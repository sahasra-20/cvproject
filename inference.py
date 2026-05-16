import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path
import torch
import psutil
from ultralytics import YOLO

# Ensure internal paths are correct
base_dir = Path(__file__).resolve().parent
if str(base_dir) not in sys.path:
    sys.path.append(str(base_dir))

from models.detectors import BikeDetector, PoseDetector, HelmetDetector, LicensePlateDetector
from models.association import RiderBikeAssociator
from models.ocr import OCRProcessor
from config import PATHS, MODEL_PATHS, INFERENCE


class Benchmark:
    def __init__(self):
        self.start_time = None
        self.stage_times = {}
        self.process = psutil.Process(os.getpid())
        
    def start(self):
        self.start_time = time.perf_counter()
        self.stage_times.clear()
        
    def record_stage(self, stage_name):
        current_time = time.perf_counter()
        elapsed = current_time - self.start_time
        self.stage_times[stage_name] = elapsed
        self.start_time = current_time 
        return elapsed
        
    def get_total_latency(self):
        return sum(self.stage_times.values())


class TrafficImagePreprocessor:
    def __init__(self, target_size=(640, 640)):
        self.target_size = target_size
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
    def is_dark_scene(self, img, threshold=80):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        return np.mean(gray) < threshold
        
    def is_noisy_scene(self, img, threshold=20):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var < threshold
        
    def apply_clahe(self, img):
        if len(img.shape) == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            cl = self.clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return self.clahe.apply(img)
        
    def apply_gamma(self, img, gamma=1.5):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(img, table)
        
    def denoise(self, img):
        return cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
        
    def resize_preserve_aspect(self, img):
        h, w = img.shape[:2]
        tw, th = self.target_size
        scale = min(tw / w, th / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        top = (th - new_h) // 2
        bottom = th - new_h - top
        left = (tw - new_w) // 2
        right = tw - new_w - left
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return padded, scale, top, left

    def process_adaptive(self, img):
        start_t = time.perf_counter()
        dark = self.is_dark_scene(img)
        noisy = self.is_noisy_scene(img)
        processed = img.copy()
        if dark:
            processed = self.apply_gamma(processed, gamma=1.5)
            processed = self.apply_clahe(processed)
        if noisy:
            processed = self.denoise(processed)
        resized, scale, pad_t, pad_l = self.resize_preserve_aspect(processed)
        latency_ms = (time.perf_counter() - start_t) * 1000
        return resized, latency_ms, {"dark": dark, "noisy": noisy, "scale": scale, "pad": (pad_t, pad_l)}


class InferenceEngine:
    def __init__(self, model_dir="./weights"):
        self.benchmark = Benchmark()
        self.gpu_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # CPU vs GPU split: Neural nets on GPU, logic/preprocessing on CPU
        self.preprocessor = TrafficImagePreprocessor()
        self.bike_detector = BikeDetector(device=self.gpu_device)
        self.pose_detector = PoseDetector(device=self.gpu_device)
        self.associator = RiderBikeAssociator()
        
        # Load specialized weights
        h_path = os.path.join(model_dir, "helmet_best.pt")
        l_path = os.path.join(model_dir, "lp_best.pt")
        
        self.helmet_detector = HelmetDetector(model_path=h_path, device=self.gpu_device)
        self.lp_detector = LicensePlateDetector(model_path=l_path, device=self.gpu_device)
        self.ocr_processor = OCRProcessor() # read license plate text.
        
        # GPU Warmup
        if self.gpu_device == 'cuda':
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            if self.helmet_detector.model is not None: self.helmet_detector.detect(dummy)
            if self.lp_detector.model is not None: self.lp_detector.detect(dummy)

    def run_full_inference(self, img):
        # 1. Adaptive Preprocessing
        processed_img, _, meta = self.preprocessor.process_adaptive(img)
        scale = meta["scale"]
        pad_t, pad_l = meta["pad"]
        
        def remap(box):
            if box is None or len(box) == 0: return box
            return [(box[0]-pad_l)/scale, (box[1]-pad_t)/scale, (box[2]-pad_l)/scale, (box[3]-pad_t)/scale]

        # Implementation of all 9 stages
        # Returns internal detailed detections
        bike_boxes, bike_confs = self.bike_detector.detect(processed_img)
        pose_boxes, _, keypoints = self.pose_detector.detect(processed_img)
        associations = self.associator.associate(bike_boxes, bike_confs, pose_boxes, keypoints)
        
        results = []
        for assoc in associations:
            bike_box = assoc['bike']
            riders = assoc['riders']
            num_riders = len(riders) if not assoc['inferred_rider'] else 1
            
            # Helmet Check
            helmet_violations = 0
            helmets_detected = []
            no_helmets_detected = []
            if self.helmet_detector.model is not None and not assoc['inferred_rider']:
                for r_box in riders:
                    # Calculate true x1, y1 with 0.1 margin to map helmet boxes back
                    x1, y1, x2, y2 = map(int, r_box)
                    bw, bh = x2 - x1, y2 - y1
                    margin = 0.1
                    crop_x1 = max(0, int(x1 - bw * margin))
                    crop_y1 = max(0, int(y1 - bh * margin))
                    
                    r_crop = self._safe_crop(processed_img, r_box, margin)
                    if r_crop.size == 0: continue
                    res = self.helmet_detector.detect(r_crop)
                    status = "unknown"
                    h_box_local = None
                    if len(res[0].boxes) > 0:
                        # Explicitly compare helmet (0) vs no_helmet (1) using highest confidence
                        best_box = max(res[0].boxes, key=lambda b: b.conf[0].item())
                        best_cls = int(best_box.cls[0].item())
                        h_box_local = best_box.xyxy[0].cpu().numpy()
                        if best_cls == 0:
                            status = "helmet"
                        elif best_cls == 1:
                            status = "no_helmet"
                            
                    if status == "helmet" and h_box_local is not None:
                        hx1, hy1, hx2, hy2 = h_box_local
                        helmets_detected.append([float(crop_x1 + hx1), float(crop_y1 + hy1), float(crop_x1 + hx2), float(crop_y1 + hy2)])
                    else:
                        # Default to violation if unknown or explicitly no_helmet
                        helmet_violations += 1
                        if h_box_local is not None:
                            hx1, hy1, hx2, hy2 = h_box_local
                            no_helmets_detected.append([float(crop_x1 + hx1), float(crop_y1 + hy1), float(crop_x1 + hx2), float(crop_y1 + hy2)])
                        else:
                            no_helmets_detected.append(r_box.tolist())
            
            # LP & OCR
            lp_text = ""
            lp_bbox = None
            if (num_riders > 2 or helmet_violations > 0) and self.lp_detector.model is not None:
                b_crop = self._safe_crop(processed_img, bike_box)
                if b_crop.size > 0:
                    lp_res = self.lp_detector.detect(b_crop)
                    if len(lp_res[0].boxes) > 0:
                        best_lp = sorted(lp_res[0].boxes, key=lambda x: x.conf[0].item(), reverse=True)[0]
                        # Calculate LP bbox in original image coordinates
                        bx1, by1 = bike_box[:2]
                        lpx1, lpy1, lpx2, lpy2 = best_lp.xyxy[0].cpu().numpy()
                        lp_bbox = [float(bx1 + lpx1), float(by1 + lpy1), float(bx1 + lpx2), float(by1 + lpy2)]
                        
                        lp_crop = self._safe_crop(b_crop, best_lp.xyxy[0], 0.05)
                        if lp_crop.size > 0:
                            ocr_result = self.ocr_processor.read_plate(lp_crop)
                            if ocr_result and ocr_result[0]:
                                lp_text = ocr_result[0]
                            
            is_violation = (num_riders > 2 or helmet_violations > 0)
            
            # Remap everything to original image coordinates
            results.append({
                "bike_id": len(results) + 1,
                "bike_bbox": remap(bike_box.tolist()),
                "riders_count": int(num_riders),
                "riders_detected": [remap(r.tolist()) for r in riders],
                "helmets_detected": [remap(h) for h in helmets_detected],
                "no_helmets_detected": [remap(nh) for nh in no_helmets_detected],
                "lp_bbox": remap(lp_bbox),
                "license_plate": lp_text,
                "is_violation": is_violation,
                "helmet_violations": int(helmet_violations)
            })
        return results

    def _safe_crop(self, img, box, margin=0.0):
        x1, y1, x2, y2 = map(int, box)
        h, w = img.shape[:2]
        if margin > 0:
            bw, bh = x2 - x1, y2 - y1
            x1 = max(0, int(x1 - bw * margin)); y1 = max(0, int(y1 - bh * margin))
            x2 = min(w, int(x2 + bw * margin)); y2 = min(h, int(y2 + bh * margin))
        return img[y1:y2, x1:x2]
