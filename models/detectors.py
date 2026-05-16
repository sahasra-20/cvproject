import torch
import numpy as np
import cv2
import sys
from pathlib import Path
from ultralytics import YOLO

# Ensure paths
base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.append(str(base_dir))

from config import THRESHOLDS, MODEL_PATHS

class BikeDetector:
    # detects bikes : Returns boxes and confidence for all the bikes detected
    def __init__(self, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading Bike Detector (YOLOv8s) on {self.device}...")
        
        # Load the base model from our weights folder
        self.model = YOLO(MODEL_PATHS["yolov8_bike"]) 
        self.primary_conf = THRESHOLDS["bike_conf_primary"]
        self.fallback_conf = THRESHOLDS["bike_conf_fallback"]
        self.nms_iou = THRESHOLDS["nms_iou"]
        
        # COCO class 3: motorcycle
        self.target_classes = [3]

    def detect(self, img):
        """Returns bounding boxes and confidences for motorcycles."""
        # First pass with primary strict confidence
        results = self.model(img, conf=self.primary_conf, iou=self.nms_iou, classes=self.target_classes, device=self.device, verbose=False)
        
        # Dense-scene handling: if zero bikes detected, run a secondary sweep with fallback thresholds
        if len(results[0].boxes) == 0:
            results = self.model(img, conf=self.fallback_conf, iou=self.nms_iou, classes=self.target_classes, device=self.device, verbose=False)
            
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        return boxes, confs


class PoseDetector:
    def __init__(self, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading Pose Detector (YOLOv8s-pose) on {self.device}...")
        
        self.model = YOLO(MODEL_PATHS["yolov8_pose"])
        self.conf = THRESHOLDS["bike_conf_fallback"] # Lower threshold to find partially occluded riders
        self.nms_iou = THRESHOLDS["nms_iou"]
        self.kp_conf_threshold = THRESHOLDS["keypoint_conf"]
        
        # COCO class 0: person
        self.target_classes = [0]

    def detect(self, img):
        """Returns bounding boxes, confidences, and 17 COCO keypoints for riders."""
        results = self.model(img, conf=self.conf, iou=self.nms_iou, classes=self.target_classes, device=self.device, verbose=False)
        
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        
        keypoints = []
        if hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
            raw_kpts = results[0].keypoints.data.cpu().numpy()
            for kp in raw_kpts:
                # Filter individual skeleton keypoints by confidence
                filtered = kp.copy()
                filtered[filtered[:, 2] < self.kp_conf_threshold] = 0
                keypoints.append(filtered)
            
        return boxes, confs, keypoints

    def expand_crop(self, bbox, img_shape, margin=0.1):
        """Expand bounding box by margin (10%) to capture edges of helmets/bikes better"""
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        
        x1 = max(0, x1 - margin * w)
        y1 = max(0, y1 - margin * h)
        x2 = min(img_shape[1], x2 + margin * w)
        y2 = min(img_shape[0], y2 + margin * h)
        
        return [int(x1), int(y1), int(x2), int(y2)]

class HelmetDetector:
    def __init__(self, model_path=None, device=None):
        import os
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        path = model_path if model_path else MODEL_PATHS["helmet_model"]
        
        if os.path.exists(path):
            print(f"Loading Helmet Detector on {self.device}...")
            self.model = YOLO(path)
        else:
            print(f"Helmet model not found at {path}")
            self.model = None

    def detect(self, img):
        if self.model is None:
            return []
        return self.model(img, device=self.device, verbose=False)


class LicensePlateDetector:
    def __init__(self, model_path=None, device=None):
        import os
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        path = model_path if model_path else MODEL_PATHS["lp_model"]
        
        if os.path.exists(path):
            print(f"Loading LP Detector on {self.device}...")
            self.model = YOLO(path)
        else:
            print(f"LP model not found at {path}")
            self.model = None

    def detect(self, img):
        if self.model is None:
            return []
        return self.model(img, device=self.device, verbose=False)