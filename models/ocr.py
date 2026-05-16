
# Enhances blurry/tilted license plate images using preprocessing and super-resolution.
# Reads and postprocesses Indian license plate text using OCR with confidence-based fallback.

import cv2
import numpy as np
import re
import math
import sys
import os
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.append(str(base_dir))

from config import THRESHOLDS, PATHS

    
PADDLE_AVAILABLE = True
try:
    from paddleocr import PaddleOCR
except ImportError:
    PADDLE_AVAILABLE = False


class LicensePlateEnhancer:
    """improves image quality BEFORE OCR"""
    def __init__(self, enable_fsrcnn=True):
        self.enable_fsrcnn = enable_fsrcnn
        self.fsrcnn = None
        if self.enable_fsrcnn:
            # Load FSRCNN model if physically available in models/ directory
            model_path = os.path.join(PATHS["models"], "FSRCNN_x3.pb")
            if os.path.exists(model_path):
                self.fsrcnn = cv2.dnn_superres.DnnSuperResImpl_create()
                self.fsrcnn.readModel(model_path)
                self.fsrcnn.setModel("fsrcnn", 3)
            else:
                print("WARNING: FSRCNN_x3.pb not found. Reverting to Cubic Interpolation fallback.")
                self.fsrcnn = None

    def deskew(self, img):
        # Corrects tilted license plates.
        """Uses Hough Line Transform to detect crooked plates and mathematically rotate them flat."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        edges = cv2.Canny(gray, 50, 150, apertureSize=3) # EDGE DETECTION
        lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
        
         # Calculates dominant tilt angle.
        if lines is not None:
            angles = []
            for line in lines:
                rho, theta = line[0]
                angle = math.degrees(theta)
                # Filter out pure vertical/horizontal noise
                if 45 < angle < 135:
                    angles.append(angle - 90)
                    
            if angles:
                median_angle = np.median(angles)
                # Only correct if skew is noticeable but not completely flipped
                if abs(median_angle) > 2 and abs(median_angle) < 25:
                    (h, w) = img.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                    # Rotates plate back to straight.
                    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                    return rotated
        return img

    def enhance(self, img):
        """Full image restoration pipeline for blurry/tiny license plates."""
        # 1. Deskew
        processed = self.deskew(img)
        
        # 2. Super Resolution
        if self.fsrcnn is not None:
            processed = self.fsrcnn.upsample(processed)
        else:
            # Fallback algorithmic resize
            processed = cv2.resize(processed, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            
        # 3. Bilateral Filtering (removes noise while keeping edge sharpness)
        processed = cv2.bilateralFilter(processed, 9, 75, 75)
        
        # 4. CLAHE (Contrast fixing for shadow/glare)
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY) if len(processed.shape) == 3 else processed
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        processed = clahe.apply(gray)
        
        return processed # Returns Grayscale array

class OCRProcessor:
    def __init__(self):
        self.enhancer = LicensePlateEnhancer()
        self.min_conf = THRESHOLDS.get("ocr_char_conf_min", 0.6)
        
        # Initialize Primary Model (PaddleOCR)
        if PADDLE_AVAILABLE:
            # PaddleOCR is significantly faster on CPU for small crops due to PCIe overhead.
            # enable_mkldnn=True provides Intel CPU acceleration.
            self.paddle = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=False, enable_mkldnn=True, show_log=False)
        else:
            print("PaddleOCR library not found. Run pip install paddlepaddle paddleocr")
            self.paddle = None
            
        self.trocr_processor = None
        self.trocr_model = None

    def postprocess_text(self, text):
        """Standard cleaning for all international plate formats."""
        # Clean string: uppercase, remove spaces, keep only alphanumeric
        clean = re.sub(r'[^A-Z0-9]', '', text.upper())
        if len(clean) < 3:
            return None
        return clean

    def _run_paddle(self, img):
        if not self.paddle: return None, 0.0
        results = self.paddle.ocr(img, cls=False)
        if not results or not results[0]: return None, 0.0
        
        full_text = ""
        total_conf = 0.0
        count = 0
        for line in results[0]:
            text, conf = line[1]
            full_text += text
            total_conf += conf
            count += 1
            
        avg_conf = total_conf / count if count > 0 else 0.0
        return full_text, avg_conf



    def read_plate(self, img):
        """
        Executes the entire Phase 8 pipeline:
        Image Enhancement -> Primary PaddleOCR -> Condition Check -> Fallback TrOCR -> Postprocessing
        """
        enhanced_img = self.enhancer.enhance(img)
        
        # PRIMARY: PaddleOCR
        text, conf = self._run_paddle(enhanced_img)
        used_model = "PaddleOCR"
        
        # FALLBACK: Removed per user request
        if text is None or conf < self.min_conf:
            pass # PaddleOCR is now the single source of truth

                
        if not text:
            return None, 0.0, "Failed", enhanced_img
            
        # POSTPROCESSING
        final_text = self.postprocess_text(text)
        
        return final_text, conf, used_model, enhanced_img
