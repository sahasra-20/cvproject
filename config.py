import os
from pathlib import Path



BASE_DIR = Path(__file__).resolve().parent

PATHS = {
    "models": os.path.join(BASE_DIR, "models"),
    "weights": os.path.join(BASE_DIR, "weights"),
    "outputs": os.path.join(BASE_DIR, "outputs"),
    "logs": os.path.join(BASE_DIR, "outputs", "logs"),
    "visualizations": os.path.join(BASE_DIR, "outputs", "visualizations"),
    "test_images": os.path.join(BASE_DIR, "test_evaluation")
}

MODEL_PATHS = {
    "yolov8_bike": "yolov8s.pt",
    "yolov8_pose": "yolov8s-pose.pt",
    "helmet_model": os.path.join(PATHS["weights"], "helmet_best.pt"),
    "lp_model": os.path.join(PATHS["weights"], "lp_best.pt")
}

THRESHOLDS = {
    "bike_conf_primary": 0.4,
    "bike_conf_fallback": 0.25,
    "nms_iou": 0.5, # Controls duplicate box removal
    "keypoint_conf": 0.3,
    "rider_bike_iou_fallback": 0.15, # If overlap > 15%:rider belongs to bike
    "ocr_char_conf_min": 0.6 # Minimum confidence for license plate characters.
}


AUGMENTATIONS = {
    "mosaic": True,
    "hsv_jitter": True, # Random brightness/color shifts.
    "perspective": 15, # Random camera angle transformation.
    "horizontal_flip": 0.5,
    "cutout": 0.3, #Randomly hides image parts.
    "gamma_darkening": True,
    "motion_blur": True,
    "jpeg_compression": True,
    "rain_fog": True
}


TRAINING = {
    "helmet": {
        "epochs": 50,
        "resolution": 640, # Images resized to 640×640.
        "patience": 10
    },
    "license_plate": {
        "epochs": 60,
        "resolution": 960, # Images resized to 960×960.
        "patience": 12
    }
}


INFERENCE = {
    "fast_mode": False,
    "use_cpu": False,
    "enable_fsrcnn": True
}

HELMET_DATASET = {
    "path": str(BASE_DIR / "clean_dataset" / "helmet"),
    "train": "images/train",
    "val":   "images/val",
    "test":  "images/val",
    "nc": 2,
    "names": ["helmet", "no_helmet"]
}
LP_DATASET = {
    "path": str(BASE_DIR / "clean_dataset" / "lp"),
    "train": "images/train",
    "val":   "images/val",
    "test":  "images/val",
    "nc": 1,
    "names": ["license_plate"]
}
INDIAN_STATE_CODES = [
    "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD", "DN", "DL", "GA", "GJ",
    "HR", "HP", "JK", "JH", "KA", "KL", "LA", "LD", "MP", "MH", "MN", "ML",
    "MZ", "NL", "OD", "OR", "PY", "PB", "RJ", "SK", "TN", "TS", "TR", "UP",
    "UK", "UA", "WB"
]
