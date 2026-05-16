import os
import sys
import cv2
from pathlib import Path

base_dir = Path(__file__).resolve().parent
if str(base_dir) not in sys.path:
    sys.path.append(str(base_dir))
from inference import InferenceEngine

class TrafficViolationDetector:
    def __init__(self, model_dir="./weights"):
        """
        Initialize and load all models here.
        model_dir: path to directory containing model weights.
        """
        # Initialize the high-performance engine (handles all stages and device routing)
        self.engine = InferenceEngine(model_dir=model_dir)

    def predict(self, image_path: str) -> dict:
        """
        Input:
            image_path: Path to input image
        Output:
            {
                "violations": [
                    {
                        "num_riders": int,
                        "helmet_violations": int,
                        "license_plate": "string"
                    }
                ]
            }
        """
        try:
            # 1. Load Image
            if not os.path.exists(image_path):
                return {"violations": [], "all_detections": [], "metadata": {}}
            img = cv2.imread(image_path)
            if img is None:
                return {"violations": [], "all_detections": [], "metadata": {}}

            # 2. Run the 9-stage inference pipeline via the engine
            import time
            t0 = time.perf_counter()
            all_results = self.engine.run_full_inference(img)
            latency_ms = (time.perf_counter() - t0) * 1000

            # 3. Filter for violations only (Triple riding or Missing helmets)
            # NOTE: inference.py uses "riders_count" as the key name
            violations = []
            for res in all_results:
                if res["riders_count"] > 2 or res["helmet_violations"] > 0:
                    violations.append(res)

            return {
                "violations": violations,
                "all_detections": all_results,   # full per-bike detail for visualizer
                "metadata": {"total_latency_ms": latency_ms}
            }

        except Exception as e:
            # Absolute crash safety
            print(f"Error during prediction: {e}")
            import traceback; traceback.print_exc()
            return {"violations": [], "all_detections": [], "metadata": {}}
