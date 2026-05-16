import cv2
import json
import argparse
import sys
from pathlib import Path

# Fix Windows encoding for cleaner terminal output
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from solution import TrafficViolationDetector
from config import PATHS

# Set up paths
base_dir = Path(__file__).resolve().parent

def draw_info_on_image(img, payload):
    """Draws detailed boxes for riders, helmets, and plates with per-bike stats."""
    
    h, w = img.shape[:2]
    
    # Adaptive scaling for premium look on any resolution(scaling for labels)
    scale = min(w, h) / 700.0
    font = cv2.FONT_HERSHEY_DUPLEX 
    font_scale = max(0.35, 0.5 * scale)
    thickness = max(1, int(1.5 * scale))
    box_thickness = max(1, int(2 * scale))
    
    # Colors (BGR)
    CLR_VIOLATION = (40, 40, 255)    # Vibrant Red
    CLR_COMPLIANT = (70, 255, 70)    # Emerald Green
    CLR_RIDER     = (255, 255, 255)  # White
    CLR_HELMET    = (0, 255, 0)      # Neon Green
    CLR_NO_HELMET = (0, 140, 255)    # Orange/Red
    CLR_PLATE     = (0, 220, 255)    # Gold
    
    total_bikes = len(payload.get("all_detections", []))
    total_riders = 0
    total_helmets = 0
    total_no_helmets = 0
    
    # Sort bikes from left to right based on bounding box x1 coordinate
    detections = payload.get("all_detections", [])
    detections = sorted(detections, key=lambda d: d["bike_bbox"][0])
    
    for i, det in enumerate(detections):
        det['bike_id'] = i + 1 # Reassign ID based on left-to-right order
        
        total_riders += det["riders_count"]
        h_on_bike = len(det["helmets_detected"])
        nh_on_bike = len(det["no_helmets_detected"])
        total_helmets += h_on_bike
        total_no_helmets += nh_on_bike
        
        bx1, by1, bx2, by2 = map(int, det["bike_bbox"])
        is_v = det["is_violation"]
        color = CLR_VIOLATION if is_v else CLR_COMPLIANT
        
        # 1. Draw Bike Container (Main Box)
        cv2.rectangle(img, (bx1, by1), (bx2, by2), color, box_thickness)
        
        # 2. Draw Riders on this bike
        for r_box in det.get("riders_detected", []):
            rx1, ry1, rx2, ry2 = map(int, r_box)
            cv2.rectangle(img, (rx1, ry1), (rx2, ry2), CLR_RIDER, thickness)
            
        # 3. Draw Helmets
        for h_box in det.get("helmets_detected", []):
            hx1, hy1, hx2, hy2 = map(int, h_box)
            cv2.rectangle(img, (hx1, hy1), (hx2, hy2), CLR_HELMET, thickness)
            
        for nh_box in det.get("no_helmets_detected", []):
            nx1, ny1, nx2, ny2 = map(int, nh_box)
            cv2.rectangle(img, (nx1, ny1), (nx2, ny2), CLR_NO_HELMET, thickness)
            
        # 4. Draw License Plate
        if det.get("lp_bbox"):
            lx1, ly1, lx2, ly2 = map(int, det["lp_bbox"])
            cv2.rectangle(img, (lx1, ly1), (lx2, ly2), CLR_PLATE, box_thickness)
            lp_text = det.get('license_plate') or "PLATE"
            cv2.putText(img, lp_text, (lx1, max(15, ly1 - 5)), font, font_scale, CLR_PLATE, thickness)
            
        # 5. Per-Bike Summary Label
        label = f"BIKE {det['bike_id']} | Riders:{det['riders_count']} | H:{h_on_bike} NH:{nh_on_bike}"
        label_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        cv2.rectangle(img, (bx1, by1 - label_size[1] - 10), (bx1 + label_size[0] + 10, by1), color, -1)
        cv2.putText(img, label, (bx1 + 5, by1 - 7), font, font_scale, (0,0,0), thickness)
        
    # Premium HUD Overlay
    hud_w = int(220 * scale)
    hud_h = int(120 * scale)
    pad = 20
    overlay = img.copy()
    cv2.rectangle(overlay, (pad, pad), (pad + hud_w, pad + hud_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    
    ys = [pad + int(hud_h * r) for r in [0.22, 0.45, 0.68, 0.90]]
    cv2.putText(img, f"TOTAL BIKES: {total_bikes}", (pad+10, ys[0]), font, font_scale, (255,255,255), 1)
    cv2.putText(img, f"TOTAL RIDERS: {total_riders}", (pad+10, ys[1]), font, font_scale, (255,255,255), 1)
    cv2.putText(img, f"HELMETS OK: {total_helmets}", (pad+10, ys[2]), font, font_scale, CLR_COMPLIANT, 1)
    
    status_text = "OVERALL: VIOLATION" if total_no_helmets > 0 else "OVERALL: CLEAR"
    status_clr = CLR_VIOLATION if total_no_helmets > 0 else CLR_COMPLIANT
    cv2.putText(img, status_text, (pad+10, ys[3]), font, font_scale, status_clr, 1)

    return img, total_bikes, total_riders, total_helmets, total_no_helmets



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Traffic Detection")
    parser.add_argument("--image", type=str, default="test_sample.jpg", help="Image to test")
    args = parser.parse_args()
    
    img_path = str(base_dir / args.image)
    print(f"Loading pipeline for image: {img_path}")
    
    # 1. Initialize Pipeline
    detector = TrafficViolationDetector()
    
    # 2. Run Inference
    payload = detector.predict(img_path)
    
    # 3. Read image for visualization
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to read image at {img_path}")
        exit()
        
    # 4. Draw boxes and exact info
    out_img, b_cnt, r_cnt, h_cnt, nh_cnt = draw_info_on_image(img, payload)
    
    # 5. Save the output
    out_dir = Path(PATHS["visualizations"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"visualized_{Path(args.image).name}"
    
    cv2.imwrite(str(out_path), out_img)
    
    print("\n" + "="*50)
    print("INFERENCE SUMMARY (Left to Right):")
    print("="*50)
    
    # Sort and print detailed info per bike
    detections = payload.get("all_detections", [])
    detections = sorted(detections, key=lambda d: d["bike_bbox"][0])
    
    for i, det in enumerate(detections):
        b_id = i + 1
        bx1, by1, bx2, by2 = map(int, det["bike_bbox"])
        riders = det.get("riders_detected", [])
        
        # Sort riders left to right
        riders = sorted(riders, key=lambda r: r[0])
        
        print(f"\n[Bike {b_id}] (Position x={bx1}):")
        
        lp_text = det.get('license_plate')
        if lp_text:
            print(f"  - License Plate: {lp_text}")
        else:
            print("  - License Plate: Not Detected")
            
        print(f"  - Total Riders: {len(riders)}")
        
        for j, r_box in enumerate(riders):
            rx1, ry1, rx2, ry2 = r_box
            status = "Status Unknown"
            
            # Simple spatial association: check if helmet box center is inside rider box
            for h_box in det.get("helmets_detected", []):
                hx1, hy1, hx2, hy2 = h_box
                if rx1 <= (hx1+hx2)/2 <= rx2 and ry1 <= (hy1+hy2)/2 <= ry2:
                    status = "Helmet WORN"
                    break
            
            if status == "Status Unknown":
                for nh_box in det.get("no_helmets_detected", []):
                    nx1, ny1, nx2, ny2 = nh_box
                    if rx1 <= (nx1+nx2)/2 <= rx2 and ry1 <= (ny1+ny2)/2 <= ry2:
                        status = "NO HELMET (Violation)"
                        break
            
            print(f"    -> Rider {j+1}: {status}")

    print("\n" + "-"*50)
    print("TOTALS:")
    print(f"Total Bikes Detected:      {b_cnt}")
    print(f"Total Riders Detected:     {r_cnt}")
    print(f"Total Helmets Worn:        {h_cnt}")
    print(f"Total Riders w/o Helmets:  {nh_cnt}")
    
    latency_ms = payload.get('metadata', {}).get('total_latency_ms', 0)
    latency_sec = latency_ms / 1000

    print(f"Latency: {latency_sec:.2f} seconds")
    print(f"\nVisualization saved successfully to: {out_path}")
    print("="*50)