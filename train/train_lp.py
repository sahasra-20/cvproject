import os
import sys
import shutil
import yaml
import random
import numpy as np
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.append(str(base_dir))

import torch
from ultralytics import YOLO
from config import TRAINING, LP_DATASET, MODEL_PATHS

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

def train_lp_model():
    """
    Single-stage license plate detection training.
    Uses YOLOv8s (small) for faster convergence.
    copy_paste + scale augmentation handles tiny object learning.
    """
    set_seed(42)
    temp_yaml_path = str(base_dir / ".temp_lp_data.yaml")
    try:
        with open(temp_yaml_path, 'w') as f:
            yaml.dump(LP_DATASET, f, default_flow_style=False)
    except Exception as e:
        print(f"Failed to generate dynamic YAML: {e}")
        return

    imgsz    = TRAINING["license_plate"]["resolution"]  # 640
    epochs   = TRAINING["license_plate"]["epochs"]       # 60
    patience = TRAINING["license_plate"]["patience"]     # 12

    print(f"\n{'='*55}")
    print(f"  LP TRAINING  |  {epochs} epochs  |  {imgsz}px  |  batch=16")
    print(f"{'='*55}")

    best_epoch = 0
    best_val_loss = float('inf')
    weights_dir = base_dir / "weights"
    weights_dir.mkdir(exist_ok=True)
    final_dest = weights_dir / "lp_best.pt"

    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"  Training on: {device if device != 0 else 'GPU (CUDA)'}")

    # Start with YOLOv8m (medium) — as specified for LP
    model = YOLO(MODEL_PATHS["yolov8_medium"])

    # Cumulative progress tracking
    current_stage_offset = 0

    # This runs after every epoch to update our progress
    def on_epoch_end(trainer):
        global best_epoch, best_val_loss
        ep = trainer.epoch + 1 + current_stage_offset
        
        # Extract training losses
        t_losses = trainer.label_loss_items(trainer.tloss)
        t_box, t_cls = t_losses[0], t_losses[1]
        
        # Validation metrics
        v_box, v_cls, map50 = 0.0, 0.0, 0.0
        if hasattr(trainer, 'validator') and trainer.validator:
            v_box = trainer.validator.loss[0]
            v_cls = trainer.validator.loss[1]
            map50 = trainer.validator.results_dict.get('metrics/mAP50(B)', 0.0)
            
        # Get learning rate
        lr = trainer.optimizer.param_groups[1]['lr']
        
        # Check for best run
        current_val_total = v_box + v_cls
        if current_val_total < best_val_loss:
            best_val_loss = current_val_total
            best_epoch = ep
            marker = "  <-- NEW BEST"
            best_pt = Path(trainer.save_dir) / "weights" / "best.pt"
            if best_pt.exists():
                shutil.copy(str(best_pt), str(final_dest))
        else:
            marker = ""
            
        print(f"  Epoch {ep:>3}/{epochs} | train_box_loss: {t_box:.3f} | train_cls_loss: {t_cls:.3f} | map_50: {map50:.3f} | val_box_loss: {v_box:.3f} | val_cls_loss: {v_cls:.3f} | learning_rate: {lr:.5f}{marker}")


    try:
        model.add_callback('on_train_epoch_end', on_epoch_end)

        # Exact Tiny-Object focused Augmentations from your snippet
        aug_params = {
            'mosaic': 1.0, # Helps in tiny-object detection.
            'perspective': 0.002, # Simulates angled plates
            'hsv_v': 0.5, # Aggressive Shadow/Glare simulation
            'copy_paste': 0.2, # Extremely useful for tiny-object density
            'mixup': 0.1,
            'degrees': 10.0, # Rotation for crooked plates
            'scale': 0.5, # Scale reduction forces model to learn smaller details
        }

        #stage 1
        print("\n--- STAGE 1: WARMUP (Tiny Objects) ---")
        current_stage_offset = 0
        model.train(
            data=temp_yaml_path,
            epochs=5,
            imgsz=imgsz,
            freeze=10, 
            lr0=1e-3,
            lrf=0.01,
            patience=patience,
            batch=16,
            workers=8,
            project=str(base_dir / "outputs" / "training" / "lp"),
            name="stage1_frozen",
            exist_ok=True,
            device=device,
            **aug_params
        )
        
        last_weight_path1 = Path(base_dir / "outputs" / "training" / "lp" / "stage1_frozen" / "weights" / "last.pt")

    
        #stage 2
        print("\n--- STAGE 2: MAIN TRAINING ---")
        current_stage_offset = 5
        if last_weight_path1.exists():
            model = YOLO(str(last_weight_path1))
            model.add_callback('on_train_epoch_end', on_epoch_end)
        
        model.train(
            data=temp_yaml_path,
            epochs=35, 
            imgsz=imgsz,
            freeze=0, 
            lr0=0.01,
            lrf=0.1, 
            patience=patience,
            batch=16,
            workers=8,
            project=str(base_dir / "outputs" / "training" / "lp"),
            name="stage2_unfrozen",
            exist_ok=True,
            device=device,
            **aug_params
        )
        
        last_weight_path2 = Path(base_dir / "outputs" / "training" / "lp" / "stage2_unfrozen" / "weights" / "last.pt")

        #stage 3
        print("\n--- STAGE 3: FINE TUNING ---")
        current_stage_offset = 40
        if last_weight_path2.exists():
            model = YOLO(str(last_weight_path2))
            model.add_callback('on_train_epoch_end', on_epoch_end)
        
        model.train(
            data=temp_yaml_path,
            epochs=20, 
            imgsz=imgsz,
            freeze=0,
            lr0=0.001,
            lrf=0.01, 
            patience=patience,
            batch=16,
            workers=8,
            project=str(base_dir / "outputs" / "training" / "lp"),
            name="stage3_finetune",
            exist_ok=True,
            device=device,
            **aug_params
        )

        # Route final best weight to weights folder
        best_weight = Path(base_dir / "outputs" / "training" / "lp" / "stage3_finetune" / "weights" / "best.pt")
        if best_weight.exists():
            shutil.copy(str(best_weight), str(final_dest))
            print(f"\nSUCCESS: Tiny-Object Model saved to {final_dest}")

        # Save last model for reproducibility
        last_pt = Path(base_dir / "outputs" / "training" / "lp" / "stage3_finetune" / "weights" / "last.pt")
        last_dest = weights_dir / "lp_last.pt"
        if last_pt.exists():
            shutil.copy(str(last_pt), str(last_dest))
            print(f"  Last model saved to: {last_dest}")

        # Final summary
        print(f"\n{'='*55}")
        print(f"  TRAINING COMPLETE")
        print(f"  Best Epoch      : {best_epoch}/{epochs}")
        print(f"  Best Val Loss   : {best_val_loss:.4f}")
        print(f"  Final Model     : {final_dest}")
        print(f"{'='*55}\n")

    except Exception as e:
        print(f"LP Training Failed: {e}")
        print(f"CRITICAL ERROR: {e}")

    finally:
        if os.path.exists(temp_yaml_path):
            os.remove(temp_yaml_path)

if __name__ == "__main__":
    train_lp_model()