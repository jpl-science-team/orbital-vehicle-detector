import argparse
import os
import torch
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="COWC OBB Resolution Study Training Script")
    parser.add_argument("--img_size", type=int, default=512, help="Input image resolution size")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--data", type=str, default="data/cowc_512/dataset.yaml", help="Path to dataset.yaml")
    parser.add_argument("--gpus", type=str, default="2,3", help="GPU IDs to use (e.g., '2,3')")
    parser.add_argument("--workers", type=int, default=4, help="Number of data loader workers")
    return parser.parse_args()

def run_training(args):
    # Set device configuration based on input gpus string
    if torch.cuda.is_available():
        device_str = args.gpus
        workers = args.workers
    else:
        device_str = "cpu"
        workers = 0

    print(f"--- Starting Oriented Object Detection (OBB) Training ---")
    print(f"Target Model:       YOLO11 Nano OBB")
    print(f"Target Resolution:  {args.img_size}x{args.img_size}")
    print(f"Target Devices:     GPU {device_str}")
    print(f"Batch Size:         {args.batch_size}")
    print(f"Dataset Config:     {args.data}\n")

    # STRICTLY LOCKED: YOLO11 Nano OBB Deployment Anchor
    weights_filename = "yolo11n-obb.pt"
    
    if not os.path.exists(weights_filename):
        print(f"--- Enforcing Strict YOLO11 Release Asset Download ---")
        # Direct authenticated binary stream path bypassing package version mapping
        url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-obb.pt"
        
        # Pull down the raw un-redirected file
        torch.hub.download_url_to_file(url, weights_filename, progress=True)
    
    # Initialize framework strictly on the downloaded YOLO11 architecture
    model = YOLO(weights_filename)

    # Execute training with strict parameters optimized for OBB on Snapdragon hardware
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.img_size,
        task="obb",
        
        # Directory Management
        project="runs/detect",
        name=f"20260714_yolo11n_res{args.img_size}",
        exist_ok=True,
        
        # --- OBB Training Stabilization & Regularization ---
        optimizer="AdamW",          # Dampens chaotic OBB angular gradient explosions
        lr0=0.001,                  # Lower initial learning rate to prevent loss spikes
        cos_lr=True,                # Smooth cosine learning rate curve
        warmup_epochs=5,            # Establishes solid object orientation early on
        patience=15,                # Early stopping to lock in peak weights before overfitting
        weight_decay=0.0005,        # Penalize overly large weights
        dropout=0.15,               # Drop nodes in the head to prevent memorization
        
        # --- Safe Aerial Augmentations (No Resolution Impact) ---
        flipud=0.5,                 # Top-down imagery has no "up", safe to flip
        fliplr=0.5,                 # Safe horizontal flip
        hsv_h=0.015,                # Color/lighting variation
        hsv_s=0.7,                  # Saturation variation
        hsv_v=0.4,                  # Brightness variation
        
        # Isolated Resolution Augmentations (Adjust degrees for OBB orientation stability)
        scale=0.0,        
        mosaic=0.0,       
        mixup=0.0,        
        degrees=180.0,              # Full rotation to learn multi-angle vehicle placements
        
        # Hardware Tuning
        device=device_str,
        workers=workers,
        plots=True
    )

    print("\nTraining completed successfully! Initiating automated export pipeline...")
    
    # Extract the absolute path of the best saved weights file from this specific run
    best_weights_path = os.path.join(model.trainer.save_dir, "weights", "best.pt")
    trained_model = YOLO(best_weights_path)

    print(f"Exporting {best_weights_path} to deployment-optimized ONNX...")
    
    # Generate the target file for your deployment team
    onnx_path = trained_model.export(
        format="onnx",
        imgsz=args.img_size,       # Dynamically locks NPU buffer sizes to your dataset resolution
        dynamic=False,              # Static shapes provide optimal performance on Snapdragon NPUs
        simplify=True,              # Drastically optimizes graph nodes for the QNN compiler
        nms=True                    # Fuses NMS processing logic directly into the model graph
    )
    
    print(f"\n🚀 Target deployment build generated: {onnx_path}")

if __name__ == "__main__":
    args = parse_args()
    run_training(args)
