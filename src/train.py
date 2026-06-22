import argparse
import torch
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="COWC OBB Resolution Study Training Script")
    parser.add_argument("--img_size", type=int, default=512, help="Input image resolution size")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--data", type=str, default="data/cowc_512/dataset.yaml", help="Path to dataset.yaml")
    parser.add_argument("--gpus", type=str, default="0,1", help="GPU IDs to use (e.g., '0,1')")
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
    print(f"Target Resolution: {args.img_size}x{args.img_size}")
    print(f"Target Devices:     GPU {device_str}")
    print(f"Batch Size:         {args.batch_size}")
    print(f"Dataset Config:     {args.data}\n")

    # CRITICAL: Initialize with the OBB architecture model weights
    model = YOLO("yolov8n-obb.pt")

    # Execute training with strict parameters to protect the resolution study metrics
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.img_size,
        
        # Explicitly declare Oriented Bounding Box task pipeline
        task="obb",
        
        # Directory Management
        project="runs/detect",
        name=f"20260622_yolov8n_res{args.img_size}",
        exist_ok=True,
        
        # Isolated Resolution Augmentations (frozen to protect study legitimacy)
        scale=0.0,        
        mosaic=0.0,       
        mixup=0.0,        
        degrees=0.0,      
        
        # Hardware Tuning
        device=device_str,
        workers=workers,
        plots=True
    )

if __name__ == "__main__":
    args = parse_args()
    run_training(args)