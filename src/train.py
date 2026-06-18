#!/usr/bin/env python3

"""
=============================================================================
HOW TO RUN THIS SCRIPT
=============================================================================

Basic execution (requires image size, GPUs, and the dataset folder name):
    python train.py --img_size 512 --gpus 1 --dataset 20260615_42_cowc_base

Full execution with optional batch size and epochs:
    python train.py --img_size 512 --gpus 2 --dataset 20260615_42_cowc_base --batch_size 64 --epochs 150

=============================================================================
"""

import os
import argparse
from pathlib import Path
from ultralytics import YOLO

# ---------------------------------------------------------
# 1. Paths & Configuration
# ---------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_VARIANT = "yolov8n.pt" 

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 Resolution & Multi-GPU Study")
    
    # Required Arguments
    parser.add_argument("--img_size", type=int, required=True, help="Target imagery resolution (e.g., 256, 512)")
    parser.add_argument("--gpus", type=int, required=True, help="Number of GPUs to use (e.g., 1 or 2)")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset folder inside the 'data' directory")
    
    # Optional Arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per run")
    parser.add_argument("--epochs", type=int, default=100, help="Total training epochs")
    
    return parser.parse_args()

def run_training(args):
    # Dynamically resolve the dataset path
    data_yaml_path = REPO_ROOT / "data" / args.dataset / "data.yaml"
    
    # Fail fast if the user provided an incorrect dataset name
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"🚨 Dataset configuration not found at: {data_yaml_path}")

    # Format the device string for YOLO (e.g., "0" for 1 GPU, "0,1" for 2 GPUs)
    device_str = ",".join(str(i) for i in range(args.gpus))
    
    # Scale CPU workers dynamically based on the number of GPUs
    workers = 4 * args.gpus 

    print("="*60)
    print(" 🚀 ORBITAL VEHICLE DETECTOR: RESOLUTION STUDY")
    print("="*60)
    print(f"Loading Configuration: {data_yaml_path}")
    print(f"Target Resolution:    {args.img_size}x{args.img_size}")
    print(f"Target Hardware:      {args.gpus} GPU(s) (CUDA:{device_str})")
    print(f"Batch Size:           {args.batch_size}")
    print("-" * 60)

    # Initialize model with pre-trained coco weights
    model = YOLO(MODEL_VARIANT)

    # Execute training loop
    model.train(
        data=str(data_yaml_path),
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.img_size,
        
        # Directory Management (Dynamically names folder based on resolution)
        project="runs/detect",
        name=f"20260618_yolov8n_res{args.img_size}",
        exist_ok=True,
        
        # Isolated Resolution Augmentations (Omitted for Study)
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