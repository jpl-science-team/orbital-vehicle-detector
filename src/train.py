#!/usr/bin/env python3

import os
from pathlib import Path
from ultralytics import YOLO

# ---------------------------------------------------------
# 1. Paths & Configuration (Single GPU Setup)
# ---------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML_PATH = REPO_ROOT / "data" / "20260615_42_cowc_base" / "data.yaml"

# Single GPU Hyperparameters
EPOCHS = 100
BATCH_SIZE = 32   # Cut in half since we are only using 1 GPU now
IMG_SIZE = 640    # Kept constant to lock GSD baseline variables
MODEL_VARIANT = "yolov8n.pt" 

def run_single_gpu_training():
    print("="*60)
    print(" 🚀 ORBITAL VEHICLE DETECTOR: SINGLE-GPU RUN KICKOFF")
    print("="*60)
    print(f"Loading Configuration: {DATA_YAML_PATH}")
    print(f"Target Hardware:      NVIDIA Tesla M60 (CUDA:0)")
    print(f"Batch Size:           {BATCH_SIZE}")
    print("-" * 60)

    # Initialize model with pre-trained coco weights
    model = YOLO(MODEL_VARIANT)

    # Execute training loop on device 0
    model.train(
        data=str(DATA_YAML_PATH),
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        
        # Directory Management
        project="runs/detect",
        name="20260615_yolov8n_base",
        exist_ok=True,
        
        # Isolated Resolution Augmentations (Omitted for baseline)
        scale=0.0,        
        mosaic=0.0,       
        mixup=0.0,        
        degrees=0.0,      
        
        # Single GPU Tuning
        device=0,         # Run strictly on device 0
        workers=4,        # Scaled down to 4 CPU workers for a single GPU
        plots=True        # Generates metrics, charts, and predictions
    )

if __name__ == "__main__":
    run_single_gpu_training()
