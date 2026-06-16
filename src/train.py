#!/usr/bin/env python3

import os
from pathlib import Path
from ultralytics import YOLO

# ---------------------------------------------------------
# 1. Paths & Configuration (HPC Cluster Compliant)
# ---------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML_PATH = REPO_ROOT / "data" / "20260615_42_cowc_base" / "data.yaml"

# HPC Baseline Hyperparameters (Optimized for Dual M60)
EPOCHS = 100
BATCH_SIZE = 64   # Total batch size (32 images per GPU x 2 GPUs)
IMG_SIZE = 640    # Kept constant to lock GSD baseline variables
MODEL_VARIANT = "yolov8n.pt" 

def run_hpc_dual_gpu_training():
    print("="*60)
    print(" 🚀 ORBITAL VEHICLE DETECTOR: DUAL-GPU HPC RUN KICKOFF")
    print("="*60)
    print(f"Loading Configuration: {DATA_YAML_PATH}")
    print(f"Target Hardware:      2x NVIDIA Tesla M60 (CUDA:0,1)")
    print(f"Distributed Batch:    {BATCH_SIZE} total (32 per GPU)")
    print("-" * 60)

    # Initialize model with pre-trained coco weights
    model = YOLO(MODEL_VARIANT)

    # Execute DDP multi-GPU training loop
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
        
        # HPC Multi-GPU Infrastructure Tuning
        device=[0, 1],    # Directs YOLO to initialize Distributed Data Parallel on both cores
        workers=8,        # Slices data loading tasks over 8 total CPU workers (4 per GPU)
        plots=True        # Generates metrics, charts, and predictions directly in the run folder
    )

if __name__ == "__main__":
    run_hpc_dual_gpu_training()