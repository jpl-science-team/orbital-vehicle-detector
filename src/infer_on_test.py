#!/usr/bin/env python3

from ultralytics import YOLO
from pathlib import Path
import torch

def evaluate_test_set(weights_path, dataset_yaml, img_size=1024):
    print("="*60)
    print(" 🚀 YOLOv8 TEST DATASET EVALUATION")
    print("="*60)
    
    # Determine the best available hardware
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        
    print(f"Target Hardware:   {device.upper()}")
    print(f"Model Weights:     {weights_path}")
    print(f"Dataset Config:    {dataset_yaml}")
    print(f"Inference Size:    {img_size}x{img_size}")
    print("-" * 60)

    # 1. Load the trained model weights (best.pt)
    model = YOLO(weights_path)

    # 2. Execute the evaluation specifically on the 'test' split
    metrics = model.val(
        data=dataset_yaml,
        split='test',             # CRITICAL: Forces the engine to use images/test
        imgsz=img_size,
        device=device,
        batch=16,                 # Adjust down to 8 or 4 if you run out of memory
        conf=0.25,                # Confidence threshold for detections
        iou=0.45,                 # NMS IoU threshold 
        save_json=False,          # Set to True if you need raw JSON coordinate outputs
        name="test_evaluation"    # The folder name it will save results into
    )

    # 3. Print the final results to the terminal
    print("\n" + "="*60)
    print(" 🎯 FINAL TEST DATASET METRICS")
    print("="*60)
    # YOLOv8 metrics objects contain arrays for map50-95, map50, map75
    print(f"mAP@50:       {metrics.box.map50:.4f}")
    print(f"mAP@50-95:    {metrics.box.map:.4f}")
    print(f"Results saved: {metrics.save_dir}")
    print("="*60)

if __name__ == "__main__":
    # --- UPDATE THESE PATHS TO MATCH YOUR LOCAL SYSTEM ---
    
    # The path to your trained model weights (usually inside runs/detect/.../weights/best.pt)
    WEIGHTS_FILE = "runs/detect/train/weights/best.pt" 
    
    # The path to the dataset.yaml we generated in the previous script
    DATASET_YAML = "datasets/cowc_1024_persam_refined/dataset.yaml"
    
    # The resolution the model was trained at (e.g., 512 or 1024)
    IMAGE_SIZE = 1024
    
    # -----------------------------------------------------

    weights_path = Path(WEIGHTS_FILE)
    yaml_path = Path(DATASET_YAML)

    if not weights_path.exists():
        print(f"❌ ERROR: Cannot find trained weights at: {weights_path}")
    elif not yaml_path.exists():
        print(f"❌ ERROR: Cannot find dataset configuration at: {yaml_path}")
    else:
        evaluate_test_set(str(weights_path), str(yaml_path), img_size=IMAGE_SIZE)
