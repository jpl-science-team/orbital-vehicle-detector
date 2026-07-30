# src/eval/parsers.py
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="RQ1 Resolution Scaling Evaluator - YOLO")
    
    # Changed to --models with nargs="+" to accept multiple paths
    parser.add_argument("--models", nargs="+", 
                        default=[
                            "models/weights/test_weights/YOLO11n_15GSD.pt",
                            # Add more model paths here to evaluate them in batch:
                            # "models/weights/test_weights/YOLO11s_MultiScale.pt",
                            # "models/weights/test_weights/YOLO11m_FocalLoss.pt"
                        ], 
                        help="List of model paths to evaluate")
                        
    parser.add_argument("--test-dir", type=str, default="test_data")
    parser.add_argument("--conf", type=float, default=0.01, help="Low threshold strictly for ROC generation")
    parser.add_argument("--iou-thr", type=float, default=0.50)
    parser.add_argument("--dist-thr", type=float, default=25.0)
    parser.add_argument("--imgsz", type=int, default=None)
    return parser.parse_args()

def parse_gsd_and_deg_type(variant_name):
    if "50cm" in variant_name: gsd = 50
    elif "15cm" in variant_name: gsd = 15
    elif "30cm" in variant_name: gsd = 30
    elif "60cm" in variant_name: gsd = 60
    else: gsd = 15

    if "base" in variant_name: deg_type = "Base"
    elif "upscale" in variant_name: deg_type = "Upscale"
    elif "naive" in variant_name: deg_type = "Naive"
    elif "scalenorm" in variant_name: deg_type = "Scale-Normalized"
    else: deg_type = "Base"

    return gsd, deg_type

def parse_label_file(label_path, is_cowc=False, is_vedai=False):
    gt_items = []
    if not os.path.exists(label_path):
        return gt_items
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            try:
                if is_cowc:
                    if len(parts) >= 3: gt_items.append([float(parts[1]), float(parts[2])])
                    elif len(parts) == 2: gt_items.append([float(parts[0]), float(parts[1])])
                elif is_vedai:
                    if len(parts) >= 14:
                        eval_class = 0 if int(parts[3]) in [1, 11] else -1
                        x_coords, y_coords = [float(x) for x in parts[6:10]], [float(y) for y in parts[10:14]]
                        polygon = [x_coords[0], y_coords[0], x_coords[1], y_coords[1], 
                                   x_coords[2], y_coords[2], x_coords[3], y_coords[3]]
                        gt_items.append({"class": eval_class, "polygon": polygon})
                else:
                    if len(parts) >= 8: gt_items.append([float(x) for x in parts[:8]])
            except (ValueError, IndexError):
                continue
    return gt_items