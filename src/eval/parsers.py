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
    parser.add_argument("--dist-thr", type=float, default=25.0,
                        help="COWC point-match distance in pixels at 15cm GSD (scaled per variant to keep the ground distance fixed)")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="Fixed inference size for every image (overrides --train-imgsz scaling)")
    parser.add_argument("--train-imgsz", type=int, default=512,
                        help="imgsz the model was trained at; each image is enlarged by train-imgsz/train-tile-size at inference")
    parser.add_argument("--train-tile-size", type=int, default=512,
                        help="Size of the training tiles (cowc_512 = 512)")
    parser.add_argument("--threshold-source", choices=["val", "test"], default="val",
                        help="Pick each variant's F1 confidence threshold on matching degraded copies of the validation set (val), "
                             "or on the test data itself (test; optimistic, the pre-2026-11 behavior)")
    parser.add_argument("--val-data", type=str, default="data/cowc_512/dataset.yaml",
                        help="Ultralytics dataset.yaml whose 'val' split (15cm GSD) is used to calibrate thresholds")
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

def get_pixel_gsd(variant_name):
    """GSD (cm/px) of the saved image pixels. Scale-normalized variants are resampled back to the 15cm grid."""
    gsd, deg_type = parse_gsd_and_deg_type(variant_name)
    return 15 if deg_type == "Scale-Normalized" else gsd

def get_content_gsd(dataset, variant_name):
    """GSD (cm/px) of the image content, before any resampling back up. Angel is natively 50cm, so its upscale variants keep 50cm content."""
    if dataset == "Angel": return 50
    return parse_gsd_and_deg_type(variant_name)[0]

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