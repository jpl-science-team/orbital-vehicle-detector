# src/eval/parsers.py
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="RQ1 Resolution Scaling Evaluator - YOLO")
    
    parser.add_argument("--models", nargs="+", required=True,
                        help="One or more YOLO .pt weights to evaluate, all trained at --train-imgsz")
    parser.add_argument("--test-dir", type=str, required=True,
                        help="Test data folder from README Step 5.1 (e.g. test_data)")
    parser.add_argument("--conf", type=float, default=0.01, help="Low threshold strictly for ROC generation")
    parser.add_argument("--iou-thr", type=float, default=0.50)
    parser.add_argument("--cowc-scoring", choices=["box", "point"], default="box",
                        help="Match COWC predictions to the label boxes by IoU >= --iou-thr (box), "
                             "or to the box centers within --dist-thr pixels (point; the pre-2026-11-23 behavior)")
    parser.add_argument("--dist-thr", type=float, default=25.0,
                        help="COWC point-match distance in pixels at 15cm GSD (scaled per variant to keep the ground distance fixed)")
    parser.add_argument("--cowc-dir", type=str, default=None,
                        help="COWC folder name under --test-dir. Default: COWC (label-refiner boxes) for box scoring, "
                             "COWC_point (test_data.zip center points) for point scoring")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="Fixed inference size for every image (overrides --train-imgsz scaling)")
    parser.add_argument("--train-imgsz", type=int, required=True,
                        help="--img_size the model was trained at (train.py); each image is enlarged by train-imgsz/train-tile-size at inference")
    parser.add_argument("--train-tile-size", type=int, default=512,
                        help="Size of the training tiles (cowc_512 = 512)")
    parser.add_argument("--threshold-source", choices=["val", "test"], default="val",
                        help="Pick each variant's F1 confidence threshold on matching degraded copies of the validation set (val), "
                             "or on the test data itself (test; optimistic, the pre-2026-11 behavior)")
    parser.add_argument("--val-data", type=str, required=True,
                        help="Ultralytics dataset.yaml whose 'val' split (15cm GSD) is used to calibrate thresholds "
                             "(e.g. data/cowc_512/dataset.yaml); unused with --threshold-source test")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Only evaluate these datasets (e.g. VEDAI). Default: all in DATASET_VARIANTS")
    parser.add_argument("--variants", nargs="+", default=None,
                        help="Only evaluate these variants (e.g. 15cm_base). Default: all variants of each dataset")
    parser.add_argument("--image-glob", type=str, default="*.*",
                        help="Glob for image files inside each variant's images/ folder (e.g. '*_co.*' for VEDAI color only)")
    parser.add_argument("--out-dir", type=str, default=".",
                        help="Folder to write results to; each model gets its own Eval_Results_YOLO_<model_name>[_<tag>]/ inside it")
    parser.add_argument("--tag", type=str, default="",
                        help="Suffix for the output folder, so subset runs don't overwrite full runs")
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

def cowc_dir_name(args):
    """Box labels live in test_data/COWC, the test_data.zip center points in test_data/COWC_point."""
    return args.cowc_dir or ("COWC_point" if args.cowc_scoring == "point" else "COWC")

def parse_label_file(label_path, is_cowc=False, is_vedai=False, cowc_boxes=False):
    """COWC labels are "0 x1 y1 .. x4 y4" OBBs in pixels; returns their center points unless cowc_boxes is set."""
    gt_items = []
    if not os.path.exists(label_path):
        return gt_items
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            try:
                if is_cowc:
                    if len(parts) >= 9:
                        polygon = [float(x) for x in parts[1:9]]
                        gt_items.append(polygon if cowc_boxes else [sum(polygon[0::2]) / 4, sum(polygon[1::2]) / 4])
                    elif cowc_boxes:
                        raise RuntimeError(f"{label_path} has COWC center points, not boxes; rerun src/preprocess_rq1_datasets.py")
                    elif len(parts) >= 3: gt_items.append([float(parts[1]), float(parts[2])])
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