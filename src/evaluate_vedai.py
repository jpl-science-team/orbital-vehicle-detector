import os
import glob
import cv2
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from shapely.geometry import MultiPoint
from ultralytics import YOLO

warnings.filterwarnings("ignore", category=UserWarning)

def parse_args():
    parser = argparse.ArgumentParser(description="VEDAI 15cm Baseline Evaluator for Deployment")
    parser.add_argument("--model", type=str, default="models/weights/test_weights/YOLO11n_15GSD.pt", help="Path to YOLO weights")
    parser.add_argument("--test-dir", type=str, default="test_data", help="Sandbox test data directory containing VEDAI subfolder")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for deployment baseline")
    parser.add_argument("--iou-thr", type=float, default=0.50, help="IoU threshold for matching")
    parser.add_argument("--imgsz", type=int, default=512, help="Image size forced to 512 for target model evaluation")
    return parser.parse_args()


def parse_vedai_label(label_path):
    """Parses VEDAI labels into target classes (Car=1, Pickup=11 -> class 0) and ignored classes (-1)."""
    gt_items = []
    if not os.path.exists(label_path):
        return gt_items
    
    target_classes = [1, 11]  # Target vehicles
    
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            try:
                if len(parts) >= 14:
                    original_class = int(parts[3])
                    eval_class = 0 if original_class in target_classes else -1
                    
                    x_coords = [float(x) for x in parts[6:10]]
                    y_coords = [float(y) for y in parts[10:14]]
                    
                    polygon = [
                        x_coords[0], y_coords[0], 
                        x_coords[1], y_coords[1], 
                        x_coords[2], y_coords[2], 
                        x_coords[3], y_coords[3]
                    ]
                    gt_items.append({"class": eval_class, "polygon": polygon})
            except (ValueError, IndexError):
                continue
    return gt_items


def compute_polygon_iou(poly1_coords, poly2_coords):
    """Calculates IoU between two 8-point polygons using Shapely convex hull."""
    try:
        pts1 = np.array(poly1_coords, dtype=np.float32).reshape(4, 2)
        pts2 = np.array(poly2_coords, dtype=np.float32).reshape(4, 2)
        p1 = MultiPoint(pts1).convex_hull
        p2 = MultiPoint(pts2).convex_hull

        if not p1.is_valid or not p2.is_valid or p1.area == 0 or p2.area == 0:
            return 0.0

        intersection = p1.intersection(p2).area
        union = p1.area + p2.area - intersection
        return float(intersection / union) if union > 0 else 0.0
    except Exception:
        return 0.0


def filter_ignored_predictions(pred_polys, pred_scores, ground_truths, iou_threshold=0.5):
    """Filters out predictions that match non-target / DontCare VEDAI objects so model isn't penalized."""
    valid_predictions = []
    ignore_gts = [gt for gt in ground_truths if gt['class'] == -1]
    
    for p_box, score in zip(pred_polys, pred_scores):
        hit_ignore = False
        for ig_gt in ignore_gts:
            if compute_polygon_iou(p_box, ig_gt['polygon']) > iou_threshold:
                hit_ignore = True
                break
        if not hit_ignore:
            valid_predictions.append({"polygon": p_box, "score": score})
            
    return valid_predictions, [gt['polygon'] for gt in ground_truths if gt['class'] == 0]


def match_polygons(pred_polys, pred_scores, gt_boxes, iou_thr=0.5):
    """Greedy matching of predicted polygons against ground truth polygons."""
    matched_gt = set()
    tps, fps, fns = [], [], []

    sort_indices = np.argsort(pred_scores)[::-1]

    for idx in sort_indices:
        p_box = pred_polys[idx]
        score = pred_scores[idx]
        best_iou = 0.0
        best_gt_idx = -1

        for g_idx, g_box in enumerate(gt_boxes):
            if g_idx in matched_gt:
                continue
            iou = compute_polygon_iou(p_box, g_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx

        if best_iou >= iou_thr and best_gt_idx != -1:
            matched_gt.add(best_gt_idx)
            tps.append({"pred_box": p_box, "gt_box": gt_boxes[best_gt_idx], "score": score, "iou": best_iou})
        else:
            fps.append({"pred_box": p_box, "score": score})

    for g_idx, g_box in enumerate(gt_boxes):
        if g_idx not in matched_gt:
            fns.append({"gt_box": g_box})

    return tps, fps, fns


def generate_baseline_plots(total_tps, total_fps, total_fns, all_pred_scores, all_tp_flags, output_dir, model_name):
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # 1. Confusion Matrix Plot
    # Matrix structure: [[TP, FP], [FN, TN (N/A for detection)]]
    cm = np.array([[total_tps, total_fps], [total_fns, 0]])
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    labels = ["Vehicle", "Background"]

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax[0])
    ax[0].set_title(f"[{model_name}] Confusion Matrix (Counts)", fontweight="bold")
    ax[0].set_ylabel("True Class")
    ax[0].set_xlabel("Predicted Class")

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax[1])
    ax[1].set_title(f"[{model_name}] Confusion Matrix (Normalized)", fontweight="bold")
    ax[1].set_ylabel("True Class")
    ax[1].set_xlabel("Predicted Class")
    plt.tight_layout()
    plt.savefig(plots_dir / "confusion_matrix.png", dpi=300)
    plt.close()

    # 2. Precision-Recall & F1 Confidence Curves
    if len(all_pred_scores) > 0:
        thresholds = np.linspace(0.01, 0.99, 100)
        p_list, r_list, f1_list = [], [], []

        for th in thresholds:
            tp_th = sum(1 for score, is_tp in zip(all_pred_scores, all_tp_flags) if score >= th and is_tp)
            fp_th = sum(1 for score, is_tp in zip(all_pred_scores, all_tp_flags) if score >= th and not is_tp)
            fn_th = total_tps + total_fns - tp_th

            p = tp_th / (tp_th + fp_th) if (tp_th + fp_th) > 0 else 1.0
            r = tp_th / (tp_th + fn_th) if (tp_th + fn_th) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

            p_list.append(p)
            r_list.append(r)
            f1_list.append(f1)

        # Precision-Recall Curve
        plt.figure(figsize=(7, 6))
        plt.plot(r_list, p_list, color='b', lw=2)
        plt.title(f"[{model_name}] Precision-Recall Curve (15cm_base)", fontweight="bold")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.xlim(0, 1)
        plt.ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(plots_dir / "precision_recall_curve.png", dpi=300)
        plt.close()

        # F1 Confidence Curve
        plt.figure(figsize=(8, 5))
        plt.plot(thresholds, f1_list, color='g', lw=2, label="F1 Score")
        plt.plot(thresholds, p_list, color='b', lw=1.5, linestyle="--", label="Precision")
        plt.plot(thresholds, r_list, color='r', lw=1.5, linestyle="--", label="Recall")
        plt.title(f"[{model_name}] Metrics vs Confidence Threshold", fontweight="bold")
        plt.xlabel("Confidence Threshold")
        plt.ylabel("Score")
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(plots_dir / "confidence_metrics_curve.png", dpi=300)
        plt.close()


def main():
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model weights not found at '{args.model}'")
        return

    model_name = model_path.stem
    output_dir = Path(f"VEDAI_15cm_512_Baseline_{model_name}")
    output_dir.mkdir(exist_ok=True)

    img_dir = Path(args.test_dir) / "VEDAI" / "15cm_base" / "images"
    lbl_dir = Path(args.test_dir) / "VEDAI" / "15cm_base" / "labels"

    if not img_dir.exists():
        print(f"Error: 15cm_base directory not found at '{img_dir.resolve()}'")
        return

    img_paths = sorted(glob.glob(str(img_dir / "*.png")) + glob.glob(str(img_dir / "*.jpg")) + glob.glob(str(img_dir / "*.tif")))
    if not img_paths:
        print(f"Error: No images found in '{img_dir.resolve()}'")
        return

    model = YOLO(args.model)
    print(f"=== Running Deployment Baseline Evaluation ===")
    print(f"Model: {model_name} | Dataset: VEDAI 15cm_base | Image Size: {args.imgsz}")

    total_tps, total_fps, total_fns = 0, 0, 0
    all_ious, all_pred_scores, all_tp_flags = [], [], []

    for img_p in img_paths:
        stem = Path(img_p).stem
        gt_items = parse_vedai_label(lbl_dir / f"{stem}.txt")

        raw_img = cv2.imread(img_p)
        if raw_img is None:
            continue

        results = model.predict(img_p, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]

        pred_polys, pred_scores = [], []
        if hasattr(results, 'obb') and results.obb is not None:
            boxes_obb = results.obb.xyxyxyxy.cpu().numpy()
            pred_polys = [b.reshape(8).tolist() for b in boxes_obb]
            pred_scores = results.obb.conf.cpu().numpy().tolist()

        valid_preds, valid_gts = filter_ignored_predictions(pred_polys, pred_scores, gt_items, iou_threshold=args.iou_thr)
        f_polys = [p['polygon'] for p in valid_preds]
        f_scores = [p['score'] for p in valid_preds]

        tps, fps, fns = match_polygons(f_polys, f_scores, valid_gts, iou_thr=args.iou_thr)

        total_tps += len(tps)
        total_fps += len(fps)
        total_fns += len(fns)

        for tp in tps:
            all_pred_scores.append(tp["score"])
            all_tp_flags.append(True)
            all_ious.append(tp["iou"])

        for fp in fps:
            all_pred_scores.append(fp["score"])
            all_tp_flags.append(False)

    prec = total_tps / (total_tps + total_fps) if (total_tps + total_fps) > 0 else 0.0
    rec = total_tps / (total_tps + total_fns) if (total_tps + total_fns) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    map50 = prec * rec
    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0

    print("\n--- Baseline Results ---")
    print(f"True Positives (TP): {total_tps}")
    print(f"False Positives (FP): {total_fps} (Ignored non-targets filtered out)")
    print(f"False Negatives (FN): {total_fns}")
    print(f"Precision:            {prec:.4f}")
    print(f"Recall:               {rec:.4f}")
    print(f"F1 Score:             {f1:.4f}")
    print(f"mAP@50:               {map50:.4f}")
    print(f"Mean IoU:             {mean_iou:.4f}")

    # Output baseline CSV report
    summary_df = pd.DataFrame([{
        "Model": model_name,
        "Dataset": "15cm_base",
        "ImgSize": args.imgsz,
        "TP": total_tps,
        "FP": total_fps,
        "FN": total_fns,
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1_Score": round(f1, 4),
        "mAP50": round(map50, 4),
        "Mean_IoU": round(mean_iou, 4)
    }])
    summary_df.to_csv(output_dir / "vedai_15cm_512_baseline.csv", index=False)

    print("\nGenerating baseline plots & confusion matrix...")
    generate_baseline_plots(total_tps, total_fps, total_fns, all_pred_scores, all_tp_flags, output_dir, model_name)
    print(f"Done! Results and plots saved to '{output_dir.resolve()}'")


if __name__ == "__main__":
    main()