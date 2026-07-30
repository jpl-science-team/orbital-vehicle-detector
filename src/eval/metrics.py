# src/eval/metrics.py
import numpy as np
from shapely.geometry import MultiPoint, Point
from sklearn.metrics import auc

def compute_polygon_iou(poly1_coords, poly2_coords):
    try:
        p1 = MultiPoint(np.array(poly1_coords, dtype=np.float32).reshape(4, 2)).convex_hull
        p2 = MultiPoint(np.array(poly2_coords, dtype=np.float32).reshape(4, 2)).convex_hull
        if not p1.is_valid or not p2.is_valid or p1.area == 0 or p2.area == 0: return 0.0
        intersection = p1.intersection(p2).area
        union = p1.area + p2.area - intersection
        return float(intersection / union) if union > 0 else 0.0
    except Exception: return 0.0

def get_poly_area(poly_coords):
    try: return float(MultiPoint(np.array(poly_coords, dtype=np.float32).reshape(4, 2)).convex_hull.area)
    except Exception: return 0.0

def filter_ignored_predictions(pred_polys, pred_scores, ground_truths, iou_threshold=0.5):
    valid_predictions = []
    valid_gts = [gt['polygon'] for gt in ground_truths if gt['class'] == 0]
    ignore_gts = [gt for gt in ground_truths if gt['class'] == -1]
    for p_box, score in zip(pred_polys, pred_scores):
        if not any(compute_polygon_iou(p_box, ig_gt['polygon']) > iou_threshold for ig_gt in ignore_gts):
            valid_predictions.append({"polygon": p_box, "score": score})
    return valid_predictions, valid_gts

def match_cowc_points(pred_polys, pred_scores, gt_points, max_dist=25.0):
    matched_gt, tps, fps, fns, match_distances = set(), [], [], [], []
    sort_indices = np.argsort(pred_scores)[::-1]
    for idx in sort_indices:
        p_box, score = pred_polys[idx], pred_scores[idx]
        try:
            pts = np.array(p_box, dtype=np.float32).reshape(4, 2)
            pred_poly, pred_center, valid_poly = MultiPoint(pts).convex_hull, np.mean(pts, axis=0), MultiPoint(pts).convex_hull.is_valid
        except Exception:
            valid_poly, pred_center = False, np.mean(np.array(p_box, dtype=np.float32).reshape(-1, 2), axis=0)
        best_dist, best_gt_idx = float('inf'), -1
        for g_idx, (gx, gy) in enumerate(gt_points):
            if g_idx in matched_gt: continue
            dist = float(np.linalg.norm(pred_center - np.array([gx, gy])))
            if (valid_poly and pred_poly.contains(Point(gx, gy))) or dist <= max_dist:
                if dist < best_dist: best_dist, best_gt_idx = dist, g_idx
        if best_gt_idx != -1:
            matched_gt.add(best_gt_idx)
            tps.append({"pred_box": p_box, "gt_pt": gt_points[best_gt_idx], "score": score, "dist": best_dist})
            match_distances.append(best_dist)
        else: fps.append({"pred_box": p_box, "score": score})
    fns = [{"gt_pt": pt} for g_idx, pt in enumerate(gt_points) if g_idx not in matched_gt]
    return tps, fps, fns, match_distances

def match_polygons(pred_polys, pred_scores, gt_boxes, iou_thr=0.5):
    matched_gt, tps, fps, fns, match_ious = set(), [], [], [], []
    sort_indices = np.argsort(pred_scores)[::-1]
    for idx in sort_indices:
        p_box, score = pred_polys[idx], pred_scores[idx]
        best_iou, best_gt_idx = 0.0, -1
        for g_idx, g_box in enumerate(gt_boxes):
            if g_idx in matched_gt: continue
            iou = compute_polygon_iou(p_box, g_box)
            if iou > best_iou: best_iou, best_gt_idx = iou, g_idx
        if best_iou >= iou_thr and best_gt_idx != -1:
            matched_gt.add(best_gt_idx)
            tps.append({"pred_box": p_box, "gt_box": gt_boxes[best_gt_idx], "score": score, "iou": best_iou})
            match_ious.append(best_iou)
        else: fps.append({"pred_box": p_box, "score": score})
    fns = [{"gt_box": g_box} for g_idx, g_box in enumerate(gt_boxes) if g_idx not in matched_gt]
    return tps, fps, fns, match_ious

def get_peak_metrics_and_roc(var_preds, total_gts):
    """Sweeps for max F1 and computes mAP50 using full low-conf predictions."""
    best_f1, best_conf, best_p, best_r = 0.0, 0.01, 0.0, 0.0
    for th in np.linspace(0.01, 0.90, 50):
        th_tps = sum(1 for p in var_preds if p['score'] >= th and p['is_tp'])
        th_fps = sum(1 for p in var_preds if p['score'] >= th and not p['is_tp'])
        th_fns = total_gts - th_tps
        p = th_tps / (th_tps + th_fps) if (th_tps + th_fps) > 0 else 0.0
        r = th_tps / (th_tps + th_fns) if (th_tps + th_fns) > 0 else 0.0
        th_f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
        if th_f1 > best_f1:
            best_f1, best_conf, best_p, best_r = th_f1, th, p, r

    var_preds_sorted = sorted(var_preds, key=lambda x: x['score'], reverse=True)
    precisions, recalls, run_tps, run_fps = [1.0], [0.0], 0, 0
    for p_data in var_preds_sorted:
        if p_data['is_tp']: run_tps += 1
        else: run_fps += 1
        precisions.append(run_tps / (run_tps + run_fps))
        recalls.append(run_tps / total_gts if total_gts > 0 else 0.0)
    map50 = auc(recalls, precisions) if len(recalls) > 1 else 0.0

    return {
        "best_conf": best_conf,
        "TP": sum(1 for p in var_preds if p['score'] >= best_conf and p['is_tp']),
        "FP": sum(1 for p in var_preds if p['score'] >= best_conf and not p['is_tp']),
        "FN": total_gts - sum(1 for p in var_preds if p['score'] >= best_conf and p['is_tp']),
        "Precision": best_p, "Recall": best_r, "F1_Score": best_f1, "mAP50": map50
    }