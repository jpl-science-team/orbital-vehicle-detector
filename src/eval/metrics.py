# src/eval/metrics.py
import numpy as np
from shapely.geometry import MultiPoint, Point

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

def compute_ap(recalls, precisions):
    """COCO-style 101-point interpolated AP over the precision envelope (matches Ultralytics' compute_ap)."""
    mrec = np.concatenate(([0.0], recalls, [recalls[-1] if len(recalls) else 1.0], [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0], [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))  # precision envelope
    x = np.linspace(0, 1, 101)
    return float(np.trapz(np.interp(x, mrec, mpre), x))

def counts_at_threshold(var_preds, total_gts, th):
    """TP/FP/FN, precision, recall and F1 when keeping predictions with score >= th."""
    tps = sum(1 for p in var_preds if p['score'] >= th and p['is_tp'])
    fps = sum(1 for p in var_preds if p['score'] >= th and not p['is_tp'])
    fns = total_gts - tps
    prec = tps / (tps + fps) if (tps + fps) > 0 else 0.0
    rec = tps / (tps + fns) if (tps + fns) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return tps, fps, fns, prec, rec, f1

def find_best_f1_threshold(var_preds, total_gts):
    """Sweeps confidence thresholds and returns (threshold, F1) with the highest F1."""
    best_conf, best_f1 = 0.01, 0.0
    for th in np.linspace(0.01, 0.90, 50):
        f1 = counts_at_threshold(var_preds, total_gts, th)[5]
        if f1 > best_f1:
            best_conf, best_f1 = float(th), f1
    return best_conf, best_f1

def get_peak_metrics_and_roc(var_preds, total_gts, conf_thr=None):
    """Computes counts/P/R/F1 at conf_thr (e.g. a validation-calibrated threshold) and mAP50 from the full
    low-conf predictions. If conf_thr is None, the test-set-optimal threshold is used. Oracle_* always report
    the best F1 achievable on this data, for comparison."""
    oracle_conf, oracle_f1 = find_best_f1_threshold(var_preds, total_gts)
    best_conf = oracle_conf if conf_thr is None else conf_thr
    tps, fps, fns, best_p, best_r, best_f1 = counts_at_threshold(var_preds, total_gts, best_conf)

    var_preds_sorted = sorted(var_preds, key=lambda x: x['score'], reverse=True)
    precisions, recalls, run_tps, run_fps = [], [], 0, 0
    for p_data in var_preds_sorted:
        if p_data['is_tp']: run_tps += 1
        else: run_fps += 1
        precisions.append(run_tps / (run_tps + run_fps))
        recalls.append(run_tps / total_gts if total_gts > 0 else 0.0)
    map50 = compute_ap(recalls, precisions) if total_gts > 0 else 0.0

    return {
        "best_conf": best_conf,
        "TP": tps, "FP": fps, "FN": fns,
        "Precision": best_p, "Recall": best_r, "F1_Score": best_f1, "mAP50": map50,
        "Oracle_conf": oracle_conf, "Oracle_F1": oracle_f1
    }