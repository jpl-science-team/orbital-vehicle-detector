import os
import sys
import argparse
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Ensure local repository paths are accessible
sys.path.append(os.path.abspath('./O2-RT-DETR'))
from mmdet.apis import init_detector, inference_detector

# --- Safe Registry Monkey-Patch to prevent import crashes ---
try:
    import mmdet.structures.bbox.box_type as box_type
    _orig_register_box_converter = box_type._register_box_converter
    def safe_register_box_converter(*args, **kwargs):
        try: return _orig_register_box_converter(*args, **kwargs)
        except KeyError: return None
    box_type._register_box_converter = safe_register_box_converter
except Exception: pass
# --------------------------------------------------------------

# Attempt to load Shapely for Rotated IoU calculation
try:
    from shapely.geometry import Polygon
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False
    print("[!] Warning: Shapely is not installed. Rotated IoU evaluation will fall back to Horizontal IoU.")

def parse_args():
    parser = argparse.ArgumentParser(description="Orbital Vehicle Detector Research Suite")
    parser.add_argument('--config', default='O2-RT-DETR/projects/rotated_rtdetr/configs/cowc_rtdetr_512.py', help='Model config file')
    parser.add_argument('--checkpoint', required=True, help='Path to trained .pth weights')
    
    # Inputs
    parser.add_argument('--single_img', required=True, help='Path to the unannotated 50GSD Angle Fire image')
    parser.add_argument('--cowc_dir', default='data/cowc_512/val/images/', help='COWC test images')
    parser.add_argument('--cowc_anno', default='data/cowc_512/val/annfiles/', help='COWC centerpoint annotations')
    parser.add_argument('--vedai_color_dir', default='data/vedai/val/color_images/', help='VEDAI RGB images')
    parser.add_argument('--vedai_ir_dir', default='data/vedai/val/ir_images/', help='VEDAI Infrared images')
    parser.add_argument('--vedai_anno', default='data/vedai/val/annfiles/', help='VEDAI OBB annotations')
    
    # Thresholds
    parser.add_argument('--score_thr', type=float, default=0.3, help='Confidence threshold for predictions')
    parser.add_argument('--iou_threshold', type=float, default=0.5, help='IoU threshold to count as an OBB match')
    parser.add_argument('--dist_threshold', type=float, default=25.0, help='Pixel threshold to count as a centerpoint match')
    
    parser.add_argument('--output_dir', default='./research_analysis', help='Where to save plots and images')
    return parser.parse_args()

def calculate_hbb_iou(box1_pts, box2_pts):
    """Standard horizontal bounding box IoU fallback."""
    x1_min, y1_min = box1_pts.min(axis=0)
    x1_max, y1_max = box1_pts.max(axis=0)
    x2_min, y2_min = box2_pts.min(axis=0)
    x2_max, y2_max = box2_pts.max(axis=0)
    
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    inter_w = max(0, inter_xmax - inter_xmin)
    inter_h = max(0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h
    
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def calculate_rotated_iou(box1_pts, box2_pts):
    """Computes accurate Rotated IoU (RIoU) using Shapely."""
    if not HAS_SHAPELY:
        return calculate_hbb_iou(box1_pts, box2_pts)
    try:
        p1 = Polygon(box1_pts)
        p2 = Polygon(box2_pts)
        if not p1.is_valid or not p2.is_valid:
            return calculate_hbb_iou(box1_pts, box2_pts)
        intersection = p1.intersection(p2).area
        union = p1.area + p2.area - intersection
        return intersection / union if union > 0 else 0.0
    except Exception:
        return calculate_hbb_iou(box1_pts, box2_pts)

def get_rotated_box_points(box):
    """Converts prediction format [cx, cy, w, h, angle_rad] to 4 corner coordinates."""
    cx, cy, w, h, angle = box
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    corners = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
    rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated_corners = np.dot(corners, rot_matrix.T) + [cx, cy]
    return rotated_corners.astype(np.float32)

def load_annotations(anno_path):
    """
    Loads annotations and auto-detects format.
    Returns: (centers_array, boxes_pts_array)
    """
    gt_centers = []
    gt_boxes_pts = []
    if not os.path.exists(anno_path):
        return np.empty((0, 2)), np.empty((0, 4, 2))
        
    with open(anno_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            
            try:
                # Filter out trailing class label strings
                coords = []
                for p in parts:
                    try: coords.append(float(p))
                    except ValueError: break
                
                # Check parsed length to determine format
                if len(coords) >= 8:
                    # Oriented Bounding Box: [x1, y1, x2, y2, x3, y3, x4, y4]
                    pts = np.array(coords[:8]).reshape(4, 2)
                    gt_boxes_pts.append(pts)
                    gt_centers.append(pts.mean(axis=0))
                elif len(coords) >= 2:
                    # Raw Centerpoint: [cx, cy]
                    gt_centers.append([coords[0], coords[1]])
            except Exception:
                continue
                
    return np.array(gt_centers), np.array(gt_boxes_pts)

def match_by_distance(pred_centers, gt_centers, threshold):
    """Distance-based matching (for COWC)."""
    if len(gt_centers) == 0: return 0, len(pred_centers), 0
    if len(pred_centers) == 0: return 0, 0, len(gt_centers)
    
    dists = np.linalg.norm(pred_centers[:, None, :] - gt_centers[None, :, :], axis=2)
    matched_gts = set()
    tp = 0
    
    min_dist_idx = np.argsort(dists.min(axis=1))
    for p_idx in min_dist_idx:
        best_gt = np.argmin(dists[p_idx])
        if dists[p_idx, best_gt] < threshold and best_gt not in matched_gts:
            tp += 1
            matched_gts.add(best_gt)
            
    fp = len(pred_centers) - tp
    fn = len(gt_centers) - tp
    return tp, fp, fn

def match_by_iou(pred_boxes_pts, pred_scores, gt_boxes_pts, threshold=0.5):
    """Score-sorted Rotated IoU matching (for VEDAI)."""
    if len(gt_boxes_pts) == 0: return 0, len(pred_boxes_pts), 0
    if len(pred_boxes_pts) == 0: return 0, 0, len(gt_boxes_pts)
    
    # Sort predictions by score descending
    sort_idx = np.argsort(-pred_scores)
    pred_boxes_pts = pred_boxes_pts[sort_idx]
    
    matched_gts = set()
    tp = 0
    
    for p_pts in pred_boxes_pts:
        best_iou = -1.0
        best_gt_idx = -1
        for gt_idx, gt_pts in enumerate(gt_boxes_pts):
            iou = calculate_rotated_iou(p_pts, gt_pts)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
                
        if best_iou >= threshold and best_gt_idx not in matched_gts:
            tp += 1
            matched_gts.add(best_gt_idx)
            
    fp = len(pred_boxes_pts) - tp
    fn = len(gt_boxes_pts) - tp
    return tp, fp, fn

def evaluate_dataset(model, img_dir, anno_dir, res_factor=1.0, force_grayscale=False, force_ir=False, args=None):
    """Runs pipeline over chosen directory with dynamic resolution/color scaling."""
    if not os.path.exists(img_dir):
        return None
        
    total_tp, total_fp, total_fn = 0, 0, 0
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
    
    for f in img_files:
        img_path = os.path.join(img_dir, f)
        img = cv2.imread(img_path)
        if img is None: continue
        
        orig_h, orig_w = img.shape[:2]
        
        # Q4 Modification: Grayscale transform
        if force_grayscale:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.merge([gray, gray, gray])
            
        # Q1 Modification: Resolution resizing
        if res_factor != 1.0:
            new_w, new_h = int(orig_w * res_factor), int(orig_h * res_factor)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
        result = inference_detector(model, img)
        pred_instances = result.pred_instances
        keep = pred_instances.scores >= args.score_thr
        pred_boxes = pred_instances.bboxes[keep].cpu().numpy()
        pred_scores = pred_instances.scores[keep].cpu().numpy()
        
        # Build predicted box coordinates
        pred_boxes_pts = np.array([get_rotated_box_points(box) for box in pred_boxes]) if len(pred_boxes) > 0 else np.empty((0, 4, 2))
        
        # Rescale predictions back to baseline scale if scaled down
        if res_factor != 1.0 and len(pred_boxes_pts) > 0:
            pred_boxes_pts /= res_factor
            
        # Load corresponding ground truth
        anno_name = os.path.splitext(f)[0] + '.txt'
        gt_centers, gt_boxes_pts = load_annotations(os.path.join(anno_dir, anno_name))
        
        if len(gt_boxes_pts) > 0:
            # Human Oriented BBoxes (VEDAI) -> Evaluation via True Rotated IoU
            tp, fp, fn = match_by_iou(pred_boxes_pts, pred_scores, gt_boxes_pts, args.iou_threshold)
        else:
            # Automated Centerpoints (COWC) -> Evaluation via Center Distance
            pred_centers = pred_boxes[:, :2] if len(pred_boxes) > 0 else np.empty((0, 2))
            if res_factor != 1.0 and len(pred_centers) > 0:
                pred_centers /= res_factor
            tp, fp, fn = match_by_distance(pred_centers, gt_centers, args.dist_threshold)
            
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {"precision": precision, "recall": recall, "f1": f1, "fp": total_fp, "tp": total_tp}

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("[-] Spawning detector engine from checkpoints...")
    model = init_detector(args.config, args.checkpoint, device='cuda:0')
    print("[+] Framework loaded successfully.")

    # ==========================================================
    # STEP 1: VISUAL CONFIRMATION (Single 50GSD Unannotated Image)
    # ==========================================================
    print(f"\n[-] Running visual check on unannotated image: {args.single_img}")
    single_img = cv2.imread(args.single_img)
    if single_img is not None:
        vis_result = inference_detector(model, single_img)
        instances = vis_result.pred_instances
        keep = instances.scores >= args.score_thr
        boxes = instances.bboxes[keep].cpu().numpy()
        scores = instances.scores[keep].cpu().numpy()
        
        canvas = single_img.copy()
        for box, score in zip(boxes, scores):
            pts = get_rotated_box_points(box).astype(np.int32)
            cv2.polylines(canvas, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(canvas, f"{score:.2f}", (int(box[0]), int(box[1])), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
                        
        vis_path = os.path.join(args.output_dir, "visual_predictions_angle_fire.png")
        cv2.imwrite(vis_path, canvas)
        print(f"[+] Visual confirmation output saved to: {vis_path}")
    else:
        print(f"[!] Warning: Unable to open unannotated image at {args.single_img}")

    # ==========================================================
    # STEP 2: MULTI-RESOLUTION & SPECTRUM EXPERIMENTS
    # ==========================================================
    print("\n[-] Starting Grid Experiments across scale and color spectrums...")
    
    # Resolutions to test: 100%, 75%, 50%, 25% (Simulates spatial GSD degradation)
    resolutions = [1.0, 0.75, 0.5, 0.25]
    
    # Track metrics
    results_cowc = []
    results_vedai_rgb = []
    results_vedai_gray = []
    results_vedai_ir = []

    for res in tqdm(resolutions, desc="Resolutions Sweeps"):
        # 1. COWC (Centerpoint evaluation)
        res_cowc = evaluate_dataset(model, args.cowc_dir, args.cowc_anno, res_factor=res, args=args)
        if res_cowc: results_cowc.append(res_cowc)
        
        # 2. VEDAI Color (True Rotated IoU evaluation)
        res_vrgb = evaluate_dataset(model, args.vedai_color_dir, args.vedai_anno, res_factor=res, force_grayscale=False, args=args)
        if res_vrgb: results_vedai_rgb.append(res_vrgb)
        
        # 3. VEDAI Grayscale (Strip color)
        res_vgray = evaluate_dataset(model, args.vedai_color_dir, args.vedai_anno, res_factor=res, force_grayscale=True, args=args)
        if res_vgray: results_vedai_gray.append(res_vgray)
        
        # 4. VEDAI Infrared (IR)
        res_vir = evaluate_dataset(model, args.vedai_ir_dir, args.vedai_anno, res_factor=res, force_grayscale=False, args=args)
        if res_vir: results_vedai_ir.append(res_vir)

    # ==========================================================
    # STEP 3: ANALYTICAL REPORT PLOTTING
    # ==========================================================
    print("\n[-] Compiling visual graphs and scientific metrics...")
    
    # Chart A: Resolution Degradation Slopes (Question 1 & 3)
    plt.figure(figsize=(10, 6))
    if results_cowc:
        plt.plot(resolutions, [x['f1'] for x in results_cowc], '-o', label='COWC (Distance)', color='teal')
    if results_vedai_rgb:
        plt.plot(resolutions, [x['f1'] for x in results_vedai_rgb], '-s', label='VEDAI RGB (Rotated IoU)', color='blue')
    if results_vedai_ir:
        plt.plot(resolutions, [x['f1'] for x in results_vedai_ir], '-^', label='VEDAI Infrared (Rotated IoU)', color='red')
    
    plt.title('Performance Scaling Across Spatial Resolution Degradation')
    plt.xlabel('Resolution Scale Factor (Simulated Ground Sample Distance)')
    plt.ylabel('F1 Target Tracking Metric')
    plt.gca().invert_xaxis()  # Order from 100% to 25% scale
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'resolution_scaling_f1.png'), dpi=300)
    plt.close()

    # Chart B: False Positive Clutter Analysis (Question 4)
    plt.figure(figsize=(10, 6))
    x_axis = np.arange(len(resolutions))
    
    if results_vedai_rgb and results_vedai_gray:
        plt.bar(x_axis - 0.2, [x['fp'] for x in results_vedai_rgb], 0.4, label='Color (RGB) False Positives', color='royalblue')
        plt.bar(x_axis + 0.2, [x['fp'] for x in results_vedai_gray], 0.4, label='Grayscale (Panchromatic) False Positives', color='darkorange')
        plt.xticks(x_axis, [f"{r*100:.0f}%" for r in resolutions])
        plt.title('Chromatic Vulnerability vs. Shape Induction (False Positive Swings)')
        plt.xlabel('Resolution Scale Factor')
        plt.ylabel('Total Count of False Positive Clutter')
        plt.legend()
        plt.savefig(os.path.join(args.output_dir, 'chromatic_fp_clutter.png'), dpi=300)
        plt.close()

    print(f"\n[+] Analysis phase complete! All graphs saved to: '{args.output_dir}/'")
    
    # Quick Print of Baseline Figures
    if results_vedai_rgb:
        print(f"    - Baseline VEDAI RGB (True RIoU): {results_vedai_rgb[0]['f1']:.4f}")
    if results_vedai_ir:
        print(f"    - Baseline VEDAI Infrared (True RIoU): {results_vedai_ir[0]['f1']:.4f}")
    if results_cowc:
        print(f"    - Baseline COWC (Distance): {results_cowc[0]['f1']:.4f}")

if __name__ == '__main__':
    main()