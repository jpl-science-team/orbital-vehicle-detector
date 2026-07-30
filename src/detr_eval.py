import os
import sys
import re
import glob
import cv2
import inspect
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from functools import partial

# Import directly from your modular eval package
from eval.config import DATASET_VARIANTS
from eval.parsers import parse_label_file, parse_gsd_and_deg_type
from eval.metrics import (
    filter_ignored_predictions, 
    match_cowc_points, 
    match_polygons, 
    get_peak_metrics_and_roc, 
    get_poly_area
)
from eval.visualization import (
    draw_annotations, 
    generate_powerpoint_plots, 
    generate_extended_plots
)

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------
# 1. AUTO-PATCHER & DETR DEPENDENCIES
# ---------------------------------------------------------
def patch_prob_iou():
    target_file = os.path.abspath('./O2-RT-DETR/projects/rotated_rtdetr/rotated_rtdetr/prob_iou.py')
    if os.path.exists(target_file):
        with open(target_file, 'r') as f:
            content = f.read()
        if '| np.ndarray' in content:
            fixed_content = re.sub(
                r'obb1:\s*torch\.Tensor\s*\|\s*np\.ndarray,\s*obb2:\s*torch\.Tensor\s*\|\s*np\.ndarray',
                'obb1, obb2', 
                content
            )
            with open(target_file, 'w') as f:
                f.write(fixed_content)

patch_prob_iou()

sys.path.append(os.path.abspath('./O2-RT-DETR'))
sys.path.append(os.path.abspath('./O2-RT-DETR/projects/rotated_rtdetr'))

import torch
import mmcv
import mmdet.structures.bbox.box_type as box_type_module
from mmengine.registry import TRANSFORMS
from mmdet.models.detectors.dino import DINO
from mmengine.config import Config
from mmdet.apis import init_detector, inference_detector

# --- GLOBAL PATCHES ---
_orig_register_box = box_type_module._register_box
def _force_register_box(name, box_type, *args, **kwargs):
    old_type_mappings = dict(box_type_module._box_type_to_name)
    kwargs['force'] = True
    res = _orig_register_box(name, box_type, *args, **kwargs)
    for b_type, b_info in old_type_mappings.items():
        if b_type not in box_type_module._box_type_to_name:
            box_type_module._box_type_to_name[b_type] = b_info
    return res
box_type_module._register_box = _force_register_box

_orig_register_converter = box_type_module._register_box_converter
def _force_register_converter(*args, **kwargs):
    kwargs['force'] = True
    return _orig_register_converter(*args, **kwargs)
box_type_module._register_box_converter = _force_register_converter

_orig_transforms_build = TRANSFORMS.build
def _auto_resolve_transforms_build(cfg, *args, **kwargs):
    try:
        return _orig_transforms_build(cfg, *args, **kwargs)
    except KeyError as e:
        t_name = cfg.get('type') if isinstance(cfg, dict) else getattr(cfg, 'type', None)
        if isinstance(t_name, str):
            clean_name = t_name.split('::')[-1]
            mod = None
            for module_path in ['mmdet.datasets.transforms', 'mmrotate.datasets.transforms', 'mmengine.dataset.transforms']:
                try:
                    imported_mod = __import__(module_path, fromlist=[clean_name])
                    if hasattr(imported_mod, clean_name):
                        mod = getattr(imported_mod, clean_name)
                        break
                except ImportError:
                    continue
            if mod is not None:
                TRANSFORMS.register_module(name=clean_name, module=mod, force=True)
                TRANSFORMS.register_module(name=f'ai4rs::{clean_name}', module=mod, force=True)
                TRANSFORMS.register_module(name=t_name, module=mod, force=True)
                return _orig_transforms_build(cfg, *args, **kwargs)
        raise e
TRANSFORMS.build = _auto_resolve_transforms_build

_orig_dino_forward_transformer = DINO.forward_transformer
def _patched_dino_forward_transformer(self, *args, **kwargs):
    if hasattr(self, 'decoder') and hasattr(self, 'bbox_head'):
        cls_b = getattr(self.bbox_head, 'cls_branches', None)
        reg_b = getattr(self.bbox_head, 'reg_branches', None)
        setattr(self.decoder, '_patch_cls_branches', cls_b)
        setattr(self.decoder, '_patch_reg_branches', reg_b)
        if not hasattr(self.decoder, '_forward_patched'):
            orig_decoder_forward = self.decoder.forward
            dec_sig = inspect.signature(orig_decoder_forward)
            dec_has_var = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in dec_sig.parameters.values())
            def _robust_decoder_forward(*d_args, **d_kwargs):
                if 'cls_branches' not in d_kwargs and 'cls_branches' in dec_sig.parameters:
                    d_kwargs['cls_branches'] = getattr(self.decoder, '_patch_cls_branches', None)
                if 'reg_branches' not in d_kwargs and 'reg_branches' in dec_sig.parameters:
                    d_kwargs['reg_branches'] = getattr(self.decoder, '_patch_reg_branches', None)
                if not dec_has_var:
                    valid_dec_params = set(dec_sig.parameters.keys())
                    d_kwargs = {k: v for k, v in d_kwargs.items() if k in valid_dec_params}
                return orig_decoder_forward(*d_args, **d_kwargs)
            self.decoder.forward = _robust_decoder_forward
            self.decoder._forward_patched = True
    orig_fd = getattr(self, 'forward_decoder', None)
    if orig_fd:
        fd_sig = inspect.signature(orig_fd)
        fd_has_var = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in fd_sig.parameters.values())
        if not fd_has_var:
            valid_fd_params = set(fd_sig.parameters.keys())
            def _robust_fd(*f_args, **f_kwargs):
                if 'cls_branches' in f_kwargs:
                    setattr(self.decoder, '_patch_cls_branches', f_kwargs['cls_branches'])
                if 'reg_branches' in f_kwargs:
                    setattr(self.decoder, '_patch_reg_branches', f_kwargs['reg_branches'])
                filtered_fd_kwargs = {k: v for k, v in f_kwargs.items() if k in valid_fd_params}
                return orig_fd(*f_args, **filtered_fd_kwargs)
            self.__dict__['forward_decoder'] = _robust_fd
    try:
        return _orig_dino_forward_transformer(self, *args, **kwargs)
    finally:
        self.__dict__.pop('forward_decoder', None)
DINO.forward_transformer = _patched_dino_forward_transformer
torch.load = partial(torch.load, weights_only=False)

# ---------------------------------------------------------
# 2. EVALUATION CONFIGURATION & HELPERS
# ---------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="RQ1 Resolution Evaluator - RT-DETR")
    parser.add_argument("--config", type=str, default="runs/20260715_rtdetr_obb_512/cowc_rtdetr_512.py", help="Path to MMDetection config file (.py)")
    parser.add_argument("--models", nargs="+", default=["runs/20260715_rtdetr_obb_512/best_dota_mAP_epoch_27.pth"], help="Path(s) to PyTorch model weights (.pth)")
    parser.add_argument("--test-dir", type=str, default="test_data", help="Sandbox test data directory")
    
    parser.add_argument("--conf", type=float, default=0.01, help="Confidence threshold (lowered for ROC generation)")
    parser.add_argument("--iou-thr", type=float, default=0.50, help="IoU threshold for polygon datasets")
    parser.add_argument("--dist-thr", type=float, default=25.0, help="Max pixel distance for COWC point matching")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run inference on")
    return parser.parse_args()

def obb_to_poly(obb):
    """Converts RT-DETR [cx, cy, w, h, angle] to standard 8-point polygon [x1, y1, x2, y2, x3, y3, x4, y4]"""
    if len(obb) == 8: return obb
    cx, cy, w, h, angle = obb[:5]
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    vec1 = np.array([w / 2 * cos_a, w / 2 * sin_a])
    vec2 = np.array([-h / 2 * sin_a, h / 2 * cos_a])
    pt1 = np.array([cx, cy]) + vec1 + vec2
    pt2 = np.array([cx, cy]) + vec1 - vec2
    pt3 = np.array([cx, cy]) - vec1 - vec2
    pt4 = np.array([cx, cy]) - vec1 + vec2
    return [pt1[0], pt1[1], pt2[0], pt2[1], pt3[0], pt3[1], pt4[0], pt4[1]]


# ---------------------------------------------------------
# 3. MAIN EVALUATION LOOP
# ---------------------------------------------------------
def main():
    args = parse_args()

    for model_path_str in args.models:
        model_path = Path(model_path_str)
        if not model_path.exists():
            print(f"\n[WARNING] Skipping model '{model_path_str}' - File not found.")
            continue
            
        model_name = model_path.stem
        output_dir = Path(f"Eval_Results_RTDETR_{model_name}")
        output_dir.mkdir(exist_ok=True)
        qual_dir = output_dir / "qualitative"
        qual_dir.mkdir(exist_ok=True)

        print(f"\n=======================================================")
        print(f" STARTING EVALUATION: {model_name} (RT-DETR)")
        print(f"=======================================================")
        
        # 1. Load and Patch Configuration
        print(f"Loading Config: {args.config}")
        cfg = Config.fromfile(args.config)
        
        if 'test_dataloader' in cfg and 'dataset' in cfg.test_dataloader:
            if cfg.test_dataloader.dataset.get('type') == 'DOTADataset':
                cfg.test_dataloader.dataset.type = 'mmrotate.DOTADataset'
            cfg.test_dataloader.dataset.metainfo = dict(classes=('vehicle',))
            
        if 'val_dataloader' in cfg and 'dataset' in cfg.val_dataloader:
            if cfg.val_dataloader.dataset.get('type') == 'DOTADataset':
                cfg.val_dataloader.dataset.type = 'mmrotate.DOTADataset'
            cfg.val_dataloader.dataset.metainfo = dict(classes=('vehicle',))

        # 2. Initialize Model
        print(f"Loading Model Weights: {model_path_str}")
        model = init_detector(cfg, model_path_str, device=args.device)

        summary_records, all_roc_data = [], []
        vedai_iou_data, vedai_size_data = [], []
        test_data_dir = Path(args.test_dir)

        for ds, variants in DATASET_VARIANTS.items():
            is_cowc = (ds == "COWC")
            is_vedai = (ds == "VEDAI")
            print(f"\n--- DATASET: {ds} ---")
            
            for deg in variants:
                img_dir, lbl_dir = test_data_dir / ds / deg / "images", test_data_dir / ds / deg / "labels"
                img_paths = sorted(glob.glob(str(img_dir / "*.png")) + glob.glob(str(img_dir / "*.tif")) + glob.glob(str(img_dir / "*.jpg")))
                if not img_paths: 
                    print(f"  [WARNING] No images found in {img_dir}")
                    continue

                total_gts = 0
                all_ious, all_distances, frame_evaluations = [], [], []

                for img_p in img_paths:
                    stem = Path(img_p).stem
                    gt_items = parse_label_file(lbl_dir / f"{stem}.txt", is_cowc=is_cowc, is_vedai=is_vedai)
                    total_gts += len([gt for gt in gt_items if not isinstance(gt, dict) or gt.get('class') == 0])

                    # 3. Inference via MMDetection
                    result = inference_detector(model, img_p)
                    
                    # 4. Extract boxes and filter by low ROC confidence
                    pred_instances = result.pred_instances
                    valid_mask = pred_instances.scores > args.conf
                    
                    scores = pred_instances.scores[valid_mask].cpu().numpy().tolist()
                    boxes_obb = pred_instances.bboxes[valid_mask].cpu().numpy().tolist()
                    
                    # 5. Convert [cx, cy, w, h, angle] to standard 8-point polygons
                    pred_polys = [obb_to_poly(b) for b in boxes_obb]

                    # 6. Modular Matching Engine
                    if is_cowc:
                        tps, fps, fns, match_dists = match_cowc_points(pred_polys, scores, gt_items, max_dist=args.dist_thr)
                        all_distances.extend(match_dists)
                    elif is_vedai:
                        v_preds, v_gts = filter_ignored_predictions(pred_polys, scores, gt_items, iou_threshold=args.iou_thr)
                        tps, fps, fns, match_ious = match_polygons([p['polygon'] for p in v_preds], [p['score'] for p in v_preds], v_gts, iou_thr=args.iou_thr)
                        all_ious.extend(match_ious)
                        
                        for tp in tps:
                            vedai_iou_data.append({"variant": deg, "iou": tp["iou"]})
                            vedai_size_data.append({"variant": deg, "area": get_poly_area(tp["gt_box"]), "matched": True})
                        for fn in fns:
                            vedai_size_data.append({"variant": deg, "area": get_poly_area(fn["gt_box"]), "matched": False})
                    else:
                        tps, fps, fns, match_ious = match_polygons(pred_polys, scores, gt_items, iou_thr=args.iou_thr)
                        all_ious.extend(match_ious)

                    for tp in tps: all_roc_data.append({"score": tp["score"], "is_tp": True, "variant": deg, "dataset": ds})
                    for fp in fps: all_roc_data.append({"score": fp["score"], "is_tp": False, "variant": deg, "dataset": ds})

                    frame_evaluations.append({
                        "img_path": img_p, "stem": stem,
                        "tps": tps, "fps": fps, "fns": fns
                    })

                var_preds = [p for p in all_roc_data if p["dataset"] == ds and p["variant"] == deg]
                
                if len(var_preds) > 0:
                    # Execute Threshold Sweep
                    metrics = get_peak_metrics_and_roc(var_preds, total_gts)
                    gsd_val, deg_type = parse_gsd_and_deg_type(deg)
                    
                    mean_iou = float(np.mean(all_ious)) if len(all_ious) > 0 else np.nan
                    mean_dist = float(np.mean(all_distances)) if len(all_distances) > 0 else np.nan
                    
                    print(f"  > {deg:<16} | Peak F1: {metrics['F1_Score']:.4f} (@conf {metrics['best_conf']:.2f}) | mAP50: {metrics['mAP50']:.4f}")

                    summary_records.append({
                        "Dataset": ds, "Variant": deg, "GSD_cm": gsd_val, "DegradationType": deg_type,
                        **metrics,
                        "Mean_IoU": round(mean_iou, 4) if not np.isnan(mean_iou) else "-",
                        "Mean_Point_Dist_px": round(mean_dist, 4) if not np.isnan(mean_dist) else "-"
                    })

                    # Qualitative Filtering (Filter bounding boxes at the newly discovered Peak Confidence)
                    deg_qual_dir = qual_dir / f"{ds}_{deg}"
                    deg_qual_dir.mkdir(exist_ok=True)
                    
                    for frame in frame_evaluations:
                        best_tps = [tp for tp in frame["tps"] if tp["score"] >= metrics["best_conf"]]
                        best_fps = [fp for fp in frame["fps"] if fp["score"] >= metrics["best_conf"]]
                        
                        dropped_tps_as_fns = [{"gt_box": tp["gt_box"]} if "gt_box" in tp else {"gt_pt": tp["gt_pt"]} for tp in frame["tps"] if tp["score"] < metrics["best_conf"]]
                        new_fns = frame["fns"] + dropped_tps_as_fns
                        
                        f_tp, f_fp, f_fn = len(best_tps), len(best_fps), len(new_fns)
                        f_prec = f_tp / (f_tp + f_fp) if (f_tp + f_fp) > 0 else 0.0
                        f_rec = f_tp / (f_tp + f_fn) if (f_tp + f_fn) > 0 else 0.0
                        frame["opt_f1"] = 2 * (f_prec * f_rec) / (f_prec + f_rec) if (f_prec + f_rec) > 0 else 0.0
                        frame["opt_tp_count"] = f_tp
                        frame["opt_tps"], frame["opt_fps"], frame["opt_fns"] = best_tps, best_fps, new_fns

                    # Save Clean Visual Artifacts
                    sorted_frames = sorted(frame_evaluations, key=lambda x: (x["opt_f1"], x["opt_tp_count"]), reverse=True)
                    for idx, frame in enumerate(sorted_frames[:2], 1):
                        raw_img = cv2.imread(frame["img_path"])
                        if raw_img is not None:
                            vis = draw_annotations(raw_img, frame["opt_tps"], frame["opt_fps"], frame["opt_fns"], is_cowc)
                            cv2.imwrite(str(deg_qual_dir / f"best_{idx}_{frame['stem']}.png"), vis)
                    for idx, frame in enumerate(sorted_frames[-2:], 1):
                        raw_img = cv2.imread(frame["img_path"])
                        if raw_img is not None:
                            vis = draw_annotations(raw_img, frame["opt_tps"], frame["opt_fps"], frame["opt_fns"], is_cowc)
                            cv2.imwrite(str(deg_qual_dir / f"worst_{idx}_{frame['stem']}.png"), vis)

        df_results = pd.DataFrame(summary_records)
        df_results.to_csv(output_dir / "rq1_summary_metrics.csv", index=False)
        
        # Pass to the modular drawing functions
        generate_powerpoint_plots(df_results, all_roc_data, output_dir, model_name)
        generate_extended_plots(df_results, vedai_iou_data, vedai_size_data, output_dir, model_name)
        
        print(f"\n=== SUCCESS: {model_name} ===")

if __name__ == "__main__":
    main()