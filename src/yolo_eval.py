# src/run_eval.py
import glob
import cv2
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

from eval.config import DATASET_VARIANTS
from eval.parsers import parse_args, parse_label_file, parse_gsd_and_deg_type
from eval.metrics import filter_ignored_predictions, match_cowc_points, match_polygons, get_peak_metrics_and_roc, get_poly_area
from eval.visualization import draw_annotations, generate_powerpoint_plots, generate_extended_plots

warnings.filterwarnings("ignore", category=UserWarning)

def main():
    args = parse_args()
    
    for model_path_str in args.models:
        model_path = Path(model_path_str)
        if not model_path.exists():
            print(f"\n[WARNING] Skipping model '{model_path_str}' - File not found.")
            continue
            
        model_name = model_path.stem
        output_dir = Path(f"Eval_Results_YOLO_{model_name}")
        output_dir.mkdir(exist_ok=True)
        qual_dir = output_dir / "qualitative"
        qual_dir.mkdir(exist_ok=True)

        print(f"\n=======================================================")
        print(f" STARTING EVALUATION: {model_name}")
        print(f"=======================================================")
        
        model = YOLO(model_path_str)
        summary_records, all_roc_data = [], []
        vedai_iou_data, vedai_size_data = [], []

        for ds, variants in DATASET_VARIANTS.items():
            is_cowc = (ds == "COWC")
            is_vedai = (ds == "VEDAI")
            print(f"\n--- DATASET: {ds} ---")
            
            for deg in variants:
                img_paths = sorted(glob.glob(str(Path(args.test_dir) / ds / deg / "images" / "*.*")))
                lbl_dir = Path(args.test_dir) / ds / deg / "labels"
                total_gts = 0
                
                all_ious, all_distances, frame_evaluations = [], [], []
                
                for img_p in img_paths:
                    if not img_p.lower().endswith(('.png', '.jpg', '.tif')): continue
                    
                    gt_items = parse_label_file(lbl_dir / f"{Path(img_p).stem}.txt", is_cowc=is_cowc, is_vedai=is_vedai)
                    total_gts += len([gt for gt in gt_items if not isinstance(gt, dict) or gt.get('class') == 0])

                    raw_img = cv2.imread(img_p)
                    if raw_img is None: continue
                    
                    h_img, w_img = raw_img.shape[:2]
                    eval_imgsz = args.imgsz if args.imgsz else int(np.ceil(max(h_img, w_img) / 32.0) * 32)
                    
                    results = model.predict(img_p, conf=args.conf, imgsz=eval_imgsz, verbose=False)[0]
                    
                    pred_polys, pred_scores = [], []
                    if hasattr(results, 'obb') and results.obb is not None:
                        pred_polys = [b.reshape(8).tolist() for b in results.obb.xyxyxyxy.cpu().numpy()]
                        pred_scores = results.obb.conf.cpu().numpy().tolist()

                    if is_cowc:
                        tps, fps, fns, match_dists = match_cowc_points(pred_polys, pred_scores, gt_items, args.dist_thr)
                        all_distances.extend(match_dists)
                    elif is_vedai:
                        v_preds, v_gts = filter_ignored_predictions(pred_polys, pred_scores, gt_items, args.iou_thr)
                        tps, fps, fns, match_ious = match_polygons([p['polygon'] for p in v_preds], [p['score'] for p in v_preds], v_gts, args.iou_thr)
                        all_ious.extend(match_ious)
                        
                        for tp in tps:
                            vedai_iou_data.append({"variant": deg, "iou": tp["iou"]})
                            vedai_size_data.append({"variant": deg, "area": get_poly_area(tp["gt_box"]), "matched": True})
                        for fn in fns:
                            vedai_size_data.append({"variant": deg, "area": get_poly_area(fn["gt_box"]), "matched": False})
                    else:
                        tps, fps, fns, match_ious = match_polygons(pred_polys, pred_scores, gt_items, args.iou_thr)
                        all_ious.extend(match_ious)

                    for tp in tps: all_roc_data.append({"score": tp["score"], "is_tp": True, "variant": deg, "dataset": ds})
                    for fp in fps: all_roc_data.append({"score": fp["score"], "is_tp": False, "variant": deg, "dataset": ds})

                    frame_evaluations.append({
                        "img_path": img_p, "stem": Path(img_p).stem,
                        "tps": tps, "fps": fps, "fns": fns
                    })

                var_preds = [p for p in all_roc_data if p["dataset"] == ds and p["variant"] == deg]
                
                if len(var_preds) > 0:
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

                    # Qualitative Filtering & Output
                    deg_qual_dir = qual_dir / f"{ds}_{deg}"
                    deg_qual_dir.mkdir(exist_ok=True)
                    
                    # Filter frame annotations using the discovered best_conf
                    for frame in frame_evaluations:
                        best_tps = [tp for tp in frame["tps"] if tp["score"] >= metrics["best_conf"]]
                        best_fps = [fp for fp in frame["fps"] if fp["score"] >= metrics["best_conf"]]
                        
                        # TPs that fell below threshold are now FNs
                        dropped_tps_as_fns = [{"gt_box": tp["gt_box"]} if "gt_box" in tp else {"gt_pt": tp["gt_pt"]} for tp in frame["tps"] if tp["score"] < metrics["best_conf"]]
                        new_fns = frame["fns"] + dropped_tps_as_fns
                        
                        f_tp, f_fp, f_fn = len(best_tps), len(best_fps), len(new_fns)
                        f_prec = f_tp / (f_tp + f_fp) if (f_tp + f_fp) > 0 else 0.0
                        f_rec = f_tp / (f_tp + f_fn) if (f_tp + f_fn) > 0 else 0.0
                        frame["opt_f1"] = 2 * (f_prec * f_rec) / (f_prec + f_rec) if (f_prec + f_rec) > 0 else 0.0
                        frame["opt_tp_count"] = f_tp
                        frame["opt_tps"], frame["opt_fps"], frame["opt_fns"] = best_tps, best_fps, new_fns

                    # Sort and save best/worst images
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
        generate_powerpoint_plots(df_results, all_roc_data, output_dir, model_name)
        generate_extended_plots(df_results, vedai_iou_data, vedai_size_data, output_dir, model_name)
        print(f"\n=== SUCCESS: {model_name} ===")

if __name__ == "__main__":
    main()