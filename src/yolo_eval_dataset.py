# src/yolo_eval_dataset.py
"""Evaluates YOLO OBB models on the test split of any dataset YAML (README Step 7.5), at native resolution.

Unlike yolo_eval.py (the RQ1 resolution study on COWC/VEDAI/Angel), this reads the same dataset file used for
training, so the test labels are in the training format (class x1 y1 .. x4 y4, normalized). Each class is scored
separately: a prediction is a true positive if it has the same class as a labeled box and overlaps it with
IoU >= --iou-thr. Each class's confidence threshold is the one with the best F1 on the val split, so the test
F1 isn't tuned on the test data."""
import argparse
import glob
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset, img2label_paths

from eval.metrics import match_polygons, find_best_f1_threshold, get_peak_metrics_and_roc
from eval.visualization import draw_annotations

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def parse_args():
    parser = argparse.ArgumentParser(description="Per-class evaluation of YOLO OBB models on a dataset YAML's test split")
    parser.add_argument("--models", nargs="+", required=True, help="One or more YOLO .pt weights, all trained at --imgsz")
    parser.add_argument("--data", type=str, required=True,
                        help="Dataset YAML the models were trained with (train.py --data); needs 'val' and 'test' splits")
    parser.add_argument("--imgsz", type=int, required=True, help="--img_size the models were trained at (train.py)")
    parser.add_argument("--iou-thr", type=float, default=0.50, help="Minimum IoU for a prediction to match a labeled box")
    parser.add_argument("--conf", type=float, default=0.01,
                        help="Lowest confidence kept from the model; thresholds and mAP are computed above this")
    parser.add_argument("--threshold-source", choices=["val", "test"], default="val",
                        help="Pick each class's F1 threshold on the val split (val), or on the test split itself (test; optimistic)")
    parser.add_argument("--out-dir", type=str, default=".",
                        help="Folder to write results to; each model gets its own Eval_Results_YOLO_<model>_<data>[_<tag>]/ inside it")
    parser.add_argument("--tag", type=str, default="", help="Suffix for the output folder")
    parser.add_argument("--device", type=str, default=None, help="Device for inference, e.g. 0 or cpu (default: Ultralytics' choice)")
    return parser.parse_args()


def list_split_images(data, split, yaml_path):
    """Image paths for a split of a resolved dataset dict: a folder, a .txt list of images, or a list of folders."""
    entries = data.get(split)
    if not entries:
        raise SystemExit(f"{yaml_path} has no '{split}:' split; this evaluation needs 'val' and 'test' (see configs/template.yaml)")
    paths = []
    for entry in entries if isinstance(entries, list) else [entries]:
        entry = Path(entry)
        if entry.is_dir():
            paths += [p for p in glob.glob(str(entry / "**" / "*"), recursive=True) if p.lower().endswith(IMAGE_EXTENSIONS)]
        elif entry.suffix == ".txt" and entry.is_file():
            root = Path(data["path"])
            for line in entry.read_text().splitlines():
                if line.strip():
                    p = Path(line.strip())
                    paths.append(str(p if p.is_absolute() else (root / p).resolve()))
        else:
            raise SystemExit(f"'{split}' entry {entry} (from {yaml_path}) is not a folder or a .txt image list")
    if not paths:
        raise SystemExit(f"No images found for the '{split}' split of {yaml_path}")
    return sorted(paths)


def load_labels(img_path, w, h):
    """Labeled boxes of one image as (class IDs, pixel polygons); the label file is found as in Ultralytics training."""
    lbl_p = Path(img2label_paths([img_path])[0])
    classes, polys = [], []
    if lbl_p.exists():
        for line in lbl_p.read_text().splitlines():
            parts = line.split()
            if len(parts) < 9: continue
            xy = [float(v) for v in parts[1:9]]  # YOLO OBB: class x1 y1 ... x4 y4 (normalized)
            classes.append(int(float(parts[0])))
            polys.append([xy[i] * (w if i % 2 == 0 else h) for i in range(8)])
    return classes, polys


def predict_split(model, img_paths, args):
    """Runs the model on every image once; returns one record per image with its labels and predictions."""
    records = []
    for img_p in img_paths:
        img = cv2.imread(img_p)
        if img is None:
            print(f"  [WARNING] Could not read {img_p}; skipping")
            continue
        h, w = img.shape[:2]
        gt_cls, gt_polys = load_labels(img_p, w, h)
        res = model.predict(img, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)[0]
        pred_polys, pred_scores, pred_cls = [], [], []
        if res.obb is not None and len(res.obb):
            pred_polys = [b.reshape(8).tolist() for b in res.obb.xyxyxyxy.cpu().numpy()]
            pred_scores = res.obb.conf.cpu().numpy().tolist()
            pred_cls = res.obb.cls.int().cpu().numpy().tolist()
        records.append({"img_path": img_p, "gt_cls": gt_cls, "gt_polys": gt_polys,
                        "pred_cls": pred_cls, "pred_polys": pred_polys, "pred_scores": pred_scores})
    return records


def match_by_class(records, class_ids, iou_thr):
    """Matches predictions to labels of the same class in every image.

    Returns {class_id: {"preds": [{"score", "is_tp"}], "total_gts": n, "ious": [...]}} and, per image,
    {class_id: (tps, fps, fns)} for the qualitative images."""
    per_class = {c: {"preds": [], "total_gts": 0, "ious": []} for c in class_ids}
    per_image = []
    for r in records:
        img_matches = {}
        for c in class_ids:
            gts = [p for p, k in zip(r["gt_polys"], r["gt_cls"]) if k == c]
            idx = [i for i, k in enumerate(r["pred_cls"]) if k == c]
            tps, fps, fns, ious = match_polygons([r["pred_polys"][i] for i in idx], [r["pred_scores"][i] for i in idx], gts, iou_thr)
            for item in tps + fps + fns: item["cls"] = c  # for the class names on the qualitative images
            per_class[c]["preds"] += [{"score": t["score"], "is_tp": True} for t in tps] + [{"score": f["score"], "is_tp": False} for f in fps]
            per_class[c]["total_gts"] += len(gts)
            per_class[c]["ious"] += ious
            img_matches[c] = (tps, fps, fns)
        per_image.append(img_matches)
    return per_class, per_image


def summarize(test_per_class, thresholds, names):
    """One row per class at its threshold, plus an 'all' row that adds up the counts over all classes."""
    rows = []
    for c, d in test_per_class.items():
        m = get_peak_metrics_and_roc(d["preds"], d["total_gts"], thresholds[c])
        rows.append({"Class_ID": c, "Class": names[c], "Labels": d["total_gts"], "conf": round(m["best_conf"], 3),
                     "TP": m["TP"], "FP": m["FP"], "FN": m["FN"], "Precision": m["Precision"], "Recall": m["Recall"],
                     "F1_Score": m["F1_Score"], "Oracle_F1": m["Oracle_F1"], "mAP50": m["mAP50"],
                     "Mean_IoU": round(float(np.mean(d["ious"])), 4) if d["ious"] else np.nan})
    df = pd.DataFrame(rows)
    tp, fp, fn = df["TP"].sum(), df["FP"].sum(), df["FN"].sum()
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    labeled = df[df["Labels"] > 0]
    total = {"Class_ID": "-", "Class": "all", "Labels": int(df["Labels"].sum()), "conf": "-", "TP": tp, "FP": fp, "FN": fn,
             "Precision": p, "Recall": r, "F1_Score": 2 * p * r / (p + r) if p + r else 0.0, "Oracle_F1": np.nan,
             # mean over the classes that have test labels, like Ultralytics' mAP50
             "mAP50": float(labeled["mAP50"].mean()) if len(labeled) else 0.0, "Mean_IoU": np.nan}
    return pd.concat([df, pd.DataFrame([total])], ignore_index=True)


def save_qualitative(records, per_image, thresholds, qual_dir, names, n=2):
    """Saves the n best and n worst test images by F1 at the chosen thresholds, with class names on each box.
    Green: correct, red: false detection (or wrong class), blue: missed labeled box."""
    frames = []
    for r, matches in zip(records, per_image):
        tps, fps, fns = [], [], []
        for c, (c_tps, c_fps, c_fns) in matches.items():
            tps += [t for t in c_tps if t["score"] >= thresholds[c]]
            fps += [f for f in c_fps if f["score"] >= thresholds[c]]
            fns += c_fns + [{"gt_box": t["gt_box"], "cls": c} for t in c_tps if t["score"] < thresholds[c]]
        f1 = 2 * len(tps) / (2 * len(tps) + len(fps) + len(fns)) if tps or fps or fns else 1.0  # an empty image with no predictions is perfect
        frames.append({"img_path": r["img_path"], "tps": tps, "fps": fps, "fns": fns, "f1": f1})
    frames.sort(key=lambda f: (f["f1"], len(f["tps"])), reverse=True)
    for name, chosen in (("best", frames[:n]), ("worst", frames[::-1][:n])):
        for idx, f in enumerate(chosen, 1):
            img = cv2.imread(f["img_path"])
            if img is not None:
                cv2.imwrite(str(qual_dir / f"{name}_{idx}_{Path(f['img_path']).stem}.png"), draw_annotations(img, f["tps"], f["fps"], f["fns"], names=names))


def main():
    args = parse_args()
    data = check_det_dataset(args.data, autodownload=False)
    names = data["names"]
    class_ids = sorted(names)
    data_name = Path(args.data).stem
    test_imgs = list_split_images(data, "test", args.data)
    val_imgs = list_split_images(data, "val", args.data) if args.threshold_source == "val" else []
    print(f"Dataset {args.data}: {len(class_ids)} classes {list(names.values())}, "
          f"{len(test_imgs)} test images" + (f", {len(val_imgs)} val images" if val_imgs else ""))

    for model_path in args.models:
        if not os.path.exists(model_path):
            print(f"\n[WARNING] Skipping model '{model_path}' - File not found.")
            continue
        model_name = Path(model_path).stem
        output_dir = Path(args.out_dir) / (f"Eval_Results_YOLO_{model_name}_{data_name}" + (f"_{args.tag}" if args.tag else ""))
        qual_dir = output_dir / "qualitative"
        qual_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== EVALUATING {model_name} on {data_name} ===")

        model = YOLO(model_path)
        model_names = model.names
        if {int(k): v for k, v in model_names.items()} != {int(k): v for k, v in names.items()}:
            print(f"  [WARNING] The model's classes {model_names} differ from {args.data}'s {names}; "
                  f"predictions are matched by class ID")

        test_records = predict_split(model, test_imgs, args)
        test_per_class, test_per_image = match_by_class(test_records, class_ids, args.iou_thr)

        if args.threshold_source == "val":
            val_per_class, _ = match_by_class(predict_split(model, val_imgs, args), class_ids, args.iou_thr)
            thresholds = {c: find_best_f1_threshold(val_per_class[c]["preds"], val_per_class[c]["total_gts"])[0] for c in class_ids}
        else:
            thresholds = {c: find_best_f1_threshold(test_per_class[c]["preds"], test_per_class[c]["total_gts"])[0] for c in class_ids}

        df = summarize(test_per_class, thresholds, names)
        df.to_csv(output_dir / "summary_metrics.csv", index=False)
        with open(output_dir / "SUMMARY.md", "w") as f:
            f.write(f"# Evaluation Summary\n**Model:** `{model_name}`  \n**Data:** `{args.data}` (test split, {len(test_records)} images)  \n"
                    f"**Thresholds:** best F1 per class on the {args.threshold_source} split, IoU >= {args.iou_thr}\n\n"
                    f"{df.to_markdown(index=False, floatfmt='.4f')}\n")
        save_qualitative(test_records, test_per_image, thresholds, qual_dir, names)

        print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
