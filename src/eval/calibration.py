# src/eval/calibration.py
"""Picks per-resolution confidence thresholds on the validation split, so test-set F1 isn't tuned on the test data.

The validation split is 15cm GSD only. For each test variant, the validation images are degraded in memory the
same way preprocess_rq1_datasets.py degrades the test data (INTER_AREA down to the content GSD, then INTER_CUBIC
back up to the pixel GSD), and the best-F1 threshold is found on those predictions."""
import glob
from pathlib import Path

import cv2
import numpy as np
import yaml

from eval.metrics import match_polygons, find_best_f1_threshold

NATIVE_VAL_GSD = 15.0


def compute_eval_imgsz(h_img, w_img, args):
    """Inference size: --imgsz if given, else the image enlarged by the same factor the training tiles were."""
    if args.imgsz: return args.imgsz
    train_scale = args.train_imgsz / args.train_tile_size
    return int(np.ceil(max(h_img, w_img) * train_scale / 32.0) * 32)


def load_val_split(dataset_yaml):
    """Returns [(image_path, [8-coord pixel polygons])] for the 'val' split of an Ultralytics OBB dataset.yaml."""
    with open(dataset_yaml) as f:
        cfg = yaml.safe_load(f)
    img_dir = Path(cfg["path"]) / cfg["val"]
    lbl_dir = Path(cfg["path"]) / cfg["val"].replace("images", "labels", 1)  # Ultralytics images/ -> labels/ convention

    samples = []
    for img_p in sorted(glob.glob(str(img_dir / "*.*"))):
        if not img_p.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")): continue
        img = cv2.imread(img_p)
        if img is None: continue
        h, w = img.shape[:2]
        polys = []
        lbl_p = lbl_dir / f"{Path(img_p).stem}.txt"
        if lbl_p.exists():
            with open(lbl_p) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 9: continue
                    xy = [float(v) for v in parts[1:9]]  # YOLO OBB: class x1 y1 ... x4 y4 (normalized)
                    polys.append([xy[i] * (w if i % 2 == 0 else h) for i in range(8)])
        samples.append((img, polys))
    return samples


def degrade(img, polys, content_gsd, pixel_gsd):
    """Mimics preprocess_rq1_datasets.py: downsample to content_gsd, then resample to pixel_gsd."""
    h, w = img.shape[:2]
    out = img
    if content_gsd != NATIVE_VAL_GSD:
        s = NATIVE_VAL_GSD / content_gsd
        out = cv2.resize(out, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    if pixel_gsd != content_gsd:
        s = NATIVE_VAL_GSD / pixel_gsd
        out = cv2.resize(out, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_CUBIC)
    s = NATIVE_VAL_GSD / pixel_gsd
    return out, [[c * s for c in poly] for poly in polys]


class ValThresholdCalibrator:
    """Caches one threshold per (content GSD, pixel GSD) pair for a given model."""

    def __init__(self, model, args):
        self.model, self.args = model, args
        self.samples = load_val_split(args.val_data)
        self.cache = {}
        n_gts = sum(len(p) for _, p in self.samples)
        print(f"Threshold calibration: {len(self.samples)} validation images, {n_gts} vehicles ({args.val_data})")

    def threshold(self, content_gsd, pixel_gsd):
        key = (content_gsd, pixel_gsd)
        if key not in self.cache:
            preds, total_gts = [], 0
            for img, polys in self.samples:
                v_img, v_polys = degrade(img, polys, content_gsd, pixel_gsd)
                total_gts += len(v_polys)
                h, w = v_img.shape[:2]
                res = self.model.predict(v_img, conf=self.args.conf, imgsz=compute_eval_imgsz(h, w, self.args), verbose=False)[0]
                pred_polys, pred_scores = [], []
                if res.obb is not None:
                    pred_polys = [b.reshape(8).tolist() for b in res.obb.xyxyxyxy.cpu().numpy()]
                    pred_scores = res.obb.conf.cpu().numpy().tolist()
                tps, fps, _, _ = match_polygons(pred_polys, pred_scores, v_polys, self.args.iou_thr)
                preds += [{"score": t["score"], "is_tp": True} for t in tps]
                preds += [{"score": f["score"], "is_tp": False} for f in fps]
            conf, f1 = find_best_f1_threshold(preds, total_gts)
            print(f"  [val calib] content {content_gsd}cm / pixels {pixel_gsd}cm -> conf {conf:.3f} (val F1 {f1:.4f})")
            self.cache[key] = conf
        return self.cache[key]
