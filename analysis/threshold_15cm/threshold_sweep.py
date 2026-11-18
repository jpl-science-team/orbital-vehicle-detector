# Scratch analysis: F1 vs confidence threshold on COWC val, COWC test (per city) and VEDAI 15cm color, for both models.
import sys, glob, json, argparse
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, "src")
from ultralytics import YOLO
from eval.calibration import load_val_split, compute_eval_imgsz
from eval.parsers import parse_label_file
from eval.metrics import match_polygons, match_cowc_points, filter_ignored_predictions, counts_at_threshold

THS = np.round(np.arange(0.01, 0.91, 0.01), 2)

def predict(model, img, a):
    h, w = img.shape[:2]
    r = model.predict(img, conf=0.01, imgsz=compute_eval_imgsz(h, w, a), verbose=False)[0]
    if r.obb is None: return [], []
    return [b.reshape(8).tolist() for b in r.obb.xyxyxyxy.cpu().numpy()], r.obb.conf.cpu().numpy().tolist()

def collect(preds, tps, fps):
    preds += [{"score": t["score"], "is_tp": True} for t in tps] + [{"score": f["score"], "is_tp": False} for f in fps]

def curve(preds, n_gt):
    return [counts_at_threshold(preds, n_gt, t)[5] for t in THS]

out = {}
for s in [640, 512]:
    a = argparse.Namespace(imgsz=None, train_imgsz=s, train_tile_size=512)
    m = YOLO(f"models/weights/test_weights/YOLO11n_15GSD_{s}.pt")
    sets = {}
    # COWC val (Vaihingen), OBB polygons, IoU 0.5 -- what the calibrator uses
    p, n = [], 0
    for img, polys in load_val_split("data/cowc_512/dataset.yaml"):
        pp, sc = predict(m, img, a); t, f, _, _ = match_polygons(pp, sc, polys, 0.5); collect(p, t, f); n += len(polys)
    sets["COWC val (Vaihingen), IoU"] = (p, n)
    # cowc_512 test split, same OBB labels + IoU matching as val, so only the domain differs
    city = {}
    for ip in sorted(glob.glob("data/cowc_512/images/test/*.png")):
        img = cv2.imread(ip); h, w = img.shape[:2]; polys = []
        for line in open(f"data/cowc_512/labels/test/{Path(ip).stem}.txt"):
            xy = [float(v) for v in line.split()[1:9]]
            if len(xy) == 8: polys.append([xy[i] * (w if i % 2 == 0 else h) for i in range(8)])
        pp, sc = predict(m, img, a); t, f, _, _ = match_polygons(pp, sc, polys, 0.5)
        cp, cn = city.setdefault(Path(ip).name.split("_")[0], ([], [0])); collect(cp, t, f); cn[0] += len(polys)
    for c, (cp, cn) in city.items(): sets[f"cowc_512 test {c}, IoU"] = (cp, cn[0])
    # COWC test, center points, 25px, split by city
    city = {}
    for ip in sorted(glob.glob("test_data/COWC/15cm_base/images/*.png")):
        c = Path(ip).name.split("_")[0]
        gts = parse_label_file(f"test_data/COWC/15cm_base/labels/{Path(ip).stem}.txt", is_cowc=True)
        pp, sc = predict(m, cv2.imread(ip), a); t, f, _, _ = match_cowc_points(pp, sc, gts, 25.0)
        cp, cn = city.setdefault(c, ([], [0])); collect(cp, t, f); cn[0] += len(gts)
    allp, alln = [], 0
    for c, (cp, cn) in city.items():
        sets[f"COWC test {c}, points"] = (cp, cn[0]); allp += cp; alln += cn[0]
    sets["COWC test all, points"] = (allp, alln)
    # VEDAI 15cm color, IoU 0.5 with ignore regions
    p, n = [], 0
    for ip in sorted(glob.glob("test_data/VEDAI/15cm_base/images/*_co.png")):
        gts = parse_label_file(f"test_data/VEDAI/15cm_base/labels/{Path(ip).stem}.txt", is_vedai=True)
        pp, sc = predict(m, cv2.imread(ip), a)
        vp, vg = filter_ignored_predictions(pp, sc, gts, 0.5)
        t, f, _, _ = match_polygons([x["polygon"] for x in vp], [x["score"] for x in vp], vg, 0.5); collect(p, t, f)
        n += len([g for g in gts if g["class"] == 0])
    sets["VEDAI 15cm color, IoU"] = (p, n)
    out[s] = {k: {"n_gt": n, "f1": curve(p, n)} for k, (p, n) in sets.items()}
    print(s, {k: v["n_gt"] for k, v in out[s].items()}, flush=True)
json.dump({"ths": THS.tolist(), "res": out}, open("analysis/threshold_15cm/f1_vs_threshold.json", "w"))
