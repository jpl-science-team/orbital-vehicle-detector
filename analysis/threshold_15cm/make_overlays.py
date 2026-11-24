# Scratch: side-by-side 640 vs 512 overlays on a few 15cm test images (COWC + VEDAI color).
import sys, glob, random
from pathlib import Path
import argparse, cv2, numpy as np
sys.path.insert(0, "src")
from ultralytics import YOLO
from eval.calibration import compute_eval_imgsz
from eval.parsers import parse_label_file
from eval.metrics import match_cowc_points, match_polygons, filter_ignored_predictions

THR, LOW = 0.46, 0.10          # val-calibrated threshold; lower bound for "correct but below threshold"
OUT = Path("analysis/threshold_15cm/overlays"); OUT.mkdir(exist_ok=True)
GREEN, RED, BLUE, YELLOW, GRAY = (0, 220, 0), (0, 0, 255), (255, 80, 0), (0, 220, 255), (160, 160, 160)

def pick(paths, lbl, n, min_gt, **kw):
    rng = random.Random(0); paths = paths[:]; rng.shuffle(paths); out = []
    for p in paths:
        gts = parse_label_file(f"{lbl}/{Path(p).stem}.txt", **kw)
        if len([g for g in gts if not isinstance(g, dict) or g["class"] == 0]) >= min_gt: out.append(p)
        if len(out) == n: break
    return out

def poly(c, pts, col, t=2): cv2.polylines(c, [np.array(pts, np.int32).reshape(-1, 1, 2)], True, col, t)

def panel(model, s, img_p, gts, is_cowc):
    img = cv2.imread(img_p); h, w = img.shape[:2]
    a = argparse.Namespace(imgsz=None, train_imgsz=s, train_tile_size=512)
    r = model.predict(img_p, conf=0.01, imgsz=compute_eval_imgsz(h, w, a), verbose=False)[0]
    pp = [b.reshape(8).tolist() for b in r.obb.xyxyxyxy.cpu().numpy()]; sc = r.obb.conf.cpu().numpy().tolist()
    c = img.copy()
    if is_cowc:
        # Greedy matching in score order, so matches above THR are the same as matching only the >=THR boxes
        tps, fps, fns, _ = match_cowc_points(pp, sc, gts, 25.0); key = "gt_pt"
    else:
        for g in gts:
            if g["class"] != 0: poly(c, g["polygon"], GRAY, 1)   # non-vehicle classes = ignore regions
        vp, vg = filter_ignored_predictions(pp, sc, gts, 0.5)
        tps, fps, fns, _ = match_polygons([x["polygon"] for x in vp], [x["score"] for x in vp], vg, 0.5); key = "gt_box"
    hi_tp = [t for t in tps if t["score"] >= THR]; lo_tp = [t for t in tps if LOW <= t["score"] < THR]
    hi_fp = [f for f in fps if f["score"] >= THR]
    missed = fns + [t for t in tps if t["score"] < THR]
    for m in missed:
        if is_cowc: cv2.circle(c, (int(m[key][0]), int(m[key][1])), 5, BLUE, -1)
        else: poly(c, m[key], BLUE, 2)
    for t in lo_tp: poly(c, t["pred_box"], YELLOW, 1)
    for f in hi_fp: poly(c, f["pred_box"], RED, 2)
    for t in hi_tp: poly(c, t["pred_box"], GREEN, 2)
    n_gt = len(hi_tp) + len(missed)
    hdr = np.full((52, w, 3), 255, np.uint8)
    cv2.putText(hdr, f"YOLO11n {s}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(hdr, f"TP {len(hi_tp)}  FP {len(hi_fp)}  miss {len(missed)}/{n_gt}  (+{len(lo_tp)} correct at {LOW}-{THR})",
                (8, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return np.vstack([hdr, c])

def legend(w, is_cowc):
    l = np.full((30, w, 3), 255, np.uint8); x = 8
    items = [(GREEN, f"correct (conf>={THR})"), (RED, f"false positive (conf>={THR})"),
             (BLUE, "missed car" + (" (center)" if is_cowc else "")), (YELLOW, f"correct but conf {LOW}-{THR}")]
    if not is_cowc: items.append((GRAY, "ignored class"))
    for col, txt in items:
        cv2.rectangle(l, (x, 9), (x + 14, 23), col, -1); cv2.putText(l, txt, (x + 20, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        x += 30 + int(len(txt) * 8.2)
    return l

models = {s: YOLO(f"models/weights/test_weights/YOLO11n_15GSD_{s}.pt") for s in (640, 512)}
cl = "test_data/COWC/15cm_base/labels"; vl = "test_data/VEDAI/15cm_base/labels"
cowc = sorted(glob.glob("test_data/COWC/15cm_base/images/*.png"))
sel = [(p, cl, True) for city in ("Columbus", "Potsdam")
       for p in pick([q for q in cowc if Path(q).name.startswith(city)], cl, 3, 8, is_cowc=True)]
sel += [(p, vl, False) for p in pick(sorted(glob.glob("test_data/VEDAI/15cm_base/images/*_co.png")), vl, 4, 3, is_vedai=True)]
for i, (p, lbl, is_cowc) in enumerate(sel, 1):
    gts = parse_label_file(f"{lbl}/{Path(p).stem}.txt", is_cowc=is_cowc, is_vedai=not is_cowc)
    row = np.hstack([panel(models[640], 640, p, gts, is_cowc), np.full((0, 8, 3), 255, np.uint8)[:0] if False else np.full((cv2.imread(p).shape[0] + 52, 8, 3), 255, np.uint8),
                     panel(models[512], 512, p, gts, is_cowc)])
    name = ("COWC_" if is_cowc else "VEDAI_") + f"{i:02d}_{Path(p).stem[:60]}.png"
    cv2.imwrite(str(OUT / name), np.vstack([row, legend(row.shape[1], is_cowc)])); print(name)
