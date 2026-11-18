# 15 cm threshold analysis (2026-11-17)

YOLO11n-OBB trained at 640 vs 512, tested at 15 cm GSD on COWC (`test_data/COWC/15cm_base`) and VEDAI color images (`test_data/VEDAI/15cm_base/*_co.png`).

- `threshold_sweep.py` -> `f1_vs_threshold.json`, `sweep_summary.txt`: F1 at every confidence threshold 0.01-0.90 on COWC val, the cowc_512 test split (OBB boxes, IoU 0.5) and the RQ1 COWC test (center points, 25 px), split by city, plus VEDAI.
- `make_overlays.py` -> `overlays/`: 640 (left) vs 512 (right) on 6 COWC and 4 VEDAI test images. Green = correct at conf >= 0.46, red = false positive, blue = missed car, yellow = correct but conf 0.10-0.46, gray = VEDAI ignored class.

Run from the repo root: `~/miniforge3/envs/yolo_env/bin/python analysis/threshold_15cm/<script>.py`

## Findings

- The validation-picked threshold (0.46) works for VEDAI (best on test: 0.45/0.53) and Potsdam with box scoring (0.58/0.62), but not for COWC with center-point scoring (best 0.04-0.10).
- Columbus is the problem: box-scored F1 is only ~0.61 at any threshold (Potsdam ~0.90), and with center points its best threshold is 0.01-0.05. The model is unsure about Columbus cars overall, not just mis-thresholded.
- The cowc_512 val split is all Vaihingen, and all 268 tiles are grayscale. Training is ~99.6% color (only 10 Selwyn tiles are gray). Test Columbus and Potsdam are stored as 3-channel color (but Columbus looks gray in the overlays; to check).
- 512 vs 640 at 15 cm: tied on VEDAI and Potsdam; 512 clearly better on Columbus (point F1 0.674 vs 0.516 at best threshold).

## Next steps

1. Check whether the Columbus test images are really grayscale stored as RGB (the `gray()` check above found per-pixel channel differences, possibly from JPEG/compression noise).
2. Score COWC validation with center points too, so the COWC threshold is picked under the same rule as the test set.
3. Rebuild the validation split from several cities (color and gray) and retrain both models.
4. Report mAP50 as the headline metric, with val-picked F1 and oracle F1 (labeled as an upper bound).
