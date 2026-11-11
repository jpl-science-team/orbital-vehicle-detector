# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Research workflow for detecting vehicles in satellite/aerial imagery with oriented bounding boxes (OBB), run on the JPL MLIA cluster. There are two model pipelines, and each has its own conda env because their dependencies conflict:

- **YOLO (Ultralytics)**: YOLO11n-OBB, `yolo_env` (Python 3.10, `requirements-yolo.txt`). Working.
- **O2-RT-DETR (OpenMMLab mmdet/mmrotate)**: `detr_env` (Python 3.9, mmcv 2.0–2.2, mmdet 3.0–3.3, `requirements-detr.txt`). Still in progress.

The main research question ("RQ1") is how detection accuracy degrades as ground sample distance (GSD) changes, measured across the COWC, VEDAI and Angel datasets.

There's no test suite, linter or build step. All scripts use paths relative to the repo root, so **run everything from the repo root**.

## Commands

```bash
# YOLO training (yolo_env)
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 python src/train.py --img_size 640 --batch_size 16 --epochs 150 --data data/cowc_512/dataset.yaml --gpus 2,3

# O2-RT-DETR training on 4 GPUs (detr_env)
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 bash O2-RT-DETR/tools/dist_train.sh O2-RT-DETR/projects/rotated_rtdetr/configs/cowc_rtdetr_512.py 4

# Train vs val loss / mAP curves for a finished run -> <run_dir>/results_combined.png
python src/plot_training.py runs/obb/runs/detect/<run_name>

# Build the degraded-GSD test sandbox: data/ (read-only) -> test_data/
python src/preprocess_rq1_datasets.py

# RQ1 evaluation (default weights: models/weights/test_weights/YOLO11n_15GSD.pt)
python src/yolo_eval.py --models <a.pt> [<b.pt> ...] --train-imgsz 640   # must match train.py --img_size
python src/detr_eval.py --config <cfg.py> --models <ckpt.pth>

# YOLO vs RT-DETR comparison plots -> Comparison_Plots/
python src/compare_models.py --yolo-csv Eval_Results_YOLO_<name>/rq1_summary_metrics.csv --detr-csv Eval_Results_RTDETR_<name>/rq1_summary_metrics.csv
```

Run training inside `tmux`, because jobs can take hours or days.

## Architecture

**Data layout (gitignored):** `data/` holds the raw datasets and must be treated as read-only. `data/cowc_512/dataset.yaml` has to contain an absolute path that matches the local machine. The DETR configs read DOTA-format data from `data/cowc_512_degraded_dota/{train,val}/annfiles/`. `test_data/<Dataset>/<variant>/{images,labels}` is the evaluation sandbox. `models/`, `runs/` and `*.pt` are also gitignored. YOLO training runs land in `runs/obb/runs/detect/<timestamp>_yolo11n_res<imgsz>/` (Ultralytics prefixes `runs/obb/` to `project="runs/detect"` for OBB models).

**GSD variants:** `src/eval/config.py::DATASET_VARIANTS` is the single source of truth for which variants exist (COWC/VEDAI: `15cm_base`, `30cm_{naive,scalenorm}`, `60cm_{naive,scalenorm}`; Angel: `50cm_base`, `30cm_upscale`, `15cm_upscale`). `preprocess_rq1_datasets.py` generates them, and `eval/parsers.py::parse_gsd_and_deg_type` turns variant names back into a GSD and degradation type. If you add a variant, keep all three in sync.

**Label formats differ per dataset, and the evaluators branch on them:**
- COWC uses center points only, so it's matched by pixel distance (`match_cowc_points`, `--dist-thr`).
- VEDAI uses the raw 14/15-field format (`cx cy angle class cont occl x1..x4 y1..y4`). Only classes 1 and 11 count as vehicles. Other classes become "ignore" regions, and predictions that land on them are filtered out (`filter_ignored_predictions`).
- Angel uses 8-point polygons (optionally prefixed with a class ID), matched by polygon IoU (`match_polygons`, Shapely).

**Shared eval package (`src/eval/`):** `config`, `parsers`, `metrics` (matching, peak-F1/ROC) and `visualization` (qualitative best/worst images, PPT plots). The scripts import it as `from eval...`, which works because `src/` is on `sys.path` when you run `python src/<script>.py`. Both `yolo_eval.py` and `detr_eval.py` write `Eval_Results_{YOLO,RTDETR}_<weights-stem>/` containing `rq1_summary_metrics.csv`, `rq1_paper_table.tex`, `RQ1_ANSWER_SUMMARY.md`, `ppt_plots/` and `qualitative/`. Those result folders are committed. `evaluate_vedai.py` is an older standalone VEDAI evaluator that duplicates some of the metric code.

**O2-RT-DETR:** this is a vendored copy of the upstream repo (the `ai4rs` package plus `projects/rotated_rtdetr`). The project-specific configs are `projects/rotated_rtdetr/configs/cowc_rtdetr_{128,256,512}.py`. Each one inherits `o2_rtdetr_r50vd_2xb4_72e_dota`, sets `num_classes=1` / `classes=('vehicle',)`, and adds `./O2-RT-DETR` to `sys.path`. `detr_eval.py` and `detr_on_image.py` monkey-patch the mmdet/mmrotate registries (box converters, transforms build, prob-IoU, the DINO transformer forward) so that imports and inference don't crash. Keep those patches when you change the DETR scripts.

**Misc utilities:** `inf_on_image.py` and `detr_on_image.py` run single-image inference. `viewer.py` lets you browse predictions on dataset splits. `convert_coco.py` converts a COCO JSON export into Angel `.txt` labels. `upscale_angel.py` and `prepare_vedai.py` are one-off data prep scripts.
