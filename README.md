# Copyright

Copyright 2026, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.
 
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

# About

This repository is a complete workflow hub for training AI models to detect vehicles in satellite and aerial imagery. 

It handles two main AI training pipelines:
1. **O2-RT-DETR (OpenMMLab):** 
2. **YOLO (Ultralytics):** 

Authors:
* Bridgit Graddy
* Emily Dunkel
* Mike Burl

# Status

This is a work in progress. DETR is still in progress, but YOLO is set up and runs

---

## 📋 Table of Contents
- [0. Install Miniforge on Cluster (One-Time Setup)](#0-install-miniforge-on-cluster-one-time-setup)
- [1. Clone Repository & Setup Directories](#1-clone-repository--setup-directories)
- [2. Transfer Files from Local PC to Cluster](#2-transfer-files-from-local-pc-to-cluster)
- [3. Install Conda AI Environments](#3-install-conda-ai-environments)
- [4. Run Model Training (via tmux)](#4-run-model-training-via-tmux)
- [5. Run Testing Inference and Eval](#step-5-run-testing-inference-and-eval)

---

## 0. Install Miniforge on Cluster (One-Time Setup)

If the cluster does not already have Miniforge or Conda installed, run this single command to download and install it in your personal home folder:

```bash
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p ~/miniforge3 && ~/miniforge3/bin/conda init bash && source ~/.bashrc
```

* **What this does:** Downloads the Miniforge installer, installs it silently to `~/miniforge3`, links it to your command line interface, and reloads your terminal session so `conda` commands work immediately.

---

## 1. Clone Repository & Setup Directories

Log into your **Analysis Cluster** via your terminal, clone this codebase, and create the required data and checkpoint folders in a single step:

```bash
git lfs install

git clone https://github.com/jpl-science-team/orbital-vehicle-detector
cd orbital-vehicle-detector

mkdir -p data
```

---

## 2. Transfer Files from Local PC to Cluster

Download the COWC_512.zip from the Google Drive data folder 
https://drive.google.com/drive/folders/1FTX76Ybf0PLiqdwyKsCvyi2WsF-ccoxY

### Transfer Raw Datasets
From Your local terminal

```bash
scp -C /path/to/local/cowc_512.zip user@analysis:~/orbital-vehicle-detector/data/
```

### Unzip the Dataset (On Cluster)
```bash
cd ~/orbital-vehicle-detector/data && unzip cowc_512.zip
```

---

## 3. Install Conda AI Environments

Return to your **Cluster Terminal**. Because DETR and YOLO require different software dependencies, install these two separate environments.

### Environment 1: DETR / O2-RT-DETR (`detr_env`)

```bash
source ~/miniforge3/bin/activate
conda create -n detr_env python=3.9 -y 
conda activate detr_env
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 
pip install -U openmim 
mim install mmengine "mmcv>=2.0.0rc4,<2.2.0" "mmdet>=3.0.0,<3.3.0" "mmrotate>=1.0.0rc1"
pip install -r requirements-detr.txt
```

---

### Environment 2: YOLO (`yolo_env`)

```bash
conda create -n yolo_env python=3.10 -y
conda activate yolo_env 
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements-yolo.txt
```

---

### Step 3.1: Update the dataset.yaml
The `path:` line in `data/cowc_512/dataset.yaml` must be the absolute path to the dataset on your machine. From the repo root, this sets it:
```bash
cd ~/orbital-vehicle-detector
sed -i "s|^path:.*|path: $(pwd)/data/cowc_512|" data/cowc_512/dataset.yaml
head -1 data/cowc_512/dataset.yaml   # should print path: /home/<you>/orbital-vehicle-detector/data/cowc_512
```

## 4. Run Model Training (via `tmux`)

Training an AI model can take hours or days. We execute all training inside a `tmux` session so the process won't crash if your laptop turns off or disconnects.

### Step 4.1: Start a `tmux` Session at repo root

```bash
cd ~/orbital-vehicle-detector
tmux new -s training_session
```

* **What this does:** Opens a background-safe terminal window named `training_session`.
* **To re-open this session later if disconnected:** Type `tmux attach -t training_session`.

---

### Step 4.2: Run Training Commands (Inside `tmux`)

Choose **one** of the following commands based on which model you want to train:

#### Option A: Train O2-RT-DETR (Oriented Bounding Boxes across 4 GPUs)

```bash
source ~/miniforge3/bin/activate
cd ~/orbital-vehicle-detector 
conda activate detr_env 
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 bash O2-RT-DETR/tools/dist_train.sh O2-RT-DETR/projects/rotated_rtdetr/configs/cowc_rtdetr_512.py 4
```

#### Option B: Train YOLO

```bash
source ~/miniforge3/bin/activate
cd ~/orbital-vehicle-detector
conda activate yolo_env 
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 python src/train.py --img_size 640 --batch_size 16 --epochs 150 --data data/cowc_512/dataset.yaml --gpus 2,3
```

`--img_size`, `--data` and `--gpus` are required; `--batch_size` and `--epochs` default to 16 and 150. `--img_size` sets the size images are resized to before training. The COWC tiles are 512×512, so:
* `--img_size 512` trains at the tiles' native resolution (no resizing).
* `--img_size 640` (Ultralytics' default) enlarges every tile 1.25× before training.

Each run saves to its own timestamped folder, `runs/obb/runs/detect/<date>_<time>_yolo11n_res<img_size>/`, so runs at different sizes don't overwrite each other. Training stops early if validation mAP hasn't improved for 15 epochs.

Before starting, check with `nvidia-smi` that the GPUs in `--gpus` are free. If a previous run was killed, leftover processes can hold the GPUs and stall the new run.

---

### Step 4.3: Detach and Leave Training Running

Press `Ctrl + B`, release both keys, and then press `D`.

### Step 4.4: Plot the Training Curves

When training finishes, the run folder contains Ultralytics' own plots (`results.png`, PR/F1 curves, confusion matrix) and `results.csv`. To plot training and validation losses on the same axes, plus validation mAP and precision/recall:
```bash
python src/plot_training.py runs/obb/runs/detect/<run_name>
```
This saves `results_combined.png` in the run folder. A dotted vertical line marks the best epoch (the one saved as `weights/best.pt`).

### Step 5: Run Testing Inference and Eval

Run all of Step 5 from the repo root in `yolo_env`:
```bash
source ~/miniforge3/bin/activate
cd ~/orbital-vehicle-detector
conda activate yolo_env
```

#### Step 5.1: Get the test data
The evaluation reads `test_data/<Dataset>/<variant>/{images,labels}` (COWC and VEDAI at 15, 30 and 60 cm GSD, Angel at 50, 30 and 15 cm). COWC comes in two versions that label cars differently: center points in `test_data/COWC_point` and oriented boxes in `test_data/COWC`. Step 5.1a is required, since it is the only source of VEDAI. Step 5.1b is optional but recommended, because box scoring is the default in Step 5.3. Angel is not distributed with this repo yet: it is only evaluated if you have the raw data in `data/Angel` and run Step 5.1b; otherwise the evaluation prints a warning for each Angel variant and skips it.

**Step 5.1a: download `test_data.zip` (VEDAI, and COWC center points).** Download it from the Google Drive data folder, copy it to the repo root on the cluster (with `scp`, as in Step 2), then unzip it and rename its COWC folder:
https://drive.google.com/drive/folders/1FTX76Ybf0PLiqdwyKsCvyi2WsF-ccoxY
```bash
unzip test_data.zip
mv test_data/COWC test_data/COWC_point
```
In this zip, each COWC car is labeled with a single center point (`0 x y`), not a box. The rename keeps the points separate from the boxes that Step 5.1b writes to `test_data/COWC`. Evaluating with `--cowc-scoring point` (see Step 5.3) reads `test_data/COWC_point`. If you skip the rename, point scoring finds no COWC images and skips COWC with a warning.

**Step 5.1b: build COWC with bounding boxes from label-refiner.** The [label-refiner](https://github.jpl.nasa.gov/science-team-algorithms/label-refiner) repo turns COWC's center points into SAM-fitted oriented boxes. Check it out next to this repo so that `../label-refiner/datasets_refined/COWC_512/test/{images,labels}` exists, then run from the repo root:
```bash
python src/preprocess_rq1_datasets.py
```
This writes all five COWC variants to `test_data/COWC/` (905 tiles, 10,847 boxes, a few minutes on CPU), replacing any COWC data already there (the center points in `test_data/COWC_point` are left alone). About 7% of the boxes are label-refiner's 32×32 px fallback squares, used where SAM didn't give a usable outline. The script also rebuilds VEDAI and Angel if their raw data is in `data/VEDAI` and `data/Angel`; otherwise it skips them, so the VEDAI data from Step 5.1a is kept. Note that these tiles are not the same image files as the ones in the zip.

#### Step 5.2: Put the weights in place
Copy the trained model (`best.pt` from the run folder of Step 4) into `models/weights/test_weights/`, giving it a name that includes the training size. For a run trained with `--img_size 640`:
```bash
mkdir -p models/weights/test_weights
ls runs/obb/runs/detect/   # find your run's folder name
cp runs/obb/runs/detect/<run_name>/weights/best.pt models/weights/test_weights/YOLO11n_15GSD_640.pt
```
Results are written to `Eval_Results_YOLO_<model_name>/`, so reusing a name overwrites earlier results.

#### Step 5.3: Run the evaluation
Four inputs are required:
* `--models`: the weights from Step 5.2 (you can list several, if they were all trained at the same size).
* `--test-dir`: the test data folder from Step 5.1, `test_data`.
* `--val-data`: the training `dataset.yaml` (`data/cowc_512/dataset.yaml`), whose validation split is used to pick the confidence threshold (see the options below).
* `--train-imgsz`: the `--img_size` the model was trained at, so test images are scaled the same way as in training.

With the label-refiner boxes (Step 5.1b), COWC is scored the same way as VEDAI and the validation set: a prediction is correct if its box overlaps a labeled box with IoU >= `--iou-thr` (default 0.5). This is the default:
```bash
python src/yolo_eval.py --models models/weights/test_weights/YOLO11n_15GSD_640.pt --test-dir test_data --val-data data/cowc_512/dataset.yaml --train-imgsz 640
python src/yolo_eval.py --models models/weights/test_weights/YOLO11n_15GSD_512.pt --test-dir test_data --val-data data/cowc_512/dataset.yaml --train-imgsz 512
```

With the center points from `test_data.zip` (Step 5.1a), add `--cowc-scoring point`: a prediction is correct if its center is within `--dist-thr` pixels (default 25 at 15 cm, scaled for coarser GSDs) of a car's center. This was the only behavior before 2026-11-23. Point scoring reads COWC from `test_data/COWC_point`; to point-score the label-refiner boxes by their centers instead, add `--cowc-dir COWC`.
```bash
python src/yolo_eval.py --models models/weights/test_weights/YOLO11n_15GSD_640.pt --test-dir test_data --val-data data/cowc_512/dataset.yaml --train-imgsz 640 --cowc-scoring point
```

Other useful options:
* `--datasets COWC VEDAI` and `--variants 15cm_base` evaluate only some datasets or variants. `--image-glob '*_co.*'` keeps only matching images (here, VEDAI color). Add `--tag <name>` so a partial run writes to `Eval_Results_YOLO_<model_name>_<name>/` instead of overwriting the full results:
  ```bash
  python src/yolo_eval.py --models models/weights/test_weights/YOLO11n_15GSD_640.pt --test-dir test_data --val-data data/cowc_512/dataset.yaml --train-imgsz 640 --datasets VEDAI --variants 15cm_base --image-glob '*_co.*' --tag VEDAI15_co
  ```
* `--threshold-source val` (default) picks each variant's F1 confidence threshold on the validation set (`--val-data`), degraded to that variant's resolution. `--threshold-source test` picks it on the test data itself, which gives optimistic F1 scores. Either way, the CSV also reports `Oracle_F1`/`Oracle_conf`, the best F1 on the test data, as an upper bound.
* To choose the GPU, set `CUDA_VISIBLE_DEVICES`, e.g. `CUDA_VISIBLE_DEVICES=2 python src/yolo_eval.py ...`.

Each results folder contains `rq1_summary_metrics.csv` (one row per dataset and variant: precision, recall, F1, mAP50, thresholds), `rq1_paper_table.tex`, `RQ1_ANSWER_SUMMARY.md`, plots in `ppt_plots/`, and the best and worst images per variant in `qualitative/`. In the committed results, folders ending in `_COWC15_points` were scored by center points on the `test_data.zip` tiles (Step 5.1a), and folders ending in `_COWC15_box` by the label-refiner boxes (Step 5.1b).

To evaluate an O2-RT-DETR checkpoint instead (in `detr_env`), and to compare the two models' results:
```bash
python src/detr_eval.py --config <cfg.py> --models <ckpt.pth> --test-dir test_data
python src/compare_models.py --yolo-csv Eval_Results_YOLO_<name>/rq1_summary_metrics.csv --detr-csv Eval_Results_RTDETR_<name>/rq1_summary_metrics.csv
```
`detr_eval.py` takes the same `--cowc-scoring`, `--cowc-dir` and `--dist-thr` options as `yolo_eval.py`, and scores COWC the same way, so the two models' numbers are comparable. It doesn't support `--threshold-source val` yet: it always picks the F1 threshold on the test data, which is optimistic. To compare like with like, run `yolo_eval.py` with `--threshold-source test` too. `compare_models.py` labels the models `YOLO11n` and `RT-DETR` in the plots; change this with `--yolo-name` and `--detr-name`.
---
