# Copyright

Copyright 2026, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.
 
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

# About

This repository is a complete workflow hub for training AI models to detect vehicles in satellite and aerial imagery. The models draw an oriented (rotated) box around each vehicle. The main research question is how detection accuracy changes as image resolution gets coarser.

It handles two main AI training pipelines:
1. **O2-RT-DETR (OpenMMLab):** a transformer-based detector. Still in progress.
2. **YOLO (Ultralytics):** YOLO11n-OBB, a small, fast detector. Set up and working; this README focuses on it.

Authors:
* Bridgit Graddy
* Emily Dunkel
* Mike Burl

# Status

This is a work in progress. DETR is still in progress, but YOLO is set up and runs.

# Terms Used in This README

* **GSD (ground sample distance):** how much ground one pixel covers. At 15 cm GSD a car is about 30 pixels long; at 60 cm it is about 8.
* **OBB (oriented bounding box):** a box that can rotate to follow the vehicle, stored as its 4 corner points.
* **Datasets:** COWC and VEDAI are aerial images of cars at about 15 cm and 12.5 cm GSD. The models are trained on COWC. Angel is a third test dataset, at 50 cm GSD.
* **Variants:** each test dataset is evaluated at several GSDs, made by resampling the original images:
  * `15cm_base`: the images at 15 cm GSD.
  * `30cm_naive`, `60cm_naive`: shrunk to 30 or 60 cm GSD, so images and cars get smaller in pixels.
  * `30cm_scalenorm`, `60cm_scalenorm`: shrunk to 30 or 60 cm, then enlarged back to the 15 cm pixel size. Cars are the same size in pixels as in `15cm_base`, but blurrier.
  * Angel only: `50cm_base` (original), and `30cm_upscale`, `15cm_upscale` (enlarged).
* **Weights:** the file a trained model is saved in (`.pt`).

---

## 📋 Table of Contents
- [0. Install Miniforge on Cluster (One-Time Setup)](#0-install-miniforge-on-cluster-one-time-setup)
- [1. Clone Repository & Setup Directories](#1-clone-repository--setup-directories)
- [2. Transfer Files from Local PC to Cluster](#2-transfer-files-from-local-pc-to-cluster)
- [3. Install Conda AI Environments](#3-install-conda-ai-environments)
- [4. Run Model Training (via tmux)](#4-run-model-training-via-tmux)
- [5. Run Testing Inference and Eval](#5-run-testing-inference-and-eval)
- [6. Look at Predictions on Single Images](#6-look-at-predictions-on-single-images)
- [7. Train on New Data](#7-train-on-new-data)

---

## 0. Install Miniforge on Cluster (One-Time Setup)

If the cluster does not already have Miniforge or Conda installed (check with `conda --version`), run these commands to download and install it in your personal home folder:

```bash
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p ~/miniforge3 && ~/miniforge3/bin/conda init bash && source ~/.bashrc
```

* **What this does:** Downloads the Miniforge installer, installs it silently to `~/miniforge3`, links it to your command line interface, and reloads your terminal session so `conda` commands work immediately.

---

## 1. Clone Repository & Setup Directories

Log into your **Analysis Cluster** via your terminal, clone this codebase, and create the data folder:

```bash
git clone https://github.com/jpl-science-team/orbital-vehicle-detector
cd orbital-vehicle-detector

mkdir -p data
```

All commands in this README are run from this repo folder, `~/orbital-vehicle-detector`, unless a step says otherwise.

---

## 2. Transfer Files from Local PC to Cluster

Download `cowc_512.zip` (the COWC training and validation tiles) from the Google Drive data folder to your own computer:
https://drive.google.com/drive/folders/1FTX76Ybf0PLiqdwyKsCvyi2WsF-ccoxY

### Transfer Raw Datasets
From a terminal on your own computer (not the cluster), copy it to the cluster. Replace `/path/to/local/` with where you downloaded it, `user` with your cluster username, and `analysis` with the cluster's hostname:

```bash
scp -C /path/to/local/cowc_512.zip user@analysis:~/orbital-vehicle-detector/data/
```

### Unzip the Dataset (On Cluster)
```bash
cd ~/orbital-vehicle-detector/data && unzip cowc_512.zip && cd ..
```
This creates `data/cowc_512/` with `images/{train,val,test}` and `labels/{train,val,test}`. The repo's `configs/cowc_512.yaml` already points to this folder, so there's nothing to edit.

---

## 3. Install Conda AI Environments

Return to your **Cluster Terminal**. Because DETR and YOLO require different software dependencies, they each get their own environment. If you only use YOLO, you can skip Environment 1.

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
source ~/miniforge3/bin/activate
conda create -n yolo_env python=3.10 -y
conda activate yolo_env 
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements-yolo.txt
```

Each time you open a new terminal, activate the environment again with `source ~/miniforge3/bin/activate` and `conda activate yolo_env` (or `detr_env`).

---

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

Choose **one** of the following commands based on which model you want to train. If you're not sure, use YOLO (Option B).

Before starting, run `nvidia-smi` to see which GPUs are free: a GPU with no processes listed under it and little memory in use is free. The GPU numbers you pick go in `--gpus` (YOLO). If a previous run was killed, leftover processes can hold the GPUs and stall the new run.

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
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 python src/train.py --img_size 640 --batch_size 16 --epochs 150 --data configs/cowc_512.yaml --gpus 2,3
```

* `--img_size`, `--data` and `--gpus` are required.
  * `--img_size` sets the size images are resized to before training. The COWC tiles are 512×512, so `--img_size 512` trains at the tiles' native resolution (no resizing), and `--img_size 640` (Ultralytics' default) enlarges every tile 1.25×. Write down which one you used: the evaluation needs it (Step 5.3).
  * `--data` is the dataset file `configs/cowc_512.yaml`. It lists the training and validation image folders (relative to the repo root) and the class name, `vehicle`. Training uses the `train` images and checks the model on the `val` images after every epoch; the `test` images aren't used. To train on other data, copy this file and change the folders.
  * `--gpus` is a comma-separated list of free GPU numbers from `nvidia-smi`, e.g. `0` or `2,3`.
* `--batch_size` and `--epochs` default to 16 and 150.
* Training starts from Ultralytics' pretrained `yolo11n-obb.pt`, which is downloaded to the repo root the first time. Pass `--weights <file.pt>` to start from another checkpoint instead.
* `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1` work around a multi-GPU communication problem on the cluster; keep them.

Each run saves to its own timestamped folder, `runs/obb/runs/detect/<date>_<time>_yolo11n_res<img_size>/`, so runs at different sizes don't overwrite each other. Training stops early if validation mAP hasn't improved for 15 epochs. The terminal prints the progress of each epoch.

---

### Step 4.3: Detach and Leave Training Running

Press `Ctrl + B`, release both keys, and then press `D`. Training keeps running; use `tmux attach -t training_session` to check on it.

### Step 4.4: Plot the Training Curves

When training finishes, the run folder contains Ultralytics' own plots (`results.png`, PR/F1 curves, confusion matrix) and `results.csv`. To plot training and validation losses on the same axes, plus validation mAP and precision/recall (in `yolo_env`):
```bash
python src/plot_training.py runs/obb/runs/detect/<run_name>
# e.g. python src/plot_training.py runs/obb/runs/detect/20261106_135621_yolo11n_res640
```
`<run_name>` is the timestamped folder from Step 4.2; `ls runs/obb/runs/detect/` lists them.
This saves `results_combined.png` in the run folder. A dotted vertical line marks the best epoch (the one saved as `weights/best.pt`).

---

## 5. Run Testing Inference and Eval

This step runs the trained model on test images at each GSD variant and measures how many vehicles it finds. Run all of Step 5 from the repo root in `yolo_env`:
```bash
source ~/miniforge3/bin/activate
cd ~/orbital-vehicle-detector
conda activate yolo_env
```

### Step 5.1: Get the test data
The evaluation reads `test_data/<Dataset>/<variant>/{images,labels}` (see [Terms](#terms-used-in-this-readme) for the variants). COWC comes in two versions that label cars differently: center points in `test_data/COWC_point` and oriented boxes in `test_data/COWC`.

* **Step 5.1a is required,** since it is the only source of VEDAI.
* **Step 5.1b is optional but recommended,** because the evaluation scores COWC with the boxes by default.
* **Angel** is not distributed with this repo yet. It is only evaluated if you have the raw data in `data/Angel` and run Step 5.1b; otherwise the evaluation prints a warning for each Angel variant and skips it.

**Step 5.1a: download `test_data.zip` (VEDAI, and COWC center points).** Download it from the Google Drive data folder, copy it to the repo root on the cluster with `scp` as in Step 2, then unzip it and rename its COWC folder:
https://drive.google.com/drive/folders/1FTX76Ybf0PLiqdwyKsCvyi2WsF-ccoxY
```bash
# on your own computer:
scp -C /path/to/local/test_data.zip user@analysis:~/orbital-vehicle-detector/
# on the cluster, in the repo root:
unzip test_data.zip
mv test_data/COWC test_data/COWC_point
```
In this zip, each COWC car is labeled with a single center point (`0 x y`), not a box. The rename keeps the points separate from the boxes that Step 5.1b writes to `test_data/COWC`. Evaluating with `--cowc-scoring point` (see Step 5.3) reads `test_data/COWC_point`. If you skip the rename, point scoring finds no COWC images and skips COWC with a warning.

**Step 5.1b: build COWC with bounding boxes from label-refiner.** The [label-refiner](https://github.com/jpl-science-team/label-refiner) repo turns COWC's center points into SAM-fitted oriented boxes. Clone it next to this repo (so it is at `~/label-refiner`) and follow its README to produce the refined COWC test split. Cloning alone isn't enough, since the refined labels aren't committed there. When `../label-refiner/datasets_refined/COWC_512/test/{images,labels}` exists, run from this repo's root:
```bash
python src/preprocess_rq1_datasets.py --raw-dir data --cowc-refined-dir ../label-refiner/datasets_refined/COWC_512/test --out-dir test_data
```
* `--raw-dir` (raw VEDAI and Angel data, if any) and `--cowc-refined-dir` (the label-refiner boxes) are only read, never changed.
* `--out-dir` is where the test data is written; it's the folder you pass to the evaluation as `--test-dir`.

This writes all five COWC variants to `test_data/COWC/` (905 tiles, 10,847 boxes, a few minutes on CPU), replacing any COWC data already there. The center points in `test_data/COWC_point` are left alone. About 7% of the boxes are label-refiner's 32×32 px fallback squares, used where SAM didn't give a usable outline. The script also rebuilds VEDAI and Angel if their raw data is in `data/VEDAI` and `data/Angel`; otherwise it skips them, so the VEDAI data from Step 5.1a is kept. The COWC tiles made here are not the same image files as the ones in the zip, so box-scored and point-scored COWC results are on slightly different images.

### Step 5.2: Put the weights in place
Copy the trained model (`best.pt` from the run folder of Step 4) into `models/weights/test_weights/`, giving it a name that says what it is. Here `15GSD` means it was trained on 15 cm GSD data, and `640` is the `--img_size` it was trained at:
```bash
mkdir -p models/weights/test_weights
ls runs/obb/runs/detect/   # find your run's folder name
cp runs/obb/runs/detect/<run_name>/weights/best.pt models/weights/test_weights/YOLO11n_15GSD_640.pt
# e.g. cp runs/obb/runs/detect/20261106_135621_yolo11n_res640/weights/best.pt models/weights/test_weights/YOLO11n_15GSD_640.pt
```
Results are written to `Eval_Results_YOLO_<model_name>/` (e.g. `Eval_Results_YOLO_YOLO11n_15GSD_640/`), so reusing a name overwrites earlier results.

**No trained model yet?** Some pretrained weights are included in `vehicle_detector.zip` in the Google Drive data folder (the same link as in Step 2). You can use them to try the evaluation without training first. Copy them into `models/weights/test_weights/` as above, and pass the `--train-imgsz` they were trained at in Step 5.3.

### Step 5.3: Run the evaluation
Four inputs are required:
* `--models`: the weights from Step 5.2 (you can list several, if they were all trained at the same size).
* `--test-dir`: the test data folder from Step 5.1, `test_data`.
* `--val-data`: the dataset file the model was trained with (`configs/cowc_512.yaml`, the same as `--data` in Step 4.2). Its validation images are used to pick the confidence threshold (see the options below).
* `--train-imgsz`: the `--img_size` the model was trained at (Step 4.2), so test images are scaled the same way as in training. Getting this wrong gives worse scores without any error.

The evaluation uses a GPU and takes a while, so you may want to run it in `tmux` too (Step 4.1).

**Box scoring (default).** With the label-refiner boxes (Step 5.1b), COWC is scored the same way as VEDAI and the validation set: a prediction counts as correct if its box overlaps a labeled box with IoU (overlap area divided by combined area) >= `--iou-thr` (default 0.5):
```bash
python src/yolo_eval.py --models models/weights/test_weights/YOLO11n_15GSD_640.pt --test-dir test_data --val-data configs/cowc_512.yaml --train-imgsz 640
python src/yolo_eval.py --models models/weights/test_weights/YOLO11n_15GSD_512.pt --test-dir test_data --val-data configs/cowc_512.yaml --train-imgsz 512
```

**Point scoring.** With the center points from `test_data.zip` (Step 5.1a), add `--cowc-scoring point`: a prediction counts as correct if its center is within `--dist-thr` pixels (default 25 at 15 cm, scaled for coarser GSDs) of a car's center. Point scoring reads COWC from `test_data/COWC_point`; to point-score the label-refiner boxes by their centers instead, add `--cowc-dir COWC`.
```bash
python src/yolo_eval.py --models models/weights/test_weights/YOLO11n_15GSD_640.pt --test-dir test_data --val-data configs/cowc_512.yaml --train-imgsz 640 --cowc-scoring point
```

Other useful options:
* `--datasets COWC VEDAI` and `--variants 15cm_base` evaluate only some datasets or variants. `--image-glob '*_co.*'` keeps only matching images (here, VEDAI color). Add `--tag <name>` so a partial run writes to `Eval_Results_YOLO_<model_name>_<name>/` instead of overwriting the full results:
  ```bash
  python src/yolo_eval.py --models models/weights/test_weights/YOLO11n_15GSD_640.pt --test-dir test_data --val-data configs/cowc_512.yaml --train-imgsz 640 --datasets VEDAI --variants 15cm_base --image-glob '*_co.*' --tag VEDAI15_co
  ```
* The model gives each prediction a confidence score, and only predictions above a threshold are counted. `--threshold-source val` (default) picks each variant's threshold on the validation images (`--val-data`), resampled to that variant's GSD. `--threshold-source test` picks it on the test data itself, which gives optimistic F1 scores. Either way, the CSV also reports `Oracle_F1`/`Oracle_conf`, the best F1 possible on the test data, as an upper bound.
* `--out-dir <folder>` writes the `Eval_Results_YOLO_*` folders inside `<folder>` instead of the repo root.
* To choose the GPU, set `CUDA_VISIBLE_DEVICES`, e.g. `CUDA_VISIBLE_DEVICES=2 python src/yolo_eval.py ...`.

**Reading the results.** Each results folder contains:
* `rq1_summary_metrics.csv`: one row per dataset and variant. TP, FP and FN count correct predictions, false detections and missed vehicles. Precision is the fraction of predictions that are real vehicles, recall is the fraction of vehicles found, F1 combines the two, and mAP50 summarizes precision and recall over all thresholds at IoU 0.5. Higher is better for all four.
* `RQ1_ANSWER_SUMMARY.md`: the same table in Markdown, easy to read on GitHub; start here.
* `rq1_paper_table.tex`: the same numbers as a LaTeX table.
* `ppt_plots/`: bar charts of mAP50 and F1 per variant, and per-dataset plots of how accuracy changes with GSD.
* `qualitative/`: the two best and two worst images per variant. Correct predictions are drawn in green, false detections in red and missed vehicles in blue.

In the committed results, folders ending in `_COWC15_points` were scored by center points on the `test_data.zip` tiles (Step 5.1a), and folders ending in `_COWC15_box` by the label-refiner boxes (Step 5.1b).

To evaluate an O2-RT-DETR checkpoint instead (in `detr_env`), and to compare the two models' results:
```bash
python src/detr_eval.py --config <cfg.py> --models <ckpt.pth> --test-dir test_data
python src/compare_models.py --yolo-csv Eval_Results_YOLO_<name>/rq1_summary_metrics.csv --detr-csv Eval_Results_RTDETR_<name>/rq1_summary_metrics.csv
```
`detr_eval.py` takes the same `--cowc-scoring`, `--cowc-dir` and `--dist-thr` options as `yolo_eval.py` and is meant to score COWC the same way, but this hasn't been tested yet (see Status). The committed `Eval_Results_RTDETR_*` results predate it and were scored by center points at a fixed 25 px. It doesn't support `--threshold-source val` yet: it always picks the F1 threshold on the test data, which is optimistic. To compare like with like, run `yolo_eval.py` with `--threshold-source test` too. `compare_models.py` labels the models `YOLO11n` and `RT-DETR` in the plots; change this with `--yolo-name` and `--detr-name`.

---

## 6. Look at Predictions on Single Images

These scripts are for looking at a model's predictions, not for scoring. Run them from the repo root in `yolo_env`.

**One image:** `inf_on_image.py` runs a YOLO model on one image and saves a copy with the predicted boxes drawn (no labels or scores). Boxes below `--conf` (default 0.4) aren't drawn.
```bash
python src/inf_on_image.py --model models/weights/test_weights/YOLO11n_15GSD_640.pt --image test_data/COWC/15cm_base/images/Potsdam_ISPRS_top_potsdam_2_13_RGB_patch_512_1280.png --out predictions_drawn.jpg
```

**Browse the test data:** `viewer.py` opens a desktop window for stepping through the test data from Step 5.1 with the model's predictions drawn on it. Pick the dataset (`COWC`, `COWC_point`, `VEDAI`, ...) and GSD variant from the two dropdowns, and use the arrow keys or the buttons to move between images. Images are scaled as in `yolo_eval.py`, so pass the same `--train-imgsz`. Boxes below `--conf` (default 0.1) aren't drawn. The window needs a display, so run it on a machine with a screen or over `ssh -X`, not in a plain terminal session.
```bash
python src/viewer.py --model models/weights/test_weights/YOLO11n_15GSD_640.pt --test-dir test_data --train-imgsz 640
```

---

## 7. Train on New Data

You can train YOLO on your own oriented-box data with the same `train.py`, by describing your data in a dataset file.

### Step 7.1: Arrange the data
Put the images and labels in this layout (any folder works; `data/` is a good place):
```
data/my_data/
├── images/
│   ├── train/   tile_001.png, tile_002.png, ...
│   └── val/     tile_101.png, ...
└── labels/
    ├── train/   tile_001.txt, tile_002.txt, ...
    └── val/     tile_101.txt, ...
```
* Each image needs a label file with the same name, ending in `.txt`. An image with no objects gets an empty `.txt` file.
* Each line of a label file is one object: `<class_id> x1 y1 x2 y2 x3 y3 x4 y4`. The four points are the box corners, in order around the box. Each x is divided by the image width and each y by the image height, so all values are between 0 and 1. For example, `0 0.10 0.20 0.15 0.20 0.15 0.30 0.10 0.30` is a class-0 box near the top left. COWC's labels in `data/cowc_512/labels/` are an example of this format.
* The validation images should be separate from the training images; around 10% of the data is common. The model is checked on them after every epoch, and the best epoch is saved as `best.pt`.
* If your labels are in another format, convert them to this one first. `src/convert_coco.py` is an example of a converter.

### Step 7.2: Make a dataset file
Copy the template and edit the lines marked `EDIT`:
```bash
cp configs/template.yaml configs/my_data.yaml
```
* `path:` the data folder, `data/my_data`. Relative paths are from the repo root.
* `train:` and `val:` the image folders inside it. Change them only if your folders have other names.
* `names:` the class names, numbered from 0 in the same order as the class IDs in your labels.

### Step 7.3: Train
Train as in Step 4.2, with your dataset file:
```bash
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 python src/train.py --data configs/my_data.yaml --img_size 512 --gpus 0
```
Set `--img_size` to your images' size to train at their native resolution, or larger to enlarge them. If your images aren't square, Ultralytics resizes the longer side to `--img_size`. Then check the training curves as in Step 4.4. To start from a model you already trained instead of the pretrained `yolo11n-obb.pt`, add `--weights <best.pt>`.

### Step 7.4: Machines without internet access
Training itself doesn't need internet, but setting up the environment and the first run normally download files. Before moving to a machine without internet, collect these on a machine that has it:
1. **The `yolo_env` environment.** `pip` and `conda` can't install packages offline. One option is `conda-pack`: on a machine with internet, run `conda install -n base -c conda-forge conda-pack`, then `conda pack -n yolo_env -o yolo_env.tar.gz`. Copy the archive over, then unpack and activate it:
   ```bash
   mkdir -p ~/envs/yolo_env && tar -xzf yolo_env.tar.gz -C ~/envs/yolo_env
   source ~/envs/yolo_env/bin/activate
   conda-unpack
   ```
   The new machine needs a similar Linux system and an NVIDIA driver that supports the CUDA version of the packed PyTorch (check with `python -c "import torch; print(torch.version.cuda)"` before packing).
2. **`yolo11n-obb.pt`**, the pretrained starting weights. `train.py` downloads it to the repo root if it's missing. Copy it into the repo root on the new machine (it's gitignored, so it isn't in the repo).
3. **`yolo26n.pt`** (optional). At the start of training, Ultralytics uses it for a quick GPU check. Without it, the check is skipped with a warning and training continues. Copy it into the repo root to avoid the warning.
4. **`~/.config/Ultralytics/Arial.ttf`**, the font Ultralytics uses for its plots, downloaded the first time it draws them. Copy it to the same folder on the new machine.

On the new machine, set this before running, so Ultralytics skips its online checks instead of waiting for them to time out:
```bash
export YOLO_OFFLINE=true
```

### Evaluating on new data
`train.py`, `plot_training.py` and `inf_on_image.py` work with any data in this format. `yolo_eval.py` doesn't yet: it only knows the COWC, VEDAI and Angel test sets. To evaluate a new test set, you would need to:
* add the dataset and its variants to `DATASET_VARIANTS` in `src/eval/config.py`,
* make sure its labels can be read by `parse_label_file` in `src/eval/parsers.py`, either by converting them to the format it reads for Angel (one object per line, as the 8 pixel coordinates `x1 y1 x2 y2 x3 y3 x4 y4` with no class ID in front) or by adding a branch for the new format,
* and, if the new data isn't at 15 cm GSD, update the GSD handling (`get_pixel_gsd` and `get_content_gsd` in `src/eval/parsers.py`, and the validation threshold calibration in `src/eval/calibration.py`, which assumes 15 cm validation data).

For a quick check without those changes, `inf_on_image.py` (Step 6) shows the model's predictions on any image, and Ultralytics' own validation reports mAP on your `val` images: `yolo val task=obb model=<best.pt> data=configs/my_data.yaml imgsz=<size>`.
