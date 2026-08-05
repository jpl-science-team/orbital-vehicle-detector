# `orbital-vehicle-detector`

This repository is a complete workflow hub for training AI models to detect vehicles in satellite and aerial imagery. 

It handles two main AI training pipelines:
1. **O2-RT-DETR (OpenMMLab):** 
2. **YOLO (Ultralytics):** 

---

## 📋 Table of Contents
- [0. Install Miniforge on Cluster (One-Time Setup)](#0-install-miniforge-on-cluster-one-time-setup)
- [1. Clone Repository & Setup Directories](#1-clone-repository--setup-directories)
- [2. Transfer Files from Local PC to Cluster](#2-transfer-files-from-local-pc-to-cluster)
- [3. Install Conda AI Environments](#3-install-conda-ai-environments)
- [4. Run Model Training (via tmux)](#4-run-model-training-via-tmux)
- [5. Run Model Inference (Testing New Images)](#5-run-model-inference-testing-new-images)

---

## 0. Install Miniforge on Cluster (One-Time Setup)

If the cluster does not already have Miniforge or Conda installed, run this single command to download and install it in your personal home folder:

```bash
curl -L -O [https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh) 
bash Miniforge3-Linux-x86_64.sh -b -p ~/miniforge3 && ~/miniforge3/bin/conda init bash && source ~/.bashrc
```

* **What this does:** Downloads the Miniforge installer, installs it silently to `~/miniforge3`, links it to your command line interface, and reloads your terminal session so `conda` commands work immediately.

---

## 1. Clone Repository & Setup Directories

Log into your **Analysis Cluster** via your terminal, clone this codebase, and create the required data and checkpoint folders in a single step:

```bash
git clone https://github.jpl.nasa.gov/science-team-algorithms/orbital-vehicle-detector.git

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

### Step 3: Unzip the Dataset (On Cluster)
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
### Step 3: Unzip the Dataset (On Cluster)
```bash
cd ~/orbital-vehicle-detector/data && unzip cowc_512.zip
```

---
### Step 3.1: Updated the dataset.yaml
The dataset yaml must contain the absoulte path to you dataset and splits
```bash
cd ~/orbital-vehicle-detector/data/cowc_512
pwd
```



## 4. Run Model Training (via `tmux`)

Training an AI model can take hours or days. We execute all training inside a `tmux` session so the process won't crash if your laptop turns off or disconnects.

### Step 4.1: Start a `tmux` Session

```bash
tmux new -s training_session
```

* **What this does:** Opens a background-safe terminal window named `training_session`.
* **To re-open this session later if disconnected:** Type `tmux attach -t training_session`.

---

### Step 4.2: Run Training Commands (Inside `tmux`)

Choose **one** of the following commands based on which model you want to train:

#### Option A: Train O2-RT-DETR (Oriented Bounding Boxes across 4 GPUs)

```bash
source ~/miniforge3/etc/profile.d/conda.sh && cd ~/orbital-vehicle-detector && conda activate detr_env && NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 bash O2-RT-DETR/tools/dist_train.sh O2-RT-DETR/projects/rotated_rtdetr/configs/cowc_rtdetr_256.py 4
```

#### Option B: Train YOLO

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate yolo_env 
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 python src/train.py --img_size 512 --batch_size 16 --epochs 150 --data data/cowc_512/dataset.yaml --gpus 2,3
```

---

### Step 4.3: Detach and Leave Training Running

Press `Ctrl + B`, release both keys, and then press `D`.


---
