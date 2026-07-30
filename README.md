orbital-vehicle-detector
This repository serves as the central hub for training, hyperparameter optimization, and evaluation of computer vision models optimized for detecting vehicles in satellite and high-altitude aerial imagery.

It supports oriented object detection (OBB) pipelines using O2-RT-DETR / MMRotate as well as standard YOLO workflows.

🛠️ Environment Setup
Because OpenMMLab/DETR frameworks and YOLO have different dependency trees and underlying C++ compilation needs, we maintain two isolated Conda environments.

orbital-vehicle-detector/
├── requirements-detr.txt      # DETR / MMRotate / OpenMMLab stack
├── requirements-yolo.txt      # Ultralytics / YOLO stack
Environment 1: DETR / O2-RT-DETR (requirements-detr.txt)
This environment powers the oriented object detection pipeline (RT-DETR, MMRotate, MMDetection).

1. Create and Activate the Conda Environment
Bash
conda create -n detr_env python=3.9 -y
conda activate detr_env
2. Install PyTorch & Core Dependencies
Bash
pip install -r requirements-detr.txt
3. Build MMCV (Mac Apple Silicon / macOS Fix)
Building mmcv on macOS requires forcing the C++17 compiler standard and bypassing broken MPS Metal checks during extension building.

Bash
# Clone and build MMCV locally
git clone https://github.com/open-mmlab/mmcv.git /tmp/mmcv-src
cd /tmp/mmcv-src
git checkout v2.0.1

# Bypass MPS availability check during compilation
sed -i '' 's/torch.backends.mps.is_available()/False/g' setup.py
# Force C++17 compilation flag
sed -i '' 's/-std=c++14/-std=c++17/g' setup.py

# Export flags and compile
export CFLAGS="-Wno-invalid-specialization"
export CXXFLAGS="-Wno-invalid-specialization"
export MMCV_WITH_OPS=1

pip install .
cd -
rm -rf /tmp/mmcv-src
4. Setup Custom Repositories
Ensure submodules or custom model directories are present:

Bash
# If submodules are used
git submodule update --init --recursive
Environment 2: YOLO Pipeline (requirements-yolo.txt)
This environment handles standard bounding box training and rapid prototyping via Ultralytics.

1. Create and Activate the Conda Environment
Bash
conda create -n yolo_env python=3.10 -y
conda activate yolo_env
2. Install Dependencies
Bash
pip install -r requirements-yolo.txt
🚀 Running Inference
DETR / O2-RT-DETR
To run inference with oriented bounding boxes (OBB) on a satellite tile:

Bash
conda activate detr_env
python src/detr_on_image.py
Note: The inference script automatically handles PyTorch 2.6+ checkpoint loading rules, Python 3.9 type-hint patching, and MMDetection registry collision bypasses.

YOLO
To run YOLO inference or evaluation:

Bash
conda activate yolo_env
python src/yolo_on_image.py  # or yolo predict model=runs/yolo_best.pt source=data/sample.tif

