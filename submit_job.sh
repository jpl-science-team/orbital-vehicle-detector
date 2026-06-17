#!/bin/bash
#SBATCH --job-name=cowc_dual_gpu     # Name of your job
#SBATCH --output=runs/train_out.log # Allocation terminal printouts log file
#SBATCH --error=runs/train_err.log  # Error logs catch file
#SBATCH --partition=gpu             # Name of your cluster's GPU partition
#SBATCH --gres=gpu:2                # Allocate exactly 2 Tesla M60 GPUs
#SBATCH --cpus-per-task=8           # Allocate 8 CPU cores (4 workers per GPU)
#SBATCH --mem=32G                   # 32GB of host system memory to handle staging
#SBATCH --time=02:00:00             # Max wall-clock time allocation (2 hours)

# 1. Move cleanly into your actual repository workspace directory
cd /home/graddy/repos/orbital-vehicle-detector

# 2. Activate your isolated cluster-side native .venv 
source /home/graddy/miniforge3/etc/profile.d/conda.sh
conda activate vehicle-detector
# 3. Upgrade pip and ensure cluster-ready requirements are locked in
pip install --upgrade pip
pip install -r requirements.txt

# 4. Kick off the parallel execution engine using the dual-GPU config
python src/train.py
