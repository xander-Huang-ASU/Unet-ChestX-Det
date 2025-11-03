#!/bin/bash
#SBATCH --job-name=Unet-ChestX-Det
#SBATCH --partition=htc               # ✅ 指定 htc 分区
#SBATCH --output=Unet-ChestX-Det_%j.out
#SBATCH --error=Unet-ChestX-Det_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1                  # 先要 1 块 GPU
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# ===== 环境设置 =====
module purge
module load cuda-11.8.0-gcc-12.1.0
module load cudnn/8.9.7       # CUDA 11.8 对应的 cuDNN

source ~/.bashrc
conda activate imagePro

# 确认 GPU 信息
nvidia-smi
python -c "import torch; print('Torch CUDA:', torch.version.cuda); print('Is CUDA available?', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"

# ===== 运行训练 =====
python main.py
