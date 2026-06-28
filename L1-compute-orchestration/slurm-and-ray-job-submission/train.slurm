#!/bin/bash
#SBATCH --job-name=hello-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

module load cuda/12.6 2>/dev/null || true
source "${VENV:-$HOME/.venv}/bin/activate" 2>/dev/null || true

python -c "
import torch
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device', torch.cuda.get_device_name(0))
"
