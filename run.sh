#!/bin/bash
#SBATCH --nodelist=gpuvm24
#SBATCH --output=logs/train_%j.log
# Exit if any command fails
set -e

# Activate your virtual environment if needed
# source /path/to/venv/bin/activate

# Set working directory to project root
cd /vol/bitbucket/lb124/Thesis/shakespeer/diffusion-text-generation  # or wherever your repo lives
mkdir -p logs
# Optionally clear cache (e.g., for CUDA or logs)
export PYTHONUNBUFFERED=1  # real-time output

# Run the training + sampling script
python -u run.py | tee logs/train_${SLURM_JOB_ID}.log
