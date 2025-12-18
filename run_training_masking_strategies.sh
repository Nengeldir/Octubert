#!/bin/bash
#SBATCH --job-name=music_diff_masking_strategies
#SBATCH --output=logs/train_masking_%j.out
#SBATCH --error=logs/train_masking_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=student 
#SBATCH --account deep_learning
#SBATCH --gpus=1

# Initialize environment
. /home/lconconi/jupyter/bin/activate
nvidia-smi
echo "Starting masking strategies training job on $HOSTNAME"
echo "Date: $(date)"

# Ensure logs directory exists
mkdir -p logs

# Run the training script
python3 train_masking_strategies.py

echo "Job finished at $(date)"
