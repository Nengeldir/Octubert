#!/bin/bash
#SBATCH --job-name=musicbert_diff_train
#SBATCH --output=logs/musicbert_diff_train_%j.out
#SBATCH --error=logs/musicbert_diff_train_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=student 
#SBATCH --account deep_learning
#SBATCH --gpus=1

# Initialize environment
# . /home/lziltener/jupyter/bin/activate
source $HOME/my_env/bin/activate
nvidia-smi
echo "Starting MusicBERT diffusion training job on $HOSTNAME"
echo "Date: $(date)"

# Ensure logs directory exists
mkdir -p logs

# Run the training script
python3 train_musicBERT_diffusion.py

echo "Job finished at $(date)"
