#!/bin/bash
#SBATCH --job-name=musicbert_test
#SBATCH --output=logs/musicbert_test_%j.out
#SBATCH --error=logs/musicbert_test_%j.err
#SBATCH --time=00:15:00
#SBATCH --partition=student 
#SBATCH --account deep_learning
#SBATCH --gpus=1

# Initialize environment
. /home/lziltener/jupyter/bin/activate
pip install -r requirements.txt
nvidia-smi
echo "Starting MusicBERT diffusion test job on $HOSTNAME"
echo "Date: $(date)"

# Ensure logs directory exists
mkdir -p logs

# Run the training script with test parameters
# Short training run (100 steps) to verify pipeline works
python3 train.py \
    --model octuple_musicbert \
    --dataset_path data/processed \
    --train_steps 100 \
    --batch_size 4 \
    --steps_per_log 10 \
    --steps_per_eval 50 \
    --steps_per_checkpoint 100 \
    --steps_per_sample 100 \
    --lr 0.0001

echo "Job finished at $(date)"
