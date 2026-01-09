#!/bin/bash
#SBATCH --job-name=debug_oct_mel
#SBATCH --output=logs/debug_oct_mel_%j.out
#SBATCH --error=logs/debug_oct_mel_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=student
#SBATCH --account=deep_learning
#SBATCH --gpus=1
#SBATCH --mem=24G

set -euo pipefail

source .venv/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

mkdir -p logs
nvidia-smi || true

python3 src/smdiff/cli/train.py \
  --model octuple_ddpm \
  --dataset_id pop909_melody_octuple \
  --batch_size 2 \
  --train_steps 200 \
  --steps_per_log 10 \
  --steps_per_eval 100 \
  --steps_per_sample 200 \
  --steps_per_checkpoint 200 \
  --seed 67 \
  --wandb \
  --wandb_project "octubert-music-debug" \
  --wandb_name "debug-octuple-ddpm-melody-octuple"

echo "Job finished at $(date)"
