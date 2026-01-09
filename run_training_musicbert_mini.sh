#!/bin/bash
#SBATCH --job-name=musicbert_mini_train
#SBATCH --output=logs/musicbert_mini_%j.out
#SBATCH --error=logs/musicbert_mini_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=student
#SBATCH --account=deep_learning

set -euo pipefail

source .venv/bin/activate

# Ensure Python can import project packages when running scripts by path
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

nvidia-smi || true
echo "Starting MusicBERT mini training on $(hostname)"
echo "Date: $(date)"

# Ensure logs directory exists
mkdir -p logs

# Sanity: show Python and torch CUDA availability
python -V || python3 -V
python -c "import torch; print('Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())" || true

# Run unified training CLI with small settings
python3 src/smdiff/cli/train.py \
  --model musicbert_ddpm \
  --dataset_id pop909_melody_octuple \
  --batch_size 4 \
  --train_steps 400 \
  --steps_per_log 10 \
  --steps_per_sample 400 \
  --steps_per_eval 100 \
  --steps_per_checkpoint 400 \
  --wandb \
  --wandb_project "octubert-music" \
  --wandb_name "mini-test-musicbert-ddpm-melody"

echo "Job finished at $(date)"
