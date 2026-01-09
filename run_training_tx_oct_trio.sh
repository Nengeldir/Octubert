#!/bin/bash
#SBATCH --job-name=conv_mini_train
#SBATCH --output=logs/conv_mini_%j.out
#SBATCH --error=logs/conv_mini_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=student
#SBATCH --account=deep_learning
#SBATCH --gpus=1
#SBATCH --mem=24G

set -euo pipefail

source .venv/bin/activate

# Ensure Python can import project packages when running scripts by path
# Prefer src/ on PYTHONPATH so 'smdiff' resolves
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

nvidia-smi || true
echo "Starting conv mini training on $(hostname)"
echo "Date: $(date)"

# Ensure logs directory exists
mkdir -p logs

# Sanity: show Python and torch CUDA availability
python -V || python3 -V
python -c "import torch; print('Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())" || true

# Run unified training CLI with small settings
python3 src/smdiff/cli/train.py \
  --model octuple_ddpm \
  --set tracks=trio_octuple \
  --set dataset_path=data/POP909_trio_octuple.npy \
  --set batch_size=8 \
  --set train_steps=400 \
  --set steps_per_log=10 \
  --set steps_per_sample=400 \
  --set steps_per_eval=100 \
  --set steps_per_checkpoint=400 \
  --wandb \
  --wandb_project "octubert-music" \
  --wandb_name "mini-test-octuple-ddpm-trio" \

echo "Job finished at $(date)"