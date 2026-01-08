#!/bin/bash
#SBATCH --job-name=music_diff_sample
#SBATCH --output=logs/sample_%j.out
#SBATCH --error=logs/sample_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=student 
#SBATCH --account=deep_learning
#SBATCH --gres=gpu:1

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

# Configuration
MODEL="octuple_ddpm"
RUN_DIR="runs/octuple_ddpm_trio_octuple"
N_SAMPLES=4
SAMPLE_STEPS=100
BATCH_SIZE=4

# Ensure output directories exist
mkdir -p logs
mkdir -p ${RUN_DIR}/samples/uncond
mkdir -p ${RUN_DIR}/samples/infill

echo "============================================"
echo "Running Unconditional Generation"
echo "============================================"

python3 -m smdiff.cli.sample \
  --model ${MODEL} \
  --load_dir ${RUN_DIR} \
  --task uncond \
  --n_samples ${N_SAMPLES} \
  --sample_steps ${SAMPLE_STEPS} \
  --batch_size ${BATCH_SIZE}

echo ""
echo "============================================"
echo "Running Infilling"
echo "============================================"

python3 -m smdiff.cli.sample \
  --model ${MODEL} \
  --load_dir ${RUN_DIR} \
  --task infill \
  --input_midi data/POP909/001/001.mid \
  --mask_start_bar 4 \
  --mask_end_bar 16 \
  --n_samples ${N_SAMPLES} \
  --sample_steps ${SAMPLE_STEPS} \
  --batch_size ${BATCH_SIZE}

echo ""
echo "Job finished at $(date)"
echo "Samples saved to ${RUN_DIR}/samples/"
