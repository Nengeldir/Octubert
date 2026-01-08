#!/bin/bash
#SBATCH --job-name=eval_ddpm
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=student 
#SBATCH --account=deep_learning

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
MODEL_ID="octuple_ddpm"
RUN_DIR="runs/octuple_ddpm_trio_octuple"  # Update to your actual run directory
DATASET_ID="pop909_trio_octuple"
N_SAMPLES=4

# Navigate to repository
cd ~/Octubert  # Update to your repository path

echo "========================================"
echo "Starting Model Evaluation"
echo "Model: $MODEL_ID"
echo "Run Dir: $RUN_DIR"
echo "Dataset: $DATASET_ID"
echo "========================================"

# ============================================================
# EXAMPLE 1: UNCONDITIONAL EVALUATION - GENERATE NEW SAMPLES
# ============================================================
# This configuration generates new samples during evaluation
# Use when: You want fresh samples with specific sampling parameters
echo ""
echo "Running unconditional evaluation (generating new samples)..."
python -m smdiff.cli.evaluate \
    --task uncond \
    --model $MODEL_ID \
    --load_dir $RUN_DIR \
    --dataset_id $DATASET_ID \
    --n_samples $N_SAMPLES \
    --sample_steps 100 \
    --batch_size 16 \
    --ema \
    --save_samples \
    --device cuda

echo "Unconditional evaluation complete!"

# ============================================================
# EXAMPLE 2: INFILLING EVALUATION - USE EXISTING SAMPLES
# ============================================================
# This configuration loads pre-generated samples from a directory
# Use when: You already have samples and want to re-evaluate with different metrics
# or save computation time
echo ""
echo "Running infilling evaluation (using existing samples)..."

# Path to pre-generated samples (update to where your samples are stored)
SAMPLE_DIR="$RUN_DIR/samples/infill"

# Note: If samples don't exist yet, first generate them with sample.py:
# python -m smdiff.cli.sample \
#     --task infill \
#     --model $MODEL_ID \
#     --load_dir $RUN_DIR \
#     --dataset_id $DATASET_ID \
#     --n_samples $N_SAMPLES \
#     --mask_start_bar 16 \
#     --mask_end_bar 32 \
#     --output_dir $SAMPLE_DIR

# Run evaluation using existing samples
python -m smdiff.cli.evaluate \
    --task infill \
    --sample_dir $SAMPLE_DIR \
    --dataset_id $DATASET_ID \
    --n_samples $N_SAMPLES \
    --mask_start_bar 4 \
    --mask_end_bar 16 \
    --input_midi_dir data/POP909 \
    --output_dir $RUN_DIR/metrics \
    --device cuda

echo "Infilling evaluation complete!"

echo ""
echo "========================================"
echo "All evaluations complete!"
echo "Results saved to: $RUN_DIR/metrics/"
echo "========================================"
