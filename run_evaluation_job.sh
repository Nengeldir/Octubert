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
echo "Starting evaluation on $(hostname)"
echo "Date: $(date)"

# Ensure logs directory exists
mkdir -p logs

# Sanity: show Python and torch CUDA availability
python -V || python3 -V
python -c "import torch; print('Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())" || true

# Configuration
MODEL_ID="octuple_ddpm"
RUN_DIR="runs/schmu_tx_vae_trio"  # Update to your actual run directory
DATASET_ID="pop909_trio"

# Mini smoke-test sizes (fast). Increase once the job works end-to-end.
N_SAMPLES_UNCOND=8
N_MIDIS=2
SAMPLES_PER_MIDI=1

# Two separate infill regions (NOT simultaneous): total infill samples = N_MIDIS * 2 * SAMPLES_PER_MIDI
# NO: We simplified to single region masking.
MASK1_START_BAR=4
MASK1_END_BAR=8
# Region 2 args removed/ignored

INFILL_MIDI_DIR="data/test/POP909"  # recursively searched; excludes any 'versions/' subfolders

# Navigate to repository (prefer SLURM submission directory)
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_DIR"

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
    --n_samples $N_SAMPLES_UNCOND \
    --sample_steps 100 \
    --batch_size 16 \
    --ema \
    --save_samples \
    --device cuda

echo "Unconditional evaluation complete!"

# ============================================================
# EXAMPLE 2: INFILLING EVALUATION - GENERATE CONDITIONED SAMPLES
# ============================================================
# This configuration generates conditioned samples directly from MIDIs.
# It will recursively scan --input_midi_dir and omit any files under 'versions/'.
echo ""
echo "Running infilling evaluation (generating conditioned samples)..."

# This uses --input_midi_dir and runs each MIDI once (region1 only),
# producing N_MIDIS * 1 * SAMPLES_PER_MIDI samples.
python -m smdiff.cli.evaluate \
    --task infill \
    --model $MODEL_ID \
    --load_dir $RUN_DIR \
    --dataset_id $DATASET_ID \
    --input_midi_dir "$INFILL_MIDI_DIR" \
    --n_midis $N_MIDIS \
    --samples_per_midi $SAMPLES_PER_MIDI \
    --mask_start_bar $MASK1_START_BAR \
    --mask_end_bar $MASK1_END_BAR \
    --n_samples $((N_MIDIS * SAMPLES_PER_MIDI)) \
    --output_dir $RUN_DIR/metrics \
    --device cuda \
    --save_samples

echo "Infilling evaluation complete!"

echo ""
echo "========================================"
echo "All evaluations complete!"
echo "Results saved to: $RUN_DIR/metrics/"
echo "========================================"
