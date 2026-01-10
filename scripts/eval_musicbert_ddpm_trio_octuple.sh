#!/bin/bash
#SBATCH --job-name=eval_mb_ddpm_trio
#SBATCH --output=logs/eval_mb_ddpm_trio_%j.out
#SBATCH --error=logs/eval_mb_ddpm_trio_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=student
#SBATCH --account=deep_learning
#SBATCH --gpus=1
#SBATCH --mem=24G

set -euo pipefail

source .venv/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

mkdir -p logs
nvidia-smi || true

MODEL_ID="musicbert_ddpm"
RUN_DIR="runs/musicbert_ddpm_trio_octuple"
DATASET_ID="pop909_trio_octuple"

# Infilling: 50 MIDI files × 2 regions = 100 samples
INFILL_MIDI_DIR="data/test/POP909"
N_MIDIS=50
SAMPLES_PER_MIDI=1

# Region 1 (bars)
MASK1_START=16
MASK1_END=32

# Region 2 (bars)
MASK2_START=32
MASK2_END=48

echo "========================================"
echo "Unconditional evaluation"
echo "Model:   ${MODEL_ID}"
echo "Run dir: ${RUN_DIR}"
echo "Dataset: ${DATASET_ID}"
echo "========================================"

python3 evaluate_octuple.py \
  --task uncond \
  --model "${MODEL_ID}" \
  --load_dir "${RUN_DIR}" \
  --n_samples 100 \
  --batch_size 4

echo "========================================"
echo "Infilling evaluation"
echo "MIDI dir: ${INFILL_MIDI_DIR}"
echo "========================================"

python3 evaluate_octuple.py \
  --task infill \
  --model "${MODEL_ID}" \
  --load_dir "${RUN_DIR}" \
  --input_midi_dir "${INFILL_MIDI_DIR}" \
  --batch_size 4

echo "Job finished at $(date)"
