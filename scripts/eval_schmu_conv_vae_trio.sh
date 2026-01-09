#!/bin/bash
#SBATCH --job-name=eval_schmu_conv_trio
#SBATCH --output=logs/eval_schmu_conv_trio_%j.out
#SBATCH --error=logs/eval_schmu_conv_trio_%j.err
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

MODEL_ID="schmu_conv_vae"
RUN_DIR="runs/schmu_conv_vae"
DATASET_ID="pop909_trio"

# Infilling: 50 MIDI files × 2 regions = 100 samples
INFILL_MIDI_DIR="data/POP909/test"
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

python3 -m smdiff.cli.evaluate \
  --task uncond \
  --model "${MODEL_ID}" \
  --load_dir "${RUN_DIR}" \
  --dataset_id "${DATASET_ID}" \
  --n_samples 100 \
  --sample_steps 100 \
  --batch_size 16 \
  --ema \
  --device cuda

echo "========================================"
echo "Infilling evaluation (two regions, sequential per MIDI)"
echo "MIDI dir: ${INFILL_MIDI_DIR}"
echo "N MIDIs:  ${N_MIDIS}"
echo "Regions:  [${MASK1_START},${MASK1_END}) and [${MASK2_START},${MASK2_END})"
echo "========================================"

python3 -m smdiff.cli.evaluate \
  --task infill \
  --model "${MODEL_ID}" \
  --load_dir "${RUN_DIR}" \
  --dataset_id "${DATASET_ID}" \
  --input_midi_dir "${INFILL_MIDI_DIR}" \
  --n_midis "${N_MIDIS}" \
  --samples_per_midi "${SAMPLES_PER_MIDI}" \
  --mask_start_bar "${MASK1_START}" \
  --mask_end_bar "${MASK1_END}" \
  --mask2_start_bar "${MASK2_START}" \
  --mask2_end_bar "${MASK2_END}" \
  --n_samples 100 \
  --sample_steps 100 \
  --batch_size 16 \
  --ema \
  --device cuda

echo "Job finished at $(date)"
