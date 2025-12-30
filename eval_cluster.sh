#!/bin/bash
#SBATCH --job-name=eval_diffusion_models
#SBATCH --account=deep_learning
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --output=logs/eval_all.log

# Enable module command
. /etc/profile.d/modules.sh
module add cuda/12.8

# Set up repo directory
REPO_DIR="${HOME}/symbolic-music-discrete-diffusion-fork"
cd "${REPO_DIR}"

# Create and activate venv (first run only)
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Install dependencies if not already installed
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio
pip install --prefer-binary -r requirements.txt

# Define model configs
MODEL1_DIR="log_transformer_melody_1024"
MODEL1_STEP=36500

MODEL2_DIR="log_octuple_melody_1024"
MODEL2_STEP=32000

MODEL3_DIR="log_octuple_1_bar_all_1024"
MODEL3_STEP=32000

# Run all three evaluations sequentially
echo "Running Baseline A (Event + Random): ${MODEL1_DIR}"
python3 evaluate_metrics.py \
    --mode unconditional \
    --n_samples 64 \
    --load_dir "${MODEL1_DIR}" \
    --load_step ${MODEL1_STEP} \
    --model "transformer" \
    --tracks melody \
    --bars 64 \
    --split_partition test \
    --bootstrap_ci \
    --n_bootstrap 1000 \
    --save_midis \
    --seed 123

echo "Running Baseline B (Octuple + Random): ${MODEL2_DIR}"
python3 evaluate_metrics.py \
    --mode unconditional \
    --n_samples 64 \
    --load_dir "${MODEL2_DIR}" \
    --load_step ${MODEL2_STEP} \
    --model "transformer" \
    --tracks melody \
    --bars 64 \
    --split_partition test \
    --bootstrap_ci \
    --n_bootstrap 1000 \
    --save_midis \
    --seed 123

echo "Running Proposed (Octuple + Bar-Aligned): ${MODEL3_DIR}"
python3 evaluate_metrics.py \
    --mode unconditional \
    --n_samples 64 \
    --load_dir "${MODEL3_DIR}" \
    --load_step ${MODEL3_STEP} \
    --model "transformer" \
    --tracks melody \
    --bars 64 \
    --split_partition test \
    --bootstrap_ci \
    --n_bootstrap 1000 \
    --save_midis \
    --seed 123

echo "All evaluations complete"
