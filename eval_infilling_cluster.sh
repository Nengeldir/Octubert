#!/bin/bash
#SBATCH --job-name=eval_infilling_models
#SBATCH --account=deep_learning
#SBATCH --partition=student
#SBATCH --time=06:00:00
#SBATCH --output=logs/eval_infilling_all.log

# Enable module command
#. /etc/profile.d/modules.sh
# CUDA 12.8 is available (cuda/13.0 module doesn't exist)
#module purge
#module add cuda/12.8 || module load cuda

# Set up repo directory
REPO_DIR="${HOME}/symbolic-music-discrete-diffusion-fork"
cd "${REPO_DIR}"

# Create and activate venv (first run only)
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Install dependencies if not already installed
pip3 install --upgrade pip 'setuptools>=70.0.0' wheel
pip3 install torch torchvision
# Install key packages compatible with Python 3.12, skip old pinned versions
pip3 install --prefer-binary numpy scipy scikit-learn pandas matplotlib jupyter notebook ipywidgets statsmodels
pip3 install --prefer-binary absl-py addict aiofiles anyio audioread bleach click fastapi filelock huggingface-hub joblib librosa mido nicegui note-seq numba pillow pretty-midi pydub pyfluidsynth pypianoroll python-dotenv python-socketio pyyaml rich soundfile tensorboard transformers uvicorn visdom watchfiles websockets miditoolkit

# Verify CUDA is available
echo "CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA device count: $(python3 -c 'import torch; print(torch.cuda.device_count())')"
nvcc --version

# Define model configs
MODEL1_DIR="log_transformer_melody_1024"
MODEL1_STEP=36500

MODEL2_DIR="log_octuple_melody_1024"
MODEL2_STEP=32000

MODEL3_DIR="log_octuple_1_bar_all_1024"
MODEL3_STEP=32000

MODEL4_DIR="log_octuple_mixed_melody_1024"
MODEL4_STEP=86000

# Common infilling parameters
N_SAMPLES=64
SAMPLE_STEPS=100
GAP_START=256
GAP_END=512
N_BOOTSTRAP=1000
SEED=123
SPLIT_PATH="data/splits/pop909_split.json"
SPLIT_PARTITION="test"
PROCESSED_DIR="data/processed"

# Run all four infilling evaluations sequentially
echo "Running Infilling - Transformer Baseline: ${MODEL1_DIR}"
python3 -u evaluate_metrics.py \
    --mode infilling \
    --n_samples ${N_SAMPLES} \
    --sample_steps ${SAMPLE_STEPS} \
    --load_dir "${MODEL1_DIR}" \
    --load_step ${MODEL1_STEP} \
    --model "transformer" \
    --tracks melody \
    --bars 64 \
    --no_split \
    --gap_start ${GAP_START} \
    --gap_end ${GAP_END} \
    --save_midis \
    --bootstrap_ci --n_bootstrap 1 \
    --seed ${SEED}

# echo "Running Infilling - Octuple Baseline (Random): ${MODEL2_DIR}"
# python3 -u evaluate_metrics.py \
#     --mode infilling \
#     --n_samples ${N_SAMPLES} \
#     --sample_steps ${SAMPLE_STEPS} \
#     --load_dir "${MODEL2_DIR}" \
#     --load_step ${MODEL2_STEP} \
#     --model "octuple" \
#     --tracks melody \
#     --bars 64 \
#     --split_path "${SPLIT_PATH}" \
#     --split_partition "${SPLIT_PARTITION}" \
#     --processed_dir "${PROCESSED_DIR}" \
#     --gap_start ${GAP_START} \
#     --gap_end ${GAP_END} \
#     --save_midis \
#     --bootstrap_ci --n_bootstrap 1 \
#     --seed ${SEED}

# echo "Running Infilling - Octuple + 1_bar_all Masking: ${MODEL3_DIR}"
# python3 -u evaluate_metrics.py \
#     --mode infilling \
#     --n_samples ${N_SAMPLES} \
#     --sample_steps ${SAMPLE_STEPS} \
#     --load_dir "${MODEL3_DIR}" \
#     --load_step ${MODEL3_STEP} \
#     --model "octuple" \
#     --tracks melody \
#     --bars 64 \
#     --split_path "${SPLIT_PATH}" \
#     --split_partition "${SPLIT_PARTITION}" \
#     --processed_dir "${PROCESSED_DIR}" \
#     --masking_strategy "1_bar_all" \
#     --gap_start ${GAP_START} \
#     --gap_end ${GAP_END} \
#     --save_midis \
#     --bootstrap_ci --n_bootstrap 1 \
#     --seed ${SEED}

# echo "Running Infilling - Octuple + Mixed Masking: ${MODEL4_DIR}"
# python3 -u evaluate_metrics.py \
#     --mode infilling \
#     --n_samples ${N_SAMPLES} \
#     --sample_steps ${SAMPLE_STEPS} \
#     --load_dir "${MODEL4_DIR}" \
#     --load_step ${MODEL4_STEP} \
#     --model "octuple" \
#     --tracks melody \
#     --bars 64 \
#     --split_path "${SPLIT_PATH}" \
#     --split_partition "${SPLIT_PARTITION}" \
#     --processed_dir "${PROCESSED_DIR}" \
#     --masking_strategy "mixed" \
#     --gap_start ${GAP_START} \
#     --gap_end ${GAP_END} \
#     --bootstrap_ci --n_bootstrap 1 \
#     --save_midis \
#     --seed ${SEED}

# echo "All infilling evaluations complete"
