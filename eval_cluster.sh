#!/bin/bash
#SBATCH --job-name=eval_diffusion_models
#SBATCH --account=deep_learning
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --array=1-3
#SBATCH --output=logs/eval_%a.log

# Enable module command
. /etc/profile.d/modules.sh
module add cuda/12.9

# Set up repo directory
REPO_DIR="${HOME}/symbolic-music-discrete-diffusion-fork"
cd "${REPO_DIR}"

# Create and activate venv (first run only)
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Install dependencies if not already installed
pip install -q --upgrade pip
pip install -q -r requirements.txt
pip install -q note-seq

# Define model configs
# Model 1: Baseline A (event + random)
MODEL1_DIR="log_transformer_melody_1024"
MODEL1_STEP=36500
MODEL1_NAME="transformer"

# Model 2: Baseline B (octuple + random)
MODEL2_DIR="log_octuple_melody_1024"
MODEL2_STEP=32000
MODEL2_NAME="transformer"

# Model 3: Proposed (octuple + bar-aligned)
MODEL3_DIR="log_octuple_1_bar_all_1024"
MODEL3_STEP=32000
MODEL3_NAME="transformer"

# Select which model to run based on SLURM_ARRAY_TASK_ID
case ${SLURM_ARRAY_TASK_ID} in
    1)
        LOAD_DIR="${MODEL1_DIR}"
        LOAD_STEP=${MODEL1_STEP}
        MODEL_NAME="${MODEL1_NAME}"
        echo "Running Baseline A (Event + Random): ${LOAD_DIR}"
        ;;
    2)
        LOAD_DIR="${MODEL2_DIR}"
        LOAD_STEP=${MODEL2_STEP}
        MODEL_NAME="${MODEL2_NAME}"
        echo "Running Baseline B (Octuple + Random): ${LOAD_DIR}"
        ;;
    3)
        LOAD_DIR="${MODEL3_DIR}"
        LOAD_STEP=${MODEL3_STEP}
        MODEL_NAME="${MODEL3_NAME}"
        echo "Running Proposed (Octuple + Partial / Bar-Aligned): ${LOAD_DIR}"
        ;;
esac

# Run evaluation
python evaluate_metrics.py \
    --mode unconditional \
    --n_samples 64 \
    --load_dir "${LOAD_DIR}" \
    --load_step ${LOAD_STEP} \
    --model "${MODEL_NAME}" \
    --tracks melody \
    --bars 64 \
    --split_partition test \
    --bootstrap_ci \
    --n_bootstrap 1000 \
    --save_midis \
    --seed 123

echo "Evaluation complete for model ${SLURM_ARRAY_TASK_ID}"
