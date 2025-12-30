#!/bin/bash
#SBATCH --job-name=eval_diffusion_models
#SBATCH --account=dl_jobs
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=1-3
#SBATCH --output=logs/eval_%a.log

# Load modules (adjust based on your cluster setup)
module load cuda/11.8
module load python/3.10

# Set up paths
SCRATCH_DIR="/work/scratch/${USER}/diffusion_eval"
REPO_DIR="${SCRATCH_DIR}/symbolic-music-discrete-diffusion-fork"
LOCAL_REPO="/path/to/local/symbolic-music-discrete-diffusion-fork"

# Create scratch directories
mkdir -p "${SCRATCH_DIR}"
cd "${SCRATCH_DIR}"

# Clone repo (if not already present)
if [ ! -d "${REPO_DIR}" ]; then
    git clone <your-repo-url> "${REPO_DIR}"
fi

cd "${REPO_DIR}"

# Create virtual environment (first run only)
if [ ! -d "venv" ]; then
    python -m venv venv
    source venv/bin/activate
    pip install -q -r requirements.txt
    pip install -q note-seq
else
    source venv/bin/activate
fi

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
        echo "Running Proposed (Octuple + Bar-Aligned): ${LOAD_DIR}"
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
