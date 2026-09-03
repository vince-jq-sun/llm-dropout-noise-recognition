#!/bin/bash
#SBATCH --job-name=spe_mc
#SBATCH --account=torch_pr_1004_cds
#SBATCH --partition=h200_cds
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out

set -euo pipefail

SPE_REPO=${SPE_REPO:-/scratch/$USER/repos/llm-dropout-noise-recognition}
SPE_ENV=${SPE_ENV:-/scratch/$USER/conda_envs/spe}

MODEL=${MODEL:-qwen3_32b}
NUM_SAMPLES=${NUM_SAMPLES:-10}
ACTIVE_PERTURBATIONS=${ACTIVE_PERTURBATIONS:-DROPOUT+NOISE}
DROPOUT_RATE=${DROPOUT_RATE:-0.36}
NOISE_STD=${NOISE_STD:-0.165}
ALIASES=${ALIASES:-none}
RUN_NAME=${RUN_NAME:-qwen32_zero_shot_smoke}
SENTENCES_FILE=${SENTENCES_FILE:-$SPE_REPO/data/sentences/15tok.txt}

export HF_HOME=${HF_HOME:-/scratch/$USER/hf_cache}
export TMPDIR=${TMPDIR:-/scratch/$USER/tmp/$SLURM_JOB_ID}

RUN_DIR="$SPE_REPO/results/nyu/${RUN_NAME}_${SLURM_JOB_ID}"
HF_SNAPSHOT_REF="$HF_HOME/hub/models--Qwen--Qwen3-32B/refs/main"
HYDRA_ACTIVE_PERTURBATIONS=${ACTIVE_PERTURBATIONS//+/,}

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export WANDB_DIR="$RUN_DIR"

mkdir -p "$RUN_DIR" "$TMPDIR"

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "$SPE_ENV"

if [[ ! -x "$SPE_ENV/bin/spe-multiclass" ]]; then
    echo "ERROR: spe-multiclass is not installed in $SPE_ENV" >&2
    exit 2
fi

if [[ ! -f "$SENTENCES_FILE" ]]; then
    echo "ERROR: sentence file not found: $SENTENCES_FILE" >&2
    exit 2
fi

if [[ ! -f "$HF_SNAPSHOT_REF" ]]; then
    echo "ERROR: Qwen3-32B is not cached at $HF_SNAPSHOT_REF" >&2
    exit 2
fi

{
    echo "# date=$(date -Is)"
    echo "# job_id=$SLURM_JOB_ID"
    echo "# host=$(hostname)"
    echo "# run_dir=$RUN_DIR"
    echo "# repo_commit=$(git -C "$SPE_REPO" rev-parse HEAD)"
    echo "# repo_status_begin"
    git -C "$SPE_REPO" status --short
    echo "# repo_status_end"
    echo "# model=$MODEL"
    echo "# hf_model=Qwen/Qwen3-32B"
    echo "# hf_snapshot=$(<"$HF_SNAPSHOT_REF")"
    echo "# num_samples=$NUM_SAMPLES"
    echo "# active_perturbations=$HYDRA_ACTIVE_PERTURBATIONS"
    echo "# dropout_rate=$DROPOUT_RATE"
    echo "# noise_std=$NOISE_STD"
    echo "# aliases=$ALIASES"
    echo "# sentences_file=$SENTENCES_FILE"
    sha256sum "$SENTENCES_FILE"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    python --version
    python -m pip freeze
} | tee "$RUN_DIR/provenance.txt"

cd "$RUN_DIR"

"$SPE_ENV/bin/spe-multiclass" \
    --config-name classification_a_b \
    model="$MODEL" \
    aliases="$ALIASES" \
    experiment.num_samples="$NUM_SAMPLES" \
    experiment.sentences_file="$SENTENCES_FILE" \
    "perturbation.active_perturbations=[$HYDRA_ACTIVE_PERTURBATIONS]" \
    perturbation.dropout_rate="$DROPOUT_RATE" \
    perturbation.noise_std="$NOISE_STD" \
    wandb.name="$RUN_NAME" \
    hydra.run.dir="$RUN_DIR/hydra" \
    2>&1 | tee "$RUN_DIR/run.log"

echo "# completed=$(date -Is)" | tee -a "$RUN_DIR/provenance.txt"
