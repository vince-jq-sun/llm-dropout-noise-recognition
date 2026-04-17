#!/bin/bash
#SBATCH --job-name=wandb-sweep
#SBATCH --output=sweep_%A_%a.out
#SBATCH --error=sweep_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=04:00:00

# Usage: bash scripts/run_sweep.sh <SWEEP_ID> <NUM_JOBS> <MAX_CONCURRENT>
# Example: bash scripts/run_sweep.sh lawzero-default/llm-mechanistic-detection/6ywxddt7 20 5
# The script self-submits as an sbatch array job.

set -euo pipefail

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "Array task ${SLURM_ARRAY_TASK_ID} starting wandb agent for sweep: ${SWEEP_ID}"
    uv run wandb agent --count 1 "${SWEEP_ID}"
    exit 0
fi

SWEEP_ID="$1"
NUM_JOBS="$2"
MAX_CONCURRENT="$3"

sbatch --export=ALL,SWEEP_ID="${SWEEP_ID}" --array="0-$((NUM_JOBS - 1))%${MAX_CONCURRENT}" "$0"
