# NYU Torch runners

These scripts reproduce the paper experiments on NYU Torch. The compute nodes
have no internet access, so install dependencies and cache models on the login
node first.

## Login-node setup

```bash
module purge
module load anaconda3/2025.06

python -m venv --copies /scratch/$USER/venvs/spe
source /scratch/$USER/venvs/spe/bin/activate

cd /scratch/$USER/repos/llm-dropout-noise-recognition
python -m pip install --upgrade pip
python -m pip install -e .

export HF_HOME=/scratch/$USER/hf_cache
hf download Qwen/Qwen3-32B
```

The repository lives on scratch because the account's home quota is full:

```bash
git clone --branch reproduce/arxiv-v2 --single-branch \
  https://github.com/vince-jq-sun/llm-dropout-noise-recognition.git \
  /scratch/$USER/repos/llm-dropout-noise-recognition
```

## Balanced zero-shot smoke test

Submit from the repository root:

```bash
sbatch --export=ALL,\
RUN_NAME=qwen32_zero_shot_smoke,\
MODEL=qwen3_32b,\
NUM_SAMPLES=10,\
ACTIVE_PERTURBATIONS=DROPOUT+NOISE,\
DROPOUT_RATE=0.36,\
NOISE_STD=0.165,\
ALIASES=none \
scripts/nyu/run_spe_multiclass.sbatch
```

The job uses one H200 and writes to a unique directory under `results/nyu/`.
It keeps W&B enabled in offline mode so the raw per-sample result tables are
preserved. Do not set `wandb.enabled=false` for formal runs.

Check the job with:

```bash
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode,MaxRSS
tail -100 spe_mc_JOB_ID.out
```
