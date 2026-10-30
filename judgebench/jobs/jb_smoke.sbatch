#!/bin/bash
#SBATCH --job-name=jb_smoke
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/019111677/judgebench/logs/%x-%j.out

# 4-pair smoke test (2 math + 2 code), both modes, BF16, Qwen3-4B by default.
# Override: MODEL=/path/to/model DTYPE=float16 sbatch jobs/jb_smoke.sbatch
# Second step forces truncation (max-len 1024) to exercise that code path.
set -euo pipefail
cd /home/019111677/judgebench
source env.sh

echo "host: $(hostname)  job: ${SLURM_JOB_ID:-none}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}  SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
nvidia-smi --query-gpu=index,name,memory.used,compute_mode --format=csv
which python && python --version

MODEL=${MODEL:-/home/019111677/models/Qwen3-4B}
DTYPE=${DTYPE:-bfloat16}
TAG=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')
python jb_eval.py --mode both --limit 4 --model-path "$MODEL" --dtype "$DTYPE"
python jb_eval.py --mode both --limit 4 --model-path "$MODEL" --dtype "$DTYPE" --max-len 1024 \
    --run-name "jb_smoke_trunc1024_${TAG}_${DTYPE}"
