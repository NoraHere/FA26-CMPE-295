#!/bin/bash
#SBATCH --job-name=jb_think
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/019111677/judgebench/logs/%x-%j.out

# Exploratory thinking-on check (jb_think.py): BF16, pointwise, greedy, JudgeBench TASKS. Resumable.
# Override: MODEL=/path TASKS=reasoning BUDGET=4096 LIMIT=4 sbatch jobs/jb_think.sbatch
set -euo pipefail
cd /home/019111677/judgebench
source env.sh

echo "host: $(hostname)  job: ${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=index,name,memory.used,compute_mode --format=csv

MODEL=${MODEL:-/home/019111677/models/Qwen3-8B}
python jb_think.py --model-path "$MODEL" --tasks "${TASKS:-reasoning}" --think-budget "${BUDGET:-4096}" \
    --limit "${LIMIT:-0}"
