#!/bin/bash
#SBATCH --job-name=jb_methods
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/019111677/judgebench/logs/%x-%j.out

# One judge on JudgeBench TASKS (default reasoning), both modes, under each precision / rounding method:
#   BF16, INT8 RTN (i8g00), INT4 RTN (g00), INT4 GPTQ (gptq-g00), INT4 margin-neutral (mnr-g00).
# Resumable. Override: MODEL=/path TASKS=reasoning GRID=00 sbatch jobs/jb_methods.sbatch
set -euo pipefail
cd /home/019111677/judgebench
source env.sh

echo "host: $(hostname)  job: ${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=index,name,memory.used,compute_mode --format=csv

MODEL=${MODEL:-/home/019111677/models/Qwen3-4B}
TASKS=${TASKS:-reasoning}
GRID=${GRID:-00}
run() { echo "=== $*"; python jb_eval.py --mode both --tasks "$TASKS" --model-path "$MODEL" "$@"; }
run
run --grid "i8g$GRID"
run --grid "g$GRID"
run --grid "g$GRID" --method gptq
run --grid "g$GRID" --method mnr
