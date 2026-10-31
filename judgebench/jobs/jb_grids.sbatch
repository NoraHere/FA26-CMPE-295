#!/bin/bash
#SBATCH --job-name=jb_grids
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/019111677/judgebench/logs/%x-%j.out

# All 109 JudgeBench pairs, both modes, on each shifted weight grid (quant_grids.py). Resumable.
# Default: the 16 INT4 grids. Override: GRIDS="i8g00 i8g05" MODEL=/path sbatch jobs/jb_grids.sbatch
set -euo pipefail
cd /home/019111677/judgebench
source env.sh

echo "host: $(hostname)  job: ${SLURM_JOB_ID:-none}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}  SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
nvidia-smi --query-gpu=index,name,memory.used,compute_mode --format=csv

MODEL=${MODEL:-/home/019111677/models/Qwen3-4B}
GRIDS=${GRIDS:-$(printf 'g%02d ' $(seq 0 15))}
for grid in $GRIDS; do
    echo "=== grid $grid"
    python jb_eval.py --mode both --model-path "$MODEL" --grid "$grid"
done
