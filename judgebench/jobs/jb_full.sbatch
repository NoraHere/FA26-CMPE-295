#!/bin/bash
#SBATCH --job-name=jb_full
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/019111677/judgebench/logs/%x-%j.out

# All 109 JudgeBench GPT math+code pairs, both modes, Qwen3-4B by default. Resumable: resubmit to continue.
# Override: MODEL=/path/to/model DTYPE=float16 sbatch jobs/jb_full.sbatch
set -euo pipefail
cd /home/019111677/judgebench
source env.sh

echo "host: $(hostname)  job: ${SLURM_JOB_ID:-none}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}  SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
nvidia-smi --query-gpu=index,name,memory.used,compute_mode --format=csv

MODEL=${MODEL:-/home/019111677/models/Qwen3-4B}
DTYPE=${DTYPE:-bfloat16}
python jb_eval.py --mode both --model-path "$MODEL" --dtype "$DTYPE"
