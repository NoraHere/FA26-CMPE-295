#!/bin/bash
#SBATCH --job-name=ifeval_gen
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/019111677/judgebench/logs/%x-%j.out

# IFEval steps 1-2: generate responses (Qwen3-1.7B, SmolLM2-1.7B), label them with the official checkers,
# and build the judge set (ifeval_build.py).
# Smoke: LIMIT=8 OUT=data/ifeval_smoke sbatch jobs/ifeval_generate.sbatch
set -euo pipefail
cd /home/019111677/judgebench
source env.sh
export NLTK_DATA=/home/019111677/nltk_data

echo "host: $(hostname)  job: ${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=index,name,memory.used,compute_mode --format=csv
python ifeval_generate.py --limit "${LIMIT:-0}" --out-dir "${OUT:-data/ifeval}"
python ifeval_build.py --data-dir "${OUT:-data/ifeval}"   # once here, so parallel scoring jobs only read it
