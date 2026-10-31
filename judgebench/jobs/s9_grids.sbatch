#!/bin/bash
#SBATCH --job-name=s9_grids
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/019111677/judgebench/logs/%x-%j.out

# S9: score the injection set at BF16 + 16 shifted INT4 grids, then lottery statistics. Resumable.
# Real run only after docs/s9_preregistration.md is frozen. Smoke test (discovery only):
#   PARTITIONS=discovery LIMIT=4 GRIDS=2 RUN=s9_smoke_qwen3-4b sbatch --exclude=cs002 jobs/s9_grids.sbatch
set -euo pipefail
cd /home/019111677/judgebench
source env.sh

echo "host: $(hostname)  job: ${SLURM_JOB_ID:-none}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}  SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
nvidia-smi --query-gpu=index,name,memory.used,compute_mode --format=csv

MODEL=${MODEL:-/home/019111677/models/Qwen3-4B}
PARTITIONS=${PARTITIONS:-development,locked}
LIMIT=${LIMIT:-0}
GRIDS=${GRIDS:-16}
INT8=${INT8:-16}       # INT8 control grids (pre-registered: 16; 0 = none)
NOTES=${NOTES:-strong}  # primary note (pre-registered); the all-main-notes summary is always written too
RUN=${RUN:-s9_$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')}
EVAL=${EVAL:-locked}

[ -f data/s9_injection/items.jsonl ] || python s9_build_injection.py
python s9_score_grids.py --model-path "$MODEL" --partitions "$PARTITIONS" \
    --limit-problems "$LIMIT" --grids "$GRIDS" --int8-grids "$INT8" --run-name "$RUN"
python s9_lottery_stats.py --items "results/$RUN/items.jsonl" --eval-partition "$EVAL" --notes "$NOTES"
python s9_lottery_stats.py --items "results/$RUN/items.jsonl" --eval-partition "$EVAL"
