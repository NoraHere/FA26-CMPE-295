#!/bin/bash
#SBATCH --job-name=ifeval_score
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/019111677/judgebench/logs/%x-%j.out

# IFEval steps 3-4 for one judge (judge set built by jobs/ifeval_generate.sbatch): score it under each condition,
# then stats on PARTS (default discovery; locked only after pre-registration). Resumable.
# Override: MODEL=/path CONDS=bf16,g00 DATA=data/ifeval PARTS=discovery EXTRA="--with-notes" sbatch ...
# Smoke: RUN=ifeval_team_smoke EXTRA="--limit-items 16 --calib-samples 16 --tune-problems 4" sbatch ...
set -euo pipefail
cd /home/019111677/judgebench
source env.sh

echo "host: $(hostname)  job: ${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=index,name,memory.used,compute_mode --format=csv

MODEL=${MODEL:-/home/019111677/models/Qwen3-4B}
DATA=${DATA:-data/ifeval}
TAG=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')
RUN=${RUN:-ifeval_team_${TAG}$([ "$DATA" = data/ifeval ] || echo _$(basename "$DATA"))}
[ -f "$DATA/items.jsonl" ] || { echo "missing $DATA/items.jsonl: run jobs/ifeval_generate.sbatch first"; exit 1; }
python ifeval_score.py --model-path "$MODEL" --data-dir "$DATA" --conditions "${CONDS:-bf16,tao_w8,tao_w4,rtn_w4,gptq_w4,mnr_w4,i8g00,g00,gptq-g00}" \
    --run-name "$RUN" ${EXTRA:-}
python ifeval_stats.py --run "results/$RUN" --data-dir "$DATA" --partitions "${PARTS:-discovery}"
