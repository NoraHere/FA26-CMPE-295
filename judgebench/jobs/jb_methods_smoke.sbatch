#!/bin/bash
#SBATCH --job-name=jb_methods_smoke
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/home/019111677/judgebench/logs/%x-%j.out

# Smoke test for the new tasks and rounding methods, with small calibration/tuning sets:
#   1. quant_methods.py self-test (random layer)
#   2. JudgeBench knowledge + reasoning, 4 pairs, BF16 / gptq g05 / mnr g05
#   3. S9 scoring with mnr on 2 grids, 4 discovery problems (plumbing only)
# Override: MODEL=/path sbatch jobs/jb_methods_smoke.sbatch
set -euo pipefail
cd /home/019111677/judgebench
source env.sh

echo "host: $(hostname)  job: ${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=index,name,memory.used,compute_mode --format=csv

MODEL=${MODEL:-/home/019111677/models/Qwen3-4B}
TAG=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')
SMALL="--calib-samples 16 --tune-problems 4"
python quant_methods.py
python jb_eval.py --mode both --tasks knowledge,reasoning --limit 4 --model-path "$MODEL"
for method in gptq mnr; do
    python jb_eval.py --mode both --tasks knowledge,reasoning --limit 4 --model-path "$MODEL" \
        --method "$method" --grid g05 $SMALL
done
python s9_score_grids.py --model-path "$MODEL" --partitions discovery --limit-problems 4 --grids 2 \
    --method mnr $SMALL --run-name "s9_smoke_${TAG}_mnr"
