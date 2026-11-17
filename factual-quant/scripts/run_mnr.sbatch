#!/bin/bash
#SBATCH --job-name=qjudge-mnr
#SBATCH --partition=gpuqs,condo
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-3%4
#SBATCH --output=logs/%x-%A_%a.out
# mnr_w4 (GPTQ + margin-neutral rounding, simulated) on TruthfulQA, RB2 Factuality and JudgeBench
# knowledge. H100: the 8B judges need model + note gradient + original weights in memory at once.

RUNS=("Qwen3-4B" "Qwen3-8B" "Qwen2.5-7B-Instruct" "Phi-4-mini-instruct")
M=${RUNS[$SLURM_ARRAY_TASK_ID]}

module purge
module load python3/3.12.12 ml/lambda/gpu-ai ml/torch/2.6
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=~/qjudge/pylib
hostname; nvidia-smi --query-gpu=name,memory.total --format=csv
python -u ~/qjudge/scripts/fact_eval.py --model ~/qjudge/models/$M --quant mnr4
