#!/bin/bash
#SBATCH --job-name=qjudge-jb
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-19%4
#SBATCH --output=logs/%x-%A_%a.out
# JudgeBench knowledge (154 pairs, both orders) for runs that already have TruthfulQA + RB2:
# 4 judges x {bf16, tao_w8 = int8, tao_w4 = int4, rtn_w4 = rtn4, gptq_w4 = gptq4}.

JUDGES=("Qwen3-4B" "Qwen3-8B" "Qwen2.5-7B-Instruct" "Phi-4-mini-instruct")
QUANTS=(bf16 int8 int4 rtn4 gptq4)
M=${JUDGES[$((SLURM_ARRAY_TASK_ID / 5))]}
Q=${QUANTS[$((SLURM_ARRAY_TASK_ID % 5))]}

module purge
module load python3/3.12.12 ml/lambda/gpu-ai ml/torch/2.6
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=~/qjudge/pylib

source ~/qjudge/scripts/pick_gpu.sh
hostname; nvidia-smi --query-gpu=name,memory.total --format=csv
python -u ~/qjudge/scripts/fact_eval.py --model ~/qjudge/models/$M --quant $Q --only jb
