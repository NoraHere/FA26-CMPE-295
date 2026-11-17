#!/bin/bash
#SBATCH --job-name=qjudge-fact
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-11%4
#SBATCH --output=logs/%x-%A_%a.out
# TruthfulQA + RewardBench 2 Factuality (scripts/fact_eval.py). 2 GPUs so pick_gpu.sh can skip a broken one.

RUNS=("Qwen3-4B bf16" "Qwen3-4B int8" "Qwen3-4B int4"
      "Qwen3-8B bf16" "Qwen3-8B int8" "Qwen3-8B int4"
      "Qwen2.5-7B-Instruct bf16" "Qwen2.5-7B-Instruct int8" "Qwen2.5-7B-Instruct int4"
      "Phi-4-mini-instruct bf16" "Phi-4-mini-instruct int8" "Phi-4-mini-instruct int4")
read M Q <<< "${RUNS[$SLURM_ARRAY_TASK_ID]}"

module purge
module load python3/3.12.12 ml/lambda/gpu-ai ml/torch/2.6
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=~/qjudge/pylib

source ~/qjudge/scripts/pick_gpu.sh
hostname; nvidia-smi --query-gpu=name,memory.total --format=csv
python -u ~/qjudge/scripts/fact_eval.py --model ~/qjudge/models/$M --quant $Q
