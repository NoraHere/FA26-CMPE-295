#!/bin/bash
#SBATCH --job-name=qjudge-gptq
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-8%4
#SBATCH --output=logs/%x-%A_%a.out
# 4-bit on the factual sets (scripts/fact_eval.py):
#   gptq4 / rtn4 = team simquant, fake-quant in BF16 (same grid; only GPTQ error compensation differs)
#   Qwen3-4B-w4a16 = Red Hat GPTQ W4A16 checkpoint, real compressed-tensors kernel (loads with --quant bf16)

RUNS=("Qwen3-4B gptq4" "Qwen3-8B gptq4" "Qwen2.5-7B-Instruct gptq4" "Phi-4-mini-instruct gptq4"
      "Qwen3-4B rtn4"  "Qwen3-8B rtn4"  "Qwen2.5-7B-Instruct rtn4"  "Phi-4-mini-instruct rtn4"
      "Qwen3-4B-w4a16 bf16")
read M Q <<< "${RUNS[$SLURM_ARRAY_TASK_ID]}"

module purge
module load python3/3.12.12 ml/lambda/gpu-ai ml/torch/2.6
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=~/qjudge/pylib

source ~/qjudge/scripts/pick_gpu.sh
hostname; nvidia-smi --query-gpu=name,memory.total --format=csv
python -u ~/qjudge/scripts/fact_eval.py --model ~/qjudge/models/$M --quant $Q
