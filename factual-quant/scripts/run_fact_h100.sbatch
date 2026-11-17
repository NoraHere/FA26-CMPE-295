#!/bin/bash
#SBATCH --job-name=qjudge-fact-h100
#SBATCH --partition=gpuqs,condo
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-7%3
#SBATCH --output=logs/%x-%A_%a.out
# FP8 / INT8 W8A8 vs BF16 / INT8 weight-only, all on H100 so throughput is comparable.
# Results go to results/<model>[-quant]-h100 and do not overwrite the A100 runs.

RUNS=("Qwen3-8B bf16" "Qwen3-8B int8" "Qwen3-8B fp8" "Qwen3-8B w8a8"
      "Phi-4-mini-instruct bf16" "Phi-4-mini-instruct int8" "Phi-4-mini-instruct fp8" "Phi-4-mini-instruct w8a8")
read M Q <<< "${RUNS[$SLURM_ARRAY_TASK_ID]}"

module purge
module load python3/3.12.12 ml/lambda/gpu-ai ml/torch/2.6
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=~/qjudge/pylib

hostname; nvidia-smi --query-gpu=name,memory.total --format=csv
python -u ~/qjudge/scripts/fact_eval.py --model ~/qjudge/models/$M --quant $Q --suffix=-h100
