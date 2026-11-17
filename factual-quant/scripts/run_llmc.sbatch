#!/bin/bash
#SBATCH --job-name=qjudge-llmc
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-7%4
#SBATCH --output=logs/%x-%A_%a.out
# llm-compressor W4A16 (GPTQ / AWQ) checkpoints for the 4 fact judges, then fact_eval scoring.
# Quantize with pylib_llmc on the path; score with the normal eval stack only.

RUNS=("Qwen3-4B gptq" "Qwen3-8B gptq" "Qwen2.5-7B-Instruct gptq" "Phi-4-mini-instruct gptq"
      "Qwen3-4B awq"  "Qwen3-8B awq"  "Qwen2.5-7B-Instruct awq"  "Phi-4-mini-instruct awq")
read M METHOD <<< "${RUNS[$SLURM_ARRAY_TASK_ID]}"

module purge
module load python3/3.12.12 ml/lambda/gpu-ai ml/torch/2.6
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

source ~/qjudge/scripts/pick_gpu.sh
hostname; nvidia-smi --query-gpu=name,memory.total --format=csv
set -e
PYTHONPATH=~/qjudge/pylib_llmc:~/qjudge/pylib python -u ~/qjudge/scripts/llmc_quant.py --model ~/qjudge/models/$M --method $METHOD
PYTHONPATH=~/qjudge/pylib python -u ~/qjudge/scripts/fact_eval.py --model ~/qjudge/models/$M-llmc-$METHOD-w4a16 --quant bf16
