#!/bin/bash
#SBATCH --job-name=qjudge-hqq
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-3%4
#SBATCH --output=logs/%x-%A_%a.out
# HQQ 4-bit (group 128, calibration-free) on the 4 fact judges, via transformers HqqConfig.

RUNS=("Qwen3-4B" "Qwen3-8B" "Qwen2.5-7B-Instruct" "Phi-4-mini-instruct")
M=${RUNS[$SLURM_ARRAY_TASK_ID]}

module purge
module load python3/3.12.12 ml/lambda/gpu-ai ml/torch/2.6
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=~/qjudge/pylib_hqq:~/qjudge/pylib

source ~/qjudge/scripts/pick_gpu.sh
hostname; nvidia-smi --query-gpu=name,memory.total --format=csv
python -u ~/qjudge/scripts/fact_eval.py --model ~/qjudge/models/$M --quant hqq4
