#!/bin/bash
#SBATCH --job-name=qjudge-gptaq
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%A_%a.out
# GPTAQ-style asymmetric calibration (scripts/gptaq.py). Usage:
#   sbatch --array=0-1 run_gptaq.sbatch tune      Qwen3-4B, two correction settings, JudgeBench only (settings check)
#   sbatch --array=0-3 run_gptaq.sbatch full A C  4 judges, all 3 datasets, alpha A, cdamp C

MODE=$1; JUDGES=("Qwen3-4B" "Qwen3-8B" "Qwen2.5-7B-Instruct" "Phi-4-mini-instruct")
module purge
module load python3/3.12.12 ml/lambda/gpu-ai ml/torch/2.6
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=~/qjudge/pylib
source ~/qjudge/scripts/pick_gpu.sh
hostname
if [ "$MODE" = tune ]; then
  CFG=("0.25 0.1 tuneA" "0.25 1.0 tuneB"); read A C S <<< "${CFG[$SLURM_ARRAY_TASK_ID]}"
  python -u ~/qjudge/scripts/fact_eval.py --model ~/qjudge/models/Qwen3-4B --quant gptaq4 --only jb \
         --gptaq_alpha $A --gptaq_cdamp $C --suffix=-$S
else
  python -u ~/qjudge/scripts/fact_eval.py --model ~/qjudge/models/${JUDGES[$SLURM_ARRAY_TASK_ID]} --quant gptaq4 \
         --gptaq_alpha $2 --gptaq_cdamp $3
fi
