#!/bin/bash
#SBATCH --job-name=jb_setup
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/019111677/judgebench/logs/%x-%j.out

# One-time: install the pinned stack into ~/venvs/judgebench-gpu from ~/wheelhouse (offline).
# Wheels are fetched on the login node with `pip download` (python 3.11, cp311, cu124).
# The venv's python3.11 only exists on compute nodes, hence a job.
set -euo pipefail
cd /home/019111677/judgebench
source env.sh

echo "host: $(hostname)  job: ${SLURM_JOB_ID:-none}"
python -m pip install --no-index --find-links /home/019111677/wheelhouse \
    "torch==2.6.0" "transformers==4.57.1" "torchao==0.10.0" "numpy==1.26.4" accelerate \
    langdetect nltk immutabledict absl-py   # IFEval rule checkers; nltk data in ~/nltk_data
python - <<'EOF'
import torch, transformers, torchao
import langdetect, nltk, immutabledict, absl
nltk.word_tokenize("IFEval checkers need punkt."); nltk.sent_tokenize("One. Two.")
print("ifeval deps ok: nltk", nltk.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("transformers", transformers.__version__, "torchao", torchao.__version__)
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0), "cc", torch.cuda.get_device_capability(0))
EOF
