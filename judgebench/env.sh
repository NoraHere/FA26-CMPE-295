source ~/venvs/judgebench-gpu/bin/activate
export PYTHONPATH=/home/019111677/venvs/judgebench-gpu/lib/python3.11/site-packages
export CUDA_HOME=/fs/sfs/local/opt/ohpc/pub/apps/nvidia/nvhpc/24.11/Linux_x86_64/24.11/compilers
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRITON_CACHE_DIR=/tmp/$USER/triton
mkdir -p "$TRITON_CACHE_DIR"
