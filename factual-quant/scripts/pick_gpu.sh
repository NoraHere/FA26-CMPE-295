# source me: pick a working allocated GPU, preferring non-zero indices (cs002 GPU 0 fails intermittently with "busy or unavailable")
_c=""; for d in ${CUDA_VISIBLE_DEVICES//,/ }; do [ "$d" != 0 ] && _c="$_c $d"; done
for d in $_c ${CUDA_VISIBLE_DEVICES//,/ }; do
  if CUDA_VISIBLE_DEVICES=$d python -c "import torch; torch.cuda.mem_get_info(0); torch.zeros(1,device='cuda')" 2>/dev/null; then export CUDA_VISIBLE_DEVICES=$d; break; fi
done
echo "using GPU $CUDA_VISIBLE_DEVICES"
