#!/bin/bash
# Download HF repos with curl (hpc1 has internet; hpc3 does not, home is shared)
set -u
dl() { # kind repo dest
  local kind=$1 repo=$2 dest=$3 prefix=""
  [ "$kind" = datasets ] && prefix="datasets/"
  mkdir -p "$dest"
  curl -s -m 60 "https://huggingface.co/api/$kind/$repo" | python3 -c 'import json,sys;[print(s["rfilename"]) for s in json.load(sys.stdin)["siblings"]]' |
  grep -vE '^(\.gitattributes|LICENSE.*|README.md|.*\.ipynb|raw/.*|scripts/.*|\.gitignore)$' |
  while read -r f; do
    mkdir -p "$dest/$(dirname "$f")"
    curl -sSL --retry 5 -C - -o "$dest/$f" "https://huggingface.co/${prefix}${repo}/resolve/main/$f" && echo "ok $repo/$f" || echo "FAIL $repo/$f"
  done
}
dl datasets ScalerLab/JudgeBench data/JudgeBench &
dl datasets THU-KEG/RM-Bench data/RM-Bench &
dl datasets allenai/reward-bench data/reward-bench &
for m in Qwen3-1.7B Qwen3-4B Qwen3-8B; do dl models Qwen/$m models/$m & done
wait; echo ALL_DONE
