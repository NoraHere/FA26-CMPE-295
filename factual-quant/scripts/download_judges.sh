#!/bin/bash
# Download the extra judge models (run on hpc1, which has internet). Same curl approach as download.sh.
set -u
cd ~/qjudge
dl() { # repo dest
  local repo=$1 dest=$2
  mkdir -p "$dest"
  curl -s -m 60 "https://huggingface.co/api/models/$repo" | python3 -c 'import json,sys;[print(s["rfilename"]) for s in json.load(sys.stdin)["siblings"]]' |
  grep -vE '^(\.gitattributes|LICENSE.*|README.md|reward_server/.*)$' |
  while read -r f; do
    mkdir -p "$dest/$(dirname "$f")"
    curl -sSL --retry 5 -C - -o "$dest/$f" "https://huggingface.co/$repo/resolve/main/$f" && echo "ok $repo/$f" || echo "FAIL $repo/$f"
  done
}
dl sarosavo/Master-RM models/Master-RM &
dl IAAR-Shanghai/xVerify-3B-Ia models/xVerify-3B-Ia &
dl Qwen/Qwen2.5-7B-Instruct models/Qwen2.5-7B-Instruct &
dl RedHatAI/Qwen3-4B-quantized.w4a16 models/Qwen3-4B-w4a16 &
wait; echo ALL_DONE
