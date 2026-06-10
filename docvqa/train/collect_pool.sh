#!/usr/bin/env bash
# Optional: collect 27B-teacher trajectories over one pool dataset for SFT/OPD
# warmup or difficulty curation. RL-from-scratch does NOT need this.
# Usage: ./docvqa/train/collect_pool.sh <dataset-name>   (name must be in pool.yaml)
set -xeuo pipefail
cd /home/baris/repos/docvqa-verl
source .venv/bin/activate
NAME="$1"
QUESTIONS=$(.venv/bin/python -c "
import yaml
m=yaml.safe_load(open('docvqa/train/pool.yaml'))
print(next(d['questions'] for d in m['datasets'] if d['name']=='$NAME'))")
.venv/bin/python docvqa/scripts/eval.py \
  --questions "$QUESTIONS" \
  --base-url http://localhost:8927/v1 --model Qwen/Qwen3.5-27B \
  --vlm-base-url http://localhost:8927 --vlm-model Qwen/Qwen3.5-27B \
  --n 4 --temperature 0.8 --concurrency 12 --rollout-timeout 900 \
  --run-dir "outputs/runs/pool-collect-${NAME}" --resume
echo "POOL_COLLECT_DONE ${NAME}"
