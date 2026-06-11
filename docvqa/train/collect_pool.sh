#!/usr/bin/env bash
# Optional: collect 27B-teacher trajectories over one pool dataset for SFT/OPD
# warmup or difficulty curation. RL-from-scratch does NOT need this.
# Collects over the BALANCED sample (data/pool/sampled/<name>.json) produced by
# sample_pool.py — run that first. Falls back to the full questions.json (with a
# warning) if no sample exists.
# Usage: ./docvqa/train/collect_pool.sh <dataset-name>   (name must be in pool.yaml)
set -xeuo pipefail
cd /home/baris/repos/docvqa-verl
source .venv/bin/activate
NAME="$1"
SAMPLED="data/pool/sampled/${NAME}.json"
if [ -f "$SAMPLED" ]; then
  QUESTIONS="$SAMPLED"
else
  echo "WARN: $SAMPLED not found (run sample_pool.py); falling back to full split" >&2
  QUESTIONS=$(NAME="$NAME" .venv/bin/python -c "
import os, yaml
m=yaml.safe_load(open('docvqa/train/pool.yaml'))
print(next(d['questions'] for d in m['datasets'] if d['name']==os.environ['NAME']))")
fi
.venv/bin/python docvqa/scripts/eval.py \
  --questions "$QUESTIONS" \
  --base-url http://localhost:8927/v1 --model Qwen/Qwen3.5-27B \
  --vlm-base-url http://localhost:8927 --vlm-model Qwen/Qwen3.5-27B \
  --n 4 --temperature 0.8 --concurrency 12 --rollout-timeout 900 \
  --run-dir "outputs/runs/pool-collect-${NAME}" --resume
echo "POOL_COLLECT_DONE ${NAME}"
