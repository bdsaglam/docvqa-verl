#!/usr/bin/env bash
# Optional (SFT/OPD only): build the combined trajectory parquet from collection runs.
# RL-from-scratch skips this. Run AFTER collect_pool.sh for the datasets you collected.
set -xeuo pipefail
cd /home/baris/repos/docvqa-verl
source .venv/bin/activate
mkdir -p data/sft
for NAME in "$@"; do
  RUN="outputs/runs/pool-collect-${NAME}"
  [ -d "$RUN" ] || { echo "skip ${NAME} (no run dir ${RUN})"; continue; }
  .venv/bin/python docvqa/scripts/make_clean_sft.py \
    --in "$RUN" --out "data/sft/pool_${NAME}.parquet" --max-per-question 3
done
.venv/bin/python -c "
import pandas as pd, glob
parts=[pd.read_parquet(p) for p in glob.glob('data/sft/pool_*.parquet')]
assert parts, 'no per-dataset parquets found'
df=pd.concat(parts, ignore_index=True)
assert list(df.columns)==['messages'], df.columns
df.to_parquet('data/sft/pool_combined.parquet')
print('combined rows:', len(df))"
echo "POOL_PARQUET_DONE"
