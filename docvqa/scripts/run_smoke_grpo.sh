#!/usr/bin/env bash
# Layer-4 smoke training: ~100 GRPO steps on a 200-question slice of train.
#
# Prerequisites (NOT handled by this script):
#   1. The frozen 27B VLM is serving at $DOCVQA_VLM_BASE_URL (default
#      http://localhost:8928). Bring it up in tmux session `vllm` on the
#      GPU you've reserved for it.
#   2. Training/eval JSON files exist with the verl-compatible schema
#      that prepare_data.py emits (record_id, prompt, data_source,
#      reward_model, extra_info + our own question/doc_dir/category fields).
#      RLHFDataset reads JSON natively and concatenates a list of files.
#      DocVQA-2026 has no public train split; we train on the val split's
#      train.json and use heldout.json (1 doc/category) for eval.
#      Other DocVQA-family corpora can be appended to TRAIN_FILES later.
#
# Usage:
#   bash docvqa/scripts/run_smoke_grpo.sh
#   bash docvqa/scripts/run_smoke_grpo.sh trainer.total_epochs=2  # extra overrides

set -euo pipefail

# Repo root: this script lives at <repo>/docvqa/scripts/, so go up two levels.
cd "$(dirname "$0")/../.."

export DOCVQA_VLM_BASE_URL=${DOCVQA_VLM_BASE_URL:-http://localhost:8928}
export DOCVQA_VLM_MODEL_ID=${DOCVQA_VLM_MODEL_ID:-qwen3.6-27b}

STUDENT_MODEL=${STUDENT_MODEL:-willcb/Qwen3-8B}
TRAIN_FILE=${TRAIN_FILE:-data/docvqa-2026/val/train.json}
VAL_FILE=${VAL_FILE:-data/docvqa-2026/val/heldout.json}

if [[ ! -f "$TRAIN_FILE" ]]; then
    echo "ERROR: $TRAIN_FILE missing. Run docvqa/scripts/prepare_data.py first." >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1} python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path="$STUDENT_MODEL" \
    actor_rollout_ref.actor.lora_rank=16 \
    actor_rollout_ref.actor.lora_alpha=32 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.prompt_length=16384 \
    actor_rollout_ref.rollout.response_length=32768 \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=docvqa/agent.yaml \
    actor_rollout_ref.rollout.agent.default_agent_loop=docvqa_repl \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.apply_chat_template_kwargs.enable_thinking=true \
    custom_reward_function.path=docvqa/reward.py \
    custom_reward_function.name=compute_score \
    trainer.total_epochs=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.rollout_data_dir='${hydra:runtime.output_dir}/rollouts' \
    trainer.log_val_generations=20 \
    trainer.project_name=docvqa-verl \
    trainer.experiment_name=smoke-phase1-grpo \
    "$@"
