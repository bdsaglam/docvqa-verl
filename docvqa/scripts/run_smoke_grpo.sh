#!/usr/bin/env bash
# Phase-1 smoke training: ~100 GRPO steps on data/docvqa-2026/val/train.json.
#
# Verl spins up its own student vLLM internally for rollouts. The ONLY
# external prerequisite is the frozen 27B VLM at $DOCVQA_VLM_BASE_URL
# (default http://localhost:8928), which the agent's batch_look() tool
# hits via HTTP. Bring it up in tmux session `vllm` on the GPU reserved
# for it before running this script.
#
# Data: prepare_data.py must have been run already (RLHFDataset reads the
# emitted JSON files natively; multiple files are concatenated).
#
# trainer.val_before_train=True (verl default) gives a Layer-3 baseline at
# step 0, before any gradient update. Set trainer.val_only=true to skip
# training entirely and just emit baseline eval.
#
# Usage:
#   bash docvqa/scripts/run_smoke_grpo.sh                        # smoke training
#   bash docvqa/scripts/run_smoke_grpo.sh trainer.val_only=true  # baseline eval only

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
