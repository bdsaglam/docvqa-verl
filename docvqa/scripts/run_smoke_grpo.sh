#!/usr/bin/env bash
# Phase-1 smoke training for DocVQA. GRPO + LoRA, FSDP training, vLLM rollout.
# Modeled on examples/tuning/lora/run_qwen3_8b_fsdp.sh, customized for our
# multi-turn agent loop and the DocVQA-2026 val-train split.
#
# Verl spins up its own student vLLM internally for rollouts. The ONLY
# external prerequisite is the frozen 27B VLM at $DOCVQA_VLM_BASE_URL
# (default http://localhost:8928), which the agent's batch_look() tool
# hits via HTTP.
#
# trainer.val_before_train=True (verl default) gives a Layer-3 baseline at
# step 0, before any gradient update. Set trainer.val_only=true to skip
# training entirely and just emit baseline eval.
#
# Usage:
#   bash docvqa/scripts/run_smoke_grpo.sh                        # smoke training
#   bash docvqa/scripts/run_smoke_grpo.sh trainer.val_only=true  # baseline eval only

set -xeuo pipefail

cd "$(dirname "$0")/../.."

# ---- external services -----------------------------------------------------
export DOCVQA_VLM_BASE_URL=${DOCVQA_VLM_BASE_URL:-http://localhost:8928}
export DOCVQA_VLM_MODEL_ID=${DOCVQA_VLM_MODEL_ID:-Qwen/Qwen3.5-27B}

# ---- compute layout --------------------------------------------------------
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-2}            # GPUs 0,1 for student; GPU 2 holds the VLM
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

# ---- model / LoRA ----------------------------------------------------------
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.5-4B}
LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-32}

# ---- data ------------------------------------------------------------------
TRAIN_FILE=${TRAIN_FILE:-data/docvqa-2026/val/train.json}
VAL_FILE=${VAL_FILE:-data/docvqa-2026/val/heldout.json}
if [[ ! -f "$TRAIN_FILE" ]]; then
    echo "ERROR: $TRAIN_FILE missing. Run docvqa/scripts/prepare_data.py first." >&2
    exit 1
fi

# ---- batch / context sizes -------------------------------------------------
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}       # tiny for smoke
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-16384}  # generous headroom for the rvlm_minimal system + first-user prompt
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}  # multi-turn aggregate; capped to keep actor-update activations within 80GB
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}  # must be >= max_prompt+max_response (16384+16384). Smaller cap halves peak activation memory in actor backward.

ROLLOUT_TP=${ROLLOUT_TP:-2}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.45}  # leave more for FSDP grads/activations
ROLLOUT_N=${ROLLOUT_N:-4}                      # GRPO group size

ACTOR_LR=${ACTOR_LR:-3e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}

PROJECT_NAME=${PROJECT_NAME:-docvqa-verl}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-smoke-phase1-grpo}

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="$TRAIN_FILE"
    data.val_files="$VAL_FILE"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation='error'
    +data.apply_chat_template_kwargs.enable_thinking=true
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.lora_rank=${LORA_RANK}
    actor_rollout_ref.model.lora_alpha=${LORA_ALPHA}
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.layered_summon=True
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.rollout.prompt_length=${MAX_PROMPT_LENGTH}
    actor_rollout_ref.rollout.response_length=${MAX_RESPONSE_LENGTH}
    actor_rollout_ref.rollout.agent.agent_loop_config_path=docvqa/agent.yaml
    actor_rollout_ref.rollout.agent.default_agent_loop=docvqa_repl
    actor_rollout_ref.rollout.multi_stage_wake_up=True  # gentler vLLM wake-up after actor update; default False crashes us
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

REWARD=(
    custom_reward_function.path=docvqa/reward.py
    custom_reward_function.name=compute_score
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console","wandb"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=50
    trainer.test_freq=50
    trainer.total_epochs=1
    trainer.log_val_generations=20
    trainer.rollout_data_dir="outputs/${EXPERIMENT_NAME}/rollouts"
)

python -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "$@"
