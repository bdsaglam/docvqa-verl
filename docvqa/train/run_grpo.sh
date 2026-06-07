#!/usr/bin/env bash
# GRPO for the Qwen3.5-4B DocVQA CodeAct agent (online TITO rollouts via our agent_loop).
#
# Rung after SeqKD. Online GRPO: the 4B actor generates multi-turn agent rollouts through
# DocVQAReplAgentLoop (registered as "docvqa_repl" in docvqa/agent.yaml); each rollout's
# batch_look hits the frozen 27B VLM at :8927; reward = continuous ANLS
# (docvqa/reward.py:compute_score). adv_estimator=grpo (group-relative, no value fn).
#
# Init actor from the v1 SeqKD model (warm start → some in-group reward variance, avoids
# the zero-variance GRPO cold start).
#
# DEFAULTS BELOW ARE A TINY DRY-RUN. Scale up via env vars once the dry-run is clean.
# Run with `.venv-rl` active (the only env with a verl-compatible vllm). Requires the 27B VLM up at :8927. Pause mmlb collection first
# (shares the 27B). Read CLAUDE.md RL-training-practices BEFORE the real (scaled) run:
# trainer/rollout PRECISION MATCHING is the dominant silent RL failure — verify it.
set -xeuo pipefail

MODEL_PATH=${MODEL_PATH:-checkpoints/docvqa-verl/seqkd-transfer-mp/merged_hf}   # v1 SFT init
TRAIN_FILES=${TRAIN_FILES:-data/docvqa-2026/val/train_rl.parquet}              # 56 Q in-dist
VAL_FILES=${VAL_FILES:-data/docvqa-2026/val/heldout_rl.parquet}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-docvqa-grpo-dryrun}
PROJECT_NAME=${PROJECT_NAME:-docvqa-verl}
VLM_BASE_URL=${VLM_BASE_URL:-http://localhost:8927}
VLM_MODEL=${VLM_MODEL:-Qwen/Qwen3.5-27B}

# --- dry-run scale (override for the real run) ---
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}        # # prompts/step (real: 32-64)
PPO_MINI_BATCH=${PPO_MINI_BATCH:-4}
ROLLOUT_N=${ROLLOUT_N:-4}                       # GRPO group size (real: 8)
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
MAX_STEPS=${MAX_STEPS:-2}                        # dry-run: stop after 2 steps
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-16384}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-16384}     # agent trajectories are long; data caps ~12.5K
LR=${LR:-1e-6}
KL_COEF=${KL_COEF:-0.001}
ROLLOUT_GPU_MEM=${ROLLOUT_GPU_MEM:-0.45}        # actor+ref+rollout colocate on 1 GPU
NGPUS=${NGPUS:-1}
ROLLOUT_TIMEOUT=${ROLLOUT_TIMEOUT:-1200}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LEN} \
    data.max_response_length=${MAX_RESPONSE_LEN} \
    data.truncation=error \
    data.filter_overlong_prompts=True \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=True \
    custom_reward_function.path=docvqa/reward.py \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${KL_COEF} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=docvqa/agent.yaml \
    actor_rollout_ref.rollout.agent.default_agent_loop=docvqa_repl \
    +actor_rollout_ref.rollout.agent.docvqa.vlm_base_url="${VLM_BASE_URL}" \
    +actor_rollout_ref.rollout.agent.docvqa.vlm_model_id="${VLM_MODEL}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=naive \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=${NGPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.total_training_steps=${MAX_STEPS} \
    "$@"
