#!/usr/bin/env bash
# Async (one-step-off-policy) GRPO for the Qwen3.5-4B DocVQA CodeAct agent.
#
# This is the CANONICAL RL launcher. It runs verl's experimental
# `one_step_off_policy` trainer — DISAGGREGATED async-RL (AReaL / async-RLHF style):
# dedicated rollout GPU(s) + dedicated training GPU(s), generation overlaps training
# (one step stale), NCCL weight sync, NO colocated sleep/wake. Chosen over the sync
# colocated `verl.trainer.main_ppo` because this agentic task is ROLLOUT-BOUND (the 27B
# VLM perception is the throughput gate) and the colocated path hit a sleep/wake OOM.
# It is also the substrate OPD / Pedagogical-RL build on (see RL-OPD-design.md).
#
# The 4B actor generates multi-turn agent rollouts through DocVQAReplAgentLoop (registered
# as "docvqa_repl" in docvqa/agent.yaml); each rollout's batch_look hits the frozen 27B VLM
# at :8927; reward = continuous ANLS (docvqa/reward.py:compute_score). adv_estimator=grpo
# (group-relative, no value fn). Init from base Qwen3.5-4B by default; for an SFT warm start
# set MODEL_PATH to a merged SeqKD checkpoint (more in-group reward variance early).
#
# GPU LAYOUT (4-GPU box): the 27B perception VLM is served SEPARATELY (e.g. GPUs 2,3 via
# `vllm serve ... --data-parallel-size 2 --port 8927`). This script drives the policy on the
# remaining GPUs, disaggregated: N_ROLLOUT_GPU for generation + N_TRAIN_GPU for training.
# Point CUDA_VISIBLE_DEVICES at the policy GPUs (e.g. CUDA_VISIBLE_DEVICES=0,1).
#
# Sampling: TRAIN rollouts stay at verl defaults (temp 1.0, top_p 1.0, top_k off) — on-policy
# GRPO needs the policy's own untruncated distribution (the trainer corrects log-probs for
# temperature only, not for top-p/top-k truncation). VAL mirrors deploy sampling
# (0.6/0.95/20 = configs/lm/*.yaml in ~/repos/docvqa) so any test_freq eval measures the
# policy as deployed.
#
# Run with the RL venv active (.venv — the only env with a Qwen3.5-capable vllm 0.17).
# Requires the 27B VLM up at :8927; pause any mmlb/pool collection first (shares the 27B).
# Read CLAUDE.md RL-training-practices BEFORE scaling: trainer/rollout PRECISION MATCHING is
# the dominant silent RL failure — verify it. Full async writeup: RL-async-findings.md.
set -xeuo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$REPO_DIR/outputs/hf_datasets_cache}"
mkdir -p "$HF_DATASETS_CACHE"
export WANDB_ENTITY=${WANDB_ENTITY:-bdsaglam}
# Optional best-effort JSONL dump of rollout messages/termination (debugging). Unset to disable.
export DOCVQA_TRAJ_DUMP=${DOCVQA_TRAJ_DUMP:-$REPO_DIR/outputs/async_traj.jsonl}
rm -f "$DOCVQA_TRAJ_DUMP"

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.5-4B}                       # base; SFT init = a merged_hf ckpt
# Default = the difficulty CURRICULUM (easy->hard by num_pages, datasets interleaved), built by
# docvqa/scripts/make_curriculum_parquet.py. MUST pair with SHUFFLE=False to preserve the order.
TRAIN_FILES=${TRAIN_FILES:-data/pool/curriculum_rl.parquet}
VAL_FILES=${VAL_FILES:-data/pool/curriculum_rl.parquet}
SHUFFLE=${SHUFFLE:-False}                                       # False = keep curriculum order
EXP_NAME=${EXP_NAME:-docvqa-grpo-4b-async}
PROJECT_NAME=${PROJECT_NAME:-docvqa-verl-rl}
# '|'-separated weighted endpoint pool (see docvqa/tools.py EndpointPool): local DP2 on
# :8927 + remote DP3 tunneled on :8928. Least-loaded + health-aware — a dead endpoint is
# benched and re-probed every 60s, so the remote VLM starts absorbing traffic the moment
# it comes up, mid-run, no restart needed.
VLM_BASE_URL=${VLM_BASE_URL:-'http://localhost:8927@2|http://localhost:8928@3'}
VLM_MODEL=${VLM_MODEL:-Qwen/Qwen3.5-27B}

# --- disaggregation: rollout GPU(s) + train GPU(s) (must sum to # visible policy GPUs) ---
N_ROLLOUT_GPU=${N_ROLLOUT_GPU:-1}
N_TRAIN_GPU=${N_TRAIN_GPU:-1}

# --- scale (these defaults reproduce the first successful 4B run; shrink for a dry-run) ---
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}        # prompts/step
PPO_MINI_BATCH=${PPO_MINI_BATCH:-8}
ROLLOUT_N=${ROLLOUT_N:-8}                       # GRPO group size
ROLLOUT_TP=${ROLLOUT_TP:-1}
MAX_STEPS=${MAX_STEPS:-40}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
SAVE_FREQ=${SAVE_FREQ:-10}
LR=${LR:-1e-6}
KL_COEF=${KL_COEF:-0.001}
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-16384}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-16384}     # agent trajectories are long
# gpu_memory_utilization is intentionally LOW: the fp32 log-prob logits of the longest sequence
# (~16K tok x ~248K vocab x 4B) need ~16 GB of headroom on the rollout GPU. 0.25 leaves ~20 GB
# free > worst case. (Do NOT "fix" the resulting OOM with PYTORCH_ALLOC_CONF=expandable_segments
# — it breaks the CUDA-IPC weight transfer: pidfd_getfd "Operation not permitted" -> hang.)
ROLLOUT_GPU_MEM=${ROLLOUT_GPU_MEM:-0.25}

# CUDA graphs (enforce_eager=False) are a ~3.8x per-stream decode speedup for this GDN-hybrid
# model (profiled 15.8 -> 59.9 tok/s; rollout tail 509s -> 212s) and need TWO companion fixes:
# (1) the vllm GDN-LoRA capture patch (patches/vllm-0.17-gdn-lora-cudagraph.patch.md, applied to
# .venv) — without it dummy-LoRA warmup dies with IndexError (vllm#36372); (2) max_num_seqs
# capped to actual concurrency (>= TRAIN_BATCH_SIZE*ROLLOUT_N) so capture batches never exceed
# the GDN conv-state cache lines at our low gpu_memory_utilization (assert num_cache_lines >=
# batch). Set ENFORCE_EAGER=True to fall back if either breaks after an env change.
#
# load_format MUST be non-"dummy" for Qwen3.5 (hybrid GDN/linear-attn model). With the verl
# default `load_format=dummy`, vLLM inits the rollout engine with RANDOM weights and relies on
# verl's FSDP->vLLM base-weight sync. But that sync ships the GDN `linear_attn` projections under
# SPLIT, non-LoRA-wrapped names (in_proj_qkv/z/a/b, conv1d, out_proj) while vLLM expects them
# FUSED + LoRA-wrapped (in_proj_qkvz.base_layer.weight, in_proj_ba.base_layer.weight, ...). vLLM's
# loader does not apply the split->fused + .base_layer mapping for GDN layers, so ~6/7 linear_attn
# params per hybrid layer SILENTLY skip -> stay random -> the rollout policy emits GIBBERISH ->
# every rollout fails -> zero reward. `safetensors` makes vLLM load the REAL checkpoint at init
# (GDN correct) and sets base_sync_done=True (engine_workers.py:614 `"dummy" not in load_format`),
# so verl SKIPS the broken base push and only syncs LoRA deltas (standard layers).
LOAD_FORMAT=${LOAD_FORMAT:-safetensors}
# LoRA target modules. MUST be LM-only for Qwen3.5-VL (`Qwen3_5ForConditionalGeneration`):
# `all-linear` makes PEFT wrap the VISION tower's linears too, but vLLM only LoRA-adapts the
# language model. The mismatch makes verl's base-weight sync rename vision params to
# `...base_layer.weight`, which vLLM's unwrapped vision tower lacks -> KeyError
# 'blocks.0.attn.qkv.base_layer.weight' during rollout weight-sync. LM-only also matches project
# scope (we do NOT fine-tune the VLM). Vision linears use qkv/proj/fc1/fc2 names, so this list
# cannot accidentally match them.
TARGET_MODULES=${TARGET_MODULES:-'[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]'}

python3 -m verl.experimental.one_step_off_policy.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.shuffle=${SHUFFLE} \
    data.max_prompt_length=${MAX_PROMPT_LEN} \
    data.max_response_length=${MAX_RESPONSE_LEN} \
    data.truncation=error \
    data.filter_overlong_prompts=True \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=${ENABLE_THINKING:-False} \
    custom_reward_function.path=docvqa/reward.py \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules="${TARGET_MODULES}" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${KL_COEF} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.load_format=${LOAD_FORMAT} \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.enforce_eager=${ENFORCE_EAGER:-False} \
    actor_rollout_ref.rollout.max_num_seqs=${MAX_NUM_SEQS:-64} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=docvqa/agent.yaml \
    actor_rollout_ref.rollout.agent.default_agent_loop=docvqa_repl \
    +actor_rollout_ref.rollout.agent.docvqa.vlm_base_url="${VLM_BASE_URL}" \
    +actor_rollout_ref.rollout.agent.docvqa.vlm_model_id="${VLM_MODEL}" \
    +actor_rollout_ref.rollout.agent.docvqa.parse_first_fence=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=naive \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.val_before_train=False \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=${N_TRAIN_GPU} \
    rollout.nnodes=1 \
    rollout.n_gpus_per_node=${N_ROLLOUT_GPU} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=-1 \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.total_training_steps=${MAX_STEPS} \
    trainer.resume_mode=auto \
    algorithm.rollout_correction.bypass_mode=False \
    "$@"
