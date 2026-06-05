#!/usr/bin/env bash
# SeqKD (sequence-level KD = SFT on ANLS-passing 27B-teacher CodeAct trajectories)
# for the Qwen3.5-4B DocVQA agent. FSDP engine, LoRA by default, single GPU.
#
# This is Stage-1 / loss-ladder-rung-1 of the off-policy distillation plan.
# The data is a verl MultiTurnSFTDataset parquet (single `messages` column)
# produced by docvqa/scripts/make_sft_data.py from collect_trajectories.py output.
#
# Usage:
#   recipe/docvqa/run_seqkd.sh <train_parquet> <experiment_name> [nproc] [extra hydra overrides...]
#
# Examples:
#   # in-distribution learnability probe (1 GPU)
#   recipe/docvqa/run_seqkd.sh \
#       data/sft/dv2026_probe.parquet seqkd-probe 1
#
#   # transfer run (mmlb-train -> dv2026), 2 GPUs, override lr
#   recipe/docvqa/run_seqkd.sh \
#       data/sft/mmlb_transfer.parquet seqkd-transfer 2 optim.lr=5e-5
#
# Env overrides (with defaults):
#   MODEL_PATH=Qwen/Qwen3.5-4B
#   LORA_RANK=32  LORA_ALPHA=32        (set LORA_RANK=0 for full fine-tune)
#   LR=1e-4                            (LoRA lr; use ~2e-5 for full-FT)
#   EPOCHS=3
#   TRAIN_BATCH_SIZE=16               (global; small because the SFT set is small)
#   MICRO_BATCH_SIZE_PER_GPU=1        (CodeAct trajectories are long)
#   MAX_LENGTH=32768                  (per-sample token cap; truncation=error)
#   MAX_TOKEN_LEN_PER_GPU=32768       (dynamic-bsz packing budget per GPU)
#   SAVE_FREQ=-1                      (-1 = save once at the end)
#   VAL_FILES=null                    (optional held-out parquet for loss curve)
#   LOGGER=console                    (set to "console,wandb" if wandb is configured)
set -xeuo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <train_parquet> <experiment_name> [nproc] [extra hydra overrides...]"
    exit 1
fi

TRAIN_FILES=$1
EXPERIMENT_NAME=$2
NPROC=${3:-1}
shift 2; [ "$#" -ge 1 ] && shift   # drop nproc if it was passed positionally
# (any remaining args are passed through as hydra overrides)

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.5-4B}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-32}
LR=${LR:-1e-4}
EPOCHS=${EPOCHS:-3}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-1}
MAX_LENGTH=${MAX_LENGTH:-32768}
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-32768}
SAVE_FREQ=${SAVE_FREQ:--1}
VAL_FILES=${VAL_FILES:-null}
LOGGER=${LOGGER:-console}
PROJECT_NAME=${PROJECT_NAME:-docvqa-seqkd}
SAVE_PATH=${SAVE_PATH:-checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}

# Build the python logger list literal, e.g. console -> ['console']
LOGGER_LIST="[$(echo "$LOGGER" | sed "s/,/','/g; s/^/'/; s/$/'/")]"

mkdir -p "${SAVE_PATH}"

torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
    -m verl.trainer.sft_trainer \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.messages_key=messages \
    data.ignore_input_ids_mismatch=True \
    data.pad_mode=no_padding \
    data.use_dynamic_bsz=${USE_DYNAMIC_BSZ:-False} \
    data.max_length="${MAX_LENGTH}" \
    data.max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
    data.truncation=error \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
    model.path="${MODEL_PATH}" \
    model.lora_rank="${LORA_RANK}" \
    model.lora_alpha="${LORA_ALPHA}" \
    model.target_modules=all-linear \
    model.use_remove_padding=${USE_REMOVE_PADDING:-False} \
    model.enable_gradient_checkpointing=True \
    +model.override_config.attn_implementation=${ATTN:-sdpa} \
    engine=fsdp \
    engine.strategy=fsdp2 \
    engine.ulysses_sequence_parallel_size=1 \
    optim=fsdp \
    optim.lr="${LR}" \
    optim.lr_warmup_steps_ratio=0.03 \
    optim.weight_decay=0.0 \
    optim.warmup_style=cosine \
    optim.min_lr_ratio=0.1 \
    trainer.default_local_dir="${SAVE_PATH}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.logger="${LOGGER_LIST}" \
    trainer.total_epochs="${EPOCHS}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq=-1 \
    trainer.resume_mode=auto \
    trainer.max_ckpt_to_keep=2 \
    checkpoint.save_contents=[model,optimizer,extra] \
    "$@"
