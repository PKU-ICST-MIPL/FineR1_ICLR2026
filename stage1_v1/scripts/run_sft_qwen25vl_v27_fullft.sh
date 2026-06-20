#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/tata/tata/F/rsy/FineR1_repro}"
LLAMAFACTORY_DIR="${LLAMAFACTORY_DIR:-$REPO_ROOT/FineR1_ICLR2026_clean/LLaMA-Factory}"
PYTHON_ENV_BIN="${PYTHON_ENV_BIN:-/home/tata/tata/D/anaconda/envs/finer1_tapo/bin}"
MODEL_PATH="${MODEL_PATH:-/home/tata/tata/models/Qwen2.5-VL-3B-Instruct}"
CUSPARSELT_LIB="${CUSPARSELT_LIB:-/home/tata/tata/D/anaconda/envs/finer1_sft/lib/python3.10/site-packages/nvidia/cusparselt/lib}"

EPOCHS="${EPOCHS:-3}"
MAX_STEPS="${MAX_STEPS:-}"
LR="${LR:-1e-6}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
BF16="${BF16:-true}"
OPTIM="${OPTIM:-adamw_torch}"
RUN_VARIANTS="${RUN_VARIANTS:-plain,top2,top3}"
RUN_MULTITURN_AFTER_SINGLE="${RUN_MULTITURN_AFTER_SINGLE:-true}"
OUTPUT_TAG="${OUTPUT_TAG:-fullft_v27_e${EPOCHS}}"
DATASET_PREFIX="${DATASET_PREFIX:-Fine-R1-Stage1-cub-v27-full}"
OUTPUT_MODEL_STEM="${OUTPUT_MODEL_STEM:-qwen25vl_3b_cub_v27_full}"
FREEZE_VISION_TOWER="${FREEZE_VISION_TOWER:-false}"
FREEZE_MULTI_MODAL_PROJECTOR="${FREEZE_MULTI_MODAL_PROJECTOR:-false}"
FREEZE_LANGUAGE_MODEL="${FREEZE_LANGUAGE_MODEL:-false}"
IMAGE_MAX_PIXELS="${IMAGE_MAX_PIXELS:-589824}"
IMAGE_MIN_PIXELS="${IMAGE_MIN_PIXELS:-1024}"

if [[ -d "$CUSPARSELT_LIB" ]]; then
  export LD_LIBRARY_PATH="$CUSPARSELT_LIB:${LD_LIBRARY_PATH:-}"
fi
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$LLAMAFACTORY_DIR"

run_full_sft() {
  local dataset="$1"
  local output_dir="$2"
  local extra_steps=()

  if [[ -n "$MAX_STEPS" ]]; then
    extra_steps=(--max_steps "$MAX_STEPS")
  else
    extra_steps=(--num_train_epochs "$EPOCHS")
  fi

  echo "==== v27 full SFT dataset=$dataset output=$output_dir"
  "$PYTHON_ENV_BIN/llamafactory-cli" train \
    --stage sft \
    --do_train true \
    --model_name_or_path "$MODEL_PATH" \
    --dataset "$dataset" \
    --dataset_dir data \
    --template qwen2_vl \
    --finetuning_type full \
    --output_dir "$output_dir" \
    --overwrite_output_dir true \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate "$LR" \
    --optim "$OPTIM" \
    "${extra_steps[@]}" \
    --logging_steps "$LOGGING_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --bf16 "$BF16" \
    --report_to none \
    --plot_loss true \
    --gradient_checkpointing true \
    --freeze_vision_tower "$FREEZE_VISION_TOWER" \
    --freeze_multi_modal_projector "$FREEZE_MULTI_MODAL_PROJECTOR" \
    --freeze_language_model "$FREEZE_LANGUAGE_MODEL" \
    --image_max_pixels "$IMAGE_MAX_PIXELS" \
    --image_min_pixels "$IMAGE_MIN_PIXELS"
}

IFS="," read -r -a variants <<< "$RUN_VARIANTS"
for variant in "${variants[@]}"; do
  case "$variant" in
    plain)
      run_full_sft "${DATASET_PREFIX}-plain-cot" "saves/${OUTPUT_MODEL_STEM}_plain_cot_${OUTPUT_TAG}"
      ;;
    top2)
      run_full_sft "${DATASET_PREFIX}-filtered-grounded-top2" "saves/${OUTPUT_MODEL_STEM}_filtered_grounded_top2_${OUTPUT_TAG}"
      if [[ "$RUN_MULTITURN_AFTER_SINGLE" == "true" ]]; then
        run_full_sft "${DATASET_PREFIX}-multiturn-grounded-top2" "saves/${OUTPUT_MODEL_STEM}_multiturn_grounded_top2_${OUTPUT_TAG}"
      fi
      ;;
    top3)
      run_full_sft "${DATASET_PREFIX}-filtered-grounded-top3" "saves/${OUTPUT_MODEL_STEM}_filtered_grounded_top3_${OUTPUT_TAG}"
      if [[ "$RUN_MULTITURN_AFTER_SINGLE" == "true" ]]; then
        run_full_sft "${DATASET_PREFIX}-multiturn-grounded-top3" "saves/${OUTPUT_MODEL_STEM}_multiturn_grounded_top3_${OUTPUT_TAG}"
      fi
      ;;
    mtop2)
      run_full_sft "${DATASET_PREFIX}-multiturn-grounded-top2" "saves/${OUTPUT_MODEL_STEM}_multiturn_grounded_top2_${OUTPUT_TAG}"
      ;;
    mtop3)
      run_full_sft "${DATASET_PREFIX}-multiturn-grounded-top3" "saves/${OUTPUT_MODEL_STEM}_multiturn_grounded_top3_${OUTPUT_TAG}"
      ;;
    *)
      echo "Unknown RUN_VARIANTS item: $variant" >&2
      exit 2
      ;;
  esac
done

echo "==== v27 full SFT done"
