#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/tata/tata/F/rsy/FineR1_repro}"
PYTHON="${PYTHON:-/home/tata/tata/D/anaconda/envs/finer1_tapo/bin/python}"
LLAMAFACTORY_DIR="${LLAMAFACTORY_DIR:-$REPO_ROOT/FineR1_ICLR2026_clean/LLaMA-Factory}"

INPUT_JSON="${INPUT_JSON:-$LLAMAFACTORY_DIR/data/Fine-R1-Stage1-cub-v27c-strict-abs-filtered-grounded-top2.json}"
RIGID_JSON="${RIGID_JSON:-$LLAMAFACTORY_DIR/data/Fine-R1-Stage1-cub-v27c-strict-abs-rigid-filtered-grounded-top2.json}"
DATASET_KEY="${DATASET_KEY:-Fine-R1-Stage1-cub-v27c-strict-abs-rigid-filtered-grounded-top2}"
DATASET_PREFIX="${DATASET_PREFIX:-Fine-R1-Stage1-cub-v27c-strict-abs-rigid}"

TAG="${TAG:-frozenvis_v27c_abs_rigid_top2_e8_lr5e6}"
OUTPUT_MODEL_STEM="${OUTPUT_MODEL_STEM:-qwen25vl_3b_cub_v27c_abs_rigid_frozenvis}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/eval_outputs/qwen25vl_v27c_${TAG}_eval_50}"
EVAL_JSONL="${EVAL_JSONL:-$REPO_ROOT/prompt_debug/cub_eval_50_seed20260614.jsonl}"
TEACHER_JSONL="${TEACHER_JSONL:-$REPO_ROOT/prompt_debug/cub_visual_primitive_teacher_v27c_eval50_base_object_fallback.jsonl}"

IMAGE_MIN_PIXELS="${IMAGE_MIN_PIXELS:-1024}"
IMAGE_MAX_PIXELS="${IMAGE_MAX_PIXELS:-589824}"
MIN_TRANSFORMERS_VERSION="${MIN_TRANSFORMERS_VERSION:-4.51.0}"

WAIT_FOR_GPU="${WAIT_FOR_GPU:-1}"
GPU_ID="${GPU_ID:-0}"
MIN_GPU_FREE_MB="${MIN_GPU_FREE_MB:-70000}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-60}"

EPOCHS="${EPOCHS:-8}"
LR="${LR:-5e-6}"
OPTIM="${OPTIM:-adafactor}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-768}"
FORMAT_RETRY="${FORMAT_RETRY:-1}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"

mkdir -p "$REPO_ROOT/logs" "$OUT_DIR"

for arg in "$@"; do
  case "$arg" in
    --prepare-only)
      SKIP_TRAIN=1
      SKIP_EVAL=1
      WAIT_FOR_GPU=0
      ;;
    --skip-train)
      SKIP_TRAIN=1
      ;;
    --skip-eval)
      SKIP_EVAL=1
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

log() {
  echo "[$(date '+%F %T')] $*" >&2
}

gpu_free_mb() {
  nvidia-smi --id="$GPU_ID" --query-gpu=memory.free --format=csv,noheader,nounits | awk 'NR==1 {gsub(/ /, ""); print int($1)}'
}

wait_for_gpu() {
  if [[ "$WAIT_FOR_GPU" != "1" ]]; then
    log "WAIT_FOR_GPU=$WAIT_FOR_GPU; starting without GPU wait"
    return
  fi
  while true; do
    local free_mb
    free_mb="$(gpu_free_mb || echo 0)"
    log "GPU ${GPU_ID} free memory: ${free_mb} MiB; need >= ${MIN_GPU_FREE_MB} MiB"
    if (( free_mb >= MIN_GPU_FREE_MB )); then
      return
    fi
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits || true
    sleep "$GPU_POLL_SECONDS"
  done
}

check_transformers() {
  "$PYTHON" - "$MIN_TRANSFORMERS_VERSION" <<'PY'
import re
import sys

minimum = sys.argv[1]
import transformers

def key(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", v.split("+", 1)[0])[:4])

current = transformers.__version__
print(f"transformers={current}; required>={minimum}", flush=True)
if key(current) < key(minimum):
    raise SystemExit(f"transformers {current} < {minimum}; aborting Qwen2.5-VL grounding run")
PY
}

build_rigid_dataset() {
  if [[ "$FORCE_REBUILD" == "1" || ! -f "$RIGID_JSON" ]]; then
    log "rebuilding rigid box-top2 SFT data"
    "$PYTHON" "$REPO_ROOT/scripts/rebuild_grounded_rigid.py" \
      --in "$INPUT_JSON" \
      --out "$RIGID_JSON" \
      --target-k 2 \
      --min-k 1
  else
    log "using existing rigid SFT data: $RIGID_JSON"
  fi

  log "registering LLaMA-Factory dataset: $DATASET_KEY"
  "$PYTHON" - "$LLAMAFACTORY_DIR/data/dataset_info.json" "$DATASET_KEY" "$(basename "$RIGID_JSON")" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
key = sys.argv[2]
file_name = sys.argv[3]
data = json.loads(path.read_text(encoding="utf-8"))
data[key] = {
    "formatting": "sharegpt",
    "columns": {"messages": "messages", "images": "images"},
    "tags": {
        "role_tag": "role",
        "content_tag": "content",
        "user_tag": "user",
        "assistant_tag": "assistant",
        "system_tag": "system",
    },
    "file_name": file_name,
}
path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({key: data[key]}, ensure_ascii=False, indent=2))
PY
}

build_object_fallback_if_missing() {
  if [[ -f "$TEACHER_JSONL" ]]; then
    log "using eval teacher/object grounding targets: $TEACHER_JSONL"
    return
  fi
  log "building eval-50 object-level fallback grounding targets"
  wait_for_gpu
  "$PYTHON" "$REPO_ROOT/scripts/build_eval50_base_object_targets_v27c.py" \
    --eval-jsonl "$EVAL_JSONL" \
    --output-jsonl "$TEACHER_JSONL" \
    --summary "${TEACHER_JSONL%.jsonl}_summary.json" \
    --image-min-pixels "$IMAGE_MIN_PIXELS" \
    --image-max-pixels "$IMAGE_MAX_PIXELS"
}

run_train() {
  if [[ "$SKIP_TRAIN" == "1" ]]; then
    log "SKIP_TRAIN=1; skipping SFT"
    return
  fi
  wait_for_gpu
  log "starting rigid top2 retention SFT: tag=$TAG"
  (
    cd "$REPO_ROOT"
    env \
      RUN_VARIANTS=top2 \
      RUN_MULTITURN_AFTER_SINGLE=false \
      DATASET_PREFIX="$DATASET_PREFIX" \
      OUTPUT_MODEL_STEM="$OUTPUT_MODEL_STEM" \
      OUTPUT_TAG="$TAG" \
      EPOCHS="$EPOCHS" \
      LR="$LR" \
      OPTIM="$OPTIM" \
      BATCH_SIZE="$BATCH_SIZE" \
      GRAD_ACCUM="$GRAD_ACCUM" \
      SAVE_STEPS="$SAVE_STEPS" \
      SAVE_TOTAL_LIMIT="$SAVE_TOTAL_LIMIT" \
      FREEZE_VISION_TOWER=true \
      FREEZE_MULTI_MODAL_PROJECTOR=true \
      FREEZE_LANGUAGE_MODEL=false \
      IMAGE_MIN_PIXELS="$IMAGE_MIN_PIXELS" \
      IMAGE_MAX_PIXELS="$IMAGE_MAX_PIXELS" \
      bash run_sft_qwen25vl_v27_fullft.sh
  ) 2>&1 | tee "$REPO_ROOT/logs/${TAG}.sft.log"
}

run_eval() {
  if [[ "$SKIP_EVAL" == "1" ]]; then
    log "SKIP_EVAL=1; skipping eval"
    return
  fi
  wait_for_gpu
  log "starting eval-50 for rigid top2 retention checkpoint"
  (
    cd "$REPO_ROOT"
    env \
      RUN_VARIANTS=top2 \
      RUN_MULTITURN_AFTER_SINGLE=false \
      RUN_MULTITURN_DIALOG_EVAL=false \
      OUTPUT_MODEL_STEM="$OUTPUT_MODEL_STEM" \
      TAG="$TAG" \
      OUT_DIR="$OUT_DIR" \
      EVAL_JSONL="$EVAL_JSONL" \
      TEACHER_JSONL="$TEACHER_JSONL" \
      EVAL_NAME_VERSION=v27c \
      PROMPT_STYLE=grounded_eval \
      FORMAT_RETRY="$FORMAT_RETRY" \
      MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
      SKIP_BASE_EVAL=1 \
      RUN_STAGE2_GATE=false \
      bash run_eval_qwen25vl_v27_fullft_tag.sh
  ) 2>&1 | tee "$REPO_ROOT/logs/${TAG}.eval50.log"

  local pred="$OUT_DIR/filtered_grounded_top2_v27c_${TAG}.jsonl"
  if [[ -f "$pred" ]]; then
    log "visualizing eval-50 student boxes"
    "$PYTHON" "$REPO_ROOT/scripts/visualize_eval_ref_boxes.py" \
      --jsonl "$pred" \
      --output-dir "$OUT_DIR/vis_filtered_grounded_top2" \
      --summary "$OUT_DIR/vis_filtered_grounded_top2_summary.json" \
      --contact-sheet "$OUT_DIR/vis_filtered_grounded_top2_contact_sheet.jpg" \
      --max-records 50 \
      --coordinate-system llamafactory_qwen25_abs \
      --image-min-pixels "$IMAGE_MIN_PIXELS" \
      --image-max-pixels "$IMAGE_MAX_PIXELS" \
      --parse-loose
  else
    log "prediction JSONL not found for visualization: $pred"
  fi
}

main() {
  cd "$REPO_ROOT"
  check_transformers
  build_rigid_dataset
  build_object_fallback_if_missing
  run_train
  run_eval
  log "rigid top2 retention pipeline complete"
  log "model: $LLAMAFACTORY_DIR/saves/${OUTPUT_MODEL_STEM}_filtered_grounded_top2_${TAG}"
  log "eval:  $OUT_DIR"
}

main "$@"
