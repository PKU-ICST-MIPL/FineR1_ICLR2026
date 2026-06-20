#!/usr/bin/env bash
# ===========================================================================
# Controlled Stage2 ladder: CoT_SFT | DAPO | Intra | Inter | TAPO
# One init, one Stage2 dataset, one reward/parser/prompt, one rollout/update
# budget, one seed, one checkpoint-selection rule. Only the algorithm toggles
# in the switch table below change between rows.
#
# Usage:  bash run_stage2_variant.sh VARIANT [--smoke] [--dry-run]
#   VARIANT in {cot_sft, dapo, intra, inter, tapo}
# ===========================================================================
set -euo pipefail
VARIANT="${1:-}"; shift || true
if [[ -z "$VARIANT" ]]; then echo "VARIANT required: cot_sft|dapo|intra|inter|tapo" >&2; exit 2; fi
SMOKE=0; DRY=0
for a in "$@"; do case "$a" in --smoke) SMOKE=1;; --dry-run) DRY=1;; *) echo "bad arg $a">&2; exit 2;; esac; done

TAPO_DIR="${TAPO_DIR:-/home/tata/tata/F/rsy/FineR1_repro/FineR1_ICLR2026_clean/TAPO}"
PY="${PY:-/home/tata/tata/D/anaconda/envs/finer1_tapo/bin/python}"
CUSPARSELT_LIB="${CUSPARSELT_LIB:-/home/tata/tata/D/anaconda/envs/finer1_sft/lib/python3.10/site-packages/nvidia/cusparselt/lib}"
STAGE1_INIT="${STAGE1_INIT:?path to the SINGLE shared grounded Stage1 checkpoint}"
CONFIG="${CONFIG:-examples/configs/config_tapo.yaml}"
TRAIN_FILE="${TRAIN_FILE:-data/Fine-R1-Stage2-data/data}"
VAL_FILE="${VAL_FILE:-data/Fine-R1-Stage2-data/data}"   # OVERRIDE: released YAML val_files is PAPO math!
FORMAT_PROMPT="${FORMAT_PROMPT:-examples/format_prompt/cls_thinking.jinja}"
PROMPT_MODE="${PROMPT_MODE:-generic}"
INIT_LABEL="${INIT_LABEL:-grounded}"
# Primary fair rows MUST use the released reward. The grounding ablation swaps it.
REWARD="${REWARD:-examples/reward_function/cls_thinking.py:compute_score}"

# ----- resolution: pin to the SAME pixels as Stage1 SFT (589824/1024) ------
# The released config uses max_pixels=1003520/min=200704, which DIFFERS from the
# Stage1 SFT canvas and miscalibrates resized-absolute grounding coords. Pin both.
MAX_PIXELS="${MAX_PIXELS:-589824}"
MIN_PIXELS="${MIN_PIXELS:-1024}"

# ----- shared budget / seed / selection ------------------------------------
SEED="${SEED:-1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-10}"
ROLLOUT_BS="${ROLLOUT_BS:-384}"
MINI_ROLLOUT_BS="${MINI_ROLLOUT_BS:-128}"
GLOBAL_BS="${GLOBAL_BS:-128}"
ROLLOUT_N="${ROLLOUT_N:-5}"
TEMP="${TEMP:-1.0}"
SAVE_FREQ="${SAVE_FREQ:-5}"
N_GPU="${N_GPU:-4}"
TP="${TP:-1}"
CUDA_IDS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export LD_LIBRARY_PATH="$CUSPARSELT_LIB:${LD_LIBRARY_PATH:-}"

case "$PROMPT_MODE" in
  generic) FORMAT_PROMPT="${FORMAT_PROMPT_GENERIC:-examples/format_prompt/cls_thinking.jinja}" ;;
  grounded) FORMAT_PROMPT="${FORMAT_PROMPT_GROUNDED:-examples/format_prompt/cls_thinking_grounded.jinja}" ;;
  *) echo "PROMPT_MODE must be generic or grounded, got: $PROMPT_MODE" >&2; exit 2 ;;
esac

# ----- single-GPU overrides (set SINGLE_GPU=1) -----------------------------
if [[ "${SINGLE_GPU:-0}" == "1" ]]; then
  N_GPU=1; TP=1; ROLLOUT_BS="${ROLLOUT_BS_SG:-64}"; MINI_ROLLOUT_BS=64; GLOBAL_BS=32
  CUDA_IDS="${CUDA_VISIBLE_DEVICES:-0}"
  echo "[single-gpu] reduced batches; expect slow rollouts. Verify with --smoke first." >&2
fi
if [[ "$SMOKE" == "1" ]]; then
  TOTAL_EPOCHS=1; ROLLOUT_BS=8; MINI_ROLLOUT_BS=8; GLOBAL_BS=8; SAVE_FREQ=-1
  # rollout_bs(8) * n(5) = 40 must be divisible by the experience micro-batch.
  EXTRA_SMOKE="trainer.max_steps=1 trainer.val_before_train=false trainer.logger=[console] trainer.save_best_checkpoint=false worker.actor.micro_batch_size_per_device_for_experience=8"
else EXTRA_SMOKE=""; fi

# ===========================================================================
# SWITCH TABLE  (the ONLY thing that differs across rows)
# ===========================================================================
#                       adv   clipL clipH onlineFilt disableKL is_noisy use_prcp contr_type aug_ent ori_ent
# cot_sft : no RL (eval the Stage1 init directly)
# dapo    : dapo  0.20  0.28  true        true       false    false    -          false   false
# intra   : dapo  0.20  0.28  true        true       false    true     <INTRA>    false   false
# inter   : dapo  0.20  0.28  true        true       false    true     <INTER>    false   false
# tapo    : dapo  0.20  0.28  true        true       true     true     augmented  true    true
# ---------------------------------------------------------------------------
case "$VARIANT" in
  cot_sft)
    echo ">>> cot_sft: no RL. Run the Stage1 init through the eval matrix only."
    echo ">>> init=$INIT_LABEL prompt_mode=$PROMPT_MODE prompt=$FORMAT_PROMPT"
    exit 0;;
  dapo)  IS_NOISY=false USE_PRCP=false CONTR="" AUG_ENT=false ORI_ENT=false;;
  intra) IS_NOISY=false USE_PRCP=true  CONTR="${CONTRASTIVE_INTRA:-intra}" AUG_ENT=false ORI_ENT=false;;
  inter) IS_NOISY=false USE_PRCP=true  CONTR="${CONTRASTIVE_INTER:-inter}" AUG_ENT=false ORI_ENT=false;;
  tapo)  IS_NOISY=true  USE_PRCP=true  CONTR="augmented" AUG_ENT=true ORI_ENT=true;;
  *) echo "unknown VARIANT $VARIANT">&2; exit 2;;
esac

[[ -d "$TAPO_DIR" ]] || { echo "missing TAPO_DIR: $TAPO_DIR" >&2; exit 14; }
[[ -d "$STAGE1_INIT" ]] || { echo "missing STAGE1_INIT: $STAGE1_INIT" >&2; exit 14; }
[[ -f "$TAPO_DIR/$CONFIG" ]] || { echo "missing config: $TAPO_DIR/$CONFIG" >&2; exit 14; }
[[ -e "$TAPO_DIR/$TRAIN_FILE" ]] || { echo "missing train data: $TAPO_DIR/$TRAIN_FILE" >&2; exit 14; }
[[ -f "$TAPO_DIR/$FORMAT_PROMPT" ]] || { echo "missing prompt: $TAPO_DIR/$FORMAT_PROMPT" >&2; exit 14; }

# ----- source-truth guard for intra/inter ----------------------------------
# I cannot see the TAPO source, so I do NOT know the exact accepted strings for
# `contrastive_type`. Released config uses "augmented". Verify intra/inter map to
# real enum values; abort loudly rather than silently no-op.
if [[ "$VARIANT" == "intra" || "$VARIANT" == "inter" ]]; then
  if ! grep -RInE "contrastive_type" "$TAPO_DIR" >/tmp/_ct.txt 2>/dev/null; then
    echo "WARN: could not grep contrastive_type in $TAPO_DIR; confirm enum manually." >&2
  else
    if ! grep -qiE "\"?${CONTR}\"?" /tmp/_ct.txt; then
      echo "ABORT: contrastive_type='${CONTR}' not found in TAPO source." >&2
      echo "Found references:" >&2; sed -n '1,40p' /tmp/_ct.txt >&2
      echo "Set CONTRASTIVE_${VARIANT^^}=<real value> after confirming, then rerun." >&2
      exit 14
    fi
  fi
fi

PRCP_ARGS=()
if [[ "$USE_PRCP" == "true" ]]; then
  PRCP_ARGS+=(algorithm.use_kl_prcp=true
              algorithm.kl_prcp_coef="${KL_PRCP_COEF:-0.01}"
              algorithm.kl_prcp_penalty=low_var_kl
              algorithm.kl_prcp_apply_mode="${KL_PRCP_APPLY_MODE:-all}")
  [[ -n "$CONTR" ]] && PRCP_ARGS+=(algorithm.contrastive_type="$CONTR")
else
  PRCP_ARGS+=(algorithm.use_kl_prcp=false)
fi

CMD=( "$PY" -m verl.trainer.main
  config="$CONFIG"
  data.train_files="$TRAIN_FILE" data.val_files="$VAL_FILE"
  data.rollout_batch_size="$ROLLOUT_BS" data.mini_rollout_batch_size="$MINI_ROLLOUT_BS"
  data.format_prompt="$FORMAT_PROMPT" data.seed="$SEED"
  data.max_pixels="$MAX_PIXELS" data.min_pixels="$MIN_PIXELS"
  worker.actor.model.model_path="$STAGE1_INIT"
  worker.actor.global_batch_size="$GLOBAL_BS"
  worker.actor.clip_ratio_low=0.2 worker.actor.clip_ratio_high=0.28
  worker.actor.is_noisy="$IS_NOISY"
  worker.rollout.tensor_parallel_size="$TP" worker.rollout.n="$ROLLOUT_N"
  worker.rollout.temperature="$TEMP"
  worker.reward.reward_function="$REWARD"
  algorithm.adv_estimator=dapo
  algorithm.disable_kl=true
  algorithm.online_filtering=true algorithm.filter_key=accuracy
  algorithm.filter_low=0.01 algorithm.filter_high=0.99
  "${PRCP_ARGS[@]}"
  algorithm.use_aug_entropy_loss="$AUG_ENT" algorithm.aug_entropy_loss_coef=0.03
  algorithm.use_ori_entropy_loss="$ORI_ENT" algorithm.ori_entropy_loss_coef=0.03
  trainer.experiment_name="stage2_${INIT_LABEL}_${PROMPT_MODE}_${VARIANT}_seed${SEED}"
  trainer.n_gpus_per_node="$N_GPU" trainer.total_epochs="$TOTAL_EPOCHS"
  trainer.save_freq="$SAVE_FREQ" trainer.save_best_checkpoint=true
  $EXTRA_SMOKE )

echo "================= RESOLVED CONFIG ($INIT_LABEL/$PROMPT_MODE/$VARIANT) ================="
printf '%s\n' "${CMD[@]}"
echo "============================================================="
if [[ "$DRY" == "1" ]]; then echo "[dry-run] not launching."; exit 0; fi
cd "$TAPO_DIR"
CUDA_VISIBLE_DEVICES="$CUDA_IDS" "${CMD[@]}"
