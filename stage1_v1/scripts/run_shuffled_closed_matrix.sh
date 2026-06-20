#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/tata/tata/F/rsy/FineR1_repro}"
PY="${PY:-/home/tata/tata/D/anaconda/envs/finer1_tapo/bin/python}"
PROMPTS="$REPO_ROOT/FineR1_ICLR2026/eval/data_abs"
OUT="$REPO_ROOT/FineR1_ICLR2026_clean/results_official_bird_shuffled_closed_seed20260620"
SEED="${SEED:-20260620}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
CUSPARSELT_LIB="${CUSPARSELT_LIB:-/home/tata/tata/D/anaconda/envs/finer1_sft/lib/python3.10/site-packages/nvidia/cusparselt/lib}"
export LD_LIBRARY_PATH="$CUSPARSELT_LIB:${LD_LIBRARY_PATH:-}"

declare -A MODELS
MODELS[plain]="$REPO_ROOT/FineR1_ICLR2026_clean/LLaMA-Factory/saves/qwen25vl_3b_cub_v27c_abs_frozenvis_plain_cot_frozenvis_v27c_abs_strict_plain_top2_e3"
MODELS[rigid]="$REPO_ROOT/FineR1_ICLR2026_clean/LLaMA-Factory/saves/qwen25vl_3b_cub_v27c_abs_rigid_frozenvis_filtered_grounded_top2_frozenvis_v27c_abs_rigid_top2_e8_lr5e6"
MODELS[balanced]="$REPO_ROOT/FineR1_ICLR2026_clean/LLaMA-Factory/saves/qwen25vl_3b_cub_balanced_filtered_grounded_top2_v1_r1x_p1x_e4_lr3e-6_adamw"

mkdir -p "$OUT" "$REPO_ROOT/logs"
cd "$REPO_ROOT"

run_model() {
  local label="$1"
  local model="${MODELS[$label]}"
  local split output
  for split in seen unseen; do
    output="$OUT/${label}_${split}_closed_shuffled.json"
    "$PY" scripts/official_bird_closed_eval_fast.py \
      --model-path "$model" \
      --prompt-path "$PROMPTS/bird_${split}.jsonl" \
      --output-path "$output" \
      --device cuda:0 --batch-size "$BATCH_SIZE" \
      --max-new-tokens "$MAX_NEW_TOKENS" --resume \
      --image-min-pixels 1024 --image-max-pixels 589824 \
      --shuffle-options --option-shuffle-seed "$SEED"
  done
}

if [[ "${1:-}" == "--model-only" ]]; then
  [[ -n "${2:-}" && -n "${MODELS[$2]:-}" ]] || { echo "model must be plain, rigid, or balanced" >&2; exit 2; }
  run_model "$2"
  exit 0
fi

pids=()
for label in plain rigid balanced; do
  bash "$0" --model-only "$label" >"$REPO_ROOT/logs/shuffled_closed_${label}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

"$PY" - "$OUT" <<'PY'
import json, sys
from collections import Counter
from pathlib import Path

out = Path(sys.argv[1])
rows = []
for model in ("plain", "rigid", "balanced"):
    for split in ("seen", "unseen"):
        data = json.loads((out / f"{model}_{split}_closed_shuffled.json").read_text())
        positions = Counter()
        for result in data["results"]:
            question = result["question"]
            positions[question["options"].index(question["ground_truth"])] += 1
        rows.append({
            "model": model,
            "split": split,
            "n": len(data["results"]),
            "accuracy": data["accuracy"],
            "gt_position_counts": dict(sorted(positions.items())),
        })
(out / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
lines = ["# Deterministically shuffled closed-world control", "",
         "| model | split | n | accuracy | GT positions 0/1/2/3 |",
         "|---|---|---:|---:|---|"]
for row in rows:
    counts = row["gt_position_counts"]
    pos = "/".join(str(counts.get(str(i), counts.get(i, 0))) for i in range(4))
    lines.append(f'| {row["model"]} | {row["split"]} | {row["n"]} | {row["accuracy"]:.2f} | {pos} |')
(out / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY
