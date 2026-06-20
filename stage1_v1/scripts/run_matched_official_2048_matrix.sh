#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/tata/tata/F/rsy/FineR1_repro}"
PY="${PY:-/home/tata/tata/D/anaconda/envs/finer1_tapo/bin/python}"
CUSPARSELT_LIB="${CUSPARSELT_LIB:-/home/tata/tata/D/anaconda/envs/finer1_sft/lib/python3.10/site-packages/nvidia/cusparselt/lib}"
PROMPTS="$REPO_ROOT/FineR1_ICLR2026/eval/data_abs"
BALANCED_OUT="$REPO_ROOT/FineR1_ICLR2026_clean/results_official_bird_balanced_v1_r1x_p1x_e4_lr3e-6_adamw_paper2048"
OUT="$REPO_ROOT/FineR1_ICLR2026_clean/results_official_bird_matched2048"
MAX_NEW_TOKENS=2048
export LD_LIBRARY_PATH="$CUSPARSELT_LIB:${LD_LIBRARY_PATH:-}"
mkdir -p "$OUT"
cd "$REPO_ROOT"

declare -A MODELS
MODELS[plain]="$REPO_ROOT/FineR1_ICLR2026_clean/LLaMA-Factory/saves/qwen25vl_3b_cub_v27c_abs_frozenvis_plain_cot_frozenvis_v27c_abs_strict_plain_top2_e3"
MODELS[rigid]="$REPO_ROOT/FineR1_ICLR2026_clean/LLaMA-Factory/saves/qwen25vl_3b_cub_v27c_abs_rigid_frozenvis_filtered_grounded_top2_frozenvis_v27c_abs_rigid_top2_e8_lr5e6"
MODELS[balanced]="$REPO_ROOT/FineR1_ICLR2026_clean/LLaMA-Factory/saves/qwen25vl_3b_cub_balanced_filtered_grounded_top2_v1_r1x_p1x_e4_lr3e-6_adamw"

run_closed() {
  local label="$1"
  local split="$2"
  local model="${MODELS[$label]}"
  local output="$OUT/${label}_${split}_closed.json"
  [[ -s "$output" ]] && return
  "$PY" scripts/official_bird_closed_eval_fast.py \
    --model-path "$model" --prompt-path "$PROMPTS/bird_${split}.jsonl" \
    --output-path "$output" --device cuda:0 --batch-size 8 \
    --max-new-tokens "$MAX_NEW_TOKENS" --resume \
    --image-min-pixels 1024 --image-max-pixels 589824
}

run_open() {
  local label="$1"
  local split="$2"
  local model="${MODELS[$label]}"
  local output="$OUT/${label}_${split}_open.json"
  [[ -s "$output" ]] && return
  local tmp="$OUT/.${label}_open_${split}"
  mkdir -p "$tmp"
  "$PY" FineR1_ICLR2026/eval/evaluation.py \
    --mode open --model_path "$model" \
    --prompt_path "$PROMPTS/bird_${split}.jsonl" --output_path "$tmp" \
    --batch_size 8 --device cuda:0 --max_new_tokens "$MAX_NEW_TOKENS" \
    --siglip_path google/siglip-base-patch16-256 --domain bird \
    --image_min_pixels 1024 --image_max_pixels 589824
  local generated="$tmp/$(basename "$model")_bird_${split}_open.json"
  [[ -s "$generated" ]] || { echo "missing output: $generated" >&2; exit 14; }
  mv "$generated" "$output"
  rmdir "$tmp" 2>/dev/null || true
}

if [[ "${1:-}" == "--model-only" ]]; then
  label="${2:?model label required: plain or rigid}"
  [[ "$label" == "plain" || "$label" == "rigid" ]] || { echo "bad label: $label" >&2; exit 2; }
  run_closed "$label" seen
  run_closed "$label" unseen
  run_open "$label" seen
  run_open "$label" unseen
  echo "[$(date '+%F %T')] completed model-only $label"
  exit 0
fi

while [[ ! -s "$BALANCED_OUT/summary.json" ]]; do
  echo "[$(date '+%F %T')] waiting for balanced matched-2048 run"
  sleep 60
done
for split in seen unseen; do
  cp -f "$BALANCED_OUT/balanced_${split}_closed.json" "$OUT/balanced_${split}_closed.json"
  cp -f "$BALANCED_OUT/balanced_${split}_open.json" "$OUT/balanced_${split}_open.json"
done

# Parallel model-only runners may finish slightly after balanced. Do not race
# their resumable JSON writers or mistake a partial closed output for complete.
while pgrep -f 'run_matched_official_2048_matrix.sh --model-only' >/dev/null; do
  echo "[$(date '+%F %T')] waiting for parallel plain/rigid runners"
  sleep 60
done

for label in plain rigid; do
  run_closed "$label" seen
  run_closed "$label" unseen
  run_open "$label" seen
  run_open "$label" unseen
done

"$PY" - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1]); rows = []
for model in ("plain", "rigid", "balanced"):
    for world in ("closed", "open"):
        for split in ("seen", "unseen"):
            data = json.loads((out / f"{model}_{split}_{world}.json").read_text())
            results = data.get("results") or []
            answer_key = "parsed_answer" if world == "closed" else "answer"
            missing = sum(not (r.get(answer_key) or "") for r in results)
            grounded = sum("<ref>" in (r.get("model_output") or "") and
                           "<box>" in (r.get("model_output") or "") for r in results)
            rows.append({
                "model": model, "world": world, "split": split, "n": len(results),
                "accuracy": data.get("accuracy", data.get("accuracy_TI")),
                "semantic_bird": data.get("avg_semantic_similarity"),
                "missing_answers": missing,
                "ref_box_rate": grounded / len(results) if results else None,
            })
(out / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
headers = "| model | world | split | n | accuracy/TI | semantic | missing | ref-box |"
lines = ["# Matched 2048-token, 589824/1024 bird matrix", "", headers,
         "|---|---|---|---:|---:|---:|---:|---:|"]
for r in rows:
    sem = "NA" if r["semantic_bird"] is None else f'{100*r["semantic_bird"]:.2f}'
    lines.append(f'| {r["model"]} | {r["world"]} | {r["split"]} | {r["n"]} | '
                 f'{r["accuracy"]:.2f} | {sem} | {r["missing_answers"]} | {r["ref_box_rate"]:.3f} |')
(out / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY
