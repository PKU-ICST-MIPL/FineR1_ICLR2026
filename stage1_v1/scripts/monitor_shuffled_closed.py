#!/usr/bin/env python3
import json
from pathlib import Path

root = Path("FineR1_ICLR2026_clean/results_official_bird_shuffled_closed_seed20260620")
for model in ("plain", "rigid", "balanced"):
    for split, expected in (("seen", 3483), ("unseen", 2311)):
        path = root / f"{model}_{split}_closed_shuffled.json"
        if not path.exists():
            print(f"{model:8} {split:6} not started")
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"{model:8} {split:6} output update in progress")
            continue
        completed = data.get("completed", len(data.get("results", [])))
        print(
            f"{model:8} {split:6} {completed:4}/{expected} "
            f"({completed / expected:6.2%}) current_acc={data.get('accuracy', 0):6.2f}%"
        )
