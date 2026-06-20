#!/usr/bin/env python3
"""Create and resumably upload the Stage1 v1 checkpoint to Hugging Face."""

import os
import time
from pathlib import Path

from huggingface_hub import HfApi, get_token


MODEL_DIR = Path(os.environ["MODEL_DIR"])
REPO_ID = os.environ.get("HF_REPO_ID", "FineR1-Qwen2.5-VL-3B-Stage1-Grounded-v1")
PRIVATE = os.environ.get("HF_PRIVATE", "0") == "1"
STATUS_FILE = Path(os.environ.get("HF_STATUS_FILE", MODEL_DIR / "hf_upload_status.txt"))
RETRY_SECONDS = int(os.environ.get("HF_RETRY_SECONDS", "300"))

IGNORE = [
    "trainer_state.json",
    "trainer_log.jsonl",
    "training_args.bin",
    "training_loss.png",
    "train_results.json",
    "all_results.json",
    "hf_upload_status.txt",
]


def record(message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    STATUS_FILE.write_text(line + "\n", encoding="utf-8")


def main() -> None:
    if not MODEL_DIR.is_dir():
        raise SystemExit(f"missing model directory: {MODEL_DIR}")
    token = get_token()
    if not token:
        raise SystemExit("no Hugging Face token found; run `hf auth login`")

    required = [
        "config.json",
        "preprocessor_config.json",
        "model.safetensors.index.json",
        "tokenizer_config.json",
        "chat_template.json",
    ]
    missing = [name for name in required if not (MODEL_DIR / name).is_file()]
    if missing:
        raise SystemExit(f"incomplete checkpoint, missing: {missing}")

    api = HfApi(token=token)
    while True:
        try:
            url = api.create_repo(
                repo_id=REPO_ID,
                repo_type="model",
                private=PRIVATE,
                exist_ok=True,
            )
            record(f"repository ready: {url}")
            api.upload_large_folder(
                repo_id=str(url.repo_id),
                repo_type="model",
                folder_path=MODEL_DIR,
                private=PRIVATE,
                ignore_patterns=IGNORE,
                num_workers=4,
                print_report=True,
                print_report_every=60,
            )
            record(f"UPLOAD COMPLETE: https://huggingface.co/{url.repo_id}")
            return
        except Exception as exc:
            record(f"upload attempt failed: {type(exc).__name__}: {exc}; retrying in {RETRY_SECONDS}s")
            time.sleep(RETRY_SECONDS)


if __name__ == "__main__":
    main()
