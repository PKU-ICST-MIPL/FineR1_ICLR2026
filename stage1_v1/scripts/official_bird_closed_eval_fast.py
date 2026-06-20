#!/usr/bin/env python3
"""Fast/resumable FineR1 official bird closed-world evaluation.

Metric alignment:
- Uses the same closed-world prompt template as FineR1_ICLR2026/eval/evaluation.py.
- Uses the same answer extraction and substring correctness rule.
- Uses the same bird_seen/bird_unseen JSONL schema.

Execution changes:
- Writes results after every batch.
- Supports resume from an existing output JSON.
- Allows larger batch sizes for Qwen2.5-VL inference.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


QUESTION_TEMPLATE = (
    "Given the question: {Question}, based on the options provided in {Options}, "
    "output the thinking process in <think> </think> and final choice in <answer> </answer> tags. "
    "The response format should be: <think>...</think> <answer>choice</answer>."
)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def extract_answer(output_str: str) -> str | None:
    match = re.search(r"<answer>(.*)</answer>", output_str, re.DOTALL)
    return match.group(1).strip() if match else None


def is_correct(item: dict[str, Any], output: str) -> tuple[bool, str | None]:
    model_ans = extract_answer(output)
    gt = str(item["ground_truth"]).lower().replace("_", " ")
    if model_ans:
        norm_ans = model_ans.lower().replace("_", " ")
        return gt in norm_ans, norm_ans
    return False, None


def save_output(
    path: Path,
    results: list[dict[str, Any]],
    n_total: int,
    option_order: str,
    option_shuffle_seed: int,
) -> None:
    correct = sum(1 for row in results if row.get("correct"))
    accuracy = correct / n_total * 100 if n_total else 0.0
    payload = {
        "mode": "closedworld",
        "accuracy": accuracy,
        "completed": len(results),
        "total": n_total,
        "option_order": option_order,
        "option_shuffle_seed": option_shuffle_seed,
        "results": results,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def load_existing(path: Path, option_order: str, option_shuffle_seed: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("option_order", "original") != option_order:
        raise SystemExit(f"resume option-order mismatch in {path}")
    if option_order == "shuffled" and data.get("option_shuffle_seed") != option_shuffle_seed:
        raise SystemExit(f"resume option-shuffle-seed mismatch in {path}")
    return list(data.get("results") or [])


def shuffle_options(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    shuffled_rows = copy.deepcopy(rows)
    for row in shuffled_rows:
        key = f"{seed}:{row['image_path']}".encode("utf-8")
        row_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
        random.Random(row_seed).shuffle(row["options"])
        if row["ground_truth"] not in row["options"]:
            raise SystemExit(f"ground truth missing from options: {row['image_path']}")
    return shuffled_rows


def make_messages(items: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    messages = []
    for item in items:
        image_path = str(item["image_path"])
        if not image_path.startswith("file://"):
            image_path = f"file://{image_path}"
        messages.append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {
                            "type": "text",
                            "text": QUESTION_TEMPLATE.format(
                                Question=item["question"],
                                Options=item["options"],
                            ),
                        },
                    ],
                }
            ]
        )
    return messages


def load_model(model_path: Path, device: str, min_pixels: int | None, max_pixels: int | None):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    processor = AutoProcessor.from_pretrained(str(model_path))
    image_processor = processor.image_processor
    if min_pixels is not None:
        image_processor.min_pixels = min_pixels
        image_processor.size["shortest_edge"] = min_pixels
    if max_pixels is not None:
        image_processor.max_pixels = max_pixels
        image_processor.size["longest_edge"] = max_pixels
    if min_pixels is not None or max_pixels is not None:
        print(f"processor pixels: min={image_processor.min_pixels} max={image_processor.max_pixels}")
    processor.tokenizer.padding_side = "left"
    model.eval()
    return model, processor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--prompt-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--image-min-pixels", type=int)
    parser.add_argument("--image-max-pixels", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shuffle-options", action="store_true")
    parser.add_argument("--option-shuffle-seed", type=int, default=20260620)
    args = parser.parse_args()

    set_seed(42)
    data = read_jsonl(args.prompt_path)
    option_order = "shuffled" if args.shuffle_options else "original"
    if args.shuffle_options:
        data = shuffle_options(data, args.option_shuffle_seed)
    if args.limit:
        data = data[: args.limit]
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    results = (
        load_existing(args.output_path, option_order, args.option_shuffle_seed)
        if args.resume
        else []
    )
    start_idx = len(results)
    if start_idx > len(data):
        raise SystemExit(f"existing results exceed data length: {start_idx}>{len(data)}")

    model, processor = load_model(
        args.model_path, args.device, args.image_min_pixels, args.image_max_pixels
    )
    pbar = tqdm(range(start_idx, len(data), args.batch_size), initial=start_idx // args.batch_size)
    for start in pbar:
        batch_items = data[start : start + args.batch_size]
        batch = make_messages(batch_items)
        text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch]
        image_inputs, video_inputs = process_vision_info(batch)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(args.device)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, out)]
        outputs = processor.batch_decode(trimmed, skip_special_tokens=True)
        for item, output in zip(batch_items, outputs):
            correct, model_ans = is_correct(item, output)
            results.append(
                {
                    "question": item,
                    "ground_truth": item["ground_truth"],
                    "model_output": output,
                    "parsed_answer": model_ans,
                    "correct": correct,
                }
            )
        save_output(args.output_path, results, len(data), option_order, args.option_shuffle_seed)
        correct_so_far = sum(1 for row in results if row.get("correct"))
        pbar.set_description(f"acc={correct_so_far / len(results) * 100:.2f}% n={len(results)}")

    save_output(args.output_path, results, len(data), option_order, args.option_shuffle_seed)
    final = json.loads(args.output_path.read_text(encoding="utf-8"))
    print(f"[Closed-world] Accuracy = {final['accuracy']:.2f}% ({final['completed']}/{final['total']})")


if __name__ == "__main__":
    main()
