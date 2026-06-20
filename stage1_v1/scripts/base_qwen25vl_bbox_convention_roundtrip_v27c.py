#!/usr/bin/env python3
"""Infer base Qwen2.5-VL bbox convention on CUB images.

The script prompts the unmodified base model to locate the bird and output
`bbox_2d`, then interprets the returned numbers under three conventions:

1. qwen_resized_abs: coordinates are absolute pixels on Qwen resized canvas.
2. original_abs: coordinates are absolute pixels on the original image.
3. normalized_1000: coordinates are full-image 0..1000.

All interpretations are projected into the Qwen resized canvas and compared to
the teacher object box converted into that same canvas.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qwen25vl_coordinate_utils import (  # noqa: E402
    LLAMAFACTORY_IMAGE_MAX_PIXELS,
    LLAMAFACTORY_IMAGE_MIN_PIXELS,
    QWEN_FACTOR,
    QWEN_IMAGE_MAX_PIXELS,
    QWEN_IMAGE_MIN_PIXELS,
    box_1000_to_resized_pixels,
    clamp_box,
    coordinate_bounds_for_image,
    image_size,
    processor_grid_bounds_for_image,
    regularize_pil_for_llamafactory,
)


PROMPT = """Locate the entire bird in this image. Return JSON only, with exactly this schema:
[{"bbox_2d": [x1, y1, x2, y2], "label": "bird"}]
Do not explain."""


BOX_RE = re.compile(r'"bbox_2d"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]', re.I)
LOOSE_BOX_RE = re.compile(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_records(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if row.get("object_bbox_2d") and row.get("image_path"):
            selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def parse_bbox(text: str) -> list[float] | None:
    match = BOX_RE.search(text or "") or LOOSE_BOX_RE.search(text or "")
    if not match:
        return None
    return [float(match.group(i)) for i in range(1, 5)]


def iou(a: list[int] | None, b: list[int] | None) -> float | None:
    if a is None or b is None:
        return None
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def original_abs_to_qwen(box: list[float], original_w: int, original_h: int, qwen_w: int, qwen_h: int) -> list[int] | None:
    return clamp_box(
        [
            box[0] / original_w * qwen_w,
            box[1] / original_h * qwen_h,
            box[2] / original_w * qwen_w,
            box[3] / original_h * qwen_h,
        ],
        qwen_w,
        qwen_h,
    )


def font(size: int = 13):
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_overlay(image: Image.Image, boxes: list[dict[str, Any]], title: str) -> Image.Image:
    header_h = 96
    out = Image.new("RGB", (image.width, image.height + header_h), "white")
    out.paste(image.convert("RGB"), (0, header_h))
    draw = ImageDraw.Draw(out)
    draw.text((10, 8), title[:150], fill=(25, 30, 42), font=font(14))
    draw.text((10, 31), "red=teacher object, green=resized_abs, blue=original_abs, orange=normalized_1000", fill=(90, 98, 115), font=font(12))
    colors = {
        "teacher": (220, 50, 65),
        "qwen_resized_abs": (25, 135, 84),
        "original_abs": (55, 95, 220),
        "normalized_1000": (180, 98, 0),
    }
    for item in boxes:
        box = item.get("box")
        if box is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        color = colors.get(item["name"], (120, 120, 120))
        draw.rectangle([x1, y1 + header_h, x2, y2 + header_h], outline=color, width=3)
        draw.text((x1 + 2, max(header_h, y1 + header_h - 17)), item["name"], fill=color, font=font(12))
    return out


def build_inputs(processor: Any, image_path: str, prompt: str):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    from qwen_vl_utils import process_vision_info

    image_inputs, video_inputs = process_vision_info(messages)
    return processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")


def generate(model: Any, processor: Any, image_path: str, max_new_tokens: int) -> str:
    import torch

    inputs = build_inputs(processor, image_path, PROMPT)
    inputs = inputs.to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, out)]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0]


def load_model(model_path: str, dtype: str):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16 if dtype == "fp16" else torch.float32
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, processor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-jsonl", type=Path, required=True)
    parser.add_argument("--model-path", default="/home/tata/tata/models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--image-min-pixels", type=int, default=LLAMAFACTORY_IMAGE_MIN_PIXELS)
    parser.add_argument("--image-max-pixels", type=int, default=LLAMAFACTORY_IMAGE_MAX_PIXELS)
    parser.add_argument("--qwen-min-pixels", type=int, default=QWEN_IMAGE_MIN_PIXELS)
    parser.add_argument("--qwen-max-pixels", type=int, default=QWEN_IMAGE_MAX_PIXELS)
    parser.add_argument("--qwen-factor", type=int, default=QWEN_FACTOR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, processor = load_model(args.model_path, args.dtype)

    results = []
    win_counts: dict[str, int] = {}
    rows = select_records(read_jsonl(args.teacher_jsonl), args.limit)
    for idx, row in enumerate(rows, 1):
        image_path = str(row["image_path"])
        original_h, original_w = image_size(image_path)
        qwen_w, qwen_h = coordinate_bounds_for_image(
            image_path,
            "llamafactory_qwen25_abs",
            image_min_pixels=args.image_min_pixels,
            image_max_pixels=args.image_max_pixels,
            qwen_min_pixels=args.qwen_min_pixels,
            qwen_max_pixels=args.qwen_max_pixels,
            factor=args.qwen_factor,
        )
        processor_w, processor_h = processor_grid_bounds_for_image(
            processor,
            image_path,
            "llamafactory_qwen25_abs",
            image_min_pixels=args.image_min_pixels,
            image_max_pixels=args.image_max_pixels,
        )
        raw_output = generate(model, processor, image_path, args.max_new_tokens)
        pred = parse_bbox(raw_output)
        teacher_object = box_1000_to_resized_pixels(row.get("object_bbox_2d"), qwen_w, qwen_h)
        interpretations = {
            "qwen_resized_abs": clamp_box(pred, qwen_w, qwen_h) if pred else None,
            "original_abs": original_abs_to_qwen(pred, original_w, original_h, qwen_w, qwen_h) if pred else None,
            "normalized_1000": box_1000_to_resized_pixels(pred, qwen_w, qwen_h) if pred else None,
        }
        ious = {name: iou(box, teacher_object) for name, box in interpretations.items()}
        valid = {k: v for k, v in ious.items() if v is not None}
        winner = max(valid, key=lambda k: valid[k]) if valid else "parse_failed"
        win_counts[winner] = win_counts.get(winner, 0) + 1

        base_image = regularize_pil_for_llamafactory(
            image_path,
            image_min_pixels=args.image_min_pixels,
            image_max_pixels=args.image_max_pixels,
        ).resize((qwen_w, qwen_h), Image.Resampling.BICUBIC)
        overlay_path = args.output_dir / f"base_roundtrip_{idx:02d}.jpg"
        draw_overlay(
            base_image,
            [{"name": "teacher", "box": teacher_object}]
            + [{"name": name, "box": box} for name, box in interpretations.items()],
            f"{idx}. {row.get('label')} | winner={winner} | raw={pred}",
        ).save(overlay_path, quality=92)
        results.append(
            {
                "idx": idx,
                "label": row.get("label"),
                "image_path": image_path,
                "original_size_wh": [original_w, original_h],
                "qwen_size_wh": [qwen_w, qwen_h],
                "processor_size_wh": [processor_w, processor_h],
                "raw_output": raw_output,
                "parsed_bbox": pred,
                "teacher_object_qwen_abs": teacher_object,
                "interpretations_qwen_canvas": interpretations,
                "ious_vs_teacher_object": ious,
                "winner": winner,
                "overlay": str(overlay_path),
            }
        )

    summary = {
        "model_path": args.model_path,
        "teacher_jsonl": str(args.teacher_jsonl),
        "prompt": PROMPT,
        "win_counts": win_counts,
        "records": results,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
