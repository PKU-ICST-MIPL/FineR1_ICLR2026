#!/usr/bin/env python3
"""Validate Qwen2.5-VL coordinate convention for CUB v27c grounding data.

This script is intentionally CPU-only by default. It samples teacher records,
converts full-image 0..1000 teacher boxes into Qwen2.5-VL resized-image
absolute pixel coordinates, and writes visual overlays plus a JSON summary.

Optional model round-trip inference can be added later; the first blocking check
is whether our labels are rendered in the same image space used by training.
"""

from __future__ import annotations

import argparse
import json
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
    convert_teacher_box_to_output,
    coordinate_bounds_for_image,
    image_size,
    regularize_pil_for_llamafactory,
    resized_size_for_image,
    smart_resize,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def font(size: int = 14):
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, color: tuple[int, int, int]) -> None:
    text = text[:80]
    fnt = font(13)
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=fnt)
    draw.rectangle([bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2], fill=(255, 255, 255))
    draw.text((x, y), text, fill=color, font=fnt)


def draw_boxes(image: Image.Image, boxes: list[dict[str, Any]], title: str) -> Image.Image:
    header_h = 76
    out = Image.new("RGB", (image.width, image.height + header_h), "white")
    out.paste(image.convert("RGB"), (0, header_h))
    draw = ImageDraw.Draw(out)
    draw.text((10, 8), title[:120], fill=(28, 33, 45), font=font(15))
    draw.text((10, 34), f"canvas={image.width}x{image.height}", fill=(90, 98, 115), font=font(13))
    colors = [(222, 65, 72), (32, 128, 84), (64, 96, 220), (178, 96, 0), (142, 68, 173)]
    for idx, item in enumerate(boxes):
        box = item.get("box")
        if not box:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        color = colors[idx % len(colors)]
        y1h, y2h = y1 + header_h, y2 + header_h
        draw.rectangle([x1, y1h, x2, y2h], outline=color, width=3)
        draw_label(draw, (x1 + 2, max(header_h, y1h - 18)), f"{idx+1}:{item.get('label','')}", color)
    return out


def concat_horizontal(images: list[Image.Image]) -> Image.Image:
    height = max(img.height for img in images)
    width = sum(img.width for img in images)
    out = Image.new("RGB", (width, height), "white")
    x = 0
    for image in images:
        out.paste(image, (x, 0))
        x += image.width
    return out


def scale_1000_box_to_image(box: Any, width: int, height: int) -> list[int] | None:
    return box_1000_to_resized_pixels(box, width, height)


def select_records(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if len(row.get("accepted_evidence") or []) >= 2:
            selected.append(row)
        if len(selected) >= limit:
            return selected
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--coordinate-system", choices=["qwen25_abs", "llamafactory_qwen25_abs"], default="llamafactory_qwen25_abs")
    parser.add_argument("--image-min-pixels", type=int, default=LLAMAFACTORY_IMAGE_MIN_PIXELS)
    parser.add_argument("--image-max-pixels", type=int, default=LLAMAFACTORY_IMAGE_MAX_PIXELS)
    parser.add_argument("--qwen-min-pixels", type=int, default=QWEN_IMAGE_MIN_PIXELS)
    parser.add_argument("--qwen-max-pixels", type=int, default=QWEN_IMAGE_MAX_PIXELS)
    parser.add_argument("--qwen-factor", type=int, default=QWEN_FACTOR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = select_records(read_jsonl(args.teacher_jsonl), args.limit)
    summary = {
        "teacher_jsonl": str(args.teacher_jsonl),
        "coordinate_system": args.coordinate_system,
        "image_min_pixels": args.image_min_pixels,
        "image_max_pixels": args.image_max_pixels,
        "qwen_min_pixels": args.qwen_min_pixels,
        "qwen_max_pixels": args.qwen_max_pixels,
        "qwen_factor": args.qwen_factor,
        "records": [],
    }

    for idx, row in enumerate(rows, 1):
        image_path = str(row["image_path"])
        original_h, original_w = image_size(image_path)
        lf_h, lf_w = original_h, original_w
        if args.coordinate_system == "llamafactory_qwen25_abs":
            lf_image = regularize_pil_for_llamafactory(
                image_path,
                image_min_pixels=args.image_min_pixels,
                image_max_pixels=args.image_max_pixels,
            )
            lf_w, lf_h = lf_image.size
        else:
            lf_image = Image.open(image_path).convert("RGB")

        qwen_h, qwen_w = smart_resize(
            lf_h,
            lf_w,
            factor=args.qwen_factor,
            min_pixels=args.qwen_min_pixels,
            max_pixels=args.qwen_max_pixels,
        )
        util_h, util_w = resized_size_for_image(
            image_path,
            args.coordinate_system,
            image_min_pixels=args.image_min_pixels,
            image_max_pixels=args.image_max_pixels,
            qwen_min_pixels=args.qwen_min_pixels,
            qwen_max_pixels=args.qwen_max_pixels,
            factor=args.qwen_factor,
        )
        coord_w, coord_h = coordinate_bounds_for_image(
            image_path,
            args.coordinate_system,
            image_min_pixels=args.image_min_pixels,
            image_max_pixels=args.image_max_pixels,
            qwen_min_pixels=args.qwen_min_pixels,
            qwen_max_pixels=args.qwen_max_pixels,
            factor=args.qwen_factor,
        )
        assert (qwen_h, qwen_w) == (util_h, util_w)
        assert (coord_w, coord_h) == (qwen_w, qwen_h)

        original_image = Image.open(image_path).convert("RGB")
        qwen_image = lf_image.resize((qwen_w, qwen_h), Image.Resampling.BICUBIC)
        original_boxes = []
        qwen_boxes = []
        record_items = []
        for ev_idx, ev in enumerate((row.get("accepted_evidence") or [])[:3], 1):
            label = f"{ev.get('part')} | {ev.get('visible_attribute')}"
            box1000 = ev.get("bbox_2d")
            original_box = scale_1000_box_to_image(box1000, original_w, original_h)
            qwen_box = convert_teacher_box_to_output(
                box1000,
                image_path,
                args.coordinate_system,
                image_min_pixels=args.image_min_pixels,
                image_max_pixels=args.image_max_pixels,
                qwen_min_pixels=args.qwen_min_pixels,
                qwen_max_pixels=args.qwen_max_pixels,
                factor=args.qwen_factor,
            )
            original_boxes.append({"label": label, "box": original_box})
            qwen_boxes.append({"label": label, "box": qwen_box})
            record_items.append(
                {
                    "idx": ev_idx,
                    "part": ev.get("part"),
                    "visible_attribute": ev.get("visible_attribute"),
                    "box_1000": box1000,
                    "original_pixel_box": original_box,
                    "qwen_abs_box": qwen_box,
                }
            )

        title = f"{idx}. {row.get('label')} | {Path(image_path).name}"
        original_overlay = draw_boxes(original_image, original_boxes, title + " | original + 0..1000->original")
        qwen_overlay = draw_boxes(qwen_image, qwen_boxes, title + f" | {args.coordinate_system}")
        combined = concat_horizontal([original_overlay, qwen_overlay])
        out_path = args.output_dir / f"coord_check_{idx:02d}.jpg"
        combined.save(out_path, quality=92)
        summary["records"].append(
            {
                "idx": idx,
                "label": row.get("label"),
                "image_path": image_path,
                "original_size_wh": [original_w, original_h],
                "llamafactory_regularized_size_wh": [lf_w, lf_h],
                "qwen_smart_resized_size_wh": [qwen_w, qwen_h],
                "overlay": str(out_path),
                "evidence": record_items,
            }
        )

    summary_path = args.output_dir / "coordinate_convention_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote overlays to {args.output_dir}")


if __name__ == "__main__":
    main()
