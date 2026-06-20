#!/usr/bin/env python3
"""Score CUB eval outputs for v27 thinking-with-grounding experiments."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any

from qwen25vl_coordinate_utils import (
    LLAMAFACTORY_IMAGE_MAX_PIXELS,
    LLAMAFACTORY_IMAGE_MIN_PIXELS,
    QWEN_FACTOR,
    QWEN_IMAGE_MAX_PIXELS,
    QWEN_IMAGE_MIN_PIXELS,
    box_1000_to_resized_pixels,
    coordinate_bounds_for_image,
)


STRICT_BOX_RE = re.compile(
    r"<ref>\s*(?P<label>.*?)\s*</ref>\s*<box>\s*\[\[\s*"
    r"(?P<x1>-?\d+(?:\.\d+)?)\s*,\s*(?P<y1>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<x2>-?\d+(?:\.\d+)?)\s*,\s*(?P<y2>-?\d+(?:\.\d+)?)\s*"
    r"\]\]\s*</box>",
    re.I | re.S,
)
LOOSE_BOX_RE = re.compile(
    r"<ref>\s*(?P<label>.*?)\s*</ref>\s*(?::|：)?\s*\[\[\s*"
    r"(?P<x1>-?\d+(?:\.\d+)?)\s*,\s*(?P<y1>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<x2>-?\d+(?:\.\d+)?)\s*,\s*(?P<y2>-?\d+(?:\.\d+)?)\s*"
    r"\]\]",
    re.I | re.S,
)
JSON_BBOX_RE = re.compile(
    r'"bbox_2d"\s*:\s*\[\s*'
    r'(?P<x1>-?\d+(?:\.\d+)?)\s*,\s*(?P<y1>-?\d+(?:\.\d+)?)\s*,\s*'
    r'(?P<x2>-?\d+(?:\.\d+)?)\s*,\s*(?P<y2>-?\d+(?:\.\d+)?)\s*'
    r'\]',
    re.I | re.S,
)
THINK_RE = re.compile(r"<think>(?P<body>.*?)</think>", re.I | re.S)
PART_WORDS = (
    "head",
    "bill",
    "beak",
    "wing",
    "breast",
    "tail",
    "throat",
    "crown",
    "back",
    "belly",
    "eye",
    "leg",
)
COMPARISON_RE = re.compile(r"\b(compared?|contrast|distinguish|rules?\s+out|confus|whereas|unlike)\b", re.I)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def norm_name(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"<.*?>", " ", text)
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\b(the|a|an|bird|species|class|common)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def norm_part(text: str) -> str:
    text = norm_name(text)
    if text == "beak":
        return "bill"
    return text


def strict_relaxed_match(label: str, pred: str) -> bool:
    n_label = norm_name(label)
    n_pred = norm_name(pred)
    if not n_label or not n_pred:
        return False
    if n_label == n_pred:
        return True
    return bool(re.search(rf"\b{re.escape(n_label)}\b", n_pred))


def clamp_box(vals: list[float], width: int, height: int) -> list[int] | None:
    x1, y1, x2, y2 = [float(v) for v in vals]
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = max(0, min(width - 1, int(round(x1))))
    y1 = max(0, min(height - 1, int(round(y1))))
    x2 = max(x1 + 1, min(width, int(round(x2))))
    y2 = max(y1 + 1, min(height, int(round(y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def parse_boxes(text: str, loose: bool, width: int, height: int) -> list[dict[str, Any]]:
    boxes = []
    seen: set[tuple[str, tuple[int, int, int, int]]] = set()
    for match in STRICT_BOX_RE.finditer(text or ""):
        vals = [float(match.group(k)) for k in ("x1", "y1", "x2", "y2")]
        box = clamp_box(vals, width, height)
        if box is None:
            continue
        label = norm_part(match.group("label"))
        key = (label, tuple(box))
        seen.add(key)
        boxes.append({"label": label, "bbox_2d": box, "source": "strict", "start": match.start(), "end": match.end()})
    if loose:
        for match in LOOSE_BOX_RE.finditer(text or ""):
            vals = [float(match.group(k)) for k in ("x1", "y1", "x2", "y2")]
            box = clamp_box(vals, width, height)
            if box is None:
                continue
            label = norm_part(match.group("label"))
            key = (label, tuple(box))
            if key in seen:
                continue
            seen.add(key)
            boxes.append({"label": label, "bbox_2d": box, "source": "loose", "start": match.start(), "end": match.end()})
    return sorted(boxes, key=lambda x: x["start"])


def parse_json_bbox_count(text: str, width: int, height: int) -> int:
    count = 0
    for match in JSON_BBOX_RE.finditer(text or ""):
        vals = [float(match.group(k)) for k in ("x1", "y1", "x2", "y2")]
        if clamp_box(vals, width, height) is not None:
            count += 1
    return count


def box_area(box: list[int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def box_iou(a: list[int], b: list[int]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = box_area(a) + box_area(b) - inter
    return inter / union if union else 0.0


def center_in_box(point_box: list[int], target: list[int]) -> bool:
    cx = (point_box[0] + point_box[2]) / 2
    cy = (point_box[1] + point_box[3]) / 2
    return target[0] <= cx <= target[2] and target[1] <= cy <= target[3]


def duplicate_row(boxes: list[dict[str, Any]]) -> bool:
    seen: set[tuple[int, int, int, int]] = set()
    for item in boxes:
        key = tuple(item["bbox_2d"])
        if key in seen:
            return True
        seen.add(key)
    return False


def think_body(text: str) -> str:
    match = THINK_RE.search(text or "")
    return match.group("body") if match else ""


def mentioned_parts(text: str) -> set[str]:
    body = think_body(text)
    body = STRICT_BOX_RE.sub(" ", body)
    body = LOOSE_BOX_RE.sub(" ", body)
    body = re.sub(r"<.*?>", " ", body).lower()
    found = set()
    for part in PART_WORDS:
        if re.search(rf"\b{re.escape(part)}\b", body):
            found.add(norm_part(part))
    return found


def comparison_near_boxes(text: str, boxes: list[dict[str, Any]], window: int = 220) -> bool:
    if not boxes:
        return False
    for item in boxes:
        left = max(0, item["start"] - window)
        right = min(len(text), item["end"] + window)
        if COMPARISON_RE.search(text[left:right]):
            return True
    return False


def teacher_lookup_keys(value: Any) -> list[str]:
    raw = str(value or "").strip()
    if not raw:
        return []
    normalized = raw.replace("\\", "/")
    keys = [raw, normalized]
    path = Path(normalized)
    if path.name:
        keys.append(path.name)
    if len(path.parts) >= 2:
        keys.append("/".join(path.parts[-2:]))
    return list(dict.fromkeys(keys))


def load_teacher(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    teacher = {}
    for row in read_jsonl(path):
        for key in teacher_lookup_keys(row.get("image_path")) + teacher_lookup_keys(row.get("image")):
            teacher[key] = row
    return teacher


def find_teacher_row(teacher: dict[str, dict[str, Any]], row: dict[str, Any]) -> dict[str, Any] | None:
    for key in teacher_lookup_keys(row.get("image_path")) + teacher_lookup_keys(row.get("image")):
        if key in teacher:
            return teacher[key]
    return None


def teacher_box_to_row_space(
    box_1000: Any,
    width: int,
    height: int,
    coordinate_system: str,
) -> list[int] | None:
    if coordinate_system == "normalized_1000":
        return clamp_box(box_1000, 1000, 1000)
    return box_1000_to_resized_pixels(box_1000, width, height)


def teacher_targets(row: dict[str, Any] | None, width: int, height: int, args: argparse.Namespace) -> list[dict[str, Any]]:
    if not row:
        return []
    out = []
    for item in row.get("accepted_evidence") or []:
        box = teacher_box_to_row_space(
            item.get("bbox_2d"),
            width,
            height,
            args.output_coordinate_system,
        )
        if box is not None:
            out.append({"label": norm_part(item.get("part")), "bbox_2d": box})
    return out


def teacher_object_box(row: dict[str, Any] | None, width: int, height: int, args: argparse.Namespace) -> list[int] | None:
    if not row:
        return None
    return teacher_box_to_row_space(
        row.get("object_bbox_2d"),
        width,
        height,
        args.output_coordinate_system,
    )


def match_teacher(box: dict[str, Any], targets: list[dict[str, Any]]) -> tuple[float | None, bool | None]:
    if not targets:
        return None, None
    same_label = [x for x in targets if x["label"] == box["label"]]
    candidates = same_label or targets
    best = max(candidates, key=lambda x: box_iou(box["bbox_2d"], x["bbox_2d"]))
    return box_iou(box["bbox_2d"], best["bbox_2d"]), center_in_box(box["bbox_2d"], best["bbox_2d"])


def row_bounds(row: dict[str, Any], args: argparse.Namespace) -> tuple[int, int]:
    if args.output_coordinate_system == "normalized_1000":
        return 1000, 1000
    if row.get("resized_width") and row.get("resized_height"):
        return int(row["resized_width"]), int(row["resized_height"])
    return coordinate_bounds_for_image(
        str(row.get("image_path") or ""),
        args.output_coordinate_system,
        image_min_pixels=args.image_min_pixels,
        image_max_pixels=args.image_max_pixels,
        qwen_min_pixels=args.qwen_min_pixels,
        qwen_max_pixels=args.qwen_max_pixels,
        factor=args.qwen_factor,
    )


def score_output(path: Path, teacher: dict[str, dict[str, Any]], args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = read_jsonl(path)
    details = []
    for idx, row in enumerate(rows, 1):
        raw = row.get("raw_output") or ""
        label = row.get("label") or row.get("answer") or row.get("ground_truth") or ""
        pred = row.get("prediction") or ""
        width, height = row_bounds(row, args)
        strict_boxes = parse_boxes(raw, loose=False, width=width, height=height)
        loose_boxes = parse_boxes(raw, loose=True, width=width, height=height)
        grounding_turn_boxes = parse_json_bbox_count(row.get("grounding_output") or "", width, height)
        teacher_row = find_teacher_row(teacher, row)
        targets = teacher_targets(teacher_row, width, height, args)
        object_box = teacher_object_box(teacher_row, width, height, args)
        ious = []
        point_hits = []
        object_hits = []
        for box in loose_boxes:
            iou, hit = match_teacher(box, targets)
            if iou is not None:
                ious.append(iou)
            if hit is not None:
                point_hits.append(hit)
            if isinstance(object_box, list) and len(object_box) == 4:
                object_hits.append(center_in_box(box["bbox_2d"], object_box))
        parts = mentioned_parts(raw)
        refs = {x["label"] for x in loose_boxes if x["label"]}
        mentioned_grounded = len(parts & refs) / len(parts) if parts else None
        first_answer = raw.lower().find("<answer>")
        first_box = min([x["start"] for x in strict_boxes], default=-1)
        details.append(
            {
                "idx": idx,
                "model_name": row.get("model_name", path.stem),
                "label": label,
                "prediction": pred,
                "image_path": row.get("image_path", ""),
                "coordinate_system": args.output_coordinate_system,
                "coord_width": width,
                "coord_height": height,
                "teacher_matched": teacher_row is not None,
                "exact_match": norm_name(label) == norm_name(pred),
                "strict_relaxed_match": strict_relaxed_match(label, pred),
                "format_ok": "<think>" in raw.lower() and "</think>" in raw.lower() and "<answer>" in raw.lower() and "</answer>" in raw.lower(),
                "strict_ref_box": bool(strict_boxes),
                "loose_ref_box": bool(loose_boxes),
                "strict_box_count": len(strict_boxes),
                "loose_box_count": len(loose_boxes),
                "grounding_turn_box_count": grounding_turn_boxes,
                "grounding_turn_box_rate": grounding_turn_boxes > 0,
                "duplicate_box_row": duplicate_row(loose_boxes),
                "mean_box_area": statistics.mean([box_area(x["bbox_2d"]) for x in loose_boxes]) if loose_boxes else None,
                "mean_box_area_1000": statistics.mean([box_area(x["bbox_2d"]) for x in loose_boxes]) if loose_boxes else None,
                "teacher_target_count": len(targets),
                "mean_best_iou": statistics.mean(ious) if ious else None,
                "point_hit_rate": sum(point_hits) / len(point_hits) if point_hits else None,
                "object_containment_rate": sum(object_hits) / len(object_hits) if object_hits else None,
                "mentioned_part_count": len(parts),
                "ref_part_count": len(refs),
                "mentioned_part_grounded_rate": mentioned_grounded,
                "comparative_grounded": comparison_near_boxes(raw, strict_boxes),
                "answer_before_grounding": bool(first_answer >= 0 and (first_box < 0 or first_answer < first_box)),
                "output_chars": len(raw),
            }
        )
    n = len(details) or 1

    def avg_present(key: str) -> float | None:
        vals = [x[key] for x in details if isinstance(x.get(key), (int, float))]
        return statistics.mean(vals) if vals else None

    summary = {
        "file": str(path),
        "model_name": details[0]["model_name"] if details else path.stem,
        "n": len(details),
        "exact_acc": sum(x["exact_match"] for x in details) / n,
        "strict_relaxed_acc": sum(x["strict_relaxed_match"] for x in details) / n,
        "format_acc": sum(x["format_ok"] for x in details) / n,
        "strict_ref_box": sum(x["strict_ref_box"] for x in details) / n,
        "loose_ref_box": sum(x["loose_ref_box"] for x in details) / n,
        "avg_strict_boxes": avg_present("strict_box_count"),
        "avg_loose_boxes": avg_present("loose_box_count"),
        "grounding_turn_box_rate": sum(x["grounding_turn_box_rate"] for x in details) / n,
        "avg_grounding_turn_boxes": avg_present("grounding_turn_box_count"),
        "duplicate_box_row_rate": sum(x["duplicate_box_row"] for x in details) / n,
        "teacher_match_rate": sum(x["teacher_matched"] for x in details) / n,
        "coordinate_system": args.output_coordinate_system,
        "mean_box_area": avg_present("mean_box_area"),
        "mean_box_area_1000": avg_present("mean_box_area_1000"),
        "mean_best_iou": avg_present("mean_best_iou"),
        "point_hit": avg_present("point_hit_rate"),
        "object_containment": avg_present("object_containment_rate"),
        "reasoning_grounding_alignment": avg_present("mentioned_part_grounded_rate"),
        "comparative_grounded_rate": sum(x["comparative_grounded"] for x in details) / n,
        "answer_before_grounding_rate": sum(x["answer_before_grounding"] for x in details) / n,
        "avg_output_chars": avg_present("output_chars"),
    }
    return summary, details


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", nargs="+", required=True)
    parser.add_argument("--teacher-jsonl", type=Path, default=None)
    parser.add_argument(
        "--output-coordinate-system",
        choices=["normalized_1000", "qwen25_abs", "llamafactory_qwen25_abs"],
        default="llamafactory_qwen25_abs",
    )
    parser.add_argument("--image-max-pixels", type=int, default=LLAMAFACTORY_IMAGE_MAX_PIXELS)
    parser.add_argument("--image-min-pixels", type=int, default=LLAMAFACTORY_IMAGE_MIN_PIXELS)
    parser.add_argument("--qwen-min-pixels", type=int, default=QWEN_IMAGE_MIN_PIXELS)
    parser.add_argument("--qwen-max-pixels", type=int, default=QWEN_IMAGE_MAX_PIXELS)
    parser.add_argument("--qwen-factor", type=int, default=QWEN_FACTOR)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--details-csv", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    args = parser.parse_args()

    teacher = load_teacher(args.teacher_jsonl)
    summaries = []
    details = []
    for output in args.outputs:
        summary, rows = score_output(Path(output), teacher, args)
        summaries.append(summary)
        details.extend(rows)

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(args.summary_csv, summaries)
    write_csv(args.details_csv, details)

    columns = [
        "model_name",
        "n",
        "exact_acc",
        "strict_relaxed_acc",
        "format_acc",
        "strict_ref_box",
        "loose_ref_box",
        "avg_loose_boxes",
        "grounding_turn_box_rate",
        "avg_grounding_turn_boxes",
        "duplicate_box_row_rate",
        "teacher_match_rate",
        "mean_best_iou",
        "point_hit",
        "object_containment",
        "reasoning_grounding_alignment",
        "answer_before_grounding_rate",
    ]
    lines = ["# v27 Thinking-with-Grounding Eval Summary", ""]
    if args.teacher_jsonl:
        lines.append(f"Teacher pseudo targets: `{args.teacher_jsonl}`")
        lines.append("")
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join(["---"] * len(columns)) + "|")
    for row in summaries:
        lines.append("| " + " | ".join(fmt(row.get(col)) for col in columns) + " |")
    lines.extend(
        [
            "",
            "Notes:",
            "- `mean_best_iou`, `point_hit`, and `object_containment` are only populated when `--teacher-jsonl` covers the evaluated image paths.",
            f"- `coordinate_system`: `{args.output_coordinate_system}`. For Qwen2.5-VL abs modes, teacher 0-1000 boxes are converted to resized-image pixels before matching.",
            "- `reasoning_grounding_alignment` is an automatic proxy: part words mentioned in `<think>` outside tags that also appear in `<ref>` labels.",
            "- `answer_before_grounding_rate` flags outputs that start the final answer before any strict grounded evidence.",
        ]
    )
    args.summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved {args.summary_md}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
