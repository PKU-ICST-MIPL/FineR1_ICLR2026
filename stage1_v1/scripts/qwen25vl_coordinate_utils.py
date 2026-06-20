#!/usr/bin/env python3
"""Coordinate helpers for Qwen2.5-VL CUB grounding experiments.

Teacher boxes in the v27 pipeline are stored on a full-image 0..1000 grid.
Qwen2.5-VL grounding targets should use absolute pixel coordinates on the
image after the same resize policy used by training/inference.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from PIL import Image


CoordinateSystem = Literal["normalized_1000", "qwen25_abs", "llamafactory_qwen25_abs"]

QWEN_FACTOR = 28
LLAMAFACTORY_IMAGE_MAX_PIXELS = 768 * 768
LLAMAFACTORY_IMAGE_MIN_PIXELS = 32 * 32
QWEN_IMAGE_MIN_PIXELS = 56 * 56
QWEN_IMAGE_MAX_PIXELS = 14 * 14 * 4 * 1280


def round_by_factor(value: float, factor: int) -> int:
    return round(value / factor) * factor


def ceil_by_factor(value: float, factor: int) -> int:
    return math.ceil(value / factor) * factor


def floor_by_factor(value: float, factor: int) -> int:
    return math.floor(value / factor) * factor


def smart_resize(
    height: int,
    width: int,
    *,
    factor: int = QWEN_FACTOR,
    min_pixels: int = QWEN_IMAGE_MIN_PIXELS,
    max_pixels: int = QWEN_IMAGE_MAX_PIXELS,
) -> tuple[int, int]:
    """Match the Qwen2-VL/Qwen2.5-VL image processor resize rule."""

    if height <= 0 or width <= 0:
        raise ValueError(f"invalid image size: {width}x{height}")
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200, got {width}x{height}")
    if max_pixels < min_pixels:
        raise ValueError("max_pixels must be >= min_pixels")

    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, floor_by_factor(height / beta, factor))
        w_bar = max(factor, floor_by_factor(width / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return int(h_bar), int(w_bar)


def llamafactory_regularized_size(
    height: int,
    width: int,
    *,
    image_min_pixels: int = LLAMAFACTORY_IMAGE_MIN_PIXELS,
    image_max_pixels: int = LLAMAFACTORY_IMAGE_MAX_PIXELS,
) -> tuple[int, int]:
    """Mirror LLaMA-Factory Qwen2VLPlugin._preprocess_image area regularization."""

    if height <= 0 or width <= 0:
        raise ValueError(f"invalid image size: {width}x{height}")

    area = width * height
    if area > image_max_pixels:
        resize_factor = math.sqrt(image_max_pixels / area)
        width = int(width * resize_factor)
        height = int(height * resize_factor)

    area = width * height
    if area < image_min_pixels:
        resize_factor = math.sqrt(image_min_pixels / area)
        width = int(width * resize_factor)
        height = int(height * resize_factor)

    if min(width, height) < 28:
        width, height = max(width, 28), max(height, 28)

    if width / height > 200:
        width, height = height * 180, height
    if height / width > 200:
        width, height = width, width * 180

    return int(height), int(width)


@lru_cache(maxsize=8192)
def image_size(path: str) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.height, image.width


@lru_cache(maxsize=8192)
def resized_size_for_image(
    image_path: str,
    coordinate_system: CoordinateSystem = "llamafactory_qwen25_abs",
    image_min_pixels: int = LLAMAFACTORY_IMAGE_MIN_PIXELS,
    image_max_pixels: int = LLAMAFACTORY_IMAGE_MAX_PIXELS,
    qwen_min_pixels: int = QWEN_IMAGE_MIN_PIXELS,
    qwen_max_pixels: int = QWEN_IMAGE_MAX_PIXELS,
    factor: int = QWEN_FACTOR,
) -> tuple[int, int]:
    """Return final resized (height, width) for the selected coordinate system."""

    height, width = image_size(str(Path(image_path)))
    if coordinate_system == "normalized_1000":
        return 1000, 1000
    if coordinate_system == "llamafactory_qwen25_abs":
        height, width = llamafactory_regularized_size(
            height,
            width,
            image_min_pixels=image_min_pixels,
            image_max_pixels=image_max_pixels,
        )
    elif coordinate_system != "qwen25_abs":
        raise ValueError(f"unknown coordinate system: {coordinate_system}")

    return smart_resize(
        height,
        width,
        factor=factor,
        min_pixels=qwen_min_pixels,
        max_pixels=qwen_max_pixels,
    )


def clamp_box(box: Any, width: int, height: int) -> list[int] | None:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except Exception:
        return None
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = max(0, min(width - 1, int(round(x1))))
    y1 = max(0, min(height - 1, int(round(y1))))
    x2 = max(x1 + 1, min(width, int(round(x2))))
    y2 = max(y1 + 1, min(height, int(round(y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def box_1000_to_resized_pixels(box: Any, resized_width: int, resized_height: int) -> list[int] | None:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except Exception:
        return None
    return clamp_box(
        [
            x1 / 1000.0 * resized_width,
            y1 / 1000.0 * resized_height,
            x2 / 1000.0 * resized_width,
            y2 / 1000.0 * resized_height,
        ],
        resized_width,
        resized_height,
    )


def resized_pixels_to_1000(box: Any, resized_width: int, resized_height: int) -> list[int] | None:
    pixel_box = clamp_box(box, resized_width, resized_height)
    if pixel_box is None:
        return None
    return clamp_box(
        [
            pixel_box[0] / resized_width * 1000,
            pixel_box[1] / resized_height * 1000,
            pixel_box[2] / resized_width * 1000,
            pixel_box[3] / resized_height * 1000,
        ],
        1000,
        1000,
    )


def resized_pixels_to_original(box: Any, resized_width: int, resized_height: int, original_width: int, original_height: int) -> list[int] | None:
    pixel_box = clamp_box(box, resized_width, resized_height)
    if pixel_box is None:
        return None
    return clamp_box(
        [
            pixel_box[0] / resized_width * original_width,
            pixel_box[1] / resized_height * original_height,
            pixel_box[2] / resized_width * original_width,
            pixel_box[3] / resized_height * original_height,
        ],
        original_width,
        original_height,
    )


def convert_teacher_box_to_output(
    box_1000: Any,
    image_path: str,
    coordinate_system: CoordinateSystem = "llamafactory_qwen25_abs",
    *,
    image_min_pixels: int = LLAMAFACTORY_IMAGE_MIN_PIXELS,
    image_max_pixels: int = LLAMAFACTORY_IMAGE_MAX_PIXELS,
    qwen_min_pixels: int = QWEN_IMAGE_MIN_PIXELS,
    qwen_max_pixels: int = QWEN_IMAGE_MAX_PIXELS,
    factor: int = QWEN_FACTOR,
) -> list[int] | None:
    if coordinate_system == "normalized_1000":
        return clamp_box(box_1000, 1000, 1000)
    resized_height, resized_width = resized_size_for_image(
        image_path,
        coordinate_system,
        image_min_pixels,
        image_max_pixels,
        qwen_min_pixels,
        qwen_max_pixels,
        factor,
    )
    return box_1000_to_resized_pixels(box_1000, resized_width, resized_height)


def coordinate_bounds_for_image(
    image_path: str,
    coordinate_system: CoordinateSystem = "llamafactory_qwen25_abs",
    *,
    image_min_pixels: int = LLAMAFACTORY_IMAGE_MIN_PIXELS,
    image_max_pixels: int = LLAMAFACTORY_IMAGE_MAX_PIXELS,
    qwen_min_pixels: int = QWEN_IMAGE_MIN_PIXELS,
    qwen_max_pixels: int = QWEN_IMAGE_MAX_PIXELS,
    factor: int = QWEN_FACTOR,
) -> tuple[int, int]:
    resized_height, resized_width = resized_size_for_image(
        image_path,
        coordinate_system,
        image_min_pixels,
        image_max_pixels,
        qwen_min_pixels,
        qwen_max_pixels,
        factor,
    )
    return resized_width, resized_height


def regularize_pil_for_llamafactory(
    image_path: str,
    *,
    image_min_pixels: int = LLAMAFACTORY_IMAGE_MIN_PIXELS,
    image_max_pixels: int = LLAMAFACTORY_IMAGE_MAX_PIXELS,
) -> Image.Image:
    """Load and resize a PIL image like LLaMA-Factory before HF processing."""

    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    height, width = image.height, image.width
    regularized_h, regularized_w = llamafactory_regularized_size(
        height,
        width,
        image_min_pixels=image_min_pixels,
        image_max_pixels=image_max_pixels,
    )
    if (regularized_w, regularized_h) != image.size:
        image = image.resize((regularized_w, regularized_h))
    return image


def processor_patch_size(processor: Any) -> int:
    image_processor = getattr(processor, "image_processor", processor)
    patch_size = getattr(image_processor, "patch_size", None)
    if patch_size is None:
        patch_size = getattr(processor, "patch_size", None)
    if isinstance(patch_size, (list, tuple)):
        patch_size = patch_size[0]
    if patch_size is None:
        raise ValueError("processor has no image patch_size")
    return int(patch_size)


def processor_merge_size(processor: Any) -> int:
    image_processor = getattr(processor, "image_processor", processor)
    merge_size = getattr(image_processor, "merge_size", None)
    if merge_size is None:
        merge_size = getattr(image_processor, "spatial_merge_size", None)
    if merge_size is None:
        merge_size = getattr(processor, "spatial_merge_size", None)
    return int(merge_size or 1)


def _grid_to_list(grid: Any) -> list[int]:
    if hasattr(grid, "detach"):
        grid = grid.detach().cpu()
    if hasattr(grid, "tolist"):
        grid = grid.tolist()
    if isinstance(grid, list) and grid and isinstance(grid[0], list):
        grid = grid[0]
    if not isinstance(grid, (list, tuple)) or len(grid) < 3:
        raise ValueError(f"invalid image_grid_thw: {grid!r}")
    return [int(grid[0]), int(grid[1]), int(grid[2])]


def grid_bounds_from_inputs(inputs: dict[str, Any], processor: Any) -> tuple[int, int]:
    """Return actual resized (width, height) from processor output image_grid_thw."""

    grid = inputs.get("image_grid_thw")
    if grid is None:
        raise ValueError("processor inputs did not include image_grid_thw")
    _, grid_h, grid_w = _grid_to_list(grid)
    patch_size = processor_patch_size(processor)
    return grid_w * patch_size, grid_h * patch_size


def processor_image_for_coordinate_system(
    image_path: str,
    coordinate_system: CoordinateSystem = "llamafactory_qwen25_abs",
    *,
    image_min_pixels: int = LLAMAFACTORY_IMAGE_MIN_PIXELS,
    image_max_pixels: int = LLAMAFACTORY_IMAGE_MAX_PIXELS,
) -> Image.Image:
    if coordinate_system == "llamafactory_qwen25_abs":
        return regularize_pil_for_llamafactory(
            image_path,
            image_min_pixels=image_min_pixels,
            image_max_pixels=image_max_pixels,
        )
    return Image.open(image_path).convert("RGB")


def processor_grid_bounds_for_image(
    processor: Any,
    image_path: str,
    coordinate_system: CoordinateSystem = "llamafactory_qwen25_abs",
    *,
    image_min_pixels: int = LLAMAFACTORY_IMAGE_MIN_PIXELS,
    image_max_pixels: int = LLAMAFACTORY_IMAGE_MAX_PIXELS,
) -> tuple[int, int]:
    """Return actual processor resized (width, height) for one image."""

    image = processor_image_for_coordinate_system(
        image_path,
        coordinate_system,
        image_min_pixels=image_min_pixels,
        image_max_pixels=image_max_pixels,
    )
    image_processor = getattr(processor, "image_processor", processor)
    inputs = image_processor(images=[image], return_tensors="pt")
    return grid_bounds_from_inputs(dict(inputs), processor)


def direct_processor_grid_bounds_for_image(
    processor: Any,
    image_path: str,
    *,
    image_min_pixels: int = LLAMAFACTORY_IMAGE_MIN_PIXELS,
    image_max_pixels: int = LLAMAFACTORY_IMAGE_MAX_PIXELS,
) -> tuple[int, int]:
    """Return processor resized (width, height) when min/max are passed directly."""

    image = Image.open(image_path).convert("RGB")
    image_processor = getattr(processor, "image_processor", processor)
    try:
        inputs = image_processor(
            images=[image],
            return_tensors="pt",
            min_pixels=image_min_pixels,
            max_pixels=image_max_pixels,
        )
    except TypeError:
        inputs = image_processor(images=[image], return_tensors="pt")
    return grid_bounds_from_inputs(dict(inputs), processor)
