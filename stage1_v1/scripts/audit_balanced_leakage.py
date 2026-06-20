#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path


def load_rows(path: Path):
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def images(row):
    value = row.get("images") or row.get("image") or row.get("image_path") or []
    return [value] if isinstance(value, str) else value


def canonical(path):
    return str(Path(path).expanduser().resolve())


def class_name(path):
    parts = Path(path).parts
    for marker in ("images", "images_train", "images_test"):
        if marker in parts:
            index = parts.index(marker)
            if index + 1 < len(parts):
                return parts[index + 1]
    return Path(path).parent.name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--seen", type=Path, required=True)
    parser.add_argument("--unseen", type=Path, required=True)
    args = parser.parse_args()

    train_rows = load_rows(args.train)
    train_paths = [canonical(path) for row in train_rows for path in images(row)]
    train_bird = [path for path in train_paths if "bird-200" in path or "bird_200" in path]
    train_names = Counter(Path(path).name for path in train_bird)

    report = {
        "train_rows": len(train_rows),
        "train_images": len(train_paths),
        "train_bird_images": len(train_bird),
        "train_bird_unique": len(set(train_bird)),
    }
    for split, path in (("seen", args.seen), ("unseen", args.unseen)):
        rows = load_rows(path)
        test_paths = [canonical(image) for row in rows for image in images(row)]
        exact = sorted(set(train_bird) & set(test_paths))
        basename = sorted(set(train_names) & {Path(item).name for item in test_paths})
        report[split] = {
            "rows": len(rows),
            "images": len(test_paths),
            "exact_path_overlap": len(exact),
            "basename_overlap": len(basename),
            "class_overlap": len({class_name(x) for x in train_bird} & {class_name(x) for x in test_paths}),
            "exact_examples": exact[:10],
            "basename_examples": basename[:10],
        }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
