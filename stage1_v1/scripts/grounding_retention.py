#!/usr/bin/env python3
"""Measure grounding retention from RL rollouts or eval outputs (label-free).

Parser rule matches the acceptance checklist: the final answer is taken AFTER
</think>. Works on JSON list / JSONL with a text field (default: raw_output, then
model_output, then response). Emits strict_ref_box, answer_before_grounding,
duplicate_box, avg_boxes for decay tracking at any saved checkpoint/step.
"""
import argparse, json, re, sys
from pathlib import Path

STRICT_REF_BOX_RE = re.compile(
    r"<ref>\s*[^<]+?\s*</ref>\s*<box>\s*\[\[\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*,"
    r"\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\]\]\s*</box>", re.IGNORECASE | re.DOTALL)
REF_BOX_TAG_RE = re.compile(
    r"<ref>\s*[^<]+?\s*</ref>\s*<box>.*?</box>", re.IGNORECASE | re.DOTALL)
ANSWER_TAG_RE = re.compile(r"<answer\b", re.IGNORECASE)
BOX_RE = re.compile(r"<box>\s*(\[\[.*?\]\])\s*</box>", re.IGNORECASE | re.DOTALL)

def final_answer(text):
    m = re.search(r"</think>", text, re.IGNORECASE)
    tail = text[m.end():] if m else text
    a = list(re.finditer(r"<answer>\s*(.*?)\s*</answer>", tail, re.IGNORECASE | re.DOTALL))
    if a:
        return re.sub(r"\s+", " ", a[-1].group(1)).strip()
    a = list(re.finditer(r"<answer>\s*(.*?)\s*</answer>", text, re.IGNORECASE | re.DOTALL))
    return re.sub(r"\s+", " ", a[-1].group(1)).strip() if a else ""

def row_signals(text):
    refs = list(STRICT_REF_BOX_RE.finditer(text))
    tagged_refs = list(REF_BOX_TAG_RE.finditer(text))
    ans = ANSWER_TAG_RE.search(text)
    coords = [m.group(1) for m in BOX_RE.finditer(text)]
    return {
        "ref_box_tag": bool(tagged_refs),
        "strict_ref_box": bool(refs),
        "answer_before_grounding": bool(ans and refs and ans.start() < refs[0].start())
                                   or (bool(ans) and not refs),
        "duplicate_box": len(coords) != len(set(coords)) and len(coords) > 0,
        "n_boxes": len(coords),
    }

def load_rows(path):
    p = Path(path); txt = p.read_text(encoding="utf-8")
    if p.suffix == ".jsonl":
        return [json.loads(l) for l in txt.splitlines() if l.strip()]
    data = json.loads(txt)
    if isinstance(data, dict):
        data = data.get("results") or data.get("outputs") or [data]
    return data

def get_text(row, fields):
    for f in fields:
        v = row.get(f)
        if isinstance(v, str) and v:
            return v
    return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--text-fields", default="raw_output,model_output,response,output")
    ap.add_argument("--step", default=None, help="optional step/checkpoint tag for logging")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    fields = args.text_fields.split(",")
    rows = load_rows(args.inp)
    sigs = [row_signals(get_text(r, fields)) for r in rows]
    n = len(sigs) or 1
    summary = {
        "step": args.step, "n": len(sigs),
        "ref_box_tag_rate": sum(s["ref_box_tag"] for s in sigs) / n,
        "strict_ref_box": sum(s["strict_ref_box"] for s in sigs) / n,
        "answer_before_grounding": sum(s["answer_before_grounding"] for s in sigs) / n,
        "duplicate_box_rate": sum(s["duplicate_box"] for s in sigs) / n,
        "avg_boxes": sum(s["n_boxes"] for s in sigs) / n,
    }
    tagged = sum(s["ref_box_tag"] for s in sigs)
    summary["well_formed_given_tag"] = (
        sum(s["strict_ref_box"] for s in sigs) / tagged if tagged else None
    )
    js = json.dumps(summary, indent=2)
    print(js)
    if args.out:
        Path(args.out).write_text(js + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
