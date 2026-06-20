#!/usr/bin/env python3
"""
rebuild_grounded_rigid.py
=========================
Centerpiece of the protocol-retention recipe. Takes your existing box-top2
grounded SFT json and rewrites every assistant target into a LOW-VARIANCE,
GROUNDING-DENSE canonical form so the student reliably learns to emit
<ref>part</ref><box>[[...]]</box> BEFORE <answer>.

Why this raises strict_ref_box (your 0.28 -> target >0.8):
  - A rigid protocol is a high-variance generation behavior. With ~480 rows it is
    only learnable if the *surface form is near-constant*. Unconstrained prose
    makes the protocol harder to retain. This script removes that variance:
      * one or two faithful grounded evidences per row (TARGET_K is a cap),
        always before <answer>;
      * one fixed sentence frame per evidence line;
      * one fixed closing sentence;
      * rows that cannot supply MIN_K strict evidences are dropped (reported),
        NOT padded with fabricated boxes.
  - Coordinates are preserved verbatim (you already fixed them to resized-abs);
    boxes are only de-duplicated and integer-normalized.

Coordinate space is NOT touched. Box-area-fraction stats are printed so you can
confirm targets are part-sized (your train dist was ~4% mean, p90 ~8%).

Usage:
  python rebuild_grounded_rigid.py \
    --in  data/Fine-R1-Stage1-cub-v27c-abs-strict-grounded-top2.json \
    --out data/Fine-R1-Stage1-cub-v27c-abs-strict-grounded-top2-RIGID.json \
    --target-k 2 --min-k 1 --image-wh-key image_wh
"""
import argparse
import ast
import json
import re
import sys
from collections import Counter

REF_BOX_RE = re.compile(
    r"<ref>\s*(?P<part>.*?)\s*</ref>\s*<box>\s*(?P<coords>\[\[.*?\]\])\s*</box>", re.DOTALL)
THINK_RE = re.compile(r"<think>(?P<body>.*?)</think>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>\s*(?P<ans>.*?)\s*</answer>", re.DOTALL)

# fixed frames -> these constants ARE the habit signal. Keep them constant.
EVIDENCE_FRAME = "{i}. <ref>{part}</ref><box>{box}</box>: {attr}."
# Keep the claim at the level supported by the current object/part diagnostics.
CLOSING = "These localized cues support the classification."
FILLER = re.compile(r"(the\s+\w+\s+evidence\s+is|supported\s+by|localized\s+at|"
                    r"is\s+a|is\s+an|,?\s*shown\s+(?:by|at)|^\s*\d+\.\s*)", re.IGNORECASE)


def detect_layout(rec):
    if "conversations" in rec:
        get = lambda r: next((t["value"] for t in r["conversations"]
                              if t.get("from") in ("gpt", "assistant")), None)
        setv = lambda r, v: [t.__setitem__("value", v) for t in r["conversations"]
                             if t.get("from") in ("gpt", "assistant")]
        usr = lambda r: next((t["value"] for t in r["conversations"]
                             if t.get("from") in ("human", "user")), "")
        return get, setv, usr
    if "messages" in rec:
        get = lambda r: next((t["content"] for t in r["messages"]
                             if t.get("role") == "assistant"), None)
        setv = lambda r, v: [t.__setitem__("content", v) for t in r["messages"]
                            if t.get("role") == "assistant"]
        usr = lambda r: next((t["content"] for t in r["messages"]
                             if t.get("role") == "user"), "")
        return get, setv, usr
    for f in ("output", "response", "assistant"):
        if f in rec:
            return (lambda r, f=f: r[f]), (lambda r, v, f=f: r.__setitem__(f, v)), \
                   (lambda r: r.get("instruction", r.get("query", "")))
    raise ValueError("Unknown record layout")


def extract_evidences(think_body):
    """Return ordered list of (part, [x1,y1,x2,y2], attribute_text)."""
    out, last = [], 0
    for m in REF_BOX_RE.finditer(think_body):
        try:
            box = ast.literal_eval(m.group("coords"))
        except Exception:
            last = m.end(); continue
        if box and isinstance(box[0], list):
            box = box[0]
        if not box or len(box) != 4:
            last = m.end(); continue
        # attribute: clean clause in the window before this match
        window = think_body[last:m.start()]
        clause = re.split(r"[.\n]", window)[-1]
        clause = FILLER.sub(" ", clause)
        clause = re.sub(r"<[^>]+>", " ", clause)
        attr = re.sub(r"\s+", " ", clause).strip(" ,;:").strip()
        if not attr or len(attr) < 3:
            attr = m.group("part").strip()
        out.append((m.group("part").strip(), [int(round(c)) for c in box], attr))
        last = m.end()
    return out


def box_str(b):
    return "[[%d,%d,%d,%d]]" % (b[0], b[1], b[2], b[3])


def area_frac(b, wh):
    if not wh:
        return None
    W, H = wh
    return max(0, (b[2] - b[0])) * max(0, (b[3] - b[1])) / float(W * H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--target-k", type=int, default=2)
    ap.add_argument("--min-k", type=int, default=1)
    ap.add_argument("--image-wh-key", default=None,
                    help="record key holding [W,H] of the RESIZED canvas, for area stats")
    args = ap.parse_args()

    data = json.load(open(args.inp, encoding="utf-8"))
    if not isinstance(data, list):
        sys.exit("expected top-level list")
    get, setv, usr = detect_layout(data[0])

    out, stats = [], Counter()
    k_hist, area_fracs, leaks = Counter(), [], 0

    for rec in data:
        text = get(rec)
        if text is None:
            stats["no_assistant"] += 1; continue
        tm, am = THINK_RE.search(text), ANSWER_RE.search(text)
        if not tm or not am:
            stats["no_think_or_answer"] += 1; continue
        evs = extract_evidences(tm.group("body"))
        # dedupe boxes (keep first), then clip to target_k
        seen, uniq = set(), []
        for part, box, attr in evs:
            key = tuple(box)
            if key in seen:
                continue
            seen.add(key); uniq.append((part, box, attr))
        if len(uniq) < args.min_k:
            stats["dropped_below_min_k"] += 1; continue
        uniq = uniq[:args.target_k]
        k_hist[len(uniq)] += 1

        wh = rec.get(args.image_wh_key) if args.image_wh_key else None
        lines = []
        single = len(uniq) == 1
        for i, (part, box, attr) in enumerate(uniq, 1):
            if single:
                lines.append("<ref>%s</ref><box>%s</box>: %s." % (part, box_str(box), attr))
            else:
                lines.append(EVIDENCE_FRAME.format(i=i, part=part, box=box_str(box), attr=attr))
            af = area_frac(box, wh)
            if af is not None:
                area_fracs.append(af)
        new_think = "<think>\n" + "\n".join(lines) + "\n" + CLOSING + "\n</think>"
        new_text = "%s\n<answer>%s</answer>" % (new_think, am.group("ans").strip())
        setv(rec, new_text)

        if am.group("ans").strip().lower() in usr(rec).lower():
            leaks += 1
        out.append(rec); stats["kept"] += 1

    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("=== rebuild_grounded_rigid ===")
    print("input rows:            %d" % len(data))
    print("kept rows:             %d" % stats["kept"])
    print("dropped (below min-k): %d" % stats["dropped_below_min_k"])
    print("dropped (no fields):   %d" % (stats["no_assistant"] + stats["no_think_or_answer"]))
    print("k distribution:        %s" % dict(sorted(k_hist.items())))
    print("input_label_leaks:     %d  (MUST be 0)" % leaks)
    if area_fracs:
        area_fracs.sort()
        n = len(area_fracs)
        print("box area frac mean:    %.3f%%" % (100 * sum(area_fracs) / n))
        print("box area frac p90:     %.3f%%" % (100 * area_fracs[int(0.9 * n)]))
        oob = sum(1 for a in area_fracs if a < 0.005 or a > 0.20)
        print("boxes outside 0.5-20%% (suspect part size): %d / %d" % (oob, n))
    print("wrote: %s" % args.out)
    print("\nNOTE: register in LLaMA-Factory data/dataset_info.json, e.g.:")
    print('  "fine_r1_cub_box_top2_rigid": {')
    print('    "file_name": "%s",' % args.out.split("/")[-1])
    print('    "formatting": "sharegpt",')
    print('    "columns": {"messages": "conversations"},')
    print('    "tags": {"role_tag": "from", "content_tag": "value",')
    print('             "user_tag": "human", "assistant_tag": "gpt"}}')


if __name__ == "__main__":
    main()
