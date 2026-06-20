#!/usr/bin/env python3
"""Validate a LLaMA-Factory full-FT Stage1 dir before an expensive EasyR1/TAPO run.

Checks: HF dir completeness, model_type==qwen2_5_vl, sharded safetensors index
references all present shards, processor pixel bounds, and (critically) that the
processor min/max pixels MATCH the pixels the RL stage will use. A mismatch
silently miscalibrates the resized-absolute grounding coordinates.
"""
import argparse, json, sys
from pathlib import Path

REQUIRED = ["config.json", "generation_config.json", "tokenizer_config.json",
            "preprocessor_config.json"]
ONE_OF = [["model.safetensors", "model.safetensors.index.json"],
          ["tokenizer.json", "vocab.json"]]

def fail(msg, errs): errs.append(msg)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rl-max-pixels", type=int, default=None,
                    help="max_pixels the RL/eval stage will use; warns on mismatch")
    ap.add_argument("--rl-min-pixels", type=int, default=None)
    ap.add_argument("--set-processor-pixels", action="store_true",
                    help="write the supplied RL min/max into preprocessor_config.json before validating")
    args = ap.parse_args()
    d = Path(args.ckpt); errs, warns = [], []
    if not d.is_dir():
        print(f"FAIL: not a dir: {d}"); sys.exit(2)

    for f in REQUIRED:
        if not (d / f).exists(): fail(f"missing {f}", errs)
    for group in ONE_OF:
        if not any((d / g).exists() for g in group):
            fail(f"missing all of {group}", errs)

    # model_type
    cfg = {}
    if (d / "config.json").exists():
        cfg = json.loads((d / "config.json").read_text())
        mt = cfg.get("model_type")
        if mt != "qwen2_5_vl":
            fail(f"model_type={mt!r}, expected qwen2_5_vl", errs)

    # safetensors shards all present
    idx = d / "model.safetensors.index.json"
    if idx.exists():
        shards = set(json.loads(idx.read_text()).get("weight_map", {}).values())
        for s in shards:
            if not (d / s).exists(): fail(f"index references missing shard {s}", errs)

    # no leftover LoRA adapter (full-FT must be merged)
    if (d / "adapter_config.json").exists():
        fail("adapter_config.json present -> looks like LoRA, not merged full-FT", errs)

    # processor pixel bounds + RL alignment
    pp = d / "preprocessor_config.json"
    if pp.exists():
        p = json.loads(pp.read_text())
        if args.set_processor_pixels:
            if args.rl_max_pixels is None or args.rl_min_pixels is None:
                fail("--set-processor-pixels requires --rl-max-pixels and --rl-min-pixels", errs)
            else:
                p["max_pixels"] = args.rl_max_pixels
                p["min_pixels"] = args.rl_min_pixels
                size = p.setdefault("size", {})
                size["longest_edge"] = args.rl_max_pixels
                size["shortest_edge"] = args.rl_min_pixels
                tmp = pp.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(p, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                tmp.replace(pp)
                print(f"updated processor pixels: min={args.rl_min_pixels} max={args.rl_max_pixels}")
        mn = p.get("min_pixels") or (p.get("size") or {}).get("min_pixels")
        mx = p.get("max_pixels") or (p.get("size") or {}).get("max_pixels")
        print(f"processor pixels: min={mn} max={mx}")
        if args.rl_max_pixels is not None and mx not in (None, args.rl_max_pixels):
            fail(f"processor max_pixels={mx} != RL max_pixels={args.rl_max_pixels} "
                 f"(miscalibrates resized-absolute boxes)", errs)
        if args.rl_min_pixels is not None and mn not in (None, args.rl_min_pixels):
            fail(f"processor min_pixels={mn} != RL min_pixels={args.rl_min_pixels}", errs)
        if mx is None:
            warns.append("processor has no explicit max_pixels; eval/RL will use library "
                         "defaults -> pin them explicitly in the RL launcher and eval")

    for w in warns: print(f"WARN: {w}")
    if errs:
        print("PREFLIGHT FAILED:")
        for e in errs: print("  -", e)
        sys.exit(14)
    print("PREFLIGHT OK:", d)

if __name__ == "__main__":
    main()
