# Ablation-only reward: released accuracy+format PLUS a LABEL-FREE grounding-format
# term. No teacher coordinates are used. Use this ONLY for the grounding-retention
# ablation row; primary fair runs must use the unmodified released cls_thinking.py.
import re
from typing import Any, Dict, List

# --- released components (kept byte-compatible with cls_thinking.py) ---------
def format_reward(response: str) -> float:
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    return 1.0 if re.fullmatch(pattern, response) else 0.0

def extract_answer_from_text(text):
    m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    return m.group(1).strip() if m else None

def accuracy_reward(response: str, ground_truth: str) -> float:
    ans = extract_answer_from_text(response)
    if ans is None:
        return 0.0
    return 1.0 if ground_truth.lower() in ans.lower() else 0.0

# --- new: label-free grounding-format term ----------------------------------
STRICT_REF_BOX_RE = re.compile(
    r"<ref>\s*[^<]+?\s*</ref>\s*<box>\s*\[\[\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*,"
    r"\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\]\]\s*</box>",
    re.IGNORECASE | re.DOTALL)
ANSWER_TAG_RE = re.compile(r"<answer\b", re.IGNORECASE)
BOX_COORDS_RE = re.compile(r"<box>\s*(\[\[.*?\]\])\s*</box>", re.IGNORECASE | re.DOTALL)

def grounding_signals(response: str) -> Dict[str, Any]:
    refs = list(STRICT_REF_BOX_RE.finditer(response))
    ans = ANSWER_TAG_RE.search(response)
    has_ref = len(refs) > 0
    answer_before = bool(ans and refs and ans.start() < refs[0].start())
    coords = [m.group(1) for m in BOX_COORDS_RE.finditer(response)]
    dup = len(coords) != len(set(coords))
    return {"has_ref_box": has_ref, "answer_before_grounding": answer_before,
            "duplicate_box": dup, "n_boxes": len(coords)}

def grounding_reward(response: str) -> float:
    s = grounding_signals(response)
    if not s["has_ref_box"]:
        return 0.0
    score = 0.5
    if not s["answer_before_grounding"]:
        score += 0.25
    if not s["duplicate_box"]:
        score += 0.25
    return score

def compute_score(reward_inputs: List[Dict[str, Any]],
                  format_weight: float = 0.1,
                  grounding_weight: float = 0.05) -> List[Dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch`.")
    acc_w = 1.0 - format_weight - grounding_weight
    out = []
    for ri in reward_inputs:
        raw = ri["response"]
        norm = re.sub(r"\s*(<|>|/)\s*", r"\1", raw)   # released normalization for fmt/acc
        fmt = format_reward(norm)
        acc = accuracy_reward(norm, ri["ground_truth"])
        gnd = grounding_reward(raw)                    # parse RAW (keep tag spacing)
        sig = grounding_signals(raw)
        out.append({
            "overall": acc_w * acc + format_weight * fmt + grounding_weight * gnd,
            "accuracy": acc, "format": fmt, "grounding": gnd,
            "answer_before_grounding": float(sig["answer_before_grounding"]),
            "duplicate_box": float(sig["duplicate_box"]),
            "n_boxes": float(sig["n_boxes"]),
        })
    return out
