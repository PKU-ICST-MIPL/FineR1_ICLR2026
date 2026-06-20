# Stage1 v1: Thinking with Grounding for Fine-Grained Bird Recognition

**Status:** completed Stage1 candidate and evaluation package, prepared for controlled bird-only Stage2 experiments.

**Base model:** Qwen2.5-VL-3B-Instruct  
**Primary checkpoint:** `FineR1-3B-Stage1-Grounded-v1` (Hugging Face link will be recorded here after the upload job completes)  
**Date:** 2026-06-20

## Executive Summary

This branch adds a bird-focused Stage1 variant to Fine-R1 in which the model is trained to produce a localized visual cue before its final fine-grained classification answer:

```text
<think>
1. <ref>part</ref><box>[[x1,y1,x2,y2]]</box>: visible attribute.
These localized cues support the classification.
</think>
<answer>class name</answer>
```

The main result is not a claim of solved part grounding. The defensible result is narrower:

> A rigid localized-cue Stage1 objective creates a prompt-robust grounding habit and improves CUB closed-world unseen-category selection relative to an internally matched plain-CoT Stage1 checkpoint, without reducing seen-category accuracy.

On a deterministic option-shuffled control, the selected grounded checkpoint obtains **65.98% seen / 55.95% unseen**, compared with **65.69% / 51.49%** for the plain-CoT Stage1 control. On the paired unseen set, the grounded checkpoint uniquely solves 299 examples while the plain checkpoint uniquely solves 196 (`p=4.55e-6`, continuity-corrected McNemar test).

The released Fine-R1 closed-world JSONL files place the correct option first in every bird example. We therefore report both the released protocol for paper alignment and a deterministic shuffled-option control for scientific interpretation.

## Research Question

Fine-R1 improves fine-grained visual recognition using CoT SFT followed by TAPO. Stage1 v1 asks whether an explicit spatial evidence interface can provide a better initialization for later policy optimization:

1. Can Qwen2.5-VL learn to emit localized cues before answering?
2. Does that behavior survive prompts that do not explicitly demand boxes?
3. Does the initialization improve seen/unseen recognition under a controlled benchmark?
4. Does the behavior survive answer-reward-only Stage2 optimization?

This release addresses the first three questions at Stage1. The fourth is the next controlled Stage2 experiment.

## Model Variants

| Variant | Purpose | Training summary |
|---|---|---|
| Plain | Internal CoT SFT control | Plain CUB CoT, 3 epochs, low learning rate |
| Rigid | Selected grounded Stage1 v1 | 465 strict CUB grounded rows, 8 epochs, LR `5e-6`; frozen vision tower and projector; full LLM update |
| Balanced | De-specialization diagnostic | 465 grounded CUB rows + 404 released six-domain replay rows, 1:1, 4 epochs, LR `3e-6`, AdamW |

The Rigid checkpoint is selected for Stage2. Balanced is retained only as a diagnostic because replay diluted prompt-invariant grounding and amplified a closed-world option-position shortcut.

## Data and Target Construction

### Evidence policy

- Use strict teacher evidence whenever available.
- Require at least one grounded cue before the answer.
- Permit one or two cues; do not force a second cue when it cannot be independently supported.
- Canonicalize the target surface form to reduce protocol variance.
- Reject empty cue slots and malformed boxes.
- Keep the final class declaration outside `<think>`.

The student normally emits one box. This is treated as a design outcome rather than an automatic failure: forcing a second cue can introduce post-hoc or contradictory evidence.

### Coordinate convention

Qwen2.5-VL uses absolute pixel coordinates on the processor's `smart_resize` canvas. Teacher boxes originally represented in a normalized `0..1000` space are converted per image into resized-image absolute coordinates.

The complete pipeline pins:

```text
min_pixels = 1024
max_pixels = 589824
patch_size = 14
merge_size = 2
```

The following checks were completed:

- teacher normalized box -> resized-absolute overlay;
- processor `image_grid_thw` canvas agreement;
- base-model bbox convention round trip;
- identical conversion bounds in train, inference, scoring, and visualization;
- export-time processor metadata validation.

The selected checkpoint's `preprocessor_config.json` has been corrected to persist `1024/589824`, so Stage2 does not silently load the base model's larger default pixel budget.

## Training Configuration

The primary Rigid recipe updates the language model while preserving the pretrained visual representation:

```text
model                    Qwen2.5-VL-3B-Instruct
vision tower             frozen
multimodal projector     frozen
language model           trainable (full update)
epochs                    8
learning rate             5e-6
coordinate space          resized-image absolute pixels
interaction format        single turn
```

Single-turn training is intentional because the released Stage2 rollout is single-turn. Earlier multi-turn checkpoints failed when evaluated using a single-turn prompt.

## Evaluation Protocol

Stage1 v1 uses four complementary evaluation layers.

### 1. Protocol and grounding diagnostic (50 examples)

Measures answer accuracy, strict tag retention, box count, duplicate boxes, answer-before-grounding, pseudo-IoU, point hit, and containment.

Rigid result:

| Metric | Value |
|---|---:|
| Exact answer accuracy | 0.52 |
| Strict-relaxed answer accuracy | 0.56 |
| `<think>` / `<answer>` format | 1.00 |
| Strict `<ref><box>` retention | 1.00 |
| Answer before grounding | 0.00 |
| Duplicate-box row rate | 0.00 |
| Average loose boxes | 1.02 |
| Mean pseudo-box IoU | 0.191 |

The pseudo-IoU target is not a human part annotation. This evaluation certifies protocol retention and spatial sanity, not discriminative part correctness.

### 2. Prompt x initialization probe

| Initialization | Prompt | Strict grounding | Well formed given tag |
|---|---|---:|---:|
| Plain | Generic | 0.00 | N/A |
| Plain | Explicit grounded | 0.68 | 0.694 |
| Rigid | Generic | **0.88** | **1.00** |
| Rigid | Explicit grounded | **1.00** | **1.00** |
| Balanced | Generic | 0.04 | 1.00 |
| Balanced | Explicit grounded | 1.00 | 1.00 |

This establishes both initialization and prompt effects. Rigid grounding is learned in the weights, but an explicit prompt can also induce partial grounding from a plain initialization.

### 3. Released Fine-R1 matched evaluation

All three models were evaluated using the same released bird splits, prompt templates, `max_new_tokens=2048`, and `1024/589824` processor bounds.

| Model | Closed seen | Closed unseen | Open seen semantic | Open unseen semantic |
|---|---:|---:|---:|---:|
| Plain | 70.17 | 55.69 | 64.58 | **50.08** |
| Rigid | 69.65 | 61.06 | **88.12** | 48.20 |
| Balanced | **86.13** | **80.61** | 77.42 | 49.78 |

Open-world semantic is the released SigLIP-relative text similarity metric. It is not classification accuracy. Text inclusion (TI) remains a diagnostic only.

### 4. Deterministic shuffled-option control

Audit result for the released CUB closed-world files:

```text
bird_seen:   3483 / 3483 correct answers at option index 0
bird_unseen: 2311 / 2311 correct answers at option index 0
```

The released evaluator does not shuffle options at runtime. A model that always chooses the first option can therefore obtain 100%. To isolate visual selection from position preference, we deterministically shuffle options per image using seed `20260620` while retaining all other evaluation settings.

| Model | Shuffled seen | Shuffled unseen |
|---|---:|---:|
| Plain | 65.69 | 51.49 |
| Rigid | **65.98** | **55.95** |
| Balanced | 54.49 | running at the time of this v1 report |

Paired Plain vs Rigid comparison:

| Split | Plain-only correct | Rigid-only correct | McNemar p |
|---|---:|---:|---:|
| Seen | 335 | 345 | 0.730 |
| Unseen | 196 | 299 | `4.55e-6` |

Rigid is statistically tied on seen categories and significantly better on unseen categories relative to the matched Plain Stage1 control.

## Comparison with the Paper's Qwen2.5-VL-3B Row

The paper reports the off-the-shelf Qwen2.5-VL-3B at 65.40 seen / 58.98 unseen closed-world Bird accuracy. Our current shuffled Rigid result is 65.98 / 55.95.

This means:

- Rigid improves materially over our own Plain CoT SFT control.
- Rigid preserves paper-level seen performance.
- Rigid has not yet demonstrated superiority over the paper's raw-Qwen unseen result.
- A raw Qwen2.5-VL-3B baseline under the same three shuffled seeds remains a required calibration experiment.

The paper's “unseen” classes are unseen to Fine-R1 fine-tuning, not necessarily unseen during Qwen pretraining.

## What Is and Is Not Established

### Supported claims

- The Qwen2.5-VL resized-absolute coordinate pipeline is internally consistent.
- Rigid Stage1 SFT creates a prompt-robust localized-cue generation habit.
- Rigid improves shuffled closed-world unseen selection over the matched Plain Stage1 control without harming seen accuracy.
- The released closed-world bird benchmark has a correct-option position bias that materially affects some fine-tuned checkpoints.

### Unsupported claims

- The generated boxes are accurate human-level part annotations.
- The localized cue is causally responsible for the answer.
- Stage1 v1 improves open-world unseen naming.
- Stage1 v1 exceeds the final Fine-R1 Stage2 model.
- Grounding will survive unmodified answer-reward-only Stage2 training.

## Selected Checkpoint and Stage2 Readiness

The selected checkpoint is a complete Hugging Face model directory containing three safetensor shards, model index, configuration, tokenizer, processor, generation configuration, and chat template.

Before release it passed:

```text
processor min/max validation       PASS (1024 / 589824)
Hugging Face export structure      PASS
Stage2 DAPO config dry-run         PASS
single-GPU DAPO update smoke       PASS for the prepared pipeline
```

See [STAGE2_HANDOFF.md](STAGE2_HANDOFF.md) for the controlled Stage2 matrix and exact launch expectations.

## Code Map

| File | Purpose |
|---|---|
| `scripts/qwen25vl_coordinate_utils.py` | Coordinate conversion and processor-grid utilities |
| `scripts/base_qwen25vl_bbox_convention_roundtrip_v27c.py` | Base checkpoint convention check |
| `scripts/rebuild_grounded_rigid.py` | Canonical strict target rebuilding |
| `scripts/run_v27c_abs_rigid_top2_retention.sh` | Primary Stage1 v1 training recipe |
| `scripts/score_cub_eval_thinking_grounded_v27.py` | Protocol and spatial diagnostics |
| `scripts/grounding_retention.py` | Stage1/Stage2 retention metrics |
| `scripts/official_bird_closed_eval_fast.py` | Resumable official-aligned closed evaluation with optional option shuffle |
| `scripts/run_shuffled_closed_matrix.sh` | Plain/Rigid/Balanced control matrix |
| `scripts/audit_balanced_leakage.py` | Train/eval image and class overlap audit |
| `scripts/preflight_stage1_export.py` | HF/Stage2 export and pixel-budget validation |
| `scripts/run_stage2_variant.sh` | Controlled Stage2 launcher and source-truth guards |

## Reproducibility Notes

- Do not compare fast-512 and matched-2048 outputs as if they were the same protocol.
- Always pin processor pixels across target conversion, SFT, rollout, inference, and visualization.
- Report released-order and shuffled closed-world results separately.
- Use the full six-domain Stage2 data for comparisons to paper-wide averages. Bird-only Stage2 supports only Bird-column claims.
- The released source currently exposes the augmented TAPO perception path but does not provide independently executable `intra` and `inter` enums. Do not label guessed configuration switches as paper reproductions.

## Next Experiments

1. Evaluate raw Qwen2.5-VL-3B under three deterministic shuffled-option seeds.
2. Run bird-only Plain vs Rigid Stage2 DAPO and TAPO with the released generic prompt and reward.
3. Track strict grounding retention at every saved checkpoint.
4. Run an explicit-grounding prompt as a separate retention ablation.
5. Add human or CUB part-level correctness and causal masking tests before making explanation-faithfulness claims.

