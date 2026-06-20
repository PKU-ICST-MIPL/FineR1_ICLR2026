# Stage2 Handoff for Stage1 Grounded v1

This document is the execution contract for the bird-only Stage2 comparison. It is intentionally stricter than a generic checkpoint handoff because the experiment must separate initialization effects, prompt effects, policy-optimization effects, and benchmark option-position effects.

## Checkpoints

### Grounded initialization (primary)

Hugging Face model: **pending completion of the resumable upload job**.

Server source directory:

```text
/home/tata/tata/F/rsy/FineR1_repro/FineR1_ICLR2026_clean/LLaMA-Factory/saves/qwen25vl_3b_cub_v27c_abs_rigid_frozenvis_filtered_grounded_top2_frozenvis_v27c_abs_rigid_top2_e8_lr5e6
```

### Plain initialization (required control)

```text
/home/tata/tata/F/rsy/FineR1_repro/FineR1_ICLR2026_clean/LLaMA-Factory/saves/qwen25vl_3b_cub_v27c_abs_frozenvis_plain_cot_frozenvis_v27c_abs_strict_plain_top2_e3
```

Do not run only the grounded initialization. The contribution is an initialization comparison under an otherwise identical Stage2 protocol.

## Processor Invariant

Both Stage1 checkpoints now persist:

```json
{
  "min_pixels": 1024,
  "max_pixels": 589824,
  "size": {
    "shortest_edge": 1024,
    "longest_edge": 589824
  }
}
```

Validate every downloaded checkpoint:

```bash
python stage1_v1/scripts/preflight_stage1_export.py \
  --ckpt /path/to/checkpoint \
  --rl-min-pixels 1024 \
  --rl-max-pixels 589824
```

Do not continue if the preflight reports a processor mismatch. Resized-absolute grounding coordinates are only meaningful when conversion, training, rollout, and inference use the same canvas.

## Primary Bird-Only Matrix

The minimum controlled matrix is:

| Init | Method | Prompt | Reward | Purpose |
|---|---|---|---|---|
| Plain | CoT SFT | released generic | none | Stage1 control |
| Rigid | CoT SFT | released generic | none | grounded Stage1 control |
| Plain | DAPO | released generic | released | optimizer control |
| Rigid | DAPO | released generic | released | grounded-init effect |
| Plain | TAPO | released generic | released | paper method control |
| Rigid | TAPO | released generic | released | primary comparison |

Keep identical across each Plain/Rigid pair:

- Stage2 bird data and image variants;
- rollout and update budgets;
- seed;
- prompt template;
- reward/parser;
- pixel budget;
- checkpoint selection;
- evaluation scripts.

The grounded prompt is a secondary ablation, not a replacement for the released generic prompt:

| Init | Method | Prompt | Purpose |
|---|---|---|---|
| Rigid | TAPO | explicit grounded | measure prompt-assisted retention |

## Launcher

Dry run:

```bash
cd /path/to/FineR1_ICLR2026

STAGE1_INIT=/path/to/rigid-checkpoint \
INIT_LABEL=rigid \
PROMPT_MODE=generic \
SINGLE_GPU=1 \
bash stage1_v1/scripts/run_stage2_variant.sh dapo --dry-run
```

One-update smoke:

```bash
STAGE1_INIT=/path/to/rigid-checkpoint \
INIT_LABEL=rigid \
PROMPT_MODE=generic \
SINGLE_GPU=1 \
bash stage1_v1/scripts/run_stage2_variant.sh dapo --smoke
```

Formal multi-GPU runs should use the released budget unless a hardware-driven change is preregistered and applied to every compared method.

## Format and Grounding Tracking

The released Stage2 prompt requests only `<think>...</think><answer>...</answer>`. The released reward does not request or reward `<ref><box>`. Grounding may therefore decay even if answer accuracy improves.

At Stage2 step 0 and every saved checkpoint, record:

```text
think/answer format rate
ref_box_tag_rate
strict_ref_box
well_formed_given_tag
answer_before_grounding
duplicate_box_rate
average box count
```

Use `scripts/grounding_retention.py` for the retention summary. Keep the released reward unchanged in the primary runs. A label-free grounding-format reward may be tested only as a separate ablation after decay is measured.

## Evaluation

For each selected checkpoint report:

1. Bird closed seen/unseen using the released option order, for paper protocol compatibility.
2. Bird closed seen/unseen using deterministic shuffled options, for position-robust interpretation.
3. Bird open seen/unseen TI and SigLIP-relative semantic similarity.
4. Grounding retention under generic and explicit prompts.

Use:

```text
max_new_tokens = 2048
min_pixels = 1024
max_pixels = 589824
do_sample = false
```

The released Bird closed files have the ground-truth class at option index 0 for every example. Never use the released-order score alone as evidence of improved visual recognition.

## Intra and Inter Status

The current released TAPO source accepts the augmented contrastive path used by TAPO. It does not expose independently executable `intra` and `inter` contrastive enums. The launcher intentionally aborts if these names are not implemented in the source.

Do not publish rows called “CoT SFT + Intra” or “CoT SFT + Inter” until the exact executable definitions from the authors are available. Renaming a configuration is not a valid reproduction.

## Valid Claims

A bird-only Stage2 experiment can support Bird-column claims. It cannot be directly compared with the paper's six-domain average unless the full official six-domain Stage2 training and all 24 evaluation cells are run.

The central Stage2 question is:

> Under identical answer-reward policy optimization, does a grounded Stage1 initialization improve unseen bird recognition and how quickly does its grounding behavior decay?

