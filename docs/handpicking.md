# Handpicking images for the scaffolding ground truth

> **Reconstructed, not dictated.** This was written by reading the code and
> the tables on disk, not from the session that designed the workflow. The
> numbers below are recomputed from the CSVs and are current. The *intent* —
> particularly which sampling option was chosen and why — is inference.
> Correct anything that misstates the plan rather than working around it.

## Why this exists

The project needs to know which panoramas actually show a shed, scaffold or
construction fence. Two candidate sources were available and both fail:

- **CLIPSeg**, the detector: AUC 0.55 against DOB permits, 0.51 inside the
  forward cone. A coin flip. `tools/scaffold_eval.py` is the benchmark, and
  CLAUDE.md forbids reporting a scaffolding share from it.
- **DOB permits**, the city's own register: measured below, and worse than
  it looks.

So the ground truth has to be made by eye. That is what the review sheets
and the label tables are for.

## The measured case against the permit flag

`svi_scaffold_flag.py` marks an image `in_view=True` when a live permit sits
inside the match radius *and* inside the forward 180°. Its docstring is
careful to call this a candidate, not a confirmation. Scored against the
visual labels, it is weaker than that wording implies:

| sample | n | precision | recall | accuracy | trivial baseline |
|---|---|---|---|---|---|
| balanced (48 flagged / 48 not) | 96 | 0.42 | 0.54 | 0.53 | 0.50 |
| sequential sheets 1–18 | 216 | 0.29 | 0.35 | 0.68 | 0.79 |

On the balanced sample the flag performs at chance. On the sequential sample
it is *below* the trivial all-negative baseline. Both directions of error are
large: permits miss structures that are plainly there (no live permit, signed
off early, erected without one) and flag frontages where nothing is visible
(the permit point is a building address, so a shed round a corner or across a
wide avenue is in range but out of sight).

This is the finding that justifies the labelling campaign. It is not a reason
to discard the flag — it remains a useful *prioritiser*, narrowing 1,254
images to the 302 with any structure in range — but it is not a label.

## The instrument

```
data/raw/svi_180/<street>/<direction>/<seq>_<node_id>_<cardinal>.jpg
```

1,254 panoramas, 17 streets, two walks each, 1440×916. Built by
`tools/export_svi_180.py`. Not redistributable, gitignored, 30-day cache cap.

Two sheet builders, deliberately different jobs:

- `tools/svi_contact_sheets.py` — one sheet per street, sized to scan a whole
  street and spot what stands out.
- `tools/svi_review_sheets.py` — the labelling instrument. Fixed 2×6 grid,
  760 px thumbnails, a stable `[idx]` printed on every tile, sheet width held
  under 1570 px because anything wider is downsampled before it reaches a
  vision model and loses exactly the detail the sheet exists to show. **No
  DOB boxes drawn**, on purpose: on a labelling sheet a permit marker anchors
  the judgement to the thing being tested.

## The two sampling options

**Sequential.** Walk the export in order, 12 per sheet. Sheets 1–18 are done:
216 images, but only 4 of 17 streets (1st, 2nd, 3rd Avenues, E 34th). Gives
contiguous coverage of a walk; converges on the full set slowly and inherits
the natural ~24% base rate, so most tiles are negatives.

**Balanced (the handpicked option).** `results/review_sheets/balanced/` —
96 images drawn 48 flagged / 48 not, 8 sheets, `sample.csv` carrying a stable
`sid`, verdicts in `sample_labelled.csv`. This is the design that can *score*
the flag, because a balanced sample makes precision and recall meaningful and
puts real positives in front of the labeller often enough to calibrate the
eye.

The balanced sample is the one to extend. Sequential coverage is a reading
aid; balanced sampling is the measurement.

## What a label means

`results/tables/svi_180_visual_labels.csv`, one row per labelled image:

- `vlm_structure` — is any tracked structure visible in this panorama
- `vlm_type` — `facade_scaffold` | `sidewalk_shed` | `hoarding_fence`
- `in_view`, `shed_in_view`, `shed_nearest_m` — carried from the permit flag
  for comparison, **never** used to decide the verdict

Current counts across 216 sequential labels: 46 positive — `facade_scaffold`
28, `hoarding_fence` 11, `sidewalk_shed` 7. Sheds are a *minority* of what
stands on these streets, which is why `svi_scaffold_flag.py` tracks all four
DOB work types separately rather than folding them together.

## Rules

1. Label from the photograph. Never open the permit columns first.
2. Verdicts are per *image*, not per node. A node appears twice, once per
   walk; a structure behind the camera is not in the panorama. A node-level
   flag is wrong half the time by construction.
3. `node_id` is positional. If the frame is ever rebuilt, every label in
   these tables points at a different street corner and must be re-checked.
4. Nothing here becomes a reported share until it beats `scaffold_eval.py`
   and has a row in the `openvocab_eval.py` per-class AUC table.

## To extend the balanced set

```bash
.venv/Scripts/python tools/svi_review_sheets.py     # rebuild sheets
```

Draw the next balanced block from the 1,038 unlabelled rows of
`svi_180_scaffold.csv`, stratified by street so no avenue dominates, and keep
the 50/50 flagged split. Append verdicts against the printed `[idx]`.

## Related

- `tools/svi_180_segformer.py` — SegFormer-B5 Cityscapes over the same 1,254
  panoramas. A different question: streetscape composition, 19 fixed classes.
  Cityscapes has no scaffolding class, so it does **not** substitute for this
  campaign. Its `fence` class is confounded by permanent fencing.
