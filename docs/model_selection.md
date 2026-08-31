# Model selection

Why this study runs **Qwen2-VL-7B-Instruct** in 4-bit NF4, what else was tested,
and what the literature reports for everything not tested here.

## What the model has to do

Not "predict GVI". GVI is a vegetation pixel share, and a segmenter measuring
it is measuring the target directly — so scoring a VLM against it tests the VLM
on the segmenter's home turf and would prove little either way.

The model has to produce **judgements a segmenter cannot express**. There is no
ADE20K or Cityscapes class for *somewhere to sit*, *how much the frontage
varies*, or *how far greenery relieves the enclosure*. Those are the SIM terms,
and only a generative VLM can answer them in one pass.

GVI and VEI are the **validity check**: measurable quantities to score the
ratings against, so "the model rates greenery sensibly" is a number and not an
assertion.

## What Liu Liu benchmarked, and chose

MINGLE (Liu, Kudaeva, Cipriano, Al Ghannam, Tan, de Melo, Sevtsuk — AAAI 2026)
tested **ten VLMs zero-shot** on pairwise social affiliation in street imagery,
then selected Qwen2-VL-7B and LoRA fine-tuned it. This is the paper our study
cites, and the comparison it did is the one we would otherwise have to make
ourselves.

**Table 2 — zero-shot, pairwise classification, F1 on the positive class**

| model | size | F1 (Yes) | F1 (No) |
|---|---|---|---|
| Qwen2-VL | **72B** | **0.62** | 0.58 |
| Qwen2.5-VL | 72B | 0.59 | 0.56 |
| Phi 3.5 | 4.2B | 0.57 | 0.24 |
| **Qwen2-VL** | **7B** | **0.53** | 0.54 |
| Pixtral | 12B | 0.52 | 0.51 |
| LLama3.2-Vision | 11B | 0.46 | 0.30 |
| Gemma 3 | 24B | 0.27 | 0.67 |
| Claude-Sonnet | — | 0.24 | 0.66 |
| Qwen2.5-VL | 7B | 0.10 | 0.69 |
| GPT-4o | — | **0.00** | 0.70 |

GPT-4o scores 0.00 on the positive class and 0.70 on the negative: it answered
"no" to everything. Qwen2.5-VL-7B at 0.10 / 0.69 is nearly the same failure.
Both are degenerate answers rather than weak ones — the same pattern we measured
when Qwen2.5-VL-3B returned `{"enclosure": 3}` for every image in Murray Hill.

**Table 3 — region detection, mIoU**

| model | mIoU | F1 |
|---|---|---|
| **MINGLE** (detector + VLM) | **0.64** | 0.60 |
| Claude-Sonnet | 0.02 | 0.02 |
| GPT-4o | 0.01 | 0.01 |
| Claude-Opus | 0.01 | 0.02 |
| Qwen2-VL (7B) | 0.00 | 0.00 |
| LLama3.2-Vision | 0.00 | 0.00 |

Every VLM collapses on localisation, the frontier models included. Only the
hybrid — an off-the-shelf detector finding the regions, the VLM judging them —
works at all.

**Their stated reason for choosing 7B over 72B:**

> *"smaller variants — especially Qwen2-VL (7B) — show just marginally lower
> performance, while being more robust to overfitting, have lower memory
> requirements and significantly faster inference"*, which mattered because
> *"urban analysis requires working with vast amounts of data."*

72B scores 0.62 against 7B's 0.53 — a real gap, traded away deliberately for
throughput and memory. We reach the same conclusion from a harder constraint:
72B does not fit a 12.9 GB card at any quantisation worth using.

**ELSA** (Hosseini, Cipriano, Eslami, Hodczak, Liu, Sevtsuk, de Melo) benchmarks
the open-vocabulary side on 900+ manually annotated NYC street images:
**Grounding DINO, Detic, OWL, MDETR**, all at single-digit mAP, and reports they
are *"sensitive to slight variations in input phrasing"*. We reproduced that
independently: five prompts for the same structure, scored against the same
labels on the same images, ranged 0.555 to 0.757 on wording alone.

## Why only two generative models were tested here

MINGLE already benchmarked ten, including GPT-4o, Claude Sonnet, Claude Opus
and both 72B variants. Repeating that comparison would add nothing; citing it
is the correct move, and it is why that table appears first.

The question left over is narrower: **does the selected model work on this
imagery and this schema, and is 7B worth its cost over 3B?** Those two decide
it, and the answer is decisive -- +0.24 on greenery, and the difference between
a usable enclosure rating and a constant.

### The hardware decides most of it

One 12 GB card, run locally. That single constraint removes most of the field
before any measurement:

- **72B** needs more than 40 GB, or quantisation heavy enough to change what is
  being measured.
- **7B in bf16** is 16.6 GB and does not fit either. 4-bit NF4 brings it to
  **5.9 GB**, which is why the study runs a quantised 7B rather than a
  full-precision anything.

### Local beats metered, and the reason is iteration not price

**6,271 generative VLM calls were made on this corpus in a single day** --
across a morphology schema, a SIM schema, a scaffolding pass, a cone test, a
projection test, a partial 90-degree run and a three-model benchmark.

At API rates that is roughly **$25 to $95** depending on the provider. The cost
is not the point. The point is that the schema changed six times in that day,
and each change meant re-running everything. Under a per-call meter that
iteration does not happen -- you commit to a schema early to avoid paying for
mistakes, which is exactly the wrong incentive when the schema is what is being
worked out. Counts were dropped, enclosure was reformulated five times, the
60-degree split was tested and abandoned; none of that survives a budget.

Local inference also keeps Street View imagery on the machine, which matters
given Google's redistribution terms.

The remaining alternatives are ruled out on published evidence rather than by
testing:

- **GPT-4o and Claude** sit below the 7B in MINGLE's table on the nearest
  published task, on top of being metered.
- **Phi 3.5, Pixtral, Gemma 3, LLama3.2-Vision** all sit below Qwen2-VL-7B in
  MINGLE's table, so testing them locally would be re-deriving a published
  negative.

Counting the open-vocabulary models, **four VLMs were tested here**, not two:
Qwen2-VL-7B, Qwen2.5-VL-3B, CLIPSeg and OWLv2.

## Measured here

Spearman against measured GVI / VEI, bootstrap clustered on `face_id`
(Moran's I is 0.62–0.66, so rows are not independent).

### Generative VLMs — image and text in, text out

| model | greenery vs GVI | enclosure vs VEI | VRAM |
|---|---|---|---|
| **Qwen2-VL-7B** (4-bit) | **+0.733** [+0.632, +0.792] | **+0.480** [+0.340, +0.585] | 5.9 GB |
| Qwen2.5-VL-3B (4-bit) | +0.494 [+0.359, +0.603] | **constant — 3 on 83 of 85 measured** | ~3 GB |

Same question, same images. The 3B does not discriminate enclosure at all.
Measured directly on 85 images it returned `{"enclosure": 3}` on all 83 that
parsed, the other two being unreadable -- a degenerate answer rather than a
parsing failure, and reported as such. Greenery does vary for the same model
(five distinct values), which is why that column still correlates.

**"Discriminating" means the answer changes with the image.** A model that says
the same thing about every street carries no information about any street,
however confident the wording. That is the failure the two-column F1 in MINGLE's
table exists to expose, and it is why a correlation is refused rather than
reported when a column has fewer than two distinct values.

### How to read a two-column F1

Read the **weaker** of the two columns, never the higher one. A model answering
"no" to everything scores 0.70 on the No column and 0.00 on the Yes column: the
0.70 is an artefact of class balance, and only the 0.00 says it never decided
anything. A coin flip on a balanced task gives about 0.50 on **both** sides.

| weaker side | reading |
|---|---|
| >= 0.50 | discriminates both ways |
| 0.30 - 0.50 | lopsided |
| 0.15 - 0.30 | mostly answers one way |
| < 0.15 | degenerate -- answers one way regardless of input |

Sorted that way, **only four of MINGLE's ten models clear 0.50**, and two of
those are 72B. Phi 3.5 shows why the higher column misleads: its F1(Yes) of 0.57
beats the selected model's 0.53, while its weaker side is 0.24.

### Open-vocabulary VLMs — text-conditioned, not generative

| model | greenery vs GVI | enclosure vs VEI |
|---|---|---|
| **CLIPSeg** | **+0.809** [+0.725, +0.854] | +0.414 [+0.178, +0.591] |
| OWLv2 `'a street tree'` | +0.519 [+0.384, +0.606] | — |
| OWLv2 `'scaffolding'` *(negative control)* | −0.133 [−0.246, −0.025] | — |

CLIPSeg is the best greenery predictor of any VLM here. It **cannot produce the
SIM ratings** — it returns a mask for a text prompt, not a judgement on a scale
— so it is a strong single-concept detector, not a candidate for this schema.

The `'scaffolding'` row is a **negative control, not a comparison**: a
scaffolding prompt should not predict greenery, so near zero is the correct
result and reading it as a weakness would be wrong.

The genuine phrasing sensitivity is elsewhere. Scored against the same
scaffolding labels, on the same images, wording alone moved OWLv2 from 0.555
(`'scaffolding'`) to 0.757 (`'metal scaffold poles'`), with
`'a sidewalk shed'` at 0.564 and `'a plywood construction barrier'` at 0.693.

## Reported elsewhere — different tasks and metrics, not comparable

| model | task, metric | value | source |
|---|---|---|---|
| CLIPSeg | scaffolding vs DOB permits, AUC | 0.55 | this repo, `scaffold_eval.py` |
| CLIPSeg | same, forward cone only, AUC | 0.51 | this repo, CLAUDE.md |
| open-vocab detectors | street trees vs city register, AUC | 0.78 | this repo, `openvocab_eval.py` |
| closed-set detector | street trees, same | 0.83 | this repo |
| open-vocab detectors | sidewalk sheds, AUC | 0.51 | this repo |
| GPT-4o | urban region detection, mIoU | **0.01** | MINGLE, AAAI 2026, Table 3 |
| Claude Sonnet | same | 0.02 | MINGLE |
| Claude Opus | same | benchmarked, not selected | MINGLE |
| Qwen2-VL-7B zero-shot | same | 0.00 | MINGLE |
| **MINGLE pipeline** (detector + VLM) | same | **0.64** | MINGLE |
| Grounding DINO, Detic, OWL, MDETR | social activity detection, mAP | single-digit | ELSA, arXiv 2406.01551 |
| CAT-Seg | this imagery | reported unusable | collaborator, `blockology-gvi` |
| FGA-Seg | CAT-Seg fork | not tested; same stack | arXiv 2501.00877 |

The recurring pattern across all of it: **open-vocabulary models sit near chance
on street furniture, generative VLMs near zero on localisation, and a detector
plus a VLM beats either alone.** This study reproduces that shape.

## Why Qwen2-VL-7B

1. **It is the only tested model that can answer the SIM schema.** CLIPSeg
   scores higher on greenery but returns masks, not ratings. That is the
   deciding constraint, not the correlation.
2. **Best VLM on enclosure**, +0.480, where CLIPSeg reaches +0.414 and the 3B
   returns a constant.
3. **Size is load-bearing.** 7B beats 3B by +0.24 on greenery and is the
   difference between a usable enclosure rating and no variance at all.
4. **It fits.** 16.6 GB in bf16 does not fit a 12.9 GB card; 4-bit NF4 brings it
   to 5.9 GB with no measured loss.
5. **Independently selected by MINGLE** (AAAI 2026) after benchmarking eleven
   VLMs including GPT-4o, Claude Sonnet and Claude Opus, for the same reasons:
   *"smaller variants show just marginally lower performance, while being more
   robust to overfitting"* and *"significantly faster inference, crucial for
   processing vast urban datasets."*
6. **Open weights, run locally, no per-image cost**, so the whole corpus can be
   re-run when the schema changes — which it has, repeatedly.

## What would change the choice

**Qwen2-VL-72B** would need >40 GB or heavier quantisation; untested here.
**GPT-4o or Claude** would remove the local-cost advantage and MINGLE measures
them below the 7B on the nearest published task. A **detector + VLM** hybrid is
the one direction the literature says is clearly better, and it is what
`SIM_hybrid` already does in miniature by taking enclosure from measured VEI
rather than from the model.

## One caveat on every number above

Qwen's enclosure rating scores **+0.480 asked as a standalone question** and
**+0.012 as the tenth field of a twelve-field JSON**. Same model, same images.
Schema length costs accuracy, and the in-schema figures in this document should
be read with that in mind.
