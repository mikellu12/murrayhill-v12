# Murray Hill streetscape pipeline

Measuring the Street Interface Matrix — Place Imageability, Place Identity,
Place Dependence — at 20 m sampling nodes across Murray Hill, Manhattan, from
Google Street View imagery.

Two tracks run over the same frame. A **segmentation track** produces class
shares from two open-weight semantic segmenters, plus Green View Index, Visual
Enclosure Index and azimuthal profiles. A **VLM track** rates ten perceptual
fields with a vision-language model, for the terms no segmenter can produce.
The first exists largely to validate the second: **eight of the ten** VLM
fields now have a measured counterpart in the same 90 degrees of view, and
those eight are the only evidence that the model is reading this imagery
sensibly rather than producing plausible numbers.

Read the twin table in section 4 before trusting any field. The correlations
run from +0.718 down to +0.143, two fields have no usable counterpart at all,
and the three greenery fields turn out not to be separable from one another —
which matters, because eye-level greenness is the framework's headline claim.

The specification is `Nature.8.31spacesyntax.docx` with
`formulawithspacesyntax.docx` alongside it. The two currently disagree on one
point: the canyon penalty is `Ω_i` and trails the product in the second, and is
`A_i` and leads it in the first. They are mathematically identical, but `A_i`
collides with the Stayability Amplification Factor, which the first document
renames `F_i` and the second still calls `A_i`. Where the code departs from the
specification, the departure is stated here and in `config.yaml`, never left
implicit.

---

## The pipeline

```
1  nodes          OSM network -> 20 m nodes, headings, sequence
2  imagery        Street View metadata -> panoramas -> reprojected views
3  segmentation   two segmenters -> class shares; azimuthal profiles -> GVI, VEI
4  VLM ratings    ten fields, one call per field, logits read not generated
5  compose        I, Y, D, Omega -> M, and validate against the segmenters
6  figures        maps, sections, cubes, ribbons
```

Stages 1–3 need a GPU and a `GMAPS_KEY`; 4 needs a GPU; 5–6 run on a laptop
with only `data/processed/` copied across.

---

## 1. Nodes

`src/s01_frame.py` downloads the drive network, chains ways into streets,
resamples every 20 m, and clips to the study area.

**The study area is a quadrilateral, not the bbox.** `study_area.bbox` is a
download bound — 3.04 km², deliberately 250–400 m larger on every side so the
corner intersections sit inside the graph and street chains run past the clip
line. `corner_pairs` names four intersections (Madison × E34, Madison × E42,
1st Ave × E34, 1st Ave × E42); the quadrilateral between them, buffered by
`clip_buffer_m`, is the study area. It is 0.67 km² and yields **766 nodes
across 19 street segments**. Everything in the margin is discarded here and
never reaches stage 2.

The boundary is also written out as `data/processed/study_area.geojson`, so it
can be shared rather than reconstructed from a regex over street names.

**Node IDs are positional.** `n00000` upward, assigned against the current
geometry. Rebuild the frame with different settings and the same ID points at
a different corner, so cached imagery and profiles pair with the wrong place.
Stages 2 and 4 abort rather than allow it. Changing anything under
`study_area:` or `sampling:` means deleting `data/processed/` and
`data/raw/svi/`.

**Street labelling** follows `common.street_grouping()`: `street_segment` from
an imported coordinate mapping if present, else `cleaned_street`, else the raw
chains. `tools/import_segments.py` attaches the first of those by spatial
join, refusing any node that snaps further than 2 m.

---

## 2. Imagery

`src/s02_imagery.py` probes the free metadata endpoint first — coverage,
capture date, and how far Google's nearest panorama sits from the node — then
requests imagery **by `pano_id`, never by coordinate**, because a coordinate
request can snap to a different panorama for each heading and corrupt the
four-way sum.

Four headings per node at FOV 90 give a 360-degree cover. Those are the source
frames; the analysis views are reprojections of them.

### Reprojected views

```
svi_90    two 90-degree halves, left and right of the walk direction
          766 nodes x 2 walks x 2 sides = 3,064 images
          1440 x 1833 px, 16 px/degree          <- the rating set

svi_180   one 180-degree strip centred on the walk bearing
          766 nodes x 2 walks = 1,532 images
          1440 x 916 px, 8 px/degree
```

Both are rendered from the **raw four frames**, not from each other:
`export_svi_90.py` and `export_svi_180.py` each call the same cylindrical
reprojection on `data/raw/svi`, differing only in output field of view. One
resample each, no intermediate.

Rendered, not cropped. Halving the field of view doubles both the angular
resolution and the vertical extent, so the 90-degree halves carry twice the
detail of a crop from the 180 strip.

Both sets now cover **all 766 nodes**. svi_180 previously held 627: a
sky-share tunnel heuristic and a hardcoded four-node viaduct list were dropping
139, of which 106 were Park Avenue in one contiguous run — the deepest canyons
in the study area, absent from one imagery set and present in the other. Tunnel
and bridge nodes are carried with their OSM flags and filtered downstream,
which is what svi_90 already did.

Left and right are relative to **direction of travel**, not compass. Two walks
per street and two halves per walk give four views per node, covering two
frontages each seen from both approaches — a free consistency check, since a
frontage that rates differently depending on which way it is passed is either
a real directional effect or model noise.

### Why the field of view should depend on the street

Street View is captured **from a vehicle on the roadway**, and that fixes where
the camera stands relative to what a pedestrian sees.

On a **wide vehicular street** the camera sits in the carriageway, tens of
metres from either frontage. What a walker on that sidewalk experiences is
mostly the building line on their own side; the opposite frontage is far away
and read peripherally. A 90-degree half centred on each side captures that
frontage squarely, and at 16 px/degree — twice the angular resolution of the
wider render. Splitting into two halves also gives one observation per
*sidewalk* rather than one per street, which is the unit a pedestrian actually
occupies: `L` and `R` are two different walking experiences, not two views of
one thing to be averaged.

On a **narrow pedestrian way** the camera and the walker are in the same place,
and both frontages are within a few metres. Enclosure there is experienced
across nearly the whole visual field, not in a 90-degree cone facing one wall,
and a half-view would cut out most of what makes the space feel as it does. The
180-degree strip is the honest render of that situation.

So the field of view is not a nuisance parameter to be corrected away — it is
**part of what is being measured**, chosen to match the perceptual situation.
That has a consequence worth stating: an `M` from a 180-degree walkway and an
`M` from a 90-degree street are not strictly on the same scale, and the paper
should report them as typology-specific rather than compare them directly.

**Planned, not implemented.** Nothing in the code makes that choice today;
every node has both renders and everything downstream uses svi_90.

Three things are known about that plan, none of them blocking:

**OSM states the street type.** `highway=footway | pedestrian | steps | path |
cycleway | living_street` against `residential | primary | secondary | service`.
The study bbox holds 3,609 pedestrian-class ways against 1,084 vehicular. But
`s01` builds the frame from the *street* network, so only 6 of our 766 nodes
land on a pedestrian way, and those look like snapping noise at intersections.
Murray Hill's frame contains no pedestrian-only routes to test the rule on.

**The two field of views are more comparable than expected.** Measured on 1,524
matched captures — same nodes, same source frames, differing only in render —
rank correlations run **0.92 to 0.99** on the large classes and the means move
by 3% or less: building +3%, sky −5 to −7%. Both renders are cylindrical and
cover the same solid angle, so mixing them needs no correction for seven of the
fields.

**Except on the ground plane.** `map_Sidewalk` comes out at a ratio of 0.84 and
`Lane Marking - Crosswalk` at 0.69 — thin features, and svi_180 renders at 8
px/degree against svi_90's 16. That is a resolution difference, not a
projection one, and rendering the 180s at `--width 2880` would match the two
exactly. Untested.

**Imagery is not redistributable.** Google caps caching at 30 days. Derived
profiles and metrics are ours to publish; the JPEGs are scratch and are
git-ignored.

---

## 3. Segmentation

Two segmenters run over every rendered view in one process,
`tools/seg_two_model.py`, at **0.38 s per image** — 3,064 frames in 19 minutes:

```
Mapillary Vistas   65 classes, street-level.  Curb, Curb Cut, Pedestrian Area,
                   Bench, Trash Can, Banner, Billboard, front/back Traffic
                   Signs, Bridge, Tunnel.
ADE20K            150 classes.  windowpane and door, which Mapillary lacks.
```

Neither covers the ten fields alone. Mapillary supplies six twins, ADE20K
contributes to one (`stairs` for IAS), and one field needs a third source.

**This replaced a four-model taxonomy pipeline** that took 15.3 hours and whose
30 purpose-built classes scored *worse* than plain ADE20K wherever both had a
class — greenery +0.686 against +0.720, sky +0.548 against +0.584, hardscape
+0.178 against +0.388. Five of its classes never fire at all (`shrub_hedge`,
`ground_vegetation`, `vertical_green_wall`, `arcade_column`, `arcade_soffit`,
each written into all 3,064 files with exactly zero pixels). It is still
required for one field: its `ground_floor_glazing` fires on 93% of frames and
gives GFAPI its only positive twin, where ADE20K's `windowpane` fires on 1%.
`data/processed/seg90_shares.csv` holds that run.

### Google's camera mast

Every frame carries the capture vehicle's mast rising from the bottom edge with
the Google wordmark on it. Neither segmenter has a class for it, so both label
it as something: masking removes **47% of Mapillary's `Pole`, 60% of ADE20K's
`pole`, and 29% of ADE20K's `signboard`** — that last being the wordmark read
as a sign. The VLM is fooled differently, and only by the text: erasing the
mast moves `signage_detail` by −0.104 of a rung (Wilcoxon p<0.0001, n=47) while
`walkable_ground` and `vertical_hardscape` do not move at all.

`src/mast.py` detects it without either model — keying off `Pole` would miss
the frames ADE20K called `signboard`. Three measured properties do the work:
the blob reaches the bottom edge, is a fixed 16.1% of frame height, and a fixed
5.4% of width. A real pole is detected as a blob floating higher: of 143 blobs,
85% of those stopping below 20% height touched the bottom and **0%** of every
other height band did.

Calibration is **per imagery set**, named by the source folder. The mast
subtends a fixed angle, so svi_180 — twice the horizontal span in the same
pixel width — carries *two* masts at 2.71% wide against svi_90's one at 5.4%,
almost exactly half, which is the check that these are the same object. An
uncalibrated set raises rather than guessing. To calibrate a new one, average a
few hundred frames: fixed elements stay sharp while streetscape blurs away.

Effect on the twins is small — only `signage_detail` moves, +0.420 to +0.439.
Half of `Pole` was camera hardware and no twin used `Pole`. This is a
correctness fix, not an improvement.

**A prompt line telling the VLM to ignore the mast was tested and rejected.** It
moved signage the *wrong way* (+0.237 against the pixel edit's −0.103) and
shifted `walkable_ground` by +0.469 — a field the mast does not affect at all.
Adding a sentence perturbs the model globally rather than surgically.

### Azimuthal profiles

`src/s03_profiles.py` segments the four frames per node with Mask2Former and
projects the class shares into **360 one-degree azimuth bins**.

`data/processed/azimuth_profiles.npz` is the keystone: 766 arrays of shape
(4, 360) — vegetation, sky, building, weight — in 3.9 MB. Every GVI and VEI in
the study is a *slice* of it. The five view conditions in stage 4, the
90-degree arc targets used to validate the VLM, the 180-degree halves: all
sums over different bin ranges, no re-segmentation.

Regeneration takes about 11 minutes on an RTX 3080 Ti **provided the source
JPEGs are still on disk**. They expire under the 30-day cap, after which the
file cannot be regenerated at all without re-fetching — and the panoramas may
have changed by then. Back it up before touching it.

`SVF_band` is **not** a sky view factor. The imagery spans ±45 degrees of
elevation, so the zenith is never sampled. Do not compare it to published SVF.

---

## 4. VLM ratings

`tools/sim_vlm_run.py`, Qwen2-VL-7B-Instruct in 4-bit NF4.

**One field per call, ten calls per image, batched.** A twelve-field schema
was tried first and collapsed — the model wrote one number into every slot, so
every field in a family returned an identical correlation. Asked one field at
a time, `green_eye_level` moved from ρ +0.403 to +0.787 against the greenery
measured in the same frame.

**Seven named rungs, not two anchors.** With only the ends of the scale named,
the model never answered 4 on any field: the midpoint carried real probability
but was never the tallest single token, so greedy decoding could not emit it.
Naming all seven rungs fixed that — every field now uses 4.

**One metric per ladder.** Three fields changed what they measured partway up
and have been rewritten (`src/sim_scale.py` carries the old wording and the
measured justification in comments beside each):

| field | the defect | result |
|---|---|---|
| `walkable_ground` | width at rungs 1/2/4/5/6, obstruction at 3, share of view at 7 | no accuracy change on 3,064 |
| `resting_affordance` | counted objects at 2–4, switched to extent at 5–7 | bimodality 35% → 5%, decisive 12% → 22% |
| `green_softening` | asked for a judgement about an *effect*, and rungs 3–4 described a ratio between two subjects | bimodality 60% → 8% |

`walkable_ground` looked like a large win on a 300-image pilot (+0.188 →
+0.333, CI just clearing zero) and **did not replicate** on the full set
(+0.342 → +0.336). The pilot was a false positive. What survives is the shape
change on the other two — the models commit far more often — and that is why
the rewrites are kept.

A separate check found nothing wrong with the *other* seven ladders. Embedding
the rung sentences and projecting them onto their own scale flagged inversions
in six of ten fields and near-duplicate pairs in all ten, which looked
damning — but grouping nodes by assigned rung and reading the measured share of
each group came out **monotone on every field**, including at exactly the rungs
the embedding called inverted. The embedding measures sentence similarity, not
whether the model uses the rungs in order. It is recorded here so nobody
repeats it.

**Logits are read, not generated.** The assistant turn is prefixed with
`{"field": ` so the next token must be the digit, and the full distribution
over `1`–`7` is read at that one position. Verified identical to
`generate(do_sample=False)` on 6 of 6 images, so nothing is lost and the other
six probabilities survive. Three readings are stored per field:

| column | what it is |
|---|---|
| `<field>` | `round(EV)` — the survey answer |
| `<field>_ev` | the expected value before rounding |
| `<field>_argmax` | what a `generate()` call would have written |
| `<field>_p1..p7` | the full distribution |

The run is deterministic across a model reload: a clean single pass over all
3,064 images matched an incrementally-built table on **27,576 of 27,576
ratings**, with a maximum EV drift of 9e-16.

### The ten fields

| field | manuscript term | measured twin | ρ |
|---|---|---|---|
| `vertical_greenery` | V_nat | `map_Vegetation`, whole frame | +0.718 |
| `green_eye_level` | GVI_eye | `map_Vegetation` 0–15° below horizon | +0.601 |
| `sky_openness` | SVF | `map_Sky` | +0.482 |
| `vertical_hardscape` | V_built | `map_Building + map_Wall` | +0.479 |
| `signage_detail` | V_sign | `map_Billboard` | +0.439 |
| `walkable_ground` | V_pave | `map_Sidewalk + Curb + Curb Cut + Pedestrian Area` | +0.248 |
| `resting_affordance` | IAS | `map_Bench + ade_stairs + ade_step + ade_bench` | +0.234 |
| `ground_floor_activity` | GFAPI | `ground_floor_glazing` (four-model taxonomy) | +0.143 |
| `green_softening` | GMI | greenery on the lower 3 m of facade | +0.446 |
| `facade_variation` | SFV | — every source tried is negative | — |

`V_nat`'s twin is whole-frame by the manuscript's own definition — it counts
"foveal **and peripheral**" vegetation — so `veg_all` is correct there, not a
convenience.

**`GVI_eye` is listed against its own definition, not its best score.** The
manuscript defines it as *"the region 0° to 15° below the horizontal vanishing
line of sight"*, and `tools/seg_bands.py` computes exactly that band from the
render geometry (`φ = arctan((H/2 − y)/fc)`, so 15° below centre is 246 px for
svi_90). Against that band it scores **+0.601**. Against whole-frame vegetation
it scores +0.694 — higher, but that is validating it against overhead canopy,
which is the thing GVI_eye exists to exclude. The weaker number against the
right region is the honest one.

That gap is itself the finding: **the model resolves how much greenery there is
far better than where it is.** The bands are genuinely distinct (`veg_above15` vs `veg_eye0_15` correlate
only +0.510, at shares of 15.25% and 3.95%), yet all three green fields track
total vegetation better than their own region, and correlate +0.73 to +0.84
with each other. Asked about greenery at eye height, the model reports
greenery. Whether `GVI_eye` and `V_nat` are separable at all is a question for
the manuscript, and it matters because eye-level greenness is the framework's
headline claim.

Eight of ten now have a counterpart, against four before the two-model
segmentation. `resting_affordance` in particular went from nothing to a
validated +0.234, which is weak but real and clustered on `face_id` excludes
zero. Two things are worth knowing about that table:

`ade_signboard` is **excluded** from the signage twin despite being the obvious
candidate. 29% of its pixels sit on Google's camera mast — the wordmark read as
a sign — and adding it drops the correlation from +0.439 to +0.375.

`wall_ledge` is **excluded** from the IAS twin for the same class of reason. It
fires on 83% of frames but anti-correlates with rated seating, flipping the
twin from +0.14 to −0.07; it appears to be catching facade trim and sills at
height rather than perchable surfaces.

**GMI's twin is built from the manuscript's own definition.** The paper
specifies greenery covering "the lower 3 metres of the building facades" -- a
distance in the world, not a fraction of the image. `tools/seg_gmi_band.py`
computes exactly that, per node, from the band probe's measured facade
distance: `phi = arctan((z - h_cam)/d)` with `d = W_facade/2` and the camera at
2.5 m, giving a band that runs +2.3 to -11.5 degrees on average and varies from
77 to 451 rows as the street widens. A fixed image band cannot express it: 3 m
subtends about 14 degrees at a 23 m street and 8 on Park Avenue at 43.6 m.

It scores **+0.446**, against +0.617 for whole-frame vegetation. The lower
number is the right one, for the same reason GVI_eye is listed at +0.601: the
higher score comes from counting greenery the definition excludes.

"Structural interaction variable" in the manuscript describes the phenomenon --
greenery changing how hard surfaces are perceived -- not a multiplicative form.
The operational heuristic is one measurable quantity, coverage of a surface. So
GMI is a SUBSET of V_nat, greenery in a particular place, and correlating with
total greenery is what a correct twin does rather than evidence of redundancy.
Its -0.323 against `vertical_hardscape` is likewise the expected direction:
greenery covering a wall means less bare wall in view.

Two weaker constructions are recorded for completeness. Vegetation adjacent to
a building at ANY height correlates +0.857 with total vegetation, so it adds
nothing over plain greenery. The RATIO of adjacent-to-total is distinct but
inverts (-0.866 with total): sparse greenery in a dense street hugs the facade,
so a high fraction means little greenery rather than more softening.

Coverage is the live constraint: `W_facade` exists for 584 of 764 nodes. The
rest carry inherited H/W and a band from a borrowed width would be a guess, so
`HW_source` rides along and those rows can be dropped. The 33 open-one-side
nodes arguably have no GMI at all rather than zero -- with no opposite wall
there is nothing to soften, the same geometric fact that gives them Omega = 1.0.

**`green_softening`'s best number passes and fails on the construct.**
+0.701 against `map_Vegetation` is a good correlation, and it is the *same*
class `vertical_greenery` is validated against at +0.718. The two fields
correlate **+0.836** with each other, so the table would be showing twice that
a greenery field tracks greenery. GMI is defined as an interaction — "how
eye-level greenery softens the visual impact of vertical hardscape walls" — and
its correlation with `vertical_hardscape` is **−0.323**, so it carries almost
none of the hardscape half.

The rung rewrite made this worse rather than better: alignment with
`vertical_greenery` went +0.726 → +0.836. Recasting the question as one
observable quantity fixed the field's real pathology (60% of its answers were
two-peaked, 58% landed on a rung the model ranked below its own runner-up) by
removing the interaction that made it hard. Whether GMI should be an
interaction term or a greenery measure is a question for the manuscript, not
one to settle by wording, and until it is settled this row should not be read
as GMI being validated.

`tools/sim_vlm_run.py --show-prompts` prints every prompt exactly as sent.

---

## 5. Compose and validate

`tools/sim_compute.py` implements section 2.7:

```
M = I^a · Y^b · D^c · Ω

I = sigmoid(I_raw, κ_I, τ_I)   I_raw = mean(V_nat/V_built, GVI_eye, GMI)
Y =         Y_raw              Y     = mean(V_sign, 1 − SVF, SFV)
D = sigmoid(D_raw, κ_D, τ_D)   D_raw = mean(V_pave, IAS, GFAPI)
Ω = exp(−ψ · max(0, H/W − Ω_th))
```

Exponents shift by morphological regime (section 2.8), classified on H/W.

### Canyon geometry

H/W is measured from building footprints, not imagery. `W` comes from a
**band probe**: a parallel-sided corridor one node spacing wide, sampled by
nine rays, reduced by the nearest hit, reaching 40 m each side. Adjacent nodes
tile the street rather than overlapping, and the probe cannot widen at range
and punch through onto the next block.

| `HW_source` | nodes | meaning |
|---|---|---|
| `measured` | 584 | both walls found |
| `radius_mean` | 75 | mean of measured nodes on the same street within 20√2 m |
| `series` | 72 | nearest measured node walking the street's own order |
| `open_one_side` | 33 | one wall, other side genuinely open — no ratio exists |

`open_one_side` is not missing data: those nodes have no opposite wall, so
`HW_effective` is NaN by construction while Ω = 1.0 — the value the formula
already gives any street below the comfort threshold. They also take the
manuscript's POPS/porous elasticities, which section 2.8 gives no H/W band and
which are otherwise unreachable.

Validated against the segmenter: **ρ +0.429 between measured H/W and VEI**
(n = 583).

### VLM validation

Each rating is scored against the pixels in **its own 90 degrees**, not the
node's — a half facing a blank wall while the trees stand opposite should read
low, and a node-level share would count them.

| field | vs | ρ round(EV) | ρ true EV |
|---|---|---|---|
| `vertical_greenery` | arc vegetation | +0.720 | **+0.751** |
| `green_eye_level` | arc vegetation | +0.715 | **+0.751** |
| `green_softening` | arc vegetation | +0.573 | **+0.625** |
| `sky_openness` | arc sky | +0.584 | **+0.605** |
| `vertical_hardscape` | arc building | +0.388 | **+0.446** |

The unrounded EV wins on all five. Rounding discards information the index
never needed, since every rating is normalised to [0,1] before use.

#### Which readout — and why it is not settled

The seven rungs are **ordered categories**, and an expected value over them
assumes the step from rung 2 to 3 equals the step from 3 to 4. That is
measurably false here. Grouping nodes by the rung the model assigned and taking
the median measured share of each group gives, for `vertical_greenery`:

```
rung      2       3       4       5       6
share  0.0016  0.0220  0.0608  0.1689  0.4334
ratio      -    14.0x    2.8x    2.8x    2.6x
```

Each rung is roughly a constant **multiple** of the previous, not a constant
increment: `log(share) ~ rung` fits at R² 0.941 against 0.809 for the linear
form. The scale is perceptual (Weber-Fechner), which is arguably the right
thing for a study of experienced quality, but it means a mean over rung
*numbers* is not defined on the quantity being measured, and it must be stated
rather than assumed.

Every scale is nonetheless **monotone in practice** — the medians above rise
without a single reversal, on every field with a twin — so the instrument
works even where the arithmetic on top of it is questionable.

Five alternatives were tested against the measured twins on the full 3,064:

| readout | assumes | result |
|---|---|---|
| expected value | even spacing | the current pipeline |
| argmax | ordering only | −0.003 vs EV |
| interpolated median | ordering only | **+0.010 to +0.014 vs EV** |
| prune below 1/7, then EV | even spacing | +0.026 [+0.015, +0.040] |
| prune, then re-ask the model | even spacing | +0.047 [+0.004, +0.106] |
| re-ask to a single rung | nothing | **−0.043**, worse than EV |

The interpolated median is the defensible choice — it needs only ordering, and
it beats the mean on four of five fields. Pruning scores marginally higher but
is a mean, so it inherits the problem. Eliminating to a single rung is the most
defensible of all and the worst performing, which is the trade stated plainly.
`results/tables/sim_vlm_converged.csv` holds that run.

Nothing is switched over: `sim_compute.py` still stores `round(EV)`.

### Outputs

```
results/tables/vlm_observations.csv    3,064 x 121   everything observed
results/tables/vlm_calculations.csv   3,064 x  27   everything derived
results/tables/vlm_sections.csv         121 x   5   per-section aggregate
results/tables/README.md                            generated data dictionary

data/processed/seg90_two_model.csv    3,064 x 217   class shares, svi_90
data/processed/seg180_two_model.csv   1,532 x 217   class shares, svi_180
data/processed/seg90_shares.csv       3,064 x  30   four-model taxonomy run
```

Alternative runs kept beside the live tables because the questions they answer
are open, not because they are in use:

```
results/tables/sim_vlm_v2.csv         3,064 x 107   rungs rewritten for three fields
results/tables/sim_vlm_converged.csv  3,064 x 101   rated by elimination, not EV
```

Joined on `file`.

---

## 6. Figures

`src/s08_figures.py` and the `tools/sim_*` generators produce the maps,
section scores, the I/Y/D cube, the ribbon heatmaps and the exploded
axonometric. `make_dashboard.py` writes a self-contained HTML report.

---

## Known departures from the specification

Stated here because a reader should not have to diff the code against the
document to find them.

**τ is hardcoded where the document says calibrate it.** Section 2.7 defines
the thresholds as `τ_I(s_i)` and `τ_D(s_i)` — *"context-dependent, empirically
calibrated parameters … dynamically estimated for the specific area being
evaluated."* We use fixed 0.20 and 0.15, which were set for **pixel shares**.
Our inputs are normalised 1–7 ratings, and the same street reads 14–24× higher
on the rating scale than on the pixel scale. The result is that **99.5% of
rows saturate D** and 31% saturate I: `D_raw` runs 0.222–0.778 against a
threshold of 0.15, so every node clears it before the sigmoid begins and D
contributes nothing to M. This is the single largest open problem.

**Sub-index weights are normalised, which the document does not state.** It
gives `α_k = β_k = γ_k = 1.0` alongside a summation formula, which read
literally is an unnormalised sum reaching 3. We divide by the number of terms.
Under the literal reading both I and D saturate at 100%.

**The regime bands do not tile H/W.** Section 2.8 names H/W ≥ 3.0 and
0.8 ≤ H/W ≤ 1.2; 1.2 < H/W < 3.0 and H/W < 0.8 belong to no regime, so **52%
of half-views take the global exponents**. Table 2 presents three typologies
as exhaustive.

**Table 2's k-means does not reproduce.** Our silhouette is 0.440 against a
reported 0.784, and the reported centroids are unreachable: Avenue Canyons
sits at D = 0.159 where our minimum is 0.747. Same cause as the τ problem.

**No outcome variable exists.** The GWR calibration in section 2.8 regresses
on `t_base` — baseline pedestrian dwell time. NYC Open Data has no sensor
within 1 km of the study area, and the one manual count site measures flow,
not dwell. The machinery is implemented and checkable
(`tools/gwr_machinery.py`); the regression is not runnable. `dwell_lambda` in
`config.yaml` is a placeholder and must not be reported as fitted.

**Not implemented:** the Stayability Amplification Factor `A_i`, peripheral
GVI (`pGVI`), and the generative ANN/NURBS design loop of section 3.

**Two of ten fields have no usable measured counterpart** — SFV, where every
class tried comes out negative, and GMI, where the twin correlates well but
with the wrong thing (see section 4). This was five before the two-model
segmentation; V_sign, GFAPI and IAS now have one.

**Place Dependence remains the weakest founded of the three dimensions.**
`resting_affordance` reaches only +0.234 against actual seating classes where
greenery reaches +0.718, and neither rewording the rungs nor any of five
readouts moved it — the model can barely see seating in a 90-degree view.
`walkable_ground` reaches +0.248. Street View is captured from the roadway, so
the near sidewalk is often cropped or occluded, and that is the likely cause
for both.

---

## Statistical rules specific to this study

**Nodes are not independent.** Moran's I is +0.835 for GVI and +0.889 for VEI.
Every inferential number must cluster standard errors by `face_id` or
aggregate to block faces. Naive node-level p-values may be printed to show the
size of the gap, never reported as results.

**Read R², not p.** At this n a slope is significant while explaining nothing.

**Report rank correlation alongside OLS slopes.** OLS slope is cov/var, so a
handful of extreme-x points move it several-fold while Spearman barely shifts.

**Park Avenue moves everything.** Its planted median sits at GVI 14–17 where
every other street is under 5. Any fit that could be driven by it must be
reported with and without.

**Do not fit a functional form because the previous one failed.** A low linear
R² carries no information about curvature.

---

## Running it

```bash
python preflight.py                       # what is present, what is missing

.venv/Scripts/python main.py --from s04   # analysis, seconds, no GPU or key
.venv/Scripts/python tools/sim_compute.py # compose the index and validate
.venv/Scripts/python tools/sim_readme.py  # regenerate the data dictionary
python make_dashboard.py                  # results/dashboard.html
```

GPU stages, in the order they would be re-run from scratch:

```bash
.venv/Scripts/python tools/export_svi_90.py                     # raw -> svi_90
.venv/Scripts/python tools/export_svi_180.py --out data/raw/svi_180 --keep-tunnels
.venv-gpu/Scripts/python tools/seg_two_model.py                 # 19 min
.venv-gpu/Scripts/python tools/seg_two_model.py --src data/raw/svi_180 \
    --out data/processed/seg180_two_model.csv                   # 12 min
.venv-gpu/Scripts/python tools/sim_vlm_run.py --src data/raw/svi_90 \
    --table results/tables/sim_vlm.csv                          # 4.3 h
```

`--keep-tunnels` on the 180 export is not optional: without it a sky-share
heuristic and a hardcoded viaduct list drop 139 nodes, 106 of them Park Avenue
in one run, and the two imagery sets stop covering the same frame.

`seg_two_model.py` picks its mast calibration from the source folder name, so
`--src data/raw/svi_180` selects the two-mast one automatically. A folder with
no calibration raises rather than guessing.

**Long GPU jobs must be launched detached.** The harness reaps background
processes during model loading; the segmentation batch and two VLM runs were
killed that way before this was understood. Use PowerShell:

```powershell
Start-Process -FilePath ".venv-gpu\Scripts\python.exe" `
  -ArgumentList "tools/sim_vlm_run.py","--src","data/raw/svi_90" `
  -RedirectStandardOutput "run.log" -RedirectStandardError "run.err" `
  -WindowStyle Hidden
```

Full VLM re-rating is about 4.3 hours on an RTX 3080 Ti for 3,064 images ×
ten fields; rating by elimination is 9.4 h. Never run the full `main.py` to
test a change to stages 4–8; it re-checks every image and can trigger a
re-fetch.

Two environments, deliberately split. `.venv` is analysis-only and is the
mirror of what runs on a laptop with only `data/processed/` copied across.
`.venv-gpu` has torch and transformers. There is no `python` on PATH; call the
interpreter directly.
