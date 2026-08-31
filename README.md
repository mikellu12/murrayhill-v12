# Murray Hill streetscape pipeline

Measuring the Street Interface Matrix — Place Imageability, Place Identity,
Place Dependence — at 20 m sampling nodes across Murray Hill, Manhattan, from
Google Street View imagery.

Two tracks run over the same frame. A **segmentation track** produces Green
View Index, Visual Enclosure Index and azimuthal class profiles with an
open-weight semantic segmenter. A **VLM track** rates ten perceptual fields
with a vision-language model, for the terms no segmenter can produce. The
first exists largely to validate the second: five of the ten VLM fields have a
measured counterpart in the same 90 degrees of view, and those five are the
only evidence that the five without one come from a model reading this imagery
sensibly.

The specification is `Nature.8.27.docx`, sections 2.7 and 2.8. Where the code
departs from it, the departure is stated here and in `config.yaml`, never left
implicit.

---

## The pipeline

```
1  nodes          OSM network -> 20 m nodes, headings, sequence
2  imagery        Street View metadata -> panoramas -> reprojected views
3  segmentation   per-node azimuthal profiles -> GVI, VEI, arc shares
4  VLM ratings    ten fields, one call per field, logits read not generated
5  compose        I, Y, D, Omega -> M, and validate against the segmenter
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
          627 nodes x 2 walks = 1,254 images
          8 px/degree                            <- superseded; holdout only
```

Rendered, not cropped. Halving the field of view doubles both the angular
resolution and the vertical extent, so the 90-degree halves carry twice the
detail of a crop from the 180 strip.

Left and right are relative to **direction of travel**, not compass. Two walks
per street and two halves per walk give four views per node, covering two
frontages each seen from both approaches — a free consistency check, since a
frontage that rates differently depending on which way it is passed is either
a real directional effect or model noise.

**Planned, not implemented:** selecting the field of view by street type — 90
degrees where the camera sits far from the frontage on a wide vehicular road,
180 degrees on narrow pedestrian ways. Nothing in the code makes that choice
today; `svi_180` is a strict subset of `svi_90` covering the same streets, and
every node has both.

**Imagery is not redistributable.** Google caps caching at 30 days. Derived
profiles and metrics are ours to publish; the JPEGs are scratch and are
git-ignored.

---

## 3. Segmentation

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

| field | manuscript term | measured twin |
|---|---|---|
| `vertical_greenery` | V_nat | vegetation share over the arc |
| `vertical_hardscape` | V_built | building share over the arc |
| `green_eye_level` | GVI_eye | vegetation below the horizon |
| `sky_openness` | SVF | sky share over the arc |
| `green_softening` | GMI | — |
| `walkable_ground` | V_pave | — |
| `signage_detail` | V_sign | — |
| `facade_variation` | SFV | — |
| `ground_floor_activity` | GFAPI | — |
| `resting_affordance` | IAS | — |

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

### Outputs

```
results/tables/vlm_observations.csv       3,064 x 111   everything observed
results/tables/vlm_calculations.csv   3,064 x  26   everything derived
results/tables/vlm_sections.csv         121 x   5   per-section aggregate
results/tables/README.md                            generated data dictionary
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

**Five of ten fields have no measured counterpart** — GMI, V_sign, SFV, GFAPI
and IAS. Nothing in this repository validates them. Place Dependence is the
weakest founded of the three dimensions: two of its three terms are
unvalidated and the third, `walkable_ground`, correlates with nothing
measurable (ρ −0.04 against building share, +0.04 against sky). Street View is
captured from the roadway, so the near sidewalk is often cropped or occluded.

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

Full VLM re-rating is about 4.3 hours on an RTX 3080 Ti for 3,064 images ×
ten fields. Never run the full `main.py` to test a change to stages 4–8; it
re-checks every image and can trigger a re-fetch.

Two environments, deliberately split. `.venv` is analysis-only and is the
mirror of what runs on a laptop with only `data/processed/` copied across.
`.venv-gpu` has torch and transformers. There is no `python` on PATH; call the
interpreter directly.
