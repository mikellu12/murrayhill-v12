# Murray Hill streetscape pipeline — v12

Green View Index (GVI) and Visual Enclosure Index (VEI) at 20 m sampling
nodes across Murray Hill, Manhattan, from Google Street View imagery, using
open-weight semantic segmentation; validated against building-footprint
geometry.

## What is new in v12

**GVI against H/W on pre-specified bands (Stage 7).** The framework
document's section 4 fixes four cut points — 0.3, 0.8, 1.5, 3.0 — before
any GVI in this frame was inspected, which makes a shape hypothesis over
them testable. Stage 7 asks whether GVI peaks inside the human-scale band
and tests it with the Lind & Mehlum (2010) joint test rather than a
significant squared term. **It does not: GVI declines monotonically.** See
"The enclosure result" below — including what that failure does and does
not license, because the document does not actually predict a GVI peak.

**Zones are gone.** The three named zones were a latitude cut across a grid
rotated ~29°, so the boundary ran diagonally through the street pattern and
split all fifteen streets between two categories — East 38th Street had 43
nodes in one and 1 in another. They are replaced by `northing_m`, metres
along the grid's uptown axis, in `gridaxis.py`. The continuous version is
strictly better: at block-face level the zones gave adj R² 0.02 (p = 0.33)
while `northing_m` gives R² 0.219 (p = 0.028). The binning was destroying a
real gradient.

**`skyview.py` is gone.** `SVF_band` and `BVF_band` are computed inside
Stage 4 straight from the azimuthal profiles, which is where the profiles
already were. Two files computing the same quantity is two places for the
definition to drift. The closed-form canyon check moved to `common.py` as
`theoretical_svf()` / `theoretical_svf_band()`.

**Directional inference fixed (Stage 6).** The per-view regressions now
cluster by block face (25 groups) rather than by street (15 — too few for
cluster-robust standard errors), and the output table carries `rho`
alongside `slope` so the leverage problem is visible. A new
along-street/cross-street regression reports the contrast the design
actually supports.

**Coverage audit records the reason.** Stage 4 joins the Stage 2 metadata
probe, so `nodes_without_metrics.csv` says *why* each node has none and
flags runs of consecutive IDs. This matters: in the current run all 22
excluded nodes have working panoramas within 5.4 m and are excluded purely
by the capture-date filter. None is a coverage gap. The old message said
otherwise.

## The enclosure result

Reported here because it is the headline and it is negative. Read the
scope note at the end of this section before quoting it: what fails is a
proxy of our own construction, not the framework's envelope.

| specification | R² | AIC | CV RMSE |
|---|---|---|---|
| linear | 0.1129 | 2452.9 | 4.504 |
| quadratic | 0.1278 | 2447.7 | 4.473 |
| log | 0.0984 | 2459.7 | 4.540 |
| reciprocal | 0.0155 | 2496.7 | 4.864 |

Lind & Mehlum joint test, all three parts required:

| condition | value | result |
|---|---|---|
| b₂ < 0 (concave) | **+0.128** (p = 0.003) | FAIL |
| slope > 0 at the low end | −1.79 (t = −4.08) | FAIL |
| slope < 0 at the high end | +0.78 (t = +1.62) | FAIL |

The quadratic term is **positive**. That is a U, not an inverted-U — the
curve is convex, and the fitted turning point sits at H/W 7.03 with 18 of
421 nodes beyond it. What the data describe is a monotone decline that
flattens near zero, not a peak at human scale.

The framework's own bands agree:

| band | n | median GVI | median GVI, no Park Ave |
|---|---|---|---|
| <0.3 under-enclosed | 8 | 3.93 | 3.93 |
| 0.3–0.8 transitional | 98 | 4.06 | 3.99 |
| 0.8–1.5 human scale | 135 | 3.83 | 2.90 |
| 1.5–3.0 enclosed | 95 | 2.15 | 2.15 |
| >3.0 deep canyon | 85 | 0.23 | 0.23 |

Flat, then falling. Excluding Park Avenue it is monotone decreasing
throughout. The *mean* shows a bump at human scale (6.17) which the median
does not — that bump is Park Avenue's planted median, one street at
H/W ≈ 1.2, not an envelope effect.

Two scope limits worth stating in the paper rather than burying:

- **The under-enclosed band holds 8 nodes on 7 faces.** Murray Hill has
  essentially no under-enclosed streets, so the framework's low-end
  prediction is *out of scope here*, not falsified.
- **The envelope itself is untested here, and this is not a test of it.**
  Section 4's three bullets, verbatim, say that under-enclosure leaves
  eye-level greenery *visually diluted*, that human scale *frames* it, and
  that a deep canyon is oppressive *unless offset by* rich greenery. Every
  one is about what enclosure does to the experience of greenery. None
  says greenery is more abundant at human scale. The document's own model
  puts GVI under Place Attachment and canyon H/W under Place Identity —
  separate arguments to the same function, not one predicting the other.

  So "GVI peaks at human scale" is a proxy we chose, not the framework's
  proposition, and its failure is a finding about greenery and geometry in
  Murray Hill rather than a verdict on the envelope. What carries over
  from the document is the part that makes the test well-posed: the band
  edges, fixed in advance.

- **There is no outcome variable, so the claim the document *does* make
  cannot be estimated at all.** The third bullet — a deep canyon is
  oppressive unless offset by rich eye-level greenery — is an explicit
  GVI × H/W interaction on sense of place. Nothing in this dataset
  measures sense of place: no survey, no dwell time, no intercept counts.
  That interaction is not weakly supported here; it is unmeasured, and it
  should be written up as a hypothesis awaiting outcome data.

### Why a curve with a turning point, and not a correlation line

Because the band edges were fixed in advance, so a non-monotone shape over
them was specified before the data were seen — not because a straight line
fitted poorly. The distinction decides whether the result is publishable:

1. **A low linear R² carries no information about curvature.** It is
   equally consistent with no relationship, a monotone one buried in noise,
   and any non-linear shape whatever. Rejecting a line tells you the line
   is wrong, never what is right.
2. **The quadratic nests the linear**, so adding x² cannot lower in-sample
   R². "The curve fits better" is arithmetic. Only the *sign* of b₂ and the
   location of the turning point say anything about shape — which is why
   Stage 7 reports those and not the R² gain.
3. **Choosing the form after seeing the first fit fail, on the same data,
   is specification search.** The p-value that comes out is not the p-value
   it claims to be.

Stage 7 is defensible because the bands came from the framework document
before any GVI was inspected, and because it reports the answer whichever
way it falls. Here it falls against the proxy hypothesis — which is a
smaller claim than it first appears, for the reason set out above.

## Geometry notes

Manhattan's grid is rotated ~29° east of true north (Commissioners' Plan,
1811, set perpendicular to the Hudson shoreline), so travel bearings are
029/119/209/299 rather than 0/90/180/270, and the north–south gradient is
projected onto 029 rather than onto latitude — worth about double the
explained variance (R² 0.097 against 0.053).

Degrees map to pixel columns **gnomonically**, not linearly: a fov-90 image
is a perspective projection, so the column at 29° off centre is 497 of 640,
not 526. A linear reading displaces every direction boundary by ~2°.

Pixels are weighted by solid angle, `(1 + x² + y²)^(-3/2)`.

### The four travel directions are two partitions, not four samples

N and S are disjoint halves that tile the circle; so are E and W. Each pair
is a complete decomposition of the same node. Reading five rows of a
by-direction table as five independent findings is a mistake the v12 output
now warns about inline.

Which views are *along-street* also flips with typology — on the avenues
(axis ~029) N and S run along the corridor; on the cross streets (axis
~119) E and W do. So every compass row mixes both viewing situations in
whatever ratio the node counts give. Report the along/cross contrast.

## Known limitation: vertical coverage

Pitch 0 at fov 90 sees only ±45° of elevation. Overhead canopy and most of
the visible sky are never sampled, which biases **GVI low** and **VEI high**
by an unmeasured amount. Sky fractions are named `SVF_band` throughout and
must never be reported as SVF.

`tools/cubemap_check.py` measures the bias. Adding pitch ±90 completes a
six-face cubemap that tiles the sphere exactly once — strictly better than
Treepedia's 6 headings × 3 pitches, which overlaps heavily and has no
closed-form solid angle. It runs on a stratified subsample first, because
it is the only thing in the repo that issues fresh API requests (6 per
node, ~$4 for 100 nodes). Until it has been run, the direction of the bias
is known and the magnitude is not, so the bias must not be described as
small.

This matters more in v12 than it did before: the framework's first claim is
that *eye-level* vegetation drives pedestrian GVI more than overhead
canopy. The ±45° band is centred on the horizon, so it is reasonably well
aligned with that claim — but it cannot separate eye-level from canopy
without pitch information. **"Eye-level GVI" is therefore not yet a
quantity this pipeline measures**, and the framework's first claim cannot
be tested until the cubemap runs. Until then, write the GVI here as
horizon-band GVI and say what it does not distinguish.

## Layout

```
config.yaml              every parameter; nothing hardcoded
main.py                  runs the pipeline
gridaxis.py              grid-axis position; replaces the zone categories
migrate_gridaxis.py      upgrade a v11 run in place, no refetch
make_dashboard.py        self-contained HTML report
run_analysis.py          laptop entry point, no GPU

src/common.py            config, paths, typology, image geometry
src/s01_frame.py         street network -> 20 m frame -> typology, grid axis
src/s02_imagery.py       metadata probe -> Street View retrieval
src/s03_profiles.py      segmentation -> 360-bin azimuthal profiles
src/s04_metrics.py       GVI/VEI x 5 views, SVF_band, coverage audit,
                         block faces, Moran
src/s05_geometry.py      footprints, measured facade width, H/W
src/s06_analysis.py      regressions, typology contrasts, gradient, per street
src/s07_enclosure.py     GVI vs H/W, inverted-U test, pre-specified bands
src/s08_figures.py       maps, scatters, leverage, directional panels, roses

tools/pedestrian.py      pedestrian-realm segmentation, per street
tools/sidewalk.py        sidewalk width and setback vs directional GVI
tools/fov_check.py       is 180 deg the right forward view? FOV sweep
tools/cubemap_check.py   measures the vertical-coverage bias  [GPU, $]
tools/factor_check.py    factor analysis of the directional metrics
```

## Moving an existing run into this folder

Copy `data/` across from the v11 run, then check before you run:

```bash
python preflight.py
```

It reports what is present, what each missing file would cost, and whether
the frame and the profiles actually agree.

The minimum worth copying:

```
data/processed/azimuth_profiles.npz     <- the expensive one
data/processed/nodes.gpkg               <- avoids an OSM re-query
data/processed/manifest.csv
data/processed/scaffold_by_node.csv
data/raw/metadata.csv
data/raw/building_footprints.geojson
data/raw/svi/                           <- only for re-segmentation and
                                           tools/pedestrian.py
```

`azimuth_profiles.npz` is the output of Stage 3 and the JPEGs are its
input, so bringing `svi/` without the profiles re-segments everything.
Everything under "optional" in the preflight output is rebuilt by Stage 4
and copying it saves nothing.

## Upgrading from v11

Removing the zones does **not** invalidate the frame. Zone was computed
from lat/lon *after* node IDs were assigned; it never touched placement,
dedupe or ordering. No refetch:

```bash
python migrate_gridaxis.py           # dry run
python migrate_gridaxis.py --apply
python main.py --from s04            # seconds, no GPU, no key, no network
```

This edits `nodes.gpkg` in place rather than re-running Stage 1. Re-running
Stage 1 would also be free, but it re-queries OSM — and if the cache has
expired and anything upstream changed, node IDs shift and Stage 2 aborts on
the stale-imagery check, which *would* then force a refetch.

## Setup

Python 3.12 — no CUDA wheels exist for 3.14.

```bash
python3.12 -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

`.env` beside `config.yaml`:

```
GMAPS_KEY=your_key
```

On macOS omit the `--index-url`; the default PyPI wheel includes MPS.

## Running

```bash
python main.py                  # everything, ~90 min on a 12 GB GPU
python main.py --from s04       # resume from a stage
python main.py --only s07 s08   # enclosure and figures only
python make_dashboard.py        # then open results/dashboard.html
```

Stages 1–3 need a GPU and a key. Stages 4–8 run on CPU from saved data in
under five minutes, which is the path for re-running analysis.

**If you change the frame geometry, delete `data/processed/` first** — node
IDs restart at n00000 against the new geometry, so stale imagery would be
silently reused for the wrong locations. Changing only `directional:`,
`enclosure:` or `spatial:` is safe; `--from s04` is enough.

## Inference

Moran's I is ~0.62–0.66: nodes 20 m apart on a block face are not
independent, and node-level p-values are inflated. Report either the
face-clustered standard errors or the block-face aggregation. GVI is a
bounded proportion with a floor at zero, so a quasi-binomial GLM is
reported alongside OLS.

Park Avenue is the single most influential street in this frame — its
planted median sits at GVI 14–17 where every other street is under 5. Every
stage that can be moved by it now reports the fit with and without it. That
is not a licence to drop it; the median is real streetscape. It is a
requirement to say which fit the text is quoting.

## Data provenance

| source | licence |
|---|---|
| Google Street View Static API | © Google. Imagery not redistributable (30-day cache limit); derived profiles and metrics are |
| NYC Building Footprints | NYC Open Data, public domain |
| OpenStreetMap via OSMnx | © OSM contributors, ODbL |

## Expected wall time, ~600 nodes

| machine | stage 3 (segmentation) | full pipeline |
|---|---|---|
| RTX 3080 Ti | ~35 min | ~60 min |
| M3 Pro / M4 (MPS) | ~60–90 min | ~90–120 min |
| M1 / M2 base (MPS) | ~2–3 h | ~2.5–3.5 h |
| CPU only | ~4–6 h | ~5–7 h |

Stage 3 is the only stage needing a GPU. Run 1–3 once on the CUDA machine,
copy `data/processed/` (a few MB without the JPEGs) to any laptop, then:

```bash
pip install -r requirements-analysis.txt   # no torch
python run_analysis.py
```
