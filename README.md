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

