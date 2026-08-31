"""Write the data dictionary for vlm_observations.csv and vlm_calculations.csv.

Generated rather than hand-written so the column lists, ranges and row counts
come from the files themselves and cannot drift from them. Re-run after any
recompute.

    .venv/Scripts/python tools/sim_readme.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import CFG, RES, banner
from sim_fields import FIELDS

OUT = RES / "tables" / "README.md"

DESC = {
    "file": "path under data/raw/svi_90, and the join key between the two files",
    "node_id": "positional id against the frame geometry, from stage 1",
    "street": "folder name, the cleaned-frame street label",
    "walk": "direction of travel; each street has two",
    "side": "L or R half of the forward view, relative to the direction of travel",
    "seq": "position along this walk, 1 = first node. The two directions sum to N+1",
    "cardinal": "compass direction of travel, N/E/S/W",
    "osm_name": "street name from the network",
    "typology": "avenue_canyon, avenue_secondary, mid_block or other",
    "face_id": "block face, the unit standard errors are clustered on",
    "arc_vegetation": "vegetation share over this half-view's own 90 degrees",
    "arc_sky": "sky share over the same arc",
    "arc_building": "building share over the same arc",
    "H_m": "median building height within 35 m of the node, metres",
    "W_facade": "facade-to-facade width from the cone probe, metres",
    "HW_facade": "H_m / W_facade. Blank where the probe found no facade both sides",
    "HW_effective": "HW_facade, or the street-segment median where it is blank",
    "HW_source": "how the H/W was obtained; see the table below",
    "node_GVI": "node-level Green View Index over the full 360, per cent",
    "node_VEI": "node-level Visual Enclosure Index, building / (sky + building)",
    "node_SVF_band": "node-level sky share of the +-45 degree band. NOT a sky view factor",
    "nat_built": "V_nat / V_built, from the two raw ratings. Spans 0.125 to 0.875",
    "GVI_eye": "green_eye_level normalised",
    "GMI": "green_softening normalised",
    "V_sign": "signage_detail normalised",
    "SVF": "sky_openness normalised. Y uses (1 - SVF)",
    "SFV": "facade_variation normalised. Enters Y and D_raw both",
    "V_pave": "walkable_ground normalised",
    "GFAPI": "ground_floor_activity normalised",
    "IAS": "resting_affordance normalised",
    "I_raw": "equal thirds of nat_built, GVI_eye, GMI",
    "I": "Imageability, I_raw through the perceptual sigmoid",
    "Y": "Identity. Linear, no sigmoid",
    "D_raw": "equal thirds of V_pave, SFV, GFAPI",
    "D": "Dependence, D_raw through its own sigmoid",
    "Omega": "canyon discount. Exactly 1 below the threshold",
    "a": "Imageability elasticity, set by the H/W regime",
    "b": "Identity elasticity",
    "c": "Dependence elasticity. a + b + c = 1 at every row",
    "M": "the Street Interface Matrix score",
}


# The manuscript's own definition of each term, section 2.7 and Table 1.
# Quoted rather than paraphrased: the prompt anchors are how the field was
# ASKED, these are what the paper says the field IS, and the two should be
# checkable against each other without going back to the docx.
PAPER = {
 "vertical_greenery": (
   "V_nat", "Natural Elements Above-Ground. V_tree_canopy + V_green_facade + "
   "V_elevated_planters + V_hedge_wall. Counts vertical and overhead foveal and "
   "peripheral natural vegetation; excludes flat grass lawns and ground cover."),
 "vertical_hardscape": (
   "V_built", "Built Elements Above-Ground. V_building_wall + V_glazing + "
   "V_structural_column + V_overhead_barrier. Counts vertical built facade "
   "structures and elevated built hardscape; explicitly excludes all walkable "
   "ground surfaces like asphalt, concrete paving, and sidewalks."),
 "green_eye_level": (
   "GVI_eye", "Eye-level Green View. Eye-level foliage triggers soft "
   "fascination and stress reduction."),
 "sky_openness": (
   "SVF", "Sky View Factor. 1 - SVF is standardized morphological canyon "
   "enclosure in [0,1]. Vertical building proportions and sky exposure "
   "structure spatial containment."),
 "walkable_ground": (
   "V_pave", "Sidewalk &amp; Paver Grounding. V_sidewalk + V_paver: all "
   "walkable ground surfacing."),
 "green_softening": (
   "GMI", "Green Mitigation Interaction. Models how eye-level greenness softens "
   "spatial enclosure (H/W), mitigating perceived oppression to sharpen place "
   "imageability."),
 "signage_detail": (
   "V_sign", "Architectural detail &amp; signage density in [0,1]. Articulated "
   "facade details, historic elements and commercial signage create memorable "
   "visual anchors."),
 "facade_variation": (
   "SFV", "Street facade variation / articulation index in [0,1]. Prevents "
   "corridors from feeling homogeneous."),
 "ground_floor_activity": (
   "GFAPI", "Ground-Floor Permeability. Active ground-floor glazing and "
   "micro-rest infrastructure (benches, planter ledges). Minimizes spatial "
   "friction, extending dwelling times."),
 "resting_affordance": (
   "IAS", "Interface Affordance Score: micro-resting infrastructure and "
   "tactile seating ledges in [0, 1]. Stoops, seating ledges, low walls, "
   "benches -- somewhere to stop. Section 2.7 places it in Place Dependence; "
   "facade_variation stood in for it until this field existed, which left the "
   "dimension with no affordance term and put SFV in both Y and D."),
}


def ratings_table(d):
    """Every field on the same 1-7 scale, with the manuscript's definition.

    Which points the model actually reached for is a finding about the model,
    not part of the definition, so it sits under the table rather than in it.
    """
    out = ["| column | term | scale | what the manuscript says it is |",
           "|---|---|---|---|"]
    for f in FIELDS:
        term, meaning = PAPER[f]
        out.append(f"| `{f}` | `{term}` | 1-7 | {meaning} |")
    return "\n".join(out)


def table(d, cols, ratings=False):
    out = ["| column | range in this file | meaning |", "|---|---|---|"]
    for c in cols:
        if c not in d.columns:
            continue
        s = d[c].dropna()
        if s.empty:
            rng = "all blank"
        elif pd.api.types.is_numeric_dtype(s):
            rng = (f"{s.min():.0f} to {s.max():.0f}" if ratings
                   else f"{s.min():.3f} to {s.max():.3f}")
        else:
            v = s.unique()
            rng = ", ".join(map(str, sorted(v)[:4])) + ("&hellip;" if len(v) > 4 else "")
        out.append(f"| `{c}` | {rng} | {DESC.get(c, '')} |")
    return "\n".join(out)


def main():
    banner("data dictionary")
    m = pd.read_csv(RES / "tables" / "vlm_observations.csv")
    c = pd.read_csv(RES / "tables" / "vlm_calculations.csv")
    R = list(FIELDS)
    ID = ["file", "node_id", "street", "walk", "side", "seq", "cardinal"]
    CD = CFG["sim_vlm"]["cobb_douglas"]
    src = m.HW_source.value_counts()
    G = CFG["geometry"]
    rad = G["hw_fill_radius_m"]
    bw = CFG["sampling"]["grid_spacing_m"]
    nray = G.get("hw_band_rays", 9)
    half = G["facade_half_m"]
    oreach = G.get("facade_open_reach_m", half)
    p4 = [(m[f] == 4).mean() * 100 for f in R]
    p4lo, p4hi = min(p4), max(p4)
    wg = (m.walkable_ground == 4).mean() * 100
    tail = [c for c in m.columns if c.endswith("_ev")
            or c.endswith("_argmax") or (c[-2] == "p" and c[-1].isdigit())]
    tailfrom = m.shape[1] - len(tail) + 1
    tailto = m.shape[1]

    md = f"""# SIM run tables

Two tables from the same run, one row per 90-degree half-view, joined on
`file`. {len(m):,} rows, {m.node_id.nunique()} nodes, four half-views each:
two walk directions by two sides.

| file | rows | columns | what it holds |
|---|---|---|---|
| `vlm_observations.csv` | {len(m):,} | {m.shape[1]} | everything observed |
| `vlm_calculations.csv` | {len(c):,} | {c.shape[1]} | everything derived from it |

`vlm_observations.csv` holds two kinds of measurement side by side: the ratings
the VLM gave, and the `arc_*` shares a segmenter gave over the same arc. That
pairing is the point of the file, but the `arc_*` columns did not come from
the model.

## Reading a filename

    35_n00045_S_L.jpg
    │   │      │  └─ side: LEFT half of the forward view
    │   │      └──── compass direction of travel
    │   └─────────── node id, positional against the frame
    └─────────────── 35th node along this walk

The sequence prefix makes a sorted directory listing the walk itself. A node's
two sequence numbers, one per direction, always sum to N+1 for a street of N
nodes.

## vlm_observations.csv

### identity

{table(m, ID + ["osm_name", "typology", "face_id"])}

### the nine VLM ratings, 1-7

Each was asked in its own call, one field per prompt, with both ends of the
scale named. The meanings below are the manuscript's own, from section 2.7 and
Table 1 -- what the term *is*. How it was *asked* is a separate thing:
`tools/sim_vlm_run.py --show-prompts` prints the exact prompts, and the two
should be read against each other.

{ratings_table(m)}

Every field uses the same 1-7 scale, and the rating stored is `round(EV)` --
the rung nearest the model's expected value across all seven, not the single
most likely token. Both alternatives are kept: see the trailing columns below.

An earlier run named only the two ends of the scale and **never answered 4 on
any field**; the midpoint carried real probability but was never the tallest
single bar, so greedy decoding could not emit it. Naming all seven rungs, and
reading the expected value rather than the argmax, fixed that: every field now
uses 4, from {p4lo:.1f}% to {p4hi:.1f}% of rows.

Spread is still uneven. `walkable_ground` uses three values and sits on 4 for
{wg:.0f}% of rows, so it carries little information; `vertical_greenery` and
`sky_openness` use six. Treating any of them as an interval measure assumes
equal spacing the model may not be using, which is what `(r - 1) / 6` does.

### measured over the same 90 degrees

Sliced from `azimuth_profiles.npz` at the exact bearing the half-view was
rendered from, not over the node. A half facing a blank wall while the trees
stand opposite should read low, and a node-level share would count them.

{table(m, ["arc_vegetation", "arc_sky", "arc_building"])}

### canyon geometry, and node-level context

{table(m, ["H_m", "W_facade", "HW_facade", "HW_effective", "HW_source",
           "node_GVI", "node_VEI", "node_SVF_band"])}

**`HW_source` matters, and anything treating H/W as measured must filter on
it.** Four values:

| value | rows | what it means |
|---|---|---|
| `measured` | {int(src.get('measured', 0)):,} | both walls found by the probe |
| `radius_mean` | {int(src.get('radius_mean', 0)):,} | mean of measured nodes on the same street within {rad:.1f} m |
| `series` | {int(src.get('series', 0)):,} | nearest measured node walking along the street |
| `open_one_side` | {int(src.get('open_one_side', 0)):,} | one wall, the other side genuinely open -- no ratio exists |

`W` comes from a **band** probe: a parallel-sided corridor one node spacing
({bw:g} m) wide, sampled by {nray} rays, reduced by the nearest hit, reaching
{half:g} m each side. Adjacent nodes therefore tile the street rather than
overlapping, and the probe cannot widen at range and punch through onto the
next block. A separate, longer {oreach:g} m fan is used *only* to decide why a
side came back empty, never to measure a width.

`open_one_side` is not missing data. Those nodes have no opposite wall, so
`HW_effective` is NaN by construction while Omega is 1.0 -- the value the
formula already gives any street below the comfort threshold. They also take
the manuscript's POPS/porous elasticities, which section 2.8 gives no H/W band
and which are otherwise unreachable.

### the other three readings, columns {tailfrom}-{tailto}

The same VLM answer, three more ways. Parked at the end of the file because
they are rarely needed day to day.

| suffix | n | what it is |
|---|---|---|
| `*_ev` | 9 | expected value over the seven rungs, before rounding |
| `*_argmax` | 9 | the rung a `generate()` call would have written |
| `*_p1` .. `*_p7` | 63 | the full probability distribution per field |

The distributions are the raw evidence: `round(EV)` and `argmax` are both
derivable from them, so any other reading can be recomputed without the GPU.

## vlm_calculations.csv

Applied in this order: normalise, compose, threshold, discount, combine.

### normalised inputs

Every rating maps to [0,1] by `(r - 1) / 6`. `nat_built` is the exception: it
is a ratio of two raw ratings, so it spans 0.125 to 0.875 and cannot reach
either end.

{table(c, ["nat_built", "GVI_eye", "GMI", "V_sign", "SVF", "SFV", "V_pave",
           "GFAPI"])}

### dimensions

{table(c, ["I_raw", "I", "Y", "D_raw", "D"])}

### discount, elasticities, score

{table(c, ["HW_effective", "Omega", "a", "b", "c", "M"])}

## The equation

    M = I^a * Y^b * D^c * Omega

    I = 1 / (1 + exp(-kI * (I_raw - tauI)))    I_raw = mean(V_nat/V_built, GVI_eye, GMI)
    Y = mean(V_sign, 1 - SVF, SFV)                                    linear, no sigmoid
    D = 1 / (1 + exp(-kD * (D_raw - tauD)))    D_raw = mean(V_pave, SFV, GFAPI)
    Omega = exp(-psi * max(0, H/W - Omega_th))

    kI = {CD['imageability_sigmoid']['kappa']:g}   tauI = {CD['imageability_sigmoid']['tau']:g}
    kD = {CD['dependence_sigmoid']['kappa']:g}   tauD = {CD['dependence_sigmoid']['tau']:g}
    psi = {CD['omega']['psi']:g}   Omega_th = {CD['omega']['hw_threshold']:g}

Section 2.7 of the manuscript. Multiplicative, not additive: a dimension at
zero collapses the whole score, which a weighted sum cannot express. The
exponents are elasticities summing to 1, not mixing weights, and section 2.8
shifts them by H/W regime.

## Known limits

- **{int(c.M.isna().sum())} rows have no `M`**, and they are all the same two
  nodes. Both sit *inside a road tunnel*: tiled walls, overhead lighting, and
  no sky, vegetation or building pixels anywhere in frame. Stage 4 refuses to
  produce GVI/VEI from a profile whose class rows are all zero, because that
  is the absence of a measurement rather than a measurement of nothing, so
  they never reach the geometry stage and carry a blank `HW_source`. They were
  included deliberately -- the frame keeps tunnel and viaduct nodes -- and the
  VLM rated them, so the ratings are present and only the score is missing.
  Every other node has an H/W.
- **Dependence is saturated.** `tauD` = {CD['dependence_sigmoid']['tau']:g}
  against a `D_raw` median of {c.D_raw.median():.3f}, so
  {((c.D > 0.95) | (c.D < 0.05)).mean()*100:.0f}% of rows sit at the rail and D
  carries little into M. The manuscript calibrates tau per study area; these
  values were set for pixel shares, not normalised ratings.
- **`node_SVF_band` is not SVF.** The imagery spans +-45 degrees of elevation,
  so the zenith is never sampled. Do not compare it to published SVF values.
- **Five of the nine ratings have no measured counterpart** -- GMI, V_sign,
  SFV, GFAPI and V_pave. Nothing in these files validates them; that is what
  the segmentation-mask comparison is for.

## Provenance

| step | tool | output |
|---|---|---|
| render half-views | `tools/export_svi_90.py` | `data/raw/svi_90/` |
| rate them | `tools/sim_vlm_run.py` | `results/tables/sim_vlm_v3.csv` |
| canyon geometry | `src/s05_geometry.py` | `data/processed/metrics.csv` |
| compose the index | `tools/sim_compute.py` | `vlm_observations.csv` + `vlm_calculations.csv` |
| split these two | `tools/sim_export.py` | this pair |
| this file | `tools/sim_readme.py` | `README.md` |

Field definitions and prompts live in `src/sim_fields.py`; the equations in
`src/sim_core.py`; every constant in `config.yaml`.
"""
    OUT.write_text(md, encoding="utf-8")
    print(f"wrote {OUT}  ({len(md.splitlines())} lines)")


if __name__ == "__main__":
    main()
