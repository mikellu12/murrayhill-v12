# SIM run tables

Two tables from the same run, one row per 90-degree half-view, joined on
`file`. 3,064 rows, 766 nodes, four half-views each:
two walk directions by two sides.

| file | rows | columns | what it holds |
|---|---|---|---|
| `vlm_observations.csv` | 3,064 | 121 | everything observed |
| `vlm_calculations.csv` | 3,064 | 27 | everything derived from it |

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

| column | range in this file | meaning |
|---|---|---|
| `file` | 1st_avenue/north_to_south/001_n00045_S_L.jpg, 1st_avenue/north_to_south/001_n00045_S_R.jpg, 1st_avenue/north_to_south/002_n00044_S_L.jpg, 1st_avenue/north_to_south/002_n00044_S_R.jpg&hellip; | path under data/raw/svi_90, and the join key between the two files |
| `node_id` | n00000, n00001, n00002, n00003&hellip; | positional id against the frame geometry, from stage 1 |
| `street` | 1st_avenue, 2nd_avenue, 3rd_avenue, Park_Ave_East&hellip; | folder name, the cleaned-frame street label |
| `walk` | east_to_west, north_to_south, south_to_north, west_to_east | direction of travel; each street has two |
| `side` | L, R | L or R half of the forward view, relative to the direction of travel |
| `seq` | 1.000 to 57.000 | position along this walk, 1 = first node. The two directions sum to N+1 |
| `cardinal` | E, N, S, W | compass direction of travel, N/E/S/W |
| `osm_name` | 1st Avenue, 2nd Avenue, 3rd Avenue, East 34th Street&hellip; | street name from the network |
| `typology` | avenue_canyon, avenue_secondary, mid_block, other | avenue_canyon, avenue_secondary, mid_block or other |
| `face_id` | f1, f10, f11, f12&hellip; | block face, the unit standard errors are clustered on |

### the nine VLM ratings, 1-7

Each was asked in its own call, one field per prompt, with both ends of the
scale named. The meanings below are the manuscript's own, from section 2.7 and
Table 1 -- what the term *is*. How it was *asked* is a separate thing:
`tools/sim_vlm_run.py --show-prompts` prints the exact prompts, and the two
should be read against each other.

| column | term | scale | what the manuscript says it is |
|---|---|---|---|
| `vertical_greenery` | `V_nat` | 1-7 | Natural Elements Above-Ground. V_tree_canopy + V_green_facade + V_elevated_planters + V_hedge_wall. Counts vertical and overhead foveal and peripheral natural vegetation; excludes flat grass lawns and ground cover. |
| `vertical_hardscape` | `V_built` | 1-7 | Built Elements Above-Ground. V_building_wall + V_glazing + V_structural_column + V_overhead_barrier. Counts vertical built facade structures and elevated built hardscape; explicitly excludes all walkable ground surfaces like asphalt, concrete paving, and sidewalks. |
| `green_eye_level` | `GVI_eye` | 1-7 | Eye-level Green View. Eye-level foliage triggers soft fascination and stress reduction. |
| `sky_openness` | `SVF` | 1-7 | Sky View Factor. 1 - SVF is standardized morphological canyon enclosure in [0,1]. Vertical building proportions and sky exposure structure spatial containment. |
| `walkable_ground` | `V_pave` | 1-7 | Sidewalk &amp; Paver Grounding. V_sidewalk + V_paver: all walkable ground surfacing. |
| `green_softening` | `GMI` | 1-7 | Green Mitigation Interaction. Models how eye-level greenness softens spatial enclosure (H/W), mitigating perceived oppression to sharpen place imageability. |
| `signage_detail` | `V_sign` | 1-7 | Architectural detail &amp; signage density in [0,1]. Articulated facade details, historic elements and commercial signage create memorable visual anchors. |
| `facade_variation` | `SFV` | 1-7 | Street facade variation / articulation index in [0,1]. Prevents corridors from feeling homogeneous. |
| `ground_floor_activity` | `GFAPI` | 1-7 | Ground-Floor Permeability. Active ground-floor glazing and micro-rest infrastructure (benches, planter ledges). Minimizes spatial friction, extending dwelling times. |
| `resting_affordance` | `IAS` | 1-7 | Interface Affordance Score: micro-resting infrastructure and tactile seating ledges in [0, 1]. Stoops, seating ledges, low walls, benches -- somewhere to stop. Section 2.7 places it in Place Dependence; facade_variation stood in for it until this field existed, which left the dimension with no affordance term and put SFV in both Y and D. |

Every field uses the same 1-7 scale, and the rating stored is `round(EV)` --
the rung nearest the model's expected value across all seven, not the single
most likely token. Both alternatives are kept: see the trailing columns below.

An earlier run named only the two ends of the scale and **never answered 4 on
any field**; the midpoint carried real probability but was never the tallest
single bar, so greedy decoding could not emit it. Naming all seven rungs, and
reading the expected value rather than the argmax, fixed that: every field now
uses 4, from 4.6% to 73.3% of rows.

Spread is still uneven. `walkable_ground` uses three values and sits on 4 for
73% of rows, so it carries little information; `vertical_greenery` and
`sky_openness` use six. Treating any of them as an interval measure assumes
equal spacing the model may not be using, which is what `(r - 1) / 6` does.

### measured over the same 90 degrees

Sliced from `azimuth_profiles.npz` at the exact bearing the half-view was
rendered from, not over the node. A half facing a blank wall while the trees
stand opposite should read low, and a node-level share would count them.

| column | range in this file | meaning |
|---|---|---|
| `arc_vegetation` | 0.000 to 0.546 | vegetation share over this half-view's own 90 degrees |
| `arc_sky` | 0.000 to 0.445 | sky share over the same arc |
| `arc_building` | 0.000 to 0.805 | building share over the same arc |

### canyon geometry, and node-level context

| column | range in this file | meaning |
|---|---|---|
| `H_m` | 4.511 to 180.760 | median building height within 35 m of the node, metres |
| `W_facade` | 12.177 to 71.794 | facade-to-facade width from the cone probe, metres |
| `HW_facade` | 0.250 to 9.800 | H_m / W_facade. Blank where the probe found no facade both sides |
| `HW_effective` | 0.250 to 9.800 | HW_facade, or the street-segment median where it is blank |
| `HW_source` | measured, open_one_side, radius_mean, series | how the H/W was obtained; see the table below |
| `node_GVI` | 0.000 to 30.594 | node-level Green View Index over the full 360, per cent |
| `node_VEI` | 0.001 to 0.980 | node-level Visual Enclosure Index, building / (sky + building) |
| `node_SVF_band` | 0.000 to 0.237 | node-level sky share of the +-45 degree band. NOT a sky view factor |

**`HW_source` matters, and anything treating H/W as measured must filter on
it.** Four values:

| value | rows | what it means |
|---|---|---|
| `measured` | 2,336 | both walls found by the probe |
| `radius_mean` | 300 | mean of measured nodes on the same street within 28.3 m |
| `series` | 288 | nearest measured node walking along the street |
| `open_one_side` | 132 | one wall, the other side genuinely open -- no ratio exists |

`W` comes from a **band** probe: a parallel-sided corridor one node spacing
(20 m) wide, sampled by 9 rays, reduced by the nearest hit, reaching
40 m each side. Adjacent nodes therefore tile the street rather than
overlapping, and the probe cannot widen at range and punch through onto the
next block. A separate, longer 90 m fan is used *only* to decide why a
side came back empty, never to measure a width.

`open_one_side` is not missing data. Those nodes have no opposite wall, so
`HW_effective` is NaN by construction while Omega is 1.0 -- the value the
formula already gives any street below the comfort threshold. They also take
the manuscript's POPS/porous elasticities, which section 2.8 gives no H/W band
and which are otherwise unreachable.

### the other three readings, columns 32-121

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

| column | range in this file | meaning |
|---|---|---|
| `nat_built` | 0.125 to 0.600 | V_nat / V_built, from the two raw ratings. Spans 0.125 to 0.875 |
| `GVI_eye` | 0.000 to 0.833 | green_eye_level normalised |
| `GMI` | 0.167 to 0.833 | green_softening normalised |
| `V_sign` | 0.000 to 0.833 | signage_detail normalised |
| `SVF` | 0.000 to 0.833 | sky_openness normalised. Y uses (1 - SVF) |
| `SFV` | 0.167 to 0.833 | facade_variation normalised. Enters Y and D_raw both |
| `V_pave` | 0.333 to 0.667 | walkable_ground normalised |
| `GFAPI` | 0.167 to 0.833 | ground_floor_activity normalised |

### dimensions

| column | range in this file | meaning |
|---|---|---|
| `I_raw` | 0.097 to 0.756 | equal thirds of nat_built, GVI_eye, GMI |
| `I` | 0.226 to 0.999 | Imageability, I_raw through the perceptual sigmoid |
| `Y` | 0.222 to 0.833 | Identity. Linear, no sigmoid |
| `D_raw` | 0.167 to 0.667 | equal thirds of V_pave, SFV, GFAPI |
| `D` | 0.562 to 1.000 | Dependence, D_raw through its own sigmoid |

### discount, elasticities, score

| column | range in this file | meaning |
|---|---|---|
| `HW_effective` | 0.250 to 9.800 | HW_facade, or the street-segment median where it is blank |
| `Omega` | 0.310 to 1.000 | canyon discount. Exactly 1 below the threshold |
| `a` | 0.300 to 0.500 | Imageability elasticity, set by the H/W regime |
| `b` | 0.100 to 0.200 | Identity elasticity |
| `c` | 0.300 to 0.600 | Dependence elasticity. a + b + c = 1 at every row |
| `M` | 0.242 to 0.947 | the Street Interface Matrix score |

## The equation

    M = I^a * Y^b * D^c * Omega

    I = 1 / (1 + exp(-kI * (I_raw - tauI)))    I_raw = mean(V_nat/V_built, GVI_eye, GMI)
    Y = mean(V_sign, 1 - SVF, SFV)                                    linear, no sigmoid
    D = 1 / (1 + exp(-kD * (D_raw - tauD)))    D_raw = mean(V_pave, SFV, GFAPI)
    Omega = exp(-psi * max(0, H/W - Omega_th))

    kI = 12   tauI = 0.2
    kD = 15   tauD = 0.15
    psi = 0.15   Omega_th = 2

Section 2.7 of the manuscript. Multiplicative, not additive: a dimension at
zero collapses the whole score, which a weighted sum cannot express. The
exponents are elasticities summing to 1, not mixing weights, and section 2.8
shifts them by H/W regime.

## Known limits

- **8 rows have no `M`**, and they are all the same two
  nodes. Both sit *inside a road tunnel*: tiled walls, overhead lighting, and
  no sky, vegetation or building pixels anywhere in frame. Stage 4 refuses to
  produce GVI/VEI from a profile whose class rows are all zero, because that
  is the absence of a measurement rather than a measurement of nothing, so
  they never reach the geometry stage and carry a blank `HW_source`. They were
  included deliberately -- the frame keeps tunnel and viaduct nodes -- and the
  VLM rated them, so the ratings are present and only the score is missing.
  Every other node has an H/W.
- **Dependence is saturated.** `tauD` = 0.15
  against a `D_raw` median of 0.500, so
  96% of rows sit at the rail and D
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
