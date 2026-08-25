# Frame source of record

The sampling frame is not built by `s01_frame.py` in this version. It was
authored externally and imported, so `python main.py` from a clean clone would
produce a *different* frame. These files are that missing input, committed so
the frame is reproducible rather than merely present.

## Lineage

```
final_nodes_output.gpkg      1532 rows, 2026-08-20   image-keyed, pre-cleaning
final_coordinates_mapping.csv 1532 rows, 2026-08-20   positions + grid coordinates
final_nodes_cleaned.gpkg     1424 rows, 2026-08-21   712 locations, hand-checked
        |
        |  tools/import_frame.py
        v
data/processed/nodes_v13.gpkg   766 nodes
        |  tools/repair_chain_pos.py --apply
        |  tools/study_area_filter.py --apply
        |  tools/cleaned_frame_flag.py <cleaned> --apply
        v
data/processed/nodes.gpkg       766 nodes, 657 in_study, 712 in_cleaned
```

## What the cleaning decided

`final_nodes_cleaned.gpkg` keeps 712 of the 766 locations, matching them to
**0.00 m**, so it is a strict subset: a set of decisions about the frame, not
a different frame. The 54 it removes are coherent:

| | dropped (54) | kept (712) |
|---|---|---|
| `is_tunnel` | 36 (67%) | 42 (5.9%) |
| median GVI | 1.187 | 2.277 |
| GVI exactly 0 | 8 (14.8%) | 22 (3.1%) |

They are Park Avenue's tunnel segment, its surplus carriageways, and the four
viaduct nodes `export_svi_180.py` carries as a hardcoded `VIADUCT_NODES` list.
Tunnel Exit Street is kept — the same call `study_area_filter.py` argues in
prose, since that street has one roadway and duplicates nothing.

## Why it is applied as a flag

`node_id` is positional. Re-importing renumbers every node, and 2,940 image
filenames, both profile arrays, `metrics.csv`, `sim_index.csv`, the
1,254-image export and every table built from them are keyed to the current
ids. `cleaned_frame_flag.py` adds columns instead:

    in_cleaned          kept by the cleaning
    cleaned_street      its street label
    cleaned_direction   Northbound / Southbound / Eastbound / Westbound
    cleaned_id          its own id, e.g. 1st_avenue_001

Impact on current results: **2 nodes, 4 images** of 1,254.

## The street organisation is the valuable part

Ordering measured as Spearman of position-along-street against each street's
principal axis:

| | streets ordered perfectly |
|---|---|
| cleaned file | **19 of 20** |
| live `chain_pos_m` | 15 of 19 |

The live frame still has `Park_Ave_Tunnel_Segment` at 0.318 and
`Park_Ave_West` at 0.474 *after* `repair_chain_pos.py` ran. The cleaned file
avoids both by splitting the tangled chains into `1st_avenue_west_branch` and
`tunnel_approach_street`. Both broken chains are `in_study = False`, so the
bug is quarantined and has not corrupted results — but it is real.

## final_coordinates_mapping.csv

`along_dist` and `cross_dist` are **grid** coordinates on Manhattan's rotated
axes, not per-street positions. Each street is ordered perfectly by one of
them, never both:

    avenues (N-S)       along_dist   |rho| 1.000 on all 7
    cross streets (E-W) cross_dist   |rho| 1.000 on all 9

Read the wrong column and half the streets appear scrambled. The exception is
`Park_Ave_Tunnel_Segment`, which orders on *cross* (0.984) despite being on a
north-south avenue — independent confirmation of the chain running east-west.

`original_node_id` is `1st_avenue_001`, not `n00043`. There is no id crosswalk
to the pipeline; the join is spatial and exact at 0.00 m.

## superseded_nodes_2026-08-18.gpkg

DO NOT JOIN ON THIS. 601 nodes, and it carries `node_id` strings that collide
with the live frame while sitting a **median 356.7 m** away from where the live
frame puts the same id. It also has the deprecated `zone` column CLAUDE.md
says never to reintroduce. Kept as history only.
