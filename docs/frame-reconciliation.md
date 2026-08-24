# Node frame reconciliation: `blockology-gvi` vs `murrayhill-v12`

Before any metrics move between the two repos, the node frames have to be
reconciled. They are currently different frames, and because `node_id` is
positional in both, merging results across them mis-attributes every row
without raising an error.

Measured 2026-08-20 against `jling888/blockology-gvi@main`
(`out/nodes/nodes.gpkg`, `out/metadata/metadata.csv`) and
`mikellu12/murrayhill-v12@main` (`data/processed/nodes.gpkg`).

## The problem in one line

Both frames number sequentially from `n00000`, but they are offset: my
`n00000` sits **8.3 m** from theirs. **Zero of 601 nodes have both matching
geometry (<1 m) and a matching ID.**

Copying GVI/VEI values keyed to one frame into the other attaches them to
the wrong street corner. Nothing errors; the run looks clean.

## Side by side

|                  | `blockology-gvi` | `murrayhill-v12` |
|------------------|------------------|------------------|
| nodes            | 697              | 601              |
| usable           | 680 (2026-04)    | 579 analysed     |
| streets          | 16               | 15               |
| 5th Avenue       | 33 nodes         | excluded         |
| typology classes | `avenue`, `mid_block` | `avenue_canyon`, `avenue_secondary`, `mid_block` |
| extra columns    | —                | `chain`, `chain_pos_m`, `northing_m`, `easting_m` |
| capture target   | 2026-04          | 2026-04 ✓        |
| CRS              | EPSG:32618       | EPSG:32618 ✓     |

Capture epoch and CRS agree, so the imagery is comparable in principle.
The sampling geometry is what diverges.

## How far apart

Distance from each of my 601 nodes to the nearest node in their frame,
against a 20 m sampling interval:

```
   0-1  m   210   ##########################################
   1-2  m     1
   2-4  m    79   ###############
   4-6  m   104   ####################
   6-8  m    44   ########
   8-10 m   121   ########################
  10-15 m    13   ##
  15+  m     29   #####
```

Median 4.5 m, p90 9.5 m. 35% within 1 m, 60% within 5 m.

The spread across 0–10 m — half the sampling interval — is the signature of
independent phase along each street chain, not a uniform shift. There is no
constant offset to correct for, and no rigid transform that aligns them.
Nearest-neighbour matching at any tolerance would silently pair ~40% of
nodes with a neighbour one position up or down the block.

## Where the differences come from

Explained and deliberate on the `murrayhill-v12` side:

- **5th Avenue (33 nodes).** Excluded via `study_area.corner_pairs`, which
  set Madison Avenue as the western edge. Fifth Avenue and the Empire State
  block fall in Midtown South / NoMad on most delimitations. Manhattan
  neighbourhood boundaries are not official — this is a citable choice, not
  a bug, but it is a choice the paper has to state either way. All 697 of
  their nodes fall inside my bounding box; the difference is the corner-pair
  polygon, not the bbox.
- **Three-class typology.** `avenue` is split into `avenue_canyon`
  (Madison, Park, Lexington) and `avenue_secondary` (1st, 2nd, 3rd),
  because H/W differs sharply between them and the split is load-bearing
  for the enclosure analysis.
- **Grid-axis coordinates.** `northing_m` / `easting_m` along bearing 029°
  replace an earlier latitude-based zone categorisation, which cut
  diagonally across a grid rotated 29° and split every street between two
  categories.
- **Phase.** My chains start at `chain_pos_m = 80`, theirs at 0.

Not yet explained, and worth checking:

- **1st Avenue: 21 nodes theirs, 44 mine.** Too large to be phase or
  typology. Suggests a different clip or a break in the OSM way. Flagging
  rather than guessing.
- Cross streets run consistently ~15% longer in their frame (e.g. E 36th:
  52 vs 36), consistent with a wider clip at the avenue ends.

## The decision

Which frame is canonical for the paper? The options are not symmetric:

**A. Their frame becomes canonical.** Stages 2–8 re-run on their 697 nodes.
Cost: 697 × 4 headings = 2,788 Street View requests ≈ **$19.52** at list
price, plus a full segmentation pass. Gains 5th Avenue and ~100 more usable
nodes. Every number currently in `murrayhill-v12/README.md` changes, and
the typology split and grid-axis work need porting forward to survive.

**B. My frame becomes canonical.** The stage-1 refinements move into
`blockology-gvi` alongside stages 3–8, and their repo regenerates this
frame end-to-end. No refetch; existing results stay valid. Costs the 5th
Avenue coverage.

**C. Neither — a third frame.** If a better frame is already in progress,
both A and B are wasted work, and the right move is to fix the *contract*
first (below) and re-run once, when that frame lands.

## Proposed contract, whichever frame wins

Given the frame is expected to change again, the cross-repo failure mode
gets worse rather than better. `murrayhill-v12` already aborts stages 2 and
4 on frame mismatch, but that check does not span repositories.

Suggestion: **a frame fingerprint.** Hash the rounded node coordinates,
write it into `nodes.gpkg` and into every downstream artifact, and have each
stage refuse to run when the fingerprint it reads disagrees with the frame
it was handed. Roughly fifteen lines, and it converts the silent
mis-attribution above into a hard failure.

The general rule it encodes: results keyed to a frame should not be
committed as data. Commit the code that regenerates them, take the frame as
an input, and re-run when the frame changes. Committed numbers from a
superseded frame are worse than absent ones — they still look valid.

## What has not happened

Nothing has been pushed to `jling888/blockology-gvi`, and no fork has been
created. `mikellu12` has pull access only, so any contribution would arrive
as a pull request from a fork.

Reproduce the numbers above from `murrayhill-v12` with the node frames from
both repos; the comparison uses geometry only and needs no imagery.
