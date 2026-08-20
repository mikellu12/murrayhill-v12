# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

A research pipeline measuring Green View Index (GVI) and Visual Enclosure
Index (VEI) at 20 m intervals across Murray Hill, Manhattan, from Google
Street View imagery, validated against building-footprint geometry. The
output is a paper, so **the standard is what a reviewer would accept, not
what runs without error.**

Read `README.md` first. It carries the findings and the reasoning; this
file carries the rules.

## Run it

```bash
python preflight.py                # what's present, what's missing costs what
python migrate_gridaxis.py --apply # only when upgrading a v11 run
python main.py --from s04          # seconds, CPU only, no key, no network
python make_dashboard.py           # results/dashboard.html
```

Stages 1–3 need a GPU and `GMAPS_KEY` in `.env`. Stages 4–8 do not.

Two environments on this machine, and the split is deliberate. `.venv` is
analysis-only — `requirements-analysis.txt` says "no GPU, no torch, no
transformers" and it should stay that way, because it is the mirror of what
runs on a laptop with only `data/processed/` copied across. `.venv-gpu` has
torch + transformers and runs the segmentation tools on Apple MPS. There is
no `python` on the PATH; call the interpreter directly.

```bash
.venv/bin/python main.py --from s04          # analysis, seconds
.venv-gpu/bin/python tools/face_samples.py   # anything that segments
```

`transformers` pulls Mask2Former's loss module at load time, which imports
`scipy` even for inference — it is not optional in the GPU env.
`python main.py --only s07 s08` re-runs the enclosure analysis and figures.

Never run the full `python main.py` to test a change to stages 4–8. It
re-checks 2,300 images and can trigger a re-fetch.

To test a change to stage 3, do not re-run stage 3. `python
tools/s03_subset.py` segments a dozen stratified nodes from the JPEGs
already on disk, writes to its own npz, and prints the GVI/VEI movement
against the saved profiles.

## Invariants — breaking these silently corrupts results

**Node IDs are positional.** Stage 1 assigns `node_id` sequentially from
`n00000` against the current geometry. Rebuild the frame with different
settings and the same ID points at a different street corner, so cached
imagery and profiles pair with the wrong location. s02 and s04 abort rather
than allow it. If you touch anything under `study_area:` or `sampling:` in
`config.yaml`, `data/processed/` and `data/raw/svi/` must be deleted.
Changing `directional:`, `enclosure:` or `spatial:` is safe.

**Image paths in `manifest.csv` are absolute and often stale.** This frame
inherited a v11 manifest addressing `~/Downloads/murrayhill_v11`, so all
2,316 paths were dead while the JPEGs sat in `data/raw/svi`. Open imagery
through `common.image_path()`, which prefers the stored path and falls back
to `paths.imagery` by basename. Never call `Image.open(row.path)` directly:
`s03.profile_node` drops a node whose image will not open, so a stale
manifest produces an empty run and a cheerful "0 nodes profiled".

**`azimuth_profiles.npz` is expensive and irreplaceable.** It is 35 minutes
to 6 hours of segmentation. Never delete it, never write to it without a
backup, and never suggest re-running s03 as a debugging step.

**`SVF_band` is not SVF.** The imagery covers ±45° of elevation, so the
zenith — where most of the sky is — is never sampled. The name says `band`
everywhere on purpose. Do not rename it, do not compare it to published SVF
values, and do not let it be described as a sky view factor in any output.

**Street View imagery is not redistributable.** Google's terms cap caching
at 30 days. Derived profiles and metrics are fine to keep and publish; the
JPEGs are scratch. Do not add code that commits, uploads or bundles them.

**Open-vocabulary grounding is validated per class, never assumed.**
`tools/openvocab_eval.py` scores a detector against a geocoded city
register on the 180-degree forward view, balanced, by AUC. Measured so far:
street trees 0.83 closed-set against 0.78 open-vocab, benches 0.66/0.60,
bus shelters 0.56/0.65, sidewalk sheds 0.51 open-vocab. Closed-set wins
where ADE20K has the class; open vocabulary wins only where it does not,
and only to 0.65. `street_tree` is the positive control — if a change makes
that score fall, the harness broke, not the detector. Do not wire a new
class into any index before it has a row in that table.

**The scaffolding detector does not work, and this is measured.** CLIPSeg
with contrastive prompts scores AUC 0.55 against DOB sidewalk-shed permits
on a balanced 60-node sample, and 0.51 once the label is restricted to
permits inside the forward cone. That is a coin flip. Do not report a
scaffolding share from it, and do not tune the threshold expecting to
rescue it — there is no signal to threshold. `tools/scaffold_eval.py` is
the benchmark any replacement must beat before it is wired into anything.

**Do not reintroduce `zone`.** The three named zones were a latitude cut
across a grid rotated 29°, splitting all fifteen streets between two
categories. They are replaced by `northing_m` (`gridaxis.py`). If a
`zone` column appears in a dataframe, it came from a stale cached file and
should be dropped, not used.

## Statistical rules specific to this project

**Nodes are not independent.** Moran's I is ~0.62–0.66; nodes 20 m apart on
a block face photograph nearly the same scene. Every inferential number
must either cluster standard errors by `face_id` or aggregate to block
faces. There are 22–25 faces and 15 streets — clustering by street is too
few groups and was a bug in v11. Naive node-level p-values may be printed
only to show the size of the gap, never reported as results.

**Read R², not p.** At n=579 a slope is significant while explaining
nothing. Several existing results have p < 0.001 and R² < 0.05.

**Report rank correlation alongside OLS slopes.** OLS slope is cov/var, so
a handful of extreme-x points move it several-fold while Spearman rho
barely shifts. That divergence is the signature of leverage, and it is
exactly what produced v11's spurious "west-only" directional finding.

**The four travel directions are two partitions, not four samples.** N and
S are disjoint halves that tile the circle; so are E and W. Which views are
*along-street* flips with typology. Never present five view rows as five
independent findings.

**Park Avenue moves everything.** Its planted median sits at GVI 14–17
where every other street is under 5. Any fit that could be driven by it
must be reported with and without it. This is not licence to drop it — the
median is real streetscape — it is a requirement to say which fit the text
quotes.

**Do not fit a functional form because the previous one failed.** A low
linear R² carries no information about curvature, and the quadratic nests
the linear so its R² can only rise. Stage 7's inverted-U test is legitimate
only because the band edges come from the framework document and were fixed
before any GVI was inspected. If a new shape is proposed, it needs a prior
reason, not a failed fit.

## Working style here

**Verify against the real data before claiming a fix works.** The repo has
a history of plausible-sounding diagnoses that the data contradicted — the
"coverage gaps" that turned out to be date-filter exclusions with perfect
imagery, the "west-only effect" that was leverage. Run the numbers.

**Negative results are results.** GVI declines monotonically in H/W here —
no peak at human scale — and the README says so in the second section. Do
not soften that, and do not go looking for a specification that rescues it.
State it as what it is: the failure of a proxy we chose. Section 4 of the
framework document claims enclosure changes how greenery is *experienced*
(diluted, framed, offsetting oppression), never that greenery is more
abundant at human scale, and its outcome is sense of place, which nothing
here measures. Do not write that the enclosure envelope was falsified.

**Distinguish out-of-scope from falsified.** The under-enclosed band holds
8 nodes on 7 faces; Murray Hill has essentially no under-enclosed streets.
That prediction is untestable here, not disproved. `min_band_n` in
`config.yaml` triggers this warning automatically.

**Comments explain why, not what.** The existing code comments record
decisions and the traps behind them — why matching is exact rather than
substring, why the degree-to-column mapping is gnomonic, why the weight row
exists in the profile array. Match that. Do not add comments that restate
the line below them.

**Nothing hardcoded.** Every parameter lives in `config.yaml`. If a number
appears in a stage, it should have come from there.

## Layout

```
config.yaml              every parameter
main.py                  stage runner
preflight.py             input check before running
gridaxis.py              grid-axis position; replaces the zone categories
migrate_gridaxis.py      upgrade a v11 run in place, no refetch
make_dashboard.py        self-contained HTML report
run_analysis.py          laptop entry point, no GPU

src/common.py            config, paths, typology, image geometry
src/s01_frame.py         network -> 20 m frame -> typology, grid axis
src/s02_imagery.py       metadata probe -> Street View retrieval
src/s03_profiles.py      segmentation -> 360-bin azimuthal profiles   [GPU]
src/s04_metrics.py       GVI/VEI x 5 views, SVF_band, audit, faces, Moran
src/s05_geometry.py      footprints, measured facade width, H/W
src/s06_analysis.py      regressions, typology, gradient, confound check, per street
src/s07_enclosure.py     GVI vs H/W: bands, inverted-U, envelope by direction
src/s08_figures.py       maps, scatters, directional panels, roses

tools/s03_subset.py      re-profile a few nodes from cached JPEGs, diff  [GPU]
tools/eyelevel.py        vegetation by elevation band, eye-level vs canopy [GPU]
tools/face_samples.py    180 deg stitched sample + overlay per block face [GPU]
tools/dob_sheds.py       DOB permits as scaffolding ground truth  [network]
tools/scaffold_eval.py   scores a scaffolding detector against permits  [GPU]
tools/openvocab_eval.py  per-class AUC vs city registers, both detectors [GPU]
tools/pedestrian.py      pedestrian-realm segmentation, per street   [GPU]
tools/fov_check.py       is 180 deg the right forward view?
tools/cubemap_check.py   measures the vertical-coverage bias  [GPU, $]
tools/sidewalk.py        sidewalk width and setback vs directional GVI
tools/factor_check.py    factor analysis of the directional metrics
```

`tools/` is optional analysis, run by hand, not part of `main.py`.

**`tools/cubemap_check.py` is the only thing here that spends money.** It
issues 6 fresh Street View requests per sampled node — ~600 requests at the
default subsample. Never invoke it as a casual step, never widen `--n`
without saying what it will cost, and use `--dry-run` to check the estimate
first. Everything else after s02 reads saved data.

## Known open items

- The vertical bias (±45° elevation only) is **not yet measured**.
  `tools/cubemap_check.py` measures it but has not been run. Until it has,
  "eye-level GVI" — the framework's first claim — is not a quantity this
  pipeline can produce, and the magnitude of the bias is unknown.
- No outcome variable exists. The framework's dependent variable is sense
  of place; GVI and H/W are both inputs. The GVI × H/W interaction it
  proposes needs survey or dwell-time data and cannot be fitted here.
- 22 frame nodes are excluded by the capture-date filter, in two
  contiguous runs (E 37th, Park Ave). They have usable imagery from other
  dates. Relaxing the filter would mix leaf-on and leaf-off seasons into
  the variable being measured, so it is the wrong fix.
