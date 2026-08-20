"""Re-profile a handful of nodes from cached imagery, then diff the result.

Stage 3 is 35 minutes to 6 hours of segmentation across 579 nodes, and it
writes to azimuth_profiles.npz every 50 nodes. That makes the real stage
the worst possible place to test a change to it: the slowest loop in the
repo, pointed at the one file that cannot be regenerated cheaply.

This runs the same segmentation -- it imports s03_profiles.profile_node
rather than copying the loop -- over a stratified handful of nodes, writes
to a separate npz, and reports how far the result moved from the saved
profiles in the terms anyone actually reads: GVI and VEI under the five
view conditions of stage 4. A refactor that changes nothing prints zeros.
Anything else prints the size of what you changed, per node and per view,
before it costs you a full run.

Nothing here touches the network or Street View: it reads the JPEGs
already under data/raw/svi. The baseline npz is opened read-only, and an
--out that resolves to it is refused rather than merged into.

    python tools/s03_subset.py                  # n from config, profile + diff
    python tools/s03_subset.py --n 4
    python tools/s03_subset.py --nodes n00000,n00417
    python tools/s03_subset.py --diff-only data/processed/azimuth_profiles_subset.npz

--diff-only re-reads an existing subset file and re-runs the comparison
with no GPU and no torch, which is also how this runs on the laptop.
"""
import argparse, sys, time
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from common import (CFG, PROC, banner, DIRECTIONS, slice_metrics,  # noqa: E402
                    missing_images)

SUB = CFG["s03_subset"]
D = CFG["directional"]
BASE = PROC / "azimuth_profiles.npz"

# The same five views stage 4 reports. Diffing raw 360-bin arrays would
# flag bin-boundary jitter that no reported number is sensitive to; these
# are the quantities a change has to leave alone.
VIEWS = [(lab, b, D["fov"]) for lab, b in DIRECTIONS.items()] + \
        [("full360", 0.0, 360.0)]


def pick(mf, n, explicit):
    """Choose which nodes to re-profile.

    Stratified by GVI quartile, as tools/cubemap_check.py does, so a small
    subset still spans bare mid-block to Park Avenue median. A change that
    only shows up in heavy canopy is exactly the kind that a subset drawn
    off the top of a sorted node list would miss.
    """
    have = sorted(set(mf.node_id))
    if explicit:
        want = [s.strip() for s in explicit.split(",") if s.strip()]
        missing = [w for w in want if w not in set(have)]
        if missing:
            sys.exit(f"not in the manifest: {', '.join(missing)}")
        return want

    m = PROC / "metrics.csv"
    if not m.exists():
        return have[:: max(1, len(have) // n)][:n]
    g = pd.read_csv(m)
    g = g[g.node_id.isin(have) & g.GVI.notna()].copy()
    if g.empty:
        return have[:: max(1, len(have) // n)][:n]
    g["q"] = pd.qcut(g.GVI, 4, labels=False, duplicates="drop")
    sel = list(g.groupby("q", group_keys=False)
                .apply(lambda d: d.sample(min(len(d), max(1, n // 4)),
                                          random_state=CFG["seed"]))
                .node_id)
    # Park Avenue moves everything: its planted median sits at GVI 14-17
    # where every other street is under 5. A change to the vegetation
    # labels or the class shares shows up there first, and a four-node
    # quartile draw usually lands in the top quartile well below it. The
    # groups come out in quartile order, so the last pick is the one to
    # give up.
    top = g.loc[g.GVI.idxmax(), "node_id"]
    if top not in sel:
        sel[-1] = top
    return sorted(sel)


def profile(nodes, out):
    """Segment the chosen nodes through stage 3's own function."""
    if out.resolve() == BASE.resolve():
        sys.exit(f"--out is the real profile file ({BASE}). It is hours of "
                 f"segmentation and this tool will not write to it.")
    try:
        from s03_profiles import profile_node
        from common import device_and_batch, load_segmenter, resolve_classes
    except ModuleNotFoundError as e:
        sys.exit(f"{e.name} is missing: this is the analysis environment "
                 f"(requirements-analysis.txt), which has no segmenter. "
                 f"Segment on the GPU box, or use --diff-only here.")

    mf = pd.read_csv(PROC / "manifest.csv")
    by_node = {k: g for k, g in mf.groupby("node_id")}
    # profile_node drops a node whose image will not open, so a manifest
    # written on another machine produced an empty npz and a cheerful
    # "0 nodes profiled" rather than an error. Check up front.
    by_node_paths = [q for n in nodes for q in by_node[n].path]
    gone = missing_images(by_node_paths)
    if gone:
        sys.exit(f"{len(gone)} of {len(by_node_paths)} images are not on "
                 f"disk, e.g. {gone[0]}. manifest.csv stores absolute paths "
                 f"and this frame's were written on another machine; "
                 f"common.image_path falls back to paths.imagery by "
                 f"basename, so a miss here means the JPEG is genuinely "
                 f"absent. Re-run s02 to fetch it.")
    dev, _ = device_and_batch()
    proc, model = load_segmenter(dev)
    classes = resolve_classes(model.config.id2label)

    prof, t0, n_img = {}, time.time(), 0
    for nid in nodes:
        rows = by_node[nid]
        acc = profile_node(rows, proc, model, dev, classes)
        n_img += len(rows)
        if acc is None:
            print(f"  {nid}: an image failed to open, node dropped")
            continue
        prof[nid] = acc
        print(f"  {nid}: {len(rows)} images, {time.time() - t0:5.1f}s elapsed")
    np.savez_compressed(out, **prof)

    dt = time.time() - t0
    total = len(mf)
    print(f"\nwrote {out} ({len(prof)} nodes)")
    print(f"{n_img} images in {dt:.1f}s = {dt / max(n_img, 1):.2f}s/image "
          f"-> {dt / max(n_img, 1) * total / 60:.0f} min for all {total}")
    return prof


def diff(sub, tol):
    """Compare a subset profile against the saved one, per view."""
    if not BASE.exists():
        sys.exit(f"no baseline at {BASE} -- nothing to diff against")
    z = np.load(BASE)
    shared = [k for k in sub if k in z.files]
    if not shared:
        sys.exit("no nodes in common with the baseline")
    if len(shared) < len(sub):
        print(f"note: {len(sub) - len(shared)} subset nodes are absent from "
              f"the baseline and are not compared")

    rows = []
    for nid in shared:
        a, b = z[nid], sub[nid]
        for lab, ctr, fov in VIEWS:
            g0, v0 = slice_metrics(a, ctr, fov)
            g1, v1 = slice_metrics(b, ctr, fov)
            rows.append({"node_id": nid, "view": lab,
                         "GVI_base": g0, "GVI_new": g1, "dGVI": g1 - g0,
                         "VEI_base": v0, "VEI_new": v1, "dVEI": v1 - v0})
    d = pd.DataFrame(rows)

    banner(f"DIFF  {len(shared)} nodes x {len(VIEWS)} views")
    for c, unit in (("dGVI", "GVI points"), ("dVEI", "VEI")):
        ad = d[c].abs()
        print(f"  {c:5s}  max {ad.max():.4f}  mean {ad.mean():.4f}  "
              f"p95 {ad.quantile(0.95):.4f}  ({unit})")

    # The weight row is structural: it depends only on geometry, not on
    # what the segmenter saw. If it moves, the degree-to-column mapping or
    # the solid-angle weighting moved, which is a different and worse class
    # of change than a shifted class share.
    wmax = max(float(np.abs(z[nid][3] - sub[nid][3]).max()) for nid in shared)
    print(f"  weight row (row 3) max abs change: {wmax:.2e}")

    moved = d[(d.dGVI.abs() > tol) | (d.dVEI.abs() > tol)]
    if moved.empty:
        print(f"\nno view moved by more than {tol} -- unchanged")
    else:
        print(f"\n{len(moved)} of {len(d)} node-views moved by more "
              f"than {tol}:")
        print(moved.sort_values("dGVI", key=abs, ascending=False)
                   .head(20).to_string(index=False,
                                       float_format=lambda x: f"{x:8.3f}"))
    out = ROOT / CFG["paths"]["results"] / "tables" / "s03_subset_diff.csv"
    d.to_csv(out, index=False)
    print(f"\nwrote {out}")
    return moved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=SUB["n_nodes"],
                    help=f"nodes to profile (default {SUB['n_nodes']}, "
                         f"stratified by GVI quartile)")
    ap.add_argument("--nodes", help="explicit node_ids, comma separated")
    ap.add_argument("--out", default=str(PROC / SUB["out_name"]),
                    help="where to write the subset profiles")
    ap.add_argument("--diff-only", metavar="NPZ",
                    help="skip segmentation, diff this file (no GPU needed)")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if any view moved beyond the tolerance")
    a = ap.parse_args()

    banner("STAGE 3 SUBSET  cached imagery only, no requests")
    if a.diff_only:
        z = np.load(a.diff_only)
        sub = {k: z[k] for k in z.files}
        print(f"{a.diff_only}: {len(sub)} nodes")
    else:
        mf = pd.read_csv(PROC / "manifest.csv")
        nodes = pick(mf, a.n, a.nodes)
        print(f"{len(nodes)} nodes: {', '.join(nodes)}")
        sub = profile(nodes, Path(a.out))

    moved = diff(sub, SUB["report_tol"])
    if a.strict and moved is not None and not moved.empty:
        sys.exit(1)


if __name__ == "__main__":
    main()
