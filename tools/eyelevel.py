"""Eye-level greenery against overhead canopy: the framework's first claim.

Section 1 of the framework document says low-tier foliage between 0.5 m and
2.5 m drives the pedestrian GVI far more than canopy above 4 m, because
sight concentrates in a 30-60 degree vertical cone. The pipeline's GVI
cannot speak to that: stage 3 averages each image down its columns, so the
vertical axis is gone by the time anything is saved. This measures it.

WHY A HEIGHT BAND IS RECOVERABLE WITHOUT DEPTH
----------------------------------------------
A point at height h and horizontal distance d appears at elevation angle
atan((h - camera) / d). The distance sets how far from the horizon it
lands, but only the SIGN decides which side: everything standing below the
camera appears below the horizon, at any distance, and everything taller
appears above it. Street View's mast is about 2.5 m, which is the
framework's own eye-level ceiling.

So the split below/above the horizon row is the split below/above roughly
2.5 m, exactly, with no depth estimate anywhere. What it is NOT is a
0.5-2.5 m band: the 0.5 m floor cannot be recovered this way, and the
columns are reported in degrees of elevation for the same reason SVF_band
carries the word band. Do not describe the lower band as "0.5-2.5 m".

Two further limits, both real. The camera height varies by vehicle by
maybe 10 cm, which moves the boundary a little at close range and not at
all far away. And overhanging things that are not attached to the ground --
awnings, balconies, hanging baskets -- sit above the horizon while being
"eye level" in any pedestrian sense.

    python tools/eyelevel.py --selftest     # geometry only, no GPU, no model
    python tools/eyelevel.py                # subset from config, then diff
    python tools/eyelevel.py --n 4
    python tools/eyelevel.py --all          # the whole frame; hours on GPU

Reads the JPEGs already in data/raw/svi. No network, no Street View
requests, and it never touches azimuth_profiles.npz.
"""
import argparse, sys, time
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from common import CFG, PROC, RES, banner, image_path, missing_images  # noqa: E402

EL = CFG["eyelevel"]
EDGES = EL["bands_deg"]
LABELS = EL["band_labels"]
FOV = CFG["sampling"]["fov"]


def row_elevations(height, fov=FOV):
    """Elevation angle of each pixel row, in degrees, positive upwards.

    The vertical twin of common.column_bearings, and gnomonic for the same
    reason: a perspective image is not linear in angle. Reading rows as a
    linear fraction of the field of view puts the horizon in the right
    place (it is the centre either way) but misplaces every other boundary,
    by ~4 degrees at the 15 degree edge of a fov-90 image.
    """
    y = (np.arange(height) + 0.5) / height * 2 - 1
    return -np.degrees(np.arctan(y * np.tan(np.radians(fov / 2))))


def cell_weights(height, width, fov=FOV):
    """Solid angle of every pixel, normalised to sum to 1.

    common.column_weights already sums this over rows; the vertical split
    needs it un-summed, because a band's weight is exactly the solid angle
    it occupies. Corner pixels subtend about a fifth of what centre pixels
    do, so counting rows uniformly would over-credit the band edges.
    """
    if not CFG["sampling"]["solid_angle_weighting"]:
        return np.full((height, width), 1.0 / (height * width))
    x = (np.arange(width) + 0.5) / width * 2 - 1
    y = (np.arange(height) + 0.5) / height * 2 - 1
    t = np.tan(np.radians(fov / 2))
    X, Y = np.meshgrid(x * t, y * t)
    w = (1 + X ** 2 + Y ** 2) ** -1.5
    return w / w.sum()


def band_masks(height, fov=FOV):
    """Row masks for each elevation band, and the band each row falls in."""
    el = row_elevations(height, fov)
    return [(el >= lo) & (el < hi) for lo, hi in zip(EDGES[:-1], EDGES[1:])]


def image_shares(seg, veg_ids, fov=FOV):
    """Vegetation share of solid angle within each elevation band.

    Returns (shares, weights): the vegetation fraction of each band and the
    band's share of the image's total solid angle. The weights are what
    lets several images be combined without a band that happens to be
    steeply foreshortened counting as much as one that is not.
    """
    H, W = seg.shape
    cw = cell_weights(H, W, fov)
    isveg = np.isin(seg, veg_ids)
    shares, weights = [], []
    for m in band_masks(H, fov):
        w = cw[m].sum()
        shares.append(float((cw[m] * isveg[m]).sum() / w) if w > 0 else np.nan)
        weights.append(float(w))
    return np.array(shares), np.array(weights)


def selftest():
    """Verify the geometry against cases with a known answer, without torch."""
    banner("SELFTEST  geometry only")
    H = W = CFG["sampling"]["image_size"]
    el = row_elevations(H)
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")

    check("top row looks up", el[0] > 0, f"{el[0]:+.2f} deg")
    check("bottom row looks down", el[-1] < 0, f"{el[-1]:+.2f} deg")
    check("horizon at the centre", abs(el[H // 2]) < 0.2, f"{el[H//2]:+.3f} deg")
    check("range is +/- fov/2", abs(abs(el[0]) - FOV / 2) < .2,
          f"{el[0]:+.2f} vs {FOV/2:+.1f}")
    # Gnomonic, not linear: the 15 degree row must NOT sit at 15/45 of the way.
    r15 = int(np.argmin(np.abs(el - 15)))
    lin = H / 2 - H / 2 * (15 / (FOV / 2))
    check("row spacing is gnomonic", abs(r15 - lin) > 5,
          f"row {r15}, linear reading would say {lin:.0f}")

    cw = cell_weights(H, W)
    check("weights sum to 1", abs(cw.sum() - 1) < 1e-9)
    check("centre pixel outweighs corner", cw[H // 2, W // 2] > 4 * cw[0, 0],
          f"{cw[H//2, W//2] / cw[0, 0]:.1f}x")

    masks = band_masks(H)
    check("bands tile the image", sum(m.sum() for m in masks) == H,
          f"{sum(m.sum() for m in masks)} of {H} rows")

    # A synthetic scene: vegetation exactly on the rows below the horizon.
    veg_id, other_id = 4, 1
    seg = np.full((H, W), other_id)
    seg[masks[0], :] = veg_id
    sh, wt = image_shares(seg, [veg_id])
    check("below-horizon band reads 100%", abs(sh[0] - 1) < 1e-9, f"{sh[0]:.4f}")
    check("upper bands read 0%", np.allclose(sh[1:], 0), f"{sh[1:]}")
    check("band weights sum to 1", abs(wt.sum() - 1) < 1e-9)
    check("below-horizon holds half the solid angle", abs(wt[0] - .5) < .01,
          f"{wt[0]:.3f}")

    print(f"\n  {'all checks passed' if ok else 'FAILURES ABOVE'}")
    return 0 if ok else 1


def run(nodes, out):
    """Segment the chosen nodes and split vegetation by elevation band."""
    try:
        import torch
        from common import device_and_batch, load_segmenter, resolve_classes
        from PIL import Image
    except ModuleNotFoundError as e:
        sys.exit(f"{e.name} is missing: this is the analysis environment, "
                 f"which has no segmenter. Run --selftest here and do the "
                 f"segmentation on the GPU box.")

    mf = pd.read_csv(PROC / "manifest.csv")
    by_node = {k: g for k, g in mf.groupby("node_id")}
    # Fail here rather than per image: a manifest written on another
    # machine used to leave every open() failing one node at a time.
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
    veg_ids, _, _ = resolve_classes(model.config.id2label)

    rows, t0 = [], time.time()
    for j, nid in enumerate(nodes, 1):
        acc_s = np.zeros(len(LABELS))
        acc_w = np.zeros(len(LABELS))
        for _, r in by_node[nid].iterrows():
            img = Image.open(image_path(r.path)).convert("RGB")
            inp = proc(images=[img], return_tensors="pt").to(dev)
            with torch.inference_mode():
                out_ = model(**inp)
            out_.class_queries_logits = out_.class_queries_logits.cpu().contiguous()
            out_.masks_queries_logits = out_.masks_queries_logits.cpu().contiguous()
            seg = proc.post_process_semantic_segmentation(
                out_, target_sizes=[img.size[::-1]])[0].numpy()
            sh, wt = image_shares(seg, veg_ids)
            # Weight each image's band by that band's solid angle, so the
            # four headings combine the way stage 3 combines them.
            acc_s += np.nan_to_num(sh) * wt
            acc_w += wt
        rec = {"node_id": nid}
        for lab, s, w in zip(LABELS, acc_s, acc_w):
            rec[f"GVI_{lab}"] = 100 * s / w if w > 0 else np.nan
        rows.append(rec)
        print(f"  [{j}/{len(nodes)}] {nid}  " + "  ".join(
            f"{lab}={rec[f'GVI_{lab}']:.2f}%" for lab in LABELS)
            + f"   {time.time() - t0:5.1f}s")

    d = pd.DataFrame(rows)
    d.to_csv(out, index=False)
    print(f"\nwrote {out} ({len(d)} nodes, {time.time() - t0:.0f}s)")
    return d


def report(d):
    """What the framework claims, against what the bands say."""
    m = pd.read_csv(PROC / "metrics.csv")[["node_id", "GVI", "osm_name",
                                           "typology"]]
    d = d.merge(m, on="node_id", how="left")
    eye, upper = f"GVI_{LABELS[0]}", [f"GVI_{l}" for l in LABELS[1:]]
    d["GVI_above"] = d[upper].mean(axis=1)

    banner(f"EYE-LEVEL vs OVERHEAD  ({len(d)} nodes)")
    print(d[["node_id", "osm_name", "GVI"] + [f"GVI_{l}" for l in LABELS]]
          .round(2).to_string(index=False))
    print(f"\n  median GVI below camera height : {d[eye].median():.2f}%")
    print(f"  median GVI above camera height : {d.GVI_above.median():.2f}%")
    print(f"  median full-image GVI (stage 4): {d.GVI.median():.2f}%")

    share = d[eye].sum() / (d[eye].sum() + d.GVI_above.sum())
    print(f"\n  {share:.0%} of the vegetation a pedestrian sees in this "
          f"subset sits below camera height.")
    print("  The framework predicts eye-level foliage dominates. This is the")
    print("  quantity that claim is about; the pipeline's GVI is not.")
    if len(d) < 30:
        print(f"\n  !! {len(d)} nodes is a prototype run, not a result. It")
        print("     shows the measurement works, not what the study area is")
        print("     like. Run --all before quoting any of these numbers.")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=EL["n_nodes"],
                    help=f"nodes to segment (default {EL['n_nodes']}, "
                         f"stratified by GVI quartile)")
    ap.add_argument("--nodes", help="explicit node_ids, comma separated")
    ap.add_argument("--all", action="store_true",
                    help="every node with imagery -- hours on a GPU")
    ap.add_argument("--out", default=str(PROC / EL["out_name"]))
    ap.add_argument("--selftest", action="store_true",
                    help="check the geometry without a model or a GPU")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(selftest())

    from s03_subset import pick
    mf = pd.read_csv(PROC / "manifest.csv")
    nodes = (sorted(set(mf.node_id)) if a.all
             else pick(mf, a.n, a.nodes))
    banner(f"EYE-LEVEL BAND  {len(nodes)} nodes, cached imagery only")
    d = run(nodes, Path(a.out))
    d = report(d)
    (RES / "tables").mkdir(parents=True, exist_ok=True)
    d.to_csv(RES / "tables" / "eyelevel_bands.csv", index=False)
    print(f"wrote {RES / 'tables' / 'eyelevel_bands.csv'}")


if __name__ == "__main__":
    main()
