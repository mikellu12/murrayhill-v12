"""
Cubemap vs 4-image sampling: how much does the missing sky bias GVI and VEI?

Four images at fov=90, pitch=0 cover only the band +/-45 degrees from
horizontal. Everything above 45 degrees elevation -- overhead canopy, most
of the visible sky -- is never sampled. Adding pitch=+90 and pitch=-90
completes a cubemap: six faces tiling the sphere exactly once, no gaps and
no overlap.

That is a strictly better sampling scheme than Treepedia's 6 headings x 3
pitches, which double-counts the mid-elevations and has no closed-form
solid angle. Same API, 6 requests per node instead of 4.

THIS IS THE ONLY THING IN THE REPO THAT SPENDS MONEY
----------------------------------------------------
Every other stage after s02 reads saved data. This one issues fresh Street
View requests: 6 per sampled node. At the default 100-node subsample that
is 600 requests, about $4.20 at list price and comfortably inside the
10,000/month free tier. It is a stratified subsample by design -- run it
before deciding whether a full re-fetch of all ~600 nodes is worth it.

WHY IT MATTERS HERE
-------------------
The framework's first claim is that eye-level vegetation drives pedestrian
GVI more than overhead canopy. Pitch-0 imagery cannot separate the two, so
that claim is currently untestable with this pipeline. The zenith face is
what would separate them. Until this runs, GVI here is horizon-band GVI and
should be written as such, and the direction of the bias is known (GVI low,
VEI high) but the magnitude is not.

WHAT TO CONCLUDE
----------------
The question is not whether the level shifts -- it will. It is whether the
VALIDATED RELATIONSHIP changes. If cube VEI correlates better with
geometric H/W than the 4-image VEI does, the current VEI is degraded by the
missing sky and the re-fetch is justified. If it correlates the same or
worse, the 4-image scheme is adequate and the bias can be reported as a
known offset.

    python tools/cubemap_check.py                 # 100 nodes, stratified
    python tools/cubemap_check.py --n 40 --dry-run

Or from a notebook where the segmenter is already loaded:

    import cubemap_check as cc
    cmp = cc.run(metrics, meta, OUT, GMAPS_KEY,
                 seg_proc, seg_model, DEV, VEG_IDS, SKY_IDS, BLDG_IDS)
"""
import argparse, sys
from io import BytesIO
from pathlib import Path

import numpy as np, pandas as pd, requests
from PIL import Image
from tqdm.auto import tqdm
from scipy.stats import wilcoxon, spearmanr

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import CFG, PROC, RAW, RES, banner, require

CUBE = CFG.get("cubemap", {})
N_SUBSAMPLE = CUBE.get("n_subsample", 100)
SOLID_ANGLE_WEIGHTING = CFG["sampling"]["solid_angle_weighting"]
COST_PER_REQUEST = CUBE.get("cost_per_request", 0.007)

CUBE_VIEWS = [("F", 0, 0), ("R", 90, 0), ("B", 180, 0), ("L", 270, 0),
              ("U", 0, 90), ("D", 0, -90)]
EQUATOR_ONLY = {"F", "R", "B", "L"}

SV = "https://maps.googleapis.com/maps/api/streetview"


# --------------------------------------------------------------- weighting
def solid_angle_weights(size):
    """Per-pixel solid angle on one cube face.

    A cube face is a plane, so pixels near the corners subtend less solid
    angle than pixels at the centre -- by up to a factor of ~3.6. Summing
    raw pixel counts therefore over-weights the corners. The correct weight
    for a pixel at normalised face coordinates (x, y) in [-1, 1] is
    (1 + x^2 + y^2)^(-3/2).

    This is the 2-D form. common.column_weights() collapses the same
    quantity over rows because the profile stages only ever need a
    per-column weight; the zenith and nadir faces have real vertical
    structure, so the full map is needed here.

    Treepedia does not do this. It is a small correction (a few percent on
    GVI) but it is free and it makes the estimator genuinely unbiased.
    """
    c = (np.arange(size) + 0.5) / size * 2 - 1
    x, y = np.meshgrid(c, c)
    w = (1 + x**2 + y**2) ** -1.5
    return w / w.sum()          # normalise so each face sums to 1


def fetch(pano_id, heading, pitch, key, size=640):
    r = requests.get(SV, params={"pano": pano_id, "size": f"{size}x{size}",
                                 "heading": heading, "fov": 90,
                                 "pitch": pitch, "key": key}, timeout=30)
    if r.status_code != 200 or len(r.content) < 5000:
        return None
    return Image.open(BytesIO(r.content)).convert("RGB")


def classify(img, seg_proc, seg_model, dev, veg_ids, sky_ids, bldg_ids, W):
    """Weighted class fractions for one face."""
    import torch
    inp = seg_proc(images=[img], return_tensors="pt").to(dev)
    with torch.inference_mode():
        out = seg_model(**inp)
    # Same MPS contiguity issue as s03_profiles.
    out.class_queries_logits = out.class_queries_logits.cpu().contiguous()
    out.masks_queries_logits = out.masks_queries_logits.cpu().contiguous()
    m = seg_proc.post_process_semantic_segmentation(
        out, target_sizes=[img.size[::-1]])[0].numpy()

    if W is None:
        W = np.full(m.shape, 1.0 / m.size)
    elif W.shape != m.shape:
        # nearest-neighbour resize of the weight map
        yi = (np.arange(m.shape[0]) * W.shape[0] // m.shape[0])
        xi = (np.arange(m.shape[1]) * W.shape[1] // m.shape[1])
        W = W[np.ix_(yi, xi)]
        W = W / W.sum()

    return {"veg":  W[np.isin(m, veg_ids)].sum(),
            "sky":  W[np.isin(m, sky_ids)].sum(),
            "bldg": W[np.isin(m, bldg_ids)].sum()}


def subsample(metrics, n):
    """n nodes spanning the GVI range, not whichever look best."""
    m = metrics.dropna(subset=["GVI"]).copy()
    m["q"] = pd.qcut(m.GVI, 4, labels=False, duplicates="drop")
    return (m.groupby("q", group_keys=False)
             .apply(lambda g: g.sample(min(len(g), max(1, n // 4)),
                                       random_state=CFG["seed"])))


def run(metrics, meta, OUT, GMAPS_KEY, seg_proc, seg_model, DEV,
        VEG_IDS, SKY_IDS, BLDG_IDS, n=None, dry_run=False):
    n = n or N_SUBSAMPLE
    OUT = Path(OUT)
    sub = subsample(metrics, n)
    panos = meta.dropna(subset=["pano_id"]).set_index("node_id").pano_id.to_dict()
    sub = sub[sub.node_id.isin(panos)]
    cost = len(sub) * 6 * COST_PER_REQUEST
    print(f"subsample: {len(sub)} nodes, {len(sub)*6} images (${cost:.2f})")
    if dry_run:
        print("  --dry-run: stopping before any request is issued.")
        print(f"  GVI quartile spread of the subsample: "
              f"{sub.GVI.min():.2f} to {sub.GVI.max():.2f}")
        return None

    W = solid_angle_weights(CFG["sampling"]["image_size"]) \
        if SOLID_ANGLE_WEIGHTING else None

    rows = []
    for nid in tqdm(sub.node_id, desc="cubemap", mininterval=2.0):
        pid = panos.get(nid)
        if not pid:
            continue
        faces = {}
        for tag, h, p in CUBE_VIEWS:
            img = fetch(pid, h, p, GMAPS_KEY)
            if img is None:
                break
            faces[tag] = classify(img, seg_proc, seg_model, DEV,
                                  VEG_IDS, SKY_IDS, BLDG_IDS, W)
        if len(faces) != 6:
            continue

        def agg(tags):
            v = sum(faces[t]["veg"] for t in tags)
            s = sum(faces[t]["sky"] for t in tags)
            b = sum(faces[t]["bldg"] for t in tags)
            k = len(tags)
            return (100 * v / k, b / (s + b) if (s + b) > 0 else np.nan)

        gvi6, vei6 = agg([t for t, _, _ in CUBE_VIEWS])
        gvi4, vei4 = agg(sorted(EQUATOR_ONLY))
        gvi5, vei5 = agg([t for t in faces if t != "D"])

        rows.append({"node_id": nid,
                     "GVI_cube6": gvi6, "VEI_cube6": vei6,
                     "GVI_cube5": gvi5, "VEI_cube5": vei5,
                     "GVI_eq4": gvi4, "VEI_eq4": vei4,
                     "sky_up": faces["U"]["sky"], "veg_up": faces["U"]["veg"],
                     "bldg_up": faces["U"]["bldg"]})

    if not rows:
        print("no nodes completed all six faces -- check the key and quota")
        return None

    keep = [c for c in ["node_id", "GVI", "VEI", "typology", "HW_ratio",
                        "northing_m"] if c in metrics.columns]
    cmp = pd.DataFrame(rows).merge(metrics[keep], on="node_id", how="left")
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    cmp.to_csv(OUT / "tables" / "cubemap_comparison.csv", index=False)

    # ------------------------------------------------------------ report
    print(f"\n=== n={len(cmp)} ===\n")
    print("The zenith face alone, mean class shares:")
    print(f"  sky {cmp.sky_up.mean():.1%}   vegetation {cmp.veg_up.mean():.1%}"
          f"   building {cmp.bldg_up.mean():.1%}")
    print("  -> this is the view the 4-image scheme never sees.\n")

    print(f"{'metric':12s} {'4-img (current)':>16s} {'cube 6':>10s} "
          f"{'cube 5 (no ground)':>20s}")
    print("-" * 62)
    print(f"{'GVI mean':12s} {cmp.GVI_eq4.mean():16.2f} "
          f"{cmp.GVI_cube6.mean():10.2f} {cmp.GVI_cube5.mean():20.2f}")
    print(f"{'VEI mean':12s} {cmp.VEI_eq4.mean():16.3f} "
          f"{cmp.VEI_cube6.mean():10.3f} {cmp.VEI_cube5.mean():20.3f}")

    print("\npaired tests, current scheme vs full cubemap:")
    for a_, b_, lab in [("GVI_eq4", "GVI_cube6", "GVI"),
                        ("VEI_eq4", "VEI_cube6", "VEI")]:
        d = cmp[[a_, b_]].dropna()
        if len(d) < 10:
            continue
        stat, p = wilcoxon(d[a_], d[b_])
        bias = (d[b_] - d[a_]).mean()
        print(f"  {lab}: mean difference {bias:+.3f}  p={p:.2e}")
    print("  A significant shift is expected and is not the finding. The")
    print("  finding is whether the RELATIONSHIP below changes.")

    # The question that actually matters: does the correction change the
    # validated relationship, or just shift the level?
    if "HW_ratio" in cmp.columns and cmp.HW_ratio.notna().sum() > 20:
        print("\nrho vs geometric H/W (the validation that matters):")
        for col in ["VEI_eq4", "VEI_cube5", "VEI_cube6"]:
            d = cmp.dropna(subset=[col, "HW_ratio"])
            r, p = spearmanr(d[col], d.HW_ratio)
            print(f"  {col:12s} rho={r:+.3f}  p={p:.2e}  n={len(d)}")
        print("  If cube VEI correlates BETTER, the current VEI is degraded")
        print("  by the missing sky and the full re-fetch is justified.")
        print("  If it correlates the same or worse, the 4-image scheme is")
        print("  adequate and you can report the bias as a known offset.")

    print(f"\nwrote {OUT/'tables'/'cubemap_comparison.csv'}")
    return cmp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_SUBSAMPLE,
                    help=f"nodes to sample (default {N_SUBSAMPLE}, "
                         f"stratified by GVI quartile)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the cost and stop before any request")
    a = ap.parse_args()

    banner("cubemap vs 4-image sampling")
    import geopandas as gpd
    from common import device_and_batch, load_segmenter, resolve_classes

    mp = PROC / "metrics.gpkg"
    md = RAW / "metadata.csv"
    for p in (mp, md):
        if not p.exists():
            sys.exit(f"{p} not found -- run the pipeline first")
    metrics = gpd.read_file(mp)
    meta = pd.read_csv(md)

    if a.dry_run:
        return run(metrics, meta, RES, None, None, None, None,
                   None, None, None, n=a.n, dry_run=True)

    key = require("GMAPS_KEY")
    dev, _ = device_and_batch()
    proc, model = load_segmenter(dev)
    VEG, SKY, BLD = resolve_classes(model.config.id2label)
    return run(metrics, meta, RES, key, proc, model, dev,
               VEG, SKY, BLD, n=a.n)


if __name__ == "__main__":
    main()
