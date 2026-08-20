"""Does the scaffolding detector agree with the permits? A balanced test.

The open-vocabulary detector has never been scored against anything. The
25-face sample panel suggested it does not track DOB permits, but 5
positives cannot settle that: it is equally consistent with a detector that
works and a sample that missed.

This runs the detector over a balanced subsample -- half the nodes drawn
from those with a permitted shed nearby, half from those without -- and
scores it properly. Balanced by design: at the frame's base rate of 32% a
detector that always says "no shed" scores 68% accuracy, and accuracy is
the wrong summary anyway. What is reported is the rank statistic, AUC,
which asks whether a node with a shed scores higher than one without more
often than chance. 0.5 is a coin flip.

TWO THINGS THIS FIXES ABOUT THE EARLIER COMPARISON
--------------------------------------------------
The detector is scored on the 180-degree forward view, not the raw frames,
so it is measured on the thing the panels draw and the metrics describe.

The permit label is restricted to the same cone. A shed two metres behind
the camera is inside a 30 m radius and outside the view, and counting it as
a positive would penalise the detector for not seeing through its own head.
Both labels are reported, because the difference between them is itself
worth knowing.

    python tools/scaffold_eval.py            # 60 nodes, balanced
    python tools/scaffold_eval.py --n 24     # quicker
"""
import argparse, sys, time
import numpy as np, pandas as pd, geopandas as gpd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from common import CFG, PROC, RAW, RES, banner, image_path, DIRECTIONS  # noqa: E402
from face_samples import scaffold_masks, panorama                       # noqa: E402

D = CFG["dob"]


def auc(scores, labels):
    """Rank AUC via the Mann-Whitney U identity. Ties get half credit."""
    labels = np.asarray(labels, dtype=bool)
    pos, neg = np.asarray(scores)[labels], np.asarray(scores)[~labels]
    if not len(pos) or not len(neg):
        return np.nan
    from scipy.stats import rankdata
    r = rankdata(np.r_[pos, neg])
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def cone_label(nodes, permits, bearing_by_node, radius, fov=180.0):
    """Is a live shed permit inside this node's forward cone?"""
    p = gpd.GeoDataFrame(
        permits,
        geometry=gpd.points_from_xy(permits.longitude.astype(float),
                                    permits.latitude.astype(float)),
        crs=4326).to_crs(nodes.crs)
    shed = p[p.work_type.eq("Sidewalk Shed")]
    px, py = shed.geometry.x.values, shed.geometry.y.values
    out = {}
    for _, n in nodes.iterrows():
        dx, dy = px - n.geometry.x, py - n.geometry.y
        d = np.hypot(dx, dy)
        near = d <= radius
        if not near.any():
            out[n.node_id] = (False, float(d.min()) if len(d) else np.nan)
            continue
        brg = (np.degrees(np.arctan2(dx[near], dy[near]))) % 360
        off = np.abs(((brg - bearing_by_node[n.node_id] + 180) % 360) - 180)
        out[n.node_id] = (bool((off <= fov / 2).any()), float(d[near].min()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60,
                    help="nodes to score, balanced across the permit label")
    a = ap.parse_args()

    banner("SCAFFOLD DETECTOR against DOB permits")
    dob = pd.read_csv(PROC / "dob_shed_by_node.csv")
    permits = pd.read_csv(RAW / "dob_permits.csv").dropna(
        subset=["latitude", "longitude"])
    nodes = gpd.read_file(PROC / "nodes.gpkg")
    metrics = pd.read_csv(PROC / "metrics.csv")[["node_id", "street_axis_deg"]]
    manifest = pd.read_csv(PROC / "manifest.csv")
    have = set(manifest.node_id)

    d = dob[dob.node_id.isin(have)].merge(metrics, on="node_id", how="left")
    rng = np.random.default_rng(CFG["seed"])
    half = max(4, a.n // 2)
    pos = d[d.dob_shed].sample(min(half, int(d.dob_shed.sum())),
                               random_state=CFG["seed"])
    neg = d[~d.dob_shed].sample(min(half, int((~d.dob_shed).sum())),
                                random_state=CFG["seed"])
    sel = pd.concat([pos, neg])
    print(f"  {len(pos)} nodes with a permitted shed within "
          f"{D['match_radius_m']} m, {len(neg)} without")

    bearing = {}
    for _, r in sel.iterrows():
        ax = r.street_axis_deg
        if pd.isna(ax):
            bearing[r.node_id] = 0.0
        else:
            k = min(DIRECTIONS, key=lambda v:
                    abs(((DIRECTIONS[v] - float(ax) + 180) % 360) - 180))
            bearing[r.node_id] = DIRECTIONS[k]

    cone = cone_label(nodes[nodes.node_id.isin(sel.node_id)], permits,
                      bearing, D["match_radius_m"])

    from PIL import Image
    rows, t0 = [], time.time()
    by_node = {k: g for k, g in manifest.groupby("node_id")}
    for i, (_, r) in enumerate(sel.iterrows(), start=1):
        images, frames = {}, {}
        for _, q in by_node[r.node_id].iterrows():
            images[float(q.heading)] = Image.open(
                image_path(q.path)).convert("RGB")
        masks = scaffold_masks(None, images)
        for h, im in images.items():
            frames[h] = (im, masks[h].astype(np.int32))
        _, lab, _ = panorama(frames, bearing[r.node_id])
        valid = lab >= 0
        share = float((lab[valid] == 1).mean())
        in_cone, nearest = cone[r.node_id]
        rows.append({"node_id": r.node_id, "osm_name": r.osm_name,
                     "bearing": bearing[r.node_id], "scaffold_share": share,
                     "dob_shed_radius": bool(r.dob_shed),
                     "dob_shed_cone": in_cone, "nearest_m": nearest})
        if i % 10 == 0:
            print(f"    {i}/{len(sel)}  {time.time() - t0:5.0f}s")

    t = pd.DataFrame(rows)
    t.to_csv(RES / "tables" / "scaffold_eval.csv", index=False)

    print(f"\n  {'label':22s} {'n pos':>6s} {'median + ':>10s} "
          f"{'median - ':>10s} {'AUC':>6s}")
    for lab in ["dob_shed_radius", "dob_shed_cone"]:
        y = t[lab].values.astype(bool)
        print(f"  {lab:22s} {y.sum():6d} "
              f"{100 * t.scaffold_share[y].median():9.2f}% "
              f"{100 * t.scaffold_share[~y].median():9.2f}% "
              f"{auc(t.scaffold_share.values, y):6.2f}")
    print("\n  AUC 0.5 is a coin flip. Below ~0.65 there is no threshold")
    print("  worth choosing, and no share worth reporting.")
    print(f"\nwrote {RES / 'tables' / 'scaffold_eval.csv'}"
          f"  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
