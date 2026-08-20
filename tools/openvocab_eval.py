"""Per-class validation: does open-vocabulary grounding actually see it?

The morphology study argues that closed-set segmentation must give way to a
VLM, because ADE20K has no class for arcades, planters, bollards or hedges.
The premise is right -- measured on the sample panels, three of the four
Street Interface Matrix terms come out near zero for exactly that reason.
The conclusion does not follow automatically, and the one class already
tested says so: contrastive CLIPSeg scored AUC 0.51 against DOB sidewalk
shed permits, which is a coin flip.

So each class gets its own test, against a geocoded city register:

    street_tree   2015 Street Tree Census        POSITIVE CONTROL
    bench         DOT City Bench locations       the SAI term
    bus_shelter   DOT Bus Stop Shelters          a TEF proxy

street_tree is the control that makes the rest interpretable. ADE20K
certainly has a tree class, so if the harness cannot score that one, the
harness is wrong rather than the detector. And because both detectors run
on the same nodes against the same labels, the comparison that matters --
closed-set against open-vocabulary, per class -- is a difference of two
numbers rather than an argument.

WHAT IS SCORED
--------------
Both detectors are measured on the 180-degree forward view, and a label
counts only if the object falls inside that cone: a bench two metres behind
the camera is within the radius and outside the view, and scoring it as a
miss would penalise the detector for not seeing through its own head.

AUC, not accuracy. Sampling is balanced, so accuracy has a 50% floor and
tells you nothing; AUC asks whether a node with the object scores above one
without, more often than chance. 0.5 is a coin flip.

    python tools/openvocab_eval.py                 # every configured class
    python tools/openvocab_eval.py --class bench
    python tools/openvocab_eval.py --n 12          # quicker, noisier
"""
import argparse, sys, time
import numpy as np, pandas as pd, geopandas as gpd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from common import (CFG, PROC, RAW, RES, banner, image_path,      # noqa: E402
                    DIRECTIONS)
from face_samples import clipseg_masks, panorama                  # noqa: E402
from scaffold_eval import auc                                     # noqa: E402

E = CFG["openvocab_eval"]
BB = CFG["study_area"]["bbox"]


def truth_points(spec, name):
    """Fetch a city register once, cache it, return it as points."""
    import requests
    cache = RAW / f"truth_{name}.csv"
    if cache.exists():
        d = pd.read_csv(cache)
    else:
        where = (f"latitude between {BB['south']} and {BB['north']} AND "
                 f"longitude between {BB['west']} and {BB['east']}")
        if spec.get("where"):
            where += f" AND {spec['where']}"
        rows, offset = [], 0
        while True:
            r = requests.get(
                f"https://data.cityofnewyork.us/resource/{spec['dataset']}.json",
                params={"$where": where, "$limit": 50000, "$offset": offset},
                headers={"User-Agent": "murrayhill-gvi-research/1.0"}, timeout=120)
            r.raise_for_status()
            page = r.json()
            rows += page
            if len(page) < 50000:
                break
            offset += 50000
        d = pd.DataFrame(rows)
        d.to_csv(cache, index=False)
    d = d.dropna(subset=["latitude", "longitude"])
    print(f"  {name}: {len(d)} objects in the study bbox")
    return d


def in_cone(nodes, pts, bearing, radius, fov=180.0):
    """Count register points inside each node's forward cone."""
    p = gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(
        pts.longitude.astype(float), pts.latitude.astype(float)),
        crs=4326).to_crs(nodes.crs)
    px, py = p.geometry.x.values, p.geometry.y.values
    out = {}
    for _, n in nodes.iterrows():
        dx, dy = px - n.geometry.x, py - n.geometry.y
        d = np.hypot(dx, dy)
        near = d <= radius
        if not near.any():
            out[n.node_id] = 0
            continue
        brg = np.degrees(np.arctan2(dx[near], dy[near])) % 360
        off = np.abs(((brg - bearing[n.node_id] + 180) % 360) - 180)
        out[n.node_id] = int((off <= fov / 2).sum())
    return out


def bearings(nodes, metrics):
    """The along-street travel bearing for each node."""
    ax = metrics.set_index("node_id").street_axis_deg
    out = {}
    for nid in nodes.node_id:
        a = ax.get(nid, np.nan)
        if pd.isna(a):
            out[nid] = 0.0
        else:
            k = min(DIRECTIONS, key=lambda v:
                    abs(((DIRECTIONS[v] - float(a) + 180) % 360) - 180))
            out[nid] = DIRECTIONS[k]
    return out


def evaluate(name, spec, nodes, metrics, manifest, n_nodes, cache):
    from PIL import Image
    banner(f"CLASS  {name}")
    pts = truth_points(spec["truth"], name)
    brg = bearings(nodes, metrics)
    counts = in_cone(nodes, pts, brg, E["radius_m"])
    have = set(manifest.node_id)
    d = pd.DataFrame({"node_id": list(counts), "n_truth": list(counts.values())})
    d = d[d.node_id.isin(have)]

    pos_pool, neg_pool = d[d.n_truth > 0], d[d.n_truth == 0]
    rule = "in cone" 
    if len(pos_pool) < 6 or len(neg_pool) < 6:
        # Too common or too rare to split on presence: fall back to the
        # tails of the count, and say so rather than reporting a balanced
        # test that was not balanced.
        q = d.n_truth.quantile([1/3, 2/3]).values
        pos_pool, neg_pool = d[d.n_truth > q[1]], d[d.n_truth <= q[0]]
        rule = f"tertiles of count (>{q[1]:.0f} vs <={q[0]:.0f})"
    half = max(5, n_nodes // 2)
    sel = pd.concat([pos_pool.sample(min(half, len(pos_pool)),
                                     random_state=CFG["seed"]),
                     neg_pool.sample(min(half, len(neg_pool)),
                                     random_state=CFG["seed"])])
    print(f"  label rule: {rule} within {E['radius_m']} m; "
          f"{(sel.n_truth > 0).sum()} positive, {(sel.n_truth == 0).sum()} negative")

    by_node = {k: g for k, g in manifest.groupby("node_id")}
    rows, t0 = [], time.time()
    for i, (_, r) in enumerate(sel.iterrows(), start=1):
        images = {float(q.heading): Image.open(image_path(q.path)).convert("RGB")
                  for _, q in by_node[r.node_id].iterrows()}
        # Closed set, cached: the segmentation does not depend on the class.
        if r.node_id not in cache:
            cache[r.node_id] = segment(images)
        seg = cache[r.node_id]
        ade_ids = [i_ for i_, lab in cache["_labels"].items()
                   if lab.strip().lower() in [a.lower() for a in spec["ade"]]]
        frames = {h: (images[h], np.isin(seg[h], ade_ids).astype(np.int32))
                  for h in images}
        _, closed, _ = panorama(frames, brg[r.node_id])
        masks = clipseg_masks(images, spec["prompts"], spec.get("negatives", []),
                              CFG["open_vocabulary"].get("scaffold_contrast"))
        frames = {h: (images[h], masks[h].astype(np.int32)) for h in images}
        _, openv, _ = panorama(frames, brg[r.node_id])
        rows.append({"class": name, "node_id": r.node_id,
                     "n_truth": int(r.n_truth), "label": r.n_truth > 0,
                     "closed_set": float((closed[closed >= 0] == 1).mean()),
                     "open_vocab": float((openv[openv >= 0] == 1).mean())})
        if i % 10 == 0:
            print(f"    {i}/{len(sel)}  {time.time() - t0:5.0f}s")
    t = pd.DataFrame(rows)
    y = t.label.values
    print(f"\n  {'detector':12s} {'median +':>9s} {'median -':>9s} {'AUC':>6s}")
    for col, lab in [("closed_set", "Mask2Former"), ("open_vocab", "CLIPSeg")]:
        print(f"  {lab:12s} {100 * t[col][y].median():8.2f}% "
              f"{100 * t[col][~y].median():8.2f}% "
              f"{auc(t[col].values, y):6.2f}")
    return t


_SEG = {}


def segment(images):
    """Mask2Former label maps for one node's frames."""
    import torch
    from common import device_and_batch, load_segmenter
    if "model" not in _SEG:
        dev, _ = device_and_batch()
        proc, model = load_segmenter(dev)
        _SEG.update(dev=dev, proc=proc, model=model)
    dev, proc, model = _SEG["dev"], _SEG["proc"], _SEG["model"]
    out = {}
    for h, img in images.items():
        inp = proc(images=[img], return_tensors="pt").to(dev)
        with torch.inference_mode():
            o = model(**inp)
        o.class_queries_logits = o.class_queries_logits.cpu().contiguous()
        o.masks_queries_logits = o.masks_queries_logits.cpu().contiguous()
        out[h] = proc.post_process_semantic_segmentation(
            o, target_sizes=[img.size[::-1]])[0].numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--class", dest="only", help="evaluate one class")
    ap.add_argument("--n", type=int, default=E["n_nodes"])
    a = ap.parse_args()

    nodes = gpd.read_file(PROC / "nodes.gpkg")
    metrics = pd.read_csv(PROC / "metrics.csv")[["node_id", "street_axis_deg"]]
    manifest = pd.read_csv(PROC / "manifest.csv")
    classes = {k: v for k, v in E["classes"].items()
               if not a.only or k == a.only}

    cache = {}
    from common import load_segmenter, device_and_batch
    dev, _ = device_and_batch()
    proc, model = load_segmenter(dev)
    _SEG.update(dev=dev, proc=proc, model=model)
    cache["_labels"] = model.config.id2label

    out = [evaluate(k, v, nodes, metrics, manifest, a.n, cache)
           for k, v in classes.items()]
    t = pd.concat(out, ignore_index=True)
    t.to_csv(RES / "tables" / "openvocab_eval.csv", index=False)

    banner("SUMMARY  AUC against the city registers")
    print(f"  {'class':14s} {'n':>4s} {'Mask2Former':>12s} {'CLIPSeg':>9s}")
    for name, g in t.groupby("class"):
        y = g.label.values
        print(f"  {name:14s} {len(g):4d} {auc(g.closed_set.values, y):12.2f} "
              f"{auc(g.open_vocab.values, y):9.2f}")
    print("\n  0.5 is a coin flip. street_tree is the control: a harness that")
    print("  cannot score the class ADE20K certainly has is measuring itself.")
    print(f"\nwrote {RES / 'tables' / 'openvocab_eval.csv'}")


if __name__ == "__main__":
    main()
