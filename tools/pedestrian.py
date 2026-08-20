"""
Pedestrian-realm composition, by node and by travel direction.

GVI and VEI describe the built envelope -- how green, how enclosed. They
say nothing about the ground plane a pedestrian actually negotiates:
sidewalk area, seating, barriers, steps, planting at eye level. Those are
the elements a woonerf-style intervention adds or removes, so a streetscape
study that only reports canopy and enclosure cannot speak to them.

TWO DETECTORS, AND THE REASON FOR BOTH
--------------------------------------
Mask2Former on ADE20K has native classes for sidewalk, bench, stairs,
fence, railing, pole, streetlight, ashcan and vehicles. Those are dense,
exhaustive and reproducible, so they carry the quantitative claims.

It has no class for bollards, planters, tree pits, tactile paving, bike
racks or play equipment. CLIPSeg fills those from text prompts. Its soft
masks need a threshold, that threshold is not tuned against hand-labelled
data here, and prompt-driven detection is sensitive to wording -- so treat
the CLIPSeg classes as INDICATIVE, report the threshold, and do not build
a headline result on them.

DIRECTIONAL ATTRIBUTION
-----------------------
Each pixel column maps to an absolute bearing gnomonically, exactly as in
s03_profiles. Class shares are accumulated into 1-degree bins, so the same
profile yields the 360 composition and each 180-degree forward view.

Sampling 5 nodes costs a few seconds of GPU time. Nothing is re-fetched.

    python pedestrian.py                 # 5 stratified nodes
    python pedestrian.py --n 8 --all     # more nodes, all of them scored
"""
import argparse, sys
from pathlib import Path

import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import (CFG, PROC, RES, DIRECTIONS, column_bearings,
                    column_weights, bin_mask, device_and_batch,
                    load_segmenter, banner, image_path)

NB = CFG["directional"]["n_bins"]

# The class definitions live in config.yaml under `pedestrian:`, with the
# reasoning for the exact-match rule and the omitted synonyms beside them.
# tools/face_samples.py reads the same three, so a legend drawn there and a
# share reported here cannot disagree.
PED = CFG["pedestrian"]
ADE_GROUPS = PED["ade_groups"]
CLIPSEG_PROMPTS = PED["clipseg_prompts"]
PALETTE = PED["palette"]
FOV = CFG["directional"]["fov"]


def resolve(id2label, needles):
    """Exact match against ADE20K synonym parts -- never substring.

    Substring matching silently mis-assigns: "tree" is inside "street
    lamp", "car" inside "placard", "sky" inside "skyscraper". Each of
    those put a class in the wrong group, and none of them is visible in
    the output. Exact matching on the comma-separated synonyms is the only
    safe rule, which means every synonym has to be listed explicitly.
    """
    want = {n.strip().lower() for n in needles}
    out = []
    for i, lab in id2label.items():
        parts = {p.strip().lower() for p in lab.split(",")}
        if parts & want:
            out.append(int(i))
    return out


def build_profiles(node_ids, manifest, proc, model, dev, groups,
                   clipseg=None, thresh=0.4):
    """(class x bin) profiles plus a weight row, per node."""
    names = list(groups) + (list(CLIPSEG_PROMPTS) if clipseg else [])
    by_node = {n: g for n, g in manifest.groupby("node_id")}
    out, maps = {}, {}
    CW = None

    for nid in node_ids:
        acc = np.zeros((len(names) + 1, NB))
        for _, r in by_node[nid].iterrows():
            img = Image.open(image_path(r.path)).convert("RGB")
            inp = proc(images=[img], return_tensors="pt").to(dev)
            with torch.inference_mode():
                o = model(**inp)
            # Same MPS contiguity issue as s03_profiles.
            o.class_queries_logits = o.class_queries_logits.cpu().contiguous()
            o.masks_queries_logits = o.masks_queries_logits.cpu().contiguous()
            a = proc.post_process_semantic_segmentation(
                o, target_sizes=[img.size[::-1]])[0].numpy()
            H, W = a.shape
            if CW is None or CW.shape[0] != W:
                CW = column_weights(H, W)
            idx = np.floor(column_bearings(float(r.heading), W)).astype(int) % NB

            for ci, g in enumerate(groups):
                np.add.at(acc[ci], idx, np.isin(a, groups[g]).mean(axis=0) * CW)

            if clipseg:
                cp, cm = clipseg
                texts = list(CLIPSEG_PROMPTS.values())
                ci_inp = cp(text=texts, images=[img] * len(texts),
                            padding=True, return_tensors="pt").to(
                                "cpu" if dev == "mps" else dev)
                with torch.inference_mode():
                    pr = torch.sigmoid(cm(**ci_inp).logits).cpu().numpy()
                for k, _ in enumerate(CLIPSEG_PROMPTS):
                    m = pr[k] > thresh
                    # CLIPSeg emits at 352x352; resample columns to native.
                    xi = (np.arange(W) * m.shape[1] // W)
                    col = m[:, xi].mean(axis=0) * CW
                    np.add.at(acc[len(groups) + k], idx, col)

            np.add.at(acc[-1], idx, CW)
            if r.heading == 90:
                maps[nid] = (img, a)
        out[nid] = acc
    return out, names, maps


def shares(acc, names, centre, fov):
    m = bin_mask(centre, fov)
    W = acc[-1][m].sum()
    if W <= 0:
        return {}
    d = {n: acc[i][m].sum() / W for i, n in enumerate(names)}
    known = sum(v for k, v in d.items() if k in ADE_GROUPS)
    d["other"] = max(0.0, 1.0 - known)
    return d


def figure_overlay(maps, metrics, groups, out, name=None):
    """Original above, role overlay below -- the VSeg panel, extended."""
    ids = list(maps)
    fig, ax = plt.subplots(2, len(ids), figsize=(3.3 * len(ids), 7.4),
                           gridspec_kw={"hspace": .02, "wspace": .02})
    if len(ids) == 1:
        ax = ax.reshape(2, 1)
    order = list(groups)
    lut = np.zeros(200, dtype=int)
    for gi, g in enumerate(order, start=1):
        for cid in groups[g]:
            lut[cid] = gi
    cols = ["#3f3f3f"] + [PALETTE[g] for g in order]
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(cols)

    for c, nid in enumerate(ids):
        img, a = maps[nid]
        r = metrics[metrics.node_id.eq(nid)]
        lab = (f"{r.osm_name.iloc[0]}\nGVI {r.GVI.iloc[0]:.1f}%  "
               f"VEI {r.VEI.iloc[0]:.2f}") if len(r) else nid
        ax[0, c].imshow(img)
        ax[0, c].set_title(lab, fontsize=9)
        ax[1, c].imshow(img)
        ax[1, c].imshow(lut[a], cmap=cmap, vmin=0, vmax=len(order),
                        alpha=.55, interpolation="nearest")
        for a_ in (ax[0, c], ax[1, c]):
            a_.set_xticks([]); a_.set_yticks([])
    ax[0, 0].set_ylabel("Street View", fontsize=10)
    ax[1, 0].set_ylabel("pedestrian realm", fontsize=10)
    fig.legend(handles=[Patch(color=PALETTE[g], label=g) for g in order],
               loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(.5, -.04), fontsize=9)
    fn = f"overlay_{name}.png" if name else "figure_pedestrian_overlay.png"
    plt.savefig(out / fn, dpi=300, bbox_inches="tight")
    plt.close()


def figure_directions(df, out):
    """Composition per view: 360 and each 180-degree forward direction."""
    views = ["full360"] + list(DIRECTIONS)
    nodes = df.node_id.unique()
    if len(nodes) > 8:
        # One row per node stops being legible past about eight.
        nodes = nodes[:8]
        print(f"  direction figure limited to the first 8 of "
              f"{df.node_id.nunique()} nodes; the CSV has them all")
    stack = [k for k in list(ADE_GROUPS) + ["other"] if k in df.columns]

    fig, axes = plt.subplots(len(nodes), 1,
                             figsize=(11, 2.3 * len(nodes)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, nid in zip(axes, nodes):
        sub = df[df.node_id.eq(nid)].set_index("view").reindex(views)
        left = np.zeros(len(views))
        for k in stack:
            v = sub[k].fillna(0).values * 100
            ax.barh(range(len(views)), v, left=left, color=PALETTE[k],
                    height=.72, label=k)
            left += v
        ax.set_yticks(range(len(views)))
        ax.set_yticklabels([v.replace("_", " ") for v in views], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f"{sub.osm_name.dropna().iloc[0]}  ({nid})", fontsize=9,
                     loc="left")
        ax.set_xlim(0, 100)
    axes[-1].set_xlabel("share of view (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(.5, -.02), fontsize=9)
    plt.tight_layout()
    plt.savefig(out / "figure_pedestrian_directions.png", dpi=300,
                bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5,
                    help="nodes to sample (default 5, stratified by GVI)")
    ap.add_argument("--per-street", action="store_true",
                    help="sample --n nodes from EACH street rather than "
                         "--n in total; writes one overlay figure per street")
    ap.add_argument("--all", action="store_true",
                    help="score every node (slow, but gives per-street stats)")
    ap.add_argument("--no-clipseg", action="store_true")
    a = ap.parse_args()

    banner("pedestrian realm")
    import geopandas as gpd
    metrics = gpd.read_file(PROC / "metrics.gpkg")
    manifest = pd.read_csv(PROC / "manifest.csv")
    dev, _ = device_and_batch()
    proc, model = load_segmenter(dev)

    groups, seen = {}, {}
    for g, needles in ADE_GROUPS.items():
        ids = resolve(model.config.id2label, needles)
        if ids:
            groups[g] = ids
            print(f"  {g:11s} "
                  f"{[model.config.id2label[i].split(',')[0] for i in ids]}")
            for i in ids:
                seen.setdefault(i, []).append(g)
        else:
            print(f"  {g:11s} -- no ADE20K class matched, dropped")

    # A class in two groups would be double-counted in the shares, which
    # is invisible in the output unless it is checked for explicitly.
    dup = {model.config.id2label[i]: v for i, v in seen.items() if len(v) > 1}
    if dup:
        print("\n  !! double-assigned classes -- fix ADE_GROUPS:")
        for k, v in dup.items():
            print(f"     {k!r} -> {v}")
        sys.exit("aborting: shares would not sum correctly")

    clipseg = None
    if not a.no_clipseg:
        from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
        mid = CFG["open_vocabulary"]["model"]
        clipseg = (CLIPSegProcessor.from_pretrained(mid),
                   CLIPSegForImageSegmentation.from_pretrained(mid).to(
                       "cpu" if dev == "mps" else dev).eval())
        print(f"  CLIPSeg: {list(CLIPSEG_PROMPTS)} "
              f"(threshold {CFG['open_vocabulary']['threshold']}, indicative only)")

    have = set(manifest.node_id)
    pool = metrics[metrics.node_id.isin(have)].dropna(subset=["GVI"])

    def stratified(df, k):
        """k nodes spanning the GVI range, not whichever look best."""
        if len(df) <= k:
            return df.node_id.tolist()
        q = pd.qcut(df.GVI, min(k, 4), labels=False, duplicates="drop")
        out = (df.assign(q=q).groupby("q", group_keys=False)
                 .apply(lambda g: g.sample(min(len(g), max(1, k // q.nunique())),
                                           random_state=CFG["seed"]))
                 .node_id.tolist())
        # Top up if integer division left the quota short.
        rest = [n for n in df.node_id if n not in set(out)]
        return (out + rest)[:k]

    groups_by_street = None
    if a.all:
        sel = pool.node_id.tolist()
    elif a.per_street:
        groups_by_street = {st: stratified(g, a.n)
                            for st, g in pool.groupby("osm_name")}
        sel = [n for v in groups_by_street.values() for n in v]
        print(f"\n{a.n} node(s) from each of {len(groups_by_street)} streets")
    else:
        sel = stratified(pool, a.n)
    print(f"\nscoring {len(sel)} node(s)")

    prof, names, maps = build_profiles(
        sel, manifest, proc, model, dev, groups, clipseg,
        CFG["open_vocabulary"]["threshold"])

    rows = []
    for nid, acc in prof.items():
        for view, centre, fov in ([("full360", 0.0, 360.0)] +
                                  [(k, b, FOV) for k, b in DIRECTIONS.items()]):
            d = shares(acc, names, centre, fov)
            d.update(node_id=nid, view=view)
            r = metrics[metrics.node_id.eq(nid)]
            if len(r):
                d["osm_name"] = r.osm_name.iloc[0]
                d["typology"] = r.typology.iloc[0]
            rows.append(d)

    df = pd.DataFrame(rows)
    front = ["node_id", "osm_name", "typology", "view"]
    df = df[front + [c for c in df.columns if c not in front]]
    (RES / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(RES / "tables" / "pedestrian_realm.csv", index=False)

    show = [c for c in ["sidewalk", "road", "vegetation", "seating",
                        "barrier", "steps", "furniture"] if c in df.columns]
    print("\nshare of view (%), by node and direction:")
    print((df.set_index(["osm_name", "view"])[show] * 100)
          .round(1).to_string())

    print("\nmean across sampled nodes, by view:")
    print((df.groupby("view")[show].mean() * 100).round(1).to_string())
    print("\n  Differences between directions at the same node are the")
    print("  anisotropy a 360 index averages away. Sidewalk share in")
    print("  particular should differ along-street versus cross-street.")

    figdir = RES / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    if groups_by_street:
        # One overlay per street: 15 streets x 5 nodes in a single panel
        # would be unreadable, so they are written separately.
        sub = figdir / "pedestrian_by_street"
        sub.mkdir(exist_ok=True)
        for st, ids in groups_by_street.items():
            mm = {k: v for k, v in maps.items() if k in ids}
            if mm:
                slug = "".join(c if c.isalnum() else "_" for c in st).strip("_")
                figure_overlay(mm, metrics, groups, sub, name=slug)
        print(f"\nwrote {len(groups_by_street)} per-street overlays to {sub}")

        cols = [c for c in ADE_GROUPS if c in df.columns]
        sub360 = df[df.view.eq("full360")]
        # Flatten before writing. A MultiIndex column frame round-trips
        # through CSV as two header rows, so read_csv puts the words
        # mean/count in the first data row and every column becomes object
        # dtype -- the table then renders as text in the dashboard.
        per = sub360.groupby("osm_name")[cols].mean().round(5)
        per.columns = [f"{c}_mean" for c in per.columns]
        per.insert(0, "n_nodes", sub360.groupby("osm_name").node_id.nunique())
        per = per.reset_index()
        per.to_csv(RES / "tables" / "pedestrian_by_street.csv", index=False)
        show2 = [c for c in ["sidewalk", "vegetation", "seating", "barrier",
                             "steps", "furniture"] if c in df.columns]
        print("\nmean share of view (%) per street, 360 view:")
        print((df[df.view.eq("full360")].groupby("osm_name")[show2].mean() * 100)
              .round(1).sort_values("vegetation", ascending=False).to_string())
    else:
        figure_overlay(maps, metrics, groups, figdir)

    figure_directions(df, figdir)
    print(f"\nwrote pedestrian_realm.csv and the figures to {figdir}")


if __name__ == "__main__":
    main()
