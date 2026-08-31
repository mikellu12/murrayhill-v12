"""Every model on the same two targets, the same images, the same metric.

Earlier comparisons in this project scored different models on different
tasks -- one on scaffolding labels, another on greenery, a third on open-
vocabulary boxes -- which cannot rank anything. This scores all of them
against the two quantities the pipeline actually measures, GVI and VEI, over
the same 1,254 panoramas, by Spearman with a bootstrap clustered on face_id.

Spearman, not R2: these predictors are on wildly different scales -- pixel
shares, 1-7 ratings, detector logits, ridge predictions -- and only their
ordering is comparable. Clustered on faces because Moran's I is 0.62-0.66 and
nodes 20 m apart photograph nearly the same street, so an unclustered interval
would be far too narrow.

TWO ENTRIES ARE CIRCULAR AND ARE LABELLED, NOT HIDDEN:

  Mask2Former produces GVI and VEI. Scoring it against them would return 1.0
  and mean nothing, so it is excluded rather than reported.

  SegFormer's vegetation share is a vegetation pixel share, and GVI is a
  vegetation pixel share under a different taxonomy. It is close to the same
  measurement twice. That is not cheating -- it is the correct baseline, and
  the right way to read a VLM beating or losing to it -- but the comparison is
  not arms-length and the table says so.

The VLM entries are arms-length: an ordinal judgement by eye, never shown a
pixel count.

    .venv/Scripts/python tools/model_benchmark.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import PROC, RES, banner

RNG = np.random.default_rng(0)


def spearman_ci(x, y, groups, n=3000):
    s = pd.DataFrame({"x": x, "y": y, "g": groups}).dropna()
    if len(s) < 25 or s.x.nunique() < 2:
        return None
    r = s.x.corr(s.y, method="spearman")
    uniq = pd.unique(s.g)
    idx = {q: np.flatnonzero(s.g.to_numpy() == q) for q in uniq}
    bs = []
    for _ in range(n):
        pick = np.concatenate([idx[q] for q in RNG.choice(uniq, len(uniq))])
        sub = s.iloc[pick]
        if sub.x.nunique() > 1:
            bs.append(sub.x.corr(sub.y, method="spearman"))
    return r, np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5), len(s)


def dinov2_oof(files, target, groups):
    """Out-of-fold ridge predictions from frozen embeddings, faces held out."""
    p = PROC / "svi_180_dinov2.npz"
    if not p.exists():
        return None
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    z = np.load(p)
    keep = [i for i, f in enumerate(files) if f in z and np.isfinite(target[i])]
    if len(keep) < 50:
        return None
    X = np.stack([np.asarray(z[files[i]]).ravel() for i in keep])
    y = target[keep]
    g = groups[keep]
    oof = np.full(len(y), np.nan)
    k = min(5, len(np.unique(g)))
    for tr, te in GroupKFold(n_splits=k).split(X, y, groups=g):
        m = make_pipeline(StandardScaler(), Ridge(alpha=100.0))
        m.fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
    return keep, oof, y, g


def main():
    banner("all models, same targets, same images")
    met = pd.read_csv(PROC / "metrics.csv")[["node_id", "GVI", "VEI", "face_id"]]

    def load(name, cols):
        p = RES / "tables" / f"{name}.csv"
        if not p.exists():
            return None
        d = pd.read_csv(p)
        keep = ["file", "node_id"] + [c for c in cols if c in d.columns]
        return d[keep]

    base = load("svi_180_segformer", ["vegetation", "building", "sky", "sidewalk"])
    if base is None:
        sys.exit("need results/tables/svi_180_segformer.csv")
    d = base.merge(met, on="node_id", how="left")

    for name, cols, pre in (
        ("svi_180_qwen_morphology", ["greenery_dominance", "framing_score"], "qwen1"),
        ("svi_180_sim_vlm", ["green_eye_level", "vertical_greenery", "enclosure",
                             "vertical_hardscape", "Imageability_vlm",
                             "Identity_vlm"], "qwen2"),
        ("svi_180_openvocab", ["a_street_tree", "scaffolding"], "owl"),
    ):
        t = load(name, cols)
        if t is None:
            continue
        t = t.drop(columns=["node_id"]).rename(
            columns={c: f"{pre}_{c}" for c in t.columns if c != "file"})
        d = d.merge(t, on="file", how="left")

    g = d.face_id.fillna(d.node_id).to_numpy()
    rows = []

    ENTRIES = [
        # label, column, target, arms-length?
        ("SegFormer-B5  vegetation share", "vegetation", "GVI", False),
        ("SegFormer-B5  building share", "building", "VEI", False),
        ("SegFormer-B5  sky share", "sky", "VEI", False),
        ("Qwen2-VL-7B   greenery rating", "qwen1_greenery_dominance", "GVI", True),
        ("Qwen2-VL-7B   green_eye_level", "qwen2_green_eye_level", "GVI", True),
        ("Qwen2-VL-7B   vertical_greenery", "qwen2_vertical_greenery", "GVI", True),
        ("Qwen2-VL-7B   Imageability", "qwen2_Imageability_vlm", "GVI", True),
        ("Qwen2-VL-7B   enclosure rating", "qwen2_enclosure", "VEI", True),
        ("Qwen2-VL-7B   vertical_hardscape", "qwen2_vertical_hardscape", "VEI", True),
        ("Qwen2-VL-7B   framing_score", "qwen1_framing_score", "VEI", True),
        ("Qwen2-VL-7B   Identity", "qwen2_Identity_vlm", "VEI", True),
        ("OWLv2         'a street tree'", "owl_a_street_tree", "GVI", True),
        # Negative control, not a comparison: a scaffolding prompt should NOT
        # predict greenery. Near zero is the correct result, and reading it as a
        # model weakness would be a mistake.
        ("OWLv2  'scaffolding' (neg. control)", "owl_scaffolding", "GVI", True),
    ]
    for label, col, tgt, arms in ENTRIES:
        if col not in d.columns:
            continue
        r = spearman_ci(d[col].to_numpy(float), d[tgt].to_numpy(float), g)
        if r:
            rows.append((label, tgt, r[0], r[1], r[2], r[3], arms))

    # DINOv2 needs fitting, so it is out-of-fold with faces held out
    for tgt in ("GVI", "VEI"):
        got = dinov2_oof(d.file.tolist(), d[tgt].to_numpy(float), g)
        if got:
            keep, oof, y, gg = got
            r = spearman_ci(oof, y, gg)
            if r:
                rows.append((f"DINOv2-large  ridge probe (oof)", tgt,
                             r[0], r[1], r[2], r[3], True))

    out = pd.DataFrame(rows, columns=["model", "target", "rho", "lo", "hi",
                                      "n", "arms_length"])
    for tgt in ("GVI", "VEI"):
        sub = out[out.target == tgt].sort_values("rho", ascending=False)
        print(f"\n=== predicting measured {tgt} ===")
        print(f"  {'model':<36}{'rho':>8}   95% CI (faces)     n")
        for r in sub.itertuples():
            flag = "" if r.arms_length else "   <- same quantity, not arms-length"
            print(f"  {r.model:<36}{r.rho:>8.3f}   [{r.lo:+.3f},{r.hi:+.3f}]  "
                  f"{r.n:>5}{flag}")
    print("\n  Mask2Former is excluded: it produces GVI and VEI, so scoring it")
    print("  against them would return 1.0 and mean nothing.")
    p = RES / "tables" / "model_benchmark.csv"
    out.to_csv(p, index=False)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
