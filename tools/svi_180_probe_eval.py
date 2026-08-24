"""Score candidate scaffolding signals against the labels, not the permits.

tools/scaffold_eval.py grades a detector against DOB permits. Those permits
were measured at chance against what is visually in the panorama
(docs/handpicking.md), so that grade cannot separate a bad detector from a
bad label. This grades against the visual verdicts instead, and reports the
permit flag as one more candidate so the two graders can be compared on the
same axis.

Clustering is not optional here. Moran's I is 0.62-0.66 and images 20 m apart
on one block face photograph the same scaffolding, so the effective sample is
the number of FACES, not the number of images. The probe is cross-validated
with GroupKFold on face_id and the bootstrap resamples faces, not rows.
Reporting a row-level interval would repeat v11's mistake.

Nothing here is wired into an index. It answers one question: does any
candidate beat 0.5, by enough to matter, on the labels that exist.

    .venv/Scripts/python tools/svi_180_probe_eval.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import PROC, RES, banner

RNG = np.random.default_rng(0)


def auc(scores, labels) -> float:
    """Rank AUC via the Mann-Whitney U identity. Ties get half credit."""
    labels = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype=float)
    pos, neg = s[labels], s[~labels]
    if not len(pos) or not len(neg):
        return float("nan")
    r = pd.Series(s).rank().to_numpy()
    return (r[labels].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def boot_ci(scores, labels, groups, n=2000):
    """Percentile interval, resampling FACES so correlated rows move together."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    uniq = pd.unique(groups)
    idx_by = {g: np.flatnonzero(groups == g) for g in uniq}
    out = []
    for _ in range(n):
        pick = np.concatenate([idx_by[g] for g in RNG.choice(uniq, len(uniq))])
        if labels[pick].all() or not labels[pick].any():
            continue
        out.append(auc(scores[pick], labels[pick]))
    return (np.nanpercentile(out, 2.5), np.nanpercentile(out, 97.5)) if out \
        else (float("nan"), float("nan"))


def main():
    banner("candidate scaffolding signals vs the visual labels")

    # Ground truth: both labelling passes, deduped, permits never consulted.
    truth = []
    b = RES / "review_sheets" / "balanced" / "sample_labelled.csv"
    if b.exists():
        d = pd.read_csv(b)
        truth.append(d.assign(y=d.vlm.astype(bool), src="balanced")[
            ["file", "node_id", "y", "src", "in_view"]])
    s = RES / "tables" / "svi_180_visual_labels.csv"
    if s.exists():
        d = pd.read_csv(s)
        truth.append(d.assign(y=d.vlm_structure.astype(bool), src="sequential")[
            ["file", "node_id", "y", "src", "in_view"]])
    if not truth:
        sys.exit("no visual labels found")
    t = pd.concat(truth, ignore_index=True).drop_duplicates("file")

    faces = pd.read_csv(PROC / "nodes_with_faces.csv")[["node_id", "face_id"]]
    t = t.merge(faces, on="node_id", how="left")
    t["face_id"] = t.face_id.fillna(t.node_id)

    ov = pd.read_csv(RES / "tables" / "svi_180_openvocab.csv")
    prompt_cols = [c for c in ov.columns
                   if c not in ("file", "node_id", "cardinal", "seq", "street",
                                "direction") and not c.endswith("__crop")]
    d = t.merge(ov[["file"] + prompt_cols], on="file", how="inner")

    vlm = RES / "tables" / "svi_180_vlm.csv"
    if vlm.exists():
        d = d.merge(pd.read_csv(vlm)[["file", "vlm_p_yes"]], on="file", how="left")

    seg = RES / "tables" / "svi_180_segformer.csv"
    if seg.exists():
        sg = pd.read_csv(seg)
        for c in ("fence", "building", "wall", "vegetation"):
            if c in sg.columns:
                d = d.merge(sg[["file", c]].rename(columns={c: f"seg_{c}"}),
                            on="file", how="left")

    n_face = d.face_id.nunique()
    print(f"{len(d)} labelled images, {int(d.y.sum())} positive, "
          f"{n_face} distinct faces")
    print(f"  by pass: {d.src.value_counts().to_dict()}")
    print(f"\nEffective sample is {n_face} faces, not {len(d)} images.\n")

    cands = [("permit flag (current label)", d.in_view.astype(float))]
    if "vlm_p_yes" in d.columns:
        cands.append(("GENERATIVE VLM  Qwen2.5-VL-3B P(yes)", d.vlm_p_yes))
    cands += [(c.replace("_", " "), d[c]) for c in prompt_cols]
    cands += [(c, d[c]) for c in d.columns if c.startswith("seg_")]

    # Scaffolding lives on dense built frontage, so a candidate can rank well
    # by recognising the neighbourhood. Within a band of similar building
    # share that route is closed; if AUC survives, the signal is structural.
    strat = None
    if "seg_building" in d.columns:
        strat = pd.qcut(d.seg_building, 3, labels=False, duplicates="drop")

    def strat_auc(sc):
        if strat is None:
            return float("nan")
        num = den = 0.0
        for b in pd.unique(strat):
            m = strat == b
            yy = d.y[m]
            if yy.nunique() < 2:
                continue
            w = int(yy.sum()) * int((~yy).sum())
            num += auc(sc[m.to_numpy()], yy) * w
            den += w
        return num / den if den else float("nan")

    print(f"{'candidate':<46}{'AUC':>7}   95% CI (faces)   {'strat':>6}")
    print("-" * 86)
    rows = []
    for name, sc in cands:
        sc = pd.to_numeric(sc, errors="coerce").fillna(0).to_numpy()
        a = auc(sc, d.y)
        lo, hi = boot_ci(sc, d.y, d.face_id.to_numpy())
        rows.append((name, a, lo, hi, strat_auc(sc)))
    for name, a, lo, hi, sa in sorted(rows, key=lambda r: -abs(r[1] - 0.5)):
        star = "  *" if lo > 0.5 or hi < 0.5 else "   "
        print(f"{name:<46}{a:>7.3f}   [{lo:.3f}, {hi:.3f}]{star}{sa:>6.3f}")

    # Linear probe on frozen DINOv2 features, out-of-fold, grouped by face.
    emb_p = PROC / "svi_180_dinov2.npz"
    if emb_p.exists():
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline

        z = np.load(emb_p)
        have = [f for f in d.file if f in z]
        sub = d[d.file.isin(have)].reset_index(drop=True)
        # Four crops per image; concatenated so the probe can weight the
        # thirds differently from the whole strip.
        X = np.stack([np.asarray(z[f]).ravel() for f in sub.file])
        y = sub.y.to_numpy()
        g = sub.face_id.to_numpy()
        k = min(5, len(np.unique(g)))
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=k).split(X, y, groups=g):
            m = make_pipeline(StandardScaler(),
                              LogisticRegression(C=0.01, max_iter=5000))
            m.fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        a = auc(oof, y)
        lo, hi = boot_ci(oof, y, g)
        print("-" * 78)
        sa = strat_auc(oof) if strat is not None else float("nan")
        print(f"{'PURE VISION  DINOv2 probe (oof, ' + str(k) + '-fold by face)':<46}"
              f"{a:>7.3f}   [{lo:.3f}, {hi:.3f}]"
              + ("  *" if lo > 0.5 else "   ") + f"{sa:>6.3f}")
        print(f"  {X.shape[1]}-dim features, {len(y)} images, "
              f"{len(np.unique(g))} faces, {int(y.sum())} positive")

    print("\n*  interval excludes 0.5")
    print("CLIPSeg's published bar on this question is AUC 0.55 (0.51 in cone).")


if __name__ == "__main__":
    main()
