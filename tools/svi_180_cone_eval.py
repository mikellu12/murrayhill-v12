"""Which weighting of the three cones best reproduces the VLM's own whole-view
judgement?

Reading raw regression coefficients is the obvious approach and the wrong one
here. Adjacent 60-degree cones of the same street are strongly correlated --
they look at overlapping frontage from the same point -- and collinear
predictors give coefficients that swing on small changes in the data while the
fit itself barely moves. A "centre weight of 0.72" read off an unstable
regression is not a finding.

So this reports four things, in order of how much they should be trusted:

1. COLLINEARITY FIRST. Correlation between cones and the design matrix's
   condition number. If the cones are near-collinear, every coefficient below
   is unstable and should be read as such -- stated rather than buried.

2. MODEL COMPARISON, the primary result. Rather than estimating weights, score
   candidate weightings against the whole-panorama rating the model already
   gave: centre-only, the literature's 0.10/0.80/0.10, uniform thirds, and a
   free fit. Out-of-sample R2 under GroupKFold on face_id. This answers the
   question that matters -- which weighting reproduces the model's own
   judgement -- and is unaffected by collinearity, because a fixed weighting
   has no coefficients to destabilise.

3. RELATIVE IMPORTANCE (LMG). Each cone's share of explained variance,
   averaged over every ordering of the predictors. Robust to collinearity
   where raw coefficients are not, and reads directly as "the centre accounts
   for X per cent of what the whole-view rating tracks".

4. FREE WEIGHTS, last and with a clustered bootstrap interval, so the width of
   that interval carries the collinearity warning rather than a point estimate
   implying precision it does not have.

Everything clusters on face_id. Moran's I here is 0.62-0.66 and nodes 20 m
apart on one block face see nearly the same street, so rows are not
independent and an unclustered interval would be far too narrow -- the mistake
CLAUDE.md records as v11's.

    .venv/Scripts/python tools/svi_180_cone_eval.py
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import PROC, RES, banner

# Listed here rather than imported from svi_180_sim_vlm, which pulls torch --
# absent from .venv by design, since that env is analysis-only.
RATE = ["vertical_greenery", "vertical_hardscape", "green_eye_level",
        "green_softening", "signage_detail", "enclosure", "facade_variation",
        "walking_room", "ground_floor_activity"]

CONES = ("left", "centre", "right")
CANDIDATES = {
    "centre only":       np.array([0.00, 1.00, 0.00]),
    "literature 10/80/10": np.array([0.10, 0.80, 0.10]),
    "uniform thirds":    np.array([1 / 3, 1 / 3, 1 / 3]),
}


def r2(y, pred):
    ss = ((y - y.mean()) ** 2).sum()
    return 1 - ((y - pred) ** 2).sum() / ss if ss > 0 else np.nan


def nnls_w(X, y):
    try:
        from scipy.optimize import nnls
        w, _ = nnls(X, y)
    except Exception:
        w = np.linalg.lstsq(X, y, rcond=None)[0].clip(0)
    return w / w.sum() if w.sum() > 0 else w


def lmg(X, y):
    """Each predictor's share of R2, averaged over all orderings (Lindeman,
    Merenda and Gold). Collinearity splits credit between correlated
    predictors instead of handing it arbitrarily to one."""
    k = X.shape[1]
    share = np.zeros(k)
    for order in itertools.permutations(range(k)):
        prev, used = 0.0, []
        for j in order:
            used.append(j)
            A = X[:, used]
            beta = np.linalg.lstsq(A, y, rcond=None)[0]
            cur = r2(y, A @ beta)
            share[j] += max(cur - prev, 0.0)
            prev = cur
    import math
    share /= math.factorial(k)
    return share / share.sum() if share.sum() > 0 else share


def grouped_cv_r2(X, y, groups, weights=None, k=5):
    """Out-of-sample R2 with whole faces held out."""
    uniq = pd.unique(groups)
    k = min(k, len(uniq))
    if k < 2:
        return np.nan
    folds = np.array_split(uniq, k)
    pred = np.full(len(y), np.nan)
    for f in folds:
        te = np.isin(groups, f)
        tr = ~te
        if tr.sum() < 4 or te.sum() == 0:
            continue
        if weights is None:
            w = nnls_w(X[tr], y[tr])
        else:
            w = weights
        pred[te] = X[te] @ w
    ok = ~np.isnan(pred)
    return r2(y[ok], pred[ok]) if ok.sum() > 3 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cones", type=Path,
                    default=RES / "tables" / "svi_180_cone_test.csv")
    ap.add_argument("--whole", type=Path,
                    default=RES / "tables" / "svi_180_sim_vlm.csv")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    banner("which cone weighting reproduces the whole-view rating?")

    c = pd.read_csv(args.cones)
    w = pd.read_csv(args.whole)
    faces = pd.read_csv(PROC / "metrics.csv")[["node_id", "face_id"]]
    j = c.merge(w, on="file", how="inner", suffixes=("", "_w")).merge(
        faces, on="node_id", how="left")
    j["face_id"] = j.face_id.fillna(j.node_id)
    print(f"{len(j)} images, {j.face_id.nunique()} faces, "
          f"{j.street.nunique()} streets\n")

    rng = np.random.default_rng(0)
    rows, imp_rows, wt_rows = [], [], []

    for f in RATE:
        cols = [f"{cn}_{f}" for cn in CONES]
        if f not in j.columns or not all(x in j.columns for x in cols):
            continue
        s = j[cols + [f, "face_id"]].dropna()
        if len(s) < 25 or s[f].nunique() < 2:
            continue
        X = s[cols].to_numpy(float)
        y = s[f].to_numpy(float)
        g = s.face_id.to_numpy()

        # 1. collinearity
        cc = np.corrcoef(X.T)
        cond = np.linalg.cond(X)

        # 2. model comparison, out of sample, faces held out
        scores = {k: grouped_cv_r2(X, y, g, wts) for k, wts in CANDIDATES.items()}
        scores["free fit"] = grouped_cv_r2(X, y, g, None)

        # 3. relative importance
        imp = lmg(X, y)
        imp_rows.append(imp)

        # 4. free weights with a face-clustered bootstrap
        w_hat = nnls_w(X, y)
        uniq = pd.unique(g)
        bw = []
        for _ in range(args.boot):
            pick = np.concatenate([np.flatnonzero(g == q)
                                   for q in rng.choice(uniq, len(uniq))])
            bw.append(nnls_w(X[pick], y[pick]))
        bw = np.array(bw)
        wt_rows.append(w_hat)

        rows.append({
            "field": f, "n": len(s),
            "r_LC": cc[0, 1], "r_CR": cc[1, 2], "r_LR": cc[0, 2], "cond": cond,
            **{k: v for k, v in scores.items()},
            "w_left": w_hat[0], "w_centre": w_hat[1], "w_right": w_hat[2],
            "c_lo": np.nanpercentile(bw[:, 1], 2.5),
            "c_hi": np.nanpercentile(bw[:, 1], 97.5),
        })

    if not rows:
        sys.exit("not enough rated fields in common -- has the cone run finished?")
    d = pd.DataFrame(rows)

    print("=== 1. collinearity between cones (read this first) ===")
    print(f"  mean r(left,centre) {d.r_LC.mean():.3f}   r(centre,right) "
          f"{d.r_CR.mean():.3f}   r(left,right) {d.r_LR.mean():.3f}")
    print(f"  mean condition number {d.cond.mean():.1f}"
          + ("   -> severe; coefficients unstable" if d.cond.mean() > 30 else
             "   -> moderate" if d.cond.mean() > 10 else "   -> mild"))

    print("\n=== 2. out-of-sample R2 by weighting, faces held out (PRIMARY) ===")
    keys = list(CANDIDATES) + ["free fit"]
    print(f"  {'field':<24}" + "".join(f"{k:>21}" for k in keys))
    for r in d.itertuples():
        print(f"  {r.field:<24}" + "".join(
            f"{getattr(r, k.replace(' ', '_').replace('/', '_')) if False else d.loc[r.Index, k]:>21.3f}"
            for k in keys))
    print(f"  {'MEAN':<24}" + "".join(f"{d[k].mean():>21.3f}" for k in keys))
    best = max(keys, key=lambda k: d[k].mean())
    print(f"\n  best mean out-of-sample R2: {best}")

    print("\n=== 3. relative importance, LMG share of explained variance ===")
    I = np.array(imp_rows)
    print(f"  {'':<12}{'left':>10}{'centre':>10}{'right':>10}")
    print(f"  {'mean':<12}{I[:,0].mean():>10.3f}{I[:,1].mean():>10.3f}{I[:,2].mean():>10.3f}")
    print(f"  {'median':<12}{np.median(I[:,0]):>10.3f}{np.median(I[:,1]):>10.3f}"
          f"{np.median(I[:,2]):>10.3f}")
    print(f"  {'literature':<12}{0.10:>10.3f}{0.80:>10.3f}{0.10:>10.3f}")
    print(f"  {'uniform':<12}{0.333:>10.3f}{0.333:>10.3f}{0.333:>10.3f}")

    print("\n=== 4. free weights, face-clustered bootstrap (least trustworthy) ===")
    print(f"  {'field':<24}{'left':>8}{'centre':>8}{'right':>8}   centre 95% CI")
    for r in d.itertuples():
        print(f"  {r.field:<24}{r.w_left:>8.3f}{r.w_centre:>8.3f}{r.w_right:>8.3f}"
              f"   [{r.c_lo:.3f}, {r.c_hi:.3f}]")
    W = np.array(wt_rows)
    print(f"  {'MEAN':<24}{W[:,0].mean():>8.3f}{W[:,1].mean():>8.3f}{W[:,2].mean():>8.3f}")

    out = RES / "tables" / "svi_180_cone_eval.csv"
    d.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
