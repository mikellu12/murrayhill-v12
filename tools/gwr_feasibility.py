"""Can the GWR calibration in section 2.8 be estimated on this frame?

The section proposes fitting

    ln(t_base) = b0(s) + b_Img(s) ln I + b_Id(s) ln Y + b_Dep(s) ln D + e

separately at every node, weighting neighbours by a Gaussian kernel, then
normalising |b| into the elasticities a, b, c. It needs t_base -- localized
pedestrian stayability counts -- which this study does not have, so the
regression cannot be run.

What CAN be tested is whether it would be identifiable if the counts arrived.
Three preconditions, none of which involve the outcome:

  1. The three regressors must be separable. If ln I, ln Y and ln D move
     together, no amount of data separates b_Img from b_Id -- the local solve
     is degenerate whatever t_base turns out to be.
  2. Each local fit needs enough effective observations for four parameters.
     A Gaussian kernel does not have a hard edge, so the count that matters is
     the effective sample size, sum(w)^2 / sum(w^2), not how many nodes are
     "nearby".
  3. X'WX must be invertible and well conditioned at every node, or the
     coefficients are noise dressed as local variation.

Failing any of these means the section is not merely blocked on data; it would
not work on this frame even with the data.

    .venv/Scripts/python tools/gwr_feasibility.py
"""
import sys
from pathlib import Path

import json

import geopandas as gpd
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import PROJ_CRS, PROC, RES, banner

UTM = PROJ_CRS       # metre CRS from config; 32618 for Manhattan
BANDWIDTHS = [60, 100, 150, 250, 400]   # metres


def vif(X):
    out = []
    for k in range(X.shape[1]):
        y = X[:, k]
        A = np.delete(X, k, axis=1)
        A = np.c_[np.ones(len(A)), A]
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        r2 = 1 - ((y - A @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        out.append(np.inf if r2 >= 1 else 1 / (1 - r2))
    return out


def main():
    banner("is the GWR of section 2.8 identifiable here?")
    v = pd.read_csv(RES / "tables" / "vlm_calculations.csv")
    n = v.groupby("node_id")[["I", "Y", "D"]].mean().reset_index()
    g = gpd.read_file(PROC / "nodes.gpkg").to_crs(UTM)[["node_id", "geometry"]]
    d = n.merge(g, on="node_id", how="inner").dropna()
    d = gpd.GeoDataFrame(d, geometry="geometry", crs=UTM)
    print(f"{len(d)} nodes with all three dimensions\n")

    # the regressors are logs of the dimensions, so they must be positive
    for c in ("I", "Y", "D"):
        bad = int((d[c] <= 0).sum())
        print(f"  {c}: min {d[c].min():.4f}"
              + (f"   {bad} non-positive -> ln undefined" if bad else "   ln ok"))
    L = np.column_stack([np.log(d[c].to_numpy()) for c in ("I", "Y", "D")])

    print("\n1. are the regressors separable?")
    C = np.corrcoef(L.T)
    names = ["ln I", "ln Y", "ln D"]
    print(f"     {'':<7}" + "".join(f"{x:>8}" for x in names))
    for i, nm in enumerate(names):
        print(f"     {nm:<7}" + "".join(f"{C[i, j]:>8.3f}" for j in range(3)))
    vs = vif(L)
    print(f"     VIF  " + "  ".join(f"{nm} {x:.2f}" for nm, x in zip(names, vs)))
    worst = max(vs)
    print(f"     -> {'PASS' if worst < 5 else 'FAIL'}: "
          f"highest VIF {worst:.2f} ({'under' if worst < 5 else 'over'} the "
          f"conventional 5)")

    print("\n2. effective sample size per local fit  (needs > 4 for 4 params)")
    xy = np.column_stack([d.geometry.x, d.geometry.y])
    dist = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
    print(f"     {'bandwidth':>10}{'median ESS':>12}{'min ESS':>10}"
          f"{'nodes ESS<8':>13}")
    for b in BANDWIDTHS:
        W = np.exp(-0.5 * (dist / b) ** 2)
        ess = W.sum(1) ** 2 / (W ** 2).sum(1)
        print(f"     {b:>10}{np.median(ess):>12.1f}{ess.min():>10.1f}"
              f"{int((ess < 8).sum()):>13}")

    print("\n3. is X'WX invertible and well conditioned?")
    X = np.c_[np.ones(len(L)), L]
    print(f"     {'bandwidth':>10}{'median cond':>14}{'worst cond':>13}"
          f"{'singular':>10}")
    for b in BANDWIDTHS:
        W = np.exp(-0.5 * (dist / b) ** 2)
        conds, sing = [], 0
        for i in range(len(X)):
            XtWX = X.T @ (W[i][:, None] * X)
            c = np.linalg.cond(XtWX)
            if not np.isfinite(c) or c > 1e12:
                sing += 1
            else:
                conds.append(c)
        print(f"     {b:>10}{np.median(conds):>14.3e}{max(conds):>13.3e}"
              f"{sing:>10}")
    print("\n     a condition number over 1e6 means the local coefficients are")
    print("     numerically unstable: small changes in t_base would swing them.")

    print("\n4. what the normalisation does to a negative coefficient")
    print("     w_k = |b_k| / sum|b_j| discards the sign. In a Cobb-Douglas,")
    print("     I^+0.5 and I^-0.5 move M in opposite directions, so a dimension")
    print("     that reduces dwell would be reported as one that matters a lot")
    print("     in the same direction as the others. Worth resolving before the")
    print("     counts arrive, not after.")


if __name__ == "__main__":
    main()
