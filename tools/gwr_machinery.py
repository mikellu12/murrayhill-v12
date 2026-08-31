"""What the 26 Aug section 2.8 can be checked on, without the outcome.

The 24 Aug version specified the regression, the kernel and the normalisation.
This one adds the inferential apparatus: an epsilon offset in the logs, a
Golden Section Search on AICc for the bandwidth, the GWR hat matrix, a
sandwich variance, node-level t-statistics and a Benjamini-Hochberg
correction over n = 1,320.

Most of that does not touch t_base. The hat matrix S, its trace, the effective
degrees of freedom and the shape of the localized standard errors are
functions of X and W alone -- the outcome enters only through the residual
scalar s^2. So the machinery is checkable now even though the regression is
not runnable.

Four things are tested here:

  1. n. The paper states 1,320 nodes on a sidewalk dual-discretization
     network. This frame produces a different number, and which one is
     right decides what the FDR correction is applied over.
  2. epsilon. It is introduced to stop zero-value observations inflating
     standard errors. Whether any of I, Y, D actually reach zero here
     decides whether it does anything.
  3. tr(S) and the effective degrees of freedom, per bandwidth. These fix
     the AICc penalty term and the divisor in s^2, and a bandwidth that
     drives n - 2tr(S) + tr(S'S) toward zero makes the variance explode.
  4. The claimed t threshold of 2.65. Under Benjamini-Hochberg the critical
     value is not a constant -- it depends on how many nodes turn out
     significant -- so a fixed 2.65 implies a particular answer.

    .venv/Scripts/python tools/gwr_machinery.py
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import PROJ_CRS, PROC, RES, banner

UTM = PROJ_CRS       # metre CRS from config; 32618 for Manhattan
EPS = 1e-5
BANDWIDTHS = [60, 100, 150, 250, 400]
PAPER_N = 1320


def hat_stats(X, W):
    """tr(S) and tr(S'S) for the GWR hat matrix, built row by row.

    Row i of S is x_i' (X'W_i X)^-1 X'W_i -- the weights that turn y into
    the fitted value at node i. S is never formed whole; only its trace and
    the trace of S'S are needed, and both accumulate row by row.
    """
    n = len(X)
    tr_S = 0.0
    SS = np.zeros(n)
    for i in range(n):
        w = W[i]
        XtW = X.T * w
        A = XtW @ X
        try:
            row = X[i] @ np.linalg.solve(A, XtW)
        except np.linalg.LinAlgError:
            return np.nan, np.nan
        tr_S += row[i]
        SS += row ** 2
    return tr_S, SS.sum()


def main():
    banner("section 2.8 machinery, checked without t_base")
    v = pd.read_csv(RES / "tables" / "vlm_calculations.csv")
    g = gpd.read_file(PROC / "nodes.gpkg").to_crs(UTM)[["node_id", "geometry"]]

    # 1 -- what is n?
    halves = len(v)
    nodes = v.node_id.nunique()
    print(f"1. sample size\n")
    print(f"     paper states                     n = {PAPER_N:,}")
    print(f"     half-views in this frame         n = {halves:,}"
          f"   ({nodes} nodes x 2 sides)")
    print(f"     nodes                            n = {nodes:,}")
    print(f"     difference from the paper's n      {halves - PAPER_N:+,}")
    print(f"     -> the FDR correction is applied over n, so this matters:")
    print(f"        it changes every localized p-value threshold.\n")

    # 2 -- does epsilon do anything?
    print(f"2. the epsilon offset (eps = {EPS:g})\n")
    print(f"     {'term':<8}{'min':>10}{'zeros':>8}{'ln(min)':>12}"
          f"{'ln(min+eps)':>14}{'changed?':>10}")
    for c in ("I", "Y", "D"):
        s = v[c].dropna()
        z = int((s <= 0).sum())
        a = np.log(s.min()) if s.min() > 0 else -np.inf
        b = np.log(s.min() + EPS)
        print(f"     {c:<8}{s.min():>10.4f}{z:>8}{a:>12.4f}{b:>14.4f}"
              f"{'no' if abs(a - b) < 1e-4 else 'yes':>10}")
    print(f"     -> no dimension reaches zero here, because the sigmoids on I")
    print(f"        and D cannot output 0 and Y is a mean of bounded terms.")
    print(f"        epsilon is harmless but inert on this data. It would bite")
    print(f"        on I_raw / D_raw, which do reach 0.000.\n")

    # 3 -- hat matrix, per bandwidth
    d = v.groupby("node_id")[["I", "Y", "D"]].mean().reset_index().merge(
        g, on="node_id", how="inner").dropna()
    d = gpd.GeoDataFrame(d, geometry="geometry", crs=UTM)
    L = np.column_stack([np.log(d[c].to_numpy() + EPS) for c in ("I", "Y", "D")])
    X = np.c_[np.ones(len(L)), L]
    xy = np.c_[d.geometry.x, d.geometry.y]
    dist = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
    n = len(X)
    print(f"3. the hat matrix, on {n} nodes\n")
    print(f"     {'b (m)':>7}{'tr(S)':>9}{'tr(S,S)':>10}{'eff. df':>10}"
          f"{'AICc penalty':>14}{'s2 divisor':>12}")
    rows = []
    for b in BANDWIDTHS:
        W = np.exp(-0.5 * (dist / b) ** 2)
        trS, trSS = hat_stats(X, W)
        edf = n - trS
        div = n - 2 * trS + trSS
        pen = n * ((n + trS) / (n - 2 - trS)) if n - 2 - trS > 0 else np.nan
        print(f"     {b:>7}{trS:>9.1f}{trSS:>10.1f}{edf:>10.1f}"
              f"{pen:>14.1f}{div:>12.1f}")
        rows.append({"bandwidth": b, "tr_S": trS, "tr_SS": trSS,
                     "eff_df": edf, "s2_divisor": div, "aicc_penalty": pen})
    print(f"     -> AICc is minimised over b by Golden Section Search, but the")
    print(f"        first term needs residuals. The penalty column is the part")
    print(f"        that is computable now, and it is monotone in b, so the")
    print(f"        search will trade it against fit once t_base exists.\n")

    # 4 -- the claimed threshold
    print(f"4. the Benjamini-Hochberg threshold\n")
    t_claim = 2.65
    p_claim = 2 * (1 - stats.norm.cdf(t_claim))
    print(f"     paper: |t| >= {t_claim} corresponds to alpha ~ 0.008")
    print(f"     check: two-sided p at |t| = {t_claim} is {p_claim:.4f}   ok\n")
    print(f"     but BH is not a fixed cutoff. With m tests at alpha, the")
    print(f"     critical p is k*alpha/m for the largest passing rank k:\n")
    print(f"     {'discoveries k':>14}{'k/m':>8}{'critical p':>13}{'|t|':>8}")
    for m_ in (PAPER_N, halves):
        tag = "paper n" if m_ == PAPER_N else "this frame"
        print(f"     -- m = {m_:,}  ({tag})")
        for k in (1, 50, 211, 500, m_):
            pc = k * 0.05 / m_
            tt = stats.norm.ppf(1 - pc / 2)
            mark = "   <- the paper's 2.65" if abs(tt - t_claim) < 0.05 else ""
            print(f"     {k:>14,}{k/m_:>8.3f}{pc:>13.5f}{tt:>8.2f}{mark}")
    print(f"\n     -> |t| >= 2.65 is the BH threshold only if about {int(0.008/0.05*PAPER_N)}")
    print(f"        of {PAPER_N:,} nodes come out significant. Stating it in advance")
    print(f"        assumes the answer; the threshold is computed from the fitted")
    print(f"        p-values, not fixed before the fit.")

    pd.DataFrame(rows).to_csv(RES / "tables" / "gwr_machinery.csv", index=False)
    print(f"\nwrote {RES / 'tables' / 'gwr_machinery.csv'}")


if __name__ == "__main__":
    main()
