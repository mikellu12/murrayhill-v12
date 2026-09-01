"""Turn each field's seven-rung distribution into one number, by pruning then
taking the interpolated median.

WHY NOT THE MEAN. The rungs are ordered events, not quantities: rung 5 is a
described streetscape, not five of something. Averaging them asserts that the
gap from 2 to 3 equals the gap from 6 to 7 and that a bimodal frame -- half the
mass on 2, half on 6 -- is a 4, which is a rung the model positively rejected.
The manuscript reads the scale as ordinal and the readout has to as well.

WHY NOT PLAIN ARGMAX. It throws away everything except the winner, so a frame
at p = (.30, .28, .27, ...) reads identically to one at (.95, .02, .01, ...).
That is most of why M looked flat between streets: the readout, not the model,
was the bottleneck.

WHAT THIS DOES INSTEAD. One round of pruning, then the grouped median.

  PRUNE removes the single least-likely rung and renormalises. A seven-way
  softmax always puts some mass everywhere, and the tail rung is the one the
  model is most confident is wrong; dropping it stops that mass dragging the
  quantile. One round, not many: pruning to a single survivor is the
  elimination readout, which is a separate and more expensive procedure.

  THE INTERPOLATED MEDIAN is a quantile, so it needs only the order of the
  rungs, never the spacing between them. Rung k covers [k-0.5, k+0.5], and the
  reported value is where the cumulative distribution crosses one half:

      median = (k - 0.5) + (0.5 - F_{k-1}) / p_k

  It lands on a rung when the mass is concentrated there and between two rungs
  when the model is split, which is the behaviour argmax cannot express and the
  mean expresses wrongly.

    .venv/Scripts/python tools/sim_readout.py --table results/tables/sim_vlm_v3.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import banner

K = np.arange(1, 8)


def prune_once(P):
    """Zero the least-likely rung of each row and renormalise."""
    P = P.copy()
    P[np.arange(len(P)), P.argmin(axis=1)] = 0.0
    return P / P.sum(axis=1, keepdims=True)


def interpolated_median(P):
    """Grouped median: rung k spans [k-0.5, k+0.5]."""
    F = P.cumsum(axis=1)
    k = (F >= 0.5).argmax(axis=1)                 # first rung crossing one half
    r = np.arange(len(P))
    F_below = np.where(k > 0, F[r, np.maximum(k - 1, 0)], 0.0)
    p_k = P[r, k]
    # p_k is never 0 at the crossing rung: F only reaches 0.5 by adding mass
    return (k + 1 - 0.5) + (0.5 - F_below) / p_k


def fields_in(d):
    return sorted({c.rsplit("_p", 1)[0] for c in d.columns
                   if "_p" in c and c.rsplit("_p", 1)[-1].isdigit()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None,
                    help="default: alongside --table, suffixed _readout")
    args = ap.parse_args()
    banner("readout: prune one rung, then the interpolated median")

    d = pd.read_csv(args.table)
    fl = fields_in(d)
    print(f"{len(d)} rows, {len(fl)} fields\n")

    keep = [c for c in ("file", "street", "walk", "seq", "node_id",
                        "cardinal", "side") if c in d.columns]
    out = d[keep].copy()
    print(f"  {'field':<24}{'argmax':>9}{'readout':>9}{'diff':>8}"
          f"{'sd argmax':>11}{'sd readout':>12}")
    for f in fl:
        P = d[[f"{f}_p{k}" for k in K]].to_numpy(float)
        P = P / P.sum(axis=1, keepdims=True)
        v = interpolated_median(prune_once(P))
        out[f] = v
        am = P.argmax(axis=1) + 1
        print(f"  {f:<24}{am.mean():>9.2f}{v.mean():>9.2f}"
              f"{v.mean()-am.mean():>8.2f}{am.std():>11.2f}{v.std():>12.2f}")

    o = args.out or args.table.with_name(args.table.stem + "_readout.csv")
    out.to_csv(o, index=False)
    print(f"\nwrote {o}")

    # The complaint this readout exists to answer is that M could not tell two
    # streets apart, so the number to report is how separable the streets are.
    #
    # NOT the spread of street means on its own. That falls for 8 of 10 fields
    # here, which reads as the readout making things worse and is an artefact:
    # the readout also cuts the scatter WITHIN a street, and faster. What
    # matters is the ratio of the two -- between-street sd over the pooled
    # within-street sd -- and on that the readout gains on every field.
    if "street" in out.columns:
        def separability(v):
            g = pd.Series(v).groupby(out.street.values)
            n = g.size()
            within = np.sqrt(np.average(g.var(ddof=1).fillna(0), weights=n))
            return g.mean().std() / max(within, 1e-9)

        print(f"\n  street separability, "
              f"between-street sd over within-street sd:")
        print(f"    {'field':<24}{'argmax':>9}{'readout':>10}{'gain':>8}")
        gains = []
        for f in fl:
            P = d[[f"{f}_p{k}" for k in K]].to_numpy(float)
            P = P / P.sum(axis=1, keepdims=True)
            a, b = separability(P.argmax(axis=1) + 1.0), separability(out[f])
            gains.append(b / max(a, 1e-9))
            print(f"    {f:<24}{a:>9.3f}{b:>10.3f}{gains[-1]:>7.2f}x")
        g = np.array(gains)
        print(f"\n    median gain {np.median(g):.2f}x, "
              f"improved on {(g > 1).sum()}/{len(g)} fields")


if __name__ == "__main__":
    main()
