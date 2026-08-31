"""Is a low-confidence field measuring the street, or rolling dice?

facade_variation puts only 0.226 of its probability on the digit it emits,
against 0.528 for vertical_greenery. A near-uniform distribution means the
model is unsure about that image -- it does not follow that the field carries
no information, and the difference is testable two ways without any measured
counterpart:

  SAME FRONTAGE, OPPOSITE APPROACH. Walking north, the left half faces the
  west side of the street; walking south, the right half faces the same west
  side. Those two ratings are of one physical frontage, taken from opposite
  directions in different light. A field that is decided at random cannot
  agree with itself across that pair. A field that is measuring the frontage
  should.

  SPATIAL STRUCTURE. Noise has no geography. If neighbouring nodes rate alike
  more than distant ones do -- positive Moran's I -- the field is tracking
  something that varies over space, which random selection cannot produce.

Neither test needs a pixel counterpart, which is the point: the four
judgement-only fields have none, and this is the evidence available for them.

    .venv/Scripts/python tools/rating_reliability.py
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import PROJ_CRS, PROC, RES, banner
from sim_fields import FIELDS

RNG = np.random.default_rng(0)


def morans_i(v, xy, band=120.0):
    """Row-standardised inverse-distance weights within `band` metres."""
    ok = np.isfinite(v)
    v, xy = v[ok], xy[ok]
    z = v - v.mean()
    d = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
    W = np.where((d > 0) & (d <= band), 1.0 / np.maximum(d, 1.0), 0.0)
    rs = W.sum(1, keepdims=True)
    W = np.divide(W, rs, out=np.zeros_like(W), where=rs > 0)
    num = (W * np.outer(z, z)).sum()
    return float(len(z) * num / (W.sum() * (z ** 2).sum()))


def main():
    banner("is a low-confidence field random?")
    d = pd.read_csv(RES / "tables" / "sim_vlm.csv")
    g = gpd.read_file(PROC / "nodes.gpkg").to_crs(PROJ_CRS)[["node_id", "geometry"]]

    # walk bearing decides which compass side L and R face. Two walks per
    # street, so for each node exactly one L and one R look at the same side.
    card = {"N": 0, "E": 90, "S": 180, "W": 270}
    d["facing"] = [(card.get(c, 0) + (-90 if s == "L" else 90)) % 360
                   for c, s in zip(d.cardinal, d.side)]

    print(f"  {'field':<22}{'same frontage':>14}{'n pairs':>9}"
          f"{'Moran I':>10}{'confidence':>12}")
    logit = pd.read_csv(RES / "tables" / "rating_logits.csv")
    node = d.groupby("node_id")
    rows = []
    for f in FIELDS:
        # pair the two half-views of each node that face the same way
        pairs = []
        for nid, grp in node:
            for facing, gg in grp.groupby("facing"):
                v = gg[f].dropna().to_numpy()
                if len(v) == 2:
                    pairs.append(v)
        pairs = np.array(pairs) if pairs else np.empty((0, 2))
        r = (pd.Series(pairs[:, 0]).corr(pd.Series(pairs[:, 1]), method="spearman")
             if len(pairs) > 20 else np.nan)

        nd = d.groupby("node_id")[f].mean().reset_index().merge(g, on="node_id")
        nd = gpd.GeoDataFrame(nd, geometry="geometry")
        mi = morans_i(nd[f].to_numpy(float),
                      np.c_[nd.geometry.x, nd.geometry.y])
        conf = (logit[f + "_top"].mean() if f + "_top" in logit.columns
                else np.nan)
        cs = f"{conf:.3f}" if np.isfinite(conf) else "--"
        print(f"  {f:<22}{r:>+14.3f}{len(pairs):>9}{mi:>+10.3f}{cs:>12}")
        rows.append({"field": f, "same_frontage_rho": r, "n_pairs": len(pairs),
                     "morans_i": mi, "top_p": conf})

    print("\n  a field decided at random would score 0 on both:")
    print("    no agreement between two views of one frontage,")
    print("    no tendency for neighbouring nodes to rate alike.")
    print("\n  shuffled control, same values in random spatial order:")
    for f in ["facade_variation", "vertical_greenery"]:
        nd = d.groupby("node_id")[f].mean().reset_index().merge(g, on="node_id")
        nd = gpd.GeoDataFrame(nd, geometry="geometry")
        v = nd[f].to_numpy(float)
        xy = np.c_[nd.geometry.x, nd.geometry.y]
        sh = [morans_i(RNG.permutation(v), xy) for _ in range(20)]
        print(f"    {f:<22}real {morans_i(v, xy):+.3f}   "
              f"shuffled {np.mean(sh):+.3f} +/- {np.std(sh):.3f}")
    pd.DataFrame(rows).to_csv(RES / "tables" / "rating_reliability.csv",
                              index=False)


if __name__ == "__main__":
    main()
