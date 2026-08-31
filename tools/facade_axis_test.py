"""Does the cleaned-frame street axis recover the nodes s05 loses?

s05 casts the width ray perpendicular to street_axis(nodes), which fits a
bearing from each node's `chain` neighbours. `chain` is the field that puts
both of 1st Avenue's kerbs in one street 26 m apart, so the fitted axis is
pulled off the street and the ray runs diagonally down the block instead of
across it. The signature is visible in the output: cross streets come out at
19-21 m and Park Avenue at 43.7 m, both correct, while 1st Avenue comes out at
78.6 m and loses 35 of its 44 nodes.

The colleague's cleaned frame already splits those cases -- 1st Avenue into two
branches with their own fitted axes -- and nodes.gpkg carries the labels. This
recomputes the width both ways on the same nodes, same footprints, same ray
length, and changes nothing else.

Writes nothing. Run s05 only if this says the axis is the cause.

    .venv/Scripts/python tools/facade_axis_test.py
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString
from shapely.ops import nearest_points

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
from common import PROJ_CRS, CFG, PROC, RAW, banner, street_axis
from export_svi_180 import _street_axis

G = CFG["geometry"]
HALF = G["facade_half_m"]
UTM = PROJ_CRS       # metre CRS from config; 32618 for Manhattan


def cleaned_axis(nodes):
    """One fitted bearing per cleaned street, applied to its nodes."""
    if "cleaned_street" not in nodes.columns or nodes.cleaned_street.isna().all():
        return {}
    n = nodes[nodes.cleaned_street.notna()].copy()
    utm = n.to_crs(UTM)
    n["_e"], n["_n"] = utm.geometry.x.values, utm.geometry.y.values
    out = {}
    for street, g in n.groupby("cleaned_street"):
        ax = _street_axis(g._e.to_numpy(), g._n.to_numpy())
        for nid in g.node_id:
            out[nid] = ax
    return out


def widths(m, bf, sidx, axis):
    res = {}
    for _, r in m.iterrows():
        ax = axis.get(r.node_id, np.nan)
        if not np.isfinite(ax):
            continue
        ux, uy = np.sin(np.radians(ax)), np.cos(np.radians(ax))
        px, py = -uy, ux
        pt = r.geometry
        out = []
        for sgn in (1, -1):
            ray = LineString([(pt.x, pt.y),
                              (pt.x + sgn * px * HALF, pt.y + sgn * py * HALF)])
            best = None
            for k in sidx.intersection(ray.bounds):
                g = bf.geometry.iloc[k]
                if not ray.intersects(g):
                    continue
                i = ray.intersection(g)
                if i.is_empty:
                    continue
                dd = pt.distance(nearest_points(pt, i)[1])
                best = dd if best is None or dd < best else best
            out.append(best)
        if None in out:
            continue
        w = out[0] + out[1]
        if 4 < w < 120:
            res[r.node_id] = w
    return pd.Series(res, name="w")


def main():
    banner("chain axis against cleaned-street axis")
    bf = gpd.read_file(RAW / "building_footprints.geojson").to_crs(UTM)
    hcol = "height_roof" if "height_roof" in bf.columns else "heightroof"
    bf["h_m"] = pd.to_numeric(bf[hcol], errors="coerce") * 0.3048
    bf = bf[bf.h_m.between(G["height_min_m"], G["height_max_m"])]
    sidx = bf.sindex

    m = gpd.read_file(PROC / "metrics.gpkg").to_crs(UTM)
    nodes = gpd.read_file(PROC / "nodes.gpkg")

    a_chain = street_axis(nodes)
    a_clean = cleaned_axis(nodes)
    print(f"{len(m)} nodes, {len(bf)} footprints, ray {HALF} m each side")
    print(f"cleaned axis available for {len(a_clean)} nodes\n")

    w_chain = widths(m, bf, sidx, a_chain)
    w_clean = widths(m, bf, sidx, a_clean)
    print(f"  {'':<22}{'measured':>10}{'of':>5}{'':>4}{'median width':>14}")
    print(f"  {'chain axis (current)':<22}{len(w_chain):>10}{len(m):>5}"
          f"{'':>4}{w_chain.median():>13.1f} m")
    print(f"  {'cleaned-street axis':<22}{len(w_clean):>10}{len(m):>5}"
          f"{'':>4}{w_clean.median():>13.1f} m")
    gained = set(w_clean.index) - set(w_chain.index)
    lost = set(w_chain.index) - set(w_clean.index)
    print(f"\n  recovered {len(gained)} nodes, lost {len(lost)}\n")

    j = pd.DataFrame({"chain": w_chain, "clean": w_clean})
    j = j.join(m.set_index("node_id")[["osm_name"]], how="left")
    print(f"  median width by street (a Manhattan side street is ~18-20 m,")
    print(f"  a 100 ft avenue ~30 m, Park Avenue 140 ft ~43 m)\n")
    print(f"  {'street':<22}{'chain n':>8}{'chain w':>9}{'clean n':>9}{'clean w':>9}")
    for s, g in j.groupby("osm_name"):
        c, k = g.chain.dropna(), g.clean.dropna()
        flag = ""
        if len(c) and len(k) and abs(k.median() - c.median()) > 8:
            flag = "   <--"
        print(f"  {s:<22}{len(c):>8}{c.median() if len(c) else np.nan:>9.1f}"
              f"{len(k):>9}{k.median() if len(k) else np.nan:>9.1f}{flag}")


if __name__ == "__main__":
    main()
