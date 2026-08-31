"""Score the 90-degree half ratings against the GVI of that same 90 degrees.

The halves looked far worse than the 180-degree view -- greenery rho +0.45
against +0.70 -- but the comparison was not fair. Both were scored against
`metrics.csv` GVI, which is a per-node quantity over the whole forward view.
A half facing a blank wall while the trees stand on the other side of the
street should rate 1, and scoring it against a GVI that counted those trees
marks a correct answer wrong.

The azimuthal profiles carry the per-bearing data, so the fair target exists:
slice_metrics(profile, bearing +/- 45, 90) is the vegetation share over
exactly the arc each half was rendered from. This rescores the ratings we
already have against that, and reports both targets side by side.

If the halves recover against their own arc, they were never broken and the
validation was; if they stay flat, the 180-degree view genuinely carries
signal the halves do not.

No GPU and no model calls -- this only re-scores tables already on disk.

    .venv/Scripts/python tools/half_target.py
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
from common import PROJ_CRS, PROC, RES, banner, slice_metrics
from export_svi_180 import _street_axis, _walks

UTM = PROJ_CRS       # metre CRS from config; 32618 for Manhattan
HALF_FOV, OFFSET = 90.0, 45.0
SIDE_OFF = {"L": -OFFSET, "R": +OFFSET}


def walk_bearings():
    """(street, walk) -> bearing, rebuilt the way the export did it."""
    nodes = gpd.read_file(PROC / "nodes.gpkg")
    if "cleaned_street" in nodes.columns and nodes.cleaned_street.notna().any():
        nodes = nodes[nodes.in_cleaned & nodes.cleaned_street.notna()].copy()
        nodes["folder"] = nodes.cleaned_street
    else:
        nodes["folder"] = nodes.chain.str.split("#").str[0]
    utm = nodes.to_crs(UTM)
    nodes["_e"], nodes["_n"] = utm.geometry.x.values, utm.geometry.y.values
    out = {}
    for street, g in nodes.groupby("folder"):
        axis = _street_axis(g._e.to_numpy(), g._n.to_numpy())
        for bearing, walk in _walks(axis):
            out[(street, walk)] = bearing
    return out


def add_arc_target(d):
    """Attach GVI/VEI over each row's own 90-degree arc."""
    z = np.load(PROC / "azimuth_profiles.npz")
    prof = {k: z[k] for k in z.files}
    bear = walk_bearings()

    # the probe tables carry `file` as street/walk/name.jpg
    parts = d.file.str.split("/", expand=True)
    d = d.assign(street=parts[0], walk=parts[1])
    gvi, vei, miss = [], [], 0
    for r in d.itertuples():
        p = prof.get(r.node_id)
        b = bear.get((r.street, r.walk))
        if p is None or b is None:
            gvi.append(np.nan); vei.append(np.nan); miss += 1
            continue
        # "W" is a whole forward view: its own arc is the full 180 on the
        # walk bearing, not a 90-degree half off it.
        if r.side == "W":
            g, v = slice_metrics(p, b % 360, 180.0)
        elif r.side in SIDE_OFF:
            g, v = slice_metrics(p, (b + SIDE_OFF[r.side]) % 360, HALF_FOV)
        else:
            gvi.append(np.nan); vei.append(np.nan); miss += 1
            continue
        gvi.append(g); vei.append(v)
    d["GVI_arc"], d["VEI_arc"] = gvi, vei
    if miss:
        print(f"  {miss} row(s) without a profile or bearing")
    return d


def rho(d, x, y):
    s = d[[x, y]].dropna()
    if len(s) < 12 or s[x].nunique() < 2:
        return np.nan, len(s)
    return s[x].corr(s[y], method="spearman"), len(s)


def report(d, label):
    print(f"\n  {label}  ({len(d)} rows)")
    print(f"    {'field':<22}{'vs node GVI':>14}{'vs its own arc':>17}{'n':>7}")
    for f, node_t, arc_t in [("vertical_greenery", "GVI", "GVI_arc"),
                             ("green_eye_level", "GVI", "GVI_arc"),
                             ("green_softening", "GVI", "GVI_arc"),
                             ("enclosure", "VEI", "VEI_arc"),
                             ("vertical_hardscape", "VEI", "VEI_arc")]:
        if f not in d.columns or d[f].dropna().empty:
            continue
        a, n1 = rho(d, f, node_t)
        b, n2 = rho(d, f, arc_t)
        mark = ""
        if not np.isnan(a) and not np.isnan(b) and abs(b) - abs(a) > 0.10:
            mark = "   <-- recovers on its own arc"
        print(f"    {f:<22}{a:>+14.3f}{b:>+17.3f}{min(n1, n2):>7}{mark}")


def main():
    banner("do the halves recover when scored against their own 90 degrees?")
    met = pd.read_csv(PROC / "metrics.csv")[["node_id", "GVI", "VEI"]]

    partial = RES / "tables" / "svi_90_sim.partial_fullschema.csv"
    if partial.exists():
        d = pd.read_csv(partial)
        d = d.drop(columns=[c for c in ("GVI", "VEI") if c in d.columns])
        d = d.merge(met, on="node_id", how="left")
        report(add_arc_target(d), "full schema, the run that was stopped")

    probe = RES / "tables" / "prompt_probe.csv"
    if probe.exists():
        p = pd.read_csv(probe)
        p = p.drop(columns=[c for c in ("GVI", "VEI") if c in p.columns])
        p = p.merge(met, on="node_id", how="left")
        p = add_arc_target(p)
        for v in p.variant.unique():
            report(p[p.variant == v], f"probe, {v} prompt")

    w = RES / "tables" / "prompt_probe_180.csv"
    if w.exists():
        p = pd.read_csv(w)
        p = p.drop(columns=[c for c in ("GVI", "VEI") if c in p.columns])
        p = p.merge(met, on="node_id", how="left")
        p = add_arc_target(p)
        for v in p.variant.unique():
            report(p[p.variant == v], f"180-degree whole view, {v} prompt")

    ns = RES / "tables" / "prompt_probe_90_noside.csv"
    if ns.exists():
        p = pd.read_csv(ns)
        p = p.drop(columns=[c for c in ("GVI", "VEI") if c in p.columns])
        p = p.merge(met, on="node_id", how="left")
        report(add_arc_target(p), "probe, solo prompt, no preamble")

    print("\n  180-degree solo, for reference: greenery +0.701, "
          "green_eye_level +0.768")


if __name__ == "__main__":
    main()
