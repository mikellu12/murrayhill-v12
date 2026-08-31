"""Exploded axonometric of the SIM layers over the built fabric.

Six strata over Murray Hill: extruded footprints at the bottom, then eye-level
greenery, G, M, P and the composite. Bar height is the value at each 20 m node,
red marks the upper tail.

COORDINATES COME FROM AN EXACT JOIN, not a positional key. sim_index.csv
carries `node_id`, and merging it to nodes.gpkg on that string matches 634 of
634 rows. An earlier rendering of this figure carried a provisional banner
warning that node positions "assume a 0-based positional key into the 766-node
mapping frame" and that a 1-based reading would shift every value one node
along its street -- that ambiguity only exists if node_id is discarded and rows
are indexed by position. It is not discarded here.

The off-by-one is easy to believe because sim_index starts at n00001 while
nodes.gpkg starts at n00000: n00000 is filtered out upstream, so a positional
read appears to line up. It does not.

The greenery layer is measured GVI from metrics.csv, which joins to sim_index
on node_id at 100 per cent. `green_eye` is a SIM term, not GVI, and the two
differ by more than a name -- green_eye has a median share of 0.0009.

Matplotlib rather than a modelling package on purpose: this regenerates from
sim_index.csv, metrics.csv, nodes.gpkg and the footprint file in seconds, and
stays correct when the data changes. tools/sim_axonometric_blender.py exports
the same geometry for a rendered version.

    .venv/Scripts/python tools/sim_axonometric.py
    .venv/Scripts/python tools/sim_axonometric.py --layers GVI G M P SIM
"""
import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import PROC, RAW, RES, banner

UTM = 32618
# Vertical gap between strata, in metres of the same UTM space the plan uses,
# so the explosion reads at the same scale as the neighbourhood.
GAP = 620.0
BAR = 150.0          # tallest bar in a layer
TAIL = 0.80          # quantile above which a bar is drawn red
GREY, RED = "#9a9a9a", "#a83a32"


def _footprints(nodes):
    p = RAW / "building_footprints.geojson"
    if not p.exists():
        return None
    f = gpd.read_file(p).to_crs(UTM)
    f["h"] = pd.to_numeric(f.height_roof, errors="coerce").fillna(0.0)
    # A 1,401 ft outlier is a survey error, not a building; clip so one bad
    # row does not set the vertical scale for the whole plate.
    f["h"] = f.h.clip(0, f.h.quantile(0.995)) * 0.3048
    return f.clip(nodes.total_bounds_poly) if hasattr(nodes, "total_bounds_poly") else f


def _prisms(f, z0, scale):
    """Extruded footprints as one Poly3DCollection: walls plus roof."""
    polys = []
    for geom, h in zip(f.geometry, f.h):
        gs = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        for g in gs:
            x, y = np.array(g.exterior.coords).T
            top = z0 + h * scale
            for i in range(len(x) - 1):
                polys.append([(x[i], y[i], z0), (x[i + 1], y[i + 1], z0),
                              (x[i + 1], y[i + 1], top), (x[i], y[i], top)])
            polys.append([(a, b, top) for a, b in zip(x, y)])
    return polys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", nargs="+",
                    default=["GVI", "G", "M", "P", "SIM"],
                    help="bottom to top; any column of the joined table")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--nodes", type=Path, default=PROC / "nodes.gpkg",
                    help="coordinate source. A file without node_id (e.g. "
                         "final_nodes_cleaned.gpkg) is matched spatially, and "
                         "any sim_index node it does not contain is dropped.")
    ap.add_argument("--source", default="pixel", choices=["pixel", "vlm"],
                    help="pixel reads sim_index.csv (G/M/P/SIM); vlm reads "
                         "vlm_calculations.csv (I_raw/Y/D_raw/M), pooling the "
                         "four half-views of each node")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--elev", type=float, default=28.0)
    ap.add_argument("--azim", type=float, default=-72.0)
    args = ap.parse_args()
    if args.out is None:
        args.out = RES / "figures" / (
            "figure_axonometric_vlm.png" if args.source == "vlm"
            else "figure_axonometric_sim.png")
    banner(f"exploded axonometric, {args.source} layers")

    nodes = gpd.read_file(PROC / "nodes.gpkg").to_crs(UTM)
    alt = None
    if args.nodes.resolve() != (PROC / "nodes.gpkg").resolve():
        alt = gpd.read_file(args.nodes).to_crs(UTM)
        if "original_id" in alt.columns:
            alt = alt.drop_duplicates("original_id")
        print(f"coordinate source: {args.nodes.name} ({len(alt)} locations)")
    if args.source == "vlm":
        # Four half-views per node -- two walks by two sides -- and the
        # strata are drawn per node, so they pool here. Mean, not median:
        # the layers are meant to show where a dimension is high, and the
        # median of four discards the pair that differ.
        v = pd.read_csv(RES / "tables" / "vlm_calculations.csv")
        sim = (v.groupby("node_id")[["I_raw", "Y", "D_raw", "M"]]
                 .mean().reset_index())
        print(f"{len(v)} half-views pooled to {len(sim)} nodes")
        if args.layers == ["GVI", "G", "M", "P", "SIM"]:
            args.layers = ["I_raw", "Y", "D_raw", "M"]
    else:
        sim = pd.read_csv(PROC / "sim_index.csv")
    met = pd.read_csv(PROC / "metrics.csv")[["node_id", "GVI", "VEI"]]

    # The join the banner was worried about. On node_id, exactly.
    met = met.drop(columns=[c for c in met.columns
                            if c != "node_id" and c in sim.columns])
    d = sim.merge(met, on="node_id", how="left").merge(
        nodes[["node_id", "geometry"]], on="node_id", how="left")
    miss = d.geometry.isna().sum()
    print(f"{len(d)} rows; matched to a coordinate: {len(d) - miss}, unmatched: {miss}")
    if miss:
        sys.exit("unmatched node_id -- refusing to draw a figure with guessed positions")
    # A layer with no value at a node would draw a zero-height bar, which
    # reads as a measured minimum rather than an absence. Drop those rows.
    need = [c for c in args.layers if c in d.columns]
    before = len(d)
    d = d.dropna(subset=need).reset_index(drop=True)
    if len(d) < before:
        print(f"  dropped {before - len(d)} node(s) missing a layer value")
    d = gpd.GeoDataFrame(d, geometry="geometry", crs=UTM)
    if alt is not None:
        # The alternative frame carries no node_id, so membership is decided
        # by position. The two agree to 0.00 m where they overlap, so this is
        # a filter rather than a re-registration -- nothing moves.
        ax_ = np.column_stack([alt.geometry.x, alt.geometry.y])
        dx = np.column_stack([d.geometry.x, d.geometry.y])
        near = np.sqrt(((dx[:, None] - ax_[None]) ** 2).sum(-1)).min(1)
        keep = near < 1.0
        print(f"  in {args.nodes.name}: {int(keep.sum())} of {len(d)}"
              f"   dropped {int((~keep).sum())}")
        d = d[keep].reset_index(drop=True)
    x, y = d.geometry.x.to_numpy(), d.geometry.y.to_numpy()

    layers = [c for c in args.layers if c in d.columns]
    if len(layers) != len(args.layers):
        print("skipping absent columns:", set(args.layers) - set(layers))

    fp = _footprints(nodes)
    fig = plt.figure(figsize=(11, 15))
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

    if fp is not None:
        ax.add_collection3d(Poly3DCollection(
            _prisms(fp, 0.0, 1.0), facecolors="white", edgecolors="#6f6f6f",
            linewidths=0.25, alpha=1.0))
        print(f"{len(fp)} footprints")

    xr = x.max() - x.min()
    for i, col in enumerate(layers):
        z0 = GAP * (i + 1) + 120.0
        v = pd.to_numeric(d[col], errors="coerce").to_numpy(dtype=float)
        lo, hi = np.nanmin(v), np.nanmax(v)
        rng = hi - lo if hi > lo else 1.0
        hgt = (v - lo) / rng * BAR
        cut = np.nanquantile(v, TAIL)

        # A dotted plane under each stratum so the layer reads as a sheet
        # even where its bars are near zero -- which is the whole point of
        # the greenery layer.
        ax.scatter(x, y, np.full_like(x, z0), s=1.2, c="#c8c8c8", depthshade=False)
        segs = [[(xi, yi, z0), (xi, yi, z0 + h)] for xi, yi, h in zip(x, y, hgt)]
        cols = [RED if vi >= cut else GREY for vi in v]
        ax.add_collection3d(Line3DCollection(segs, colors=cols, linewidths=1.5))
        ax.text(x.min() - xr * 0.42, y.mean(), z0 + BAR * 0.75, col,
                fontsize=15, color="#333333", zdir=None)
        ax.text(x.min() - xr * 0.42, y.mean(), z0 + BAR * 0.30,
                f"{col}  {lo:.2f}–{hi:.2f}", fontsize=9, color="#666666", zdir=None)
        print(f"  {col:<5} {lo:.3f} – {hi:.3f}   red above {cut:.3f}")

    if fp is not None:
        ax.text(x.min() - xr * 0.42, y.mean(), 150, "BUILT FABRIC",
                fontsize=15, color="#333333")
        ax.text(x.min() - xr * 0.42, y.mean(), 60,
                f"{len(fp)} footprints", fontsize=9, color="#666666")

    ax.set_xlim(x.min() - xr * 0.45, x.max() + xr * 0.05)
    ax.set_ylim(y.min(), y.max())
    ax.set_zlim(0, GAP * (len(layers) + 1) + 400)
    ax.set_box_aspect((1, 1, 1.5))
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.set_axis_off()
    ax.legend(handles=[Line2D([], [], color=RED, lw=2,
                              label=f"upper {100*(1-TAIL):.0f}% of each layer")],
              loc="lower right", frameon=False, fontsize=9)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
