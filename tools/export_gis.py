"""
Build the layered stack -- GeoPackage for QGIS2threejs, JSON for Blender.

WHY THIS EXISTS
---------------
Everything downstream of it is generic. Hand it a CSV of nodes carrying any
set of metric columns and it produces one plane per metric, bottom to top,
with the built fabric underneath. Two metrics make two planes; adding a
third column later adds a third plane and needs no edit here.

    .venv/bin/python tools/export_gis.py                       # repo metrics
    .venv/bin/python tools/export_gis.py --csv mine.csv --metrics GVI VEI dwell

GEOMETRY, AND THE TRAP UNDER IT
-------------------------------
A CSV carrying lat/lon is used as-is. A CSV carrying only node_id has to be
joined to a frame, and node ids in this project have been positional --
`n00042` means a different street corner in every frame that was ever built.
So the join is checked and reported, and refuses to run below a coverage
floor rather than quietly drawing a tenth of the city. Prefer lat/lon.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import LineString

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from common import CFG, PROC, RAW                                # noqa: E402

CRS = "EPSG:32618"
RESERVED = {"node_id", "lat", "lon", "geometry", "osm_name", "typology",
            "chain", "chain_pos_m", "face_id", "northing_m", "easting_m",
            "street_axis_deg", "H_m", "W_facade"}


def check_coords(df):
    """Sanity-check lat/lon before anything is drawn from them.

    Carrying coordinates in the CSV avoids the positional-node_id trap, but
    it has a failure mode of its own: swapped columns parse perfectly and
    silently relocate the study area. In New York lon is about -74 and lat
    about +40, so a swap is unambiguous and worth refusing outright.
    """
    lat, lon = pd.to_numeric(df.lat, errors="coerce"), pd.to_numeric(df.lon, errors="coerce")
    bad = int(lat.isna().sum() + lon.isna().sum())
    if bad:
        sys.exit(f"ERROR: {bad} lat/lon values are not numeric.")
    if lat.abs().max() > 90:
        sys.exit("ERROR: lat outside [-90, 90] -- lat and lon look swapped.")
    if lon.abs().max() > 180:
        sys.exit("ERROR: lon outside [-180, 180].")
    if lat.median() < 0 and lon.median() > 0:
        sys.exit("ERROR: lat/lon look swapped (lat negative, lon positive).")
    dup = int(df.duplicated(["lat", "lon"]).sum())
    if dup:
        print(f"  note: {dup} rows share coordinates with another row; "
              f"their bars will overlap exactly.")
    return lat, lon


def load_nodes(csv, frame):
    """Return a GeoDataFrame of nodes in CRS, however the caller keyed them."""
    if csv is None:
        g = gpd.read_file(PROC / "metrics.gpkg").to_crs(CRS)
        print(f"nodes: {len(g)} from metrics.gpkg")
        return g
    df = pd.read_csv(csv)
    if {"lat", "lon"} <= set(df.columns):
        check_coords(df)
        g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat),
                             crs="EPSG:4326").to_crs(CRS)
        print(f"nodes: {len(g)} from {csv} (lat/lon in file)")
        return g
    if "node_id" not in df.columns:
        sys.exit("ERROR: csv needs lat/lon columns, or node_id to join on.")
    fr = gpd.read_file(frame or PROC / "nodes.gpkg").to_crs(CRS)
    g = fr[["node_id", "geometry"]].merge(df, on="node_id", how="inner")
    cov = len(g) / max(len(df), 1)
    print(f"nodes: joined {len(g)} of {len(df)} csv rows to "
          f"{frame or 'data/processed/nodes.gpkg'} by node_id ({cov:.0%})")
    if cov < 0.999:
        miss = sorted(set(df.node_id) - set(fr.node_id))[:5]
        print(f"  WARNING {len(df) - len(g)} rows had no match, e.g. {miss}")
        print("  node ids are positional in this project -- a partial match "
              "usually means the csv came from a different frame, not that "
              "rows are missing. Check before trusting the drawing.")
    if cov < CFG["diagram"]["min_layer_coverage"]:
        sys.exit("ERROR: join coverage below diagram.min_layer_coverage.")
    return gpd.GeoDataFrame(g, geometry="geometry", crs=CRS)


def pick_metrics(g, asked):
    if asked is not None:
        missing = [m for m in asked if m not in g.columns]
        if missing:
            sys.exit(f"ERROR: columns not in input: {missing}")
        return asked
    auto = [c for c in g.columns
            if c not in RESERVED and pd.api.types.is_numeric_dtype(g[c])]
    print(f"metrics: auto-detected {auto}")
    return auto


def build_pairs(m, pairs):
    """Reparameterise opposed view pairs into level and asymmetry.

    Two 180-degree views that face opposite ways tile the full circle, so
    their mean IS the 360-degree value -- verified on this repo's own
    directional metrics to 3.6e-15 for GVI. All the information the pair
    adds over full360 sits in the difference. Drawing or modelling them as
    two independent layers would double the apparent sample on a quantity
    that is algebraically determined by the other half.

    A is taken as the positive direction. Order the pair so A is the uptown
    (grid bearing) or grid-east member on EVERY street, or the asymmetries
    cancel when averaged across streets.
    """
    specs, used = [], set()
    for spec in pairs or []:
        if len(spec) < 2:
            sys.exit(f"ERROR: --pair needs two columns, got {spec}")
        a, b = spec[0], spec[1]
        name = spec[2] if len(spec) > 2 else f"{a}_vs_{b}"
        for c in (a, b):
            if c not in m.columns:
                sys.exit(f"ERROR: --pair column not in input: {c}")
        used |= {a, b}
        lvl, asy = f"{name}_level", f"{name}_asym"
        m[lvl] = (m[a] + m[b]) / 2.0
        m[asy] = m[a] - m[b]
        specs += [(lvl, False), (asy, True)]
        print(f"pair {name}: level=({a}+{b})/2, asymmetry={a}-{b} "
              f"(positive means {a})")
        d = m[asy].dropna()
        print(f"  asymmetry: mean {d.mean():+.3f}  sd {d.std():.3f}  "
              f"range {d.min():+.3f}..{d.max():+.3f}")
        # A sign convention that flips between streets shows up as street
        # means of opposite sign around a near-zero pooled mean.
        if "osm_name" in m.columns:
            g = m.groupby("osm_name")[asy].mean()
            pos, neg = (g > 0).sum(), (g < 0).sum()
            if pos and neg:
                print(f"  NOTE {pos} streets have positive mean asymmetry and "
                      f"{neg} negative. If the pair is not ordered by a single "
                      f"axis on every street, these are cancelling and the "
                      f"pooled mean is meaningless.")
    return specs, used


def block_face_lines(m):
    if "face_id" not in m.columns or "chain_pos_m" not in m.columns:
        return gpd.GeoDataFrame({"face_id": [], "GVI": []},
                                geometry=[], crs=CRS)
    rows = []
    for fid, g in m.dropna(subset=["face_id"]).groupby("face_id"):
        g = g.sort_values("chain_pos_m")
        if len(g) < 2:
            continue
        rows.append({"face_id": str(fid), "n_nodes": len(g),
                     "geometry": LineString(list(zip(g.geometry.x, g.geometry.y)))})
    return gpd.GeoDataFrame(rows, crs=CRS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None,
                    help="node table; needs lat/lon, or node_id to join")
    ap.add_argument("--frame", default=None,
                    help="frame gpkg to join node_id against")
    ap.add_argument("--pair", nargs="+", action="append", default=None,
                    metavar="A B [NAME]",
                    help="two opposed 180-deg views of the same node. Drawn "
                         "as level=(A+B)/2 and asymmetry=A-B, never as two "
                         "planes: opposed halves tile the circle, so they are "
                         "one measurement plus a contrast, not two samples. "
                         "Repeatable.")
    ap.add_argument("--metrics", nargs="*", default=None,
                    help="columns to draw, bottom to top. default: all numeric")
    ap.add_argument("--out", default="results/gis/murrayhill_layers.gpkg")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    D = CFG["diagram"]; gap = D["layer_gap_m"]
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    m = load_nodes(a.csv, a.frame)
    pair_specs, pair_used = build_pairs(m, a.pair)
    base = [c for c in pick_metrics(m, a.metrics)
            if c not in pair_used and not c.endswith(("_level", "_asym"))]
    specs = [(c, False) for c in base] + pair_specs
    metrics = [c for c, _ in specs]
    if not metrics:
        sys.exit("ERROR: no numeric metric columns found.")

    fp = gpd.read_file(RAW / "building_footprints.geojson").to_crs(CRS)
    fb, nb = fp.total_bounds, m.total_bounds
    inside = ((m.geometry.x.between(fb[0], fb[2]))
              & (m.geometry.y.between(fb[1], fb[3]))).mean()
    if inside < 0.5:
        print(f"  WARNING only {inside:.0%} of nodes fall inside the building "
              f"footprint extent. Nodes span x {nb[0]:.0f}..{nb[2]:.0f}, "
              f"footprints span {fb[0]:.0f}..{fb[2]:.0f}. Wrong study area, "
              f"wrong CRS, or swapped coordinates.")
    fp["height_m"] = pd.to_numeric(fp.get("height_roof"), errors="coerce") * 0.3048
    fp["height_m"] = fp.height_m.fillna(fp.height_m.median())
    fp["extrude_m"] = fp.height_m * D.get("z_exaggeration", 1.0)
    fp["z_m"] = 0.0
    fabric = fp[[c for c in ["bin", "height_m", "extrude_m", "z_m", "geometry"]
                 if c in fp.columns]]
    faces = block_face_lines(m); faces["z_m"] = float(gap)

    B = m.total_bounds
    cx, cy = (B[0] + B[2]) / 2, (B[1] + B[3]) / 2

    if out.exists():
        out.unlink()
    fabric.to_file(out, layer="fabric", driver="GPKG")
    if len(faces):
        faces.to_file(out, layer="faces", driver="GPKG")
    print(f"  fabric     {len(fabric):5d}   z =     0")
    print(f"  faces      {len(faces):5d}   z = {gap:5.0f}")

    layers, z = [], gap
    for i, (col, diverging) in enumerate(specs):
        z = gap * (i + 2)
        sub = m.dropna(subset=[col])
        cov = len(sub) / max(len(m), 1)
        if cov < D["min_layer_coverage"]:
            print(f"  SKIP {col}: only {len(sub)} of {len(m)} nodes "
                  f"({cov:.0%}) carry a value; below min_layer_coverage.")
            continue
        gl = sub.copy(); gl["z_m"] = float(z)
        keep = [c for c in ["node_id", "osm_name", "typology", col, "z_m",
                            "geometry"] if c in gl.columns]
        gl[keep].to_file(out, layer=col, driver="GPKG")
        v = sub[col].astype(float)
        lo, hi = float(v.min()), float(v.max())
        if diverging:
            # Signed, symmetric about zero: a bar's direction carries the
            # sign and zero asymmetry sits flat on the plane.
            scale = max(abs(lo), abs(hi), 1e-9)
            norm = lambda x: round(x / scale, 4)
        else:
            rng = max(hi - lo, 1e-9)
            norm = lambda x: round((x - lo) / rng, 4)
        meta = D.get("labels", {}).get(col, {})
        layers.append({
            "key": col, "z": z, "diverging": bool(diverging),
            "title": meta.get("title", col.replace("_", " ").upper()),
            "unit": meta.get("unit", ""), "lo": lo, "hi": hi,
            "coverage": round(cov, 4),
            "nodes": [{"x": round(r.geometry.x - cx, 2),
                       "y": round(r.geometry.y - cy, 2),
                       "v": norm(getattr(r, col))}
                      for r in sub.itertuples()]})
        print(f"  {col:<10} {len(sub):5d}   z = {z:5.0f}   "
              f"range {lo:.2f}..{hi:.2f}   coverage {cov:.0%}")

    if not layers:
        sys.exit("ERROR: every metric fell below min_layer_coverage.")

    def rings(gdf):
        r = []
        for geom in gdf.geometry:
            gs = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
            for p in gs:
                r.append([[round(x - cx, 2), round(y - cy, 2)]
                          for x, y in p.exterior.coords])
        return r

    doc = {"crs": CRS, "accent": D["accent"],
           "bar_height_m": D["bar_height_m"], "bar_width_m": D["bar_width_m"],
           "extent_m": [round(B[2] - B[0], 1), round(B[3] - B[1], 1)],
           "faces_z": gap, "n_nodes": len(m),
           "fabric": [{"ring": rg, "h": round(float(h), 1)}
                      for rg, h in zip(rings(fabric), fabric.height_m)],
           "faces": [[[round(x - cx, 2), round(y - cy, 2)] for x, y in g.coords]
                     for g in faces.geometry],
           "layers": layers}
    jp = Path(a.json) if a.json else out.with_suffix(".json")
    jp.write_text(json.dumps(doc))
    print(f"\nwrote {out}\nwrote {jp}  ({jp.stat().st_size/1e6:.1f} MB, "
          f"{len(layers)} value layers)")


if __name__ == "__main__":
    main()
