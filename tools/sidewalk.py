"""
Sidewalk width and building setback vs directional greenness.

THE HYPOTHESIS
--------------
Street trees need planting space. Facade-to-facade width did not predict
GVI (rho = 0.08), but that measure includes the roadway, which is where
trees cannot go. Sidewalk width is the physically relevant quantity, and
setback -- the gap between the legal right-of-way line and the building
face -- is where forecourt planting and street walls sit.

If sidewalk width predicts GVI where street width did not, greenness is
constrained by planting space rather than by street proportions. That is
a substantive finding and a straightforward one to state.

METHOD
------
The same perpendicular transect used for facade width, but recording more
along the way. From each node, on each side:

    curb distance      node -> first sidewalk polygon edge
    facade distance    node -> first building face
    sidewalk width     the sidewalk polygon's extent along the transect
    setback            facade distance - outer sidewalk edge

Setback is the residual: where a building sits back from the walkable
edge, that gap is forecourt, areaway or plaza.

DIRECTIONAL JOIN
----------------
A pedestrian walks on ONE sidewalk, so left and right are kept separate
rather than summed. For each travel direction the left side is the one at
bearing - 90 and the right at bearing + 90; both are in the 180-degree
forward view, but they are different walking experiences.

CPU only. No GPU, no imagery, no model. Runs on a laptop in seconds once
the pipeline outputs exist.
"""
import sys
import numpy as np, pandas as pd, geopandas as gpd, requests
from pathlib import Path
from shapely.geometry import LineString, shape
from shapely.ops import nearest_points, unary_union

HALF_M = 60.0          # transect half-length
# Candidate Socrata datasets. IDs get retired, so probe and report rather
# than assume; the field list is printed so a schema change is visible.
SIDEWALK_DATASETS = ["vfx9-tbb6", "b6gu-cbhh", "2bnn-yakx"]
FOOTPRINT_DATASETS = ["5zhs-2jue", "qb5r-6dgf", "nqwf-w8eh"]


def socrata_bbox(datasets, bounds, path, label):
    """Download the first dataset that answers, clipped to a bbox."""
    if Path(path).exists():
        print(f"{label}: cached")
        return gpd.read_file(path)
    pad = 0.002
    where = (f"within_box(the_geom, {bounds[3]+pad}, {bounds[0]-pad}, "
             f"{bounds[1]-pad}, {bounds[2]+pad})")
    for ds in datasets:
        url = f"https://data.cityofnewyork.us/resource/{ds}.json"
        try:
            t = requests.get(url, params={"$limit": 1}, timeout=30)
            if t.status_code != 200 or not t.json():
                print(f"  {ds}: HTTP {t.status_code}")
                continue
            if "the_geom" not in t.json()[0]:
                print(f"  {ds}: no geometry field "
                      f"({sorted(t.json()[0])[:6]}...)")
                continue
        except Exception as e:
            print(f"  {ds}: {e}")
            continue
        print(f"{label}: dataset {ds}, fields "
              f"{sorted(t.json()[0])[:8]}...")
        rows, off = [], 0
        while True:
            r = requests.get(url, params={"$where": where, "$limit": 5000,
                                          "$offset": off}, timeout=90)
            r.raise_for_status()
            ch = r.json()
            if not ch:
                break
            rows += ch
            off += 5000
        if not rows:
            print(f"  {ds}: no features in bbox")
            continue
        g = gpd.GeoDataFrame(pd.DataFrame(rows),
                             geometry=[shape(x["the_geom"]) for x in rows],
                             crs=4326)
        g.to_file(path, driver="GeoJSON")
        print(f"  {len(g)} features -> {path}")
        return g
    print(f"{label}: no dataset available. Download manually from "
          f"data.cityofnewyork.us and save to {path}")
    return None


def transect_profile(pt, axis_deg, sidewalks, buildings, sw_idx, bd_idx):
    """Per-side distances along a perpendicular transect."""
    if np.isnan(axis_deg):
        return {}
    ux, uy = np.sin(np.radians(axis_deg)), np.cos(np.radians(axis_deg))
    px, py = -uy, ux                       # perpendicular to the street
    out = {}
    for sgn, side in ((1, "L"), (-1, "R")):
        ray = LineString([(pt.x, pt.y),
                          (pt.x + sgn * px * HALF_M, pt.y + sgn * py * HALF_M)])

        facade = None
        for k in bd_idx.intersection(ray.bounds):
            g = buildings.geometry.iloc[k]
            if ray.intersects(g):
                d = pt.distance(nearest_points(pt, ray.intersection(g))[1])
                facade = d if facade is None or d < facade else facade

        curb_in, curb_out = None, None
        if sidewalks is not None:
            hits = [sidewalks.geometry.iloc[k]
                    for k in sw_idx.intersection(ray.bounds)
                    if ray.intersects(sidewalks.geometry.iloc[k])]
            if hits:
                seg = ray.intersection(unary_union(hits))
                pieces = list(seg.geoms) if hasattr(seg, "geoms") else [seg]
                ds = []
                for s in pieces:
                    if s.is_empty or s.length == 0:
                        continue
                    cs = list(s.coords)
                    ds += [pt.distance(LineString([c, c]).centroid) for c in cs]
                if ds:
                    curb_in, curb_out = min(ds), max(ds)

        out[f"facade_{side}"] = facade
        out[f"curb_{side}"] = curb_in
        out[f"sidewalk_w_{side}"] = (
            curb_out - curb_in if curb_in is not None and curb_out is not None
            else np.nan)
        # Setback: building face beyond the outer sidewalk edge.
        out[f"setback_{side}"] = (
            facade - curb_out
            if facade is not None and curb_out is not None else np.nan)
    return out


def measure(nodes, sidewalks, buildings):
    from sys import path as _p
    sw_idx = sidewalks.sindex if sidewalks is not None else None
    bd_idx = buildings.sindex

    key = "chain" if "chain" in nodes.columns else "osm_name"
    axis = {}
    for _, grp in nodes.groupby(key):
        grp = grp.sort_values("chain_pos_m") if "chain_pos_m" in grp else grp
        xs, ys = grp.geometry.x.values, grp.geometry.y.values
        for i, nid in enumerate(grp.node_id.values):
            j = min(max(i + 1 if i == 0 else i - 1, 0), len(xs) - 1)
            dx, dy = xs[i] - xs[j], ys[i] - ys[j]
            axis[nid] = (np.degrees(np.arctan2(dx, dy)) % 180
                         if np.hypot(dx, dy) > 1 else np.nan)

    rows = []
    for _, r in nodes.iterrows():
        d = transect_profile(r.geometry, axis.get(r.node_id, np.nan),
                             sidewalks, buildings, sw_idx, bd_idx)
        d["node_id"] = r.node_id
        d["street_axis_deg"] = axis.get(r.node_id, np.nan)
        rows.append(d)

    t = pd.DataFrame(rows)
    for c in ["sidewalk_w", "setback", "facade"]:
        L, R = f"{c}_L", f"{c}_R"
        if L in t and R in t:
            t[f"{c}_mean"] = t[[L, R]].mean(axis=1)
            t[f"{c}_min"] = t[[L, R]].min(axis=1)
    if "facade_L" in t and "facade_R" in t:
        t["W_facade_total"] = t.facade_L + t.facade_R
    return t


def regressions(dm, t, out):
    """Directional GVI on sidewalk width and setback."""
    import statsmodels.api as sm
    from scipy.stats import spearmanr

    d = dm.merge(t, on="node_id", how="left")
    preds = [c for c in ["sidewalk_w_mean", "sidewalk_w_min", "setback_mean",
                         "W_facade_total"] if c in d.columns]
    if not preds:
        print("no predictors measured")
        return

    print(f"\n{'='*76}")
    print("GVI ~ sidewalk geometry, per travel direction "
          "(SEs clustered by street)")
    print(f"{'='*76}")
    print(f"{'direction':13s} {'predictor':17s} {'slope':>8s} {'p':>10s} "
          f"{'R2':>7s} {'rho':>7s} {'n':>6s}")
    print("-" * 76)

    rows = []
    for direction in d.direction.dropna().unique():
        sub = d[d.direction.eq(direction)]
        for c in preds:
            dd = sub[["GVI", c, "osm_name"]].dropna()
            if len(dd) < 40:
                continue
            r = sm.OLS(dd.GVI, sm.add_constant(dd[[c]])).fit(
                cov_type="cluster", cov_kwds={"groups": dd.osm_name})
            rho, _ = spearmanr(dd[c], dd.GVI)
            print(f"{direction:13s} {c:17s} {r.params.iloc[1]:8.3f} "
                  f"{r.pvalues.iloc[1]:10.2e} {r.rsquared:7.4f} "
                  f"{rho:+7.3f} {len(dd):6d}")
            rows.append({"direction": direction, "predictor": c,
                         "slope": r.params.iloc[1], "p": r.pvalues.iloc[1],
                         "r2": r.rsquared, "rho": rho, "n": len(dd)})

    print("\n  Read R2, not p. The comparison that matters is sidewalk width")
    print("  against W_facade_total: if the sidewalk predicts GVI where the")
    print("  full street width did not, greenness is limited by planting")
    print("  space rather than by street proportions.")

    if rows:
        tab = pd.DataFrame(rows)
        Path(out).mkdir(parents=True, exist_ok=True)
        tab.to_csv(Path(out) / "regression_sidewalk.csv", index=False)
        best = tab.loc[tab.groupby("predictor").r2.idxmax()]
        print("\nbest R2 per predictor, across directions:")
        print(best[["predictor", "direction", "r2", "rho", "n"]]
              .round(4).to_string(index=False))
        print(f"\nwrote {Path(out)/'regression_sidewalk.csv'}")
    return d


def run(nodes_path="data/processed/nodes.gpkg",
        dm_path="data/processed/directional_metrics.csv",
        raw_dir="data/raw", out_dir="results/tables"):
    nodes = gpd.read_file(nodes_path).to_crs(PROJ_CRS)
    dm = pd.read_csv(dm_path)
    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    b = nodes.to_crs(4326).total_bounds

    sw = socrata_bbox(SIDEWALK_DATASETS, b, raw / "sidewalks.geojson",
                      "sidewalks")
    bd_path = raw / "building_footprints.geojson"
    bd = (gpd.read_file(bd_path) if bd_path.exists()
          else socrata_bbox(FOOTPRINT_DATASETS, b, bd_path, "footprints"))
    if bd is None:
        sys.exit("no building footprints; cannot measure setback")

    bd = bd.to_crs(PROJ_CRS)
    if sw is not None:
        sw = sw.to_crs(PROJ_CRS)

    t = measure(nodes, sw, bd)
    t.to_csv(Path(raw).parent / "processed" / "sidewalk_geometry.csv",
             index=False)

    print("\nmeasured (m):")
    cols = [c for c in t.columns if c not in ("node_id", "street_axis_deg")]
    print(t[cols].describe(percentiles=[.1, .5, .9]).round(2).T.to_string())
    for c in cols:
        miss = t[c].isna().mean()
        if miss > 0.25:
            print(f"  !! {c}: {miss:.0%} missing -- interpret with care")

    return regressions(dm, t, out_dir)
