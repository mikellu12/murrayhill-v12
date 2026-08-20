"""Stage 5 -- building heights, measured street width, H/W.

H/W is the standard enclosure ratio. The denominator matters more than it
looks: an earlier version assigned W by typology (30.5 m avenues, 18.3 m
cross streets), which made H/W partly definitional -- typology determined
the denominator, so the typology contrast was guaranteed. Measured
facade-to-facade width removes that circularity and reverses the result.

W is measured by casting a perpendicular transect from each node and
finding the first building face on either side. That is the distance that
actually bounds the view, and unlike the legal right-of-way it accounts for
setbacks, plazas and arcades.
"""
import sys
import numpy as np, pandas as pd, geopandas as gpd, requests
from pathlib import Path
from shapely.geometry import LineString, shape
from shapely.ops import nearest_points, unary_union
from scipy.stats import spearmanr, mannwhitneyu

sys.path.insert(0, str(Path(__file__).parent))
from common import CFG, PROC, RAW, banner, street_axis

G = CFG["geometry"]


def fetch_footprints(nodes, path):
    if path.exists():
        return
    b = nodes.to_crs(4326).total_bounds
    pad = 0.001
    where = f"within_box(the_geom, {b[3]+pad}, {b[0]-pad}, {b[1]-pad}, {b[2]+pad})"
    for ds in G["footprint_datasets"]:
        url = f"https://data.cityofnewyork.us/resource/{ds}.json"
        try:
            t = requests.get(url, params={"$limit": 1}, timeout=30)
            if t.status_code != 200 or not t.json():
                continue
        except Exception:
            continue
        print(f"footprints dataset {ds}")
        rows, off = [], 0
        while True:
            r = requests.get(url, params={"$where": where, "$limit": 5000,
                                          "$offset": off}, timeout=60)
            r.raise_for_status()
            ch = r.json()
            if not ch:
                break
            rows += ch
            off += 5000
        gpd.GeoDataFrame(pd.DataFrame(rows),
                         geometry=[shape(x["the_geom"]) for x in rows],
                         crs=4326).to_file(path, driver="GeoJSON")
        print(f"  {len(rows)} buildings -> {path}")
        return
    print("no working footprint dataset; download manually from NYC Open Data")


def main():
    banner("STAGE 5  geometry")
    metrics = gpd.read_file(PROC / "metrics.gpkg")
    nodes = gpd.read_file(PROC / "nodes.gpkg")
    fp = RAW / "building_footprints.geojson"
    fetch_footprints(nodes, fp)
    if not fp.exists():
        print("skipping Stage 5")
        return

    bf = gpd.read_file(fp).to_crs(32618)
    hcol = "height_roof" if "height_roof" in bf.columns else "heightroof"
    bf["h_m"] = pd.to_numeric(bf[hcol], errors="coerce") * 0.3048
    bf = bf[bf.h_m.between(G["height_min_m"], G["height_max_m"])]
    print(f"{len(bf)} buildings with usable heights")

    buf = metrics[["node_id", "geometry"]].copy()
    buf["geometry"] = buf.buffer(G["corridor_m"])
    j = gpd.sjoin(bf[["h_m", "geometry"]], buf, predicate="intersects")
    H = j.groupby("node_id").h_m.median().rename("H_m")   # median, not mean
    metrics = metrics.merge(H, on="node_id", how="left")

    # ---------------------------------------------- facade-to-facade width
    axis = street_axis(nodes)
    sidx = bf.sindex
    half = G["facade_half_m"]

    def width(pt, ax):
        if np.isnan(ax):
            return np.nan
        ux, uy = np.sin(np.radians(ax)), np.cos(np.radians(ax))
        px, py = -uy, ux
        out = []
        for sgn in (1, -1):
            ray = LineString([(pt.x, pt.y),
                              (pt.x + sgn * px * half, pt.y + sgn * py * half)])
            best = None
            for k in sidx.intersection(ray.bounds):
                geom = bf.geometry.iloc[k]
                if not ray.intersects(geom):
                    continue
                inter = ray.intersection(geom)
                if inter.is_empty:
                    continue
                d = pt.distance(nearest_points(pt, inter)[1])
                best = d if best is None or d < best else best
            out.append(best)
        return np.nan if None in out else out[0] + out[1]

    ws, why = {}, {"no_axis": 0, "no_facade": 0, "out_of_range": 0}
    nm = metrics.to_crs(32618)
    for _, r in nm.iterrows():
        ax = axis.get(r.node_id, np.nan)
        if np.isnan(ax):
            why["no_axis"] += 1
            continue
        w = width(r.geometry, ax)
        if not np.isfinite(w):
            why["no_facade"] += 1
        elif not (4 < w < 120):
            why["out_of_range"] += 1
        else:
            ws[r.node_id] = w
    print(f"facade width failures: {why}")

    metrics = metrics.merge(
        pd.Series(ws, name="W_facade").rename_axis("node_id"),
        on="node_id", how="left")
    print(f"W_facade measured for {metrics.W_facade.notna().sum()}/"
          f"{len(metrics)} nodes")
    cov = metrics.assign(has=metrics.W_facade.notna())
    print("coverage by typology:")
    print(cov.groupby("typology").has.agg(["mean", "size"]).round(3).to_string())
    print("  Clustered missingness would bias any H/W contrast; check this")
    print("  before interpreting the numbers below.")

    metrics["HW_facade"] = metrics.H_m / metrics.W_facade
    metrics["HW_assigned"] = metrics.H_m / np.where(
        metrics.typology.eq("avenue_canyon"), 30.5, 18.3)
    metrics["HW_ratio"] = metrics.HW_facade

    print("\nH/W by typology (measured width):")
    print(metrics.groupby("typology")[["H_m", "W_facade", "HW_facade"]]
                 .median().round(2).to_string())

    for col in ["HW_assigned", "HW_facade"]:
        a = metrics.loc[metrics.typology.eq("avenue_canyon"), col].dropna()
        b = metrics.loc[metrics.typology.eq("mid_block"), col].dropna()
        if len(a) > 5 and len(b) > 5:
            u, p = mannwhitneyu(a, b)
            r = 1 - (2 * u) / (len(a) * len(b))
            print(f"{col:13s} canyon={a.median():.2f} mid={b.median():.2f} "
                  f"p={p:.2e} r={r:+.3f}")
    print("  HW_assigned is retained ONLY to document the artifact: because")
    print("  typology sets its denominator, its contrast is definitional.")

    print("\nVEI vs measured H/W (the validation that matters):")
    s = metrics.dropna(subset=["VEI", "HW_facade"])
    rho, p = spearmanr(s.VEI, s.HW_facade)
    print(f"  rho={rho:+.3f} p={p:.2e} n={len(s)}")

    metrics.to_file(PROC / "metrics.gpkg", driver="GPKG")
    metrics.drop(columns="geometry").to_csv(PROC / "metrics.csv", index=False)
    print("\nupdated metrics.csv / .gpkg")


if __name__ == "__main__":
    main()
