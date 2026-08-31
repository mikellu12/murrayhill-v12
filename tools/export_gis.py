"""Every layer the figures draw, as one GeoPackage for QGIS or ArcGIS.

The figures in results/figures are rendered from GIS data through GeoPandas
rather than drawn by hand, but a rendered PNG is a dead end for anyone who
wants to restyle it, add a layer, or check a geometry. This writes the same
inputs out as a single .gpkg so the drawing can be reopened, re-symbolised
and re-exported in a desktop GIS without touching the pipeline.

One file with many layers rather than a folder of shapefiles: shapefile
truncates field names at ten characters, cannot hold a CRS unambiguously,
and splits every layer across five files. GeoPackage is a single SQLite
database, is the OGC standard, and both QGIS and ArcGIS Pro open it
natively.

Everything is written in EPSG:4326 except the analysis columns, which carry
their own units and are metric regardless of the geometry's CRS.

    .venv/Scripts/python tools/export_gis.py
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, box

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import PROJ_CRS, CFG, PROC, RAW, RES, banner

GIS = PROC / "nyc_gis"
AVENUE_HALF_M = 15.0        # 100 ft right of way, 1811 grid
STREET_HALF_M = 9.0         # 60 ft


def _sampled_lines(nodes):
    rows = []
    for (name, chain), g in nodes.groupby(["osm_name", "chain"]):
        g = g.sort_values("chain_pos_m")
        if len(g) > 1:
            rows.append({"osm_name": name, "chain": chain, "n_nodes": len(g),
                         "geometry": LineString(list(zip(g.geometry.x, g.geometry.y)))})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=nodes.crs)


def _corridors():
    """Centrelines buffered to their right of way, dissolved."""
    cl = gpd.read_file(GIS / "centerline_mh.gpkg").to_crs(PROJ_CRS)
    name = cl.get("stname_label", cl.get("street", "")).astype(str).str.upper()
    half = np.where(name.str.contains("AVENUE|AVE\\b|BROADWAY", regex=True),
                    AVENUE_HALF_M, STREET_HALF_M)
    cl["geometry"] = [g.buffer(h, cap_style=2, join_style=2)
                      for g, h in zip(cl.geometry, half)]
    return gpd.GeoDataFrame(geometry=[cl.union_all()], crs=PROJ_CRS)


def main():
    banner("export GIS package")
    out = RES / "gis" / "murrayhill_sim.gpkg"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()            # append mode would stack duplicate layers

    nodes = gpd.read_file(PROC / "nodes.gpkg")
    if "in_study" in nodes.columns:
        nodes = nodes[nodes.in_study]

    # Join the measured values onto the points, so a symbology can be built
    # on SIM in the GIS directly instead of against a separate CSV.
    sim_path = RES / "tables" / "sim_index.csv"
    if sim_path.exists():
        sim = pd.read_csv(sim_path)
        keep = [c for c in ("node_id", "SIM", "G", "M", "P") if c in sim.columns]
        nodes = nodes.merge(sim[keep], on="node_id", how="left")

    b = CFG["study_area"]["bbox"]
    layers = {
        "nodes": nodes.to_crs(4326),
        "sampled_streets": _sampled_lines(nodes).to_crs(4326),
        "street_corridors": _corridors().to_crs(4326),
        "centerline_all": gpd.read_file(GIS / "centerline_mh.gpkg").to_crs(4326),
        "boroughs": gpd.read_file(GIS / "boroughs.gpkg").to_crs(4326),
        "parks": gpd.read_file(GIS / "parks_mn.gpkg").to_crs(4326),
        "study_bbox": gpd.GeoDataFrame(
            geometry=[box(b["west"], b["south"], b["east"], b["north"])], crs=4326),
    }
    footprints = RAW / "building_footprints.geojson"
    if footprints.exists():
        layers["building_footprints"] = gpd.read_file(footprints).to_crs(4326)

    for name, gdf in layers.items():
        # Columns of dtype object that hold no geometry break the GPKG
        # writer; drop rather than guess a type for them.
        bad = [c for c in gdf.columns
               if c != gdf.geometry.name and gdf[c].dtype == "O"
               and not gdf[c].map(lambda v: isinstance(v, (str, type(None)))).all()]
        gdf.drop(columns=bad).to_file(out, layer=name, driver="GPKG")
        print(f"  {name:<22} {len(gdf):>6} features  {gdf.geom_type.iloc[0]}")

    print(f"\nwrote {out}  ({out.stat().st_size / 1024 / 1024:.1f} MB)")
    print("open in QGIS: Layer > Add Layer > Add Vector Layer, or drag the file in")


if __name__ == "__main__":
    main()
