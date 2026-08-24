"""
Build a v12-schema nodes.gpkg from an external coordinate mapping CSV.

WHY THIS EXISTS
---------------
The frame is being rebuilt outside this repo and arrives as a flat CSV of
image_id -> lat/lng. Stages 2-8 need the v12 node schema (chain,
chain_pos_m, typology, northing_m, easting_m). This converts one to the
other without hand-editing a GeoPackage, so the frame stays reproducible
from its source file rather than becoming an opaque binary.

It never writes data/processed/nodes.gpkg. That file is the frame the
cached imagery and azimuth_profiles.npz are keyed to; replacing it in place
would silently repoint every saved profile at a different street corner.
Output goes to its own directory and is adopted deliberately, if at all.

NOTHING IS DROPPED
------------------
Every input node reaches the output. Nodes that fail a check are flagged in
a column, never removed -- which run to analyse is a decision for the
analysis, not for a format converter.

    .venv/bin/python tools/build_frame_from_mapping.py MAPPING.csv --out frames/mapping_v1
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from gridaxis import grid_northing, grid_easting          # noqa: E402
from common import CFG, typology_of                       # noqa: E402

# The mapping file uses its own street vocabulary. Typology is matched by
# regex against OSM-style names (common.typology_of), so the categories have
# to be spelled the way config.yaml's patterns expect. Anything absent here
# is title-cased and flagged rather than guessed into a typology.
STREET_NAMES = {
    "1st_avenue": "1st Avenue", "2nd_avenue": "2nd Avenue",
    "3rd_avenue": "3rd Avenue", "madison_avenue": "Madison Avenue",
    "lexington_avenue": "Lexington Avenue",
    "Park_Ave_East": "Park Avenue", "Park_Ave_West": "Park Avenue",
    **{f"east_{n}{s}_street": f"East {n}{s} Street" for n, s in
       [(34,"th"),(35,"th"),(36,"th"),(37,"th"),(38,"th"),
        (39,"th"),(40,"th"),(41,"st"),(42,"nd")]},
}
# Deliberately absent: Park_Ave_Tunnel_Segment, Tunnel_Exit_Street,
# tudor_city_place. The first is not one street (see the QA report); the
# other two have no counterpart in the current study area definition.


def spatial_runs(g, max_gap_m):
    """Split a street category into spatially contiguous runs.

    A category that jumps hundreds of metres is not one chain, and
    chain_pos_m measured along it would be meaningless. Single linkage on
    the 2-D positions is the right tool: it needs no assumption that the ID
    order follows the street, and unlike sorting along one axis it does not
    fuse two parallel runs of the same street or shred an east-west street
    into fragments.
    """
    p = g[["easting_m", "northing_m"]].values
    if len(p) < 2:
        return np.zeros(len(p), dtype=int)
    from scipy.cluster.hierarchy import fcluster, linkage
    return fcluster(linkage(p, method="single"), max_gap_m,
                    criterion="distance") - 1


def order_along_axis(g):
    """Position in metres along the run's own principal axis, from its
    southern (or western) end. Distances are cumulative between consecutive
    nodes, so a gap inside a run shows up as a gap in chain_pos_m."""
    p = g[["easting_m", "northing_m"]].values
    if len(p) < 2:
        return np.zeros(len(p)), np.arange(len(p))
    c = p - p.mean(0)
    v = np.linalg.svd(c, full_matrices=False)[2][0]
    if v[1] < 0 or (abs(v[1]) < 1e-9 and v[0] < 0):
        v = -v                                     # point it north, else east
    idx = np.argsort(c @ v)
    d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(p[idx], axis=0).T))]
    pos = np.empty(len(p)); pos[idx] = d
    return pos, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="frames/mapping_v1")
    ap.add_argument("--max-gap-m", type=float, default=60.0,
                    help="split a street category into separate chains "
                         "across a spatial gap wider than this")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(a.csv)

    # One row per node. The CSV is per image; lat/lng is constant within a
    # node (verified: max one distinct coordinate per original_node_id).
    n = (raw.groupby("original_node_id")
            .agg(street_category=("street_category", "first"),
                 lat=("lat", "first"), lon=("lng", "first"),
                 cross_dist=("cross_dist", "first"),
                 along_dist=("along_dist", "first"),
                 n_images=("image_id", "size"),
                 directions=("direction", lambda s: "/".join(sorted(set(s)))))
            .reset_index().rename(columns={"original_node_id": "node_id"}))

    # node_id is carried through verbatim. It is street-scoped and stable,
    # unlike the positional n00000 scheme, so a frame edit no longer
    # repoints every downstream ID.
    n["osm_name"] = n.street_category.map(STREET_NAMES)
    n["name_resolved"] = n.osm_name.notna()
    n["osm_name"] = n.osm_name.fillna(
        n.street_category.str.replace("_", " ").str.title())
    n["typology"] = typology_of(n.osm_name)

    n["northing_m"] = grid_northing(n.lat, n.lon)
    n["easting_m"] = grid_easting(n.lat, n.lon)

    # chain / chain_pos_m, split across spatial discontinuities
    chains, pos = [], np.zeros(len(n))
    for cat, g in n.groupby("street_category"):
        lab = spatial_runs(g, a.max_gap_m)
        for r in np.unique(lab):
            sel = g.index[lab == r]
            p, _ = order_along_axis(n.loc[sel])
            pos[n.index.get_indexer(sel)] = p
            chains += [(i, f"{cat}#{r}") for i in sel]
    n["chain"] = pd.Series(dict(chains))
    n["chain_pos_m"] = pos

    # --- checks, all recorded as columns -------------------------------
    ex = CFG["study_area"]["exclude_pattern"]
    n["excluded_by_pattern"] = n.osm_name.str.contains(ex, case=False, na=False)

    # Nodes on different chains sitting on top of each other: the classic
    # double count where an avenue meets a cross street.
    rad = CFG["sampling"]["dedupe_radius_m"] / 2
    p = n[["easting_m", "northing_m"]].values
    D = np.hypot(p[:, None, 0] - p[None, :, 0], p[:, None, 1] - p[None, :, 1])
    np.fill_diagonal(D, np.inf)
    cross = n.chain.values[:, None] != n.chain.values[None, :]
    n["collides_m"] = np.where(cross, D, np.inf).min(1)
    n["intersection_dup"] = n.collides_m < rad

    # Does the travel-direction label agree with the run's geometry? A
    # north-south street labelled Eastbound/Westbound sends the directional
    # metrics down the wrong axis.
    ns = n.directions.str.contains("Northbound")
    run_is_ns = n.groupby("chain").apply(
        lambda g: abs(g.northing_m.max() - g.northing_m.min())
                > abs(g.easting_m.max() - g.easting_m.min()),
        include_groups=False)
    n["axis_mismatch"] = ns != n.chain.map(run_is_ns)

    n["frame_ok"] = ~(n.excluded_by_pattern | n.intersection_dup
                      | n.axis_mismatch | ~n.name_resolved)

    gdf = gpd.GeoDataFrame(
        n, geometry=gpd.points_from_xy(n.lon, n.lat), crs="EPSG:4326"
    ).to_crs("EPSG:32618")
    cols = ["node_id", "osm_name", "street_category", "chain", "chain_pos_m",
            "lat", "lon", "typology", "northing_m", "easting_m",
            "along_dist", "cross_dist", "n_images", "directions",
            "name_resolved", "excluded_by_pattern", "collides_m",
            "intersection_dup", "axis_mismatch", "frame_ok", "geometry"]
    gdf = gdf[cols]
    gdf.to_file(out / "nodes.gpkg", driver="GPKG")
    gdf.drop(columns="geometry").to_csv(out / "nodes_qa.csv", index=False)

    print(f"wrote {out/'nodes.gpkg'}  ({len(gdf)} nodes, "
          f"{gdf.chain.nunique()} chains, CRS EPSG:32618)")
    print(f"\ntypology:\n{gdf.typology.value_counts().to_string()}")
    print(f"\nflags (nodes are flagged, never dropped):")
    for c in ["name_resolved", "excluded_by_pattern", "intersection_dup",
              "axis_mismatch", "frame_ok"]:
        print(f"  {c:<20} {int(gdf[c].sum()):4d}")
    bad = gdf[~gdf.frame_ok]
    if len(bad):
        print(f"\n{len(bad)} nodes flagged, by street:")
        print(bad.groupby("street_category").size().to_string())
    print(f"\nchains split across a >{a.max_gap_m:.0f} m gap:")
    for cat, g in gdf.groupby("street_category"):
        if g.chain.nunique() > 1:
            print(f"  {cat}: {g.chain.nunique()} runs "
                  f"({', '.join(f'{k.split(chr(35))[1]}:n={len(v)}' for k, v in g.groupby('chain'))})")


if __name__ == "__main__":
    main()
