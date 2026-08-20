"""Stage 1 -- sampling frame.

Fixed BEFORE any imagery request. Coverage gaps are recorded later as
attributes of these nodes, never as absences from the frame.

Three traps handled here, each of which silently corrupted an earlier run:

1. network_type must be "drive", not "walk". In Manhattan the walk network
   returns sidewalk and footway geometry -- both sides of every street as
   separate unnamed ways, plus plazas and building passages. ~90% carry no
   street name, which makes any typology split impossible.

2. Interpolate along the MERGED geometry of each street. OSM splits a
   street at every intersection; stepping 20 m along each edge restarts the
   count at every corner. And linemerge only joins lines whose endpoints
   match exactly, while OSM leaves sub-metre gaps -- so endpoints are
   snapped first. Without this, East 34th came out as nine fragments.

3. The study area is bounded by STREETS, not a lat/lon rectangle.
   Manhattan's grid is rotated ~29 deg, so any rectangle wide enough to
   hold all of Madison reaches well past First at the opposite end.
"""
import sys
import numpy as np, pandas as pd, geopandas as gpd, osmnx as ox
from pathlib import Path
from shapely.geometry import MultiPoint, MultiLineString, LineString
from shapely.ops import unary_union, linemerge, snap
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from common import (CFG, PROC, banner, norm_name, typology_of,
                    grid_northing, grid_easting)

SA = CFG["study_area"]
SP = CFG["sampling"]


def merge_street(geoms, tol):
    lines = []
    for g in geoms:
        if g is None or g.is_empty:
            continue
        if isinstance(g, LineString):
            lines.append(g)
        elif isinstance(g, MultiLineString):
            lines.extend(g.geoms)
    if not lines:
        return []
    if len(lines) == 1:
        return lines
    u = unary_union(lines)
    m = linemerge(snap(u, u, tol))
    return list(m.geoms) if isinstance(m, MultiLineString) else [m]


def corner(edges, gnodes, av, st):
    a = edges[edges.nm.str.contains(av, case=False, na=False)]
    s = edges[edges.nm.str.contains(st, case=False, na=False)]
    if a.empty or s.empty:
        return None
    na = set(a.index.get_level_values("u")) | set(a.index.get_level_values("v"))
    ns = set(s.index.get_level_values("u")) | set(s.index.get_level_values("v"))
    both = na & ns
    if both:
        return gnodes.loc[list(both)].geometry.union_all().centroid
    inter = (unary_union(a.geometry.tolist()).buffer(12)
             .intersection(unary_union(s.geometry.tolist()).buffer(12)))
    return None if inter.is_empty else inter.centroid


def main():
    banner("STAGE 1  sampling frame")
    ox.settings.use_cache = True
    b = SA["bbox"]
    G = ox.graph_from_bbox(bbox=(b["west"], b["south"], b["east"], b["north"]),
                           network_type=SA["network_type"], simplify=True)
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True).to_crs(32618)
    gnodes = ox.graph_to_gdfs(G, nodes=True, edges=False).to_crs(32618)
    edges["nm"] = edges["name"].map(norm_name)

    # Structural exclusion by TAG. The Park Avenue viaduct around Grand
    # Central is bridge=yes but still named "Park Avenue"; name matching
    # cannot catch it. Slip roads are highway=*_link.
    def has(col):
        if col not in edges.columns:
            return pd.Series(False, index=edges.index)
        v = edges[col]
        return v.notna() & ~v.astype(str).isin(["no", "False", "nan", ""])

    link = edges.get("highway", pd.Series("", index=edges.index)) \
                .astype(str).str.contains("_link", na=False)
    drop = has("bridge") | has("tunnel") | link
    print(f"dropping {int(drop.sum())} edges "
          f"({int((has('bridge') | has('tunnel')).sum())} bridge/tunnel, "
          f"{int(link.sum())} link roads)")
    edges = edges[~drop].copy()

    corners, missing = [], []
    for av, st in SA["corner_pairs"]:
        c = corner(edges, gnodes, av, st)
        (corners if c is not None else missing).append(c if c is not None
                                                       else f"{av} x {st}")
    print(f"corners found: {len(corners)}/4")
    for m in missing:
        print(f"  MISSING {m}")
    if len(corners) < 3:
        sys.exit("fewer than 3 corners -- widen bbox or fix corner_pairs")

    tight = MultiPoint(corners).convex_hull
    hull = tight.buffer(SA["corner_buffer_m"])
    print(f"study area: {tight.area/1e6:.3f} km2")

    keep = (f"(?:{SA['canyon_pattern']})|(?:{SA['secondary_pattern']})"
            f"|(?:{SA['midblock_pattern']})")
    sel = edges[edges.nm.str.contains(keep, case=False, na=False)]
    sel = sel[~sel.nm.str.contains(SA["exclude_pattern"], case=False, na=False)]

    pts = []
    for street, grp in sel.groupby("nm"):
        for pi, line in enumerate(merge_street(grp.geometry.tolist(),
                                               SP["snap_tol_m"])):
            if line.length < SP["grid_spacing_m"]:
                continue
            for i in range(int(line.length // SP["grid_spacing_m"]) + 1):
                d = i * SP["grid_spacing_m"]
                pts.append({"geometry": line.interpolate(d), "osm_name": street,
                            "chain": f"{street}#{pi}", "chain_pos_m": float(d)})

    nodes = gpd.GeoDataFrame(pts, crs=32618)
    nodes = nodes[nodes.within(hull)]
    nodes = nodes[nodes.within(tight.buffer(SA["clip_buffer_m"]))]
    nodes = nodes.reset_index(drop=True)

    # Dedupe only where different streets cross; never inside a chain.
    order = np.lexsort((nodes.chain_pos_m.values, nodes.chain.values))
    xy = np.c_[nodes.geometry.x.values, nodes.geometry.y.values]
    tree, chains = cKDTree(xy), nodes.chain.values
    keep_idx, taken = [], {}
    for idx in order:
        if not any(j in taken and taken[j] != chains[idx]
                   for j in tree.query_ball_point(xy[idx], SP["dedupe_radius_m"])):
            keep_idx.append(idx)
            taken[idx] = chains[idx]
    nodes = nodes.iloc[sorted(keep_idx)].reset_index(drop=True)
    nodes["node_id"] = [f"n{i:05d}" for i in range(len(nodes))]

    w = nodes.to_crs(4326)
    nodes["lat"], nodes["lon"] = w.geometry.y, w.geometry.x
    nodes["typology"] = typology_of(nodes.osm_name)
    # Continuous grid-axis position, replacing the old zone categories.
    nodes["northing_m"] = grid_northing(nodes.lat, nodes.lon)
    nodes["easting_m"] = grid_easting(nodes.lat, nodes.lon)

    gaps = np.concatenate([np.diff(np.sort(g.chain_pos_m.values))
                           for _, g in nodes.groupby("chain") if len(g) > 1])
    gaps = gaps[gaps < 100]
    on = np.abs(gaps - SP["grid_spacing_m"]) < 0.5
    # A double gap is an intersection: the dedupe deliberately drops a node
    # where two streets cross, leaving ~40 m. That is correct behaviour, not
    # a spacing failure, so count it separately.
    dbl = np.abs(gaps - 2 * SP["grid_spacing_m"]) < 1.0
    print(f"spacing: median {np.median(gaps):.1f} m | "
          f"{on.mean():.1%} at 20 m | {dbl.mean():.1%} at 40 m "
          f"(intersection dedupe) | {1 - on.mean() - dbl.mean():.1%} other")

    per = nodes.groupby("osm_name").agg(n=("node_id", "size"),
                                        chains=("chain", "nunique"))
    frag = per[per.chains > 2]
    if len(frag):
        print(f"  multi-chain: {list(frag.index)}")
        print("    Some of this is topological, not a tolerance problem:")
        print("    linemerge also stops where 3+ lines meet, so a service")
        print("    spur joining an avenue splits the chain no matter how")
        print("    large snap_tol_m is. It only affects where block-face")
        print("    boundaries fall, not the 20 m spacing itself.")
    div = per[per.chains == 2]
    if len(div):
        print(f"  two chains (divided carriageway, or one split): "
              f"{list(div.index)}")

    nodes.to_file(PROC / "nodes.gpkg", driver="GPKG")
    print(f"\n{len(nodes)} nodes -> {len(nodes)*4} images")
    print(nodes.typology.value_counts().to_string())
    print()
    print(per.sort_values("n", ascending=False).to_string())
    print(f"\nwrote {PROC/'nodes.gpkg'}")


if __name__ == "__main__":
    main()
