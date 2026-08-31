"""Stage 4 -- metrics, coverage audit, spatial structure.

Produces GVI and VEI under five view conditions from the azimuthal
profiles: the four pedestrian travel directions on the Manhattan grid
(029 / 119 / 209 / 299, each a 180-degree forward view) and the original
full 360.

Also runs the coverage audit. Any node in the frame with no metric is
listed with the reason, so a hole in a map is explained rather than
noticed.
"""
import sys
import numpy as np, pandas as pd, geopandas as gpd
from pathlib import Path
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from common import (CFG, PROC, RAW, RES, banner, DIRECTIONS, slice_metrics,
                    street_axis, typology_of, TYPOLOGY_ORDER)

D = CFG["directional"]
SPAT = CFG["spatial"]


def morans_i(vals, xy, radius, nperm, seed):
    tree = cKDTree(xy)
    pairs = tree.query_pairs(radius, output_type="ndarray")
    if len(pairs) == 0:
        return np.nan, np.nan
    d = np.linalg.norm(xy[pairs[:, 0]] - xy[pairs[:, 1]], axis=1)
    w = 1.0 / np.maximum(d, 1.0)
    z = vals - vals.mean()
    den, S0 = (z ** 2).sum(), w.sum() * 2
    I = (len(vals) / S0) * ((w * z[pairs[:, 0]] * z[pairs[:, 1]]).sum() * 2 / den)
    rng = np.random.default_rng(seed)
    null = [(len(vals) / S0) * ((w * (zp := rng.permutation(z))[pairs[:, 0]]
                                 * zp[pairs[:, 1]]).sum() * 2 / den)
            for _ in range(nperm)]
    return I, (np.sum(np.abs(null) >= abs(I)) + 1) / (nperm + 1)


def main():
    banner("STAGE 4  metrics")
    nodes = gpd.read_file(PROC / "nodes.gpkg")
    z = np.load(PROC / "azimuth_profiles.npz")
    prof = {k: z[k] for k in z.files}
    axis = street_axis(nodes)
    info = nodes.set_index("node_id")

    rows, empty = [], []
    for nid, p in prof.items():
        # A profile whose class rows are all zero is not a measurement of an
        # empty street, it is the absence of a measurement. The weight row
        # still says images were segmented, so nothing downstream notices:
        # GVI comes out a plausible 0.0 and is averaged in as real, while VEI
        # divides by a zero denominator and drops to NaN. That split is the
        # tell -- the node silently leaves the enclosure analyses and stays in
        # the greenery ones. Drop it here, once, where the profile is read.
        cls = p[:-1] if p.shape[0] > 3 else p
        if not np.any(cls):
            empty.append(nid)
            continue
        ax = axis.get(nid, np.nan)
        views = [(lab, b, D["fov"]) for lab, b in DIRECTIONS.items()]
        views.append(("full360", 0.0, 360.0))
        for lab, b, f in views:
            gvi, vei = slice_metrics(p, b, f)
            al = np.nan
            if lab != "full360" and not np.isnan(ax):
                dd = abs((b - ax) % 180)
                al = bool(min(dd, 180 - dd) <= D["along_street_tol"])
            rows.append({"node_id": nid, "direction": lab, "bearing_deg":
                         np.nan if lab == "full360" else b, "fov_deg": f,
                         "street_axis_deg": ax, "along_street": al,
                         "GVI": gvi, "VEI": vei})

    if empty:
        print(f"\ndropped {len(empty)} node(s) with an empty profile "
              f"(no pixels in any class): {', '.join(sorted(empty))}")

    dm = pd.DataFrame(rows)
    for c in ["osm_name", "typology", "northing_m", "lat", "lon"]:
        if c in info.columns:
            dm[c] = dm.node_id.map(info[c])
    # Re-derive typology from the single canonical definition so the label
    # set can never drift between frame and metrics.
    dm["typology"] = typology_of(dm.osm_name)
    dm.to_csv(PROC / "directional_metrics.csv", index=False)

    wide = dm.pivot_table(index="node_id", columns="direction",
                          values=["GVI", "VEI"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    for c in ["osm_name", "typology", "northing_m", "lat", "lon"]:
        if c in info.columns:
            wide[c] = wide.node_id.map(info[c])
    wide.to_csv(PROC / "directional_metrics_wide.csv", index=False)

    # Canonical 360 table, for everything downstream.
    metrics = (dm[dm.direction.eq("full360")]
               .drop(columns=["direction", "bearing_deg", "fov_deg",
                              "along_street"])
               .merge(nodes[["node_id", "geometry", "chain", "chain_pos_m"]
                            + [c for c in ("street_segment", "cross_dist",
                                           "along_dist", "mapping_id")
                               if c in nodes.columns]],
                      on="node_id"))
    # Assert the CRS from what nodes.gpkg actually carries, never a literal.
    # Stamping crs=32618 onto lat/lon degrees produces a frame that looks
    # right, plots right and silently breaks every distance downstream: s05's
    # 90 m probe found 0 of 764 facades because it was casting metres across
    # a space measured in degrees.
    metrics = gpd.GeoDataFrame(metrics, geometry="geometry", crs=nodes.crs)
    if metrics.crs is None:
        sys.exit("nodes.gpkg has no CRS -- refusing to guess")
    metrics = metrics.to_crs(32618)
    sc = PROC / "scaffold_by_node.csv"
    if sc.exists():
        metrics = metrics.merge(pd.read_csv(sc), on="node_id", how="left")

    # ------------------------------------------------ coverage audit
    extra = sorted(set(prof) - set(nodes.node_id))
    if extra:
        sys.exit(f"{len(extra)} profiled nodes are not in the current frame "
                 f"(e.g. {extra[:3]}). data/processed is from an older frame "
                 f"-- delete it and re-run from s01.")
    missing = sorted(set(nodes.node_id) - set(prof))
    print(f"\n--- coverage audit ---")
    print(f"  nodes in frame     {len(nodes)}")
    print(f"  with a profile     {len(prof)}")
    print(f"  no metric          {len(missing)}")
    if missing:
        mrep = nodes[nodes.node_id.isin(missing)].copy()
        # The reason is already known -- s02 probed it. Carrying it here
        # turns "22 nodes have no metric" into an auditable statement.
        # Do not guess: a date exclusion and a coverage gap have opposite
        # implications, and only one of them is fixable by re-fetching.
        mpath = RAW / "metadata.csv"
        if mpath.exists():
            meta = pd.read_csv(mpath).set_index("node_id")

            def why(nid):
                if nid not in meta.index:
                    return "not probed"
                r = meta.loc[nid]
                if r.get("status") != "OK":
                    return f"status={r.get('status')}"
                tgt = CFG["capture"]["target"]
                if tgt and r.get("pano_date") != tgt:
                    return f"capture {r.get('pano_date')}, wanted {tgt}"
                off = r.get("pano_offset_m")
                if pd.notna(off) and off > CFG["sampling"]["max_pano_offset_m"]:
                    return f"pano {off:.0f} m from node"
                return "usable but unprofiled -- check s03"

            mrep["reason"] = [why(n) for n in mrep.node_id]
            mrep["pano_date"] = mrep.node_id.map(meta.get("pano_date"))
        else:
            mrep["reason"] = "metadata.csv absent"
        print(mrep.groupby(["osm_name"]).size().to_string())
        print("\n  reasons:")
        print("  " + mrep.reason.value_counts().to_string().replace("\n", "\n  "))
        cols = [c for c in ["node_id", "osm_name", "typology", "reason",
                            "pano_date", "lat", "lon"] if c in mrep.columns]
        mrep[cols].to_csv(RES / "tables" / "nodes_without_metrics.csv",
                          index=False)
        # A run of consecutive IDs on one street is a stretch the capture
        # drive missed, not scattered dropout, and it thins that block face
        # systematically. Worth seeing without reading the CSV.
        ids = sorted(int(n[1:]) for n in missing)
        runs, start = [], ids[0]
        for a_, b_ in zip(ids, ids[1:] + [None]):
            if b_ is None or b_ != a_ + 1:
                if a_ - start >= 2:
                    runs.append((start, a_))
                start = b_
        if runs:
            print("\n  consecutive runs (a missed stretch, not random "
                  "dropout):")
            for s_, e_ in runs:
                st = nodes.loc[nodes.node_id.eq(f"n{s_:05d}"), "osm_name"]
                nm = st.iloc[0] if len(st) else "?"
                print(f"    n{s_:05d}-n{e_:05d}  {e_-s_+1} nodes on {nm}")
        print("  -> nodes_without_metrics.csv")
    nanvei = metrics.VEI.isna().sum()
    if nanvei:
        print(f"  VEI is NaN for {nanvei} nodes (no sky AND no building "
              f"pixels -- typically a fully vegetated or fully occluded view)")

    print("\ntypology of the analytic sample:")
    print(metrics.typology.value_counts().reindex(TYPOLOGY_ORDER)
                 .dropna().to_string())

    # ------------------------------------------------ band sky fractions
    # Folded in from the old standalone skyview.py. It read the same
    # profiles and recomputed the same quantity, so keeping it separate
    # only created a second place for the definition to drift.
    #
    # This is NOT the sky view factor. The imagery covers +/-45 deg of
    # elevation, so this is the sky share OF THAT BAND; the zenith, where
    # most of the sky is, is never sampled. The name says band and the
    # paper must too.
    #
    # Not redundant with VEI, either:
    #     VEI      = building / (sky + building)   -- conditions on the pair
    #     SVF_band = sky / everything in frame     -- vegetation reduces it
    # The gap between (1 - VEI) and SVF_band is the share of view that is
    # neither sky nor building, which is mostly foliage. That residual is
    # the point of computing both.
    def _band_shares(p_):
        W = p_[3].sum()
        if W <= 0:
            return np.nan, np.nan
        return p_[1].sum() / W, p_[2].sum() / W

    sv = pd.DataFrame(
        [{"node_id": k, "SVF_band": a, "BVF_band": b}
         for k, (a, b) in ((k, _band_shares(p_)) for k, p_ in prof.items())])
    metrics = metrics.merge(sv, on="node_id", how="left")
    metrics["sky_of_pair"] = 1 - metrics.VEI
    metrics["sky_veg_residual"] = metrics.sky_of_pair - metrics.SVF_band
    print(f"\nband sky fractions (NOT SVF -- zenith unsampled):")
    print(f"  SVF_band median {metrics.SVF_band.median():.3f}   "
          f"BVF_band median {metrics.BVF_band.median():.3f}")
    print(f"  mean gap between (1-VEI) and SVF_band: "
          f"{metrics.sky_veg_residual.mean():+.3f}")
    print("  That gap is view which is neither sky nor building -- foliage,")
    print("  road, vehicles. Report as SVF_band, never as SVF.")

    # ------------------------------------------------ block faces
    m = metrics.sort_values(["chain", "chain_pos_m"]).copy()
    new = ((m.chain != m.chain.shift()) |
           (m.chain_pos_m.diff().abs() > SPAT["face_gap_m"]))
    m["face_id"] = "f" + new.cumsum().astype(str)
    face = (m.groupby("face_id")
              .agg(osm_name=("osm_name", "first"), typology=("typology", "first"),
                   northing_m=("northing_m", "mean"),
                   GVI=("GVI", "median"),
                   VEI=("VEI", "median"), n_nodes=("node_id", "size"))
              .reset_index())
    face = face[face.n_nodes >= 2]
    m.drop(columns="geometry").to_csv(PROC / "nodes_with_faces.csv", index=False)
    face.to_csv(PROC / "block_faces.csv", index=False)
    metrics["face_id"] = m.set_index("node_id").face_id.reindex(
        metrics.node_id).values

    print(f"\nblock faces: {len(face)} units from {len(m)} nodes "
          f"(median {face.n_nodes.median():.0f} nodes/face)")

    # ------------------------------------------------ autocorrelation
    xy = np.c_[metrics.geometry.x.values, metrics.geometry.y.values]
    print("\nMoran's I (inverse distance, "
          f"{SPAT['moran_radius_m']} m band, {SPAT['moran_permutations']} perms):")
    for col in ["GVI", "VEI"]:
        v = metrics[col].values
        ok = ~np.isnan(v)
        I, p = morans_i(v[ok], xy[ok], SPAT["moran_radius_m"],
                        SPAT["moran_permutations"], CFG["seed"])
        print(f"  {col}: I={I:+.3f} p={p:.3f} "
              f"{'clustered' if I > 0.1 else 'weak'}")
    print("  Node-level p-values are inflated by this. Report block-face n,")
    print("  or standard errors clustered by face.")

    metrics.to_file(PROC / "metrics.gpkg", driver="GPKG")
    metrics.drop(columns="geometry").to_csv(PROC / "metrics.csv", index=False)
    print(f"\nwrote metrics.csv / .gpkg, directional_metrics.csv, "
          f"block_faces.csv")


if __name__ == "__main__":
    main()
