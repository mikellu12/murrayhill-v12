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

The transect is a BAND, not a cone: a parallel-sided corridor one node
spacing wide, sampled by nine rays, reduced by the nearest hit. Adjacent
nodes therefore tile the street instead of overlapping, and the probe does
not widen with range, so it cannot punch through a gap onto the next block
and report 110 m on a 20 m street. config.yaml carries the measurement
behind that choice, including why the cone's better pooled correlation is a
Simpson's paradox. `hw_probe: cone` restores the old behaviour.
"""
import sys
import numpy as np, pandas as pd, geopandas as gpd, requests
from pathlib import Path
from shapely.geometry import LineString, Point, Polygon, shape
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
    # s05 writes back into metrics.gpkg, so a second run would merge its own
    # output and silently produce W_facade_x / W_facade_y. Drop what this
    # stage owns before recomputing it.
    metrics = metrics.drop(columns=[c for c in
        ("H_m", "W_facade", "HW_facade", "HW_assigned", "HW_ratio",
         "HW_effective", "HW_source", "facade_sides_hit",
         "facade_sides_open", "facade_empty_bearing")
        if c in metrics.columns])
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

    cone = G["facade_cone_deg"]

    open_test = G["facade_open_test_deg"]
    open_reach = G.get("facade_open_reach_m", half)

    def _wedge(pt, bearing, steps=9, deg=None):
        """Fan from the node, cone degrees either side of `bearing`.

        A zero-width ray slides between footprint polygons and past building
        edges. The error it has to tolerate is angular -- the street bearing
        is fitted, not known -- so the probe widens with distance rather than
        staying parallel: tight where the facade should be, forgiving at range.
        """
        span = cone if deg is None else deg
        reach = half if deg is None else open_reach
        pts = [(pt.x, pt.y)]
        for tdeg in np.linspace(-span, span, steps):
            b = np.radians(bearing + tdeg)
            pts.append((pt.x + np.sin(b) * reach, pt.y + np.cos(b) * reach))
        return Polygon(pts)

    band_w = CFG["sampling"]["grid_spacing_m"]
    n_rays = G.get("hw_band_rays", 9)
    band_stat = G.get("hw_band_stat", "min")

    def _band_reach(pt, bearing, deg=None):
        """Nearest (or median) wall across a parallel-sided corridor.

        One node-spacing wide, so adjacent nodes tile the street instead of
        overlapping, and the width does not grow with range. `deg` widens the
        corridor for the open-side test only, mirroring the cone's signature.
        """
        w = band_w if deg is None else band_w * (deg / max(cone, 1e-9))
        reach = half if deg is None else open_reach
        r = np.radians(bearing)
        ux, uy = np.sin(r), np.cos(r)
        px, py = uy, -ux
        hits = []
        for off in np.linspace(-w / 2, w / 2, n_rays):
            ox, oy = pt.x + px * off, pt.y + py * off
            o = Point(ox, oy)
            ray = LineString([(ox, oy), (ox + ux * reach, oy + uy * reach)])
            best = None
            for k in sidx.intersection(ray.bounds):
                geom = bf.geometry.iloc[k]
                if not ray.intersects(geom):
                    continue
                inter = ray.intersection(geom)
                if inter.is_empty:
                    continue
                d = o.distance(nearest_points(o, inter)[1])
                best = d if best is None or d < best else best
            if best is not None:
                hits.append(best)
        if not hits:
            return None
        return float(min(hits) if band_stat == "min" else np.median(hits))

    def _cone_reach(pt, bearing, deg=None):
        """Distance to the nearest footprint inside the fan, or None."""
        probe = _wedge(pt, bearing, deg=deg)
        best = None
        for k in sidx.intersection(probe.bounds):
            geom = bf.geometry.iloc[k]
            if not probe.intersects(geom):
                continue
            inter = probe.intersection(geom)
            if inter.is_empty:
                continue
            d = pt.distance(nearest_points(pt, inter)[1])
            best = d if best is None or d < best else best
        return best

    _reach = _band_reach if G.get("hw_probe", "cone") == "band" else _cone_reach
    print(f"width probe: {G.get('hw_probe', 'cone')}"
          + (f", {n_rays} rays, {band_stat}" if _reach is _band_reach
             else f", +/-{cone} deg"))

    def width(pt, ax):
        """(width, n_sides_hit, n_sides_genuinely_open).

        A side that comes back empty is two different things, and they need
        different treatment downstream. Re-probing the empty side with a much
        wider fan separates them: if a footprint appears there, the narrow
        cone aimed past a wall that exists; if nothing appears even at
        open_test degrees, there is no wall on that bearing at all.
        """
        if np.isnan(ax):
            return np.nan, 0, 0, np.nan
        out, opens, empty_b = [], 0, np.nan
        for sgn in (1, -1):
            b = (ax + sgn * 90.0) % 360
            d = _reach(pt, b)
            out.append(d)
            if d is None and _reach(pt, b, deg=open_test) is None:
                opens += 1
                empty_b = b
        hits = sum(d is not None for d in out)
        w = out[0] + out[1] if hits == 2 else np.nan
        return w, hits, opens, empty_b

    ws, sides, why = {}, {}, {"no_axis": 0, "no_facade": 0, "out_of_range": 0}
    nm = metrics.to_crs(32618)
    for _, r in nm.iterrows():
        ax = axis.get(r.node_id, np.nan)
        if np.isnan(ax):
            why["no_axis"] += 1
            continue
        w, hits, opens, empty_b = width(r.geometry, ax)
        sides[r.node_id] = (hits, opens, empty_b)
        if not np.isfinite(w):
            why["no_facade"] += 1
        elif not (G['hw_min_w_m'] < w < G['hw_max_w_m']):
            # 0 m means the probe hit a wall at zero distance on both sides:
            # the node is inside a footprint, a snap error in the network.
            # 120+ means it punched through a gap onto the next block. Both
            # are failures, neither is an open street.
            why["out_of_range"] += 1
        else:
            ws[r.node_id] = w
    print(f"facade width failures: {why}")
    metrics["facade_sides_hit"] = metrics.node_id.map(
        lambda n: sides.get(n, (0, 0, np.nan))[0])
    metrics["facade_sides_open"] = metrics.node_id.map(
        lambda n: sides.get(n, (0, 0, np.nan))[1])
    metrics["facade_empty_bearing"] = metrics.node_id.map(
        lambda n: sides.get(n, (0, 0, np.nan))[2])

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
    # ------------------------------------------- corner nodes inherit H/W
    # A node inside a crossing has no facade on the perpendicular: the probe
    # points down the cross street, and 87 per cent of nodes within 15 m of a
    # crossing return nothing against a flat 12 per cent elsewhere. That is
    # not a failed measurement, it is an undefined one -- there is no street
    # wall on that bearing. The canyon such a node sits in is still the
    # street's, so it inherits the median of measured nodes on the same
    # street within hw_fill_radius_m. HW_facade stays measured-only and
    # HW_source says which is which; nothing downstream should confuse them.
    # street_segment is the colleague's coordinate mapping, which splits Park
    # Avenue into west / tunnel / east. Those three are morphologically
    # different -- the tunnel approach runs 100 m facade to facade against 44 m
    # on the boulevard -- so pooling them lets a gap inherit across the break.
    street = (metrics.street_segment if "street_segment" in metrics.columns
              and metrics.street_segment.notna().any()
              else (metrics.cleaned_street
                    if "cleaned_street" in metrics.columns
                    and metrics.cleaned_street.notna().any()
                    else metrics.osm_name))
    metrics["_street"] = street
    um = metrics.to_crs(32618)
    ex, ny = um.geometry.x.to_numpy(), um.geometry.y.to_numpy()
    hw = metrics.HW_facade.to_numpy(float)
    # object dtype, not np.where: a fixed-width string array from np.where is
    # <U8 here and truncates "segment_median" to "segment_" on assignment.
    filled = hw.copy()
    source = np.array(["measured" if np.isfinite(v) else "none" for v in hw],
                      dtype=object)
    # One side built, the other genuinely empty even under the wide fan. There
    # is no facade-to-facade distance to measure, but that is not missing
    # data: the street has no opposite wall, so it is not a canyon. Marked
    # before the segment fill runs, because inheriting a neighbour's aspect
    # ratio would hand these nodes the very enclosure they do not have. 1st
    # Avenue along the tunnel approach is most of them. HW_effective stays
    # NaN -- there is no ratio -- and sim_core.omega reads the flag instead.
    openish = ((metrics.facade_sides_hit.to_numpy() == 1)
               & (metrics.facade_sides_open.to_numpy() == 1)
               & ~np.isfinite(hw))

    # A cross street is also empty for 90 m, so a corner node whose empty
    # bearing points down one is indistinguishable from an open lot by
    # footprints alone. Separate them by aim: if the empty bearing lines up
    # with a nearby node on a different street, the probe is looking along a
    # street, not across an open side, and the node is a crossing after all.
    # Real open lots here sit 40-80 degrees off the nearest cross street.
    # Both sides of the "is this a different street" test must speak the same
    # vocabulary. `_street` is street_segment when the coordinate mapping is
    # present, so comparing it against osm_name would never match -- every
    # node's nearest "cross street" node would be its own neighbour, whose
    # axis is parallel to the street, giving 90 degrees and demoting nothing.
    other = nodes.to_crs(32618)
    oname_col = ("street_segment" if "street_segment" in other.columns
                 and other.street_segment.notna().any() else "osm_name")
    other = other[other[oname_col].notna()]
    ox, oy = other.geometry.x.to_numpy(), other.geometry.y.to_numpy()
    oname = other[oname_col].to_numpy()
    eb = metrics.facade_empty_bearing.to_numpy(float)
    sname = metrics._street.to_numpy()
    oax = np.array([axis.get(k, np.nan) for k in other.node_id])
    for i in np.flatnonzero(openish):
        if not np.isfinite(eb[i]):
            continue
        far = oname != sname[i]
        d = np.hypot(ox[far] - ex[i], oy[far] - ny[i])
        if not len(d) or d.min() > G["open_cross_dist_m"]:
            continue
        j = d.argmin()
        ca = oax[far][j]
        if not np.isfinite(ca):
            continue
        # Compare against the cross street's AXIS, not the direction to its
        # nearest node. The question is whether the probe is looking ALONG
        # that street -- which is empty for 90 m because it is a corridor --
        # or ACROSS it, which can be genuinely open. The bearing to a single
        # node answers neither: it depends on where that node happens to sit.
        # n00030 reads 18.4 deg to the nearest Park Avenue tunnel node and was
        # demoted, but 84.8 deg to that road's axis. Its empty side is the
        # East River, and the imagery shows open water to the horizon.
        along = abs((eb[i] - ca + 90) % 180 - 90)
        if along < G["open_cross_angle_deg"]:
            openish[i] = False        # falls through to the segment fill
    source[openish] = "open_one_side"

    R = G["hw_fill_radius_m"]
    fill_stat = G.get("hw_fill_stat", "mean")
    todo = ~np.isfinite(hw) & ~openish
    seg = metrics._street.to_numpy()

    # Order every street along its own axis once. H/W is a property of a
    # street section, so both the radius pool and the fallback walk the same
    # ordered series rather than treating nodes as a cloud of points.
    order = {}
    for s_ in pd.unique(seg):
        k = np.flatnonzero(seg == s_)
        a = np.nanmedian([axis.get(metrics.node_id.iloc[i], np.nan) for i in k])
        if not np.isfinite(a) or len(k) < 2:
            order[s_] = k
            continue
        ar = np.radians(a)
        order[s_] = k[np.argsort(ex[k] * np.sin(ar) + ny[k] * np.cos(ar))]

    for i in np.flatnonzero(todo):
        same = (seg == seg[i]) & np.isfinite(hw)
        if not same.any():
            continue
        d = np.hypot(ex[same] - ex[i], ny[same] - ny[i])
        near = hw[same][d <= R]
        if len(near):
            filled[i] = (near.mean() if fill_stat == "mean"
                         else float(np.median(near)))
            source[i] = "radius_mean"
            continue

        # Nothing measured within the radius, which happens whenever the
        # neighbours failed too -- every node inside a run of failures has an
        # unmeasured neighbour by construction. Walk the street's series
        # outward from this node and take the first measured value found in
        # either direction, nearer side winning. Every street here has at
        # least one measured node, so this always resolves.
        k = order[seg[i]]
        pos = int(np.flatnonzero(k == i)[0])
        step = 1
        while True:
            lo = pos - step
            hi = pos + step
            got = None
            if lo >= 0 and np.isfinite(hw[k[lo]]):
                got = hw[k[lo]]
            if hi < len(k) and np.isfinite(hw[k[hi]]):
                # nearer side wins; on a tie take their mean
                if got is None:
                    got = hw[k[hi]]
                else:
                    got = float(np.mean([got, hw[k[hi]]]))
            if got is not None:
                filled[i] = float(got)
                source[i] = "series"
                break
            if lo < 0 and hi >= len(k):
                break
            step += 1

    metrics["HW_effective"], metrics["HW_source"] = filled, source
    metrics = metrics.drop(columns="_street")
    for tag, note in (("measured", "measured wall to wall"),
                      ("radius_mean", "mean of measured nodes within the radius"),
                      ("series", "nearest measured node along the street"),
                      ("open_one_side", "no opposite wall; not a canyon"),
                      ("none", "still absent")):
        print(f"H/W  {int((source == tag).sum()):>4}  {tag:<15} {note}")
    print(f"H/W  {int((source != 'none').sum()):>4}  usable "
          f"({int((source != 'none').sum()) / len(source) * 100:.0f}% of "
          f"{len(source)} nodes)")

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
