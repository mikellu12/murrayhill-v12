"""Stage 2 -- Street View metadata and imagery.

The metadata endpoint is free and unlimited, so coverage and capture date
are checked before a single paid request. It also reports how far Google's
nearest panorama sits from each node: a pano 40 m away is sampling a
different part of the street, and accepting it silently destroys the
regular 20 m spacing Stage 1 guarantees.

Imagery is requested by pano_id, never by coordinate. A coordinate request
can snap to a different panorama for each heading, which corrupts the
four-way sum in the GVI denominator.
"""
import sys
import pandas as pd, geopandas as gpd, requests
from pathlib import Path
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from common import PROJ_CRS, CFG, PROC, RAW, IMG, banner, require

SP, CAP = CFG["sampling"], CFG["capture"]
META = "https://maps.googleapis.com/maps/api/streetview/metadata"
SV = "https://maps.googleapis.com/maps/api/streetview"


def main():
    banner("STAGE 2  metadata and imagery")
    key = require("GMAPS_KEY")
    nodes = gpd.read_file(PROC / "nodes.gpkg")

    mpath = RAW / "metadata.csv"
    meta = None
    if mpath.exists():
        cached = pd.read_csv(mpath)
        # A cache from a different frame is worse than no cache: node IDs
        # restart at n00000 whenever the frame is rebuilt, so stale rows
        # would silently pair each node with another location's panorama.
        same = (len(cached) == len(nodes)
                and set(cached.node_id) == set(nodes.node_id))
        if same:
            meta = cached
            print(f"metadata cached: {len(meta)} rows")
        else:
            print(f"metadata cache is from a different frame "
                  f"({len(cached)} rows vs {len(nodes)} nodes) -- re-probing. "
                  f"Metadata requests are free.")
    if meta is None:
        rows = []
        for _, r in tqdm(list(nodes.iterrows()), desc="metadata",
                         mininterval=2.0):
            try:
                js = requests.get(META, params={"location": f"{r.lat},{r.lon}",
                                                "source": "outdoor",
                                                "key": key}, timeout=15).json()
            except Exception as e:
                js = {"status": f"ERROR {e}"}
            loc = js.get("location") or {}
            rows.append({"node_id": r.node_id, "status": js.get("status"),
                         "pano_id": js.get("pano_id"), "pano_date": js.get("date"),
                         "pano_lat": loc.get("lat"), "pano_lon": loc.get("lng"),
                         "node_lat": r.lat, "node_lon": r.lon})
        meta = pd.DataFrame(rows)
        meta.to_csv(mpath, index=False)

    dt = pd.to_datetime(meta.pano_date, format="%Y-%m", errors="coerce")
    meta["month"], meta["year"] = dt.dt.month, dt.dt.year

    g = meta.dropna(subset=["pano_lat"])
    if len(g):
        a = gpd.GeoSeries(gpd.points_from_xy(g.pano_lon, g.pano_lat),
                          crs=4326).to_crs(PROJ_CRS)
        b = gpd.GeoSeries(gpd.points_from_xy(g.node_lon, g.node_lat),
                          crs=4326).to_crs(PROJ_CRS)
        meta.loc[g.index, "pano_offset_m"] = a.distance(b, align=False).values

    ok = meta.status.eq("OK")
    # `target` pins a single capture campaign, which is what Murray Hill has.
    # `months` is the looser option for a frame with no single campaign. Both
    # are opt-in: with neither set every node with imagery is kept, and the
    # capture spread is reported rather than filtered. pano_date rides along in
    # the manifest either way, so an analysis can condition on it after the
    # fact instead of losing the nodes here.
    if CAP.get("target"):
        on_date = meta.pano_date.eq(CAP["target"])
    elif CAP.get("months"):
        lo, hi = CAP["months"]
        on_date = meta.month.between(lo, hi)
        print(f"capture window: months {lo}-{hi} "
              f"({int((meta.month.between(lo, hi) & ok).sum())} nodes)")
    else:
        on_date = True
    near = meta.pano_offset_m.le(SP["max_pano_offset_m"])
    meta["usable"] = ok & on_date & near
    # Drop columns carried in from a cache written AFTER a previous merge;
    # re-merging them produces typology_x / typology_y and the plain name
    # disappears.
    meta = meta.drop(columns=[c for c in ["typology", "osm_name", "northing_m", "zone"]
                              if c in meta.columns])
    meta = meta.merge(nodes[["node_id", "typology", "osm_name", "northing_m"]],
                      on="node_id", how="left")
    meta.to_csv(mpath, index=False)

    print("\n--- sample flow ---")
    print(f"  nodes in frame              {len(meta)}")
    print(f"  with coverage               {int(ok.sum())}")
    print(f"  captured {CAP['target']}            {int((ok & on_date).sum())}")
    print(f"  pano within {SP['max_pano_offset_m']} m           "
          f"{int(meta.usable.sum())}   <- analytic n")
    print("\npano offset (m):")
    print(meta.loc[ok, "pano_offset_m"].describe(
        percentiles=[.5, .9, .99]).round(1).to_string())
    print("\ncoverage by typology:")
    print(meta.groupby("typology").usable.agg(["mean", "size"]).round(3).to_string())
    print("\nall captures available:")
    print(pd.crosstab(meta.loc[ok, "year"], meta.loc[ok, "month"]).to_string())

    if meta.usable.sum() == 0:
        if (meta.status == "REQUEST_DENIED").any():
            print("\nREQUEST_DENIED -> Street View Static API not enabled, or "
                  "no billing account, or the key is restricted.")
        elif (meta.status == "ZERO_RESULTS").all():
            print("\nZERO_RESULTS everywhere -> bbox is wrong.")
        sys.exit("no usable nodes")

    # Imagery is named by node_id, and Stage 1 reassigns those from n00000
    # on every rebuild. Files from an older frame would be silently reused
    # for the wrong locations, so refuse to proceed with a mismatched cache.
    stale = [p for p in IMG.glob("*.jpg")
             if p.stem.rsplit("_", 1)[0] not in set(nodes.node_id)]
    if stale:
        sys.exit(f"{len(stale)} images in {IMG} belong to node IDs that are "
                 f"not in the current frame. Delete {IMG} and re-run.")

    # Off-target captures are excluded by default because mixing capture dates
    # mixes seasons into the greenery variables the study measures. Enabled
    # deliberately here: 29 of the 31 affected nodes are leaf-on (May-Sep), so
    # the seasonal objection applies to two nodes rather than all of them, and
    # the built morphology those nodes contribute is stable across the gap.
    # pano_date rides along in the manifest so any analysis can exclude them.
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-offtarget", action="store_true")
    args, _ = ap.parse_known_args()

    use = meta[meta.usable]
    if args.include_offtarget:
        extra = meta[ok & near & ~meta.usable]
        if len(extra):
            print(f"\n--include-offtarget: adding {len(extra)} node(s) whose "
                  f"panorama is not {CAP['target']}")
            print(extra.pano_date.value_counts().to_string())
            use = pd.concat([use, extra], ignore_index=True)
    # Count what is actually missing, not len(use)*4. The loop skips any file
    # already on disk, so quoting the full total reads as a bill for a re-fetch
    # that never happens -- 3,064 requests announced against 124 performed.
    todo = sum(1 for _, r in use.iterrows() for h in SP["headings"]
               if not (IMG / f"{r.node_id}_{h:03d}.jpg").exists())
    cached = len(use) * len(SP["headings"]) - todo
    print(f"\nimagery: {todo} request(s) to make, {cached} already cached "
          f"(${todo * 0.007:.2f} at list price; free tier covers 10,000/month)")

    man, failed = [], 0
    for _, r in tqdm(list(use.iterrows()), desc="imagery", mininterval=2.0):
        for h in SP["headings"]:
            fp = IMG / f"{r.node_id}_{h:03d}.jpg"
            if not fp.exists():
                try:
                    resp = requests.get(SV, params={
                        "pano": r.pano_id, "size": f"{SP['image_size']}x{SP['image_size']}",
                        "heading": h, "fov": SP["fov"], "pitch": SP["pitch"],
                        "key": key}, timeout=30)
                except Exception:
                    failed += 1
                    continue
                if resp.status_code != 200 or len(resp.content) < 5000:
                    failed += 1
                    if failed <= 3:
                        print(f"\n  HTTP {resp.status_code}: {resp.content[:120]}")
                    continue
                fp.write_bytes(resp.content)
            man.append({"node_id": r.node_id, "heading": h, "path": str(fp),
                        "pano_date": r.pano_date,
                        "on_target": bool(r.pano_date == CAP["target"])})

    mf = pd.DataFrame(man)
    mf.to_csv(PROC / "manifest.csv", index=False)
    print(f"{len(mf)} images across {mf.node_id.nunique()} nodes, "
          f"{failed} failed")
    if not len(mf):
        sys.exit("no imagery downloaded")

    # Google's ToS restricts caching Street View imagery beyond 30 days.
    # Retain the derived profiles and metrics; treat the JPEGs as scratch.


if __name__ == "__main__":
    main()
