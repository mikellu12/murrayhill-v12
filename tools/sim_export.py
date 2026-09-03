"""Split the SIM run into two tables: what was observed, and what was derived.

One computed frame holds both, which is convenient for code and confusing for a
reader -- a raw 1-7 rating and a sigmoid output sit in adjacent columns with
nothing saying which is which. These two are keyed on `file` and join
one-to-one.

  vlm_observations.csv      one row per half-view. The ten VLM ratings exactly as
                        returned, the pixel shares measured over the same 90
                        degrees, and the canyon geometry. Nothing here is
                        computed from anything else here.

  vlm_calculations.csv  the same rows, every intermediate the manuscript's
                        section 2.7 defines, in the order it is applied:
                        normalise, compose, threshold, discount, combine.

Every arc share is measured over the exact 90 degrees the half-view was
rendered from, not over the node -- a half facing a blank wall while the trees
stand opposite should read low, and a node-level share would count them.

    .venv/Scripts/python tools/sim_export.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
from common import CFG, PROC, RES, banner, bin_mask
from sim_fields import FIELDS
from sim_readout import prune_once, interpolated_median
from half_target import walk_bearings, SIDE_OFF

RATINGS = list(FIELDS)
ID = ["file", "node_id", "street", "walk", "side", "seq", "cardinal"]


def main():
    """Standalone entry point -- recompute, then split."""
    import sim_compute
    sim_compute.main()


def write_split(sc, dropped=None):
    """Split a computed frame into the two study tables and write them.

    `dropped` carries the raw rating rows of nodes tagged usable: False. They
    are OBSERVATIONS -- the model rated the frame -- so they appear in
    vlm_observations with the tag, and never in vlm_calculations: a tourist
    bus interior must be visible in the record and invisible to the score.
    """
    banner("split the run into observed and derived")
    sc = sc.copy()
    sc["usable"] = True
    if dropped is not None and len(dropped):
        dropped = dropped.copy()
        dropped["usable"] = False
        sc = pd.concat([sc, dropped], ignore_index=True)
    reason_map = {}
    npath = PROC / "nodes.csv"
    if npath.exists():
        nu = pd.read_csv(npath)
        if "exclude_reason" in nu.columns:
            reason_map = dict(zip(nu.node_id, nu.exclude_reason.fillna("")))
    sc["exclude_reason"] = sc.node_id.map(reason_map).fillna("")

    # THE MEDIAN, NOT JUST round(EV) AND argmax. The bare column used to be the
    # rounded expected value with nothing in its name saying so -- a reader
    # opening the file could not tell it apart from a raw survey answer.
    # interpolated_median(prune_once(p)) is what the study actually builds M
    # from (sim_compute.rung); it belongs in the table that claims to hold
    # every observation, not only inside the function that consumes it once
    # and discards it.
    # median_round is what the interface displays and median is what M is
    # built from; they lead, because the reader should meet the readout the
    # study uses before its alternatives. Both EV columns are gone -- round(EV)
    # was the deprecated readout and keeping it "for continuity" next to the
    # median just invited picking the wrong column.
    for f in RATINGS:
        pcols = [f"{f}_p{k}" for k in range(1, 8)]
        if all(c in sc.columns for c in pcols):
            P = sc[pcols].to_numpy(float)
            P = P / P.sum(axis=1, keepdims=True)
            med = interpolated_median(prune_once(P))
            sc[f + "_median"] = med
            sc[f + "_median_round"] = np.clip(np.round(med), 1, 7).astype(int)

    # Both of the joins below are Murray Hill artefacts: metrics.csv comes from
    # s05's footprint geometry and azimuth_profiles.npz from s03's
    # segmentation. A frame without footprints has neither, and the ratings and
    # the computed M are still the deliverable, so the geometry columns are
    # filled with NaN rather than the split being skipped. The two files then
    # have the same shape in every study area, which is what makes them
    # stackable for a cross-city table.
    mpath, zpath = PROC / "metrics.csv", PROC / "azimuth_profiles.npz"
    geo = mpath.exists() and zpath.exists()
    if not geo:
        missing = [q.name for q in (mpath, zpath) if not q.exists()]
        print(f"no {', '.join(missing)}: writing ratings and M without the "
              f"footprint geometry and arc shares")

    GEO_COLS = ["typology", "osm_name", "face_id", "H_m", "W_facade",
                "GVI", "VEI", "SVF_band"]
    ARC = ["arc_vegetation", "arc_sky", "arc_building"]
    if not geo:
        for c in GEO_COLS:
            if c not in sc.columns:
                sc[c] = np.nan
        for c in ARC:
            sc[c] = np.nan
        if "osm_name" in sc.columns and sc.osm_name.isna().all():
            sc["osm_name"] = sc.street
        return _write(sc)

    met = pd.read_csv(mpath)

    # ---- arc shares, measured over each half-view's own 90 degrees ---------
    z = np.load(zpath)
    prof = {k: z[k] for k in z.files}
    bear = walk_bearings()
    parts = sc.file.str.split("/", expand=True)
    # not _st/_wk: itertuples() renames any leading-underscore column to a
    # positional _1, _2 and the attribute lookup below fails silently late.
    sc["dir_street"], sc["dir_walk"] = parts[0], parts[1]
    cols = {0: "arc_vegetation", 1: "arc_sky", 2: "arc_building"}
    acc = {v: [] for v in cols.values()}
    for r in sc.itertuples():
        p, b = prof.get(r.node_id), bear.get((r.dir_street, r.dir_walk))
        if p is None or b is None or r.side not in SIDE_OFF:
            for v in acc.values():
                v.append(np.nan)
            continue
        m = bin_mask((b + SIDE_OFF[r.side]) % 360, 90.0)
        W = p[3][m].sum()
        for i, name in cols.items():
            acc[name].append(p[i][m].sum() / W if W > 0 else np.nan)
    for k, v in acc.items():
        sc[k] = v

    extra = met[["node_id", "typology", "osm_name", "face_id", "H_m",
                 "W_facade", "GVI", "VEI", "SVF_band"]]
    sc = sc.merge(extra, on="node_id", how="left", suffixes=("", "_node"))
    return _write(sc)


def _write(sc):
    """The split itself, once the optional geometry columns are present."""
    # ---- observed ---------------------------------------------------------
    lead = [c for f in RATINGS
            for c in (f + "_median_round", f + "_median")
            if c in sc.columns]
    obs = sc[ID + ["usable", "exclude_reason",
                   "osm_name", "typology", "face_id"] + lead
             + ["arc_vegetation", "arc_sky", "arc_building",
                "H_m", "W_facade", "HW_facade", "HW_effective", "HW_source",
                "GVI", "VEI", "SVF_band"]].copy()
    obs = obs.rename(columns={"GVI": "node_GVI", "VEI": "node_VEI",
                              "SVF_band": "node_SVF_band"})

    # The other three readings of the same VLM answer ride along at the end.
    # `round(EV)` above is the survey response; the expected value it rounds
    # from, the argmax a generate() call would have written, and the full
    # distribution over the seven rungs are all observations too, so they
    # belong here rather than in the calculations. Placed last so opening the
    # file shows the ratings and the geometry without scrolling past 90
    # columns nobody reads day to day.
    tail = ([f + "_argmax" for f in RATINGS]
            + [f"{f}_p{k}" for f in RATINGS for k in range(1, 8)])
    have = [c for c in tail if c in sc.columns]
    obs = pd.concat([obs, sc[have]], axis=1)
    # The interior flag rides along when it exists. Anyone working from this
    # table would otherwise include station subways and shop interiors without
    # knowing they are there. It is a candidate flag, not a verdict: measured
    # against 16 hand-labelled frames it runs 0.70 precision and 0.70 recall,
    # so it wrongly flags roofed public streets and misses about a third of
    # true interiors. Filter on it only after checking the frames.
    ipath = PROC / "indoor_flag.csv"
    if ipath.exists():
        fl = pd.read_csv(ipath)[["file", "indoor", "sky_share", "road_share",
                                 "ceiling_share"]]
        obs = obs.merge(fl, on="file", how="left")
        n = int(obs.indoor.fillna(False).sum())
        print(f"  indoor flag merged: {n} of {len(obs)} frames "
              f"({n/len(obs)*100:.1f}%) -- UNVALIDATED, see the docstring")

    obs = obs.sort_values(["osm_name", "walk", "seq", "side"])
    print(f"  readouts per field, in order: <field>_median_round (the rung "
          f"the interface shows), <field>_median (what M is built from), "
          f"then <field>_argmax and the full <field>_p1.._p7 distribution")

    # ---- derived ----------------------------------------------------------
    # The alternative calibrations and the un-penalised score ride along.
    # M alone is not enough to compare two cities: M_noA is the like-for-like
    # column, because a study area without building heights cannot compute A_i
    # at all and its M is I^a Y^b D^c with nothing discounting it. Reading one
    # city's M against another's M_noA compares the presence of the term, not
    # the streets -- which is exactly the mistake this column exists to stop,
    # and it was unavailable to anyone reading the exported table.
    EXTRA = [c for c in ("M_local", "M_noA", "M_local_noA",
                         "I_local", "D_local", "Omega_local")
             if c in sc.columns]
    sc = sc[sc.usable].copy()   # calculations never see an unusable frame
    calc = sc[ID + ["nat_built", "GVI_eye", "GMI", "V_sign", "SVF", "SFV",
                    "V_pave", "GFAPI", "IAS", "I_raw", "I", "Y", "D_raw", "D",
                    "HW_effective", "Omega", "a", "b", "c", "M"]
              + EXTRA].copy()
    calc = calc.sort_values(["file"])
    calc = calc.reindex(columns=ID + [
        # normalised inputs, (r-1)/6 except nat_built which is a raw ratio
        "nat_built", "GVI_eye", "GMI", "V_sign", "SVF", "SFV", "V_pave", "GFAPI",
        "IAS",
        # composed, thresholded
        "I_raw", "I", "Y", "D_raw", "D",
        # discount and elasticities
        "HW_effective", "Omega", "a", "b", "c",
        "M"] + EXTRA)

    # The city rides in a column so the two areas can be concatenated without
    # the reader having to remember which file was which.
    name = CFG.get("study_area_name", "")
    if name:
        obs.insert(0, "city", name)
        calc.insert(0, "city", name)

    out = RES / "tables"
    obs.to_csv(out / "vlm_observations.csv", index=False)
    calc.to_csv(out / "vlm_calculations.csv", index=False)

    # ...and again under a name that survives leaving the repository. Ten tools
    # read the plain names, so those stay canonical; these are the copies to
    # hand to somebody. Written in the same call, so they cannot drift.
    slug = CFG.get("study_area_slug")
    if slug:
        obs.to_csv(out / f"vlm_observations_{slug}.csv", index=False)
        calc.to_csv(out / f"vlm_calculations_{slug}.csv", index=False)
        print(f"shareable copies: vlm_observations_{slug}.csv, "
              f"vlm_calculations_{slug}.csv")

    print(f"vlm_observations.csv        {len(obs):>5} rows x {obs.shape[1]:>2} cols")
    print(f"  identity            {', '.join(ID)}")
    print(f"  VLM ratings, 1-7    <field>_median_round / _median for: "
          f"{', '.join(RATINGS)}")
    print(f"  measured over arc   arc_vegetation, arc_sky, arc_building")
    print(f"  canyon geometry     H_m, W_facade, HW_facade, HW_effective, HW_source")
    print(f"  node-level context  node_GVI, node_VEI, node_SVF_band")
    print(f"\nvlm_calculations.csv  {len(calc):>5} rows x {calc.shape[1]:>2} cols")
    print(f"  normalised          nat_built, GVI_eye, GMI, V_sign, SVF, SFV,")
    print(f"                      V_pave, GFAPI, IAS")
    print(f"  dimensions          I_raw, I, Y, D_raw, D")
    print(f"  discount, weights   HW_effective, Omega, a, b, c")
    print(f"  score               M")
    print(f"\n  join on `file`. {int(calc.M.notna().sum())} rows carry a score; "
          f"{int(calc.M.isna().sum())} have no H/W.")

    # ---- sections ---------------------------------------------------------
    # Built here rather than only inside sim_section_map.py, which wrote it as
    # a side effect of drawing the figure. That meant it refreshed only when
    # someone made the map: it sat two days stale beside two current tables,
    # carrying an M range of 0.129-0.826 against the live 0.242-0.947. The
    # `sections()` rule is imported, not copied, so there is still one
    # definition of where a section starts and ends.
    try:
        import geopandas as gpd
        from sim_section_map import sections, UTM
        g = gpd.read_file(PROC / "metrics.gpkg").to_crs(UTM).reset_index(drop=True)
        g = g[g.node_id.isin(set(calc.node_id))].reset_index(drop=True)
        g = sections(g, 15.0)
        g = g.merge(calc.groupby("node_id").M.mean().rename("M"),
                    on="node_id", how="left")
        agg = (g[g.section > 0].groupby("section")
               .agg(street=("osm_name", "first"), n=("M", "count"),
                    M=("M", "mean"), sd=("M", "std"))
               .reset_index())
        agg = agg[agg.n > 0]
        agg.to_csv(out / "vlm_sections.csv", index=False)
        print(f"\nvlm_sections.csv      {len(agg):>5} rows  "
              f"(section, street, n, M, sd)")
        print(f"  one block-length run of a street between crossings; "
              f"{int((g.section < 0).sum())} nodes sit at a crossing and "
              f"belong to none")
    except Exception as e:
        print(f"\nvlm_sections.csv      SKIPPED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
