"""
Field of view as an analysis parameter: is 180 degrees the right choice?

WHY THIS EXISTS
---------------
The azimuthal profiles were built so that field of view would not be baked
into data collection. Any centre bearing and any FOV is a sum over a
contiguous range of bins, so the choice can be interrogated after the fact
instead of defended as a collection decision. Nothing here re-fetches or
re-segments; it is arithmetic over `azimuth_profiles.npz`.

180 degrees is defended in the write-up as "a pedestrian does not look
behind". That is a claim about human vision, and it is worth checking what
it costs and what it buys before the paper rests on it.

WHAT 180 ACTUALLY CORRESPONDS TO
--------------------------------
The human binocular field is roughly 120 degrees; the full field including
monocular periphery is roughly 200-220 degrees horizontally, though acuity
outside the central 60 is very low. So 180 is not a physiological constant.
It is a convenient half-circle that happens to sit inside the plausible
range. The honest defence is that the result does not depend on the exact
number -- which is testable, and is what this script tests.

THE THREE QUESTIONS
-------------------
1. LEVEL. How much do GVI and VEI shift as FOV widens? A monotone drift is
   expected and uninteresting; what matters is whether it is large relative
   to the effects being reported.

2. RANK STABILITY. Do nodes, and more importantly block faces, keep their
   ordering? If the ranking of streets is the same at 120 and at 240, the
   substantive conclusions do not depend on the choice. Spearman rho
   against the 180 reference is the measure.

3. CONCLUSION STABILITY. Does the reported relationship change? The
   directional GVI ~ VEI fits are re-run at each FOV. If a finding appears
   only at one width, it is a specification artifact.

A NOTE ON THE COMPLEMENTARITY THAT ONLY HOLDS AT 180
-----------------------------------------------------
At exactly 180, N and S are disjoint halves that tile the circle, and so
are E and W. At any other width they overlap (below 180 they leave a gap).
That is a genuine property of the 180 choice and it is why the four views
behave as two partitions rather than four samples. Widening past 180 makes
opposite views share bins, which manufactures correlation between them --
worth knowing before reading the rho table at 240.

    python tools/fov_check.py
    python tools/fov_check.py --fovs 90 120 180 240 --metric GVI
"""
import argparse, sys
from pathlib import Path

import numpy as np, pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import CFG, PROC, RES, DIRECTIONS, banner, slice_metrics

REFERENCE_FOV = 180.0


def metrics_at(prof, fov):
    """GVI and VEI for every node under every travel direction, at one FOV."""
    rows = []
    for nid, p in prof.items():
        for lab, b in DIRECTIONS.items():
            gvi, vei = slice_metrics(p, b, fov)
            rows.append({"node_id": nid, "direction": lab, "fov": fov,
                         "GVI": gvi, "VEI": vei})
        gvi, vei = slice_metrics(p, 0.0, 360.0)
        rows.append({"node_id": nid, "direction": "full360", "fov": 360.0,
                     "GVI": gvi, "VEI": vei})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fovs", type=float, nargs="*",
                    default=[60, 90, 120, 150, 180, 210, 240, 300])
    ap.add_argument("--metric", default="GVI", choices=["GVI", "VEI"])
    a = ap.parse_args()

    banner("field of view sensitivity")
    npz = PROC / "azimuth_profiles.npz"
    if not npz.exists():
        sys.exit(f"{npz} not found -- run stage 3, or copy it across")
    z = np.load(npz)
    prof = {k: z[k] for k in z.files}
    print(f"{len(prof)} profiled nodes, no imagery needed")

    import geopandas as gpd
    faces = None
    mp = PROC / "metrics.gpkg"
    if mp.exists():
        m = gpd.read_file(mp)
        if "face_id" in m.columns:
            faces = m.set_index("node_id")[["face_id", "osm_name",
                                            "typology"]]

    print(f"\nreference FOV = {REFERENCE_FOV:.0f} deg\n")
    ref = metrics_at(prof, REFERENCE_FOV)
    refw = ref[ref.direction.ne("full360")].pivot_table(
        index=["node_id", "direction"], values=a.metric)

    # ------------------------------------------------------------ 1. level
    print("=" * 74)
    print(f"1. LEVEL -- median {a.metric} by FOV, across all four directions")
    print("=" * 74)
    print(f"  {'FOV':>6s} {'median':>9s} {'mean':>9s} {'sd':>8s} "
          f"{'vs 180':>9s}   {'coverage'}")
    print("  " + "-" * 62)
    allm = {}
    for f in a.fovs:
        d = metrics_at(prof, f)
        d = d[d.direction.ne("full360")]
        allm[f] = d
        med = d[a.metric].median()
        rel = med - ref[ref.direction.ne("full360")][a.metric].median()
        # Four windows of width f centred 90 deg apart: total angular
        # coverage counting overlap, as a multiple of the circle.
        cov = 4 * f / 360
        note = ("gap between adjacent views" if f < 90 else
                "adjacent views overlap" if f > 90 else "adjacent views abut")
        star = "  <- reference" if f == REFERENCE_FOV else ""
        print(f"  {f:6.0f} {med:9.3f} {d[a.metric].mean():9.3f} "
              f"{d[a.metric].std():8.3f} {rel:+9.3f}   {cov:.2f}x  {note}{star}")
    print("\n  Coverage is 4 x FOV / 360. Only at FOV=90 do the four windows")
    print("  tile the circle once. At 180 each pair (N/S, E/W) tiles it, so")
    print("  the four views are two complete partitions -- which is the")
    print("  property the directional analysis relies on.")

    # --------------------------------------------------- 2. rank stability
    from scipy.stats import spearmanr
    print("\n" + "=" * 74)
    print(f"2. RANK STABILITY -- Spearman rho against the 180 deg values")
    print("=" * 74)
    print(f"  {'FOV':>6s} {'node rho':>10s} {'face rho':>10s} "
          f"{'street rho':>11s}")
    print("  " + "-" * 42)
    for f in a.fovs:
        d = allm[f].pivot_table(index=["node_id", "direction"],
                                values=a.metric)
        j = refw.join(d, lsuffix="_ref", rsuffix="_f").dropna()
        rn, _ = spearmanr(j[f"{a.metric}_ref"], j[f"{a.metric}_f"])
        rf = rs = np.nan
        if faces is not None:
            k = (allm[f].join(faces, on="node_id")
                 .groupby("face_id")[a.metric].median())
            kr = (ref[ref.direction.ne("full360")].join(faces, on="node_id")
                  .groupby("face_id")[a.metric].median())
            jj = pd.concat([kr, k], axis=1).dropna()
            if len(jj) > 5:
                rf, _ = spearmanr(jj.iloc[:, 0], jj.iloc[:, 1])
            s = (allm[f].join(faces, on="node_id")
                 .groupby("osm_name")[a.metric].median())
            sr = (ref[ref.direction.ne("full360")].join(faces, on="node_id")
                  .groupby("osm_name")[a.metric].median())
            js = pd.concat([sr, s], axis=1).dropna()
            if len(js) > 5:
                rs, _ = spearmanr(js.iloc[:, 0], js.iloc[:, 1])
        star = "  <- reference" if f == REFERENCE_FOV else ""
        print(f"  {f:6.0f} {rn:10.4f} {rf:10.4f} {rs:11.4f}{star}")
    print("\n  Street- and face-level rho are the ones that matter. If the")
    print("  ordering of streets survives from 120 to 240, no substantive")
    print("  conclusion depends on the exact width and the 180 choice needs")
    print("  no physiological defence -- only a stated convention.")

    # --------------------------------------------- 3. conclusion stability
    print("\n" + "=" * 74)
    print("3. CONCLUSION STABILITY -- GVI ~ VEI at each FOV")
    print("=" * 74)
    if faces is None:
        print("  metrics.gpkg has no face_id -- skipping (run s04 first)")
    else:
        import statsmodels.api as sm
        print(f"  {'FOV':>6s} {'direction':11s} {'slope':>9s} {'p':>10s} "
              f"{'R2':>8s} {'rho':>8s}")
        print("  " + "-" * 58)
        rows = []
        for f in a.fovs:
            d = allm[f].join(faces, on="node_id").dropna(
                subset=["GVI", "VEI", "face_id"])
            for lab in DIRECTIONS:
                dd = d[d.direction.eq(lab)]
                if len(dd) < 40:
                    continue
                r = sm.OLS(dd.GVI, sm.add_constant(dd[["VEI"]])).fit(
                    cov_type="cluster", cov_kwds={"groups": dd.face_id})
                rho, _ = spearmanr(dd.VEI, dd.GVI)
                rows.append({"fov": f, "direction": lab,
                             "slope": r.params.iloc[1],
                             "p": r.pvalues.iloc[1], "r2": r.rsquared,
                             "rho": rho, "n": len(dd)})
                star = " <-" if f == REFERENCE_FOV else ""
                print(f"  {f:6.0f} {lab:11s} {r.params.iloc[1]:9.2f} "
                      f"{r.pvalues.iloc[1]:10.2e} {r.rsquared:8.4f} "
                      f"{rho:+8.3f}{star}")
        tab = pd.DataFrame(rows)
        (RES / "tables").mkdir(parents=True, exist_ok=True)
        tab.to_csv(RES / "tables" / "fov_sensitivity.csv", index=False)
        print(f"\n  wrote {RES/'tables'/'fov_sensitivity.csv'}")
        print("\n  Read `rho` down each direction. If it is flat across FOV")
        print("  while `slope` and `p` move, the movement is leverage from")
        print("  the low-VEI tail, not a width effect -- the same diagnosis")
        print("  that applies across directions at fixed width.")
        if len(tab):
            w = tab.pivot_table(index="direction", columns="fov", values="rho")
            print("\n  rho by direction x FOV:")
            print("  " + w.round(3).to_string().replace("\n", "\n  "))


if __name__ == "__main__":
    main()
