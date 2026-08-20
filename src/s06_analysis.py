"""Stage 6 -- regression and group contrasts.

Everything inferential lives here, and everything inferential accounts for
spatial dependence. Nodes 20 m apart on the same block face photograph
nearly the same scene; treating them as independent inflates significance
by roughly the square root of the cluster size.

Two honest options, both reported: standard errors clustered by block face
(keeps all nodes in the model, widens the interval), and OLS on block-face
medians (fewer units, all independent). The naive version is printed only
to show the size of the gap.
"""
import sys
import numpy as np, pandas as pd, geopandas as gpd
from pathlib import Path
from scipy.stats import (kruskal, mannwhitneyu, friedmanchisquare,
                         spearmanr, t as stats_t)

sys.path.insert(0, str(Path(__file__).parent))
from common import CFG, PROC, RES, banner, DIRECTIONS, TYPOLOGY_ORDER


def fit(d, y="GVI", x="VEI", cluster=None):
    import statsmodels.api as sm
    d = d.dropna(subset=[y, x])
    m = sm.OLS(d[y], sm.add_constant(d[[x]]))
    if cluster is not None:
        return m.fit(cov_type="cluster", cov_kwds={"groups": d[cluster]}), d
    return m.fit(), d


def partial_spearman(d, x, y, z):
    """Spearman rho between x and y with z partialled out.

    Standard first-order partial correlation on the rank correlations:
    (r_xy - r_xz r_yz) / sqrt((1 - r_xz^2)(1 - r_yz^2)). Rank-based rather
    than Pearson-based because the marginal relationships here are driven
    by a handful of extreme-GVI nodes, which is exactly the case where a
    Pearson partial reports the leverage rather than the association.
    """
    d = d.dropna(subset=[x, y, z])
    if len(d) < 8:
        return np.nan, np.nan, np.nan, len(d)
    rxy = spearmanr(d[x], d[y])[0]
    rxz = spearmanr(d[x], d[z])[0]
    ryz = spearmanr(d[y], d[z])[0]
    den = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    r = (rxy - rxz * ryz) / den if den > 0 else np.nan
    # t on n - 3 df: one covariate partialled out.
    n = len(d)
    t = r * np.sqrt((n - 3) / max(1 - r ** 2, 1e-12))
    p = 2 * stats_t.sf(abs(t), n - 3) if np.isfinite(t) else np.nan
    return rxy, r, p, n


def robust_associations(metrics, faces):
    """OLS, Theil-Sen and Spearman side by side, with face-clustered CIs.

    Three columns that answer three different questions, and they disagree
    here in a way that is itself the result:

      OLS slope     covariance over variance, so points with ordinary x and
                    extreme y drag it. Park Avenue's mall does exactly that.
      Theil-Sen     the median of all pairwise slopes. Half the points would
                    have to move before it does, so it survives Park Avenue
                    without anyone having to decide whether to delete a real
                    street.
      Spearman rho  agreement of rank orders only. Immune to magnitude, and
                    the statistic that was stable while the OLS slope swung
                    fivefold.

    Every interval is bootstrapped over block faces, never over nodes.
    Moran's I is ~0.62 here: resampling nodes would treat 20 m neighbours as
    independent draws and return an interval several times too narrow.
    """
    import statsmodels.api as sm
    from scipy.stats import theilslopes
    rng = np.random.default_rng(CFG["seed"])
    B = CFG["spatial"]["bootstrap_n"]
    rows = []

    def boot_rho(d, x, y):
        # Faces are held as plain arrays and concatenated by index: a
        # pd.concat of 22 frames, 2000 times, per row of the table, is
        # about twenty times slower for an identical answer.
        groups = [(g[x].values, g[y].values) for _, g in d.groupby("face_id")]
        if len(groups) < 5:
            return np.nan, np.nan
        idx = rng.integers(0, len(groups), (B, len(groups)))
        out = np.empty(B)
        for i in range(B):
            xs = np.concatenate([groups[j][0] for j in idx[i]])
            ys = np.concatenate([groups[j][1] for j in idx[i]])
            out[i] = spearmanr(xs, ys)[0]
        return tuple(np.nanpercentile(out, [2.5, 97.5]))

    for unit, src in [("node", metrics), ("face", faces)]:
        for x in ["VEI", "HW_ratio"]:
            if x not in src.columns:
                continue
            for lab, d in [("all", src),
                           ("no Park Ave", src[src.osm_name.ne("Park Avenue")])]:
                d = d.dropna(subset=[x, "GVI"])
                if len(d) < 10:
                    continue
                ols = sm.OLS(d.GVI, sm.add_constant(d[[x]]))
                ols = (ols.fit(cov_type="cluster", cov_kwds={"groups": d.face_id})
                       if unit == "node" else ols.fit())
                ts = theilslopes(d.GVI.values, d[x].values, 0.95)
                rho, prho = spearmanr(d[x], d.GVI)
                lo, hi = boot_rho(d, x, "GVI") if "face_id" in d.columns \
                    else (np.nan, np.nan)
                rows.append({"unit": unit, "x": x, "subset": lab, "n": len(d),
                             "ols_slope": ols.params.iloc[1],
                             "ols_p": ols.pvalues.iloc[1],
                             "r2": ols.rsquared,
                             "theilsen_slope": ts[0], "ts_lo": ts[2],
                             "ts_hi": ts[3], "rho": rho, "rho_p": prho,
                             "rho_lo": lo, "rho_hi": hi})
    t = pd.DataFrame(rows)
    t.to_csv(RES / "tables" / "robust_associations.csv", index=False)
    print("\n--- OLS against robust alternatives (CIs bootstrapped over faces) ---")
    print(f"  {'unit':5s} {'x':9s} {'subset':12s} {'n':>4s} {'OLS':>7s} "
          f"{'R2':>6s} {'TheilSen':>9s} {'rho':>6s} {'rho 95% CI':>16s}")
    for _, r in t.iterrows():
        ci = (f"[{r.rho_lo:+.2f}, {r.rho_hi:+.2f}]"
              if pd.notna(r.rho_lo) else "")
        print(f"  {r.unit:5s} {r.x:9s} {r.subset:12s} {int(r.n):4d} "
              f"{r.ols_slope:7.2f} {r.r2:6.3f} {r.theilsen_slope:9.2f} "
              f"{r.rho:+6.2f} {ci:>16s}")
    print("  Where OLS moves between the all and no-Park-Ave rows and")
    print("  Theil-Sen and rho do not, the OLS number was reporting one")
    print("  street's leverage rather than the association.")
    return t


def main():
    banner("STAGE 6  analysis")
    import statsmodels.api as sm

    metrics = gpd.read_file(PROC / "metrics.gpkg")
    dm = pd.read_csv(PROC / "directional_metrics.csv")
    face = pd.read_csv(PROC / "block_faces.csv")

    # ------------------------------------------------ typology contrasts
    print("--- GVI / VEI by typology ---")
    for lvl, df in [("node", metrics), ("block face", face)]:
        print(f"\n{lvl} level (n={len(df)}):")
        print(df.groupby("typology")[["GVI", "VEI"]]
                .agg(["median", "count"]).round(3).to_string())
        for col in ["GVI", "VEI"]:
            grps = [df.loc[df.typology.eq(t), col].dropna()
                    for t in TYPOLOGY_ORDER]
            grps = [g for g in grps if len(g) >= 5]
            if len(grps) >= 2:
                h, p = kruskal(*grps)
                print(f"  {col}: Kruskal-Wallis H={h:.1f} p={p:.2e} "
                      f"({len(grps)} groups)")
            a = df.loc[df.typology.eq("avenue_canyon"), col].dropna()
            b = df.loc[df.typology.eq("mid_block"), col].dropna()
            if len(a) >= 5 and len(b) >= 5:
                u, p = mannwhitneyu(a, b)
                r = 1 - (2 * u) / (len(a) * len(b))
                print(f"       canyon vs mid-block p={p:.2e} r={r:+.3f}")
    print("\n  Compare the two levels. A result that survives aggregation is")
    print("  real; one that vanishes was pseudoreplication.")

    # ------------------------------------------------ north-south gradient
    print("\n--- north-south gradient ---")
    if "northing_m" in metrics.columns:
        d = metrics.dropna(subset=["GVI", "northing_m"])
        r = sm.OLS(d.GVI, sm.add_constant(d[["northing_m"]])).fit(
            cov_type="cluster", cov_kwds={"groups": d.face_id})
        rho, pr = spearmanr(d.northing_m, d.GVI)
        print(f"  GVI ~ metres uptown (bearing 029): "
              f"{100*r.params.iloc[1]:+.2f} pp per 100 m")
        print(f"  R2={r.rsquared:.4f}  p={r.pvalues.iloc[1]:.2e}  "
              f"rho={rho:+.3f}  n={len(d)}  (SEs clustered by face)")
        # block_faces.csv written by a pre-v12 s04 has no northing_m; fall
        # back to averaging the node values rather than failing the stage.
        if "northing_m" not in face.columns and "face_id" in metrics.columns:
            face = face.merge(
                metrics.groupby("face_id").northing_m.mean().reset_index(),
                on="face_id", how="left")
        f = (face.dropna(subset=["GVI", "northing_m"])
             if "northing_m" in face.columns else face.iloc[:0])
        if len(f) > 10:
            rf = sm.OLS(f.GVI, sm.add_constant(f[["northing_m"]])).fit()
            print(f"  block-face level: R2={rf.rsquared:.4f} "
                  f"adj={rf.rsquared_adj:.4f} p={rf.f_pvalue:.3f} n={len(f)}")
        f2 = f[f.osm_name.ne("Park Avenue")]
        if len(f2) > 10:
            r2f = sm.OLS(f2.GVI, sm.add_constant(f2[["northing_m"]])).fit()
            print(f"  excluding Park Avenue: R2={r2f.rsquared:.4f} "
                  f"adj={r2f.rsquared_adj:.4f} p={r2f.f_pvalue:.3f} n={len(f2)}")
            print("    Park Avenue's planted medians sit mid-frame at GVI 14-17.")
            print("    Report both fits; they are a real part of the streetscape.")
        print("  A gradient, not a boundary. Calling it a neighbourhood effect")
        print("  needs a land-use covariate, not a latitude proxy.")
    else:
        print("  no northing_m in metrics -- re-run s01 and s04")

    # ------------------------------------------------ GVI ~ VEI
    print("\n--- GVI ~ VEI, full 360 view ---")
    naive, _ = fit(metrics)
    clust, _ = fit(metrics, cluster="face_id")
    agg, _ = fit(face)
    print(f"{'model':32s} {'slope':>8s} {'95% CI':>19s} {'p':>9s} {'R2':>6s} {'n':>5s}")
    print("-" * 84)
    # Spearman alongside every slope. The slope is cov/var, so the low-VEI
    # tail moves it several-fold; rank correlation asks only whether the
    # greener nodes are the less enclosed ones, and barely shifts. Where the
    # two disagree, the disagreement is the finding.
    gv_rows = []
    face_np = face[face.osm_name.ne("Park Avenue")] if "osm_name" in face else None
    models = [("node OLS (INVALID SEs)", naive, metrics, "node"),
              ("node OLS, clustered by face", clust, metrics, "node"),
              ("block-face medians", agg, face, "face")]
    if face_np is not None and len(face_np) > 5:
        models.append(("block-face medians, no Park Ave",
                       fit(face_np)[0], face_np, "face"))
    for lab, r, src, unit in models:
        lo, hi = r.conf_int().iloc[1]
        dd = src.dropna(subset=["GVI", "VEI"])
        rho, p_rho = spearmanr(dd.VEI, dd.GVI)
        print(f"{lab:32s} {r.params.iloc[1]:8.2f} [{lo:8.2f},{hi:8.2f}] "
              f"{r.pvalues.iloc[1]:9.2e} {r.rsquared:6.3f} {len(dd):5d}"
              f"   rho={rho:+.2f}")
        gv_rows.append({"model": lab, "unit": unit, "n": len(dd),
                        "slope": r.params.iloc[1], "ci_lo": lo, "ci_hi": hi,
                        "p": r.pvalues.iloc[1], "r2": r.rsquared,
                        "spearman_rho": rho, "rho_p": p_rho})
    pd.DataFrame(gv_rows).to_csv(RES / "tables" / "regression_gvi_vei.csv",
                                 index=False)

    # GVI is a bounded proportion with a floor at zero; OLS can predict
    # negative greenness. Agreement in sign and significance is the check.
    d = metrics.dropna(subset=["GVI", "VEI"])
    glm = sm.GLM(d.GVI / 100, sm.add_constant(d[["VEI"]]),
                 family=sm.families.Binomial()).fit(scale="X2")
    print(f"\nquasi-binomial GLM (logit): VEI coef {glm.params.iloc[1]:+.3f} "
          f"p={glm.pvalues.iloc[1]:.2e}")

    if metrics.typology.nunique() > 1:
        X = pd.get_dummies(d[["VEI", "typology"]], columns=["typology"],
                           drop_first=True).astype(float)
        r2 = sm.OLS(d.GVI, sm.add_constant(X)).fit(
            cov_type="cluster", cov_kwds={"groups": d.face_id})
        print(f"with typology controlled: VEI slope {r2.params['VEI']:+.2f} "
              f"p={r2.pvalues['VEI']:.2e}")
        print("  A large drop from the clustered slope means the relationship")
        print("  is mostly between-typology, not enclosure per se.")

    # ------------------------------------------------ directional
    print("\n--- by travel direction ---")
    print(dm.groupby("direction")[["GVI", "VEI"]]
            .agg(["median", "mean", "count"]).round(3).to_string())

    sub = dm[dm.direction.ne("full360") & dm.along_street.notna()]
    if len(sub):
        print("\nalong-street vs cross-street:")
        print(sub.groupby("along_street")[["GVI", "VEI"]]
                 .agg(["median", "count"]).round(3).to_string())
        for col in ["GVI", "VEI"]:
            a = sub.loc[sub.along_street.eq(True), col].dropna()
            b = sub.loc[sub.along_street.eq(False), col].dropna()
            if len(a) > 10 and len(b) > 10:
                u, p = mannwhitneyu(a, b)
                print(f"  {col}: along={a.median():.3f} "
                      f"cross={b.median():.3f} p={p:.2e}")
        print("  Along-street looks down the corridor, cross-street at the")
        print("  facades. The 360 index averages the two together.")

    w = dm.pivot_table(index="node_id", columns="direction", values="GVI")
    dirs = [x for x in DIRECTIONS if x in w.columns]
    ww = w[dirs].dropna()
    if len(ww) > 20:
        st, p = friedmanchisquare(*[ww[x].values for x in dirs])
        sp = ww.max(axis=1) - ww.min(axis=1)
        print(f"\nFriedman across directions (GVI): chi2={st:.1f} p={p:.2e}")
        print(f"within-node GVI spread: median {sp.median():.2f} pp, "
              f"90th pct {sp.quantile(.9):.2f} pp")
        print("  That spread is what a single 360 value hides.")

    # ------------------------------------------------ per view
    # The four views are NOT four independent samples. N and S are disjoint
    # halves that tile the full circle, and so are E and W -- each pair is a
    # complete decomposition of the same node. Adjacent views share a 90 deg
    # quadrant. So structure across the rows is partly geometric by design,
    # and five p-values here are five looks at two partitions.
    #
    # Cluster by face, not by street. There are only 15 streets; cluster
    # -robust SEs need roughly 30-50 groups and are erratic below that, too
    # narrow as often as too wide.
    grp = "face_id" if "face_id" in dm.columns else "osm_name"
    if grp == "osm_name" and "face_id" in metrics.columns:
        dm = dm.merge(metrics[["node_id", "face_id"]], on="node_id", how="left")
        grp = "face_id"

    rows = []
    for lab in list(DIRECTIONS) + ["full360"]:
        d2 = dm[dm.direction.eq(lab)].dropna(subset=["GVI", "VEI", grp])
        if len(d2) < 30:
            continue
        r = sm.OLS(d2.GVI, sm.add_constant(d2[["VEI"]])).fit(
            cov_type="cluster", cov_kwds={"groups": d2[grp]})
        lo, hi = r.conf_int().iloc[1]
        rho, prho = spearmanr(d2.VEI, d2.GVI)
        rows.append({"view": lab, "n": len(d2), "slope": r.params.iloc[1],
                     "ci_lo": lo, "ci_hi": hi, "p": r.pvalues.iloc[1],
                     "r2": r.rsquared, "rho": rho, "p_rho": prho,
                     "VEI_sd": d2.VEI.std(), "VEI_lo_tail": int((d2.VEI < 0.5).sum()),
                     "GVI_med": d2.GVI.median(), "VEI_med": d2.VEI.median()})
    tab = pd.DataFrame(rows)
    tab.to_csv(RES / "tables" / "regression_by_direction.csv", index=False)
    print(f"\nGVI ~ VEI by view (SEs clustered by {grp}, {dm[grp].nunique()} groups):")
    print(tab.round(3).to_string(index=False))
    print("\n  Read the `rho` column against `slope`. If rho is near-constant")
    print("  across views while the OLS slope swings by several fold, the")
    print("  spread is leverage from the low-VEI tail (VEI_lo_tail), not")
    print("  directional signal -- OLS slope is cov/var and a handful of")
    print("  extreme-x points move it a long way. Rank correlation does not")
    print("  care, so a stable rho with an unstable slope means the views")
    print("  are measuring the same weak relationship.")
    print("  Five views, two partitions: do not read one significant row as")
    print("  a directional finding without adjusting for the other four.")

    # ------------------------------------------------ along vs cross street
    # This is the contrast the design supports, and it is NOT what the
    # compass rows show. Which views are along-street flips with typology:
    # on the avenues (axis ~029) N and S run along the corridor, on the
    # cross streets (axis ~119) E and W do. So every compass row mixes both
    # viewing situations in whatever ratio the node counts happen to give.
    sub2 = dm[dm.direction.ne("full360")].dropna(
        subset=["GVI", "VEI", "along_street", grp])
    if len(sub2) > 100:
        print("\nGVI ~ VEI, pooled by viewing situation rather than compass:")
        print(f"  {'situation':10s} {'slope':>8s} {'p':>10s} {'R2':>8s} "
              f"{'rho':>7s} {'n':>6s}")
        arows = []
        for al, name in [(True, "along"), (False, "cross")]:
            d3 = sub2[sub2.along_street.eq(al)]
            r = sm.OLS(d3.GVI, sm.add_constant(d3[["VEI"]])).fit(
                cov_type="cluster", cov_kwds={"groups": d3[grp]})
            rho, _ = spearmanr(d3.VEI, d3.GVI)
            print(f"  {name:10s} {r.params.iloc[1]:8.2f} "
                  f"{r.pvalues.iloc[1]:10.2e} {r.rsquared:8.4f} "
                  f"{rho:+7.3f} {len(d3):6d}")
            arows.append({"situation": name, "slope": r.params.iloc[1],
                          "p": r.pvalues.iloc[1], "r2": r.rsquared,
                          "rho": rho, "n": len(d3)})
        pd.DataFrame(arows).to_csv(
            RES / "tables" / "regression_along_cross.csv", index=False)
        print("  Along-street looks down the corridor, cross-street at the")
        print("  facades. Report this contrast, not the compass rows.")

    fmet = metrics.copy()
    ffac = (metrics.groupby("face_id")
                   .agg(GVI=("GVI", "median"), VEI=("VEI", "median"),
                        HW_ratio=("HW_ratio", "median"),
                        osm_name=("osm_name", "first"),
                        n_nodes=("node_id", "size"))
                   .reset_index())
    ffac = ffac[ffac.n_nodes >= 2]
    robust_associations(fmet, ffac)

    # ------------------------------------ is enclosure just position?
    # Streets get taller and narrower towards Grand Central and greener
    # towards the south, so grid-axis position is correlated with both
    # sides of the enclosure result. If the association does not survive
    # partialling it out, what looks like enclosure is geography.
    if "northing_m" in metrics.columns:
        print("\n--- confound check: enclosure against grid-axis position ---")
        f2 = (metrics.groupby("face_id")
                     .agg(GVI=("GVI", "median"), VEI=("VEI", "median"),
                          HW_ratio=("HW_ratio", "median"),
                          northing_m=("northing_m", "mean"),
                          n_nodes=("node_id", "size"))
                     .reset_index())
        # Same >=2-node rule s04 applies to block_faces.csv, so the face n
        # here is the same 22 units reported everywhere else.
        f2 = f2[f2.n_nodes >= 2]
        prows = []
        print(f"  {'unit':6s} {'pair':16s} {'rho':>7s} {'rho | northing':>15s} "
              f"{'p':>9s} {'n':>5s}")
        for unit, d4 in [("node", metrics), ("face", f2)]:
            for xcol in ["HW_ratio", "VEI"]:
                if xcol not in d4.columns:
                    continue
                raw, par, pv, n = partial_spearman(d4, xcol, "GVI", "northing_m")
                print(f"  {unit:6s} {xcol + ' ~ GVI':16s} {raw:+7.2f} "
                      f"{par:+15.2f} {pv:9.3f} {n:5d}")
                prows.append({"unit": unit, "x": xcol, "y": "GVI",
                              "control": "northing_m", "rho_raw": raw,
                              "rho_partial": par, "p_partial": pv, "n": n})
        pd.DataFrame(prows).to_csv(
            RES / "tables" / "partial_correlations.csv", index=False)
        print("  The face row is the one to read: nodes 20 m apart are not")
        print("  independent, so the node partial is computed on an n the")
        print("  data do not have. If the face partial collapses towards")
        print("  zero, the enclosure-greenery association in this frame is")
        print("  largely a north-south gradient, and grid-axis position is")
        print("  the covariate that shows it rather than a finding of its own.")

    # ------------------------------------------------ per street
    rng = np.random.default_rng(CFG["seed"])

    def ci(v):
        v = v.dropna().values
        if len(v) < 5:
            return np.nan, np.nan
        b = rng.choice(v, (CFG["spatial"]["bootstrap_n"], len(v)), replace=True)
        return tuple(np.percentile(np.median(b, axis=1), [2.5, 97.5]))

    st = (metrics.groupby("osm_name")
          .agg(n=("node_id", "size"), typology=("typology", "first"),
               GVI=("GVI", "median"), VEI=("VEI", "median"),
               HW=("HW_ratio", "median") if "HW_ratio" in metrics else ("VEI", "median"),
               W_n=("W_facade", "count") if "W_facade" in metrics else ("node_id", "size"))
          .reset_index().sort_values("GVI", ascending=False))
    lo, hi = zip(*[ci(metrics.loc[metrics.osm_name.eq(s), "GVI"])
                   for s in st.osm_name])
    st["GVI_lo"], st["GVI_hi"] = np.round(lo, 2), np.round(hi, 2)
    st.to_csv(RES / "tables" / "by_street.csv", index=False)
    print("\n--- per street (bootstrap CI on the median) ---")
    print(st.round(2).to_string(index=False))
    print("  Overlapping CIs are the honest picture; a street-level ranking")
    print("  without them invites over-reading.")


if __name__ == "__main__":
    main()
