"""Stage 7 -- figures.

Every group is drawn from the values present in the data, never from a
hardcoded list, and every legend entry carries its n. An empty group then
shows as an absent legend entry rather than a legend entry with no points --
which is how a missing typology went unnoticed once already.
"""
import sys
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import CFG, PROC, RES, banner, PALETTE, DIRECTIONS

FIG = RES / "figures"
# Zone colours are gone with the zones themselves; the north-south
# gradient is continuous now and gets a colour ramp, not a legend.


def groups(df, col="typology"):
    """Iterate the groups actually present, with counts."""
    for t in df[col].value_counts().index:
        s = df[df[col].eq(t)]
        yield t, s, PALETTE.get(t, "#777")


def maps(metrics):
    m = metrics.to_crs(4326)
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # Frame nodes with no metric, drawn as open rings. Without them a hole
    # in the map reads as "no street here" when it is really "sampled, then
    # excluded" -- the two runs on E 37th and Park Ave that the capture-date
    # filter drops. The audit table says which; the map should not hide that
    # the location was in the frame.
    absent = None
    npath = PROC / "nodes.gpkg"
    if npath.exists():
        allnodes = gpd.read_file(npath).to_crs(4326)
        absent = allnodes[~allnodes.node_id.isin(m.node_id)]

    m.plot(column="GVI", cmap="YlGn", legend=True, markersize=12, ax=ax[0])
    ax[0].set_title("Green View Index (%)")

    # VEI is bunched near 0.9 with a long left tail, so a linear ramp spends
    # most of its range on values almost nothing occupies. Decile bands put
    # the contrast where the data is.
    v = m.VEI.dropna()
    if len(v) > 10:
        bounds = np.unique(np.quantile(v, np.linspace(0, 1, 11)))
        m.plot(column="VEI", cmap="magma_r", norm=BoundaryNorm(bounds, 256),
               legend=True, markersize=12, ax=ax[1])
    ax[1].set_title("Visual Enclosure Index (decile bands)")

    if "northing_m" in m.columns:
        sc = ax[2].scatter(m.geometry.x, m.geometry.y, s=12, c=m.northing_m,
                           cmap="cividis")
        plt.colorbar(sc, ax=ax[2], label="metres uptown (bearing 029)")
        ax[2].set_title("Grid-axis position")
    else:
        ax[2].set_title("no northing_m")

    if absent is not None and len(absent):
        for a in ax:
            a.scatter(absent.geometry.x, absent.geometry.y, s=14,
                      facecolors="none", edgecolors="#999", linewidths=.7,
                      zorder=5, label=f"no metric ({len(absent)})")
        ax[0].legend(fontsize=7, loc="lower left", frameon=False)

    # Panels 0 and 1 are GeoPandas plots, which set an aspect for a
    # geographic CRS; panel 2 is a bare scatter, which does not. Left alone
    # the same street grid comes out at two different shapes side by side.
    aspect = 1 / np.cos(np.radians(float(m.geometry.y.mean())))
    for a in ax:
        a.set_aspect(aspect)
        a.set_axis_off()
    plt.tight_layout()
    plt.savefig(FIG / "figure_maps.png", dpi=300, bbox_inches="tight")
    plt.close()


def scatter(metrics, face):
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.6))
    for t, s, c in groups(metrics):
        ax[0].scatter(s.VEI, s.GVI, s=13, alpha=.55, c=c, label=f"{t} ({len(s)})")
    ax[0].set_title("By typology, nodes")
    if "northing_m" in metrics.columns:
        sc = ax[1].scatter(metrics.VEI, metrics.GVI, s=13, alpha=.65,
                           c=metrics.northing_m, cmap="cividis")
        plt.colorbar(sc, ax=ax[1], label="metres uptown")
    ax[1].set_title("Coloured by grid-axis position, nodes")
    for t, s, c in groups(face):
        ax[2].scatter(s.VEI, s.GVI, s=30, alpha=.8, c=c, label=f"{t} ({len(s)})")
    ax[2].set_title("By typology, block faces")
    for i, a in enumerate(ax):
        a.set_xlabel("VEI"); a.set_ylabel("GVI (%)")
        if i != 1:
            a.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "figure_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()


def directional(dm):
    import statsmodels.api as sm
    labels = list(DIRECTIONS) + ["full360"]
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    for a, lab in zip(axes.ravel(), labels):
        d = dm[dm.direction.eq(lab)].dropna(subset=["GVI", "VEI"])
        a.scatter(d.VEI, d.GVI, s=9, alpha=.4, c="#3b6ea5", zorder=3)
        if len(d) > 30:
            r = sm.OLS(d.GVI, sm.add_constant(d[["VEI"]])).fit(
                cov_type="cluster", cov_kwds={"groups": d.osm_name})
            xs = np.linspace(d.VEI.min(), d.VEI.max(), 80)
            pr = r.get_prediction(sm.add_constant(
                pd.DataFrame({"VEI": xs}))).summary_frame(alpha=.05)
            a.plot(xs, pr["mean"], color="#c0392b", lw=2, zorder=6)
            a.fill_between(xs, pr["mean_ci_lower"], pr["mean_ci_upper"],
                           color="#c0392b", alpha=.18, zorder=5, lw=0)
            b = DIRECTIONS.get(lab)
            tag = "" if b is None else f" ({b:.0f}°)"
            a.set_title(f"{lab}{tag}  slope {r.params.iloc[1]:+.1f}, "
                        f"R²={r.rsquared:.3f}, n={len(d)}", fontsize=10)
        a.set_xlabel("VEI"); a.set_ylabel("GVI (%)")
    axes.ravel()[-1].axis("off")
    plt.tight_layout()
    plt.savefig(FIG / "figure_directional.png", dpi=300, bbox_inches="tight")
    plt.close()


def rose(dm, metrics):
    """Azimuthal roses: where the greenery and the sky actually are."""
    npz = PROC / "azimuth_profiles.npz"
    if not npz.exists():
        print(f"  no {npz.name} -- skipping the azimuth roses")
        return
    z = np.load(npz)
    keys = list(z.files)
    if not keys:
        return
    prof = np.stack([z[k] for k in keys])          # nodes x 3 x bins
    theta = np.radians(np.arange(prof.shape[2]) + 0.5)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2),
                             subplot_kw={"projection": "polar"})
    for a, (ci, name, col) in zip(axes, [(0, "vegetation", "#2c7a4b"),
                                         (1, "sky", "#3b6ea5")]):
        mean = prof[:, ci, :].mean(axis=0)
        a.plot(theta, mean, color=col, lw=1.6)
        a.fill(theta, mean, color=col, alpha=.25)
        a.set_theta_zero_location("N")
        a.set_theta_direction(-1)
        for b in DIRECTIONS.values():
            a.plot([np.radians(b)] * 2, [0, mean.max()], color="#888",
                   ls="--", lw=.9)
        a.set_title(f"mean {name} share by bearing\n"
                    f"dashed = grid axes ({CFG['directional']['grid_bearing']:.0f}° "
                    f"and orthogonals)", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG / "figure_rose.png", dpi=300, bbox_inches="tight")
    plt.close()


def leverage(metrics):
    """What one street does to a slope, and what it does not do to a rank.

    Park Avenue sits at GVI 14-17 with a planted central mall where every
    other street is under 5, at H/W near 1.2 -- the middle of the range.
    An OLS slope is covariance over variance, so a cluster of points with
    ordinary x and extreme y pulls the fitted line towards flat and takes
    R2 with it. Spearman sees only that those faces are the top few in the
    greenness order, which they are with or without the mall.

    This panel exists because the first reading of these data was "no
    relationship" -- an R2 of 0.002 and p = 0.85 at face level. That was
    one street's leverage, not an absence.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    f = (metrics.groupby("face_id")
                .agg(GVI=("GVI", "median"), VEI=("VEI", "median"),
                     HW=("HW_ratio", "median"), n=("node_id", "size"),
                     osm_name=("osm_name", "first"))
                .reset_index())
    f = f[f.n >= 2]
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.4))
    for a, xcol, xlab in [(ax[0], "VEI", "VEI (block-face median)"),
                          (ax[1], "HW", "H/W (block-face median)")]:
        d = f.dropna(subset=[xcol, "GVI"])
        park = d.osm_name.eq("Park Avenue")
        a.scatter(d.loc[~park, xcol], d.loc[~park, "GVI"], s=46,
                  c="#3b6ea5", alpha=.8, label=f"other faces ({(~park).sum()})")
        a.scatter(d.loc[park, xcol], d.loc[park, "GVI"], s=90, c="#c0392b",
                  marker="D", label=f"Park Avenue ({park.sum()})")
        xs = np.linspace(d[xcol].min(), d[xcol].max(), 50)
        for mask, colr, style, name in [
                (slice(None), "#c0392b", "-", "all faces"),
                (~park, "#22406b", "--", "without Park Ave")]:
            sub = d[mask] if not isinstance(mask, slice) else d
            if len(sub) < 5:
                continue
            b1, b0 = np.polyfit(sub[xcol], sub.GVI, 1)
            r2 = np.corrcoef(sub[xcol], sub.GVI)[0, 1] ** 2
            rho = spearmanr(sub[xcol], sub.GVI)[0]
            a.plot(xs, b0 + b1 * xs, color=colr, ls=style, lw=1.8,
                   label=f"{name}: R\u00b2={r2:.3f}, rho={rho:+.2f}")
        a.set_xlabel(xlab)
        a.set_ylabel("GVI (%)")
        a.legend(fontsize=8)
        a.set_title(f"GVI against {xcol}: the slope moves, the rank does not",
                    fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG / "figure_leverage.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    banner("STAGE 7  figures")
    metrics = gpd.read_file(PROC / "metrics.gpkg")
    face = pd.read_csv(PROC / "block_faces.csv")
    dm = pd.read_csv(PROC / "directional_metrics.csv")
    maps(metrics)
    scatter(metrics, face)
    leverage(metrics)
    directional(dm)
    rose(dm, metrics)
    print(f"wrote figures to {FIG}")
    for p in sorted(FIG.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
