"""Stage 7 -- GVI against H/W, on the framework's pre-specified bands.

WHAT SECTION 4 ACTUALLY CLAIMS
------------------------------
Read the three bullets of "The Enclosure Envelope" as written:

    H/W < 0.3        under-enclosed: feels sprawling and exposed;
                     eye-level greenery gets VISUALLY DILUTED, weakening
                     place identity
    H/W 0.8 - 1.5    human scale: an intimate "outdoor room" that
                     naturally FRAMES eye-level green features and
                     optimises solar shading
    H/W > 3.0        deep canyon: can induce spatial oppression UNLESS
                     OFFSET BY rich eye-level greenery and high
                     ground-floor glass permeability

Not one of them says greenery is more abundant at human scale. Each says
enclosure changes what greenery DOES to the experience of a street --
diluted, framed, compensating. The quantity with a maximum in the middle
is perceived quality, not GVI.

The document's own model says the same thing structurally: sense of place
is a function of grouped terms, and GVI sits under Place Attachment while
canyon H/W sits under Place Identity. They are separate arguments to the
same function, not one predicting the other. And the third bullet is an
explicit interaction -- a deep canyon is oppressive UNLESS greenery is
rich -- which is a claim about GVI x H/W acting on an outcome, not about
GVI as a function of H/W.

SO WHAT IS THIS STAGE
---------------------
It tests a PROXY, and it should be described as one. The proxy is: does
GVI vary non-monotonically with H/W, peaking inside the human-scale band?
That is not asserted by the document. It is a reasonable thing to ask
alongside it -- if greenery reads best at human scale, one might expect
planting to have accumulated there -- but the expectation is ours.

What survives from the document is the part that makes the test
well-posed: the cut points 0.3, 0.8, 1.5 and 3.0 are its numbers, written
down before any GVI in this frame was inspected. A shape hypothesis with
boundaries fixed in advance is testable. That is the justification for
fitting a curve with a turning point, and it is unaffected by the proxy
question -- what changes is what a failure licenses you to say.

WHAT A FAILURE HERE DOES AND DOES NOT MEAN
------------------------------------------
Does: GVI in Murray Hill declines monotonically as streets get narrower
and taller, flattening near zero, with no peak at human scale. That is a
real finding about greenery and geometry, on pre-specified bands.

Does NOT: falsify the enclosure envelope. The envelope is a claim about
sense of place, and nothing in this dataset measures sense of place. The
envelope remains untested here, for want of an outcome variable -- which
is a different sentence from "the envelope fails", and the two must not
be swapped in the write-up.

WHAT IS *NOT* A REASON TO FIT A CURVE, AND WHY IT MATTERS
---------------------------------------------------------
"The straight line gave a low R2, so the relationship must be an
inverted-U" does not follow, and building the analysis on it would make
the result unpublishable. Three separate problems:

1. A low linear R2 carries no information about curvature. It is equally
   consistent with no relationship, with a monotone relationship buried in
   noise, and with any non-linear shape whatever. Rejecting a line tells
   you the line is wrong, never what is right.

2. The quadratic NESTS the linear. Adding x^2 cannot lower in-sample R2,
   so "the quadratic fits better" is arithmetic, not evidence. Only the
   SIGN of the quadratic term and the location of the turning point say
   anything about shape.

3. Choosing the functional form after seeing the first fit fail, on the
   same data, is specification search. The p-value that comes out is not
   the p-value it claims to be.

WHAT THE TEST ACTUALLY IS
-------------------------
A significant squared term is not sufficient for an inverted-U: a
monotone-but-curved relationship over the observed range produces one
routinely. Lind & Mehlum (2010, Oxford Bulletin of Economics and
Statistics) give the correct joint test, and all three parts must hold:

    (a) the quadratic coefficient is negative (concave)
    (b) the fitted slope is positive at the low end of the data
    (c) the fitted slope is negative at the high end of the data

plus the turning point falling strictly inside the observed range, with a
Fieller interval rather than a delta-method one because the turning point
is a ratio of estimates and its sampling distribution is skewed.

The curve is then raced against monotone alternatives and scored out of
sample with folds split on block face. In-sample fit always rewards the
larger model; held-out error does not.

THE OUTCOME VARIABLE PROBLEM
----------------------------
The framework's dependent variable is sense of place. No sense-of-place
measure exists in this dataset -- no survey, no dwell time, no intercept
counts -- so the interaction the document does predict, greenery offsetting
the oppression of a deep canyon, cannot be estimated here at all. It is not
weakly supported or poorly identified. It is unmeasured, and it stays a
hypothesis requiring outcome data.
"""
import sys
from pathlib import Path

import numpy as np, pandas as pd, geopandas as gpd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from common import CFG, PROC, RES, banner, DIRECTIONS

BANDS = CFG["enclosure"]["bands"]
BAND_LABELS = CFG["enclosure"]["band_labels"]
MIN_BAND_N = CFG["enclosure"]["min_band_n"]
N_BINS = CFG["enclosure"]["envelope_bins"]
BOOT_N = CFG["spatial"]["bootstrap_n"]


def _cluster_ols(X, y, groups):
    import statsmodels.api as sm
    return sm.OLS(y, sm.add_constant(X)).fit(
        cov_type="cluster", cov_kwds={"groups": groups})


def fieller_turning_point(fit, alpha=0.05):
    """Fieller interval for -b1/(2*b2), the turning point of a quadratic.

    The turning point is a ratio of two estimates. The delta method assumes
    that ratio is normal, which it is not when the denominator is anywhere
    near zero -- and a weakly identified curvature term is exactly that
    case. Fieller inverts the t-test on (b1 + 2*t*b2) instead and can
    honestly return an unbounded interval, which is the right answer when
    the data do not locate a turning point at all.
    """
    b1, b2 = fit.params[1], fit.params[2]
    V = np.asarray(fit.cov_params())
    v11, v12, v22 = V[1, 1], V[1, 2], V[2, 2]
    z = stats.norm.ppf(1 - alpha / 2)
    # Solve  (b1 + 2 t b2)^2 = z^2 Var(b1 + 2 t b2)  for t.
    A = 4 * (b2 ** 2 - z ** 2 * v22)
    B = 4 * (b1 * b2 - z ** 2 * v12)
    C = b1 ** 2 - z ** 2 * v11
    if abs(A) < 1e-12:
        return np.nan, np.nan, False
    disc = B ** 2 - 4 * A * C
    if disc < 0:
        return np.nan, np.nan, False
    r = np.sqrt(disc)
    t1, t2 = (-B - r) / (2 * A), (-B + r) / (2 * A)
    lo, hi = sorted((-t1, -t2))
    return lo, hi, A > 0          # A > 0 -> bounded interval


def lind_mehlum(fit, x, label="GVI"):
    """The three-part joint test. Returns a dict; prints the verdict."""
    b1, b2 = fit.params[1], fit.params[2]
    V = np.asarray(fit.cov_params())
    xl, xh = float(np.min(x)), float(np.max(x))

    def slope_and_t(xv):
        s = b1 + 2 * b2 * xv
        grad = np.array([0.0, 1.0, 2 * xv])
        se = float(np.sqrt(grad @ V @ grad))
        return s, s / se if se > 0 else np.nan

    s_lo, t_lo = slope_and_t(xl)
    s_hi, t_hi = slope_and_t(xh)
    crit = stats.norm.ppf(0.95)
    a = b2 < 0 and fit.pvalues[2] < 0.05
    b = s_lo > 0 and t_lo > crit
    c = s_hi < 0 and t_hi < -crit
    tp = -b1 / (2 * b2) if b2 != 0 else np.nan
    lo, hi, bounded = fieller_turning_point(fit)

    print(f"\n  Lind-Mehlum joint test for an inverted-U in {label}:")
    print(f"    (a) concave, b2 < 0             b2 = {b2:+.4f}  "
          f"p = {fit.pvalues[2]:.2e}   {'PASS' if a else 'FAIL'}")
    print(f"    (b) slope > 0 at H/W = {xl:5.2f}    {s_lo:+8.3f}  "
          f"t = {t_lo:+6.2f}          {'PASS' if b else 'FAIL'}")
    print(f"    (c) slope < 0 at H/W = {xh:5.2f}    {s_hi:+8.3f}  "
          f"t = {t_hi:+6.2f}          {'PASS' if c else 'FAIL'}")
    verdict = a and b and c
    print(f"    joint verdict: {'INVERTED-U SUPPORTED' if verdict else 'NOT SUPPORTED'}")
    ci_txt = (f"95% CI [{lo:.2f}, {hi:.2f}]"
              if bounded and np.isfinite(lo) and np.isfinite(hi)
              else "95% CI unbounded (the data do not locate a turning point)")
    print(f"    turning point  H/W = {tp:.3f}   {ci_txt}")
    print(f"    inside the observed range [{xl:.2f}, {xh:.2f}]: "
          f"{'yes' if xl < tp < xh else 'NO'}   "
          f"({int((x > tp).sum())} of {len(x)} nodes lie beyond it)")
    if b2 > 0:
        print("    NOTE: b2 is POSITIVE. That is a U, not an inverted-U --")
        print("    the curve is convex. If the fitted minimum sits near the")
        print("    top of the range, what this describes is a monotone")
        print("    DECLINE that flattens, not a peak at human scale.")
    return {"b2": b2, "p_b2": fit.pvalues[2], "slope_lo": s_lo,
            "slope_hi": s_hi, "t_lo": t_lo, "t_hi": t_hi, "turning_point": tp,
            "tp_ci_lo": lo, "tp_ci_hi": hi, "tp_ci_bounded": bounded,
            "pass_a": a, "pass_b": b, "pass_c": c, "inverted_u": verdict}


def shape_race(x, y, g):
    """Quadratic against monotone alternatives, in sample and out."""
    import statsmodels.api as sm
    specs = {
        "linear      y ~ x":        lambda v: v.reshape(-1, 1),
        "quadratic   y ~ x + x^2":  lambda v: np.c_[v, v ** 2],
        "log         y ~ log x":    lambda v: np.log(np.maximum(v, 1e-6)).reshape(-1, 1),
        "sqrt        y ~ sqrt x":   lambda v: np.sqrt(v).reshape(-1, 1),
        "reciprocal  y ~ 1/x":      lambda v: (1 / np.maximum(v, 1e-6)).reshape(-1, 1),
    }
    print(f"\n  {'specification':26s} {'R2':>8s} {'AIC':>9s} {'CV RMSE':>9s}")
    print("  " + "-" * 56)
    rows = []
    try:
        from sklearn.model_selection import GroupKFold
        from sklearn.linear_model import LinearRegression
        gkf = GroupKFold(n_splits=min(5, pd.Series(g).nunique()))
        have_cv = True
    except ImportError:
        have_cv = False
    for lab, f in specs.items():
        m = _cluster_ols(f(x), y, g)
        cv = np.nan
        if have_cv:
            err = []
            for tr, te in gkf.split(x, y, g):
                lm = LinearRegression().fit(f(x[tr]), y[tr])
                err.append(np.mean((y[te] - lm.predict(f(x[te]))) ** 2))
            cv = float(np.sqrt(np.mean(err)))
        print(f"  {lab:26s} {m.rsquared:8.4f} {m.aic:9.1f} {cv:9.3f}")
        rows.append({"spec": lab.split()[0], "r2": m.rsquared, "aic": m.aic,
                     "cv_rmse": cv})
    print("  In-sample R2 can only rise with the squared term -- it nests the")
    print("  line. Held-out RMSE is the column that can fall either way, and")
    print("  folds are split on block face so no face straddles train/test.")
    return pd.DataFrame(rows)


def band_model(d, x_col="HW_ratio", y_col="GVI"):
    """The framework's own bands, as a factor with a slope inside each."""
    import statsmodels.api as sm
    d = d.copy()
    d["band"] = pd.cut(d[x_col], BANDS, labels=BAND_LABELS)
    t = (d.groupby("band", observed=True)
           .agg(n=(y_col, "size"), faces=("face_id", "nunique"),
                median=(y_col, "median"), mean=(y_col, "mean"),
                q25=(y_col, lambda v: v.quantile(.25)),
                q75=(y_col, lambda v: v.quantile(.75)))
           .round(2))
    print("\n  GVI by the framework's pre-specified enclosure bands:")
    print("  " + t.to_string().replace("\n", "\n  "))

    thin = t[t.n < MIN_BAND_N]
    for band, r in thin.iterrows():
        print(f"\n  !! band '{band}' holds {int(r.n)} nodes on "
              f"{int(r.faces)} face(s).")
        print("     The framework's prediction for this regime cannot be")
        print("     tested in this study area. Report it as out of scope,")
        print("     not as a null result.")

    grp = [v[y_col].dropna().values
           for _, v in d.groupby("band", observed=True) if len(v) >= 5]
    if len(grp) >= 2:
        H, p = stats.kruskal(*grp)
        print(f"\n  Kruskal-Wallis across bands: H={H:.1f} p={p:.2e}")
        print("  A difference between bands is NOT a peak. Read the medians")
        print("  in order: monotone decline and a mid-range maximum both")
        print("  produce a significant H.")

    # Monotone-ordering check: does the median ever go UP as H/W rises?
    med = t["median"].values
    rises = [(BAND_LABELS[i], BAND_LABELS[i + 1])
             for i in range(len(med) - 1) if med[i + 1] > med[i]]
    print(f"\n  band-to-band median increases: "
          f"{rises if rises else 'none -- medians are monotone decreasing'}")
    if not rises:
        print("  A peak at human scale requires at least one increase before")
        print("  the decline. There is none. That refutes the proxy -- GVI")
        print("  peaking at human scale -- not the envelope, which is a claim")
        print("  about sense of place and is not measured here.")
    return t.reset_index()


def envelope_curve(d, n_bins=N_BINS, boot=BOOT_N, seed=None):
    """Median GVI in equal-count H/W bins, with a face-clustered bootstrap CI.

    The descriptive form of the same question the Lind-Mehlum test asks,
    and the one that can be read without accepting a functional form. A
    quadratic answers "is there a peak" only by assuming the shape has at
    most one; binned medians let the data show a peak, a plateau, a decline
    or a kink without being asked to choose in advance.

    Resampling is over face_id, not over nodes. Moran's I here is ~0.62, so
    a node-level bootstrap would treat 20 m neighbours as independent draws
    and return an interval several times too narrow.
    """
    seed = CFG["seed"] if seed is None else seed
    d = d.dropna(subset=["HW_ratio", "GVI", "face_id"]).copy()
    edges = np.unique(np.quantile(d.HW_ratio, np.linspace(0, 1, n_bins + 1)))
    d["bin"] = pd.cut(d.HW_ratio, edges, include_lowest=True, labels=False)

    by_face = {f: g for f, g in d.groupby("face_id")}
    faces = np.array(list(by_face))
    rng = np.random.default_rng(seed)
    draws = np.full((boot, len(edges) - 1), np.nan)
    for i in range(boot):
        rs = pd.concat([by_face[f] for f in rng.choice(faces, len(faces))])
        m = rs.groupby("bin", observed=True).GVI.median()
        draws[i, m.index.astype(int)] = m.values

    obs = d.groupby("bin", observed=True).agg(
        n=("GVI", "size"), faces=("face_id", "nunique"),
        hw_mid=("HW_ratio", "median"), median=("GVI", "median"))
    obs["lo"] = np.nanquantile(draws, .025, axis=0)[obs.index.astype(int)]
    obs["hi"] = np.nanquantile(draws, .975, axis=0)[obs.index.astype(int)]
    return obs.reset_index(drop=True)


def directional_envelopes(d, out):
    """The envelope curve as a pedestrian actually meets it.

    A pedestrian walks along a street and looks where they are going, so
    the view to model is the 180-degree forward cone on the bearing of
    travel -- peripheral vision included, which is what makes 180 rather
    than 60 the right cone. Two things follow, and both remove panels that
    an earlier version of this figure drew.

    The cross-street views are not a pedestrian condition. Facing east
    while walking north is a side glance at a wall, not a way anyone
    experiences a street, so those rows are excluded here rather than
    plotted as a contrast. (They remain in the stage 6 along/cross
    regression, which asks a different question: whether the anisotropy of
    a node depends on enclosure.)

    The full 360 index is not a pedestrian condition either. It is the
    average of what a person sees and what is behind their head, and the
    premise of this figure is that the second half does not reach them.
    Plotting it as a reference line invited exactly the comparison the
    design says is meaningless.

    What is left is four legitimate conditions, each restricted to the
    nodes where that bearing runs along the street: walking uptown or
    downtown on an avenue, walking east or west on a cross street. N and S
    still describe the same avenue nodes seen forwards and backwards, so
    the gap between those two curves is a within-node contrast -- what
    changes when you turn around -- not two independent samples.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = d[d.along_street.eq(True)].dropna(subset=["GVI", "HW_ratio", "face_id"])
    if len(d) < 80:
        return None

    labels = {"N_uptown": "walking uptown (avenues)",
              "S_downtown": "walking downtown (avenues)",
              "E_east": "walking east (cross streets)",
              "W_west": "walking west (cross streets)"}
    colours = {"N_uptown": "#22406b", "S_downtown": "#7fb0e0",
               "E_east": "#a8442a", "W_west": "#e0a08a"}

    rows, curves = [], {}
    todo = list(labels) + ["all along-street"]
    # Park Avenue is 35 of the 67 avenue nodes that have an H/W, and its
    # planted mall sits at H/W ~1.2. Walking an avenue means looking down
    # the corridor, so the mall lands squarely in the forward view and one
    # street can carry the whole avenue curve. Drawn both ways for that
    # reason, not because the mall is an artefact.
    if "osm_name" in d.columns:
        todo += ["N_uptown_noPark", "S_downtown_noPark"]
    for lab in todo:
        base = lab.replace("_noPark", "")
        sub = d if base == "all along-street" else d[d.direction.eq(base)]
        if lab.endswith("_noPark"):
            sub = sub[sub.osm_name.ne("Park Avenue")]
        if len(sub) < 60:
            continue
        # Fewer bins where there are fewer nodes: eight equal-count bins of
        # 135 node-views would put 17 in each, which a median cannot carry.
        env = envelope_curve(sub, n_bins=N_BINS if len(sub) > 250 else 5)
        curves[lab] = env
        for _, r in env.iterrows():
            rows.append({"view": lab, **r.to_dict()})

    if "all along-street" not in curves:
        return None

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    for a in ax:
        for e in BANDS[1:-1]:
            a.axvline(e, color="#b9b3a8", lw=.9, ls=":", zorder=1)
        a.axvspan(0.8, 1.5, color="#2c7a4b", alpha=.09, zorder=0)
        a.set_xlabel("H/W (measured facade-to-facade)")

    e = curves["all along-street"]
    ax[0].fill_between(e.hw_mid, e.lo, e.hi, color="#2c7a4b", alpha=.18,
                       zorder=3, label="95% CI, bootstrap over faces")
    ax[0].plot(e.hw_mid, e["median"], "-o", color="#1d5334", lw=2.2, ms=5,
               zorder=6, label=f"forward view while walking (n={int(e.n.sum())})")
    ax[0].set_ylabel("median GVI (%)")
    ax[0].set_title("What a pedestrian sees ahead, along the street",
                    fontsize=10)
    ax[0].legend(fontsize=8)

    for lab, nice in labels.items():
        if lab not in curves:
            continue
        c = curves[lab]
        ax[1].plot(c.hw_mid, c["median"], "-o", ms=4, lw=1.8,
                   color=colours[lab], label=f"{nice}, n={int(c.n.sum())}")
        if f"{lab}_noPark" in curves:
            c2 = curves[f"{lab}_noPark"]
            ax[1].plot(c2.hw_mid, c2["median"], ":s", ms=3.5, lw=1.3,
                       color=colours[lab], alpha=.85,
                       label=f"{nice.split(' (')[0]}, no Park Ave")
    ax[1].set_title("By direction of travel", fontsize=10)
    ax[1].legend(fontsize=8)

    plt.tight_layout()
    pth = out / "figures" / "figure_enclosure_directional.png"
    plt.savefig(pth, dpi=300, bbox_inches="tight")
    plt.close()
    t = pd.DataFrame(rows)
    t.to_csv(out / "tables" / "enclosure_envelope_directional.csv", index=False)
    print(f"  wrote {pth} and enclosure_envelope_directional.csv")
    return t


def figure(d, fit_q, out):
    """Three panels, left to right as the argument is made.

    The scatter and its fitted curve come first because that is where the
    shape claim can be checked against the points themselves -- a reader
    should see the raw cloud before seeing any summary of it. The binned
    medians restate the same shape without assuming a functional form, and
    the framework's own categories come last.

    The order is presentational. The quadratic is still not the headline
    result: with 421 nodes on 25 faces it implies a precision the
    clustering does not support. What it licenses is the phrase "not
    supported" in place of "looks flat", which the medians alone cannot.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x, y = d.HW_ratio.values, d.GVI.values
    fig, axes = plt.subplots(1, 3, figsize=(19.5, 5.2))
    # Reading order: the scatter and its fitted curve first, so the shape
    # claim is seen on the raw points; then the same shape as binned
    # medians; then the framework's own categories.
    ax_fit, ax_env, ax_band = axes

    # ---------------------------------------------- centre: the medians
    def draw_regimes(a):
        for e in BANDS[1:-1]:
            a.axvline(e, color="#b9b3a8", lw=.9, ls=":", zorder=1)
        a.axvspan(0.8, 1.5, color="#2c7a4b", alpha=.09, zorder=0)
        a.axvspan(BANDS[0], 0.3, color="#c9a227", alpha=.07, zorder=0)
        a.axvspan(3.0, max(3.01, x.max()), color="#7b3f9d", alpha=.07, zorder=0)

    env = envelope_curve(d)
    draw_regimes(ax_env)
    ax_env.fill_between(env.hw_mid, env.lo, env.hi, color="#3b6ea5", alpha=.18,
                       zorder=3, label="95% CI, bootstrap over faces")
    ax_env.plot(env.hw_mid, env["median"], "-o", color="#22406b", lw=2, ms=5,
               zorder=6, label=f"median GVI, {len(env)} equal-count bins")

    # Park Avenue's planted median sits near the predicted optimum and is
    # the one street that could manufacture a peak there. Drawn separately
    # so the reader sees whether the shape depends on it.
    d2 = d[d.osm_name.ne("Park Avenue")]
    if len(d2) > 50:
        e2 = envelope_curve(d2)
        ax_env.plot(e2.hw_mid, e2["median"], "--s", color="#c0392b", lw=1.4,
                   ms=4, zorder=7, label="excluding Park Avenue")
    ax_env.set_xlabel("H/W (measured facade-to-facade)")
    ax_env.set_ylabel("median GVI (%)")
    ax_env.set_title("GVI across the enclosure range, as the data draw it",
                     fontsize=10)
    ax_env.legend(fontsize=7.5, loc="upper right")
    ax_env.text(0.15, ax_env.get_ylim()[1] * .06, "under-\nenclosed", fontsize=7,
               ha="center", color="#8a6d0b")
    ax_env.text(1.15, ax_env.get_ylim()[1] * .06, "human\nscale", fontsize=7,
               ha="center", color="#2c7a4b")
    ax_env.text(min(3.6, x.max() * .95), ax_env.get_ylim()[1] * .06,
               "deep\ncanyon", fontsize=7, ha="center", color="#7b3f9d")

    # ---------------------------------------------- right: framework bands
    d3 = d.assign(band=pd.cut(d.HW_ratio, BANDS, labels=BAND_LABELS))
    order = [b for b in BAND_LABELS if (d3.band == b).any()]
    data = [d3.loc[d3.band.eq(b), "GVI"].dropna().values for b in order]
    bp = ax_band.boxplot(data, vert=True, patch_artist=True, showfliers=False,
                       widths=.62)
    for patch, b in zip(bp["boxes"], order):
        patch.set_facecolor("#2c7a4b" if "human" in b else "#c9c3b8")
        patch.set_alpha(.65)
    ax_band.set_xticks(range(1, len(order) + 1))
    ax_band.set_xticklabels([b.replace(" ", "\n", 1) for b in order], fontsize=7.5)
    ax_band.set_ylabel("GVI (%)")
    ax_band.set_title("Pre-specified bands (n, faces shown)", fontsize=10)
    for i, (v, b) in enumerate(zip(data, order), start=1):
        nf = d3.loc[d3.band.eq(b), "face_id"].nunique()
        ax_band.text(i, ax_band.get_ylim()[1] * .96, f"n={len(v)}\n{nf} faces",
                   ha="center", fontsize=7, color="#555")

    # ---------------------------------------------- left: the formal test
    ax_fit.scatter(x, y, s=11, alpha=.38, c="#3b6ea5", zorder=3)
    xs = np.linspace(x.min(), x.max(), 250)
    b0, b1, b2 = fit_q.params
    ax_fit.plot(xs, b0 + b1 * xs + b2 * xs ** 2, color="#c0392b", lw=2.1,
               zorder=6, label="quadratic (fitted)")
    lin = _cluster_ols(x.reshape(-1, 1), y, d.face_id)
    ax_fit.plot(xs, lin.params[0] + lin.params[1] * xs, color="#7d7d7d",
               lw=1.5, ls="--", zorder=5, label="linear")
    draw_regimes(ax_fit)
    ax_fit.set_xlabel("H/W (measured facade-to-facade)")
    ax_fit.set_ylabel("GVI (%)")
    ax_fit.set_title(f"Fitted shape: b2 = {b2:+.3f} "
                    f"({'convex, a U' if b2 > 0 else 'concave'})", fontsize=10)
    ax_fit.legend(fontsize=8)

    plt.tight_layout()
    p = out / "figures" / "figure_enclosure.png"
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    env.to_csv(out / "tables" / "enclosure_envelope.csv", index=False)
    print(f"\n  wrote {p} and enclosure_envelope.csv")
    return env


def main():
    banner("STAGE 7  GVI against H/W, on the pre-specified bands")
    metrics = gpd.read_file(PROC / "metrics.gpkg")
    need = {"GVI", "HW_ratio", "face_id"}
    if not need.issubset(metrics.columns):
        print(f"missing {need - set(metrics.columns)} -- run s05 first")
        return
    d = metrics.dropna(subset=["GVI", "HW_ratio", "face_id"]).copy()
    print(f"n = {len(d)} nodes on {d.face_id.nunique()} block faces")
    print(f"H/W observed range {d.HW_ratio.min():.3f} to {d.HW_ratio.max():.3f}")
    print(f"H/W measured for {len(d)}/{len(metrics)} nodes "
          f"({len(d)/len(metrics):.0%}) -- the rest lack a facade on one side")

    x, y, g = d.HW_ratio.values, d.GVI.values, d.face_id.values
    lin = _cluster_ols(x.reshape(-1, 1), y, g)
    quad = _cluster_ols(np.c_[x, x ** 2], y, g)
    print(f"\n  linear     R2 = {lin.rsquared:.4f}  "
          f"slope = {lin.params[1]:+.3f}  p = {lin.pvalues[1]:.2e}")
    print(f"  quadratic  R2 = {quad.rsquared:.4f}  "
          f"gain = {quad.rsquared - lin.rsquared:+.4f}")

    lm = lind_mehlum(quad, x)
    race = shape_race(x, y, g)
    bands = band_model(d)

    # Park Avenue carries a planted central mall at H/W near the predicted
    # optimum, so it is the single most influential street for this
    # hypothesis. If a peak exists only with it in, that is not an envelope
    # effect, it is one street.
    d2 = d[d.osm_name.ne("Park Avenue")]
    if len(d2) > 50:
        print("\n  --- excluding Park Avenue (planted median, H/W ~1.2) ---")
        x2, y2, g2 = d2.HW_ratio.values, d2.GVI.values, d2.face_id.values
        q2 = _cluster_ols(np.c_[x2, x2 ** 2], y2, g2)
        lind_mehlum(q2, x2, label="GVI, no Park Ave")
        band_model(d2)

    (RES / "tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([lm]).to_csv(RES / "tables" / "enclosure_invertedU.csv",
                              index=False)
    race.to_csv(RES / "tables" / "enclosure_shape_race.csv", index=False)
    bands.to_csv(RES / "tables" / "enclosure_bands.csv", index=False)
    # The pedestrian reading: the same curve per 180-degree travel view.
    dmp = PROC / "directional_metrics.csv"
    if dmp.exists():
        print("\n  --- by travel direction ---")
        dd = pd.read_csv(dmp).merge(
            d[["node_id", "HW_ratio", "face_id"]], on="node_id", how="inner")
        dirtab = directional_envelopes(dd, RES)
        if dirtab is not None:
            # Each view has its own equal-count bin edges, so a shared-index
            # pivot would be mostly blanks. Read each row left to right.
            print("  median GVI by H/W bin, forward view while walking:")
            for view, g in dirtab.groupby("view"):
                cells = "  ".join(f"{r.hw_mid:.2f}->{r['median']:5.2f}"
                                  for _, r in g.iterrows())
                print(f"    {view:18s} n={int(g.n.sum()):4d}  {cells}")
            print("  Along-street views only. Facing across the street is a")
            print("  side glance at a wall, and the 360 index averages in")
            print("  what is behind the pedestrian's head; neither is a")
            print("  condition anyone walks in. Uptown against downtown is")
            print("  the same avenue nodes turned around -- a within-node")
            print("  contrast, not two samples.")

    env = figure(d, quad, RES)
    print("\n  envelope diagram, median GVI by equal-count H/W bin:")
    print("  " + env.round(2).to_string().replace("\n", "\n  "))
    print("  Read the medians in order. A peak at human scale requires a")
    print("  rise into 0.8-1.5 and a fall after it; a monotone decline with")
    print("  overlapping intervals is the same picture the quadratic gives,")
    print("  without asking the data to be a parabola.")
    print(f"\nwrote enclosure_invertedU.csv, enclosure_shape_race.csv, "
          f"enclosure_bands.csv, enclosure_envelope.csv")


if __name__ == "__main__":
    main()
