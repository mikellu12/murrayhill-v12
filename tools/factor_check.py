"""
Factor analysis of the directional metrics.

THE QUESTION
------------
Each node yields GVI and VEI under four travel directions. If a node's four
values are essentially one number plus noise, direction adds nothing and the
360 index is a sufficient summary. If two or more factors are needed, the
streetscape is anisotropic at the sampling point and averaging over the
horizon discards real structure.

This is a sharper test than "do the direction means differ", because it asks
about covariance within nodes rather than differences between marginal
distributions. Two directions can have identical means while being
uncorrelated across nodes.

WHAT TO EXPECT, AND WHY IT NEEDS CARE
-------------------------------------
The four 180-degree views OVERLAP: N and E share a 90-degree quadrant, N and
S share nothing. So even under pure noise the correlation matrix has
structure -- adjacent directions correlate, opposite directions do not.
A single dominant factor is therefore the null, not the finding. What would
be interesting is a clear TWO-factor solution splitting the axes, i.e. one
factor loading on N and S, another on E and W. That would say the along-
street and cross-street views measure different things.

Sampling adequacy is checked first: with four indicators the KMO statistic
is often marginal, and a factor solution on inadequate data is not
interpretable no matter how tidy the loadings look.

Usage:
    import factor_check as fc
    fc.run(dm)                # dm = directional_metrics.csv
"""
import numpy as np, pandas as pd

DIRS = ["N_uptown", "E_east", "S_downtown", "W_west"]


def _kmo(R):
    """Kaiser-Meyer-Olkin sampling adequacy from a correlation matrix."""
    Ri = np.linalg.pinv(R)
    d = np.sqrt(np.diag(Ri))
    P = -Ri / np.outer(d, d)          # partial correlations
    np.fill_diagonal(P, 0)
    Roff = R.copy()
    np.fill_diagonal(Roff, 0)
    overall = (Roff ** 2).sum() / ((Roff ** 2).sum() + (P ** 2).sum())
    per = ((Roff ** 2).sum(axis=0) /
           ((Roff ** 2).sum(axis=0) + (P ** 2).sum(axis=0)))
    return overall, per


def _bartlett(R, n):
    """Bartlett's test of sphericity: is R distinguishable from identity?"""
    from scipy.stats import chi2
    p = R.shape[0]
    sign, logdet = np.linalg.slogdet(R)
    stat = -((n - 1) - (2 * p + 5) / 6) * logdet
    df = p * (p - 1) / 2
    return stat, df, chi2.sf(stat, df)


def analyse(X, label):
    from sklearn.decomposition import FactorAnalysis, PCA

    X = X.dropna()
    n, p = X.shape
    Z = (X - X.mean()) / X.std(ddof=1)
    R = np.corrcoef(Z.values, rowvar=False)

    print(f"\n{'='*66}\n{label}   n={n}, {p} indicators\n{'='*66}")
    print("\ncorrelation matrix:")
    print(pd.DataFrame(R, index=X.columns, columns=X.columns).round(3).to_string())
    print("  Adjacent directions share a 90-degree quadrant; opposite ones")
    print("  share nothing. Structure here is partly geometric by design.")

    kmo, per = _kmo(R)
    stat, df, pv = _bartlett(R, n)
    print(f"\nKMO overall {kmo:.3f}", end="  ")
    print("(<0.5 unacceptable, 0.5-0.7 mediocre, >0.8 good)")
    print("  per indicator:", dict(zip(X.columns, per.round(3))))
    print(f"Bartlett chi2={stat:.1f} df={df:.0f} p={pv:.2e}"
          f"  {'-> correlated enough to factor' if pv < 0.05 else '-> NOT factorable'}")
    if kmo < 0.5:
        print("  KMO below 0.5: a factor solution here is not interpretable.")

    ev = np.linalg.eigvalsh(R)[::-1]
    print(f"\neigenvalues: {np.round(ev, 3)}")
    print(f"variance explained: {np.round(100*ev/ev.sum(), 1)} %")
    keep = int((ev > 1).sum())
    print(f"Kaiser criterion (eigenvalue > 1): {keep} factor(s)")
    print(f"first factor alone explains {100*ev[0]/ev.sum():.1f}% of variance")

    # Parallel analysis -- Kaiser over-retains, especially with few indicators.
    rng = np.random.default_rng(42)
    sim = np.array([np.linalg.eigvalsh(np.corrcoef(
        rng.standard_normal((n, p)), rowvar=False))[::-1] for _ in range(500)])
    thr = np.percentile(sim, 95, axis=0)
    npar = int((ev > thr).sum())
    print(f"parallel analysis (95th pct of random data): {np.round(thr, 3)}")
    print(f"  -> retain {npar} factor(s)")

    nf = max(1, min(npar, p - 1))
    fa = FactorAnalysis(n_components=nf, rotation="varimax" if nf > 1 else None,
                        random_state=42).fit(Z.values)
    load = pd.DataFrame(fa.components_.T, index=X.columns,
                        columns=[f"F{i+1}" for i in range(nf)])
    print(f"\nloadings ({nf} factor{'s' if nf > 1 else ''}, "
          f"{'varimax' if nf > 1 else 'unrotated'}):")
    print(load.round(3).to_string())
    comm = (load ** 2).sum(axis=1)
    print("communalities:", comm.round(3).to_dict())

    if nf == 1:
        print("\n  One factor: the four directional values at a node are one")
        print("  number plus noise. The 360 index is a sufficient summary and")
        print("  direction adds no dimension -- though it can still shift the")
        print("  LEVEL, which the marginal medians test separately.")
    else:
        print("\n  More than one factor: check whether the split follows the")
        print("  street axes (N/S on one factor, E/W on the other). If it")
        print("  does, along-street and cross-street views are measuring")
        print("  different things and should be reported separately.")
    return load, ev


def run(dm, metrics=None):
    for m in ["GVI", "VEI"]:
        w = dm.pivot_table(index="node_id", columns="direction", values=m)
        cols = [c for c in DIRS if c in w.columns]
        if len(cols) < 3:
            print(f"skipping {m}: only {len(cols)} directions present")
            continue
        analyse(w[cols], f"{m} across travel directions")

        # How much does a node's own 360 value predict its directional ones?
        if "full360" in w.columns:
            r = w[cols].corrwith(w["full360"], method="spearman")
            print(f"\nrho of each direction with the same node's 360 {m}:")
            print(r.round(3).to_string())

    # A joint solution over both metrics asks a different question: are
    # greenness and enclosure separate constructs, or one "openness" axis?
    if metrics is not None:
        cand = [c for c in ["GVI", "VEI", "HW_ratio", "H_m", "W_facade",
                            "scaffold_frac"] if c in metrics.columns]
        if len(cand) >= 3:
            analyse(metrics[cand], "streetscape variables (joint)")
            print("\n  If GVI and VEI load on separate factors they are")
            print("  distinct constructs, which is the assumption the paper")
            print("  makes by reporting them separately.")
