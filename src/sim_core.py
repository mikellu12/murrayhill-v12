"""The Street Interface Matrix exactly as the manuscript defines it.

Section 2.7 of morphology_streetinterfacematrix (24 Aug 2026):

    M_i = I_i^a * Y_i^b * D_i^c * Omega_i

    I_i = sigmoid(kappa_I * (I_raw - tau_I))
          I_raw = a1*(V_nat/V_built) + a2*GVI_eye + a3*GMI
    Y_i = b1*V_sign + b2*(1 - SVF) + b3*SFV          <- no sigmoid
    D_i = sigmoid(kappa_D * (D_raw - tau_D))
          D_raw = c1*V_pave + c2*SFV + c3*GFAPI
    Omega_i = exp(-psi * max(0, H/W - Omega_th))

Three things here are easy to get wrong, so they are stated once:

MULTIPLICATIVE, NOT ADDITIVE. Every earlier implementation in this repo
composed the three dimensions as a weighted sum. The manuscript is explicit
that this is the error being corrected -- "if physical utility or walkability
drops to zero the entire index collapses", which a sum cannot express. The
exponents are elasticities and must sum to 1; they are not mixing weights.

ONLY I AND D ARE SIGMOIDAL. Y is linear. That asymmetry is deliberate:
greenery and walkability have activation thresholds below which they do not
register (tau_I = 0.20 foveal greenness, tau_D = 0.15 sidewalk clearance),
whereas legibility has no such floor.

OMEGA IS A DISCOUNT, NOT A DIMENSION. It is 1.0 everywhere H/W <= 2.0 and
decays only in deep canyons. Without a measured H/W it cannot be computed,
and this module refuses to invent one -- see omega().

SFV enters Y and D_raw both, as the manuscript specifies. That is not a
double-count to be corrected; it is why facade variation carries more
influence than any other single input.
"""
import numpy as np

# The manuscript writes "sub-index weights alpha_k = 1.0" in the same sentence
# as "a_i + b_i + c_i = 1.0". Read as a sum constraint, I_raw stays on the same
# scale as its components, which is what makes tau_I = 0.20 comparable to the
# quoted GVI_eye values (0.06 in canyons, 0.22 in mid-blocks). Read as "each
# weight is 1.0", I_raw would be a sum of three shares and tau_I would sit far
# below the typical value, saturating the sigmoid. The first reading is used
# here; the alternative is a one-line change and is flagged in the docs.
SUBINDEX_WEIGHTS_SUM_TO_ONE = True


def sigmoid(raw, kappa, tau):
    """Perceptual activation. Below tau the dimension barely registers."""
    return 1.0 / (1.0 + np.exp(-kappa * (np.asarray(raw, float) - tau)))


def omega(hw, psi, threshold, open_one_side=None):
    """Canyon oppression discount, 1.0 until H/W passes the comfort threshold.

    Returns NaN where H/W is missing rather than 1.0. A missing aspect ratio
    is not a comfortable street -- silently substituting the neutral value
    would let every un-measured node keep its full score, which is the same
    class of bug as a stale manifest producing a cheerful zero.

    `open_one_side` is the exception, and it is a different fact: the probe
    found a wall on one side and nothing on the other even under a 25-degree
    fan, so there is no facade-to-facade distance to compute. That is not a
    failed measurement. A street with no opposite wall imposes no canyon
    oppression, which is exactly what this term discounts, so those nodes
    take 1.0 -- the same value the formula already gives any street below
    the threshold. Pass s05's HW_source == "open_one_side".
    """
    hw = np.asarray(hw, float)
    out = np.where(np.isnan(hw), np.nan,
                   np.exp(-psi * np.maximum(0.0, hw - threshold)))
    if open_one_side is not None:
        out = np.where(np.asarray(open_one_side, bool), 1.0, out)
    return out


def imageability(nat_built, gvi_eye, gmi, w, kappa, tau):
    raw = w["nat_built"] * np.asarray(nat_built, float) \
        + w["gvi_eye"] * np.asarray(gvi_eye, float) \
        + w["gmi"] * np.asarray(gmi, float)
    return sigmoid(raw, kappa, tau), raw


def identity(v_sign, svf, sfv, w):
    """Linear in the manuscript -- no sigmoid. 1 - SVF is the enclosure term."""
    return (w["signboard"] * np.asarray(v_sign, float)
            + w["enclosure"] * (1.0 - np.asarray(svf, float))
            + w["sfv"] * np.asarray(sfv, float))


def dependence(v_pave, sfv, gfapi, w, kappa, tau):
    raw = w["sidewalk_paver"] * np.asarray(v_pave, float) \
        + w["sfv"] * np.asarray(sfv, float) \
        + w["gfapi"] * np.asarray(gfapi, float)
    return sigmoid(raw, kappa, tau), raw


def regime_exponents(hw, cfg, porous=None):
    """Section 2.8: elasticities shift with the local morphological regime.

    Mid-block is the one case the manuscript leaves partly open -- it states
    a -> 0.50 and nothing about b or c, so b holds at its global value and c
    takes the remainder. That inference is recorded in config, not buried
    here, so it can be overridden without touching code.

    `porous` selects the third regime, POPS and setback plazas. It is the one
    the manuscript gives no H/W band for, and deliberately so: it describes
    an interface that "transitions from a rigid, sheer street wall to a
    porous, fractured block edge", which is a street with no second wall and
    therefore no aspect ratio to band. Classified on H/W alone the regime is
    unreachable, so it needs its own flag -- pass s05's
    HW_source == "open_one_side", which is that condition measured.
    """
    hw = np.asarray(hw, float)
    g = cfg["exponents"]
    out = {k: np.full(hw.shape, v, float) for k, v in g.items()}
    R = cfg.get("elasticity_by_regime") or {}

    canyon = hw >= cfg.get("canyon_hw", 3.0)
    lo, hi = cfg.get("midblock_hw", [0.8, 1.2])
    midblock = (hw >= lo) & (hw <= hi)
    for mask, key in ((canyon, "avenue_canyon"), (midblock, "covenant_midblock")):
        if key in R:
            for k, v in R[key].items():
                out[k] = np.where(mask, v, out[k])
    for k in out:
        out[k] = np.where(np.isnan(hw), g[k], out[k])

    # POPS is applied last, and must be: it is defined by the ABSENCE of an
    # aspect ratio, so every porous node has hw = NaN and the reset above
    # would otherwise overwrite it back to the global exponents.
    if porous is not None and "pops_setback" in R:
        pm = np.asarray(porous, bool)
        for k, v in R["pops_setback"].items():
            out[k] = np.where(pm, v, out[k])

    total = sum(out.values())
    assert np.allclose(total[~np.isnan(total)], 1.0, atol=1e-9), \
        "elasticities must sum to 1 at every node"
    return out


def matrix_score(I, Y, D, Om, a, b, c):
    """M = I^a * Y^b * D^c * Omega, with 0^positive = 0 kept, not clipped.

    A zero dimension collapsing the score is the behaviour the manuscript
    asks for, so it is not floored. Negative or NaN inputs propagate as NaN
    rather than raising, because a partial run should lose the affected rows
    and keep the rest.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        I, Y, D = (np.clip(np.asarray(x, float), 0.0, None) for x in (I, Y, D))
        return (I ** a) * (Y ** b) * (D ** c) * np.asarray(Om, float)


def effective_dwell(t_base, M, lam):
    """t_effective = t_base * (1 + lambda * M)."""
    return np.asarray(t_base, float) * (1.0 + lam * np.asarray(M, float))
