"""The nine VLM ratings that supply every term in the manuscript's SIM.

    VLM field              manuscript term
    --------------------------------------
    vertical_greenery      V_nat
    vertical_hardscape     V_built
    green_eye_level        GVI_eye
    sky_openness           SVF
    walkable_ground        V_pave
    green_softening        GMI
    signage_detail         V_sign
    facade_variation       SFV
    ground_floor_activity  GFAPI

All nine come from the model. The study is about how a VLM reads a street, so
the ratings are the measurement rather than an estimate of one, and comparison
against segmentation masking is a separate arm run elsewhere.

H/W is deliberately absent. It is a plan-geometry quantity measured from
building footprints and facade width in s05, and a single eye-level view cannot
see the street width in the denominator. The manuscript's section 2.8 does
speak of a "VLM-grounded canyon aspect ratio", but grounding a regime
classification is a different job from producing the number Omega discounts.

ANCHORS ARE WRITTEN OUT PER FIELD, BOTH ENDS. The twelve-field schema this
replaces stated the scale once for all ten fields -- "none/absent at 1 to
dominant/continuous at 7" -- and the fields whose real endpoints did not match
that sentence are the ones that collapsed, several answering a single value on
90 per cent of images. Asked one field at a time with both poles named, over
2,724 half-views, no field exceeds 73 per cent on its modal answer and the
three greenery fields inter-correlate at 0.79/0.75/0.92 rather than the 1.00
that showed the model was writing one number into every slot.

sky_openness replaces the old `enclosure` field rather than rephrasing it.
"How strongly this side's wall encloses the street" asked for a property one
wall does not have, and a single 90-degree view cannot see a canyon. Sky share
is visible in the frame and is what (1 - SVF) needs.
"""

from common import CFG

# field -> (low anchor, high anchor, manuscript term, measured counterpart)
FIELDS = {
    "vertical_greenery": (
        "no canopy, green facade or hedge wall at all",
        "canopy and green facade filling the view",
        "V_nat", "vegetation share over the arc"),
    "vertical_hardscape": (
        "almost no built vertical surface in view",
        "building wall, glazing and columns filling the view",
        "V_built", "building share over the arc"),
    "green_eye_level": (
        "no greenery at or below a standing person's eye",
        "greenery at eye height along the whole frontage",
        "GVI_eye", "vegetation below the horizon"),
    "sky_openness": (
        "almost no sky visible, buildings closing overhead",
        "wide open sky across the whole view",
        "SVF", "SVF_band or theoretical_svf(H/W)"),
    # naming the sidewalk directly beats asking for a walkable share of the
    # ground: 54% on the modal answer against 83% for the share phrasing.
    "walkable_ground": (
        "the sidewalk is very narrow or missing on this side",
        "the sidewalk is very wide, taking up most of the ground in view",
        "V_pave", "sidewalk share over the arc"),
    "green_softening": (
        "greenery does nothing to relieve the enclosure",
        "greenery substantially relieves the enclosure",
        "GMI", None),
    "signage_detail": (
        "blank, no signage, cornices, mouldings or lettering",
        "dense signage, cornices and shopfront lettering",
        "V_sign", None),
    # "changes every few metres" asks about a sequence along the street that a
    # single frame cannot show -- it answered one value on 96% of images, with
    # two distinct values, while carrying 22% of the index. Naming what changes
    # takes it to 42% across four values.
    "facade_variation": (
        "a single unbroken building face, one material, one window pattern",
        "many different building faces, materials, window patterns and "
        "entrances along the street",
        "SFV", None),
    "ground_floor_activity": (
        "continuous blank wall at street level",
        "continuous active glazed shopfronts at street level",
        "GFAPI", None),
    # The manuscript's third Dependence term, and the one this study was
    # missing. Section 2.7 defines D_raw = g1*V_pave + g2*IAS + g3*GFAPI, and
    # IAS as "micro-resting infrastructure and tactile seating ledges in
    # [0,1]". Until this field existed, facade_variation stood in for it --
    # which put SFV in both Y and D and left Place Dependence with no
    # affordance term at all, so D measured walkable ground, an Identity
    # trait, and ground-floor glazing.
    "resting_affordance": (
        "nothing to sit on or lean against anywhere in view",
        "continuous seating and resting places along the whole frontage",
        "IAS", None),
}

MEASURED_TWIN = {k: v[3] for k, v in FIELDS.items() if v[3]}
JUDGEMENT_ONLY = [k for k, v in FIELDS.items() if not v[3]]

SYSTEM = ("You are an expert urban morphologist evaluating streetscape "
          "quality at eye-level (1.5m).")


# Named to the model in every prompt; see config.yaml: prompt_place.
# "Rate this street view." with no place name unless config supplies one; see
# config.yaml: prompt_place for why none is the right default.
_place = CFG.get("prompt_place") or ""
PLACE = f"{_place} " if _place else ""


def prompt(field):
    """One field, both poles named, an explicit instruction to use the range."""
    lo, hi = FIELDS[field][0], FIELDS[field][1]
    return (f"Rate this {PLACE}street view. Reply with ONE JSON object and "
            f"nothing else: {{\"{field}\": <1-7>}} where 1 is {lo} and 7 is "
            f"{hi}. Use the whole 1-7 range.")


def all_prompts():
    return {f: prompt(f) for f in FIELDS}
