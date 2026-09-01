"""Seven verbal anchors per field, not two.

The two-anchor prompts name 1 and 7 and leave the model to interpolate. It
never answers 4 on any of the nine fields, and reading the logits shows why:
the midpoint carries real probability -- 16 per cent on sky_openness -- but is
never the single most likely digit, so greedy decoding cannot emit it. Part of
that is the decoder. Part is that the model was given no language for the
middle of a scale it was asked to use.

Each step below is a real rung, not padding: it describes a street you could
photograph, and the progression is monotone in the quantity the manuscript
defines. 4 is deliberately the balance point -- half and half, neither
dominant -- because that is what a midpoint means and what the model had no
way to say.

Longer prompts are the thing that broke the twelve-field schema, so this is a
hypothesis to test, not an improvement to assume: the length here is in one
field's scale rather than in eleven competing fields, which may or may not
behave the same way. tools/scale_probe.py measures it.
"""

from common import CFG

SCALE = {
 "vertical_greenery": [
   "no canopy, green facade or hedge wall anywhere in view",
   "one or two bare or sparse trees and nothing else",
   "scattered canopy over part of the frontage",
   "canopy or green facade along about half the frontage",
   "near-continuous canopy along the frontage",
   "dense canopy overhead plus green facade or hedge wall",
   "canopy and green facade filling the view"],
 # rewritten. The first version described BOTH SIDES of the street --
 # "built frontage on both sides", "tall continuous walls on both sides" --
 # which a 90-degree half-view does not show. It is the defect that sank the
 # old `enclosure` field: a property of the canyon asked of one frontage.
 # Measured against building share over the arc it fell from +0.373 to
 # +0.278. These rungs describe the one frontage in frame, by how much of
 # the view it fills.
 "vertical_hardscape": [
   "almost no built surface in view",
   "a low building or a gap, most of the view open",
   "a modest frontage occupying part of the view",
   "a frontage filling about half the view",
   "a tall frontage filling most of the view",
   "a tall unbroken wall across nearly the whole view",
   "building wall, glazing and columns filling the view"],
 # rewritten. The first version counted objects -- "a single planter or
 # tree pit", "a few planters or low shrubs" -- which is a harder question
 # than how much greenery is there, and the field fell from +0.737 under two
 # anchors to +0.649. It reads well when asked about amount, so these rungs
 # ask about amount and let the model decide what counts.
 "green_eye_level": [
   "no greenery at or below a standing person's eye",
   "a trace of greenery at eye height, easily missed",
   "a little greenery at eye height, here and there",
   "greenery at eye height along about half the frontage",
   "greenery at eye height along most of the frontage",
   "greenery at eye height along nearly all of it",
   "greenery at eye height along the whole frontage, in layers"],
 # rewritten. The first version described walls at 2, 3 and 4 -- "between
 # tall walls", "the walls dominate", "sky and built edges in balance" --
 # which asks the enclosure question, not the sky one. Its correlation with
 # measured sky share went to -0.034. These rungs vary one quantity only.
 "sky_openness": [
   "almost no sky visible",
   "only a narrow strip of sky",
   "a small amount of sky above",
   "a moderate amount of sky",
   "a good deal of sky",
   "sky across most of the view",
   "wide open sky across the whole view"],
 # rewritten. Seven rungs carried three metrics at once: WIDTH (1, 2, 4, 5, 6),
 # OBSTRUCTION (3) and SHARE OF THE VIEW (7), so the model could not tell which
 # question it was answering. It was the flattest field in the set -- not one
 # image of 3,064 had a rung above 0.5, and 54% were ties within 0.05. Extent of
 # unobstructed sidewalk is what the manuscript's V_pave actually defines.
 # A/B against sidewalk + curb_edge share: +0.188 -> +0.333, 95% CI
 # [+0.005, +0.246] clustered on face. What counts moved to DEFINITION.
 "walkable_ground": [
   "no usable sidewalk on this side",
   "a trace of clear sidewalk, blocked almost throughout",
   "clear sidewalk along a small part of the frontage",
   "clear sidewalk along about half the frontage",
   "clear sidewalk along most of the frontage",
   "clear sidewalk along nearly all of the frontage",
   "clear sidewalk along the whole frontage"],
 # rewritten. Every rung asked for a judgement about an EFFECT ("relieves the
 # enclosure", "softens the walls") and rungs 3-4 described a ratio between two
 # subjects. That is an inference, not an observation, and it produced the worst
 # distribution in the set: 60% of images two-peaked and 58% of stored answers
 # on a rung the model ranked below its own runner-up -- it was answering "is
 # there greenery" and "are there walls" alternately rather than combining them.
 #
 # Recast as one observable quantity: how much of the building surface has
 # greenery in front of it. That is the manuscript's own heuristic -- "score 1.00
 # if vertical green walls or dense planters completely cover the lower 3 metres
 # of the building facades" -- and it is coverage of a surface, not a product of
 # two terms. "Structural interaction variable" in the paper names the
 # PHENOMENON, greenery changing how hard surfaces are perceived; it does not
 # prescribe a multiplicative form. GMI is therefore a SUBSET of V_nat, greenery
 # in a particular place, and its correlation with total greenery is what a
 # correct measure looks like rather than evidence it has collapsed into one.
 #
 # Bimodality 56% -> 8%, decisive 1% -> 9%. Twin is greenery on the lower 3 m of
 # facade, computed per node from the measured facade distance by
 # tools/seg_gmi_band.py: +0.446.
 "green_softening": [
   "the building surface is bare, no greenery in front of it",
   "a trace of greenery in front of the building surface",
   "greenery in front of a small part of the building surface",
   "greenery in front of about half the building surface",
   "greenery in front of most of the building surface",
   "greenery in front of nearly all the building surface",
   "greenery hides the building surface almost completely"],
 "signage_detail": [
   "blank, no signage, cornices, mouldings or lettering",
   "a single sign or a plain entrance",
   "occasional signage on an otherwise plain frontage",
   "signage and architectural detail along about half the frontage",
   "frequent signage, cornices or mouldings",
   "dense signage and articulated facade detail",
   "dense signage, cornices and shopfront lettering throughout"],
 "facade_variation": [
   "a single unbroken building face, one material, one window pattern",
   "one building face with a slight change of material",
   "two building faces, similar in material and rhythm",
   "three or four distinct faces along the frontage",
   "frequent changes of material, window pattern or entrance",
   "many different faces, each clearly its own building",
   "many different building faces, materials, window patterns and entrances"],
 "ground_floor_activity": [
   "continuous blank wall at street level",
   "blank wall with a single doorway",
   "mostly blank with occasional windows or entrances",
   "about half glazed shopfront, half blank wall",
   "mostly active glazed frontage",
   "continuous shopfronts with entrances and displays",
   "continuous active glazed shopfronts at street level"],
 # IAS. The rungs count how much of the frontage offers somewhere to stop,
 # not how good it is: stoops, ledges, low walls, benches, wide window sills.
 # One frontage only -- this is a 90-degree half-view and the far side is not
 # in it, which is the defect that sank the first vertical_hardscape set.
 # rewritten. Rungs 2-4 COUNTED objects ("a single step", "one stoop", "a few
 # stoops") while 5-7 switched to EXTENT ("along much of the frontage") -- two
 # metrics on one ladder. green_eye_level had the same defect and moved +0.649
 # -> +0.737 against measured pixels when it stopped counting planters.
 # Here the A/B left accuracy unchanged (+0.134 -> +0.100, CI spans zero) and
 # bimodality fell 27% -> 6%, so this is kept for the cleaner distribution and
 # NOT for a score it did not win. IAS stays the weakest field in the set;
 # against stoop_stair + bench_seating share it reaches only +0.14, against
 # +0.72 for greenery, and no rewording or readout has moved it. That is a
 # limit of what the model can see in a 90-degree view, not of the wording.
 "resting_affordance": [
   "nothing to sit or lean on anywhere in view",
   "a trace: one perchable ledge or step, nothing more",
   "places to sit along a small part of the frontage",
   "places to sit along about half the frontage",
   "places to sit along most of the frontage",
   "places to sit along nearly all of the frontage",
   "continuous places to sit along the whole frontage"],
}


# What COUNTS, deliberately kept out of the rungs. The rungs say HOW MUCH, and
# conflating the two is the defect the rewrites above remove -- so a field
# needing an operational definition gets one sentence here rather than smuggling
# criteria into rung text. Both are the manuscript appendix's, not invented.
DEFINITION = {
 "resting_affordance":
   "Places to sit means masonry stoops 0.9-1.5 m high, ledges or planter rims "
   "0.4-0.6 m high and at least 0.3 m deep, low walls, or benches. Elements "
   "blocked by spikes or railings do not count.",
 "walkable_ground":
   "Clear sidewalk means the continuous unobstructed paving a pedestrian can "
   "walk on. Scaffolding, bins, parked vehicles and construction reduce it.",
}


# Named to the model in every prompt; see config.yaml: prompt_place.
# "Rate this street view." with no place name unless config supplies one; see
# config.yaml: prompt_place for why none is the right default.
_place = CFG.get("prompt_place") or ""
PLACE = f"{_place} " if _place else ""


def prompt7(field):
    """One field, every rung named."""
    # Instruction first, scale second -- the same order as the two-anchor
    # prompt. The first version led with the scale and ended with the JSON
    # instruction, so a comparison between them confounded length with
    # structure.
    steps = "\n".join(f"{i + 1} = {s}" for i, s in enumerate(SCALE[field]))
    d = DEFINITION.get(field)
    d = f"{d} " if d else ""
    return (f"Rate this {PLACE}street view. {d}Reply with ONE JSON object "
            f"and nothing else: {{\"{field}\": <1-7>}}, using this scale:\n\n"
            f"{steps}")
