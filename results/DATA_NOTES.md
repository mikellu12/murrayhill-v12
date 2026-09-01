# Reading these tables

Four files, two per study area.

    results/tables/                 Murray Hill, Manhattan   766 nodes,  3,064 half-views
    results/london/tables/          City of London         1,806 nodes, 6,422 views

`vlm_observations.csv` is what was observed: the ten VLM ratings per view, the
full seven-rung probability distribution behind each, and the measured geometry
where it exists. `vlm_calculations.csv` is what was derived from them: the
normalised terms, the three dimensions, and M. Join them on `file`.

## Read these before using the numbers

**The two cities are not yet on one instrument.** Every rating prompt used to
name a city -- "Rate this Manhattan street view" -- in both study areas. London
was re-rated with a placeless prompt, "Rate this street view", so the two
cities are compared fairly only after Murray Hill is re-rated the same way.
That run has not happened. London's internal numbers are unaffected; a
Murray Hill against London difference is not yet clean.

**Use M_noA to compare the cities, not M.** London has no building footprints,
so it cannot compute the canyon penalty A_i and its A_i is 1 everywhere.
Murray Hill can and does. Reading one city's M against the other's compares the
presence of the term rather than the streets: Murray Hill's median is 0.630
with the penalty and 0.661 without.

**M_local answers a different question and reverses the ordering.** Local tau
recentres the sigmoid on each city's own median, so it measures relativity
within a study area. On that column London is 0.517 and Murray Hill 0.439 --
the opposite of the global-tau comparison, and both are correct for what they
ask. Use the global columns across cities and the local ones for ranking
streets inside one.

**London's 90-degree and 180-degree views are not on one scale.** A pedestrian
way is rendered as a single 180-degree strip and a vehicular street as two
90-degree halves, because Street View is captured from the roadway and the two
situations are geometrically different. `side` is L/R for halves and F for a
strip. Report them separately; pooled means mix two fields of view.
Murray Hill is entirely 90-degree: it has no pedestrian ways.

**The `indoor` flag is a candidate, not a verdict.** 269 London frames (4.2%)
are flagged as interiors or station subways by a segmentation rule. Against 16
hand-labelled frames it scored 0.70 precision and 0.70 recall: it wrongly flags
roofed public streets such as Leadenhall Market, and misses about a third of
true interiors. Excluding the flagged set moves the city-wide median M from
0.497 to 0.503, and individual streets by up to 0.22. Do not filter on it
without looking at the frames.

**Three London fields sit against a rail.** 82% of frames read rung 1 on
resting_affordance, 74% rung 1 on vertical_greenery, 61% rung 7 on
vertical_hardscape. The rung sets were calibrated on Murray Hill's range. Those
fields carry little between-street variance in London -- partly because the
City really is that extreme, partly because the scale bottoms out.

**London's imagery spans many years.** Of the 45 streets with ten or more
nodes, none was captured in a single year; the median street spans 10 years and
the widest 13. A street's profile is a composite of frontages photographed up
to a decade apart. `pano_date` is in the manifest for anyone who wants to
condition on it. See docs/london.md.

## How a rating becomes a number

Each field is read from the model's probability over the seven rungs, not from
a generated token. The columns `<field>_p1` .. `<field>_p7` are that
distribution. The value used in M is the **pruned interpolated median**: drop
the single least-likely rung, renormalise, then take the quantile where the
cumulative crosses one half. Expected value is not used -- the rungs are
ordered categories, so a mean would assert that the step from 2 to 3 equals the
step from 6 to 7. `<field>_ev` and `<field>_argmax` are kept for comparison.
