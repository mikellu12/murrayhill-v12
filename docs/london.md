# The London frame: what the imagery is, and what it costs

The City of London extension reuses the Murray Hill instrument unchanged.
What differs is the imagery underneath it, and the difference is not
incidental -- it is the main threat to any comparison drawn between the two
cities, so it is stated here rather than left to a footnote.

## Murray Hill is one capture; London is fifteen years of them

Google drove Murray Hill in a single pass, so every node there carries a
2026-04 panorama and the study area is a synchronic slice. The pipeline
enforces that with an exact capture-date filter, which costs 22 nodes.

The City of London has no equivalent pass. Its 1,806 nodes carry panoramas
from **2008 to 2026**, median 2022, interquartile range 2021-2024. No single
date covers more than 464 of them, so the Murray Hill rule -- applied
literally -- would discard three quarters of the frame.

Every node is therefore kept, and the spread is reported as a limitation.
`pano_date` is carried through the manifest so any later analysis can
condition on it.

## The year spread is the larger problem, and it is within-street

Aggregated, 15.6% of nodes predate 2020 and 4.8% predate 2015. That
understates it. Of the 45 streets carrying ten or more nodes, **not one was
captured in a single year**; the median street spans 10 years of capture and
the widest spans 13.

So a street's SIM profile is not a reading of that street at a moment. It is
a composite of frontages photographed up to a decade apart, in a district
that rebuilt substantially over the period. Where a street-level result
turns on facade condition, ground-floor activity or scaffolding -- all of
which change on a shorter cycle than the capture span -- the reading should
be treated as an average over the period, not a snapshot.

## Season is the smaller problem, but it concentrates

225 nodes (12.5%) were captured November-March, December alone accounting
for 115. Spread across 90 streets, that sounds harmless. At street level it
is not:

| street              | nodes | leaf-off |
|---------------------|------:|---------:|
| Cornhill            |    16 |    93.8% |
| Lothbury            |    13 |    76.9% |
| King William Street |    27 |    63.0% |
| Threadneedle Street |    14 |    50.0% |
| Aldermanbury        |    12 |    41.7% |
| Leadenhall Street   |    15 |    40.0% |

Three of the 45 streets are majority leaf-off and seven are above a quarter,
while nineteen contain none at all. A comparison of eye-level greenery
between Cornhill and a fully leaf-on street is therefore partly a comparison
of season, and any greenery result that ranks those streets against each
other must say so.

This is a caveat on street-level greenery contrasts specifically. It does
not bear on the aggregate distribution, where 87.5% of nodes are leaf-on,
nor on the non-greenery dimensions.

## Why not filter it out

Three reasons, in order of weight.

The manuscript does not restrict the measurement to leaf-on imagery. A
seasonal filter would be a rule invented at analysis time, not a stated part
of the method.

It would not fix the confound it targets. Dropping leaf-off nodes removes
Cornhill's coverage almost entirely (15 of 16 nodes) rather than making
Cornhill comparable -- the street would leave the frame instead of the
season leaving the street.

And it would cost the more important spread nothing. Year, not season, is
where London's imagery is most heterogeneous, and no date filter narrow
enough to control season leaves a usable frame.

Conditioning after the fact is available to anything that needs it, and is
the honest place for it.

## Who took the pictures

192 of the 1,806 nodes (10.6%) draw on a panorama somebody uploaded rather than
one Google captured. Most are individuals -- one contributor accounts for 29
nodes -- and some are commercial photosphere firms.

They are full 360-degree photospheres, so the directional split holds on them:
mean between-heading difference 52.5 against Google's 50.4, closest pair 43.5
against 41.5, and no blank or duplicated heading in either group. Street View
ingests uploads only in equirectangular form, so this is structural rather than
luck.

**They do not come in runs.** 128 of the 157 consecutive-user stretches are a
single isolated node and the longest is three. So along one street the sequence
alternates between car panoramas and the occasional photosphere taken on a
different day, at a different height, with a different camera -- heterogeneity
within a street, on top of the year spread above. Four streets lean on them
heavily: Throgmorton Avenue 5 of 7 nodes, Custom House Walkway 4 of 7,
St Paul's Churchyard 5 of 10, Saint Mary Axe 5 of 12.

Provenance is recorded in `provenance.csv` and is **not** used as a filter.
Against hand-labelled frames, excluding every upload would delete four
legitimate street views while still missing sixteen interiors that Google
captured itself -- it drives station subways. 84 of the 150 off-street nodes
are uploads, which corroborates the ground test without deciding anything.

## What the off-street filter cannot catch

`offstreet_flag.csv` keys on visible public ground, which separates interiors,
subways and vehicle photospheres in one test because none of them has pavement
in frame. It cannot separate public pavement from private pavement. A patio
reached 0.640 ground share and is kept; it is a paved outdoor space that is not
a street. Segmentation has no class for who owns the ground, so this residual
is logged rather than thresholded away -- and it fails toward keeping a frame
rather than deleting a real street, which is the safer direction.
