"""A walk-through page: the view, the scores, and what the model said.

Writes a self-contained HTML file that steps along one street node by node --
the rendered view on the left, the ten field ratings and M on the right, and
the model's own description beneath them.

FRAMES ARE INLINED, so the page works wherever it is opened. Referencing them
by relative path made a smaller file that only rendered from results/figures,
and moving it produced a page of broken images rather than an error --  which
is the first thing that happens to any file.

LOCAL ONLY. Street View imagery is not redistributable: Google caps caching at
30 days, and the derived measurements are ours to publish while the photographs
are not. Inlining makes the file portable, which makes it easier to share by
accident -- so it must not be uploaded or handed out. --link restores the
reference-on-disk build.

It degrades. Ratings, descriptions and M each appear if their table exists and
are quietly absent if not, so the page can be built and looked at before the
qualitative pass has finished rather than only afterwards.

ORDER COMES FROM THE FILENAME SEQUENCE within a corridor, which is the one
ordering that follows the walk on every street: chain_pos_m runs in an
arbitrary direction per chain, and the frame's own seq restarts inside each
corridor of a split street.

THE RATINGS ARE RECOMPUTED, not read off. The stored point columns are
round(EV), which this study does not use -- a rung index is ordinal, so its
summary has to be a quantile. Where the seven rung probabilities are present
they are pruned once and read at the interpolated median, the same numbers
sim_compute builds M from, so the panel and the map cannot disagree. Reading
the stored column instead put round(EV) beside an M built from the median.

BOTH HALVES OF A VEHICULAR NODE ARE SHOWN. A street rendered as 90 degree
halves has two frames per node; keeping one dropped a frontage the panel's own
scores still averaged. The two are composed with a seam, as walk_gif does.

THE MAST CALIBRATION IS PER FRAME. A corridor can mix geometries -- Cannon
Street has a 180 degree strip at n00290 between 90 degree halves -- so one set
for the street leaves the odd frame's mast standing, and the mast is the one
thing in the view the model provably never saw.

PAYLOAD LAST. The interface and its script are written before the frames, and
the frames are read on DOMContentLoaded. A viewer that truncates a large file
then loses only the tail: the page says how many frames arrived and walks them.
Ordered the other way, one cut blanked the whole page.

    .venv/Scripts/python tools/build_walk_interface.py --street east_38th_street
    SIM_CONFIG=config_london.yaml .venv/Scripts/python \
        tools/build_walk_interface.py --street london_wall
"""
import argparse
import html
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as _mpl  # get_cmap moved in matplotlib 3.9

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
from common import CFG, PROC, RAW, RES, banner
from sim_readout import K, interpolated_median, prune_once

NAME = re.compile(r"^(\d+)_(n\d+)_([NESW])(?:_([LRF]))?\.jpg$")
FIELDS = ["vertical_greenery", "green_eye_level", "green_softening",
          "vertical_hardscape", "sky_openness", "signage_detail",
          "walkable_ground", "resting_affordance", "ground_floor_activity",
          "facade_variation"]
DIM = {"I_raw": "imageability", "Y": "identity", "D_raw": "dependence"}
QCOLS = ["scene", "greenery", "ground", "frontage"]

# WHICH FIELD FEEDS WHICH TERM, read off sim_compute's formulas rather than
# guessed: I_raw from nat_built + GVI_eye + GMI, Y from V_sign + (1-SVF) + SFV,
# D_raw from V_pave + IAS + GFAPI. Two fields do not enter as themselves and
# the panel says so, because a reader who sees sky_openness under identity will
# otherwise assume the panel is wrong: vertical_hardscape enters only through
# its ratio with vertical_greenery, and sky_openness enters inverted, as
# enclosure. Without the grouping the ten fields are a list; with it they are
# the three terms taken apart.
# The flag marks a field that pushes its term the OTHER way: vertical_hardscape
# enters only as vg/(vg+vh), so more hardscape lowers imageability, and
# sky_openness enters as (1-SVF), so more sky lowers identity. Both are drawn
# in red -- a reader scanning the panel needs to know the arrow points the
# other way, and does not need the algebra.
GROUPS = [
    ("imageability", "I_raw",
     [("vertical_greenery", 0), ("vertical_hardscape", 1),
      ("green_eye_level", 0), ("green_softening", 0)]),
    ("identity", "Y",
     [("signage_detail", 0), ("sky_openness", 1), ("facade_variation", 0)]),
    ("dependence", "D_raw",
     [("walkable_ground", 0), ("resting_affordance", 0),
      ("ground_floor_activity", 0)]),
]


def load(path, cols=None):
    if not path.exists():
        print(f"  (no {path.name})")
        return None
    d = pd.read_csv(path)
    return d[[c for c in cols if c in d.columns]] if cols else d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--street", required=True,
                    help="one street folder, several separated by commas, or 'all' for every street in --src")
    ap.add_argument("--walk", default=None)
    ap.add_argument("--exclude", default="",
                    help="street folders to leave out, comma separated")
    ap.add_argument("--src", type=Path, default=None)
    ap.add_argument("--ratings", type=Path, default=None)
    ap.add_argument("--descriptions", type=Path, default=None)
    ap.add_argument("--greenery", type=Path, default=None,
                    help="a re-run of one question, merged over the "
                         "descriptions table without touching it on disk")
    ap.add_argument("--calc", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--embed", action="store_true", default=True,
                    help="inline the frames as data URIs so the file works "
                         "wherever it is opened")
    ap.add_argument("--link", dest="embed", action="store_false",
                    help="reference the JPEGs on disk instead; smaller, but "
                         "only opens correctly from inside the repository")
    ap.add_argument("--mast-set", default="auto",
                    help="mast calibration for the erase: 'auto' picks it per "
                         "frame from the geometry, a set name forces one, "
                         "'none' leaves the camera mast in frame")
    ap.add_argument("--ramp-floor", type=float, default=0.40,
                    help="drop the darkest fraction of each term ramp in the "
                         "panel bars, for legibility; 0 keeps the exact ramp "
                         "the term maps use")
    ap.add_argument("--embed-width", type=int, default=680)
    ap.add_argument("--quality", type=int, default=62)
    ap.add_argument("--frames-dir", type=Path, default=None,
                    help="write the cleaned frames as files beside the page "
                         "and link them, instead of inlining every one. Over a "
                         "server the browser then fetches a frame per step, so "
                         "the page can carry EVERY node rather than every "
                         "second one -- inlining is what forced the thinning.")
    ap.add_argument("--both-walks", action="store_true",
                    help="carry both traversals of each street so the page "
                         "can switch walking direction. Doubles the frames; "
                         "meant for the served build, and for a study area "
                         "whose walks are clean opposites -- London's are "
                         "not, so it stays off there.")
    ap.add_argument("--phone", action="store_true",
                    help="a build a phone can hold: fewer nodes, smaller "
                         "frames. 25 MB of base64 is not something a phone "
                         "browser opens comfortably from a message.")
    ap.add_argument("--every", type=int, default=1,
                    help="keep every Nth node, to hold the file small enough "
                         "that a viewer does not truncate it")
    args = ap.parse_args()
    if args.phone:
        args.embed_width = min(args.embed_width, 560)
        args.quality = min(args.quality, 55)
        args.every = max(args.every, 3)
        print(f"phone build: every {args.every}, {args.embed_width}px, "
              f"q{args.quality}")
    if str(args.mast_set).lower() == "none":
        args.mast_set = None
    area = CFG.get("study_area_name", "study area")
    banner(f"walk-through page: {args.street}")

    src = args.src or (RAW / "svi_180")
    if args.street.strip().lower() == "all":
        names = sorted(d.name for d in src.iterdir() if d.is_dir())
    else:
        names = [t.strip() for t in args.street.split(",") if t.strip()]
    drop = {t.strip() for t in args.exclude.split(",") if t.strip()}
    if drop:
        names = [n for n in names if n not in drop]
        print(f"excluded: {', '.join(sorted(drop))}")
    missing = [n for n in names if not (src / n).is_dir()]
    if missing:
        sys.exit(f"no such street folder: {', '.join(missing)}")
    print(f"{len(names)} street(s): {', '.join(names)}")

    def _dedupe_runs(g):
        """One node per position along a run; the rest marked auxiliary.

        Parallel service-road strips sit beside the main line at the same
        block position, and walking them in sequence hops across the street.
        Grouping by proximity along the run and keeping the node nearest its
        centre line leaves one node per position, in order.
        """
        if len(g) < 8:
            return g
        import numpy as _np
        lat0 = float(g.lat.mean())
        _x = (g.lon - g.lon.mean()).to_numpy() * 111320 * _np.cos(_np.radians(lat0))
        _y = (g.lat - g.lat.mean()).to_numpy() * 110540
        _v = _np.linalg.svd(_np.c_[_x, _y], full_matrices=False)[2][0]
        along = _x * _v[0] + _y * _v[1]
        latr = _x * (-_v[1]) + _y * _v[0]
        if latr.max() - latr.min() <= 8.0:
            return g
        gap = 10.0
        order = _np.argsort(along)
        keep = _np.zeros(len(g), dtype=bool)
        grp = [order[0]]
        for k in range(1, len(order)):
            if along[order[k]] - along[grp[-1]] > gap:
                arr = _np.array(grp)
                keep[arr[_np.argmin(_np.abs(latr[arr]))]] = True
                grp = []
            grp.append(order[k])
        if grp:
            arr = _np.array(grp)
            keep[arr[_np.argmin(_np.abs(latr[arr]))]] = True
        g = g.copy()
        g.loc[g.index[~keep], "auxflag"] = 1
        return g

    def frames_for(name):
        """Each walk of one street as (walk_name, nodes, files) tuples.

        One walk by default, both with --both-walks. The reverse walk is not
        the same frames backwards -- it is the OTHER render, facing the other
        way down the street -- so switching direction needs both folders.
        """
        base = src / name
        walks = ([base / args.walk] if args.walk
                 else sorted(q for q in base.iterdir() if q.is_dir()))
        if not args.both_walks:
            walks = walks[:1]
        out_walks = []
        for w in walks:
            r = _one_walk(w, name)
            if r is not None:
                out_walks.append((w.name,) + r)
        return out_walks

    def _one_walk(w, name):
        files, seqs = {}, {}
        for q in sorted(w.glob("*.jpg")):
            m = NAME.match(q.name)
            if m:
                files.setdefault(m.group(2), {})[m.group(4) or "F"] = q
                seqs[m.group(2)] = int(m.group(1))
        if not files:
            return None
        nf = ALLNODES[ALLNODES.node_id.isin(files)].copy()
        if not len(nf):
            return None
        nf["fseq"] = nf.node_id.map(seqs)
        # THE FILENAME SEQUENCE IS THE WALK, wherever it is trustworthy.
        # The exporter numbers every frame of a walk by its projection along
        # the travel bearing, both kerbs interleaved correctly -- so when the
        # seq values are unique, sorting on seq alone IS the walk. Grouping by
        # chain first (added to stop dropping the smaller corridors) walked
        # each chain separately, and on Park Avenue and 1st Avenue, whose one
        # walk spans several chain labels over the same stretch, that meant
        # walking down the street and teleporting back up to walk it again:
        # east kerb, west kerb, east. Chain-grouping is now reserved for the
        # one case that needs it -- duplicated seq values, where the sequence
        # genuinely restarts per corridor and seq alone would interleave them.
        if nf.fseq.is_unique:
            nf = nf.sort_values("fseq")
        else:
            nf = nf.sort_values(["segname", "fseq"])
        # ONE NODE PER POSITION ALONG THE STREET, not one run per street.
        #
        # An avenue's folder can carry parallel runs: the main line of nodes
        # plus strips on the service roads either side. Walked in sequence the
        # view hopped across the street and back, and the arrow pointed at the
        # hop -- so the first fix kept only the largest run. That was wrong
        # here: on 1st Avenue the main run STOPS at the tunnel portal and the
        # northern 120 m exists only as the two flanking runs, so dropping
        # them deleted the north end from the walk entirely.
        #
        # The stride instead takes ONE node per position along the street --
        # the one nearest the street's own centre line -- so it is continuous
        # wherever any node exists, and never doubles back. Everything else is
        # marked auxiliary: still in the file, still on the map, still
        # clickable with its frame and scores, just not part of the stride.
        #
        # auxflag, not _aux: itertuples() renames leading-underscore columns
        # to positional _1, _2, and the flag silently read 0 for every row.
        nf["auxflag"] = 0
        if nf.cstreet.notna().all():
            # the cleaning already resolved which street each node is on and
            # where it sits: nothing to reconstruct, nothing to de-duplicate
            nf = nf.sort_values("cseq")
        elif nf.segname.nunique() > 1:
            # dedupe within each chain; across disjoint chains "position along
            # the street" has no meaning and grouped unrelated nodes together
            parts = []
            for _sg, _g in nf.groupby("segname", sort=False):
                parts.append(_dedupe_runs(_g))
            nf = pd.concat(parts)
        else:
            nf = _dedupe_runs(nf)
        seg = ", ".join(str(x) for x in nf.segname.unique())
        if args.every > 1:
            nf = nf.iloc[::args.every]
        print(f"  {name:<26}{w.name:<16}{len(nf):>4} nodes  ({seg})")
        return nf, files


    # A NODE CAN HAVE TWO FRAMES. A vehicular street is rendered as two 90
    # degree halves, left and right of the direction of travel; a pedestrian
    # way gets one 180 degree strip. Keeping one file per node silently threw
    # the other half away, so half of every London avenue was missing from a
    # page whose scores averaged both halves -- a view that did not match its
    # own numbers. Both halves are composed into one frame instead, seam down
    # the middle, which is the same pairing tools/walk_gif.py makes.
    ALLNODES = pd.read_csv(PROC / "nodes.csv")
    # usable: False never reaches the page -- not as a step, not as a map dot.
    # The tag (tools/node_usability.py) marks tunnel interiors, the viaduct
    # deck, and user-contributed panoramas; a map that shows them anyway
    # re-includes by picture what the calculations exclude by rule.
    UNUSABLE = (set(ALLNODES.loc[~ALLNODES.usable.astype(bool), "node_id"])
                if "usable" in ALLNODES.columns else set())
    if UNUSABLE:
        print(f"  {len(UNUSABLE)} nodes tagged unusable: excluded from the "
              f"walk and the map")
        ALLNODES = ALLNODES[ALLNODES.usable.astype(bool)].copy()
    # NO LEADING UNDERSCORES ON COLUMNS READ BACK THROUGH itertuples(): it
    # renames them to positional _1, _2 and every getattr silently returns the
    # default. It cost this file a broken chain split and a broken street
    # split before the rule was finally applied everywhere.
    #
    # STREET AND SEQUENCE COME FROM cleaned_id, the hand-cleaned labelling
    # carried in nodes.csv: "1st_avenue_west_branch_004" names the street and
    # its position on it. That column already separates the branch from the
    # avenue and gives the tunnel approach its own name, which is exactly what
    # this tool spent several rounds trying to reconstruct from geometry --
    # badly. Where it is absent (54 nodes the cleaning did not cover) the
    # render folder and filename sequence stand in.
    if "cleaned_id" in ALLNODES.columns and ALLNODES.cleaned_id.notna().any():
        cid = ALLNODES.cleaned_id.astype(str)
        ok = ALLNODES.cleaned_id.notna()
        ALLNODES["cstreet"] = cid.str.rsplit("_", n=1).str[0].where(ok)
        ALLNODES["cseq"] = pd.to_numeric(
            cid.str.rsplit("_", n=1).str[1], errors="coerce").where(ok)
    else:
        ALLNODES["cstreet"], ALLNODES["cseq"] = None, None
    ALLNODES["segname"] = (
        ALLNODES.source_id.astype(str).str.rsplit("_", n=1).str[0]
        if "source_id" in ALLNODES.columns and ALLNODES.source_id.notna().any()
        else ALLNODES.get("chain", "all"))

    rt = load(args.ratings or (RES / "tables" / "sim_vlm_180_placeless.csv"))
    if rt is None:
        rt = load(RES / "tables" / "sim_vlm_v3.csv")
    de = load(args.descriptions or (RES / "tables" / "vlm_descriptions_180.csv"))
    if args.greenery and args.greenery.exists() and de is not None:
        g = pd.read_csv(args.greenery)[["file", "greenery"]]
        de = de.drop(columns=[c for c in ["greenery"] if c in de.columns])                .merge(g, on="file", how="left")
        print(f"  greenery from {args.greenery.name}: "
              f"{int(de.greenery.notna().sum())} frames")
    ca = load(args.calc or (RES / "tables" / "vlm_calculations.csv"))

    def with_node_id(d):
        """node_id from the frame path when the table does not carry one.

        The London ratings are keyed only by `file`; without this the merge
        matched nothing and the panel drew empty beside a perfectly good frame,
        which reads as the model having no opinion rather than as a join that
        missed.
        """
        if d is None or "node_id" in d.columns or "file" not in d.columns:
            return d
        d = d.copy()
        d["node_id"] = d.file.astype(str).str.extract(r"(n\d+)")[0]
        return d

    def readout(d):
        """The study's own readout, recomputed from the rung probabilities.

        The stored point columns are round(EV), which this study does not use:
        a rung index is ordinal, so the summary has to be a quantile. Where the
        seven p-columns are present they are pruned once and read at the
        interpolated median -- the same numbers sim_compute builds M from, so
        the panel and the map cannot disagree.
        """
        if d is None:
            return d
        pcols = [[f"{f}_p{k}" for k in K] for f in FIELDS]
        if not all(all(c in d.columns for c in cs) for cs in pcols):
            return d
        d = d.copy()
        for f, cs in zip(FIELDS, pcols):
            P = d[cs].to_numpy(float)
            P = P / P.sum(axis=1, keepdims=True)
            d[f] = interpolated_median(prune_once(P))
        return d

    rt = readout(with_node_id(rt))
    de = with_node_id(de)
    ca = with_node_id(ca)

    def per_node(d, cols):
        if d is None:
            return {}
        have = [c for c in cols if c in d.columns]
        if not have or "node_id" not in d.columns:
            return {}
        g = d.groupby("node_id")[have]
        num = [c for c in have if pd.api.types.is_numeric_dtype(d[c])]
        out = {}
        for nid, sub in g:
            rec = {}
            for c in have:
                v = sub[c]
                rec[c] = (float(v.mean()) if c in num
                          else str(v.dropna().iloc[0]) if v.notna().any() else None)
            out[nid] = rec
        return out

    R = per_node(rt, FIELDS)
    D = per_node(de, QCOLS)
    C = per_node(ca, list(DIM) + ["M", "M_noA", "Omega"])

    out = args.out or (RES / "figures" / f"walk_{args.street}.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Embedded by default. A page that references ../../data/raw only opens
    # from results/figures, and the first thing anyone does with a file is move
    # it -- which silently produces a page of broken images rather than an
    # error. Inlining costs size and makes it work anywhere.
    if args.embed:
        import base64, io
        from PIL import Image
        from mast import erase_mast
    def mast_for(side):
        """The calibration that matches this frame's geometry.

        A STREET CAN MIX THE TWO. The export splits by street type, and a
        corridor that changes type partway -- Cannon Street has a 180 degree
        strip at n00290 and 90 degree halves either side of it -- ends up with
        both in one folder. Erasing every frame with one set left the strip's
        two masts standing, and the mast is the one thing in the frame the
        model provably never saw. 'auto' reads the geometry off the filename;
        an explicit --mast-set still overrides for the whole street.
        """
        if args.mast_set is None:
            return None
        if str(args.mast_set).lower() != "auto":
            return args.mast_set
        return "svi_180" if side == "F" else "svi_90"

    def one_step(r, files, street_folder, wk_tag=""):
        """One node: the frame, composed and erased, plus its street."""
        rec = _render(r, files, street_folder, wk_tag)
        rec["sf"] = street_folder
        return rec

    def _fill(arr, mask):
        """Interpolate across masked pixels, row by row, from their edges.

        np.interp per row rather than a Python loop over the run: the loop
        version was correct and took minutes per street, because a 2,880-wide
        frame with a 200px mask is 366,000 iterations of interpreted
        arithmetic. Only rows that actually carry mask are touched.
        """
        rows = np.nonzero(mask.any(axis=1))[0]
        if not len(rows):
            return arr
        W = mask.shape[1]
        xs = np.arange(W)
        for y in rows:
            m = mask[y]
            good = ~m
            if not good.any():
                continue
            gx = xs[good]
            for c in range(arr.shape[2]):
                arr[y, m, c] = np.interp(xs[m], gx,
                                         arr[y, good, c].astype(np.float32)
                                         ).astype(arr.dtype)
        return arr

    def clean(im, mast_set):
        """Erase the mast and the attribution bars, filling both by
        interpolation rather than with a flat patch.

        TWO ARTEFACTS, one fix. erase_mast finds the mast but fills it with a
        flat grey block that reads as a patch on the road; and the 180 strips
        carry two near-black columns with the Street View attribution burnt
        in, of which the calibration reliably catches only one -- 22 per cent
        of frames reached the page still showing the other. Taking the mask
        rather than the eraser's own fill, adding the bars to it, and
        interpolating across from the edges handles both and leaves nothing to
        see on a smooth road surface.

        DISPLAY ONLY, and measured rather than assumed. The bars were present
        when the frames were rated, so a cleaned frame differs from what the
        model saw. On 300 frames the ratings of frames with and without a
        residual bar differ by at most 0.12 of a rung and none approaches
        significance (walkable_ground -0.12, p=0.088; everything else p>0.35),
        so the bar does not move the numbers and the page can be clean without
        the panel beside it becoming a lie. Repairing erase_mast itself would
        be the real fix and would mean re-rating every frame.
        """
        arr = np.asarray(im.convert("RGB")).copy()
        mask = np.zeros(arr.shape[:2], dtype=bool)
        if mast_set:
            _, m = erase_mast(im, mast_set)
            m = np.asarray(m)
            if m.shape == mask.shape:
                mask |= m.astype(bool)
        g = np.asarray(im.convert("L"), dtype=np.float32)
        H, W = g.shape
        y0 = int(H * 0.78)
        dark = (g[y0:, :] < 45).mean(axis=0) > 0.45
        if dark.any():
            mask[y0:, dark] = True
        if not mask.any():
            return im
        return Image.fromarray(_fill(arr, mask))

    def prepare(path, mast_set):
        """One half or strip: mast erased, because that is what was rated."""
        im = Image.open(path).convert("RGB")
        # The model never saw the camera mast; the ratings beside each frame
        # were made on an erased image, so a page showing the mast shows
        # something other than what was rated.
        return clean(im, mast_set)

    def _render(r, files, street_folder="", wk_tag=""):
        sides = files[r.node_id]
        if args.embed:
            if "L" in sides and "R" in sides:
                # Both halves, left of travel then right, so the frame is the
                # forward 180 degrees and the two frontages pass together.
                half = (args.embed_width - SEAM) // 2
                parts = []
                for k in ("L", "R"):
                    im = prepare(sides[k], mast_for(k))
                    parts.append(im.resize(
                        (half, round(half * im.height / im.width)),
                        Image.LANCZOS))
                im = Image.new("RGB", (args.embed_width,
                                       max(q.height for q in parts)),
                               (18, 18, 20))
                for j, q in enumerate(parts):
                    im.paste(q, (j * (half + SEAM), 0))
            else:
                side, path = next(iter(sides.items()))
                im = prepare(path, mast_for(side))
                wdt = min(args.embed_width, im.width)
                im = im.resize((wdt, round(wdt * im.height / im.width)),
                               Image.LANCZOS)
            if args.frames_dir:
                args.frames_dir.mkdir(parents=True, exist_ok=True)
                fp = (args.frames_dir /
                      f"{r.node_id}_{street_folder}_{wk_tag}.jpg")
                im.save(fp, "JPEG", quality=args.quality, optimize=True)
                rel = (os.path.relpath(fp, out.parent).replace("\\", "/"))
            else:
                buf = io.BytesIO()
                im.save(buf, "JPEG", quality=args.quality, optimize=True)
                rel = ("data:image/jpeg;base64,"
                       + base64.b64encode(buf.getvalue()).decode())
        else:
            rel = os.path.relpath(next(iter(sides.values())),
                                  out.parent).replace("\\", "/")
        return {"node": r.node_id, "img": rel,
                "street": str(getattr(r, "osm_name", None)
                              or getattr(r, "street_name", "")),
                "r": R.get(r.node_id, {}), "d": D.get(r.node_id, {}),
                "c": C.get(r.node_id, {})}

    SEAM = 3
    steps = []
    # EVERY STREET IN ONE FILE. The frames carry which street they belong to,
    # so the selector switches street without a page load and without the
    # walk-through becoming a folder of files that have to travel together.
    for name in names:
        got = frames_for(name)
        if not got:
            print(f"  {name:<26}skipped: no frames")
            continue
        for wname, nf, files in got:
            # A FOLDER IS NOT ALWAYS ONE STREET. Park Avenue's tunnel segment
            # is five chains scattered over 970 by 690 m -- disjoint runs the
            # frame happens to label with one name. Walked as a single street
            # the stride is nonsense (57 nodes collapsed to 4 positions), and
            # the selector claims four unrelated places are the same street.
            # Chains of a folder therefore become their own selector entries
            # whenever the folder holds more than one; a chain too short to
            # walk rides along as auxiliary rather than becoming an entry of
            # its own.
            segs = list(nf.segname.unique())
            multi = len(segs) > 1
            for r in nf.itertuples():
                st = one_step(r, files, name, wname)
                st["wk"] = wname
                st["aux"] = int(getattr(r, "auxflag", 0))
                cs = getattr(r, "cstreet", None)
                if isinstance(cs, str) and cs:
                    st["sf"] = cs              # the cleaned street name
                else:
                    sg = str(getattr(r, "segname", name))
                    if multi:
                        if int((nf.segname == sg).sum()) < 4:
                            st["aux"] = 1      # stray, not a street
                        st["sf"] = sg
                steps.append(st)

    # THE SAME RAMP AND THE SAME CLIP AS THE MAPS. sim_vlm_maps normalises to
    # the 2nd-98th percentile of the study area's own distribution, so a colour
    # on the walk means what the identical colour means on the heat map. A bar
    # normalised to this street's own range instead would recolour the same
    # node differently on every street, which is the one thing a shared ramp
    # exists to prevent.
    from matplotlib.colors import Normalize, to_hex
    from common import sim_cmap
    import cmcrameri.cm as cmc
    import cmocean

    # THE THREE TERMS KEEP THEIR OWN RAMPS. sim_terms_maps draws imageability
    # in viridis, identity in lajolla and dependence in ice, each normalised to
    # the 2nd-98th percentile POOLED OVER BOTH CITIES so the two rows of that
    # figure are comparable. The panel reuses both, so a bar here is the same
    # colour as that node on the term map; a per-city clip would have made the
    # same value a different colour in each city.
    TERM_CMAP = {"I_raw": _mpl.colormaps["viridis"], "Y": cmc.lajolla,
                 "D_raw": cmocean.cm.ice}
    term_ramp = {}
    pooled = []
    for q in (RES.parent / "tables" / "vlm_calculations.csv",
              Path("results/tables/vlm_calculations.csv"),
              Path("results/london/tables/vlm_calculations.csv"),
              Path("results/london/tables/vlm_calculations_london.csv")):
        if q.exists():
            try:
                pooled.append(pd.read_csv(q, usecols=list(DIM)))
            except ValueError:
                pass
    tnorm = {}
    if pooled:
        allc = pd.concat(pooled, ignore_index=True)
        for k in DIM:
            lo_, hi_ = np.percentile(allc[k].dropna(), [2, 98])
            tnorm[k] = Normalize(float(lo_), float(hi_))
            # THE BAR IS A WINDOW ONTO THE RAMP, not a block of one colour.
            # Position x along the FULL bar carries the colour that value x
            # has on the term map, and the fill simply stops at the value --
            # so a full bar is the whole ramp, a bar at 40 per cent is its
            # left 40 per cent, and the colour at the tip is still exactly the
            # colour that node has on the map. Recolouring the whole bar each
            # step made the panel flicker through hues that meant nothing.
            # LEGIBILITY, NOT ACCURACY, and only in these three bars. Most
            # nodes sit low enough that the exact ramp renders them as near
            # black on a dark panel, where three different terms all read as
            # "dark" and the bar stops carrying information. Compressing into
            # the ramp's upper part keeps the ORDER intact -- higher is still
            # further up the ramp -- while giving the eye something to
            # separate. The cost is real and worth stating: the tip no longer
            # matches the term map exactly, so the maps stay the reference and
            # this is a reading aid. --ramp-floor 0 restores the true ramp.
            f0 = float(np.clip(args.ramp_floor, 0.0, 0.9))
            term_ramp[k] = [to_hex(TERM_CMAP[k](f0 + (1 - f0) * tnorm[k](x)))
                            for x in np.linspace(0, 1, 32)]
        print("  term ramps pooled over "
              f"{len(pooled)} table(s), {len(allc)} rows")
    # WHICH COMPOSITE, DECIDED BY WHETHER THE GEOMETRY TERM EXISTS. Murray Hill
    # has Omega from 2,924 measured H/W values, so its M is the full
    # Cobb-Douglas and showing M_noA there dropped a term the study computed.
    # The City of London has no facade heights: Omega is identically 1.0 and
    # HW_effective is empty, so M and M_noA are the same number and the panel
    # says so rather than implying an enclosure term it does not have.
    has_omega = (ca is not None and "Omega" in ca.columns
                 and ca.Omega.notna().any()
                 and float(ca.Omega.min()) < 0.999)
    mcol = "M" if (has_omega and "M" in ca.columns) else "M_noA"
    if ca is None or mcol not in ca.columns:
        mcol = "M"
    if ca is not None and mcol in ca.columns:
        lo, hi = ca[mcol].quantile([.02, .98])
    else:
        lo, hi = 0.0, 1.0
    # magma, because that is what sim_vlm_maps draws the composite in; the
    # hand-picked green ramp belongs to the dimension maps, not to M.
    ramp = _mpl.colormaps["magma"]
    norm = Normalize(float(lo), float(hi))
    print(f"  composite {mcol}  (Omega {'present' if has_omega else 'absent'})")
    # the legend: the ramp itself, sampled, plus ticks on round values inside
    # the clip -- so a colour on the walk can be read back as a number
    legend = [to_hex(ramp(x)) for x in np.linspace(0, 1, 24)]

    # WHERE THIS NODE SITS. A composite of 0.446 says nothing on its own; the
    # same number is unremarkable in one city and an outlier in another. The
    # curve is the study area's own distribution of M -- every node, not just
    # this street -- so the marker answers "compared with what". Percentile is
    # given as well, because reading a position off a curve is a guess and the
    # number is not.
    dist = {"xs": [], "ys": [], "lo": 0.0, "hi": 1.0}
    node_M = None
    if ca is not None and mcol in ca.columns:
        node_M = ca.groupby("node_id")[mcol].mean().dropna()
        if len(node_M) > 8:
            from scipy.stats import gaussian_kde
            x0, x1 = np.percentile(node_M, [0.5, 99.5])
            pad = (x1 - x0) * 0.04
            xs = np.linspace(x0 - pad, x1 + pad, 96)
            ys = gaussian_kde(node_M.to_numpy())(xs)
            ys = ys / ys.max()
            dist = {"xs": [round(float(v), 5) for v in xs],
                    "ys": [round(float(v), 4) for v in ys],
                    "lo": float(xs[0]), "hi": float(xs[-1])}
            print(f"  distribution over {len(node_M)} nodes of {mcol}")
    sorted_M = np.sort(node_M.to_numpy()) if node_M is not None else None

    # THE WHOLE STUDY AREA, not the street. A rating panel says what this view
    # is like; the map says where it is, which is the question a walk-through
    # otherwise leaves the viewer to hold in their head. Every node is drawn on
    # the same M ramp as the profile and the heat map, so the locator and the
    # figures in the paper are the same picture.
    MAP = {"nodes": [], "w": 1000, "h": 1000}
    pos = {}
    try:
        import geopandas as gpd
        from common import PROJ_CRS
        gdf = gpd.read_file(PROC / "nodes.gpkg").to_crs(PROJ_CRS)
        if UNUSABLE:
            gdf = gdf[~gdf.node_id.isin(UNUSABLE)].copy()
        mser = (ca.groupby("node_id")[mcol].mean()
                if ca is not None and mcol in ca.columns else None)
        xs_, ys_ = gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy()
        x0, x1 = float(xs_.min()), float(xs_.max())
        y0, y1 = float(ys_.min()), float(ys_.max())
        span = max(x1 - x0, y1 - y0) or 1.0
        pad = 24.0
        sc = (1000.0 - 2 * pad) / span
        for nid, X, Y in zip(gdf.node_id, xs_, ys_):
            # y is flipped: projected northing grows upward, SVG grows down
            mx = pad + (X - x0) * sc
            my = 1000.0 - pad - (Y - y0) * sc
            v = None if mser is None or nid not in mser.index else float(mser[nid])
            col = "#3a3f46" if v is None or not np.isfinite(v) else                 to_hex(ramp(norm(v)))
            MAP["nodes"].append([nid, round(mx, 1), round(my, 1), col])
            pos[nid] = (round(mx, 1), round(my, 1))
        # EVERY NODE CARRIES ITS SCORES, not only the ones whose frames are in
        # this file. Embedding a street's imagery costs megabytes and a study
        # area has hundreds of streets, so a file can only ever hold some of
        # them -- but the numbers are small. Without this the locator showed a
        # thousand nodes and let you click twenty, which reads as a broken map
        # rather than as a file that holds a subset of the imagery.
        NODEDATA = {}
        for nid in gdf.node_id:
            rec = {}
            if nid in R: rec["r"] = R[nid]
            if nid in C: rec["c"] = C[nid]
            if nid in D: rec["d"] = D[nid]
            if rec:
                v = (C.get(nid) or {}).get(mcol)
                rec["k"] = "#3a3f46" if v is None else to_hex(ramp(norm(float(v))))
                rec["kf"] = rec["k"]
                rec["kc"] = {k: to_hex(TERM_CMAP[k](tnorm[k](float(dv))))
                             for k, dv in (C.get(nid) or {}).items()
                             if k in tnorm and dv is not None}
                rec["pc"] = (None if sorted_M is None or v is None else
                             round(float(np.searchsorted(sorted_M, float(v),
                                                         side="right"))
                                   / len(sorted_M) * 100))
                NODEDATA[nid] = rec
        # A SCALE BAR IN METRES, computed from the projection rather than
        # guessed: sc is map-units per metre, so a round distance that fills
        # about a quarter of the frame is 1/2/5 x 10^k below that quarter.
        raw = (1000.0 * 0.28) / sc
        mag = 10 ** int(np.floor(np.log10(max(raw, 1.0))))
        nice = next((m * mag for m in (5, 2, 1) if m * mag <= raw), mag)
        MAP["bar"] = {"len": round(float(nice * sc), 1),
                      "label": (f"{int(nice)} m" if nice < 1000
                                else f"{nice/1000:g} km")}
        MAP["data"] = NODEDATA
        print(f"  locator map: {len(MAP['nodes'])} nodes, "
              f"{len(NODEDATA)} with scores")
    except Exception as e:
        print(f"  no locator map ({e.__class__.__name__}: {e})")

    step = 0.1 if (hi - lo) > 0.28 else 0.05
    ticks = [round(t, 2) for t in np.arange(np.ceil(lo / step) * step,
                                            hi + 1e-9, step)]
    ticks = [t for t in ticks if lo <= t <= hi]
    for st in steps:
        v = st["c"].get(mcol, st["c"].get("M"))
        st["k"] = "#3a3f46" if v is None else to_hex(ramp(norm(float(v))))
        # THE FACE IS THE COMPOSITE'S OWN COLOUR, exactly the one the profile
        # strip, the scale bar and the M map give this node -- no lift. The
        # term bars are lifted for legibility and the face is not, because the
        # face restates M and must not drift from the number beside it.
        st["kf"] = st["k"]
        st["pc"] = (None if (sorted_M is None or v is None) else
                    round(float(np.searchsorted(sorted_M, float(v),
                                                side="right"))
                          / len(sorted_M) * 100))
        st["mx"], st["my"] = pos.get(st["node"], (None, None))
        st["kc"] = {}
        for k in DIM:
            dv = st["c"].get(k)
            if dv is not None and k in tnorm:
                st["kc"][k] = to_hex(TERM_CMAP[k](tnorm[k](float(dv))))
    print(f"  ramp {mcol} clipped to {float(lo):.3f}-{float(hi):.3f}")

    # A SINGLE STRAY IS NOT A STREET. Park Ave Tunnel Segment #4 is one node;
    # as its own selector entry it offers a walk of length zero. Entries with
    # fewer than three walkable nodes become auxiliary -- still on the map,
    # still clickable, just not somewhere the selector offers to walk.
    from collections import Counter
    stride_n = Counter((st["sf"], st.get("wk")) for st in steps if not st.get("aux"))
    thin = {sf for (sf, _), c in stride_n.items() if c < 3}
    if thin:
        for st in steps:
            if st["sf"] in thin:
                st["aux"] = 1
        print(f"  {len(thin)} entr{'y' if len(thin)==1 else 'ies'} too short to "
              f"walk, kept on the map only: {', '.join(sorted(thin))}")

    have_q = sum(1 for s in steps if s["d"])
    print(f"  ratings on {sum(1 for s in steps if s['r'])} nodes, "
          f"descriptions on {have_q}, M on {sum(1 for s in steps if s['c'])}")

    # EACH FRAME IS ITS OWN ELEMENT, not one JSON blob. A viewer that
    # truncates a large file cuts the blob mid-array, the parse throws, and the
    # whole page goes blank -- image, scores and text at once, which reads as
    # the page being broken rather than the file being cut. As elements, a
    # truncated file simply ends early: every frame before the cut still works.
    frag = []
    for st in steps:
        meta = json.dumps({k: st[k] for k in ("node", "street", "r", "d", "c", "k", "kc", "pc", "sf", "kf", "mx", "my", "wk", "aux")})
        frag.append('<i class="f" data-meta="'
                    + html.escape(meta, quote=True)
                    + '" data-src="' + html.escape(st["img"], quote=True)
                    + '"></i>')
    page = TEMPLATE.replace("__FRAMES__", "\n".join(frag))
    page = page.replace("__TOTAL__", str(len(steps)))
    page = page.replace("__MCOL__", json.dumps(mcol))
    page = page.replace("__LEGEND__", json.dumps(legend))
    page = page.replace("__DIST__", json.dumps(dist))
    page = page.replace("__TERMRAMP__", json.dumps(term_ramp))
    page = page.replace("__MAP__", json.dumps(MAP))
    page = page.replace("__TICKS__", json.dumps(ticks))
    page = page.replace("__LO__", f"{float(lo):.6f}")
    page = page.replace("__HI__", f"{float(hi):.6f}")
    page = page.replace("__TITLE__", html.escape(names[0].replace("_", " ")))
    page = page.replace("__AREA__", html.escape(area))
    page = page.replace("__FIELDS__", json.dumps(FIELDS))
    page = page.replace("__GROUPS__", json.dumps(GROUPS))
    page = page.replace("__DIMS__", json.dumps(DIM))
    page = page.replace("__QCOLS__", json.dumps(QCOLS))
    out.write_text(page, encoding="utf-8")
    print(f"\nwrote {out}")
    print("  open it from the repository -- it points at the JPEGs on disk and")
    print("  must not be uploaded: Street View imagery is not redistributable.")


TEMPLATE = r"""<!doctype html>
<meta charset="utf-8">
<!-- Without this a phone lays the page out at 980px and then zooms out to
     fit, so every control is a third of its intended size and the ratings are
     unreadable. It is one line and it is the whole difference between the
     page working on a phone and not. -->
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>__TITLE__</title>
<style>
:root{--bg:#0e0f12;--fg:#e8e6e1;--mut:#9a9aa2;--line:#23262b;--acc:#5fbf6a}
*{box-sizing:border-box}
/* ONE SCREEN, whatever the screen is. The page is a fixed-height column and
   the frame takes whatever is left after the controls and the composite, so
   nothing below the fold and no scrollbar on a laptop. Below 900px it gives
   up and scrolls: a phone cannot hold a panorama, ten ratings and four
   paragraphs at once, and squeezing them until it does makes all three
   unreadable rather than one of them absent. */
html{height:100%}
body{margin:0;background:var(--bg);color:var(--fg);
     font:14px/1.5 "Segoe UI",system-ui,sans-serif;
     height:100dvh;overflow:hidden;display:flex;flex-direction:column}
header{display:flex;align-items:baseline;gap:14px;padding:12px 18px;
       border-bottom:1px solid var(--line)}
/* The title IS the selector: a street name that can be changed reads as a
   control, and a separate dropdown beside a heading says the same word twice. */
#pick{font:600 17px/1.2 "Segoe UI",system-ui,sans-serif;color:var(--fg);
      background:transparent;border:1px solid transparent;border-radius:4px;
      padding:3px 6px;margin-left:-6px;cursor:pointer;max-width:44vw}
#pick:hover{border-color:var(--line);background:#171a1f}
#pick:focus-visible{outline:2px solid var(--acc)}
#pick option{background:#171a1f;color:var(--fg);font-size:14px}
header .a{color:var(--mut);font-size:13px}
main{display:grid;grid-template-columns:1fr 380px;gap:18px;padding:14px 18px;
     align-items:stretch;flex:1;min-height:0}
#left{display:flex;flex-direction:column;min-height:0}
/* The sidebar is a column: tabs and the shared box are fixed, and the
   description takes what is left and scrolls inside itself if a node's answers
   are unusually long. Letting the whole sidebar scroll instead moved the map
   and the ratings off the top of the screen on five nodes out of sixteen. */
aside{min-height:0;overflow:hidden;padding-right:4px;
      display:flex;flex-direction:column}
aside>#tabs,aside>#pane,aside>h2{flex:none}
#qual{flex:1;min-height:0;overflow-y:auto;margin:0}
/* stack on anything narrow -- a side-by-side grid at phone width squeezes the
   image into a sliver and the panel off the screen */
@media (max-width:900px){
  html,body{height:auto;overflow:auto}
  main{grid-template-columns:1fr;padding:12px;gap:12px;min-height:0}
  #view{flex:none;height:auto}
  aside{overflow:visible}
  header{flex-wrap:wrap;gap:6px 12px;padding:10px 12px}
  aside{border-top:1px solid var(--line);padding-top:8px}
}
/* A grid track will not shrink below its content's min-content width, and an
   img's min-content width is its INTRINSIC width -- so the 680 px frame held
   the left column open at 680 px and pushed the panel off a phone screen,
   width:100% notwithstanding. min-width:0 lets the track shrink. */
main>*{min-width:0}
#view{width:100%;max-width:100%;border:1px solid var(--line);border-radius:3px;
      display:block;background:#171a1f;flex:1;min-height:0;object-fit:contain}
#warn{display:none;padding:10px 12px;margin:8px 0;border:1px solid #7a3b3b;
      border-radius:3px;background:#241a1a;color:#e8b4b4;font-size:13px}
.bar{height:20px;background:#171a1f;border-radius:2px;position:relative;
     overflow:hidden}
.bar i{position:absolute;inset:0 auto 0 0;background:var(--acc);opacity:.75}
.row{display:grid;grid-template-columns:142px 1fr 30px;gap:8px;
     align-items:center;margin:3px 0}
.row span{color:var(--mut);font-size:12px;overflow-wrap:anywhere}
.row b{font-weight:600;text-align:right;font-variant-numeric:tabular-nums}
h2{font-size:11px;text-transform:uppercase;letter-spacing:.08em;
   color:var(--mut);margin:11px 0 5px;font-weight:600}
/* Two modes over one panel: the ratings say what this view is like, the map
   says where it is. They answer different questions about the same node and
   neither needs to be visible while the other is being read. */
#tabs{display:flex;gap:4px;margin:2px 0 7px}
#tabs button{font-size:10px;text-transform:uppercase;letter-spacing:.08em;
   color:var(--mut);background:transparent;border:1px solid transparent;
   padding:4px 8px;border-radius:3px;font-weight:600}
#tabs button:hover{color:var(--fg)}
#tabs button.on{color:var(--fg);border-color:var(--line);background:#171a1f}
/* BOTH TABS OCCUPY ONE BOX. The ratings stay in flow and set the height --
   hidden with visibility, not display, so the box does not collapse -- and the
   map is laid over them. Switching tabs then moves nothing below: the
   description stays exactly where the eye left it. */
/* THE MAP SETS THE HEIGHT, not the ratings. The locator is square and holds
   hundreds of nodes, so sizing the box to the ten rating rows made it a
   thumbnail and clicking a node a game of darts; the ratings are happy with
   space below them and the map is not happy without it. The box is square to
   the sidebar's width, capped against the viewport so the description under it
   still fits on a short screen without scrolling. */
#pane{position:relative;min-height:180px}
#fields{height:100%;overflow-y:auto}
.grp{display:flex;align-items:center;gap:6px;font-size:10px;font-weight:600;
     text-transform:uppercase;letter-spacing:.07em;color:var(--mut);
     margin:9px 0 3px}
.grp:first-child{margin-top:0}
.gdot{width:8px;height:8px;border-radius:2px;flex:none}
#fields.off{visibility:hidden}
#mapwrap{position:absolute;inset:0;display:flex;flex-direction:column;
         align-items:center;justify-content:flex-start;gap:4px}
#mapwrap.off{visibility:hidden;pointer-events:none}
#map{flex:1;min-height:0;width:100%;height:100%;display:block}
#maphint{flex:none;text-align:center}
#map circle.n{cursor:pointer}
#map circle.n:hover{stroke:var(--fg);stroke-width:14}
#maphint{margin-top:6px}
.q{margin:0 0 12px}
.q dt{color:var(--mut);font-size:9.5px;text-transform:uppercase;
      letter-spacing:.07em;margin-bottom:1px}
.q dd{margin:0 0 6px;font-size:11.5px;line-height:1.36}
/* two rows, three columns: the legend sits in the middle column so its scale
   lines up with the profile it explains rather than with the page. */
#ctl{display:grid;grid-template-columns:auto 1fr auto;gap:6px 10px;
     align-items:center;margin-top:10px}
#btns{display:flex;gap:8px}
#dir{white-space:nowrap}
/* A SCALE BAR, not a full-width colourbar. It does the job the 200 m ruler
   does on the maps: small, in the corner, there when you look for it and out
   of the way when you are not. Spanning the profile made the legend louder
   than the data it explains. */
#legend{display:flex;align-items:center;gap:7px;padding-top:5px;
        font-size:10px;color:var(--mut)}
#lgrad{width:128px;height:6px;border-radius:1px;border:1px solid var(--line);
       flex:none}
#legend b{font-weight:600;letter-spacing:.06em;text-transform:uppercase}
#legend .lv{font-variant-numeric:tabular-nums}
/* The profile carries the street's own M on the map's ramp: the walk and the
   heat map become the same picture read two ways. */
/* THE PROFILE IS THE CONTROL. A native range under it drew the same position
   twice in two different visual languages; the strip already shows where you
   are, so it takes the clicks and the second bar goes. */
#sl{flex:1;position:relative;height:18px;cursor:pointer;outline:none}
#strip{position:absolute;top:3px;left:0;right:0;height:12px;border-radius:2px;
       border:1px solid var(--line)}
#sl:focus-visible{box-shadow:0 0 0 2px var(--acc);border-radius:3px}
#mark{position:absolute;top:0;width:3px;height:18px;background:var(--fg);
      border-radius:1px;box-shadow:0 0 0 1px rgba(0,0,0,.7);
      transform:translateX(-1.5px);pointer-events:none}
button{background:#171a1f;color:var(--fg);border:1px solid var(--line);
       border-radius:3px;padding:6px 12px;font-size:13px;cursor:pointer}
button:hover{border-color:#3a3f46}
/* NEVER WRAPS, AND NEVER CHANGES WIDTH. "9 / 16" fits on one line and
   "10 / 16" did not, so the counter wrapped, the control row grew a line, and
   every node with a two-digit index pushed the page down a step. */
#pos{color:var(--mut);font-variant-numeric:tabular-nums;white-space:nowrap;
     min-width:7ch;text-align:right}
input[type=range]{display:block}
.M{font-size:26px;font-weight:600;font-variant-numeric:tabular-nums}
/* The composite sits under the view, not in the sidebar: three numbers and a
   distribution take a strip of width far better than a column, and the space
   they were using is what the descriptions needed. */
#foot{display:grid;grid-template-columns:auto minmax(230px,1fr) 260px;
      gap:20px;align-items:start;margin-top:10px;padding-top:10px;
      border-top:1px solid var(--line);flex:none}
#mbox{min-width:150px}
/* The composite as a face: the same number the panel already gives, in the
   one encoding nobody has to be taught to read. It is a restatement, not a
   new measurement -- the curve of the mouth is M and nothing else. */
/* The face sits beside the number, not in a column of its own: it restates M,
   so it belongs to M rather than standing as a third thing to read. */
#mrow{display:flex;align-items:center;gap:14px}
#face{display:flex;align-items:center}
#smiley{width:52px;height:52px;display:block}
#foot h2{margin:0 0 4px}
#dims{padding-top:2px}
#dist svg{width:100%;height:46px;display:block;margin-top:3px}
#dist .note{min-height:15px}
@media (max-width:900px){#foot{grid-template-columns:1fr;gap:12px}}
.none{color:#6b7078;font-style:italic}
.note{color:var(--mut);font-size:11px;line-height:1.45;margin:-2px 0 4px}

/* ---- PHONE ---------------------------------------------------------------
   LAST IN THE FILE, and that is the whole point. These rules first sat above
   the base layout, and a media query adds no specificity, so every later #map
   and #pane rule quietly overrode them: the map kept height:100% from the
   desktop rule and rendered 480x150 inside a 180px box, which is why it stayed
   small however large it was told to be. Overrides go last.

   The page scrolls vertically on a phone and that is the right answer -- a
   panorama, ten ratings and four paragraphs cannot share a 390px screen at a
   readable size. What it must NOT do is scroll sideways: horizontal panning
   fights the draggable profile and makes the whole page feel loose. */
@media (max-width:900px){
  html,body{height:auto;overflow-y:auto;overflow-x:hidden;
            max-width:100%;overscroll-behavior-x:none}
  body{touch-action:pan-y}
  main{grid-template-columns:1fr;padding:10px;gap:14px;min-height:0;
       max-width:100%}
  #left,aside,#foot,#pane,#fields,#mapwrap,#dims,#dist{min-width:0;
       max-width:100%}
  #left{display:block}
  #view{flex:none;height:auto;width:100%}
  aside{overflow:visible;display:block;padding-right:0;
        border-top:1px solid var(--line);padding-top:8px}
  #qual{overflow:visible;max-height:none;flex:none}
  #pane{height:auto !important;min-height:0;aspect-ratio:auto}
  #fields{height:auto;overflow:visible}
  #fields.off{display:none;visibility:visible}
  #mapwrap{display:block;position:static;visibility:visible}
  #mapwrap.off{display:none}
  #map{width:100%;height:auto !important;aspect-ratio:1/1;
       max-height:none !important;flex:none}
  #maphint{margin-top:8px;text-align:left}
  #foot{grid-template-columns:1fr;gap:14px}
  #dist svg{height:64px}
  header{flex-wrap:wrap;gap:4px 12px;padding:10px 12px}
  #pick{max-width:100%}
  button{padding:10px 14px;font-size:14px}
  #ctl{gap:6px 8px}
  #sl{height:26px}
  #strip{top:6px;height:14px}
  #mark{height:26px;width:4px}
  .q dd{font-size:13.5px;line-height:1.45}
  .row{grid-template-columns:minmax(96px,42%) 1fr 26px;gap:6px}
  .row span{overflow-wrap:anywhere}
  #legend{flex-wrap:wrap}
}
</style>
<header>
  <select id="pick" aria-label="street"></select><span class="a">__AREA__</span>
  <span class="a" id="node"></span>
</header>
<main>
  <div id="left">
    <div id="warn">The frames did not load. This build references the JPEGs on
      disk, so it only works from inside the repository -- rebuild with
      --embed for a file that opens anywhere.</div>
    <img id="view" alt="street view">
    <div id="ctl">
      <div id="btns">
        <button id="prev">&larr;</button>
        <button id="play">play</button>
        <button id="next">&rarr;</button>
        <button id="dir" hidden title="walk the street the other way"></button>
      </div>
      <div id="sl" tabindex="0" role="slider" aria-label="position along the walk">
        <div id="strip"></div>
        <i id="mark"></i>
      </div>
      <span id="pos"></span>
      <div></div>
      <div id="legend"><b id="lname"></b><span class="lv" id="llo"></span>
        <div id="lgrad"></div><span class="lv" id="lhi"></span></div>
      <div></div>
    </div>
    <div id="foot">
      <div id="mbox">
        <h2 id="Mh">composite</h2>
        <div id="mrow">
          <div class="M" id="M">--</div>
          <div id="face">
        <svg id="smiley" viewBox="0 0 64 64" aria-label="composite as a face">
          <circle cx="32" cy="32" r="27" fill="none" stroke-width="4"/>
          <circle id="eyeL" cx="23" cy="25" r="3.4" stroke="none"/>
          <circle id="eyeR" cx="41" cy="25" r="3.4" stroke="none"/>
          <path id="mouth" fill="none" stroke-width="4" stroke-linecap="round"/>
          </svg>
          </div>
        </div>
        <div class="note" id="Mnote"></div>
      </div>
      <div id="dims"></div>
      <div id="dist">
        <h2>where this node sits</h2>
        <div class="note" id="pct"></div>
        <svg id="curve" viewBox="0 0 260 62" preserveAspectRatio="none"></svg>
      </div>
    </div>
  </div>
  <aside>
    <div id="tabs">
      <button id="tabR" class="on">ratings, 1 to 7</button>
      <button id="tabM">where</button>
    </div>
    <div id="pane">
      <div id="fields"></div>
      <div id="mapwrap" class="off">
        <svg id="map" viewBox="0 0 1000 1000"></svg>
        <div class="note" id="maphint">every node in the study area, on the M
          ramp. click one to jump there.</div>
      </div>
    </div>
    <h2>what the model says</h2>
    <dl class="q" id="qual"></dl>
  </aside>
</main>
<script>
const FIELDS=__FIELDS__, GROUPS=__GROUPS__, DIMS=__DIMS__, QCOLS=__QCOLS__, MCOL=__MCOL__;
const LEGEND=__LEGEND__, TICKS=__TICKS__, LO=__LO__, HI=__HI__;
const DIST=__DIST__, TERMRAMP=__TERMRAMP__, MAP=__MAP__;
// THE FRAMES LIVE AFTER THIS SCRIPT, and are read once the document has
// finished parsing. Order matters: interface and logic first, payload last, so
// a viewer that truncates a large file cuts only frames off the end. Whatever
// arrived still works; nothing above the cut depends on what is below it.
let STEPS=[], ALL=[], CUR="", CURWK="";
let i=0, timer=null;
const $=id=>document.getElementById(id);
const TOTAL=__TOTAL__;

$("view").onerror=()=>{ $("warn").style.display="block"; };
function draw(){ paint(STEPS[i], false); }

function paint(s, scoresOnly){
  if(scoresOnly){
    $("view").removeAttribute("src");
    $("view").alt="no view in this file for this node";
  } else {
    $("view").src=s.img;
    $("view").alt="street view";
  }
  $("node").textContent=s.node+(s.street?"  ·  "+s.street:"")
    +(scoresOnly?"   scores only -- this view is not in this file":"");
  $("pos").textContent=scoresOnly?"--":((i+1)+" / "+STEPS.length);
  if(!scoresOnly && i >= 0)
    $("mark").style.left=(STEPS.length<2?0:i/(STEPS.length-1)*100)+"%";

  // NAME THE NUMBER. Neither city's vlm_calculations carries A_i, so what is
  // shown is M without the geometry term -- Cobb-Douglas over I, Y and D
  // alone. Labelling it "composite" let a number that excludes enclosure read
  // as M, which is the one misreading a slide cannot afford.
  // MCOL is chosen at build time: M where the study area has a geometry term,
  // M_noA where it has none. The caption names the formula actually used.
  const m = s.c[MCOL];
  const full = MCOL === "M";
  $("Mh").textContent = full ? "composite M" : "composite M, no geometry term";
  // the mouth IS the composite: corners fixed, the middle driven by where M
  // falls between the ramp's ends, so the face and the colour say one thing
  const t = (m==null||HI<=LO) ? 0.5
            : Math.max(0, Math.min(1, (m - LO) / (HI - LO)));
  $("mouth").setAttribute("d",
    "M20,40 Q32," + (40 + (t - 0.5) * 26).toFixed(1) + " 44,40");
  ["smiley"].forEach(id=>$(id).setAttribute("stroke", s.kf));
  $("eyeL").setAttribute("fill", s.kf);
  $("eyeR").setAttribute("fill", s.kf);
  $("smiley").querySelector("circle").setAttribute("stroke", s.kf);
  $("mouth").setAttribute("stroke", s.kf);
  markCurve(m, s.pc);
  if(scoresOnly){
    const g=document.getElementById("here");
    if(g && s.mx!=null){ g.setAttribute("opacity",".55");
      g.setAttribute("transform",`translate(${s.mx},${s.my}) rotate(0)`); }
  } else markMap();
  $("Mnote").textContent = full
    ? "I⁰·⁴ · Y⁰·² · D⁰·⁴ · Ω"
    : "I⁰·⁴ · Y⁰·² · D⁰·⁴ — Ω = 1: no facade heights for this study area, so no H/W";
  $("M").textContent = m==null ? "--" : m.toFixed(3);
  $("dims").innerHTML = Object.entries(DIMS).map(([k,label])=>{
    const v=s.c[k];
    // each term wears its own ramp -- viridis, lajolla, ice -- so the bar is
    // the colour this node has on the three-term map
    if(v==null) return "";
    const pct=Math.max(0.6,Math.min(100,v*100));
    const stops=TERMRAMP[k];
    // the gradient is sized to the WHOLE bar, then cropped by the fill's own
    // width, so the visible run is the ramp up to this value
    const fill = stops
      ? `background-image:linear-gradient(to right,${stops.join(",")});
         background-size:${(100/pct*100).toFixed(1)}% 100%;
         background-repeat:no-repeat;opacity:1`
      : `background:${s.kc[k]||"var(--acc)"};opacity:1`;
    return `<div class="row"><span>${label}</span>
        <div class="bar"><i style="width:${pct}%;${fill}"></i></div>
        <b>${v.toFixed(2)}</b></div>`;
  }).join("");

  $("fields").innerHTML = GROUPS.map(([label, term, members])=>{
    const rows = members.map(([f, inv])=>{
      const v=s.r[f];
      if(v==null) return "";
      // SHOWN AS A RUNG, NOT AS A MEASUREMENT. The seven rungs exist so a
      // rating reads the way somebody on the pavement would say it; printing
      // 4.9 turns a judgement back into an instrument reading. The
      // interpolated median still drives M -- the display rounds, the
      // calculation does not.
      const r=Math.max(1,Math.min(7,Math.round(v)));
      return `<div class="row"><span
        title="${inv?"enters its term inverted: a higher rung lowers it":""}"
        >${f.replace(/_/g," ")}</span>
        <div class="bar"><i style="width:${(r-1)/6*100}%"></i></div>
        <b>${r}</b></div>`;
    }).join("");
    if(!rows) return "";
    return `<div class="grp"><span class="gdot"
      style="background:${s.kc[term]||"var(--acc)"}"></span>${label}</div>${rows}`;
  }).join("") || '<div class="none">no ratings for this node</div>';

  const q=QCOLS.filter(c=>s.d[c]).map(c=>
    `<dt>${c}</dt><dd>${s.d[c]}</dd>`).join("");
  $("qual").innerHTML = q || '<div class="none">not described yet</div>';
  if(PANE_H) $("pane").style.height = PANE_H + "px";
}
function go(n){ i=(n+STEPS.length)%STEPS.length; draw(); }
$("prev").onclick=()=>go(i-1);
$("next").onclick=()=>go(i+1);
// THE LOCATOR. Every node in the study area is drawn once; only the arrow
// moves. Nodes that appear in this file are clickable and jump there --
// switching street first if the node belongs to another one -- and nodes that
// are not in the file are drawn anyway, because a map showing only the streets
// that happen to be loaded misrepresents the study area.
let NEAR = 0;
function drawMap(){
  if(!MAP.nodes || !MAP.nodes.length){ $("tabM").style.display="none"; return; }
  // "near enough to be the same place": a couple of steps of the walk, taken
  // from the actual spacing between consecutive frames rather than guessed, so
  // it holds for either city and any --every.
  const gaps = [];
  for(let j = 1; j < ALL.length; j++){
    const a = ALL[j-1], b = ALL[j];
    if(a.sf === b.sf && a.mx != null && b.mx != null)
      gaps.push(Math.hypot(b.mx - a.mx, b.my - a.my));
  }
  gaps.sort((x,y)=>x-y);
  NEAR = gaps.length ? gaps[Math.floor(gaps.length/2)] * 1.6 : 40;
  const inFile = new Set(ALL.map(s=>s.node));
  // every node with scores is clickable; the ones whose frames are in this
  // file are drawn larger, because those are the ones that also show a view
  const dots = MAP.nodes.map(([id,x,y,c])=>{
    const has = inFile.has(id), scored = MAP.data && MAP.data[id];
    return `<circle class="${has||scored?"n":""}" data-id="${id}" cx="${x}"
      cy="${y}" r="${has?7:5}" fill="${c}" opacity="${has?1:.5}"
      ><title>${id}${has?"":" (scores only)"}</title></circle>`;
  }).join("");
  // North is straight up: the projection maps northing to screen-y inverted,
  // so the compass is a fact about the drawing rather than a decoration. The
  // ramp legend repeats the scale bar because the map can be read on its own.
  // North is straight up: the projection maps northing to screen-y inverted,
  // so the compass is a fact about the drawing, not a decoration. The scale is
  // in metres -- the colour scale already sits under the profile, and a map
  // wants to say how far, not how green.
  const b = MAP.bar || {len: 200, label: ""};
  const x0 = 28, yb = 962;
  const chrome =
    `<g id="compass" style="pointer-events:none" opacity=".85">
       <path d="M952,26 L972,80 L952,66 L932,80 Z" fill="var(--fg)"/>
       <text x="952" y="116" fill="var(--fg)" font-size="40"
             text-anchor="middle" font-family="inherit">N</text>
     </g>
     <g id="mscale" style="pointer-events:none" font-family="inherit">
       <line x1="${x0}" y1="${yb}" x2="${x0 + b.len}" y2="${yb}"
             stroke="var(--fg)" stroke-width="5" opacity=".8"/>
       <line x1="${x0}" y1="${yb-11}" x2="${x0}" y2="${yb+11}"
             stroke="var(--fg)" stroke-width="5" opacity=".8"/>
       <line x1="${x0 + b.len}" y1="${yb-11}" x2="${x0 + b.len}" y2="${yb+11}"
             stroke="var(--fg)" stroke-width="5" opacity=".8"/>
       <text x="${x0 + b.len/2}" y="${yb-20}" fill="var(--mut)" font-size="34"
             text-anchor="middle">${b.label}</text>
     </g>`;
  $("map").innerHTML = chrome + dots +
    `<g id="here" style="pointer-events:none">
       <circle r="30" fill="#3aa0ff" opacity=".16"/>
       <circle r="30" fill="none" stroke="#0b1520" stroke-width="9"/>
       <circle r="30" fill="none" stroke="#3aa0ff" stroke-width="5"/>
       <path d="M0,-46 L20,17 L0,6 L-20,17 Z" fill="#0b1520"
             transform="scale(1.18)"/>
       <path d="M0,-46 L20,17 L0,6 L-20,17 Z" fill="#3aa0ff"/>
     </g>`;
  $("map").querySelectorAll("circle.n").forEach(el=>{
    el.addEventListener("click", ()=>jumpTo(el.dataset.id));
  });
}
function jumpTo(id){
  let k = STEPS.findIndex(s=>s.node===id);
  if(k >= 0){ go(k); return; }
  const other = ALL.find(s=>s.node===id);          // another street in the file
  if(other && other.aux){
    // an auxiliary-run node: real frame, real scores, just not on the
    // stride. Shown standalone; the counter shows a dot.
    if(other.sf !== CUR) pickStreet(other.sf, other.wk);
    i = -1;
    paint(other, false);
    $("pos").textContent = "·";
    const g = document.getElementById("here");
    if(g && other.mx != null){
      g.setAttribute("opacity", "1");
      g.setAttribute("transform",
        "translate(" + other.mx + "," + other.my + ") rotate(0)");
    }
    return;
  }
  if(other){ pickStreet(other.sf, other.wk);
             const kk = STEPS.findIndex(s=>s.node===id);
             if(kk >= 0){ go(kk); return; } }

  // NEAREST VIEW, not a dead end. The file holds every Nth node, so most of
  // the dots on a loaded street have no frame of their own -- and clicking one
  // used to drop into scores-only on a street that is right there in the file,
  // which reads as broken. Jump to the closest node that does have a view, if
  // one is near enough to be the same place; only somewhere genuinely not in
  // the file falls back to scores.
  const at = MAP.nodes.find(n=>n[0]===id);
  if(at){
    let best = null, bd = Infinity;
    ALL.forEach(st=>{
      if(st.mx==null) return;
      const d = Math.hypot(st.mx - at[1], st.my - at[2]);
      if(d < bd){ bd = d; best = st; }
    });
    if(best && bd <= NEAR){
      if(best.sf !== CUR || best.wk !== CURWK) pickStreet(best.sf, best.wk);
      go(STEPS.indexOf(best));
      return;
    }
  }
  showScoresOnly(id);
}

// A node whose imagery is not in this file still has numbers, and they are
// worth showing: the panel fills, the view says why it is blank, and stepping
// returns to the walk.
function showScoresOnly(id){
  const rec = MAP.data && MAP.data[id];
  if(!rec) return;
  const st = Object.assign({node:id, street:"", img:null, r:{}, d:{}, c:{},
                            kc:{}, k:"#3a3f46", kf:"#6b7078"}, rec);
  st.mx = (MAP.nodes.find(n=>n[0]===id)||[])[1];
  st.my = (MAP.nodes.find(n=>n[0]===id)||[])[2];
  paint(st, true);
}
// The arrow points the way the walk is going, taken from the next node along
// -- at the last node from the previous one, so the heading never flips at the
// end of a street.
function markMap(){
  const g = document.getElementById("here");
  if(!g) return;
  const s = STEPS[i];
  if(!s || s.mx==null){ g.setAttribute("opacity","0"); return; }
  const nb = STEPS[i+1] || STEPS[i-1];
  let ang = 0;
  if(nb && nb.mx!=null){
    const dx = (STEPS[i+1] ? nb.mx - s.mx : s.mx - nb.mx);
    const dy = (STEPS[i+1] ? nb.my - s.my : s.my - nb.my);
    ang = Math.atan2(dx, -dy) * 180 / Math.PI;
  }
  g.setAttribute("opacity","1");
  g.setAttribute("transform", `translate(${s.mx},${s.my}) rotate(${ang.toFixed(1)})`);
}

// The distribution is static -- it is the study area, not the street -- so it
// is drawn once and only the marker moves.
const W=260, H=62, PAD=6;
function dx(v){ return (v-DIST.lo)/(DIST.hi-DIST.lo)*W; }
function drawCurve(){
  if(!DIST.xs.length) { $("dist").style.display="none"; return; }
  const pts=DIST.xs.map((x,j)=>[dx(x), H-PAD-DIST.ys[j]*(H-PAD*2)]);
  const path="M0,"+H+" L"+pts.map(p=>p[0].toFixed(1)+","+p[1].toFixed(1))
             .join(" L")+" L"+W+","+H+" Z";
  const stops=LEGEND.map((c,j)=>
    `<stop offset="${(j/(LEGEND.length-1)*100).toFixed(1)}%" stop-color="${c}"/>`
  ).join("");
  $("curve").innerHTML=
    `<defs><linearGradient id="g" x1="0" x2="1">${stops}</linearGradient></defs>`
    + `<path d="${path}" fill="url(#g)" opacity=".55"/>`
    + `<path d="${"M"+pts.map(p=>p[0].toFixed(1)+","+p[1].toFixed(1)).join(" L")}"
         fill="none" stroke="var(--fg)" stroke-width="1.1" opacity=".55"/>`
    + `<line id="nowline" y1="0" y2="${H}" stroke="var(--fg)" stroke-width="2"/>`;
}
function markCurve(v, pc){
  const ln=document.getElementById("nowline");
  if(!ln) return;
  if(v==null){ ln.setAttribute("opacity","0"); $("pct").textContent=""; return; }
  const x=Math.max(0,Math.min(W,dx(v)));
  ln.setAttribute("opacity","1");
  ln.setAttribute("x1",x); ln.setAttribute("x2",x);
  $("pct").textContent = pc==null ? ""
    : pc+(pc%10===1&&pc%100!==11?"st":pc%10===2&&pc%100!==12?"nd":
          pc%10===3&&pc%100!==13?"rd":"th")
      +" percentile of every node in this study area";
}

// scrub on the profile: click or drag anywhere along it
function seek(ev){
  const r=$("sl").getBoundingClientRect();
  if(r.width<=0||STEPS.length<2) return;
  const f=Math.min(1,Math.max(0,(ev.clientX-r.left)/r.width));
  go(Math.round(f*(STEPS.length-1)));
}
$("sl").addEventListener("pointerdown",e=>{
  $("sl").setPointerCapture(e.pointerId); seek(e);
});
$("sl").addEventListener("pointermove",e=>{ if(e.buttons&1) seek(e); });
$("dir").onclick=()=>{
  const wks=walksOf(CUR);
  if(wks.length<2) return;
  const other=wks[(wks.indexOf(CURWK)+1)%wks.length];
  const was=STEPS[i];
  pickStreet(CUR, other);
  // same node in the other direction -- and when --every sampled the two
  // walks onto different node subsets, the nearest position instead of the
  // start of the street
  let k=STEPS.findIndex(s=>s.node===(was&&was.node));
  if(k<0 && was && was.mx!=null){
    let bd=Infinity;
    STEPS.forEach((s,j)=>{ if(s.mx!=null){
      const d=Math.hypot(s.mx-was.mx, s.my-was.my);
      if(d<bd){ bd=d; k=j; } }});
  }
  if(k>=0) go(k);
};
$("play").onclick=function(){
  if(timer){clearInterval(timer);timer=null;this.textContent="play";}
  else{timer=setInterval(()=>go(i+1),900);this.textContent="pause";}
};
addEventListener("keydown",e=>{
  if(e.key==="ArrowLeft")go(i-1);
  if(e.key==="ArrowRight")go(i+1);
});
function start(){
  ALL=[...document.querySelectorAll("#frames .f")].map(el=>{
    let m={}; try{ m=JSON.parse(el.dataset.meta||"{}"); }catch(e){}
    return {img:el.dataset.src, node:m.node||"", street:m.street||"",
            r:m.r||{}, d:m.d||{}, c:m.c||{}, k:m.k||"#3a3f46", kc:m.kc||{}, pc:m.pc, sf:m.sf||"", kf:m.kf||"#6b7078",
          mx:m.mx, my:m.my, wk:m.wk||"", aux:m.aux||0};
  });
  // one <option> per street, in the order the build wrote them
  const seen=[];
  ALL.forEach(s=>{ if(!seen.includes(s.sf)) seen.push(s.sf); });
  $("pick").innerHTML=seen.map(n=>
    `<option value="${n}">${n.replace(/_/g," ")}</option>`).join("");
  $("pick").onchange=e=>pickStreet(e.target.value);
  if(seen.length<2) $("pick").style.pointerEvents="none";
  CUR=seen[0]||"";
  drawCurve();
  drawMap();
  $("tabR").onclick=()=>setTab(false);
  $("tabM").onclick=()=>setTab(true);

  $("lgrad").style.background="linear-gradient(to right,"+LEGEND.map(
    (c,j)=>c+" "+(j/(LEGEND.length-1)*100).toFixed(2)+"%").join(",")+")";
  $("lname").textContent = MCOL === "M" ? "M" : "M (no Ω)";
  $("llo").textContent = LO.toFixed(2);
  $("lhi").textContent = HI.toFixed(2);
  if(!ALL.length){
    $("warn").style.display="block";
    $("warn").textContent="No frames in this file -- it was cut short in transit.";
    return;
  }
  if(ALL.length<TOTAL){
    $("warn").style.display="block";
    $("warn").textContent="Showing "+ALL.length+" of "+TOTAL+
      " frames; the file was cut short in transit.";
  }
  pickStreet(CUR);
  measurePane();
}

// Switching street rebuilds the walk and its profile, but not the
// distribution: that is the study area, and it does not change.
// AS BIG AS IT CAN BE WITHOUT PUSHING ANYTHING OFF. CSS can express "square"
// or "42vh", but not "whatever is left after the description", and the
// description's length changes with the node -- so a fixed cap either wasted
// space on a tall screen or scrolled the panel on a short one. Measure what
// the sidebar has spare and give the map that, never more than square.
let PANE_H = 0;

// ONE SIZE, SET ONCE. sizePane ran on every draw, and the description's length
// changes from node to node, so the map grew and shrank as you walked -- the
// thing you are trying to read position off was never the same size twice.
// Measure the WORST case instead: the longest description in the file. The map
// is then fixed, and no node can overflow the sidebar.
function measurePane(){
  const qual = $("qual");
  let worst = null, n = -1;
  ALL.forEach(st=>{
    const L = QCOLS.reduce((a,c)=>a + (((st.d||{})[c])||"").length, 0);
    if(L > n){ n = L; worst = st; }
  });
  const saved = qual.innerHTML;
  if(worst && n > 0){
    qual.innerHTML = QCOLS.filter(c=>worst.d[c])
      .map(c=>`<dt>${c}</dt><dd>${worst.d[c]}</dd>`).join("");
  }
  // The box must never be shorter than the ratings need, or the last group is
  // clipped and the list scrolls -- which is what happened when the worst-case
  // description alone decided the height.
  const pane = $("pane"), fields = $("fields");
  const prevH = pane.style.height;
  pane.style.height = "auto";
  const natural = fields.scrollHeight + 2;
  pane.style.height = prevH;
  PANE_H = Math.max(sizePane(true), natural);
  qual.innerHTML = saved;
  pane.style.height = PANE_H + "px";
}
addEventListener("resize", measurePane);

function sizePane(measuring){
  const aside = document.querySelector("aside"), pane = $("pane");
  if(!aside || !pane) return;
  // Collapse the pane AND clip it, so nothing inside contributes to the
  // measurement, then read where the description actually ends. scrollHeight
  // is useless here: it never reports less than clientHeight, so with the pane
  // hidden it returned the full sidebar height and the space always looked
  // spent -- which pinned the map at its floor and let the ratings overflow
  // into the text beneath.
  // min-height must go too, or the collapse does not collapse: the pane kept
  // its 180px floor and measured itself as part of what the sidebar needs.
  const prevH = pane.style.height, prevO = pane.style.overflow,
        prevM = pane.style.minHeight;
  pane.style.height = "0px"; pane.style.overflow = "hidden";
  pane.style.minHeight = "0px";
  const top = aside.getBoundingClientRect().top;
  const end = $("qual").getBoundingClientRect().bottom;
  const used = end - top;
  pane.style.height = prevH; pane.style.overflow = prevO;
  pane.style.minHeight = prevM;
  const avail = aside.clientHeight - used - 8;
  const w = pane.getBoundingClientRect().width;
  const h = Math.max(180, Math.min(w, avail));
  pane.style.height = h + "px";
  return h;
}

function setTab(mapOn){
  $("fields").classList.toggle("off", mapOn);
  $("mapwrap").classList.toggle("off", !mapOn);
  $("tabR").classList.toggle("on", !mapOn);
  $("tabM").classList.toggle("on", mapOn);
  if(mapOn) markMap();
}

// THE REVERSE WALK IS THE OTHER RENDER, not the same frames backwards:
// facing the other way down the street is a different set of images. The
// button shows the direction you are heading, read off the frame filenames'
// cardinal, and only appears when this file carries both traversals.
function walksOf(name){
  const seen=[];
  ALL.forEach(s=>{ if(s.sf===name && !seen.includes(s.wk)) seen.push(s.wk); });
  return seen;
}
function headingOf(steps){
  // the cardinal letter the exporter put in every filename of this walk
  const m=(steps[0]||{}).img && steps[0].img.match(/_([NSEW])(_[LRF])?\.jpg/);
  if(m) return m[1];
  const t={east_to_west:"W", west_to_east:"E",
           north_to_south:"S", south_to_north:"N"}[ (steps[0]||{}).wk ];
  return t||"";
}
function pickStreet(name, wk){
  CUR=name; $("pick").value=name;
  const wks=walksOf(name);
  CURWK = (wk && wks.includes(wk)) ? wk : wks[0]||"";
  STEPS=ALL.filter(s=>s.sf===name && s.wk===CURWK && !s.aux);
  const d=$("dir");
  if(wks.length>1){
    d.hidden=false;
    d.textContent="heading "+headingOf(STEPS)+"  \u21c4";
  } else d.hidden=true;
  i=0;
  if(STEPS.length>1){
    const n=STEPS.length;
    $("strip").style.background="linear-gradient(to right,"+STEPS.map(
      (s,j)=>s.k+" "+(j/(n-1)*100).toFixed(2)+"%").join(",")+")";
  } else {
    $("strip").style.background=STEPS.length?STEPS[0].k:"#171a1f";
  }
  draw();
}
if(document.readyState==="loading")
  document.addEventListener("DOMContentLoaded",start);
else start();
</script>
<div id="frames" hidden>__FRAMES__</div>
"""

if __name__ == "__main__":
    main()
