"""Export the forward view, split by street type: 90-degree halves or one 180.

A vehicular street gets two 90-degree halves, left and right of the walk. A
pedestrian way gets a single 180-degree strip. Which a node takes is read from
data/processed/street_type.csv; with no such file every node takes halves,
which is the Murray Hill case.

The reason is capture geometry. Street View is shot from a vehicle on the
roadway, so on a wide street the camera stands tens of metres off either
frontage and a 90-degree half faces one frontage squarely. On a narrow
pedestrian way the camera and the walker share a position, both frontages are
metres away, and enclosure fills the visual field -- a half-view would cut away
most of what makes that space feel as it does.

Both renders are held at the same pixels per degree, so they differ in how much
horizon is in frame and in nothing else: at the default width a half is
1440x1833 and a strip is 2880x1833, both 16 px per degree with the same
vertical field. The field of view is therefore part of what is measured. The
consequence belongs in the methods: an M from a 180-degree walkway and an M
from a 90-degree street are not strictly on one scale and should be reported as
typology-specific.

The 180-degree export renders one strip centred on the street axis. This
splits that field into two 90-degree views -- the left half of what a walker
faces, and the right half -- each rendered in its own right rather than cropped
out of the wider strip.

Rendered, not cropped, and the difference is not cosmetic. Cropping svi_180 in
half gives 720x916 at 8 px per degree with the vertical field still fitted to
180 degrees. Rendering at FOV 90 gives 1440x1833 at 16 px per degree, twice the
angular resolution, with the vertical field fitted to the view actually being
shown. Since out_h = 2*fc*tan(45) and fc = out_w/radians(FOV), halving the FOV
doubles both.

Left and right are relative to the DIRECTION OF TRAVEL, not to the compass.
Each half is centred 45 degrees off the walk bearing, so the two together tile
the same forward 180 degrees the wider export covers, with no overlap and no
gap.

Two walks per street and two halves per walk gives four views per node. They
cover two physical frontages, each seen from both approaches, which is a free
consistency check: a frontage that rates differently depending on which way it
is passed is either a real directional effect or model noise, and the data can
say which.

Why halves rather than the pedestrian literature's 60-degree cones with 80/10/10
weighting: Street View is captured from a vehicle on the roadway, not from the
sidewalk, so a frontal-cone model of pedestrian attention does not match the
capture geometry. Halves that each face a frontage do.

Street View imagery is not redistributable and Google caps caching at 30 days.
These are reprojections of that imagery, so the same limit applies.

INPUTS. Only two things define the frame: data/processed/nodes.gpkg and the
coordinate CSV imported onto it by tools/import_segments.py, which supplies
`street_segment`. Grouping follows common.street_grouping -- street_segment
first, then cleaned_street, then the raw chains -- so dropping in a new frame
and its CSV is enough to re-run everything. --no-segments forces the older
labelling.

    .venv/Scripts/python tools/import_segments.py --csv <coords.csv>
    .venv/Scripts/python tools/export_svi_90.py
    .venv-gpu/Scripts/python tools/sim_vlm_run.py --src data/raw/svi_90         --table results/tables/sim_vlm_v3.csv
    .venv/Scripts/python tools/sim_compute.py
"""
import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
from common import PROJ_CRS, CFG, PROC, banner, street_grouping
from export_svi_180 import (_cardinal, _street_axis, _walks, _load,
                            _tunnel_nodes, VIADUCT_NODES)


def _walk_name(bearing):
    """Folder name for a travel bearing -- the naming half of _walks."""
    east, north = np.sin(np.radians(bearing)), np.cos(np.radians(bearing))
    if abs(east) > abs(north):
        return "west_to_east" if east > 0 else "east_to_west"
    return "south_to_north" if north > 0 else "north_to_south"


def _circmean(deg):
    """Mean of bearings, which cannot be averaged arithmetically."""
    r = np.radians(np.asarray(deg, dtype=float))
    return float(np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())) % 360)


def _plan_street(g, use_headings):
    """The two walks of a street: (folder name, ordered rows, bearing column).

    With per-node headings supplied by the frame, both the bearing and the
    node ordering are read off the frame rather than re-derived here. That
    matters wherever streets curve: in the City of London 51 of the 72 streets
    with eight or more nodes span more than 30 degrees of heading and 25 span
    more than 60, so one fitted axis points a good fraction of a street's
    renders at the wrong frontage -- individual nodes sit up to 87 degrees off
    their own street's fitted axis. Manhattan's grid hides this, which is why
    the fitted axis was adequate there and is not here.
    """
    if use_headings:
        out = []
        for hcol, scol in use_headings:
            out.append((_walk_name(_circmean(g[hcol])),
                        g.sort_values(scol), hcol))
        return out

    axis = _street_axis(g._e.to_numpy(), g._n.to_numpy())
    out = []
    for bearing, walk in _walks(axis):
        e, n_ = np.sin(np.radians(bearing)), np.cos(np.radians(bearing))
        ordered = g.assign(_proj=np.round(g._e * e + g._n * n_),
                           _perp=g._e * n_ - g._n * e
                           ).sort_values(["_proj", "_perp"])
        out.append((walk, ordered, float(bearing)))
    return out

UTM = PROJ_CRS       # metre CRS from config; 32618 for Manhattan
HALF_FOV = 90.0
WIDE_FOV = 180.0
# Each half is centred 45 degrees off the walk bearing, so left spans -90 to 0
# and right spans 0 to +90 -- together exactly the forward 180.
OFFSET = 45.0
SIDES = (("L", -OFFSET), ("R", +OFFSET))
SEQ_WIDTH = 3          # zero-pad the sequence to this many digits


def panorama90(frames, centre, out_w, fov=None):
    """Reproject the source frames into a cylindrical strip of width `fov`.

    Same construction as export_svi_180.panorama, with FOV as a parameter
    rather than the module constant. Halving the FOV doubles fc, which doubles
    both the pixels per degree and the vertical extent.

    Holding px/degree fixed instead of holding the output width fixed is what
    lets a 90 and a 180 render be compared. At 16 px/degree both come out
    1833 px tall with the same vertical field, and the only difference between
    them is how much of the horizon is in frame -- which is the thing the
    street-type split is meant to vary.
    """
    fov = HALF_FOV if fov is None else fov
    H, W = next(iter(frames.values())).shape[:2]
    f = W / 2.0
    fc = out_w / np.radians(fov)
    out_h = int(2 * fc * np.tan(np.radians(45.0)))

    theta = np.radians(centre) + (np.arange(out_w) - out_w / 2) / fc
    phi = np.arctan((out_h / 2 - np.arange(out_h)) / fc)

    heads = np.array(sorted(frames))
    dh = (np.degrees(theta)[:, None] - heads[None, :] + 180) % 360 - 180
    pick = np.abs(dh).argmin(axis=1)
    alpha = np.radians(dh[np.arange(out_w), pick])

    rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    xs = f * np.tan(alpha) + W / 2.0
    for k, h in enumerate(heads):
        col = np.where(pick == k)[0]
        if not len(col):
            continue
        src_x = np.clip(np.round(xs[col]).astype(int), 0, W - 1)
        ys = H / 2.0 - f * np.tan(phi)[:, None] / np.cos(alpha[col])[None, :]
        src_y = np.clip(np.round(ys).astype(int), 0, H - 1)
        rgb[:, col] = frames[h][src_y, src_x[None, :]]
    return rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/raw/svi_90"))
    ap.add_argument("--width", type=int, default=1440,
                    help="width per half in px (default 1440, 16 px per degree)")
    ap.add_argument("--quality", type=int, default=88)
    ap.add_argument("--keep-tunnels", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N streets, for a quick check")
    ap.add_argument("--no-segments", action="store_true",
                    help="ignore street_segment and fall back to "
                         "cleaned_street; segments are the default")
    ap.add_argument("--nodes", nargs="+", default=None,
                    help="export only these node_ids; everything already on "
                         "disk is left untouched")
    ap.add_argument("--missing-only", action="store_true",
                    help="export only nodes with no image under --out yet")
    args = ap.parse_args()
    banner("export 90-degree halves, left and right of the walk")

    manifest = pd.read_csv(PROC / "manifest.csv")
    nodes = gpd.read_file(PROC / "nodes.gpkg")

    nodes, label = street_grouping(nodes, prefer_segments=not args.no_segments)
    print(f"grouping by {label} ({nodes.folder.nunique()} streets, "
          f"{len(nodes)} nodes)")
    info = nodes[nodes.node_id.isin(set(manifest.node_id))].copy()

    # `seq` numbers a node's position along the WHOLE street, so the ordering
    # set must stay complete no matter what this run exports. Filtering here
    # and enumerating the remainder restarts the count at 1 for every partial
    # pass: a --missing-only run over 4 nodes wrote seq 1-4 into a folder that
    # already had 1-4, duplicating them and interleaving two orderings. The
    # filter therefore selects what to WRITE, never what to number.
    to_write = set(info.node_id)
    if args.missing_only:
        import re as _re
        have = {_re.search(r"(n\d+)", p.name).group(1)
                for p in args.out.rglob("*.jpg")}
        to_write -= have
        print(f"--missing-only: {len(have)} node(s) already exported, "
              f"{len(to_write)} left to do")
    if args.nodes:
        to_write &= set(args.nodes)
        print(f"--nodes: restricted to {len(to_write)} node(s)")

    tunnels = {} if args.keep_tunnels else _tunnel_nodes(info.node_id)
    viaduct = set() if args.keep_tunnels else (VIADUCT_NODES & set(info.node_id))
    if tunnels or viaduct:
        info = info[~info.node_id.isin(set(tunnels) | viaduct)]
        print(f"dropped {len(tunnels)} tunnel and {len(viaduct)} viaduct nodes")
    utm = info.to_crs(UTM)
    info["_e"], info["_n"] = utm.geometry.x.values, utm.geometry.y.values

    streets = sorted(info.folder.unique())
    if args.limit:
        streets = streets[:args.limit]
    fc = args.width / np.radians(HALF_FOV)
    out_h = int(2 * fc * np.tan(np.radians(45.0)))
    print(f"{len(info)} nodes, {len(streets)} streets x 2 walks x 2 halves")
    print(f"each half {args.width}x{out_h} px, "
          f"{args.width / HALF_FOV:.1f} px per degree\n")

    # Which nodes take the wide render. Absent the file every node is
    # vehicular, which is the Murray Hill case and the conservative default:
    # the 90-degree half is what the instrument was validated on.
    stf = PROC / "street_type.csv"
    if stf.exists():
        st = pd.read_csv(stf)
        wide = set(st.loc[st.is_pedestrian.fillna(False), "node_id"])
        print(f"street type: {len(wide)} pedestrian node(s) -> one "
              f"{WIDE_FOV:.0f} deg strip, {len(info) - len(wide)} vehicular "
              f"-> two {HALF_FOV:.0f} deg halves")
    else:
        wide = set()
        print("no street_type.csv; every node rendered as 90-degree halves")

    # Derived from --width rather than fixed, so the two renders stay locked
    # at the same pixels per degree whatever width is asked for. Equal px/deg
    # is what makes them comparable: both come out the same height with the
    # same vertical field, differing only in how much horizon is in frame.
    wide_w = int(args.width * WIDE_FOV / HALF_FOV)

    # Prefer bearings and orderings supplied with the frame over ones fitted
    # from coordinates here. Murray Hill's frame has neither column and falls
    # back to the fitted street axis.
    # The generator (github.com/ex032895-crypto/street-view-nodes) computes
    # each heading from the local street tangent, which is exactly the bearing
    # a render wants to be centred on. Fitting an axis here re-derives that
    # worse. Murray Hill's import renames the columns, so both spellings are
    # accepted; a frame carrying neither falls back to the fitted axis.
    use_headings = None
    for fwd, rev in (("heading_fwd_deg", "heading_rev_deg"),
                     ("osm_heading_fwd", "osm_heading_rev")):
        if {fwd, rev, "seq_fwd", "seq_rev"}.issubset(info.columns):
            use_headings = ((fwd, "seq_fwd"), (rev, "seq_rev"))
            print(f"walk bearings: per-node, from the frame ({fwd})")
            break
    if use_headings is None:
        have = [c for c in ("heading_fwd_deg", "osm_heading_fwd") if c in info.columns]
        print("walk bearings: one fitted axis per street" + (
            f" -- {have[0]} is present but seq_fwd/seq_rev are not, so the "
            f"walk ordering cannot be read from the frame" if have else ""))

    by_node = {n: g for n, g in manifest.groupby("node_id")}
    written, missing, summary = 0, [], []
    n_wide = n_half = 0

    for street in tqdm(streets, desc="streets", mininterval=1.0):
        g = info[info.folder == street]
        for walk, ordered, bearing_src in _plan_street(g, use_headings):
            # FIXED width, never len(ordered). Deriving it from the batch
            # made the filename depend on how many nodes that run happened to
            # export: a --missing-only pass over 4 nodes wrote 1_, 2_ into a
            # folder the full run had written 01_, 02_ into. Same node, two
            # names, and a sorted listing that puts 10_ before 1_.
            width = SEQ_WIDTH
            folder = args.out / street / walk
            folder.mkdir(parents=True, exist_ok=True)

            for seq, row in enumerate(ordered.itertuples(), start=1):
                if row.node_id not in to_write:
                    continue          # numbered, just not written this run
                frames = _load(by_node[row.node_id])
                if frames is None:
                    missing.append(row.node_id)
                    continue
                n = str(seq).zfill(width)
                # one bearing for the whole street, or this node's own
                bearing = (getattr(row, bearing_src)
                           if isinstance(bearing_src, str) else bearing_src)
                # A pedestrian way gets one 180-degree strip; a vehicular
                # street gets two 90-degree halves. See the module docstring
                # for why the field of view is part of the measurement rather
                # than a nuisance to standardise away.
                if row.node_id in wide:
                    img = panorama90(frames, bearing, wide_w, WIDE_FOV)
                    Image.fromarray(img).save(
                        folder / f"{n}_{row.node_id}_{_cardinal(bearing)}_F.jpg",
                        quality=args.quality, optimize=True)
                    written += 1
                    n_wide += 1
                else:
                    for side, off in SIDES:
                        img = panorama90(frames, (bearing + off) % 360,
                                         args.width, HALF_FOV)
                        Image.fromarray(img).save(
                            folder / f"{n}_{row.node_id}_{_cardinal(bearing)}_{side}.jpg",
                            quality=args.quality, optimize=True)
                        written += 1
                        n_half += 1
            summary.append((street, walk, len(list(folder.glob("*.jpg")))))

    print(f"\nwrote {written} halves in {len(streets)} street folders\n")
    for street, walk, n in summary:
        print(f"  {street:<22} {walk:<15} {n:>4} images")
    if missing:
        u = sorted(set(missing))
        print(f"\n{len(u)} node(s) skipped for an incomplete frame set")
    total = sum(p.stat().st_size for p in args.out.rglob("*.jpg"))
    print(f"\ntotal {total / 1024 / 1024:.0f} MB in {args.out}")


if __name__ == "__main__":
    main()
