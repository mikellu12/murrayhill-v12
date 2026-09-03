"""Export 180-degree along-street panoramas as two walks per street.

Each street gets two subfolders, one per direction of travel, and inside
each the images are numbered in the order you would meet them walking that
way. Flipping through a folder is then the walk itself: the same block face
approached from opposite ends, which is the comparison the study is about.

Manhattan's grid is rotated about 29 degrees off true north, so no street
here runs at a compass cardinal. Two consequences, handled separately:

  - The IMAGE is centred on the street's true axis, not on true east. A
    panorama centred at 90 degrees on a street running at 118 would put the
    vanishing point 28 degrees off centre, which is precisely the view a
    walker never has.
  - The FOLDER is named by the dominant component of that axis, so cross
    streets read east/west and avenues north/south, the way the grid is
    spoken about. Every street here sits within 2 degrees of 28 or 118, far
    from the 45-degree line where the two would be ambiguous.

One bearing per street, fitted from that street's node positions, rather
than the per-node street_axis_deg -- see _street_axis for why. These streets
are straight to within 0.2 degrees, so nothing is lost, and a single bearing
also guarantees the two walks are exact opposites.

Ordering is each node's projection onto the travel bearing: along-street
distance measured in the direction of travel, with distance across the
street breaking ties.

Sequence numbers are zero padded by default. The point of numbering is that
the folder sorts into walking order, and a file browser sorts names as text:
unpadded, 10 lands between 1 and 2. Pass --no-pad for bare numbers.

Street View imagery is not redistributable and Google caps caching at 30
days. These are reprojections of that imagery, not derived metrics, so the
same limit applies: keep them private and treat them as scratch.

    .venv/Scripts/python tools/export_svi_180.py --out D:/svi_180
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
from common import PROJ_CRS, CFG, PROC, banner, street_grouping, image_path

FOV = CFG["directional"]["fov"]           # 180
UTM = PROJ_CRS       # metre CRS from config; 32618 for Manhattan
SEQ_WIDTH = 3          # zero-pad the sequence to this many digits
CARDINALS = {"N": 0, "E": 90, "S": 180, "W": 270}


def _cardinal(bearing: float) -> str:
    """Nearest compass letter, for a filename a person can read."""
    return min(CARDINALS, key=lambda k: min(abs(bearing - CARDINALS[k]),
                                            360 - abs(bearing - CARDINALS[k])))


def _street_axis(east: np.ndarray, north: np.ndarray) -> float:
    """Principal direction of a street's own node positions, as a bearing.

    Fitted from the coordinates rather than averaged from metrics.csv's
    street_axis_deg. That column is fitted per node from its five nearest
    chain-mates, and on 1st Avenue -- the one street here with node runs on
    both kerbs, 27 m apart -- the five nearest include the far side, so ten
    nodes come out 30 to 90 degrees across the avenue. Averaging those drags
    the street mean off by more than a degree; a fit over all the street's
    points is dominated by its 700 m length and ignores the 27 m width.
    """
    xy = np.column_stack([east - east.mean(), north - north.mean()])
    vx, vy = np.linalg.svd(xy, full_matrices=False)[2][0]
    return np.degrees(np.arctan2(vx, vy)) % 180


# Being inside a road tunnel has a two-part signature, and both parts are
# needed. No sky alone also describes the Tunnel Exit Street trench, which
# is open above and is real streetscape the study deliberately keeps. A
# large unnameable share alone also describes a scaffolding deck or a
# retaining wall, which are likewise real. Together they mean the camera is
# in a tiled tube: no sky, and a third of the field that ADE20K has no
# street class for.
TUNNEL_MAX_SKY = 0.02
TUNNEL_MAX_MASS = 0.78

# Park Avenue leaves grade north of East 41st and climbs onto the viaduct
# that carries it around Grand Central. The deck is a vehicular flyover with
# no sidewalk at all, so its nodes are not pedestrian streetscape in any
# sense the study measures -- they photograph a balustrade and the terminal's
# cornice from a roadway no one can stand on.
#
# This is a LIST, not a rule, because no rule was available. CSCL maps the
# ramp as ordinary PARK AVE, rw_type 1: only a five-segment stub over
# Pershing Square carries the PARK AVE VIADUCT name, and selecting on
# proximity to it instead flags the East 42nd Street nodes passing
# underneath. The segmentation signature (no sidewalk, heavy railing) does
# separate these four within Park Avenue but collides with ordinary nodes
# whose sidewalk is merely occluded by traffic.
#
# node_id is POSITIONAL. Rebuild the frame and these ids
# point somewhere else, so this list must be re-checked by eye, not trusted.
# From config, because node IDs are positional and a hardcoded set silently
# follows the code into another study area. See config.yaml: excluded_nodes.
# every named exclusion list in config counts, not only "viaduct" -- the
# approach ramp got its own list and silently did not exist to this set
VIADUCT_NODES = set().union(*(v for v in
    CFG.get("excluded_nodes", {}).values())) if CFG.get("excluded_nodes")     else set()


def _tunnel_nodes(node_ids) -> dict[str, tuple[float, float]]:
    """Nodes whose panorama is a tunnel interior rather than a street."""
    path = PROC / "sim_profiles.npz"
    if not path.exists():
        return {}
    z = np.load(path, allow_pickle=True)
    rows = [str(r) for r in z["__rows__"]]
    sky_row = rows.index("sky")
    found = {}
    for nid in node_ids:
        if nid not in z:
            continue
        a = z[nid]
        w = a[-1].sum()
        if w <= 0:
            continue
        sky, mass = a[sky_row].sum() / w, a[:-1].sum() / w
        if sky < TUNNEL_MAX_SKY and mass < TUNNEL_MAX_MASS:
            found[nid] = (sky, mass)
    return found


def _walks(axis: float) -> list[tuple[float, str]]:
    """The two travel bearings of a street, with a folder name for each."""
    out = []
    for bearing in (axis % 360, (axis + 180) % 360):
        east, north = np.sin(np.radians(bearing)), np.cos(np.radians(bearing))
        if abs(east) > abs(north):
            name = "west_to_east" if east > 0 else "east_to_west"
        else:
            name = "south_to_north" if north > 0 else "north_to_south"
        out.append((bearing, name))
    return out


def panorama(frames: dict, axis: float, out_w: int) -> np.ndarray:
    """Reproject fov=90 frames into one cylindrical strip centred on `axis`."""
    H, W = next(iter(frames.values())).shape[:2]
    f = W / 2.0                                   # tan(45 deg) = 1
    fc = out_w / np.radians(FOV)
    out_h = int(2 * fc * np.tan(np.radians(45.0)))

    theta = np.radians(axis) + (np.arange(out_w) - out_w / 2) / fc
    # Row 0 is the top of the output, which is positive elevation. Getting
    # this sign wrong returns the panorama flipped: sky underfoot.
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


def _load(frames_df) -> dict | None:
    """A node's four frames, or None if any will not open.

    A node built from three headings carries a quarter of the horizon black,
    which in a sequence reads as a real gap in the street rather than as a
    missing file.
    """
    frames = {}
    for f in frames_df.itertuples():
        try:
            frames[float(f.heading)] = np.asarray(
                Image.open(image_path(f.path)).convert("RGB"))
        except Exception:
            return None
    return frames if len(frames) >= 4 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True,
                    help="destination; street/direction folders are made inside")
    # 2880, not 1440. Both exports defaulted to 1440, but this one spreads it
    # over 180 degrees and export_svi_90 spreads it over 90, so the same number
    # meant half the angular resolution here -- 8 px/degree against 16 -- and,
    # because the vertical field is derived from fc, half the height as well:
    # 1440x916 against 1440x1833, from identical source pixels. That was a
    # copied default, never a choice. Matching px/degree makes the two
    # comparable, which is what the street-type split in export_svi_90 needs.
    #
    # Existing data/raw/svi_180 was written at 1440 and is 8 px/degree. It fed
    # only exploratory tools -- no rating in results/tables came from it -- but
    # re-running here at the new default will put two sizes in one folder
    # unless the old tree is cleared first.
    ap.add_argument("--width", type=int, default=2880,
                    help="panorama width in px (default 2880, 16 px per "
                         "degree, matching export_svi_90)")
    ap.add_argument("--quality", type=int, default=88)
    ap.add_argument("--no-pad", action="store_true",
                    help="bare sequence numbers instead of zero padded")
    ap.add_argument("--keep-tunnels", action="store_true",
                    help="keep tunnel interiors and viaduct deck nodes")
    ap.add_argument("--only", default="",
                    help="render only these street folders, comma separated; "
                         "for re-rendering one street after a fix rather than "
                         "the whole frame")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N streets, for a quick check")
    ap.add_argument("--no-segments", action="store_true",
                    help="ignore street_segment and fall back to "
                         "cleaned_street; segments are the default")
    args = ap.parse_args()
    banner("export 180-degree panoramas")

    manifest = pd.read_csv(PROC / "manifest.csv")
    nodes = gpd.read_file(PROC / "nodes.gpkg")
    nodes, label = street_grouping(nodes, prefer_segments=not args.no_segments)
    print(f"grouping by {label} ({nodes.folder.nunique()} streets, "
          f"{len(nodes)} nodes)")

    info = nodes[nodes.node_id.isin(set(manifest.node_id))].copy()

    tunnels = {} if args.keep_tunnels else _tunnel_nodes(info.node_id)
    viaduct = set() if args.keep_tunnels else (VIADUCT_NODES & set(info.node_id))
    if tunnels or viaduct:
        info = info[~info.node_id.isin(set(tunnels) | viaduct)]
    utm = info.to_crs(UTM)
    info["_e"], info["_n"] = utm.geometry.x.values, utm.geometry.y.values

    streets = sorted(info.folder.unique())
    if args.limit:
        streets = streets[:args.limit]
    if tunnels:
        print(f"dropped {len(tunnels)} node(s) inside a road tunnel "
              f"(sky < {TUNNEL_MAX_SKY:.0%}, classified < {TUNNEL_MAX_MASS:.0%}):")
        for nid, (sky, mass) in sorted(tunnels.items()):
            street = nodes.loc[nodes.node_id == nid, "osm_name"].iloc[0]
            print(f"  {nid}  {street:<20} sky {sky:.3%}  classified {mass:.1%}")
    if viaduct:
        print(f"dropped {len(viaduct)} node(s) on the Park Avenue viaduct "
              f"(elevated roadway, no sidewalk): {', '.join(sorted(viaduct))}")
    print(f"{len(info)} of {len(nodes)} study nodes exported")
    print(f"{len(streets)} streets x 2 directions -> {args.out}\n")

    by_node = {n: g for n, g in manifest.groupby("node_id")}
    written, missing, summary = 0, [], []

    only = {t.strip() for t in args.only.split(",") if t.strip()}
    if only:
        streets = [s_ for s_ in streets if s_ in only]
        print(f"rendering only: {', '.join(streets)}")
    for street in tqdm(streets, desc="streets", mininterval=1.0):
        g = info[info.folder == street]
        # FIT ON ONE CHAIN, NOT THE WHOLE FOLDER. The docstring below assumes a
        # folder is one straight street, and for eighteen of nineteen it is --
        # pooled and per-chain axes agree to within a degree. Park Avenue's
        # tunnel segment is four disjoint chains spread over 970 by 690 m, and
        # pooling them returned 99.5 degrees against the chains' own 28.5, so
        # every one of its 57 frames was rendered facing across the avenue
        # instead of along it. The dominant chain is the street; the others are
        # the same street resumed after an interruption, and are parallel to it.
        if "chain" in g.columns and g.chain.notna().any():
            dom = g.chain.value_counts().idxmax()
            gm = g[g.chain == dom]
            axis = _street_axis(gm._e.to_numpy(), gm._n.to_numpy())
            pooled = _street_axis(g._e.to_numpy(), g._n.to_numpy())
            off = min(abs(pooled - axis), 180 - abs(pooled - axis))
            if off > 5:
                print(f"  {street}: chains disagree with the pooled fit by "
                      f"{off:.1f} deg -- using the dominant chain "
                      f"({axis:.1f}, pooled was {pooled:.1f})")
        else:
            axis = _street_axis(g._e.to_numpy(), g._n.to_numpy())

        for bearing, walk in _walks(axis):
            # Along-street distance measured in the direction of travel,
            # with distance across the street as the tiebreak. 1st Avenue
            # carries two node runs 27 m apart -- both kerbs of a 100 ft
            # avenue -- which share an along-street position, so without the
            # tiebreak the two sides interleave in an arbitrary order.
            e, n_ = np.sin(np.radians(bearing)), np.cos(np.radians(bearing))
            ordered = g.assign(
                _proj=np.round(g._e * e + g._n * n_),
                _perp=g._e * n_ - g._n * e,
            ).sort_values(["_proj", "_perp"])
            # FIXED width, never len(ordered). Deriving it from the batch
            # made the filename depend on how many nodes that run happened to
            # export: a --missing-only pass over 4 nodes wrote 1_, 2_ into a
            # folder the full run had written 01_, 02_ into. Same node, two
            # names, and a sorted listing that puts 10_ before 1_.
            width = SEQ_WIDTH
            folder = args.out / street / walk
            folder.mkdir(parents=True, exist_ok=True)

            for seq, row in enumerate(ordered.itertuples(), start=1):
                frames = _load(by_node[row.node_id])
                if frames is None:
                    missing.append(row.node_id)
                    continue
                # The street bearing renders the view, not the node's own
                # street_axis_deg. That axis is fitted from the five nearest
                # chain-mates, which on 1st Avenue includes nodes on the far
                # kerb, so ten of them come out 30 to 90 degrees across the
                # avenue. Rendering from those would centre the panorama on
                # a building face. Every street here is straight to within
                # 0.2 degrees, so one bearing per street loses nothing.
                img = panorama(frames, bearing, args.width)
                n = str(seq) if args.no_pad else str(seq).zfill(width)
                Image.fromarray(img).save(
                    folder / f"{n}_{row.node_id}_{_cardinal(bearing)}.jpg",
                    quality=args.quality, optimize=True)
                written += 1
            summary.append((street, walk, len(list(folder.glob("*.jpg")))))

    print(f"\nwrote {written} panoramas in {len(streets)} street folders\n")
    for street, walk, n in summary:
        print(f"  {street:<22} {walk:<15} {n:>4} images")
    if missing:
        u = sorted(set(missing))
        print(f"\n{len(u)} node(s) skipped for an incomplete frame set: "
              f"{', '.join(u[:8])}{' ...' if len(u) > 8 else ''}")
    total = sum(p.stat().st_size for p in args.out.rglob("*.jpg"))
    print(f"\ntotal {total / 1024 / 1024:.0f} MB in {args.out}")


if __name__ == "__main__":
    main()
