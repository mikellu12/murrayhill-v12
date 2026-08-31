"""Export the forward view as two 90-degree halves, left and right of the walk.

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

UTM = PROJ_CRS       # metre CRS from config; 32618 for Manhattan
HALF_FOV = 90.0
# Each half is centred 45 degrees off the walk bearing, so left spans -90 to 0
# and right spans 0 to +90 -- together exactly the forward 180.
OFFSET = 45.0
SIDES = (("L", -OFFSET), ("R", +OFFSET))
SEQ_WIDTH = 3          # zero-pad the sequence to this many digits


def panorama90(frames, centre, out_w):
    """Reproject fov=90 frames into a 90-degree cylindrical strip.

    Same construction as export_svi_180.panorama, with FOV as a parameter
    rather than the module constant. Halving the FOV doubles fc, which doubles
    both the pixels per degree and the vertical extent.
    """
    H, W = next(iter(frames.values())).shape[:2]
    f = W / 2.0
    fc = out_w / np.radians(HALF_FOV)
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

    by_node = {n: g for n, g in manifest.groupby("node_id")}
    written, missing, summary = 0, [], []

    for street in tqdm(streets, desc="streets", mininterval=1.0):
        g = info[info.folder == street]
        axis = _street_axis(g._e.to_numpy(), g._n.to_numpy())
        for bearing, walk in _walks(axis):
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
                if row.node_id not in to_write:
                    continue          # numbered, just not written this run
                frames = _load(by_node[row.node_id])
                if frames is None:
                    missing.append(row.node_id)
                    continue
                n = str(seq).zfill(width)
                for side, off in SIDES:
                    img = panorama90(frames, (bearing + off) % 360, args.width)
                    Image.fromarray(img).save(
                        folder / f"{n}_{row.node_id}_{_cardinal(bearing)}_{side}.jpg",
                        quality=args.quality, optimize=True)
                    written += 1
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
