"""A walk down one street, as a GIF, from the rendered half-views.

Two shapes, because the export makes two.

  A VEHICULAR street is rendered as two 90-degree halves per node, left and
  right of the direction of travel. Both are shown, side by side with a seam
  down the middle, so one frame is the whole forward 180 degrees and walking
  the street shows both frontages passing at once. That pairing is the point:
  a node's two halves are the same instant looking two ways, and the study
  rates them separately.

  A PEDESTRIAN way is rendered as a single 180-degree strip, so a frame is
  already the whole view and the GIF is just the strips in order.

ORDER COMES FROM THE FILENAME, not from a directory listing. The exporter
zero-pads a sequence number that counts along the walk, so sorting on it walks
the street; sorting on the node id would walk the order the nodes were
generated in, which is a different street entirely on a frame where ids were
renumbered.

ONE WALK PER FILE. The two walks of a street are opposite traversals of the
same frontages, so joining them would run to the far end and teleport back.

    .venv/Scripts/python tools/walk_gif.py --street london_wall
    SIM_CONFIG=config_london.yaml .venv/Scripts/python tools/walk_gif.py \\
        --street watling_street
"""
import argparse
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import CFG, PROC, RAW, RES, banner

NAME = re.compile(r"^(\d+)_(n\d+)_([NESW])_([LRF])\.jpg$")
SEAM = (18, 18, 20)


def frames_for(walk_dir):
    """Files in walking order, grouped by node."""
    got = {}
    for p in sorted(walk_dir.glob("*.jpg")):
        m = NAME.match(p.name)
        if not m:
            continue
        seq, node, card, side = m.groups()
        got.setdefault((int(seq), node), {})[side] = p
    return [got[k] for k in sorted(got)]


def compose(sides, width, seam_px):
    """One frame: L|R side by side for a vehicular node, or the F strip."""
    if "F" in sides:
        im = Image.open(sides["F"]).convert("RGB")
        h = int(width * im.height / im.width)
        return im.resize((width, h), Image.LANCZOS)
    if "L" not in sides or "R" not in sides:
        return None
    half = (width - seam_px) // 2
    out = None
    for i, s in enumerate(("L", "R")):
        im = Image.open(sides[s]).convert("RGB")
        h = int(half * im.height / im.width)
        im = im.resize((half, h), Image.LANCZOS)
        if out is None:
            out = Image.new("RGB", (width, h), SEAM)
        out.paste(im, (i * (half + seam_px), 0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=None)
    ap.add_argument("--street", required=True, help="street folder name")
    ap.add_argument("--walk", default=None, help="one walk; default is each")
    ap.add_argument("--width", type=int, default=900)
    ap.add_argument("--seam", type=int, default=3)
    ap.add_argument("--ms", type=int, default=260, help="frame duration")
    ap.add_argument("--label", action="store_true", default=True)
    ap.add_argument("--no-label", dest="label", action="store_false")
    ap.add_argument("--sheet", type=int, default=0,
                    help="write a still contact sheet of N evenly spaced "
                         "frames instead of a GIF; a slide plays a still "
                         "reliably and a GIF does not")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    banner(f"walk: {args.street}")

    src = args.src or (RAW / "svi_90")
    base = src / args.street
    if not base.exists():
        sys.exit(f"no such street folder: {base}")
    walks = ([base / args.walk] if args.walk
             else sorted(p for p in base.iterdir() if p.is_dir()))
    out_dir = args.out or (RES / "figures" / "walks")
    out_dir.mkdir(parents=True, exist_ok=True)

    for w in walks:
        nodes = frames_for(w)
        if not nodes:
            print(f"  {w.name}: no frames")
            continue
        kind = "180 strip" if "F" in nodes[0] else "90 halves, L | R"
        imgs = []
        for i, sides in enumerate(nodes, 1):
            f = compose(sides, args.width, args.seam)
            if f is None:
                continue
            if args.label:
                d = ImageDraw.Draw(f)
                node = next(iter(sides.values())).name.split("_")[1]
                d.rectangle([0, f.height - 22, f.width, f.height], fill=SEAM)
                d.text((7, f.height - 17),
                       f"{args.street.replace('_', ' ')}   {w.name.replace('_', ' ')}"
                       f"   {i}/{len(nodes)}   {node}   {kind}",
                       fill=(232, 230, 225))
            imgs.append(f)
        if not imgs:
            print(f"  {w.name}: nothing composed")
            continue

        if args.sheet:
            # evenly spaced along the walk, so the sheet is the street rather
            # than its first N nodes
            k = min(args.sheet, len(imgs))
            idx = [round(i * (len(imgs) - 1) / max(k - 1, 1)) for i in range(k)]
            picks = [imgs[i] for i in idx]
            W = picks[0].width
            H = sum(p.height for p in picks)
            sheet = Image.new("RGB", (W, H), SEAM)
            y = 0
            for p_ in picks:
                sheet.paste(p_, (0, y)); y += p_.height
            o = out_dir / f"{args.street}__{w.name}.jpg"
            sheet.save(o, quality=86, optimize=True)
            print(f"  {w.name:<18}{k:>3} of {len(imgs)} frames  {kind:<18}"
                  f"{o.stat().st_size/1048576:>6.1f} MB  {o.name}")
            continue

        o = out_dir / f"{args.street}__{w.name}.gif"
        imgs[0].save(o, save_all=True, append_images=imgs[1:],
                     duration=args.ms, loop=0, optimize=True)
        mb = o.stat().st_size / 1048576
        print(f"  {w.name:<18}{len(imgs):>3} frames  {kind:<18}{mb:>6.1f} MB  {o.name}")


if __name__ == "__main__":
    main()
