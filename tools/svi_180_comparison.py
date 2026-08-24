"""Photo above mask, in a tree you can arrow through like the export itself.

`svi_180_segformer.py --overlays` writes a flat directory keyed by a mangled
filename, which is fine for spot checks and useless for reading a street. This
rebuilds the same pairs as

    <street>/<direction>/<seq>_<node_id>_<cardinal>.jpg

mirroring data/raw/svi_180 exactly, so the folder sorts into walking order and
holding the arrow key down walks the block face with the segmentation moving
underneath it. That is the comparison the export was shaped for; see the
sequence-numbering note in export_svi_180.py for why the padding matters.

Stacked rather than side by side because the panoramas are 1440 wide and a
180-degree view laid beside itself is 2900 px, wider than the screen it has to
be read on. Stacked, the two halves share a horizontal axis, so a facade in
the photo sits directly above the pixels claimed for it.

No GPU and no model. It reads the masks the segmentation run already wrote,
so it is seconds to re-run with a different width, and it cannot change a
result -- if a mask is missing it is reported, never regenerated silently.

Street View pixels, so the output tree is gitignored like every other
directory here that draws them.

    .venv/Scripts/python tools/svi_180_comparison.py
    .venv/Scripts/python tools/svi_180_comparison.py --width 1024 --street 1st_avenue
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import RES, banner

CLASS_COLS = ["road", "sidewalk", "building", "wall", "fence", "pole",
              "traffic_light", "traffic_sign", "vegetation", "terrain", "sky",
              "person", "rider", "car", "truck", "bus", "train", "motorcycle",
              "bicycle"]


def _font(size: int):
    """A real face if Pillow shipped one; its bitmap font is 11 px and the
    caption has to stay readable on a 1440 px panorama."""
    for name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/raw/svi_180"))
    ap.add_argument("--masks", type=Path, default=RES / "svi_180_seg")
    ap.add_argument("--out", type=Path,
                    default=RES / "svi_180_segmented_comparison")
    ap.add_argument("--table", type=Path,
                    default=RES / "tables" / "svi_180_segformer.csv")
    ap.add_argument("--width", type=int, default=1440,
                    help="output width in px; halves the file size at 1024")
    ap.add_argument("--street", default=None, help="only this street folder")
    ap.add_argument("--quality", type=int, default=85)
    ap.add_argument("--masks-only", action="store_true",
                    help="write the mask alone -- derived data, no Street View "
                         "pixels, so the result is publishable where the "
                         "photo-over-mask tree is not")
    args = ap.parse_args()
    banner("photo over mask, in walking order")

    if not args.masks.exists():
        sys.exit(f"no masks at {args.masks} -- run tools/svi_180_segformer.py "
                 f"--overlays first")

    shares = {}
    if args.table.exists():
        t = pd.read_csv(args.table)
        cols = [c for c in CLASS_COLS if c in t.columns]
        for r in t.itertuples():
            s = {c: getattr(r, c) for c in cols}
            top = sorted(s, key=s.get, reverse=True)[:5]
            shares[r.file] = "   ".join(f"{c} {s[c]:.1f}%" for c in top)

    photos = sorted(args.src.rglob("*.jpg"))
    if args.street:
        photos = [p for p in photos if p.parent.parent.name == args.street]
    if not photos:
        sys.exit(f"no panoramas under {args.src}"
                 + (f" for street {args.street}" if args.street else ""))
    print(f"{len(photos)} panoramas")

    font = _font(15)
    cap = 20
    written, missing = 0, []

    for p in photos:
        rel = str(p.relative_to(args.src)).replace("\\", "/")
        mask_p = args.masks / (rel.replace("/", "__")[:-4] + "_mask.png")
        if not mask_p.exists():
            missing.append(rel)
            continue

        photo = Image.open(p).convert("RGB")
        mask = Image.open(mask_p).convert("RGB")
        w = args.width
        h = round(w * photo.height / photo.width)
        if photo.size != (w, h):
            photo = photo.resize((w, h), Image.LANCZOS)
        # NEAREST on the mask: it is a palette of flat class colours, and any
        # smoothing invents boundary colours that belong to no class.
        if mask.size != (w, h):
            mask = mask.resize((w, h), Image.NEAREST)

        rows = 1 if args.masks_only else 2
        out = Image.new("RGB", (w, cap + h * rows + (rows - 1)), "white")
        d = ImageDraw.Draw(out)
        d.text((4, 3), f"{rel}      {shares.get(rel, '')}", fill="black",
               font=font)
        if args.masks_only:
            out.paste(mask, (0, cap))
        else:
            out.paste(photo, (0, cap))
            out.paste(mask, (0, cap + h + 1))

        dest = args.out / Path(rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        out.save(dest, quality=args.quality, optimize=True)
        written += 1

    mb = sum(f.stat().st_size for f in args.out.rglob("*.jpg")) / 1024 / 1024
    print(f"\nwrote {written} comparisons -> {args.out}  ({mb:.0f} MB)")
    print(f"each {args.width}x{cap + 2 * round(args.width * 916 / 1440) + 1} px,"
          f" photo over mask")
    if missing:
        print(f"\n{len(missing)} without a mask (segmentation not run for them): "
              f"{', '.join(missing[:5])}{' ...' if len(missing) > 5 else ''}")
    for street in sorted({p.parent.parent.name for p in photos}):
        n = len(list((args.out / street).rglob("*.jpg"))) if (args.out / street).exists() else 0
        print(f"  {street:<24}{n:>4} images")


if __name__ == "__main__":
    main()
