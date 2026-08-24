"""Contact sheets of the exported panoramas, one per street, DOB flags marked.

Every image in the export at thumbnail size, in walking order, so a street
can be reviewed in one pass. Images whose DOB label says a sidewalk shed
sits in the forward cone are boxed and captioned with the distance.

The boxes are a reading aid, not a verdict. The permit label is a candidate
list: a permit's point is the building address rather than the structure,
sign-offs run late enough that a live record can outlast the shed, and a
shed across a wide avenue falls inside the match radius while being barely
visible. Spot checks found both false positives and real sheds. So the box
says "look here first", and the sheet exists because looking is the only
thing that has reliably settled these questions.

Sheets are built per street rather than per direction so both walks of the
same block face sit near each other: a shed missed from one approach is
usually obvious from the other.

    .venv/Scripts/python tools/svi_contact_sheets.py
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import RES, banner

THUMB_W = 330
COLS = 6
PAD = 4
CAP = 15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/raw/svi_180"))
    ap.add_argument("--flags", type=Path,
                    default=RES / "tables" / "svi_180_scaffold.csv")
    ap.add_argument("--out", type=Path, default=RES / "contact_sheets")
    args = ap.parse_args()
    banner("contact sheets")

    flag = {}
    if args.flags.exists():
        f = pd.read_csv(args.flags)
        flag = {r.file: (bool(r.shed_in_view), r.shed_nearest_m)
                for r in f.itertuples()}
        print(f"{sum(v[0] for v in flag.values())} images flagged by DOB")

    args.out.mkdir(parents=True, exist_ok=True)
    streets = sorted(p for p in args.src.iterdir() if p.is_dir())
    total, marked = 0, 0

    for street in streets:
        files = []
        # Walk order within each direction, directions in a stable order, so
        # the sheet reads the way the folders do.
        for direction in sorted(street.iterdir()):
            if direction.is_dir():
                files += sorted(direction.glob("*.jpg"),
                                key=lambda p: int(re.match(r"(\d+)_", p.name).group(1)))
        if not files:
            continue

        probe = Image.open(files[0])
        tw, th = THUMB_W, int(THUMB_W * probe.height / probe.width)
        rows = (len(files) + COLS - 1) // COLS
        sheet = Image.new("RGB",
                          (COLS * (tw + PAD) + PAD,
                           rows * (th + CAP + PAD) + PAD + 22), "white")
        d = ImageDraw.Draw(sheet)
        n_flag = sum(flag.get(str(p.relative_to(args.src)).replace("\\", "/"),
                              (False, 0))[0] for p in files)
        d.text((PAD, 6), f"{street.name}   {len(files)} images   "
                         f"{n_flag} DOB shed candidates (boxed)", fill="black")

        for i, p in enumerate(files):
            key = str(p.relative_to(args.src)).replace("\\", "/")
            is_flag, dist = flag.get(key, (False, None))
            x = PAD + (i % COLS) * (tw + PAD)
            y = 22 + PAD + (i // COLS) * (th + CAP + PAD)
            label = f"{p.parent.name[:1]}{p.stem}"
            if is_flag:
                label += f"  SHED {dist}m"
                marked += 1
            d.text((x + 1, y), label, fill="#b00000" if is_flag else "#333333")
            sheet.paste(Image.open(p).resize((tw, th)), (x, y + CAP))
            if is_flag:
                d.rectangle([x - 1, y + CAP - 1, x + tw, y + CAP + th],
                            outline="#e00000", width=3)
            total += 1

        dest = args.out / f"{street.name}.jpg"
        sheet.save(dest, quality=72, optimize=True)
        print(f"  {street.name:<24}{len(files):>4} images  {n_flag:>3} flagged  -> {dest.name}")

    size = sum(p.stat().st_size for p in args.out.glob("*.jpg")) / 1024 / 1024
    print(f"\n{total} images across {len(streets)} sheets, {marked} boxed, {size:.0f} MB")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
