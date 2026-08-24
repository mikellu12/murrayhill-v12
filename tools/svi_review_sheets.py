"""Review sheets: the export in fixed-size batches, big enough to judge from.

Different job from svi_contact_sheets.py. That one is per street, sized to
scan a whole street at once and spot what stands out. This one is a labelling
instrument: fixed batch size, fixed grid, thumbnails large enough that a
facade scaffold -- not just an obvious sidewalk shed -- is decidable, and a
stable index printed on every tile so a verdict can be written down against
it without ambiguity.

Sheet width is held under 1570 px on purpose. Anything wider is downsampled
before it reaches a vision model, which silently shrinks every thumbnail and
costs exactly the detail the sheet exists to show.

No DOB boxes here. The permit flag is drawn on the contact sheets, where it
belongs as a reading aid; on a labelling sheet it would anchor the judgement
to the thing being tested.

    .venv/Scripts/python tools/svi_review_sheets.py
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

COLS = 2
ROWS = 6
THUMB_W = 760
CAP = 16
PAD = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/raw/svi_180"))
    ap.add_argument("--out", type=Path, default=RES / "review_sheets")
    args = ap.parse_args()
    banner("review sheets")

    files = []
    for street in sorted(p for p in args.src.iterdir() if p.is_dir()):
        for direction in sorted(d for d in street.iterdir() if d.is_dir()):
            files += sorted(direction.glob("*.jpg"),
                            key=lambda p: int(re.match(r"(\d+)_", p.name).group(1)))
    print(f"{len(files)} images")

    args.out.mkdir(parents=True, exist_ok=True)
    for p in args.out.glob("*.jpg"):
        p.unlink()

    per = COLS * ROWS
    probe = Image.open(files[0])
    tw, th = THUMB_W, int(THUMB_W * probe.height / probe.width)
    index = []

    for s in range(0, len(files), per):
        batch = files[s:s + per]
        sheet_no = s // per + 1
        rows = (len(batch) + COLS - 1) // COLS
        sheet = Image.new("RGB", (COLS * (tw + PAD) + PAD,
                                  rows * (th + CAP + PAD) + PAD + 20), "white")
        d = ImageDraw.Draw(sheet)
        d.text((PAD, 5), f"sheet {sheet_no}   items {s + 1}-{s + len(batch)}",
               fill="black")
        for i, f in enumerate(batch):
            n = s + i + 1
            x = PAD + (i % COLS) * (tw + PAD)
            y = 20 + PAD + (i // COLS) * (th + CAP + PAD)
            rel = str(f.relative_to(args.src)).replace("\\", "/")
            d.text((x + 2, y + 1), f"[{n}]  {rel}", fill="black")
            sheet.paste(Image.open(f).resize((tw, th)), (x, y + CAP))
            index.append({"idx": n, "sheet": sheet_no, "file": rel})
        sheet.save(args.out / f"sheet_{sheet_no:03d}.jpg", quality=80, optimize=True)

    pd.DataFrame(index).to_csv(args.out / "index.csv", index=False)
    size = sum(p.stat().st_size for p in args.out.glob("*.jpg")) / 1024 / 1024
    print(f"{len(index)} items across {index[-1]['sheet']} sheets, {size:.0f} MB")
    print(f"grid {COLS}x{ROWS}, thumbnails {tw}x{th} px")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
