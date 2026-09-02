"""Check an imagery set's mast detection against its src/mast.py calibration.

WHAT THIS DOES NOT ASSUME. An earlier version of this tool averaged frames and
looked for a low-variance region, on the reasoning that the mast is the part of
the frame that never changes. That is true only where the render bearings sit
at a constant offset from the source headings, which is a property of Murray
Hill's grid, not of the method: its streets run 28 and 118 degrees against
source frames fetched at 0/90/180/270, so the mast lands in the same pixels
every time and its anchor x has an interquartile range of 0.002.

Give the exporter per-node bearings off a curving street and that stops being
true -- in the City of London the anchor x runs from 0.03 to 0.85 -- and the
variance test reports failure on a set where detection is in fact working. It
was wrong about London, and it was wrong in a way that read as a data problem
rather than a tool problem.

src/mast.py never assumed a fixed position. It anchors per frame, and its own
docstring says so: the mast is a camera part, its size is a constant and only
its position moves. So the thing to measure is not whether the mast holds
still, it is whether the detector finds it and whether what it finds is the
right SIZE -- size being the constant, and the one thing a wrong calibration
would get wrong.

    .venv/Scripts/python tools/mast_calibrate.py --src data/london/raw/svi_90
    .venv/Scripts/python tools/mast_calibrate.py --src data/london/raw/svi_90 \
        --pattern "*_F.jpg" --name svi_180
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import banner
from mast import anchors, mast_mask, SETS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--pattern", default="*.jpg")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--name", default=None, help="SETS entry to test against")
    args = ap.parse_args()
    banner(f"mast detection in {args.src}")

    name = args.name or args.src.name
    if name not in SETS:
        sys.exit(f"no calibration named {name!r}; have {sorted(SETS)}")
    files = sorted(args.src.rglob(args.pattern))
    if not files:
        sys.exit(f"no frames matching {args.pattern!r} under {args.src}")
    rng = np.random.default_rng(args.seed)
    if len(files) > args.n:
        files = [files[i] for i in sorted(rng.choice(len(files), args.n, replace=False))]

    xs, ns, shares, size = [], [], [], None
    for f in tqdm(files, desc="frames", mininterval=2.0):
        im = Image.open(f).convert("RGB")
        size = im.size
        a = anchors(im, name)
        ns.append(len(a))
        xs.extend([x / im.size[0] for x, _ in a])
        shares.append(float(mast_mask(im, name).mean()))
    ns, xs, shares = np.array(ns), np.array(xs), np.array(shares)
    c = SETS[name]
    found = (ns > 0).mean()

    print(f"\n  {len(files)} frames at {size[0]}x{size[1]}, calibration {name!r}\n")
    print(f"  detected in            {found*100:>6.1f}% of frames")
    print(f"  masts per frame        {ns.mean():>6.2f}   (calibration expects "
          f"up to {c['masts']})")
    print(f"  frame erased           {shares.mean()*100:>6.2f}%   "
          f"(one mast is {c['w']*c['h']*100:.2f}% by construction)")
    if len(xs):
        print(f"\n  anchor x, as a fraction of width:")
        print(f"    median {np.median(xs):.3f}   IQR {np.percentile(xs,25):.3f}"
              f"-{np.percentile(xs,75):.3f}   range {xs.min():.3f}-{xs.max():.3f}")
        spread = np.percentile(xs, 75) - np.percentile(xs, 25)
        print(f"    spread {spread:.3f} -- {'the mast holds still in this set' if spread < 0.02 else 'the mast moves between frames, which the per-frame anchor handles'}")

    # The size is the constant, so it is the only thing a wrong calibration
    # shows up in. Detection rate and position spread are reported, not judged:
    # a set whose bearings vary will scatter, and that is not a fault.
    print()
    ok = True
    exp = c["w"] * c["h"] * c["masts"]
    for lab, got, want, tol in (
            ("detection rate", found, 0.90, 0.15),
            ("erased share", shares.mean(), exp, exp * 0.6)):
        good = abs(got - want) <= tol
        ok &= good
        print(f"    {lab:<18}{got:>8.3f}  expect ~{want:<8.3f}"
              f"{'ok' if good else 'OUT OF RANGE'}")
    print(f"\n  {'calibration holds' if ok else 'inspect masked frames before trusting this set'}")


if __name__ == "__main__":
    main()
