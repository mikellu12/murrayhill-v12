"""Check that an imagery set's camera mast matches its src/mast.py calibration.

The detection METHOD never varies by city: bottom-anchored dark blob, anchor on
the leftmost, erase a fixed rectangle. Nor do the constants, really -- the mast
is one object of one angular size, so a set's w and h are that constant divided
by the field of view, and a new render geometry is arithmetic rather than a new
measurement. What a new city genuinely raises is whether the capture rig is the
same object at all, since Google's cars changed over the years and one frame is
a poor witness. This answers that, from the imagery, in about a minute.

THE MAST IS THE PART OF THE FRAME THAT NEVER CHANGES. Averaging a few hundred
frames blurs the streetscape away and leaves fixed elements sharp, which is how
the original numbers were found. The per-pixel standard deviation is the
sharper instrument and is what this uses: the mast is the same object in the
same place in every frame, so its variance across frames collapses toward zero
while real streetscape -- which differs at every node -- stays high. A dark
average alone would also flag a persistently shaded corner; near-zero variance
will not.

The candidate region is therefore low-variance AND dark AND touching the bottom
edge, which is the same three-property test src/mast.py applies to a single
frame. Frames are sampled across the whole set rather than taken in order, so a
single street cannot dominate the average.

    .venv/Scripts/python tools/mast_calibrate.py --src data/raw/svi_90
    .venv/Scripts/python tools/mast_calibrate.py --src data/london/raw/svi_90 \
        --pattern "*_F.jpg" --name london_wide
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from tqdm.auto import tqdm

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import banner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--pattern", default="*.jpg",
                    help="which frames to average; use *_F.jpg to calibrate "
                         "the wide strip separately from the 90 halves")
    ap.add_argument("--n", type=int, default=300,
                    help="frames to average (300 is ample; the mast is "
                         "identical in all of them)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--name", default=None, help="label for the SETS entry")
    ap.add_argument("--band", type=float, default=0.35,
                    help="bottom fraction of the frame to search")
    ap.add_argument("--out", type=Path, default=None,
                    help="write a diagnostic PNG of the variance map")
    args = ap.parse_args()
    banner(f"measure the mast in {args.src}")

    files = sorted(args.src.rglob(args.pattern))
    if not files:
        sys.exit(f"no frames matching {args.pattern!r} under {args.src}")
    rng = np.random.default_rng(args.seed)
    if len(files) > args.n:
        files = [files[i] for i in sorted(rng.choice(len(files), args.n, replace=False))]
    print(f"{len(files)} frame(s) sampled")

    W, H = Image.open(files[0]).size
    # Welford, so a few hundred full-size frames never sit in memory at once.
    n = 0
    mean = np.zeros((H, W), dtype=np.float64)
    m2 = np.zeros((H, W), dtype=np.float64)
    odd = 0
    for f in tqdm(files, desc="frames", mininterval=1.0):
        im = Image.open(f).convert("L")
        if im.size != (W, H):
            odd += 1
            continue
        x = np.asarray(im, dtype=np.float64)
        n += 1
        d = x - mean
        mean += d / n
        m2 += d * (x - mean)
    if odd:
        print(f"  {odd} frame(s) skipped for a different size than {W}x{H}")
    sd = np.sqrt(m2 / max(n - 1, 1))
    print(f"averaged {n} frame(s) at {W}x{H}\n")

    b0 = int(H * (1 - args.band))
    msd, mmean = sd[b0:], mean[b0:]
    # Low variance and dark, both relative to this band rather than absolute,
    # so the test does not depend on exposure. This finds the mast's dark core,
    # which is what mast.anchors keys on in a single frame.
    quiet = msd < np.percentile(msd, 10)
    dark = mmean < np.median(mmean) * 0.75
    lab, k = ndimage.label(quiet & dark)
    if not k:
        sys.exit("no low-variance dark region found in the bottom band")

    keep = []
    for i in range(1, k + 1):
        ys, xs = np.where(lab == i)
        if ys.max() < msd.shape[0] - 2:      # must touch the bottom edge
            continue
        if len(ys) < 0.0005 * msd.size:
            continue
        keep.append((xs.min(), xs.max(), ys.min(), len(ys)))
    if not keep:
        sys.exit("no bottom-anchored candidate; try --band larger")
    keep.sort()

    # Merge fragments that sit within 2% of the frame width of each other: a
    # low-contrast mast breaks into slivers and they are one object.
    merged, gap = [], 0.02 * W
    for x0, x1, y0, npx in keep:
        if merged and x0 - merged[-1][1] < gap:
            p = merged[-1]
            merged[-1] = (p[0], max(p[1], x1), min(p[2], y0), p[3] + npx)
        else:
            merged.append((x0, x1, y0, npx))

    print(f"  {len(merged)} mast(s) found in the bottom {args.band:.0%} "
          f"of the frame\n")
    print(f"  {'#':<3}{'x px':>14}{'x frac':>16}{'w core':>9}{'height':>9}")
    ws, hs = [], []
    for i, (x0, x1, y0, _) in enumerate(merged, 1):
        wc = (x1 - x0 + 1) / W
        h = (msd.shape[0] - y0) / H
        ws.append(wc); hs.append(h)
        print(f"  {i:<3}{f'{x0}-{x1}':>14}"
              f"{f'{x0/W:.3f}-{x1/W:.3f}':>16}{wc:>9.3f}{h:>9.3f}")

    # Compare against the configured entry rather than propose a replacement.
    # What this measurement can see is the DARK CORE, its height and how many
    # masts are in frame -- which is exactly what answers "is this the same
    # capture rig?", the question a new city raises. It cannot see the full
    # erase width: the wordmark is painted on the camera but reprojection
    # warps it differently at every heading, so it does not hold still across
    # frames and no variance test will find it. SETS w is wider than the core
    # for that reason and was fixed by looking at masked frames.
    name = args.name or args.src.name
    from mast import SETS
    print()
    if name not in SETS:
        print(f"  no SETS entry named {name!r}. Measured: core width "
              f"{max(ws):.3f}, height {max(hs):.3f}, {len(merged)} mast(s).")
        print(f"  Derive w from an existing set by the ratio of fields of view "
              f"rather than from the core alone.")
        return
    c = SETS[name]
    print(f"  against SETS[{name!r}]:")
    ok = True
    for lab, got, want, tol in (
            ("mast count", len(merged), c["masts"], 0),
            ("height", max(hs), c["h"], 0.03),
            ("core width <= max_w", max(ws), c["max_w"], None)):
        if tol is None:
            good = got <= want
            print(f"    {lab:<22}{got:>8.3f}  limit {want:<8.3f}"
                  f"{'  ok' if good else '  EXCEEDS'}")
        else:
            good = abs(got - want) <= tol
            print(f"    {lab:<22}{got:>8.3f}  expect {want:<7.3f}"
                  f"{'  ok' if good else '  DIFFERS'}")
        ok &= bool(good)
    verdict = ("same rig, calibration holds" if ok else
               "DIFFERENT from the configured set -- inspect masked frames")
    print(f"\n  {verdict}")

    if args.out:
        v = (sd - sd.min()) / max(sd.ptp(), 1e-9)
        Image.fromarray((v * 255).astype(np.uint8)).save(args.out)
        print(f"\n  variance map -> {args.out} (dark = never changes)")


if __name__ == "__main__":
    main()
