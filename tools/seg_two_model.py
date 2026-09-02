"""Per-image class shares from two segmenters: Mapillary Vistas and ADE20K.

WHY TWO, AND WHY THESE TWO. Neither covers all ten fields alone.

  Mapillary Vistas (65 classes) is a street-level dataset and carries what a
  sidewalk study needs: Curb, Curb Cut, Pedestrian Area, Bench, Trash Can,
  Banner, Billboard, and separate front/back Traffic Signs -- the last matters
  for a directional index, since a sign facing away contributes nothing to
  legibility from this viewpoint. It also has Bridge and Tunnel, which give a
  pixel-level check on the OSM flags imported from the colleague's frame.

  ADE20K (150 classes) has windowpane and door, which Mapillary lacks
  entirely. Those are the only handles on facade_variation and
  ground_floor_activity, so dropping ADE20K would leave two fields unmeasured.

WHY NOT THE FOUR-MODEL TAXONOMY RUN. It took 15.3 h and its 30 classes are
worse than plain ADE20K wherever both have a class -- greenery +0.686 against
+0.720, sky +0.548 against +0.584, hardscape +0.178 against +0.388. Five of its
classes never fire at all (shrub_hedge, ground_vegetation, vertical_green_wall,
arcade_column, arcade_soffit: written into all 3,064 files with zero pixels).
It earns its place only for stoop_stair and bench_seating, which gave IAS its
first real twin. This run keeps the coverage and drops the cost.

ONE PROCESS, NOT ONE PER IMAGE. The taxonomy pipeline forks a subprocess per
image because its 29 notebook cells share module-level state and a leak would
be silent. These two models are stateless calls, so they load once and loop.

Shares are of the WHOLE frame, matching seg90_shares.csv, so the two tables can
sit side by side. Class names are prefixed map_ and ade_ because both datasets
have a "sky" and a "building" and merging them unprefixed would silently
collide.

    .venv-gpu/Scripts/python tools/seg_two_model.py
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import PROC, banner
from mast import mast_mask

MODELS = {
    "map": "facebook/mask2former-swin-large-mapillary-vistas-semantic",
    "ade": "facebook/mask2former-swin-large-ade-semantic",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/raw/svi_90"))
    ap.add_argument("--out", type=Path, default=PROC / "seg90_two_model.csv")
    ap.add_argument("--checkpoint", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--no-mast-mask", action="store_true",
                    help="count every pixel, including Google's camera mast")
    ap.add_argument("--mast-set", default=None,
                    help="mast calibration to use; defaults to the source "
                         "folder name, which is what names the calibration")
    args = ap.parse_args()
    banner("two-model segmentation: Mapillary Vistas + ADE20K")
    # The mast's share of the frame scales with the field of view, so the
    # calibration is per imagery set and the source folder names it. Guessing
    # would silently apply svi_90's single 14%-wide mast to an svi_180 frame
    # that has two at 7%.
    WIDE_SET = "svi_180"   # the 2880x1833 strip
    mset = args.mast_set or args.src.name
    if not args.no_mast_mask:
        print(f"mast calibration: {mset}")

    imgs = sorted(args.src.rglob("*.jpg"))
    rel = [str(p.relative_to(args.src)).replace("\\", "/") for p in imgs]
    done = pd.DataFrame()
    if args.out.exists():
        done = pd.read_csv(args.out)
        have = set(done.file)
        pairs = [(p, r) for p, r in zip(imgs, rel) if r not in have]
        print(f"resuming: {len(have)} already done")
    else:
        pairs = list(zip(imgs, rel))
    if args.limit:
        pairs = pairs[:args.limit]
    print(f"{len(pairs)} images x {len(MODELS)} models\n")
    if not pairs:
        print("nothing to do")
        return

    import torch
    from PIL import Image
    from tqdm.auto import tqdm
    from transformers import (AutoImageProcessor,
                              Mask2FormerForUniversalSegmentation)

    nets = {}
    for tag, name in MODELS.items():
        proc = AutoImageProcessor.from_pretrained(name)
        net = Mask2FormerForUniversalSegmentation.from_pretrained(name)
        net = net.to("cuda").eval()
        if args.fp16:
            net = net.half()
        lab = {int(i): v for i, v in net.config.id2label.items()}
        nets[tag] = (proc, net, lab)
        print(f"  {tag}: {len(lab)} classes  {name.split('/')[-1]}")
    print()

    out = []
    t0 = time.time()
    for img, r in tqdm(pairs, desc="images", mininterval=30.0):
        im = Image.open(img).convert("RGB")
        # The mast is excluded from the DENOMINATOR as well as the numerator:
        # it is not part of the street, so a share should be of the street.
        # per image, not per folder: the street-type split puts 90-degree
        # halves and 180-degree strips in one tree, and the mast's share of
        # the frame width differs between them
        mset_i = WIDE_SET if img.stem.endswith("_F") else mset
        keep = ~mast_mask(im, mset_i) if not args.no_mast_mask else None
        rec = {"file": r, "mast_share": 0.0 if keep is None
               else float(1.0 - keep.mean())}
        for tag, (proc, net, lab) in nets.items():
            inp = proc(images=im, return_tensors="pt").to("cuda")
            if args.fp16:
                inp["pixel_values"] = inp["pixel_values"].half()
            with torch.no_grad():
                o = net(**inp)
            seg = proc.post_process_semantic_segmentation(
                o, target_sizes=[im.size[::-1]])[0].cpu().numpy()
            sel = seg if keep is None else seg[keep]
            n = sel.size
            ids, cnt = np.unique(sel, return_counts=True)
            # every class gets a column, present or not: a zero share and an
            # absent column are different facts, and the four-model run's five
            # dead classes were only visible because they were written as zeros
            for i in lab:
                rec[f"{tag}_{lab[i]}"] = 0.0
            for i, c in zip(ids, cnt):
                rec[f"{tag}_{lab[int(i)]}"] = round(float(c) / n, 6)
        out.append(rec)
        if len(out) % args.checkpoint == 0:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            pd.concat([done, pd.DataFrame(out)], ignore_index=True).to_csv(
                args.out, index=False)

    d = pd.concat([done, pd.DataFrame(out)], ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(args.out, index=False)
    el = time.time() - t0
    print(f"\n{len(out)} images in {el/60:.1f} min ({el/max(len(out),1):.2f} s/image)")
    print(f"wrote {args.out}  ({d.shape[1]-1} class columns)\n")

    # what never fires is worth printing: the four-model run had five such
    # classes and nothing surfaced them until they were looked for.
    cols = [c for c in d.columns if c != "file"]
    dead = [c for c in cols if (d[c] > 0).sum() == 0]
    print(f"  {len(dead)} classes never fire:")
    for c in sorted(dead):
        print(f"    {c}")
    live = [(c, (d[c] > 0).mean() * 100, d[c].mean()) for c in cols
            if c not in dead]
    live.sort(key=lambda x: -x[2])
    print(f"\n  the 15 largest of {len(live)} live classes:")
    print(f"    {'class':<34}{'% images':>10}{'mean share':>12}")
    for c, pz, mn in live[:15]:
        print(f"    {c:<34}{pz:>9.1f}%{mn:>12.5f}")


if __name__ == "__main__":
    main()
