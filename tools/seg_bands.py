"""Vegetation share by elevation band, so the three green fields stop sharing one twin.

vertical_greenery, green_eye_level and green_softening were all validated
against whole-frame vegetation, which is why they all score around +0.70 and
why green_softening looks validated when it is really just tracking greenery:
the three correlate +0.72 to +0.84 with each other. They measure different
REGIONS of the same frame, and the twin should too.

  vertical_greenery   canopy and green facade      -> above the horizon
  green_eye_level     greenery at eye height       -> 0-15 deg below horizon,
                                                      which is the manuscript
                                                      appendix's own definition
                                                      of the foveal cone
  green_softening     greenery against the facade  -> vegetation adjacent to
                                                      building pixels

The band edges are exact rather than fractions of frame height, because the
render is cylindrical: phi = arctan((H/2 - y) / fc) with fc = out_w /
radians(FOV). Row H/2 is the horizon by construction, and 15 degrees below it
is fc*tan(15) = 246 px for svi_90. Using a percentage of frame height instead
would put the boundary in the wrong place and differently in each imagery set.

The mast is excluded from numerator and denominator both, as in
seg_two_model.py: a share should be of the street.

    .venv-gpu/Scripts/python tools/seg_bands.py --src data/raw/svi_90
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import PROC, banner
from mast import mast_mask

MODEL = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
# horizontal span of each render; sets fc, and so where the horizon and the
# 15-degree lines fall
FOV = {"svi_90": 90.0, "svi_180": 180.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/raw/svi_90"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--checkpoint", type=int, default=200)
    args = ap.parse_args()
    tag = args.src.name
    out = args.out or PROC / f"{tag.replace('svi_', 'seg')}_bands.csv"
    banner(f"vegetation by elevation band -- {tag}")

    imgs = sorted(args.src.rglob("*.jpg"))
    rel = [str(p.relative_to(args.src)).replace("\\", "/") for p in imgs]
    done = pd.DataFrame()
    if out.exists():
        done = pd.read_csv(out)
        have = set(done.file)
        pairs = [(p, r) for p, r in zip(imgs, rel) if r not in have]
        print(f"resuming: {len(have)} done")
    else:
        pairs = list(zip(imgs, rel))
    if args.limit:
        pairs = pairs[:args.limit]
    print(f"{len(pairs)} images\n")
    if not pairs:
        print("nothing to do")
        return

    import torch
    from PIL import Image
    from tqdm.auto import tqdm
    from transformers import (AutoImageProcessor,
                              Mask2FormerForUniversalSegmentation)
    proc = AutoImageProcessor.from_pretrained(MODEL)
    net = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL)
    net = net.to("cuda").eval().half()
    inv = {v: int(k) for k, v in net.config.id2label.items()}
    VEG, BLD, SKY = inv["Vegetation"], inv["Building"], inv["Sky"]
    print(f"  {MODEL.split('/')[-1]}")
    print(f"  bands from phi = arctan((H/2 - y)/fc), fc = W/radians("
          f"{FOV.get(tag, 90.0):g})\n")

    rows = []
    t0 = time.time()
    for img, r in tqdm(pairs, desc="images", mininterval=30.0):
        im = Image.open(img).convert("RGB")
        W, H = im.size
        inp = proc(images=im, return_tensors="pt").to("cuda")
        inp["pixel_values"] = inp["pixel_values"].half()
        with torch.no_grad():
            o = net(**inp)
        seg = proc.post_process_semantic_segmentation(
            o, target_sizes=[(H, W)])[0].cpu().numpy()
        keep = ~mast_mask(im, tag)
        veg, bld = (seg == VEG) & keep, (seg == BLD) & keep

        fc = W / np.radians(FOV.get(tag, 90.0))
        row = lambda deg: int(np.clip(H / 2 - fc * np.tan(np.radians(deg)), 0, H))
        bands = {"above15": (0, row(15)),
                 "up0_15": (row(15), row(0)),
                 "eye0_15": (row(0), row(-15)),
                 "below15": (row(-15), H)}
        rec = {"file": r, "mast_share": float(1.0 - keep.mean())}
        for k, (a, b) in bands.items():
            denom = keep[a:b].sum()
            rec[f"veg_{k}"] = round(float(veg[a:b].sum() / max(denom, 1)), 6)
        rec["veg_all"] = round(float(veg.sum() / max(keep.sum(), 1)), 6)
        rec["veg_above_horizon"] = round(
            float(veg[:row(0)].sum() / max(keep[:row(0)].sum(), 1)), 6)
        # greenery touching a facade: the interaction GMI is defined as, rather
        # than greenery anywhere in a frame that happens to contain a building
        near = ndimage.binary_dilation(bld, iterations=max(1, int(0.015 * W) // 3))
        rec["veg_on_bld"] = round(float((veg & near).sum() / max(keep.sum(), 1)), 6)
        rec["bld_all"] = round(float(bld.sum() / max(keep.sum(), 1)), 6)
        rec["sky_all"] = round(float(((seg == SKY) & keep).sum()
                                     / max(keep.sum(), 1)), 6)
        rows.append(rec)
        if len(rows) % args.checkpoint == 0:
            out.parent.mkdir(parents=True, exist_ok=True)
            pd.concat([done, pd.DataFrame(rows)], ignore_index=True).to_csv(
                out, index=False)

    d = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out, index=False)
    el = time.time() - t0
    print(f"\n{len(rows)} images in {el/60:.1f} min ({el/max(len(rows),1):.2f} s/image)")
    print(f"wrote {out}\n")
    cols = [c for c in d.columns if c.startswith(("veg_", "bld_", "sky_"))]
    print(f"    {'band':<22}{'mean share':>12}{'% frames >0':>14}")
    for c in cols:
        print(f"    {c:<22}{d[c].mean()*100:>11.2f}%{(d[c] > 0).mean()*100:>13.1f}%")


if __name__ == "__main__":
    main()
