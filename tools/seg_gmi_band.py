"""GMI's twin: greenery on the lower 3 m of the facade, per node.

The manuscript defines GMI as greenery that "completely covers the lower 3
metres of the building facades", which is a distance in the WORLD, not a
fraction of the image. A fixed image band cannot express it: 3 m subtends about
14 degrees at a 23 m street and 8 degrees on Park Avenue at 43.6 m, so the same
band is too wide on the avenues and too narrow on the mid-blocks.

We already measure the distance. s05's band probe gives W_facade -- a
parallel-sided 20 m corridor, nine rays, reduced by the minimum -- and the
render is cylindrical, so the rows those 3 metres occupy are exact:

    d   = W_facade / 2                        distance to the facade
    phi = arctan((z - h_cam) / d)             elevation of a point z m up
    y   = H/2 - fc*tan(phi)                   the row it lands on
    fc  = W / radians(FOV)

Symmetric street assumed. The probe measures both sides and s05 keeps only
their sum, so each half-view is given W_facade/2. On an asymmetric section --
a setback on one side, Park Avenue's median -- the two halves' facades are at
different distances and this is wrong for both. Keeping the per-side reaches is
a small change to s05 if this version earns it.

W_facade exists for 584 of 764 nodes. The rest carry inherited H/W (radius
mean, series walk, open-one-side), and a band from a borrowed width would be a
guess dressed as a measurement, so `hw_source` rides along and those rows can
be dropped. The 33 open-one-side nodes arguably have no GMI at all rather than
zero: with no opposite wall there is nothing for greenery to soften, which is
the same geometric fact that gives them Omega = 1.0.

    .venv-gpu/Scripts/python tools/seg_gmi_band.py --n 500
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import PROC, RES, banner
from mast import mast_mask

MODEL = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
# Street View's camera sits on the vehicle roof, not at a pedestrian's eye.
# The horizon in the frame is at the CAPTURE height, so that is what converts
# metres to rows -- the appendix's 1.5 m is the protocol's ideal, not this rig.
H_CAM = 2.5
FACADE_M = 3.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/raw/svi_90"))
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=19)
    ap.add_argument("--out", type=Path, default=PROC / "seg90_gmi_band.csv")
    args = ap.parse_args()
    banner("GMI twin: greenery on the lower 3 m of facade")

    # vlm_observations.csv already carries W_facade and HW_source from the
    # export; merging metrics.csv in again suffixes both away to _x/_y.
    o = pd.read_csv(RES / "tables" / "vlm_observations.csv")
    o = o[[(args.src / f).exists() for f in o.file]]
    have = o.dropna(subset=["W_facade"])
    print(f"{len(have)} of {len(o)} half-views have a measured W_facade")
    take = have.sample(min(args.n, len(have)), random_state=args.seed)
    print(f"{len(take)} sampled\n")

    import torch
    from PIL import Image
    from tqdm.auto import tqdm
    from transformers import (AutoImageProcessor,
                              Mask2FormerForUniversalSegmentation)
    proc = AutoImageProcessor.from_pretrained(MODEL)
    net = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL)
    net = net.to("cuda").eval().half()
    inv = {v: int(k) for k, v in net.config.id2label.items()}
    VEG, BLD = inv["Vegetation"], inv["Building"]

    rows = []
    for r in tqdm(list(take.itertuples()), desc="frames", mininterval=20.0):
        im = Image.open(args.src / r.file).convert("RGB")
        W, H = im.size
        inp = proc(images=im, return_tensors="pt").to("cuda")
        inp["pixel_values"] = inp["pixel_values"].half()
        with torch.no_grad():
            out = net(**inp)
        seg = proc.post_process_semantic_segmentation(
            out, target_sizes=[(H, W)])[0].cpu().numpy()
        keep = ~mast_mask(im, args.src.name)
        veg, bld = (seg == VEG) & keep, (seg == BLD) & keep

        d = float(r.W_facade) / 2.0
        fc = W / np.radians(90.0)
        row = lambda z: int(np.clip(
            H / 2 - fc * np.tan(np.arctan((z - H_CAM) / d)), 0, H))
        y_top, y_bot = row(FACADE_M), row(0.0)      # 3 m up, and ground
        near = ndimage.binary_dilation(bld, iterations=max(1, int(0.015 * W) // 3))

        band = np.zeros_like(keep)
        band[y_top:y_bot] = True
        denom = max((keep & band).sum(), 1)
        rows.append(dict(
            file=r.file, node_id=r.node_id, W_facade=r.W_facade,
            hw_source=r.HW_source, d=d,
            band_deg_top=float(np.degrees(np.arctan((FACADE_M - H_CAM) / d))),
            band_deg_bot=float(np.degrees(np.arctan((0.0 - H_CAM) / d))),
            band_rows=int(y_bot - y_top),
            # the interaction: greenery, against a facade, in the lower 3 m
            gmi_band=float((veg & near & band).sum() / denom),
            # each ingredient alone, to show what the interaction adds
            veg_band=float((veg & band).sum() / denom),
            veg_all=float(veg.sum() / max(keep.sum(), 1)),
            veg_near=float((veg & near).sum() / max(keep.sum(), 1))))
    d = pd.DataFrame(rows)
    d.to_csv(args.out, index=False)

    from scipy.stats import spearmanr
    j = d.merge(o[["file", "green_softening", "vertical_greenery",
                   "vertical_hardscape"]], on="file")
    print(f"\n  band geometry: {d.band_deg_top.mean():+.1f} to "
          f"{d.band_deg_bot.mean():+.1f} deg, {d.band_rows.mean():.0f} rows "
          f"(varies {d.band_rows.min()}-{d.band_rows.max()})\n")
    print(f"    {'candidate':<16}{'vs green_softening':>20}{'vs vertical_greenery':>22}")
    for c in ("gmi_band", "veg_band", "veg_near", "veg_all"):
        print(f"    {c:<16}{spearmanr(j[c], j.green_softening).statistic:>20.3f}"
              f"{spearmanr(j[c], j.vertical_greenery).statistic:>22.3f}")
    print(f"\n  a GMI twin has to beat veg_all on green_softening AND be less")
    print(f"  aligned with vertical_greenery, or it is just greenery again.")
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
