"""Render what the two segmenters actually see, side by side with the photo.

seg_two_model.py writes shares and no pixels, which is enough to score a twin
and useless for judging one. A share of 0.046 for ground_floor_glazing and a
share of 0.00002 for ade_windowpane are both just numbers until you look at
where they land.

Three panels per image: the source frame, Mapillary Vistas, ADE20K. Colours
come from a fixed hash of the class name, so the same class is the same colour
in every image and across runs -- a per-image palette would make two frames
incomparable, which is the one thing this figure exists to allow.

Only classes above `--min-share` are drawn and named. A Manhattan street view
has fifty classes at trace level and labelling them all produces a legend
nobody reads.

    .venv-gpu/Scripts/python tools/seg_two_model_render.py --n 10
"""
import argparse
import hashlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import PROC, RES, banner

MODELS = {
    "Mapillary Vistas": "facebook/mask2former-swin-large-mapillary-vistas-semantic",
    "ADE20K": "facebook/mask2former-swin-large-ade-semantic",
}
BG, FG, DIM = "#0d1117", "#e6edf3", "#8b949e"


def colour(name):
    """Deterministic per-class colour: same class, same colour, every run."""
    h = hashlib.md5(name.encode()).digest()
    # bias away from very dark: the ground is dark and a near-black class
    # would read as unlabelled rather than as a class
    return tuple(0.25 + 0.72 * (b / 255) for b in h[:3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/raw/svi_90"))
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--min-share", type=float, default=0.015)
    ap.add_argument("--out", type=Path,
                    default=RES / "figures" / "seg_two_model_samples.png")
    args = ap.parse_args()
    banner("render the two segmenters against the photo")

    # stratify by street so ten samples are not ten frames of one avenue
    obs = pd.read_csv(RES / "tables" / "vlm_observations.csv")
    obs = obs[[(args.src / f).exists() for f in obs.file]]
    per = max(1, args.n // max(obs.osm_name.nunique(), 1))
    take = pd.concat([g.sample(min(len(g), per), random_state=args.seed)
                      for _, g in obs.groupby("osm_name")])
    take = take.sample(min(args.n, len(take)), random_state=args.seed)
    files = list(take.file)
    print(f"{len(files)} frames across {take.osm_name.nunique()} streets\n")

    import torch
    from PIL import Image
    from transformers import (AutoImageProcessor,
                              Mask2FormerForUniversalSegmentation)
    nets = {}
    for tag, name in MODELS.items():
        proc = AutoImageProcessor.from_pretrained(name)
        net = Mask2FormerForUniversalSegmentation.from_pretrained(name)
        nets[tag] = (proc, net.to("cuda").eval().half(),
                     {int(i): v for i, v in net.config.id2label.items()})
        print(f"  {tag}: {len(nets[tag][2])} classes")

    n = len(files)
    # svi_90 frames are portrait (1440x1833). A row shorter than the panel
    # width * 1.27 makes matplotlib fit by height and the image shrinks to a
    # sliver, which is what the first version did.
    w = 5.1
    fig, axes = plt.subplots(n, 3, figsize=(3 * w + 0.6, (w * 1.30) * n),
                             facecolor=BG,
                             gridspec_kw=dict(hspace=.10, wspace=.015))
    if n == 1:
        axes = axes[None, :]
    for r, f in enumerate(files):
        im = Image.open(args.src / f).convert("RGB")
        axes[r, 0].imshow(im)
        axes[r, 0].set_title(f, color=FG, fontsize=9, loc="left", pad=6)
        for c, (tag, (proc, net, lab)) in enumerate(nets.items(), start=1):
            inp = proc(images=im, return_tensors="pt").to("cuda")
            inp["pixel_values"] = inp["pixel_values"].half()
            with torch.no_grad():
                o = net(**inp)
            seg = proc.post_process_semantic_segmentation(
                o, target_sizes=[im.size[::-1]])[0].cpu().numpy()
            ids, cnt = np.unique(seg, return_counts=True)
            share = {int(i): c_ / seg.size for i, c_ in zip(ids, cnt)}
            rgb = np.zeros(seg.shape + (3,), float)
            named = []
            for i, s in sorted(share.items(), key=lambda kv: -kv[1]):
                if s < args.min_share:
                    continue
                rgb[seg == i] = colour(lab[i])
                named.append((lab[i], s))
            axes[r, c].imshow(rgb)
            # wrapped, not truncated: the two titles collided in the first
            # version and the ADE20K label overprinted Mapillary's.
            body = "  ".join(f"{k} {v*100:.0f}%" for k, v in named[:5])
            axes[r, c].set_title(tag + "\n" + body, color=DIM, fontsize=8,
                                 loc="left", pad=6)
        for a in axes[r]:
            a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(False)
    fig.suptitle("source frame  |  Mapillary Vistas  |  ADE20K        "
                 f"classes above {args.min_share*100:.1f}% of the frame are "
                 "coloured and named",
                 color=FG, fontsize=15, y=0.998)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=96, bbox_inches="tight", facecolor=BG)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
