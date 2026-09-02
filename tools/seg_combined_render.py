"""One overlay per frame, each class drawn from the model the study uses for it.

seg_two_model_render.py shows Mapillary and ADE20K as separate panels, which is
right for judging the segmenters and wrong for seeing the measurement: no field
is scored from a whole panel. The twin table in README pairs each field with
specific classes from a specific model, and this draws exactly those, layered
into a single image.

WHICH MODEL SUPPLIES WHAT IS NOT A PREFERENCE. Mapillary Vistas is a
street-level dataset and carries Vegetation, Sky, Building, Wall, Billboard,
Sidewalk, Curb, Curb Cut, Pedestrian Area and Bench. ADE20K has none of the
kerb vocabulary but does have stairs, step, windowpane and door, which
Mapillary lacks entirely -- and those are the only handles on resting
affordance's stair component and on ground-floor activity. So the split follows
the classes that exist, not a judgement about which model is better.

PAINT ORDER IS DELIBERATE. A pixel can satisfy more than one role: a bench sits
on a pedestrian area, a billboard hangs on a building. Sparse, specific roles
are painted last so they survive; sky and hardscape are the ground layers. Any
other order hides exactly the classes that matter, because they are the small
ones.

    .venv-gpu/Scripts/python tools/seg_combined_render.py --n 6
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import banner
from mast import mast_mask

MODELS = {"map": "facebook/mask2former-swin-large-mapillary-vistas-semantic",
          "ade": "facebook/mask2former-swin-large-ade-semantic"}
BG, FG = "#0e0f12", "#e8e6e1"

# role, the field it feeds, [(model, class), ...], colour.
# Ground layers first, specific ones last -- see the paint-order note above.
ROLES = [
    ("sky",        "sky_openness",          [("map", "Sky")],              "#5aa9e6"),
    ("hardscape",  "vertical_hardscape",    [("map", "Building"),
                                             ("map", "Wall")],             "#8c8177"),
    ("greenery",   "vertical_greenery / GVI", [("map", "Vegetation")],     "#5fbf6a"),
    ("walkable",   "walkable_ground",       [("map", "Sidewalk"),
                                             ("map", "Curb"),
                                             ("map", "Curb Cut"),
                                             ("map", "Pedestrian Area")],  "#c9a227"),
    ("glazing",    "ground_floor_activity", [("ade", "windowpane"),
                                             ("ade", "door")],             "#d2704f"),
    ("signage",    "signage_detail",        [("map", "Billboard")],        "#e05780"),
    ("resting",    "resting_affordance",    [("map", "Bench"),
                                             ("ade", "bench"),
                                             ("ade", "stairs"),
                                             ("ade", "step")],             "#b07de0"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/raw/svi_90"))
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.55)
    ap.add_argument("--files", nargs="+", default=None)
    ap.add_argument("--mast-set", default=None)
    ap.add_argument("--out", type=Path,
                    default=Path("results/figures/seg_combined.png"))
    args = ap.parse_args()
    banner("combined overlay: each class from the model the study uses")

    files = args.files
    if not files:
        pool = sorted(str(p.relative_to(args.src)).replace("\\", "/")
                      for p in args.src.rglob("*.jpg"))
        rng = np.random.default_rng(args.seed)
        files = [pool[i] for i in sorted(rng.choice(len(pool), args.n, replace=False))]
    print(f"{len(files)} frame(s)")

    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    nets = {}
    for tag, name in MODELS.items():
        proc = AutoImageProcessor.from_pretrained(name)
        net = Mask2FormerForUniversalSegmentation.from_pretrained(name)
        net = net.to("cuda").eval().half()
        lab = {int(i): v for i, v in net.config.id2label.items()}
        nets[tag] = (proc, net, {v: k for k, v in lab.items()})
        print(f"  {tag}: {len(lab)} classes")

    n = len(files)
    # The legend needs real width -- its class lists run long and were being
    # clipped mid-word -- so the panels stop at .62 and the key gets the rest.
    fig, axes = plt.subplots(n, 2, figsize=(13.4, 5.1 * n), facecolor=BG,
                             gridspec_kw=dict(hspace=.05, wspace=.010,
                                              left=.004, right=.615,
                                              top=.972, bottom=.008))
    if n == 1:
        axes = axes[None, :]

    for r, f in enumerate(files):
        im = Image.open(args.src / f).convert("RGB")
        W, H = im.size
        drop = mast_mask(im, "svi_180" if str(f).endswith("_F.jpg")
                         else (args.mast_set or args.src.name))
        segs = {}
        for tag, (proc, net, name2id) in nets.items():
            inp = proc(images=im, return_tensors="pt").to("cuda")
            inp["pixel_values"] = inp["pixel_values"].half()
            with torch.no_grad():
                o = net(**inp)
            segs[tag] = proc.post_process_semantic_segmentation(
                o, target_sizes=[(H, W)])[0].cpu().numpy()

        rgb = np.asarray(im).astype(float)
        over = rgb.copy()
        shares = {}
        for role, field, members, colour in ROLES:
            m = np.zeros((H, W), bool)
            for tag, cls in members:
                cid = nets[tag][2].get(cls)
                if cid is not None:
                    m |= segs[tag] == cid
            m &= ~drop
            shares[role] = m.mean()
            c = np.array([int(colour[i:i + 2], 16) for i in (1, 3, 5)], float)
            over[m] = (1 - args.alpha) * rgb[m] + args.alpha * c
        over[drop] = 0.15 * rgb[drop]           # mast dimmed, not painted

        axes[r, 0].imshow(np.asarray(im))
        axes[r, 1].imshow(over.astype(np.uint8))
        for c in (0, 1):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            for s in axes[r, c].spines.values():
                s.set_color("#23262b")
        axes[r, 0].set_ylabel(Path(f).name, color="#9a9aa2", fontsize=7.5)
        txt = "   ".join(f"{k} {v*100:.1f}%" for k, v in shares.items() if v > .005)
        axes[r, 1].set_title(txt, color="#9a9aa2", fontsize=7.5, pad=3)

    handles = [mpatches.Patch(color=c, label=f"{role}  ->  {field}\n"
                              + ", ".join(f"{t}_{n}" for t, n in mem))
               for role, field, mem, c in ROLES]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(.725, .5),
               facecolor="#15171b", edgecolor="#23262b", labelcolor=FG,
               fontsize=8.5, borderpad=1.1, labelspacing=1.25)
    fig.text(.004, .988, "photograph  |  what the segmenters call it. Each "
             "class is drawn from the model the study scores that field with; "
             "the Google mast is dimmed, not classified.",
             color=FG, fontsize=10.5, va="top")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=155, facecolor=BG)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
