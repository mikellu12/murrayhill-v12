"""A walk down one street twice: as photographed, and as segmented.

Two GIFs from the same frames in the same order, so the pair answers the
question a reviewer asks about any segmentation-backed measure -- does the
mask actually land on the thing it is named after -- by letting the eye check
it along a whole street rather than on one cherry-picked frame.

Classes are drawn from the model the study scores each field with, and in the
same paint order as tools/seg_combined_render.py: sparse specific roles last,
so a bench survives the pedestrian area it stands on.
"""
import argparse, re, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src")); sys.path.insert(0, str(HERE))
from common import RAW, RES, banner
from mast import mast_mask

NAME = re.compile(r"^(\d+)_(n\d+)_([NESW])_([LRF])\.jpg$")
SEAM = (18, 18, 20)
MODELS = {"map": "facebook/mask2former-swin-large-mapillary-vistas-semantic",
          "ade": "facebook/mask2former-swin-large-ade-semantic"}
ROLES = [("sky", [("map", "Sky")], "#5aa9e6"),
         ("hardscape", [("map", "Building"), ("map", "Wall")], "#8c8177"),
         ("greenery", [("map", "Vegetation")], "#5fbf6a"),
         ("walkable", [("map", "Sidewalk"), ("map", "Curb"),
                       ("map", "Curb Cut"), ("map", "Pedestrian Area")], "#c9a227"),
         ("glazing", [("ade", "windowpane"), ("ade", "door")], "#d2704f"),
         ("signage", [("map", "Billboard")], "#e05780"),
         ("resting", [("map", "Bench"), ("ade", "bench"),
                      ("ade", "stairs"), ("ade", "step")], "#b07de0")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--street", required=True)
    ap.add_argument("--walk", default=None)
    ap.add_argument("--width", type=int, default=760)
    ap.add_argument("--ms", type=int, default=280)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    banner(f"walk gifs: {args.street}")

    base = RAW / "svi_90" / args.street
    walks = [base / args.walk] if args.walk else sorted(
        p for p in base.iterdir() if p.is_dir())
    w = walks[0]
    got = {}
    for p in sorted(w.glob("*.jpg")):
        m = NAME.match(p.name)
        if m:
            got.setdefault((int(m.group(1)), m.group(2)), {})[m.group(4)] = p
    nodes = [got[k] for k in sorted(got)]
    if args.limit:
        nodes = nodes[:args.limit]
    print(f"{w.name}: {len(nodes)} nodes")

    import torch
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
    nets = {}
    for tag, name in MODELS.items():
        pr = AutoImageProcessor.from_pretrained(name)
        nt = Mask2FormerForUniversalSegmentation.from_pretrained(name).to("cuda").eval().half()
        nets[tag] = (pr, nt, {v: k for k, v in nt.config.id2label.items()})
        print(f"  {tag} loaded")

    def overlay(path):
        im = Image.open(path).convert("RGB")
        W, H = im.size
        drop = mast_mask(im, "svi_90_wide" if path.name.endswith("_F.jpg") else "svi_90")
        segs = {}
        for tag, (pr, nt, _) in nets.items():
            inp = pr(images=im, return_tensors="pt").to("cuda")
            inp["pixel_values"] = inp["pixel_values"].half()
            with torch.no_grad():
                o = nt(**inp)
            segs[tag] = pr.post_process_semantic_segmentation(o, target_sizes=[(H, W)])[0].cpu().numpy()
        rgb = np.asarray(im).astype(float); out = rgb.copy()
        for role, members, colour in ROLES:
            m = np.zeros((H, W), bool)
            for tag, cls in members:
                cid = nets[tag][2].get(cls)
                if cid is not None:
                    m |= segs[tag] == cid
            m &= ~drop
            c = np.array([int(colour[i:i+2], 16) for i in (1, 3, 5)], float)
            out[m] = (1 - args.alpha) * rgb[m] + args.alpha * c
        out[drop] = 0.18 * rgb[drop]
        return im, Image.fromarray(out.astype(np.uint8))

    def pair(a, b, width):
        half = (width - 3) // 2
        ims = [x.resize((half, int(half * x.height / x.width)), Image.LANCZOS) for x in (a, b)]
        o = Image.new("RGB", (width, ims[0].height), SEAM)
        o.paste(ims[0], (0, 0)); o.paste(ims[1], (half + 3, 0))
        return o

    raw, seg = [], []
    from tqdm.auto import tqdm
    for i, sides in enumerate(tqdm(nodes, desc="frames"), 1):
        if "F" in sides:
            p0, s0 = overlay(sides["F"])
            r = p0.resize((args.width, int(args.width * p0.height / p0.width)), Image.LANCZOS)
            s = s0.resize(r.size, Image.LANCZOS)
        else:
            pl, sl = overlay(sides["L"]); prr, sr = overlay(sides["R"])
            r = pair(pl, prr, args.width); s = pair(sl, sr, args.width)
        for img, tag in ((r, "street view"), (s, "segmented")):
            d = ImageDraw.Draw(img)
            d.rectangle([0, img.height - 20, img.width, img.height], fill=SEAM)
            d.text((6, img.height - 15),
                   f"{args.street.replace('_',' ')}  {w.name.replace('_',' ')}"
                   f"  {i}/{len(nodes)}  {tag}", fill=(232, 230, 225))
        raw.append(r); seg.append(s)

    out = RES / "figures" / "walks"; out.mkdir(parents=True, exist_ok=True)
    for imgs, tag in ((raw, "raw"), (seg, "segmented")):
        o = out / f"{args.street}__{w.name}__{tag}.gif"
        imgs[0].save(o, save_all=True, append_images=imgs[1:],
                     duration=args.ms, loop=0, optimize=True)
        print(f"  wrote {o.name}  {o.stat().st_size/1048576:.1f} MB")


if __name__ == "__main__":
    main()
