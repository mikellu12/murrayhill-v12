"""Does the VLM see Google's camera mast as part of the street?

The segmenters do: masking the mast removes 47% of Mapillary's Pole pixels,
60% of ADE20K's pole, and 29% of ADE20K's signboard -- the last being the
watermark read as a sign. Whether the VLM is fooled the same way is a separate
question, because it is answering "how much signage is there" rather than
labelling pixels, and a rating may be robust to an object a segmenter
miscounts.

Each frame is rated twice: as captured, and with the mast painted out in the
median colour of the surrounding band. Same model, same prompt, same seed --
the only difference is roughly 2% of the frame. If the ratings move, the VLM
is reading Google's hardware as streetscape.

Three fields, chosen because they are the ones the segmentation evidence
implicates: signage_detail (the watermark), walkable_ground (the mast stands
in the road/sidewalk zone) and vertical_hardscape (it is a dark vertical
surface, which is what that field counts).

Painted, not cropped. Cropping changes the framing and the field of view,
which would confound the thing being measured; filling with the local median
leaves the geometry identical.

    .venv-gpu/Scripts/python tools/vlm_mast_probe.py --n 50
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import wilcoxon

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import RES, banner
from sim_fields import SYSTEM
from sim_scale import prompt7

MODEL = "Qwen/Qwen2-VL-7B-Instruct"
FIELDS = ["signage_detail", "walkable_ground", "vertical_hardscape"]


def mast(im, band=0.22, dark=0.55, grow=2.6, cap=0.05):
    """Bottom-anchored dark column in the lower band, widened to the whole part.

    Model-free on purpose: the mast is labelled Pole by Mapillary and signboard
    by ADE20K, so keying off either model's classes misses what the other one
    called it. `cap` refuses a runaway mask -- off-centre instances sometimes
    grow to 19% of the frame, which would erase real street rather than the
    camera.
    """
    a = np.asarray(im.convert("L"), float) / 255.0
    H, W = a.shape
    lo = int(H * (1 - band))
    sub = a[lo:]
    L, n = ndimage.label(sub < np.median(sub) * dark)
    boxes = []
    for k in range(1, n + 1):
        ys, xs = np.where(L == k)
        if len(ys) < 0.0004 * sub.size:
            continue
        tall = (ys.max() - ys.min() + 1) / sub.shape[0] > 0.45
        narrow = (xs.max() - xs.min() + 1) / W < 0.25
        if ys.max() >= sub.shape[0] - 2 and tall and narrow:
            boxes.append((xs.min(), xs.max(), ys.min()))
    m = np.zeros((H, W), bool)
    if not boxes:
        return m
    x0 = min(b[0] for b in boxes)
    x1 = max(b[1] for b in boxes)
    yt = min(b[2] for b in boxes)
    xa = max(0, int(x0 - 0.004 * W))
    xb = min(W, int(xa + (x1 - x0 + 1) * grow))
    m[lo + yt:, xa:xb] = True
    return m if m.mean() <= cap else np.zeros((H, W), bool)


def erase(im):
    a = np.asarray(im.convert("RGB")).copy()
    m = mast(im)
    if not m.any():
        return im, False
    from PIL import Image
    band = a[int(a.shape[0] * 0.80):]
    a[m] = np.median(band.reshape(-1, 3), axis=0).astype(np.uint8)
    return Image.fromarray(a), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=57)
    args = ap.parse_args()
    banner("does the VLM read the camera mast as streetscape?")

    import torch
    from PIL import Image
    from tqdm.auto import tqdm
    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              Qwen2VLForConditionalGeneration)
    o = pd.read_csv(RES / "tables" / "vlm_observations.csv")
    src = Path("data/raw/svi_90")
    o = o[[(src / f).exists() for f in o.file]]
    take = o.sample(args.n, random_state=args.seed)

    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.bfloat16,
                           bnb_4bit_use_double_quant=True)
    proc = AutoProcessor.from_pretrained(MODEL, max_pixels=768 * 768)
    proc.tokenizer.padding_side = "left"
    net = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL, quantization_config=q, device_map="cuda").eval()
    tok = proc.tokenizer
    ids = [tok.encode(str(k), add_special_tokens=False)[0] for k in range(1, 8)]
    ks = np.arange(1, 8)

    def rate(field, im):
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt7(field)}]}]
        t = proc.apply_chat_template(msgs, tokenize=False,
                                     add_generation_prompt=True)
        inp = proc(text=[t + '{"' + field + '": '], images=[im],
                   return_tensors="pt").to("cuda")
        with torch.no_grad():
            lg = net(**inp).logits[:, -1, :].float()
        p = torch.softmax(lg[0, ids], -1).cpu().numpy()
        return float(((p / p.sum()) * ks).sum())

    rows, skipped = [], 0
    for r in tqdm(list(take.itertuples()), desc="frames", mininterval=20.0):
        im = Image.open(src / r.file).convert("RGB")
        cln, found = erase(im)
        if not found:
            skipped += 1
            continue
        rec = {"file": r.file}
        for f in FIELDS:
            rec[f + "_raw"] = rate(f, im)
            rec[f + "_clean"] = rate(f, cln)
        rows.append(rec)

    d = pd.DataFrame(rows)
    out = RES / "tables" / "vlm_mast_probe.csv"
    d.to_csv(out, index=False)
    print(f"\n  n={len(d)} frames rated both ways ({skipped} had no mast found)\n")
    print(f"    {'field':<22}{'with mast':>11}{'erased':>10}{'mean shift':>12}{'p':>9}")
    for f in FIELDS:
        a, b = d[f + "_raw"], d[f + "_clean"]
        p = wilcoxon(a, b).pvalue if (a - b).abs().sum() > 0 else 1.0
        print(f"    {f:<22}{a.mean():>11.3f}{b.mean():>10.3f}"
              f"{(b - a).mean():>+12.3f}{p:>9.4f}")
    print(f"\n    {'field':<22}{'moved >=0.25 rung':>20}{'max move':>11}")
    for f in FIELDS:
        dif = d[f + "_clean"] - d[f + "_raw"]
        print(f"    {f:<22}{(dif.abs() >= 0.25).mean()*100:>19.0f}%"
              f"{dif.abs().max():>11.3f}")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
