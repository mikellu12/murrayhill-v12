"""How much does the place named in the prompt move the ratings?

The prompts said "Rate this Manhattan street view" for every study area. That
is the only context the model gets beyond the pixels, so naming the wrong city
is not cosmetic -- it tells the model to expect a streetscape it is not looking
at. The question is whether it moves the numbers enough to justify rerating.

Same images, same fields, same seed, one prompt word different.

    .venv-gpu/Scripts/python tools/prompt_place_ab.py --n 40
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
import sim_scale
from sim_fields import FIELDS, SYSTEM
from mast import erase_mast
from sim_readout import prune_once, interpolated_median

ORDER = list(FIELDS)
MODEL = "Qwen/Qwen2-VL-7B-Instruct"
MAX_PIXELS = 1024 * 28 * 28


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/london/raw/svi_90"))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--places", nargs=2, default=["Manhattan", "London"])
    args = ap.parse_args()

    files = sorted(args.src.rglob("*_[LR].jpg"))
    rng = np.random.default_rng(args.seed)
    files = [files[i] for i in sorted(rng.choice(len(files), args.n, replace=False))]

    import torch
    from PIL import Image
    from tqdm.auto import tqdm
    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              Qwen2VLForConditionalGeneration)

    # Same quantisation as the rating pass, so the comparison is against the
    # numbers the study actually produces and not a different regime.
    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True)
    proc = AutoProcessor.from_pretrained(MODEL, max_pixels=MAX_PIXELS)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL, quantization_config=qcfg, device_map="cuda").eval()
    tok = proc.tokenizer
    dig = [tok.encode(str(k), add_special_tokens=False)[0] for k in range(1, 8)]

    ims = [erase_mast(Image.open(f).convert("RGB"), "svi_90")[0] for f in files]
    res = {}
    for place in args.places:
        sim_scale.PLACE = place                 # the one word under test
        texts = []
        for f in ORDER:
            msg = [{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": [{"type": "image"},
                                                {"type": "text",
                                                 "text": sim_scale.prompt7(f)}]}]
            texts.append(proc.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True)
                + f'{{"{f}": ')
        rows = []
        for im in tqdm(ims, desc=place, mininterval=5.0):
            inp = proc(text=texts, images=[im] * len(ORDER), padding=True,
                       return_tensors="pt").to("cuda")
            with torch.no_grad():
                lg = model(**inp).logits[:, -1, :].float()
            rec = {}
            for i, f in enumerate(ORDER):
                p = torch.softmax(lg[i, dig], -1).cpu().numpy()
                rec[f] = p / p.sum()
            rows.append(rec)
        res[place] = rows

    a, b = args.places
    print(f"\n  {len(ims)} London frames, prompt says {a!r} vs {b!r}\n")
    print(f"  {'field':<24}{a:>12}{b:>12}{'shift':>9}{'|shift|':>9}{'flip':>7}")
    shifts = []
    for f in ORDER:
        PA = np.array([r[f] for r in res[a]])
        PB = np.array([r[f] for r in res[b]])
        va = interpolated_median(prune_once(PA))
        vb = interpolated_median(prune_once(PB))
        flip = (PA.argmax(1) != PB.argmax(1)).mean() * 100
        shifts.append(np.abs(vb - va).mean())
        print(f"  {f:<24}{va.mean():>12.2f}{vb.mean():>12.2f}"
              f"{(vb-va).mean():>9.2f}{np.abs(vb-va).mean():>9.2f}{flip:>6.0f}%")
    print(f"\n  mean |shift| across fields: {np.mean(shifts):.3f} of a rung")
    print(f"  a rung is 1/6 of the normalised range, so this is "
          f"{np.mean(shifts)/6*100:.1f}% of the 0-1 scale each field feeds M")


if __name__ == "__main__":
    main()
