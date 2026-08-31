"""Ask the model like a survey: many respondents, one number each.

A survey gets one answer per respondent and reports the distribution. Greedy
decoding gives one answer, always the same, so it is a survey of one person
asked repeatedly. Sampling at temperature 1 draws a fresh answer each time,
which is the closest thing to a panel: N respondents drawn from the same head.

The point of this probe is that the panel and the logit read give the same
answer. Sampling N times and averaging is a Monte Carlo estimate of
sum(p_k * k) -- it converges to the expected value, with error falling as
1/sqrt(N). The logits give that number exactly, in one forward pass, for less
compute than a single sample.

So "make it pick a number like a survey" is already what the ratings are; the
question is only whether to read one respondent's pick, poll many, or take the
distribution they are all drawn from.

Output is masked to the digits 1-7, so the model cannot answer anything else --
no prose, no JSON, no refusal. That is the forced-choice part of a survey.

    .venv-gpu/Scripts/python tools/survey_probe.py --draws 200
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
from common import RES, banner
from sim_fields import SYSTEM, prompt

MODEL = "Qwen/Qwen2-VL-7B-Instruct"
MAX_PIXELS = 1024 * 28 * 28


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--field", default="facade_variation")
    ap.add_argument("--n-images", type=int, default=3)
    args = ap.parse_args()
    banner("one respondent, a panel, or the distribution")

    d = pd.read_csv(RES / "tables" / "scale_probe.csv")
    picks = d.head(args.n_images)

    import torch
    from PIL import Image
    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              Qwen2VLForConditionalGeneration)
    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True)
    proc = AutoProcessor.from_pretrained(MODEL, max_pixels=MAX_PIXELS)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL, quantization_config=qcfg, device_map="cuda").eval()
    ids = [proc.tokenizer.encode(str(k), add_special_tokens=False)[0]
           for k in range(1, 8)]
    f = args.field

    for r in picks.itertuples():
        im = Image.open(Path("data/raw/svi_90") / r.file).convert("RGB")
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": prompt(f)}]}]
        t = proc.apply_chat_template(msgs, tokenize=False,
                                     add_generation_prompt=True) + f'{{"{f}": '
        inp = proc(text=[t], images=[im], return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = model(**inp).logits[0, -1].float()

        # the distribution, masked to the seven digits -- forced choice
        p = torch.softmax(logits[ids] / args.temp, -1).cpu().numpy()
        p = p / p.sum()
        ev = float((p * np.arange(1, 8)).sum())
        am = int(np.argmax(p) + 1)

        # a panel: `draws` independent respondents from that same head
        rng = np.random.default_rng(0)
        draws = rng.choice(np.arange(1, 8), size=args.draws, p=p)

        print(f"\n  {r.file}")
        print(f"    {'digit':<8}" + "".join(f"{k:>7}" for k in range(1, 8)))
        print(f"    {'model p':<8}" + "".join(f"{x:>7.3f}" for x in p))
        cnt = np.array([(draws == k).sum() / args.draws for k in range(1, 8)])
        print(f"    {'panel':<8}" + "".join(f"{x:>7.3f}" for x in cnt))
        print(f"    one respondent, greedy   : {am}")
        print(f"    panel of {args.draws:<4} mean       : {draws.mean():.3f} "
              f"(sd {draws.std():.2f}, se {draws.std()/np.sqrt(args.draws):.3f})")
        print(f"    the distribution itself   : {ev:.3f}")
        print(f"    panel minus exact         : {draws.mean() - ev:+.3f}")

    print(f"\n  how many respondents to match the exact value?")
    print(f"    the panel mean has standard error sd/sqrt(N), so halving the")
    print(f"    error costs four times the compute. The logit read has no")
    print(f"    sampling error at all and costs one forward pass -- less than")
    print(f"    a single respondent, because nothing is generated.")


if __name__ == "__main__":
    main()
