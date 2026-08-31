"""How many images can share one generate() call before the card runs out?

The nine prompts for one image already batch together, which bought 1.60x.
Stacking several images into the same call batches the vision encoder too.
The card is 12.9 GB and the model is 5.9 GB in 4-bit, so the headroom is real
but finite -- this measures where the returns stop and checks the answers do
not drift, because padded batched attention can move logits.

    .venv-gpu/Scripts/python tools/batch_size_probe.py
"""
import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import banner
from sim_fields import FIELDS, SYSTEM, prompt
from sim_vlm_run import NAME_90, one

MODEL = "Qwen/Qwen2-VL-7B-Instruct"
MAX_PIXELS = 1024 * 28 * 28
ORDER = list(FIELDS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/raw/svi_90"))
    ap.add_argument("--images", type=int, default=12)
    ap.add_argument("--sizes", nargs="+", type=int, default=[1, 2, 3, 4])
    args = ap.parse_args()
    banner("images per generate() call")

    files = [p for p in sorted(args.src.rglob("*.jpg")) if NAME_90.search(p.name)]
    files = files[::max(1, len(files) // args.images)][:args.images]
    print(f"{len(files)} images x {len(ORDER)} fields\n")

    import torch
    from PIL import Image
    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              Qwen2VLForConditionalGeneration)
    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True)
    proc = AutoProcessor.from_pretrained(MODEL, max_pixels=MAX_PIXELS)
    proc.tokenizer.padding_side = "left"
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL, quantization_config=qcfg, device_map="cuda").eval()

    texts = []
    for f in ORDER:
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": prompt(f)}]}]
        texts.append(proc.apply_chat_template(msgs, tokenize=False,
                                              add_generation_prompt=True))

    def run(n_img):
        got = {}
        for start in range(0, len(files), n_img):
            chunk = files[start:start + n_img]
            ims = [Image.open(p).convert("RGB") for p in chunk]
            t = texts * len(chunk)
            imgs = [im for im in ims for _ in ORDER]
            inp = proc(text=t, images=imgs, padding=True,
                       return_tensors="pt").to("cuda")
            with torch.no_grad():
                g = model.generate(**inp, max_new_tokens=24, do_sample=False,
                                   pad_token_id=proc.tokenizer.eos_token_id)
            cut = inp.input_ids.shape[1]
            for k, row in enumerate(g):
                p, f = chunk[k // len(ORDER)], ORDER[k % len(ORDER)]
                got[(p.name, f)] = one(
                    proc.tokenizer.decode(row[cut:], skip_special_tokens=True), f)
        return got

    ref, base = None, None
    print(f"  {'imgs/call':>10}{'rows/call':>11}{'total s':>10}{'s/image':>10}"
          f"{'speedup':>9}{'peak GB':>9}   answers")
    for n in args.sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            t0 = time.perf_counter()
            got = run(n)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
        except torch.cuda.OutOfMemoryError:
            print(f"  {n:>10}{n*len(ORDER):>11}     out of memory")
            torch.cuda.empty_cache()
            continue
        peak = torch.cuda.max_memory_allocated() / 1024**3
        if ref is None:
            ref, base = got, dt
            same = "reference"
        else:
            ok = sum(1 for k in ref if got.get(k) == ref[k]
                     or (np.isnan(ref[k]) and np.isnan(got.get(k, np.nan))))
            same = f"{ok}/{len(ref)} identical"
        print(f"  {n:>10}{n*len(ORDER):>11}{dt:>10.1f}{dt/len(files):>10.2f}"
              f"{base/dt:>8.2f}x{peak:>9.2f}   {same}")

    print(f"\n  2724 half-views at each rate:")
    print(f"    (multiply s/image above by 2724 / 3600)")


if __name__ == "__main__":
    main()
