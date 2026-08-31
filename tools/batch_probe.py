"""Does asking the nine fields as a batch cost less than nine calls?

Solo prompts are the only variant whose accuracy gain replicated, but nine of
them cost 0.93 s each against 4.43 s for the single twelve-field call. Almost
all of that is fixed cost -- the same image pushed through the vision encoder
nine times -- not token generation.

Batching the nine prompts into one generate() does not remove those nine
encodes, but it runs them in one forward pass instead of nine sequential ones,
which is where the GPU is idle. This measures whether that is worth having,
and checks the answers are the same either way: padding and batched attention
can change logits slightly, and a batched run that quietly answers differently
is worse than no speedup at all.

Only the nine fields the SIM formula actually reads are asked. rest_affordance
and the two categorical fields are in the current schema and are consumed by
nothing downstream.

    .venv-gpu/Scripts/python tools/batch_probe.py --n 8
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
from common import PROC, RES, banner
from prompt_probe import ANCHOR, NAME_RE, SYSTEM, prompt_solo

MODEL = "Qwen/Qwen2-VL-7B-Instruct"
MAX_PIXELS = 1024 * 28 * 28

# exactly what sim_terms() reads, nothing else
SIM_FIELDS = ["vertical_greenery", "vertical_hardscape", "green_eye_level",
              "green_softening", "signage_detail", "enclosure",
              "facade_variation", "walking_room", "ground_floor_activity"]


def one(txt, field):
    m = re.search(r"\{.*\}", txt, re.S)
    if m:
        try:
            v = json.loads(m.group()).get(field)
            return float(v)
        except Exception:
            pass
    m = re.search(r"([1-7])", txt)
    return float(m.group(1)) if m else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/raw/svi_90"))
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()
    banner("nine solo calls, sequential against batched")

    files = []
    for jpg in sorted(args.src.rglob("*.jpg")):
        if NAME_RE.search(jpg.name):
            files.append(jpg)
        if len(files) >= args.n * 40:
            break
    files = files[::max(1, len(files) // args.n)][:args.n]
    print(f"{len(files)} images x {len(SIM_FIELDS)} fields\n")

    import torch
    from PIL import Image
    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              Qwen2VLForConditionalGeneration)
    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True)
    proc = AutoProcessor.from_pretrained(MODEL, max_pixels=MAX_PIXELS)
    # generation needs the pad on the left or the batched rows decode from
    # the wrong offset
    proc.tokenizer.padding_side = "left"
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL, quantization_config=qcfg, device_map="cuda").eval()

    def texts():
        out = []
        for f in SIM_FIELDS:
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": [{"type": "image"},
                                                 {"type": "text",
                                                  "text": prompt_solo(f)}]}]
            out.append(proc.apply_chat_template(msgs, tokenize=False,
                                                add_generation_prompt=True))
        return out

    T = texts()
    seq, bat = {}, {}

    # warm up so the first call does not carry cuda init
    im0 = Image.open(files[0]).convert("RGB")
    inp = proc(text=[T[0]], images=[im0], return_tensors="pt").to("cuda")
    with torch.no_grad():
        model.generate(**inp, max_new_tokens=8, do_sample=False,
                       pad_token_id=proc.tokenizer.eos_token_id)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for p in files:
        im = Image.open(p).convert("RGB")
        for f, t in zip(SIM_FIELDS, T):
            inp = proc(text=[t], images=[im], return_tensors="pt").to("cuda")
            with torch.no_grad():
                g = model.generate(**inp, max_new_tokens=24, do_sample=False,
                                   pad_token_id=proc.tokenizer.eos_token_id)
            txt = proc.tokenizer.decode(g[0][inp.input_ids.shape[1]:],
                                        skip_special_tokens=True)
            seq[(p.name, f)] = one(txt, f)
    torch.cuda.synchronize()
    t_seq = time.perf_counter() - t0

    t0 = time.perf_counter()
    for p in files:
        im = Image.open(p).convert("RGB")
        inp = proc(text=T, images=[im] * len(SIM_FIELDS), padding=True,
                   return_tensors="pt").to("cuda")
        with torch.no_grad():
            g = model.generate(**inp, max_new_tokens=24, do_sample=False,
                               pad_token_id=proc.tokenizer.eos_token_id)
        cut = inp.input_ids.shape[1]
        for f, row in zip(SIM_FIELDS, g):
            txt = proc.tokenizer.decode(row[cut:], skip_special_tokens=True)
            bat[(p.name, f)] = one(txt, f)
    torch.cuda.synchronize()
    t_bat = time.perf_counter() - t0

    n = len(files)
    print(f"  sequential  {t_seq:7.1f} s   {t_seq/n:6.2f} s per image")
    print(f"  batched     {t_bat:7.1f} s   {t_bat/n:6.2f} s per image")
    print(f"  speedup     {t_seq/t_bat:7.2f}x\n")

    same = sum(1 for k in seq if (seq[k] == bat[k])
               or (pd.isna(seq[k]) and pd.isna(bat[k])))
    print(f"  identical answers: {same}/{len(seq)} "
          f"({same/len(seq)*100:.0f}%)")
    diff = [(k, seq[k], bat[k]) for k in seq
            if seq[k] != bat[k] and not (pd.isna(seq[k]) and pd.isna(bat[k]))]
    for (fn, f), a, b in diff[:10]:
        print(f"    {fn:<28}{f:<24}sequential {a}  batched {b}")
    if diff:
        d = pd.DataFrame(diff, columns=["k", "seq", "bat"])
        print(f"  mean absolute difference {(d.seq - d.bat).abs().mean():.2f} "
              f"on {len(d)} disagreeing answers")

    for label, per in (("full 12-field schema", 4.43),
                       ("9 solo, sequential", t_seq / n),
                       ("9 solo, batched", t_bat / n)):
        print(f"  {label:<26}{per:>6.2f} s/image   "
              f"2724 halves {2724*per/3600:>5.2f} h   "
              f"1254 whole {1254*per/3600:>5.2f} h")


if __name__ == "__main__":
    main()
