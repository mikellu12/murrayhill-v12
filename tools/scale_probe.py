"""Two anchors against seven, read as argmax and as expected value.

Four cells on the same images in the same pass, so nothing differs but the
prompt and the read:

    2-anchor / argmax   what the 3.7-hour run recorded
    2-anchor / EV       the same call, distribution instead of token
    7-anchor / argmax   every rung named
    7-anchor / EV       both changes

Scored against the share measured over each half-view's own 90 degrees.
Answers come from one forward pass with the JSON prefix forced, so the
argmax here is exactly the token generate() would have emitted -- the
comparison is paired, not two runs.

    .venv-gpu/Scripts/python tools/scale_probe.py --n 64
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
from common import PROC, RES, banner
from sim_fields import SYSTEM, prompt
from sim_scale import prompt7
from sim_vlm_run import NAME_90

MODEL = "Qwen/Qwen2-VL-7B-Instruct"
MAX_PIXELS = 1024 * 28 * 28
FIELDS = ["green_eye_level", "vertical_greenery", "sky_openness",
          "facade_variation", "walkable_ground"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/raw/svi_90"))
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()
    banner("two anchors against seven")

    rows = []
    for jpg in sorted(args.src.rglob("*.jpg")):
        m = NAME_90.search(jpg.name)
        if m:
            rows.append({"src_path": jpg, "node_id": m.group(2),
                         "side": m.group(4),
                         "file": str(jpg.relative_to(args.src)).replace("\\", "/")})
    fl = pd.DataFrame(rows)
    met = pd.read_csv(PROC / "metrics.csv")[["node_id", "GVI"]]
    fl = fl.merge(met, on="node_id", how="left").dropna(subset=["GVI"])
    fl["q"] = pd.qcut(fl.GVI, 4, labels=False, duplicates="drop")
    per = max(1, args.n // fl.q.nunique())
    fl = pd.concat([g.sample(min(len(g), per), random_state=args.seed)
                    for _, g in fl.groupby("q")]).head(args.n).reset_index(drop=True)
    print(f"{len(fl)} half-views x {len(FIELDS)} fields x 2 prompts\n")

    import torch
    from PIL import Image
    from tqdm.auto import tqdm
    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              Qwen2VLForConditionalGeneration)
    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True)
    proc = AutoProcessor.from_pretrained(MODEL, max_pixels=MAX_PIXELS)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL, quantization_config=qcfg, device_map="cuda").eval()
    tok = proc.tokenizer
    ids = [tok.encode(str(k), add_special_tokens=False)[0] for k in range(1, 8)]

    def ask(im, text, field):
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": text}]}]
        t = proc.apply_chat_template(msgs, tokenize=False,
                                     add_generation_prompt=True)
        t += f'{{"{field}": '
        inp = proc(text=[t], images=[im], return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = model(**inp).logits[0, -1].float()
        p = torch.softmax(logits, -1)[ids].cpu().numpy()
        p = p / p.sum()
        return int(np.argmax(p) + 1), float((p * np.arange(1, 8)).sum()), float(p[3])

    out = []
    for r in tqdm(list(fl.itertuples()), desc="images", mininterval=10.0):
        im = Image.open(r.src_path).convert("RGB")
        rec = {"file": r.file, "node_id": r.node_id, "side": r.side}
        for f in FIELDS:
            for tag, text in (("a2", prompt(f)), ("a7", prompt7(f))):
                am, ev, p4 = ask(im, text, f)
                rec[f"{f}__{tag}_argmax"] = am
                rec[f"{f}__{tag}_ev"] = ev
                rec[f"{f}__{tag}_p4"] = p4
        out.append(rec)
    d = pd.DataFrame(out)
    p = RES / "tables" / "scale_probe.csv"
    d.to_csv(p, index=False)

    print(f"\n  does 4 appear now?")
    print(f"  {'field':<22}{'2-anchor P(4)':>15}{'7-anchor P(4)':>15}"
          f"{'2a argmax=4':>13}{'7a argmax=4':>13}")
    for f in FIELDS:
        print(f"  {f:<22}{d[f'{f}__a2_p4'].mean():>15.3f}"
              f"{d[f'{f}__a7_p4'].mean():>15.3f}"
              f"{int((d[f'{f}__a2_argmax'] == 4).sum()):>13}"
              f"{int((d[f'{f}__a7_argmax'] == 4).sum()):>13}")

    print(f"\n  spread of the argmax")
    print(f"  {'field':<22}{'2-anchor':>10}{'7-anchor':>10}   values used")
    for f in FIELDS:
        v7 = sorted(d[f"{f}__a7_argmax"].unique())
        print(f"  {f:<22}{d[f'{f}__a2_argmax'].nunique():>10}"
              f"{d[f'{f}__a7_argmax'].nunique():>10}   {v7}")
    print(f"\nwrote {p}")
    print("  score with tools/scale_probe_score.py")


if __name__ == "__main__":
    main()
