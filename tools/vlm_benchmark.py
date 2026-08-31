"""Several VLMs on the same two questions, plus what other papers report.

WHAT THIS IS AND IS NOT. Predicting measured GVI and VEI is a VALIDITY CHECK,
not a competition. GVI is a vegetation pixel share, so a segmenter measuring it
is measuring the target directly, and a VLM estimating it by eye is being
tested on the segmenter's home turf. Losing that comparison would not mean the
VLM is worse at anything the study needs.

The distinction that matters is capability. A segmenter can only emit shares of
the classes it was trained on. There is no ADE20K or Cityscapes class for
"somewhere to sit", "how much the frontage varies", or "how far greenery
relieves the enclosure" -- those are judgements, and only a VLM can produce
them. What GVI and VEI buy is the one thing those judgements otherwise lack: a
measured quantity to check the ratings against, so "the model rates greenery
sensibly" stops being an assertion.

So the vision models are here as a floor and a sanity check, clearly labelled,
and the question asked of each VLM is: does its judgement track a quantity we
can measure independently?

Every model sees the same stratified sample, is asked in the same words where
the interface allows, and is scored by Spearman with a bootstrap clustered on
face_id -- Moran's I is 0.62-0.66 here, so rows are not independent.

LITERATURE ROWS are included in the output and marked, because published
numbers on the same family of task are findings too, and reproducing the
pattern they report is worth more than another in-house number. They are NOT
comparable to our rho values -- different tasks, metrics and data -- and the
table says so rather than lining them up as if they were.

    .venv-gpu/Scripts/python tools/vlm_benchmark.py --sample 300
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import PROC, RES, banner

NAME_RE = re.compile(r"(\d+)_(n\d+)_([NESW])\.jpg$")

GREEN_Q = ("Rate the greenery visible in this Manhattan street view: trees, "
           "canopy, planting, green facades. Reply with ONE JSON object and "
           "nothing else: {\"greenery\": <1-7>} where 1 is no greenery at all "
           "and 7 is greenery dominating the view.")
ENCL_Q = ("Rate how strongly buildings enclose this Manhattan street into a "
          "room. Reply with ONE JSON object and nothing else: "
          "{\"enclosure\": <1-7>} where 1 is wide open with sky dominating "
          "and 7 is a deep canyon where the sky is a slot.")

# Reported elsewhere, on related tasks. Not comparable to our numbers.
LITERATURE = [
    ("CLIPSeg", "open-vocab seg", "scaffolding vs DOB permits, AUC",
     0.55, "this repo, tools/scaffold_eval.py"),
    ("CLIPSeg", "open-vocab seg", "same, restricted to the forward cone, AUC",
     0.51, "this repo, CLAUDE.md"),
    ("open-vocab detectors", "CLIP-family", "street trees vs city register, AUC",
     0.78, "this repo, tools/openvocab_eval.py"),
    ("closed-set detector", "vision model", "street trees vs city register, AUC",
     0.83, "this repo, tools/openvocab_eval.py"),
    ("open-vocab detectors", "CLIP-family", "sidewalk sheds vs register, AUC",
     0.51, "this repo, tools/openvocab_eval.py"),
    ("GPT-4o", "generative VLM", "urban region detection, mIoU",
     0.01, "MINGLE, AAAI 2026, Table 3"),
    ("Claude Sonnet", "generative VLM", "urban region detection, mIoU",
     0.02, "MINGLE, AAAI 2026, Table 3"),
    ("Qwen2-VL-7B zero-shot", "generative VLM", "urban region detection, mIoU",
     0.00, "MINGLE, AAAI 2026, Table 3"),
    ("MINGLE pipeline", "detector + VLM", "urban region detection, mIoU",
     0.64, "MINGLE, AAAI 2026, Table 3"),
    ("CAT-Seg", "open-vocab seg", "reported unusable on this imagery",
     float("nan"), "collaborator, blockology-gvi"),
]


def sample(src, n, seed=5):
    rows = []
    for jpg in sorted(Path(src).rglob("*.jpg")):
        m = NAME_RE.search(jpg.name)
        if m:
            rows.append({"file": str(jpg.relative_to(src)).replace("\\", "/"),
                         "path": jpg, "street": jpg.parent.parent.name,
                         "node_id": m.group(2)})
    fl = pd.DataFrame(rows)
    per = max(1, n // max(1, fl.street.nunique()))
    return pd.concat([g.sample(min(len(g), per), random_state=seed)
                      for _, g in fl.groupby("street")]).head(n)


def one_number(txt, key):
    m = re.search(r"\{.*\}", txt, re.S)
    if m:
        try:
            v = json.loads(m.group()).get(key)
            return float(v)
        except Exception:
            pass
    m = re.search(r"([1-7])(?:\s*/\s*7)?", txt)
    return float(m.group(1)) if m else np.nan


def run_generative(model_id, files, quant4=True):
    """Any Qwen2/2.5-VL or LLaVA-style chat model, asked both questions."""
    import torch
    from PIL import Image
    from tqdm.auto import tqdm
    from transformers import AutoProcessor, BitsAndBytesConfig, AutoModelForVision2Seq
    kw = {"device_map": "cuda"}
    if quant4:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    proc = AutoProcessor.from_pretrained(model_id, max_pixels=1024 * 28 * 28)
    model = AutoModelForVision2Seq.from_pretrained(model_id, **kw).eval()
    out = {"greenery": [], "enclosure": []}
    for p in tqdm(files, desc=Path(model_id).name, mininterval=15.0):
        im = Image.open(p).convert("RGB")
        for key, q in (("greenery", GREEN_Q), ("enclosure", ENCL_Q)):
            msgs = [{"role": "user", "content": [{"type": "image"},
                                                 {"type": "text", "text": q}]}]
            text = proc.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True)
            inp = proc(text=[text], images=[im], return_tensors="pt").to("cuda")
            with torch.no_grad():
                g = model.generate(**inp, max_new_tokens=24, do_sample=False,
                                   pad_token_id=proc.tokenizer.eos_token_id)
            out[key].append(one_number(
                proc.tokenizer.decode(g[0][inp.input_ids.shape[1]:],
                                      skip_special_tokens=True), key))
    del model
    import torch as T
    T.cuda.empty_cache()
    return out


def run_clipseg(files):
    """CLIPSeg: masked area for a text prompt, as a continuous score."""
    import torch
    from PIL import Image
    from tqdm.auto import tqdm
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
    mid = "CIDAS/clipseg-rd64-refined"
    proc = CLIPSegProcessor.from_pretrained(mid)
    model = CLIPSegForImageSegmentation.from_pretrained(mid).to("cuda").eval()
    prompts = ["trees and greenery", "tall buildings enclosing the street"]
    out = {"greenery": [], "enclosure": []}
    for p in tqdm(files, desc="CLIPSeg", mininterval=15.0):
        im = Image.open(p).convert("RGB")
        inp = proc(text=prompts, images=[im] * 2, padding=True,
                   return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = model(**inp).logits
        s = torch.sigmoid(logits).flatten(1).mean(1).float().cpu().numpy()
        out["greenery"].append(float(s[0]))
        out["enclosure"].append(float(s[1]))
    del model
    torch.cuda.empty_cache()
    return out


def spearman_ci(x, y, g, n=3000, rng=np.random.default_rng(0)):
    s = pd.DataFrame({"x": x, "y": y, "g": g}).dropna()
    if len(s) < 25 or s.x.nunique() < 2:
        return None
    r = s.x.corr(s.y, method="spearman")
    uniq = pd.unique(s.g)
    idx = {q: np.flatnonzero(s.g.to_numpy() == q) for q in uniq}
    bs = []
    for _ in range(n):
        sub = s.iloc[np.concatenate([idx[q] for q in rng.choice(uniq, len(uniq))])]
        if sub.x.nunique() > 1:
            bs.append(sub.x.corr(sub.y, method="spearman"))
    return r, np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5), len(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/raw/svi_180"))
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--table", type=Path,
                    default=RES / "tables" / "vlm_benchmark.csv")
    ap.add_argument("--models", nargs="+", default=["qwen2", "qwen25", "clipseg"])
    args = ap.parse_args()
    banner("several VLMs, the same two questions")

    fl = sample(args.src, args.sample)
    met = pd.read_csv(PROC / "metrics.csv")[["node_id", "GVI", "VEI", "face_id"]]
    fl = fl.merge(met, on="node_id", how="left")
    fl["face_id"] = fl.face_id.fillna(fl.node_id)
    print(f"{len(fl)} images across {fl.street.nunique()} streets\n")

    SPECS = {
        "qwen2":  ("Qwen2-VL-7B", "generative VLM",
                   lambda f: run_generative("Qwen/Qwen2-VL-7B-Instruct", f)),
        "qwen25": ("Qwen2.5-VL-3B", "generative VLM",
                   lambda f: run_generative("Qwen/Qwen2.5-VL-3B-Instruct", f)),
        "clipseg": ("CLIPSeg", "open-vocab seg (CLIP-family)",
                    lambda f: run_clipseg(f)),
    }
    files = fl.path.tolist()
    rows = []
    for key in args.models:
        if key not in SPECS:
            print(f"unknown model key {key}")
            continue
        label, cls, fn = SPECS[key]
        print(f"\n--- {label} ---")
        try:
            got = fn(files)
        except Exception as e:
            print(f"  failed: {type(e).__name__}: {e}")
            continue
        for q, tgt in (("greenery", "GVI"), ("enclosure", "VEI")):
            r = spearman_ci(np.asarray(got[q], float),
                            fl[tgt].to_numpy(float), fl.face_id.to_numpy())
            if r:
                rows.append({"model": label, "class": cls, "question": q,
                             "target": tgt, "rho": r[0], "lo": r[1],
                             "hi": r[2], "n": r[3], "source": "measured here"})
                print(f"  {q:<10} vs {tgt}  rho {r[0]:+.3f}  [{r[1]:+.3f},{r[2]:+.3f}]")

    d = pd.DataFrame(rows)
    if len(d):
        args.table.parent.mkdir(parents=True, exist_ok=True)
        d.to_csv(args.table, index=False)
        print(f"\n=== measured here, same {len(fl)} images, Spearman vs measured ===")
        for tgt in ("GVI", "VEI"):
            s = d[d.target == tgt].sort_values("rho", ascending=False)
            if not len(s):
                continue
            print(f"\n  predicting {tgt}")
            for r in s.itertuples():
                print(f"    {r.model:<16}{r._2:<30}{r.rho:>7.3f}  "
                      f"[{r.lo:+.3f},{r.hi:+.3f}]")

    print("\n=== reported elsewhere, related tasks, NOT comparable to the above ===")
    print(f"  {'model':<24}{'class':<22}{'task and metric':<44}{'value':>7}")
    for m, c, t, v, src in LITERATURE:
        vs = "n/a" if v != v else f"{v:.2f}"
        print(f"  {m:<24}{c:<22}{t:<44}{vs:>7}   {src}")
    print("\n  Different tasks, metrics and data. Listed because the pattern they")
    print("  report -- open-vocab models near chance on street furniture, generative")
    print("  VLMs near zero on localisation, a detector plus a VLM far ahead of")
    print("  either alone -- is itself a finding this study reproduces.")


if __name__ == "__main__":
    main()
