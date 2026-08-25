"""Does the VLM already weight the centre of a view, or does it read it flat?

The manuscript's case for a VLM is that it judges a scene the way a person
does, "in a single operational pass", rather than tallying surfaces. If that
holds, the perceptual weighting the literature reports -- roughly 80 per cent
of attention in the central 60 degrees, 10 each side -- is already inside the
model's judgement, and imposing those weights again in code would count them
twice.

That is an assumption, not a fact, and this measures it.

The 1440 px panorama covers 180 degrees at exactly 8 px per degree, so the
three 60-degree cones are clean crops at 0-480, 480-960 and 960-1440. Each is
rated with the SAME schema used on the whole panorama, and the whole-panorama
rating already exists for all 1,254 images in svi_180_sim_vlm.csv.

Two readings of the result:

  regress  whole ~ w_L*left + w_C*centre + w_R*right

  centre coefficient near 0.8, peripherals near 0.1
      the model already weights as the literature describes; the split is
      redundant and the paper can say so with a number.

  all three near 0.33
      the model reads the panorama flat, and a 60-degree framing would be
      adding something the whole-view rating does not capture.

Coefficients are fitted without an intercept and constrained non-negative,
because a negative weight on a cone has no perceptual reading; they are then
scaled to sum to 1 so they can be compared with 0.8/0.1/0.1 directly.

    .venv-gpu/Scripts/python tools/svi_180_cone_test.py --sample 102
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import (AutoProcessor, BitsAndBytesConfig,
                          Qwen2VLForConditionalGeneration)

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
from common import PROC, RES, banner
from svi_180_sim_vlm import (MODEL, MAX_PIXELS, SYSTEM, SCHEMA, RATE, COUNT,
                             CAT, NAME_RE, parse)

CONES = ("left", "centre", "right")
# Neutral labels. "peripheral" tells the model the view matters less, which
# would produce the very weighting the test is trying to detect.
CONE_LINE = ("This is the {} 60-degree third of a pedestrian's forward view, "
             "looking along the street in the direction of travel.\n\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/raw/svi_180"))
    ap.add_argument("--table", type=Path,
                    default=RES / "tables" / "svi_180_cone_test.csv")
    ap.add_argument("--sample", type=int, default=102)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--max-new-tokens", type=int, default=260)
    ap.add_argument("--restart", action="store_true")
    args = ap.parse_args()
    banner("does the vlm centre-weight a 180-degree view?")

    rows = []
    for jpg in sorted(args.src.rglob("*.jpg")):
        m = NAME_RE.search(jpg.name)
        if m:
            rows.append({"file": str(jpg.relative_to(args.src)).replace("\\", "/"),
                         "path": jpg, "street": jpg.parent.parent.name,
                         "direction": jpg.parent.name, "node_id": m.group(2),
                         "seq": int(m.group(1))})
    fl = pd.DataFrame(rows)

    whole_p = RES / "tables" / "svi_180_sim_vlm.csv"
    if not whole_p.exists():
        sys.exit(f"need {whole_p} -- the whole-panorama ratings are the baseline")
    whole = pd.read_csv(whole_p)
    fl = fl[fl.file.isin(set(whole.file))]

    done = pd.DataFrame()
    if args.table.exists() and not args.restart:
        done = pd.read_csv(args.table)
        fl = fl[~fl.file.isin(set(done.file))]
        print(f"{len(done)} already done, {len(fl)} remaining")
    per = max(1, args.sample // max(1, fl.street.nunique()))
    fl = pd.concat([g.sample(min(len(g), per), random_state=args.seed)
                    for _, g in fl.groupby("street")]).head(args.sample)
    print(f"{len(fl)} images across {fl.street.nunique()} streets, "
          f"{len(fl) * 3} calls")

    if len(fl):
        print(f"loading {MODEL} in 4-bit NF4")
        qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch.bfloat16,
                                  bnb_4bit_use_double_quant=True)
        proc = AutoProcessor.from_pretrained(MODEL, max_pixels=MAX_PIXELS)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL, quantization_config=qcfg, device_map="cuda").eval()

        def judge(img, cone):
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text",
                         "text": CONE_LINE.format(cone.upper()) + SCHEMA}]}]
            text = proc.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True)
            inputs = proc(text=[text], images=[img], return_tensors="pt").to("cuda")
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                     do_sample=False,
                                     pad_token_id=proc.tokenizer.eos_token_id)
            return parse(proc.tokenizer.decode(gen[0][inputs.input_ids.shape[1]:],
                                               skip_special_tokens=True))

        out = []
        for r in tqdm(list(fl.itertuples()), desc="images", mininterval=10.0):
            im = Image.open(r.path).convert("RGB")
            W, H = im.size
            third = W // 3
            rec = {"file": r.file, "street": r.street, "node_id": r.node_id}
            for i, cone in enumerate(CONES):
                crop = im.crop((i * third, 0, (i + 1) * third if i < 2 else W, H))
                got = judge(crop, cone)
                for k, v in got.items():
                    rec[f"{cone}_{k}"] = v
            out.append(rec)
            if len(out) % 20 == 0:
                pd.concat([done, pd.DataFrame(out)], ignore_index=True).to_csv(
                    args.table, index=False)
        done = pd.concat([done, pd.DataFrame(out)], ignore_index=True)
        args.table.parent.mkdir(parents=True, exist_ok=True)
        done.to_csv(args.table, index=False)
    print(f"\n{len(done)} images rated per cone -> {args.table}")

    # --- the actual question -------------------------------------------------
    j = done.merge(whole, on="file", how="inner", suffixes=("", "_w"))
    print(f"\njoined to whole-panorama ratings: {len(j)}\n")
    rng = np.random.default_rng(0)

    print(f"{'field':<24}{'rho vs left':>13}{'rho vs centre':>15}{'rho vs right':>14}")
    print("-" * 66)
    implied = []
    for f in RATE:
        cols = [f"{c}_{f}" for c in CONES]
        if f not in j.columns or not all(c in j.columns for c in cols):
            continue
        s = j[cols + [f]].dropna()
        if len(s) < 20 or s[f].nunique() < 2:
            continue
        rr = [s[c].corr(s[f], method="spearman") for c in cols]
        print(f"{f:<24}{rr[0]:>13.3f}{rr[1]:>15.3f}{rr[2]:>14.3f}")

        # implied weights: non-negative least squares, no intercept, rescaled
        X = s[cols].to_numpy(float)
        y = s[f].to_numpy(float)
        try:
            from scipy.optimize import nnls
            w, _ = nnls(X, y)
        except Exception:
            w = np.linalg.lstsq(X, y, rcond=None)[0].clip(0)
        if w.sum() > 0:
            implied.append(w / w.sum())

    if implied:
        W = np.array(implied)
        print("\n=== implied weights, non-negative least squares, rescaled to 1 ===")
        print(f"  {'':<10}{'left':>9}{'centre':>9}{'right':>9}")
        print(f"  {'mean':<10}{W[:,0].mean():>9.3f}{W[:,1].mean():>9.3f}{W[:,2].mean():>9.3f}")
        print(f"  {'median':<10}{np.median(W[:,0]):>9.3f}{np.median(W[:,1]):>9.3f}"
              f"{np.median(W[:,2]):>9.3f}")
        print(f"  {'literature':<10}{0.10:>9.3f}{0.80:>9.3f}{0.10:>9.3f}")
        print(f"  {'uniform':<10}{0.333:>9.3f}{0.333:>9.3f}{0.333:>9.3f}")
        c = np.median(W[:, 1])
        print(f"\n  centre weight {c:.3f}: "
              + ("close to the literature's 0.80 -- the model already "
                 "centre-weights" if c > 0.6 else
                 "close to uniform -- the model reads the panorama flat"
                 if c < 0.45 else "between the two; neither reading is clean"))


if __name__ == "__main__":
    main()
