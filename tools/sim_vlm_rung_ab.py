"""A/B the current rungs against a rewrite, scored on the measured twin.

Three fields change what they measure partway up the ladder:

  resting_affordance  rungs 2-4 COUNT objects ("a single step", "one stoop",
                      "a few stoops"), rungs 5-7 switch to EXTENT ("along much
                      of the frontage").
  walkable_ground     three metrics in seven rungs -- width (1,2,4,5,6),
                      obstruction (3), share of the view (7).
  green_softening     every rung asks for a judgment about an EFFECT
                      ("relieves the enclosure"), and rungs 3-4 describe a
                      ratio between two subjects. That is an inference, not an
                      observation, and it is the field with 60% of its images
                      two-peaked.

The rewrite holds one observable quantity per ladder and uses extent all the
way up. There is precedent: green_eye_level moved from +0.649 to +0.737
against measured pixels when its rungs stopped counting planters and started
asking about amount.

SCORED, NOT ASSERTED. Two of the three have a measured twin over the same 90
degrees, so the rewrite has to beat the current wording on that or it is not an
improvement. green_softening has no twin and is carried for its bimodality rate
alone -- the thing the rewrite is meant to fix.

The embedding ladder check is deliberately NOT used here. It flagged all ten
current fields as defective; the empirical medians then came out monotone on
every field it flagged, so it measures sentence similarity rather than whether
the model uses the rungs in order.

    .venv-gpu/Scripts/python tools/sim_vlm_rung_ab.py --n 300
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
from common import RES, banner
from sim_fields import SYSTEM
from sim_scale import SCALE

MODEL = "Qwen/Qwen2-VL-7B-Instruct"
MAX_PIXELS = 768 * 768

NEW = {
 "resting_affordance": [
   "nothing to sit or lean on anywhere in view",
   "a trace: one perchable ledge or step, nothing more",
   "places to sit along a small part of the frontage",
   "places to sit along about half the frontage",
   "places to sit along most of the frontage",
   "places to sit along nearly all of the frontage",
   "continuous places to sit along the whole frontage"],
 "walkable_ground": [
   "no usable sidewalk on this side",
   "a trace of clear sidewalk, blocked almost throughout",
   "clear sidewalk along a small part of the frontage",
   "clear sidewalk along about half the frontage",
   "clear sidewalk along most of the frontage",
   "clear sidewalk along nearly all of the frontage",
   "clear sidewalk along the whole frontage"],
 "green_softening": [
   "the building surface is bare, no greenery in front of it",
   "a trace of greenery in front of the building surface",
   "greenery in front of a small part of the building surface",
   "greenery in front of about half the building surface",
   "greenery in front of most of the building surface",
   "greenery in front of nearly all the building surface",
   "greenery hides the building surface almost completely"],
}

# Moved into the system turn rather than the rung text: the mechanical criteria
# say what COUNTS, which is a definition, while the rungs say HOW MUCH, which is
# the scale. Mixing the two is what the rewrite is removing.
EXTRA = {
 "resting_affordance":
   " Places to sit means masonry stoops 0.9-1.5 m high, ledges or planter rims"
   " 0.4-0.6 m high and at least 0.3 m deep, low walls, or benches. Elements"
   " blocked by spikes or railings do not count.",
 "walkable_ground":
   " Clear sidewalk means the continuous unobstructed paving a pedestrian can"
   " walk on. Scaffolding, bins, parked vehicles and construction reduce it.",
}

# vertical_greenery is the untouched control: if BOTH arms move on a field whose
# rungs did not change, the difference is run-to-run noise, not the rewrite.
FIELDS = ["resting_affordance", "walkable_ground", "green_softening",
          "vertical_greenery"]
TWIN = {"resting_affordance": ["stoop_stair", "bench_seating"],
        "walkable_ground": ["sidewalk", "curb_edge"],
        "vertical_greenery": ["tree", "planter_container"]}


def prompt(field, rungs):
    steps = "\n".join(f"{i + 1} = {s}" for i, s in enumerate(rungs))
    return (f"Rate this Manhattan street view. Reply with ONE JSON object and "
            f"nothing else: {{\"{field}\": <1-7>}}, using this scale:\n\n{steps}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    banner("current rungs vs rewrite, scored on the measured twin")

    obs = pd.read_csv(RES / "tables" / "vlm_observations.csv")
    src = Path("data/raw/svi_90")
    obs = obs[[(src / f).exists() for f in obs.file]]
    per = max(1, args.n // max(obs.typology.nunique(), 1))
    take = pd.concat([g.sample(min(len(g), per), random_state=args.seed)
                      for _, g in obs.groupby("typology")])
    take = take.sample(min(args.n, len(take)), random_state=args.seed)
    print(f"{len(take)} images x {len(FIELDS)} fields x 2 wordings\n")

    import torch
    from PIL import Image
    from tqdm.auto import tqdm
    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              Qwen2VLForConditionalGeneration)
    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True)
    proc = AutoProcessor.from_pretrained(MODEL, max_pixels=MAX_PIXELS)
    proc.tokenizer.padding_side = "left"
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL, quantization_config=qcfg, device_map="cuda").eval()
    tok = proc.tokenizer
    ids = [tok.encode(str(k), add_special_tokens=False)[0] for k in range(1, 8)]
    ks = np.arange(1, 8)

    def ask(field, rungs, im, extra=""):
        msgs = [{"role": "system", "content": SYSTEM + extra},
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt(field, rungs)}]}]
        t = proc.apply_chat_template(msgs, tokenize=False,
                                     add_generation_prompt=True)
        inp = proc(text=[t + '{"' + field + '": '], images=[im],
                   return_tensors="pt").to("cuda")
        with torch.no_grad():
            lg = model(**inp).logits[:, -1, :].float()
        p = torch.softmax(lg[0, ids], -1).cpu().numpy()
        return p / p.sum()

    rows = []
    t0 = time.time()
    for r in tqdm(list(take.itertuples()), desc="images"):
        im = Image.open(src / r.file).convert("RGB")
        for f in FIELDS:
            pa = ask(f, SCALE[f], im)
            pb = (ask(f, NEW[f], im, EXTRA.get(f, "")) if f in NEW else pa)
            rec = dict(file=r.file, field=f, typology=r.typology)
            for tag, p in (("old", pa), ("new", pb)):
                s = np.sort(p)[::-1]
                a = np.argsort(p)[::-1]
                rec[f"{tag}_ev"] = float((p * ks).sum())
                rec[f"{tag}_ent"] = float(-(p * np.log(p + 1e-12)).sum()
                                          / np.log(7))
                rec[f"{tag}_bim"] = int(abs(a[0] - a[1]) > 1)
                rec[f"{tag}_dec"] = int(s[0] > 0.5)
            rows.append(rec)
    d = pd.DataFrame(rows)
    out = RES / "tables" / "vlm_rung_ab.csv"
    d.to_csv(out, index=False)

    S = pd.read_csv("data/processed/seg90_shares.csv")
    o = pd.read_csv(RES / "tables" / "vlm_observations.csv")[["file", "face_id"]]
    m = d.merge(S, on="file", how="left").merge(o, on="file", how="left")
    from scipy.stats import spearmanr

    print("\n  ACCURACY vs the measured twin\n")
    print(f"    {'field':<22}{'old':>9}{'new':>9}{'diff':>9}"
          f"{'95% CI (face-clustered)':>28}")
    rng = np.random.default_rng(0)
    for f, cls in TWIN.items():
        g = m[m.field == f].dropna(subset=["face_id"])
        y = g[[c for c in cls if c in g.columns]].sum(axis=1).values
        a = spearmanr(g.old_ev, y).statistic
        b = spearmanr(g.new_ev, y).statistic
        faces = g.face_id.unique()
        idxs = [np.flatnonzero(g.face_id.values == fc) for fc in faces]
        bs = []
        for _ in range(2000):
            ix = np.concatenate([idxs[j] for j in
                                 rng.integers(0, len(faces), len(faces))])
            bs.append(spearmanr(g.new_ev.values[ix], y[ix]).statistic
                      - spearmanr(g.old_ev.values[ix], y[ix]).statistic)
        v = np.array(bs)
        lo, hi = np.percentile(v, [2.5, 97.5])
        tag = " better" if lo > 0 else (" worse" if hi < 0 else "")
        note = "  (control, rungs unchanged)" if f == "vertical_greenery" else ""
        print(f"    {f:<22}{a:>+9.3f}{b:>+9.3f}{b-a:>+9.3f}"
              f"   [{lo:+.3f}, {hi:+.3f}]{tag}{note}")

    print("\n  DISTRIBUTION SHAPE\n")
    print(f"    {'field':<22}{'entropy':>16}{'bimodal':>16}{'decisive':>16}")
    print(f"    {'':<22}{'old':>8}{'new':>8}{'old':>8}{'new':>8}{'old':>8}{'new':>8}")
    for f in FIELDS:
        g = d[d.field == f]
        print(f"    {f:<22}{g.old_ent.mean():>8.3f}{g.new_ent.mean():>8.3f}"
              f"{g.old_bim.mean()*100:>7.0f}%{g.new_bim.mean()*100:>7.0f}%"
              f"{g.old_dec.mean()*100:>7.0f}%{g.new_dec.mean()*100:>7.0f}%")
    print(f"\n  {time.time()-t0:.0f} s   wrote {out}")


if __name__ == "__main__":
    main()
