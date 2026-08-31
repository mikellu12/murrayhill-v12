"""Does giving the model room to think before it answers change the answer?

sim_vlm_run.py forces the assistant prefix and reads the digit distribution at
the very next token position. The model gets ZERO tokens of computation between
seeing the image and committing to a number. That is a constraint we imposed,
not a property of the model, and it is the leading suspect for why
resting_affordance is flat -- entropy 0.837, 26% near-ties, and no alternative
readout recovering more than 1.2x the between-street structure.

This runs the SAME images three ways.

  A  the current prefix-read, reproduced here rather than imported so the arms
     differ in exactly one thing.
  B  describe first, then append the same prefix and read the same seven
     logits. Identical readout; only the working space differs.
  C  six cumulative binary questions -- "at least as much as rung k?" for
     k=2..7 -- scored 1 + sum P(>=k). No seven-way softmax anywhere, one
     anchor visible at a time, and straight off the image so perception is
     held fixed.

A and B share the seven-bin readout, so B tests working space while holding
binning constant. C is the reverse: it changes the readout and holds working
space constant. Between them the two hypotheses are separable, which neither
arm manages alone.

C also yields something the others cannot: a coherence rate. If the model says
"yes, more than rung 5" and "no, less than rung 3" on the same image, the
ladder is not being read as ordered, and that is measurable here rather than
inferred.

vertical_greenery is carried as a control. It correlates +0.72 with measured
pixels and is the healthiest field in the set, so if arm B moves IT too, the
effect is the prompt shape in general rather than this field's defect.

If entropy falls toward vertical_greenery's 0.682, the no-working-space
explanation holds and a full re-run is worth costing. If it does not move, the
problem is perception, no restructuring rescues it, and this cost twenty
minutes instead of a night.

Do not run against a busy GPU -- the segmentation batch reserves 6.5 GB per
worker and both the timing and the memory headroom will be wrong.

    .venv-gpu/Scripts/python tools/sim_vlm_describe_probe.py --n 100
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
from sim_scale import SCALE, prompt7

MODEL = "Qwen/Qwen2-VL-7B-Instruct"
MAX_PIXELS = 768 * 768
FIELDS_TESTED = ["resting_affordance", "vertical_greenery"]

# Deliberately NOT the rung language. Asking the model to describe in the
# scale's own words would leak the answer into the description and the two arms
# would no longer differ in one thing.
DESCRIBE = {
    "resting_affordance":
        "Look at this Manhattan street view. Describe only what a person could "
        "sit or lean on: stoops, steps, ledges, low walls, planter rims, "
        "benches. Say where they are and roughly how much of the frontage they "
        "cover. If there is nothing, say so plainly. Two or three sentences.",
    "vertical_greenery":
        "Look at this Manhattan street view. Describe only the vegetation: "
        "tree canopy, green walls, hedges, planters. Say where it is and "
        "roughly how much of the frontage it covers. If there is none, say so "
        "plainly. Two or three sentences.",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-new", type=int, default=90)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    banner("does room to think change the answer?")

    obs = pd.read_csv(RES / "tables" / "vlm_observations.csv")
    src = Path("data/raw/svi_90")
    # Built by concat, not groupby().apply(): on this pandas the grouping key
    # is dropped from the frame apply() receives, so `typology` came back
    # missing and the run died after loading the model.
    obs = obs[[(src / f).exists() for f in obs.file]]
    per = max(1, args.n // max(obs.typology.nunique(), 1))
    take = pd.concat([g.sample(min(len(g), per), random_state=args.seed)
                      for _, g in obs.groupby("typology")])
    take = take.sample(min(args.n, len(take)), random_state=args.seed)
    print(f"{len(take)} images across {take.typology.nunique()} typologies\n")

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
    # "yes"/"no" as they appear after the forced opening quote. Encoded once
    # and asserted distinct -- a tokenizer that maps both to the same id would
    # make arm C silently return 0.5 everywhere.
    yn = [tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "no")]
    assert yn[0] != yn[1], "yes/no share a token id"

    def read(text, im):
        """Seven-way distribution at the token after `text`. Same in A and B."""
        inp = proc(text=[text], images=[im], return_tensors="pt").to("cuda")
        with torch.no_grad():
            lg = model(**inp).logits[:, -1, :].float()
        p = torch.softmax(lg[0, ids], -1).cpu().numpy()
        return p / p.sum()

    def read_yes(text, im):
        """P(yes) over just the two tokens, at the same position."""
        inp = proc(text=[text], images=[im], return_tensors="pt").to("cuda")
        with torch.no_grad():
            lg = model(**inp).logits[:, -1, :].float()
        p = torch.softmax(lg[0, yn], -1).cpu().numpy()
        return float(p[0])

    rows, spent = [], {"A": 0.0, "B": 0.0, "C": 0.0}
    for r in tqdm(list(take.itertuples()), desc="images"):
        im = Image.open(src / r.file).convert("RGB")
        for f in FIELDS_TESTED:
            ask = prompt7(f)
            base = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": [{"type": "image"},
                                                 {"type": "text", "text": ask}]}]
            t0 = time.time()
            pre = proc.apply_chat_template(base, tokenize=False,
                                           add_generation_prompt=True)
            pa = read(pre + '{"' + f + '": ', im)
            spent["A"] += time.time() - t0

            t0 = time.time()
            dmsg = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": [{"type": "image"},
                                                 {"type": "text",
                                                  "text": DESCRIBE[f]}]}]
            dtext = proc.apply_chat_template(dmsg, tokenize=False,
                                             add_generation_prompt=True)
            inp = proc(text=[dtext], images=[im], return_tensors="pt").to("cuda")
            with torch.no_grad():
                gen = model.generate(**inp, max_new_tokens=args.max_new,
                                     do_sample=False)
            desc = tok.decode(gen[0][inp.input_ids.shape[1]:],
                              skip_special_tokens=True).strip()
            full = dmsg + [{"role": "assistant", "content": desc},
                           {"role": "user", "content": ask}]
            ftext = proc.apply_chat_template(full, tokenize=False,
                                             add_generation_prompt=True)
            pb = read(ftext + '{"' + f + '": ', im)
            spent["B"] += time.time() - t0

            # ---- arm C: six cumulative binary questions ---------------------
            t0 = time.time()
            ge = []
            for k in range(2, 8):
                q = ("Look at this Manhattan street view. Is there AT LEAST as "
                     "much as this describes?\n\n\"%s\"\n\nReply with one JSON "
                     "object and nothing else: {\"at_least\": \"yes\"} or "
                     "{\"at_least\": \"no\"}." % SCALE[f][k - 1])
                cm = [{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": [{"type": "image"},
                                                   {"type": "text", "text": q}]}]
                ct = proc.apply_chat_template(cm, tokenize=False,
                                              add_generation_prompt=True)
                ge.append(read_yes(ct + '{"at_least": "', im))
            spent["C"] += time.time() - t0

            rec = dict(file=r.file, field=f, typology=r.typology, desc=desc)
            rec.update({"A_p%d" % (k + 1): pa[k] for k in range(7)})
            rec.update({"B_p%d" % (k + 1): pb[k] for k in range(7)})
            rec.update({"C_ge%d" % (k + 2): ge[k] for k in range(6)})
            rec["C_score"] = 1.0 + float(np.sum(ge))
            # a coherent answer set is non-increasing in k: anything at least
            # as good as rung 6 is at least as good as rung 3.
            rec["C_coherent"] = bool(all(ge[i] >= ge[i + 1] - 1e-9
                                         for i in range(5)))
            rows.append(rec)

    d = pd.DataFrame(rows)
    out = RES / "tables" / "vlm_describe_probe.csv"
    d.to_csv(out, index=False)

    def stats(P):
        H = -(P * np.log(P + 1e-12)).sum(1) / np.log(7)
        s = np.sort(P, 1)[:, ::-1]
        return (H.mean(), (s[:, 0] - s[:, 1]).mean(),
                (s[:, 0] - s[:, 1] < 0.05).mean() * 100,
                (s[:, 0] > 0.5).mean() * 100, (P * ks).sum(1).std())

    def faithful(P):
        """Does the reported integer land on a rung the model endorses?

        The seven bins are interpretable only if the number we store means what
        its prose says. Under the current readout it often does not: round(EV)
        on a diffuse or two-peaked distribution can land on a rung ranked below
        the runner-up, so a reader looking up rung 3 reads a sentence the model
        was arguing against. Reported as the share where the stored rung IS the
        model's top choice, and the share where it is not even top two.
        """
        st = np.clip(np.round((P * ks).sum(1)), 1, 7).astype(int)
        top = P.argmax(1) + 1
        s = np.sort(P, 1)[:, ::-1]
        ps = P[np.arange(len(P)), st - 1]
        return (st == top).mean() * 100, (ps < s[:, 1]).mean() * 100

    print()
    for f in FIELDS_TESTED:
        g = d[d.field == f]
        A = g[["A_p%d" % k for k in range(1, 8)]].to_numpy()
        B = g[["B_p%d" % k for k in range(1, 8)]].to_numpy()
        print("  %s   n=%d" % (f, len(g)))
        print("    %-12s%9s%9s%10s%10s%9s%10s%9s"
              % ("", "entropy", "margin", "near-tie", "decisive", "sd(EV)",
                 "=argmax", "valley"))
        for nm, P in (("A commit", A), ("B describe", B)):
            h, m, t_, dec, ev = stats(P)
            fa, va = faithful(P)
            print("    %-12s%9.3f%9.3f%9.1f%%%9.1f%%%9.3f%9.0f%%%8.0f%%"
                  % (nm, h, m, t_, dec, ev, fa, va))
        # C has no seven-way distribution, so entropy/margin do not apply.
        # sd is the comparable column: it is what "can this tell streets
        # apart" means for a continuous score.
        # C endorses named rungs directly, so "does the reported integer land
        # on a rung the model endorses" is asked differently: is the rounded
        # score a threshold it actually cleared?
        r = np.clip(np.round(g.C_score), 1, 7).astype(int)
        ge = g[["C_ge%d" % k for k in range(2, 8)]].to_numpy()
        end = np.mean([1.0 if rr <= 1 else ge[i, min(rr, 7) - 2] > 0.5
                       for i, rr in enumerate(r)]) * 100
        print("    %-12s%9s%9s%10s%10s%9.3f%9.0f%%%8s"
              % ("C anchors", "-", "-", "-", "-", g.C_score.std(), end, "-"))
        print("      C: mean %.2f  range %.2f-%.2f  coherent on %.0f%% of images"
              % (g.C_score.mean(), g.C_score.min(), g.C_score.max(),
                 g.C_coherent.mean() * 100))
        print()

    n = max(len(d), 1)
    print("  timing over %d field-calls:" % n)
    print("    A commit    %5.0f s   %4.0f ms per call"
          % (spent["A"], spent["A"] / n * 1000))
    print("    B describe  %5.0f s   %4.0f ms per call   -> %.1fx"
          % (spent["B"], spent["B"] / n * 1000,
             spent["B"] / max(spent["A"], 1e-9)))
    print("    C anchors   %5.0f s   %4.0f ms per call   -> %.1fx"
          % (spent["C"], spent["C"] / n * 1000,
             spent["C"] / max(spent["A"], 1e-9)))
    calls = 3064 * 10
    print("\n  extrapolated to all %s field-calls in a full run:" % f"{calls:,}")
    print("    A  %.1f h     B  %.1f h     C  %.1f h"
          % (spent["A"] / n * calls / 3600, spent["B"] / n * calls / 3600,
             spent["C"] / n * calls / 3600))
    print("\n  wrote %s -- the descriptions are in it, read some" % out)


if __name__ == "__main__":
    main()
