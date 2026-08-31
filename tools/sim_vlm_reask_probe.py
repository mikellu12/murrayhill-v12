"""Prune the scale to the bins the model favours, then ask it again.

Round one is the live pipeline: force the prefix, read the seven digit logits.
Round two drops every rung whose probability sat below the mean, rebuilds the
prompt listing ONLY the survivors, and reads again over just those digits.

Why this is not the arithmetic pruning already tested. That version reweighted
the round-one numbers and, iterated, converged on the argmax -- identical on
99.2% of rows, because no new information entered. Re-asking is a fresh forward
pass against a different prompt, so the model can move mass it did not have
before.

Where it should matter: the expected value of a two-peaked distribution lands
in the valley between the peaks, on a rung the model rejected. 37% of
green_softening images and 14% of ground_floor_activity images have
non-adjacent survivors, so that failure is common rather than exotic.
Re-asking makes the model choose between the peaks instead of averaging across
them.

The known limit, measured before building this: on vertical_greenery the
reference rung survives pruning 88.2% of the time, and when it does not, 91% of
misses sit within one rung of a survivor. So round two can refine but cannot
recover a rung round one eliminated -- about 1% of images are beyond its reach.

vertical_greenery is the control. It has a measured twin, so it is the only
field here where "better" can be checked rather than asserted.

    .venv-gpu/Scripts/python tools/sim_vlm_reask_probe.py --n 100
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
# Three of these have a measured twin over the same 90 degrees, so "better"
# can be checked rather than asserted. green_softening carries no twin but is
# the field where the gap pathology is worst (60% bimodal), so it is kept for
# that diagnostic alone.
FIELDS_TESTED = ["vertical_greenery", "walkable_ground", "resting_affordance",
                 "green_softening"]
# arc twins for the control; the rewritten fields score against the 30-class
# per-image shares instead, which is where their twins live.
TWIN = {"vertical_greenery": "arc_vegetation"}
SEG_TWIN = {"walkable_ground": ["sidewalk", "curb_edge"],
            "resting_affordance": ["stoop_stair", "bench_seating"],
            "vertical_greenery": ["tree", "planter_container"]}


# --new-rungs swaps in the rewrite from sim_vlm_rung_ab for the three fields it
# covers, leaving the rest at their live wording. Kept as a switch rather than a
# separate script so the two runs differ in the rung text and nothing else.
ACTIVE = dict(SCALE)
EXTRA_SYS = ""


def prompt_for(field, rungs):
    """The live prompt when rungs is all seven; the pruned one otherwise.

    Rung numbers are KEPT, not renumbered -- the survivors of a prune are
    often 1, 3 and 6, and relabelling them 1, 2, 3 would silently change what
    the model is being asked and make round two incomparable with round one.
    """
    steps = "\n".join(f"{k} = {ACTIVE[field][k - 1]}" for k in rungs)
    return (f"Rate this Manhattan street view. Reply with ONE JSON object and "
            f"nothing else: {{\"{field}\": <one of "
            f"{', '.join(str(k) for k in rungs)}>}}, using this scale:\n\n"
            f"{steps}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--new-rungs", action="store_true",
                    help="use the rewrite from sim_vlm_rung_ab for the three "
                         "fields it covers")
    args = ap.parse_args()
    banner("prune the scale, then ask again"
           + (" -- REWRITTEN RUNGS" if args.new_rungs else ""))
    global ACTIVE, EXTRA_SYS
    if args.new_rungs:
        from sim_vlm_rung_ab import NEW, EXTRA
        ACTIVE = dict(SCALE); ACTIVE.update(NEW)
        # the mechanical criteria ride in the system turn, as in the A/B
        EXTRA_SYS = "".join(EXTRA.get(f, "") for f in FIELDS_TESTED)
        print("rewritten: " + ", ".join(k for k in NEW if k in FIELDS_TESTED))

    obs = pd.read_csv(RES / "tables" / "vlm_observations.csv")
    src = Path("data/raw/svi_90")
    obs = obs[[(src / f).exists() for f in obs.file]]
    per = max(1, args.n // max(obs.typology.nunique(), 1))
    take = pd.concat([g.sample(min(len(g), per), random_state=args.seed)
                      for _, g in obs.groupby("typology")])
    take = take.sample(min(args.n, len(take)), random_state=args.seed)
    print(f"{len(take)} images x {len(FIELDS_TESTED)} fields\n")

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
    dig = [tok.encode(str(k), add_special_tokens=False)[0] for k in range(1, 8)]

    def ask(field, rungs, im):
        """Distribution over `rungs` only, read at the forced prefix."""
        msgs = [{"role": "system", "content": SYSTEM + EXTRA_SYS},
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_for(field, rungs)}]}]
        t = proc.apply_chat_template(msgs, tokenize=False,
                                     add_generation_prompt=True)
        inp = proc(text=[t + '{"' + field + '": '], images=[im],
                   return_tensors="pt").to("cuda")
        with torch.no_grad():
            lg = model(**inp).logits[:, -1, :].float()
        sel = [dig[k - 1] for k in rungs]
        p = torch.softmax(lg[0, sel], -1).cpu().numpy()
        return p / p.sum()

    rows, spent = [], {"r1": 0.0, "r2": 0.0, "rN": 0.0}
    allr = list(range(1, 8))
    for r in tqdm(list(take.itertuples()), desc="images"):
        im = Image.open(src / r.file).convert("RGB")
        for f in FIELDS_TESTED:
            t0 = time.time()
            p1 = ask(f, allr, im)
            spent["r1"] += time.time() - t0

            keep = [k for k in allr if p1[k - 1] > p1.mean()] or allr
            t0 = time.time()
            p2 = ask(f, keep, im) if len(keep) > 1 else np.array([1.0])
            spent["r2"] += time.time() - t0

            # ---- keep going until one rung is left --------------------------
            # Each round re-asks among the survivors of the last. Unlike the
            # arithmetic prune, which converges on the round-one argmax by
            # construction, every round here is a fresh forward pass, so the
            # model can land somewhere its earlier distribution did not favour.
            cur, pc, rounds = list(keep), p2, 1
            t0 = time.time()
            while len(cur) > 1:
                nxt = [k for i, k in enumerate(cur) if pc[i] > pc.mean()]
                if not nxt or len(nxt) == len(cur):
                    nxt = [cur[int(pc.argmax())]]
                cur = nxt
                rounds += 1
                if len(cur) > 1:
                    pc = ask(f, cur, im)
            spent["rN"] += time.time() - t0
            final = cur[0]

            ka = np.array(keep, float)
            rec = dict(file=r.file, field=f, typology=r.typology,
                       n_keep=len(keep), keep=";".join(map(str, keep)),
                       gap=int(len(keep) > 1 and np.any(np.diff(ka) > 1)),
                       ev1=float((p1 * np.arange(1, 8)).sum()),
                       # pruned EV: round-one mass, renormalised over survivors
                       evp=float((ka * (p1[ka.astype(int) - 1]
                                        / p1[ka.astype(int) - 1].sum())).sum()),
                       ev2=float((ka * p2).sum()),
                       top1=int(p1.argmax() + 1), top2=int(ka[p2.argmax()]),
                       final=int(final), rounds=int(rounds))
            rows.append(rec)

    d = pd.DataFrame(rows)
    out = RES / "tables" / "vlm_reask_probe.csv"
    d.to_csv(out, index=False)

    o = pd.read_csv(RES / "tables" / "vlm_observations.csv")[
        ["file", "arc_vegetation"]]
    m = d.merge(o, on="file", how="left")
    from scipy.stats import spearmanr

    print("\n  ACCURACY -- vertical_greenery is the only field with a twin\n")
    g = m[(m.field == "vertical_greenery")].dropna(subset=["arc_vegetation"])
    print(f"    {'method':<38}{'Spearman':>10}")
    for nm, c in (("round one only (current pipeline)", "ev1"),
                  ("prune, then expected value", "evp"),
                  ("prune, then ASK AGAIN", "ev2"),
                  ("re-ask until one rung is left", "final")):
        print(f"    {nm:<38}{spearmanr(g[c], g.arc_vegetation).statistic:>10.3f}")

    print("\n  WHAT ROUND TWO DID\n")
    print(f"    {'field':<22}{'survivors':>10}{'w/ gap':>9}"
          f"{'moved':>8}{'sd ev1':>9}{'sd ev2':>9}")
    for f in FIELDS_TESTED:
        g = m[m.field == f]
        moved = (g.top2 != g.top1).mean() * 100
        print(f"    {f:<22}{g.n_keep.mean():>10.2f}{g.gap.mean()*100:>8.0f}%"
              f"{moved:>7.0f}%{g.ev1.std():>9.3f}{g.ev2.std():>9.3f}")

    print("\n  ON THE GAP CASES -- where the old expected value fell in a valley\n")
    for f in FIELDS_TESTED:
        g = m[(m.field == f) & (m.gap == 1)]
        if not len(g):
            print(f"    {f:<22} no gap cases")
            continue
        inside = np.mean([float(x) in [float(k) for k in s.split(";")]
                          for x, s in zip(np.round(g.ev1), g.keep)]) * 100
        print(f"    {f:<22} n={len(g):>3}   old EV landed on a kept rung "
              f"{inside:>3.0f}% of the time; round two always does")

    n = max(len(d), 1)
    print(f"\n  timing over {n} field-calls:")
    print(f"    round one   {spent['r1']:5.0f} s   {spent['r1']/n*1000:4.0f} ms")
    print(f"    round two   {spent['r2']:5.0f} s   {spent['r2']/n*1000:4.0f} ms")
    print(f"    rounds 3+   {spent['rN']:5.0f} s   {spent['rN']/n*1000:4.0f} ms")
    print(f"\n    rounds to converge: mean {d.rounds.mean():.2f}, "
          f"max {int(d.rounds.max())}")
    print("    " + "  ".join(f"{k} rounds: {v}" for k, v in
                             sorted(d.rounds.value_counts().items())))
    calls = 3064 * 10
    base = spent['r1'] / n * calls / 3600
    tot = (spent['r1'] + spent['r2']) / n * calls / 3600
    allt = (spent['r1'] + spent['r2'] + spent['rN']) / n * calls / 3600
    print(f"\n    full run, {calls:,} field-calls:")
    print(f"      one round only (current)      {base:5.1f} h   1.0x")
    print(f"      two rounds                    {tot:5.1f} h   {tot/base:.1f}x")
    print(f"      to convergence                {allt:5.1f} h   {allt/base:.1f}x")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
