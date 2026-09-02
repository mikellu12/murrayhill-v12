"""Rate every half-view by elimination: prune to the rungs above chance, ask
again among the survivors, repeat until one rung is left.

WHY NOT AN EXPECTED VALUE. The seven rungs are ordered categories describing
states of a street, not points on a number line. Averaging them assumes the
step from rung 2 to 3 equals the step from 3 to 4, and that is measurably false
here: each rung corresponds to roughly 2-4x the previous rung's pixel share, so
the scale is multiplicative in physical units and the mean is not defined on it.
Elimination returns a rung the model actually chose, which needs no spacing
assumption at all.

THE PRICE, MEASURED, NOT GUESSED. On the three fields with a measured twin,
converging scored 0.03-0.07 BELOW simply reading round one, and significantly
worse on vertical_hardscape. The last round discards the "it is between 4 and
5" information that a distribution carries. This run buys measurement validity
and pays for it in correlation; that trade was made deliberately.

THE RULE. Keep every rung whose probability exceeds 1/k, k = rungs offered this
round. Since probabilities sum to 1, not all can exceed 1/k, so at least one is
eliminated per round -- convergence is bounded at 6 rounds and observed at 2.3
on average. Rung NUMBERS are preserved when the prompt is rebuilt: survivors
1, 3 and 6 are offered as "one of 1, 3, 6", never renumbered, or round two would
be answering a different question than round one.

ROUND ONE IS RE-RUN RATHER THAN READ FROM sim_vlm_v2.csv. That table batches
ten fields into one padded forward pass; this asks one field at a time, and the
two differ in the fourth decimal. Harmless anywhere except at the prune
boundary -- and 18% of field-calls have a bin within 0.005 of 1/7, where a
shift that small changes which rungs survive. So the whole ladder runs in one
regime.

    .venv-gpu/Scripts/python tools/sim_vlm_converge.py --src data/raw/svi_90
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import CFG, RES, banner
from mast import erase_mast
from sim_fields import SYSTEM, FIELDS
from sim_scale import SCALE, DEFINITION

MODEL = "Qwen/Qwen2-VL-7B-Instruct"
MAX_PIXELS = 768 * 768
ORDER = list(FIELDS)


# Named to the model in every prompt; see config.yaml: prompt_place.
# "Rate this street view." with no place name unless config supplies one; see
# config.yaml: prompt_place for why none is the right default.
_place = CFG.get("prompt_place") or ""
PLACE = f"{_place} " if _place else ""


def prompt_for(field, rungs):
    """The live prompt when rungs is all seven; the pruned one otherwise."""
    steps = "\n".join(f"{k} = {SCALE[field][k - 1]}" for k in rungs)
    d = DEFINITION.get(field)
    d = f"{d} " if d else ""
    if len(rungs) == 7:
        which = "<1-7>"
    else:
        which = "<one of " + ", ".join(str(k) for k in rungs) + ">"
    return (f"Rate this {PLACE}street view. {d}Reply with ONE JSON object "
            f"and nothing else: {{\"{field}\": {which}}}, using this "
            f"scale:\n\n{steps}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/raw/svi_90"))
    ap.add_argument("--table", type=Path,
                    default=RES / "tables" / "sim_vlm_converged.csv")
    ap.add_argument("--checkpoint", type=int, default=25)
    ap.add_argument("--mast-set", default=None,
                    help="mast calibration; defaults to the --src folder "
                         "name. 'none' disables the erase")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    banner("rate by elimination -- prune to above chance, re-ask, repeat")

    imgs = sorted(args.src.rglob("*.jpg"))
    rel = [str(p.relative_to(args.src)).replace("\\", "/") for p in imgs]
    done = pd.DataFrame()
    if args.table.exists():
        done = pd.read_csv(args.table)
        have = set(done.file)
        keep = [(p, r) for p, r in zip(imgs, rel) if r not in have]
        print(f"resuming: {len(have)} already done")
    else:
        keep = list(zip(imgs, rel))
    if args.limit:
        keep = keep[:args.limit]
    print(f"{len(keep)} images x {len(ORDER)} fields = {len(keep)*len(ORDER)} "
          f"ladders\n")
    if not keep:
        print("nothing to do")
        return

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
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_for(field, rungs)}]}]
        t = proc.apply_chat_template(msgs, tokenize=False,
                                     add_generation_prompt=True)
        inp = proc(text=[t + '{"' + field + '": '], images=[im],
                   return_tensors="pt").to("cuda")
        with torch.no_grad():
            lg = model(**inp).logits[:, -1, :].float()
        p = torch.softmax(lg[0, [dig[k - 1] for k in rungs]], -1).cpu().numpy()
        return p / p.sum()

    # Same erase as the single-pass runner. The mast is a camera part, not
    # street, and the model reads it as a pole and its wordmark as signage;
    # leaving it in would make this ladder answer a different question than
    # the pass it is meant to be compared against. Chosen per image because
    # the street-type split puts 90-degree halves and 180-degree strips in one
    # tree and the mast covers a different share of each.
    WIDE_SET = "svi_180"   # the 2880x1833 strip
    mast_set = args.mast_set or args.src.name
    if str(mast_set).lower() == "none":
        mast_set = None
        print("mast erase: DISABLED")
    else:
        print(f"mast erase: calibration set {mast_set!r}")

    out = []
    t0 = time.time()
    for img, r in tqdm(keep, desc="images", mininterval=30.0):
        im = Image.open(img).convert("RGB")
        if mast_set:
            im, _ = erase_mast(im, WIDE_SET if img.stem.endswith("_F")
                               else mast_set)
        rec = {"file": r}
        for f in ORDER:
            cur = list(range(1, 8))
            p = ask(f, cur, im)
            # round one kept in full: it is the only distribution over all
            # seven rungs, and every later round is conditioned on this prune.
            for k in range(7):
                rec[f"{f}_p{k + 1}"] = round(float(p[k]), 4)
            path, rounds = [tuple(cur)], 1
            while len(cur) > 1:
                nxt = [k for i, k in enumerate(cur) if p[i] > p.mean()]
                if not nxt or len(nxt) == len(cur):
                    # every rung at or below chance, or none eliminated: take
                    # the leader rather than loop. Rare, but the loop must end.
                    nxt = [cur[int(p.argmax())]]
                cur = nxt
                path.append(tuple(cur))
                rounds += 1
                if len(cur) > 1:
                    p = ask(f, cur, im)
            rec[f] = int(cur[0])
            rec[f"{f}_rounds"] = rounds
            rec[f"{f}_path"] = "|".join(",".join(map(str, s)) for s in path)
        out.append(rec)
        if len(out) % args.checkpoint == 0:
            args.table.parent.mkdir(parents=True, exist_ok=True)
            pd.concat([done, pd.DataFrame(out)], ignore_index=True).to_csv(
                args.table, index=False)

    d = pd.concat([done, pd.DataFrame(out)], ignore_index=True)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(args.table, index=False)
    el = time.time() - t0
    print(f"\n{len(out)} images in {el/3600:.2f} h "
          f"({el/max(len(out),1):.2f} s/image)")
    print(f"wrote {args.table}\n")
    for f in ORDER:
        v = d[f].value_counts().sort_index()
        print(f"  {f:<24} rungs " + " ".join(f"{k}:{v.get(k,0)}"
                                             for k in range(1, 8))
              + f"   mean rounds {d[f'{f}_rounds'].mean():.2f}")


if __name__ == "__main__":
    main()
