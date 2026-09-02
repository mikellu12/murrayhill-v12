"""Rate the nine SIM fields, one field per call, batched per image.

Sampling follows the manuscript: a dual-directional hemispheric field of view,
forward and backward along the street vector, which is what data/raw/svi_180
already contains -- two walks per street, one 180-degree render each.

WHY NINE CALLS AND NOT ONE. The twelve-field schema this replaces collapsed:
vertical_greenery answered 1 on 90 per cent of images, enclosure 7 on 89 per
cent, and every field in a family returned the identical correlation because
the model was writing one number into every slot. Asked one field at a time
with both scale poles named, green_eye_level moved from rho +0.403 to +0.787
against the greenery measured in the same frame. The schema was not diluting
attention across twelve questions; it was collapsing them into one.

WHY BATCHED. Nine sequential calls cost 8.30 s per image against 4.43 s for
the single schema call, almost all of it re-encoding the same image nine
times. Running the nine prompts as one batch costs 5.18 s and returns answers
identical to the sequential version -- verified on 72 of 72 answers, which is
the check that matters, because padded batched attention can shift logits.

Five of the nine have a measured counterpart in the profiles. Those are not
redundant: they are the only evidence that the four judgement fields, which
have no measured counterpart at all and carry the larger share of the index,
come from a model that rates this imagery sensibly.

    .venv-gpu/Scripts/python tools/sim_vlm_run.py --sample 24   # smoke test
    .venv-gpu/Scripts/python tools/sim_vlm_run.py
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
from sim_fields import FIELDS, SYSTEM, prompt
from sim_scale import prompt7
from mast import erase_mast

MODEL = "Qwen/Qwen2-VL-7B-Instruct"
MAX_PIXELS = 1024 * 28 * 28
NAME_180 = re.compile(r"(\d+)_(n\d+)_([NESW])\.jpg$")
# L and R are the two halves of a vehicular street; F is the single wide strip
# a pedestrian way gets instead. See tools/export_svi_90.py.
NAME_90 = re.compile(r"(\d+)_(n\d+)_([NESW])_([LRF])\.jpg$")
ORDER = list(FIELDS)


def one(txt, field):
    m = re.search(r"\{.*\}", txt, re.S)
    if m:
        try:
            v = json.loads(m.group()).get(field)
            if isinstance(v, dict):
                v = v.get("score", v.get("value"))
            return float(v)
        except Exception:
            pass
    m = re.search(r"\b([1-7])\b", txt)
    return float(m.group(1)) if m else np.nan


def dist(d, field):
    """The stored distribution as an n x 7 array of probabilities."""
    return d[[f"{field}_p{k}" for k in range(1, 8)]].to_numpy(float)


def column_order(fields):
    """Ratings first, then EV, then argmax, then the probabilities.

    The 63 probability columns are ordinary columns, just placed last, so
    opening the file shows the nine ratings without scrolling and the
    distributions sit out of the way on the right.
    """
    ident = ["file", "street", "walk", "seq", "node_id", "cardinal", "side"]
    return (ident + list(fields)
            + [f + "_ev" for f in fields]
            + [f + "_argmax" for f in fields]
            + [f"{f}_p{k}" for f in fields for k in range(1, 8)])


def index(src):
    rows = []
    for jpg in sorted(Path(src).rglob("*.jpg")):
        m = NAME_90.search(jpg.name)
        side = m.group(4) if m else None
        if not m:
            m = NAME_180.search(jpg.name)
        if not m:
            continue
        rows.append({"file": str(jpg.relative_to(src)).replace("\\", "/"),
                     "src_path": jpg, "street": jpg.parent.parent.name,
                     "walk": jpg.parent.name, "seq": int(m.group(1)),
                     "node_id": m.group(2), "cardinal": m.group(3),
                     "side": side or "W"})
    return pd.DataFrame(rows)


def main():
    global ORDER          # --fields narrows it; must precede any use below
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/raw/svi_180"))
    ap.add_argument("--mast-set", default=None,
                    help="mast calibration set (default: the --src folder "
                         "name); 'none' disables the erase")
    ap.add_argument("--table", type=Path, default=RES / "tables" / "sim_vlm.csv")
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--checkpoint", type=int, default=25)
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--anchors", default="7", choices=["2", "7"],
                    help="2 names only the ends of the scale; 7 names every rung")
    ap.add_argument("--fields", nargs="+", default=None,
                    help="rerun only these fields; the output carries just "
                         "them plus identity, for merging into a full table")
    ap.add_argument("--show-prompts", action="store_true")
    args = ap.parse_args()

    if args.show_prompts:
        # take `ask` from the same expression the run uses, so this can never
        # print one prompt while the model receives another
        ask = prompt7 if args.anchors == "7" else prompt
        banner(f"the nine prompts as sent, --anchors {args.anchors}")
        print("SYSTEM\n" + SYSTEM + "\n")
        for f in ORDER:
            twin = FIELDS[f][3] or "no measured counterpart"
            print("-" * 72)
            print(f"{f}   [{FIELDS[f][2]}]   twin: {twin}\n")
            print(ask(f) + "\n")
            print(f'  ...then the assistant turn is prefixed {{"{f}": ')
            print("  and the distribution over the next token is read.\n")
        return

    if args.fields:
        bad = [f for f in args.fields if f not in FIELDS]
        if bad:
            sys.exit(f"unknown field(s): {bad}")
        ORDER = list(args.fields)
        print(f"rerunning {len(ORDER)} field(s) only: {', '.join(ORDER)}")
    banner("SIM ratings, one field per call, batched per image")
    fl = index(args.src)
    if fl.empty:
        sys.exit(f"no images under {args.src}")
    print(f"{len(fl)} images, {fl.node_id.nunique()} nodes, "
          f"{fl.street.nunique()} streets, walks {sorted(fl.walk.unique())}")

    done = pd.DataFrame()
    if args.table.exists() and not args.restart:
        done = pd.read_csv(args.table)
        fl = fl[~fl.file.isin(set(done.file))]
        print(f"{len(done)} already done, {len(fl)} remaining")
    if args.sample:
        met = pd.read_csv(PROC / "metrics.csv")[["node_id", "GVI"]]
        fl = fl.merge(met, on="node_id", how="left").dropna(subset=["GVI"])
        fl["q"] = pd.qcut(fl.GVI, 4, labels=False, duplicates="drop")
        per = max(1, args.sample // max(1, fl.q.nunique()))
        fl = pd.concat([g.sample(min(len(g), per), random_state=args.seed)
                        for _, g in fl.groupby("q")]).head(args.sample)
        print(f"stratified smoke test: {len(fl)} images, "
              f"GVI {fl.GVI.min():.1f} to {fl.GVI.max():.1f}")
    if fl.empty:
        print("nothing to do")
        return
    print(f"{len(fl)} images x {len(ORDER)} fields = {len(fl)*len(ORDER)} calls\n")

    import torch
    from PIL import Image
    from tqdm.auto import tqdm
    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              Qwen2VLForConditionalGeneration)
    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True)
    proc = AutoProcessor.from_pretrained(MODEL, max_pixels=MAX_PIXELS)
    # left padding or the batched rows decode from the wrong offset
    proc.tokenizer.padding_side = "left"
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL, quantization_config=qcfg, device_map="cuda").eval()

    # The answer prefix is forced, so the very next token is the digit and its
    # whole distribution is readable at one position. generate() would take the
    # argmax of exactly this and discard the other six numbers -- verified
    # identical on 6 of 6 images -- so nothing is lost by reading it here, and
    # the six survive.
    ask = prompt7 if args.anchors == "7" else prompt
    texts = []
    for f in ORDER:
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": ask(f)}]}]
        texts.append(proc.apply_chat_template(msgs, tokenize=False,
                                              add_generation_prompt=True)
                     + f'{{"{f}": ')
    tok = proc.tokenizer
    ids = [tok.encode(str(k), add_special_tokens=False)[0] for k in range(1, 8)]
    ks = np.arange(1, 8)

    WIDE_SET = "svi_180"   # the 2880x1833 strip
    mast_set = args.mast_set or args.src.name
    if str(mast_set).lower() == "none":
        mast_set = None
        print("mast erase: DISABLED")
    else:
        print(f"mast erase: calibration set {mast_set!r}")

    out = []
    for r in tqdm(list(fl.itertuples()), desc="images", mininterval=10.0):
        im = Image.open(r.src_path).convert("RGB")
        # Erase the Google camera mast before the model sees it. The
        # segmentation arm can exclude the mast from both numerator and
        # denominator and leave the pixels alone; a VLM has no such option --
        # it reads whatever is in frame, and it read the mast as a pole and
        # the "Google" wordmark on it as signage. Erasing is the only way to
        # keep it out of a rating.
        if mast_set:
            # per image, not per folder: a wide strip and a 90 half sit side
            # by side in the same tree and need different width fractions
            im, _ = erase_mast(im, WIDE_SET if r.side == "F" else mast_set)
        inp = proc(text=texts, images=[im] * len(ORDER), padding=True,
                   return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = model(**inp).logits[:, -1, :].float()
        rec = {"file": r.file, "street": r.street, "walk": r.walk,
               "seq": r.seq, "node_id": r.node_id, "cardinal": r.cardinal,
               "side": r.side}
        for i, f in enumerate(ORDER):
            p = torch.softmax(logits[i, ids], -1).cpu().numpy()
            p = p / p.sum()
            ev = float((p * ks).sum())
            # `f` is the survey answer: the rung closest to what the model
            # thinks, not the tallest single bar. The other two reads and the
            # full distribution are kept so this choice can be revisited
            # without a rerun.
            rec[f] = int(np.clip(round(ev), 1, 7))
            rec[f + "_ev"] = ev
            rec[f + "_argmax"] = int(ks[p.argmax()])
            # four decimals: the eventual use is a weighted mean over seven
            # terms, and more precision than that is noise.
            for k in range(7):
                rec[f"{f}_p{k + 1}"] = round(float(p[k]), 4)
        out.append(rec)
        if len(out) % args.checkpoint == 0:
            args.table.parent.mkdir(parents=True, exist_ok=True)
            pd.concat([done, pd.DataFrame(out)], ignore_index=True).to_csv(
                args.table, index=False)

    d = pd.concat([done, pd.DataFrame(out)], ignore_index=True)
    d = d.sort_values(["street", "walk", "seq"])
    d = d.reindex(columns=[c for c in column_order(ORDER) if c in d.columns])
    args.table.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(args.table, index=False)
    print(f"\n{len(d)} rows -> {args.table}\n")

    print(f"  {'field':<23}{'survey':>7}{'sd':>6}{'mode%':>7}"
          f"{'EV mean':>9}{'argmax':>8}{'P(4)':>7}{'agree':>7}")
    for f in ORDER:
        s = d[f].dropna()
        if s.empty:
            print(f"  {f:<23}   all NaN")
            continue
        mo = (s == s.mode().iloc[0]).mean() * 100
        am = d[f + "_argmax"]
        agree = (s == am).mean() * 100
        flag = "   <-- degenerate" if mo > 80 else ""
        print(f"  {f:<23}{s.mean():>7.2f}{s.std():>6.2f}{mo:>6.0f}%"
              f"{d[f + '_ev'].mean():>9.2f}{am.mean():>8.2f}"
              f"{d[f + '_p4'].mean():>7.3f}{agree:>6.0f}%{flag}")
    print("\n  survey = the rung closest to the model's belief, round(EV).")
    print("  agree  = how often that matches the argmax, which is what a")
    print("           generate() call would have written.")
    print(f"\n  every column kept: {len(ORDER)} fields x (survey, EV, argmax,")
    print(f"  p1..p7) = {len(ORDER) * 10} columns, so any other read is")
    print("  derivable without touching the GPU again.")


if __name__ == "__main__":
    main()
