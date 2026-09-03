"""Ask the model to say what it sees, in its own words.

The ratings are read from logits at a forced prefix -- the model never writes a
token, it only reveals how it would have distributed one. That is the right
instrument for a number: it cannot ramble, cannot refuse, cannot be talked into
a different answer by its own preamble, and it costs one forward pass.

It also means nobody has ever heard the model explain itself, which is a real
gap when the numbers go in front of people who reasonably ask what the model
was looking at. This asks in open text instead, on a sample rather than on
everything, and keeps the answers beside the ratings so the two can be read
together.

WHAT THIS IS NOT. It is not validation and must not be quoted as any. A model
asked to justify a rating will produce a fluent justification whether or not
that reasoning drove the rating -- the text is generated after the fact and by
a different decoding path than the logit read. Treat it as an illustration of
what the imagery contains, and keep the measured twins in
tools/validation_figure.py as the evidence.

SAMPLED ACROSS THE RANGE, not at random. A random draw of a hundred frames from
Murray Hill is a hundred mid-scoring streets; stratifying by M means the extremes
-- the ones anybody will ask about -- are actually in the sample.

    .venv-gpu/Scripts/python tools/sim_vlm_describe.py --src data/raw/svi_180 \
        --mast-set svi_90_wide --n 90
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
from common import RES, banner
from mast import erase_mast

MODEL = "Qwen/Qwen2-VL-7B-Instruct"
MAX_PIXELS = 1024 * 28 * 28
NAME = re.compile(r"(\d+)_(n\d+)_([NESW])(?:_([LRF]))?\.jpg$")

# Every constraint here answers something the first test did. Two sentences,
# because ground and frontage ran to numbered lists and truncated mid-word.
# No text on signs or vehicles, because frontage was answered with the logo on
# a parked truck. No season or purpose, because bare trees drew "likely late
# autumn or winter" from imagery captured in April.
SYSTEM = ("You are describing a street for an urban design study. Answer in at "
          "most two sentences. Describe only what is visible in the image. Do "
          "not read out text on signs or vehicles, do not name companies, and "
          "do not speculate about the season, the weather, the purpose of a "
          "building, or the neighbourhood. If something asked about is absent, "
          "say so plainly.")

# Open questions, deliberately not the rung prompts. Asking "why did you say 5"
# invites the model to reverse-engineer a justification; asking what is there
# gets a description that can be checked against the photograph.
QUESTIONS = {
    "scene": "What is it like to walk down this street?",
    # NOT ENUMERATED, and the change is measured. The earlier wording listed
    # the options -- "street trees, planting at ground level, greenery on the
    # buildings, or none" -- and the model answered the LIST instead of the
    # image: on 15 vegetation-free frames it claimed greenery on the buildings
    # 5 times with none present, and on 4 frames with real planters it replied
    # "no visible greenery", because a potted plant fits none of the offered
    # categories and "none" does. The open wording made no error on the same
    # 30 frames. tools/describe_prompt_ab.py is the test; the crops are in
    # results/tables/describe_prompt_ab.csv.
    "greenery": "What vegetation is visible in this view, and where?",
    "ground": "Describe the footway only: how wide it is, its surface, whether "
              "there is a kerb, and what stands on it. Ignore the roadway and "
              "ignore vehicles.",
    "frontage": "Describe the buildings at street level only -- entrances, "
                "windows, shopfronts, or blank wall. Ignore vehicles, people "
                "and anything in the road.",
    "standout": "What is the single most noticeable thing in this view?",
}


def _write(rows, prior, table):
    """The new rows plus whatever the table already held, never just the new.

    THE TABLE CANNOT SHRINK. Hours of generation live in this file and it has
    been destroyed twice -- once by a restarted stage rewriting from row one,
    once by an exit path that skipped the merge. Both had the same signature:
    a write smaller than the file it replaced. So that signature is now
    checked at the last moment before the bytes move, where no bug upstream
    can argue with it: a shrinking write is diverted to <table>.rej and the
    table is left alone. Shrinking on purpose (dropping rows) is done by the
    tools that own the table, not through this function.
    """
    d = pd.DataFrame(rows)
    if prior is not None and len(prior):
        d = pd.concat([prior, d], ignore_index=True)
        d = d.drop_duplicates(subset="file", keep="last")
    table.parent.mkdir(parents=True, exist_ok=True)
    if table.exists():
        try:
            existing = sum(1 for _ in open(table, encoding="utf-8")) - 1
        except OSError:
            existing = 0
        if len(d) < existing:
            rej = table.with_suffix(table.suffix + ".rej")
            d.to_csv(rej, index=False)
            print(f"REFUSED to shrink {table.name}: {existing} rows on disk, "
                  f"{len(d)} about to be written. New rows saved to "
                  f"{rej.name}; the table is untouched.")
            return d
    d.to_csv(table, index=False)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/raw/svi_180"))
    ap.add_argument("--calc", type=Path, default=None,
                    help="vlm_calculations.csv, to stratify the sample by M")
    ap.add_argument("--table", type=Path, default=None)
    ap.add_argument("--n", type=int, default=90)
    ap.add_argument("--all", action="store_true",
                    help="describe every frame rather than a sample, which is "
                         "what a walk-through interface needs")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--mast-set", default=None)
    ap.add_argument("--max-new", type=int, default=110)
    ap.add_argument("--fields", nargs="+", default=list(QUESTIONS))
    ap.add_argument("--resume", action="store_true",
                    help="keep what the table already holds and describe only "
                         "the frames missing from it")
    ap.add_argument("--checkpoint", type=int, default=10)
    args = ap.parse_args()
    banner("what the model says it sees")

    files = sorted(args.src.rglob("*.jpg"))
    rows = []
    for p in files:
        m = NAME.search(p.name)
        if m:
            rows.append({"file": str(p.relative_to(args.src)).replace("\\", "/"),
                         "path": p, "node_id": m.group(2)})
    fl = pd.DataFrame(rows)
    print(f"{len(fl)} frames in {args.src}")

    # stratify by M so the sample spans the range rather than clustering at the
    # median, where a random draw would put nearly all of it
    calc = args.calc or (RES / "tables" / "vlm_calculations.csv")
    if calc.exists():
        c = pd.read_csv(calc)
        mcol = "M_noA" if "M_noA" in c.columns else "M"
        per = c.groupby("node_id")[mcol].mean().rename("M").reset_index()
        fl = fl.merge(per, on="node_id", how="left")
    else:
        fl["M"] = np.nan
        print(f"no {calc.name}: M will be blank")

    if args.all:
        # Every frame. A walk-through interface shows a verdict at each step,
        # so a sample leaves the sidebar empty most of the way down a street.
        print(f"describing all {len(fl)} frames")
    else:
        # Stratified by M, so the sample spans the range instead of clustering
        # at the median where a random draw would put nearly all of it.
        ok = fl.M.notna()
        if ok.any():
            band = pd.qcut(fl.loc[ok, "M"], 5, labels=False, duplicates="drop")
            per_band = max(1, args.n // max(int(band.nunique()), 1))
            # index-based rather than groupby.apply: pandas 2 drops the
            # grouping column from the result, which then cannot be reported
            take = []
            for b in sorted(band.dropna().unique()):
                idx = band.index[band == b]
                k = min(per_band, len(idx))
                take.extend(pd.Series(idx).sample(k, random_state=args.seed))
            fl = fl.loc[take]
            print(f"sampled {len(fl)} frames across {int(band.nunique())} "
                  f"bands of M ({fl.M.min():.3f} to {fl.M.max():.3f})")
        elif len(fl) > args.n:
            fl = fl.sample(args.n, random_state=args.seed)

    # RESUME RATHER THAN REWRITE. Without this the pass starts from an empty
    # table every time, so anything that restarts the script destroys a
    # finished run: a completed 1,514-frame table was lost exactly that way,
    # rewritten from row one while a later stage was being resumed. Generation
    # is deterministic here, so a frame already described does not need doing
    # twice, and the cost of being wrong about that is only a repeated frame.
    table = args.table or (RES / "tables" / "vlm_descriptions.csv")
    prior = None
    if args.resume and table.exists():
        prior = pd.read_csv(table)
        need = [c for c in args.fields if c in QUESTIONS]
        have = prior.dropna(subset=[c for c in need if c in prior.columns])             if all(c in prior.columns for c in need) else prior.iloc[0:0]
        done = set(have.file.astype(str))
        before = len(fl)
        fl = fl[~fl.file.isin(done)]
        print(f"resuming: {len(done)} frames already described, "
              f"{len(fl)} of {before} left")
        if not len(fl):
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
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL, quantization_config=qcfg, device_map="cuda").eval()

    mast_set = args.mast_set or args.src.name
    if str(mast_set).lower() == "none":
        mast_set = None
        print("mast erase: DISABLED")
    else:
        print(f"mast erase: calibration set {mast_set!r}")

    out, ask = [], [(k, QUESTIONS[k]) for k in args.fields if k in QUESTIONS]
    for i, r in enumerate(tqdm(list(fl.itertuples()), desc="frames",
                               mininterval=10.0), 1):
        im = Image.open(r.path).convert("RGB")
        if mast_set:
            im, _ = erase_mast(im, mast_set)
        rec = {"file": r.file, "node_id": r.node_id,
               "M": getattr(r, "M", np.nan)}
        for key, q in ask:
            msg = [{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": [{"type": "image"},
                                                {"type": "text", "text": q}]}]
            text = proc.apply_chat_template(msg, tokenize=False,
                                            add_generation_prompt=True)
            inp = proc(text=[text], images=[im], return_tensors="pt").to("cuda")
            with torch.no_grad():
                gen = model.generate(**inp, max_new_tokens=args.max_new,
                                     do_sample=False)
            reply = proc.batch_decode(
                gen[:, inp["input_ids"].shape[1]:],
                skip_special_tokens=True)[0].strip().replace("\n", " ")
            rec[key] = reply
        out.append(rec)
        if i % args.checkpoint == 0:
            table.parent.mkdir(parents=True, exist_ok=True)
            _write(out, prior, table)

    # THE FINAL WRITE GOES THROUGH _write TOO. This used to be a bare
    # to_csv of the run's own rows, so every run ENDED by discarding
    # whatever the table already held -- the checkpoints merged with
    # prior and the last write threw prior away. It destroyed a
    # 1,254-row table twice before being caught: the resume machinery
    # is worthless if the exit path does not use it.
    d = _write(out, prior, table)
    print(f"\nwrote {table}  ({len(d)} rows now in the table)")

    if "M" in d.columns and d.M.notna().any():
        lo = d.nsmallest(1, "M").iloc[0]
        hi = d.nlargest(1, "M").iloc[0]
        for lab, r in (("lowest M", lo), ("highest M", hi)):
            print(f"\n  {lab} = {r.M:.3f}   {r.file}")
            for key, _ in ask[:3]:
                print(f"    {key:<10}{str(r[key])[:150]}")


if __name__ == "__main__":
    main()
