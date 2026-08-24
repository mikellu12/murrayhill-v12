"""Put a human in front of thirty tiles, blind, and measure the label itself.

Every number in the scaffolding comparison is scored against verdicts a
vision-language model wrote. Qwen2.5-VL then scored 0.935 against them --
which is partly a measurement of scaffolding and partly a measurement of how
much two models agree. Nothing in the pipeline can separate those, because
nothing in the pipeline has ever been checked by eye.

This builds the check. A stratified thirty, half labelled positive and half
negative, shuffled, printed at review-sheet size with a stable index and NO
verdict, permit column or filename hint visible. You write Y or N against
each index; --score then joins your answers to the stored labels and reports
agreement, Cohen's kappa, and which direction the disagreements run.

Thirty is not a sample size for an AUC. It is a sample size for finding out
whether the labels are trustworthy at all, which is the question blocking
everything downstream. A kappa near 0.8 says the VLM labels can stand in for
human judgement; near 0.4 says the 0.935 is largely two models agreeing with
each other.

Blind by construction: the sheet is built from a shuffled order and the key
is written to a separate file this tool does not print.

    .venv/Scripts/python tools/svi_180_spotcheck.py            # build sheets
    .venv/Scripts/python tools/svi_180_spotcheck.py --score    # after filling in
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import PROC, RES, banner

COLS, ROWS, THUMB_W, CAP, PAD = 2, 3, 760, 18, 3


def _font(size):
    for n in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _labels() -> pd.DataFrame:
    out = []
    b = RES / "review_sheets" / "balanced" / "sample_labelled.csv"
    if b.exists():
        d = pd.read_csv(b)
        out.append(d.assign(y=d.vlm.astype(bool))[["file", "node_id", "y"]])
    s = RES / "tables" / "svi_180_visual_labels.csv"
    if s.exists():
        d = pd.read_csv(s)
        out.append(d.assign(y=d.vlm_structure.astype(bool))[["file", "node_id", "y"]])
    if not out:
        sys.exit("no labels found")
    return pd.concat(out, ignore_index=True).drop_duplicates("file")


def build(args):
    t = _labels()
    faces = pd.read_csv(PROC / "nodes_with_faces.csv")[["node_id", "face_id"]]
    t = t.merge(faces, on="node_id", how="left")
    t["face_id"] = t.face_id.fillna(t.node_id)

    rng = np.random.default_rng(args.seed)
    half = args.n // 2
    pick = []
    for val, k in ((True, half), (False, args.n - half)):
        g = t[t.y == val]
        # Spread across faces first, so thirty tiles are not thirty views of
        # one block; the effective sample is faces, not images.
        g = g.sample(frac=1, random_state=args.seed).sort_values(
            "face_id", key=lambda s: s.map(
                {f: i for i, f in enumerate(rng.permutation(g.face_id.unique()))}))
        pick.append(g.groupby("face_id", sort=False).head(
            max(1, k // max(1, g.face_id.nunique()) + 1)).head(k))
    sel = pd.concat(pick).sample(frac=1, random_state=args.seed + 1).reset_index(drop=True)
    sel["idx"] = np.arange(1, len(sel) + 1)

    args.out.mkdir(parents=True, exist_ok=True)
    for p in args.out.glob("sheet_*.jpg"):
        p.unlink()

    font = _font(15)
    probe = Image.open(args.src / sel.file.iloc[0])
    tw = THUMB_W
    th = int(tw * probe.height / probe.width)
    per = COLS * ROWS
    for s0 in range(0, len(sel), per):
        batch = sel.iloc[s0:s0 + per]
        no = s0 // per + 1
        rows = (len(batch) + COLS - 1) // COLS
        sheet = Image.new("RGB", (COLS * (tw + PAD) + PAD,
                                  rows * (th + CAP + PAD) + PAD + 22), "white")
        d = ImageDraw.Draw(sheet)
        d.text((PAD, 5), f"spot-check sheet {no}   write Y or N against each "
                         f"number   items {batch.idx.min()}-{batch.idx.max()}",
               fill="black", font=font)
        for i, r in enumerate(batch.itertuples()):
            x = PAD + (i % COLS) * (tw + PAD)
            y = 22 + PAD + (i // COLS) * (th + CAP + PAD)
            # Index only. No filename, no verdict, no permit column: any of
            # them would anchor the judgement to what is being tested.
            d.text((x + 2, y + 2), f"[{r.idx}]", fill="black", font=font)
            sheet.paste(Image.open(args.src / r.file).resize((tw, th)),
                        (x, y + CAP))
        sheet.save(args.out / f"sheet_{no:02d}.jpg", quality=82, optimize=True)

    sel[["idx"]].assign(human="").to_csv(args.out / "answers.csv", index=False)
    sel[["idx", "file", "node_id", "face_id", "y"]].to_csv(
        args.out / "key.csv", index=False)

    print(f"{len(sel)} tiles across {(len(sel) - 1) // per + 1} sheets, "
          f"{sel.face_id.nunique()} faces")
    print(f"  sheets  {args.out / 'sheet_*.jpg'}")
    print(f"  fill in {args.out}\answers.csv   (write Y or N in the human column)")
    print(f"  key      written to key.csv -- do not open it before answering")
    print(f"\nthen: .venv/Scripts/python tools/svi_180_spotcheck.py --score")


def score(args):
    a = pd.read_csv(args.out / "answers.csv")
    k = pd.read_csv(args.out / "key.csv")
    a["human"] = a.human.astype(str).str.strip().str.upper()
    a = a[a.human.isin(["Y", "N"])]
    if a.empty:
        sys.exit("answers.csv has no Y/N entries yet")
    m = k.merge(a, on="idx")
    m["h"] = m.human.eq("Y")

    n = len(m)
    agree = (m.h == m.y).mean()
    po = agree
    pe = (m.h.mean() * m.y.mean()) + ((1 - m.h.mean()) * (1 - m.y.mean()))
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")

    print(f"{n} of {len(k)} answered, {m.face_id.nunique()} faces\n")
    print(pd.crosstab(m.y, m.h, rownames=["VLM label"],
                      colnames=["human"]).to_string())
    print(f"\nraw agreement   {agree:.1%}")
    print(f"Cohen's kappa   {kappa:.3f}")
    fp = int(((~m.h) & m.y).sum())
    fn = int((m.h & (~m.y)).sum())
    print(f"\nVLM said structure, human disagreed : {fp}")
    print(f"VLM said none, human saw one        : {fn}")
    if kappa >= 0.8:
        print("\nkappa >= 0.8: labels can stand in for human judgement.")
    elif kappa >= 0.6:
        print("\nkappa 0.6-0.8: usable, but the disagreement rate belongs in the paper.")
    else:
        print("\nkappa < 0.6: the labels are not reliable ground truth. Every AUC")
        print("scored against them, including the 0.935, is suspect.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/raw/svi_180"))
    ap.add_argument("--out", type=Path, default=RES / "review_sheets" / "spotcheck")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()
    banner("blind spot-check of the vlm labels")
    (score if args.score else build)(args)


if __name__ == "__main__":
    main()
