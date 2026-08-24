"""Two measurements over the 180-degree panoramas, one pass, no wiring.

Neither output is a detector and neither may be reported. Both exist so that
a scaffolding detector can be *scored* against the visual labels rather than
against DOB permits, which were measured at chance (docs/handpicking.md).

  1. OWLv2 open-vocabulary detection, one max score per text prompt per
     image. CLIPSeg scored AUC 0.55 here and is from 2022; this asks whether
     a current open-vocabulary model does better on the same question.
     `street tree` rides along as the positive control CLAUDE.md requires --
     if that column stops separating, the harness broke, not the detector.

  2. DINOv2 embeddings, for a linear probe on the visual labels. A frozen
     backbone plus logistic regression needs far fewer labels than
     fine-tuning and cannot quietly memorise the permit geometry, because
     the permits are never shown to it.

Both are computed on four crops -- the three thirds of the panorama and the
whole -- and the per-prompt score is the max across them. A sidewalk shed
occupies a fraction of a 180-degree view, and a model fed the whole strip
squashed to its input size sees it at a few dozen pixels. Thirds also keep
the embeddings from averaging a shed away against 120 degrees of empty
street.

Scores come straight off the logits rather than the post-processing helper,
whose name has moved between transformers versions; sigmoid of the max
logit per query is the same rank statistic and is stable across releases.

    .venv-gpu/Scripts/python tools/svi_180_probe_features.py --limit 8
    .venv-gpu/Scripts/python tools/svi_180_probe_features.py
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import (AutoImageProcessor, AutoModel, Owlv2ForObjectDetection,
                          Owlv2Processor)

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import PROC, RES, banner

OWL = "google/owlv2-base-patch16-ensemble"
DINO = "facebook/dinov2-large"

# Phrasings, not synonyms: an open-vocabulary model is sensitive to the exact
# noun phrase, so the same structure is asked for several ways and the columns
# are kept separate rather than averaged. Which phrasing wins is itself a
# result. `street tree` is the control and is not a scaffolding prompt.
PROMPTS = [
    "a sidewalk shed",
    "scaffolding",
    "construction scaffolding on a building facade",
    "a green wooden shed over the sidewalk",
    "metal scaffold poles",
    "a construction fence",
    "a plywood construction barrier",
    "a street tree",
]

NAME_RE = re.compile(r"(\d+)_(n\d+)_([NESW])\.jpg$")


def _crops(img: Image.Image) -> list[Image.Image]:
    """Three thirds plus the whole panorama."""
    w, h = img.size
    third = w // 3
    return [img.crop((i * third, 0, min((i + 1) * third, w), h))
            for i in range(3)] + [img]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/raw/svi_180"))
    ap.add_argument("--table", type=Path,
                    default=RES / "tables" / "svi_180_openvocab.csv")
    ap.add_argument("--emb", type=Path, default=PROC / "svi_180_dinov2.npz")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--restart", action="store_true")
    args = ap.parse_args()
    banner("open-vocabulary scores and dinov2 embeddings")

    files = []
    for jpg in sorted(args.src.rglob("*.jpg")):
        m = NAME_RE.search(jpg.name)
        if m:
            files.append((str(jpg.relative_to(args.src)).replace("\\", "/"), jpg,
                          m.group(2), m.group(3), int(m.group(1))))
    if not files:
        sys.exit(f"no panoramas under {args.src}")
    print(f"{len(files)} panoramas")

    done, emb_done = pd.DataFrame(), {}
    if args.table.exists() and not args.restart:
        done = pd.read_csv(args.table)
        if args.emb.exists():
            emb_done = {k: v for k, v in np.load(args.emb).items()}
        seen = set(done.file)
        files = [f for f in files if f[0] not in seen]
        print(f"{len(done)} already scored, {len(files)} to do")
    if args.limit:
        files = files[:args.limit]
    if not files:
        print("nothing to do")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}")
    print(f"loading {OWL}")
    owl_p = Owlv2Processor.from_pretrained(OWL)
    owl = Owlv2ForObjectDetection.from_pretrained(OWL).to(device).eval()
    print(f"loading {DINO}")
    dino_p = AutoImageProcessor.from_pretrained(DINO)
    dino = AutoModel.from_pretrained(DINO).to(device).eval()

    rows, embs, since = [], dict(emb_done), 0
    for rel, path, node_id, cardinal, seq in tqdm(files, desc="panoramas",
                                                  mininterval=2.0):
        img = Image.open(path).convert("RGB")
        crops = _crops(img)

        with torch.no_grad():
            oi = owl_p(text=[PROMPTS] * len(crops), images=crops,
                       return_tensors="pt").to(device)
            # (crops, patches, queries) -> best patch per query, per crop.
            logits = owl(**oi).logits.sigmoid().amax(dim=1)
            best = logits.amax(dim=0).float().cpu().numpy()
            which = logits.argmax(dim=0).cpu().numpy()

            di = dino_p(images=crops, return_tensors="pt").to(device)
            # CLS token: one vector per crop, kept separately so a later probe
            # can pool them however it likes.
            vec = dino(**di).last_hidden_state[:, 0].float().cpu().numpy()

        rec = {"file": rel, "node_id": node_id, "cardinal": cardinal, "seq": seq,
               "street": path.parent.parent.name, "direction": path.parent.name}
        for i, p in enumerate(PROMPTS):
            key = p.replace(" ", "_")
            rec[key] = round(float(best[i]), 5)
            # 3 == the whole strip; 0-2 are left, middle, right thirds.
            rec[key + "__crop"] = int(which[i])
        rows.append(rec)
        embs[rel] = vec.astype(np.float32)

        since += 1
        if since >= 100:
            pd.concat([done, pd.DataFrame(rows)], ignore_index=True).to_csv(
                args.table, index=False)
            np.savez_compressed(args.emb, **embs)
            since = 0

    out = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    out = out.sort_values(["street", "direction", "seq"])
    args.table.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.table, index=False)
    np.savez_compressed(args.emb, **embs)

    print(f"\n{len(out)} rows -> {args.table}")
    print(f"{len(embs)} embeddings {vec.shape} -> {args.emb}")
    print("\nmax score by prompt (rank statistic only, NOT a detection rate):")
    for p in PROMPTS:
        c = out[p.replace(" ", "_")]
        print(f"  {p:<44}median {c.median():.4f}  p95 {c.quantile(.95):.4f}")


if __name__ == "__main__":
    main()
