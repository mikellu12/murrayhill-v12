"""A generative VLM's verdict on every panorama, as a score that can be ranked.

The third arm of the comparison. tools/svi_180_probe_features.py measures an
open-vocabulary detector (OWLv2, text-conditioned but not generative) and a
pure-vision backbone (DINOv2, no language at all); this measures a generative
vision-language model, asked the question in words and answering in words.
All three are scored the same way against the same labels, so the paper can
report what the language conditioning buys and what it costs.

The score is P(Yes) from the first answer token, not the generated string. A
yes/no string collapses to 0 or 1 and throws away every distinction AUC needs
-- two images the model is 51% and 99% sure about would tie. Reading the
softmax over the Yes/No tokens keeps the ranking the metric is built on.

One panorama per call, which is NOT how the labels were made: the review
sheets show twelve at once, and a model judging a grid has neighbouring
context an isolated image does not. That difference is a confound between
this arm and the labels, and it is why this is a fresh measurement rather
than a reproduction of the labelling.

Prompt names all four DOB structure types rather than "scaffolding", because
sheds are a minority of what stands on these streets -- facade scaffold 28,
hoarding fence 11, sidewalk shed 7 in the labels so far.

    .venv-gpu/Scripts/python tools/svi_180_vlm_scan.py --limit 8
    .venv-gpu/Scripts/python tools/svi_180_vlm_scan.py
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import RES, banner

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

QUESTION = (
    "This is a 180-degree street-level photograph of a Manhattan street. "
    "Is there any temporary construction structure visible: a sidewalk shed "
    "(an overhead deck pedestrians walk under), a supported scaffold (a frame "
    "up a building facade), a suspended scaffold (a swing stage), or a "
    "construction fence or plywood hoarding? "
    "Permanent features such as ordinary railings, garden fences, chain-link "
    "around a park or ballfield, and retaining walls do NOT count. "
    "Answer with one word, Yes or No."
)

# Capping the vision tokens is what keeps a 1440x916 panorama inside 12 GB
# alongside the weights. Left unset the processor scales to its own maximum
# and the first wide image runs the card out of memory.
MAX_PIXELS = 1024 * 28 * 28

NAME_RE = re.compile(r"(\d+)_(n\d+)_([NESW])\.jpg$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/raw/svi_180"))
    ap.add_argument("--table", type=Path,
                    default=RES / "tables" / "svi_180_vlm.csv")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--restart", action="store_true")
    args = ap.parse_args()
    banner("generative vlm verdict per panorama")

    files = []
    for jpg in sorted(args.src.rglob("*.jpg")):
        m = NAME_RE.search(jpg.name)
        if m:
            files.append((str(jpg.relative_to(args.src)).replace("\\", "/"),
                          jpg, m.group(2), m.group(3), int(m.group(1))))
    if not files:
        sys.exit(f"no panoramas under {args.src}")
    print(f"{len(files)} panoramas")

    done = pd.DataFrame()
    if args.table.exists() and not args.restart:
        done = pd.read_csv(args.table)
        seen = set(done.file)
        files = [f for f in files if f[0] not in seen]
        print(f"{len(done)} already scored, {len(files)} to do")
    if args.limit:
        files = files[:args.limit]
    if not files:
        print("nothing to do")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {MODEL} on {device}")
    proc = AutoProcessor.from_pretrained(MODEL, max_pixels=MAX_PIXELS)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map=device).eval()

    # Both capitalisations, because the tokenizer treats a leading space as
    # part of the token and the chat template may or may not emit one.
    tok = proc.tokenizer
    yes_ids = {i for w in ("Yes", " Yes", "yes", " yes")
               for i in tok.encode(w, add_special_tokens=False)[:1]}
    no_ids = {i for w in ("No", " No", "no", " no")
              for i in tok.encode(w, add_special_tokens=False)[:1]}
    yes_ids, no_ids = sorted(yes_ids), sorted(no_ids)

    rows, since = [], 0
    for rel, path, node_id, cardinal, seq in tqdm(files, desc="panoramas",
                                                  mininterval=5.0):
        img = Image.open(path).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": QUESTION}]}]
        text = proc.apply_chat_template(msgs, tokenize=False,
                                        add_generation_prompt=True)
        inputs = proc(text=[text], images=[img], return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1].float()
        p = torch.softmax(
            torch.stack([logits[yes_ids].max(), logits[no_ids].max()]), 0)

        rows.append({"file": rel, "street": path.parent.parent.name,
                     "direction": path.parent.name, "seq": seq,
                     "node_id": node_id, "cardinal": cardinal,
                     "vlm_p_yes": round(float(p[0]), 5)})
        since += 1
        if since >= 50:
            pd.concat([done, pd.DataFrame(rows)], ignore_index=True).to_csv(
                args.table, index=False)
            since = 0

    out = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    out = out.sort_values(["street", "direction", "seq"])
    args.table.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.table, index=False)
    print(f"\n{len(out)} rows -> {args.table}")
    print(f"P(Yes): median {out.vlm_p_yes.median():.4f}  "
          f"p95 {out.vlm_p_yes.quantile(.95):.4f}  "
          f"share > 0.5 {(out.vlm_p_yes > .5).mean():.1%}")


if __name__ == "__main__":
    main()
