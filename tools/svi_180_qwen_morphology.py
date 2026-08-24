"""Qwen2-VL-7B rating micro-scale morphology and sense of place, per panorama.

The model is Qwen2-VL-7B-Instruct in 4-bit NF4, which is the same model MINGLE
(AAAI 2026) selected after benchmarking eleven VLMs including GPT-4o and Claude
Opus, for the reason that applies here too: the small variant loses little and
runs fast enough to cross a whole city. 16.6 GB of bf16 weights do not fit a
12 GB card; NF4 brings them to about 5 GB and leaves room for the vision
tokens a 1440-px panorama produces.

Output is a strict JSON object per image, parsed into flat columns. Generation
is greedy: these ratings are compared across 1,254 images, and sampling would
put noise on that axis for no benefit.

READ BEFORE USING THE sense_of_place COLUMNS. CLAUDE.md states that no outcome
variable exists in this pipeline -- the framework's dependent variable is sense
of place, and it "needs survey or dwell-time data and cannot be fitted here."
A VLM's 1-7 rating is not that data. It is a model's guess at what a person
would feel, unvalidated against any person. On the far easier binary question
of whether a structure is visible, VLM labels scored kappa 0.667 against one
human rater; three subjective 7-point scales will be looser, and nothing here
measures how much. These columns are exploratory until human ratings exist to
score them against, and must not be presented as the framework's outcome.

Written beside the photo and mask so a rating can be read against the scene it
describes, and to a CSV so it can be joined to metrics.csv on node_id.

    .venv-gpu/Scripts/python tools/svi_180_qwen_morphology.py --limit 4
    .venv-gpu/Scripts/python tools/svi_180_qwen_morphology.py --street 1st_avenue
"""
import argparse
import json
import re
import sys
import textwrap
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm
from transformers import (AutoProcessor, BitsAndBytesConfig,
                          Qwen2VLForConditionalGeneration)

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import RES, banner

MODEL = "Qwen/Qwen2-VL-7B-Instruct"
MAX_PIXELS = 1024 * 28 * 28

SYSTEM = ("You are an expert urban morphologist evaluating streetscape quality "
          "at eye-level (1.5m).")

SCHEMA = """Analyze the image and evaluate micro-scale spatial attributes and Sense of Place.

Reply with ONE JSON object and nothing else, exactly this shape:

{
  "eye_level_greenery": {
    "layer_type": "low_planter|hedge|tree_canopy|vertical_wall|none",
    "visual_dominance": 1-7
  },
  "edge_effect_interface": {
    "barrier_present": true,
    "type": "stoop|yard|planter_wall|none",
    "buffering_quality": "low|medium|high"
  },
  "street_canyon_enclosure": {
    "perceived_hw_ratio": "under_enclosed|human_scale|deep_canyon",
    "framing_score": 1-7
  },
  "sense_of_place": {
    "place_identity": {"score": 1-7, "rationale": "one short morphological clause"},
    "place_attachment": {"score": 1-7, "rationale": "one short microclimate or restorative clause"},
    "place_dependence": {"score": 1-7, "rationale": "one short lingering or seating affordance clause"}
  }
}

human_scale means a height-to-width ratio of roughly 0.8-1.5. barrier_present
is true or false. Keep every rationale under 15 words.

The three sense_of_place scores measure DIFFERENT things and will usually
DIFFER from each other. Score each independently:
- place_identity: how distinctive and legible is this place? Could a person
  tell it apart from any other street? Generic frontage scores low even when
  the street is pleasant.
- place_attachment: would a person feel comfortable and want to be here?
  Shade, shelter, enclosure, quiet, greenery. A distinctive but hostile
  street scores low here while scoring high on identity.
- place_dependence: could a person actually stop and stay? Seating, stoops,
  width, forecourts. A street with nowhere to stand scores low here even if
  it is beautiful.

Do not give all three the same score unless the evidence genuinely warrants
it. Output JSON only."""

NAME_RE = re.compile(r"(\d+)_(n\d+)_([NESW])\.jpg$")


def _font(sz):
    for n in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(n, sz)
        except OSError:
            continue
    return ImageFont.load_default()


def _num(v):
    """A rating as a number, whatever shape the model wrapped it in."""
    if isinstance(v, dict):
        v = v.get("score")
    try:
        return float(v)
    except (TypeError, ValueError):
        m = re.search(r"\d+(\.\d+)?", str(v))
        return float(m.group()) if m else None


def _why(v):
    if isinstance(v, dict):
        return str(v.get("rationale", ""))[:200]
    return ""


def parse(txt):
    """Flatten the model's JSON. Missing keys stay None rather than guessed."""
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return {}
    try:
        j = json.loads(m.group())
    except json.JSONDecodeError:
        return {}
    g = j.get("eye_level_greenery") or {}
    e = j.get("edge_effect_interface") or {}
    c = j.get("street_canyon_enclosure") or {}
    s = j.get("sense_of_place") or {}
    return {
        "greenery_layer": g.get("layer_type"),
        "greenery_dominance": _num(g.get("visual_dominance")),
        "barrier_present": e.get("barrier_present"),
        "edge_type": e.get("type"),
        "buffering_quality": e.get("buffering_quality"),
        "perceived_hw_ratio": c.get("perceived_hw_ratio"),
        "framing_score": _num(c.get("framing_score")),
        "place_identity": _num(s.get("place_identity")),
        "place_identity_why": _why(s.get("place_identity")),
        "place_attachment": _num(s.get("place_attachment")),
        "place_attachment_why": _why(s.get("place_attachment")),
        "place_dependence": _num(s.get("place_dependence")),
        "place_dependence_why": _why(s.get("place_dependence")),
    }


def dash(v):
    return "-" if v is None or (isinstance(v, float) and v != v) else v


def panel(photo, mask, rec, width, font, small):
    """Photo, mask, then the ratings as text under the scene they describe."""
    w = width
    h = round(w * photo.height / photo.width)
    photo = photo.resize((w, h), Image.LANCZOS)
    mask = mask.resize((w, h), Image.NEAREST)

    lines = [
        "greenery   {:<14} dominance {}/7".format(
            dash(rec.get("greenery_layer")), dash(rec.get("greenery_dominance"))),
        "edge       barrier={}  {}  buffering={}".format(
            dash(rec.get("barrier_present")), dash(rec.get("edge_type")),
            dash(rec.get("buffering_quality"))),
        "canyon     {:<14} framing {}/7".format(
            dash(rec.get("perceived_hw_ratio")), dash(rec.get("framing_score"))),
        "place      identity {}/7   attachment {}/7   dependence {}/7".format(
            dash(rec.get("place_identity")), dash(rec.get("place_attachment")),
            dash(rec.get("place_dependence"))),
    ]
    for k, lab in (("place_identity_why", "identity"),
                   ("place_attachment_why", "attachment"),
                   ("place_dependence_why", "dependence")):
        t = (rec.get(k) or "").strip()
        if t:
            lines += textwrap.wrap("  {}: {}".format(lab, t),
                                   width=max(40, w // 8))[:2]

    cap, lh = 20, 19
    foot = 10 + lh * len(lines) + 8
    out = Image.new("RGB", (w, cap + h * 2 + 1 + foot), "white")
    d = ImageDraw.Draw(out)
    d.text((4, 3), rec["file"], fill="black", font=font)
    out.paste(photo, (0, cap))
    out.paste(mask, (0, cap + h + 1))
    y = cap + h * 2 + 8
    for ln in lines:
        d.text((6, y), ln, fill="black", font=small)
        y += lh
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/raw/svi_180"))
    ap.add_argument("--masks", type=Path, default=RES / "svi_180_seg")
    ap.add_argument("--out", type=Path,
                    default=RES / "svi_180_seg_comparison_QWEN2-VL_7B")
    ap.add_argument("--table", type=Path,
                    default=RES / "tables" / "svi_180_qwen_morphology.csv")
    ap.add_argument("--width", type=int, default=1100)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--street", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-images", action="store_true",
                    help="ratings only, skip the composited panels")
    ap.add_argument("--restart", action="store_true")
    args = ap.parse_args()
    banner("qwen2-vl-7b micro-scale morphology and sense of place")

    files = []
    for jpg in sorted(args.src.rglob("*.jpg")):
        m = NAME_RE.search(jpg.name)
        if not m:
            continue
        if args.street and jpg.parent.parent.name != args.street:
            continue
        files.append((str(jpg.relative_to(args.src)).replace("\\", "/"), jpg,
                      m.group(2), m.group(3), int(m.group(1))))
    if not files:
        sys.exit("no panoramas under {}".format(args.src))
    print("{} panoramas".format(len(files)))

    done = pd.DataFrame()
    if args.table.exists() and not args.restart:
        done = pd.read_csv(args.table)
        seen = set(done.file)
        files = [f for f in files if f[0] not in seen]
        print("{} already rated, {} to do".format(len(done), len(files)))
    if args.limit:
        files = files[:args.limit]
    if not files:
        print("nothing to do")
        return

    print("loading {} in 4-bit NF4".format(MODEL))
    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True)
    proc = AutoProcessor.from_pretrained(MODEL, max_pixels=MAX_PIXELS)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL, quantization_config=qcfg, device_map="cuda").eval()
    print("VRAM in use {:.1f} GB".format(torch.cuda.memory_allocated() / 1e9))

    font, small = _font(15), _font(14)
    rows, bad, since = [], 0, 0
    for rel, path, node_id, cardinal, seq in tqdm(files, desc="panoramas",
                                                  mininterval=10.0):
        img = Image.open(path).convert("RGB")
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": SCHEMA}]}]
        text = proc.apply_chat_template(msgs, tokenize=False,
                                        add_generation_prompt=True)
        inputs = proc(text=[text], images=[img], return_tensors="pt").to("cuda")
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=False,
                                 pad_token_id=proc.tokenizer.eos_token_id)
        txt = proc.tokenizer.decode(gen[0][inputs.input_ids.shape[1]:],
                                    skip_special_tokens=True)
        rec = parse(txt)
        if not rec:
            bad += 1
        rec.update({"file": rel, "street": path.parent.parent.name,
                    "direction": path.parent.name, "seq": seq,
                    "node_id": node_id, "cardinal": cardinal,
                    "parsed": rec.get("place_identity") is not None})
        rows.append(rec)

        if not args.no_images:
            mp = args.masks / (rel.replace("/", "__")[:-4] + "_mask.png")
            if mp.exists():
                dest = args.out / Path(rel)
                dest.parent.mkdir(parents=True, exist_ok=True)
                panel(img, Image.open(mp).convert("RGB"), rec, args.width,
                      font, small).save(dest, quality=85, optimize=True)

        since += 1
        if since >= 25:
            pd.concat([done, pd.DataFrame(rows)], ignore_index=True).to_csv(
                args.table, index=False)
            since = 0

    out = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    out = out.sort_values(["street", "direction", "seq"])
    args.table.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.table, index=False)

    print("\n{} rows -> {}".format(len(out), args.table))
    if bad:
        print("{} replies did not parse as JSON (columns left empty)".format(bad))
    if not args.no_images:
        print("panels -> {}".format(args.out))
    for c in ("greenery_dominance", "framing_score", "place_identity",
              "place_attachment", "place_dependence"):
        if c in out and out[c].notna().any():
            print("  {:<22} mean {:.2f}  sd {:.2f}".format(
                c, out[c].mean(), out[c].std()))
    for c in ("greenery_layer", "perceived_hw_ratio", "buffering_quality"):
        if c in out:
            print("  {}: {}".format(c, out[c].value_counts().head(5).to_dict()))


if __name__ == "__main__":
    main()
