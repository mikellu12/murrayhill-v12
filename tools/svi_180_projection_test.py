"""Does the cylindrical reprojection cost the VLM anything?

The panoramas the study reads are cylindrical: azimuth is linear in x, so the
sides are not stretched, but straight world lines bow and a pixel at the
+/-45 degree edge subtends about a third of the solid angle of one at the
horizon. The raw frames those panoramas are built from are ordinary
perspective renders, 90 degrees and 640 px square, with no curvature at all.

This runs the SAME schema over the same nodes in both projections and scores
both against the same measured quantities. Identical prompt, identical model,
identical sample -- the projection is the only thing that differs, which is
what makes the comparison worth anything.

Two frames per walk: the headings nearest the walk bearing, which together
span 180 degrees around it. Each is rated on its own and the numeric fields
averaged; categorical fields take the first frame's answer. Averaging two
undistorted views is a real alternative pipeline, not a workaround -- it is
what you would build if you never reprojected in the first place.

SCHEMA, SYSTEM and parse() are imported rather than copied. A comparison
where the two arms could drift apart in wording would measure the wording.

    .venv-gpu/Scripts/python tools/svi_180_projection_test.py --sample 68
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import (AutoProcessor, BitsAndBytesConfig,
                          Qwen2VLForConditionalGeneration)

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
from common import PROC, RES, banner
from svi_180_sim_vlm import (MODEL, MAX_PIXELS, SYSTEM, SCHEMA, RATE, COUNT,
                             CAT, NAME_RE, parse)


def forward_frames(bearing, by_node, node_id, k=2):
    """The k headings whose centres sit closest to the walk bearing."""
    g = by_node.get(node_id)
    if g is None or g.empty:
        return []
    off = np.abs(((g.heading.to_numpy() - bearing + 180) % 360) - 180)
    return [g.path.to_numpy()[i] for i in np.argsort(off)[:k]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/raw/svi_180"))
    ap.add_argument("--table", type=Path,
                    default=RES / "tables" / "svi_180_projection_test.csv")
    ap.add_argument("--sample", type=int, default=68)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=260)
    args = ap.parse_args()
    banner("cylindrical panorama vs raw perspective frames")

    rows = []
    for jpg in sorted(args.src.rglob("*.jpg")):
        m = NAME_RE.search(jpg.name)
        if m:
            rows.append({"file": str(jpg.relative_to(args.src)).replace("\\", "/"),
                         "street": jpg.parent.parent.name,
                         "direction": jpg.parent.name, "node_id": m.group(2),
                         "seq": int(m.group(1))})
    fl = pd.DataFrame(rows)

    # Same stratified draw as the panorama trial, so the two arms cover the
    # same streets and the comparison is not confounded by the sample.
    per = max(1, args.sample // max(1, fl.street.nunique()))
    fl = pd.concat([g.sample(min(len(g), per), random_state=args.seed)
                    for _, g in fl.groupby("street")]).head(args.sample)
    print(f"{len(fl)} walks across {fl.street.nunique()} streets")

    # Bearings come from svi_180_scaffold.csv, which svi_scaffold_flag.py
    # already built with export_svi_180's own _street_axis/_walks. Reading
    # them avoids importing geopandas into the GPU env, which carries neither
    # it nor any other analysis dependency on purpose.
    sc = RES / "tables" / "svi_180_scaffold.csv"
    if not sc.exists():
        sys.exit(f"need {sc} for the walk bearings")
    b = pd.read_csv(sc)
    bearing = {(r.street, r.direction): r.bearing for r in b.itertuples()}

    man = pd.read_csv(PROC / "manifest.csv")
    by_node = {n: g for n, g in man.groupby("node_id")}

    print(f"loading {MODEL} in 4-bit NF4")
    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True)
    proc = AutoProcessor.from_pretrained(MODEL, max_pixels=MAX_PIXELS)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL, quantization_config=qcfg, device_map="cuda").eval()

    def judge(img):
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
        return parse(proc.tokenizer.decode(gen[0][inputs.input_ids.shape[1]:],
                                           skip_special_tokens=True))

    out, missing = [], 0
    for r in tqdm(list(fl.itertuples()), desc="walks", mininterval=10.0):
        b = bearing.get((r.street, r.direction))
        paths = forward_frames(b, by_node, r.node_id) if b is not None else []
        if len(paths) < 2:
            missing += 1
            continue
        recs = []
        for p in paths:
            try:
                recs.append(judge(Image.open(p).convert("RGB")))
            except Exception:
                pass
        recs = [x for x in recs if x]
        if not recs:
            missing += 1
            continue
        rec = {k: float(np.nanmean([x.get(k, np.nan) for x in recs]))
               for k in RATE + COUNT}
        for k in CAT:
            rec[k] = recs[0].get(k)
        rec.update({"file": r.file, "street": r.street, "direction": r.direction,
                    "node_id": r.node_id, "seq": r.seq, "n_frames": len(recs)})
        out.append(rec)

    d = pd.DataFrame(out)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(args.table, index=False)
    print(f"\n{len(d)} walks rated from raw frames -> {args.table}"
          + (f"   {missing} skipped" if missing else ""))

    pano = RES / "tables" / "svi_180_sim_vlm.csv"
    if not pano.exists():
        return
    p = pd.read_csv(pano)
    m = pd.read_csv(PROC / "metrics.csv")[["node_id", "GVI", "VEI"]]
    rng = np.random.default_rng(0)

    def ci(j, a, b):
        s = j[[a, b]].dropna()
        if len(s) < 8 or s[a].nunique() < 2:
            return None
        r = s[a].corr(s[b], method="spearman")
        bs = [s.iloc[rng.integers(0, len(s), len(s))].corr(
            method="spearman").iloc[0, 1] for _ in range(3000)]
        return r, np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5), len(s)

    print(f"\n{'field':<22}{'vs':<6}{'FRAMES (rectilinear)':<28}{'PANORAMA (cylindrical)'}")
    print("-" * 92)
    jf = d.merge(m, on="node_id", how="inner")
    jp = p[p.file.isin(set(d.file))].merge(m, on="node_id", how="inner")
    for a, b in (("green_eye_level", "GVI"), ("vertical_greenery", "GVI"),
                 ("green_softening", "GVI"), ("street_trees", "GVI"),
                 ("enclosure", "VEI"), ("vertical_hardscape", "VEI"),
                 ("walking_room", "VEI"), ("signage_detail", "GVI")):
        rf, rp = ci(jf, a, b), ci(jp, a, b)
        if not rf and not rp:
            continue
        f = f"{rf[0]:+.3f} [{rf[1]:+.3f},{rf[2]:+.3f}] n={rf[3]}" if rf else "-"
        q = f"{rp[0]:+.3f} [{rp[1]:+.3f},{rp[2]:+.3f}] n={rp[3]}" if rp else "-"
        print(f"{a:<22}{b:<6}{f:<28}{q}")
    print("\nSame prompt, same model, same nodes. Projection is the only difference.")


if __name__ == "__main__":
    main()
