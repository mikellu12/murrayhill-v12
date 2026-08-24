"""SIM-class azimuthal profiles: the 11 Street Interface Matrix terms.

s03 stores three classes -- vegetation, sky, building -- because that is all
GVI and VEI need. Every SIM term beyond greenery and enclosure (benches,
awnings, sidewalk, signboards, railings) was never written, so the saved
profiles cannot answer them no matter how they are sliced.

This writes a SECOND array rather than widening the first. azimuth_profiles.npz
is 35 minutes to 6 hours of segmentation and every existing GVI/VEI result is
reproducible from it; a wider array would change its shape and silently break
common.slice_metrics, which indexes rows 0-2 by position. Two files, one
shape each, no ambiguity about which produced a number.

Vegetation is split at the horizon here rather than at draw time. The
azimuthal profile collapses the vertical axis, so once a column is averaged
the eye-level and canopy contributions are indistinguishable -- the split has
to happen before the collapse or not at all. At pitch=0 the horizon is the
centre row, so rows [0, H/2) are above it and [H/2, H) below. Both halves are
divided by the FULL column height, so eye_green + canopy_green equals the
vegetation share s03 would have recorded.

    .venv-gpu/Scripts/python tools/s03_sim_profiles.py --limit 12   # smoke test
    .venv-gpu/Scripts/python tools/s03_sim_profiles.py              # all nodes
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import (CFG, PROC, banner, column_bearings, column_weights,
                    device_and_batch, image_path, load_segmenter)

NB = CFG["directional"]["n_bins"]
OUT = PROC / "sim_profiles.npz"

# Row order is written to the npz alongside the data. Nothing downstream
# should index these by position -- s03's three-row layout is exactly the
# assumption that makes widening an array dangerous.
SPLIT_GREEN = "eye_green"          # the one group split at the horizon
_CW = {}


def resolve_groups(id2label):
    """Map config's sim.groups to class ids by LABEL NAME, never index.

    ADE20K ordering varies between checkpoints, and this checkpoint returns
    bare names ("pot", not "pot, flowerpot"). Matching a synonym string
    against the wrong checkpoint would put "pot" and "stool" on the toilet
    class, so an unresolved name is reported rather than silently dropped.
    """
    lut = {i: [p.strip().lower() for p in lab.split(",")]
           for i, lab in id2label.items()}
    out, report = {}, []
    for group, names in CFG["sim"]["groups"].items():
        ids = [i for i, labs in lut.items() if any(n in labs for n in names)]
        missing = [n for n in names
                   if not any(n in labs for labs in lut.values())]
        out[group] = ids
        report.append((group, [id2label[i] for i in ids], missing))
    return out, report


def profile_node(rows, proc, model, dev, groups):
    """Segment one node's images into a (len(rows)+1) x NB accumulator.

    Mirrors s03.profile_node, including the weight row: class shares are
    acc[c]/acc[-1], which normalises away both the number of images covering
    a bin and the varying solid angle across an image face.

    Returns None if any image fails to open, for the same reason s03 does --
    a node built from three of its four headings carries a quarter of the
    horizon at zero weight, which reads downstream as sampled-and-empty
    rather than as a gap.
    """
    names = list(groups) + [f"canopy_{SPLIT_GREEN.split('_')[1]}"]
    acc = np.zeros((len(names) + 1, NB))
    gi = {g: k for k, g in enumerate(names)}
    for _, r in rows.iterrows():
        try:
            img = Image.open(image_path(r.path)).convert("RGB")
        except Exception:
            return None, names
        inp = proc(images=[img], return_tensors="pt").to(dev)
        with torch.inference_mode():
            out = model(**inp)
        # MPS returns non-contiguous tensors where CUDA returns contiguous
        # ones and post_process calls .view() internally; moving to CPU
        # first is free at this size and works on both.
        out.class_queries_logits = out.class_queries_logits.cpu().contiguous()
        out.masks_queries_logits = out.masks_queries_logits.cpu().contiguous()
        a = proc.post_process_semantic_segmentation(
            out, target_sizes=[img.size[::-1]])[0].numpy()
        H, W = a.shape
        if W not in _CW:
            _CW[W] = column_weights(H, W)
        cw = _CW[W]
        idx = np.floor(column_bearings(float(r.heading), W)).astype(int) % NB
        mid = H // 2
        for g, ids in groups.items():
            hit = np.isin(a, ids)
            if g == SPLIT_GREEN:
                # Divide both halves by the full height so the two rows sum
                # to the share s03 records for vegetation.
                np.add.at(acc[gi[g]], idx, hit[mid:].sum(axis=0) / H * cw)
                np.add.at(acc[gi[names[-1]]], idx,
                          hit[:mid].sum(axis=0) / H * cw)
            else:
                np.add.at(acc[gi[g]], idx, hit.mean(axis=0) * cw)
        np.add.at(acc[-1], idx, cw)
    return acc, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="profile only the first N nodes (smoke test)")
    args = ap.parse_args()

    banner("SIM-class azimuthal profiles")
    mf = pd.read_csv(PROC / "manifest.csv")
    dev, _ = device_and_batch()
    proc, model = load_segmenter(dev)
    groups, report = resolve_groups(model.config.id2label)

    print(f"device: {dev}\n")
    for group, got, missing in report:
        print(f"  {group:<14} {len(got):>2} classes  {sorted(got)}")
        if missing:
            print(f"  {'':<14}    unresolved: {missing}")
    empty = [g for g, ids in groups.items() if not ids]
    if empty:
        raise SystemExit(f"\nno class resolved for {empty} -- refusing to "
                         "write a profile with a term that is always zero")

    ids = sorted(mf.node_id.unique())
    if args.limit:
        ids = ids[:args.limit]
    print(f"\nprofiling {len(ids)} nodes")

    prof, skipped, names = {}, [], None
    for nid in tqdm(ids, desc="nodes"):
        acc, names = profile_node(mf[mf.node_id == nid], proc, model, dev,
                                  groups)
        if acc is None:
            skipped.append(nid)
            continue
        prof[nid] = acc

    if skipped:
        print(f"\nskipped {len(skipped)} node(s) whose imagery would not "
              f"open: {', '.join(skipped[:12])}"
              f"{' ...' if len(skipped) > 12 else ''}")

    # Row names travel with the data. Positional row indexing is what makes
    # a widened array dangerous, so nothing downstream has to guess.
    np.savez_compressed(OUT, __rows__=np.array(names + ["weight"]), **prof)
    print(f"\nwrote {OUT}  ({len(prof)} nodes, "
          f"{len(names) + 1} x {NB} each)")
    print("rows:", ", ".join(names + ["weight"]))


if __name__ == "__main__":
    main()
