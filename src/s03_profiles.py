"""Stage 3 -- azimuthal class profiles.

Instead of one pixel count per image, this builds a 360-bin profile of
absolute bearing per node: for each 1-degree slice of the horizon, the
solid-angle-weighted share of vegetation, sky and building.

Why: any view -- any centre bearing, any field of view -- is then a sum
over a contiguous range of bins. The 360 index, a 180-degree pedestrian
forward view heading uptown, a narrow 90-degree gaze: all fall out of the
same profile in microseconds, with no re-fetching and no re-segmenting.
Field of view becomes an analysis parameter rather than a decision baked
into data collection.

The mapping from degrees to pixel columns is gnomonic, not linear. A
fov-90 image is a perspective projection: the column at 29 degrees off
centre is 497 of 640, not the 526 a linear reading gives. Getting this
wrong displaces every direction boundary by ~2 degrees.

Vertical coverage is unchanged: pitch 0 at fov 90 sees +/-45 degrees of
elevation. Overhead canopy and most sky are not sampled, which biases GVI
low and VEI high by an unmeasured amount. That is a separate limitation
from the azimuthal one and is stated in the README.
"""
import sys
import numpy as np, pandas as pd, torch
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from common import (CFG, PROC, banner, device_and_batch, load_segmenter,
                    resolve_classes, column_bearings, column_weights,
                    image_path)

NB = CFG["directional"]["n_bins"]


_CW = {}


def profile_node(rows, proc, model, dev, classes):
    """Segment one node's images into a 4 x NB azimuthal accumulator.

    Row 3 accumulates total column weight per bin. Class shares are then
    acc[c]/acc[3], which self-normalises: it does not matter how many
    images cover a bin, nor that solid angle varies across a face. Summing
    raw class weight without this denominator scales GVI by the number of
    contributing images.

    Returns None if any image fails to open. A node built from three of its
    four headings would carry a quarter of the horizon at zero weight,
    which reads downstream as sampled-and-empty rather than as a gap.

    Factored out so tools/s03_subset.py profiles a handful of nodes through
    this exact code rather than a copy of it that can drift.
    """
    acc = np.zeros((4, NB))
    for _, r in rows.iterrows():
        try:
            img = Image.open(image_path(r.path)).convert("RGB")
        except Exception:
            return None
        inp = proc(images=[img], return_tensors="pt").to(dev)
        with torch.inference_mode():
            out = model(**inp)
        # MPS returns non-contiguous tensors where CUDA returns contiguous
        # ones, and post_process_semantic_segmentation calls .view()
        # internally. Moving to CPU first sidesteps it and costs nothing at
        # this size.
        out.class_queries_logits = out.class_queries_logits.cpu().contiguous()
        out.masks_queries_logits = out.masks_queries_logits.cpu().contiguous()
        a = proc.post_process_semantic_segmentation(
            out, target_sizes=[img.size[::-1]])[0].numpy()
        H, W = a.shape
        if W not in _CW:
            _CW[W] = column_weights(H, W)
        cw = _CW[W]
        idx = np.floor(column_bearings(float(r.heading), W)).astype(int) % NB
        for ci, ids in enumerate(classes):
            np.add.at(acc[ci], idx, np.isin(a, ids).mean(axis=0) * cw)
        np.add.at(acc[3], idx, cw)
    return acc


def main():
    banner("STAGE 3  azimuthal profiles")
    mf = pd.read_csv(PROC / "manifest.csv")
    dev, batch = device_and_batch()
    proc, model = load_segmenter(dev)
    VEG, SKY, BLD = resolve_classes(model.config.id2label)

    npz = PROC / "azimuth_profiles.npz"
    prof = {}
    if npz.exists():
        z = np.load(npz)
        prof = {k: z[k] for k in z.files}
        print(f"resuming: {len(prof)} nodes profiled")

    by_node = {n: g for n, g in mf.groupby("node_id")}
    todo = sorted(set(by_node) - set(prof))
    print(f"{len(todo)} nodes to profile (no network access)")

    for k, nid in enumerate(tqdm(todo, desc="profiling", mininterval=2.0)):
        acc = profile_node(by_node[nid], proc, model, dev, (VEG, SKY, BLD))
        if acc is not None:
            prof[nid] = acc
        if k % 50 == 0 and prof:
            np.savez_compressed(npz, **prof)

    # Only write when something was added. Re-running s03 for the
    # scaffolding pass alone used to rewrite this file for no reason, and
    # it is the one artefact in the repo that cannot be regenerated
    # cheaply -- a crash mid-write costs hours of segmentation.
    if todo:
        np.savez_compressed(npz, **prof)
        print(f"wrote {npz} ({len(prof)} nodes)")
    else:
        print(f"{npz} unchanged ({len(prof)} nodes already profiled)")

    # Sanity: the four images must tile 360 degrees with total weight 4.
    if prof:
        tot = np.stack(list(prof.values())).sum(axis=(0, 1))
        print(f"bins with data: {int((tot > 0).sum())}/{NB}")

    # ------------------------------------------------- scaffolding
    ov = CFG["open_vocabulary"]
    if not ov["enabled"]:
        return
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
    cp = CLIPSegProcessor.from_pretrained(ov["model"])
    cm = CLIPSegForImageSegmentation.from_pretrained(ov["model"]).to(dev).eval()

    ck = PROC / "scaffold.csv"
    done, recs = set(), []
    if ck.exists():
        prev = pd.read_csv(ck)
        recs, done = prev.to_dict("records"), set(prev.path)
    todo2 = [p for p in mf.path if p not in done]
    print(f"\nscaffolding: {len(todo2)} images to score")

    P = ov["scaffold_prompts"]
    for i, p in enumerate(tqdm(todo2, desc="scaffold", mininterval=2.0)):
        im = Image.open(image_path(p)).convert("RGB")
        inp = cp(text=P, images=[im] * len(P), padding=True,
                 return_tensors="pt").to(dev)
        with torch.inference_mode():
            pr = torch.sigmoid(cm(**inp).logits).max(dim=0).values
        recs.append({"path": p,
                     "f_scaffold": float((pr > ov["threshold"]).float().mean())})
        if i % 200 == 0:
            pd.DataFrame(recs).to_csv(ck, index=False)
    sc = pd.DataFrame(recs)
    sc.to_csv(ck, index=False)
    m = sc.merge(mf, on="path").groupby("node_id").f_scaffold.mean()
    m.rename("scaffold_frac").to_csv(PROC / "scaffold_by_node.csv")
    print(f"scaffolding: {m.mean():.2%} of pixels, "
          f"{(m > 0.05).mean():.1%} of nodes above 5%")


if __name__ == "__main__":
    main()
