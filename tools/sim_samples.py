"""Segmentation sample panels over the 180-degree along-street view.

Shows what the SIM class grouping picks out of the view the study actually
analyses. The class shares alone cannot tell you whether "rest 0.07%" is a
real scarcity of benches or a detector that never fires; the picture can.

The source frames are fov=90 rectilinear, so a single frame is a quarter of
the horizon and not the window any metric is computed over. This reprojects
four frames into one 180-degree cylindrical panorama centred on the street
axis -- the same window slice_metrics integrates.

Reprojection, not stitching. A rectilinear frame has focal length
f = (W/2)/tan(45 deg) = W/2, so for a bearing alpha from the frame centre and
an elevation phi:

    x = f*tan(alpha) + W/2
    y = H/2 - f*tan(phi)/cos(alpha)

Pasting frames side by side instead would leave the horizon bent at every
seam, because tan(alpha) is not linear in alpha. The 1/cos(alpha) term is
what keeps a building edge straight across a seam.

The class map is reprojected through the identical mapping with nearest
neighbour, so the mask corresponds pixel for pixel to the photograph rather
than being segmented from a resampled image.

    .venv-gpu/Scripts/python tools/sim_samples.py --n 5
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import CFG, PROC, RES, banner, image_path, load_segmenter

OUT_W = 1440                    # 180 deg across
FOV = CFG["directional"]["fov"]


def hex2rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.uint8)


def panorama(frames, axis):
    """Reproject fov=90 frames into one cylindrical strip centred on `axis`.

    frames maps heading -> (rgb HxWx3, class map HxW). Returns the same pair
    resampled onto the 180-degree window, plus the elevation of every output
    row so the horizon split can be applied to the panorama.
    """
    H, W = next(iter(frames.values()))[1].shape
    f = W / 2.0                                   # tan(45 deg) = 1
    fc = OUT_W / np.radians(FOV)                  # cylindrical focal length
    out_h = int(2 * fc * np.tan(np.radians(45.0)))

    theta = np.radians(axis) + (np.arange(OUT_W) - OUT_W / 2) / fc
    # Row 0 is the top of the output, which is positive elevation. Getting
    # this sign wrong silently returns the panorama flipped: sky underfoot.
    phi = np.arctan((out_h / 2 - np.arange(out_h)) / fc)

    heads = np.array(sorted(frames))
    # Each output column is taken from the frame whose centre it is nearest,
    # which keeps |alpha| <= 45 deg and the resampling well conditioned.
    dh = (np.degrees(theta)[:, None] - heads[None, :] + 180) % 360 - 180
    pick = np.abs(dh).argmin(axis=1)
    alpha = np.radians(dh[np.arange(OUT_W), pick])

    rgb = np.zeros((out_h, OUT_W, 3), dtype=np.uint8)
    cls = np.full((out_h, OUT_W), -1, dtype=np.int32)
    xs = f * np.tan(alpha) + W / 2.0
    for k, h in enumerate(heads):
        col = np.where(pick == k)[0]
        if not len(col):
            continue
        src_x = np.clip(np.round(xs[col]).astype(int), 0, W - 1)
        # 1/cos(alpha) is the rectilinear stretch away from frame centre.
        ys = H / 2.0 - f * np.tan(phi)[:, None] / np.cos(alpha[col])[None, :]
        src_y = np.clip(np.round(ys).astype(int), 0, H - 1)
        r, c = frames[h]
        rgb[:, col] = r[src_y, src_x[None, :]]
        cls[:, col] = c[src_y, src_x[None, :]]
    return rgb, cls, phi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()
    banner("SIM segmentation samples -- 180 deg along-street")

    m = pd.read_csv(PROC / "metrics.csv").dropna(subset=["GVI", "street_axis_deg"])
    # sim_index.csv holds exactly the nodes inside the study area with a
    # usable along-street window, so it is the right sampling pool and does
    # not drag geopandas into the GPU environment.
    _ix = PROC / "sim_index.csv"
    if _ix.exists():
        keep = set(pd.read_csv(_ix).node_id)
        m = m[m.node_id.isin(keep)]
        print(f"study-area filter: sampling from {len(m)} nodes")
    mf = pd.read_csv(PROC / "manifest.csv")
    picks = [m.iloc[(m.GVI - m.GVI.quantile(t)).abs().argmin()]
             for t in np.linspace(0.02, 0.98, args.n)]

    proc, model = load_segmenter("cuda")
    lut = {i: [p.strip().lower() for p in l.split(",")]
           for i, l in model.config.id2label.items()}
    groups = {g: [i for i, labs in lut.items() if any(n in labs for n in names)]
              for g, names in CFG["sim"]["groups"].items()}
    pal = CFG["sim"]["palette"]

    out = RES / "figures" / "sim_samples"
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    for k, r in enumerate(picks, 1):
        frames = {}
        for _, h in mf[mf.node_id == r.node_id].iterrows():
            img = Image.open(image_path(h.path)).convert("RGB")
            inp = proc(images=[img], return_tensors="pt").to("cuda")
            with torch.inference_mode():
                o = model(**inp)
            o.class_queries_logits = o.class_queries_logits.cpu().contiguous()
            o.masks_queries_logits = o.masks_queries_logits.cpu().contiguous()
            a = proc.post_process_semantic_segmentation(
                o, target_sizes=[img.size[::-1]])[0].numpy()
            frames[float(h.heading)] = (np.asarray(img), a)

        # Face the street: the axis is mod 180, so pick the direction whose
        # frames are actually present rather than assuming one of them.
        rgb, cls, phi = panorama(frames, float(r.street_axis_deg))

        mask = np.full(rgb.shape, hex2rgb(pal["other"]), dtype=np.uint8)
        below = (phi < 0)[:, None]                # horizon at phi = 0
        for g, ids in groups.items():
            hit = np.isin(cls, ids)
            if g == "eye_green":
                mask[hit & below] = hex2rgb(pal["eye_green"])
                mask[hit & ~below] = hex2rgb(pal["canopy_green"])
            else:
                mask[hit] = hex2rgb(pal[g])
        mask[cls < 0] = 255                       # unsampled: leave white

        over = (0.55 * rgb.astype(np.float32)
                + 0.45 * mask.astype(np.float32)).astype(np.uint8)

        h_out = rgb.shape[0]
        strip = Image.new("RGB", (OUT_W, h_out * 3 + 16), (255, 255, 255))
        for j, im in enumerate((rgb, over, mask)):
            strip.paste(Image.fromarray(im), (0, j * (h_out + 8)))
        name = f"sample_{k:02d}_{r.node_id}.png"
        strip.save(out / name)

        shares = {g: float(np.isin(cls, ids).mean()) for g, ids in groups.items()}
        rows.append({"file": name, "node_id": r.node_id, "street": r.osm_name,
                     "typology": r.typology, "axis_deg": round(float(r.street_axis_deg), 1),
                     "GVI": round(float(r.GVI), 2), "VEI": round(float(r.VEI), 3),
                     **{f"share_{g}": round(v, 5) for g, v in shares.items()}})
        print(f"  {name}  {r.osm_name:<18} axis={r.street_axis_deg:5.1f}deg  "
              f"GVI={r.GVI:5.2f} VEI={r.VEI:.3f}  panorama {OUT_W}x{h_out}")

    pd.DataFrame(rows).to_csv(RES / "tables" / "sim_samples.csv", index=False)
    print(f"\nwrote {len(rows)} panels to {out}")


if __name__ == "__main__":
    main()
