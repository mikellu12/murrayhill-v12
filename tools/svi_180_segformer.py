"""SegFormer-B5 Cityscapes over the 180-degree along-street panoramas.

A second opinion, not a replacement. The pipeline's own segmentation is
Mask2Former on ADE20K and lives in `azimuth_profiles.npz`; this is SegFormer
on Cityscapes, a different model and a different taxonomy. The two disagree
by construction -- Cityscapes folds all greenery into `vegetation` plus
`terrain`, where ADE20K separates tree, grass, plant and palm -- so the
`vegetation` share written here is NOT GVI and must not be substituted into
it. Wire nothing from this table into an index before it has a row in the
per-class AUC table (`tools/openvocab_eval.py`); see CLAUDE.md.

Cityscapes has no scaffolding class. The nineteen classes include `fence`,
`wall` and `building`, and a sidewalk shed reads as some mixture of those,
so this does not answer the shed question that `svi_scaffold_flag.py` and
the review sheets exist to answer. `fence` looks like the one class with a
DOB counterpart -- Construction Fence -- but it is confounded: on the first
eight panoramas the pixels it claims are a ballfield's chain-link and a
roadway barrier, both permanent. Score it against that column before
believing it; the output carries the join keys so that is one merge.

One row per image, keyed by street/direction/seq/node_id, so this table
joins to `svi_180_scaffold.csv` and `svi_180_visual_labels.csv` on `file`.

Overlays draw Street View pixels under the mask, so they go to a directory
git ignores, for the same reason `results/figures/face_samples/` is ignored.
Only the CSV is derived metric and safe to commit.

Preprocessing is left exactly as the source script had it: the checkpoint's
own processor squashes to 1024x1024 and the logits are interpolated back, so
a 1440x916 panorama is stretched vertically on the way in. Faithful beats
clever here -- change it and the numbers stop matching the run this came
from. `--keep-aspect` letterboxes instead, for comparison.

    .venv-gpu/Scripts/python tools/svi_180_segformer.py --limit 8 --overlays
    .venv-gpu/Scripts/python tools/svi_180_segformer.py
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F
from tqdm.auto import tqdm
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import RES, banner

MODEL_NAME = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"

# NVIDIA published this checkpoint as pytorch_model.bin only, and transformers
# refuses torch.load under torch < 2.6 (CVE-2025-32434). Upgrading torch in
# .venv-gpu to satisfy that would put the Mask2Former stage the whole pipeline
# depends on at risk for the sake of one tool, so the weights are converted to
# safetensors once, outside the repo, and loaded from there afterwards. Same
# tensors, same results; only the container changes.
LOCAL_CKPT = Path.home() / ".cache" / "murrayhill" / "segformer-b5-cityscapes"

CLASS_NAMES = {
    0: "road", 1: "sidewalk", 2: "building", 3: "wall", 4: "fence",
    5: "pole", 6: "traffic_light", 7: "traffic_sign", 8: "vegetation",
    9: "terrain", 10: "sky", 11: "person", 12: "rider", 13: "car",
    14: "truck", 15: "bus", 16: "train", 17: "motorcycle", 18: "bicycle",
}

PALETTE = {
    0: (128, 64, 128), 1: (238, 0, 220), 2: (70, 70, 70), 3: (102, 102, 156),
    4: (190, 153, 153), 5: (153, 153, 153), 6: (250, 170, 30), 7: (220, 220, 0),
    8: (107, 142, 35), 9: (152, 251, 152), 10: (70, 130, 180), 11: (220, 20, 60),
    12: (255, 0, 0), 13: (0, 0, 142), 14: (0, 0, 70), 15: (0, 60, 100),
    16: (0, 80, 100), 17: (0, 0, 230), 18: (119, 11, 32),
}

# Same pattern the export writes and svi_scaffold_flag.py reads. Anything
# that does not match is not a panorama and is skipped rather than guessed at.
NAME_RE = re.compile(r"(\d+)_(n\d+)_([NESW])\.jpg$")


def _lut() -> np.ndarray:
    """Class id -> RGB as a lookup table, so colouring is one indexing op."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    for k, v in PALETTE.items():
        lut[k] = v
    return lut


LUT = _lut()


def _inventory(src: Path) -> pd.DataFrame:
    rows = []
    for jpg in sorted(src.rglob("*.jpg")):
        m = NAME_RE.search(jpg.name)
        if not m:
            continue
        rows.append({
            "street": jpg.parent.parent.name,
            "direction": jpg.parent.name,
            "seq": int(m.group(1)),
            "node_id": m.group(2),
            "cardinal": m.group(3),
            "file": str(jpg.relative_to(src)).replace("\\", "/"),
            "src_path": jpg,
        })
    return pd.DataFrame(rows)


def _segment(model, processor, img: Image.Image, device, keep_aspect: bool):
    """Per-pixel class ids at the image's own resolution."""
    W, H = img.size
    src = img
    if keep_aspect:
        # Letterbox to square on grey, so the model sees undistorted geometry.
        side = max(W, H)
        src = Image.new("RGB", (side, side), (128, 128, 128))
        src.paste(img, (0, (side - H) // 2))

    inputs = processor(images=src, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits

    if keep_aspect:
        side = max(W, H)
        logits = F.interpolate(logits, size=(side, side), mode="bilinear",
                               align_corners=False)
        top = (side - H) // 2
        logits = logits[:, :, top:top + H, :W]
    else:
        logits = F.interpolate(logits, size=(H, W), mode="bilinear",
                               align_corners=False)
    return logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)


def _panel(photo: Image.Image, mask_rgb: np.ndarray, caption: str) -> Image.Image:
    """Photo beside mask, captioned. PIL rather than matplotlib on purpose:
    the GPU env carries neither matplotlib nor cv2, and adding either to run
    a two-panel figure is a worse trade than compositing it here."""
    w, h = photo.size
    cap = 16
    out = Image.new("RGB", (w * 2 + 3, h + cap), "white")
    ImageDraw.Draw(out).text((2, 3), caption, fill="black")
    out.paste(photo, (0, cap))
    out.paste(Image.fromarray(mask_rgb), (w + 3, cap))
    return out


def _checkpoint() -> Path:
    """Local safetensors copy of the checkpoint, converted on first use."""
    if (LOCAL_CKPT / "model.safetensors").exists():
        return LOCAL_CKPT
    import shutil
    from huggingface_hub import hf_hub_download
    from safetensors.torch import save_file

    print(f"converting {MODEL_NAME} to safetensors (once)")
    LOCAL_CKPT.mkdir(parents=True, exist_ok=True)
    for f in ("config.json", "preprocessor_config.json"):
        shutil.copy(hf_hub_download(MODEL_NAME, f), LOCAL_CKPT / f)
    sd = torch.load(hf_hub_download(MODEL_NAME, "pytorch_model.bin"),
                    map_location="cpu", weights_only=True)
    save_file({k: v.contiguous().clone() for k, v in sd.items()},
              str(LOCAL_CKPT / "model.safetensors"), metadata={"format": "pt"})
    return LOCAL_CKPT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/raw/svi_180"))
    ap.add_argument("--out", type=Path, default=RES / "svi_180_seg")
    ap.add_argument("--table", type=Path,
                    default=RES / "tables" / "svi_180_segformer.csv")
    ap.add_argument("--overlays", action="store_true",
                    help="also write mask PNG + photo/mask panel per image")
    ap.add_argument("--keep-aspect", action="store_true",
                    help="letterbox instead of squashing to 1024x1024")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N images, for a smoke test")
    ap.add_argument("--restart", action="store_true",
                    help="ignore an existing table instead of resuming")
    args = ap.parse_args()
    banner("segformer b5 cityscapes over 180-degree panoramas")

    if not args.src.exists():
        sys.exit(f"no such directory: {args.src}")
    df = _inventory(args.src)
    if df.empty:
        sys.exit(f"no panoramas matching {NAME_RE.pattern} under {args.src}")
    print(f"{len(df)} panoramas in {args.src}")

    # Resume: a full pass is thousands of forward passes, and losing it to a
    # closed laptop or a dropped SSH session should cost only what is unwritten.
    done = pd.DataFrame()
    if args.table.exists() and not args.restart:
        done = pd.read_csv(args.table)
        df = df[~df.file.isin(set(done.file))]
        print(f"{len(done)} already in {args.table}, {len(df)} to do")
    if args.limit:
        df = df.head(args.limit)
    if df.empty:
        print("nothing to do")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}"
          + (f"  {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""))
    ckpt = _checkpoint()
    print(f"loading {MODEL_NAME}")
    processor = SegformerImageProcessor.from_pretrained(ckpt)
    model = SegformerForSemanticSegmentation.from_pretrained(ckpt)
    model.to(device).eval()

    if args.overlays:
        args.out.mkdir(parents=True, exist_ok=True)
    args.table.parent.mkdir(parents=True, exist_ok=True)

    rows, since_flush = [], 0
    for r in tqdm(list(df.itertuples()), desc="panoramas", mininterval=1.0):
        photo = Image.open(r.src_path).convert("RGB")
        pred = _segment(model, processor, photo, device, args.keep_aspect)

        rec = {"street": r.street, "direction": r.direction, "seq": r.seq,
               "node_id": r.node_id, "cardinal": r.cardinal, "file": r.file}
        # Shares, not counts: panoramas are a fixed size now but the column
        # should survive a re-export at another width.
        share = np.bincount(pred.ravel(), minlength=19) / pred.size * 100
        for cid, name in CLASS_NAMES.items():
            rec[name] = round(float(share[cid]), 4)
        rows.append(rec)

        if args.overlays:
            mask_rgb = LUT[pred]
            stem = r.file.replace("/", "__")[:-4]
            Image.fromarray(mask_rgb).save(args.out / f"{stem}_mask.png",
                                           optimize=True)
            top = ", ".join(f"{CLASS_NAMES[c]} {share[c]:.1f}%"
                            for c in np.argsort(share)[::-1][:5])
            _panel(photo, mask_rgb, f"{r.file}   {top}").save(
                args.out / f"{stem}_panel.jpg", quality=82, optimize=True)

        since_flush += 1
        if since_flush >= 50:
            out = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
            out.to_csv(args.table, index=False)
            since_flush = 0

    out = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    out = out.sort_values(["street", "direction", "seq"])
    out.to_csv(args.table, index=False)

    print(f"\n{len(out)} rows -> {args.table}")
    if args.overlays:
        mb = sum(p.stat().st_size for p in args.out.glob("*")) / 1024 / 1024
        print(f"overlays -> {args.out}  ({mb:.0f} MB)")
    print("\nmean pixel share by class:")
    for name in sorted(CLASS_NAMES.values(), key=lambda n: -out[n].mean()):
        m = out[name].mean()
        if m >= 0.05:
            print(f"  {name:<15}{m:>7.2f}%")


if __name__ == "__main__":
    main()
