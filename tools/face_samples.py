"""One segmented sample per block face, with the class legend.

The metrics are shares of a segmentation, and a share is only as good as
the labelling underneath it. This puts a face of the actual imagery next to
what the model called it, once per block face, so the categories can be
checked by eye rather than trusted.

WHICH IMAGE, AND WHY IT IS NOT THE PRETTIEST ONE
------------------------------------------------
Per face, the node whose GVI is closest to that face's median, and the
heading closest to the street axis -- the along-street forward view a
pedestrian walking the block actually has. Both choices are made by rule so
that the panel cannot drift towards the photogenic end of the block. The
node's rank within its face is printed with it: if the sample sits at the
median of 20 nodes it is representative, and if the face holds 2 nodes it
is not, and the reader can see which.

The imagery was fetched on 0/90/180/270 true north, while the street grid
runs 029/119/209/299, so the chosen heading is up to 29 degrees off the
street axis. At a 90 degree field of view the corridor is still well inside
the frame; it is simply not centred, and no measurement in the repo depends
on the frame edges -- the azimuthal profiles are assembled from all four.

Categories come from config.yaml -- the same ADE_GROUPS and the
same palette -- so the legend here and the pedestrian-realm numbers can
never disagree. Matching is exact against ADE20K's comma-separated
synonyms, never substring: "tree" is inside "street lamp" and "sky" is
inside "skyscraper".

    python tools/face_samples.py --categories   # what each class contains
    python tools/face_samples.py --dry-run      # which node per face
    python tools/face_samples.py                # segment and draw   [GPU]

Reads cached JPEGs. No network, no Street View requests.
"""
import argparse, sys
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from common import (CFG, PROC, RES, banner, image_path,        # noqa: E402
                    missing_images, DIRECTIONS)

# Read from config, not from tools/pedestrian.py: that module imports torch
# at the top, and the class table and the dry run have to work on a laptop
# with no segmenter. Same three objects either way.
PED = CFG["pedestrian"]
CLIPSEG_PROMPTS = PED["clipseg_prompts"]
# The classes drawn are the Street Interface Matrix terms from the
# morphology study, not the pedestrian-realm grouping. One domain per SIM
# term plus the three denominators the formula names, so a panel can be
# read against the equation rather than against a different taxonomy.
ADE_GROUPS = CFG["sim"]["groups"]
PALETTE = dict(PED["palette"], **CFG["sim"]["palette"])

FIG = RES / "figures" / "face_samples"
# Elevation the stitched strip can carry everywhere: tan(e) <= cos(45).
EL_MAX = 35.0
# A synthetic label id for scaffolding, outside ADE20K's 150. Sidewalk
# sheds are ubiquitous in Manhattan and ADE20K has no class for them, so
# without this they are labelled building or road and disappear into the
# shares of things they are covering.
SCAFFOLD_ID = 300
# Vegetation split at the horizon. A point below the horizon of a pitch-0
# frame stands below the camera -- about 2.5 m, the framework's own
# eye-level ceiling -- at any distance, so the split needs no depth. Above
# it is canopy and balcony planting. Two greens rather than two unrelated
# colours, because they are the same material at different heights.
VEG_EYE_ID, VEG_CANOPY_ID = 301, 302
PER_FIG = 4


def categories():
    """The class table: what each role contains, and its colour."""
    rows = [{"category": g, "colour": PALETTE.get(g, "#3f3f3f"),
             "source": "ADE20K semantic class",
             "matches": ", ".join(v)} for g, v in ADE_GROUPS.items()]
    rows.append({"category": "scaffolding",
                 "colour": PALETTE.get("scaffolding", "#e6550d"),
                 "source": f"open-vocabulary, drawn (threshold "
                           f"{CFG['open_vocabulary']['threshold']}, untuned)",
                 "matches": " | ".join(CFG["open_vocabulary"]["scaffold_prompts"])})
    rows += [{"category": k, "colour": PALETTE.get(k, "#3f3f3f"),
              "source": "open-vocabulary, not drawn (indicative only)",
              "matches": v} for k, v in CLIPSEG_PROMPTS.items()]
    t = pd.DataFrame(rows)
    (RES / "tables").mkdir(parents=True, exist_ok=True)
    out = RES / "tables" / "segmentation_categories.csv"
    t.to_csv(out, index=False)
    banner("SEGMENTATION CATEGORIES")
    for _, r in t.iterrows():
        print(f"  {r.category:12s} {r.colour}  {r.matches[:64]}")
    print(f"\n  ADE20K rows carry the quantitative claims. The"
          f" open-vocabulary rows\n  are indicative: their threshold"
          f" ({CFG['open_vocabulary']['threshold']}) has never been tuned"
          f"\n  against hand-labelled data, so they locate features rather"
          f" than measure them.")
    print(f"\nwrote {out}")
    return t


def pick(nodes, manifest, metrics):
    """One node per block face, and the travel bearing to face from it.

    The bearing is the grid direction closest to the street's own axis, so
    the sample shows the corridor a pedestrian walks rather than an
    arbitrary compass view. All four frames are kept: the 180-degree view
    is assembled from three of them and the fourth is the one behind.
    """
    dm = None
    dmp = PROC / "directional_metrics.csv"
    if dmp.exists():
        dm = pd.read_csv(dmp)
    rows = []
    for fid, g in nodes.groupby("face_id"):
        g = g.dropna(subset=["GVI"])
        if g.empty:
            continue
        med = g.GVI.median()
        r = g.iloc[(g.GVI - med).abs().argsort().iloc[0]]
        axis = r.get("street_axis_deg", np.nan)
        if manifest[manifest.node_id.eq(r.node_id)].empty:
            continue
        if pd.notna(axis):
            d = {k: abs(((v - float(axis) + 180) % 360) - 180)
                 for k, v in DIRECTIONS.items()}
            direction = min(d, key=d.get)
        else:
            direction = list(DIRECTIONS)[0]
        rec = {"face_id": fid, "node_id": r.node_id, "osm_name": r.osm_name,
               "n_nodes": len(g), "GVI_360": r.GVI,
               "face_median_GVI": med, "direction": direction,
               "travel_bearing": DIRECTIONS[direction],
               "street_axis_deg": axis}
        if dm is not None:
            v = dm[dm.node_id.eq(r.node_id) & dm.direction.eq(direction)]
            rec["GVI"] = float(v.GVI.iloc[0]) if len(v) else np.nan
            rec["VEI"] = float(v.VEI.iloc[0]) if len(v) else np.nan
            rec["along_street"] = bool(v.along_street.iloc[0]) if len(v) else None
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["osm_name", "face_id"])


def clipseg_masks(images, pos, neg=(), margin=None, thresh=None):
    """Contrastive CLIPSeg masks for one class, at image resolution.

    score = max over positive prompts - max over negative prompts. A bare
    probability answers "does this look somewhat like X", which a distant
    glass facade answers yes to; subtracting what the same pixels score for
    the ordinary street asks whether it looks MORE like X than like its
    surroundings. With no negatives it falls back to a plain threshold.

    CPU deliberately: CLIPSeg's decoder hits a non-contiguous-tensor error
    under MPS, and at 0.7 s a frame the device is not the bottleneck.
    """
    import torch
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
    ov = CFG["open_vocabulary"]
    thresh = ov["threshold"] if thresh is None else thresh
    cp = CLIPSegProcessor.from_pretrained(ov["model"])
    cm = CLIPSegForImageSegmentation.from_pretrained(ov["model"]).to("cpu").eval()

    def heat(im, prompts):
        inp = cp(text=list(prompts), images=[im] * len(prompts), padding=True,
                 return_tensors="pt")
        with torch.inference_mode():
            pr = torch.sigmoid(cm(**inp).logits).max(dim=0).values
        return torch.nn.functional.interpolate(
            pr[None, None], size=im.size[::-1], mode="bilinear",
            align_corners=False)[0, 0].numpy()

    out = {}
    for key, im in images.items():
        p_ = heat(im, pos)
        out[key] = ((p_ - heat(im, neg)) > margin) if (neg and margin is not None) \
            else (p_ > thresh)
    return out


def scaffold_masks(paths, images, thresh=None):
    """The scaffolding class, as configured. See clipseg_masks."""
    ov = CFG["open_vocabulary"]
    return clipseg_masks(images, ov["scaffold_prompts"],
                         ov.get("scaffold_negative_prompts", []),
                         ov.get("scaffold_contrast"), thresh)


def panorama(frames, centre, fov=None, el_max=None, width=1440):
    """Stitch the cached 90-degree frames into the 180-degree forward view.

    The metrics are computed over a 180-degree cone centred on the bearing
    of travel -- 029 walking uptown, 119 walking east -- but the imagery was
    fetched on 0/90/180/270 and no single frame is that view. A sample panel
    drawn from one raw frame therefore shows something the numbers were
    never computed from, off-axis by up to 29 degrees. This builds the view
    the numbers describe.

    The reprojection is exact, not a paste. Each source frame is a gnomonic
    projection, so a ray at azimuth offset `a` from the frame centre and
    elevation `e` lands at

        x = tan(a) / tan(fov/2)        y = tan(e) / (cos(a) tan(fov/2))

    in normalised image coordinates -- the cos(a) is the part a naive
    horizontal concatenation gets wrong, and it is what makes straight
    verticals bend as they approach a frame edge. Inverting that per output
    cell resamples all three contributing frames onto one azimuth-elevation
    canvas with the seams in the right place.

    Vertical extent is the limit here: a frame covers +/-45 degrees of
    elevation at its centre but only +/-35.3 at its corners, since
    tan(e) <= cos(45). The strip is therefore drawn to +/-35 degrees, which
    is less than the raw frames carry and is stated on the figure. It does
    not change any metric -- the profiles are built per frame, before this.
    """
    # The per-frame capture fov (90), NOT directional.fov -- that one is the
    # 180-degree pedestrian cone this function is assembling, and using it
    # here makes tan(fov/2) diverge and every ray sample the centre pixel.
    fov = fov or CFG["sampling"]["fov"]
    el_max = el_max or EL_MAX
    span = 180.0
    height = int(round(width * (2 * el_max) / span))
    az = centre + np.linspace(-span / 2, span / 2, width)
    el = np.linspace(el_max, -el_max, height)
    A, E = np.meshgrid(np.radians(az), np.radians(el))
    t = np.tan(np.radians(fov / 2))

    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    lab = np.full((height, width), -1, dtype=np.int32)
    for head, (img, seg) in frames.items():
        arr = np.asarray(img)
        H, W = seg.shape
        d = np.radians(((az - head + 180) % 360) - 180)          # per column
        D = np.broadcast_to(d, (height, width))
        near = np.abs(D) <= np.radians(fov / 2)
        x = np.tan(D) / t
        y = np.tan(E) / (np.cos(D) * t)
        ok = near & (np.abs(x) <= 1) & (np.abs(y) <= 1)
        c = np.clip(((x + 1) / 2 * W - 0.5).round().astype(int), 0, W - 1)
        r = np.clip(((1 - y) / 2 * H - 0.5).round().astype(int), 0, H - 1)
        rgb[ok] = arr[r[ok], c[ok]]
        lab[ok] = seg[r[ok], c[ok]]
    return rgb, lab, np.degrees(E)


def draw(sel, strips, groups, chunk, seg=True):
    """Stacked 180-degree strips: what is seen, and what it was called."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    order = list(groups)
    lut = np.zeros(400, dtype=int)
    for gi, g in enumerate(order, start=1):
        for cid in groups[g]:
            lut[cid] = gi
    cmap = ListedColormap(["#3f3f3f"] + [PALETTE[g] for g in order])

    n = len(sel)
    rows = 2 * n if seg else n
    fig, ax = plt.subplots(rows, 1, figsize=(12, 2.55 * rows + 1.1))
    ax = np.atleast_1d(ax)
    for i, (_, r) in enumerate(sel.iterrows()):
        rgb, lab = strips[r.node_id]
        a0 = ax[2 * i] if seg else ax[i]
        a0.imshow(rgb, extent=[-90, 90, -EL_MAX, EL_MAX], aspect="auto")
        gvi = f"GVI {r.GVI:.1f}%" if pd.notna(r.get("GVI")) else ""
        vei = f"VEI {r.VEI:.2f}" if pd.notna(r.get("VEI")) else ""
        a0.set_title(f"{r.osm_name} · {r.face_id} · {r.node_id} — "
                     f"{r.direction.replace('_', ' ')}, bearing "
                     f"{int(r.travel_bearing):03d}, 180° forward view\n"
                     f"{gvi}  {vei}   (face median GVI "
                     f"{r.face_median_GVI:.1f}%, {int(r.n_nodes)} node"
                     f"{'s' if r.n_nodes != 1 else ''})", fontsize=8.5)
        if seg:
            a1 = ax[2 * i + 1]
            a1.imshow(rgb, extent=[-90, 90, -EL_MAX, EL_MAX], aspect="auto")
            a1.imshow(lut[np.where(lab < 0, 0, lab)],
                      extent=[-90, 90, -EL_MAX, EL_MAX], aspect="auto",
                      cmap=cmap, vmin=0, vmax=len(order), alpha=.5,
                      interpolation="nearest")
        for a_ in ([a0, ax[2 * i + 1]] if seg else [a0]):
            a_.set_yticks([-30, 0, 30])
            a_.set_yticklabels(["-30°", "horizon", "+30°"], fontsize=7)
            a_.set_xticks([-90, -45, 0, 45, 90])
            a_.set_xticklabels(["90° left", "45°", "ahead", "45°",
                                "90° right"], fontsize=7)
    handles = [Patch(color=PALETTE[g], label=g) for g in order] if seg else []
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
                   bbox_to_anchor=(.5, -.004), fontsize=9)
    fig.suptitle("Vertical extent is ±35°: a 90° frame reaches ±45° at its "
                 "centre but only ±35.3° at its corners, and the strip is "
                 "drawn to what every column can carry.",
                 y=.998, fontsize=8, color="#666")
    plt.tight_layout(rect=[0, .03 if handles else 0, 1, .985])
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / (f"face_samples_{chunk:02d}.png" if seg
                 else f"face_images_{chunk:02d}.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", action="store_true",
                    help="print and write the class table, then stop")
    ap.add_argument("--dry-run", action="store_true",
                    help="show the chosen node per face without segmenting")
    ap.add_argument("--images-only", action="store_true",
                    help="draw the chosen frames with no overlay (no GPU)")
    a = ap.parse_args()

    if a.categories:
        categories()
        return

    nodes = pd.read_csv(PROC / "nodes_with_faces.csv")
    manifest = pd.read_csv(PROC / "manifest.csv")
    metrics = pd.read_csv(PROC / "metrics.csv")
    if "street_axis_deg" not in nodes.columns:
        nodes = nodes.merge(metrics[["node_id", "street_axis_deg"]],
                            on="node_id", how="left")
    sel = pick(nodes, manifest, metrics)
    banner(f"FACE SAMPLES  {len(sel)} block faces")
    cols = [c for c in ["face_id", "osm_name", "node_id", "n_nodes",
                        "direction", "travel_bearing", "GVI", "VEI",
                        "GVI_360", "face_median_GVI"] if c in sel.columns]
    print(sel[cols].round(2).to_string(index=False))
    thin = sel[sel.n_nodes < 2]
    if len(thin):
        print(f"\n  {len(thin)} face(s) hold a single node "
              f"({', '.join(thin.face_id)}). They are drawn because a"
              f" sample of\n  the imagery should show every face, but they"
              f" are below the >=2-node\n  rule s04 applies to"
              f" block_faces.csv and carry no inference.")
    sel.to_csv(RES / "tables" / "face_samples.csv", index=False)
    gone = missing_images(manifest[manifest.node_id.isin(sel.node_id)].path)
    if gone:
        print(f"\n  !! {len(gone)} sample images are not on disk, "
              f"e.g. {gone[0]}")
    if a.dry_run:
        print("\ndry run -- nothing segmented")
        return

    manifest_by_node = {k: g for k, g in manifest.groupby("node_id")}

    if a.images_only:
        from PIL import Image
        strips = {}
        for _, r in sel.iterrows():
            frames = {}
            for _, q in manifest_by_node[r.node_id].iterrows():
                im = Image.open(image_path(q.path)).convert("RGB")
                frames[float(q.heading)] = (
                    im, np.zeros(np.asarray(im).shape[:2], dtype=np.int32))
            rgb, lab, _ = panorama(frames, float(r.travel_bearing))
            strips[r.node_id] = (rgb, lab)
        written = [draw(sel.iloc[i:i + PER_FIG], strips, {}, i // PER_FIG + 1,
                        seg=False)
                   for i in range(0, len(sel), PER_FIG)]
        categories()
        print(f"\nwrote {len(written)} image panels to {FIG}")
        print("No segmentation: run without --images-only on a machine with"
              " torch\nto add the overlay row.")
        return

    try:
        import torch
        from PIL import Image
        from common import device_and_batch, load_segmenter
        from pedestrian import resolve
    except ModuleNotFoundError as e:
        sys.exit(f"{e.name} is missing: this is the analysis environment, "
                 f"which has no segmenter. --categories, --dry-run and "
                 f"--images-only work here; segment on the GPU box.")

    dev, _ = device_and_batch()
    proc, model = load_segmenter(dev)
    groups = {g: resolve(model.config.id2label, v)
              for g, v in ADE_GROUPS.items()}
    groups = {g: v for g, v in groups.items() if v}

    strips, rows = {}, []
    for k, (_, r) in enumerate(sel.iterrows(), start=1):
        frames, images = {}, {}
        for _, q in manifest_by_node[r.node_id].iterrows():
            img = Image.open(image_path(q.path)).convert("RGB")
            inp = proc(images=[img], return_tensors="pt").to(dev)
            with torch.inference_mode():
                out = model(**inp)
            out.class_queries_logits = out.class_queries_logits.cpu().contiguous()
            out.masks_queries_logits = out.masks_queries_logits.cpu().contiguous()
            seg = proc.post_process_semantic_segmentation(
                out, target_sizes=[img.size[::-1]])[0].numpy()
            frames[float(q.heading)] = (img, seg)
            images[float(q.heading)] = img
        # Reproject the LABELS, never re-segment the stitched strip: the
        # metrics come from the four frames, so an overlay built any other
        # way could disagree with the number printed beside it.
        rgb, lab, elev = panorama(frames, float(r.travel_bearing))
        # Split vegetation at the horizon, on the strip rather than on the
        # frames: elevation is a property of the output grid, so it is one
        # comparison here instead of a per-frame row mask.
        veg = np.isin(lab, groups.get("eye_green", []))
        lab = np.where(veg & (elev < 0), VEG_EYE_ID,
                       np.where(veg, VEG_CANOPY_ID, lab))
        strips[r.node_id] = (rgb, lab)
        rec = {"face_id": r.face_id, "node_id": r.node_id,
               "osm_name": r.osm_name, "direction": r.direction,
               "travel_bearing": r.travel_bearing}
        valid = lab >= 0
        share_groups = dict(groups)
        share_groups.pop("eye_green", None)
        share_groups["eye_green"] = [VEG_EYE_ID]
        share_groups["canopy_green"] = [VEG_CANOPY_ID]
        for g, ids in share_groups.items():
            rec[g] = float(np.isin(lab[valid], ids).mean())
        rows.append(rec)
        print(f"  [{k}/{len(sel)}] {r.face_id:5s} {r.osm_name:18s} "
              f"{r.node_id}  {r.direction}")

    pd.DataFrame(rows).to_csv(RES / "tables" / "face_sample_shares.csv",
                              index=False)
    # Draw the split classes, not the parent: "vegetation" no longer
    # appears in any label map once the horizon test has run.
    groups = {k: v for k, v in groups.items() if k != "eye_green"}
    groups["eye_green"] = [VEG_EYE_ID]
    groups["canopy_green"] = [VEG_CANOPY_ID]
    written = [draw(sel.iloc[i:i + PER_FIG], strips, groups, i // PER_FIG + 1)
               for i in range(0, len(sel), PER_FIG)]
    categories()
    print(f"\nwrote {len(written)} panels to {FIG}")
    print("wrote face_samples.csv, face_sample_shares.csv")
    print("Shares are pixel counts over the drawn strip: same 180-degree"
          " azimuth\nas the printed GVI, but +/-35 degrees of elevation"
          " rather than +/-45, and\nnot solid-angle weighted. They track"
          " the metric closely (rho 0.99) and\nrun about a point high,"
          " because the excluded top band is mostly sky.\nRead them as a"
          " check on the labelling, never as the metric.")


if __name__ == "__main__":
    main()
