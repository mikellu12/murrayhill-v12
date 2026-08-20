"""Shared configuration, paths, keys and geometry helpers."""
import os, sys, yaml
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
SEED = CFG["seed"]

RAW = ROOT / CFG["paths"]["raw"]
PROC = ROOT / CFG["paths"]["processed"]
IMG = ROOT / CFG["paths"]["imagery"]
RES = ROOT / CFG["paths"]["results"]
for _d in (RAW, PROC, IMG, RES / "figures", RES / "tables"):
    _d.mkdir(parents=True, exist_ok=True)

TYPOLOGY_ORDER = ["avenue_canyon", "avenue_secondary", "mid_block"]
PALETTE = {"avenue_canyon": "#c0392b", "avenue_secondary": "#e08214",
           "mid_block": "#2c7a4b"}

GRID = CFG["directional"]["grid_bearing"]
DIRECTIONS = {"N_uptown": GRID % 360, "E_east": (GRID + 90) % 360,
              "S_downtown": (GRID + 180) % 360, "W_west": (GRID + 270) % 360}


def key(name):
    v = os.environ.get(name, "")
    if not v and (ROOT / ".env").exists():
        for line in (ROOT / ".env").read_text().splitlines():
            if line.strip().startswith(name):
                v = line.split("=", 1)[1].strip().strip("\"'")
    return v


def require(name):
    v = key(name)
    if not v:
        sys.exit(f"{name} not set. Export it or add it to {ROOT/'.env'}")
    return v


def banner(text):
    print(f"\n{'='*72}\n{text}\n{'='*72}")


def norm_name(v):
    """OSM returns a list when an edge carries several names."""
    if isinstance(v, list):
        return " | ".join(str(x) for x in v)
    return str(v) if pd.notna(v) else ""


def typology_of(names):
    """Three-way split from street name. THE canonical definition.

    Every stage derives typology from here, so the label set cannot drift
    between the frame and the metrics -- which is how 'avenue_canyon'
    previously ended up absent from the plots while present in the frame.
    """
    sa = CFG["study_area"]
    s = pd.Series(list(names)).astype(str)
    return np.select(
        [s.str.contains(sa["canyon_pattern"], case=False, na=False),
         s.str.contains(sa["secondary_pattern"], case=False, na=False)],
        ["avenue_canyon", "avenue_secondary"], default="mid_block")


# zone_of() is gone. It cut the frame by latitude across a grid rotated
# ~29 deg, so the boundary ran diagonally through the street pattern and
# split all 15 streets between two zones. The north-south gradient it was
# encoding is now continuous; see gridaxis.py at the project root.
sys.path.insert(0, str(ROOT))
from gridaxis import grid_northing, grid_easting        # noqa: E402


def device_and_batch():
    """Pick a compute device: CUDA, then Apple MPS, then CPU.

    v11 uses only Mask2Former and CLIPSeg, both of which run on MPS. (The
    earlier LLaVA stage did not -- bitsandbytes has no MPS backend -- which
    is why previous versions were CUDA-only.)

    CPU works and is the fallback, but expect roughly 4x the MPS time.
    """
    import torch, platform
    b = CFG["segmentation"]["batch_size"]

    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 2**30
        if b == "auto":
            b = 16 if vram >= 32 else 8 if vram >= 14 else 4 if vram >= 10 else 2
        print(f"device: cuda -- {torch.cuda.get_device_name(0)} "
              f"({vram:.1f} GiB), batch {b}")
        return "cuda", int(b)

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        # Some ops still lack MPS kernels; this makes them fall back to CPU
        # rather than raising. Set before torch dispatches anything.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        if b == "auto":
            b = 4
        print(f"device: mps -- Apple Silicon ({platform.machine()}), batch {b}")
        print("  roughly 1.5-2x the wall time of a mid-range NVIDIA card")
        return "mps", int(b)

    if b == "auto":
        b = 2
    print(f"device: cpu ({platform.machine()}), batch {b}")
    print("  WARNING: segmentation on CPU takes hours, not minutes.")
    print("  If a GPU machine has already produced "
          "data/processed/azimuth_profiles.npz, copy it across and run")
    print("  the analysis stages only: python main.py --from s04")
    return "cpu", int(b)


def load_segmenter(dev):
    from transformers import (AutoImageProcessor,
                              Mask2FormerForUniversalSegmentation)
    m = CFG["segmentation"]["model"]
    proc = AutoImageProcessor.from_pretrained(m)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(m).to(dev).eval()
    return proc, model


def resolve_classes(id2label):
    """Match by LABEL NAME, never index -- ADE20K ordering varies."""
    seg = CFG["segmentation"]

    def ids(names):
        return [int(i) for i, lab in id2label.items()
                if any(n in [p.strip().lower() for p in lab.split(",")]
                       for n in names)]

    veg, sky, bld = (ids(seg["vegetation_labels"]), ids(seg["sky_labels"]),
                     ids(seg["building_labels"]))
    for nm, v in [("vegetation", veg), ("sky", sky), ("building", bld)]:
        print(f"  {nm:12s} {[id2label[i] for i in v]}")
    assert veg and sky and bld, "class resolution failed"
    return veg, sky, bld


def image_path(p):
    """Locate a manifest image regardless of where the manifest was written.

    manifest.csv stores absolute paths, so a run carried between machines
    or checkouts points at a directory that no longer exists -- this v12
    frame inherited a v11 manifest addressing ~/Downloads, and every one of
    its 2,316 paths was dead while the JPEGs sat in data/raw/svi. Filenames
    are unique per node and heading, so resolving by basename against
    paths.imagery is unambiguous and survives the move.

    The stored path wins when it exists, so pointing the manifest at a
    deliberate alternate location still works.
    """
    p = Path(p)
    return p if p.exists() else IMG / p.name


def missing_images(paths):
    """Which of these are not on disk, after resolution. Empty is good."""
    return [str(q) for q in paths if not image_path(q).exists()]


# ------------------------------------------------------- image geometry
def column_bearings(heading, width, fov=None):
    """Absolute bearing of each pixel column in a gnomonic image.

    A fov-90 image is a perspective projection: column at normalised x sits
    at atan(x * tan(fov/2)) from centre, NOT at a linear fraction. At 29 deg
    off centre the correct column is 497 of 640, not 526 -- a linear reading
    misplaces every boundary by ~29 px.
    """
    fov = fov or CFG["sampling"]["fov"]
    x = (np.arange(width) + 0.5) / width * 2 - 1
    return (heading + np.degrees(np.arctan(x * np.tan(np.radians(fov / 2))))) % 360


def column_weights(height, width, fov=None):
    """Solid angle per column, summed over rows, normalised to 1.

    A plane projection compresses solid angle toward the edges: corner
    pixels subtend ~1/5 of centre pixels, so uniform counting over-weights
    corners by 2.7x. Treepedia does not correct for this.
    """
    fov = fov or CFG["sampling"]["fov"]
    if not CFG["sampling"]["solid_angle_weighting"]:
        return np.full(width, 1.0 / width)
    x = (np.arange(width) + 0.5) / width * 2 - 1
    y = (np.arange(height) + 0.5) / height * 2 - 1
    t = np.tan(np.radians(fov / 2))
    X, Y = np.meshgrid(x * t, y * t)
    w = (1 + X**2 + Y**2) ** -1.5
    return w.sum(axis=0) / w.sum()


def bin_mask(centre, fov, nbins=None):
    nbins = nbins or CFG["directional"]["n_bins"]
    d = (np.arange(nbins) + 0.5 - centre) % 360
    d = np.where(d > 180, d - 360, d)
    return np.abs(d) <= fov / 2


def slice_metrics(prof, centre, fov):
    """GVI (%) and VEI over one angular window.

    prof is 4 x nbins: vegetation, sky, building, and total column weight.
    Dividing by the weight row is what makes this correct -- it normalises
    away both the number of images covering a bin and the varying solid
    angle across an image face. Without it, GVI scales with the number of
    contributing images rather than being a share.
    """
    m = bin_mask(centre, fov)
    W = prof[3][m].sum() if prof.shape[0] > 3 else m.sum() / prof.shape[1]
    if W <= 0:
        return np.nan, np.nan
    veg, sky, bld = (prof[i][m].sum() for i in range(3))
    gvi = 100 * veg / W
    vei = bld / (sky + bld) if (sky + bld) > 0 else np.nan
    return gvi, vei


def street_axis(nodes):
    """Street bearing mod 180 per node, from neighbours on the same chain."""
    key_col = "chain" if "chain" in nodes.columns else "osm_name"
    out = {}
    for _, grp in nodes.groupby(key_col):
        grp = grp.sort_values("chain_pos_m") if "chain_pos_m" in grp else grp
        xs, ys = grp.geometry.x.values, grp.geometry.y.values
        for i, nid in enumerate(grp.node_id.values):
            j = min(max(i + 1 if i == 0 else i - 1, 0), len(xs) - 1)
            dx, dy = xs[i] - xs[j], ys[i] - ys[j]
            out[nid] = (np.degrees(np.arctan2(dx, dy)) % 180
                        if np.hypot(dx, dy) > 1 else np.nan)
    return pd.Series(out)


# ----------------------------------------------- canyon sky geometry
# Kept from the retired skyview.py: the closed-form check that the
# measured band sky fraction behaves like the geometry says it should.
BAND_DEG = 45.0        # elevation half-angle actually imaged


def theoretical_svf(hw):
    """Hemispherical SVF at the floor of an infinite symmetric canyon."""
    return np.cos(np.arctan(2.0 * np.asarray(hw, dtype=float)))


def theoretical_svf_band(hw, band_deg=BAND_DEG):
    """The same canyon restricted to the imaged elevation band.

    Sky is visible above elevation arctan(2H/W); within a band of half-angle
    b the visible share is the fraction of elevations between that angle and
    b. It hits zero once buildings rise above the band entirely, which is why
    band-limited values go flat at low H/W -- a real limitation of pitch-0
    imagery, not a modelling choice.
    """
    hw = np.asarray(hw, dtype=float)
    blocked = np.degrees(np.arctan(2.0 * hw))
    return np.clip(band_deg - blocked, 0, band_deg) / band_deg
