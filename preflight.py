"""
Preflight: is this folder ready to run, and what will it cost?

    python preflight.py

Checks which inputs are present and reports, for each one that is missing,
which stage will regenerate it and what that stage costs in time, money and
risk. Run it before `main.py`, not after.

THE ONE THAT MATTERS
--------------------
`data/processed/azimuth_profiles.npz` is the segmented output of Stage 3.
It is a few MB and it represents 35 minutes to 6 hours of GPU or CPU work
depending on the machine. Copy it across and Stage 3 skips entirely.

The JPEGs in `data/raw/svi/` are NOT a substitute for it. They are the
INPUT to segmentation, not the output. Bringing the images without the
profiles means re-segmenting all ~2,300 of them.

THE TRAP WORTH KNOWING ABOUT
----------------------------
If `data/processed/nodes.gpkg` is missing, Stage 1 rebuilds the frame by
re-querying OSM. That is free, but node IDs are assigned sequentially from
n00000 against whatever geometry comes back. If OSM has changed at all
since the original run -- a single added service road inside the bbox is
enough -- the count shifts, every ID after that point moves, and the
existing JPEGs and profiles now belong to different locations.

Stage 2 catches this and aborts rather than letting it through, which is
correct but means you are then forced into a full re-fetch and re-segment.
Copying nodes.gpkg across avoids the whole question.
"""
import sys
from pathlib import Path

HERE = Path(__file__).parent
PROC = HERE / "data" / "processed"
RAW = HERE / "data" / "raw"
SVI = RAW / "svi"

# path, label, regenerating stage, cost if absent, blocking?
CHECKS = [
    (PROC / "azimuth_profiles.npz", "segmented profiles", "s03",
     "35 min - 6 h of GPU/CPU segmentation", "expensive"),
    (PROC / "nodes.gpkg", "sampling frame", "s01",
     "free, but re-queries OSM -- node IDs may shift, see the docstring",
     "risky"),
    (PROC / "manifest.csv", "image manifest", "s02",
     "free if the JPEGs are present; s02 rebuilds it by walking them",
     "cheap"),
    (RAW / "metadata.csv", "Street View metadata probe", "s02",
     "free (the metadata endpoint is unmetered) but needs GMAPS_KEY",
     "cheap"),
    (RAW / "building_footprints.geojson", "NYC building footprints", "s05",
     "one download from NYC Open Data, needs network", "cheap"),
    (PROC / "scaffold_by_node.csv", "scaffolding fractions", "s03",
     "re-runs CLIPSeg over every image", "expensive"),
]

OPTIONAL = [
    (PROC / "metrics.gpkg", "s04 rebuilds this from the profiles"),
    (PROC / "directional_metrics.csv", "s04 rebuilds this"),
    (PROC / "block_faces.csv", "s04 rebuilds this"),
    (PROC / "nodes_with_faces.csv", "s04 rebuilds this"),
]

MARK = {"expensive": "!!", "risky": "!!", "cheap": " ~"}

print(__doc__.split("    python preflight.py")[0].strip())
print("\n" + "=" * 72)
print("REQUIRED INPUTS")
print("=" * 72)

missing_expensive, missing_risky = [], []
for path, label, stage, cost, kind in CHECKS:
    ok = path.exists()
    rel = path.relative_to(HERE)
    if ok:
        size = path.stat().st_size / 1e6
        print(f"  ok  {label:28s} {str(rel):42s} {size:7.1f} MB")
    else:
        print(f"  {MARK[kind]}  {label:28s} {str(rel):42s}  MISSING")
        print(f"      -> {stage} would regenerate it: {cost}")
        if kind == "expensive":
            missing_expensive.append(label)
        elif kind == "risky":
            missing_risky.append(label)

n_jpg = len(list(SVI.glob("*.jpg"))) if SVI.is_dir() else 0
print(f"\n  {'ok' if n_jpg else ' ~'}  {'Street View JPEGs':28s} "
      f"{str(SVI.relative_to(HERE)):42s} {n_jpg:5d} files")
if n_jpg and (PROC / "azimuth_profiles.npz").exists():
    print("      The profiles already exist, so these are not needed for the")
    print("      analysis stages. Keep them only if you plan to re-segment,")
    print("      or for tools/pedestrian.py, which reads the imagery")
    print("      directly. tools/fov_check.py needs only the profiles;")
    print("      tools/cubemap_check.py fetches its own from the API.")
elif n_jpg:
    print("      Present, but without the profiles these must all be")
    print("      re-segmented. Copy azimuth_profiles.npz if you have it.")

print("\n" + "=" * 72)
print("OPTIONAL (regenerated downstream -- copying them saves nothing)")
print("=" * 72)
for path, note in OPTIONAL:
    print(f"  {'ok ' if path.exists() else '   '} "
          f"{str(path.relative_to(HERE)):46s} {note}")

# ------------------------------------------------------------ frame check
print("\n" + "=" * 72)
print("CONSISTENCY")
print("=" * 72)
try:
    import geopandas as gpd
    import numpy as np
    import pandas as pd
except ImportError as e:
    sys.exit(f"  cannot check further: {e}")

np_ = PROC / "nodes.gpkg"
if np_.exists():
    nodes = gpd.read_file(np_)
    ids = set(nodes.node_id)
    print(f"  frame holds {len(nodes)} nodes")
    print(f"  grid axis present: "
          f"{'yes' if 'northing_m' in nodes.columns else 'NO -- run migrate_gridaxis.py --apply'}")
    if "zone" in nodes.columns:
        print("  !! a `zone` column is still on the frame -- "
              "run migrate_gridaxis.py --apply")

    npz = PROC / "azimuth_profiles.npz"
    if npz.exists():
        keys = set(np.load(npz).files)
        extra = keys - ids
        print(f"  profiled nodes: {len(keys)}"
              f"{f'  ({len(keys & ids)} match the frame)' if keys else ''}")
        if extra:
            print(f"  !! {len(extra)} profiled node IDs are NOT in the frame "
                  f"(e.g. {sorted(extra)[:3]}).")
            print("     data/processed is from a different frame. s04 will")
            print("     abort. Do not proceed -- start from a clean copy of")
            print("     one run, not a mix of two.")
        else:
            print("  profiles and frame agree -- safe to run --from s04")

    if n_jpg:
        stale = [p for p in SVI.glob("*.jpg")
                 if p.stem.rsplit("_", 1)[0] not in ids]
        if stale:
            print(f"  !! {len(stale)} JPEGs belong to node IDs not in the "
                  f"frame. s02 will abort. Same diagnosis as above.")
else:
    print("  no frame -- cannot cross-check IDs")

# ------------------------------------------------------------ verdict
print("\n" + "=" * 72)
print("WHAT TO RUN")
print("=" * 72)
have_prof = (PROC / "azimuth_profiles.npz").exists()
have_frame = np_.exists()

if have_prof and have_frame:
    print("  python migrate_gridaxis.py --apply     # if the axis check said NO")
    print("  python main.py --from s04              # seconds, CPU only")
    print("  python make_dashboard.py")
    print("\n  No GPU, no GMAPS_KEY, no network needed (except s05's footprint")
    print("  download, and only if building_footprints.geojson is absent).")
elif have_frame and not have_prof:
    print("  Stage 3 will re-segment. Check the machine first:")
    print("    python -c \"import torch; print(torch.cuda.is_available())\"")
    print("  Then: python main.py --from s03")
    print("\n  If azimuth_profiles.npz exists on the machine that ran v11,")
    print("  copy it instead. It is a few MB and saves the whole stage.")
else:
    print("  No frame present. python main.py will run everything from s01,")
    print("  which needs GMAPS_KEY in a .env beside config.yaml and will")
    print("  re-fetch ~2,300 images (~$16 at list price, free tier covers it).")
    print("\n  If a v11 run exists anywhere, copy data/processed/ across first.")

if missing_expensive:
    print(f"\n  !! re-doing expensive work for: {', '.join(missing_expensive)}")
if missing_risky:
    print(f"  !! frame will be rebuilt -- see the docstring on ID drift")
