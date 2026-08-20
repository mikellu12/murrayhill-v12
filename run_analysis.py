"""Laptop entry point -- every analysis stage, no GPU required.

The only stage that needs a GPU is s03 (segmentation). Once
data/processed/azimuth_profiles.npz exists, everything downstream is CPU
work that finishes in seconds. So copy data/processed/ from the machine
that ran the pipeline, and run this anywhere.

    python run_analysis.py              everything available
    python run_analysis.py --skip sidewalk

Needs no torch, no CUDA, no API key, no network -- except the sidewalk
step, which downloads NYC Open Data polygons once and caches them.

skyview.py is gone in v12: SVF_band and BVF_band are computed inside s04
straight from the azimuthal profiles, so there is no separate stage to run
and no chance of the two definitions drifting apart.
"""
import argparse, subprocess, sys, importlib
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE / "tools"))

ap = argparse.ArgumentParser()
ap.add_argument("--skip", nargs="*", default=[])
a = ap.parse_args()

PROC = HERE / "data" / "processed"
need = {"metrics.gpkg": "s04", "directional_metrics.csv": "s04",
        "azimuth_profiles.npz": "s03"}
missing = [f for f in need if not (PROC / f).exists()]
if missing:
    print("missing inputs in data/processed:")
    for f in missing:
        print(f"  {f}  (produced by {need[f]})")
    sys.exit("\nCopy data/processed/ from the machine that ran the pipeline.")


def stage(name, fn):
    if name in a.skip:
        print(f"\n--- skipping {name} ---")
        return
    print(f"\n{'#'*74}\n# {name}\n{'#'*74}")
    try:
        fn()
    except ImportError as e:
        print(f"  needs a package: {e}")
    except Exception as e:
        print(f"  failed: {type(e).__name__}: {e}")


def _script(mod):
    def go():
        r = subprocess.run([sys.executable, str(HERE / "src" / f"{mod}.py")])
        if r.returncode:
            raise RuntimeError(f"exit {r.returncode}")
    return go


stage("s06 analysis", _script("s06_analysis"))
stage("s07 enclosure", _script("s07_enclosure"))
stage("s08 figures", _script("s08_figures"))


def _sidewalk():
    sw = importlib.import_module("sidewalk")
    sw.run(nodes_path=str(PROC / "nodes.gpkg"),
           dm_path=str(PROC / "directional_metrics.csv"),
           raw_dir=str(HERE / "data" / "raw"),
           out_dir=str(HERE / "results" / "tables"))


stage("sidewalk and setback", _sidewalk)

print("\nresults/tables and results/figures updated")
