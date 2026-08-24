"""
One command: a CSV of nodes in, a rendered exploded axonometric out.

    .venv/bin/python tools/render_stack.py --csv nodes.csv
    .venv/bin/python tools/render_stack.py --csv nodes.csv --metrics GVI VEI dwell
    .venv/bin/python tools/render_stack.py            # repo metrics.gpkg

The CSV needs one row per node and either lat/lon columns or a node_id to
join against a frame. Every numeric column that is not frame bookkeeping
becomes a plane, bottom to top, in the order given by --metrics or in
column order if that is omitted. Adding a metric later means adding a
column; nothing in this repo needs editing for it to be drawn.

Two stages, either runnable alone:
  tools/export_gis.py    csv  -> GeoPackage (QGIS/Qgis2threejs) + JSON
  tools/blender_axo.py   JSON -> rendered PNG            [needs Blender]
"""
import argparse, os, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLENDER_CANDIDATES = [
    os.environ.get("BLENDER"),
    "/Applications/Blender.app/Contents/MacOS/Blender",
    shutil.which("blender"),
]


def find_blender():
    for c in BLENDER_CANDIDATES:
        if c and Path(c).exists():
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--frame", default=None)
    ap.add_argument("--metrics", nargs="*", default=None)
    ap.add_argument("--pair", nargs="+", action="append", default=None,
                    metavar="A B [NAME]",
                    help="opposed 180-deg view pair -> level + asymmetry "
                         "planes. Repeatable. See tools/export_gis.py.")
    ap.add_argument("--gpkg", default="results/gis/murrayhill_layers.gpkg")
    ap.add_argument("--out", default="results/figures/figure_axo_3d.png")
    ap.add_argument("--res", type=int, default=2200)
    ap.add_argument("--samples", type=int, default=96)
    ap.add_argument("--z-scale", type=float, default=3.4)
    ap.add_argument("--no-render", action="store_true",
                    help="build the GeoPackage and JSON, skip Blender")
    a = ap.parse_args()

    py = sys.executable
    cmd = [py, str(ROOT / "tools/export_gis.py"), "--out", a.gpkg]
    if a.csv:
        cmd += ["--csv", a.csv]
    if a.frame:
        cmd += ["--frame", a.frame]
    if a.metrics is not None:
        cmd += ["--metrics", *a.metrics]
    for pr in a.pair or []:
        cmd += ["--pair", *pr]
    print("$ " + " ".join(cmd))
    if subprocess.call(cmd, cwd=ROOT) != 0:
        sys.exit("export stage failed")
    if a.no_render:
        return

    blender = find_blender()
    if not blender:
        print("\nBlender not found. The GeoPackage and JSON are written, so "
              "the QGIS/Qgis2threejs route still works. Set $BLENDER or "
              "install Blender to render here.")
        return
    js = str(Path(a.gpkg).with_suffix(".json"))
    cmd = [blender, "--background", "--python",
           str(ROOT / "tools/blender_axo.py"), "--",
           "--json", js, "--out", a.out, "--res", str(a.res),
           "--samples", str(a.samples), "--z-scale", str(a.z_scale)]
    print("\n$ " + " ".join(cmd[:4]) + " ... ")
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.startswith("WROTE") or "Saved:" in line:
            print(line)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        sys.exit("render stage failed")


if __name__ == "__main__":
    main()
