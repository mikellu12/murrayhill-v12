"""The SIM axonometric, rendered in Blender rather than drawn in matplotlib.

Same figure as tools/sim_axonometric.py and the same exact node_id join --
only the renderer differs. Blender buys ambient occlusion, real shadows and
clean line work; it changes no number.

Runs headless and is driven entirely by this file, so it stays reproducible:

    .venv/Scripts/python tools/sim_axonometric_blender.py

That builds a geometry JSON, then invokes

    blender --background --python <this file> -- <json> <out.png>

A hand-built .blend would not be reproducible, which is the whole reason the
scene is constructed in script rather than saved.

The two halves live in one file on purpose. Blender ships its own Python and
cannot import geopandas, while the analysis env cannot import bpy; splitting
them across two files invites the prompt and the geometry drifting apart, the
same failure the projection test avoids by importing its schema.
"""
import json
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------- blender side
try:
    import bpy  # noqa: F401  only inside Blender
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False


def render(spec_path, out_path):
    import bpy

    spec = json.loads(Path(spec_path).read_text())
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene

    def mat(name, rgba, rough=0.6):
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        b = m.node_tree.nodes["Principled BSDF"]
        b.inputs["Base Color"].default_value = rgba
        b.inputs["Roughness"].default_value = rough
        # Workbench's MATERIAL colour mode reads the VIEWPORT display colour,
        # not the BSDF. Set only the BSDF and every object renders in the
        # default grey -- which is what hid the red tail bars entirely.
        m.diffuse_color = rgba
        m.roughness = rough
        return m

    white = mat("fabric", (0.97, 0.97, 0.97, 1.0), 0.85)
    grey = mat("bar", (0.62, 0.62, 0.62, 1.0), 0.6)
    red = mat("tail", (0.62, 0.16, 0.13, 1.0), 0.5)

    def mesh_from(name, verts, faces, material):
        me = bpy.data.meshes.new(name)
        me.from_pydata(verts, [], faces)
        me.validate()
        ob = bpy.data.objects.new(name, me)
        ob.data.materials.append(material)
        sc.collection.objects.link(ob)
        return ob

    # Footprints: one mesh for all of them, so the scene stays light.
    verts, faces = [], []
    for poly, h in spec["footprints"]:
        n = len(poly)
        base = len(verts)
        verts += [(x, y, 0.0) for x, y in poly] + [(x, y, h) for x, y in poly]
        for i in range(n):
            j = (i + 1) % n
            faces.append([base + i, base + j, base + n + j, base + n + i])
        faces.append([base + n + i for i in range(n)])
    if verts:
        mesh_from("built_fabric", verts, faces, white)

    # Bars: square prisms, batched into one mesh per colour.
    w = spec["bar_width"]
    for key, material in (("grey", grey), ("red", red)):
        verts, faces = [], []
        for x, y, z0, h in spec["bars"][key]:
            base = len(verts)
            sq = [(x - w, y - w), (x + w, y - w), (x + w, y + w), (x - w, y + w)]
            verts += [(a, b, z0) for a, b in sq] + [(a, b, z0 + h) for a, b in sq]
            for i in range(4):
                j = (i + 1) % 4
                faces.append([base + i, base + j, base + 4 + j, base + 4 + i])
            faces.append([base + 4 + i for i in range(4)])
        if verts:
            mesh_from(f"bars_{key}", verts, faces, material)

    span = spec["span"]
    zmid, zext = spec["zmid"], spec["zext"]

    # Aim with a TRACK_TO constraint rather than hand-set Euler angles: the
    # first attempt guessed the rotation and rendered an empty frame. An
    # empty at the scene centre cannot miss.
    target = bpy.data.objects.new("target", None)
    target.location = (0.0, 0.0, zmid)
    sc.collection.objects.link(target)

    cam_data = bpy.data.cameras.new("cam")
    cam_data.type = "ORTHO"          # axonometric, not perspective
    # The stack is taller than the plan is wide, so the vertical extent sets
    # the frame, not the span.
    cam_data.ortho_scale = max(span, zext) * 1.25
    cam_data.clip_end = max(span, zext) * 12
    cam = bpy.data.objects.new("cam", cam_data)
    sc.collection.objects.link(cam)
    d_ = max(span, zext) * 2.2
    cam.location = (d_ * 0.62, -d_ * 0.72, zmid + d_ * 0.55)
    con = cam.constraints.new("TRACK_TO")
    con.target = target
    con.track_axis = "TRACK_NEGATIVE_Z"
    con.up_axis = "UP_Y"
    sc.camera = cam

    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = 2.0
    sun.rotation_euler = (0.6, 0.1, 0.9)
    sc.collection.objects.link(sun)
    sc.world = bpy.data.worlds.new("w")
    sc.world.use_nodes = True
    sc.world.node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)
    sc.world.node_tree.nodes["Background"].inputs[1].default_value = 1.0

    # WORKBENCH, not EEVEE. The reference plate is a drafted axonometric --
    # flat solids with a dark outline on every edge -- which is what Workbench
    # draws natively. EEVEE rendered it correctly and washed out: white solids
    # lit by a white world sit at the same value as the background, so the
    # geometry vanished into it.
    # Try the assignment rather than testing membership first: in background
    # mode the engine enum reports only its CURRENT value, so a membership
    # test rejects every engine except the default and silently leaves EEVEE
    # in place. Assignment itself works and raises TypeError if it does not.
    for want in ("BLENDER_WORKBENCH", "BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        try:
            sc.render.engine = want
            break
        except TypeError:
            continue
    print("engine:", sc.render.engine)
    if sc.render.engine == "BLENDER_WORKBENCH":
        sh = sc.display.shading
        # Workbench ignores the world entirely; its background comes from
        # shading.background_type, which defaults to the theme grey. Left
        # alone it renders light geometry on mid-grey and everything washes.
        sh.background_type = "VIEWPORT"
        sh.background_color = (1.0, 1.0, 1.0)
        sh.light = "STUDIO"
        sh.color_type = "MATERIAL"
        sh.show_object_outline = True
        sh.object_outline_color = (0.18, 0.18, 0.18)
        sh.show_shadows = True
        sh.shadow_intensity = 0.25
        sh.show_cavity = True
        sh.cavity_type = "BOTH"
        sc.display.render_aa = "32"

    # Standard, not AgX. Blender's default view transform is a filmic tone
    # map that renders pure white at about 197/255 -- which is why the first
    # plates came out washed grey on grey. A drafted figure wants the values
    # it was given, not a cinematic response curve.
    try:
        sc.view_settings.view_transform = "Standard"
        sc.view_settings.look = "None"
    except TypeError:
        pass

    sc.render.resolution_x, sc.render.resolution_y = 2000, 2800
    sc.render.film_transparent = False
    sc.render.filepath = str(out_path)
    bpy.ops.render.render(write_still=True)
    print(f"rendered {out_path}")


# ------------------------------------------------------------- analysis side
def build_spec():
    import geopandas as gpd
    import numpy as np
    import pandas as pd

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from common import PROC, RAW, banner
    banner("axonometric geometry for blender")

    UTM, GAP, BAR, TAIL = 32618, 620.0, 150.0, 0.80
    nodes = gpd.read_file(PROC / "nodes.gpkg").to_crs(UTM)
    sim = pd.read_csv(PROC / "sim_index.csv")
    met = pd.read_csv(PROC / "metrics.csv")[["node_id", "GVI"]]

    d = sim.merge(met, on="node_id", how="left").merge(
        nodes[["node_id", "geometry"]], on="node_id", how="left")
    if d.geometry.isna().any():
        sys.exit("unmatched node_id -- refusing to build a scene with guessed positions")
    d = gpd.GeoDataFrame(d, geometry="geometry", crs=UTM)
    x, y = d.geometry.x.to_numpy(), d.geometry.y.to_numpy()
    ox, oy = x.mean(), y.mean()
    print(f"{len(d)} nodes, all matched on node_id")

    fps = []
    p = RAW / "building_footprints.geojson"
    if p.exists():
        f = gpd.read_file(p).to_crs(UTM)
        h = pd.to_numeric(f.height_roof, errors="coerce").fillna(0.0)
        h = h.clip(0, h.quantile(0.995)) * 0.3048
        for geom, hh in zip(f.geometry, h):
            gs = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
            for g in gs:
                ring = [(px - ox, py - oy) for px, py in g.exterior.coords[:-1]]
                if len(ring) >= 3:
                    fps.append([ring, float(hh)])
        print(f"{len(fps)} footprint rings")

    bars = {"grey": [], "red": []}
    for i, col in enumerate(["GVI", "G", "M", "P", "SIM"]):
        v = pd.to_numeric(d[col], errors="coerce").to_numpy(float)
        lo, hi = np.nanmin(v), np.nanmax(v)
        hgt = (v - lo) / (hi - lo if hi > lo else 1.0) * BAR
        cut = np.nanquantile(v, TAIL)
        z0 = GAP * (i + 1) + 120.0
        for xi, yi, hh, vi in zip(x, y, hgt, v):
            bars["red" if vi >= cut else "grey"].append(
                [float(xi - ox), float(yi - oy), z0, float(hh)])
        print(f"  {col:<4} {lo:.2f}-{hi:.2f}")

    span = float(max(x.max() - x.min(), y.max() - y.min()))
    ztop = GAP * 5 + 120.0 + BAR
    return {"footprints": fps, "bars": bars, "bar_width": 3.0,
            "span": span, "zmid": ztop / 2.0, "zext": ztop}


if IN_BLENDER:
    argv = sys.argv[sys.argv.index("--") + 1:]
    render(argv[0], argv[1])
else:
    import shutil
    spec = build_spec()
    out_dir = Path(__file__).parent.parent / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "_axon_spec.json"
    tmp.write_text(json.dumps(spec))
    print(f"spec {tmp.stat().st_size/1e6:.1f} MB")

    blender = shutil.which("blender")
    if not blender:
        cands = sorted(Path(r"C:\Program Files\Blender Foundation").glob("*/blender.exe"),
                       reverse=True)
        blender = str(cands[0]) if cands else None
    if not blender:
        sys.exit("blender not found; winget install --id BlenderFoundation.Blender -e")
    out = out_dir / "figure_axonometric_sim_blender.png"
    print(f"rendering with {blender}")
    r = subprocess.run([blender, "--background", "--python", str(Path(__file__).resolve()),
                        "--", str(tmp), str(out)], capture_output=True, text=True)
    print(r.stdout[-1500:] if r.returncode == 0 else r.stderr[-2500:])
    tmp.unlink(missing_ok=True)
    print(f"\n{'wrote ' + str(out) if out.exists() else 'render failed'}")
