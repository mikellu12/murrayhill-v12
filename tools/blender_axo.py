"""
Exploded axonometric, rendered in 3-D by Blender.

Run through Blender's own interpreter, which is where bpy lives:

    /Applications/Blender.app/Contents/MacOS/Blender --background \
        --python tools/blender_axo.py -- \
        --json results/gis/murrayhill_layers.json \
        --out results/figures/figure_axo_3d.png

WHY A JSON HANDOFF
------------------
Blender ships its own Python with no geopandas or fiona, and installing
them into an app bundle to draw a picture is not a trade worth making.
tools/export_gis.py writes the same geometry as plain lists; this reads it.

WHY BARS
--------
The flat version encoded GVI as dot area, which is a poor channel -- area
is judged badly and the Park Avenue median swamped the ramp. Height is read
accurately, and a bar sitting above its own street corner keeps the value
attached to the place it was measured. Bars are normalised within their own
layer, so a bar's height is a drawing unit and never a metre.
"""
import bpy, bmesh, json, math, sys, argparse
from pathlib import Path
from mathutils import Vector


def argv():
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--json", default="results/gis/murrayhill_layers.json")
    p.add_argument("--out", default="results/figures/figure_axo_3d.png")
    p.add_argument("--res", type=int, default=2000)
    p.add_argument("--samples", type=int, default=48)
    p.add_argument("--z-scale", type=float, default=3.4)
    return p.parse_args(a)


def clear():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def mat(name, rgb, rough=0.62):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1)
    b.inputs["Roughness"].default_value = rough
    return m


def hex_rgb(h):
    h = h.lstrip("#")
    return tuple((int(h[i:i + 2], 16) / 255) ** 2.2 for i in (0, 2, 4))


def mesh_from(name, verts, faces, material, z=0.0):
    me = bpy.data.meshes.new(name)
    me.from_pydata([(x, y, zz + z) for x, y, zz in verts], [], faces)
    me.validate()
    me.update()
    ob = bpy.data.objects.new(name, me)
    ob.data.materials.append(material)
    bpy.context.collection.objects.link(ob)
    return ob


def prisms(rings_h, zbase):
    """Extruded footprints as one mesh. Ngon caps, quad sides."""
    V, F = [], []
    for ring, h in rings_h:
        r = ring[:-1] if len(ring) > 2 and ring[0] == ring[-1] else ring
        if len(r) < 3:
            continue
        n, o = len(r), len(V)
        V += [(x, y, 0.0) for x, y in r] + [(x, y, h) for x, y in r]
        F.append(list(range(o + n, o + 2 * n)))                    # roof
        F += [[o + i, o + (i + 1) % n, o + n + (i + 1) % n, o + n + i]
              for i in range(n)]                                   # walls
    return V, F


def bars(nodes, zbase, hmax, w, signed=False):
    """Value bars. A signed layer draws below its plane for negatives, so a
    contrast reads as a contrast and zero sits flat on the plane."""
    V, F = [], []
    floor = hmax * 0.012
    for nd in nodes:
        if signed:
            h = nd["v"] * hmax
            h = (floor if h >= 0 else -floor) if abs(h) < floor else h
        else:
            h = max(nd["v"] * hmax, floor)
        x, y = nd["x"], nd["y"]
        o = len(V)
        V += [(x - w, y - w, 0), (x + w, y - w, 0), (x + w, y + w, 0), (x - w, y + w, 0),
              (x - w, y - w, h), (x + w, y - w, h), (x + w, y + w, h), (x - w, y + w, h)]
        F += [[o, o + 1, o + 2, o + 3], [o + 4, o + 5, o + 6, o + 7],
              [o, o + 1, o + 5, o + 4], [o + 1, o + 2, o + 6, o + 5],
              [o + 2, o + 3, o + 7, o + 6], [o + 3, o, o + 4, o + 7]]
    return V, F


def ribbons(faces, zbase, w=3.0, t=1.2):
    """Block faces as thin slabs, so they read as objects in 3-D."""
    V, F = [], []
    for c in faces:
        for (x0, y0), (x1, y1) in zip(c[:-1], c[1:]):
            d = Vector((x1 - x0, y1 - y0)); L = d.length
            if L < 1e-6:
                continue
            nx, ny = -d.y / L * w, d.x / L * w
            o = len(V)
            quad = [(x0 + nx, y0 + ny), (x1 + nx, y1 + ny),
                    (x1 - nx, y1 - ny), (x0 - nx, y0 - ny)]
            V += [(x, y, 0) for x, y in quad] + [(x, y, t) for x, y in quad]
            F += [[o, o+1, o+2, o+3], [o+4, o+5, o+6, o+7],
                  [o, o+1, o+5, o+4], [o+1, o+2, o+6, o+5],
                  [o+2, o+3, o+7, o+6], [o+3, o, o+4, o+7]]
    return V, F


def main():
    a = argv()
    doc = json.loads(Path(a.json).read_text())
    zs = a.z_scale
    hmax, bw = doc["bar_height_m"], doc["bar_width_m"] / 2
    acc = hex_rgb(doc["accent"])

    clear()
    m_fab = mat("fabric", (0.80, 0.80, 0.79))
    m_low = mat("value", (0.42, 0.42, 0.43))
    m_acc = mat("accent", acc, rough=0.5)
    m_face = mat("face", (0.35, 0.35, 0.36))

    V, F = prisms([(f["ring"], f["h"]) for f in doc["fabric"]], 0)
    mesh_from("fabric", V, F, m_fab, z=0.0)

    V, F = ribbons(doc["faces"], 0)
    if F:
        mesh_from("faces", V, F, m_face, z=doc["faces_z"] * zs)

    # One plane per metric column, in the order the exporter emitted them.
    # Nothing here knows what the metrics are, so a new column drawn later
    # needs no edit to this file.
    lab = []
    for L in doc["layers"]:
        nd, dv = L["nodes"], L.get("diverging", False)
        # On a signed layer the accent marks the largest departures in
        # EITHER direction, so magnitude drives it, not sign.
        key = (lambda n: abs(n["v"])) if dv else (lambda n: n["v"])
        cut = sorted(key(n) for n in nd)[int(len(nd) * 0.9)] if nd else 1.1
        for tag, sel, mt in [("lo", [n for n in nd if key(n) < cut], m_low),
                             ("hi", [n for n in nd if key(n) >= cut], m_acc)]:
            if not sel:
                continue
            V, F = bars(sel, 0, hmax, bw, signed=dv)
            mesh_from(f"{L['key']}_{tag}", V, F, mt, z=L["z"] * zs)
        rng = (f"{L['lo']:+.2f}..{L['hi']:+.2f}{L['unit']}" if dv
               else f"{L['lo']:.2f}-{L['hi']:.2f}{L['unit']}")
        note = "" if L["coverage"] > 0.995 else f"   {L['coverage']:.0%} of nodes"
        lab.append((L["z"], L["title"], f"{L['key']}  {rng}{note}"))
    if doc["faces"]:
        lab.append((doc["faces_z"], "BLOCK FACES", f"{len(doc['faces'])} faces"))
    lab.append((0, "BUILT FABRIC", f"{len(doc['fabric'])} footprints"))

    # ---- screen-aligned labels -----------------------------------------
    # Text objects share the camera's rotation so they read flat on the page
    # rather than lying on the planes in perspective.
    m_txt = mat("txt", (0.06, 0.06, 0.07), rough=0.9)
    # Place labels in the camera's own basis instead of guessing world
    # offsets. RIGHT/UP are the camera's local axes expressed in world
    # space for this fixed isometric rotation; VIEW is what it looks along.
    # Because the camera is orthographic, sliding a label along VIEW moves
    # it in front of the model without shifting it on the page at all.
    RIGHT = Vector((0.70711, 0.70711, 0.0))
    UP = Vector((-0.40825, 0.40825, 0.81650))
    VIEW = Vector((-0.57735, 0.57735, -0.57735))
    for zz, title, sub in lab:
        for dy, body, size in [(0, title, 40), (-52, sub, 24)]:
            c = bpy.data.curves.new(type="FONT", name=title + body)
            c.body, c.size, c.align_x = body, size, "RIGHT"
            ob = bpy.data.objects.new(title + body, c)
            ob.data.materials.append(m_txt)
            ob.rotation_euler = (math.radians(54.736), 0.0, math.radians(45.0))
            ob.location = (Vector((0, 0, zz * zs))
                           + RIGHT * -660 + UP * dy - VIEW * 2200)
            bpy.context.collection.objects.link(ob)

    # ---- camera: true isometric, orthographic -------------------------
    sc = bpy.context.scene
    cam_d = bpy.data.cameras.new("cam"); cam_d.type = "ORTHO"
    ex = max(doc["extent_m"])
    cam_d.ortho_scale = ex * 2.40
    # The scene sits ~2.7 km from an orthographic camera placed well off the
    # model. Blender's default clip_end is 100 m, which silently clips the
    # entire city and renders nothing but world colour.
    cam_d.clip_start, cam_d.clip_end = 1.0, 20000.0
    cam = bpy.data.objects.new("cam", cam_d)
    sc.collection.objects.link(cam); sc.camera = cam
    cam.rotation_euler = (math.radians(54.736), 0.0, math.radians(45.0))
    top = max(L["z"] for L in doc["layers"]) * zs
    cam.location = Vector((1, -1, 1)).normalized() * ex * 2.4 + Vector((0, 0, top * 0.58))

    # ---- light and world ----------------------------------------------
    sun_d = bpy.data.lights.new("sun", "SUN"); sun_d.energy = 3.1
    sun_d.angle = math.radians(12)
    sun = bpy.data.objects.new("sun", sun_d)
    sc.collection.objects.link(sun)
    sun.rotation_euler = (math.radians(48), math.radians(14), math.radians(-125))
    w = bpy.data.worlds.new("w"); w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)
    w.node_tree.nodes["Background"].inputs[1].default_value = 1.05
    sc.world = w

    # ---- render --------------------------------------------------------
    # Cycles on CPU: EEVEE wants a GL context, which a --background run on
    # macOS does not reliably get.
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = a.samples
    sc.cycles.use_denoising = True
    sc.render.film_transparent = False
    sc.render.resolution_x = a.res
    sc.render.resolution_y = int(a.res * 1.3)
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    # AgX renders a pure-white world as ~0.8 grey. This is a line drawing on
    # white paper, not a photograph, so tone-map straight through.
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"

    # Freestyle gives the drawn-edge quality the reference has; without it
    # a white massing model on a white ground loses all its corners.
    sc.render.use_freestyle = True
    vl = sc.view_layers[0]
    vl.use_freestyle = True
    fs = vl.freestyle_settings
    ls = fs.linesets.new("edges")
    ls.select_silhouette = ls.select_border = ls.select_crease = True
    ls.linestyle.thickness = 0.75
    ls.linestyle.color = (0.12, 0.12, 0.13)

    for ob in bpy.data.objects:
        if ob.type == "MESH":
            ws = [ob.matrix_world @ Vector(c) for c in ob.bound_box]
            print("OBJ %-16s verts=%-7d x %8.0f..%-8.0f y %8.0f..%-8.0f z %8.0f..%-8.0f"
                  % (ob.name, len(ob.data.vertices),
                     min(v.x for v in ws), max(v.x for v in ws),
                     min(v.y for v in ws), max(v.y for v in ws),
                     min(v.z for v in ws), max(v.z for v in ws)))
    print("CAM loc %s ortho %.0f" % (tuple(round(v) for v in cam.location), cam_d.ortho_scale))

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    sc.render.filepath = str(out.resolve())
    bpy.ops.render.render(write_still=True)
    print(f"WROTE {out}")


main()
