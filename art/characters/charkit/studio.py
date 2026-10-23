"""Shared materials, studio and renders for the scripted characters (Blender 4.5, Cycles).

Every character is shot in the same studio: a rig that carries the camera AND the
lights (so a back view is lit like a front view), a flat backdrop, a shadow-catching
floor. A turnaround is orthographic at a fixed metres-per-pixel scale, so two
characters -- or a character and its reference sheet -- can be compared like for like.
"""
import json
import math
import os

import bpy
import numpy as np
from mathutils import Vector

import kit

BACKDROP = (0.36, 0.375, 0.41)  # grey-blue, scene-linear


# ------------------------------------------------------------------------ images
def load_rgba(path):
    """Image as float32 (h, w, 4), row 0 at the TOP, display-referred values."""
    img = bpy.data.images.load(path, check_existing=False)
    w, h = img.size
    buf = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(buf)
    bpy.data.images.remove(img)
    return buf.reshape(h, w, 4)[::-1].copy()


def save_rgba(arr, path):
    h, w = arr.shape[:2]
    img = bpy.data.images.new("_out", w, h, alpha=True)
    img.pixels.foreach_set(np.ascontiguousarray(arr[::-1]).astype(np.float32).ravel())
    img.filepath_raw = path
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)


def to_srgb(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(np.maximum(c, 1e-9), 1 / 2.4) - 0.055)


# ---------------------------------------------------------------------- materials
def rgb(r, g, b):
    return (r, g, b, 1.0)


GRIME = rgb(0.24, 0.20, 0.15)
BLACK = rgb(0.012, 0.012, 0.014)


class NT:
    """Tiny node-graph builder."""

    def __init__(self, mat, grime=GRIME):
        mat.use_nodes = True
        self.t = mat.node_tree
        self.bsdf = self.t.nodes["Principled BSDF"]
        self.grime = grime

    def n(self, typ, **kw):
        node = self.t.nodes.new(typ)
        for k, v in kw.items():
            setattr(node, k, v)
        return node

    def put(self, sock, v):
        if isinstance(v, bpy.types.NodeSocket):
            self.t.links.new(v, sock)
        elif v is not None:
            sock.default_value = v

    def set(self, **kw):
        for k, v in kw.items():
            self.put(self.bsdf.inputs[k.replace("_", " ")], v)

    def math(self, op, a, b=None, clamp=False):
        node = self.n("ShaderNodeMath", operation=op, use_clamp=clamp)
        self.put(node.inputs[0], a)
        self.put(node.inputs[1], b)
        return node.outputs[0]

    def mix(self, fac, a, b, blend="MIX"):
        node = self.n("ShaderNodeMix", data_type="RGBA", blend_type=blend)
        self.put(node.inputs[0], fac)
        self.put(node.inputs[6], a)
        self.put(node.inputs[7], b)
        return node.outputs[2]

    def mixf(self, fac, a, b):
        node = self.n("ShaderNodeMix", data_type="FLOAT")
        self.put(node.inputs[0], fac)
        self.put(node.inputs[2], a)
        self.put(node.inputs[3], b)
        return node.outputs[0]

    def ramp(self, x, lo, hi, out_lo=0.0, out_hi=1.0):
        node = self.n("ShaderNodeMapRange", interpolation_type="SMOOTHSTEP")
        self.put(node.inputs["Value"], x)
        for k, v in (("From Min", lo), ("From Max", hi), ("To Min", out_lo), ("To Max", out_hi)):
            node.inputs[k].default_value = v
        return node.outputs[0]

    def attr(self, name):
        return self.n("ShaderNodeAttribute", attribute_name=name).outputs["Fac"]

    def pos(self):
        return self.n("ShaderNodeNewGeometry").outputs["Position"]

    def noise(self, scale, detail=2.0, rough=0.5, distortion=0.0, kind="FBM", vec=None, stretch=None):
        node = self.n("ShaderNodeTexNoise", noise_type=kind)
        v = vec if vec is not None else self.pos()
        if stretch is not None:
            mp = self.n("ShaderNodeMapping")
            self.put(mp.inputs["Vector"], v)
            mp.inputs["Scale"].default_value = stretch
            v = mp.outputs[0]
        self.put(node.inputs["Vector"], v)
        for k, val in (("Scale", scale), ("Detail", detail), ("Roughness", rough), ("Distortion", distortion)):
            node.inputs[k].default_value = val
        return node.outputs[0]

    def ao(self, dist=0.06):
        node = self.n("ShaderNodeAmbientOcclusion", samples=6)
        node.inputs["Distance"].default_value = dist
        return node.outputs["AO"]

    def bump(self, height, strength=0.5, dist=0.003, normal=None):
        node = self.n("ShaderNodeBump")
        node.inputs["Strength"].default_value = strength
        node.inputs["Distance"].default_value = dist
        self.put(node.inputs["Height"], height)
        self.put(node.inputs["Normal"], normal)
        return node.outputs[0]

    def low(self, top=0.55):
        """1 at the floor fading to 0 at height `top`: where the dust collects."""
        sep = self.n("ShaderNodeSeparateXYZ")
        self.put(sep.inputs[0], self.pos())
        return self.math("SUBTRACT", 1.0, self.ramp(sep.outputs["Z"], 0.03, top))

    def dust(self, col, amount=0.45):
        f = self.math("MULTIPLY", self.math("MULTIPLY", self.low(), self.ramp(self.noise(5.0, 4.0, 0.6), 0.30, 0.75)), amount)
        return self.mix(f, col, self.grime)

    def line(self, name="seam", lo=0.0006, hi=0.0018):
        """1 on a line stored by kit.seam_attr / kit.tape_attr, 0 elsewhere; (lo, hi) is its half-width ramp in metres."""
        d = self.math("MULTIPLY", self.math("SUBTRACT", 1.0, self.attr(name)), kit.SEAM_CAP)
        return self.math("SUBTRACT", 1.0, self.ramp(d, lo, hi))

    def seam(self):
        return self.line("seam")


def _new(name, grime=GRIME):
    m = bpy.data.materials.new(name)
    return m, NT(m, grime)


def mat_cloth(name, a, b, flex_col=BLACK, grime=GRIME, seam_col=(0.20, 0.17, 0.13), crumple=1.0, bump=(0.5, 0.005),
              rough=(0.84, 0.62), sheen=0.25, stains=0.32, crease_dirt=0.65, dust=0.45, tape=None, tape_w=0.0045):
    """Fabric. The `flex` vertex mask turns it into ribbed stretch panels of `flex_col`,
    `seam` draws stitched lines, and `tape` (a colour) draws bonded seam tape along the `tape` attribute."""
    m, t = _new(name, grime)
    flex = t.ramp(t.attr("flex"), 0.40, 0.60)
    seam = t.seam()
    ao = t.ramp(t.ao(0.07), 0.30, 0.95)
    col = t.mix(t.noise(2.6, 3.0), a, b)
    col = t.mix(t.math("MULTIPLY", t.ramp(t.noise(7.0, 4.0, 0.6), 0.45, 0.8), stains), col, grime)  # stains
    col = t.mix(t.math("MULTIPLY", t.math("SUBTRACT", 1.0, ao), crease_dirt), col, grime)  # dirt in the creases
    col = t.dust(col, dust)
    col = t.mix(t.math("MULTIPLY", seam, 0.7), col, rgb(*seam_col))
    col = t.mix(flex, col, flex_col)
    cr = t.math("ADD", t.math("MULTIPLY", t.noise(24.0, 2.0, 0.55, 0.6, "RIDGED_MULTIFRACTAL"), 0.7),
                t.math("MULTIPLY", t.noise(75.0, 3.0, 0.6), 0.3))
    weave = t.noise(650.0, 1.0)
    z = t.n("ShaderNodeSeparateXYZ")
    t.put(z.inputs[0], t.pos())
    ribs = t.math("SINE", t.math("MULTIPLY", z.outputs["Z"], kit.TAU / 0.0065))
    h = t.mixf(flex, t.math("ADD", t.math("MULTIPLY", cr, crumple), t.math("MULTIPLY", weave, 0.10)), t.math("MULTIPLY", ribs, 0.25))
    h = t.math("SUBTRACT", h, t.math("MULTIPLY", seam, 0.9))
    if tape is not None:
        band = t.line("tape", tape_w - 0.0008, tape_w + 0.0008)
        col = t.mix(band, col, rgb(*tape))
        h = t.math("ADD", h, t.math("MULTIPLY", band, 0.5))
    t.set(Base_Color=col, Roughness=t.mixf(flex, *rough), Sheen_Weight=t.mixf(flex, sheen, 0.0), Sheen_Roughness=0.6,
          Normal=t.bump(h, *bump))
    return m


def mat_armor(name, a, b, rough=(0.42, 0.66), lift=1.05, grime=GRIME, wear_col=(0.30, 0.27, 0.23), wear=0.45, stains=0.36):
    """Moulded hard shell: worn edges, grime in the recesses, panel lines from the `seam` attribute."""
    m, t = _new(name, grime)
    bev = t.n("ShaderNodeBevel", samples=4)
    bev.inputs["Radius"].default_value = 0.0025
    geo = t.n("ShaderNodeNewGeometry")
    dot = t.n("ShaderNodeVectorMath", operation="DOT_PRODUCT")
    t.put(dot.inputs[0], bev.outputs[0])
    t.put(dot.inputs[1], geo.outputs["Normal"])
    edge = t.ramp(t.math("SUBTRACT", 1.0, dot.outputs["Value"]), 0.01, 0.12)
    ao = t.ramp(t.ao(0.05), 0.30, 0.95)
    col = t.mix(t.noise(3.0, 3.0), a, b)
    col = t.mix(1.0, col, rgb(lift, lift, lift), "MULTIPLY")
    col = t.mix(t.math("MULTIPLY", t.ramp(t.noise(11.0, 5.0, 0.65), 0.5, 0.85), stains), col, grime)
    col = t.dust(col)
    col = t.mix(t.math("MULTIPLY", t.math("SUBTRACT", 1.0, ao), 0.6), col, grime)
    w = t.math("MULTIPLY", edge, t.ramp(t.noise(60.0, 3.0), 0.35, 0.7))
    col = t.mix(t.math("MULTIPLY", w, wear), col, rgb(*wear_col))
    seam = t.seam()
    col = t.mix(t.math("MULTIPLY", seam, 0.75), col, rgb(0.12, 0.10, 0.08))
    scratch = t.noise(160.0, 2.0, 0.5, 0.0, "FBM", stretch=(1.0, 1.0, 0.08))
    h = t.math("SUBTRACT", t.math("MULTIPLY", scratch, 0.06), seam)
    t.set(Base_Color=col, Roughness=t.mixf(t.noise(14.0, 3.0), *rough), Normal=t.bump(h, 0.35, 0.002, bev.outputs[0]))
    return m


def mat_plain(name, color, rough=0.6, metallic=0.0, bump_scale=None, bump=0.2, sheen=0.0, coat=0.0):
    m, t = _new(name)
    ao = t.ramp(t.ao(0.04), 0.25, 0.95)
    col = t.mix(t.math("MULTIPLY", t.math("SUBTRACT", 1.0, ao), 0.5), color, rgb(*(0.4 * c for c in color[:3])))
    kw = dict(Base_Color=col, Roughness=t.mixf(t.noise(18.0, 3.0), rough - 0.08, rough + 0.08), Metallic=metallic,
              Sheen_Weight=sheen, Coat_Weight=coat)
    if bump_scale:
        kw["Normal"] = t.bump(t.noise(bump_scale, 2.0), bump, 0.002)
    t.set(**kw)
    return m


def mat_visor(name="visor_glass", facing=(0.003, 0.014, 0.006), edge=(0.010, 0.045, 0.018)):
    """Glossy visor glass. `facing` / `edge` (scene-linear RGB) are its colour seen head-on and at a
    grazing angle; the defaults are the original dark green glass (the space marine passes amber)."""
    m, t = _new(name)
    lw = t.n("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.35
    col = t.mix(lw.outputs["Facing"], rgb(*facing), rgb(*edge))
    t.set(Base_Color=col, Roughness=0.045, Coat_Weight=1.0, Coat_Roughness=0.02, IOR=1.6)
    return m


# -------------------------------------------------------------------------- studio
def _look_at(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def _area(name, loc, size, energy, rig, target=(0.0, 0.0, 1.0), color=(1.0, 1.0, 1.0)):
    li = bpy.data.lights.new(name, "AREA")
    li.shape, li.size, li.energy, li.color = "SQUARE", size, energy, color
    ob = bpy.data.objects.new(name, li)
    ob.location = loc
    _look_at(ob, target)
    ob.parent = rig
    bpy.context.scene.collection.objects.link(ob)
    return ob


def setup(samples=64, backdrop=BACKDROP):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    gpu = False
    for kind in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = kind
            (getattr(prefs, "refresh_devices", None) or prefs.get_devices)()
            devs = [d for d in prefs.devices if d.type == kind]
            for d in prefs.devices:
                d.use = d.type == kind
            if devs:
                gpu = True
                break
        except Exception:
            continue
    sc.cycles.device = "GPU" if gpu else "CPU"
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.cycles.use_adaptive_sampling = True
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    print("render device:", sc.cycles.device, prefs.compute_device_type)

    world = bpy.data.worlds.new("Studio")
    world.use_nodes = True
    sc.world = world
    nt = world.node_tree
    nt.nodes.clear()
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = -1.0, 1.0
    mr.inputs["To Min"].default_value, mr.inputs["To Max"].default_value = 0.05, 0.24
    amb = nt.nodes.new("ShaderNodeBackground")
    amb.inputs["Color"].default_value = (0.93, 0.96, 1.0, 1.0)
    cam = nt.nodes.new("ShaderNodeBackground")
    cam.inputs["Color"].default_value = (*backdrop, 1.0)
    lp = nt.nodes.new("ShaderNodeLightPath")
    mix = nt.nodes.new("ShaderNodeMixShader")
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(tc.outputs["Generated"], sep.inputs[0])
    nt.links.new(sep.outputs["Z"], mr.inputs["Value"])
    nt.links.new(mr.outputs[0], amb.inputs["Strength"])
    nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs[0])
    nt.links.new(amb.outputs[0], mix.inputs[1])
    nt.links.new(cam.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])

    rig = bpy.data.objects.new("StudioRig", None)
    sc.collection.objects.link(rig)
    _area("Key", (-2.4, -3.4, 3.6), 2.2, 330.0, rig)
    _area("Fill", (3.4, -3.0, 1.5), 3.5, 55.0, rig, color=(0.95, 0.97, 1.0))
    _area("Rim", (1.8, 3.6, 3.0), 2.5, 190.0, rig)
    _area("Top", (0.0, -0.6, 4.6), 3.0, 40.0, rig)

    floor = kit.new_mesh("Floor", [[-6, -6, 0], [6, -6, 0], [6, 6, 0], [-6, 6, 0]], [[0, 1, 2, 3]], coll="Studio", smooth=False)
    floor.is_shadow_catcher = True

    cd = bpy.data.cameras.new("Cam")
    cam_ob = bpy.data.objects.new("Cam", cd)
    cam_ob.parent = rig
    sc.collection.objects.link(cam_ob)
    sc.camera = cam_ob
    return rig, cam_ob, floor


def render(path, res, transparent=False):
    sc = bpy.context.scene
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = transparent
    jpg = path.lower().endswith(".jpg")
    sc.render.image_settings.file_format = "JPEG" if jpg else "PNG"
    sc.render.image_settings.color_mode = "RGB" if jpg else "RGBA"
    sc.render.image_settings.quality = 92
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)


def ortho_panels(rig, cam, floor, out_dir, views, m_per_px, height_px, centre_z, tag="sheet"):
    """Orthographic views at a fixed scale. `views` = [(name, rig azimuth deg, width px)];
    returns {name: RGBA array} with the figure's alpha (the floor is hidden)."""
    floor.hide_render = True
    cam.data.type = "ORTHO"
    cam.data.clip_end = 40.0
    cam.location = (0.0, -10.0, centre_z)
    cam.rotation_euler = (math.radians(90.0), 0.0, 0.0)
    panels = {}
    for name, az, w in views:
        cam.data.ortho_scale = max(w, height_px) * m_per_px
        rig.rotation_euler = (0.0, 0.0, math.radians(az))
        path = os.path.join(out_dir, "_%s_%s.png" % (tag, name))
        render(path, (w, height_px), transparent=True)
        panels[name] = load_rgba(path)
        os.remove(path)
    rig.rotation_euler = (0.0, 0.0, 0.0)
    floor.hide_render = False
    return panels


def compose(panels, order, backdrop=BACKDROP):
    """Panels side by side over the flat backdrop -> RGBA array."""
    bg = to_srgb(backdrop)
    h = panels[order[0]].shape[0]
    sheet = np.ones((h, sum(panels[n].shape[1] for n in order), 4), np.float32)
    x0 = 0
    for n in order:
        img = panels[n]
        a = img[..., 3:4]
        sheet[:, x0:x0 + img.shape[1], :3] = img[..., :3] * a + bg * (1 - a)
        x0 += img.shape[1]
    return sheet


# Turnaround used by characters that have no reference sheet of their own.
TURNAROUND = (("front", 0.0, 560), ("side", -90.0, 440), ("back", 180.0, 560), ("quarter", 35.0, 560))


def turnaround(rig, cam, floor, out_dir, height_m, tag="turnaround", views=TURNAROUND, height_px=1100, backdrop=BACKDROP):
    m_per_px = height_m * 1.12 / height_px
    panels = ortho_panels(rig, cam, floor, out_dir, views, m_per_px, height_px, height_m * 0.5, tag)
    save_rgba(compose(panels, [v[0] for v in views], backdrop), os.path.join(out_dir, tag + ".png"))
    return panels


def render_beauty(rig, cam, view, name, out_dir, scale=1.0):
    """view = (rig azimuth, camera azimuth within the rig, elevation, distance, target height, lens mm, resolution)."""
    rig_az, az, el, dist, tz, lens, res = view
    cam.data.type = "PERSP"
    cam.data.lens = lens
    cam.data.clip_end = 60.0
    rig.rotation_euler = (0.0, 0.0, math.radians(rig_az))
    a, e = math.radians(az), math.radians(el)
    cam.location = (dist * math.sin(a) * math.cos(e), -dist * math.cos(a) * math.cos(e), tz + dist * math.sin(e))
    _look_at(cam, (0.0, 0.0, tz))
    render(os.path.join(out_dir, name + ".jpg"), (int(res[0] * scale), int(res[1] * scale)))
    rig.rotation_euler = (0.0, 0.0, 0.0)


def mesh_stats(path=None):
    dg = bpy.context.evaluated_depsgraph_get()
    out, tot_src, tot_eval, count = {}, 0, 0, 0
    for ob in bpy.data.objects:
        if ob.type != "MESH" or ob.name == "Floor":
            continue
        ev = ob.evaluated_get(dg)
        me = ev.to_mesh()
        me.calc_loop_triangles()
        src = sum(len(p.vertices) - 2 for p in ob.data.polygons)
        c = out.setdefault(ob.users_collection[0].name, dict(objects=0, source_tris=0, evaluated_tris=0))
        c["objects"] += 1
        c["source_tris"] += src
        c["evaluated_tris"] += len(me.loop_triangles)
        tot_src += src
        tot_eval += len(me.loop_triangles)
        count += 1
        ev.to_mesh_clear()
    out["TOTAL"] = dict(objects=count, source_tris=tot_src, evaluated_tris=tot_eval)
    if path:
        with open(path, "w") as f:
            json.dump(out, f, indent=1)
    return out
