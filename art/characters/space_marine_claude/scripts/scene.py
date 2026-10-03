"""Materials, studio, cameras and renders for the space marine.

The turnaround sheet is re-shot with orthographic cameras that reproduce the
reference sheet's own framing (`refkit`), so a render and the artwork can be
compared silhouette for silhouette.
"""
import math
import os

import bpy
import numpy as np
from mathutils import Vector

import kit
import refkit

BACKDROP = (0.36, 0.375, 0.41)  # the sheet's grey-blue, scene-linear


# ---------------------------------------------------------------------- materials
class NT:
    """Tiny node-graph builder."""

    def __init__(self, mat):
        mat.use_nodes = True
        self.t = mat.node_tree
        self.bsdf = self.t.nodes["Principled BSDF"]

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

    def seam(self):
        """1 on a seam line, 0 elsewhere (see kit.seam_attr)."""
        d = self.math("MULTIPLY", self.math("SUBTRACT", 1.0, self.attr("seam")), kit.SEAM_CAP)
        return self.math("SUBTRACT", 1.0, self.ramp(d, 0.0006, 0.0018))


def rgb(r, g, b):
    return (r, g, b, 1.0)


IVORY_A, IVORY_B = rgb(0.52, 0.45, 0.34), rgb(0.62, 0.55, 0.43)
GRIME = rgb(0.24, 0.20, 0.15)
BLACK = rgb(0.012, 0.012, 0.014)


def _new(name):
    m = bpy.data.materials.new(name)
    return m, NT(m)


def mat_suit():
    """Padded ivory fabric; the `flex` mask turns it into black ribbed stretch panels."""
    m, t = _new("suit_fabric")
    flex = t.ramp(t.attr("flex"), 0.40, 0.60)
    seam = t.seam()
    ao = t.ramp(t.ao(0.07), 0.30, 0.95)
    ivory = t.mix(t.noise(2.6, 3.0), IVORY_A, IVORY_B)
    ivory = t.mix(t.math("MULTIPLY", t.ramp(t.noise(7.0, 4.0, 0.6), 0.45, 0.8), 0.32), ivory, GRIME)  # stains
    ivory = t.mix(t.math("MULTIPLY", t.math("SUBTRACT", 1.0, ao), 0.65), ivory, GRIME)  # dirt in the creases
    ivory = t.mix(t.math("MULTIPLY", seam, 0.7), ivory, rgb(0.20, 0.17, 0.13))
    col = t.mix(flex, ivory, BLACK)
    crumple = t.math("ADD", t.math("MULTIPLY", t.noise(24.0, 2.0, 0.55, 0.6, "RIDGED_MULTIFRACTAL"), 0.7),
                     t.math("MULTIPLY", t.noise(75.0, 3.0, 0.6), 0.3))
    weave = t.noise(650.0, 1.0)
    z = t.n("ShaderNodeSeparateXYZ")
    t.put(z.inputs[0], t.pos())
    ribs = t.math("SINE", t.math("MULTIPLY", z.outputs["Z"], kit.TAU / 0.0065))
    h = t.mixf(flex, t.math("ADD", t.math("MULTIPLY", crumple, 1.0), t.math("MULTIPLY", weave, 0.10)), t.math("MULTIPLY", ribs, 0.25))
    h = t.math("SUBTRACT", h, t.math("MULTIPLY", seam, 0.9))
    t.set(Base_Color=col, Roughness=t.mixf(flex, 0.84, 0.62), Sheen_Weight=t.mixf(flex, 0.25, 0.0), Sheen_Roughness=0.6,
          Normal=t.bump(h, 0.5, 0.005))
    return m


def mat_armor(name="armor_ivory", a=IVORY_A, b=IVORY_B, rough=(0.42, 0.66), lift=1.05):
    """Moulded hard shell: slightly lighter than the cloth, worn edges, grime in the recesses."""
    m, t = _new(name)
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
    col = t.mix(t.math("MULTIPLY", t.ramp(t.noise(11.0, 5.0, 0.65), 0.5, 0.85), 0.25), col, GRIME)
    col = t.mix(t.math("MULTIPLY", t.math("SUBTRACT", 1.0, ao), 0.6), col, GRIME)
    wear = t.math("MULTIPLY", edge, t.ramp(t.noise(60.0, 3.0), 0.35, 0.7))
    col = t.mix(t.math("MULTIPLY", wear, 0.45), col, rgb(0.30, 0.27, 0.23))
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


def mat_visor():
    m, t = _new("visor_glass")
    lw = t.n("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.35
    col = t.mix(lw.outputs["Facing"], rgb(0.003, 0.014, 0.006), rgb(0.010, 0.045, 0.018))
    t.set(Base_Color=col, Roughness=0.045, Coat_Weight=1.0, Coat_Roughness=0.02, IOR=1.6)
    return m


def make_materials():
    return dict(
        suit=mat_suit(),
        armor=mat_armor(),
        blue=mat_armor("panel_blue", rgb(0.065, 0.105, 0.17), rgb(0.085, 0.13, 0.21), (0.42, 0.6), 1.0),
        rubber=mat_plain("glove_rubber", BLACK, 0.55, bump_scale=420.0, bump=0.25, sheen=0.3),
        strap=mat_plain("strap_webbing", rgb(0.035, 0.035, 0.04), 0.7, bump_scale=900.0, bump=0.3),
        sole=mat_plain("boot_sole", rgb(0.03, 0.03, 0.032), 0.75, bump_scale=300.0, bump=0.2),
        dark=mat_plain("dark_fitting", rgb(0.02, 0.02, 0.022), 0.45),
        metal=mat_plain("fastener_metal", rgb(0.30, 0.29, 0.27), 0.4, metallic=1.0),
        visor=mat_visor(),
    )


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


def setup(samples=64):
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
    cam.inputs["Color"].default_value = (*BACKDROP, 1.0)
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
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)


# The three sheet views: (name, rig azimuth). The rig carries camera AND lights, so
# every view is lit like a front view, as on the reference sheet.
SHEET_VIEWS = (("front", 0.0), ("side", -90.0), ("back", 180.0))


def render_sheet(rig, cam, floor, out_dir, tag="sheet"):
    """Orthographic front/side/back in the reference sheet's framing; returns alpha masks."""
    floor.hide_render = True
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = refkit.SHEET_H * refkit.M_PER_PX
    cam.data.clip_end = 40.0
    cam.location = (0.0, -10.0, (refkit.FOOT_ROW - refkit.SHEET_H / 2) * refkit.M_PER_PX)
    cam.rotation_euler = (math.radians(90.0), 0.0, 0.0)
    panels = {}
    for (name, az), (_, _, w) in zip(SHEET_VIEWS, refkit.PANELS):
        rig.rotation_euler = (0.0, 0.0, math.radians(az))
        path = os.path.join(out_dir, "_%s_%s.png" % (tag, name))
        render(path, (w, refkit.SHEET_H), transparent=True)
        panels[name] = refkit.load_rgba(path)
        os.remove(path)
    rig.rotation_euler = (0.0, 0.0, 0.0)
    floor.hide_render = False
    return panels


def to_srgb(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(np.maximum(c, 1e-9), 1 / 2.4) - 0.055)


def compare_sheet(panels, out_dir, tag="sheet", verbose=True):
    """Write the model's sheet + a silhouette overlay against the artwork; print mismatches."""
    ref = refkit.load_rgba(refkit.REF_PATH)
    rmask, _ = refkit.ref_mask(ref)
    bg = to_srgb(BACKDROP)
    widths = [w for _, _, w in refkit.PANELS]
    W = sum(widths)
    sheet = np.ones((refkit.SHEET_H, W, 4), np.float32)
    over = np.ones_like(sheet)
    x0 = 0
    z_levels = [round(z, 3) for z in np.arange(1.86, 0.05, -0.04)]
    for name, cx, w in refkit.PANELS:
        img = panels[name]
        a = img[..., 3:4]
        sheet[:, x0:x0 + w, :3] = img[..., :3] * a + bg * (1 - a)
        mine = img[..., 3] > 0.5
        sl = refkit.panel_slice(name)[1]
        rm = rmask[:, sl]
        o = np.zeros((refkit.SHEET_H, w, 3), np.float32) + 0.93
        o[rm & mine] = (0.62, 0.62, 0.62)
        o[rm & ~mine] = (0.90, 0.25, 0.20)  # artwork only
        o[~rm & mine] = (0.20, 0.40, 0.90)  # model only
        over[:, x0:x0 + w, :3] = o
        iou = (rm & mine).sum() / max(1, (rm | mine).sum())
        print("\n== %s  IoU %.3f  (red = artwork only, blue = model only) ==" % (name, iou))
        if verbose:
            full = np.zeros_like(rmask)
            full[:, sl] = mine
            ra, rb = refkit.runs_table(rmask, name, z_levels), refkit.runs_table(full, name, z_levels)
            for z in z_levels:
                r, m = ra.get(z, []), rb.get(z, [])
                bad = len(r) != len(m) or any(abs(p[0] - q[0]) > 0.012 or abs(p[1] - q[1]) > 0.012 for p, q in zip(r, m))
                if bad:
                    print("z=%.2f ref %s\n       got %s" % (z, refkit.fmt_runs(r), refkit.fmt_runs(m)))
        x0 += w
    refkit.save_rgba(sheet, os.path.join(out_dir, tag + ".png"))
    refkit.save_rgba(over, os.path.join(out_dir, tag + "_overlay.png"))


# Beauty views: name -> (rig azimuth, camera azimuth within the rig, elevation, distance,
# target height, lens mm, resolution). The rig turns the lights with the camera.
BEAUTY = {
    "hero": (0.0, 32.0, 8.0, 4.7, 0.95, 85.0, (1200, 1600)),
    "hero_back": (180.0, 32.0, 8.0, 4.7, 0.95, 85.0, (1200, 1600)),
    "closeup": (0.0, 24.0, 4.0, 2.7, 1.50, 85.0, (1400, 1400)),
    "closeup_back": (180.0, 25.0, 10.0, 2.7, 1.48, 85.0, (1400, 1400)),
    "elevated": (0.0, 35.0, 52.0, 5.2, 0.95, 85.0, (1200, 1400)),
    "legs": (0.0, 28.0, 6.0, 2.9, 0.48, 85.0, (1400, 1400)),
    "top": (0.0, 0.0, 89.0, 4.6, 0.9, 85.0, (1200, 1200)),
}


def render_beauty(rig, cam, name, out_dir, scale=1.0):
    rig_az, az, el, dist, tz, lens, res = BEAUTY[name]
    cam.data.type = "PERSP"
    cam.data.lens = lens
    cam.data.clip_end = 60.0
    rig.rotation_euler = (0.0, 0.0, math.radians(rig_az))
    a, e = math.radians(az), math.radians(el)
    cam.location = (dist * math.sin(a) * math.cos(e), -dist * math.cos(a) * math.cos(e), tz + dist * math.sin(e))
    _look_at(cam, (0.0, 0.0, tz))
    render(os.path.join(out_dir, name + ".png"), (int(res[0] * scale), int(res[1] * scale)))
    rig.rotation_euler = (0.0, 0.0, 0.0)
