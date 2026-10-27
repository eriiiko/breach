"""Blender 4.5: the octopus skin-pattern variants (issue #113) -- hero renders + game albedo bakes.

blender -b --factory-startup -P build_patterns.py -- [--only zebra,ghost] [--no-renders] [--save-blend]

Opens the committed hero .blend READ-ONLY (it is never saved over), adds the
pattern coordinates of `pattern_math.skin_coords` to the hero mesh as point
attributes, and for every row of `pattern_math.PATTERNS`:

1. builds the row's hero materials (the "default" row IS build_octopus.skin_material /
   cup_material, unchanged);
2. bakes its albedo onto the SAME game mesh with build_octopus.bake_pass (the build's
   own bake), keeps the committed gloss alpha, and writes
   prototype/octopus/variants/octopus_albedo_gloss_<name>.png;
3. renders the rest pose from the top (the game camera's view) and three-quarter, in the
   build's own studio, plus the game mesh with the new texture from the top.

The default row is checked against the committed octopus_albedo_gloss.png (and its gloss
re-baked against the committed alpha) and the result goes in patterns_report.json.
The GLBs are made from the PNGs by package_patterns.py (no Blender).
"""
from pathlib import Path
import argparse
import json
import math
import sys
import time

import bpy
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_octopus as bo
import octopus_math as om
import pattern_math as pm

OUT = bo.OUT / "variants"
PREVIEW = bo.HOME / "previews" / "patterns"
TEX = bo.TEX
GAMMA_NOTE = "colours in pattern_math are sRGB 0..255, converted with pattern_math.linear"


# --------------------------------------------------------------------------
class Graph:
    """Tiny shader-node builder: every method returns an output socket."""

    def __init__(self, tree):
        self.t = tree

    def _set(self, sock, v):
        if isinstance(v, bpy.types.NodeSocket):
            self.t.links.new(v, sock)
        else:
            sock.default_value = v

    def attr(self, name, out="Vector"):
        n = self.t.nodes.new("ShaderNodeAttribute")
        n.attribute_type, n.attribute_name = "GEOMETRY", name
        return n.outputs[out]

    def m(self, op, a, b=0.0, clamp=False):
        n = self.t.nodes.new("ShaderNodeMath")
        n.operation, n.use_clamp = op, clamp
        self._set(n.inputs[0], a)
        self._set(n.inputs[1], b)
        return n.outputs[0]

    def v(self, op, a, b=None, scale=None):
        n = self.t.nodes.new("ShaderNodeVectorMath")
        n.operation = op
        self._set(n.inputs[0], a)
        if b is not None:
            self._set(n.inputs[1], b)
        if scale is not None:
            self._set(n.inputs[3], scale)
        return n.outputs["Value" if op in ("DOT_PRODUCT", "LENGTH", "DISTANCE") else "Vector"]

    def xyz(self, vec):
        n = self.t.nodes.new("ShaderNodeSeparateXYZ")
        self._set(n.inputs[0], vec)
        return n.outputs["X"], n.outputs["Y"], n.outputs["Z"]

    def rgb(self, col):
        n = self.t.nodes.new("ShaderNodeSeparateColor")
        self._set(n.inputs[0], col)
        return n.outputs["Red"], n.outputs["Green"], n.outputs["Blue"]

    def noise(self, vec, size_m, detail=2.0, rough=.55, out="Fac"):
        n = self.t.nodes.new("ShaderNodeTexNoise")
        n.noise_dimensions = "3D"
        self._set(n.inputs["Vector"], vec)
        n.inputs["Scale"].default_value = 1.0 / size_m
        n.inputs["Detail"].default_value = detail
        n.inputs["Roughness"].default_value = rough
        return n.outputs[out]

    def warp(self, vec, size_m, amount_m):
        """vec + amount * (noise colour - 0.5): organic, skin-bound distortion."""
        c = self.noise(vec, size_m, detail=2.0, out="Color")
        d = self.v("SUBTRACT", c, (.5, .5, .5))
        return self.v("ADD", vec, self.v("SCALE", d, scale=2 * amount_m))

    def vor(self, vec, cell_m, feature="F1", rand=1.0):
        n = self.t.nodes.new("ShaderNodeTexVoronoi")
        n.voronoi_dimensions, n.feature = "3D", feature
        self._set(n.inputs["Vector"], vec)
        n.inputs["Scale"].default_value = 1.0 / cell_m
        n.inputs["Randomness"].default_value = rand
        return n.outputs["Distance"], (n.outputs["Color"] if feature != "DISTANCE_TO_EDGE" else None)

    def smooth(self, x, a, b):
        """0 at a, 1 at b (smoothstep; a > b gives the falling edge)."""
        n = self.t.nodes.new("ShaderNodeMapRange")
        n.interpolation_type, n.clamp = "SMOOTHSTEP", True
        self._set(n.inputs["Value"], x)
        self._set(n.inputs["From Min"], a)
        self._set(n.inputs["From Max"], b)
        return n.outputs["Result"]

    def band(self, x, a0, a1, b0, b1):
        """1 between the rising edge a0->a1 and the falling edge b0->b1."""
        return self.m("MULTIPLY", self.smooth(x, a0, a1), self.smooth(x, b1, b0))

    def lerp(self, f, a, b):
        n = self.t.nodes.new("ShaderNodeMix")
        n.data_type, n.clamp_factor = "FLOAT", True
        self._set(n.inputs[0], f)
        self._set(n.inputs[2], a)
        self._set(n.inputs[3], b)
        return n.outputs[0]

    def mix(self, f, a, b):
        n = self.t.nodes.new("ShaderNodeMix")
        n.data_type, n.clamp_factor = "RGBA", True
        self._set(n.inputs[0], f)
        self._set(n.inputs[6], (*a, 1) if isinstance(a, tuple) else a)
        self._set(n.inputs[7], (*b, 1) if isinstance(b, tuple) else b)
        return n.outputs[2]


class Skin:
    """The coordinates a pattern function draws in (see pattern_math docstring)."""

    def __init__(self, g):
        self.g = g
        self.body = g.attr("pat_body")
        self.arm = g.attr("pat_arm")
        self.w = g.attr("pat_w", "Fac")
        self.s = g.attr("pat_s", "Fac")

    def field(self, fn):
        """fn(P, on_arm) -> float socket, evaluated on the body and the arm coordinate and
        morphed by pat_w, so a pattern runs continuously from the body into each arm."""
        return self.g.lerp(self.w, fn(self.body, False), fn(self.arm, True))

    def mottle(self, size_m=.09):
        return self.field(lambda P, a: self.g.noise(P, size_m, detail=4.0, rough=.6))


def shade(row, key, k):
    return tuple(c * k for c in pm.linear(row[key]))


def mottled(g, sk, row, key, lo=.84, size_m=.09):
    return g.mix(sk.mottle(size_m), shade(row, key, lo), shade(row, key, 1.06))


# --------------------------------------------------------------------------
# Pattern functions: (g, skin, row) -> the DORSAL colour socket. Countershading,
# the mouth and the BSDF are shared (pattern_skin_material).
def zebra(g, sk, row):
    duty = row["band_duty"]
    thr = .5 + .5 * math.sin(math.pi * (.5 - duty))       # sin > thr for `duty` of a period

    def bands(P, on_arm):
        if on_arm:
            x, _, _ = g.xyz(P)
            q = g.m("ADD", x, g.m("MULTIPLY", g.m("SUBTRACT", g.noise(P, .06, detail=1.0), .5), .022))
            period = row["band_m"]
        else:                                             # chevrons across the mantle axis
            axis = tuple((om.MANTLE_C - om.HEAD_C) / np.linalg.norm(om.MANTLE_C - om.HEAD_C))
            rel = g.v("SUBTRACT", P, tuple(om.HEAD_C))
            x, _, _ = g.xyz(rel)
            q = g.m("ADD", g.v("DOT_PRODUCT", rel, axis), g.m("MULTIPLY", g.m("ABSOLUTE", x), .45))
            q = g.m("ADD", q, g.m("MULTIPLY", g.m("SUBTRACT", g.noise(P, .11, detail=3.0), .5), .16))
            period = row["stripe_m"]
        wave = g.m("SINE", g.m("MULTIPLY", q, 2 * math.pi / period))
        return g.smooth(g.m("ADD", g.m("MULTIPLY", wave, .5), .5), thr - .07, thr + .07)

    dark = sk.field(bands)
    return g.mix(dark, mottled(g, sk, row, "base", lo=.9), mottled(g, sk, row, "dark", lo=.75))


def reticulated(g, sk, row):
    cell, nw = row["cell_m"], row["net_w"]
    cache = {}

    def net(P, on_arm):
        Pw = g.warp(P, cell * .9, cell * .16)
        e, _ = g.vor(Pw, cell, "DISTANCE_TO_EDGE")
        d, c = g.vor(Pw, cell, "F1")
        cache[on_arm] = g.rgb(c)[0]
        return g.smooth(e, nw, nw * .55)

    pale = sk.field(net)
    var = g.lerp(sk.w, cache[False], cache[True])
    patch = g.mix(var, mottled(g, sk, row, "dark", lo=.85), mottled(g, sk, row, "base", lo=.85))
    return g.mix(pale, patch, mottled(g, sk, row, "net", lo=.9, size_m=.05))


def rosette(g, sk, row):
    cell = row["cell_m"]
    parts = {}

    def rings(P, on_arm):
        Pw = g.warp(P, cell * .8, cell * .1)
        d, c = g.vor(Pw, cell, "F1", rand=.85)
        R = g.m("ADD", g.m("MULTIPLY", g.rgb(c)[0], .06), row.get("ring_r", .40))
        at = lambda k: g.m("MULTIPLY", R, k)
        annulus = g.band(d, at(.5), at(.62), at(.92), at(1.05))
        db, _ = g.vor(Pw, cell * .34, "F1")              # the ring breaks into 4-6 arcs
        ring = annulus if row.get("ring_whole", False) else g.m("MULTIPLY", annulus, g.smooth(db, .62, .5))
        centre = g.smooth(d, at(.62), at(.5))
        d2, _ = g.vor(P, cell * .4, "F1")                 # solid spots between the rosettes
        spot = g.m("MULTIPLY", g.smooth(d2, .24, .16), g.smooth(d, at(1.1), at(1.3)))
        if on_arm and row.get("arm_solid", True):         # down the limb: rosettes close into solid spots
            solid = g.smooth(d, at(.95), at(.8))
            t = g.smooth(sk.s, .32, .58)
            ring = g.lerp(t, ring, solid)
            centre = g.lerp(t, centre, 0.0)
        parts[on_arm] = centre
        return g.m("MAXIMUM", ring, spot)

    dark = sk.field(rings)
    centre = g.lerp(sk.w, parts[False], parts[True])
    col = g.mix(g.m("MULTIPLY", centre, .9), mottled(g, sk, row, "base", lo=.86), mottled(g, sk, row, "centre", lo=.9))
    return g.mix(dark, col, mottled(g, sk, row, "dark", lo=.8))


def red_whitespot(g, sk, row):
    cell = row["cell_m"]

    def spots(P, on_arm):
        if on_arm:                                        # oval spots, long axis along the arm
            P = g.v("MULTIPLY", P, (.75, 1., 1.))
        Pw = g.warp(P, cell * .7, cell * .06)
        d, c = g.vor(Pw, cell, "F1", rand=1.0)
        r, gg, _ = g.rgb(c)
        R = g.m("ADD", g.m("MULTIPLY", r, .14), .14)
        keep = g.smooth(gg, .07, .11)                      # some cells carry no spot: scattered
        return g.m("MULTIPLY", g.smooth(d, R, g.m("MULTIPLY", R, .78)), keep)

    spot = sk.field(spots)
    base = g.mix(g.smooth(sk.mottle(.08), .3, .7), shade(row, "deep", 1.0), shade(row, "base", 1.05))
    return g.mix(spot, base, pm.linear(row["spot"]))


def blue_ringed(g, sk, row):
    cell = row["cell_m"]
    layers = {}

    def rings(P, on_arm):
        Pw = g.warp(P, cell * .8, cell * .08)
        d, c = g.vor(Pw, cell, "F1", rand=.8)
        R = g.m("ADD", g.m("MULTIPLY", g.rgb(c)[0], .06), .29)
        at = lambda k: g.m("MULTIPLY", R, k)
        blotch = g.smooth(d, at(1.5), at(1.2))
        edge = g.band(d, at(.46), at(.54), at(1.02), at(1.1))
        blue = g.band(d, at(.6), at(.66), at(.9), at(.96))
        layers[on_arm] = (blotch, edge)
        return blue

    blue = sk.field(rings)
    blotch = g.lerp(sk.w, layers[False][0], layers[True][0])
    edge = g.lerp(sk.w, layers[False][1], layers[True][1])
    col = g.mix(g.m("MULTIPLY", blotch, .85), mottled(g, sk, row, "base", lo=.85), mottled(g, sk, row, "blotch"))
    col = g.mix(edge, col, pm.linear(row["edge"]))
    return g.mix(blue, col, pm.linear(row["blue"]))


def ghost(g, sk, row):
    def blush(P, on_arm):
        b = g.m("MULTIPLY", g.smooth(g.noise(P, .16, detail=1.5), .42, .66), .45)
        if on_arm:
            b = g.m("ADD", b, g.m("MULTIPLY", g.smooth(sk.s, .3, .95), .45))
        return b

    def specks(P, on_arm):
        d, c = g.vor(P, .028, "F1")
        return g.m("MULTIPLY", g.smooth(d, .2, .1), g.m("MULTIPLY", g.smooth(g.rgb(c)[0], .70, .74), .6))

    col = g.mix(sk.field(blush), pm.linear(row["base"]), pm.linear(row["blush"]))
    return g.mix(sk.field(specks), col, pm.linear(row["speck"]))


BUILDERS = dict(zebra=zebra, reticulated=reticulated, rosette=rosette, red_whitespot=red_whitespot,
                blue_ringed=blue_ringed, ghost=ghost)


def pattern_skin_material(name, row, baking=False):
    """A patterned skin: the row's dorsal colour, countershaded by `pale`, the build's mouth
    darkening, BSDF and pore bump (as build_octopus.skin_material)."""
    m = bpy.data.materials.new(f"Octopus_skin_{name}")
    m.use_nodes = True
    t = m.node_tree
    bsdf = t.nodes["Principled BSDF"]
    g = Graph(t)
    sk = Skin(g)
    dorsal = BUILDERS[row["builder"]](g, sk, row)
    pale, mouth, _ = g.rgb(g.attr("tint", "Color"))
    col = g.mix(g.m("MULTIPLY", pale, .85), dorsal, pm.linear(row["oral"]))
    col = g.mix(mouth, col, (.16, .06, .05))
    t.links.new(col, bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = .44
    bsdf.inputs["Specular IOR Level"].default_value = .45
    sss = row.get("subsurface", .06)
    bsdf.inputs["Subsurface Weight"].default_value = .06 if baking else sss
    S = om.OCTOPUS_SCALE                      # world-unit lengths follow the size row (#106)
    bsdf.inputs["Subsurface Radius"].default_value = (.02 * S, .008 * S, .005 * S) if sss <= .06 else (1., .45, .35)
    if sss > .06:
        bsdf.inputs["Subsurface Scale"].default_value = .03 * S
    # the build's pore bump, on the same 'Generated' coordinates (skin_material)
    tc = t.nodes.new("ShaderNodeTexCoord")
    mp = t.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = MAPPING or bo.skin_mapping_scale()
    t.links.new(tc.outputs["Generated"], mp.inputs["Vector"])
    pores = bo.node(t, "ShaderNodeTexVoronoi", Scale=55.0, Randomness=.9)
    t.links.new(mp.outputs["Vector"], pores.inputs["Vector"])
    bump = t.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = .08
    bump.inputs["Distance"].default_value = .004 * S
    t.links.new(pores.outputs["Distance"], bump.inputs["Height"])
    t.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    m["pattern"] = name
    return m


MAPPING = None      # the hero mesh's 'Generated' mapping scale (build_octopus.skin_mapping_scale), set in main


def materials(name, baking=False):
    row = pm.PATTERNS[name]
    if row["builder"] == "default":
        return bo.skin_material(MAPPING), bo.cup_material()
    return (pattern_skin_material(name, row, baking),
            bo.cup_material(pm.linear(row["rim"]), pm.linear(row["bowl"]), f"Octopus_suckers_{name}"))


# --------------------------------------------------------------------------
def add_coords(obj, data):
    """Write pattern_math.skin_coords as point attributes (after checking vertex order)."""
    mesh = obj.data
    co = np.zeros(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", co)
    rest = co.reshape(-1, 3) @ bo.B / om.OCTOPUS_SCALE
    err = float(np.abs(rest - data["vertices"]).max())
    if err > 1e-4:
        raise SystemExit(f"hero mesh vertex order differs from octopus_math.assemble ({err})")
    c = pm.skin_coords(data)
    for key in ("pat_body", "pat_arm"):
        a = mesh.attributes.new(key, "FLOAT_VECTOR", "POINT")
        a.data.foreach_set("vector", c[key].astype(np.float32).ravel())
    for key in ("pat_w", "pat_s"):
        a = mesh.attributes.new(key, "FLOAT", "POINT")
        a.data.foreach_set("value", c[key].astype(np.float32))
    return err


def assign(obj, skin, cups):
    obj.data.materials[0] = skin
    obj.data.materials[1] = cups


def pixels(img):
    return np.array(img.pixels[:], np.float32).reshape(img.size[1], img.size[0], 4)


def bake(hero_obj, game_obj, btype, label):
    mat = game_obj.data.materials[0]
    t = mat.node_tree
    img = bpy.data.images.new(f"bake_{label}", TEX, TEX, alpha=False)
    n = t.nodes.new("ShaderNodeTexImage")
    n.image = img
    bo.bake_pass(hero_obj, game_obj, t, n, btype)
    out = pixels(img)
    t.nodes.remove(n)
    bpy.data.images.remove(img)
    return out


def save_png(rgba, path):
    img = bpy.data.images.new(path.stem, TEX, TEX, alpha=True)
    img.pixels.foreach_set(rgba.astype(np.float32).ravel())
    img.filepath_raw = str(path)
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--no-renders", action="store_true")
    ap.add_argument("--save-blend", action="store_true", help="also write a per-variant hero .blend (gitignored)")
    args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    names = [n for n in pm.ORDER if not args.only or n in args.only.split(",")]
    OUT.mkdir(parents=True, exist_ok=True)
    PREVIEW.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=str(bo.OUT / "octopus_hero.blend"))
    bo.gpu()
    scene = bpy.context.scene
    hero_obj, hero_rig = bpy.data.objects["Octopus_hero_mesh"], bpy.data.objects["Octopus_hero_rig"]
    game_obj, game_rig = bpy.data.objects["Octopus_game_mesh"], bpy.data.objects["Octopus_game_rig"]
    cam, floor = bpy.data.objects["Cam"], bpy.data.objects["Studio_floor_not_exported"]
    for r in (hero_rig, game_rig):
        r.animation_data_clear()
        r.hide_set(False)          # a hidden rig is not evaluated: the game LOD would render in bind pose
        r.hide_render = True
    t0 = time.time()
    data = om.assemble("hero")
    global MAPPING
    MAPPING = bo.skin_mapping_scale(data["vertices"])
    order_err = add_coords(hero_obj, data)
    bo.log("pattern coords", round(time.time() - t0, 1), "s, vertex-order check", order_err)
    committed = bpy.data.images.load(str(bo.OUT / "octopus_albedo_gloss.png"))
    committed_px = pixels(committed)
    gloss = committed_px[..., 3].copy()
    game_tex = next(n for n in game_obj.data.materials[0].node_tree.nodes
                    if n.type == "TEX_IMAGE" and n.image and n.image.name.startswith("Octopus_albedo_opaque"))
    report_path = OUT / "patterns_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    report.update(coords=dict(r_ref_m=pm.R_REF, taper_gamma=pm.TAPER_GAMMA, blend_s=pm.BLEND_S,
                              vertex_order_max_err_m=order_err), colour_note=GAMMA_NOTE)
    S = om.OCTOPUS_SCALE
    poses = om.named_poses()
    variants = report.setdefault("variants", {})
    for name in names:
        tv = time.time()
        rec = dict(animal=pm.PATTERNS[name]["animal"])
        # ---- bake (bind pose, as the build) -----------------------------------
        bo.set_pose(hero_rig, om.BodyParams(), None, "hero")
        bo.set_pose(game_rig, om.BodyParams(), None, "game")
        scene.render.engine, scene.cycles.samples = "CYCLES", 16
        assign(hero_obj, *materials(name, baking=True))
        for o in (game_obj, hero_obj):
            o.hide_set(False)
            o.hide_render = False
        alb = bake(hero_obj, game_obj, "DIFFUSE", name)
        rgba = alb.copy()
        rgba[..., 3] = gloss
        png = OUT / f"octopus_albedo_gloss_{name}.png"
        save_png(rgba, png)
        if name == "default":
            rough = bake(hero_obj, game_obj, "ROUGHNESS", "rough")[..., 0]
            regloss = np.clip(1 - rough / .65, 0, 1)
            d_rgb = np.abs(np.round(alb[..., :3] * 255) - np.round(committed_px[..., :3] * 255))
            d_a = np.abs(np.round(regloss * 255) - np.round(gloss * 255))
            rec["default_vs_committed"] = dict(
                albedo_rgb_max_diff_255=float(d_rgb.max()), albedo_rgb_mean_diff_255=float(d_rgb.mean()),
                albedo_texels_differing=int((d_rgb.max(-1) > 0).sum()),
                gloss_rebaked_max_diff_255=float(d_a.max()), gloss_texels_differing=int((d_a > 0).sum()))
            bo.log("DEFAULT vs committed", rec["default_vs_committed"])
        rec["bake_s"] = round(time.time() - tv, 1)
        # ---- renders (rest pose, the build's studio and cameras) ---------------
        if not args.no_renders:
            assign(hero_obj, *materials(name))
            scene.cycles.samples = 48
            scene.render.image_settings.file_format = "JPEG"
            scene.render.image_settings.quality = 90
            game_obj.hide_render, hero_obj.hide_render = True, False
            bo.set_pose(hero_rig, *poses["rest"], "hero")
            c = (0, .15 * S, -.05 * S)
            bo.shot(cam, PREVIEW / f"{name}_top.jpg", bo.to_b((0, 6 * S, -.05 * S)), bo.to_b(c),
                    ortho=3.15 * S, res=(1400, 1400))
            bo.shot(cam, PREVIEW / f"{name}_34.jpg", bo.to_b((2.3 * S, 2.1 * S, 2.6 * S)),
                    bo.to_b((0, .12 * S, -.05 * S)), lens=50, res=(1600, 1100))
            # the game LOD with this albedo (opaque in Blender; gloss is the GLB's alpha)
            opaque = alb.copy()
            opaque[..., 3] = 1
            game_tex.image.pixels.foreach_set(opaque.ravel())
            game_tex.image.update()
            bo.set_pose(game_rig, *poses["rest"], "game")
            game_obj.hide_render, hero_obj.hide_render = False, True
            bo.shot(cam, PREVIEW / f"game_{name}_top.jpg", bo.to_b((0, 6 * S, -.05 * S)), bo.to_b(c),
                    ortho=3.15 * S, res=(900, 900))
            game_obj.hide_render, hero_obj.hide_render = True, False
        if args.save_blend:
            assign(hero_obj, *materials(name))
            bpy.ops.wm.save_as_mainfile(filepath=str(OUT / f"octopus_hero_{name}.blend"), copy=True)
        rec["seconds"] = round(time.time() - tv, 1)
        variants[name] = rec
        bo.log("variant", name, rec)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    bo.log("PATTERNS_OK", [n for n in names])


if __name__ == "__main__":
    main()
