"""An anime / toon look for a character's high-resolution renders (Blender 4.5, EEVEE) -- Beatrice B2b.

Render-time only: `apply(look, rig, cam, ...)` converts the OPENED scene (a saved build, posed or not) and is never saved
with it, so the build, the game-ready step and the physically-lit look (`--look pbr`, Cycles) are untouched. A later
character uses it with its own `look` table; nothing here holds a colour or a measure of hers.

Why EEVEE with Shader-to-RGB (B2b's choice): the look needs the LIGHT as a number the material can ramp -- a two- or
three-tone terminator, a highlight that is either there or not -- and real cast shadows (the fringe across the forehead,
the chin over the neck). EEVEE's Shader-to-RGB gives a material its own diffuse and glossy lighting, shadows included, as a
colour; Cycles cannot (a ramp on a hand-made N.L has no shadows). Lines are an INVERTED HULL per object (a Solidify shell,
flipped, back-face culled, its own darker colour per material, its width set per shot in pixels): Freestyle and Line Art
are minutes per frame on a 3 M-triangle figure, the hull is free and draws the silhouette and every place one part
overlaps another (the chin on the neck, the hair on the face, an arm on the body), as the art inks them.

Each material of the scene named in `look["materials"]` (by its name less the prefix) is rebuilt round what fed its
Principled BSDF's Base Color and Normal (its painted masks, nets and piping kept):

  L      = luminance(Shader-to-RGB(Diffuse(normal))) / look["calib"]   the light, 1 = a white surface square to the key
  colour = base x ramp(L)                     `ramp`: ((position, multiplier rgb), ...), `interp` EASE / LINEAR / CONSTANT
         + streaks: map(luminance(Shader-to-RGB(Glossy(rough))), lo, hi) x colour     `spec` (the lights' specular only)
         + rim: map(facing, lo, hi) x colour, optionally only on the lit side          `rim`
         + `ring` (hair): a band of the attribute `toon_ring` (see `ring_attr`) x the light x colour
  emitted as is (no further lighting); `flat` blends the ramp towards 1 (the eyes: painted, barely lit)

The light rig is the look's own (`lights`, parented to the studio rig so a back view is lit like a front view; the studio's
lights are hidden); a light's `diffuse` / `specular` factors let narrow strips throw streaks on the gloss without lighting
the skin. The world: the backdrop for camera rays, a flat ambient for the lighting. The floor: a shadow catcher of its own
(the backdrop, darkened where the key is shadowed).
"""
import fnmatch
import math

import bpy
import numpy as np

import studio


def _lin(c):
    """'#rrggbb' (sRGB) or an (r, g, b) tuple (linear) -> linear RGBA."""
    if isinstance(c, str):
        h = c.lstrip("#")
        v = [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
        v = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in v]
        return (v[0], v[1], v[2], 1.0)
    return (c[0], c[1], c[2], 1.0)


# ------------------------------------------------------------------------------------------------ engine, world, lights
def engine(spec):
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_EEVEE_NEXT"
    e = sc.eevee
    e.taa_render_samples = spec.get("samples", 64)
    e.use_shadows = True
    e.shadow_ray_count = spec.get("shadow_rays", 3)
    e.shadow_step_count = spec.get("shadow_steps", 12)
    e.shadow_resolution_scale = 1.0
    e.use_raytracing = False
    e.use_gtao = False
    sc.render.film_transparent = False
    sc.render.filter_size = spec.get("filter", 1.2)
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"


def world(spec, backdrop):
    """The backdrop for camera rays, a flat ambient (`ambient` rgb, `strength`) for the lighting."""
    w = bpy.data.worlds.new("ToonWorld")
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    amb = nt.nodes.new("ShaderNodeBackground")
    amb.inputs["Color"].default_value = _lin(spec.get("ambient", (1.0, 1.0, 1.0)))
    amb.inputs["Strength"].default_value = spec.get("strength", 0.2)
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (*backdrop, 1.0)
    lp = nt.nodes.new("ShaderNodeLightPath")
    mix = nt.nodes.new("ShaderNodeMixShader")
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs[0])
    nt.links.new(amb.outputs[0], mix.inputs[1])
    nt.links.new(bg.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    bpy.context.scene.world = w


def lights(specs, rig):
    """Hide the studio's lights; add the look's: (name, kind, location, target, size (x, y) or angle, energy, colour,
    options: diffuse, specular, radius (soft shadow), shadow), parented to the rig."""
    for ob in bpy.data.objects:
        if ob.type == "LIGHT":
            ob.hide_render = True
    out = []
    for name, kind, loc, target, size, energy, col, opt in specs:
        li = bpy.data.lights.new(name, kind)
        li.energy, li.color = energy, col
        if kind == "AREA":
            li.shape = "RECTANGLE"
            li.size, li.size_y = size
        elif kind == "SUN":
            li.angle = math.radians(size)
        else:
            li.shadow_soft_size = size
        li.diffuse_factor = opt.get("diffuse", 1.0)
        li.specular_factor = opt.get("specular", 1.0)
        li.use_shadow = opt.get("shadow", True)
        li.use_shadow_jitter = opt.get("jitter", True)
        li.shadow_filter_radius = opt.get("filter", 1.0)
        if "res" in opt:   # B2c: the shadow map's finest texel (m); a crisp toon shadow from a lock 10 mm off the skin
            li.shadow_maximum_resolution = opt["res"]
        ob = bpy.data.objects.new(name, li)
        ob.location = loc
        studio._look_at(ob, target)
        ob.parent = rig
        bpy.context.scene.collection.objects.link(ob)
        out.append(ob)
    return out


# ------------------------------------------------------------------------------------------------ materials
class _G:
    """A few node helpers on a material's tree (no Principled needed)."""

    def __init__(self, m):
        self.nt = m.node_tree
        self.t = studio.NT.__new__(studio.NT)
        self.t.t = self.nt

    def n(self, typ, **kw):
        return self.t.n(typ, **kw)

    def put(self, sock, v):
        self.t.put(sock, v)

    def bw(self, colour):
        n = self.n("ShaderNodeRGBToBW")
        self.put(n.inputs[0], colour)
        return n.outputs[0]

    def lit(self, kind, normal, rough=0.5):
        """luminance(Shader-to-RGB(a white Diffuse / Glossy BSDF))."""
        b = self.n("ShaderNodeBsdfDiffuse" if kind == "diffuse" else "ShaderNodeBsdfGlossy")
        b.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        b.inputs["Roughness"].default_value = rough
        if normal is not None:
            self.put(b.inputs["Normal"], normal)
        s = self.n("ShaderNodeShaderToRGB")
        self.put(s.inputs[0], b.outputs[0])
        return self.bw(s.outputs[0])

    def colour_ramp(self, fac, stops, interp="EASE"):
        n = self.n("ShaderNodeValToRGB")
        cr = n.color_ramp
        cr.interpolation = interp
        while len(cr.elements) > 1:
            cr.elements.remove(cr.elements[-1])
        cr.elements[0].position, cr.elements[0].color = stops[0][0], _lin(stops[0][1])
        for pos, col in stops[1:]:
            e = cr.elements.new(pos)
            e.color = _lin(col)
        self.put(n.inputs[0], fac)
        return n.outputs[0]


def convert(m, spec, calib, palette=None):
    """Rebuild material `m` in the look (see the module docstring). Returns the base colour socket's source (for the hull)."""
    nt = m.node_tree
    bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
    out = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output), None)
    if bsdf is None or out is None:
        return None
    g = _G(m)
    t = g.t
    if not spec.get("ao", False):   # no ambient occlusion in a drawn look: every AO node reads 1 (unoccluded)
        for n in [n for n in nt.nodes if n.type == "AMBIENT_OCCLUSION"]:
            for o in n.outputs:
                for lk in list(o.links):
                    sock = lk.to_socket
                    nt.links.remove(lk)
                    if hasattr(sock, "default_value") and not hasattr(sock.default_value, "__len__"):
                        sock.default_value = 1.0
    src = bsdf.inputs["Base Color"]
    base = src.links[0].from_socket if src.is_linked else tuple(src.default_value)
    nin = bsdf.inputs["Normal"]
    normal = nin.links[0].from_socket if nin.is_linked else None
    if spec.get("bump", 1.0) != 1.0 and normal is not None and normal.node.type == "BUMP":
        normal.node.inputs["Strength"].default_value *= spec["bump"]
    L = t.math("DIVIDE", g.lit("diffuse", normal), calib)
    lobe = spec.get("lobe")
    if lobe:   # B2e: the light as PAINTED -- a lobe of the normal in camera space (see `normal_lobe`), mixed into L
        lb = t.ramp(normal_lobe(g, normal, lobe["dir"]), *lobe.get("map", (-0.2, 1.0, 0.0, 1.0)))
        if lobe.get("side"):   # only on one side of the head (`toon_x`, see `ring_attr`): (x where it starts, x where full)
            x0, x1 = lobe["side"]
            sgn = 1.0 if x1 > x0 else -1.0
            lb = t.math("MULTIPLY", lb, t.ramp(t.math("MULTIPLY", t.attr("toon_x"), sgn), sgn * x0, sgn * x1))
        if lobe.get("world"):   # and only where the surface faces that side in the WORLD (axis, from, to): seen from that
            # side (the profile) the hair facing the camera stays dark, as the profile close-up paints it
            ax, w0, w1 = lobe["world"]
            geo = g.n("ShaderNodeNewGeometry")
            dw = g.n("ShaderNodeVectorMath", operation="DOT_PRODUCT")
            g.put(dw.inputs[0], normal if normal is not None else geo.outputs["Normal"])
            dw.inputs[1].default_value = tuple(float(v) for v in ax)
            lb = t.math("MULTIPLY", lb, t.ramp(dw.outputs["Value"], w0, w1))
        if lobe.get("view"):   # and only while the camera looks at the head from that way (world axis, from, to): the
            # close-ups paint the lit side seen from the front, the side and back views the whole mass dark
            ax, v0, v1 = lobe["view"]
            vt = g.n("ShaderNodeVectorTransform", vector_type="VECTOR", convert_from="CAMERA", convert_to="WORLD")
            vt.inputs[0].default_value = (0.0, 0.0, -1.0)   # camera space looks down +z here: -z points back at the camera
            dv = g.n("ShaderNodeVectorMath", operation="DOT_PRODUCT")
            g.put(dv.inputs[0], vt.outputs[0])
            dv.inputs[1].default_value = tuple(float(v) for v in ax)
            lb = t.math("MULTIPLY", lb, t.ramp(dv.outputs["Value"], v0, v1))
        if lobe.get("per_lock"):   # and only as much as each lock takes (its `hlit`: the outer layer, not the fringe)
            lb = t.math("MULTIPLY", lb, t.attr("hlit"))
        L = t.mixf(lobe.get("weight", 1.0), L, lb)
    mult = g.colour_ramp(L, spec["ramp"], spec.get("interp", "EASE"))
    if spec.get("flat"):
        mult = t.mix(spec["flat"], mult, (1.0, 1.0, 1.0, 1.0))
    col = t.mix(1.0, base, mult, "MULTIPLY")
    sps = spec.get("spec") or ()
    sheen = spec.get("sheen")
    if sheen:   # B2d: a broad soft sheen under the streaks (see `sheen_lobe`), only where the key reaches
        sh = t.ramp(sheen_lobe(g, normal, sheen["dir"]), sheen["lo"], sheen["hi"])
        sh = t.math("MULTIPLY", sh, t.ramp(L, *sheen.get("lit", (0.15, 0.45))))
        if sheen.get("curv"):   # and more of it on the rounder forms
            sh = t.math("MULTIPLY", sh, t.ramp(t.attr("toon_curv"), *sheen["curv"][:2], out_lo=sheen["curv"][2]))
        col = t.mix(t.math("MULTIPLY", sh, sheen.get("amount", 1.0)), col, _lin(_col(sheen["colour"], palette)))
    for sp in ((sps,) if isinstance(sps, dict) else sps):   # one or more highlight layers (a broad sheen, then the streaks)
        s = g.lit("glossy", normal, sp["rough"])
        hl = t.ramp(s, sp["lo"], sp["hi"])
        if sp.get("curv"):   # B2d: the crisp white kept to the rounder forms (`curvature_attr`): no blobs on the flat upper chest
            hl = t.math("MULTIPLY", hl, t.ramp(t.attr("toon_curv"), *sp["curv"]))
        if sp.get("break"):  # B2d: long streaks broken along their length (the flank's "racing stripe")
            scale, lo_b, hi_b = sp["break"]
            hl = t.math("MULTIPLY", hl, t.ramp(t.noise(scale, 2.0, 0.5), lo_b, hi_b))
        hl = t.math("MULTIPLY", hl, sp.get("amount", 1.0))
        col = t.mix(hl, col, _lin(_col(sp["colour"], palette)), "ADD" if sp.get("add", False) else "MIX")
    rim = spec.get("rim")
    if rim:
        lw = g.n("ShaderNodeLayerWeight")
        lw.inputs["Blend"].default_value = rim.get("blend", 0.5)
        if normal is not None:
            g.put(lw.inputs["Normal"], normal)
        r = t.ramp(lw.outputs["Facing"], rim["lo"], rim["hi"])
        if rim.get("lit"):   # only where the light reaches (the shadow side keeps a dark edge)
            r = t.math("MULTIPLY", r, t.mixf(rim["lit"], 1.0, t.ramp(L, 0.25, 0.6)))
        r = t.math("MULTIPLY", r, rim.get("amount", 1.0))
        col = t.mix(r, col, _lin(_col(rim["colour"], palette)), "ADD" if rim.get("add", False) else "MIX")
    ring = spec.get("ring")
    if ring:
        band = t.attr("toon_ring")
        if ring.get("strands"):   # B2e: the band broken along the strands (the drawn gloss: streaks, not a block)
            sc_, lo_s, hi_s = ring["strands"]
            band = t.math("MULTIPLY", band, t.ramp(_strand_noise(g, sc_), lo_s, hi_s))
        lit_lo = ring.get("lit_floor", 0.0)   # B2e: some of the band on the shadow side (the drawing's cool sheen there)
        r = t.math("MULTIPLY", band, t.ramp(L, ring.get("lo", 0.3), ring.get("hi", 0.7), out_lo=lit_lo))
        r = t.math("MULTIPLY", r, ring.get("amount", 1.0))
        col = t.mix(r, col, _lin(_col(ring["colour"], palette)), "ADD" if ring.get("add", False) else "MIX")
    em = g.n("ShaderNodeEmission")
    g.put(em.inputs["Color"], col)
    em.inputs["Strength"].default_value = 1.0
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    m.use_backface_culling = spec.get("cull", False)
    return base


def _strand_noise(g, scale):
    """B2e: a noise stretched along a hair lock (its `hv` across and `hu` along, metres): `scale` = (features per metre
    across, per metre along) -- long thin streaks that follow each lock."""
    cx = g.n("ShaderNodeCombineXYZ")
    g.put(cx.inputs[0], g.t.math("MULTIPLY", g.t.attr("hv"), scale[0]))
    g.put(cx.inputs[1], g.t.math("MULTIPLY", g.t.attr("hu"), scale[1]))
    return g.t.noise(1.0, 3.0, 0.55, 0.0, "FBM", vec=cx.outputs[0])


def normal_lobe(g, normal, direction):
    """B2e: the cosine between the shading normal in CAMERA space and a fixed `direction` (x right, y up, z towards the
    viewer): the light as the art PAINTS it on anime hair -- the side of the mass turned towards the picture's light (the
    close-up's and the concept's: the viewer's left, a little above) goes ash to pale, the parts facing the viewer stay the
    dark brown -- whichever way the head is seen; a real light from her right would light the profile too, which the profile
    close-up paints dark."""
    geo = g.n("ShaderNodeNewGeometry")
    n = normal if normal is not None else geo.outputs["Normal"]
    vt = g.n("ShaderNodeVectorTransform", vector_type="NORMAL", convert_from="WORLD", convert_to="CAMERA")
    g.put(vt.inputs[0], n)
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    dot = g.n("ShaderNodeVectorMath", operation="DOT_PRODUCT")
    g.put(dot.inputs[0], vt.outputs[0])
    dot.inputs[1].default_value = (float(d[0]), float(d[1]), float(-d[2]))
    return dot.outputs["Value"]


def sheen_lobe(g, normal, direction):
    """B2d: a broad sheen as a reflection lobe in CAMERA space (a matcap): the cosine between the view reflected about the
    normal and a fixed `direction` (camera space: x right, y up, z towards the viewer). No light is needed, so the sheen keeps
    its size at any distance (a lit glossy on a sun or a strip is a speck or a flood) and stays with the view as the look's
    lights stay with the rig; the caller gates it by the key's light so shadowed sides keep no sheen."""
    geo = g.n("ShaderNodeNewGeometry")
    n = normal if normal is not None else geo.outputs["Normal"]
    inc = geo.outputs["Incoming"]
    neg = g.n("ShaderNodeVectorMath", operation="SCALE")
    g.put(neg.inputs[0], inc)
    neg.inputs["Scale"].default_value = -1.0
    ref = g.n("ShaderNodeVectorMath", operation="REFLECT")
    g.put(ref.inputs[0], neg.outputs[0])
    g.put(ref.inputs[1], n)
    vt = g.n("ShaderNodeVectorTransform", vector_type="VECTOR", convert_from="WORLD", convert_to="CAMERA")
    g.put(vt.inputs[0], ref.outputs[0])
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    # Blender's camera space looks down -z: the lobe's (x, y, z towards the viewer) is (x, y, -z) there
    dot = g.n("ShaderNodeVectorMath", operation="DOT_PRODUCT")
    g.put(dot.inputs[0], vt.outputs[0])
    dot.inputs[1].default_value = (float(d[0]), float(d[1]), float(-d[2]))
    return dot.outputs["Value"]


def curvature_attr(obs, spec):
    """B2d: a point attribute `toon_curv` (1/m) on each object: how round the surface is -- per vertex the largest normal
    curvature over its edges, (n_i - n_j).(p_i - p_j) / |p_i - p_j|^2, smoothed `smooth` times over the neighbours (a
    plate reads ~0, a breast or an arm 15-30). The streak layers read it (`spec["curv"]`) so the crisp white stays on the
    rounder forms, as the art paints latex, and a broad flat area (the upper chest) never floods white."""
    for ob in obs:
        me = ob.data
        n = len(me.vertices)
        if n == 0 or len(me.edges) == 0:
            continue
        P = np.empty(n * 3)
        me.vertices.foreach_get("co", P)
        P = P.reshape(-1, 3)
        N = np.empty(n * 3)
        me.vertices.foreach_get("normal", N)
        N = N.reshape(-1, 3)
        E = np.empty(len(me.edges) * 2, np.int64)
        me.edges.foreach_get("vertices", E)
        E = E.reshape(-1, 2)
        dp = P[E[:, 0]] - P[E[:, 1]]
        k = np.sum((N[E[:, 0]] - N[E[:, 1]]) * dp, axis=1) / np.maximum(np.sum(dp * dp, axis=1), 1e-12)
        kv = np.full(n, -1e9)
        np.maximum.at(kv, E[:, 0], k)
        np.maximum.at(kv, E[:, 1], k)
        kv = np.where(kv < -1e8, 0.0, kv) * spec.get("scale", 1.0)
        cnt = np.bincount(E.ravel(), minlength=n).astype(float) + 1.0
        for _ in range(spec.get("smooth", 4)):
            acc = kv.copy()
            np.add.at(acc, E[:, 0], kv[E[:, 1]])
            np.add.at(acc, E[:, 1], kv[E[:, 0]])
            kv = acc / cnt
        a = me.attributes.get("toon_curv") or me.attributes.new("toon_curv", "FLOAT", "POINT")
        a.data.foreach_set("value", kv.astype(np.float32))


def _col(c, palette):
    return palette[c] if (isinstance(c, str) and palette and c in palette) else c


def hull_material(name, colour):
    """The outline shell's material: a flat colour on the back faces only, invisible to shadows."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = colour
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    lp = nt.nodes.new("ShaderNodeLightPath")
    mix = nt.nodes.new("ShaderNodeMixShader")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(lp.outputs["Is Shadow Ray"], mix.inputs[0])
    nt.links.new(em.outputs[0], mix.inputs[1])
    nt.links.new(tr.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    m.use_backface_culling = True
    m.use_backface_culling_shadow = True
    m.use_transparent_shadow = True
    return m


def floor_material(spec, backdrop, calib):
    """The floor as a shadow catcher: the backdrop where lit, `colour` where the key's shadow falls (alpha = the shadow,
    so a transparent render keeps only the shadow)."""
    m = bpy.data.materials.new("toon_floor")
    m.use_nodes = True
    g = _G(m)
    t = g.t
    nt = m.node_tree
    for n in list(nt.nodes):
        if n.type != "OUTPUT_MATERIAL":
            nt.nodes.remove(n)
    out = nt.nodes[0]
    L = t.math("DIVIDE", g.lit("diffuse", None), calib)
    sh = t.math("MULTIPLY", t.math("SUBTRACT", 1.0, t.ramp(L, spec["lo"], spec["hi"])), spec.get("amount", 0.5))
    em = g.n("ShaderNodeEmission")
    em.inputs["Color"].default_value = _lin(spec.get("colour", (0.05, 0.05, 0.06)))
    tr = g.n("ShaderNodeBsdfTransparent")
    mix = g.n("ShaderNodeMixShader")
    g.put(mix.inputs[0], sh)
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(em.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    m.surface_render_method = "BLENDED"
    return m


# ------------------------------------------------------------------------------------------------ outlines
HULL = "ToonHull"


def outlines(spec, palette=None):
    """An inverted hull on every mesh not excluded (`exclude` name patterns): a Solidify shell outside the surface
    (offset +1, normals flipped, no rim), its faces given hull materials appended after the object's own (one per slot,
    `colours`[material name less prefix] or the local colour x `darken`). The width is set per shot (`set_width`)."""
    made = {}
    obs = []
    for ob in bpy.data.objects:
        if (ob.type != "MESH" or ob.name == "Floor" or not ob.visible_camera
                or any(fnmatch.fnmatch(ob.name, p) for p in spec.get("exclude", ()))):
            continue
        slots = [s.material for s in ob.material_slots]
        n = len(slots)
        if n == 0:
            continue
        for mat in slots:
            key = mat.name if mat else "_none"
            if key not in made:
                short = (key.split("_", 1)[1] if "_" in key else key).split(".")[0]
                c = spec.get("colours", {}).get(short)
                colour = _lin(_col(c, palette)) if c is not None else (0.02, 0.02, 0.02, 1.0)
                made[key] = hull_material("hull_" + key, colour)
            ob.data.materials.append(made[key])
        md = ob.modifiers.new(HULL, "SOLIDIFY")
        md.thickness = 0.001
        md.offset = 1.0
        md.use_flip_normals = True
        md.use_rim = False
        md.use_even_offset = False
        md.material_offset = n
        md.material_offset_rim = n
        if ob.vertex_groups.get("toon_line"):   # a per-vertex width factor (`line_group`)
            md.vertex_group = "toon_line"
            md.thickness_vertex_group = 0.0
        ob["toon_scale"] = spec.get("scale", {}).get(ob.name, 1.0)
        obs.append(ob)
    return obs


def metres_per_px(cam, target, res):
    """The size of a pixel at the target (orthographic: everywhere)."""
    d = cam.data
    big = max(res)
    if d.type == "ORTHO":
        return d.ortho_scale / big
    dist = (np.asarray(cam.matrix_world.translation) - np.asarray(target, float))
    dist = float(np.linalg.norm(dist))
    return dist * d.sensor_width / d.lens / big


def set_width(px, m_per_px, cam=None):
    """Every hull's width: `px` pixels at the subject (each object's `toon_scale` times); with `cam`, the view-dependent
    outline groups (`line_group`) are re-set for its direction."""
    if cam is not None:
        _view_lines(cam)
    for ob in bpy.data.objects:
        md = ob.modifiers.get(HULL) if ob.type == "MESH" else None
        if md is not None:
            md.thickness = px * m_per_px * ob.get("toon_scale", 1.0)


# ------------------------------------------------------------------------------------------------ attributes
def ring_attr(obs, centre, spec):
    """The hair's highlight band ("angel ring") as a point attribute `toon_ring` (0..1): a band of elevation about the
    head's `centre` (beta0 deg, half-width deg, softness deg), its edge broken per lock by a ripple along the azimuth
    (`jag` deg, `freq` per turn), only on the front and sides (`back` deg: faded out behind)."""
    c = np.asarray(centre, float)
    for ob in obs:
        me = ob.data
        P = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", P)
        P = P.reshape(-1, 3) @ np.asarray(ob.matrix_world)[:3, :3].T + np.asarray(ob.matrix_world)[:3, 3]
        d = P - c
        r = np.linalg.norm(d, axis=1)
        beta = np.degrees(np.arcsin(np.clip(d[:, 2] / np.maximum(r, 1e-9), -1.0, 1.0)))
        alpha = np.degrees(np.arctan2(np.abs(d[:, 0]), -d[:, 1]))
        b0, hw, soft = spec["beta"]
        jag = spec.get("jag", 0.0) * np.sin(np.radians(alpha) * spec.get("freq", 9.0)) \
            + 0.5 * spec.get("jag", 0.0) * np.sin(np.radians(alpha) * spec.get("freq", 9.0) * 2.3 + 1.1)
        lock = spec.get("per_lock")
        if lock and me.attributes.get("ring_off") is not None and me.attributes.get("hw") is not None:
            # B2c: a band PER LOCK, broken between locks as drawn: each lock's band shifted by its own `ring_off` (deg),
            # bowed across the lock (`sag` deg x across^2), its edges serrated along the strands (`saw` deg, `teeth` per
            # lock width) and narrowing to nothing at the lock's edges (`edge`: the across-position where it is gone)
            ro, hwv = np.empty(len(me.vertices)), np.empty(len(me.vertices))
            me.attributes["ring_off"].data.foreach_get("value", ro)
            me.attributes["hw"].data.foreach_get("value", hwv)
            jag = ro + lock.get("sag", 0.0) * hwv ** 2 + lock.get("saw", 0.0) * np.abs(np.sin(hwv * np.pi * lock.get("teeth", 2.0) + ro))
            b0, hw, soft = lock.get("beta", spec["beta"])
            # a lens per lock: the band's half-width shrinks to nothing towards the lock's edges
            lens = np.clip(1.0 - (hwv / lock.get("edge", 1.0)) ** 2, 0.0, 1.0)
            hw, soft = hw * lens, np.maximum(soft * lens, 0.2)
        q = np.abs(beta - b0 - jag)
        band = np.clip((hw + soft - q) / np.maximum(soft, 1e-6), 0.0, 1.0)
        band = band * band * (3 - 2 * band)
        bk = spec.get("back", 150.0)
        band *= np.clip((bk - alpha) / 25.0, 0.0, 1.0)
        a = me.attributes.get("toon_ring") or me.attributes.new("toon_ring", "FLOAT", "POINT")
        a.data.foreach_set("value", band.astype(np.float32))
        # B2e: the head-relative sideways position (m, + her left), written before any posing so it moves with the hair
        a = me.attributes.get("toon_x") or me.attributes.new("toon_x", "FLOAT", "POINT")
        a.data.foreach_set("value", d[:, 0].astype(np.float32))


# ------------------------------------------------------------------------------------------------ the whole look
def _verts(ob):
    V = np.empty(len(ob.data.vertices) * 3)
    ob.data.vertices.foreach_get("co", V)
    M = np.asarray(ob.matrix_world)
    return V.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]


def radial_normals(obs, centre, spec):
    """Custom normals blended from the mesh's own towards the direction from `centre` (the hair shaded as one smooth
    mass, not lock by lock, as anime hair is): `weight`, full above world z `z`[1], none below `z`[0]."""
    c = np.asarray(centre, float)
    z0, z1 = spec["z"]
    for ob in obs:
        me = ob.data
        V = _verts(ob)
        N = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("normal", N)
        N = N.reshape(-1, 3)
        R = V - c
        R /= np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-9)
        q = np.clip((V[:, 2] - z0) / (z1 - z0), 0.0, 1.0)
        q = q * q * (3 - 2 * q)
        w = spec["weight"] * q
        if spec.get("hang"):   # B2e: the hanging hair shaded as one mass too -- towards the horizontal from the head's
            # vertical axis below z0 (weight `hang`), so a curtain's sides read as the mass's sides, not ribbon by ribbon
            A = (V - c) * np.array([1.0, 1.0, 0.0])
            A /= np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)
            R = R * q[:, None] + A * (1.0 - q)[:, None]
            R /= np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-9)
            w = w + spec["hang"] * (1.0 - q)
        M = N * (1 - w)[:, None] + R * w[:, None]
        M /= np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-9)
        me.normals_split_custom_set_from_vertices([tuple(n) for n in M])


def shadow_proxy(name, V, Q, casters_off=()):
    """A mesh that only casts shadows (invisible to the camera), standing in for objects that no longer cast
    (`casters_off`, name patterns): the face casts as its simple shape, so no nose, lid or socket shadow falls on it."""
    import kit
    ob = kit.new_mesh(name, V, Q, None, None, "Toon")
    ob.visible_camera = False
    ob.visible_diffuse = ob.visible_glossy = ob.visible_transmission = ob.visible_volume_scatter = False
    for o in bpy.data.objects:
        if o.type == "MESH" and any(fnmatch.fnmatch(o.name, p) for p in casters_off):
            o.visible_shadow = False
    return ob


_LINE_MASKS = {}   # object name -> (mask, forward, front): the view-dependent part of its outline (`line_group`)


def _set_group(ob, w, name="toon_line"):
    vg = ob.vertex_groups.get(name) or ob.vertex_groups.new(name=name)
    w = np.round(np.asarray(w, float) * 20.0) / 20.0
    for val in np.unique(w):
        vg.add([int(i) for i in np.where(w == val)[0]], float(val), "REPLACE")


def line_group(ob, mask, forward=(0.0, -1.0, 0.0), front=(0.70, 0.92)):
    """The outline's width factor per vertex as a vertex group (`outlines` reads it): 1 - mask x how frontal the view is,
    re-set for every shot by `set_width` -- the cosine between the camera's view and `forward` (the face's) ramped over
    `front`. So a line the art leaves out from the front (the nose's sides) is thinned there and still drawn in profile."""
    _LINE_MASKS[ob.name] = (np.clip(np.asarray(mask, float), 0.0, 1.0), np.asarray(forward, float), front)
    _set_group(ob, 1.0 - _LINE_MASKS[ob.name][0])


def _view_lines(cam):
    """Re-set the view-dependent outline groups for the camera's current direction."""
    if not _LINE_MASKS:
        return
    M = np.asarray(cam.matrix_world)
    view = -M[:3, 2]          # the camera looks down its -Z
    for name, (mask, fwd, (c0, c1)) in _LINE_MASKS.items():
        ob = bpy.data.objects.get(name)
        if ob is None:
            continue
        c = float(-view @ fwd)    # 1 when the camera faces the face
        q = min(1.0, max(0.0, (c - c0) / (c1 - c0)))
        _set_group(ob, 1.0 - mask * q * q * (3 - 2 * q))


def prepare(look, normals=None, ring_centre=None, hair=None, proxies=(), lines=None):
    """The look's GEOMETRY-level data, set on the opened scene BEFORE any posing (it then deforms with the parts):
    `normals` = {object name: f(world verts) -> world normals} for custom normals (the face shaded as a simple head),
    `ring_centre` the head's centre for the hair's highlight band (`ring_attr`) and its smooth shading (`radial_normals`,
    `look["hair_normals"]`), `proxies` = [(name, verts, faces, casters_off)] shadow stand-ins (`shadow_proxy`)."""
    for ob_name, f in (normals or {}).items():
        ob = bpy.data.objects.get(ob_name)
        if ob is None:
            continue
        N = f(_verts(ob))
        ob.data.normals_split_custom_set_from_vertices([tuple(n) for n in N])
    if ring_centre is not None and look.get("ring"):
        R = look["ring"]
        hair_obs = [o for o in bpy.data.objects if any(fnmatch.fnmatch(o.name, p) for p in R["objects"])]
        if look.get("hair_normals"):
            radial_normals(hair_obs, ring_centre, look["hair_normals"])
        ring_attr(hair_obs, ring_centre, R)
    for name, V, Q, off in proxies:
        shadow_proxy(name, V, Q, off)
    cu = look.get("curvature")
    if cu:   # B2d: how round each surface is, for the latex's streaks (`curvature_attr`)
        def short(m):
            return (m.name.split("_", 1)[1] if "_" in m.name else m.name).split(".")[0] if m else ""
        curvature_attr([o for o in bpy.data.objects if o.type == "MESH"
                        and (any(fnmatch.fnmatch(o.name, p) for p in cu.get("objects", ()))
                             or any(short(s.material) in cu.get("materials", ()) for s in o.material_slots))], cu)
    for ob_name, f in (lines or {}).items():   # f(world verts) -> the mask of lines left out from the front
        ob = bpy.data.objects.get(ob_name)
        if ob is not None:
            line_group(ob, f(_verts(ob)), look.get("outline", {}).get("forward", (0.0, -1.0, 0.0)),
                       look.get("outline", {}).get("front", (0.70, 0.92)))


def apply(look, rig, cam, palette=None, backdrop=studio.BACKDROP):
    """Convert the opened scene to the look (render settings, world, lights, materials, the floor, the outlines; after
    `prepare` and any posing). Returns the hull objects."""
    engine(look.get("engine", {}))
    world(look.get("world", {}), backdrop)
    lights(look["lights"], rig)
    calib = look.get("calib", 1.0)
    mats = look["materials"]
    for m in list(bpy.data.materials):
        if not m.use_nodes or m.name.startswith(("hull_", "toon_")):
            continue
        short = (m.name.split("_", 1)[1] if "_" in m.name else m.name).split(".")[0]   # "beatrice_hair.001" -> "hair"
        spec = mats.get(short, mats.get("*"))
        if spec:
            convert(m, spec, calib, palette)
    fl = bpy.data.objects.get("Floor")
    if fl is not None:
        fl.is_shadow_catcher = False
        if look.get("floor"):
            fl.data.materials.clear()
            fl.data.materials.append(floor_material(look["floor"], backdrop, calib))
        else:
            fl.hide_render = True
    hulls = outlines(look["outline"], palette) if look.get("outline") else []
    return hulls


def width_for(cam, target, res, look):
    """Set the hull width for a shot (`look["outline"]["px"]` pixels at `target`)."""
    if look.get("outline"):
        set_width(look["outline"]["px"], metres_per_px(cam, target, res), cam)
