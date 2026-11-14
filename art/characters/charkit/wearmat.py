"""Materials for bare-skinned characters in workwear and bodysuits (Blender 4.5, Cycles).

Every builder takes its colours as arguments: a character's palette table is the only
place a colour lives. Colours are given as sRGB hex strings ("#4a5468") or linear
tuples; `lin()` converts.

Vertex attributes the materials read (all default to 0 where a mesh lacks them):
  seam     distance to a sewn seam (kit.seam_attr): a groove with a row of stitching each side
  stitch   distance to a stitch line with no groove (a patch's topstitching)
  dirt     where grime collects (0..1); scaled by the material's `dirt_strength` value node
  lips, stubble, brow, nostril   head masks (parts.face_field)
  iris     cosine to the gaze direction (parts.eyeball)
  cover    hair coverage (parts.hair)
"""
import bpy

import kit
from studio import NT, rgb


def lin(c):
    """'#rrggbb' (sRGB) or a linear (r, g, b[, a]) -> linear RGBA."""
    if isinstance(c, str):
        h = c.lstrip("#")
        v = [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
        v = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in v]
        return (v[0], v[1], v[2], 1.0)
    c = tuple(c)
    return c if len(c) == 4 else (c[0], c[1], c[2], 1.0)


def scale(c, k):
    c = lin(c)
    return (c[0] * k, c[1] * k, c[2] * k, 1.0)


def _new(name):
    m = bpy.data.materials.new(name)
    return m, NT(m)


def _value(t, name, v):
    """A named Value node: a parameter to change in the .blend without a rebuild."""
    node = t.n("ShaderNodeValue", name=name, label=name)
    node.outputs[0].default_value = v
    return node.outputs[0]


def _stitches(t, attr, d0=0.0030, half=0.00055, dash=0.0042):
    """Thread rows at distance d0 either side of the line stored in `attr`, dashed in 3-D."""
    d = t.math("MULTIPLY", t.math("SUBTRACT", 1.0, t.attr(attr)), kit.SEAM_CAP)
    row = t.math("SUBTRACT", 1.0, t.ramp(t.math("ABSOLUTE", t.math("SUBTRACT", d, d0)), half * 0.4, half))
    sep = t.n("ShaderNodeSeparateXYZ")
    t.put(sep.inputs[0], t.pos())
    s = t.math("ADD", t.math("ADD", sep.outputs["X"], sep.outputs["Y"]), sep.outputs["Z"])
    on = t.math("GREATER_THAN", t.math("SINE", t.math("MULTIPLY", s, kit.TAU / dash)), -0.35)
    return t.math("MULTIPLY", row, on)


def mat_coverall(name, base, stitch, grime, dirt=1.0, fade=0.10, rough=0.86):
    """Work cloth (twill). `base` the cloth colour, `stitch` the thread, `grime` the colour the
    dirt MULTIPLIES the cloth by at full strength. Dirt is its own layer: the `dirt`
    attribute (chest, knees, cuffs, seat ...) plus grime in the creases, all scaled by the
    `dirt_strength` value node (0 = clean), and multiplied over the clean cloth, never
    painted into it."""
    m, t = _new(name)
    strength = _value(t, "dirt_strength", dirt)
    base = lin(base)
    # clean cloth: a slow fade (lighter, greyer where it has been washed and rubbed)
    worn = (min(1.0, base[0] * 1.35 + 0.012), min(1.0, base[1] * 1.35 + 0.012), min(1.0, base[2] * 1.30 + 0.012), 1.0)
    col = t.mix(t.math("MULTIPLY", t.ramp(t.noise(3.2, 3.0, 0.55), 0.35, 0.75), fade), base, worn)
    seam_d = t.math("MULTIPLY", t.math("SUBTRACT", 1.0, t.attr("seam")), kit.SEAM_CAP)
    groove = t.math("SUBTRACT", 1.0, t.ramp(seam_d, 0.0004, 0.0012))
    col = t.mix(t.math("MULTIPLY", groove, 0.55), col, scale(base, 0.45))
    thread = t.math("MAXIMUM", _stitches(t, "seam"), _stitches(t, "stitch"))
    col = t.mix(t.math("MULTIPLY", thread, 0.85), col, lin(stitch))
    # the dirt layer
    ao = t.ramp(t.ao(0.06), 0.25, 0.95)
    blot = t.ramp(t.noise(6.0, 4.0, 0.62, 0.3), 0.38, 0.72)
    speck = t.ramp(t.noise(38.0, 3.0, 0.6), 0.55, 0.80)
    zone = t.math("MULTIPLY", t.math("MULTIPLY", t.attr("dirt"), 1.6), t.math("ADD", t.math("MULTIPLY", blot, 0.9), 0.35), clamp=True)
    crease = t.math("MULTIPLY", t.math("SUBTRACT", 1.0, ao), 0.55)
    amount = t.math("ADD", t.math("ADD", zone, crease), t.math("MULTIPLY", speck, 0.12), clamp=True)
    amount = t.math("MULTIPLY", amount, strength, clamp=True)
    tint = t.mix(amount, rgb(1.0, 1.0, 1.0), lin(grime))
    col = t.mix(1.0, col, tint, "MULTIPLY")
    # twill weave + crumple + stitch relief
    sep = t.n("ShaderNodeSeparateXYZ")
    t.put(sep.inputs[0], t.pos())
    twill = t.math("SINE", t.math("MULTIPLY", t.math("ADD", t.math("ADD", sep.outputs["X"], sep.outputs["Y"]), sep.outputs["Z"]),
                                  kit.TAU / 0.0011))
    h = t.math("ADD", t.math("MULTIPLY", twill, 0.05), t.math("MULTIPLY", t.noise(70.0, 3.0, 0.6), 0.35))
    h = t.math("ADD", h, t.math("MULTIPLY", t.noise(20.0, 2.0, 0.55, 0.5, "RIDGED_MULTIFRACTAL"), 0.45))
    h = t.math("SUBTRACT", h, t.math("MULTIPLY", groove, 0.9))
    h = t.math("ADD", h, t.math("MULTIPLY", thread, 0.35))
    # a little sheen, tinted by the cloth itself: a white sheen washes a dark cloth (and its dirt) out
    t.set(Base_Color=col, Roughness=t.mixf(amount, rough, rough - 0.14), Sheen_Weight=0.08, Sheen_Roughness=0.6, Sheen_Tint=col,
          Normal=t.bump(h, 0.45, 0.004))
    return m


def mat_skin(name, skin, hair, lip_tint=(1.0, 0.80, 0.78), sss=0.18):
    """Skin with lips, eyebrows, stubble and nostrils from the head's mask attributes;
    the brows and stubble take the HAIR colour, the lips are the skin times `lip_tint`."""
    m, t = _new(name)
    s = lin(skin)
    hc = lin(hair)
    col = t.mix(t.math("MULTIPLY", t.ramp(t.noise(9.0, 3.0, 0.6), 0.3, 0.8), 0.35), s, (s[0] * 1.06, s[1] * 0.90, s[2] * 0.86, 1.0))
    col = t.mix(t.math("MULTIPLY", t.ramp(t.noise(30.0, 2.0, 0.5), 0.55, 0.85), 0.25), col, (s[0] * 0.82, s[1] * 0.74, s[2] * 0.70, 1.0))
    lips = t.attr("lips")
    col = t.mix(lips, col, (s[0] * lip_tint[0], s[1] * lip_tint[1], s[2] * lip_tint[2], 1.0))
    dots = t.ramp(t.noise(1400.0, 1.0, 0.5), 0.42, 0.62)
    stub = t.math("MULTIPLY", t.attr("stubble"), t.math("ADD", t.math("MULTIPLY", dots, 0.5), 0.35), clamp=True)
    col = t.mix(t.math("MULTIPLY", stub, 0.75), col, (0.55 * hc[0] + 0.45 * s[0] * 0.6, 0.55 * hc[1] + 0.45 * s[1] * 0.6, 0.55 * hc[2] + 0.45 * s[2] * 0.6, 1.0))
    strands = t.ramp(t.noise(260.0, 2.0, 0.5, 0.0, "FBM", stretch=(1.0, 1.0, 0.25)), 0.35, 0.6)
    col = t.mix(t.math("MULTIPLY", t.attr("brow"), t.math("ADD", t.math("MULTIPLY", strands, 0.4), 0.55), clamp=True), col, hc)
    col = t.mix(t.math("MULTIPLY", t.attr("nostril"), 0.45), col, (s[0] * 0.18, s[1] * 0.12, s[2] * 0.11, 1.0))
    ao = t.ramp(t.ao(0.03), 0.2, 0.95)
    col = t.mix(t.math("MULTIPLY", t.math("SUBTRACT", 1.0, ao), 0.35), col, (s[0] * 0.45, s[1] * 0.33, s[2] * 0.30, 1.0))
    pores = t.noise(900.0, 2.0, 0.6)
    h = t.math("ADD", t.math("MULTIPLY", pores, 0.25), t.math("MULTIPLY", t.noise(120.0, 3.0, 0.6), 0.25))
    h = t.math("ADD", h, t.math("MULTIPLY", stub, t.math("MULTIPLY", dots, 0.4)))
    t.set(Base_Color=col, Roughness=t.mixf(lips, t.mixf(t.noise(20.0, 2.0), 0.48, 0.60), 0.38), Subsurface_Weight=sss,
          Subsurface_Radius=(1.0, 0.45, 0.28), Subsurface_Scale=0.004, Normal=t.bump(h, 0.22, 0.0015))
    m["game_gloss"] = 0.0  # matte in the game: its roughness would bake the armour's sheen (gameready.gloss_signal)
    return m


def mat_hair(name, hair, grey=0.0):
    """Short hair: strand streaks along the head (z), clumps darker in their roots."""
    m, t = _new(name)
    hc = lin(hair)
    light = (min(1.0, hc[0] * 1.6 + 0.01 + grey * 0.25), min(1.0, hc[1] * 1.6 + 0.01 + grey * 0.25), min(1.0, hc[2] * 1.6 + 0.01 + grey * 0.25), 1.0)
    strands = t.noise(420.0, 3.0, 0.6, 0.0, "FBM", stretch=(1.0, 1.0, 0.12))
    col = t.mix(t.ramp(strands, 0.45, 0.75), hc, light)
    ao = t.ramp(t.ao(0.02), 0.2, 0.95)
    col = t.mix(t.math("MULTIPLY", t.math("SUBTRACT", 1.0, ao), 0.7), col, (hc[0] * 0.35, hc[1] * 0.35, hc[2] * 0.35, 1.0))
    h = t.math("ADD", t.math("MULTIPLY", strands, 0.8), t.math("MULTIPLY", t.noise(60.0, 3.0, 0.6), 0.4))
    t.set(Base_Color=col, Roughness=0.6, Sheen_Weight=0.12, Sheen_Roughness=0.4, Sheen_Tint=col,
          Normal=t.bump(h, 0.6, 0.002))
    m["game_gloss_scale"] = 0.25  # a faint sheen at most in the game (gameready.gloss_signal)
    return m


def mat_eye(name, sclera, iris):
    m, t = _new(name)
    c = t.attr("iris")
    sc = lin(sclera)
    col = t.mix(t.ramp(c, 0.80, 0.30), sc, (sc[0] * 0.80, sc[1] * 0.62, sc[2] * 0.58, 1.0))
    ir = lin(iris)
    streak = t.noise(900.0, 2.0, 0.5, 0.0, "FBM")
    iris_col = t.mix(t.ramp(streak, 0.35, 0.7), (ir[0] * 0.6, ir[1] * 0.6, ir[2] * 0.6, 1.0), (min(1, ir[0] * 1.4), min(1, ir[1] * 1.4), min(1, ir[2] * 1.4), 1.0))
    # iris ~28 deg across its radius (an 11-12 mm iris on a 24 mm eye), a dark limbal ring, pupil ~3.5 mm
    col = t.mix(t.ramp(c, 0.872, 0.884), col, (ir[0] * 0.35, ir[1] * 0.35, ir[2] * 0.35, 1.0))
    col = t.mix(t.ramp(c, 0.884, 0.900), col, iris_col)
    col = t.mix(t.ramp(c, 0.981, 0.985), col, rgb(0.004, 0.004, 0.005))
    t.set(Base_Color=col, Roughness=0.08, Coat_Weight=1.0, Coat_Roughness=0.03)
    return m


def mat_leather(name, base, grime, dirt=1.0, rough=(0.62, 0.82)):
    """Worn work-boot leather: scuffed lighter on the edges and toes, grime in the creases
    (scaled by `dirt_strength`)."""
    m, t = _new(name)
    strength = _value(t, "dirt_strength", dirt)
    b = lin(base)
    bev = t.n("ShaderNodeBevel", samples=4)
    bev.inputs["Radius"].default_value = 0.004
    geo = t.n("ShaderNodeNewGeometry")
    dot = t.n("ShaderNodeVectorMath", operation="DOT_PRODUCT")
    t.put(dot.inputs[0], bev.outputs[0])
    t.put(dot.inputs[1], geo.outputs["Normal"])
    edge = t.ramp(t.math("SUBTRACT", 1.0, dot.outputs["Value"]), 0.01, 0.10)
    col = t.mix(t.noise(8.0, 3.0), (b[0] * 0.85, b[1] * 0.85, b[2] * 0.85, 1.0), (b[0] * 1.15, b[1] * 1.12, b[2] * 1.10, 1.0))
    scuff = t.math("MULTIPLY", edge, t.ramp(t.noise(40.0, 3.0), 0.35, 0.7))
    col = t.mix(t.math("MULTIPLY", scuff, 0.7), col, (min(1, b[0] * 1.9), min(1, b[1] * 1.85), min(1, b[2] * 1.8), 1.0))
    ao = t.ramp(t.ao(0.04), 0.25, 0.95)
    amount = t.math("MULTIPLY", t.math("ADD", t.math("MULTIPLY", t.math("SUBTRACT", 1.0, ao), 0.6),
                                       t.math("MULTIPLY", t.ramp(t.noise(12.0, 4.0, 0.6), 0.5, 0.8), 0.35)), strength, clamp=True)
    col = t.mix(1.0, col, t.mix(amount, rgb(1.0, 1.0, 1.0), lin(grime)), "MULTIPLY")
    seam = t.math("MAXIMUM", _stitches(t, "seam", 0.0025), _stitches(t, "stitch", 0.0025))
    col = t.mix(t.math("MULTIPLY", seam, 0.6), col, (b[0] * 0.5, b[1] * 0.5, b[2] * 0.5, 1.0))
    crease = t.noise(55.0, 3.0, 0.55, 0.4, "RIDGED_MULTIFRACTAL")
    h = t.math("ADD", t.math("MULTIPLY", crease, 0.6), t.math("MULTIPLY", t.noise(500.0, 2.0), 0.25))
    t.set(Base_Color=col, Roughness=t.mixf(scuff, rough[0], rough[1]), Coat_Weight=0.08, Coat_Roughness=0.5,
          Normal=t.bump(h, 0.5, 0.002, bev.outputs[0]))
    return m


def mat_rubber(name, base, rough=0.78):
    m, t = _new(name)
    b = lin(base)
    ao = t.ramp(t.ao(0.03), 0.25, 0.95)
    col = t.mix(t.math("MULTIPLY", t.math("SUBTRACT", 1.0, ao), 0.5), b, (b[0] * 0.4, b[1] * 0.4, b[2] * 0.4, 1.0))
    t.set(Base_Color=col, Roughness=rough, Normal=t.bump(t.noise(300.0, 2.0), 0.2, 0.002))
    return m


def mat_metal(name, base, rough=0.38):
    m, t = _new(name)
    b = lin(base)
    col = t.mix(t.ramp(t.noise(30.0, 3.0), 0.4, 0.8), b, (b[0] * 0.55, b[1] * 0.52, b[2] * 0.48, 1.0))
    t.set(Base_Color=col, Metallic=1.0, Roughness=t.mixf(t.noise(40.0, 2.0), rough - 0.1, rough + 0.15))
    return m


def mat_gloss(name, base, rough=0.16, coat=1.0, coat_rough=0.06, specular=0.5, piping=0.6, groove=0.35):
    """A glossy stretch garment (latex-look bodysuit, patent boots). The gloss is three named
    Value nodes (`roughness`, `coat`, `coat_roughness`) so it can be tuned in the .blend. The
    `seam` attribute (kit.seam_attr / kit.tape_attr) or the signed `pipe` one is a narrow raised piping line with a
    darker core, drawn in relief (bump) so the gloss picks it out; `stitch` the same, finer."""
    m, t = _new(name)
    b = lin(base)
    rough_v, coat_v, coat_r = _value(t, "roughness", rough), _value(t, "coat", coat), _value(t, "coat_roughness", coat_rough)
    col = t.mix(t.ramp(t.noise(4.0, 2.0, 0.5), 0.35, 0.75), (b[0] * 0.92, b[1] * 0.92, b[2] * 0.92, 1.0), (b[0] * 1.08, b[1] * 1.08, b[2] * 1.08, 1.0))
    d = t.math("MULTIPLY", t.math("SUBTRACT", 1.0, t.attr("seam")), kit.SEAM_CAP)
    # `pipe`: a SIGNED distance to a line (0.5 on it, see bodysuit.pipe_attr), which interpolates
    # exactly across a face, so a diagonal line stays unbroken on a coarse grid; absent = 0 = far
    dp = t.math("MULTIPLY", t.math("ABSOLUTE", t.math("SUBTRACT", t.attr("pipe"), 0.5)), 2.0 * kit.SEAM_CAP)
    d = t.math("MINIMUM", d, dp)
    ridge = t.math("SUBTRACT", 1.0, t.ramp(d, 0.0004, 0.0013))
    core = t.math("SUBTRACT", 1.0, t.ramp(d, 0.00008, 0.00030))
    ds = t.math("MULTIPLY", t.math("SUBTRACT", 1.0, t.attr("stitch")), kit.SEAM_CAP)
    fine = t.math("SUBTRACT", 1.0, t.ramp(ds, 0.0003, 0.0008))
    col = t.mix(t.math("MULTIPLY", core, groove), col, (b[0] * 0.35, b[1] * 0.35, b[2] * 0.35, 1.0))
    h = t.math("SUBTRACT", t.math("ADD", ridge, t.math("MULTIPLY", fine, 0.6)), t.math("MULTIPLY", core, 0.8))
    h = t.math("ADD", h, t.math("MULTIPLY", t.noise(160.0, 2.0, 0.5), 0.02))
    t.set(Base_Color=col, Roughness=rough_v, Specular_IOR_Level=specular, Coat_Weight=coat_v, Coat_Roughness=coat_r,
          Normal=t.bump(h, piping, 0.0012))
    return m


def mat_mesh(name, thread, under, cell=0.0032, width=0.30, show=0.55, rough=0.45, surface=False, coat=0.0):
    """An open-mesh insert (fishnet stretch panel): `thread` the net's colour, `under` what shows
    through it (the skin), dimmed to `show`. The net is a lattice of fine thread planes in world
    space, `cell` metres apart; `width` the thread's share of a cell. `surface`: a diamond net laid
    IN the surface instead, from the panel's developed coordinates (`mesh_u` / `mesh_v`, see
    garment.panel(uv=True)): world planes cut a curved panel in contour rings (a wood grain that
    shimmers at full-figure distance); a net in the surface's own coordinates cannot. `coat`: a
    clear coat over the net (the suit's gloss carried over the insert)."""
    m, t = _new(name)
    th, u = lin(thread), lin(under)
    k = kit.TAU / cell

    def lines(v):
        s = t.math("ABSOLUTE", t.math("SINE", t.math("MULTIPLY", v, k * 0.5)))
        return t.ramp(s, 1.0 - width * 1.6, 1.0 - width * 0.9)

    if surface:
        su, sv = t.attr("mesh_u"), t.attr("mesh_v")
        net = t.math("MAXIMUM", lines(t.math("ADD", su, sv)), lines(t.math("SUBTRACT", su, sv)))
    else:
        sep = t.n("ShaderNodeSeparateXYZ")
        t.put(sep.inputs[0], t.pos())
        x, y, z = sep.outputs["X"], sep.outputs["Y"], sep.outputs["Z"]
        # a cubic lattice of thread planes: any surface cuts at least two families, so the net never
        # collapses into parallel stripes whatever way the panel faces
        net = t.math("MAXIMUM", t.math("MAXIMUM", lines(x), lines(y)), lines(z))
    under_col = (u[0] * show, u[1] * show, u[2] * show, 1.0)
    col = t.mix(net, under_col, th)
    t.set(Base_Color=col, Roughness=t.mixf(net, 0.55, rough), Coat_Weight=coat, Coat_Roughness=0.25, Normal=t.bump(net, 0.35, 0.0006))
    return m


def mat_knit(name, base, rough=0.9):
    """A plain jersey knit (an undershirt): fine rib bump, no grime."""
    m, t = _new(name)
    b = lin(base)
    col = t.mix(t.noise(6.0, 2.0), (b[0] * 0.9, b[1] * 0.9, b[2] * 0.9, 1.0), (b[0] * 1.1, b[1] * 1.1, b[2] * 1.1, 1.0))
    sep = t.n("ShaderNodeSeparateXYZ")
    t.put(sep.inputs[0], t.pos())
    rib = t.math("SINE", t.math("MULTIPLY", sep.outputs["X"], kit.TAU / 0.0012))
    t.set(Base_Color=col, Roughness=rough, Sheen_Weight=0.08, Sheen_Tint=col, Normal=t.bump(t.math("ADD", t.math("MULTIPLY", rib, 0.2), t.noise(200.0, 2.0)), 0.25, 0.002))
    return m
