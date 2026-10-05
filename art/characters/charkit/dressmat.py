"""Materials for dress uniforms (Blender 4.5, Cycles): wool broadcloth with gold lace drawn
from a `braid` attribute, fur, feathers, gold cord, polished leather, a barrel sash, gloves.

Every builder takes its colours as arguments (sRGB hex or linear tuples, `wearmat.lin`): a
character's palette table is the only place a colour lives. Roughness is part of the look in
the game (the game step bakes it into a gloss mask, `gameready.gloss_signal`): polished
leather and metal glossy, gold lace semi-gloss, cloth, fur and feathers matte.

Vertex attributes read (0 where a mesh lacks them):
  braid    distance to a lace centre line, in the kit's line encoding (`braid.line_value`)
  tip      0 -> 1 along a plume (`fur.plume`)
  barrel   distance to a sash barrel's centre line (the same encoding)
"""
import kit
import wearmat
from studio import rgb
from wearmat import lin, scale, _new, _value


def _lace(t, attr, width):
    """1 on a lace band of half-width `width` round the line stored in `attr`, 0 off it."""
    d = t.math("MULTIPLY", t.math("SUBTRACT", 1.0, t.attr(attr)), kit.SEAM_CAP)
    return t.math("SUBTRACT", 1.0, t.ramp(d, width - 0.0006, width + 0.0004))


def _pos_sum(t):
    sep = t.n("ShaderNodeSeparateXYZ")
    t.put(sep.inputs[0], t.pos())
    return t.math("ADD", t.math("ADD", sep.outputs["X"], sep.outputs["Y"]), sep.outputs["Z"]), sep


def mat_wool(name, base, lace=None, lace_w=0.0030, rough=0.82, lace_rough=0.42, sheen=0.04, lace_metal=0.6):
    """Wool broadcloth: a fine felted nap, a little sheen, AO darkening in the folds. Where the
    `braid` attribute marks a line, a band `lace_w` either side of it is gold lace (`lace`),
    metallic and semi-gloss, with a woven ridge across it and a little relief."""
    m, t = _new(name)
    b = lin(base)
    col = t.mix(t.ramp(t.noise(5.0, 3.0, 0.55), 0.3, 0.7), (b[0] * 0.92, b[1] * 0.92, b[2] * 0.92, 1.0), (b[0] * 1.08, b[1] * 1.08, b[2] * 1.08, 1.0))
    ao = t.ramp(t.ao(0.05), 0.25, 0.95)
    col = t.mix(t.math("MULTIPLY", t.math("SUBTRACT", 1.0, ao), 0.55), col, scale(base, 0.45))
    h = t.math("ADD", t.math("MULTIPLY", t.noise(160.0, 3.0, 0.6), 0.35), t.math("MULTIPLY", t.noise(22.0, 2.0, 0.55, 0.3, "RIDGED_MULTIFRACTAL"), 0.25))
    metal, r = 0.0, rough
    if lace is not None:
        band = _lace(t, "braid", lace_w)
        s, _ = _pos_sum(t)
        weave = t.math("SINE", t.math("MULTIPLY", s, kit.TAU / 0.0016))
        lc = lin(lace)
        gold = t.mix(t.ramp(weave, -0.6, 0.9), (lc[0] * 0.55, lc[1] * 0.52, lc[2] * 0.45, 1.0), lc)
        gold = t.mix(t.math("MULTIPLY", t.math("SUBTRACT", 1.0, ao), 0.5), gold, (lc[0] * 0.35, lc[1] * 0.32, lc[2] * 0.25, 1.0))
        col = t.mix(band, col, gold)
        h = t.math("ADD", t.math("MULTIPLY", t.math("ADD", band, t.math("MULTIPLY", weave, t.math("MULTIPLY", band, 0.25))), 1.6), h)
        metal = t.mixf(band, 0.0, lace_metal)
        r = t.mixf(band, rough, lace_rough)
    t.set(Base_Color=col, Roughness=r, Metallic=metal, Sheen_Weight=sheen, Sheen_Roughness=0.5, Sheen_Tint=col, Normal=t.bump(h, 0.35, 0.002))
    return m


def mat_sash(name, base, stripe, barrel, rough=0.7, barrel_rough=0.34, barrel_w=0.0045):
    """A barrel sash: rows of crimson cord (`base`, darker `stripe` between them, round the
    waist) gathered by gold barrels where the `barrel` attribute marks a line."""
    m, t = _new(name)
    sep = t.n("ShaderNodeSeparateXYZ")
    t.put(sep.inputs[0], t.pos())
    rows = t.math("SINE", t.math("MULTIPLY", sep.outputs["Z"], kit.TAU / 0.0115))
    s, _ = _pos_sum(t)
    twist = t.math("SINE", t.math("MULTIPLY", t.math("ADD", sep.outputs["Z"], t.math("MULTIPLY", sep.outputs["X"], 0.5)), kit.TAU / 0.0030))
    col = t.mix(t.ramp(rows, -0.75, -0.2), lin(stripe), lin(base))
    col = t.mix(t.math("MULTIPLY", t.ramp(twist, 0.2, 1.0), 0.25), col, scale(base, 1.25))
    band = _lace(t, "barrel", barrel_w)
    bc = lin(barrel)
    gold = t.mix(t.ramp(t.math("SINE", t.math("MULTIPLY", sep.outputs["Z"], kit.TAU / 0.004)), -0.5, 1.0), (bc[0] * 0.6, bc[1] * 0.55, bc[2] * 0.45, 1.0), bc)
    col = t.mix(band, col, gold)
    ao = t.ramp(t.ao(0.03), 0.25, 0.95)
    col = t.mix(t.math("MULTIPLY", t.math("SUBTRACT", 1.0, ao), 0.5), col, rgb(0.02, 0.01, 0.01))
    h = t.math("ADD", t.math("MULTIPLY", rows, 0.5), t.math("MULTIPLY", twist, 0.15))
    h = t.math("ADD", h, t.math("MULTIPLY", band, 0.8))
    t.set(Base_Color=col, Roughness=t.mixf(band, rough, barrel_rough), Metallic=t.mixf(band, 0.0, 0.85), Normal=t.bump(h, 0.45, 0.002))
    return m


def mat_gold_cord(name, gold, rough=0.44, metal=0.6):
    """Gold cord and lace standing free (cords, tassels, a cross-belt, boot trim): metallic,
    semi-gloss, a woven ridge pattern."""
    m, t = _new(name)
    g = lin(gold)
    s, _ = _pos_sum(t)
    weave = t.math("SINE", t.math("MULTIPLY", s, kit.TAU / 0.0022))
    col = t.mix(t.ramp(weave, -0.5, 0.9), (g[0] * 0.55, g[1] * 0.52, g[2] * 0.45, 1.0), g)
    ao = t.ramp(t.ao(0.02), 0.2, 0.95)
    col = t.mix(t.math("MULTIPLY", t.math("SUBTRACT", 1.0, ao), 0.6), col, (g[0] * 0.3, g[1] * 0.27, g[2] * 0.2, 1.0))
    t.set(Base_Color=col, Metallic=metal, Roughness=t.mixf(t.noise(60.0, 2.0), rough - 0.06, rough + 0.08),
          Normal=t.bump(t.math("ADD", weave, t.math("MULTIPLY", t.noise(400.0, 2.0), 0.3)), 0.35, 0.0015))
    return m


def mat_polished(name, base, rough=0.16, coat=0.7):
    """Polished black leather (boots, a shako's peak): a clear coat over a dark base, a few
    creases and a soft broken highlight."""
    m, t = _new(name)
    b = lin(base)
    col = t.mix(t.noise(6.0, 3.0), (b[0] * 0.85, b[1] * 0.85, b[2] * 0.85, 1.0), (b[0] * 1.2, b[1] * 1.2, b[2] * 1.2, 1.0))
    ao = t.ramp(t.ao(0.04), 0.25, 0.95)
    col = t.mix(t.math("MULTIPLY", t.math("SUBTRACT", 1.0, ao), 0.5), col, scale(base, 0.4))
    crease = t.noise(18.0, 3.0, 0.55, 0.4, "RIDGED_MULTIFRACTAL")
    t.set(Base_Color=col, Roughness=t.mixf(t.noise(9.0, 2.0), rough, rough + 0.12), Coat_Weight=coat, Coat_Roughness=0.08,
          Normal=t.bump(t.math("ADD", t.math("MULTIPLY", crease, 0.4), t.math("MULTIPLY", t.noise(300.0, 2.0), 0.1)), 0.25, 0.002))
    return m


def mat_fur(name, base, tip, rough=0.92):
    """Brown fur: strands (noise stretched along z) lighter at their `tip` colour, dark in the
    pile's depths (AO), matte, a soft sheen."""
    m, t = _new(name)
    b, tp = lin(base), lin(tip)
    strands = t.noise(240.0, 3.0, 0.6, 0.5, "FBM", stretch=(1.0, 1.0, 0.35))
    clumps = t.noise(30.0, 3.0, 0.6, 0.8)
    col = t.mix(t.ramp(t.math("ADD", t.math("MULTIPLY", strands, 0.7), t.math("MULTIPLY", clumps, 0.5)), 0.45, 0.85), b, tp)
    ao = t.ramp(t.ao(0.025), 0.15, 0.95)
    col = t.mix(t.math("MULTIPLY", t.math("SUBTRACT", 1.0, ao), 0.8), col, (b[0] * 0.25, b[1] * 0.22, b[2] * 0.2, 1.0))
    h = t.math("ADD", t.math("MULTIPLY", strands, 1.0), t.math("MULTIPLY", clumps, 0.6))
    t.set(Base_Color=col, Roughness=rough, Sheen_Weight=0.5, Sheen_Roughness=0.35, Sheen_Tint=col, Normal=t.bump(h, 0.8, 0.003))
    return m


def mat_feather(name, base, tip=None, tip_from=0.75, rough=0.85):
    """Feathers: fine barbs (noise stretched along the plume's height), matte with a sheen;
    with `tip`, the plume turns that colour from `tip_from` of its length (the `tip`
    attribute) upwards."""
    m, t = _new(name)
    b = lin(base)
    barbs = t.noise(380.0, 2.0, 0.6, 0.6, "FBM", stretch=(1.0, 1.0, 0.25))
    col = t.mix(t.ramp(barbs, 0.35, 0.75), (b[0] * 0.6, b[1] * 0.6, b[2] * 0.6, 1.0), (min(1, b[0] * 1.4 + 0.01), min(1, b[1] * 1.4 + 0.01), min(1, b[2] * 1.4 + 0.01), 1.0))
    if tip is not None:
        tc = lin(tip)
        tcol = t.mix(t.ramp(barbs, 0.35, 0.75), (tc[0] * 0.6, tc[1] * 0.6, tc[2] * 0.6, 1.0), (min(1, tc[0] * 1.3), min(1, tc[1] * 1.3), min(1, tc[2] * 1.3), 1.0))
        frac = t.ramp(t.math("ADD", t.attr("tip"), t.math("MULTIPLY", t.noise(25.0, 2.0), 0.08)), tip_from - 0.02, tip_from + 0.06)
        col = t.mix(frac, col, tcol)
    ao = t.ramp(t.ao(0.03), 0.15, 0.95)
    col = t.mix(t.math("MULTIPLY", t.math("SUBTRACT", 1.0, ao), 0.7), col, rgb(0.003, 0.003, 0.003))
    t.set(Base_Color=col, Roughness=rough, Sheen_Weight=0.4, Sheen_Roughness=0.3, Sheen_Tint=col, Normal=t.bump(barbs, 0.7, 0.003))
    return m


def mat_glove(name, base, rough=0.62):
    """White kid gloves: smooth, a faint sheen, creases dark in the AO."""
    m, t = _new(name)
    b = lin(base)
    ao = t.ramp(t.ao(0.02), 0.2, 0.95)
    col = t.mix(t.math("MULTIPLY", t.math("SUBTRACT", 1.0, ao), 0.55), b, scale(base, 0.55))
    t.set(Base_Color=col, Roughness=t.mixf(t.noise(30.0, 2.0), rough - 0.05, rough + 0.08), Sheen_Weight=0.15,
          Normal=t.bump(t.noise(140.0, 2.0, 0.6), 0.15, 0.0015))
    return m


def mat_gold_metal(name, base, rough=0.26):
    """Struck gilt metal (buttons, plates, the chin scales): wearmat's metal, glossier."""
    return wearmat.mat_metal(name, base, rough=rough)


def mat_shako(name, base, rough=0.55):
    """A shako's felt-covered body: dark, a soft nap, a little sheen."""
    m, t = _new(name)
    b = lin(base)
    ao = t.ramp(t.ao(0.04), 0.25, 0.95)
    col = t.mix(t.math("MULTIPLY", t.math("SUBTRACT", 1.0, ao), 0.5), t.mix(t.noise(8.0, 2.0), scale(base, 0.85), scale(base, 1.15)), scale(base, 0.4))
    t.set(Base_Color=col, Roughness=rough, Sheen_Weight=0.2, Sheen_Tint=col, Normal=t.bump(t.noise(200.0, 2.0), 0.2, 0.0015))
    return m
