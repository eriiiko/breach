"""One tone-map (graphics arc G0 P1, docs/graphics_direction_2026-10-18.md §6).

Every world layer writes LINEAR HDR into the RGBA16F scene RT;
``WorldComposite.resolve()`` (shaders/world_resolve.fs) is the ONLY place
ACES, the gamma encode and the dither run. A layer that tone-maps itself is
the bug: summed over an already tone-mapped floor it re-curves and clips (the
#74 "fires read too white" mechanism). These tests scan the CODE (comments
stripped) of every world shader and the GLSL composed in renderer/ modules,
so they hold for shaders not yet written. Pyray-free; no GL context.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

SHADERS = ROOT / "shaders"
RESOLVE = SHADERS / "world_resolve.fs"

# What a tone-map / encode / dither looks like in GLSL code.
_FORBIDDEN = {
    "an ACES tone-map": re.compile(r"\baces", re.IGNORECASE),
    "a gamma encode": re.compile(r"linear_to_srgb|1\.0\s*/\s*2\.2"),
    "a dither": re.compile(r"dither", re.IGNORECASE),
}


def _code(glsl: str) -> str:
    """GLSL with // and /* */ comments removed (rules apply to code, and the
    layers' comments rightly point at the resolve)."""
    glsl = re.sub(r"/\*.*?\*/", "", glsl, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", glsl)


def _layer_shader_sources():
    """(name, code) of every world-layer shader: the .fs/.vs files except the
    resolve, plus the fragment sources the renderer composes in Python."""
    out = [(p.name, _code(p.read_text(encoding="utf-8")))
           for p in sorted(SHADERS.glob("*.[fv]s")) if p != RESOLVE]
    from renderer import marine_shader, static_props
    out.append(("marine_shader.MARINE_FS", _code(marine_shader.MARINE_FS)))
    out.append(("static_props.PROP_FS", _code(static_props.PROP_FS)))
    # graphics G3: the flame VFX passes as COMPOSED in Python (each stage
    # .fs with shaders/fire_vfx_common.glsl spliced in -- the .glsl part is
    # not a *.[fv]s file, so the composed source is what must be clean)
    from renderer import fire_vfx
    for st in fire_vfx.STAGES + ("debug",):
        out.append((f"fire_vfx.{st} (composed)",
                    _code(fire_vfx.compose_shader_source(st))))
    # graphics G3 F3 (design §13.14 minor 6): THE fire's draw -- the flame,
    # its compensation under the ceiling soot -- as COMPOSED (fire_glow.fs
    # with the smoke cast product's shared GLSL spliced in)
    out.append(("fire_vfx.fire_glow (composed)",
                _code(fire_vfx.compose_glow_source())))
    return out


def test_no_world_layer_shader_tone_maps_encodes_or_dithers():
    """Property: no world-layer shader applies ACES, a gamma encode or a
    dither -- the resolve does, once. Breaks the moment any layer (an
    existing one or a new one dropped into shaders/) re-adds its own."""
    bad = []
    for name, code in _layer_shader_sources():
        for what, pat in _FORBIDDEN.items():
            if pat.search(code):
                bad.append(f"{name}: {what}")
    assert not bad, ("world layers must write linear HDR; the one tone-map "
                     "is shaders/world_resolve.fs: " + "; ".join(bad))


def test_the_resolve_is_the_one_place_that_does_all_three():
    """Property: the resolve shader really carries the ACES curve, the
    encode and the dither the layers gave up. Breaks if the one tone-map is
    deleted or hollowed out (the world would show raw linear HDR)."""
    code = _code(RESOLVE.read_text(encoding="utf-8"))
    for what, pat in _FORBIDDEN.items():
        assert pat.search(code), f"world_resolve.fs lacks {what}"


# The ONE named exception (graphics G0 review): the LEGACY HEAT LAYER keeps
# its pre-G0 display-space look -- ACES-shaped display colour, added on screen
# AFTER the resolve -- because its hue, saturation and halo were tuned as a
# display colour and no linear-scene conversion keeps them
# (blackbody.pack_emissive_rgba explains). G3 F3 RE-JUSTIFIED it (design
# §13.14 minor 6): the hue gate (tools/fire_look_gate.py) measured the move
# into the linear scene failing, so the exception stays until its own
# retirement patch; the same module's glow_share uses the curve as the
# relief's 0-1 own-light share (not a drawn colour). It shapes its colour
# with the resolve's own curve, through a CPU twin that lives in that one
# module.
CPU_TWIN_HOME = {"blackbody.py": "legacy heat layer, display space (kept "
                                 "by G3 F3's hue gate) + glow_share"}


def test_no_cpu_side_tone_map_in_the_renderer():
    """Property: the only CPU tone-map in renderer/ is the resolve's CPU twin
    inside the allow-listed legacy heat layer's module, and no other module
    uses it. Breaks if a CPU-packed layer starts compressing its own light
    again (the pre-G0 ``blackbody.aces_tonemap`` fed the heat AND the gas
    packs), or a second module starts using the twin."""
    defs = re.compile(r"^\s*def\s+\w*(tone_?map|aces)\w*\s*\(",
                      re.IGNORECASE | re.MULTILINE)
    uses = re.compile(r"\baces_curve\b")
    bad = []
    for p in sorted((ROOT / "renderer").glob("*.py")):
        src = p.read_text(encoding="utf-8")
        if p.name not in CPU_TWIN_HOME and defs.search(src):
            bad.append(f"{p.name} defines a tone-map")
        if p.name not in CPU_TWIN_HOME and uses.search(src):
            bad.append(f"{p.name} uses the CPU twin")
    assert not bad, "; ".join(bad)


def test_the_cpu_twin_is_the_resolves_curve():
    """Property: the CPU ACES twin carries the resolve shader's exact
    constants. Breaks if either side is retuned alone (the legacy heat layer
    would then be shaped by a different curve than the world)."""
    from renderer.blackbody import ACES_ABCDE
    code = _code(RESOLVE.read_text(encoding="utf-8"))
    consts = [float(v) for v in re.findall(
        r"const float [abcde] = ([0-9.]+);", code)]
    assert tuple(consts) == ACES_ABCDE
