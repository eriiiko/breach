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


def test_no_cpu_side_tone_map_in_the_renderer():
    """Property: no renderer module defines a CPU tone-map function (the
    pre-G0 ``blackbody.aces_tonemap`` fed the heat and gas packs). Breaks if
    a CPU-packed layer starts compressing its own light again."""
    defs = re.compile(r"^\s*def\s+\w*(tone_?map|aces)\w*\s*\(",
                      re.IGNORECASE | re.MULTILINE)
    bad = [p.name for p in sorted((ROOT / "renderer").glob("*.py"))
           if defs.search(p.read_text(encoding="utf-8"))]
    assert not bad, f"CPU tone-map defined in: {bad}"
