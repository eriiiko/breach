"""The smoke's CAST SHADOW (graphics arc G2, docs/graphics_direction_
2026-10-18.md §2.3, §12): one GLSL copy in ``renderer/lit3d.py`` run by every
receiver (the floor's ``shaders/lighting.fs`` and the lit-3D consumers), fed
by the relief's one cast product (``GasReliefPass.product_texture``).

Headless: the shader sources are plain strings, the uniforms a pure function.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from renderer import gas_relief as gr  # noqa: E402
from renderer import lit3d  # noqa: E402


def _settings(**over):
    from config import CFG
    s = gr.relief_settings(CFG)
    return s.__class__(**{**s.__dict__, **over})


def test_every_receiver_runs_the_one_shadow_copy():
    """Property: the floor and every lit-3D consumer run the SAME shadow GLSL
    -- lit3d's ``_SMOKE_SHADOW_GLSL``, verbatim, exactly once -- and the
    on-disk ``lighting.fs`` defines no shadow function of its own (it is
    spliced in at the marker). Breaks when a receiver grows an inline copy
    that could drift (the lit3d seam's one-copy rule)."""
    from renderer.lighting import load_lighting_fs
    from renderer.marine_shader import MARINE_FS
    from renderer.static_props import PROP_FS
    block = lit3d._SMOKE_SHADOW_GLSL
    for name, src in (("lighting.fs", load_lighting_fs()),
                      ("MARINE_FS", MARINE_FS), ("PROP_FS", PROP_FS)):
        assert src.count(block) == 1, name
        assert len(re.findall(r"float\s+smokeShadow\s*\(", src)) == 1, name
    raw = (ROOT / "shaders" / "lighting.fs").read_text(encoding="utf-8")
    assert not re.search(r"float\s+smokeShadow\s*\(", raw)


def test_every_receiver_applies_the_shadow_to_its_incoming_light():
    """Property: each receiver multiplies its INCOMING light by the shadow
    (the ambient floor is not a lamp and casts nothing). Breaks if a
    receiver declares the block but never calls it."""
    from renderer.lighting import load_lighting_fs
    assert "incoming_rgb *= smoke_sh" in load_lighting_fs()
    assert re.search(r"incoming_rgb\s*\*=\s*smokeShadow\(",
                     lit3d._FIELD_SAMPLE_GLSL)


def test_inactive_relief_casts_no_shadow():
    """Property: with the relief off (or the product inactive) both shadow
    strengths are 0, which the GLSL reads as "no shadow" before sampling the
    product. Breaks if a stale product could darken the floor while the
    relief is not drawn (the B2 A/B look, smoke hidden)."""
    s = _settings(cast_shadow_ceiling=0.9, cast_shadow_floor=0.7)
    on, _ = gr.cast_uniforms(s, active=True)
    assert on[0] == pytest.approx(0.9) and on[1] == pytest.approx(0.7)
    off, _ = gr.cast_uniforms(s, active=False)
    assert off[0] == 0.0 and off[1] == 0.0
    disabled, _ = gr.cast_uniforms(_settings(enabled=False), active=True)
    assert disabled[0] == 0.0 and disabled[1] == 0.0
    assert "if (u_smoke_cast.x + u_smoke_cast.y <= 0.0) return 1.0;" \
        in lit3d._SMOKE_SHADOW_GLSL


@pytest.mark.parametrize("over", [
    {"cast_shadow_ceiling": 1.5},
    {"cast_shadow_floor": -0.1},
    {"cast_parallax": -1.0},
    {"cast_lift_ceiling": -2.0},
])
def test_cast_dials_are_refused_out_of_range(over):
    """Property: a cast-shadow strength outside [0, 1] or a negative
    geometry dial is REFUSED at the settings door, not drawn. Breaks if the
    door stops range-checking the G2 dials."""
    from config import CFG
    section = dict(vars(CFG.render.gas_relief))
    section.update(over)

    class NS:
        pass
    cfg, render, rel = NS(), NS(), NS()
    for k, v in section.items():
        setattr(rel, k, v)
    render.gas_relief = rel
    cfg.render = render
    with pytest.raises(ValueError):
        gr.relief_settings(cfg)


def test_product_mode_writes_the_layers_own_coverage():
    """Property: the cast product comes from the relief shader itself
    (``u_mode`` 1), from the same coverage C and height h the layer draw
    composites -- one smoke model -- and each group writes its own channel
    pair (ceiling RG, floor BA) so the two sum without mixing. Breaks if the
    product is computed by a second smoke model or the groups' channels
    collide."""
    src = (ROOT / "shaders" / "gas_relief.fs").read_text(encoding="utf-8")
    m = re.search(r"if \(u_mode == 1\) \{(.*?)\n    \}", src, re.S)
    assert m, "no product branch in gas_relief.fs"
    body = m.group(1)
    assert "C;" in body and ", h," in body
    assert "vec4(ap, h, 0.0, 0.0)" in body and "vec4(0.0, 0.0, ap, h)" in body
