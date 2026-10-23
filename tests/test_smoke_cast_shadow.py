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


# --- fire glow-through (G2) ----------------------------------------------------

def test_glow_below_is_only_a_bodys_glow_beyond_the_hot_gas_around_it():
    """Property: the share of a tile's glow that lies BELOW the smoke is 0 on
    a gas tile (a hot smoke glows by itself, #95's question is untouched),
    1 on a body burning alone in cool gas, 0 on a body inside an equally hot
    cloud (no dark hole where the crate was), and always in [0, 1]. Breaks
    if the glow-through starts hiding hot gas under its own depth, or
    punches holes into a glowing cloud."""
    import numpy as np
    body = np.zeros((7, 7), bool)
    body[3, 3] = True
    alone = np.zeros((7, 7), np.float32)
    alone[3, 3] = 0.9
    f = gr.glow_below_fraction(alone, body)
    assert f[3, 3] == pytest.approx(1.0)
    assert f.min() >= 0.0 and f.max() <= 1.0
    cloud = np.full((7, 7), 0.9, np.float32)
    assert gr.glow_below_fraction(cloud, body)[3, 3] == pytest.approx(0.0)
    no_body = gr.glow_below_fraction(cloud, np.zeros((7, 7), bool))
    assert np.all(no_body == 0.0)


def test_glow_below_spills_onto_the_bodys_neighbours_only():
    """Property: the below share reaches one tile past a glowing body (the
    heat layer is bilinear, so the body's glow spills half a tile; without
    it a hidden crate wore a bright frame) and no further. Breaks if the
    spill is dropped or widened."""
    import numpy as np
    body = np.zeros((7, 7), bool)
    body[3, 3] = True
    bright = np.zeros((7, 7), np.float32)
    bright[3, 3] = 0.8
    f = gr.glow_below_fraction(bright, body)
    assert f[3, 4] == pytest.approx(1.0) and f[2, 2] == pytest.approx(1.0)
    assert f[3, 5] == 0.0 and f[0, 0] == 0.0


def test_inactive_relief_lets_the_heat_glow_through_untouched():
    """Property: with the relief off (or no product) the glow-through
    weights are 0 -- the heat layer draws exactly as before G2. Breaks if a
    stale product could hide fires while the relief is not drawn."""
    s = _settings(glow_through_ceiling=1.0, glow_through_floor=0.5)
    assert gr.glow_through_weights(s, active=False) == (0.0, 0.0)
    assert gr.glow_through_weights(_settings(enabled=False), True) == (0.0, 0.0)
    assert gr.glow_through_weights(s, active=True) == (1.0, 0.5)


# --- draw order (G2) -----------------------------------------------------------

def test_compose_world_draws_by_height():
    """Property: GameRenderer.compose_world draws bottom to top -- the cast
    product before the scene RT is bound, then floor, water (the floor's
    surface), the FLOOR smoke layer (haze lying on the water), the 3D units,
    the CEILING smoke layer (Erik's ruling 2), the resolve, and the heat
    layer after it (display space, glow-through). Breaks if a layer is moved
    out of its height (water over the haze, soot under the marines, the
    product rendered inside the scene RT)."""
    import inspect
    from renderer.game_renderer import GameRenderer
    src = inspect.getsource(GameRenderer.compose_world)
    order = ["self.gas_relief.render_product(",
             "self.world.begin(",
             "self.lighting.draw_lit_world(",
             "self.water_pass.draw(",
             'self.gas_relief.draw("floor"',
             "self._draw_units_3d_world(",
             'self.gas_relief.draw("ceiling"',
             "self.world.resolve("]
    pos = [src.index(k) for k in order]
    assert pos == sorted(pos), [k for k, _ in sorted(zip(order, pos),
                                                       key=lambda t: t[1])]


# --- the synthetic low band (G2, doc §2.5 A) -----------------------------------

@pytest.mark.parametrize("over", [
    {"low_band_tiles": 0.0},
    {"low_band_push_tiles": -0.5},
    {"low_band_height": 1.5},
    {"low_band_tau0": 1.0, "low_band_tau1": 0.5},
])
def test_low_band_dials_are_refused_out_of_range(over):
    """Property: a non-positive lump size, a negative push, a carve outside
    [0, 1] or an inverted gate is REFUSED at the settings door. Breaks if
    the door stops checking the low band's dials."""
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


def test_low_band_is_gated_and_acts_through_the_coverage_edge():
    """Property: the low band is render-only structure that (a) is skipped
    entirely at push 0 and carve 0 (the A/B), (b) is gated by tau over the
    coverage depth (only inside plumes), (c) moves the coverage contour by
    |grad ln tau| x its displacement (a flat field is untouched, Erik's
    ruling 7: only where the field changes), and (d) rides the relief's own
    two-phase clock offsets (the real wind, one noise clock). Breaks if the
    band is applied ungated, outside the coverage contour, or on a clock of
    its own."""
    src = (ROOT / "shaders" / "gas_relief.fs").read_text(encoding="utf-8")
    assert "if (u_low_push > 0.0 || u_low_height > 0.0)" in src
    assert re.search(r"smoothstep\(u_low_gate\.x,\s*u_low_gate\.y,\s*"
                     r"tau_ext\s*/\s*max\(u_cov_tau", src)
    assert "+ grad * g_low_d;" in src
    assert "g_o1 * sc" in src and "g_o0 * sc" in src and "g_w0" in src
