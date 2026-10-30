"""The fire's own light in the picture (graphics arc G3 "Fire", patch F3;
docs/g3_fire_design_2026-10-23.md §3.5, §3.6, §13.4, §13.7, §13.8, §13.14).

Pure numpy / source-inspection properties of the draw: the heat layer's
linear pack and its two shares, the cover weight, the relief's own-light
share (``blackbody.glow_share``), the flame's emission law and its M4 bound
on the numpy spec twin, the flame noise, the draw helpers, and the §13.7
draw order. The GL side (the shader on a real field, A != B on the studio,
no fire -> no draws) is ``tests/gl_fire_vfx_check.py``. Pyray-free.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from renderer import blackbody as bb  # noqa: E402
from renderer.blackbody import BlackbodyRamp, TEMP_SCALE  # noqa: E402


def _ramp():
    from config import CFG
    return BlackbodyRamp.from_config(CFG)


def _q16_from_kelvin(ramp, kelvin):
    t_game = (np.asarray(kelvin, dtype=np.float64) - ramp.kelvin_ambient) \
        / ramp.k_temp_to_kelvin
    return np.round(t_game * TEMP_SCALE).astype(np.int32)


# --- the own-light share (design §13.14, round-2 M3) ---------------------------

def test_glow_share_is_the_reliefs_previous_brightness_share():
    """Property: ``blackbody.glow_share`` is the 0-1 brightness share the
    relief's own-light cut has always read -- before F3 the display-space
    heat pack's alpha over its ceiling: ``floor(max_c ACES(chroma *
    intensity) * 220) / 220`` -- to within that pack's own quantisation.
    Breaks if the share is re-derived differently (a linear radiance, an
    unshaped intensity) and the relief's tuned cut silently moves."""
    r = _ramp()
    kelvin = np.linspace(300.0, 9000.0, 400).reshape(20, 20)
    q = _q16_from_kelvin(r, kelvin)
    rgb, inten = r.chroma_intensity(q)
    old = (np.floor(bb.aces_curve(rgb * inten[..., None]).max(axis=-1) * 220.0)
           / 220.0)
    got = bb.glow_share(r, q)
    assert got.shape == q.shape and got.dtype == np.float32
    assert np.all((got >= 0.0) & (got <= 1.0))
    assert np.max(np.abs(got - old)) <= 1.0 / 220.0 + 1e-6


def test_glow_share_is_zero_when_cold_and_monotone_in_temperature():
    """Property: a tile at or below the glow floor has share 0, and a hotter
    tile never has a smaller share. Breaks if a cold tile gains a cut or the
    share inverts somewhere along the ramp."""
    r = _ramp()
    kelvin = np.linspace(250.0, 9000.0, 600)[None, :]
    s = bb.glow_share(r, _q16_from_kelvin(r, kelvin))[0]
    assert np.all(s[kelvin[0] <= r.kelvin_glow_min] == 0.0)
    assert np.all(np.diff(s) >= -1e-6)


# --- the cover weight (design §13.4) -------------------------------------------

def test_cover_weight_is_the_emission_capped_at_the_reference():
    """Property (§13.4, as F3 measured it): an emitter at or below
    ``kelvin_glow_min`` covers nothing, one at or above ``kelvin_ref`` (the
    exposure reference) covers fully, and between them the cover never
    exceeds the ramp's own intensity -- a dim emitter hides little. Breaks
    if a cool flame tip starts covering more than it glows (dark tips over a
    lit floor) or the cover moves off the ONE ramp."""
    r = _ramp()
    g, ref_k = r.kelvin_glow_min, r.kelvin_ref
    k = np.linspace(g - 300.0, ref_k + 400.0, 1401)
    w = r.cover_weight(k)
    _, inten = r.emission_at_kelvin(k)
    assert np.all(w[k <= g] == 0.0)
    assert np.all(w[k >= ref_k] == 1.0)
    assert np.all(w <= inten + 1e-12)
    assert np.all(np.diff(w) >= 0.0)


# --- the flame's emission law on the spec twin (design §3.5, §13.4) ------------

def test_flame_radiance_never_exceeds_its_feeder_and_cover_never_exceeds_E():
    """Property (round 1's M4 bound, §13.4): over a twin run of a burning
    tile, for both laws, every cell's flame radiance ``Le(T_vfx) E`` is at
    most ``Le(T_src)`` of the tile that feeds it, ``0 <= cover <= E <= 1``,
    and a flame at or below the glow floor neither emits nor covers. Breaks
    if the draw law lets a flame outshine its fuel (double counting over the
    body glow) or cover where it does not glow."""
    from renderer import fire_vfx_ref as ref
    scales = ref.load_scales()
    ramp = scales.ramp
    d = ref.FireVFXDials(m=4)
    h = w = 12
    fire = np.zeros((h, w))
    fire[6, 5:7] = 1.0
    t_src = 1330.0
    t_game = np.zeros((h, w))
    t_game[6, 5:7] = t_src - scales.t_amb_k
    z = np.zeros((h, w), bool)
    fl = z.copy()
    fl[6, 5:7] = True
    tick = ref.prepare_tick(fire, t_game, z, fl, z, np.zeros((h, w, 2)), tps=24.0,
                            tile_size_m=0.333, scales=scales, dials=d)
    st = ref.FireVFXState.zeros(h * d.m, w * d.m)
    for _ in range(150):
        st = ref.substep(st, tick, d)
    assert st.R.max() > 0.1
    ch, it = ramp.emission_at_kelvin(np.array([t_src]))
    le_src = (ch * it[:, None])[0]
    T = st.temperature(d.r_eps)
    for mode in ("A", "B"):
        Le, E, cover = ref.flame_emission(st, ramp, mode, d)
        L = Le * E[..., None]
        assert np.all(L <= le_src[None, None, :] * (1 + 1e-9) + 1e-12)
        assert np.all((cover >= 0.0) & (cover <= E + 1e-12) & (E <= 1.0))
        cold = T <= ramp.kelvin_glow_min
        assert np.all(L[cold] == 0.0) and np.all(cover[cold] == 0.0)


def test_emission_clip_cuts_faint_flame_and_keeps_strong_flame():
    """Property: the emission threshold draws nothing at or below ``e_min``
    and leaves an emission at or above ``2 e_min`` untouched, monotone
    between. Breaks if faint R (the margin boxes, B's never-zero tail) glows
    again or the threshold starts dimming a real flame."""
    from renderer.fire_vfx_ref import emission_clip
    e_min = 0.04
    E = np.linspace(0.0, 1.0, 1001)
    c = emission_clip(E, e_min)
    assert np.all(c[E <= e_min] == 0.0)
    assert np.all(c[E >= 2 * e_min] == 1.0)
    assert np.all(np.diff(E * c) >= 0.0)


def test_a_and_b_differ_on_a_twin_flame():
    """Property: the two laws give visibly different opacity on a real flame
    field (design §8 'A != B'): mean |E_A - E_B| over the flame > 0.05. Breaks
    if a dial change makes the A/B meaningless (both laws saturate the same
    way). The GL check measures it on the studio row too."""
    from renderer import fire_vfx_ref as ref
    scales = ref.load_scales()
    d = ref.FireVFXDials(m=4)
    h = w = 12
    fire = np.zeros((h, w))
    fire[6, 4:8] = 1.0
    t_game = np.zeros((h, w))
    t_game[6, 4:8] = 1330.0 - scales.t_amb_k
    z = np.zeros((h, w), bool)
    fl = z.copy()
    fl[6, 4:8] = True
    tick = ref.prepare_tick(fire, t_game, z, fl, z, np.zeros((h, w, 2)), tps=24.0,
                            tile_size_m=0.333, scales=scales, dials=d)
    st = ref.FireVFXState.zeros(h * d.m, w * d.m)
    for _ in range(150):
        st = ref.substep(st, tick, d)
    _, ea, _ = ref.flame_emission(st, scales.ramp, "A", d)
    _, eb, _ = ref.flame_emission(st, scales.ramp, "B", d)
    sel = st.R > d.r_cut
    assert np.mean(np.abs(ea - eb)[sel]) > 0.05


# --- the flame noise (design §3.5, M12) ----------------------------------------

def test_flame_noise_tiles_and_is_not_the_billow():
    """Property: the flame noise grain tiles seamlessly (period 1 on both
    axes), spans [0, 1], and is a different field from the smoke's billow.
    Breaks if the bake loses its periodicity (a seam every repeat on the
    flame's edge) or the flame reuses the billow (the cauliflower M12
    rejected)."""
    from renderer import noise_bake as nb
    xy = np.stack(np.meshgrid(np.linspace(0, 1, 33), np.linspace(0, 1, 33)),
                  axis=-1)
    f = nb.flame_noise_field(xy)
    np.testing.assert_allclose(f[0, :], f[-1, :], atol=1e-9)
    np.testing.assert_allclose(f[:, 0], f[:, -1], atol=1e-9)
    assert f.min() >= 0.0 and f.max() <= 1.0 and f.std() > 0.1
    img = nb.bake_flame_noise_rgba(64)
    bill = nb.bake_billow_rgba(64)
    assert img.shape == (64, 64, 4) and img.dtype == np.uint8
    assert np.corrcoef(img[..., 0].ravel().astype(float),
                       bill[..., 0].ravel().astype(float))[0, 1] < 0.5


# --- the draw helpers -----------------------------------------------------------

def test_noise_phases_stay_small_and_continuous():
    """Property: the flow map's phases are in [0, 1), the two phases sit half
    a cycle apart, and the grain offsets stay inside one grain repeat at any
    game age -- the GPU never sees absolute time. Breaks if raw time reaches
    a float32 uniform (the noise would stutter after hours)."""
    from renderer.fire_vfx import noise_phases
    for t in (0.0, 0.31, 12.5, 3600.0 * 7 + 0.123):
        p0, p1, offs = noise_phases(t, 0.6, 4.0, 1234)
        assert 0.0 <= p0 < 1.0 and 0.0 <= p1 < 1.0
        assert abs(((p1 - p0) % 1.0) - 0.5) < 1e-6
        assert all(0.0 <= o < 4.0 for o in offs)


def test_clip_rects_is_active_intersect_view():
    """Property: the draw rects are the active boxes intersected with the
    view, empty ones dropped (§13.10: draws are clipped to active ∩ view).
    Breaks if the draw covers the whole map or forgets the view."""
    from renderer.fire_vfx import clip_rects
    boxes = [(2, 2, 8, 6), (20, 20, 30, 30)]
    assert clip_rects(boxes, (5.5, 0.0, 25.2, 4.5)) == [(5, 2, 8, 5)]
    assert clip_rects(boxes, None) == boxes
    assert clip_rects([], (0, 0, 10, 10)) == []


def test_emission_config_is_refused_to_the_two_laws():
    """Property: ``[render.fire_vfx] emission`` is one of the two laws; an
    unknown value draws A and says so. Breaks if a typo silently selects an
    undefined shader branch."""
    from types import SimpleNamespace
    from renderer import fire_vfx as fv
    cfg = SimpleNamespace(render=SimpleNamespace(
        fire_vfx=SimpleNamespace(emission="thicknes")))
    assert fv.settings_from_config(cfg).emission == "mask"
    cfg.render.fire_vfx.emission = "thickness"
    assert fv.settings_from_config(cfg).emission == "thickness"
    assert set(fv.EMISSION_LAWS) == {"mask", "thickness"}
    assert fv.EMISSION_CYCLE[-1] == "off"


# --- the draw order (design §13.7, §13.8 fallback) ----------------------------

def test_compose_world_draws_the_fire_in_the_ruled_order():
    """Property: ``compose_world`` draws, bottom to top: floor, water, the
    floor gas, the FLAME, the units / props, the ceiling soot, the flame's
    COMPENSATION -- all in the linear scene, before the one resolve -- and
    the display-space HEAT layer after it (design §13.7; §13.8's fallback:
    the heat layer stays G0's named display exception). Erik's rulings:
    units above the flame, soot above the units. Breaks if the flame slips
    past the resolve (into display space) or out of its height, or the heat
    layer is drawn somewhere else than the one fire shader's display stage."""
    from renderer.game_renderer import GameRenderer
    src = inspect.getsource(GameRenderer.compose_world)
    order = ["self.gas_relief.render_product(",
             "self.world.begin(",
             "self.lighting.draw_lit_world(",
             "self.water_pass.draw(",
             'self.gas_relief.draw("floor"',
             'fx.draw_glow("flame"',
             "self._draw_units_3d_world(",
             'self.gas_relief.draw("ceiling"',
             'fx.draw_glow("compensation"',
             "self.world.resolve(",
             'fx.draw_glow("heat"']
    pos = [src.index(k) for k in order]
    assert pos == sorted(pos), [k for k, _ in sorted(zip(order, pos),
                                                       key=lambda t: t[1])]
    after = src[src.index("self.world.resolve("):]
    assert 'draw_glow("flame"' not in after
    assert 'draw_glow("compensation"' not in after
    assert "temperature_overlay.draw(" not in src


def _glow_stage(src: str, n: int) -> str:
    body = src[src.index("void main()"):]
    key = f"u_stage == {n}"
    if key in body:
        return body[body.index(key):].split("} else", 1)[0]
    return body[body.rindex("} else {"):]


def test_the_heat_stage_scales_only_the_display_layers_weight():
    """Property (G2's glow-through, re-homed in G3 F3's one fire shader): the
    display-space heat stage passes the layer's colour through and scales
    only its additive weight -- by the glow-through and by ``1 - eps`` of the
    flame over it -- with no tone-map or linearisation. Breaks if the heat
    layer is converted (G0 and F3 measured that losing the fire's hue) or
    stops yielding to the flame (double counting, M4)."""
    from renderer.fire_vfx import GLOW_STAGES, compose_glow_source
    src = compose_glow_source()
    heat = _glow_stage(src, GLOW_STAGES["heat"])
    assert "finalColor = vec4(c.rgb * wgt, 0.0)" in heat
    assert "(1.0 - flame_at(uv).a)" in heat
    assert "u_glow_through" in heat
    low = heat.lower()
    assert "aces" not in low and "pow(" not in low and "srgb" not in low


def test_the_flame_covers_by_its_emission_and_compensation_adds():
    """Property: the flame stage writes premultiplied ``(Le E, E w(T))`` --
    its cover IS its emission's opacity times the cover weight -- and the
    compensation re-adds only the flame, weighted by the ceiling alpha times
    ``1 - glow_through_ceiling``. Breaks if the cover decouples from the
    emission (dark tips) or the compensation stops honouring Erik's 0.2."""
    from renderer.fire_vfx import GLOW_STAGES, compose_glow_source
    src = compose_glow_source()
    assert "return vec4(c.rgb * c.a * E, E * min(c.a, 1.0));" in src
    comp = _glow_stage(src, GLOW_STAGES["compensation"])
    assert "(1.0 - u_glow_through.x)" in comp and "cov.r" in comp
    assert "flame_at(uv).rgb * wgt" in comp
