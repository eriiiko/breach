"""The G1 smoke-relief pass's CPU half (renderer/gas_relief.py, graphics arc
G1, docs/graphics_direction_2026-10-18.md §2). Headless: the packing is
pyray-free; the shader runs on the GPU and is judged in the look shots
(tools/look_shots.toml). What can break silently on the CPU is what is
held here: the dials' door, the species grouping, the one-tau identity with
the B2 medium, the exact glow split, and the masks the shader trusts to
discard pixels.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pytest

from renderer import gas_relief as gr
from renderer.gas_medium import gas_medium_layer, trace_gas_optics
from simulation.gases import GasTable, N_TRACE_GASES, SMOKE, STEAM

TABLE = GasTable.from_config()
OPTICS = trace_gas_optics(TABLE)
SHADER = ROOT / "shaders" / "gas_relief.fs"


def _cfg_namespace(**over):
    """A config-shaped namespace holding the SHIPPED [render.gas_relief]
    section, with ``over`` replacing (or, with None, deleting) keys."""
    from config import CFG
    section = dict(vars(CFG.render.gas_relief))
    for k, v in over.items():
        if v is None:
            section.pop(k, None)
        else:
            section[k] = v
    ns = type("NS", (), {})
    cfg, render, gas_relief = ns(), ns(), ns()
    for k, v in section.items():
        setattr(gas_relief, k, v)
    render.gas_relief = gas_relief
    cfg.render = render
    return cfg


def _layers(density, member):
    glow = np.ones(density.shape[1:] + (3,), np.float32)
    return gr.relief_group_layers(density, OPTICS, glow, member,
                                  tau_curve_a=1.6, tau_curve_b=1.0,
                                  glow_gain=1.0)


# --- the dials' door ---------------------------------------------------------

def test_shipped_config_parses_with_every_dial():
    """Property: the shipped config.toml carries every [render.gas_relief]
    dial with a valid type and range (the pass reads it every frame and
    refuses a bad one). Breaks when a dial is added to ReliefSettings
    without its config row, or a shipped value leaves its range."""
    from config import CFG
    s = gr.relief_settings(CFG)
    assert isinstance(s, gr.ReliefSettings)


@pytest.mark.parametrize("over", [
    {"cov_tau_ceiling": None},          # missing
    {"enabled": 1},                     # wrong type (int, not bool)
    {"billow_octaves": 4},              # out of range
    {"boil_t1": 10.0},                  # boil_t1 <= boil_t0
    {"ceiling_gases": "smoke"},         # a string, not a list
])
def test_settings_door_refuses_bad_dials(over):
    """Property: a missing, mistyped or out-of-range dial is REFUSED with a
    ValueError, never silently defaulted (a typo while tuning live must
    show). Breaks if relief_settings grows a getattr default."""
    with pytest.raises(ValueError):
        gr.relief_settings(_cfg_namespace(**over))


def test_shader_declares_every_uniform_the_pass_sets():
    """Property: every uniform name GasReliefPass binds is declared in
    shaders/gas_relief.fs (an undeclared one is a dial that silently does
    nothing). Breaks when a uniform is renamed on one side only."""
    src = SHADER.read_text(encoding="utf-8")
    declared = set(re.findall(r"uniform\s+\w+\s+(\w+)\s*;", src))
    missing = [u for u in gr.GasReliefPass._UNIFORMS if u not in declared]
    assert not missing, missing


# --- species groups (ruling 2) ---------------------------------------------

def test_every_trace_gas_lands_in_exactly_one_group():
    """Property: the ceiling and floor groups partition the trace gases --
    nothing is drawn twice or dropped -- and the named gases are the
    ceiling. Breaks if the grouping stops being a complement."""
    m = gr.group_members(OPTICS.names, ["smoke"])
    assert m["ceiling"].shape == (N_TRACE_GASES,)
    assert np.array_equal(m["ceiling"] ^ m["floor"],
                          np.ones(N_TRACE_GASES, bool))
    assert m["ceiling"][SMOKE] and m["floor"][STEAM]


def test_unknown_ceiling_gas_is_refused():
    """Property: naming a gas that does not exist is an error (a typo would
    otherwise move soot under the marines). Breaks if unknown names are
    ignored."""
    with pytest.raises(ValueError):
        gr.group_members(OPTICS.names, ["smok"])


# --- the per-tile layers -------------------------------------------------------

def _mixed_density(h=6, w=7, seed=3):
    rng = np.random.default_rng(seed)
    d = np.zeros((N_TRACE_GASES, h, w), np.float32)
    d[SMOKE] = rng.random((h, w)) * 0.02
    d[STEAM] = rng.random((h, w)) * 0.8
    return d


def test_occlusion_is_the_b2_medium_tau():
    """Property: with every gas in one group, the relief's occlusion depth
    gives EXACTLY the B2 medium's alpha (1 - exp(-tau')): one tau, one
    curve, shared (gas_optical_depth + tau_curve). Breaks if the relief
    re-derives the optical depth or its curve."""
    d = _mixed_density()
    medium, _ = _layers(d, np.ones(N_TRACE_GASES, bool))
    _, alpha = gas_medium_layer(
        d, OPTICS.k_s, base_absorb_scale=1.0, plume_k_scale=1.0,
        tau_curve_a=1.6, tau_curve_b=1.0,
        glow=np.zeros(d.shape[1:] + (3,), np.float32), glow_gain=1.0)
    relief_alpha = 1.0 - np.exp(-medium[..., 3].astype(np.float64))
    np.testing.assert_allclose(relief_alpha, alpha, atol=2e-3)


def test_group_glow_shares_sum_to_the_whole_glow():
    """Property: the two groups' glow shares add up to the sweep's glow
    (times glow_gain) on every tile -- the split moves light between layers
    and invents or loses none. Breaks if the share is not normalised by the
    all-gas in-scatter sum."""
    d = _mixed_density()
    glow = np.random.default_rng(5).random(d.shape[1:] + (3,)).astype(
        np.float32)
    m = gr.group_members(OPTICS.names, ["smoke"])
    a, _ = gr.relief_group_layers(d, OPTICS, glow, m["ceiling"],
                                  tau_curve_a=1.0, tau_curve_b=1.0,
                                  glow_gain=2.0)
    b, _ = gr.relief_group_layers(d, OPTICS, glow, m["floor"],
                                  tau_curve_a=1.0, tau_curve_b=1.0,
                                  glow_gain=2.0)
    total = a[..., :3].astype(np.float64) + b[..., :3]
    np.testing.assert_allclose(total, 2.0 * glow, rtol=3e-3, atol=1e-3)


def test_a_white_scatterer_has_a_body():
    """Property: steam, which barely absorbs, still has an extinction depth
    that grows with its density -- the coverage and height are driven by
    absorbers AND scatterers, so a white cloud has form. Breaks if the
    coverage driver falls back to the absorption tau alone."""
    lo = np.zeros((N_TRACE_GASES, 2, 2), np.float32)
    hi = lo.copy()
    lo[STEAM] = 0.2
    hi[STEAM] = 0.8
    member = gr.group_members(OPTICS.names, ["smoke"])["floor"]
    _, c_lo = _layers(lo, member)
    _, c_hi = _layers(hi, member)
    assert (c_hi[..., 3] > c_lo[..., 3]).all()
    k_scat_steam = float(OPTICS.k_scat[STEAM].mean())
    assert float(c_lo[0, 0, 3]) >= 0.2 * k_scat_steam * 0.99


def test_bulk_reflectance_is_monotone_and_below_single_scatter():
    """Property: a thick cloud's reflectance rises with its single-scatter
    albedo, is 0 at 0 and 1 at 1, and is below the single-scatter albedo in
    between (many scatterings lose light). Breaks if the surface is lit with
    the raw single-scatter albedo again (soot read near-white)."""
    w = np.linspace(0.0, 1.0, 101)
    r = gr.bulk_reflectance(w)
    assert np.all(np.diff(r) > 0)
    assert r[0] == pytest.approx(0.0) and r[-1] == pytest.approx(1.0)
    assert np.all(r[1:-1] < w[1:-1])


def test_layers_hold_nothing_where_the_group_has_no_gas():
    """Property: a group with no gas on a tile packs zero depth, zero
    extinction and zero glow there (the shader then discards it). Breaks if
    another group's gas leaks into this group's planes."""
    d = np.zeros((N_TRACE_GASES, 3, 3), np.float32)
    d[STEAM] = 0.5
    ceiling = gr.group_members(OPTICS.names, ["smoke"])["ceiling"]
    medium, cover = _layers(d, ceiling)
    assert not medium.any() and not cover[..., 3].any()


# --- the masks and fields the shader trusts ------------------------------------

def test_presence_mask_covers_the_bicubic_reach():
    """Property: the presence mask is 1 on every tile within 2 tiles of any
    smoke (the bicubic reconstruction's reach, so a discarded pixel can see
    no smoke) and 0 three tiles away. Breaks if the dilation shrinks below
    the shader's support, which would cut the edges of every plume."""
    tau = np.zeros((11, 11), np.float32)
    tau[5, 5] = 1e-4
    m = gr.presence_mask(tau)
    yy, xx = np.mgrid[:11, :11]
    cheb = np.maximum(abs(yy - 5), abs(xx - 5))
    assert (m[cheb <= 2] == 1).all() and (m[cheb >= 3] == 0).all()


def test_top_light_reaches_in_and_falls_off():
    """Property: the top light carries a bright tile's light onto dark
    neighbours within its radius, less the farther they are (falloff), and
    nothing beyond the radius plus the smoothing. Breaks if the falloff is
    dropped (a fire's light carried undiminished) or the reach is lost."""
    e = np.zeros((15, 15, 3), np.float32)
    e[7, 7] = 1.0
    top = gr.top_light_field(e, 2, 0.5)[..., 0]
    assert top[7, 8] > top[7, 9] > 0.0
    assert top[7, 7 + 2 + gr._TOP_LIGHT_SMOOTH_PASSES + 1] == 0.0
    flat = gr.top_light_field(e, 2, 1.0)[..., 0]
    assert flat[7, 9] > top[7, 9]


def test_boil_is_zero_cold_one_hot_and_monotone():
    """Property: boil is 0 at or below boil_t0, 1 at or above boil_t1 and
    never decreases with temperature. Breaks if the clamp or the ramp
    direction is lost (cold smoke churning)."""
    t = np.linspace(-100.0, 1000.0, 221)
    b = gr.boil_field(t, 50.0, 600.0)
    assert np.all(b[t <= 50.0] == 0.0) and np.all(b[t >= 600.0] == 1.0)
    assert np.all(np.diff(b) >= 0.0)


def test_clip_rect_is_inside_the_grid():
    """Property: the view rect the pass draws is clamped to the grid and
    covers the asked span (floor of the start, ceiling of the end); None is
    the whole grid. Breaks if a partial tile at the view's edge is dropped
    (a strip of smoke missing at the screen border)."""
    assert gr.clip_tile_rect(None, 48, 32) == (0, 0, 48, 32)
    assert gr.clip_tile_rect((-2.0, 3.5, 60.0, 10.2), 48, 32) == (0, 3, 48, 11)
