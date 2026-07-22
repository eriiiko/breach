"""The physical gas-medium pass — pyray-free numpy tests (Fire & Heat Beauty
B2 P2). Exercises the pure core of ``renderer/gas_medium.py``
(``gas_medium_layer`` / ``pack_premult_rgba`` / ``pack_gas_medium_rgba``) —
no GL context, no window. Fixtures are built from the REAL GasTable
(``simulation.gases``) so ``k_s`` / albedo / effect shapes are authentic.

Design: docs/fire_b2_smoke_honesty_design_2026-07-21.md §3 gate. This file
also folds in the intent of the retired tests/test_smoke_render_gamma.py — the
render-contrast curve now lives in TAU-space (tau' = a·tau^b), not on alpha.

Run:
    conda run -n data python -m pytest tests/test_gas_medium.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))

import numpy as np
import pytest

from renderer.gas_medium import (
    gas_medium_layer, pack_premult_rgba, pack_gas_medium_rgba,
    _EFFECT_GAS_TAGS,
)
from simulation.gases import (
    GasTable, STEAM, SMOKE, POISON, TEARGAS, N_TRACE_GASES,
)

# --- authentic fixtures from the real GasTable ------------------------------
TABLE = GasTable.from_config()
# k_s = mean over the absorption RGB triple — the panchromatic collapse the ray
# march (and beam_absorb_q16) use. Steam ~0.10, soot ~0.90 (soot dominates).
K_S = np.asarray(TABLE.absorption[:N_TRACE_GASES], dtype=np.float32).mean(axis=1)
SCATTER = np.asarray(TABLE.scatter_albedo[:N_TRACE_GASES], dtype=np.float32)
EFFECT_MASK = np.array(
    [str(e) in _EFFECT_GAS_TAGS for e in TABLE.effect[:N_TRACE_GASES]], dtype=bool)


def _single_gas(gas_id: int, density: float, h: int = 8, w: int = 8):
    g = np.zeros((N_TRACE_GASES, h, w), dtype=np.float32)
    g[gas_id] = float(density)
    return g


def _zero_glow(h: int = 8, w: int = 8):
    return np.zeros((h, w, 3), dtype=np.float32)


def _base_kw(**over):
    kw = dict(base_absorb_scale=1.4, plume_k_scale=1.0, tau_curve_a=1.0,
              tau_curve_b=1.0, glow_gain=1.0)
    kw.update(over)
    return kw


# ---------------------------------------------------------------------------
# Authentic-fixture sanity: soot is the dark, high-extinction gas; the effect
# mask picks the gameplay gases (poison + teargas), not fire-smoke/steam.
# ---------------------------------------------------------------------------
def test_fixture_shapes_and_identities():
    assert K_S.shape == (N_TRACE_GASES,)
    assert SCATTER.shape == (N_TRACE_GASES, 3)
    assert EFFECT_MASK.shape == (N_TRACE_GASES,)
    # Soot's extinction dwarfs steam's (the whole point of the crossover).
    assert K_S[SMOKE] > 5.0 * K_S[STEAM]
    # Gameplay ("effect") gases get the legibility floor; fire-smoke/steam don't.
    assert EFFECT_MASK[POISON] and EFFECT_MASK[TEARGAS]
    assert not EFFECT_MASK[SMOKE] and not EFFECT_MASK[STEAM]


# ---------------------------------------------------------------------------
# 1. Thin limit: for small tau at a=b=1, d(alpha)/d(rho) -> base*plume*k_s
#    (Beer-Lambert linearization 1 - exp(-x) ~ x). With base_absorb_scale = 1
#    this is the design's shorthand slope ~ plume_k_scale * k_s.
# ---------------------------------------------------------------------------
def test_thin_limit_slope_is_base_plume_k():
    rho = 1e-4
    for gas_id in range(N_TRACE_GASES):
        k = float(K_S[gas_id])
        for base, plume in ((1.0, 1.0), (1.0, 2.0), (1.4, 1.0)):
            g = _single_gas(gas_id, rho, 2, 2)
            _, alpha = gas_medium_layer(
                g, K_S, smoke_glow=_zero_glow(2, 2),
                **_base_kw(base_absorb_scale=base, plume_k_scale=plume))
            slope = float(alpha[0, 0]) / rho
            assert slope == pytest.approx(base * plume * k, rel=1e-3, abs=1e-6)


# ---------------------------------------------------------------------------
# 2. Soot-dominates-steam crossover: a soot+steam mix blackens — soot's high
#    k_s carries the alpha, and with no light the lit half is ~black.
# ---------------------------------------------------------------------------
def test_soot_dominates_steam_crossover():
    d = 0.5
    kw = _base_kw()
    _, a_steam = gas_medium_layer(_single_gas(STEAM, d), K_S,
                                  smoke_glow=_zero_glow(), **kw)
    _, a_soot = gas_medium_layer(_single_gas(SMOKE, d), K_S,
                                 smoke_glow=_zero_glow(), **kw)
    rgb_mix, a_mix = gas_medium_layer(
        _single_gas(STEAM, d) + _single_gas(SMOKE, d), K_S,
        smoke_glow=_zero_glow(), **kw)
    a_steam, a_soot, a_mix = a_steam[0, 0], a_soot[0, 0], a_mix[0, 0]
    # The mix is far more opaque than steam alone.
    assert a_mix > a_steam
    # Soot carries the great majority of the mix's optical depth.
    soot_frac = float(K_S[SMOKE]) / float(K_S[SMOKE] + K_S[STEAM])
    assert soot_frac > 0.8
    # ... so the mix's alpha tracks SOOT-only far closer than STEAM-only.
    assert abs(a_mix - a_soot) < abs(a_mix - a_steam)
    # Dark room: soot barely scatters -> the lit half is black (black occluder).
    assert rgb_mix.max() < 1e-6


# ---------------------------------------------------------------------------
# 3. Alpha monotone in tau and bounded [0,1] — including under the artistic
#    tau-curve (a, b), which must never break monotonicity or the [0,1] bound.
# ---------------------------------------------------------------------------
def test_alpha_monotone_and_bounded():
    densities = np.linspace(0.0, 5.0, 60)
    for a_curve, b_curve in [(1.0, 1.0), (1.5, 2.0), (0.5, 0.7), (2.0, 3.0)]:
        alphas = []
        for d in densities:
            _, al = gas_medium_layer(
                _single_gas(SMOKE, float(d), 1, 1), K_S,
                smoke_glow=_zero_glow(1, 1),
                **_base_kw(tau_curve_a=a_curve, tau_curve_b=b_curve))
            alphas.append(float(al[0, 0]))
        alphas = np.array(alphas)
        assert alphas.min() >= 0.0 and alphas.max() <= 1.0
        assert np.all(np.diff(alphas) >= -1e-7), (
            f"non-monotone at a={a_curve} b={b_curve}")


# ---------------------------------------------------------------------------
# 4. All-zero gas + zero glow -> fully transparent black (RGBA all ~0).
# ---------------------------------------------------------------------------
def test_all_zero_gas_and_glow_is_transparent_black():
    g = np.zeros((N_TRACE_GASES, 6, 6), dtype=np.float32)
    packed = pack_gas_medium_rgba(g, K_S, smoke_glow=_zero_glow(6, 6),
                                  **_base_kw())
    assert packed.shape == (6, 6, 4)
    assert packed.dtype == np.uint8
    assert np.all(packed == 0)      # transparent black: RGBA all zero


# ---------------------------------------------------------------------------
# 5. ADDITIVE LIMIT (premult's additive case): zero optical depth with NONZERO
#    smoke_glow -> alpha ~ 0 with RGB > 0. This is steam glowing in a beam and
#    MUST be allowed. RGB is deliberately NOT bounded by alpha — do not "fix" it.
# ---------------------------------------------------------------------------
def test_additive_limit_zero_tau_nonzero_glow():
    g = np.zeros((N_TRACE_GASES, 4, 4), dtype=np.float32)     # no gas -> tau = 0
    glow = np.full((4, 4, 3), 0.5, dtype=np.float32)          # a lit beam
    rgb, alpha = gas_medium_layer(g, K_S, smoke_glow=glow, **_base_kw())
    assert np.allclose(alpha, 0.0)          # no occlusion at all
    assert np.all(rgb > 0.0)                # but the inscatter is visible
    # The premultiplied pack keeps RGB > alpha here (additive case).
    packed = pack_premult_rgba(rgb, alpha)
    assert np.all(packed[..., 3] == 0)      # alpha byte 0
    assert np.all(packed[..., 0] > 0)       # RGB bytes nonzero — RGB NOT <= alpha
    assert np.all(packed[..., 1] > 0)
    assert np.all(packed[..., 2] > 0)


# ---------------------------------------------------------------------------
# tau-curve (folds the retired smoke_render_gamma test's intent): a=b=1 is the
# honest identity alpha = 1 - exp(-tau); b>1 steepens edges (thin crushed
# toward clear, thick pushed toward opaque), the contrast now living in
# TAU-space rather than on the packed alpha.
# ---------------------------------------------------------------------------
def test_tau_curve_honest_at_unit_and_steepens_with_b():
    k = float(K_S[SMOKE])
    for dd in np.linspace(0.01, 2.0, 20):
        _, a_honest = gas_medium_layer(
            _single_gas(SMOKE, float(dd), 1, 1), K_S,
            smoke_glow=_zero_glow(1, 1),
            **_base_kw(base_absorb_scale=1.0))
        tau = k * float(dd)
        assert float(a_honest[0, 0]) == pytest.approx(1.0 - np.exp(-tau),
                                                       abs=1e-6)
    kw = dict(base_absorb_scale=1.0, plume_k_scale=1.0, glow_gain=1.0,
              smoke_glow=_zero_glow(1, 1))
    # tau ~ 0.18 (< 1): b>1 shrinks it -> more transparent than honest.
    thin = _single_gas(SMOKE, 0.2, 1, 1)
    _, thin_h = gas_medium_layer(thin, K_S, tau_curve_a=1.0, tau_curve_b=1.0, **kw)
    _, thin_s = gas_medium_layer(thin, K_S, tau_curve_a=1.0, tau_curve_b=2.0, **kw)
    assert thin_s[0, 0] < thin_h[0, 0]
    # tau ~ 2.7 (> 1): b>1 grows it -> more opaque than honest.
    thick = _single_gas(SMOKE, 3.0, 1, 1)
    _, thick_h = gas_medium_layer(thick, K_S, tau_curve_a=1.0, tau_curve_b=1.0, **kw)
    _, thick_s = gas_medium_layer(thick, K_S, tau_curve_a=1.0, tau_curve_b=2.0, **kw)
    assert thick_s[0, 0] > thick_h[0, 0]


# ---------------------------------------------------------------------------
# Non-physical legibility floor: with the floor raised, gameplay gases
# (poison/teargas) self-emit in the dark for legibility; fire-smoke/steam stay
# black occluders. Off (default 0) everything is a black occluder in the dark.
# ---------------------------------------------------------------------------
def test_effect_gas_floor_lifts_gameplay_gases_only():
    d = 0.8
    on = _base_kw(effect_gas_floor=0.5)
    rgb_poison, _ = gas_medium_layer(_single_gas(POISON, d), K_S,
                                     smoke_glow=_zero_glow(),
                                     effect_gas_mask=EFFECT_MASK,
                                     scatter_albedo=SCATTER, **on)
    rgb_smoke, _ = gas_medium_layer(_single_gas(SMOKE, d), K_S,
                                    smoke_glow=_zero_glow(),
                                    effect_gas_mask=EFFECT_MASK,
                                    scatter_albedo=SCATTER, **on)
    assert rgb_poison.max() > 1e-3    # poison glows (green legibility floor)
    assert rgb_smoke.max() < 1e-6     # fire-smoke stays a black occluder
    # Floor OFF (default): even poison is black in the dark (honest default).
    rgb_poison_off, _ = gas_medium_layer(
        _single_gas(POISON, d), K_S, smoke_glow=_zero_glow(),
        effect_gas_mask=EFFECT_MASK, scatter_albedo=SCATTER,
        **_base_kw(effect_gas_floor=0.0))
    assert rgb_poison_off.max() < 1e-6


if __name__ == "__main__":
    test_fixture_shapes_and_identities()
    test_thin_limit_slope_is_base_plume_k()
    test_soot_dominates_steam_crossover()
    test_alpha_monotone_and_bounded()
    test_all_zero_gas_and_glow_is_transparent_black()
    test_additive_limit_zero_tau_nonzero_glow()
    test_tau_curve_honest_at_unit_and_steepens_with_b()
    test_effect_gas_floor_lifts_gameplay_gases_only()
    print("OK — gas_medium: thin-limit, soot crossover, monotone, transparent "
          "black, additive limit, tau-curve, effect floor")
