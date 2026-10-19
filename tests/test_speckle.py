"""Headless tests for the B2 P4 dirty-Planck speckle (renderer/speckle.py) — the
flame-mottle that modulates the B1 black-body overlay. Pure numpy, no GL context:
the module is pyray-free by construction (it owns no GPU resource), so its
chemistry seam (amplitude field), its motion (the shared advected-phase crossfade
on the sim tick), and the ``pack_emissive_rgba`` intensity_mod seam are all pinned
here without a window.

The load-bearing invariants (design §5 + the P4 gate):
  - off mode / zero amp is a strict NO-OP (byte-for-byte the B1 pack);
  - soot mode amplitude SCALES with the real local soot density;
  - a pure-steam cell speckles at most ~10% of an equal-density soot cell
    (the steam bound — steam must not sparkle like dirty flame);
  - the pattern is a function of the SIM TICK phase, so consecutive ticks DIFFER
    (it MOVES with the flow) and a repeated tick is bit-identical (replay-safe).

Run:
    conda run -n data python -m pytest tests/test_speckle.py -q
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

from renderer.speckle import (
    MODES, clamp_mode_idx, mode_name, mode_index,
    dirtiness, amplitude_field, advected_speckle, SpeckleField,
    _STEAM_MOTTLE_FRAC, _SOOT_REF,
)
from renderer.blackbody import BlackbodyRamp, pack_emissive_rgba, TEMP_SCALE

TPS = 24.0
CYC = 2.5


def _field(t_game_grid):
    return (np.asarray(t_game_grid, dtype=np.float64) * TEMP_SCALE).astype(np.int32)


# ---------------------------------------------------------------------------
# mode <-> index helpers (game F10 cycle + harness stepped slider)
# ---------------------------------------------------------------------------
def test_mode_index_roundtrip_and_order():
    assert MODES == ("off", "noise", "soot")            # order is load-bearing
    for i, name in enumerate(MODES):
        assert mode_name(i) == name
        assert mode_index(name) == i
    # A continuous slider value rounds/clamps into a valid step.
    assert clamp_mode_idx(1.4) == 1 and clamp_mode_idx(1.6) == 2
    assert clamp_mode_idx(-3) == 0 and clamp_mode_idx(99) == 2
    # Unknown name -> the shipped default 'soot'.
    assert mode_index("bogus") == MODES.index("soot")


# ---------------------------------------------------------------------------
# amplitude field — the chemistry seam (WHERE + HOW HARD it speckles)
# ---------------------------------------------------------------------------
def test_off_amplitude_is_zero():
    soot = np.full((6, 6), 0.5, np.float32)
    a = amplitude_field("off", 0.25, soot, np.zeros_like(soot))
    assert np.all(a == 0.0)


def test_noise_amplitude_is_uniform():
    # 'noise' = pure render noise: amplitude is `amp` everywhere, independent of
    # the gas field (the naive A/B baseline).
    soot = np.linspace(0, 1, 36, dtype=np.float32).reshape(6, 6)
    a = amplitude_field("noise", 0.25, soot, np.zeros_like(soot))
    assert np.allclose(a, 0.25)


def test_soot_amplitude_scales_with_soot_density():
    # The dirty Planck: amplitude tracks the real local soot density (below the
    # saturation ref it is monotone; the ref clamps it to `amp`).
    amp = 0.4
    densities = np.array([0.0, 0.1, 0.25, _SOOT_REF, 2.0], dtype=np.float32)
    got = np.array([
        float(amplitude_field("soot", amp,
                              np.full((1, 1), d, np.float32),
                              np.zeros((1, 1), np.float32))[0, 0])
        for d in densities])
    assert got[0] == 0.0                                # no soot -> no mottle
    assert np.all(np.diff(got) >= -1e-7)                # monotone non-decreasing
    assert got[-2] == pytest.approx(amp, rel=1e-6)      # saturates at the ref
    assert got[-1] == pytest.approx(amp, rel=1e-6)      # and clamps above it


def test_steam_mottle_is_bounded_to_ten_percent():
    # THE steam bound (design §5): a pure-steam cell must speckle at most
    # ~_STEAM_MOTTLE_FRAC (<=10%) of an equal-density soot cell — steam is clean,
    # it must never sparkle like dirty flame. Use a sub-saturation density so the
    # comparison is in the linear regime (not both clamped to `amp`).
    amp, d = 0.5, 0.2 * _SOOT_REF
    soot_amp = float(amplitude_field(
        "soot", amp, np.full((1, 1), d, np.float32),
        np.zeros((1, 1), np.float32))[0, 0])
    steam_amp = float(amplitude_field(
        "soot", amp, np.zeros((1, 1), np.float32),
        np.full((1, 1), d, np.float32))[0, 0])
    assert soot_amp > 0.0
    assert steam_amp <= _STEAM_MOTTLE_FRAC * soot_amp + 1e-7
    assert _STEAM_MOTTLE_FRAC <= 0.1 + 1e-9             # the design's <=~10%


def test_dirtiness_is_bounded_unit_interval():
    soot = np.array([[0.0, 10.0], [_SOOT_REF, 0.3]], np.float32)
    steam = np.array([[5.0, 0.0], [0.0, 0.1]], np.float32)
    d = dirtiness(soot, steam)
    assert d.dtype == np.float32
    assert np.all(d >= 0.0) and np.all(d <= 1.0)


# ---------------------------------------------------------------------------
# motion — the shared advected-phase crossfade on the SIM TICK
# ---------------------------------------------------------------------------
def _fbm():
    # Small deterministic fBm R field for the motion tests (same recipe the field
    # bakes, kept tiny for speed).
    sp = SpeckleField(1, 1)          # borrow its bake
    return sp._fbm_r


def test_advected_speckle_is_signed_and_bounded():
    n = advected_speckle(_fbm(), 12, 16, sim_tick=7,
                         cycle_seconds=CYC, tps=TPS)
    assert n.shape == (12, 16) and n.dtype == np.float32
    assert np.all(n >= -1.0 - 1e-6) and np.all(n <= 1.0 + 1e-6)


def test_speckle_moves_consecutive_ticks_differ():
    # THE hard rule: a static speckle reads as a screen overlay and is wrong. The
    # pattern is a pure function of the sim tick, so stepping the tick MUST change
    # it. Stack several ticks and require real per-cell variation (it boils).
    fbm = _fbm()
    frames = np.stack([
        advected_speckle(fbm, 24, 24, sim_tick=t, cycle_seconds=CYC, tps=TPS)
        for t in range(0, 60, 6)])
    assert frames.std(axis=0).max() > 0.05             # it genuinely moves
    # And two ADJACENT ticks already differ (not just far-apart ones).
    a = advected_speckle(fbm, 24, 24, sim_tick=11, cycle_seconds=CYC, tps=TPS)
    b = advected_speckle(fbm, 24, 24, sim_tick=12, cycle_seconds=CYC, tps=TPS)
    assert not np.allclose(a, b)


def test_speckle_is_tick_deterministic():
    # Same tick -> byte-identical field (replays / spectators render identical
    # flame; the clock is the integer tick, never wall time).
    fbm = _fbm()
    a = advected_speckle(fbm, 20, 20, sim_tick=123, cycle_seconds=CYC, tps=TPS)
    b = advected_speckle(fbm, 20, 20, sim_tick=123, cycle_seconds=CYC, tps=TPS)
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# SpeckleField.modulation — the per-frame multiplicative intensity field
# ---------------------------------------------------------------------------
def test_off_mode_modulation_is_none():
    sp = SpeckleField(8, 8, mode="off", amp=0.25)
    soot = np.full((8, 8), 0.6, np.float32)
    assert sp.modulation(soot, np.zeros_like(soot), sim_tick=5) is None


def test_zero_amp_modulation_is_none():
    sp = SpeckleField(8, 8, mode="soot", amp=0.0)
    soot = np.full((8, 8), 0.6, np.float32)
    assert sp.modulation(soot, np.zeros_like(soot), sim_tick=5) is None


def test_soot_mode_no_soot_is_none():
    # soot mode with an all-clean field -> nothing to dirty -> identity (None).
    sp = SpeckleField(8, 8, mode="soot", amp=0.25)
    zero = np.zeros((8, 8), np.float32)
    assert sp.modulation(zero, zero, sim_tick=5) is None


def test_modulation_bounded_positive_and_centered_near_one():
    sp = SpeckleField(16, 16, mode="noise", amp=0.25)
    soot = np.full((16, 16), 0.3, np.float32)
    mod = sp.modulation(soot, np.zeros_like(soot), sim_tick=9)
    assert mod is not None and mod.shape == (16, 16) and mod.dtype == np.float32
    assert np.all(mod >= 0.0)                           # never drives intensity <0
    assert np.all(mod <= 1.0 + 0.25 + 1e-6)             # 1 + amp*[-1,1]
    assert abs(float(mod.mean()) - 1.0) < 0.1           # a mottle AROUND 1.0


def test_cycle_mode_advances():
    sp = SpeckleField(4, 4, mode="off")
    assert sp.mode == "off"
    assert sp.cycle_mode() == "noise"
    assert sp.cycle_mode() == "soot"
    assert sp.cycle_mode() == "off"                     # wraps


def test_from_config_reads_section():
    class _NS:
        pass
    sp_cfg = _NS(); sp_cfg.mode = "noise"; sp_cfg.amp = 0.4
    gd = _NS(); gd.cycle_seconds = 3.0
    clock = _NS(); clock.ticks_per_second = 30
    render = _NS(); render.speckle = sp_cfg; render.gas_detail = gd
    cfg = _NS(); cfg.render = render; cfg.clock = clock
    sp = SpeckleField.from_config(6, 6, cfg)
    assert sp.mode == "noise" and sp.amp == 0.4
    assert sp.cycle_seconds == 3.0 and sp.tps == 30.0


def test_from_config_missing_section_uses_defaults():
    class _NS:
        pass
    sp = SpeckleField.from_config(6, 6, _NS())
    assert sp.mode == "soot" and sp.amp == 0.25         # shipped defaults


# ---------------------------------------------------------------------------
# the pack_emissive_rgba intensity_mod seam (B1 overlay stays byte-identical)
# ---------------------------------------------------------------------------
def test_intensity_mod_none_is_byte_identical_to_b1():
    r = BlackbodyRamp()
    temps = _field([[0, 300, 900], [1500, 3000, 9000]])
    base = pack_emissive_rgba(r, temps)
    modless = pack_emissive_rgba(r, temps, intensity_mod=None)
    assert np.array_equal(base, modless)                # strict no-op


def test_intensity_mod_changes_the_pack_on_hot_tiles():
    r = BlackbodyRamp()
    temps = _field([[3000, 3000], [3000, 3000]])        # all hot -> all glow
    base = pack_emissive_rgba(r, temps)
    dim = pack_emissive_rgba(r, temps, intensity_mod=np.full((2, 2), 0.5))
    # A 0.5x intensity mod dims the tone-mapped brightness (alpha) on hot tiles.
    assert np.all(dim[..., 3] <= base[..., 3])
    assert np.any(dim[..., 3] < base[..., 3])


def test_intensity_mod_leaves_cold_tiles_invisible():
    r = BlackbodyRamp()
    temps = _field([[0, 0], [0, 0]])                    # cold -> intensity 0
    # Even a huge mod cannot resurrect a cold tile (0 * mod = 0 -> alpha 0).
    packed = pack_emissive_rgba(r, temps, intensity_mod=np.full((2, 2), 5.0))
    assert np.all(packed[..., 3] == 0)


def test_field_modulation_does_not_mutate_inputs():
    sp = SpeckleField(8, 8, mode="soot", amp=0.3)
    soot = np.full((8, 8), 0.4, np.float32)
    steam = np.full((8, 8), 0.2, np.float32)
    s0, st0 = soot.copy(), steam.copy()
    sp.modulation(soot, steam, sim_tick=3)
    assert np.array_equal(soot, s0) and np.array_equal(steam, st0)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("OK — speckle: modes, chemistry seam, steam bound, motion, pack seam")
