"""Headless plumbing tests for the B2 P3 gas-detail shader (Fire & Heat Beauty
arc). The GLSL runs on the GPU, but its INPUTS are numpy-packed on the CPU and
those are what break silently — so the uniform/texture packing, the
Q16.16->tiles/tick wind dequantization, the noise-bake determinism + tiling, and
the two-layer crossfade phase math are all pinned here WITHOUT a GL context.

The one that matters most (critique finding): a KNOWN wind_x Q16.16 value must
map to the EXPECTED RG-float advection in tiles/tick, or the advection is
invisibly wrong and reads as a shader bug.

Run:
    conda run -n data python -m pytest tests/test_gas_detail.py -q
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

from renderer.advected_noise import (
    bake_fbm_rgba, bake_jitter_rgba, advection_phase, layer_ages_weight,
)
from renderer.gas_detail import GasDetailPass, WIND_V_REF, tame_wind
from simulation.atmosphere_fixed import quantize as wind_quantize

TPS = 24.0
SIM_DT = 1.0 / TPS
V_REF, V_SENS = 0.08, 0.5          # the pass's wind-taming defaults


def _tamed_speed(dq_speed, v_ref=V_REF, v_sens=V_SENS):
    return v_ref * (1.0 - np.exp(-dq_speed / v_sens))


# ---------------------------------------------------------------------------
# WIND UNITS — the raw -grad(P) is fire-spiked + unusable as a velocity (the
# critique's premise was empirically wrong; module header). The pass TAMES it:
# smooth -> direction -> saturating small speed. On a UNIFORM field the 3x3
# box-smooth is identity, so the tamed velocity is exact and assertable.
# ---------------------------------------------------------------------------
def test_uniform_wind_tamed_to_expected_velocity():
    dq = 0.4                                       # dequantized raw |wind|
    q = int(wind_quantize(dq))
    wx = np.full((6, 6), q, dtype=np.int32)
    wy = np.zeros((6, 6), dtype=np.int32)
    dyn = GasDetailPass.pack_dynamics(wx, wy, np.zeros((6, 6), np.float32),
                                      v_ref=V_REF, v_sens=V_SENS)
    assert dyn.dtype == np.float16
    vmag = _tamed_speed(dq)                         # direction (1,0)
    assert float(dyn[3, 3, 0]) == pytest.approx(vmag, abs=2e-3)
    assert float(dyn[3, 3, 1]) == 0.0


def test_wind_direction_sign_preserved():
    q = lambda v: int(wind_quantize(v))
    wx = np.full((6, 6), q(-0.3), dtype=np.int32)
    wy = np.full((6, 6), q(0.5), dtype=np.int32)
    dyn = GasDetailPass.pack_dynamics(wx, wy, np.zeros((6, 6), np.float32),
                                      v_ref=V_REF, v_sens=V_SENS)
    # a negative x-gradient / positive y-gradient advects the noise that way.
    assert float(dyn[3, 3, 0]) < 0.0 and float(dyn[3, 3, 1]) > 0.0
    # direction matches the (normalized) wind vector.
    ratio_in = 0.5 / -0.3
    ratio_out = float(dyn[3, 3, 1]) / float(dyn[3, 3, 0])
    assert ratio_out == pytest.approx(ratio_in, rel=1e-2)


def test_wind_magnitude_saturates_and_is_bounded():
    # The tamed speed rises monotonically toward v_ref and NEVER exceeds it —
    # this is what bounds the fire-spiked cells so they can't shear the noise.
    prev = -1.0
    for dq in (0.01, 0.1, 1.0, 100.0, 1000.0):
        q = int(wind_quantize(dq))
        dyn = GasDetailPass.pack_dynamics(
            np.full((5, 5), q, np.int32), np.zeros((5, 5), np.int32),
            np.zeros((5, 5), np.float32), v_ref=V_REF, v_sens=V_SENS)
        mag = np.hypot(dyn[..., 0].astype(np.float64),
                       dyn[..., 1].astype(np.float64))
        assert mag.max() <= V_REF + 1e-3           # bounded by v_ref
        v = float(dyn[2, 2, 0])
        assert v >= prev - 1e-6                     # monotone increasing
        prev = v
    # A huge wind saturates to ~v_ref.
    assert prev == pytest.approx(V_REF, rel=2e-2)


def test_small_wind_is_approximately_linear():
    # For |wind| << v_sens the saturating curve is ~linear: v ~ v_ref*|w|/v_sens.
    dq = 0.01
    q = int(wind_quantize(dq))
    dyn = GasDetailPass.pack_dynamics(
        np.full((5, 5), q, np.int32), np.zeros((5, 5), np.int32),
        np.zeros((5, 5), np.float32), v_ref=V_REF, v_sens=V_SENS)
    assert float(dyn[2, 2, 0]) == pytest.approx(V_REF * dq / V_SENS, rel=5e-2)


def test_dynamics_layout_b_density_a_zero():
    wx = np.full((3, 5), int(wind_quantize(0.4)), dtype=np.int32)
    wy = np.full((3, 5), int(wind_quantize(-0.2)), dtype=np.int32)
    density = np.linspace(0, 1, 15, dtype=np.float32).reshape(3, 5)
    dyn = GasDetailPass.pack_dynamics(wx, wy, density,
                                      v_ref=V_REF, v_sens=V_SENS)
    assert dyn.shape == (3, 5, 4)
    # B carries the density solidity verbatim; A is zero.
    assert np.allclose(dyn[..., 2].astype(np.float32), density, atol=1e-3)
    assert np.all(dyn[..., 3] == 0.0)


def test_zero_wind_gives_zero_advection():
    dyn = GasDetailPass.pack_dynamics(
        np.zeros((4, 4), np.int32), np.zeros((4, 4), np.int32),
        np.full((4, 4), 0.5, np.float32), v_ref=V_REF, v_sens=V_SENS)
    assert np.all(dyn[..., 0] == 0.0) and np.all(dyn[..., 1] == 0.0)
    # density still carried (the layer breathes via the crossfade, not wind).
    assert np.allclose(dyn[..., 2].astype(np.float32), 0.5, atol=1e-3)


# ---------------------------------------------------------------------------
# tame_wind — THE render-side wind seam (props & vegetation arc #60 P4 made it
# public: prop sway is its second consumer, design §4.3 F3). Same math as
# before, now assertable on its own and re-usable without a GL context.
# ---------------------------------------------------------------------------
def test_tame_wind_shape_and_units():
    wx = np.full((4, 7), int(wind_quantize(0.3)), dtype=np.int32)
    wy = np.zeros((4, 7), dtype=np.int32)
    tamed = tame_wind(wx, wy)
    assert tamed.shape == (4, 7, 2) and tamed.dtype == np.float32
    assert float(tamed[2, 3, 0]) == pytest.approx(_tamed_speed(0.3), abs=2e-3)


def test_tame_wind_smooths_a_spike_into_its_neighbourhood():
    """A single fire-spiked cell is the failure mode the smoothing exists for:
    after taming, the spike's own cell is far below its raw share and its
    NEIGHBOURS have picked up flow — a coherent direction, not a delta."""
    wx = np.zeros((9, 9), dtype=np.int32)
    wy = np.zeros((9, 9), dtype=np.int32)
    wx[4, 4] = int(wind_quantize(500.0))       # a plume cell's raw -grad(P)
    tamed = tame_wind(wx, wy)
    peak = float(tamed[4, 4, 0])
    # Gain-limited: never above the saturation ceiling, spike or no spike.
    assert np.abs(tamed).max() <= WIND_V_REF + 1e-6
    # Smoothed: the untouched neighbours now carry a real share of the flow.
    assert float(tamed[4, 5, 0]) > 0.25 * peak
    assert float(tamed[3, 3, 0]) > 0.0
    # ...and it stays local — two box passes reach 2 cells, not the whole map.
    assert float(tamed[0, 0, 0]) == pytest.approx(0.0, abs=1e-9)


def test_tame_wind_is_monotone_and_bounded_over_five_decades():
    prev = -1.0
    for dq in (0.001, 0.01, 0.1, 1.0, 10.0, 1000.0):
        tamed = tame_wind(np.full((5, 5), int(wind_quantize(dq)), np.int32),
                          np.zeros((5, 5), np.int32))
        v = float(tamed[2, 2, 0])
        assert 0.0 <= v <= WIND_V_REF + 1e-9
        assert v >= prev - 1e-9
        prev = v
    assert prev == pytest.approx(WIND_V_REF, rel=2e-2)


def test_tame_wind_preserves_direction():
    wx = np.full((6, 6), int(wind_quantize(-0.3)), dtype=np.int32)
    wy = np.full((6, 6), int(wind_quantize(0.5)), dtype=np.int32)
    tamed = tame_wind(wx, wy)
    assert float(tamed[3, 3, 0]) < 0.0 < float(tamed[3, 3, 1])
    assert float(tamed[3, 3, 1]) / float(tamed[3, 3, 0]) == \
        pytest.approx(0.5 / -0.3, rel=1e-2)


def test_pack_dynamics_consumes_tame_wind_rather_than_forking_it():
    """The detail pass and prop sway must never disagree about which way the
    air moves in a room: pack_dynamics' RG IS tame_wind's output."""
    rng = np.random.default_rng(7)
    wx = rng.integers(-3 << 16, 3 << 16, size=(8, 6)).astype(np.int32)
    wy = rng.integers(-3 << 16, 3 << 16, size=(8, 6)).astype(np.int32)
    dyn = GasDetailPass.pack_dynamics(wx, wy, np.zeros((8, 6), np.float32))
    tamed = tame_wind(wx, wy)
    # float16 round-trip is the only difference.
    assert np.allclose(dyn[..., 0].astype(np.float32), tamed[..., 0], atol=1e-3)
    assert np.allclose(dyn[..., 1].astype(np.float32), tamed[..., 1], atol=1e-3)


# ---------------------------------------------------------------------------
# NOISE BAKE — determinism (byte-identical) + seamless tiling.
# ---------------------------------------------------------------------------
def test_fbm_bake_is_deterministic():
    a = bake_fbm_rgba(128, 4, 0.56, seed=0xB2F1)
    b = bake_fbm_rgba(128, 4, 0.56, seed=0xB2F1)
    assert a.shape == (128, 128, 4) and a.dtype == np.uint8
    assert np.array_equal(a, b)                    # byte-identical
    # A different seed changes it (not a constant field).
    c = bake_fbm_rgba(128, 4, 0.56, seed=0x1234)
    assert not np.array_equal(a, c)
    assert a[..., 3].min() == 255                  # opaque alpha


def test_fbm_channels_decorrelated():
    a = bake_fbm_rgba(256, 4, 0.56)
    r = a[..., 0].astype(np.float64)
    g = a[..., 1].astype(np.float64)
    # R (coverage) and G (warp) are baked from independent RNG draws, so they are
    # not the SAME field (the warp doesn't just echo the coverage). They share
    # the fBm's coarse-octave envelope statistics, so a modest correlation is
    # expected; the guard only needs to reject "identical" (corr == 1).
    cc = np.corrcoef(r.ravel(), g.ravel())[0, 1]
    assert abs(cc) < 0.5


def test_fbm_tiles_seamlessly():
    # Periodic construction => no discontinuity at the wrap seam: the mean
    # step across the col 255<->0 seam is comparable to an interior step.
    a = bake_fbm_rgba(256, 4, 0.56)[..., 0].astype(np.int32)
    seam = np.abs(a[:, 0] - a[:, -1]).mean()
    interior = np.abs(np.diff(a, axis=1)).mean()
    assert seam <= 3.0 * interior + 2.0            # continuous across the seam
    # Same on the vertical seam.
    seam_v = np.abs(a[0, :] - a[-1, :]).mean()
    interior_v = np.abs(np.diff(a, axis=0)).mean()
    assert seam_v <= 3.0 * interior_v + 2.0


def test_fbm_octave_counts_1_to_6_bake_cleanly():
    # Every octave count the slider allows must divide 256 seamlessly (no
    # assertion in the periodic-noise builder) and produce a valid texture.
    for oc in range(1, 7):
        img = bake_fbm_rgba(256, oc, 0.56)
        assert img.shape == (256, 256, 4)
        assert np.ptp(img[..., 0]) > 0             # actually varies


def test_jitter_bake_deterministic_and_independent():
    a = bake_jitter_rgba(64, seed=0xB217)
    b = bake_jitter_rgba(64, seed=0xB217)
    assert a.shape == (64, 64, 4) and np.array_equal(a, b)
    # Phase channel (R) and dither channel (G) are independent draws.
    cc = np.corrcoef(a[..., 0].ravel().astype(float),
                     a[..., 1].ravel().astype(float))[0, 1]
    assert abs(cc) < 0.2


# ---------------------------------------------------------------------------
# CROSSFADE PHASE — the two-layer ping-pong math (shared with P4).
# ---------------------------------------------------------------------------
def test_advection_phase_rides_the_sim_tick():
    cyc = 2.5
    tau_ticks = cyc * TPS                          # 60 ticks per cycle
    # tick 0 -> phase 0; a full cycle later -> phase ~0 (wrapped).
    assert advection_phase(0, cyc, TPS).phase == 0.0
    assert advection_phase(0, cyc, TPS).tau_ticks == pytest.approx(tau_ticks)
    assert advection_phase(int(tau_ticks), cyc, TPS).phase == pytest.approx(
        0.0, abs=1e-9)
    # Half a cycle -> phase 0.5.
    assert advection_phase(int(tau_ticks // 2), cyc, TPS).phase == pytest.approx(
        0.5, abs=1e-9)
    # Monotone increasing within a cycle.
    ph = [advection_phase(t, cyc, TPS).phase for t in range(0, 30)]
    assert all(b > a for a, b in zip(ph, ph[1:]))


def test_phase_stays_bounded_for_huge_tick():
    # A long session's large tick must still yield a bounded phase (the whole
    # point of decomposing on the CPU in float64 before handing the GPU a
    # small float — no float32 precision cliff).
    for t in (10**6, 10**7, 2 * 10**8):
        p = advection_phase(t, 2.5, TPS).phase
        assert 0.0 <= p < 1.0


def test_layer_weights_hide_each_reset():
    # Layer 0 resets at phase 0 (age0 -> 0); its weight w0 must be 0 there so the
    # UV pop is invisible. Layer 1 resets at phase 0.5 (age1 -> 0); w0 must be 1
    # there (all weight on layer 0). This is the Vlachos ping-pong guarantee.
    a0, a1, w0 = layer_ages_weight(0.0)
    assert a0 == pytest.approx(0.0) and w0 == pytest.approx(0.0, abs=1e-9)
    a0, a1, w0 = layer_ages_weight(0.5)
    assert a1 == pytest.approx(0.0) and w0 == pytest.approx(1.0, abs=1e-9)
    # Weight is a valid crossfade in [0,1] across the cycle.
    for p in np.linspace(0, 1, 41):
        _, _, w = layer_ages_weight(float(p))
        assert 0.0 - 1e-9 <= w <= 1.0 + 1e-9


def test_layer_ages_span_the_cycle():
    # Over a cycle each layer's age fraction sweeps [0,1) once; the two are
    # exactly half a cycle out of phase (the tau/2 offset).
    for p in np.linspace(0, 1, 17, endpoint=False):
        a0, a1, _ = layer_ages_weight(float(p))
        assert 0.0 <= a0 < 1.0 and 0.0 <= a1 < 1.0
        assert abs(((a0 - a1) % 1.0) - 0.5) < 1e-9


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("OK — gas_detail plumbing: wind units, dynamics layout, noise bake, "
          "phase math")
