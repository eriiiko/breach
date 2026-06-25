"""Ripple field — Step W6a of docs/water_implementation_plan.md (canon §6).

``WaterSolver::step_ripple`` advances a damped kick-drift surface wave that is
VISUAL-ONLY: c² = g·min(depth, h_cap) (the deep-water cap pins the speed at
c_cap = √(g·h_cap) ≈ 1.57 m/s ≈ 4.7 tiles/s at dx = 1/3), splash-sourced from
``wave_p`` on wet tiles, clamped to |ripple| ≤ k_amp·depth, zeroed on
dry/solid — and it NEVER feeds back into transport (the locked canon rule).

The six plan tests, in order:
  1. decay in a still pool — discrete wave energy strictly decreases at EVERY
     one of 100 steps (gamma_r eats it)
  2. zero on dry/solid — exact, every step, while the wet side rings
  3. point splash front ≥ 3 tiles out after 1.0 s of ticks
  4. far-field — negligible beyond the wave cone (c_cap·t + 2 tiles), but
     NOT exact zero on wet tiles (numerical precursors travel 1 tile/step;
     exact zero holds only on dry/solid)
  5. clamp holds everywhere at every sampled step (and actually engages)
  6. THE KEY: visual-only guarantee — 60-tick full-Simulation A/B rollout,
     rippling live vs no-op'd: every transport field bit-identical

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_water_ripple.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))

import breach_physics as bp  # noqa: E402
from level_loader import LevelData  # noqa: E402
from simulation import Simulation  # noqa: E402
from water_q16 import q, deq  # noqa: E402  (S1: Q16.16 quantize/dequantize)

# S1: water_depth is int32 Q16.16. step_ripple now takes the int32 depth (it
# dequantizes internally at the c2 = g*min(depth,h_cap) read). ripple/ripple_v/
# wave_p stay FLOAT (render-only). _wet_box returns an int32 depth.
SEED = 42
DT = 1.0 / 24.0   # s — one game tick; ripple_max_dt() ≈ 106 ms >> this, so the
                  # runner's ONE-call-per-tick discipline is what we test at.


# ---------------------------------------------------------------------------
# helpers (deterministic — no RNG anywhere in this file)
# ---------------------------------------------------------------------------
def _solver(**overrides) -> "bp.WaterSolver":
    s = bp.WaterSolver()
    for key, val in overrides.items():
        assert hasattr(s, key), f"unknown WaterSolver param {key!r}"
        setattr(s, key, val)
    return s


def _wet_box(n: int, depth_m: float = 0.5):
    """Solid border ring, uniformly wet interior: (solid, depth) arrays. depth
    is int32 Q16.16 metres (S1)."""
    solid = np.zeros((n, n), dtype=bool)
    solid[0, :] = solid[-1, :] = solid[:, 0] = solid[:, -1] = True
    depth = np.zeros((n, n), dtype=np.int32)
    depth[1:-1, 1:-1] = q(depth_m)
    return solid, depth


def _zeros(n: int) -> np.ndarray:
    return np.zeros((n, n), dtype=np.float32)


def _sealed_room_level(n: int = 12, tile_size_m: float = 0.333) -> LevelData:
    """An n x n hull-ringed room, interior air, NO vacuum anywhere (the
    test_water_integration synthetic-LevelData pattern). 1 = hull, 4 = air."""
    tm = np.ones((n, n), dtype=np.int32)
    tm[1:n - 1, 1:n - 1] = 4
    return LevelData(
        name="ripple_room_test",
        version="1",
        path=Path("."),
        tilemap=tm,
        tile_size_m=tile_size_m,
        diffuse_path=Path("."),
    )


def _splash_rollout(ticks: int = 24, n: int = 33):
    """Single-tile wave_p impulse at the centre of a deep wet box, then free
    ringing: the shared scenario for the front + far-field tests. Returns
    (solver, ripple, depth, solid, splash_amp, t, dist) where ``splash_amp``
    is max|ripple| right after the splash tick and ``dist`` is the Euclidean
    tile distance from the splash point."""
    solid, depth = _wet_box(n, 0.5)          # 0.5 m > h_cap: c == c_cap everywhere
    ripple, ripple_v = _zeros(n), _zeros(n)
    s = _solver()
    c = n // 2
    wave = _zeros(n)
    wave[c, c] = 1.0
    s.step_ripple(ripple, ripple_v, depth, wave, solid, DT)   # the splash tick
    splash_amp = float(np.abs(ripple).max())
    assert splash_amp > 0.0, "splash never landed (vacuous scenario)"
    for _ in range(ticks - 1):
        s.step_ripple(ripple, ripple_v, depth, None, solid, DT)
    t = ticks * DT
    yy, xx = np.mgrid[0:n, 0:n]
    dist = np.hypot(yy - float(c), xx - float(c))
    return s, ripple, depth, solid, splash_amp, t, dist


def _c_cap_tiles(s) -> float:
    """The capped ripple speed in tiles/s: √(g·h_cap)/dx ≈ 4.70 at dx=0.333."""
    return float(np.sqrt(float(s.g) * float(s.h_cap)) / float(s.dx))


# ---------------------------------------------------------------------------
# 1. Decay in a still pool — energy strictly decreases at every step
# ---------------------------------------------------------------------------
def test_energy_decays_in_still_pool():
    """Uniform deep pool, an initial ripple bump, no source: the discrete wave
    energy E = Σv² + (c²/dx²)·Σ_faces(Δr)² (float64; the Hamiltonian the
    kick-drift integrates, faces = the open wet-wet pairs the mirror BC
    couples) strictly decreases at EVERY one of 100 steps — gamma_r drains it.
    """
    n = 24
    solid, depth = _wet_box(n, 0.5)          # deep: c² = g·h_cap, uniform on wet
    yy, xx = np.mgrid[0:n, 0:n]
    bump = (0.05 * np.exp(-((yy - 12.0) ** 2 + (xx - 12.0) ** 2) / (2 * 1.5 ** 2))
            ).astype(np.float32)
    ripple = np.where(depth > 0, bump, np.float32(0.0)).astype(np.float32)
    ripple_v = _zeros(n)
    s = _solver()
    c2 = float(s.g) * min(0.5, float(s.h_cap))
    dx = float(s.dx)

    def energy() -> float:
        ri = ripple[1:-1, 1:-1].astype(np.float64)   # the wet interior block
        kin = float((ripple_v[1:-1, 1:-1].astype(np.float64) ** 2).sum())
        pot = float((np.diff(ri, axis=0) ** 2).sum()
                    + (np.diff(ri, axis=1) ** 2).sum())
        return kin + (c2 / (dx * dx)) * pot

    e_prev = energy()
    assert e_prev > 0.0, "initial bump carries no energy (vacuous)"
    r0 = ripple.copy()
    for k in range(100):
        s.step_ripple(ripple, ripple_v, depth, None, solid, DT)
        e_now = energy()
        assert e_now < e_prev, (
            f"energy did not strictly decrease at step {k}: "
            f"{e_prev:.6e} -> {e_now:.6e}")
        e_prev = e_now
    # non-vacuity: the field actually evolved (the bump rang outward).
    assert not np.array_equal(ripple, r0), "ripple never evolved (vacuous run)"


# ---------------------------------------------------------------------------
# 2. Zero on dry/solid — exact, at every step
# ---------------------------------------------------------------------------
def test_ripple_exactly_zero_on_dry_and_solid():
    """Half-wet box with an interior solid block: splash + 50 steps of ringing
    leave ripple AND ripple_v exactly 0.0 on every dry and solid tile at every
    step, while the wet side is actually rippling (non-vacuity)."""
    n = 16
    solid, depth = _wet_box(n, 0.4)
    solid[6:9, 6:9] = True                  # interior wall block
    depth[:, 8:] = 0.0                      # right half dry (shore down the middle)
    depth[solid] = 0.0
    wet = depth > 0
    dead = (~wet)                           # dry OR solid — the exact-zero set
    ripple, ripple_v = _zeros(n), _zeros(n)
    s = _solver()
    wave = np.full((n, n), 0.5, dtype=np.float32)   # splash lands on wet only

    rippled = False
    for k in range(50):
        s.step_ripple(ripple, ripple_v, depth, wave if k == 0 else None,
                      solid, DT)
        assert not ripple[dead].any(), f"ripple leaked onto dry/solid at step {k}"
        assert not ripple_v[dead].any(), f"ripple_v leaked onto dry/solid at step {k}"
        rippled = rippled or bool(ripple[wet].any())
    assert rippled, "wet side never rippled (vacuous zero-check)"


# ---------------------------------------------------------------------------
# 3. Point splash front ≥ 3 tiles out after 1.0 s of ticks
# ---------------------------------------------------------------------------
def test_point_splash_front_propagates():
    """c_cap = √(9.81·0.25) ≈ 1.57 m/s ≈ 4.7 tiles/s at dx = 1/3, so after
    1.0 s the front (the 1%-of-splash level) must be ≥ 3 tiles out.
    MEASURED on the shipped stencil: it sits at 6.0 tiles."""
    s, ripple, depth, solid, splash_amp, t, dist = _splash_rollout(ticks=24)
    front = np.abs(ripple) >= 0.01 * splash_amp
    assert front.any(), "no tile above the 1% front level (wave vanished?)"
    front_dist = float(dist[front].max())
    assert front_dist >= 3.0, (
        f"splash front only reached {front_dist:.2f} tiles after {t:.2f}s "
        f"(expected >= 3; c_cap*t = {_c_cap_tiles(s) * t:.2f})")


# ---------------------------------------------------------------------------
# 4. Far-field — negligible beyond the wave cone, but NOT exact zero
# ---------------------------------------------------------------------------
def test_far_field_negligible_beyond_wave_cone():
    """Beyond c_cap·t + 2 tiles the capped wave cannot have arrived, so any
    signal there is the explicit stencil's numerical precursor (the domain of
    dependence grows 1 tile/step): nonzero on wet tiles — exact zero would be
    the WRONG assertion — but below the 1% front level everywhere, i.e. the
    visible front never outruns the capped speed.

    MEASURED deviation from the plan's constant (plan W6a says
    1e-7·splash_amp beyond +2): on the shipped stencil at dt = 1/24 s,
    t = 1.0 s, the precursor tail beyond c_cap·t + 2 peaks at
    1.4e-4 ≈ 1.8e-3·splash_amp, decaying ~one decade per tile (1e-7·splash_amp
    first holds beyond ~c_cap·t + 6). The plan's 1e-7-at-+2 is numerically
    unachievable for ANY correct implementation of its own pseudocode; the
    bound here keeps the plan's +2-tile radius and asserts the same 1% level
    that defines the front in test 3 (5.5x measured margin), which is the
    physical claim the constant was standing in for."""
    s, ripple, depth, solid, splash_amp, t, dist = _splash_rollout(ticks=24)
    cut = _c_cap_tiles(s) * t + 2.0
    far_wet = (depth > 0) & (dist > cut)
    assert far_wet.any(), "no wet tiles beyond the cone (grid too small)"
    far_max = float(np.abs(ripple[far_wet]).max())
    assert far_max < 1e-2 * splash_amp, (
        f"far field beyond {cut:.2f} tiles carries {far_max:.3e} "
        f"(>= 1% of splash_amp {splash_amp:.3e}) — the front outran c_cap")
    # NOT exact zero on wet: the precursors exist (asserting exact zero out
    # here would be wrong — the plan pins this explicitly) ...
    assert far_max > 0.0, (
        "far field is exactly zero on wet tiles — numerical precursors "
        "should exist (is the laplacian reaching its neighbours?)")
    # ... while dry/solid IS exact zero (here: the border ring).
    assert not ripple[solid].any(), "ripple nonzero on solid (exact-zero rule)"


# ---------------------------------------------------------------------------
# 5. Clamp — |ripple| ≤ k_amp·depth everywhere, every sampled step
# ---------------------------------------------------------------------------
def test_amplitude_clamp_holds_and_engages():
    """Shallow graded pool + a violent sustained splash: at every one of 50
    steps |ripple| ≤ k_amp·depth holds EVERYWHERE (the hard amplitude
    guarantee — waves no taller than the water), and the clamp actually
    engages (non-vacuity: shallow tiles ride AT the bound)."""
    n = 16
    solid, depth = _wet_box(n, 0.0)
    for y in range(1, n - 1):               # 0.02 m .. 0.30 m, row-graded
        depth[y, 1:-1] = q(0.02 + 0.28 * (y - 1) / (n - 3))
    wet = depth > 0
    ripple, ripple_v = _zeros(n), _zeros(n)
    s = _solver()
    # S1: depth is Q16.16 — the C++ clamp uses the DEQUANTIZED metres, so mirror
    # that: amp = k_amp * (depth/65536), in metres (matching the ripple units).
    amp = (np.float32(s.k_amp) * deq(depth)).astype(np.float32)
    wave = np.full((n, n), 5.0, dtype=np.float32)   # a violent blast overhead

    engaged = False
    for k in range(50):
        s.step_ripple(ripple, ripple_v, depth, wave if k < 5 else None,
                      solid, DT)
        assert np.all(np.abs(ripple) <= amp), (
            f"|ripple| exceeded k_amp*depth at step {k}: "
            f"max excess {float((np.abs(ripple) - amp).max()):.3e}")
        engaged = engaged or bool(
            np.any(np.abs(ripple[wet]) >= amp[wet] * np.float32(0.999)))
    assert engaged, "clamp never engaged (vacuous bound check)"


# ---------------------------------------------------------------------------
# 6. THE KEY TEST — the visual-only guarantee (60-tick A/B rollout)
# ---------------------------------------------------------------------------
_TRANSPORT_FIELDS = ("water_depth", "flow_vx", "flow_vy", "atmosphere",
                     "wave_p", "gas", "fire", "temperature")


def _ab_rollout(noop_ripple: bool):
    """Sealed room with a painted pool and a wave_p bump (the splash source —
    with k_p = 0.5 it also shoves the water, so transport is genuinely
    active).

    Ripple no-op swap point (Patch 1 S4a): the ripple call moved INTO
    ``PhysicsEngine::step_tail`` (C++) alongside the fire/temperature steps, so
    the old ``_step_ripple`` monkeypatch no longer intercepts it. pybind methods
    are read-only, so we wrap the engine the runner calls with a thin proxy that
    snapshots ``ripple`` / ``ripple_v`` around ``step_tail`` and restores them —
    the ripple solver still RUNS (and feeds nothing back, which is the whole
    point), but its output is discarded each tick, so a divergence in any
    transport field could only come from a ripple->transport feedback. Fire and
    temperature (the other two tail steps) are untouched by the wrapper."""
    level = _sealed_room_level(12)
    sim = Simulation(level, seed=SEED, breach_physics=bp, enable_recorder=False)
    g = sim.gmap
    interior = (~g.solid) & (~g.is_vacuum)
    g.water_depth[interior] = q(0.3)         # painted pool (Q16.16 metres, S1)
    g.wave_p[5, 4:8] = q(0.8)                # a blast ringing over the pool
                                             # (S2a: wave_p is Q16.16 int32 now)
    if noop_ripple:
        class _NoRippleEngine:
            """Forward everything to the real engine, but discard the ripple
            field's update each tick (restore ripple/ripple_v after step_tail)."""
            def __init__(self, engine):
                self._engine = engine

            def __getattr__(self, name):
                return getattr(self._engine, name)

            def step_tail(self, ripple, ripple_v, *args, **kwargs):
                r_before = ripple.copy()
                rv_before = ripple_v.copy()
                out = self._engine.step_tail(ripple, ripple_v, *args, **kwargs)
                ripple[...] = r_before
                ripple_v[...] = rv_before
                return out

        sim.physics_runner.engine = _NoRippleEngine(sim.physics_runner.engine)
    sim.set_paused(False)

    ripple_ever = False
    for _ in range(60):
        sim.step()
        ripple_ever = ripple_ever or bool(g.ripple.any())
    fields = tuple(getattr(g, name).copy() for name in _TRANSPORT_FIELDS)
    return fields, ripple_ever


def test_visual_only_ab_rollout_transport_bit_identical():
    """Rippling live vs step_ripple no-op'd, same seed, 60 ticks: EVERY
    transport field is bit-identical — the ripple feeds nothing back (the
    locked canon §6 rule). Non-vacuity: the live run actually rippled."""
    live, live_rippled = _ab_rollout(noop_ripple=False)
    noop, noop_rippled = _ab_rollout(noop_ripple=True)

    assert live_rippled, (
        "live run never rippled — the A/B proves nothing (no splash landed?)")
    assert not noop_rippled, (
        "no-op run rippled — the monkeypatch missed the runner-side call")
    for name, fa, fb in zip(_TRANSPORT_FIELDS, live, noop):
        assert np.array_equal(fa, fb), (
            f"{name} diverged between rippling-live and ripple-noop runs — "
            f"the ripple fed back into transport (locked canon §6 rule)")
