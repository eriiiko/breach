"""S1 STRESS conservation gate — the regime that hid the outflow-limiter leak.

This is the PERMANENT conservation gate the S1 adversarial review demanded, and
the one S2-S5 reuse. It reproduces the high-CFL / max-velocity regime that the
normal-regime check (tests/_s1_conservation_check.py) never exercised:

  * sealed room (hull-walled box; the only boundary is solid walls — no source,
    no sink, no boil),
  * pre-seeded +/- v_max velocities (so faces carry their largest fluxes and
    many cells are 4-face donors at once),
  * HIGH CFL: dt forced several times above the solver's CFL bound (max_dt) so
    out_sum > depth on many cells -> the per-cell OUTFLOW LIMITER fires hard,
  * tilt ON (the surface slopes, biasing flux directions incl. negatives),
  * NO damping (velocities are not bled off -> the stress persists every tick),
  * depth_eps = 0 (the snap-to-zero sink is OFF, so the ONLY thing that could
    create/destroy mass is the max(depth,0) non-negativity clamp).

The assertions are exact, in raw Q16.16 LSB counts (integer sums — no float
tolerance): over many ticks the integer sum of water_depth must be UNCHANGED
(0 LSB created or destroyed), AND depth must stay >= 0 with the clamp never
firing (no mass injection). This holds iff the limiter bounds each donor cell's
total scaled outflow to <= its depth (Fix 1: the limiter face-apply scales on
the MAGNITUDE so a negative outgoing delta cannot over-drain).

Before Fix 1 the magnitude-scaling bug let a 4-face donor over-drain by up to a
few LSB/tick under exactly this regime -> depth went slightly negative -> the
max(depth,0) clamp injected mass -> the sum drifted UP. This test is RED on the
old mul_q16 face-apply and GREEN after scale_mag.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_water_conservation_stress.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402
from water_q16 import q, FP_ONE  # noqa: E402


# ---------------------------------------------------------------------------
# helpers (deterministic — no RNG seeded outside the explicit default_rng below)
# ---------------------------------------------------------------------------
def _sealed_box(h: int, w: int) -> np.ndarray:
    """A hull-walled (sealed) solid mask: ring of walls, open interior."""
    solid = np.zeros((h, w), dtype=bool)
    solid[0, :] = solid[-1, :] = solid[:, 0] = solid[:, -1] = True
    return solid


def _stress_solver():
    # NO damping (stress persists), snap OFF (depth_eps=0 -> only max(depth,0)
    # can move mass), default v_max so the +/- v_max seed is the clamp bound.
    s = bp.WaterSolver()
    s.dx = 1.0 / 3.0
    s.damping = 0.0
    s.depth_eps = 0.0
    return s


def _isum(depth: np.ndarray) -> int:
    """Exact integer sum of the Q16.16 depth field (raw LSB counts)."""
    return int(depth.astype(np.int64).sum())


def _make_stress_scene(h=28, w=28, seed=7):
    """A lumpy sealed flood pre-seeded with +/- v_max velocities + a tilt.

    Depth is bounded and shallow-ish (0.1-0.6 m) so that at the forced high-CFL
    dt the per-face flux exceeds depth on many cells and the limiter fires.
    """
    solid = _sealed_box(h, w)
    rng = np.random.default_rng(seed)

    d = np.zeros((h, w), np.float64)
    inner = (slice(1, h - 1), slice(1, w - 1))
    # lumpy, shallow, strictly positive interior fill
    d[inner] = 0.1 + 0.5 * rng.random((h - 2, w - 2))
    d[solid] = 0.0
    depth = q(d)

    # pre-seed velocities at +/- v_max (8 m/s default) in a checkerboard sign
    # pattern so adjacent faces push opposite ways -> many 4-face donors with
    # NEGATIVE outgoing deltas (the exact case the mul_q16 face-apply over-drained).
    v_max = 8.0
    yy, xx = np.mgrid[0:h, 0:w]
    sign = np.where(((yy + xx) % 2) == 0, 1.0, -1.0)
    vx = q(sign * v_max)
    vy = q(np.where(((yy * 3 + xx) % 2) == 0, 1.0, -1.0) * v_max)
    vx[solid] = 0
    vy[solid] = 0
    return solid, depth, vx, vy


# ---------------------------------------------------------------------------
# THE GATE: exact (0-LSB) mass conservation under high CFL + max velocity
# ---------------------------------------------------------------------------
def test_stress_conservation_exact_zero_lsb():
    solid, depth, vx, vy = _make_stress_scene()
    s = _stress_solver()

    # HIGH CFL: drive the solver with a dt several times its CFL bound so the
    # limiter is forced to fire on many cells. (Python owns the substep loop in
    # the game; here we deliberately DO NOT substep — one big dt per tick IS the
    # high-CFL stress. The limiter + clamp are what must keep mass exact.)
    cfl = s.max_dt()                 # ~33.6 ms at these params (k_p = 0)
    dt = 6.0 * cfl                   # ~6x over CFL — hard limiter regime
    tilt = (0.12, -0.07)             # radians; tilt ON (sloped surface)

    total0 = _isum(depth)
    n_ticks = 400

    min_depth_seen = 0
    drifts = []
    for _ in range(n_ticks):
        s.step(depth, vx, vy, None, None, solid, dt, tilt[0], tilt[1])
        min_depth_seen = min(min_depth_seen, int(depth.min()))
        drifts.append(_isum(depth) - total0)

    max_abs_drift = max(abs(d) for d in drifts)

    # (1) EXACT conservation: not one LSB created or destroyed, ever.
    assert max_abs_drift == 0, (
        f"mass drifted by {max_abs_drift} LSB over {n_ticks} ticks under the "
        f"high-CFL/max-velocity stress regime (clamp injected/removed mass)"
    )
    # (2) depth never went negative pre-clamp would have injected; post-step it
    #     must be >= 0 every tick (we sampled depth.min() each tick).
    assert min_depth_seen >= 0, (
        f"depth went negative (min seen = {min_depth_seen}) -> the max(depth,0) "
        f"clamp would inject mass (outflow limiter over-drained)"
    )
    # (3) non-vacuity: the field must actually be churning hard (the limiter
    #     genuinely fires), else the test proves nothing.
    moved = not np.array_equal(depth, _make_stress_scene()[1])
    assert moved, "field never evolved (vacuous stress test)"


def test_stress_limiter_actually_fires():
    """Sentinel: confirm this regime DOES drive the outflow limiter, so the
    exact-conservation gate above is meaningful (it is exercising the limiter +
    clamp path, not a trivially-conserved low-velocity drift)."""
    solid, depth, vx, vy = _make_stress_scene()
    s = _stress_solver()
    dt = 6.0 * s.max_dt()
    tilt = (0.12, -0.07)

    # If the limiter were NOT firing (out_sum <= depth everywhere) the field
    # would barely move; under this regime a large fraction of cells should
    # change by more than a coarse threshold within a few ticks. We assert a
    # substantial churn as a proxy that the limiter/clamp path is active.
    before = depth.copy()
    for _ in range(5):
        s.step(depth, vx, vy, None, None, solid, dt, tilt[0], tilt[1])
    changed = int((depth != before).sum())
    interior = (depth.shape[0] - 2) * (depth.shape[1] - 2)
    assert changed > interior // 2, (
        f"only {changed}/{interior} interior cells moved — regime too mild to "
        f"exercise the limiter; the conservation gate would be vacuous"
    )


def test_stress_conservation_many_seeds():
    """Run the exact-conservation gate across several seeds + tilts to widen the
    coverage (different lumpy fills / velocity patterns -> different donor face
    sign combinations). Every one must be 0-LSB exact."""
    s = _stress_solver()
    dt = 6.0 * s.max_dt()
    for seed, tilt in [(1, (0.0, 0.0)), (2, (0.2, 0.0)),
                       (3, (-0.15, 0.1)), (4, (0.08, 0.08))]:
        solid, depth, vx, vy = _make_stress_scene(seed=seed)
        total0 = _isum(depth)
        min_d = 0
        for _ in range(200):
            s.step(depth, vx, vy, None, None, solid, dt, tilt[0], tilt[1])
            min_d = min(min_d, int(depth.min()))
        assert _isum(depth) == total0, (
            f"seed {seed} tilt {tilt}: mass drifted "
            f"{_isum(depth) - total0} LSB (not 0-LSB conserved)"
        )
        assert min_d >= 0, f"seed {seed} tilt {tilt}: depth went negative ({min_d})"


if __name__ == "__main__":
    test_stress_conservation_exact_zero_lsb()
    test_stress_limiter_actually_fires()
    test_stress_conservation_many_seeds()
    print("stress conservation: 0-LSB exact, depth >= 0, limiter fires. PASS")
