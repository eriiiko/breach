"""Semi-Lagrangian smoke advection — sanity tests (smoke v2, step 1).

The smoke advection was changed from a central-difference stencil (which
checkerboards and can oscillate near strong wind) to an unconditionally
stable semi-Lagrangian back-trace with permeability-aware bilinear sampling
(see docs/architecture/engine/05_smoke.md §2).

These tests exercise ``SmokeDynamics.step`` directly on synthetic fields so
the advection is isolated from the full atmosphere/fire pipeline:

1. Constant wind translates the cloud in the wind direction.
2. Values stay in [0, 1] with no NaN/blow-up over many steps.
3. In a closed (walled) domain total smoke does not grow — semi-Lagrangian
   is slightly diffusive, so we only assert "no growth" within tolerance.
4. A single off cell does NOT explode into a checkerboard over many steps.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_smoke_semilagrangian.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))

import breach_physics as bp


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _make_solver(advection_rate=1.0, d_smoke=0.0,
                 wind_diffusion_scale=0.0):
    """A solver with diffusion off by default so advection is isolated.

    Patch 2b: dt_scale is gone — smoke moves on the real dt. The back-trace
    distance per step is now
        dt_adv = advection_rate * dt
    so a constant wind w moves the field by w * dt_adv each step.
    """
    s = bp.SmokeDynamics()
    s.d_smoke = d_smoke
    s.advection_rate = advection_rate
    s.wind_diffusion_scale = wind_diffusion_scale
    return s


def _open_domain(h, w):
    """All-air fields: nothing solid, nothing vacuum, fully permeable."""
    obstacles = np.zeros((h, w), dtype=bool)
    is_wall = np.zeros((h, w), dtype=bool)
    is_vacuum = np.zeros((h, w), dtype=bool)
    perm = np.ones((h, w), dtype=np.float32)
    return obstacles, is_wall, is_vacuum, perm


def _walled_box(h, w):
    """Air interior with a one-cell solid wall border (closed domain)."""
    obstacles = np.zeros((h, w), dtype=bool)
    obstacles[0, :] = obstacles[-1, :] = True
    obstacles[:, 0] = obstacles[:, -1] = True
    is_wall = obstacles.copy()
    is_vacuum = np.zeros((h, w), dtype=bool)
    perm = np.ones((h, w), dtype=np.float32)
    perm[obstacles] = 0.0
    return obstacles, is_wall, is_vacuum, perm


def _step(s, smoke, wind_x, wind_y, obstacles, is_wall, is_vacuum, perm, dt):
    """Call ``SmokeDynamics.step`` (Patch 2b: WIND-ONLY).

    The breach sink-pull is no longer fused into the advection back-trace (it
    is the standalone ``sink_hop``), so ``step`` takes no sink field — these S1
    advection tests have no breach anyway, so this is exactly the plain
    semi-Lagrangian wind advection they assert.
    """
    s.step(smoke, wind_x, wind_y,
           obstacles, is_wall, is_vacuum, perm, dt)


def _center_of_mass(field):
    total = field.sum()
    assert total > 0
    ys, xs = np.mgrid[0:field.shape[0], 0:field.shape[1]]
    return (ys * field).sum() / total, (xs * field).sum() / total


# --------------------------------------------------------------------------
# 1. Translation
# --------------------------------------------------------------------------
def test_constant_wind_translates_in_wind_direction():
    """A blob under constant rightward wind moves right (and only right)."""
    h, w = 24, 48
    smoke = np.zeros((h, w), dtype=np.float32)
    smoke[10:14, 8:12] = 1.0  # blob on the left

    obstacles, is_wall, is_vacuum, perm = _open_domain(h, w)
    wind_x = np.full((h, w), 1.0, dtype=np.float32)  # rightward (+x)
    wind_y = np.zeros((h, w), dtype=np.float32)

    s = _make_solver(advection_rate=1.0)  # dt_adv = dt per step
    dt = 0.5

    cy0, cx0 = _center_of_mass(smoke)
    for _ in range(20):
        _step(s, smoke, wind_x, wind_y, obstacles, is_wall, is_vacuum, perm, dt)
    cy1, cx1 = _center_of_mass(smoke)

    assert cx1 > cx0 + 3.0, f"cloud did not move right enough: {cx0:.2f} -> {cx1:.2f}"
    assert abs(cy1 - cy0) < 0.75, f"cloud drifted in y unexpectedly: {cy0:.2f} -> {cy1:.2f}"


def test_translation_speed_matches_wind_times_dt_adv():
    """Center of mass advances ~ wind * advection_rate * dt per step.

    Sanity-checks that the back-trace distance preserves the intended units
    (we did not silently change the feel of the advection knobs).
    """
    h, w = 16, 64
    smoke = np.zeros((h, w), dtype=np.float32)
    smoke[6:10, 6:10] = 1.0

    obstacles, is_wall, is_vacuum, perm = _open_domain(h, w)
    wind_x = np.full((h, w), 2.0, dtype=np.float32)
    wind_y = np.zeros((h, w), dtype=np.float32)

    advection_rate = 1.0
    dt = 0.5
    s = _make_solver(advection_rate=advection_rate)

    n_steps = 10
    cy0, cx0 = _center_of_mass(smoke)
    for _ in range(n_steps):
        _step(s, smoke, wind_x, wind_y, obstacles, is_wall, is_vacuum, perm, dt)
    cy1, cx1 = _center_of_mass(smoke)

    expected = wind_x[0, 0] * advection_rate * dt * n_steps  # = 2*1*0.5*10 = 10
    moved = cx1 - cx0
    # Allow generous tolerance: numerical diffusion smears the blob and the
    # leading edge piles against the (far) wall is avoided by the 64-wide domain.
    assert abs(moved - expected) < 0.25 * expected, \
        f"moved {moved:.2f}, expected ~{expected:.2f}"


# --------------------------------------------------------------------------
# 2. Bounded / no NaN
# --------------------------------------------------------------------------
def test_values_stay_in_range_and_finite():
    """Strong, swirling wind over many steps: stays in [0,1], no NaN/Inf."""
    h, w = 32, 32
    rng = np.random.default_rng(0)
    smoke = rng.random((h, w), dtype=np.float32)

    obstacles, is_wall, is_vacuum, perm = _open_domain(h, w)
    # A large rotational wind field — the regime where the old central
    # difference would oscillate.
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = h / 2, w / 2
    wind_x = -(ys - cy).astype(np.float32) * 0.5
    wind_y = (xs - cx).astype(np.float32) * 0.5

    s = _make_solver(advection_rate=5.0, d_smoke=0.05, wind_diffusion_scale=10.0)
    dt = 0.3

    for _ in range(200):
        _step(s, smoke, wind_x, wind_y, obstacles, is_wall, is_vacuum, perm, dt)
        assert np.all(np.isfinite(smoke)), "NaN/Inf appeared in smoke field"
        assert smoke.min() >= 0.0 and smoke.max() <= 1.0, \
            f"smoke escaped [0,1]: min={smoke.min()}, max={smoke.max()}"


# --------------------------------------------------------------------------
# 3. Closed-domain conservation (no growth)
# --------------------------------------------------------------------------
def test_closed_domain_no_mass_growth():
    """In a sealed box, advection must NOT create smoke.

    Semi-Lagrangian is not strictly conservative (mildly diffusive), so the
    task asks only that total smoke does not *grow*. We check two regimes:

    * zero wind  -> the field must be left exactly unchanged (identity), the
      strongest possible no-growth statement; and
    * a gentle interior wind (displacement well under a cell, kept away from
      slamming the blob into a wall) -> total is conserved to within float
      noise and certainly does not grow.

    (Driving a blob hard into a wall for many steps legitimately *loses* mass:
    the sealed boundary is a sink — smoke piles against it and is clamped. That
    is correct behaviour, not a conservation violation, so we do not test it
    here as if mass should survive.)
    """
    h, w = 28, 28
    obstacles, is_wall, is_vacuum, perm = _walled_box(h, w)

    # (a) Zero wind: the advection step must be the identity.
    smoke = np.zeros((h, w), dtype=np.float32)
    smoke[4:10, 4:10] = 0.8
    base = smoke.copy()
    zero = np.zeros((h, w), dtype=np.float32)
    s0 = _make_solver(advection_rate=2.0, d_smoke=0.0)  # diffusion off
    for _ in range(50):
        _step(s0, smoke, zero, zero, obstacles, is_wall, is_vacuum, perm, 0.4)
    assert np.allclose(smoke, base, atol=1e-6), \
        "zero-wind advection changed the field (should be identity)"

    # (b) Gentle interior wind, small displacement, kept off the walls.
    smoke = np.zeros((h, w), dtype=np.float32)
    smoke[12:16, 8:12] = 0.8  # centred blob
    wind_x = np.full((h, w), 0.2, dtype=np.float32)
    wind_y = np.zeros((h, w), dtype=np.float32)
    s = _make_solver(advection_rate=1.0, d_smoke=0.0)  # dt_adv = 0.3 -> ~0.06/step
    dt = 0.3

    total0 = float(smoke.sum())
    for _ in range(40):
        _step(s, smoke, wind_x, wind_y, obstacles, is_wall, is_vacuum, perm, dt)
    total1 = float(smoke.sum())

    # No growth (allow a tiny epsilon for float accumulation).
    assert total1 <= total0 + 1e-3, \
        f"closed-domain smoke GREW: {total0:.4f} -> {total1:.4f}"
    # Transport happened but mass is essentially preserved (not a vanishing).
    assert total1 > 0.9 * total0, \
        f"gentle interior advection lost too much mass: {total0:.4f} -> {total1:.4f}"

    # Smoke must never have leaked into the walls.
    assert np.all(smoke[is_wall] == 0.0), "smoke leaked into wall cells"


def test_no_smoke_pulled_through_wall():
    """Smoke on one side of a wall must not teleport across it via back-trace.

    A vertical wall splits the domain. Smoke sits on the right of the wall;
    wind points left (toward the wall). The back-trace from the left-side
    cells must NOT sample through the wall and pull the right-side smoke over.
    """
    h, w = 20, 20
    obstacles = np.zeros((h, w), dtype=bool)
    wall_x = 10
    obstacles[:, wall_x] = True
    is_wall = obstacles.copy()
    is_vacuum = np.zeros((h, w), dtype=bool)
    perm = np.ones((h, w), dtype=np.float32)
    perm[obstacles] = 0.0

    smoke = np.zeros((h, w), dtype=np.float32)
    smoke[:, wall_x + 1:] = 0.7  # all smoke strictly right of the wall

    # Wind pointing LEFT everywhere — would drag smoke across the wall if the
    # bilinear sample ignored permeability.
    wind_x = np.full((h, w), -2.0, dtype=np.float32)
    wind_y = np.zeros((h, w), dtype=np.float32)

    s = _make_solver(advection_rate=2.0)
    dt = 0.5

    for _ in range(40):
        _step(s, smoke, wind_x, wind_y, obstacles, is_wall, is_vacuum, perm, dt)

    # Left of the wall must remain (essentially) clear.
    left = smoke[:, :wall_x]
    assert left.max() < 1e-4, \
        f"smoke teleported through the wall: max left-of-wall = {left.max()}"


# --------------------------------------------------------------------------
# 4. No checkerboard from a single off cell
# --------------------------------------------------------------------------
def test_single_cell_does_not_checkerboard():
    """A lone dense cell under wind must not grow an odd/even checkerboard.

    The old central-difference advection developed alternating-sign artifacts
    in high-wind regions. Semi-Lagrangian is monotone-ish (it only averages
    existing values), so values stay bounded by the source field's range and
    no negative/oscillating pattern appears. We measure the high-frequency
    (checkerboard) energy and assert it stays tiny.
    """
    h, w = 40, 40
    smoke = np.zeros((h, w), dtype=np.float32)
    smoke[20, 20] = 1.0  # single off cell

    obstacles, is_wall, is_vacuum, perm = _open_domain(h, w)
    # Strong wind — the destabilising regime for the old scheme.
    wind_x = np.full((h, w), 1.0, dtype=np.float32)
    wind_y = np.full((h, w), 0.7, dtype=np.float32)

    s = _make_solver(advection_rate=4.0, d_smoke=0.0)  # diffusion OFF: harshest test
    dt = 0.5

    for _ in range(60):
        _step(s, smoke, wind_x, wind_y, obstacles, is_wall, is_vacuum, perm, dt)

    # Bounded, non-negative (semi-Lagrangian only averages existing values, so
    # it can never exceed the source range or go negative).
    assert np.all(np.isfinite(smoke))
    assert smoke.min() >= -1e-6, f"negative smoke (overshoot): {smoke.min()}"
    assert smoke.max() <= 1.0 + 1e-6, f"smoke overshoot above source: {smoke.max()}"

    # Checkerboard detector: convolve with the (+ - / - +) Nyquist stencil.
    # A checkerboard would put large energy here; a smooth advected blob
    # should leave the high-frequency mode tiny relative to the total.
    cb = np.zeros_like(smoke)
    cb[1:-1, 1:-1] = (
        4.0 * smoke[1:-1, 1:-1]
        - smoke[:-2, 1:-1] - smoke[2:, 1:-1]
        - smoke[1:-1, :-2] - smoke[1:-1, 2:]
    )
    # For a genuine checkerboard, |cb| ~ 8*amplitude per cell; for a smooth
    # field it is the (small) Laplacian. Compare its energy to the field's.
    cb_energy = float(np.abs(cb).sum())
    field_energy = float(smoke.sum())
    ratio = cb_energy / max(field_energy, 1e-6)
    assert ratio < 2.0, f"checkerboard energy too high (ratio={ratio:.3f})"


if __name__ == "__main__":
    test_constant_wind_translates_in_wind_direction()
    test_translation_speed_matches_wind_times_dt_adv()
    test_values_stay_in_range_and_finite()
    test_closed_domain_no_mass_growth()
    test_no_smoke_pulled_through_wall()
    test_single_cell_does_not_checkerboard()
    print("OK: all semi-Lagrangian smoke advection sanity tests passed")
