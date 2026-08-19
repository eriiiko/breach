"""VELOCITY-CLAMP (P-V1) — property gates 1-2: diagonal-leak closure and cap
locality, exact via the P6.4 CPU reference (the state the cap derives from);
plus a loose full-engine smoke wiring check.

Design: docs/velocity_clamp_impl_2026-08-19.md (v3). Audit:
docs/velocity_clamp_audit_2026-08-19.md.

Gate 1 — diagonal leak closed (exact): construct winds with strong diagonal
components (each BELOW a constructed scalar cap, magnitude ABOVE it — the
audit's Defect 2 shape, the old component Chebyshev pre-test's hole) against
a constructed cap2 plane; assert every open cell's post-kick
wx^2+wy^2 <= (floor(sqrt(cap2)) + 2)^2 (D6's exact-rescale overshoot bound —
umag = floor(sqrt(rad)) >= |u| - 1, so the rescale can land up to ~2 raw
counts above cap). Squares computed in int64 (int32 squares wrap silently).

Gate 2 — cap locality (exact, same harness): hot cells (fire-range T) in one
region of a constructed state, blast-scale wind in a COOL region; the plane
is folded from that T with formula A (the eos_solver.cpp / cuda_eos_step.cu
scan, ported here in plain Python ints — c_amb2*ratio overflows numpy
int64) — assert the cool region's post-kick velocity obeys the AMBIENT cap
bound (gate 1's form): the remote hot cell must not raise it (the audit's
Defect 1 shape — the old global scalar cap let a hot cell anywhere raise
EVERY cell's ceiling).

Plus one full-engine smoke: a playground-style blast scenario driven through
the real engine, post-tick snapshot, assert wx^2+wy^2 <= cap2(T_snap)*1.5 for
every open cell — a loose e2e wiring check only (T moves after the fold
within the tick, hence the slack; the exact assertions live in gates 1-2).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))

import numpy as np

import breach_physics as bp  # noqa: E402

FP_ONE = 65536

CONSTS = dict(
    c_max=300.0, dx=1.0 / 3.0, adiabatic_index=1.4, absorb_strength=8.0,
    n_floor_solver=1e-3, t_min=-289.0, t_work_clamp=0.5,
    t_max_phys=16000.0, u_max=1000.0,
)


def _q(x):
    """Round-to-nearest Q16.16 (matches fixedpoint::quantize)."""
    x = np.asarray(x, dtype=np.float64)
    return np.int32(np.where(x >= 0, np.floor(x * FP_ONE + 0.5),
                             np.ceil(x * FP_ONE - 0.5)))


def _fold_cap2_plane_formula_a(temperature, solid, is_vacuum, ts,
                               s_eos_q, t_amb_q, c_amb_q, u_max_q):
    """VELOCITY-CLAMP formula A, ported verbatim (the eos_solver.cpp /
    cuda_eos_step.cu per-tick scan fold, D2v2/D4/D1) for gate 2's cap-
    locality property and the full-engine smoke. c_amb2*ratio reaches ~2^68
    and WRAPS SILENTLY in numpy int64 — the fold multiply runs in PLAIN
    PYTHON INTS (arbitrary precision); only the masking and the ratio_umax
    comparison are vectorized (ratio itself, bounded by ratio_umax ~7e5, is
    safely int64-representable).
    """
    c_amb2 = c_amb_q * c_amb_q
    u_max2 = u_max_q * u_max_q
    ru = u_max_q / c_amb_q
    ratio_umax = int(ru * ru * 65536.0) + 1

    t_abs = ((s_eos_q * temperature.astype(np.int64)) >> 16) + t_amb_q
    floor_mask = ts | (t_abs < t_amb_q)                 # D4 + D1
    t_abs_cap = np.where(floor_mask, t_amb_q, t_abs)
    ratio = (t_abs_cap.astype(np.int64) << 16) // t_amb_q   # int64-safe

    rails = solid | is_vacuum | (ratio >= ratio_umax)   # filler + U_MAX rail
    cap2 = np.full(temperature.shape, u_max2, dtype=np.int64)
    idx = np.nonzero(~rails)
    for pos, r in zip(zip(*idx), ratio[idx].tolist()):
        cap2[pos] = (c_amb2 * int(r)) >> 16
    return np.ascontiguousarray(cap2)


def _run_kick(wind_x, wind_y, temperature, cap2_plane, solid=None,
             is_vacuum=None, p_new=None, gas=None, gas_conservative=None,
             absorb=None, consts=CONSTS):
    """Drive the P6.4 CPU reference (the exact-state kick+compression tail)
    on constructed fields — the harness gates 1-2 share."""
    h, w = wind_x.shape
    if solid is None:
        solid = np.zeros((h, w), dtype=bool)
    if is_vacuum is None:
        is_vacuum = np.zeros((h, w), dtype=bool)
    if p_new is None:
        p_new = np.full((h, w), FP_ONE, dtype=np.int32)   # uniform -> zero grad(P)
    if gas is None:
        gas = np.zeros((3, h, w), dtype=np.int32)
        gas[0] = int(0.21 * FP_ONE)
        gas[1] = int(0.79 * FP_ONE)
    if gas_conservative is None:
        gas_conservative = np.array([True, True, False])
    if absorb is None:
        absorb = np.zeros((h, w), dtype=np.float32)
    wx = np.ascontiguousarray(wind_x.copy())
    wy = np.ascontiguousarray(wind_y.copy())
    t = np.ascontiguousarray(temperature.copy())
    res = bp.eos_kick_compression_ref(
        wx, wy, t, p_new, gas, gas_conservative, solid, is_vacuum, absorb,
        1.0 / 24.0, np.ascontiguousarray(cap2_plane, dtype=np.int64), **consts)
    return wx, wy, t, res


def _rad_bound(cap2_val):
    """D6's exact-rescale overshoot bound: |u| <= floor(sqrt(cap2)) + 2 raw
    counts (umag = floor(sqrt(rad)) >= |u| - 1, both directions of the
    rescale can land up to ~1 count high); squared, int64."""
    bound_mag = int(np.floor(np.sqrt(float(cap2_val)))) + 2
    return bound_mag * bound_mag


# ---------------------------------------------------------------------------
# Gate 1 — diagonal leak closed, exact.
# ---------------------------------------------------------------------------
def test_gate1_diagonal_leak_closed_exact():
    """Audit defect 2: components each < cap, magnitude > cap (the diagonal
    leak) — must be closed exactly by the exact rad > cap^2 test (no
    Chebyshev component pre-test) + D6's exact rescale."""
    h = w = 12
    for cap_val in (200.0, 550.0, 999.0):
        cap_q = int(_q(cap_val))
        cap2_val = cap_q * cap_q
        cap2 = np.full((h, w), cap2_val, dtype=np.int64)

        # Pure 45-degree diagonal at 0.75*cap per component: EACH component
        # individually BELOW cap, magnitude (0.75*sqrt(2) ~ 1.06x) ABOVE it
        # — the exact hole the old Chebyshev pre-test left open.
        comp = int(cap_q * 0.75)
        wind_x = np.full((h, w), comp, dtype=np.int32)
        wind_y = np.full((h, w), comp, dtype=np.int32)
        temperature = np.zeros((h, w), dtype=np.int32)
        wx, wy, t, res = _run_kick(wind_x, wind_y, temperature, cap2)

        assert res[2] > 0, (
            f"cap={cap_val}: the diagonal-leak forcer never engaged the clamp "
            "(gate is vacuous)")

        rad = wx.astype(np.int64) ** 2 + wy.astype(np.int64) ** 2
        bound_rad = _rad_bound(cap2_val)
        assert np.all(rad <= bound_rad), (
            f"cap={cap_val}: post-clamp |u|^2 exceeds (floor(sqrt(cap2))+2)^2 "
            f"— diagonal leak NOT closed (max rad={int(rad.max())} > {bound_rad})")

    # Random fuzz: many angles/magnitudes/caps, always squares in int64.
    rng = np.random.default_rng(20260819)
    for _ in range(40):
        cap_val = float(rng.uniform(50.0, 950.0))
        cap_q = int(_q(cap_val))
        cap2_val = cap_q * cap_q
        cap2 = np.full((h, w), cap2_val, dtype=np.int64)
        wind_x = _q((rng.random((h, w)) * 2 - 1) * 1400.0).astype(np.int32)
        wind_y = _q((rng.random((h, w)) * 2 - 1) * 1400.0).astype(np.int32)
        temperature = np.zeros((h, w), dtype=np.int32)
        wx, wy, t, res = _run_kick(wind_x, wind_y, temperature, cap2)
        rad = wx.astype(np.int64) ** 2 + wy.astype(np.int64) ** 2
        bound_rad = _rad_bound(cap2_val)
        assert np.all(rad <= bound_rad), (
            f"cap={cap_val}: fuzz found a post-clamp violation "
            f"(max rad={int(rad.max())} > {bound_rad})")


# ---------------------------------------------------------------------------
# Gate 2 — cap locality, exact.
# ---------------------------------------------------------------------------
def test_gate2_cap_locality_hot_cell_does_not_raise_a_remote_cap():
    """Audit defect 1: a hot cell anywhere used to raise EVERY cell's cap
    (the global-scalar bug). Fold the per-cell plane via formula A from a
    state with a fire-range hot region and assert a remote COOL region's
    kick still obeys the AMBIENT cap — the hot cell may not leak into it."""
    h = w = 20
    t_amb_k = 290.0
    s_eos_q = int(_q(1.0))       # S_EOS frozen at identity (design v2.2 D-A)
    t_amb_q = max(1, int(_q(t_amb_k)))
    c_amb_q = int(_q(CONSTS["c_max"]))
    u_max_q = int(_q(CONSTS["u_max"]))

    solid = np.zeros((h, w), dtype=bool)
    is_vacuum = np.zeros((h, w), dtype=bool)
    ts = solid   # no thermal_solid distinction in this constructed state

    temperature = np.zeros((h, w), dtype=np.int32)   # everywhere at ambient...
    HOT = (slice(2, 6), slice(2, 6))
    temperature[HOT] = int(_q(900.0))                # ...except a fire-range pocket

    cap2 = _fold_cap2_plane_formula_a(
        temperature, solid, is_vacuum, ts, s_eos_q, t_amb_q, c_amb_q, u_max_q)

    # The cool region's cap must be EXACTLY the ambient cap (c_amb^2) — the
    # hot cell at (2..6, 2..6) must not have raised it one raw count.
    COOL = (slice(12, 18), slice(12, 18))
    c_amb2 = c_amb_q * c_amb_q
    assert np.all(cap2[COOL] == c_amb2), (
        "a hot cell raised the COOL region's cap — the global-scalar leak "
        "(audit defect 1) is back")

    # Blast-scale diagonal wind EVERYWHERE (incl. the cool region) — engage
    # the clamp there and assert it obeys the ambient bound (gate 1's form),
    # unaffected by the remote hot cell.
    wind_x = np.full((h, w), int(400.0 * FP_ONE), dtype=np.int32)
    wind_y = np.full((h, w), int(400.0 * FP_ONE), dtype=np.int32)
    wx, wy, t, res = _run_kick(wind_x, wind_y, temperature, cap2)

    assert res[2] > 0, "the cap-locality forcer never engaged the clamp (vacuous gate)"
    rad_cool = wx[COOL].astype(np.int64) ** 2 + wy[COOL].astype(np.int64) ** 2
    bound_rad = _rad_bound(c_amb2)
    assert np.all(rad_cool <= bound_rad), (
        f"the cool region exceeded the ambient cap bound "
        f"(max rad={int(rad_cool.max())} > {bound_rad}) — a remote hot cell "
        f"leaked into a cold cell's ceiling")


# ---------------------------------------------------------------------------
# Gate 2 (cont.) — loose full-engine smoke.
# ---------------------------------------------------------------------------
def test_gate2_full_engine_smoke_playground_blast():
    """Loose e2e wiring check: a real engine tick on a playground-style
    blast scenario, post-tick snapshot, assert wx^2+wy^2 <= cap2(T_snap)*1.5
    for every open cell (T moves after the fold within the tick, hence the
    slack — the exact assertions live in the two gates above)."""
    from config import CFG
    from level_loader import LevelData
    from simulation import atmosphere_fixed
    from simulation.gamemap import GameMap
    from simulation.gases import O2
    from simulation.physics_runner import PhysicsRunner

    h = w = 32
    tm = np.zeros((h, w), dtype=np.int32)
    tm[1:31, 1:31] = 1
    tm[2:30, 2:30] = 4
    level = LevelData(name="velocity_clamp_smoke", version="1", path=Path("."),
                      tilemap=tm, tile_size_m=1.0 / 3.0,
                      diffuse_path=Path("."))
    g = GameMap(level)
    g.stamp_units([])

    q = atmosphere_fixed.quantize_scalar
    g.temperature[10:16, 10:16] += q(4000.0)
    g.gas[O2, 11:14, 11:14] += q(4.0)

    runner = PhysicsRunner(bp)
    runner.eos.dx = float(g.tile_size_m)
    eos = runner.engine.eos
    inert_n2_idx = int(g.gases.name_to_id["inert_n2"])
    dt = 1.0 / float(CFG.clock.ticks_per_second)

    t_snap = np.ascontiguousarray(g.temperature.copy())   # tick-entry T
    for _ in range(5):
        runner.engine.run_substeps(
            g.wave_p, g.atmosphere, g.wind_x, g.wind_y, g.temperature,
            g.obstacles, g.solid, g.is_vacuum, g.dyn_permeability,
            g.dyn_wave_absorb, g.gas, g.gases.diffusion, g.gases.conservative,
            g.gases.decay, inert_n2_idx, dt)
        t_snap = np.ascontiguousarray(g.temperature.copy())

    s_eos_q = int(_q(float(eos.S_EOS)))
    t_amb_q = max(1, int(_q(float(eos.T_AMB_K))))
    c_amb_q = int(_q(float(eos.c_max)))
    u_max_q = int(_q(float(eos.U_MAX)))
    ts = g.solid   # no thermal_solid plane on this scenario
    cap2 = _fold_cap2_plane_formula_a(
        t_snap, g.solid, g.is_vacuum, ts, s_eos_q, t_amb_q, c_amb_q, u_max_q)

    open_mask = ~(g.solid | g.is_vacuum)
    rad = g.wind_x.astype(np.int64) ** 2 + g.wind_y.astype(np.int64) ** 2
    loose_bound = cap2.astype(np.float64) * 1.5
    viol = open_mask & (rad.astype(np.float64) > loose_bound)
    assert not np.any(viol), (
        f"{int(viol.sum())} open cells exceed cap2(T_snap)*1.5 — the "
        f"per-cell clamp wiring looks broken end to end "
        f"(max rad={int(rad[open_mask].max())}, "
        f"worst allowed={float(loose_bound[open_mask].max()):.3e})")
