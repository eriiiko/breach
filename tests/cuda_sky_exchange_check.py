"""CUDA sky-exchange lockstep — resident vs CPU bit-identity (gate e).

docs/sky_exchange_design_2026-07-24.md gate e. The sky-exchange pass runs
HOST-side on the numpy mirror, immediately after combustion, in BOTH the normal
:meth:`PhysicsRunner.step` and the GPU-resident :meth:`_step_resident` (design
finding, Erik-approved 2026-07-24: combustion is a host bracket on the mirror in
the resident tick, so the pass rides it — no device kernel). This gate proves a
full RESIDENT tick with the pass ACTIVE stays bit-identical to a full CPU tick:
the pass introduces no CPU↔GPU divergence.

Scenario: a planetside ambient map with sky_tau_s = 60 s (the pass live), a fire
seed (combustion vitiates local O2) AND a hand-depleted O2 patch (guarantees the
sky pass has real work → the gate is not vacuous). A/B over N ticks on two
independently built worlds — CPU (residency OFF) vs GPU-RESIDENT (residency ON +
all backends) — asserting byte-for-byte identity (tol 0) of every synced field
each tick, plus the vacuousness guards (sky_flux went non-zero; residency live).

Runs on a CUDA build (the cuda_*_check.py convention — the plain pytest suite
skips these without CUDA). Prints ``SKY_RESULT: PASS``/``FAIL``, exits 0/1.
"""
from __future__ import annotations

import sys

import numpy as np

import breach_physics as bp  # the CUDA build, imported first

FP_ONE = 65536
N_TICKS = 30
SKY_TAU_S = 60.0

_BACKENDS = (
    "set_temperature_backend", "set_water_backend", "set_smoke_backend",
    "set_fire_backend", "set_raycaster_backend",
    "set_bulk_flux_backend", "set_sl_advection_backend",
    "set_mg_solve_backend", "set_kick_compression_backend",
    "set_combustion_backend",
)

_FIELDS = ("atmosphere", "wave_p", "wind_x", "wind_y", "temperature", "heat",
           "fire", "wall_hp", "water_depth", "flow_vx", "flow_vy", "gas",
           "ripple", "ripple_v")


def _set_backends(on: bool) -> None:
    for name in _BACKENDS:
        getattr(bp, name)(bool(on))


def _residency(on: bool) -> None:
    from simulation import physics_runner
    physics_runner.set_residency(bool(on))


def _build_scenario():
    """A planetside ambient world with the sky pass LIVE (sky_tau_s=60), a fire
    seed, and a hand-depleted O2 patch (composition ≠ ambient → the sky pass has
    work every tick)."""
    from pathlib import Path

    from config import CFG
    from level_loader import LevelData
    from simulation import atmosphere_fixed, fire_fixed
    from simulation.ambient import derive_ambient
    from simulation.gamemap import GameMap
    from simulation.gases import INERT_N2, O2
    from simulation.physics_runner import PhysicsRunner

    H = W = 48
    tm = np.full((H, W), 9, dtype=np.int32)            # interior air
    tm[0, :] = tm[-1, :] = tm[:, 0] = tm[:, -1] = 0    # SPACE ring → is_ambient
    level = LevelData(name="sky_ab", version="1", path=Path("."),
                      tilemap=tm, tile_size_m=1.0 / 3.0, diffuse_path=Path("."),
                      boundary="ambient",
                      ambient=derive_ambient(sky_tau_s=SKY_TAU_S))
    g = GameMap(level)
    g.stamp_units([])
    assert g.is_ambient.any(), "ambient routing expected"
    assert g.sky_mask.any(), "interior sky mask must be non-empty"

    q = atmosphere_fixed.quantize_scalar
    # A fire seed (combustion vitiates local O2 → composition drifts below ambient).
    g.temperature[20:26, 20:26] += q(5000.0)
    g.fire[22:24, 22:24] = fire_fixed.quantize_scalar(0.8)
    # A hand-depleted O2 patch, N_total conserved (move O2 → inert): guarantees
    # the sky pass has real work from tick 1, independent of combustion timing.
    dep = (slice(30, 40), slice(8, 20))
    move = g.gas[O2][dep] // 2
    g.gas[O2][dep] -= move
    g.gas[INERT_N2][dep] += move

    runner = PhysicsRunner(bp)
    g.bind_physics_engine(runner.engine)
    dt = 1.0 / float(CFG.clock.ticks_per_second)
    return runner, g, dt


def _one_tick(runner, g, dt):
    g.stamp_units([])
    destroyed = runner.step(g, dt)
    for (yy, xx) in destroyed:
        g.destroy_wall(yy, xx)
    g.heat.fill(0)


def _compare(t, g_cpu, g_gpu):
    bad = 0
    for f in _FIELDS:
        a, b = getattr(g_cpu, f), getattr(g_gpu, f)
        if not np.array_equal(a, b):
            bad += 1
            mism = int(np.count_nonzero(a != b))
            print(f"  tick {t}: field {f}: {mism} mismatch(es)")
    return bad


def main() -> int:
    if not getattr(bp, "HAS_CUDA", False) or not bp.cuda_available():
        print("SKY_RESULT: FAIL (no CUDA build / device)")
        return 1
    try:
        import cupy  # noqa: F401
    except Exception as e:
        print(f"SKY_RESULT: FAIL (cupy not importable: {e!r})")
        return 1
    print("device:", bp.cuda_device_info())

    _residency(False); _set_backends(False)
    runner_cpu, g_cpu, dt = _build_scenario()
    runner_gpu, g_gpu, dt2 = _build_scenario()
    assert dt == dt2
    for f in _FIELDS:
        assert np.array_equal(getattr(g_cpu, f), getattr(g_gpu, f)), \
            f"scenario construction not deterministic on {f}"

    bad = 0
    sky_worked = 0
    for t in range(N_TICKS):
        _residency(False); _set_backends(False)
        _one_tick(runner_cpu, g_cpu, dt)
        _residency(True); _set_backends(True)
        _one_tick(runner_gpu, g_gpu, dt)
        _residency(False); _set_backends(False)
        bad += _compare(t, g_cpu, g_gpu)
        # vacuousness: the CPU runner's per-tick sky rail must fire (real work).
        if runner_cpu._sky_flux is not None and int(np.abs(runner_cpu._sky_flux).max()) > 0:
            sky_worked += 1
        if bad >= 8:
            print("  aborting after 8 divergences")
            break

    if sky_worked == 0:
        print("  the sky pass never did work (sky_flux stayed zero) — gate vacuous")
        return 1
    if not (bool(g_gpu.residency_on()) and hasattr(g_gpu, "_dev")):
        print("  the GPU world never entered residency mode — gate vacuous")
        return 1
    if bad != 0:
        print("SKY_RESULT: FAIL")
        return 1
    print(f"  {N_TICKS} ticks bit-identical (CPU vs resident) across all synced "
          f"fields; sky pass did work on {sky_worked}/{N_TICKS} ticks; "
          f"residency confirmed live.")
    print("SKY_RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
