"""B4 — CUDA-vs-CPU lockstep on a planetside AMBIENT map (runs in the GPU
subprocess). Proves the .cu mirror of the BC arc (B3a/b/c) is bit-identical
(tol 0) to the CPU reference: the shift trick, the ring→Dirichlet excl + the
σ-diagonal (via the shared host mg_build_levels), the per-substep ring reset +
the int64 boundary_flux rail, the u/T widenings, the u-damping band, and the
step-5 add-back — plus the temperature Pass-0 ambient wipe and the joins-ambient
structural edit.

A full ``PhysicsRunner.step`` tick is run TWICE on two independently built,
identically seeded ambient worlds: once with every GPU backend flag OFF (the CPU
reference) and once with the four EOS kernel flags + the temperature flag ON
(the GPU chain — dispatch proven via eos_step_cuda_calls). Per tick, asserts
bit-identity of every EOS-owned field, all six solver digests, the five rail
counters, AND the new per-plane boundary_flux rail. A scripted destroy_wall on a
ring-adjacent hull tile mid-run exercises the joins-ambient twin + the rail.

Prints ``AMBIENT_RESULT: PASS``/``FAIL`` and exits 0/1.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import breach_physics as bp  # the CUDA build (subprocess sys.path[0] == build_cuda)

FP_ONE = 65536

_EOS_SETTERS = ("set_sl_advection_backend", "set_bulk_flux_backend",
                "set_mg_solve_backend", "set_kick_compression_backend")


def _set_backends(on: bool) -> None:
    """Flip the four EOS kernel flags AND the temperature flag together, so the
    GPU run exercises the whole ambient GPU surface (EOS shift/reset/widenings/
    u-damping/rail + the temperature Pass-0 wipe)."""
    for name in _EOS_SETTERS:
        getattr(bp, name)(bool(on))
    bp.set_temperature_backend(bool(on))


# Ring-adjacent hull stub we breach mid-run (joins-ambient twin coverage).
_STUB_ROW = 1
_STUB_COLS = range(28, 32)
_BREACH_TICK = 25


def _build_scenario():
    """An independently constructed ambient world: a SPACE ring border (v1 code
    0 -> is_ambient) around an open-air interior (code 9), with a short hull stub
    (code 1) at row 1 adjacent to the ring (the destroy_wall target). The open
    interior gives the u-damping band real cells; a hot-core + O2 detonation
    drives flow into the band and across the ring."""
    from config import CFG
    from level_loader import LevelData
    from simulation import atmosphere_fixed
    from simulation.gamemap import GameMap
    from simulation.gases import O2
    from simulation.physics_runner import PhysicsRunner

    H = W = 64
    tm = np.full((H, W), 9, dtype=np.int32)          # interior air
    tm[0, :] = tm[-1, :] = tm[:, 0] = tm[:, -1] = 0   # SPACE ring border
    for c in _STUB_COLS:
        tm[_STUB_ROW, c] = 1                          # ring-adjacent hull stub
    level = LevelData(name="bc_ambient_lockstep", version="1", path=Path("."),
                      tilemap=tm, tile_size_m=1.0 / 3.0,
                      diffuse_path=Path("."), boundary="ambient")
    g = GameMap(level)
    g.stamp_units([])
    assert g.is_ambient.any() and not g.is_vacuum.any(), \
        "ambient map: wholesale SPACE->is_ambient routing expected"
    assert g.sponge_udamp.any(), \
        "u-damping band must be active (default k_max) for a meaningful gate"

    # Detonation: a hot core (outward shock) + an O2 overpressure pocket
    # (density spike venting toward the ring). Placed off-centre so the front
    # reaches the near ring within a few ticks.
    q = atmosphere_fixed.quantize_scalar
    g.temperature[16:26, 16:26] += q(5000.0)
    g.gas[O2, 18:24, 18:24] += q(4.0)

    runner = PhysicsRunner(bp)
    runner.eos.dx = float(g.tile_size_m)
    dt = 1.0 / float(CFG.clock.ticks_per_second)
    return runner, g, dt


_DIGESTS = ("digest_advect", "digest_bulk_flux", "digest_pstar",
            "digest_helmholtz", "digest_velocity", "digest_compression")
_COUNTERS = ("u_clamp_hits", "u_max_hits", "work_clamp_hits",
             "energy_floor_hits", "t_max_phys_hits")
_FIELDS = ("wave_p", "atmosphere", "wind_x", "wind_y", "temperature", "gas")


def run_lockstep() -> bool:
    print("AMBIENT lockstep — full runner.step tick, CPU (flags off) vs GPU "
          "(4 EOS + temperature flags on), per-tick fields + digests + rails:")
    _set_backends(False)
    runner_cpu, g_cpu, dt = _build_scenario()
    runner_gpu, g_gpu, dt2 = _build_scenario()
    assert dt == dt2
    eos_cpu = runner_cpu.engine.eos
    eos_gpu = runner_gpu.engine.eos

    for f in _FIELDS:
        assert np.array_equal(getattr(g_cpu, f), getattr(g_gpu, f)), \
            f"scenario construction not deterministic on {f}"
    # Prove the composite flag actually flips the dispatch predicate on.
    _set_backends(True)
    assert bp.get_eos_step_backend(), "EOS flags on but predicate False"
    assert bp.get_temperature_backend(), "temperature flag on but predicate False"
    _set_backends(False)

    n_ticks = 60
    bad = 0
    calls0 = int(bp.eos_step_cuda_calls())
    rail_seen_nonzero = False
    breached = False

    for tick in range(n_ticks):
        # Scripted structural edit (identical on both worlds): breach the
        # ring-adjacent hull stub -> joins-ambient twin + rail vent.
        if tick == _BREACH_TICK:
            for c in _STUB_COLS:
                g_cpu.destroy_wall(_STUB_ROW, c)
                g_gpu.destroy_wall(_STUB_ROW, c)
            breached = True

        _set_backends(False)
        runner_cpu.step(g_cpu, dt)
        _set_backends(True)
        runner_gpu.step(g_gpu, dt)
        _set_backends(False)

        for f in _FIELDS:
            a, b = getattr(g_cpu, f), getattr(g_gpu, f)
            if not np.array_equal(a, b):
                bad += 1
                print(f"  tick {tick}: field {f}: "
                      f"{int(np.count_nonzero(a != b))} MISMATCH(es)")
        for d in _DIGESTS:
            dc, dg = int(getattr(eos_cpu, d)), int(getattr(eos_gpu, d))
            if dc != dg:
                bad += 1
                print(f"  tick {tick}: {d} mismatch "
                      f"(cpu={dc:#018x} gpu={dg:#018x})")
        for c in _COUNTERS:
            cc, cg = int(getattr(eos_cpu, c)), int(getattr(eos_gpu, c))
            if cc != cg:
                bad += 1
                print(f"  tick {tick}: counter {c} mismatch (cpu={cc} gpu={cg})")
        # The BC rail (spec §5): per-plane int64 boundary_flux — byte-identical.
        rc, rg = list(eos_cpu.boundary_flux()), list(eos_gpu.boundary_flux())
        if rc != rg:
            bad += 1
            print(f"  tick {tick}: boundary_flux mismatch cpu={rc} gpu={rg}")
        if any(v != 0 for v in rg):
            rail_seen_nonzero = True
        if bad >= 10:
            print("  aborting after 10 divergences")
            break

    ok = (bad == 0)
    calls = int(bp.eos_step_cuda_calls()) - calls0
    if calls != n_ticks and bad < 10:
        ok = False
        print(f"  dispatch fired {calls}/{n_ticks} GPU ticks — not on the chain")
    if not breached:
        ok = False
        print("  structural edit never fired")
    if not rail_seen_nonzero:
        ok = False
        print("  boundary_flux rail never went non-zero — the ring exchange "
              "was not exercised (scenario too tame)")
    # The ring must stay pinned to the effective pin on BOTH paths.
    pin = g_cpu._ambient.pin_q
    if not (np.all(g_cpu.atmosphere[g_cpu.is_ambient] == pin)
            and np.all(g_gpu.atmosphere[g_gpu.is_ambient] == pin)):
        ok = False
        print("  ring did not materialize the effective pin on both paths")

    print(f"  divergences={bad}  dispatch={calls}/{n_ticks}  "
          f"rail_nonzero={rail_seen_nonzero}  breached={breached}")
    return ok


def main() -> int:
    ok = run_lockstep()
    print(f"AMBIENT_RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
