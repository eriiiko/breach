"""EOS P6.5 — chained full-eos.step engine-dispatch bit-identity check (runs
inside the GPU subprocess).

THE gate that proves the CHAINED dispatch — not just each kernel in isolation
— is bit-identical: P6.1–P6.4 each re-proved one kernel surface against a
replayed CPU reference; P6.5 wires them into one per-tick GPU orchestration
(cuda_eos_step.cu: device-resident substep loop chaining SL advection + bulk
flux, then the MG solve, the kick + compression, and the step-5 P
materialization) dispatched from PhysicsEngine::run_substeps when all four
EOS kernel-surface backend flags are on.

Two gates:

  PART 1 — CHAINED TRAJECTORY (the review's §4 P6.5 row): a breach-to-vacuum
  + blast scenario (hot core + O2 overpressure + a trace cloud in a
  hull-ringed room breached to vacuum — sustained sonic venting pins n_sub at
  N_SUB_MAX, steep N gradients at the breach, warm-start reuse, deep
  V-cycles) driven through the REAL engine path (run_substeps) for 120 ticks
  TWICE, on two independently built GameMaps/runners: once with the flags OFF
  (the CPU reference trajectory) and once with all four flags ON (every tick
  dispatches the GPU chain — proven via eos_step_cuda_calls, so a
  silently-CPU run cannot pass vacuously). Per tick, asserts bit-identity of
    * ALL EOS-owned fields: wind_x / wind_y / temperature / atmosphere /
      p_prev (wave_p) / every gas plane (bulk AND traces — the once-per-tick
      trace advection rides the GPU-corrected wind);
    * ALL SIX solver digests (advect, bulk_flux, pstar, helmholtz, velocity,
      compression);
    * the five cumulative rail counters (u_clamp / u_max / work_clamp /
      energy_floor / t_max_phys);
    * the schedule telemetry (dbg_last_n_sub, dbg_last_c_local_q).
  Also measures per-tick wall cost CPU vs GPU (informational — the review is
  explicit the per-call port will NOT beat the CPU at this size).

  PART 2 — the CUDA build's CPU path (flags off) still reproduces the
  committed default-scenario golden (the s4a-check idiom; proves the P6.5
  dispatch code is strictly additive — no CPU trajectory changed).

Prints ``P65_RESULT: PASS``/``FAIL`` and exits 0/1.
"""
from __future__ import annotations

import sys
import time

import numpy as np

# Import the CUDA build FIRST so it is the cached `breach_physics` before
# anything else (field_ab_harness inserts cpp/build/Release) imports it.
import breach_physics as bp

FP_ONE = 65536

# The four EOS kernel-surface flags the P6.5 dispatch ANDs.
_EOS_SETTERS = ("set_sl_advection_backend", "set_bulk_flux_backend",
                "set_mg_solve_backend", "set_kick_compression_backend")


def _set_eos_backends(on: bool) -> None:
    for name in _EOS_SETTERS:
        getattr(bp, name)(bool(on))


def _build_scenario():
    """One independently constructed runner + map (the P6.3 blast+vent
    scenario, + a trace cloud so the once-per-tick trace advection is
    exercised on a non-zero plane)."""
    from pathlib import Path

    from config import CFG
    from level_loader import LevelData
    from simulation import atmosphere_fixed
    from simulation.gamemap import GameMap
    from simulation.gases import O2
    from simulation.physics_runner import PhysicsRunner

    H = W = 96
    # v1 tilemap vocabulary: 0 = outer space (vacuum), 1 = hull wall,
    # 4 = interior air. A vacuum band, a hull ring, interior air, and a
    # 4-tile breach carved through the east hull — sustained venting into
    # vacuum (96x96 -> an 8-level pyramid, fused-tail entry at level 2).
    tm = np.zeros((H, W), dtype=np.int32)
    tm[2:94, 2:94] = 1
    tm[3:93, 3:93] = 4
    tm[46:50, 93] = 4          # the breach: hull ring opened to the vacuum band
    level = LevelData(name="eos_p65_blast_vent", version="1", path=Path("."),
                      tilemap=tm, tile_size_m=1.0 / 3.0,   # ship tile scale
                      diffuse_path=Path("."))
    g = GameMap(level)
    g.stamp_units([])
    assert g.is_vacuum.any(), "scenario must have vacuum to vent into"

    # THE BLAST: a hot core (raises p* hard -> outward shock) + an O2
    # overpressure pocket (density spike venting toward the breach) + a trace
    # cloud (rides the once-per-tick trace advection on the corrected wind —
    # CPU in both runs, so it transitively asserts the GPU wind bytes).
    q = atmosphere_fixed.quantize_scalar
    g.temperature[20:32, 20:32] += q(5000.0)
    g.gas[O2, 22:28, 22:28] += q(4.0)
    trace_ids = [gi for gi in range(g.gas.shape[0])
                 if not bool(g.gases.conservative[gi])]
    assert trace_ids, "scenario needs a trace plane"
    g.gas[trace_ids[0], 40:60, 40:60] += q(0.5)

    runner = PhysicsRunner(bp)
    runner.eos.dx = float(g.tile_size_m)
    inert_n2_idx = int(g.gases.name_to_id["inert_n2"])
    dt = 1.0 / float(CFG.clock.ticks_per_second)
    return runner, g, inert_n2_idx, dt


def _tick(runner, g, inert_n2_idx, dt):
    runner.engine.run_substeps(
        g.wave_p, g.atmosphere,
        g.wind_x, g.wind_y,
        g.temperature, g.gas_energy,   # arc #54 §2.2 (MECHANICAL)
        g.obstacles, g.solid, g.is_vacuum,
        g.dyn_permeability, g.dyn_wave_absorb,
        g.gas, g.gases.diffusion, g.gases.conservative,
        g.gases.decay, inert_n2_idx,
        dt,
    )


_DIGESTS = ("digest_advect", "digest_bulk_flux", "digest_pstar",
            "digest_helmholtz", "digest_velocity", "digest_compression")
_COUNTERS = ("u_clamp_hits", "u_max_hits", "work_clamp_hits",
             "energy_floor_hits", "t_max_phys_hits")
_FIELDS = ("wave_p", "atmosphere", "wind_x", "wind_y", "temperature", "gas")


def part1_chained_trajectory() -> bool:
    print("PART 1 — chained dispatch, breach+blast trajectory, CPU run vs "
          "GPU run (per-tick fields + six digests + rail counters):")
    # Two independent, identically seeded worlds. Flags are process-global,
    # so they are flipped around each tick (the s2b A/B idiom).
    _set_eos_backends(False)
    runner_cpu, g_cpu, n2_cpu, dt = _build_scenario()
    runner_gpu, g_gpu, n2_gpu, dt2 = _build_scenario()
    assert dt == dt2
    eos_cpu = runner_cpu.engine.eos
    eos_gpu = runner_gpu.engine.eos

    # The two worlds must START identical (same construction).
    for f in _FIELDS:
        assert np.array_equal(getattr(g_cpu, f), getattr(g_gpu, f)), \
            f"scenario construction not deterministic on {f}"

    _set_eos_backends(True)
    assert bp.get_eos_step_backend(), \
        "all four EOS flags on but get_eos_step_backend() is False"
    _set_eos_backends(False)

    n_ticks = 120
    bad = 0
    calls0 = int(bp.eos_step_cuda_calls())
    t_cpu = 0.0
    t_gpu = 0.0
    max_pstar = 0
    max_n_sub = 0
    for tick in range(n_ticks):
        # -- CPU reference tick (flags off) --------------------------------
        _set_eos_backends(False)
        t0 = time.perf_counter()
        _tick(runner_cpu, g_cpu, n2_cpu, dt)
        t_cpu += time.perf_counter() - t0

        # scenario-hardness telemetry (CPU side owns the dbg caches)
        ps, _dv, _nt = eos_cpu.dbg_mg_inputs()
        max_pstar = max(max_pstar, int(ps.max()))
        max_n_sub = max(max_n_sub, int(eos_cpu.dbg_last_n_sub))

        # -- GPU chained tick (all four flags on) --------------------------
        _set_eos_backends(True)
        t0 = time.perf_counter()
        _tick(runner_gpu, g_gpu, n2_gpu, dt)
        t_gpu += time.perf_counter() - t0
        _set_eos_backends(False)

        # -- per-tick bit-identity -----------------------------------------
        for f in _FIELDS:
            a, b = getattr(g_cpu, f), getattr(g_gpu, f)
            if not np.array_equal(a, b):
                bad += 1
                mism = int(np.count_nonzero(a != b))
                print(f"  tick {tick}: field {f}: {mism} MISMATCH(es)")
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
                print(f"  tick {tick}: counter {c} mismatch "
                      f"(cpu={cc} gpu={cg})")
        if int(eos_cpu.dbg_last_n_sub) != int(eos_gpu.dbg_last_n_sub):
            bad += 1
            print(f"  tick {tick}: n_sub mismatch "
                  f"(cpu={eos_cpu.dbg_last_n_sub} gpu={eos_gpu.dbg_last_n_sub})")
        if int(eos_cpu.dbg_last_c_local_q) != int(eos_gpu.dbg_last_c_local_q):
            bad += 1
            print(f"  tick {tick}: c_local_q mismatch")
        if bad >= 10:
            print("  aborting after 10 divergences")
            break

    ok = (bad == 0)

    # The dispatch must have FIRED every GPU tick (a silently-CPU "GPU run"
    # would make the whole gate vacuous).
    calls = int(bp.eos_step_cuda_calls()) - calls0
    if calls != n_ticks and bad < 10:
        ok = False
        print(f"  dispatch fired {calls}/{n_ticks} GPU ticks — "
              f"the GPU run was not actually on the GPU chain")

    # The scenario must actually stress the chain (hardness on the solve
    # INPUTS — the solved P stays near ambient by design, see the P6.3 gate).
    if max_pstar < 20 * FP_ONE:
        ok = False
        print(f"  scenario too tame: peak p* = {max_pstar / FP_ONE:.1f} atm "
              f"< 20 atm")
    if max_n_sub < int(eos_cpu.N_SUB_MAX):
        ok = False
        print(f"  scenario too tame: max n_sub {max_n_sub} never hit "
              f"N_SUB_MAX={int(eos_cpu.N_SUB_MAX)}")

    print(f"  per-tick wall cost (informational, review §0 predicts GPU "
          f"slower per-call at this size): CPU {1e3 * t_cpu / n_ticks:.2f} "
          f"ms/tick vs GPU {1e3 * t_gpu / n_ticks:.2f} ms/tick "
          f"({t_gpu / max(t_cpu, 1e-9):.1f}x).")
    if ok:
        print(f"  {n_ticks} ticks bit-identical across all EOS fields "
              f"(wind/T/P/P_prev/all gas planes), all six digests, all five "
              f"rail counters; dispatch fired {calls}/{n_ticks}; peak p* = "
              f"{max_pstar / FP_ONE:.1f} atm; n_sub pinned at {max_n_sub}.")
    return ok


def part2_golden() -> bool:
    print("PART 2 — CUDA build's CPU path (flags off) vs the committed golden:")
    _set_eos_backends(False)
    from field_ab_harness import capture_trajectory
    from field_digest import trajectory_digest

    # The committed default-scenario golden (see cuda_s4a_check.py history;
    # last re-baselined 2026-07-10, eos-p3fix-thermal-ceiling).
    # P-R4 GOLDEN REBASE (2026-08-01, the arc's ONE deliberate rebase —
    # ruling amendment 5 D2, Erik's approval). The canonical A/B scenario seeds
    # fire at (8,8)/(8,9) on AIR tiles (material 0, heat_atten 0,
    # flammable.sum() == 0) — a GHOST fire whose only observable was the retired
    # painter's air deposit. Under Kirchhoff a body that cannot absorb cannot
    # emit (a_s == 0), so that heat is now correctly ZERO and every trajectory
    # carrying it moves. Folded into the SAME one-shot rebase: D1's demand
    # accumulator (digest spec v2 -> v3, +dem_acc), D3's radiant-flux sensor and
    # D4's per-tick fan rotation. ONE approved change-set, ONE rebase event.
    # P-O2b GOLDEN REBASE (2026-08-02) - the fire-realism arc's OWN single
    # deliberate rebase (design v5.2 section 5: "this arc carries its own
    # single deliberate rebase"; the arc-local golden the design budgets).
    # THE EXTENDED OXYGEN DRAW (Erik's Option 2b) widens `dem_acc` from the 4
    # faces to the 2*R*(R+1) SOURCE OFFSETS within BFS hop-radius DRAW_R -
    # (12, h, w) at the shipped DRAW_R = 2. The shape rides the hashed
    # per-field header, so this is a DIGEST-SPEC VERSION BUMP (v3 -> v4) taken
    # per tests/field_digest_spec.toml's own change procedure, with every
    # committed golden regenerated in the same commit.
    # The A/B scenario carries no flammable tiles, so the LAW itself moves
    # nothing here: the entire delta is dem_acc's layout. That is deliberate
    # and separately gated - at DRAW_R = 1 the offset table's ring 1 IS D4's
    # order, so the plane is bit-for-bit the v3 plane and the full engine
    # reproduces every pre-patch field, byte for byte, over 45 ticks.
    # (was e73f130ea6f514fc285825d1efc828202bfc7e2e77dee3212bed2aa822e45f8a)
    # SINGLE-SOURCED 2026-08-18: was a hardcoded copy of the golden.
    # 11 scripts each carried their own, so ONE deliberate re-baseline
    # left 11 tests red. The sanctioned golden is OWNED by
    # tests/_xarch_perfield_digest.py (its lineage block carries every
    # rebase + rationale); import it, per test_w6_armory.py's own rule.
    from _xarch_perfield_digest import GOLDEN_AGGREGATE as GOLDEN
    base = capture_trajectory(n_steps=30)
    dig = trajectory_digest(base)
    if dig != GOLDEN:
        print(f"  GOLDEN MISMATCH: {dig[:16]}... != {GOLDEN[:16]}...")
        return False
    print(f"  CUDA build CPU path reproduces the golden ({dig[:12]}...).")
    return True


def main() -> int:
    if not getattr(bp, "HAS_CUDA", False) or not bp.cuda_available():
        print("P65_RESULT: FAIL (no CUDA build / device)")
        return 1
    print("device:", bp.cuda_device_info())
    p1 = part1_chained_trajectory()
    p2 = part2_golden()
    if p1 and p2:
        print("P65_RESULT: PASS")
        return 0
    print("P65_RESULT: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
