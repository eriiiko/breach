"""CUDA-S8a — GPU residency bit-identity + payoff gate (in the subprocess).

THE gate for the S8a residency build: Path B (Rung 1, the framework + leaf
solvers — docs/cuda_s8a_residency_spec_2026-07-19, the ★ BUILD FINDING block)
EXTENDED for Path A (the fully resident EOS stage —
docs/cuda_s8a_path_a_impl_2026-07-21.md §7). Parts:

  PART 1a — BIT-IDENTITY, SPACE MAP: a >=30-tick FULL-ENGINE A/B on a seeded
  scenario with detonations / water / fire / a scripted structural edit, driven
  through the REAL per-tick path (PhysicsRunner.step) on two independently
  built worlds — CPU (residency OFF, backends OFF) vs GPU-RESIDENT (residency
  ON, backends ON; Path A: water + the WHOLE EOS stage + traces resident, the
  Path-B EOS bracket GONE). Per tick, asserts byte-for-byte identity (tol 0)
  of ALL synced fields (incl. host heat/ripple/ripple_v) AND the resident-
  maintained telemetry (dbg_last_n_sub / dbg_last_c_local_q / the five rail
  counters). Vacuousness guards: eos_resident_calls() advanced every tick AND
  eos_step_cuda_calls() did NOT (the bracket is gone, not silently per-call).
  LIVE self-referential A/B — no stored golden; "no re-baseline" means the
  per-kernel digest baselines the existing CUDA gates assert are NOT touched.

  PART 1b — BIT-IDENTITY, AMBIENT MAP (Path A critique blocker): the same A/B
  on a planetside ambient-ring world with NONZERO sponge_sigma + sponge_udamp
  and a scripted ring-adjacent breach — covers the resident device ambient
  branches (shift/re-shift, ring excl, σ-fold, ring div_u=0, masked +P_amb
  store, u-damp kick) + the per-tick boundary_flux rail A/B.

  PART 1c — DEVICE MG-BUILD PARITY (Path A critique blocker): after real
  CPU-path ticks on several scenarios (space, ambient+σ, odd-dimension), feed
  the solver's exact solve-input caches to bp.eos_mg_build_parity — the host
  mg_build_levels vs the PRODUCTION device build kernels, byte-compared per
  level per array (excl/m/gE/gS/recip/b/P) against a poisoned hierarchy.
  Localized proof for the hardest port.

  PART 2 — THE PAYOFF: (i) the Path-B isolated water/smoke tax bench; (ii) the
  Path-A isolated EOS-stage bench — per-call run_substeps(do_traces=False) vs
  from_host + run_substeps_resident + to_host at >=2 grid sizes: resident must
  clearly win and win MORE on the bigger grid. (Bench note: the per-call side
  also pays ~6 host FNV plane digests the resident path skips by design —
  part of the margin is digest removal, not transfer removal.)

Prints ``S8A_RESULT: PASS``/``FAIL`` and exits 0/1.
"""
from __future__ import annotations

import sys
import time

import numpy as np

# Import the CUDA build FIRST so it is the cached `breach_physics`.
import breach_physics as bp

FP_ONE = 65536

# The GPU backends flipped on for the resident/per-call runs (run_on_cuda's set +
# combustion). In the resident tick the water/smoke per-call flags are inert (those
# stages run resident), but the EOS/combustion/fire/temperature/raycaster brackets
# use their per-call GPU path — so a resident tick is a genuine all-GPU tick.
_BACKENDS = (
    "set_temperature_backend", "set_water_backend", "set_smoke_backend",
    "set_fire_backend", "set_raycaster_backend",
    "set_bulk_flux_backend", "set_sl_advection_backend",
    "set_mg_solve_backend", "set_kick_compression_backend",
    "set_combustion_backend",
)


def _set_backends(on: bool) -> None:
    for name in _BACKENDS:
        getattr(bp, name)(bool(on))


def _residency(on: bool) -> None:
    from simulation import physics_runner
    physics_runner.set_residency(bool(on))


# Fields compared for bit-identity (ALL synced state, incl. the host-path float
# ripple/ripple_v and the heat deposit). `gas` covers all N planes (bulk+trace).
_FIELDS = ("atmosphere", "wave_p", "wind_x", "wind_y", "temperature", "heat",
           "fire", "wall_hp", "water_depth", "flow_vx", "flow_vy", "gas",
           "ripple", "ripple_v")


def _build_scenario(H, W):
    """One independently constructed runner + map: a hull-ringed room breached to
    a vacuum band (sonic venting), with a blast (hot core + O2 overpressure), a
    standing-water pool (the water stage active), a fire seed (plume + smoke
    emission), and a trace cloud (the resident trace loop on a non-zero plane)."""
    from pathlib import Path

    from config import CFG
    from level_loader import LevelData
    from simulation import atmosphere_fixed, fire_fixed, water_fixed
    from simulation.gamemap import GameMap
    from simulation.gases import O2
    from simulation.physics_runner import PhysicsRunner

    # v1 tilemap vocabulary: 0 = vacuum, 1 = hull wall, 4 = interior air.
    tm = np.zeros((H, W), dtype=np.int32)
    tm[2:H - 2, 2:W - 2] = 1
    tm[3:H - 3, 3:W - 3] = 4
    tm[H // 2 - 2:H // 2 + 2, W - 3] = 4     # breach the east hull to the vacuum band

    level = LevelData(name="s8a_resident", version="1", path=Path("."),
                      tilemap=tm, tile_size_m=1.0 / 3.0, diffuse_path=Path("."))
    g = GameMap(level)
    g.stamp_units([])

    q = atmosphere_fixed.quantize_scalar
    # THE BLAST: a hot core + an O2 overpressure pocket (density spike -> venting).
    g.temperature[8:18, 8:18] += q(5000.0)
    g.gas[O2, 10:16, 10:16] += q(4.0)
    # A standing-water pool on the interior floor (keeps the water stage live).
    g.water_depth[H - 8:H - 4, 6:W // 2] = water_fixed.quantize_scalar(0.4)
    # A fire seed (burns -> plume into atmosphere + smoke emission into gas).
    g.fire[12:15, 12:15] = fire_fixed.quantize_scalar(0.8)
    # A trace cloud (the resident trace loop must transport a non-zero plane).
    trace_ids = [gi for gi in range(g.gas.shape[0])
                 if not bool(g.gases.conservative[gi])]
    assert trace_ids, "scenario needs a trace plane"
    g.gas[trace_ids[0], 20:40, 20:40] += q(0.5)

    runner = PhysicsRunner(bp)
    # Bind the engine so stamp_units takes the C++ (in-place) path, exactly as
    # Simulation does — the Python fallback reassigns `obstacles` (would trip the
    # residency stale-pointer guard, and is not the live game path anyway).
    g.bind_physics_engine(runner.engine)
    dt = 1.0 / float(CFG.clock.ticks_per_second)
    return runner, g, dt


# The scripted structural edit (a combat wall breach) fired mid-run on BOTH worlds.
_EDIT_TICK = 12


def _one_tick(runner, g, dt, tick_idx):
    """One full engine tick, mirroring Simulation.step's physics slice: the
    scripted structural edit (combat), the pre-physics unit stamp, the physics
    step, burn-through wall destruction, and the end-of-tick heat clear."""
    if tick_idx == _EDIT_TICK:
        # A scripted structural edit — a wall breached by "combat" this tick, on
        # the mirror, between ticks (picked up by next tick's batched H2D).
        g.destroy_wall(6, 6)
    g.stamp_units([])
    destroyed = runner.step(g, dt)
    for (yy, xx) in destroyed:
        g.destroy_wall(yy, xx)
    g.heat.fill(0)


# Telemetry the RESIDENT path maintains (design §3.3) — A/B'd per tick.
# (The six digest_* members are a documented resident-path gap — NOT compared.)
_COUNTERS = ("u_clamp_hits", "u_max_hits", "work_clamp_hits",
             "energy_floor_hits", "t_max_phys_hits")


def _compare_tick(t, g_cpu, g_gpu, eos_cpu, eos_gpu, ambient):
    """Per-tick A/B: all synced fields + the resident-maintained telemetry.
    Returns the number of divergences found this tick."""
    bad = 0
    for f in _FIELDS:
        a, b = getattr(g_cpu, f), getattr(g_gpu, f)
        if not np.array_equal(a, b):
            bad += 1
            mism = int(np.count_nonzero(a != b))
            amax = float(np.abs(a.astype(np.int64) - b.astype(np.int64)).max()) \
                if a.dtype != np.float32 else float(np.abs(a - b).max())
            print(f"  tick {t}: field {f}: {mism} MISMATCH(es), max|delta|={amax}")
    if int(eos_cpu.dbg_last_n_sub) != int(eos_gpu.dbg_last_n_sub):
        bad += 1
        print(f"  tick {t}: dbg_last_n_sub mismatch "
              f"(cpu={int(eos_cpu.dbg_last_n_sub)} gpu={int(eos_gpu.dbg_last_n_sub)})")
    if int(eos_cpu.dbg_last_c_local_q) != int(eos_gpu.dbg_last_c_local_q):
        bad += 1
        print(f"  tick {t}: dbg_last_c_local_q mismatch "
              f"(cpu={int(eos_cpu.dbg_last_c_local_q)} "
              f"gpu={int(eos_gpu.dbg_last_c_local_q)})")
    for c in _COUNTERS:
        cc, cg = int(getattr(eos_cpu, c)), int(getattr(eos_gpu, c))
        if cc != cg:
            bad += 1
            print(f"  tick {t}: counter {c} mismatch (cpu={cc} gpu={cg})")
    if ambient:
        rc, rg = list(eos_cpu.boundary_flux()), list(eos_gpu.boundary_flux())
        if rc != rg:
            bad += 1
            print(f"  tick {t}: boundary_flux mismatch cpu={rc} gpu={rg}")
    return bad


def _run_ab(label, build, n_ticks, tick_fn, ambient):
    """The shared A/B driver: two independently built worlds, CPU vs resident,
    per-tick compare + the Path-A vacuousness guards."""
    _residency(False); _set_backends(False)
    runner_cpu, g_cpu, dt = build()
    runner_gpu, g_gpu, dt2 = build()
    assert dt == dt2
    eos_cpu, eos_gpu = runner_cpu.engine.eos, runner_gpu.engine.eos

    for f in _FIELDS:
        assert np.array_equal(getattr(g_cpu, f), getattr(g_gpu, f)), \
            f"{label}: scenario construction not deterministic on {f}"

    res_calls0 = int(bp.eos_resident_calls())
    percall0 = int(bp.eos_step_cuda_calls())
    bad = 0
    for t in range(n_ticks):
        _residency(False); _set_backends(False)
        tick_fn(runner_cpu, g_cpu, dt, t)
        _residency(True); _set_backends(True)
        tick_fn(runner_gpu, g_gpu, dt, t)
        _residency(False); _set_backends(False)
        bad += _compare_tick(t, g_cpu, g_gpu, eos_cpu, eos_gpu, ambient)
        if bad >= 8:
            print("  aborting after 8 divergences")
            break

    # Vacuousness guards (design §7): the resident EOS actually ran every GPU
    # tick, and the per-call EOS path did NOT (the bracket is gone).
    res_delta = int(bp.eos_resident_calls()) - res_calls0
    percall_delta = int(bp.eos_step_cuda_calls()) - percall0
    if res_delta < n_ticks and bad < 8:
        print(f"  {label}: resident EOS ran only {res_delta}/{n_ticks} ticks "
              f"— gate is vacuous")
        return None
    if percall_delta != 0:
        print(f"  {label}: eos_step_cuda_calls advanced by {percall_delta} "
              f"during the resident run — the EOS bracket is NOT gone")
        return None
    if not (bool(g_gpu.residency_on()) and hasattr(g_gpu, "_dev")):
        print(f"  {label}: the GPU world never entered residency mode")
        return None
    return bad, runner_cpu, g_cpu


def part1a_bit_identity() -> bool:
    print("PART 1a — SPACE map full-engine A/B (residency ON vs CPU), 40 ticks, "
          "ALL synced fields tol 0 + telemetry (n_sub/c_local/rail counters); "
          "resident-EOS + no-per-call guards:")
    H = W = 72
    n_ticks = 40
    out = _run_ab("part1a", lambda: _build_scenario(H, W), n_ticks,
                  _one_tick, ambient=False)
    if out is None:
        return False
    bad, runner_cpu, g_cpu = out

    # The scenario must be non-trivial (fields actually present after the run).
    active = int(np.count_nonzero(g_cpu.gas)) + int(np.count_nonzero(g_cpu.water_depth))
    if active == 0:
        print("  scenario went inert (no gas/water) — not a real exercise")
        return False

    ok = (bad == 0)
    if ok:
        print(f"  {n_ticks} ticks bit-identical across all synced fields + "
              f"telemetry; resident EOS confirmed live (bracket gone); "
              f"scripted breach @ t={_EDIT_TICK}.")
    return ok


def _build_ambient_scenario():
    """A planetside ambient world (the cuda_ambient_check pattern): a SPACE
    ring border routed to is_ambient around an open-air interior, a
    ring-adjacent hull stub (the breach target), a detonation, a water pool,
    a fire seed, and NONZERO σ (the sponge_sigma grid ships 0 by default —
    written identically on both worlds so the resident σ-fold is exercised)."""
    from pathlib import Path

    from config import CFG
    from level_loader import LevelData
    from simulation import atmosphere_fixed, fire_fixed, water_fixed
    from simulation.gamemap import GameMap
    from simulation.gases import O2
    from simulation.physics_runner import PhysicsRunner

    H = W = 64
    tm = np.full((H, W), 9, dtype=np.int32)           # interior air
    tm[0, :] = tm[-1, :] = tm[:, 0] = tm[:, -1] = 0   # SPACE ring border
    for c in range(28, 32):
        tm[1, c] = 1                                  # ring-adjacent hull stub
    level = LevelData(name="s8a_patha_ambient", version="1", path=Path("."),
                      tilemap=tm, tile_size_m=1.0 / 3.0,
                      diffuse_path=Path("."), boundary="ambient")
    g = GameMap(level)
    g.stamp_units([])
    assert g.is_ambient.any(), "ambient routing expected"
    assert g.sponge_udamp.any(), "u-damping band must be active"
    # Force a NONZERO σ band (identical on both worlds — in-place write).
    g.sponge_sigma[:] = g.sponge_udamp // 4

    q = atmosphere_fixed.quantize_scalar
    g.temperature[16:26, 16:26] += q(5000.0)
    g.gas[O2, 18:24, 18:24] += q(4.0)
    g.water_depth[H - 10:H - 6, 8:24] = water_fixed.quantize_scalar(0.4)
    g.fire[20:23, 40:43] = fire_fixed.quantize_scalar(0.8)

    runner = PhysicsRunner(bp)
    g.bind_physics_engine(runner.engine)
    dt = 1.0 / float(CFG.clock.ticks_per_second)
    return runner, g, dt


_AMBIENT_BREACH_TICK = 18


def _one_tick_ambient(runner, g, dt, tick_idx):
    if tick_idx == _AMBIENT_BREACH_TICK:
        for c in range(28, 32):
            g.destroy_wall(1, c)
    g.stamp_units([])
    destroyed = runner.step(g, dt)
    for (yy, xx) in destroyed:
        g.destroy_wall(yy, xx)
    g.heat.fill(0)


def part1b_ambient() -> bool:
    print("PART 1b — AMBIENT map full-engine A/B (sigma + u-damp bands live, "
          "ring-adjacent breach), 40 ticks, fields + telemetry + boundary_flux "
          "rail tol 0:")
    n_ticks = 40
    out = _run_ab("part1b", _build_ambient_scenario, n_ticks,
                  _one_tick_ambient, ambient=True)
    if out is None:
        return False
    bad, runner_cpu, g_cpu = out
    rail = list(runner_cpu.engine.eos.boundary_flux())
    if not any(v != 0 for v in rail):
        print("  boundary_flux rail never went non-zero — ring exchange "
              "not exercised")
        return False
    ok = (bad == 0)
    if ok:
        print(f"  {n_ticks} ambient ticks bit-identical (fields + telemetry + "
              f"per-plane rail); sigma-fold + u-damp + ring breach exercised.")
    return ok


def part1c_build_parity() -> bool:
    print("PART 1c — device MG-build parity (host mg_build_levels vs the "
          "production device build, poisoned hierarchy, per-level byte "
          "compare):")
    scenarios = [
        ("space 72x72", lambda: _build_scenario(72, 72), False),
        ("space 71x53 (odd dims)", lambda: _build_scenario(71, 53), False),
        ("ambient 64x64 (sigma live)", _build_ambient_scenario, True),
    ]
    ok = True
    for label, build, ambient in scenarios:
        _residency(False); _set_backends(False)
        runner, g, dt = build()
        # A few real CPU-path ticks so the solve-input caches are live state.
        for t in range(6):
            g.stamp_units([])
            runner.step(g, dt)
            g.heat.fill(0)
        eos = runner.engine.eos
        h, w = g.solid.shape
        ps, du, nt = (np.asarray(a, dtype=np.int32).reshape(h, w)
                      for a in eos.dbg_mg_inputs())
        amb = runner._ambient_args(g)
        mism, report = bp.eos_mg_build_parity(
            eos, ps, du, nt, g.wave_p, g.solid, g.is_vacuum,
            g.dyn_permeability, dt,
            is_ambient=amb[0], p_amb=amb[2], sponge_sigma=amb[3])
        if mism != 0:
            ok = False
            print(f"  {label}: {mism} mismatched cells:\n{report}")
        else:
            print(f"  {label}: device build == host build "
                  f"(every level, every array, tol 0)")
    return ok


def _bench_run(H, W, n_ticks, residency, backends, repeats=3):
    """Best-of-`repeats` ms/tick for `n_ticks` full engine ticks in one mode."""
    _residency(residency); _set_backends(backends)
    runner, g, dt = _build_scenario(H, W)
    _one_tick(runner, g, dt, -1)   # warm-up (lazy device allocs) — not timed
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        for t in range(n_ticks):
            _one_tick(runner, g, dt, -1)   # -1 => no scripted edit (pure throughput)
        best = min(best, time.perf_counter() - t0)
    _residency(False); _set_backends(False)
    return 1e3 * best / n_ticks    # ms/tick


def _bench_tax(H, W, reps=150):
    """Isolate the substep-/plane-MULTIPLIED transfer tax: time ONLY the water
    substep loop + the 5-plane smoke trace loop, per-call (each substep/plane a
    separate cudaMalloc + H2D + D2H) vs resident (two batched round-trips on
    persistent buffers). This is exactly the tax the FLOOR kills — the full-tick
    win (informational, below) buries it under the still-bracketed EOS. Returns
    (per_call_ms, resident_ms) per iteration."""
    import cupy as cp

    _residency(False); _set_backends(True)
    runner, g, dt = _build_scenario(H, W)
    g.enable_residency()
    dev = g.device_ptrs()
    wp = runner.water
    dx = float(g.tile_size_m)
    n_sub = int(runner.engine.water_substep_count(dt))
    wdt = dt / n_sub
    trace_ids = [gi for gi in range(g.gas.shape[0])
                 if not bool(g.gases.conservative[gi])]
    n2 = int(g.gases.name_to_id["inert_n2"])
    diffusion = g.gases.diffusion
    adv = float(np.float32(1.0) / max(np.float32(dx), np.float32(1e-3)))

    def per_call():   # the MULTIPLIED per-call path (n_sub + 5 malloc/H2D/D2H)
        for _ in range(n_sub):
            bp.cuda_water_step(g.water_depth, g.flow_vx, g.flow_vy,
                               g.floor_height, g.atmosphere, g.solid,
                               wdt, g.tilt_x, g.tilt_y, wp.g, wp.damping,
                               dx, wp.k_p, wp.v_max, wp.depth_eps)
        for gi in trace_ids:
            bp.cuda_smoke_step(g.gas[gi], g.wind_x, g.wind_y,
                               g.solid, g.solid, g.is_vacuum, g.dyn_permeability,
                               dt, float(diffusion[gi]), 0.0, adv)

    def resident():   # the RESIDENT path (2 batched round-trips)
        g.from_host(["water_depth", "flow_vx", "flow_vy", "atmosphere", "solid"])
        bp.water_substeps_resident(
            dev["water_depth"], dev["flow_vx"], dev["flow_vy"],
            dev["floor_height"], dev["atmosphere"], dev["solid"],
            H, W, n_sub, wdt, g.tilt_x, g.tilt_y,
            wp.g, wp.damping, dx, wp.k_p, wp.v_max, wp.depth_eps)
        g.to_host(["water_depth", "flow_vx", "flow_vy"])
        g.from_host(["gas", "wind_x", "wind_y", "dyn_permeability",
                     "is_vacuum", "solid"])
        bp.trace_smoke_resident(
            dev["gas"], dev["wind_x"], dev["wind_y"],
            dev["solid"], dev["is_vacuum"], dev["dyn_permeability"], 0,
            H, W, g.gas.shape[0], n2,
            g.gases.conservative, g.gases.diffusion, g.gases.decay, dt, adv, 0.0)
        g.to_host(["gas"])

    per_call(); resident(); cp.cuda.Stream.null.synchronize()   # warm-up

    def timeit(fn):
        best = float("inf")
        for _ in range(3):
            t0 = time.perf_counter()
            for _ in range(reps):
                fn()
            cp.cuda.Stream.null.synchronize()
            best = min(best, time.perf_counter() - t0)
        return 1e3 * best / reps
    pc, res = timeit(per_call), timeit(resident)
    _residency(False); _set_backends(False)
    return pc, res, n_sub, len(trace_ids)


def _bench_eos_stage(H, W, reps=60):
    """Path A payoff: the ISOLATED EOS stage, per-call (the Path-B bracket —
    run_substeps with do_traces=False, internal malloc/H2D/D2H + the MG
    hierarchy upload every call) vs fully resident (from_host(8) +
    run_substeps_resident + to_host(6) — honestly charged its bracket).
    NOTE: part of the per-call cost is its ~6 host FNV plane digests, which
    the resident path skips by design (documented gap) — the margin is
    transfer + malloc + digest removal combined."""
    import cupy as cp

    def build(resident):
        _residency(False); _set_backends(True)
        runner, g, dt = _build_scenario(H, W)
        # Prime the runner-lazy ids + dx exactly as PhysicsRunner.step does
        # (we drive engine.run_substeps* directly here).
        if runner._o2_idx is None:
            runner._o2_idx = int(g.gases.name_to_id["o2"])
            runner._inert_n2_idx = int(g.gases.name_to_id["inert_n2"])
            runner._black_smoke_idx = int(g.gases.name_to_id["black_smoke"])
        runner.eos.dx = float(g.tile_size_m)
        if resident:
            g.enable_residency()
        return runner, g, dt

    def make_percall(runner, g, dt):
        amb = runner._ambient_args(g)

        def fn():
            runner.engine.run_substeps(
                g.wave_p, g.atmosphere, g.wind_x, g.wind_y, g.temperature,
                g.obstacles, g.solid, g.is_vacuum,
                g.dyn_permeability, g.dyn_wave_absorb,
                g.gas, g.gases.diffusion, g.gases.conservative,
                g.gases.decay, runner._inert_n2_idx,
                dt, is_ambient=amb[0], n_amb=amb[1], p_amb=amb[2],
                sponge_sigma=amb[3], sponge_udamp=amb[4], do_traces=False)
        return fn

    def make_resident(runner, g, dt):
        amb = runner._ambient_args(g)
        dev = g.device_ptrs()

        def fn():
            g.from_host(["atmosphere", "wind_x", "wind_y", "temperature",
                         "gas", "solid", "is_vacuum", "is_ambient",
                         "dyn_permeability"])
            runner.engine.run_substeps_resident(
                g.wave_p, g.atmosphere, g.wind_x, g.wind_y, g.temperature,
                g.solid, g.is_vacuum, g.dyn_permeability, g.dyn_wave_absorb,
                g.gas, g.gases.conservative, dt,
                is_ambient=amb[0], n_amb=amb[1], p_amb=amb[2],
                d_atmosphere=dev["atmosphere"], d_wave_p=dev["wave_p"],
                d_wind_x=dev["wind_x"], d_wind_y=dev["wind_y"],
                d_temperature=dev["temperature"], d_gas=dev["gas"],
                d_solid=dev["solid"], d_is_vacuum=dev["is_vacuum"],
                d_dyn_permeability=dev["dyn_permeability"])
            g.to_host(["atmosphere", "wave_p", "wind_x", "wind_y",
                       "temperature", "gas"])
        return fn

    def timeit(fn):
        fn(); cp.cuda.Stream.null.synchronize()   # warm-up (lazy allocs)
        best = float("inf")
        for _ in range(3):
            t0 = time.perf_counter()
            for _ in range(reps):
                fn()
            cp.cuda.Stream.null.synchronize()
            best = min(best, time.perf_counter() - t0)
        return 1e3 * best / reps

    r_pc, g_pc, dt = build(resident=False)
    pc = timeit(make_percall(r_pc, g_pc, dt))
    r_rs, g_rs, dt = build(resident=True)
    res = timeit(make_resident(r_rs, g_rs, dt))
    _residency(False); _set_backends(False)
    return pc, res


def part2_payoff() -> bool:
    print("PART 2 — the payoff (the substep-/plane-MULTIPLIED transfer tax must be "
          "gone). ISOLATED water-substep + 5-plane-smoke loop, per-call vs resident "
          "at >=2 grid sizes (the tax is a grid-AREA cost -> resident wins MORE on "
          "bigger grids):")
    sizes = [(128, 128), (256, 256), (384, 384)]
    ratios = []
    for (H, W) in sizes:
        pc, res, n_sub, n_tr = _bench_tax(H, W)
        ratio = pc / max(res, 1e-9)
        ratios.append(ratio)
        print(f"  {H:3d}x{W:<3d} (water x{n_sub} substeps + smoke x{n_tr} planes): "
              f"per-call {pc:7.3f} | RESIDENT {res:7.3f} ms  "
              f"({ratio:.2f}x faster resident)")
    # The tax is GONE iff resident clearly beats per-call at EVERY size AND the
    # advantage remains MULTIPLIED at the biggest grid (the area-scaling win).
    # Path-A robustness fix (2026-07-21): the original monotone-growth ordering
    # (asserted ratios grow with size) was demonstrated definitively at the
    # Path-B merge (1.75x -> 3.20x -> 4.77x @128/256/384^2, recorded in the
    # ledger) but is NOT a stable regression signal on a laptop GPU: per-call
    # timings swing ~2x with power/thermal state (measured 4.9/4.4/4.6 and
    # 5.4/5.0/4.5 across otherwise-green runs), so adjacent-size ratio ORDER is
    # bench noise while the ratios themselves stay huge. The durable claims: a
    # clear win at every size, and a >=3x multiplied advantage at 384^2 — a
    # real residency regression (per-substep transfers reappearing) would
    # collapse both long before the noise band matters.
    wins = all(r > 1.05 for r in ratios)
    big = ratios[-1] > 3.0
    ok = wins and big
    if not wins:
        print("  FAIL: resident did not clearly beat per-call at every size "
              "(the multiplied transfer tax is not gone)")
    if not big:
        print(f"  FAIL: the resident advantage at the biggest grid fell to "
              f"{ratios[-1]:.2f}x (>= 3x expected — the multiplied tax is back?)")
    if ok:
        print(f"  the multiplied transfer tax is GONE: resident is "
              f"{ratios[0]:.2f}x -> {ratios[-1]:.2f}x faster than per-call as the "
              f"grid grows (the tax scales with area; residency removes it).")

    # Path A: the ISOLATED EOS-stage bench — the transfer/malloc/digest tax of
    # the per-call bracket must be gone (design §7 PART 2).
    print("  Path A — isolated EOS stage, per-call bracket vs fully resident:")
    eos_ratios = []
    for (H, W) in [(128, 128), (256, 256)]:
        pc, res = _bench_eos_stage(H, W)
        ratio = pc / max(res, 1e-9)
        eos_ratios.append(ratio)
        print(f"  {H:3d}x{W:<3d} EOS stage: per-call {pc:7.3f} | "
              f"RESIDENT {res:7.3f} ms  ({ratio:.2f}x faster resident)")
    # Same robustness shape as the water/smoke assert above: wins at every
    # size + a strong absolute floor at the biggest grid (growth 1.8x -> 2.2x
    # was demonstrated consistently at build time; adjacent-size ORDER is
    # laptop-bench noise, the multiplied advantage at scale is the signal).
    eos_wins = all(r > 1.05 for r in eos_ratios)
    eos_big = eos_ratios[-1] > 1.5
    if not eos_wins:
        print("  FAIL: resident EOS did not clearly beat the per-call bracket")
    if not eos_big:
        print(f"  FAIL: the resident-EOS advantage at the biggest grid fell "
              f"to {eos_ratios[-1]:.2f}x (>= 1.5x expected)")
    if eos_wins and eos_big:
        print(f"  the EOS transfer tax is GONE: resident EOS is "
              f"{eos_ratios[0]:.2f}x / {eos_ratios[-1]:.2f}x faster than the "
              f"per-call bracket at 128^2 / 256^2. (Margin = transfers + "
              f"~40 mallocs + MG-hierarchy upload + host digests per call.)")
    ok = ok and eos_wins and eos_big

    # Full-engine context (informational): with Path A the EOS bracket is gone —
    # only combustion + the tail remain bracketed (S8c).
    cpu = _bench_run(256, 256, 20, residency=False, backends=False)
    percall = _bench_run(256, 256, 20, residency=False, backends=True)
    resident = _bench_run(256, 256, 20, residency=True, backends=True)
    print(f"  [full tick @256x256, informational: CPU {cpu:.1f} | per-call GPU "
          f"{percall:.1f} | RESIDENT {resident:.1f} ms/tick -- water+EOS+smoke "
          f"resident, combustion/tail still bracketed (S8c)]")
    return ok


def main() -> int:
    if not getattr(bp, "HAS_CUDA", False) or not bp.cuda_available():
        print("S8A_RESULT: FAIL (no CUDA build / device)")
        return 1
    try:
        import cupy  # noqa: F401
    except Exception as e:
        print(f"S8A_RESULT: FAIL (cupy not importable: {e!r})")
        return 1
    print("device:", bp.cuda_device_info())
    p1a = part1a_bit_identity()
    p1b = part1b_ambient()
    p1c = part1c_build_parity()
    p2 = part2_payoff()
    if p1a and p1b and p1c and p2:
        print("S8A_RESULT: PASS")
        return 0
    print("S8A_RESULT: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
