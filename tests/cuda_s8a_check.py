"""CUDA-S8a Path B — GPU residency bit-identity + payoff gate (in the subprocess).

THE gate for the Rung-1 residency build (docs/cuda_s8a_residency_spec_2026-07-19,
the ★ BUILD FINDING + RUNG-1 SPLIT block). Two parts:

  PART 1 — BIT-IDENTITY (spec §6): a >=30-tick FULL-ENGINE A/B on a seeded
  scenario with detonations / water / fire / a scripted structural edit, driven
  through the REAL per-tick path (PhysicsRunner.step) on two independently built
  worlds — once on the CPU path (residency OFF, all backends OFF) and once on the
  GPU-RESIDENT path (residency ON, all backends ON: the water substep loop + the
  smoke trace loop run resident on persistent device buffers; EOS + combustion +
  the tail are bracketed). Per tick, asserts byte-for-byte identity (tol 0) of
  ALL synced fields, INCLUDING the host-path heat / ripple / ripple_v. This is a
  LIVE self-referential A/B (resident-ON vs CPU) — there is no stored full-engine
  golden to reproduce; "no re-baseline" means the per-kernel digest baselines the
  existing CUDA gates assert are NOT touched (this build changes no kernel math).

  PART 2 — THE PAYOFF (spec §6): benchmark N ticks CPU vs per-call GPU vs resident
  at two grid sizes. The substep-/plane-MULTIPLIED transfer tax must be gone —
  resident clearly beats per-call GPU, and by MORE on the bigger grid (the tax
  scales with grid area x substep/plane count).

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


def part1_bit_identity() -> bool:
    print("PART 1 — full-engine A/B (residency ON vs CPU), 40 ticks, ALL synced "
          "fields tol 0 (incl host heat/ripple/ripple_v):")
    H = W = 72
    n_ticks = 40

    _residency(False); _set_backends(False)
    runner_cpu, g_cpu, dt = _build_scenario(H, W)
    runner_gpu, g_gpu, dt2 = _build_scenario(H, W)
    assert dt == dt2

    # The two worlds must START identical.
    for f in _FIELDS:
        assert np.array_equal(getattr(g_cpu, f), getattr(g_gpu, f)), \
            f"scenario construction not deterministic on {f}"

    bad = 0
    for t in range(n_ticks):
        # -- CPU reference tick (residency + backends OFF) ------------------
        _residency(False); _set_backends(False)
        _one_tick(runner_cpu, g_cpu, dt, t)

        # -- GPU-resident tick (residency + backends ON) -------------------
        _residency(True); _set_backends(True)
        _one_tick(runner_gpu, g_gpu, dt, t)
        _residency(False); _set_backends(False)

        # -- per-tick bit-identity -----------------------------------------
        for f in _FIELDS:
            a, b = getattr(g_cpu, f), getattr(g_gpu, f)
            if not np.array_equal(a, b):
                bad += 1
                mism = int(np.count_nonzero(a != b))
                amax = float(np.abs(a.astype(np.int64) - b.astype(np.int64)).max()) \
                    if a.dtype != np.float32 else float(np.abs(a - b).max())
                print(f"  tick {t}: field {f}: {mism} MISMATCH(es), max|delta|={amax}")
        if bad >= 8:
            print("  aborting after 8 divergences")
            break

    # The resident path must actually have RUN resident (guard against a silently-
    # CPU "GPU run" making the gate vacuous): the GPU world's map is in residency
    # mode and its device buffers exist.
    resident_fired = bool(g_gpu.residency_on()) and hasattr(g_gpu, "_dev")
    if not resident_fired:
        print("  the GPU world never entered residency mode — gate is vacuous")
        return False

    # The scenario must be non-trivial (fields actually present after the run).
    active = int(np.count_nonzero(g_cpu.gas)) + int(np.count_nonzero(g_cpu.water_depth))
    if active == 0:
        print("  scenario went inert (no gas/water) — not a real exercise")
        return False

    ok = (bad == 0)
    if ok:
        print(f"  {n_ticks} ticks bit-identical across all synced fields "
              f"(P/P_prev/wind/T/heat/fire/wall_hp/water/flow/all gas planes/"
              f"ripple); residency confirmed live; scripted breach @ t={_EDIT_TICK}.")
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
    # The tax is GONE iff resident clearly beats per-call at EVERY size, and the
    # advantage STRENGTHENS with grid area (it is a grid-scaling transfer cost).
    wins = all(r > 1.05 for r in ratios)
    grows = ratios[-1] > ratios[0]
    ok = wins and grows
    if not wins:
        print("  FAIL: resident did not clearly beat per-call at every size "
              "(the multiplied transfer tax is not gone)")
    if not grows:
        print(f"  FAIL: the resident advantage did not grow with grid size "
              f"({ratios[0]:.2f}x -> {ratios[-1]:.2f}x)")
    if ok:
        print(f"  the multiplied transfer tax is GONE: resident is "
              f"{ratios[0]:.2f}x -> {ratios[-1]:.2f}x faster than per-call as the "
              f"grid grows (the tax scales with area; residency removes it).")

    # Full-engine context (informational — Rung-1 keeps EOS bracketed, so the
    # whole-tick margin is modest; Path-A removing the EOS bracket is the big win).
    cpu = _bench_run(256, 256, 20, residency=False, backends=False)
    percall = _bench_run(256, 256, 20, residency=False, backends=True)
    resident = _bench_run(256, 256, 20, residency=True, backends=True)
    print(f"  [full tick @256x256, informational: CPU {cpu:.1f} | per-call GPU "
          f"{percall:.1f} | RESIDENT {resident:.1f} ms/tick -- water+smoke resident, "
          f"EOS/combustion/tail still bracketed (Path-A's win)]")
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
    p1 = part1_bit_identity()
    p2 = part2_payoff()
    if p1 and p2:
        print("S8A_RESULT: PASS")
        return 0
    print("S8A_RESULT: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
