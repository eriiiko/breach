"""CUDA-S2 LIVE raycaster bit-identity check (runs inside the GPU subprocess).

S2 proved the GPU directional march's `heat` == the CPU cast byte-for-byte in an
ISOLATED harness (cuda_s2_check). This gate proves the LIVE WIRING preserves that:
the fire->heat ray cast that actually runs each tick — PhysicsRunner.cast_fire_heat,
the per-burning-tile source loop that CLEARS nothing (the per-tick heat clear is at
the end of Simulation.step) and ACCUMULATES each source's deposit into gmap.heat —
produces a byte-identical `heat` field whether cast_fire_heat dispatches to the CPU
(Raycaster.cast_source_directional) or the GPU (bp.cuda_raycaster_cast). It also
proves the full 7/7 (all field solvers + the raycaster) is bit-identical end-to-end
over a real 30-tick trajectory, including the synced `heat`/`temperature` fields,
and still reproduces the committed default-scenario golden.

Two parts:

  PART 1 — LIVE cast_fire_heat, multi-source, heat tol 0. Build a real GameMap
  with MANY burning tiles (so cast_fire_heat enumerates many LightSources and
  ACCUMULATES their saturating-add heat deposits into one shared gmap.heat), plus
  smoke/gas in the path (the gas optics `expf` lives on the RGB survival only and
  must NEVER perturb the heat-touched set) and heat_atten occluders. Run the SAME
  PhysicsRunner.cast_fire_heat with the raycaster backend OFF then ON; assert the
  resulting gmap.heat is byte-for-byte equal. This drives the production dispatch
  site, not a re-implementation — so it catches a wiring bug (wrong clear/accumulate,
  a dropped source, a buffer-aliasing slip) that the isolated S2 gate cannot.

  PART 2 — 7/7 INTEGRATION. The default A/B scenario (fire seeded -> cast_fire_heat
  runs) stepped 30 ticks with ALL 7 backends ON vs ALL 7 OFF (CPU). Assert the full
  per-tick trajectory of EVERY synced field (incl. heat + temperature, which the
  raycaster feeds) is bit-identical (diff_trajectories tol 0), and the CPU path
  still reproduces the golden. This is the proof that --cuda is a full 7/7.

Prints ``S2_LIVE_RESULT: PASS``/``FAIL`` and exits 0/1, plus the headline
``RAYCASTER_LIVE_RESULT: PASS``/``FAIL`` the task asks for.
"""
from __future__ import annotations

import sys

import numpy as np

# Import the CUDA build FIRST so it is the cached `breach_physics` before
# field_ab_harness / level_loader (which insert cpp/build/Release on sys.path)
# import it.
import breach_physics as bp

INT32_MAX = (1 << 31) - 1

# The committed default-scenario golden (CUDA-S2 re-baseline, 2026-06-28) — the
# raycaster being live on the GPU must NOT change the CPU-path digest.
# Re-baselined 2026-07-04 (Q2-lift): pure-integer trig kit wired into the
# raycaster ray dirs/cone cos + unit facing + Q16.16-snapped HP deltas —
# the trajectory legitimately moved by quantization-scale deltas.
# (was 60bd331faccc0b08c11e1ccad3ca75fa6f2aa26232b0b04c1a070b6c65c86ba1)
# Re-baselined 2026-07-04 (spawn-stat pin): unit spawn stats switched from
# rng.multivariate_normal (LAPACK/BLAS -- CPU-dispatch-dependent, caused the
# Ada tick-0 __unit_hp__ cross-machine divergence, lenovo_dev_setup.md 8b)
# to Q16.16-quantized species MEANS (ingress door 2): spawn hp now exactly
# 100.0. Only __unit_hp__ moved, from tick 0; all field trajectories identical.
# (was 453829a67a38d79e0befd01d591cb19bdeb19f49d9234fb4d27a5083d126501a)
# Re-baselined 2026-07-05 (P3 statuses): the synced unit record grows the
# status list (__unit_status__ sub-hash); no field trajectory moved.
# (was ae1164ca163b4bf49a86694ba78ea5319f86cfff46301c6aa59190207e6c1a12)
GOLDEN = "6d690fda8259b392be9029082013623fbef0fc0322ed3089107d5db220e1b441"


# ----------------------------------------------------------------------------
# PART 1 — the LIVE cast_fire_heat, multi-source, CPU vs GPU heat (tol 0).
# ----------------------------------------------------------------------------
def _build_runner_and_map(seed):
    """A real PhysicsRunner + a default-scenario GameMap with MANY burning tiles,
    smoke/gas in the path, and heat_atten occluders — so cast_fire_heat builds a
    multi-source list and accumulates the saturating-add heat across overlapping
    rays (the multi-source accumulation the live gate must preserve)."""
    from field_ab_harness import default_scenario_sim
    from simulation import fire_fixed, gas_fixed
    from simulation.physics_runner import PhysicsRunner

    sim = default_scenario_sim()
    g = sim.gmap
    rng = np.random.default_rng(seed)

    interior = (~g.solid) & (~g.is_vacuum)
    ys, xs = np.nonzero(interior)
    h, w = g.solid.shape

    # Light MANY interior tiles on fire (varied intensity -> varied range/heat),
    # so cast_fire_heat enumerates a big source list. Quantize to the Q16.16 fire
    # field exactly as the sim does (a raw assignment would store ~0 counts).
    g.fire[...] = 0
    n_fire = max(8, len(ys) // 3)
    pick = rng.choice(len(ys), size=min(n_fire, len(ys)), replace=False)
    for k in pick:
        yy, xx = int(ys[k]), int(xs[k])
        g.fire[yy, xx] = fire_fixed.quantize_scalar(float(rng.uniform(0.3, 1.0)))

    # Smoke/gas across the beams' paths (so the gas `expf` runs on the RGB path —
    # it must not touch the heat-set). Fill a couple of int32 gas planes with a
    # random interior cloud (Q16.16 counts).
    for gi in range(g.gas.shape[0]):
        if rng.random() < 0.6:
            plane = np.zeros((h, w), dtype=np.int32)
            blob = (rng.random((h, w)) * 0.8 * gas_fixed.FP_ONE_F).astype(np.int32)
            plane[interior] = blob[interior]
            g.gas[gi] = plane

    # A few heat_atten occluders cutting across the room (partial + full), so the
    # heat survival decays and the occlusion branch is exercised live.
    g.heat_atten[h // 2, :] = 0.7
    if w > 4:
        g.heat_atten[:, w // 2] = 1.0

    runner = PhysicsRunner(bp)
    # Mirror the binds the live Simulation does for the heat-cast params (the
    # PhysicsRunner __init__ already binds them from config; nothing extra needed).
    return runner, g


def _cast_live(runner, g):
    """Run the production cast_fire_heat once into a freshly-zeroed gmap.heat and
    return the resulting heat field (the per-tick clear is the sim's job; we zero
    it here to isolate THIS cast's accumulation)."""
    g.heat[...] = 0
    runner.cast_fire_heat(g)
    return g.heat.copy()


def part1_live_cast() -> bool:
    print("PART 1 — LIVE cast_fire_heat heat bit-identity (multi-source, tol 0):")
    ok = True
    n_scen = 0
    for seed in (20260628, 3, 17, 51, 88):
        # Rebuild a fresh runner+map per backend so neither run sees the other's
        # accumulated state (the runner caches scratch buffers; a fresh instance
        # mirrors the live game's single-runner-per-session but guarantees the A/B
        # starts from identical zeroed scratch).
        runner_cpu, g_cpu = _build_runner_and_map(seed)
        bp.set_raycaster_backend(False)
        heat_cpu = _cast_live(runner_cpu, g_cpu)

        runner_gpu, g_gpu = _build_runner_and_map(seed)
        bp.set_raycaster_backend(True)
        heat_gpu = _cast_live(runner_gpu, g_gpu)
        bp.set_raycaster_backend(False)   # restore
        n_scen += 1

        # Sanity: both maps must be the SAME scenario (same fire layout) — the A/B
        # is only meaningful if the inputs match. (default_scenario_sim + the same
        # seed make them identical; assert the fire fields agree.)
        if not np.array_equal(g_cpu.fire, g_gpu.fire):
            ok = False
            print(f"  seed {seed}: SCENARIO MISMATCH (fire layout differs) — "
                  f"the A/B inputs are not identical, gate invalid.")
            continue

        if not np.array_equal(heat_cpu, heat_gpu):
            ok = False
            mism = int(np.count_nonzero(heat_cpu != heat_gpu))
            idx = int(np.argmax(heat_cpu != heat_gpu))
            ry, rx = divmod(idx, heat_cpu.shape[1])
            print(f"  seed {seed}: {mism} HEAT MISMATCH "
                  f"(first @ ({ry},{rx}): cpu={heat_cpu.flat[idx]} "
                  f"gpu={heat_gpu.flat[idx]})")
        else:
            nz = int(np.count_nonzero(heat_cpu))
            nfire = int(np.count_nonzero(g_cpu.fire))
            peak = int(heat_cpu.max())
            print(f"  seed {seed}: bit-identical "
                  f"({nfire} fire sources -> {nz} heated tiles, peak={peak}).")
            if nz == 0 or nfire < 2:
                ok = False
                print(f"  seed {seed}: SCENARIO TOO WEAK (nfire={nfire}, "
                      f"heated={nz}) — multi-source accumulation not exercised.")
    if ok:
        print(f"  all {n_scen} live multi-source casts: GPU heat == CPU heat "
              f"byte-for-byte through PhysicsRunner.cast_fire_heat (per-tick "
              f"clear + per-source saturating accumulate preserved).")
    return ok


def part1b_multitick_live() -> bool:
    """The live cast_fire_heat across an EVOLVING fire field — the production
    tick is fire-solver -> cast_fire_heat each tick, so the cast sees a CHANGING
    source list (fire grows, decays, saturates). Step ONE sim with the real fire
    solver; each tick, before the sim clears heat, cast on BOTH backends into a
    scratch heat buffer and assert byte-identical. Proves the wiring holds tick
    after tick on real evolving state (not just one frozen frame)."""
    print("PART 1b — LIVE cast over EVOLVING fire, per tick, heat tol 0:")
    from field_ab_harness import default_scenario_sim

    sim = default_scenario_sim()
    g = sim.gmap
    runner = sim.physics_runner if sim.physics_runner is not None else None
    if runner is None:
        print("  no physics_runner on the sim — cannot drive the live cast.")
        return False

    ok = True
    n_tick = 0
    max_peak = 0
    for t in range(20):
        # Cast THIS tick's fire on both backends into a fresh scratch heat buffer
        # (don't disturb the sim's own gmap.heat — we use a private copy of the
        # field state for the A/B, casting the SAME source list both ways).
        scratch = np.zeros_like(g.heat)

        saved = g.heat
        g.heat = scratch.copy()
        bp.set_raycaster_backend(False)
        g.heat[...] = 0
        runner.cast_fire_heat(g)
        heat_cpu = g.heat.copy()

        g.heat[...] = 0
        bp.set_raycaster_backend(True)
        runner.cast_fire_heat(g)
        heat_gpu = g.heat.copy()
        bp.set_raycaster_backend(False)
        g.heat = saved   # restore the sim's real heat buffer reference

        if not np.array_equal(heat_cpu, heat_gpu):
            ok = False
            mism = int(np.count_nonzero(heat_cpu != heat_gpu))
            print(f"  tick {t}: {mism} HEAT MISMATCH on the live evolving cast.")
            break
        max_peak = max(max_peak, int(heat_cpu.max()))
        n_tick += 1

        # Advance the sim one real tick so the fire field evolves for the next cast.
        sim.set_paused(False)
        sim.step()
    if ok:
        print(f"  {n_tick} ticks of the live evolving fire->heat cast: GPU heat == "
              f"CPU heat byte-for-byte every tick (peak |heat| over the run = "
              f"{max_peak} counts).")
        if max_peak == 0:
            ok = False
            print("  SCENARIO WEAK: heat never non-zero — vacuous.")
    return ok


# ----------------------------------------------------------------------------
# PART 2 — 7/7 all-backends-on 30-tick integration vs CPU (tol 0) + golden.
# ----------------------------------------------------------------------------
_SETTERS = ("set_temperature_backend", "set_water_backend", "set_smoke_backend",
            "set_wave_backend", "set_fire_backend", "set_atmos_backend",
            "set_raycaster_backend")


def _set_all(on):
    for name in _SETTERS:
        getattr(bp, name)(bool(on))


def part2_integration() -> bool:
    print("PART 2 — 7/7 all-backends-on 30-tick trajectory vs CPU (tol 0) + golden:")
    from field_ab_harness import capture_trajectory, diff_trajectories
    from field_digest import trajectory_digest

    # ALL 7 backends ON (incl. raycaster) — the full GPU tick, fire seeded so
    # cast_fire_heat runs the GPU ray cast each tick.
    _set_all(True)
    traj_gpu = capture_trajectory(n_steps=30)
    # ALL 7 OFF — the pure CPU reference.
    _set_all(False)
    traj_cpu = capture_trajectory(n_steps=30)

    diffs = diff_trajectories(traj_cpu, traj_gpu, tol=0.0)
    ok = (len(diffs) == 0)
    if not ok:
        print(f"  {len(diffs)} field divergence(s) over 30 ticks; "
              f"first 5:")
        for d in diffs[:5]:
            print(f"    {d}")
    else:
        # `heat` is a per-tick deposit buffer CLEARED at the end of Simulation.step
        # (so the post-step snapshot sees 0 — its bit-identity over the trajectory
        # is real but the NON-VACUOUS heat proof is PART 1, which reads heat before
        # the clear). The witness that the GPU raycaster actually RAN each tick is
        # that fire is present (cast_fire_heat enumerates burning tiles every tick).
        peak_fire = max(int(np.abs(s["fire"]).max()) for s in traj_cpu)
        nfields = len(set(traj_cpu[0]) | set(traj_gpu[0]))
        print(f"  CPU vs 7/7-GPU: ALL {nfields} synced fields bit-identical over "
              f"30 ticks (incl. heat + temperature; fire present each tick -> the "
              f"GPU fire->heat cast ran, peak |fire|={peak_fire} counts).")
        if peak_fire == 0:
            ok = False
            print("  SCENARIO WEAK: no fire over the run -> cast_fire_heat never "
                  "cast on the GPU; the 7/7 integration is vacuous for the raycaster.")

    # The CPU path (all backends OFF) must still reproduce the committed golden —
    # the raycaster being live-wired changes NOTHING when the flag is off.
    _set_all(False)
    base = capture_trajectory(n_steps=30)
    dig = trajectory_digest(base)
    if dig != GOLDEN:
        ok = False
        print(f"  GOLDEN MISMATCH: {dig[:16]}... != {GOLDEN[:16]}...")
    else:
        print(f"  CUDA build CPU path reproduces the golden ({dig[:12]}...).")
    return ok


def main() -> int:
    if not getattr(bp, "HAS_CUDA", False) or not bp.cuda_available():
        print("RAYCASTER_LIVE_RESULT: FAIL (no CUDA build / device)")
        print("S2_LIVE_RESULT: FAIL (no CUDA build / device)")
        return 1
    print("device:", bp.cuda_device_info())
    p1 = part1_live_cast()
    p1b = part1b_multitick_live()
    p2 = part2_integration()
    ok = p1 and p1b and p2
    print("RAYCASTER_LIVE_RESULT:", "PASS" if ok else "FAIL")
    print("S2_LIVE_RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
