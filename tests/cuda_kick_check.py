"""EOS P6.4 — momentum kick + compression work bit-identity check (runs inside
the GPU subprocess).

Three gates:

  PART 1 — ISOLATED (synthetic, all branches + ALL FIVE RAILS FORCED): crafted
  and random inputs that hit every branch of the step-4/4c tail — the
  zero-gradient fast path, the kick's reciprocal chain against floored and
  ambient N̂, absorption a=0 / 0<a<1 / a>=1 (kk=0), the ±2^30 RAD_SAFE
  component pre-clamp, the counted |u| clamp with BOTH binding caps (c_LOCAL
  when c_local_q < U_MAX; U_MAX when the state-derived cap exceeds it), the
  4c work clamp on both signs, the T_MIN energy floor and the T_MAX_PHYS
  ceiling, solid/vacuum cells, trace-plane Dalton weighting, degenerate
  1xN / Nx1 grids. Run BOTH the GPU chain (bp.cuda_eos_kick_compression) and
  the CPU reference (bp.eos_kick_compression_ref — the step-4/4c loops copied
  line for line from EOSSolver::step) on identical copies and assert
  byte-for-byte equality on (wind_x, wind_y, temperature) + digest equality +
  BIT-EXACT equality of all five rail counters. The config set is asserted to
  engage every counter at least once (u_clamp, u_max, work_clamp,
  energy_floor, t_max_phys) — the rails and their telemetry are digest-covered,
  not just the happy path.

  PART 2 — TRAJECTORY (the review's §4 P6.4 digest gate): a blast + breach
  venting scenario (a 5000 K core + O2 overpressure, PLUS a near-ceiling
  15500 K pocket so the v2.4 rails engage in-engine) driven through the REAL
  engine path (PhysicsEngine.run_substeps -> EOSSolver::step) for 120 ticks
  on the CPU. Per tick: snapshot the step-1-entry (wind, T) state and the
  solver's cumulative rail counters, run the real tick, reconstruct the
  step-4-entry state by replaying the advection substeps on the snapshot
  (eos_sl_advect_ref + dbg_last_n_sub, asserted == digest_advect — the P6.2
  contract), then replay the isolated kick+compression on that state (p_new =
  the post-tick atmosphere, which step 5 copies verbatim from the solved P;
  N̂ from the post-tick gas planes — valid because run_substeps' post-step
  trace advection/decay only moves NON-ZERO trace planes and this scenario
  keeps every trace plane zero, asserted; c_local_q = dbg_last_c_local_q)
  through BOTH the CPU reference and the GPU chain, asserting
      ref digests == EOSSolver.digest_velocity / digest_compression
      ref fields  == the engine's own post-tick wind/temperature bytes
      ref counter deltas == the solver's cumulative-counter deltas
      GPU == ref on fields, digests, AND counters
  — a full per-tick digest + rail-telemetry trajectory, CPU vs GPU, over the
  whole run. The scenario is asserted to actually drive the tail hard (n_sub
  pins at N_SUB_MAX, the counted |u| clamp engages with U_MAX as the binding
  cap, the work clamp engages). The T_MIN/T_MAX_PHYS field rails do not fire
  in a pure run_substeps loop (the v2.4 measured outcome — they need
  fire/combustion heat, outside this pass's surface); their coverage is
  Part 1's deterministic forcer.

  PART 3 — the CUDA build's CPU path still reproduces the committed
  default-scenario golden (the s4a-check idiom; proves the P6.4 additions
  changed no CPU trajectory).

Prints ``P64_RESULT: PASS``/``FAIL`` and exits 0/1.
"""
from __future__ import annotations

import sys

import numpy as np

# Import the CUDA build FIRST so it is the cached `breach_physics` before
# anything else (field_ab_harness inserts cpp/build/Release) imports it.
import breach_physics as bp

FP_ONE = 65536

# EOSSolver's config defaults (eos_solver.h) — the isolated Part-1 constants.
CONSTS = dict(
    c_max=300.0, dx=1.0 / 3.0, adiabatic_index=1.4, absorb_strength=8.0,
    n_floor_solver=1e-3, t_min=-289.0, t_work_clamp=0.5,
    t_max_phys=16000.0, u_max=1000.0, trace_mass_scale=0.02,
)

COUNTER_NAMES = ("u_clamp", "u_max", "work_clamp", "energy_floor", "t_max_phys")


def _quantize(x):
    """Round-to-nearest Q16.16 (matches fixedpoint::quantize)."""
    x = np.asarray(x, dtype=np.float64)
    return np.int32(np.where(x >= 0, np.floor(x * FP_ONE + 0.5),
                             np.ceil(x * FP_ONE - 0.5)))


def _run_pair(inp, dt, c_local_q, consts):
    """Run CPU reference + GPU chain on identical copies; return everything."""
    args_tail = (inp["p_new"], inp["gas"], inp["gas_conservative"],
                 inp["solid"], inp["is_vacuum"], inp["absorb"])
    f_ref = {k: inp[k].copy() for k in ("wind_x", "wind_y", "temperature")}
    res_ref = bp.eos_kick_compression_ref(
        f_ref["wind_x"], f_ref["wind_y"], f_ref["temperature"],
        *args_tail, dt, int(c_local_q), **consts)
    f_gpu = {k: inp[k].copy() for k in ("wind_x", "wind_y", "temperature")}
    res_gpu = bp.cuda_eos_kick_compression(
        f_gpu["wind_x"], f_gpu["wind_y"], f_gpu["temperature"],
        *args_tail, dt, int(c_local_q), **consts)
    return f_ref, res_ref, f_gpu, res_gpu


def _compare(tag, f_ref, res_ref, f_gpu, res_gpu):
    ok = True
    for k in ("wind_x", "wind_y", "temperature"):
        if not np.array_equal(f_ref[k], f_gpu[k]):
            ok = False
            mism = int(np.count_nonzero(f_ref[k] != f_gpu[k]))
            idx = int(np.argmax(f_ref[k] != f_gpu[k]))
            print(f"  {tag}: {k} {mism} MISMATCH (first @ {idx}: "
                  f"cpu={f_ref[k].flat[idx]} gpu={f_gpu[k].flat[idx]})")
    if tuple(res_ref) != tuple(res_gpu):
        ok = False
        print(f"  {tag}: result mismatch\n    ref={res_ref}\n    gpu={res_gpu}")
    return ok


def _make_random_inputs(rng, h, w, wind_mag, t_lo, t_hi, p_mag, n_scale):
    n = h * w
    wx = _quantize(((rng.random(n) * 2.0 - 1.0) * wind_mag)).reshape(h, w)
    wy = _quantize(((rng.random(n) * 2.0 - 1.0) * wind_mag)).reshape(h, w)
    zero_mask = rng.random((h, w)) < 0.15
    wx[zero_mask] = 0
    wy[zero_mask] = 0

    t = rng.random(n) * (t_hi - t_lo) + t_lo
    t[rng.random(n) < 0.08] = -289.0          # at the T_MIN floor
    t[rng.random(n) < 0.05] = 15999.0         # a hair under the ceiling
    temperature = _quantize(t).reshape(h, w)

    p_new = _quantize(rng.random(n) * p_mag).reshape(h, w)
    p_new[rng.random((h, w)) < 0.10] = 0      # vacuum-zeroed patches

    solid = (rng.random(n) < 0.10).reshape(h, w)
    is_vacuum = (rng.random(n) < 0.08).reshape(h, w)

    # 3 gas planes: O2-like + N2-like (conservative) + one trace.
    gas = np.zeros((3, h, w), dtype=np.int32)
    gas[0] = _quantize(rng.random((h, w)) * 0.30 * n_scale)
    gas[1] = _quantize(rng.random((h, w)) * 0.80 * n_scale)
    gas[2] = _quantize(rng.random((h, w)) * 0.9)          # trace opacity
    gas[0][rng.random((h, w)) < 0.10] = 0                 # sub-floor N̂ cells
    gas[1][rng.random((h, w)) < 0.10] = 0
    gas_conservative = np.array([True, True, False])

    absorb = (rng.random((h, w)) * 1.2).astype(np.float32)
    absorb[rng.random((h, w)) < 0.30] = 0.0               # a == 0 branch
    absorb[rng.random((h, w)) < 0.05] = 4.0               # a >= 1 -> kk == 0

    return {
        "wind_x": np.ascontiguousarray(wx.astype(np.int32)),
        "wind_y": np.ascontiguousarray(wy.astype(np.int32)),
        "temperature": np.ascontiguousarray(temperature.astype(np.int32)),
        "p_new": np.ascontiguousarray(p_new.astype(np.int32)),
        "gas": np.ascontiguousarray(gas),
        "gas_conservative": np.ascontiguousarray(gas_conservative),
        "solid": np.ascontiguousarray(solid),
        "is_vacuum": np.ascontiguousarray(is_vacuum),
        "absorb": np.ascontiguousarray(absorb),
    }


def _make_rail_forcer(h=24, w=24):
    """A deterministic config engaging every rail in one call:
    - a blast quadrant: huge P spike against near-zero N̂ -> RAD_SAFE + |u| clamp;
    - a convergent-flow band around a near-ceiling-hot column -> work clamp
      (k pinned negative) + T_MAX_PHYS ceiling;
    - the same band crossing a floor-cold column -> T_MIN energy floor.
    """
    inp = {
        "wind_x": np.zeros((h, w), dtype=np.int32),
        "wind_y": np.zeros((h, w), dtype=np.int32),
        "temperature": np.zeros((h, w), dtype=np.int32),
        "p_new": np.full((h, w), _quantize(1.0), dtype=np.int32),
        "gas": np.zeros((3, h, w), dtype=np.int32),
        "gas_conservative": np.array([True, True, False]),
        "solid": np.zeros((h, w), dtype=bool),
        "is_vacuum": np.zeros((h, w), dtype=bool),
        "absorb": np.zeros((h, w), dtype=np.float32),
    }
    inp["gas"][0][:, :] = _quantize(0.21)
    inp["gas"][1][:, :] = _quantize(0.79)
    # Blast quadrant: P spike vs floor-N neighbors (rows 2..8, cols 2..8).
    inp["p_new"][4:7, 4:7] = _quantize(3000.0)
    inp["gas"][0][2:9, 2:9] = 0
    inp["gas"][1][2:9, 2:9] = 0
    # Direct overspeed seeds (RAD_SAFE: |u| raw > 2^30 == 16384 m/s).
    inp["wind_x"][1, 12] = _quantize(25000.0)
    inp["wind_y"][1, 14] = _quantize(-25000.0)
    # Convergent-flow bands (div < 0 at the meeting cell): +u on the left
    # half, -u on the right. Band 1's center is near-ceiling hot (k pinned
    # at -T_WORK_CLAMP -> T*(1.5) crosses T_MAX_PHYS); band 2's center is at
    # the T_MIN floor (-289*(1.5) = -433.5 crosses the floor).
    cy, cx = 16, 12
    inp["wind_x"][cy, :cx] = _quantize(400.0)
    inp["wind_x"][cy, cx:] = _quantize(-400.0)
    inp["temperature"][cy, cx] = _quantize(15000.0)   # -> past T_MAX_PHYS
    cy2 = 20
    inp["wind_x"][cy2, :cx] = _quantize(400.0)
    inp["wind_x"][cy2, cx:] = _quantize(-400.0)
    inp["temperature"][cy2, cx] = _quantize(-289.0)   # -> past T_MIN (k<0)
    for k in inp:
        inp[k] = np.ascontiguousarray(inp[k])
    return inp


def part1_isolated() -> bool:
    print("PART 1 — isolated GPU vs CPU reference (synthetic, all rails):")
    ok = True
    rng = np.random.default_rng(20260711)
    totals = np.zeros(5, dtype=np.int64)   # ref counter engagement coverage
    n_cfg = 0

    # (a) the deterministic all-rails forcer, both cap regimes.
    for c_local, tag in ((2300.0, "forcer c_LOCAL>U_MAX (U_MAX binds)"),
                         (300.0, "forcer c_LOCAL<U_MAX (c_LOCAL binds)")):
        n_cfg += 1
        inp = _make_rail_forcer()
        f_ref, res_ref, f_gpu, res_gpu = _run_pair(
            inp, 1.0 / 24.0, _quantize(c_local), CONSTS)
        ok &= _compare(tag, f_ref, res_ref, f_gpu, res_gpu)
        totals += np.array(res_ref[2:], dtype=np.int64)
        if c_local > CONSTS["u_max"]:
            if res_ref[2] == 0 or res_ref[3] == 0:
                ok = False
                print(f"  {tag}: U_MAX rail did NOT engage "
                      f"(u_clamp={res_ref[2]} u_max={res_ref[3]})")
        else:
            if res_ref[2] == 0 or res_ref[3] != 0:
                ok = False
                print(f"  {tag}: c_LOCAL cap regime wrong "
                      f"(u_clamp={res_ref[2]} u_max={res_ref[3]})")
        if res_ref[4] == 0 or res_ref[5] == 0 or res_ref[6] == 0:
            ok = False
            print(f"  {tag}: 4c rails did not all engage "
                  f"(work={res_ref[4]} floor={res_ref[5]} tmax={res_ref[6]})")

    # (b) random fuzz over sizes/regimes (incl. degenerate 1xN / Nx1).
    configs = [
        (16, 16, 1.0 / 24.0, 0.0,    -50.0,   300.0,   1.5, 1.0, 300.0),
        (16, 16, 1.0 / 24.0, 5.0,    -289.0,  9000.0,  2.0, 1.0, 300.0),
        (24, 32, 1.0 / 24.0, 200.0,  -289.0, 15999.0,  8.0, 1.0, 1281.0),
        (31, 17, 0.5,        900.0,  -289.0,  6000.0, 40.0, 0.2, 2300.0),
        (40, 40, 1.0 / 24.0, 1500.0, -289.0, 15999.0, 500.0, 0.05, 2300.0),
        (12, 20, 1.0 / 24.0, 400.0,  -100.0,  1200.0,  3.0, 1.0, 800.0),
        (1, 50,  1.0 / 24.0, 300.0,  -289.0,  2000.0,  4.0, 1.0, 300.0),
        (50, 1,  1.0 / 24.0, 300.0,  -289.0,  2000.0,  4.0, 1.0, 300.0),
        (8, 8,   2.0,        700.0,  -289.0, 14000.0, 20.0, 0.5, 1500.0),
    ]
    for (h, w, dt, wmag, tlo, thi, pmag, nsc, c_local) in configs:
        for _ in range(4):
            n_cfg += 1
            inp = _make_random_inputs(rng, h, w, wmag, tlo, thi, pmag, nsc)
            f_ref, res_ref, f_gpu, res_gpu = _run_pair(
                inp, dt, _quantize(c_local), CONSTS)
            ok &= _compare(f"{h}x{w} dt={dt} wmag={wmag} c_loc={c_local}",
                           f_ref, res_ref, f_gpu, res_gpu)
            totals += np.array(res_ref[2:], dtype=np.int64)

    # Coverage: every rail counter must have engaged somewhere in Part 1.
    for i, name in enumerate(COUNTER_NAMES):
        if totals[i] == 0:
            ok = False
            print(f"  COVERAGE HOLE: rail counter {name!r} never engaged")
    if ok:
        print(f"  all {n_cfg} configs bit-identical on (wind_x, wind_y, T), "
              f"digests + ALL rail counters equal; rail engagements "
              f"(ref totals): "
              + ", ".join(f"{n}={int(t)}" for n, t in zip(COUNTER_NAMES, totals)))
    return ok


def part2_trajectory() -> bool:
    print("PART 2 — blast + venting trajectory (real engine, per-tick digest):")
    from pathlib import Path

    from config import CFG
    from level_loader import LevelData
    from simulation import atmosphere_fixed
    from simulation.gamemap import GameMap
    from simulation.gases import O2
    from simulation.physics_runner import PhysicsRunner

    H = W = 48
    # v1 tilemap vocabulary: 0 = outer space (vacuum), 1 = hull wall,
    # 4 = interior air. A vacuum band, a hull ring, interior air, and a
    # 4-tile breach carved through the east hull — sustained venting
    # (the P6.2 scenario, plus a near-ceiling pocket for the v2.4 rails).
    tm = np.zeros((H, W), dtype=np.int32)
    tm[2:46, 2:46] = 1
    tm[3:45, 3:45] = 4
    tm[22:26, 45] = 4          # the breach: hull ring opened to the vacuum band
    level = LevelData(name="eos_p64_blast_vent", version="1", path=Path("."),
                      tilemap=tm, tile_size_m=1.0 / 3.0,   # ship tile scale
                      diffuse_path=Path("."))
    g = GameMap(level)
    g.stamp_units([])
    assert g.is_vacuum.any(), "scenario must have vacuum to vent into"

    # THE BLAST: a hot core + an O2 overpressure pocket (the P6.2 drivers),
    # plus a NEAR-CEILING pocket (15500 K on a 16000 K rail) so compression
    # pockets push T onto the T_MAX_PHYS rail and c_LOCAL rises past U_MAX
    # (c_amb*sqrt(15790/290) ~ 2214 m/s) — the counted rails engage in-engine.
    q = atmosphere_fixed.quantize_scalar
    g.temperature[10:16, 10:16] += q(5000.0)
    g.gas[O2, 11:14, 11:14] += q(4.0)
    g.temperature[30:36, 30:36] += q(15500.0)

    runner = PhysicsRunner(bp)
    runner.eos.dx = float(g.tile_size_m)
    eos = runner.engine.eos
    inert_n2_idx = int(g.gases.name_to_id["inert_n2"])
    dt = 1.0 / float(CFG.clock.ticks_per_second)

    consts = dict(
        c_max=float(eos.c_max), dx=float(eos.dx),
        adiabatic_index=float(eos.adiabatic_index),
        absorb_strength=float(eos.absorb_strength),
        n_floor_solver=float(eos.N_FLOOR_SOLVER),
        t_min=float(eos.T_MIN), t_work_clamp=float(eos.T_WORK_CLAMP),
        t_max_phys=float(eos.T_MAX_PHYS), u_max=float(eos.U_MAX),
        trace_mass_scale=float(eos.trace_mass_scale),
    )
    trace_planes = [gi for gi in range(g.gas.shape[0])
                    if not bool(g.gases.conservative[gi])]

    def counters():
        return np.array([eos.u_clamp_hits, eos.u_max_hits,
                         eos.work_clamp_hits, eos.energy_floor_hits,
                         eos.t_max_phys_hits], dtype=np.int64)

    n_ticks = 120
    max_n_sub = 0
    max_u_counts = 0
    totals = np.zeros(5, dtype=np.int64)
    bad = 0
    for tick in range(n_ticks):
        # Snapshot the eos.step step-1-entry state (run_substeps calls
        # eos.step FIRST; step 0 copies P only — u/T enter advection as-is).
        wx0 = np.ascontiguousarray(g.wind_x.copy())
        wy0 = np.ascontiguousarray(g.wind_y.copy())
        t0 = np.ascontiguousarray(g.temperature.copy())
        cnt0 = counters()

        runner.engine.run_substeps(
            g.wave_p, g.atmosphere,
            g.wind_x, g.wind_y,
            g.temperature,
            g.obstacles, g.solid, g.is_vacuum,
            g.dyn_permeability, g.dyn_wave_absorb,
            g.gas, g.gases.diffusion, g.gases.conservative,
            g.gases.decay, inert_n2_idx,
            dt,
        )
        n_sub = int(eos.dbg_last_n_sub)
        c_local_q = int(eos.dbg_last_c_local_q)
        cnt_delta = counters() - cnt0
        totals += cnt_delta
        max_n_sub = max(max_n_sub, n_sub)
        max_u_counts = max(max_u_counts,
                           int(np.abs(g.wind_x).max()), int(np.abs(g.wind_y).max()))

        # The N̂-reconstruction premise: every trace plane is zero, so the
        # post-step trace advection/decay in run_substeps was a no-op and the
        # post-tick gas planes ARE step 2's Dalton input.
        for gi in trace_planes:
            assert not g.gas[gi].any(), \
                f"tick {tick}: trace plane {gi} non-zero — N̂ reconstruction invalid"

        # Reconstruct the step-4-entry (u, T): replay the advection substeps
        # on the snapshot (the P6.2-proven contract).
        dig_adv = bp.eos_sl_advect_ref(wx0, wy0, t0, g.solid, g.is_vacuum,
                                       g.dyn_permeability, dt, n_sub)
        if dig_adv != int(eos.digest_advect):
            bad += 1
            print(f"  tick {tick}: advection replay != solver digest_advect "
                  f"(ref={dig_adv:#018x} solver={int(eos.digest_advect):#018x})")

        inp = {"wind_x": wx0, "wind_y": wy0, "temperature": t0,
               "p_new": np.ascontiguousarray(g.atmosphere),
               "gas": np.ascontiguousarray(g.gas),
               "gas_conservative": np.ascontiguousarray(g.gases.conservative),
               "solid": g.solid, "is_vacuum": g.is_vacuum,
               "absorb": np.ascontiguousarray(g.dyn_wave_absorb)}
        f_ref, res_ref, f_gpu, res_gpu = _run_pair(inp, dt, c_local_q, consts)

        # The CPU reference must reproduce the REAL solver: digests, the
        # engine's own post-tick bytes, AND the per-tick counter deltas —
        # proves the input reconstruction and the reference itself.
        if res_ref[0] != int(eos.digest_velocity) or \
           res_ref[1] != int(eos.digest_compression):
            bad += 1
            print(f"  tick {tick}: CPU ref digests != solver "
                  f"(ref=({res_ref[0]:#018x}, {res_ref[1]:#018x}) solver="
                  f"({int(eos.digest_velocity):#018x}, "
                  f"{int(eos.digest_compression):#018x}) n_sub={n_sub})")
        if not (np.array_equal(f_ref["wind_x"], g.wind_x)
                and np.array_equal(f_ref["wind_y"], g.wind_y)
                and np.array_equal(f_ref["temperature"], g.temperature)):
            bad += 1
            print(f"  tick {tick}: CPU ref fields != engine post-tick fields")
        if tuple(int(v) for v in res_ref[2:]) != tuple(int(v) for v in cnt_delta):
            bad += 1
            print(f"  tick {tick}: CPU ref counters {tuple(res_ref[2:])} != "
                  f"solver deltas {tuple(cnt_delta)}")

        # The GPU chain must be bit-identical to the reference — fields,
        # digests, AND rail counters.
        if not _compare(f"tick {tick}", f_ref, res_ref, f_gpu, res_gpu):
            bad += 1
        if bad >= 10:
            print("  aborting after 10 divergences")
            break

    ok = (bad == 0)
    # The scenario must actually drive the tail HARD — a quiescent trajectory
    # (or one that never touches the rails) would make this gate vacuous.
    if max_n_sub < int(eos.N_SUB_MAX):
        ok = False
        print(f"  scenario too tame: max n_sub {max_n_sub} never hit "
              f"N_SUB_MAX={int(eos.N_SUB_MAX)}")
    if max_u_counts < 30 * FP_ONE:
        ok = False
        print(f"  scenario too tame: peak |u| {max_u_counts / FP_ONE:.1f} m/s "
              f"< 30 m/s")
    # Required in-engine rails: the |u| clamp (both counters — the hot pocket
    # pushes c_LOCAL past U_MAX, so engagements bind on U_MAX) and the work
    # clamp. The T_MIN/T_MAX_PHYS field rails do NOT fire in a pure
    # run_substeps loop (the v2.4 measured outcome: "rails untouched" in the
    # game-faithful loop — they need fire/combustion heat, which is not part
    # of this pass's surface); their digest + counter coverage is Part 1's
    # deterministic forcer.
    for i, name in enumerate(("u_clamp", "u_max", "work_clamp")):
        if totals[i] == 0:
            ok = False
            print(f"  scenario too tame: rail {name!r} never engaged in-engine")
    if ok:
        print(f"  {n_ticks} ticks bit-identical (per-tick digest_velocity/"
              f"digest_compression == CPU ref == GPU, counters matched; "
              f"peak |u| = {max_u_counts / FP_ONE:.1f} m/s, n_sub pinned at "
              f"{max_n_sub}; in-engine rail engagements: "
              + ", ".join(f"{n}={int(t)}" for n, t in zip(COUNTER_NAMES, totals))
              + ").")
    return ok


def part3_golden() -> bool:
    print("PART 3 — CUDA build's CPU path vs the committed golden:")
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
    GOLDEN = "28678e9d6210533f63cc701bba8f93194e23df9ebbdfa5f75f5d26681e897040"
    base = capture_trajectory(n_steps=30)
    dig = trajectory_digest(base)
    if dig != GOLDEN:
        print(f"  GOLDEN MISMATCH: {dig[:16]}... != {GOLDEN[:16]}...")
        return False
    print(f"  CUDA build CPU path reproduces the golden ({dig[:12]}...).")
    return True


def main() -> int:
    if not getattr(bp, "HAS_CUDA", False) or not bp.cuda_available():
        print("P64_RESULT: FAIL (no CUDA build / device)")
        return 1
    print("device:", bp.cuda_device_info())
    p1 = part1_isolated()
    p2 = part2_trajectory()
    p3 = part3_golden()
    if p1 and p2 and p3:
        print("P64_RESULT: PASS")
        return 0
    print("P64_RESULT: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
