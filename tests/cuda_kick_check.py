"""EOS P6.4 — momentum kick + compression work bit-identity check (runs inside
the GPU subprocess).

Three gates:

  PART 1 — ISOLATED (synthetic, all branches + ALL FIVE RAILS FORCED): crafted
  and random inputs that hit every branch of the step-4/4c tail — the
  zero-gradient fast path, the kick's reciprocal chain against floored and
  ambient N̂, absorption a=0 / 0<a<1 / a>=1 (kk=0), the ±2^30 RAD_SAFE
  component pre-clamp, the counted |u| clamp with BOTH binding caps
  (VELOCITY-CLAMP, P-V1, D2v2: a per-cell cap2_plane, uniform per config here
  — an ambient-scale plane where the ambient cap binds; a U_MAX² plane where
  the U_MAX rail binds), the 4c work clamp on both signs, the T_MIN energy
  floor and the T_MAX_PHYS ceiling, solid/vacuum cells, trace-plane Dalton
  weighting, degenerate 1xN / Nx1 grids. Run BOTH the GPU chain
  (bp.cuda_eos_kick_compression) and the CPU reference
  (bp.eos_kick_compression_ref — the step-4/4c loops copied line for line
  from EOSSolver::step) on identical copies and assert byte-for-byte equality
  on (wind_x, wind_y, temperature) + digest equality + BIT-EXACT equality of
  all five rail counters. The config set is asserted to engage every counter
  at least once (u_clamp, u_max, work_clamp, energy_floor, t_max_phys) — the
  rails and their telemetry are digest-covered, not just the happy path.

  PART 2 — TRAJECTORY (the review's §4 P6.4 digest gate). REPAIRED at P-E4
  (design §6 P-E4 row, item C) to its NEW, TRUE contract — the original
  premise (reconstruct the exact step-4-entry TEMPERATURE by replaying the
  advection substeps, then check the isolated tail against the real
  engine's post-tick bytes) died at P-E1: `eos_sl_advect_reference` became
  u-only (temperature is read-only there now, its `.t` slot discarded) and
  `digest_advect` moved to hash (wx, wy, T-AFTER-RECOVERY) — T evolution now
  happens entirely inside the energy-transport pass, which has no isolated
  Python entry point at this granularity. P-E2a's investigation (as-built
  §8.3) found this via revert+rebuild: PART 2 was ALREADY diverging on that
  base, from tick 1, digest_compression only — digest_velocity matched
  exactly. That is the key fact the repair rests on: step 4 (the momentum
  kick) reads pressure/N/absorb and NEVER reads temperature, so wind
  reconstruction is completely unaffected by T's now-stale-by-one-tick
  reconstruction — only step 4c (compression work) consumes T, and its
  output is what can no longer be checked against ground truth this way.
  P-E1's own gate (`cuda_bulk_flux_check` PART 3) already proves the energy-
  transport pass that produces the TRUE step-4-entry T is bit-identical
  CPU<->GPU and closes the energy books every tick — so T-side coverage
  was relocated to its rightful owner, not lost.

  A blast + breach venting scenario (a 5000 K core + O2 overpressure, PLUS a
  near-ceiling 15500 K pocket so the v2.4 rails engage in-engine) driven
  through the REAL engine path (PhysicsEngine.run_substeps -> EOSSolver::step)
  for 120 ticks on the CPU. Per tick: snapshot the step-1-entry (wind, T)
  state and the solver's cumulative rail counters, run the real tick, replay
  the advection substeps on the snapshot to reconstruct step-4-entry WIND
  (eos_sl_advect_ref + dbg_last_n_sub — no digest comparison; see above),
  then replay the isolated kick+compression on (that wind, the STALE t0,
  p_new = the post-tick atmosphere, N̂ from the post-tick gas planes — valid
  because run_substeps' post-step trace advection/decay only moves NON-ZERO
  trace planes and this scenario keeps every trace plane zero, asserted;
  cap2_plane REBUILT FROM t0 via formula A — VELOCITY-CLAMP, P-V1, D2v2: t0
  IS tick-entry T, never mutated, so the rebuild stays exact and this gate is
  RESTORED to full validity, not merely patched around the T-staleness below
  — the scenario has no ts-gas cells, so ts == solid on both sides) through
  BOTH the CPU reference and the GPU chain, asserting
      ref digest_velocity          == EOSSolver.digest_velocity
      ref wind_x/wind_y            == the engine's own post-tick wind bytes
      ref T-INDEPENDENT counters   == the solver's cumulative deltas
                                       (u_clamp, u_max, work_clamp — the
                                       SS2.4 fade + the ±T_WORK_CLAMP compare
                                       key off N/divergence, never T)
      GPU == ref on EVERYTHING — fields, BOTH digests, ALL rail counters
                                       (the core P6.4 proof: unaffected by
                                       t0's staleness, since CPU ref and GPU
                                       are fed the identical input)
  — a per-tick wind-side ground-truth trajectory PLUS a full CPU<->GPU
  bit-identity trajectory, over the whole run. The scenario is asserted to
  actually drive the tail hard (n_sub pins at N_SUB_MAX, the counted |u|
  clamp engages, the work clamp engages). VELOCITY-CLAMP (P-V1, D2v2): u_max
  is NOT required to engage here any more — pre-D2v2 the near-ceiling hot
  pocket alone pushed the (global-scalar) cap to U_MAX for the WHOLE map
  regardless of a cell's own temperature (audit defect 1); post-D2v2 the cap
  is per-cell, so u_max only binds where a cell is both hot AND carries fast
  local flow, which this static hot pocket (no local pressure driver) never
  does — see the assertion below for the full rationale. The T_MIN/T_MAX_PHYS
  field rails do not fire in a pure run_substeps loop (the v2.4 measured
  outcome — they need fire/combustion heat, outside this pass's surface);
  their coverage is Part 1's deterministic forcer.

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
    t_max_phys=16000.0, u_max=1000.0,
    # trace_mass_scale key RETIRED (P-T0, design §2.6)
    # P-E3 (design §2.8): interior drag + heat counterparty. Shipped
    # defaults here (k_drag=0.0) so the base CONSTS dict stays DORMANT —
    # every Part 1/2 config below therefore ALSO doubles as drag-dormancy
    # CPU<->GPU parity coverage. The dedicated drag-active config below
    # (CONSTS_DRAG) is what exercises the mechanism itself.
    k_drag=0.0, k_drag_heat_frac=1.0, c_v=1.0,
)

# A drag-ACTIVE variant: k_drag > 0 (dormancy branch open) and
# k_drag_heat_frac < 1 (so e_drag_drop_sum is non-vacuous too).
CONSTS_DRAG = dict(CONSTS, k_drag=0.02, k_drag_heat_frac=0.5)

COUNTER_NAMES = ("u_clamp", "u_max", "work_clamp", "energy_floor", "t_max_phys",
                 "ke_drag_removed", "e_drag_deposit", "e_drag_drop_sum",
                 "e_drag_rail_clipped")


def _quantize(x):
    """Round-to-nearest Q16.16 (matches fixedpoint::quantize)."""
    x = np.asarray(x, dtype=np.float64)
    return np.int32(np.where(x >= 0, np.floor(x * FP_ONE + 0.5),
                             np.ceil(x * FP_ONE - 0.5)))


def _run_pair(inp, dt, cap2_plane, consts):
    """Run CPU reference + GPU chain on identical copies; return everything.

    VELOCITY-CLAMP (P-V1, D2v2): `cap2_plane` drops into the old `c_local_q`
    positional slot as an (h,w) int64 array (Q32.32 raw, >= 0 everywhere —
    the hard contract both bindings document).
    """
    args_tail = (inp["p_new"], inp["gas"], inp["gas_conservative"],
                 inp["solid"], inp["is_vacuum"], inp["absorb"])
    cap2_plane = np.ascontiguousarray(cap2_plane, dtype=np.int64)
    f_ref = {k: inp[k].copy() for k in ("wind_x", "wind_y", "temperature")}
    res_ref = bp.eos_kick_compression_ref(
        f_ref["wind_x"], f_ref["wind_y"], f_ref["temperature"],
        *args_tail, dt, cap2_plane, **consts)
    f_gpu = {k: inp[k].copy() for k in ("wind_x", "wind_y", "temperature")}
    res_gpu = bp.cuda_eos_kick_compression(
        f_gpu["wind_x"], f_gpu["wind_y"], f_gpu["temperature"],
        *args_tail, dt, cap2_plane, **consts)
    return f_ref, res_ref, f_gpu, res_gpu


def _uniform_cap2(h, w, cap_val):
    """A uniform (h,w) int64 cap2 plane at the given real-units cap value —
    Part 1's constructed-config idiom (D5: the kick trusts the plane
    verbatim, so a uniform fill reproduces the old uniform-scalar regimes
    exactly)."""
    cap_q = int(_quantize(cap_val))
    return np.full((h, w), cap_q * cap_q, dtype=np.int64)


def _fold_cap2_plane(temperature, solid, is_vacuum, ts, eos):
    """VELOCITY-CLAMP (P-V1, D2v2) formula A, ported to Python verbatim (the
    eos_solver.cpp / cuda_eos_step.cu scan) — used by PART 2 to rebuild the
    exact per-tick cap plane from `temperature` (must be tick-entry T, never
    mutated after capture — the plane-from-t0 rebuild is what restores PART
    2's wind-side gate to full validity under D2v2).

    c_amb2*ratio reaches ~2^68 and WRAPS SILENTLY in numpy int64 (the same
    hazard fixed_point.h's mul128_shr sidesteps in C++) — the fold multiply
    runs in PLAIN PYTHON INTS (arbitrary precision); only the masking and the
    ratio_umax comparison are vectorized (ratio itself is bounded by
    ratio_umax, ~7e5, safely int64-representable).
    """
    s_eos_q = int(_quantize(eos.S_EOS))
    t_amb_q = max(1, int(_quantize(eos.T_AMB_K)))
    c_amb_q = int(_quantize(eos.c_max))
    u_max_q = int(_quantize(eos.U_MAX))
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


def _make_drag_forcer(h=24, w=24):
    """P-E3 (design §2.8) drag-rail forcer: substantial velocity everywhere
    (real KE to remove), a near-ceiling hot band (a sustained drag deposit
    there clips at T_MAX_PHYS — e_drag_rail_clipped), ordinary ambient N
    (n_bulk >= 1, so the deposit WRITES most places), and a near-vacuum strip
    (n_bulk < 1 raw count — the phantom-T guard: still priced, n-weighted to
    ~0, T never written there)."""
    inp = {
        "wind_x": np.full((h, w), _quantize(900.0), dtype=np.int32),
        "wind_y": np.full((h, w), _quantize(-900.0), dtype=np.int32),
        "temperature": np.full((h, w), _quantize(100.0), dtype=np.int32),
        "p_new": np.full((h, w), _quantize(1.0), dtype=np.int32),
        "gas": np.zeros((3, h, w), dtype=np.int32),
        "gas_conservative": np.array([True, True, False]),
        "solid": np.zeros((h, w), dtype=bool),
        "is_vacuum": np.zeros((h, w), dtype=bool),
        "absorb": np.zeros((h, w), dtype=np.float32),
    }
    inp["gas"][0][:, :] = _quantize(0.21)
    inp["gas"][1][:, :] = _quantize(0.79)
    # Near-ceiling hot band: the drag deposit here should clip at T_MAX_PHYS.
    inp["temperature"][8:16, :] = _quantize(15990.0)
    # Near-vacuum strip: n_bulk < 1 raw count -> the phantom-T guard engages.
    inp["gas"][0][20:24, :] = 0
    inp["gas"][1][20:24, :] = 0
    for k in inp:
        inp[k] = np.ascontiguousarray(inp[k])
    return inp


def part1_isolated() -> bool:
    print("PART 1 — isolated GPU vs CPU reference (synthetic, all rails):")
    ok = True
    rng = np.random.default_rng(20260711)
    totals = np.zeros(9, dtype=np.int64)   # ref counter engagement coverage
    n_cfg = 0

    # (a) the deterministic all-rails forcer, both cap regimes — VELOCITY-
    # CLAMP (P-V1, D2v2): two constructed UNIFORM planes (D5: the kick
    # trusts the plane verbatim). A u_max2 plane reaches the U_MAX rail (the
    # forcer's RAD_SAFE-clamped seeds far exceed 1000 m/s — reachable); a
    # c_amb2 plane keeps the ambient cap binding (u_max_hits == 0).
    RAIL_FORCER_H, RAIL_FORCER_W = 24, 24
    for cap_val, tag, is_umax in ((CONSTS["u_max"], "forcer u_max2 plane (U_MAX binds)", True),
                                  (CONSTS["c_max"], "forcer c_amb2 plane (ambient binds)", False)):
        n_cfg += 1
        inp = _make_rail_forcer()
        cap2 = _uniform_cap2(RAIL_FORCER_H, RAIL_FORCER_W, cap_val)
        f_ref, res_ref, f_gpu, res_gpu = _run_pair(inp, 1.0 / 24.0, cap2, CONSTS)
        ok &= _compare(tag, f_ref, res_ref, f_gpu, res_gpu)
        totals += np.array(res_ref[2:], dtype=np.int64)
        if is_umax:
            if res_ref[2] == 0 or res_ref[3] == 0:
                ok = False
                print(f"  {tag}: U_MAX rail did NOT engage "
                      f"(u_clamp={res_ref[2]} u_max={res_ref[3]})")
        else:
            if res_ref[2] == 0 or res_ref[3] != 0:
                ok = False
                print(f"  {tag}: ambient cap regime wrong "
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
            # VELOCITY-CLAMP (P-V1, D2v2): a uniform plane per config (D5:
            # trusted verbatim — no implicit re-min against U_MAX here, so
            # a > u_max fill value legitimately means an ambient cap ABOVE
            # U_MAX, exercised deliberately by several of the configs above).
            cap2 = _uniform_cap2(h, w, c_local)
            f_ref, res_ref, f_gpu, res_gpu = _run_pair(inp, dt, cap2, CONSTS)
            ok &= _compare(f"{h}x{w} dt={dt} wmag={wmag} c_loc={c_local}",
                           f_ref, res_ref, f_gpu, res_gpu)
            totals += np.array(res_ref[2:], dtype=np.int64)

    # (c) P-E3 (design §2.8): the dedicated drag-active forcer, both cap
    # regimes — CONSTS_DRAG (k_drag=0.02, k_drag_heat_frac=0.5) so all four
    # new counters engage (ke_drag_removed, e_drag_deposit non-vacuous from
    # the heat_frac<1 split; e_drag_drop_sum non-vacuous for the same
    # reason; e_drag_rail_clipped from the near-ceiling hot band). The two
    # regimes use a c_amb2 plane and a u_max2 plane DIRECTLY (D5: NOT a
    # 2300²-style plane — the drag forcer's ~1272 m/s is below 2300, so
    # that literal fill would DISENGAGE the clamp under D5's no-re-min
    # contract and lose the U_MAX-regime coverage the old min(c_local,
    # u_max)=u_max semantics gave for free).
    DRAG_FORCER_H, DRAG_FORCER_W = 24, 24
    for cap_val, tag in ((CONSTS["c_max"], "drag forcer c_amb2 plane (ambient binds)"),
                         (CONSTS["u_max"], "drag forcer u_max2 plane (U_MAX binds)")):
        n_cfg += 1
        inp = _make_drag_forcer()
        cap2 = _uniform_cap2(DRAG_FORCER_H, DRAG_FORCER_W, cap_val)
        f_ref, res_ref, f_gpu, res_gpu = _run_pair(
            inp, 1.0 / 24.0, cap2, CONSTS_DRAG)
        ok &= _compare(tag, f_ref, res_ref, f_gpu, res_gpu)
        totals += np.array(res_ref[2:], dtype=np.int64)
        if res_ref[7] == 0 or res_ref[8] == 0 or res_ref[9] == 0 or res_ref[10] == 0:
            ok = False
            print(f"  {tag}: drag rails did not all engage "
                  f"(ke_removed={res_ref[7]} deposit={res_ref[8]} "
                  f"drop={res_ref[9]} clipped={res_ref[10]})")

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
        # trace_mass_scale key RETIRED (P-T0, design §2.6) — the EOSSolver
        # member is gone; eos.trace_mass_scale no longer exists to read.
        # P-E3 (design §2.8) BUGFIX (VELOCITY-CLAMP arc investigation,
        # 2026-08-19): read the LIVE k_drag/k_drag_heat_frac/c_v from `eos`
        # instead of leaving them at the pybind dormant default — the shipped
        # config ([physics.eos] k_drag = 0.5) has drag ACTIVE by default, so
        # the old "matches the live engine's shipped default" comment here
        # was stale (true only before P-E3 shipped a nonzero k_drag) and the
        # CPU reference was silently running drag-dormant against a
        # drag-ACTIVE live solver — the actual source of this gate's
        # wind-side digest_velocity/wind_x/wind_y divergence (confirmed by
        # reproducing it bit-for-bit on the UNMODIFIED pre-P-V1 tree; the
        # velocity-clamp per-cell cap plane was not involved). Pre-existing,
        # unrelated to D2v2 — fixed here because it was silently defeating
        # THIS gate's wind-side ground-truth check.
        k_drag=float(eos.k_drag), k_drag_heat_frac=float(eos.k_drag_heat_frac),
        c_v=float(eos.c_v),
        # P-E4 (design §2.4): the trust gate's reference density.
        n_work_ref=float(eos.n_work_ref),
        # T_ABS COMPRESSION WORK (P-W1a, design §5): thread the solver's own
        # T_AMB_K explicitly — a silent pybind default here would compare a
        # defaulted reference against a solver-valued device path, exactly
        # the quiet hole this lockstep gate exists to close.
        t_amb_k=float(eos.T_AMB_K),
    )
    trace_planes = [gi for gi in range(g.gas.shape[0])
                    if not bool(g.gases.conservative[gi])]

    def counters():
        # u_clamp/u_max/work_clamp/energy_floor/t_max_phys: CUMULATIVE (the
        # caller diffs two snapshots). ke_drag_removed/e_drag_deposit/
        # e_drag_drop_sum/e_drag_rail_clipped (P-E3, design §2.8): PER-TICK
        # (reset at every step() entry) — at this scenario's default
        # k_drag=0.0 (dormant, matching the live engine everywhere else)
        # these four stay 0 throughout, so a plain diff is harmless here,
        # but the caller still takes them as a direct per-tick READ (not a
        # diff) to stay correct in general — see the loop below.
        return np.array([eos.u_clamp_hits, eos.u_max_hits,
                         eos.work_clamp_hits, eos.energy_floor_hits,
                         eos.t_max_phys_hits,
                         eos.ke_drag_removed, eos.e_drag_deposit,
                         eos.e_drag_drop_sum, eos.e_drag_rail_clipped],
                        dtype=np.int64)

    n_ticks = 120
    max_n_sub = 0
    max_u_counts = 0
    totals = np.zeros(9, dtype=np.int64)
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
        cnt_now = counters()
        cnt_delta = cnt_now - cnt0
        # P-E3 (design §2.8): indices 5..8 are PER-TICK (reset every
        # step() entry) — READ directly rather than diffed against cnt0
        # (which holds the PREVIOUS tick's per-tick snapshot, not a base to
        # subtract from). Harmless at this scenario's k_drag=0.0, correct
        # in general.
        cnt_delta[5:] = cnt_now[5:]
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

        # Reconstruct the step-4-entry WIND: replay the advection substeps
        # on the snapshot (the P6.2-proven contract, still exactly valid —
        # advection/kick never depended on T even before P-E1).
        #
        # P-E4 REPAIR (design §6 P-E4 row, item C — inherited debt from
        # P-E1, verified by revert+rebuild at P-E2a §8.3): this call used to
        # ALSO be compared against `eos.digest_advect`. That comparison is
        # now STRUCTURALLY invalid, not merely stale: P-E1 made
        # `eos_sl_advect_reference` u-only (`temperature` is read-only, its
        # `.t` slot discarded) and moved `digest_advect` to hash
        # (wx, wy, T-AFTER-RECOVERY) — a 3-field chained digest — whereas
        # this replay returns a 2-field (wx, wy)-only digest. No wind/T
        # values were ever compared by that check; two DIFFERENTLY-SHAPED
        # digests were being compared for equality, which cannot pass by
        # construction. Dropped, not worked around.
        bp.eos_sl_advect_ref(wx0, wy0, t0, g.solid, g.is_vacuum,
                             g.dyn_permeability, dt, n_sub)
        # `t0` is INTENTIONALLY left as the tick-START temperature — P-E1
        # retired the SL T-copy, so nothing advects it any more (T evolution
        # now happens entirely inside the energy-transport pass, which has
        # no isolated Python entry point at this granularity). `t0` is
        # therefore NOT the true step-4-entry T from here on (it is missing
        # this tick's own transport delta) — this is the SAME thing that
        # made P-E2a's investigation find PART 2 diverging from tick 1.

        # VELOCITY-CLAMP (P-V1, D2v2): rebuild the per-cell cap2 plane from
        # t0 (tick-entry T, never mutated by the advection replay above —
        # see the comment there) via formula A — this is what RESTORES the
        # wind-side ground-truth gate to full validity: the cap is now
        # derivable from the replay's own inputs, not a telemetry scalar
        # (dbg_last_c_local_q) the isolated tail can no longer even use.
        # The scenario has no ts-gas cells (no thermal_solid plane), so
        # ts == solid on both sides (D4's fallback).
        cap2 = _fold_cap2_plane(t0, g.solid, g.is_vacuum, g.solid, eos)

        inp = {"wind_x": wx0, "wind_y": wy0, "temperature": t0,
               "p_new": np.ascontiguousarray(g.atmosphere),
               "gas": np.ascontiguousarray(g.gas),
               "gas_conservative": np.ascontiguousarray(g.gases.conservative),
               "solid": g.solid, "is_vacuum": g.is_vacuum,
               "absorb": np.ascontiguousarray(g.dyn_wave_absorb)}
        f_ref, res_ref, f_gpu, res_gpu = _run_pair(inp, dt, cap2, consts)

        # THE NEW, TRUE CONTRACT (repair, see above): step 4 (the momentum
        # kick) reads pressure/N/absorb — NEVER temperature — so its output
        # is independent of t0's staleness and MUST still reproduce the
        # real solver exactly: digest_velocity, wind_x, wind_y, and the
        # T-independent rail counters (u_clamp, u_max, work_clamp — the
        # SS2.4 trust-gate fade and the ±T_WORK_CLAMP compare both key off N
        # and the divergence, never T, so work_clamp_hits is T-independent
        # too). Step 4c (compression work) DOES read T, so its output
        # (digest_compression, the temperature field, energy_floor_hits,
        # t_max_phys_hits) is downstream of the now-unreconstructible t0 and
        # is NO LONGER checked against the real solver here — P-E1's own
        # gate (`cuda_bulk_flux_check` PART 3) already proves the energy-
        # transport pass that actually produces the true step-4-entry T is
        # bit-identical CPU<->GPU and closes the energy books every tick;
        # that coverage was never lost, it lives at its rightful owner.
        if res_ref[0] != int(eos.digest_velocity):
            bad += 1
            print(f"  tick {tick}: CPU ref digest_velocity != solver "
                  f"(ref={res_ref[0]:#018x} solver={int(eos.digest_velocity):#018x} "
                  f"n_sub={n_sub})")
        if not (np.array_equal(f_ref["wind_x"], g.wind_x)
                and np.array_equal(f_ref["wind_y"], g.wind_y)):
            bad += 1
            print(f"  tick {tick}: CPU ref wind != engine post-tick wind")
        wind_counter_names = ("u_clamp", "u_max", "work_clamp")
        wind_counters_ref = tuple(int(v) for v in res_ref[2:5])
        wind_counters_solver = tuple(int(v) for v in cnt_delta[0:3])
        if wind_counters_ref != wind_counters_solver:
            bad += 1
            print(f"  tick {tick}: CPU ref T-independent counters "
                  f"{dict(zip(wind_counter_names, wind_counters_ref))} != "
                  f"solver deltas "
                  f"{dict(zip(wind_counter_names, wind_counters_solver))}")

        # The GPU chain must be bit-identical to the CPU reference on
        # EVERYTHING — fields, digests, and ALL rail counters (including the
        # T-dependent ones): both are fed the SAME (possibly-stale-T) input,
        # so their outputs must agree with EACH OTHER regardless of whether
        # that input matches the real engine. This remains the core P6.4
        # proof, untouched by the repair above.
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
    # Required in-engine rails: the |u| clamp and the work clamp. The
    # T_MIN/T_MAX_PHYS field rails do NOT fire in a pure run_substeps loop
    # (the v2.4 measured outcome: "rails untouched" in the game-faithful
    # loop — they need fire/combustion heat, which is not part of this
    # pass's surface); their digest + counter coverage is Part 1's
    # deterministic forcer.
    #
    # VELOCITY-CLAMP (P-V1, D2v2) — u_max is DELIBERATELY not required here
    # any more. Pre-D2v2, "u_max" bound trivially: c_LOCAL was a GLOBAL
    # scalar, so the far-away near-ceiling hot pocket alone pushed EVERY
    # cell's ceiling to U_MAX regardless of that cell's own temperature —
    # exactly the audit's defect 1. Post-D2v2 the cap is per-cell, so u_max
    # only binds where a cell is BOTH hot enough (T ≳ 2930 K, design's own
    # claims section) AND carries fast local flow — this scenario's hot
    # pocket is a static thermal anomaly with no local pressure driver, so
    # it never engages the rail, which is now CORRECT behavior, not a
    # scenario weakness (P-V2's job is to measure how often this combination
    # occurs in a real blast, not this gate's).
    for name in ("u_clamp", "work_clamp"):
        i = COUNTER_NAMES.index(name)   # totals is 9-wide, positional — NOT
                                        # the (shortened) loop tuple's own index
        if totals[i] == 0:
            ok = False
            print(f"  scenario too tame: rail {name!r} never engaged in-engine")
    if ok:
        print(f"  {n_ticks} ticks: wind-side ground truth held (per-tick "
              f"digest_velocity + wind_x/wind_y + T-independent counters == "
              f"solver) AND CPU ref == GPU on EVERYTHING (both digests, all "
              f"fields, all rail counters); peak |u| = "
              f"{max_u_counts / FP_ONE:.1f} m/s, n_sub pinned at {max_n_sub}; "
              f"in-engine rail engagements: "
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
