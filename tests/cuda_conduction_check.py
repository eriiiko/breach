"""EOS P6.6 — unified temperature (conduction) bit-identity check (runs inside
the GPU subprocess).

Three gates, CPU reference = the real bound `TemperatureSolver` (the source of
truth), GPU = `bp.cuda_temperature_step`; both live in the one CUDA-build module.

  PART 1 — ISOLATED (synthetic, ALL branches + the T_MAX_PHYS rail FORCED):
  crafted configs that hit every path of the four passes —
    * Pass 0a zero-at-open-vacuum + Pass 0b semi-Lagrangian advection (a wind +
      dt>0 config so the DDA march / bilinear renorm run);
    * Pass 1 solid bit-shift deposit AND the open-air v2.4 absorption-∝-density
      reciprocal deposit across the three N regimes (N>=ambient exact path,
      floor<=N<ambient thin-gas absorption, N<floor near-vacuum), with the
      T_MAX_PHYS rail driven in BOTH branches (a near-ceiling solid + a huge
      deposit into ambient gas), the per-call rail-hit count asserted bit-equal;
    * Pass 2 conduction over solid-solid / air-air / solid-air interfaces;
    * Pass 3 cooling in both the interior and the vacuum-exposed shift.
  Plus the degenerate/edge masks: all-solid, all-open-vacuum, all-open-air,
  mixed, and 1xN / Nx1 grids. Each config: run BOTH backends on identical copies
  and assert byte-for-byte equality on `temperature` + bit-exact equality of the
  T_MAX_PHYS hit count.

  PART 2 — TRAJECTORY (the review's §4 P6.6 gate): a hot fire-core-vs-cold-hull
  scenario with thin / near-vacuum interior gas, a continuous per-tick heat
  deposit (constant re-injection), and a synthetic T_MAX_PHYS forcer cell, driven
  for 120 ticks. Each tick: run the CPU solver and the GPU kernel on identical
  copies of the evolving field, assert `temperature` byte-identity AND rail-hit
  parity, then advance the trajectory on the CPU result. The scenario is asserted
  to actually drive the passes hard (the rail engages every tick, conduction
  moves heat, cooling sheds it) so the gate is not vacuous.

  PART 3 — ENGINE DISPATCH A/B: flip PhysicsEngine's temperature pass between the
  CPU and GPU backends (set_temperature_backend) and assert the real 30-tick field
  trajectory is byte-identical — exercising the new step_tail dispatch
  (physics_engine.cpp) with the real n_bulk = O2+N2 sum. Also asserts the default
  scenario's CPU-path digest still equals the committed golden.

Prints ``P66_RESULT: PASS``/``FAIL`` and exits 0/1.
"""
from __future__ import annotations

import sys

import numpy as np

# Import the CUDA build FIRST so it is the cached `breach_physics`.
import breach_physics as bp

FP_ONE = 65536

# TemperatureSolver config defaults (temperature_solver.h) — the dials both
# backends run with. The CPU solver carries these as members; the GPU free
# function takes them explicitly. Kept identical so the two divide/clamp/cool
# with the same constants.
DIALS = dict(
    no_face=63, cool_shift=5, cool_shift_vacuum=3, o2_vacuum_thresh=0.3,
    c_v=1.0, n_floor_heat=0.05, gas_advection_rate=900.0, t_max_phys=16000.0,
)


def _q(x):
    """Round-to-nearest Q16.16 (matches fixedpoint::quantize)."""
    x = np.asarray(x, dtype=np.float64)
    return np.int32(np.where(x >= 0, np.floor(x * FP_ONE + 0.5),
                             np.ceil(x * FP_ONE - 0.5)))


def _make_solver():
    s = bp.TemperatureSolver()
    s.no_face = DIALS["no_face"]
    s.cool_shift = DIALS["cool_shift"]
    s.cool_shift_vacuum = DIALS["cool_shift_vacuum"]
    s.o2_vacuum_thresh = DIALS["o2_vacuum_thresh"]
    s.c_v = DIALS["c_v"]
    s.n_floor_heat = DIALS["n_floor_heat"]
    s.gas_advection_rate = DIALS["gas_advection_rate"]
    s.T_MAX_PHYS = DIALS["t_max_phys"]
    return s


def _build_face_shift(solid, is_vacuum, shift=3):
    """Symmetric per-tile face-shift cache (h,w,4) in dir order N,S,E,W. A face
    conducts (shift `shift`) iff BOTH cells participate in conduction — anything
    except an OPEN vacuum cell (solid, or open-air with air conductivity); grid
    edges and open-vacuum faces are NO_FACE. This exercises solid-solid, air-air,
    and solid-air interfaces at once. The value fed to both backends is the SAME
    array, so whatever it is the two must agree; `shift`>=3 keeps sum(rate)<=1/2
    (stable over the long trajectory)."""
    h, w = solid.shape
    NO = DIALS["no_face"]
    fs = np.full((h, w, 4), NO, dtype=np.int32)
    conductive = solid | (~solid & ~is_vacuum)   # everything except open vacuum
    DY = (-1, 1, 0, 0)
    DX = (0, 0, 1, -1)
    for d in range(4):
        for y in range(h):
            for x in range(w):
                ny, nx = y + DY[d], x + DX[d]
                if 0 <= ny < h and 0 <= nx < w and conductive[y, x] and conductive[ny, nx]:
                    fs[y, x, d] = shift
    return fs


# P-E2a/P-E2b: the SEVEN energy counters the conduction rewrite (P-E2a) and
# the Pass-1 attenuation drop (P-E2b) are gated on, in the pinned slot order
# of cuda_temperature.h / the C_* enum / the CPU field order.
E_COUNTERS = ("e_cond_trunc_sum", "e_cond_cap_sum", "cond_limit_hits",
              "e_cool_sum", "e_vac_wipe_sum", "e_ring_pin_sum",
              "e_deposit_drop_sum")


def _run_pair(solver, temperature, heat, his, fs, solid, is_vacuum,
              atmosphere, n_bulk, wind_x, wind_y, dt):
    """Run CPU reference + GPU kernel on identical copies; return
    (t_cpu, cnt_cpu, t_gpu, cnt_gpu) where cnt is
    (t_max_phys_hits,) + the seven P-E2a/P-E2b energy counters, all per-call."""
    t_cpu = np.ascontiguousarray(temperature.copy())
    c0 = (int(solver.t_max_phys_hits),) + tuple(
        int(getattr(solver, nm)) for nm in E_COUNTERS)
    solver.step(t_cpu, heat, his, fs, solid, is_vacuum, atmosphere,
                wind_x=wind_x, wind_y=wind_y, dt=dt, n_bulk=n_bulk)
    c1 = (int(solver.t_max_phys_hits),) + tuple(
        int(getattr(solver, nm)) for nm in E_COUNTERS)
    cnt_cpu = tuple(b - a for a, b in zip(c0, c1))

    t_gpu = np.ascontiguousarray(temperature.copy())
    # P-E2a/P-E2b: the isolated GPU entry returns (hits, *energy_counters).
    # NOT this file's sole caller — cuda_thermal_mass_check.py and
    # cuda_cool_shift_check.py also call it (P-E2a as-built §8.1) and are
    # updated in lockstep whenever this tuple grows.
    cnt_gpu = tuple(int(v) for v in bp.cuda_temperature_step(
        t_gpu, heat, his, fs, solid, is_vacuum, atmosphere,
        n_bulk=n_bulk, wind_x=wind_x, wind_y=wind_y, dt=dt, **DIALS))
    return t_cpu, cnt_cpu, t_gpu, cnt_gpu


def _compare(tag, t_cpu, cnt_cpu, t_gpu, cnt_gpu):
    ok = True
    if not np.array_equal(t_cpu, t_gpu):
        ok = False
        mism = int(np.count_nonzero(t_cpu != t_gpu))
        idx = int(np.argmax(t_cpu != t_gpu))
        print(f"  {tag}: temperature {mism} MISMATCH (first @ {idx}: "
              f"cpu={t_cpu.flat[idx]} gpu={t_gpu.flat[idx]})")
    names = ("t_max_phys_hits",) + E_COUNTERS
    for nm, a, b in zip(names, cnt_cpu, cnt_gpu):
        if a != b:
            ok = False
            print(f"  {tag}: {nm} mismatch cpu={a} gpu={b}")
    return ok


def _mask_grid(h, w, rng, kind):
    """Build (solid, is_vacuum) for an edge/mask config."""
    solid = np.zeros((h, w), dtype=bool)
    is_vacuum = np.zeros((h, w), dtype=bool)
    if kind == "all_solid":
        solid[:] = True
    elif kind == "all_vacuum":
        is_vacuum[:] = True
    elif kind == "all_air":
        pass
    elif kind == "mixed":
        solid = rng.random((h, w)) < 0.35
        is_vacuum = (~solid) & (rng.random((h, w)) < 0.25)
    elif kind == "hull":
        solid[0, :] = solid[-1, :] = solid[:, 0] = solid[:, -1] = True
        is_vacuum[0, :] = is_vacuum[-1, :] = True   # space-facing hull rows
    return np.ascontiguousarray(solid), np.ascontiguousarray(is_vacuum)


def _synth_fields(h, w, rng, solid, is_vacuum, thin=False, wind=False):
    n = h * w
    # temperature: spread incl. negatives, zeros, and near-ceiling values.
    t = rng.integers(-8 * FP_ONE, 40 * FP_ONE, size=n, dtype=np.int64).astype(np.int32)
    t[rng.random(n) < 0.15] = 0
    temperature = np.ascontiguousarray(t.reshape(h, w))

    # heat: non-negative saturating deposits, mostly small, some large.
    heat = _q(rng.random((h, w)) * 200.0)
    heat[rng.random((h, w)) < 0.4] = 0
    heat[rng.random((h, w)) < 0.05] = _q(30000.0)   # rail-forcing deposits
    heat = np.ascontiguousarray(np.maximum(heat, 0).astype(np.int32))

    his = np.ascontiguousarray(
        rng.integers(0, 5, size=(h, w)).astype(np.int32))   # log2(thermal_mass)

    # atmosphere: ambient inside, ~0 at vacuum (exposure + N proxy).
    atm = _q(np.where(is_vacuum, 0.0, 0.6 + rng.random((h, w)) * 0.8))
    atmosphere = np.ascontiguousarray(atm.astype(np.int32))

    # n_bulk: the real O2+N2 sum; thin/near-vacuum pockets when `thin`.
    if thin:
        nb = _q(rng.random((h, w)) * 1.2)
        nb[rng.random((h, w)) < 0.20] = _q(0.01)    # below N_FLOOR_HEAT
        nb[rng.random((h, w)) < 0.20] = _q(0.5)     # thin gas (absorption path)
    else:
        nb = _q(0.8 + rng.random((h, w)) * 0.5)
    n_bulk = np.ascontiguousarray(np.maximum(nb, 0).astype(np.int32))

    if wind:
        wx = _q((rng.random((h, w)) * 2 - 1) * 6.0)
        wy = _q((rng.random((h, w)) * 2 - 1) * 6.0)
        wind_x = np.ascontiguousarray(wx.astype(np.int32))
        wind_y = np.ascontiguousarray(wy.astype(np.int32))
        dt = 1.0 / 24.0
    else:
        wind_x = wind_y = None
        dt = 0.0

    fs = np.ascontiguousarray(_build_face_shift(solid, is_vacuum))
    return dict(temperature=temperature, heat=heat, his=his, fs=fs,
                atmosphere=atmosphere, n_bulk=n_bulk,
                wind_x=wind_x, wind_y=wind_y, dt=dt)


def _rail_forcer():
    """Deterministic 12x12 config engaging the T_MAX_PHYS rail in BOTH branches:
    a near-ceiling solid cell + a full-mass deposit, and an ambient-gas cell with
    a >ceiling deposit. Returns fields + expected (>0) hit count."""
    h = w = 12
    solid = np.zeros((h, w), dtype=bool)
    is_vacuum = np.zeros((h, w), dtype=bool)
    solid[3:9, 3:6] = True                         # a solid block
    temperature = np.zeros((h, w), dtype=np.int32)
    temperature[5, 4] = _q(15000.0)                # solid, near the 16000 ceiling
    heat = np.zeros((h, w), dtype=np.int32)
    heat[5, 4] = _q(5000.0)                         # 15000+5000 -> past ceiling (solid)
    heat[2, 8] = _q(30000.0)                        # ambient-gas cell -> past ceiling
    his = np.zeros((h, w), dtype=np.int32)         # shift 0 -> full deposit
    atmosphere = _q(np.ones((h, w)))               # ambient
    n_bulk = _q(np.ones((h, w)))                   # N == ambient -> exact deposit
    fs = _build_face_shift(solid, is_vacuum)
    return dict(
        temperature=np.ascontiguousarray(temperature),
        heat=np.ascontiguousarray(heat),
        his=np.ascontiguousarray(his),
        fs=np.ascontiguousarray(fs),
        solid=solid, is_vacuum=is_vacuum,
        atmosphere=np.ascontiguousarray(atmosphere),
        n_bulk=np.ascontiguousarray(n_bulk),
        wind_x=None, wind_y=None, dt=0.0), 2


def part1_isolated() -> bool:
    print("PART 1 — isolated GPU vs CPU reference (all branches + rail):")
    ok = True
    solver = _make_solver()
    rng = np.random.default_rng(20260711)
    total_hits = 0
    n_cfg = 0

    # (a) the deterministic rail forcer (both branches).
    f, expect_hits = _rail_forcer()
    t_cpu, hc, t_gpu, hg = _run_pair(
        solver, f["temperature"], f["heat"], f["his"], f["fs"], f["solid"],
        f["is_vacuum"], f["atmosphere"], f["n_bulk"], f["wind_x"], f["wind_y"], f["dt"])
    ok &= _compare("rail_forcer", t_cpu, hc, t_gpu, hg)
    total_hits += hc[0]
    e_touched = [abs(v) for v in hc[1:]]
    n_cfg += 1
    if hc[0] < expect_hits:
        ok = False
        print(f"  rail_forcer: T_MAX_PHYS rail under-engaged "
              f"(cpu hits={hc}, expected >= {expect_hits})")

    # (b) edge / mask configs across sizes, with thin gas + wind variants.
    sizes = [(16, 16), (24, 20), (1, 40), (40, 1), (13, 17), (8, 8)]
    kinds = ["all_solid", "all_vacuum", "all_air", "mixed", "hull"]
    for (h, w) in sizes:
        for kind in kinds:
            for thin in (False, True):
                for wind in (False, True):
                    solid, is_vacuum = _mask_grid(h, w, rng, kind)
                    fld = _synth_fields(h, w, rng, solid, is_vacuum,
                                        thin=thin, wind=wind)
                    t_cpu, hc, t_gpu, hg = _run_pair(
                        solver, fld["temperature"], fld["heat"], fld["his"],
                        fld["fs"], solid, is_vacuum, fld["atmosphere"],
                        fld["n_bulk"], fld["wind_x"], fld["wind_y"], fld["dt"])
                    tag = f"{h}x{w}/{kind}/thin={thin}/wind={wind}"
                    ok &= _compare(tag, t_cpu, hc, t_gpu, hg)
                    total_hits += hc[0]
                    e_touched = [a + abs(b) for a, b in zip(e_touched, hc[1:])]
                    n_cfg += 1

    if total_hits == 0:
        ok = False
        print("  COVERAGE HOLE: the T_MAX_PHYS rail never engaged in Part 1")
    # P-E2a: a counter that is 0 on every config proves nothing when compared.
    # Conduction's truncation, the capacity floor (thin-gas configs), the
    # cooling channel and the vacuum wipe must all be non-trivially exercised.
    for nm, tot in zip(E_COUNTERS, e_touched):
        if nm == "cond_limit_hits":
            if tot != 0:
                ok = False
                print(f"  the per-face limiter ENGAGED ({tot}) at shipped shifts")
            continue
        if nm == "e_ring_pin_sum":
            continue          # no ambient-ring mask in this isolated harness
        if tot == 0:
            ok = False
            print(f"  COVERAGE HOLE: {nm} stayed 0 across all {n_cfg} configs")
    if ok:
        print(f"  all {n_cfg} configs bit-identical on `temperature` + rail hits "
              f"+ the seven P-E2a/P-E2b energy counters (total rail engagements: "
              f"{total_hits}; |counters| {dict(zip(E_COUNTERS, e_touched))}).")
    return ok


def part2_trajectory() -> bool:
    print("PART 2 — hot-core vs cold-hull thin-gas trajectory (120 ticks):")
    solver = _make_solver()
    H = W = 40

    # A hull ring (solid), space-facing top/bottom rows (is_vacuum + solid), an
    # interior of open air, and a solid "fire core" block. Interior gas is thin
    # (drives the absorption-∝-density path + the per-tile reciprocal).
    solid = np.zeros((H, W), dtype=bool)
    is_vacuum = np.zeros((H, W), dtype=bool)
    solid[0, :] = solid[-1, :] = solid[:, 0] = solid[:, -1] = True
    is_vacuum[0, :] = is_vacuum[-1, :] = True        # radiate-to-space hull rows
    solid[18:23, 18:23] = True                       # the fire core (solid mass)

    fs = np.ascontiguousarray(_build_face_shift(solid, is_vacuum))
    his = np.ascontiguousarray(np.full((H, W), 2, dtype=np.int32))

    # Thin interior gas + ambient; vacuum rows ~0.
    atm = np.where(is_vacuum, 0.0, 0.9)
    atmosphere = np.ascontiguousarray(_q(atm).astype(np.int32))
    nb = np.where(is_vacuum, 0.0, 0.15)              # thin interior gas
    nb[10:14, 10:14] = 0.02                          # a near-vacuum pocket (< floor)
    n_bulk = np.ascontiguousarray(_q(nb).astype(np.int32))

    # Constant per-tick deposit: a hot core + a rail-forcing pocket in thin gas.
    heat = np.zeros((H, W), dtype=np.int32)
    heat[19:22, 19:22] = _q(400.0)                   # into the solid core
    heat[24:27, 24:27] = _q(300.0)                   # radiant into interior gas
    heat[11, 11] = _q(30000.0)                        # forces T_MAX_PHYS every tick
    heat = np.ascontiguousarray(heat)

    temperature = np.zeros((H, W), dtype=np.int32)
    temperature[19:22, 19:22] = _q(1000.0)           # a pre-warmed core

    n_ticks = 120
    bad = 0
    total_hits = 0
    max_T = 0
    moved = False
    for tick in range(n_ticks):
        t_cpu, hc, t_gpu, hg = _run_pair(
            solver, temperature, heat, his, fs, solid, is_vacuum,
            atmosphere, n_bulk, None, None, 0.0)
        if not _compare(f"tick {tick}", t_cpu, hc, t_gpu, hg):
            bad += 1
        total_hits += hc[0]
        max_T = max(max_T, int(np.abs(t_cpu).max()))
        # conduction must have spread heat off the deposit cells at least once.
        if int((t_cpu[17, 19] != 0)) or int((t_cpu[23, 20] != 0)):
            moved = True
        temperature = np.ascontiguousarray(t_cpu)   # advance on the CPU result
        if bad >= 10:
            print("  aborting after 10 divergences")
            break

    ok = (bad == 0)
    # The rail must engage on essentially every tick (a clamped cell re-clamps
    # each tick once warmed); allow a few ticks of warm-up before it latches.
    if total_hits < n_ticks - 5:
        ok = False
        print(f"  scenario too tame: rail engaged {total_hits} times "
              f"(< {n_ticks - 5} — expected ~every tick)")
    if not moved:
        ok = False
        print("  scenario too tame: conduction never spread heat off the core")
    if max_T < DIALS["t_max_phys"] * FP_ONE // 2:
        ok = False
        print(f"  scenario too tame: peak |T| {max_T} counts too low")
    if ok:
        print(f"  {n_ticks} ticks bit-identical on `temperature` + rail hits "
              f"+ the seven P-E2a/P-E2b energy counters "
              f"(total rail engagements {total_hits}, peak |T| "
              f"{max_T / FP_ONE:.0f} K-rel).")
    return ok


def part3_integration() -> bool:
    print("PART 3 — engine dispatch A/B (set_temperature_backend) + golden:")
    from field_ab_harness import (capture_trajectory, default_scenario_sim,
                                  diff_trajectories)
    from field_digest import trajectory_digest

    # The committed default-scenario golden (== cuda_kick_check part3 /
    # cuda_s*_check; last re-baselined 2026-07-10, eos-p3fix-thermal-ceiling).
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

    def make_hot():
        sim = default_scenario_sim()
        g = sim.gmap
        # Seed a temperature hotspot on solid tiles so conduction + cooling
        # evolve through the REAL engine step_tail dispatch (the default scenario
        # already seeds fire, the heat source for Pass 1).
        ys, xs = np.where(g.solid)
        for k in range(min(5, len(ys))):
            g.temperature[ys[k], xs[k]] = (50 + 10 * k) * FP_ONE
        return sim

    bp.set_temperature_backend(False)
    traj_cpu = capture_trajectory(make_sim=make_hot, n_steps=30)
    bp.set_temperature_backend(True)
    traj_gpu = capture_trajectory(make_sim=make_hot, n_steps=30)
    bp.set_temperature_backend(False)   # restore the CPU path

    diffs = diff_trajectories(traj_cpu, traj_gpu, tol=0.0)
    ok = (len(diffs) == 0)
    if not ok:
        print(f"  {len(diffs)} field divergence(s); first: {diffs[0]}")
    else:
        peak = max(int(np.abs(s["temperature"]).max()) for s in traj_cpu)
        print(f"  CPU vs GPU engine backend: bit-identical over 30 ticks "
              f"(peak |temperature| = {peak} counts).")

    # The default (un-seeded) scenario's CPU-backend digest must still match the
    # golden — proves the P6.6 dispatch/n_bulk-hoist changed no CPU trajectory.
    bp.set_temperature_backend(False)
    dig = trajectory_digest(capture_trajectory(n_steps=30))
    if dig != GOLDEN:
        ok = False
        print(f"  GOLDEN MISMATCH: {dig[:16]}... != {GOLDEN[:16]}...")
    else:
        print(f"  CPU-path default-scenario golden intact ({dig[:12]}...).")
    return ok


def main() -> int:
    if not getattr(bp, "HAS_CUDA", False) or not bp.cuda_available():
        print("P66_RESULT: FAIL (no CUDA build / device)")
        return 1
    print("device:", bp.cuda_device_info())
    p1 = part1_isolated()
    p2 = part2_trajectory()
    p3 = part3_integration()
    if p1 and p2 and p3:
        print("P66_RESULT: PASS")
        return 0
    print("P66_RESULT: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
