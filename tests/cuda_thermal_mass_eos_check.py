"""THERMAL-MASS AXIS **P-EOS** gate (d) — the CPU<->CUDA lockstep proof.

Gate (d) of docs/thermal_mass_eos_ruling_2026-07-30.md §3: the GPU mirror of the
ruling's EOS edits — step-1b's SKIPPED `temperature` write + its T-ONLY occluder
mask, step-4c's skipped write, and the combustion deposit's OBJECT conversion —
is bit-identical to the CPU at TOLERANCE ZERO on a FURNITURE-BURN scenario (the
case where `thermal_solid != solid` and the pre-patch kernels were wrong), for
BOTH the per-call step path and the GPU-resident path.

Sibling of tests/cuda_thermal_mass_check.py (P2, the TemperatureSolver mirror);
this file gates the EOS/combustion kernels instead. Same scenario builder, same
tol-0 comparison, same non-vacuousness discipline.

  PART 1 — ISOLATED KERNEL lockstep + the NON-VACUOUSNESS CONTROLS. For each of
    the three touched kernels the GPU entry is driven WITH the mask (must match
    the CPU reference driven with the same mask, tol 0) and WITHOUT it (must
    DIVERGE — otherwise the mask is not reaching the kernel and the gate proves
    nothing):
      * SL advection      cuda_eos_sl_advect        vs eos_sl_advect_ref
      * kick+compression  cuda_eos_kick_compression vs eos_kick_compression_ref
      * combustion        cuda_combustion_step      vs CombustionSolver.step
  PART 2 — FURNITURE-BURN engine A/B, STEP path: the same burning-crate world
    built twice and stepped 40 ticks through the real PhysicsRunner, with ONLY
    the EOS + combustion backends flipped (so any divergence is one of the
    kernels this patch touched). All synced fields + the 5 EOS rail counters
    compared per tick, tol 0, plus the dispatch-fired and burn-activity
    vacuousness guards.
  PART 3 — same scenario, RESIDENT path: CPU (residency off, all backends off)
    vs residency ON + every backend ON. Per-tick tol 0, the resident EOS is
    asserted to have actually run, and the resident device set is asserted to
    carry `thermal_solid` (which the resident SL/compression kernels now READ,
    so it also rides the per-tick from_host upload).

Prints ``THERMAL_MASS_EOS_RESULT: PASS``/``FAIL`` and exits 0/1.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import breach_physics as bp   # the CUDA build (sys.path[0] == cpp/build_cuda)

FP_ONE = 65536


def _q(x):
    """Round-to-nearest Q16.16 (matches fixedpoint::quantize)."""
    x = np.asarray(x, dtype=np.float64)
    return np.int32(np.where(x >= 0, np.floor(x * FP_ONE + 0.5),
                             np.ceil(x * FP_ONE - 0.5)))


def _c(a):
    return np.ascontiguousarray(a)


# ===========================================================================
# PART 1 — isolated kernels + the mask-is-live controls
# ===========================================================================
def _iso_world(h, w, rng, crate=True):
    """A hull-shelled box with an air interior, a vacuum band, and (optionally) a
    FURNITURE block — the only material that is permeable AND thermally solid, so
    it is the only place the thermal and gas media diverge."""
    solid = np.zeros((h, w), dtype=bool)
    solid[0, :] = solid[-1, :] = solid[:, 0] = solid[:, -1] = True
    solid[h // 3, 1:w - 3] = True                      # an interior wall
    is_vacuum = np.zeros((h, w), dtype=bool)
    is_vacuum[:, w - 2:] = True                        # a vacuum band
    thermal_solid = solid.copy()
    perm = np.where(solid, 0.0, 1.0).astype(np.float32)
    furn = np.zeros((h, w), dtype=bool)
    if crate:
        furn[h // 2:h // 2 + 3, w // 2 - 2:w // 2 + 2] = True
        furn &= ~solid & ~is_vacuum
        thermal_solid |= furn
        perm[furn] = 0.5                               # shield, not seal
    return solid, is_vacuum, thermal_solid, perm, furn


def _iso_fields(h, w, rng, solid, is_vacuum):
    t = _q(rng.uniform(-50.0, 900.0, size=(h, w)))
    t[is_vacuum & ~solid] = 0
    wx = _q(rng.uniform(-6.0, 6.0, size=(h, w)))
    wy = _q(rng.uniform(-6.0, 6.0, size=(h, w)))
    wx[solid] = 0
    wy[solid] = 0
    return _c(t), _c(wx), _c(wy)


def part1_isolated() -> bool:
    print("PART 1 — ISOLATED kernel lockstep + mask-is-live controls:")
    ok = True

    # ---------------- 1a: SL advection (step-1b) ---------------------------
    n_cfg = n_div = 0
    for size in (16, 24, 33):
        for seed in (11, 22, 33):
            rng = np.random.default_rng(seed * 1000 + size)
            h = w = size
            solid, vac, tsol, perm, furn = _iso_world(h, w, rng)
            if not furn.any():
                continue
            t, wx, wy = _iso_fields(h, w, rng, solid, vac)
            for n_sub in (1, 4):
                n_cfg += 1
                a = (t.copy(), wx.copy(), wy.copy())
                b = (t.copy(), wx.copy(), wy.copy())
                c = (t.copy(), wx.copy(), wy.copy())
                d_cpu = bp.eos_sl_advect_ref(
                    a[1], a[2], a[0], solid, vac, perm,
                    dt=1.0 / 24.0, n_sub=n_sub, thermal_solid=tsol)
                d_gpu = bp.cuda_eos_sl_advect(
                    b[1], b[2], b[0], solid, vac, perm,
                    dt=1.0 / 24.0, n_sub=n_sub, thermal_solid=tsol)
                # the MASK-OMITTED control (the pre-patch behaviour)
                bp.cuda_eos_sl_advect(
                    c[1], c[2], c[0], solid, vac, perm,
                    dt=1.0 / 24.0, n_sub=n_sub, thermal_solid=None)
                if d_cpu != d_gpu or not np.array_equal(a[0], b[0]) \
                        or not np.array_equal(a[1], b[1]) \
                        or not np.array_equal(a[2], b[2]):
                    ok = False
                    print(f"  1a {size}^2 seed{seed} n_sub{n_sub}: CPU/GPU "
                          f"DIVERGE (digests {d_cpu} vs {d_gpu}, "
                          f"{int(np.count_nonzero(a[0] != b[0]))} T cells)")
                # P-E1 (energy-books design SS2.1.1, AUTHORIZED REWRITE):
                # the "mask-is-live" control INVERTS here. It used to require
                # that omitting `thermal_solid` CHANGED the T result, because
                # the mask's whole job at this entry was guarding the SL T
                # sample. That sample is retired (T-WRITE SITE 1/2, the
                # measured mint) — SL advection is u-only now — so the mask
                # must change NOTHING at this entry, and temperature must come
                # back exactly as it went in on BOTH twins. The mask's live
                # roles moved to 1b (step-4c) and to the energy books' ts-face
                # rule (d), both still controlled below / at the engine level.
                if not np.array_equal(b[0], c[0]):
                    ok = False
                    print(f"  1a {size}^2 seed{seed} n_sub{n_sub}: the mask "
                          f"changed TEMPERATURE — SL advection is u-only now")
                for tag, fields in (("cpu", a), ("gpu", b)):
                    if not np.array_equal(fields[0], t):
                        ok = False
                        print(f"  1a {size}^2 seed{seed} n_sub{n_sub}: {tag} "
                              f"twin WROTE temperature — the retired SL "
                              f"T-copy is back")
                # velocity must be UNTOUCHED by the mask (ruling §4 item 4)
                if not (np.array_equal(b[1], c[1]) and np.array_equal(b[2], c[2])):
                    ok = False
                    print(f"  1a {size}^2 seed{seed} n_sub{n_sub}: the mask "
                          f"CHANGED VELOCITY — cmask must be untouched")
                # non-vacuity: the pass really ran and really advected u
                if np.array_equal(b[1], wx) and np.array_equal(b[2], wy):
                    n_div += 1
    if n_div:
        ok = False
        print(f"  1a: {n_div} configs never moved velocity — VACUOUS")
    else:
        print(f"  1a SL advection: {n_cfg} configs bit-identical CPU<->GPU "
              f"(tol 0); u-only (P-E1) — temperature untouched by both twins "
              f"and by the mask-omitted control; velocity moved everywhere.")

    # ---------------- 1b: kick + compression (step-4c) ---------------------
    # arc #54 (P-G2): step 4c (compression work) is DELETED. `temperature` is
    # no longer written by EITHER twin (moved to the §2.6 recovery, outside
    # this isolated tail); `t_work_clamp`/`k_drag_heat_frac`/`c_v` are RETIRED
    # from both signatures (the CPU reference keeps t_min/t_max_phys in its
    # own signature, D10 layout, but they are dormant here — the GPU entry
    # dropped them outright). `gas_energy` (arc #54 §2.2/§2.3) is now the
    # field the kick's KE brackets debit/credit; the ts mask's live role at
    # this site moved from "gates a temperature write" to "gates whether the
    # KE debit lands in gas_energy (bulk) or is exported (F5, ts cells never
    # carry gas_energy)" — the mask-omitted control below now diverges there
    # instead of on temperature.
    n_cfg = n_div = 0
    for size in (16, 24, 33):
        for seed in (44, 55):
            rng = np.random.default_rng(seed * 1000 + size)
            h = w = size
            solid, vac, tsol, perm, furn = _iso_world(h, w, rng)
            t, wx, wy = _iso_fields(h, w, rng, solid, vac)
            p_new = _c(_q(rng.uniform(-0.2, 0.4, size=(h, w))))
            n_gases = 3
            gas = _c(_q(rng.uniform(0.0, 1.2, size=(n_gases, h, w))))
            cons = np.array([True, True, False], dtype=bool)
            wabs = _c(rng.uniform(0.0, 1.0, size=(h, w)).astype(np.float32))
            # VELOCITY-CLAMP (P-V1, D2v2): a uniform (h,w) cap² plane at the
            # old scalar c_local_q's value (300 m/s) — this gate compares
            # CPU vs GPU, not clamp behavior, so any valid (>= 0) plane
            # works; a uniform fill reproduces the old regime exactly (D5).
            cap2 = np.full((h, w), (300 * FP_ONE) ** 2, dtype=np.int64)
            ref_args = dict(dt=1.0 / 24.0, cap2_plane=cap2,
                             c_max=300.0, dx=1.0 / 3.0, adiabatic_index=1.4,
                             absorb_strength=8.0, n_floor_solver=1e-3,
                             # G12 (issue #12): config's T_MIN/eos_t_amb_k,
                             # -289/290 -> -292/293. Explicit CPU/GPU parity
                             # params, not tied to the (unchanged) C++ struct
                             # defaults.
                             t_min=-292.0, t_max_phys=16000.0, u_max=1000.0,
                             t_amb_k=293.0)
            gpu_args = dict(dt=1.0 / 24.0, cap2_plane=cap2,
                             c_max=300.0, dx=1.0 / 3.0, adiabatic_index=1.4,
                             absorb_strength=8.0, n_floor_solver=1e-3,
                             u_max=1000.0, t_amb_k=293.0)
            n_cfg += 1
            ge0 = _c((_q(rng.uniform(200.0, 400.0, size=(h, w))).astype(np.int64)) << 16)
            A = (wx.copy(), wy.copy(), t.copy(), ge0.copy())
            B = (wx.copy(), wy.copy(), t.copy(), ge0.copy())
            C = (wx.copy(), wy.copy(), t.copy(), ge0.copy())
            r_cpu = bp.eos_kick_compression_ref(
                A[0], A[1], A[2], p_new, gas, cons, solid, vac, wabs,
                thermal_solid=tsol, gas_energy=A[3], **ref_args)
            r_gpu = bp.cuda_eos_kick_compression(
                B[0], B[1], B[3], p_new, gas, cons, solid, vac, wabs,
                thermal_solid=tsol, **gpu_args)
            bp.cuda_eos_kick_compression(
                C[0], C[1], C[3], p_new, gas, cons, solid, vac, wabs,
                thermal_solid=None, **gpu_args)
            dig_cpu = r_cpu[0]
            cnts_cpu = tuple(int(v) for v in r_cpu[2:11])   # the 9 D10-layout slots
            dig_gpu, cnts_gpu_full = r_gpu
            cnts_gpu = tuple(int(v) for v in cnts_gpu_full[:9])
            if (dig_cpu != dig_gpu or cnts_cpu != cnts_gpu
                    or not np.array_equal(A[0], B[0])
                    or not np.array_equal(A[1], B[1])
                    or not np.array_equal(A[3], B[3])
                    or not np.array_equal(A[2], t)):
                ok = False
                print(f"  1b {size}^2 seed{seed}: CPU/GPU DIVERGE "
                      f"(digest_velocity {dig_cpu} vs {dig_gpu}, counters "
                      f"{cnts_cpu} vs {cnts_gpu})")
            # the mask-is-live control: with the mask, furniture (ts) cells
            # export their KE debit instead of storing it in gas_energy; drop
            # the mask (falls back to `solid`, and furniture is NOT solid) and
            # those same cells switch to storing — gas_energy must diverge.
            if not np.array_equal(B[3], C[3]):
                n_div += 1
            if not (np.array_equal(B[0], C[0]) and np.array_equal(B[1], C[1])):
                ok = False
                print(f"  1b {size}^2 seed{seed}: the mask CHANGED VELOCITY — "
                      f"the kick must be untouched")
    if n_div == 0:
        ok = False
        print("  1b: the mask-omitted control never diverged — VACUOUS")
    else:
        print(f"  1b kick+compression: {n_cfg} configs bit-identical CPU<->GPU "
              f"(tol 0, digest_velocity + gas_energy + the 9 D10-layout rail "
              f"counters); the mask-omitted control diverged (gas_energy's "
              f"ts-export split, F5) in {n_div}/{n_cfg} and NEVER moved "
              f"velocity.")

    # ---------------- 1c: combustion deposit (ruling §2 site 3) -----------
    from simulation.materials import MAT_AIR, MAT_FURNITURE  # noqa: F401
    n_cfg = n_div = 0
    solver = bp.CombustionSolver()
    for size in (16, 24):
        for seed in (66, 77):
            rng = np.random.default_rng(seed * 100 + size)
            h = w = size
            solid, vac, tsol, perm, furn = _iso_world(h, w, rng)
            if not furn.any():
                continue
            n_gases = 3
            o2, n2, soot = 0, 1, 2
            gas0 = _q(np.stack([
                np.full((h, w), 0.21 * 1.0), np.full((h, w), 0.79 * 1.0),
                np.zeros((h, w))]))
            # flammable = the furniture block + a wood stub; the wood stub is
            # SOLID so its neighbours are air burn sites, the crate itself is an
            # OPEN burn site — the two conversion paths, side by side.
            flam = furn.copy()
            flam[h // 3, 3:8] = True
            wall_hp = _c(_q(np.where(flam, 30.0, 0.0)))
            fire = _c(_q(np.where(flam, 0.7, 0.0)))
            ign = _c(_q(np.where(flam, 280.0, 0.0)))
            t0 = _q(np.full((h, w), 400.0))     # everything above ignition
            shift = np.where(tsol, 3, 0).astype(np.int32)
            n_cfg += 1
            outs = {}
            for tag, mask in (("mask", tsol), ("fallback", None)):
                gas = _c(gas0.copy())
                t = _c(t0.copy())
                whp = _c(wall_hp.copy())
                solver.step(gas, o2, n2, soot, t, whp, fire, flam, solid, vac,
                            ign, dt=1.0 / 24.0, c_v=1.0, n_floor_heat=0.05,
                            thermal_solid=mask,
                            heat_inv_shift=(_c(shift) if mask is not None else None))
                gas_g = _c(gas0.copy())
                t_g = _c(t0.copy())
                whp_g = _c(wall_hp.copy())
                bp.cuda_combustion_step(
                    gas_g, o2, n2, soot, t_g, whp_g, fire, flam, solid, vac, ign,
                    dt=1.0 / 24.0, c_v=1.0, n_floor_heat=0.05,
                    burn_rate=solver.burn_rate, o2_thresh_burn=solver.o2_thresh_burn,
                    H_fuel=solver.H_fuel, soot_yield=solver.soot_yield,
                    fuel_per_o2=solver.fuel_per_o2,
                    o2_frac_ext=solver.o2_frac_ext, o2_frac_full=solver.o2_frac_full,
                    T_MAX_PHYS=solver.T_MAX_PHYS,
                    thermal_solid=mask,
                    heat_inv_shift=(_c(shift) if mask is not None else None))
                if not (np.array_equal(t, t_g) and np.array_equal(gas, gas_g)
                        and np.array_equal(whp, whp_g)):
                    ok = False
                    print(f"  1c {size}^2 seed{seed} [{tag}]: CPU/GPU DIVERGE "
                          f"({int(np.count_nonzero(t != t_g))} T cells, "
                          f"{int(np.count_nonzero(gas != gas_g))} gas cells)")
                outs[tag] = t
            if not np.array_equal(outs["mask"], outs["fallback"]):
                n_div += 1
    if n_div == 0:
        ok = False
        print("  1c: the object-deposit branch never changed the answer — VACUOUS")
    else:
        print(f"  1c combustion deposit: {n_cfg} configs bit-identical CPU<->GPU "
              f"(tol 0) on BOTH the object path and the gas fallback; the object "
              f"branch changed the answer in {n_div}/{n_cfg}.")
    return ok


# ===========================================================================
# PARTS 2 & 3 — the FURNITURE-BURN engine A/B (step path + resident path)
# ===========================================================================
# The kernels P-EOS touched (plus the ones they chain with on the resident path).
_EOS_BACKENDS = ("set_bulk_flux_backend", "set_sl_advection_backend",
                 "set_mg_solve_backend", "set_kick_compression_backend",
                 "set_combustion_backend")
_ALL_BACKENDS = ("set_temperature_backend", "set_water_backend",
                 "set_smoke_backend", "set_fire_backend",
                 "set_raycaster_backend") + _EOS_BACKENDS

_FIELDS = ("atmosphere", "wave_p", "wind_x", "wind_y", "temperature", "heat",
           "fire", "wall_hp", "water_depth", "flow_vx", "flow_vy", "gas",
           "ripple", "ripple_v")
_COUNTERS = ("u_clamp_hits", "u_max_hits", "work_clamp_hits",
             "energy_floor_hits", "t_max_phys_hits")

_CRATE = (slice(9, 14), slice(9, 14))
_FIRE_SEED = (slice(10, 13), slice(7, 9))
_EDIT_TICK = 14


def _flip(names, on):
    for n in names:
        getattr(bp, n)(bool(on))


def _residency(on: bool) -> None:
    from simulation import physics_runner
    physics_runner.set_residency(bool(on))


def _build_furniture_scenario(H=48, W=48):
    """The P2 gate's FURNITURE-BURN world, verbatim (one scenario, both gates):
    a hull-shelled room with a vacuum breach, a FURNITURE block, a fire seeded
    beside AND on it, a wood partition, a water pool and a trace cloud."""
    from config import CFG
    from level_loader import LevelData
    from simulation import atmosphere_fixed, fire_fixed, water_fixed
    from simulation.gamemap import GameMap
    from simulation.gases import O2
    from simulation.materials import MAT_FURNITURE
    from simulation.physics_runner import PhysicsRunner

    tm = np.full((H, W), 9, dtype=np.int32)
    tm[2:H - 2, 2:W - 2] = 1
    tm[3:H - 3, 3:W - 3] = 0
    tm[_CRATE] = 6
    tm[20, 6:30] = 2
    tm[H // 2 - 2:H // 2 + 2, W - 3] = 0

    level = LevelData(name="thermal_mass_eos_furniture_burn", version="2",
                      path=Path("."), tilemap=tm, tile_size_m=1.0 / 3.0,
                      diffuse_path=Path("."))
    g = GameMap(level)
    g.stamp_units([])
    furn = (g.material == MAT_FURNITURE)
    assert furn.any(), "the scenario must carry furniture"
    assert np.array_equal(g.thermal_solid != g.solid, furn), \
        "furniture must be the ONLY divergence between the thermal and flow masks"

    q = atmosphere_fixed.quantize_scalar
    g.temperature[6:12, 6:12] += q(3000.0)
    g.gas[O2, 7:11, 7:11] += q(3.0)
    g.fire[_FIRE_SEED] = fire_fixed.quantize_scalar(0.9)
    g.fire[11, 11] = fire_fixed.quantize_scalar(0.8)
    g.temperature[_CRATE] += q(500.0)
    g.water_depth[H - 8:H - 5, 6:W // 2] = water_fixed.quantize_scalar(0.4)
    trace_ids = [gi for gi in range(g.gas.shape[0])
                 if not bool(g.gases.conservative[gi])]
    assert trace_ids, "scenario needs a trace plane"
    g.gas[trace_ids[0], 30:44, 20:40] += q(0.5)

    runner = PhysicsRunner(bp)
    g.bind_physics_engine(runner.engine)
    dt = 1.0 / float(CFG.clock.ticks_per_second)
    return runner, g, dt


def _one_tick(runner, g, dt, tick_idx, observe=None):
    if tick_idx == _EDIT_TICK:
        g.destroy_wall(11, 11)
    g.stamp_units([])
    destroyed = runner.step(g, dt)
    for (yy, xx) in destroyed:
        g.destroy_wall(yy, xx)
    if observe is not None:
        observe(g)
    g.heat.fill(0)


def _compare_tick(t, g_cpu, g_gpu, eos_cpu, eos_gpu):
    bad = 0
    for f in _FIELDS:
        a, b = getattr(g_cpu, f), getattr(g_gpu, f)
        if not np.array_equal(a, b):
            bad += 1
            mism = int(np.count_nonzero(a != b))
            amax = (float(np.abs(a.astype(np.int64) - b.astype(np.int64)).max())
                    if a.dtype != np.float32 else float(np.abs(a - b).max()))
            print(f"  tick {t}: field {f}: {mism} MISMATCH(es), max|delta|={amax}")
    for c in _COUNTERS:
        cc, cg = int(getattr(eos_cpu, c)), int(getattr(eos_gpu, c))
        if cc != cg:
            bad += 1
            print(f"  tick {t}: counter {c} mismatch (cpu={cc} gpu={cg})")
    return bad


def _burn_activity(g, acc):
    from simulation.materials import MAT_FURNITURE
    furn = (g.material == MAT_FURNITURE)
    if furn.any():
        acc["crate_heat"] += int(np.count_nonzero(g.heat[furn] > 0))
        acc["crate_T"] = max(acc["crate_T"], int(np.abs(g.temperature[furn]).max()))
        acc["crate_burn"] += int(np.count_nonzero(g.fire[furn] > 0))
        acc["mask_divergent"] |= bool((g.thermal_solid != g.solid).any())
    acc["peak_T"] = max(acc["peak_T"], int(np.abs(g.temperature).max()))
    acc["fire"] += int(np.count_nonzero(g.fire))


def _run_ab(label, n_ticks, gpu_setup, cpu_setup, extra_check=None):
    cpu_setup()
    runner_cpu, g_cpu, dt = _build_furniture_scenario()
    runner_gpu, g_gpu, dt2 = _build_furniture_scenario()
    assert dt == dt2
    eos_cpu, eos_gpu = runner_cpu.engine.eos, runner_gpu.engine.eos
    for f in _FIELDS:
        assert np.array_equal(getattr(g_cpu, f), getattr(g_gpu, f)), \
            f"{label}: scenario construction not deterministic on {f}"

    acc = dict(crate_heat=0, crate_T=0, crate_burn=0, peak_T=0, fire=0,
               mask_divergent=False)
    bad = 0
    for t in range(n_ticks):
        cpu_setup()
        _one_tick(runner_cpu, g_cpu, dt, t,
                  observe=lambda g: _burn_activity(g, acc))
        gpu_setup()
        _one_tick(runner_gpu, g_gpu, dt, t)
        cpu_setup()
        bad += _compare_tick(t, g_cpu, g_gpu, eos_cpu, eos_gpu)
        if bad >= 8:
            print("  aborting after 8 divergences")
            break

    ok = (bad == 0)
    if not acc["mask_divergent"]:
        ok = False
        print(f"  {label}: thermal_solid never differed from solid — VACUOUS")
    if acc["crate_T"] == 0:
        ok = False
        print(f"  {label}: the crate never held a temperature — VACUOUS")
    if acc["crate_burn"] == 0:
        ok = False
        print(f"  {label}: fire never burned ON a furniture tile — the "
              f"combustion object-deposit branch may be unexercised")
    if acc["fire"] == 0:
        ok = False
        print(f"  {label}: fire went out immediately — not a burn scenario")
    if extra_check is not None and not extra_check(g_cpu, g_gpu):
        ok = False
    if ok:
        print(f"  {label}: {n_ticks} ticks bit-identical across all synced "
              f"fields + the 5 EOS rail counters (tol 0). crate held up to "
              f"{acc['crate_T'] / FP_ONE:.0f} K-rel, heat landed on a crate tile "
              f"{acc['crate_heat']} cell-ticks, fire burned on a crate tile "
              f"{acc['crate_burn']} cell-ticks, peak |T| "
              f"{acc['peak_T'] / FP_ONE:.0f} K-rel, scripted crate burn-out @ "
              f"t={_EDIT_TICK}.")
    return ok


def part2_step_path() -> bool:
    print("PART 2 — FURNITURE-BURN engine A/B, STEP path (ONLY the EOS + "
          "combustion backends flip), 40 ticks, all synced fields tol 0:")
    _flip(_EOS_BACKENDS, True)
    live = bp.get_eos_step_backend() and bp.get_combustion_backend()
    _flip(_EOS_BACKENDS, False)
    if not live:
        print("  the EOS/combustion backend flags did not take — cannot gate")
        return False

    calls = {}

    def gpu_setup():
        _residency(False)
        _flip(_EOS_BACKENDS, True)
        assert bp.get_eos_step_backend() and bp.get_combustion_backend()

    def cpu_setup():
        _residency(False)
        _flip(_EOS_BACKENDS, False)

    c0 = int(bp.eos_step_cuda_calls())

    def dispatch_fired(g_cpu, g_gpu):
        delta = int(bp.eos_step_cuda_calls()) - c0
        calls["eos_step_cuda"] = delta
        if delta < 40:
            print(f"  the GPU eos.step chain ran only {delta}/40 ticks — the "
                  f"dispatch was not really exercised")
            return False
        print(f"  GPU eos.step dispatch confirmed live ({delta} chained calls).")
        return True

    ok = _run_ab("PART 2 (step)", 40, gpu_setup, cpu_setup,
                 extra_check=dispatch_fired)
    _residency(False)
    _flip(_ALL_BACKENDS, False)
    return ok


def part3_resident_path() -> bool:
    print("PART 3 — FURNITURE-BURN engine A/B, RESIDENT path (residency + every "
          "backend ON vs CPU), 40 ticks, all synced fields tol 0:")
    try:
        import cupy  # noqa: F401
    except Exception as exc:      # pragma: no cover — reported, never skipped
        print(f"  cupy unavailable ({exc}) — the resident leg CANNOT be gated")
        return False

    def gpu_setup():
        _residency(True)
        _flip(_ALL_BACKENDS, True)

    def cpu_setup():
        _residency(False)
        _flip(_ALL_BACKENDS, False)

    def resident_mask_present(g_cpu, g_gpu):
        """The resident device set must carry `thermal_solid` — the resident SL
        advection and compression kernels READ it now, so a missing (or stale)
        device copy is a determinism bug, not an optimisation."""
        good = True
        if not (bool(g_gpu.residency_on()) and hasattr(g_gpu, "_dev")):
            print("  the GPU world never entered residency mode — VACUOUS")
            good = False
        elif "thermal_solid" not in g_gpu.device_ptrs():
            print("  `thermal_solid` is NOT in the resident device set — the "
                  "resident half is missing")
            good = False
        else:
            dev = g_gpu._dev["thermal_solid"]
            host = g_gpu.thermal_solid
            if tuple(dev.shape) != tuple(host.shape):
                print(f"  resident thermal_solid shape {tuple(dev.shape)} != "
                      f"host {tuple(host.shape)}")
                good = False
            elif not bool((dev.get() == host).all()):
                print("  the resident device thermal_solid is STALE vs the "
                      "mirror — the per-tick from_host upload is missing")
                good = False
            else:
                print(f"  resident device set carries a CURRENT thermal_solid "
                      f"({dev.dtype}, {tuple(dev.shape)}); "
                      f"{int(np.count_nonzero(host))} thermal-solid tiles "
                      f"(post-burn-out, so the on_tile_changed patch rode up).")
        return good

    res0 = int(bp.eos_resident_calls())
    ok = _run_ab("PART 3 (resident)", 40, gpu_setup, cpu_setup,
                 extra_check=resident_mask_present)
    res_delta = int(bp.eos_resident_calls()) - res0
    if res_delta < 40 and ok:
        print(f"  the resident EOS ran only {res_delta}/40 ticks — the resident "
              f"path was not really exercised")
        ok = False
    elif ok:
        print(f"  resident EOS confirmed live ({res_delta} resident calls).")
    _residency(False)
    _flip(_ALL_BACKENDS, False)
    return ok


def main() -> int:
    if not getattr(bp, "HAS_CUDA", False) or not bp.cuda_available():
        print("THERMAL_MASS_EOS_RESULT: FAIL (no CUDA build / device)")
        return 1
    print("device:", bp.cuda_device_info())
    p1 = part1_isolated()
    p2 = part2_step_path()
    p3 = part3_resident_path()
    if p1 and p2 and p3:
        print("THERMAL_MASS_EOS_RESULT: PASS")
        return 0
    print("THERMAL_MASS_EOS_RESULT: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
