"""THERMAL-MASS AXIS, P2 — CPU<->CUDA lockstep gate at tolerance ZERO (in the
GPU subprocess).

THE gate for the P2 patch (docs/thermal_mass_axis_design_2026-07-25.md §3 gate
(d); build addendum 2026-07-30 §3 "P2 (CUDA)"). P1 routed the CPU thermal pass's
six MEDIUM tests from the FLOW mask ``solid`` (permeability <= 0) onto the
derived THERMAL mask ``thermal_solid`` (thermal_mass > 0) and left CUDA on
``solid``; P2 mirrors those same six sites in ``cuda_temperature.cu``. This file
proves the mirror is bit-exact — on a map that CARRIES FURNITURE, where the two
masks actually differ (furniture is the only material that is permeable AND
thermally solid, addendum D4), which is precisely the case the pre-P2 GPU kernel
got wrong.

  PART 1 — ISOLATED, mask-divergent (the mask swap itself): synthetic configs
  where ``thermal_solid != solid`` on a scattered "crate" set — every pass and
  branch driven (Pass 0a zero-at-open-vacuum incl. a SPACE-EXPOSED CRATE, Pass 0b
  semi-Lagrangian advection so the DDA occluder + the bilinear sealed-corner
  gather see a crate, Pass 1 both convert branches with the T_MAX_PHYS rail
  forced, Pass 2 conduction, Pass 3 COOL_SHIFT), across grid sizes and mask
  shapes, plus a 120-tick trajectory. CPU reference = the real bound
  ``TemperatureSolver``; GPU = ``bp.cuda_temperature_step``; both given the SAME
  ``thermal_solid``. temperature byte-identity + rail-hit parity, tol 0.
  NON-VACUOUSNESS CONTROL: the same configs run on the GPU with the mask OMITTED
  (nullptr -> the ``solid`` fallback, i.e. the pre-P2 GPU behaviour) MUST differ
  from the CPU reference. If that control ever passed, the gate would be proving
  nothing.

  PART 2 — FURNITURE-BURN engine A/B, STEP path: a burning-crate world built
  twice and driven 40 ticks through the REAL per-tick path
  (``PhysicsRunner.step`` -> ``step_tail``), once with every GPU backend OFF (the
  CPU reference) and once with the temperature backend ON. Per tick, all synced
  fields byte-identical (tol 0) + the rail counter. Guards: furniture present,
  ``thermal_solid != solid`` exactly on it, heat landing on crate tiles, crate T
  going non-zero, the dispatch predicate actually True.

  PART 3 — FURNITURE-BURN engine A/B, RESIDENT path: the same scenario, CPU
  (residency OFF, backends OFF) vs GPU-RESIDENT (``physics_runner.set_residency``
  ON + every backend ON — the whole EOS stage/water/traces resident, the
  temperature tail bracketed on the mirror with ``GameMap.thermal_solid``
  supplied). Per-tick tol 0 on all synced fields + telemetry, and the resident
  device set is asserted to carry the ``thermal_solid`` mask.

Prints ``THERMAL_MASS_RESULT: PASS``/``FAIL`` and exits 0/1.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import breach_physics as bp   # the CUDA build (sys.path[0] == cpp/build_cuda)

FP_ONE = 65536

# TemperatureSolver config defaults (temperature_solver.h) — the dials both
# backends run with (the CPU solver carries them as members, the GPU free
# function takes them explicitly). Same set cuda_conduction_check uses.
DIALS = dict(
    no_face=63, cool_shift=5, cool_shift_vacuum=3, o2_vacuum_thresh=0.3,
    c_v=1.0, n_floor_heat=0.05, gas_advection_rate=900.0, t_max_phys=16000.0,
)


def _q(x):
    """Round-to-nearest Q16.16 (matches fixedpoint::quantize)."""
    x = np.asarray(x, dtype=np.float64)
    return np.int32(np.where(x >= 0, np.floor(x * FP_ONE + 0.5),
                             np.ceil(x * FP_ONE - 0.5)))


# ===========================================================================
# PART 1 — isolated, mask-divergent
# ===========================================================================
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
    """Symmetric per-tile face-shift cache (h,w,4), dir order N,S,E,W: a face
    conducts iff BOTH cells are anything except an OPEN vacuum cell. Fed
    IDENTICALLY to both backends (conduction is kappa-keyed and deliberately NOT
    one of the six medium sites, so it must not move)."""
    h, w = solid.shape
    NO = DIALS["no_face"]
    fs = np.full((h, w, 4), NO, dtype=np.int32)
    conductive = solid | (~solid & ~is_vacuum)
    DY = (-1, 1, 0, 0)
    DX = (0, 0, 1, -1)
    for d in range(4):
        for y in range(h):
            for x in range(w):
                ny, nx = y + DY[d], x + DX[d]
                if 0 <= ny < h and 0 <= nx < w and conductive[y, x] and conductive[ny, nx]:
                    fs[y, x, d] = shift
    return fs


def _crate_mask(solid, rng, kind):
    """Return ``thermal_solid``: `solid` PLUS a set of "crate" tiles that are
    permeable (not in `solid`) but thermally solid — the furniture case, the ONLY
    way the two masks can differ (addendum D4). At least one crate is placed on a
    vacuum cell where possible, to exercise MEDIUM-TEST SITE 1/6's load-bearing
    guard (a space-exposed crate must keep its object temperature, exactly as an
    intact hull tile does)."""
    ts = solid.copy()
    h, w = solid.shape
    free = ~solid
    if kind == "scatter":
        ts |= free & (rng.random((h, w)) < 0.20)
    elif kind == "block":
        y0, x0 = h // 3, w // 3
        blk = np.zeros((h, w), dtype=bool)
        blk[y0:y0 + max(1, h // 4), x0:x0 + max(1, w // 4)] = True
        ts |= free & blk
    elif kind == "wall_line":
        blk = np.zeros((h, w), dtype=bool)
        blk[h // 2, :] = True
        blk[:, w // 2] = True
        ts |= free & blk
    elif kind == "all_free":
        ts |= free
    return np.ascontiguousarray(ts)


def _mask_grid(h, w, rng, kind):
    """(solid, is_vacuum) for an edge/mask config — the cuda_conduction_check
    set, so PART 1 covers the same structural shapes the P6.6 gate does."""
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
        is_vacuum[0, :] = is_vacuum[-1, :] = True
    return np.ascontiguousarray(solid), np.ascontiguousarray(is_vacuum)


def _synth_fields(h, w, rng, solid, is_vacuum, thin=False, wind=False):
    n = h * w
    t = rng.integers(-8 * FP_ONE, 40 * FP_ONE, size=n, dtype=np.int64).astype(np.int32)
    t[rng.random(n) < 0.15] = 0
    temperature = np.ascontiguousarray(t.reshape(h, w))

    heat = _q(rng.random((h, w)) * 200.0)
    heat[rng.random((h, w)) < 0.4] = 0
    heat[rng.random((h, w)) < 0.05] = _q(30000.0)      # rail-forcing deposits
    heat = np.ascontiguousarray(np.maximum(heat, 0).astype(np.int32))

    his = np.ascontiguousarray(
        rng.integers(0, 5, size=(h, w)).astype(np.int32))   # log2(thermal_mass)

    atm = _q(np.where(is_vacuum, 0.0, 0.6 + rng.random((h, w)) * 0.8))
    atmosphere = np.ascontiguousarray(atm.astype(np.int32))

    if thin:
        nb = _q(rng.random((h, w)) * 1.2)
        nb[rng.random((h, w)) < 0.20] = _q(0.01)
        nb[rng.random((h, w)) < 0.20] = _q(0.5)
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


# P-E2a/P-E2b: the seven energy counters `cuda_temperature_step` now returns
# alongside the T_MAX_PHYS hit count (design §2.3/§2.2). This gate is a
# TEMPERATURE-SOLVER lockstep, so it compares them too — the conduction
# rewrite's books (P-E2a) and the Pass-1 attenuation drop (P-E2b) must agree
# bit-for-bit across the backends here as well.
E_COUNTERS = ("e_cond_trunc_sum", "e_cond_cap_sum", "cond_limit_hits",
              "e_cool_sum", "e_vac_wipe_sum", "e_ring_pin_sum",
              "e_deposit_drop_sum")


def _run_pair(solver, f, solid, is_vacuum, ts):
    """CPU reference vs GPU kernel on identical copies, BOTH given `ts`; plus the
    GPU control run with the mask OMITTED (the pre-P2 `solid` fallback).
    Returns (t_cpu, cnt_cpu, t_gpu, cnt_gpu, t_ctl), where cnt is
    (t_max_phys_hits,) + the six P-E2a energy counters, all per-call."""
    t_cpu = np.ascontiguousarray(f["temperature"].copy())
    c0 = (int(solver.t_max_phys_hits),) + tuple(
        int(getattr(solver, nm)) for nm in E_COUNTERS)
    solver.step(t_cpu, f["heat"], f["his"], f["fs"], solid, is_vacuum,
                f["atmosphere"], wind_x=f["wind_x"], wind_y=f["wind_y"],
                dt=f["dt"], n_bulk=f["n_bulk"], thermal_solid=ts)
    c1 = (int(solver.t_max_phys_hits),) + tuple(
        int(getattr(solver, nm)) for nm in E_COUNTERS)
    hits_cpu = tuple(b - a for a, b in zip(c0, c1))

    t_gpu = np.ascontiguousarray(f["temperature"].copy())
    hits_gpu = tuple(int(v) for v in bp.cuda_temperature_step(
        t_gpu, f["heat"], f["his"], f["fs"], solid, is_vacuum, f["atmosphere"],
        n_bulk=f["n_bulk"], wind_x=f["wind_x"], wind_y=f["wind_y"], dt=f["dt"],
        thermal_solid=ts, **DIALS))

    t_ctl = np.ascontiguousarray(f["temperature"].copy())
    bp.cuda_temperature_step(
        t_ctl, f["heat"], f["his"], f["fs"], solid, is_vacuum, f["atmosphere"],
        n_bulk=f["n_bulk"], wind_x=f["wind_x"], wind_y=f["wind_y"], dt=f["dt"],
        thermal_solid=None, **DIALS)
    return t_cpu, hits_cpu, t_gpu, hits_gpu, t_ctl


def _compare(tag, t_cpu, hits_cpu, t_gpu, hits_gpu):
    ok = True
    if not np.array_equal(t_cpu, t_gpu):
        ok = False
        mism = int(np.count_nonzero(t_cpu != t_gpu))
        idx = int(np.argmax(t_cpu != t_gpu))
        print(f"  {tag}: temperature {mism} MISMATCH (first @ {idx}: "
              f"cpu={t_cpu.flat[idx]} gpu={t_gpu.flat[idx]})")
    # P-E2a: `hits_*` is now (t_max_phys_hits,) + the six energy counters.
    for nm, a, b in zip(("t_max_phys_hits",) + E_COUNTERS, hits_cpu, hits_gpu):
        if a != b:
            ok = False
            print(f"  {tag}: {nm} mismatch cpu={a} gpu={b}")
    return ok


def part1_isolated() -> bool:
    print("PART 1 — isolated CPU vs GPU with thermal_solid != solid (all passes, "
          "rail forced) + the mask-matters control:")
    ok = True
    solver = _make_solver()
    rng = np.random.default_rng(20260730)
    n_cfg = 0
    total_hits = 0
    control_differed = 0
    divergent_masks = 0

    sizes = [(16, 16), (24, 20), (1, 40), (40, 1), (13, 17), (8, 8)]
    kinds = ["all_solid", "all_vacuum", "all_air", "mixed", "hull"]
    crates = ["scatter", "block", "wall_line", "all_free"]
    for (h, w) in sizes:
        for kind in kinds:
            for crate in crates:
                for wind in (False, True):
                    solid, is_vacuum = _mask_grid(h, w, rng, kind)
                    ts = _crate_mask(solid, rng, crate)
                    fld = _synth_fields(h, w, rng, solid, is_vacuum,
                                        thin=True, wind=wind)
                    if not np.array_equal(ts, solid):
                        divergent_masks += 1
                    t_cpu, hc, t_gpu, hg, t_ctl = _run_pair(
                        solver, fld, solid, is_vacuum, ts)
                    tag = f"{h}x{w}/{kind}/crate={crate}/wind={wind}"
                    ok &= _compare(tag, t_cpu, hc, t_gpu, hg)
                    total_hits += hc[0]
                    n_cfg += 1
                    if not np.array_equal(t_cpu, t_ctl):
                        control_differed += 1

    if total_hits == 0:
        ok = False
        print("  COVERAGE HOLE: the T_MAX_PHYS rail never engaged")
    if divergent_masks == 0:
        ok = False
        print("  COVERAGE HOLE: thermal_solid never differed from solid — the "
              "gate would be a re-run of the P6.6 gate")
    # The control is the whole point: with the mask omitted the GPU takes the OLD
    # `solid` medium, which MUST give a different answer on a divergent mask.
    if control_differed == 0:
        ok = False
        print("  VACUOUS: the `solid`-fallback control never differed from the "
              "CPU reference — the mask does not reach the kernel")
    if ok:
        print(f"  all {n_cfg} configs bit-identical on temperature + rail hits "
              f"({divergent_masks} with a divergent mask, rail engagements "
              f"{total_hits}); the mask-omitted control diverged in "
              f"{control_differed}/{n_cfg} configs (proof the mask is live).")
    return ok


def part1b_trajectory() -> bool:
    print("PART 1b — 120-tick crate-in-a-hull trajectory (mask divergent, rail "
          "forced every tick):")
    solver = _make_solver()
    H = W = 40

    solid = np.zeros((H, W), dtype=bool)
    is_vacuum = np.zeros((H, W), dtype=bool)
    solid[0, :] = solid[-1, :] = solid[:, 0] = solid[:, -1] = True
    is_vacuum[0, :] = is_vacuum[-1, :] = True          # radiate-to-space hull rows
    solid[18:23, 18:23] = True                          # a solid fire core

    # The crates: permeable but thermally solid. One sits ON a vacuum cell (row 1,
    # just inside the space-facing hull) once that row is flagged vacuum below —
    # MEDIUM-TEST SITE 1/6's guard for a space-exposed crate.
    is_vacuum[1, 5:9] = True
    thermal_solid = solid.copy()
    thermal_solid[10:14, 24:29] = True                  # a crate stack
    thermal_solid[1, 6:8] = True                        # a SPACE-EXPOSED crate
    thermal_solid[25, 10:30] = True                     # a crate line (occluder)
    thermal_solid = np.ascontiguousarray(thermal_solid)
    assert not np.array_equal(thermal_solid, solid)

    fs = np.ascontiguousarray(_build_face_shift(solid, is_vacuum))
    his = np.ascontiguousarray(np.full((H, W), 3, dtype=np.int32))   # thermal_mass 8

    atm = np.where(is_vacuum, 0.0, 0.9)
    atmosphere = np.ascontiguousarray(_q(atm).astype(np.int32))
    nb = np.where(is_vacuum, 0.0, 0.15)
    nb[10:14, 10:14] = 0.02
    n_bulk = np.ascontiguousarray(_q(nb).astype(np.int32))

    heat = np.zeros((H, W), dtype=np.int32)
    heat[19:22, 19:22] = _q(400.0)                      # into the solid core
    heat[11:13, 25:28] = _q(400.0)                      # into the CRATE (shift path)
    heat[24:27, 32:35] = _q(300.0)                      # radiant into interior gas
    # Force the T_MAX_PHYS rail every tick ON A CRATE TILE: shift 0 there, so the
    # whole (>ceiling) deposit lands through MEDIUM-TEST SITE 5/6's shift branch.
    # (An open-air forcer cannot latch here — Pass 0b advection rewrites gas cells
    # every tick, unlike the wind-free cuda_conduction_check trajectory.)
    his[12, 27] = 0
    heat[12, 27] = _q(20000.0)
    heat = np.ascontiguousarray(heat)

    # Wind: a steady plume so Pass 0b advection runs every tick and the crate
    # line is a real occluder / sealed corner for the ray-walk.
    wind_x = np.ascontiguousarray(_q(np.full((H, W), 4.0)).astype(np.int32))
    wind_y = np.ascontiguousarray(_q(np.full((H, W), -3.0)).astype(np.int32))

    temperature = np.zeros((H, W), dtype=np.int32)
    temperature[19:22, 19:22] = _q(1000.0)
    temperature[11:13, 25:28] = _q(600.0)               # a pre-warmed crate
    temperature[1, 6:8] = _q(500.0)                     # the space-exposed crate

    n_ticks = 120
    bad = 0
    total_hits = 0
    crate_seen = 0
    exposed_survived = 0
    for tick in range(n_ticks):
        fld = dict(temperature=temperature, heat=heat, his=his, fs=fs,
                   atmosphere=atmosphere, n_bulk=n_bulk,
                   wind_x=wind_x, wind_y=wind_y, dt=1.0 / 24.0)
        t_cpu, hc, t_gpu, hg, t_ctl = _run_pair(
            solver, fld, solid, is_vacuum, thermal_solid)
        if not _compare(f"tick {tick}", t_cpu, hc, t_gpu, hg):
            bad += 1
        total_hits += hc[0]
        if int(t_cpu[11, 26]) != 0:
            crate_seen += 1
        if int(t_cpu[1, 7]) != 0:
            exposed_survived += 1
        temperature = np.ascontiguousarray(t_cpu)
        if bad >= 10:
            print("  aborting after 10 divergences")
            break

    ok = (bad == 0)
    if crate_seen < n_ticks // 2:
        ok = False
        print(f"  scenario too tame: the crate held a temperature on only "
              f"{crate_seen}/{n_ticks} ticks")
    if exposed_survived < n_ticks // 2:
        ok = False
        print(f"  SITE 1/6 not exercised: the space-exposed crate kept its T on "
              f"only {exposed_survived}/{n_ticks} ticks (it should survive the "
              f"Pass-0a wipe exactly as a hull tile does)")
    if total_hits < n_ticks - 5:
        ok = False
        print(f"  scenario too tame: rail engaged {total_hits} times")
    if ok:
        print(f"  {n_ticks} ticks bit-identical (rail engagements {total_hits}; "
              f"crate held T {crate_seen}/{n_ticks} ticks; space-exposed crate "
              f"survived the Pass-0a wipe {exposed_survived}/{n_ticks} ticks).")
    return ok


# ===========================================================================
# PARTS 2 & 3 — the FURNITURE-BURN engine A/B (step path + resident path)
# ===========================================================================
_BACKENDS = (
    "set_temperature_backend", "set_water_backend", "set_smoke_backend",
    "set_fire_backend", "set_raycaster_backend",
    "set_bulk_flux_backend", "set_sl_advection_backend",
    "set_mg_solve_backend", "set_kick_compression_backend",
    "set_combustion_backend",
)

# All synced state (the cuda_s8a_check set): a divergence anywhere downstream of
# temperature (ignition -> fire -> heat -> wall_hp) shows up here too.
_FIELDS = ("atmosphere", "wave_p", "wind_x", "wind_y", "temperature", "heat",
           "fire", "wall_hp", "water_depth", "flow_vx", "flow_vy", "gas",
           "ripple", "ripple_v")
_COUNTERS = ("u_clamp_hits", "u_max_hits", "work_clamp_hits",
             "energy_floor_hits", "t_max_phys_hits")

# The crate block + the fire seed that lights it.
_CRATE = (slice(9, 14), slice(9, 14))
_FIRE_SEED = (slice(10, 13), slice(7, 9))


def _set_backends(on: bool) -> None:
    for name in _BACKENDS:
        getattr(bp, name)(bool(on))


def _only_temperature_backend(on: bool) -> None:
    """PART 2 isolates the patched kernel: ONLY the temperature backend flips, so
    any divergence is the temperature pass and nothing else."""
    bp.set_temperature_backend(bool(on))


def _residency(on: bool) -> None:
    from simulation import physics_runner
    physics_runner.set_residency(bool(on))


def _build_furniture_scenario(H=48, W=48):
    """A FURNITURE-BURN world: a hull-shelled room with a breach to a vacuum
    band, a block of FURNITURE (the only permeable-yet-thermally-solid material)
    in the middle, a fire seeded right beside it (radiant heat -> the crate's
    thermal mass), a water pool and a trace cloud (so the water/trace resident
    loops are live too). Independently constructed and fully deterministic."""
    from config import CFG
    from level_loader import LevelData
    from simulation import atmosphere_fixed, fire_fixed, water_fixed
    from simulation.gamemap import GameMap
    from simulation.gases import O2
    from simulation.materials import MAT_FURNITURE
    from simulation.physics_runner import PhysicsRunner

    # v2 vocabulary: codes ARE material ids, plus 9 == SPACE.
    tm = np.full((H, W), 9, dtype=np.int32)             # outer space band
    tm[2:H - 2, 2:W - 2] = 1                            # hull shell
    tm[3:H - 3, 3:W - 3] = 0                            # interior air (MAT_AIR)
    tm[_CRATE] = 6                                      # MAT_FURNITURE (crates)
    tm[20, 6:30] = 2                                    # a wood partition (more fuel)
    tm[H // 2 - 2:H // 2 + 2, W - 3] = 0                # breach the east hull -> vacuum

    level = LevelData(name="thermal_mass_furniture_burn", version="2",
                      path=Path("."), tilemap=tm, tile_size_m=1.0 / 3.0,
                      diffuse_path=Path("."))
    g = GameMap(level)
    g.stamp_units([])
    furn = (g.material == MAT_FURNITURE)
    assert furn.any(), "the scenario must carry furniture"
    assert np.array_equal(g.thermal_solid != g.solid, furn), \
        "furniture must be the ONLY divergence between the thermal and flow masks"

    q = atmosphere_fixed.quantize_scalar
    # A hot core + an O2 overpressure pocket (wind/venting -> a real plume over
    # the crate, which is what used to advect the crate's T away).
    g.temperature[6:12, 6:12] += q(3000.0)
    g.gas[O2, 7:11, 7:11] += q(3.0)
    # THE BURN: fire seeded next to the crate stack, plus one tile ON it.
    g.fire[_FIRE_SEED] = fire_fixed.quantize_scalar(0.9)
    g.fire[11, 11] = fire_fixed.quantize_scalar(0.8)
    # Pre-warm the crate so its object temperature is live from tick 0 (the EOS
    # pass strips T off a furniture tile in BOTH backends — the P1 escalation,
    # out of P2's scope — so this keeps the medium branches genuinely engaged).
    g.temperature[_CRATE] += q(500.0)
    g.water_depth[H - 8:H - 5, 6:W // 2] = water_fixed.quantize_scalar(0.4)
    trace_ids = [gi for gi in range(g.gas.shape[0])
                 if not bool(g.gases.conservative[gi])]
    assert trace_ids, "scenario needs a trace plane"
    g.gas[trace_ids[0], 30:44, 20:40] += q(0.5)

    runner = PhysicsRunner(bp)
    # Bind so stamp_units takes the C++ in-place path (the live game path; the
    # Python fallback reassigns `obstacles` and would trip the residency guard).
    g.bind_physics_engine(runner.engine)
    dt = 1.0 / float(CFG.clock.ticks_per_second)
    return runner, g, dt


_EDIT_TICK = 14        # a scripted structural edit, on BOTH worlds


def _one_tick(runner, g, dt, tick_idx, observe=None):
    """One full engine tick, mirroring Simulation.step's physics slice. `observe`
    (if given) runs BEFORE the end-of-tick heat clear — `heat` is a per-tick
    scratch field, so the burn evidence must be read while it is still live."""
    if tick_idx == _EDIT_TICK:
        g.destroy_wall(11, 11)      # burn out one CRATE tile: the mask must patch
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
    """Accumulate the non-vacuousness evidence for the furniture-burn A/B."""
    from simulation.materials import MAT_FURNITURE
    furn = (g.material == MAT_FURNITURE)
    if furn.any():
        acc["crate_heat"] += int(np.count_nonzero(g.heat[furn] > 0))
        acc["crate_T"] = max(acc["crate_T"], int(np.abs(g.temperature[furn]).max()))
        acc["mask_divergent"] |= bool((g.thermal_solid != g.solid).any())
    acc["peak_T"] = max(acc["peak_T"], int(np.abs(g.temperature).max()))
    acc["fire"] += int(np.count_nonzero(g.fire))


def _run_ab(label, n_ticks, gpu_setup, cpu_setup, extra_check=None):
    """Two independently built furniture-burn worlds, CPU vs GPU, per-tick tol 0."""
    cpu_setup()
    runner_cpu, g_cpu, dt = _build_furniture_scenario()
    runner_gpu, g_gpu, dt2 = _build_furniture_scenario()
    assert dt == dt2
    eos_cpu, eos_gpu = runner_cpu.engine.eos, runner_gpu.engine.eos
    for f in _FIELDS:
        assert np.array_equal(getattr(g_cpu, f), getattr(g_gpu, f)), \
            f"{label}: scenario construction not deterministic on {f}"

    acc = dict(crate_heat=0, crate_T=0, peak_T=0, fire=0, mask_divergent=False)
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
    if acc["crate_heat"] == 0:
        ok = False
        print(f"  {label}: no heat ever landed on a furniture tile — the "
              f"convert-branch medium test (site 5/6) was never exercised")
    if acc["crate_T"] == 0:
        ok = False
        print(f"  {label}: the crate never held a temperature — VACUOUS")
    if acc["fire"] == 0:
        ok = False
        print(f"  {label}: fire went out immediately — not a burn scenario")
    if extra_check is not None and not extra_check(g_cpu, g_gpu):
        ok = False
    if ok:
        print(f"  {label}: {n_ticks} ticks bit-identical across all synced "
              f"fields + rail counters (tol 0). crate held up to "
              f"{acc['crate_T'] / FP_ONE:.0f} K-rel, heat landed on a crate tile "
              f"{acc['crate_heat']} cell-ticks, peak |T| "
              f"{acc['peak_T'] / FP_ONE:.0f} K-rel, scripted crate burn-out @ "
              f"t={_EDIT_TICK}.")
    return ok


def part2_step_path() -> bool:
    print("PART 2 — FURNITURE-BURN engine A/B, STEP path (only the temperature "
          "backend flips), 40 ticks, all synced fields tol 0:")
    _only_temperature_backend(True)
    if not bp.get_temperature_backend():
        print("  temperature backend flag did not take — cannot gate")
        return False
    _only_temperature_backend(False)

    def gpu_setup():
        _residency(False)
        _only_temperature_backend(True)
        assert bp.get_temperature_backend()

    def cpu_setup():
        _residency(False)
        _only_temperature_backend(False)

    def mask_is_live(g_cpu, g_gpu):
        """Direct proof, on this scenario's REAL end-of-run fields, that the mask
        reaches the kernel: the same GPU call with the mask omitted (the pre-P2
        `solid` fallback) must give a different temperature field."""
        from config import CFG
        thermal = CFG.physics.thermal
        dials = dict(
            no_face=int(getattr(thermal, "NO_FACE", 63)),
            cool_shift=int(getattr(thermal, "COOL_SHIFT", 5)),
            cool_shift_vacuum=int(getattr(thermal, "COOL_SHIFT_VACUUM", 3)),
            o2_vacuum_thresh=float(getattr(thermal, "o2_vacuum_thresh", 0.3)),
            c_v=float(getattr(thermal, "c_v", 1.0)),
            n_floor_heat=float(getattr(thermal, "n_floor_heat", 0.05)),
            gas_advection_rate=float(getattr(thermal, "gas_advection_rate", 900.0)),
            t_max_phys=float(getattr(thermal, "T_MAX_PHYS", 16000.0)),
        )
        heat = np.ascontiguousarray(np.full(g_cpu.temperature.shape,
                                            _q(200.0), dtype=np.int32))
        out = {}
        for tag, mask in (("mask", g_cpu.thermal_solid), ("fallback", None)):
            t = np.ascontiguousarray(g_cpu.temperature.copy())
            bp.cuda_temperature_step(
                t, heat, g_cpu.heat_inv_shift, g_cpu.face_shift,
                g_cpu.solid, g_cpu.is_vacuum, g_cpu.atmosphere,
                n_bulk=None, wind_x=g_cpu.wind_x, wind_y=g_cpu.wind_y,
                dt=1.0 / 24.0, thermal_solid=mask, **dials)
            out[tag] = t
        if np.array_equal(out["mask"], out["fallback"]):
            print("  VACUOUS: on the real furniture-bearing state the GPU gave "
                  "the SAME answer with and without the thermal mask")
            return False
        n = int(np.count_nonzero(out["mask"] != out["fallback"]))
        print(f"  mask-is-live control: the `solid`-fallback GPU call differs "
              f"from the masked one on {n} cells of the live scenario state.")
        return True

    ok = _run_ab("PART 2 (step)", 40, gpu_setup, cpu_setup,
                 extra_check=mask_is_live)
    _residency(False); _set_backends(False)
    return ok


def part3_resident_path() -> bool:
    print("PART 3 — FURNITURE-BURN engine A/B, RESIDENT path (residency + every "
          "backend ON vs CPU), 40 ticks, all synced fields tol 0:")
    try:
        import cupy  # noqa: F401
    except Exception as exc:      # pragma: no cover — reported, never skipped
        print(f"  cupy unavailable ({exc}) — the resident leg CANNOT be gated")
        return False

    seen = {}

    def gpu_setup():
        _residency(True)
        _set_backends(True)

    def cpu_setup():
        _residency(False)
        _set_backends(False)

    def resident_mask_present(g_cpu, g_gpu):
        """The resident device set must actually carry `thermal_solid` (the one
        static mask upload this patch adds) and the GPU world must really have
        entered residency mode."""
        good = True
        if not (bool(g_gpu.residency_on()) and hasattr(g_gpu, "_dev")):
            print("  the GPU world never entered residency mode — VACUOUS")
            good = False
        elif "thermal_solid" not in g_gpu.device_ptrs():
            print("  `thermal_solid` is NOT in the resident device set "
                  "(GameMap._RESIDENT_MASKS) — the resident half is missing")
            good = False
        else:
            dev = g_gpu._dev["thermal_solid"]
            host = g_gpu.thermal_solid
            if tuple(dev.shape) != tuple(host.shape):
                print(f"  resident thermal_solid shape {tuple(dev.shape)} != "
                      f"host {tuple(host.shape)}")
                good = False
            else:
                print(f"  resident device set carries thermal_solid "
                      f"({dev.dtype}, {tuple(dev.shape)}); "
                      f"{int(np.count_nonzero(host))} thermal-solid tiles.")
        seen["resident"] = good
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
    _residency(False); _set_backends(False)
    return ok


def main() -> int:
    if not getattr(bp, "HAS_CUDA", False) or not bp.cuda_available():
        print("THERMAL_MASS_RESULT: FAIL (no CUDA build / device)")
        return 1
    print("device:", bp.cuda_device_info())
    p1 = part1_isolated()
    p1b = part1b_trajectory()
    p2 = part2_step_path()
    p3 = part3_resident_path()
    if p1 and p1b and p2 and p3:
        print("THERMAL_MASS_RESULT: PASS")
        return 0
    print("THERMAL_MASS_RESULT: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
