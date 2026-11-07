"""CUDA-S8a — GPU residency bit-identity + payoff gate (in the subprocess).

THE gate for the S8a residency build: Path B (Rung 1, the framework + leaf
solvers — docs/cuda_s8a_residency_spec_2026-07-19, the ★ BUILD FINDING block)
EXTENDED for Path A (the fully resident EOS stage —
docs/cuda_s8a_path_a_impl_2026-07-21.md §7). Parts:

  #29 P4a (the brackets gone): the resident tick runs the water tail,
  combustion, fire, the radiation sweep and temperature on the device too, so
  PART 1a / 1b and the new 1d / 1e / 1f compare EVERY GameMap array and EVERY
  integer book (combustion, EOS, temperature, engine, the trace books, the
  seam net, the solid books, the n_sub / c_LOCAL / stream telemetry) at tol 0
  and evaluate the ENERGY CLOSURE IDENTITY on both worlds (equal residuals,
  0 from tick 1), and assert that each covered stage ran on the device (the
  resident stage counters) and the per-call twins did not:
    1a space map + a blast (#31's divergence source) + burning wood, 40 ticks
    1b ambient map + the sky exchange live (its host bracket), 40 ticks
    1d WATER: a flooding compartment breached to vacuum (W5 + W3), 60 ticks
    1e fire_smoke_studio burning, LIGHT requested (cone, sky, flicker), 60
    1f ship_demo, the real match (bots + horde, headless), 120 ticks
  The per-part notes below predate P4a where they say "bracketed".

  PART 1a — BIT-IDENTITY, SPACE MAP: a >=30-tick FULL-ENGINE A/B on a seeded
  scenario with detonations / water / fire / a scripted structural edit, driven
  through the REAL per-tick path (PhysicsRunner.step) on two independently
  built worlds — CPU (residency OFF, backends OFF) vs GPU-RESIDENT (residency
  ON, backends ON; Path A: water + the WHOLE EOS stage (trace planes riding
  its substeps) + the trace tail resident, the
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

  PART 2 — THE PAYOFF: (i) the Path-B isolated WATER tax bench; (ii) the
  Path-A isolated EOS-stage bench — per-call run_substeps(do_traces=False) vs
  from_host + run_substeps_resident + to_host at >=2 grid sizes: resident must
  clearly win and win MORE on the bigger grid. (Bench note: the per-call side
  also pays ~6 host FNV plane digests the resident path skips by design —
  part of the margin is digest removal, not transfer removal.) (iii) smoke
  transport v2 (#12, P2b): THE RESIDENT TRACE TAIL MOVES NO PLANE ACROSS THE
  BUS — at every grid size, one resident tail (run_trace_tail_resident:
  stranded zeroing + Jacobi diffusion + decay on the device gas block, one
  small H2D of folds, one small D2H of books) costs less than ONE host
  round-trip of the gas block it would need if the tail ran on the host.
  BREAKS IF the tail grows a per-plane (or whole-block) transfer.
  (P2b: the smoke half of (i) is GONE with its subject — it timed the per-call
  per-plane GPU semi-Lagrangian smoke step, cuda_smoke_step, against the
  resident SL loop, trace_smoke_resident; both were deleted with the old law.
  The trace planes now ride the bulk face flux inside the EOS substeps on every
  path, so there is no per-plane trace transfer tax left to bench; (iii) is
  the residency property of what replaced the loop.)

  Per tick, PART 1a/1b also A/B the four TRACE BOOKS (boundary_flux() — the
  vent channel in its trace slots — trace_wipe_sum, trace_sink_sum,
  trace_decay_sum): the resident EOS's stages 3b/3c and the resident trace
  tail book exactly what the CPU books (smoke transport v2 V6-resident).

Prints ``S8A_RESULT: PASS``/``FAIL`` and exits 0/1.
"""
from __future__ import annotations

import sys
import time

import numpy as np

# Import the CUDA build FIRST so it is the cached `breach_physics`.
import breach_physics as bp

FP_ONE = 65536

# (#29 P4a: on the resident path the per-call flags below no longer select the
# tail's stages -- the water tail, combustion, fire, the sweep and
# temperature run as resident drivers whatever they say.)
# The GPU backends flipped on for the resident/per-call runs (run_on_cuda's set,
# combustion included since #71 / #29 P2 — the scenarios carry burning wood so
# the combustion twin's pass actually runs). In the resident tick the water
# per-call flag is inert (that
# stages run resident), but the EOS/combustion/fire/temperature/radiation-sweep
# brackets use their per-call GPU path — so a resident tick is a genuine all-GPU
# tick. (ray-engine-v2 P4: set_radiation_backend replaces the vestigial
# set_raycaster_backend, deleted with cuda_raycaster.{cu,h}.)
_BACKENDS = (
    "set_temperature_backend", "set_water_backend",
    "set_fire_backend", "set_radiation_backend",
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


def _light_wood(g, cells):
    """#29 P2 (#71): light wood tiles the CUDA-combustion way the playground
    repro does — heated to the 1263-game plateau AND lit (CLAUDE.md "Starting
    a fire": heat delivered, never `fire` alone). Without a flammable source
    the combustion backend this gate turns on had no claimant and never ran
    its pass (the air-cell fire seed below is not a fuel tile)."""
    from simulation import fire_fixed
    from simulation.materials import MAT_WOOD
    for (y, x) in cells:
        assert int(g.material[y, x]) == MAT_WOOD and bool(g.flammable[y, x])
        g.temperature[y, x] = 1263 << 16
        g.fire[y, x] = fire_fixed.quantize_scalar(0.8)


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
    tm[46:50, 50] = 2                        # a wood wall segment (#71: the fuel)

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
    # A trace cloud (the resident EOS's stages 3b/3c and the resident trace
    # tail must carry a non-zero plane).
    trace_ids = [gi for gi in range(g.gas.shape[0])
                 if not bool(g.gases.conservative[gi])]
    assert trace_ids, "scenario needs a trace plane"
    g.gas[trace_ids[0], 20:40, 20:40] += q(0.5)
    # A BURNING wood segment (#71): the combustion twin's claimants.
    _light_wood(g, [(47, 50), (48, 50)])

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
    tm[1, 10:12] = 2       # #71: ring-adjacent wood (ring donors + ring sites)
    tm[40:43, 30] = 2      # #71: an interior wood segment
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
    _light_wood(g, [(1, 10), (1, 11), (41, 30)])

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


# ===========================================================================
# #29 P4a — THE BRACKETS GONE: every synced field, every book, the closure
# identity, on five scenes (residency design §9's P4a row).
# ===========================================================================
# The resident tick now runs the water tail, combustion, fire, the radiation
# sweep and temperature on the device. Each leg below steps a CPU world
# (residency OFF, backends OFF) and a resident world (residency ON) built
# identically, and per tick asserts, at tol 0:
#   * EVERY numpy array on the GameMap (the synced fields, the masks, the
#     sweep and light planes, ripple) — the headless match leg downloads its
#     three render-only sweep planes test-side, outside the tick;
#   * EVERY integer book of the combustion, EOS, temperature solvers and the
#     engine (the e_* channels, the *_hits counters, the *_sum snapshots incl.
#     the solid books), the EOS trace books, the gas-energy seam net, the
#     GameMap trace books, the EOS n_sub / c_LOCAL telemetry and the sweep's
#     stream telemetry;
#   * THE ENERGY CLOSURE IDENTITY (CLAUDE.md), evaluated on BOTH worlds:
#     equal residuals every tick, and 0 from tick 1 (tick 0 starts from the
#     solid books' never-taken snapshot, the same on both worlds).
# and, per leg, that the stage it covers RAN ON THE DEVICE (the resident
# stage counters advanced; P2 once found a gate whose fire was on air cells)
# and that the per-call twins did NOT (the sweep's and the EOS's per-call
# counters, a counting shim on the combustion twin).

def _int_attrs(obj, prefix: str) -> dict:
    """Every integer book on a solver: the ``e_*`` channels, the ``*_hits``
    counters and the ``*_sum`` snapshots (readonly properties; methods are
    skipped and listed separately) — cuda_combustion_burn_check's idiom."""
    out = {}
    for name in dir(obj):
        if name.startswith("_"):
            continue
        if not (name.startswith("e_") or name.endswith("_hits")
                or name.endswith("_sum")):
            continue
        try:
            v = getattr(obj, name)
        except Exception:
            continue
        if callable(v):
            continue
        try:
            out[f"{prefix}.{name}"] = int(v)
        except (TypeError, ValueError):
            continue
    return out


def _all_books(g, runner) -> dict:
    eng = runner.engine
    eos = runner.eos
    out = {}
    out.update(_int_attrs(runner.combustion, "combustion"))
    out.update(_int_attrs(eos, "eos"))
    out.update(_int_attrs(eng.temperature, "temperature"))
    out.update(_int_attrs(eng, "engine"))
    for m in ("boundary_flux", "trace_wipe_sum", "trace_sink_sum",
              "trace_decay_sum", "trace_clip_sum", "trace_mint_sum",
              "trace_neg_hits"):
        out[f"eos.{m}()"] = [int(v) for v in getattr(eos, m)()]
    out["eos.dbg_last_n_sub"] = int(eos.dbg_last_n_sub)
    out["eos.dbg_last_c_local_q"] = int(eos.dbg_last_c_local_q)
    rad = eng.radiation
    out["radiation.min_stream"] = int(rad.min_stream)
    out["radiation.max_stream"] = int(rad.max_stream)
    if eng.light_requested:
        out["radiation.light_books"] = {k: (list(v) if hasattr(v, "__len__") else int(v))
                                        for k, v in dict(rad.light_books).items()}
    out["gmap.gas_energy_seam_net()"] = int(g.gas_energy_seam_net())
    out["gmap.solid_energy_books"] = {k: int(v) for k, v in g.solid_energy_books.items()}
    out["gmap.trace_deposit_books"] = [int(v) for v in g.trace_deposit_books]
    out["gmap.trace_clip_books"] = [int(v) for v in g.trace_clip_books]
    out["gmap.div_source_sat_hits"] = int(g.div_source_sat_hits)
    out["gmap.div_source_cut_q"] = int(g.div_source_cut_q)
    return out


def _gas_total(g) -> int:
    return int(g.gas_energy[g._gas_energy_accountable()].astype(object).sum())


def _terms(g, runner):
    """The closure identity's right-hand side (cuda_combustion_burn_check's
    `_terms`, on a map + runner): EOS (absolute per step), the thermal
    solver's gas side, combustion, the seams, water evac (absolute), the
    solid side."""
    eos, ts, comb, eng = (runner.eos, runner.engine.temperature,
                          runner.combustion, runner.engine)
    return (
        int(eos.e_entry_resync_sum) + int(eos.e_transport_net_sum)
        - int(eos.e_wipe_sum) - int(eos.e_kick_ke_sum)
        + int(eos.e_drag_heat_sum) - int(eos.e_work_export_sum)
        + int(eos.e_rail_sum),
        int(ts.e_gas_deposit_sum) + int(ts.e_gas_cond_sum) + int(ts.e_gas_rail_sum),
        -int(comb.e_comb_draw_sum) + int(comb.e_comb_deliver_sum)
        + int(comb.e_comb_heat_sum) + int(comb.e_comb_rail_sum),
        int(g.gas_energy_seam_net()),
        -int(eng.e_water_evac_export_sum),
        int(ts.e_solid_deposit_sum) + int(ts.e_solid_cond_sum)
        + int(comb.e_comb_solid_heat_sum)
        + int(ts.e_solid_low_rail_sum) + int(ts.e_solid_seam_sum),
    )


def _ledger(g, runner):
    return (_gas_total(g) + int(runner.engine.temperature.solid_energy_books_sum),
            _terms(g, runner))


def _closure_residual(before, after) -> int:
    (tot0, p), (tot1, t) = before, after
    expected = (t[0] + (t[1] - p[1]) + (t[2] - p[2]) + (t[3] - p[3]) + t[4]
                + (t[5] - p[5]))
    return (tot1 - tot0) - expected


def _arrays(g) -> dict:
    return {k: v for k, v in vars(g).items() if isinstance(v, np.ndarray)}


def _compare_all(tag, t, g_cpu, g_gpu, r_cpu, r_gpu) -> int:
    """Every GameMap array and every book, tol 0. Returns the divergences."""
    bad = 0
    a_cpu, a_gpu = _arrays(g_cpu), _arrays(g_gpu)
    if set(a_cpu) != set(a_gpu):
        print(f"  {tag} tick {t}: the two worlds hold different array sets "
              f"({sorted(set(a_cpu) ^ set(a_gpu))})")
        return 1
    for k in sorted(a_cpu):
        x, y = a_cpu[k], a_gpu[k]
        eq = (np.array_equal(x, y, equal_nan=True) if x.dtype.kind == "f"
              else np.array_equal(x, y))
        if not eq:
            bad += 1
            n = int(np.count_nonzero(x != y)) if x.shape == y.shape else -1
            print(f"  {tag} tick {t}: gmap.{k} differs ({n} cells)")
    b_cpu, b_gpu = _all_books(g_cpu, r_cpu), _all_books(g_gpu, r_gpu)
    if set(b_cpu) != set(b_gpu):
        print(f"  {tag} tick {t}: the two worlds expose different books")
        return bad + 1
    for k in sorted(b_cpu):
        if b_cpu[k] != b_gpu[k]:
            bad += 1
            print(f"  {tag} tick {t}: {k} cpu={b_cpu[k]} gpu={b_gpu[k]}")
    return bad


class _CombustionTwinShim:
    """Counts calls into the PER-CALL combustion twin — the resident world
    must make none (its combustion is the resident driver)."""

    def __init__(self):
        self.n = 0
        self._orig = bp.cuda_combustion_step

    def __enter__(self):
        orig = self._orig

        def counted(*a, **k):
            self.n += 1
            return orig(*a, **k)

        bp.cuda_combustion_step = counted
        return self

    def __exit__(self, *exc):
        bp.cuda_combustion_step = self._orig
        return False


def _counters() -> dict:
    return dict(bp.resident_stage_counters())


def _run_leg(label, world_cpu, world_gpu, n_ticks, step_fn, evidence,
             headless_render=False, post_tick=None):
    """THE P4a A/B driver. ``world_*`` = (gmap, runner) of two identically
    built worlds; ``step_fn(world, t)`` steps one world one tick (the caller
    holds the residency switch); ``evidence`` = {counter: minimum delta} the
    resident stage counters must advance by over the run. ``headless_render``:
    the resident world is headless (its render-only sweep planes stay on the
    device), so they are downloaded TEST-side before each compare.
    ``post_tick(t, g_cpu, g_gpu)`` may return extra divergences.
    Returns True on a clean leg."""
    g_c, r_c = world_cpu[0], world_cpu[1]
    g_g, r_g = world_gpu[0], world_gpu[1]
    if _compare_all(label, "init", g_c, g_g, r_c, r_g):
        print(f"  {label}: the two worlds differ before the first tick")
        return False
    c0 = _counters()
    eos_pc0 = int(bp.eos_step_cuda_calls())
    sweep_pc0 = int(bp.radiation_sweep_cuda_calls())
    res0 = int(bp.eos_resident_calls())
    bad = 0
    closed = 0
    with _CombustionTwinShim() as shim:
        for t in range(n_ticks):
            led_c, led_g = _ledger(g_c, r_c), _ledger(g_g, r_g)
            _residency(False); _set_backends(False)
            step_fn(world_cpu, t)
            _residency(True); _set_backends(True)
            step_fn(world_gpu, t)
            _residency(False); _set_backends(False)
            if headless_render:
                g_g.to_host(["rad_net_sweep", "rad_amb_sweep", "rad_fluence"])
            res_c = _closure_residual(led_c, _ledger(g_c, r_c))
            res_g = _closure_residual(led_g, _ledger(g_g, r_g))
            if res_c != res_g:
                bad += 1
                print(f"  {label} tick {t}: the closure residual differs "
                      f"(cpu {res_c}, gpu {res_g})")
            elif t >= 1 and res_c != 0:
                bad += 1
                print(f"  {label} tick {t}: the closure identity does not close "
                      f"(residual {res_c} on BOTH worlds -- a CPU-law finding)")
            elif t >= 1:
                closed += 1
            bad += _compare_all(label, t, g_c, g_g, r_c, r_g)
            if post_tick is not None:
                bad += post_tick(t, g_c, g_g)
            if bad >= 8:
                print(f"  {label}: aborting after 8 divergences")
                break
        percall_comb = shim.n
    c1 = _counters()
    delta = {k: c1[k] - c0[k] for k in c1}
    ok = (bad == 0)
    if int(bp.eos_resident_calls()) - res0 < n_ticks:
        ok = False
        print(f"  {label}: the resident EOS ran only "
              f"{int(bp.eos_resident_calls()) - res0}/{n_ticks} ticks")
    if int(bp.eos_step_cuda_calls()) != eos_pc0:
        ok = False
        print(f"  {label}: the PER-CALL EOS ran in the resident world")
    if int(bp.radiation_sweep_cuda_calls()) != sweep_pc0:
        ok = False
        print(f"  {label}: the PER-CALL sweep ran in the resident world")
    if percall_comb:
        ok = False
        print(f"  {label}: the PER-CALL combustion twin ran {percall_comb} times")
    for k, need in evidence.items():
        if delta.get(k, 0) < need:
            ok = False
            print(f"  {label}: resident stage counter {k} advanced "
                  f"{delta.get(k, 0)} < {need} -- the stage did not run on the "
                  f"device (vacuous)")
    print(f"  {label}: {n_ticks} ticks, "
          f"{'tol 0' if bad == 0 else f'{bad} DIVERGENCES'} on every GameMap "
          f"array ({len(_arrays(g_c))}) and every book "
          f"({len(_all_books(g_c, r_c))}); closure 0 on {closed}/{n_ticks - 1} "
          f"ticks (both worlds); device stages "
          + ", ".join(f"{k} +{v}" for k, v in delta.items() if v))
    return ok


def _raw_step(tick_fn, dt):
    def step(world, t):
        g, runner = world
        tick_fn(runner, g, dt, t)
    return step


def _sim_step(world, t):
    sim = world[2]
    sim.set_paused(False)
    sim.step()


def _session_step(world, t):
    world[2].tick()


def _blast_tick(runner, g, dt, tick_idx):
    """part1a's tick + a BLAST (#31's divergence source, the way the generic
    explosion lands it: add_source over the O2 overpressure pocket) on tick
    5, and a second one on tick 6 at the burning wood."""
    from simulation import gas_fixed
    from simulation.blast_expansion import add_source
    if tick_idx == 5:
        add_source(g, [(y, x) for y in range(10, 16) for x in range(10, 16)],
                   gas_fixed.quantize_q16_scalar(4.0))
    if tick_idx == 6:
        add_source(g, [(y, x) for y in range(45, 51) for x in range(47, 50)],
                   gas_fixed.quantize_q16_scalar(2.0))
    _one_tick(runner, g, dt, tick_idx)


def part1a_bit_identity() -> bool:
    print("PART 1a — SPACE map with a BLAST and burning wood (P4a: every field, "
          "every book, the closure identity), 40 ticks, resident vs CPU, tol 0:")
    H = W = 72
    _residency(False); _set_backends(False)
    r_c, g_c, dt = _build_scenario(H, W)
    r_g, g_g, _ = _build_scenario(H, W)
    ok = _run_leg("1a space+blast", (g_c, r_c), (g_g, r_g), 40,
                  _raw_step(_blast_tick, dt),
                  {"combustion": 40, "fire_gate_open": 40, "sweep": 40,
                   "temperature": 40, "water_tail": 40, "books_downloads": 40})
    if int(r_g.combustion.e_comb_heat_sum) <= 0:
        ok = False
        print("  1a: the resident combustion never burned (vacuous)")
    return ok


def _build_ambient_scenario_sky():
    """part1b's ambient world with the SKY EXCHANGE live (sky_tau_s > 0,
    identical on both worlds): the resident tick's one remaining host
    bracket (ambient maps only, an ACCEPTED GAP) must hold the gate too."""
    import dataclasses
    runner, g, dt = _build_ambient_scenario()
    g._ambient = dataclasses.replace(g._ambient, sky_tau_s=20.0)
    g._sky_mask_dirty = True
    return runner, g, dt


def part1b_ambient() -> bool:
    print("PART 1b — AMBIENT map (sigma + u-damp bands, ring-adjacent breach, the "
          "sky exchange live), 40 ticks, every field / book / closure tol 0:")
    _residency(False); _set_backends(False)
    r_c, g_c, dt = _build_ambient_scenario_sky()
    r_g, g_g, _ = _build_ambient_scenario_sky()
    sky_ticks = [0]

    def step(world, t):
        g, runner = world
        if runner is r_g and runner._sky_exchange_plan(g, dt) is not None:
            sky_ticks[0] += 1
        _one_tick_ambient(runner, g, dt, t)

    ok = _run_leg("1b ambient+sky", (g_c, r_c), (g_g, r_g), 40, step,
                  {"combustion": 40, "sweep": 40, "temperature": 40})
    rail = list(r_c.engine.eos.boundary_flux())
    if not any(v != 0 for v in rail):
        ok = False
        print("  1b: the boundary_flux rail never went non-zero (ring not exercised)")
    if sky_ticks[0] == 0:
        ok = False
        print("  1b: the sky exchange never had work (its bracket not exercised)")
    else:
        print(f"  1b: the sky-exchange bracket ran on {sky_ticks[0]} ticks")
    return ok


def _build_water_scenario():
    """A FLOODING compartment BREACHED TO VACUUM (the water stage's two tail
    laws, live together): a hull-ringed room with a breach in its east wall
    to a vacuum band; a deep pool against the breach (the room vents, its
    pressure falls under boil_p_thresh, the pool BOILS — W5) and a water
    SOURCE in the far corner pouring into dry floor (the front's depth rises
    every tick: the occupancy transition EVACUATES gas — W3, and floods cells
    to the seal). A trace cloud rides along (W3's trace receivers)."""
    from pathlib import Path

    from config import CFG
    from level_loader import LevelData
    from simulation import atmosphere_fixed, water_fixed
    from simulation.gamemap import GameMap
    from simulation.physics_runner import PhysicsRunner

    H, W = 30, 40
    tm = np.zeros((H, W), dtype=np.int32)
    tm[2:H - 2, 2:W - 2] = 1
    tm[3:H - 3, 3:W - 3] = 4
    tm[12:18, W - 3] = 4                     # breach the east hull to the vacuum band
    tm[20:23, 18] = 2                        # a wood stub (thermal solid beside water)
    level = LevelData(name="s8a_water", version="1", path=Path("."),
                      tilemap=tm, tile_size_m=1.0 / 3.0, diffuse_path=Path("."))
    g = GameMap(level)
    g.stamp_units([])
    q = atmosphere_fixed.quantize_scalar
    g.water_depth[8:22, W - 12:W - 3] = water_fixed.quantize_scalar(0.6)
    g.water_sources = [(5, 5, 1.8), (6, 5, 1.8)]
    trace_ids = [gi for gi in range(g.gas.shape[0])
                 if not bool(g.gases.conservative[gi])]
    g.gas[trace_ids[0], 4:12, 4:14] += q(0.5)
    runner = PhysicsRunner(bp)
    g.bind_physics_engine(runner.engine)
    dt = 1.0 / float(CFG.clock.ticks_per_second)
    return runner, g, dt


def _water_tick(runner, g, dt, tick_idx):
    g.stamp_units([])
    destroyed = runner.step(g, dt)
    for (yy, xx) in destroyed:
        g.destroy_wall(yy, xx)
    g.heat.fill(0)


def part1d_water() -> bool:
    print("PART 1d — WATER: a flooding compartment breached to vacuum (W5 boil + "
          "W3 evacuation and seal on the device), 60 ticks, every field / book / "
          "closure tol 0:")
    _residency(False); _set_backends(False)
    r_c, g_c, dt = _build_water_scenario()
    r_g, g_g, _ = _build_water_scenario()
    return _run_leg("1d water", (g_c, r_c), (g_g, r_g), 60,
                    _raw_step(_water_tick, dt),
                    {"water_tail": 60, "water_tail_boil": 1,
                     "water_tail_evac": 1, "sweep": 60, "temperature": 60})


def _studio_with_light():
    """levels/fire_smoke_studio (cuda_combustion_burn_check's live scene: its
    own timed charges heat the fuel at 0.5 s, the HEPA vent loop runs) with
    LIGHT REQUESTED — an omni cone lamp, a sky and the fire-flicker rows on
    the fuel tiles — so the resident sweep's light group and the entry-phase
    light upload are in the gate (the played --resident game requests light)."""
    from level_loader import load as load_level
    from simulation import Simulation
    from simulation.ruleset import ContinuousRealtime
    sim = Simulation(load_level("fire_smoke_studio"), seed=42, breach_physics=bp,
                     enable_recorder=False, ruleset=ContinuousRealtime())
    g = sim.gmap
    ys, xs = np.where(~g.solid & ~g.is_vacuum)
    k = len(ys) // 2
    lamp = np.array([[int(ys[k]), int(xs[k]), 1 << 37, 1 << 37, 1 << 36,
                      0, 65536]], dtype=np.int64)
    sky = np.full((16, 3), 1 << 30, dtype=np.int64)
    fy, fx = np.where(g.flammable & ~g.solid)
    emit = np.array([[int(y), int(x), 2 * 65536] for y, x in zip(fy, fx)],
                    dtype=np.int64).reshape(-1, 3)
    sim.set_light(True, lamp, sky, emit if len(emit) else None)
    return sim


def part1e_fire_studio() -> bool:
    print("PART 1e — fire_smoke_studio burning (lit at 0.5 s by its charges), "
          "LIGHT requested (cone, sky, flicker), 60 ticks, every field / book / "
          "closure tol 0:")
    _residency(False); _set_backends(False)
    s_c, s_g = _studio_with_light(), _studio_with_light()
    w_c = (s_c.gmap, s_c.physics_runner, s_c)
    w_g = (s_g.gmap, s_g.physics_runner, s_g)
    ok = _run_leg("1e fire_smoke_studio+light", w_c, w_g, 60, _sim_step,
                  {"combustion": 60, "fire_gate_open": 20, "sweep_light": 60,
                   "temperature": 60})
    comb = s_g.physics_runner.combustion
    if int(comb.e_comb_heat_sum) <= 0 or int(comb.e_comb_draw_sum) <= 0:
        ok = False
        print("  1e: the resident combustion never burned (vacuous)")
    if not np.any(s_g.gmap.light_q):
        ok = False
        print("  1e: the resident light field is dark (the light group did not ride)")
    return ok


def part1f_ship_demo() -> bool:
    print("PART 1f — ship_demo, the real match (4 v 4 bots + 4 zombies, seed 1, "
          "headless MatchSession), the first 120 ticks, every field / book / "
          "closure tol 0:")
    from level_loader import load as load_level
    from simulation.match_session import MatchSession
    _residency(False); _set_backends(False)
    ses_c = MatchSession(load_level("ship_demo"), seed=1, breach_physics=bp)
    ses_g = MatchSession(load_level("ship_demo"), seed=1, breach_physics=bp)
    for ses in (ses_c, ses_g):
        assert ses.sim.physics_runner.headless, "a MatchSession runs headless"
    w_c = (ses_c.sim.gmap, ses_c.sim.physics_runner, ses_c)
    w_g = (ses_g.sim.gmap, ses_g.sim.physics_runner, ses_g)
    return _run_leg("1f ship_demo", w_c, w_g, 120, _session_step,
                    {"combustion": 120, "fire": 120, "sweep": 120,
                     "temperature": 120, "books_downloads": 120},
                    headless_render=True)


def part1g_ignition_gather() -> bool:
    """#29 P4a: the resident combustion gathers its per-tile ignition
    threshold on the device from `material` (int8) and the per-material
    column, where the host path builds `ignition_temp_q16[material].astype(
    np.int32)` in numpy every tick. Property: the device gather (the tick's
    own kernel, through the test-only probe) equals numpy's on random
    material planes over every row id, -1 (numpy's wrap to the last row)
    included, for the LIVE table and for a table with values past int32
    (the astype wrap). Breaks if the gather mis-indexes, drops the negative
    wrap, or narrows differently from astype."""
    print("PART 1g — the device ignition-plane gather == numpy's "
          "table[material].astype(int32), negative ids included:")
    from config import CFG  # noqa: F401  (the live tables load through it)
    from level_loader import load as load_level
    from simulation.gamemap import GameMap
    g = GameMap(load_level("fire_smoke_studio"))
    live = np.asarray(g.materials.ignition_temp_q16)
    tables = [live, np.concatenate([live, np.array([(1 << 31) + 12345, -7],
                                                   dtype=np.int64)])]
    rng = np.random.default_rng(20261107)
    ok = True
    for k, tbl in enumerate(tables):
        n_mat = len(tbl)
        for shape in ((70, 100), (1, 17), (33, 5)):
            mat = rng.integers(-1, n_mat, size=shape).astype(np.int8)
            want = tbl[mat].astype(np.int32)
            got = bp.resident_ignition_probe(mat, tbl.astype(np.int32))
            if not np.array_equal(np.asarray(got).reshape(shape), want):
                ok = False
                print(f"  table {k} shape {shape}: device gather != numpy "
                      f"({int(np.count_nonzero(np.asarray(got).reshape(shape) != want))} cells)")
    if ok:
        print(f"  {len(tables)} tables x 3 shapes: device == numpy, tol 0")
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
    """Isolate the substep-MULTIPLIED transfer tax of the WATER loop: per-call
    (each substep a separate cudaMalloc + H2D + D2H) vs resident (one batched
    round-trip on persistent buffers). (The 5-plane smoke half this bench used
    to carry is gone with the deleted per-call smoke step — see the docstring.)
    Returns (per_call_ms, resident_ms, n_sub) per iteration."""
    import cupy as cp

    _residency(False); _set_backends(True)
    runner, g, dt = _build_scenario(H, W)
    g.enable_residency()
    dev = g.device_ptrs()
    wp = runner.water
    dx = float(g.tile_size_m)
    n_sub = int(runner.engine.water_substep_count(dt))
    wdt = dt / n_sub

    def per_call():   # the MULTIPLIED per-call path (n_sub malloc/H2D/D2H)
        for _ in range(n_sub):
            bp.cuda_water_step(g.water_depth, g.flow_vx, g.flow_vy,
                               g.floor_height, g.atmosphere, g.solid,
                               wdt, g.tilt_x, g.tilt_y, wp.g, wp.damping,
                               dx, wp.k_p, wp.v_max, wp.depth_eps)

    def resident():   # the RESIDENT path (1 batched round-trip)
        g.from_host(["water_depth", "flow_vx", "flow_vy", "atmosphere", "solid"])
        bp.water_substeps_resident(
            dev["water_depth"], dev["flow_vx"], dev["flow_vy"],
            dev["floor_height"], dev["atmosphere"], dev["solid"],
            H, W, n_sub, wdt, g.tilt_x, g.tilt_y,
            wp.g, wp.damping, dx, wp.k_p, wp.v_max, wp.depth_eps)
        g.to_host(["water_depth", "flow_vx", "flow_vy"])

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
    return pc, res, n_sub


def _bench_trace_tail(H, W, reps=100):
    """Smoke transport v2 (#12, P2b): the resident trace tail vs ONE host
    round-trip of the gas block (to_host + from_host of `gas` — the minimum a
    host-side tail would add to the resident tick). Every trace plane carries
    a live, non-uniform field so the tail does real work on every plane.
    Returns (tail_ms, roundtrip_ms, n_trace) per iteration."""
    import cupy as cp

    _residency(False); _set_backends(True)
    runner, g, dt = _build_scenario(H, W)
    q = 65536
    trace_ids = [gi for gi in range(g.gas.shape[0])
                 if not bool(g.gases.conservative[gi])]
    for k, gi in enumerate(trace_ids):
        g.gas[gi, 4:H - 4, 4:W // 2] += (q // 4) + (k + 1) * 97
    g.enable_residency()
    g.from_host(["gas", "solid", "is_vacuum", "dyn_permeability"])
    dev = g.device_ptrs()
    eos = runner.engine.eos

    def tail():
        runner.engine.run_trace_tail_resident(
            g.gases.conservative, g.gases.diffusion, g.gases.decay,
            H, W, dt, d_gas=dev["gas"], d_solid=dev["solid"],
            d_is_vacuum=dev["is_vacuum"], d_is_ambient=0,
            d_dyn_permeability=dev["dyn_permeability"])

    def roundtrip():
        g.to_host(["gas"])
        g.from_host(["gas"])

    tail(); roundtrip(); cp.cuda.Stream.null.synchronize()   # warm-up
    if int(sum(eos.trace_decay_sum())) <= 0:
        raise AssertionError("trace-tail bench is vacuous: the tail decayed nothing")

    def timeit(fn):
        best = float("inf")
        for _ in range(3):
            t0 = time.perf_counter()
            for _ in range(reps):
                fn()
            cp.cuda.Stream.null.synchronize()
            best = min(best, time.perf_counter() - t0)
        return 1e3 * best / reps
    t_tail, t_rt = timeit(tail), timeit(roundtrip)
    _residency(False); _set_backends(False)
    return t_tail, t_rt, len(trace_ids)


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
            runner._black_smoke_idx = int(g.gases.name_to_id["smoke"])
        runner.eos.dx = float(g.tile_size_m)
        if resident:
            g.enable_residency()
        return runner, g, dt

    def make_percall(runner, g, dt):
        amb = runner._ambient_args(g)

        def fn():
            runner.engine.run_substeps(
                g.wave_p, g.atmosphere, g.wind_x, g.wind_y, g.temperature,
                g.gas_energy,          # arc #54 §2.2 (MECHANICAL)
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
            # arc #54 §2.2 (P-G0/P-G2): `gas_energy` rides the resident
            # upload/D2H pair beside `temperature` (physics_runner.py's
            # `_step_resident` precedent) — K1's KE brackets, K3's face-flux
            # step and the once-per-tick recovery all read/write its device
            # buffer now, so an omitted d_gas_energy is a null device pointer
            # the resident bulk-flux D2D e_pre snapshot then faults on.
            g.from_host(["atmosphere", "wind_x", "wind_y", "temperature",
                         "gas", "solid", "is_vacuum", "is_ambient",
                         "dyn_permeability", "gas_energy"])
            runner.engine.run_substeps_resident(
                g.wave_p, g.atmosphere, g.wind_x, g.wind_y, g.temperature,
                g.solid, g.is_vacuum, g.dyn_permeability, g.dyn_wave_absorb,
                g.gas, g.gases.conservative, dt,
                is_ambient=amb[0], n_amb=amb[1], p_amb=amb[2],
                d_atmosphere=dev["atmosphere"], d_wave_p=dev["wave_p"],
                d_wind_x=dev["wind_x"], d_wind_y=dev["wind_y"],
                d_temperature=dev["temperature"], d_gas=dev["gas"],
                d_solid=dev["solid"], d_is_vacuum=dev["is_vacuum"],
                d_dyn_permeability=dev["dyn_permeability"],
                d_gas_energy=dev["gas_energy"])
            g.to_host(["atmosphere", "wave_p", "wind_x", "wind_y",
                       "temperature", "gas", "gas_energy"])
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
    print("PART 2 — the payoff (the substep-MULTIPLIED transfer tax must be "
          "gone). ISOLATED water-substep loop, per-call vs resident at >=2 grid "
          "sizes (the tax is a grid-AREA cost -> resident wins MORE on bigger "
          "grids):")
    sizes = [(128, 128), (256, 256), (384, 384)]
    ratios = []
    for (H, W) in sizes:
        pc, res, n_sub = _bench_tax(H, W)
        ratio = pc / max(res, 1e-9)
        ratios.append(ratio)
        print(f"  {H:3d}x{W:<3d} (water x{n_sub} substeps): "
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
    # Same robustness shape as the water assert above: wins at every
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

    # Smoke transport v2 (#12, P2b) — (iii): the resident trace tail is
    # transfer-free. Property: at every size it costs less than ONE host
    # round-trip of the gas block (what any host-side tail would add), so no
    # plane crosses the bus. The tail is a handful of launches + ~100 B of
    # fold/book traffic, the round-trip two full-block copies; a regression to
    # a host tail or a per-plane transfer makes the tail at least as slow as
    # the round-trip it would contain.
    print("  smoke transport v2 — the resident trace tail vs ONE host "
          "round-trip of the gas block:")
    tail_ok = True
    for (H, W) in [(128, 128), (256, 256), (384, 384)]:
        t_tail, t_rt, n_tr = _bench_trace_tail(H, W)
        print(f"  {H:3d}x{W:<3d} ({n_tr} trace planes): resident tail "
              f"{t_tail:7.3f} | gas round-trip {t_rt:7.3f} ms")
        if not t_tail < t_rt:
            tail_ok = False
            print(f"  FAIL: the resident trace tail ({t_tail:.3f} ms) is not "
                  f"cheaper than a gas-block round-trip ({t_rt:.3f} ms) at "
                  f"{H}x{W} — has the tail grown a host transfer?")
    if tail_ok:
        print("  the resident trace tail moves no plane across the bus "
              "(cheaper than one gas round-trip at every size).")
    ok = ok and tail_ok

    # Full-engine context (informational): with Path A the EOS bracket is gone,
    # and since #29 P4a the water tail, combustion and the tail (fire, the
    # sweep, temperature) run on the device too -- no bracket remains on a
    # space map.
    cpu = _bench_run(256, 256, 20, residency=False, backends=False)
    percall = _bench_run(256, 256, 20, residency=False, backends=True)
    resident = _bench_run(256, 256, 20, residency=True, backends=True)
    print(f"  [full tick @256x256, informational: CPU {cpu:.1f} | per-call GPU "
          f"{percall:.1f} | RESIDENT {resident:.1f} ms/tick -- the whole of slot 7 "
          f"resident (#29 P4a)]")
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
    p1d = part1d_water()
    p1e = part1e_fire_studio()
    p1f = part1f_ship_demo()
    p1g = part1g_ignition_gather()
    p2 = part2_payoff()
    if p1a and p1b and p1c and p1d and p1e and p1f and p1g and p2:
        print("S8A_RESULT: PASS")
        return 0
    print("S8A_RESULT: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
