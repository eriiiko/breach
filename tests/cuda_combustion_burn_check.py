"""#71 / residency arc #29 P2 — the CUDA combustion twin on a BURNING scene.

Runs inside the GPU subprocess (tests/cuda_harness.run_cuda_script; wrapper
tests/test_cuda_combustion_burn.py). THE gate the existing combustion checks
(cuda_combustion_check, cuda_po2b_check, ...) never reached: those drive the
pass in isolation WITHOUT the arc #54 gas energy field, so a twin that
predated the seam passed them while it parted from the CPU solver on tick 0 of
every burning scene (#71: 6 cells of temperature / gas_energy around the fire
on tick 0, the whole field by tick 1).

  PART 1 — ISOLATED, ENERGY ON: CPU ``CombustionSolver.step`` vs
  ``cuda_combustion_step`` with ``gas_energy`` / ``is_ambient`` / ``t_amb_q``
  on random scenes built to reach every branch of the ledger: draw radius 1, 2
  and 3 (the hop >= 2 remainder's energy split), donors on the ambient ring
  (parcels MINTED at ambient), furniture burn sites (the object deposit, the
  parcel to a thermal solid, the solid-heat book), ring deposit sites (the
  export), thin cells under ``n_floor_heat`` (the floor and its drop book) and
  hot cells at the ``T_MAX_PHYS`` rail (the railed seam's write-back). Every
  plane and every book at tol 0, and the scenes are asserted to have driven
  every channel (a vacuity guard per book).

  PART 2 — THE LIVE TICK, TWO BURNING SCENES: the #71 repro (the playground,
  a wood tile heated to the 1263-game plateau and lit, a marine beside it) and
  the fire + smoke studio (five furniture tiles and a lone one lit at 0.5 s by
  its own timed charges, a HEPA vent loop over them) — each built twice
  through the real ``Simulation`` and stepped through ``PhysicsRunner.step``
  with ONLY the combustion backend flipped. Per tick, tol 0 on EVERY GameMap
  array and EVERY integer book of the combustion, EOS and temperature solvers
  and the engine (rail hits, floor hits, every ``e_*`` channel, the trace
  books, the seam net), and the across-tick energy closure identity
  (``Δ(Σ gas_energy + solid books) == EOS + thermal + combustion + seams +
  water``) evaluated on BOTH worlds with the residuals required equal (and
  reported). Vacuity guards: the GPU dispatch fired on every tick (a counting
  shim on ``bp.cuda_combustion_step``), a tile burned, and the GPU world's
  ``e_comb_heat_sum`` / ``e_comb_draw_sum`` / ``e_comb_deliver_sum`` moved.

Prints ``COMB_BURN_RESULT: PASS`` / ``FAIL`` and exits 0 / 1.
"""
from __future__ import annotations

import random
import sys

import numpy as np

import breach_physics as bp   # the CUDA build (cuda_harness bootstrap)

FP_ONE = 65536
_FAILS: list = []


def _fail(msg: str) -> None:
    _FAILS.append(msg)
    print(f"  FAIL: {msg}")


_ALL_BACKENDS = (
    "set_temperature_backend", "set_water_backend",
    "set_fire_backend", "set_radiation_backend",
    "set_bulk_flux_backend", "set_sl_advection_backend",
    "set_mg_solve_backend", "set_kick_compression_backend",
    "set_combustion_backend",
)


def _only_combustion(on: bool) -> None:
    """Every backend OFF and residency OFF, then the combustion backend alone."""
    from simulation import physics_runner
    physics_runner.set_residency(False)
    for name in _ALL_BACKENDS:
        getattr(bp, name)(False)
    bp.set_combustion_backend(bool(on))


# ---------------------------------------------------------------------------
# the books
# ---------------------------------------------------------------------------
def _int_attrs(obj, prefix: str) -> dict:
    """Every integer book on a solver: the ``e_*`` channels, the ``*_hits``
    counters and the ``*_sum`` snapshots (readonly properties; methods are
    skipped and listed separately)."""
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


def _books(sim) -> dict:
    rp = sim.physics_runner
    eng = rp.engine
    g = sim.gmap
    out = {}
    out.update(_int_attrs(rp.combustion, "combustion"))
    out.update(_int_attrs(rp.eos, "eos"))
    out.update(_int_attrs(eng.temperature, "temperature"))
    out.update(_int_attrs(eng, "engine"))
    for m in ("boundary_flux", "trace_wipe_sum", "trace_sink_sum",
              "trace_decay_sum", "trace_clip_sum", "trace_mint_sum",
              "trace_neg_hits"):
        out[f"eos.{m}()"] = [int(v) for v in getattr(rp.eos, m)()]
    out["gmap.gas_energy_seam_net()"] = int(g.gas_energy_seam_net())
    out["gmap.trace_deposit_books"] = [int(v) for v in g.trace_deposit_books]
    out["gmap.trace_clip_books"] = [int(v) for v in g.trace_clip_books]
    return out


# THE ENERGY CLOSURE IDENTITY (CLAUDE.md "Energy closure identity") — the
# full ledger's right-hand side, the shape tests/test_heat_source.py and
# tests/test_thermostat_books.py use: EOS (absolute per step), the thermal
# solver's gas side, combustion, the seams, water evac (absolute), the solid side.
def _gas_total(g) -> int:
    return int(g.gas_energy[g._gas_energy_accountable()].astype(object).sum())


def _terms(sim):
    g = sim.gmap
    rp = sim.physics_runner
    eos, ts, comb, eng = rp.eos, rp.engine.temperature, rp.combustion, rp.engine
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


def _ledger(sim):
    return (_gas_total(sim.gmap)
            + int(sim.physics_runner.engine.temperature.solid_energy_books_sum),
            _terms(sim))


def _closure_residual(before, after) -> int:
    (tot0, p), (tot1, t) = before, after
    expected = (t[0] + (t[1] - p[1]) + (t[2] - p[2]) + (t[3] - p[3]) + t[4]
                + (t[5] - p[5]))
    return (tot1 - tot0) - expected


# ---------------------------------------------------------------------------
# PART 1 — isolated, energy on
# ---------------------------------------------------------------------------
_E_BOOKS = ("e_comb_draw_sum", "e_comb_mint_sum", "e_comb_deliver_sum",
            "e_soot_shed_sum", "e_ts_products_sum", "e_comb_export_sum",
            "e_comb_heat_sum", "e_comb_rail_sum", "e_comb_solid_heat_sum")
_RAIL_BOOKS = ("heat_floor_hits", "t_max_phys_hits", "e_deposit_drop_sum")
_DIALS = ("burn_rate", "o2_frac_ext", "o2_frac_full", "o2_thresh_burn",
          "H_FUEL_M", "H_FUEL_SHIFT", "soot_yield", "fuel_per_o2",
          "fire_T_ext", "fire_T_span", "hotf_cap", "p_ext_q", "p_full_q",
          "H_BED_M", "H_BED_SHIFT", "T_MAX_PHYS")


def _live_dials():
    """The LIVE engine's combustion dials + the deposit currency, read off a
    real Simulation (so PART 1 runs the shipped arithmetic, not a fixture's)."""
    from level_loader import load as load_level
    from simulation import Simulation
    sim = Simulation(load_level("playground"), seed=1, breach_physics=bp,
                     enable_recorder=False)
    rp = sim.physics_runner
    dials = {k: getattr(rp.combustion, k) for k in _DIALS}
    return (dials, float(rp.temperature.c_v), float(rp.temperature.n_floor_heat),
            int(rp._eos_t_amb_raw()))


def _solver(dials):
    s = bp.CombustionSolver()
    for k, v in dials.items():
        setattr(s, k, v)
    return s


def _scene(rng: random.Random, h: int, w: int, t_amb_q: int, t_max_q: int,
           n_floor_q: int, ring: bool) -> dict:
    """One random scene reaching the ledger's branches (see PART 1 above)."""
    from simulation.gases import INERT_N2, O2, SMOKE
    nr = np.random.default_rng(rng.randrange(1 << 30))
    solid = nr.random((h, w)) < 0.22
    is_vacuum = (~solid) & (nr.random((h, w)) < 0.04)
    furn = (~solid) & (~is_vacuum) & (nr.random((h, w)) < 0.14)
    is_ambient = np.zeros((h, w), dtype=bool)
    if ring:
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
        is_ambient = border & ~solid & ~is_vacuum & ~furn
    flammable = (solid & (nr.random((h, w)) < 0.7)) | furn
    thermal_solid = furn | (solid & (nr.random((h, w)) < 0.6))
    heat_inv_shift = np.where(thermal_solid, nr.integers(-2, 5, (h, w)), 0).astype(np.int32)
    open_ = ~solid & ~is_vacuum
    n_gases = 7
    gas = np.zeros((n_gases, h, w), dtype=np.int32)
    n_tot = np.where(nr.random((h, w)) < 0.15,
                     nr.integers(1, max(2, n_floor_q), (h, w)),        # thin
                     nr.integers(FP_ONE // 3, 3 * FP_ONE, (h, w)))      # ~ambient
    frac = nr.uniform(0.14, 0.7, (h, w))
    o2 = (n_tot * frac).astype(np.int64)
    gas[O2] = np.where(open_, o2, 0)
    gas[INERT_N2] = np.where(open_, n_tot - o2, 0)
    gas[SMOKE] = np.where(open_, nr.integers(0, 1 << 22, (h, w)), 0)
    # Temperatures (relative, Q16 game): thermal solids around their ignition
    # range, a few cells at the rail; gas cells from their stored energy.
    t_rel = nr.integers(-150 * FP_ONE, 1500 * FP_ONE, (h, w)).astype(np.int64)
    hot = nr.random((h, w)) < 0.25
    t_rel = np.where(hot, t_max_q - nr.integers(0, 2 * FP_ONE, (h, w)), t_rel)
    n_b = (gas[O2].astype(np.int64) + gas[INERT_N2])
    acct = open_ & ~thermal_solid & ~is_ambient
    gas_energy = np.where(
        acct, n_b * (t_rel + t_amb_q) + nr.integers(0, 1 << 30, (h, w)) % np.maximum(n_b, 1),
        0).astype(np.int64)
    temperature = t_rel.astype(np.int32).copy()
    floor_div = np.where(n_b > 0, gas_energy // np.maximum(n_b, 1) - t_amb_q, 0)
    temperature[acct] = floor_div[acct].astype(np.int32)
    ign = np.where(flammable, nr.choice([0, 280, 450, 600], (h, w)) * FP_ONE, 0).astype(np.int32)
    fire = np.where(flammable & (nr.random((h, w)) < 0.6),
                    nr.integers(FP_ONE // 20, FP_ONE, (h, w)), 0).astype(np.int32)
    wall_hp = np.where(flammable, nr.choice([1, 2, 30 * FP_ONE, 60 * FP_ONE], (h, w)),
                       0).astype(np.int32)
    perm = np.where(solid, 0.0, np.where(furn, 0.5, 1.0)).astype(np.float32)
    atm = nr.integers(FP_ONE // 20, 2 * FP_ONE, (h, w)).astype(np.int32)
    fpo = np.where(flammable, nr.integers(FP_ONE // 4, 2 * FP_ONE, (h, w)), 0).astype(np.int32)
    t_ext = np.where(flammable, nr.choice([180, 350], (h, w)) * FP_ONE, 0).astype(np.int32)
    return dict(gas=gas, temperature=temperature, wall_hp=wall_hp, fire=fire,
                flammable=flammable, solid=solid, is_vacuum=is_vacuum,
                ignition_temp_q16=ign, thermal_solid=thermal_solid,
                heat_inv_shift=heat_inv_shift,
                heat=np.zeros((h, w), dtype=np.int32),
                dem_acc=nr.integers(0, 1 << 31, (24, h, w)).astype(np.int32),
                dyn_permeability=perm, gas_energy=gas_energy, is_ambient=is_ambient,
                fire_T_ext_plane=t_ext, fuel_per_o2_plane=fpo, atmosphere=atm)


def _copy(st: dict) -> dict:
    return {k: v.copy() for k, v in st.items()}


def _run_pair(st, dials, c_v, n_floor, t_amb_q, draw_r, dt):
    from simulation.gases import INERT_N2, O2, SMOKE
    cpu, gpu = _copy(st), _copy(st)
    s_cpu, s_gpu = _solver(dials), _solver(dials)
    s_cpu.step(cpu["gas"], O2, INERT_N2, SMOKE, cpu["temperature"], cpu["wall_hp"],
               cpu["fire"], cpu["flammable"], cpu["solid"], cpu["is_vacuum"],
               cpu["ignition_temp_q16"], dt, c_v, n_floor,
               thermal_solid=cpu["thermal_solid"], heat_inv_shift=cpu["heat_inv_shift"],
               heat=cpu["heat"], dem_acc=cpu["dem_acc"], draw_r=draw_r,
               dyn_permeability=cpu["dyn_permeability"], max_claimants=24,
               gas_energy=cpu["gas_energy"], is_ambient=cpu["is_ambient"],
               t_amb_q=t_amb_q, fire_T_ext_plane=cpu["fire_T_ext_plane"],
               fuel_per_o2_plane=cpu["fuel_per_o2_plane"], atmosphere=cpu["atmosphere"])
    d = dials
    bp.cuda_combustion_step(
        gpu["gas"], O2, INERT_N2, SMOKE, gpu["temperature"], gpu["wall_hp"],
        gpu["fire"], gpu["flammable"], gpu["solid"], gpu["is_vacuum"],
        gpu["ignition_temp_q16"], dt, c_v, n_floor,
        d["burn_rate"], d["o2_thresh_burn"], d["H_FUEL_M"], d["H_FUEL_SHIFT"],
        d["soot_yield"], d["fuel_per_o2"], d["o2_frac_ext"], d["o2_frac_full"],
        d["T_MAX_PHYS"], gpu["thermal_solid"], gpu["heat_inv_shift"],
        gpu["heat"], d["H_BED_M"], d["H_BED_SHIFT"], gpu["dem_acc"],
        draw_r, gpu["dyn_permeability"], 24,
        d["fire_T_ext"], d["fire_T_span"], d["hotf_cap"],
        gpu["fire_T_ext_plane"], gpu["fuel_per_o2_plane"],
        atmosphere=gpu["atmosphere"], p_ext_q=d["p_ext_q"], p_full_q=d["p_full_q"],
        gas_energy=gpu["gas_energy"], is_ambient=gpu["is_ambient"],
        t_amb_q=t_amb_q, combustion=s_gpu)
    return cpu, gpu, s_cpu, s_gpu


def part1_isolated() -> None:
    print("PART 1 -- isolated, energy on: CPU step vs cuda_combustion_step, tol 0")
    dials, c_v, n_floor, t_amb_q = _live_dials()
    from simulation import gas_fixed
    t_max_q = gas_fixed.quantize_q16_scalar(float(dials["T_MAX_PHYS"]))
    n_floor_q = gas_fixed.quantize_q16_scalar(n_floor)
    # The scenes run with p_full dormant half the time so both branches of
    # the pressure factor are exercised (the live edges otherwise).
    driven = {k: 0 for k in _E_BOOKS + _RAIL_BOOKS}
    n_cases = 0
    for case in range(48):
        rng = random.Random(20261107 + case)
        h, w = rng.choice([(9, 11), (14, 10), (17, 23), (6, 6), (1, 12), (12, 1)])
        draw_r = (1, 2, 3)[case % 3]
        st = _scene(rng, h, w, t_amb_q, t_max_q, n_floor_q, ring=(case % 2 == 0))
        dl = dict(dials)
        if case % 4 == 3:
            dl["p_ext_q"], dl["p_full_q"] = 0, 0
        cpu, gpu, s_cpu, s_gpu = _run_pair(st, dl, c_v, n_floor, t_amb_q, draw_r,
                                           1.0 / 24.0)
        tag = f"P1 case {case} ({h}x{w}, draw_r {draw_r})"
        for k in cpu:
            if not np.array_equal(cpu[k], gpu[k]):
                _fail(f"{tag}: {k} differs ({int(np.count_nonzero(cpu[k] != gpu[k]))} cells)")
        for k in _E_BOOKS + _RAIL_BOOKS:
            a, b = int(getattr(s_cpu, k)), int(getattr(s_gpu, k))
            if a != b:
                _fail(f"{tag}: {k} cpu={a} gpu={b}")
            driven[k] += int(a != 0)
        n_cases += 1
    for k, v in driven.items():
        if v == 0:
            _fail(f"PART 1 never drove {k} -- the fuzz is vacuous for that book")
    print(f"  {n_cases} cases; cases driving each book: "
          + ", ".join(f"{k}={v}" for k, v in driven.items()))


# ---------------------------------------------------------------------------
# PART 2 — the live tick, burning scenes
# ---------------------------------------------------------------------------
def _burning_playground():
    """THE #71 REPRO — tests/cuda_radiation_sweep_check.py's live scene (and
    tests/test_radiation_sweep_shadow_wiring.py's): the playground with a wood
    tile HEATED to the 1263-game plateau and lit (a fire is started by
    delivering heat, CLAUDE.md "Starting a fire"), a marine on open air beside
    it."""
    from level_loader import load as load_level
    from simulation import Simulation, fire_fixed
    from simulation.materials import MAT_WOOD
    from simulation.unit import Unit
    sim = Simulation(load_level("playground"), seed=1, breach_physics=bp,
                     enable_recorder=False)
    g = sim.gmap
    ys, xs = np.where((g.material == MAT_WOOD) & g.thermal_solid)
    pick = None
    for y, x in zip(ys, xs):
        if 0 < y < g.material.shape[0] - 1 and 0 < x < g.material.shape[1] - 1:
            if not g.thermal_solid[y, x + 1] or not g.thermal_solid[y + 1, x]:
                pick = (int(y), int(x))
                break
    assert pick is not None, "the playground carries no wood tile beside air"
    # The repro's DIRECT write on a thermal solid is outside every book, so the
    # solid side's snapshot jumps by cap_real * dT on tick 0 (the capacity
    # law of conduction::cell_capacity_q: 2^(clamp(s, -16, 30) + 16)).
    s_real = min(max(int(g.heat_inv_shift[pick]), -16), 30)
    injected = ((1263 << 16) - int(g.temperature[pick])) * (1 << (s_real + 16))
    g.temperature[pick] = 1263 << 16
    g.fire[pick] = fire_fixed.quantize_scalar(0.8)
    y0, x0 = pick
    spot = None
    for r in (2, 3, 1, 4):
        for dy, dx in ((0, -r), (0, r), (r, 0), (-r, 0), (r, r), (-r, -r)):
            y, x = y0 + dy, x0 + dx
            if (0 < y < g.material.shape[0] - 1 and 0 < x < g.material.shape[1] - 1
                    and not g.solid[y, x] and not g.thermal_solid[y, x]
                    and not g.is_vacuum[y, x]):
                spot = (y, x)
                break
        if spot:
            break
    assert spot is not None, "no open air cell near the fire for the marine"
    sim.add_unit(Unit("M1", x=spot[1], y=spot[0], team=0))
    sim._p2_fixture_injection = injected
    return sim


def _fire_studio():
    """levels/fire_smoke_studio as the fire-smoke probe runs it: its own timed
    charges heat each fuel tile past ignition at 0.5 s and the ordinary
    temperature ignition lights it a tick later; the HEPA vent loop runs."""
    from level_loader import load as load_level
    from simulation.ruleset import ContinuousRealtime
    from simulation import Simulation
    return Simulation(load_level("fire_smoke_studio"), seed=42, breach_physics=bp,
                      enable_recorder=False, ruleset=ContinuousRealtime())


def _arrays(g) -> dict:
    return {k: v for k, v in vars(g).items() if isinstance(v, np.ndarray)}


def _compare(tag, t, s_cpu, s_gpu) -> int:
    bad = 0
    a_cpu, a_gpu = _arrays(s_cpu.gmap), _arrays(s_gpu.gmap)
    if set(a_cpu) != set(a_gpu):
        _fail(f"{tag} tick {t}: the two worlds hold different array sets")
        return 1
    for k in sorted(a_cpu):
        x, y = a_cpu[k], a_gpu[k]
        eq = (np.array_equal(x, y, equal_nan=True) if x.dtype.kind == "f"
              else np.array_equal(x, y))
        if not eq:
            bad += 1
            n = int(np.count_nonzero(x != y)) if x.shape == y.shape else -1
            _fail(f"{tag} tick {t}: gmap.{k} differs ({n} cells)")
    b_cpu, b_gpu = _books(s_cpu), _books(s_gpu)
    if set(b_cpu) != set(b_gpu):
        _fail(f"{tag} tick {t}: the two worlds expose different books")
        return bad + 1
    for k in sorted(b_cpu):
        if b_cpu[k] != b_gpu[k]:
            bad += 1
            _fail(f"{tag} tick {t}: {k} cpu={b_cpu[k]} gpu={b_gpu[k]}")
    return bad


class _CallCounter:
    """Counts the runner's calls into the GPU twin (the dispatch looks the
    function up on the module every tick, so a module attribute shim sees
    every one) — a 'GPU run' that silently took the CPU branch is vacuous."""

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


def _run_live(tag: str, builder, n_ticks: int) -> None:
    fails_before = len(_FAILS)
    s_cpu, s_gpu = builder(), builder()
    if _compare(tag, "init", s_cpu, s_gpu):
        return
    comb_g = s_gpu.physics_runner.combustion
    heat0, draw0, deliv0 = (int(comb_g.e_comb_heat_sum), int(comb_g.e_comb_draw_sum),
                            int(comb_g.e_comb_deliver_sum))
    burning = 0
    res_cpu_nz = res_gpu_nz = 0
    worst = 0
    with _CallCounter() as calls:
        for t in range(n_ticks):
            led_c, led_g = _ledger(s_cpu), _ledger(s_gpu)
            _only_combustion(False)
            s_cpu.set_paused(False)
            s_cpu.step()
            c0 = calls.n
            _only_combustion(True)
            s_gpu.set_paused(False)
            s_gpu.step()
            _only_combustion(False)
            if calls.n != c0 + 1:
                _fail(f"{tag} tick {t}: the GPU combustion dispatch ran "
                      f"{calls.n - c0} times (expected once)")
            r_c = _closure_residual(led_c, _ledger(s_cpu))
            r_g = _closure_residual(led_g, _ledger(s_gpu))
            if t == 0:   # a fixture's direct write before tick 0, if any
                inj = int(getattr(s_cpu, "_p2_fixture_injection", 0))
                r_c, r_g = r_c - inj, r_g - inj
            res_cpu_nz += int(r_c != 0)
            res_gpu_nz += int(r_g != 0)
            worst = max(worst, abs(r_c), abs(r_g))
            if r_c != r_g:
                _fail(f"{tag} tick {t}: the closure residual differs "
                      f"(cpu {r_c}, gpu {r_g})")
            elif r_c != 0:
                _fail(f"{tag} tick {t}: the closure identity does not close "
                      f"(residual {r_c} on BOTH worlds -- a CPU-law finding)")
            burning += int(np.any(s_cpu.gmap.fire > 0))
            if _compare(tag, t, s_cpu, s_gpu) >= 4:
                print("  aborting after 4 divergent books / arrays")
                break
        n_calls = calls.n
    if n_calls != n_ticks:
        _fail(f"{tag}: {n_calls} GPU combustion calls in {n_ticks} ticks")
    moved = {
        "e_comb_heat_sum": int(comb_g.e_comb_heat_sum) - heat0,
        "e_comb_draw_sum": int(comb_g.e_comb_draw_sum) - draw0,
        "e_comb_deliver_sum": int(comb_g.e_comb_deliver_sum) - deliv0,
    }
    for k, v in moved.items():
        if v <= 0:
            _fail(f"{tag}: the GPU world's {k} did not move ({v}) -- the scene "
                  "never burned through the twin")
    if burning == 0:
        _fail(f"{tag}: no tile ever burned -- the live scene is vacuous")
    verdict = "tol 0" if len(_FAILS) == fails_before else "FAILED"
    print(f"  {tag}: {n_ticks} ticks, {verdict} on every GameMap array and "
          f"{len(_books(s_cpu))} books; GPU calls {n_calls}; burning ticks "
          f"{burning}; closure residual 0 on "
          f"{n_ticks - res_cpu_nz}/{n_ticks} ticks (both worlds); GPU world "
          + ", ".join(f"d{k} {v}" for k, v in moved.items()))


def part2_live() -> None:
    print("PART 2 -- the live tick, burning scenes, ONLY the combustion backend flipped")
    _run_live("2a the #71 repro (playground, a lit 1263-game wood tile)",
              _burning_playground, 40)
    _run_live("2b fire_smoke_studio (lit at 0.5 s by its charges)",
              _fire_studio, 60)
    _only_combustion(False)


def main() -> int:
    if not getattr(bp, "HAS_CUDA", False) or not bp.cuda_available():
        print("COMB_BURN_RESULT: FAIL (no CUDA build / device)")
        return 1
    part1_isolated()
    part2_live()
    if _FAILS:
        print(f"COMB_BURN_RESULT: FAIL ({len(_FAILS)} failures)")
        return 1
    print("COMB_BURN_RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
