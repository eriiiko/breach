"""P5c -- the temperature twin's GAS radiation branch, CPU vs CUDA at tol 0
(runs inside the GPU subprocess; ray-engine-v2 issue #12, design v3 §2.8 / §6.3).

P5c opens the Pass-1 fold to accountable gas cells on BOTH temperature backends
in one patch (CLAUDE.md "Radiation sweep": the CUDA twin lands with the CPU
change): cuda_temperature.cu's temp_convert_unified carries the CPU block
verbatim -- the staged chain through the FP_HD kit, the clamp's energy form, the
railed seam deposit, e_rad_clamp_drop_sum in the APPENDED slot 12 -- and, since
the P5c follow-up (design 8.4), the boundary's two other exits in the APPENDED
slots 13 and 14: e_rad_boundary_export_sum (rad_net on a gas cell outside the
accountable set, exported) and e_rad_floor_drop_sum (the floored chain's
unlanded remainder below n_floor_heat). This check holds the two to each other.

  PART 1 -- ISOLATED (TemperatureSolver.step vs cuda_temperature_step, one
  call): random scenes with accountable gas cells (stored residuals E mod N,
  bulk from the N_EPS edge to 3 atm, rad_net of both signs, caps above and
  below), thermal solids (thin rows included), vacuum and ambient-ring cells, a
  HEAT deposit on some of every kind (Pass 1's second half runs after the fold
  on the same cell) and live conduction faces (Pass 2 reads the fold's mirror)
  -- clamp on and off, LIVE and resolving tables. Tol 0 on `temperature`,
  `gas_energy` and every counter the isolated entry returns, including slots
  12-14. Non-vacuous per config: the gas branch booked, with the clamp on it
  bound on a gas cell, and both boundary counters moved. P5d (brief §5.6):
  per table, the clamp binds AT the headroom ceiling e_ceiling_q(Phi) -- the
  top of the first E° bucket out-emitting Phi, where the pre-P5d E°⁻¹(Phi)
  would land one to two buckets lower -- on thermal solids AND on accountable
  gas cells, counted from the scene's inputs through the reference's arithmetic
  (which the CPU fold equals bit for bit), so a device clamp still at E°⁻¹
  cannot pass the tol-0 comparison (measured: a build with only the device's
  two clamp lines back on e_inv_q fails it).

  PART 2 -- THE LIVE CONDUCTOR: Simulation.step on a sealed room of hot smoke
  radiating into its hull (the shipped coefficient), on the smoke-shield room
  (a held 1263-game column shining through a smoke layer at a marine), on the
  same room OVER-DRIVEN (the column held at 8000 game; #78), and on
  the two breached rooms of test_temperature_gas_radiation's closure test (a
  burning smoky room vented to space; one breached into an ambient ring whose
  air absorbs -- a fixture), the temperature backend flipping CPU <-> CUDA
  every tick in one world against a CPU-only world: every GameMap array and
  every temperature counter (slots 12-14 included) at tol 0, tick for tick.
  Non-vacuous: the gas branch booked on the GPU ticks; on the over-driven
  shield the clamp withheld energy (#78: at the 1263-game plateau it no longer
  does -- its P5c binding there was the sweep's per-ordinate floor artifact,
  gone with the fine currency); the vented room's floor remainder and the
  ring's export moved on GPU ticks.

Prints ``TGR_RESULT: PASS``/``FAIL`` and exits 0/1.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

# Import the CUDA build FIRST so it is the cached `breach_physics` (the
# harness below inserts the CPU build dir on sys.path, which must not win).
import breach_physics as bp

from _radiation_sweep_harness import ONE, R, live_table, reference_table  # noqa: E402

FP_ONE = 65536
T_AMB_Q = R.T_AMB_Q
DIALS = dict(no_face=63, o2_vacuum_thresh=0.3, c_v=R.C_V_LIVE,
             n_floor_heat=R.N_FLOOR_HEAT_LIVE, gas_advection_rate=900.0,
             t_max_phys=16000.0)
# The isolated GPU entry's tuple, in order (bindings.cpp cuda_temperature_step):
GPU_TUPLE = ("t_max_phys_hits", "e_cond_trunc_sum", "e_cond_cap_sum", "cond_limit_hits",
             "e_vac_wipe_sum", "e_ring_pin_sum", "e_deposit_drop_sum",
             "e_gas_deposit_sum", "e_gas_cond_sum", "e_gas_rail_sum",
             "e_solid_deposit_sum", "e_solid_cond_sum", "rad_clamp_hits",
             "solid_energy_books_sum", "e_rad_clamp_drop_sum",
             # P5c follow-up (slots 13, 14, appended)
             "e_rad_boundary_export_sum", "e_rad_floor_drop_sum")
# The boundary counters sum rn << 16 over the cells they read (outside the
# books; below n_floor): capped there at 2^34, a 16x margin over the physical
# bound E[3999] ~ 2^30 on the live table (test_temperature_gas_radiation.
# RN_PHYS_MAX), so the int64 sums cannot wrap; the 2^40 stress stays on every
# other cell.
RN_PHYS_MAX = 1 << 34
_FAILS: list[str] = []


def _fail(msg: str) -> None:
    _FAILS.append(msg)
    print("  FAIL: " + msg)


def _solver():
    s = bp.TemperatureSolver()
    s.no_face = DIALS["no_face"]
    s.o2_vacuum_thresh = DIALS["o2_vacuum_thresh"]
    s.c_v = DIALS["c_v"]
    s.n_floor_heat = DIALS["n_floor_heat"]
    s.gas_advection_rate = DIALS["gas_advection_rate"]
    s.T_MAX_PHYS = DIALS["t_max_phys"]
    return s


# ---------------------------------------------------------------------------
# PART 1 -- isolated
# ---------------------------------------------------------------------------
def _scene(rng, h, w, tref):
    """Every kind of cell the fold distinguishes, in one random grid."""
    kind = rng.choice(["gas", "gas", "gas", "solid", "vac", "ring", "flow"], size=(h, w))
    ts = kind == "solid"
    solid = ts | (kind == "flow")
    vac = kind == "vac"
    ring = kind == "ring"
    his = np.where(ts, rng.choice([-4, -2, 0, 3, 5], size=(h, w)), 0).astype(np.int32)
    nb = rng.choice([1, 200, R.N_FLOOR_Q_LIVE, 13107, ONE, 3 * ONE], size=(h, w))
    nb = np.where(ts | vac, 0, nb).astype(np.int32)
    t_game = rng.choice([-150, 0, 5, 300, 804, 1263, 3000], size=(h, w))
    T = ((t_game << 16) + rng.integers(0, 1 << 16, size=(h, w))).astype(np.int64)
    E = np.zeros((h, w), dtype=np.int64)
    gas_like = ~ts
    resid = rng.integers(0, np.maximum(nb, 1))
    E[gas_like] = nb[gas_like].astype(np.int64) * (T[gas_like] + T_AMB_Q) + resid[gas_like]
    for y, x in zip(*np.nonzero(gas_like)):
        T[y, x] = R.gas_mirror_q(int(E[y, x]), int(nb[y, x]))
    mag = rng.choice([1, 77, 1 << 12, 1 << 20, 1 << 30, 1 << 40], size=(h, w))
    rn = np.where(rng.random((h, w)) < 0.6, mag, -mag).astype(np.int64)
    rn[rng.random((h, w)) < 0.1] = 0
    counted = (vac | ring) | (~ts & ~solid & (nb < R.N_FLOOR_Q_LIVE))
    rn = np.where(counted, np.clip(rn, -RN_PHYS_MAX, RN_PHYS_MAX), rn)
    caps = rng.choice([0, 4, 290, 804, 1263, 5000, 15996], size=(h, w)) // 4
    phi = np.asarray([[tref[min(R.E_TABLE_SIZE - 1, int(c))] for c in row]
                      for row in caps], dtype=np.int64)
    phi += rng.choice([0, 1, 999], size=(h, w))
    heat = np.where(rng.random((h, w)) < 0.2,
                    rng.choice([1, 5000, 1 << 20], size=(h, w)), 0).astype(np.int32)
    fs = np.full((h, w, 4), DIALS["no_face"], dtype=np.int32)
    live_face = rng.random((h, w, 4)) < 0.5
    fs[live_face] = rng.choice([2, 5, 10, 16], size=int(live_face.sum()))
    return dict(T=T.astype(np.int32), E=E, rn=rn, phi=phi, nb=nb, his=his, ts=ts,
                solid=solid, vac=vac, ring=ring, heat=heat, fs=fs)


def _at_headroom(sc, tref):
    """P5d §5.6: (solid, gas) counts of cells whose Pass-1 radiative sub-step is
    clamped AT the headroom ceiling -- Phi >= E°[0] (a real ceiling, not row
    31's 0), the ceiling at or above T_before, and the unclamped T_after above
    it -- from the scene's INPUTS through the reference's arithmetic (the fold
    runs first in Pass 1, so the heat deposit and conduction after it cannot
    move the sub-step)."""
    ts, solid, vac, ring = sc["ts"], sc["solid"], sc["vac"], sc["ring"]
    acct = ~ts & ~solid & ~vac & ~ring
    fb = R.fine_bits_of(tref)          # #78: rn is in the table's currency
    n_s = n_g = 0
    for y, x in zip(*np.nonzero(sc["rn"])):
        if not (ts[y, x] or acct[y, x]):
            continue
        phi = int(sc["phi"][y, x])
        if phi < tref[0]:
            continue
        r = int(sc["rn"][y, x])
        if ts[y, x]:
            t0 = int(sc["T"][y, x])
            t_after = R.sat_add_q16(t0, R.fine_heat_shr(r, int(sc["his"][y, x]), fb))
        else:
            n = int(sc["nb"][y, x])
            t0 = R.gas_mirror_q(int(sc["E"][y, x]), max(0, n))
            t_after = R.sat_add_q16(t0, R.gas_rad_dT_q(r, n, fine_bits=fb))
        ceiling = R.e_ceiling_q(phi, tref)
        if ceiling >= t0 and t_after > ceiling:
            if ts[y, x]:
                n_s += 1
            else:
                n_g += 1
    return n_s, n_g


def _run_cpu(sc, table, clamp):
    s = _solver()
    T = np.ascontiguousarray(sc["T"].copy())
    E = np.ascontiguousarray(sc["E"].copy())
    h, w = T.shape
    s.step(T, sc["heat"], sc["his"], sc["fs"], np.ascontiguousarray(sc["solid"]),
           np.ascontiguousarray(sc["vac"]), np.full((h, w), ONE, np.int32),
           n_bulk=np.ascontiguousarray(sc["nb"]), thermal_solid=np.ascontiguousarray(sc["ts"]),
           rad_net=np.ascontiguousarray(sc["rn"]),
           rad_fluence=np.ascontiguousarray(sc["phi"]) if clamp else None,
           e_table=table if clamp else None, gas_energy=E, t_amb_q=T_AMB_Q,
           is_ambient=np.ascontiguousarray(sc["ring"]))
    cnt = {k: int(getattr(s, k)) for k in GPU_TUPLE}
    return T, E, cnt


def _run_gpu(sc, table, clamp):
    T = np.ascontiguousarray(sc["T"].copy())
    E = np.ascontiguousarray(sc["E"].copy())
    h, w = T.shape
    out = bp.cuda_temperature_step(
        T, sc["heat"], sc["his"], sc["fs"], np.ascontiguousarray(sc["solid"]),
        np.ascontiguousarray(sc["vac"]), np.full((h, w), ONE, np.int32),
        n_bulk=np.ascontiguousarray(sc["nb"]), thermal_solid=np.ascontiguousarray(sc["ts"]),
        gas_energy=E, t_amb_k=float(R.K_AMB), rad_net=np.ascontiguousarray(sc["rn"]),
        rad_fluence=np.ascontiguousarray(sc["phi"]) if clamp else None,
        e_table=table if clamp else None,
        is_ambient=np.ascontiguousarray(sc["ring"]), **DIALS)
    cnt = {k: int(v) for k, v in zip(GPU_TUPLE, out)}
    return T, E, cnt


def part1_isolated():
    print("PART 1 -- isolated: TemperatureSolver.step vs cuda_temperature_step, gas branch live")
    n = 0
    booked = bound = exits = 0
    at_ceiling = {}                    # P5d §5.6: per table, (solid, gas)
    fails_before = len(_FAILS)
    for tname, table, tref in (("live", live_table(), R.E_LIVE),
                               ("resolving", reference_table(), R.E)):
        for seed in range(4):
            for clamp in (True, False):
                for (h, w) in ((9, 13), (33, 65)):
                    rng = np.random.default_rng(20260925 + seed)
                    sc = _scene(rng, h, w, tref)
                    Tc, Ec, cc = _run_cpu(sc, table, clamp)
                    Tg, Eg, cg = _run_gpu(sc, table, clamp)
                    tag = f"P1 {tname} seed={seed} clamp={clamp} {h}x{w}"
                    if not np.array_equal(Tc, Tg):
                        _fail(f"{tag}: temperature differs at "
                              f"{int(np.count_nonzero(Tc != Tg))} cells")
                    if not np.array_equal(Ec, Eg):
                        _fail(f"{tag}: gas_energy differs at "
                              f"{int(np.count_nonzero(Ec != Eg))} cells")
                    for k in GPU_TUPLE:
                        if cc[k] != cg[k]:
                            _fail(f"{tag}: {k} cpu={cc[k]} gpu={cg[k]}")
                    # non-vacuity: the gas branch booked (heat deposits book
                    # too, so compare with the same scene's rad_net zeroed)
                    sc0 = dict(sc)
                    sc0["rn"] = np.zeros_like(sc["rn"])
                    _T0, _E0, c0 = _run_cpu(sc0, table, clamp)
                    if cc["e_gas_deposit_sum"] == c0["e_gas_deposit_sum"]:
                        _fail(f"{tag}: the gas branch booked nothing -- vacuous")
                    else:
                        booked += 1
                    if clamp:
                        if cc["e_rad_clamp_drop_sum"] <= 0:
                            _fail(f"{tag}: the clamp withheld nothing -- vacuous")
                        else:
                            bound += 1
                        a_s, a_g = _at_headroom(sc, tref)
                        prev = at_ceiling.get(tname, (0, 0))
                        at_ceiling[tname] = (prev[0] + a_s, prev[1] + a_g)
                    elif cc["rad_clamp_hits"] or cc["e_rad_clamp_drop_sum"]:
                        _fail(f"{tag}: the clamp counted with no fluence")
                    if not cc["e_rad_boundary_export_sum"] or not cc["e_rad_floor_drop_sum"]:
                        _fail(f"{tag}: a boundary counter never moved -- vacuous")
                    else:
                        exits += 1
                    n += 1
    for tname, (a_s, a_g) in at_ceiling.items():
        if a_s == 0 or a_g == 0:
            _fail(f"P1 {tname}: the clamp never bound AT the headroom ceiling on "
                  f"{'thermal solids' if a_s == 0 else 'gas cells'} -- vacuous (P5d §5.6)")
    verdict = "at tol 0" if len(_FAILS) == fails_before else "with FAILURES"
    print(f"  {n} configurations {verdict} on temperature, gas_energy and all "
          f"{len(GPU_TUPLE)} counters; the gas branch booked in {booked}, the clamp "
          f"withheld energy in {bound}, both boundary counters moved in {exits}")
    print("  P5d: cells clamped AT the headroom ceiling e_ceiling_q(Phi) in the "
          "clamp-on configurations, (thermal solid, gas): "
          + ", ".join(f"{t} {v}" for t, v in at_ceiling.items()))


# ---------------------------------------------------------------------------
# PART 2 -- the live conductor, the temperature backend flipping
# ---------------------------------------------------------------------------
_TEMP_COUNTERS = ("t_max_phys_hits", "t_low_rail_hits", "rad_clamp_hits",
                  "e_rad_clamp_drop_sum", "e_rad_boundary_export_sum",
                  "e_rad_floor_drop_sum", "e_cond_trunc_sum", "e_cond_cap_sum",
                  "cond_limit_hits", "e_vac_wipe_sum", "e_ring_pin_sum",
                  "e_deposit_drop_sum", "e_gas_deposit_sum", "e_gas_cond_sum",
                  "e_gas_rail_sum", "e_solid_deposit_sum", "e_solid_cond_sum",
                  "solid_energy_books_sum")


def _worlds():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import test_temperature_gas_radiation as T5c
    # The second element is the temperature (game) the shield's emitter column
    # is HELD at before every tick, or None. #78: at the 1263-game plateau
    # nothing over-drives the ambient-density smoke once the sweep's
    # per-ordinate floors are resolved (its P5c clamp binding was that
    # artifact), so the clamp's non-vacuity lives on the OVER-DRIVEN shield,
    # whose emitter is held at 8000 game (measured: ~1200 clamped cell-ticks
    # in 30 ticks on the CPU engine).
    return {"hot smoky room": lambda: (T5c._sealed_hot_smoky_room()[0], None),
            "smoke shield": lambda: (T5c._shield_room(0.2)[0], 1263),
            "over-driven shield": lambda: (T5c._shield_room(0.2)[0], 8000),
            # the P5c follow-up's two breached rooms (the closure test's scenes)
            "vented to space": lambda: (T5c._breached_room("space", fire=True,
                                                           T_game=300), None),
            "breached into the ring": lambda: (T5c._breached_room(
                "ambient", fire=True, n2_absorb=0.5), None)}


def _hold(sim, T_game):
    """The shield world's emitter is held at T_game before every tick."""
    g = sim.gmap
    col = np.zeros(g.material.shape, dtype=bool)
    col[1:8, 2] = True
    g.temperature[col] = int(T_game) << 16


def part2_live():
    print("PART 2 -- the live conductor: temperature backend CPU vs CUDA, gas radiation live")
    for name, build in _worlds().items():
        fails_before = len(_FAILS)
        s_cpu, held = build()
        s_gpu, _h = build()
        booked = 0
        drop = 0
        exits = dict(e_rad_boundary_export_sum=0, e_rad_floor_drop_sum=0)
        ticks = 72 if name in ("vented to space", "breached into the ring") else 30
        for t in range(ticks):
            for s, on in ((s_cpu, False), (s_gpu, True)):
                if held is not None:
                    _hold(s, held)
                bp.set_temperature_backend(on)
                ts = s.physics_runner.engine.temperature
                d0, c0 = int(ts.e_gas_deposit_sum), int(ts.e_rad_clamp_drop_sum)
                x0 = {k: int(getattr(ts, k)) for k in exits}
                s.set_paused(False)
                s.step()
                bp.set_temperature_backend(False)
                if on:
                    booked += int(int(ts.e_gas_deposit_sum) != d0)
                    drop += int(ts.e_rad_clamp_drop_sum) - c0
                    for k in exits:
                        exits[k] += int(int(getattr(ts, k)) != x0[k])
            a = {k: v for k, v in vars(s_cpu.gmap).items() if isinstance(v, np.ndarray)}
            b = {k: v for k, v in vars(s_gpu.gmap).items() if isinstance(v, np.ndarray)}
            bad = [k for k in a if not np.array_equal(a[k], b[k])]
            tc = s_cpu.physics_runner.engine.temperature
            tg = s_gpu.physics_runner.engine.temperature
            bad += [c for c in _TEMP_COUNTERS if int(getattr(tc, c)) != int(getattr(tg, c))]
            if bad:
                _fail(f"P2 {name} tick {t}: differs in {bad[:6]}")
                break
        if booked == 0:
            _fail(f"P2 {name}: the GPU fold's gas branch never booked -- vacuous")
        if name == "over-driven shield" and drop <= 0:
            _fail(f"P2 {name}: the GPU clamp never withheld energy -- vacuous")
        if name == "vented to space" and not exits["e_rad_floor_drop_sum"]:
            _fail(f"P2 {name}: the GPU floor remainder never moved -- vacuous")
        if name == "breached into the ring" and not exits["e_rad_boundary_export_sum"]:
            _fail(f"P2 {name}: the GPU ring export never moved -- vacuous")
        verdict = "at tol 0" if len(_FAILS) == fails_before else "FAILED"
        print(f"  {name}: {ticks} ticks {verdict} on every GameMap array and "
              f"{len(_TEMP_COUNTERS)} temperature counters; GPU ticks booking the gas "
              f"branch {booked}, GPU clamp drop {drop}, GPU ticks moving the export "
              f"{exits['e_rad_boundary_export_sum']} / the floor "
              f"{exits['e_rad_floor_drop_sum']}")


def main() -> int:
    if not getattr(bp, "HAS_CUDA", False) or not bp.cuda_available():
        print("TGR_RESULT: FAIL (no CUDA build / device)")
        return 1
    print("device:", bp.cuda_device_info())
    part1_isolated()
    part2_live()
    bp.set_temperature_backend(False)
    if _FAILS:
        print(f"TGR_RESULT: FAIL ({len(_FAILS)} failures)")
        return 1
    print("TGR_RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
