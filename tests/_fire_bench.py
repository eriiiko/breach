"""FIRE gate — a burning crate stack (issue #54, P-G1a).

Design: ``docs/gas_energy_conservation_design_2026-08-29.md`` §6 "FIRE".

``ignite_ring`` lights the playground's crate stack at (26, 41) and the sim
runs 10 s. Two asks, and one deliberately DEFERRED:

  (1) THE CLOSURE IDENTITY IS EXACT — measured WITHIN the EOS step. At P-G1a
      combustion is still on the T side (§2.7 row 2 is P-G1b's), so it writes
      `temperature` between two EOS steps and the solver's entry re-sync
      absorbs it; the identity books that as `e_entry_resync_sum` and stays
      exact. Bracketing the whole TICK instead would just be measuring
      combustion, which this patch does not own.
  (2) ROOMS ELSEWHERE STAY ~0. This is #54's actual signature: on HEAD the
      crate fire drove a sealed box 20 tiles away to +115 and cooled the hall
      by -16, with no mass crossing anything. Under the flux form a room the
      fire cannot reach must not move.

  DEFERRED: the flame-cell E/N bound (R3-#9's "no per-tick compounding"). That
  bound is about the combustion products' two-hop energy ledger and the soot
  shed row, which land at P-G1b — there is nothing to measure here yet. The
  flame cell's T is REPORTED so the P-G1b run has a before-number.

HARNESS, not a pytest gate (``_`` prefix): prints the table, exits 0.

Run:
    conda run -n data python tests/_fire_bench.py
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402
from level_loader import load as load_level  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation import materials  # noqa: E402
from simulation.payloads import ignite_ring  # noqa: E402

TPS = 24
RUN_TICKS = 10 * TPS
IGNITE_TICK = 2 * TPS
CRATE = (26, 41)                 # the scenario's crate stack
AQ_BOX = (50, 58, 24, 32)        # a glass box sealed at t=0, 20+ tiles away
AQ_IN = np.s_[51:58, 25:32]
BUNKER = np.s_[27:42, 83:96]     # #54 bench R6 (steel, doored)
PEN = np.s_[49:66, 83:96]        # #54 bench R8 (glass, sealed)
Q = 65536.0


def _n_plane(g):
    n = np.zeros(g.temperature.shape, dtype=np.int64)
    for gi in np.flatnonzero(g.gases.conservative):
        n += g.gas[gi].astype(np.int64)
    return n


def _sum_obj(a):
    return int(a.astype(object).sum())


def main() -> None:
    lvl = load_level("playground", levels_dir=str(ROOT / "levels"))
    lvl = replace(lvl, entities=[e for e in lvl.entities
                                 if e.class_name not in ("vent", "duct")])
    sim = Simulation(lvl, seed=1, breach_physics=bp, enable_recorder=False)
    g = sim.gmap
    eos = sim.physics_runner.eos

    # A sealed glass box on open arena floor — the #54 repro's own probe.
    r0, r1, c0, c1 = AQ_BOX
    ring = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
            if min(r - r0, r1 - r, c - c0, c1 - c) == 0]
    # P-G1b: no re-derive here -- `seal_tiles` is an energy writer now
    # (design 2.7: the evacuated mass is MOVED at the sealed tile's own T_abs
    # and the sub-count remainder retires), so the field is already correct.
    g.seal_tiles(ring, materials.MAT_GLASS)

    open0 = ~g.solid.copy()
    T0 = g.temperature.astype(np.int64).copy()
    t_amb_raw = g._gas_energy_t_amb_raw()

    def region_T(sl):
        """N-weighted mean T over a region's accountable cells, game-deg."""
        m = np.zeros_like(open0)
        m[sl] = True
        m &= g._gas_energy_accountable()
        if not m.any():
            return float("nan")
        n = _n_plane(g)
        N = _sum_obj(n[m])
        if N == 0:
            return float("nan")
        return (_sum_obj(g.gas_energy[m]) / N - t_amb_raw) / Q

    box0, bunk0, pen0 = region_T(AQ_IN), region_T(BUNKER), region_T(PEN)

    tally = dict(ticks=0, bad=0, worst=0, soot_note=0)
    tsolver = sim.physics_runner.engine.temperature
    comb = sim.physics_runner.combustion

    # arc #54 P-G1b: the closure identity is checked ACROSS WHOLE TICKS now.
    # P-G1a could only bracket `run_substeps` (the writers outside the EOS
    # still wrote `temperature` and were swept up by the entry re-sync); with
    # D1 live the honest bracket is the whole tick, and it has to account for
    # all four groups of counters that may move the field.
    def _terms():
        return (
            int(eos.e_entry_resync_sum) + int(eos.e_transport_net_sum)
            - int(eos.e_wipe_sum) - int(eos.e_kick_ke_sum)
            + int(eos.e_drag_heat_sum) - int(eos.e_work_export_sum)
            + int(eos.e_rail_sum),
            int(tsolver.e_gas_deposit_sum) + int(tsolver.e_gas_cond_sum)
            + int(tsolver.e_gas_rail_sum),
            -int(comb.e_comb_draw_sum) + int(comb.e_comb_deliver_sum)
            + int(comb.e_comb_heat_sum) + int(comb.e_comb_rail_sum),
            int(g.gas_energy_seam_net()),
            # the water-displacement evacuation's export (R3-#10) -- host-side,
            # before the EOS, reset per call like the EOS group.
            -int(sim.physics_runner.engine.e_water_evac_export_sum),
            # P-G5 (design gas_energy_thermostat_ledger_2026-08-30.md): the
            # SOLID side's own channels — Pass 1/Pass 2 landings on thermal
            # solids, the thermostat (Pass 3 relax-to-ambient), and
            # combustion's own object-site solid heat deposit (bypasses
            # TemperatureSolver's Pass 1 entirely — the crate fuel itself).
            # Accumulating.
            int(tsolver.e_solid_deposit_sum) + int(tsolver.e_solid_cond_sum)
            + int(tsolver.e_thermostat_sum) + int(comb.e_comb_solid_heat_sum),
        )

    def _e_acct():
        return _sum_obj(g.gas_energy[g._gas_energy_accountable()])

    def _solid_books():
        """(P-G5) Σ thermal_mass_raw·T_raw over thermal_solid cells."""
        return int(tsolver.solid_energy_books_sum)

    flame_peak = 0.0
    en_peak = 0.0
    tally_total = dict(ticks=0, bad=0, worst=0)
    prev_e, prev_terms = _e_acct(), _terms()
    prev_solid = _solid_books()
    for t in range(1, RUN_TICKS + 1):
        if t == IGNITE_TICK:
            ignite_ring(g, sim.edit_queue, *CRATE, 2.5, 1.0)
        sim.set_paused(False)
        sim.step()
        e_now, terms = _e_acct(), _terms()
        solid_now = _solid_books()
        expected = (terms[0] + (terms[1] - prev_terms[1])
                    + (terms[2] - prev_terms[2]) + (terms[3] - prev_terms[3])
                    + terms[4])
        resid = (e_now - prev_e) - expected
        tally["ticks"] += 1
        if resid:
            tally["bad"] += 1
            tally["worst"] = max(tally["worst"], abs(resid))
        # P-G5: the TOTAL ledger — gas books + solid books.
        expected_total = expected + (terms[5] - prev_terms[5])
        resid_total = ((e_now + solid_now) - (prev_e + prev_solid)) - expected_total
        tally_total["ticks"] += 1
        if resid_total:
            tally_total["bad"] += 1
            tally_total["worst"] = max(tally_total["worst"], abs(resid_total))
        prev_e, prev_terms, prev_solid = e_now, terms, solid_now
        burning = g.fire > 0
        if burning.any():
            tally["soot_note"] += 1     # ticks with live fire (non-vacuity)
            flame_peak = max(flame_peak,
                             float(g.temperature[burning].max()) / Q)
            n = _n_plane(g)
            live = burning & g._gas_energy_accountable() & (n >= 1)
            if live.any():
                en = (g.gas_energy[live].astype(np.float64)
                      / n[live].astype(np.float64) - t_amb_raw) / Q
                en_peak = max(en_peak, float(en.max()))

    box1, bunk1, pen1 = region_T(AQ_IN), region_T(BUNKER), region_T(PEN)
    arena = np.s_[3:67, 3:58]
    dT_arena = float(((g.temperature.astype(np.int64) - T0)[arena]
                      )[open0[arena]].mean()) / Q

    print(f"FIRE gate — ignite_ring on the crate stack at {CRATE}, "
          f"{RUN_TICKS / TPS:.0f} s")
    print(f"  (1) CLOSURE IDENTITY across {tally['ticks']} TICKS: "
          f"{'EXACT' if tally['bad'] == 0 else 'BROKEN'}"
          + ("" if tally["bad"] == 0 else
             f" ({tally['bad']} bad, worst |resid| {tally['worst']})"))
    print(f"  (2) rooms elsewhere (N-weighted mean T, game-deg):")
    print(f"        sealed box   {box0:+7.2f} -> {box1:+7.2f}  "
          f"(d {box1 - box0:+7.2f})")
    print(f"        bunker R6    {bunk0:+7.2f} -> {bunk1:+7.2f}  "
          f"(d {bunk1 - bunk0:+7.2f})")
    print(f"        pen R8       {pen0:+7.2f} -> {pen1:+7.2f}  "
          f"(d {pen1 - pen0:+7.2f})")
    print(f"        arena mirror mean dT {dT_arena:+7.2f}")
    parcel = (int(comb.e_comb_draw_sum) + int(comb.e_comb_mint_sum)
              - int(comb.e_comb_deliver_sum) - int(comb.e_soot_shed_sum)
              - int(comb.e_ts_products_sum) - int(comb.e_comb_export_sum))
    print(f"  (3) COMBUSTION PARCEL identity (combustion.h (B)): "
          f"{'EXACT' if parcel == 0 else f'BROKEN by {parcel}'}")
    print(f"        drawn={int(comb.e_comb_draw_sum)} "
          f"mint={int(comb.e_comb_mint_sum)} "
          f"deliver={int(comb.e_comb_deliver_sum)} "
          f"soot_shed={int(comb.e_soot_shed_sum)} "
          f"ts_products={int(comb.e_ts_products_sum)} "
          f"export={int(comb.e_comb_export_sum)}")
    print(f"        heat={int(comb.e_comb_heat_sum)} "
          f"rail={int(comb.e_comb_rail_sum)}")
    print(f"  (4) FLAME CELL bound (R3-#9, no per-tick compounding): "
          f"peak T = {flame_peak:.1f}, peak gas E/N - T_AMB = {en_peak:.1f} "
          f"game-deg over {tally['soot_note']} ticks with live fire")
    print(f"      hits: rad_clip={int(eos.rad_clip_hits)} "
          f"p_floor={int(eos.p_face_floor_hits)} "
          f"p_ceil={int(eos.p_face_ceil_hits)} "
          f"flux_sat={int(eos.flux_sat_hits)} "
          f"t_max={int(eos.t_max_phys_hits)}")
    # P-G5 (design gas_energy_thermostat_ledger_2026-08-30.md): the EXTENDED
    # identity — gas books + solid books — over the same run.
    print(f"  (5) P-G5 TOTAL ledger (gas+solid) across "
          f"{tally_total['ticks']} TICKS: "
          f"{'EXACT' if tally_total['bad'] == 0 else 'BROKEN'}"
          + ("" if tally_total["bad"] == 0 else
             f" ({tally_total['bad']} bad, worst |resid| "
             f"{tally_total['worst']})"))
    print(f"      e_solid_deposit_sum={int(tsolver.e_solid_deposit_sum)} "
          f"e_solid_cond_sum={int(tsolver.e_solid_cond_sum)} "
          f"e_thermostat_sum={int(tsolver.e_thermostat_sum)} "
          f"e_comb_solid_heat_sum={int(comb.e_comb_solid_heat_sum)}")

    assert tally["bad"] == 0, (
        f"FIRE gate gas-books identity BROKEN: {tally['bad']} bad tick(s), "
        f"worst |resid| {tally['worst']}")
    assert tally_total["bad"] == 0, (
        f"P-G5 TOTAL ledger BROKEN: {tally_total['bad']} bad tick(s), "
        f"worst |resid| {tally_total['worst']}")


if __name__ == "__main__":
    main()
