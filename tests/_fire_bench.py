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
    g.seal_tiles(ring, materials.MAT_GLASS)
    g.refresh_gas_energy()

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

    state = {"pre": 0, "acct": None}
    tally = dict(ticks=0, bad=0, worst=0, soot_note=0)

    class _EngineProbe:
        def __init__(self, inner):
            object.__setattr__(self, "_inner", inner)

        def __getattr__(self, k):
            return getattr(object.__getattribute__(self, "_inner"), k)

        def run_substeps(self, *a, **kw):
            inner = object.__getattribute__(self, "_inner")
            acct = g._gas_energy_accountable()
            pre = _sum_obj(g.gas_energy[acct])
            inner.run_substeps(*a, **kw)
            post = _sum_obj(g.gas_energy[acct])
            expected = (int(eos.e_entry_resync_sum)
                        + int(eos.e_transport_net_sum)
                        - int(eos.e_wipe_sum) - int(eos.e_kick_ke_sum)
                        + int(eos.e_drag_heat_sum)
                        - int(eos.e_work_export_sum) + int(eos.e_rail_sum))
            resid = (post - pre) - expected
            tally["ticks"] += 1
            if resid:
                tally["bad"] += 1
                tally["worst"] = max(tally["worst"], abs(resid))

    sim.physics_runner.engine = _EngineProbe(sim.physics_runner.engine)

    flame_peak = 0.0
    en_peak = 0.0
    for t in range(1, RUN_TICKS + 1):
        if t == IGNITE_TICK:
            ignite_ring(g, sim.edit_queue, *CRATE, 2.5, 1.0)
        sim.set_paused(False)
        sim.step()
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
    print(f"  (1) CLOSURE IDENTITY over {tally['ticks']} EOS steps: "
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
    print(f"  DEFERRED (P-G1b): flame-cell peak T = {flame_peak:.1f}, "
          f"peak gas E/N - T_AMB = {en_peak:.1f} game-deg over "
          f"{tally['soot_note']} ticks with live fire "
          f"(the R3-#9 compounding bound lands with the combustion ledger)")
    print(f"      hits: rad_clip={int(eos.rad_clip_hits)} "
          f"p_floor={int(eos.p_face_floor_hits)} "
          f"p_ceil={int(eos.p_face_ceil_hits)} "
          f"flux_sat={int(eos.flux_sat_hits)} "
          f"t_max={int(eos.t_max_phys_hits)}")


if __name__ == "__main__":
    main()
