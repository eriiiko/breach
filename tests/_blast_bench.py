"""BLAST gate — one frag grenade mid-arena (issue #54, P-G1a).

Design: ``docs/gas_energy_conservation_design_2026-08-29.md`` §6 "BLAST",
prediction §3: "Blast cores: COOLER than HEAD, not hotter — HEAD's 4c pumped
energy into cavities; the flux form bounds a cell's heating by its
neighbours' energy. (Corrects v1 §8.) Fire tuning (#5/#8) will see lower
core T."

One ``frag_standard`` payload is executed mid-arena on the playground and the
sim runs 3 s. Two asks:

  (1) NO ``T_MAX_PHYS`` hit OUTSIDE the blast disc. The rail is allowed to
      engage inside the disc — that is a genuinely near-vacuum, genuinely
      ill-defined regime and the rail is its stand-in (eos_solver.h) — but a
      hit at range means the flux step is throwing energy somewhere it has no
      business being. Measured as: the hottest cell outside the disc, and the
      per-tick t_max_phys_hits delta correlated with the disc mask.
  (2) CORE T BELOW HEAD's. Run this bench on the base commit FIRST and record
      the number; §3 predicts the flux form comes in under it.

Both are REPORTED, not asserted: this is a harness (``_`` prefix), and the
HEAD comparison has to come from a second run on a different build anyway.

Run:
    conda run -n data python tests/_blast_bench.py
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
from simulation.payloads import execute_payload  # noqa: E402

TPS = 24
RUN_TICKS = 3 * TPS
BLAST_TICK = 4              # a few ticks in, so the map is settled
CENTER = (35, 40)           # mid-arena, open floor, away from walls
PAYLOAD = "frag_standard"
DISC_R = 8                  # Chebyshev radius counted as "the blast disc"
Q = 65536.0


def _n_plane(g):
    n = np.zeros(g.temperature.shape, dtype=np.int64)
    for gi in np.flatnonzero(g.gases.conservative):
        n += g.gas[gi].astype(np.int64)
    return n


def main() -> None:
    lvl = load_level("playground", levels_dir=str(ROOT / "levels"))
    lvl = replace(lvl, entities=[e for e in lvl.entities
                                 if e.class_name not in ("vent", "duct")])
    sim = Simulation(lvl, seed=1, breach_physics=bp, enable_recorder=False)
    g = sim.gmap
    eos = sim.physics_runner.eos

    cy, cx = CENTER
    assert not g.solid[cy, cx], "blast centre must be open floor"
    ys, xs = np.ogrid[:g.temperature.shape[0], :g.temperature.shape[1]]
    cheb = np.maximum(np.abs(ys - cy), np.abs(xs - cx))
    disc = cheb <= DISC_R
    open_air = (~g.solid) & (~g.is_vacuum)

    peak_core = -1e9
    peak_out = -1e9
    peak_out_at = None
    hits_prev = 0
    hits_with_hot_outside = 0
    t_hist = []

    for t in range(1, RUN_TICKS + 1):
        if t == BLAST_TICK:
            execute_payload(g, sim.edit_queue, sim.units, cy, cx,
                            sim.weapons_tables.payloads.by_name[PAYLOAD],
                            sim.rng)
        sim.set_paused(False)
        sim.step()
        T = g.temperature.astype(np.int64) / Q
        core = float(T[disc & open_air].max()) if (disc & open_air).any() else 0.0
        outm = (~disc) & open_air
        out = float(T[outm].max()) if outm.any() else 0.0
        peak_core = max(peak_core, core)
        if out > peak_out:
            peak_out = out
            idx = np.argmax(np.where(outm, T, -1e18))
            peak_out_at = np.unravel_index(idx, T.shape)
        hits = int(eos.t_max_phys_hits)
        if hits > hits_prev:
            # A rail hit fired this tick: was there ANY cell outside the disc
            # sitting at the ceiling? (The rail is legitimate inside it.)
            ceiling = float(eos.T_MAX_PHYS)
            if (T[outm] >= ceiling - 1e-6).any():
                hits_with_hot_outside += 1
            hits_prev = hits
        t_hist.append((t, core, out, hits))

    print(f"BLAST gate — {PAYLOAD} at {CENTER}, {RUN_TICKS / TPS:.0f} s, "
          f"disc = Chebyshev r<={DISC_R}")
    print(f"  (1) T_MAX_PHYS: total hits={int(eos.t_max_phys_hits)}  "
          f"ticks with a railed cell OUTSIDE the disc={hits_with_hot_outside}  "
          f"{'PASS' if hits_with_hot_outside == 0 else 'FAIL'}")
    print(f"      hottest cell OUTSIDE the disc: {peak_out:+.1f} game-deg "
          f"at {peak_out_at}")
    print(f"  (2) PEAK CORE T (inside the disc): {peak_core:+.1f} game-deg   "
          f"<-- compare against the same line on the base commit")
    print(f"      peak |u| = "
          f"{float(np.hypot(g.wind_x[open_air] / Q, g.wind_y[open_air] / Q).max()):.1f} m/s")
    print(f"      counters (last tick): kick={int(eos.e_kick_ke_sum)} "
          f"drag={int(eos.e_drag_heat_sum)} clamp={int(eos.e_clamp_destroyed_sum)} "
          f"absorb={int(eos.e_absorb_export_sum)} rail={int(eos.e_rail_sum)}")
    print(f"      hits: rad_clip={int(eos.rad_clip_hits)} "
          f"p_floor={int(eos.p_face_floor_hits)} "
          f"p_ceil={int(eos.p_face_ceil_hits)} "
          f"flux_sat={int(eos.flux_sat_hits)} "
          f"u_clamp={int(eos.u_clamp_hits)}")
    print("  per-tick (tick, core T, outside T, cumulative t_max hits):")
    for row in t_hist[:12]:
        print(f"      {row[0]:3d}  {row[1]:9.1f}  {row[2]:9.1f}  {row[3]}")


if __name__ == "__main__":
    main()
