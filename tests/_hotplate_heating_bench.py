"""Issue #54 repro harness — sealed-room heating under sustained forcing.

A 5x5 patch mid-arena held at +600 game-deg, headless, NO vents (stripped
in-memory), NO fires, NO player: 30 s later the playground's closed rooms
heat by ~+75..+105 game-deg while the arena (which CONTAINS the heat
source) cools below ambient and drains pressure, with 10-33 m/s winds.
Found 2026-08-25 via #48's feel patch; forcing-generic (fires and vents
force the same signature); the T_abs arc's "quiet-room drift, monitored"
escalated. Full forensics: issue #54.

This is a HARNESS, not a pytest gate (the ``_`` prefix — tests/
convention): it prints the region table and exits 0. FIXED behavior =
room means stay ~0 while only the plate's neighbourhood warms.

Bisection use: toggle one energy-chain term per run (k_drag_heat_frac=0,
compression-work off, MG vs point-GS solve, velocity clamp off) and watch
which toggle kills the room heating. Diagnosis step 1 per #54: the
mass-books audit — rooms heat while LOSING mass; find where mass goes
(tools/analyze_blowup_dump.py::mass_books is the instrument pattern).

Run:
    conda run -n data python tests/_hotplate_heating_bench.py
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

# Playground region boxes (row, col slices) — see the #54 evidence table.
R5 = np.s_[27:42, 61:77]      # mid-inner room (doored)
R6 = np.s_[27:42, 83:96]      # mid-far room (doored, NO vent ever)
R8 = np.s_[49:66, 83:96]      # bottom-far room (sealed)
ARENA = np.s_[3:67, 3:58]     # the big open area (holds the plate)
PLATE = np.s_[33:38, 28:33]   # ~30 tiles from the nearest room
T_HOLD_Q = 600 * 65536        # +600 game-deg, Q16.16
TICKS = 720                   # 30 s at 24 tps


def main() -> None:
    lvl = load_level("playground", levels_dir=str(ROOT / "levels"))
    # Strip vents/ducts so the harness isolates #54 from #48 regardless of
    # whether the level currently carries the vent layout.
    lvl = replace(lvl, entities=[e for e in lvl.entities
                                 if e.class_name not in ("vent", "duct")])
    sim = Simulation(lvl, seed=1, breach_physics=bp, enable_recorder=False)
    g = sim.gmap

    print(f"{'t(s)':>5} {'T_r5':>7} {'T_r6':>7} {'T_r8':>7} {'T_arena':>8} "
          f"{'P_r8':>6} {'P_arena':>8} {'u_max':>7}")

    def row(t: int) -> None:
        T = g.temperature / 65536.0
        P = g.atmosphere / 65536.0
        u = np.sqrt((g.wind_x / 65536.0) ** 2 + (g.wind_y / 65536.0) ** 2)
        print(f"{t/24:5.0f} {T[R5].mean():7.1f} {T[R6].mean():7.1f} "
              f"{T[R8].mean():7.1f} {T[ARENA].mean():8.1f} "
              f"{P[R8].mean():6.3f} {P[ARENA].mean():8.3f} {u.max():7.2f}")

    row(0)
    for t in range(1, TICKS + 1):
        # gas-energy conservation arc #54, design §2.7 last row (P-G0): PLATE
        # is open air (mid-arena), so its temperature seed goes through the
        # seam primitive that keeps gas_energy in sync — not a raw
        # `temperature[...] =` write.
        g.seed_gas_temperature(PLATE, T_HOLD_Q)   # the held hot plate
        sim.set_paused(False)
        sim.step()
        if t % 144 == 0 or t == TICKS:
            row(t)


if __name__ == "__main__":
    main()
