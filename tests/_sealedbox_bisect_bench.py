"""Issue #54 bisection bench — the sealed-box probe, one solver term per run.

The all-systems scenario (2026-08-27) found the cleanest #54 repro yet: a
glass box born sealed at t=0 via ``seal_tiles`` (no doors, no history, no
interior heat source) heats +124 game-deg and self-pressurizes 1.0->1.52 atm
in 18 s from a crate fire ~20 tiles away, while the arena around it COOLS.
P/T ratio ~ constant-N heating: energy, not mass, crosses the sealed wall.

This bench reduces that to the minimal deterministic probe (fire + sealed
dry box, no water/blasts/breach) and re-runs it with ONE energy-chain term
disabled per pass — the bisection the #54 session plan prescribes:

    baseline        as configured
    drag_heat       k_drag_heat_frac = 0
    drag            k_drag = 0 (whole staged momentum drag off)
    comp_work       adiabatic_index = 1.0 (compression work off)
    flat_gs         use_multigrid = False (flat RB-GS — MG wall suspect)
    no_vrail        U_MAX = 1e9 (v2.4 store-clamp rail effectively off)

All fields are live ``def_readwrite`` members of the C++ EOSSolver
(bindings.cpp), set on ``sim.physics_runner.eos`` post-construction, fresh
Simulation per variant. FIXED behavior = box dT ~ 0 while only the crate's
neighbourhood warms. The toggle that kills the box heating names the
mechanism (or flat_gs indicts the MG wall handling specifically).

HARNESS, not a pytest gate (``_`` prefix): prints the table, exits 0.

Run:
    conda run -n data python tests/_sealedbox_bisect_bench.py
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
END_TICK = 18 * TPS
IGNITE_TICK = 2 * TPS
CRATE = (26, 41)                 # mid-arena crate stack — the only heat source
AQ_BOX = (50, 58, 24, 32)        # sealed box, built on open arena floor
AQ_IN = np.s_[51:58, 25:32]
BUNKER = np.s_[27:42, 83:96]     # #54 bench R6 (steel, doored)
PEN = np.s_[49:66, 83:96]        # #54 bench R8 (glass, sealed)
ARENA = np.s_[3:67, 3:58]

VARIANTS = [
    ("baseline",  {}),
    ("drag_heat", {"k_drag_heat_frac": 0.0}),
    ("drag",      {"k_drag": 0.0}),
    ("comp_work", {"adiabatic_index": 1.0}),
    # 2026-08-29: adiabatic_index is CONFOUNDED — it also sets the kick's
    # pressure stiffness K = c_max^2/gamma (cuda_kick_compression.cu
    # kick_scalar_folds), so comp_work stiffens the EOS by 1.4x as well.
    # T_WORK_CLAMP = 0 zeroes ONLY the step-4c work term (w_mag clamps to
    # 0 -> t_new == T exactly on both rails); stiff_K isolates the K change.
    ("comp_clamp0", {"T_WORK_CLAMP": 0.0}),
    ("stiff_K",     {"adiabatic_index": 1.0, "T_WORK_CLAMP": 0.0}),
    ("flat_gs",   {"use_multigrid": False}),
    ("no_vrail",  {"U_MAX": 1e9}),
    # MG thin-wall probe (2026-08-29): same box, 2- and 3-tile glass walls.
    ("wall2",       {}, 2),
    ("wall3",       {}, 3),
    ("wall2_clamp0", {"T_WORK_CLAMP": 0.0}, 2),
]


def run_variant(name, overrides, wall_thick=1):
    """One fresh Simulation; ``wall_thick`` = glass ring thickness in tiles
    (2026-08-29: the MG thin-wall probe — if a coarse cell straddling a
    1-tile wall is the leak, thicker walls should shrink it)."""
    lvl = load_level("playground", levels_dir=str(ROOT / "levels"))
    lvl = replace(lvl, entities=[e for e in lvl.entities
                                 if e.class_name not in ("vent", "duct")])
    sim = Simulation(lvl, seed=1, breach_physics=bp, enable_recorder=False)
    g = sim.gmap
    for field, value in overrides.items():
        setattr(sim.physics_runner.eos, field, value)

    r0, r1, c0, c1 = AQ_BOX
    box_in = np.s_[r0 + wall_thick:r1 + 1 - wall_thick,
                   c0 + wall_thick:c1 + 1 - wall_thick]
    # Seal one layer per call, INNERMOST first: seal_tiles evacuates each
    # tile's gas to an OPEN non-span neighbour and refuses a tile with none
    # (its sealed-pocket guard) — an inner layer's corners only have open
    # neighbours while the layer outside them is still open.
    for k in reversed(range(wall_thick)):
        layer = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                 if min(r - r0, r1 - r, c - c0, c1 - c) == k]
        g.seal_tiles(layer, materials.MAT_GLASS)
    open0 = ~g.solid.copy()
    T0 = g.temperature.astype(np.int64)
    P0_aq = float(g.atmosphere[box_in][open0[box_in]].mean()) / 65536.0
    # MASS vs PRESSURE-FIELD (2026-08-29, Erik's question): N inside the box
    # from the two conservative bulk planes — if P rises while N holds, the
    # pressure SOLVE is contaminated (no mass moved); if N rises, mass
    # actually crossed the sealed faces.
    o2 = int(g.gases.name_to_id["o2"])
    n2 = int(g.gases.name_to_id["inert_n2"])

    def n_box():
        return int(g.gas[o2][box_in].sum(dtype=np.int64) +
                   g.gas[n2][box_in].sum(dtype=np.int64))
    N0 = n_box()

    def dT(sl):
        d = (g.temperature.astype(np.int64) - T0)[sl]
        return float(d[open0[sl]].mean()) / 65536.0

    for t in range(1, END_TICK + 1):
        if t == IGNITE_TICK:
            ignite_ring(g, sim.edit_queue, *CRATE, 2.5, 1.0)
        sim.set_paused(False)
        sim.step()

    P_aq = float(g.atmosphere[box_in][open0[box_in]].mean()) / 65536.0
    u = np.sqrt((g.wind_x / 65536.0) ** 2 + (g.wind_y / 65536.0) ** 2)
    print(f"{name:>11}: box dT={dT(box_in):+7.1f}  box P {P0_aq:.3f}->{P_aq:.3f}"
          f"  box N x{n_box()/N0:5.3f}"
          f"  bunker dT={dT(BUNKER):+7.1f}  pen dT={dT(PEN):+7.1f}"
          f"  arena dT={dT(ARENA):+6.1f}  u_max={float(u.max()):5.1f}"
          f"  wall={wall_thick}"
          + (f"  [{overrides}]" if overrides else ""))


def main() -> None:
    print(f"sealed-box bisection — crate fire only, {END_TICK/TPS:.0f} s, "
          f"FIXED = box dT ~ 0")
    wanted = set(sys.argv[1:])          # optional: run only the named variants
    for spec in VARIANTS:
        name, overrides = spec[0], spec[1]
        if wanted and name not in wanted:
            continue
        run_variant(name, overrides, *spec[2:])


if __name__ == "__main__":
    main()
