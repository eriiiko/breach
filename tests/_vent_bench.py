"""VENT gate — a 1-atm sealed room breached to vacuum (issue #54, P-G1a).

Design: ``docs/gas_energy_conservation_design_2026-08-29.md`` §6 "VENT".

The scenario, built on the playground with the canonical seams (never a hand
edit): the level's own HULL already encloses a 1-atm volume against hard
vacuum, so after letting the map settle, ``destroy_wall`` opens three hull
tiles and the gas vents to space for 5 s.

(The first cut of this bench sealed a fresh glass room in the middle of the
arena and breached THAT — which vents into ordinary air, not vacuum, so
``e_work_export_sum`` was identically 0 and ask (4) was vacuous. The gate
needs a face whose far side is genuinely ``is_vacuum``: that is the only
place §2.4's OUTFLOW class exists.)

The four things this measures — the four §6 asks — and why each one is the
right question for the FLUX form specifically:

  (1) ``gas_energy >= 0`` everywhere, EVERY tick. The face-flux step moves
      energy out through the mouth at a material Courant number that is NOT
      bounded by 1 (§2.4: |div|·dt up to ~8 during venting), so positivity is
      not free — it is bought by the donor-only two-pass rail (F3/F13). A
      single negative cell means that rail is not holding.
  (2) ``n_sub`` STABLE. Venting pins the substep count at N_SUB_MAX for the
      whole post-breach phase (eos_solver.h's own note), and the energy pass
      rides the SAME n_sub — so a runaway would show as n_sub sticking high
      long after the pressure has equalized.
  (3) THE MOUTH COOLS. §3: "the mouth's net cooling comes from the outflow
      face export + the KE debit, not from a per-cell expansion factor —
      expect a RING around the mouth rather than a core." Reported as the
      profile, not just a number, because the SHAPE is the claim.
  (4) ``e_work_export_sum`` MATCHES the room's energy loss, to the LSB of the
      counters. This is the one that proves the outflow faces are booked: the
      room's ΔΣE must be the sum of the counted channels and nothing else.

HARNESS, not a pytest gate (``_`` prefix): prints the table, exits 0.

Run:
    conda run -n data python tests/_vent_bench.py
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

TPS = 24
SETTLE_TICKS = 1 * TPS      # let the level's own start transient decay
RUN_TICKS = 5 * TPS
# Three adjacent HULL tiles on the playground's north wall: solid, with hard
# vacuum on one side and 1-atm interior air on the other (verified by the
# fixture assert below rather than assumed).
MOUTH_TILES = [(2, 5), (2, 6), (2, 7)]
MOUTH = MOUTH_TILES[1]
Q = 65536.0


def _n_plane(g):
    n = np.zeros(g.temperature.shape, dtype=np.int64)
    for gi in np.flatnonzero(g.gases.conservative):
        n += g.gas[gi].astype(np.int64)
    return n


def _sum_obj(a):
    """Sum as a PYTHON int — §2.2 forbids absolute int64 sums, and a venting
    mouth is exactly where a bench must not be the thing that wraps."""
    return int(a.astype(object).sum())


class _EngineProbe:
    """pybind methods are read-only, so the per-EOS-step bracket rides a
    transparent proxy over the engine (the SB bench's idiom)."""

    def __init__(self, inner, on_call):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_on", on_call)

    def __getattr__(self, k):
        return getattr(object.__getattribute__(self, "_inner"), k)

    def run_substeps(self, *a, **kw):
        inner = object.__getattribute__(self, "_inner")
        on = object.__getattribute__(self, "_on")
        on("pre")
        inner.run_substeps(*a, **kw)
        on("post")


def main() -> None:
    lvl = load_level("playground", levels_dir=str(ROOT / "levels"))
    lvl = replace(lvl, entities=[e for e in lvl.entities
                                 if e.class_name not in ("vent", "duct")])
    sim = Simulation(lvl, seed=1, breach_physics=bp, enable_recorder=False)
    g = sim.gmap
    eos = sim.physics_runner.eos

    # --- fixture check: these really are hull tiles ------------------------
    for (y, x) in MOUTH_TILES:
        nb = [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]
        assert g.solid[y, x], f"{(y, x)} is not a wall"
        assert any(g.is_vacuum[a, b] for a, b in nb), f"{(y, x)} has no vacuum side"
        assert any((not g.solid[a, b]) and (not g.is_vacuum[a, b])
                   for a, b in nb), f"{(y, x)} has no air side"

    for _ in range(SETTLE_TICKS):
        sim.set_paused(False)
        sim.step()
    # P-G1b: no re-derive -- D1 is live and the field is the truth, so
    # re-deriving it from the mirror here would destroy the settle's own
    # sub-count state before the gate ever measured it.

    room = g._gas_energy_accountable().copy()   # the whole pressurized volume
    t_amb_raw = g._gas_energy_t_amb_raw()
    n0 = _n_plane(g)
    E0 = _sum_obj(g.gas_energy[room])
    N0 = _sum_obj(n0[room])
    T0_room = (E0 / N0 - t_amb_raw) / Q if N0 else 0.0
    T0_plane = g.temperature.astype(np.int64).copy()

    print(f"VENT gate — playground hull breached at {MOUTH_TILES} to vacuum, "
          f"{RUN_TICKS / TPS:.0f} s")
    print(f"  t=0 (post-settle): interior T = {T0_room:+.2f} game-deg, "
          f"P = {float(g.atmosphere[room].mean()) / Q:.3f} atm, "
          f"N = {N0 / Q:.1f} atm-equiv over {int(room.sum())} cells")

    # --- THE BREACH --------------------------------------------------------
    # P-G1b: `destroy_wall` is an energy writer now (design 2.7) -- the
    # breached tile's stored energy retires and, where a tile joins open air
    # instead of the boundary, its seed is born at ambient. No re-derive.
    for (y, x) in MOUTH_TILES:
        g.destroy_wall(y, x)

    # --- instrumentation ---------------------------------------------------
    state = {"pre": 0, "acct": None}
    tally = dict(neg_cells=0, neg_worst=0, ticks=0, bad_closure=0,
                 worst_resid=0, work_export=0, dE_room=0, n_sub=[],
                 rail=0, wipe=0, kick=0, drag=0, transport=0, resync=0)

    def _bracket(phase):
        if phase == "pre":
            state["acct"] = g._gas_energy_accountable()
            state["pre"] = _sum_obj(g.gas_energy[state["acct"]])
            return
        acct = state["acct"]
        post = _sum_obj(g.gas_energy[acct])
        expected = (int(eos.e_entry_resync_sum) + int(eos.e_transport_net_sum)
                    - int(eos.e_wipe_sum) - int(eos.e_kick_ke_sum)
                    + int(eos.e_drag_heat_sum) - int(eos.e_work_export_sum)
                    + int(eos.e_rail_sum))
        resid = (post - state["pre"]) - expected
        tally["dE_inside"] = tally.get("dE_inside", 0) + (post - state["pre"])
        tally["counted"] = tally.get("counted", 0) + expected
        tally["ticks"] += 1
        if resid:
            tally["bad_closure"] += 1
            tally["worst_resid"] = max(tally["worst_resid"], abs(resid))
        tally["work_export"] += int(eos.e_work_export_sum)
        tally["rail"] += int(eos.e_rail_sum)
        tally["wipe"] += int(eos.e_wipe_sum)
        tally["kick"] += int(eos.e_kick_ke_sum)
        tally["drag"] += int(eos.e_drag_heat_sum)
        tally["transport"] += int(eos.e_transport_net_sum)
        tally["resync"] += int(eos.e_entry_resync_sum)
        tally["n_sub"].append(int(eos.dbg_last_n_sub))

    sim.physics_runner.engine = _EngineProbe(sim.physics_runner.engine, _bracket)

    for _ in range(RUN_TICKS):
        sim.set_paused(False)
        sim.step()
        # (1) positivity, EVERY tick, over the whole accountable set.
        acct = g._gas_energy_accountable()
        neg = g.gas_energy[acct] < 0
        if neg.any():
            tally["neg_cells"] += int(neg.sum())
            tally["neg_worst"] = min(tally["neg_worst"],
                                     int(g.gas_energy[acct][neg].min()))

    # --- report ------------------------------------------------------------
    n1 = _n_plane(g)
    room_now = room & g._gas_energy_accountable()
    E1 = _sum_obj(g.gas_energy[room_now])
    N1 = _sum_obj(n1[room_now])
    T1_room = (E1 / N1 - t_amb_raw) / Q if N1 else 0.0
    ns = tally["n_sub"]

    print(f"  (1) gas_energy >= 0 everywhere, every tick: "
          f"{'PASS' if tally['neg_cells'] == 0 else 'FAIL'}"
          + ("" if tally["neg_cells"] == 0 else
             f" ({tally['neg_cells']} cell-ticks, worst {tally['neg_worst']})"))
    print(f"  (2) n_sub: first={ns[0]} max={max(ns)} last={ns[-1]} "
          f"mean={sum(ns) / len(ns):.2f}  "
          f"{'STABLE' if ns[-1] <= max(2, ns[0]) else 'STILL ELEVATED'}")
    # (3) the mouth cools — as a RING profile, which is the actual §3 claim.
    dT = (g.temperature.astype(np.int64) - T0_plane) / Q
    my, mx = MOUTH
    rings = []
    for rad in range(0, 5):
        m = np.zeros_like(room)
        ys, xs = np.ogrid[:m.shape[0], :m.shape[1]]
        d = np.maximum(np.abs(ys - my), np.abs(xs - mx))
        m = (d == rad) & (~g.solid) & (~g.is_vacuum)
        rings.append(float(dT[m].mean()) if m.any() else float("nan"))
    print("  (3) dT by Chebyshev ring from the mouth (game-deg): "
          + "  ".join(f"r{r}={v:+7.2f}" for r, v in enumerate(rings)))
    print(f"      room mean T {T0_room:+.2f} -> {T1_room:+.2f} "
          f"(N-weighted, {'COOLED' if T1_room < T0_room else 'WARMED'})")
    # (4) the room's energy loss vs the counted channels.
    dE_room = E1 - E0
    inside = tally.get("dE_inside", 0)
    print(f"  (4) interior d(Sum E) over the run = {dE_room}")
    print(f"      ... INSIDE the EOS step  = {inside}")
    print(f"      ... outside it           = {dE_room - inside}   (the thermal "
          f"solver + the P-G0 per-tick refresh + the changing accountable set)")
    print(f"      counted channels (per-tick sums): "
          f"work_export={tally['work_export']} transport={tally['transport']} "
          f"kick={tally['kick']} drag={tally['drag']} rail={tally['rail']} "
          f"wipe={tally['wipe']} resync={tally['resync']} (retired, 0)")
    print(f"      Sum(counted) = {tally.get('counted', 0)}   "
          f"residual vs INSIDE = {inside - tally.get('counted', 0)}  "
          f"(this is the LSB-exactness ask -- it must be 0)")
    print(f"      CLOSURE IDENTITY over {tally['ticks']} EOS steps: "
          f"{'EXACT' if tally['bad_closure'] == 0 else 'BROKEN'}"
          + ("" if tally["bad_closure"] == 0 else
             f" ({tally['bad_closure']} bad, worst |resid| "
             f"{tally['worst_resid']})"))
    print(f"      hits: rad_clip={int(eos.rad_clip_hits)} "
          f"p_floor={int(eos.p_face_floor_hits)} "
          f"p_ceil={int(eos.p_face_ceil_hits)} "
          f"flux_sat={int(eos.flux_sat_hits)} "
          f"t_max={int(eos.t_max_phys_hits)} "
          f"rail_shortfall={int(eos.e_energy_floor_sum)}")


if __name__ == "__main__":
    main()
