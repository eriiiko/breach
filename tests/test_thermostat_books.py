"""arc #54 P-G5 — the SOLID-side / thermostat books gate.

Design: ``docs/gas_energy_thermostat_ledger_2026-08-30.md``. Erik's ruling
(2026-08-30): walls decaying to ambient (``cool_shift``, TemperatureSolver
Pass 3, solids only) is a deliberate modelling boundary — "the ship's heating
system, not simulated further" — and a TWO-WAY thermostat: it also warms a
sub-ambient wall back up. This patch adds the SOLID side's own books
(``solid_energy_books_sum``) and the three counters that close them
(``e_solid_deposit_sum``, ``e_solid_cond_sum``, ``e_thermostat_sum``), so the
TOTAL ledger — gas books (arc #54's own truth) PLUS solid books — closes
exactly against every named external channel. COUNTER ONLY: no physics
changed, so every field trajectory must stay byte-identical to the base
commit (45050f3, pre-P-G5) — see ``test_thermostat_books_byte_identical``.

Scenario: a small sealed hull room (``field_ab_harness``'s canonical 16x16
box — hull border, carved-out air interior, NO breach), gas seeded well
above ambient via ``seed_gas_temperature``, NO fire. Nothing drives the room
after the seed except its own conduction and the thermostat relaxing the
(now gas-warmed) walls back toward ambient — the minimal repro for the
solid-side identity, mirroring how ``_quiet_books_bench.py`` isolates the
gas-side one.

Run:
    conda run -n data python -m pytest tests/test_thermostat_books.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402
import field_ab_harness as fab  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation import gas_fixed  # noqa: E402

TICKS = 200
BASE_TRAJECTORY = ROOT / "tests" / "_pg5_base_trajectory_45050f3.pkl"


def _sealed_hot_room_sim():
    """field_ab_harness's canonical sealed 16x16 hull room, gas seeded to
    +300 game-deg above ambient, no fire/water/wave and no breach — the
    minimal P-G5 repro."""
    sim = Simulation(fab._scenario_level(), seed=1, breach_physics=bp,
                     enable_recorder=False)
    g = sim.gmap
    interior = (~g.solid) & (~g.is_vacuum)
    g.seed_gas_temperature(interior, gas_fixed.quantize_scalar(300.0))
    sim.set_paused(False)
    return sim


def _gas_books(g):
    """Sum gas_energy over the accountable set as a PYTHON int (design §2.2
    forbids an absolute int64 sum — the gate must not be the thing that
    wraps)."""
    return int(g.gas_energy[g._gas_energy_accountable()].astype(object).sum())


def _terms(g, eos, tsolver, comb, engine):
    """The FULL P-G5 total-ledger RHS: the four pre-existing gas-side groups
    (design §2.8 / temperature_solver.h's gas identity / combustion.h /
    GameMap.gas_energy_seam_net, unchanged since arc #54 P-G1b) plus the
    water-evac export, plus the new SOLID-side group (this patch). The EOS
    and water groups reset every step (read absolutely); the rest accumulate
    (differenced tick to tick by the caller)."""
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
        -int(engine.e_water_evac_export_sum),
        # P-G5: the solid side's own channels (accumulating) — the thermal
        # solver's three (Pass 1 deposit, Pass 2 conduction, Pass 3
        # thermostat) PLUS combustion's own `e_comb_solid_heat_sum`, the
        # object-site fuel deposit that bypasses TemperatureSolver's Pass 1
        # entirely (combustion.cpp writes `temperature[s]` directly).
        int(tsolver.e_solid_deposit_sum) + int(tsolver.e_solid_cond_sum)
        + int(tsolver.e_thermostat_sum) + int(comb.e_comb_solid_heat_sum),
    )


def test_thermostat_books_close_and_decay():
    sim = _sealed_hot_room_sim()
    g = sim.gmap
    eos = sim.physics_runner.eos
    tsolver = sim.physics_runner.engine.temperature
    comb = sim.physics_runner.combustion
    engine = sim.physics_runner.engine

    prev_total = _gas_books(g) + int(tsolver.solid_energy_books_sum)
    prev_terms = _terms(g, eos, tsolver, comb, engine)
    totals = [prev_total]
    bad = worst = worst_tick = 0

    for t in range(1, TICKS + 1):
        sim.step()
        gas_now = _gas_books(g)
        solid_now = int(tsolver.solid_energy_books_sum)
        total_now = gas_now + solid_now
        terms = _terms(g, eos, tsolver, comb, engine)
        expected = (
            terms[0]                        # EOS: absolute (reset/step)
            + (terms[1] - prev_terms[1])     # thermal solver gas side
            + (terms[2] - prev_terms[2])     # combustion
            + (terms[3] - prev_terms[3])     # python seams
            + terms[4]                       # water evac: absolute
            + (terms[5] - prev_terms[5])     # P-G5: thermal solver solid side
        )
        resid = (total_now - prev_total) - expected
        if resid:
            bad += 1
            if abs(resid) > abs(worst):
                worst, worst_tick = resid, t
        totals.append(total_now)
        prev_total, prev_terms = total_now, terms

    # (a) THE TOTAL LEDGER (gas books + solid books) closes EXACTLY, every
    # tick, in int64 — the P-G5 extension of arc #54's own gas-only identity.
    assert bad == 0, (
        f"P-G5 total ledger broken on {bad}/{TICKS} ticks, "
        f"worst |resid|={worst} @ tick {worst_tick}")

    # (b) a hot-seeded sealed room with no fire only ever pushes its walls
    # ABOVE ambient (the gas warms them via conduction), so Pass 3's
    # relax-to-ambient is a pure SINK here: e_thermostat_sum must be negative
    # (energy leaving to the thermostat), and the room's total (gas+solid)
    # energy must decay monotonically toward ambient — the thermostat is the
    # only channel with anywhere to put net energy in this closed, fireless
    # scenario.
    assert int(tsolver.e_thermostat_sum) < 0, (
        "e_thermostat_sum should be negative (heat leaving to the "
        "thermostat) in a hot-seeded sealed room with no sub-ambient cells")
    diffs = np.diff(np.array(totals, dtype=object))
    bad_rises = [(i + 1, int(d)) for i, d in enumerate(diffs) if d > 0]
    assert not bad_rises, (
        f"room energy (gas+solid) must decay monotonically toward ambient; "
        f"rose on {len(bad_rises)} tick(s), first {bad_rises[:5]}")


def test_thermostat_books_byte_identical_to_base():
    """(c): P-G5 is counters ONLY, so field_ab_harness's canonical scenario
    (fire + water + wave + smoke + a real breach — the same trajectory every
    other A/B gate in this repo uses) must still be byte-identical to a
    capture taken on the base commit (45050f3), BEFORE this patch's C++
    edits landed. Counters are not fields, so nothing here may move."""
    if not BASE_TRAJECTORY.exists():
        import pytest
        pytest.skip(
            f"base trajectory fixture missing: {BASE_TRAJECTORY} "
            "(regenerate from 45050f3 with field_ab_harness.capture_trajectory "
            "+ save_trajectory before editing, per the P-G5 patch note)")
    base = fab.load_trajectory(str(BASE_TRAJECTORY))
    new = fab.capture_trajectory(n_steps=len(base))
    fab.assert_trajectories_match(base, new, tol=0.0)
