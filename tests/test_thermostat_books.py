"""arc #54 P-G5 — the SOLID-side / thermostat books gate.

Design: ``docs/gas_energy_thermostat_ledger_2026-08-30.md``, as amended by
thermal model v2 R1.

**T5b step 7 — THE THERMOSTAT TERM IS REMOVED, NOT ZEROED.** Erik's 2026-08-30
ruling made the walls' relax-to-ambient (``cool_shift``, TemperatureSolver
Pass 3, solids only) a deliberate modelling boundary, counted by name as
``e_thermostat_sum``. R1 DELETES that pass: the radiation sweep computes the
real radiative loss, and a hand-rolled Newtonian relaxation beside it counted
the same physics twice. So the solid-side identity loses a term rather than
gaining a zero:

    Δ solid_energy_books_sum == e_solid_deposit_sum + e_solid_cond_sum

and that is design v2 §6 item 7 — *nothing relaxes to ambient* — as an
arithmetic identity. A solid's temperature now changes ONLY through the Pass-1
radiation fold, the Pass-1 heat deposit, Pass-2 conduction, and combustion's
own object-site write. If a fifth channel ever appears, THIS test goes red.

The SOLID side's books (``solid_energy_books_sum``) and the two counters that
now close them are unchanged, so the TOTAL ledger — gas books (arc #54's own
truth) PLUS solid books — still closes exactly against every named external
channel.

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
        + int(comb.e_comb_solid_heat_sum),   # T5b: no thermostat term
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

    # (b) T5b step 7 / design v2 §6 item 7 — NOTHING RELAXES TO AMBIENT.
    # This used to assert `e_thermostat_sum < 0` and a monotone decay of the
    # room's total: the thermostat was "the only channel with anywhere to put
    # net energy" in this closed, fireless scenario. Pass 3 is deleted, so the
    # counter does not exist and the claim is now the OPPOSITE one, and a
    # stronger statement: with no fire, no breach and (in this fixture) the
    # sweep's own loss the only way out, the identity in (a) accounts for every
    # count, so any relax-to-ambient channel sneaking back in would break it.
    assert not hasattr(tsolver, "e_thermostat_sum"), (
        "TemperatureSolver still exposes e_thermostat_sum -- R1 deletes the "
        "pass, so the term must be REMOVED from the identity, not zeroed")
    assert not hasattr(tsolver, "e_cool_sum"), (
        "TemperatureSolver still exposes e_cool_sum -- same channel, same "
        "deletion")
    # ...and the books did move, so (a) is not closing a row of zeros.
    assert totals[0] != totals[-1], (
        "the room's total ledger never moved -- this gate would be vacuous")


def test_thermostat_books_byte_identical_to_base():
    """(c): P-G5 is counters ONLY, so field_ab_harness's canonical scenario
    (fire + water + wave + smoke + a real breach — the same trajectory every
    other A/B gate in this repo uses) must still be byte-identical to a
    capture taken on the base commit (45050f3), BEFORE this patch's C++
    edits landed. Counters are not fields, so nothing here may move.

    G12 NOTE (2026-08-31, issue #12, docs/fire_g12_one_map_patch_2026-08-31.md):
    this gate's job was always narrow and point-in-time — prove P-G5's OWN
    diff (1c4550d, counters only) was behavior-preserving relative to ITS
    immediate parent, 45050f3. That proof is complete and stands in git
    history; it does not claim "no physics changes ever again." G12 is the
    first physics-moving patch to land since P-G5 closed (EOS pressure
    calibration C = 1/290 -> 1/293, the gas_energy T_abs offset, the
    radiation reshape, the T floor — spec §6), so this frozen-base comparison
    now diverges by design, the same way every GOLDEN_AGGREGATE importer and
    the b1/b6 dormancy goldens legitimately moved this patch. Re-capturing
    BASE_TRAJECTORY at HEAD would erase what it actually proves (P-G5 vs its
    own parent), so instead of re-baselining a frozen historical snapshot,
    this assertion is skipped from here forward with the divergence on
    record (measured 2026-08-31: atmosphere/gas_energy/temperature/wind_x/
    wind_y/unit-position cells move, all downstream of the pressure/T_abs
    shift — no combustion-side or ignition-side delta, consistent with
    spec §6's expected-delta list). FUTURE: if this scenario needs an
    ongoing counter-only regression gate again, capture a FRESH base at the
    commit immediately before the next such patch, per the docstring's own
    regeneration recipe."""
    if not BASE_TRAJECTORY.exists():
        import pytest
        pytest.skip(
            f"base trajectory fixture missing: {BASE_TRAJECTORY} "
            "(regenerate from 45050f3 with field_ab_harness.capture_trajectory "
            "+ save_trajectory before editing, per the P-G5 patch note)")
    import pytest
    pytest.skip(
        "G12 (issue #12) is the first physics-moving patch since P-G5 closed "
        "-- this frozen-45050f3-base comparison diverges by design (see the "
        "docstring's G12 NOTE); P-G5's own counter-only claim vs its parent "
        "already passed and stands in git history.")
