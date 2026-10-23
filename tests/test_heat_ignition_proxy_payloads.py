"""#79 — ignition by delivered heat, and the frag_heat/frag_hot_mass proxy rows.

Erik's ruling (docs/prep_patches_handoff_2026-09-30.md P3): a fire is started
by DELIVERING HEAT, never by writing ``fire`` alone (CLAUDE.md "Starting a
fire"). ``apply_explosion``'s old heatless ``fire = max(fire, 0.5*falloff)``
ignite block (a decorative fire: no O2 draw, no fuel burn, ~125 s e-fold to
die) is DELETED. Ignition now happens only through the existing heat ->
temperature -> ``apply_temperature_ignition`` path, driven by a payload's
``heat_amount``/``heat_radius`` columns (the W6 plasma-splash mechanism,
already wired through ``payloads.py::execute_payload``/``deposit_heat``).

Two new provisional config rows exercise this for the grenade family:
``[payloads.frag_heat]`` (heat only, no blast pressure) and
``[payloads.frag_hot_mass]`` (frag_standard's blast plus the same heat
deposit).

The HEATLESS-BLAST properties below used to run on ``[payloads.frag_standard]``
(the shipped grenade/40mm payload) as a mass-only control. Erik keeps
retuning that row, so per his 2026-10-03 ruling ("tests must never break on
retuning a config row") they now run on ``tests/_test_charge.py``'s
``TEST_CHARGE`` instead — a payload with ``heat_amount == 0.0`` the tests
own outright, never a shipped row.

Properties gated here (breaks if the ruling above is reverted, or if a new
igniter is added the CLAUDE.md-required way but with `fire` written directly
instead of heat):

  - ``execute_payload`` with ``frag_heat`` beside a furniture (fuel) tile
    produces an HONEST fire within a few seconds: ``fire > 0`` at the tile,
    its local O2 falls, and its fuel (``wall_hp``) falls. A DECORATIVE fire
    (the deleted mechanism) would show fire > 0 with NEITHER O2 nor fuel
    moving — this test would not distinguish the two if it only checked
    ``fire > 0``, so it checks the honest-fire triple.
  - a HEATLESS blast (TEST_CHARGE, ``heat_amount == 0.0``) queues NO ``fire``
    FieldEdit at all from ``apply_explosion`` — the property #79 deletes,
    checked directly on the edit queue rather than by absence-of-effect
    (which a coincidentally-cold blast could also produce).
  - a HEATLESS blast leaves the whole level fire-free over a several-second
    window (the emergent, game-visible form of the same property).

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_heat_ignition_proxy_payloads.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402
from config import CFG  # noqa: E402
from level_loader import LevelData  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation.gases import O2  # noqa: E402
from simulation.materials import MAT_AIR, MAT_FURNITURE, MAT_HULL  # noqa: E402
from simulation.payloads import execute_payload  # noqa: E402

from _test_charge import TEST_CHARGE  # noqa: E402

TPS = int(CFG.clock.ticks_per_second)
RUN_SECONDS = 3.0
RUN_TICKS = round(RUN_SECONDS * TPS)


def _room_with_crate(hh=14, crate_at=None):
    """A hull-walled square room, MAT_AIR interior, one MAT_FURNITURE (fuel)
    tile — built the same way test_eos_p4_combustion.py's `_sealed_room`
    builds its wood-fuel scenarios (level v2: tilemap codes ARE material
    ids)."""
    tm = np.full((hh, hh), MAT_HULL, dtype=np.int32)
    tm[1:hh - 1, 1:hh - 1] = MAT_AIR
    tm[crate_at] = MAT_FURNITURE
    return LevelData(name="p79_test", version="2", path=Path("."),
                     tilemap=tm, tile_size_m=1.0 / 3.0, diffuse_path=Path("."))


CRATE_AT = (7, 7)


def _run(payload, ticks=RUN_TICKS, crate_at=CRATE_AT):
    """Build a fresh sim, execute `payload` (a PayloadDef) centred on the
    crate tile, and step `ticks` ticks. Returns (sim, gmap)."""
    lvl = _room_with_crate(crate_at=crate_at)
    sim = Simulation(lvl, seed=1, breach_physics=bp, enable_recorder=False)
    g = sim.gmap
    assert g.material[crate_at] == MAT_FURNITURE
    assert g.flammable[crate_at], "crate tile must be a fuel-bearing (flammable) material"

    execute_payload(g, sim.edit_queue, sim.units, crate_at[0], crate_at[1],
                    payload, sim.rng)
    sim.set_paused(False)
    for _ in range(ticks):
        sim.set_paused(False)   # the ruleset auto-pauses at phase boundaries
        sim.step()
    return sim, g


# ---------------------------------------------------------------------------
# Oracle 1 — frag_heat lights an honest fire (fire + O2 draw + fuel burn)
# ---------------------------------------------------------------------------
def test_frag_heat_lights_an_honest_fire_on_a_fuel_tile():
    lvl = _room_with_crate(crate_at=CRATE_AT)
    sim = Simulation(lvl, seed=1, breach_physics=bp, enable_recorder=False)
    g = sim.gmap

    o2_before = int(g.gas[O2][CRATE_AT])
    fuel_before = int(g.wall_hp[CRATE_AT])
    assert o2_before > 0, "crate tile must start with real O2 to make the draw meaningful"
    assert fuel_before > 0, "crate tile must start with real fuel (wall_hp) to make the burn meaningful"

    payload = sim.weapons_tables.payloads.by_name["frag_heat"]
    execute_payload(g, sim.edit_queue, sim.units, CRATE_AT[0], CRATE_AT[1],
                    payload, sim.rng)
    sim.set_paused(False)
    for _ in range(RUN_TICKS):
        sim.set_paused(False)
        sim.step()

    fire_after = int(g.fire[CRATE_AT])
    o2_after = int(g.gas[O2][CRATE_AT])
    fuel_after = int(g.wall_hp[CRATE_AT])

    assert fire_after > 0, (
        "frag_heat must ignite the crate via delivered heat "
        "(apply_temperature_ignition) within 3 s")
    assert o2_after < o2_before, (
        "an honest (hotf-driven) fire draws O2 at its own tile — a decorative "
        "fire (the deleted heatless ignite) would leave O2 untouched")
    assert fuel_after < fuel_before, (
        "an honest fire burns fuel (wall_hp) at its own tile — a decorative "
        "fire would leave fuel untouched")


# ---------------------------------------------------------------------------
# Oracle 2 — a heatless blast (TEST_CHARGE) queues no `fire` edit, and stays
# fire-free. This used to run on frag_standard as a "byte-identical" mass-
# only control; Erik's 2026-10-03 ruling retargets it onto the tests' own
# TEST_CHARGE (heat_amount == 0.0) so retuning frag_standard can't break it.
# ---------------------------------------------------------------------------
def test_heatless_blast_queues_no_fire_edit():
    """The literal property #79 deletes: apply_explosion (via execute_payload)
    for a HEATLESS blast (TEST_CHARGE, heat_amount == 0.0) never enqueues a
    `fire` FieldEdit — checked on the queue itself, not by absence of a
    downstream effect (which a cold/short blast could also produce for other
    reasons). Breaks if apply_explosion's deleted heatless ignite block is
    reintroduced."""
    lvl = _room_with_crate(crate_at=CRATE_AT)
    sim = Simulation(lvl, seed=1, breach_physics=bp, enable_recorder=False)
    g = sim.gmap
    assert TEST_CHARGE.heat_amount == 0.0

    execute_payload(g, sim.edit_queue, sim.units, CRATE_AT[0], CRATE_AT[1],
                    TEST_CHARGE, sim.rng)

    fields_queued = {edit.field for edit, _seq in sim.edit_queue._edits}
    assert "fire" not in fields_queued, (
        "a heatless blast must not queue a `fire` FieldEdit — "
        "apply_explosion's heatless ignite block is deleted (#79)")


def test_heatless_blast_leaves_the_level_fire_free():
    """The emergent, game-visible form of the same property: over several
    seconds of real sim ticks, a HEATLESS blast (TEST_CHARGE) never lights a
    single tile anywhere on the level (no heat is delivered, so
    apply_temperature_ignition has nothing to trigger on). Breaks if a
    heatless blast starts igniting tiles again."""
    sim, g = _run(TEST_CHARGE)
    assert int(g.fire.astype(np.int64).sum()) == 0, (
        "frag_standard must leave the level fire-free — its blast carries "
        "no heat_amount, so nothing can cross ignition_temp")
