"""The HEAT TRIGGER on `timed_charge` (levelgen vocabulary E.2 row 1,
docs/levelgen_zone_vocabulary_2026-10-29.md; Erik's ruling 2026-10-31).

A ``trigger = "heat"`` charge waits until its own tile's Q16.16 temperature
mirror reads at or above ``trigger_temp`` (game degrees above ambient), then
latches its synced ``fired`` row and runs ``first_at_s`` as a fuse before it
executes its payload through THE payload executor, once. The department
payload rows (``magazine_cookoff``, ``hazmat_release``) and gen_ship's
``hazard`` room tag ride on it.

Each test names the property it protects and the change that must break it.
Heat is delivered through the canonical deposit path (``payloads.deposit_
heat``: a ``heat`` FieldEdit the temperature solver converts) -- never a bare
``temperature[...] =``, which moves no books and is vacuous on a gas cell
(CLAUDE.md "Gas temperature is a mirror").

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_timed_charge_heat.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests", ROOT / "tools",
           ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402
import level_lib  # noqa: E402
import level_loader  # noqa: E402
from config import CFG, ticks_from_seconds  # noqa: E402
from field_ab_harness import capture_snapshot  # noqa: E402
from field_digest import tick_digest  # noqa: E402
from level_loader import EntityInstance, LevelData  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation.entities import REGISTRY  # noqa: E402
from simulation.entities.serialize import entity_records  # noqa: E402
from simulation.events import ExplosionEvent  # noqa: E402
from simulation.field_edit import EditQueue  # noqa: E402
from simulation.gas_fixed import quantize_q16_scalar  # noqa: E402
from simulation.materials import (MAT_AIR, MAT_FURNITURE, MAT_HULL,  # noqa: E402
                                  MAT_WOOD)
from simulation.payloads import deposit_heat, execute_payload  # noqa: E402
from simulation.timed_charge_system import (EVENT_KIND,  # noqa: E402
                                            TimedChargeError)

TPS = int(CFG.clock.ticks_per_second)
CHARGE_AT = (6, 6)          # (y, x)
CRATE_AT = (6, 9)
THRESHOLD = 50.0            # game degrees above ambient
HEAT_AT_STEP = 5            # the step whose start carries the deposit
HEAT = 3000.0               # far past THRESHOLD on one gas tile


def _room(entities=(), hh=14, crate=True):
    tm = np.full((hh, hh), MAT_HULL, dtype=np.int32)
    tm[1:hh - 1, 1:hh - 1] = MAT_AIR
    if crate:
        tm[CRATE_AT] = MAT_FURNITURE
    return LevelData(name="heat_charge", version="2", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."),
                     entities=list(entities), wires=[])


def _charge(eid="c1", ordinal=0, **over):
    fields = {f.name: f.default for f in REGISTRY["timed_charge"].FIELDS}
    fields.update(x=CHARGE_AT[1], y=CHARGE_AT[0], payload="studio_small",
                  first_at_s=0.5, period_s=0.0, enabled=True,
                  trigger="heat", trigger_temp=THRESHOLD)
    fields.update(over)
    return EntityInstance(id=eid, class_name="timed_charge", ordinal=ordinal,
                          tags=(), fields=fields)


def _sim(entities, seed=3, crate=True, physics=bp):
    return Simulation(_room(entities, crate=crate), seed=seed,
                      breach_physics=physics, enable_recorder=False)


def _step(sim):
    sim.set_paused(False)            # the ruleset auto-pauses at seams
    sim.step()
    return [e for e in sim.tick_events
            if isinstance(e, ExplosionEvent) and e.kind == EVENT_KIND]


def _run_heated(sim, n, heat_at=HEAT_AT_STEP, heat=HEAT):
    """Step ``n`` times, delivering ``heat`` on the charge tile through the
    canonical deposit just before step ``heat_at``. Returns (the step the
    latch closed, the steps that carried a detonation, the tile's Q16
    temperature at the latch step)."""
    rt = sim._timed_charges[0]
    latched, fired, t_at_latch = None, [], None
    for i in range(n):
        if i == heat_at:
            deposit_heat(sim.gmap, sim.edit_queue, CHARGE_AT[0], CHARGE_AT[1],
                         heat, 1.5)
        before = rt.fired
        if _step(sim):
            fired.append(i)
        if rt.fired and not before:
            latched = i
            t_at_latch = int(sim.gmap.temperature[CHARGE_AT])
    return latched, fired, t_at_latch


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

def test_a_heat_charge_on_an_ambient_tile_never_fires():
    """Property: a heat charge in an ambient room never detonates and never
    latches -- the timer clock means nothing to it. Breaks if a heat charge
    falls through to the timer countdown (first_at_s read as a deadline), if
    the latch compares against 0 instead of the threshold, or if the
    threshold is quantized to 0."""
    sim = _sim([_charge(first_at_s=0.1)])
    rt = sim._timed_charges[0]
    for i in range(10 * TPS):
        assert not _step(sim), f"an unheated heat charge fired at step {i}"
    assert rt.fired == 0 and rt.ticks_to_fire == -1
    assert int(sim.gmap.fire.astype(np.int64).sum()) == 0


def test_heat_past_the_threshold_fires_once_after_its_fuse():
    """Property: heat delivered through the canonical deposit path latches
    the charge in the first step its tile reads >= trigger_temp, and the
    payload detonates exactly ticks_from_seconds(first_at_s) steps later,
    once, through the executor (ExplosionEvent at the charge's tile) --
    never again however hot the tile stays. Breaks if the fuse is counted
    from load instead of from the crossing, if it is off by one against the
    timer's countdown semantics, if the charge re-arms (a heat charge
    repeating), or if the payload bypasses execute_payload."""
    fuse = ticks_from_seconds(0.5, TPS)
    sim = _sim([_charge(first_at_s=0.5)])
    latched, fired, t_q = _run_heated(sim, HEAT_AT_STEP + fuse + 4 * TPS)
    assert latched is not None and latched >= HEAT_AT_STEP, \
        "the heat never latched the charge (or it latched before any heat)"
    assert t_q >= quantize_q16_scalar(THRESHOLD)
    assert fired == [latched + fuse], (latched, fired)
    rt = sim._timed_charges[0]
    assert rt.fired == 1 and rt.ticks_to_fire == -1   # spent, latched


def test_a_zero_fuse_fires_in_the_crossing_step():
    """Property: first_at_s = 0 on a heat charge means no fuse -- the
    detonation is in the very step the threshold is crossed. Breaks if the
    fuse goes through ticks_from_seconds' one-tick floor, or if the latch
    and the countdown run in different sweeps."""
    sim = _sim([_charge(first_at_s=0.0)])
    latched, fired, _ = _run_heated(sim, HEAT_AT_STEP + 2 * TPS)
    assert latched is not None and fired == [latched]


def test_a_disabled_heat_charge_never_fires_however_hot():
    """Property: enabled = false holds for heat charges too. Breaks if the
    heat latch ignores ``enabled``."""
    sim = _sim([_charge(first_at_s=0.0, enabled=False)])
    latched, fired, _ = _run_heated(sim, HEAT_AT_STEP + 2 * TPS)
    assert latched is None and fired == []


def test_a_timer_charge_still_fires_on_its_clock():
    """Property: an explicit trigger = "timer" behaves exactly as the
    pre-heat charge did -- fires at ticks_from_seconds(first_at_s) from load,
    whatever its tile's temperature. Breaks if the trigger switch changed
    the timer path."""
    first = ticks_from_seconds(1.0, TPS)
    sim = _sim([_charge(trigger="timer", trigger_temp=0.0, first_at_s=1.0)])
    fired = [i for i in range(first + 2 * TPS) if _step(sim)]
    assert fired == [first]


# ---------------------------------------------------------------------------
# Digest
# ---------------------------------------------------------------------------

def test_the_heat_latch_is_digested_and_timer_records_carry_none():
    """Property: a heat charge's ENTITY_SECT record carries its ``fired``
    latch, so two sims differing only in it hash differently (record bytes
    and tick_digest); a timer charge's record carries no ``fired`` row, so
    every pre-existing timer charge serializes byte-identically. Breaks if
    the latch is dropped from runtime_digest_rows, or emitted on timer
    charges (which would move every charge-bearing digest)."""
    a = _sim([_charge()], crate=False, physics=None)
    b = _sim([_charge()], crate=False, physics=None)
    b._timed_charges[0].fired = 1
    (ra,), (rb,) = entity_records(a.entities), entity_records(b.entities)
    assert b"fired|" in ra and ra != rb
    assert (tick_digest(capture_snapshot(a, capture_units=False))
            != tick_digest(capture_snapshot(b, capture_units=False)))
    t = _sim([_charge(trigger="timer", first_at_s=1.0)], crate=False,
             physics=None)
    (rt,) = entity_records(t.entities)
    assert b"fired|" not in rt and b"ticks_to_fire|" in rt


# ---------------------------------------------------------------------------
# The door
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("over, words", [
    ({"trigger_temp": 0.0}, "trigger_temp"),
    ({"trigger": "smoke"}, "'smoke'"),
    ({"period_s": 2.0}, "period_s"),
])
def test_the_door_refuses_a_bad_heat_charge_by_name(over, words):
    """Property: a heat charge without a positive threshold, an unknown
    trigger, and a repeating heat charge are load errors naming the field.
    Breaks if any of the three is silently accepted (a heat charge that can
    never fire, a typo'd trigger running as a timer, a cook-off that
    repeats)."""
    with pytest.raises(TimedChargeError, match=words):
        _sim([_charge(**over)], crate=False, physics=None)


def test_level_lib_round_trips_the_heat_fields(tmp_path):
    """Property: the one writer writes trigger / trigger_temp and the loader
    reads them back equal, re-formatting byte-identically. Breaks if a new
    field's kind cannot be written or parsed."""
    lvl = tmp_path / "lvl"
    lvl.mkdir()
    level_lib.write_level_header(lvl, name="rt", tile_size_m=1.0)
    tm = np.full((6, 6), MAT_HULL, dtype=np.int32)
    tm[1:5, 1:5] = MAT_AIR
    level_lib.write_tilemap_csv(lvl, tm, csv_bak=False)
    from PIL import Image
    Image.new("RGB", (6, 6)).save(lvl / "diffuse.png")
    keys = ("x", "y", "payload", "trigger", "trigger_temp", "first_at_s")
    inst = EntityInstance(id="c1", class_name="timed_charge", ordinal=0,
                          fields={"x": 2, "y": 3,
                                  "payload": "magazine_cookoff",
                                  "trigger": "heat", "trigger_temp": 120.0,
                                  "first_at_s": 1.5},
                          authored_keys=keys)
    level_lib.write_managed_blocks(
        lvl / "level.toml",
        {"entity": lambda nl: level_lib.format_entity_lines([inst], nl)})
    data = level_loader.load("lvl", str(tmp_path))
    (back,) = data.entities
    assert {k: back.fields[k] for k in keys} == {k: inst.fields[k] for k in keys}
    assert (level_lib.format_entity_lines([back])
            == level_lib.format_entity_lines([inst]))


# ---------------------------------------------------------------------------
# The department payload rows
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("row", ["magazine_cookoff", "hazmat_release"])
def test_the_department_rows_execute_through_the_executor(row):
    """Property: the cook-off and hazmat rows exist and execute through THE
    payload executor on a small map, their edits flush, and the sim steps
    on. Breaks if a row is missing, names an unknown explosive or gas, or
    asks the executor for a conservative gas (emit_gas refuses bulk rows)."""
    sim = _sim([], crate=False)
    payload = sim.weapons_tables.payloads.by_name[row]
    events = []
    execute_payload(sim.gmap, sim.edit_queue, sim.units, CHARGE_AT[0],
                    CHARGE_AT[1], payload, sim.rng, events=events,
                    kind=EVENT_KIND)
    assert any(isinstance(e, ExplosionEvent) for e in events)
    for _ in range(3):
        _step(sim)
    if row == "hazmat_release":
        gid = int(sim.gmap.gases.name_to_id["fuel_gas"])
        assert int(sim.gmap.gas[gid].astype(np.int64).sum()) > 0, \
            "the hazmat release put no fuel_gas in the air"


# ---------------------------------------------------------------------------
# gen_ship's hazard tag
# ---------------------------------------------------------------------------

def test_a_hazard_tagged_room_gets_one_heat_charge_and_loads(tmp_path):
    """Property: a gen_ship room tagged ``hazard`` is written as exactly one
    heat-triggered timed_charge, on a floor tile of that room with no fuel
    in its 3x3, carrying the tag's payload / threshold / fuse, and the
    level loads and steps headless with the charge waiting. Breaks if stage
    2 drops the tag, places the charge on a wall or fuel, mistranslates the
    fuse, or writes a field the loader's door refuses."""
    import gen_ship as G
    hz = {"payload": "magazine_cookoff", "trigger_temp": 150.0,
          "fuse_s": 1.0}
    rooms = [G.Room(0, 4, 4, 10, 10, "spawn", {"spawn_team": 0}),
             G.Room(1, 15, 4, 10, 10, "section", {"fuel": 2, "hazard": hz}),
             G.Room(2, 26, 4, 10, 10, "spawn", {"spawn_team": 1})]
    edges = [G.Edge(0, 1, "door", at=4), G.Edge(1, 2, "door", at=4)]
    g = G.ShipGraph(width=40, height=18, hull=(3, 3, 36, 14), rooms=rooms,
                    edges=edges, mirror_x=False, max_per_team=4, zombies=0)
    emb = G.embed(g, np.random.Generator(np.random.PCG64(0)))
    assert G.validate(emb.tilemap, emb.spawn_rooms, mirrored=False,
                      graph=g) == []
    plan = G.ShipPlan(seed=0, size="fixture", attempt=0, graph=g,
                      embedding=emb)
    G.write(plan, tmp_path / "hz")
    lv = level_loader.load("hz", levels_dir=str(tmp_path))
    charges = [e for e in lv.entities if e.class_name == "timed_charge"]
    assert len(charges) == 1
    f = charges[0].fields
    assert (f["trigger"], f["payload"], f["trigger_temp"], f["first_at_s"]) \
        == ("heat", "magazine_cookoff", 150.0, 1.0)
    x, y = f["x"], f["y"]
    x0, y0, x1, y1 = rooms[1].box
    assert x0 <= x <= x1 and y0 <= y <= y1
    tm = plan.tilemap
    assert tm[y, x] == MAT_AIR
    assert not np.isin(tm[y - 1:y + 2, x - 1:x + 2],
                       (MAT_FURNITURE, MAT_WOOD)).any()
    sim = Simulation(lv, seed=1, breach_physics=bp, enable_recorder=False)
    (rt,) = sim._timed_charges
    for _ in range(5):
        assert not _step(sim)
    assert rt.heat and rt.fired == 0


def test_a_malformed_hazard_tag_is_refused():
    """Property: the hazard tag is schema-checked like every room tag.
    Breaks if a hazard without a positive threshold passes stage 1."""
    import gen_ship as G
    bad = {"payload": "magazine_cookoff", "trigger_temp": 0, "fuse_s": 1.0}
    rooms = [G.Room(0, 4, 4, 10, 10, "spawn", {"spawn_team": 0}),
             G.Room(1, 15, 4, 10, 10, "section", {"hazard": bad}),
             G.Room(2, 26, 4, 10, 10, "spawn", {"spawn_team": 1})]
    edges = [G.Edge(0, 1, "door", at=4), G.Edge(1, 2, "door", at=4)]
    with pytest.raises(G.GraphSchemaError, match="hazard"):
        G.ShipGraph(width=40, height=18, hull=(3, 3, 36, 14), rooms=rooms,
                    edges=edges, mirror_x=False, max_per_team=4, zombies=0)
