"""#31 P5 — the `timed_charge` entity (docs/prep_patches_handoff_2026-09-30.md).

A level-authored charge executes one ``[payloads.*]`` row through THE payload
executor on the sim clock: first after ``first_at_s``, then every
``period_s`` (0 = once). Its synced state is one runtime row,
``ticks_to_fire``, in the ENTITY digest section — present only when a charge
exists, so every charge-free digest (every golden) is untouched.

Properties gated here, each naming what must break it:

- it fires at its tick, not before (the payload's ExplosionEvent appears in
  exactly that step; a fire-free level with fuel stays fire-free until then);
- it repeats at its period, across round boundaries (TwoPhaseWEGO rewinds
  ``sim.tick``); ``period_s = 0`` fires once;
- ``enabled = false`` never fires;
- the serializer carries the LIVE countdown, and a level_lib-written
  ``[[entity]]`` row round-trips through the loader;
- presence-gating: a charge-free sim never runs the sweep and serializes no
  charge record;
- determinism: two runs of the studio level give equal tick digests;
- an unknown payload name is a load error.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_timed_charge.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402
import level_lib  # noqa: E402
import level_loader  # noqa: E402
from config import CFG, ticks_from_seconds  # noqa: E402
from level_loader import EntityInstance, LevelData  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation import timed_charge_system  # noqa: E402
from simulation.entities import REGISTRY  # noqa: E402
from simulation.entities.serialize import entity_records  # noqa: E402
from simulation.events import ExplosionEvent  # noqa: E402
from simulation.materials import MAT_AIR, MAT_FURNITURE, MAT_HULL  # noqa: E402
from simulation.timed_charge_system import EVENT_KIND, TimedChargeError  # noqa: E402

TPS = int(CFG.clock.ticks_per_second)
CHARGE_AT = (6, 6)          # (y, x)
CRATE_AT = (6, 9)


def _room(entities=(), hh=14, crate=True):
    tm = np.full((hh, hh), MAT_HULL, dtype=np.int32)
    tm[1:hh - 1, 1:hh - 1] = MAT_AIR
    if crate:
        tm[CRATE_AT] = MAT_FURNITURE
    return LevelData(name="p5_charge", version="2", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."),
                     entities=list(entities), wires=[])


def _charge(eid="c1", ordinal=0, **over):
    fields = {f.name: f.default for f in REGISTRY["timed_charge"].FIELDS}
    fields.update(x=CHARGE_AT[1], y=CHARGE_AT[0], payload="studio_small",
                  first_at_s=1.0, period_s=0.0, enabled=True)
    fields.update(over)
    return EntityInstance(id=eid, class_name="timed_charge", ordinal=ordinal,
                          tags=(), fields=fields)


def _charge_steps(sim, n):
    """Step n times; return the step indices (0-based) whose tick_events
    carry a timed_charge ExplosionEvent."""
    fired = []
    for i in range(n):
        sim.set_paused(False)          # the ruleset auto-pauses at seams
        sim.step()
        if any(isinstance(e, ExplosionEvent) and e.kind == EVENT_KIND
               for e in sim.tick_events):
            fired.append(i)
    return fired


def test_charge_fires_at_its_tick_and_the_level_is_fire_free_until_then():
    """Property: a charge detonates in exactly step ticks_from_seconds(
    first_at_s) — its ExplosionEvent appears there and nowhere earlier — and
    before it nothing burns on a level with fuel beside it. Breaks if the
    sweep fires early/late (off-by-one in the countdown), if it bypasses the
    executor (no ExplosionEvent), or if the charge heats the world before it
    fires."""
    first = ticks_from_seconds(1.0, TPS)
    sim = Simulation(_room([_charge()]), seed=3, breach_physics=bp,
                     enable_recorder=False)
    g = sim.gmap
    for i in range(first):
        sim.set_paused(False)
        sim.step()
        assert not any(isinstance(e, ExplosionEvent) for e in sim.tick_events), \
            f"no detonation may happen before step {first} (saw one at {i})"
        assert int(g.fire.astype(np.int64).sum()) == 0, \
            "the level must stay fire-free until the charge fires"
    sim.set_paused(False)
    sim.step()
    ev = [e for e in sim.tick_events if isinstance(e, ExplosionEvent)]
    assert len(ev) == 1 and ev[0].kind == EVENT_KIND
    assert ev[0].pos == (CHARGE_AT[1], CHARGE_AT[0])   # (fx, fy)
    # and it is a real detonation through the executor: its heat lights the
    # crate within a few seconds (delivered heat -> apply_temperature_ignition)
    for _ in range(3 * TPS):
        sim.set_paused(False)
        sim.step()
    assert int(g.fire[CRATE_AT]) > 0, "studio_small's heat must ignite the crate"


def test_charge_repeats_at_its_period_across_round_boundaries():
    """Property: fires at first + k*period for every k, counted in sim steps,
    independent of the ruleset's round rewind of sim.tick. Breaks if the
    schedule is keyed to sim.tick (TwoPhaseWEGO rewinds it each round, so an
    absolute deadline re-fires or never fires), or if the period reload is
    off by one."""
    first = ticks_from_seconds(0.5, TPS)
    period = ticks_from_seconds(1.5, TPS)
    sim = Simulation(_room([_charge(first_at_s=0.5, period_s=1.5)], crate=False),
                     seed=1, breach_physics=None, enable_recorder=False)
    n = 2 * sim._ticks_per_round + 1     # crosses two round rewinds
    fired = _charge_steps(sim, n)
    assert fired == list(range(first, n, period))


def test_zero_period_fires_once():
    """Property: period_s = 0 means exactly one detonation. Breaks if a zero
    period reloads the countdown (fires every step or periodically)."""
    first = ticks_from_seconds(0.25, TPS)
    sim = Simulation(_room([_charge(first_at_s=0.25, period_s=0.0)], crate=False),
                     seed=1, breach_physics=None, enable_recorder=False)
    assert _charge_steps(sim, first + 4 * TPS) == [first]


def test_disabled_charge_never_fires():
    """Property: enabled = false never detonates. Breaks if the runtime
    ignores `enabled`."""
    sim = Simulation(_room([_charge(first_at_s=0.25, period_s=0.5,
                                    enabled=False)], crate=False),
                     seed=1, breach_physics=None, enable_recorder=False)
    assert _charge_steps(sim, 3 * TPS) == []


def test_serializer_carries_the_live_countdown():
    """Property: the ENTITY_SECT record of a charge carries its CURRENT
    ticks_to_fire (the synced runtime row), so the digest sees the schedule
    advance. Breaks if the runtime row is dropped, frozen at its load value,
    or read off the load-time instance instead of the runtime wrapper."""
    sim = Simulation(_room([_charge(first_at_s=1.0)], crate=False), seed=1,
                     breach_physics=None, enable_recorder=False)
    rt = sim._timed_charges[0]
    seen = set()
    for _ in range(5):
        (rec,) = entity_records(sim.entities)
        assert b"|timed_charge\n" in rec
        row = b"ticks_to_fire|" + rt.ticks_to_fire.to_bytes(8, "little",
                                                             signed=True)
        assert row in rec
        seen.add(rec)
        sim.set_paused(False)
        sim.step()
    assert len(seen) == 5, "every step's countdown must serialize differently"


def test_level_lib_round_trip(tmp_path):
    """Property: a timed_charge [[entity]] written by level_lib (the one
    writer) loads back with the same fields, and re-formats byte-identically.
    Breaks if a field kind cannot be written/parsed (e.g. the float seconds
    or the payload string) or if the writer materializes defaults."""
    lvl = tmp_path / "lvl"
    lvl.mkdir()
    level_lib.write_level_header(lvl, name="rt", tile_size_m=1.0)
    tm = np.full((6, 6), MAT_HULL, dtype=np.int32)
    tm[1:5, 1:5] = MAT_AIR
    level_lib.write_tilemap_csv(lvl, tm, csv_bak=False)
    from PIL import Image
    Image.new("RGB", (6, 6)).save(lvl / "diffuse.png")
    keys = ("x", "y", "payload", "first_at_s", "period_s", "enabled")
    inst = EntityInstance(id="c1", class_name="timed_charge", ordinal=0,
                          fields={"x": 2, "y": 3, "payload": "studio_large",
                                  "first_at_s": 6.0, "period_s": 5.0,
                                  "enabled": False},
                          authored_keys=keys)
    level_lib.write_managed_blocks(
        lvl / "level.toml",
        {"entity": lambda nl: level_lib.format_entity_lines([inst], nl)})
    data = level_loader.load("lvl", str(tmp_path))
    (back,) = data.entities
    assert back.class_name == "timed_charge"
    assert {k: back.fields[k] for k in keys} == {k: inst.fields[k] for k in keys}
    assert (level_lib.format_entity_lines([back])
            == level_lib.format_entity_lines([inst]))


def test_charge_free_level_never_runs_the_sweep(monkeypatch):
    """Property (presence gating): a level without charges builds no charge
    runtime, never calls the 9e sweep, and serializes no timed_charge record
    — so its digest is byte-identical to a build without this system (the
    GOLDEN_AGGREGATE staying put is the whole-suite evidence). Breaks if the
    sweep runs unconditionally or the section gains a record for a charge
    that does not exist."""
    def _boom(sim):
        raise AssertionError("the charge sweep ran on a charge-free level")
    monkeypatch.setattr("simulation.simulation.sweep_timed_charges", _boom)
    sim = Simulation(_room([]), seed=1, breach_physics=bp,
                     enable_recorder=False)
    assert sim._timed_charges == []
    for _ in range(10):
        sim.set_paused(False)
        sim.step()
    assert sim.get_state().entity_state["n_entities"] == 0


def test_studio_level_is_deterministic():
    """Property: two runs of levels/explosion_studio from the same seed give
    identical per-tick digests (fields + units + the entity section with the
    charges' countdowns), through two detonations. Breaks if the charge path
    draws RNG out of order, iterates in a non-ordinal order, or reads
    wall-clock time."""
    from field_ab_harness import capture_trajectory
    from field_digest import tick_digest

    level = level_loader.load("explosion_studio", str(ROOT / "levels"))
    n = ticks_from_seconds(4.0, TPS) + 2           # small + medium have fired

    def make():
        return Simulation(level, seed=7, breach_physics=bp,
                          enable_recorder=False)

    a = [tick_digest(s) for s in capture_trajectory(make, n_steps=n)]
    b = [tick_digest(s) for s in capture_trajectory(make, n_steps=n)]
    assert a == b
    # the charges really are in the hashed section (not a vacuous equality)
    sim = make()
    assert len(sim._timed_charges) == 4
    assert len(set(a)) == len(a), "the countdown must move the digest each tick"


def test_unknown_payload_is_a_load_error():
    """Property: a charge naming no [payloads.*] row fails LOUDLY at load
    (Simulation construction), naming the charge. Breaks if the name is
    resolved lazily at fire time or silently skipped."""
    with pytest.raises(TimedChargeError, match="c_bad"):
        Simulation(_room([_charge(eid="c_bad", payload="no_such_row")]),
                   seed=1, breach_physics=None, enable_recorder=False)


def test_charge_off_the_grid_is_a_load_error():
    """Property: a charge outside the grid fails at load. Breaks if the
    bounds check is dropped (the executor would clip silently)."""
    with pytest.raises(TimedChargeError, match="out of the"):
        Simulation(_room([_charge(x=99)]), seed=1, breach_physics=None,
                   enable_recorder=False)


def test_the_module_draws_no_rng():
    """Property: the sweep hands the sim rng to the executor and never draws
    from it itself (the rng state is unchanged by a fire-free step with a
    charge ticking down). Breaks if the charge path adds a draw."""
    sim = Simulation(_room([_charge(first_at_s=5.0)], crate=False), seed=1,
                     breach_physics=None, enable_recorder=False)
    before = sim.rng.bit_generator.state
    timed_charge_system.sweep_timed_charges(sim)
    assert sim.rng.bit_generator.state == before
