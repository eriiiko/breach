"""#29 P6.1 -- THE DEVICE UNIT STORE's host half (``simulation.unit_store``):
the row layout, its generated C++ header, and the bridge (``pack`` /
``unpack``) -- the IDENTITY gate of design §4.4b decision 4.

PROPERTIES PROTECTED (each test names its own and the change that breaks it)
  a. The checked-in ``cpp/src/unit_store_layout.h`` is the schema's layout.
  b. Every schema column sits in the row at its encoded width, in schema
     order, contiguous, ``valid`` last.
  c. IDENTITY on live matches, every tick: ``unpack(pack(units))`` into blank
     units leaves every group-S attribute equal, the unit digest computed
     from the ROWS equals the reference's ``unit_section_bytes``, and the
     blank units pack back to the same rows -- over a ship_demo bots match
     (shots, hits, throws, deaths, a conversion) and the scripted arena of
     ``tests/_p3a_sole_cause.py`` (every verb, a manual and an automatic
     reload), each asserted non-vacuous.
  d. In a batch, env i's rows are env i packed alone; a padding row carries
     nothing.
  e. The bridge refuses, by name, what a row cannot hold.

CPU build only; the device half is ``tests/cuda_unit_store_check.py``.
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tools", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402

import level_loader  # noqa: E402
from simulation import unit_schema, unit_store  # noqa: E402
from simulation.unit_schema import COLUMNS, column_width  # noqa: E402

SHIP_TICKS = 700       # ship_demo seed 1: first hit ~422, first kill ~457,
                       # first conversion ~673 (measured 2026-11-14)


# ---------------------------------------------------------------------------
# a / b: the layout and its header
# ---------------------------------------------------------------------------
def test_the_checked_in_header_is_the_schemas_layout():
    """Property (a): ``cpp/src/unit_store_layout.h`` is exactly the text the
    schema's layout generates. Breaks when a schema column is added,
    removed or re-widened (a config row that moves a block's width included)
    without regenerating the header -- the device kernels would read the
    wrong offsets."""
    path = ROOT / unit_store.HEADER_REL
    want = unit_store.header_text()
    have = path.read_text(encoding="utf-8") if path.is_file() else ""
    assert have == want, (
        f"{unit_store.HEADER_REL} is stale (the unit schema moved): run "
        f"`C:/Users/steen/anaconda3/envs/data/python.exe "
        f"{unit_store.GENERATOR}` and commit the header")


def test_every_column_sits_at_its_encoded_width_in_schema_order():
    """Property (b): the row is the schema's group-S columns in schema order,
    each at ``column_width`` (the width its encoding has on a real unit),
    contiguous from 0, then ``valid``, and W is one past it. Breaks if the
    layout reorders, skips or re-widens a column, or puts ``valid``
    anywhere but last."""
    from simulation.unit import Unit
    lay = unit_store.layout()
    u = Unit("probe", x=3, y=4, team=0)
    enc = unit_schema.encode_unit(u)
    off = 0
    assert [c.name for c in lay.columns] == [c.name for c in COLUMNS]
    for sc, c in zip(lay.columns, COLUMNS):
        assert sc.offset == off, sc
        assert sc.width == column_width(c) >= 1, sc
        v = enc[c.name]
        assert (len(v) if isinstance(v, tuple) else 1) == sc.width, sc
        assert sc.tuple_valued == isinstance(v, tuple), sc
        off += sc.width
    assert lay.valid == off and lay.width == off + 1


# ---------------------------------------------------------------------------
# c: IDENTITY on live matches
# ---------------------------------------------------------------------------
class _Identity:
    """Per tick: the identity checks over ``sim.units`` and coverage
    counters (so a scene that never exercised a verb shows)."""

    def __init__(self, name):
        self.name = name
        self.ticks = 0
        self.cov = collections.Counter()
        self.sim = None

    def __call__(self, sim):
        self.sim = sim
        units = list(sim.units)
        rows = unit_store.pack(units)
        blank = unit_store.blank_units(len(units))
        unit_store.unpack(rows, blank)
        for u, b in zip(units, blank):
            for c in COLUMNS:
                ref, got = getattr(u, c.store), getattr(b, c.store)
                assert got == ref, (
                    f"{self.name} tick {sim.tick}: unit {u.id} column "
                    f"{c.name!r}: {got!r} != {ref!r}")
        ref_bytes = unit_schema.unit_section_bytes(
            unit_schema.digest_record(u)
            for u in sorted(units, key=lambda u: int(u.id)))
        assert unit_store.section_bytes(rows) == ref_bytes, (
            f"{self.name} tick {sim.tick}: the digest from the rows differs")
        assert np.array_equal(unit_store.pack(blank), rows), (
            f"{self.name} tick {sim.tick}: the unpacked units pack to other "
            f"rows")
        self.ticks += 1
        for e in sim.tick_events:
            self.cov["ev:" + type(e).__name__] += 1
        for u in units:
            if u.is_zombie and u.loadout_name is not None:
                self.cov["converted_unit_ticks"] += 1
            if any(u.exposure):
                self.cov["exposed_unit_ticks"] += 1
            if u.statuses:
                self.cov["status_unit_ticks"] += 1
            if not u.alive:
                self.cov["dead_unit_ticks"] += 1
            if u.live_aim is not None or u.live_move_dir is not None:
                self.cov["held_intent_unit_ticks"] += 1
            if u.current_mag not in (unit_schema.MAG_UNBOUND,):
                self.cov["bound_mag_unit_ticks"] += 1


@pytest.fixture(scope="module")
def identity_runs():
    """ship_demo's match (bots + the horde) for SHIP_TICKS ticks and the
    scripted arena, each checked on every tick; the final sims for (d)."""
    from simulation.match_session import MatchSession
    level = level_loader.load("ship_demo", str(ROOT / "levels"))
    session = MatchSession(level, seed=1, breach_physics=bp)
    ship = _Identity("ship_demo")
    for _ in range(SHIP_TICKS):
        if session.is_terminal():
            break
        session.tick()
        ship(session.sim)

    # The scripted arena of tests/_p3a_sole_cause.py, its per-tick tape
    # swapped for the identity check (the scenario calls tape(sim) after
    # every step).
    import _p3a_sole_cause as p3a
    scripted = _Identity("scripted")
    orig = p3a._Tape
    p3a._Tape = lambda: scripted
    try:
        p3a.scenario_scripted()
    finally:
        p3a._Tape = orig
    return {"ship": ship, "scripted": scripted,
            "ship_sim": session.sim, "scripted_sim": scripted.sim}


def test_identity_ship_demo_every_tick(identity_runs):
    """Property (c), the match half: on every tick of ship_demo's bots match
    the rows alone rebuild every group-S attribute, the digest from the rows
    is the reference's bytes, and the rebuilt units pack to the same rows;
    the scene saw shots, hits, throws, kills, a conversion, exposure and
    statuses. Breaks if an encoding loses a value (a sentinel collides, a
    block is truncated, an optional drops a field), the layout misplaces a
    column, or :func:`record_from_row` diverges from ``digest_record``."""
    run = identity_runs["ship"]
    assert run.ticks >= 650, run.ticks
    for k in ("ev:ShotFiredEvent", "ev:UnitHitEvent", "ev:ThrowEvent",
              "ev:UnitKilledEvent", "converted_unit_ticks",
              "exposed_unit_ticks", "status_unit_ticks", "dead_unit_ticks",
              "held_intent_unit_ticks", "bound_mag_unit_ticks"):
        assert run.cov[k] > 0, (k, dict(run.cov))


def test_identity_scripted_arena_every_tick(identity_runs):
    """Property (c), the verb half: the same identity on every tick of the
    scripted arena (every grenade kind thrown, a C4 planted, melee, shove, a
    magazine emptied and reloaded by hand, the flashlight toggled, a
    knockdown). Breaks as (c) does, on a column only a verb writes (a throw
    or plant in progress, a pending latch, a magazine, a cool-down)."""
    run = identity_runs["scripted"]
    assert run.ticks >= 400, run.ticks
    for k in ("ev:ShotFiredEvent", "ev:UnitHitEvent", "ev:ThrowEvent",
              "ev:ReloadEvent", "ev:PlantEvent", "ev:MeleeEvent",
              "ev:ShoveEvent", "status_unit_ticks", "bound_mag_unit_ticks"):
        assert run.cov[k] > 0, (k, dict(run.cov))


# ---------------------------------------------------------------------------
# d: the batch
# ---------------------------------------------------------------------------
def test_a_batch_env_is_its_sim_alone(identity_runs):
    """Property (d): three different sims (ship_demo's match, the scripted
    arena, a two-unit sim) pack as a batch to (3, U, W) with U the largest;
    env i's rows equal env i packed alone at that U, a padding row is all
    zero with ``valid`` 0, and every env unpacks back to its own units'
    state. Breaks if a batch shares or shifts rows between envs, pads with
    anything but zeros, or marks a padding row valid."""
    from simulation.unit import Unit
    small = [Unit("a", x=5, y=5, team=0), Unit("b", x=9, y=5, team=1)]
    for i, u in enumerate(small):
        u.id = 1000 + i
    lists = [identity_runs["ship_sim"].units,
             identity_runs["scripted_sim"].units, small]
    U = max(len(us) for us in lists)
    assert len({len(us) for us in lists}) > 1   # padding is exercised
    batch = unit_store.pack_batch(lists)
    lay = unit_store.layout()
    assert batch.shape == (3, U, lay.width)
    for e, us in enumerate(lists):
        alone = unit_store.pack(us, U)
        assert np.array_equal(batch[e], alone), e
        pad = batch[e, len(us):]
        assert not pad.any(), e
        assert (batch[e, :len(us), lay.valid] == 1).all(), e
        blank = unit_store.blank_units(len(us))
        unit_store.unpack(batch[e], blank)
        for u, b in zip(us, blank):
            assert unit_schema.digest_record(b) == \
                unit_schema.digest_record(u), (e, u.id)


# ---------------------------------------------------------------------------
# e: the refusals
# ---------------------------------------------------------------------------
def _two():
    from simulation.unit import Unit
    us = [Unit("a", x=5, y=5, team=0), Unit("b", x=9, y=5, team=1)]
    us[0].id, us[1].id = 3, 7
    return us


def test_the_bridge_refuses_what_a_row_cannot_hold():
    """Property (e): pack refuses more units than rows, an id order that is
    not ascending (the laws' order) and a value outside int64; unpack
    refuses a padding row under a unit, a valid row with no unit and
    another unit's row. Breaks if a refusal is dropped (a store silently
    truncated, reordered or wrapped)."""
    us = _two()
    with pytest.raises(ValueError, match="do not fit"):
        unit_store.pack(us, 1)
    with pytest.raises(ValueError, match="ascending id"):
        unit_store.pack(us[::-1])
    big = _two()
    big[1].spare_mags = 2 ** 63
    with pytest.raises(ValueError, match="outside int64"):
        unit_store.pack(big)
    rows = unit_store.pack(us, 3)
    with pytest.raises(ValueError, match="padding"):
        unit_store.unpack(rows, unit_store.blank_units(3))
    with pytest.raises(ValueError, match="no unit was given"):
        unit_store.unpack(rows, unit_store.blank_units(1))
    other = _two()
    other[0].id = 4
    with pytest.raises(ValueError, match="unit 3's, not unit 4's"):
        unit_store.unpack(rows[:2], other)
