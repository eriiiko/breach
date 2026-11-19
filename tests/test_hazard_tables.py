"""#29 P6b.4 -- the timed-charge and heat-source tables and the digest export,
on the CPU (no device): the host side of tests/cuda_unit_hazards_check.py.

Each test names the property it protects and the change that must break it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
for p in (_ROOT / "src", _ROOT / "tests", _ROOT / "tools"):
    if str(p) not in sys.path:
        sys.path.append(str(p))

bp = pytest.importorskip("breach_physics")

from _hazard_scene import HazardScene  # noqa: E402
from simulation import object_tables as ot, unit_kernels as uk, unit_schema as us  # noqa: E402


@pytest.fixture(scope="module")
def scene_ticks():
    """The hazard scene stepped 90 ticks (the chain, the grenade's kill, the
    heat triggers, the door's death, the planted C4 all happen by tick 75),
    the tables' records kept at every tick."""
    sc = HazardScene(0, bp=bp)
    out = []
    for _ in range(90):
        sc.tick()
        out.append((uk.timed_charges_of(sc.sim), uk.heat_sources_of(sc.sim)))
    return sc, out


def test_the_tables_hold_every_value_the_sweeps_move(scene_ticks):
    """Property: unpack(records) re-encodes to the same records, and over the
    scene every moving column of each table (the countdown, the heat latch,
    hp, alive, enabled) changes at some tick -- so the tables carry the
    runtimes' cross-tick state. Breaks if a column is dropped from
    TIMED_CHARGE_SCHEMA / HEAT_SOURCE_SCHEMA or its encode / decode stops
    being an inverse."""
    sc, recs = scene_ticks
    seen = {c: False for c in uk.TIMED_STATE + uk.HEAT_STATE}
    lt = ot.table_layout(us.TIMED_CHARGE_TABLE)
    lh = ot.table_layout(us.HEAT_SOURCE_TABLE)
    for k, (t, h) in enumerate(recs):
        assert uk._records_of(us.TIMED_CHARGE_TABLE, uk.timed_objects(t)).tolist() == t.tolist()
        assert uk._records_of(us.HEAT_SOURCE_TABLE, uk.heat_objects(h)).tolist() == h.tolist()
        if k:
            for c in uk.TIMED_STATE:
                o = lt.column(c).offset
                seen[c] |= bool((t[:, o] != recs[k - 1][0][:, o]).any())
            for c in uk.HEAT_STATE:
                o = lh.column(c).offset
                seen[c] |= bool((h[:, o] != recs[k - 1][1][:, o]).any())
    assert all(seen.values()), f"a moving column never moved (vacuous scene): {seen}"


def test_landing_writes_the_state_back_and_refuses_foreign_rows(scene_ticks):
    """Property: the device's rows land on a twin's own runtimes (the twin's
    records become the rows), and a row whose load constant is not the
    runtime's is refused by name with nothing landed. Breaks if the landing
    skips a state column or lands a row onto another runtime."""
    sc, recs = scene_ticks
    twin = HazardScene(0, bp=bp)
    t, h = recs[-1]
    uk.timed_charges_to_host(twin.sim, t)
    uk.heat_sources_to_host(twin.sim, h)
    assert uk.timed_charges_of(twin.sim).tolist() == t.tolist()
    assert uk.heat_sources_of(twin.sim).tolist() == h.tolist()
    bad = h.copy()
    bad[0, ot.table_layout(us.HEAT_SOURCE_TABLE).column("heat_q16").offset] += 1
    before = uk.heat_sources_of(twin.sim).tolist()
    with pytest.raises(uk.HazardTableMismatch):
        uk.heat_sources_to_host(twin.sim, bad)
    assert uk.heat_sources_of(twin.sim).tolist() == before


def test_entity_rows_are_the_serializers(scene_ticks):
    """Property: a timed charge's / heat source's ENTITY_SECT_V1 tail read
    from its table row is the serializer's (the free alive row, then the
    class's runtime rows: fired only on a heat charge). Breaks if
    timed_entity_rows / heat_entity_rows and runtime_digest_rows drift."""
    from simulation.entities import REGISTRY
    sc, _ = scene_ticks
    by_ord = {int(e.ordinal): e for e in sc.sim.entities}
    kinds = set()
    for rec in uk.timed_charges_of(sc.sim):
        e = by_ord[int(rec[0])]
        want = (("alive", 1),) + tuple(REGISTRY["timed_charge"].runtime_digest_rows(e))
        assert uk.timed_entity_rows(rec) == want
        kinds.add(len(want))
    for rec in uk.heat_sources_of(sc.sim):
        e = by_ord[int(rec[0])]
        want = (("alive", 1 if e.alive else 0),) + tuple(
            REGISTRY["heat_source"].runtime_digest_rows(e))
        assert uk.heat_entity_rows(rec) == want
    assert kinds == {2, 3}, "vacuous: no timer and heat charge side by side"


def _drivers():
    import level_loader
    from simulation import Simulation
    from simulation.match_session import MatchSession
    from simulation.ruleset import ContinuousRealtime

    class Plain:
        def __init__(self, name):
            self.sim = Simulation(level_loader.load(name, str(_ROOT / "levels")), seed=1,
                                  breach_physics=bp, enable_recorder=False,
                                  ruleset=ContinuousRealtime())
            self.sim.set_paused(False)

        def tick(self):
            self.sim.step()
            self.sim.set_paused(False)

    class Match:
        def __init__(self, name):
            self.s = MatchSession(level_loader.load(name, str(_ROOT / "levels")), seed=1,
                                  breach_physics=bp)
            self.sim = self.s.sim

        def tick(self):
            if not self.s.is_terminal():
                self.s.tick()
    return [("hazard scene", HazardScene(0, bp=bp)), ("airlock_demo", Plain("airlock_demo")),
            ("playground", Plain("playground")), ("fire_smoke_studio", Plain("fire_smoke_studio")),
            ("larva_studio", Plain("larva_studio")),       # arc #63 P4: broods
            ("ship_demo", Match("ship_demo"))]


def test_the_export_from_the_tables_is_the_serializers_bytes():
    """Property: EntityExport composes, from the table records alone (and the
    load-constant heads), the very ENTITY_SECT_V1 and SIGNAL_SECT_V1 bytes
    the host serializer hashes, every tick, on the hazard scene, airlock_demo,
    playground, fire_smoke_studio, larva_studio (two swarm broods, a heat
    source) and ship_demo. Breaks if an entity class's
    rows, its order, a column or the signal section's alive exclusion
    differs from entities/serialize."""
    from simulation.entities.serialize import serialize_entity_state, serialize_signal_state
    for name, d in _drivers():
        ex = uk.EntityExport(d.sim)
        for _ in range(30):
            d.tick()
            got = ex.entity_section(uk.export_records(d.sim))
            assert got == serialize_entity_state(d.sim.entities), f"{name} tick {d.sim.tick}"
            bus = uk.signals_of(d.sim) if ex.has_bus else None
            assert ex.signal_section(bus) == serialize_signal_state(d.sim._digest_signals()), \
                f"{name} tick {d.sim.tick}"


def test_a_heat_sources_wired_alive_is_served_as_the_reference_emits_it(scene_ticks):
    """Property: the bus row of h1's wired ``alive`` is emitted as the value
    the reference publishes, before and after h1 dies (the reference reads
    the source's EntityInstance from _entity_by_ordinal, so it stays 1: a
    reported finding, not changed). Breaks if the device emit and the host's
    _signal_emit disagree (a fix of the reference without the device, or a
    device that reads the runtime's alive)."""
    sc, _ = scene_ticks
    sim = sc.sim
    assert not sc.byid["h1"].alive, "vacuous: h1 never died"
    rows = uk.signal_rows(sim)
    o = int(sc.byid["h1"].ordinal)
    k = [i for i, r in enumerate(rows) if r.ordinal == o and r.name == "alive"][0]
    assert rows[k].emit == "constant"
    assert int(sim._signal_bus.pub[k]) == rows[k].emit_ref


def test_the_chain_bound_counts_the_sources_that_can_append_a_disc(scene_ticks):
    """Property: heat_chain_bound is the log's discs plus the LIVING sources
    whose death payload has radius > 0 (a dead source or one with no / a
    radius-0 payload never appends). Breaks if the bound counts a source that
    cannot append (too strict) or misses one (unsafe)."""
    sc, recs = scene_ticks
    h0 = recs[0][1]
    objs = uk.heat_objects(h0)
    want = sum(1 for h in objs if h.alive and h.death_payload is not None
               and int(h.death_payload.radius) > 0)
    assert uk.heat_chain_bound(h0, 3) == 3 + want
    assert 0 < want < len(objs), "vacuous: every source or none can append"
    end = recs[-1][1]
    assert uk.heat_chain_bound(end, 0) < want, "the dead appended no more"
