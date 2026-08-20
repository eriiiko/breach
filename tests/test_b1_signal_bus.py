"""B1 — SignalBus + [[wire]] format + slot-9e logic + accessor seam.

Arc B impl doc (docs/arc_b_impl_2026-07-21.md v2), patch B1. Gates:

- DORMANCY (escalation-trigger 3): a door-present, WIRE-FREE level's multi-tick
  digest is byte-identical to the captured pre-B1 baseline (the fragile case —
  an entity-free fixture would trivially pass). Frozen constant below.
- SignalBus construction / gating (built only when wires exist, D1; alive
  excluded from __signals__).
- [[wire]] parse + every §1b validation branch.
- level_lib [[wire]] managed-block byte-stable round-trip.
- A 2-door wire scenario (door_A.is_open → door_B.close): is_open emit + the
  per-door wire-drive + the exact flip tick (§2c) + __signals__ population.
- The §5a EntityFieldAccessor sample/area against a hand-built gmap.

Fixture levels are programmatic LevelData (the A5/A6 idiom) or synthetic
EntityInstance lists — no repo level is mutated, no golden moves.

Run:
    conda run -n data python -m pytest tests/test_b1_signal_bus.py -q
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import level_loader  # noqa: E402
from level_loader import (  # noqa: E402
    EntityInstance, LevelData, SpawnEntry, Wire, WireSpec, _parse_wires,
)
from level_lib import (MANAGED_FAMILIES, format_wire_lines,  # noqa: E402
                       write_managed_blocks)
from simulation import Simulation  # noqa: E402
from simulation.entities import door as door_mod  # noqa: E402
from simulation.entities.serialize import entity_carrier  # noqa: E402
from simulation.gamemap import GameMap, N_GASES  # noqa: E402
from simulation.gases import SMOKE, O2  # noqa: E402
from simulation.sensor_accessor import (  # noqa: E402
    Channel, EntityFieldAccessor, SiteIndex, build_site_index,
)
from simulation.signal_bus import SignalBus, build_signal_bus  # noqa: E402

CLOSED, OPEN = door_mod.DOOR_CLOSED, door_mod.DOOR_OPEN


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _door_inst(eid, ordinal, x, y, orientation="v", length_m=1.0,
               initial_state="closed", tags=()):
    fields = {f.name: f.default for f in door_mod.door.FIELDS}
    fields.update(x=x, y=y, orientation=orientation, length_m=length_m,
                  initial_state=initial_state)
    return EntityInstance(id=eid, class_name="door", ordinal=ordinal,
                          tags=tuple(tags), fields=fields)


def _level(tm, entities=(), wires=(), name="b1_fix", version="1",
           tile_size_m=1.0, **kw):
    return LevelData(name=name, version=version, path=Path("."), tilemap=tm,
                     tile_size_m=tile_size_m, diffuse_path=Path("."),
                     entities=list(entities), wires=list(wires), **kw)


def _split_box_tm(h=12, w=12, wall_x=6, gap_rows=()):
    tm = np.ones((h, w), dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = 4
    tm[1:h - 1, wall_x] = 1
    for r in gap_rows:
        tm[r, wall_x] = 4
    return tm


def _step(sim, n=1):
    for _ in range(n):
        sim.set_paused(False)
        sim.step()


# ---------------------------------------------------------------------------
# DORMANCY — the escalation-trigger-3 gate (door-present, wire-free)
# ---------------------------------------------------------------------------
# Captured from the pre-B1 code path on door_test (4 doors, ZERO wires),
# physics=None so the constant is deterministic and machine-independent. An
# O-key latch toggle at ticks 1/4 exercises the door sweep + entity carrier +
# signal path (the exact B1 surface). If B1 (or any later patch) perturbs a
# wire-free level's bytes, this fails — new logic MUST stay dormant (§8).
DOORTEST_NOPHYS_TRAJ_DIGEST = \
    "5d944aa8b085fa24a100575a1292196058f15953e0c0726f95342650cb685d8b"


def test_dormancy_door_present_wire_free_digest_byte_identical():
    from field_ab_harness import _capture_unit_state, _sim_entities, UNIT_DIGEST_KEY
    from simulation.entities.serialize import ENTITY_DIGEST_KEY
    from field_digest import DIGEST_FIELDS, trajectory_digest

    lvl = level_loader.load("door_test", levels_dir=str(ROOT / "levels"))
    sim = Simulation(lvl, seed=42, breach_physics=None, enable_recorder=False)
    # A wire-free level builds NO bus — the dormancy pin (D1).
    assert sim._signal_bus is None
    assert sim._digest_signals() == ()

    names = {n for n, _ in DIGEST_FIELDS}
    traj = []
    for t in range(8):
        if t == 1:
            sim.door_at(14, 8).want_open = True
        if t == 4:
            sim.door_at(14, 8).want_open = False
        _step(sim)
        snap = {n: np.copy(getattr(sim.gmap, n)) for n in names
                if hasattr(sim.gmap, n)}
        snap[UNIT_DIGEST_KEY] = _capture_unit_state(sim)
        snap[ENTITY_DIGEST_KEY] = entity_carrier(_sim_entities(sim))
        # A door-only level emits NO is_open into __signals__ (no wire ⇒ no
        # slot ⇒ no bus, D1): the carrier's signals stay ().
        assert snap[ENTITY_DIGEST_KEY]["signals"] == ()
        traj.append(snap)
    assert trajectory_digest(traj) == DOORTEST_NOPHYS_TRAJ_DIGEST


# ---------------------------------------------------------------------------
# SignalBus — construction / gating / digest_rows (D1, alive-exclusion)
# ---------------------------------------------------------------------------

def test_bus_not_built_without_wires():
    assert build_signal_bus(_level(_split_box_tm())) is None
    # Explicitly empty wires (the Arc-A shape) → still no bus.
    assert build_signal_bus(_level(_split_box_tm(), wires=())) is None


def test_bus_built_from_wire_sources_sorted():
    wires = [
        Wire(2, "is_open", 0, "close", "held"),
        Wire(0, "is_open", 1, "open", "held"),
        Wire(0, "alive", 1, "open", "held"),
    ]
    bus = build_signal_bus(_level(_split_box_tm(), wires=wires))
    assert isinstance(bus, SignalBus)
    # Slots are the DISTINCT wire sources, ordered (ordinal, name).
    assert bus.slots == ((0, "alive"), (0, "is_open"), (2, "is_open"))
    # digest_rows excludes `alive` (hashed only as the __entity__ row, A4 c7).
    bus.set_pub(bus.slot(0, "alive"), 1)
    bus.set_pub(bus.slot(0, "is_open"), 1)
    bus.set_pub(bus.slot(2, "is_open"), 0)
    assert bus.digest_rows() == ((0, "is_open", 1), (2, "is_open", 0))


def test_bus_swap_node_signals_noop_in_b1():
    bus = SignalBus([(0, "is_open")])     # no node slots
    bus.set_pub(0, 1)
    bus.stg[0] = 99
    bus.swap_node_signals()               # B1: sensor/is_open slots not swapped
    assert int(bus.pub[0]) == 1


# ---------------------------------------------------------------------------
# [[wire]] parse + validation (§1b) — via _parse_wires on synthetic entities
# ---------------------------------------------------------------------------

def _entities_two_doors():
    return [_door_inst("d0", 0, x=6, y=3), _door_inst("d1", 1, x=6, y=8)]


def _wires_raw(*pairs):
    return {"wire": [{"from": f, "to": t} for f, t in pairs]}


def test_wire_resolves_source_and_target_ordinals():
    ents = _entities_two_doors()
    wires, specs = _parse_wires(
        _wires_raw(("d0.is_open", "d1.close")), "L.toml", ents, [])
    assert wires == [Wire(0, "is_open", 1, "close", "held")]
    assert specs == [WireSpec("d0.is_open", "d1.close")]


def test_wire_tag_fanout_ordinal_order():
    a = _door_inst("d_a", 0, x=6, y=3, tags=("bank",))
    src = _door_inst("src", 1, x=6, y=8)
    b = _door_inst("d_b", 2, x=6, y=10, tags=("bank",))
    wires, _ = _parse_wires(
        _wires_raw(("src.is_open", "tag:bank.close")),
        "L.toml", [a, src, b], [])
    # One Wire per member, expanded in ORDINAL order (0 then 2).
    assert wires == [Wire(1, "is_open", 0, "close", "held"),
                     Wire(1, "is_open", 2, "close", "held")]


def test_wire_source_signal_must_exist_hard_error():
    ents = _entities_two_doors()
    with pytest.raises(ValueError, match="no signal 'bogus'"):
        _parse_wires(_wires_raw(("d0.bogus", "d1.close")), "L.toml", ents, [])


def test_wire_dangling_source_id_warns_and_drops():
    ents = _entities_two_doors()
    with pytest.warns(UserWarning, match="dangling wire dropped"):
        wires, specs = _parse_wires(
            _wires_raw(("ghost.is_open", "d1.close")), "L.toml", ents, [])
    assert wires == []                    # dropped
    assert len(specs) == 1                # authored spec preserved (round-trip)


def test_wire_dangling_target_id_warns_and_drops():
    ents = _entities_two_doors()
    with pytest.warns(UserWarning, match="dangling wire dropped"):
        wires, _ = _parse_wires(
            _wires_raw(("d0.is_open", "ghost.close")), "L.toml", ents, [])
    assert wires == []


def test_wire_bad_target_input_name_hard_error():
    ents = _entities_two_doors()
    with pytest.raises(ValueError, match="no input 'bogus'"):
        _parse_wires(_wires_raw(("d0.is_open", "d1.bogus")), "L.toml", ents, [])


def test_wire_tag_member_lacking_input_hard_error():
    # `button` declares no inputs → a tag fan-out to `.close` hard-errors,
    # naming the member (no silent partial fan-out, §1b).
    door = _door_inst("d0", 0, x=6, y=3)
    btn = EntityInstance(id="b0", class_name="button", ordinal=1,
                         tags=("grp",), fields={"x": 1, "y": 1})
    with pytest.raises(ValueError, match=r"tag member 'b0'.*no input 'close'"):
        _parse_wires(_wires_raw(("d0.is_open", "tag:grp.close")),
                     "L.toml", [door, btn], [])


def test_wire_unit_rejection_both_ends_hard_error():
    ents = _entities_two_doors()
    spawns = [SpawnEntry(name="marine_1", team=0, x=1, y=1)]
    with pytest.raises(ValueError, match="units are NOT entities"):
        _parse_wires(_wires_raw(("marine_1.is_open", "d1.close")),
                     "L.toml", ents, spawns)
    with pytest.raises(ValueError, match="units are NOT entities"):
        _parse_wires(_wires_raw(("d0.is_open", "marine_1.close")),
                     "L.toml", ents, spawns)


def test_wire_malformed_endpoint_hard_error():
    ents = _entities_two_doors()
    with pytest.raises(ValueError, match="not dotted"):
        _parse_wires(_wires_raw(("d0", "d1.close")), "L.toml", ents, [])
    with pytest.raises(ValueError, match="non-empty"):
        _parse_wires({"wire": [{"to": "d1.close"}]}, "L.toml", ents, [])


def test_wire_many_wire_input_accepts_multiple_no_arity_error():
    # door `close` is OR/held (many-wire): two wires into it is LEGAL — the
    # generic single-arity check (§2d) is a no-op for many-wire modes in B1.
    ents = _entities_two_doors()
    wires, _ = _parse_wires(
        _wires_raw(("d0.is_open", "d1.close"), ("d1.is_open", "d1.close")),
        "L.toml", ents, [])
    assert len(wires) == 2


# ---------------------------------------------------------------------------
# level_lib — [[wire]] managed family + byte-stable round-trip
# ---------------------------------------------------------------------------

def test_wire_family_registered():
    assert "wire" in MANAGED_FAMILIES
    assert MANAGED_FAMILIES["wire"].array is True


def test_wire_round_trip_byte_stable(tmp_path):
    body = ('[[wire]]\n'
            'from = "door_a.is_open"\n'
            'to = "door_b.close"\n'
            '\n'
            '[[wire]]\n'
            'from = "sensor_1.pressure"\n'
            'to = "tag:blast_doors.close"\n')
    toml = tmp_path / "level.toml"
    header = ('version = "2"\nname = "T"\ntilemap = "t.csv"\n'
              'tile_size_m = 0.333\ndiffuse = "d.png"\n\n')
    before = (header + body).encode("utf-8")
    toml.write_bytes(before)
    specs = [WireSpec("door_a.is_open", "door_b.close"),
             WireSpec("sensor_1.pressure", "tag:blast_doors.close")]
    # Load -> format -> write lands byte-identically (authored from/to verbatim).
    write_managed_blocks(toml, {"wire": lambda nl: format_wire_lines(specs, nl)})
    assert toml.read_bytes() == before


# ---------------------------------------------------------------------------
# The 2-door wire scenario — is_open emit + wire-drive + exact flip tick (§2c)
# ---------------------------------------------------------------------------

def test_two_door_wire_drive_and_latency_and_signals():
    tm = _split_box_tm(gap_rows=(3, 8))
    d_a = _door_inst("door_A", 0, x=6, y=3, initial_state="closed")
    d_b = _door_inst("door_B", 1, x=6, y=8, initial_state="open")
    wire = Wire(0, "is_open", 1, "close", "held")   # door_A.is_open → door_B.close
    sim = Simulation(_level(tm, [d_a, d_b], [wire]), seed=1,
                     breach_physics=None, enable_recorder=False)
    a = sim.door_at(3, 6)
    b = sim.door_at(8, 6)
    assert a.state == CLOSED and b.state == OPEN

    # The bus exists (wires present) and carries exactly door_A's is_open slot.
    assert sim._signal_bus is not None
    assert sim._signal_bus.slots == ((0, "is_open"),)

    # Drive door_A open via its free latch (it has no incoming wire, D3).
    a.want_open = True

    # Tick 0: (a) emits door_A.is_open = 0 (A still closed at emit); (c) close
    # inactive → door_B retains its open latch; (d) door_A flips OPEN, B stays.
    _step(sim)
    assert a.state == OPEN and b.state == OPEN          # B has NOT followed yet
    assert sim._digest_signals() == ((0, "is_open", 0),)  # emit was pre-flip

    # Tick 1: (a) emits door_A.is_open = 1; (c) door_B.close active → want_open
    # False; (d) door_B closes. The 1-tick door-state latency (§2c): B's flip
    # is EXACTLY one tick after A's is_open went high.
    _step(sim)
    assert a.state == OPEN and b.state == CLOSED
    assert sim._digest_signals() == ((0, "is_open", 1),)

    # __signals__ rides the get_state carrier (present only because a bus exists).
    st = sim.get_state()
    assert st.entity_state["signals"] == ((0, "is_open", 1),)

    # Held/close-priority latch: door_A staying open holds door_B shut (close
    # remains active); nothing reopens B.
    _step(sim, 3)
    assert b.state == CLOSED


def test_unwired_door_keeps_arc_a_latch():
    # A door that is NEITHER a wire source nor a target of an open/close wire
    # keeps its Arc-A want_open latch (the O-key path), untouched by 9e(c).
    tm = _split_box_tm(gap_rows=(3, 8))
    d_a = _door_inst("door_A", 0, x=6, y=3, initial_state="closed")
    d_b = _door_inst("door_B", 1, x=6, y=8, initial_state="closed")
    # A wire that references door_A.is_open (so a bus exists) but drives d_b —
    # d_a itself has no INCOMING open/close wire, so its latch stays free.
    wire = Wire(0, "is_open", 1, "close", "held")
    sim = Simulation(_level(tm, [d_a, d_b], [wire]), seed=1,
                     breach_physics=None, enable_recorder=False)
    a = sim.door_at(3, 6)
    a.want_open = True
    _step(sim)
    assert a.state == OPEN                # latch honored despite the bus


# ---------------------------------------------------------------------------
# §5a accessor seam — sample / area against a hand-built gmap
# ---------------------------------------------------------------------------

class _FakeGmap:
    """Minimal mirror exposing only the fields the accessor gathers."""
    def __init__(self, h=4, w=4):
        self.atmosphere = np.zeros((h, w), dtype=np.int32)
        self.gas = np.zeros((N_GASES, h, w), dtype=np.int32)
        self.water_depth = np.zeros((h, w), dtype=np.int32)
        self.temperature = np.zeros((h, w), dtype=np.int32)
        self.fire = np.zeros((h, w), dtype=np.int32)
        self.solid = np.zeros((h, w), dtype=bool)


def test_accessor_sample_reads_mirror_no_dequantize():
    g = _FakeGmap()
    g.atmosphere[1, 2] = 65536            # 1.0 atm in Q16.16 — returned RAW
    g.gas[SMOKE][1, 2] = 4242
    g.gas[O2][1, 2] = 13107
    g.temperature[3, 3] = 777
    g.fire[0, 0] = 55
    g.water_depth[2, 1] = 999
    g.solid[3, 3] = True
    acc = EntityFieldAccessor(g)
    assert acc.sample(Channel.PRESSURE, 1, 2) == 65536   # no /65536 dequantize
    assert acc.sample(Channel.SMOKE, 1, 2) == 4242
    assert acc.sample(Channel.O2, 1, 2) == 13107
    assert acc.sample(Channel.TEMPERATURE, 3, 3) == 777
    assert acc.sample(Channel.FIRE, 0, 0) == 55
    assert acc.sample(Channel.WATER_DEPTH, 2, 1) == 999
    assert acc.sample(Channel.SOLID, 3, 3) == 1          # bool → 0/1 int


def test_accessor_area_masks_solid_tiles():
    g = _FakeGmap()
    disc = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for t in disc:
        g.atmosphere[t] = 100
    g.solid[1, 1] = True                  # one solid tile in the disc
    acc = EntityFieldAccessor(g)
    total, n = acc.area(Channel.PRESSURE, disc)
    assert (total, n) == (300, 3)         # solid tile excluded from sum + count
    # A destroyed wall (now non-solid) legitimately re-enters the count.
    g.solid[1, 1] = False
    assert acc.area(Channel.PRESSURE, disc) == (400, 4)


def test_build_site_index_empty_in_b1():
    idx = build_site_index([])            # no sensor classes until B3
    assert isinstance(idx, SiteIndex) and len(idx) == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
