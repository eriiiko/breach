"""B5 — the airlock_controller state machine + the E2E airlock fixture level.

Arc B impl doc (docs/arc_b_impl_2026-07-21.md v2), patch B5 (§7, D12). This is
the FEEL-ADJACENT patch: built green + deterministic here, but it does NOT merge
until Erik plays levels/airlock_demo (the HUMAN-TEST gate). Gates:

- STATE GRAPH (drive AirlockControllerRuntime directly over a SignalBus): the
  full automatic cycle IDLE→CLOSING→EQUALIZE→OPEN_FAR→RESEAL→REPRESSURIZE→IDLE,
  asserting the exact state + command outputs at each tick.
- FAULT breach-abort (D12): a door reading alive==0 during CLOSING aborts to
  FAULT (never pump into a breach), releases the doors, drops busy, and LATCHES;
  a far door reading alive==0 during RESEAL likewise aborts.
- OCCUPANCY-blocks-close STALL (the accepted v1 gap, §7): a door that never
  reports is_open==0 (a unit parked on its span) holds CLOSING forever.
- at_target GATING: EQUALIZE / REPRESSURIZE wait on the pump's at_target.
- pump DIRECTION: the far/near targets pick inject vs extract per phase.
- The FIXTURE level levels/airlock_demo LOADS, wires the whole graph, STEPS
  deterministically, and CYCLES through every state via the real SignalBus.
- DORMANCY: the B1 door-present, wire-free digest is still byte-identical.

The state-graph tests need no units and no physics (the transition function is
pure integer SignalBus I/O); the fixture-cycle test drives presence through a
controllable stand-in for the plate sensor so the whole wired graph — sensor →
controller → doors/pump, the 9e(e) swap, the digest — runs for real. The
unit-walks-in feel is Erik's HUMAN-TEST.

Run:
    conda run -n data python -m pytest tests/test_b5_airlock.py -q
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import level_loader  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation.entities import REGISTRY  # noqa: E402
from simulation.entities.actuators import (  # noqa: E402
    AIRLOCK_CLOSING, AIRLOCK_EQUALIZE, AIRLOCK_FAULT, AIRLOCK_IDLE,
    AIRLOCK_OPEN_FAR, AIRLOCK_REPRESSURIZE, AIRLOCK_RESEAL,
    AIRLOCK_STATE_NAMES,
)
from simulation.logic_nodes import AirlockControllerRuntime  # noqa: E402
from simulation.signal_bus import SignalBus  # noqa: E402


# ---------------------------------------------------------------------------
# A controller wired to a synthetic bus — the state-graph harness
# ---------------------------------------------------------------------------
# Distinct (ordinal, name) source slots for each wired input, plus the seven
# command outputs on the controller's own ordinal (4). node_slots = the outputs
# so swap_node_signals mirrors stg→pub, exactly like 9e(e).
_IN = {
    "presence": (10, "value"),
    "inner_open": (0, "is_open"), "inner_alive": (0, "alive"),
    "outer_open": (1, "is_open"), "outer_alive": (1, "alive"),
    "at_target": (2, "at_target"),
}
_OUT = ("inner_close", "inner_open_cmd", "outer_close", "outer_open_cmd",
        "pump_inject", "pump_extract", "busy")
_CTRL_ORD = 4


def _make_controller(far_is_outer=True, equalize_cmd=-1):
    slots = list(_IN.values()) + [(_CTRL_ORD, s) for s in _OUT]
    slots.sort(key=lambda s: (s[0], s[1]))
    out_keys = {(_CTRL_ORD, s) for s in _OUT}
    node_slots = tuple(i for i, s in enumerate(slots) if s in out_keys)
    bus = SignalBus(slots, node_slots=node_slots)
    inst = types.SimpleNamespace(ordinal=_CTRL_ORD, id="ctrl",
                                 class_name="airlock_controller", fields={})
    in_slots = {name: bus.slot(*key) for name, key in _IN.items()}
    out_slots = {s: bus.slot(_CTRL_ORD, s) for s in _OUT}
    ctrl = AirlockControllerRuntime(inst, far_is_outer, equalize_cmd,
                                    in_slots, out_slots)
    return ctrl, bus


class _Driver:
    """Feed the controller's inputs on ``pub``, evaluate, swap, read outputs."""

    def __init__(self, ctrl, bus):
        self.ctrl, self.bus = ctrl, bus

    def tick(self, presence, inner_open, outer_open,
             inner_alive=1, outer_alive=1, at_target=0):
        b = self.bus
        b.set_pub(b.slot(*_IN["presence"]), presence)
        b.set_pub(b.slot(*_IN["inner_open"]), inner_open)
        b.set_pub(b.slot(*_IN["outer_open"]), outer_open)
        b.set_pub(b.slot(*_IN["inner_alive"]), inner_alive)
        b.set_pub(b.slot(*_IN["outer_alive"]), outer_alive)
        b.set_pub(b.slot(*_IN["at_target"]), at_target)
        self.ctrl.evaluate(b)
        b.swap_node_signals()               # 9e(e): stg → pub for node slots
        return self.ctrl.state

    def out(self, name):
        return int(self.bus.read(_CTRL_ORD, name))


# ---------------------------------------------------------------------------
# The full automatic cycle — exact state at each tick (§7)
# ---------------------------------------------------------------------------

def test_full_cycle_exact_states_and_commands():
    ctrl, bus = _make_controller(far_is_outer=True, equalize_cmd=-1)
    d = _Driver(ctrl, bus)

    # IDLE: empty chamber, near (inner) door held open for entry, far shut.
    assert d.tick(presence=0, inner_open=1, outer_open=0) == AIRLOCK_IDLE
    assert (d.out("inner_open_cmd"), d.out("outer_close"), d.out("busy")) == (1, 1, 0)

    # A unit enters → CLOSING (busy). Drive BOTH doors closed.
    assert d.tick(presence=1, inner_open=1, outer_open=0) == AIRLOCK_CLOSING
    assert (d.out("inner_close"), d.out("outer_close"), d.out("busy")) == (1, 1, 1)

    # Inner still reads open → hold CLOSING (not both sealed yet).
    assert d.tick(presence=1, inner_open=1, outer_open=0) == AIRLOCK_CLOSING

    # Both doors alive AND is_open==0 → EQUALIZE; pump EXTRACTS (far < near).
    assert d.tick(presence=1, inner_open=0, outer_open=0) == AIRLOCK_EQUALIZE
    assert (d.out("pump_extract"), d.out("pump_inject"), d.out("busy")) == (1, 0, 1)

    # at_target still 0 → hold EQUALIZE.
    assert d.tick(presence=1, inner_open=0, outer_open=0, at_target=0) == AIRLOCK_EQUALIZE
    # at_target latches → OPEN_FAR; the far (outer) door is commanded open.
    assert d.tick(presence=1, inner_open=0, outer_open=0, at_target=1) == AIRLOCK_OPEN_FAR
    assert (d.out("outer_open_cmd"), d.out("inner_close"), d.out("busy")) == (1, 1, 1)

    # Chamber still occupied → hold OPEN_FAR.
    assert d.tick(presence=1, inner_open=0, outer_open=1) == AIRLOCK_OPEN_FAR
    # Chamber clears → RESEAL (drive the far door shut again).
    assert d.tick(presence=0, inner_open=0, outer_open=1) == AIRLOCK_RESEAL
    assert (d.out("outer_close"), d.out("busy")) == (1, 1)

    # Far door still open → hold RESEAL.
    assert d.tick(presence=0, inner_open=0, outer_open=1) == AIRLOCK_RESEAL
    # Far sealed (alive AND is_open==0) → REPRESSURIZE; pump INJECTS (toward near).
    assert d.tick(presence=0, inner_open=0, outer_open=0) == AIRLOCK_REPRESSURIZE
    assert (d.out("pump_inject"), d.out("pump_extract"), d.out("busy")) == (1, 0, 1)

    # at_target latches → back to IDLE (the near door reopens).
    assert d.tick(presence=0, inner_open=0, outer_open=0, at_target=1) == AIRLOCK_IDLE
    assert (d.out("inner_open_cmd"), d.out("busy")) == (1, 0)


def test_dwell_counts_ticks_in_state():
    ctrl, bus = _make_controller()
    d = _Driver(ctrl, bus)
    d.tick(0, 1, 0)                          # IDLE, first tick
    assert ctrl.state == AIRLOCK_IDLE and ctrl.dwell == 1
    d.tick(0, 1, 0)
    assert ctrl.dwell == 2                    # held → dwell grows
    d.tick(1, 1, 0)                          # → CLOSING (transition)
    assert ctrl.state == AIRLOCK_CLOSING and ctrl.dwell == 0   # reset on entry


# ---------------------------------------------------------------------------
# FAULT breach-abort (D12) — never pump into a hole
# ---------------------------------------------------------------------------

def test_fault_on_dead_door_during_closing_and_latches():
    ctrl, bus = _make_controller()
    d = _Driver(ctrl, bus)
    d.tick(1, 1, 0)                          # IDLE → CLOSING
    assert ctrl.state == AIRLOCK_CLOSING
    # The inner door is DESTROYED: is_open reads 0 ("looks sealed") but alive==0
    # (a venting hole). Must abort to FAULT, not proceed to EQUALIZE.
    assert d.tick(presence=1, inner_open=0, outer_open=0, inner_alive=0) == AIRLOCK_FAULT
    # FAULT releases the doors to manual, drops busy, drives NO pump.
    assert (d.out("inner_close"), d.out("inner_open_cmd"), d.out("outer_close"),
            d.out("outer_open_cmd"), d.out("pump_inject"), d.out("pump_extract"),
            d.out("busy")) == (0, 0, 0, 0, 0, 0, 0)
    # LATCHED: even with everything healthy again it stays FAULT (until reset).
    assert d.tick(0, 0, 0, inner_alive=1, outer_alive=1, at_target=1) == AIRLOCK_FAULT


def test_fault_on_dead_far_door_during_reseal():
    ctrl, bus = _make_controller(far_is_outer=True)
    d = _Driver(ctrl, bus)
    d.tick(1, 1, 0)                          # CLOSING
    d.tick(1, 0, 0)                          # EQUALIZE
    d.tick(1, 0, 0, at_target=1)             # OPEN_FAR
    d.tick(0, 0, 1)                          # RESEAL
    assert ctrl.state == AIRLOCK_RESEAL
    # The far (outer) door is destroyed mid-reseal (alive==0) → FAULT, so
    # REPRESSURIZE never pumps into the breach.
    assert d.tick(0, 0, 0, outer_alive=0) == AIRLOCK_FAULT


# ---------------------------------------------------------------------------
# Occupancy-blocks-close STALL — the accepted v1 gap (§7)
# ---------------------------------------------------------------------------

def test_occupancy_blocks_close_holds_closing_forever():
    ctrl, bus = _make_controller()
    d = _Driver(ctrl, bus)
    d.tick(1, 1, 0)                          # → CLOSING
    # A unit is parked on the inner door span, so the door can never seal — it
    # keeps reading is_open==1. CLOSING is a PERMANENT stall (no timeout, §7).
    for _ in range(50):
        assert d.tick(presence=1, inner_open=1, outer_open=0) == AIRLOCK_CLOSING
    assert d.out("busy") == 1                 # still cycling, just blocked


# ---------------------------------------------------------------------------
# at_target gating — EQUALIZE / REPRESSURIZE wait on the pump
# ---------------------------------------------------------------------------

def test_equalize_waits_for_at_target():
    ctrl, bus = _make_controller()
    d = _Driver(ctrl, bus)
    d.tick(1, 1, 0)                          # CLOSING
    d.tick(1, 0, 0)                          # EQUALIZE
    for _ in range(20):                      # pump not settled → hold EQUALIZE
        assert d.tick(1, 0, 0, at_target=0) == AIRLOCK_EQUALIZE
    assert d.tick(1, 0, 0, at_target=1) == AIRLOCK_OPEN_FAR


def test_repressurize_waits_for_at_target():
    ctrl, bus = _make_controller()
    d = _Driver(ctrl, bus)
    d.tick(1, 1, 0); d.tick(1, 0, 0); d.tick(1, 0, 0, at_target=1)   # OPEN_FAR
    d.tick(0, 0, 1)                          # RESEAL
    d.tick(0, 0, 0)                          # REPRESSURIZE
    assert ctrl.state == AIRLOCK_REPRESSURIZE
    for _ in range(20):
        assert d.tick(0, 0, 0, at_target=0) == AIRLOCK_REPRESSURIZE
    assert d.tick(0, 0, 0, at_target=1) == AIRLOCK_IDLE


# ---------------------------------------------------------------------------
# Pump direction + far_door mapping
# ---------------------------------------------------------------------------

def test_pump_direction_follows_targets():
    # far > near → EQUALIZE injects (pressurize toward the far side).
    ctrl, bus = _make_controller(equalize_cmd=+1)
    d = _Driver(ctrl, bus)
    d.tick(1, 1, 0); d.tick(1, 0, 0)         # → EQUALIZE
    assert ctrl.state == AIRLOCK_EQUALIZE
    assert (d.out("pump_inject"), d.out("pump_extract")) == (1, 0)


def test_equal_targets_drive_no_pump():
    ctrl, bus = _make_controller(equalize_cmd=0)
    d = _Driver(ctrl, bus)
    d.tick(1, 1, 0); d.tick(1, 0, 0)
    assert (d.out("pump_inject"), d.out("pump_extract")) == (0, 0)


def test_far_door_inner_mapping():
    # far_door = inner: the INNER door is the one opened after EQUALIZE.
    ctrl, bus = _make_controller(far_is_outer=False, equalize_cmd=-1)
    d = _Driver(ctrl, bus)
    d.tick(1, 1, 0)                          # CLOSING
    d.tick(1, 0, 0)                          # EQUALIZE
    d.tick(1, 0, 0, at_target=1)             # OPEN_FAR
    assert ctrl.state == AIRLOCK_OPEN_FAR
    # The far side is inner → inner opens, outer stays shut.
    assert (d.out("inner_open_cmd"), d.out("outer_close")) == (1, 1)
    assert (d.out("outer_open_cmd"), d.out("inner_close")) == (0, 0)


# ---------------------------------------------------------------------------
# runtime_digest_rows — synced state + dormancy
# ---------------------------------------------------------------------------

def test_runtime_digest_rows_report_state_and_dwell():
    ctrl, bus = _make_controller()
    d = _Driver(ctrl, bus)
    d.tick(1, 1, 0)                          # CLOSING, dwell 0
    rows = REGISTRY["airlock_controller"].runtime_digest_rows(ctrl)
    assert rows == (("state", AIRLOCK_CLOSING), ("dwell", 0))


def test_bare_instance_has_no_runtime_rows_dormant():
    # A bare EntityInstance (no runtime object) raises — digests only come from
    # constructed sims, mirroring the door/filter/pump loud path (§8).
    bare = types.SimpleNamespace(id="c", ordinal=0)
    with pytest.raises(AttributeError):
        REGISTRY["airlock_controller"].runtime_digest_rows(bare)


# ===========================================================================
# The E2E fixture level — loads, wires, steps, cycles (levels/airlock_demo)
# ===========================================================================

def _load_fixture():
    lvl = level_loader.load("airlock_demo", levels_dir=str(ROOT / "levels"))
    sim = Simulation(lvl, seed=1, breach_physics=None, enable_recorder=False)
    return lvl, sim


def _controller(sim):
    return next(e for e in sim.entities
               if isinstance(e, AirlockControllerRuntime))


def test_fixture_loads_and_wires_the_whole_graph():
    lvl, sim = _load_fixture()
    assert lvl.name == "airlock_demo"
    assert len(lvl.wires) == 12               # the full command + sense graph
    # A bus exists (wires present); the controller, pump, sensor, doors built.
    assert sim._signal_bus is not None
    assert len(sim._logic_nodes) == 1 and len(sim._pumps) == 1
    assert len(sim._sensors) == 1 and len(sim._doors) == 2
    ctrl = _controller(sim)
    assert ctrl.state == AIRLOCK_IDLE and ctrl.far_is_outer
    assert ctrl.equalize_cmd == -1            # far (0) < near (1 atm) → extract


def test_fixture_idle_when_empty_and_deterministic():
    # No units in range → presence 0 → holds IDLE; two fresh runs step
    # bit-identically (the same signal digest + controller rows each tick).
    def _run():
        _, sim = _load_fixture()
        ctrl = _controller(sim)
        trace = []
        for _ in range(12):
            sim.set_paused(False)
            sim.step()
            trace.append((sim._digest_signals(), ctrl.state, ctrl.dwell))
        return ctrl.state, trace
    s0, t0 = _run()
    s1, t1 = _run()
    assert s0 == AIRLOCK_IDLE
    assert t0 == t1                           # deterministic stepping


class _Presence:
    """A controllable stand-in for the plate sensor: publishes ``n`` at 9e(a)
    so the fixture-cycle test drives presence deterministically without the
    unit-spawn pipeline. Everything else (doors, pump, controller, the wired
    SignalBus, the 9e(e) swap) runs for real."""

    def __init__(self, value_slot):
        self.value_slot = value_slot
        self.inst = types.SimpleNamespace(alive=True)
        self.n = 0

    def evaluate(self, sim):
        return self.n


def test_fixture_cycles_through_every_state():
    _, sim = _load_fixture()
    ctrl = _controller(sim)
    inner, outer = sim.door_at(4, 7), sim.door_at(4, 12)
    port = (3, 9)
    pres = _Presence(sim._sensors[0].value_slot)
    sim._sensors[0] = pres                    # drive presence deterministically

    def _until(state, cap, evacuate=False):
        for _ in range(cap):
            if evacuate:
                sim.gmap.atmosphere[port] = 0   # so the pump latches at_target
            sim.set_paused(False)
            sim.step()
            if ctrl.state == state:
                return True
        return False

    from simulation.entities.door import DOOR_OPEN, DOOR_CLOSED

    # A unit is in the chamber → both doors seal → EQUALIZE.
    pres.n = 1
    assert _until(AIRLOCK_EQUALIZE, 20), AIRLOCK_STATE_NAMES[ctrl.state]
    assert inner.state == DOOR_CLOSED and outer.state == DOOR_CLOSED
    # The pump evacuates the chamber → at_target → OPEN_FAR.
    assert _until(AIRLOCK_OPEN_FAR, 10, evacuate=True)
    # The far (outer) door opens a tick or two after OPEN_FAR commands it
    # (presence still 1 → the machine holds OPEN_FAR meanwhile).
    for _ in range(3):
        sim.set_paused(False)
        sim.step()
        if outer.state == DOOR_OPEN:
            break
    assert outer.state == DOOR_OPEN
    # The unit exits to space → RESEAL.
    pres.n = 0
    assert _until(AIRLOCK_RESEAL, 10), AIRLOCK_STATE_NAMES[ctrl.state]
    # The outer door reseals → REPRESSURIZE → IDLE (inner reopens).
    assert _until(AIRLOCK_IDLE, 20, evacuate=True), AIRLOCK_STATE_NAMES[ctrl.state]
    # Back at rest: one more step and the near (inner) door is open again.
    sim.set_paused(False)
    sim.step()
    assert inner.state == DOOR_OPEN


def test_fixture_level_lib_round_trips_byte_stable(tmp_path):
    # level_lib re-emits the fixture's managed families (spawn/entity/wire)
    # byte-for-byte — the authoring-round-trip contract (§1c).
    from level_lib import (format_entity_lines, format_spawn_lines,
                           format_wire_lines, write_managed_blocks)
    src = ROOT / "levels" / "airlock_demo" / "level.toml"
    before = src.read_bytes()
    lvl = level_loader.load("airlock_demo", levels_dir=str(ROOT / "levels"))
    dst = tmp_path / "level.toml"
    dst.write_bytes(before)
    write_managed_blocks(dst, {
        "spawn": lambda nl: format_spawn_lines(lvl.spawns, nl),
        "entity": lambda nl: format_entity_lines(lvl.entities, nl),
        "wire": lambda nl: format_wire_lines(lvl.wire_specs, nl),
    })
    assert dst.read_bytes() == before


# ---------------------------------------------------------------------------
# DORMANCY — the B1 door-present, wire-free digest is untouched (re-run)
# ---------------------------------------------------------------------------

def test_b1_dormancy_still_byte_identical():
    import test_b1_signal_bus as b1
    b1.test_dormancy_door_present_wire_free_digest_byte_identical()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
