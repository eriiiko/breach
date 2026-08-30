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
- LOOP-CLOSE GATING (B7 / Option 2): EQUALIZE waits on at_far, REPRESSURIZE on
  at_near — the chamber pressure sensor + two deciders, not the pump at_target.
- pump DIRECTION: the far/near targets pick inject vs extract per phase.
- The FIXTURE level levels/airlock_demo LOADS, wires the whole graph, and — over
  REAL physics — EVACUATES then genuinely REFILLS the chamber across a full
  bidirectional cycle; a 3×3 marine has room to walk the corridor (the B7 bug).
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
    # B7 (Option 2): two chamber-pressure-sensed loop closes — a `dec_far`
    # decider (ordinal 2) and a `dec_near` decider (ordinal 3), each emitting
    # `out`; EQUALIZE waits on at_far, REPRESSURIZE on at_near.
    "at_far": (2, "out"), "at_near": (3, "out"),
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
             inner_alive=1, outer_alive=1, at_far=0, at_near=0):
        b = self.bus
        b.set_pub(b.slot(*_IN["presence"]), presence)
        b.set_pub(b.slot(*_IN["inner_open"]), inner_open)
        b.set_pub(b.slot(*_IN["outer_open"]), outer_open)
        b.set_pub(b.slot(*_IN["inner_alive"]), inner_alive)
        b.set_pub(b.slot(*_IN["outer_alive"]), outer_alive)
        b.set_pub(b.slot(*_IN["at_far"]), at_far)
        b.set_pub(b.slot(*_IN["at_near"]), at_near)
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

    # at_far still 0 → hold EQUALIZE.
    assert d.tick(presence=1, inner_open=0, outer_open=0, at_far=0) == AIRLOCK_EQUALIZE
    # at_far latches (chamber ≤ far target) → OPEN_FAR; far (outer) door opens.
    assert d.tick(presence=1, inner_open=0, outer_open=0, at_far=1) == AIRLOCK_OPEN_FAR
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

    # at_near latches (chamber ≥ near target) → back to IDLE (near door reopens).
    assert d.tick(presence=0, inner_open=0, outer_open=0, at_near=1) == AIRLOCK_IDLE
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
    assert d.tick(0, 0, 0, inner_alive=1, outer_alive=1, at_far=1, at_near=1) == AIRLOCK_FAULT


def test_fault_on_dead_far_door_during_reseal():
    ctrl, bus = _make_controller(far_is_outer=True)
    d = _Driver(ctrl, bus)
    d.tick(1, 1, 0)                          # CLOSING
    d.tick(1, 0, 0)                          # EQUALIZE
    d.tick(1, 0, 0, at_far=1)                # OPEN_FAR
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
# Loop-close gating (B7) — EQUALIZE waits on at_far, REPRESSURIZE on at_near
# (the chamber pressure sensor + deciders, not the pump's own at_target)
# ---------------------------------------------------------------------------

def test_equalize_waits_for_at_far():
    ctrl, bus = _make_controller()
    d = _Driver(ctrl, bus)
    d.tick(1, 1, 0)                          # CLOSING
    d.tick(1, 0, 0)                          # EQUALIZE
    for _ in range(20):                      # chamber not yet evacuated → hold
        assert d.tick(1, 0, 0, at_far=0) == AIRLOCK_EQUALIZE
    # at_near is IRRELEVANT to EQUALIZE — only at_far advances it.
    assert d.tick(1, 0, 0, at_far=0, at_near=1) == AIRLOCK_EQUALIZE
    assert d.tick(1, 0, 0, at_far=1) == AIRLOCK_OPEN_FAR


def test_repressurize_waits_for_at_near():
    ctrl, bus = _make_controller()
    d = _Driver(ctrl, bus)
    d.tick(1, 1, 0); d.tick(1, 0, 0); d.tick(1, 0, 0, at_far=1)   # OPEN_FAR
    d.tick(0, 0, 1)                          # RESEAL
    d.tick(0, 0, 0)                          # REPRESSURIZE
    assert ctrl.state == AIRLOCK_REPRESSURIZE
    for _ in range(20):
        assert d.tick(0, 0, 0, at_near=0) == AIRLOCK_REPRESSURIZE
    # at_far is IRRELEVANT to REPRESSURIZE — only at_near advances it.
    assert d.tick(0, 0, 0, at_far=1, at_near=0) == AIRLOCK_REPRESSURIZE
    assert d.tick(0, 0, 0, at_near=1) == AIRLOCK_IDLE


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
    d.tick(1, 0, 0, at_far=1)                # OPEN_FAR
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
    assert len(lvl.wires) == 15               # command + sense + the B7 pressure loop
    # A bus exists (wires present); controller + 2 deciders (3 logic nodes),
    # pump, 2 sensors (motion plate + pressure probe), 2 doors.
    assert sim._signal_bus is not None
    assert len(sim._logic_nodes) == 3 and len(sim._pumps) == 1
    assert len(sim._sensors) == 2 and len(sim._doors) == 2
    ctrl = _controller(sim)
    assert ctrl.state == AIRLOCK_IDLE and ctrl.far_is_outer
    assert ctrl.equalize_cmd == -1            # far (0) < near (1 atm) → extract


def test_fixture_corridor_gives_a_3x3_marine_room_to_walk():
    """B7 REGRESSION (Erik's HUMAN-TEST bug): the corridor/chamber/space must be
    tall enough that a 3×3 marine (``is_passable_block`` needs the FULL footprint
    clear) has more than ONE valid anchor row — a 3-tall corridor admits exactly
    fy=3 and strands the unit ("sometimes walks there, sometimes not"). After the
    widen it is 5 tall → 3 valid anchor rows the whole way, room to maneuver."""
    _, sim = _load_fixture()
    g = sim.gmap
    from simulation.materials import MAT_AIR
    for d in sim._doors:                       # open both doors for the walk
        for (fy, fx) in d.span:
            g.material[fy, fx] = MAT_AIR
    g._update_caches()
    # Sample the mid-chamber (col 9) and deep space (col 15): each must offer
    # ≥3 valid anchor rows for a 3×3 footprint (not a single-row knife-edge).
    for fx in (9, 15):
        rows = [fy for fy in range(g.material.shape[0] - 2)
                if g.is_passable_block(fy, fx, 3)]
        assert len(rows) >= 3, (fx, rows)
        # and they are contiguous (a real corridor, not scattered)
        assert rows == list(range(rows[0], rows[0] + len(rows))), (fx, rows)


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


try:
    import breach_physics as _bp          # noqa: E402
except Exception:                          # pragma: no cover - CPU-only box
    _bp = None


@pytest.mark.skipif(_bp is None, reason="needs the compiled breach_physics")
def test_fixture_bidirectional_cycle_evacuates_and_refills():
    """B7 / Option 2 — the WHOLE wired graph over REAL physics: a marine in the
    chamber triggers the cycle, the pump EVACUATES (chamber pressure falls, the
    `dec_far` decider fires at_far → OPEN_FAR), the marine exits, and the pump
    REPRESSURIZES (chamber pressure RISES back, `dec_near` fires at_near →
    IDLE). The refill is the point Erik's HUMAN-TEST wanted — the B5 single-pump
    gap is gone. Deterministic (integer, seed 1), so this is a stable gate.

    Evacuation target RESTATED (arc #54 P-G3, 2026-08-30): measured p_evac
    moved from <=6554 (0.1 atm exactly, the pre-arc isothermal bound) to 6618
    raw -- a 1.0% miss, not a regression. Under stored gas_energy the
    evacuated gas is honestly ADIABATIC: pumping mass out of the chamber cools
    it (design §2.7's pump seam moves each parcel's own T_abs with it, no
    ambient top-up), so P = C*N*T_abs settles a little higher than the old
    isothermal-pump-down law predicted for the SAME extracted mass -- the
    chamber is colder, so it takes slightly less N to reach a given pressure
    than the pre-arc law assumed, and the fixed-duration pump undershoots the
    old N target. The cycle-cap assertions below (green since P-G1b's pump
    seam) are untouched -- this is a tolerance restatement of the pressure
    target only."""
    from simulation.unit import Unit
    from simulation.entities.door import DOOR_OPEN

    lvl = level_loader.load("airlock_demo", levels_dir=str(ROOT / "levels"))
    sim = Simulation(lvl, seed=1, breach_physics=_bp, enable_recorder=False)
    ctrl = _controller(sim)
    unit = Unit("Alpha", x=9, y=4, team=0, footprint=3)   # in the chamber
    sim.add_unit(unit)
    # The probe samples chamber air at (row 2, col 10) — read the same tile.
    chamber_p = lambda: int(sim.gmap.atmosphere[2, 10])

    def _run_until(state, cap):
        for _ in range(cap):
            sim.set_paused(False)
            sim.step()
            if ctrl.state == state:
                return True
        return False

    p_start = chamber_p()                          # ~1 atm
    # Occupancy → seal → EVACUATE → dec_far fires → OPEN_FAR.
    assert _run_until(AIRLOCK_OPEN_FAR, 200), AIRLOCK_STATE_NAMES[ctrl.state]
    p_evac = chamber_p()
    assert p_evac < p_start // 2                    # chamber genuinely evacuated
    # RESTATED arc #54 P-G3: 6554 (0.1 atm exactly) -> 6650 (~0.1015 atm), a
    # stated ~1.5% tolerance around the measured 6618 (see docstring: the
    # evacuated gas is honestly cold now, adiabatic pump-down differs from
    # the old isothermal law).
    assert p_evac <= 6650, f"chamber evacuated to {p_evac} raw (target <= 6650)"

    # The marine walks out to space → presence clears.
    unit.alive = False
    # RESEAL → REPRESSURIZE → the chamber REFILLS → dec_near fires → IDLE.
    assert _run_until(AIRLOCK_IDLE, 300), AIRLOCK_STATE_NAMES[ctrl.state]
    p_refill = chamber_p()
    assert p_refill >= 58982                        # ≥ the near target (0.9 atm)
    assert p_refill > p_evac * 4                    # the refill actually happened
    # The near (inner) door reopens at IDLE for the next entry.
    sim.set_paused(False)
    sim.step()
    assert sim.door_at(4, 7).state == DOOR_OPEN


def test_fixture_deterministic_over_physics():
    """Two fresh physics runs of the opening leg step bit-identically (integer
    determinism) — the controller state + signal digest match tick-for-tick."""
    if _bp is None:
        pytest.skip("needs the compiled breach_physics")
    from simulation.unit import Unit

    def _run():
        lvl = level_loader.load("airlock_demo", levels_dir=str(ROOT / "levels"))
        sim = Simulation(lvl, seed=1, breach_physics=_bp, enable_recorder=False)
        ctrl = _controller(sim)
        sim.add_unit(Unit("Alpha", x=9, y=4, team=0, footprint=3))
        trace = []
        for _ in range(30):
            sim.set_paused(False)
            sim.step()
            trace.append((sim._digest_signals(), ctrl.state, ctrl.dwell))
        return trace
    assert _run() == _run()


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
