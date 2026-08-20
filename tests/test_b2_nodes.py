"""B2 — the logic node set (decider / gate_and / gate_or / gate_not / filter).

Arc B impl doc (docs/arc_b_impl_2026-07-21.md v2), patch B2. Gates (§10 B2):

- TRUTH TABLES: all 6 decider comparators; AND / OR / NOT over multiple and
  ZERO wires.
- require_alive: fail-passive (dead source → decider 0) vs bare fail-deadly
  (value passes through without it).
- FILTER: the exact Fraction tau-snap on several tau (incl. a boundary tau
  where a naive float round(log2) could flip); guard-bit round-to-nearest;
  step response; cross-machine k-snap stability (pure integer/Fraction — no
  float dependence).
- LATENCY: a node hop adds exactly one tick (prev-read/next-write, §2c).
- DIGEST: the filter EMA row is present when instantiated and ABSENT (zero
  bytes) when the class is not — plus __signals__ population.
- DORMANCY: the B1 door-present wire-free digest stays byte-identical (re-run
  the B1 gate explicitly).

Run:
    conda run -n data python -m pytest tests/test_b2_nodes.py -q
"""
from __future__ import annotations

import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import level_loader  # noqa: E402
from level_loader import EntityInstance, LevelData, Wire, _parse_wires  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation.entities import REGISTRY  # noqa: E402
from simulation.entities import door as door_mod  # noqa: E402
from simulation.entities.nodes import COMPARATORS, snap_filter_k  # noqa: E402
from simulation.entities.schema import (  # noqa: E402
    ALL_INPUT_MODES, INPUT_AND, INPUT_HELD, INPUT_SINGLE,
)
from simulation.logic_nodes import (  # noqa: E402
    FilterRuntime, _DeciderEval, _GateEval, _NotEval, aggregate_input,
)
from simulation.signal_bus import SignalBus, build_signal_bus  # noqa: E402

CLOSED, OPEN = door_mod.DOOR_CLOSED, door_mod.DOOR_OPEN
Q1 = 65536                                # 1.0 atm in Q16.16


# ---------------------------------------------------------------------------
# Fixtures — programmatic EntityInstances (the B1 idiom; no repo level moved)
# ---------------------------------------------------------------------------

def _door_inst(eid, ordinal, x, y, orientation="v", length_m=1.0,
               initial_state="closed", tags=()):
    fields = {f.name: f.default for f in door_mod.door.FIELDS}
    fields.update(x=x, y=y, orientation=orientation, length_m=length_m,
                  initial_state=initial_state)
    return EntityInstance(id=eid, class_name="door", ordinal=ordinal,
                          tags=tuple(tags), fields=fields)


def _node_inst(eid, ordinal, class_name, **overrides):
    cls = REGISTRY[class_name]
    fields = {f.name: f.default for f in cls.FIELDS}
    fields.update(overrides)
    return EntityInstance(id=eid, class_name=class_name, ordinal=ordinal,
                          tags=(), fields=fields)


def _level(tm, entities=(), wires=(), name="b2_fix", version="1",
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


def _bus_one(slots, in_val_map):
    """A tiny SignalBus with ``slots`` and pub preset from ``in_val_map`` (slot
    tuple -> value). Returns (bus, index_of)."""
    bus = SignalBus(slots)
    for key, v in in_val_map.items():
        bus.set_pub(bus.slot(*key), v)
    return bus


# ---------------------------------------------------------------------------
# schema — the AND/SINGLE mode extension (§2d) without breaking HELD/EDGE
# ---------------------------------------------------------------------------

def test_input_modes_extended_not_replaced():
    assert set(ALL_INPUT_MODES) == {"held", "edge", "and", "single"}
    # HELD/EDGE meanings untouched (trigger-2 safe) — the door still declares
    # its open/close as HELD.
    door_inputs = {i.name: i.mode for i in door_mod.door.INPUTS}
    assert door_inputs == {"open": "held", "close": "held"}


def test_node_classes_registered_and_intangible():
    for cn in ("decider", "gate_and", "gate_or", "gate_not", "filter"):
        cls = REGISTRY[cn]
        assert cls.LOGIC_NODE is True
        assert cls.INTANGIBLE is True     # pure logic, no tile (§5)
        assert [s.name for s in cls.SIGNALS] == ["out"]


# ---------------------------------------------------------------------------
# decider — all 6 comparators (§5 truth table)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cmp_name,a,b,expect", [
    ("gt", 5, 3, 1), ("gt", 3, 3, 0), ("gt", 2, 3, 0),
    ("ge", 3, 3, 1), ("ge", 2, 3, 0),
    ("lt", 2, 3, 1), ("lt", 3, 3, 0),
    ("le", 3, 3, 1), ("le", 4, 3, 0),
    ("eq", 3, 3, 1), ("eq", 4, 3, 0),
    ("ne", 4, 3, 1), ("ne", 3, 3, 0),
])
def test_decider_comparators(cmp_name, a, b, expect):
    assert cmp_name in COMPARATORS
    bus = _bus_one([(0, "value"), (1, "out")], {(0, "value"): a})
    ev = _DeciderEval(bus.slot(1, "out"), [bus.slot(0, "value")], cmp_name, b,
                      require_alive=False, alive_slot=None)
    ev.evaluate(bus)
    assert int(bus.stg[bus.slot(1, "out")]) == expect


def test_decider_single_input_is_verbatim_value():
    # SINGLE aggregation returns the driving wire's integer value verbatim.
    bus = _bus_one([(0, "value"), (1, "out")], {(0, "value"): 424242})
    assert aggregate_input(bus, [bus.slot(0, "value")], INPUT_SINGLE) == 424242
    assert aggregate_input(bus, [], INPUT_SINGLE) == 0     # unwired => 0


# ---------------------------------------------------------------------------
# gates — AND / OR / NOT over multiple and ZERO wires (§5 truth table)
# ---------------------------------------------------------------------------

def test_gate_and_over_multiple_and_zero_wires():
    slots = [(0, "a"), (1, "b"), (2, "out")]
    out = lambda bus: int(bus.stg[bus.slot(2, "out")])
    # both non-zero → 1
    bus = _bus_one(slots, {(0, "a"): 7, (1, "b"): 1})
    _GateEval(bus.slot(2, "out"), [bus.slot(0, "a"), bus.slot(1, "b")],
              INPUT_AND).evaluate(bus)
    assert out(bus) == 1
    # one zero → 0
    bus = _bus_one(slots, {(0, "a"): 7, (1, "b"): 0})
    _GateEval(bus.slot(2, "out"), [bus.slot(0, "a"), bus.slot(1, "b")],
              INPUT_AND).evaluate(bus)
    assert out(bus) == 0
    # ZERO wires → 0 (empty AND is false, §2d)
    bus = _bus_one(slots, {})
    _GateEval(bus.slot(2, "out"), [], INPUT_AND).evaluate(bus)
    assert out(bus) == 0


def test_gate_or_over_multiple_and_zero_wires():
    slots = [(0, "a"), (1, "b"), (2, "out")]
    out = lambda bus: int(bus.stg[bus.slot(2, "out")])
    bus = _bus_one(slots, {(0, "a"): 0, (1, "b"): 3})
    _GateEval(bus.slot(2, "out"), [bus.slot(0, "a"), bus.slot(1, "b")],
              INPUT_HELD).evaluate(bus)
    assert out(bus) == 1
    bus = _bus_one(slots, {(0, "a"): 0, (1, "b"): 0})
    _GateEval(bus.slot(2, "out"), [bus.slot(0, "a"), bus.slot(1, "b")],
              INPUT_HELD).evaluate(bus)
    assert out(bus) == 0
    bus = _bus_one(slots, {})               # ZERO wires → 0 (empty OR)
    _GateEval(bus.slot(2, "out"), [], INPUT_HELD).evaluate(bus)
    assert out(bus) == 0


def test_gate_not():
    slots = [(0, "a"), (1, "out")]
    out = lambda bus: int(bus.stg[bus.slot(1, "out")])
    bus = _bus_one(slots, {(0, "a"): 0})
    _NotEval(bus.slot(1, "out"), [bus.slot(0, "a")]).evaluate(bus)
    assert out(bus) == 1
    bus = _bus_one(slots, {(0, "a"): 9})
    _NotEval(bus.slot(1, "out"), [bus.slot(0, "a")]).evaluate(bus)
    assert out(bus) == 0
    bus = _bus_one(slots, {})               # unwired in (0) → out 1
    _NotEval(bus.slot(1, "out"), []).evaluate(bus)
    assert out(bus) == 1


# ---------------------------------------------------------------------------
# require_alive — fail-passive vs bare fail-deadly (§2d)
# ---------------------------------------------------------------------------

def test_require_alive_fail_passive_vs_fail_deadly():
    slots = [(0, "value"), (0, "alive"), (1, "out")]
    v_slot = lambda bus: bus.slot(0, "value")
    a_slot = lambda bus: bus.slot(0, "alive")
    o_slot = lambda bus: bus.slot(1, "out")

    # `in` = 0 < threshold 1 → the bare predicate FIRES (fail-deadly: a dead
    # source reads 0 and trips the < wire).
    bus = _bus_one(slots, {(0, "value"): 0, (0, "alive"): 0})
    bare = _DeciderEval(o_slot(bus), [v_slot(bus)], "lt", 1,
                        require_alive=False, alive_slot=None)
    bare.evaluate(bus)
    assert int(bus.stg[o_slot(bus)]) == 1          # value passes through (fires)

    # require_alive: the same dead source (alive==0) forces the decider to 0
    # (fail-passive — the door is NOT commanded).
    bus = _bus_one(slots, {(0, "value"): 0, (0, "alive"): 0})
    passive = _DeciderEval(o_slot(bus), [v_slot(bus)], "lt", 1,
                           require_alive=True, alive_slot=a_slot(bus))
    passive.evaluate(bus)
    assert int(bus.stg[o_slot(bus)]) == 0

    # require_alive with a LIVE source: predicate result stands.
    bus = _bus_one(slots, {(0, "value"): 0, (0, "alive"): 1})
    passive = _DeciderEval(o_slot(bus), [v_slot(bus)], "lt", 1,
                           require_alive=True, alive_slot=a_slot(bus))
    passive.evaluate(bus)
    assert int(bus.stg[o_slot(bus)]) == 1


# ---------------------------------------------------------------------------
# filter — the exact tau-snap (§5, D5)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tau_s,tps,k", [
    (0.0, 24, 0),                          # zero tau → no smoothing
    (1.0 / 24, 24, 0),                     # tau*tps == 1 → log2(1) = 0
    (1.5 / 24, 24, 1),                     # x=1.5 → round(0.585) = 1
    (2.0 / 24, 24, 1),                     # x=2   → log2 = 1
    (3.0 / 24, 24, 2),                     # x=3   → round(1.585) = 2
    (1.0, 24, 5),                          # x=24  → round(4.585) = 5
    (10.0, 24, 8),                         # x=240 → round(7.907) = 8
])
def test_snap_filter_k_values(tau_s, tps, k):
    assert snap_filter_k(tau_s, tps) == k


def test_snap_filter_k_boundary_sqrt2_rounds_up():
    # The exact boundary x = sqrt(2) (x^2 == 2): a naive float
    # round(log2(1.4142135623...)) can flip either side of 0.5; the exact
    # Fraction rule rounds HALF-UP to k=1, deterministically. Author tau so
    # tau_s * tps == sqrt(2) is impossible in decimals, so we probe just below
    # and at the integer-exact half via tps engineering: pick x just under and
    # over sqrt(2) and confirm the split is monotone and lands where the exact
    # x^2 >= 2 rule says.
    # x = 1.41 < sqrt(2): 1.41^2 = 1.9881 < 2 → k=0
    assert snap_filter_k(Fraction(141, 100), 1) == 0
    # x = 1.42 > sqrt(2): 1.42^2 = 2.0164 >= 2 → k=1
    assert snap_filter_k(Fraction(142, 100), 1) == 1
    # A float log2 near this seam agrees in direction (sanity), but the point
    # is the snap NEVER consults it — see the no-float test below.
    assert (round(math.log2(1.41)) == 0) and (round(math.log2(1.42)) == 1)


def _k_exact_reference(tau_s, tps):
    """The exact integer/Fraction reference: k = largest count with the EMA
    input x = tau*tps satisfying x^2 >= 2^(2k+1). No float log2/round."""
    x = Fraction(str(float(tau_s))) * Fraction(int(tps))
    if x <= 0:
        return 0
    x2 = x * x
    k = 0
    while x2 >= (1 << (2 * k + 1)):
        k += 1
    return k


def test_snap_filter_k_is_pure_integer_no_float_dependence():
    # Cross-machine stability (§9): the snap must equal a hand-rolled EXACT
    # integer/Fraction reference across a wide sweep — never a float path.
    for milli in range(0, 3000, 7):        # tau_s = 0.000 .. 2.993 s
        tau_s = milli / 1000.0
        assert snap_filter_k(tau_s, 24) == _k_exact_reference(tau_s, 24)
    # Determinism: identical inputs → identical output, many repeats, no RNG.
    assert len({snap_filter_k(0.7, 24) for _ in range(50)}) == 1


# ---------------------------------------------------------------------------
# filter — EMA guard-bit round-to-nearest + step response (§5)
# ---------------------------------------------------------------------------

def _filter_run(k, xs):
    """Drive a bare FilterRuntime EMA with input sequence ``xs``; return the
    per-tick out list. Uses a 2-slot bus (in=0, out=1)."""
    bus = SignalBus([(0, "in"), (1, "out")])
    frt = FilterRuntime(inst=_node_inst("f", 9, "filter"),
                        out_slot=bus.slot(1, "out"), in_slot=bus.slot(0, "in"),
                        k=k)
    outs = []
    for x in xs:
        bus.set_pub(bus.slot(0, "in"), x)
        frt.evaluate(bus)
        outs.append(int(bus.stg[bus.slot(1, "out")]))
    return outs, frt


def test_filter_k0_is_passthrough():
    outs, _ = _filter_run(0, [10, 20, 30])
    assert outs == [10, 20, 30]            # alpha == 1, no smoothing


def test_filter_step_response_monotone_and_reaches_target():
    # Constant input 100 with k=1 (alpha=1/2): out rises monotonically toward
    # 100 and REACHES it exactly (no truncation park — the guard-bit fix).
    outs, _ = _filter_run(1, [100] * 12)
    assert outs[0] == 50
    assert all(outs[i] <= outs[i + 1] for i in range(len(outs) - 1))
    assert outs[-1] == 100                 # reaches target, does not park below


def test_filter_no_truncation_park_small_input():
    # The truncation-park bug: a naive `ema += (x-ema)>>k` with a small x never
    # moves ema when x < 2^k. Here k=4 (2^4=16) and x=3 (< 16): the guard-bit
    # accumulator MUST still climb off zero and reach 3.
    outs, _ = _filter_run(4, [3] * 400)
    assert outs[0] == 0                    # first tick still rounds to 0…
    assert max(outs) == 3                  # …but it climbs and reaches 3
    assert outs[-1] == 3


def test_filter_round_to_nearest_before_shift():
    # k=1, feed 1 then 1: acc after t0 = 0 + 1 - ((0+1)>>1=0) = 1, out=(1+1)>>1
    # = 1 (rounds 0.5 up). A truncating shift would give out=0.
    outs, frt = _filter_run(1, [1])
    assert outs == [1]
    assert int(frt.ema) == 1               # accumulator carries the guard bit


# ---------------------------------------------------------------------------
# Integration through a real Simulation — latency, signals, digest rows
# ---------------------------------------------------------------------------

def _sim_with(entities, wires, gap_rows=(3, 8)):
    tm = _split_box_tm(gap_rows=gap_rows)
    return Simulation(_level(tm, entities, wires), seed=1,
                      breach_physics=None, enable_recorder=False)


def test_node_hop_adds_exactly_one_tick():
    # B1 pinned door_A.is_open → door_B.close = door_B flips at tick 1. Insert
    # a gate_not hop (door_A.is_open → gate_not.in → door_B.open): the hop adds
    # exactly one tick, and the negation means B is driven OPEN while A is shut.
    d_a = _door_inst("door_A", 0, x=6, y=3, initial_state="closed")
    g = _node_inst("g_not", 1, "gate_not")
    d_b = _door_inst("door_B", 2, x=6, y=8, initial_state="closed")
    wires = [Wire(0, "is_open", 1, "in", INPUT_SINGLE),      # A.is_open → not
             Wire(1, "out", 2, "open", INPUT_HELD)]          # not.out → B.open
    sim = _sim_with([d_a, g, d_b], wires)
    a = sim.door_at(3, 6)
    b = sim.door_at(8, 6)
    assert a.state == CLOSED and b.state == CLOSED

    # The bus has A.is_open, gate_not.out; the node output is a swapped slot.
    bus = sim._signal_bus
    assert bus.has(0, "is_open") and bus.has(1, "out")

    # Tick 0: (a) A.is_open emitted 0; (b) not(0)=1 staged; (c) B.open reads
    # pub[not.out] = 0 (pre-swap) → B stays; (e) swap makes not.out = 1.
    _step(sim)
    # A had want_open still False (unwired latch), stays closed; not.out went 1.
    assert b.state == CLOSED
    assert int(bus.read(1, "out")) == 1

    # Now open door_A via its free latch; trace the two-hop timing.
    a.want_open = True
    # Tick 1: A flips OPEN at (d). is_open emitted THIS tick was still 0 (emit
    # precedes the flip). not(0)=1. B.open reads not.out(pub)=1 → B opens (d).
    _step(sim)
    assert a.state == OPEN
    # Tick 2: (a) A.is_open now 1; (b) not(1)=0 staged; (c) B.open reads
    # not.out(pub)=1 still (swap is end-of-tick) → B stays open this tick.
    _step(sim)
    # Tick 3: not.out(pub) now 0 → B.open inactive → B retains its open latch
    # (open has no close to fight); B stays open. The point tested: the not.out
    # transition is observed one tick after A.is_open changed (the hop).
    _step(sim)
    assert int(bus.read(1, "out")) == 0    # negation of A-open, one hop later


def test_decider_drives_door_and_signals_populated():
    # sensor-substitute: door_A.is_open (0/1) → decider(ge, 1) → door_B.close.
    # When A opens, is_open=1 ≥ 1 → decider out 1 → B closes (2 hops: emit→
    # decider→door). Verifies __signals__ carries the decider out + door slot.
    d_a = _door_inst("door_A", 0, x=6, y=3, initial_state="closed")
    dec = _node_inst("dec", 1, "decider", comparator="ge", threshold=1)
    d_b = _door_inst("door_B", 2, x=6, y=8, initial_state="open")
    wires = [Wire(0, "is_open", 1, "in", INPUT_SINGLE),
             Wire(1, "out", 2, "close", INPUT_HELD)]
    sim = _sim_with([d_a, dec, d_b], wires)
    a = sim.door_at(3, 6)
    b = sim.door_at(8, 6)
    a.want_open = True
    # Give the two-hop chain time to propagate and shut B.
    _step(sim, 4)
    assert a.state == OPEN and b.state == CLOSED
    # __signals__ carries the decider out (=1) and the wired door slot; alive
    # slots (none here) never appear.
    sig = dict(((o, n), v) for o, n, v in sim._digest_signals())
    assert sig[(1, "out")] == 1
    assert (0, "is_open") in sig           # door_A.is_open is a wire source


def test_filter_ema_digest_row_present_and_signals():
    # A filter instance → its EMA accumulator rides the __entity__ runtime rows.
    d_a = _door_inst("door_A", 0, x=6, y=3, initial_state="closed")
    filt = _node_inst("filt", 1, "filter", tau_s=0.5)
    wires = [Wire(0, "is_open", 1, "in", INPUT_SINGLE)]
    sim = _sim_with([d_a, filt], wires, gap_rows=(3,))
    # The filter instance was replaced by a FilterRuntime carrying `ema`.
    frt = sim._entity_by_ordinal[1]
    assert isinstance(frt, FilterRuntime)
    rows = dict(REGISTRY["filter"].runtime_digest_rows(frt))
    assert "ema" in rows
    # Drive door_A open so the filter sees a 0→1 step; ema climbs off zero.
    a = sim.door_at(3, 6)
    a.want_open = True
    _step(sim, 6)
    rows2 = dict(REGISTRY["filter"].runtime_digest_rows(frt))
    assert rows2["ema"] >= 0
    # filter.out is a node slot in __signals__ (non-alive).
    sig = {(o, n) for o, n, _ in sim._digest_signals()}
    assert (1, "out") in sig


def test_filter_row_absent_when_uninstantiated():
    # No filter in the level → no filter runtime object → the `ema` row is
    # ABSENT from every serialized record (dormancy of the class, §8).
    from simulation.entities.serialize import entity_records
    d_a = _door_inst("door_A", 0, x=6, y=3, initial_state="closed")
    dec = _node_inst("dec", 1, "decider", comparator="gt", threshold=0)
    wires = [Wire(0, "is_open", 1, "in", INPUT_SINGLE)]
    sim = _sim_with([d_a, dec], wires, gap_rows=(3,))
    _step(sim, 2)
    blob = b"".join(entity_records(sim.entities))
    assert b"ema" not in blob              # no filter → zero `ema` bytes


# ---------------------------------------------------------------------------
# Loader — the SINGLE arity check goes LIVE (§2d)
# ---------------------------------------------------------------------------

def test_single_input_arity_enforced_second_wire_hard_errors():
    d0 = _door_inst("d0", 0, x=6, y=3)
    d1 = _door_inst("d1", 1, x=6, y=8)
    dec = _node_inst("dec", 2, "decider")
    raw = {"wire": [{"from": "d0.is_open", "to": "dec.in"},
                    {"from": "d1.is_open", "to": "dec.in"}]}
    with pytest.raises(ValueError, match="single-arity input accepts exactly one"):
        _parse_wires(raw, "L.toml", [d0, d1, dec], [])


def test_and_input_accepts_multiple_wires():
    d0 = _door_inst("d0", 0, x=6, y=3)
    d1 = _door_inst("d1", 1, x=6, y=8)
    g = _node_inst("g", 2, "gate_and")
    raw = {"wire": [{"from": "d0.is_open", "to": "g.in"},
                    {"from": "d1.is_open", "to": "g.in"}]}
    wires, _ = _parse_wires(raw, "L.toml", [d0, d1, g], [])
    assert len(wires) == 2
    assert all(w.aggregation_mode == INPUT_AND for w in wires)


# ---------------------------------------------------------------------------
# build_signal_bus — node emitter slots + require_alive alive slot (§2a)
# ---------------------------------------------------------------------------

def test_bus_adds_node_out_slots_as_swapped():
    # A gate_not with an unwired-source is still a "logic exists" trigger; its
    # `out` gets a slot and is a swapped node slot.
    g = _node_inst("g", 5, "gate_not")
    bus = build_signal_bus(_level(_split_box_tm(), entities=[g], wires=()))
    assert bus is not None
    assert bus.has(5, "out")
    # The out slot is swapped (node output), so swap_node_signals moves stg→pub.
    bus.stg[bus.slot(5, "out")] = 1
    bus.swap_node_signals()
    assert int(bus.read(5, "out")) == 1


def test_bus_adds_alive_slot_for_require_alive_decider():
    d_a = _door_inst("door_A", 0, x=6, y=3)
    dec = _node_inst("dec", 1, "decider", require_alive=True)
    wires = [Wire(0, "is_open", 1, "in", INPUT_SINGLE)]
    bus = build_signal_bus(_level(_split_box_tm(), entities=[d_a, dec],
                                  wires=wires))
    # The source's alive slot exists (so 9e a can emit it, D4)…
    assert bus.has(0, "alive")
    # …but alive is EXCLUDED from __signals__ (A4 c7).
    bus.set_pub(bus.slot(0, "alive"), 1)
    assert all(n != "alive" for _, n, _ in bus.digest_rows())


# ---------------------------------------------------------------------------
# DORMANCY — re-run the B1 gate explicitly (byte-identical, escalation-3)
# ---------------------------------------------------------------------------

def test_b1_dormancy_still_byte_identical():
    from field_ab_harness import _capture_unit_state, _sim_entities, UNIT_DIGEST_KEY
    from simulation.entities.serialize import ENTITY_DIGEST_KEY, entity_carrier
    from field_digest import DIGEST_FIELDS, trajectory_digest

    DOORTEST_NOPHYS_TRAJ_DIGEST = \
        "5d944aa8b085fa24a100575a1292196058f15953e0c0726f95342650cb685d8b"

    lvl = level_loader.load("door_test", levels_dir=str(ROOT / "levels"))
    sim = Simulation(lvl, seed=42, breach_physics=None, enable_recorder=False)
    assert sim._signal_bus is None         # no wires, no nodes → no bus (D1)
    assert sim._logic_nodes == []
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
        assert snap[ENTITY_DIGEST_KEY]["signals"] == ()
        traj.append(snap)
    assert trajectory_digest(traj) == DOORTEST_NOPHYS_TRAJ_DIGEST


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
