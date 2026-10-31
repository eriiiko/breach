"""Properties of the INTERIOR airlock (#133; Erik's ruling 2026-10-31,
.claude/skills/level-design-lead/SKILL.md §3): "an interior airlock pair (two
pressurised cells) only sequences its doors and equalises the vestibule to
the side being opened; only a HULL airlock (one side is space) evacuates.
The generator picks the mode at generation time."

The pieces: ``pump`` ``feed = "transfer"`` (GameMap.transfer_gas_n, a
conservative port <-> source move), ``airlock_controller`` ``mode =
"interior"`` (each leg equalises the chamber to the side about to open, on
three wired pressure readings), and gen_ship's ``airlock_mode`` /
``airlock_entities``. Every test names the property it protects and the
change that must break it. The level is gen_ship's airlock fixture (the
host section and its annex, tests/test_gen_ship_vents_doors.py); its hull
twin is the same geometry with the hull recipe.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests", ROOT / "tools",
           ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import gen_ship as G  # noqa: E402
import level_lib  # noqa: E402
import level_loader  # noqa: E402
import test_gen_ship_vents_doors as V  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation.entities import REGISTRY  # noqa: E402
from simulation.entities.actuators import (  # noqa: E402
    AIRLOCK_EQUALIZE, AIRLOCK_IDLE, AIRLOCK_OPEN_FAR, AIRLOCK_REPRESSURIZE,
)
from simulation.entities.door import DOOR_OPEN  # noqa: E402
from simulation.gamemap import INERT_N2, O2  # noqa: E402
from simulation.logic_nodes import AirlockControllerRuntime  # noqa: E402
from simulation.signal_bus import SignalBus  # noqa: E402
from simulation.unit import Unit  # noqa: E402

try:
    import breach_physics as _bp  # noqa: E402
except Exception:                  # pragma: no cover - a box without the build
    _bp = None
needs_bp = pytest.mark.skipif(_bp is None,
                              reason="needs the compiled breach_physics")

FP_ONE = 65536
P_B_LOW = 45875                    # the annex seeded at 0.7 atm (air_init)


# ---------------------------------------------------------------------------
# Fixtures: the generated airlock level, interior or its hull twin
# ---------------------------------------------------------------------------

def _write(tmp_path, monkeypatch=None, hull=False, p_b=None, name="fx",
           edit=None):
    """Write gen_ship's airlock fixture. ``hull``: force the hull recipe
    on the same geometry (the twin). ``p_b``: seed the annex (room 3) at
    this pressure through air_init.npy, the rest at 1 atm. ``edit``: a
    callable on the plan's embedding before the write (refusal tests)."""
    if hull:
        monkeypatch.setattr(G, "airlock_mode", lambda tm, al: "hull")
    g = V._airlock_graph()
    plan = V._plan(g)
    if edit is not None:
        edit(plan.embedding)
    G.write(plan, tmp_path / name)
    if p_b is not None:
        air = np.full(plan.tilemap.shape, FP_ONE, dtype=np.int32)
        x0, y0, x1, y1 = g.rooms[3].box
        air[y0:y1 + 1, x0:x1 + 1] = p_b
        air[plan.tilemap == level_loader.SPACE_CODE] = 0
        level_lib.write_air_init_npy(tmp_path / name, air, npy_bak=False)
    (_i, al), = G._airlocks(g)
    return g, al


def _sim(tmp_path, name="fx"):
    lv = level_loader.load(name, levels_dir=str(tmp_path))
    return Simulation(lv, seed=1, breach_physics=_bp, enable_recorder=False)


def _bulk_total(gm) -> int:
    return int(gm.gas[O2].astype(np.int64).sum()
               + gm.gas[INERT_N2].astype(np.int64).sum())


def _box_mask(shape, box):
    m = np.zeros(shape, dtype=bool)
    x0, y0, x1, y1 = box
    m[y0:y1 + 1, x0:x1 + 1] = True
    return m


def _cycle(sim, g, al, settle=60, cap=1500, after=200):
    """Settle, put a unit in the vestibule, run the cycle A -> B (the unit
    leaves once the far door opens) and ``after`` ticks past the return to
    IDLE. Returns the per-tick record the tests read."""
    gm = sim.gmap
    ctrl = next(e for e in sim.entities
                if isinstance(e, AirlockControllerRuntime))
    outer = next(d for d in sim._doors if d.id.endswith("_outer"))
    inner = next(d for d in sim._doors if d.id.endswith("_inner"))
    air = ~gm.solid & ~gm.is_vacuum
    chamber = _box_mask(gm.solid.shape, al.chamber) & air
    vest = _box_mask(gm.solid.shape, al.chamber)
    for (y, x) in al.carve:
        vest[y, x] = True
    side_b = _box_mask(gm.solid.shape, g.rooms[3].box) & air
    side_a = _box_mask(gm.solid.shape, g.rooms[1].box) & air & ~vest

    def mean(m):
        return int(gm.atmosphere[m].astype(np.int64).mean())

    def n_of(m):
        return int(gm.gas[O2][m].astype(np.int64).sum()
                   + gm.gas[INERT_N2][m].astype(np.int64).sum())

    for _ in range(settle):
        sim.set_paused(False)
        sim.step()
    rec = types.SimpleNamespace(
        n0=_bulk_total(gm), n=[], both_open=0, states=[],
        p_a0=mean(side_a), p_b0=mean(side_b), nb0=n_of(side_b),
        outer_open_at=None, inner_reopen_at=None, idle_at=None,
        chamber_tiles=int(chamber.sum()),
        quantum=max(p.delta_n_q for p in sim._pumps))
    unit = Unit("Alpha", x=al.chamber[0] + 1, y=al.chamber[1] + 1, team=0,
                footprint=3)
    sim.add_unit(unit)
    prev_outer = prev_inner = None
    for t in range(cap):
        sim.set_paused(False)
        sim.step()
        rec.n.append(_bulk_total(gm))
        rec.states.append(ctrl.state)
        o, i = outer.state == DOOR_OPEN, inner.state == DOOR_OPEN
        rec.both_open += int(o and i)
        if o and prev_outer is False and rec.outer_open_at is None:
            rec.outer_open_at = (t, mean(chamber), mean(side_b))
        if (i and prev_inner is False and rec.outer_open_at is not None
                and rec.inner_reopen_at is None):
            rec.inner_reopen_at = (t, mean(chamber), mean(side_a))
        prev_outer, prev_inner = o, i
        if ctrl.state == AIRLOCK_OPEN_FAR and unit.alive:
            unit.alive = False            # the unit walks on into B
        if (rec.idle_at is None and ctrl.state == AIRLOCK_IDLE
                and AIRLOCK_OPEN_FAR in rec.states):
            rec.idle_at = t
        if rec.idle_at is not None and t >= rec.idle_at + after:
            break
    rec.p_a1, rec.p_b1, rec.nb1 = mean(side_a), mean(side_b), n_of(side_b)
    return rec


# ---------------------------------------------------------------------------
# The cycle, headless, on real physics
# ---------------------------------------------------------------------------

@needs_bp
def test_interior_cycle_moves_a_unit_and_no_gas_leaves_while_the_hull_twin_vents(
        tmp_path, monkeypatch):
    """PROPERTY: an interior airlock between two pressurised rooms cycles a
    unit from A to B (far door opens, then the cycle returns to IDLE) and
    the ship's total bulk gas N never moves by more than one pump quantum at
    any tick -- no gas leaves -- while the HULL twin on the same geometry
    sheds at least half a vestibule of gas on its evacuate leg. Neither mode
    ever has both doors open at once. BREAKS IF the interior controller
    falls back to the external pump (evacuate / refill), the transfer pump
    mints or destroys, or the door sequencing lets both doors open."""
    g, al = _write(tmp_path / "i")
    rec = _cycle(_sim(tmp_path / "i"), g, al)
    assert rec.outer_open_at is not None and rec.idle_at is not None
    assert max(abs(n - rec.n0) for n in rec.n) <= rec.quantum
    assert rec.both_open == 0

    g, al = _write(tmp_path / "h", monkeypatch, hull=True)
    hull = _cycle(_sim(tmp_path / "h"), g, al)
    assert hull.outer_open_at is not None and hull.idle_at is not None
    assert min(hull.n) < hull.n0 - hull.chamber_tiles * FP_ONE // 2
    assert hull.both_open == 0


@needs_bp
def test_the_vestibule_matches_the_side_being_opened_when_its_door_opens(
        tmp_path):
    """PROPERTY: with the far room at 0.7 atm and the near side at 1 atm,
    the vestibule's mean pressure is within the controller's equalize band
    of the far room's when the OUTER door opens, and of the near room's
    when the INNER
    door reopens -- the airlock equalises to the side it opens, so a door
    never opens onto a pressure step. BREAKS IF a leg ends on the wrong
    side's reading, the pump runs the wrong way, or the leg ends before
    the chamber gets there (a band wider than 0.05 atm)."""
    g, al = _write(tmp_path, p_b=P_B_LOW)
    rec = _cycle(_sim(tmp_path), g, al)
    assert rec.p_a0 - rec.p_b0 > FP_ONE // 5      # the step is really there
    # The tolerance is the controller's own band (0.05 atm), well inside
    # the 0.3 atm step: the per-TILE pump quantum (rate / tps, 1/3 atm at
    # the fixture's rate) would accept the step itself.
    band = G.AIRLOCK_EQUALIZE_BAND
    _t, p_ch, p_far = rec.outer_open_at
    assert abs(p_ch - p_far) <= band, (p_ch, p_far)
    _t, p_ch, p_near = rec.inner_reopen_at
    assert abs(p_ch - p_near) <= band, (p_ch, p_near)
    assert rec.both_open == 0


@needs_bp
def test_a_pressure_difference_survives_a_cycle_up_to_one_vestibule(
        tmp_path):
    """PROPERTY: the airlock is not a pump between cells. Cycling a unit
    from the 1 atm side into the 0.7 atm room carries at most one
    vestibule's worth of the difference into the low room (ΔN_B <= chamber
    tiles x (p_A - p_B), with 25 % slack for the chamber's own ripple),
    the high side stays high, and the ship's total N is conserved. BREAKS
    IF a leg keeps pumping past equality (draining A into B), the transfer
    pump's source is the wrong side, or gas is created or lost."""
    g, al = _write(tmp_path, p_b=P_B_LOW)
    rec = _cycle(_sim(tmp_path), g, al)
    dp0 = rec.p_a0 - rec.p_b0
    gained = rec.nb1 - rec.nb0
    assert 0 <= gained <= rec.chamber_tiles * dp0 * 5 // 4, (gained, dp0)
    assert rec.p_a1 - rec.p_b1 > 0                  # not reversed, not erased
    assert abs(rec.n[-1] - rec.n0) <= rec.quantum


# ---------------------------------------------------------------------------
# The controller's interior legs (a synthetic bus, no physics)
# ---------------------------------------------------------------------------

_IN = {
    "presence": (10, "value"),
    "inner_open": (0, "is_open"), "inner_alive": (0, "alive"),
    "outer_open": (1, "is_open"), "outer_alive": (1, "alive"),
    "at_far": (2, "out"), "at_near": (3, "out"),
    "chamber_p": (11, "value"), "inner_side_p": (12, "value"),
    "outer_side_p": (13, "value"),
}
_OUT = tuple(sig.name for sig in REGISTRY["airlock_controller"].SIGNALS)
_CTRL = 4
_BAND = 3277


def _controller(interior):
    slots = sorted(list(_IN.values()) + [(_CTRL, s) for s in _OUT])
    outs = {(_CTRL, s) for s in _OUT}
    bus = SignalBus(slots, node_slots=tuple(i for i, s in enumerate(slots)
                                            if s in outs))
    inst = types.SimpleNamespace(ordinal=_CTRL, id="c",
                                 class_name="airlock_controller", fields={})
    ctrl = AirlockControllerRuntime(
        inst, True, -1, {k: bus.slot(*v) for k, v in _IN.items()},
        {s: bus.slot(_CTRL, s) for s in _OUT}, interior=interior, band=_BAND)
    return ctrl, bus


def _tick(ctrl, bus, presence=1, inner_open=0, outer_open=0, ch=FP_ONE,
          p_in=FP_ONE, p_out=FP_ONE, at_far=0, at_near=0):
    vals = dict(presence=presence, inner_open=inner_open,
                outer_open=outer_open, inner_alive=1, outer_alive=1,
                at_far=at_far, at_near=at_near, chamber_p=ch,
                inner_side_p=p_in, outer_side_p=p_out)
    for k, v in vals.items():
        bus.set_pub(bus.slot(*_IN[k]), v)
    ctrl.evaluate(bus)
    bus.swap_node_signals()
    return {s: int(bus.read(_CTRL, s)) for s in _OUT}


def test_interior_legs_pump_toward_the_side_about_to_open_and_end_in_band():
    """PROPERTY: in interior mode EQUALIZE drives the FAR side's pump toward
    the far reading (inject when the far side is higher, extract when
    lower) and ends inside ``equalize_band``; REPRESSURIZE drives only the
    NEAR side's pump toward the near reading and ends the same way; the
    deciders' at_far / at_near are ignored. BREAKS IF a leg reuses the
    hull's fixed direction, drives the other side's pump, or gates on the
    decider inputs."""
    ctrl, bus = _controller(interior=True)
    _tick(ctrl, bus)                                   # IDLE -> CLOSING
    out = _tick(ctrl, bus, ch=FP_ONE, p_out=P_B_LOW, at_far=1)
    assert ctrl.state == AIRLOCK_EQUALIZE              # at_far ignored
    assert (out["pump_inject"], out["pump_extract"]) == (0, 1)
    assert out["near_pump_inject"] == out["near_pump_extract"] == 0
    out = _tick(ctrl, bus, ch=P_B_LOW - 2 * _BAND, p_out=P_B_LOW)
    assert (out["pump_inject"], out["pump_extract"]) == (1, 0)
    out = _tick(ctrl, bus, ch=P_B_LOW + _BAND, p_out=P_B_LOW)
    assert ctrl.state == AIRLOCK_OPEN_FAR and out["outer_open_cmd"] == 1
    assert out["pump_inject"] == out["pump_extract"] == 0
    _tick(ctrl, bus, presence=0, outer_open=1)         # -> RESEAL
    out = _tick(ctrl, bus, presence=0, ch=P_B_LOW, p_in=FP_ONE, at_near=1)
    assert ctrl.state == AIRLOCK_REPRESSURIZE
    assert (out["near_pump_inject"], out["near_pump_extract"]) == (1, 0)
    assert out["pump_inject"] == out["pump_extract"] == 0
    out = _tick(ctrl, bus, presence=0, ch=FP_ONE - _BAND, p_in=FP_ONE)
    assert ctrl.state == AIRLOCK_IDLE and out["inner_open_cmd"] == 1


def test_hull_mode_never_drives_the_near_pump_and_keeps_its_direction():
    """PROPERTY: hull mode (the default) is the pre-#133 cycle: EQUALIZE
    and REPRESSURIZE drive the one pump in the targets' fixed directions,
    gated by at_far / at_near, and the near_pump_* outputs stay 0 whatever
    the pressure readings say. BREAKS IF the interior logic leaks into hull
    mode (the shipped airlock_demo would change behaviour)."""
    ctrl, bus = _controller(interior=False)
    _tick(ctrl, bus)
    out = _tick(ctrl, bus, ch=0, p_out=FP_ONE)
    assert ctrl.state == AIRLOCK_EQUALIZE
    assert (out["pump_inject"], out["pump_extract"]) == (0, 1)
    out = _tick(ctrl, bus, ch=0, p_out=0)              # equal, but no at_far
    assert ctrl.state == AIRLOCK_EQUALIZE
    _tick(ctrl, bus, at_far=1)
    _tick(ctrl, bus, presence=0, outer_open=1)
    out = _tick(ctrl, bus, presence=0)
    assert ctrl.state == AIRLOCK_REPRESSURIZE
    assert (out["pump_inject"], out["pump_extract"]) == (1, 0)
    assert out["near_pump_inject"] == out["near_pump_extract"] == 0


# ---------------------------------------------------------------------------
# The doors: defaults, refusals
# ---------------------------------------------------------------------------

def test_a_level_without_mode_or_feed_loads_as_hull_and_external():
    """PROPERTY: an existing level that names no ``mode`` / ``feed``
    (levels/airlock_demo) loads as a hull airlock with an external pump --
    nothing shipped changes. BREAKS IF either default flips."""
    lv = level_loader.load("airlock_demo", levels_dir=str(ROOT / "levels"))
    sim = Simulation(lv, seed=1, breach_physics=None, enable_recorder=False)
    ctrl = next(e for e in sim.entities
                if isinstance(e, AirlockControllerRuntime))
    assert ctrl.fields["mode"] == "hull" and not ctrl.interior
    pump, = sim._pumps
    assert pump.fields["feed"] == "external" and pump.source is None


def test_an_unknown_mode_or_feed_is_refused_by_name(tmp_path):
    """PROPERTY: the entity door refuses a controller ``mode`` or a pump
    ``feed`` outside its declared choices, naming the field. BREAKS IF the
    field stops being a closed enum (a typo would silently run hull)."""
    def bad_mode(emb):
        next(e for e in emb.entities
             if e.class_name == "airlock_controller").fields["mode"] = "x"

    def bad_feed(emb):
        next(e for e in emb.entities
             if e.class_name == "pump").fields["feed"] = "x"

    for k, edit in enumerate((bad_mode, bad_feed)):
        with pytest.raises(Exception, match="mode|feed"):
            _write(tmp_path / str(k), edit=edit)
            _sim(tmp_path / str(k))


def test_an_interior_controller_without_its_pressure_wires_is_refused(
        tmp_path):
    """PROPERTY: an interior controller with an unwired pressure reading
    is refused at load, naming the input -- an unwired reading is 0, which
    would equalise to vacuum. BREAKS IF the load check is dropped."""
    def drop(emb):
        emb.wires[:] = [w for w in emb.wires
                        if not w[1].endswith(".chamber_p")]
    _write(tmp_path, edit=drop)
    with pytest.raises(ValueError, match="chamber_p"):
        _sim(tmp_path)


def test_a_transfer_pump_needs_a_source_distinct_from_its_port(tmp_path):
    """PROPERTY: a ``feed = "transfer"`` pump whose source is its own port
    is refused at load. BREAKS IF the check is dropped (the pump would move
    gas from a tile to itself and the airlock would never equalise)."""
    def same(emb):
        p = next(e for e in emb.entities if e.id.endswith("_pump_far"))
        p.fields["source_dx"] = p.fields["port_dx"]
        p.fields["source_dy"] = p.fields["port_dy"]
    _write(tmp_path, edit=same)
    with pytest.raises(ValueError, match="source"):
        _sim(tmp_path)


# ---------------------------------------------------------------------------
# The conservative primitive
# ---------------------------------------------------------------------------

def test_transfer_gas_n_moves_air_trace_and_energy_without_minting(tmp_path):
    """PROPERTY: ``GameMap.transfer_gas_n`` moves bulk air, the trace it
    prices and the parcel's energy from one tile to another: every gas
    slice's two-cell sum is unchanged, the seam books net to zero, the
    trace leaves at the donor's air ratio, and an over-ask takes only what
    the donor holds. BREAKS IF the move mints / destroys (the external
    pump's path), the energy goes unbooked, or trace is left behind."""
    _g, al = _write(tmp_path)
    gm = _sim(tmp_path).gmap
    x0, y0, x1, y1 = al.chamber                    # two air tiles
    (sy, sx), (dy, dx) = (y0, x0), (y1, x1)
    smoke = next(k for k in range(gm.gas.shape[0])
                 if not gm.gases.conservative[k])
    gm.gas[smoke][sy, sx] = 1 << 24
    before = [int(gm.gas[k][sy, sx]) + int(gm.gas[k][dy, dx])
              for k in range(gm.gas.shape[0])]
    e_before = int(gm.gas_energy[sy, sx]) + int(gm.gas_energy[dy, dx])
    seam0 = gm.gas_energy_seam_net()
    n_src = gm.gas_bulk_n_at(sy, sx)
    moved = gm.transfer_gas_n(sy, sx, dy, dx, n_src // 4)
    assert moved == n_src // 4
    after = [int(gm.gas[k][sy, sx]) + int(gm.gas[k][dy, dx])
             for k in range(gm.gas.shape[0])]
    assert after == before
    assert int(gm.gas_energy[sy, sx]) + int(gm.gas_energy[dy, dx]) == e_before
    assert gm.gas_energy_seam_net() == seam0
    assert int(gm.gas[smoke][dy, dx]) == ((1 << 24) * moved) // n_src
    left = gm.gas_bulk_n_at(sy, sx)
    assert gm.transfer_gas_n(sy, sx, dy, dx, 10 * FP_ONE) == left
    assert gm.gas_bulk_n_at(sy, sx) == 0


# ---------------------------------------------------------------------------
# The generator picks the mode
# ---------------------------------------------------------------------------

def test_the_generator_derives_interior_between_rooms_and_hull_onto_space():
    """PROPERTY: ``airlock_mode`` reads the tiles beyond the outer door:
    two rooms -> ``interior`` (and the emitted controller says so), space
    beyond -> ``hull``; ``airlock_entities`` for hull is the demo recipe
    (an external pump, two deciders, a controller with no ``mode`` key,
    i.e. hull by default) and refuses an unknown mode by name. BREAKS IF
    the derivation ignores the far side, the generator writes the hull
    recipe between two pressurised rooms, or the hull recipe drifts from
    the demo."""
    g = V._airlock_graph()
    emb, reasons = V._run(g)
    assert reasons == []
    (_i, al), = G._airlocks(g)
    assert G.airlock_mode(emb.tilemap, al) == "interior"
    ctrl, = [e for e in emb.entities if e.class_name == "airlock_controller"]
    assert ctrl.fields["mode"] == "interior"
    assert all(e.fields["feed"] == "transfer" for e in emb.entities
               if e.class_name == "pump")

    tm = emb.tilemap.copy()
    for (y, x) in al.beyond:
        tm[y, x] = level_loader.SPACE_CODE
    assert G.airlock_mode(tm, al) == "hull"
    ents, wires = G.airlock_entities("x", al, "hull")
    by = {}
    for e in ents:
        by.setdefault(e.class_name, []).append(e)
    assert len(by["decider"]) == 2 and len(by["pump"]) == 1
    assert "feed" not in by["pump"][0].fields
    assert "mode" not in by["airlock_controller"][0].fields
    with pytest.raises(ValueError, match="sideways"):
        G.airlock_entities("x", al, "sideways")
