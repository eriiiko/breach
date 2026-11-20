"""G7p (#183) -- the FLASHLIGHT BIT: ``Unit.flashlight_on``,
``Simulation.set_flashlight`` (a hold verb in its OFF sense), the unit digest
key, the frame-light assembly's cones, the direct control's Q toggle.

PROPERTIES PROTECTED (each test names its own and the change that breaks it)
  1. The bit is ON by default and the verb holds whatever it is last told.
  2. It is synced and ABSENCE-TRANSPARENT: hashed only while off, so no
     digest moves until a unit's light is darkened.
  3. The assembly's cone rows follow the bit and nothing else moves: on -> the
     unit's cone row, off -> none, every other unit's row byte for byte the
     same; a zombie never carries one; direct control lights its possessed
     unit only, a watched match every living marine.
  4. Nothing in the sim reads the bit (light as game state is #81): a match
     plays the same with every light off.
  5. The direct control TOGGLES on the key's press and holds the state every
     frame; a zero policy row never darkens a unit.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tools", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402

import ui  # noqa: E402
from config import CFG  # noqa: E402
from control_direct import (DeviceFrame, DirectControl, INERT_ACTIONS,  # noqa: E402
                            key_table)
from field_ab_harness import _capture_unit_state, _unit_record  # noqa: E402
from level_loader import LevelData  # noqa: E402
from renderer import frame_lights as fl  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation.factions import ZOMBIE_TEAM  # noqa: E402
from simulation.materials import MAT_AIR, MAT_HULL  # noqa: E402
from simulation.ruleset import ContinuousRealtime  # noqa: E402
from simulation.turning import PI_Q16, place  # noqa: E402
from simulation.unit import LifeState, Unit  # noqa: E402

FB = int(bp.L_FINE_BITS)
EAST = 0


def _sim(h=40, w=40):
    tm = np.full((h, w), MAT_HULL, dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = MAT_AIR
    lvl = LevelData(name="light", version="1", path=Path("."), tilemap=tm,
                    tile_size_m=1.0 / 3.0, diffuse_path=Path("."))
    sim = Simulation(lvl, seed=1, breach_physics=bp, enable_recorder=False,
                     ruleset=ContinuousRealtime())
    sim.set_paused(False)
    return sim


def _unit(sim, x, y, team=0, facing=EAST):
    u = sim.get_unit(sim.add_unit(Unit("U", x=x, y=y, team=team)))
    place(u, facing)
    return u


def _rows(sim, cones):
    specs = fl.flashlight_specs(
        cones, lambda uid: int(sim.get_unit(uid).footprint))
    h, w = sim.gmap.solid.shape
    return fl.cone_rows(specs, w, h, FB)


# ---------------------------------------------------------------------------
# 1 -- the verb
# ---------------------------------------------------------------------------
def test_the_light_is_on_by_default_and_the_verb_holds_what_it_is_told():
    """Property: a new unit's flashlight is on; set_flashlight(uid, off) holds
    the OFF sense (True darkens, False lights) and a call for an unknown unit
    is ignored. Breaks if the default flips (every digest would move) or the
    argument's sense is inverted."""
    sim = _sim()
    u = _unit(sim, 10, 10)
    assert u.flashlight_on is True
    sim.set_flashlight(u.id, True)
    assert u.flashlight_on is False
    sim.set_flashlight(u.id, True)                 # a hold: idempotent
    assert u.flashlight_on is False
    sim.set_flashlight(u.id, False)
    assert u.flashlight_on is True
    sim.set_flashlight(9999, True)                 # unknown: no effect, no crash


# ---------------------------------------------------------------------------
# 2 -- the digest
# ---------------------------------------------------------------------------
def test_the_digest_is_unchanged_while_the_light_is_on_and_moves_when_it_is_off():
    """Property: a unit's digest record has no flashlight key while the light
    is on (so its hash is the one it had before G7p), gains one when off, and
    returns to the same hash when the light comes back on. Breaks if the bit
    is hashed at its default (every pinned digest would move) or never hashed
    (a desync of the bit would go unseen)."""
    sim = _sim()
    a = _unit(sim, 10, 10)
    b = _unit(sim, 20, 10, team=1)
    sim.step()
    base = _capture_unit_state(sim)["hash"]
    assert "flashlight_on" not in _unit_record(a)
    sim.set_flashlight(a.id, True)
    dark = _capture_unit_state(sim)["hash"]
    assert dark != base and "flashlight_on" in _unit_record(a)
    assert "flashlight_on" not in _unit_record(b)
    sim.set_flashlight(a.id, False)
    assert _capture_unit_state(sim)["hash"] == base


def test_nothing_in_the_sim_reads_the_bit():
    """Property (light as game state is #81): a scene played with every unit's
    light off ends in the same state as the same scene with every light on,
    but for the bit's own digest key. Breaks if the sim, a law or a
    controller starts reading the flashlight (a stealth system by drift)."""
    def play(off):
        sim = _sim()
        s = _unit(sim, 10, 20)
        _unit(sim, 14, 20, team=1)
        z = _unit(sim, 30, 20, team=ZOMBIE_TEAM)
        z.zombie_activated = True
        sim.set_aim(s.id, 65536, 0)
        sim.set_trigger(s.id, True)
        for _ in range(80):
            for u in sim.units:
                sim.set_flashlight(u.id, off)
            sim.step()
        recs = []
        for u in sorted(sim.units, key=lambda u: u.id):
            r = dict(_unit_record(u))
            r.pop("flashlight_on", None)
            recs.append(r)
        return recs
    assert play(False) == play(True)


# ---------------------------------------------------------------------------
# 3 -- the assembly
# ---------------------------------------------------------------------------
def test_the_cone_rows_follow_the_bit_and_leave_the_others_untouched():
    """Property: in a watched match every living marine has one cone row;
    switching one unit's light off removes exactly its row and leaves every
    other row byte for byte as it was; a zombie never has one and a dead unit
    has none. Breaks if the bit is ignored, removes a neighbour's row or
    perturbs the others, or a zombie gets a flashlight."""
    sim = _sim()
    _unit(sim, 8, 8)
    b = _unit(sim, 20, 8, team=1, facing=PI_Q16 // 2)
    _unit(sim, 8, 24)
    _unit(sim, 30, 30, team=ZOMBIE_TEAM)
    watch = lambda: _rows(sim, ui.carried_flashlight_cones(  # noqa: E731
        sim, watching=True))
    all_on = watch()
    assert len(all_on) == 3
    before = {tuple(r) for r in all_on}
    only_b = _rows(sim, [c for c in ui.flashlight_cones(sim, mode="all")
                         if c.unit_id == b.id])
    assert len(only_b) == 1
    b_row = tuple(only_b[0])
    sim.set_flashlight(b.id, True)
    after = {tuple(r) for r in watch()}
    assert after == before - {b_row}             # exactly b's row is gone
    sim.set_flashlight(b.id, False)
    assert {tuple(r) for r in watch()} == before
    sim.get_unit(0).life_state = LifeState.DEAD  # a corpse carries no light
    assert len(watch()) == 2


def test_direct_control_lights_the_possessed_unit_only_and_follows_its_bit():
    """Property: with a possessed unit the assembly's cones are that unit's
    alone (its squad-mates have none) while its bit is on, and none when it
    is off; with nothing possessed and no watched match there are none (and
    unit id 0 is a legal possessed unit). Breaks if direct control lights the
    whole squad, or its own unit regardless of the bit, or loses id 0 to a
    falsy test."""
    sim = _sim()
    p = _unit(sim, 8, 8)
    mate = _unit(sim, 20, 8)
    on = ui.carried_flashlight_cones(sim, possessed_id=p.id)
    assert [c.unit_id for c in on] == [p.id]
    assert len(_rows(sim, on)) == 1
    sim.set_flashlight(p.id, True)
    assert ui.carried_flashlight_cones(sim, possessed_id=p.id) == []
    sim.set_flashlight(mate.id, True)
    assert ui.carried_flashlight_cones(sim) == []
    sim0 = _sim()
    z0 = _unit(sim0, 8, 8)
    assert z0.id == 0
    assert [c.unit_id for c in
            ui.carried_flashlight_cones(sim0, possessed_id=0)] == [0]


def test_the_onephase_flashlights_also_follow_the_bit():
    """Property: the onephase planning flashlights (the old path of the same
    assembly) honour the bit too. Breaks if the bit gates only the new
    callers and a darkened squad member still shines in onephase."""
    sim = _sim()
    a = _unit(sim, 8, 8)
    b = _unit(sim, 20, 8)
    assert {c.unit_id for c in ui.flashlight_cones(sim, team=0)} == {a.id, b.id}
    sim.set_flashlight(a.id, True)
    assert {c.unit_id for c in ui.flashlight_cones(sim, team=0)} == {b.id}


def test_the_emitter_inputs_name_the_bit():
    """Property: the assembly's input table says where the flashlight's
    inputs come from, including the unit's synced bit. Breaks if the table is
    not updated when a new input joins the cone."""
    assert "flashlight bit" in fl.EMITTER_INPUTS["flashlight"][1]


# ---------------------------------------------------------------------------
# 5 -- the control and the policy
# ---------------------------------------------------------------------------
class _FakeSim:
    def __init__(self, units):
        self.units = units
        self.player_team = 0
        self.calls = []

    def get_unit(self, uid):
        return next((u for u in self.units if u.id == uid), None)

    def __getattr__(self, name):
        if name.startswith(("set_", "clear_", "throw_", "use_", "reload_",
                            "shove_", "melee_")):
            return lambda *a, **k: self.calls.append((name, a))
        raise AttributeError(name)


def _fake_unit(uid, on=True):
    return SimpleNamespace(id=uid, team=0, alive=True, is_zombie=False,
                           flashlight_on=on, grenade_counts=[1, 0, 0])


def _frame(**kw):
    return DeviceFrame(move=None, aim=None, trigger=False, throw=None,
                       use=False, **kw)


def test_the_toggle_flips_on_a_press_and_the_state_is_held_every_frame():
    """Property: a flashlight press darkens, a second press relights, and
    the control holds the state with set_flashlight on EVERY frame (a hold
    verb, as the trigger is). Breaks if the toggle fires on hold, is sent
    only on the edge, or starts a frame late."""
    dc = DirectControl(keys=key_table(CFG))
    sim = _FakeSim([_fake_unit(1)])
    dc._sync_flashlight(sim.units[0])
    seq = []
    for toggle in (False, True, False, False, True, False):
        sim.calls.clear()
        dc._emit(sim, 1, _frame(flashlight_toggle=toggle))
        seq.append([a[1] for n, a in sim.calls if n == "set_flashlight"])
    assert seq == [[False], [True], [True], [True], [False], [False]]


def test_a_newly_possessed_dark_unit_is_not_relit_by_the_control():
    """Property: the control seeds its state from the possessed unit's own
    bit, so possessing a unit whose light is off does not switch it on the
    next frame. Breaks if the control state is independent of the unit."""
    dc = DirectControl(keys=key_table(CFG))
    sim = _FakeSim([_fake_unit(1, on=False)])
    dc._sync_flashlight(sim.units[0])
    dc._emit(sim, 1, _frame())
    assert [a for n, a in sim.calls if n == "set_flashlight"] == [(1, True)]


def test_the_melee_button_sends_the_verb_only_while_it_is_down():
    """Property: a frame with melee set emits melee_intent, a frame without
    emits none; the melee and flashlight keys are no longer inert. Breaks if
    the verb is not wired to the frame, or fires without the button."""
    dc = DirectControl(keys=key_table(CFG))
    sim = _FakeSim([_fake_unit(1)])
    dc._emit(sim, 1, _frame(melee=True))
    assert ("melee_intent", (1,)) in sim.calls
    sim.calls.clear()
    dc._emit(sim, 1, _frame())
    assert not any(n == "melee_intent" for n, _ in sim.calls)
    assert "melee" not in INERT_ACTIONS and "flashlight" not in INERT_ACTIONS


def test_a_policy_row_darkens_exactly_the_unit_whose_off_column_is_set():
    """Property: the action row's ``set_flashlight.off`` column darkens the
    unit it belongs to and a zero column keeps the light on, through the real
    facade. Breaks if the column's sense is inverted (a zero row would darken
    every unit, and every reference-equality digest would move)."""
    import control_policy as cp
    sim = _sim()
    a = _unit(sim, 8, 8)
    b = _unit(sim, 20, 8)
    schema = cp.action_schema()
    plan = cp._Plan(schema)
    rows = cp.zero_rows(2)
    rows[1, schema.index("set_flashlight.off")] = 1
    cp.issue_row(sim, a.id, rows[0], True, plan)
    cp.issue_row(sim, b.id, rows[1], True, plan)
    assert a.flashlight_on is True and b.flashlight_on is False
