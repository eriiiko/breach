"""No firing while winding up a throw (183-no-fire-windup; Erik's ruling of
2026-11-13: "a marine should not be able to fire when throwing a grenade").

The law: while a unit winds up a throw (``throw_release_tick >= 0`` at the
shooting slot, the Grenade kinds row's wind-up) its trigger does nothing --
no round, no beam, no spray burst (a burst under way is interrupted as a
stun interrupts it), no noise, no magazine spend. A held trigger is kept and
fires again from the release tick (slot 1b releases the grenade before slot
4 shoots). The decision is ONE pure read, ``verb_gates.can_fire``, read by
the shooting slot, the sprays, the v1 trigger mask and the fire lanes.

Properties (each test's docstring names the change that must break it):
a winding-up unit fires nothing whatever its archetype and fires again at
the release; a spray burst under way stops at the wind-up; a unit not
winding up fires exactly as with the gate removed; the trigger mask is the
gate (for every decision length) and a masked hold fires nothing; the fire
lanes skip a winding-up unit; the 1j door releases a masked trigger and
never sends one on a whole ship_demo match.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tools", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402

import control_bots_door as door  # noqa: E402
import level_loader  # noqa: E402
from level_loader import LevelData  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation import action_spec_v1 as A  # noqa: E402
from simulation import noise as noise_mod  # noqa: E402
from simulation import team_picture as tp  # noqa: E402
from simulation import verb_gates  # noqa: E402
from simulation.events import (LaserFiredEvent, ShotFiredEvent,  # noqa: E402
                               SprayJetEvent)
from simulation.intents import FP_ONE  # noqa: E402
from simulation.match_session import MatchSession  # noqa: E402
from simulation.ruleset import ContinuousRealtime  # noqa: E402
from simulation.status import STUNNED, apply_status  # noqa: E402
from simulation.turning import place as place_facing  # noqa: E402
from simulation.unit import Unit  # noqa: E402
from simulation.weapons import get_tables  # noqa: E402

TABLES = get_tables()
FRAG = TABLES.hand_grenade_rows.index("grenade_frag")
WINDUP = int(TABLES.weapons.by_name["hand_grenade"].throw_windup_ticks)
FIRE_EVENT = {"k5_carbine": ShotFiredEvent, "lance_3": LaserFiredEvent,
              "dragon_7": SprayJetEvent}


def _arena(h=24, w=48) -> LevelData:
    tm = np.ones((h, w), dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = 4
    return LevelData(name="nfw_arena", version="1", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


def _sim() -> Simulation:
    sim = Simulation(_arena(), seed=7, breach_physics=bp,
                     enable_recorder=False, ruleset=ContinuousRealtime())
    sim.set_paused(False)
    return sim


def _shooter(sim, weapon="k5_carbine", x=8, y=10, team=0):
    uid = sim.add_unit(Unit("S", x=x, y=y, team=team))
    u = sim.get_unit(uid)
    place_facing(u, 0)                                   # east
    u.weapon_id = weapon
    u.grenade_counts[:] = [0] * len(u.grenade_counts)
    u.grenade_counts[FRAG] = 1
    return u


def _fired(sim, u, weapon) -> int:
    t = FIRE_EVENT[weapon]
    return sum(1 for e in sim.tick_events
               if isinstance(e, t) and int(e.unit_id) == int(u.id)
               and getattr(e, "launch", True))


def _noises(monkeypatch, sim):
    heard = []
    real = noise_mod.emit

    def emit(gmap, fy, fx, radius):
        heard.append((int(sim.tick), int(fy), int(fx), int(radius)))
        real(gmap, fy, fx, radius)
    monkeypatch.setattr(noise_mod, "emit", emit)
    return heard


def _gun_state(u):
    return (u.current_mag, u.spare_mags, u.reload_done_tick,
            u.last_fire_tick, u.spray_ticks_left)


@pytest.mark.parametrize("weapon", sorted(FIRE_EVENT))
def test_a_winding_up_unit_fires_nothing_and_fires_again_at_the_release(
        weapon, monkeypatch):
    """Property: a unit holding its trigger while it winds up a throw fires
    nothing on every tick of the wind-up (the tick the throw starts
    included) -- no round / beam / spray jet, no noise, its magazine,
    cadence and burst state untouched, the trigger still held -- whatever
    the archetype (a marcher, a hitscan beam, a spray); on the release tick
    (the grenade leaves at 1b, before slot 4) it fires again with the same
    held trigger. Breaks if the shooting slot ignores the wind-up, the gate
    clears the trigger, or the boundary is one tick late or early."""
    assert WINDUP >= 2
    sim = _sim()
    u = _shooter(sim, weapon)
    heard = _noises(monkeypatch, sim)
    sim.set_trigger(u.id, True)
    sim.throw_grenade_intent(u.id, FP_ONE, 0, 72, kind=FRAG)
    t0 = sim.tick
    for _ in range(WINDUP):
        before = _gun_state(u)
        t = sim.tick
        sim.step()
        assert u.throw_release_tick == t0 + WINDUP, "not winding up"
        assert _fired(sim, u, weapon) == 0, f"fired at tick {t}"
        assert not [n for n in heard if n[0] == t], f"a noise at tick {t}"
        assert _gun_state(u) == before
        assert u.live_trigger
    sim.step()                                          # the release tick
    assert u.throw_release_tick < 0 and u.grenade_counts[FRAG] == 0
    assert _fired(sim, u, weapon) == 1
    assert [n for n in heard if n[0] == t0 + WINDUP]


def test_a_spray_burst_under_way_stops_at_the_wind_up():
    """Property: a spray burst running when a throw's wind-up starts is
    interrupted on that tick (no jet on any wind-up tick, the burst's state
    cleared, not resumed), as a stun interrupts it; the held trigger starts
    a fresh burst at the release. Breaks if the sprays keep depositing
    through a wind-up or resume the old burst."""
    sim = _sim()
    u = _shooter(sim, "dragon_7")
    sim.set_trigger(u.id, True)
    for _ in range(3):
        sim.step()
    assert u.spray_ticks_left > 0 and _fired(sim, u, "dragon_7") == 1
    mag = u.current_mag
    sim.throw_grenade_intent(u.id, FP_ONE, 0, 72, kind=FRAG)
    for _ in range(WINDUP):
        sim.step()
        assert _fired(sim, u, "dragon_7") == 0
        assert u.spray_ticks_left == 0 and u.spray_target is None
    sim.step()
    assert _fired(sim, u, "dragon_7") == 1 and u.spray_ticks_left > 0
    assert u.current_mag == mag - 1                     # a fresh burst


def _trace(gate_on: bool, monkeypatch, weapon):
    if not gate_on:
        monkeypatch.setattr(verb_gates, "can_fire",
                            lambda *a, **k: True)
    sim = _sim()
    u = _shooter(sim, weapon)
    foe = sim.add_unit(Unit("F", x=30, y=10, team=1))
    out = []
    sim.set_trigger(u.id, True)
    for _ in range(90):
        sim.step()
        out.append((sim.tick, _gun_state(u),
                    [repr(e) for e in sim.tick_events],
                    sim.get_unit(foe).hp_q, u.x_q, u.y_q))
    monkeypatch.undo()
    return out


@pytest.mark.parametrize("weapon", sorted(FIRE_EVENT))
def test_a_unit_not_winding_up_fires_exactly_as_without_the_gate(
        weapon, monkeypatch):
    """Property: with no wind-up the gate is transparent: a held trigger
    (every archetype, a target in the line) plays tick for tick -- events,
    magazine, cadence, the target's hp -- as with ``can_fire`` replaced by
    "always". Breaks if the gate reads anything but the wind-up (a cadence,
    a magazine, a status) or the sprays' interruption fires without one."""
    with_gate = _trace(True, monkeypatch, weapon)
    without = _trace(False, monkeypatch, weapon)
    assert any(t[2] for t in with_gate), "vacuous: nothing happened"
    assert with_gate == without


def _mask_row(sim, u, D):
    tb = A.action_tables(sim, 1)
    sizes = tb.sizes
    E = tb.n_entries
    out = {k: np.zeros((1, n), np.uint8) for k, n in zip(A.MASK_KEYS, sizes)}
    A.build_masks(sim, [u], np.zeros((1, E)), np.zeros((1, E)),
                  np.full((1, E), -1), tb, D, out)
    return out["mask_trigger"][0]


def test_the_trigger_mask_is_the_gate_and_a_masked_hold_fires_nothing():
    """Property: at the start of every tick around a throw, for decisions
    of 1, 2, 4 and 8 ticks, the trigger head's release is legal and its
    hold is legal exactly when ``can_fire`` asked for the decision's last
    tick says so -- i.e. iff the wind-up does not outlast the decision
    (``throw_release_tick > T + D - 1`` masks it); with D = 1 a masked hold
    fires nothing that tick (the one-sided promise), and the hold is masked
    on exactly the wind-up's ticks after its first. A stunned unit's
    wind-up (dropped at 1b) masks nothing. Breaks if the mask copies the
    decision instead of calling the gate, ignores the decision's length,
    or ever masks both options."""
    sim = _sim()
    u = _shooter(sim)
    sim.set_trigger(u.id, True)
    sim.throw_grenade_intent(u.id, FP_ONE, 0, 72, kind=FRAG)
    t0 = sim.tick
    masked_ticks = []
    for _ in range(WINDUP + 3):
        T = sim.tick
        for D in (1, 2, 4, 8):
            m = _mask_row(sim, u, D)
            assert m[0] == 1
            gate = verb_gates.can_fire(u, T + D - 1, at_tick_start=True)
            assert m[1] == gate
            assert gate == (u.throw_release_tick <= T + D - 1)
        m1 = _mask_row(sim, u, 1)[1]
        sim.step()
        if not m1:
            masked_ticks.append(T)
            assert _fired(sim, u, "k5_carbine") == 0
    assert masked_ticks == list(range(t0 + 1, t0 + WINDUP))
    # a stunned unit: 1b drops its wind-up, so the hold is not masked
    u2 = _shooter(sim, x=8, y=16)
    sim.throw_grenade_intent(u2.id, FP_ONE, 0, 72, kind=FRAG)
    sim.step()
    assert u2.throw_release_tick > sim.tick
    assert _mask_row(sim, u2, 1)[1] == 0
    apply_status(u2, STUNNED, magnitude=0, duration_ticks=10)
    assert _mask_row(sim, u2, 1)[1] == 1


def test_a_winding_up_unit_draws_no_fire_lane():
    """Property: a unit holding its trigger draws its fire lane, except
    while a wind-up will still keep its hands busy at the coming tick's
    shooting slot; at the start of the release tick it draws it again.
    Breaks if the lanes read ``live_trigger`` alone."""
    sim = _sim()
    u = _shooter(sim)
    sim.set_trigger(u.id, True)
    assert (tp.fire_lanes(sim, 0, 1, 0) == u.id).any()
    sim.throw_grenade_intent(u.id, FP_ONE, 0, 72, kind=FRAG)
    t0 = sim.tick
    sim.step()
    while sim.tick < t0 + WINDUP:
        assert (tp.fire_lanes(sim, 0, 1, 0) == -1).all(), sim.tick
        sim.step()
    assert u.throw_release_tick == sim.tick           # releases at this 1b
    assert (tp.fire_lanes(sim, 0, 1, 0) == u.id).any()


# ---------------------------------------------------------------------------
# The 1j door
# ---------------------------------------------------------------------------
def _door_tables():
    return A.ActionTables(
        n_slots=4, n_entries=37, hand_grenade_rows=("a", "b", "c"),
        throttles=(0, 32768, 65536), fuse_ticks=72,
        reaches_q=tuple((j + 1) * (12 << 16) // 4 for j in range(4)),
        plant_timers=(120, 240, 480, 1440), aim_lo=3, aim_hi=19)


class _U:
    def center_tile_x(self):
        return 10

    def center_tile_y(self):
        return 10


def _slot(tb, trigger_mask, trigger):
    sz = tb.sizes
    m = {"mask_feet": np.ones(sz[0], np.uint8),
         "mask_aim": np.zeros(sz[1], np.uint8),
         "mask_trigger": np.array(trigger_mask, np.uint8),
         "mask_hand": np.zeros(sz[3], np.uint8)}
    m["mask_aim"][:A.AIM_KEEP] = 1
    m["mask_hand"][A.HAND_NOTHING] = 1
    return door.SlotInput(
        unit=_U(), move=None, aim=None, trigger=trigger, window=[],
        list_uid=np.full(tb.n_entries, -1, np.int32),
        list_type=np.zeros(tb.n_entries, np.float32), masks=m, records={},
        tick=8)


def test_the_door_releases_a_masked_trigger_and_counts_it():
    """Property: a captured held trigger whose hold the decision's mask
    forbids is sent as release and counted ``trigger_throw``; a legal hold
    is sent as hold, uncounted. Breaks if the translator sends the bot's
    trigger without reading the mask."""
    tb, st = _door_tables(), door.DoorSettings()
    five, loss = door.translate(_slot(tb, (1, 0), True), tb, st)
    assert five[2] == 0 and loss.bits & door._BIT["trigger_throw"]
    five, loss = door.translate(_slot(tb, (1, 1), True), tb, st)
    assert five[2] == 1 and not loss.bits & door._BIT["trigger_throw"]


def test_the_door_never_sends_a_masked_trigger_on_a_whole_match(
        monkeypatch):
    """Property: on a whole ship_demo door match (seed 4, all-round sight,
    the horde in) at a 2-tick decision -- so a wind-up outlasts whole
    decisions and their trigger masks forbid the hold -- with the bots made
    to hold the trigger on every decision (the worst teacher), the door
    sends no masked choice (``DoorRefused`` would stop the match,
    ``masked_sent`` 0) and the masked holds it met are released. Breaks if
    the translator sends the captured trigger without reading the mask."""
    real = door.translate
    met = {"masked": 0}

    def held(si, tables, settings):
        si = replace(si, trigger=True)
        if not si.masks["mask_trigger"][1]:
            met["masked"] += 1
        five, loss = real(si, tables, settings)
        if not si.masks["mask_trigger"][1]:
            assert five[2] == 0
        return five, loss
    monkeypatch.setattr(door, "translate", held)
    lvl = level_loader.load("ship_demo", str(ROOT / "levels"))
    t0, t1 = lvl.match.teams
    spec = replace(lvl.match, teams=(replace(t0, controller="bots_door"),
                                     replace(t1, controller="bots")),
                   time_limit_s=120.0, sight="all_round")
    ses = MatchSession(lvl, spec, seed=4, breach_physics=bp)
    ctl = [c for c in ses.controllers
           if isinstance(c, door.BotsThroughDoor)]
    assert len(ctl) == 1
    ctl[0].settings = door.DoorSettings(decision_ticks=2)
    while not ses.is_terminal():
        ses.tick()
    st = ctl[0].stats
    assert st["masked_sent"] == 0
    assert met["masked"] > 0, "vacuous: no wind-up outlasted a decision"
    assert st["trigger_throw"] >= met["masked"]
