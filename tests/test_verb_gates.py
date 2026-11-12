"""#195 1e: the hand verbs' gates (``simulation.verb_gates``) are exact reads of
the verbs' own decisions.

The harness: a continuous arena (a marine ``A`` facing east, an enemy ``B``
adjacent in its arc, a door beside a third marine in a second arena); for a
state S set up at tick start, read the gate, issue the verb ALONE, step once,
and read the act from the verb's own event (or the door's ``want_open``
flip). Every test docstring states its property and the change that breaks it;
the oversights (a stunned unit reloads and opens doors) are pinned, so a later
ruling is a deliberate test change.
"""
from __future__ import annotations

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
from field_ab_harness import SIM_FIELDS, _snapshot, capture_snapshot  # noqa: E402
from field_digest import field_digest, tick_digest  # noqa: E402
from level_loader import LevelData  # noqa: E402
from simulation import Simulation, combat, unit_impulse, unit_schema  # noqa: E402
from simulation import verb_gates  # noqa: E402
from simulation.events import (MeleeEvent, PlantEvent, ReloadEvent,  # noqa: E402
                               ShoveEvent, ThrowEvent, UnitHitEvent)
from simulation.factions import ZOMBIE_TEAM  # noqa: E402
from simulation.materials import MAT_AIR, MAT_HULL  # noqa: E402
from simulation.ruleset import ContinuousRealtime  # noqa: E402
from simulation.status import (BLINDED, KNOCKED_DOWN, PARALYZED,  # noqa: E402
                               STUNNED, apply_status, composed_flags,
                               tick_statuses)
from simulation.turning import PI_Q16, place  # noqa: E402
from simulation.unit import Unit  # noqa: E402
from simulation.combat import SPARES_UNTRACKED  # noqa: E402
from simulation.unit_schema import MAG_UNBOUND  # noqa: E402
from simulation.weapons import get_tables  # noqa: E402

ONE = 1 << 16
TABLES = get_tables()
ROF = TABLES.weapons.by_name["rifle_butt"].rof_interval_ticks
WINDUP = TABLES.weapons.by_name["hand_grenade"].throw_windup_ticks
PLANT_TICKS = TABLES.weapons.by_name["c4_satchel"].plant_ticks
RELOAD_TICKS = TABLES.weapons.by_name["k5_carbine"].reload_ticks
MAG = TABLES.weapons.by_name["k5_carbine"].mag_size

_EVENT = {"throw": ThrowEvent, "plant": PlantEvent, "shove": ShoveEvent,
          "melee": MeleeEvent, "reload": ReloadEvent}
_INTENT = {"throw": lambda sim, u: sim.throw_grenade_intent(
               u.id, ONE, 0, 24, 5 * ONE, 0),
           "plant": lambda sim, u: sim.plant_charge_intent(u.id, 0),
           "use": lambda sim, u: sim.use_intent(u.id),
           "reload": lambda sim, u: sim.reload_intent(u.id),
           "shove": lambda sim, u: sim.shove_intent(u.id),
           "melee": lambda sim, u: sim.melee_intent(u.id)}
VERBS = tuple(_INTENT)


def _gate(verb, sim, u):
    """The gate as the mask reads it, at the start of the tick."""
    if verb == "throw":
        return verb_gates.can_throw(sim, u, 0)
    if verb == "melee":
        return verb_gates.can_melee(u, sim.tick, sim.weapons_tables,
                                    at_tick_start=True)
    return getattr(verb_gates, "can_" + verb)(sim, u)


def _level(door=False):
    from test_horde_behaviours import _door_inst
    h, w = 20, 40
    tm = np.full((h, w), MAT_HULL, dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = MAT_AIR
    ents = []
    if door:
        tm[1:h - 1, 20] = MAT_HULL
        tm[4:7, 20] = MAT_AIR
        ents = [_door_inst("d", 0, 20, 4)]
    return LevelData(name="gates", version="3", path=Path("."), tilemap=tm,
                     tile_size_m=1.0, diffuse_path=Path("."), entities=ents)


class World:
    """``a`` (a rifleman, team 0, facing east) and ``b`` (team 1) one
    footprint to its east -- in reach and arc of a shove and a blow -- on a
    sim with no physics (the verbs read none). With ``door`` the pair stands
    where ``a`` is beside the interior door (b is out of the way)."""

    def __init__(self, door=False, *, b=True, seed=5, physics=False):
        self.sim = Simulation(_level(door), seed=seed,
                              breach_physics=bp if physics else None,
                              enable_recorder=False,
                              ruleset=ContinuousRealtime())
        self.sim.set_paused(False)
        self.a = self.add("A", 17 if door else 10, 4 if door else 8, 0, 0)
        self.b = (self.add("B", 25 if door else 13, 4 if door else 8, 1, PI_Q16)
                  if b else None)

    def add(self, name, x, y, team, facing):
        u = self.sim.get_unit(self.sim.add_unit(Unit(name, x=x, y=y, team=team)))
        place(u, facing)
        if not u.is_zombie:
            u.current_hp = 10000.0
        return u

    def step(self, n=1):
        evs = []
        for _ in range(n):
            self.sim.set_paused(False)
            self.sim.step()
            evs += list(self.sim.tick_events)
        return evs

    def issue_and_step(self, verb, u=None):
        """Issue ``verb`` for ``u`` alone, step once; True iff it acted."""
        u = u or self.a
        door = self.sim._doors[0] if verb == "use" else None
        was = None if door is None else bool(door.want_open)
        _INTENT[verb](self.sim, u)
        evs = self.step()
        if verb == "use":
            return bool(door.want_open) != was
        return any(isinstance(e, _EVENT[verb]) and e.unit_id == u.id
                   for e in evs)


# ---------------------------------------------------------------------------
# the states the sweep walks (each: name, fn(world))
# ---------------------------------------------------------------------------
def _stunned(k):
    return lambda w: apply_status(w.a, STUNNED, 0, k)


def _winding(left):
    """A throw's wind-up that releases ``left`` ticks from now (left >= 0)."""
    def go(w):
        w.sim.throw_grenade_intent(w.a.id, ONE, 0, 24, 5 * ONE, 0)
        w.step(1 + WINDUP - 1 - left)
        assert w.a.throw_release_tick - w.sim.tick == left
    return go


def _planting(left):
    def go(w):
        w.sim.plant_charge_intent(w.a.id, 0)
        w.step(1 + PLANT_TICKS - 1 - left)
        assert w.a.plant_done_tick - w.sim.tick == left
    return go


def _cooled(attr, d):
    """The unit last shoved / struck ``d`` ticks before the tick start."""
    def go(w):
        w.step(40)                              # tick > d: a real last tick
        setattr(w.a, attr, w.sim.tick - d)
    return go


def _bound(rounds=MAG, spare=2):
    def go(w):
        combat.mag_bind(w.a, TABLES.weapons.by_name["k5_carbine"])
        w.a.current_mag = rounds
        w.a.spare_mags = spare
    return go


def _stall(left, spare=1):
    """A reload stall with ``left`` ticks to run (0 = it ends this tick)."""
    def go(w):
        w.step(3)                               # tick > 0 for a negative left
        _bound(0, spare)(w)
        w.a.reload_done_tick = w.sim.tick + left
    return go


def _gone(w):
    w.a.grenade_counts[0] = 0


def _no_charge(w):
    w.a.has_explosive = 0


def _remaining_zero_stun(w):
    """A stun that the coming sweep releases: ``remaining_ticks`` 0 at the
    start of the tick (what a slot-2 effect of duration 1 leaves one tick
    later)."""
    st = apply_status(w.a, STUNNED, 0, 1)
    st.remaining_ticks = 0


def _none(w):
    pass


STATES = {
    "throw": [("rest", _none), ("stunned 3", _stunned(3)),
              ("stunned remaining 0", _remaining_zero_stun),
              ("winding, release in 3", _winding(3)),
              ("winding, release now", _winding(0)),
              ("planting, done in 3", _planting(3)),
              ("planting, done now", _planting(0)), ("count 0", _gone)],
    "plant": [("rest", _none), ("stunned 3", _stunned(3)),
              ("stunned remaining 0", _remaining_zero_stun),
              ("winding, release in 3", _winding(3)),
              ("winding, release now", _winding(0)),
              ("planting, done in 3", _planting(3)),
              ("planting, done now", _planting(0)),
              ("no charge", _no_charge)],
    "use": [("rest", _none), ("stunned 3", _stunned(3)),
            ("three tiles from the door", lambda w: w.a.set_pos(8.0, w.a.y))],
    "reload": [("never bound (full)", _none), ("bound, full", _bound()),
               ("bound, partial, 2 spare", _bound(10, 2)),
               ("bound, partial, untracked spares",
                _bound(10, SPARES_UNTRACKED)),
               ("bound, partial, no spare", _bound(10, 0)),
               ("dry, no spare", _bound(0, 0)),
               ("stall, 5 left", _stall(5)),
               ("stall, ends now", _stall(0)),
               ("stall ran out a tick ago", _stall(-1)),
               ("stunned, partial", lambda w: (_bound(10, 2)(w),
                                               _stunned(3)(w))),
               ("winding, partial", lambda w: (_bound(10, 2)(w),
                                               _winding(3)(w)))],
    "melee": [("rest", _none), ("stunned 3", _stunned(3)),
              ("stunned remaining 1", lambda w: apply_status(
                  w.a, STUNNED, 0, 1)),
              ("stunned remaining 0", _remaining_zero_stun),
              ("winding, release in 3", _winding(3)),
              ("winding, release now", _winding(0)),
              ("planting, done in 3", _planting(3)),
              ("planting, done now", _planting(0)),
              ("last blow rof-1 ago", _cooled("last_blow_tick", ROF - 1)),
              ("last blow rof ago", _cooled("last_blow_tick", ROF))],
}


def _shove_cd(w):
    return w.sim.impulse_table.shove_cooldown_ticks


STATES["shove"] = [
    ("rest", _none), ("stunned 3", _stunned(3)),
    ("stunned remaining 0", _remaining_zero_stun),
    ("last shove cd-1 ago",
     lambda w: _cooled("last_shove_tick", _shove_cd(w) - 1)(w)),
    ("last shove cd ago",
     lambda w: _cooled("last_shove_tick", _shove_cd(w))(w))]


def _cases():
    return [(v, name, fn) for v in VERBS for name, fn in STATES[v]]


@pytest.mark.parametrize("verb,name,setup", _cases(),
                         ids=[f"{v}:{n}" for v, n, _ in _cases()])
def test_the_gate_at_tick_start_is_whether_the_verb_acted(verb, name, setup):
    """Property: for every verb, over a sweep of states with no forbidding
    event in the step, the gate read at the start of the tick equals whether
    the verb issued alone then acted (its own event, or the door flip). Breaks
    if a gate term differs from the verb's decision, in either direction."""
    w = World(door=(verb == "use"))
    setup(w)
    g = _gate(verb, w.sim, w.a)
    acted = w.issue_and_step(verb)
    assert g == acted, f"{verb} / {name}: gate {g}, acted {acted}"


def test_the_sweep_reaches_both_answers_for_every_verb():
    """Property: the sweep above is not vacuous -- each verb is walked
    through at least one state where it acts and one where it does not.
    Breaks if a state builder stops building its state."""
    for verb in VERBS:
        seen = set()
        for name, fn in STATES[verb]:
            w = World(door=(verb == "use"))
            fn(w)
            seen.add(_gate(verb, w.sim, w.a))
        assert seen == {True, False}, verb


# ---------------------------------------------------------------------------
def test_reload_gate_reads_the_settled_magazine_status():
    """Property: the reload gate is False on an untracked weapon, a full
    magazine, a never-bound magazine (full), during the stall and on its last
    tick (the settle then makes it full), and dry with no spare; True on a
    partial magazine with counted spares and with untracked spares. Breaks if
    the gate reads the raw fields instead of ``mag_status``, or reads "a
    counted spare" (the design's wording) where the code also reloads with
    untracked spares."""
    def gate(setup, **kw):
        w = World(b=False)
        for k, v in kw.items():
            setattr(w.a, k, v)
        setup(w)
        return verb_gates.can_reload(w.sim, w.a)

    assert gate(_none, weapon_id="lance_3") is False        # untracked weapon
    assert gate(_none) is False                             # never bound, full
    assert gate(_bound()) is False                          # full
    assert gate(_stall(5)) is False                         # the stall
    assert gate(_stall(0)) is False                         # its last tick
    assert gate(_bound(0, 0)) is False                      # dry, no spare
    assert gate(_bound(10, 0)) is False                     # no spare
    assert gate(_bound(10, 2)) is True                      # counted spares
    assert gate(_bound(10, SPARES_UNTRACKED)) is True                     # untracked spares
    assert gate(_none, magazines=0) is False                # untracked, full


def test_every_gate_is_pure():
    """Property: calling any gate leaves the unit's schema record, its
    statuses and the tick's events as they were -- on a never-bound magazine
    and on a finished stall included (the verb's settle is the verb's, not
    the gate's). Breaks if a gate settles, binds, spends a cool-down, or
    emits."""
    for setup in (_none, _stall(-1), _bound(10, 2), _winding(3),
                  _planting(3), _stunned(3)):
        w = World()
        setup(w)
        w.sim.tick_events.clear()
        before = (unit_schema.digest_record(w.a),
                  [(s.kind, s.remaining_ticks) for s in w.a.statuses],
                  list(w.sim.tick_events), w.a.current_mag,
                  w.a.reload_done_tick)
        for v in VERBS:
            _gate(v, w.sim, w.a)
        verb_gates.use_door(w.sim, w.a)
        for kind in range(len(TABLES.hand_grenade_rows)):
            verb_gates.can_throw(w.sim, w.a, kind)
        verb_gates.can_melee(w.a, w.sim.tick, w.sim.weapons_tables,
                             at_tick_start=False)
        after = (unit_schema.digest_record(w.a),
                 [(s.kind, s.remaining_ticks) for s in w.a.statuses],
                 list(w.sim.tick_events), w.a.current_mag,
                 w.a.reload_done_tick)
        assert before == after


def test_a_plant_on_a_wind_ups_release_tick_acts_and_its_gate_says_so():
    """Property: a wind-up releasing on this very tick has let go of the hand
    by the plant's consumption, so the plant gate is True at tick start and
    the plant acts. Breaks if the plant gate reads ``throw_release_tick >=
    0`` (the mask would forbid a plant the game performs, against the
    one-sided promise)."""
    w = World(b=False)
    _winding(0)(w)
    assert w.a.throw_release_tick == w.sim.tick
    assert verb_gates.can_plant(w.sim, w.a)
    assert w.issue_and_step("plant")
    assert w.a.plant_done_tick >= 0 and w.a.throw_release_tick == -1


def test_a_throw_on_a_plants_last_tick_and_a_wind_ups_release_tick_is_refused():
    """Property: the throw is decided before the advances, so on the tick a
    plant ends or a wind-up releases its gate is False and nothing new is
    thrown. Breaks if the throw gate reads ``> tick`` like the plant's."""
    w = World(b=False)
    _planting(0)(w)
    assert not verb_gates.can_throw(w.sim, w.a, 0)
    assert not w.issue_and_step("throw")
    w = World(b=False)
    _winding(0)(w)
    assert not verb_gates.can_throw(w.sim, w.a, 0)
    n = w.a.grenade_counts[0]
    evs = w.step()
    assert not any(isinstance(e, ThrowEvent) for e in evs)
    assert w.a.grenade_counts[0] == n - 1       # the old wind-up's release


def test_a_unit_whose_stun_reaches_zero_blows_but_does_not_shove():
    """Property: on the tick a STUN reaches ``remaining_ticks == 0`` at its
    start, slot 1b (shove) still sees it and slot 4a (melee, after the 2b
    sweep) does not: the shove gate is False and nothing is shoved; the melee
    gate read at tick start is True and the blow lands (a hit on B). Breaks
    if the mask reads the 4a moment wrong, or the shove gate projects the
    sweep."""
    w = World()
    apply_status(w.a, STUNNED, 0, 3)
    w.step(3)
    assert [s.remaining_ticks for s in w.a.statuses] == [0]
    assert not verb_gates.can_shove(w.sim, w.a)
    assert verb_gates.can_melee(w.a, w.sim.tick, w.sim.weapons_tables,
                                at_tick_start=True)
    assert not verb_gates.can_melee(w.a, w.sim.tick, w.sim.weapons_tables,
                                    at_tick_start=False)
    w.sim.shove_intent(w.a.id)
    w.sim.melee_intent(w.a.id)
    evs = w.step()
    assert not any(isinstance(e, ShoveEvent) for e in evs)
    assert any(isinstance(e, MeleeEvent) for e in evs)
    assert any(isinstance(e, UnitHitEvent) and e.unit_id == w.b.id
               for e in evs)


def test_melee_reads_the_advance_at_slot_1b():
    """Property: a wind-up releasing at T, a plant ending at T, and a wind-up
    DROPPED at 1b by a status the 2b sweep releases, all leave the hands free
    for the blow at 4a: the mask's gate is True and the blow lands. Breaks if
    the busy terms ignore what ``_advance_throw`` / ``_advance_plant`` do at
    1b."""
    for setup in (_winding(0), _planting(0)):
        w = World()
        setup(w)
        assert verb_gates.can_melee(w.a, w.sim.tick, w.sim.weapons_tables,
                                    at_tick_start=True)
        assert w.issue_and_step("melee")
    w = World()
    _winding(3)(w)
    _remaining_zero_stun(w)
    assert verb_gates.can_melee(w.a, w.sim.tick, w.sim.weapons_tables,
                                at_tick_start=True)
    assert w.issue_and_step("melee")
    assert w.a.throw_release_tick == -1                 # dropped at 1b


def test_a_gate_that_says_no_at_tick_start_means_nothing_happens():
    """Property (the one-sided promise): with a slot-1a knockdown (a pending
    impulse over the fall limit), gate False => the verb does nothing; and a
    gate True may still do nothing (a throw, plant, shove or blow by a unit
    the impulse knocks down). Breaks if a gate allows less than the game in
    some state."""
    allowed_to_fail = {"throw", "plant", "shove", "melee"}
    saw_true_no_act = set()
    for verb in VERBS:
        for name, setup in STATES[verb]:
            w = World(door=(verb == "use"))
            setup(w)
            g = _gate(verb, w.sim, w.a)
            unit_impulse.add(w.sim.impulse_table, w.a, 0, w.a.mass_g * 4 * ONE)
            acted = w.issue_and_step(verb)
            assert any(s.kind == KNOCKED_DOWN for s in w.a.statuses)
            if not g:
                assert not acted, f"{verb} / {name}: gate False but acted"
            elif not acted:
                saw_true_no_act.add(verb)
    assert saw_true_no_act >= allowed_to_fail
    assert "reload" not in saw_true_no_act      # the oversight: it still acts


def test_cooldowns_at_their_boundary_and_a_miss_spends_none():
    """Property: shove at ``tick - last == cooldown - 1`` False, ``==
    cooldown`` True; melee the same at ``rof_interval_ticks``; a shove or a
    blow with no one in reach spends neither. Breaks on an off-by-one, or if a
    miss spends a cool-down."""
    cd = World().sim.impulse_table.shove_cooldown_ticks
    for d, want in ((cd - 1, False), (cd, True)):
        w = World()
        _cooled("last_shove_tick", d)(w)
        assert verb_gates.can_shove(w.sim, w.a) is want
        assert w.issue_and_step("shove") is want
    for d, want in ((ROF - 1, False), (ROF, True)):
        w = World()
        _cooled("last_blow_tick", d)(w)
        assert _gate("melee", w.sim, w.a) is want
        assert w.issue_and_step("melee") is want
    w = World(b=False)                                  # nobody in reach
    last_s, last_m = w.a.last_shove_tick, w.a.last_blow_tick
    assert verb_gates.can_shove(w.sim, w.a) and _gate("melee", w.sim, w.a)
    assert not w.issue_and_step("shove") and not w.issue_and_step("melee")
    assert (w.a.last_shove_tick, w.a.last_blow_tick) == (last_s, last_m)


def test_the_two_oversights_are_kept():
    """Property (Erik's to rule on, design 11): a stunned unit's reload and
    door use are legal -- both gates True and both verbs act -- and the
    reload also ignores a wind-up and a plant. Breaks if the oversights are
    'fixed' without Erik's ruling."""
    w = World(b=False)
    _bound(10, 2)(w)
    _stunned(5)(w)
    assert not composed_flags(w.a).can_act
    assert verb_gates.can_reload(w.sim, w.a) and w.issue_and_step("reload")
    w = World(door=True, b=False)
    _stunned(5)(w)
    assert verb_gates.can_use(w.sim, w.a) and w.issue_and_step("use")
    for busy in (_winding(3), _planting(3)):
        w = World(b=False)
        _bound(10, 2)(w)
        busy(w)
        assert verb_gates.can_reload(w.sim, w.a) and w.issue_and_step("reload")


@pytest.mark.parametrize("verb", VERBS)
@pytest.mark.parametrize("who", ["dead", "zombie"])
def test_the_dead_and_the_zombie_have_every_gate_false_and_never_act(who, verb):
    """Property: a dead unit and a zombie are refused by every gate and act
    on nothing, in a state where a living marine's gate is True. Breaks if
    the 1b envelope (alive, not a zombie) is missing from a gate."""
    def build(kind):
        w = World(door=(verb == "use"))
        _bound(10, 2)(w)
        if kind == "zombie":
            w.a.team = ZOMBIE_TEAM
            w.a.is_zombie = True
        elif kind == "dead":
            w.a.alive = False
        return w
    living = World(door=(verb == "use"))
    _bound(10, 2)(living)
    assert _gate(verb, living.sim, living.a), "the control must be True"
    w = build(who)
    assert w.a.is_zombie == (who == "zombie")
    assert not _gate(verb, w.sim, w.a)
    assert not w.issue_and_step(verb)


def test_can_throw_refuses_an_unknown_kind_by_name():
    """Property: ``can_throw`` refuses what the verb refuses -- a kind outside
    the hand-grenade family, by name -- instead of indexing the counts.
    Breaks if the gate accepts it (or raises an IndexError)."""
    w = World(b=False)
    for bad in (-1, len(TABLES.hand_grenade_rows), True):
        with pytest.raises(ValueError, match="unknown grenade kind"):
            verb_gates.can_throw(w.sim, w.a, bad)


def test_the_sweep_predicate_is_stated_once():
    """Property: ``composed_flags(u, after_sweep=True)`` equals the flags of
    the same unit after one ``tick_statuses`` (no damage-over-time kinds), over
    every kind and remaining count. Breaks if the sweep rule is stated twice
    and the copies drift."""
    kinds = (STUNNED, PARALYZED, KNOCKED_DOWN, BLINDED)
    for k1 in kinds:
        for r1 in (0, 1, 2):
            for k2 in kinds:
                for r2 in (0, 1, 3):
                    w = World(b=False)
                    for k, r in ((k1, r1), (k2, r2)):
                        st = apply_status(w.a, k, 0, max(1, r))
                        st.remaining_ticks = r
                    predicted = composed_flags(w.a, after_sweep=True)
                    tick_statuses([w.a])
                    assert composed_flags(w.a) == predicted


@pytest.mark.parametrize("verb", VERBS)
def test_the_verb_is_its_gate(monkeypatch, verb):
    """Property: each verb's decision IS its gate -- with the gate forced
    False the verb does nothing in a state where it would act; forced True
    in a state where it would not, the verb carries on to its own act (or a
    law the gate does not own: a target). Breaks if a verb keeps a private
    copy of the decision."""
    name = {"throw": "can_throw", "plant": "can_plant", "use": "can_use",
            "reload": "can_reload", "shove": "can_shove",
            "melee": "can_melee"}[verb]
    if verb == "use":
        name = "use_door"                       # the verb flips what it finds
    w = World(door=(verb == "use"))
    if verb == "reload":
        _bound(10, 2)(w)
    if verb == "use":
        monkeypatch.setattr(verb_gates, name, lambda sim, u: None)
    elif verb == "throw":
        monkeypatch.setattr(verb_gates, name, lambda sim, u, k: False)
    elif verb == "melee":
        monkeypatch.setattr(verb_gates, name, lambda *a, **k: False)
    else:
        monkeypatch.setattr(verb_gates, name, lambda sim, u: False)
    assert not w.issue_and_step(verb)


def test_a_throw_kind_and_a_plant_spend_through_their_own_laws():
    """Property: the gates leave the verbs' spending where it was -- a
    gate-passed throw spends its count at the RELEASE and a plant its charge
    at the plant's last tick, not at the gate. Breaks if a gate (or a verb
    moved onto it) spends early."""
    w = World(b=False)
    n, c = w.a.grenade_counts[0], w.a.has_explosive
    assert verb_gates.can_throw(w.sim, w.a, 0)
    w.sim.throw_grenade_intent(w.a.id, ONE, 0, 24, 5 * ONE, 0)
    w.step(WINDUP)                              # the release tick is the next
    assert w.a.grenade_counts[0] == n
    w.step(1)
    assert w.a.grenade_counts[0] == n - 1
    w2 = World(b=False)
    w2.sim.plant_charge_intent(w2.a.id, 0)
    w2.step(PLANT_TICKS)
    assert w2.a.has_explosive == c
    w2.step(1)
    assert w2.a.has_explosive == c - 1


# ---------------------------------------------------------------------------
# way (a): a refused reload writes nothing (the lazy settle is deferred)
# ---------------------------------------------------------------------------
def _digests(w):
    return (field_digest(_snapshot(w.sim.gmap, SIM_FIELDS)),
            tick_digest(capture_snapshot(w.sim)))


def _fresh(w):
    """Match start: the magazine was never bound."""
    assert w.a.current_mag == MAG_UNBOUND


def _ran_out(w):
    """Right after a stall has run out: fire to bind, reload, let the stall
    end with nothing touching the unit -- the finished stall is still
    unsettled (``reload_done_tick`` stale) at this tick start."""
    w.sim.set_aim(w.a.id, 0, -ONE)
    w.sim.set_trigger(w.a.id, True)
    w.step(3)
    w.sim.set_trigger(w.a.id, False)
    w.sim.reload_intent(w.a.id)
    w.step(1)
    w.step(RELOAD_TICKS)                       # the stall's last tick passed
    assert 0 <= w.a.reload_done_tick <= w.sim.tick
    assert w.a.current_mag == 0                # ... and nothing settled it


@pytest.mark.parametrize("start", [_fresh, _ran_out],
                         ids=["match start", "after a stall ran out"])
def test_a_refused_reload_by_force_leaves_the_digests_where_a_do_nothing_tick_does(
        start):
    """Property (the 1h gate, way (a)): a reload issued by force that the
    gate refuses -- the gun full and never bound, or full by a stall that has
    run out -- leaves ``field_digest`` AND ``tick_digest`` where a twin that
    did nothing leaves them, every tick of the step and after. Breaks if the
    verb settles the magazine before it knows the reload is legal (the lazy
    bind or the completed stall then moves the unit record)."""
    forced, idle = World(physics=True), World(physics=True)
    for w in (forced, idle):
        start(w)
    assert not verb_gates.can_reload(forced.sim, forced.a)
    assert _digests(forced) == _digests(idle)
    forced.sim.reload_intent(forced.a.id)
    for _ in range(4):
        forced.step()
        idle.step()
        assert _digests(forced) == _digests(idle)
    rec = unit_schema.digest_record
    assert rec(forced.a) == rec(idle.a)


@pytest.mark.parametrize("start", [_fresh, _ran_out],
                         ids=["match start", "after a stall ran out"])
def test_the_deferred_settle_still_happens(start):
    """Property: deferring the settle changes nothing a player, a bot or an
    agent can observe -- ``mag_status`` reads the settled magazine before and
    after the refused reload alike, and the next trigger pull leaves the
    unit exactly where the old eager settle then the pull left it (a twin
    that settled by hand at the reload's tick, as the old verb did), the
    round count and the spares included. Breaks if the trigger no longer
    settles lazily, or ``mag_status`` stops reading the settled view."""
    w, old = World(physics=True), World(physics=True)
    for x in (w, old):
        start(x)
    weapon = TABLES.weapons.by_name["k5_carbine"]
    status = combat.mag_status(w.a, weapon, w.sim.tick)
    assert status[0] == MAG and status[2] is False
    w.sim.reload_intent(w.a.id)
    old.sim.reload_intent(old.a.id)
    combat.mag_settle(old.a, weapon, old.sim.tick)   # the old verb's write
    w.step()
    old.step()
    assert combat.mag_status(w.a, weapon, w.sim.tick) == status
    assert combat.mag_status(old.a, weapon, old.sim.tick) == status
    for x in (w, old):                               # the next trigger pull
        x.sim.set_aim(x.a.id, 0, -ONE)
        x.sim.set_trigger(x.a.id, True)
    for _ in range(6):
        w.step()
        old.step()
        assert (combat.mag_status(w.a, weapon, w.sim.tick)
                == combat.mag_status(old.a, weapon, old.sim.tick))
    assert w.a.current_mag < MAG and w.a.reload_done_tick == -1
    assert (unit_schema.digest_record(w.a)
            == unit_schema.digest_record(old.a))
    assert _digests(w) == _digests(old)
