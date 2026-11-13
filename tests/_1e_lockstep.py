"""#195 1e's LOCKSTEP driver: the verbs' gates are a refactor, so the game must
not move -- the same scenarios, run on two trees, must write the same tape.

Imports ``_p3a_sole_cause``'s four scenarios (``ship_demo`` seed 1, the same
with ``bots_v1``, the #122 conversion box, the scripted arena) and adds
``gates``: EVERY hand verb driven directly on a level with an interior door,
reaching every refusal edge the 1e note lists at least once and every act at
least once. Per tick the tape records ``(tick, schema unit records, planted
charges, synced events, field_digest, tick_digest)`` -- ``tick_digest`` over
``field_ab_harness.capture_snapshot`` (fields, units, charges, entities: a
door's ``want_open`` is hashed), so the door verb is covered too.

    <py> tests/_1e_lockstep.py OUT.pkl [scenario ...]
    <py> tests/_1e_lockstep.py --compare A.pkl B.pkl

``--compare`` says EQUAL per scenario, or where the tapes differ: the first
differing tick, the unit-record keys that differ (and over how many ticks),
and whether EVERY ``field_digest`` is still identical -- how way (a)'s commit
(the reload settles only when the gate says yes) is shown to move the
magazine columns only.

The ``gates`` scenario prints a table per verb: issues, acts (by the verb's
own event, or a door's ``want_open`` flip), and every labelled edge whose act
differed from what the note's table says it should be (an empty list = the
script reads the decision as the note does).
"""
from __future__ import annotations

import collections
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tools", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

import _p3a_sole_cause as p3a  # noqa: E402
import breach_physics as bp  # noqa: E402
from field_ab_harness import capture_snapshot  # noqa: E402
from field_digest import tick_digest  # noqa: E402

_TICK_FIELDS = ("tick", "units", "charges", "events", "field_digest",
                "tick_digest")


class _Tape(p3a._Tape):
    """p3a's tape plus the tick digest of the whole snapshot."""

    def __call__(self, sim):
        super().__call__(sim)
        tick, recs, charges, evs, fd = self.ticks.pop()
        self.ticks.append((tick, recs, charges, evs, fd,
                           tick_digest(capture_snapshot(sim))))


p3a._Tape = _Tape


# ---------------------------------------------------------------------------
# the `gates` scenario
# ---------------------------------------------------------------------------
_INTENT = {"throw": "throw_grenade_intent", "plant": "plant_charge_intent",
           "use": "use_intent", "reload": "reload_intent",
           "shove": "shove_intent", "melee": "melee_intent"}
_VERB_EVENT = {"throw": "ThrowEvent", "plant": "PlantEvent",
               "shove": "ShoveEvent", "melee": "MeleeEvent",
               "reload": "ReloadEvent"}


def _gates_level():
    """A 40 x 24 room split by a wall at x = 20 with a 3-tile closed door in
    it (rows 4..6): the left room for use / reload / throw / plant, the right
    for the shove and melee pairs (against the east wall, so a push never
    breaks the contact)."""
    from level_loader import LevelData
    from simulation.materials import MAT_AIR, MAT_HULL
    from test_horde_behaviours import _door_inst
    h, w = 24, 40
    tm = np.full((h, w), MAT_HULL, dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = MAT_AIR
    tm[1:h - 1, 20] = MAT_HULL
    tm[4:7, 20] = MAT_AIR
    return LevelData(name="gates", version="3", path=Path("."), tilemap=tm,
                     tile_size_m=1.0, diffuse_path=Path("."),
                     entities=[_door_inst("d", 0, 20, 4)])


def scenario_gates(n_ticks=200):
    from simulation import Simulation
    from simulation.factions import ZOMBIE_TEAM
    from simulation.ruleset import ContinuousRealtime
    from simulation.status import STUNNED, apply_status
    from simulation.turning import PI_Q16, place
    from simulation.unit import Unit
    from simulation.weapons import get_tables

    sim = Simulation(_gates_level(), seed=3, breach_physics=bp,
                     enable_recorder=False, ruleset=ContinuousRealtime())
    sim.set_paused(False)
    tables = get_tables()
    rof = tables.weapons.by_name["rifle_butt"].rof_interval_ticks
    one = 1 << 16
    east, west, north = 0, PI_Q16, -(PI_Q16 // 2)
    units = {}

    def add(name, x, y, team, facing):
        u = sim.get_unit(sim.add_unit(Unit(name, x=x, y=y, team=team)))
        place(u, facing)
        if not u.is_zombie:
            u.current_hp = 10000.0
        units[name] = u
        return u

    # --- the left room: use, reload, throw, plant -------------------------
    add("U", 17, 4, 0, east)         # beside the door
    add("Ufar", 8, 4, 0, east)       # three tiles from anything
    for name, x, y in (("R1", 2, 8), ("R2", 6, 8), ("R3", 10, 8),
                       ("R4", 14, 8), ("R5", 2, 12), ("R6", 6, 12)):
        add(name, x, y, 0, north)
    add("T1", 10, 16, 0, west)       # the thrower
    add("TP", 2, 20, 0, west)        # plants, then is asked to throw
    add("PW", 6, 20, 0, west)        # throws, then is asked to plant
    # --- the right room: shove and melee pairs against the east wall ------
    for name, y in (("S", 1), ("M1", 5), ("M2", 9), ("M3", 13), ("M4", 17)):
        add(name, 33, y, 0, east)
        add("X" + name, 36, y, 1, west)
    add("W", 26, 12, 0, east)        # alone: its blows and shoves whiff
    # --- the dead and the zombie -----------------------------------------
    d = add("D", 17, 7, 0, east)     # beside the door, then dead
    d.alive = False
    add("Zr", 21, 4, ZOMBIE_TEAM, east)   # beside the door, dormant
    # R5 carries untracked spares (a loadout without `magazines`); R6 holds
    # a weapon with no magazine at all.
    units["R5"].magazines = 0
    units["R6"].weapon_id = "lance_3"
    for k in range(2, len(tables.hand_grenade_rows)):
        units["T1"].grenade_counts[k] = 1      # one of every kind

    sched = collections.defaultdict(list)
    log = []                         # (label, verb, uid, expect, acted)
    pend = []                        # this tick's issues, judged after step
    issues = collections.Counter()

    def at(t, fn):
        sched[t].append(fn)

    def issue(t, label, verb, name, expect, *args):
        def go():
            u = units[name]
            if verb == "use":
                door = sim._doors[0]
                pend.append((label, verb, int(u.id), expect, door,
                             bool(door.want_open)))
                sim.use_intent(u.id)
            else:
                pend.append((label, verb, int(u.id), expect, None, None))
                getattr(sim, _INTENT[verb])(u.id, *args)
            issues[verb] += 1
        at(t, go)

    def throw(t, label, name, kind, expect):
        issue(t, label, "throw", name, expect, 0, one, 24, 5 * one, kind)
        # the direction of a throw is the facing at release; (0, one) only
        # serves a 0-tick wind-up

    def fire(name, t0, t1, aim):
        at(t0, lambda: sim.set_aim(units[name].id, *aim))
        at(t0, lambda: sim.set_trigger(units[name].id, True))
        at(t1, lambda: sim.set_trigger(units[name].id, False))

    def stun(t, name, ticks):
        at(t, lambda: apply_status(units[name], STUNNED, 0, ticks))

    # use ------------------------------------------------------------------
    issue(5, "use beside the door (opens)", "use", "U", True)
    issue(10, "use three tiles away", "use", "Ufar", False)
    issue(40, "use beside the door (closes)", "use", "U", True)
    stun(60, "U", 20)
    issue(62, "use while stunned (the oversight)", "use", "U", True)

    # reload ---------------------------------------------------------------
    issue(3, "reload never-bound full", "reload", "R1", False)
    fire("R2", 3, 6, (0, -one))      # a few rounds bind and part-empty it
    issue(10, "reload partial with spares", "reload", "R2", True)
    issue(20, "reload during the stall", "reload", "R2", False)

    def after_r2():
        done = int(units["R2"].reload_done_tick)
        issue(done - 1, "reload one tick before the stall ends", "reload",
              "R2", False)
        issue(done, "reload on the stall's last tick", "reload", "R2", False)
    at(11, after_r2)

    fire("R3", 3, 4, (0, -one))
    at(6, lambda: (setattr(units["R3"], "spare_mags", 0),
                   setattr(units["R3"], "current_mag", 0)))
    issue(8, "reload dry with no spare", "reload", "R3", False)
    fire("R4", 3, 4, (0, -one))
    stun(8, "R4", 30)
    issue(10, "reload while stunned (the oversight)", "reload", "R4", True)
    fire("R5", 3, 4, (0, -one))
    issue(10, "reload with untracked spares", "reload", "R5", True)
    issue(10, "reload an untracked weapon", "reload", "R6", False)

    # throw ----------------------------------------------------------------
    throw(5, "throw frag (acts)", "T1", 0, True)
    throw(8, "throw during a wind-up", "T1", 0, False)
    throw(12, "throw on a wind-up's release tick", "T1", 0, False)
    throw(14, "throw frag again (acts)", "T1", 0, True)
    throw(30, "throw a kind at count 0", "T1", 0, False)
    for k in range(1, len(tables.hand_grenade_rows)):
        throw(70 + 10 * k, f"throw kind {k} ({tables.hand_grenade_rows[k]})",
              "T1", k, True)
    throw(130, "throw smoke at count 0", "T1", 1, False)
    issue(5, "plant (acts)", "plant", "TP", True, 0)
    throw(10, "throw during a plant", "TP", 0, False)
    issue(12, "plant during a plant", "plant", "TP", False, 0)
    throw(29, "throw on a plant's last tick", "TP", 0, False)
    throw(30, "throw after the plant (acts)", "TP", 0, True)
    issue(40, "plant with no charge left", "plant", "TP", False, 0)

    # plant ----------------------------------------------------------------
    throw(5, "throw (acts), then a plant asks", "PW", 0, True)
    issue(8, "plant during a wind-up", "plant", "PW", False, 0)
    issue(12, "plant on a wind-up's release tick", "plant", "PW", True, 0)
    issue(20, "plant during a plant", "plant", "PW", False, 0)
    issue(40, "plant with no charge (spent)", "plant", "PW", False, 0)

    # shove ----------------------------------------------------------------
    issue(5, "shove (acts)", "shove", "S", True)
    issue(10, "shove on cool-down", "shove", "S", False)
    issue(28, "shove one tick before the cool-down runs out", "shove", "S",
          False)
    issue(29, "shove on the tick it runs out", "shove", "S", True)
    stun(60, "S", 6)
    issue(61, "shove while stunned", "shove", "S", False)
    issue(70, "shove after the stun (acts)", "shove", "S", True)
    issue(5, "shove with no target", "shove", "W", False)
    issue(6, "shove with no target again", "shove", "W", False)

    # melee ----------------------------------------------------------------
    issue(5, "melee (acts)", "melee", "M1", True)
    issue(5 + rof - 1, "melee one tick before the cadence", "melee", "M1",
          False)
    issue(5 + rof, "melee on the cadence boundary", "melee", "M1", True)
    issue(5, "melee a whiff", "melee", "W", False)
    issue(6, "melee a whiff again", "melee", "W", False)
    throw(5, "throw smoke (acts), then a blow asks", "M2", 1, True)
    issue(8, "melee during a wind-up", "melee", "M2", False)
    issue(12, "melee on a wind-up's release tick", "melee", "M2", True)
    issue(5, "plant (acts), then a blow asks", "plant", "M3", True, 0)
    issue(10, "melee during a plant", "melee", "M3", False)
    issue(29, "melee on a plant's last tick", "melee", "M3", True)
    stun(20, "M4", 3)
    issue(22, "melee stunned, remaining 1 at tick start", "melee", "M4",
          False)
    issue(23, "melee stunned, remaining 0 at tick start", "melee", "M4",
          True)
    issue(23, "shove stunned, remaining 0 at tick start", "shove", "M4",
          False)
    throw(23, "throw stunned, remaining 0 at tick start", "M4", 1, False)

    # the dead and the zombie: every verb, no act ---------------------------
    for who in ("D", "Zr"):
        throw(5, f"throw by {who}", who, 0, False)
        issue(6, f"plant by {who}", "plant", who, False, 0)
        issue(7, f"use by {who}", "use", who, False)
        issue(8, f"reload by {who}", "reload", who, False)
        issue(9, f"shove by {who}", "shove", who, False)
        issue(10, f"melee by {who}", "melee", who, False)

    tape = p3a._Tape()
    acts = collections.Counter()
    for _ in range(n_ticks):
        t = int(sim.tick)
        for fn in list(sched.get(t, ())):
            fn()
        sim.set_paused(False)
        sim.step()
        by = collections.defaultdict(set)
        for e in sim.tick_events:
            by[type(e).__name__].add(int(getattr(e, "unit_id", -1)))
        for label, verb, uid, expect, door, was in pend:
            if verb == "use":
                acted = bool(door.want_open) != was
            else:
                acted = uid in by[_VERB_EVENT[verb]]
            acts[verb] += acted
            log.append((t, label, verb, expect, acted))
        pend.clear()
        tape(sim)
    tape.cov["conversions"] = p3a._conversions(sim)
    wrong = [(t, lb, v, ex, ac) for t, lb, v, ex, ac in log if ex != ac]
    print("   gates: verb      issues  acts")
    for v in ("throw", "plant", "use", "reload", "shove", "melee"):
        print(f"          {v:<8} {issues[v]:>6}  {acts[v]:>4}")
    print(f"   gates: {len(log)} labelled issues; act != the note's table: "
          f"{wrong if wrong else 'none'}")
    tape.cov["labelled_issues"] = len(log)
    tape.cov["act_differs_from_table"] = len(wrong)
    return tape, ("gates", int(sim.tick))


SCENARIOS = dict(p3a.SCENARIOS, gates=scenario_gates)


# ---------------------------------------------------------------------------
# dump / compare
# ---------------------------------------------------------------------------
def run(out: str, names=None):
    dump = {}
    for name in names or SCENARIOS:
        tape, outcome = SCENARIOS[name]()
        print(f"{name}: {len(tape.ticks)} ticks, outcome {outcome}")
        print("   coverage: " + ", ".join(
            f"{k}={v}" for k, v in sorted(tape.cov.items())))
        dump[name] = dict(ticks=tape.ticks, outcome=outcome,
                          cov=dict(tape.cov))
    with open(out, "wb") as fh:
        pickle.dump(dump, fh, protocol=pickle.HIGHEST_PROTOCOL)


def _unit_diffs(x, y):
    """{(unit id, key): (a, b)} between two ticks' unit records."""
    out = {}
    for ra, rb in zip(x[1], y[1]):
        for key in sorted(set(ra) | set(rb)):
            if ra.get(key) != rb.get(key):
                out[(ra.get("id"), key)] = (ra.get(key), rb.get(key))
    return out


def compare(a_path: str, b_path: str) -> int:
    with open(a_path, "rb") as fh:
        a = pickle.load(fh)
    with open(b_path, "rb") as fh:
        b = pickle.load(fh)
    bad = 0
    for name in sorted(set(a) | set(b)):
        if name not in a or name not in b:
            print(f"{name}: in one dump only")
            bad += 1
            continue
        ta, tb = a[name]["ticks"], b[name]["ticks"]
        if len(ta) != len(tb) or a[name]["outcome"] != b[name]["outcome"]:
            print(f"{name}: DIFFERENT -- {len(ta)} vs {len(tb)} ticks, "
                  f"outcome {a[name]['outcome']} vs {b[name]['outcome']}")
            bad += 1
            continue
        diff_ticks = [i for i, (x, y) in enumerate(zip(ta, tb)) if x != y]
        if not diff_ticks:
            n_units = sum(len(t[1]) for t in ta)
            print(f"{name}: EQUAL over {len(ta)} ticks ({n_units} unit "
                  f"records, every field digest and tick digest identical); "
                  f"coverage {a[name]['cov']}")
            continue
        bad += 1
        keys = collections.Counter()
        fd_equal = all(x[4] == y[4] for x, y in zip(ta, tb))
        other = collections.Counter()
        for i in diff_ticks:
            for k, (p, q) in enumerate(zip(ta[i], tb[i])):
                if p != q and _TICK_FIELDS[k] not in ("units", "tick_digest"):
                    other[_TICK_FIELDS[k]] += 1
            for (uid, key) in _unit_diffs(ta[i], tb[i]):
                keys[key] += 1
        first = diff_ticks[0]
        print(f"{name}: DIFFERENT at {len(diff_ticks)} of {len(ta)} ticks, "
              f"first at tick {ta[first][0]}")
        print(f"   unit-record keys that differ (ticks x units): "
              f"{dict(keys)}")
        print(f"   every field_digest identical: {fd_equal}; other tape "
              f"columns that differ: {dict(other) or 'none'}")
        for (uid, key), (p, q) in list(_unit_diffs(ta[first],
                                                   tb[first]).items())[:6]:
            print(f"   first tick, unit {uid} {key}: {p!r} != {q!r}")
    return 1 if bad else 0


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        sys.exit(compare(sys.argv[2], sys.argv[3]))
    run(sys.argv[1], sys.argv[2:] or None)
