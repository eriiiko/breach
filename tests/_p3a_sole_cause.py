"""#29 P3a's SOLE-CAUSE driver: the game must not change, only the digest's
definition.

Runs the same scenarios on any tree and writes, per tick, the schema's unit
records (``simulation.unit_schema.digest_record`` -- built by ``getattr``, so
it reads a pre-P3a ``Unit`` as well as a slotted one), the planted charges'
rows (``charge_record``), the synced events and the field digest. Run it on
main before P3a (a throwaway worktree with ``src/simulation/unit_schema.py``
and this file copied in, nothing else changed) and on the P3a tree, then
``--compare`` the two dumps: equal records at every tick and every field
digest identical = the re-pin moves the HASH, not the game.

    <py> tests/_p3a_sole_cause.py OUT.pkl            # all scenarios
    <py> tests/_p3a_sole_cause.py --compare A.pkl B.pkl

Scenarios: ``ship_demo`` seed 1 (its match.toml: bots vs bots, the horde);
the same with ``bots_v1`` on both teams; the #122 box (idle marines, a horde
that bites and CONVERTS them); a scripted arena where the verbs are driven
directly -- every hand-grenade kind thrown, a C4 planted and detonated,
melee, shove, a held trigger that empties a magazine and a manual reload,
the flashlight off and on, a knockdown. Each scenario prints its coverage
(events by kind, conversions, knockdowns, dark ticks) so a vacuous scenario
shows.
"""
from __future__ import annotations

import collections
import pickle
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tools", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

import breach_physics as bp  # noqa: E402
import level_lib  # noqa: E402
import level_loader  # noqa: E402
from field_ab_harness import SIM_FIELDS, _event_record, _snapshot  # noqa: E402
from field_digest import field_digest  # noqa: E402
from simulation import unit_schema  # noqa: E402

_SYNCED = ("UnitHitEvent", "UnitKilledEvent")


class _Tape:
    """Per tick: (tick, unit records, charge rows, synced events, field
    digest), plus coverage counters."""

    def __init__(self):
        self.ticks = []
        self.cov = collections.Counter()

    def __call__(self, sim):
        units = sorted(sim.units, key=lambda u: int(u.id))
        recs = [unit_schema.digest_record(u) for u in units]
        charges = [unit_schema.charge_record(c)
                   for c in getattr(sim, "planted_charges", ()) or ()
                   if c.live]
        evs = [_event_record(e) for e in sim.tick_events
               if type(e).__name__ in _SYNCED]
        fd = field_digest(_snapshot(sim.gmap, SIM_FIELDS))
        self.ticks.append((int(sim.tick), recs, charges, evs, fd))
        for e in sim.tick_events:
            self.cov["ev:" + type(e).__name__] += 1
        from simulation.status import KNOCKED_DOWN
        for u in units:
            if any(st.kind == KNOCKED_DOWN for st in u.statuses):
                self.cov["knocked_down_unit_ticks"] += 1
            if not u.flashlight_on:
                self.cov["dark_unit_ticks"] += 1
            if u.alive and (u.pvx or u.pvy):
                self.cov["pushed_unit_ticks"] += 1
        if charges:
            self.cov["charge_ticks"] += 1
        self.last_conversions = _conversions(sim)


def _conversions(sim):
    return sum(1 for u in sim.units
               if u.is_zombie and getattr(u, "loadout_name", None) is not None)


def _match(level, spec=None):
    from run_match import run_match
    tape = _Tape()
    o = run_match(level, spec, on_tick=tape)
    tape.cov["conversions"] = tape.last_conversions
    return tape, (o.result.outcome, o.result.team, o.result.ticks,
                  o.field_digest)


def scenario_ship_demo():
    level = level_loader.load("ship_demo", str(ROOT / "levels"))
    return _match(level)


def scenario_ship_demo_bots_v1():
    level = level_loader.load("ship_demo", str(ROOT / "levels"))
    spec = level.match
    spec = replace(spec, teams=tuple(replace(t, controller="bots_v1")
                                     for t in spec.teams))
    return _match(level, spec)


def scenario_box_conversions():
    from simulation.factions import ZOMBIE_TEAM
    from simulation.match import MatchSpec, MatchTeam, SpawnRoom
    from simulation.materials import MAT_AIR, MAT_HULL
    idle = MatchTeam(n=2, loadout="rifleman", env_profile="default",
                     controller="idle")
    rooms = [SpawnRoom(team=0, x=2, y=2, w=8, h=8, max_per_team=4),
             SpawnRoom(team=1, x=30, y=14, w=8, h=8, max_per_team=4),
             SpawnRoom(team=ZOMBIE_TEAM, x=12, y=2, w=8, h=8,
                       max_per_team=4)]
    spec = MatchSpec(teams=(idle, idle), seed=1, zombies=3,
                     time_limit_s=20.0)
    tmp = Path(tempfile.mkdtemp(prefix="p3a_box_"))
    lvl = tmp / "box"
    lvl.mkdir()
    h, w = 24, 40
    level_lib.write_level_header(lvl, name="box", tile_size_m=1.0)
    tm = np.full((h, w), MAT_HULL, dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = MAT_AIR
    level_lib.write_tilemap_csv(lvl, tm, csv_bak=False)
    from PIL import Image
    Image.new("RGB", (w, h)).save(lvl / "diffuse.png")
    level_lib.write_spawn_rooms(lvl / "level.toml", rooms)
    level_lib.write_match(lvl, spec)
    level = level_loader.load("box", str(tmp))
    return _match(level)


def scenario_scripted(n_ticks=420):
    """Every verb driven directly on a continuous arena (1/3 m tiles)."""
    from level_loader import LevelData
    from simulation import Simulation
    from simulation.factions import ZOMBIE_TEAM
    from simulation.materials import MAT_AIR, MAT_HULL
    from simulation.ruleset import ContinuousRealtime
    from simulation.turning import PI_Q16, place
    from simulation.unit import Unit
    from simulation.weapons import get_tables

    h = w = 64
    tm = np.full((h, w), MAT_HULL, dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = MAT_AIR
    level = LevelData(name="p3a", version="1", path=Path("."), tilemap=tm,
                      tile_size_m=1.0 / 3.0, diffuse_path=Path("."))
    sim = Simulation(level, seed=7, breach_physics=bp, enable_recorder=False,
                     ruleset=ContinuousRealtime())
    sim.set_paused(False)

    def add(name, x, y, team, facing):
        u = sim.get_unit(sim.add_unit(Unit(name, x=x, y=y, team=team)))
        place(u, facing)
        if not u.is_zombie:
            u.current_hp = 10000.0
        return u

    s = add("S", 20, 20, 0, 0)                      # striker, facing east
    e = add("E", 23, 20, 1, PI_Q16)                 # enemy, adjacent east
    f = add("F", 20, 30, 0, -(PI_Q16 // 2))         # thrower, facing south
    p = add("P", 40, 12, 0, 0)                      # the planter, alone
    add("Z", 50, 50, ZOMBIE_TEAM, 0)                # a zombie the conductor walks
    one = 1 << 16
    kinds = get_tables().hand_grenade_rows
    tape = _Tape()
    for t in range(n_ticks):
        if t < 60 and t % 3 == 0:
            sim.melee_intent(s.id)
        if t in (5, 45, 130):
            sim.shove_intent(s.id)
        if t == 8:
            sim.set_flashlight(s.id, True)
        if t == 70:
            sim.set_flashlight(s.id, False)
        for k in range(len(kinds)):                 # every kind, thrown south
            if t == 12 + 30 * k:
                sim.throw_grenade_intent(f.id, 0, one, 24, 5 * one, k)
        if t == 110:                                # a frag between S and E
            sim.throw_grenade_intent(f.id, 0, -one, 20, 9 * one, 0)
        if t == 30:
            sim.plant_charge_intent(p.id, 0)        # clamped to the minimum
        if 50 <= t < 70:
            sim.set_move_dir(e.id, 0, -one, one)
        if t == 70:
            sim.clear_move_dir(e.id)
        if 80 <= t < 260:
            sim.set_aim(s.id, one, 0)
            sim.set_trigger(s.id, True)
        if t == 150:
            sim.reload_intent(s.id)
        if t == 260:
            sim.set_trigger(s.id, False)
        sim.set_paused(False)
        sim.step()
        tape(sim)
    tape.cov["conversions"] = _conversions(sim)
    return tape, ("scripted", int(sim.tick))


SCENARIOS = {
    "ship_demo": scenario_ship_demo,
    "ship_demo_bots_v1": scenario_ship_demo_bots_v1,
    "box_conversions": scenario_box_conversions,
    "scripted": scenario_scripted,
}


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
        n_units = sum(len(t[1]) for t in ta)
        if a[name]["outcome"] != b[name]["outcome"]:
            print(f"{name}: outcome {a[name]['outcome']} != "
                  f"{b[name]['outcome']}")
            bad += 1
        if len(ta) != len(tb):
            print(f"{name}: {len(ta)} ticks != {len(tb)}")
            bad += 1
        first = None
        for i, (x, y) in enumerate(zip(ta, tb)):
            if x != y:
                first = i
                break
        if first is not None:
            bad += 1
            x, y = ta[first], tb[first]
            what = ("tick", "units", "charges", "events", "field_digest")
            for k, (p, q) in enumerate(zip(x, y)):
                if p != q:
                    print(f"{name}: first difference at tick index {first}, "
                          f"in {what[k]}")
                    if k == 1:
                        for ra, rb in zip(p, q):
                            if ra != rb:
                                for key in sorted(set(ra) | set(rb)):
                                    if ra.get(key) != rb.get(key):
                                        print(f"   unit {ra.get('id')} {key}: "
                                              f"{ra.get(key)!r} != "
                                              f"{rb.get(key)!r}")
                                break
        else:
            print(f"{name}: EQUAL over {len(ta)} ticks ({n_units} unit "
                  f"records, every field digest identical); coverage "
                  f"{a[name]['cov']}")
    return 1 if bad else 0


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        sys.exit(compare(sys.argv[2], sys.argv[3]))
    run(sys.argv[1], sys.argv[2:] or None)
