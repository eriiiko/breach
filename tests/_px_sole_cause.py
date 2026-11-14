"""#29 PX's SOLE-CAUSE driver: the poison law changes nothing where there is
no poison.

The scenarios of ``tests/_p3a_sole_cause.py`` (imported, never copied) with
its per-tick tape -- the schema's unit records (``unit_schema.digest_record``,
built by ``getattr``, so a tree without the poison columns records them
absent, as a tree with them does at zero), the planted charges, the synced
events and the field digest -- plus the scripted arena with every hand-grenade
kind BUT the poison grenade (``scripted_no_poison``). Run it on main before
PX (a throwaway checkout with this file copied in) and on the PX tree, then
``--compare`` the two dumps (``_p3a_sole_cause.compare``): equal records at
every tick and every field digest identical on every scenario without poison
gas = the law is dormant there.

``scripted_poison`` (every kind, the thrower handed one poison grenade at its
feet: its kit carries none, so ``_p3a_sole_cause``'s own scenario throws no
poison) turned out EQUAL too: the arena is built with no air, and a cloud in
it deposits nothing. ``ship_demo_poison`` (the bench's match scene: a poison
grenade thrown into a squad's room on ship_demo) is the scenario EXPECTED to
differ; it is run so the report can name its first difference and the poison
hits on each tree.

    <py> tests/_px_sole_cause.py OUT.pkl [scenario ...]
    <py> tests/_px_sole_cause.py --compare A.pkl B.pkl
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tools", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

import breach_physics as bp  # noqa: E402
import _p3a_sole_cause as p3a  # noqa: E402

POISON_ROW = "grenade_poison"


def scenario_scripted_arena(n_ticks=420, with_poison=False):
    """``_p3a_sole_cause.scenario_scripted``'s arena and verbs, the
    hand-grenade kinds filtered: every kind, or every kind but the poison
    grenade (its throw tick then throws nothing: the other kinds keep their
    ticks)."""
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

    s = add("S", 20, 20, 0, 0)
    e = add("E", 23, 20, 1, PI_Q16)
    f = add("F", 20, 30, 0, -(PI_Q16 // 2))
    p = add("P", 40, 12, 0, 0)
    add("Z", 50, 50, ZOMBIE_TEAM, 0)
    one = 1 << 16
    kinds = get_tables().hand_grenade_rows
    if with_poison:
        # The thrower's kit carries no poison grenade (a throw of a kind it
        # lacks is ignored): hand it one, so the cloud is real.
        f.grenade_counts[kinds.index(POISON_ROW)] += 1
    tape = p3a._Tape()
    for t in range(n_ticks):
        if t < 60 and t % 3 == 0:
            sim.melee_intent(s.id)
        if t in (5, 45, 130):
            sim.shove_intent(s.id)
        if t == 8:
            sim.set_flashlight(s.id, True)
        if t == 70:
            sim.set_flashlight(s.id, False)
        for k in range(len(kinds)):
            if t == 12 + 30 * k and (with_poison or kinds[k] != POISON_ROW):
                # the poison grenade at the thrower's feet: a unit in the cloud
                reach = one if kinds[k] == POISON_ROW else 5 * one
                sim.throw_grenade_intent(f.id, 0, one, 24, reach, k)
        if t == 110:
            sim.throw_grenade_intent(f.id, 0, -one, 20, 9 * one, 0)
        if t == 30:
            sim.plant_charge_intent(p.id, 0)
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
        for ev in sim.tick_events:
            if type(ev).__name__ == "UnitHitEvent" and ev.source == "poison_gas":
                tape.cov["poison_hits"] += 1
    tape.cov["conversions"] = p3a._conversions(sim)
    return tape, ("scripted", int(sim.tick))


def scenario_ship_demo_poison(seconds=60):
    """The bench's match scene (``tools/bench_poison.py::run_scene``): a
    poison grenade thrown into a squad's room on ship_demo (real air; the
    scripted arena above has none, so a cloud there deposits nothing)."""
    import bench_poison
    tape = p3a._Tape()

    def on_tick(sim):
        tape(sim)
        for ev in sim.tick_events:
            if type(ev).__name__ == "UnitHitEvent" and ev.source == "poison_gas":
                tape.cov["poison_hits"] += 1
    _sq, _lost, death, hits, _d, _m = bench_poison.run_scene(seconds,
                                                             on_tick=on_tick)
    tape.cov["deaths"] = len(death)
    return tape, ("ship_demo_poison", hits)


SCENARIOS = {
    "ship_demo": p3a.scenario_ship_demo,
    "ship_demo_bots_v1": p3a.scenario_ship_demo_bots_v1,
    "box_conversions": p3a.scenario_box_conversions,
    "scripted_no_poison": lambda: scenario_scripted_arena(with_poison=False),
    "scripted_poison": lambda: scenario_scripted_arena(with_poison=True),
    "ship_demo_poison": scenario_ship_demo_poison,
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


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        sys.exit(p3a.compare(sys.argv[2], sys.argv[3]))
    run(sys.argv[1], sys.argv[2:] or None)
