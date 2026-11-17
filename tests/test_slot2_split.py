"""#29 P7.0b -- the reference's slot 2 split in two conductor calls
(``Simulation._update_grenades`` then ``Simulation._update_rounds``), each
with its own ``tools/tick_stage_profile.py`` label (the harness and the twin
manifest name slots by these labels).

The split is a pure refactor; its proof that the game did not move is the
sole-cause driver (``tests/_p3a_sole_cause.py``, run before and after) and
the match pins / golden. These tests hold its SHAPE.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402
import tick_stage_profile as tsp  # noqa: E402

GRENADES, ROUNDS = "2 grenades", "2' rounds"


def _arena():
    """A thrower lobbing a frag and a shooter holding the trigger, so a
    grenade and rounds are in flight on the same ticks."""
    from level_loader import LevelData
    from simulation import Simulation
    from simulation.materials import MAT_AIR, MAT_HULL
    from simulation.ruleset import ContinuousRealtime
    from simulation.turning import place
    from simulation.unit import Unit
    h, w = 40, 400
    tm = np.full((h, w), MAT_HULL, dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = MAT_AIR
    level = LevelData(name="slot2", version="1", path=Path("."), tilemap=tm,
                      tile_size_m=1.0 / 3.0, diffuse_path=Path("."))
    sim = Simulation(level, seed=3, breach_physics=bp, enable_recorder=False,
                     ruleset=ContinuousRealtime())
    sim.set_paused(False)
    t = sim.get_unit(sim.add_unit(Unit("T", x=10, y=10, team=0)))
    s = sim.get_unit(sim.add_unit(Unit("S", x=10, y=30, team=0)))
    place(t, 0)
    place(s, 0)
    return sim, t, s


def test_both_slot2_labels_resolve_to_the_conductors_calls():
    """Each label names exactly one Simulation method that exists, grenades
    before rounds in slot order, and the old single label is gone. Breaks:
    a label left on a removed callable, the two swapped, or a third callable
    joining a label."""
    rows = [(m, q, lab) for m, q, lab in tsp.SLOT_WRAPS
            if lab in (GRENADES, ROUNDS)]
    assert [lab for _m, _q, lab in rows] == [GRENADES, ROUNDS]
    from simulation import Simulation
    for _m, q, _lab in rows:
        assert callable(getattr(Simulation, q.split(".")[1]))
    assert not any(lab == "2 projectiles" for _m, _q, lab in tsp.SLOT_WRAPS)
    assert not hasattr(Simulation, "_update_projectiles")


def test_the_profiler_sees_both_with_grenades_and_rounds_in_flight():
    """On a tick with a grenade AND a round in flight both labels are timed,
    once each per tick, the grenades first. Breaks: a label the profiler
    cannot see (the time lands in the rest of the step), a call made twice or
    skipped, or the order reversed (rounds marching before a grenade's
    blast)."""
    from simulation import Simulation
    sim, t, s = _arena()
    one = 1 << 16
    order, both = [], 0
    og, orr = Simulation._update_grenades, Simulation._update_rounds

    def g(self, __o=og):
        order.append("g")
        return __o(self)

    def r(self, __o=orr):
        nonlocal both
        order.append("r")
        live_g = any(not p.detonated for p in self.projectiles)
        both += bool(live_g and self.bullets)
        return __o(self)

    prof = tsp.TickStageProfiler(skip=0)
    Simulation._update_grenades, Simulation._update_rounds = g, r
    prof.install(bp)
    try:
        for k in range(60):
            if k == 2:
                sim.throw_grenade_intent(t.id, one, 0, 40, 12 * one, 0)
            sim.set_aim(s.id, one, 0)
            sim.set_trigger(s.id, True)
            sim.set_paused(False)
            sim.step()
    finally:
        prof.uninstall()
        Simulation._update_grenades, Simulation._update_rounds = og, orr
    assert both > 0, "no tick had a grenade and a round in flight together"
    assert order == ["g", "r"] * 60
    assert len(prof.calls) == 60
    for c in prof.calls:
        assert c.get(GRENADES) == 1 and c.get(ROUNDS) == 1, c
