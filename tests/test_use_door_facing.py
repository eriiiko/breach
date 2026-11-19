"""#215 (Erik, 2026-11-18): USE toggles the door the unit FACES when several
are in reach ("what if I stay near 2 doors and open the wrong one by
mistake"), reach alone when one is ("when only 1 door available, only reach
is fine"). ``verb_gates.use_door`` / ``facing_off``; the device form is
``use_door`` in ``cpp/src/cuda_unit_verbs.cu`` (held equal by
``tests/cuda_unit_verbs_check.py``, its two-door scene)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tools", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from level_loader import EntityInstance, LevelData  # noqa: E402
from simulation import Simulation, verb_gates  # noqa: E402
from simulation.entities import door as door_mod  # noqa: E402
from simulation.materials import MAT_AIR, MAT_HULL  # noqa: E402
from simulation.ruleset import ContinuousRealtime  # noqa: E402
from simulation.turning import PI_Q16, place  # noqa: E402
from simulation.unit import Unit  # noqa: E402

EAST = 0


def _door(eid, ordinal, x, y):
    f = {fl.name: fl.default for fl in door_mod.door.FIELDS}
    f.update(x=x, y=y, orientation="v", length_m=3.0, initial_state="closed")
    return EntityInstance(id=eid, class_name="door", ordinal=ordinal, fields=f)


def _sim(two=True):
    """A room cut by a wall at x = 10 with door A at y 10..12 and (``two``)
    door B at y 14..16, one wall tile between them."""
    h, w = 30, 20
    tm = np.full((h, w), MAT_HULL, dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = MAT_AIR
    tm[1:h - 1, 10] = MAT_HULL
    tm[10:13, 10] = MAT_AIR
    ents = [_door("A", 0, 10, 10)]
    if two:
        tm[14:17, 10] = MAT_AIR
        ents.append(_door("B", 1, 10, 14))
    lvl = LevelData(name="two_doors", version="3", path=Path("."), tilemap=tm,
                    tile_size_m=1.0, diffuse_path=Path("."), entities=ents)
    sim = Simulation(lvl, seed=3, breach_physics=None, enable_recorder=False,
                     ruleset=ContinuousRealtime())
    sim.set_paused(False)
    return sim


def _unit(sim, x=7, y=12, facing=EAST):
    u = sim.get_unit(sim.add_unit(Unit("U", x=float(x), y=float(y), team=0)))
    place(u, facing)
    return u


def test_between_two_doors_the_faced_one_is_chosen_at_a_sweep_of_facings():
    """Property: a unit with both doors in reach gets the door whose nearest
    span tile lies least off its facing, for every facing of a full turn in
    1/64 steps; at the exact tie (straight east, the doors mirror-equal:
    the kit's atan2 is odd in y)
    the scan order decides (A, whose ring tile comes first). Breaks if the
    choice ignores the facing, picks the farther-off door or breaks a tie
    any other way."""
    sim = _sim()
    a, b = sim._doors
    u = _unit(sim)
    assert verb_gates.door_in_reach(u, a) and verb_gates.door_in_reach(u, b)
    seen = set()
    for i in range(64):
        f = -PI_Q16 + (2 * PI_Q16 * i) // 64
        place(u, f)
        oa, ob = verb_gates.facing_off(u, a), verb_gates.facing_off(u, b)
        want = a if oa <= ob else b
        got = verb_gates.use_door(sim, u)
        assert got is want, (i, oa, ob)
        seen.add(got.id)
    assert seen == {"A", "B"}
    place(u, EAST)
    assert verb_gates.facing_off(u, a) == verb_gates.facing_off(u, b)
    assert verb_gates.use_door(sim, u) is a                 # the tie: scan order
    place(u, EAST - PI_Q16 // 8)                            # toward the south: B
    assert verb_gates.use_door(sim, u) is b
    place(u, EAST + PI_Q16 // 8)                            # toward the north: A
    assert verb_gates.use_door(sim, u) is a


def test_one_door_in_reach_ignores_facing():
    """Property: with ONE door in reach the use picks it whatever the facing,
    even facing straight away from it. Breaks if facing becomes a term of
    the single-door case (Erik: reach alone is fine)."""
    sim = _sim(two=False)
    u = _unit(sim, y=10)
    for i in range(16):
        place(u, -PI_Q16 + (2 * PI_Q16 * i) // 16)
        assert verb_gates.use_door(sim, u) is sim._doors[0]
        assert verb_gates.can_use(sim, u)


@pytest.mark.parametrize("facing, which",
                         [(EAST, 0), (EAST - PI_Q16 // 6, 1),
                          (EAST + PI_Q16 // 6, 0)])
def test_one_use_toggles_the_faced_door_and_only_it(facing, which):
    """Property: one USE flips the chosen door's latch and the slot-9e sweep
    opens it; the other door stays closed; a second USE closes it again.
    Breaks if the verb flips another door than ``use_door`` names, or both."""
    sim = _sim()
    u = _unit(sim, facing=facing)
    want = sim._doors[which]
    other = sim._doors[1 - which]
    assert verb_gates.use_door(sim, u) is want
    sim.use_intent(u.id)
    sim.step()
    assert want.state == door_mod.DOOR_OPEN and other.state == door_mod.DOOR_CLOSED
    sim.set_paused(False)
    place(u, facing)
    assert verb_gates.use_door(sim, u) is want      # an open door stays in reach
    sim.use_intent(u.id)
    sim.step()
    assert want.state == door_mod.DOOR_CLOSED and other.state == door_mod.DOOR_CLOSED


def test_the_gate_and_the_mask_still_mean_a_door_in_reach():
    """Property: ``can_use`` is True exactly when some live door is in reach,
    at every facing (facing chooses WHICH door, never WHETHER). Breaks if the
    facing enters the gate."""
    sim = _sim()
    u = _unit(sim)
    for i in range(8):
        place(u, -PI_Q16 + (2 * PI_Q16 * i) // 8)
        assert verb_gates.can_use(sim, u)
    far = _unit(sim, x=3, y=24)
    assert not verb_gates.can_use(sim, far)
