"""#33 -- which model each unit is drawn with (renderer/unit_looks.py).

The assignment is render-only and raylib-free, so it is tested headless on
made-up units and a made-up table: the rules, never the shipped counts.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_unit_looks.py -q
"""
from __future__ import annotations

import copy
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from renderer.unit_looks import LookAssigner, looks_table  # noqa: E402


class _FakeUnit:
    """Just what a renderer reads off a unit: id, is_zombie (+ alive)."""

    def __init__(self, uid: int, is_zombie: bool) -> None:
        self.id = uid
        self.is_zombie = is_zombie
        self.alive = True


_TABLE = {"player": ["p_a", "p_b", "p_c"], "zombie": ["z_a", "z_b"]}


def test_a_unit_keeps_its_look_when_it_turns_zombie():
    """PROPERTY: a unit's look is fixed the first time it is seen; flipping
    ``is_zombie`` afterwards (a player turning mid-match) does not change it.

    BREAKS IF: the look is re-derived from the unit's current role on every
    frame (or the cache is keyed on role), so a turned player would suddenly
    be drawn as a zombie model -- against #33's "turning is an animation
    swap, not a skin swap".
    """
    looks = LookAssigner(_TABLE)
    player = _FakeUnit(3, is_zombie=False)
    other = _FakeUnit(7, is_zombie=True)
    (first,) = looks.looks_for([player])
    assert first in _TABLE["player"]
    player.is_zombie = True
    for _ in range(3):
        assert looks.looks_for([other, player])[1] == first


def test_units_of_a_role_are_dealt_its_list_round_robin_by_id():
    """PROPERTY: among the units of one role seen together, the k-th in
    ascending id gets entry ``k mod n`` of that role's list, whatever order
    the units arrive in and however the roles interleave.

    BREAKS IF: looks are dealt in arrival order instead of id order, one
    counter is shared across roles, the deal stops cycling (index error or
    clamping to the last entry), or a role's list is read out of order.
    """
    rng = random.Random(1234)
    ids = rng.sample(range(1000), 23)
    units = [_FakeUnit(i, is_zombie=rng.random() < 0.5) for i in ids]
    rng.shuffle(units)
    names = dict(zip((u.id for u in units),
                     LookAssigner(_TABLE).looks_for(units)))
    for role, zombie in (("player", False), ("zombie", True)):
        of_role = sorted(u.id for u in units if u.is_zombie == zombie)
        assert of_role, "the made-up roster must hold both roles"
        want = _TABLE[role]
        for k, uid in enumerate(of_role):
            assert names[uid] == want[k % len(want)], (role, k, uid)


def test_assigning_looks_never_touches_a_unit():
    """PROPERTY: dealing and re-reading looks leaves every attribute of
    every unit exactly as it was (the look lives only in the render layer).

    BREAKS IF: the assigner caches the look (or anything) on the unit
    object -- the first step towards render state leaking into ``Unit`` and
    its digest.
    """
    units = [_FakeUnit(i, is_zombie=(i % 3 == 0)) for i in range(9)]
    before = [copy.deepcopy(vars(u)) for u in units]
    looks = LookAssigner(_TABLE)
    looks.looks_for(units)
    looks.looks_for(list(reversed(units)))
    assert [vars(u) for u in units] == before


def test_a_look_table_that_draws_nobody_is_refused():
    """PROPERTY: a ``[render.unit_looks]`` role that is missing or empty is
    a config error (ValueError naming the key), not a silent fallback.

    BREAKS IF: ``looks_table`` returns an empty list for a role, which would
    make the round-robin divide by zero at draw time instead of saying which
    key is wrong at load.
    """
    from config import Namespace

    cfg = Namespace({"render": {"unit_looks": {"player": ["x"], "zombie": []}}})
    with pytest.raises(ValueError, match="zombie"):
        looks_table(cfg)
    cfg = Namespace({"render": {"unit_looks": {"zombie": ["x"]}}})
    with pytest.raises(ValueError, match="player"):
        looks_table(cfg)
