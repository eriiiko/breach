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


def test_every_draw_call_of_a_frame_advances_the_animation_alike():
    """PROPERTY: the renderer's draw calls of one frame (the game makes one
    for the players and one for the zombies, with the same clock) all
    advance the animation by that frame's wall-clock step, and the step
    follows the clock from frame to frame.

    BREAKS IF: the step is taken as ``clock - last call's clock`` per call
    again -- the second call of every frame then gets 0 and the zombies never
    animate (found and fixed in #33's unit-looks patch).
    """
    from renderer.unit_model_renderer import UnitModelRenderer

    r = UnitModelRenderer()   # no GL needed: nothing is loaded
    r._frame_dt(10.0)
    for clock, step in ((10.5, 0.5), (10.75, 0.25)):
        assert r._frame_dt(clock) == pytest.approx(step)   # players
        assert r._frame_dt(clock) == pytest.approx(step)   # zombies


_SHADING_KEYS = ("rim_albedo", "gloss_strength", "gloss_shininess",
                 "normal_map", "normal_strength", "blob_shadow")


def test_shipped_unit_shading_has_every_setting_and_zero_means_off():
    """PROPERTY: the shipped ``[render.unit_shading]`` holds every setting the
    unit renderer reads each frame, and a zero / false value reaches it as
    zero / false (gloss_strength 0 = no highlight, normal_map false = no map,
    blob_shadow false = no disc) -- never replaced by a default.

    BREAKS IF: a setting is dropped from config.toml or renamed on one side
    only, or the reader substitutes a default for a falsy value (``x or 1.0``),
    so "off" in the file could not switch the effect off.
    """
    from config import CFG, Namespace
    from renderer.unit_looks import unit_shading

    shipped = unit_shading(CFG)
    assert {k for k in _SHADING_KEYS} <= set(vars(shipped))
    off = {"rim_albedo": 0.0, "gloss_strength": 0.0, "gloss_shininess": 1.0,
           "normal_map": False, "normal_strength": 0.0, "blob_shadow": False}
    got = unit_shading(Namespace({"render": {"unit_shading": dict(off)}}))
    assert got.gloss_strength == 0.0 and got.normal_strength == 0.0
    assert got.normal_map is False and got.blob_shadow is False


def test_a_missing_or_mistyped_unit_shading_setting_fails_loudly():
    """PROPERTY: every ``[render.unit_shading]`` key is required: leaving one
    out, or giving an on/off switch a number (or a number a string), raises a
    ValueError naming the key.

    BREAKS IF: the reader falls back to a default for an absent key (a typo
    in config.toml would then silently draw the default) or coerces
    ``normal_map = 0`` / ``"2.0"`` instead of refusing them.
    """
    from config import Namespace
    from renderer.unit_looks import unit_shading

    full = {"rim_albedo": 0.9, "gloss_strength": 2.0, "gloss_shininess": 48.0,
            "normal_map": True, "normal_strength": 1.0, "blob_shadow": True}
    for key in _SHADING_KEYS:
        partial = {k: v for k, v in full.items() if k != key}
        with pytest.raises(ValueError, match=key):
            unit_shading(Namespace({"render": {"unit_shading": partial}}))
    for key, bad in (("normal_map", 0), ("blob_shadow", 1), ("gloss_strength", "2.0")):
        with pytest.raises(ValueError, match=key):
            unit_shading(Namespace({"render": {"unit_shading": dict(full, **{key: bad})}}))
