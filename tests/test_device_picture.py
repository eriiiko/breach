"""#29 P8.3 -- the device team picture's host side
(``simulation.device_picture``): its block layout, its generated constants,
the normal form and the block conversion a per-env reset restages through.
CPU only; the card's every-tick gate is ``tests/cuda_obs_check.py`` (wrapper
``tests/test_cuda_obs.py``).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release", ROOT / "tools",
           ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402

from simulation import device_obs, device_picture as dp  # noqa: E402


def test_the_block_sections_tile_the_block_exactly():
    """Property: the sections of one env's block follow one another with no
    gap and no overlap, every shape sized from (T, U) and the two object
    tables' capacities, and the width is their sum. Breaks when a section
    is added or resized in one place (the kernels' ``pic_layout`` is checked
    against this list at every build on the card)."""
    for T, U in ((1, 1), (2, 12), (3, 40)):
        off = 0
        for name, o, shape in dp.layout(T, U):
            assert o == off, (T, U, name)
            off += int(np.prod(shape))
        assert dp.width(T, U) == off
        sec = {n: s for n, _o, s in dp.layout(T, U)}
        assert sec["REC"] == (T, U, len(dp.REC_FIELDS))
        assert sec["UNIT"] == (U, len(dp.UNIT_FIELDS))
        assert sec["DMG"] == (T, dp.DMG_CAP, len(dp.DMG_FIELDS))
        g, c = dp._caps()
        assert sec["OBJ"] == sec["SEEN"] == (T, g + c, 2)


def test_the_picture_constants_reach_the_generated_header():
    """Property: every constant the picture's kernels read is a row of the
    observation's ONE generated header (the object id base the reference's
    own ``OBJECT_UID_BASE``, the fault bits distinct powers of two). Breaks
    when a field or a fault is added without ``tools/gen_obs_layout.py
    --write``."""
    from simulation.team_picture import OBJECT_UID_BASE
    consts = dict(device_obs._constants())
    for k, v in dp.constants():
        assert consts[k] == v
    assert consts["OBS_OBJECT_UID_BASE"] == OBJECT_UID_BASE
    bits = [v for k, v in dp.constants() if k.startswith("OBS_PIC_FAULT_")]
    assert len(bits) == len(dp.FAULT_BITS) and len(set(bits)) == len(bits)
    assert all(b & (b - 1) == 0 for b in bits)
    text = (ROOT / device_obs.HEADER_REL).read_text(encoding="utf-8")
    assert "OBS_PIC_REC_W" in text and "OBS_PIC_DMG_CAP" in text


def _hand_session():
    """A ship_demo bots match with grenades thrown by hand (the tracker and
    the damage window move), played on the CPU."""
    import _pinned_matches as pm
    from simulation import unit_fixed
    from simulation.match_session import MatchSession
    lv = pm.ship_demo()
    s = MatchSession(lv, pm.spec_of(lv, "bots"), seed=31, breach_physics=bp)
    return s, unit_fixed.FP_ONE


def test_the_block_carries_the_whole_picture():
    """Property: the normal form of the REFERENCE picture, put into a device
    block (``block_from_form``: what a per-env reset restages) and read back
    (``device_form``: what the gate reads off the card), is the same form,
    field for field, at every tick of a ship_demo bots match (520 ticks) with grenades
    in flight and a damage window that fills -- so the block holds every
    piece of the picture's state. Breaks when a tracker, a record field, the
    damage window's order or the object positions are dropped or mangled by
    the layout."""
    s, one = _hand_session()
    sim = s.sim
    pic = sim.team_picture
    T, U = len(pic.teams), len(sim.units)
    uids = [int(u.id) for u in sim.units]
    params = (pic.memory_ticks, pic.window_ticks, len(pic.teams))
    marines = sorted((u for u in sim.units if not u.is_zombie), key=lambda u: int(u.id))
    seen_objects = seen_damage = 0
    for k in range(520):
        if k in (4, 9, 15) and marines[k % len(marines)].alive:
            sim.throw_grenade_intent(int(marines[k % len(marines)].id), one, 0, 60, kind=1)
        s.tick()
        form = dp.host_form(pic, sim)
        blk = dp.block_from_form(form, uids, T, U, params)
        back = dp.device_form(blk, uids, T, T, U, len(sim.projectiles),
                              len(sim.planted_charges))
        assert dp.first_difference(form, back) is None, (k, dp.first_difference(form, back))
        seen_objects += sum(len(o) for o in form["objects"].values())
        seen_damage += sum(len(q) for d in form["damage"].values() for q in d.values())
    assert seen_objects > 0 and seen_damage > 0, (seen_objects, seen_damage)


def test_a_difference_is_named_by_its_field():
    """Property: ``first_difference`` names the first differing field of two
    forms by its path and both values, and returns None for equal forms
    (the gate's message). Breaks if a nested difference goes unseen."""
    a = {"records": {0: {7: (1, 2)}}, "tick": 3}
    assert dp.first_difference(a, {"records": {0: {7: (1, 2)}}, "tick": 3}) is None
    assert dp.first_difference(a, {"records": {0: {7: (1, 9)}}, "tick": 3}) == \
        (("records", 0, 7), (1, 2), (1, 9))
    assert dp.first_difference(a, {"records": {0: {}}, "tick": 3})[2] == "<absent>"


def test_every_fault_bit_has_a_name_and_a_reason():
    """Property: a fault word names every bit it carries (the books read's
    message). Breaks when a fault bit is added without its text."""
    every = (1 << len(dp.FAULT_BITS)) - 1
    assert dp.fault_names(every) == [n for n, _ in dp.FAULT_BITS]
    assert all(why in dp.fault_text(every) for _n, why in dp.FAULT_BITS)
    assert dp.fault_names(0) == []
