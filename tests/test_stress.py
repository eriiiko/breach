"""The stress level and the heart rate (#215; ``docs/body_temperature_design_
2026-11-17.md`` section 2b) and the shared clearing step.

``src/simulation/stress.py``: one per-unit level fed ONLY through its producer
door (the one producer: damage actually lost, from ``damage.apply_packet``),
cleared ONLY by its own tick at a 2-minute half-life through the clearing step
lifted from the poison law (``src/simulation/clearing.py``); the heart rate a
Hill curve DERIVED from it, read by nothing in the game yet. Each test names
the property it protects and the change that must break it.
"""
from __future__ import annotations

import ast
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config import CFG  # noqa: E402
from simulation import clearing, poison, stress, unit_fixed, unit_schema  # noqa: E402
from simulation.ai_zombie import convert_marines_to_zombies  # noqa: E402
from simulation.damage import (HEAL, KINETIC, DamagePacket,  # noqa: E402
                               apply_packet)
from simulation.factions import ZOMBIE_TEAM  # noqa: E402
from simulation.unit import Unit  # noqa: E402

ONE = unit_fixed.FP_ONE
TPS = int(CFG.clock.ticks_per_second)


def _marine(hp=100.0, uid=1, team=0):
    u = Unit("M", x=8, y=8, team=team)
    u.id = uid
    u.current_hp = hp
    return u


def _hit(u, amount, dtype=KINETIC):
    return apply_packet(u, DamagePacket(amount=amount, dtype=dtype), None,
                        source="bullet")


def _cfg(**over):
    sec = dict(stress_half_life_s=120.0, stress_per_hp=1.0, heart_rate_rest=60.0,
               heart_rate_max=190.0, heart_half_stress=50.0, heart_hill_n=2)
    sec.update(over)
    sec = {k: v for k, v in sec.items() if v is not None}
    return SimpleNamespace(damage=SimpleNamespace(stress=sec),
                           clock=SimpleNamespace(ticks_per_second=24))


# ---------------------------------------------------------------------------
# The clearing step
# ---------------------------------------------------------------------------
def _old_poison_step(x, a, flush=True):
    """poison.step as it stood before #215 lifted it (the frozen reference)."""
    if x <= 0:
        return 0
    if flush and x <= poison.FLUSH_Q:
        return x
    m = (x * a) >> 16
    return m if m >= 1 else 1


def test_the_lifted_step_is_the_poison_step_bit_for_bit():
    """Property: ``poison.step`` through the shared ``clearing.drain`` is the
    step the poison law carried, on every input (random levels, fractions,
    both flush modes, the edges). Breaks if the shared step changes the
    poison's arithmetic (its flush, its floor of one, its shift)."""
    rng = random.Random(215)
    cases = [(0, 683, True), (-5, 683, True), (1, 1, False), (poison.FLUSH_Q, 137, True),
             (poison.FLUSH_Q + 1, 137, True), (poison.X_MAX, ONE, False)]
    cases += [(rng.randrange(-10, poison.X_MAX), rng.randrange(1, ONE + 1),
               rng.random() < 0.5) for _ in range(20000)]
    for x, a, f in cases:
        assert poison.step(x, a, f) == _old_poison_step(x, a, f), (x, a, f)


@pytest.mark.parametrize("seconds", (120.0, 1200.0))
def test_the_half_life_comes_out_within_one_percent(seconds):
    """Property: from any level well above the flush, the integer stress
    tick halves the level in the authored half-life to within 1 % (120 s:
    Erik's; 1200 s: the later drug's), measured on the law itself. Breaks
    if the rate goes back to a Q16 fraction (16 counts at 120 s: 1.5 % off;
    2 counts at 1200 s: 15 %) or the door's rate formula is wrong."""
    rate = clearing.half_life_rate(seconds, TPS)
    for start_pts in (10, 100, 5000):
        x = start_pts * stress.LEVEL_ONE
        t = 0
        while x > start_pts * stress.LEVEL_ONE // 2:
            x -= clearing.drain(x, rate, clearing.RATE_SHIFT, stress.FLUSH_Q)
            t += 1
        assert abs(t / TPS - seconds) <= 0.01 * seconds, (seconds, start_pts, t)


def test_a_level_returns_to_exactly_zero():
    """Property: stress drains to EXACTLY 0 and the unit then hashes as never
    stressed. Breaks if the flush is removed (the floor-of-one tail would
    still end it, but hundreds of ticks later: the bound below fails)."""
    u = _marine(hp=1000.0)
    _hit(u, 30.0)
    assert u.stress_q > 0
    ticks = 0
    while u.stress_q:
        stress.tick([u])
        ticks += 1
        assert ticks < 24 * 120 * 12, "the level never reached 0"
    assert "stress_q" not in unit_schema.digest_record(u)


# ---------------------------------------------------------------------------
# The producer door
# ---------------------------------------------------------------------------
def test_stress_is_fed_by_the_hit_points_actually_lost():
    """Property: a living marine's stress rises by ``stress_per_hp`` points
    per hit point the packet actually took (after mitigation), and by
    nothing for a heal or for a packet that took nothing. Breaks if the
    producer reads the packet's raw amount, or a heal or a zero hit stresses."""
    s = stress.get_settings()
    u = _marine(hp=1000.0)
    dmg = _hit(u, 12.5)
    assert u.stress_q == unit_fixed.mul_q16_round(
        unit_fixed.quantize_scalar(dmg), s.weights_q["damage"])
    before = u.stress_q
    _hit(u, -5.0, HEAL)
    assert u.stress_q == before, "a heal added stress"
    _hit(u, 0.0)
    assert u.stress_q == before, "a hit that took nothing added stress"


def test_a_zombie_and_a_corpse_take_no_stress():
    """Property (the patch's choice, Erik's to overturn): a zombie takes no
    stress, a killing blow's stress is cleared with the rest, a hit on a
    corpse adds none. Breaks if the producer skips the zombie gate or the
    death clear."""
    z = _marine(hp=1000.0, uid=2, team=ZOMBIE_TEAM)
    z.is_zombie = True
    _hit(z, 20.0)
    assert z.stress_q == 0
    m = _marine(hp=10.0)
    _hit(m, 30.0)
    assert not m.alive and m.stress_q == 0
    _hit(m, 30.0)
    assert m.stress_q == 0


def test_conversion_clears_the_level():
    """Property: a marine a zombie killed turns with no stress (and an
    ambient body). Breaks if the conversion keeps the level."""
    m = _marine(hp=1000.0)
    _hit(m, 20.0)
    m.stress_q = 7 * stress.LEVEL_ONE        # a fixture between ticks: a corpse's leftover
    m.body_temp_q = 99 * ONE
    m.alive = False
    m.killed_by_zombie = True
    convert_marines_to_zombies([m])
    assert m.is_zombie and m.stress_q == 0
    assert m.body_temp_q == unit_schema.COLUMN_BY_NAME["body_temp_q"].host_default()


def test_the_door_refuses_an_unknown_producer():
    """Property: a producer is a named row; any other name is refused by
    name. Breaks if the door accepts a free-form source."""
    with pytest.raises(ValueError, match="no producer row"):
        stress.produce(_marine(), "near_miss", ONE)


def test_the_door_refuses_bad_dials_by_name():
    """Property: the ``[damage.stress]`` door refuses a missing or unknown
    key, a half-life below a tick, a curve upside down and a non-whole Hill
    exponent, naming the key. Breaks if a bad dial is taken silently."""
    stress.settings_from_config(_cfg())
    for over, needle in ((dict(stress_per_hp=None), "missing 'stress_per_hp'"),
                         (dict(extra=1.0), "unknown key"),
                         (dict(stress_half_life_s=0.01), "stress_half_life_s"),
                         (dict(heart_rate_max=50.0), "heart_rate_rest"),
                         (dict(heart_hill_n=2.5), "heart_hill_n"),
                         (dict(heart_half_stress=0.0), "heart_half_stress")):
        with pytest.raises(ValueError, match=needle):
            stress.settings_from_config(_cfg(**over))


# ---------------------------------------------------------------------------
# The heart rate
# ---------------------------------------------------------------------------
def test_the_hill_curve_has_its_shape():
    """Property: the heart rate is monotone in the stress, the rest rate at
    0, half way at the half stress, never above the max, and approaches the
    max at a high stress. Breaks if the table is built off its formula, the
    lookup interpolates past an entry, or a rounding lets it overshoot."""
    s = stress.get_settings()
    h = s.hill
    assert stress.heart_rate_q(0) == h.rest_q
    assert stress.heart_rate_q(h.half_q) == (h.rest_q + h.max_q) // 2 \
        or abs(stress.heart_rate_q(h.half_q) - (h.rest_q + h.max_q) / 2) <= 1
    prev = -1
    for k in range(0, 40000, 7):
        v = stress.heart_rate_q(k * ONE // 37)
        assert prev <= v <= h.max_q
        prev = v
    assert stress.heart_rate_q(10**4 * stress.LEVEL_ONE) > h.max_q - (h.max_q - h.rest_q) // 1000
    # a unit's rate is the curve at its level
    u = _marine()
    u.stress_q = 3 * h.half_q
    assert stress.heart_rate_q(u) == stress.heart_rate_q(3 * h.half_q)


# ---------------------------------------------------------------------------
# The scans: one writer, one door, no reader
# ---------------------------------------------------------------------------
_SRC = ROOT / "src"


def _py_files():
    return sorted(p for p in _SRC.rglob("*.py") if "__pycache__" not in p.parts)


def test_only_the_stress_module_writes_the_level():
    """Property: nothing in ``src/`` but ``simulation/stress.py`` assigns
    ``stress_q`` (the schema's default aside). Breaks if a second writer
    appears (a law adding to the level itself instead of a producer row)."""
    bad = []
    for p in _py_files():
        if p.name in ("stress.py", "unit_schema.py"):
            continue
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                targets = [node.target]
            for t in targets:
                for sub in ast.walk(t):
                    if isinstance(sub, ast.Attribute) and sub.attr == "stress_q":
                        bad.append(f"{p.relative_to(ROOT)}:{node.lineno}")
    assert not bad, f"stress_q written outside stress.py: {bad}"


def test_the_producer_door_is_called_from_apply_packet_alone():
    """Property: the only callers of ``stress.produce`` / ``on_damage`` in
    ``src/`` are ``damage.apply_packet`` (the damage producer) and the stress
    module itself. Breaks if another site feeds stress without its own
    producer row and ruling."""
    sites = []
    for p in _py_files():
        if p.name == "stress.py":
            continue
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr in ("produce", "on_damage"):
                sites.append(p.name)
    assert sites == ["damage.py"], sites


#: The files of ``src/`` that may name the stress level or the heart rate:
#: the law, the damage producer, the schema, the tick's call, the death and
#: conversion clears, the device tables and the device kernels' Python face.
_STRESS_ALLOWED = {"stress.py", "damage.py", "unit_schema.py", "simulation.py",
                   "ai_zombie.py", "unit_tables.py", "unit_kernels.py",
                   "clearing.py"}


def test_nothing_in_the_game_reads_stress_or_the_heart_rate():
    """Property (Erik, 2026-11-17: "readout in first patch is good, we wait
    with effects"): no module of ``src/`` outside the law, its producer, the
    schema, the clears and the device face names ``stress_q``,
    ``heart_rate_q`` or the stress module -- no bot, observation, reward or
    law reads it (the readouts live in ``ui/`` and ``renderer/``). Breaks
    when an effect of stress lands without its ruling (it then adds its file
    here, in the patch that rules it)."""
    hits = []
    for p in _py_files():
        if p.name in _STRESS_ALLOWED:
            continue
        text = p.read_text(encoding="utf-8")
        for needle in ("stress_q", "heart_rate", "simulation import stress",
                       "simulation.stress"):
            if needle in text:
                hits.append(f"{p.relative_to(ROOT)}: {needle}")
    assert not hits, hits


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
