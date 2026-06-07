"""Tick-rate independence of per-second tunables.

The engine consumes integer tick counts, but the tunables are authored
per-second in ``config.toml`` so that ``[clock] ticks_per_second`` can later
be raised (24/60) without silently changing gameplay speed. ``config.py``
derives the integer tick counts via the pure helper ``ticks_from_seconds``.

This test locks two things:

  1. At the shipped 12 tps the derived counts equal the legacy integers
     (attack 9, cover 6, sprint 4, zombie 7, attack_cooldown 12, rifle burst 2)
     and the recorder capacity is 1200 — i.e. the per-second refactor is
     behaviour-identical to the old hard-coded tick values.
  2. The derivation scales correctly at 24 tps (18, 12, 8, 14, 24, 4;
     capacity 2400).

The scaling case exercises the pure helper directly rather than mutating the
global ``CFG`` singleton, keeping the test side-effect free.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_tick_rate.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config import CFG, ticks_from_seconds


# Per-second source values authored in config.toml (kept here so the test is
# self-describing; an accidental edit to config.toml would diverge and fail).
MARINE_ATTACK_S = 0.75
MARINE_COVER_S = 0.5
MARINE_SPRINT_S = 0.33333333
ZOMBIE_S = 0.58333333          # 7/12
ATTACK_COOLDOWN_S = 1.0
RIFLE_BURST_S = 0.16666667     # 2/12
REPLAY_S = 100.0


def test_live_config_is_consistent_with_helper():
    """The live config's derived tick counts match the pure helper at whatever
    ``ticks_per_second`` is shipped (24 currently). This stays green when the
    default tps changes; the legacy-12 equivalence is locked separately in
    ``test_helper_reproduces_legacy_at_12_tps``."""
    tps = CFG.clock.ticks_per_second
    assert tps > 0

    assert CFG.movement.marine_attack_ticks_per_tile == ticks_from_seconds(MARINE_ATTACK_S, tps)
    assert CFG.movement.marine_cover_ticks_per_tile == ticks_from_seconds(MARINE_COVER_S, tps)
    assert CFG.movement.marine_sprint_ticks_per_tile == ticks_from_seconds(MARINE_SPRINT_S, tps)
    assert CFG.zombie.ticks_per_tile == ticks_from_seconds(ZOMBIE_S, tps)
    assert CFG.zombie.attack_cooldown_ticks == ticks_from_seconds(ATTACK_COOLDOWN_S, tps)
    assert CFG.weapons.rifle.burst_interval_ticks == ticks_from_seconds(RIFLE_BURST_S, tps)
    assert CFG.recorder.capacity == round(REPLAY_S * tps)


def test_helper_reproduces_legacy_at_12_tps():
    """The pure derivation reproduces the legacy integers at 12 tps."""
    tps = 12
    assert ticks_from_seconds(MARINE_ATTACK_S, tps) == 9
    assert ticks_from_seconds(MARINE_COVER_S, tps) == 6
    assert ticks_from_seconds(MARINE_SPRINT_S, tps) == 4
    assert ticks_from_seconds(ZOMBIE_S, tps) == 7
    assert ticks_from_seconds(ATTACK_COOLDOWN_S, tps) == 12
    assert ticks_from_seconds(RIFLE_BURST_S, tps) == 2
    assert round(REPLAY_S * tps) == 1200


def test_helper_scales_at_24_tps():
    """Doubling the tick rate doubles the derived counts (same wall-clock speed)."""
    tps = 24
    assert ticks_from_seconds(MARINE_ATTACK_S, tps) == 18
    assert ticks_from_seconds(MARINE_COVER_S, tps) == 12
    assert ticks_from_seconds(MARINE_SPRINT_S, tps) == 8
    assert ticks_from_seconds(ZOMBIE_S, tps) == 14
    assert ticks_from_seconds(ATTACK_COOLDOWN_S, tps) == 24
    assert ticks_from_seconds(RIFLE_BURST_S, tps) == 4
    assert round(REPLAY_S * tps) == 2400


def test_helper_floor_is_one_tick():
    """A non-zero duration never collapses below a single tick."""
    assert ticks_from_seconds(0.001, 12) == 1
    assert ticks_from_seconds(0.0, 12) == 1
