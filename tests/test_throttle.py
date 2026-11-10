"""G1p (gameplay pack v1, #183) -- continuous speed: ``set_move_dir``'s THROTTLE.

The commanded speed is ``v_walk + t * (v_sprint - v_walk)`` tiles per second
from the walk and sprint cadence dials (``movement.throttle_tiles_per_tick``),
each reduced by terrain and wind. Properties:

* throttle 0 IS the old walk and ``ONE`` the old sprint, BIT FOR BIT, on the
  law and on the sim's movement step (against the pre-G1p expression,
  ``1.0 / default_speed(samples, cadence)``), over terrain and wind costs;
* the speed is monotone in the throttle;
* the door clamps (as a throw's reach is clamped) and the tap records the
  clamped value.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tools", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402

from level_loader import LevelData  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation.intents import FP_ONE  # noqa: E402
from simulation.movement import (FootprintSamples, SCALE_ONE,  # noqa: E402
                                 THROTTLE_ONE, default_speed,
                                 throttle_tiles_per_tick)
from simulation.orders import ORDER_MOVE_ATTACK, ORDER_SPRINT  # noqa: E402
from simulation.ruleset import ContinuousRealtime  # noqa: E402
from simulation.simulation import _ticks_per_tile  # noqa: E402
from simulation.unit import Unit  # noqa: E402

WALK = _ticks_per_tile(ORDER_MOVE_ATTACK)
SPRINT = _ticks_per_tile(ORDER_SPRINT)

#: Footprint samples spanning the terrain + wind reductions: open air,
#: furniture, a mixed footprint, a gale's half-speed scale, an odd scale.
SAMPLES = [
    FootprintSamples(mobility=[1000] * 9),
    FootprintSamples(mobility=[400] * 9),
    FootprintSamples(mobility=[1000, 400, 700, 1000, 1000, 250, 1000, 900, 1]),
    FootprintSamples(mobility=[1000] * 9, wind_scale_q16=SCALE_ONE // 2),
    FootprintSamples(mobility=[600] * 9, wind_scale_q16=40961),
]


def _costs(s):
    return default_speed(s, WALK), default_speed(s, SPRINT)


@pytest.mark.parametrize("s", SAMPLES)
def test_throttle_zero_is_the_walk_and_one_the_sprint_bit_for_bit(s):
    """Property: at throttle 0 the law returns exactly the pre-G1p walk
    step ``1.0 / walk_cost`` and at ONE exactly the sprint's ``1.0 /
    sprint_cost`` (the same float, not just close), on every terrain and
    wind cost. Breaks if the interpolation is formed in a way that rounds
    differently at an end (``a + t * (b - a)`` does at t = 1)."""
    cw, cs = _costs(s)
    assert throttle_tiles_per_tick(cw, cs, 0) == 1.0 / cw
    assert throttle_tiles_per_tick(cw, cs, THROTTLE_ONE) == 1.0 / cs


@pytest.mark.parametrize("s", SAMPLES)
def test_the_speed_is_monotone_in_the_throttle(s):
    """Property: more throttle is never slower (the sprint cadence is the
    faster dial): over every 37th throttle value and a seeded random set, the
    speed never decreases, and it spans walk..sprint. Breaks if the law is
    not linear in tiles per second or rounds non-monotonically."""
    cw, cs = _costs(s)
    ts = sorted(set(range(0, THROTTLE_ONE + 1, 37))
                | {int(t) for t in np.random.default_rng(1).integers(
                    0, THROTTLE_ONE + 1, size=2000)}
                | {THROTTLE_ONE - 1, THROTTLE_ONE})
    v = [throttle_tiles_per_tick(cw, cs, t) for t in ts]
    assert all(b >= a for a, b in zip(v, v[1:]))
    assert v[0] == 1.0 / cw and v[-1] == 1.0 / cs
    if cs < cw:
        assert v[-1] > v[0]


def test_the_law_refuses_a_throttle_outside_its_range():
    """Property: the law itself refuses a throttle the door should have
    clamped (never extrapolates past the sprint). Breaks if it silently
    extrapolates."""
    with pytest.raises(ValueError):
        throttle_tiles_per_tick(18, 8, THROTTLE_ONE + 1)
    with pytest.raises(ValueError):
        throttle_tiles_per_tick(18, 8, -1)


def _arena(h=16, w=16):
    tm = np.ones((h, w), dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = 4
    return LevelData(name="throttle_open", version="1", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


def _step_once(throttle_q):
    sim = Simulation(_arena(), seed=1, breach_physics=bp,
                     enable_recorder=False, ruleset=ContinuousRealtime())
    sim.set_paused(False)
    uid = sim.add_unit(Unit("M", x=6, y=6, team=0))
    u = sim.get_unit(uid)
    x0 = u.x
    s = FootprintSamples(mobility=sim.gmap.footprint_mobility(
        u.tile_y, u.tile_x, u.footprint))
    sim.set_move_dir(uid, FP_ONE, 0, throttle_q)
    sim.step()
    return sim, u, x0, s


def test_the_movement_step_at_throttle_zero_and_one_is_the_old_walk_and_sprint():
    """Property: through the sim's movement step (slot 3), one tick at
    throttle 0 advances exactly the pre-G1p walk step and at ONE exactly the
    pre-G1p sprint step (``x0 + 1.0 * (1.0 / cost)``). Breaks if the step
    reads another cadence, or the throttle is applied anywhere but the law."""
    _, u, x0, s = _step_once(0)
    assert u.x == x0 + 1.0 * (1.0 / default_speed(s, WALK))
    _, u, x0, s = _step_once(THROTTLE_ONE)
    assert u.x == x0 + 1.0 * (1.0 / default_speed(s, SPRINT))


def test_the_door_clamps_the_throttle_and_the_tap_records_the_clamp():
    """Property: ``set_move_dir`` clamps a throttle below 0 to 0 and above
    ONE to ONE (as the neighbouring throw clamps its reach: no intent of the
    facade refuses), the unit moves as at the clamp, and the replay tap sees
    the clamped value. Breaks if an out-of-range throttle reaches the law or
    the recording holds a value the sim did not use."""
    import match_replay as mr
    from simulation.match import MatchSpec
    for given, clamped in ((-5, 0), (THROTTLE_ONE + 7, THROTTLE_ONE),
                           (2 ** 40, THROTTLE_ONE)):
        sim, u, x0, _ = _step_once(given)
        _, ref, rx0, _ = _step_once(clamped)
        assert u.live_move_dir.throttle_q == clamped
        assert u.x - x0 == ref.x - rx0
        rec = mr.Recording(level="x", spec=MatchSpec(teams=(), seed=1), seed=1,
                           checkpoint_every_ticks=60)
        sim.attach_intent_log(mr.IntentLog(rec))
        sim.set_move_dir(u.id, FP_ONE, 0, given)
        assert rec.intents[sim.tick][-1] == ("set_move_dir",
                                             (u.id, FP_ONE, 0, clamped))
