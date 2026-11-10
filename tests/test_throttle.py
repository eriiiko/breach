"""G1p-c (gameplay pack v1, #183) -- the continuous path's speed law.

``set_move_dir``'s THROTTLE between the ``[speeds]`` table's marine walk and
full speed (m/s, quantized once per map into Q16 ticks per tile), with Erik's
direction rule (2026-11-10)::

    speed = f(theta) * v_walk + throttle * (v_full - v_walk) * max(0, cos theta)
    f = 1 forward, sideways_frac at 90 deg, backward_frac at 180, linear in cos

(``movement.move_speed_tiles_per_tick``; theta between the move and the
facing, ``movement.facing_cos_q16``).

REPLACED PROPERTY (G1p-c, deliberately): G1p's version of this file pinned
"throttle 0 / ONE equal the old walk / sprint cadence dials bit for bit". The
continuous path no longer reads those dials (they are WEGO's alone), so that
property is gone; its successors are below: throttle 0 forward IS the table's
walk and ONE its full speed (exactly the rounded rationals), monotone in the
throttle and in cos theta, nobody sprints backwards or sideways, terrain and
wind scale the new base exactly as they scaled the old one, and the door
still clamps.
"""
from __future__ import annotations

import math
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

from config import CFG  # noqa: E402
from level_loader import LevelData  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation.intents import FP_ONE  # noqa: E402
from simulation.movement import (FootprintSamples, SCALE_ONE,  # noqa: E402
                                 THROTTLE_ONE, default_speed, facing_cos_q16,
                                 move_speed_tiles_per_tick)
from simulation.ruleset import ContinuousRealtime  # noqa: E402
from simulation.speeds import speed_settings, speed_table  # noqa: E402
from simulation.turning import PI_Q16  # noqa: E402
from simulation.unit import Unit  # noqa: E402
from simulation.unit_fixed import dequantize_scalar  # noqa: E402
from simulation.turning import place as _place_facing  # noqa: E402
from simulation.unit_fixed import quantize_scalar as _q_rad  # noqa: E402

#: The shipped table on the shipped 0.333 m tile.
TABLE = speed_table(CFG, 0.333)
CW, CF, SIDE, BACK = (TABLE.walk_cost_q, TABLE.full_cost_q, TABLE.side_q,
                      TABLE.back_q)

#: Footprint samples spanning the terrain + wind reductions: open air,
#: furniture, a mixed footprint, a gale's half-speed scale, an odd scale.
SAMPLES = [
    FootprintSamples(mobility=[1000] * 9),
    FootprintSamples(mobility=[400] * 9),
    FootprintSamples(mobility=[1000, 400, 700, 1000, 1000, 250, 1000, 900, 1]),
    FootprintSamples(mobility=[1000] * 9, wind_scale_q16=SCALE_ONE // 2),
    FootprintSamples(mobility=[600] * 9, wind_scale_q16=40961),
]

#: Throttles: the ends, a 37-step sweep, a seeded random set.
THROTTLES = sorted(set(range(0, THROTTLE_ONE + 1, 997))
                   | {int(t) for t in np.random.default_rng(1).integers(
                       0, THROTTLE_ONE + 1, size=300)}
                   | {0, 1, THROTTLE_ONE - 1, THROTTLE_ONE})


def _v(t, c, cw=CW, cf=CF):
    return move_speed_tiles_per_tick(cw, cf, SIDE, BACK, t, c)


def test_forward_throttle_zero_is_the_walk_and_one_the_full_speed():
    """Property: on open floor in calm air, moving along the facing, throttle
    0 is EXACTLY the quantized ``marine_walk_mps`` (``ONE / walk_cost_q``
    tiles a tick) and THROTTLE_ONE exactly the quantized ``marine_full_mps``,
    and in m/s those are the config's numbers to within one Q16 count of the
    cost. Breaks if the law forms the ends with an extra rounding, reads
    another dial, or the table quantizes the m/s wrongly."""
    assert _v(0, FP_ONE) == FP_ONE / CW
    assert _v(THROTTLE_ONE, FP_ONE) == FP_ONE / CF
    s = speed_settings(CFG)
    tps = CFG.clock.ticks_per_second
    for v, mps, cost in ((_v(0, FP_ONE), s.marine_walk_mps, CW),
                         (_v(THROTTLE_ONE, FP_ONE), s.marine_full_mps, CF)):
        got = v * 0.333 * tps
        assert abs(got - mps) <= mps / cost      # one count of the cost
        assert got == pytest.approx(TABLE.realised_mps(cost), rel=1e-12)


@pytest.mark.parametrize("c", [FP_ONE, FP_ONE // 3, 0, -FP_ONE // 2, -FP_ONE])
@pytest.mark.parametrize("s", SAMPLES)
def test_the_speed_is_monotone_in_the_throttle(s, c):
    """Property: more throttle is never slower, at every angle to the facing
    and on every terrain / wind cost. Breaks if the throttle term can go
    negative (full slower than walk) or the law rounds non-monotonically."""
    cw, cf = default_speed(s, CW), default_speed(s, CF)
    v = [_v(t, c, cw, cf) for t in THROTTLES]
    assert all(b >= a for a, b in zip(v, v[1:]))
    if c > 0:
        assert v[-1] > v[0]


@pytest.mark.parametrize("t", [0, THROTTLE_ONE // 2, THROTTLE_ONE])
def test_the_speed_is_monotone_in_cos_theta(t):
    """Property: turning the move away from the facing never makes a unit
    faster: over every 61st cos value from -ONE to ONE the speed never
    decreases, forward is the fastest and straight backward the slowest.
    Breaks if f(theta) is not monotone (backward_frac above sideways_frac,
    a factor above 1) or the sprint term survives behind the unit."""
    cs = sorted(set(range(-FP_ONE, FP_ONE + 1, 61)) | {-FP_ONE, 0, FP_ONE})
    v = [_v(t, c) for c in cs]
    assert all(b >= a for a, b in zip(v, v[1:]))
    assert v[0] < v[-1]


def test_nobody_sprints_backward_or_sideways():
    """Property (Erik's ruling, 2026-11-10: sprinting is forward only):
    straight backward the speed IS ``backward_frac x walk`` and at a right
    angle ``sideways_frac x walk``, the same at EVERY throttle (each the
    exactly rounded rational ``frac_q / walk_cost_q``). Breaks if the
    throttle term reads ``|cos|`` or ``cos`` without the max, or the
    direction factor is applied after the throttle."""
    for t in THROTTLES:
        assert _v(t, -FP_ONE) == BACK / CW
        assert _v(t, 0) == SIDE / CW
    s = speed_settings(CFG)
    assert BACK / CW == pytest.approx(s.backward_frac * FP_ONE / CW, rel=1e-4)


@pytest.mark.parametrize("s", SAMPLES)
def test_terrain_and_wind_scale_the_new_base_as_they_scaled_the_old(s):
    """Property: terrain mobility and the wind scale reach the new law
    through the SAME function the old cadence went through
    (``default_speed``, here at the Q16 cost currency): the reduced walk is
    exactly ``ONE / default_speed(samples, walk_cost_q)``, and the factor it
    applies to the base speed equals the factor the old integer cadence got
    (18 ticks per tile, the old walk dial) to within that integer
    reduction's own half-up rounding. Breaks if the new law re-derives the
    terrain or wind reduction, drops one of them, or applies it twice."""
    cw, cf = default_speed(s, CW), default_speed(s, CF)
    assert _v(0, FP_ONE, cw, cf) == FP_ONE / cw
    assert _v(THROTTLE_ONE, FP_ONE, cw, cf) == FP_ONE / cf
    plain = FootprintSamples(mobility=[1000] * 9)
    new_factor = _v(0, FP_ONE, cw, cf) / _v(0, FP_ONE)
    old = 18
    old_factor = default_speed(plain, old) / default_speed(s, old)
    # The integer reduction rounds its cost by half a tick per stage (the
    # mobility stage's error then scaled by the wind's 1/scale <= 2): 1.5.
    assert abs(new_factor - old_factor) <= old_factor * 1.5 / default_speed(s, old)
    exact = (sum(s.mobility) / (len(s.mobility) * 1000)) * (s.wind_scale_q16 / SCALE_ONE)
    assert new_factor == pytest.approx(exact, rel=1e-5)


def test_the_cos_is_exact_at_the_cardinal_angles():
    """Property: the kit's cos of the angle between the move and the facing
    is exactly ONE along the facing, -ONE against it and 0 at a right angle
    (so the three exact speeds above are what a unit actually gets), and
    stays in [-ONE, ONE] all round. Breaks if theta is taken without the
    world's y flip, or cos is formed off the kit."""
    east, north = 0, 102944                       # atan2_q16 of (+x), (-y)
    assert facing_cos_q16(FP_ONE, 0, east) == FP_ONE
    assert facing_cos_q16(-FP_ONE, 0, east) == -FP_ONE
    assert facing_cos_q16(0, -FP_ONE, east) == 0       # north, at 90 deg
    assert facing_cos_q16(0, FP_ONE, east) == 0
    assert facing_cos_q16(0, -FP_ONE, north) == FP_ONE
    assert facing_cos_q16(0, FP_ONE, north) == -FP_ONE
    assert facing_cos_q16(FP_ONE, 0, PI_Q16) == -FP_ONE
    for k in range(64):
        a = 2 * math.pi * k / 64
        c = facing_cos_q16(int(math.cos(a) * FP_ONE), int(math.sin(a) * FP_ONE),
                           east)
        assert -FP_ONE <= c <= FP_ONE
        assert abs(c / FP_ONE - math.cos(a)) < 1e-3


def test_the_law_refuses_out_of_range_inputs():
    """Property: the law itself refuses a throttle the door should have
    clamped and a cos outside [-ONE, ONE] (never extrapolates past the full
    speed or the facing). Breaks if it silently extrapolates."""
    for t, c in ((THROTTLE_ONE + 1, FP_ONE), (-1, FP_ONE),
                 (0, FP_ONE + 1), (0, -FP_ONE - 1)):
        with pytest.raises(ValueError):
            _v(t, c)


# ---------------------------------------------------------------------------
# Through the sim's movement step (slot 3)
# ---------------------------------------------------------------------------
def _arena(h=16, w=16):
    tm = np.ones((h, w), dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = 4
    return LevelData(name="throttle_open", version="1", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


def _step_once(throttle_q, move=(FP_ONE, 0), facing_q=None, aim=None):
    """One tick of a unit on open floor; ``facing_q`` / ``aim`` set the
    facing already where the aim names (no turn this tick)."""
    sim = Simulation(_arena(), seed=1, breach_physics=bp,
                     enable_recorder=False, ruleset=ContinuousRealtime())
    sim.set_paused(False)
    uid = sim.add_unit(Unit("M", x=6, y=6, team=0))
    u = sim.get_unit(uid)
    if facing_q is not None:
        _place_facing(u, facing_q)
    if aim is not None:
        sim.set_aim(uid, *aim)
    x0, y0 = u.x, u.y
    sim.set_move_dir(uid, move[0], move[1], throttle_q)
    sim.step()
    return sim, u, x0, y0


def test_the_movement_step_walks_at_the_table_and_slows_off_the_facing():
    """Property: through slot 3, on open floor, one tick advances exactly
    the law's speed for the map's own table (the arena's 1 m tile): forward
    with the aim along the move, the walk at 0 and the full speed at ONE;
    aiming east while moving west, ``backward_frac x walk`` at both
    throttles; aiming north while moving east, ``sideways_frac x walk``.
    Breaks if the step reads another table or tile size, takes theta from
    anything but this tick's facing, or ignores the direction rule."""
    t = speed_table(CFG, 1.0)
    east, north = (FP_ONE, 0), (0, -FP_ONE)
    sim, u, x0, _ = _step_once(0, aim=east, facing_q=0)
    assert sim.speed_table == t
    assert u.x == x0 + FP_ONE / t.walk_cost_q
    _, u, x0, _ = _step_once(THROTTLE_ONE, aim=east, facing_q=0)
    assert u.x == x0 + FP_ONE / t.full_cost_q
    for th in (0, THROTTLE_ONE):
        _, u, x0, _ = _step_once(th, move=(-FP_ONE, 0), aim=east, facing_q=0)
        assert u.x == x0 - t.back_q / t.walk_cost_q
        _, u, x0, _ = _step_once(th, aim=north, facing_q=102944)
        assert u.x == x0 + t.side_q / t.walk_cost_q


def test_with_no_aim_the_direction_rule_reads_the_facing_and_travel_is_the_target():
    """Property (G5p-a1, which deletes the "no aim means theta 0" shortcut):
    a unit holding no aim is held to the direction rule against THIS tick's
    facing like everyone else -- facing north (the spawn facing) and stepping
    east it gets the sideways walk at either throttle, stepping south the
    backward walk -- and the move becomes its facing TARGET (the turning law
    turns it there from the next tick); facing its move it walks at the
    table's walk / full speed. Breaks if the shortcut returns, or the travel
    writes the facing at once."""
    t = speed_table(CFG, 1.0)
    for th in (0, THROTTLE_ONE):
        _, u, x0, _ = _step_once(th, move=(FP_ONE, 0))      # east, no aim
        assert u.x == x0 + t.side_q / t.walk_cost_q
        assert u.facing_q == 102944 and u.facing_target_q == 0
    _, u, _, y0 = _step_once(0, move=(0, FP_ONE))          # south, no aim
    assert u.y == y0 + t.back_q / t.walk_cost_q
    _, u, x0, _ = _step_once(THROTTLE_ONE, move=(FP_ONE, 0), facing_q=0)
    assert u.x == x0 + FP_ONE / t.full_cost_q


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


# ---------------------------------------------------------------------------
# End to end: the real main.py in a hidden window, direct control
# ---------------------------------------------------------------------------
def test_e2e_the_keyboard_walks_runs_strafes_and_backs_at_the_table():
    """Property (the real window, ``--control direct`` on door_test, the
    mouse holding the aim north of the marine): every tick a key moves the
    marine freely it steps EXACTLY the law's speed at that tick's facing --
    R forward, R+A forward at full speed, D / G strafing, F backward -- and
    along / against the aim the runs average the table's m/s (walk, full,
    ``backward_frac x walk``) to within the few degrees the fixed cursor lets
    the aim drift. Ticks against a wall (no step) are left out. Breaks if the
    keys reach the sim with another throttle, the step reads another table,
    or theta is not this tick's facing."""
    import json
    import os
    import subprocess
    from simulation.unit_fixed import quantize_scalar
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run(
        [sys.executable, "tests/_e2e_speed_probe.py", "--level", "door_test",
         "--control", "direct", "--mouse", "640,0@1", "--frames", "240",
         # strafes, a walk, a back-step, then the run (it ends short of
         # the room's north wall)
         "--hold", "D@30-60", "--hold", "G@75-105", "--hold", "R@120-150",
         "--hold", "F@165-195", "--hold", "R@210-230", "--hold", "A@210-230"],
        cwd=str(ROOT), env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    d = json.loads(next(ln for ln in r.stdout.splitlines()
                        if ln.startswith("SPEEDPROBE "))[len("SPEEDPROBE "):])
    tile, tps = d["tile_size_m"], d["tps"]
    t = speed_table(CFG, tile)
    assert d["table"] == dict(walk_cost_q=t.walk_cost_q,
                              full_cost_q=t.full_cost_q, side_q=t.side_q,
                              back_q=t.back_q)
    runs: dict = {}
    for row in d["log"]:
        step = math.hypot(row[3] - row[1], row[4] - row[2])
        if row[6] is not None and step > 0.0:          # moved freely
            runs.setdefault((row[6], row[7], row[8]), []).append((row, step))

    def law(key, row):
        return move_speed_tiles_per_tick(
            t.walk_cost_q, t.full_cost_q, t.side_q, t.back_q, key[2],
            facing_cos_q16(key[0], key[1], quantize_scalar(row[5])))

    north, south = (0, -FP_ONE), (0, FP_ONE)
    table = {(*north, 0): FP_ONE / t.walk_cost_q,
             (*north, THROTTLE_ONE): FP_ONE / t.full_cost_q,
             (*south, 0): t.back_q / t.walk_cost_q,
             (-FP_ONE, 0, 0): None, (FP_ONE, 0, 0): None}
    for key, v in table.items():
        rows = runs.get(key, [])
        assert len(rows) >= 5, (key, len(rows))
        for row, step in rows:
            assert abs(step - law(key, row)) < 1e-12, (key, row)
        if v is not None:
            mps = sum(s for _, s in rows) / len(rows) * tile * tps
            assert abs(mps - v * tile * tps) <= 1e-3 * v * tile * tps,                 (key, mps, v * tile * tps)
