"""G1p (gameplay pack v1, #183) -- the turn-rate cap (``simulation/turning.py``).

``set_aim`` names the facing a unit WANTS; each tick its facing turns toward
it by at most the unit's turn rate, the shorter way round, in integer Q16.16
radians (the kit's ``atan2_q16`` currency), reaching it exactly. Properties:

* the law: one tick never turns more than the cap, always the shorter way,
  and lands on the aim exactly (no overshoot, no residual) -- on a sweep of
  angles across the wrap, and through the sim's AIM slot;
* the door refuses a missing or non-positive rate and quantizes once;
* the bots never fire while this tick's turn leaves them outside their
  weapon's spread of the aim (the trigger discipline);
* THE GATE of the patch: with the cap at half a turn per tick, the pinned
  ship_demo match is the PRE-G1p match exactly -- so the cap is the only
  thing that moved its re-pinned digests.
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

from config import CFG, Namespace  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation import turning, unit_fixed  # noqa: E402
from simulation.intents import FP_ONE  # noqa: E402
from simulation.ruleset import ContinuousRealtime  # noqa: E402
from simulation.turning import (PI_Q16, TWO_PI_Q16, turn_rate_q16,  # noqa: E402
                                turn_toward_q16, wrap_q16)
from simulation.unit import Unit  # noqa: E402

#: 15 degrees in Q16.16 radians -- the shipped placeholder's per-tick cap.
CAP_15 = unit_fixed.quantize_scalar(math.radians(15.0))


def _dist(a, b) -> int:
    """The shorter-way angular distance |a - b| in Q16.16 radians."""
    return abs(wrap_q16(a - b))


# ---------------------------------------------------------------------------
# (a) the law
# ---------------------------------------------------------------------------
def test_the_kit_constants_are_the_kits_own():
    """Property: PI_Q16 is the trig kit's own pi (atan2_q16 of the negative
    x axis), so both ends of the facing range name one angle under the wrap.
    Breaks if the constant is re-derived from a float pi or the period is not
    twice it."""
    assert unit_fixed.atan2_q16(0, -FP_ONE) == PI_Q16
    assert TWO_PI_Q16 == 2 * PI_Q16
    assert wrap_q16(PI_Q16) == PI_Q16 and wrap_q16(-PI_Q16) == PI_Q16


@pytest.mark.parametrize("cap", [1, 977, CAP_15, PI_Q16 // 2, PI_Q16])
def test_a_turn_never_exceeds_the_cap_takes_the_shorter_way_and_lands(cap):
    """Property: for any facing and any aim (a seeded sweep across the wrap,
    plus the range ends and the exact half turn), one tick moves the facing
    by at most ``cap`` along the shorter arc, never past the aim, and
    repeated ticks reach the aim EXACTLY in ceil(distance / cap) ticks, each
    tick closing the distance by exactly the cap until the last (walked to
    the end for aims within 64 caps, the first 65 ticks beyond). Breaks if
    the turn takes the long way round the wrap, overshoots, stops a residual
    short, or steps by anything but the cap."""
    rng = np.random.default_rng(183)
    angles = [-PI_Q16, PI_Q16, 0, 1, -1, PI_Q16 - 1, -PI_Q16 + 1]
    angles += [int(a) for a in rng.integers(-PI_Q16, PI_Q16 + 1, size=60)]
    for cur in angles:
        for target in angles[:12] + [cur + PI_Q16, cur - PI_Q16 + 1,
                                     cur + 3 * cap // 2, cur - 5 * cap // 2]:
            target = turning.normalize_q16(target)
            d0 = _dist(target, cur)
            if d0 > 64 * cap:
                # far for this cap: walk the first ticks only (the step law
                # is the same on every tick; the near targets above walk in)
                d0 = None
            f = cur
            for k in range(1, 66 if d0 is None else (d0 + cap - 1) // cap + 2):
                nxt = turn_toward_q16(f, target, cap)
                step = wrap_q16(nxt - f)
                assert abs(step) <= cap
                # the shorter way: the step goes the way of the short arc
                if nxt != target:
                    assert abs(step) == cap
                    assert step == (cap if wrap_q16(target - f) > 0
                                    else -cap)
                    assert _dist(target, nxt) == _dist(target, f) - cap
                f = nxt
                if f == target:
                    break
            if d0 is not None:
                assert f == target, (cur, target, cap)
                assert k == max(1, (d0 + cap - 1) // cap)


def test_a_cap_of_half_a_turn_is_the_instant_facing():
    """Property: at a cap of half a turn (or more) every aim is reached in
    one tick -- the pre-G1p law. Breaks if the wrap leaves any aim farther
    than half a turn."""
    rng = np.random.default_rng(7)
    for cur, target in rng.integers(-PI_Q16, PI_Q16 + 1, size=(500, 2)):
        assert turn_toward_q16(int(cur), int(target), PI_Q16) == int(target)


def _arena(h=20, w=20):
    from level_loader import LevelData
    tm = np.ones((h, w), dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = 4
    return LevelData(name="turn_open", version="1", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


def test_the_sim_turns_a_unit_at_its_cap_and_reaches_the_aim_exactly():
    """Property: through ``set_aim`` and the sim's AIM slot, a marine facing
    north that aims south-west turns by at most its cap per tick, the shorter
    way (counter-clockwise from north, through west), and ends on exactly the
    facing the instant law gave (``atan2_rad`` of the aim). Breaks if the
    slot writes the aim at once, turns the long way, or ends a residual away."""
    sim = Simulation(_arena(), seed=1, breach_physics=bp,
                     enable_recorder=False, ruleset=ContinuousRealtime())
    sim.set_paused(False)
    uid = sim.add_unit(Unit("M", x=8, y=8, team=0))
    u = sim.get_unit(uid)
    cap = sim.unit_turn_rate_q16(u)
    assert cap == turn_rate_q16(CFG)
    dx_q, dy_q = -46341, 46341                      # south-west (y down)
    want = unit_fixed.atan2_rad(-dy_q / FP_ONE, dx_q / FP_ONE)
    prev = unit_fixed.quantize_scalar(u.facing)
    for _ in range(40):
        sim.set_aim(uid, dx_q, dy_q)
        sim.step()
        now = unit_fixed.quantize_scalar(u.facing)
        assert abs(wrap_q16(now - prev)) <= cap
        assert wrap_q16(now - prev) >= 0            # CCW: north -> west -> SW
        prev = now
        if u.facing == want:
            break
    assert u.facing == want


# ---------------------------------------------------------------------------
# (b) the door
# ---------------------------------------------------------------------------
def _cfg(rate, tps=24):
    return Namespace({"marine": ({} if rate is None
                                 else {"turn_rate_deg_per_s": rate}),
                      "clock": {"ticks_per_second": tps}})


def test_the_door_quantizes_the_shipped_rate_once():
    """Property: 360 deg/s at 24 tps is quantize(15 deg) per tick. Breaks if
    the door reads another key, forgets the tick rate or rounds another way."""
    assert turn_rate_q16(_cfg(360.0)) == CAP_15
    assert turn_rate_q16(_cfg(4320.0)) == PI_Q16      # half a turn per tick


@pytest.mark.parametrize("bad", [None, 0, -10.0, True, "fast", 1e-9])
def test_the_door_refuses_a_bad_rate_by_name(bad):
    """Property: a missing, non-positive, non-numeric or sub-count rate is
    refused naming ``turn_rate_deg_per_s``. Breaks if a typo silently turns
    units at a default rate (or never)."""
    with pytest.raises(ValueError, match="turn_rate_deg_per_s"):
        turn_rate_q16(_cfg(bad))


# ---------------------------------------------------------------------------
# (c) the bots' trigger discipline
# ---------------------------------------------------------------------------
def test_a_bot_fires_only_once_its_turn_lands_within_the_spread(tmp_path):
    """Property: a bot whose seen foe is ~22 degrees off its facing holds the
    trigger released while this tick's turn leaves it outside its weapon's
    spread of the aim, and pulls it once inside; every tick it pulls the
    trigger, the facing the tick gives it is within the spread. Breaks if
    the bot fires mid-turn (spraying the arc) or never fires."""
    from test_bots import BotTeam, _level, _sim
    lvl = _level(tmp_path, "turnfire")
    sim, (bot, foe) = _sim(lvl, [(0, 10, 25), (1, 16, 10)])
    team = BotTeam(0)
    spread_q = team._weapon_spread_q(bot)
    pulls = []
    for _ in range(12):
        team.handle_frame(sim, None)
        pulled = bool(bot.live_trigger)
        aim = bot.live_aim
        sim.step()
        if pulled:
            want = unit_fixed.atan2_q16(-aim.dy_q, aim.dx_q)
            assert _dist(want, unit_fixed.quantize_scalar(bot.facing)) \
                <= spread_q
        pulls.append(pulled)
    assert pulls[0] is False          # 22 deg off, a 15 deg cap: not yet
    assert any(pulls)


# ---------------------------------------------------------------------------
# (d) THE GATE: at half a turn per tick the pinned match is the pre-G1p one
# ---------------------------------------------------------------------------
#: ship_demo seed 1 at an INSTANT turn: the pre-G1p match (main da6a8d9e,
#: win 1 at 722, field da05242a / tick f35ed2be / units 4408c874) until
#: #183 G2p's carbine rate (one round per pull, 12 a second) moved the match
#: for another cause -- the retirement this gate's docstring announced. The
#: values below are the same instant-turn run RE-MEASURED at the G2p tree
#: (2026-11-10): they hold the turn cap as the only G1p cause, as before.
#: (G1p's instant-turn values were ticks 722, field da05242a99122428c903ad8a
#: 96e9373a0e12006c03abdbec1e849f4aca7dc4b0, tick f35ed2beadb77141ccea20d944
#: a70a66e93c70818ae4c44bbf6ba1edc01024b3, units 4408c8740425593b0d47d35e7d0
#: 658c4efbd62caed87f5fd69a9e8f894ac02dd.)
#: RE-MEASURED again at #183 G1p-c (2026-11-10, speeds in m/s): the same
#: instant-turn run at the new speeds. SOLE CAUSE, verified on the same
#: tree: with the speed table at the old cadences (walk 18, full 8, zombie
#: 14 ticks per tile, the direction factors 1.0) the instant-turn run gives
#: exactly the G2p values (win 1 at 700, field 8deadde6d6d9c9f3f87a3a565fb1
#: bc57054d9e6882f79abbbcd79ce01a3d52e2, tick e4e39176f31246219319a3c72bbc
#: 598f5da16aa7d02def7469fedde2da966324, units bc1166d31b6df445503733eca435
#: f24ac2ad48fe788d50b5eb89d19460f7a3a6).
#: RE-MEASURED again at #183 G3p (2026-11-10, the throw wind-up): the same
#: instant-turn run with the grenade leaving the hand 7 ticks after the verb.
#: SOLE CAUSE, verified on the same tree: with `throw_windup_seconds = 0` the
#: instant-turn run gives exactly the G1p-c values (win 1 at 237, field
#: 305c4e13667c1262869ab2d8d4e15e6c6fdff661d8a6cadc49b7ce05d499876e, tick
#: 7e58771a0fa44b9539260472ab5cec22edc1a60e885c1e9c39ae69fd60f173df, units
#: e0b47942bd115b8d72070494597061f7727126ef442caf452dc62cb42e5e7233).
#: RE-MEASURED again at #183 G4p (2026-11-10, the focus-fire law): the same
#: instant-turn run with exposure live (the shipped [damage.exposure] dials).
#: SOLE CAUSE, verified on the same tree: with `k_expose = 0` the
#: instant-turn run gives exactly the G3p values (win 1 at 244, field
#: 8cb12fa8d9b2028a3507013627b3a46acf7fbd51ce68ad6129da34949caf3a03, tick
#: b435f76ecdedf393e6417be05caf6f2a0988d7f903cde82a1307afd6acbe47ba, units
#: 8c9d7b3af1423b147386f066dd36cb135c28e4c12a73152b447713ca39d42937).
#: RE-MEASURED at #183 G5p-a1 commit 1 (2026-11-10, the turning law): the
#: instant turn is now a top speed of half a turn per tick AND unlimited
#: acceleration (a >= 2w; the run sets both). Outcome, ticks and field_digest
#: UNCHANGED; tick_digest and the unit stream move. SOLE CAUSES, verified on
#: the same tree: with every non-aim facing writer placing the facing at
#: once again and the old float representation hashed, the run gives exactly
#: the G4p values above (d74c0c9c / cfc087f4 / 030e28aa); with the
#: representation alone undone only the unit stream moves -- even at an
#: instant turn the zombie walk's facing is hashed a tick later (its target
#: is set in slot 5, turned in the next tick's slot 1a), while every read of
#: it in play comes after that slot 1a and is unchanged.
#: (was tick cfc087f4ff68f8395881ba629d4de73f70bb63edd0088789bcf8281e3decb819,
#: units 030e28aaf6a4d3c64242ff617bb9ab554ffbd91449b1d3b8667f9201f94ab19e)
#: RE-MEASURED at #183 G5p-a2 re-pin 1 (2026-11-10, the unit impulse law built with every input at zero -- [impulse]
#: bullet_push_mult = recoil_mult = 0, no shove -- and the MOVE_DIR walk
#: on the Q16 grid (the step is the speed law's rational rounded ONCE to
#: Q16 tiles per tick, critique B M6)):
#: the same instant-turn run on the new tree. Outcome and ticks UNCHANGED
#: (win 1 at 244); field_digest, tick_digest and the unit stream move. SOLE CAUSE, verified on the same tree: with the pre-G5p-a2 float walk
#: step reinstated (a scratch monkeypatch of Simulation._step_move_dir)
#: this run gives EXACTLY the previous values -- slot 1a's integration,
#: slot 3's displacement loop, the effort member, the digest keys, the
#: events and the door change nothing while every input is off.
#: (was field d74c0c9c3e4efb2bab446009bd7b9c8f7e05ca1fa976b29fa45df0de63d2e2a0,
#: tick 74f88f8db69c11c9ac93accdd48a49ac9b90e046f22b3f57dc9421f8dde846bf,
#: units a402a66787dde297706acd243a3b2c108c854339f2bf30d7ef074bd514b1e53c)
#: RE-MEASURED at #183 G5p-a2 re-pin 2 (2026-11-10, hits push and guns recoil: [impulse] bullet_push_mult =
#: recoil_mult = 1.0, the RULED real momentum -- a carbine round 3.6 N s on
#: the target, its recoil on the shooter):
#: the same instant-turn run with hits pushing and guns recoiling. Outcome,
#: ticks and field_digest UNCHANGED; tick_digest and the unit stream move.
#: SOLE CAUSE, verified on the same tree: with both
#: multipliers at 0 this run gives exactly re-pin 1's values (the comment above).
#: (was tick 6363856b30393dbcfef1c9801f03b3cfa423652cbd8f95f72f269ab59a394f36,
#: units 02e8717ae42a475dd629ff355596da554f5771f91d258b8da83143e27af0dcb7)
#: RE-MEASURED at #183 G5p-b (2026-11-10, the wind and the blast onto the
#: unit impulse law: the dynamic pressure at each footprint is a force
#: q * drag_area through unit_impulse.add_wind; #140's cadence scale and
#: bespoke knockdown DELETED in the same commit).
#: The instant-turn run's WINNER CHANGES: win 0 at 244 (was win 1 at
#: 244); all three digests and the unit stream move.
#: SOLE CAUSE, verified on the same build: the pre-G5p-b tree (0b57fc6e)
#: with #140's wind law off (wind_q_slow_pa / wind_q_fall_pa = 1e15 /
#: 2e15) and this tree with the wind's input off ([impulse]
#: wind_push_mult = 0) give exactly the same values (win 0 at 244, 55a525f9 / 23b0bdfe / 1af4fccd) --
#: replacing the law is the whole change.
#: So the change of winner is #140's law leaving, not the force: with
#: it off the frag of ticks 218-219 floors nobody (the old law floored
#: units 1, 3, 6, 7, 10, 11 on q alone), units 6 and 7 die on ticks
#: 219-220 instead of surviving to 244, and team 0 wins; the force then
#: pushes 1, 3, 7, 10, 11 by up to 1.1 tiles without changing the
#: winner or the end tick.
#: (was win 1 at 244; field f479f16f80b1071550eda76c4630e5380f15c501f89e
#: 9160c4966fed9b0634ad, tick 16cfecdec6e602d9b547f9d37019b63fd309764dc1
#: cd282ef9665a5c597453c6, units 72b7eba882b7bcb97c382e525690666e83135343
#: c8f045d042c93671b130fd1c)
_PRE_G1P_SHIP_DEMO = dict(
    outcome="win", team=0, ticks=244,
    field_digest="a669d9431e8ba01e9c13d8dc9397ac91"
                 "100eeecc2f1e69d41c6821c6a8f39cae",
    tick_digest="94c39cffdb18d4740053e7d643d91b2f"
                "eb453f468834b781dd752d4ace8e2e5c",
    unit_stream="13124e00feec4fe69d07512b624595d0"
                "0f9bb797a649ae71d2dbe0a62a431cf0")


def test_at_half_a_turn_per_tick_the_pinned_match_is_the_pre_g1p_match(
        monkeypatch):
    """Property: with ``[marine] turn_rate_deg_per_s`` at half a turn per
    tick, ship_demo seed 1 (bots vs bots + the horde) plays the pre-G1p match
    tick for tick -- the throttle (bots walk at 0), the format bump, the
    bots' trigger discipline and the direct-control scheme change nothing in
    it, so the turn cap is the ONLY cause of the G1p re-pin. Breaks if
    throttle 0 is not today's walk, if the bots fire differently at an
    instant turn, or if the cap at half a turn is not the instant facing. The
    next DELIBERATE re-pin of this match (another cause) retires this gate's
    values with it."""
    import level_loader
    from run_match import run_match
    from test_zombie_control import _UnitStream
    tps = int(CFG.clock.ticks_per_second)
    monkeypatch.setattr(CFG.marine, "turn_rate_deg_per_s", 180.0 * tps)
    # G5p-a1: an instant turn is also an unlimited acceleration (a >= 2w).
    monkeypatch.setattr(CFG.marine, "turn_accel_rad_per_s2", 1e7)
    level = level_loader.load("ship_demo", str(ROOT / "levels"))
    stream = _UnitStream()
    o = run_match(level, on_tick=stream)
    got = dict(outcome=o.result.outcome, team=o.result.team,
               ticks=o.result.ticks, field_digest=o.field_digest,
               tick_digest=o.tick_digest, unit_stream=stream.h.hexdigest())
    assert got == _PRE_G1P_SHIP_DEMO
