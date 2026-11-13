"""G5p-a1 (#183) -- the turning law: every unit's facing has angular inertia
and ONE writer (``simulation/turning.py``; ``docs/unit_impulse_design_
2026-11-10.md`` §5.1, §13; critique B's B3 and gates G2, G9, G10).

PROPERTIES PROTECTED
  1. For a FIXED target, from rest and from any state the law can reach, the
     facing lands on the target EXACTLY in bounded ticks, the own torque
     never drives |omega| past the top speed, and afterwards facing and omega
     stay put (no limit cycle); from rest it never overshoots. Swept over
     many (start, target, a, w), targets across the wrap included.
  2. An incoming omega ABOVE the top speed or pointing AWAY from the target
     (an outside spin, run 2) slows at exactly ``a`` per tick, comes back and
     settles exactly; it never speeds up past what it came in with.
  3. The turn goes the shorter way; at the shipped dials 90 degrees land in
     11 ticks (0.458 s) and 180 in 17 (0.708 s), as the law predicts.
  4. G2: at ``a >= 2 w`` the law IS G1p's cap ``turn_toward_q16``, bit for
     bit, over a sweep.
  5. The doors: ``turn_accel_rad_per_s2`` quantized once, refused by name
     when missing / non-positive / sub-count / needing too many trials.
  6. G10: ``Unit.facing`` has no setter, and no module of ``src/`` writes the
     facing's state outside ``turning.py`` (whose ``place`` is the one
     constructor-time door, called by ``Unit.__init__``).
  7. One writer, every reason: the travel direction of a walker holding no
     aim, the zombie walk and the attack snap set TARGETS that slot 1a turns
     toward through the law -- never an instant write; a target persists and
     a zero aim holds the last one.
  8. The Speed table's direction rule reads the post-turn facing: a walker
     holding no aim that reverses turns over several ticks, moves at the
     reduced sideways / backward speed meanwhile, then walks at full speed.
  9. G9: under a bots match, the ``facing_after_turn_q16`` a bot reads in
     ``handle_frame`` is the facing the shooting slot fires along, every tick.
"""
from __future__ import annotations

import ast
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
from simulation import simulation as simulation_mod  # noqa: E402
from simulation import turning, unit_fixed  # noqa: E402
from simulation.factions import ZOMBIE_TEAM  # noqa: E402
from simulation.intents import FP_ONE  # noqa: E402
from simulation.ruleset import ContinuousRealtime  # noqa: E402
from simulation.simultaneous import SimultaneousEffects  # noqa: E402
from simulation.turning import (  # noqa: E402
    OMEGA_MAX_Q16, PI_Q16, braking_distance_q16, law_q16, top_speed_within,
    turn_accel_q16, turn_toward_q16, wrap_q16,
)
from simulation.unit import Unit  # noqa: E402

TPS = int(CFG.clock.ticks_per_second)
#: The shipped dials (360 deg/s, 30 rad/s^2 at 24 tps) in the law's currency.
W_SHIP = unit_fixed.quantize_scalar(math.radians(360.0) / TPS)
A_SHIP = unit_fixed.quantize_scalar(30.0 / (TPS * TPS))


def _settle(facing, omega, target, a, w, max_ticks):
    """Run the law to rest on ``target``; returns the trajectory of
    ``(facing, omega)`` up to and including the first tick at rest on it."""
    traj = []
    for _ in range(max_ticks):
        facing, omega = law_q16(facing, omega, target, a, w)
        traj.append((facing, omega))
        if facing == target and omega == 0:
            return traj
    raise AssertionError(f"no rest in {max_ticks} ticks: start "
                         f"{traj[0]}, target {target}, a {a}, w {w}")


def _bound(a, w):
    """A generous tick bound for any reachable start: shed up to a half turn
    per tick of spin at ``a``, then cross a half turn twice."""
    return 4 * (-(-PI_Q16 // min(w, PI_Q16)) + -(-OMEGA_MAX_Q16 // a)) + 16


#: (a, w) pairs swept by the trajectory gates: the shipped dials, a slow and
#: a fast acceleration, a top speed below one tick of acceleration, the
#: instant limit and a large top speed (a sweep over tiny dials would be the
#: same law over hundreds of thousands of ticks; the s(e) gate takes those).
_DIALS = [(A_SHIP, W_SHIP), (977, 17157), (3413, 1000), (40000, 17157),
          (2 * W_SHIP, W_SHIP), (PI_Q16, PI_Q16 // 3), (5000, 60000)]
#: ... and the s(e) gate's, tiny dials included.
_SMALL_DIALS = _DIALS + [(1, 7), (5, 3), (2, 40)]


# ---------------------------------------------------------------------------
# 1 -- the law on a fixed target
# ---------------------------------------------------------------------------
def test_s_is_the_largest_speed_that_can_still_stop_by_the_target():
    """Property: ``s(e)`` brakes within ``e`` and one count more does not
    (unless capped at ``w``), and ``s(e) <= e`` (the step never overshoots).
    Breaks if the closed form searches the wrong segment, rounds up, or
    stops the search early."""
    rng = np.random.default_rng(51)
    for a, w in _SMALL_DIALS:
        w = min(w, PI_Q16)
        for e in list(range(0, 40)) + [int(x) for x in
                                       rng.integers(0, PI_Q16 + 1, 300)]:
            s = top_speed_within(e, a, w)
            assert 0 <= s <= min(e, w)
            assert braking_distance_q16(s, a) <= e
            if s < w:
                assert braking_distance_q16(s + 1, a) > e, (e, a, w, s)


@pytest.mark.parametrize("a,w", _DIALS)
def test_from_rest_a_fixed_target_is_reached_exactly_never_overshot(a, w):
    """Property 1 (from rest): the facing reaches the target exactly in
    bounded ticks, the remaining turn never changes sign (no overshoot),
    |omega| never exceeds ``min(w, pi)``, and once there it stays for good.
    Swept over starts and targets across the wrap. Breaks if the braking
    distance is underestimated (the M3 chatter), the cap is ignored, or the
    landing leaves a residual."""
    rng = np.random.default_rng(183)
    angles = [-PI_Q16 + 1, PI_Q16, 0, 1, -1, PI_Q16 - 1]
    angles += [int(x) for x in rng.integers(-PI_Q16 + 1, PI_Q16 + 1, 24)]
    for start in angles:
        for target in angles[:10] + [turning.normalize_q16(start + PI_Q16),
                                     turning.normalize_q16(start - 3)]:
            e0 = wrap_q16(target - start)
            traj = _settle(start, 0, target, a, w, _bound(a, w))
            for f, om in traj:
                assert abs(om) <= min(w, PI_Q16)
                e = wrap_q16(target - f)
                assert e == 0 or (e > 0) == (e0 > 0), (start, target, a, w)
            f, om = traj[-1]
            for _ in range(5):                  # no limit cycle
                f, om = law_q16(f, om, target, a, w)
                assert (f, om) == (target, 0)


@pytest.mark.parametrize("a,w", _DIALS)
def test_from_any_reachable_state_a_new_target_is_reached_and_held(a, w):
    """Property 1 (any reachable state): turn toward one target for a few
    ticks, then switch to another -- whatever (facing, omega) the first turn
    left, the second lands exactly in bounded ticks, |omega| stays within the
    top speed (own torque only), and the unit stays put. Breaks if a moving
    start can orbit the target or the own torque accelerates past ``w``."""
    rng = np.random.default_rng(7)
    for _ in range(150):
        start, first, second = (int(x) for x in
                                rng.integers(-PI_Q16 + 1, PI_Q16 + 1, 3))
        f, om = start, 0
        for _ in range(int(rng.integers(1, 12))):
            f, om = law_q16(f, om, first, a, w)
        traj = _settle(f, om, second, a, w, _bound(a, w))
        assert all(abs(o) <= min(w, PI_Q16) for _, o in traj)
        f, om = traj[-1]
        for _ in range(5):
            f, om = law_q16(f, om, second, a, w)
            assert (f, om) == (second, 0)


# ---------------------------------------------------------------------------
# 2 -- an outside spin decays at a and returns
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("a,w", _DIALS)
def test_a_spin_above_the_top_speed_or_away_decays_at_a_and_settles(a, w):
    """Property 2: an incoming omega above the top speed, or pointing away
    from the target, loses exactly ``a`` per tick while it points away or
    exceeds what can still stop (never more, never a jump), the unit comes
    back and settles exactly, and |omega| never grows past what it came in
    with (or the top speed). The incoming spin is clamped below half a turn
    per tick. Breaks if the top speed clamps the total omega (an unbooked
    loss), the decay is not ``a``, or the spin aliases."""
    rng = np.random.default_rng(29)
    for _ in range(120):
        start, target = (int(x) for x in
                         rng.integers(-PI_Q16 + 1, PI_Q16 + 1, 2))
        spin = int(rng.integers(-3 * PI_Q16, 3 * PI_Q16))
        f, om = start, spin
        cap = max(min(abs(spin), OMEGA_MAX_Q16), min(w, PI_Q16))
        for _ in range(_bound(a, w)):
            om_in = max(-OMEGA_MAX_Q16, min(OMEGA_MAX_Q16, om))
            e = wrap_q16(target - f)
            f, om = law_q16(f, om, target, a, w)
            assert abs(om) <= cap
            assert abs(om - om_in) <= a
            away = om_in != 0 and (e == 0 or (e > 0) != (om_in > 0))
            if away and abs(om_in) >= a:
                assert om == om_in - (a if om_in > 0 else -a)
            if f == target and om == 0:
                break
        assert (f, om) == (target, 0), (start, target, spin, a, w)


# ---------------------------------------------------------------------------
# 3 -- the shorter way, the shipped times
# ---------------------------------------------------------------------------
def test_the_turn_goes_the_shorter_way():
    """Property 3: from rest the first step goes the way of
    ``wrap(target - facing)`` (an exact half turn goes counter-clockwise).
    Breaks if the law turns the long way round the wrap."""
    rng = np.random.default_rng(3)
    for f, t in rng.integers(-PI_Q16 + 1, PI_Q16 + 1, (400, 2)):
        f, t = int(f), int(t)
        e = wrap_q16(t - f)
        _, om = law_q16(f, 0, t, A_SHIP, W_SHIP)
        assert (om > 0) == (e > 0) and (om == 0) == (e == 0)


def test_the_shipped_dials_turn_90_and_180_degrees_in_the_predicted_time():
    """Property 3: at the shipped 360 deg/s and 30 rad/s^2 the config's
    dials quantize to the law's (w, a), and from rest 90 degrees land in 11
    ticks (0.458 s) and 180 degrees in 17 (0.708 s): within one tick of the
    continuous bang-bang time ``D / w + w / a`` (0.459 s, 0.709 s). Breaks
    if a dial or its quantization moves, or the discrete law loses ticks."""
    cfg = Namespace({"marine": {"turn_rate_deg_per_s": 360.0,
                                "turn_accel_rad_per_s2": 30.0},
                     "clock": {"ticks_per_second": TPS}})
    assert turning.turn_rate_q16(cfg) == W_SHIP
    assert turn_accel_q16(cfg) == A_SHIP
    for deg, want in ((90, 11), (180, 17)):
        target = turning.normalize_q16(
            unit_fixed.quantize_scalar(math.radians(deg)))
        traj = _settle(0, 0, target, A_SHIP, W_SHIP, 100)
        land = next(i + 1 for i, (f, _) in enumerate(traj) if f == target)
        assert land == want
        cont = math.radians(deg) / (2 * math.pi) + (2 * math.pi) / 30.0
        assert abs(land / TPS - cont) <= 1.0 / TPS


# ---------------------------------------------------------------------------
# 4 -- G2: unlimited acceleration is G1p's cap
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("w", [1, 977, W_SHIP, PI_Q16 // 2, PI_Q16])
def test_at_a_at_least_2w_the_law_is_the_g1p_cap_bit_for_bit(w):
    """G2: at ``a >= 2 w`` one tick of the law from any facing, any
    (normalized) target and any own-torque omega equals
    ``turn_toward_q16(facing, target, w)`` exactly, and chained ticks stay
    equal. Breaks if the braking law leaks into the instant limit."""
    rng = np.random.default_rng(2)
    for a in (2 * w, 2 * w + 1, 50 * w):
        for _ in range(400):
            f, t = (turning.normalize_q16(int(x)) for x in
                    rng.integers(-PI_Q16, PI_Q16 + 1, 2))
            om = int(rng.integers(-w, w + 1))
            g, g_om = f, om
            for _ in range(4):
                nxt, g_om = law_q16(g, g_om, t, a, w)
                assert nxt == turn_toward_q16(g, t, w), (g, t, w, a)
                g = nxt


# ---------------------------------------------------------------------------
# 5 -- the door
# ---------------------------------------------------------------------------
def _cfg(rate=360.0, accel=30.0):
    m = {"turn_rate_deg_per_s": rate}
    if accel is not None:
        m["turn_accel_rad_per_s2"] = accel
    return Namespace({"marine": m, "clock": {"ticks_per_second": TPS}})


@pytest.mark.parametrize("bad", [None, 0, -1.0, True, "fast", 1e-9, 0.2])
def test_the_accel_door_refuses_by_name(bad):
    """Property 5: a missing, non-positive, non-numeric, sub-count or
    too-slow acceleration (more than ``TURN_TRIALS_MAX`` trials against the
    top speed: 0.2 rad/s^2 at 360 deg/s) is refused naming
    ``turn_accel_rad_per_s2``. Breaks if a typo silently turns units at a
    default, or the trial count is unbounded."""
    with pytest.raises(ValueError, match="turn_accel_rad_per_s2"):
        turn_accel_q16(_cfg(accel=bad))


def test_the_shipped_config_bounds_the_trials():
    """Property 5: the shipped config passes the door, and its trial count
    ``ceil(min(w, pi) / a)`` is within the bound. Breaks if the shipped
    dials need an unbounded search."""
    w = turning.turn_rate_q16(CFG)
    a = turn_accel_q16(CFG)
    assert -(-min(w, PI_Q16) // a) <= turning.TURN_TRIALS_MAX


# ---------------------------------------------------------------------------
# 6 -- G10: one writer
# ---------------------------------------------------------------------------
_STATE = {"facing", "_facing_q", "facing_q", "facing_target_q", "omega_q"}


def test_facing_has_no_setter():
    """G10: assigning ``Unit.facing`` (or ``facing_q``) raises -- a missed
    writer fails at its own line. Breaks if a setter is added."""
    u = Unit("M", 5, 5)
    with pytest.raises(AttributeError):
        u.facing = 0.0
    with pytest.raises(AttributeError):
        u.facing_q = 0
    assert u.facing == u.facing_q / 65536.0
    assert u.facing_q == u.facing_target_q == turning.FACING_NORTH_Q16


def test_no_src_module_writes_the_facing_outside_turning():
    """G10: parse every module under ``src/``; no assignment (plain,
    augmented, annotated, tuple) to an attribute named like the facing's
    state, and no ``setattr`` naming one, outside
    ``simulation/turning.py``. Breaks if a new site writes a facing instead
    of handing the law a target."""
    offenders = []
    for path in sorted((ROOT / "src").rglob("*.py")):
        rel = path.relative_to(ROOT / "src").as_posix()
        if rel == "simulation/turning.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                targets = [node.target]
            elif (isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Name)
                  and node.func.id == "setattr" and len(node.args) >= 2
                  and isinstance(node.args[1], ast.Constant)
                  and node.args[1].value in _STATE):
                offenders.append(f"{rel}:{node.lineno} setattr")
            for t in targets:
                for sub in ast.walk(t):
                    if isinstance(sub, ast.Attribute) and sub.attr in _STATE:
                        offenders.append(f"{rel}:{sub.lineno} .{sub.attr}")
    assert not offenders, offenders


# ---------------------------------------------------------------------------
# 7 -- every reason to face somewhere is a target
# ---------------------------------------------------------------------------
def _arena(h=40, w=40):
    from level_loader import LevelData
    tm = np.ones((h, w), dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = 4
    return LevelData(name="turn_law", version="1", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


def _sim(**kw):
    sim = Simulation(_arena(**kw), seed=1, breach_physics=bp,
                     enable_recorder=False, ruleset=ContinuousRealtime())
    sim.set_paused(False)
    return sim


def test_the_attack_snap_is_a_target_turned_toward_from_the_next_tick():
    """Property 7: a recorded facing snap (``SimultaneousEffects.face``)
    changes the TARGET at resolve and never the facing; the next ticks turn
    the facing there through the law. Breaks if the snap writes the facing
    (an instant turn) or is lost."""
    sim = _sim()
    u = sim.get_unit(sim.add_unit(Unit("M", x=8, y=8, team=0)))
    f0 = u.facing_q
    fx = SimultaneousEffects()
    fx.face(u, 0)                                 # east
    assert u.facing_target_q == f0                # recorded, not applied
    fx.resolve()
    assert u.facing_target_q == 0 and u.facing_q == f0
    a, w = sim.unit_turn_accel_q16(u), sim.unit_turn_rate_q16(u)
    want = law_q16(f0, 0, 0, a, w)
    sim.step()
    assert (u.facing_q, u.omega_q) == want
    for _ in range(40):
        sim.step()
    assert u.facing_q == 0 and u.omega_q == 0


def test_a_target_persists_and_a_zero_aim_holds_the_last_one():
    """Property 7: after an aim turned the unit, a zero aim (the policy's
    "facing untouched" row) keeps turning toward / holding the last aimed
    angle, never back to an older target. Breaks if the aim is not stored
    as the target or a zero aim releases it."""
    sim = _sim()
    uid = sim.add_unit(Unit("M", x=8, y=8, team=0))
    u = sim.get_unit(uid)
    sim.set_aim(uid, 0, FP_ONE)                   # south (y down)
    sim.step()
    want = unit_fixed.atan2_q16(-FP_ONE, 0)
    assert u.facing_target_q == want
    sim.set_aim(uid, 0, 0)
    for _ in range(40):
        sim.step()
    assert u.facing_q == want and u.omega_q == 0


def test_a_zombie_turns_toward_its_walk_through_the_law():
    """Property 7: a walking zombie's facing never jumps: each tick it moves
    by exactly one step of the law toward its target, the target is the
    direction of its last step, and it gets there. Breaks if the zombie walk
    writes the facing (an instant turn) again."""
    sim = _sim()
    sim.add_unit(Unit("M", x=30, y=8, team=0))
    z = sim.get_unit(sim.add_unit(Unit("Z", x=8, y=30, team=ZOMBIE_TEAM)))
    z.zombie_activated = True
    a, w = sim.unit_turn_accel_q16(z), sim.unit_turn_rate_q16(z)
    steps = turned = 0
    reached = False
    for _ in range(400):
        before = (z.facing_q, z.omega_q, z.facing_target_q)
        x0, y0 = z.x, z.y
        sim.step()
        assert (z.facing_q, z.omega_q) == law_q16(*before[:2], before[2],
                                                  a, w)
        if (z.x, z.y) != (x0, y0):
            steps += 1
            assert z.facing_target_q == unit_fixed.atan2_q16(
                unit_fixed.quantize_scalar(-(z.y - y0)),
                unit_fixed.quantize_scalar(z.x - x0))
        if z.facing_q != before[0]:
            turned += 1
            reached |= z.facing_q == z.facing_target_q
    assert steps > 3 and turned > 1 and reached


def test_a_walker_holding_no_aim_turns_into_a_reversal_at_the_reduced_speed():
    """Property 8: a unit walking east with no aim faces east at full walk
    speed; reversed to the west it keeps moving at less than full speed while
    its facing turns (several ticks, through the law), and walks at full
    speed again once it faces west. Breaks if the no-aim shortcut
    (cos = ONE) returns, or the travel facing is written at once."""
    sim = _sim()
    uid = sim.add_unit(Unit("M", x=20, y=20, team=0))
    u = sim.get_unit(uid)

    def walk(dx_q, ticks):
        out = []
        for _ in range(ticks):
            x0 = u.x
            sim.set_move_dir(uid, dx_q, 0, 0)           # walk, throttle 0
            sim.step()
            out.append((abs(u.x - x0), u.facing_q))
        return out

    east = walk(FP_ONE, 30)
    full = east[-1][0]

    def is_full(d):                              # float positions: ULPs
        return d == pytest.approx(full, rel=1e-9)
    assert east[-1][1] == 0 and is_full(east[-2][0])  # facing east, steady
    assert east[0][0] < full and not is_full(east[0][0])  # it faced north
    west = walk(-FP_ONE, 40)
    slow = [i for i, (d, _) in enumerate(west) if not is_full(d)]
    assert len(slow) >= 5                        # several ticks of turning
    assert all(west[i][0] < full for i in slow)
    assert west[-1][1] == PI_Q16 and is_full(west[-1][0])
    assert all(is_full(d) for d, _ in west[slow[-1] + 1:])
    turning_ticks = [f for _, f in west[:slow[-1] + 1]]
    assert len(set(turning_ticks)) >= 5          # the facing moved gradually


# ---------------------------------------------------------------------------
# 9 -- G9: the bots' predictor is the shooting slot's facing
# ---------------------------------------------------------------------------
def test_g9_the_bots_prediction_is_the_facing_the_shot_leaves_along(
        monkeypatch):
    """G9: in ship_demo seed 1 (bots vs bots + the horde), every
    ``facing_after_turn_q16`` a bot reads in ``handle_frame`` (its trigger
    rule) equals the facing that unit has in the shooting slot of the same
    tick. Breaks if the turn moves after a producer of the facing, or the
    predictor is a copy of the law that drifts from slot 1a."""
    import control_bots
    import level_loader
    from run_match import run_match

    predicted = {}
    seen = {"n": 0}
    orig_after = simulation_mod.Simulation.facing_after_turn_q16

    def after(self, u, dx_q, dy_q):
        q = orig_after(self, u, dx_q, dy_q)
        predicted[(self.tick, u.id)] = q
        return q

    orig_shoot = simulation_mod.process_shooting

    def shoot(gmap, units, tick, *args, **kwargs):
        for (t, uid), q in list(predicted.items()):
            if t == tick:
                u = next(x for x in units if x.id == uid)
                assert u.facing_q == q, (tick, uid)
                seen["n"] += 1
                del predicted[(t, uid)]
        return orig_shoot(gmap, units, tick, *args, **kwargs)

    monkeypatch.setattr(simulation_mod.Simulation, "facing_after_turn_q16",
                        after)
    monkeypatch.setattr(simulation_mod, "process_shooting", shoot)
    assert control_bots.BotTeam._on_target    # the reader this gate covers
    level = level_loader.load("ship_demo", str(ROOT / "levels"))
    run_match(level)
    # Non-vacuity, not a count pin: the bots read the predictor dozens of
    # times in the match (44 at G9p, where the horde eats half the squad).
    assert seen["n"] > 20


# ---------------------------------------------------------------------------
# G3: at unlimited acceleration the pinned matches are re-pin 1's
# ---------------------------------------------------------------------------
#: ship_demo seed 1 and the box at G5p-a1's re-pin 1 (the representation and
#: the routing, unlimited acceleration): the values the pins held before the
#: shipped 30 rad/s^2 moved them (tests/test_zombie_control.py).
#: RE-MEASURED at #183 G5p-a2 re-pin 1 (2026-11-10, the unit impulse law built with every input at zero -- [impulse]
#: bullet_push_mult = recoil_mult = 0, no shove -- and the MOVE_DIR walk
#: on the Q16 grid (the step is the speed law's rational rounded ONCE to
#: Q16 tiles per tick, critique B M6)):
#: the same unlimited-acceleration run on the new tree. Outcome, ticks and
#: field_digest UNCHANGED; tick_digest and the unit stream move. SOLE CAUSE, verified on the same tree: with the pre-G5p-a2 float walk
#: step reinstated (a scratch monkeypatch of Simulation._step_move_dir)
#: this run gives EXACTLY the previous values -- slot 1a's integration,
#: slot 3's displacement loop, the effort member, the digest keys, the
#: events and the door change nothing while every input is off.
#: (was tick 6e408f1955d11e567974b29252a93ecb080cff000e6d38a7d9653476d095267a,
#: units c792732fc27000f1adf5d04c223c598022016368c9dbd81c79ffba31bac3e5b5;
#: the box is UNCHANGED: its idle marines never walk.)
#: RE-MEASURED at #183 G5p-b (2026-11-10, the wind and the blast onto the
#: unit impulse law: the dynamic pressure at each footprint is a force
#: q * drag_area through unit_impulse.add_wind; #140's cadence scale and
#: bespoke knockdown DELETED in the same commit).
#: Every impulse input off is now the wind's too (wind_push_mult = 0,
#: set by the gate), and #140's law is gone with it: the ship_demo run
#: ends win 1 at 241 (was 246; outcome and team UNCHANGED), all three
#: digests and the unit stream move; the box is UNCHANGED.
#: SOLE CAUSE, verified on the same build: the pre-G5p-b tree (0b57fc6e)
#: at unlimited acceleration with the push and recoil off AND #140's
#: wind law off (1e15 / 2e15 Pa) gives exactly these values -- the
#: law's removal is the whole move; nothing of the impulse law runs
#: here.
#: (was field 4a30c340a0982a80080a1d29f85cdc377bb579e9ff85c6f685699c86fe
#: 725440, tick 9932ea99d2b3149ed58be5ddf07c91972b1b531087e4f66a196c2698
#: 7a7cba2e, units 6d94773610af96c82da8ce023d69d496a97c8175d65d7c59b80219
#: fea77a1f46)
#: RE-MEASURED at #183 G9p STAGE 1 (2026-11-11, the horde field law,
#: residency P3d: the horde descends an integer BFS field, no A*; see
#: tests/test_zombie_control.py's pins for the law and its proof). This
#: gate's dial is untouched; the horde under it walks the field now, so
#: the match it pins is re-measured on this tree with the gate's own
#: settings -- the law is the whole move.
#: (was win 1 at 241; field
#: c198fe24ac6dd5b0201619a60476d4f81ea00f5c5a2527b7146716a300759744; tick
#: f6545050e5ba3f41c54f7d0a8ea2dbed9934dfa92606fd6ef2f6aa082f139002; units
#: 1d1573ec408624eccf25481d02bcec9ac62ad03262b58aa06ece5682fca740a1)
#: RE-MEASURED at #183 G9p STAGE 2 (2026-11-11, the horde behaviours
#: on the field: wander, hearing, fear of fire, door bashing and the
#: sprint, [zombie]; see tests/test_zombie_control.py's pins). This
#: gate's dial is untouched; the match it pins is re-measured on this
#: tree with the gate's own settings -- the behaviours are the whole
#: move (tests/test_horde_behaviours.py: with all of them off, the
#: stage-1 law exactly).
#: (was win 1 at 244; field
#: 87c4dff86973a2ebc2310ea0f6c235dac18826ca0675b6f6774357758838a191; tick
#: 0d8dff655ad15306c696896a2b6ac7be0ddc77b01789503d79eeadafa6abef71; units
#: db7c0e8376ea39397a0d2c079f5243d24c3ac86a25618ceea57ed807e08a4317)
_RE_PIN_1_SHIP_DEMO = dict(
    outcome="win", team=1, ticks=244,
    # #29 P3e re-pin (2026-11-12, the match path to integer): the shot
    # cone, the exposure / crit rolls and the FieldEdit noise are integer draws
    # (simulation.draws: new random values), and the blast falloff, the payload
    # deposits, the heat and poison rows, mitigation and the hp are integer
    # (re-rounded) -- the GAME moved by design, not only the hash.
    # docs/p3e_ab_2026-11-12.md: win rate, survivors, rounds per kill and the
    # blast outcomes within seed noise over 24-72 seeds.
    # (was 841869decf00a406da93a7e35a561d81dc2d581cc090309ea33f0b957e7625ba)
    # #29 P3b re-pin (2026-11-12, Philox at the match draw sites): every
    # draw is keyed Philox (seed; tick, index, salt, id) in place of one
    # numpy stream -- WHICH random numbers are drawn moved, nothing else.
    # SOLE CAUSE, verified on the same tree: tests/_p3b_sole_cause.py feeds
    # main's stream words through the keyed sites and this pin's old value
    # comes back bit for bit; A/B in docs/p3b_ab_2026-11-12.md.
    # (was 309590f66a864df062ccb7c9cdf92caba03dfbfe8bdad86f22e435ab30b2a5be)
    field_digest="d6ff90dbce559fa603805b9eaf8139bd"
                 "5e18b34c0ac39b251894c3686140d780",
    # #29 P3a re-pin (2026-11-12, the unit schema): the unit record is now
    # the schema's (simulation.unit_schema.digest_record, as name|int64 rows
    # under UNIT_SECT_V2): ap / n_orders and the derived tile / alive /
    # faction / offsets left it, the hp hashes as its Q16 count, and every
    # synced column hashes while it differs from its default (the kit, the
    # magazine, the grenades, the cool-downs, the throw / plant in progress,
    # the intents). The GAME is unchanged: tests/_p3a_sole_cause.py gives
    # equal per-tick records and identical field digests on main edc0d2b1 and
    # on this tree. Only the hash's definition moved.
    # (was df36873fbb14eda3ff329f89e89f20d735202b973638f6f3e4d29b724197a9bc)
    # #29 P3c re-pin (2026-11-12, positions to Q16): the unit record hashes a
    # unit's position as its Q16 ints x_q / y_q instead of round(x * 1e9) under
    # x / y. The GAME is unchanged: tests/_p3c_sole_cause.py gives equal per-tick
    # records (x_q := round(x * 65536), every position already on the grid) and
    # identical field digests on main 229f7097 and on this tree; outcome, ticks
    # and field_digest here are unmoved. Only the hash's definition moved.
    # (was ddda3bb5d992ff8071f36cbf4d7550e5525baada5dd12738ed6c7c7c3a6c6943)
    # #29 P3e re-pin (2026-11-12, the match path to integer): the shot
    # cone, the exposure / crit rolls and the FieldEdit noise are integer draws
    # (simulation.draws: new random values), and the blast falloff, the payload
    # deposits, the heat and poison rows, mitigation and the hp are integer
    # (re-rounded) -- the GAME moved by design, not only the hash.
    # docs/p3e_ab_2026-11-12.md: win rate, survivors, rounds per kill and the
    # blast outcomes within seed noise over 24-72 seeds.
    # (was cdb2152bc2d764dce97483c26f70152b9017c463083d53b7275529f447e8b9b4)
    # #29 P3b re-pin (2026-11-12, Philox at the match draw sites): every
    # draw is keyed Philox (seed; tick, index, salt, id) in place of one
    # numpy stream -- WHICH random numbers are drawn moved, nothing else.
    # SOLE CAUSE, verified on the same tree: tests/_p3b_sole_cause.py feeds
    # main's stream words through the keyed sites and this pin's old value
    # comes back bit for bit; A/B in docs/p3b_ab_2026-11-12.md.
    # (was 3d21436d8858106d5d9f26c57829d7b5b9e6de232d0bd2492e7a841c749d4bb8)
    # 183-no-fire-windup re-pin (2026-11-13, Erik's ruling: no firing while
    # winding up a throw). Outcome, team, ticks and field_digest UNCHANGED;
    # the tick_digest moves. SOLE CAUSE, verified on the same tree: with
    # verb_gates.can_fire forced True this test gives exactly the old
    # value. A bot holding its trigger on the tick its throw starts fired
    # on that tick (the bots pin: ticks 554 and 657, unit 5); now that
    # round does not leave, and the cadence moves after it.
    # (was 1775613f19c56ce58fb53975fa1ad64ce472ccb59073b404578eaff67c99ed93)
    tick_digest="bc211fbe91ab81cde55e1fcf4085a611"
                "af30c2cfdc34f1431392f4c049d6cba9",
    # #29 P3a re-pin (2026-11-12, the unit schema): the unit record is now
    # the schema's (simulation.unit_schema.digest_record, as name|int64 rows
    # under UNIT_SECT_V2): ap / n_orders and the derived tile / alive /
    # faction / offsets left it, the hp hashes as its Q16 count, and every
    # synced column hashes while it differs from its default (the kit, the
    # magazine, the grenades, the cool-downs, the throw / plant in progress,
    # the intents). The GAME is unchanged: tests/_p3a_sole_cause.py gives
    # equal per-tick records and identical field digests on main edc0d2b1 and
    # on this tree. Only the hash's definition moved.
    # (was 6480df664eb192d90ff0be9dc50e8f9ca9263a89f4b85c050b9a27a7183eefc9)
    # #29 P3c re-pin (2026-11-12, positions to Q16): the unit record hashes a
    # unit's position as its Q16 ints x_q / y_q instead of round(x * 1e9) under
    # x / y. The GAME is unchanged: tests/_p3c_sole_cause.py gives equal per-tick
    # records (x_q := round(x * 65536), every position already on the grid) and
    # identical field digests on main 229f7097 and on this tree; outcome, ticks
    # and field_digest here are unmoved. Only the hash's definition moved.
    # (was e8c7ef9db453387ff803d525b5577b3582fe008dd458bf14708612c429ab85d3)
    # #29 P3e re-pin (2026-11-12, the match path to integer): the shot
    # cone, the exposure / crit rolls and the FieldEdit noise are integer draws
    # (simulation.draws: new random values), and the blast falloff, the payload
    # deposits, the heat and poison rows, mitigation and the hp are integer
    # (re-rounded) -- the GAME moved by design, not only the hash.
    # docs/p3e_ab_2026-11-12.md: win rate, survivors, rounds per kill and the
    # blast outcomes within seed noise over 24-72 seeds.
    # (was da9febf7321dbe971ef9e5fa7c3ed599c8453f7cc969f18aad6251dc3c07d529)
    # #29 P3b re-pin (2026-11-12, Philox at the match draw sites): every
    # draw is keyed Philox (seed; tick, index, salt, id) in place of one
    # numpy stream -- WHICH random numbers are drawn moved, nothing else.
    # SOLE CAUSE, verified on the same tree: tests/_p3b_sole_cause.py feeds
    # main's stream words through the keyed sites and this pin's old value
    # comes back bit for bit; A/B in docs/p3b_ab_2026-11-12.md.
    # (was 7e4d0161cf924be4dfe170b52d7d2386b77c59c984a5a27bdea519d2431ffdcb)
    # 183-no-fire-windup re-pin (2026-11-13, Erik's ruling: no firing while
    # winding up a throw). Outcome, team, ticks and field_digest UNCHANGED;
    # the unit_stream moves. SOLE CAUSE, verified on the same tree: with
    # verb_gates.can_fire forced True this test gives exactly the old
    # value. A bot holding its trigger on the tick its throw starts fired
    # on that tick (the bots pin: ticks 554 and 657, unit 5); now that
    # round does not leave, and the cadence moves after it.
    # (was 27cefe3132431e8d06ca4838afe4b4d48219ea21403d53c2d300fdecfdf8643b)
    unit_stream="2d57334578451b54a9c0caa62c728f26"
                "0a91eed499dca0da5f2b0ed2cf0bf739")
#: RE-MEASURED at #183 G9p STAGE 1 (2026-11-11, the horde field law,
#: residency P3d: the horde descends an integer BFS field, no A*; see
#: tests/test_zombie_control.py's pins for the law and its proof). This
#: gate's dial is untouched; the horde under it walks the field now, so
#: the match it pins is re-measured on this tree with the gate's own
#: settings -- the law is the whole move.
#: (was win 1 at 130; field
#: a672c7249e18d0eebbb8e398b696bd9c3d34417445f66c3cab498ec7cd783ab1; units
#: a8f571251bbb5cc7572f7c4711dc1a3186b67dcdfd4c42837dacfcaeff1fefa0)
#: RE-MEASURED at #183 G9p STAGE 2 (2026-11-11, the horde behaviours
#: on the field: wander, hearing, fear of fire, door bashing and the
#: sprint, [zombie]; see tests/test_zombie_control.py's pins). This
#: gate's dial is untouched; the match it pins is re-measured on this
#: tree with the gate's own settings -- the behaviours are the whole
#: move (tests/test_horde_behaviours.py: with all of them off, the
#: stage-1 law exactly).
#: (was win 1 at 130; field
#: 0fdc79f228cc2e64ca6936cf5a2ec41c52642adfca545cbe51324cfad2e21631; units
#: 19d264664c2b3885c6aec68e6634eb2c79e450266bdb88cc6677aea5bda38248)
_RE_PIN_1_BOX = dict(
    outcome="win", team=1, ticks=92,
    field_digest="2a9339a53e67d28c29ff513f6942dab9"
                 "1586f5e85c6eb552885554c16ebe4c3b",
    # #29 P3a re-pin (2026-11-12, the unit schema): the unit record is now
    # the schema's (simulation.unit_schema.digest_record, as name|int64 rows
    # under UNIT_SECT_V2): ap / n_orders and the derived tile / alive /
    # faction / offsets left it, the hp hashes as its Q16 count, and every
    # synced column hashes while it differs from its default (the kit, the
    # magazine, the grenades, the cool-downs, the throw / plant in progress,
    # the intents). The GAME is unchanged: tests/_p3a_sole_cause.py gives
    # equal per-tick records and identical field digests on main edc0d2b1 and
    # on this tree. Only the hash's definition moved.
    # (was 035304b74d47d0dfcd73e661ebe4401121994f1a121abcc851c301e7a3c588c7)
    # #29 P3c re-pin (2026-11-12, positions to Q16): the unit record hashes a
    # unit's position as its Q16 ints x_q / y_q instead of round(x * 1e9) under
    # x / y. The GAME is unchanged: tests/_p3c_sole_cause.py gives equal per-tick
    # records (x_q := round(x * 65536), every position already on the grid) and
    # identical field digests on main 229f7097 and on this tree; outcome, ticks
    # and field_digest here are unmoved. Only the hash's definition moved.
    # (was 1e05debc38bf2ad1aebc1c8b5a75811019205bc09e78bc506b7fe0aab20f0f57)
    unit_stream="764ee188445aa1bf3e4f8357a44d2576"
                "edabd6d0f301b9ce6b87c435f2d3879d")


def test_g3_at_unlimited_acceleration_the_pins_are_re_pin_1s(
        monkeypatch, tmp_path):
    """G3 (permanent): with ``[marine] turn_accel_rad_per_s2`` at or above
    twice the top speed per tick (1000 rad/s^2 against 302) and every
    impulse input off (G5p-a2: the push and recoil multipliers at 0; G5p-b:
    the wind's), ship_demo
    seed 1 and the box play exactly the matches of re-pin 1 -- so the shipped
    acceleration is the ONLY cause of re-pin 2, and the law changes nothing
    its dials do not control. Breaks if the braking law leaks into the
    instant limit, or anything else moved with the acceleration."""
    import level_loader
    from run_match import run_match
    from test_zombie_control import _UnitStream, _box
    monkeypatch.setattr(CFG.marine, "turn_accel_rad_per_s2", 1000.0)
    # G5p-a2 (critique B's G3: "and every input off"): no push, no recoil;
    # G5p-b: and no wind.
    monkeypatch.setattr(CFG.impulse, "bullet_push_mult", 0.0)
    monkeypatch.setattr(CFG.impulse, "recoil_mult", 0.0)
    monkeypatch.setattr(CFG.impulse, "wind_push_mult", 0.0)
    w = turning.turn_rate_q16(CFG)
    assert turn_accel_q16(CFG) >= 2 * w
    stream = _UnitStream()
    from test_bots_v1 import v1_spec
    ship = level_loader.load("ship_demo", str(ROOT / "levels"))
    # G8p: the first bots (bots_v1) played this pin; the layered bots
    # moved the shipped match, so this gate plays it with them.
    o = run_match(ship, v1_spec(ship), on_tick=stream)
    assert dict(outcome=o.result.outcome, team=o.result.team,
                ticks=o.result.ticks, field_digest=o.field_digest,
                tick_digest=o.tick_digest,
                unit_stream=stream.h.hexdigest()) == _RE_PIN_1_SHIP_DEMO
    stream = _UnitStream()
    o = run_match(_box(tmp_path, "box"), on_tick=stream)
    assert dict(outcome=o.result.outcome, team=o.result.team,
                ticks=o.result.ticks, field_digest=o.field_digest,
                unit_stream=stream.h.hexdigest()) == _RE_PIN_1_BOX


# ---------------------------------------------------------------------------
# #195 1g -- ticks_to_within: the trigger aid is the law iterated, not a copy
# ---------------------------------------------------------------------------
def test_ticks_to_within_is_the_law_stepped_by_hand():
    """Property (#195 1g, obs spec 1.7): ``ticks_to_within`` is the count of
    ``law_q16`` steps until the facing lies within ``tol_q`` of the target
    -- 0 when already within, the hand-stepped count for a quarter turn
    (with and without an outside spin), and ``cap`` when the turn cannot
    land within the cap. Breaks if the aid stops reading the law (a copy, a
    closed form) or counts the landing tick off by one."""
    u = Unit("t", x=5, y=5, team=0, footprint=3)
    tol = unit_fixed.quantize_scalar(math.radians(3.0))
    quarter = turning.normalize_q16(unit_fixed.quantize_scalar(math.pi / 2))
    # already within: zero ticks, whatever the spin
    turning.place(u, 0)
    assert turning.ticks_to_within(u, tol // 2, A_SHIP, W_SHIP, 0, tol,
                                   48) == 0
    # a quarter turn from rest: step the law by hand to the same answer
    turning.place(u, 0)
    facing, omega, n = 0, 0, 0
    while abs(wrap_q16(quarter - facing)) > tol:
        facing, omega = law_q16(facing, omega, quarter, A_SHIP, W_SHIP)
        n += 1
    assert 1 < n < 11
    assert turning.ticks_to_within(u, quarter, A_SHIP, W_SHIP, 0, tol,
                                   48) == n
    # the same with an outside spin ``dw`` added first, as slot 1a adds it
    dw = -3 * A_SHIP
    facing, omega, m = 0, dw, 0
    while abs(wrap_q16(quarter - facing)) > tol:
        facing, omega = law_q16(facing, omega, quarter, A_SHIP, W_SHIP)
        m += 1
    assert m != n
    assert turning.ticks_to_within(u, quarter, A_SHIP, W_SHIP, dw, tol,
                                   48) == m
    # the cap: a half turn at a tiny top speed is not reached in 4 ticks
    assert turning.ticks_to_within(u, PI_Q16, A_SHIP, 100, 0, tol, 4) == 4
    assert u.facing_q == 0 and u.omega_q == 0      # a pure read
