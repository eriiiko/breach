"""#106 P6f -- ANALOG SPEED for creatures, and a human driving the octopus.

Erik's ruling of 2026-10-31: "I'd rather we assume we can control the speed
from 0 to its max speed, as if we had an Xbox controller; we have good
control over the actual speed. If we build this, we can always reduce
later." A MOVE_DIR carries a direction AND ``speed_q`` (Q16, 0 .. ONE of the
unit's maximum); the creature's HUB picks the gait from it
(``creature_primitives``' speed law over ``[creatures.octopus.locomotion]``).

Properties protected (each test names the change that must break it):

* THE SPEED LAW: 0 idles, up to ``crawl_until`` crawls, up to
  ``swim_until`` the umbrella, above the jet train (band tops inclusive);
  the medium factor is the identity at ONE; each band's TOP is the P6d2
  gait EXACTLY (p_recruit, the stroke ticks, the jet dv untouched), and
  each gait's tempo is monotone in the speed.
* SPEED IS MONOTONE: asked 0.2 / 0.5 / 0.8 / 1.0 of v_max, the octopus
  covers strictly more ground in 20 s, and the top speed is the row's
  declared ``v_max`` within a band; inside the crawl a slower ask crawls
  slower (the crawl's lever, the recruitment probability).
* HYSTERESIS: ramped through both boundaries (with a jitter that crosses
  them faster than the hold), the travel goal changes exactly once per
  boundary, each after ``gait_hold_ticks`` ticks of asking; a flicker
  across a boundary changes nothing; stopping is immediate.
* SPEED 0 IDLES (a MOVE_DIR with no speed is no travel goal).
* MARINES: a marine's movement is byte-identical whatever ``speed_q`` says
  (the marines' law reads ``speed_mode`` only).
* CONTROL: the gamepad's stick magnitude -> ``speed_q`` past the deadzone
  (pure); the keyboard map (pure); ``--play creature`` selects the creature
  and routes a MOVE_DIR to its hub (headless, through the control seam,
  renderer None), refused for a non-direct scheme; the ``creature``
  controller hunts at the row's ``hunt_speed``.
* DETERMINISM: a run whose speed ramps across all three gaits is
  byte-identical twice, record for record.

The 20 s runs take ~1 min together (CPU). The CPU == CUDA lockstep is
``tests/cuda_creature_check.py`` (it drives the crawl at its band top).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release", ROOT / "tests",
           ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

bp = pytest.importorskip("breach_physics")
if not hasattr(bp, "arm_chain_tick"):
    pytest.skip("breach_physics predates the arm chain solver (#106 P6c): rebuild cpp/",
                allow_module_level=True)

import _creature_scenarios as SC  # noqa: E402
from simulation import creature_fixed as CF  # noqa: E402
from simulation import creature_primitives as P  # noqa: E402
from simulation import creature_system as CS  # noqa: E402
from simulation.creatures import get_creature  # noqa: E402

ONE = 1 << 16
TPS = SC.TPS


def _dials():
    return P.PrimDials(get_creature("octopus"))


class _Body:
    """The hub fields the tempo functions read (pure tests, no solver)."""

    def __init__(self, speed):
        self.dials = _dials()
        self.goal_speed = int(speed)


# ---------------------------------------------------------------------------
# The speed law (pure)
# ---------------------------------------------------------------------------
def test_the_speed_law_is_the_goal_table():
    """0 -> none; (0, crawl_until] -> crawl; (crawl_until, swim_until] ->
    the umbrella; above -> the jet train; the medium factor at ONE is the
    identity and scales otherwise. Breaks if a band edge moves off its dial
    or the hook stops being exact at ONE."""
    d = _dials()
    c, w = d.crawl_until, d.swim_until
    assert 0 < c < w <= ONE
    table = [(0, P.GOAL_NONE), (1, P.GOAL_CRAWL), (c, P.GOAL_CRAWL),
             (c + 1, P.GOAL_CRUISE), (w, P.GOAL_CRUISE), (w + 1, P.GOAL_ESCAPE),
             (ONE, P.GOAL_ESCAPE)]
    for s, goal in table:
        assert P.speed_band(s, d) == goal, (s, P.speed_band(s, d), goal)
    for s in (0, 1, c, w, ONE, ONE + 5, -3):
        assert P.speed_law(s) == max(0, min(ONE, s))
    assert P.speed_law(ONE, ONE // 2) == ONE // 2
    assert P.speed_law(ONE, 2 * ONE) == ONE          # clamped to the maximum


def test_each_band_top_is_the_p6d2_gait_and_tempo_is_monotone():
    """At a band's top the gait's integers are P6d2's exactly (p_recruit;
    the umbrella's open / close ticks; the jet's dv), and below it the crawl
    recruits less, the umbrella strokes slower, the jet pushes less --
    monotonically. Breaks if a tempo factor of ONE perturbs a gait or the
    tempo stops ordering with the speed."""
    d = _dials()
    c, w = d.crawl_until, d.swim_until
    assert P.crawl_p_recruit(_Body(c)) == d.p_recruit
    assert P._stroke_ticks(_Body(w)) == (d.open_ticks, d.close_ticks)
    assert P._jet_scale(_Body(ONE), d.jet_dv_q24) == d.jet_dv_q24
    ps = [P.crawl_p_recruit(_Body(s)) for s in np.linspace(1, c, 40).astype(int)]
    assert all(a <= b for a, b in zip(ps, ps[1:])) and ps[0] < ps[-1]
    st = [sum(P._stroke_ticks(_Body(s))) for s in np.linspace(c + 1, w, 40).astype(int)]
    assert all(a >= b for a, b in zip(st, st[1:])) and st[0] > st[-1]
    dv = [P._jet_scale(_Body(s), d.jet_dv_q24) for s in np.linspace(w + 1, ONE, 40).astype(int)]
    assert all(a <= b for a, b in zip(dv, dv[1:])) and dv[0] < dv[-1]


# ---------------------------------------------------------------------------
# Speed through the sim
# ---------------------------------------------------------------------------
def _travel(speed_q, seconds, *, name, w=300, h=40, at=(12.0, 18.5), seed=7, size=None,
            deg=0.0):
    if size is not None:
        sc = SC.Scenario(name, size=size, seed=seed)
    else:
        sc = SC.Scenario(name, w=w, h=h, at=at, seed=seed)
    sc.seconds(seconds, each=lambda s: s.move(deg, int(speed_q)))
    r = sc.finish()
    return r, float(np.hypot(r.X[-1, 0] - r.X[0, 0], r.X[-1, 1] - r.X[0, 1]))


def test_asked_speed_orders_the_distance_travelled_over_20_s():
    """Asked 0.2 / 0.5 / 0.8 / 1.0 of v_max East from the deck (heading
    East), the octopus travels STRICTLY farther in 20 s at each step (by at
    least 5 %), and its top speed is the row's declared v_max within
    0.5x-1.5x. Breaks if a gait band inverts the order (the umbrella above
    the jet), the jet stops scaling with the speed, or v_max stops
    describing the animal. (P6f measured 0.47 / 2.13 / 3.27 / 4.01 m/s.)"""
    secs = 20.0
    dist = []
    for f in (0.2, 0.5, 0.8, 1.0):
        _, d = _travel(round(f * ONE), secs, name=f"mono_{int(f * 100)}")
        dist.append(d)
    print("P6f speeds (m/s):", [round(x / secs, 3) for x in dist])
    assert all(b > 1.05 * a for a, b in zip(dist, dist[1:])), dist
    v_max = get_creature("octopus").locomotion.v_max
    assert 0.5 * v_max <= dist[-1] / secs <= 1.5 * v_max, (dist[-1] / secs, v_max)


def test_a_slower_ask_crawls_slower():
    """Inside the crawl band, asked a quarter of crawl_until the octopus
    crawls toward the goal at most 0.7x the distance it crawls at the band
    top over 20 s -- and still forward. Breaks if the crawl's speed lever
    (the recruitment probability x tempo^2) stops reaching the gait. (P6f
    measured 0.19 vs 0.43 m/s, seed 1; the recruitment x tempo alone kept
    0.37.)"""
    c = _dials().crawl_until
    slow, d_slow = _travel(c // 4, 20.0, name="crawl_q", size=110, seed=1)
    full, d_full = _travel(c, 20.0, name="crawl_f", size=110, seed=1)
    print(f"P6f crawl: tempo 0.25 {d_slow / 20:.3f} m/s, tempo 1 {d_full / 20:.3f} m/s")
    assert slow.records[-1]["goal_kind"] == full.records[-1]["goal_kind"] == P.GOAL_CRAWL
    assert slow.X[-1, 0] - slow.X[0, 0] > 0.5
    assert d_slow <= 0.7 * d_full, (d_slow, d_full)


def test_the_gait_switches_once_per_boundary_after_the_hold():
    """A speed ramped 0.05 -> 1.0 over 6 s with a +-0.03 jitter flipping
    every tick (so it straddles each boundary for a while): the travel goal
    runs crawl -> cruise -> escape, each switch EXACTLY once, on the tick
    the new band has been asked ``gait_hold`` ticks in a row; a speed
    flickering across crawl_until every tick never switches; dropping to 0
    idles on that tick. Breaks if the hysteresis is lost (flicker), stops
    being an elapsed counter, or delays a stop."""
    sc = SC.Scenario("hyst", size=40)
    b = sc.body
    d = b.dials
    n = 6 * TPS
    asks, goals = [], []
    for k in range(n):
        base = 0.05 + 0.95 * k / (n - 1)
        s = int(round(min(1.0, max(0.0, base + (0.03 if k % 2 else -0.03))) * ONE))
        asks.append(P.speed_band(s, d))
        sc.step(1, each=lambda q, s=s: q.move(0.0, s))
        goals.append(b.goal_kind)
    runs = [g for i, g in enumerate(goals) if i == 0 or g != goals[i - 1]]
    assert runs == [P.GOAL_CRAWL, P.GOAL_CRUISE, P.GOAL_ESCAPE], runs
    for i in range(1, n):
        if goals[i] != goals[i - 1]:
            # asked the new band on this tick and the hold - 1 before it
            assert all(a == goals[i] for a in asks[i - d.gait_hold + 1:i + 1]), (i, asks[i - 8:i + 1])
            assert asks[i - d.gait_hold] != goals[i]
    print(f"P6f hysteresis: switches at ticks "
          f"{[i for i in range(1, n) if goals[i] != goals[i - 1]]}, hold {d.gait_hold} ticks")
    # flicker across crawl_until: never leaves the crawl
    fl = SC.Scenario("flicker", size=40)
    c = fl.body.dials.crawl_until
    for k in range(48):
        s = c + 64 if k % 2 else c          # starts in the crawl
        fl.step(1, each=lambda q, s=s: q.move(0.0, s))
        assert fl.body.goal_kind == P.GOAL_CRAWL, k
    # a stop is immediate, from any travel goal
    sc.step(1, each=lambda q: q.move(0.0, 0))
    assert b.goal_kind == P.GOAL_NONE and b.gait == P.GAIT_IDLE


def test_speed_zero_idles():
    """A MOVE_DIR at speed 0 is no travel goal: the hub idles (seated) and
    the crown stays put over 3 s. Breaks if a zero speed starts a gait."""
    sc = SC.Scenario("zero", size=40)
    x0 = sc.body.bufs.body[0, 0, :2].copy()
    sc.seconds(3.0, each=lambda q: q.move(0.0, 0))
    assert sc.body.goal_kind == P.GOAL_NONE and sc.body.gait == P.GAIT_IDLE
    moved = np.hypot(*(sc.body.bufs.body[0, 0, :2] - x0).astype(float)) / CF.CHAIN_ONE
    assert moved < 0.05, moved


def test_a_ramp_across_every_gait_is_byte_identical_twice():
    """Two runs ramping the speed 0 -> 1 -> 0 over 4 s: every tick's
    ``__creature__`` record is the same bytes. Breaks on a float or any
    hidden state in the speed law."""
    def go():
        sc = SC.Scenario("det_ramp", size=60, seed=5)
        for k in range(4 * TPS):
            f = 1.0 - abs(1.0 - 2.0 * k / (4 * TPS - 1))
            sc.step(1, each=lambda q, f=f: q.move(30.0, int(round(f * ONE))))
        return sc.run.raw
    a, b = go(), go()
    assert a == b
    assert a[0] != a[-1]


# ---------------------------------------------------------------------------
# Marines: the analog speed is not their law
# ---------------------------------------------------------------------------
def test_a_marines_movement_ignores_speed_q(tmp_path):
    """A marine driven by MOVE_DIR for 2 s: the per-tick digest is the same
    with ``speed_q`` absent (ONE), 0.3 and 0 -- the marines' law reads only
    ``speed_mode``. Breaks if the marines' continuous move reads the analog
    speed (which would move every golden)."""
    from field_ab_harness import capture_snapshot
    from field_digest import tick_digest
    from simulation import Simulation
    from simulation.ruleset import ContinuousRealtime
    from simulation.unit import Unit
    level = SC.write_room(tmp_path, "marine", w=40, h=30)

    def go(speed):
        sim = Simulation(level, seed=3, breach_physics=bp, enable_recorder=False,
                         ruleset=ContinuousRealtime())
        u = Unit("m", 10, 12, team=0)
        sim.add_unit(u)
        out = []
        for _ in range(48):
            if speed is None:
                sim.set_move_dir(u.id, 46341, -46341, 0)
            else:
                sim.set_move_dir(u.id, 46341, -46341, 0, speed_q=speed)
            sim.set_paused(False)
            sim.step()
            out.append(tick_digest(capture_snapshot(sim)))
        return out, (u.x, u.y)
    base, pos = go(None)
    assert pos != (10, 12), "the marine did not move"
    for sp in (ONE, round(0.3 * ONE), 0):
        assert go(sp)[0] == base, sp


# ---------------------------------------------------------------------------
# Control: the stick, the keys, --play creature, the controller
# ---------------------------------------------------------------------------
def test_the_stick_magnitude_past_the_deadzone_is_the_speed():
    """analog_speed_q: 0 inside the deadzone, rising from ~0 just past it to
    ONE at full deflection (a corner reading > 1 clamped), monotone in the
    magnitude and the same in every direction. Breaks if the deadzone is
    not rescaled out or the clamp is lost."""
    from control_source import analog_speed_q
    dz = 0.2
    assert analog_speed_q(0.0, 0.0, dz) == 0
    assert analog_speed_q(0.19, 0.0, dz) == 0
    assert 0 < analog_speed_q(0.21, 0.0, dz) < ONE // 50
    assert analog_speed_q(1.0, 0.0, dz) == ONE
    assert analog_speed_q(0.9, 0.9, dz) == ONE
    assert analog_speed_q(0.6, 0.0, dz) == analog_speed_q(0.0, -0.6, dz) == round(0.5 * ONE)
    mags = np.linspace(0.0, 1.2, 61)
    sp = [analog_speed_q(float(m), 0.0, dz) for m in mags]
    assert all(a <= b for a, b in zip(sp, sp[1:]))


def test_the_keyboard_map():
    """keyboard_move: WASD -> a unit vector (screen: W = -y), opposite keys
    cancel (None), full speed by default, Shift -> slow_speed, Ctrl ->
    mid_speed, Shift wins. Breaks if a binding or a speed moves."""
    from control_keyboard import keyboard_move, keyboard_settings
    slow, mid = keyboard_settings()
    assert 0.0 <= slow <= mid <= 1.0
    assert keyboard_move(False, False, False, False, False, False, slow, mid) is None
    assert keyboard_move(True, True, False, False, False, False, slow, mid) is None
    dx, dy, s = keyboard_move(True, False, False, False, False, False, slow, mid)
    assert (dx, dy, s) == (0, -ONE, ONE)
    dx, dy, s = keyboard_move(True, False, False, True, True, False, slow, mid)
    assert dx > 0 and dy < 0 and abs(dx + dy) <= 1 and s == round(slow * ONE)
    assert keyboard_move(False, True, False, False, False, True, slow, mid)[2] == round(mid * ONE)
    assert keyboard_move(False, True, False, False, True, True, slow, mid)[2] == round(slow * ONE)


def _tank_sim():
    """levels/octopus_tank as main.py builds it for a direct scheme: the
    squad from [[spawn]], the octopus from [[creature]]."""
    import level_loader
    from simulation import Simulation
    from simulation.ruleset import ContinuousRealtime
    from simulation.unit import Unit
    level = level_loader.load("octopus_tank", str(ROOT / "levels"))
    sim = Simulation(level, seed=42, breach_physics=bp, enable_recorder=False,
                     ruleset=ContinuousRealtime())
    for s in level.spawns:
        sim.add_unit(Unit(s.name, x=s.x, y=s.y, team=s.team, footprint=s.footprint))
    (octo,) = sim.spawn_creatures(level.creatures)
    return sim, octo


def test_play_creature_drives_the_octopus_through_the_control_seam():
    """--play creature + the keyboard scheme, headless (renderer None): the
    driven unit is the octopus, W + Shift sends it a MOVE_DIR North at the
    slow speed, and after the hold its hub crawls North while the squad has
    no intent and stands; W alone asks the jet train. A WEGO scheme and an
    unknown name are refused. Breaks if --play stops selecting the creature,
    the scheme stops routing to its hub, or the squad gets a driver."""
    import control_source
    from control_keyboard import KeyboardDirect
    from simulation.factions import CREATURE_TEAM
    held = set()
    kd = KeyboardDirect(key_down=lambda k: k in held, key_pressed=lambda k: False,
                        mouse_down=lambda b: False)
    sim, octo = _tank_sim()
    control_source.select_play(sim, kd, "creature")
    assert sim.player_team == CREATURE_TEAM
    body = sim._creature_system.bodies[octo.id]
    y0 = int(body.bufs.body[0, 0, 1])
    held.update({"W", "LEFT_SHIFT"})
    for _ in range(3 * TPS):
        kd.handle_frame(sim, None)
        sim.set_paused(False)
        sim.step()
    assert kd.selected_unit_id == octo.id
    mv = octo.live_move_dir
    assert (mv.dx_q, mv.dy_q) == (0, -ONE) and mv.speed_q == round(kd.slow_speed * ONE)
    assert body.goal_kind == P.GOAL_CRAWL and body.gait == P.GAIT_CRAWL
    assert (y0 - int(body.bufs.body[0, 0, 1])) / CF.CHAIN_ONE > 0.2   # North = -y
    for u in sim.units:
        if u is not octo:
            assert getattr(u, "live_move_dir", None) is None
    held.discard("LEFT_SHIFT")
    for _ in range(body.dials.gait_hold):
        kd.handle_frame(sim, None)
        sim.set_paused(False)
        sim.step()
    assert octo.live_move_dir.speed_q == ONE and body.goal_kind == P.GOAL_ESCAPE
    held.clear()
    kd.handle_frame(sim, None)
    sim.set_paused(False)
    sim.step()
    assert octo.live_move_dir is None and body.goal_kind == P.GOAL_NONE
    # refusals: a WEGO scheme cannot play a creature; an unknown name
    sim2, _ = _tank_sim()
    with pytest.raises(SystemExit, match="direct control scheme"):
        control_source.select_play(sim2, control_source.IdleControl(), "creature")
    with pytest.raises(SystemExit, match="--play must be one of"):
        control_source.select_play(sim2, kd, "zombie")
    control_source.select_play(sim2, control_source.IdleControl(), None)   # absent: no-op
    assert sim2.player_team == 0


def test_the_creature_controller_hunts_at_the_rows_hunt_speed(tmp_path):
    """The ``creature`` controller's MOVE_DIR carries the row's
    ``locomotion.hunt_speed`` (the shipped row: crawl_until, so it crawls,
    as P6d2's controller did), and a row with another hunt_speed sends that
    one. Breaks if the controller hard-codes its speed."""
    import control_source
    from control_creature import hunt_speed_q16
    from simulation import Simulation
    from simulation.factions import CREATURE_TEAM
    from simulation.match import CreatureSpawn
    from simulation.ruleset import ContinuousRealtime
    from simulation.unit import Unit
    row = get_creature("octopus")
    assert row.locomotion.hunt_speed == row.locomotion.crawl_until
    for hunt, name in ((None, "h1"), (0.5, "h2")):
        level = SC.write_room(tmp_path, name, w=60, h=30,
                              creatures=(CreatureSpawn("octopus", 12.0, 13.5),))
        ctx = SC.row_override(locomotion={"hunt_speed": hunt}) if hunt is not None else None
        if ctx is not None:
            ctx.__enter__()
        try:
            sim = Simulation(level, seed=7, breach_physics=bp, enable_recorder=False,
                             ruleset=ContinuousRealtime())
            (octo,) = sim.spawn_creatures(level.creatures)
            sim.add_unit(Unit("m", 48, 14, team=0))
            ctl = control_source.create_controller("creature", CREATURE_TEAM)
            ctl.handle_frame(sim, None)
            want = CF.q16_from_scalar(hunt if hunt is not None else row.locomotion.hunt_speed)
            assert octo.live_move_dir.speed_q == want == hunt_speed_q16(octo)
        finally:
            if ctx is not None:
                ctx.__exit__(None, None, None)


def test_the_door_refuses_a_bad_locomotion_row():
    """The [locomotion] group goes through the one door: bands out of order
    (crawl_until >= swim_until), a fraction above 1, a missing key are each
    refused by name; gait_hold_s is quantized to ticks (>= 1). Breaks if a
    speed dial can slip past the door."""
    import copy
    import tomllib
    from config import CFG
    from simulation.creatures import build_creature
    with open(ROOT / "config.toml", "rb") as f:
        raw = tomllib.load(f)["creatures"]["octopus"]
    tps = int(CFG.clock.ticks_per_second)
    assert build_creature("octopus", raw, tps).locomotion.gait_hold_ticks >= 1
    bad = copy.deepcopy(raw)
    bad["locomotion"]["crawl_until"] = bad["locomotion"]["swim_until"]
    with pytest.raises(ValueError, match="locomotion"):
        build_creature("octopus", bad, tps)
    bad = copy.deepcopy(raw)
    bad["locomotion"]["swim_until"] = 1.5
    with pytest.raises(ValueError, match="locomotion.swim_until"):
        build_creature("octopus", bad, tps)
    bad = copy.deepcopy(raw)
    del bad["locomotion"]["hunt_speed"]
    with pytest.raises(ValueError, match="locomotion.hunt_speed: missing"):
        build_creature("octopus", bad, tps)
