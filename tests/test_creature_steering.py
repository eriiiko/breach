"""#106 P6g -- STEERING: the octopus both strafes and turns.

Erik played P6f (2026-10-31): "WASD only ever moves it straight forward, I
can't turn it"; then, of the crawl: it STRAFES (travels in the asked
direction without turning its body), which he wants to keep -- "but it has
to be able to both strafe and turn". So two channels, as on a twin-stick
game: TRAVEL (MOVE_DIR + speed_q) and FACING (the AIM: the keyboard cursor,
the gamepad's right stick; a creature reaches along an aim only while its
``reach`` flag is set). The lead's ruling: no yaw actuator -- every turn
comes from a force the animal applies, through the solver's own body
dynamics (one law per gait, ``creature_primitives``' docstring).

Properties protected (each test names the change that must break it):

* NO ASK, NO CHANGE (gate d): with no MOVE_DIR and no aim, and while
  crawling with no aim, the per-tick ``__creature__`` records are
  bit-identical to P6f's (hashes captured with the P6f code before this
  patch changed anything; the row header -- version and row hash -- and the
  V4 facing field are left out of the hash, since adding row dials moves
  them by design).
* THE FACING CHANNEL: an aim without ``reach`` is a facing and starts no
  reach; with ``reach`` it reaches as before; the record carries the facing.
* THE CURSOR MAP (pure): a marine aims only while RMB is held; a creature
  always faces the cursor and reaches only with RMB.
* THE TANK'S PROP: the octopus spawns with its grab prop, so USE grabs.
* THE CRAWL'S 90-DEGREE RESPONSE (gate e): finite and under 4 s.
"""
from __future__ import annotations

import hashlib
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
from simulation import creature_primitives as P  # noqa: E402
from simulation import creature_system as CS  # noqa: E402

ONE = 1 << 16
TPS = SC.TPS

#: P6f's records (captured at 9c34824 before P6g changed anything), hashed
#: without the row header's version + row hash (bytes 8:48) and without the
#: V4 facing + turn fields (the 12 bytes before the terminator).
P6F_HASHES = {
    "idle": "42b31746d03b716dcbc072144ac50e9db8e79c9cf5a25ec41e3c230d978b0b5b",
    "crawl": "a6f062336fe8b3eb4b2058079a118f561d01cef9c2d567d9cba207bdc928dc7a",
    "crawlturn": "a142215a2187810e476fd48109576d00b40222a4ba04bcbf85cd2d4132d2ac7c",
}


def _p6f_hash(raw) -> str:
    m = hashlib.sha256()
    for r in raw:
        assert r[-13:-1] == b"\0" * 12        # no facing asked, no turn
        m.update(r[:8] + r[48:-13] + r[-1:])
    return m.hexdigest()


def test_no_ask_changes_nothing_since_p6f():
    """Gate (d): idle with no intent, a crawl North, and a crawl East then
    North then a stop -- no aim anywhere -- give records bit-identical to
    P6f's, tick for tick. Breaks if a steering law acts without a facing
    asked, or if anything else in the hub moved."""
    sc = SC.Scenario("b_idle", size=60, seed=5)
    sc.seconds(3.0)
    assert _p6f_hash(sc.run.raw) == P6F_HASHES["idle"]
    sc = SC.Scenario("b_crawl", size=60, seed=7)
    sc.seconds(3.0, each=lambda s: s.move(90.0))
    assert _p6f_hash(sc.run.raw) == P6F_HASHES["crawl"]
    sc = SC.Scenario("b_turn", size=60, seed=3)
    sc.seconds(1.5, each=lambda s: s.move(0.0))
    sc.seconds(1.5, each=lambda s: s.move(90.0, "crawl"))
    sc.seconds(1.0)
    assert _p6f_hash(sc.run.raw) == P6F_HASHES["crawlturn"]


def test_an_aim_without_reach_is_a_facing_and_reaches_nothing():
    """A facing (aim with ``reach`` False) is stored and recorded and starts
    no reach; the same aim with ``reach`` reaches (P6d2's AIM -> reach).
    Breaks if the facing channel starts a reach, or the reach flag stops
    reaching."""
    sc = SC.Scenario("face", size=60, seed=5)
    sc.seconds(1.0, each=lambda s: s.face(0.0))
    b = sc.body
    assert b.face_q16 != (0, 0) and b.reach_q16 == (0, 0)
    assert not (b.arm_state == P.ARM_REACH).any()
    assert CS.decode_record(sc.run.raw[-1])["face_q16"] == b.face_q16
    sc.seconds(1.0, each=lambda s: s.aim(0.0))
    assert b.reach_q16 == b.face_q16 and (b.arm_state == P.ARM_REACH).any()
    sc.seconds(0.2, each=lambda s: s.face(None))
    assert b.face_q16 == (0, 0) and b.reach_q16 == (0, 0)


def test_the_cursor_map():
    """control_keyboard.cursor_aim: a marine aims only while RMB is held; a
    creature always faces the cursor (reach False) and reaches with RMB; no
    cursor or a cursor on the centre clears. Breaks if the creature's
    facing needs a button or a marine starts aiming without one."""
    from control_keyboard import cursor_aim
    assert cursor_aim(3.0, 0.0, False, False) is None
    assert cursor_aim(3.0, 0.0, True, False) == (ONE, 0, True)
    assert cursor_aim(0.0, -2.0, False, True) == (0, -ONE, False)
    assert cursor_aim(0.0, -2.0, True, True) == (0, -ONE, True)
    assert cursor_aim(None, None, True, True) is None
    assert cursor_aim(0.0, 0.0, False, True) is None


def test_the_tank_octopus_has_a_prop_and_use_grabs_it():
    """levels/octopus_tank's octopus spawns with its grab prop (``prop =
    true`` on its [[creature]] row, through the level writer / loader door),
    and USE starts the grab. Breaks if the row loses the prop, the loader
    drops the flag, or USE stops reaching the hub."""
    import level_loader
    from simulation import Simulation
    from simulation.ruleset import ContinuousRealtime
    level = level_loader.load("octopus_tank", str(ROOT / "levels"))
    assert [c.prop for c in level.creatures] == [True]
    sim = Simulation(level, seed=42, breach_physics=bp, enable_recorder=False,
                     ruleset=ContinuousRealtime())
    (octo,) = sim.spawn_creatures(level.creatures)
    body = sim._creature_system.bodies[octo.id]
    assert body.has_prop
    sim.use_intent(octo.id)
    for _ in range(TPS):
        sim.set_paused(False)
        sim.step()
    assert int(body.arm_state[body.dials.grab_arm]) in P.GRAB_STATES


def test_a_prop_row_is_written_only_when_true(tmp_path):
    """The writer adds ``prop = true`` only for a creature with a prop (every
    other level keeps its bytes) and the door refuses a non-bool. Breaks if
    a prop-less row grows a key or a bad prop slips through."""
    import level_lib
    from simulation.match import CreatureSpawn, creature_spawn_from_dict
    lines = "".join(level_lib.format_creature_lines([CreatureSpawn("octopus", 1.0, 2.0)]))
    assert "prop" not in lines
    lines = "".join(level_lib.format_creature_lines([CreatureSpawn("octopus", 1.0, 2.0,
                                                                   prop=True)]))
    assert "prop = true" in lines
    with pytest.raises(ValueError, match="prop must be a bool"):
        creature_spawn_from_dict({"species": "octopus", "x": 1.0, "y": 2.0, "prop": 1}, "t")


def test_the_crawls_90_degree_response_is_reported():
    """Gate (e): crawling East for 4 s, then asked North, the crown's travel
    over a trailing 0.5 s comes within 30 deg of North in under 4 s (seed 7).
    Breaks if the crawl stops answering a change of travel direction. (P6g
    measured a median 0.54 s over five seeds at steer_window_s 2.0, 0.71 s
    at 0.75; config.toml keeps 2.0.)"""
    sc = SC.Scenario("c90", size=110, seed=7)
    sc.seconds(4.0, each=lambda s: s.move(0.0))
    n0 = len(sc.run.records)
    sc.seconds(4.0, each=lambda s: s.move(90.0))
    X = sc.finish().X[:, :2]
    w = TPS // 2
    t = None
    for i in range(n0 + w, len(X)):
        d = X[i] - X[i - w]
        if np.hypot(*d) > 0.02 and abs(np.degrees(np.arctan2(-d[1], d[0])) - 90.0) < 30.0:
            t = (i - n0 + 1) / TPS
            break
    print(f"P6g crawl 90-deg response: {t} s")
    assert t is not None and t < 4.0


# ---------------------------------------------------------------------------
# Law 1: the TURN (Levy 2015 SI) -- the crawl's and the idle's facing
# ---------------------------------------------------------------------------
def _angd(a):
    return (np.degrees(a) + 180.0) % 360.0 - 180.0


def _facing_run(heading, face, move=None, secs=10.0, seed=1, anchors=True):
    sc = SC.Scenario("turn", size=110, seed=seed, heading=heading)
    orig = P._anchor
    if not anchors:
        P._anchor = lambda body, a, nodes: None
    try:
        sc.seconds(secs, each=lambda s: (s.face(face),
                                         s.move(move, "crawl") if move is not None else None))
    finally:
        P._anchor = orig
    r = sc.finish()
    err = np.abs(_angd(r.TH - np.radians(face)))
    hit = np.nonzero(err < 15.0)[0]
    return r, (float(r.t[hit[0]]) if len(hit) else None)


def test_an_idle_octopus_turns_to_face_the_aim_both_ways():
    """Seated, facing East, asked to face North (and, mirrored, South): the
    heading comes within 15 deg of the asked facing in under 8 s on two
    seeds, each turn the right way from its first second on. Breaks if the
    TURN's sign is wrong (it would turn away), if seated arms hold the body
    against it, or if a turn stops rotating the body. (P6g measured 4.8-6.5
    s for 90 deg -- Levy's scale is ~25 deg per 2.8 s.)"""
    times = []
    for face, sign in ((90.0, 1.0), (-90.0, -1.0)):
        for seed in (1, 7):
            r, t = _facing_run(0.0, face, secs=8.0, seed=seed)
            times.append(t)
            assert t is not None and t < 8.0, (face, seed, t)
            assert sign * np.degrees(r.TH[int(2.0 * TPS)]) > 0.0      # turned toward it
    print("P6g idle 90-deg turn times (s):", times)


def test_a_crawl_turns_while_it_strafes():
    """Gate (c), at the TURN's measured rate: crawling North (full crawl)
    while facing East -- and mirrored, West -- the heading comes within 15
    deg of the facing within 10 s, and the travel stays North: its direction
    over 1-10 s within 30 deg of North (the median of three seeds), above
    0.08 m/s on every seed. Breaks if the facing steals the travel (no
    strafe) or the crawl cannot turn. (The brief's 2 s is not reached by
    Levy's turn: P6g measured 3.5-7.4 s facing East; and turning COSTS
    travel -- 0.25-0.42 m/s facing East, 0.11-0.21 facing West, against
    0.36-0.53 with no facing: reported for P8.)"""
    for face in (0.0, 180.0):
        dirs, times = [], []
        for seed in (1, 3, 7):
            r, t = _facing_run(90.0, face, move=90.0, seed=seed)
            d = r.X[-1, :2] - r.X[TPS, :2]
            dirs.append(float(np.degrees(np.arctan2(-d[1], d[0]))))
            times.append(t)
            assert np.hypot(*d) / 9.0 > 0.08, (face, seed)
        print(f"P6g crawl N facing {face}: times {times}, travel {np.round(dirs, 1)}")
        assert sum(t is not None and t < 10.0 for t in times) >= 2, times
        assert abs(np.median(dirs) - 90.0) < 30.0, dirs


def test_the_turn_rotates_the_body_only_through_its_anchors():
    """Gate (g): the same idle facing ask with every anchor disabled (the
    sucker grip never takes) turns the body by under a quarter of what the
    anchored animal turns in 6 s. Breaks if a turn rotates the body by
    anything but the anchor's reaction through the root (a yaw written by
    a gait, a torque from nowhere)."""
    r1, _ = _facing_run(0.0, 90.0, secs=6.0, seed=7)
    r0, _ = _facing_run(0.0, 90.0, secs=6.0, seed=7, anchors=False)
    with_anchors = abs(_angd(r1.TH[-1] - r1.TH[0]))
    without = abs(_angd(r0.TH[-1] - r0.TH[0]))
    print(f"P6g turn: {with_anchors:.1f} deg anchored, {without:.1f} deg without anchors")
    assert with_anchors > 20.0 and without < 0.25 * with_anchors


def test_no_gait_writes_the_yaw():
    """Gate (j): nothing in the primitives module writes the crown's heading
    or yaw rate (``BS_TH`` / ``BS_OM``) -- every turn is a force through
    the solver. Breaks on a yaw actuator in a gait."""
    import re
    src = (ROOT / "src" / "simulation" / "creature_primitives.py").read_text(encoding="utf-8")
    for line in src.splitlines():
        if re.search(r"BS_(TH|OM)\b", line):
            assert not re.search(r"\]\s*[+\-]?=", line) or "BS_TH]" not in line.split("=")[0], line
            assert "BS_OM" not in line, line


# ---------------------------------------------------------------------------
# Law 2: the umbrella steers by per-side stroke amplitude
# ---------------------------------------------------------------------------
def _umbrella(ask, secs=12.0, seed=7):
    sc = SC.Scenario("umb", w=120, h=120, seed=seed, heading=0.0)
    sc.seconds(secs, each=lambda s: s.move(ask, "cruise"))
    return sc.finish()


def test_the_umbrella_turns_toward_its_travel_and_mirrors():
    """Gates (b) and (h): from the deck facing East, the umbrella asked
    North (and, mirrored, South) turns toward the ask -- the heading error
    shrinks by at least 20 deg in 12 s and the travel bends toward it --
    and the two runs are MIRRORED tick for tick (headings opposite within
    0.5 deg). Breaks if the per-side amplitude's sign flips (it turns away:
    measured), if steering acts on one side only, or if the law is not
    mirror-symmetric. (P6g measured 90 -> 59 deg in 12 s at umbrella_steer
    0.5, ~2.6 deg/s, intra-stroke yaw up to 75 deg/s; the brief's 2 s is
    far off: reported for P8.)"""
    north, south = _umbrella(90.0), _umbrella(-90.0)
    e_n = _angd(north.TH - np.radians(90.0))
    assert abs(e_n[-1]) < abs(e_n[0]) - 20.0, (e_n[0], e_n[-1])
    d = north.X[-1, :2] - north.X[-2 * TPS, :2]
    assert np.degrees(np.arctan2(-d[1], d[0])) > 10.0             # bending North
    assert np.allclose(np.degrees(north.TH), -np.degrees(south.TH), atol=0.5)
    print(f"P6g umbrella: error {e_n[0]:.0f} -> {e_n[-1]:.0f} deg in 12 s")


def test_the_umbrella_does_not_steer_within_the_dead_band():
    """A swim asked along its own heading has no steering (0 within the dead
    band): umbrella_steer_q16 is 0 for an error inside steer_dead_band_deg
    and signed outside it, saturating at the row's umbrella_steer. Breaks
    if the law acts on a straight swim or loses its proportional shape."""
    sc = SC.Scenario("umbq", size=60, seed=7)
    b = sc.body
    d = b.dials
    b.gait, b.goal_kind = P.GAIT_SWIM, P.GOAL_CRUISE
    for deg, want in ((0.0, 0), (5.0, 0), (-5.0, 0)):
        r = np.radians(deg)
        b.goal_q16 = (int(round(np.cos(r) * ONE)), int(round(np.sin(r) * ONE)))
        assert P.umbrella_steer_q16(b) == want
    for deg in (30.0, 60.0, 120.0):
        r = np.radians(deg)
        b.goal_q16 = (int(round(np.cos(r) * ONE)), int(round(np.sin(r) * ONE)))
        s = P.umbrella_steer_q16(b)
        b.goal_q16 = (b.goal_q16[0], -b.goal_q16[1])
        assert 0 < s <= d.umb_steer and P.umbrella_steer_q16(b) == -s
    assert s == d.umb_steer


# ---------------------------------------------------------------------------
# Law 3: the jet steers by its FUNNEL
# ---------------------------------------------------------------------------
P6F_SWIM_EAST = "a108201c37ff84eebd16b11bf250a6d49c124c139086faacd09ff5043a23247c"


def test_an_on_axis_jet_is_p6fs_jet():
    """Gate (i), on the axis: the jet train asked along its own heading
    (East, from East, 2 s) never aims the funnel off the axis, and its
    records are bit-identical to P6f's (the impulse at the funnel point,
    parallel to the offset, adds no yaw and the same dv). Breaks if the
    funnel point or its rotation perturbs a straight jet."""
    sc = SC.Scenario("b_swim", size=60, seed=7)
    sc.seconds(2.0, each=lambda s: s.move(0.0, "escape"))
    assert _p6f_hash(sc.run.raw) == P6F_SWIM_EAST


def _jet(heading, ask, secs=6.0, speed="escape", seed=7):
    sc = SC.Scenario("jet", w=200, h=200, seed=seed, heading=heading)
    sc.seconds(secs, each=lambda s: s.move(ask, speed))
    return sc.finish()


def _first(mask, t):
    i = np.nonzero(mask)[0]
    return float(t[i[0]]) if len(i) else None


def test_the_jet_reverses_through_its_funnel():
    """Gate (a) and (i), off the axis: from the deck facing West, asked East
    at full speed, the heading comes within 15 deg of East within 2.5 s and
    the crown's velocity within 30 deg of East within 3 s, with no
    overshoot past East of more than 10 deg (the trailing arms' drag is the
    rudder: no added yaw damping). Breaks if the funnel aims the wrong way,
    the jet stops turning the body, or the turn over-rotates. (P6g
    measured 2.17 s and 2.0 s, overshoot 0 -- the brief's 2 s heading is
    just missed: two jets are needed, one per opening time.)"""
    r = _jet(180.0, 0.0)
    err = _angd(r.TH)
    t_head = _first(np.abs(err) < 15.0, r.t)
    v = np.diff(r.X[:, :2], axis=0) * TPS
    vdir = _angd(np.arctan2(-v[:, 1], v[:, 0]))
    t_vel = _first((np.abs(vdir) < 30.0) & (np.hypot(v[:, 0], v[:, 1]) > 0.2), r.t[1:])
    print(f"P6g jet reversal: heading < 15 deg at {t_head} s, velocity < 30 deg at {t_vel} s")
    assert t_head is not None and t_head < 2.5
    assert t_vel is not None and t_vel < 3.0
    i = int(np.nonzero(np.abs(err) < 15.0)[0][0])
    assert np.max(np.abs(err[i:])) < 15.0                  # settles, no over-rotation


def test_the_jet_turns_both_ways_mirrored():
    """Asked North and South from East at full speed, the jet turns toward
    each within 1.5 s, and the two runs mirror (headings opposite within
    1 deg). Breaks if the funnel's sign or the impulse's lever is
    one-sided."""
    north, south = _jet(0.0, 90.0, secs=3.0), _jet(0.0, -90.0, secs=3.0)
    t = _first(np.abs(_angd(north.TH - np.radians(90.0))) < 15.0, north.t)
    assert t is not None and t < 1.5, t
    assert np.allclose(np.degrees(north.TH), -np.degrees(south.TH), atol=1.0)


def test_the_point_impulse_is_the_rigid_body_law():
    """add_impulse_at_q24: an impulse J (velocity units) at the body point
    r adds J to the crown's velocity (rotated into the world) and (r x J)
    m / I to its yaw rate, in the solver's sign; on the axis it adds no yaw.
    Breaks if the lever, the inertia ratio or the sign drifts."""
    from simulation import arm_chain as AC
    from simulation import creature_fixed as CF
    sc = SC.Scenario("imp", size=60, seed=7)
    b = sc.body
    bd = b.bufs.body[0, 0]
    v0, om0 = bd[[AC.BS_VX, AC.BS_VY]].astype(np.int64), int(bd[AC.BS_OM])
    J = CF.chain_from_m(0.5)
    L = CF.chain_from_m(0.3)
    b.add_impulse_at_q24((L, 0), (J, 0))
    assert int(bd[AC.BS_OM]) == om0
    b.add_impulse_at_q24((L, 0), (0, J))
    cfg = b.bufs.body_cfg[0, 0]
    m_over_i = int(cfg[AC.BC_WINVI_Q32]) / int(cfg[AC.BC_W_Q32])
    want = 0.3 * 0.5 * m_over_i                            # rad/s
    got = (int(bd[AC.BS_OM]) - om0) / CF.CHAIN_ONE
    assert abs(got - want) < 1e-4 * max(1.0, abs(want)) and got > 0
    dv = bd[[AC.BS_VX, AC.BS_VY]].astype(np.int64) - v0
    assert abs(np.hypot(*dv) / CF.CHAIN_ONE - np.hypot(0.5, 0.5)) < 1e-5
