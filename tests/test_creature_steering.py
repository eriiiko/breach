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
#: V4 facing field (the 8 bytes before the terminator).
P6F_HASHES = {
    "idle": "42b31746d03b716dcbc072144ac50e9db8e79c9cf5a25ec41e3c230d978b0b5b",
    "crawl": "a6f062336fe8b3eb4b2058079a118f561d01cef9c2d567d9cba207bdc928dc7a",
    "crawlturn": "a142215a2187810e476fd48109576d00b40222a4ba04bcbf85cd2d4132d2ac7c",
}


def _p6f_hash(raw) -> str:
    m = hashlib.sha256()
    for r in raw:
        assert r[-9:-1] == b"\0" * 8          # no facing asked
        m.update(r[:8] + r[48:-9] + r[-1:])
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
