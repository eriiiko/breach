"""#106 P7 -- THE chain accessor (``creature_system.read_chain`` /
``Simulation.creature_chains``): the one read of a creature's chain for the
renderer, tools and rules. Float32 views dequantized once, never written
back; the render read changes no synced byte.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_chain_accessor.py -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

bp = pytest.importorskip("breach_physics")
if not hasattr(bp, "arm_chain_tick"):
    pytest.skip("this engine build has no arm chain solver", allow_module_level=True)

import _creature_scenarios as SC  # noqa: E402
from field_ab_harness import capture_snapshot  # noqa: E402
from field_digest import tick_digest  # noqa: E402
from simulation import arm_chain as AC  # noqa: E402
from simulation import creature_fixed as CF  # noqa: E402
from simulation import creature_system as CS  # noqa: E402


@pytest.fixture(scope="module")
def crawled():
    """An octopus 1.5 s into a crawl toward 70 deg, at heading 30 deg."""
    sc = SC.Scenario("acc", size=60, heading=30.0, seed=4)
    sc.seconds(1.5, each=lambda s: s.move(70.0))
    return sc


def test_the_view_is_the_buffers_at_float32(crawled):
    """PROPERTY: every view field is its integer buffer dequantized once at
    float32 (nodes, velocities, crown, its velocity, the yaw rate), the
    hub's per-arm state, the anchors, last_hit and the hit tiles the hit
    seam tests; each arm's current length is its polyline's.

    BREAKS IF: the view reads a stale copy, applies a second scaling, or a
    field is taken from somewhere other than the synced buffers."""
    body, unit = crawled.body, crawled.unit
    v = CS.read_chain(body, unit)
    b = body.bufs
    assert v.nodes.dtype == np.float32 and v.nodes.shape == b.p.shape[1:]
    assert np.array_equal(v.nodes, CF.chain_to_f32(b.p[0]))
    assert np.array_equal(v.velocities, CF.chain_to_f32(b.v[0]))
    bd = b.body[0, 0]
    assert np.array_equal(v.crown, CF.chain_to_f32(bd[[AC.BS_X, AC.BS_Y, AC.BS_Z]]))
    assert np.array_equal(v.crown_velocity, CF.chain_to_f32(bd[[AC.BS_VX, AC.BS_VY, AC.BS_VZ]]))
    assert v.yaw_rate == -float(bd[AC.BS_OM]) / CF.CHAIN_ONE
    assert np.array_equal(v.arm_state, body.arm_state)
    assert np.array_equal(v.anchored, b.anchored[0].astype(bool))
    assert v.last_hit == tuple(body.last_hit)
    assert v.hit_tiles == unit.hit_shape.tiles
    seg = np.diff(CF.chain_to_m(b.p[0]), axis=1)
    assert np.allclose(v.length, np.linalg.norm(seg, axis=-1).sum(1), rtol=1e-6)
    assert np.all(v.rest_length == np.float32(body._cdef.body.arm_length_m))
    assert (v.unit_id, v.creature, v.species_id, v.alive) == (unit.id, "octopus",
                                                             unit.species_id, True)
    assert v.footprint == unit.footprint and v.tile_size_m == body.tile_size_m


def test_the_heading_is_the_facing_convention(crawled):
    """PROPERTY: the view's heading is the pose heading through
    ``heading_rad_to_facing`` -- equal to the unit's ``facing`` the sim wrote
    -- and it MEANS the facing convention: on the rest pose, arm a points
    ``arm_angles_deg[a]`` counter-clockwise on screen from it (screen y is
    -grid y).

    BREAKS IF: the view hands out the solver's yaw (the opposite sign), or a
    convention change lands anywhere but ``creature_fixed``."""
    body, unit = crawled.body, crawled.unit
    v = CS.read_chain(body, unit)
    assert v.heading == CF.heading_rad_to_facing(CF.heading_q24_to_rad(body.heading_q24))
    assert v.heading == unit.facing
    sc = SC.Scenario("acc_rest", size=60, heading=40.0)
    r = CS.read_chain(sc.body, sc.unit)
    assert abs(r.heading - math.radians(40.0)) < 1e-6
    for a, deg in enumerate(sc.body._cdef.body.arm_angles_deg):
        d = r.nodes[a, -1, :2].astype(float) - r.nodes[a, 0, :2]
        screen = math.atan2(-d[1], d[0])
        want = r.heading + math.radians(deg)
        assert abs((screen - want + math.pi) % (2 * math.pi) - math.pi) < 1e-3, a


def test_reading_is_pure_and_the_digest_ignores_the_renderer():
    """PROPERTY: two identical sims, one READ every tick through the
    accessor and posed by the renderer's pose module, the other never read,
    stay byte-identical: the field digest and the ``__creature__`` bytes at
    every tick (the renderer attached or detached cannot move a digest).

    BREAKS IF: the accessor or the pose writes into a buffer (a view that
    aliases ``bufs.p`` and an in-place lift, say), or reading advances any
    sim state."""
    from renderer import creature_pose as CP
    from config import CFG
    rig = CP.build_rig(SC.Scenario("tmp_rig", size=40).body._cdef)
    dials = CP.creature_render_settings(CFG)
    a = SC.Scenario("pure_a", size=60, seed=9)
    b = SC.Scenario("pure_b", size=60, seed=9)
    for t in range(36):
        for sc in (a, b):
            sc.move(120.0)
            sc.sim.set_paused(False)
            sc.sim.step()
        views = a.sim.creature_chains()
        for v in views.values():
            CP.pose_from_chain(v, rig, dials, t * 0.04)
        sa, sb = capture_snapshot(a.sim), capture_snapshot(b.sim)
        assert tick_digest(sa) == tick_digest(sb), t
        assert a.body.bufs.p.tobytes() == b.body.bufs.p.tobytes()
        assert CS.serialize_record(a.body) == CS.serialize_record(b.body)


def test_dead_creatures_are_read_frozen():
    """PROPERTY: ``Simulation.creature_chains`` includes a dead creature
    (alive False), and its chain does not move once dead (slot 5b skips
    it) -- the renderer draws it frozen at its last pose.

    BREAKS IF: the facade drops dead creatures (the body would vanish), or
    a dead body keeps being integrated."""
    sc = SC.Scenario("dead", size=60)
    sc.seconds(0.5, each=lambda s: s.move(0.0))
    sc.unit.alive = False
    v0 = sc.sim.creature_chains()[sc.unit.id]
    sc.step(10)
    v1 = sc.sim.creature_chains()[sc.unit.id]
    assert v0.alive is False and v1.alive is False
    assert np.array_equal(v0.nodes, v1.nodes)


def test_octopus_clips_draw_through_the_accessor():
    """PROPERTY: ``tools/octopus_clips.clip_frames`` -- re-pointed at the
    accessor -- draws the same arms, crown, prop and arm colours as the
    decoded ``__creature__`` record of every tick (float32 precision).

    BREAKS IF: the clip tool reads another source than the accessor, or the
    accessor's frame differs from the record's (an origin, an axis, a sign)."""
    sys.path.insert(0, str(ROOT / "tools"))
    import octopus_clips as OC
    from simulation import creature_primitives as P
    sc = SC.Scenario("clips", size=110, seed=3)
    sc.body.add_prop()
    sc.seconds(1.0, each=lambda s: s.move(0.0))
    sc.use()
    sc.seconds(1.5, each=lambda s: s.move(0.0))
    run = sc.finish()
    frames = OC.clip_frames(run, every=3)
    one = float(CF.CHAIN_ONE)
    for i, f in enumerate(frames):
        rec = run.records[3 * i]
        arms = rec["p"].astype(float) / one
        arms[..., 1] *= -1
        assert np.allclose(f["arms"], arms, atol=2e-6, rtol=0)
        crown = rec["body"][:3].astype(float) / one
        crown[1] *= -1
        assert np.allclose(f["crown"], crown, atol=2e-6, rtol=0)
        prop = rec["bodies"][0][:3].astype(float) / one
        prop[1] *= -1
        assert np.allclose(f["prop"], prop, atol=2e-6, rtol=0)
        assert f["colours"] == [OC.STATE_COLOURS[P.ARM_STATE_NAMES[int(s)]]
                                for s in rec["arm_state"]]
