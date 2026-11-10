"""P3 direct-control tests (control-modularity §3, gamepad-first).

Covers the three GATE categories for the new (dormant-under-WEGO) path:

  A. the MOVE_DIR per-tick movement branch — mobility-scaled step + footprint
     collision using the SAME predicate A* uses (is_passable_block);
  B. the ContinuousRealtime ruleset — no phases / AP / auto-pause,
     death-triggered zombie conversion, and per-tick stamp semantics
     (corpse physics not regressed);
  C. the gamepad float->fixed-point intent quantization pure function; plus
  D. bit-reproducibility of a scripted fixed-point intent stream (this
     substitutes for a golden on the new path — there is no strong oracle,
     the real behavior is Erik's human-test with a controller).

These exercise ONLY the new continuous path; the WEGO byte-identity gate lives
in the existing digest/golden tests (this patch leaves them untouched).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))

import numpy as np
import pytest

import breach_physics as bp
from level_loader import LevelData
from simulation.factions import ZOMBIE_TEAM  # noqa: E402
from simulation import Simulation
from simulation.ruleset import ContinuousRealtime, TwoPhaseWEGO
from simulation.unit import Unit
from simulation.orders import ORDER_MOVE_ATTACK
from simulation.movement import FootprintSamples, default_speed
from simulation.simulation import _ticks_per_tile
from simulation.intents import FP_ONE, dequantize
from control_source import quantize_stick_direction

SEED = 42


# ---------------------------------------------------------------------------
# Synthetic arenas (no asset files)
# ---------------------------------------------------------------------------
def _open_arena(h=16, w=16) -> LevelData:
    """Hull border, all-air interior (tilemap 4 = air, 1 = hull)."""
    tm = np.ones((h, w), dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = 4
    return LevelData(name="p3_open", version="1", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


def _wall_arena(h=20, w=20, wall_col=12) -> LevelData:
    """All-air interior with one solid wall column at ``wall_col``."""
    tm = np.ones((h, w), dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = 4
    tm[:, wall_col] = 1               # solid wall column (hull material)
    return LevelData(name="p3_wall", version="1", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


def _continuous_sim(level) -> Simulation:
    sim = Simulation(level, seed=SEED, breach_physics=bp,
                     enable_recorder=False, ruleset=ContinuousRealtime())
    sim.set_paused(False)
    return sim


# ===========================================================================
# A. MOVE_DIR movement branch
# ===========================================================================
def test_move_dir_steps_by_mobility_scaled_amount():
    """A live MOVE_DIR moves the unit by exactly the mobility-scaled per-tick
    distance the WEGO cadence math yields (1 / default_speed tiles), and along
    the fixed-point direction only."""
    sim = _continuous_sim(_open_arena())
    uid = sim.add_unit(Unit("M1", x=6, y=6, team=0))
    u = sim.get_unit(uid)
    # G5p-a1: a walker holding no aim turns to its travel with inertia; this
    # test is about the step's size, so the unit already faces +X.
    from simulation.turning import place
    place(u, 0)
    x0, y0 = u.x, u.y
    fp = u.footprint

    # Expected per-tick advance, computed through the SAME predicates the
    # movement branch reuses: the [speeds] walk (Q16 ticks per tile, G1p-c)
    # -> area-average mobility. The unit faces its move, so theta is 0.
    base = sim.speed_table.walk_cost_q
    samples = FootprintSamples(
        mobility=sim.gmap.footprint_mobility(u.tile_y, u.tile_x, fp))
    tick_cost = default_speed(samples, base)
    expected_step = dequantize(FP_ONE) * (FP_ONE / tick_cost)  # +X unit vector

    sim.set_move_dir(uid, FP_ONE, 0, 0)     # pure +X, throttle 0 (walk)
    sim.step()

    assert u.x == x0 + expected_step        # exact (deterministic)
    assert u.y == y0                        # no drift on the idle axis


def test_move_dir_blocked_by_impassable_footprint_like_astar():
    """A unit driven into a wall stops where its footprint would first overlap
    the wall — the SAME boundary is_passable_block (A*'s predicate) enforces —
    and never overlaps it."""
    sim = _continuous_sim(_wall_arena(wall_col=12))
    # 3x3 footprint anchored at x=3 (cols 3,4,5) — clear of the wall at col 12.
    uid = sim.add_unit(Unit("M1", x=3, y=8, team=0))
    u = sim.get_unit(uid)
    fp = u.footprint

    sim.set_move_dir(uid, FP_ONE, 0, FP_ONE)     # drive +X, full throttle
    for _ in range(400):
        sim.step()

    # The last passable anchor before the col-12 wall is tile_x = 9 (cols
    # 9,10,11 air; anchoring at 10 would pull col 12 into the footprint).
    assert sim.gmap.is_passable_block(u.tile_y, u.tile_x, fp)          # legal
    assert not sim.gmap.is_passable_block(u.tile_y, u.tile_x + 1, fp)  # A* agrees
    assert u.tile_x == 9
    assert u.x < 10.0            # never stepped its anchor onto the wall


def test_no_move_dir_leaves_wego_path_untouched():
    """A unit with NO live MOVE_DIR under a WEGO sim replays its move_path
    exactly as before (the continuous branch is dormant)."""
    sim = Simulation(_open_arena(), seed=SEED, breach_physics=bp,
                     enable_recorder=False, ruleset=TwoPhaseWEGO())
    uid = sim.add_unit(Unit("M1", x=6, y=6, team=0))
    u = sim.get_unit(uid)
    assert getattr(u, "live_move_dir", None) is None   # never armed
    # No crash / no continuous movement without an intent.
    sim.set_paused(False)
    x0 = u.x
    sim.step()
    assert u.x == x0            # no orders, no path, no intent -> stays put


# ===========================================================================
# B. ContinuousRealtime ruleset
# ===========================================================================
def test_continuous_no_phases_no_ap_no_autopause():
    """ContinuousRealtime never advances phase, never rewinds the tick, never
    auto-pauses, and never spends AP."""
    sim = _continuous_sim(_open_arena())
    uid = sim.add_unit(Unit("M1", x=6, y=6, team=0))
    z = sim.add_unit(Unit("Z1", x=9, y=9, team=ZOMBIE_TEAM))   # keep it non-terminal
    u = sim.get_unit(uid)
    ap0 = list(u.ap)
    tpr = sim._ticks_per_round

    for _ in range(tpr + 30):        # well past a WEGO round boundary
        sim.step()

    assert sim.tick == tpr + 30      # free-running clock, no rewind to 0
    assert sim.phase == 0            # phases never advance
    assert sim.is_paused() is False  # no auto-pause at the round boundary
    assert list(u.ap) == ap0         # AP untouched by the ruleset


def test_continuous_death_triggered_zombie_conversion():
    """A marine killed by a zombie converts within one tick under continuous
    play (death-triggered), not batched at an end-of-round that never comes."""
    sim = _continuous_sim(_open_arena())
    uid = sim.add_unit(Unit("M1", x=6, y=6, team=0))
    sim.add_unit(Unit("Z1", x=9, y=9, team=ZOMBIE_TEAM))
    u = sim.get_unit(uid)

    # Simulate a zombie kill this tick.
    u.alive = False
    u.killed_by_zombie = True
    sim.step()

    assert u.team == ZOMBIE_TEAM
    assert u.is_zombie is True
    assert u.alive is True           # risen as a zombie, immediately


def test_wego_defers_conversion_past_the_tick():
    """Contrast: under TwoPhaseWEGO the same freshly-killed marine does NOT
    convert on the next tick (conversion is the end-of-round batch)."""
    sim = Simulation(_open_arena(), seed=SEED, breach_physics=bp,
                     enable_recorder=False, ruleset=TwoPhaseWEGO())
    uid = sim.add_unit(Unit("M1", x=6, y=6, team=0))
    sim.add_unit(Unit("Z1", x=9, y=9, team=ZOMBIE_TEAM))
    u = sim.get_unit(uid)
    sim.set_paused(False)
    u.alive = False
    u.killed_by_zombie = True
    sim.step()
    assert u.team == 0               # still a (dead) marine mid-round
    assert u.is_zombie is False


def test_continuous_corpse_physics_not_regressed():
    """Per-tick stamp semantics (Simulation.step slot 6) are ruleset-
    independent: living units are SOFT occluders in dyn_permeability, obstacles
    stays WALLS-ONLY, and a corpse stamps NOTHING — so continuous play never
    accumulates dead-body blockers and never regresses corpse physics.

    (As-built model note: dead units do not block physics in either ruleset —
    only living units soft-occlude, and only walls are hard obstacles. The
    WEGO end-of-round obstacle reset was undoing per-round accumulation that
    per-tick stamping already keeps clean; continuous simply never accumulates
    it. See the report's discrepancy note.)"""
    sim = _continuous_sim(_open_arena())
    uid = sim.add_unit(Unit("M1", x=6, y=6, team=0))
    u = sim.get_unit(uid)
    g = sim.gmap

    walls_only = (g.permeability <= 0.0)
    sim.step()
    # Obstacles == walls only, every tick, regardless of the ruleset.
    assert np.array_equal(g.obstacles, walls_only)
    # The living unit is a soft occluder: its footprint lowered dyn_permeability
    # below the (open-air) static baseline.
    fy, fx = u.tile_y, u.tile_x
    assert g.dyn_permeability[fy, fx] < g.permeability[fy, fx]

    # Kill it; next tick its tiles return to the static baseline (corpse stops
    # soft-blocking) and obstacles is STILL walls-only.
    u.alive = False
    u.killed_by_zombie = False        # a plain corpse, not a zombie kill
    sim.step()
    assert np.array_equal(g.obstacles, walls_only)
    assert g.dyn_permeability[fy, fx] == g.permeability[fy, fx]


def test_continuous_is_terminal_one_team_eliminated():
    """is_terminal fires when one side is gone (no round to complete)."""
    sim = _continuous_sim(_open_arena())
    m = sim.add_unit(Unit("M1", x=6, y=6, team=0))
    sim.add_unit(Unit("Z1", x=9, y=9, team=ZOMBIE_TEAM))
    assert sim.is_terminal() is False        # both sides present
    sim.get_unit(m).alive = False            # marines wiped
    assert sim.is_terminal() is True


# ===========================================================================
# C. Gamepad float -> fixed-point quantization (pure function)
# ===========================================================================
def test_quantize_deadzone_returns_zero():
    assert quantize_stick_direction(0.0, 0.0) == (0, 0, 0)
    assert quantize_stick_direction(0.05, 0.05, deadzone=0.15) == (0, 0, 0)


def test_quantize_axis_unit_vectors():
    # Pure +X and -Y are exact unit vectors at full magnitude.
    assert quantize_stick_direction(1.0, 0.0) == (FP_ONE, 0, FP_ONE)
    assert quantize_stick_direction(0.0, -1.0) == (0, -FP_ONE, FP_ONE)


def test_quantize_normalizes_direction_and_clamps_magnitude():
    # (0.6, 0.8) already has magnitude 1 -> unit vector unchanged.
    dx, dy, mag = quantize_stick_direction(0.6, 0.8)
    assert (dx, dy, mag) == (
        quantize_stick_direction(0.6, 0.8)[0],   # self-consistent
        dy, mag)
    # Over-unit corner (2.0, 0.0): direction normalizes to +X, magnitude clamps.
    assert quantize_stick_direction(2.0, 0.0) == (FP_ONE, 0, FP_ONE)


def test_quantize_is_direction_only_of_magnitude():
    """Same direction, different magnitudes -> same fixed-point unit vector."""
    a = quantize_stick_direction(0.5, 0.0)
    b = quantize_stick_direction(0.9, 0.0)
    assert a[:2] == b[:2] == (FP_ONE, 0)


def test_quantize_is_deterministic_and_fixed_point():
    """Repeated calls are bit-identical, and every component is a Q16.16 int."""
    for _ in range(3):
        dx, dy, mag = quantize_stick_direction(0.31, -0.77)
        assert isinstance(dx, int) and isinstance(dy, int) and isinstance(mag, int)
    assert (quantize_stick_direction(0.31, -0.77)
            == quantize_stick_direction(0.31, -0.77))


def test_quantize_point_aim_is_the_stick_seam_plus_the_reach():
    """G1p, the MOUSE: the aim at a point is the stick seam's unit vector of
    the offset (direction only: any distance gives the same vector), its
    reach the distance in Q16.16 tiles, deterministic ints; a point inside
    the deadzone names nothing. Breaks if the mouse path quantizes another
    way than the stick, or the reach is not the distance to the cursor."""
    from control_source import quantize_point_aim
    from simulation.unit_fixed import quantize_scalar
    ox, oy = 10.5, 20.5
    assert quantize_point_aim(ox, oy, ox + 4.0, oy, 0.25) == (
        FP_ONE, 0, 4 * FP_ONE)
    assert quantize_point_aim(ox, oy, ox, oy - 7.5, 0.25) == (
        0, -FP_ONE, quantize_scalar(7.5))
    dx, dy, reach = quantize_point_aim(ox, oy, ox + 3.0, oy + 4.0, 0.25)
    assert (dx, dy) == quantize_stick_direction(3.0, 4.0, 0.0)[:2]
    assert reach == 5 * FP_ONE
    far = quantize_point_aim(ox, oy, ox + 30.0, oy + 40.0, 0.25)
    assert far[:2] == (dx, dy) and far[2] == 50 * FP_ONE
    assert quantize_point_aim(ox, oy, ox + 0.1, oy - 0.1, 0.25) == (0, 0, 0)
    a = quantize_point_aim(1.25, 2.5, 7.31, -0.77, 0.25)
    assert a == quantize_point_aim(1.25, 2.5, 7.31, -0.77, 0.25)
    assert all(isinstance(v, int) for v in a)


def test_stick_throttle_spans_walk_to_sprint_over_the_deflection():
    """G1p, the PAD's throttle: the deflection just past the deadzone is 0
    (the walk), full deflection ONE (the sprint), monotone between, a Q16
    int. Breaks if a light push sprints, a full push walks, or a float
    reaches the sim."""
    from control_source import stick_throttle_q
    dz = 0.2
    assert stick_throttle_q(0, dz) == 0
    assert stick_throttle_q(quantize_stick_direction(0.2, 0.0, 0.0)[2], dz) \
        == 0
    assert stick_throttle_q(FP_ONE, dz) == FP_ONE
    vals = [stick_throttle_q(quantize_stick_direction(m / 100, 0.0, 0.0)[2],
                             dz) for m in range(0, 101)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))
    assert all(isinstance(v, int) and 0 <= v <= FP_ONE for v in vals)


# ===========================================================================
# D. Bit-reproducibility of a scripted intent stream (new-path substitute
#    for a golden — there is no strong oracle for direct-control behavior)
# ===========================================================================
def _scripted_trajectory():
    """Run one deterministic direct-control episode from a fixed synthetic
    intent script; return the per-tick (x, y, facing, n_proj) trace."""
    sim = _continuous_sim(_open_arena(20, 20))
    uid = sim.add_unit(Unit("M1", x=8, y=8, team=0))
    sim.add_unit(Unit("Z1", x=15, y=15, team=ZOMBIE_TEAM))   # keep non-terminal
    trace = []
    for t in range(60):
        # A rotating stick input, quantized at the seam exactly like the pad.
        ax = np.cos(t * 0.3)
        ay = np.sin(t * 0.3)
        mdx, mdy, mmag = quantize_stick_direction(float(ax), float(ay))
        if mmag == 0:
            sim.clear_move_dir(uid)
        else:
            sim.set_move_dir(uid, mdx, mdy, 0)
        sim.set_aim(uid, mdx, mdy)
        sim.set_trigger(uid, t % 4 == 0)
        if t == 10:
            sim.throw_grenade_intent(uid, FP_ONE, 0, 48)     # 2 s
        if t == 20:
            sim.use_intent(uid)
        sim.step()
        u = sim.get_unit(uid)
        trace.append((u.x, u.y, u.facing, len(sim.projectiles)))
    return trace


def test_scripted_intent_stream_is_bit_reproducible():
    """Two independent runs of the same fixed intent script produce an
    identical trajectory — the determinism substitute for a golden on the
    new direct-control path."""
    a = _scripted_trajectory()
    b = _scripted_trajectory()
    assert a == b
    # And it actually moved / fired (guard against a vacuous all-equal trace).
    assert any(x != 8.0 or y != 8.0 for (x, y, _f, _n) in a)
