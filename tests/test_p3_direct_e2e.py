"""Free-aim directional shooting — headless END-TO-END (F2/F3).

Drives a real :class:`Simulation` under :class:`ContinuousRealtime` through the
same ``set_aim`` / ``set_trigger`` facade ``GamepadDirect`` uses (no controller),
and asserts the free-aim firing model (free_aim_shooting_design_2026-07-23).

  F2 §2.1 REGRESSION — a possessed marine with a 90-range weapon, facing an
       enemy ~4 tiles ahead whose *nominal max-range endpoint tile is occluded
       by a wall beyond it*, HITS the enemy. The deleted band-aid
       (``_aim_fire_order``) fired at that far endpoint tile and its range+LOS
       pre-gate FAILED on the wall -> silent no-fire. The march bypasses the
       pre-gate and stops on the first thing the ray crosses.
  F2 bit-reproducibility — a scripted aim+trigger+move stream, run twice,
       produces an identical synced (x, y, hp, facing, alive) digest.

  F3 (added below in the F3 commit) — per-archetype pass + stress harness.

The WEGO byte-identity gate lives in the existing digest/golden suite; this
file exercises ONLY the new continuous path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402
from level_loader import LevelData  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation.ruleset import ContinuousRealtime  # noqa: E402
from simulation.unit import Unit  # noqa: E402
from simulation.intents import FP_ONE  # noqa: E402
from control_source import quantize_stick_direction  # noqa: E402

SEED = 20260723


# ---------------------------------------------------------------------------
# Arenas (synthetic — no asset files)
# ---------------------------------------------------------------------------
def _open_arena(h=20, w=20) -> LevelData:
    tm = np.ones((h, w), dtype=np.int32)   # hull border
    tm[1:h - 1, 1:w - 1] = 4               # air interior
    return LevelData(name="fa_open", version="1", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


def _wall_arena(h=20, w=24, wall_col=14) -> LevelData:
    tm = np.ones((h, w), dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = 4
    tm[:, wall_col] = 1                     # solid wall column
    return LevelData(name="fa_wall", version="1", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


def _sim(level) -> Simulation:
    sim = Simulation(level, seed=SEED, breach_physics=bp,
                     enable_recorder=False, ruleset=ContinuousRealtime())
    sim.set_paused(False)
    return sim


def _dummy_enemy(sim, x, y):
    """A stationary team-1 target (is_zombie flipped off so the zombie AI never
    moves it), so a geometry assertion holds still."""
    eid = sim.add_unit(Unit("E", x=x, y=y, team=1))
    e = sim.get_unit(eid)
    e.is_zombie = False
    return e


def _aim_east(sim, uid):
    sim.set_aim(uid, FP_ONE, 0)     # +X unit vector -> facing 0 (east)


# ===========================================================================
# F2 — the §2.1 regression (the bug that motivated the whole design)
# ===========================================================================
def test_directional_fire_hits_enemy_past_occluded_endpoint():
    """A possessed marine facing an enemy ~4 tiles east — with a WALL beyond
    the enemy that occludes the weapon's nominal max-range endpoint tile —
    HITS the enemy while holding TRIGGER. The old band-aid fired at the far
    endpoint and its LOS pre-gate failed on the wall (silent no-fire)."""
    sim = _sim(_wall_arena(wall_col=14))
    uid = sim.add_unit(Unit("M1", x=5, y=8, team=0))
    u = sim.get_unit(uid)
    assert u.weapon_id == "k5_carbine"       # projectile, range 90
    enemy = _dummy_enemy(sim, x=9, y=8)       # centre ~4 tiles east of the marine
    hp0 = enemy.current_hp

    cx, cy = u.center_tile_x(), u.center_tile_y()
    # Precondition that makes this the §2.1 case: the nominal max-range endpoint
    # tile (east, clamped to the grid) is occluded by the wall — the exact
    # pre-gate the old band-aid failed. The enemy itself is CLOSER than the wall.
    assert not sim.gmap.has_los(cy, cx, cy, 22), "endpoint must be wall-occluded"

    _aim_east(sim, uid)
    sim.set_trigger(uid, True)
    for _ in range(12):                       # a few cadence intervals (rof 4)
        sim.step()

    assert enemy.current_hp < hp0, "directional fire must hit the point-blank enemy"


# ===========================================================================
# F2 — bit-reproducibility (the new-path golden substitute)
# ===========================================================================
def _digest(sim):
    return tuple(
        (int(u.id), round(float(u.x), 9), round(float(u.y), 9),
         round(float(u.current_hp), 9), round(float(u.facing), 9), bool(u.alive))
        for u in sorted(sim.units, key=lambda z: z.id))


def _scripted_run():
    sim = _sim(_open_arena(20, 20))
    uid = sim.add_unit(Unit("M1", x=8, y=8, team=0))
    _dummy_enemy(sim, x=13, y=8)
    trace = []
    from simulation.orders import ORDER_MOVE_ATTACK
    for t in range(50):
        ax, ay = np.cos(t * 0.25), np.sin(t * 0.25)
        mdx, mdy, mmag = quantize_stick_direction(float(ax), float(ay))
        if mmag == 0:
            sim.clear_move_dir(uid)
        else:
            sim.set_move_dir(uid, mdx, mdy, ORDER_MOVE_ATTACK)
        sim.set_aim(uid, mdx, mdy)
        sim.set_trigger(uid, t % 3 == 0)
        sim.step()
        trace.append(_digest(sim))
    return trace


def test_scripted_aim_trigger_move_stream_is_bit_reproducible():
    a = _scripted_run()
    b = _scripted_run()
    assert a == b
    # And it actually fired — the enemy lost HP somewhere in the run.
    last = {rec[0]: rec[3] for rec in a[-1]}
    first = {rec[0]: rec[3] for rec in a[0]}
    assert any(last[i] < first[i] for i in last), "the stream never dealt damage"


# ===========================================================================
# F3 — per-archetype pass (projectile / hitscan / spray / grenade)
# ===========================================================================
def test_directional_projectile_hits_along_facing():
    """Projectile (default k5) marches along facing and hits the first unit."""
    sim = _sim(_open_arena())
    uid = sim.add_unit(Unit("M1", x=4, y=9, team=0))
    enemy = _dummy_enemy(sim, x=10, y=9)
    hp0 = enemy.current_hp
    _aim_east(sim, uid)
    sim.set_trigger(uid, True)
    for _ in range(8):
        sim.step()
    assert enemy.current_hp < hp0


def test_directional_hitscan_skewers_and_bites_wall():
    """Hitscan (Lance-3) skewers a unit along facing AND stops on / bites the
    first solid tile it crosses (design §4c)."""
    # Skewer: an enemy directly east takes beam damage.
    sim = _sim(_wall_arena(wall_col=16))
    uid = sim.add_unit(Unit("M1", x=4, y=9, team=0))
    sim.get_unit(uid).weapon_id = "lance_3"    # hitscan
    enemy = _dummy_enemy(sim, x=10, y=9)
    hp0 = enemy.current_hp
    _aim_east(sim, uid)
    sim.set_trigger(uid, True)
    for _ in range(30):                        # lance rof is 12 ticks
        sim.step()
    assert enemy.current_hp < hp0, "hitscan must skewer the enemy along facing"

    # Bite: with no enemy, the beam stops on the wall and chews it.
    sim2 = _sim(_wall_arena(wall_col=16))
    uid2 = sim2.add_unit(Unit("M2", x=4, y=9, team=0))
    sim2.get_unit(uid2).weapon_id = "lance_3"
    wall_hp0 = float(sim2.gmap.wall_hp[9, 16])
    _aim_east(sim2, uid2)
    sim2.set_trigger(uid2, True)
    for _ in range(30):
        sim2.step()
    assert float(sim2.gmap.wall_hp[9, 16]) < wall_hp0, "beam must bite the wall it stops on"


def test_directional_spray_cone_follows_facing():
    """The spray cone bearing is shooter.facing (design §4c), NOT a target
    tile: aiming EAST heats the tiles east of the shooter and none to the
    west."""
    sim = _sim(_open_arena(20, 20))
    uid = sim.add_unit(Unit("M1", x=9, y=9, team=0))
    u = sim.get_unit(uid)
    u.weapon_id = "dragon_7"                   # spray (heat cone)
    cx, cy = u.center_tile_x(), u.center_tile_y()
    _aim_east(sim, uid)
    sim.set_trigger(uid, True)
    for _ in range(3):                         # let a burst deposit a few ticks
        sim.step()
    # The spray's ``heat`` deposit is a per-tick ingress buffer converted into
    # ``temperature`` by the C++ TemperatureSolver within the same step, so we
    # read the persistent field: aiming east heats the tiles east of the
    # shooter and leaves the west (behind the cone) untouched.
    temp = sim.gmap.temperature
    east = float(temp[cy, cx + 3])
    west = float(temp[cy, cx - 3])
    assert east > 0.0, "aiming east must heat the tiles east of the shooter"
    assert west == 0.0, "no heat behind the shooter (cone follows facing)"


def test_directional_grenade_still_throws():
    """THROW is consumed by the rewritten _consume_direct_intents unchanged."""
    sim = _sim(_open_arena(20, 20))
    uid = sim.add_unit(Unit("M1", x=8, y=9, team=0))
    n0 = len(sim.projectiles)
    sim.throw_grenade_intent(uid, FP_ONE, 0, 2.0)   # lob east
    sim.step()
    assert len(sim.projectiles) == n0 + 1


# ===========================================================================
# F3 — STRESS (must not raise; probes the human-test crash)
# ===========================================================================
def test_directional_fire_stress_never_raises():
    """Spam TRIGGER with zero-vector aim, cycle every trigger archetype, fire
    into a wall and off the map edge, kill the possessed unit and possess a
    fresh one (rebind) — no exception. Probes the human-test crash (the deleted
    _aim_fire_order was a prime suspect)."""
    sim = _sim(_wall_arena(h=16, w=16, wall_col=10))
    uid = sim.add_unit(Unit("M1", x=6, y=7, team=0))
    _dummy_enemy(sim, x=8, y=7)

    weapons = ["k5_carbine", "lance_3", "dragon_7"]
    for t in range(60):
        u = sim.get_unit(uid)
        if u is not None and u.alive:
            u.weapon_id = weapons[t % len(weapons)]   # cycle archetypes
            # Aim at everything hostile: east into the wall, west off toward the
            # border, zero-vector (deadzone), and a rotating direction.
            if t % 5 == 0:
                sim.set_aim(uid, 0, 0)                 # zero vector (deadzone)
            elif t % 5 == 1:
                sim.set_aim(uid, -FP_ONE, 0)           # west, toward the border
            else:
                ax, ay = np.cos(t), np.sin(t)
                dx, dy, mag = quantize_stick_direction(float(ax), float(ay))
                if mag:
                    sim.set_aim(uid, dx, dy)
            sim.set_trigger(uid, True)                 # spam trigger every tick
        sim.step()

        if t == 30:
            # Possessed unit dies mid-stream; possess a fresh marine (rebind).
            sim.get_unit(uid).alive = False
            uid = sim.add_unit(Unit("M2", x=3, y=3, team=0))

    # A marine wedged into the corner firing off the map edge — still no raise.
    corner = sim.add_unit(Unit("M3", x=1, y=1, team=0))
    sim.set_aim(corner, -FP_ONE, 0)
    sim.set_trigger(corner, True)
    for _ in range(10):
        sim.step()
