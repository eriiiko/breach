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
