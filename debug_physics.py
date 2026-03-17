"""
Physics debug test — headless simulation with automated events.
No player input, no rendering. Logs atmosphere stats to CSV each tick.

Usage:
    python debug_physics.py

Output:
    debug_physics_log.csv
"""
import sys
import os
import csv
import numpy as np
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cpp', 'build', 'Release'))

from config import CFG

# Try C++ physics
try:
    import breach_physics
    HAS_CPP = True
    print("[debug] C++ physics loaded")
except ImportError:
    HAS_CPP = False
    print("[debug] C++ physics NOT found, using Python")

# Material constants (must match game.py)
MAT_AIR = 0
MAT_HULL = 1
MAT_WOOD = 2
MAT_DOOR = 3


def build_test_map():
    """Simple sealed room with wood walls inside."""
    fw = CFG.display.fine_w   # 120
    fh = CFG.display.fine_h   # 75
    co = CFG.display.coarse   # 3

    material = np.full((fh, fw), MAT_AIR, dtype=np.int8)

    # Hull border
    material[0:co, :] = MAT_HULL
    material[fh-co:, :] = MAT_HULL
    material[:, 0:co] = MAT_HULL
    material[:, fw-co:] = MAT_HULL

    # Internal wood wall at x=60 (middle)
    for y in range(co, fh - co):
        material[y, 60] = MAT_WOOD

    # Door in the middle of the wood wall
    for dy in range(co):
        material[37 + dy, 60] = MAT_DOOR

    # Build derived arrays
    is_wall = np.isin(material, [MAT_HULL, MAT_WOOD])
    is_vacuum = np.zeros((fh, fw), dtype=bool)
    is_vacuum[0:co, :] = True
    is_vacuum[fh-co:, :] = True
    is_vacuum[:, 0:co] = True
    is_vacuum[:, fw-co:] = True
    flammable = (material == MAT_WOOD)
    obstacles = is_wall.copy()

    # Set wall HP
    wall_hp = np.zeros((fh, fw), dtype=np.float32)
    wall_hp[material == MAT_HULL] = 300.0
    wall_hp[material == MAT_WOOD] = 60.0
    wall_hp[material == MAT_DOOR] = 40.0

    # Physics fields
    atmosphere = np.ones((fh, fw), dtype=np.float32)
    atmosphere[is_wall] = 0.0
    atmosphere[is_vacuum] = 0.0
    wave_p = np.zeros((fh, fw), dtype=np.float32)
    wave_v = np.zeros((fh, fw), dtype=np.float32)
    wave_source = np.zeros((fh, fw), dtype=np.float32)
    smoke = np.zeros((fh, fw), dtype=np.float32)
    fire = np.zeros((fh, fw), dtype=np.float32)
    light_map = np.zeros((fh, fw), dtype=np.float32)

    return {
        'material': material, 'is_wall': is_wall, 'is_vacuum': is_vacuum,
        'flammable': flammable, 'obstacles': obstacles, 'wall_hp': wall_hp,
        'atmosphere': atmosphere, 'wave_p': wave_p, 'wave_v': wave_v,
        'wave_source': wave_source, 'smoke': smoke, 'fire': fire,
        'light_map': light_map, 'fh': fh, 'fw': fw,
    }


def apply_explosion(m, fy, fx, radius, pressure, wall_damage):
    """Same logic as game.py Physics.apply_explosion."""
    fh, fw = m['fh'], m['fw']
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ny, nx = fy + dy, fx + dx
            if 0 <= ny < fh and 0 <= nx < fw:
                dist = math.sqrt(dy * dy + dx * dx)
                if dist <= radius:
                    falloff = 1.0 - (dist / radius)
                    mat = m['material'][ny, nx]
                    if mat in (MAT_HULL, MAT_WOOD, MAT_DOOR):
                        m['wall_hp'][ny, nx] -= wall_damage * falloff
                        if m['wall_hp'][ny, nx] <= 0:
                            m['material'][ny, nx] = MAT_AIR
                            m['is_wall'][ny, nx] = False
                            m['flammable'][ny, nx] = False
                            m['obstacles'][ny, nx] = False
                    if not m['is_wall'][ny, nx] and not m['is_vacuum'][ny, nx]:
                        m['wave_source'][ny, nx] += pressure * falloff
                    # Ignite flammable
                    if m['flammable'][ny, nx] and dist <= radius * 0.7:
                        m['fire'][ny, nx] = max(m['fire'][ny, nx], 0.5 * falloff)

    # Center-only atmosphere deposit
    if not m['is_wall'][fy, fx] and not m['is_vacuum'][fy, fx]:
        m['atmosphere'][fy, fx] += pressure * 0.3


def ignite_wall(m, fy, fx, length=5):
    """Set fire on a stretch of wood wall."""
    for dy in range(-length // 2, length // 2 + 1):
        y = fy + dy
        if 0 <= y < m['fh'] and m['flammable'][y, fx]:
            m['fire'][y, fx] = 0.8


def run_test():
    m = build_test_map()
    sim_time = 1.0 / CFG.clock.ticks_per_second

    # Set up C++ solvers
    if HAS_CPP:
        wave = breach_physics.WaveSolver()
        wave.c = 300.0
        wave.damping = 3.0
        wave.transfer = 0.5
        wave.feed_rate = 200.0

        atmo = breach_physics.AtmoDiffusion()
        atmo.d_atm = CFG.physics.d_atm

        smoke_solver = breach_physics.SmokeDynamics()
        smoke_solver.d_smoke = CFG.physics.d_smoke
        smoke_solver.advection_rate = CFG.physics.advection_rate

        fire_solver = breach_physics.FireSimulation()

        raycaster = breach_physics.Raycaster()
        raycaster.coarse_cluster = CFG.display.coarse

    # Scheduled events
    events = {
        10: ("grenade at (30, 37)", lambda: apply_explosion(m, 37, 30, 6, 10.0, 200)),
        60: ("grenade at (90, 37)", lambda: apply_explosion(m, 37, 90, 6, 10.0, 200)),
        120: ("ignite wood wall at x=60", lambda: ignite_wall(m, 37, 60)),
        200: ("explosive on wood wall at (60, 20)", lambda: apply_explosion(m, 20, 60, 3, 5.0, 500)),
    }

    total_ticks = 400
    log_file = "debug_physics_log.csv"

    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'tick', 'event',
            'atm_min', 'atm_max', 'atm_mean',
            'wave_p_min', 'wave_p_max',
            'fire_max', 'smoke_max',
            'wave_source_max'
        ])

        for tick in range(total_ticks):
            event_name = ""

            # Trigger scheduled events
            if tick in events:
                event_name, action = events[tick]
                print(f"  Tick {tick}: {event_name}")
                action()

            # Run physics
            if HAS_CPP:
                wave.step(m['wave_p'], m['wave_v'], m['wave_source'],
                          m['atmosphere'], m['obstacles'], m['is_wall'],
                          m['is_vacuum'], sim_time)
                atmo.step(m['atmosphere'], m['obstacles'], m['is_wall'],
                          m['is_vacuum'], sim_time)
                smoke_solver.step(m['smoke'], m['atmosphere'], m['wave_p'],
                                  m['obstacles'], m['is_wall'], m['is_vacuum'],
                                  sim_time)
                destroyed = fire_solver.step(m['fire'], m['atmosphere'], m['smoke'],
                                             m['wall_hp'], m['is_wall'],
                                             m['flammable'], sim_time)
                for fy, fx in destroyed:
                    m['material'][fy, fx] = MAT_AIR
                    m['is_wall'][fy, fx] = False
                    m['flammable'][fy, fx] = False
                    m['obstacles'][fy, fx] = False

                raycaster.update_from_fire(m['light_map'], m['fire'],
                                           m['smoke'], m['is_wall'])

            # Log stats
            interior = ~m['is_wall'] & ~m['is_vacuum']
            atm = m['atmosphere'][interior]
            writer.writerow([
                tick, event_name,
                f"{atm.min():.4f}", f"{atm.max():.4f}", f"{atm.mean():.4f}",
                f"{m['wave_p'].min():.4f}", f"{m['wave_p'].max():.4f}",
                f"{m['fire'].max():.4f}", f"{m['smoke'].max():.4f}",
                f"{m['wave_source'].max():.4f}"
            ])

            # Alert on anomaly
            if abs(atm.max()) > 5 or abs(atm.min()) > 5:
                print(f"  !! Tick {tick}: atmosphere out of range [{atm.min():.2f}, {atm.max():.2f}]")

    print(f"\nDone. {total_ticks} ticks logged to {log_file}")


if __name__ == "__main__":
    run_test()
