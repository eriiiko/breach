"""Breach main entry point (Raylib renderer).

Loads a level, initializes the C++ physics module, runs a basic game loop
with a cursor-driven flashlight. Marines, orders, and full turn system are
TODO — this is the foundation that replaces the pygame entry in game.py.

Run:
    C:/Users/steen/anaconda3/python.exe main.py
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

# Ensure we can import the C++ physics module + project modules
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))
sys.path.insert(0, str(ROOT))

import numpy as np
import pyray as rl

import breach_physics as bp
from config import CFG
from level_loader import load as load_level, materials_from_tilemap
from renderer import GameRenderer
from renderer.game_renderer import RenderConfig


# ---------------------------------------------------------------------------
# Minimal GameMap shim — replaces game.GameMap for the renderer's needs.
# ---------------------------------------------------------------------------

class GameMap:
    """Bare GameMap: just the fields the renderer + physics need. Full version
    in game.py — we'll migrate it later. For now this is enough to test the
    pipeline end-to-end with the new renderer."""

    def __init__(self, level):
        h, w = level.height, level.width
        self.level = level
        self.material  = np.zeros((h, w), dtype=np.int8)
        self.is_wall   = np.zeros((h, w), dtype=bool)
        self.is_vacuum = np.zeros((h, w), dtype=bool)
        self.flammable = np.zeros((h, w), dtype=bool)
        self.wall_hp   = np.zeros((h, w), dtype=np.float32)
        self.atmosphere = np.ones((h, w), dtype=np.float32)
        self.wave_p     = np.zeros((h, w), dtype=np.float32)
        self.wave_v     = np.zeros((h, w), dtype=np.float32)
        self.wave_source= np.zeros((h, w), dtype=np.float32)
        self.wind_x     = np.zeros((h, w), dtype=np.float32)
        self.wind_y     = np.zeros((h, w), dtype=np.float32)
        self.smoke      = np.zeros((h, w), dtype=np.float32)
        self.fire       = np.zeros((h, w), dtype=np.float32)
        self.obstacles  = np.zeros((h, w), dtype=bool)

        # Populate from level
        mat, vac = materials_from_tilemap(level.tilemap)
        self.material[:] = mat
        self.is_wall[:]  = (mat == 1)   # MAT_HULL
        self.is_vacuum[:] = vac
        self.atmosphere[vac] = 0.0
        self.flammable[:] = False  # No wood in this level (yet)
        self.obstacles[:] = self.is_wall

        # HP from config (only hull cells need HP)
        try:
            hull_hp = CFG.materials.hull[0]
        except Exception:
            hull_hp = 300.0
        self.wall_hp[self.is_wall] = float(hull_hp)


# ---------------------------------------------------------------------------
# Physics step adapter
# ---------------------------------------------------------------------------

class PhysicsRunner:
    """Wraps the C++ atmosphere/smoke/fire solvers."""

    def __init__(self):
        self.atmos = bp.AtmosphereSolver()
        self.atmos.c             = float(CFG.physics.wave_c)
        self.atmos.damping       = float(CFG.physics.wave_damping)
        self.atmos.transfer      = float(CFG.physics.wave_transfer)
        self.atmos.d_atm         = float(CFG.physics.d_atm)
        self.atmos.feed_rate     = float(CFG.physics.source_feed_rate)
        self.atmos.breach_rate   = float(CFG.physics.breach_rate)

        self.smoke = bp.SmokeDynamics()
        self.smoke.d_smoke              = float(CFG.physics.d_smoke)
        self.smoke.advection_rate       = float(CFG.physics.advection_rate)
        self.smoke.dt_scale             = float(CFG.physics.smoke_dt_scale)
        self.smoke.wind_diffusion_scale = float(CFG.physics.wind_diffusion_scale)

        self.fire = bp.FireSimulation()

    def step(self, gmap: GameMap, sim_time: float) -> None:
        """Advance physics by sim_time seconds (default: 1 game tick)."""
        dt = self.atmos.max_dt()
        n = max(1, int(math.ceil(sim_time / dt)))
        dt_actual = sim_time / n
        for _ in range(n):
            self.atmos.step(
                gmap.wave_p, gmap.wave_v, gmap.wave_source, gmap.atmosphere,
                gmap.wind_x, gmap.wind_y,
                gmap.obstacles, gmap.is_wall, gmap.is_vacuum,
                dt_actual,
            )
            self.smoke.step(
                gmap.smoke, gmap.wind_x, gmap.wind_y,
                gmap.obstacles, gmap.is_wall, gmap.is_vacuum,
                dt_actual * self.smoke.dt_scale,
            )
        # Fire step at full sim_time
        destroyed = self.fire.step(
            gmap.fire, gmap.atmosphere, gmap.smoke, gmap.wall_hp,
            gmap.is_wall, gmap.flammable,
            sim_time,
        )
        return destroyed


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    # 1. Load level
    level_name = getattr(CFG.display, "level", "unhcr_vessel")
    print(f"Loading level: {level_name}")
    level = load_level(level_name)
    print(f"  {level.name} — {level.width}x{level.height} tiles, "
          f"tile size {level.tile_size_m} m")

    # 2. Build map + physics
    gmap = GameMap(level)
    physics = PhysicsRunner()

    # 3. Render config — borderless windowed at monitor resolution.
    # We need to open a temporary window first so Raylib can query the
    # monitor; then we resize. Simpler: open borderless directly with a
    # placeholder size; Raylib's borderless mode picks up the monitor.
    from renderer.camera import Camera2D
    from renderer import core as rcore

    BORDERLESS = "--windowed" not in sys.argv
    if BORDERLESS:
        # Will be sized to monitor by Raylib; supply hints anyway.
        screen_w, screen_h = 1920, 1080
    else:
        screen_w, screen_h = 1280, 720

    panel_px_w = 280
    map_px_w   = screen_w - panel_px_w
    map_px_h   = screen_h
    cfg = RenderConfig(
        map_px_w=map_px_w, map_px_h=map_px_h,
        panel_px_w=panel_px_w,
        grid_w=level.width, grid_h=level.height,
        world_px_per_tile=24.0,
    )
    print(f"  Window: {screen_w}x{screen_h} "
          f"(borderless={BORDERLESS}, world RT "
          f"{int(level.width*cfg.world_px_per_tile)}x"
          f"{int(level.height*cfg.world_px_per_tile)})")
    print(f"  WASD/arrows pan | Q/E or mousewheel zoom | [/] light Z | F1-F5,B,G,H toggles")

    # Zoom: fit the world WIDTH in the viewport (vertical scrolling for tall
    # ships). Bounded to keep tiles readable.
    fit_w_zoom = map_px_w / max(level.width, 1)
    initial_zoom = max(20.0, min(64.0, fit_w_zoom))
    initial_camera = Camera2D(
        pos_tile_x=0.0, pos_tile_y=0.0,
        zoom_px_per_tile=initial_zoom,
        viewport_px_w=map_px_w, viewport_px_h=map_px_h,
        world_size_tile_w=level.width, world_size_tile_h=level.height,
    )
    renderer = GameRenderer(level, bp, cfg,
                            initial_camera=initial_camera,
                            borderless=BORDERLESS)

    # Dramatic dark scene so lighting is obvious
    renderer.lighting.set_ambient((0.10, 0.10, 0.13))

    # ----- Demo scene setup -----
    # 1. Persistent smoke source in mid-ship (re-deposited each tick)
    SMOKE_SOURCE = (slice(45, 50), slice(22, 28))
    # 2. Fire intensity bound to wall tiles (won't actually spread without
    #    flammable walls but the orange overlay is visible)
    FIRE_SOURCE = (slice(75, 80), slice(22, 28))
    gmap.flammable[FIRE_SOURCE] = True  # let it persist
    gmap.fire[FIRE_SOURCE] = 0.8
    # 3. Pre-breach the hull on the starboard side, mid-ship — air will rush out
    BREACH = (slice(58, 64), slice(33, 36))
    gmap.material[BREACH] = 0  # MAT_AIR
    gmap.is_wall[BREACH] = False
    gmap.is_vacuum[BREACH] = True
    gmap.atmosphere[BREACH] = 0.0
    gmap.obstacles[BREACH] = False

    # Static emergency lights — always-on, scattered through the ship
    static_lights = []
    for (lx, ly, color_unused) in [
        (25, 10, None),   # cockpit
        (25, 30, None),   # crew quarters
        (25, 55, None),   # lab
        (25, 88, None),   # plants
        (25, 110, None),  # storage
    ]:
        src = bp.LightSource()
        src.x, src.y = float(lx), float(ly)
        src.max_range = 18
        src.intensity = 0.9
        src.angle_spread = 6.283
        static_lights.append(src)

    # 4. Run loop
    last_time = time.perf_counter()
    sim_time_per_tick = 1.0 / float(CFG.clock.ticks_per_second)
    tick_accum = 0.0

    try:
        while not renderer.should_close():
            now = time.perf_counter()
            dt = now - last_time
            last_time = now

            # Step physics at game tick rate
            tick_accum += dt
            while tick_accum >= sim_time_per_tick:
                # Re-deposit smoke + fire each tick so the demo stays visible
                gmap.smoke[SMOKE_SOURCE] = np.maximum(gmap.smoke[SMOKE_SOURCE], 0.7)
                gmap.fire[FIRE_SOURCE]   = np.maximum(gmap.fire[FIRE_SOURCE],   0.8)
                physics.step(gmap, sim_time_per_tick)
                tick_accum -= sim_time_per_tick

            renderer.poll_toggles()
            renderer.update_camera(dt)

            # Flashlight at mouse cursor + static emergency lights.
            # Use the FRACTIONAL cursor coordinate so the brightness peak
            # appears exactly under the cursor (not half-a-tile south of it
            # due to bilinear texel-center sampling).
            sources = list(static_lights)
            mouse_f = renderer.mouse_to_tile_float()
            if mouse_f is not None:
                src = bp.LightSource()
                src.x = float(mouse_f[0])
                src.y = float(mouse_f[1])
                src.max_range = 25
                src.intensity = 2.5
                src.angle_spread = 6.283
                sources.append(src)

            renderer.upload_state(gmap, light_sources=sources)

            renderer.begin_frame()
            renderer.compose_world(units_marines=[], units_zombies=[])
            renderer.blit_world_to_screen()
            renderer.draw_panel(None)
            renderer.end_frame()
    finally:
        renderer.shutdown()


if __name__ == "__main__":
    main()
