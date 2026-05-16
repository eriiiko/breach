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

    # 3. Render config: derive map area pixels from window-friendly dims
    target_h = 960
    fine_tile_px = target_h / level.height          # fit height into 960px
    map_px_w = int(round(level.width * fine_tile_px))
    map_px_h = int(round(level.height * fine_tile_px))
    panel_px_w = 280
    cfg = RenderConfig(
        map_px_w=map_px_w, map_px_h=map_px_h,
        panel_px_w=panel_px_w,
        fine_tile_px=fine_tile_px,
        grid_w=level.width, grid_h=level.height,
    )
    print(f"  Window: {map_px_w + panel_px_w}x{map_px_h} "
          f"(tile {fine_tile_px:.1f} px)")

    renderer = GameRenderer(level, bp, cfg)

    # Demo: drop some smoke + a tiny fire so we can see overlays
    gmap.smoke[40:60, 18:32] = 0.4

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
                physics.step(gmap, sim_time_per_tick)
                tick_accum -= sim_time_per_tick

            renderer.poll_toggles()

            # Flashlight at mouse cursor
            sources = []
            mouse = renderer.mouse_to_tile()
            if mouse is not None:
                src = bp.LightSource()
                src.x = float(mouse[0])
                src.y = float(mouse[1])
                src.max_range = 20
                src.intensity = 1.4
                src.angle_spread = 6.283  # omni for now (cone later)
                sources.append(src)

            renderer.upload_state(gmap, light_sources=sources)

            renderer.begin_frame()
            renderer.draw_world()
            renderer.draw_units([], [])
            renderer.draw_panel(None)
            renderer.end_frame()
    finally:
        renderer.shutdown()


if __name__ == "__main__":
    main()
