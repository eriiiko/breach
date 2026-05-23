"""Smoke test for main.py: open the wired-up renderer + Simulation, run a
handful of frames, and exit cleanly.

Mirrors ``tests/test_renderer_smoke.py`` but for the full game loop —
proves that the Simulation facade can drive the renderer end-to-end without
exceptions (input handler, projectile drawing, event consumption, panel
update with sim, all in one).

Run:
    C:/Users/steen/anaconda3/python.exe tests/test_main_smoke.py --auto
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pyray as rl

import breach_physics as bp
from config import CFG
from input_handler import InputHandler
from level_loader import load as load_level
from renderer import GameRenderer
from renderer.camera import Camera2D
from renderer.game_renderer import RenderConfig
from simulation import Simulation
from simulation.unit import Unit


def main():
    level = load_level("unhcr_vessel")
    sim = Simulation(level, seed=42, breach_physics=bp,
                     enable_recorder=False)

    # Spawn a small squad so units/effects exercise their code paths.
    # Coords are physics-tile positions inside the ship interior.
    sim.add_unit(Unit("Alpha", x=22, y=50, team=0))
    sim.add_unit(Unit("Bravo", x=28, y=50, team=0))
    sim.add_unit(Unit("Zomb1", x=22, y=70, team=1))

    cfg = RenderConfig(
        map_px_w=400, map_px_h=960, panel_px_w=280,
        grid_w=level.width, grid_h=level.height,
        world_px_per_tile=8.0,
    )
    initial_camera = Camera2D(
        pos_tile_x=0.0, pos_tile_y=0.0,
        zoom_px_per_tile=8.0,
        viewport_px_w=400, viewport_px_h=960,
        world_size_tile_w=level.width, world_size_tile_h=level.height,
    )
    renderer = GameRenderer(level, bp, cfg, initial_camera=initial_camera,
                            borderless=False)
    input_handler = InputHandler()

    # Run a brief unpaused sim so events fire.
    sim.set_paused(False)

    last_time = time.perf_counter()
    sim_time_per_tick = 1.0 / float(CFG.clock.ticks_per_second)
    tick_accum = 0.0

    frames = 0
    try:
        while not renderer.should_close():
            now = time.perf_counter()
            dt = now - last_time
            last_time = now

            renderer.poll_toggles()
            input_handler.handle_frame(sim, renderer)

            if not sim.is_paused():
                tick_accum += dt
                steps = 0
                while tick_accum >= sim_time_per_tick and steps < 5:
                    sim.step()
                    tick_accum -= sim_time_per_tick
                    steps += 1

            renderer.upload_state(sim.gmap, light_sources=[])
            renderer.begin_frame()
            renderer.compose_world(
                units_marines=sim.marines(),
                units_zombies=sim.zombies(),
                projectiles=sim.projectiles,
                orders_phase1=sim.orders_for_phase(0),
                orders_phase2=sim.orders_for_phase(1),
                current_phase=input_handler.planning_phase,
            )
            renderer.draw_background_to_screen()
            renderer.blit_world_to_screen()
            renderer.consume_events(sim.tick_events)
            renderer._advance_effects(dt)
            renderer.draw_panel(sim=sim, selected_unit=None,
                                planning_phase=input_handler.planning_phase,
                                current_mode=input_handler.current_mode)
            renderer.end_frame()

            frames += 1
            if frames >= 600 and "--auto" in sys.argv:
                break
    finally:
        renderer.shutdown()

    print(f"OK — main_smoke rendered {frames} frames; "
          f"sim.tick={sim.tick}, phase={sim.phase}, paused={sim.paused}")


if __name__ == "__main__":
    main()
