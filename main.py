"""Breach main entry point (Raylib renderer + Simulation facade).

Wires the new :class:`simulation.Simulation` to the existing pyray
renderer. This is the canonical entry point — ``game.py`` (the pygame
legacy) is scheduled for deletion as step 13 of the migration.

Game loop (real time + pause):

  - Sim starts paused. Player plans all orders for the round across
    Phase 1 (preparation) and Phase 2 (engagement) — Tab switches
    which phase the next order belongs to.
  - Spacebar resumes execution; sim ticks at CFG.clock.ticks_per_second
    (12 Hz by default).
  - Sim runs the full round (120 ticks) in one go without pausing
    between phases — phase 1 and phase 2 play through smoothly like
    a movie. Auto-pause fires only at end of round, returning to
    planning for the next round.
  - Backspace undoes last order; Tab switches planning phase; Esc
    clears selection; Ctrl+R reloads config; F8 dumps physics .npz.

Input is in :mod:`input_handler`. The renderer reads ``sim.get_state()``
and ``sim.tick_events`` each frame — it never writes back into the sim.

Run:
    C:/Users/steen/anaconda3/python.exe main.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure we can import the C++ physics module + project modules.
# ``src/`` hosts the new ``simulation`` package — imported as
# ``from simulation import X`` everywhere (no ``src.`` prefix). See
# src/simulation/__init__.py for the convention.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pyray as rl

import breach_physics as bp
from config import CFG
from level_loader import load as load_level
from renderer import GameRenderer
from renderer.game_renderer import RenderConfig
from simulation import Simulation
from simulation.unit import Unit
from input_handler import InputHandler


def main():
    # 1. Load level + build the simulation.
    level_name = getattr(CFG.display, "level", "unhcr_vessel")
    print(f"Loading level: {level_name}")
    level = load_level(level_name)
    print(f"  {level.name} — {level.width}x{level.height} tiles, "
          f"tile size {level.tile_size_m} m")

    sim = Simulation(level, seed=42, breach_physics=bp,
                     enable_recorder=True)

    # ----- Demo scene: pre-breach + smoke source + persistent fire patch.
    # Gameplay isn't fully content-driven yet, so we keep the visible
    # ambient effects from the prior main loop so the renderer has
    # something to show. Replace with level-defined hazards later.
    SMOKE_SOURCE = (slice(45, 50), slice(22, 28))
    FIRE_SOURCE  = (slice(75, 80), slice(22, 28))
    sim.gmap.flammable[FIRE_SOURCE] = True
    sim.gmap.fire[FIRE_SOURCE] = 0.8
    BREACH = (slice(58, 64), slice(33, 36))
    sim.gmap.material[BREACH] = 0  # MAT_AIR
    sim.gmap.solid[BREACH] = False
    sim.gmap.is_vacuum[BREACH] = True
    sim.gmap.atmosphere[BREACH] = 0.0
    sim.gmap.obstacles[BREACH] = False

    # ----- Spawn units from level.toml [[spawn]] entries.
    if not level.spawns:
        raise RuntimeError(
            f"Level '{level.name}' has no [[spawn]] entries — nothing to play."
        )
    for s in level.spawns:
        sim.add_unit(Unit(s.name, x=s.x, y=s.y, team=s.team,
                          footprint=s.footprint))
    print(f"  Spawned {len(level.spawns)} units from level.toml")

    # 2. Render config — borderless windowed at monitor resolution.
    from renderer.camera import Camera2D

    BORDERLESS = "--windowed" not in sys.argv
    if BORDERLESS:
        # Borderless windowed mode uses the actual monitor size, not the
        # numbers we pass to init_window. Open the window first so we can
        # query the real dimensions and lay the panel/map out to fit.
        from renderer import core as rcore
        rcore.init_window(0, 0, title=f"Breach — {level.name}",
                          borderless=True)
        screen_w, screen_h = rcore.get_monitor_size()
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
    print(f"  WASD/arrows pan | Q/E or wheel zoom | "
          f"Space resume | Tab phase | Bksp undo | Ctrl+R reload")
    print(f"  DEBUG: T toggles temperature overlay (black-body heat ramp) | "
          f"I ignites the tile under the cursor")

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
    renderer.lighting.set_ambient((0.10, 0.10, 0.13))

    input_handler = InputHandler()

    # Static emergency lights — always-on, scattered through the ship.
    static_lights = []
    for (lx, ly) in [(25, 10), (25, 30), (25, 55), (25, 88), (25, 110)]:
        src = bp.LightSource()
        src.x, src.y = float(lx), float(ly)
        src.max_range = 18
        src.intensity = 0.9
        src.angle_spread = 6.283
        # Emergency lighting — red (profile: emergency_light).
        src.color = (1.0, 0.1, 0.05)
        # Fire-style sources default to jitter=0.0 — natural smoke
        # advection creates the flicker we want without C++ RNG drift.
        src.jitter = 0.0
        static_lights.append(src)

    # 3. Main loop.
    last_time = time.perf_counter()
    sim_time_per_tick = 1.0 / float(CFG.clock.ticks_per_second)
    tick_accum = 0.0

    try:
        while not renderer.should_close():
            now = time.perf_counter()
            dt = now - last_time
            last_time = now

            # ----- Input first (may toggle pause / queue orders) -----
            renderer.poll_toggles()
            renderer.update_camera(dt)
            input_handler.handle_frame(sim, renderer)

            # ----- Tick the simulation while not paused -----
            if not sim.is_paused():
                # Persistent demo hazards.
                sim.gmap.smoke[SMOKE_SOURCE] = np.maximum(
                    sim.gmap.smoke[SMOKE_SOURCE], 0.7)
                sim.gmap.fire[FIRE_SOURCE] = np.maximum(
                    sim.gmap.fire[FIRE_SOURCE], 0.8)

                tick_accum += dt
                # Cap the per-frame catch-up to avoid spirals if a
                # background pause stalls the loop.
                max_catch_up = 5
                steps = 0
                while tick_accum >= sim_time_per_tick and steps < max_catch_up:
                    sim.step()
                    tick_accum -= sim_time_per_tick
                    steps += 1
                    # Sim may auto-pause mid-batch (phase boundary).
                    if sim.is_paused():
                        break

            # ----- Lights: mouse flashlight + static emergencies -----
            sources = list(static_lights)
            mouse_f = renderer.mouse_to_tile_float()
            if mouse_f is not None:
                src = bp.LightSource()
                src.x = float(mouse_f[0])
                src.y = float(mouse_f[1])
                src.max_range = 25
                src.intensity = 2.5
                src.angle_spread = 6.283
                # Flashlight — cool white (profile: flashlight).
                src.color = (1.0, 1.0, 0.95)
                src.jitter = 0.0
                sources.append(src)

            # ----- Upload + draw -----
            renderer.upload_state(sim.gmap, light_sources=sources)
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
            renderer.draw_debug_hud(sim.gmap)

            # Pull tick events into the renderer's effect queue, then age
            # the queue. tick_events is cleared by the next sim.step(),
            # so we MUST consume here every frame.
            renderer.consume_events(sim.tick_events)
            renderer._advance_effects(dt)

            selected = sim.get_unit(input_handler.selected_unit_id) \
                if input_handler.selected_unit_id is not None else None
            renderer.draw_panel(
                sim=sim,
                selected_unit=selected,
                planning_phase=input_handler.planning_phase,
                current_mode=input_handler.current_mode,
            )
            renderer.end_frame()
    finally:
        renderer.shutdown()


if __name__ == "__main__":
    main()
