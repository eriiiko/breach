"""Smoke test: open the renderer with a real level, render a few frames with
a synthetic flashlight at the cursor, and exit cleanly.

Run:
    C:/Users/steen/anaconda3/python.exe tests/test_renderer_smoke.py
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

# Add project root + C++ build dir to import path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))

import numpy as np
import pyray as rl

import breach_physics as bp
from level_loader import load as load_level, materials_from_tilemap
from renderer import GameRenderer
from renderer.game_renderer import RenderConfig


def main():
    # Load level
    level = load_level("unhcr_vessel")
    print(f"Loaded level: {level.name} ({level.width}x{level.height})")

    # Build a minimal gmap-shim with just the fields the renderer needs.
    class Shim:
        pass
    g = Shim()
    g.smoke = np.zeros((level.height, level.width), dtype=np.float32)
    g.fire  = np.zeros((level.height, level.width), dtype=np.float32)
    mat, vac = materials_from_tilemap(level.tilemap, level.version)
    g.material  = mat
    g.is_vacuum = vac
    g.solid     = np.isin(mat, [1])     # MAT_HULL only for now
    g.obstacles = g.solid.copy()       # no units in this test
    # Static per-channel light attenuation (ch.03 march input). Derive from the
    # material table so opaque tiles ([1,1,1]) block light like the old wall
    # hard-stop. upload_state reads gmap.light_atten now (replaces the bool mask).
    from simulation.materials import MaterialTable
    from config import CFG
    g.light_atten = np.ascontiguousarray(
        MaterialTable.from_config(CFG).light_atten[mat], dtype=np.float32)
    # Dynamic per-channel attenuation field (ch.03 §units): the live field the
    # march reads = static material atten MAX'd with stamped-unit opacity. With
    # no units in this smoke test it equals the static field. (A real GameMap
    # rebuilds this in stamp_units each tick; the shim has no units.)
    g.dyn_light_atten = g.light_atten.copy()

    # Drop some smoke and fire for visual test
    g.smoke[60:80, 20:30] = 0.7
    g.fire[110:115, 25:30] = 1.0

    # Render config: window size = map area + panel
    MAP_PX_W = 400          # 8 px per tile horizontally (50 tiles)
    MAP_PX_H = 960          # 8 px per tile vertically  (120 tiles)
    PANEL_W  = 280
    cfg = RenderConfig(
        map_px_w=MAP_PX_W, map_px_h=MAP_PX_H,
        panel_px_w=PANEL_W,
        grid_w=level.width, grid_h=level.height,
        world_px_per_tile=8.0,
    )

    renderer = GameRenderer(level, bp, cfg)
    renderer.show_lighting = True

    frames = 0
    try:
        while not renderer.should_close():
            renderer.poll_toggles()

            # Build one moving light source at the mouse
            sources = []
            mouse = renderer.mouse_to_tile()
            if mouse is not None:
                src = bp.LightSource()
                src.x = float(mouse[0])
                src.y = float(mouse[1])
                src.max_range = 25
                src.intensity = 1.5
                src.angle_spread = 6.283  # omni
                sources.append(src)

            renderer.upload_state(g, light_sources=sources)

            renderer.begin_frame()
            renderer.compose_world(units_marines=[], units_zombies=[])
            renderer.draw_background_to_screen()
            renderer.blit_world_to_screen()
            renderer.draw_panel(None)
            renderer.end_frame()

            frames += 1
            if frames >= 600 and "--auto" in sys.argv:
                break
    finally:
        renderer.shutdown()

    print(f"OK — rendered {frames} frames")


if __name__ == "__main__":
    main()
