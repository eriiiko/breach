"""Smoke test: open the renderer with a real level, render a few frames and
exit cleanly. (Its synthetic mouse flashlight and the march-input shim fields
went with the render march at ray-engine-v2 P6c; the end-to-end check is
tools/e2e_drive.py.)

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

            # (The old synthetic mouse flashlight -- a bp.LightSource cast by
            # the render march -- went with the march at ray-engine-v2 P6c;
            # the light is the sweep's, read from gmap through the accessor.
            # The end-to-end check is tools/e2e_drive.py.)
            renderer.upload_state(g)

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
