"""Breach renderer package: pyray-based replacement for pygame.

Public interface:
    from renderer import GameRenderer
    renderer = GameRenderer(level_data, screen_size, panel_width)
    renderer.upload_state(gmap)
    renderer.begin_frame()
    renderer.draw_world()
    renderer.draw_units(units)
    renderer.draw_overlays(state)
    renderer.end_frame()
"""

from .game_renderer import GameRenderer

__all__ = ["GameRenderer"]
