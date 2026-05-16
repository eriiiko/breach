"""Breach renderer package: pyray-based replacement for pygame.

Public interface:
    from renderer import GameRenderer, RenderConfig
    from renderer.camera import Camera2D

    renderer = GameRenderer(level_data, breach_physics, cfg, initial_camera=cam)

Each frame:
    renderer.upload_state(gmap, light_sources=...)
    renderer.begin_frame()
    renderer.compose_world(units_marines=..., units_zombies=...,
                           orders_per_unit=...)
    renderer.blit_world_to_screen()
    renderer.draw_panel(state)
    renderer.end_frame()

Architecture:
    All world-space drawing happens inside compose_world() into a single
    RenderTexture2D owned by WorldComposite. The camera transform is a
    single DrawTexturePro blit from that RT to the screen, applied in
    blit_world_to_screen(). This means shaders never need camera UV
    uniforms — fragTexCoord 0..1 always corresponds to the full world.

    See docs/design_camera_and_coordinate_systems_research.md for the
    rationale and step-by-step migration map.
"""

from .game_renderer import GameRenderer, RenderConfig

__all__ = ["GameRenderer", "RenderConfig"]
