"""GameRenderer: thin orchestrator over core/lighting/overlays/world_composite.

Public API (used by main.py):
    renderer.upload_state(gmap, light_sources)
    renderer.begin_frame()
    renderer.compose_world(...)        # everything in world space, inside RT
    renderer.blit_world_to_screen()    # camera blit from RT to map area
    renderer.draw_panel(state)         # right-side UI on top of screen
    renderer.end_frame()

State flow:
    Game logic ---> upload_state ---> textures (light/smoke/fire)
    Render: compose_world -> world RT -> blit_to_screen -> screen
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import pyray as rl

from . import core
from .camera import Camera2D
from .lighting import LightingPass
from .overlays import (
    FieldOverlay, FireOverlay,
    draw_unit, draw_waypoint_line, draw_grid, draw_text, draw_panel_background,
)
from .world_composite import WorldComposite


@dataclass
class RenderConfig:
    """Static configuration set at construction time. Camera state lives on
    the Camera2D object, not here — keep RenderConfig immutable after init.

    `world_px_per_tile` is the only place this value is configured; the
    WorldComposite reads it via cfg.world_px_per_tile at construction. Do
    not duplicate this value elsewhere — single source of truth.
    """
    map_px_w: int          # pixels for the map area (window minus panel)
    map_px_h: int
    panel_px_w: int        # right-side panel width
    grid_w: int            # physics grid width in tiles
    grid_h: int            # physics grid height in tiles
    world_px_per_tile: float = 24.0   # world RT resolution (independent of zoom)


class GameRenderer:
    """Encapsulates pyray drawing. One per game session."""

    def __init__(self, level_data, breach_physics, cfg: RenderConfig,
                 initial_camera: Optional[Camera2D] = None):
        self.level = level_data
        self.bp = breach_physics
        self.cfg = cfg

        total_w = cfg.map_px_w + cfg.panel_px_w
        total_h = cfg.map_px_h
        core.init_window(total_w, total_h, title=f"Breach — {level_data.name}")

        # Camera (default: top-left of world, zoom set to fit width or fixed)
        if initial_camera is not None:
            self.camera = initial_camera
        else:
            # Pick a zoom such that the world width fits in the viewport
            default_zoom = cfg.map_px_w / cfg.grid_w
            self.camera = Camera2D(
                pos_tile_x=0.0, pos_tile_y=0.0,
                zoom_px_per_tile=default_zoom,
                viewport_px_w=cfg.map_px_w, viewport_px_h=cfg.map_px_h,
                world_size_tile_w=cfg.grid_w, world_size_tile_h=cfg.grid_h,
            )

        # World RT — all world-space draws go into this.
        self.world = WorldComposite(
            world_tile_w=cfg.grid_w, world_tile_h=cfg.grid_h,
            world_px_per_tile=cfg.world_px_per_tile,
        )

        # Level textures + lighting + overlays
        self.textures = core.load_level_textures(level_data)
        self.raycaster = breach_physics.Raycaster()
        self.raycaster.smoke_absorption = 0.8
        self.lighting = LightingPass(self.raycaster, cfg.grid_h, cfg.grid_w)
        self.smoke_overlay = FieldOverlay(cfg.grid_h, cfg.grid_w,
                                          tint=(190, 195, 210), max_alpha=180)
        self.fire_overlay = FireOverlay(cfg.grid_h, cfg.grid_w)

        # Toggles
        self.show_grid = False
        self.show_smoke = True
        self.show_fire = True
        self.show_lighting = True
        self.show_normal_map = True
        self.normal_y_flipped = False
        self.srgb_decode = True

        # Frame timing
        self.last_frame_ms = 0.0
        self.last_raycast_ms = 0.0

        self.lighting.set_ambient((0.18, 0.18, 0.22))

    # ---- per-frame physics->GPU upload ---------------------------------

    def upload_state(self, gmap, light_sources: Optional[List] = None) -> None:
        t_start = time.perf_counter()

        # Light field
        if self.show_lighting and light_sources:
            t_ray = time.perf_counter()
            self.lighting.compute_light_field(light_sources, gmap.smoke, gmap.is_wall)
            self.last_raycast_ms = (time.perf_counter() - t_ray) * 1000
        else:
            self.lighting.light_map.fill(0)
            self.lighting.light_dx.fill(0)
            self.lighting.light_dy.fill(0)
            self.lighting.packed.fill(0)
            self.lighting.packed[..., 3] = 255
            core.update_rgba_texture(self.lighting.light_tex, self.lighting.packed)
            self.last_raycast_ms = 0.0

        # Smoke + fire overlays
        light_mod = self.lighting.light_map if self.show_lighting else None
        if self.show_smoke:
            self.smoke_overlay.update(gmap.smoke, light_modulation=light_mod)
        if self.show_fire:
            self.fire_overlay.update(gmap.fire)

        self.lighting.set_use_normal(self.show_normal_map)
        self.last_frame_ms = (time.perf_counter() - t_start) * 1000

    # ---- frame lifecycle ------------------------------------------------

    def begin_frame(self) -> None:
        core.begin_frame(clear_color=(0, 0, 0, 255))

    def end_frame(self) -> None:
        core.end_frame()

    def should_close(self) -> bool:
        return core.should_close()

    # ---- world-space compose phase --------------------------------------

    def compose_world(self, units_marines: Sequence = (),
                      units_zombies: Sequence = (),
                      orders_per_unit: Optional[dict] = None) -> None:
        """Draw every world-space layer into the world RT.

        Order: lit ship (diffuse + normal + light), smoke, fire, units,
        waypoints, grid. Each is drawn at world-pixel coordinates inside
        the RT — no camera math; the RT IS the world.
        """
        self.world.begin(clear_color=(0, 0, 0, 255))

        # 1. Lit ship — covers the entire world RT
        if self.textures.diffuse:
            self.lighting.draw_lit_world(
                self.textures.diffuse,
                self.textures.normal,
                world_px_w=self.world.world_px_w,
                world_px_h=self.world.world_px_h,
            )

        # 2. Smoke + fire overlays — stretched to world RT bounds
        if self.show_smoke:
            self._draw_overlay_to_world(self.smoke_overlay.tex)
        if self.show_fire:
            rl.begin_blend_mode(rl.BlendMode.BLEND_ADDITIVE)
            self._draw_overlay_to_world(self.fire_overlay.tex)
            rl.end_blend_mode()

        # 3. Units, waypoints, grid — drawn in world-pixel space
        if orders_per_unit:
            self._draw_orders_world(orders_per_unit)
        self._draw_units_world(units_marines, units_zombies)
        if self.show_grid:
            self._draw_grid_world()

        self.world.end()

    def _draw_overlay_to_world(self, field_tex: rl.Texture) -> None:
        """Stretch a physics-resolution texture across the full world RT."""
        src = rl.Rectangle(0, 0, float(field_tex.width), float(field_tex.height))
        dst = rl.Rectangle(0, 0, float(self.world.world_px_w),
                           float(self.world.world_px_h))
        rl.draw_texture_pro(field_tex, src, dst, rl.Vector2(0, 0), 0.0, rl.WHITE)

    def _draw_units_world(self, marines: Sequence, zombies: Sequence) -> None:
        wpt = self.world.world_px_per_tile
        for m in marines:
            if not getattr(m, "alive", True):
                continue
            draw_unit(m.fx, m.fy, wpt, (60, 180, 60, 255),
                      label=getattr(m, "name", ""))
        for z in zombies:
            if not getattr(z, "alive", True):
                continue
            draw_unit(z.fx, z.fy, wpt, (200, 50, 50, 255))

    def _draw_orders_world(self, orders_per_unit: dict) -> None:
        wpt = self.world.world_px_per_tile
        for waypoints in orders_per_unit.values():
            if len(waypoints) < 2:
                continue
            for a, b in zip(waypoints, waypoints[1:]):
                draw_waypoint_line(a, b, wpt)

    def _draw_grid_world(self) -> None:
        wpt = self.world.world_px_per_tile
        draw_grid(self.cfg.grid_w, self.cfg.grid_h, wpt, step=3)

    # ---- final blit -----------------------------------------------------

    def blit_world_to_screen(self) -> None:
        """Single DrawTexturePro from world RT to the map area of the screen.
        Camera transform happens here and nowhere else.

        No scissor — the destination rectangle == the map area, so there is
        nothing to clip. Scissor would only be needed if we drew into the
        panel by accident (which we don't).
        """
        self.world.blit_to_screen(
            self.camera,
            self.camera.viewport_screen_x, self.camera.viewport_screen_y,
            self.cfg.map_px_w, self.cfg.map_px_h,
        )

    # ---- panel ----------------------------------------------------------

    def draw_panel(self, state) -> None:
        cfg = self.cfg
        panel_x = cfg.map_px_w
        draw_panel_background(panel_x, 0, cfg.panel_px_w, cfg.map_px_h)
        x = panel_x + 12
        y = 12
        draw_text(self.level.name, x, y, 20)
        y += 28
        draw_text(f"{cfg.grid_w} x {cfg.grid_h} tiles", x, y, 14)
        y += 22
        draw_text(f"Camera: ({self.camera.pos_tile_x:.1f}, "
                  f"{self.camera.pos_tile_y:.1f})", x, y, 13)
        y += 18
        draw_text(f"Zoom:   {self.camera.zoom_px_per_tile:.1f} spx/tile", x, y, 13)
        y += 22
        draw_text(f"FPS: {rl.get_fps()}", x, y, 14)
        y += 18
        draw_text(f"Raycast: {self.last_raycast_ms:.1f} ms", x, y, 14)
        y += 18
        draw_text(f"Frame:   {self.last_frame_ms:.1f} ms", x, y, 14)
        y += 28
        draw_text("Toggles:", x, y, 14, color=(180, 200, 255, 255))
        y += 20
        for label, on in [
            ("F1 grid",        self.show_grid),
            ("F2 smoke",       self.show_smoke),
            ("F3 fire",        self.show_fire),
            ("F4 light",       self.show_lighting),
            ("F5 normal map",  self.show_normal_map),
            ("B  bilinear",    self.lighting.bilinear),
            ("G  sRGB",        self.srgb_decode),
            ("H  flip-Y norm", self.normal_y_flipped),
        ]:
            color = (180, 255, 180, 255) if on else (140, 140, 140, 255)
            draw_text(label, x, y, 13, color=color)
            y += 16

    # ---- input ----------------------------------------------------------

    def poll_toggles(self) -> None:
        if rl.is_key_pressed(rl.KeyboardKey.KEY_F1):
            self.show_grid = not self.show_grid
        if rl.is_key_pressed(rl.KeyboardKey.KEY_F2):
            self.show_smoke = not self.show_smoke
        if rl.is_key_pressed(rl.KeyboardKey.KEY_F3):
            self.show_fire = not self.show_fire
        if rl.is_key_pressed(rl.KeyboardKey.KEY_F4):
            self.show_lighting = not self.show_lighting
        if rl.is_key_pressed(rl.KeyboardKey.KEY_F5):
            self.show_normal_map = not self.show_normal_map
        if rl.is_key_pressed(rl.KeyboardKey.KEY_B):
            self.lighting.toggle_bilinear()
        if rl.is_key_pressed(rl.KeyboardKey.KEY_H):
            self.normal_y_flipped = not self.normal_y_flipped
            self.lighting.set_normal_y_sign(-1.0 if self.normal_y_flipped else 1.0)
        if rl.is_key_pressed(rl.KeyboardKey.KEY_G):
            self.srgb_decode = not self.srgb_decode
            self.lighting.set_srgb_decode(self.srgb_decode)

    def update_camera(self, dt: float, pan_speed_tiles_per_s: float = 30.0) -> None:
        K = rl.KeyboardKey
        dx = dy = 0.0
        if rl.is_key_down(K.KEY_A) or rl.is_key_down(K.KEY_LEFT):  dx -= 1
        if rl.is_key_down(K.KEY_D) or rl.is_key_down(K.KEY_RIGHT): dx += 1
        if rl.is_key_down(K.KEY_W) or rl.is_key_down(K.KEY_UP):    dy -= 1
        if rl.is_key_down(K.KEY_S) or rl.is_key_down(K.KEY_DOWN):  dy += 1
        if dx == 0 and dy == 0:
            return
        speed = pan_speed_tiles_per_s
        if rl.is_key_down(K.KEY_LEFT_SHIFT) or rl.is_key_down(K.KEY_RIGHT_SHIFT):
            speed *= 3
        self.camera.pan(dx * speed * dt, dy * speed * dt)

    # ---- coordinate conversions -----------------------------------------

    def mouse_to_tile(self) -> Optional[tuple]:
        """Mouse screen position -> integer world tile. None if mouse is
        outside the camera's viewport. Uses the camera's anchor so multi-
        camera (security cam, split screen) will route clicks correctly when
        we have multiple cameras."""
        mx = rl.get_mouse_x()
        my = rl.get_mouse_y()
        if not self.camera.contains_screen_px(mx, my):
            return None
        tx, ty = self.camera.screen_px_to_world_tile(mx, my)
        return int(tx), int(ty)

    # ---- shutdown -------------------------------------------------------

    def shutdown(self) -> None:
        self.textures.unload_all()
        rl.unload_shader(self.lighting.shader)
        rl.unload_texture(self.lighting.light_tex)
        rl.unload_texture(self.smoke_overlay.tex)
        rl.unload_texture(self.fire_overlay.tex)
        self.world.unload()
        core.shutdown()


__all__ = ["GameRenderer", "RenderConfig"]
