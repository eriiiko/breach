"""GameRenderer: thin orchestrator over core/lighting/overlays.

The game code holds one GameRenderer instance and calls the public methods:
    upload_state(gmap)         — push physics state to GPU
    begin_frame()              — clear backbuffer
    draw_world()               — draw lit ship + smoke/fire
    draw_units(units)          — draw squad + zombies
    draw_orders(orders, phase) — draw planned waypoints, throws
    draw_panel(state)          — draw the right-side UI panel
    end_frame()                — present
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pyray as rl

from . import core
from .lighting import LightingPass
from .overlays import (
    FieldOverlay, FireOverlay,
    draw_unit, draw_waypoint_line, draw_grid, draw_text, draw_panel_background,
)


@dataclass
class RenderConfig:
    map_px_w: int          # pixels for the map area (window minus panel)
    map_px_h: int
    panel_px_w: int        # right-side panel width
    fine_tile_px: float    # pixels per tile in the map area
    grid_w: int            # physics grid width in tiles
    grid_h: int            # physics grid height in tiles
    # Camera offset in tile units. (0,0) means top-left of map is shown.
    camera_x: float = 0.0
    camera_y: float = 0.0


class GameRenderer:
    """Encapsulates pyray drawing. One per game session."""

    def __init__(self, level_data, breach_physics, cfg: RenderConfig):
        """level_data: LevelData from level_loader.load(...)
        breach_physics: the imported breach_physics module
        cfg: RenderConfig with window dimensions
        """
        self.level = level_data
        self.bp = breach_physics
        self.cfg = cfg

        total_w = cfg.map_px_w + cfg.panel_px_w
        total_h = cfg.map_px_h
        core.init_window(total_w, total_h, title=f"Breach — {level_data.name}")

        # Load level textures
        self.textures = core.load_level_textures(level_data)

        # Lighting pass (owns shader + light field texture + raycaster scratch)
        self.raycaster = breach_physics.Raycaster()
        self.raycaster.smoke_absorption = 0.8
        self.lighting = LightingPass(self.raycaster, cfg.grid_h, cfg.grid_w)

        # Field overlays (smoke + fire) at physics resolution
        self.smoke_overlay = FieldOverlay(cfg.grid_h, cfg.grid_w,
                                          tint=(190, 195, 210), max_alpha=180)
        self.fire_overlay = FireOverlay(cfg.grid_h, cfg.grid_w)

        # Toggles
        self.show_grid = False
        self.show_smoke = True
        self.show_fire = True
        self.show_lighting = True
        self.show_normal_map = True

        # Frame timing
        self.last_frame_ms = 0.0
        self.last_raycast_ms = 0.0

        # Default lighting: bright ambient so initial scene is visible (no lights yet)
        self.lighting.set_ambient((0.5, 0.5, 0.55))

    # ---- per-frame inputs ----------------------------------------------

    def upload_state(self, gmap, light_sources: Optional[List] = None) -> None:
        """Push physics state to GPU textures: smoke, fire, light field."""
        import time
        t_start = time.perf_counter()

        # Use the most recently computed light map for smoke modulation.
        # This makes smoke only visible where light reaches — much more atmospheric.
        light_mod = self.lighting.light_map if self.show_lighting else None
        if self.show_smoke:
            self.smoke_overlay.update(gmap.smoke, light_modulation=light_mod)
        if self.show_fire:
            self.fire_overlay.update(gmap.fire)

        if self.show_lighting and light_sources:
            t_ray = time.perf_counter()
            self.lighting.compute_light_field(light_sources, gmap.smoke, gmap.is_wall)
            self.last_raycast_ms = (time.perf_counter() - t_ray) * 1000
        else:
            # Clear the light field so nothing illuminates the ship
            self.lighting.light_map.fill(0)
            self.lighting.light_dx.fill(0)
            self.lighting.light_dy.fill(0)
            self.lighting.packed.fill(0)
            self.lighting.packed[..., 3] = 255
            core.update_rgba_texture(self.lighting.light_tex, self.lighting.packed)
            self.last_raycast_ms = 0.0

        self.lighting.set_use_normal(self.show_normal_map)
        self.last_frame_ms = (time.perf_counter() - t_start) * 1000

    # ---- frame lifecycle ------------------------------------------------

    def begin_frame(self) -> None:
        core.begin_frame(clear_color=(0, 0, 0, 255))

    def end_frame(self) -> None:
        core.end_frame()

    def should_close(self) -> bool:
        return core.should_close()

    # ---- drawing layers -------------------------------------------------

    def draw_world(self) -> None:
        """Lit ship + smoke + fire. Honors camera offset + zoom."""
        cfg = self.cfg
        # Compute the rectangle of world (in tiles) that the map area shows.
        # We render the world at zoom = fine_tile_px (px per tile). The visible
        # tile range is camera..(camera + map_px / fine_tile_px).
        ft = cfg.fine_tile_px
        tiles_x = cfg.map_px_w / ft
        tiles_y = cfg.map_px_h / ft

        # Scissor map area so nothing draws into the panel
        rl.begin_scissor_mode(0, 0, cfg.map_px_w, cfg.map_px_h)

        # Destination is full map area; source rect in the texture
        if self.textures.diffuse:
            # Diffuse spans 0..grid_w in U (across tile coords).
            tex = self.textures.diffuse
            u0 = cfg.camera_x / cfg.grid_w
            v0 = cfg.camera_y / cfg.grid_h
            u1 = (cfg.camera_x + tiles_x) / cfg.grid_w
            v1 = (cfg.camera_y + tiles_y) / cfg.grid_h
            src = rl.Rectangle(u0 * tex.width, v0 * tex.height,
                               (u1 - u0) * tex.width, (v1 - v0) * tex.height)
            dst = rl.Rectangle(0, 0, float(cfg.map_px_w), float(cfg.map_px_h))
            # Use the lighting shader by drawing the diffuse with shader bound
            normal = self.textures.normal
            if normal is not None:
                rl.set_shader_value_texture(self.lighting.shader,
                                            self.lighting._loc_normal_tex, normal)
            rl.set_shader_value_texture(self.lighting.shader,
                                        self.lighting._loc_light_tex,
                                        self.lighting.light_tex)
            rl.begin_shader_mode(self.lighting.shader)
            rl.draw_texture_pro(tex, src, dst, rl.Vector2(0, 0), 0.0, rl.WHITE)
            rl.end_shader_mode()

        # Overlays: smoke + fire, sampled with same camera rect
        def _draw_overlay(field_tex):
            u0 = cfg.camera_x / cfg.grid_w
            v0 = cfg.camera_y / cfg.grid_h
            u1 = (cfg.camera_x + tiles_x) / cfg.grid_w
            v1 = (cfg.camera_y + tiles_y) / cfg.grid_h
            src = rl.Rectangle(u0 * field_tex.width, v0 * field_tex.height,
                               (u1 - u0) * field_tex.width, (v1 - v0) * field_tex.height)
            dst = rl.Rectangle(0, 0, float(cfg.map_px_w), float(cfg.map_px_h))
            rl.draw_texture_pro(field_tex, src, dst, rl.Vector2(0, 0), 0.0, rl.WHITE)

        if self.show_smoke:
            _draw_overlay(self.smoke_overlay.tex)
        if self.show_fire:
            rl.begin_blend_mode(rl.BlendMode.BLEND_ADDITIVE)
            _draw_overlay(self.fire_overlay.tex)
            rl.end_blend_mode()
        if self.show_grid:
            draw_grid(cfg.grid_w, cfg.grid_h, ft, step=3)

        rl.end_scissor_mode()

    def draw_units(self, marines: Sequence, zombies: Sequence) -> None:
        ft = self.cfg.map_px_w / self.cfg.grid_w
        for m in marines:
            if not getattr(m, "alive", True):
                continue
            color = (60, 180, 60, 255)
            draw_unit(m.fx, m.fy, ft, color, label=getattr(m, "name", ""))
        for z in zombies:
            if not getattr(z, "alive", True):
                continue
            color = (200, 50, 50, 255)
            draw_unit(z.fx, z.fy, ft, color)

    def draw_orders(self, orders_per_unit: dict, phase: int) -> None:
        """orders_per_unit: {unit_id: [list of (waypoint_fx, waypoint_fy)]}"""
        ft = self.cfg.map_px_w / self.cfg.grid_w
        for waypoints in orders_per_unit.values():
            if len(waypoints) < 2:
                continue
            for a, b in zip(waypoints, waypoints[1:]):
                draw_waypoint_line(a, b, ft)

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
        draw_text(f"FPS: {rl.get_fps()}", x, y, 14)
        y += 18
        draw_text(f"Raycast: {self.last_raycast_ms:.1f} ms", x, y, 14)
        y += 18
        draw_text(f"Frame:   {self.last_frame_ms:.1f} ms", x, y, 14)
        y += 28
        draw_text("Toggles:", x, y, 14, color=(180, 200, 255, 255))
        y += 20
        for label, on in [
            ("F1 grid",   self.show_grid),
            ("F2 smoke",  self.show_smoke),
            ("F3 fire",   self.show_fire),
            ("F4 light",  self.show_lighting),
            ("F5 normal", self.show_normal_map),
        ]:
            color = (180, 255, 180, 255) if on else (140, 140, 140, 255)
            draw_text(label, x, y, 13, color=color)
            y += 16

    # ---- input -----------------------------------------------------------

    def poll_toggles(self) -> None:
        """Check for F1-F5 toggles. Called once per frame."""
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

    # ---- coordinate conversions -----------------------------------------

    def mouse_to_tile(self) -> Optional[tuple]:
        """Returns (fx, fy) tile coords under mouse (in world space), or None if
        outside map area. Accounts for camera offset and zoom."""
        cfg = self.cfg
        mx = rl.get_mouse_x()
        my = rl.get_mouse_y()
        if mx < 0 or mx >= cfg.map_px_w or my < 0 or my >= cfg.map_px_h:
            return None
        ft = cfg.fine_tile_px
        tx = int(cfg.camera_x + mx / ft)
        ty = int(cfg.camera_y + my / ft)
        return tx, ty

    def update_camera(self, dt: float, pan_speed_tiles_per_s: float = 30.0) -> None:
        """WASD / arrow keys pan the camera. Clamp to map bounds."""
        cfg = self.cfg
        K = rl.KeyboardKey
        dx = dy = 0.0
        if rl.is_key_down(K.KEY_A) or rl.is_key_down(K.KEY_LEFT):  dx -= 1
        if rl.is_key_down(K.KEY_D) or rl.is_key_down(K.KEY_RIGHT): dx += 1
        if rl.is_key_down(K.KEY_W) or rl.is_key_down(K.KEY_UP):    dy -= 1
        if rl.is_key_down(K.KEY_S) or rl.is_key_down(K.KEY_DOWN):  dy += 1
        if dx == 0 and dy == 0:
            return
        # Hold shift to pan faster
        speed = pan_speed_tiles_per_s
        if rl.is_key_down(K.KEY_LEFT_SHIFT) or rl.is_key_down(K.KEY_RIGHT_SHIFT):
            speed *= 3
        cfg.camera_x += dx * speed * dt
        cfg.camera_y += dy * speed * dt
        # Clamp so camera doesn't show past the world bounds
        tiles_x = cfg.map_px_w / cfg.fine_tile_px
        tiles_y = cfg.map_px_h / cfg.fine_tile_px
        cfg.camera_x = max(0.0, min(cfg.grid_w - tiles_x, cfg.camera_x))
        cfg.camera_y = max(0.0, min(cfg.grid_h - tiles_y, cfg.camera_y))

    # ---- shutdown -------------------------------------------------------

    def shutdown(self) -> None:
        self.textures.unload_all()
        # Light texture + overlay textures + shader unload by Raylib at exit
        rl.unload_shader(self.lighting.shader)
        rl.unload_texture(self.lighting.light_tex)
        rl.unload_texture(self.smoke_overlay.tex)
        rl.unload_texture(self.fire_overlay.tex)
        core.shutdown()


__all__ = ["GameRenderer", "RenderConfig"]
