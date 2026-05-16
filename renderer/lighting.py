"""Lighting renderer: computes the directional light field via the C++
raycaster, packs it into a small RGBA texture, and draws the diffuse
ship lit by the lighting shader.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, List

import numpy as np
import pyray as rl

from . import core


SHADERS_DIR = Path(__file__).resolve().parent.parent / "shaders"


class LightingPass:
    """Owns the lighting shader and the dynamic light-field texture."""

    def __init__(self, raycaster, grid_h: int, grid_w: int):
        self.raycaster = raycaster
        self.h = grid_h
        self.w = grid_w

        # CPU-side scratch buffers for the raycaster
        self.light_map = np.zeros((grid_h, grid_w), dtype=np.float32)
        self.light_dx  = np.zeros((grid_h, grid_w), dtype=np.float32)
        self.light_dy  = np.zeros((grid_h, grid_w), dtype=np.float32)
        # Packed RGBA8: R=intensity, G=dx (0.5-centered), B=dy (0.5-centered)
        self.packed = np.zeros((grid_h, grid_w, 4), dtype=np.uint8)

        # GPU resources
        self.light_tex = core.create_dynamic_rgba_texture(grid_w, grid_h)
        # Toggle bilinear vs nearest on the light texture
        self.bilinear = True

        self.shader = core.load_shader_with_fallback(
            str(SHADERS_DIR / "lighting.vs"),
            str(SHADERS_DIR / "lighting.fs"),
        )
        # Look up uniform locations once. Warn (but continue) on any -1.
        self._loc_normal_tex      = self._lookup("u_normal")
        self._loc_light_tex       = self._lookup("u_light")
        self._loc_ambient         = self._lookup("u_ambient")
        self._loc_normal_strength = self._lookup("u_normal_strength")
        self._loc_use_normal      = self._lookup("u_use_normal")
        self._loc_normal_y_sign   = self._lookup("u_normal_y_sign")
        self._loc_srgb_decode     = self._lookup("u_srgb_decode")

        # Default uniforms
        self.set_ambient((0.18, 0.18, 0.22))
        self.set_normal_strength(1.0)
        self.set_normal_y_sign(1.0)   # OpenGL convention; flip to -1 if needed
        self.set_srgb_decode(True)    # PNG diffuse art is sRGB

    # ---- uniform setters -----------------------------------------------

    def _lookup(self, name: str) -> int:
        loc = rl.get_shader_location(self.shader, name)
        if loc == -1:
            print(f"[lighting] WARN: shader uniform '{name}' not found (loc=-1)")
        return loc

    def set_ambient(self, rgb):
        val = rl.ffi.new("float[3]", [float(rgb[0]), float(rgb[1]), float(rgb[2])])
        rl.set_shader_value(self.shader, self._loc_ambient, val,
                            rl.ShaderUniformDataType.SHADER_UNIFORM_VEC3)

    def set_normal_strength(self, s: float):
        val = rl.ffi.new("float[1]", [float(s)])
        rl.set_shader_value(self.shader, self._loc_normal_strength, val,
                            rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT)

    def set_use_normal(self, on: bool):
        val = rl.ffi.new("int[1]", [1 if on else 0])
        rl.set_shader_value(self.shader, self._loc_use_normal, val,
                            rl.ShaderUniformDataType.SHADER_UNIFORM_INT)

    def set_normal_y_sign(self, sign: float):
        """Set +1 for OpenGL convention (Y up), -1 for DirectX (Y down)."""
        val = rl.ffi.new("float[1]", [float(sign)])
        rl.set_shader_value(self.shader, self._loc_normal_y_sign, val,
                            rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT)

    def set_srgb_decode(self, on: bool):
        """When True, treat the diffuse texture as sRGB-encoded and do lighting
        math in linear space, re-encoding on output."""
        val = rl.ffi.new("int[1]", [1 if on else 0])
        rl.set_shader_value(self.shader, self._loc_srgb_decode, val,
                            rl.ShaderUniformDataType.SHADER_UNIFORM_INT)

    def toggle_bilinear(self):
        self.bilinear = not self.bilinear
        filt = (rl.TextureFilter.TEXTURE_FILTER_BILINEAR
                if self.bilinear else rl.TextureFilter.TEXTURE_FILTER_POINT)
        rl.set_texture_filter(self.light_tex, filt)

    # ---- light field computation ---------------------------------------

    def compute_light_field(self, sources: List, smoke: np.ndarray, is_wall: np.ndarray) -> None:
        """Cast all sources, accumulate intensity + direction, normalize, pack."""
        self.light_map.fill(0)
        self.light_dx.fill(0)
        self.light_dy.fill(0)

        for src in sources:
            self.raycaster.cast_source_directional(
                src, self.light_map, self.light_dx, self.light_dy,
                smoke, is_wall,
            )
        # Normalize direction to unit vectors (vector-magnitude normalization).
        # See expert review notes in docs/patch_level_pipeline_v1.md.
        type(self.raycaster).normalize_directions(self.light_dx, self.light_dy)

        # Pack into RGBA8:
        #   R = intensity clamped 0..1
        #   G = (dx + 1) / 2  (0.5 = no direction)
        #   B = (dy + 1) / 2
        #   A = 255 (unused, reserved)
        np.clip(self.light_map, 0.0, 1.0, out=self.light_map)
        self.packed[..., 0] = (self.light_map * 255).astype(np.uint8)
        self.packed[..., 1] = ((self.light_dx * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        self.packed[..., 2] = ((self.light_dy * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        self.packed[..., 3] = 255

        core.update_rgba_texture(self.light_tex, self.packed)

    # ---- drawing --------------------------------------------------------

    def draw_lit_world(self, diffuse: rl.Texture, normal: Optional[rl.Texture],
                       world_px_w: int, world_px_h: int) -> None:
        """Draw the lit diffuse over the full world render target.

        Caller must already be inside BeginTextureMode(world_rt). The diffuse
        covers (0, 0) to (world_px_w, world_px_h), so fragTexCoord runs 0..1
        over the world — light field UV matches naturally, no camera math.

        Sampler bindings are issued INSIDE BeginShaderMode so they apply to
        the active shader. Calling set_shader_value_texture before
        BeginShaderMode can target the wrong shader in some raylib versions
        (see research note Gotcha 3).
        """
        rl.begin_shader_mode(self.shader)
        if normal is not None:
            rl.set_shader_value_texture(self.shader, self._loc_normal_tex, normal)
        rl.set_shader_value_texture(self.shader, self._loc_light_tex, self.light_tex)

        src = rl.Rectangle(0, 0, float(diffuse.width), float(diffuse.height))
        dst = rl.Rectangle(0, 0, float(world_px_w), float(world_px_h))
        rl.draw_texture_pro(diffuse, src, dst, rl.Vector2(0, 0), 0.0, rl.WHITE)
        rl.end_shader_mode()


__all__ = ["LightingPass"]
