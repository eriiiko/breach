"""Lighting renderer: computes the RGB directional light field via the C++
raycaster, packs it into two small RGBA16F textures (ch.05), and draws the
diffuse ship lit by the lighting shader.

Texture A = light_rgb (RGB) + light_dir.x (A, signed).
Texture B = smoke_glow (RGB, reserved/zero this slice) + light_dir.y (A, signed).
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
        # RGB light accumulator (f32), shape (h, w, 3) — interleaved per ch.03.
        self.light_rgb = np.zeros((grid_h, grid_w, 3), dtype=np.float32)
        self.light_dx  = np.zeros((grid_h, grid_w), dtype=np.float32)
        self.light_dy  = np.zeros((grid_h, grid_w), dtype=np.float32)
        # Legacy scalar light field, derived from light_rgb (max over channels)
        # for the render-side unit/smoke tinting consumers that still read it.
        self.light_map = np.zeros((grid_h, grid_w), dtype=np.float32)
        # Packed RGBA16F render textures (ch.05):
        #   Texture A = light_rgb (RGB) + light_dir.x (A, signed)
        #   Texture B = smoke_glow (RGB, reserved/zero this slice) + light_dir.y (A, signed)
        self.packed_a = np.zeros((grid_h, grid_w, 4), dtype=np.float16)
        self.packed_b = np.zeros((grid_h, grid_w, 4), dtype=np.float16)

        # GPU resources
        self.light_tex_a = core.create_dynamic_rgba16f_texture(grid_w, grid_h)
        self.light_tex_b = core.create_dynamic_rgba16f_texture(grid_w, grid_h)
        # Vacuum mask texture (R=255 where vacuum, R=0 elsewhere). Built once
        # at level load via set_vacuum_mask; used in the shader to discard
        # vacuum pixels so the screen-space background can show through.
        self.vacuum_tex = core.create_dynamic_rgba_texture(grid_w, grid_h)
        rl.set_texture_filter(self.vacuum_tex,
                              rl.TextureFilter.TEXTURE_FILTER_POINT)
        # Toggle bilinear vs nearest on the light texture
        self.bilinear = True

        self.shader = core.load_shader_with_fallback(
            str(SHADERS_DIR / "lighting.vs"),
            str(SHADERS_DIR / "lighting.fs"),
        )
        # Look up uniform locations once. Warn (but continue) on any -1.
        self._loc_normal_tex      = self._lookup("u_normal")
        self._loc_light_tex_a     = self._lookup("u_light_a")
        self._loc_light_tex_b     = self._lookup("u_light_b")
        self._loc_vacuum_tex      = self._lookup("u_vacuum")
        self._loc_ambient         = self._lookup("u_ambient")
        self._loc_normal_strength = self._lookup("u_normal_strength")
        self._loc_use_normal      = self._lookup("u_use_normal")
        self._loc_normal_y_sign   = self._lookup("u_normal_y_sign")
        self._loc_srgb_decode     = self._lookup("u_srgb_decode")
        self._loc_light_z         = self._lookup("u_light_z")

        # Cached state (for HUD display + bound checks)
        self.light_z = 0.5            # default — overhead lamp feel

        # Default uniforms
        self.set_ambient((0.18, 0.18, 0.22))
        self.set_normal_strength(1.0)
        self.set_normal_y_sign(1.0)   # OpenGL convention; flip to -1 if needed
        self.set_srgb_decode(True)    # PNG diffuse art is sRGB
        self.set_light_z(self.light_z)

    # ---- uniform setters -----------------------------------------------

    def _lookup(self, name: str) -> int:
        loc = rl.get_shader_location(self.shader, name)
        if loc == -1:
            print(f"[lighting] WARN: shader uniform '{name}' not found (loc=-1)")
        return loc

    def set_ambient(self, rgb):
        # Cache as a Python tuple so non-shader consumers (e.g. unit sprite
        # tinting in game_renderer._draw_units_world) can read the same
        # ambient value the ship is lit by. Single source of truth.
        self.ambient = (float(rgb[0]), float(rgb[1]), float(rgb[2]))
        val = rl.ffi.new("float[3]", list(self.ambient))
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

    def set_vacuum_mask(self, is_vacuum: np.ndarray) -> None:
        """Upload the vacuum mask once at level load. (H, W) bool array.
        Vacuum tiles will be discarded by the shader."""
        packed = np.zeros((is_vacuum.shape[0], is_vacuum.shape[1], 4),
                          dtype=np.uint8)
        packed[is_vacuum, 0] = 255
        packed[..., 3] = 255
        core.update_rgba_texture(self.vacuum_tex, packed)

    def set_light_z(self, z: float):
        """0.0 = light skims along the floor (high relief).
        0.5 = standing-height / overhead lamp feel.
        1.0 = light from directly above (flat shading)."""
        z = max(0.0, min(1.5, float(z)))
        self.light_z = z
        val = rl.ffi.new("float[1]", [z])
        rl.set_shader_value(self.shader, self._loc_light_z, val,
                            rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT)

    def toggle_bilinear(self):
        self.bilinear = not self.bilinear
        filt = (rl.TextureFilter.TEXTURE_FILTER_BILINEAR
                if self.bilinear else rl.TextureFilter.TEXTURE_FILTER_POINT)
        rl.set_texture_filter(self.light_tex_a, filt)
        rl.set_texture_filter(self.light_tex_b, filt)

    # ---- light field computation ---------------------------------------

    def compute_light_field(self, sources: List, smoke: np.ndarray,
                            occluders: np.ndarray) -> None:
        """Cast all sources, accumulate intensity + direction, normalize, pack.

        `occluders` is a bool mask that blocks rays. Pass `gmap.obstacles`
        (walls + stamped units) so units cast shadows, OR pass `gmap.is_wall`
        if you only want static geometry to occlude.
        """
        self.light_rgb.fill(0)
        self.light_dx.fill(0)
        self.light_dy.fill(0)

        for src in sources:
            self.raycaster.cast_source_directional(
                src, self.light_rgb, self.light_dx, self.light_dy,
                smoke, occluders,
            )
        # Normalize direction to unit vectors (vector-magnitude normalization).
        # See expert review notes in docs/patch_level_pipeline_v1.md.
        type(self.raycaster).normalize_directions(self.light_dx, self.light_dy)

        # Legacy scalar field for render-side unit/smoke tinting: max over the
        # RGB channels (a brightness proxy). Render-only; not in the sim.
        self.light_map[:] = self.light_rgb.max(axis=2)

        # Pack into two RGBA16F textures (ch.05). 16F stores HDR RGB and
        # SIGNED light_dir directly (no 0.5-centered encode):
        #   Texture A = light_rgb (RGB) + light_dir.x (A)
        #   Texture B = smoke_glow (RGB, zero this slice) + light_dir.y (A)
        self.packed_a[..., 0:3] = self.light_rgb
        self.packed_a[..., 3]   = self.light_dx
        self.packed_b[..., 0:3] = 0.0          # smoke_glow reserved (later slice)
        self.packed_b[..., 3]   = self.light_dy

        core.update_rgba16f_texture(self.light_tex_a, self.packed_a)
        core.update_rgba16f_texture(self.light_tex_b, self.packed_b)

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
        rl.set_shader_value_texture(self.shader, self._loc_light_tex_a, self.light_tex_a)
        rl.set_shader_value_texture(self.shader, self._loc_light_tex_b, self.light_tex_b)
        rl.set_shader_value_texture(self.shader, self._loc_vacuum_tex, self.vacuum_tex)

        src = rl.Rectangle(0, 0, float(diffuse.width), float(diffuse.height))
        dst = rl.Rectangle(0, 0, float(world_px_w), float(world_px_h))
        rl.draw_texture_pro(diffuse, src, dst, rl.Vector2(0, 0), 0.0, rl.WHITE)
        rl.end_shader_mode()


__all__ = ["LightingPass"]
