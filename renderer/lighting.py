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

    def compute_light_field(self, sources: List, gas: np.ndarray,
                            gas_absorption: np.ndarray, gas_scatter: np.ndarray,
                            light_atten: np.ndarray,
                            heat: Optional[np.ndarray] = None,
                            smoke_glow: Optional[np.ndarray] = None,
                            heat_atten: Optional[np.ndarray] = None) -> None:
        """Cast all sources, accumulate intensity + direction, normalize, pack.

        `gas` is the multi-gas density field, shape (n_gases, h, w) f32 — pass
        `gmap.gas`. `gas_absorption` / `gas_scatter` are the per-gas per-channel
        tables (n_gases, 3) f32 from `GasTable` (`gmap.gases.absorption` /
        `.scatter_albedo`). The march sums the two decoupled optical budgets
        (Beer-Lambert transmission + additive scatter/glow) density-weighted
        across all gases (engine/05 §6.2), so coexisting gases mix automatically
        and each gas tints the beam by its own colour.

        `light_atten` is the per-channel attenuation field the march reads,
        shape (h, w, 3) f32 — pass `gmap.dyn_light_atten` (the live dynamic
        field = static material attenuation MAX'd with stamped-unit opacity,
        rebuilt each tick in `stamp_units`). Occlusion is per-channel (ch.03
        §the march): opaque tiles ([1,1,1]) kill the ray exactly like the old
        wall hard-stop, glass transmits dimmed, an unequal triple tints the
        surviving light. Units stamped into this field restore their shadows
        (default opacity [1,1,1]); smoke remains the separate live input passed
        as `smoke`. (Folding smoke/water into the dynamic field is a later
        slice.)

        `heat` (Q16.16 int32, (h,w)) and `smoke_glow` (f32 RGB, (h,w,3)) are the
        Slice-4 march outputs — pass `gmap.heat` / `gmap.smoke_glow`. They are
        accumulators: cleared here before the frame's sources, then written
        IN-PLACE by the C++ march. `heat` is the sim-affecting deposit (nothing
        reads it yet, ch.04); `smoke_glow` is the render-only god-ray glow that
        supersedes the old surface-tint light_modulation. Both may be None
        (the cast skips that deposit) — kept optional during the renderer-owns-
        the-cast phase (the cast moves into the sim in S5).

        `heat_atten` (f32 (h,w)) is the per-tile heat-ray attenuation field
        (engine/06 §1), the heat analogue of `light_atten` — pass
        `gmap.heat_atten`. It attenuates the march's independent heat channel
        exactly as `light_atten` attenuates the RGB channels, so heat and light
        occlusion diverge for materials transparent to one but not the other.
        May be None (heat is not attenuated — the pre-S6 behaviour).
        """
        self.light_rgb.fill(0)
        self.light_dx.fill(0)
        self.light_dy.fill(0)
        # Zero the per-frame render accumulator IN-PLACE (never reassign — a C++
        # view of the buffer must stay valid). smoke_glow re-accumulates every
        # frame from scratch.
        #
        # NOTE: `heat` is NOT cleared here. It is the sim-owned per-tick deposit
        # and is now cleared at END OF TICK in PhysicsRunner.step (engine/06
        # §1.3) — AFTER the heat -> temperature conversion has read it. Clearing
        # it here (before the cast) used to wipe the deposit before any consumer
        # existed; that clear has moved to the sim so conversion reads it first.
        # The ray march below deposits into `heat` with a SATURATING add, so the
        # sim's end-of-tick clear is what keeps it a per-tick (not cross-tick)
        # buffer.
        if smoke_glow is not None:
            smoke_glow.fill(0)

        for src in sources:
            self.raycaster.cast_source_directional(
                src, self.light_rgb, self.light_dx, self.light_dy,
                gas, gas_absorption, gas_scatter, light_atten,
                heat, smoke_glow, heat_atten,
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
        #   Texture B = smoke_glow (RGB) + light_dir.y (A)
        # smoke_glow is the god-ray glow deposited by the march (ch.03 C16):
        # the light the smoke absorbed, per channel. Drawn additively as the
        # volumetric shaft (ch.05 §God-rays). When no smoke_glow buffer is
        # passed it stays zero (no glow), matching the pre-slice behaviour.
        self.packed_a[..., 0:3] = self.light_rgb
        self.packed_a[..., 3]   = self.light_dx
        if smoke_glow is not None:
            self.packed_b[..., 0:3] = smoke_glow
        else:
            self.packed_b[..., 0:3] = 0.0
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
