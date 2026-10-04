"""Lighting renderer: packs the RGB directional light field into two small
RGBA16F textures (ch.05) and draws the diffuse ship lit by the lighting shader.

Texture A = light_rgb (RGB) + light_dir.x (A, signed).
Texture B = glow (RGB) + light_dir.y (A, signed).

RAY-ENGINE-V2 P6b/P6c (design v3 §4.3, §7.1): LightingPass is a CONSUMER, and
since P6c nothing else. It uploads the one accessor's field
(``simulation.light_field.read_light`` -- the radiation sweep's light channels)
into the two textures once per sim tick (:meth:`LightingPass.consume_light_view`,
packed by the pure :func:`pack_light_view`); the shader, the 3D units
(renderer/lit3d.py) and the water pass sample them per frame. It never calls a
producer. (The old render march that used to fill the same textures -- the C++
Raycaster behind P6b's F11 toggle -- was deleted at P6c.) The flat term is the
small FLOOR dial (``[render.lighting] floor``; design §2.7: a separate dial,
never a substitute for the sky).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import pyray as rl

from . import core
from config import CFG


SHADERS_DIR = Path(__file__).resolve().parent.parent / "shaders"


def art_src_and_uv_rect(art_align, grid_w: int, grid_h: int,
                        art_w: float, art_h: float):
    """Pure [art.align] draw math (level format v2 §1.3) — unit-testable.

    ``art_align`` is ``None`` (legacy full-stretch draw) or the 4-tuple
    ``(off_x, off_y, ppt_x, ppt_y)`` stored by :meth:`LightingPass.set_art_align`.
    Returns ``(src_rect, uv_rect)`` as plain ``(x, y, w, h)`` tuples:
    ``src_rect`` is the art-pixel region spanning the grid (offset_px +
    per-axis px_per_tile * grid), ``uv_rect`` the same rect normalized by the
    art dimensions per axis — exactly what is pushed as ``u_art_uv_rect``.
    """
    if art_align is None:
        return (0.0, 0.0, float(art_w), float(art_h)), (0.0, 0.0, 1.0, 1.0)
    off_x, off_y, ppt_x, ppt_y = art_align
    src = (off_x, off_y, ppt_x * float(grid_w), ppt_y * float(grid_h))
    aw, ah = float(art_w), float(art_h)
    return src, (src[0] / aw, src[1] / ah, src[2] / aw, src[3] / ah)


def pack_light_view(view, gain: float, light_rgb: np.ndarray, light_dx: np.ndarray,
                    light_dy: np.ndarray, light_map: np.ndarray,
                    packed_a: np.ndarray, packed_b: np.ndarray,
                    glow_rgb: Optional[np.ndarray] = None) -> None:
    """THE NEW PATH'S PACK (P6b), pure numpy, headless-testable: the accessor's
    :class:`simulation.light_field.LightView` -> the texture contract the ship
    shader, the 3D units and the water pass already read.

      light_rgb = view.rgb * gain          the calibrated field (render dial
                                           ``[render.lighting] sweep_gain``;
                                           light units -> the old march's scale)
      light_dir = -view.flux_dir           toward the light, the contract's
                                           convention (the sweep's flux points
                                           the way light TRAVELS)
      light_map = max over the channels    the scalar the sprite tint reads
      packed_a  = (light_rgb, dir.x) ; packed_b = (glow * gain, dir.y)   16F

    Every output is a MONOTONE non-decreasing function of the integer field
    per channel (a positive scale, then float rounding, which is monotone) --
    design §8.5, held by tests/test_light_field.py. A post-process that mixes
    cells or channels here would break it."""
    np.multiply(view.rgb, np.float32(gain), out=light_rgb)
    np.negative(view.flux_dir[..., 0], out=light_dx)
    np.negative(view.flux_dir[..., 1], out=light_dy)
    np.max(light_rgb, axis=2, out=light_map)
    packed_a[..., 0:3] = light_rgb
    packed_a[..., 3] = light_dx
    glow = view.glow * np.float32(gain)
    packed_b[..., 0:3] = glow
    packed_b[..., 3] = light_dy
    if glow_rgb is not None:
        glow_rgb[...] = glow


class LightingPass:
    """Owns the lighting shader and the dynamic light-field texture."""

    def __init__(self, grid_h: int, grid_w: int):
        self.h = grid_h
        self.w = grid_w
        _lcfg = getattr(getattr(CFG, "render", None), "lighting", None)
        # [render.lighting] sweep_gain: light units -> the old march's field
        # scale, a RENDER dial (dequantized at the render read, never written
        # back). Measured 1:1 (report: lamp 0.996, flashlight 1.05, crate 0.83).
        self.sweep_gain = float(getattr(_lcfg, "sweep_gain", 1.0))
        # [render.lighting] floor: the small flat FLOOR (design §2.7) -- the
        # only flat term since P6c (the old path's flat ambient is gone).
        self.floor = tuple(float(v) for v in getattr(_lcfg, "floor", (0.03, 0.03, 0.04)))
        # the calibrated glow (for the gas-medium pass), (h, w, 3)
        self.glow_rgb = np.zeros((grid_h, grid_w, 3), dtype=np.float32)

        # CPU-side scratch the pack writes (pack_light_view):
        # RGB light field (f32), shape (h, w, 3) — interleaved per ch.03.
        self.light_rgb = np.zeros((grid_h, grid_w, 3), dtype=np.float32)
        self.light_dx  = np.zeros((grid_h, grid_w), dtype=np.float32)
        self.light_dy  = np.zeros((grid_h, grid_w), dtype=np.float32)
        # The renderer's SCALAR light field, the brightest channel of light_rgb
        # (i.e. of the accessor's view), for the sprite tint.
        self.light_map = np.zeros((grid_h, grid_w), dtype=np.float32)
        # Packed RGBA16F render textures (ch.05):
        #   Texture A = light_rgb (RGB) + light_dir.x (A, signed)
        #   Texture B = glow (RGB) + light_dir.y (A, signed)
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
        # [render.lighting] bilinear: bilinear vs nearest on the light
        # texture. Read ONCE here (#59: was a live B-key toggle, which
        # collided with ORDER_EXPLOSIVE's B binding and confounded every look
        # judgement during a session with explosives armed all the time --
        # Erik's ruling 2026-09-29, fewer shortcut keys, a switch is a
        # config.toml setting, edit + restart). Default True = today's
        # shipped state (create_dynamic_rgba16f_texture already creates the
        # two textures bilinear-filtered; only a False here needs to change
        # anything).
        self.bilinear = bool(getattr(_lcfg, "bilinear", True))
        if not self.bilinear:
            point = rl.TextureFilter.TEXTURE_FILTER_POINT
            rl.set_texture_filter(self.light_tex_a, point)
            rl.set_texture_filter(self.light_tex_b, point)

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
        self._loc_light_gain      = self._lookup("u_light_gain")
        self._loc_art_uv_rect     = self._lookup("u_art_uv_rect")

        # Cached state (for HUD display + bound checks)
        self.light_z = 0.5            # default — overhead lamp feel
        # [art.align] transform (level format v2 §1.3): None = legacy
        # stretch-art-to-grid-rect draw (bit-identical to pre-F2 output);
        # (offset_x, offset_y, ppt_x, ppt_y) = explicit alignment (per-axis
        # px_per_tile), consumed by draw_lit_world. Set via set_art_align at
        # level-load time.
        self.art_align = None

        # Default uniforms
        self._push_flat_term()
        self.set_normal_strength(1.0)
        self.set_normal_y_sign(1.0)   # OpenGL convention; flip to -1 if needed
        self.set_srgb_decode(True)    # PNG diffuse art is sRGB
        self.set_light_z(self.light_z)
        # Render exposure: light_rgb is a PHYSICAL field (light units, the
        # sweep's irradiance scaled by [render.lighting] sweep_gain). This gain
        # maps physical power -> display brightness so sources are tuned by
        # physics, exposure by one master dial. Default 1.0 (neutral);
        # game_renderer / the demo override from config.
        # Config-bound 2026-08-25 ([render.lighting] master_gain): the game ran
        # the neutral 1.0 while the demo harness tunes at ~10 — the "too dark
        # everywhere" gap (Erik). Distinct from the blackbody light_gain dial.
        self.light_gain = float(getattr(
            getattr(getattr(CFG, "render", None), "lighting", None),
            "master_gain", 1.0))
        self.set_light_gain(self.light_gain)
        # u_art_uv_rect MUST be initialised: an unset vec4 uniform reads as
        # (0,0,0,0) and the shader's world_uv division would produce NaNs.
        self._set_art_uv_rect((0.0, 0.0, 1.0, 1.0))

    # ---- uniform setters -----------------------------------------------

    def _lookup(self, name: str) -> int:
        loc = rl.get_shader_location(self.shader, name)
        if loc == -1:
            print(f"[lighting] WARN: shader uniform '{name}' not found (loc=-1)")
        return loc

    def set_floor(self, rgb):
        """The small flat FLOOR (design §2.7) -- a render dial, never a
        substitute for the sky: a sealed room reads exactly this."""
        self.floor = (float(rgb[0]), float(rgb[1]), float(rgb[2]))
        self._push_flat_term()

    def _push_flat_term(self):
        # Cache as a Python tuple so non-shader consumers (e.g. unit sprite
        # tinting in game_renderer._draw_units_world, the water pass, the 3D
        # units' LightFieldCtx) read the same flat term the ship is lit by.
        # Single source of truth: `ambient` is the flat term (the floor) --
        # the shader's u_ambient uniform keeps its name.
        self.ambient = self.floor
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

    def set_light_gain(self, gain: float):
        """Render exposure: multiply the physical light field before tone-map.
        Physics tunes power; this tunes look ([render.lighting] master_gain)."""
        gain = max(0.0, float(gain))
        self.light_gain = gain
        val = rl.ffi.new("float[1]", [gain])
        rl.set_shader_value(self.shader, self._loc_light_gain, val,
                            rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT)

    def set_art_align(self, offset_px, px_per_tile) -> None:
        """Bind the level's explicit [art.align] transform (format v2 §1.3).

        Art pixel ``offset_px`` lands on grid (0, 0); ``px_per_tile`` art
        pixels span one tile — a scalar (same both axes) or an ``(x, y)``
        pair (LevelData always carries the pair; the v2 art's proportions
        differ from the tilemap per axis). draw_lit_world derives the src
        rect + the shader's u_art_uv_rect from this each draw (the art
        dimensions are only known from the bound texture). Never called for
        levels without an explicit [art.align] — those keep the legacy
        full-stretch draw.
        """
        if isinstance(px_per_tile, (list, tuple)):
            ppt_x, ppt_y = float(px_per_tile[0]), float(px_per_tile[1])
        else:
            ppt_x = ppt_y = float(px_per_tile)
        self.art_align = (float(offset_px[0]), float(offset_px[1]),
                          ppt_x, ppt_y)

    def _set_art_uv_rect(self, rect) -> None:
        """Upload u_art_uv_rect (art-UV subrect: x, y, w, h)."""
        val = rl.ffi.new("float[4]", [float(v) for v in rect])
        rl.set_shader_value(self.shader, self._loc_art_uv_rect, val,
                            rl.ShaderUniformDataType.SHADER_UNIFORM_VEC4)

    # ---- the light field: the one accessor's, uploaded ---------------------

    def consume_light_view(self, view) -> None:
        """Upload the accessor's field (``light_field.read_light``) into
        light_tex_a / light_tex_b, once per sim tick (the caller watches
        Simulation.light_serial). Packed by :func:`pack_light_view`; the 3D
        units and the water pass read the same two textures."""
        pack_light_view(view, self.sweep_gain, self.light_rgb, self.light_dx,
                        self.light_dy, self.light_map, self.packed_a,
                        self.packed_b, glow_rgb=self.glow_rgb)
        core.update_rgba16f_texture(self.light_tex_a, self.packed_a)
        core.update_rgba16f_texture(self.light_tex_b, self.packed_b)

    def clear_light(self) -> None:
        """Lighting OFF (F4): zero every light buffer and upload the dark
        textures, so no stale field lingers (the flat floor still lights the
        ship)."""
        self.light_rgb.fill(0)
        self.light_map.fill(0)
        self.light_dx.fill(0)
        self.light_dy.fill(0)
        self.packed_a.fill(0)
        self.packed_b.fill(0)
        self.glow_rgb.fill(0)
        core.update_rgba16f_texture(self.light_tex_a, self.packed_a)
        core.update_rgba16f_texture(self.light_tex_b, self.packed_b)

    # ---- drawing --------------------------------------------------------

    def draw_lit_world(self, diffuse: rl.Texture, normal: Optional[rl.Texture],
                       world_px_w: int, world_px_h: int) -> None:
        """Draw the lit diffuse over the full world render target.

        Caller must already be inside BeginTextureMode(world_rt). The drawn
        quad covers (0, 0) to (world_px_w, world_px_h).

        Art -> grid mapping (level format v2 §1.3): without an explicit
        [art.align] (``self.art_align is None``) the src rect is the FULL
        diffuse — the legacy stretch-art-to-grid-rect draw, fragTexCoord runs
        0..1 over the world, u_art_uv_rect stays (0,0,1,1) and the output is
        bit-identical to pre-F2. With an explicit align, the src rect is the
        art region that spans the grid (offset_px + per-axis px_per_tile·grid), so
        fragTexCoord interpolates over that art-UV subrect; the same subrect
        is pushed as u_art_uv_rect so the shader can recover world UV for the
        grid-resolution samplers (light field, vacuum mask). The normal map
        is sampled at fragTexCoord (art space) — it must cover the same image
        extent as the diffuse (any resolution).

        Sampler bindings are issued INSIDE BeginShaderMode so they apply to
        the active shader. Calling set_shader_value_texture before
        BeginShaderMode can target the wrong shader in some raylib versions
        (see research note Gotcha 3).
        """
        src_rect, uv_rect = art_src_and_uv_rect(
            self.art_align, self.w, self.h,
            float(diffuse.width), float(diffuse.height))
        src = rl.Rectangle(*src_rect)

        rl.begin_shader_mode(self.shader)
        if normal is not None:
            rl.set_shader_value_texture(self.shader, self._loc_normal_tex, normal)
        else:
            # No normal map for this level: never let the shader sample an
            # UNBOUND u_normal. It reads whatever the texture unit holds -- in
            # practice art -- as a normal, so a flat grey floor (0.55) decodes
            # to a ~55-degree diagonal tilt and every lamp lights one side and
            # "shadows" the other, and a near-black floor tilts below the
            # horizon and takes no light at all (found in the smoke-light
            # studio, 2026-10-04, #12). Re-asserted each draw: the F5 toggle
            # sets the flag every frame before this.
            self.set_use_normal(False)
        rl.set_shader_value_texture(self.shader, self._loc_light_tex_a, self.light_tex_a)
        rl.set_shader_value_texture(self.shader, self._loc_light_tex_b, self.light_tex_b)
        rl.set_shader_value_texture(self.shader, self._loc_vacuum_tex, self.vacuum_tex)
        self._set_art_uv_rect(uv_rect)

        dst = rl.Rectangle(0, 0, float(world_px_w), float(world_px_h))
        rl.draw_texture_pro(diffuse, src, dst, rl.Vector2(0, 0), 0.0, rl.WHITE)
        rl.end_shader_mode()


__all__ = ["LightingPass", "art_src_and_uv_rect", "pack_light_view"]
