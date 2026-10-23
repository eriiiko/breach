"""WorldComposite — the world's render targets, its ONE tone-map, and the blit.

All world-space drawing (lit ship, smoke, fire, units, particles, debug)
happens *into* world-sized render targets inside BeginTextureMode. The camera
is then a single DrawTexturePro blit to the screen, with a source rectangle
computed from the camera's visible world rectangle.

Two stages, two targets (graphics arc G0 P1, docs/graphics_direction_
2026-10-18.md §6):

  1. SCENE  -- ``scene_rt``, RGBA16F. Every lit layer (the ship, the gas
     medium, the black-body heat, the water, the 3D units and props) writes
     LINEAR, un-tone-mapped HDR here. ``begin()`` binds it. Exposure is
     already in the layers (``u_light_gain``, the one exposure dial).
  2. DISPLAY -- ``rt``, RGBA8. ``resolve()`` draws the scene RT through
     ``shaders/world_resolve.fs`` -- ACES, the gamma encode and the dither,
     the ONLY place the world is tone-mapped -- and leaves this target bound
     for the display-referred layers that follow (debug colormaps, doors,
     waypoints, sprite units, effects, the grid, UI overlays). Those are
     authored in display sRGB and must blend in display space: drawn into
     the HDR scene, a translucent UI line would be tone-mapped too (a 35 %
     cyan line reads near-opaque white after ACES). ``end()`` unbinds.

``blit_to_screen`` reads the display RT, exactly as before G0.

Benefits:
  - Every shader samples fragTexCoord 0..1 over the world — no camera UVs.
  - Adding a new layer is one more draw call inside compose(), not a shader.
  - Screen-space distortion (heat shimmer, G3 F4a; refraction, shockwaves)
    is a TERM of the resolve's distortion block, never another pass.
  - Multiple cameras (security cam, split screen, replay) = re-blit same RT.

Cost: two world-sized RTs. For the playground (100x70 tiles at 48 wpx/tile,
4800x3360) that is ~129 MB (16F) + ~65 MB (8-bit) + their depth buffers, and
one full-world resolve pass per frame.

Y-FLIP: Raylib stores RT contents with origin at bottom-left (OpenGL).
       The screen has origin top-left. Reading an RT back out (the resolve
       into the display RT, the blit to the screen) MUST use a negative
       source height to flip Y back. resolve() and blit_to_screen() handle
       this — every other consumer reads from the RT via this class.
"""
from __future__ import annotations

from pathlib import Path

import pyray as rl

from . import core
from .camera import Camera2D

SHADERS_DIR = Path(__file__).resolve().parent.parent / "shaders"

# Dither amplitude at the resolve, in 8-bit LSBs (1.0 = +-0.5 LSB): enough to
# break the banding of a dark gradient quantised to 8 bits, too little to see.
RESOLVE_DITHER_LSB = 1.0


class WorldComposite:
    """Owns the world-space render targets, the resolve, and the blit."""

    def __init__(self, world_tile_w: int, world_tile_h: int,
                 world_px_per_tile: float):
        self.world_tile_w = world_tile_w
        self.world_tile_h = world_tile_h
        self.world_px_per_tile = float(world_px_per_tile)
        self.world_px_w = int(round(world_tile_w * world_px_per_tile))
        self.world_px_h = int(round(world_tile_h * world_px_per_tile))

        # SCENE: linear HDR, the target every lit layer draws into.
        self.scene_rt = core.load_render_texture_rgba16f(self.world_px_w,
                                                         self.world_px_h)
        # Point-sampled: the resolve is a 1:1 texel copy (no filtering wanted).
        rl.set_texture_filter(self.scene_rt.texture,
                              rl.TextureFilter.TEXTURE_FILTER_POINT)
        rl.set_texture_wrap(self.scene_rt.texture,
                            rl.TextureWrap.TEXTURE_WRAP_CLAMP)
        # DISPLAY: tone-mapped sRGB, the target the camera blits from.
        self.rt = rl.load_render_texture(self.world_px_w, self.world_px_h)
        # Smooth zoom when blitting to screen; turn to POINT for pixel-art crisp.
        rl.set_texture_filter(self.rt.texture,
                              rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
        # Clamp so the final blit doesn't sample garbage past world bounds.
        rl.set_texture_wrap(self.rt.texture,
                            rl.TextureWrap.TEXTURE_WRAP_CLAMP)

        self.resolve_shader = core.load_shader_with_fallback(
            str(SHADERS_DIR / "lighting.vs"),      # the passthrough VS
            str(SHADERS_DIR / "world_resolve.fs"))
        self._loc_encode = rl.get_shader_location(self.resolve_shader,
                                                  "u_encode_srgb")
        self._loc_dither = rl.get_shader_location(self.resolve_shader,
                                                  "u_dither_amp")
        # The DISTORTION BLOCK's inputs (graphics G3 F4a: heat shimmer).
        self._dloc = {}
        self._dcache = {}
        self._stage = None         # None | "scene" | "display"

    # ---- Compose-phase lifecycle ----------------------------------------

    def begin(self, clear_color=(0, 0, 0, 255)) -> None:
        """Bind the SCENE RT (linear HDR) as the draw target. Clear to color."""
        rl.begin_texture_mode(self.scene_rt)
        rl.clear_background(rl.Color(*clear_color))
        self._stage = "scene"

    def resolve(self, encode_srgb: bool = True, shimmer=None) -> None:
        """The ONE tone-map: scene RT -> [distortion block] -> ACES ->
        gamma encode -> dither -> display RT, which is left BOUND for the
        display-referred layers.

        ``encode_srgb`` is the output half of the G toggle (LightingPass's
        sRGB flag): off, the resolve skips the encode exactly as the layers
        used to when it was off.

        ``shimmer`` (graphics G3 F4a) = the frame's
        :class:`renderer.heat_shimmer.HeatShimmer`, the first term of the
        shader's DISTORTION BLOCK (``world_resolve.fs`` ``distortion_px``):
        every screen-space distortion is a term there, read while the
        resolve samples the scene RT -- never a second pass. None (or
        inactive) = the term is off and the read is the plain point fetch."""
        if self._stage != "scene":
            raise RuntimeError("WorldComposite.resolve() outside begin()")
        rl.end_texture_mode()
        rl.begin_texture_mode(self.rt)
        rl.clear_background(rl.Color(0, 0, 0, 0))
        rl.set_shader_value(self.resolve_shader, self._loc_encode,
                            rl.ffi.new("int[1]", [1 if encode_srgb else 0]),
                            rl.ShaderUniformDataType.SHADER_UNIFORM_INT)
        rl.set_shader_value(self.resolve_shader, self._loc_dither,
                            rl.ffi.new("float[1]", [RESOLVE_DITHER_LSB]),
                            rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT)
        # Premultiplied "over" a transparent-cleared target == a copy of the
        # premultiplied resolve output (alpha included).
        rl.begin_blend_mode(rl.BlendMode.BLEND_ALPHA_PREMULTIPLY)
        rl.begin_shader_mode(self.resolve_shader)
        bound = self._set_distortion(shimmer)
        rl.draw_texture_rec(
            self.scene_rt.texture,
            rl.Rectangle(0, 0, float(self.world_px_w), -float(self.world_px_h)),
            rl.Vector2(0, 0), rl.WHITE)
        rl.end_shader_mode()                 # flushes: drawn now
        if bound is not None:
            bound.unbind()
        rl.end_blend_mode()
        self._stage = "display"

    def _set_distortion(self, shimmer):
        """Set the distortion block's uniforms (only those that changed)
        and bind its samplers; returns what to unbind after the draw."""
        from .heat_shimmer import HeatShimmer, SAMPLER_UNITS
        if not self._dloc:
            self._dloc = {u: rl.get_shader_location(self.resolve_shader, u)
                          for u in HeatShimmer.UNIFORMS}
            for name, unit in SAMPLER_UNITS.items():
                self._dset(name, (int(unit),))
        on = shimmer is not None and shimmer.active(self.world_px_per_tile)
        if not on:
            self._dset("u_shimmer_on", (0,))
            return None
        for name, v in shimmer.uniforms(self.world_px_per_tile).items():
            self._dset(name, v)
        shimmer.bind()
        return shimmer

    def _dset(self, name: str, v: tuple) -> None:
        loc = self._dloc.get(name, -1)
        if loc < 0 or self._dcache.get(name) == v:
            return
        self._dcache[name] = v
        if len(v) == 1 and isinstance(v[0], int):
            rl.set_shader_value(self.resolve_shader, loc,
                                rl.ffi.new("int[1]", [int(v[0])]),
                                rl.ShaderUniformDataType.SHADER_UNIFORM_INT)
            return
        kind = {1: rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT,
                2: rl.ShaderUniformDataType.SHADER_UNIFORM_VEC2,
                4: rl.ShaderUniformDataType.SHADER_UNIFORM_VEC4}[len(v)]
        rl.set_shader_value(self.resolve_shader, loc,
                            rl.ffi.new(f"float[{len(v)}]",
                                       [float(x) for x in v]), kind)

    def end(self) -> None:
        """Unbind. A caller that never resolved still gets a tone-mapped
        display RT (resolve with the default encode)."""
        if self._stage == "scene":
            self.resolve()
        rl.end_texture_mode()
        self._stage = None

    # ---- Blit to screen --------------------------------------------------

    def blit_to_screen(self, camera: Camera2D,
                       dst_x: int, dst_y: int,
                       dst_w: int, dst_h: int) -> None:
        """Blit the camera's visible world rectangle from the RT to the screen.

        Letterboxes when the camera's visible region extends past the world
        bounds (i.e. zoomed too far out). The source rect is clipped to the
        world; the destination rect is shrunk proportionally and centered so
        the camera's visible region "frame" still spans dst_w x dst_h on
        screen — empty space ends up as black bars.

        Y-FLIP: Raylib RTs are Y-up internally. Standard idiom: src.y at
        the BOTTOM edge of the sampled region (in Y-up coords), negative
        source height for the flip on draw.
        """
        x_wpx, y_wpx, w_wpx, h_wpx = camera.visible_world_rect_world_px(
            self.world_px_per_tile
        )

        # Clip the visible region to the world bounds.
        clip_x0 = max(0.0, x_wpx)
        clip_y0 = max(0.0, y_wpx)
        clip_x1 = min(float(self.world_px_w), x_wpx + w_wpx)
        clip_y1 = min(float(self.world_px_h), y_wpx + h_wpx)
        clip_w = max(0.0, clip_x1 - clip_x0)
        clip_h = max(0.0, clip_y1 - clip_y0)

        if clip_w <= 0 or clip_h <= 0:
            return  # camera is entirely outside the world; nothing to draw

        # Pixel scale: how many screen pixels per world pixel under this camera
        scale_x = dst_w / max(w_wpx, 1.0)
        scale_y = dst_h / max(h_wpx, 1.0)
        # The dst rectangle for the clipped portion is also clipped and shifted
        dst_off_x = (clip_x0 - x_wpx) * scale_x
        dst_off_y = (clip_y0 - y_wpx) * scale_y
        dst_real_w = clip_w * scale_x
        dst_real_h = clip_h * scale_y

        src = rl.Rectangle(
            float(clip_x0),
            float(self.world_px_h - (clip_y0 + clip_h)),
            float(clip_w),
            -float(clip_h),
        )
        dst = rl.Rectangle(
            float(dst_x + dst_off_x),
            float(dst_y + dst_off_y),
            float(dst_real_w),
            float(dst_real_h),
        )
        # Use premultiplied alpha for the RT-to-screen blit. The contents
        # of the world RT are already in premultiplied form: any overlay
        # drawn into it with default alpha blending produces
        # rt.rgb = src.rgb * src.a + dest.rgb * (1 - src.a), and where
        # dest is transparent (vacuum tiles), this leaves rt.rgb as the
        # premultiplied smoke contribution. Blitting with normal alpha
        # would multiply by alpha AGAIN — producing smoke.rgb * a² and
        # excessive background bleed. PREMUL gives correct compositing.
        rl.begin_blend_mode(rl.BlendMode.BLEND_ALPHA_PREMULTIPLY)
        rl.draw_texture_pro(self.rt.texture, src, dst,
                            rl.Vector2(0, 0), 0.0, rl.WHITE)
        rl.end_blend_mode()

    # ---- Shutdown -------------------------------------------------------

    def unload(self) -> None:
        rl.unload_shader(self.resolve_shader)
        rl.unload_render_texture(self.scene_rt)
        rl.unload_render_texture(self.rt)
