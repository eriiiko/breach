"""WorldComposite — single render target sized to the entire world.

All world-space drawing (lit ship, smoke, fire, units, particles, debug)
happens *into* this RT inside BeginTextureMode. The camera is then a single
DrawTexturePro blit from this RT to the screen, with a source rectangle
computed from the camera's visible world rectangle.

Benefits:
  - Every shader samples fragTexCoord 0..1 over the world — no camera UVs.
  - Adding a new layer is one more draw call inside compose(), not a shader.
  - Post-FX (bloom, vignette, CRT) becomes one extra full-screen pass.
  - Multiple cameras (security cam, split screen, replay) = re-blit same RT.

Cost: one RT allocation. For our typical ship (50x120 tiles at 24 wpx/tile)
that is 1200x2880 RGBA8 = ~14 MB. Trivial on modern GPUs.

Y-FLIP: Raylib stores RT contents with origin at bottom-left (OpenGL).
       The screen has origin top-left. When blitting, the source rectangle
       MUST use a negative height to flip Y back. blit_to_screen() handles
       this — every other consumer reads from the RT via this class.
"""
from __future__ import annotations

import pyray as rl

from .camera import Camera2D


class WorldComposite:
    """Owns the world-space render target and its lifecycle."""

    def __init__(self, world_tile_w: int, world_tile_h: int,
                 world_px_per_tile: float):
        self.world_tile_w = world_tile_w
        self.world_tile_h = world_tile_h
        self.world_px_per_tile = float(world_px_per_tile)
        self.world_px_w = int(round(world_tile_w * world_px_per_tile))
        self.world_px_h = int(round(world_tile_h * world_px_per_tile))

        self.rt = rl.load_render_texture(self.world_px_w, self.world_px_h)
        # Smooth zoom when blitting to screen; turn to POINT for pixel-art crisp.
        rl.set_texture_filter(self.rt.texture,
                              rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
        # Clamp so the final blit doesn't sample garbage past world bounds.
        rl.set_texture_wrap(self.rt.texture,
                            rl.TextureWrap.TEXTURE_WRAP_CLAMP)

    # ---- Compose-phase lifecycle ----------------------------------------

    def begin(self, clear_color=(0, 0, 0, 255)) -> None:
        """Bind RT as the draw target. Clear to color."""
        rl.begin_texture_mode(self.rt)
        rl.clear_background(rl.Color(*clear_color))

    def end(self) -> None:
        """Unbind RT."""
        rl.end_texture_mode()

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
        rl.draw_texture_pro(self.rt.texture, src, dst,
                            rl.Vector2(0, 0), 0.0, rl.WHITE)

    # ---- Shutdown -------------------------------------------------------

    def unload(self) -> None:
        rl.unload_render_texture(self.rt)
