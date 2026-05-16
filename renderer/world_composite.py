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

        Y-FLIP: source rect uses negative height because Raylib RTs are
        Y-up internally but the screen is Y-down.
        """
        # Visible region in world pixels
        x_wpx, y_wpx, w_wpx, h_wpx = camera.visible_world_rect_world_px(
            self.world_px_per_tile
        )
        # Build source rectangle, flipped vertically.
        src = rl.Rectangle(
            float(x_wpx),
            float(self.world_px_h - y_wpx),   # Y flip: top of viewport is
                                              # high in world-RT Y space
            float(w_wpx),
            -float(h_wpx),                    # negative height = vertical flip
        )
        dst = rl.Rectangle(float(dst_x), float(dst_y),
                           float(dst_w), float(dst_h))
        rl.draw_texture_pro(self.rt.texture, src, dst,
                            rl.Vector2(0, 0), 0.0, rl.WHITE)

    # ---- Shutdown -------------------------------------------------------

    def unload(self) -> None:
        rl.unload_render_texture(self.rt)
