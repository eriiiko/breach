"""tilt_view — the tilted perspective SCREEN-STAGE view (issue #90).

The world render target keeps being composed top-down exactly as today
(``compose_world`` is untouched: floors, walls, lights, the 2D overlays, the
3D marines and props all land in it). This module replaces the final 2D blit
with a perspective one: the world RT becomes a textured GROUND QUAD under a
tilted ``Camera3D``, the volumetric smoke (``renderer/gas_volume.py``) is
raymarched above it, and the result is blitted to the map area. Walls and
units therefore read as flat decals on the floor in this view — accepted for
the prototype (design §3); 3D walls / depth-composited units come later.

Axis convention = ``renderer/lit3d.py``'s: 3D X = x_wpx, 3D Z = y_wpx,
3D Y = up, all in WORLD PIXELS. This is a different camera from
``lit3d.make_camera`` (which frames the world RT orthographically for things
drawn INTO it); it is the ONE screen-stage perspective camera — anything
drawn in perspective over the world RT goes through here (design §5).

The camera DERIVES from the existing ``Camera2D`` so panning/zooming keep
their keys: target = the visible rect's centre on the ground, distance so the
ground width at the target matches the 2D zoom, pitch/fovy from
``[render.tilt_view]``. The pure math (:func:`tilt_camera_params`,
:func:`ground_hit`) is pyray-free and tested headless.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pyray as rl

from .camera import Camera2D

# raylib's own defaults (rlgl RL_CULL_DISTANCE_NEAR / _FAR) — restored after
# our 3D pass so the ortho lit3d camera inside the world RT keeps its framing.
_RL_NEAR_DEFAULT = 0.01
_RL_FAR_DEFAULT = 1000.0
# Our clip planes, world px: the largest world RT is ~5760 px and the camera
# sits a few thousand px back, so 1000 would clip the ground.
NEAR_PX = 4.0
FAR_PX = 60000.0
# rlgl's RL_QUADS primitive id (== GL_QUADS, 0x0007); pyray exposes it but the
# stub does not, so resolve it with a fallback.
_RL_QUADS = int(getattr(rl, "RL_QUADS", 0x0007))


@dataclass
class TiltSettings:
    pitch_deg: float = 55.0     # 90 = straight down (the top-down look)
    fovy_deg: float = 45.0

    @classmethod
    def from_config(cls, cfg) -> "TiltSettings":
        tv = getattr(getattr(cfg, "render", None), "tilt_view", None)
        return cls(pitch_deg=float(getattr(tv, "pitch_deg", 55.0)),
                   fovy_deg=float(getattr(tv, "fovy_deg", 45.0)))


def tilt_camera_params(cam: Camera2D, world_px_per_tile: float,
                       viewport_w: int, viewport_h: int,
                       pitch_deg: float, fovy_deg: float
                       ) -> Tuple[Tuple[float, float, float],
                                  Tuple[float, float, float], float]:
    """(position, target, distance) in world px for the tilt camera.

    Target = the 2D camera's visible-rect centre on the ground (Y = 0). The
    camera sits ``distance`` back along a direction pitched ``pitch_deg``
    above the ground, looking "up the screen" (toward smaller y_wpx, i.e. -Z),
    so screen-up stays map-north as in the 2D view. ``distance`` is chosen so
    the ground width at the target equals the 2D camera's visible width:
    ``w = 2 d aspect tan(fovy/2)``. Pure math (no GL)."""
    x_wpx, y_wpx, w_wpx, h_wpx = cam.visible_world_rect_world_px(world_px_per_tile)
    cx = x_wpx + 0.5 * w_wpx
    cz = y_wpx + 0.5 * h_wpx
    aspect = float(viewport_w) / max(float(viewport_h), 1.0)
    half = math.tan(math.radians(fovy_deg) * 0.5)
    dist = w_wpx / max(2.0 * aspect * half, 1e-6)
    pitch = math.radians(pitch_deg)
    pos = (cx, dist * math.sin(pitch), cz + dist * math.cos(pitch))
    return pos, (cx, 0.0, cz), dist


def ground_hit(ray_pos, ray_dir) -> Optional[Tuple[float, float]]:
    """Intersect a ray with the ground plane Y = 0 -> (x_wpx, y_wpx) or None
    when the ray points away from / along the ground."""
    dy = float(ray_dir[1])
    if not math.isfinite(dy) or abs(dy) < 1e-9:
        return None
    t = -float(ray_pos[1]) / dy
    if not math.isfinite(t) or t < 0.0:
        return None
    x = float(ray_pos[0]) + t * float(ray_dir[0])
    z = float(ray_pos[2]) + t * float(ray_dir[2])
    if not (math.isfinite(x) and math.isfinite(z)):
        return None
    return (x, z)


def matrix_to_numpy(m: rl.Matrix) -> np.ndarray:
    """raylib Matrix (fields m0..m15, column-major: m[c*4+r]) -> (4,4) with
    M[r][c], so ``M @ [x, y, z, 1]`` is raylib's ``MatrixMultiply`` order."""
    out = np.empty((4, 4), dtype=np.float64)
    for c in range(4):
        for r in range(4):
            out[r, c] = float(getattr(m, f"m{c * 4 + r}"))
    return out


class TiltView:
    """Owns the viewport-sized RT, the derived Camera3D and the ground draw."""

    def __init__(self, map_px_w: int, map_px_h: int, world_px_w: int,
                 world_px_h: int, world_px_per_tile: float,
                 settings: Optional[TiltSettings] = None):
        self.map_px_w = int(map_px_w)
        self.map_px_h = int(map_px_h)
        self.world_px_w = float(world_px_w)
        self.world_px_h = float(world_px_h)
        self.wpt = float(world_px_per_tile)
        self.settings = settings or TiltSettings()
        self.rt = rl.load_render_texture(self.map_px_w, self.map_px_h)
        rl.set_texture_filter(self.rt.texture,
                              rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
        self.cam = rl.Camera3D()
        self.cam.up = rl.Vector3(0.0, 1.0, 0.0)
        self.cam.projection = rl.CameraProjection.CAMERA_PERSPECTIVE
        self.cam.fovy = float(self.settings.fovy_deg)
        # A valid placeholder framing (the world centre from above) until the
        # first update_camera(): the unproject must never see a zero camera.
        cx, cz = self.world_px_w * 0.5, self.world_px_h * 0.5
        self.cam.position = rl.Vector3(cx, max(self.world_px_h, 1.0), cz + 1.0)
        self.cam.target = rl.Vector3(cx, 0.0, cz)
        self.cam_pos = (cx, max(self.world_px_h, 1.0), cz + 1.0)
        self.inv_vp = np.eye(4)
        self.distance = 1.0
        self.cam_valid = False

    # ---- camera ----------------------------------------------------------

    def update_camera(self, cam2d: Camera2D) -> None:
        """Derive this frame's Camera3D (+ the inverse view-projection the
        raymarcher needs) from the 2D camera."""
        pos, tgt, dist = tilt_camera_params(
            cam2d, self.wpt, self.map_px_w, self.map_px_h,
            self.settings.pitch_deg, self.settings.fovy_deg)
        self.cam.position = rl.Vector3(*pos)
        self.cam.target = rl.Vector3(*tgt)
        self.cam.fovy = float(self.settings.fovy_deg)
        self.cam_pos = pos
        self.distance = dist
        view = matrix_to_numpy(rl.get_camera_matrix(self.cam))
        aspect = self.map_px_w / max(self.map_px_h, 1)
        proj = matrix_to_numpy(rl.matrix_perspective(
            math.radians(self.settings.fovy_deg), aspect, NEAR_PX, FAR_PX))
        self.inv_vp = np.linalg.inv(proj @ view)
        self.cam_valid = True

    def screen_ray(self, u: float, v: float) -> Tuple[np.ndarray, np.ndarray]:
        """(origin, unit direction) of the view ray through viewport fraction
        (u, v), (0,0) = top-left — the SAME reconstruction the raymarcher does
        in GLSL (ndc = (2u-1, 1-2v), inverse view-projection), in float64.
        raylib's own GetScreenToWorldRayEx is float32 against a 0.01..1000
        clip range and is ~1e-3 rad off at world-px scale, so it is not used."""
        ndc = np.array([u * 2.0 - 1.0, 1.0 - v * 2.0])
        pn = self.inv_vp @ np.array([ndc[0], ndc[1], -1.0, 1.0])
        pf = self.inv_vp @ np.array([ndc[0], ndc[1], 1.0, 1.0])
        pn = pn[:3] / pn[3]
        pf = pf[:3] / pf[3]
        d = pf - pn
        return np.asarray(self.cam_pos, dtype=np.float64), d / np.linalg.norm(d)

    def mouse_to_world_px(self, mx: float, my: float, viewport_x: int,
                          viewport_y: int) -> Optional[Tuple[float, float]]:
        """Screen px -> ground (x_wpx, y_wpx) through the current camera."""
        lx = float(mx - viewport_x)
        ly = float(my - viewport_y)
        if not (0 <= lx < self.map_px_w and 0 <= ly < self.map_px_h):
            return None
        if not self.cam_valid:
            return None
        ro, rd = self.screen_ray(lx / self.map_px_w, ly / self.map_px_h)
        return ground_hit(ro, rd)

    # ---- draw ------------------------------------------------------------

    def begin(self) -> None:
        """Open the view RT and draw the world RT as the ground quad. The
        caller then draws the volume pass (and anything else in screen
        space) and calls :meth:`end`."""
        rl.begin_texture_mode(self.rt)
        rl.clear_background(rl.Color(0, 0, 0, 0))

    def draw_ground(self, world_rt: rl.RenderTexture) -> None:
        W, H = self.world_px_w, self.world_px_h
        rl.rl_set_clip_planes(NEAR_PX, FAR_PX)
        rl.begin_mode_3d(self.cam)
        rl.rl_disable_backface_culling()
        # The world RT is premultiplied (vacuum = alpha 0) — keep it so, and
        # let the cleared-transparent backdrop show through there.
        rl.begin_blend_mode(rl.BlendMode.BLEND_ALPHA_PREMULTIPLY)
        rl.rl_set_texture(world_rt.texture.id)
        rl.rl_begin(_RL_QUADS)
        rl.rl_color4ub(255, 255, 255, 255)
        rl.rl_normal3f(0.0, 1.0, 0.0)
        # RT textures are stored bottom-up: v = 1 is the RT's top row
        # (y_wpx = 0), v = 0 its bottom (y_wpx = H) — the same flip the 2D
        # blit does with a negative source height.
        rl.rl_tex_coord2f(0.0, 1.0); rl.rl_vertex3f(0.0, 0.0, 0.0)
        rl.rl_tex_coord2f(0.0, 0.0); rl.rl_vertex3f(0.0, 0.0, H)
        rl.rl_tex_coord2f(1.0, 0.0); rl.rl_vertex3f(W, 0.0, H)
        rl.rl_tex_coord2f(1.0, 1.0); rl.rl_vertex3f(W, 0.0, 0.0)
        rl.rl_end()
        rl.rl_set_texture(0)
        rl.end_blend_mode()
        rl.rl_enable_backface_culling()
        rl.end_mode_3d()
        rl.rl_set_clip_planes(_RL_NEAR_DEFAULT, _RL_FAR_DEFAULT)

    def end(self) -> None:
        rl.end_texture_mode()

    def blit_to_screen(self, dst_x: int, dst_y: int) -> None:
        """The view RT -> the map area, y-flipped, premultiplied over the
        screen-fixed background (the WorldComposite blit's contract)."""
        src = rl.Rectangle(0.0, 0.0, float(self.map_px_w), -float(self.map_px_h))
        dst = rl.Rectangle(float(dst_x), float(dst_y),
                           float(self.map_px_w), float(self.map_px_h))
        rl.begin_blend_mode(rl.BlendMode.BLEND_ALPHA_PREMULTIPLY)
        rl.draw_texture_pro(self.rt.texture, src, dst, rl.Vector2(0, 0), 0.0,
                            rl.WHITE)
        rl.end_blend_mode()

    def unload(self) -> None:
        rl.unload_render_texture(self.rt)


__all__ = ["TiltSettings", "TiltView", "tilt_camera_params", "ground_hit",
           "matrix_to_numpy", "NEAR_PX", "FAR_PX"]
