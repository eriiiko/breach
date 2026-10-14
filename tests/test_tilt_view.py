"""Property gates for the tilted screen-stage camera (issue #90,
renderer/tilt_view.py) — HEADLESS: only pyray's raymath (no window).

Each test names the property it protects and the change that must break it.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pyray as rl

from renderer.camera import Camera2D
from renderer import tilt_view as tv


def _cam2d(x=10.0, y=20.0, zoom=30.0, vw=1600, vh=900, W=100, H=70):
    return Camera2D(pos_tile_x=x, pos_tile_y=y, zoom_px_per_tile=zoom,
                    viewport_px_w=vw, viewport_px_h=vh,
                    world_size_tile_w=W, world_size_tile_h=H)


# ---------------------------------------------------------------------------
# The tilt camera LOOKS AT the 2D camera's visible-rect centre, from above
# the ground, pitched as asked, and "up the screen" (camera south of the
# target, so screen-up stays map-north). Breaks if the derivation changes
# which point is framed or flips the view direction.
# ---------------------------------------------------------------------------
def test_tilt_camera_targets_visible_centre_with_requested_pitch():
    cam = _cam2d()
    wpt = 48.0
    pos, tgt, dist = tv.tilt_camera_params(cam, wpt, 1600, 900, 55.0, 45.0)
    x, y, w, h = cam.visible_world_rect_world_px(wpt)
    assert tgt == pytest_approx((x + w / 2, 0.0, y + h / 2))
    d = np.subtract(pos, tgt)
    assert d[1] > 0                           # above the ground
    assert d[2] > 0                           # south of the target (+Z)
    pitch = math.degrees(math.atan2(d[1], math.hypot(d[0], d[2])))
    assert abs(pitch - 55.0) < 1e-6
    assert abs(np.linalg.norm(d) - dist) < 1e-6


def pytest_approx(seq, tol=1e-6):
    class _A:
        def __eq__(self, other):
            return all(abs(a - b) < tol for a, b in zip(seq, other)) and len(seq) == len(other)
    return _A()


# ---------------------------------------------------------------------------
# ZOOM PARITY: the ground width seen across the viewport AT THE TARGET equals
# the 2D camera's visible width, so zooming in 2D zooms the tilt view by the
# same amount. Checked through raylib's own unproject (the same math the
# renderer uses): rays through the left and right viewport edges, at the
# target's depth, are one visible-width apart. Breaks if `distance` stops
# being derived from fovy/aspect.
# ---------------------------------------------------------------------------
def test_ground_width_at_target_matches_2d_visible_width():
    cam = _cam2d()
    wpt = 48.0
    vw, vh = 1600, 900
    pos, tgt, dist = tv.tilt_camera_params(cam, wpt, vw, vh, 55.0, 45.0)
    c = rl.Camera3D()
    c.position = rl.Vector3(*pos)
    c.target = rl.Vector3(*tgt)
    c.up = rl.Vector3(0, 1, 0)
    c.fovy = 45.0
    c.projection = rl.CameraProjection.CAMERA_PERSPECTIVE
    fwd = np.subtract(tgt, pos) / dist

    def plane_hit(sx):
        ray = rl.get_screen_to_world_ray_ex(rl.Vector2(sx, vh / 2), c, vw, vh)
        o = np.array([ray.position.x, ray.position.y, ray.position.z])
        d = np.array([ray.direction.x, ray.direction.y, ray.direction.z])
        # intersect the plane through the target perpendicular to the view
        t = np.dot(np.array(tgt) - o, fwd) / np.dot(d, fwd)
        return o + t * d

    left = plane_hit(0.0)
    right = plane_hit(float(vw))
    _, _, w_wpx, _ = cam.visible_world_rect_world_px(wpt)
    assert abs(np.linalg.norm(right - left) - w_wpx) < 1e-2 * w_wpx


# ---------------------------------------------------------------------------
# The centre-of-viewport ray hits the ground exactly at the target: the
# cursor->tile unproject and the raymarcher agree with the camera about
# where the middle of the screen is. Breaks if ground_hit or the camera
# matrices disagree with raylib's unproject.
# ---------------------------------------------------------------------------
def test_centre_ray_hits_the_target_on_the_ground():
    cam = _cam2d()
    wpt = 48.0
    vw, vh = 1600, 900
    pos, tgt, _ = tv.tilt_camera_params(cam, wpt, vw, vh, 60.0, 40.0)
    c = rl.Camera3D()
    c.position = rl.Vector3(*pos)
    c.target = rl.Vector3(*tgt)
    c.up = rl.Vector3(0, 1, 0)
    c.fovy = 40.0
    c.projection = rl.CameraProjection.CAMERA_PERSPECTIVE
    ray = rl.get_screen_to_world_ray_ex(rl.Vector2(vw / 2, vh / 2), c, vw, vh)
    hit = tv.ground_hit((ray.position.x, ray.position.y, ray.position.z),
                        (ray.direction.x, ray.direction.y, ray.direction.z))
    assert hit is not None
    assert abs(hit[0] - tgt[0]) < 1e-2 and abs(hit[1] - tgt[2]) < 1e-2
    # a ray pointing up never hits
    assert tv.ground_hit((0, 10, 0), (0, 1, 0)) is None


# ---------------------------------------------------------------------------
# The raymarcher's ray reconstruction uses inverse(proj @ view) built from
# raylib's matrices via matrix_to_numpy: unprojecting the centre pixel's NDC
# must reproduce raylib's own centre ray. Breaks if the m[c*4+r] layout or
# the NDC-y convention in the shader's Python counterpart drift.
# ---------------------------------------------------------------------------
def test_inverse_view_projection_reproduces_raylib_ray():
    cam = _cam2d()
    wpt = 48.0
    vw, vh = 1600, 900
    pos, tgt, _ = tv.tilt_camera_params(cam, wpt, vw, vh, 55.0, 45.0)
    c = rl.Camera3D()
    c.position = rl.Vector3(*pos)
    c.target = rl.Vector3(*tgt)
    c.up = rl.Vector3(0, 1, 0)
    c.fovy = 45.0
    c.projection = rl.CameraProjection.CAMERA_PERSPECTIVE
    view = tv.matrix_to_numpy(rl.get_camera_matrix(c))
    proj = tv.matrix_to_numpy(rl.matrix_perspective(
        math.radians(45.0), vw / vh, tv.NEAR_PX, tv.FAR_PX))
    inv_vp = np.linalg.inv(proj @ view)

    def shader_ray(u, v):                     # the GLSL main()'s first lines
        ndc = np.array([u * 2 - 1, 1 - v * 2])
        pn = inv_vp @ np.array([ndc[0], ndc[1], -1.0, 1.0])
        pf = inv_vp @ np.array([ndc[0], ndc[1], 1.0, 1.0])
        pn, pf = pn[:3] / pn[3], pf[:3] / pf[3]
        d = pf - pn
        return d / np.linalg.norm(d)

    for (u, v) in ((0.5, 0.5), (0.2, 0.8), (0.9, 0.1)):
        ray = rl.get_screen_to_world_ray_ex(rl.Vector2(u * vw, v * vh), c, vw, vh)
        ref = np.array([ray.direction.x, ray.direction.y, ray.direction.z])
        assert np.allclose(shader_ray(u, v), ref, atol=1e-4), (u, v)
