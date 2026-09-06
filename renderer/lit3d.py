"""lit3d — the shared "lit 3D in the world RT" seam.

Everything drawn as real 3D geometry inside the 2D world render target (today:
the marine's skinned glTF body; next: static props/vegetation; later: 3D
walls) shares three things, extracted here so a second 3D consumer never
forks them:

  * ``_COMMON_GLSL`` — the sRGB decode/encode + ACES tone-map GLSL helpers,
    string-concatenated into a consumer's fragment shader. Kept numerically
    identical to ``shaders/lighting.fs`` (which keeps its OWN inline copies —
    golden-gated, never touched by this module) so every 3D-lit thing
    tone-maps and gamma-matches the 2D ship exactly.
  * ``LightFieldCtx`` — the ship's baked light field (``light_tex_a`` /
    ``light_tex_b`` + world dims + ambient/gain), built once per frame by
    ``game_renderer`` from its ``LightingPass`` + ``WorldComposite`` (single
    source of truth) and handed to every 3D draw call so each consumer
    samples EXACTLY what the ship samples.
  * ``make_camera`` — the top-down orthographic ``Camera3D`` factory, built
    UNCONDITIONALLY (not gated on any per-consumer toggle) so a second
    consumer never needs its own camera.

Coordinate mapping (calibrated in prototypes/scratchpad, see
``calib_camera.py``):
  * The world render target is ``world_px_w × world_px_h`` world-pixels, drawn
    top-left origin, y-down. A top-down ORTHOGRAPHIC ``Camera3D`` measured in
    world-pixels maps 3D X = x_wpx, 3D Z = y_wpx, 3D Y = height-up, with
    ``fovy = world_px_h`` and camera up = (0, 0, -1). Verified: 3D primitives
    land exactly on the same-coordinate 2D draws (calib_camera.py).
  * The RT's own depth buffer occludes correctly (verified: a near model drawn
    FIRST fully occludes a far model drawn SECOND), so NO world-Y painter's
    fallback is needed.

Extracted 2026-09 (props & vegetation arc #60, P1) from
``renderer/marine_shader.py`` (``_COMMON_GLSL``) and
``renderer/unit_model_renderer.py`` (``LightFieldCtx``, ``make_camera``,
``_CAM_HEIGHT``) — pure move, no behavior change. See
``docs/architecture/graphics/props_and_vegetation.md`` §4.3.
"""
from __future__ import annotations

from dataclasses import dataclass

import pyray as rl

# Shared helpers — string-concatenated into a consumer's fragment shader. Kept
# numerically identical to shaders/lighting.fs so every 3D-lit thing
# tone-maps and gamma-matches the ship exactly, without forking the
# golden-gated file.
_COMMON_GLSL = """
// Cheap sRGB <-> linear (gamma 2.2), matching shaders/lighting.fs.
vec3 srgb_to_linear(vec3 c) { return pow(c, vec3(2.2)); }
vec3 linear_to_srgb(vec3 c) { return pow(c, vec3(1.0 / 2.2)); }

// ACES filmic tone-map (Narkowicz), identical to shaders/lighting.fs.
vec3 aces_tonemap(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}
"""


@dataclass
class LightFieldCtx:
    """The ship's baked light field + scalars, handed to a 3D draw call so its
    shader samples EXACTLY what the ship samples. Built by game_renderer from
    its LightingPass + WorldComposite (single source of truth). ``tex_a`` /
    ``tex_b`` are the ``light_tex_a`` / ``light_tex_b`` RGBA16F textures."""
    tex_a: object
    tex_b: object
    world_px_w: float
    world_px_h: float
    ambient: tuple
    light_gain: float
    normal_y_sign: float = 1.0


# Top-down camera height above the floor, in world px. Ortho => this does NOT
# affect on-screen size, only near/far framing. It MUST stay below raylib's
# orthographic far-clip (empirically < ~5000 in this build), so it CANNOT scale
# with world size: a tall level (RT up to 5760 px) pushed the old
# max(w,h)*2 height past the far plane and culled every model (the "press M and
# everything vanishes" bug). 500 clears the tallest model (~3*wpt) with margin
# and is safely inside the far plane at every level size (verified 2400x5760).
_CAM_HEIGHT = 500.0


def make_camera(world_px_w: int, world_px_h: int) -> rl.Camera3D:
    """Top-down orthographic Camera3D framed to the world RT in world-px.

    3D X = x_wpx, 3D Z = y_wpx, 3D Y = up. ``fovy = world_px_h`` makes the
    ortho view span exactly the RT (aspect = w/h fills the width), so 3D
    world-px coords land on the same texels as the 2D world-px draws.
    Camera up = (0,0,-1) so world +Z (screen-down, i.e. y-down) reads down.
    """
    cam = rl.Camera3D()
    cx, cy = world_px_w / 2.0, world_px_h / 2.0
    # Fixed height (NOT world-size-scaled): ortho => distance doesn't change
    # on-screen size, only near/far framing, and it must stay under raylib's
    # ortho far-clip. See _CAM_HEIGHT (the old max(w,h)*2 culled everything on
    # tall levels).
    cam.position = rl.Vector3(cx, _CAM_HEIGHT, cy)
    cam.target = rl.Vector3(cx, 0.0, cy)
    cam.up = rl.Vector3(0.0, 0.0, -1.0)
    cam.fovy = float(world_px_h)
    cam.projection = rl.CameraProjection.CAMERA_ORTHOGRAPHIC
    return cam


__all__ = ["LightFieldCtx", "make_camera", "_COMMON_GLSL", "_CAM_HEIGHT"]
