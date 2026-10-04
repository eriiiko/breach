"""unit_shadow -- the 3D units' sharp projected shadows (#33, tier 1).

The old Quake 2 trick: each unit's already-skinned model is drawn a second
time through a matrix that flattens it onto the floor plane, slid away from the
light, as one flat see-through dark shape. It sits on top of the soft shadows
(units absorb light in the sweep) and the blob under each unit, and shows the
unit's own silhouette -- limbs, backpack.

What this module owns (render-only; nothing here touches a ``Unit`` or the
simulation):

  * :func:`unit_shadow_settings` -- ``[render.unit_shadow]``, read every frame
    (Ctrl+R retunes it); a missing or mistyped key raises, naming it.
  * :func:`read_unit_light` -- the light around one unit, from the light
    accessor's :class:`simulation.light_field.LightView` (the renderer's ONE
    light read): brightness, the net direction the light travels, and how
    one-sided it is (``LightView.directionality``).
  * :func:`shadow_cast` -- PURE: that light + the settings -> how dark the
    shadow is and which way / how far it is slid. The fade rules live here.
  * :func:`smooth_shadow` -- PURE: the per-unit low-pass on the shadow vector,
    so a direction change fades through zero instead of jumping.
  * :func:`projection_matrix` -- PURE: the flattening matrix, in raylib's
    float[16] (column-major) order.
  * :class:`ShadowPainter` -- the GL side: the flat shader, and the depth trick
    that darkens each floor pixel ONCE (see its docstring).

Tier 1: ONE elevation for every light -- the shadow of a point at height h is
slid ``length * h`` along the light's horizontal direction. The shadow lies on
the floor plane only; it does not climb walls.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UnitShadowSettings:
    """``[render.unit_shadow]``. ``enabled`` false draws exactly the picture
    without projected shadows."""
    enabled: bool
    length: float              # slide per unit of height (shadow = length x height)
    strength: float            # the most the floor is darkened, 0 .. 1
    min_brightness: float      # fade-in: 0 at/below this, full at twice it
    min_directionality: float  # fade-in: 0 at/below this, full at twice it


_SHADOW_KEYS = {"enabled": bool, "length": float, "strength": float,
                "min_brightness": float, "min_directionality": float}


def unit_shadow_settings(cfg) -> UnitShadowSettings:
    """The ``[render.unit_shadow]`` settings of a loaded config (``CFG``).
    Raises ValueError naming the key when the section or a key is missing,
    of the wrong type or out of range."""
    section = getattr(getattr(cfg, "render", None), "unit_shadow", None)
    if section is None:
        raise ValueError("config.toml has no [render.unit_shadow] section")
    values = {}
    for key, kind in _SHADOW_KEYS.items():
        if not hasattr(section, key):
            raise ValueError(f"[render.unit_shadow] is missing {key!r}")
        v = getattr(section, key)
        ok = isinstance(v, bool) if kind is bool else (
            isinstance(v, (int, float)) and not isinstance(v, bool))
        if not ok:
            raise ValueError(f"[render.unit_shadow] {key} must be a "
                             f"{kind.__name__}, got {v!r}")
        values[key] = kind(v)
    if values["length"] < 0.0:
        raise ValueError(f"[render.unit_shadow] length must be >= 0, got {values['length']!r}")
    if not 0.0 <= values["strength"] <= 1.0:
        raise ValueError(f"[render.unit_shadow] strength must be in [0, 1], "
                         f"got {values['strength']!r}")
    if values["min_brightness"] <= 0.0:
        raise ValueError(f"[render.unit_shadow] min_brightness must be > 0, "
                         f"got {values['min_brightness']!r}")
    if not 0.0 < values["min_directionality"] <= 0.5:
        raise ValueError(f"[render.unit_shadow] min_directionality must be in "
                         f"(0, 0.5], got {values['min_directionality']!r}")
    return UnitShadowSettings(**values)


# ---------------------------------------------------------------------------
# The light around a unit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UnitLight:
    """The light around one unit, as the shadow reads it."""
    brightness: float        # brightest channel, mean over the patch, in the
                             # ship shader's units (incoming x exposure)
    directionality: float    # |net flux| / irradiance over the patch, 0 .. 1
    dir_x: float             # the way the light TRAVELS, unit vector
    dir_y: float             # (+y = increasing row); (0, 0) when no net flux


_DARK = UnitLight(0.0, 0.0, 0.0, 0.0)


def read_unit_light(view, x0: int, y0: int, footprint: int,
                    exposure: float) -> UnitLight:
    """Read the light around a unit from the accessor's ``LightView``, over
    the unit's footprint tiles -- where the light ARRIVES at the unit (a
    cell's stream is counted before the body absorbs it). Not a ring around
    it: a marine's own flashlight leaves from the first cell outside its
    footprint (``frame_lights.flashlight_lens``), and reading it would throw
    his shadow forward into his own beam. The net flux is summed over the patch in light units
    (``flux_dir * directionality * Σ_c rgb`` per cell) and divided by the
    patch's summed irradiance, so a lamp on one side reads one-sided and two
    equal lamps on opposite sides cancel. Pure numpy; reads, never writes."""
    if view is None:
        return _DARK
    planes = _shadow_planes(view)
    h, w = planes.shape[:2]
    ya, yb = max(0, y0), min(h, y0 + footprint)
    xa, xb = max(0, x0), min(w, x0 + footprint)
    if ya >= yb or xa >= xb:
        return _DARK
    r, g, b, fx, fy = planes[ya:yb, xa:xb].sum(axis=(0, 1)).tolist()
    total = r + g + b
    if total <= 0.0:
        return _DARK
    norm = math.hypot(fx, fy)
    brightness = max(r, g, b) / float((yb - ya) * (xb - xa)) * float(exposure)
    if norm <= 0.0:
        return UnitLight(brightness, 0.0, 0.0, 0.0)
    return UnitLight(brightness, min(1.0, norm / total), fx / norm, fy / norm)


_planes_cache: list = [None, None]     # [the LightView, its planes]


def _shadow_planes(view) -> np.ndarray:
    """(h, w, 5) float64: r, g, b and the net flux (x, y) in light units --
    built once per LightView (once per sim tick), so each unit's read is one
    slice-and-sum."""
    if _planes_cache[0] is view:
        return _planes_cache[1]
    rgb = view.rgb.astype(np.float64)
    mag = view.directionality.astype(np.float64) * rgb.sum(axis=2)
    planes = np.empty(rgb.shape[:2] + (5,), dtype=np.float64)
    planes[..., 0:3] = rgb
    planes[..., 3] = view.flux_dir[..., 0] * mag
    planes[..., 4] = view.flux_dir[..., 1] * mag
    _planes_cache[0], _planes_cache[1] = view, planes
    return planes


# ---------------------------------------------------------------------------
# Pure: light -> shadow
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShadowCast:
    """One unit's shadow. ``alpha`` 0 = none. (``vx``, ``vy``) = how far a
    point slides per unit of its height, in world (x, y-down) -- the light's
    travel direction times ``length``."""
    alpha: float
    vx: float
    vy: float


NO_SHADOW = ShadowCast(0.0, 0.0, 0.0)


def _ramp(value: float, threshold: float) -> float:
    """0 at/below ``threshold``, rising linearly to 1 at twice it."""
    return min(1.0, max(0.0, (value - threshold) / threshold))


def shadow_cast(light: UnitLight, s: UnitShadowSettings) -> ShadowCast:
    """THE fade + direction rule. Dark patch -> no shadow; light with no clear
    direction -> no shadow; it grows to ``strength`` as the light gets both
    bright and one-sided, so the direction is only ever shown where it is
    well defined (no flicker from a weak or balanced flux)."""
    if not s.enabled or s.strength <= 0.0:
        return NO_SHADOW
    fade = (_ramp(light.brightness, s.min_brightness)
            * _ramp(light.directionality, s.min_directionality))
    if fade <= 0.0 or (light.dir_x == 0.0 and light.dir_y == 0.0):
        return NO_SHADOW
    return ShadowCast(s.strength * fade, s.length * light.dir_x,
                      s.length * light.dir_y)


def smooth_shadow(prev: Optional[ShadowCast], target: ShadowCast,
                  dt: float, tau: float) -> ShadowCast:
    """Low-pass the shadow as ONE vector (alpha x direction): a turn of the
    light shrinks the shadow through faint instead of snapping it across the
    unit. ``prev`` None starts at ``target``. The result's alpha never exceeds
    the larger of the two inputs' (a convex combination)."""
    if prev is None or tau <= 0.0:
        return target
    k = 1.0 - math.exp(-max(0.0, dt) / tau)

    def vec(c):
        n = math.hypot(c.vx, c.vy)
        return (0.0, 0.0) if n == 0.0 else (c.alpha * c.vx / n, c.alpha * c.vy / n)

    (px, py), (tx, ty) = vec(prev), vec(target)
    ax, ay = px + (tx - px) * k, py + (ty - py) * k
    alpha = math.hypot(ax, ay)
    if alpha < 1e-4:
        return NO_SHADOW
    # The slide length follows the target (it is the setting, not a light
    # quantity); the direction follows the smoothed vector.
    length = math.hypot(target.vx, target.vy) or math.hypot(prev.vx, prev.vy)
    return ShadowCast(alpha, length * ax / alpha, length * ay / alpha)


def projection_matrix(vx: float, vy: float, plane_y: float = 0.0) -> Tuple[float, ...]:
    """The flattening matrix, raylib float[16] order (``m0 .. m15``, column-
    major), applied AFTER the model's own transform (rlgl's transform stack),
    i.e. to world points in the 3D frame (X = world x, Y = up, Z = world y):

        X' = X + vx * Y      Y' = plane_y      Z' = Z + vy * Y

    so a point at height Y lands ``Y * (vx, vy)`` away along the light's travel
    direction, and the feet (Y = 0) stay where they stand."""
    return (1.0, 0.0, 0.0, 0.0,
            float(vx), 0.0, float(vy), 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, float(plane_y), 0.0, 1.0)


def project_point(m: Tuple[float, ...], x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Apply a float[16] raylib matrix to a point (the column-major product
    rlgl performs) -- the testable twin of what the GPU does with it."""
    return (m[0] * x + m[4] * y + m[8] * z + m[12],
            m[1] * x + m[5] * y + m[9] * z + m[13],
            m[2] * x + m[6] * y + m[10] * z + m[14])


# ---------------------------------------------------------------------------
# GL side
# ---------------------------------------------------------------------------

SHADOW_VS = """#version 330
in vec3 vertexPosition;
uniform mat4 mvp;
void main() { gl_Position = mvp * vec4(vertexPosition, 1.0); }
"""

# Premultiplied black at u_alpha; every fragment at ONE depth (see ShadowPainter).
SHADOW_FS = """#version 330
uniform float u_alpha;
uniform float u_depth;
out vec4 finalColor;
void main() {
    finalColor = vec4(0.0, 0.0, 0.0, u_alpha);
    gl_FragDepth = u_depth;
}
"""

# The shadow's depth: far behind everything on the floor (a unit's feet sit at
# ~0.5 in the top-down camera's depth range) yet in front of the cleared 1.0.
SHADOW_DEPTH = 0.999
_GL_LESS = 0x0201
_GL_LEQUAL = 0x0203      # rlgl's own depth function (rlglInit)


def _load_gl_depth_func():
    """``glDepthFunc`` straight from the system GL library (a GL 1.0 entry
    point every driver exports; rlgl does not expose it). None if absent."""
    try:
        if os.name == "nt":
            lib = ctypes.WinDLL("opengl32")
        else:
            name = ctypes.util.find_library("GL")
            if not name:
                return None
            lib = ctypes.CDLL(name)
        fn = lib.glDepthFunc
        fn.argtypes = [ctypes.c_uint]
        fn.restype = None
        return fn
    except (OSError, AttributeError):
        return None


class ShadowPainter:
    """Draws a flattened model as a flat see-through shadow, darkening each
    floor pixel ONCE however many triangles -- or units -- cover it.

    How: every shadow fragment is written at the SAME depth (``SHADOW_DEPTH``,
    set in the fragment shader) with depth writes on and the depth test
    switched to ``GL_LESS`` for the shadow draw. The first fragment to land on
    a pixel passes against the cleared 1.0 and writes ``SHADOW_DEPTH``; every
    later shadow fragment there is equal, not less, and is rejected -- one
    darkening per pixel per frame, across limbs and across units. The same
    depth keeps a shadow off every unit: a body (nearer than the floor) drawn
    before it has written a smaller depth that rejects it, a body drawn after
    it passes over it. Backface culling is off for the draw (the flattened
    mesh's triangles face both ways; overdraw is free under this test).
    Blending is premultiplied (``BLEND_ALPHA_PREMULTIPLY``) so the world RT's
    destination alpha stays 1."""

    def __init__(self, rl) -> None:
        self._rl = rl
        self.shader = None
        self._loc_alpha = -1
        self._loc_depth = -1
        self._depth_func = _load_gl_depth_func()
        if self._depth_func is None:
            print("[unit_shadow] WARN: glDepthFunc unavailable; projected "
                  "shadows disabled")
            return
        sh = rl.load_shader_from_memory(SHADOW_VS, SHADOW_FS)
        if sh.id == 0:
            print("[unit_shadow] WARN: shadow shader failed to compile; "
                  "projected shadows disabled")
            return
        self.shader = sh
        self._loc_alpha = rl.get_shader_location(sh, "u_alpha")
        self._loc_depth = rl.get_shader_location(sh, "u_depth")
        self._set_float(self._loc_depth, SHADOW_DEPTH)

    @property
    def ready(self) -> bool:
        return self.shader is not None

    def _set_float(self, loc: int, v: float) -> None:
        rl = self._rl
        rl.set_shader_value(self.shader, loc, rl.ffi.new("float[1]", [float(v)]),
                            rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT)

    def draw(self, model, materials, cast: ShadowCast, position, yaw: float,
             scale: float) -> None:
        """Draw ``model`` (already skinned) flattened by ``cast``. Must run
        inside an open ``begin_mode_3d``. ``materials`` = the indices whose
        shader is swapped for the draw and restored after."""
        rl = self._rl
        if not self.ready or cast.alpha <= 0.0:
            return
        self._set_float(self._loc_alpha, cast.alpha)
        rl.rl_draw_render_batch_active()          # flush batched draws first
        self._depth_func(_GL_LESS)
        rl.rl_disable_backface_culling()
        rl.begin_blend_mode(rl.BlendMode.BLEND_ALPHA_PREMULTIPLY)
        saved = []
        for mi in materials:
            # BY VALUE: cffi hands back a nested struct field as a view into
            # the material, so keeping the field itself would "restore" the
            # shadow shader onto the body.
            cur = model.materials[mi].shader
            saved.append((mi, rl.Shader(cur.id, cur.locs)))
            model.materials[mi].shader = self.shader
        m = rl.ffi.new("float[16]", list(projection_matrix(cast.vx, cast.vy)))
        rl.rl_push_matrix()
        try:
            # a 'float *' (pyray would take the address of a bare array)
            rl.rl_mult_matrixf(rl.ffi.cast("float *", m))
            rl.draw_model_ex(model, position, rl.Vector3(0.0, 1.0, 0.0), yaw,
                             rl.Vector3(scale, scale, scale), rl.WHITE)
        finally:
            rl.rl_pop_matrix()
            for mi, sh in saved:
                model.materials[mi].shader = sh
            rl.end_blend_mode()
            rl.rl_enable_backface_culling()
            self._depth_func(_GL_LEQUAL)

    def unload(self) -> None:
        if self.shader is not None:
            self._rl.unload_shader(self.shader)
            self.shader = None


__all__ = ["UnitShadowSettings", "unit_shadow_settings", "UnitLight",
           "read_unit_light", "ShadowCast", "NO_SHADOW", "shadow_cast",
           "smooth_shadow", "projection_matrix", "project_point",
           "ShadowPainter", "SHADOW_DEPTH"]
