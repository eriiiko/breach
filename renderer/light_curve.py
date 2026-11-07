"""The light curve (graphics G5, #64): a Photoshop-style Curves map on the light.

Input: the light reaching a pixel (Rec.709 luminance of the light field x
exposure). Output: how much light that pixel takes instead. Every lit surface
runs the same map per fragment (``lit3d._LIGHT_CURVE_GLSL``: the floor, the 3D
units, the props), so a steep segment is a crisp edge that follows the
bilinear light field. The map scales the light and keeps its colour, and is
applied BEFORE albedo and N.L, so the art and its normal relief survive.

Render-only: nothing here reaches the sim, a digest or a golden. With
``enabled = false`` the shader leaves the light untouched, bit for bit.

The axes are gamma-encoded (``x = (L / range) ** (1 / CURVE_GAMMA)``), as an
image editor's are, so the dim end -- where this look lives -- gets the room.
A curve is control points in that square, interpolated by a monotone cubic
(Fritsch & Carlson 1980, "Monotone Piecewise Cubic Interpolation", SIAM J.
Numer. Anal. 17(2)): unlike a plain spline it cannot overshoot a steep step,
so a threshold stays a clean step. Light above ``range`` keeps the top of the
axis's ratio, so the identity curve is the identity everywhere.

This file is pyray-free and is the spec of the GLSL: ``apply_curve`` is the
numpy twin of ``lightCurveGain``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

LUT_SIZE = 256          # the shader's uniform float array; a step in the
                        # curve is one entry wide, 1/255 of the x axis
CURVE_GAMMA = 2.2       # the axes' encoding
LUMA = (0.2126, 0.7152, 0.0722)     # Rec.709
MIN_POINT_GAP = 1.0 / (LUT_SIZE - 1)    # two points this close are a step
CORNER_RATIO = 4.0      # slopes this unequal at a point make it a corner: the
                        # tangent is the gentler slope, so the curve stays
                        # straight right up to a step (still monotone: the
                        # gentler slope is inside Fritsch-Carlson's region)
L_EPS = 1e-6            # below this a pixel has no light to scale

IDENTITY_POINTS = ((0.0, 0.0), (1.0, 1.0))


@dataclass(frozen=True)
class LightCurve:
    """The ``[render.light_curve]`` table (config.toml has each key's story)."""
    enabled: bool
    points: tuple           # ((x, y), ...) strictly increasing in x, in [0, 1]
    range: float            # light units at x = 1
    dark_saturation: float  # the art's saturation in the dark; 1 = off
    dark_knee: float        # encoded output where saturation is full again
    panel: bool             # show the on-screen curve editor


def check_points(points) -> tuple:
    """Normalise control points to a tuple of float pairs. Raises ValueError
    naming the fault: under two points, a point outside the unit square, or
    x values that do not increase by at least ``MIN_POINT_GAP``."""
    try:
        pts = tuple((float(p[0]), float(p[1])) for p in points)
    except (TypeError, ValueError, IndexError) as e:
        raise ValueError(f"[render.light_curve] points: not [x, y] pairs "
                         f"({e})") from e
    if len(pts) < 2:
        raise ValueError("[render.light_curve] points: at least two needed")
    for x, y in pts:
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(f"[render.light_curve] points: ({x}, {y}) is "
                             f"outside the unit square")
    for (x0, _), (x1, _) in zip(pts, pts[1:]):
        if x1 - x0 < MIN_POINT_GAP - 1e-12:
            raise ValueError(f"[render.light_curve] points: x {x0} then {x1} "
                             f"(must increase by at least {MIN_POINT_GAP:.4f})")
    return pts


def light_curve_settings(cfg) -> LightCurve:
    """The ``[render.light_curve]`` settings of a loaded config (``CFG``).
    Raises ValueError naming the key when the section or a key is missing or
    mistyped: a dial that silently fell back would hide a typo while tuning."""
    section = getattr(getattr(cfg, "render", None), "light_curve", None)
    if section is None:
        raise ValueError("config.toml has no [render.light_curve] section")
    values = {}
    for name in ("enabled", "panel"):
        if not hasattr(section, name):
            raise ValueError(f"[render.light_curve] is missing {name!r}")
        v = getattr(section, name)
        if not isinstance(v, bool):
            raise ValueError(f"[render.light_curve] {name} must be a bool")
        values[name] = v
    for name, lo, hi in (("range", 1e-3, 1e6), ("dark_saturation", 0.0, 1.0),
                         ("dark_knee", 1e-3, 1.0)):
        if not hasattr(section, name):
            raise ValueError(f"[render.light_curve] is missing {name!r}")
        v = getattr(section, name)
        if isinstance(v, bool) or not isinstance(v, (int, float)) \
                or not lo <= v <= hi:
            raise ValueError(f"[render.light_curve] {name} must be a number "
                             f"in [{lo}, {hi}], got {v!r}")
        values[name] = float(v)
    if not hasattr(section, "points"):
        raise ValueError("[render.light_curve] is missing 'points'")
    values["points"] = check_points(section.points)
    return LightCurve(**values)


def bake_lut(points, size: int = LUT_SIZE) -> np.ndarray:
    """The curve sampled at ``size`` even x in [0, 1], float32, in [0, 1].
    Flat outside the first and last point. Monotone between control points
    (Fritsch-Carlson's harmonic-mean tangents, a corner beside a step), so it
    never overshoots and a step leaves the light below it alone."""
    pts = check_points(points)
    px = np.array([p[0] for p in pts], dtype=np.float64)
    py = np.array([p[1] for p in pts], dtype=np.float64)
    h = np.diff(px)
    delta = np.diff(py) / h
    m = np.zeros(len(px), dtype=np.float64)
    m[0], m[-1] = delta[0], delta[-1]
    for i in range(1, len(px) - 1):
        d0, d1 = delta[i - 1], delta[i]
        if d0 * d1 > 0.0:
            if max(abs(d0), abs(d1)) >= CORNER_RATIO * min(abs(d0), abs(d1)):
                m[i] = d0 if abs(d0) < abs(d1) else d1
            else:
                w0 = 2.0 * h[i] + h[i - 1]
                w1 = h[i] + 2.0 * h[i - 1]
                m[i] = (w0 + w1) / (w0 / d0 + w1 / d1)
    x = np.linspace(0.0, 1.0, size)
    u = np.clip(x, px[0], px[-1])
    i = np.clip(np.searchsorted(px, u, side="right") - 1, 0, len(h) - 1)
    t = (u - px[i]) / h[i]
    t2, t3 = t * t, t * t * t
    y = ((2.0 * t3 - 3.0 * t2 + 1.0) * py[i]
         + (t3 - 2.0 * t2 + t) * h[i] * m[i]
         + (-2.0 * t3 + 3.0 * t2) * py[i + 1]
         + (t3 - t2) * h[i] * m[i + 1])
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def encode(light, range_: float):
    """Light units -> the curve's x axis (clipped to [0, 1])."""
    return np.clip(np.asarray(light, dtype=np.float64) / range_, 0.0, 1.0) \
        ** (1.0 / CURVE_GAMMA)


def decode(y, range_: float):
    """The curve's y axis -> light units."""
    return np.asarray(y, dtype=np.float64) ** CURVE_GAMMA * range_


def luminance(rgb) -> np.ndarray:
    """Rec.709 luminance over the last axis (no BLAS: a per-tick reader)."""
    rgb = np.asarray(rgb)
    return (rgb[..., 0] * LUMA[0] + rgb[..., 1] * LUMA[1]
            + rgb[..., 2] * LUMA[2])


def curve_gain(light_rgb, lut: np.ndarray, range_: float) -> np.ndarray:
    """The twin of GLSL ``lightCurveGain``: the factor on a pixel's light."""
    L = np.maximum(luminance(np.maximum(light_rgb, 0.0)), 0.0)
    x = encode(L, range_)
    n = len(lut)
    u = x * (n - 1)
    i = np.minimum(u.astype(np.int64), n - 2)
    y = lut[i] + (lut[i + 1] - lut[i]) * (u - i)
    out = decode(y, range_) * np.maximum(L / range_, 1.0)
    return np.where(L > L_EPS, out / np.maximum(L, L_EPS), 1.0)


def apply_curve(light_rgb, lut: np.ndarray, range_: float) -> np.ndarray:
    """A pixel's light after the curve (float64 ``(..., 3)``)."""
    rgb = np.asarray(light_rgb, dtype=np.float64)
    return rgb * curve_gain(rgb, lut, range_)[..., None]


def light_histogram(light_rgb, range_: float, bins: int = 64) -> np.ndarray:
    """Share of the lit tiles per bin of the x axis, float64 summing to 1 (all
    zeros with no lit tile). The panel draws it behind the curve, as an image
    editor does, so it shows where this room's light actually sits."""
    L = luminance(np.maximum(light_rgb, 0.0))
    lit = L[L > L_EPS]
    if lit.size == 0:
        return np.zeros(bins, dtype=np.float64)
    h, _ = np.histogram(encode(lit, range_), bins=bins, range=(0.0, 1.0))
    return h / lit.size


def format_points(points) -> str:
    """The TOML value of ``points``, 3 decimals (the LUT's resolution)."""
    return "[" + ", ".join(f"[{x:.3f}, {y:.3f}]" for x, y in points) + "]"


def write_settings(path, curve: LightCurve) -> None:
    """Rewrite ``enabled``, ``points`` and ``dark_saturation`` in the file's
    ``[render.light_curve]`` table, each on its own line, and nothing else:
    every other byte of config.toml, comments included, is kept. Raises
    ValueError if the table or one of the three lines is not there."""
    path = Path(path)
    raw = path.read_bytes().decode("utf-8")
    start = raw.find("[render.light_curve]\n")
    if start < 0:
        start = raw.find("[render.light_curve]\r\n")
    if start < 0:
        raise ValueError(f"{path.name} has no [render.light_curve] table")
    next_table = re.compile(r"^\[", re.M).search(raw, start + 1)
    end = next_table.start() if next_table else len(raw)
    body = raw[start:end]
    new = {"enabled": "true" if curve.enabled else "false",
           "points": format_points(curve.points),
           "dark_saturation": f"{curve.dark_saturation:.3f}"}
    for key, value in new.items():
        body, n = re.subn(rf"^{key}[ \t]*=[ \t]*[^#\r\n]*?([ \t]*(?:#[^\r\n]*)?)$",
                          lambda m: f"{key} = {value}{m.group(1)}",
                          body, count=1, flags=re.M)
        if n != 1:
            raise ValueError(f"[render.light_curve] has no {key} line")
    path.write_bytes((raw[:start] + body + raw[end:]).encode("utf-8"))


__all__ = ["CURVE_GAMMA", "IDENTITY_POINTS", "LUT_SIZE", "LightCurve",
           "MIN_POINT_GAP", "apply_curve", "bake_lut", "check_points",
           "curve_gain", "decode", "encode", "format_points",
           "light_curve_settings", "light_histogram", "luminance", "replace",
           "write_settings"]
