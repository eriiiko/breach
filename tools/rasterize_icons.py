"""tools/rasterize_icons.py — self-contained SVG -> PNG rasterizer (Arc C8).

Editor doc §8 (LOCKED): entity palette icons are authored as small SVG
sources in ``art/entities/icons/<class>.svg`` and committed as PNGs
(``art/entities/icons/<class>.png``, ``rasterize_svg`` below); a class
without an icon renders the pre-existing generated colour chip + class
initial fallback (``tools/entity_editor_ui.py``) — that fallback is
PERMANENT, never an error, so icon coverage can be partial.

**No native rasterizer dependency.** There is no working cairosvg/svglib PNG
backend on this machine (both need native `cairo`, which is not installed —
do NOT reach for them). This module parses a small, CONSTRAINED subset of
SVG using only the Python stdlib XML parser + Pillow's ``ImageDraw`` and
renders directly, at a fixed supersample factor, then downsamples — fully
deterministic (same Pillow install in, same PNG bytes out every time; no
randomness anywhere in the pipeline).

Supported subset (anything else raises ``UnsupportedSvgError`` — LOUDLY, on
purpose: a silently-wrong icon is worse than a build failure, canon "no
icon is better than a wrong one"):

  - ``<svg viewBox="minx miny w h">`` — required; ``w``/``h`` must match
    (square viewBox) so one scale factor applies to both axes and to
    stroke widths.
  - ``<rect x y width height rx?>`` — ``rx`` (if present) renders rounded
    corners (uniform radius; SVG's separate ``ry`` is not supported).
  - ``<circle cx cy r>``, ``<ellipse cx cy rx ry>``.
  - ``<line x1 y1 x2 y2>``.
  - ``<polyline points="...">`` (fill implicitly closes like a polygon,
    per SVG; stroke does NOT — it follows only the explicit segments).
  - ``<polygon points="...">`` (implicitly closed for both fill and stroke).
  - ``<path d="...">`` using ONLY ``M``/``m``, ``L``/``l``, ``H``/``h``,
    ``V``/``v``, ``Z``/``z`` (absolute or relative moves/lines; NO curve or
    arc commands — ``C``/``S``/``Q``/``T``/``A`` in any case raise).
  - ``<text x y font-size fill>GLYPH</text>`` — a single glyph, centered on
    ``(x, y)`` (``text-anchor="middle"``/``dominant-baseline="middle"``
    behaviour is hardcoded, not read from attributes). Uses Pillow's bundled
    default font (``ImageFont.load_default(size=...)``) — no system-font
    lookup, so it renders identically regardless of what fonts are installed
    on the machine.

  Common presentation attributes on every shape: ``fill`` (colour name,
  ``#rgb``/``#rrggbb``, or ``none``; defaults to ``black`` per SVG, matching
  upstream semantics), ``stroke`` (defaults to ``none``), ``stroke-width``
  (user units, scaled like geometry; defaults to ``1``), ``opacity``
  (0..1, multiplies the alpha of both fill and stroke; defaults to ``1``).
  No CSS ``style="..."`` attribute, no ``<g>``/``<defs>``/``<use>``, no
  transforms — keeps the renderer tiny and its behaviour obvious by
  inspection.

CLI: ``python tools/rasterize_icons.py`` regenerates every
``art/entities/icons/<class>.png`` from its ``.svg`` sibling.
"""
from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
ICONS_DIR = ROOT / "art" / "entities" / "icons"

# Output contract: fixed size, deterministic supersample-then-downscale AA.
OUTPUT_SIZE = 64
SUPERSAMPLE = 4
CANVAS_SIZE = OUTPUT_SIZE * SUPERSAMPLE

SVG_NS = "{http://www.w3.org/2000/svg}"

_NAMED_COLORS = {
    "none": None,
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (255, 0, 0),
    "green": (0, 128, 0),
    "blue": (0, 0, 255),
    "grey": (128, 128, 128),
    "gray": (128, 128, 128),
    "transparent": (0, 0, 0, 0),
}

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


class UnsupportedSvgError(ValueError):
    """Raised loudly for any element/attribute/path-command outside the
    constrained subset this rasterizer supports — see module docstring."""


def _tag(elem) -> str:
    t = elem.tag
    return t[len(SVG_NS):] if t.startswith(SVG_NS) else t


def _floats(s: str) -> list:
    return [float(x) for x in _NUM_RE.findall(s)]


def _parse_color(s: Optional[str]):
    if s is None:
        return None
    s = s.strip()
    if s in _NAMED_COLORS:
        return _NAMED_COLORS[s]
    if s.startswith("#"):
        hexpart = s[1:]
        if len(hexpart) == 3:
            hexpart = "".join(c * 2 for c in hexpart)
        if len(hexpart) != 6:
            raise UnsupportedSvgError(f"unsupported color literal: {s!r}")
        r, g, b = (int(hexpart[i:i + 2], 16) for i in (0, 2, 4))
        return (r, g, b)
    raise UnsupportedSvgError(f"unsupported color literal: {s!r}")


class _Transform:
    """viewBox user-units -> supersampled canvas pixels. Requires a SQUARE
    viewBox (w == h) so one scalar scale applies uniformly to points AND to
    stroke widths — anything else would need independent x/y stroke scaling,
    which SVG doesn't even define sanely."""

    def __init__(self, minx: float, miny: float, w: float, h: float):
        if abs(w - h) > 1e-9:
            raise UnsupportedSvgError(
                f"viewBox must be square (got w={w}, h={h}) — this "
                f"rasterizer applies one uniform scale to geometry and "
                f"stroke widths alike")
        self.minx, self.miny = minx, miny
        self.scale = CANVAS_SIZE / w

    def pt(self, x: float, y: float):
        return ((x - self.minx) * self.scale, (y - self.miny) * self.scale)

    def length(self, v: float) -> float:
        return v * self.scale


def _alpha_scale(color, opacity: float):
    if color is None:
        return None
    if opacity >= 1.0:
        return tuple(color[:3]) + (255,) if len(color) == 3 else color
    a = int(round((color[3] if len(color) == 4 else 255) * opacity))
    return (color[0], color[1], color[2], max(0, min(255, a)))


def _common_style(elem, xf: _Transform):
    fill = _parse_color(elem.get("fill", "black"))
    stroke = _parse_color(elem.get("stroke", "none"))
    opacity = float(elem.get("opacity", "1"))
    sw = xf.length(float(elem.get("stroke-width", "1")))
    fill = _alpha_scale(fill, opacity)
    stroke = _alpha_scale(stroke, opacity)
    return fill, stroke, sw


def _draw_rect(draw, elem, xf: _Transform):
    x, y = float(elem.get("x", "0")), float(elem.get("y", "0"))
    w, h = float(elem["width"]), float(elem["height"])
    fill, stroke, sw = _common_style(elem, xf)
    p0 = xf.pt(x, y)
    p1 = xf.pt(x + w, y + h)
    bbox = [p0, p1]
    rx = elem.get("rx")
    kwargs = dict(fill=fill, outline=stroke,
                  width=max(1, round(sw)) if stroke is not None else None)
    if rx is not None:
        draw.rounded_rectangle(bbox, radius=xf.length(float(rx)), **kwargs)
    else:
        draw.rectangle(bbox, **kwargs)


def _draw_circle(draw, elem, xf: _Transform):
    cx, cy, r = float(elem["cx"]), float(elem["cy"]), float(elem["r"])
    _draw_ellipse_like(draw, elem, xf, cx, cy, r, r)


def _draw_ellipse(draw, elem, xf: _Transform):
    cx, cy = float(elem["cx"]), float(elem["cy"])
    rx, ry = float(elem["rx"]), float(elem["ry"])
    _draw_ellipse_like(draw, elem, xf, cx, cy, rx, ry)


def _draw_ellipse_like(draw, elem, xf, cx, cy, rx, ry):
    fill, stroke, sw = _common_style(elem, xf)
    p0 = xf.pt(cx - rx, cy - ry)
    p1 = xf.pt(cx + rx, cy + ry)
    draw.ellipse([p0, p1], fill=fill, outline=stroke,
                width=max(1, round(sw)) if stroke is not None else None)


def _draw_line(draw, elem, xf: _Transform):
    x1, y1 = float(elem["x1"]), float(elem["y1"])
    x2, y2 = float(elem["x2"]), float(elem["y2"])
    _, stroke, sw = _common_style(elem, xf)
    stroke_or_fill = stroke if stroke is not None else _parse_color(
        elem.get("fill", "black"))
    draw.line([xf.pt(x1, y1), xf.pt(x2, y2)], fill=stroke_or_fill,
              width=max(1, round(sw)))


def _parse_points(s: str) -> list:
    nums = _floats(s)
    if len(nums) % 2 != 0:
        raise UnsupportedSvgError(f"odd number of coordinates in points: {s!r}")
    return list(zip(nums[0::2], nums[1::2]))


def _draw_poly(draw, elem, xf: _Transform, *, closed: bool):
    pts = [xf.pt(x, y) for x, y in _parse_points(elem["points"])]
    if len(pts) < 2:
        raise UnsupportedSvgError("polyline/polygon needs >= 2 points")
    fill, stroke, sw = _common_style(elem, xf)
    # SVG semantics: FILL always treats the point list as implicitly closed
    # (a <polyline> fills exactly like a <polygon> would), but the STROKE
    # only follows the explicit segments — a polyline's stroke never closes
    # the last point back to the first; a polygon's always does.
    if fill is not None:
        draw.polygon(pts, fill=fill)
    if stroke is not None:
        stroke_pts = pts + [pts[0]] if closed else pts
        draw.line(stroke_pts, fill=stroke, width=max(1, round(sw)))


_PATH_TOKEN_RE = re.compile(r"([MmLlHhVvZz])|(" + _NUM_RE.pattern + r")")

_SUPPORTED_PATH_CMDS = set("MmLlHhVvZz")


def _parse_path_d(d: str) -> list:
    """Straight-only path grammar -> list of (points, closed) subpaths.
    Any command letter outside M/L/H/V/Z (case-insensitive) — the curve/arc
    family C/S/Q/T/A — raises immediately, by name, so an unsupported SVG
    can never silently mis-render (module contract)."""
    tokens = []
    for cmd_m, num_m in _PATH_TOKEN_RE.findall(d):
        if cmd_m:
            tokens.append(("cmd", cmd_m))
        elif num_m:
            tokens.append(("num", float(num_m)))
    # Guard: any letter token elsewhere in the string that our regex simply
    # skipped (e.g. "C", "A") would otherwise vanish silently — scan raw text.
    for ch in d:
        if ch.isalpha() and ch not in _SUPPORTED_PATH_CMDS:
            raise UnsupportedSvgError(
                f"unsupported path command {ch!r} in d={d!r} — only "
                f"M/L/H/V/Z (straight commands) are supported")

    subpaths = []
    cur_pts = []
    cur_closed = False
    x = y = 0.0
    start_x = start_y = 0.0
    i = 0
    cmd = None
    while i < len(tokens):
        kind, val = tokens[i]
        if kind == "cmd":
            cmd = val
            i += 1
            continue
        if cmd is None:
            raise UnsupportedSvgError(f"path data starts with a number: {d!r}")
        if cmd in "Mm":
            nx, ny = tokens[i][1], tokens[i + 1][1]
            i += 2
            if cmd == "m":
                nx, ny = x + nx, y + ny
            if cur_pts:
                subpaths.append((cur_pts, cur_closed))
            cur_pts = [(nx, ny)]
            cur_closed = False
            x, y = nx, ny
            start_x, start_y = x, y
            cmd = "L" if cmd == "M" else "l"  # implicit lineto after moveto
        elif cmd in "Ll":
            nx, ny = tokens[i][1], tokens[i + 1][1]
            i += 2
            if cmd == "l":
                nx, ny = x + nx, y + ny
            cur_pts.append((nx, ny))
            x, y = nx, ny
        elif cmd in "Hh":
            nx = tokens[i][1]
            i += 1
            if cmd == "h":
                nx = x + nx
            cur_pts.append((nx, y))
            x = nx
        elif cmd in "Vv":
            ny = tokens[i][1]
            i += 1
            if cmd == "v":
                ny = y + ny
            cur_pts.append((x, ny))
            y = ny
        elif cmd in "Zz":
            cur_closed = True
            x, y = start_x, start_y
            # Z takes no operands; do not advance i (there is none to skip).
        else:  # pragma: no cover — pre-filtered above
            raise UnsupportedSvgError(f"unsupported path command {cmd!r}")
    if cur_pts:
        subpaths.append((cur_pts, cur_closed))
    return subpaths


def _draw_path(draw, elem, xf: _Transform):
    d = elem["d"]
    fill, stroke, sw = _common_style(elem, xf)
    for pts, closed in _parse_path_d(d):
        px = [xf.pt(x, y) for x, y in pts]
        if len(px) < 2:
            continue
        if closed:
            if fill is not None:
                draw.polygon(px, fill=fill)
            if stroke is not None:
                draw.line(px + [px[0]], fill=stroke, width=max(1, round(sw)))
        else:
            if fill is not None:
                draw.polygon(px, fill=fill)
            if stroke is not None:
                draw.line(px, fill=stroke, width=max(1, round(sw)))


_FONT_CACHE = {}


def _font(size_px: int):
    size_px = max(1, size_px)
    if size_px not in _FONT_CACHE:
        _FONT_CACHE[size_px] = ImageFont.load_default(size=size_px)
    return _FONT_CACHE[size_px]


def _draw_text(draw, elem, xf: _Transform):
    x, y = float(elem["x"]), float(elem["y"])
    fill, _stroke, _sw = _common_style(elem, xf)
    font_size = xf.length(float(elem.get("font-size", "1")))
    glyph = (elem.text or "").strip()
    if not glyph:
        raise UnsupportedSvgError("<text> element has no glyph content")
    px, py = xf.pt(x, y)
    draw.text((px, py), glyph, font=_font(int(round(font_size))),
              fill=fill if fill is not None else (0, 0, 0, 255), anchor="mm")


_HANDLERS = {
    "rect": _draw_rect,
    "circle": _draw_circle,
    "ellipse": _draw_ellipse,
    "line": _draw_line,
    "polyline": lambda d, e, xf: _draw_poly(d, e, xf, closed=False),
    "polygon": lambda d, e, xf: _draw_poly(d, e, xf, closed=True),
    "path": _draw_path,
    "text": _draw_text,
}


class _AttrView(dict):
    """`.get`/`__getitem__` over an Element's attrib with a clearer
    KeyError message for a missing required attribute."""

    def __init__(self, elem):
        super().__init__(elem.attrib)
        self.text = elem.text

    def __getitem__(self, key):
        try:
            return dict.__getitem__(self, key)
        except KeyError:
            raise UnsupportedSvgError(
                f"missing required attribute {key!r} on element") from None


def rasterize_svg(path) -> Image.Image:
    """Parse + render one SVG file to a deterministic 64x64 RGBA
    :class:`PIL.Image.Image`. Raises :class:`UnsupportedSvgError` on any
    element/attribute/path-command outside the supported subset (module
    docstring) — never silently mis-renders."""
    path = Path(path)
    root = ET.parse(str(path)).getroot()
    if _tag(root) != "svg":
        raise UnsupportedSvgError(f"{path}: root element is not <svg>")
    view_box = root.get("viewBox")
    if not view_box:
        raise UnsupportedSvgError(f"{path}: <svg> has no viewBox")
    minx, miny, w, h = _floats(view_box)
    xf = _Transform(minx, miny, w, h)

    canvas = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    for child in root:
        tag = _tag(child)
        handler = _HANDLERS.get(tag)
        if handler is None:
            raise UnsupportedSvgError(
                f"{path}: unsupported element <{tag}> — supported subset is "
                f"{sorted(_HANDLERS)}")
        handler(draw, _AttrView(child), xf)

    return canvas.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.LANCZOS)


def regenerate_all(icons_dir: Path = ICONS_DIR) -> list:
    """Rasterize every ``<class>.svg`` in ``icons_dir`` to its sibling
    ``<class>.png``, overwriting. Returns the sorted list of class names
    processed. This is the dev-only CLI path — the editor RUNTIME never
    calls this, it only ever loads the already-committed PNGs."""
    svgs = sorted(icons_dir.glob("*.svg"))
    names = []
    for svg_path in svgs:
        img = rasterize_svg(svg_path)
        png_path = svg_path.with_suffix(".png")
        img.save(png_path)
        names.append(svg_path.stem)
    return names


def main(argv=None) -> int:
    names = regenerate_all()
    for name in names:
        print(f"rasterized {name}.svg -> {name}.png")
    print(f"{len(names)} icon(s) written to {ICONS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
