"""tools/rasterize_icons.py — icon-rasterizer subset + PNG-freshness gate
(Arc C8: icons pipeline, editor doc §8).

Pins:
  - every committed ``art/entities/icons/<class>.png`` is byte-identical to
    re-rasterizing its ``<class>.svg`` sibling RIGHT NOW — editing an SVG
    without regenerating its PNG (``python tools/rasterize_icons.py``), or a
    stale/missing PNG, FAILS this test. This is the freshness gate the
    kickoff doc asks for; it runs in CI (PIL is available, no skip needed).
  - every committed SVG stays within the rasterizer's supported subset — the
    renderer raising IS the test failing, with a clear message, not a
    silent partial render.
  - the constrained-subset renderer itself: supported primitives render
    without error; unsupported elements/path-commands/colors raise
    ``UnsupportedSvgError`` loudly rather than mis-rendering.
  - output contract: fixed 64x64 RGBA, deterministic (two rasterizations of
    the same SVG are pixel-identical).

Run:
    python -m pytest tests/test_rasterize_icons.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import rasterize_icons as ri  # noqa: E402

ICONS_DIR = ROOT / "art" / "entities" / "icons"


# ---------------------------------------------------------------------------
# Freshness: every committed PNG must equal a fresh re-rasterization of its
# SVG, right now. This is the gate that makes "edit the SVG, forget to
# regenerate" impossible to merge.
# ---------------------------------------------------------------------------

def _committed_svgs():
    svgs = sorted(ICONS_DIR.glob("*.svg"))
    assert svgs, f"no icon SVGs found under {ICONS_DIR}"
    return svgs


@pytest.mark.parametrize("svg_path", _committed_svgs(), ids=lambda p: p.stem)
def test_committed_png_matches_fresh_rasterization(svg_path):
    png_path = svg_path.with_suffix(".png")
    assert png_path.is_file(), (
        f"{svg_path.name} has no committed PNG sibling — run "
        f"`python tools/rasterize_icons.py` and commit the result")
    committed = Image.open(png_path).convert("RGBA")
    fresh = ri.rasterize_svg(svg_path)
    assert committed.size == fresh.size, (
        f"{png_path.name}: committed size {committed.size} != fresh "
        f"rasterization size {fresh.size} (stale PNG)")
    assert list(committed.getdata()) == list(fresh.getdata()), (
        f"{png_path.name} is STALE — its pixels don't match rasterizing "
        f"{svg_path.name} right now. Run `python tools/rasterize_icons.py` "
        f"and commit the regenerated PNG.")


@pytest.mark.parametrize("svg_path", _committed_svgs(), ids=lambda p: p.stem)
def test_committed_svg_stays_within_supported_subset(svg_path):
    """A committed SVG that needs an unsupported feature must fail HERE,
    loudly, with the offending element/command named — never silently
    mis-render (module contract)."""
    ri.rasterize_svg(svg_path)  # raises UnsupportedSvgError on any violation


def test_every_registry_class_with_a_committed_png_has_a_valid_stem():
    """PNG filename (minus extension) must be a real registry class name —
    the palette maps class -> icon purely by filename, so a typo here is a
    silent no-op (the icon just never gets picked up)."""
    import json
    registry = json.loads((ROOT / "entity_registry.json").read_text(
        encoding="utf-8"))
    classes = set(registry["classes"])
    for png_path in sorted(ICONS_DIR.glob("*.png")):
        assert png_path.stem in classes, (
            f"{png_path.name}: {png_path.stem!r} is not a registered "
            f"entity class — check for a typo (registry has: "
            f"{sorted(classes)})")


# ---------------------------------------------------------------------------
# Rasterizer subset — direct unit coverage on synthetic SVG text, independent
# of whatever happens to be committed under art/entities/icons/.
# ---------------------------------------------------------------------------

def _write_svg(tmp_path, body: str, *, view_box: str = "0 0 24 24") -> Path:
    svg = tmp_path / "case.svg"
    svg.write_text(f'<svg viewBox="{view_box}">{body}</svg>',
                   encoding="utf-8")
    return svg


def test_output_is_fixed_size_rgba(tmp_path):
    svg = _write_svg(tmp_path, '<rect x="2" y="2" width="20" height="20" '
                               'fill="#ff0000"/>')
    img = ri.rasterize_svg(svg)
    assert img.size == (ri.OUTPUT_SIZE, ri.OUTPUT_SIZE)
    assert img.mode == "RGBA"


def test_deterministic_two_rasterizations_are_pixel_identical(tmp_path):
    svg = _write_svg(tmp_path, '''
        <circle cx="12" cy="12" r="9" fill="#3388ff" stroke="#000000"
                stroke-width="1"/>
        <polygon points="12,4 20,12 12,20 4,12" fill="none"
                 stroke="#ffffff" stroke-width="1.2"/>
    ''')
    a = ri.rasterize_svg(svg)
    b = ri.rasterize_svg(svg)
    assert list(a.getdata()) == list(b.getdata())


@pytest.mark.parametrize("body", [
    '<rect x="4" y="4" width="16" height="16" fill="#ff0000"/>',
    '<rect x="4" y="4" width="16" height="16" rx="3" fill="#00ff00" '
    'stroke="#000000" stroke-width="1"/>',
    '<circle cx="12" cy="12" r="8" fill="#0000ff"/>',
    '<ellipse cx="12" cy="12" rx="9" ry="5" fill="#ff00ff"/>',
    '<line x1="2" y1="2" x2="22" y2="22" stroke="#ffffff" '
    'stroke-width="2"/>',
    '<polyline points="2,2 12,20 22,2" fill="none" stroke="#ffff00" '
    'stroke-width="1"/>',
    '<polygon points="12,2 22,20 2,20" fill="#00ffff"/>',
    '<path d="M2,2 L22,2 L22,22 L2,22 Z" fill="#888888"/>',
    '<path d="M2,2 h20 v20 h-20 Z" fill="#444444"/>',
    '<text x="12" y="12" font-size="10" fill="#ffffff">A</text>',
])
def test_supported_primitives_render_without_error(tmp_path, body):
    svg = _write_svg(tmp_path, body)
    img = ri.rasterize_svg(svg)
    assert img.size == (ri.OUTPUT_SIZE, ri.OUTPUT_SIZE)


def test_bezier_path_command_raises(tmp_path):
    svg = _write_svg(tmp_path, '<path d="M2,2 C4,4 6,6 8,8 Z" fill="#fff" '
                               'stroke="none"/>')
    with pytest.raises(ri.UnsupportedSvgError, match="path command"):
        ri.rasterize_svg(svg)


def test_arc_path_command_raises(tmp_path):
    svg = _write_svg(tmp_path, '<path d="M2,2 A5,5 0 0 1 8,8" fill="none" '
                               'stroke="#fff" stroke-width="1"/>')
    with pytest.raises(ri.UnsupportedSvgError, match="path command"):
        ri.rasterize_svg(svg)


def test_unsupported_element_raises(tmp_path):
    svg = _write_svg(tmp_path, '<g><rect x="2" y="2" width="10" height="10" '
                               'fill="#fff"/></g>')
    with pytest.raises(ri.UnsupportedSvgError, match="unsupported element"):
        ri.rasterize_svg(svg)


def test_missing_viewbox_raises(tmp_path):
    svg = tmp_path / "case.svg"
    svg.write_text('<svg><rect x="0" y="0" width="10" height="10" '
                   'fill="#fff"/></svg>', encoding="utf-8")
    with pytest.raises(ri.UnsupportedSvgError, match="viewBox"):
        ri.rasterize_svg(svg)


def test_non_square_viewbox_raises(tmp_path):
    svg = _write_svg(tmp_path,
                     '<rect x="0" y="0" width="10" height="10" fill="#fff"/>',
                     view_box="0 0 24 12")
    with pytest.raises(ri.UnsupportedSvgError, match="square"):
        ri.rasterize_svg(svg)


def test_unsupported_color_literal_raises(tmp_path):
    svg = _write_svg(tmp_path, '<rect x="2" y="2" width="10" height="10" '
                               'fill="cornflowerblue"/>')
    with pytest.raises(ri.UnsupportedSvgError, match="color"):
        ri.rasterize_svg(svg)


def test_missing_required_attribute_raises(tmp_path):
    svg = _write_svg(tmp_path, '<circle cx="12" cy="12" fill="#fff"/>')
    with pytest.raises(ri.UnsupportedSvgError):
        ri.rasterize_svg(svg)


def test_hex3_color_expands_like_hex6():
    assert ri._parse_color("#f00") == ri._parse_color("#ff0000")


def test_polygon_is_implicitly_closed_for_both_fill_and_stroke(tmp_path):
    """A polygon's stroke DOES close back to the first point (unlike a
    polyline) — draw a polygon whose declared points don't touch the last
    edge and confirm the closing edge pixel is stroked."""
    # A thin polygon along the top+right+bottom edges only; if the LEFT
    # closing edge were not drawn, the leftmost column would stay empty.
    svg = _write_svg(tmp_path,
                     '<polygon points="4,4 20,4 20,20 4,20" fill="none" '
                     'stroke="#ffffff" stroke-width="2"/>')
    img = ri.rasterize_svg(svg)
    px = img.load()
    # Sample near the left edge (x~4 in a 0..24 viewBox -> scaled into the
    # 64x64 output) for a non-transparent (stroked) pixel.
    left_col_x = round(4 / 24 * ri.OUTPUT_SIZE)
    mid_y = ri.OUTPUT_SIZE // 2
    assert px[left_col_x, mid_y][3] > 0, (
        "polygon's implicit closing edge (last point -> first point) was "
        "not stroked")


def test_open_path_stroke_does_not_close(tmp_path):
    """An open <path> (no trailing Z) must NOT have its stroke closed back
    to the start — only <polygon> and Z-closed paths close their stroke."""
    svg = _write_svg(tmp_path,
                     '<path d="M4,4 L20,4 L20,20 L4,20" fill="none" '
                     'stroke="#ffffff" stroke-width="2"/>')
    img = ri.rasterize_svg(svg)
    px = img.load()
    left_col_x = round(4 / 24 * ri.OUTPUT_SIZE)
    mid_y = ri.OUTPUT_SIZE // 2
    assert px[left_col_x, mid_y][3] == 0, (
        "open path's stroke should NOT close back to its start point")


def test_regenerate_all_writes_every_svg_sibling(tmp_path):
    icons_dir = tmp_path / "icons"
    icons_dir.mkdir()
    (icons_dir / "widget.svg").write_text(
        '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="8" '
        'fill="#123456"/></svg>', encoding="utf-8")
    names = ri.regenerate_all(icons_dir)
    assert names == ["widget"]
    assert (icons_dir / "widget.png").is_file()
    img = Image.open(icons_dir / "widget.png")
    assert img.size == (ri.OUTPUT_SIZE, ri.OUTPUT_SIZE)
