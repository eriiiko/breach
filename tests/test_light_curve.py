"""The light curve (graphics G5, #64): its map, its door, its file write and its
editor, headless. The GL half (the shaders run this map) is
``gl_light_curve_check.py``, through the ``test_light_curve_gl`` wrapper below.
"""
from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from renderer.light_curve import (IDENTITY_POINTS, LUT_SIZE, LightCurve,
                                  apply_curve, bake_lut, check_points,
                                  curve_gain, encode, light_curve_settings,
                                  light_histogram, luminance, write_settings)
from ui.light_curve_panel import CurvePanel, panel_layout, to_screen

ROOT = Path(__file__).resolve().parent.parent
STEP = ((0.0, 0.0), (0.5, 0.5), (0.52, 0.95), (1.0, 1.0))


def _curve(**over) -> LightCurve:
    base: dict = dict(enabled=True, points=IDENTITY_POINTS, range=4.0,
                dark_saturation=1.0, dark_knee=0.25, panel=True)
    base.update(over)
    return LightCurve(**base)


def test_the_identity_curve_changes_no_light_at_any_level():
    """PROPERTY: a straight curve is the identity, above ``range`` too.
    BREAKS IF: the axes' encode and decode disagree, or light past the top of
    the axis is clipped."""
    lut = bake_lut(IDENTITY_POINTS)
    light = np.array([[v, v * 0.5, v * 0.25] for v in
                      (1e-4, 0.01, 0.1, 1.0, 3.9, 4.0, 9.0, 80.0)])
    assert apply_curve(light, lut, 4.0) == pytest.approx(light, rel=2e-3)


def test_the_curve_keeps_the_colour_of_the_light():
    """PROPERTY: the map scales a pixel's light, so a red lamp stays red.
    BREAKS IF: the curve is applied per channel."""
    lut = bake_lut(STEP)
    light = np.array([[0.9, 0.2, 0.05], [0.05, 0.1, 0.7]])
    out = apply_curve(light, lut, 4.0)
    assert out / out[:, :1] == pytest.approx(light / light[:, :1])


def test_erik_threshold_keeps_the_dim_light_and_lifts_the_bright():
    """PROPERTY: below a step the light stays continuous, above it it is at the
    fully-lit level. BREAKS IF: the interpolation overshoots or smooths out
    the step, which is what a plain cubic spline would do."""
    lut = bake_lut(STEP)
    grey = np.ones(3)
    below = encode(0.5, 4.0)          # x ~ 0.39, under the step at 0.5
    assert below < 0.5
    out = apply_curve(grey * 0.5, lut, 4.0)
    assert out == pytest.approx(grey * 0.5, rel=0.05)
    above = apply_curve(grey * 1.5, lut, 4.0)
    assert luminance(above) > 3.0 * 0.6 > 1.5


@pytest.mark.parametrize("seed", range(6))
def test_the_curve_never_overshoots_its_control_points(seed):
    """PROPERTY: between two points the curve stays between their y.
    BREAKS IF: the tangents stop being limited (Fritsch-Carlson)."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 9))
    xs = np.unique(np.round(np.sort(rng.random(n)), 2))
    pts = list(zip(xs.tolist(), rng.random(len(xs)).tolist()))
    lut = bake_lut(pts, 2048)
    x = np.linspace(0.0, 1.0, 2048)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        seg = lut[(x >= x0) & (x <= x1)]
        assert seg.min() >= min(y0, y1) - 1e-5
        assert seg.max() <= max(y0, y1) + 1e-5


def test_a_tile_with_no_light_stays_at_the_flat_floor():
    """PROPERTY: light 0 is left alone even by a lifted curve: the dark's level
    is the flat floor's, and a division by 0 never reaches a pixel.
    BREAKS IF: the guard on L goes."""
    lut = bake_lut(((0.0, 0.5), (1.0, 1.0)))
    assert curve_gain(np.zeros(3), lut, 4.0) == 1.0


@pytest.mark.parametrize("points", [
    [[0.0, 0.0]], [[0.0, 0.0], [1.2, 1.0]], [[0.5, 0.0], [0.5, 1.0]],
    [[0.6, 0.0], [0.5, 1.0]], [[0.0, 0.0], [1.0, -0.1]], "no",
])
def test_the_door_refuses_a_bad_curve(points):
    """PROPERTY: a curve that cannot be baked is refused by name.
    BREAKS IF: ``check_points`` loses a case."""
    with pytest.raises(ValueError, match="points"):
        check_points(points)


def _section(**over):
    v = dict(enabled=False, panel=False, range=4.0, dark_saturation=1.0,
             dark_knee=0.25, points=[[0.0, 0.0], [1.0, 1.0]])
    v.update(over)
    return SimpleNamespace(render=SimpleNamespace(
        light_curve=SimpleNamespace(**v)))


@pytest.mark.parametrize("over", [
    {"enabled": 1}, {"panel": "yes"}, {"range": 0.0}, {"range": True},
    {"dark_saturation": 1.5}, {"dark_knee": 0.0},
])
def test_the_settings_door_refuses_a_mistyped_dial(over):
    """PROPERTY: a mistyped dial raises, it never falls back.
    BREAKS IF: a key's check is dropped."""
    with pytest.raises(ValueError, match="render.light_curve"):
        light_curve_settings(_section(**over))


def test_the_shipped_table_passes_the_door_with_the_editor_closed():
    """PROPERTY: the shipped curve is one the door takes, and the shipped game
    opens no tuning panel. BREAKS IF: a tuning session lands ``panel = true``
    (Save leaves that line to the hand) or a hand edit breaks the table."""
    with open(ROOT / "config.toml", "rb") as f:
        table = tomllib.load(f)["render"]["light_curve"]
    cfg = SimpleNamespace(render=SimpleNamespace(
        light_curve=SimpleNamespace(**table)))
    assert light_curve_settings(cfg).panel is False


def test_save_rewrites_only_its_three_lines(tmp_path):
    """PROPERTY: Save writes the curve back and keeps every other byte of
    config.toml, comments and line ends included, and the file re-parses to
    the saved curve. BREAKS IF: the writer re-serialises the toml or touches a
    neighbouring table's key of the same name."""
    src = (ROOT / "config.toml").read_bytes()
    path = tmp_path / "config.toml"
    path.write_bytes(src)
    write_settings(path, _curve(points=STEP, dark_saturation=0.15))
    out = path.read_bytes()
    table = tomllib.loads(out.decode("utf-8"))["render"]["light_curve"]
    assert table["enabled"] is True
    assert check_points(table["points"]) == STEP
    assert table["dark_saturation"] == pytest.approx(0.15)
    a, b = src.split(b"\n"), out.split(b"\n")
    assert len(a) == len(b)
    changed = [x for x, y in zip(a, b) if x != y]
    assert changed
    assert all(x.split(b"=")[0].strip() in
               (b"enabled", b"points", b"dark_saturation") for x in changed)
    assert all(x.split(b"#")[-1] == y.split(b"#")[-1]
               for x, y in zip(a, b) if x != y and b"#" in x)


def test_a_save_with_no_table_is_refused(tmp_path):
    """PROPERTY: Save never appends to a config it cannot place itself in.
    BREAKS IF: the writer falls through to a blind write."""
    path = tmp_path / "config.toml"
    path.write_text("[render]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no \\[render.light_curve\\]"):
        write_settings(path, _curve())


def test_the_histogram_counts_lit_tiles_only():
    """PROPERTY: the panel's histogram sums to 1 over the lit tiles and is
    empty in the dark. BREAKS IF: dark tiles swamp the first bin."""
    dark = np.zeros((8, 8, 3))
    assert light_histogram(dark, 4.0).sum() == 0.0
    dark[0, 0] = 1.0
    dark[3, 3] = 4.0
    h = light_histogram(dark, 4.0)
    assert h.sum() == pytest.approx(1.0)
    assert np.count_nonzero(h) == 2


def test_the_editor_adds_drags_and_deletes_a_point():
    """PROPERTY: press on the plot adds a point there and holds it, a drag
    moves it and cannot pass a neighbour, a right press deletes it, and the
    ends cannot be deleted. BREAKS IF: an edit leaves the points unsorted,
    which the door would refuse mid-drag."""
    lay = panel_layout(1280)
    panel = CurvePanel(_curve())
    sx, sy = to_screen(lay, 0.5, 0.25)
    panel.press(lay, sx, sy)
    assert len(panel.curve.points) == 3 and panel.drag == 1
    assert panel.saved is False
    panel.move(lay, *to_screen(lay, 1.0, 0.75))
    x, y = panel.curve.points[1]
    assert x < 1.0 and y == pytest.approx(0.75, abs=0.01)
    panel.release()
    panel.right_press(lay, *to_screen(lay, x, y))
    assert len(panel.curve.points) == 2
    panel.right_press(lay, *to_screen(lay, 0.0, 0.0))
    assert len(panel.curve.points) == 2


def test_the_panel_keeps_the_mouse_from_the_game():
    """PROPERTY: over the panel, and during any drag that left it, the mouse is
    the panel's. BREAKS IF: a drag off the panel starts ordering units."""
    lay = panel_layout(1280)
    panel = CurvePanel(_curve())
    assert panel.captures(lay, lay.plot.x + 3, lay.plot.y + 3)
    assert not panel.captures(lay, 10, 500)
    panel.press(lay, *to_screen(lay, 0.5, 0.5))
    assert panel.captures(lay, 10, 500)
    panel.release()
    assert not panel.captures(lay, 10, 500)


def test_reset_is_the_identity_and_the_buttons_act():
    """PROPERTY: the on button flips the curve, Reset restores the identity
    and full colour, Save asks the shell to write. BREAKS IF: a button's rect
    moves off its action."""
    lay = panel_layout(1280)
    panel = CurvePanel(_curve(points=STEP, dark_saturation=0.2))
    panel.press(lay, lay.toggle.x + 1, lay.toggle.y + 1)
    assert panel.curve.enabled is False
    panel.press(lay, lay.reset.x + 1, lay.reset.y + 1)
    assert panel.curve.points == IDENTITY_POINTS
    assert panel.curve.dark_saturation == 1.0
    assert panel.press(lay, lay.save.x + 1, lay.save.y + 1) == "save"


def test_the_lut_is_the_size_the_shader_declares():
    """PROPERTY: the array the shader indexes is the array the bake fills.
    BREAKS IF: ``LUT_SIZE`` and the GLSL's ``u_curve_lut[256]`` drift."""
    from renderer.lit3d import _LIGHT_CURVE_GLSL
    assert f"u_curve_lut[{LUT_SIZE}]" in _LIGHT_CURVE_GLSL
    assert f"{float(LUT_SIZE - 1)}" in _LIGHT_CURVE_GLSL


def test_light_curve_gl():
    """The GL half, in its own process (the ``gl_*_check`` convention)."""
    r = subprocess.run(
        [sys.executable, str(ROOT / "tests" / "gl_light_curve_check.py")],
        capture_output=True, text=True, cwd=ROOT, timeout=120)
    lines = r.stdout.strip().splitlines()
    assert lines and lines[-1] == "LIGHT_CURVE_GL_RESULT: PASS", (
        r.stdout[-2000:] + r.stderr[-2000:])
