"""tools/lighting_demo.py's right panel scrolls (#97 follow-up, 2026-11-21).

Found live: on a 1920x1200 screen only the first ~20 of the "Water" sliders
were on the window; every slider below (swell, foam, wake, height, among them
swell_strength_still) was off it and could not be reached. The panel is
immediate-mode raygui (a running y), so these tests run the REAL `_draw_panel`
against a recording stand-in for pyray (no window) and read where it put each
slider.

Run:
    C:/Users/steen/anaconda3/envs/data/python.exe -m pytest tests/test_lighting_demo_panel_scroll.py -q
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pyray as real_rl
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))

_DEMO_PATH = ROOT / "tools" / "lighting_demo.py"


def _load_demo():
    spec = importlib.util.spec_from_file_location("_lighting_demo_scroll",
                                                  _DEMO_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


demo = _load_demo()

VIEW_H = 600          # a small window: far shorter than the panel's content
MAP_W = 800
PANEL_W = demo.PANEL_W


class _RecordingRl:
    """pyray stand-in: constructors and ffi are the real ones; every draw and
    gui call is recorded; the mouse is wherever the test puts it."""

    def __init__(self):
        self.sliders = []        # (label_text_before, Rectangle) per slider bar
        self.scissors = []       # (x, y, w, h) per begin_scissor_mode
        self.open_scissors = 0
        self.mouse_x = MAP_W + 50
        self.wheel = 0.0
        self._last_label = ""

    def __getattr__(self, name):
        return getattr(real_rl, name)

    def get_mouse_x(self):
        return self.mouse_x

    def get_mouse_wheel_move(self):
        return self.wheel

    def get_fps(self):
        return 60

    def draw_rectangle(self, *_a):
        pass

    def draw_line_ex(self, *_a):
        pass

    def draw_text(self, *_a):
        pass

    def gui_label(self, _rect, text):
        self._last_label = text

    def gui_slider_bar(self, rect, _l, _r, _ptr, _lo, _hi):
        self.sliders.append((self._last_label, rect))

    def gui_check_box(self, *_a):
        return False

    def gui_button(self, *_a):
        return False

    def gui_text_box(self, *_a):
        return False

    def gui_dropdown_box(self, *_a):
        return False

    def begin_scissor_mode(self, x, y, w, h):
        self.scissors.append((x, y, w, h))
        self.open_scissors += 1

    def end_scissor_mode(self):
        self.open_scissors -= 1


@pytest.fixture
def panel(monkeypatch):
    fake = _RecordingRl()
    monkeypatch.setattr(demo, "rl", fake)
    monkeypatch.setattr(demo, "list_presets", lambda: [])
    renderer = SimpleNamespace(
        cfg=SimpleNamespace(map_px_w=MAP_W, panel_px_w=PANEL_W,
                            map_px_h=VIEW_H),
        legacy_smoke_on=False, last_light_ms=0.0,
        gas_detail=SimpleNamespace(enabled=False),
        speckle=SimpleNamespace(mode=0),
        gas_medium=SimpleNamespace(effect_gas_floor=0.0))
    state = demo.PanelState()

    def draw():
        fake.sliders.clear()
        demo._draw_panel(state, renderer, 0.0)
        return list(fake.sliders)

    return fake, state, draw


def _content_rows(state, draw):
    """Every slider's top in CONTENT coordinates (drawn at scroll 0)."""
    state.scroll = 0.0
    rows = draw()          # first frame also measures content_h
    return [(label, rect.y) for label, rect in rows]


def test_content_taller_than_window_is_measured(panel):
    """Property: the draw measures the content's height, which includes the
    last slider and exceeds the small window (so there is something to
    scroll). Breaks if content_h is not updated from the drawn y."""
    fake, state, draw = panel
    rows = _content_rows(state, draw)
    assert rows[-1][1] + demo.SLIDER_H <= state.content_h
    assert state.content_h > VIEW_H
    assert fake.open_scissors == 0


def test_every_slider_can_be_scrolled_into_view(panel):
    """Property: with a window far shorter than the panel, EVERY slider
    (every group, the water dial `swell_strength_still` among them) has a
    legal scroll offset (inside the clamp) at which its whole row lies inside
    the visible rect. Breaks if the clamp's bound is shorter than the drawn
    content (the last rows unreachable), if the offset is not applied to the
    rows, or if a group is drawn outside the measured run."""
    fake, state, draw = panel
    rows = _content_rows(state, draw)
    drawn_labels = {label for label, _y in rows}
    for _k, _d, label, _lo, _hi in demo._WATER_DIALS:
        assert label in drawn_labels            # no slider lost to the panel
    water_key = next(label for k, _d, label, _lo, _hi in demo._WATER_DIALS
                     if k == "swell_strength_still")
    assert water_key in drawn_labels

    for label, content_y in rows:
        want = demo.clamp_panel_scroll(content_y - demo.PANEL_PAD,
                                       state.content_h, VIEW_H)
        state.scroll = want
        shown = [r for lab, r in draw() if lab == label]
        assert shown, label
        # draw may clamp again; the legal offset must be a fixed point
        assert state.scroll == pytest.approx(want, abs=1.0)
        top = shown[0].y
        assert top >= 0 and top + demo.SLIDER_H <= VIEW_H, (label, top)


def test_wheel_over_panel_scrolls_and_is_clamped(panel):
    """Property: a wheel turn over the panel moves the content by whole
    notches, the offset never leaves [0, content_h - view_h], and a wheel over
    the MAP does not move the panel. Breaks if the clamp is dropped or the
    mouse-over test is."""
    fake, state, draw = panel
    _content_rows(state, draw)
    max_scroll = state.content_h - VIEW_H

    fake.wheel = -1.0                         # one notch down
    draw()
    assert state.scroll == pytest.approx(demo.PANEL_SCROLL_STEP)

    fake.wheel = 1.0                          # back up, then far past the top
    for _ in range(5):
        draw()
    assert state.scroll == 0.0

    fake.wheel = -1.0                         # far past the bottom
    for _ in range(int(max_scroll // demo.PANEL_SCROLL_STEP) + 20):
        draw()
    assert state.scroll == pytest.approx(max_scroll, abs=1.0)

    state.scroll = 100.0
    fake.mouse_x = MAP_W - 1                  # over the map
    fake.wheel = -1.0
    draw()
    assert state.scroll == 100.0


def test_panel_is_clipped_to_its_strip(panel):
    """Property: the content is drawn inside a scissor of exactly the panel's
    rect (so scrolled content never paints over the map or off the window).
    Breaks if the scissor is dropped or sized differently."""
    fake, state, draw = panel
    draw()
    assert fake.scissors == [(MAP_W, 0, PANEL_W, VIEW_H)]


def test_clamp_is_pure_and_total():
    """Property: clamp lands in [0, max(0, content - view)] for any input,
    including content shorter than the view."""
    c = demo.clamp_panel_scroll
    assert c(-5, 1000, 600) == 0
    assert c(10_000, 1000, 600) == 400
    assert c(50, 300, 600) == 0
