"""#59 — the B key stops toggling bilinear filtering.

Bilinear vs nearest on the light texture used to be a live B-key toggle in
``GameRenderer.poll_debug_toggles`` (``renderer/game_renderer.py``), which
collides with ``ORDER_EXPLOSIVE``'s B binding in ``input_handler.py`` --
selecting/arming an explosive would also flip the filter mode, confounding
every look judgement in a session where explosives are armed throughout
(Erik's ruling 2026-09-29: fewer shortcut keys, a switch is a config.toml
setting). The property this test protects: pressing B in the debug-key block
touches nothing on ``self.lighting`` at all -- no toggle exists any more, on
any code path. A regression that reintroduces a `KEY_B` branch calling into
`self.lighting` (whether `toggle_bilinear` or anything else) breaks it.

Pattern: ``poll_debug_toggles`` is unbound-called on a minimal stand-in
object (no GL context / window needed -- the same style
``tests/test_onephase_control.py`` uses for ``control_onephase``), with the
module's ``rl`` swapped for a fake that reports only the key presses a test
sets, per ``renderer/game_renderer.py``'s own docstring convention.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pytest

from renderer import game_renderer


class _Keys:
    """Key ids as plain strings (KEY_B -> "B"), matching
    test_onephase_control.py's FakeRl convention."""

    def __getattr__(self, name):
        if name.startswith("KEY_"):
            return name[4:]
        raise AttributeError(name)


class FakeRl:
    KeyboardKey = _Keys()

    def __init__(self, pressed):
        self._pressed = set(pressed)

    def is_key_pressed(self, key):
        return key in self._pressed

    def is_key_down(self, key):
        return False


class _NoTouchLighting:
    """Any attribute access is a test failure: poll_debug_toggles must not
    reach ``self.lighting`` at all while only B is pressed (no branch keys
    off B any more)."""

    def __getattr__(self, name):
        raise AssertionError(
            f"poll_debug_toggles touched self.lighting.{name} on a lone "
            "B keypress -- the B/bilinear branch must be gone entirely"
        )


@pytest.fixture
def fake_rl(monkeypatch):
    def _make(pressed):
        fake = FakeRl(pressed)
        monkeypatch.setattr(game_renderer, "rl", fake)
        return fake
    return _make


def _bare_renderer():
    """A GameRenderer built with object.__new__: no window, no GPU -- only
    the attribute poll_debug_toggles is documented to touch when B alone is
    pressed (self.lighting)."""
    r = object.__new__(game_renderer.GameRenderer)
    r.lighting = _NoTouchLighting()
    return r


def test_b_key_press_never_touches_lighting(fake_rl):
    fake_rl({"B"})
    r = _bare_renderer()
    # Must not raise: poll_debug_toggles has no B branch left, so it never
    # reaches self.lighting for any reason on this keypress.
    r.poll_debug_toggles()


def test_b_key_still_does_not_exist_as_a_toggle_branch():
    """A direct source-level property, redundant with the behavioural test
    above but cheap: no code path in poll_debug_toggles fetches
    ``rl.KeyboardKey.KEY_B`` (the toggle branch was deleted, not merely
    neutered elsewhere)."""
    import inspect
    src = inspect.getsource(game_renderer.GameRenderer.poll_debug_toggles)
    assert "KEY_B" not in src, (
        "poll_debug_toggles still references KEY_B -- the #59 debug-key "
        "branch for bilinear filtering was supposed to be deleted, not "
        "just disconnected from toggle_bilinear"
    )
