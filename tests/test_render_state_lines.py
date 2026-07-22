"""Headless tests for the B2 P5 active render-state HUD cluster
(renderer.game_renderer.render_state_lines) — design §7 P5.

The function is pure (plain str/bool/float in, ``(text, RGBA)`` pairs out) so
it is testable without a GL context: no ``GameRenderer`` instance (which would
need a real window) is constructed here — only the module-level helper,
matching the ``hover_readout.py`` / ``gas_medium.py`` pyray-free-core pattern.

Pinned invariants (the HUD annotations gate, design §7 P5):
  - Medium reads NEW/LEGACY off the ``legacy_smoke_on`` flag, one line only
    (no duplicate F9 row elsewhere — the fold this patch exists to do);
  - Detail's colour is "live" (green) ONLY when both enabled AND the legacy
    path isn't shadowing it — compose_world never samples gas_detail while
    legacy_smoke_on is True, so the HUD must not claim it is;
  - Speckle always shows the current mode name;
  - the non-physical gas-floor flag appears ONLY when raised AND the new
    (non-legacy) medium is active — legacy_smoke_on never reads the floor.

Run:
    C:/Users/steen/miniconda3/envs/data/python.exe -m pytest tests/test_render_state_lines.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))

from renderer.game_renderer import render_state_lines

ON = (180, 255, 180, 255)
OFF = (140, 140, 140, 255)
FLAG = (255, 200, 120, 255)


def _labels(lines):
    return [text for text, _color in lines]


def test_new_medium_default_reads_green():
    lines = render_state_lines(legacy_smoke_on=False, detail_enabled=True,
                               speckle_mode="soot", effect_gas_floor=0.0)
    text, color = lines[0]
    assert text == "Medium:  NEW gas-medium"
    assert color == ON


def test_legacy_medium_flags_amber():
    lines = render_state_lines(legacy_smoke_on=True, detail_enabled=True,
                               speckle_mode="soot", effect_gas_floor=0.0)
    text, color = lines[0]
    assert text == "Medium:  LEGACY smoke"
    assert color == FLAG


def test_only_one_medium_line_no_duplicate_f9_row():
    """The fold this patch does: exactly one Medium: line, never two, and
    the raw text never leaks the old 'F9' HUD spelling."""
    for legacy in (True, False):
        lines = render_state_lines(legacy_smoke_on=legacy, detail_enabled=True,
                                   speckle_mode="off", effect_gas_floor=0.0)
        medium_lines = [t for t in _labels(lines) if t.startswith("Medium:")]
        assert len(medium_lines) == 1
        assert all("F9" not in t for t in _labels(lines))


def test_detail_live_green_when_enabled_and_new_medium():
    lines = render_state_lines(legacy_smoke_on=False, detail_enabled=True,
                               speckle_mode="off", effect_gas_floor=0.0)
    text, color = next(l for l in lines if l[0].startswith("Detail:"))
    assert text == "Detail:  on"
    assert color == ON


def test_detail_greyed_when_disabled():
    lines = render_state_lines(legacy_smoke_on=False, detail_enabled=False,
                               speckle_mode="off", effect_gas_floor=0.0)
    text, color = next(l for l in lines if l[0].startswith("Detail:"))
    assert text == "Detail:  off"
    assert color == OFF


def test_detail_greyed_when_shadowed_by_legacy_even_if_enabled():
    """gas_detail never draws while legacy_smoke_on is True (compose_world
    bypasses it entirely) -- the HUD must not claim it is live."""
    lines = render_state_lines(legacy_smoke_on=True, detail_enabled=True,
                               speckle_mode="off", effect_gas_floor=0.0)
    text, color = next(l for l in lines if l[0].startswith("Detail:"))
    assert text == "Detail:  on"          # the raw flag state is still shown
    assert color == OFF                    # but greyed -- it isn't live


def test_speckle_shows_current_mode():
    for mode, expect_color in (("off", OFF), ("noise", ON), ("soot", ON)):
        lines = render_state_lines(legacy_smoke_on=False, detail_enabled=True,
                                   speckle_mode=mode, effect_gas_floor=0.0)
        text, color = next(l for l in lines if l[0].startswith("Speckle:"))
        assert text == f"Speckle: {mode}"
        assert color == expect_color


def test_gas_floor_flag_only_on_new_medium_when_raised():
    lines = render_state_lines(legacy_smoke_on=False, detail_enabled=True,
                               speckle_mode="off", effect_gas_floor=0.35)
    flag = [l for l in lines if l[0].startswith("Gas floor")]
    assert len(flag) == 1
    assert flag[0][0] == "Gas floor 0.35 (non-physical)"
    assert flag[0][1] == FLAG


def test_gas_floor_flag_absent_when_zero():
    lines = render_state_lines(legacy_smoke_on=False, detail_enabled=True,
                               speckle_mode="off", effect_gas_floor=0.0)
    assert not any(t.startswith("Gas floor") for t in _labels(lines))


def test_gas_floor_flag_suppressed_under_legacy_even_if_raised():
    """The legacy path never reads effect_gas_floor (gas_medium.py isn't even
    sampled) -- the HUD must not show a flag that isn't actually doing anything."""
    lines = render_state_lines(legacy_smoke_on=True, detail_enabled=True,
                               speckle_mode="off", effect_gas_floor=0.5)
    assert not any(t.startswith("Gas floor") for t in _labels(lines))


if __name__ == "__main__":
    test_new_medium_default_reads_green()
    test_legacy_medium_flags_amber()
    test_only_one_medium_line_no_duplicate_f9_row()
    test_detail_live_green_when_enabled_and_new_medium()
    test_detail_greyed_when_disabled()
    test_detail_greyed_when_shadowed_by_legacy_even_if_enabled()
    test_speckle_shows_current_mode()
    test_gas_floor_flag_only_on_new_medium_when_raised()
    test_gas_floor_flag_absent_when_zero()
    test_gas_floor_flag_suppressed_under_legacy_even_if_raised()
    print("OK — render_state_lines: medium/detail/speckle/gas-floor cluster")
