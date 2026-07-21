"""Unit tests for renderer/fire_lights.py — the B1 brightest-K fire lights.

Headless / pure-numpy: fire_lights.py + blackbody.py are imported in ISOLATION
(importlib from file) so neither touches renderer/__init__ -> pyray. No GL, no
breach_physics. Covers the design's P3 gate
(docs/fire_b1_blackbody_fire_lights_design_2026-07-21.md §5):
  - NMS correctness on a synthetic field,
  - the brightest-K cap respected,
  - source params in range + the structural heat/jitter zeroes,
  - zero sources when the field is cold.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np

_RENDERER = Path(__file__).resolve().parent.parent / "renderer"


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"_isolated_{name}", _RENDERER / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


blackbody = _load("blackbody")
fire_lights = _load("fire_lights")

BlackbodyRamp = blackbody.BlackbodyRamp
select_fire_light_params = fire_lights.select_fire_light_params
FireLightSelector = fire_lights.FireLightSelector
TEMP_SCALE = fire_lights.TEMP_SCALE
TAU = fire_lights.TAU

RAMP = BlackbodyRamp()


def _field(shape, hot=None):
    """Zero Q16.16 field of `shape`; hot = {(row, col): t_game}."""
    f = np.zeros(shape, dtype=np.int32)
    for (r, c), tg in (hot or {}).items():
        f[r, c] = int(round(tg * TEMP_SCALE))
    return f


# ---- zero sources when cold --------------------------------------------

def test_cold_field_no_lights():
    lights, n = select_fire_light_params(_field((20, 20)), RAMP)
    assert lights == []
    assert n == 0


def test_below_threshold_no_lights():
    # A warm-but-sub-threshold tile (t_light_min default 250) yields nothing.
    f = _field((10, 10), {(5, 5): 240.0})
    lights, n = select_fire_light_params(f, RAMP, t_light_min=250.0)
    assert lights == [] and n == 0
    # Nudge it above threshold -> one light.
    f2 = _field((10, 10), {(5, 5): 260.0})
    lights2, n2 = select_fire_light_params(f2, RAMP, t_light_min=250.0)
    assert len(lights2) == 1 and n2 == 1


# ---- NMS correctness ----------------------------------------------------

def test_nms_isolated_peaks():
    # Three isolated hot tiles (>= 2 apart) are each a strict 3x3 local max.
    f = _field((20, 20), {(3, 3): 1000.0, (3, 10): 1500.0, (12, 6): 2000.0})
    lights, n = select_fire_light_params(f, RAMP, nms_window=3)
    assert n == 3
    assert len(lights) == 3


def test_nms_suppresses_slope_to_single_peak():
    # A monotone bump: one true summit surrounded by strictly cooler tiles.
    f = np.zeros((9, 9), dtype=np.int32)
    for r in range(9):
        for c in range(9):
            d = max(abs(r - 4), abs(c - 4))          # Chebyshev distance
            if d <= 3:
                f[r, c] = int((3000 - d * 500) * TEMP_SCALE)   # 3000 at centre
    lights, n = select_fire_light_params(f, RAMP, nms_window=3, t_light_min=250.0)
    # Exactly the summit survives NMS (its 8 neighbours are all cooler).
    assert n == 1
    assert lights[0]["x"] == 4 + 0.5 and lights[0]["y"] == 4 + 0.5


def test_wider_nms_window_suppresses_more():
    # Two peaks 2 apart: distinct under 3x3 (radius 1), merged under 5x5
    # (radius 2 -> the cooler peak sits inside the hotter's window, so only the
    # hotter survives).
    f = _field((15, 15), {(7, 5): 1500.0, (7, 7): 1200.0})
    _, n3 = select_fire_light_params(f, RAMP, nms_window=3)
    _, n5 = select_fire_light_params(f, RAMP, nms_window=5)
    assert n3 == 2
    assert n5 == 1


# ---- brightest-K cap ----------------------------------------------------

def test_cap_respected_and_reports_peaks():
    # Nine isolated peaks, cap at 4: four lights emitted, n_peaks reports 9.
    hot = {}
    coords = [(r, c) for r in (1, 5, 9) for c in (1, 5, 9)]
    for i, rc in enumerate(coords):
        hot[rc] = 500.0 + 100.0 * i            # distinct temps
    f = _field((12, 12), hot)
    lights, n = select_fire_light_params(f, RAMP, max_lights=4)
    assert n == 9
    assert len(lights) == 4


def test_cap_picks_the_hottest():
    hot = {(1, 1): 500.0, (1, 6): 3000.0, (6, 1): 900.0, (6, 6): 2000.0}
    f = _field((10, 10), hot)
    lights, n = select_fire_light_params(f, RAMP, max_lights=2)
    assert n == 4 and len(lights) == 2
    kept = {(l["y"] - 0.5, l["x"] - 0.5) for l in lights}   # (row, col)
    assert kept == {(1, 6), (6, 6)}          # the two hottest tiles


def test_max_lights_zero_emits_nothing():
    f = _field((8, 8), {(4, 4): 1000.0})
    lights, n = select_fire_light_params(f, RAMP, max_lights=0)
    assert lights == []


# ---- source params in range + structural zeroes ------------------------

def test_source_params_valid():
    f = _field((10, 10), {(4, 4): 3000.0, (7, 7): 1200.0})
    lights, _ = select_fire_light_params(f, RAMP, light_range=18.0, light_gain=1.0)
    for l in lights:
        # tile-centre coords (matches cast_fire_heat's +0.5)
        assert l["x"] % 1.0 == 0.5 and l["y"] % 1.0 == 0.5
        assert l["max_range"] == 18.0
        assert math.isclose(l["angle_spread"], TAU)
        assert l["angle_center"] == 0.0
        r, g, b = l["color"]
        assert all(0.0 <= ch <= 1.0 for ch in (r, g, b))
        assert l["intensity"] >= 0.0
        # STRUCTURAL: render lights never write heat or pull jitter RNG.
        assert l["heat"] == 0.0
        assert l["jitter"] == 0.0


def test_light_gain_scales_intensity():
    f = _field((8, 8), {(4, 4): 2000.0})
    (l1,), _ = select_fire_light_params(f, RAMP, light_gain=1.0)
    (l2,), _ = select_fire_light_params(f, RAMP, light_gain=2.0)
    assert math.isclose(l2["intensity"], 2.0 * l1["intensity"], rel_tol=1e-6)


def test_field_not_mutated():
    f = _field((10, 10), {(4, 4): 3000.0})
    before = f.copy()
    select_fire_light_params(f, RAMP)
    assert np.array_equal(f, before)


# ---- FireLightSelector wrapper + config --------------------------------

def test_selector_disabled_emits_nothing():
    sel = FireLightSelector(enabled=False)
    f = _field((8, 8), {(4, 4): 3000.0})
    assert sel.select(f, RAMP) == ([], 0)


def test_selector_even_window_rejected():
    for bad in (2, 4, 0):
        try:
            FireLightSelector(nms_window=bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for nms_window={bad}")


def test_selector_from_config():
    class _NS:
        pass
    fl = _NS()
    fl.max_lights = 8
    fl.light_range = 25.0
    fl.nms_window = 5
    render = _NS()
    render.fire_lights = fl
    cfg = _NS()
    cfg.render = render
    sel = FireLightSelector.from_config(cfg)
    assert sel.max_lights == 8
    assert sel.light_range == 25.0
    assert sel.nms_window == 5
    assert sel.enabled is True          # default
    assert sel.t_light_min == 250.0     # default


def test_selector_from_config_missing_uses_defaults():
    class _NS:
        pass
    sel = FireLightSelector.from_config(_NS())
    assert sel.enabled is True
    assert sel.max_lights == 16
    assert sel.light_range == 18.0
