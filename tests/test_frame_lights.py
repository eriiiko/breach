"""Equivalence gate for renderer/frame_lights.py (B2 P1).

Render behaviour is NOT digest-caught, so the sources-block extraction (main.py's
per-frame light assembly -> the shared helper) is proven here EXPLICITLY: the
pre-extraction inline logic is transcribed verbatim as an independent ORACLE and
compared field-for-field, in order, against
:func:`renderer.frame_lights.build_frame_light_sources` for a known
(statics, beacon-angle/tick, fire set, fire on/off) input. If the helper ever
drifts from what main.py used to do, this fails.

Not a headless-window test — it needs breach_physics (the compiled
``LightSource`` struct) but no GL context, so it runs in the normal
``pytest tests -q`` suite.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_frame_lights.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import breach_physics as bp  # noqa: E402
from level_loader import LightEntry  # noqa: E402
from level_lights import light_source_params, monotonic_total_tick  # noqa: E402
from renderer.blackbody import BlackbodyRamp  # noqa: E402
from renderer.fire_lights import FireLightSelector, TEMP_SCALE  # noqa: E402
from renderer.frame_lights import (  # noqa: E402
    build_frame_light_sources, build_static_light_sources)

GRID_W, GRID_H = 40, 30
DT = 1.0 / 24.0                       # sim_time_per_tick @ 24 Hz
RAMP = BlackbodyRamp()

# Attributes the pybind LightSource exposes (introspected) — the full struct.
_ATTRS = ("x", "y", "max_range", "intensity", "color", "angle_center",
          "angle_spread", "heat", "jitter", "ray_count")


def _snap(src):
    """LightSource -> comparable tuple of every bound attribute."""
    return tuple(tuple(getattr(src, a)) if a == "color" else getattr(src, a)
                 for a in _ATTRS)


def _snaps(sources):
    return [_snap(s) for s in sources]


def _known_lights():
    """Two static lamps (in-bounds) + one beacon (in-bounds) + one off-grid."""
    return [
        LightEntry(x=6.5, y=4.5, color=(1.0, 0.1, 0.05), intensity=0.9,
                   range=18.0, kind="static"),
        LightEntry(x=30.5, y=4.5, color=(0.6, 0.7, 1.0), intensity=1.2,
                   range=14.0, kind="static"),
        LightEntry(x=18.5, y=12.5, color=(1.0, 0.63, 0.16), intensity=3.0,
                   range=12.0, kind="beacon", period_s=2.0, beam_deg=30.0,
                   phase=0.0),
        # Off the 40x30 grid — must be partitioned out (never built).
        LightEntry(x=100.0, y=200.0, color=(1.0, 1.0, 1.0), intensity=1.0,
                   range=10.0, kind="static"),
    ]


def _hot_temp_field():
    """A Q16.16 temperature field with two well-separated hot peaks."""
    f = np.zeros((GRID_H, GRID_W), dtype=np.int32)
    f[10, 8] = int(round(900.0 * TEMP_SCALE))
    f[20, 25] = int(round(1500.0 * TEMP_SCALE))
    return f


def _old_inline(static_lights, beacon_lights, total_tick, selector,
                temp_field, show_fire_lights):
    """The pre-extraction main.py per-frame block, transcribed VERBATIM.

    This is the golden oracle: it uses its OWN setattr builder (never the
    helper's) so a drift in the helper cannot hide behind shared code.
    """
    def _old_build(params):
        src = bp.LightSource()
        for key, val in params.items():   # pybind class: setattr only
            setattr(src, key, val)
        return src

    sources = list(static_lights)
    if beacon_lights:
        total = total_tick
        sources += [
            _old_build(light_source_params(e, total, DT))
            for e in beacon_lights
        ]
    if show_fire_lights:
        fire_params, fire_peaks = selector.select(temp_field, RAMP)
    else:
        fire_params, fire_peaks = [], 0
    sources += [_old_build(p) for p in fire_params]
    return sources, len(fire_params), fire_peaks


def _setup():
    lights = _known_lights()
    static_lights, beacon_lights, off_grid = build_static_light_sources(
        bp, lights, GRID_W, GRID_H, DT)
    selector = FireLightSelector(enabled=True, t_light_min=250.0, nms_window=3,
                                 max_lights=8, light_range=18.0, light_gain=1.0)
    return static_lights, beacon_lights, off_grid, selector


def test_static_partition_drops_off_grid():
    static_lights, beacon_lights, off_grid, _ = _setup()
    # 2 in-bounds statics built; 1 beacon entry held; 1 off-grid skipped.
    assert len(static_lights) == 2
    assert len(beacon_lights) == 1 and beacon_lights[0].kind == "beacon"
    assert len(off_grid) == 1 and off_grid[0].x == 100.0


def test_frame_equals_old_inline_fire_on():
    static_lights, beacon_lights, _, selector = _setup()
    temp = _hot_temp_field()
    total_tick = monotonic_total_tick(turn_number=3, ticks_per_round=48, tick=17)

    exp_sources, exp_count, exp_peaks = _old_inline(
        static_lights, beacon_lights, total_tick, selector, temp,
        show_fire_lights=True)
    frame = build_frame_light_sources(
        bp, static_lights, beacon_lights, total_tick=total_tick,
        sim_time_per_tick=DT, fire_selector=selector, temperature_field=temp,
        blackbody_ramp=RAMP, show_fire_lights=True)

    assert _snaps(frame.sources) == _snaps(exp_sources)
    assert (frame.fire_count, frame.fire_peaks) == (exp_count, exp_peaks)
    # Sanity: statics + beacon + 2 fire peaks are all present.
    assert len(frame.sources) == 2 + 1 + 2
    assert frame.fire_count == 2


def test_frame_equals_old_inline_fire_off():
    static_lights, beacon_lights, _, selector = _setup()
    temp = _hot_temp_field()
    total_tick = monotonic_total_tick(turn_number=1, ticks_per_round=48, tick=0)

    exp_sources, exp_count, exp_peaks = _old_inline(
        static_lights, beacon_lights, total_tick, selector, temp,
        show_fire_lights=False)
    frame = build_frame_light_sources(
        bp, static_lights, beacon_lights, total_tick=total_tick,
        sim_time_per_tick=DT, fire_selector=selector, temperature_field=temp,
        blackbody_ramp=RAMP, show_fire_lights=False)

    assert _snaps(frame.sources) == _snaps(exp_sources)
    assert (frame.fire_count, frame.fire_peaks) == (0, 0)
    assert len(frame.sources) == 2 + 1        # statics + beacon only


def test_beacon_angle_advances_with_tick():
    """The beacon's angle_center is a pure function of the monotonic tick —
    the drift-critical property the studio depends on (sweeping beam)."""
    static_lights, beacon_lights, _, selector = _setup()
    temp = _hot_temp_field()

    def beacon_angle_at(tick):
        frame = build_frame_light_sources(
            bp, static_lights, beacon_lights,
            total_tick=monotonic_total_tick(1, 48, tick), sim_time_per_tick=DT,
            fire_selector=selector, temperature_field=temp, blackbody_ramp=RAMP,
            show_fire_lights=False)
        # sources[-1] is the beacon (statics first, then beacon, no fire).
        return frame.sources[-1].angle_center

    a0, a1 = beacon_angle_at(0), beacon_angle_at(6)
    assert a0 != a1, "beacon must sweep as the sim tick advances"


if __name__ == "__main__":
    test_static_partition_drops_off_grid()
    test_frame_equals_old_inline_fire_on()
    test_frame_equals_old_inline_fire_off()
    test_beacon_angle_advances_with_tick()
    print("OK — frame_lights == pre-extraction inline path (statics/beacon/fire)")
