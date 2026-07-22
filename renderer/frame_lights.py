"""Per-frame raycaster ``LightSource`` assembly — shared by the game + harness.

Fire & Heat Beauty arc, beat B2 patch P1 (design
docs/fire_b2_smoke_honesty_design_2026-07-21.md §2). This module deduplicates
the ONE thing main.py and tools/lighting_demo.py must not let drift apart: the
per-frame assembly of the raycaster light-source list from whatever supplies
lights today — the levels-w1 ``[[light]]`` rows (static lamps + the rotating
beacon; see :mod:`level_lights`) and the B1 fire tiles
(:class:`renderer.fire_lights.FireLightSelector`).

**NOT an entity system** (Erik's explicit concern, 2026-07-22): this creates
ZERO entity machinery. Lamps/beacons are NOT entities today — they predate the
Arc A/B entity layer (doors/sensors/nodes/pump/airlock). Migrating lights INTO
the entity system is Arc C's convergence item; when it happens only this
helper's INPUT changes and the assembly seam survives. Do not build any
lamp/beacon entity here.

Deliberately importable WITHOUT the renderer package touching this module's
consumers: it takes the compiled ``breach_physics`` module as an argument
(never imports it) and imports only :mod:`level_lights` (plain math on plain
data), so the assembly is headless-testable in isolation. The RENDERER never
writes sim fields; this only READS a temperature field to select fire lights.

The caller-specific tail — the mouse flashlight and the W6 transient emitters —
stays in each caller (they differ: main.py's flashlight is fixed, the harness's
is slider-driven), matching the pre-extraction seam (main.py's sources block was
lines ~313-407; the flashlight/transient appends followed it). This helper
covers exactly the shared, drift-critical core: level statics + beacons + fire.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from level_lights import light_source_params, monotonic_total_tick, partition_lights


def build_light_source(bp, params: dict):
    """``dict`` -> ``bp.LightSource`` via setattr.

    ``bp.LightSource`` is a pybind class with only ``py::init<>()`` — it takes
    no keyword constructor (``bp.LightSource(**params)`` would raise), so every
    caller maps the param dict onto a fresh struct with a setattr loop. Factored
    here so main.py's setup path and the per-frame assembly below share one
    copy. Every key must be a bound attribute name (writing an unbound name onto
    the pybind class raises AttributeError).
    """
    src = bp.LightSource()
    for key, val in params.items():
        setattr(src, key, val)
    return src


def build_static_light_sources(bp, lights, grid_w: int, grid_h: int,
                               sim_time_per_tick: float):
    """Level ``[[light]]`` rows -> (compiled statics, beacon entries, off-grid).

    Called ONCE at setup (never per frame): partitions the level's
    :class:`level_loader.LightEntry` list against the physics grid, builds the
    non-beacon lights into compiled ``bp.LightSource`` structs (their params are
    tick-independent, so ``total_tick=0``), and returns the beacon entries
    unbuilt (their facing angle is a per-frame function of the sim tick — see
    :func:`build_frame_light_sources`). ``off_grid`` is the skipped list so the
    caller can warn ONCE at load (never per frame). Mirrors main.py's original
    setup block verbatim.
    """
    lights_in, lights_off = partition_lights(lights, grid_w, grid_h)
    static = [
        build_light_source(bp, light_source_params(e, 0, sim_time_per_tick))
        for e in lights_in if e.kind != "beacon"
    ]
    beacons = [e for e in lights_in if e.kind == "beacon"]
    return static, beacons, lights_off


@dataclass
class FrameLights:
    """The per-frame assembly result.

    ``sources`` is the compiled ``bp.LightSource`` list (order: statics,
    beacons, fire) for :meth:`GameRenderer.upload_state`. ``fire_count`` /
    ``fire_peaks`` feed the HUD light counter via
    :meth:`GameRenderer.set_fire_light_stats` (count < peaks == the brightest-K
    cap truncated; surfaced so tuning sessions see saturation — no silent caps).
    """
    sources: list
    fire_count: int
    fire_peaks: int


def build_frame_light_sources(bp, static_lights: Sequence, beacon_lights: Sequence,
                              *, total_tick: int, sim_time_per_tick: float,
                              fire_selector=None, temperature_field=None,
                              blackbody_ramp=None,
                              show_fire_lights: bool = False) -> FrameLights:
    """Assemble the per-frame light-source list (statics + beacons + fire).

    Order is fixed and MUST match the pre-extraction main.py block: the prebuilt
    ``static_lights`` first (a fresh list copy — callers keep their originals),
    then each beacon rebuilt from the MONOTONIC sim tick (``total_tick``, on the
    SIM clock — beacons freeze on pause and replay exactly; never wall dt), then
    the brightest-K fire lights when ``show_fire_lights`` is set.

    ``fire_selector`` is a :class:`renderer.fire_lights.FireLightSelector`;
    ``temperature_field`` (read, never written) and ``blackbody_ramp`` are its
    inputs. When fire lights are off (or no selector), the fire contribution is
    empty and ``fire_count``/``fire_peaks`` are 0.

    The mouse flashlight and W6 transient emitters are appended by each CALLER
    afterward (they differ per caller) — see the module docstring.
    """
    sources: List = list(static_lights)

    for e in beacon_lights:
        sources.append(build_light_source(
            bp, light_source_params(e, total_tick, sim_time_per_tick)))

    if show_fire_lights and fire_selector is not None:
        fire_params, fire_peaks = fire_selector.select(
            temperature_field, blackbody_ramp)
    else:
        fire_params, fire_peaks = [], 0
    sources += [build_light_source(bp, p) for p in fire_params]

    return FrameLights(sources=sources, fire_count=len(fire_params),
                       fire_peaks=int(fire_peaks))


__all__ = [
    "build_light_source",
    "build_static_light_sources",
    "build_frame_light_sources",
    "FrameLights",
]
