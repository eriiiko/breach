"""Pure ``[[light]]`` -> light parameter helpers (P4; P6b: :class:`LightSpec`).

Since ray-engine-v2 P6b a level light is first a :class:`LightSpec` (what it
is), then an output row: the sweep's cone-emitter row (renderer/frame_lights.py
``cone_rows``, live) or, behind the P6b old/new toggle only, the old march's
``LightSource`` via :func:`light_source_params` (deleted at P6c).

Render-side module (deliberately NOT under ``src/simulation/`` — render
channels are ingress-exempt, engine/14 synced-vs-local) and importable
WITHOUT ``breach_physics``: everything here is plain math on plain data, so
the beacon/parameter behaviour is headless-testable and ``main.py``'s
per-frame code stays a thin setattr loop over the dicts built here.

Binding design calls (docs/patch_levels_p4_lights.md):

- **Beacons freeze with the sim** (Erik's locked call 2026-07-07): the
  facing angle is a PURE function of the sim tick — never ``+=`` per frame,
  never the wall-clock frame dt (critique B1: wall dt = wobble, no
  pause-freeze, no replay). Callers pass ``sim_time_per_tick``
  (= 1 / CFG.clock.ticks_per_second).
- **The tick must be monotonic across rounds** (critique M1):
  ``Simulation.tick`` rewinds to 0 each round while ``turn_number``
  increments, so use :func:`monotonic_total_tick` — still a pure function
  of sim state, replay-exact, no phase-0 snap at round boundaries.
- **heat=0.0 / jitter=0.0 are STRUCTURAL** (critique M2): ``heat`` is the
  only synced ray output and headless goldens never execute the render
  light pass — a leak would silently diverge interactive sessions from
  their replays. :func:`light_source_params` hard-pins both; the loader
  additionally rejects the keys in ``[[light]]`` toml.
- **No falloff field**: the Python bindings (cpp/src/bindings.cpp:847-865)
  expose no ``falloff`` — every Python-built source is uniform. The dicts
  built here contain ONLY bound attribute names; writing an unbound name
  onto the pybind class would raise AttributeError.
- Beacon stepping granularity is the tick rate (24 Hz -> 7.5 deg/step at
  period 2 s): accepted, on record — do NOT smooth with wall-clock
  interpolation later; that breaks freeze/replay (critique N2).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# angle_spread >= 2*pi = omnidirectional emission (raycaster cone contract).
STATIC_SPREAD = math.tau


# ---------------------------------------------------------------------------
# ray-engine-v2 P6b: THE LIGHT SPEC -- what a light IS, before its output row.
# ---------------------------------------------------------------------------
# The frame-lights assembly (renderer/frame_lights.py, the ONE assembly)
# enumerates every light the game shows as a LightSpec, once; two output rows
# are built from the SAME list: the sweep's CONE-EMITTER row (frame_lights.
# cone_rows, the live one) and, for P6b's old/new toggle only, the old render
# march's bp.LightSource (frame_lights.light_sources, deleted at P6c). Plain
# data, importable without breach_physics or the renderer.
@dataclass(frozen=True)
class LightSpec:
    """One light, in the old LightSource's float convention.

    x, y          tile coordinates (float; the cell is (floor(y), floor(x)))
    color         (r, g, b) tint, max channel ~1
    intensity     TOTAL emitted power in LIGHT UNITS (1.0 = the blackbody
                  ramp's intensity 1.0) -- the old LightSource.intensity's unit;
                  measured 1:1 against the old march (the P6b calibration)
    angle_center  beam centre, radians, SCREEN convention (dx = cos, dy = sin,
                  +y the increasing row -- the ordinates' own)
    angle_spread  full beam angle, radians; >= 2*pi - 0.01 is omni (the old
                  march's own cone test)
    max_range     the old march's reach in tiles (the OLD row only; a sweep
                  has no range)
    kind          "lamp" | "beacon" | "flashlight" | "cursor" | "transient" |
                  "fire" (the old path's fire lights; the sweep's fire light is
                  the thermal emission, never a row)
    source        "sim" or "render": where the light's INPUTS come from -- the
                  determinism map of P6b brief decision 1 (render-sourced
                  lights keep light_q from being sim-pure until P7)
    """
    x: float
    y: float
    color: tuple
    intensity: float
    angle_center: float = 0.0
    angle_spread: float = STATIC_SPREAD
    max_range: float = 20.0
    kind: str = "lamp"
    source: str = "sim"


def light_spec(entry, total_tick: int, tick_dt_s: float) -> LightSpec:
    """``LightEntry`` -> its :class:`LightSpec`, the SAME numbers
    :func:`light_source_params` puts in the old row (position, colour,
    intensity, the beacon's sim-tick facing and beam, the range). Level data
    and the monotonic sim tick: a SIM-sourced light."""
    p = light_source_params(entry, total_tick, tick_dt_s)
    return LightSpec(x=p["x"], y=p["y"], color=p["color"], intensity=p["intensity"],
                     angle_center=p["angle_center"], angle_spread=p["angle_spread"],
                     max_range=p["max_range"],
                     kind="beacon" if entry.kind == "beacon" else "lamp",
                     source="sim")


def beacon_angle(total_tick: int, tick_dt_s: float, period_s: float,
                 phase: float) -> float:
    """Beacon facing angle in radians — a pure function of the sim tick.

    Frozen when the sim is paused (the tick does not advance), exact under
    replay, no drift (never accumulated per frame). ``phase`` is a fraction
    of a turn (a red/blue cop-car pair = phases 0.0 / 0.5). The caller may
    reduce mod 2*pi.
    """
    return math.tau * (float(phase)
                       + (int(total_tick) * float(tick_dt_s))
                       / float(period_s))


def monotonic_total_tick(turn_number: int, ticks_per_round: int,
                         tick: int) -> int:
    """Total sim ticks since match start — monotonic ACROSS rounds.

    ``Simulation.tick`` rewinds to 0 at every round boundary exactly when
    ``Simulation.turn_number`` (1-based) increments, so
    ``(turn_number - 1) * ticks_per_round + tick`` never decreases and
    advances by 1 through the boundary — beacons sweep smoothly instead of
    snapping to phase 0 each round (critique M1).
    """
    return (int(turn_number) - 1) * int(ticks_per_round) + int(tick)


def light_source_params(entry, total_tick: int, tick_dt_s: float) -> dict:
    """``LightEntry`` -> kwargs dict for ``bp.LightSource``.

    Pure: main.py maps the dict onto the compiled struct with a setattr
    loop (``bp.LightSource`` has only ``py::init<>()`` — never
    ``bp.LightSource(**d)``). Beacons get ``angle_center`` from
    :func:`beacon_angle` (reduced mod 2*pi) + ``angle_spread`` from
    ``beam_deg``; static lights emit uniformly (spread = 2*pi).
    ALWAYS emits ``heat=0.0`` and ``jitter=0.0`` — structural, see module
    docstring. Every key is a bound LightSource attribute; ``ray_count``
    stays at the compiled default.
    """
    params = {
        "x": float(entry.x),
        "y": float(entry.y),
        "max_range": float(entry.range),
        "intensity": float(entry.intensity),
        "color": (float(entry.color[0]), float(entry.color[1]),
                  float(entry.color[2])),
        # STRUCTURAL zeroes (critique M2): level lights never write the
        # synced heat channel and never pull C++ RNG jitter.
        "heat": 0.0,
        "jitter": 0.0,
    }
    if entry.kind == "beacon":
        params["angle_center"] = beacon_angle(
            total_tick, tick_dt_s, entry.period_s, entry.phase) % math.tau
        params["angle_spread"] = math.radians(float(entry.beam_deg))
    else:
        params["angle_center"] = 0.0
        params["angle_spread"] = STATIC_SPREAD
    return params


def partition_lights(lights, grid_w: int, grid_h: int) -> tuple:
    """Split entries into (in_bounds, off_grid) against the physics grid.

    Same bounds rule as the retired hardcoded-lamp block
    (``0 <= x < width and 0 <= y < height``): lamp positions authored for
    the 50x120 vessel are skipped on a smaller level. main.py warns ONCE at
    load for the off-grid list — never per frame.
    """
    in_bounds, off_grid = [], []
    for entry in lights:
        if (0.0 <= float(entry.x) < float(grid_w)
                and 0.0 <= float(entry.y) < float(grid_h)):
            in_bounds.append(entry)
        else:
            off_grid.append(entry)
    return in_bounds, off_grid
