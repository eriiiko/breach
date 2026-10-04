"""Pure ``[[light]]`` -> light helpers (P4; P6b: :class:`LightSpec`).

Since ray-engine-v2 P6b a level light is first a :class:`LightSpec` (what it
is), then the sweep's cone-emitter row (renderer/frame_lights.py
``cone_rows``). Since P6c that is its ONLY output row: the old render march's
``LightSource`` row (``light_source_params``) was deleted with the march.

Render-side module (deliberately NOT under ``src/simulation/`` — render
channels are ingress-exempt, engine/14 synced-vs-local) and importable
WITHOUT ``breach_physics``: everything here is plain math on plain data, so
the beacon behaviour is headless-testable.

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
- **No heat, no jitter** (critique M2): a level light never writes synced
  state and never draws random numbers — the loader rejects ``heat`` /
  ``jitter`` keys in ``[[light]]``, and the cone-emitter row has no field
  for either (a sweep has no RNG and its light channels carry no heat).
- **No range**: a ``[[light]]``'s ``range`` is still parsed (level data is
  not migrated) but a sweep has no reach, so nothing here reads it
  (ray-engine-v2 P6c brief decision 9).
- Beacon stepping granularity is the tick rate (24 Hz -> 7.5 deg/step at
  period 2 s): accepted, on record — do NOT smooth with wall-clock
  interpolation later; that breaks freeze/replay (critique N2).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# angle_spread >= 2*pi = omnidirectional emission (the cone door's omni test,
# renderer/frame_lights.py spec_cone_row).
STATIC_SPREAD = math.tau


# ---------------------------------------------------------------------------
# ray-engine-v2 P6b: THE LIGHT SPEC -- what a light IS, before its output row.
# ---------------------------------------------------------------------------
# The frame-lights assembly (renderer/frame_lights.py, the ONE assembly)
# enumerates every light the game shows as a LightSpec, once, and builds the
# sweep's CONE-EMITTER rows from that list (frame_lights.cone_rows). Plain
# data, importable without breach_physics or the renderer.
@dataclass(frozen=True)
class LightSpec:
    """One light, in render floats.

    x, y          tile coordinates (float; the cell is (floor(y), floor(x)))
    color         (r, g, b) tint, max channel ~1
    intensity     TOTAL emitted power in LIGHT UNITS (1.0 = the blackbody
                  ramp's intensity 1.0) -- measured 1:1 against the old render
                  march at P6b (the calibration)
    angle_center  beam centre, radians, SCREEN convention (dx = cos, dy = sin,
                  +y the increasing row -- the ordinates' own)
    angle_spread  full beam angle, radians; >= 2*pi - 0.01 is omni
    kind          "lamp" | "beacon" | "spot" | "flashlight" | "cursor" |
                  "transient"
                  (a fire is never a spec: its light is the sweep's own
                  thermal emission)
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
    kind: str = "lamp"
    source: str = "sim"


def light_spec(entry, total_tick: int, tick_dt_s: float) -> LightSpec:
    """``LightEntry`` -> its :class:`LightSpec`: position, colour, intensity,
    and the beam -- a static lamp is omni (centre 0, spread 2*pi); a beacon's
    centre is :func:`beacon_angle` at the MONOTONIC sim tick reduced into
    [0, 2*pi), its spread ``beam_deg``; a spot is a beacon that never turns
    (centre ``phase`` turns, spread ``beam_deg``). Level data and the sim tick: a
    SIM-sourced light. (``entry.range`` is not read: a sweep has no range.)"""
    if entry.kind == "beacon":
        center = beacon_angle(total_tick, tick_dt_s, entry.period_s,
                              entry.phase) % math.tau
        spread = math.radians(float(entry.beam_deg))
        kind = "beacon"
    elif entry.kind == "spot":
        # a beacon that never turns: aimed at `phase`, independent of the tick
        center = (math.tau * float(entry.phase)) % math.tau
        spread = math.radians(float(entry.beam_deg))
        kind = "spot"
    else:
        center = 0.0
        spread = STATIC_SPREAD
        kind = "lamp"
    return LightSpec(x=float(entry.x), y=float(entry.y),
                     color=(float(entry.color[0]), float(entry.color[1]),
                            float(entry.color[2])),
                     intensity=float(entry.intensity),
                     angle_center=center, angle_spread=spread,
                     kind=kind, source="sim")


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
