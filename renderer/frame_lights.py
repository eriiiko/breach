"""The ONE light assembly -- shared by the game + harness (B2 P1; P6b: cone rows).

Fire & Heat Beauty arc, beat B2 patch P1 (design
docs/fire_b2_smoke_honesty_design_2026-07-21.md §2). This module deduplicates
the ONE thing main.py and tools/lighting_demo.py must not let drift apart: the
per-tick assembly of the light list from whatever supplies lights today — the
levels-w1 ``[[light]]`` rows (static lamps + the rotating beacon; see
:mod:`level_lights`) plus the caller's render-sourced lights.

**NOT an entity system** (Erik's explicit concern, 2026-07-22): this creates
ZERO entity machinery. Lamps/beacons are NOT entities today — they predate the
Arc A/B entity layer (doors/sensors/nodes/pump/airlock). Migrating lights INTO
the entity system is Arc C's convergence item; when it happens only this
helper's INPUT changes and the assembly seam survives. Do not build any
lamp/beacon entity here.

Deliberately importable WITHOUT the renderer package touching this module's
consumers: it takes the compiled ``breach_physics`` module as an argument
(never imports it) and imports only :mod:`level_lights` (plain math on plain
data), so the assembly is headless-testable in isolation. It never writes sim
fields.

RAY-ENGINE-V2 P6b/P6c (design v3 §4.1, §4.4; docs/ray_engine_v2_p6b_lit_world_
brief_2026-09-25.md §1.3): THE OUTPUT ROW CHANGED, THE SEAM SURVIVED. Every
light the game shows is enumerated ONCE as a :class:`level_lights.LightSpec`
(:func:`frame_light_specs` -- the level's lamps and beacons, plus the caller's
render-sourced specs: the onephase flashlights at their lens
(:func:`flashlight_specs`), WEGO's cursor lamp (:func:`cursor_lamp_spec`), the
W6 transient emitters (:func:`transient_specs`)), and that ONE list becomes
:func:`cone_rows` -- THE SWEEP'S CONE-EMITTER ROWS (y, x, r, g, b, center_q,
spread_q), int64, handed to the engine once per sim tick
(``Simulation.set_light``). (The old render march's ``bp.LightSource`` row and
the brightest-K fire lights, P6b's F11 A/B, were deleted with the march at P6c.)

The fire's light is NOT a row: every burning cell emits through L°[T] in the
sweep itself (design §7.3). :func:`sky_for_level` is the sky's door (the light
ring's per-ordinate RGB from the level's boundary type). :data:`EMITTER_INPUTS`
names where every light type's inputs come from -- sim or render state --
because ``light_q`` is not sim-pure until P7 separates them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from level_lights import (LightSpec, light_spec, monotonic_total_tick,
                          partition_lights)


# ===========================================================================
# ray-engine-v2 P6b: THE SPECS, THE CONE ROWS, THE SKY
# ===========================================================================
#: Where every light type's INPUTS come from (P6b brief decision 1). "sim":
#: level data, config, or synced sim state -- replay-exact. "render": UI /
#: control-source / renderer state -- the reason ``light_q`` is not sim-pure
#: until P7 separates them. Every light the old path showed has a form
#: here (tests/test_frame_lights_cones.py holds the map to the assembly).
EMITTER_INPUTS = {
    "lamp":       ("sim",    "level [[light]] row (position, colour, intensity)"),
    "beacon":     ("sim",    "level [[light]] row + the monotonic sim tick (facing)"),
    "flashlight": ("render", "unit position + facing (sim) + the flashlight mode and "
                             "selected unit (control-source state) + the planning aim "
                             "at the cursor while paused (mouse)"),
    "cursor":     ("render", "the mouse position (WEGO's cursor lamp)"),
    "transient":  ("render", "the renderer's W6 effect queue (sim events, aged and "
                             "faded on the wall clock)"),
    "fire":       ("sim",    "temperature -- the sweep's own thermal emission "
                             "a_c * L°[T], never a row"),
    "sky":        ("sim",    "the level's boundary type + [light.sky] config (the ring)"),
}

CONE_TURN = 65536            # one turn in Q16 turns == RadiationSweep.CONE_TURN
OMNI_EPS = 0.01              # spread >= 2*pi - eps is omni (the old march's cone test, kept)
LIGHT_CONE_BUDGET = 1 << 44  # == RadiationSweep.LIGHT_CONE_BUDGET (the sweep's door)
LIGHT_SKY_MAX = 1 << 40      # == RadiationSweep.LIGHT_SKY_MAX

# The shipped flashlight / cursor-lamp numbers (moved verbatim from main.py).
FLASHLIGHT_INTENSITY = 2.5
FLASHLIGHT_COLOR = (1.0, 1.0, 0.95)
CURSOR_INTENSITY = 2.5
CURSOR_COLOR = (1.0, 1.0, 0.95)


def flashlight_lens(cx: float, cy: float, angle: float, footprint: int):
    """The tile a flashlight's light leaves from: the first cell OUTSIDE its
    carrier's square footprint along the beam (Chebyshev distance
    footprint // 2 + 1 from the centre tile). The body is stamped light-opaque
    (units block light), so a lens inside it lights nothing past the body --
    which is exactly what the old march did with the centre-tile source (its
    rays died in the carrier's own cell). Returns the lens tile's (x, y)."""
    dx, dy = math.cos(angle), math.sin(angle)
    m = max(abs(dx), abs(dy))
    off = int(footprint) // 2 + 1
    return int(cx) + int(round(dx / m * off)), int(cy) + int(round(dy / m * off))


def flashlight_specs(cones, footprint_of) -> list:
    """The onephase flashlights (``ui.flashlight_cones``) as specs: a cone of
    the facing half-angle x 2, centred on the facing (screen convention: the
    one negation of Unit.facing's y-up), at the LENS outside the carrier's
    footprint (:func:`flashlight_lens`). ``footprint_of(unit_id)`` returns the
    carrier's footprint in tiles (``cone.range_tiles`` is not read: a sweep has
    no range). RENDER-sourced (control-source state)."""
    out = []
    for cone in cones:
        angle = -float(cone.facing)
        lx, ly = flashlight_lens(cone.x, cone.y, angle, footprint_of(cone.unit_id))
        out.append(LightSpec(x=lx + 0.5, y=ly + 0.5, color=FLASHLIGHT_COLOR,
                             intensity=FLASHLIGHT_INTENSITY, angle_center=angle,
                             angle_spread=math.radians(float(cone.half_deg) * 2.0),
                             kind="flashlight", source="render"))
    return out


def cursor_lamp_spec(mouse_tile_f) -> Optional[LightSpec]:
    """WEGO's cursor lamp: an omni lamp at the mouse (None off the map).
    RENDER-sourced."""
    if mouse_tile_f is None:
        return None
    return LightSpec(x=float(mouse_tile_f[0]), y=float(mouse_tile_f[1]),
                     color=CURSOR_COLOR, intensity=CURSOR_INTENSITY,
                     angle_spread=6.283, kind="cursor", source="render")


def transient_specs(transients) -> list:
    """The W6 transient emitters (``GameRenderer.transient_light_specs()``
    dicts: x, y, intensity, color; their ``max_range`` is not read -- a sweep
    has no range) as omni specs. RENDER-sourced."""
    return [LightSpec(x=float(t["x"]), y=float(t["y"]), color=tuple(t["color"]),
                      intensity=float(t["intensity"]), angle_spread=6.283,
                      kind="transient", source="render") for t in transients]


@dataclass
class FrameLightSpecs:
    """One assembly: every light as a spec."""
    specs: list


def frame_light_specs(lights_in: Sequence, *, total_tick: int, sim_time_per_tick: float,
                      extra: Sequence = ()) -> FrameLightSpecs:
    """THE ASSEMBLY (P6b): every light the game shows, once, as specs.

    ``lights_in`` are the level's in-grid ``[[light]]`` entries (statics first,
    then beacons), each through :func:`level_lights.light_spec` at the
    MONOTONIC sim tick; ``extra`` the caller's render-sourced specs
    (flashlights, the cursor lamp, the W6 transients). A fire is never a spec:
    the sweep's fire light is its own thermal emission."""
    statics = [e for e in lights_in if e.kind != "beacon"]
    beacons = [e for e in lights_in if e.kind == "beacon"]
    specs = [light_spec(e, 0, sim_time_per_tick) for e in statics]
    specs += [light_spec(e, total_tick, sim_time_per_tick) for e in beacons]
    specs += list(extra)
    return FrameLightSpecs(specs=specs)


def spec_cone_row(spec: LightSpec, grid_w: int, grid_h: int, fine_bits: int):
    """One spec -> its cone-emitter row [y, x, r, g, b, center_q, spread_q], or
    None when its cell is off the grid. THE DOOR from render floats to the
    sweep's integers (a render-side quantization, once): the cell is the floor
    of the position; rgb = intensity x colour in L°'s currency (2^fine_bits
    per light unit, rounded, never negative); the angles in Q16 turns, the
    centre reduced mod one turn, a spread >= 2*pi - OMNI_EPS omni (the old
    march's test)."""
    x, y = int(math.floor(spec.x)), int(math.floor(spec.y))
    if not (0 <= x < grid_w and 0 <= y < grid_h):
        return None
    scale = float(1 << int(fine_bits))
    rgb = [max(0, int(round(max(0.0, float(spec.intensity) * float(c)) * scale)))
           for c in spec.color]
    center = int(round(float(spec.angle_center) / math.tau * CONE_TURN)) % CONE_TURN
    if float(spec.angle_spread) >= math.tau - OMNI_EPS:
        spread = CONE_TURN
    else:
        spread = max(0, min(CONE_TURN, int(round(float(spec.angle_spread)
                                                 / math.tau * CONE_TURN))))
    return [y, x, rgb[0], rgb[1], rgb[2], center, spread]


def cone_rows(specs: Sequence, grid_w: int, grid_h: int, fine_bits: int) -> np.ndarray:
    """THE OUTPUT ROW: every spec as a cone-emitter row, (n, 7) int64, for
    ``Simulation.set_light`` (the engine reads it once per tick). Off-grid
    specs are skipped (the ``partition_lights`` rule). Refuses a frame whose
    summed rgb passes the cone budget (the sweep's int64 argument) here, by
    name, rather than let the sweep throw mid-tick."""
    rows = []
    for s in specs:
        r = spec_cone_row(s, grid_w, grid_h, fine_bits)
        if r is not None:
            rows.append(r)
    a = np.asarray(rows, dtype=np.int64).reshape(len(rows), 7)
    if len(rows) and np.any(a[:, 2:5].sum(axis=0) > LIGHT_CONE_BUDGET):
        raise ValueError(f"frame lights: the cones' summed rgb {a[:, 2:5].sum(axis=0)} "
                         f"passes LIGHT_CONE_BUDGET ({LIGHT_CONE_BUDGET}) -- "
                         f"{len(rows)} lights this frame")
    return np.ascontiguousarray(a)


def sky_for_level(boundary: str, cfg, bp, n_ordinates: int = 16):
    """THE SKY's door (P6b brief §1.4; design §2.7): the light ring's per-
    ordinate RGB, (n_ordinates, 3) int64 in L°'s currency, from the level's
    BOUNDARY TYPE and ``[light.sky.<boundary>]`` -- or None (the dark ring)
    when the type has no table or its sky is all zero. Space is dark; an
    ``ambient`` (planetside) boundary is open air: ``overcast`` is the
    IRRADIANCE a clear boundary cell receives from a uniform sky, in light
    units, split evenly over the ordinates (a uniform sky then carries exactly
    that into every clear cell); an optional SUN adds ``sun`` light units of
    irradiance travelling along ``sun_bearing_deg`` (screen convention: 0 =
    toward +x, 90 = toward +y) inside ``sun_spread_deg``, projected onto the
    ordinates by the engine's own ``RadiationSweep.cone_emission`` (the ONE
    projection)."""
    sky_cfg = getattr(getattr(cfg, "light", None), "sky", None)
    t = getattr(sky_cfg, str(boundary), None)
    if t is None:
        return None
    scale = float(1 << int(bp.L_FINE_BITS))

    def _rgb(name):
        v = [float(c) for c in getattr(t, name, (0.0, 0.0, 0.0))]
        if len(v) != 3 or any((not math.isfinite(c)) or c < 0.0 for c in v):
            raise ValueError(f"[light.sky.{boundary}] {name} must be 3 "
                             f"non-negative numbers")
        return v

    over = _rgb("overcast")
    sky = np.zeros((n_ordinates, 3), dtype=np.int64)
    for c in range(3):
        sky[:, c] = int(round(over[c] / n_ordinates * scale))
    sun = _rgb("sun")
    if any(v > 0.0 for v in sun):
        bearing = float(getattr(t, "sun_bearing_deg", 0.0))
        spread = float(getattr(t, "sun_spread_deg", 10.0))
        center_q = int(round(bearing / 360.0 * CONE_TURN)) % CONE_TURN
        spread_q = max(0, min(CONE_TURN, int(round(spread / 360.0 * CONE_TURN))))
        sun_q = [int(round(v * scale)) for v in sun]
        sky += np.asarray(bp.RadiationSweep.cone_emission(sun_q, center_q, spread_q,
                                                          n_ordinates), dtype=np.int64)
    if np.any(sky > LIGHT_SKY_MAX):
        raise ValueError(f"[light.sky.{boundary}]: a per-ordinate sky value passes "
                         f"LIGHT_SKY_MAX ({LIGHT_SKY_MAX})")
    if not np.any(sky):
        return None
    return np.ascontiguousarray(sky)


__all__ = [
    "EMITTER_INPUTS",
    "FrameLightSpecs",
    "frame_light_specs",
    "flashlight_lens",
    "flashlight_specs",
    "cursor_lamp_spec",
    "transient_specs",
    "spec_cone_row",
    "cone_rows",
    "sky_for_level",
    "monotonic_total_tick",
    "partition_lights",
]
