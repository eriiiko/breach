"""THE ASSEMBLY'S OUTPUT ROW -- cone emitters from every light (ray-engine-v2 P6b).

docs/ray_engine_v2_p6b_lit_world_brief_2026-09-25.md sections 1.3, 2.1 and 3.4;
design v3 sections 4.1 and 4.4. renderer/frame_lights.py enumerates every
light the game shows ONCE as a LightSpec and builds the sweep's cone-emitter
rows from it (since P6c the only row: the old march's LightSource list went
with the march). These tests hold that seam: every old light type has a new
form and lights the world through the real conductor; the float -> integer
door; the flashlight's lens; the sky's door; and -- moved here at P6c from
the deleted old-row oracle tests/test_frame_lights.py -- the assembly
enumerates each in-grid level light exactly once, and a beacon's row turns
with the sim tick and only with it.

Every test's docstring names its property and the change that breaks it.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_frame_lights_cones.py -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402

from level_loader import LevelData, LightEntry  # noqa: E402
from renderer import frame_lights as fl  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation.unit import Unit  # noqa: E402

FB = int(bp.L_FINE_BITS)
TURN = 65536
DT = 1.0 / 24.0
#: The light types the OLD path showed (pre-P6b main.py + frame_lights): the
#: historical set every one of which must have a new form. Fixed by history,
#: not a list designed to grow.
OLD_LIGHT_TYPES = ("lamp", "beacon", "flashlight", "cursor", "transient", "fire")


def _room(n=40):
    tm = np.full((n, n), 4, dtype=np.int32)          # all air
    tm[0, :] = tm[-1, :] = tm[:, 0] = tm[:, -1] = 1  # a hull ring
    return LevelData(name="p6b_lights", version="1", path=Path("."), tilemap=tm,
                     tile_size_m=1.0, diffuse_path=Path("."))


def _lit(sim, rows):
    sim.set_light(True, rows, None)
    sim.relight()
    return sim.gmap.light_q


def _around(q, x, y, r=4):
    """The light a few tiles from (x, y), excluding the cell itself."""
    h, w = q.shape[:2]
    y0, y1, x0, x1 = max(0, y - r), min(h, y + r + 1), max(0, x - r), min(w, x + r + 1)
    box = q[y0:y1, x0:x1, 0].copy()
    box[y - y0, x - x0] = 0
    return int(box.sum())


# ---------------------------------------------------------------------------
# 1. EVERY OLD LIGHT TYPE HAS A NEW FORM, AND IT LIGHTS THE WORLD
# ---------------------------------------------------------------------------
def test_every_old_light_type_has_a_new_form_that_lights_the_world():
    """PROPERTY (brief 3.4, design 4.4): each light type the old path showed --
    a level lamp, a beacon, an onephase flashlight, WEGO's cursor lamp, a W6
    transient emitter, a fire -- has a NEW form, and each one, alone in a dark
    room, lights the cells around it through the real conductor (the one
    assembly -> cone_rows -> Simulation.set_light -> the sweep; the fire through
    the sweep's own thermal emission, never a row). Every type the assembly
    produces is named in EMITTER_INPUTS with its input source, sim or render.

    BREAKS IF: a light type is dropped on the way to the sweep (the assembly
    forgets it, cone_rows skips it, the facade loses the rows) or a type is
    produced whose inputs are not declared.
    """
    from simulation import fire_fixed
    sim = Simulation(_room(), seed=1, breach_physics=bp, enable_recorder=False)
    g = sim.gmap
    h, w = g.solid.shape
    marine = Unit("M", x=8, y=28, team=0)
    sim.add_unit(marine)
    sim.set_paused(False)
    sim.step()                                     # stamp the marine's body
    lamp = LightEntry(x=10.5, y=10.5, color=(1.0, 0.8, 0.6), intensity=1.0,
                      range=18.0, kind="static")
    beacon = LightEntry(x=30.5, y=10.5, color=(1.0, 0.5, 0.2), intensity=2.0, range=12.0,
                        kind="beacon", period_s=2.0, beam_deg=40.0, phase=0.25)
    cone = NS(unit_id=marine.id, x=marine.center_tile_x(), y=marine.center_tile_y(),
              facing=0.0, half_deg=55.0, range_tiles=25.0)
    produced = {}
    produced["lamp"] = fl.frame_light_specs([lamp], total_tick=0, sim_time_per_tick=DT).specs
    produced["beacon"] = fl.frame_light_specs([beacon], total_tick=5,
                                              sim_time_per_tick=DT).specs
    produced["flashlight"] = fl.flashlight_specs([cone], lambda uid: marine.footprint)
    produced["cursor"] = [fl.cursor_lamp_spec((20.3, 30.7))]
    produced["transient"] = fl.transient_specs([{"x": 30.2, "y": 30.6, "max_range": 10,
                                                 "intensity": 1.2,
                                                 "color": (1.0, 0.65, 0.35)}])
    for kind, specs in produced.items():
        assert specs and all(s.kind == kind for s in specs), kind
        assert kind in fl.EMITTER_INPUTS and fl.EMITTER_INPUTS[kind][0] in ("sim", "render")
        rows = fl.cone_rows(specs, w, h, FB)
        assert len(rows) == len(specs), f"{kind}: cone_rows dropped a light"
        q = _lit(sim, rows)
        s = specs[0]
        assert _around(q, int(s.x), int(s.y)) > 0, f"{kind}: its new form lights nothing"
    # the fire: its new form is the sweep's thermal emission -- a burning cell
    # lights its surroundings with NO row at all
    g.temperature[20, 20] = 1263 << 16
    g.fire[20, 20] = fire_fixed.quantize_scalar(0.8)
    g.light_atten_q[20, 20] = g.dyn_light_atten_q[20, 20] = 36045   # furniture 0.55
    q = _lit(sim, None)
    assert _around(q, 20, 20) > 0, "fire: the thermal emission lights nothing"
    assert "fire" in fl.EMITTER_INPUTS
    assert set(OLD_LIGHT_TYPES) <= set(produced) | {"fire"}


# ---------------------------------------------------------------------------
# 2. THE DOOR: render floats -> the sweep's integers
# ---------------------------------------------------------------------------
def test_the_cone_row_door_quantizes_once_and_keeps_the_old_conventions():
    """PROPERTY (the render-side door, frame_lights.spec_cone_row): the cell is
    the floor of the position; rgb is intensity x colour in L°'s currency
    (2^L_FINE_BITS per light unit, rounded, never negative); the centre is the
    screen-convention angle in Q16 turns reduced mod one turn (so -pi/2 and
    3pi/2 are one row); a spread at or above the old march's omni test
    (2*pi - 0.01, which the cursor lamp's 6.283 meets) is omni; an off-grid
    light is skipped; a frame over the cone budget is refused by name.

    BREAKS IF: the door truncates the position differently, forgets the
    currency, lets a negative colour through, stops reducing the angle, treats
    the cursor lamp as a 65534-count cone, or lets the sweep throw mid-tick.
    """
    s = fl.LightSpec(x=7.9, y=3.2, color=(1.0, 0.5, 0.0), intensity=2.5,
                     angle_center=-math.pi / 2, angle_spread=math.radians(60))
    row = fl.spec_cone_row(s, 20, 10, FB)
    assert row[:2] == [3, 7]
    assert row[2:5] == [round(2.5 * (1 << FB)), round(1.25 * (1 << FB)), 0]
    assert row[5] == 3 * TURN // 4 and row[6] == round(TURN / 6)
    s2 = fl.LightSpec(x=7.9, y=3.2, color=(1.0, 0.5, 0.0), intensity=2.5,
                      angle_center=3 * math.pi / 2, angle_spread=math.radians(60))
    assert fl.spec_cone_row(s2, 20, 10, FB)[5] == row[5]
    assert fl.spec_cone_row(fl.cursor_lamp_spec((1.0, 1.0)), 20, 10, FB)[6] == TURN
    neg = fl.LightSpec(x=1.0, y=1.0, color=(-1.0, 1.0, 1.0), intensity=1.0)
    assert fl.spec_cone_row(neg, 20, 10, FB)[2] == 0
    assert fl.spec_cone_row(fl.LightSpec(x=25.0, y=1.0, color=(1, 1, 1), intensity=1.0),
                            20, 10, FB) is None
    huge = [fl.LightSpec(x=1.0, y=1.0, color=(1, 1, 1), intensity=40.0)] * 4
    with pytest.raises(ValueError, match="LIGHT_CONE_BUDGET"):
        fl.cone_rows(huge, 20, 10, FB)
    assert fl.CONE_TURN == int(bp.RadiationSweep.CONE_TURN)
    assert fl.LIGHT_CONE_BUDGET == int(bp.RadiationSweep.LIGHT_CONE_BUDGET)
    assert fl.LIGHT_SKY_MAX == int(bp.RadiationSweep.LIGHT_SKY_MAX)


# ---------------------------------------------------------------------------
# 3. THE FLASHLIGHT'S LENS
# ---------------------------------------------------------------------------
def test_a_flashlight_shines_from_its_lens_outside_the_carriers_body():
    """PROPERTY (the flashlight's cone-emitter form): the lens is exactly
    footprint // 2 + 1 tiles (Chebyshev) from the carrier's centre along its
    facing, for every facing -- outside the light-opaque body -- so the beam
    lights the floor ahead of the marine. PAIR, the old placement: the same
    cone at the marine's CENTRE tile lights nothing beyond the body (the old
    march's rays died in that cell -- the onephase flashlights were dark).

    BREAKS IF: the lens falls back inside the footprint (the beam goes dark
    again) or leaves the facing direction.
    """
    for fp in (1, 3, 5):
        for k in range(16):
            a = k * math.tau / 16
            lx, ly = fl.flashlight_lens(10, 10, a, fp)
            assert max(abs(lx - 10), abs(ly - 10)) == fp // 2 + 1
            assert (lx - 10) * math.cos(a) + (ly - 10) * math.sin(a) > 0
    sim = Simulation(_room(), seed=1, breach_physics=bp, enable_recorder=False)
    h, w = sim.gmap.solid.shape
    m = Unit("M", x=8, y=18, team=0)
    sim.add_unit(m)
    sim.set_paused(False)
    sim.step()
    cone = NS(unit_id=m.id, x=m.center_tile_x(), y=m.center_tile_y(), facing=0.0,
              half_deg=55.0, range_tiles=25.0)
    spec = fl.flashlight_specs([cone], lambda uid: m.footprint)[0]
    q = _lit(sim, fl.cone_rows([spec], w, h, FB))
    cx, cy = m.center_tile_x(), m.center_tile_y()
    ahead = q[cy, cx + 5:cx + 12, 0]
    assert np.all(ahead > 0), "the beam does not reach the floor ahead"
    centre = fl.LightSpec(x=cx + 0.5, y=cy + 0.5, color=spec.color,
                          intensity=spec.intensity, angle_center=0.0,
                          angle_spread=spec.angle_spread, kind="flashlight")
    q0 = _lit(sim, fl.cone_rows([centre], w, h, FB))
    assert np.all(q0[cy, cx + 2:, 0] == 0), "a centre-tile source should be swallowed"


# ---------------------------------------------------------------------------
# 4. THE ASSEMBLY ENUMERATES EACH LEVEL LIGHT ONCE; THE BEACON RIDES THE TICK
#    (moved at P6c from the deleted tests/test_frame_lights.py, the old-row
#    oracle: its partition and its beacon-sweep properties, on the cone row)
# ---------------------------------------------------------------------------
def _level_lights():
    """Two static lamps + one beacon in a 40x30 grid, and one lamp off it."""
    return [
        LightEntry(x=6.5, y=4.5, color=(1.0, 0.1, 0.05), intensity=0.9,
                   range=18.0, kind="static"),
        LightEntry(x=30.5, y=4.5, color=(0.6, 0.7, 1.0), intensity=1.2,
                   range=14.0, kind="static"),
        LightEntry(x=18.5, y=12.5, color=(1.0, 0.63, 0.16), intensity=3.0,
                   range=12.0, kind="beacon", period_s=2.0, beam_deg=30.0,
                   phase=0.0),
        LightEntry(x=100.0, y=200.0, color=(1.0, 1.0, 1.0), intensity=1.0,
                   range=10.0, kind="static"),
    ]


def test_the_assembly_gives_each_in_grid_level_light_one_row_and_no_fire():
    """PROPERTY (moved from test_frame_lights.py's partition + old-inline
    oracle): through the one assembly (partition_lights -> frame_light_specs ->
    cone_rows) every in-grid [[light]] becomes exactly ONE cone row at its own
    cell with its own colour x intensity, an off-grid light none, and the
    caller's extras are appended after -- whatever the map's temperature: the
    assembly never adds a fire row (the sweep's fire light is its own).

    BREAKS IF: the assembly enumerates a light twice or drops one, lets an
    off-grid light through, reorders a light's numbers, or grows a fire row.
    """
    lights = _level_lights()
    lights_in, off = fl.partition_lights(lights, 40, 30)
    assert [e.x for e in off] == [100.0]
    extra = [fl.cursor_lamp_spec((2.5, 2.5))]
    specs = fl.frame_light_specs(lights_in, total_tick=17, sim_time_per_tick=DT,
                                 extra=extra).specs
    rows = fl.cone_rows(specs, 40, 30, FB)
    assert len(rows) == len(lights_in) + len(extra) == 4
    scale = 1 << FB
    for e, r in zip(lights_in, rows[:3]):
        assert (int(r[0]), int(r[1])) == (int(e.y), int(e.x))
        assert [int(v) for v in r[2:5]] == [round(e.intensity * c * scale)
                                            for c in e.color]
    assert (int(rows[3][0]), int(rows[3][1])) == (2, 2)
    assert sorted(s.kind for s in specs) == ["beacon", "cursor", "lamp", "lamp"]


def test_a_beacons_cone_row_turns_with_the_sim_tick_and_only_with_it():
    """PROPERTY (moved from test_frame_lights.py's beacon-sweep test; P4's
    freeze ruling): a beacon's cone-row centre is a pure function of the
    MONOTONIC sim tick -- the same tick gives the same row (a paused sim is a
    frozen beacon, a replay the same beam), a later tick turns it by
    period_s's rate in Q16 turns; the static lamps' rows never move.

    BREAKS IF: the beacon's facing reads the wall clock or accumulates per
    call, the assembly feeds statics the tick, or the row door stops
    converting the angle.
    """
    lights_in, _off = fl.partition_lights(_level_lights(), 40, 30)

    def rows_at(tick):
        specs = fl.frame_light_specs(lights_in, total_tick=tick,
                                     sim_time_per_tick=DT).specs
        return fl.cone_rows(specs, 40, 30, FB)

    r0, r0b, r6 = rows_at(0), rows_at(0), rows_at(6)
    assert np.array_equal(r0, r0b)
    assert np.array_equal(r0[:2], r6[:2]), "a static lamp moved with the tick"
    turn = (int(r6[2][5]) - int(r0[2][5])) % TURN
    # 6 ticks at 24 Hz = 0.25 s of a 2 s period = 1/8 turn (row door rounding)
    assert abs(turn - TURN // 8) <= 1, turn


# ---------------------------------------------------------------------------
# 5. THE SKY'S DOOR
# ---------------------------------------------------------------------------
def test_the_sky_door_builds_overcast_and_sun_from_the_boundary_type():
    """PROPERTY (brief 1.4): sky_for_level picks the table by the level's
    boundary type; an overcast irradiance is split EVENLY over the 16
    ordinates (so it sums back to the configured value in L°'s currency); a
    SUN adds light only in the ordinates its beam overlaps -- through the
    engine's own projection (RadiationSweep.cone_emission), never a copy; an
    all-zero sky is the dark ring (None); a negative value is refused.

    BREAKS IF: the door reads the wrong table, splits the overcast unevenly,
    re-implements the projection (a sun leaking into the wrong ordinates), or
    lets a negative sky through.
    """
    cfg = NS(light=NS(sky=NS(
        ambient=NS(overcast=[0.08, 0.08, 0.1], sun=[0.0, 0.0, 0.0]),
        sunny=NS(overcast=[0.0, 0.0, 0.0], sun=[0.5, 0.4, 0.3], sun_bearing_deg=90.0,
                 sun_spread_deg=10.0),
        dark=NS(overcast=[0.0, 0.0, 0.0], sun=[0.0, 0.0, 0.0]),
        bad=NS(overcast=[-0.1, 0.0, 0.0], sun=[0.0, 0.0, 0.0]))))
    sky = fl.sky_for_level("ambient", cfg, bp)
    assert np.all(sky == sky[0]) and abs(int(sky[:, 0].sum()) - 0.08 * (1 << FB)) <= 16
    sun = fl.sky_for_level("sunny", cfg, bp)
    exp = np.asarray(bp.RadiationSweep.cone_emission(
        [round(0.5 * (1 << FB)), round(0.4 * (1 << FB)), round(0.3 * (1 << FB))],
        TURN // 4, round(TURN * 10 / 360), 16))
    assert np.array_equal(sun, exp)
    lit = [m for m in range(16) if sun[m, 0] > 0]
    assert lit and all(m in (3, 4) for m in lit)       # the bins around +y (90 deg)
    assert fl.sky_for_level("dark", cfg, bp) is None
    with pytest.raises(ValueError):
        fl.sky_for_level("bad", cfg, bp)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
