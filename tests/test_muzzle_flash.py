"""#33 muzzle flash: a bullet's shot lights a short lamp at the muzzle.

The pure part is ``renderer/frame_lights.py::muzzle_flash_lights`` (the
renderer's effect queue in, transient-light dicts out -- the same dicts the
W6 jets and plasma bolts feed ``transient_specs``); the sim's only part is the
render-only ``ShotFiredEvent.launch`` flag, which tells the segment that leaves
the barrel from the later segments of a slow round.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_muzzle_flash.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from renderer.frame_lights import MuzzleFlash, muzzle_flash_lights  # noqa: E402

FLASH = MuzzleFlash(intensity=6.0, duration_s=0.1, color=(1.0, 0.8, 0.5))
FOOTPRINT = 3


def _tracer(t, *, unit=1, frm=(10.0, 10.0), to=(30.0, 16.0), launch=True):
    return {"kind": "tracer", "from": frm, "to": to, "t": t, "life": 0.18,
            "unit": unit, "launch": launch}


def _lights(effects, flash=FLASH):
    return muzzle_flash_lights(effects, flash, lambda uid: FOOTPRINT)


def test_a_fresh_shot_lights_its_muzzle_outside_the_shooter():
    """PROPERTY: a fresh tracer gives exactly one lamp, on a cell OUTSIDE the
    shooter's footprint (a unit's own tiles are light-opaque, so a lamp inside
    them lights nothing) and on the shot's side of the shooter. BREAKS if the
    flash is placed at the tracer's ``from`` (the shooter's centre tile), at
    its far end, or behind the shooter."""
    (light,) = _lights([_tracer(0.0)])
    cx, cy = 10, 10
    lx, ly = int(light["x"]), int(light["y"])
    assert max(abs(lx - cx), abs(ly - cy)) > FOOTPRINT // 2
    assert (lx - cx) * (30 - 10) + (ly - cy) * (16 - 10) > 0
    assert max(abs(lx - cx), abs(ly - cy)) <= FOOTPRINT // 2 + 1  # at the muzzle, not downrange
    assert light["intensity"] > 0.0


def test_the_flash_fades_with_age_and_is_gone_after_its_duration():
    """PROPERTY: the lamp's strength strictly falls as the tracer ages, and a
    tracer older than ``duration_s`` lights nothing. BREAKS if the flash
    holds full strength for the tracer's whole life, lingers past its
    duration, or brightens."""
    ages = np.linspace(0.0, FLASH.duration_s * 0.95, 6)
    strengths = [_lights([_tracer(float(t))])[0]["intensity"] for t in ages]
    assert all(a > b for a, b in zip(strengths, strengths[1:]))
    assert _lights([_tracer(FLASH.duration_s)]) == []
    assert _lights([_tracer(FLASH.duration_s * 1.5)]) == []


def test_intensity_zero_gives_no_muzzle_light():
    """PROPERTY: ``[render.muzzle_flash] intensity = 0`` is exactly the game
    without the flash -- no light at all, however many shots are live.
    BREAKS if a zero-strength lamp row is still emitted (it would change the
    cone rows the sweep is handed) or the switch is ignored."""
    off = MuzzleFlash(intensity=0.0, duration_s=0.1, color=(1.0, 0.8, 0.5))
    effects = [_tracer(0.0, unit=u) for u in range(4)]
    assert _lights(effects, off) == []


def test_one_flash_per_shooter_not_per_bullet():
    """PROPERTY: the bullets of one burst leave one muzzle -- a shooter's live
    tracers give ONE lamp (the freshest's strength), so the frame's light
    budget grows with the shooters, never with the bullets; and a later
    segment of a slow round (``launch`` False) is not a shot. BREAKS if each
    pellet of a shotgun or each round of a burst adds its own lamp (eight
    pellets x four marines summed toward LIGHT_CONE_BUDGET), or if a plasma
    bolt re-flashes at the muzzle every tick it flies."""
    burst = [_tracer(0.0, unit=1) for _ in range(8)] + [_tracer(0.05, unit=1)]
    lights = _lights(burst)
    assert len(lights) == 1
    assert lights[0]["intensity"] == _lights([_tracer(0.0)])[0]["intensity"]
    assert _lights([_tracer(0.0, launch=False)]) == []


def test_only_the_first_segment_of_a_slow_round_is_a_launch():
    """PROPERTY (the sim side): a slow marching round's FIRST ShotFiredEvent
    carries ``launch`` and every later one does not. BREAKS if the flag is
    set on every tick's segment (the plasma bolt would flash at its shooter
    for its whole flight) or on none (slow rounds would never flash)."""
    from level_loader import LevelData
    from simulation import Simulation
    from simulation.events import ShotFiredEvent
    from simulation.orders import ORDER_FIRE, Order
    from simulation.unit import Unit

    tm = np.zeros((24, 24), dtype=np.int32)
    tm[0, :] = tm[-1, :] = tm[:, 0] = tm[:, -1] = 1
    level = LevelData(name="muzzle", version="2", path=Path("."), tilemap=tm,
                      tile_size_m=1.0, diffuse_path=Path("."))
    sim = Simulation(level, seed=20261004, breach_physics=None,
                     enable_recorder=False)
    shooter = Unit("S", x=3, y=9, team=0)
    shooter.weapon_id = "sunspot"                     # 1.5 t/t plasma bolt
    sid = sim.add_unit(shooter)
    assert sim.apply_action(sid, Order(ORDER_FIRE, target_fx=20, target_fy=10,
                                       phase=0))
    flags = []
    for _ in range(14):
        sim.set_paused(False)
        sim.step()
        flags += [e.launch for e in sim.tick_events
                  if isinstance(e, ShotFiredEvent)]
    assert len(flags) >= 3, "the bolt should fly several ticks"
    assert flags[0] is True and not any(flags[1:])


def test_a_big_volley_stays_inside_the_light_budget():
    """PROPERTY: however many units fire in the same frame, their flashes
    together carry at most ``MUZZLE_FLASH_MAX_TOTAL`` light units (half the
    sweep's cone budget), every shooter still flashes, and the cone rows they
    make pass ``cone_rows``' budget door. BREAKS if the flashes are summed
    unbounded -- a firefight of enough shooters would make ``cone_rows``
    raise mid-game -- or if the cap drops shooters instead of dimming them."""
    from renderer.frame_lights import (LIGHT_CONE_BUDGET, MUZZLE_FLASH_MAX_TOTAL,
                                       cone_rows, transient_specs)
    shooters = 200
    effects = [_tracer(0.0, unit=u, frm=(10.0 + 3 * (u % 20), 10.0 + 3 * (u // 20)),
                       to=(80.0, 50.0)) for u in range(shooters)]
    lights = _lights(effects)
    assert len(lights) == shooters
    assert sum(d["intensity"] for d in lights) * max(FLASH.color) \
        <= MUZZLE_FLASH_MAX_TOTAL * (1 + 1e-9)
    rows = cone_rows(transient_specs(lights), 128, 128, 37)   # L_FINE_BITS
    assert np.all(rows[:, 2:5].sum(axis=0) <= LIGHT_CONE_BUDGET // 2 + shooters)
