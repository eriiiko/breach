"""The fire + smoke studio (#12 fire + smoke session; Erik, 2026-10-22): a room
that is burning from the start, the squad north of the fire.

Only the level's PURPOSE is asserted -- every igniter fires once and has fuel
inside its radius, and the squad stands north of the fuel. Placement, sizes and
timing are Erik's to retune and no test reads them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import level_loader  # noqa: E402
import tomllib  # noqa: E402

LEVELS = str(ROOT / "levels")
with open(ROOT / "config.toml", "rb") as _fh:
    CFG = tomllib.load(_fh)


def test_every_igniter_fires_once_onto_fuel_south_of_the_squad():
    """PROPERTY: the studio burns without anyone lighting it -- each timed
    charge fires ONCE (`period_s == 0`), its payload ignites or heats, at
    least one flammable tile lies inside the disc it lights or heats -- and every marine spawn
    stands north of all the fuel, so the fire throws their shadows north.

    BREAKS IF: an igniter is moved off its fuel, its payload stops
    igniting or heating, a charge is given a repeat period, or the squad is placed
    level with or south of the fire.
    """
    lvl = level_loader.load("fire_smoke_studio", levels_dir=LEVELS)
    tm = np.asarray(lvl.tilemap)
    # material ids are positional in [materials.*] (MaterialTable asserts it)
    # ...minus 9: foliage's id is numerically the CSV's SPACE code (CLAUDE.md,
    # "Material / gas tables"), and a tilemap's 9 is space.
    flammable_ids = [i for i, row in enumerate(CFG["materials"].values())
                     if row.get("flammable", False) and i != 9]
    fuel = np.isin(tm, flammable_ids)
    assert fuel.any()
    charges = [e for e in lvl.entities if e.class_name == "timed_charge"]
    assert charges, "nothing lights the studio"
    ys, xs = np.mgrid[0:tm.shape[0], 0:tm.shape[1]]
    for e in charges:
        assert float(e.fields.get("period_s", 0.0)) == 0.0
        row = CFG["payloads"][e.fields["payload"]]
        # a fire is started by delivering heat (CLAUDE.md "Starting a fire"):
        # an ignition ring, or heat alone that apply_temperature_ignition lights
        r = max(float(row.get("ignite_radius", 0.0)) if row.get("ignite_radius", 0.0) else 0.0,
                float(row.get("heat_radius", 0.0)) if row.get("heat_amount", 0.0) else 0.0)
        assert r > 0, f"{e.id}'s payload delivers no ignition and no heat"
        near = (xs - e.fields["x"]) ** 2 + (ys - e.fields["y"]) ** 2 <= r * r
        assert (near & fuel).any(), f"{e.id} has no fuel inside its ring"
    fuel_top = int(np.nonzero(fuel.any(axis=1))[0].min())
    for s in lvl.spawns:
        assert s.y + s.footprint <= fuel_top, f"{s.name} is not north of the fire"
