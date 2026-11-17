"""#29 P6b.4 -- THE HAZARD SCENE: timed charges and heat sources, built for the
P6b.4 gates (tests/cuda_unit_hazards_check.py, tests/test_hazard_tables.py).
Not a test module (no ``test_`` prefix): a harness both import.

1 m tiles, 24 x 48, hull border, a wood wall column at x 24 with a two-tile
gap (y 11, 12), a hull wall at x 36 (y 13..22) holding a closed door (y 14..16).

TIMED CHARGES (ordinals shuffled, never row indices; the list order is not
the ordinal order):
- ``t_frag`` frag_standard (a STRENGTH row) at (6, 5), first 1.0 s, every 2 s:
  its first blast starts the CHAIN of three heat sources in one tick;
- ``t_once`` studio_ignite (a DISC row) at (12, 30), once at 0.1 s;
- ``t_heat0`` a HEAT charge (fuse 0) by the heater h_heat, plasma_splash_small;
- ``t_heatf`` a HEAT charge with a 0.3 s fuse by h_heat, studio_ignite;
- ``t_heatoff`` a disabled heat charge; ``t_off`` a disabled timer;
- ``t_breaker`` studio_hull at (3, 22), once at 3.0 s, ordinal BEFORE
  ``t_wall``: a heat charge ON the wood wall tile (3, 24), heated by h_wall,
  ``t_wallhit`` plasma_splash_large at (8, 23): its disc hurts h_behind across
  the intact wood wall (the disc law reads no wall), h_gap is hit through the
  gap by ``t_gap``;
  disabled until the tick before t_breaker fires: the blast destroys its tile
  (T := 0) in the same sweep, so it reads a cold tile (the AT ONCE case);
- ``t_room`` studio_medium (an explosive DISC row) by the door, a vent and a
  pump in the right room, once at 2.0 s; ``t_sync`` frag_standard firing on
  the same tick as a planted C4 (placed by the fixture).

HEAT SOURCES: the chain h1 -> h2 -> h3 (h1 engine_death, h2
plasma_splash_large, h3 no death payload); h_gren (killed by a thrown frag);
h_c4 (killed by the planted C4); h_heat (radius 2, the heat charges' heater);
h_cap (radius 0, max_temp: capped); h_off (disabled); h_wall (heats the
wall); h_behind (behind the wood wall from t_frag's second blast: the disc
law reads no wall) and h_gap (across the gap); h_tough (hp large: survives).

A bus: h1's ``alive`` wired to a decider driving a pump (the reference's
emit reads h1's EntityInstance, so the signal stays 1: a reported finding).

``variant``: -1 the same walls with NO charge and NO heat source (and no
bus); odd: planetside; >= 2: fewer sources, other timings (the batch).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

H, W = 24, 48


def _entity(eid, cls, ordinal, **fields):
    from level_loader import EntityInstance
    from simulation.entities import REGISTRY
    f = {x.name: x.default for x in REGISTRY[cls].FIELDS}
    f.update(fields)
    return EntityInstance(id=eid, class_name=cls, ordinal=ordinal, fields=f)


def _tc(eid, x, y, payload, first=0.0, period=0.0, enabled=True, trigger="timer",
        temp=0.0):
    return _entity(eid, "timed_charge", 0, x=x, y=y, payload=payload, first_at_s=first,
                   period_s=period, enabled=enabled, trigger=trigger, trigger_temp=temp)


def _hs(eid, x, y, heat, radius=0, hp=200, death="", enabled=True, max_temp=0.0):
    return _entity(eid, "heat_source", 0, x=x, y=y, heat_per_s=heat, radius=radius,
                   hp=hp, death_payload=death, enabled=enabled, max_temp=max_temp)


def hazard_level(variant=0):
    from level_loader import LevelData, SPACE_CODE, Wire
    from simulation.entities import REGISTRY
    from simulation.materials import MAT_AIR, MAT_HULL, MAT_WOOD
    v = variant
    tm = np.full((H, W), MAT_HULL, dtype=np.int32)
    tm[1:H - 1, 1:W - 1] = MAT_AIR
    tm[1:H - 1, 24] = MAT_WOOD
    tm[11:13, 24] = MAT_AIR                       # the gap
    tm[13:H - 1, 36] = MAT_HULL                   # the right room's wall (door span y 14..16)
    tm[14:17, 36] = MAT_AIR
    if v % 2 == 1:
        tm[0, 30:34] = SPACE_CODE                 # never reached: planetside keeps the ring
    doors = [_entity("dr", "door", 0, x=36, y=14, orientation="v", length_m=3.0,
                     initial_state="closed")]
    ents = list(doors)
    if v >= 0:
        sh = 0 if v < 2 else v
        ents += [
            _tc("t_frag", 5, 6, "frag_standard", first=1.0, period=2.0),
            _tc("t_once", 30, 12, "studio_ignite", first=0.1),
            _tc("t_heat0", 12, 18, "plasma_splash_small", trigger="heat", temp=30.0),
            _tc("t_heatf", 12, 20, "studio_ignite", first=0.3, trigger="heat", temp=20.0),
            _tc("t_heatoff", 11, 18, "studio_ignite", enabled=False, trigger="heat", temp=0.5),
            _tc("t_off", 2, 2, "frag_standard", first=0.5, enabled=False),
            _tc("t_room", 34, 15, "studio_medium", first=2.0 + 0.25 * sh),
            _tc("t_sync", 44, 3, "frag_standard", first=2.5),
            _hs("h1", 6, 6, 40.0, hp=50, death="engine_death"),
            _hs("h2", 8, 6, 40.0, hp=60, death="plasma_splash_large"),
            _hs("h3", 10, 6, 40.0, hp=10),
            _hs("h_gren", 7, 16, 30.0, hp=40, death="plasma_splash_small"),
            _hs("h_c4", 42, 20, 30.0, hp=100, death="studio_ignite"),
            _hs("h_heat", 10, 19, 3.0 + 0.5 * sh, radius=2, hp=5000),
            _hs("h_cap", 16, 4, 3000.0, hp=5000, max_temp=1.0),
            _hs("h_off", 18, 4, 3000.0, hp=5000, enabled=False),
            _hs("h_behind", 25, 8, 20.0, hp=5000),
            _tc("t_wallhit", 23, 8, "plasma_splash_large", first=1.2),
            _hs("h_gap", 25, 11, 20.0, hp=5000),
            _tc("t_gap", 23, 11, "plasma_splash_large", first=1.5),
            _hs("h_tough", 33, 15, 20.0, radius=1, hp=100000),
            _entity("pm", "pump", 0, x=33, y=17, port_dy=-1, rate=4.0, target_atm=0,
                    hysteresis_band=13107),
            _entity("dh", "decider", 0, comparator="ge", threshold=1),
            _entity("dk", "duct", 0, filter="derelict"),
        ]
        if v < 2:
            ents += [_tc("t_breaker", 22, 3, "studio_hull", first=3.0),
                     _tc("t_wall", 24, 3, "studio_ignite", enabled=False, trigger="heat",
                         temp=0.05),
                     _hs("h_wall", 23, 4, 4000.0, radius=1, hp=100000)]
        if v in (0, 1, 3):
            ents += [_hs("h_extra", 30, 4, 25.0, radius=1, hp=300)]
    by = {e.id: e for e in ents}
    if "dk" in by:
        ents += [_entity("vs", "vent", 0, x=33, y=14, duct="dk", role="supply", q_circ=1.0),
                 _entity("vr", "vent", 0, x=40, y=19, duct="dk", role="return", q_circ=1.5)]
    rng = np.random.default_rng(9100 + variant)
    ords = list(rng.permutation(len(ents)) + 100)          # never a row index
    for e, o in zip(ents, ords):
        e.ordinal = int(o)
    # t_breaker sweeps BEFORE t_wall (ordinal order is the sweep's)
    by = {e.id: e for e in ents}
    if "t_breaker" in by and by["t_breaker"].ordinal > by["t_wall"].ordinal:
        by["t_breaker"].ordinal, by["t_wall"].ordinal = by["t_wall"].ordinal, \
            by["t_breaker"].ordinal
    wires = []

    def wire(src, sig, dst, inp):
        mode = {i.name: i.mode for i in REGISTRY[by[dst].class_name].INPUTS}[inp]
        wires.append(Wire(source_ordinal=int(by[src].ordinal), signal=sig,
                          target_ordinal=int(by[dst].ordinal), input=inp,
                          aggregation_mode=mode))
    if v >= 0:
        wire("h1", "alive", "dh", "in")           # a heat source's alive into a node
        wire("dh", "out", "pm", "inject")
    order = list(rng.permutation(len(ents)))
    ents = [ents[i] for i in order]               # the list order is not the ordinal order
    return LevelData(name=f"p6b4_hazards{variant}", version="3", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."), entities=ents,
                     wires=wires, boundary="ambient" if variant % 2 else "vacuum")


class HazardScene:
    """The hazard scene, driven by fixtures between ticks: a frag lob at
    h_gren, a planted C4 timed onto t_sync's tick, t_wall enabled the tick
    before t_breaker fires. ``bp`` the breach_physics module (CPU or CUDA)."""

    def __init__(self, variant=0, bp=None):
        if bp is None:
            import breach_physics as bp
        from simulation import Simulation
        from simulation.ruleset import ContinuousRealtime
        self.v, self.k = variant, 0
        sim = Simulation(hazard_level(variant), seed=64400 + max(variant, 0) * 11,
                         breach_physics=bp, enable_recorder=False,
                         ruleset=ContinuousRealtime())
        sim.set_paused(False)
        self.sim = sim
        self.byid = {e.id: e for e in sim.entities}
        from simulation.unit import Unit
        self.units = []
        for name, x, y, team in (("U0", 5.0, 7.0, 0), ("U1", 38.0, 18.0, 2),
                                 ("U2", 12.0, 13.0, 0)):
            if variant == -1 and name == "U2":
                continue
            u = Unit(name, x=x, y=y, team=team, footprint=1)
            sim.add_unit(u)
            u.current_hp = 1.0e6
            self.units.append(u)
        self.planted_tick = None

    def is_terminal(self):
        return False

    def _ttf(self, eid):
        e = self.byid.get(eid)
        return None if e is None else int(e.ticks_to_fire)

    def tick(self):
        from simulation import charges
        sim, k = self.sim, self.k
        if self.v >= 0:
            if k == 30:                            # a frag lob onto h_gren (fuse 4 ticks)
                from simulation.combat import Projectile
                from simulation.simulation import ORDER_GRENADE
                p = Projectile(ORDER_GRENADE, 6.0, 16.0, 7.0, 15.0, 0.0, sim.tick,
                               ammo_name="grenade_frag", owner_id=self.units[0].id,
                               fuse_ticks=4)
                sim.projectiles.append(p)
            t = self._ttf("t_sync")
            if self.planted_tick is None and t is not None and 0 < t <= 20:
                # a C4 timed onto t_sync's tick (charges.tick: det_tick <= tick)
                c = charges.PlantedCharge(42, 18, "demo_c4", self.units[1].id,
                                          det_tick=sim.tick + t, approach=None)
                charges.place(sim, c)
                self.planted_tick = c.det_tick
            tb = self._ttf("t_breaker")
            if tb == 0 and "t_wall" in self.byid:  # t_wall waits from the firing tick
                self.byid["t_wall"].enabled = True
        sim.step()
        sim.set_paused(False)
        self.k += 1
