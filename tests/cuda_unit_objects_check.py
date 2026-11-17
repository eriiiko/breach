"""#29 P7b -- THE GRENADES AND CHARGES GATE: the grenade table, slot 1b's
throw / plant appends, slot 2's grenades ("2 grenades": the flight and the
fuse-out as PAYLOAD records), slot 9e(d') ``charges.tick`` (its fires as
PAYLOAD records, its prune) and slot 10's grenade prune on the device object
tables, each against its Python law at tol 0 (design 4.4c, the cut table's
P7b row).

Runs inside the GPU subprocess (tests/cuda_harness.run_cuda_script; wrapper
tests/test_cuda_unit_objects.py). The tables are ``simulation.object_tables.
DeviceTable`` over ``unit_schema.GRENADE_TABLE`` / ``CHARGE_TABLE``; the
PAYLOAD records go to a ``simulation.applier.DeviceApplier`` effect list; the
kernels are ``simulation.unit_kernels.table_appends`` / ``grenades`` /
``charges_tick`` / ``prune_grenades`` (``cpp/src/cuda_unit_objects.cu``); the
harness is ``tests/cuda_unit_lockstep.py`` with the lists ``grenades``,
``charges`` (the host lists through ``object_tables.records``) and
``payloads`` (the host's ``execute_payload`` calls at the two P7b call sites,
through ``applier.record_of``), compared record for record, in order.

PART TABLES -- every P7b device table row equals its reference object's
value; the grenade table's fingerprint is compiled in.
PART IDENTITY -- on the grenade scene, every tick: ``unpack(pack(list))`` is
the host's grenade list attribute for attribute (``vars``), and a device
round trip of the packed rows is bit for bit.
PART FLIGHT -- synthetic full tables (random starts, targets, speeds, fuses,
elapsed ticks over the proved range; the edge cases: a zero flight, a flight
one below / at the landing epsilon, an arrival exactly on the tick, a row
already detonated, a proj_type that is not a lob, a fuse due exactly now):
the device's rows and PAYLOAD records equal ``Simulation._update_grenades``
on the same list (its ``execute_payload`` recorded, not run); a WEGO row (no
integer fuse) refused by name and left as it was; a row past the range
refused (UnitKernelFault) and left as it was.
PART SCENE -- the SHADOW LOCKSTEP of slot 1b (+ its appends), slot 2's
grenades, slot 9e(d') and slot 10 on the grenade scene (every grenade kind
thrown, several in flight, six fuse-outs on one tick whose records differ --
order proven to matter --, a throw released on a fuse-out tick, a grenade
landing at once, a plant, timers running out, two charges firing on one
tick, a remote charge, a full charge table refusing a plant, an empty table
and a non-empty one): tables, PAYLOAD records, rows and requests equal every
tick; a CENSUS of the Python law's branches, each required, the unreachable
ones named; the hooked match equal to an unhooked twin.
PART BOUNCE (#215) -- the bouncing scene (BounceScene: a long wall, a gap
SEALED while a grenade flies at it, a wall tile DESTROYED while one flies at
it, a fuse-out mid-leg in front of the wall, a horizontal wall, a pillar, a
corridor three tiles wide) at the shipped restitution, at 0 and at 1, each
shadowed every tick (the v0 run against an unhooked twin); its law facts on
the host (the closed gap bounced it, the destroyed tile let it through, the
mid-leg fuse-out stayed on the thrower's side); the three and a grenade scene
as a batch of N = 4 (env i == env i alone); the bounce branches' census; the
chained 1b + 1b' + 1c + 2 on the bouncing scene. PART FLIGHT's synthetic rows
start on the grid and aim mostly off it (most meet the hull), include exact
45-degree corners, and run at the three restitutions, the bounce census
required there too.
PART BATCH -- N = 2 and 8 of differing envs (the scene's variants: another
room, seed, unit count and schedule; P6.4a's verb scene; the charge scene;
ship_demo): env i == env i alone (the harness's alone run) every tick.
PART PADDING -- garbage rows past each table's count (live, due, fused-out
values) neither read nor written: the valid rows and the records equal a
clean run.
PART OVERFLOW -- a grenade table one row short on a tick with several
throws, a charge table one row short on a tick with two accepted plants, an
effect list of capacity 1 on a tick with several fuse-outs: the first rows /
records written (the host's first), the rest counted, the run refused by
name (the table and the env; EffectOverflow).
PART REFUSE -- the device door refuses a ruleset that drives the units
(``charges.detonate_owned``) and a WEGO grenade, by name.
PART CHAINED -- 1b (+ appends) + 1b' + 1c + 2 grenades, and 9e(d') alone,
replacing the host's slots in a twin match (the tables landed back, the
PAYLOAD records replayed through the host's call): the unit hash, the field
digest and both lists equal an untouched reference every tick.
PART SHIP -- ship_demo's bots match, 500 ticks, the four slots shadowed;
how often each acts is printed.

PROPERTY: the grenade table is the host list; slot 1b's throws and accepted
plants land in it in the reference's order; the flight is
``Projectile.fly`` / ``settle`` (#215: the leg, the wall walk over the solid
plane as the slot found it, the bounce) integer for integer; every fuse-out
(at ``combat.detonation_tile``) and every
due charge is ONE PAYLOAD record equal to the host's ``execute_payload``
call, in list order; the prunes are the stable compaction; in a batch as
alone. BREAKS IF a column's encode / decode, the flight's distance, landing
test or rounding, the wall walk's tile order, stop point or flips, the
bounce's length, rounding or rest rules, the settle, the detonation tile,
the fuse test, a record field, the record order, an
append's fields or order, a compaction, the batch indexing or a capacity
check differs from the law.

Prints per-part lines and ``UNIT_OBJECTS_RESULT: PASS`` / ``FAIL``; exits
0 / 1. ``bench()`` prints the TIMING table.
"""
from __future__ import annotations

import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

import breach_physics as bp   # the CUDA build (cuda_harness bootstrap): FIRST

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "tools") not in sys.path:
    sys.path.append(str(_ROOT / "tools"))   # appended: never ahead of the CUDA build

import level_loader  # noqa: E402
from cuda_unit_lockstep import (ChainedSlots, DeviceSlot,  # noqa: E402
                                ShadowLockstep)
from simulation import (applier, object_tables as ot, unit_kernels,  # noqa: E402
                        unit_requests, unit_schema, unit_store, unit_tables)

FAILS: list = []
ONE = 65536
PI_Q16 = 205887
GT = unit_schema.GRENADE_TABLE
CT = unit_schema.CHARGE_TABLE
GLAY = ot.table_layout(GT)
CLAY = ot.table_layout(CT)
SCENE_TICKS = int(os.environ.get("UNIT_OBJECTS_SCENE_TICKS", "340"))
BATCH_TICKS = int(os.environ.get("UNIT_OBJECTS_BATCH_TICKS", "200"))
SHIP_TICKS = int(os.environ.get("UNIT_OBJECTS_SHIP_TICKS", "500"))
BATCH_NS = [int(x) for x in os.environ.get("UNIT_OBJECTS_BATCH_NS", "2,8").split(",")]

#: What the PAYLOAD law writes on the host inside slot 2 / 9e(d') -- the
#: attributes the WRITE-SET gate allows as host-only: execute_payload is
#: recorded here, applied by P7c.1 (these are what P7c.1 inherits).
PAYLOAD_WRITES = (
    "blast_log", "noise_log", "damage_credit", "render_dirty",
    "div_source", "expansions", "wall_hp", "material", "solid", "obstacles",
    "is_vacuum", "is_ambient", "thermal_solid", "flammable", "sight_screen",
    "heat_inv_shift", "face_shift", "conductivity", "permeability",
    "wave_absorb", "heat_atten", "heat_atten_q", "light_atten",
    "light_atten_q", "fuel_recip", "fire_T_ext_plane", "fuel_per_o2_plane",
    "gas", "gas_energy", "temperature", "fire", "atmosphere",
    "n_destruction_seed_sum", "gas_energy_books", "solid_energy_books",
    "_tile_serial", "_sky_mask_dirty", "div_source_sat_hits",
    "div_source_cut_q", "dyn_heat_atten_q", "dyn_light_atten_q",
)
HOST_ONLY = tuple((a, "the PAYLOAD law's write: execute_payload is a RECORD "
                      "here, applied from P7c.1") for a in PAYLOAD_WRITES)


def fail(msg):
    FAILS.append(msg)
    print("FAIL:", msg)


# ---------------------------------------------------------------------------
# The slots
# ---------------------------------------------------------------------------
def _run_1b(st, inp, req, lists):
    cov = unit_kernels.verbs(st, req, inp.tick, inp.tables["door_spans"],
                             inp.scalars["charges_live"])
    unit_kernels.table_appends(st, req, lists["grenades"], lists["charges"])
    return cov


def _flight_fields(inp):
    """#215: the padded solid plane (``cuda_unit_lockstep``'s
    ``flight_solid`` input) as the kernel's device field."""
    import cupy as cp
    return {"solid": cp.ascontiguousarray(cp.asarray(inp.tables["flight_solid"]))}


def _run_2(st, inp, lists):
    return unit_kernels.grenades(st, lists["grenades"], lists["payloads"], inp.tick,
                                 _flight_fields(inp))


def _run_9e(st, inp, lists):
    return unit_kernels.charges_tick(st, lists["charges"], lists["payloads"], inp.tick)


def _run_10(st, inp, lists):
    unit_kernels.prune_grenades(lists["grenades"])
    return np.zeros(st.shape[0], dtype=np.int64)


SLOT_1B = DeviceSlot(label="1b intents", run=_run_1b,
                     inputs=("tick", "door_spans", "charges_live", "list:grenades",
                             "list:charges"),
                     requests=True, outputs=("list:grenades", "list:charges"))
# The grenade / charge slots write NO unit row (columns=("id",): the compare
# holds the rows the device never touches); the host's row changes inside
# them are the PAYLOAD law's (its blast damage), recorded here, P7c.1's.
SLOT_2 = DeviceSlot(label="2 grenades", run=_run_2, columns=("id",),
                    inputs=("tick", "list:grenades", "flight_solid"),
                    outputs=("list:grenades", "list:payloads"), host_only=HOST_ONLY)
SLOT_9E = DeviceSlot(label="9e(d') planted charges", run=_run_9e, columns=("id",),
                     skip_ok=True,    # the conductor runs it only while a charge lies
                     inputs=("tick", "list:charges"),
                     outputs=("list:charges", "list:payloads"), host_only=HOST_ONLY)
SLOT_10 = DeviceSlot(label="10 tick end (ruleset)", run=_run_10, columns=("id",),
                     inputs=("list:grenades",), outputs=("list:grenades",))
SLOTS = [SLOT_1B, SLOT_2, SLOT_9E, SLOT_10]


# ---------------------------------------------------------------------------
# The reference values of the P7b table rows (cuda_unit_kernels_check's
# every-row gate reads this too)
# ---------------------------------------------------------------------------
def table_reference(sim) -> dict:
    from config import CFG
    from simulation import combat
    from simulation.orders import ORDER_GRENADE
    wt = sim.weapons_tables
    ammo = list(wt.ammo.by_name)
    pl = list(wt.payloads.by_name)
    return {
        "clock_tps": int(CFG.clock.ticks_per_second),
        "grenade_flight_eps_q": int(combat._FLIGHT_EPS_Q),
        "order_grenade": int(ORDER_GRENADE),
        "hand_grenade_ammo_row": tuple(ammo.index(n) for n in wt.hand_grenade_rows),
        "hand_grenade_speed_q": tuple(
            combat.Projectile(ORDER_GRENADE, 0, 0, 1, 1, 0.0, 0, ammo_name=n).travel_speed_q
            for n in wt.hand_grenade_rows),
        "ammo_payload_row": tuple(
            pl.index(wt.payload_for_ammo(n).name) if wt.ammo.by_name[n].payload else -1
            for n in ammo),
    }


# ---------------------------------------------------------------------------
# The scenes
# ---------------------------------------------------------------------------
def _room(variant):
    from level_loader import LevelData
    from simulation.materials import MAT_AIR, MAT_HULL
    h, w = 48 + 4 * (variant % 2), 72 + 8 * (variant % 3)
    tm = np.full((h, w), MAT_HULL, dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = MAT_AIR
    return LevelData(name=f"p7b_room{variant}", version="3", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."), entities=[])


NORTH, SW = PI_Q16 // 2, 3 * PI_Q16 // 4      # SW: release (c, -s) = (-0.71, -0.71)


class GrenadeScene:
    """Throwers of every grenade kind and planters in one room, driven by
    scripted intents through the intent facade (see the module docstring).
    ``variant`` shifts the room, the seed, the schedule and adds bystanders;
    variant 0 alone pre-fills the charge table to one below its capacity
    before two planters lay on one tick (the second refused: a full
    table)."""

    def __init__(self, variant=0):
        from simulation import Simulation
        from simulation.ruleset import ContinuousRealtime
        self.v, self.k = variant, 0
        sim = Simulation(_room(variant), seed=7700 + variant, breach_physics=bp,
                         enable_recorder=False, ruleset=ContinuousRealtime())
        sim.set_paused(False)
        self.sim = sim
        self.n_kinds = len(sim.weapons_tables.hand_grenade_rows)
        self.t = [self._add(f"T{i}", 6 + 6 * i, 40, NORTH) for i in range(self.n_kinds)]
        for u in self.t:
            u.grenade_counts = [4] * self.n_kinds
        self.lander = self._add("L", 60, 12, SW)
        self.lander.grenade_counts = [2] * self.n_kinds
        self.p = [self._add(f"P{i}", 8 + 6 * i, 6, NORTH) for i in range(3)]
        for i in range(variant % 3):
            self._add(f"B{i}", 50 + 3 * i, 30, NORTH)

    def _add(self, name, x, y, facing):
        from simulation.turning import place
        from simulation.unit import Unit
        u = Unit(name, x=float(x), y=float(y), team=0)
        self.sim.add_unit(u)
        place(u, facing)
        u.current_hp = 1.0e6
        return u

    def _throw(self, u, kind, fuse, reach_t):
        self.sim.throw_grenade_intent(u.id, 0, -ONE, fuse, reach_q=reach_t * ONE,
                                      kind=kind)

    def _lay(self, det, x):
        from simulation import charges
        charges.place(self.sim, charges.PlantedCharge(
            x, 20, "demo_c4", -1, det_tick=det, approach=None))

    def tick(self):
        from simulation import charges
        sim, k, v = self.sim, self.k, self.v
        s = 2 + v                                   # the schedule's offset
        if k == s:                                  # every kind, one fuse: 6 out on one tick
            for i, u in enumerate(self.t):
                self._throw(u, i, 30, 8 + i % 3)
        if k == s + 30:                             # released ON that fuse-out tick
            self._throw(self.t[0], 1, 12, 10)
        if k == s + 3:                              # several in flight, staggered
            for i, u in enumerate(self.t[1:4]):
                self._throw(u, (i + 2) % self.n_kinds, 50 + 9 * i, 14)
        if k == s + 5:                              # a flight shorter than the epsilon
            sim.throw_grenade_intent(self.lander.id, 0, 0, 20, reach_q=46341, kind=0)
        if k == s + 20:                             # a 0-tick fuse: out on its release
            self._throw(self.t[5], 0, 0, 6)
        if k == 1 and v == 0:                       # a remote charge and a full table
            self._lay(None, 60)
            while unit_requests.charges_live(sim) < unit_schema.CHARGE_CAPACITY - 1:
                self._lay(10 ** 9, 64)
        if k == s + 10:                             # two plants, one tick: same det_tick
            for u in self.p[:2]:
                sim.plant_charge_intent(u.id, 120)
        if k == s + 12:
            sim.plant_charge_intent(self.p[2].id, 150)
        if k == s + 90 and v != 0:                  # timers running out (fixtures)
            for j, d in enumerate((5, 5, 9)):
                self._lay(sim.tick + d, 10 + 4 * j)
        if k == s + 200:                            # a late wave
            for i, u in enumerate(self.t):
                self._throw(u, (i + 3) % self.n_kinds, 20 + 2 * i, 9)
        if k == 260 and v == 0:                     # the table empties: the remote fires
            for c in sim.planted_charges:           # a fixture: its timer set now
                if c.det_tick is None:
                    c.det_tick = sim.tick + 3
        sim.step()
        sim.set_paused(False)
        self.k += 1
        _ = charges


def _bounce_room(variant):
    """#215: the bouncing scene's map -- a hull box with a long wall (column
    32, a two-tile gap at rows 18-19), a horizontal wall (row 30, columns
    40-60), a pillar (row 8, column 12) and a corridor three tiles wide
    (walls at columns 46 and 50, rows 2-12)."""
    from level_loader import LevelData
    from simulation.materials import MAT_AIR, MAT_HULL
    h, w = 40, 64
    tm = np.full((h, w), MAT_HULL, dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = MAT_AIR
    tm[2:38, 32] = MAT_HULL
    tm[18:20, 32] = MAT_AIR
    tm[30, 40:61] = MAT_HULL
    tm[8, 12] = MAT_HULL
    tm[2:13, 46] = MAT_HULL
    tm[2:13, 50] = MAT_HULL
    return LevelData(name=f"p215_bounce{variant}", version="3", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."), entities=[])


EAST, SOUTH_EAST = 0, 7 * PI_Q16 // 4


class BounceScene:
    """#215: throws that meet walls, driven through the intent facade: one
    into the long wall (a vertical edge), one through the gap whose tiles
    are SEALED while it flies (a door closing mid-flight), one at a wall
    tile DESTROYED while it flies (it passes), one whose short fuse runs out
    mid-leg in front of the wall (the rest of the flight at once), one into
    the horizontal wall (a horizontal edge), one at the pillar, one down the
    corridor (a bounce per three tiles: bounce_max ends it at restitution 1).
    ``variant`` 0 keeps the shipped restitution, 1 sets it to 0, 2 to 1
    (the sim's own hand-grenade row, which the device table row reads)."""

    def __init__(self, variant=0):
        from simulation import Simulation
        from simulation.ruleset import ContinuousRealtime
        self.v, self.k = variant, 0
        sim = Simulation(_bounce_room(variant), seed=21500 + variant, breach_physics=bp,
                         enable_recorder=False, ruleset=ContinuousRealtime())
        sim.set_paused(False)
        self.sim = sim
        hand = sim.weapons_tables.weapons.by_name["hand_grenade"]
        if variant == 1:
            hand.bounce_restitution_q = 0
        elif variant == 2:
            hand.bounce_restitution_q = ONE
        add = GrenadeScene._add
        self.e = [add(self, f"E{i}", 21, y, EAST) for i, y in enumerate((9, 17, 25, 3))]
        self.n = add(self, "N", 50, 34, NORTH)
        self.d = add(self, "D", 4, 1, SOUTH_EAST)
        self.c = add(self, "C", 47, 8, EAST)
        self.thrown = {}
        for u in self.e + [self.n, self.d, self.c]:
            u.grenade_counts = [3] * len(sim.weapons_tables.hand_grenade_rows)

    def tick(self):
        sim, k = self.sim, self.k
        throw = GrenadeScene._throw
        if k == 2:
            throw(self, self.e[0], 0, 40, 16)          # into the long wall
            throw(self, self.e[1], 0, 60, 16)          # through the gap ...
            throw(self, self.e[2], 0, 50, 16)          # at a wall tile ...
            throw(self, self.n, 0, 45, 12)             # into the horizontal wall
            throw(self, self.d, 0, 45, 14)             # at the pillar
            throw(self, self.c, 0, 90, 30)             # down the corridor
        if k == 3:
            throw(self, self.e[3], 0, 3, 20)           # a fuse out mid-leg
        if k == 12:                                    # ... the gap closes
            from simulation.materials import MAT_DOOR_CLOSED
            sim.gmap.seal_tiles([(18, 32), (19, 32)], MAT_DOOR_CLOSED)
        if k == 13:                                    # ... the tile goes
            for y in (25, 26, 27):
                sim.gmap.destroy_wall(y, 32)
        sim.step()
        sim.set_paused(False)
        for p in sim.projectiles:                      # kept past the prune
            self.thrown.setdefault(id(p), p)
        self.k += 1

    def by_owner(self, u):
        return [p for p in self.thrown.values() if p.owner_id == u.id]


class ShipDriver:
    """ship_demo's bots match (MatchSession)."""

    def __init__(self, seed=1):
        from simulation.match_session import MatchSession
        level = level_loader.load("ship_demo", str(_ROOT / "levels"))
        self.session = MatchSession(level, seed=seed, breach_physics=bp)
        self.sim = self.session.sim

    def tick(self):
        self.session.tick()

    def is_terminal(self):
        return self.session.is_terminal()


# ---------------------------------------------------------------------------
# The census: the Python law's branches, counted on the host's own decisions
# ---------------------------------------------------------------------------
class Census:
    """Wraps (reads only) ``Projectile.update_position``,
    ``Simulation._update_grenades``, ``charges.tick`` and
    ``ContinuousRealtime.on_tick_end`` and counts each branch the law takes."""

    def __init__(self):
        self.c = Counter()
        self._orig = []

    def install(self):
        from config import CFG
        from simulation import charges, combat, unit_fixed
        from simulation.orders import ORDER_GRENADE
        from simulation.ruleset import ContinuousRealtime
        from simulation.simulation import Simulation
        c = self.c
        P = combat.Projectile
        o_up = P.update_position

        def up(p, t, __o=o_up):
            tps = int(CFG.clock.ticks_per_second)
            el = int(t) - int(p.leg_tick)
            dx, dy = p.target_x_q - p.start_x_q, p.target_y_q - p.start_y_q
            d = unit_fixed.isqrt(dx * dx + dy * dy)
            if d < combat._FLIGHT_EPS_Q:
                c["flight: shorter than the epsilon (lands at once)"] += 1
            elif el * p.travel_speed_q >= d * tps:
                c["flight: arrived"] += 1
                if el * p.travel_speed_q == d * tps:
                    c["flight: arrives exactly on the tick"] += 1
            else:
                c["flight: in the air (the rounded rational)"] += 1
            return __o(p, t)
        P.update_position = up
        self._orig.append((P, "update_position", o_up))

        # #215: the bounce law's branches
        o_bn = P.bounce

        def bn(p, px, py, flip_x, flip_y, tick, rq, bmax, __o=o_bn):
            c["bounce: an exact corner (both flipped)" if flip_x and flip_y
              else "bounce: a vertical edge (x flipped)" if flip_x
              else "bounce: a horizontal edge (y flipped)"] += 1
            at_max = p.bounces >= int(bmax)
            r = __o(p, px, py, flip_x, flip_y, tick, rq, bmax)
            c["bounce: bounce_max reached (rests at the wall)" if at_max
              else "bounce: a leg under a quarter tile (rests)" if not r
              else "bounce: a mirrored leg"] += 1
            if not at_max and int(rq) == 0:
                c["bounce: restitution 0 (drops at the wall)"] += 1
            if r and int(rq) == ONE:
                c["bounce: restitution 1 (a mirror)"] += 1
            return r
        P.bounce = bn
        self._orig.append((P, "bounce", o_bn))

        o_st = P.settle

        def stl(p, gmap, tick, rq, bmax, __o=o_st):
            mid = (p.pos_x_q, p.pos_y_q) != (p.target_x_q, p.target_y_q)
            b0 = p.bounces
            r = __o(p, gmap, tick, rq, bmax)
            if mid:
                c["settle: a fuse-out mid-leg (the rest flown at once)"] += 1
                if p.bounces > b0 or (p.start_x_q, p.start_y_q) == (p.target_x_q,
                                                                    p.target_y_q):
                    c["settle: a wall met while settling"] += 1
            return r
        P.settle = stl
        self._orig.append((P, "settle", o_st))

        o_ug = Simulation._update_grenades

        def ug(sim, __o=o_ug):
            due = 0
            kinds = set()
            for p in sim.projectiles:
                if p.detonated:
                    c["grenades: a detonated row skipped"] += 1
                    continue
                if sim.tick >= p.get_detonate_tick():
                    due += 1
                    kinds.add(p.ammo_name)
                    c["grenades: fuse-out of a lob (PAYLOAD)" if p.proj_type == ORDER_GRENADE
                      else "grenades: fuse-out of a non-lob (no payload)"] += 1
                    if sim.tick == p.thrown_tick:
                        c["grenades: out on its release tick"] += 1
                else:
                    c["grenades: not due"] += 1
            if due >= 2:
                c["grenades: two or more fuse-outs on one tick"] += 1
            if len(kinds) >= 2:
                c["grenades: fuse-outs of different kinds on one tick"] += 1
            if due and any(p.thrown_tick == sim.tick and sim.tick < p.get_detonate_tick()
                           for p in sim.projectiles):
                c["grenades: a throw and a fuse-out on one tick"] += 1
            if len([p for p in sim.projectiles if not p.detonated]) >= 3:
                c["grenades: three or more in flight"] += 1
            for p in sim.projectiles:
                c[f"kind thrown: {p.ammo_name}"] = max(1, c[f"kind thrown: {p.ammo_name}"])
            return __o(sim)
        Simulation._update_grenades = ug
        self._orig.append((Simulation, "_update_grenades", o_ug))

        o_ct = charges.tick

        def ct(sim, __o=o_ct):
            fired = 0
            if not sim.planted_charges:
                c["charges.tick: an empty list (no prune)"] += 1
            for ch in sim.planted_charges:
                if not ch.live:
                    c["charges.tick: a dead charge skipped"] += 1
                elif ch.det_tick is None:
                    c["charges.tick: a remote charge (no timer)"] += 1
                elif sim.tick >= ch.det_tick:
                    fired += 1
                    c["charges.tick: a due charge fires (PAYLOAD)"] += 1
                    c["charges.tick: fires with an approach" if ch.approach is not None
                      else "charges.tick: fires without an approach"] += 1
                else:
                    c["charges.tick: a timer running"] += 1
            if fired >= 2:
                c["charges.tick: two or more fire on one tick"] += 1
            if len(sim.planted_charges) >= unit_schema.CHARGE_CAPACITY:
                c["charges: a full table"] += 1
            if fired and fired == len(sim.planted_charges):
                c["charges.tick: every charge fires (the table empties)"] += 1
            return __o(sim)
        charges.tick = ct
        self._orig.append((charges, "tick", o_ct))

        o_te = ContinuousRealtime.on_tick_end

        def te(rules, sim, __o=o_te):
            if any(p.detonated for p in sim.projectiles):
                c["slot 10: detonated grenades pruned"] += 1
            elif sim.projectiles:
                c["slot 10: grenades in flight, none pruned"] += 1
            return __o(rules, sim)
        ContinuousRealtime.on_tick_end = te
        self._orig.append((ContinuousRealtime, "on_tick_end", o_te))
        return self

    def uninstall(self):
        for cont, attr, orig in reversed(self._orig):
            setattr(cont, attr, orig)
        self._orig.clear()


#: Every branch the scene must take; and the ones it cannot, with the reason.
NEED = (
    "flight: shorter than the epsilon (lands at once)", "flight: arrived",
    "flight: in the air (the rounded rational)",
    "grenades: fuse-out of a lob (PAYLOAD)", "grenades: not due",
    "grenades: out on its release tick", "grenades: two or more fuse-outs on one tick",
    "grenades: fuse-outs of different kinds on one tick",
    "grenades: a throw and a fuse-out on one tick", "grenades: three or more in flight",
    "charges.tick: a remote charge (no timer)",
    "charges.tick: a due charge fires (PAYLOAD)", "charges.tick: fires with an approach",
    "charges.tick: fires without an approach", "charges.tick: a timer running",
    "charges.tick: two or more fire on one tick", "charges: a full table",
    "charges.tick: every charge fires (the table empties)",
    "slot 10: detonated grenades pruned", "slot 10: grenades in flight, none pruned",
    "1b: a throw appended", "1b: several throws on one tick",
    "1b: an accepted plant appended", "1b: a plant refused at a full table",
    "1b: two plants on one tick",
)
#: #215: the bounce law's branches PART FLIGHT's synthetic rows must take.
NEED_FLIGHT = (
    "bounce: an exact corner (both flipped)", "bounce: a vertical edge (x flipped)",
    "bounce: a horizontal edge (y flipped)", "bounce: bounce_max reached (rests at the wall)",
    "bounce: a leg under a quarter tile (rests)", "bounce: a mirrored leg",
    "bounce: restitution 0 (drops at the wall)", "bounce: restitution 1 (a mirror)",
    "settle: a fuse-out mid-leg (the rest flown at once)",
    "settle: a wall met while settling",
)
UNREACHABLE = {
    "charges.tick: an empty list (no prune)":
        "the conductor calls charges.tick only while a charge is planted (slot "
        "9e(d')'s gate); the harness compares the gated-off ticks as a run "
        "on the empty table that must record nothing",
    "grenades: a detonated row skipped":
        "slot 10's prune (ContinuousRealtime.on_tick_end) leaves no detonated "
        "grenade at slot 2's entry; PART FLIGHT holds the branch on synthetic rows",
    "grenades: fuse-out of a non-lob (no payload)":
        "every continuous throw is ORDER_GRENADE (the other proj_types are "
        "WEGO's); PART FLIGHT holds it",
    "charges.tick: a dead charge skipped":
        "on the continuous path a charge is dead only between charges.fire and "
        "the prune inside one charges.tick call (detonate_owned is WEGO's)",
    "flight: arrives exactly on the tick":
        "needs elapsed x speed == dist x tps with an irrational-free distance; "
        "PART FLIGHT holds it on constructed rows",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _store(sims, U=2):
    return unit_store.DeviceStore(len(sims), U, tables=unit_tables.values_batch(sims))


def _solid(sims):
    """#215: the sims' ``solid`` planes as the device ``(N, h, w)`` bool
    array slot 2's grenades read."""
    import cupy as cp
    return {"solid": cp.ascontiguousarray(cp.asarray(np.stack(
        [np.asarray(s.gmap.solid, dtype=np.bool_) for s in sims])))}


def _rows_of(table, objs):
    return np.array(ot.records(table, objs), dtype=np.int64).reshape(
        -1, ot.table_layout(table).width)


def _host_grenade_step(sim, objs, tick):
    """``Simulation._update_grenades`` on ``objs`` at ``tick`` with its
    execute_payload recorded and NOT run: (the objects after, the PAYLOAD
    records in call order)."""
    from simulation import simulation as simmod, simultaneous
    calls = []
    orig = simmod.execute_payload

    def rec(*a, **kw):
        calls.append((simultaneous._WORLD, orig, a, kw))
    simmod.execute_payload = rec
    old_p, old_t = sim.projectiles, sim.tick
    try:
        sim.projectiles, sim.tick = objs, tick
        sim._update_grenades()
    finally:
        sim.projectiles, sim.tick = old_p, old_t
        simmod.execute_payload = orig
    return objs, [tuple(int(x) for x in applier.record_of(sim, c)) for c in calls]


def _first_row_diff(got, want, cols):
    d = np.argwhere(got != want)
    if not len(d):
        return None
    i, f = (int(x) for x in d[0])
    return i, cols[f], int(want[i, f]), int(got[i, f])


def _cols(lay):
    return [c.name if c.width == 1 else f"{c.name}[{j}]" for c in lay.columns
            for j in range(c.width)]


# ---------------------------------------------------------------------------
# PART TABLES
# ---------------------------------------------------------------------------
def part_tables():
    from cuda_unit_verbs_check import VerbDriver
    sims = [GrenadeScene(0).sim, VerbDriver(0).sim]
    st = _store(sims)
    got = st.tables.download()
    n = 0
    for e, sim in enumerate(sims):
        for k, v in table_reference(sim).items():
            dv = unit_tables.read(got[e], k)
            n += 1
            as_t = lambda x: tuple(x) if isinstance(x, (tuple, list)) else (x,)  # noqa: E731
            if as_t(dv) != as_t(v):
                fail(f"tables env {e} row {k}: device {dv} != reference {v}")
    if bp.object_tables_layout()["fingerprint"] != ot.layout().fingerprint:
        fail("tables: the compiled object tables are not the schema's")
    print(f"tables: {n} P7b values over 2 envs equal the reference objects; the "
          f"grenade table (W {GLAY.width}, cap {GLAY.capacity}) is compiled in")


# ---------------------------------------------------------------------------
# PART FLIGHT
# ---------------------------------------------------------------------------
def _synthetic(rng, sim, n, tick):
    """``n`` grenades over the proved range plus the edge cases."""
    from simulation import combat, unit_fixed
    from simulation.orders import ORDER_GRENADE
    kinds = sim.weapons_tables.hand_grenade_rows
    eps = combat._FLIGHT_EPS_Q
    out = []
    for i in range(n):
        name = kinds[int(rng.integers(len(kinds)))]
        p = combat.Projectile(ORDER_GRENADE, 0, 0, 1, 1, 0.0, 0, ammo_name=name,
                              owner_id=int(rng.integers(-1, 40)), fuse_ticks=0)
        # #215: a grenade starts on the grid (a thrower's tile, or a bounce's
        # point); its target anywhere within +-2^24 -- far off the grid most
        # of the time, so most rows meet the walls and bounce (a mirror of
        # such a target in a grid line stays inside the proved +-2^25)
        lim = (1 << 24) if i % 3 else (1 << 18)
        gh, gw = sim.gmap.solid.shape
        p.start_x_q = int(rng.integers(0, gw * ONE))
        p.start_y_q = int(rng.integers(0, gh * ONE))
        p.target_x_q, p.target_y_q = (int(x) for x in rng.integers(-lim, lim + 1, size=2))
        if i % 4 == 1:                                  # a short flight on the grid
            p.target_x_q = p.start_x_q + int(rng.integers(-12 * ONE, 12 * ONE))
            p.target_y_q = p.start_y_q + int(rng.integers(-12 * ONE, 12 * ONE))
        p.pos_x_q, p.pos_y_q = p.start_x_q, p.start_y_q
        p.bounces = int(rng.integers(0, 6))
        p.fuse_ticks = int(rng.integers(0, 400))
        p.thrown_tick = tick - int(rng.integers(0, min(tick, 1 << 31) + 1)) \
            if i % 5 == 0 else tick - int(rng.integers(0, 500))
        case = i % 11
        if case == 1:                                   # a zero flight
            p.target_x_q, p.target_y_q = p.start_x_q, p.start_y_q
        elif case == 2:                                 # one below the epsilon
            p.target_x_q, p.target_y_q = p.start_x_q + eps - 1, p.start_y_q
        elif case == 3:                                 # at the epsilon, at release
            # (elapsed 0: only the epsilon test lands it; one tick later the
            # arrival test would too)
            p.target_x_q, p.target_y_q = p.start_x_q + eps, p.start_y_q
            p.thrown_tick = tick
            p.fuse_ticks = 5
        elif case == 4:                                 # arrives exactly now
            # a 3-4-5 flight of length 5k with 5k x tps == elapsed x speed
            # (k integral: the speed is a multiple of 5 x tps for 30 tiles/s
            # at 24 ticks/s; otherwise this row is an ordinary one)
            tps = int(unit_tables.read(unit_tables.values(sim), "clock_tps"))
            el = int(rng.integers(1, 60))
            if (el * p.travel_speed_q) % (5 * tps) == 0:
                k = (el * p.travel_speed_q) // (5 * tps)
                p.start_x_q = p.start_y_q = 0
                p.pos_x_q = p.pos_y_q = 0
                p.target_x_q, p.target_y_q = 3 * k, 4 * k
                p.thrown_tick = tick - el
        elif case == 5:                                 # already detonated
            p.detonated = True
        elif case == 6:                                 # not a lob
            p.proj_type = ORDER_GRENADE + 1
            p.fuse_ticks = 0
        elif case == 7:                                 # due exactly now
            p.fuse_ticks = tick - p.thrown_tick
        elif case in (8, 9):                            # #215: a 45-degree flight
            # through the lattice's corners into the room's corner (an EXACT
            # corner at the hull: both flipped), from a tile corner
            k = int(rng.integers(3, 40))
            sx, sy = (gw - 2 - k // 2) * ONE, (gh - 2 - k // 2) * ONE
            p.start_x_q, p.start_y_q = sx, sy
            p.pos_x_q, p.pos_y_q = sx, sy
            p.target_x_q, p.target_y_q = sx + k * ONE, sy + k * ONE
            if case == 9:                               # up and left
                p.start_x_q = p.pos_x_q = 2 * ONE + (k // 2) * ONE
                p.start_y_q = p.pos_y_q = 2 * ONE + (k // 2) * ONE
                p.target_x_q = p.start_x_q - k * ONE
                p.target_y_q = p.start_y_q - k * ONE
        p.leg_tick = p.thrown_tick if i % 6 else max(p.thrown_tick, tick - 3)
        out.append(p)
    _ = unit_fixed
    return out


def part_flight(trials=4, n_env=4):
    import copy
    rng = np.random.default_rng(29)
    sim = GrenadeScene(0).sim
    hand = sim.weapons_tables.weapons.by_name["hand_grenade"]
    shipped = hand.bounce_restitution_q
    cols = _cols(GLAY)
    seen = Counter()
    planes = _solid([sim] * n_env)
    census = Census().install()
    # #215: the shipped restitution, then 0 (drops at the wall) and 1 (a
    # mirror, so bounce_max ends it), each over every trial
    def run_trials():
        for rq in (shipped, 0, ONE):
            hand.bounce_restitution_q = rq
            st = _store([sim] * n_env)
            for t in range(trials):
                tick = int(rng.integers(500, 1 << 30))
                envs = [_synthetic(rng, sim, GLAY.capacity - (e % 3), tick) for e in range(n_env)]
                tab = ot.DeviceTable(GT, n_env)
                fx = applier.DeviceApplier(n_env, cap=GLAY.capacity)
                tab.stage_lists([_rows_of(GT, objs) for objs in envs])
                fx.begin()
                unit_kernels.grenades(st, tab, fx, np.full(n_env, tick, dtype=np.int64), planes)
                dev = tab.records()
                drec = fx.records()
                for e, objs in enumerate(envs):
                    after, recs = _host_grenade_step(sim, copy.deepcopy(objs), tick)
                    want = _rows_of(GT, after)
                    got = np.array(dev[e], dtype=np.int64).reshape(-1, GLAY.width)
                    d = _first_row_diff(got, want, cols)
                    if d is not None:
                        fail(f"flight trial {t} env {e}: row {d[0]} column {d[1]}: host "
                             f"{d[2]} device {d[3]}")
                        return
                    if drec[e] != recs:
                        fail(f"flight trial {t} env {e}: the PAYLOAD records differ "
                             f"({len(recs)} host / {len(drec[e])} device)")
                        return
                    seen["payload records"] += len(recs)
                    seen["rows"] += len(objs)
                    seen["detonated now"] += sum(1 for a, b in zip(objs, after)
                                                 if not a.detonated and b.detonated)
                    seen["bounced"] += sum(1 for a, b in zip(objs, after)
                                           if b.bounces > a.bounces)

    try:
        run_trials()
    finally:
        hand.bounce_restitution_q = shipped
        census.uninstall()
    for b in NEED_FLIGHT:
        if not census.c[b]:
            fail(f"flight: the census branch {b!r} was never taken")
    print("flight census: " + ", ".join(f"{k} {v}" for k, v in sorted(census.c.items())
                                        if k.startswith(("bounce", "settle"))))
    st = _store([sim] * n_env)
    planes = _solid([sim])
    # a WEGO row (no integer fuse): refused by name, left as it was
    from simulation import combat
    from simulation.orders import ORDER_GRENADE
    w = combat.Projectile(ORDER_GRENADE, 3, 3, 9.5, 3.5, 2.0, 0, owner_id=1)
    tab = ot.DeviceTable(GT, 1)
    fx = applier.DeviceApplier(1)
    st1 = _store([sim])
    tab.stage_lists([_rows_of(GT, [w])])
    fx.begin()
    try:
        unit_kernels.grenades(st1, tab, fx, np.array([5]), planes)
        fail("flight: a WEGO grenade (no integer fuse) was flown, not refused")
    except unit_kernels.ObjectSlotRefused as ex:
        if "WEGO" not in str(ex):
            fail(f"flight: the WEGO refusal does not name it: {ex}")
    if tab.records()[0] != ot.records(GT, [w]) or fx.records()[0]:
        fail("flight: a refused WEGO row was written or recorded")
    # a row past the range: a fault, left as it was
    g = combat.Projectile(ORDER_GRENADE, 3, 3, 9.5, 3.5, 0.0, 0, owner_id=1, fuse_ticks=1)
    g.target_x_q = (1 << 25) + 1
    tab.stage_lists([_rows_of(GT, [g])])
    fx.begin()
    try:
        unit_kernels.grenades(st1, tab, fx, np.array([5]), planes)
        fail("flight: a row past the proved range was not refused")
    except unit_kernels.UnitKernelFault:
        pass
    if tab.records()[0] != ot.records(GT, [g]) or fx.records()[0]:
        fail("flight: a faulted row was written or recorded")
    print(f"flight: 3 restitutions x {trials} trials x {n_env} envs of synthetic full tables equal "
          f"_update_grenades row for row and record for record {dict(seen)}; a WEGO "
          f"row refused by name, a row past the range refused, both untouched")


# ---------------------------------------------------------------------------
# PART IDENTITY + SCENE
# ---------------------------------------------------------------------------
def _identity(sim, where):
    rows, n = ot.pack(GT, sim.projectiles)
    back = ot.unpack(GT, rows, n)
    for a, b in zip(sim.projectiles, back):
        if vars(a) != vars(b):
            fail(f"identity {where}: unpack(pack) differs: {vars(a)} / {vars(b)}")
            return False
    return True


def run_shadow(name, drivers, ticks, census=None, twins=None, probe=None, slots=None):
    """Shadow ``slots`` (default all four) on ``drivers``; every tick every
    slot equal (rows, requests, tables, PAYLOAD records, the alone run when
    N > 1) and slot 1b's coverage == the rows the host changed; returns the
    per-slot count of env-ticks on which the slot acted."""
    slots = slots or SLOTS
    acts = Counter()
    writes = {}
    n = 0
    dev = ot.DeviceTable(GT, len(drivers))
    t0 = time.perf_counter()
    with ShadowLockstep(drivers, list(slots)) as ls:
        for _ in range(ticks):
            if any(getattr(d, "is_terminal", lambda: False)() for d in drivers):
                break
            reps = ls.tick()
            n += 1
            for lab, rep in reps.items():
                if not rep.equal:
                    fail(f"{name}: {ls.divergences[-1]}")
                    return acts, n
            r1 = reps.get(SLOT_1B.label)
            if r1 is not None and not np.array_equal(r1.coverage, r1.host_changed):
                fail(f"{name} [1b] tick {r1.tick}: coverage {r1.coverage.tolist()} != "
                     f"host-changed {r1.host_changed.tolist()}")
                return acts, n
            for lab, rep in reps.items():
                for ws in rep.write_set or ():
                    writes.setdefault(lab, set()).update(ws)
            for e in range(len(drivers)):
                for lab, rep in reps.items():
                    hl = rep.host_lists or {}
                    if lab == SLOT_1B.label:
                        rq = rep.host_requests[e]
                        thr = sum(1 for r in rq if r[0] == unit_requests.KIND_THROW)
                        acc = sum(1 for r in rq if r[0] == unit_requests.KIND_PLANT and r[12])
                        ref = sum(1 for r in rq if r[0] == unit_requests.KIND_PLANT
                                  and not r[12])
                        acts[lab] += bool(thr or acc)
                        if census is not None:
                            census["1b: a throw appended"] += thr
                            census["1b: several throws on one tick"] += thr >= 2
                            census["1b: an accepted plant appended"] += acc
                            census["1b: a plant refused at a full table"] += ref
                            census["1b: two plants on one tick"] += (acc + ref) >= 2
                    elif "payloads" in hl:
                        acts[lab] += bool(hl["payloads"][e])
                    elif lab == SLOT_10.label:
                        acts[lab] += any(r[GLAY.live] == 0 for r in
                                         ls.last_inputs[lab].lists["grenades"][e])
            for e, d in enumerate(drivers):
                if not _identity(d.sim, f"{name} tick {d.sim.tick} env {e}"):
                    return acts, n
            # the device round trip of the packed table, bit for bit
            rows, counts = ot.pack_batch(GT, [d.sim.projectiles for d in drivers])
            dev.upload(rows, counts)
            r2, c2, _ = dev.download()
            if not (np.array_equal(r2, rows) and np.array_equal(c2, counts)):
                fail(f"{name}: the grenade table's device round trip is not bit for bit")
                return acts, n
            if probe is not None:
                probe(reps, ls)
            if twins is not None:
                from cuda_unit_verbs_check import _same
                for d, tw in zip(drivers, twins):
                    tw.tick()
                    if not _same(d.sim, tw.sim):
                        fail(f"{name}: the hooked match diverged from the unhooked "
                             f"one at tick {d.sim.tick}")
                        return acts, n
    print(f"{name}: N={len(drivers)}, {n} ticks, every slot equal every tick"
          f"{' (env i == env i alone)' if len(drivers) > 1 else ''}"
          f"{' (the unhooked twin equal)' if twins is not None else ''}; "
          f"env-ticks acted: {dict(acts)} ({time.perf_counter() - t0:.0f} s)")
    for lab, w in writes.items():
        if w:
            print(f"  [{lab}] host-only writes seen (the PAYLOAD law's, P7c.1's): "
                  f"{sorted(w)}")
    return acts, n


def part_scene():
    census = Census().install()
    try:
        acts, _n = run_shadow("scene", [GrenadeScene(0)], SCENE_TICKS,
                              census=census.c, twins=[GrenadeScene(0)])
        acts2, _ = run_shadow("scene v1", [GrenadeScene(1)], SCENE_TICKS,
                              census=census.c)
    finally:
        census.uninstall()
    c = census.c
    kinds = [k for k in c if k.startswith("kind thrown: ")]
    from simulation.weapons import get_tables
    for name in get_tables().hand_grenade_rows:
        if not c[f"kind thrown: {name}"]:
            fail(f"scene: the grenade kind {name!r} was never thrown")
    for b in NEED:
        if not c[b]:
            fail(f"scene: the census branch {b!r} was never taken")
    for b in UNREACHABLE:
        if c[b]:
            print(f"  note: {b!r} named unreachable was taken {c[b]} times")
    for lab in (s.label for s in SLOTS):
        if not acts[lab]:
            fail(f"scene: slot {lab!r} never acted (vacuous)")
    print(f"scene census ({len(kinds)} kinds thrown): " + ", ".join(
        f"{k} {v}" for k, v in sorted(c.items()) if not k.startswith("kind thrown")))
    print("scene unreachable: " + "; ".join(f"{k} -- {v}" for k, v in UNREACHABLE.items()))
    _ = acts2


# ---------------------------------------------------------------------------
# PART BOUNCE (#215)
# ---------------------------------------------------------------------------
#: The bounce law's branches the bouncing scenes must take (with PART FLIGHT
#: holding the exact corner on constructed rows: a throw's facing is a
#: rounded angle, so a scene corner is a near-corner).
NEED_BOUNCE = (
    "bounce: a vertical edge (x flipped)", "bounce: a horizontal edge (y flipped)",
    "bounce: a mirrored leg", "bounce: bounce_max reached (rests at the wall)",
    "bounce: restitution 0 (drops at the wall)", "bounce: restitution 1 (a mirror)",
    "settle: a fuse-out mid-leg (the rest flown at once)",
    "settle: a wall met while settling",
)
BOUNCE_TICKS = int(os.environ.get("UNIT_OBJECTS_BOUNCE_TICKS", "110"))


def _bounce_facts(sc, name):
    """The scene's own law facts, on the host's grenades: the one at the
    sealed gap bounced, the one at the destroyed tile went through, the
    mid-leg fuse-out detonated on the thrower's side, none rests on a
    blocked tile, each one detonated."""
    from simulation import combat
    gap, hole, mid = (sc.by_owner(sc.e[i]) for i in (1, 2, 3))
    allp = list(sc.thrown.values())
    if not allp or not all(p.detonated for p in allp):
        fail(f"{name}: not every grenade detonated within the scene")
        return
    if not gap or gap[0].bounces < 1 and sc.v != 1:
        fail(f"{name}: the grenade at the gap closed mid-flight did not bounce")
    if gap and combat.detonation_tile(sc.sim.gmap, gap[0].pos_x_q, gap[0].pos_y_q)[1] >= 32:
        fail(f"{name}: the grenade at the closed gap detonated beyond it")
    if not hole or hole[0].bounces or \
            combat.detonation_tile(sc.sim.gmap, hole[0].pos_x_q, hole[0].pos_y_q)[1] <= 32:
        fail(f"{name}: the grenade at the destroyed tile did not go through")
    if not mid or unit_fixed_tile(mid[0].pos_x_q) >= 32:
        fail(f"{name}: the mid-leg fuse-out detonated beyond the wall")


def unit_fixed_tile(q):
    return int(q) >> 16


def part_bounce():
    census = Census().install()
    try:
        scenes = {}
        for v in (0, 1, 2):
            sc = BounceScene(v)
            run_shadow(f"bounce v{v}", [sc], BOUNCE_TICKS, census=census.c,
                       twins=[BounceScene(v)] if v == 0 else None)
            _bounce_facts(sc, f"bounce v{v}")
            scenes[v] = sc
        drivers = [BounceScene(0), BounceScene(1), BounceScene(2), GrenadeScene(1)]
        acts, _ = run_shadow("bounce batch N=4", drivers, BOUNCE_TICKS)
        if not acts[SLOT_2.label]:
            fail("bounce batch: slot 2 never acted (vacuous)")
    finally:
        census.uninstall()
    for b in NEED_BOUNCE:
        if not census.c[b]:
            fail(f"bounce: the census branch {b!r} was never taken")
    print("bounce census: " + ", ".join(f"{k} {v}" for k, v in sorted(census.c.items())
                                        if k.startswith(("bounce", "settle"))))
    part_chained(scene=BounceScene, sets=("grenades",))


# ---------------------------------------------------------------------------
# PART BATCH
# ---------------------------------------------------------------------------
def _batch_drivers(n):
    from cuda_object_table_check import ChargeScene
    from cuda_unit_verbs_check import VerbDriver
    if n == 2:
        return [GrenadeScene(1), ShipDriver()]
    pool = [GrenadeScene(0), GrenadeScene(1), GrenadeScene(2), GrenadeScene(3),
            VerbDriver(0), VerbDriver(3), ChargeScene(0), ChargeScene(1)]
    return pool[:n]


def part_batch():
    for n in BATCH_NS:
        drivers = _batch_drivers(n)
        acts, _ = run_shadow(f"batch N={n}", drivers, BATCH_TICKS)
        for lab in (s.label for s in SLOTS):
            if not acts[lab]:
                fail(f"batch N={n}: slot {lab!r} never acted (vacuous)")


# ---------------------------------------------------------------------------
# PART PADDING
# ---------------------------------------------------------------------------
def part_padding():
    from simulation import combat
    from simulation.charges import PlantedCharge
    from simulation.orders import ORDER_GRENADE
    sim = GrenadeScene(0).sim
    st = _store([sim, sim])
    tick = 100
    live = [combat.Projectile(ORDER_GRENADE, 4, 4, 12.5, 4.5, 0.0, 90 - i,
                              ammo_name="grenade_frag", owner_id=i, fuse_ticks=6)
            for i in range(5)]
    chg = [PlantedCharge(10 + i, 12, "demo_c4", i, det_tick=tick - (i % 2),
                         approach=(11, 10 + i)) for i in range(4)]
    junk_g = _rows_of(GT, [combat.Projectile(ORDER_GRENADE, 1, 1, 30.5, 1.5, 0.0, 0,
                                             ammo_name="grenade_tear", owner_id=77,
                                             fuse_ticks=1)])[0]
    junk_c = _rows_of(CT, [PlantedCharge(5, 5, "demo_c4", 77, det_tick=0)])[0]

    def run(dirty):
        g = ot.DeviceTable(GT, 2)
        c = ot.DeviceTable(CT, 2)
        rg = np.zeros(g.shape, dtype=np.int64)
        rc = np.zeros(c.shape, dtype=np.int64)
        ng, nc = (5, 3), (4, 2)
        for e in range(2):
            rg[e, :ng[e]] = _rows_of(GT, live)[:ng[e]]
            rc[e, :nc[e]] = _rows_of(CT, chg)[:nc[e]]
            if dirty:
                rg[e, ng[e]:] = junk_g
                rc[e, nc[e]:] = junk_c
        g.upload(rg, np.array(ng))
        c.upload(rc, np.array(nc))
        fx = applier.DeviceApplier(2)
        fx.begin()
        unit_kernels.grenades(st, g, fx, np.array([tick, tick]), _solid([sim, sim]))
        unit_kernels.charges_tick(st, c, fx, np.array([tick, tick]))
        return g.download(), c.download(), fx.records()
    clean, dirty = run(False), run(True)
    for e, n_g, n_c in ((0, 5, 4), (1, 3, 2)):
        if not np.array_equal(clean[0][0][e, :n_g], dirty[0][0][e, :n_g]) or \
                clean[2][e] != dirty[2][e]:
            fail(f"padding env {e}: garbage past the grenade count changed the valid "
                 f"rows or the records")
        if not np.array_equal(dirty[0][0][e, n_g:], np.broadcast_to(
                junk_g, dirty[0][0][e, n_g:].shape)):
            fail(f"padding env {e}: a grenade padding row was written")
        if not np.array_equal(dirty[1][0][e, n_c:], np.broadcast_to(
                junk_c, dirty[1][0][e, n_c:].shape)):
            fail(f"padding env {e}: a charge padding row was written")
    if not clean[2][0] or len(clean[2][0]) == len(clean[2][1]):
        fail("padding: the two envs' counts made no difference (vacuous)")
    print(f"padding: garbage rows (live, due) past each table's count neither read "
          f"nor written; records {len(clean[2][0])} / {len(clean[2][1])} per env as "
          f"the clean run")


# ---------------------------------------------------------------------------
# PART OVERFLOW
# ---------------------------------------------------------------------------
def part_overflow():
    seen = Counter()
    sim0 = []

    def run_1b(st, inp, req, lists):
        cov = _run_1b(st, inp, req, lists)
        recs = req.records()[0]
        thr = [r for r in recs if r[0] == unit_requests.KIND_THROW]
        acc = [r for r in recs if r[0] == unit_requests.KIND_PLANT and r[12]]
        for name, tab_s, want, n in (("grenade", GT, thr, len(inp.lists["grenades"][0])),
                                     ("charge", CT, acc, len(inp.lists["charges"][0]))):
            if len(want) < 2 or seen[name]:
                continue
            small = ot.DeviceTable(tab_s, 1, cap=n + 1)
            small.stage_lists(inp.lists[name + "s"])
            other = ot.DeviceTable(CT if name == "grenade" else GT, 1)
            other.stage_lists(inp.lists["charges" if name == "grenade" else "grenades"])
            args = (small, other) if name == "grenade" else (other, small)
            try:
                unit_kernels.table_appends(st, req, *args)
                fail(f"overflow: a full {name} table took {len(want)} appends")
            except ot.ObjectTableOverflow as ex:
                if f"'{name}'" not in str(ex) or "env 0" not in str(ex) \
                        or f"{len(want) - 1} row" not in str(ex):
                    fail(f"overflow: the {name} refusal does not name the table, the "
                         f"env and the count: {ex}")
            rows, cnt, over = small.download()
            if int(cnt[0]) != n + 1 or int(over[0]) != len(want) - 1:
                fail(f"overflow: the {name} table holds {int(cnt[0])} rows, overflow "
                     f"{int(over[0])}")
            first = ot.table_layout(tab_s)
            _ = first
            seen[name] += 1
        return cov

    def run_2(st, inp, lists):
        if not seen["effects"]:
            fx = applier.DeviceApplier(1, cap=1)
            g = ot.DeviceTable(GT, 1)
            g.stage_lists(inp.lists["grenades"])
            fx.begin()
            try:
                unit_kernels.grenades(st, g, fx, inp.tick, _flight_fields(inp))
                n = 0
            except applier.EffectOverflow as ex:
                n = 1
                if "PAYLOAD" not in str(ex) or "env(s) [0]" not in str(ex):
                    fail(f"overflow: the effect-list refusal does not name it: {ex}")
            if n:
                full = applier.DeviceApplier(1)
                g.stage_lists(inp.lists["grenades"])
                full.begin()
                unit_kernels.grenades(st, g, full, inp.tick, _flight_fields(inp))
                if fx.records()[0] != full.records()[0][:1]:
                    fail("overflow: the record kept is not the first")
                seen["effects"] += 1
        return _run_2(st, inp, lists)

    slots = [DeviceSlot(label=SLOT_1B.label, run=run_1b, inputs=SLOT_1B.inputs,
                        requests=True, outputs=SLOT_1B.outputs),
             DeviceSlot(label=SLOT_2.label, run=run_2, columns=("id",),
                        inputs=SLOT_2.inputs, outputs=SLOT_2.outputs,
                        host_only=HOST_ONLY)]
    run_shadow("overflow", [GrenadeScene(1)], 60, slots=slots)
    for k in ("grenade", "charge", "effects"):
        if not seen[k]:
            fail(f"overflow: the {k} overflow case was never reached (vacuous)")
    _ = sim0
    print(f"overflow: a grenade table, a charge table and an effect list one short "
          f"each refused by name, the first rows / record written, the rest counted "
          f"{dict(seen)}")


# ---------------------------------------------------------------------------
# PART REFUSE
# ---------------------------------------------------------------------------
def part_refuse():
    from simulation import combat
    from simulation.orders import ORDER_GRENADE
    from simulation.ruleset import OnePhaseWEGO
    sc = GrenadeScene(0)
    try:
        unit_kernels.refuse_wego([sc.sim])
    except unit_kernels.ObjectSlotRefused as ex:
        fail(f"refuse: a continuous sim was refused: {ex}")
    rules = sc.sim.ruleset
    sc.sim.ruleset = OnePhaseWEGO()
    try:
        unit_kernels.refuse_wego([sc.sim])
        fail("refuse: a ruleset that drives the units passed the device door")
    except unit_kernels.ObjectSlotRefused as ex:
        if "detonate_owned" not in str(ex):
            fail(f"refuse: the refusal does not name detonate_owned: {ex}")
    sc.sim.ruleset = rules
    sc.sim.projectiles.append(combat.Projectile(ORDER_GRENADE, 3, 3, 9.5, 3.5, 2.0, 0))
    try:
        unit_kernels.refuse_wego([sc.sim])
        fail("refuse: a WEGO grenade passed the device door")
    except unit_kernels.ObjectSlotRefused as ex:
        if "WEGO" not in str(ex):
            fail(f"refuse: the WEGO grenade refusal does not name it: {ex}")
    print("refuse: a ruleset driving the units (detonate_owned) and a WEGO grenade "
          "refused by name at the device door; a continuous sim passes")


# ---------------------------------------------------------------------------
# PART CHAINED
# ---------------------------------------------------------------------------
_CH_REQ = {}


def _run_1b_chained(st, inp, lists):
    """Slot 1b + its appends in a chain: the request list is the chain's own
    (the throws and plants land as TABLE rows, never through apply_to_host);
    a door flip has no landing here (the scene has no door) and is refused."""
    key = st.shape
    if key not in _CH_REQ:
        _CH_REQ[key] = unit_requests.RequestList(st.shape[0], st.shape[1])
    req = _CH_REQ[key]
    req.begin()
    cov = unit_kernels.verbs(st, req, inp.tick, inp.tables["door_spans"],
                             inp.scalars["charges_live"])
    if any(r[0] == unit_requests.KIND_USE for r in req.records()[0]):
        raise RuntimeError("chained: a door flip in the grenade chain (no landing)")
    unit_kernels.table_appends(st, req, lists["grenades"], lists["charges"])
    st.chain_refused = np.asarray(st.last_refused).copy()
    return cov


SLOT_1B_CH = DeviceSlot(label="1b intents", run=_run_1b_chained,
                        inputs=SLOT_1B.inputs, outputs=SLOT_1B.outputs)


def part_chained(scene=None, sets=("grenades", "charges")):
    scene = scene or GrenadeScene
    from cuda_unit_verbs_check import (SLOT_1BP, SLOT_1C, _books_host_side,
                                       _orders_host_side, _same)

    def host_side_a(sim, pre, post):
        _books_host_side(sim, pre, post)
        _orders_host_side(sim, pre, post)
        st = ch_a.store
        sim.charges_refused = getattr(sim, "charges_refused", 0) + int(
            np.asarray(getattr(st, "chain_refused", [0]))[0])

    for name, slots, v in (("1b + 1b' + 1c + 2 grenades", [SLOT_1B_CH, SLOT_1BP,
                                                            SLOT_1C, SLOT_2], 0),
                           ("9e(d') planted charges", [SLOT_9E], 1)):
        if ("charges" if "9e" in name else "grenades") not in sets:
            continue
        name = f"{name} ({scene.__name__})"
        twin, ref = scene(v), scene(v)
        t0 = time.perf_counter()
        n = 0
        ch_a = ChainedSlots(twin, slots, host_side=host_side_a if len(slots) > 1 else None)
        payloads = 0
        with ch_a as ch:
            for _ in range(SCENE_TICKS):
                twin.tick()
                ref.tick()
                n += 1
                if not _same(twin.sim, ref.sim):
                    fail(f"chained {name}: the twin left the reference at tick "
                         f"{twin.sim.tick}")
                    break
                bad = [a for a, t in (("projectiles", GT), ("planted_charges", CT))
                       if ot.records(t, getattr(twin.sim, a))
                       != ot.records(t, getattr(ref.sim, a))]
                if bad:
                    fail(f"chained {name}: {bad} left the reference at tick "
                         f"{twin.sim.tick}")
                    break
                if getattr(twin.sim, "charges_refused", 0) != getattr(ref.sim,
                                                                      "charges_refused", 0):
                    fail(f"chained {name}: charges_refused left the reference")
                    break
            payloads = sum(len(r.get("payloads", ())) for r in ch.lists)
        print(f"chained {name}: {n} ticks, the twin's unit hash, field digest, "
              f"grenade and charge lists equal the reference every tick ({ch.runs} "
              f"chain runs, {payloads} PAYLOAD records replayed on the host; "
              f"{time.perf_counter() - t0:.0f} s)")
        # 9e(d') is gated by the conductor (a charge must lie): it runs on
        # the ticks the reference runs it, never more
        gated = all(s.skip_ok for s in slots)
        if (ch.runs > n if gated else ch.runs != n) or not ch.runs or not payloads:
            fail(f"chained {name}: {ch.runs} runs in {n} ticks, {payloads} payloads "
                 f"(vacuous)")


# ---------------------------------------------------------------------------
# PART SHIP
# ---------------------------------------------------------------------------
def part_ship():
    acts, n = run_shadow("ship", [ShipDriver()], SHIP_TICKS, twins=[ShipDriver()])
    print(f"ship: the slots acted on {dict(acts)} of {n} ticks")


# ---------------------------------------------------------------------------
PARTS = ("tables", "flight", "scene", "bounce", "batch", "padding", "overflow", "refuse",
         "chained", "ship")


def main(parts=None) -> int:
    if not getattr(bp, "HAS_CUDA", False):
        print("UNIT_OBJECTS_RESULT: FAIL (not the CUDA build)")
        return 1
    parts = parts or os.environ.get("UNIT_OBJECTS_PARTS", ",".join(PARTS)).split(",")
    t0 = time.perf_counter()
    for p in parts:
        try:
            {"tables": part_tables, "flight": part_flight, "scene": part_scene,
             "bounce": part_bounce,
             "batch": part_batch, "padding": part_padding, "overflow": part_overflow,
             "refuse": part_refuse, "chained": part_chained, "ship": part_ship}[p]()
        except Exception as ex:          # a part's crash fails it, the rest run
            import traceback
            traceback.print_exc()
            fail(f"part {p} raised {type(ex).__name__}: {ex}")
        sys.stdout.flush()
    print(f"total {time.perf_counter() - t0:.0f} s")
    ok = not FAILS
    print("UNIT_OBJECTS_RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def bench(reps: int = 200) -> int:
    """The TIMING table: slot 2's grenades (a table of 16 grenades in flight,
    4 fusing out) and charges.tick (16 charges, 4 due, then the prune), each
    call's H2D of the tick + kernels + the counts' D2H; medians over ``reps``
    at N = 1, 8, 64 (each rep re-stages the start state, not timed)."""
    import copy
    from simulation import combat
    from simulation.charges import PlantedCharge
    from simulation.orders import ORDER_GRENADE
    sim = GrenadeScene(0).sim
    tick = 1000
    gs = [combat.Projectile(ORDER_GRENADE, 4 + i, 4, 20.5, 9.5 + i, 0.0, tick - 3 - i,
                            ammo_name="grenade_frag", owner_id=i,
                            fuse_ticks=6 if i < 4 else 90) for i in range(16)]
    cs = [PlantedCharge(10 + i, 12, "demo_c4", i, det_tick=tick if i < 4 else tick + 50,
                        approach=(11, 10 + i)) for i in range(16)]
    print("TIMING (median ms over %d reps)" % reps)
    print("| slot | N=1 | N=8 | N=64 |")
    res = {}
    for n in (1, 8, 64):
        st = _store([sim] * n)
        g = ot.DeviceTable(GT, n)
        c = ot.DeviceTable(CT, n)
        fx = applier.DeviceApplier(n)
        ticks = np.full(n, tick, dtype=np.int64)
        planes = _solid([sim] * n)
        tg, tc = [], []
        for _ in range(reps):
            g.stage_lists([_rows_of(GT, copy.deepcopy(gs))] * n)
            c.stage_lists([_rows_of(CT, cs)] * n)
            fx.begin()
            s = time.perf_counter()
            unit_kernels.grenades(st, g, fx, ticks, planes)
            m = time.perf_counter()
            unit_kernels.charges_tick(st, c, fx, ticks)
            e = time.perf_counter()
            tg.append(m - s)
            tc.append(e - m)
        res[n] = (np.median(tg), np.median(tc))
    for i, name in enumerate(("2 grenades (16 in flight, 4 fuse out)",
                              "9e(d') charges.tick (16, 4 due, + prune)")):
        print(f"| {name} | " + " | ".join(f"{res[n][i] * 1e3:.3f}" for n in (1, 8, 64)) + " |")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        sys.exit(bench())
    sys.exit(main())
