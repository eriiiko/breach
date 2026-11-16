"""#29 P7a.1 -- THE EFFECT-LIST GATE: the per-env serial applier's widened
record (HIT, STATUS, PUSH, CHEW and the PAYLOAD record beside DESTROY_WALL),
its RESOLVE stage (slot 4c, ``SimultaneousEffects.resolve`` of the attack
slot, with the exposure landing as a mode of the device ``apply_packet``),
and slot 4a (the melee verb, ``combat.process_melee_intents``) as the list's
first device producer -- each against its Python law at tol 0 on every row,
every event, every plane and every book it writes.

Runs inside the GPU subprocess (tests/cuda_harness.run_cuda_script; wrapper
tests/test_cuda_unit_melee.py). The list is ``simulation.applier``'s
``DeviceApplier`` (C++ ``cpp/src/cuda_applier.cu``: a ``UnitRecordList`` the
producers append to, walked by the one applier kernel in a STAGE); slot 4a is
``cpp/src/cuda_unit_melee.cu`` over the unit library's ``can_melee`` /
``melee_arc_target`` / ``impulse_add`` (``cpp/src/cuda_unit_lib.cuh``); the
Python faces ``simulation.unit_kernels.melee`` / ``resolve`` /
``walk_effects``; the harness ``tests/cuda_unit_lockstep.py`` with the LISTS
of #29 P7a.1 (``inputs=("list:effects",)``, ``outputs=("list:effects",)``:
the host's ``SimultaneousEffects`` records turned into device records through
ONE record function, ``simulation.applier.records_of``).

PART TABLES -- every P7a.1 table row (the melee rows per weapons-table row)
equals an independent read of the sim's own weapon rows and impulse table.

PART LIB -- the device ``can_melee`` (both moments), ``melee_arc_target`` and
``melee_adjacent`` on SYNTHETIC rows (life, zombie, team, melee row, stun,
wind-up and plant ticks around the tick, the last blow around the cadence,
positions packed so adjacency, arcs, ties and coincident centres all occur,
facings over the circle, a STRIKE / PUSH hook), two envs, against the Python
on the decoded rows: both answers of every gate seen; padding rows silent.

PART SCENE -- the melee scene built for the gate (the shipped bots strike
rarely): marines with melee rows against zombies and enemy marines, two
units that kill each other on one tick, two blows on one target in one
resolve (the first packet to zero is the kill; a later STATUS and PUSH aimed
at the corpse skipped), a later blow on a slot-killed unit reading its
UNZEROED exposure (the landing changing a number), a tie to the lower id,
coincident centres, an adjacent teammate / dead enemy / enemy outside the arc
skipped, the baton (no push), every latch refusal (dead, zombie, no melee
row, stunned, winding up, planting, the cadence), STRIKE / PUSH hooks. Slots
4a and 4c SHADOWED together: 4a's rows and its effect records equal the
host's record for record; 4c fed the host's list: rows, ring and planes
equal. A census of the Python law's branches in 4a and in the resolve, each
required to be taken, the unreachable ones named with the reason.

PART RESOLVE -- SYNTHETIC lists through the RESOLVE stage on a real map:
HITs with ``expose`` (the raises held for the landing; a raise skipped for a
unit dead at the landing; the dead zeroed; the AP and HEAL doors), STATUS
and PUSH on the living and on a corpse, a PUSH with a lever and a twist, a
mutual kill, CHEW on real walls (a wall chewed to zero and destroyed, air
and dead-damage no-ops); the landing's result REQUIRED to differ from the
same list applied at once; a mixed list of HIT and CHEW whose forward and
reversed orders are required to differ (the kill's credit, a breach chain),
device equal to host in both; the books close as in P6.4b's gate. PAYLOAD:
recorded and compared, then REFUSED by name at the apply; FACE and an
unknown world call refused by name at the record function.

PART BATCH -- N = 2 and 8 melee scenes that differ in layout, seed, unit
count and fields (an extra interior wall), beside ship_demo: env i == env i
alone (rows, lists, ring, planes, books).

PART PADDING -- padding rows filled with live-looking rows (a latch, a target
in reach; valid cleared) stay untouched, never strike, never take an effect;
a HIT aimed at a padding row faults and is refused.

PART OVERFLOW -- the list's capacity and a row's staging, each past its
bound: counted and refused by name (the first records kept); a staged host
list past the capacity applies its first entries and is refused.

PART CHAINED -- 4a + 4c replaced by the device's in a running melee match:
the per-tick unit hash and field digest equal an untouched reference every
tick. PART SHIP -- ship_demo's bots match shadowed (4a + 4c) for SHIP_TICKS
ticks; how often 4a acts is printed.

PROPERTY: the effect list, the resolve stage and slot 4a are the Python's
integer for integer and record for record, in the reference's order, per env.
BREAKS IF the landing lands at once (or zeroes before the raises, or raises a
corpse), the list's order, the gather's row order or its continuation, a
kind's function (a HIT's packet field, a STATUS on a corpse, a PUSH's
angular part, a CHEW's quantization or its zero test), the arc target's tie,
the adjacency, a gate term, a melee table row or the batch rebase differs
from the reference. Each validated by breaking it (the patch's report).

Prints per-part lines and ``UNIT_MELEE_RESULT: PASS`` / ``FAIL``; exits 0 /
1. ``bench()`` prints the TIMING table.
"""
from __future__ import annotations

import os
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import breach_physics as bp   # the CUDA build (cuda_harness bootstrap): FIRST

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "tools") not in sys.path:
    sys.path.append(str(_ROOT / "tools"))   # appended: never ahead of the CUDA build

import level_loader  # noqa: E402
from cuda_unit_lockstep import (ChainedSlots, DeviceSlot,  # noqa: E402
                                ShadowLockstep, first_field_diff)
from simulation import (applier, unit_events, unit_kernels,  # noqa: E402
                        unit_store, unit_tables, unit_traits)

FAILS: list = []
ONE = unit_traits.ONE
PI_Q16 = 205887
MELEE_TICKS = int(os.environ.get("UNIT_MELEE_TICKS", "160"))
SHIP_TICKS = int(os.environ.get("UNIT_MELEE_SHIP_TICKS", "500"))
BATCH_TICKS = int(os.environ.get("UNIT_MELEE_BATCH_TICKS", "80"))
BATCH_NS = [int(x) for x in os.environ.get("UNIT_MELEE_BATCH_NS", "2,8").split(",")]
CHAIN_TICKS = int(os.environ.get("UNIT_MELEE_CHAIN_TICKS", "160"))
LIB_TRIALS = int(os.environ.get("UNIT_MELEE_LIB_TRIALS", "8"))


def fail(msg):
    FAILS.append(msg)
    print("FAIL:", msg)


# ---------------------------------------------------------------------------
# The slots
# ---------------------------------------------------------------------------
PLANE_OUTPUTS = tuple("field:" + n for n in applier.PLANE_NAMES)
BOOK_OUTPUTS = tuple("book:" + b for b in (
    "n_destruction_seed", "ge_retire", "ge_mint", "ge_destroy_seed",
    "solid_destroy_wall", "tiles_changed"))
HOST_ONLY = (
    ("render_dirty", "the wall seam's render mark (a chew's destroy; no device form)"),
    ("_sky_mask_dirty", "the sky exchange's lazy-rebuild flag (ambient maps)"),
    ("_gas_t_amb_raw_cache", "a lazily filled cache of a config constant"),
)


def _run_4a(st, inp, lists):
    return unit_kernels.melee(st, lists["effects"], inp.tick)


def _run_4c(st, inp, ring, lists):
    return unit_kernels.resolve(st, lists["effects"], inp.fields, ring)


def _conductor(sim, args, kw, caller):
    """Slot 4c is the conductor's resolve (``Simulation._step_pre_physics``),
    never the rounds' landing (slot 2', ``_update_rounds``: P7a.2's stage)."""
    return caller == "_step_pre_physics"


SLOT_4A = DeviceSlot(label="4a melee", run=_run_4a, inputs=("tick", "list:effects"),
                     outputs=("list:effects",))
SLOT_4C = DeviceSlot(label="4c resolve", run=_run_4c, inputs=("list:effects",),
                     events=True, outputs=PLANE_OUTPUTS + BOOK_OUTPUTS,
                     host_only=HOST_ONLY, select=_conductor)


# ---------------------------------------------------------------------------
# Twin compare
# ---------------------------------------------------------------------------
def _unit_hash(sim):
    from field_ab_harness import _capture_unit_state
    return _capture_unit_state(sim)["hash"]


def _field_digest(sim):
    from field_ab_harness import SIM_FIELDS, _snapshot
    from field_digest import field_digest
    return field_digest(_snapshot(sim.gmap, SIM_FIELDS))


def _same(a, b):
    return _unit_hash(a) == _unit_hash(b) and _field_digest(a) == _field_digest(b)


# ---------------------------------------------------------------------------
# The melee scene
# ---------------------------------------------------------------------------
SPACE = level_loader.SPACE_CODE
EAST, WEST, SOUTH, NORTH = 0, PI_Q16, -PI_Q16 // 2, PI_Q16 // 2
MH, MW = 30, 48


def _melee_level(variant):
    """A hull box in space, interior air, a wood wall (its row moves with the
    variant) and, for odd variants, an extra furniture block: the batch's
    envs differ in their fields."""
    from level_loader import LevelData
    from simulation.materials import MAT_AIR, MAT_FURNITURE, MAT_HULL, MAT_WOOD
    tm = np.full((MH, MW), SPACE, dtype=np.int32)
    tm[1:MH - 1, 1:MW - 1] = MAT_HULL
    tm[2:MH - 2, 2:MW - 2] = MAT_AIR
    r0 = 24 + (variant % 3)
    tm[r0, 4:20] = MAT_WOOD
    if variant % 2:
        tm[20:23, 40:44] = MAT_FURNITURE
    return LevelData(name=f"p7a1_melee{variant}", version="3", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."),
                     boundary="space")


STRIKE_PUSH = {"STRIKE": ONE * 3 // 2, "PUSH": ONE * 2}
STRIKE_PUSH_2 = {"STRIKE": ONE * 3 // 4, "PUSH": ONE // 2}


class MeleeDriver:
    """The melee scene (see the module docstring), scripted: every striker
    latches the verb every tick through the intent facade; fixtures written
    between ticks (a stun, a wind-up, a plant, the zombies held stunned so the
    horde does not wander, a target's exposure). ``variant`` > 0: another
    seed and layout, bystanders, every group shifted."""

    def __init__(self, variant=0, hooks=True):
        from simulation import Simulation
        from simulation.ruleset import ContinuousRealtime
        self.variant, self.k, self.hooks = variant, 0, hooks
        sim = Simulation(_melee_level(variant), seed=20261115 + 7 * variant,
                         breach_physics=bp, enable_recorder=False,
                         ruleset=ContinuousRealtime())
        sim.set_paused(False)
        self.sim = sim
        self.who = {}
        self.strikers = []
        # bystanders: AFTER the groups, or BEFORE them for odd variants, so the
        # batch's envs record their blows on other rows (a list of env 0 is
        # not env 1's)
        if variant % 2:
            self._bystanders()
        self._setup()
        if not variant % 2:
            self._bystanders()

    def _bystanders(self):
        for i in range(self.variant % 4):
            self._add(f"by{i}", 30 + 4 * i, 22, facing=NORTH, latch=False)

    def _add(self, name, x, y, team=0, facing=EAST, hooks=None, hp=10000.0,
             melee="rifle_butt", latch=True):
        from simulation.turning import place
        from simulation.unit import Unit
        dy = self.variant % 2
        u = Unit(name, x=float(x), y=float(y + dy), team=team)
        self.sim.add_unit(u, hooks=hooks if self.hooks else None)
        place(u, facing)
        u.current_hp = float(hp)
        u.melee_id = melee
        self.who[name] = u
        if latch:
            self.strikers.append(u)
        return u

    def _kill(self, u, zombie=False):
        from simulation.ai_zombie import convert_marines_to_zombies
        from simulation.damage import DamagePacket, apply_packet
        apply_packet(u, DamagePacket(amount=1.0e6, dtype=0, attacker_id=-1), [],
                     source="melee", mark_killed_by_zombie=zombie)
        if zombie:
            convert_marines_to_zombies([u])

    def _setup(self):
        a = self._add
        # G1: a hooked marine against a zombie (two blows kill it)
        a("A", 3, 3, facing=EAST,
          hooks=(STRIKE_PUSH, STRIKE_PUSH_2, None)[self.variant % 3])
        z1 = a("Z1", 6, 3, facing=WEST, latch=False)
        self._kill(z1, zombie=True)
        z1.current_hp = 40.0
        # G2: two marines of two teams that kill each other on one tick
        a("M", 3, 8, team=0, facing=EAST, hp=1.0)
        a("E", 6, 8, team=1, facing=WEST, hp=1.0)
        # G3: two strikers on one target the first blow kills (its exposure set)
        a("S1", 3, 13, facing=EAST)
        a("T", 6, 13, team=1, facing=NORTH, hp=5.0, latch=False)
        a("S2", 9, 13, facing=WEST, hooks=STRIKE_PUSH_2)
        # G3b: two strikers on one target the SECOND blow kills
        a("S3", 3, 18, facing=EAST)
        a("T2", 6, 18, team=1, facing=NORTH, hp=30.0, latch=False)
        a("S4", 9, 18, facing=WEST)
        # G4: a tie (two enemies at +-45 degrees), the lower id struck
        a("K", 18, 6, facing=EAST)
        a("N1", 21, 3, team=1, facing=WEST, latch=False)
        a("N2", 21, 9, team=1, facing=WEST, latch=False)
        # G5: an adjacent enemy behind (outside the arc), a teammate ahead
        a("K2", 18, 14, facing=EAST)
        a("Bk", 15, 14, team=1, facing=EAST, latch=False)
        a("Tm", 21, 14, team=0, facing=WEST, latch=False)
        # G6: coincident centres (dead ahead; the push along the facing)
        a("C", 30, 4, facing=SOUTH)
        a("X", 30, 4, team=1, facing=NORTH, latch=False)
        # G7: the baton (a stun, no push) on a zombie
        a("B", 30, 10, facing=EAST, melee="arc_baton")
        z2 = a("Z2", 33, 10, facing=WEST, latch=False)
        self._kill(z2, zombie=True)
        # G9: the latch refusals
        d = a("D", 38, 4, facing=EAST)
        self._kill(d)
        zl = a("Zl", 38, 8, facing=EAST)
        self._kill(zl, zombie=True)
        a("Nm", 38, 12, facing=EAST, melee="")
        a("St", 38, 16, facing=EAST)
        a("Wu", 42, 4, facing=EAST)
        a("Pl", 42, 8, facing=EAST)
        a("Tgt", 45, 4, team=1, facing=WEST, latch=False)

    def _before(self):
        from simulation.status import STUNNED, apply_status
        sim, w, k = self.sim, self.who, self.k
        for n in ("Z1", "Z2", "Zl"):                       # the horde held still
            if w[n].alive:
                apply_status(w[n], STUNNED, 0, 3)
        apply_status(w["St"], STUNNED, 0, 3)
        w["Wu"].throw_release_tick = sim.tick + 5
        w["Pl"].plant_done_tick = sim.tick + 5
        if w["T"].alive:                                   # a level the corpse keeps
            w["T"].exposure[0] = ONE // 2
        for u in self.strikers:
            sim.melee_intent(u.id)

    def tick(self):
        self._before()
        self.sim.step()
        self.sim.set_paused(False)
        self.k += 1

    def is_terminal(self):
        return False


class ShipDriver:
    """ship_demo's bots match (MatchSession), STRIKE / PUSH hooks on two
    thirds of the marines."""

    def __init__(self, seed=1, hooks=True):
        from simulation.match_session import MatchSession
        level = level_loader.load("ship_demo", str(_ROOT / "levels"))
        self.session = MatchSession(level, seed=seed, breach_physics=bp)
        self.sim = self.session.sim
        if hooks:
            for i, u in enumerate(u for u in self.sim.units if not u.is_zombie):
                h = (None, STRIKE_PUSH, STRIKE_PUSH_2)[i % 3]
                if h is not None:
                    unit_traits.compile_unit(self.sim, u, h)

    def tick(self):
        self.session.tick()

    def is_terminal(self):
        return self.session.is_terminal()


# ---------------------------------------------------------------------------
# The census: the host's own decisions in 4a and in the resolve, by branch
# ---------------------------------------------------------------------------
class Census:
    """Wraps slot 4a (``simulation.simulation.process_melee_intents``) and
    the resolve's internals (``simultaneous.apply_packet`` / ``_status_if_alive``
    / ``_impulse_add`` / ``ExposureLanding``) -- reads only -- and counts every
    branch the reference took, classified by the reference's own terms."""

    def __init__(self):
        self.c = Counter()
        self._saved = []
        self._kills = []          # this resolve's (attacker, victim)

    def __enter__(self):
        from simulation import simulation as simmod, simultaneous
        census = self
        o4a = simmod.process_melee_intents

        def w4a(units, tick, tables, impulse_table, events=None, effects=None,
                __o=o4a):
            census._classify_4a(units, tick, tables, impulse_table)
            n0 = len(effects._pending) if effects is not None else 0
            out = __o(units, tick, tables, impulse_table, events=events, effects=effects)
            if effects is not None:
                tg = Counter(id(e[1]) for e in effects._pending[n0:] if e[0] == 0)
                if any(v >= 2 for v in tg.values()):
                    census.c["4a: two blows on one target in one tick"] += 1
            return out
        simmod.process_melee_intents = w4a
        self._saved.append((simmod, "process_melee_intents", o4a))

        oap = simultaneous.apply_packet

        def wap(unit, packet, events=None, __o=oap, **kw):
            was = unit.alive
            nonzero_x = any(getattr(unit, "exposure", ()) or ())
            out = __o(unit, packet, events, **kw)
            if not was:
                census.c["resolve: a HIT on a corpse (no second kill)"] += 1
                if nonzero_x and kw.get("exposure_landing") is not None:
                    census.c["resolve: a HIT on a slot-killed unit reads its unzeroed "
                             "exposure"] += 1
            elif not unit.alive:
                census.c["resolve: a HIT kills"] += 1
                census._kills.append((int(packet.attacker_id), int(unit.id)))
            else:
                census.c["resolve: a HIT on the living"] += 1
            if getattr(packet, "expose", 0):
                census.c["resolve: a HIT with expose"] += 1
            return out
        simultaneous.apply_packet = wap
        self._saved.append((simultaneous, "apply_packet", oap))

        osa = simultaneous._status_if_alive

        def wsa(unit, *a, __o=osa):
            census.c["resolve: a STATUS applied" if unit.alive
                     else "resolve: a STATUS on a corpse skipped"] += 1
            return __o(unit, *a)
        simultaneous._status_if_alive = wsa
        self._saved.append((simultaneous, "_status_if_alive", osa))

        oia = simultaneous._impulse_add

        def wia(table, unit, jx, jy, lever=0, twist=ONE, __o=oia):
            census.c["resolve: a PUSH applied"] += 1
            if lever:
                census.c["resolve: a PUSH with a lever"] += 1
            return __o(table, unit, jx, jy, lever, twist)
        simultaneous._impulse_add = wia
        self._saved.append((simultaneous, "_impulse_add", oia))

        OL = simultaneous.ExposureLanding

        class CountingLanding(OL):
            __slots__ = ()

            def land(self, __c=census):
                for unit, _dt, _add in self._raises:
                    __c.c["landing: a raise lands" if unit.alive
                          else "landing: a raise skipped (dead at the landing)"] += 1
                for _u in self._dead:
                    __c.c["landing: a death's zeroing"] += 1
                kills = set(__c._kills)
                if any((v, a) in kills for a, v in kills):
                    __c.c["resolve: two units kill each other"] += 1
                __c._kills = []
                return OL.land(self)
        simultaneous.ExposureLanding = CountingLanding
        self._saved.append((simultaneous, "ExposureLanding", OL))

        ores = simultaneous.SimultaneousEffects.resolve

        def wres(fx, __o=ores):
            n_push = sum(1 for e in fx._pending if e[0] == 4)
            census.c["resolve: a world call"] += sum(1 for e in fx._pending if e[0] == 3)
            a0 = census.c["resolve: a PUSH applied"]
            out = __o(fx)
            skipped = n_push - (census.c["resolve: a PUSH applied"] - a0)
            if skipped:
                census.c["resolve: a PUSH on a corpse skipped"] += skipped
            return out
        wres.__wrapped__ = ores
        simultaneous.SimultaneousEffects.resolve = wres
        self._saved.append((simultaneous.SimultaneousEffects, "resolve", ores))
        return self

    def __exit__(self, *exc):
        for cont, attr, o in reversed(self._saved):
            setattr(cont, attr, o)
        self._saved.clear()
        return False

    def _classify_4a(self, units, tick, tables, impulse_table):
        from simulation import combat, turning, unit_fixed, unit_impulse, verb_gates
        from simulation.status import composed_flags
        c = self.c
        for u in units:
            if not getattr(u, "pending_melee", False):
                continue
            if not u.alive:
                c["4a latch: a dead unit"] += 1
                continue
            if u.is_zombie:
                c["4a latch: a zombie"] += 1
                continue
            if not u.melee_id:
                c["4a latch: no melee row"] += 1
                continue
            if not composed_flags(u).can_act:
                c["4a latch: cannot act (stunned)"] += 1
                continue
            if u.throw_release_tick >= 0:
                c["4a latch: winding up"] += 1
                continue
            if u.plant_done_tick >= 0:
                c["4a latch: planting"] += 1
                continue
            if not verb_gates.can_melee(u, tick, tables, at_tick_start=False):
                c["4a latch: the cadence"] += 1
                continue
            w = tables.weapons.by_name[u.melee_id]
            cx, cy = unit_impulse.centre_q(u)
            offs = []
            for t in units:
                if t is u or not combat.melee_adjacent(u, t):
                    continue
                tx, ty = unit_impulse.centre_q(t)
                dx, dy = tx - cx, ty - cy
                off = 0 if (dx == 0 and dy == 0) else abs(turning.wrap_q16(
                    unit_fixed.atan2_q16(-dy, dx) - u.facing_q))
                if t.team == u.team:
                    if t.alive and off <= w.melee_arc_half_q:
                        c["4a: an adjacent teammate in the arc skipped"] += 1
                    continue
                if not t.alive:
                    if off <= w.melee_arc_half_q:
                        c["4a: an adjacent dead enemy in the arc skipped"] += 1
                    continue
                if off > w.melee_arc_half_q:
                    c["4a: an adjacent enemy outside the arc skipped"] += 1
                    continue
                offs.append((off, t))
            t = combat.melee_arc_target(units, u, w)
            if t is None:
                c["4a: a whiff (no target)"] += 1
                continue
            c["4a: a blow"] += 1
            c["4a: a blow on a zombie" if t.is_zombie else "4a: a blow on a marine"] += 1
            if len([o for o, _ in offs if o == min(o for o, _ in offs)]) > 1:
                c["4a: a tie to the lower id"] += 1
            if unit_impulse.centre_q(t) == unit_impulse.centre_q(u):
                c["4a: coincident centres (dead ahead)"] += 1
            c["4a: a blow with a status" if w.status_kind_id is not None
              else "4a: a blow without a status"] += 1
            j = unit_traits.push_impulse_q(impulse_table, u, w)
            c["4a: a blow with a push" if j else "4a: a blow with no push"] += 1
            if u.traits[unit_traits.STRIKE] != ONE:
                c["4a: a STRIKE hook"] += 1
            if u.traits[unit_traits.PUSH] != ONE and j:
                c["4a: a PUSH hook"] += 1


#: The branches the melee scene must take (a 0 fails as vacuous).
SCENE_BRANCHES = (
    "4a latch: a dead unit", "4a latch: a zombie", "4a latch: no melee row",
    "4a latch: cannot act (stunned)", "4a latch: winding up", "4a latch: planting",
    "4a latch: the cadence", "4a: a whiff (no target)", "4a: a blow",
    "4a: a blow on a zombie", "4a: a blow on a marine", "4a: a tie to the lower id",
    "4a: coincident centres (dead ahead)", "4a: a blow with a status",
    "4a: a blow with a push", "4a: a blow with no push", "4a: a STRIKE hook",
    "4a: a PUSH hook", "4a: an adjacent teammate in the arc skipped",
    "4a: an adjacent dead enemy in the arc skipped",
    "4a: an adjacent enemy outside the arc skipped",
    "4a: two blows on one target in one tick",
    "resolve: a HIT on the living", "resolve: a HIT kills",
    "resolve: a HIT on a corpse (no second kill)",
    "resolve: a HIT on a slot-killed unit reads its unzeroed exposure",
    "resolve: a STATUS applied", "resolve: a STATUS on a corpse skipped",
    "resolve: a PUSH applied", "resolve: a PUSH on a corpse skipped",
    "resolve: two units kill each other", "landing: a death's zeroing",
)
#: Branches the melee scene cannot reach, each with its reason (the
#: synthetic resolve part and ship_demo take the landing's raises).
UNREACHABLE = {
    "4a: a blow without a status":
        "every melee row a loadout may carry has a status (the knife, the one "
        "row without... has a crit, refused at the loadouts door and faulted by "
        "the device)",
    "resolve: a HIT with expose / landing raises":
        "a melee packet carries no expose; the raises are PART RESOLVE's and "
        "ship_demo's (the bots' rounds)",
    "resolve: a PUSH with a lever":
        "the melee push has no lever (recoil's lever is P7a.2's); PART RESOLVE "
        "takes it",
    "resolve: a world call": "4a records none; PART RESOLVE takes CHEW / PAYLOAD",
}


# ---------------------------------------------------------------------------
# The lockstep runner
# ---------------------------------------------------------------------------
def run_lockstep(name, drivers, slots, ticks, census=None, n_units=None):
    """Shadow ``slots`` on ``drivers`` for ``ticks`` ticks: every tick each
    slot equal (rows, lists, ring, planes, books), its coverage == the rows
    the host changed. Returns per label the env-ticks with coverage, the
    effect records 4a produced by kind, the ticks 4a acted."""
    t0 = time.perf_counter()
    cov = Counter()
    kinds = Counter()
    acted = 0
    n = 0
    ok = True
    ctx = census if census is not None else _Null()
    with ctx, ShadowLockstep(drivers, list(slots), n_units=n_units) as ls:
        for _ in range(ticks):
            if any(d.is_terminal() for d in drivers):
                break
            reps = ls.tick()
            n += 1
            for lab, rep in reps.items():
                if not rep.equal:
                    fail(f"{name}: {ls.divergences[-1]}")
                    ok = False
                    break
                if not np.array_equal(rep.coverage, rep.host_changed):
                    fail(f"{name} [{lab}] tick {rep.tick}: coverage "
                         f"{rep.coverage.tolist()} != host-changed "
                         f"{rep.host_changed.tolist()}")
                    ok = False
                    break
                cov[lab] += int((rep.coverage > 0).sum())
                if lab == SLOT_4A.label and rep.lists is not None:
                    inp = ls.last_inputs[lab].lists["effects"]
                    new = [len(rep.lists["effects"][e]) - len(inp[e])
                           for e in range(len(drivers))]
                    acted += sum(1 for x in new if x > 0)
                    for e in range(len(drivers)):
                        for r in rep.lists["effects"][e][len(inp[e]):]:
                            kinds[applier.KINDS[r[0] - 1]] += 1
                if lab == SLOT_4C.label and rep.events is not None:
                    kinds["ring events (4c)"] += int(rep.events.sum())
            if not ok:
                break
    total = n * len(drivers)
    print(f"{name}: N={len(drivers)} units per env "
          f"{[len(d.sim.units) for d in drivers]} (U {ls.U}), {n} ticks"
          f"{', env i == env i alone' if len(drivers) > 1 and ok else ''}; "
          + ", ".join(f"[{lab}] coverage on {100 * cov[lab] / max(1, total):.1f} % "
                      f"of env-ticks" for lab in (s.label for s in slots))
          + f"; 4a acted on {acted} env-ticks, records {dict(kinds)}"
          + f" ({time.perf_counter() - t0:.0f} s)")
    return cov, kinds, acted


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# ---------------------------------------------------------------------------
# PART TABLES
# ---------------------------------------------------------------------------
def table_reference(sim) -> dict:
    """The P7a.1 device-table rows read from the reference objects."""
    from simulation.weapons import get_tables
    W = get_tables().weapons.by_name
    mp = sim.impulse_table.melee_push_q  # traits-class: the reference read of the class value the table uploads
    return {
        "weapon_melee_damage_q": tuple(int(w.melee_damage) * 65536 for w in W.values()),  # traits-class: as above
        "weapon_melee_dtype": tuple(-1 if w.melee_dtype_id is None else int(w.melee_dtype_id)
                                    for w in W.values()),
        "weapon_melee_status_kind": tuple(-1 if w.status_kind_id is None
                                          else int(w.status_kind_id) for w in W.values()),
        "weapon_melee_status_ticks": tuple(int(w.status_ticks) for w in W.values()),
        "weapon_melee_arc_half_q": tuple(int(w.melee_arc_half_q) for w in W.values()),
        "weapon_rof_ticks": tuple(int(w.rof_interval_ticks) for w in W.values()),
        "weapon_melee_crit": tuple(int(w.crit_chance > 0) for w in W.values()),
        "weapon_melee_push_q": tuple(int(mp.get(n, 0)) for n in W),
    }


def part_tables():
    a, b = MeleeDriver(hooks=False), ShipDriver(hooks=False)
    sims = [a.sim, b.sim]
    tab = unit_tables.values_batch(sims)
    st = unit_store.DeviceStore(2, 4, tables=tab)
    got = st.tables.download()
    n = 0
    as_t = lambda x: tuple(x) if isinstance(x, (tuple, list)) else (x,)  # noqa: E731
    for e, sim in enumerate(sims):
        for k, v in table_reference(sim).items():
            dv = unit_tables.read(got[e], k)
            n += 1
            if as_t(dv) != as_t(v):
                fail(f"tables env {e} row {k}: device {dv} != reference {v}")
    push = [unit_tables.read(got[e], "weapon_melee_push_q") for e in range(2)]
    if push[0] == push[1]:
        fail("tables: the two maps' melee pushes are equal (one tile size?)")
    if not any(unit_tables.read(got[0], "weapon_melee_crit")):
        fail("tables: no row with a crit (the knife's refusal is untested)")
    print(f"tables: {n} P7a.1 rows over 2 envs (1 m and 1/3 m tiles) equal the "
          f"reference objects (T {unit_tables.layout().width}); the applier layout "
          f"F {applier.F}, kinds {applier.KINDS}, stages {applier.STAGE_NAMES}")


# ---------------------------------------------------------------------------
# PART LIB
# ---------------------------------------------------------------------------
def part_lib(trials=LIB_TRIALS):
    """The device melee gate, target and adjacency on SYNTHETIC rows against
    the Python on the decoded rows."""
    from simulation import combat, verb_gates
    from simulation.factions import ZOMBIE_TEAM
    from simulation.status import STUNNED, apply_status
    from simulation.unit_schema import table_row
    rng = np.random.default_rng(7101)
    drv, shp = MeleeDriver(hooks=False), ShipDriver(hooks=False)
    sims = [drv.sim, shp.sim]
    lay = unit_store.layout()
    col = lambda n: lay.column(n).offset  # noqa: E731
    base = [u for u in drv.sim.units if u.alive and not u.is_zombie][:4]
    apply_status(base[1], STUNNED, 0, 4)
    n, U = 2, 24
    tabs = unit_tables.values_batch(sims)
    W = sims[0].weapons_tables.weapons.by_name
    rows_m = [-1, table_row("weapons", "rifle_butt"), table_row("weapons", "arc_baton"),
              table_row("weapons", "k5_carbine")]
    checks = Counter()
    for trial in range(trials):
        tick = int(rng.choice([0, 7, 100, 10 ** 6]))
        rows = np.zeros((n, U, lay.width), dtype=np.int64)
        for e in range(n):
            for r in range(U):
                row = unit_store.pack([base[r % len(base)]])[0]
                row[col("id")] = 100 + r
                row[col("life_state")] = int(rng.random() < 0.15)
                row[col("is_zombie")] = int(rng.random() < 0.12)
                row[col("team")] = int(rng.choice([0, 1, ZOMBIE_TEAM]))
                row[col("melee_id")] = int(rng.choice(rows_m))
                row[col("throw_release_tick")] = int(rng.choice([-1, -1, -1, tick - 1, tick,
                                                                 tick + 3]))
                row[col("plant_done_tick")] = int(rng.choice([-1, -1, -1, tick - 1, tick,
                                                              tick + 3]))
                row[col("last_blow_tick")] = int(rng.choice([-1, tick - 14, tick - 13,
                                                             tick - 19, tick]))
                # a 6 x 6 tile patch: adjacency, arcs, ties, coincident centres
                row[col("x_q")] = int(rng.integers(0, 6)) * ONE + int(rng.choice([0, ONE // 2]))
                row[col("y_q")] = int(rng.integers(0, 6)) * ONE + int(rng.choice([0, ONE // 3]))
                row[col("facing_q")] = int(rng.choice([EAST, WEST, NORTH, SOUTH, PI_Q16 // 4,
                                                       int(rng.integers(-PI_Q16, PI_Q16))]))
                tr = lay.column("traits")
                row[tr.offset + unit_traits.STRIKE] = int(rng.choice([ONE, ONE * 3 // 2]))
                row[lay.valid] = int(rng.random() < 0.9)
                if not row[lay.valid]:
                    row[lay.valid] = 0                  # live-looking, valid 0
                rows[e, r] = row
        st = unit_store.DeviceStore(n, U, tables=tabs)
        st.upload(rows)
        bidx = int(rng.integers(0, U))
        out = unit_kernels.verbs_probe(st, 0, tick, bidx)
        if not np.array_equal(st.download(), rows):
            fail(f"lib trial {trial}: the probe wrote a row")
        for e in range(n):
            tables = sims[e].weapons_tables
            units = []
            idx = {}
            for r in range(U):
                if not rows[e, r, lay.valid]:
                    units.append(None)
                    continue
                u = unit_store.blank_units(1)[0]
                unit_store.unpack(rows[e, r:r + 1], [u])
                u.offsets = list(base[r % len(base)].offsets)   # the species' square
                idx[id(u)] = r
                units.append(u)
            live = [u for u in units if u is not None]
            for r in range(U):
                o = out[e, r]
                u = units[r]
                if u is None:
                    if o.any():
                        fail(f"lib trial {trial}: a padding row reported a result")
                    continue
                want13 = int(verb_gates.can_melee(u, tick, tables, at_tick_start=False))
                want14 = int(verb_gates.can_melee(u, tick, tables, at_tick_start=True))
                if u.melee_id:
                    t = combat.melee_arc_target(live, u, W[u.melee_id])
                    want16 = -1 if t is None else idx[id(t)]
                else:
                    want16 = -2
                other = units[bidx]
                want17 = -1 if other is None else int(combat.melee_adjacent(u, other))
                dev = (int(o[13]), int(o[14]), int(o[16]), int(o[17]))
                want = (want13, want14, want16, want17)
                if dev != want or int(o[15]):
                    fail(f"lib trial {trial} env {e} row {r}: device {dev} (bad "
                         f"{int(o[15])}) != Python {want}")
                    return
                checks[("can_melee", want13)] += 1
                checks[("can_melee at the tick's start", want14)] += 1
                checks[("arc target", int(want16 >= 0))] += 1
                checks[("adjacent", want17)] += 1
    # the knife (a crit) as the melee row: the slot faults by name
    print(f"lib: {sum(v for k, v in checks.items() if k[0] == 'can_melee')} synthetic "
          f"rows over {trials} trials equal the Python ({sorted(checks.items())}); "
          f"padding rows silent")
    for name in ("can_melee", "can_melee at the tick's start", "arc target", "adjacent"):
        if not (checks[(name, 0)] and checks[(name, 1)]):
            fail(f"lib: {name!r} never gave both answers (vacuous)")


def part_knife():
    """A striker whose melee row has a crit (melee_blow would draw; the
    loadouts door refuses one): the device faults it, the Python face refuses
    the run by name."""
    drv = MeleeDriver(hooks=False)
    sim = drv.sim
    u = drv.who["A"]
    u.melee_id = "combat_knife"
    u.pending_melee = True
    st = unit_store.DeviceStore(1, len(sim.units), tables=unit_tables.values_batch([sim]))
    st.upload(unit_store.pack(sim.units)[None])
    fx = applier.DeviceApplier(1, applier.E_CAP, n_units=len(sim.units))
    fx.begin()
    try:
        unit_kernels.melee(st, fx, np.array([sim.tick]))
        fail("knife: a striker with a crit row was not refused")
    except unit_kernels.UnitKernelFault as ex:
        if "melee" not in str(ex):
            fail(f"knife: the refusal does not name the kernel: {ex}")
        print("knife: a striker holding a crit row faults; the run refused by name")


# ---------------------------------------------------------------------------
# PART SCENE / SHIP / BATCH
# ---------------------------------------------------------------------------
def part_scene():
    census = Census()
    cov, kinds, acted = run_lockstep("scene melee", [MeleeDriver()], [SLOT_4A, SLOT_4C],
                                     MELEE_TICKS, census=census)
    print("scene melee census: " + "; ".join(f"{k} = {census.c[k]}"
                                             for k in sorted(census.c)))
    for k in SCENE_BRANCHES:
        if not census.c.get(k):
            fail(f"scene melee: the branch {k!r} was never taken (vacuous)")
    print("scene melee unreachable: " + "; ".join(f"{k}: {v}" for k, v in UNREACHABLE.items()))
    if not acted or not kinds.get("HIT") or not kinds.get("STATUS") or not kinds.get("PUSH"):
        fail(f"scene melee: 4a recorded {dict(kinds)} (vacuous)")


def part_ship():
    census = Census()
    cov, kinds, acted = run_lockstep("ship (bots, seed 1)", [ShipDriver()],
                                     [SLOT_4A, SLOT_4C], SHIP_TICKS, census=census)
    print("ship census: " + "; ".join(f"{k} = {census.c[k]}" for k in sorted(census.c)))
    print(f"ship: 4a acted on {acted} ticks of {SHIP_TICKS}")


def part_batch():
    for n in BATCH_NS:
        # the planes' batch is homogeneous in its grid (one (h, w)): N melee
        # scenes of different seeds, layouts, unit counts and fields
        melee = [MeleeDriver(variant=e + 1) for e in range(n)]
        cov, kinds, acted = run_lockstep(f"batch N={n}", melee,
                                         [SLOT_4A, SLOT_4C], BATCH_TICKS)
        counts = [len(d.sim.units) for d in melee]
        mats = {d.sim.gmap.material.tobytes() for d in melee}
        print(f"batch N={n}: unit counts {counts}, {len(mats)} layouts, variants "
              f"{[d.variant for d in melee]} (seeds 20261115 + 7 v)")
        if len(set(counts)) < 2 or len(mats) < 2:
            fail(f"batch N={n}: the envs do not differ in unit count and layout")
        if not acted:
            fail(f"batch N={n}: 4a never acted (vacuous)")
        # 4a (no planes) in a batch WITH ship_demo's bots match beside them
        mixed = [MeleeDriver(variant=e + 1) for e in range(n - 1)] + [ShipDriver(seed=2)]
        run_lockstep(f"batch 4a N={n} (with ship_demo)", mixed, [SLOT_4A],
                     BATCH_TICKS // 2)


# ---------------------------------------------------------------------------
# PART RESOLVE (synthetic lists)
# ---------------------------------------------------------------------------
def _planes(gmap) -> dict:
    return {n: np.array(getattr(gmap, n), copy=True) for n in applier.PLANE_NAMES}


def _dev_planes(per_env: list) -> dict:
    import cupy as cp
    return {n: cp.asarray(np.ascontiguousarray(np.stack([p[n] for p in per_env])))
            for n in applier.PLANE_NAMES}


def _books(gmap) -> dict:
    return {b.name: applier.host_book(gmap, b.host) for b in applier.BOOKS
            if b.host is not None}


class ResolveWorld:
    """A fresh melee map with named units for a synthetic list (built the same
    every time, so two worlds take a list in two orders)."""

    def __init__(self):
        from simulation.materials import MAT_FURNITURE, MAT_HULL
        d = MeleeDriver(variant=0, hooks=False)
        self.sim, self.w = d.sim, d.who
        g = self.sim.gmap
        # the breach chain: H in the top hull (space above), P behind it
        self.H, self.P = (1, 30), (2, 30)
        g.material[2, 30] = MAT_FURNITURE
        g.on_tile_changed(2, 30)
        assert int(g.material[1, 30]) == MAT_HULL
        # a wood tile at hp 0 that no chew of no damage may destroy
        g.wall_hp[24, 9] = 0
        # exposure fixtures (the landing's reads)
        self.w["T"].exposure[0] = ONE // 4
        self.w["T2"].exposure[0] = ONE // 3

    def effects(self, items):
        from simulation.simultaneous import SimultaneousEffects
        fx = SimultaneousEffects()
        for it in items:
            it(self, fx)
        return fx


def _hit(att, tgt, dmg, expose=0, dtype=0, ap=0.0, cause="bullet"):
    from simulation.damage import DamagePacket

    def f(W, fx):
        a, t = W.w[att], W.w[tgt]
        fx.hit(t, DamagePacket(amount=float(dmg), dtype=dtype, attacker_id=a.id, ap=ap,
                               expose=int(expose), dir_x_q=3 * ONE, dir_y_q=-ONE),
               W.sim.tick_events, source=cause)
    return f


def _status(tgt, kind=None, dur=4, src="A"):
    from simulation.status import STUNNED

    def f(W, fx):
        fx.status(W.w[tgt], STUNNED if kind is None else kind, magnitude=0,
                  duration_ticks=dur, source_id=None if src is None else W.w[src].id)
    return f


def _push(tgt, jx, jy, lever=0, twist=ONE):
    def f(W, fx):
        fx.push(W.sim.impulse_table, W.w[tgt], jx, jy, lever, twist)
    return f


def _chew(y, x, dmg):
    def f(W, fx):
        from simulation.combat import chew_wall
        fx.world(chew_wall, W.sim.gmap, y, x, dmg)
    return f


def _run_list(tag, items, census, *, reverse=False, expect_payload=False):
    """One synthetic list: the host's resolve on one world, the device's walk
    (RESOLVE) on the same start state; rows, ring, planes, books compared.
    Returns the host's post state (rows, events, planes) for order checks."""
    W = ResolveWorld()
    sim = W.sim
    seq = list(reversed(items)) if reverse else list(items)
    fx = W.effects(seq)
    recs = applier.records_of(sim, fx._pending)
    for r in recs:
        census[f"kind {applier.KINDS[int(r[0]) - 1]}"] += 1
    U = len(sim.units)
    pre_rows = unit_store.pack(sim.units, U)
    pre_planes = _planes(sim.gmap)
    b0 = _books(sim.gmap)
    n0 = len(sim.tick_events)
    fx.resolve()
    post_rows = unit_store.pack(sim.units, U)
    host_ev = [unit_events.record_of(v) for v in sim.tick_events[n0:]
               if unit_events.is_synced(v)]
    post_planes = _planes(sim.gmap)
    b1 = _books(sim.gmap)
    st = unit_store.DeviceStore(1, U, tables=unit_tables.values_batch([sim]))
    st.upload(pre_rows[None])
    ring = unit_events.EventRing(1, U)
    ring.begin()
    dev = _dev_planes([pre_planes])
    dl = applier.DeviceApplier(1, applier.E_CAP, n_units=U)
    dl.stage_lists([recs])
    if [tuple(int(x) for x in r) for r in recs] != dl.records()[0]:
        fail(f"resolve {tag}: the staged list is not the host's records")
    cov, books = unit_kernels.resolve(st, dl, dev, ring)
    got = st.download()[0]
    ok = True
    if not np.array_equal(got, post_rows):
        r, c = np.argwhere(got != post_rows)[0]
        fail(f"resolve {tag}: row {r} column {unit_store.layout().column_at(int(c))}: "
             f"host {post_rows[r, c]} device {got[r, c]}")
        ok = False
    if ring.records()[0] != host_ev:
        fail(f"resolve {tag}: ring {ring.records()[0]} != host events {host_ev}")
        ok = False
    for nme in applier.PLANE_NAMES:
        fd = first_field_diff(nme, dev[nme].get(), post_planes[nme][None])
        if fd is not None:
            fail(f"resolve {tag}: plane {nme} at {fd[1]}: host {fd[2]} device {fd[3]}")
            ok = False
            break
    for k in b0:
        if int(books[k][0]) != b1[k] - b0[k]:
            fail(f"resolve {tag}: book {k} device {int(books[k][0])} != host {b1[k] - b0[k]}")
            ok = False
    host_cov = int((pre_rows != post_rows).any(axis=1).sum())
    if int(cov[0]) != host_cov:
        fail(f"resolve {tag}: coverage {int(cov[0])} != host-changed {host_cov}")
        ok = False
    return ok, (post_rows, host_ev, post_planes)


class _armored:
    """Every species' kinetic armor at 3 points for the block (the units the
    world builds stamp it; the device tables read it): no shipped profile
    has armor, so the AP door is otherwise unobservable."""

    def __enter__(self):
        from dataclasses import replace
        from simulation import species as S
        self.saved = dict(S.SPECIES_REGISTRY)
        for k, sd in self.saved.items():
            armor = list(sd.mitigation.armor)
            armor[0] = 3.0
            S.SPECIES_REGISTRY[k] = replace(sd, mitigation=replace(
                sd.mitigation, armor=tuple(armor)))
        return self

    def __exit__(self, *exc):
        from simulation import species as S
        S.SPECIES_REGISTRY.clear()
        S.SPECIES_REGISTRY.update(self.saved)
        return False


def _at_once(items):
    """The same list applied AT ONCE (no landing): what the landing must
    differ from."""
    from simulation.simultaneous import SimultaneousEffects
    W = ResolveWorld()
    fx = SimultaneousEffects(deferred=False)
    n0 = len(W.sim.tick_events)
    for it in items:
        it(W, fx)
    return (unit_store.pack(W.sim.units, len(W.sim.units)),
            [unit_events.record_of(v) for v in W.sim.tick_events[n0:]
             if unit_events.is_synced(v)])


def part_resolve():
    from simulation.damage import HEAL
    census = Counter()
    # 1. the landing list: raises held; a corpse's unzeroed exposure read;
    #    a raise skipped for a unit dead at the landing; STATUS / PUSH on the
    #    living and on a corpse; a lever and a twist; AP; a HEAL; a mutual kill
    landing = [
        _hit("A", "T2", 8, expose=ONE // 2),            # T2 lives: a raise held
        _hit("B", "T2", 8, expose=ONE // 2),            # reads T2's slot-start level
        _hit("S1", "T", 50, expose=ONE),                # T dies: its raise skipped
        _status("T"), _push("T", 5 * ONE, 0),           # aimed at the corpse: skipped
        _hit("S2", "T", 9),                             # the corpse: unzeroed exposure
        _status("T2", dur=6), _push("T2", 3 * ONE, -2 * ONE, lever=ONE // 3,
                                    twist=ONE * 3 // 2),
        _push("T2", -ONE, ONE, lever=-5 * ONE),         # the lever clamped
        _status("T2", dur=2, src=None),                 # a source of None
        _hit("B", "T2", 1, expose=3 * ONE),             # the raises pass x_max: the cap
        _hit("C", "T2", 4, ap=2.5),                     # the AP door
        _hit("C", "T2", -3, dtype=HEAL, cause="regen"), # HEAL (a negative amount): the identity
        _hit("M", "E", 50), _hit("E", "M", 50),         # a mutual kill
    ]
    ok, (rows_l, ev_l, _p) = _run_list("landing", landing, census)
    rows_o, ev_o = _at_once(landing)
    if np.array_equal(rows_l, rows_o) and ev_l == ev_o:
        fail("resolve: the landing gave the same result as applying at once (vacuous)")
    else:
        census["the landing's result differs from at once"] += 1
    # 1b. the AP door against armor (no shipped profile has any: a fixture)
    with _armored():
        ok &= _run_list("armor", [_hit("A", "T2", 6, ap=1.25), _hit("B", "T2", 2),
                                  _hit("C", "T2", 9, ap=9.0)], census)[0]
        census["armor: a packet's AP against a non-zero armor"] += 1
    # 2. CHEW: a wall chewed to zero and destroyed, air and dead damage no-ops
    chew = [_chew(24, 6, 25.0), _chew(24, 6, 35.0),       # 60 hp wood: to exactly 0
            _chew(10, 10, 50.0),                          # air: a no-op
            _chew(24, 9, 0.0),                            # no damage: a no-op (hp 0)
            _chew(24, 8, 12.5)]                           # a quantized partial chew
    ok &= _run_list("chew", chew, census)[0]
    # 3. the order: a kill's credit and a breach chain, forward vs reversed
    W0 = ResolveWorld()
    order = [_hit("S3", "T2", 60), _chew(W0.H[0], W0.H[1], 400.0),
             _chew(W0.P[0], W0.P[1], 400.0), _hit("S4", "T2", 60)]
    okf, fwd = _run_list("order forward", order, census)
    okr, rev = _run_list("order reversed", order, census, reverse=True)
    ok &= okf and okr
    if fwd[1] == rev[1] or all(np.array_equal(fwd[2][n], rev[2][n]) for n in fwd[2]):
        fail("resolve: the forward and reversed orders do not differ in the ring AND "
             "the planes (the order is not proven to matter)")
    else:
        census["forward != reversed (ring and planes)"] += 1
    # 4. PAYLOAD: recorded, compared, refused at the apply
    W = ResolveWorld()
    sim = W.sim
    from simulation.payloads import execute_payload
    from simulation.simultaneous import SimultaneousEffects
    payload = next(iter(sim.weapons_tables.payloads.by_name.values()))
    fx = SimultaneousEffects()
    fx.world(execute_payload, sim.gmap, sim.edit_queue, sim.units, 10, 12, payload,
             sim.draws, events=sim.tick_events, kind="shell", owner_id=W.w["A"].id,
             approach=(10, 11))
    recs = applier.records_of(sim, fx._pending)
    want = applier.make_record("PAYLOAD", y=10, x=12, payload=0, owner=W.w["A"].id,
                               approach_y=10, approach_x=11, has_approach=1,
                               kind=applier.PAYLOAD_KINDS.index("shell"))
    if [tuple(int(x) for x in recs[0])] != [want]:
        fail(f"payload: record {recs[0].tolist()} != {want}")
    U = len(sim.units)
    st = unit_store.DeviceStore(1, U, tables=unit_tables.values_batch([sim]))
    st.upload(unit_store.pack(sim.units, U)[None])
    ring = unit_events.EventRing(1, U)
    ring.begin()
    dl = applier.DeviceApplier(1, applier.E_CAP, n_units=U)
    dl.stage_lists([recs])
    if dl.records()[0] != [want]:
        fail("payload: the device's list is not the record")
    try:
        unit_kernels.resolve(st, dl, _dev_planes([_planes(sim.gmap)]), ring)
        fail("payload: the applier applied a PAYLOAD record")
    except applier.EffectRefused as ex:
        if "PAYLOAD" not in str(ex):
            fail(f"payload: the refusal does not name PAYLOAD: {ex}")
        census["PAYLOAD refused by name at the apply"] += 1
    # 5. FACE and an unknown world call: refused at the record function
    fx = SimultaneousEffects()
    fx.face(W.w["A"], 5)
    fx2 = SimultaneousEffects()
    fx2.world(print, "x")
    for f, what in ((fx, "FACE"), (fx2, "world call")):
        try:
            applier.records_of(sim, f._pending)
            fail(f"records: a {what} effect was not refused")
        except applier.EffectRefused:
            census[f"{what} refused at the record function"] += 1
    print("resolve: " + "; ".join(f"{k} = {v}" for k, v in sorted(census.items())))
    need = ("kind HIT", "kind STATUS", "kind PUSH", "kind CHEW",
            "armor: a packet's AP against a non-zero armor",
            "the landing's result differs from at once",
            "forward != reversed (ring and planes)", "PAYLOAD refused by name at the apply",
            "FACE refused at the record function", "world call refused at the record function")
    for k in need:
        if not census.get(k):
            fail(f"resolve: {k!r} never happened (vacuous)")


# ---------------------------------------------------------------------------
# PART PADDING
# ---------------------------------------------------------------------------
def part_padding():
    """Padding rows filled with copies of live acting rows (valid cleared):
    the valid rows, the lists, the ring and the coverage equal the clean run's,
    the padding rows come back as they went; a HIT aimed at a padding row
    faults and is refused."""
    lay = unit_store.layout()
    drivers = [MeleeDriver(variant=0), MeleeDriver(variant=1)]
    U = max(len(d.sim.units) for d in drivers) + 4
    done = False
    with ShadowLockstep(drivers, [SLOT_4A, SLOT_4C], n_units=U, check_alone=False) as ls:
        for _ in range(MELEE_TICKS):
            reps = ls.tick()
            r4a = reps[SLOT_4A.label]
            if not any(len(x) for x in r4a.lists["effects"]):
                continue
            for k, slot in enumerate((SLOT_4A, SLOT_4C)):
                clean = np.stack(ls._pre[k])
                inp = ls.last_inputs[slot.label]
                dirty = clean.copy()
                for e in range(2):
                    live = [r for r in range(U) if clean[e, r, lay.valid]
                            and clean[e, r, lay.column("pending_melee").offset]]
                    live = live or [r for r in range(U) if clean[e, r, lay.valid]]
                    for i, r in enumerate(r for r in range(U) if not clean[e, r, lay.valid]):
                        dirty[e, r] = clean[e, live[(3 * i + e) % len(live)]]
                        dirty[e, r, lay.valid] = 0
                runs = []
                for rows in (clean, dirty):
                    ls.store.upload(rows)
                    ls.ring.begin()
                    ls._stage_lists(slot, ls.dev_lists, inp)
                    import cupy as cp
                    for kk, vv in inp._host_fields.items():   # the entry's planes
                        inp.fields[kk][...] = cp.asarray(vv)
                    res = slot.call(ls.store, inp, ls.ring, None, ls.dev_lists)
                    cov = res[0] if isinstance(res, tuple) else res
                    runs.append((np.asarray(cov), ls.store.download(),
                                 ls.dev_lists["effects"].records(), ls.ring.records()))
                valid = clean[:, :, lay.valid].astype(bool)
                (c0, g0, l0, e0), (c1, g1, l1, e1) = runs
                if not np.array_equal(g1[valid], g0[valid]):
                    fail(f"padding [{slot.label}]: the valid rows differ")
                if not np.array_equal(g1[~valid], dirty[~valid]):
                    fail(f"padding [{slot.label}]: a padding row was written")
                if l1 != l0 or e1 != e0:
                    fail(f"padding [{slot.label}]: the list or the ring differs")
                if not np.array_equal(c1, c0):
                    fail(f"padding [{slot.label}]: coverage {c1.tolist()} vs {c0.tolist()}")
                if k == 0:
                    n_latched = int(sum(dirty[e, r, lay.column("pending_melee").offset] != 0
                                        for e in range(2) for r in range(U)
                                        if not dirty[e, r, lay.valid]))
            print(f"padding: tick {r4a.tick}, N=2, U={U}: {int((~valid).sum())} padding rows "
                  f"(live copies, {n_latched} latched) untouched; 4a's and 4c's valid rows, "
                  f"list and ring equal the clean run")
            if n_latched == 0:
                fail("padding: no padding row held a latch (vacuous)")
            done = True
            break
    if not done:
        fail("padding: 4a never recorded (vacuous)")
    # a HIT aimed at a padding row
    d = MeleeDriver()
    sim = d.sim
    U = len(sim.units) + 2
    st = unit_store.DeviceStore(1, U, tables=unit_tables.values_batch([sim]))
    rows = unit_store.pack(sim.units, U)[None]
    rows[0, len(sim.units)] = rows[0, 0]
    rows[0, len(sim.units), lay.valid] = 0
    st.upload(rows)
    ring = unit_events.EventRing(1, U)
    ring.begin()
    dl = applier.DeviceApplier(1, 8, n_units=U)
    rec = applier.make_record("HIT", row=len(sim.units), amount_q=10 * ONE, dtype=0,
                              attacker=-1, indirect=0, expose=0, dir_x_q=0, dir_y_q=0,
                              ap_q=0, cause=0)
    dl.stage_lists([np.array([rec])])
    try:
        unit_kernels.resolve(st, dl, _dev_planes([_planes(sim.gmap)]), ring)
        fail("padding: a HIT on a padding row was not refused")
    except applier.ApplierFault:
        if not np.array_equal(st.download()[0], rows[0]):
            fail("padding: a HIT on a padding row wrote a row")
        print("padding: a HIT aimed at a padding row faults, writes nothing, refused by name")


# ---------------------------------------------------------------------------
# PART OVERFLOW
# ---------------------------------------------------------------------------
def part_overflow():
    from simulation import simulation as simmod
    drv = MeleeDriver()
    held = {}
    orig = simmod.process_melee_intents
    tested = set()
    for _ in range(40):
        sim = drv.sim
        U = len(sim.units)

        def spy(units, tick, *a, __o=orig, **kw):
            held["pre"] = unit_store.pack(units, U)
            held["tick"] = tick
            return __o(units, tick, *a, **kw)
        simmod.process_melee_intents = spy
        try:
            drv.tick()
        finally:
            simmod.process_melee_intents = orig
        st = unit_store.DeviceStore(1, U, tables=unit_tables.values_batch([sim]))
        full = applier.DeviceApplier(1, applier.E_CAP, n_units=U)
        st.upload(held["pre"][None])
        full.begin()
        unit_kernels.melee(st, full, np.array([held["tick"]]))
        want = full.records()[0]
        if len(want) < 3:
            continue
        # the APPEND: a producer continues the list (slot 4's records first)
        lead = [applier.make_record("CHEW", y=24, x=5, damage_q=ONE, live=1),
                applier.make_record("PUSH", row=0, jx_q=ONE, jy_q=0, lever_q=0,
                                    twist_q16=ONE)]
        cont = applier.DeviceApplier(1, applier.E_CAP, n_units=U)
        st.upload(held["pre"][None])
        cont.stage_lists([np.array(lead)])
        unit_kernels.melee(st, cont, np.array([held["tick"]]))
        if cont.records()[0] != [tuple(r) for r in lead] + want:
            fail("append: 4a did not continue the staged list (its records after the "
                 "staged ones, in order)")
        else:
            tested.add("append")
        # the list's capacity
        cap = 2
        small = applier.DeviceApplier(1, cap, n_units=U)
        st.upload(held["pre"][None])
        small.begin()
        try:
            unit_kernels.melee(st, small, np.array([held["tick"]]))
            fail("overflow: a list past its capacity was not refused")
        except applier.EffectOverflow as ex:
            if "melee" not in str(ex) or "env(s) [0]" not in str(ex):
                fail(f"overflow: the refusal does not name the kernel and env: {ex}")
        recs, counts, dropped = small.download()
        if int(counts[0]) != cap or int(dropped[0]) != len(want) - cap or \
                small.records()[0] != want[:cap]:
            fail(f"overflow: count {int(counts[0])} dropped {int(dropped[0])} for "
                 f"{len(want)} records into {cap}")
        tested.add("list")
        # a row's staging (a blow is up to 3 records)
        narrow = applier.DeviceApplier(1, applier.E_CAP, n_units=U, row_cap=2)
        st.upload(held["pre"][None])
        narrow.begin()
        try:
            unit_kernels.melee(st, narrow, np.array([held["tick"]]))
            fail("overflow: a row's staging past its bound was not refused")
        except applier.EffectOverflow as ex:
            if "row's staging" not in str(ex):
                fail(f"overflow: the staging refusal does not say so: {ex}")
            tested.add("staging")
        # a staged host list past the capacity, walked: its first entries applied
        st.upload(held["pre"][None])
        ring = unit_events.EventRing(1, U)
        ring.begin()
        walk = applier.DeviceApplier(1, cap, n_units=U)
        walk.stage_lists([np.array(want)])
        try:
            unit_kernels.resolve(st, walk, _dev_planes([_planes(sim.gmap)]), ring)
            fail("overflow: a staged list past its capacity was not refused")
        except applier.EffectOverflow:
            if int(st.last_books["applied"][0]) != cap or \
                    int(st.last_books["dropped"][0]) != len(want) - cap:
                fail(f"overflow: the walk applied {int(st.last_books['applied'][0])} and "
                     f"dropped {int(st.last_books['dropped'][0])}")
            tested.add("walk")
        print(f"overflow: {len(want)} records: into a list of {cap} ({cap} kept, the first; "
              f"{len(want) - cap} counted), through a staging of 2 per row, and a staged "
              f"list of {len(want)} walked with capacity {cap} -- each refused by name")
        break
    for t in ("append", "list", "staging", "walk"):
        if t not in tested:
            fail(f"overflow: the {t} overflow was never tested (vacuous)")


# ---------------------------------------------------------------------------
# PART CHAINED
# ---------------------------------------------------------------------------
def _books_host_side(sim, pre, post):
    """What the chained 4a + 4c leave on the host besides rows: the pushed
    units' momentum books (REFERENCE_ONLY; unit_impulse.add's IN entries)."""
    from simulation.unit_impulse import IN_X, IN_Y
    lay = unit_store.layout()
    ix, iy = lay.column("imp_x").offset, lay.column("imp_y").offset
    for r, u in enumerate(sim.units):
        u.impulse_books[IN_X] += int(post[r, ix] - pre[r, ix])
        u.impulse_books[IN_Y] += int(post[r, iy] - pre[r, iy])


def part_chained():
    twin, ref = MeleeDriver(), MeleeDriver()
    t0 = time.perf_counter()
    n = 0
    with ChainedSlots(twin, [SLOT_4A, SLOT_4C], host_side=_books_host_side) as ch:
        for _ in range(CHAIN_TICKS):
            twin.tick()
            ref.tick()
            n += 1
            if not _same(twin.sim, ref.sim):
                fail(f"chained 4a + 4c: the twin left the reference at tick {twin.sim.tick}")
                break
            if any(ch.books.get(b, 0) for b in ("tiles_changed", "n_destruction_seed")):
                fail("chained 4a + 4c: a wall changed in the melee scene (its host side "
                     "is not landed here)")
    acted = sum(1 for covs in ch.coverage if int(covs[0][0]) > 0)
    recs = sum(len(x["effects"]) for x in ch.lists)
    print(f"chained 4a + 4c (melee): {n} ticks, the chained twin equal to the reference "
          f"every tick ({ch.runs} chain runs; 4a changed rows on {acted} ticks; "
          f"{recs} effect records walked; {time.perf_counter() - t0:.0f} s)")
    if ch.runs != n or acted == 0 or recs == 0:
        fail("chained 4a + 4c: vacuous")


# ---------------------------------------------------------------------------
# Main, bench
# ---------------------------------------------------------------------------
PARTS = ("tables", "lib", "knife", "scene", "resolve", "batch", "padding", "overflow",
         "chained", "ship")


def main(parts=None) -> int:
    t0 = time.perf_counter()
    want = parts or os.environ.get("UNIT_MELEE_PARTS", ",".join(PARTS)).split(",")
    for p in want:
        globals()[f"part_{p}"]()
    print(f"total {time.perf_counter() - t0:.0f} s")
    ok = not FAILS
    print(f"UNIT_MELEE_RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def bench(reps: int = 50) -> int:
    """The TIMING table: median ms of slot 4a and of the resolve stage at N =
    1 / 8 / 64 envs and 1 / 8 / 64 effects per env (synthetic HIT / STATUS /
    PUSH lists on the melee scene's rows)."""
    import cupy as cp
    drv = MeleeDriver()
    for _ in range(3):
        drv.tick()
    sim = drv.sim
    U = len(sim.units)
    rows1 = unit_store.pack(sim.units, U)
    for r in range(U):                                  # every row latched
        rows1[r, unit_store.layout().column("pending_melee").offset] = 1
    tab1 = unit_tables.values_batch([sim])
    planes1 = _planes(sim.gmap)
    lines = []
    for n_env in (1, 8, 64):
        st = unit_store.DeviceStore(n_env, U, tables=np.repeat(tab1, n_env, axis=0))
        rows = np.repeat(rows1[None], n_env, axis=0)
        dev = _dev_planes([planes1] * n_env)
        ticks = np.full(n_env, sim.tick, dtype=np.int64)
        ring = unit_events.EventRing(n_env, U, e_max=4096)
        fx = applier.DeviceApplier(n_env, 256, n_units=U)
        ts = []
        for _ in range(reps):
            st.upload(rows)
            fx.begin()
            cp.cuda.Stream.null.synchronize()
            t = time.perf_counter()
            unit_kernels.melee(st, fx, ticks)
            ts.append((time.perf_counter() - t) * 1e3)
        lines.append(("4a melee", n_env, U, float(np.median(ts))))
        for k in (1, 8, 64):
            per = []
            for i in range(k):
                kind = i % 3
                tgt = 1 + (i % (U - 1))
                if kind == 0:
                    per.append(applier.make_record(
                        "HIT", row=tgt, amount_q=ONE, dtype=0, attacker=0, indirect=0,
                        expose=ONE // 8, dir_x_q=0, dir_y_q=0, ap_q=0, cause=0))
                elif kind == 1:
                    per.append(applier.make_record("STATUS", row=tgt, status=2,
                                                   magnitude_q=0, duration=3, source=0))
                else:
                    per.append(applier.make_record("PUSH", row=tgt, jx_q=ONE, jy_q=0,
                                                   lever_q=ONE // 4, twist_q16=ONE))
            ts = []
            for _ in range(reps):
                st.upload(rows)
                ring.begin()
                fx.stage_lists([np.array(per)] * n_env)
                cp.cuda.Stream.null.synchronize()
                t = time.perf_counter()
                unit_kernels.resolve(st, fx, dev, ring)
                ts.append((time.perf_counter() - t) * 1e3)
            lines.append(("4c resolve", n_env, k, float(np.median(ts))))
    print("TIMING (median ms, the call incl. its count / book download):")
    for name, n_env, k, ms in lines:
        what = "rows" if name.startswith("4a") else "effects/env"
        print(f"  {name:11s} N={n_env:3d} {what} {k:3d}: {ms:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
