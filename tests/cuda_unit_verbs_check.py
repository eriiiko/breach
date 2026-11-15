"""#29 P6.4a -- THE VERBS GATE: slot 1b's hand verbs on the device unit store
(``Simulation._consume_direct_intents``, the continuous path), the REQUEST
LIST of what slot 1b does that is not a row write, and the DONE kernel
(``MatchRules``) -- each against its Python law at tol 0, rows AND requests.

Runs inside the GPU subprocess (tests/cuda_harness.run_cuda_script; wrapper
tests/test_cuda_unit_verbs.py). The store is ``simulation.unit_store``'s
``DeviceStore`` with its device tables; the list ``simulation.unit_requests.
RequestList`` (the event ring's own record-list class at its width); the
kernels ``simulation.unit_kernels.verbs`` / ``match_done`` (C++
``cpp/src/cuda_unit_verbs.cu`` / ``cuda_unit_done.cu`` over the device unit
library's P6.4a block); the harness ``tests/cuda_unit_lockstep.py``
(``DeviceSlot(..., requests=True)``: the list compared with what the host did
in the slot, in the host's order).

PART TABLES -- every new table row read back from the device equals the
reference object's value (the weapon rows' magazine and stall, the hand
grenade's wind-up / reach / fuse range, the plant row, the charge capacity,
the shove's impulse / cone / cooldown, the body radius, the match limit), per
env of a batch of two maps of different tile size.

PART LIB -- the gates of the device unit library (acts, can_throw, can_plant,
can_reload, can_shove, mag_status, magazines_carried, the HANDLING duration,
shove_cooled, can_act) through a TEST-ONLY probe on SYNTHETIC rows over their
range against verb_gates.py / combat.mag_status / unit_traits._handling; a
kind outside the family refused (bad), padding rows left alone.

PART SCENES -- the SHADOW LOCKSTEP of slot 1b, every tick: rows equal,
requests equal record for record in order, the kernel's coverage == the rows
the host changed, the refused plants == the host's; the hooked match equal to
the same match without the harness. Scenes:
  ship  -- ship_demo's bots match, seed 1 (the bots throw frags; doors), with
           HANDLING and PUSH hooks on two thirds of the marines;
  verbs -- a three-room level with an interior door, a hull door and a dead
           door, driven by scripted intents through the intent facade, in
           which a CENSUS of the host's own decisions counts every branch and
           the gate fails as vacuous when one is 0: each verb acting, each
           refusal reason of each gate (stunned, winding up, planting, none
           carried, full -- never bound, bound, a stall run out --, no spare,
           a stall running, an untracked weapon, the cooldown), untracked
           spares, a stunned reload (the oversight), a wind-up dropped by a
           stun, a death and a conversion during a wind-up and during a plant
           (the drop branch on a dead and a zombie row, a held trigger
           released), the release tick, every grenade kind, a reach clamped
           high and low and the fixed reach, a fuse clamped low and high, a
           plant on a wind-up's release tick, a plant refused at a full charge
           table with two planters on one tick (id order decides), a shove that
           hits, one that misses, two shovers on one target, a shove onto a
           zombie and past a dead unit, a use with a door in reach (twice on
           one tick), none in reach, a dead door in reach, a hull door;
  zero  -- the class wind-up and plant patched to 0 for the scene (both
           sides read the same rows): the pre-G3p same-tick throw and a charge
           laid on its plant's own tick.
PART BATCH -- N = 2 and 8: the verb scene's variants (another door layout,
another seed, extra bystanders, a pre-filled charge table, a tick offset)
beside ship_demo's match: env i == env i alone, rows and requests.
PART PADDING -- padding rows filled with copies of LIVE acting rows (valid
cleared) on the first tick the slot acts and requests: untouched, silent, no
shove lands on them, the valid rows / list / coverage equal the clean run.
PART OVERFLOW -- a list of capacity 1 on a tick with more requests: the first
written (the host's first), the rest counted, the run refused by name.
PART CHAINED -- slot 1b alone, and 1a + 1b + 1b' + 1c together, replacing the
host's slots in a twin match (the list turned into the host's projectiles,
charges and door flips): the per-tick unit hash and field digest equal an
untouched reference's every tick, on ship and the verb scene.
PART DONE -- the done kernel equal to MatchRules.is_terminal / result /
living_teams on every tick of matches that end by elimination (a win), at the
limit with both teams alive (a draw) and at the limit with one team (a win),
and in a batch whose envs end on different ticks.

PROPERTY: slot 1b on the device is the reference on every valid row, integer
for integer, its requests in the reference's order, the shove's impulse
summed onto its target whatever the thread schedule, in a batch as alone; the
done kernel is MatchRules. BREAKS IF a gate term, a HANDLING / PUSH hook, a
clamp, the release's target rounding, the plant's capacity rule or its id
order, the use's door scan, the reload's settle, the shove's target choice,
lever or angular term differs from the law, a dead or turned row keeps what
it held, a request is lost, reordered or misfiled, a padding row or another
env is touched, or the done kernel's team count, limit or outcome differs.

Prints per-part lines and ``UNIT_VERBS_RESULT: PASS`` / ``FAIL``; exits 0 / 1.
``bench()`` prints the TIMING table.
"""
from __future__ import annotations

import dataclasses
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
                                ShadowLockstep, capture, gather)
from simulation import (unit_kernels, unit_requests, unit_store,  # noqa: E402
                        unit_tables, unit_traits)

FAILS: list = []
ONE = unit_traits.ONE
PI_Q16 = 205887
SHIP_TICKS = int(os.environ.get("UNIT_VERBS_SHIP_TICKS", "500"))
VERB_TICKS = int(os.environ.get("UNIT_VERBS_VERB_TICKS", "200"))
BATCH_TICKS = int(os.environ.get("UNIT_VERBS_BATCH_TICKS", "120"))
BATCH_NS = [int(x) for x in os.environ.get("UNIT_VERBS_BATCH_NS", "2,8").split(",")]
DONE_TICKS = int(os.environ.get("UNIT_VERBS_DONE_TICKS", "2400"))
SCENES = os.environ.get("UNIT_VERBS_SCENES", "verbs,zero,ship").split(",")


def fail(msg):
    FAILS.append(msg)
    print("FAIL:", msg)


# ---------------------------------------------------------------------------
# The slots
# ---------------------------------------------------------------------------
def _run_verbs(st, inp, req):
    return unit_kernels.verbs(st, req, inp.tick, inp.tables["door_spans"],
                              inp.scalars["charges_live"])


SLOT_1B = DeviceSlot(label="1b intents", run=_run_verbs,
                     inputs=("tick", "door_spans", "charges_live"), requests=True)
SLOT_1A = DeviceSlot(label="1a integrate + turn", run=unit_kernels.integrate_turn)
SLOT_1BP = DeviceSlot(label="1b' orders",
                      run=lambda st, inp: unit_kernels.apply_orders(
                          st, inp.tables["order_requests"]),
                      inputs=("order_requests",))
SLOT_1C = DeviceSlot(label="1c exposure decay", run=unit_kernels.exposure_decay)


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
# The scenes
# ---------------------------------------------------------------------------
HOOK_SETS = (None, {"HANDLING": ONE * 2, "PUSH": ONE * 3 // 2},
             {"HANDLING": ONE * 3 // 4, "PUSH": ONE // 2})


def _hook(sim, units):
    for i, u in enumerate(units):
        h = HOOK_SETS[i % 3]
        if h is not None:
            unit_traits.compile_unit(sim, u, h)


class ShipDriver:
    """ship_demo's bots match (MatchSession), HANDLING / PUSH hooks on two
    thirds of the marines."""

    def __init__(self, seed=1, hooks=True):
        from simulation.match_session import MatchSession
        level = level_loader.load("ship_demo", str(_ROOT / "levels"))
        self.session = MatchSession(level, seed=seed, breach_physics=bp)
        self.sim = self.session.sim
        if hooks:
            _hook(self.sim, [u for u in self.sim.units if not u.is_zombie])

    def tick(self):
        self.session.tick()

    def is_terminal(self):
        return self.session.is_terminal()


def _door(eid, ordinal, x, y):
    from level_loader import EntityInstance
    from simulation.entities import door as door_mod
    fields = {f.name: f.default for f in door_mod.door.FIELDS}
    fields.update(x=x, y=y, orientation="v", length_m=3.0, initial_state="closed")
    return EntityInstance(id=eid, class_name="door", ordinal=ordinal, fields=fields)


def _verb_level(variant):
    """Three rooms: A (x 1..31) | door 0 at x 32 | B (x 33..63) | the dead
    door at x 64 | C (x 65..93) | the hull door at x 94 | space. Odd
    variants move door 0 (another layout)."""
    from level_loader import LevelData
    from simulation.materials import MAT_AIR, MAT_HULL
    h, w = 48, 96
    tm = np.full((h, w), MAT_HULL, dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = MAT_AIR
    dy0 = 4 + 8 * (variant % 2)
    tm[1:h - 1, 32] = MAT_HULL
    tm[dy0:dy0 + 3, 32] = MAT_AIR
    tm[1:h - 1, 64] = MAT_HULL
    tm[40:43, 64] = MAT_AIR
    tm[:, 95] = -1                       # space past the east hull
    tm[1:h - 1, 94] = MAT_HULL
    tm[4:7, 94] = MAT_AIR
    ents = [_door("d0", 0, 32, dy0), _door("hull", 1, 94, 4), _door("dead", 2, 64, 40)]
    return LevelData(name=f"p64a_verbs{variant % 2}", version="3", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."),
                     entities=ents), dy0


EAST, WEST, SOUTH, NORTH = 0, PI_Q16, -PI_Q16 // 2, PI_Q16 // 2


class VerbDriver:
    """The verb scene (see the module docstring): every branch of slot 1b by
    a scripted intent schedule through the intent facade, fixtures written
    between ticks (a stun, a death, a conversion, a magazine state, the charge
    table topped up). ``variant`` > 0: another seed, extra bystanders, odd
    variants another door layout and a pre-filled charge table, and a tick
    offset (``variant`` ticks stepped before the scene). ``zero``: the
    class wind-up and plant are 0 (the caller patches the rows)."""

    def __init__(self, variant=0, hooks=True, zero=False):
        from simulation import Simulation, combat, charges
        from simulation.ruleset import ContinuousRealtime
        from simulation.status import STUNNED, apply_status  # noqa: F401
        self.variant, self.zero, self.k = variant, zero, 0
        level, self.dy0 = _verb_level(variant)
        sim = Simulation(level, seed=20261115 + variant, breach_physics=bp,
                         enable_recorder=False, ruleset=ContinuousRealtime())
        sim.set_paused(False)
        self.sim = sim
        self.who = {}
        self.hooks = hooks
        W = sim.weapons_tables.weapons.by_name
        self.k5 = W["k5_carbine"]
        if zero:
            self._add("t0", 34, 2, facing=SOUTH)
            self._add("p0", 38, 2, facing=SOUTH)
            self._add("p1", 42, 2, facing=SOUTH)
        else:
            self._setup()
        for i in range(variant % 3):                     # bystanders
            self._add(f"B{i}", 70 + 4 * i, 30, facing=NORTH)
        if variant % 2:                                  # a pre-filled charge table
            self._top_up(10)
        for _ in range(variant):                         # a tick offset
            sim.step()
            sim.set_paused(False)

    # -- helpers -----------------------------------------------------------
    def _add(self, name, x, y, team=0, facing=EAST, hooks=None, hp=10000.0):
        from simulation.turning import place
        from simulation.unit import Unit
        u = Unit(name, x=float(x), y=float(y), team=team)
        self.sim.add_unit(u, hooks=hooks if self.hooks else None)
        place(u, facing)
        u.current_hp = float(hp)
        self.who[name] = u
        return u

    def _kill(self, u, zombie=False):
        from simulation.ai_zombie import convert_marines_to_zombies
        from simulation.damage import DamagePacket, apply_packet
        apply_packet(u, DamagePacket(amount=1.0e6, dtype=0, attacker_id=-1), [],
                     source="melee", mark_killed_by_zombie=zombie)
        if zombie:
            convert_marines_to_zombies([u])

    def _top_up(self, live):
        """Dummy charges far in the future until ``live`` are live."""
        from simulation import charges
        while unit_requests.charges_live(self.sim) < live:
            charges.place(self.sim, charges.PlantedCharge(
                80, 44, "demo_c4", -1, det_tick=10 ** 9, approach=None))

    def _setup(self):
        from simulation import combat
        from simulation.status import STUNNED, apply_status  # noqa: F401
        a = self._add
        H1, H2 = HOOK_SETS[1], HOOK_SETS[2]
        dy0 = self.dy0
        # room B: the throwers (south into the empty half of B)
        for i in range(6):
            a(f"t{i}", 34 + 4 * i, 2, facing=SOUTH, hooks=(H1, None, H2)[i % 3])
        for k in (2, 3, 5):
            self.who["t2"].grenade_counts[k] = 2
            self.who["t3"].grenade_counts[k] = 2
            self.who["t5"].grenade_counts[k] = 2
        a("u2", 61, 40, facing=EAST)                      # beside the dead door
        self.sim._doors[2].alive = False                  # (a fixture: its span stays shut)
        # room C: the hull door's user
        a("u3", 91, 4, facing=EAST)
        # room A
        a("w0", 26, 8, facing=WEST)
        for i in range(4):                                # death / conversion
            a(f"d{i}", 2 + 4 * i, 8, facing=NORTH)
        self.who["d2"].grenade_counts[2] = 1
        for i in range(7):                                # planters
            a(f"p{i}", 2 + 4 * i, 14, facing=NORTH, hooks=(None, H2, None)[i % 3])
        a("u0", 29, dy0, facing=EAST)                     # beside door 0
        a("u0b", 29, dy0 + 3, facing=EAST)
        a("u1", 10, 19, facing=EAST)                      # no door in reach
        for i in range(7):                                # reloaders
            a(f"r{i}", 2 + 4 * i, 24, facing=NORTH, hooks=(None, H2, H1)[i % 3])
        a("r7", 2, 28, facing=NORTH)
        k5 = self.k5
        for n in ("r1", "r2", "r3", "r4", "r6", "r7"):
            combat.mag_bind(self.who[n], k5)
        self.who["r2"].current_mag, self.who["r2"].spare_mags = 10, 2
        self.who["r3"].current_mag, self.who["r3"].spare_mags = 10, 0
        self.who["r4"].current_mag, self.who["r4"].spare_mags = 10, -1
        self.who["r5"].weapon_id = "lance_3"              # mag_size 0: untracked
        self.who["r6"].current_mag, self.who["r6"].spare_mags = 0, 1
        self.who["r6"].reload_done_tick = 0               # a stall that has run out
        self.who["r7"].current_mag, self.who["r7"].spare_mags = 10, 2
        # the shove rows (room A, south)
        a("s0", 2, 34, facing=EAST, hooks=H1)
        a("s1", 5, 34, team=1, facing=WEST)
        a("s2", 14, 34, facing=EAST)
        a("s3", 20, 34, facing=EAST)
        a("s4", 2, 40, facing=EAST, hooks=H1)
        a("T", 5, 40, team=1, facing=NORTH)
        a("s5", 8, 40, facing=WEST, hooks=H2)
        a("s6", 14, 40, facing=EAST)
        z = a("Z", 17, 40, facing=WEST)
        self._kill(z, zombie=True)                        # a zombie in reach
        a("s7", 22, 40, facing=EAST)
        dead = a("D", 25, 40, facing=WEST)
        self._kill(dead)                                  # dead, straight ahead
        a("L", 24, 43, facing=NORTH)                      # alive, inside the cone

    # -- the schedule ------------------------------------------------------
    def _before(self):
        from simulation.status import STUNNED, apply_status
        sim, w, k = self.sim, self.who, self.k
        if self.zero:
            if k in (1, 12):
                sim.throw_grenade_intent(w["t0"].id, 0, ONE, 240, 8 * ONE, 0)
            if k in (2,):
                sim.plant_charge_intent(w["p0"].id, 10 ** 6)
            if k in (2, 9):
                sim.plant_charge_intent(w["p1"].id, 10 ** 6)
            if k == 8:
                w["p1"].has_explosive = 1
            return
        thr = sim.throw_grenade_intent
        plant = sim.plant_charge_intent
        if k == 1:
            thr(w["t0"].id, 0, ONE, -5, None, 0)                 # fixed reach, fuse low
            apply_status(w["s3"], STUNNED, 0, 3)
            apply_status(w["r7"], STUNNED, 0, 3)
        if k == 2:
            thr(w["t0"].id, 0, ONE, 100, 6 * ONE, 0)            # winding up
            thr(w["t1"].id, 0, ONE, 100000, 100 * ONE, 1)        # both clamps high
            plant(w["p0"].id, 10 ** 6)
            for n in ("s0", "s2", "s3", "s4", "s5", "s6", "s7"):
                sim.shove_intent(w[n].id)
            for n in ("u0", "u1", "u2", "u3"):
                sim.use_intent(w[n].id)
            for i in range(8):
                sim.reload_intent(w[f"r{i}"].id)
            apply_status(w["t2"], STUNNED, 0, 3)
            apply_status(w["p2"], STUNNED, 0, 3)
            w["p1"].has_explosive = 0
        if k == 3:
            sim.shove_intent(w["s0"].id)                         # cooldown
            sim.shove_intent(w["s2"].id)                         # a miss again
            plant(w["p0"].id, 10 ** 6)                           # planting
            thr(w["p0"].id, 0, ONE, 100, 6 * ONE, 0)             # planting
            thr(w["t2"].id, 0, ONE, 100, 6 * ONE, 2)             # stunned
            thr(w["t3"].id, 0, ONE, 100, 12 * ONE, 3)
            plant(w["p1"].id, 10 ** 6)                           # none carried
            plant(w["p2"].id, 10 ** 6)                           # stunned
            sim.reload_intent(w["r2"].id)                        # a stall running
        if k == 4:
            w["t4"].grenade_counts[4] = 0
            thr(w["t4"].id, 0, ONE, 100, 6 * ONE, 4)             # none carried
            plant(w["p3"].id, 10 ** 6)
            thr(w["w0"].id, -ONE, 0, 100, 2 * ONE, 1)
        if k == 5:
            plant(w["w0"].id, 10 ** 6)                           # winding up
            apply_status(w["t3"], STUNNED, 0, 2)                 # the wind-up drops
        if k == 6:
            apply_status(w["p3"], STUNNED, 0, 2)                 # the plant drops
            w["t4"].grenade_counts[4] = 1
            thr(w["t4"].id, 0, ONE, 200, -5 * ONE, 4)            # reach low
            thr(w["t5"].id, 0, ONE, 100, 10 * ONE, 5)
            sim.set_trigger(w["d0"].id, True)
            thr(w["d0"].id, 0, ONE, 100, 6 * ONE, 0)
            plant(w["d1"].id, 10 ** 6)
            thr(w["d2"].id, 0, ONE, 100, 6 * ONE, 2)
            plant(w["d3"].id, 10 ** 6)
        if k == 9:
            self._kill(w["d0"])
            self._kill(w["d1"])
        if k == 10:
            self._kill(w["d2"], zombie=True)
            self._kill(w["d3"], zombie=True)
        if w["w0"].throw_release_tick == sim.tick and w["w0"].plant_done_tick < 0:
            plant(w["w0"].id, 10 ** 6)                           # on the release tick
        if k == 20:
            thr(w["t3"].id, 0, ONE, 100, 12 * ONE, 3)
            thr(w["t2"].id, 0, ONE, 100, 14 * ONE, 2)
            thr(w["t0"].id, 0, ONE, 100, 20 * ONE, 0)
            plant(w["p4"].id, 10 ** 6)
            plant(w["p5"].id, 10 ** 6)
        p4 = w["p4"]
        if p4.plant_done_tick >= 0 and p4.plant_done_tick == sim.tick:
            from simulation.unit_schema import CHARGE_CAPACITY
            self._top_up(CHARGE_CAPACITY - 1)                    # one place left
        if k == 50:
            plant(w["p6"].id, 10 ** 6)                           # the table is full
        if k == 60:
            sim.use_intent(w["u0"].id)                           # two on one door
            sim.use_intent(w["u0b"].id)
            sim.use_intent(w["u3"].id)
        if k == 70:
            sim.shove_intent(w["s0"].id)                         # cooled again
            sim.shove_intent(w["s4"].id)
            sim.shove_intent(w["s5"].id)
            sim.reload_intent(w["r4"].id)

    def tick(self):
        self._before()
        self.sim.step()
        self.sim.set_paused(False)
        self.k += 1

    def is_terminal(self):
        return False


# ---------------------------------------------------------------------------
# The census: the host's own decisions at slot 1b, by branch
# ---------------------------------------------------------------------------
class Census:
    """Wraps ``Simulation._consume_direct_intents``: before it runs, each
    unit's slot-1b state; after, every branch the reference took, classified
    by the reference's own gate terms (verb_gates, combat.mag_status,
    unit_impulse.shove_target), and the render-only events of the slot (the
    shove's target). Reads only."""

    def __init__(self):
        self.c = Counter()
        self._orig = None

    def __enter__(self):
        from simulation.simulation import Simulation
        orig = Simulation._consume_direct_intents
        self._orig = orig
        census = self

        def wrapped(sim, __o=orig):
            pre = census._snap(sim)
            n0 = len(sim.tick_events)
            out = __o(sim)
            census._classify(sim, pre, sim.tick_events[n0:])
            return out
        wrapped.__wrapped__ = orig
        Simulation._consume_direct_intents = wrapped
        return self

    def __exit__(self, *exc):
        from simulation.simulation import Simulation
        Simulation._consume_direct_intents = self._orig
        return False

    @staticmethod
    def _snap(sim):
        from simulation.status import composed_flags
        out = {}
        for u in sim.units:
            out[u.id] = SimpleNamespace(
                alive=u.alive, zombie=u.is_zombie, trigger=bool(u.live_trigger),
                pt=u.pending_throw, trt=int(u.throw_release_tick),
                pdt=int(u.plant_done_tick), pp=u.pending_plant,
                use=bool(u.pending_use), reload=bool(u.pending_reload),
                shove=bool(u.pending_shove), counts=list(u.grenade_counts),
                expl=int(u.has_explosive), charge=u.charge_ammo,
                act=composed_flags(u).can_act, kind=int(u.throw_kind),
                fuse=int(u.throw_fuse_ticks), reach=int(u.throw_reach_q),
                last_shove=int(u.last_shove_tick),
                mag=(int(u.current_mag), int(u.spare_mags), int(u.reload_done_tick)))
        return out

    def _release(self, sim, kind, reach, fuse):
        from simulation.weapons import fuse_ticks_from_seconds
        hand = sim.weapons_tables.weapons.by_name["hand_grenade"]
        c = self.c
        c[f"released kind {kind}"] += 1
        if reach is None:
            c["reach: the fixed 6 tiles"] += 1
        elif reach > int(hand.max_throw_range) << 16:
            c["reach: clamped high"] += 1
        elif reach < 0:
            c["reach: clamped low"] += 1
        lo = fuse_ticks_from_seconds(hand.fuse_min_seconds, sim._tps)
        hi = fuse_ticks_from_seconds(hand.fuse_max_seconds, sim._tps)
        if fuse < lo:
            c["fuse: clamped low"] += 1
        if fuse > hi:
            c["fuse: clamped high"] += 1

    def _classify(self, sim, pre, events):
        from simulation import combat, unit_impulse, unit_traits as ut, verb_gates
        from simulation.unit_schema import THROW_NO_REACH
        c, tick = self.c, sim.tick
        W = sim.weapons_tables.weapons.by_name
        for u in sim.units:
            p = pre[u.id]
            if not p.alive or p.zombie:
                tag = "a dead row" if not p.alive else "a zombie row"
                for what, held in (("a held trigger", p.trigger),
                                   ("a wind-up", p.trt >= 0),
                                   ("a plant", p.pdt >= 0)):
                    if held:
                        c[f"drop: {tag} releases {what}"] += 1
                continue
            # THROW (the gate reads the unit's state at its turn's start)
            if p.pt is not None:
                r = None
                if p.trt >= 0:
                    r = "winding up"
                elif p.pdt >= 0:
                    r = "planting"
                elif p.counts[p.pt.kind] <= 0:
                    r = "none carried"
                elif not p.act:
                    r = "stunned"
                if r:
                    c[f"throw refused: {r}"] += 1
                else:
                    c["throw acts"] += 1
                    if ut.throw_windup_ticks(u, W) == 0:
                        c["a 0-tick wind-up releases at once"] += 1
                        self._release(sim, p.pt.kind, p.pt.reach_q, p.pt.fuse_ticks)
            if p.trt >= 0:
                if not p.act:
                    c["a wind-up dropped (cannot act)"] += 1
                elif tick >= p.trt:
                    c["a wind-up released on its tick"] += 1
                    self._release(sim, p.kind,
                                  None if p.reach == THROW_NO_REACH else p.reach, p.fuse)
            # PLANT (its gate reads throw_release_tick after the throw's part)
            if p.pp is not None:
                r = None
                if p.pdt >= 0:
                    r = "planting"
                elif u.throw_release_tick > tick:
                    r = "winding up"
                elif p.expl <= 0 or not p.charge:
                    r = "none carried"
                elif not p.act:
                    r = "stunned"
                if r:
                    c[f"plant refused: {r}"] += 1
                else:
                    c["plant acts"] += 1
                    if p.trt == tick:
                        c["a plant on a wind-up's release tick"] += 1
                    if ut.plant_ticks(u, W) == 0:
                        c["a 0-tick plant lays at once"] += 1
            if p.pdt >= 0:
                if not p.act:
                    c["a plant dropped (cannot act)"] += 1
                elif tick >= p.pdt:
                    c["a plant laid on its last tick"] += 1
            # USE
            if p.use:
                door = verb_gates.use_door(sim, u)
                if door is None:
                    c["use: no door in reach"] += 1
                else:
                    c["use: a door flips"] += 1
                    if sim._doors.index(door) == 1:
                        c["use: the hull door"] += 1
                if any((not d.alive) and verb_gates._reach_key(u, d) is not None
                       for d in sim._doors):
                    c["use: a dead door in reach is skipped"] += 1
            # RELOAD
            if p.reload:
                w = W.get(u.weapon_id)
                if w is None or w.mag_size <= 0:
                    c["reload refused: an untracked weapon"] += 1
                else:
                    # mag_status on the magazine as it stood at the slot's
                    # start (nothing before the reload in 1b touches it)
                    pm = p.mag
                    rounds, spares, reloading = combat.mag_status(
                        _MagView(u, pre_mag=pm), w, tick)
                    if reloading:
                        c["reload refused: a stall running"] += 1
                    elif rounds >= w.mag_size:
                        if pm[0] == -1:
                            c["reload refused: full (never bound)"] += 1
                        elif pm[2] >= 0:
                            c["reload refused: full (its stall ran out)"] += 1
                        else:
                            c["reload refused: full (bound)"] += 1
                    elif spares == 0:
                        c["reload refused: no spare"] += 1
                    else:
                        c["reload acts" + (" (untracked spares)" if spares is None
                                           else " (a counted spare)")] += 1
                        if not p.act:
                            c["reload acts while stunned (the oversight)"] += 1
            # SHOVE
            if p.shove:
                if not p.act:
                    c["shove refused: stunned"] += 1
                elif not (p.last_shove < 0 or tick - p.last_shove
                          >= sim.impulse_table.shove_cooldown_ticks):
                    c["shove refused: the cooldown"] += 1
                else:
                    t = unit_impulse.shove_target(sim.impulse_table, sim.units, u)
                    if t is None:
                        c["shove: a miss"] += 1
                    else:
                        c["shove: a hit"] += 1
                        if t.is_zombie:
                            c["shove: onto a zombie"] += 1
                        if self._dead_nearer(sim, u, t):
                            c["shove: past a dead unit"] += 1
        hits = Counter(e.target_id for e in events if type(e).__name__ == "ShoveEvent")
        if any(n >= 2 for n in hits.values()):
            c["shove: two shovers on one target"] += 1

    @staticmethod
    def _dead_nearer(sim, u, t):
        """A dead unit adjacent and in the cone, nearer the facing than ``t``
        (what shove_target skipped)."""
        from simulation import turning, unit_fixed, unit_impulse
        tab = sim.impulse_table
        cx, cy = unit_impulse.centre_q(u)

        def key(o):
            ox, oy = unit_impulse.centre_q(o)
            dx, dy = ox - cx, oy - cy
            reach = ((int(u.footprint) + int(o.footprint)) << 15) + ONE
            if dx * dx + dy * dy > reach * reach or (dx == 0 and dy == 0):
                return None
            off = abs(turning.wrap_q16(unit_fixed.atan2_q16(-dy, dx) - u.facing_q))
            return None if off > tab.shove_cone_q else (off, int(o.id))
        kt = key(t)
        return any((not o.alive) and o is not u and key(o) is not None
                   and key(o) < kt for o in sim.units)


class _MagView:
    """The magazine columns as they stood at the slot's start (``pre_mag``:
    (current_mag, spare_mags, reload_done_tick)) over the unit's kit."""

    def __init__(self, u, pre_mag):
        self._u = u
        self.current_mag, self.spare_mags, self.reload_done_tick = pre_mag

    def __getattr__(self, name):
        return getattr(self._u, name)


VerbCensus = Census


#: The branches the verb scene must reach (a 0 fails as vacuous).
VERB_BRANCHES = (
    "throw acts", "throw refused: winding up", "throw refused: planting",
    "throw refused: none carried", "throw refused: stunned",
    "a wind-up dropped (cannot act)", "a wind-up released on its tick",
    *(f"released kind {k}" for k in range(6)),
    "reach: the fixed 6 tiles", "reach: clamped high", "reach: clamped low",
    "fuse: clamped low", "fuse: clamped high",
    "plant acts", "plant refused: planting", "plant refused: winding up",
    "plant refused: none carried", "plant refused: stunned",
    "a plant on a wind-up's release tick", "a plant dropped (cannot act)",
    "a plant laid on its last tick",
    "drop: a dead row releases a held trigger", "drop: a dead row releases a wind-up",
    "drop: a dead row releases a plant", "drop: a zombie row releases a wind-up",
    "drop: a zombie row releases a plant",
    "use: a door flips", "use: no door in reach", "use: the hull door",
    "use: a dead door in reach is skipped",
    "reload acts (a counted spare)", "reload acts (untracked spares)",
    "reload acts while stunned (the oversight)", "reload refused: full (never bound)",
    "reload refused: full (bound)", "reload refused: full (its stall ran out)",
    "reload refused: no spare", "reload refused: a stall running",
    "reload refused: an untracked weapon",
    "shove: a hit", "shove: a miss", "shove refused: the cooldown",
    "shove refused: stunned", "shove: two shovers on one target",
    "shove: onto a zombie", "shove: past a dead unit",
)
ZERO_BRANCHES = ("a 0-tick wind-up releases at once", "a 0-tick plant lays at once",
                 "throw acts", "plant acts")


# ---------------------------------------------------------------------------
# The lockstep runner
# ---------------------------------------------------------------------------
def run_lockstep(name, drivers, slots, ticks, census=None, n_units=None):
    """Shadow ``slots`` on ``drivers`` for ``ticks`` ticks: every tick each
    slot equal (rows and requests), its coverage == the host's changed rows,
    its refused plants == the host's. Returns (env-ticks with coverage,
    requests by kind, refused plants)."""
    t0 = time.perf_counter()
    cov = Counter()
    kinds = Counter()
    refused = 0
    n = 0
    ok = True
    ctx = census if census is not None else _Null()
    with ctx, ShadowLockstep(drivers, list(slots), n_units=n_units) as ls:
        for _ in range(ticks):
            if any(getattr(d, "is_terminal", lambda: False)() for d in drivers):
                break
            reps = ls.tick()
            if not isinstance(reps, dict):
                reps = {slots[0].label: reps}
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
                if rep.requests is not None:
                    host_ref = sum(1 for e in rep.host_requests for r in e
                                   if r[0] == unit_requests.KIND_PLANT and not r[12])
                    dev_ref = int(ls.store.last_refused.sum())
                    if host_ref != dev_ref:
                        fail(f"{name} tick {rep.tick}: refused plants device "
                             f"{dev_ref} != host {host_ref}")
                        ok = False
                        break
                    refused += dev_ref
                    for e in rep.requests:
                        for r in e:
                            kinds[unit_requests.KINDS[r[0] - 1]] += 1
            if not ok:
                break
    total = n * len(drivers)
    print(f"{name}: N={len(drivers)} units per env "
          f"{[len(d.sim.units) for d in drivers]} (U {ls.U}), {n} ticks"
          f"{', env i == env i alone' if len(drivers) > 1 and ok else ''}; "
          + ", ".join(f"[{lab}] coverage on {100 * cov[lab] / max(1, total):.1f} % "
                      f"of env-ticks" for lab in (s.label for s in slots))
          + f"; requests {dict(kinds)}, {refused} plants refused at the capacity"
          + f" ({time.perf_counter() - t0:.0f} s)")
    return cov, kinds, refused


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# ---------------------------------------------------------------------------
# PART TABLES
# ---------------------------------------------------------------------------
def table_reference(sim) -> dict:
    """The P6.4a device-table rows' values read from the reference objects
    themselves (what ``unit_tables.values`` must equal): also merged into
    tests/cuda_unit_kernels_check.py's table check, which refuses a row with
    no reference value."""
    from simulation.ruleset import MatchRules
    from simulation.unit_schema import CHARGE_CAPACITY
    from simulation.weapons import PLANT_WEAPON, fuse_ticks_from_seconds, get_tables
    W = get_tables().weapons.by_name
    hand = W["hand_grenade"]
    it = sim.impulse_table
    rules = getattr(sim, "ruleset", None)
    return {
        "weapon_mag_size": tuple(int(w.mag_size) for w in W.values()),
        "weapon_reload_ticks": tuple(int(w.reload_ticks) for w in W.values()),
        "hand_windup_class_ticks": int(hand.throw_windup_ticks),
        "hand_max_reach_q": int(hand.max_throw_range) * 65536,
        "hand_fuse_lo_ticks": fuse_ticks_from_seconds(hand.fuse_min_seconds, sim._tps),
        "hand_fuse_hi_ticks": fuse_ticks_from_seconds(hand.fuse_max_seconds, sim._tps),
        "plant_class_ticks": int(W[PLANT_WEAPON].plant_ticks),
        "charge_capacity": CHARGE_CAPACITY,
        "shove_q": int(it.shove_q), "shove_cone_q": int(it.shove_cone_q),
        "shove_cooldown_ticks": int(it.shove_cooldown_ticks),
        "impulse_r_body_q": int(it.r_body_q),
        "match_limit_ticks": (int(rules.limit_ticks(sim))
                              if isinstance(rules, MatchRules) else 0),
    }


def part_tables():
    a = VerbDriver(hooks=False)
    b = ShipDriver(hooks=False)
    sims = [a.sim, b.sim]
    tab = unit_tables.values_batch(sims)
    st = unit_store.DeviceStore(2, 4, tables=tab)
    got = st.tables.download()
    if not np.array_equal(got, tab):
        fail("tables: the device tables differ from the uploaded ones")
    n = 0
    as_t = lambda x: tuple(x) if isinstance(x, (tuple, list)) else (x,)  # noqa: E731
    for e, sim in enumerate(sims):
        for k, v in table_reference(sim).items():
            dv = unit_tables.read(got[e], k)
            n += 1
            if as_t(dv) != as_t(v):
                fail(f"tables env {e} row {k}: device {dv} != reference {v}")
    if unit_tables.read(got[0], "shove_q") == unit_tables.read(got[1], "shove_q"):
        fail("tables: the two maps' shove impulses are equal (one tile size?)")
    if unit_tables.read(got[1], "match_limit_ticks") <= 0:
        fail("tables: the match's limit read as none")
    req = unit_requests.RequestList(1, 1)
    print(f"tables: {n} P6.4a values over 2 envs equal the reference objects (T "
          f"{unit_tables.layout().width}); the request list's compiled layout is "
          f"unit_requests' (F {req.shape[2]}, row cap {unit_requests.ROW_CAP})")


# ---------------------------------------------------------------------------
# PART LIB
# ---------------------------------------------------------------------------
def part_lib(trials=int(os.environ.get("UNIT_VERBS_LIB_TRIALS", "6"))):
    """The device gates on SYNTHETIC rows: every column a gate reads drawn over
    its range (life and zombie states, a stun, the wind-up and plant ticks
    around the tick, the grenade counts, the charge, the weapon -- none, the
    loadout's, a swapped tracked one, an untracked one --, the magazine
    columns, the loadout's count and its weapon, the last shove, the HANDLING
    hook), two envs of different tables, against the Python gates on the
    decoded row; padding rows left alone; a kind outside the family bad."""
    from simulation import combat, unit_traits as ut, verb_gates
    from simulation.status import STUNNED, apply_status, composed_flags
    from simulation.unit_schema import table_row
    rng = np.random.default_rng(6404)
    a, b = VerbDriver(hooks=False), ShipDriver(hooks=False)
    sims = [a.sim, b.sim]
    lay = unit_store.layout()
    col = lambda n: lay.column(n).offset  # noqa: E731
    base_units = [u for u in a.sim.units if u.alive and not u.is_zombie][:6]
    stunned = base_units[1]
    apply_status(stunned, STUNNED, 0, 5)
    n, U = 2, 32
    tabs = unit_tables.values_batch(sims)
    W = sims[0].weapons_tables.weapons.by_name
    wrows = [table_row("weapons", x) for x in ("k5_carbine", "lance_3", "gl6_revolver",
                                                "p12_whisper")]
    checks = Counter()
    for trial in range(trials):
        tick = int(rng.choice([0, 5, 100, 10 ** 6]))
        kind = int(rng.integers(0, 6))
        base = int(rng.choice([0, 1, 7, 24, 120, 1 << 16]))
        rows = np.zeros((n, U, lay.width), dtype=np.int64)
        for e in range(n):
            for r in range(U):
                src = base_units[r % len(base_units)]
                row = unit_store.pack([src])[0]
                row[col("id")] = r + 1
                row[col("life_state")] = int(rng.random() < 0.15)
                row[col("is_zombie")] = int(rng.random() < 0.15)
                row[col("throw_release_tick")] = int(rng.choice(
                    [-1, -1, tick - 1, tick, tick + 3]))
                row[col("plant_done_tick")] = int(rng.choice([-1, -1, tick - 1, tick, tick + 3]))
                gc = lay.column("grenade_counts")
                row[gc.offset:gc.offset + gc.width] = rng.integers(0, 3, gc.width)
                row[col("has_explosive")] = int(rng.integers(0, 3))
                row[col("charge_ammo")] = int(rng.choice([-1, table_row("ammo", "demo_c4")]))
                row[col("weapon_id")] = int(rng.choice([-1] + wrows))
                row[col("magazines")] = int(rng.choice([0, 1, 3]))
                row[col("magazines_weapon")] = int(rng.choice([-1, wrows[0], wrows[2]]))
                row[col("current_mag")] = int(rng.choice([-1, 0, 10, 30, 6]))
                row[col("spare_mags")] = int(rng.choice([-1, 0, 2]))
                row[col("reload_done_tick")] = int(rng.choice([-1, tick - 1, tick, tick + 5]))
                cd = int(sims[e].impulse_table.shove_cooldown_ticks)
                row[col("last_shove_tick")] = int(rng.choice([-1, tick - cd, tick - cd + 1, tick]))
                tr = lay.column("traits")
                row[tr.offset + ut.HANDLING] = int(rng.choice(
                    [ONE, ONE // 8, ONE * 3 // 4, 2 * ONE, 8 * ONE,
                     int(rng.integers(ONE // 8, 8 * ONE))]))
                row[lay.valid] = int(rng.random() < 0.9)
                if not row[lay.valid]:
                    row[:] = rng.integers(-1000, 1000, lay.width)
                    row[lay.valid] = 0
                rows[e, r] = row
        st = unit_store.DeviceStore(n, U, tables=tabs)
        st.upload(rows)
        out = unit_kernels.verbs_probe(st, kind, tick, base)
        got = st.download()
        if not np.array_equal(got, rows):
            fail(f"lib trial {trial}: the probe wrote a row")
        for e in range(n):
            fake = SimpleNamespace(tick=tick, weapons_tables=sims[e].weapons_tables,
                                   impulse_table=sims[e].impulse_table)
            for r in range(U):
                o = out[e, r]
                if not rows[e, r, lay.valid]:
                    if o.any():
                        fail(f"lib trial {trial}: a padding row reported a result")
                    continue
                u = unit_store.blank_units(1)[0]
                unit_store.unpack(rows[e, r:r + 1], [u])
                w = W.get(u.weapon_id) if u.weapon_id else None
                ms = (combat.mag_status(u, w, tick) if w is not None and w.mag_size > 0
                      else None)
                want = [
                    int(verb_gates._acts(u)),
                    int(verb_gates.can_throw(fake, u, kind)),
                    int(verb_gates.can_plant(fake, u)),
                    int(verb_gates.can_reload(fake, u)),
                    int(verb_gates.can_shove(fake, u)),
                    int(ms is not None),
                    0 if ms is None else int(ms[0]),
                    -1 if ms is None or ms[1] is None else int(ms[1]),
                    0 if ms is None else int(ms[2]),
                    int(combat.magazines_carried(u)),
                    int(ut._handling(u, base)),
                    int(__import__("simulation.unit_impulse", fromlist=["x"]).shove_cooled(
                        fake.impulse_table, u, tick)),
                    int(composed_flags(u).can_act),
                ]
                dev = [int(x) for x in o[:13]]
                if dev != want or int(o[15]):
                    fail(f"lib trial {trial} env {e} row {r}: device {dev} (bad "
                         f"{int(o[15])}) != Python {want}")
                    return
                for i, name in enumerate(("acts", "throw", "plant", "reload", "shove")):
                    checks[(name, want[i])] += 1
                checks[("tracked", want[5])] += 1
                checks[("reloading", want[8])] += 1
                checks[("untracked spares", int(want[5] and want[7] == -1))] += 1
    # a kind outside the family: bad, as the Python refuses it by name
    st = unit_store.DeviceStore(1, 2, tables=tabs[:1])
    st.upload(unit_store.pack(base_units[:2], 2)[None])
    out = unit_kernels.verbs_probe(st, 6, 0, 1)
    if not out[0, :, 15].all():
        fail("lib: a throw kind outside the family was not refused (bad)")
    try:
        verb_gates.can_throw(SimpleNamespace(tick=0, weapons_tables=sims[0].weapons_tables),
                             base_units[0], 6)
        fail("lib: the Python gate accepted kind 6")
    except ValueError:
        pass
    print(f"lib: {sum(checks.values()) // 8} synthetic rows over {trials} trials equal "
          f"the Python gates (both answers of every gate: "
          f"{sorted((k, v) for k, v in checks.items())}); padding rows untouched; a "
          f"kind outside the family bad")
    for name in ("acts", "throw", "plant", "reload", "shove", "tracked", "reloading",
                 "untracked spares"):
        if not (checks[(name, 0)] and checks[(name, 1)]):
            fail(f"lib: the gate {name!r} never gave both answers (vacuous)")


# ---------------------------------------------------------------------------
# PART SCENES / BATCH
# ---------------------------------------------------------------------------
def _harmless(make, ticks, label):
    """The hooked match (the harness around slot 1b) equals the same match
    without it, every tick (per-tick unit hash and field digest)."""
    a, b = make(), make()
    with ShadowLockstep([a], [SLOT_1B]) as ls:
        for t in range(ticks):
            ls.tick()
            b.tick()
            if not _same(a.sim, b.sim):
                fail(f"harmless {label}: the hooked match diverged at tick {t}")
                return
    print(f"harmless {label}: the hooked match equals the unhooked one for "
          f"{ticks} ticks")


def part_scenes():
    from simulation.weapons import PLANT_WEAPON
    if "verbs" in SCENES:
        census = VerbCensus()
        cov, kinds, refused = run_lockstep("scene verbs", [VerbDriver()], [SLOT_1B],
                                           VERB_TICKS, census=census)
        print("scene verbs census: " + "; ".join(
            f"{k} = {census.c[k]}" for k in sorted(census.c)))
        for k in VERB_BRANCHES:
            if not census.c.get(k):
                fail(f"scene verbs: the branch {k!r} was never taken (vacuous)")
        if refused == 0 or not kinds.get("throw") or not kinds.get("plant") \
                or not kinds.get("use"):
            fail(f"scene verbs: requests {dict(kinds)}, {refused} refused (vacuous)")
        if not cov.get(SLOT_1B.label):
            fail("scene verbs: the kernel never changed a row (vacuous)")
        _harmless(lambda: VerbDriver(), min(VERB_TICKS, 100), "verbs")
    if "zero" in SCENES:
        drv = VerbDriver(zero=True)
        W = drv.sim.weapons_tables.weapons.by_name      # the rows both sides read
        hand, plant = W["hand_grenade"], W[PLANT_WEAPON]
        saved = (hand.throw_windup_ticks, plant.plant_ticks)
        hand.throw_windup_ticks, plant.plant_ticks = 0, 0
        try:
            census = VerbCensus()
            run_lockstep("scene zero (class wind-up and plant 0)",
                         [drv], [SLOT_1B], 20, census=census)
            print("scene zero census: " + "; ".join(
                f"{k} = {census.c[k]}" for k in sorted(census.c)))
            for k in ZERO_BRANCHES:
                if not census.c.get(k):
                    fail(f"scene zero: the branch {k!r} was never taken (vacuous)")
        finally:
            hand.throw_windup_ticks, plant.plant_ticks = saved
    if "ship" in SCENES:
        census = VerbCensus()
        cov, kinds, _ = run_lockstep("scene ship", [ShipDriver()], [SLOT_1B],
                                     SHIP_TICKS, census=census)
        print("scene ship census: " + "; ".join(
            f"{k} = {census.c[k]}" for k in sorted(census.c)))
        if not kinds.get("throw") or not cov.get(SLOT_1B.label):
            fail(f"scene ship: no grenade thrown or no row changed (vacuous): "
                 f"{dict(kinds)}")
        _harmless(lambda: ShipDriver(), min(SHIP_TICKS, 200), "ship")


def _batch_drivers(n):
    return [VerbDriver(variant=e + 1) for e in range(n - 1)] + [ShipDriver(seed=2)]


def part_batch():
    for n in BATCH_NS:
        drivers = _batch_drivers(n)
        cov, kinds, refused = run_lockstep(f"batch N={n}", drivers, [SLOT_1B],
                                           BATCH_TICKS)
        counts = [len(d.sim.units) for d in drivers]
        ticks = [d.sim.tick for d in drivers]
        spans = [unit_requests.door_spans(d.sim).tolist() for d in drivers]
        live = [unit_requests.charges_live(d.sim) for d in drivers]
        print(f"batch N={n}: unit counts {counts}, ticks {ticks}, live charges {live}, "
              f"{len({str(s) for s in spans})} door layouts")
        if len(set(counts)) < 2 or len(set(ticks)) < 2 or len(set(live)) < 2 \
                or len({str(s) for s in spans}) < 2:
            fail(f"batch N={n}: the envs do not differ in unit count, tick, live "
                 f"charges and doors")
        if not kinds.get("throw"):
            fail(f"batch N={n}: no request (vacuous)")


# ---------------------------------------------------------------------------
# PART PADDING
# ---------------------------------------------------------------------------
def part_padding():
    """A padding row (valid 0) is never touched, requests nothing and takes no
    shove, whatever it holds: two envs of the verb scene with U two rows past
    the larger, and on the first tick the slot acts AND requests, its entry
    rows re-run with the padding rows filled with copies of LIVE acting rows
    (a latched shove, a throw, a use; valid cleared): the valid rows, the list
    and the coverage equal the clean run's, the padding rows come back as they
    went."""
    lay = unit_store.layout()
    drivers = [VerbDriver(variant=0), VerbDriver(variant=1)]
    U = max(len(d.sim.units) for d in drivers) + 3
    done = False
    acting = ("pending_shove", "pending_use", "pending_reload", "pending_throw")
    with ShadowLockstep(drivers, [SLOT_1B], n_units=U, check_alone=False) as ls:
        for _ in range(VERB_TICKS):
            rep = ls.tick()[SLOT_1B.label]
            if rep.coverage.sum() == 0 or not any(rep.requests):
                continue
            clean = np.stack(ls._pre[0])
            inp = ls.last_inputs[SLOT_1B.label]
            dirty = clean.copy()
            for e in range(2):
                live = [r for r in range(U) if clean[e, r, lay.valid]
                        and any(clean[e, r, lay.column(c).offset] for c in acting)]
                if not live:
                    live = [r for r in range(U) if clean[e, r, lay.valid]]
                for i, r in enumerate(r for r in range(U) if not clean[e, r, lay.valid]):
                    dirty[e, r] = clean[e, live[(3 * i + e) % len(live)]]
                    dirty[e, r, lay.valid] = 0
            runs = []
            for rows in (clean, dirty):
                ls.store.upload(rows)
                ls.req.begin()
                cov = np.asarray(SLOT_1B.call(ls.store, inp, None, ls.req))
                runs.append((cov, ls.store.download(), ls.req.records()))
            valid = clean[:, :, lay.valid].astype(bool)
            (c0, g0, r0), (c1, g1, r1) = runs
            if not np.array_equal(g1[valid], g0[valid]):
                fail("padding: the valid rows differ when the padding rows hold live rows")
            if not np.array_equal(g1[~valid], dirty[~valid]):
                fail("padding: a padding row was written")
            if r1 != r0:
                fail("padding: the request list differs (a padding row requested)")
            if not np.array_equal(c1, c0):
                fail(f"padding: coverage {c1.tolist()} vs {c0.tolist()}")
            n_act = sum(int(any(dirty[e, r, lay.column(c).offset] for c in acting))
                        for e in range(2) for r in range(U) if not valid[e, r])
            print(f"padding: tick {rep.tick}, N=2, U={U}, {int((~valid).sum())} padding "
                  f"rows ({n_act} holding a latch) left untouched and silent; valid rows "
                  f"and list equal the clean run (coverage {c0.tolist()}, "
                  f"{sum(len(x) for x in r0)} requests)")
            if n_act == 0:
                fail("padding: no padding row held a latch (vacuous)")
            done = True
            break
    if not done:
        fail("padding: the slot never acted and requested (vacuous)")


# ---------------------------------------------------------------------------
# PART OVERFLOW
# ---------------------------------------------------------------------------
def part_overflow():
    """A list of capacity 1 on the verb scene's first tick with 2+ requests."""
    drv = VerbDriver()
    held = {}
    from simulation.simulation import Simulation
    orig = Simulation._consume_direct_intents
    tested = 0
    for _ in range(60):
        sim = drv.sim
        U = len(sim.units)
        n0 = len(sim.projectiles)
        n_doors = [d.want_open for d in sim._doors]

        def spy(s, __o=orig):
            held["pre"] = unit_store.pack(s.units, U)
            held["inp"] = capture(SLOT_1B.inputs, s)
            return __o(s)
        Simulation._consume_direct_intents = spy
        try:
            drv.tick()
        finally:
            Simulation._consume_direct_intents = orig
        inp = gather(SLOT_1B.inputs, [held["inp"]])
        st = unit_store.DeviceStore(1, U, tables=unit_tables.values_batch([sim]))
        full = unit_requests.RequestList(1, U)
        st.upload(held["pre"][None])
        full.begin()
        unit_kernels.verbs(st, full, inp.tick, inp.tables["door_spans"],
                           inp.scalars["charges_live"])
        want = full.records()[0]
        if len(want) < 2:
            continue
        cap = 1
        small = unit_requests.RequestList(1, U, r_max=cap)
        st.upload(held["pre"][None])
        small.begin()
        try:
            _run_verbs(st, inp, small)
            fail("overflow: the Python face did not refuse a run that dropped requests")
        except unit_requests.UnitRequestOverflow as ex:
            if "verbs" not in str(ex) or "env(s) [0]" not in str(ex):
                fail(f"overflow: the refusal does not name the kernel and env: {ex}")
        recs, counts, dropped = small.download()
        if int(counts[0]) != cap or int(dropped[0]) != len(want) - cap:
            fail(f"overflow: count {int(counts[0])} / dropped {int(dropped[0])} != "
                 f"{cap} / {len(want) - cap}")
        got = [tuple(int(x) for x in recs[0, i]) for i in range(cap)]
        if got != want[:cap]:
            fail(f"overflow: the record within the capacity {got} != the first {want[:cap]}")
        if int(st.last_dropped[0]) != len(want) - cap:
            fail("overflow: store.last_dropped is not the dropped count")
        tested += 1
        del n0, n_doors
        print(f"overflow: tick {sim.tick - 1}: {len(want)} requests into a list of "
              f"{cap}: {cap} written (the first), {int(dropped[0])} dropped and counted, "
              f"the run refused by name")
        break
    if tested == 0:
        fail("overflow: no tick of the verb scene made 2 requests (vacuous)")


# ---------------------------------------------------------------------------
# PART CHAINED
# ---------------------------------------------------------------------------
def _books_host_side(sim, pre, post):
    """What slot 1b does on the host that is not a row or a request: the
    shoved targets' momentum books (REFERENCE_ONLY; the impulse each row
    gained, unit_impulse.add's IN entries)."""
    from simulation.unit_impulse import IN_X, IN_Y
    lay = unit_store.layout()
    ix, iy = lay.column("imp_x").offset, lay.column("imp_y").offset
    for r, u in enumerate(sim.units):
        u.impulse_books[IN_X] += int(post[r, ix] - pre[r, ix])
        u.impulse_books[IN_Y] += int(post[r, iy] - pre[r, iy])


def _orders_host_side(sim, pre, post):
    """1a..1c's host side: the order requests' table dropped (1b'). (The
    momentum books, REFERENCE_ONLY, are not kept across 1a + 1b.)"""
    sim._order_requests.clear()


def _chained(name, make, slots, host_side, ticks, need_requests=True):
    twin = make()
    ref = make()
    t0 = time.perf_counter()
    n = 0
    with ChainedSlots(twin, slots, host_side=host_side) as ch:
        for _ in range(ticks):
            if getattr(ref, "is_terminal", lambda: False)():
                break
            twin.tick()
            ref.tick()
            n += 1
            if not _same(twin.sim, ref.sim):
                fail(f"{name}: the chained match left the reference at tick "
                     f"{twin.sim.tick}")
                break
    k = [s.label for s in slots].index(SLOT_1B.label)
    acted = sum(1 for covs in ch.coverage if int(covs[k][0]) > 0)
    kinds = Counter(unit_requests.KINDS[r[0] - 1] for rq in ch.requests for r in rq)
    print(f"{name}: {n} ticks, the chained twin equal to the reference every tick "
          f"({ch.runs} chain runs; [1b intents] changed rows on {acted} ticks; "
          f"requests turned into host objects {dict(kinds)}; "
          f"{time.perf_counter() - t0:.0f} s)")
    if ch.runs != n:
        fail(f"{name}: the chain ran {ch.runs} times in {n} ticks")
    if acted == 0 or (need_requests and not kinds):
        fail(f"{name}: slot 1b never acted or requested (vacuous)")


def part_chained():
    four = [SLOT_1A, SLOT_1B, SLOT_1BP, SLOT_1C]
    _chained("chained 1b (verbs)", lambda: VerbDriver(), [SLOT_1B], _books_host_side,
             VERB_TICKS)
    _chained("chained 1b (ship)", lambda: ShipDriver(), [SLOT_1B], _books_host_side,
             SHIP_TICKS)
    _chained("chained 1a + 1b + 1b' + 1c (verbs)", lambda: VerbDriver(), four,
             _orders_host_side, VERB_TICKS)
    _chained("chained 1a + 1b + 1b' + 1c (ship)", lambda: ShipDriver(), four,
             _orders_host_side, SHIP_TICKS)


# ---------------------------------------------------------------------------
# PART DONE
# ---------------------------------------------------------------------------
_ARENA_CACHE: dict = {}


def _arena(kind):
    """A hull box with two marine teams: ``bots`` (they fight to an end),
    ``idle`` (nobody fights; the limit ends it), ``horde`` (idle marines, a
    horde that turns them)."""
    if kind in _ARENA_CACHE:
        return _ARENA_CACHE[kind]
    import tempfile
    import level_lib
    from PIL import Image
    from simulation.factions import ZOMBIE_TEAM
    from simulation.match import MatchSpec, MatchTeam, SpawnRoom
    from simulation.materials import MAT_AIR, MAT_HULL
    ctrl = "bots" if kind == "bots" else "idle"
    team = MatchTeam(n=2, loadout="rifleman", env_profile="default", controller=ctrl)
    rooms = [SpawnRoom(team=0, x=2, y=2, w=8, h=8, max_per_team=4),
             SpawnRoom(team=1, x=30, y=14, w=8, h=8, max_per_team=4),
             SpawnRoom(team=ZOMBIE_TEAM, x=12, y=2, w=8, h=8, max_per_team=4)]
    spec = MatchSpec(teams=(team, team), seed=1,
                     zombies=3 if kind == "horde" else 0, time_limit_s=60.0)
    tmp = Path(tempfile.mkdtemp(prefix=f"p64a_{kind}_"))
    lvl = tmp / f"arena_{kind}"
    lvl.mkdir()
    h, w = 24, 40
    level_lib.write_level_header(lvl, name=f"arena_{kind}", tile_size_m=1.0)
    tm = np.full((h, w), MAT_HULL, dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = MAT_AIR
    level_lib.write_tilemap_csv(lvl, tm, csv_bak=False)
    Image.new("RGB", (w, h)).save(lvl / "diffuse.png")
    level_lib.write_spawn_rooms(lvl / "level.toml", rooms)
    level_lib.write_match(lvl, spec)
    level = level_loader.load(f"arena_{kind}", str(tmp))
    _ARENA_CACHE[kind] = level
    return level


class MatchDriver:
    def __init__(self, kind, limit_s, seed=1, kill_team=None, kill_at=None):
        from simulation.match_session import MatchSession
        level = _arena(kind)
        spec = dataclasses.replace(level.match, time_limit_s=float(limit_s))
        self.session = MatchSession(level, spec, seed=seed, breach_physics=bp)
        self.sim = self.session.sim
        self.kill_team, self.kill_at = kill_team, kill_at

    def tick(self):
        if self.kill_team is not None and self.sim.tick == self.kill_at:
            from simulation.damage import DamagePacket, apply_packet
            for u in self.sim.units:            # one team falls (a fixture)
                if u.team == self.kill_team and u.alive and not u.is_zombie:
                    apply_packet(u, DamagePacket(amount=1.0e6, dtype=0, attacker_id=-1),
                                 [], source="melee")
        self.session.tick()


def _done_ref(d):
    rules, sim = d.session.rules, d.sim
    term = rules.is_terminal(sim)
    res = rules.result(sim)
    teams = rules.living_teams(sim)
    outcome = (unit_kernels.DONE_NONE if res is None else
               unit_kernels.DONE_WIN if res.outcome == "win" else unit_kernels.DONE_DRAW)
    winner = -1 if res is None or res.team is None else int(res.team)
    return [int(term), outcome, winner, len(teams)]


def _run_done(name, drivers, ticks, after=30):
    """Every tick (and ``after`` ticks past the last env's end) the done kernel
    over the batch equals each env's MatchRules."""
    U = max(len(d.sim.units) for d in drivers)
    st = unit_store.DeviceStore(len(drivers), U,
                                tables=unit_tables.values_batch([d.sim for d in drivers]))
    ended = [None] * len(drivers)
    outcomes = [None] * len(drivers)
    extra = 0
    for t in range(ticks):
        for d in drivers:
            d.tick()
        rows = np.stack([unit_store.pack(d.sim.units, U) for d in drivers])
        st.upload(rows)
        got = unit_kernels.match_done(st, [d.sim.tick for d in drivers])
        for e, d in enumerate(drivers):
            want = _done_ref(d)
            if [int(x) for x in got[e]] != want:
                fail(f"{name}: tick {d.sim.tick} env {e}: device {got[e].tolist()} != "
                     f"MatchRules {want}")
                return ended, outcomes
            if want[0] and ended[e] is None:
                ended[e] = d.sim.tick
                outcomes[e] = (want[1], want[2])
        if all(x is not None for x in ended):
            extra += 1
            if extra > after:
                break
    print(f"{name}: N={len(drivers)}, the done kernel == MatchRules every tick; ended at "
          f"{ended}, outcomes (1 win / 2 draw, winner) {outcomes}")
    return ended, outcomes


def part_done():
    lim = {"bots": 60.0, "idle": 6.0, "horde": 60.0}
    tps = int(ShipDriver(hooks=False).sim._tps)
    del tps
    ended, out = _run_done("done (elimination: bots)", [MatchDriver("bots", lim["bots"])],
                           DONE_TICKS)
    if ended[0] is None or out[0][0] != unit_kernels.DONE_WIN:
        fail(f"done: the bots match did not end in a win ({ended}, {out})")
    ended, out = _run_done("done (the limit, both teams alive: a draw)",
                           [MatchDriver("idle", lim["idle"])], DONE_TICKS)
    if ended[0] is None or out[0][0] != unit_kernels.DONE_DRAW:
        fail(f"done: the idle match did not end in a draw at the limit ({ended}, {out})")
    ended, out = _run_done("done (one team falls: a win before the limit)",
                           [MatchDriver("idle", lim["idle"], kill_team=1, kill_at=40)],
                           DONE_TICKS)
    if ended[0] is None or out[0] != (unit_kernels.DONE_WIN, 0):
        fail(f"done: the fallen team's match did not end in team 0's win ({ended}, {out})")
    batch = [MatchDriver("idle", 3.0), MatchDriver("idle", 5.0, seed=2),
             MatchDriver("idle", 9.0, kill_team=0, kill_at=70), MatchDriver("horde", 40.0),
             MatchDriver("bots", 60.0, seed=3)]
    ended, out = _run_done("done (a batch ending on different ticks)", batch, DONE_TICKS)
    if None in ended or len(set(ended)) < 3:
        fail(f"done: the batch's envs did not all end, on different ticks ({ended})")


# ---------------------------------------------------------------------------
# main / bench
# ---------------------------------------------------------------------------
PARTS = ("tables", "lib", "scenes", "batch", "padding", "overflow", "chained", "done")


def main(parts=None) -> int:
    if not getattr(bp, "HAS_CUDA", False):
        print("UNIT_VERBS_RESULT: FAIL (not the CUDA build)")
        return 1
    parts = parts or os.environ.get("UNIT_VERBS_PARTS", ",".join(PARTS)).split(",")
    t0 = time.perf_counter()
    for p in parts:
        try:
            {"tables": part_tables, "lib": part_lib, "scenes": part_scenes,
             "batch": part_batch, "padding": part_padding, "overflow": part_overflow,
             "chained": part_chained, "done": part_done}[p]()
        except Exception as ex:          # a part's crash fails it, the rest run
            import traceback
            traceback.print_exc()
            fail(f"part {p} raised {type(ex).__name__}: {ex}")
    print(f"total {time.perf_counter() - t0:.0f} s")
    ok = not FAILS
    print("UNIT_VERBS_RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def bench(reps: int = 200) -> int:
    """The TIMING table: slot 1b and the done kernel, upload + kernel +
    download and the kernel call alone (launches + the list's gather + the
    plant pass + the counts' D2H and sync), medians over ``reps``, at N = 1,
    8, 64 (env rows replicated from the verb scene's busiest tick)."""
    import cupy as cp
    drv = VerbDriver()
    sim = drv.sim
    from simulation.simulation import Simulation
    orig = Simulation._consume_direct_intents
    held = {}

    def spy(s, __o=orig):
        held["pre"] = unit_store.pack(s.units)
        held["inp"] = capture(SLOT_1B.inputs, s)
        return __o(s)
    Simulation._consume_direct_intents = spy
    try:
        drv.tick()
        drv.tick()
        drv.tick()                     # k = 2: the busiest tick
    finally:
        Simulation._consume_direct_intents = orig
    rows1 = held["pre"]
    U = rows1.shape[0]
    for n in (1, 8, 64):
        rows = np.ascontiguousarray(np.broadcast_to(rows1, (n,) + rows1.shape))
        st = unit_store.DeviceStore(n, U, tables=np.tile(unit_tables.values(sim), (n, 1)))
        req = unit_requests.RequestList(n, U)
        inp = gather(SLOT_1B.inputs, [held["inp"]] * n)
        ticks = [sim.tick] * n
        kern = {
            "1b verbs": lambda: (req.begin(), _run_verbs(st, inp, req)),
            "done match_done": lambda: unit_kernels.match_done(st, ticks),
        }
        for name, f in kern.items():
            full, alone = [], []
            for _ in range(reps):
                t0 = time.perf_counter()
                st.upload(rows)
                t1 = time.perf_counter()
                f()
                t2 = time.perf_counter()
                st.download()
                req.download()
                t3 = time.perf_counter()
                full.append(t3 - t0)
                alone.append(t2 - t1)
            print(f"TIMING N={n:3d} U={U} {name:16s} upload+kernel+download "
                  f"{1e3 * float(np.median(full)):.3f} ms  kernel call "
                  f"{1e3 * float(np.median(alone)):.3f} ms")
        del st, req
        cp.get_default_memory_pool().free_all_blocks()
    return 0


if __name__ == "__main__":
    sys.exit(main())
