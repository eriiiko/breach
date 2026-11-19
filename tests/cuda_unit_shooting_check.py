"""#29 P7a.2 -- THE SHOOTING GATE: slot 4's shooting (``combat.process_shooting``
on the continuous path: ``_directional_fire``, ``fire_burst``,
``start_spray_burst_directional``), slot 2''s rounds (``Simulation.
_march_rounds``) and THE MARCH they share (``combat.BulletInFlight.advance``)
on the device, with the round table and the noise list (object tables), each
against its Python law at tol 0 on every row, record, table row and noise.

Runs inside the GPU subprocess (tests/cuda_harness.run_cuda_script; wrapper
tests/test_cuda_unit_shooting.py). The device code: ``cpp/src/cuda_unit_march.cuh``
(the march, one ``__device__`` function), ``cpp/src/cuda_unit_shooting.cu`` (the
two slots), the unit library's magazine law; the Python faces
``simulation.unit_kernels.shooting`` / ``rounds_march`` / ``resolve``; the
harness ``tests/cuda_unit_lockstep.py`` with the lists ``effects``, ``rounds``
and ``noise``.

PART TABLES -- every P7a.2 table row equals an independent read of the sim's
own weapon / ammo / payload rows, its impulse table, its materials and the
profiles, on a 1/3 m and a 1 m map (the ranges, the hearing radii and the
momenta move with the tile size).

PART MARCH -- the march on SYNTHETIC rounds through the slot-2' kernel with
the compaction off (the test-only probe form): random materials of every
class on random maps (walls, soft cover, air), units placed at random (two
on one tile, the shooter standing in its own line, a converted shooter, the
dead), rounds of every projectile row with random budgets, speeds below and
above a tile per tick, ranges, payloads, crit rows patched in; the device's
records and round rows equal the Python's, every outcome (wall stop, unit
hit, range end, grid exit, budget-starved, payload stop and its off-grid
fallback) and both outcomes of each draw seen.

PART SCENE -- the shooting scene built for the gate at 1/3 m tiles (a K5
round lives three ticks): duels that kill each other, two rounds on one
target in one resolve, a wall chewed to zero, rounds that fly over several
ticks and land later, a shooter that dies (and one that turns) while its
rounds fly, soft cover with both draw outcomes, crit rows patched in with
every arc, a zombie target, shotguns and a PDW, every gate refusal (stunned,
no weapon, no trigger, the cadence, the stall, dry), the snap cone, the
reload, recoil moving the next shot, two units on one tile, a grid exit, a
range end. Slots 2' (march), 2''s landing, 4 and 4c SHADOWED together: rows,
effect records, the round table and the noise list equal record for record;
the landing and 4c fed the host's lists: rows, ring, planes, books. A census
of the Python law's branches, each required, the unreachable named.
PART SCENEB -- payload rounds (the GL-6, a plasma bolt with a budget-starved
speed), a spray, a payload round leaving the grid: slots 2' and 4 shadowed
(the PAYLOAD records compared; their apply is P7c.1's, so no landing here).

PART BATCH -- N = 2 / 8 scenes differing in layout, seed, unit count and
fields: env i == env i alone. PART PADDING -- live-looking padding rows stay
untouched. PART OVERFLOW -- the effect list, the noise list and the round
table past their capacities: counted and refused by name. PART REFUSE -- a
beam trigger, WEGO orders, a round under physical cover: refused by name.

PART CHAINED -- slot 4 + 4a + 4c, and slot 2' with its landing, replaced by
the device's in a running scene: the twin equal to an untouched reference
every tick, the host law run 0 times in the twin. PART SHIP -- ship_demo's
bots shadowed SHIP_TICKS ticks (shots, hits and kills counted).

PROPERTY: the device shooting, the march and the rounds' slot are the
Python's integer for integer, record for record in the reference's order,
per env. BREAKS IF a step, the stop, the list order, the shooter exclusion,
a draw, a record kind's field, the recoil, the noise order, the table's
append or compaction, the batch rebase or a table row differs from the
reference. Each validated by breaking it (the patch's report).

Prints per-part lines and ``UNIT_SHOOT_RESULT: PASS`` / ``FAIL``; exits 0 / 1.
``bench()`` prints the TIMING table.
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
                                ShadowLockstep)
from simulation import (applier, object_tables, unit_events,  # noqa: E402
                        unit_kernels, unit_schema, unit_store, unit_tables,
                        unit_traits)

FAILS: list = []
ONE = unit_traits.ONE
PI_Q16 = 205887
SCENE_TICKS = int(os.environ.get("UNIT_SHOOT_TICKS", "200"))
SHIP_TICKS = int(os.environ.get("UNIT_SHOOT_SHIP_TICKS", "500"))
BATCH_TICKS = int(os.environ.get("UNIT_SHOOT_BATCH_TICKS", "60"))
BATCH_NS = [int(x) for x in os.environ.get("UNIT_SHOOT_BATCH_NS", "2,8").split(",")]
CHAIN_TICKS = int(os.environ.get("UNIT_SHOOT_CHAIN_TICKS", "160"))
MARCH_TRIALS = int(os.environ.get("UNIT_SHOOT_MARCH_TRIALS", "24"))


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
SHOOT_IN = ("tick", "draw_key", "field:solid", "field:material")


def _run_4(st, inp, lists):
    return unit_kernels.shooting(st, lists["effects"], lists["rounds"], lists["noise"],
                                 inp.tick, inp.tables["draw_key"], inp.fields)


def _run_2r(st, inp, lists):
    return unit_kernels.rounds_march(st, lists["effects"], lists["rounds"], inp.tick,
                                     inp.tables["draw_key"], inp.fields)


def _run_4a(st, inp, lists):
    return unit_kernels.melee(st, lists["effects"], inp.tick)


def _run_resolve(st, inp, ring, lists):
    return unit_kernels.resolve(st, lists["effects"], inp.fields, ring)


def _conductor(sim, args, kw, caller):
    return caller == "_step_pre_physics"


def _landing(sim, args, kw, caller):
    return caller == "_update_rounds"


SLOT_4 = DeviceSlot(label="4 shooting", run=_run_4,
                    inputs=SHOOT_IN + ("list:effects", "list:rounds", "list:noise"),
                    outputs=("list:effects", "list:rounds", "list:noise"),
                    host_only=(("noise_log", "the noise list: compared as list:noise"),))
SLOT_2R = DeviceSlot(label="2' rounds", run=_run_2r, inputs=SHOOT_IN + ("list:rounds",),
                     outputs=("list:effects", "list:rounds"))
SLOT_2L = DeviceSlot(label="4c resolve", name="2' landing", run=_run_resolve,
                     inputs=("list:effects",), events=True,
                     outputs=PLANE_OUTPUTS + BOOK_OUTPUTS, host_only=HOST_ONLY,
                     select=_landing)
SLOT_4A = DeviceSlot(label="4a melee", run=_run_4a, inputs=("tick", "list:effects"),
                     outputs=("list:effects",))
SLOT_4C = DeviceSlot(label="4c resolve", run=_run_resolve, inputs=("list:effects",),
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


def _rounds_rec(sim):
    return [tuple(r) for r in object_tables.records(unit_schema.ROUND_TABLE, sim.bullets)]


def _same(a, b):
    return (_unit_hash(a) == _unit_hash(b) and _field_digest(a) == _field_digest(b)
            and _rounds_rec(a) == _rounds_rec(b)
            and list(a.gmap.noise_log) == list(b.gmap.noise_log))


# ---------------------------------------------------------------------------
# Patched rows (a hand-built scene's data): crits on two ranged rows, a
# budget-starved plasma, a heavy recoil -- on EVERY tables object the sims
# and the law read (the process-wide tables and each sim's own)
# ---------------------------------------------------------------------------
class Patched:
    """Within the block: ``crit`` {weapon: crit_chance}, ``speed`` {ammo:
    speed_q16}, ``recoil`` {ammo: multiplier of the impulse table's recoil};
    restored at the exit."""

    def __init__(self, sims, crit=None, speed=None, recoil=None):
        self.sims, self.crit, self.speed = list(sims), crit or {}, speed or {}
        self.recoil = recoil or {}
        self.saved = []

    def __enter__(self):
        from simulation.weapons import get_tables
        tabs = {id(t): t for t in [get_tables()] + [s.weapons_tables for s in self.sims]}
        for t in tabs.values():
            for n, c in self.crit.items():
                w = t.weapons.by_name[n]
                self.saved.append((w, "crit_chance", w.crit_chance))
                w.crit_chance = c
            for n, v in self.speed.items():
                a = t.ammo.by_name[n]
                self.saved.append((a, "speed_q16", a.speed_q16))
                a.speed_q16 = v
        imps = {id(s.impulse_table): s.impulse_table for s in self.sims}
        for it in imps.values():
            for n, k in self.recoil.items():
                d = it.ammo_recoil_q
                self.saved.append((d, n, d.get(n, 0)))
                d[n] = int(d.get(n, 0)) * k
        return self

    def __exit__(self, *exc):
        for obj, k, v in reversed(self.saved):
            if isinstance(obj, dict):
                obj[k] = v
            else:
                setattr(obj, k, v)
        return False


# ---------------------------------------------------------------------------
# The shooting scene
# ---------------------------------------------------------------------------
SPACE = level_loader.SPACE_CODE
EAST, WEST, SOUTH, NORTH = 0, PI_Q16, -PI_Q16 // 2, PI_Q16 // 2
SH, SW = 96, 330


def _scene_level(variant):
    """A hull box in space at 1/3 m tiles with lanes; a glass pane to chew, a
    furniture row (soft cover), a gap in the hull (to the grid's edge); the
    variant moves the wall and adds an interior block (the batch's fields
    differ)."""
    from level_loader import LevelData
    from simulation.materials import (MAT_AIR, MAT_FURNITURE, MAT_GLASS, MAT_HULL,
                                      MAT_WOOD)
    tm = np.full((SH, SW), SPACE, dtype=np.int32)
    tm[1:SH - 1, 1:SW - 1] = MAT_HULL
    tm[2:SH - 2, 2:SW - 2] = MAT_AIR
    c = 40 + 2 * (variant % 3)
    tm[10:16, c] = MAT_GLASS                      # the wall lane (row 12): chewed to zero
    tm[20:26, 60] = MAT_FURNITURE                 # soft cover (row 22)
    tm[30:36, 60] = MAT_FURNITURE
    tm[70:76, SW - 2] = SPACE                     # the gap: the lane leaves the grid
    tm[70:76, SW - 1] = SPACE
    if variant % 2:
        tm[86:90, 100:110] = MAT_WOOD
    return LevelData(name=f"p7a2_shoot{variant}", version="3", path=Path("."),
                     tilemap=tm, tile_size_m=1.0 / 3.0, diffuse_path=Path("."),
                     boundary="space")


SPREAD_2 = {"SPREAD": ONE * 3 // 2}


class ShootDriver:
    """The shooting scene, scripted: each shooter holds the trigger; fixtures
    written between ticks (the stunned, the blinded, the zombies held, a
    shooter killed or converted while its rounds fly). ``variant`` > 0:
    another seed and layout, bystanders, every group shifted."""

    def __init__(self, variant=0, kind="a"):
        from simulation import Simulation
        from simulation.ruleset import ContinuousRealtime
        self.variant, self.k, self.kind = variant, 0, kind
        sim = Simulation(_scene_level(variant), seed=20261115 + 11 * variant,
                         breach_physics=bp, enable_recorder=False,
                         ruleset=ContinuousRealtime())
        sim.set_paused(False)
        self.sim = sim
        self.who = {}
        self.shooters = []
        self.zombies = []
        if variant % 2:
            self._bystanders()
        (self._setup_a if kind == "a" else self._setup_b)()
        if not variant % 2:
            self._bystanders()

    def _bystanders(self):
        for i in range(self.variant % 4):
            self._add(f"by{i}", 200 + 6 * i, 88, facing=NORTH, trigger=False)

    def _add(self, name, x, y, team=0, facing=EAST, weapon="k5_carbine", trigger=True,
             hp=10000.0, hooks=None, mags=None):
        from simulation.turning import place
        from simulation.unit import Unit
        dy = self.variant % 2
        u = Unit(name, x=float(x), y=float(y + dy), team=team)
        self.sim.add_unit(u, hooks=hooks)
        place(u, facing)
        u.current_hp = float(hp)
        u.weapon_id = weapon
        if mags is not None:
            u.magazines = mags
            u.magazines_weapon = weapon
        self.who[name] = u
        if trigger:
            self.shooters.append(u)
        return u

    def _kill(self, u, zombie=False):
        from simulation.ai_zombie import convert_marines_to_zombies
        from simulation.damage import DamagePacket, apply_packet
        # arc #63 P5c: a kill within the body (an overkill of a whole body
        # more destroys the corpse, which then never converts): the hp set
        # to one point, then a one-point packet (its exposure scale at most a
        # few points over)
        u.hp_q = 1 << 16                                  # fixture: one point left
        apply_packet(u, DamagePacket(amount=1.0, dtype=0, attacker_id=-1), [],
                     source="melee", mark_killed_by_zombie=zombie)
        if zombie:
            convert_marines_to_zombies([u])

    def _setup_a(self):
        a = self._add
        # D1: a duel that kills: two K5s of two teams, low hp
        a("D1", 6, 4, team=0, facing=EAST, hp=9.0)
        a("D2", 30, 4, team=1, facing=WEST, hp=9.0)
        # D2: two shooters on one target (two rounds in one resolve)
        a("P1", 6, 46, team=0, facing=EAST)
        a("P2", 6, 50, team=0, facing=EAST)
        a("PT", 20, 47, team=1, facing=WEST, hp=400.0, trigger=False)
        # W: the wood wall chewed to zero (then the rounds pass)
        a("W", 20 + 2 * (self.variant % 3), 11, facing=EAST, weapon="lr50")
        # C: soft cover (a furniture row between shooter and target)
        a("C", 40, 21, facing=EAST)
        a("CT", 61, 21, team=1, facing=WEST, hp=1.0e6, trigger=False)   # behind it
        # X: crits (p12 / mp11 patched) on targets facing three ways
        a("X", 80, 40, facing=EAST, weapon="p12_whisper")
        a("XF", 95, 40, team=1, facing=WEST, hp=1.0e6, trigger=False)    # front
        a("X2", 80, 52, facing=EAST, weapon="mp11_pdw")
        a("XB", 95, 51, team=1, facing=EAST, hp=1.0e6, trigger=False)    # behind
        a("XS", 99, 53, team=1, facing=NORTH, hp=1.0e6, trigger=False)   # flank
        # F: far targets (a K5 round lives three ticks at 1/3 m) and a shooter
        # that dies, one that turns, while their rounds fly
        a("F", 6, 60, facing=EAST)
        # FT shoots back: a round landing at 2' kills it with the trigger held,
        # so slot 4 of the same tick meets a corpse holding the trigger
        a("FT", 250, 60, team=1, facing=WEST, hp=30.0)
        a("FD", 6, 64, facing=EAST)                       # killed mid-flight
        a("FZ", 6, 68, facing=EAST)                       # converted mid-flight
        a("FT2", 250, 66, team=1, facing=WEST, hp=1.0e6, trigger=False)
        # Z: a zombie target (the bullet multiplier), held stunned
        a("Z", 120, 30, facing=EAST, weapon="jackhammer_8")
        z = a("ZT", 132, 30, facing=WEST, trigger=False, hp=1.0e6)
        self._kill(z, zombie=True)
        z.current_hp = 1.0e6
        self.zombies.append(z)
        # two units on one tile (the first in list order is hit)
        a("T1", 120, 12, facing=EAST, weapon="p12_whisper")
        a("TA", 140, 12, team=1, facing=WEST, hp=1.0e6, trigger=False)
        a("TB", 140, 12, team=1, facing=WEST, hp=1.0e6, trigger=False)
        # E: a grid exit (out of the hull gap) and a range end (jackhammer)
        a("E", 300, 72, facing=EAST)
        a("R", 160, 80, facing=WEST, weapon="jackhammer_8")
        # K: recoil moving the next shot (lr50, recoil scaled up), facing open floor
        a("K", 200, 40, facing=NORTH, weapon="lr50", hooks=SPREAD_2)
        # G: the gate refusals
        st = a("St", 220, 10, facing=EAST)                 # stunned
        a("No", 226, 10, facing=EAST, weapon="")           # no weapon
        a("Nt", 232, 10, facing=EAST, trigger=False)       # no trigger
        d = a("Dd", 238, 10, facing=EAST)                  # dead (1b drops its trigger)
        self._kill(d)
        a("Bl", 220, 20, facing=SOUTH)                     # blinded: the snap cone
        a("Dry", 226, 20, facing=SOUTH, weapon="p12_whisper", mags=1)   # goes dry
        a("Mel", 232, 20, facing=SOUTH, weapon="rifle_butt")            # a melee row
        a("Gr", 238, 20, facing=SOUTH, weapon="hand_grenade")           # lobbed
        self.stunned = [st]
        self.blinded = [self.who["Bl"]]

    def _setup_b(self):
        a = self._add
        a("G", 6, 46, facing=EAST, weapon="gl6_revolver")
        a("GT", 60, 47, team=1, facing=WEST, hp=1.0e6, trigger=False)
        a("S", 6, 60, facing=EAST, weapon="sunspot")
        a("ST", 40, 61, team=1, facing=WEST, hp=1.0e6, trigger=False)
        a("H", 6, 4, facing=EAST, weapon="helios")
        a("Sp", 80, 30, facing=EAST, weapon="dragon_7")
        a("GE", 300, 72, facing=EAST, weapon="gl6_revolver")   # leaves the grid
        a("GW", 20 + 2 * (self.variant % 3), 11, facing=EAST, weapon="gl6_revolver")
        self.stunned, self.blinded = [], []

    def _before(self):
        from simulation.status import BLINDED, STUNNED, apply_status
        sim, w, k = self.sim, self.who, self.k
        for z in self.zombies:
            if z.alive:
                apply_status(z, STUNNED, 0, 3)
        for u in self.stunned:
            apply_status(u, STUNNED, 0, 3)
        for u in self.blinded:
            apply_status(u, BLINDED, 0, 3)
        if self.kind == "a":
            if k == 9 and w["FD"].alive:
                self._kill(w["FD"])                       # its rounds still fly
            if k == 13 and w["FZ"].alive and not w["FZ"].is_zombie:
                self._kill(w["FZ"], zombie=True)          # it keeps its id
                self.zombies.append(w["FZ"])
        for u in self.shooters:
            if u.alive and not u.is_zombie:
                sim.set_trigger(u.id, True)

    def tick(self):
        self._before()
        self.sim.step()
        self.sim.set_paused(False)
        self.k += 1

    def is_terminal(self):
        return False


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
# The census: the reference's own decisions in slot 4, the march and the
# resolves, by branch (reads only)
# ---------------------------------------------------------------------------
class Census:
    def __init__(self):
        self.c = Counter()
        self._saved = []
        self._kills = []
        self._last_origin = {}
        self._in_absorb = False

    def _wrap(self, cont, attr, make):
        orig = getattr(cont, attr)
        w = make(orig)
        w.__wrapped__ = orig
        setattr(cont, attr, w)
        self._saved.append((cont, attr, orig))

    def __enter__(self):
        from simulation import attack_resolver, combat, simultaneous
        from simulation import simulation as simmod
        cen = self

        def mk_ps(o):
            def w(gmap, units, tick, *a, **kw):
                cen._classify_slot(units)
                return o(gmap, units, tick, *a, **kw)
            return w
        self._wrap(simmod, "process_shooting", mk_ps)

        def mk_df(o):
            def w(gmap, units, u, tick, shots, real_time, rng, events, bullets, weapon,
                  flags, queue=None, effects=None):
                pre = cen._classify_trigger(u, weapon, tick, flags)
                out = o(gmap, units, u, tick, shots, real_time, rng, events, bullets,
                        weapon, flags, queue=queue, effects=effects)
                cen._after_trigger(u, weapon, tick, pre)
                return out
            return w
        self._wrap(combat, "_directional_fire", mk_df)

        def mk_adv(o):
            def w(b, gmap, units, shots, real_time, rng, events=None, queue=None,
                  cover=None, facing_defense=False, effects=None):
                pre = (b.rx_q, b.ry_q, b.remaining_steps, b.budget_q16,
                       0 if effects is None else len(effects._pending))
                out = o(b, gmap, units, shots, real_time, rng, events=events,
                        queue=queue, cover=cover, facing_defense=facing_defense,
                        effects=effects)
                cen._classify_march(b, gmap, units, pre, out, effects)
                return out
            return w
        self._wrap(combat.BulletInFlight, "advance", mk_adv)

        def mk_re(o):
            def w(exposure_q, d):
                r = o(exposure_q, d)
                cen.c["draw: an exposure roll connects" if r
                      else "draw: an exposure roll absorbs"] += 1
                return r
            return w
        self._wrap(attack_resolver, "roll_exposure", mk_re)

        def mk_rc(o):
            def w(chance, mult, d):
                r = o(chance, mult, d)
                cen.c["draw: a crit roll hits" if r else "draw: a crit roll misses"] += 1
                return r
            return w
        self._wrap(attack_resolver, "roll_crit", mk_rc)

        def mk_am(o):
            def w(angle_q, target):
                m = o(angle_q, target)
                cb = attack_resolver._mult_q
                from config import CFG
                cc = CFG.combat
                name = ("front" if m == cb(getattr(cc, "crit_front_mult", 1.0)) else
                        "behind" if m == cb(getattr(cc, "crit_behind_mult", 4.0))
                        else "flank")
                cen.c[f"crit: the {name} arc"] += 1
                return m
            return w
        self._wrap(attack_resolver, "arc_multiplier", mk_am)

        def mk_cw(o):
            def w(gmap, iy, ix, wall_damage):
                from simulation.materials import MAT_AIR
                from simulation import wall_fixed
                if wall_damage > 0:
                    if int(gmap.material[iy, ix]) == MAT_AIR:
                        cen.c["resolve: a chew finds air (destroyed earlier)"] += 1
                    elif int(gmap.wall_hp[iy, ix]) - wall_fixed.quantize_scalar(
                            float(wall_damage)) <= 0:
                        cen.c["resolve: a chew destroys a wall"] += 1
                    else:
                        cen.c["resolve: a chew dents a wall"] += 1
                return o(gmap, iy, ix, wall_damage)
            return w
        self._wrap(combat, "chew_wall", mk_cw)

        def mk_ap(o):
            def w(unit, packet, events=None, **kw):
                was = unit.alive
                out = o(unit, packet, events, **kw)
                if kw.get("source") == "bullet":
                    if was and not unit.alive:
                        cen.c["resolve: a kill by a round"] += 1
                        cen._kills.append((int(packet.attacker_id), int(unit.id)))
                    elif not was:
                        cen.c["resolve: a round's HIT on a corpse"] += 1
                return out
            return w
        self._wrap(simultaneous, "apply_packet", mk_ap)

        def mk_res(o):
            def w(fx):
                tg = Counter(id(e[1]) for e in fx._pending
                             if e[0] == simultaneous._HIT and e[4] == "bullet")
                if any(v >= 2 for v in tg.values()):
                    cen.c["resolve: two rounds on one target"] += 1
                cen._kills = []
                out = o(fx)
                kills = set(cen._kills)
                if any((v, a) in kills for a, v in kills):
                    cen.c["resolve: two units kill each other"] += 1
                return out
            return w
        self._wrap(simultaneous.SimultaneousEffects, "resolve", mk_res)
        return self

    def __exit__(self, *exc):
        for cont, attr, o in reversed(self._saved):
            setattr(cont, attr, o)
        self._saved.clear()
        return False

    def _classify_slot(self, units):
        from simulation.status import composed_flags
        c = self.c
        for u in units:
            if u.is_zombie or not u.alive:
                if getattr(u, "live_trigger", False):
                    c["4 gate: a zombie or corpse holding the trigger"] += 1
                continue
            if not getattr(u, "live_trigger", False):
                c["4 gate: no trigger"] += 1
                continue
            if not composed_flags(u).can_act:
                c["4 gate: cannot act"] += 1
                continue
            if not getattr(u, "weapon_id", ""):
                c["4 gate: no weapon"] += 1

    def _classify_trigger(self, u, weapon, tick, flags):
        from simulation import combat
        c = self.c
        if tick - u.last_fire_tick < weapon.rof_interval_ticks:
            c["4 gate: the cadence"] += 1
            return None
        st = combat.mag_status(u, weapon, tick)
        if weapon.mag_size <= 0:
            c["4 gate: an untracked magazine"] += 1
        elif u.current_mag == unit_schema.MAG_UNBOUND:
            c["4 gate: the magazine bound at the trigger"] += 1
        elif u.reload_done_tick >= 0:
            if tick < u.reload_done_tick:
                c["4 gate: the reload stall"] += 1
                return None
            c["4 gate: a reload completes at the trigger"] += 1
        if st is not None and st[0] <= 0 and not st[2]:
            c["4 gate: dry"] += 1
            return None
        arch = weapon.archetype
        if arch == "melee":
            c["4: a melee row on the trigger (nothing)"] += 1
        elif arch in ("lobbed", "placed"):
            c["4: a lobbed / placed row on the trigger (nothing)"] += 1
        elif arch == "spray":
            c["4: a spray burst still running" if u.spray_ticks_left > 0
              else "4: a spray burst starts (its noise)"] += 1
        elif arch == "projectile":
            c["4: a projectile trigger (its noise)"] += 1
            if not flags.can_aim:
                c["4: the snap cone (cannot aim)"] += 1
            if u.traits[unit_traits.SPREAD] != ONE:
                c["4: a SPREAD hook"] += 1
            if weapon.shots_per_trigger > 1:
                c["4: a multi-round burst"] += 1
            org = (u.center_tile_x(), u.center_tile_y())
            prev = self._last_origin.get(u.id)
            if prev is not None and prev != org and getattr(u, "move_dir", None) is None:
                c["4: a shot from where the recoil moved the shooter"] += 1
            self._last_origin[u.id] = org
        elif arch == "hitscan":
            c["4: a beam trigger"] += 1
        return (u.current_mag, u.spare_mags)

    def _after_trigger(self, u, weapon, tick, pre):
        if pre is None or weapon.mag_size <= 0:
            return
        if pre[0] > 0 and u.current_mag == 0:
            self.c["4: the magazine empties: the reload begins" if u.reload_done_tick >= 0
                   else "4: the last magazine empties (dry)"] += 1

    def _classify_march(self, b, gmap, units, pre, flying, effects):
        from simulation import simultaneous, unit_fixed
        c = self.c
        rx0, ry0, rem0, bud0, n0 = pre
        h, w = gmap.material.shape
        new = [] if effects is None else effects._pending[n0:]
        kinds = [e[0] for e in new]
        hit = [e for e in new if e[0] == simultaneous._HIT]
        world = [e for e in new if e[0] == simultaneous._WORLD]
        chews = [e for e in world if getattr(e[1], "__name__", "") == "chew_wall"
                 or getattr(getattr(e[1], "__wrapped__", None), "__name__", "") == "chew_wall"]
        pays = [e for e in world if e not in chews]
        firing = rem0 == int(b.weapon.range_tiles)
        c["march: in the firing tick" if firing else "march: a later tick (slot 2')"] += 1
        if not firing:
            if not b.shooter.alive:
                c["2': a round whose shooter died"] += 1
            elif b.shooter.is_zombie:
                c["2': a round of a converted shooter"] += 1
        n_steps = rem0 - b.remaining_steps
        if (bud0 + b.speed_q16) >> 16 == 0:
            c["march: a budget-starved tick (no step)"] += 1
        if flying:
            c["march: still flying"] += 1
        pushes = [e for e in new if e[0] == simultaneous._PUSH]
        hit_unit = None
        if pushes:
            hit_unit = pushes[0][2]
        elif hit:
            hit_unit = hit[0][1]
        if hit_unit is not None:
            c["march: a unit hit"] += 1
            if hit_unit.is_zombie:
                c["march: a zombie hit (the bullet multiplier)"] += 1
            if not hit:
                c["march: a hit without damage (the push only)"] += 1
            if pushes:
                c["march: a push on the hit unit (its lever)" if pushes[0][5]
                  else "march: a push on the hit unit (no lever)"] += 1
            ix, iy = unit_fixed.shr_trunc(b.rx_q), unit_fixed.shr_trunc(b.ry_q)
            others = [e for e in units if e is not hit_unit and e is not b.shooter
                      and e.alive and e.tile_x <= ix < e.tile_x + e.footprint
                      and e.tile_y <= iy < e.tile_y + e.footprint]
            if others:
                c["march: two units on one tile (the first in list order)"] += 1
        elif chews and not flying:
            # a wall stop or a soft cover's absorption
            ix, iy = unit_fixed.shr_trunc(b.rx_q), unit_fixed.shr_trunc(b.ry_q)
            if 0 <= iy < h and 0 <= ix < w and gmap.solid[iy, ix]:
                c["march: a wall stop (its chew)"] += 1
            else:
                c["march: absorbed by soft cover (its chew)"] += 1
        elif not flying and not chews:
            ix, iy = unit_fixed.shr_trunc(b.rx_q), unit_fixed.shr_trunc(b.ry_q)
            if not (0 <= iy < h and 0 <= ix < w):
                c["march: leaves the grid"] += 1
            elif b.remaining_steps == 0:
                c["march: the range ends"] += 1
        if pays:
            c["march: a payload round's stop (PAYLOAD)"] += 1
            ix, iy = unit_fixed.shr_trunc(b.rx_q), unit_fixed.shr_trunc(b.ry_q)
            if not (0 <= iy < h and 0 <= ix < w):
                c["march: a payload stop off the grid (the last tile crossed)"] += 1
        # the walked tiles: the shooter's own footprint, a corpse on the line
        sx, sy = b.sx_q, b.sy_q
        rx, ry = rx0, ry0
        own = dead = False
        for _ in range(max(0, n_steps)):
            rx += sx
            ry += sy
            ix, iy = unit_fixed.shr_trunc(rx), unit_fixed.shr_trunc(ry)
            for e in units:
                if not (e.tile_x <= ix < e.tile_x + e.footprint
                        and e.tile_y <= iy < e.tile_y + e.footprint):
                    continue
                if e is b.shooter:
                    own = True
                elif not e.alive:
                    dead = True
        if own:
            c["march: the shooter's own footprint crossed (excluded)"] += 1
        if dead:
            c["march: a corpse on the line (skipped)"] += 1


#: The branches the shooting scene (A) must take (a 0 fails as vacuous).
SCENE_BRANCHES = (
    "4 gate: a zombie or corpse holding the trigger", "4 gate: no trigger",
    "4 gate: cannot act", "4 gate: no weapon", "4 gate: the cadence",
    "4 gate: the magazine bound at the trigger", "4 gate: the reload stall",
    "4 gate: a reload completes at the trigger", "4 gate: dry",
    "4 gate: an untracked magazine",
    "4: a melee row on the trigger (nothing)",
    "4: a lobbed / placed row on the trigger (nothing)",
    "4: a projectile trigger (its noise)", "4: the snap cone (cannot aim)",
    "4: a SPREAD hook", "4: a multi-round burst",
    "4: a shot from where the recoil moved the shooter",
    "4: the magazine empties: the reload begins", "4: the last magazine empties (dry)",
    "march: in the firing tick", "march: a later tick (slot 2')",
    "2': a round whose shooter died", "2': a round of a converted shooter",
    "march: still flying", "march: a unit hit",
    "march: a zombie hit (the bullet multiplier)",
    "march: a push on the hit unit (its lever)",
    "march: two units on one tile (the first in list order)",
    "march: a wall stop (its chew)", "march: absorbed by soft cover (its chew)",
    "march: leaves the grid", "march: the range ends",
    "march: the shooter's own footprint crossed (excluded)",
    "march: a corpse on the line (skipped)",
    "draw: an exposure roll connects", "draw: an exposure roll absorbs",
    "draw: a crit roll hits", "draw: a crit roll misses",
    "crit: the front arc", "crit: the behind arc", "crit: the flank arc",
    "resolve: a chew destroys a wall", "resolve: a chew dents a wall",
    "resolve: a kill by a round", "resolve: two rounds on one target",
    "resolve: two units kill each other",
)
#: Scene B's (payload rounds, a spray, a budget-starved bolt).
SCENEB_BRANCHES = (
    "4: a spray burst starts (its noise)", "4: a spray burst still running",
    "march: a payload round's stop (PAYLOAD)",
    "march: a payload stop off the grid (the last tile crossed)",
    "march: a budget-starved tick (no step)", "march: a hit without damage (the push only)",
    "march: a later tick (slot 2')",
)
#: Branches neither scene can reach, each with its reason.
UNREACHABLE = {
    "march: physical cover (march_cover) / the facing defense":
        "OnePhaseWEGO only: the round table's encoder refuses a round carrying "
        "them (PART REFUSE)",
    "4: the WEGO targeted / auto-fire branches, start_spray_burst, FACE":
        "WEGO orders only: the door refuses a unit holding orders (PART REFUSE)",
    "4: a beam trigger": "the beam is not built; refused by name (PART REFUSE)",
    "march: a push with no momentum (ammo_push_q 0)":
        "every shipped round with a body has momentum; the march probe takes a "
        "zero row",
    "resolve: a chew finds air (destroyed earlier)":
        "needs two rounds stopping on one wall tile in one resolve AND the "
        "first destroying it; the march probe and PART MARCH's records cover "
        "the CHEW record, the resolve's air no-op is P7a.1's gate",
}


# ---------------------------------------------------------------------------
# The lockstep runner
# ---------------------------------------------------------------------------
def run_lockstep(name, drivers, slots, ticks, census=None, n_units=None, check_alone=True):
    """Shadow ``slots`` on ``drivers`` for ``ticks`` ticks: every tick each
    slot equal (rows, lists, ring, planes, books), its coverage == the rows the
    host changed. Returns counts of the device's records by kind and list."""
    t0 = time.perf_counter()
    cov = Counter()
    kinds = Counter()
    n = 0
    ok = True
    ctx = census if census is not None else _Null()
    with ctx, ShadowLockstep(drivers, list(slots), n_units=n_units,
                             check_alone=check_alone) as ls:
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
                if rep.lists is not None and lab in (SLOT_4.key, SLOT_2R.key):
                    inp = ls.last_inputs[lab].lists
                    for ln, per in rep.lists.items():
                        for e in range(len(drivers)):
                            base = len(inp[ln][e]) if ln in inp and ln == "effects" else 0
                            for r in per[e][base:]:
                                if ln == "effects":
                                    kinds[f"[{lab}] {applier.KINDS[r[0] - 1]}"] += 1
                            if ln == "rounds":
                                kinds[f"[{lab}] rounds at exit"] += len(per[e])
                            if ln == "noise" and lab == SLOT_4.key:
                                kinds[f"[{lab}] noises at exit"] += len(per[e])
                if rep.events is not None:
                    kinds[f"[{lab}] ring events"] += int(rep.events.sum())
            if not ok:
                break
    total = n * len(drivers)
    print(f"{name}: N={len(drivers)} units per env "
          f"{[len(d.sim.units) for d in drivers]} (U {ls.U}), {n} ticks"
          f"{', env i == env i alone' if len(drivers) > 1 and ok and check_alone else ''}; "
          + ", ".join(f"[{lab}] coverage on {100 * cov[lab] / max(1, total):.1f} % of "
                      f"env-ticks" for lab in (s.key for s in slots))
          + f"; records {dict(kinds)} ({time.perf_counter() - t0:.0f} s)")
    return cov, kinds, n


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _check_census(name, census, branches):
    missing = [b for b in branches if census.c[b] == 0]
    print(f"{name} census: {len(branches) - len(missing)} of {len(branches)} required "
          f"branches taken: " + ", ".join(f"{k}={census.c[k]}" for k in branches))
    for b in missing:
        fail(f"{name}: branch never taken: {b!r}")


# ---------------------------------------------------------------------------
# PART TABLES
# ---------------------------------------------------------------------------
def table_reference(sim) -> dict:
    """The P7a.2 device-table rows read from the reference objects."""
    from config import CFG
    from simulation import attack_resolver, combat, unit_fixed, wall_fixed
    from simulation.loadouts import get_tables as lt
    from simulation.species import SPECIES_REGISTRY
    from simulation.unit_schema import TABLES
    T = sim.weapons_tables
    W = [T.weapons.by_name[n] for n in TABLES["weapons"]()]
    A = [T.ammo.by_name[n] for n in TABLES["ammo"]()]
    an = list(TABLES["ammo"]())
    pn = list(T.payloads.by_name)
    arch = {"projectile": 1, "hitscan": 2, "spray": 3, "melee": 4, "lobbed": 5, "placed": 6}

    def ammo_of(w):
        try:
            return an.index(T.ammo_for_weapon(w).name)
        except (KeyError, ValueError):
            return -1
    imp = sim.impulse_table
    q = unit_fixed.quantize_scalar
    cc = CFG.combat
    prof = list(lt().env_profiles.by_name.values())
    spec = [s.environment for s in SPECIES_REGISTRY.values()]
    return {
        "weapon_archetype": tuple(arch[w.archetype] for w in W),
        "weapon_shots_per_trigger": tuple(int(w.shots_per_trigger) for w in W),
        "weapon_range_tiles": tuple(int(w.range_tiles) for w in W),
        "weapon_cone_aim_q": tuple(combat.cone_q16_of_deg(w.spread_deg) for w in W),  # traits-class: the reference read of the class value the table uploads
        "weapon_cone_snap_q": tuple(combat.cone_q16_of_deg(w.spread_snap_deg) for w in W),  # traits-class: as above
        "weapon_ammo": tuple(ammo_of(w) for w in W),
        "weapon_crit_on": tuple(int(w.crit_chance > 0) for w in W),
        "weapon_crit_chance_q": tuple(q(float(w.crit_chance)) for w in W),
        "weapon_crit_mult_q": tuple(q(float(w.crit_mult)) for w in W),
        "weapon_loudness_tiles": tuple(int(getattr(w, "loudness_tiles", 0)) for w in W),
        "weapon_burst_ticks": tuple(int(getattr(w, "burst_ticks", 0)) for w in W),
        "ammo_speed_q16": tuple(int(a.speed_q16) for a in A),
        "ammo_damage": tuple(int(a.damage) for a in A),
        "ammo_dtype": tuple(int(a.dtype_id) for a in A),
        "ammo_ap_q": tuple(q(float(a.ap)) for a in A),
        "ammo_expose_q": tuple(int(getattr(a, "expose_q", 0)) for a in A),
        "ammo_wall_live": tuple(int(a.wall_damage > 0) for a in A),
        "ammo_wall_damage_q": tuple(wall_fixed.quantize_scalar(float(a.wall_damage))
                                    if a.wall_damage > 0 else 0 for a in A),
        "ammo_payload": tuple(pn.index(a.payload) if a.payload else -1 for a in A),
        "ammo_push_q": tuple(int(imp.ammo_push_q.get(a.name, 0)) for a in A),
        "ammo_recoil_q": tuple(int(imp.ammo_recoil_q.get(a.name, 0)) for a in A),
        "impulse_twist_q16": int(imp.twist_q16),
        "zombie_bullet_mult_q": q(float(CFG.zombie.bullet_damage_multiplier)),
        "crit_front_mult_q": q(float(getattr(cc, "crit_front_mult", 1.0))),
        "crit_flank_mult_q": q(float(getattr(cc, "crit_flank_mult", 2.0))),
        "crit_behind_mult_q": q(float(getattr(cc, "crit_behind_mult", 4.0))),
        "arc_front_half_profile_q": tuple(attack_resolver._half_arc_q(
            getattr(p, "front_arc_deg", 120.0)) for p in prof),
        "arc_behind_half_profile_q": tuple(attack_resolver._half_arc_q(
            getattr(p, "behind_arc_deg", 90.0)) for p in prof),
        "arc_front_half_species_q": tuple(attack_resolver._half_arc_q(
            getattr(p, "front_arc_deg", 120.0)) for p in spec),
        "arc_behind_half_species_q": tuple(attack_resolver._half_arc_q(
            getattr(p, "behind_arc_deg", 90.0)) for p in spec),
        "mat_cover_exposure_q": tuple(int(x) for x in sim.gmap.materials.cover_exposure_q16),
    }


def _plain_sim(tile_m, seed=5):
    """A plain hull box at ``tile_m`` metre tiles (a second tile size)."""
    from level_loader import LevelData
    from simulation import Simulation
    from simulation.materials import MAT_AIR, MAT_HULL
    from simulation.ruleset import ContinuousRealtime
    tm = np.full((24, 40), MAT_HULL, dtype=np.int32)
    tm[1:23, 1:39] = MAT_AIR
    level = LevelData(name="p7a2_plain", version="1", path=Path("."), tilemap=tm,
                      tile_size_m=tile_m, diffuse_path=Path("."))
    return Simulation(level, seed=seed, breach_physics=bp, enable_recorder=False,
                      ruleset=ContinuousRealtime())


def part_tables():
    sims = [ShootDriver(kind="a").sim, ShipDriver().sim, _plain_sim(1.0)]
    got = unit_store.DeviceStore(len(sims), 4,
                                 tables=unit_tables.values_batch(sims)).tables.download()
    n = 0
    as_t = lambda x: tuple(x) if isinstance(x, (tuple, list)) else (x,)  # noqa: E731
    for e, sim in enumerate(sims):
        for k, v in table_reference(sim).items():
            dv = unit_tables.read(got[e], k)
            n += 1
            if as_t(dv) != as_t(v):
                fail(f"tables env {e} row {k}: device {dv} != reference {v}")
    rng = [unit_tables.read(got[e], "weapon_range_tiles") for e in (0, 2)]
    push = [unit_tables.read(got[e], "ammo_push_q") for e in (0, 2)]
    if rng[0] == rng[1] or push[0] == push[1]:
        fail("tables: the 1/3 m and the 1 m maps share their ranges / pushes (one tile size?)")
    print(f"tables: {n} P7a.2 rows over 3 envs (the scene at 1/3 m, ship_demo, a 1 m "
          f"box) equal the reference objects (T {unit_tables.layout().width}); "
          f"archetypes {unit_tables.ARCHETYPE_CODES}")


# ---------------------------------------------------------------------------
# PART MARCH (synthetic rounds through the slot-2' kernel, the compaction off)
# ---------------------------------------------------------------------------
MARCH_BRANCHES = (
    "march: a wall stop (its chew)", "march: a unit hit", "march: the range ends",
    "march: leaves the grid", "march: still flying",
    "march: a budget-starved tick (no step)", "march: absorbed by soft cover (its chew)",
    "draw: an exposure roll connects", "draw: an exposure roll absorbs",
    "draw: a crit roll hits", "draw: a crit roll misses", "crit: the front arc",
    "crit: the behind arc", "crit: the flank arc",
    "march: two units on one tile (the first in list order)",
    "march: the shooter's own footprint crossed (excluded)",
    "march: a corpse on the line (skipped)", "march: a payload round's stop (PAYLOAD)",
    "march: a payload stop off the grid (the last tile crossed)",
    "march: a zombie hit (the bullet multiplier)", "2': a round of a converted shooter",
    "2': a round whose shooter died", "march: a hit without damage (the push only)",
)
CRITS = {"p12_whisper": 0.5, "k5_carbine": 0.3, "mp11_pdw": 0.4, "jackhammer_8": 0.2}


def _march_world(sim, rng, h, w, mat_solid, n_mat):
    """Random material / solid planes of every class: mostly air."""
    from simulation.materials import MAT_AIR, MAT_FURNITURE
    m = np.full((h, w), MAT_AIR, dtype=np.int8)
    roll = rng.random((h, w))
    m[roll < 0.10] = MAT_FURNITURE
    other = rng.integers(0, n_mat, size=(h, w)).astype(np.int8)
    sel = (roll >= 0.10) & (roll < 0.16)
    m[sel] = other[sel]
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = MAT_AIR      # the grid's edge open
    solid = np.asarray(mat_solid, dtype=bool)[m.astype(np.int64)]
    return m, solid


def part_march(trials=MARCH_TRIALS):
    import cupy as cp
    from simulation import combat, draws, unit_fixed
    from simulation.simultaneous import SimultaneousEffects
    from simulation.turning import place
    drv = ShootDriver(kind="a")
    sim = drv.sim
    U = len(sim.units)
    h, w = 40, 64
    N = 16
    K = 12
    rng = np.random.default_rng(2911)
    census = Census()
    lay_r = object_tables.table_layout(unit_schema.ROUND_TABLE)
    live = lay_r.live
    n_rec = n_rounds = 0
    with Patched([sim], crit=CRITS), census:
        tab1 = unit_tables.values_batch([sim])
        mat_solid = unit_tables.read(tab1[0], "mat_solid")
        n_mat = len(mat_solid)
        T = sim.weapons_tables
        proj = [wd for wd in T.weapons.by_name.values() if wd.archetype == "projectile"]
        pays = list(T.payloads.by_name.values())
        st = unit_store.DeviceStore(N, U, tables=np.repeat(tab1, N, axis=0))
        fx = applier.DeviceApplier(N, 1024, n_units=U)
        tbl = object_tables.DeviceTable(unit_schema.ROUND_TABLE, N, n_units=U)
        for trial in range(trials):
            rows, pre_tab, want_fx, want_tab, keys, ticks = [], [], [], [], [], []
            mats, sols = [], []
            for e in range(N):
                m, solid = _march_world(sim, rng, h, w, mat_solid, n_mat)
                for i, u in enumerate(sim.units):
                    u.set_pos_q(int(rng.integers(2 * ONE, (w - 5) * ONE)),
                                int(rng.integers(2 * ONE, (h - 5) * ONE)))
                    u.alive = bool(rng.random() > 0.15)
                    u.is_zombie = bool(rng.random() < 0.15)
                    place(u, int(rng.integers(-PI_Q16, PI_Q16)))
                for i in range(0, U - 1, 5):                   # two on one tile
                    sim.units[i + 1].set_pos_q(sim.units[i].x_q, sim.units[i].y_q)
                rows.append(unit_store.pack(sim.units, U))
                seed = int(rng.integers(0, 2 ** 40))
                tick = int(rng.integers(10, 5000))
                dr = draws.MatchDraws(seed)
                dr.set_tick(tick)
                bs = []
                for k in range(K):
                    sh = sim.units[int(rng.integers(0, U))]
                    wd = proj[int(rng.integers(0, len(proj)))]
                    am = T.ammo_for_weapon(wd)
                    if rng.random() < 0.6:                     # from the shooter's centre
                        ox = (sh.tile_x + sh.footprint // 2) << 16
                        oy = (sh.tile_y + sh.footprint // 2) << 16
                    else:
                        ox = int(rng.integers(1 * ONE, (w - 1) * ONE))
                        oy = int(rng.integers(1 * ONE, (h - 1) * ONE))
                    ox = min(max(ox, ONE), (w - 2) * ONE)
                    oy = min(max(oy, ONE), (h - 2) * ONE)
                    ang = int(rng.integers(-PI_Q16, PI_Q16))
                    b = combat.BulletInFlight(sh, sh.id, wd, am, ox, oy, ang,
                                              unit_fixed.cos_q16(ang),
                                              unit_fixed.sin_q16(ang))
                    if rng.random() < 0.3:
                        b.speed_q16 = int(rng.integers(ONE // 4, 3 * ONE))
                    b.remaining_steps = int(rng.integers(0, 90)) if rng.random() < 0.8 \
                        else int(wd.range_tiles)
                    b.budget_q16 = int(rng.integers(0, ONE))
                    b.spawn_tick = tick - int(rng.integers(0, 4))
                    b.ordinal = int(rng.integers(0, 8))
                    if rng.random() < 0.2:
                        b.payload = pays[int(rng.integers(0, len(pays)))]
                    bs.append(b)
                pre_tab.append(object_tables.records(unit_schema.ROUND_TABLE, bs))
                stub = SimpleNamespace(material=m, solid=solid, materials=sim.gmap.materials,
                                       _impulse_table=sim.impulse_table, noise_log=[])
                fake = SimpleNamespace(units=sim.units, gmap=stub,
                                       impulse_table=sim.impulse_table, tick_events=[],
                                       edit_queue=object(), weapons_tables=T)
                eff = SimultaneousEffects()
                flying = [b.advance(stub, sim.units, [], 0.0, dr, events=fake.tick_events,
                                    queue=fake.edit_queue, effects=eff) for b in bs]
                want_fx.append([tuple(int(x) for x in r)
                                for r in applier.records_of(fake, eff._pending)])
                post = object_tables.records(unit_schema.ROUND_TABLE, bs)
                want_tab.append([r[:live] + (int(f),) + r[live + 1:]
                                 for r, f in zip(post, flying)])
                keys.append([dr.key[0], dr.key[1], tick & 0xFFFFFFFF])
                ticks.append(tick)
                mats.append(m)
                sols.append(solid)
            st.upload(np.stack(rows))
            fx.begin()
            tbl.stage_lists([np.array(p, dtype=np.int64).reshape(-1, lay_r.width)
                             for p in pre_tab])
            fields = {"solid": cp.asarray(np.stack(sols)), "material": cp.asarray(np.stack(mats))}
            unit_kernels.rounds_march(st, fx, tbl, np.array(ticks), np.array(keys), fields,
                                      compact=False)
            got_fx, got_tab = fx.records(), tbl.records()
            for e in range(N):
                n_rec += len(want_fx[e])
                n_rounds += len(want_tab[e])
                if got_fx[e] != want_fx[e]:
                    i = next((i for i in range(max(len(got_fx[e]), len(want_fx[e])))
                              if i >= len(got_fx[e]) or i >= len(want_fx[e])
                              or got_fx[e][i] != want_fx[e][i]), -1)
                    fail(f"march trial {trial} env {e}: record {i}: device "
                         f"{got_fx[e][i] if i < len(got_fx[e]) else None} != host "
                         f"{want_fx[e][i] if i < len(want_fx[e]) else None}")
                    return
                if got_tab[e] != want_tab[e]:
                    i = next(i for i in range(len(want_tab[e])) if got_tab[e][i] != want_tab[e][i])
                    names = [c.name for c in lay_r.columns]
                    col = next(c for c in range(lay_r.width)
                               if got_tab[e][i][c] != want_tab[e][i][c])
                    fail(f"march trial {trial} env {e}: round {i} column {names[col]}: "
                         f"device {got_tab[e][i][col]} != host {want_tab[e][i][col]}")
                    return
            if np.asarray(st.download()).tolist() != np.stack(rows).tolist():
                fail(f"march trial {trial}: the march wrote a unit row")
    print(f"march: {trials} trials x {N} envs x {K} synthetic rounds ({n_rounds} rounds, "
          f"{n_rec} records) on random maps of every material: device == host, record "
          f"for record and column for column")
    _check_census("march", census, MARCH_BRANCHES)


# ---------------------------------------------------------------------------
# PART SCENE / SCENEB
# ---------------------------------------------------------------------------
SCENE_SLOTS = [SLOT_2R, SLOT_2L, SLOT_4, SLOT_4C]


def part_scene():
    d = ShootDriver(kind="a")
    census = Census()
    with Patched([d.sim], crit=CRITS, recoil={"rifle_50_ap": 40}):
        cov, kinds, n = run_lockstep("scene", [d], SCENE_SLOTS, SCENE_TICKS, census=census)
    _check_census("scene", census, SCENE_BRANCHES)
    for k in ("[4 shooting] HIT", "[4 shooting] PUSH", "[4 shooting] CHEW",
              "[2' rounds] HIT", "[4 shooting] rounds at exit", "[4 shooting] noises at exit"):
        if kinds[k] == 0:
            fail(f"scene: no {k} (vacuous)")
    print("scene: unreachable, with the reason: " + "; ".join(
        f"{k}: {v}" for k, v in UNREACHABLE.items()))


def part_sceneb():
    d = ShootDriver(kind="b")
    census = Census()
    with Patched([d.sim], speed={"plasma_standard": ONE // 2}):
        cov, kinds, n = run_lockstep("scene B (payloads, a spray)", [d], [SLOT_2R, SLOT_4],
                                     SCENE_TICKS, census=census)
    _check_census("scene B", census, SCENEB_BRANCHES)
    if kinds["[4 shooting] PAYLOAD"] + kinds["[2' rounds] PAYLOAD"] == 0:
        fail("scene B: no PAYLOAD record (vacuous)")


# ---------------------------------------------------------------------------
# PART BATCH, PADDING, OVERFLOW, REFUSE
# ---------------------------------------------------------------------------
def part_batch():
    for n in BATCH_NS:
        drivers = [ShootDriver(variant=v, kind="a") for v in range(n)]
        with Patched([d.sim for d in drivers], crit=CRITS):
            run_lockstep(f"batch N={n}", drivers, SCENE_SLOTS, BATCH_TICKS)


def part_padding():
    """Padding rows filled with copies of live shooting rows (valid cleared):
    the valid rows, the lists and the coverage equal the clean run's, the
    padding rows come back as they went."""
    import cupy as cp
    lay = unit_store.layout()
    drivers = [ShootDriver(variant=0), ShootDriver(variant=1)]
    U = max(len(d.sim.units) for d in drivers) + 6
    done = False
    with ShadowLockstep(drivers, [SLOT_4, SLOT_2R], n_units=U, check_alone=False) as ls:
        for _ in range(40):
            reps = ls.tick()
            if not any(len(x) for x in reps[SLOT_4.key].lists["effects"]):
                continue
            for k, slot in enumerate((SLOT_4, SLOT_2R)):
                clean = np.stack(ls._pre[k])
                inp = ls.last_inputs[slot.key]
                dirty = clean.copy()
                for e in range(2):
                    trig = [r for r in range(U) if clean[e, r, lay.valid]
                            and clean[e, r, lay.column("live_trigger").offset]]
                    for i, r in enumerate(r for r in range(U) if not clean[e, r, lay.valid]):
                        dirty[e, r] = clean[e, trig[(3 * i + e) % len(trig)]]
                        dirty[e, r, lay.valid] = 0
                runs = []
                for rows in (clean, dirty):
                    ls.store.upload(rows)
                    ls._stage_lists(slot, ls.dev_lists, inp)
                    res = slot.call(ls.store, inp, None, None, ls.dev_lists)
                    runs.append((np.asarray(res), ls.store.download(),
                                 {n: ls.dev_lists[n].records() for n in slot.list_names}))
                valid = clean[:, :, lay.valid].astype(bool)
                (c0, g0, l0), (c1, g1, l1) = runs
                if not np.array_equal(g1[valid], g0[valid]):
                    fail(f"padding [{slot.key}]: the valid rows differ")
                if not np.array_equal(g1[~valid], dirty[~valid]):
                    fail(f"padding [{slot.key}]: a padding row was written")
                if l1 != l0:
                    fail(f"padding [{slot.key}]: a list differs")
                if not np.array_equal(c1, c0):
                    fail(f"padding [{slot.key}]: coverage differs")
            n_pad = int((~valid).sum())
            print(f"padding: tick {reps[SLOT_4.key].tick}, N=2, U={U}: {n_pad} padding rows "
                  f"(copies of trigger-holding shooters) untouched; slot 4's and 2''s "
                  f"valid rows and lists equal the clean run")
            done = True
            break
    if not done:
        fail("padding: slot 4 never recorded (vacuous)")


def _one_tick_inputs(drv):
    """The rows, inputs and lists at slot 4's entry of the driver's next tick."""
    from simulation import simulation as simmod, unit_draws
    held = {}
    orig = simmod.process_shooting

    def spy(gmap, units, tick, *a, __o=orig, **kw):
        sim = drv.sim
        held.update(rows=unit_store.pack(units, len(units)), tick=tick,
                    key=unit_draws.draw_key(sim), solid=gmap.solid.copy(),
                    material=gmap.material.copy(),
                    rounds=unit_kernels.rounds_of(sim), noise=unit_kernels.noise_of(sim))
        return __o(gmap, units, tick, *a, **kw)
    simmod.process_shooting = spy
    try:
        drv.tick()
    finally:
        simmod.process_shooting = orig
    return held


def part_overflow():
    import cupy as cp
    from simulation.object_tables import ObjectTableOverflow
    drv = ShootDriver(kind="a")
    sim = drv.sim
    U = len(sim.units)
    tested = set()
    for _ in range(30):
        h = _one_tick_inputs(drv)
        st = unit_store.DeviceStore(1, U, tables=unit_tables.values_batch([sim]))
        fields = {"solid": cp.asarray(h["solid"][None]), "material": cp.asarray(h["material"][None])}

        def run(fxcap=applier.E_CAP, row_cap=applier.ROW_CAP, rcap=None, ncap=None,
                pre_noise=None):
            st.upload(h["rows"][None])
            fx = applier.DeviceApplier(1, fxcap, n_units=U, row_cap=row_cap)
            fx.begin()
            rt = object_tables.DeviceTable(unit_schema.ROUND_TABLE, 1, n_units=U,
                                           row_cap=unit_kernels.round_row_cap(), cap=rcap)
            nt = object_tables.DeviceTable(unit_schema.NOISE_TABLE, 1, n_units=U, cap=ncap)
            rt.begin()
            nt.stage_lists([h["noise"] if pre_noise is None else pre_noise])
            unit_kernels.shooting(st, fx, rt, nt, np.array([h["tick"]]), h["key"][None], fields)
            return fx, rt, nt
        fx, rt, nt = run()
        nfx, nr, nn = len(fx.records()[0]), len(rt.records()[0]), len(nt.records()[0])
        if nfx < 3 or nr < 1 or nn < 2:
            continue
        for what, kw, exc in (("effect list", {"fxcap": 2}, applier.EffectOverflow),
                              ("row staging", {"row_cap": 1}, applier.EffectOverflow),
                              ("round table", {"rcap": max(1, nr - 1)}, ObjectTableOverflow),
                              ("noise list", {"ncap": max(1, nn - 1),
                                              "pre_noise": np.zeros((0, 4), np.int64)},
                               ObjectTableOverflow)):
            try:
                run(**kw)
                fail(f"overflow: the {what} past its capacity was not refused")
            except exc as ex:
                msg = str(ex)
                if "env" not in msg:
                    fail(f"overflow: the {what}'s refusal does not name the env: {msg}")
                tested.add(what)
        print(f"overflow: {nfx} records, {nr} rounds, {nn} noises: the effect list (cap 2), "
              f"a row's staging (1), the round table ({max(1, nr - 1)}) and the noise list "
              f"({max(1, nn - 1)}) each refused by name")
        break
    for t in ("effect list", "row staging", "round table", "noise list"):
        if t not in tested:
            fail(f"overflow: the {t} overflow was never tested (vacuous)")


def part_refuse():
    import cupy as cp
    from simulation import combat
    from simulation.ruleset import OnePhaseWEGO
    # a beam trigger
    d = ShootDriver(kind="a")
    d.who["No"].weapon_id = "lance_3"
    h = _one_tick_inputs(d)
    U = len(d.sim.units)
    st = unit_store.DeviceStore(1, U, tables=unit_tables.values_batch([d.sim]))
    st.upload(h["rows"][None])
    lay = unit_store.layout()
    no = d.sim.units.index(d.who["No"])
    rows = h["rows"].copy()
    rows[no, lay.column("last_fire_tick").offset] = -999
    rows[no, lay.column("live_trigger").offset] = 1
    st.upload(rows[None])
    fx = applier.DeviceApplier(1, applier.E_CAP, n_units=U)
    fx.begin()
    rt = object_tables.DeviceTable(unit_schema.ROUND_TABLE, 1, n_units=U,
                                   row_cap=unit_kernels.round_row_cap())
    nt = object_tables.DeviceTable(unit_schema.NOISE_TABLE, 1, n_units=U)
    rt.begin()
    nt.begin()
    fields = {"solid": cp.asarray(h["solid"][None]), "material": cp.asarray(h["material"][None])}
    ok = 0
    try:
        unit_kernels.shooting(st, fx, rt, nt, np.array([h["tick"]]), h["key"][None], fields)
        fail("refuse: a beam trigger was not refused")
    except unit_kernels.ShotRefused as ex:
        if "beam" in str(ex):
            ok += 1
        if not np.array_equal(st.download()[0][no], rows[no]):
            fail("refuse: the refused beam shooter's row was written")
    # WEGO orders
    from simulation.orders import Order
    sim = d.sim
    u = d.who["Nt"]
    try:
        u.orders.append(Order(order_type=0, phase=0, target_fx=1.0, target_fy=1.0))
    except TypeError:
        u.orders.append(SimpleNamespace(order_type=0, phase=0))
    try:
        unit_kernels.rounds_of(sim)
        fail("refuse: a unit holding WEGO orders was not refused")
    except unit_kernels.ShotRefused:
        ok += 1
    u.orders.clear()
    # a ruleset that is not continuous
    rs = sim.ruleset
    sim.ruleset = OnePhaseWEGO.__new__(OnePhaseWEGO)
    try:
        unit_kernels.shooting_door(sim)
        fail("refuse: OnePhaseWEGO was not refused")
    except unit_kernels.ShotRefused:
        ok += 1
    sim.ruleset = rs
    # a round under physical cover / the facing defense
    b = combat.BulletInFlight(u, u.id, sim.weapons_tables.weapons.by_name["k5_carbine"],
                              sim.weapons_tables.ammo.by_name["rifle_556_standard"],
                              ONE * 5, ONE * 5, 0, ONE, 0)
    b.spawn_tick = sim.tick
    for attr, val in (("march_cover", []), ("march_facing_defense", True)):
        setattr(b, attr, val)
        try:
            object_tables.records(unit_schema.ROUND_TABLE, [b])
            fail(f"refuse: a round with {attr} was not refused")
        except ValueError:
            ok += 1
        setattr(b, attr, None if attr == "march_cover" else False)
    print(f"refuse: {ok} of 5 named refusals (a beam trigger, WEGO orders, OnePhaseWEGO, "
          f"physical cover, the facing defense)")
    if ok != 5:
        fail("refuse: a refusal is missing")


# ---------------------------------------------------------------------------
# PART CHAINED, SHIP
# ---------------------------------------------------------------------------
def _books_host_side(sim, pre, post):
    """What the chained slots leave on the host besides rows: the pushed
    units' momentum books (REFERENCE_ONLY; unit_impulse.add's IN entries) and
    the solid books / the tile serial of the chews' destroys."""
    from simulation.unit_impulse import IN_X, IN_Y
    lay = unit_store.layout()
    ix, iy = lay.column("imp_x").offset, lay.column("imp_y").offset
    for r, u in enumerate(sim.units):
        u.impulse_books[IN_X] += int(post[r, ix] - pre[r, ix])
        u.impulse_books[IN_Y] += int(post[r, iy] - pre[r, iy])


class _HostCalls:
    """Counts the host law's calls in the twin (the chained gate requires 0)."""

    def __init__(self, sim):
        self.sim, self.n, self._saved = sim, Counter(), []

    def __enter__(self):
        from simulation import combat
        hc = self
        for cont, attr in ((combat.BulletInFlight, "advance"), (combat, "fire_burst"),
                           (combat, "_directional_fire")):
            o = getattr(cont, attr)

            def w(*a, __o=o, __n=attr, **k):
                if any(x is hc.sim.units or x is hc.sim.gmap for x in a):
                    hc.n[__n] += 1
                return __o(*a, **k)
            w.__wrapped__ = o
            setattr(cont, attr, w)
            self._saved.append((cont, attr, o))
        return self

    def __exit__(self, *exc):
        for cont, attr, o in reversed(self._saved):
            setattr(cont, attr, o)
        return False


def _host_side_of(holder):
    """The chained slots' host-only side: the impulse books, and for a chew
    that destroyed walls the books landed (the solid book through
    _solid_energy_book), the serial, the sky flag, the render marks."""
    def hs(sim, pre, post):
        _books_host_side(sim, pre, post)
        ch = holder["ch"]
        b = ch.books
        if not b:
            return
        g = sim.gmap
        g.n_destruction_seed_sum += b["n_destruction_seed"]
        for name, key in (("ge_retire", "retire"), ("ge_mint", "mint"),
                          ("ge_destroy_seed", "destroy_seed")):
            g._gas_energy_book(key, b[name])
        g._solid_energy_book("destroy_wall", b["solid_destroy_wall"])
        if b["tiles_changed"]:
            g._tile_serial += b["tiles_changed"]
            g._sky_mask_dirty = True
            before = ch.inputs._host_fields["material"][0]
            for y, x in np.argwhere(before != g.material).tolist():
                g.render_dirty.add((int(y), int(x)))
    return hs


def part_chained():
    twin, ref = ShootDriver(kind="a"), ShootDriver(kind="a")
    t0 = time.perf_counter()
    n = 0
    h2, h4 = {}, {}
    with Patched([twin.sim, ref.sim], crit=CRITS):
        with _HostCalls(twin.sim) as hc,                 ChainedSlots(twin, [SLOT_2R, SLOT_2L], host_side=_host_side_of(h2)) as c2,                 ChainedSlots(twin, [SLOT_4, SLOT_4A, SLOT_4C],
                             host_side=_host_side_of(h4)) as c4:
            h2["ch"], h4["ch"] = c2, c4
            for _ in range(CHAIN_TICKS):
                twin.tick()
                ref.tick()
                n += 1
                if not _same(twin.sim, ref.sim):
                    fail(f"chained: the twin left the reference at tick {twin.sim.tick}")
                    break
    recs = sum(len(x["effects"]) for x in c4.lists) + sum(len(x["effects"]) for x in c2.lists)
    rounds = sum(len(x["rounds"]) for x in c2.lists)
    walls = int(ref.sim.gmap._tile_serial)
    print(f"chained (2' + landing; 4 + 4a + 4c): {n} ticks, the twin equal to the "
          f"reference every tick (unit hash, field digest, rounds in flight, noise log); "
          f"{c2.runs} + {c4.runs} chain runs, {recs} effect records walked, {rounds} "
          f"rounds carried by 2', tile serial {walls}; the host law ran "
          f"{sum(hc.n.values())} times in the twin ({time.perf_counter() - t0:.0f} s)")
    if c2.runs != n or c4.runs != n or recs == 0 or rounds == 0:
        fail("chained: vacuous")
    if sum(hc.n.values()):
        fail(f"chained: the host law ran in the twin: {dict(hc.n)}")
    if twin.sim.gmap._tile_serial != ref.sim.gmap._tile_serial:
        fail("chained: the tile serial differs (the host side of a destroy)")


LATER = "march: a later tick (slot 2')"


def part_ship():
    d = ShipDriver()
    census = Census()
    cov, kinds, n = run_lockstep("ship_demo (bots)", [d], SCENE_SLOTS, SHIP_TICKS,
                                 census=census)
    shots = census.c["4: a projectile trigger (its noise)"]
    hits = census.c["march: a unit hit"]
    kills = census.c["resolve: a kill by a round"]
    print(f"ship_demo: {n} ticks, {shots} triggers, {hits} rounds hit a unit, {kills} "
          f"kills by rounds; {census.c[LATER]} marches of rounds still flying")
    if shots == 0:
        fail("ship_demo: no shot in the match (vacuous)")


# ---------------------------------------------------------------------------
# Main, bench
# ---------------------------------------------------------------------------
PARTS = ("tables", "march", "scene", "sceneb", "batch", "padding", "overflow", "refuse",
         "chained", "ship")


def main(parts=None) -> int:
    t0 = time.perf_counter()
    want = parts or os.environ.get("UNIT_SHOOT_PARTS", ",".join(PARTS)).split(",")
    for p in want:
        globals()[f"part_{p}"]()
    print(f"total {time.perf_counter() - t0:.0f} s")
    ok = not FAILS
    print(f"UNIT_SHOOT_RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def bench(reps: int = 50) -> int:
    """The TIMING table: median ms of slot 4's shooting and of slot 2' (the
    march, then the landing) at N = 1 / 8 / 64 envs of the shooting scene."""
    import cupy as cp
    drv = ShootDriver(kind="a")
    for _ in range(4):
        h = _one_tick_inputs(drv)
    sim = drv.sim
    U = len(sim.units)
    tab1 = unit_tables.values_batch([sim])
    lay = unit_store.layout()
    rows1 = h["rows"].copy()
    rows1[:, lay.column("last_fire_tick").offset] = -999        # every trigger fires
    # a full in-flight table for slot 2': this tick's rounds of the scene
    st1 = unit_store.DeviceStore(1, U, tables=tab1)
    st1.upload(rows1[None])
    f1 = {"solid": cp.asarray(h["solid"][None]), "material": cp.asarray(h["material"][None])}
    fx1 = applier.DeviceApplier(1, 1024, n_units=U)
    fx1.begin()
    rt1 = object_tables.DeviceTable(unit_schema.ROUND_TABLE, 1, n_units=U,
                                    row_cap=unit_kernels.round_row_cap())
    nt1 = object_tables.DeviceTable(unit_schema.NOISE_TABLE, 1, n_units=U)
    rt1.begin()
    nt1.begin()
    unit_kernels.shooting(st1, fx1, rt1, nt1, np.array([h["tick"]]), h["key"][None], f1)
    flying = np.array(rt1.records()[0], dtype=np.int64)
    lines = []
    for n_env in (1, 8, 64):
        st = unit_store.DeviceStore(n_env, U, tables=np.repeat(tab1, n_env, axis=0))
        rows = np.repeat(rows1[None], n_env, axis=0)
        fields = {"solid": cp.asarray(np.repeat(h["solid"][None], n_env, axis=0)),
                  "material": cp.asarray(np.repeat(h["material"][None], n_env, axis=0))}
        ticks = np.full(n_env, h["tick"], dtype=np.int64)
        keys = np.repeat(h["key"][None], n_env, axis=0)
        fx = applier.DeviceApplier(n_env, 1024, n_units=U)
        rt = object_tables.DeviceTable(unit_schema.ROUND_TABLE, n_env, n_units=U,
                                       row_cap=unit_kernels.round_row_cap())
        nt = object_tables.DeviceTable(unit_schema.NOISE_TABLE, n_env, n_units=U)
        ring = unit_events.EventRing(n_env, U, e_max=4096)
        from simulation.applier import PLANE_NAMES
        planes = {n: cp.asarray(np.repeat(np.array(getattr(sim.gmap, n))[None], n_env, axis=0))
                  for n in PLANE_NAMES}
        ts4, ts2, ts2l = [], [], []
        for _ in range(reps):
            st.upload(rows)
            fx.begin()
            rt.begin()
            nt.begin()
            cp.cuda.Stream.null.synchronize()
            t = time.perf_counter()
            unit_kernels.shooting(st, fx, rt, nt, ticks, keys, fields)
            ts4.append((time.perf_counter() - t) * 1e3)
            st.upload(rows)
            fx.begin()
            rt.stage_lists([flying] * n_env)
            ring.begin()
            cp.cuda.Stream.null.synchronize()
            t = time.perf_counter()
            unit_kernels.rounds_march(st, fx, rt, ticks, keys, fields)
            ts2.append((time.perf_counter() - t) * 1e3)
            t = time.perf_counter()
            try:
                unit_kernels.resolve(st, fx, planes, ring)
            except (applier.EffectRefused, applier.ApplierFault):
                pass       # #29 P7c.1: a payload round's PAYLOAD faults without its lists
            ts2l.append((time.perf_counter() - t) * 1e3)
        lines.append(("4 shooting", n_env, f"{U} rows", float(np.median(ts4))))
        lines.append(("2' march", n_env, f"{len(flying)} rounds", float(np.median(ts2))))
        lines.append(("2' landing", n_env, "its records", float(np.median(ts2l))))
    print("TIMING (median ms, the call incl. its count / list download):")
    for name, n_env, what, ms in lines:
        print(f"  {name:11s} N={n_env:3d} {what:>11s}: {ms:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
