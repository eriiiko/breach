"""#29 P7a.3 -- THE SPRAYS GATE: slot 4b (``combat.process_sprays``) on the
device unit store, THE CELL BUFFER, the SPRAY_CELLS effect at the resolve
(``combat._enqueue_spray_cells``), THE DEVICE EDIT DOOR step (a) and THE
CLAIMS LIST -- each against its Python law at tol 0.

Runs inside the GPU subprocess (tests/cuda_harness.run_cuda_script; wrapper
tests/test_cuda_unit_sprays.py). Slot 4b is ``cpp/src/cuda_unit_sprays.cu``
over ``cpp/src/cuda_unit_spray.cuh`` (the cone over the PHYSICAL device sight,
the SPRAY_CELLS effect) and ``cpp/src/cuda_edit_door.cuh`` (the door, the
claims); the Python faces ``simulation.unit_kernels.sprays`` / ``resolve``
(with ``WalkLists``), ``simulation.attack_lists``, ``simulation.edit_door``;
the harness ``tests/cuda_unit_lockstep.py`` (the lists ``spray_cells``,
``edits``, ``claims``; the input ``edit_seq``).

PART TABLES -- every P7a.3 table row equals an independent read of the sim's
own weapon / ammo rows, its gas table, combat's source ids and FIELD_POLICY.

PART CONE -- the cone (``cuda_unit_spray.cuh``'s decide / emit, the TEST-ONLY
probe) on synthetic cones over four real maps (ship_demo, playground,
my_ship, test_level): apexes everywhere (walls included), aims over the
whole circle and on the axes, ranges 0 .. 60, half-angles from 0 to just
under 90 degrees, the nozzle square of side 0 .. 5 near the apex; the tile
SET and ORDER (and every divisor) equal ``combat.spray_cone_tiles``; both
answers of the physical sight seen often.

PART SCENE -- the spray scene built for the gate (no shipped level holds a
spray): every shipped spray weapon (dragon_7, miasma_vent, dragon_9_heavy),
two Dragons whose cones overlap (the second's claims lose to the first's),
a spray into a wood wall (occluded tiles), a sprayer killed by a fixture
mid-burst, a sprayer killed by a melee blow in the tick it deposits, a burst
that ends by its tick count, a stunned sprayer (the interruption), a heavy
Dragon whose box the grid edge clips, an enemy the jets' claims credit with
its death (slot 9c), and a carbine whose rounds destroy a glass wall at
the resolve BEFORE a spray whose cells hold it. Slots 4b and 4c SHADOWED:
4b's rows, its SPRAY_CELLS records and the cell buffer equal the host's; 4c
fed the host's list: rows, ring, planes, books, the door's rows (equal to
``pack_for_device`` of the host's edits, absolute seq) and the claims list
(every claim, in order) equal. A census of the Python law's branches.

PART BATCH -- N = 2 and 8 scenes differing in layout, seed, unit count and
fields: env i == env i alone. PART PADDING -- live-looking padding rows stay
untouched and record nothing. PART OVERFLOW -- the cell buffer, the effect
list, the claims list and the door's rows each past their capacity: counted
and refused by name. PART REFUSE -- a WEGO burst (no aim angle) and an
interleaved host producer refused by name. PART CHAINED -- 4b + 4c replaced
by the device's in a running scene: the unit hash and field digest equal an
untouched reference every tick (the door's rows landed through the host's
enqueue, the claims through TickCredit.claim, the walls' host side), the
host spray law run 0 times in the twin. PART SHIP -- ship_demo's bots match
(4b + 4c) for SHIP_TICKS ticks.

PROPERTY: slot 4b, the cell buffer, SPRAY_CELLS, the door's rows and the
claims list are the Python's integer for integer, record for record and in
the reference's order, per env. BREAKS IF the cone's root, angle test,
range test, nozzle rule, sight form or tile order, a spray row, the
countdown or a clear, the SPRAY_CELLS record or its span, the cell buffer's
append, an edit's amount, field, slice, source or key, the claims' order or
channel, or the batch indexing differs from the reference. Each validated by
breaking it (the patch's report).

Prints per-part lines and ``UNIT_SPRAYS_RESULT: PASS`` / ``FAIL``.
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
from simulation import (applier, attack_lists, edit_door, unit_kernels,  # noqa: E402
                        unit_store, unit_tables)

FAILS: list = []
PI_Q16 = 205887
SCENE_TICKS = int(os.environ.get("UNIT_SPRAYS_TICKS", "130"))
SHIP_TICKS = int(os.environ.get("UNIT_SPRAYS_SHIP_TICKS", "500"))
BATCH_TICKS = int(os.environ.get("UNIT_SPRAYS_BATCH_TICKS", "50"))
BATCH_NS = [int(x) for x in os.environ.get("UNIT_SPRAYS_BATCH_NS", "2,8").split(",")]
CHAIN_TICKS = int(os.environ.get("UNIT_SPRAYS_CHAIN_TICKS", "130"))
CONE_QUERIES = int(os.environ.get("UNIT_SPRAYS_CONE_QUERIES", "1500"))


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
    ("damage_credit", "the TickCredit dict: its device form is the claims list "
                      "(list:claims, every claim in order)"),
)


def _run_4b(st, inp, lists):
    return unit_kernels.sprays(st, lists["effects"], lists["spray_cells"], inp.fields)


def _run_4c(st, inp, ring, lists):
    lists["edits"].set_seq(inp.scalars["edit_seq"])
    return unit_kernels.resolve(
        st, lists["effects"], inp.fields, ring,
        unit_kernels.WalkLists(cells=lists["spray_cells"], door=lists["edits"].door,
                               claims=lists["claims"]), tick_row=getattr(inp, "tick_row", None))


def _conductor(sim, args, kw, caller):
    """Slot 4c is the conductor's resolve, never the rounds' landing."""
    return caller == "_step_pre_physics"


SLOT_4B = DeviceSlot(label="4b sprays", run=_run_4b,
                     inputs=("list:effects", "list:spray_cells", "field:solid"),
                     outputs=("list:effects", "list:spray_cells"))
SLOT_4C = DeviceSlot(label="4c resolve", run=_run_4c,
                     inputs=("list:effects", "list:spray_cells", "edit_seq"),
                     events=True,
                     outputs=PLANE_OUTPUTS + BOOK_OUTPUTS + ("list:edits", "list:claims"),
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


# ---------------------------------------------------------------------------
# The spray scene
# ---------------------------------------------------------------------------
SPACE = level_loader.SPACE_CODE
EAST, WEST, SOUTH, NORTH = 0, PI_Q16, -PI_Q16 // 2, PI_Q16 // 2
SH, SW = 40, 64


def _spray_level(variant):
    """A hull box in space (1 m tiles), interior air; a wood wall in front of
    the wall-sprayer, a glass wall the carbine chews; odd variants an extra
    furniture block, the walls' rows move with the variant."""
    from level_loader import LevelData
    from simulation.materials import (MAT_AIR, MAT_FURNITURE, MAT_HULL,
                                      MAT_GLASS, MAT_WOOD)
    dy = variant % 3
    tm = np.full((SH, SW), SPACE, dtype=np.int32)
    tm[1:SH - 1, 1:SW - 1] = MAT_HULL
    tm[2:SH - 2, 2:SW - 2] = MAT_AIR
    tm[10 + dy:17 + dy, 9] = MAT_WOOD                 # G2: the wall in D's cone
    tm[29 + dy:34 + dy, 27] = MAT_GLASS               # G6: the glass K chews, in H's cone
    if variant % 2:
        tm[22:25, 44:48] = MAT_FURNITURE
    return LevelData(name=f"p7a3_spray{variant}", version="3", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."),
                     boundary="space")


class SprayDriver:
    """The spray scene (see the module docstring), scripted: the hosing
    sprayers and the carbine hold their triggers every tick; fixtures
    between ticks (a kill, a stun, the striker's latch). ``variant`` > 0:
    another seed and layout, bystanders, every group shifted."""

    def __init__(self, variant=0):
        from simulation import Simulation
        from simulation.ruleset import ContinuousRealtime
        self.variant, self.k = variant, 0
        sim = Simulation(_spray_level(variant), seed=20261116 + 11 * variant,
                         breach_physics=bp, enable_recorder=False,
                         ruleset=ContinuousRealtime())
        sim.set_paused(False)
        self.sim = sim
        # the ids start past the rows (a unit's id is never its row index here:
        # a record field written from the row index instead of the id is seen)
        sim._next_unit_id += 3 + variant
        self.who = {}
        self.hosing = []
        if variant % 2:
            self._bystanders()
        self._setup()
        if not variant % 2:
            self._bystanders()

    def _bystanders(self):
        for i in range(self.variant % 4):
            self._add(f"by{i}", 44 + 4 * i, 4, facing=SOUTH, weapon="k5_carbine")

    def _add(self, name, x, y, team=0, facing=EAST, weapon="dragon_7", hp=10000.0,
             hose=False, melee="rifle_butt"):
        from simulation.turning import place
        from simulation.unit import Unit
        u = Unit(name, x=float(x), y=float(y + self.variant % 3), team=team)
        self.sim.add_unit(u)
        place(u, facing)
        u.current_hp = float(hp)
        u.weapon_id = weapon
        u.melee_id = melee
        self.who[name] = u
        if hose:
            self.hosing.append(u)
        return u

    def _setup(self):
        a = self._add
        # G1: two Dragons whose cones overlap; an enemy in both (G5: its death
        # credited through the claims)
        a("A", 3, 4, facing=EAST, hose=True)
        a("C", 15, 4, facing=WEST, hose=True)
        a("T", 9, 4, team=1, facing=NORTH, weapon="k5_carbine", hp=40.0)
        # G2: a spray into the wood wall
        a("D", 4, 12, facing=EAST, hose=True)
        # G3a: a miasma sprayer (gas only) killed by a fixture mid-burst
        a("E", 20, 4, facing=EAST, weapon="miasma_vent", hose=True)
        # G3b: a sprayer a melee blow kills in the tick it deposits
        a("F", 22, 12, facing=EAST, hp=1.0, hose=True)
        a("M", 25, 12, team=1, facing=WEST, weapon="k5_carbine")
        # G4: a burst that ends by its count (one trigger only)
        a("G", 40, 4, facing=SOUTH)
        # G6: a carbine chewing the glass wall, a Dragon whose cone holds it
        a("K", 31, 30, facing=WEST, weapon="k5_carbine", hose=True)
        a("H", 20, 30, facing=EAST, hose=True)
        # G7: a sprayer stunned mid-burst (the interruption)
        a("I", 40, 18, facing=WEST, hose=True)
        # G8: a heavy Dragon whose box the grid edge clips
        a("J", 54, 30, facing=EAST, weapon="dragon_9_heavy", hose=True)

    def _before(self):
        from simulation.damage import DamagePacket, apply_packet
        from simulation.status import STUNNED, apply_status
        sim, w, k = self.sim, self.who, self.k
        for u in self.hosing:
            if u.alive:
                sim.set_trigger(u.id, True)
        sim.set_trigger(w["G"].id, k == 0)
        if k == 20 and w["E"].alive:
            apply_packet(w["E"], DamagePacket(amount=1.0e6, dtype=0, attacker_id=-1), [],
                         source="melee")
        if 10 <= k <= 12 and w["I"].alive:
            apply_status(w["I"], STUNNED, 0, 3)
        if w["M"].alive:
            sim.melee_intent(w["M"].id)
        if k == 2:
            # G6: the glass's hp down to one count, so the carbine's next chew
            # destroys it at the resolve, BEFORE H's spray record (a fixture of
            # the wall's own hp: a topology value, not a FieldEdit field)
            dy = self.variant % 3
            for y in range(29 + dy, 34 + dy):
                if sim.gmap.wall_hp[y, 27] > 0:
                    sim.gmap.wall_hp[y, 27] = 1

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
# The census: the host's own decisions in 4b, the enqueue, the resolve, 9c
# ---------------------------------------------------------------------------
class Census:
    """Wraps slot 4b (``simulation.simulation.process_sprays``), the spray
    enqueue (``combat._enqueue_spray_cells``) and the resolve -- reads only --
    and counts every branch the reference took, by the reference's terms."""

    def __init__(self):
        self.c = Counter()
        self._saved = []
        self.sprayer_ids = set()

    def __enter__(self):
        from simulation import combat, simultaneous
        from simulation import simulation as simmod
        census = self
        o4b = simmod.process_sprays

        def w4b(gmap, units, queue, events=None, effects=None, __o=o4b):
            census._classify_4b(gmap, units)
            return __o(gmap, units, queue, events=events, effects=effects)
        w4b.__wrapped__ = o4b
        simmod.process_sprays = w4b
        self._saved.append((simmod, "process_sprays", o4b))

        oenq = combat._enqueue_spray_cells

        def wenq(gmap, queue, owner, heat_q, gas_id, gas_q, cells, __o=oenq):
            from simulation.damage import CREDIT_HEAT, tick_credit
            own = tick_credit(gmap)._owner
            c = census.c
            for (y, x, _d) in cells:
                if heat_q > 0:
                    c["enqueue: a heat edit"] += 1
                    c["enqueue: a heat claim lost to an earlier claim"
                      if (CREDIT_HEAT, y, x) in own else "enqueue: a heat claim kept"] += 1
                if gas_id is not None and gas_q > 0:
                    c["enqueue: a gas edit"] += 1
                    c["enqueue: a gas claim lost to an earlier claim"
                      if (gas_id, y, x) in own else "enqueue: a gas claim kept"] += 1
            return __o(gmap, queue, owner, heat_q, gas_id, gas_q, cells)
        wenq.__wrapped__ = oenq
        combat._enqueue_spray_cells = wenq
        self._saved.append((combat, "_enqueue_spray_cells", oenq))

        ores = simultaneous.SimultaneousEffects.resolve

        def wres(fx, __o=ores):
            census._classify_resolve(fx, __o)
        wres.__wrapped__ = ores
        simultaneous.SimultaneousEffects.resolve = wres
        self._saved.append((simultaneous.SimultaneousEffects, "resolve", ores))
        return self

    def __exit__(self, *exc):
        for cont, attr, o in reversed(self._saved):
            setattr(cont, attr, o)
        self._saved.clear()
        return False

    def _classify_4b(self, gmap, units):
        from simulation import combat
        from simulation.status import composed_flags
        c = self.c
        cones = []
        for u in units:
            tl = getattr(u, "spray_ticks_left", 0)
            if tl <= 0:
                continue
            if not u.alive:
                c["4b: a dead sprayer's burst cleared"] += 1
                continue
            if not composed_flags(u).can_act:
                c["4b: an interrupted burst cleared"] += 1
                if u.spray_order is not None and u.spray_order in u.orders:
                    c["4b: interrupted, its WEGO fire order consumed"] += 1
                continue
            self.sprayer_ids.add(int(u.id))
            weapon = combat.weapon_tables().weapons.by_name[u.weapon_id]
            ammo = combat.weapon_tables().ammo_for_weapon(weapon)
            c["4b: a deposit"] += 1
            heat = float(ammo.heat_deposit) > 0
            gas = bool(ammo.gas_species) and float(ammo.gas_amount) > 0
            c["4b: a deposit with heat"] += int(heat)
            c["4b: a deposit with gas"] += int(gas)
            c["4b: a gas-only deposit"] += int(gas and not heat)
            if tl - 1 <= 0:
                c["4b: a burst ends by its count"] += 1
            sight = Counter()
            orig = gmap.has_los

            def counting(*a, __o=orig, **k):
                r = __o(*a, **k)
                sight[bool(r)] += 1
                return r
            gmap.has_los = counting
            try:
                cells = list(combat.spray_cone_tiles(
                    gmap, u.center_tile_y(), u.center_tile_x(), 0.0, 0.0,
                    weapon.range_tiles, weapon.cone_half_angle_degrees,
                    exclude=set(u.occupied_tiles()), aim_q=u.spray_aim_angle))
            finally:
                del gmap.has_los
            c["4b: a cone tile in sight"] += sight[True]
            c["4b: a cone tile occluded (physical sight False)"] += sight[False]
            if not cells:
                c["4b: a cone with zero cells"] += 1
            h, w = gmap.material.shape
            ay, ax, r = u.center_tile_y(), u.center_tile_x(), weapon.range_tiles
            if ay - r < 0 or ax - r < 0 or ay + r > h - 1 or ax + r > w - 1:
                c["4b: a cone box clipped at the grid edge"] += 1
            cones.append({(y, x) for y, x, _d in cells})
        for i in range(len(cones)):
            for j in range(i + 1, len(cones)):
                if cones[i] & cones[j]:
                    c["4b: two cones overlap in one tick"] += 1

    def _classify_resolve(self, fx, orig):
        from simulation import combat, simultaneous
        c = self.c
        sprays = []          # (index, owner unit id, cell set)
        gmap = None
        for i, e in enumerate(fx._pending):
            if e[0] == simultaneous._WORLD and applier._unwrap(e[1]) is \
                    applier._unwrap(combat._enqueue_spray_cells):
                gmap = e[2][0]
                sprays.append((i, int(e[2][2]), {(y, x) for y, x, _d in e[2][6]}))
        units = {}
        alive0 = {}
        if gmap is not None:
            before = gmap.material.copy()
        # the chews that will land before each spray
        for i, e in enumerate(fx._pending):
            if e[0] == simultaneous._HIT:
                units[id(e[1])] = e[1]
                alive0[id(e[1])] = e[1].alive
        orig(fx)
        if gmap is None:
            return
        gone = {(int(y), int(x)) for y, x in np.argwhere((before != 0) & (gmap.material == 0))}
        for i, owner, cells in sprays:
            if gone & cells:
                c["resolve: a CHEW destroyed a wall that a later spray's cells hold"] += 1
        for k, u in units.items():
            if alive0[k] and not u.alive and int(u.id) in {o for _i, o, _c in sprays}:
                c["4b: a sprayer killed in this tick's resolve deposited"] += 1

    def after_tick(self, sim):
        from simulation.events import UnitKilledEvent
        for ev in sim.tick_events:
            if isinstance(ev, UnitKilledEvent) and ev.killed_by == "heat" \
                    and int(ev.attacker_id) in self.sprayer_ids:
                self.c["9c: a kill credited to a sprayer through its claims"] += 1


#: The branches the spray scene must take (a 0 fails as vacuous).
SCENE_BRANCHES = (
    "4b: a deposit", "4b: a deposit with heat", "4b: a deposit with gas",
    "4b: a gas-only deposit", "4b: a dead sprayer's burst cleared",
    "4b: an interrupted burst cleared", "4b: a burst ends by its count",
    "4b: a cone tile in sight", "4b: a cone tile occluded (physical sight False)",
    "4b: a cone box clipped at the grid edge", "4b: two cones overlap in one tick",
    "4b: a sprayer killed in this tick's resolve deposited",
    "enqueue: a heat edit", "enqueue: a gas edit", "enqueue: a heat claim kept",
    "enqueue: a heat claim lost to an earlier claim", "enqueue: a gas claim kept",
    "enqueue: a gas claim lost to an earlier claim",
    "resolve: a CHEW destroyed a wall that a later spray's cells hold",
    "9c: a kill credited to a sprayer through its claims",
)
#: Branches the scene cannot reach, each with its reason.
UNREACHABLE = {
    "4b: interrupted, its WEGO fire order consumed":
        "a continuous burst (start_spray_burst_directional) rides no fire order; "
        "start_spray_burst is WEGO, out of scope and refused at the device door",
    "4b: a cone with zero cells":
        "every shipped spray reaches >= 7 tiles and the physical sight never tests "
        "the end tile, so a wall face is always yielded; PART CONE takes range 0 and "
        "the narrow half-angles",
    "enqueue: a claim by the environment (owner -1)":
        "a sprayer is a unit with an id >= 0; the claims list's writer drops -1 like "
        "TickCredit.claim (its code is the door's claim())",
}


# ---------------------------------------------------------------------------
# The lockstep runner
# ---------------------------------------------------------------------------
class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def after_tick(self, sim):
        pass


def run_lockstep(name, drivers, slots, ticks, census=None):
    """Shadow ``slots`` on ``drivers`` for ``ticks`` ticks: every tick each slot
    equal, its coverage == the rows the host changed. Returns the counts of
    SPRAY_CELLS records, cells, door edits and claims compared."""
    t0 = time.perf_counter()
    tot = Counter()
    n = 0
    ok = True
    ctx = census if census is not None else _Null()
    with ctx, ShadowLockstep(drivers, list(slots)) as ls:
        for _ in range(ticks):
            if any(d.is_terminal() for d in drivers):
                break
            reps = ls.tick()
            for d in drivers:
                ctx.after_tick(d.sim)
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
                if lab == SLOT_4B.label and rep.lists is not None:
                    tot["4b acted (env-ticks)"] += int((rep.coverage > 0).sum())
                    for e in range(len(drivers)):
                        tot["SPRAY_CELLS"] += sum(1 for r in rep.lists["effects"][e]
                                                  if r[0] == applier.KIND_SPRAY_CELLS)
                        tot["cells"] += len(rep.lists["spray_cells"][e])
                if lab == SLOT_4C.label and rep.lists is not None:
                    for e in range(len(drivers)):
                        tot["door edits"] += len(rep.lists["edits"][e])
                        tot["claims"] += len(rep.lists["claims"][e])
                    tot["ring events (4c)"] += int(rep.events.sum())
            if not ok:
                break
    print(f"{name}: N={len(drivers)} units per env {[len(d.sim.units) for d in drivers]} "
          f"(U {ls.U}), {n} ticks{', env i == env i alone' if len(drivers) > 1 and ok else ''}"
          f"; compared {dict(tot)} ({time.perf_counter() - t0:.0f} s)")
    return tot


# ---------------------------------------------------------------------------
# PART TABLES
# ---------------------------------------------------------------------------
def table_reference(sim) -> dict:
    """The P7a.3 device-table rows read from the reference objects."""
    from simulation import combat, gas_fixed, unit_fixed
    from simulation import field_edit as fe
    tabs = sim.weapons_tables
    W = list(tabs.weapons.by_name.values())
    ammo = []
    for w in W:
        try:
            ammo.append(tabs.ammo_for_weapon(w))
        except KeyError:
            ammo.append(None)
    gid = sim.gmap.gases.name_to_id
    clamps = []
    for name in sorted(fe.FIELD_POLICY):
        pol = fe.FIELD_POLICY[name]
        cl = pol.clamp
        lo = hi = None
        if cl is not None:
            sc = fe._CURRENCY[pol.dtype]
            lo = None if cl[0] == float("-inf") else fe._quantize_in(cl[0], sc)
            hi = None if cl[1] == float("inf") else fe._quantize_in(cl[1], sc)
        clamps.append(((1 if lo is not None else 0) | (2 if hi is not None else 0),
                       lo or 0, hi or 0))
    import math
    return {
        "spray_range_tiles": tuple(int(w.range_tiles) for w in W),
        "spray_cone_half_q": tuple(unit_fixed.quantize_scalar(math.radians(
            float(w.cone_half_angle_degrees))) for w in W),
        "spray_has_ammo": tuple(int(a is not None) for a in ammo),
        "spray_heat_q": tuple(0 if a is None else unit_fixed.quantize_scalar(
            float(a.heat_deposit)) for a in ammo),
        "spray_gas_q": tuple(0 if a is None else int(gas_fixed.quantize_trace_scalar(
            float(a.gas_amount))) for a in ammo),
        "spray_gas_id": tuple(-1 if (a is None or not a.gas_species)
                              else int(gid[a.gas_species]) for a in ammo),
        "spray_src_heat": int(combat._SRC_SPRAY_HEAT),
        "spray_src_gas": int(combat._SRC_SPRAY_GAS),
        "edit_clamp_flags": tuple(c[0] for c in clamps),
        "edit_clamp_lo": tuple(c[1] for c in clamps),
        "edit_clamp_hi": tuple(c[2] for c in clamps),
    }


def part_tables():
    sims = [SprayDriver().sim, _sight_sim("playground", 0)]
    n = 0
    for e, sim in enumerate(sims):
        got = unit_tables.values(sim)
        for k, v in table_reference(sim).items():
            dv = unit_tables.read(got, k)
            as_t = lambda x: tuple(x) if isinstance(x, (tuple, list)) else (x,)  # noqa: E731
            n += 1
            if as_t(dv) != as_t(v):
                fail(f"tables env {e} row {k}: device {dv} != reference {v}")
    r0 = unit_tables.read(unit_tables.values(sims[0]), "spray_range_tiles")
    r1 = unit_tables.read(unit_tables.values(sims[1]), "spray_range_tiles")
    print(f"tables: {n} P7a.3 rows over 2 envs (tile sizes {sims[0].gmap.tile_size_m} / "
          f"{sims[1].gmap.tile_size_m} m) equal the reference objects; spray ranges "
          f"{r0} / {r1}")
    if r0 == r1:
        fail("tables: two tile sizes gave one spray range row (the per-map read)")


# ---------------------------------------------------------------------------
# PART CONE (the TEST-ONLY probe)
# ---------------------------------------------------------------------------
CONE_MAPS = ("ship_demo", "playground", "my_ship", "test_level")


def _sight_sim(name, k):
    from simulation import Simulation
    lv = level_loader.load(name, str(_ROOT / "levels"))
    return Simulation(lv, seed=1 + k, breach_physics=bp, enable_recorder=False)


def _cone_queries(rng, g, R):
    """(R, 8) probe rows and the reference's arguments for each."""
    from simulation import combat
    h, w = g.material.shape
    solid = np.argwhere(g.solid)
    rows, ref = [], []
    degs = [0.0, 0.5, 12.5, 15.0, 30.0, 45.0, 60.0, 89.0, 89.99]
    for r in range(R):
        if r % 9 == 0 and len(solid):
            ay, ax = (int(v) for v in solid[rng.integers(len(solid))])   # apex on a wall
        else:
            ay, ax = int(rng.integers(0, h)), int(rng.integers(0, w))
        kind = r % 6
        if kind == 0:
            ang = int(rng.choice([0, PI_Q16 // 2, -PI_Q16 // 2, PI_Q16, -PI_Q16,
                                  PI_Q16 // 4, -3 * PI_Q16 // 4]))
        else:
            ang = int(rng.integers(-PI_Q16, PI_Q16 + 1))
        rng_t = int(rng.choice([0, 1, 2, 3, 5, 7, 10, 14, 20, 30, 45, 60]))
        deg = float(degs[r % len(degs)] if r % 3 else rng.uniform(0.0, 89.9))
        half_q = combat.cone_q16_of_deg(deg)
        fp = int(rng.choice([0, 1, 3, 3, 5]))
        ex_x = ax - fp // 2 + int(rng.integers(-1, 2))
        ex_y = ay - fp // 2 + int(rng.integers(-1, 2))
        rows.append((ay, ax, ang, rng_t, half_q, ex_x, ex_y, fp))
        ref.append((ay, ax, ang, rng_t, deg,
                    {(ex_x + dx, ex_y + dy) for dx in range(fp) for dy in range(fp)}))
    return np.array(rows, dtype=np.int64), ref


def part_cone():
    import cupy as cp
    from simulation import combat
    tot = Counter()
    for k, name in enumerate(CONE_MAPS):
        sim = _sight_sim(name, k)
        g = sim.gmap
        h, w = g.material.shape
        rng = np.random.default_rng(9100 + k)
        args, ref = _cone_queries(rng, g, CONE_QUERIES)
        solid = cp.asarray(np.ascontiguousarray(g.solid[None]))
        box_max = (2 * 60 + 1) ** 2
        cells, meta = unit_kernels.spray_cone_probe(bp, solid, args[None], box_max, 6400)
        sight = Counter()
        orig = g.has_los

        def counting(*a, __o=orig, **kw):
            r = __o(*a, **kw)
            sight[bool(r)] += 1
            return r
        g.has_los = counting
        bad = 0
        for r, (ay, ax, ang, rt, deg, excl) in enumerate(ref):
            want = list(combat.spray_cone_tiles(g, ay, ax, 0.0, 0.0, rt, deg,
                                                exclude=excl, aim_q=ang))
            n = int(meta[0, r, 0])
            if int(meta[0, r, 1]):
                fail(f"cone {name} query {r}: the device faulted")
                bad += 1
                continue
            got = [tuple(int(v) for v in c) for c in cells[0, r, :n]]
            if n != len(want) or got != [tuple(t) for t in want]:
                fail(f"cone {name} query {r} (apex {ay},{ax} angle {ang} range {rt} deg "
                     f"{deg}): device {n} cells {got[:6]} != host {len(want)} {want[:6]}")
                bad += 1
                if bad > 3:
                    break
            tot["cells"] += len(want)
            tot["empty cones"] += int(not want)
        del g.has_los
        tot["sight True"] += sight[True]
        tot["sight False"] += sight[False]
        print(f"cone {name} ({h}x{w}): {len(ref)} queries, tile set and order equal "
              f"{'(FAILED)' if bad else ''}; physical sight True {sight[True]} / "
              f"False {sight[False]}")
    print(f"cone: {dict(tot)}")
    if tot["sight True"] < 1000 or tot["sight False"] < 1000 or tot["empty cones"] < 10:
        fail(f"cone: both sight answers / empty cones not seen often ({dict(tot)})")


# ---------------------------------------------------------------------------
# PART SCENE / BATCH / SHIP
# ---------------------------------------------------------------------------
def part_scene():
    census = Census()
    tot = run_lockstep("scene sprays", [SprayDriver()], [SLOT_4B, SLOT_4C], SCENE_TICKS,
                       census=census)
    print("scene sprays census: " + "; ".join(f"{k} = {census.c[k]}"
                                              for k in sorted(census.c)))
    for k in SCENE_BRANCHES:
        if not census.c.get(k):
            fail(f"scene sprays: the branch {k!r} was never taken (vacuous)")
    for k in UNREACHABLE:
        if census.c.get(k):
            print(f"scene sprays: NOTE the 'unreachable' branch {k!r} was taken")
    print("scene sprays unreachable: " + "; ".join(f"{k}: {v}"
                                                   for k, v in UNREACHABLE.items()))
    if not tot["SPRAY_CELLS"] or not tot["door edits"] or not tot["claims"]:
        fail(f"scene sprays: vacuous ({dict(tot)})")


def part_batch():
    for n in BATCH_NS:
        drv = [SprayDriver(variant=e + 1) for e in range(n)]
        tot = run_lockstep(f"batch N={n}", drv, [SLOT_4B, SLOT_4C], BATCH_TICKS)
        counts = [len(d.sim.units) for d in drv]
        mats = {d.sim.gmap.material.tobytes() for d in drv}
        print(f"batch N={n}: unit counts {counts}, {len(mats)} layouts")
        if len(set(counts)) < 2 or len(mats) < 2:
            fail(f"batch N={n}: the envs do not differ in unit count and layout")
        if not tot["SPRAY_CELLS"]:
            fail(f"batch N={n}: vacuous")


def part_ship():
    census = Census()
    tot = run_lockstep("ship (bots, seed 1)", [ShipDriver()], [SLOT_4B, SLOT_4C],
                       SHIP_TICKS, census=census)
    print(f"ship: sprays recorded {tot['SPRAY_CELLS']} (census {dict(census.c)})")


# ---------------------------------------------------------------------------
# Direct kernel helpers (padding, overflow, refuse, bench)
# ---------------------------------------------------------------------------
def _scene_state(ticks=3):
    """A scene sim after ``ticks`` ticks, with slot 4b's inputs captured at the
    next tick's 4b entry (rows, effect records, solid) and the host's 4b and
    4c results."""
    drv = SprayDriver()
    for _ in range(ticks):
        drv.tick()
    cap = {}
    from simulation import simulation as simmod
    from simulation import simultaneous
    o4b = simmod.process_sprays
    ores = simultaneous.SimultaneousEffects.resolve

    def w4b(gmap, units, queue, events=None, effects=None):
        cap["rows"] = unit_store.pack(units, len(units))
        cap["recs"] = applier.records_of(drv.sim, effects._pending)
        cap["solid"] = gmap.solid.copy()
        out = o4b(gmap, units, queue, events=events, effects=effects)
        cap["post_rows"] = unit_store.pack(units, len(units))
        cap["post_recs"], cap["post_cells"] = applier.records_and_cells_of(
            drv.sim, effects._pending)
        return out

    simmod.process_sprays = w4b
    try:
        drv.tick()
    finally:
        simmod.process_sprays = o4b
        simultaneous.SimultaneousEffects.resolve = ores
    return drv, cap


def _run_sprays(sims, rows, recs, solid, cap=applier.E_CAP, cell_cap=attack_lists.CELL_CAP,
                U=None):
    import cupy as cp
    n = len(rows)
    U = U or rows[0].shape[0]
    st = unit_store.DeviceStore(n, U, tables=unit_tables.values_batch(sims))
    st.upload(np.stack(rows))
    fx = applier.DeviceApplier(n, cap, n_units=U)
    fx.stage_lists(recs)
    cells = attack_lists.cell_buffer(n, cap=cell_cap)
    cells.begin()
    fields = {"solid": cp.asarray(np.ascontiguousarray(np.stack(solid)))}
    cov = unit_kernels.sprays(st, fx, cells, fields)
    return st, fx, cells, cov


def part_padding():
    drv, cap = _scene_state()
    sim = drv.sim
    U = len(sim.units)
    lay = unit_store.layout()
    vcol = lay.valid
    tl = lay.column("spray_ticks_left").offset
    rows = np.zeros((U + 4, lay.width), dtype=np.int64)
    rows[:U] = cap["rows"]
    live = [r for r in range(U) if cap["rows"][r, tl] > 0]
    for i in range(4):                       # live-looking sprayers, valid cleared
        rows[U + i] = cap["rows"][live[i % len(live)]]
        rows[U + i, vcol] = 0
    st, fx, cells, cov = _run_sprays([sim], [rows], [cap["recs"]], [cap["solid"]], U=U + 4)
    got = st.download()[0]
    if not np.array_equal(got[U:], rows[U:]):
        fail("padding: a padding row changed")
    if not np.array_equal(got[:U], cap["post_rows"]):
        fail("padding: the live rows differ from the host's with padding beside them")
    dev = fx.records()[0]
    host = [tuple(int(x) for x in r) for r in cap["post_recs"]]
    if dev != host:
        fail(f"padding: the records differ with padding ({len(dev)} vs {len(host)})")
    if cells.records()[0] != [tuple(int(x) for x in c) for c in cap["post_cells"]]:
        fail("padding: the cell buffer differs with padding")
    print(f"padding: 4 live-looking padding rows beside {U} rows ({len(live)} sprayers): "
          f"untouched, nothing recorded for them; {len(dev)} records, "
          f"{len(cap['post_cells'])} cells equal")


def part_overflow():
    drv, cap = _scene_state()
    sim = drv.sim
    n_cells = len(cap["post_cells"])
    n_recs = len(cap["post_recs"])
    n0 = len(cap["recs"])
    # the cell buffer
    try:
        _run_sprays([sim], [cap["rows"]], [cap["recs"]], [cap["solid"]],
                    cell_cap=max(1, n_cells - 3))
        fail("overflow: a cell buffer past its capacity was not refused")
    except attack_lists.ListOverflow as ex:
        print(f"overflow: the cell buffer (cap {n_cells - 3}, {n_cells} cells): refused "
              f"by name -- {ex}")
    # the effect list
    try:
        _run_sprays([sim], [cap["rows"]], [cap["recs"]], [cap["solid"]],
                    cap=max(1, n_recs - 1))
        fail("overflow: an effect list past its capacity was not refused")
    except applier.EffectOverflow as ex:
        print(f"overflow: the effect list (cap {n_recs - 1}, {n0} staged + "
              f"{n_recs - n0} SPRAY_CELLS): refused by name -- {ex}")
    # the claims list and the door's rows, at the resolve
    import cupy as cp
    st, fx, cells, _cov = _run_sprays([sim], [cap["rows"]], [cap["recs"]], [cap["solid"]])
    planes = {n: cp.asarray(np.ascontiguousarray(np.array(getattr(sim.gmap, n))[None]))
              for n in applier.PLANE_NAMES}
    from simulation import unit_events
    for what, kw in (("claims", dict(claims=3)), ("door", dict(edits=3))):
        st.upload(cap["post_rows"][None])
        fx.stage_lists([cap["post_recs"]])
        cells.stage_lists([cap["post_cells"]])
        door = edit_door.DeviceEditDoor(1, edit_cap=kw.get("edits", edit_door.EDIT_CAP))
        claims = attack_lists.claims_list(1, cap=kw.get("claims", attack_lists.CLAIM_CAP))
        door.begin()
        claims.begin()
        ring = unit_events.EventRing(1, len(sim.units))
        ring.begin()
        try:
            unit_kernels.resolve(st, fx, planes, ring,
                                 unit_kernels.WalkLists(cells=cells, door=door, claims=claims))
            fail(f"overflow: the {what} list past its capacity was not refused")
        except (attack_lists.ListOverflow, edit_door.DoorOverflow) as ex:
            print(f"overflow: the {what} list (cap 3): refused by name -- {ex}")


def part_refuse():
    drv, cap = _scene_state()
    sim = drv.sim
    lay = unit_store.layout()
    tl = lay.column("spray_ticks_left").offset
    aim = lay.column("spray_aim_angle").offset
    rows = cap["rows"].copy()
    r = int(np.nonzero(rows[:, tl] > 0)[0][0])
    rows[r, aim] = 0                       # the WEGO form: no aim angle
    rows[r, aim + 1] = 0
    try:
        _run_sprays([sim], [rows], [cap["recs"]], [cap["solid"]])
        fail("refuse: a WEGO burst was not refused")
    except unit_kernels.SpraySlotRefused as ex:
        print(f"refuse: a WEGO burst (no aim angle) refused by name -- {ex}")
    from simulation import field_edit as fe
    e = fe.FieldEdit(field="heat", region=fe.Region.TILE, coords=(3, 3), amount=1.0,
                     amount_q=65536, source_id=1)
    try:
        edit_door.door_rows(sim.gmap, [(e, 0)], {5, 6})
        fail("refuse: an interleaved host producer's edit was not refused")
    except edit_door.DoorInterleave as ex:
        print(f"refuse: an interleaved producer refused by name -- {ex}")
    # a SPRAY_CELLS whose span leaves the cell buffer faults
    import cupy as cp
    from simulation import unit_events
    st, fx, cells, _ = _run_sprays([sim], [cap["rows"]], [cap["recs"]], [cap["solid"]])
    bad = applier.make_record("SPRAY_CELLS", owner=0, heat_q=65536, gas_id=-1, gas_q=0,
                              start=0, count=10 ** 6)
    fx.stage_lists([np.array([bad])])
    planes = {n: cp.asarray(np.ascontiguousarray(np.array(getattr(sim.gmap, n))[None]))
              for n in applier.PLANE_NAMES}
    ring = unit_events.EventRing(1, len(sim.units))
    door = edit_door.DeviceEditDoor(1)
    claims = attack_lists.claims_list(1)
    try:
        unit_kernels.resolve(st, fx, planes, ring,
                             unit_kernels.WalkLists(cells=cells, door=door, claims=claims))
        fail("refuse: a SPRAY_CELLS span outside the cell buffer did not fault")
    except applier.ApplierFault as ex:
        print(f"refuse: a span outside the cell buffer faults by name -- {ex}")
    try:
        unit_kernels.resolve(st, fx, planes, ring)
        fail("refuse: a SPRAY_CELLS without its lists did not fault")
    except applier.ApplierFault:
        print("refuse: a SPRAY_CELLS walked without the cell buffer / door / claims faults")


def part_seq():
    """The door's keys are ABSOLUTE: a window whose base is not 0 (slot 2's or
    the previous tick's 9e edits before the walk) keys the device's rows from
    that base -- the SPRAY_CELLS records of a scene tick walked with the base
    1234 against ``_enqueue_spray_cells`` replayed into a queue whose seq is
    1234 (the rows through ``door_rows``, the claims folded)."""
    import cupy as cp
    from types import SimpleNamespace
    from simulation import combat, unit_events
    from simulation.field_edit import EditQueue
    drv, cap = _scene_state()
    sim = drv.sim
    base = 1234
    st, fx, cells, _ = _run_sprays([sim], [cap["rows"]], [cap["recs"]], [cap["solid"]])
    planes = {n: cp.asarray(np.ascontiguousarray(np.array(getattr(sim.gmap, n))[None]))
              for n in applier.PLANE_NAMES}
    door = edit_door.DeviceEditDoor(1)
    claims = attack_lists.claims_list(1)
    door.begin()
    claims.begin()
    door.set_seq(np.array([base]))
    ring = unit_events.EventRing(1, len(sim.units))
    ring.begin()
    st.upload(cap["post_rows"][None])
    sprays = np.array([r for r in cap["post_recs"] if r[0] == applier.KIND_SPRAY_CELLS])
    fx.stage_lists([sprays])
    cells.stage_lists([cap["post_cells"]])
    unit_kernels.resolve(st, fx, planes, ring,
                         unit_kernels.WalkLists(cells=cells, door=door, claims=claims))
    q = EditQueue()
    q._seq = base
    stand = SimpleNamespace()
    F = dict(applier.KIND_FIELDS)["SPRAY_CELLS"]
    for r in sprays:
        f = {n: int(r[applier.kind_field("SPRAY_CELLS", n)]) for n in F}
        cl = [tuple(int(v) for v in c)
              for c in cap["post_cells"][f["start"]:f["start"] + f["count"]]]
        combat._enqueue_spray_cells(stand, q, f["owner"], f["heat_q"],
                                    None if f["gas_id"] < 0 else f["gas_id"], f["gas_q"], cl)
    he, ht = edit_door.door_rows(sim.gmap, q._edits, {5, 6})
    de, dt = door.rows()[0]
    if not (np.array_equal(he, de) and np.array_equal(ht, dt)):
        fail(f"seq: the door's rows from base {base} differ from the host's "
             f"(keys device {de[:3, 8].tolist()} host {he[:3, 8].tolist()})")
    hc = np.array(attack_lists.claim_rows(stand.damage_credit), dtype=np.int64).reshape(-1, 4)
    dc = attack_lists.first_claims(np.array(claims.records()[0], dtype=np.int64))
    if not np.array_equal(hc, dc):
        fail("seq: the claims list folded differs from the host's TickCredit")
    if int(door.seq()[0]) != base + len(he):
        fail(f"seq: the door's counter ended at {int(door.seq()[0])}, not {base + len(he)}")
    print(f"seq: {len(sprays)} SPRAY_CELLS walked from the window base {base}: {len(he)} door "
          f"rows keyed {base} .. {base + len(he) - 1} equal the host's, {len(hc)} claims, the "
          f"counter at {int(door.seq()[0])}")


# ---------------------------------------------------------------------------
# PART CHAINED
# ---------------------------------------------------------------------------
def _host_side(ch):
    """What the chained 4b + 4c leave on the host besides rows and the lists:
    the pushed units' momentum books, the topology books (the solid book
    through _solid_energy_book), and a chewed wall's host-only marks."""
    from simulation.unit_impulse import IN_X, IN_Y

    def hs(sim, pre, post):
        lay = unit_store.layout()
        ix, iy = lay.column("imp_x").offset, lay.column("imp_y").offset
        for r, u in enumerate(sim.units):
            u.impulse_books[IN_X] += int(post[r, ix] - pre[r, ix])
            u.impulse_books[IN_Y] += int(post[r, iy] - pre[r, iy])
        g = sim.gmap
        b = ch.books
        if not b:
            return
        g.n_destruction_seed_sum += b["n_destruction_seed"]
        for name, key in (("ge_retire", "retire"), ("ge_mint", "mint"),
                          ("ge_destroy_seed", "destroy_seed")):
            g._gas_energy_book(key, b[name])
        g._solid_energy_book("destroy_wall", b["solid_destroy_wall"])
        g._tile_serial += b["tiles_changed"]
        mat0 = ch.inputs._host_fields["material"][0]
        changed = np.argwhere(mat0 != g.material)
        for y, x in changed:
            g.render_dirty.add((int(y), int(x)))
        if len(changed):
            g._sky_mask_dirty = True
    return hs


def part_chained():
    from simulation import combat
    twin, ref = SprayDriver(), SprayDriver()
    calls = Counter()
    o_cone, o_enq = combat.spray_cone_tiles, combat._enqueue_spray_cells

    def cone(gmap, *a, **k):
        if gmap is twin.sim.gmap:
            calls["spray_cone_tiles"] += 1
        return o_cone(gmap, *a, **k)

    def enq(gmap, *a, **k):
        if gmap is twin.sim.gmap:
            calls["_enqueue_spray_cells"] += 1
        return o_enq(gmap, *a, **k)
    enq.__wrapped__ = o_enq
    combat.spray_cone_tiles, combat._enqueue_spray_cells = cone, enq
    t0 = time.perf_counter()
    n = 0
    holder = {}
    try:
        ch = ChainedSlots(twin, [SLOT_4B, SLOT_4C], host_side=lambda s, p, q: holder["hs"](s, p, q))
        holder["hs"] = _host_side(ch)
        with ch:
            for _ in range(CHAIN_TICKS):
                twin.tick()
                ref.tick()
                n += 1
                if _unit_hash(twin.sim) != _unit_hash(ref.sim) or \
                        _field_digest(twin.sim) != _field_digest(ref.sim):
                    fail(f"chained 4b + 4c: the twin left the reference at tick "
                         f"{twin.sim.tick}")
                    break
    finally:
        combat.spray_cone_tiles, combat._enqueue_spray_cells = o_cone, o_enq
    recs = sum(sum(1 for r in x["effects"] if r[0] == applier.KIND_SPRAY_CELLS)
               for x in ch.lists)
    edits = sum(len(x["edits"]) for x in ch.lists)
    claims = sum(len(x["claims"]) for x in ch.lists)
    walls = sum(1 for x in ch.lists for r in x["effects"] if r[0] == applier.KIND_CHEW)
    print(f"chained 4b + 4c (sprays): {n} ticks, the twin equal to the reference every "
          f"tick ({ch.runs} chain runs; {recs} SPRAY_CELLS, {edits} door edits replayed, "
          f"{claims} claims, {walls} CHEWs walked); host spray law calls in the twin "
          f"{dict(calls) or 0} ({time.perf_counter() - t0:.0f} s)")
    if calls:
        fail(f"chained: the host spray law ran in the twin {dict(calls)}")
    if ch.runs != n or not recs or not edits or not claims:
        fail("chained 4b + 4c: vacuous")


def part_chained_all():
    """#29 P7a.2 merged: slots 4 (shooting, which ARMS the spray bursts), 4a,
    4b and 4c ALL on the device in the running spray scene, chained (slot 2' and
    the rest of the tick on the host): the unit hash and field digest equal an
    untouched reference every tick; the host spray law run 0 times."""
    from cuda_unit_shooting_check import SLOT_4, SLOT_4A
    from simulation import combat
    twin, ref = SprayDriver(), SprayDriver()
    calls = Counter()
    o_cone, o_enq, o_start = (combat.spray_cone_tiles, combat._enqueue_spray_cells,
                              combat.start_spray_burst_directional)

    def cone(gmap, *a, **k):
        if gmap is twin.sim.gmap:
            calls["spray_cone_tiles"] += 1
        return o_cone(gmap, *a, **k)

    def enq(gmap, *a, **k):
        if gmap is twin.sim.gmap:
            calls["_enqueue_spray_cells"] += 1
        return o_enq(gmap, *a, **k)
    enq.__wrapped__ = o_enq

    def start(u, *a, **k):
        if any(u is v for v in twin.sim.units):
            calls["start_spray_burst_directional"] += 1
        return o_start(u, *a, **k)
    combat.spray_cone_tiles, combat._enqueue_spray_cells = cone, enq
    combat.start_spray_burst_directional = start
    t0 = time.perf_counter()
    n = 0
    holder = {}
    try:
        ch = ChainedSlots(twin, [SLOT_4, SLOT_4A, SLOT_4B, SLOT_4C],
                          host_side=lambda s, p, q: holder["hs"](s, p, q))
        holder["hs"] = _host_side(ch)
        with ch:
            for _ in range(CHAIN_TICKS):
                twin.tick()
                ref.tick()
                n += 1
                if _unit_hash(twin.sim) != _unit_hash(ref.sim) or                         _field_digest(twin.sim) != _field_digest(ref.sim):
                    fail(f"chained 4 + 4a + 4b + 4c: the twin left the reference at tick "
                         f"{twin.sim.tick}")
                    break
    finally:
        combat.spray_cone_tiles, combat._enqueue_spray_cells = o_cone, o_enq
        combat.start_spray_burst_directional = o_start
    recs = Counter(applier.KINDS[r[0] - 1] for x in ch.lists for r in x["effects"])
    print(f"chained 4 + 4a + 4b + 4c (sprays): {n} ticks, the twin equal to the reference "
          f"every tick ({ch.runs} chain runs; records walked {dict(recs)}); host spray law "
          f"calls in the twin {dict(calls) or 0} ({time.perf_counter() - t0:.0f} s)")
    if calls:
        fail(f"chained all: the host spray law ran in the twin {dict(calls)}")
    if ch.runs != n or not recs.get("SPRAY_CELLS") or not recs.get("HIT"):
        fail("chained 4 + 4a + 4b + 4c: vacuous")


# ---------------------------------------------------------------------------
# Main, bench
# ---------------------------------------------------------------------------
PARTS = ("tables", "cone", "scene", "batch", "padding", "overflow", "refuse", "seq",
         "chained", "chained_all", "ship")


def main(parts=None) -> int:
    t0 = time.perf_counter()
    want = parts or os.environ.get("UNIT_SPRAYS_PARTS", ",".join(PARTS)).split(",")
    for p in want:
        globals()[f"part_{p}"]()
    print(f"total {time.perf_counter() - t0:.0f} s")
    ok = not FAILS
    print(f"UNIT_SPRAYS_RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def bench(reps: int = 50) -> int:
    """The TIMING table: median ms of slot 4b and of the SPRAY_CELLS apply
    (the resolve stage walking the tick's SPRAY_CELLS records) at N = 1 / 8 /
    64 copies of the spray scene's tick."""
    import cupy as cp
    from simulation import unit_events
    drv, cap = _scene_state(ticks=4)
    sim = drv.sim
    U = len(sim.units)
    tab1 = unit_tables.values_batch([sim])
    planes1 = {n: np.array(getattr(sim.gmap, n), copy=True) for n in applier.PLANE_NAMES}
    sprays_only = np.array([r for r in cap["post_recs"] if r[0] == applier.KIND_SPRAY_CELLS])
    lines = []
    for n in (1, 8, 64):
        st = unit_store.DeviceStore(n, U, tables=np.repeat(tab1, n, axis=0))
        rows = np.repeat(cap["rows"][None], n, axis=0)
        fx = applier.DeviceApplier(n, 256, n_units=U)
        cells = attack_lists.cell_buffer(n)
        solid = cp.asarray(np.ascontiguousarray(np.repeat(cap["solid"][None], n, axis=0)))
        bm = unit_kernels.spray_box_max(st, *cap["solid"].shape)
        ts = []
        for _ in range(reps):
            st.upload(rows)
            fx.begin()
            cells.begin()
            cp.cuda.Stream.null.synchronize()
            t = time.perf_counter()
            unit_kernels.sprays(st, fx, cells, {"solid": solid}, box_max=bm)
            ts.append((time.perf_counter() - t) * 1e3)
        lines.append(("4b sprays", n, int(np.sum(cap["rows"][:, unit_store.layout().column(
            "spray_ticks_left").offset] > 0)), float(np.median(ts))))
        dev = {k: cp.asarray(np.ascontiguousarray(np.stack([v] * n)))
               for k, v in planes1.items()}
        ring = unit_events.EventRing(n, U)
        door = edit_door.DeviceEditDoor(n)
        claims = attack_lists.claims_list(n)
        ts = []
        for _ in range(reps):
            fx.stage_lists([sprays_only] * n)
            cells.stage_lists([cap["post_cells"]] * n)
            door.begin()
            claims.begin()
            ring.begin()
            cp.cuda.Stream.null.synchronize()
            t = time.perf_counter()
            unit_kernels.resolve(st, fx, dev, ring,
                                 unit_kernels.WalkLists(cells=cells, door=door, claims=claims))
            ts.append((time.perf_counter() - t) * 1e3)
        lines.append(("SPRAY_CELLS", n, len(cap["post_cells"]), float(np.median(ts))))
    print("TIMING (median ms, the call incl. its count / book / overflow downloads):")
    for name, n, k, ms in lines:
        what = "sprayers" if name.startswith("4b") else "cells/env"
        print(f"  {name:12s} N={n:3d} {what} {k:4d}: {ms:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
