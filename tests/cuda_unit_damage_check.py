"""#29 P6.3 -- THE DAMAGE GATE: the device ``damage.apply_packet``, the EVENT
RING and the first slots on them -- 2b (statuses), 9c (heat), 9c3 (teargas,
the poison dose, the poison law) -- each against its Python law at tol 0,
rows AND events.

Runs inside the GPU subprocess (tests/cuda_harness.run_cuda_script; wrapper
tests/test_cuda_unit_damage.py). The store is ``simulation.unit_store``'s
``DeviceStore`` with its device tables (``simulation.unit_tables``); the ring
``simulation.unit_events.EventRing``; the kernels ``simulation.unit_kernels``
(C++ ``cpp/src/cuda_unit_status.cu`` / ``cuda_unit_exchange.cu`` over
``cpp/src/cuda_unit_damage.cuh``); the harness ``tests/cuda_unit_lockstep.py``
(``DeviceSlot(..., events=True)``: the ring compared with the host's
``sim.tick_events`` of the slot, record for record, in order).

PART TABLES -- every new table row read back from the device equals the
reference object's value (mitigation, exposure, the status registry's DoT
columns, the heat row's dials and bands, the gas rows' thresholds, the poison
law's class values), per env of a batch of two maps.

PART SCENES -- the SHADOW LOCKSTEP of the three slots together, every tick:
rows equal, events equal in emission order, each kernel's coverage == the rows
its host slot changed, the ring's count == the host's events of the slot; the
hooked match equal to an unhooked twin every tick. Scenes (all with
non-neutral HP / TOX_ONSET / TOX_DECAY / TOX_HARM hooks on part of the
marines):
  ship   -- ship_demo's bots match, seed 1;
  fire   -- four lanes: a marine dying in an unowned fire (the environment's
            heat), a flamethrower's jet on a marine (the credited heat), a
            marine beside a hot wall (the radiation the larger exposure), a
            zombie in a fire (the zombie's x4 resist), a marine on the
            "default" env profile (the profile branch of the band);
  gas    -- a marine walking into and out of teargas (BLINDED applied,
            refreshed, run out), a marine in a claimed poison cloud until it
            dies (the dose credited to the claimant, its later packets
            indirect), a hooked marine that leaves a cloud and recovers to
            exactly 0 (the flush, poison_src back to -1), a zombie in a cloud
            (no dose), a marine at the body's store standing in poison (the
            dose clipped and counted, == poison.SATURATION's hits);
  status -- statuses applied through status.apply_status (no shipped trigger
            applies a damage-over-time status: the registry's DoT rows are
            exercised this way): a stun running out, BURNING (max), a stack
            of POISONED at its cap, SUFFOCATING, REGEN (heal), a DoT kill of
            a marine and of a zombie.
The doomed units carry an order, exposure, an impulse and a poison depot when
they die (fixtures written between ticks), so every death clear is compared.

PART BATCH -- the fire / gas / status scenes at N = 2 and 8 with different
unit counts per env (padding rows): env i of the batch == env i alone, rows
and rings.

PART OVERFLOW -- a ring too small for a tick's events: the kernel's dropped
count == the events past the capacity, nothing written past it, the records
within it the host's first ones, and the Python face refuses the run by name.

PART CHAINED -- 2b alone, and 9c + 9c2 + 9c3 together, replacing the host's
slots in a twin match (the ring's events appended to sim.tick_events): the
twin's per-tick unit hash (which hashes the events) and field digest equal an
untouched reference's on every tick.

PROPERTY: the device damage pipeline and its three slots are their Python
laws on every valid row, integer for integer, events included in the
reference's emission order, in a batch as alone. BREAKS IF a rounding, the
exposure scale, the kill's once-only rule, a death clear, the ring's order, an
event field, the credit's channel, the heat plane's currency, the poison
flush or a TOX hook differs from the law, the zombie is dosed, a padding row
or another env is touched. Each validated by breaking it (the patch's report).

Prints per-part lines and ``UNIT_DAMAGE_RESULT: PASS`` / ``FAIL``; exits 0 / 1.
``bench()`` prints the TIMING table.
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
                                ShadowLockstep, capture, gather)
from simulation import (unit_events, unit_kernels, unit_store,  # noqa: E402
                        unit_tables, unit_traits)

FAILS: list = []
ONE = unit_traits.ONE
SHIP_TICKS = int(os.environ.get("UNIT_DAMAGE_SHIP_TICKS", "500"))
LANE_TICKS = int(os.environ.get("UNIT_DAMAGE_LANE_TICKS", "360"))
BATCH_TICKS = int(os.environ.get("UNIT_DAMAGE_BATCH_TICKS", "200"))
SCENES = os.environ.get("UNIT_DAMAGE_SCENES", "ship,fire,gas,status").split(",")
BATCH_NS = [int(x) for x in os.environ.get("UNIT_DAMAGE_BATCH_NS", "2,8").split(",")]
CHAINS = os.environ.get("UNIT_DAMAGE_CHAINS", "2b,9c").split(",")


def fail(msg):
    FAILS.append(msg)
    print("FAIL:", msg)


# ---------------------------------------------------------------------------
# The slots
# ---------------------------------------------------------------------------
SLOT_2B = DeviceSlot(label="2b statuses", run=unit_kernels.status_tick,
                     events=True)
SLOT_9C = DeviceSlot(label="9c unit heat damage",
                     run=lambda st, inp, ring: unit_kernels.heat_damage(
                         st, ring, inp.fields, inp.tables["credit_claims"]),
                     inputs=tuple("field:" + f for f in unit_kernels.HEAT_FIELDS)
                     + ("credit_claims",), events=True)
SLOT_9C3 = DeviceSlot(label="9c3 gas rows",
                      run=lambda st, inp, ring: unit_kernels.gas_rows(
                          st, ring, inp.fields, inp.tables["credit_claims"]),
                      inputs=tuple("field:" + f for f in unit_kernels.GAS_FIELDS)
                      + ("credit_claims",), events=True)
SLOT_9C2 = DeviceSlot(label="9c2 wind on units",
                      run=lambda st, inp: unit_kernels.wind_impulse(st, inp.fields),
                      inputs=tuple("field:" + f for f in unit_kernels.WIND_FIELDS))
SLOTS = [SLOT_2B, SLOT_9C, SLOT_9C3]


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
HOOK_SETS = (
    None,
    {"HP": ONE * 5 // 4, "TOX_ONSET": ONE * 8, "TOX_DECAY": ONE * 8,
     "TOX_HARM": ONE * 3 // 4},
    {"HP": ONE * 3 // 4, "TOX_ONSET": ONE // 2, "TOX_DECAY": ONE * 2,
     "TOX_HARM": ONE * 5 // 4},
)
RECOVER_HOOKS = {"TOX_ONSET": ONE * 8, "TOX_DECAY": ONE * 8}


def _hook(sim, units):
    for i, u in enumerate(units):
        h = HOOK_SETS[i % 3]
        if h is not None:
            unit_traits.compile_unit(sim, u, h)


class ShipDriver:
    """ship_demo's bots match (MatchSession), hooks on two thirds of the
    marines."""

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


HULL, AIR = 1, 4
WALL_HEAT = float(os.environ.get("UNIT_DAMAGE_WALL_HEAT", "3.0e3"))
WALL_TICKS = 100
LANE_TOPS = (1, 13, 25, 37)          # 7-row air lanes between 5-row hull bands


def _lanes(h=45, w=34):
    from level_loader import LevelData
    tm = np.full((h, w), HULL, dtype=np.int32)
    for top in LANE_TOPS:
        tm[top:top + 7, 1:w - 1] = AIR
    return LevelData(name="p63_lanes", version="1", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


class LaneDriver:
    """Four lanes (no sight, path or blast reach between them) and a scene
    script: ``kind`` is "fire", "gas" or "status"; ``variant`` (0 ..) adds
    ``variant % 3`` idle bystanders so a batch's envs differ in unit count
    and shifts the timings a little."""

    def __init__(self, kind, variant=0, hooks=True):
        from simulation import Simulation
        from simulation.ruleset import ContinuousRealtime
        self.kind, self.variant, self.k = kind, variant, 0
        sim = Simulation(_lanes(), seed=20261114 + variant, breach_physics=bp,
                         enable_recorder=False, ruleset=ContinuousRealtime())
        sim.set_paused(False)
        self.sim = sim
        self.who = {}
        self.hooks = hooks
        getattr(self, "_setup_" + kind)()
        for i in range(variant % 3):                  # bystanders (padding)
            self._add(f"B{i}", 26 + 2 * i, LANE_TOPS[i] + 1, team=0)

    # -- helpers -----------------------------------------------------------
    def _add(self, name, x, y, team=0, hooks=None, env_profile=None, hp=None):
        from simulation.unit import Unit
        u = Unit(name, x=float(x), y=float(y), team=team, env_profile=env_profile)
        self.sim.add_unit(u, hooks=hooks if self.hooks else None)
        if hp is not None:
            u.current_hp = float(hp)
        self.who[name] = u
        return u

    def _heat(self, tiles, amount):
        from simulation.field_edit import EditMode, FieldEdit, Region
        for (ty, tx) in tiles:
            self.sim.edit(FieldEdit(field="heat", region=Region.TILE,
                                    coords=(ty, tx), amount=amount,
                                    mode=EditMode.ADD, source_id=990))

    def _foot(self, u):
        return [(ty, tx) for (tx, ty) in u.occupied_tiles()]

    def _gas(self, species, fy, fx, amount, radius, owner=-1):
        from simulation import payloads
        payloads.emit_gas(self.sim.gmap, self.sim.edit_queue, fy, fx, species,
                          amount, radius, owner_id=owner)

    def _doom(self, u):
        """Fixtures on a unit about to die (written between ticks): an order,
        exposure, an impulse and (without poison of its own) a depot."""
        from simulation import poison
        if not u.alive:
            return
        u.order_what, u.order_where, u.order_until, u.order_target = 2, 0, -1, -1
        u.exposure[0] = max(u.exposure[0], 30000)
        u.exposure[2] = max(u.exposure[2], 20000)
        u.imp_x += 2000
        if not (u.poison_x0 or u.poison_x1):
            poison.dose(u, 3000, -1)

    # -- fire ----------------------------------------------------------------
    def _setup_fire(self):
        a, b, c, d = (t + 2 for t in LANE_TOPS)
        self._add("VA", 12, a, hp=30, hooks=HOOK_SETS[1])        # dies in fire
        self._add("PA", 22, a, env_profile="default", hooks=HOOK_SETS[2])
        f = self._add("F", 3, b)                                   # flamethrower
        f.weapon_id = "dragon_7"
        self._add("VB", 9, b, hp=200000, hooks=HOOK_SETS[1])       # survives the jet
        self._add("WC", 12, c + 2, hp=5000)                        # beside a wall
        self._add("ZD", 12, d, team=255)                           # zombie in fire

    def _before_fire(self):
        from simulation.turning import place
        from simulation.unit_fixed import quantize_scalar
        sim, w = self.sim, self.who
        if self.k == 0:
            place(w["F"], quantize_scalar(0.0))
            sim.set_aim(w["F"].id, ONE, 0)
        sim.set_trigger(w["F"].id, (self.k // 40) % 2 == 0)   # bursts
        if w["VA"].alive:
            self._doom(w["VA"])
            self._heat(self._foot(w["VA"]), 0.15)
        if w["PA"].alive and self.k % 3 == 0:
            self._heat(self._foot(w["PA"]), 0.05)
        if w["ZD"].alive and self.k % 2 == 0:
            self._heat(self._foot(w["ZD"]), 0.08)
        # the hot wall: the hull row under lane C, beside WC
        top_c = LANE_TOPS[2]
        wall = [(top_c + 7, x) for x in range(10, 17)]
        if self.k < WALL_TICKS:
            self._heat(wall, WALL_HEAT)

    # -- gas -------------------------------------------------------------------
    def _setup_gas(self):
        from simulation import poison
        a, b, c, d = (t + 2 for t in LANE_TOPS)
        self._add("WA", 4, a)                                      # walker
        self._add("VB", 14, b, hp=12, hooks=HOOK_SETS[2])          # dies of poison
        self._add("OB", 28, d)                                     # the claimant
        self._add("RC", 26, a, hooks=RECOVER_HOOKS)                # recovers
        cd = self._add("CD", 27, c, hp=5000)                       # at the store
        cd.poison_x0 = poison.X_MAX - 4000
        self._add("ZD", 14, d, team=255)                           # zombie in gas

    def _before_gas(self):
        from control_source import quantize_stick_direction
        sim, w, k = self.sim, self.who, self.k
        a, b, c, d = (t + 2 for t in LANE_TOPS)
        if k % 12 == 0 and k < 240:
            self._gas("teargas", a + 1, 9, 1.0, 1)
        if w["WA"].alive:
            dx = 1.0 if (k // 70) % 2 == 0 else -1.0
            mv = quantize_stick_direction(dx, 0.0, 0.0)
            sim.set_move_dir(w["WA"].id, mv[0], mv[1], 0)
        if w["VB"].alive:
            self._doom(w["VB"])
            if k % 6 == 0:
                self._gas("poison", b + 1, 15, 3.0, 2, owner=w["OB"].id)
        if k == 0 and w["RC"].alive:
            from simulation import poison
            poison.dose(w["RC"], 20000, -1)          # one dose, then it drains
        if k % 8 == 0:
            self._gas("poison", c + 1, 28, 3.0, 1)                 # CD's cloud
            self._gas("poison", d + 1, 15, 3.0, 2)                 # ZD's cloud

    # -- status ------------------------------------------------------------------
    def _setup_status(self):
        a, b, c, d = (t + 2 for t in LANE_TOPS)
        self._add("SA", 6, a, hooks=HOOK_SETS[1])
        self._add("DB", 6, b, hp=9, hooks=HOOK_SETS[2])            # DoT kill
        self._add("SB", 20, b)
        self._add("ZC", 6, c, team=255, hp=40)                     # zombie DoT kill
        self._add("RD", 6, d, hp=50)                               # healed

    def _before_status(self):
        from simulation import status as S
        w, k = self.who, self.k
        sa, db, zc, rd = w["SA"], w["DB"], w["ZC"], w["RD"]
        if sa.alive:
            if k == 3:
                S.apply_status(sa, S.STUNNED, 0, 5, None)
            if k == 8:
                S.apply_status(sa, S.BURNING, 1.25, 30, w["SB"].id)
            if k == 12:
                S.apply_status(sa, S.BURNING, 2.5, 10, None)   # max: the stronger
            if k == 20:
                S.apply_status(sa, S.SUFFOCATING, 0.75, 12, None)
            if k == 30:
                for _ in range(9):                             # 8 held, 1 refused
                    S.apply_status(sa, S.POISONED, 0.25, 40 + _, w["SB"].id)
            if k == 50:
                S.apply_status(sa, S.REGEN, 1.0, 25, None)
            if k == 90:
                S.apply_status(sa, S.STUNNED, 0, 3, w["SB"].id)
        if db.alive:
            self._doom(db)
            if k == 5:
                S.apply_status(db, S.BURNING, 1.5, 200, w["SB"].id)
                S.apply_status(db, S.STUNNED, 0, 400, None)
        if zc.alive and k == 5:
            S.apply_status(zc, S.BURNING, 0.5, 300, None)          # x4: 2 HP a tick
        if rd.alive and k % 40 == 2:
            S.apply_status(rd, S.REGEN, 0.5, 30, None)
            S.apply_status(rd, S.SUFFOCATING, 0.25, 10, None)

    def tick(self):
        getattr(self, "_before_" + self.kind)()
        self.sim.step()
        self.sim.set_paused(False)
        self.k += 1

    def is_terminal(self):
        return False


# ---------------------------------------------------------------------------
# The probes: what the scenes made the laws do
# ---------------------------------------------------------------------------
class Probe:
    """Per slot: events by kind and cause, credited hits, deaths; for 9c the
    hits whose exposure was the radiation; for 9c3 the poison levels that
    returned to exactly 0 and the doses clipped (== poison.SATURATION);
    for 2b the DoT hits by kind, statuses run out, heals."""

    def __init__(self):
        from simulation import poison
        self.c = Counter()
        self.sat0 = poison.SATURATION["hits"]

    def __call__(self, reps, ls):
        from simulation.status import STATUS_REGISTRY
        lay = ls.layout
        cols = {n: lay.column(n).offset for n in
                ("id", "poison_x0", "poison_x1", "poison_src", "life_state",
                 "current_hp")}
        sc = lay.column("statuses")
        for k, slot in enumerate(ls.slots):
            lab = slot.label
            for e in range(ls.N):
                for ev in ls._evs[k][e]:
                    kind = type(ev).__name__
                    cause = getattr(ev, "source", getattr(ev, "killed_by", ""))
                    self.c[(lab, kind, cause)] += 1
                    if kind == "UnitHitEvent" and ev.attacker_id >= 0:
                        self.c[(lab, "credited hit")] += 1
                    if kind == "UnitHitEvent" and ev.damage < 0:
                        self.c[(lab, "heal")] += 1
                    if kind == "UnitKilledEvent":
                        self.c[(lab, "death")] += 1
                pre, post = ls._pre[k][e], ls._post[k][e]
                for r in range(pre.shape[0]):
                    if not pre[r, lay.valid]:
                        continue
                    if lab == "9c3 gas rows":
                        if (pre[r, cols["poison_x0"]] or pre[r, cols["poison_x1"]]) \
                                and post[r, cols["poison_x0"]] == 0 \
                                and post[r, cols["poison_x1"]] == 0 \
                                and post[r, cols["life_state"]] == 0:
                            self.c[(lab, "levels back to 0")] += 1
                        if post[r, cols["poison_x0"]] > pre[r, cols["poison_x0"]]:
                            self.c[(lab, "dosed rows")] += 1
                        kp = pre[r, sc.offset:sc.offset + sc.width][0::4]
                        kq = post[r, sc.offset:sc.offset + sc.width][0::4]
                        if 8 in kq and (8 not in kp):
                            self.c[(lab, "blinded")] += 1
                    if lab == "2b statuses":
                        kp = [x for x in pre[r, sc.offset:sc.offset + sc.width][0::4] if x >= 0]
                        kq = [x for x in post[r, sc.offset:sc.offset + sc.width][0::4] if x >= 0]
                        if len(kq) < len(kp):
                            self.c[(lab, "statuses run out")] += 1
                        if kp.count(5) == 8:
                            self.c[(lab, "stack at its cap ticked")] += 1
                    if lab == "9c unit heat damage" and int(pre[r, cols["id"]]) in {
                            ev.unit_id for ev in ls._evs[k][e]
                            if type(ev).__name__ == "UnitHitEvent"}:
                        self._heat_probe(ls, k, e, r, pre, cols)

    def _heat_probe(self, ls, k, e, r, pre, cols):
        """Whether row r's hottest exposure was the radiation sensor."""
        from simulation import optics_fixed
        inp = ls.last_inputs.get(ls.slots[k].label)
        if inp is None or not inp._host_fields:
            return
        heat = inp._host_fields["heat"][e]
        rad = inp._host_fields["rad_flux_sweep"][e]
        lay = ls.layout
        if pre[r, lay.column("life_state").offset] != 0:
            return
        x = int(pre[r, lay.column("x_q").offset]) // 65536
        y = int(pre[r, lay.column("y_q").offset]) // 65536
        fp = int(pre[r, lay.column("footprint").offset])
        fb = optics_fixed.sweep_fine_bits(ls.drivers[e].sim.gmap)
        hv = heat[max(0, y):y + fp, max(0, x):x + fp]
        rv = optics_fixed.fine_heat_shr(rad[max(0, y):y + fp, max(0, x):x + fp], 0, fb)
        if rv.size and rv.max() > 0 and rv.max() > hv.max():
            self.c[("9c unit heat damage", "hits by radiation (the larger exposure)")] += 1

    def saturation(self):
        from simulation import poison
        return poison.SATURATION["hits"] - self.sat0

    def report(self, name):
        keys = sorted(self.c, key=str)
        print(f"{name}: " + "; ".join(f"{'/'.join(map(str, k))} = {self.c[k]}"
                                     for k in keys))


# ---------------------------------------------------------------------------
# The lockstep runner
# ---------------------------------------------------------------------------
def run_lockstep(name, drivers, slots, ticks, twins=None, min_events=None,
                 probe=None, e_max=unit_events.E_MAX):
    """Shadow ``slots`` on ``drivers`` for ``ticks`` ticks: every tick each
    slot equal (rows and events), its coverage == the host's changed rows and
    its ring count == the host's events; ``twins`` (unhooked copies) equal
    every tick. Returns {label: (env-ticks with coverage, events)}."""
    t0 = time.perf_counter()
    cov = Counter()
    evn = Counter()
    refused = Counter()
    n = 0
    ok = True
    with ShadowLockstep(drivers, list(slots), e_max=e_max) as ls:
        for _ in range(ticks):
            if any(getattr(d, "is_terminal", lambda: False)() for d in drivers):
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
                if rep.events is not None and not np.array_equal(rep.events,
                                                                  rep.host_events):
                    fail(f"{name} [{lab}] tick {rep.tick}: ring count "
                         f"{rep.events.tolist()} != host events "
                         f"{rep.host_events.tolist()}")
                    ok = False
                    break
                cov[lab] += int((rep.coverage > 0).sum())
                if rep.events is not None:
                    evn[lab] += int(rep.events.sum())
                refused[lab] += int(ls.store.last_refused.sum())
            if not ok:
                break
            if probe is not None:
                probe(reps, ls)
            if twins is not None:
                for d, tw in zip(drivers, twins):
                    tw.tick()
                    if not _same(d.sim, tw.sim):
                        fail(f"{name}: the hooked match diverged from the "
                             f"unhooked one at tick {d.sim.tick}")
                        ok = False
                if not ok:
                    break
    total = n * len(drivers)
    print(f"{name}: N={len(drivers)} units per env "
          f"{[len(d.sim.units) for d in drivers]} (U {ls.U}), {n} ticks"
          f"{' (twin equal every tick)' if twins is not None and ok else ''}"
          f"{', env i == env i alone' if len(drivers) > 1 and ok else ''}; "
          + ", ".join(f"[{lab}] coverage on {100 * cov[lab] / max(1, total):.1f} % "
                      f"of env-ticks, {evn[lab]} events"
                      for lab in (s.label for s in slots))
          + f" ({time.perf_counter() - t0:.0f} s)")
    for lab, want in (min_events or {}).items():
        if evn[lab] < want:
            fail(f"{name} [{lab}]: {evn[lab]} events, below the stated {want} "
                 f"(vacuous scene)")
    for lab in (min_events or {}):
        if cov[lab] == 0:
            fail(f"{name} [{lab}]: the kernel never changed a row (vacuous)")
    return {lab: (cov[lab], evn[lab], refused[lab]) for lab in cov}


# ---------------------------------------------------------------------------
# PART TABLES
# ---------------------------------------------------------------------------
def part_tables():
    from config import CFG
    from simulation import exchange, exposure, gas_fixed, gases, optics_fixed, poison
    from simulation.events import CAUSE_NAMES
    from simulation.loadouts import get_tables
    from simulation.species import SPECIES_REGISTRY, ZOMBIE_MITIGATION
    from simulation.status import STATUS_REGISTRY
    from simulation.unit_fixed import quantize_scalar as qs
    a = LaneDriver("fire", hooks=False)
    b = ShipDriver(hooks=False)
    sims = [a.sim, b.sim]
    tab = unit_tables.values_batch(sims)
    st = unit_store.DeviceStore(2, 4, tables=tab)
    got = st.tables.download()
    if not np.array_equal(got, tab):
        fail("tables: the device tables differ from the uploaded ones")
    profs = list(get_tables().env_profiles.by_name.values())
    specs = list(SPECIES_REGISTRY.values())
    es = exposure.get_settings()
    ps = poison.get_settings()
    gain, amb = exchange.heat_dials_q(exchange.heat_dials(CFG))
    n = 0
    for e, sim in enumerate(sims):
        flat = lambda f, xs: tuple(int(v) for x in xs for v in f(x))  # noqa: E731
        want = {
            "mitig_armor_zombie_q": tuple(ZOMBIE_MITIGATION.armor_q),
            "mitig_resist_zombie_q": tuple(ZOMBIE_MITIGATION.resist_q),
            "mitig_armor_species_q": flat(lambda s: s.mitigation.armor_q, specs),
            "mitig_resist_species_q": flat(lambda s: s.mitigation.resist_q, specs),
            "expose_k_q": es.k_q, "expose_x_max_q": es.x_max_q,
            "expose_resist_zombie_q": tuple(es.zombie_resist),
            "expose_resist_profile_q": flat(lambda p: p.expose_resist, profs),
            "expose_resist_species_q": flat(lambda s: s.environment.expose_resist, specs),
            "status_dtype": tuple(-1 if r.dtype is None else r.dtype
                                  for r in STATUS_REGISTRY),
            "status_cause": tuple(-1 if r.dtype is None else CAUSE_NAMES.index(r.name)
                                  for r in STATUS_REGISTRY),
            "heat_gain_q": gain, "heat_ambient_q": amb,
            "heat_overtemp_q": qs(float(CFG.combat.heat_overtemp_scale)),
            "heat_temp_max_profile_q": tuple(qs(p.temperature_max) for p in profs),
            "heat_temp_max_species_q": tuple(qs(s.environment.temperature_max)
                                             for s in specs),
            "heat_rate_profile_q": tuple(qs(p.environmental_damage_rate) for p in profs),
            "heat_rate_species_q": tuple(qs(s.environment.environmental_damage_rate)
                                         for s in specs),
            "heat_tps": sim._tps,
            "heat_fine_bits": optics_fixed.sweep_fine_bits(sim.gmap),
            "gas_teargas": gases.TEARGAS, "gas_poison": gases.POISON,
            "teargas_thresh_q": gas_fixed.quantize_trace_scalar(
                float(CFG.exchange.teargas_blind_density)),
            "teargas_blind_ticks": int(CFG.exchange.teargas_blind_ticks),
            "poison_thresh_q": gas_fixed.quantize_trace_scalar(
                float(CFG.exchange.poison_min_density)),
            "poison_gas_q": ps.gas_q,
            "poison_dose_den": int(gas_fixed.TRACE_ONE) * int(sim._tps),
            "poison_onset_class_q": ps.class_onset_q,  # traits-class: the table check
            "poison_decay_class_q": ps.class_decay_q,  # traits-class: the table check
            "poison_harm_class_q": ps.class_harm_q,  # traits-class: the table check
            "poison_x_max": poison.X_MAX, "poison_flush_q": poison.FLUSH_Q,
            "footprint_rowmajor": 1,
        }
        for k, v in want.items():
            dv = unit_tables.read(got[e], k)
            n += 1
            as_t = lambda x: tuple(x) if isinstance(x, (tuple, list)) else (x,)  # noqa: E731
            if as_t(dv) != as_t(v):
                fail(f"tables env {e} row {k}: device {dv} != reference {v}")
    ring = unit_events.EventRing(1, 1)
    print(f"tables: {n} P6.3 values over 2 envs equal the reference objects (T "
          f"{unit_tables.layout().width}); the ring's compiled layout is "
          f"unit_events' (F {ring.shape[2]}, row cap {unit_events.row_cap()})")


# ---------------------------------------------------------------------------
# PART SCENES / BATCH
# ---------------------------------------------------------------------------
def _scene(kind, **kw):
    if kind == "ship":
        return ShipDriver(**kw)
    return LaneDriver(kind, **kw)


#: What each scene must make the slots emit (a vacuous scene fails).
SCENE_MIN = {
    "ship": {},
    "fire": {SLOT_9C.label: 20},
    "gas": {SLOT_9C3.label: 20, SLOT_2B.label: 0},
    "status": {SLOT_2B.label: 20},
}


def part_scenes():
    for kind in SCENES:
        drv = _scene(kind)
        pr = Probe()
        out = run_lockstep(f"scene {kind}", [drv], SLOTS,
                           SHIP_TICKS if kind == "ship" else LANE_TICKS,
                           twins=None, probe=pr, min_events=SCENE_MIN[kind])
        pr.report(f"scene {kind}")
        _scene_facts(kind, pr, out)
        # the hooked match equals the same match without the harness
        _harmless(kind)


def _scene_facts(kind, pr, out):
    c = pr.c
    need = []
    if kind == "fire":
        need = [(SLOT_9C.label, "death"), (SLOT_9C.label, "credited hit"),
                (SLOT_9C.label, "hits by radiation (the larger exposure)")]
        if not any(k[0] == SLOT_9C.label and k[1] == "UnitHitEvent" and k[2] == "heat"
                   for k in c):
            fail("scene fire: no heat hit")
    if kind == "gas":
        need = [(SLOT_9C3.label, "death"), (SLOT_9C3.label, "credited hit"),
                (SLOT_9C3.label, "levels back to 0"), (SLOT_9C3.label, "blinded"),
                (SLOT_9C3.label, "dosed rows")]
        sat = pr.saturation()
        dev_ref = out.get(SLOT_9C3.label, (0, 0, 0))[2]
        print(f"scene gas: doses clipped at the body's store: host "
              f"poison.SATURATION +{sat}, the kernel's refused {dev_ref}")
        if sat == 0 or dev_ref != sat:
            fail(f"scene gas: the clip branch: host {sat} != device {dev_ref} "
                 f"(or none)")
    if kind == "status":
        need = [(SLOT_2B.label, "death"), (SLOT_2B.label, "credited hit"),
                (SLOT_2B.label, "heal"), (SLOT_2B.label, "statuses run out"),
                (SLOT_2B.label, "stack at its cap ticked")]
        for cause in ("burning", "poisoned", "suffocating", "regen"):
            if not c.get((SLOT_2B.label, "UnitHitEvent", cause)):
                fail(f"scene status: no {cause} hit (vacuous)")
    for key in need:
        if not c.get(key):
            fail(f"scene {kind}: no {'/'.join(key)} (vacuous)")


def _harmless(kind):
    """The hooked match (the harness around the slots) equals the same match
    without it, every tick (per-tick unit hash and field digest)."""
    if kind == "ship":
        a, b = ShipDriver(), ShipDriver()
        ticks = min(SHIP_TICKS, 200)
    else:
        a, b = LaneDriver(kind), LaneDriver(kind)
        ticks = min(LANE_TICKS, 200)
    ls = ShadowLockstep([a], SLOTS)
    with ls:
        for t in range(ticks):
            ls.tick()
            b.tick()
            if not _same(a.sim, b.sim):
                fail(f"harmless {kind}: the hooked match diverged at tick {t}")
                return
    print(f"harmless {kind}: the hooked match equals the unhooked one for "
          f"{ticks} ticks")


def part_batch():
    for kind in ("fire", "gas", "status"):
        for n in BATCH_NS:
            drivers = [LaneDriver(kind, variant=e) for e in range(n)]
            pr = Probe()
            run_lockstep(f"batch {kind} N={n}", drivers, SLOTS, BATCH_TICKS,
                         probe=pr)
            if len({len(d.sim.units) for d in drivers}) < 2:
                fail(f"batch {kind} N={n}: every env has the same unit count")


# ---------------------------------------------------------------------------
# PART OVERFLOW
# ---------------------------------------------------------------------------
def part_overflow():
    """A ring of capacity 2 on the status scene's busiest tick."""
    drv = LaneDriver("status")
    from simulation import status as S
    tested = 0
    for t in range(80):
        sim = drv.sim
        getattr(drv, "_before_status")()
        U = len(sim.units)
        # run the host tick; at slot 2b's entry take its rows, then its events
        evs = []
        held = {}
        orig = S.tick_statuses

        def spy(units, events=None, __o=orig):
            held["pre"] = unit_store.pack(sim.units, U)
            k0 = len(events) if events is not None else 0
            r = __o(units, events)
            evs.extend(events[k0:] if events is not None else [])
            return r
        import simulation.simulation as simmod
        simmod.tick_statuses = spy
        try:
            sim.step()
            sim.set_paused(False)
        finally:
            simmod.tick_statuses = orig
        drv.k += 1
        pre = held["pre"]
        if len(evs) < 3:
            continue
        cap = 2
        st = unit_store.DeviceStore(1, U, tables=unit_tables.values_batch([sim]))
        ring = unit_events.EventRing(1, U, e_max=cap)
        st.upload(pre[None])
        ring.begin()
        try:
            unit_kernels.status_tick(st, ring)
            fail("overflow: the Python face did not refuse a run that dropped "
                 "events")
        except unit_events.UnitEventOverflow as ex:
            if "status_tick" not in str(ex) or "env(s) [0]" not in str(ex):
                fail(f"overflow: the refusal does not name the kernel and env: {ex}")
        recs, counts, dropped = ring.download()
        want = [unit_events.record_of(e) for e in evs]
        if int(counts[0]) != cap or int(dropped[0]) != len(want) - cap:
            fail(f"overflow: count {int(counts[0])} / dropped {int(dropped[0])} != "
                 f"{cap} / {len(want) - cap}")
        got = [tuple(int(x) for x in recs[0, i]) for i in range(cap)]
        if got != want[:cap]:
            fail(f"overflow: the records within the capacity {got} != the host's "
                 f"first {want[:cap]}")
        if int(st.last_dropped[0]) != len(want) - cap:
            fail("overflow: store.last_dropped is not the dropped count")
        tested += 1
        print(f"overflow: tick {sim.tick}: {len(want)} events into a ring of "
              f"{cap}: {cap} written (the host's first), {int(dropped[0])} "
              f"dropped and counted, the run refused by name")
        break
    if tested == 0:
        fail("overflow: no tick of the status scene emitted 3 events (vacuous)")


# ---------------------------------------------------------------------------
# PART CHAINED
# ---------------------------------------------------------------------------
def _chained(name, make, slots, ticks, cov_label):
    twin = make()
    ref = make()
    t0 = time.perf_counter()
    n = 0
    with ChainedSlots(twin, slots) as ch:
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
    k = [s.label for s in slots].index(cov_label)
    acted = sum(1 for covs in ch.coverage if int(covs[k][0]) > 0)
    n_ev = sum(len(r) for r in ch.events)
    print(f"{name}: {n} ticks, the chained twin equal to the reference every tick "
          f"({ch.runs} chain runs; [{cov_label}] changed rows on {acted} ticks; "
          f"{n_ev} ring events appended; {time.perf_counter() - t0:.0f} s)")
    if ch.runs != n:
        fail(f"{name}: the chain ran {ch.runs} times in {n} ticks")
    if acted == 0 or n_ev == 0:
        fail(f"{name}: [{cov_label}] never acted or no event was appended (vacuous)")


def part_chained():
    if "2b" in CHAINS:
        _chained("chained 2b (status lanes)", lambda: LaneDriver("status"),
                 [SLOT_2B], LANE_TICKS, SLOT_2B.label)
        _chained("chained 2b (ship)", lambda: ShipDriver(), [SLOT_2B], SHIP_TICKS,
                 SLOT_2B.label)
    if "9c" in CHAINS:
        for kind in ("fire", "gas"):
            _chained(f"chained 9c + 9c2 + 9c3 ({kind} lanes)",
                     lambda kind=kind: LaneDriver(kind),
                     [SLOT_9C, SLOT_9C2, SLOT_9C3], LANE_TICKS,
                     SLOT_9C.label if kind == "fire" else SLOT_9C3.label)
        _chained("chained 9c + 9c2 + 9c3 (ship)", lambda: ShipDriver(),
                 [SLOT_9C, SLOT_9C2, SLOT_9C3], SHIP_TICKS, SLOT_9C2.label)


# ---------------------------------------------------------------------------
# main / bench
# ---------------------------------------------------------------------------
PARTS = ("tables", "scenes", "batch", "overflow", "chained")


def main(parts=None) -> int:
    if not getattr(bp, "HAS_CUDA", False):
        print("UNIT_DAMAGE_RESULT: FAIL (not the CUDA build)")
        return 1
    parts = parts or os.environ.get("UNIT_DAMAGE_PARTS", ",".join(PARTS)).split(",")
    t0 = time.perf_counter()
    for p in parts:
        try:
            {"tables": part_tables, "scenes": part_scenes, "batch": part_batch,
             "overflow": part_overflow, "chained": part_chained}[p]()
        except Exception as ex:          # a part's crash fails it, the rest run
            import traceback
            traceback.print_exc()
            fail(f"part {p} raised {type(ex).__name__}: {ex}")
    print(f"total {time.perf_counter() - t0:.0f} s")
    ok = not FAILS
    print("UNIT_DAMAGE_RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def bench(reps: int = 200) -> int:
    """The TIMING table: per kernel, upload + kernel + download and the kernel
    call alone (launch + kernel + the ring's gather + the counts' D2H and
    sync), medians over ``reps``, at N = 1, 8, 64 (env rows replicated from
    the gas scene after 40 ticks)."""
    import cupy as cp
    drv = LaneDriver("gas")
    sim = drv.sim
    names = tuple("field:" + f for f in unit_kernels.HEAT_FIELDS
                  + unit_kernels.GAS_FIELDS) + ("credit_claims",)
    for _ in range(40):
        drv.tick()
    rows1 = unit_store.pack(sim.units)
    U = rows1.shape[0]
    for n in (1, 8, 64):
        rows = np.ascontiguousarray(np.broadcast_to(rows1, (n,) + rows1.shape))
        st = unit_store.DeviceStore(n, U, tables=np.tile(unit_tables.values(sim), (n, 1)))
        ring = unit_events.EventRing(n, U)
        inp = gather(names, [capture(names, sim)] * n)
        claims = inp.tables["credit_claims"]
        kern = {
            "2b status_tick": lambda: unit_kernels.status_tick(st, ring),
            "9c heat_damage": lambda: unit_kernels.heat_damage(st, ring, inp.fields,
                                                               claims),
            "9c3 gas_rows": lambda: unit_kernels.gas_rows(st, ring, inp.fields, claims),
        }
        for name, f in kern.items():
            full, alone = [], []
            for _ in range(reps):
                t0 = time.perf_counter()
                st.upload(rows)
                ring.begin()
                t1 = time.perf_counter()
                f()
                t2 = time.perf_counter()
                st.download()
                ring.download()
                t3 = time.perf_counter()
                full.append(t3 - t0)
                alone.append(t2 - t1)
            print(f"TIMING N={n:3d} U={U} {name:16s} upload+kernel+download "
                  f"{1e3 * float(np.median(full)):.3f} ms  kernel call "
                  f"{1e3 * float(np.median(alone)):.3f} ms")
        del st, ring
        cp.get_default_memory_pool().free_all_blocks()
    return 0


if __name__ == "__main__":
    sys.exit(main())
