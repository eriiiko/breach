"""#29 P6.2 -- THE UNIT KERNELS GATE: the shared layer (the device tables, the
device unit library) and the slot kernels 1a, 9c2, 1b', 10', the tick-end
conversion and 1c again (moved), each against its Python law at tol 0.

Runs inside the GPU subprocess (tests/cuda_harness.run_cuda_script; wrapper
tests/test_cuda_unit_kernels.py). The store is ``simulation.unit_store``'s
``DeviceStore`` with its device tables (``simulation.unit_tables``); the
kernels are ``simulation.unit_kernels`` (C++ ``cpp/src/cuda_unit_*.cu`` over
``cpp/src/cuda_unit_lib.cuh``); the harness is ``tests/cuda_unit_lockstep.py``.

PART TABLES -- every uploaded table value read back from the device equals the
reference object's own value (the impulse table, the turning dials, the
stabilities, the status registry, the exposure decay, the class hp, the
zombie team, the gas planes), per env of a batch whose maps differ in tile
size (the impulse table is per map).

PART LIB -- the device unit library on synthetic rows over their whole range:
composed_flags (both moments), apply_status for every kind (the refresh /
max / stack rules, the stack cap's refusal and its count), the sweep, and the
hook accessor (every hook, values from ONE/8 to 8 ONE, class values to 2^40,
negatives), against status.py / unit_traits.py; padding rows (valid 0)
carrying the same garbage come back untouched.

PART SHIP -- the SHADOW LOCKSTEP of the six slots at once on ship_demo's bots
match (seed 1) with NON-NEUTRAL HOOKS (TURN, HOLD, HP on two thirds of the
marines) and an ORDER SCHEDULE (order_intent every few ticks: issued,
replaced in the tick, cleared, expiring after 20 or 40 ticks): every tick the
device rows equal the host's after each slot (every column), each kernel's
coverage == the rows its host slot changed; and the hooked match equals an
unhooked twin (per-tick unit hash and field digest) -- hooking is harmless.

PART DOORWAY -- slots 1a and 9c2 on tests/test_wind_doorway_e2e.py's scene (an
over-pressured ship_demo room vents through its doorway onto a squad: the jet
floors the squad -- knockdowns -- and pushes the prone bodies, whose standing
surplus the wind row keeps), N = 1 with an unhooked twin, then N = 2 and 8
(env e: 2 + e % 3 squad units, so U differs and padding rows ride along;
hooks TURN / HOLD differ per unit); env i of the batch == env i alone.

PART BATCH -- 1b', 10', 1c on ship_demo matches of different seeds (extra
idle units per env: padding), and the conversion, 10' and 1a on a box level
where a horde bites idle marines (conversions; HP hooks), N = 2 and 8.

PART CHAINED -- the two adjacent pairs this patch creates replace the host's
slots in a twin match: 1b' + 1c (ship_demo with the order schedule) and 10' +
the conversion (the box): pack at the first slot's entry, both kernels in a
row, download, unpack, the host-only side by the harness (the request table
cleared; a converted unit's name and books); the twin's per-tick unit hash and
field digest equal an untouched reference match's on every tick.

PROPERTY: every device slot kernel is its Python law on every valid row,
integer for integer, in a batch as alone, and it acts in the scenes (each
kernel's coverage > 0 on a stated share of ticks). BREAKS IF a kernel's
arithmetic differs from the law (a rounding, a dropped term, a hook ignored),
it touches a padding row or another env's rows, a table value is not the
reference's, or a chained pair leaves the match off the reference's path.
Each validated by breaking it (the patch's report).

Prints per-part lines and ``UNIT_KERNELS_RESULT: PASS`` / ``FAIL``; exits 0 / 1.
``bench()`` (a separate entry) prints the TIMING table.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

import breach_physics as bp   # the CUDA build (cuda_harness bootstrap): FIRST

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "tools") not in sys.path:
    sys.path.append(str(_ROOT / "tools"))   # appended: never ahead of the CUDA build

import level_loader  # noqa: E402
from cuda_unit_lockstep import (ChainedSlots, DeviceSlot,  # noqa: E402
                                ShadowLockstep)
from simulation import (unit_kernels, unit_schema, unit_store,  # noqa: E402
                        unit_tables, unit_traits)

FAILS: list = []
SHIP_TICKS = int(os.environ.get("UNIT_KERNELS_SHIP_TICKS", "500"))
DOOR_TICKS = int(os.environ.get("UNIT_KERNELS_DOOR_TICKS", "160"))
BATCH_TICKS = int(os.environ.get("UNIT_KERNELS_BATCH_TICKS", "160"))
BOX_TICKS = int(os.environ.get("UNIT_KERNELS_BOX_TICKS", "400"))
ONE = unit_traits.ONE


def fail(msg):
    FAILS.append(msg)
    print("FAIL:", msg)


# ---------------------------------------------------------------------------
# The slots
# ---------------------------------------------------------------------------
SLOT_1A = DeviceSlot(label="1a integrate + turn", run=unit_kernels.integrate_turn)
SLOT_1B2 = DeviceSlot(label="1b' orders",
                      run=lambda st, inp: unit_kernels.apply_orders(
                          st, inp.tables["order_requests"]),
                      inputs=("order_requests",))
SLOT_1C = DeviceSlot(label="1c exposure decay", run=unit_kernels.exposure_decay)
SLOT_9C2 = DeviceSlot(label="9c2 wind on units",
                      run=lambda st, inp: unit_kernels.wind_impulse(st, inp.fields),
                      inputs=tuple("field:" + f for f in unit_kernels.WIND_FIELDS))
SLOT_10E = DeviceSlot(label="10' order expiry",
                      run=lambda st, inp: unit_kernels.expire_orders(st, inp.tick),
                      inputs=("tick",))
SLOT_CONV = DeviceSlot(label="10 tick end (ruleset)",
                       run=unit_kernels.convert_zombies)


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
    {"TURN": ONE * 5 // 4, "HOLD": ONE * 3 // 4, "HP": ONE * 5 // 4},
    {"TURN": ONE * 3 // 4, "HOLD": ONE * 3 // 2, "HP": ONE * 3 // 4},
)


def _hook(sim, units):
    """Recompile two thirds of ``units`` with non-neutral hooks (the compile
    step, at spawn -- before the first tick)."""
    for i, u in enumerate(units):
        h = HOOK_SETS[i % 3]
        if h is not None:
            unit_traits.compile_unit(sim, u, h)


class ShipDriver:
    """ship_demo's bots match (MatchSession) with hooks and, when ``orders``,
    the order schedule issued through ``order_intent`` before each tick;
    ``extra`` idle marines added at spawn (padding in a batch)."""

    def __init__(self, seed=1, orders=True, hooks=True, extra=0):
        from simulation.match_session import MatchSession
        level = level_loader.load("ship_demo", str(_ROOT / "levels"))
        self.session = MatchSession(level, seed=seed, breach_physics=bp)
        self.sim = self.session.sim
        self.orders = orders
        self.n_rooms = len(level.rooms.rooms)
        self.issued = 0
        if extra:
            from simulation.unit import Unit
            base = [u for u in self.sim.units if not u.is_zombie][0]
            added = 0
            for dy in range(-12, 13, 4):
                for dx in range(-12, 13, 4):
                    if added >= extra:
                        break
                    x, y = base.tile_x + dx, base.tile_y + dy
                    if self.sim.gmap.is_passable_block(y, x, 3):
                        self.sim.add_unit(Unit(f"X{added}", x=float(x), y=float(y),
                                               team=int(base.team),
                                               env_profile="default"))
                        added += 1
        if hooks:
            _hook(self.sim, [u for u in self.sim.units if not u.is_zombie])
        self.k = 0

    def _orders(self):
        sim = self.sim
        t = self.k
        marines = [u for u in sim.units if u.alive and not u.is_zombie]
        for i, u in enumerate(marines):
            enemies = [v for v in sim.units if v.alive and v.team != u.team]
            if (t + 7 * i) % 23 == 0:
                what = 1 + (t // 23 + i) % 4
                where = (t + i) % self.n_rooms
                ticks = ((t + i) % 3) * 20
                target = int(enemies[0].id) if (enemies and i % 3 == 0) else -1
                sim.order_intent(u.id, what, where, ticks, target)
                self.issued += 1
                if (t + i) % 2 == 0:          # replaced within the tick
                    sim.order_intent(u.id, 1 + (what % 4), (where + 1) % self.n_rooms,
                                     20, -1)
                    self.issued += 1
            elif (t + 5 * i) % 61 == 0:
                sim.order_intent(u.id, 0, -1, 0, -1)
                self.issued += 1

    def tick(self):
        if self.orders:
            self._orders()
        self.session.tick()
        self.k += 1

    def is_terminal(self):
        return self.session.is_terminal()


# tests/test_wind_doorway_e2e.py's scene (kept in step with it).
ROOM_ROWS = (18, 36)
ROOM_COLS = (41, 58)
WEST_WALL = 40
UPPER_DOOR = range(21, 25)
LOWER_DOOR = range(29, 33)


class DoorwayDriver:
    """The over-pressured room venting through its doorway onto a squad of
    ``n_squad`` (hooked) and a walker holding MOVE_DIR west."""

    def __init__(self, n_squad=3, hooks=True):
        from control_source import quantize_stick_direction
        from simulation import Simulation
        from simulation.field_edit import EditMode, FieldEdit, Region
        from simulation.materials import MAT_DOOR
        from simulation.ruleset import ContinuousRealtime
        from simulation.unit import Unit
        level = level_loader.load("ship_demo", str(_ROOT / "levels"))
        sim = Simulation(level, seed=1, breach_physics=bp, enable_recorder=False,
                         ruleset=ContinuousRealtime())
        g = sim.gmap
        sim.edit_queue.enqueue(FieldEdit(
            "atmosphere", Region.RECT,
            (ROOM_ROWS[0], ROOM_COLS[0], ROOM_ROWS[1], ROOM_COLS[1]),
            0.44, EditMode.ADD))
        sim.set_paused(False)
        for _ in range(30):
            sim.step()
        for (r, c) in np.argwhere(g.material == MAT_DOOR).tolist():
            on_room = (ROOM_ROWS[0] - 1 <= r <= ROOM_ROWS[1] + 1
                       and ROOM_COLS[0] - 1 <= c <= ROOM_COLS[1] + 1)
            if not on_room:
                g.destroy_wall(r, c)
        for _ in range(24):
            sim.step()
        spots = [(37.0, 20.0), (37.0, 22.0), (34.0, 21.0), (34.0, 24.0),
                 (37.0, 25.0)]
        squad = []
        for i in range(n_squad):
            x, y = spots[i]
            u = Unit(f"sq_{i}", x=x, y=y, team=0)
            sim.add_unit(u, hooks=HOOK_SETS[i % 3] if hooks else None)
            squad.append(u)
        walker = Unit("walker", x=42.0, y=29.0, team=0)
        sim.add_unit(walker)
        for r in list(UPPER_DOOR) + list(LOWER_DOOR):
            g.destroy_wall(r, WEST_WALL)
        self.sim, self.squad, self.walker = sim, squad, walker
        self.move = quantize_stick_direction(-1.0, 0.0, 0.0)[:2]

    def tick(self):
        self.sim.set_move_dir(self.walker.id, self.move[0], self.move[1], 0)
        self.sim.step()


_BOX_CACHE: dict = {}


def _box_level(n_marines, n_zombies):
    """A hull box: two idle marine teams of ``n_marines`` and a horde of
    ``n_zombies`` that walks to them and bites (tests/_p3a_sole_cause.py's
    box_conversions, sized)."""
    key = (n_marines, n_zombies)
    if key in _BOX_CACHE:
        return _BOX_CACHE[key]
    import level_lib
    from PIL import Image
    from simulation.factions import ZOMBIE_TEAM
    from simulation.match import MatchSpec, MatchTeam, SpawnRoom
    from simulation.materials import MAT_AIR, MAT_HULL
    idle = MatchTeam(n=n_marines, loadout="rifleman", env_profile="default",
                     controller="idle")
    rooms = [SpawnRoom(team=0, x=2, y=2, w=8, h=8, max_per_team=4),
             SpawnRoom(team=1, x=30, y=14, w=8, h=8, max_per_team=4),
             SpawnRoom(team=ZOMBIE_TEAM, x=12, y=2, w=8, h=8, max_per_team=4)]
    spec = MatchSpec(teams=(idle, idle), seed=1, zombies=n_zombies,
                     time_limit_s=60.0)
    tmp = Path(tempfile.mkdtemp(prefix="p62_box_"))
    lvl = tmp / "box"
    lvl.mkdir()
    h, w = 24, 40
    level_lib.write_level_header(lvl, name="box", tile_size_m=1.0)
    tm = np.full((h, w), MAT_HULL, dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = MAT_AIR
    level_lib.write_tilemap_csv(lvl, tm, csv_bak=False)
    Image.new("RGB", (w, h)).save(lvl / "diffuse.png")
    level_lib.write_spawn_rooms(lvl / "level.toml", rooms)
    level_lib.write_match(lvl, spec)
    level = level_loader.load("box", str(tmp))
    _BOX_CACHE[key] = level
    return level


class BoxDriver:
    """The box match; marines hooked (HP among them: the conversion reads it)."""

    def __init__(self, n_marines=2, n_zombies=3, seed=1, hooks=True):
        from simulation.match_session import MatchSession
        self.session = MatchSession(_box_level(n_marines, n_zombies), seed=seed,
                                    breach_physics=bp)
        self.sim = self.session.sim
        if hooks:
            _hook(self.sim, [u for u in self.sim.units if not u.is_zombie])

    def tick(self):
        self.session.tick()

    def is_terminal(self):
        return self.session.is_terminal()


# ---------------------------------------------------------------------------
# The lockstep runner
# ---------------------------------------------------------------------------
def run_lockstep(name, drivers, slots, ticks, twins=None, min_share=None,
                 probes=None):
    """Shadow ``slots`` on ``drivers`` for up to ``ticks`` ticks; every tick
    each slot equal and its coverage == the host's changed rows; ``twins``
    (unhooked copies) must stay equal; returns the per-label share of
    env-ticks with coverage > 0 (each must reach ``min_share[label]``).
    ``probes(rep_by_label, ls)`` collects scene facts."""
    t0 = time.perf_counter()
    cov = {s.label: 0 for s in slots}
    total = 0
    n = 0
    ok = True
    with ShadowLockstep(drivers, list(slots)) as ls:
        for _ in range(ticks):
            if any(getattr(d, "is_terminal", lambda: False)() for d in drivers):
                break
            reps = ls.tick()
            n += 1
            total += len(drivers)
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
            if not ok:
                break
            if probes is not None:
                probes(reps, ls)
            if twins is not None:
                for d, tw in zip(drivers, twins):
                    tw.tick()
                    if not _same(d.sim, tw.sim):
                        fail(f"{name}: the hooked match diverged from the "
                             f"unhooked one at tick {d.sim.tick}")
                        ok = False
                if not ok:
                    break
    shares = {lab: cov[lab] / max(1, total) for lab in cov}
    U = ls.U
    counts = [len(d.sim.units) for d in drivers]
    print(f"{name}: N={len(drivers)} units per env {counts} (U {U}), {n} ticks"
          f"{' (twin equal every tick)' if twins is not None and ok else ''}"
          f"{', env i == env i alone' if len(drivers) > 1 and ok else ''}; "
          f"coverage > 0 on: " + ", ".join(
              f"[{lab}] {100 * s:.1f} %" for lab, s in shares.items())
          + f" of env-ticks ({time.perf_counter() - t0:.0f} s)")
    for lab, want in (min_share or {}).items():
        if shares.get(lab, 0.0) < want:
            fail(f"{name} [{lab}]: coverage > 0 on {100 * shares.get(lab, 0):.1f} % "
                 f"of env-ticks, below the stated {100 * want:.1f} % (vacuous scene)")
    return shares


# ---------------------------------------------------------------------------
# PART TABLES
# ---------------------------------------------------------------------------
def part_tables():
    from simulation import exposure
    from simulation.factions import ZOMBIE_TEAM
    from simulation.gases import INERT_N2, O2
    from simulation.loadouts import get_tables
    from simulation.species import SPECIES_REGISTRY, ZOMBIE_STABILITY
    from simulation.status import STATUS_REGISTRY
    from simulation.unit_fixed import quantize_scalar as qs
    a = ShipDriver(orders=False, hooks=False)
    b = BoxDriver(hooks=False)                 # 1 m tiles: another impulse table
    sims = [a.sim, b.sim]
    tab = unit_tables.values_batch(sims)
    st = unit_store.DeviceStore(2, 4, tables=tab)
    got = st.tables.download()
    if not np.array_equal(got, tab):
        fail("tables: the device tables differ from the uploaded ones")
    n = 0
    for e, sim in enumerate(sims):
        t = sim.impulse_table
        want = {
            "impulse_cap_q": t.cap_q, "impulse_cap_prone_q": t.cap_prone_q,
            "impulse_zombie_hold_q16": t.zombie_hold_q16,
            "impulse_v_inplace_q": tuple(t.v_inplace_q),
            "impulse_v_fall_q": tuple(t.v_fall_q),
            "impulse_react_ticks": t.react_ticks, "impulse_n_rec": t.n_rec,
            "impulse_inertia_g": t.inertia_g, "impulse_getup_ticks": t.getup_ticks,
            "impulse_wind_k_q": t.wind_k_q, "impulse_wind_k_prone_q": t.wind_k_prone_q,
            "turn_rate_class_q16": sim._marine_turn_rate_q16,
            "turn_accel_class_q16": sim._marine_turn_accel_q16,
            "stability_zombie_q16": qs(ZOMBIE_STABILITY),
            "stability_profile_q16": tuple(
                qs(p.stability) for p in get_tables().env_profiles.by_name.values()),
            "stability_species_q16": tuple(
                qs(s.environment.stability) for s in SPECIES_REGISTRY.values()),
            "status_flags": tuple(
                (0 if r.can_move else 1) | (0 if r.can_act else 2)
                | (0 if r.can_aim else 4) | (8 if r.is_prone else 0)
                for r in STATUS_REGISTRY),
            "status_stacking": tuple({"refresh": 0, "stack": 1, "max": 2}[r.stacking]
                                     for r in STATUS_REGISTRY),
            "status_stack_cap": unit_schema.STATUS_STACK_CAP,
            "exposure_decay_q": exposure.get_settings().decay_q,
            "hp_class_marine_q": unit_traits.class_hp_q(False),
            "hp_class_zombie_q": unit_traits.class_hp_q(True),
            "zombie_team": ZOMBIE_TEAM, "gas_o2": O2, "gas_n2": INERT_N2,
        }
        missing = [r.name for r in unit_tables.ROWS if r.name not in want]
        if missing:
            fail(f"tables: no reference value checked for rows {missing}")
        for k, v in want.items():
            dv = unit_tables.read(got[e], k)
            n += 1
            as_t = lambda x: tuple(x) if isinstance(x, (tuple, list)) else (x,)  # noqa: E731
            if as_t(dv) != as_t(v):
                fail(f"tables env {e} row {k}: device {dv} != reference {v}")
    if tab[0, unit_tables.layout().column("impulse_cap_q").offset] == \
            tab[1, unit_tables.layout().column("impulse_cap_q").offset] and \
            a.sim.gmap.tile_size_m != b.sim.gmap.tile_size_m:
        fail("tables: two maps of different tile size gave one impulse table")
    print(f"tables: {n} values over 2 envs equal the reference objects (W "
          f"{unit_tables.layout().width}; tile sizes "
          f"{a.sim.gmap.tile_size_m} / {b.sim.gmap.tile_size_m} m)")
    return a.sim


# ---------------------------------------------------------------------------
# PART LIB
# ---------------------------------------------------------------------------
def _random_statuses(rng, n_rows):
    """A list of StatusEffect a unit can hold (<= 1 of each non-stacking
    kind, <= the stack cap of a stacking one), fields over their range."""
    from simulation.status import STACK_STACK, STATUS_REGISTRY, StatusEffect
    out = []
    counts = {}
    big = [0, 1, -1, 65536, -65536, 2 ** 40, -(2 ** 40), 2 ** 62, -(2 ** 62)]
    for _ in range(n_rows):
        kind = int(rng.integers(0, len(STATUS_REGISTRY)))
        row = STATUS_REGISTRY[kind]
        cap = unit_schema.STATUS_STACK_CAP if row.stacking == STACK_STACK else 1
        if counts.get(kind, 0) >= cap:
            continue
        counts[kind] = counts.get(kind, 0) + 1
        mag = int(rng.choice(big)) if rng.random() < 0.3 else int(rng.integers(-5, 300000))
        ticks = int(rng.integers(-3, 50))
        src = None if rng.random() < 0.4 else int(rng.integers(0, 1000))
        out.append(StatusEffect(kind=kind, magnitude_q16=mag, remaining_ticks=ticks,
                                source_id=src))
    return out


def _block_rows(lists, n, U, rng):
    """(n, U, W) rows: the status blocks of ``lists`` on random garbage,
    valid on 70 % of rows (padding rows carry the same garbage)."""
    lay = unit_store.layout()
    sc = lay.column("statuses")
    a = rng.integers(-2 ** 40, 2 ** 40, size=(n, U, lay.width), dtype=np.int64)
    k = 0
    for e in range(n):
        for r in range(U):
            a[e, r, sc.offset:sc.offset + sc.width] = unit_schema.COLUMN_BY_NAME[
                "statuses"].encode(lists[k])
            k += 1
    a[:, :, lay.valid] = (rng.random((n, U)) < 0.7).astype(np.int64)
    return a


class _Stub:
    __slots__ = ("statuses", "traits")


def part_lib(table_vec):
    from simulation.status import (STATUS_REGISTRY, apply_status, composed_flags,
                                   survives_sweep)
    rng = np.random.default_rng(6202)
    lay = unit_store.layout()
    sc = lay.column("statuses")
    n, U = 3, 64
    st = unit_store.DeviceStore(n, U, tables=np.tile(table_vec, (n, 1)))
    enc = unit_schema.COLUMN_BY_NAME["statuses"]

    def decode(block):
        return enc.decode(tuple(int(x) for x in block))

    checks = 0
    # composed_flags, both moments
    for trial in range(6):
        lists = [_random_statuses(rng, int(rng.integers(0, 12))) for _ in range(n * U)]
        a = _block_rows(lists, n, U, rng)
        st.upload(a)
        for after in (0, 1):
            out, cov, flt, ref = unit_kernels.lib_probe(st, 0, after)
            for e in range(n):
                for r in range(U):
                    if not a[e, r, lay.valid]:
                        if out[e, r].any():
                            fail(f"lib flags: a padding row got output")
                        continue
                    s = _Stub()
                    s.statuses = decode(a[e, r, sc.offset:sc.offset + sc.width])
                    f = composed_flags(s, after_sweep=bool(after))
                    want = (f.can_move, f.can_act, f.can_aim, f.is_prone, False)
                    got = tuple(bool(x) for x in out[e, r, :5])
                    checks += 1
                    if got != want:
                        fail(f"lib flags (after_sweep {after}) env {e} row {r}: "
                             f"device {got} != {want}")
                        return
        if not np.array_equal(st.download(), a):
            fail("lib flags: composed_flags wrote the store")
    # apply_status: every kind, the three rules, the cap's refusal
    refused_seen = 0
    for kind in range(len(STATUS_REGISTRY)):
        for mag, dur, src in ((0, 1, -1), (65536 * 3, 17, 4), (-131072, 300, -1),
                              (2 ** 40, 2, 77)):
            lists = [_random_statuses(rng, int(rng.integers(0, 14))) for _ in range(n * U)]
            # some rows hold the stacking kind at the cap
            for i in range(0, n * U, 7):
                from simulation.status import StatusEffect
                lists[i] = [StatusEffect(kind=kind, magnitude_q16=1, remaining_ticks=3,
                                         source_id=None)
                            for _ in range(unit_schema.STATUS_STACK_CAP)]
            a = _block_rows(lists, n, U, rng)
            st.upload(a)
            out, cov, flt, ref = unit_kernels.lib_probe(st, 1, kind, mag, dur, src)
            got = st.download()
            want = a.copy()
            want_ref = np.zeros(n, dtype=np.int64)
            want_cov = np.zeros(n, dtype=np.int64)
            for e in range(n):
                for r in range(U):
                    if not a[e, r, lay.valid]:
                        continue
                    s = _Stub()
                    s.statuses = decode(a[e, r, sc.offset:sc.offset + sc.width])
                    before = list(enc.encode(s.statuses))
                    res = apply_status(s, kind, mag / 65536.0, dur,
                                       None if src < 0 else src)
                    if res is None:
                        want_ref[e] += 1
                    after = enc.encode(s.statuses)
                    want[e, r, sc.offset:sc.offset + sc.width] = after
                    if list(after) != before:
                        want_cov[e] += 1
                    exp_code = 1 if res is None else 0
                    if int(out[e, r, 0]) != exp_code:
                        fail(f"lib apply kind {kind}: env {e} row {r} result "
                             f"{int(out[e, r, 0])} != {exp_code}")
                        return
                    checks += 1
            if not np.array_equal(got, want):
                e, r, c = np.argwhere(got != want)[0].tolist()
                fail(f"lib apply kind {kind} ({mag}, {dur}, {src}): env {e} row {r} "
                     f"{lay.column_at(c)}: device {got[e, r, c]} != {want[e, r, c]}")
                return
            if not np.array_equal(ref, want_ref) or not np.array_equal(cov, want_cov):
                fail(f"lib apply kind {kind}: counts cov {cov.tolist()} / refused "
                     f"{ref.tolist()} != {want_cov.tolist()} / {want_ref.tolist()}")
                return
            refused_seen += int(want_ref.sum())
    if refused_seen == 0:
        fail("lib apply: no application was refused at the cap (vacuous)")
    # the sweep
    for trial in range(6):
        lists = [_random_statuses(rng, int(rng.integers(0, 14))) for _ in range(n * U)]
        a = _block_rows(lists, n, U, rng)
        st.upload(a)
        unit_kernels.lib_probe(st, 2)
        got = st.download()
        want = a.copy()
        for e in range(n):
            for r in range(U):
                if not a[e, r, lay.valid]:
                    continue
                ss = [x for x in decode(a[e, r, sc.offset:sc.offset + sc.width])
                      if survives_sweep(x)]
                want[e, r, sc.offset:sc.offset + sc.width] = enc.encode(ss)
                checks += 1
        if not np.array_equal(got, want):
            e, r, c = np.argwhere(got != want)[0].tolist()
            fail(f"lib sweep: env {e} row {r} {lay.column_at(c)}: device "
                 f"{got[e, r, c]} != {want[e, r, c]}")
            return
    # the hook accessor
    tc = lay.column("traits")
    hooks = np.array([unit_traits.HOOK_MIN, unit_traits.HOOK_MIN + 1, ONE // 3, ONE - 1,
                      ONE, ONE + 1, ONE * 3 // 2, 5 * ONE + 7, unit_traits.HOOK_MAX - 1,
                      unit_traits.HOOK_MAX], dtype=np.int64)
    for cls in (0, 1, -1, 3, 65535, 65536, -65537, 102944, 2 ** 24 + 5, -(2 ** 31),
                2 ** 40, -(2 ** 40) + 3):
        a = rng.integers(-2 ** 40, 2 ** 40, size=(n, U, lay.width), dtype=np.int64)
        a[:, :, tc.offset:tc.offset + tc.width] = rng.choice(hooks, size=(n, U, tc.width))
        a[:, :, lay.valid] = (rng.random((n, U)) < 0.7).astype(np.int64)
        st.upload(a)
        out, _c, _f, _r = unit_kernels.lib_probe(st, 3, cls)
        for e in range(n):
            for r in range(U):
                if not a[e, r, lay.valid]:
                    if out[e, r].any():
                        fail("lib hook: a padding row got output")
                    continue
                for h in range(unit_traits.N_HOOKS):
                    want = unit_traits.scaled_by(int(a[e, r, tc.offset + h]), cls)
                    checks += 1
                    if int(out[e, r, h]) != want:
                        fail(f"lib hook {unit_traits.HOOKS[h].name} x {cls}: device "
                             f"{int(out[e, r, h])} != {want}")
                        return
        if not np.array_equal(st.download(), a):
            fail("lib hook: the accessor wrote the store")
    print(f"lib: {checks} checks equal status.py / unit_traits.py (composed_flags "
          f"both moments, apply_status every kind with {refused_seen} refusals at "
          f"the cap, the sweep, the hook accessor); padding rows untouched")


# ---------------------------------------------------------------------------
# PART SHIP / DOORWAY / BATCH
# ---------------------------------------------------------------------------
def part_ship():
    drv = ShipDriver(seed=1)
    twin = ShipDriver(seed=1)
    issued0 = []

    def probes(reps, ls):
        issued0.append(drv.issued)
    run_lockstep("ship (6 slots, hooks, orders)", [drv],
                 [SLOT_1A, SLOT_1B2, SLOT_1C, SLOT_9C2, SLOT_10E, SLOT_CONV],
                 SHIP_TICKS, twins=[twin],
                 min_share={SLOT_1A.label: 0.9, SLOT_1B2.label: 0.2,
                            SLOT_9C2.label: 0.5, SLOT_10E.label: 0.02,
                            SLOT_1C.label: 0.005},
                 probes=probes)
    print(f"ship: {drv.issued} order_intent calls issued")


class _DoorProbe:
    """Scene facts of the doorway: knockdowns applied by 1a, prone rows 1a
    integrated, prone rows 9c2 gave a standing surplus."""

    def __init__(self):
        self.knock = self.prone_1a = self.prone_wind = 0

    def __call__(self, reps, ls):
        from simulation.status import KNOCKED_DOWN
        lay = ls.layout
        sc = lay.column("statuses")
        sx = lay.column("stand_x").offset
        k1 = [s.label for s in ls.slots].index(SLOT_1A.label)
        k9 = [s.label for s in ls.slots].index(SLOT_9C2.label)
        for e in range(ls.N):
            pre, post = ls._pre[k1][e], ls._post[k1][e]
            for r in range(pre.shape[0]):
                if not pre[r, lay.valid]:
                    continue
                kp = pre[r, sc.offset:sc.offset + sc.width][0::4]
                kq = post[r, sc.offset:sc.offset + sc.width][0::4]
                tp = pre[r, sc.offset:sc.offset + sc.width][2::4]
                tq = post[r, sc.offset:sc.offset + sc.width][2::4]
                if KNOCKED_DOWN in kp and (pre[r] != post[r]).any():
                    self.prone_1a += 1
                if KNOCKED_DOWN in kq and not np.array_equal(
                        np.r_[kp, tp], np.r_[kq, tq]):
                    self.knock += 1
            pre9, post9 = ls._pre[k9][e], ls._post[k9][e]
            for r in range(pre9.shape[0]):
                kp = pre9[r, sc.offset:sc.offset + sc.width][0::4]
                if pre9[r, lay.valid] and KNOCKED_DOWN in kp and \
                        pre9[r, sx] != post9[r, sx]:
                    self.prone_wind += 1


def part_doorway():
    for n in (1, 2, 8):
        drivers = [DoorwayDriver(n_squad=2 + (e % 3) if n > 1 else 3)
                   for e in range(n)]
        twins = [DoorwayDriver(n_squad=3)] if n == 1 else None
        pr = _DoorProbe()
        run_lockstep(f"doorway N={n}", drivers, [SLOT_1A, SLOT_9C2], DOOR_TICKS,
                     twins=twins, probes=pr,
                     min_share={SLOT_1A.label: 0.5, SLOT_9C2.label: 0.3})
        print(f"doorway N={n}: {pr.knock} knockdown applications by 1a, "
              f"{pr.prone_1a} prone rows integrated, {pr.prone_wind} prone rows "
              f"given a standing surplus by 9c2")
        if pr.knock == 0 or pr.prone_1a == 0 or pr.prone_wind == 0:
            fail(f"doorway N={n}: the scene did not knock down, integrate a prone "
                 f"body and push one with the wind (vacuous)")


def part_batch():
    for n in (2, 8):
        drivers = [ShipDriver(seed=1 + e, extra=e % 3) for e in range(n)]
        run_lockstep(f"ship batch N={n}", drivers, [SLOT_1B2, SLOT_10E, SLOT_1C],
                     BATCH_TICKS,
                     min_share={SLOT_1B2.label: 0.2, SLOT_10E.label: 0.02})
        if len({len(d.sim.units) for d in drivers}) < 2:
            fail(f"ship batch N={n}: every env has the same unit count (no padding)")
    for n in (2, 8):
        drivers = [BoxDriver(n_marines=2 + e % 2, n_zombies=2 + e % 3, seed=1 + e)
                   for e in range(n)]
        conv = {"n": 0}

        def probes(reps, ls, conv=conv):
            conv["n"] += int(reps[SLOT_CONV.label].coverage.sum())
        run_lockstep(f"box batch N={n}", drivers, [SLOT_10E, SLOT_CONV, SLOT_1A],
                     BOX_TICKS, probes=probes,
                     min_share={SLOT_1A.label: 0.5, SLOT_CONV.label: 0.0005})
        print(f"box batch N={n}: {conv['n']} conversions")
        if conv["n"] == 0:
            fail(f"box batch N={n}: no conversion (vacuous)")
        if len({len(d.sim.units) for d in drivers}) < 2:
            fail(f"box batch N={n}: every env has the same unit count (no padding)")


def part_box_twin():
    drv = BoxDriver()
    twin = BoxDriver()
    conv = {"n": 0}

    def probes(reps, ls):
        conv["n"] += int(reps[SLOT_CONV.label].coverage.sum())
    run_lockstep("box (conversion, hooks)", [drv], [SLOT_CONV, SLOT_10E, SLOT_1A],
                 BOX_TICKS, twins=[twin], probes=probes,
                 min_share={SLOT_CONV.label: 0.001})
    print(f"box: {conv['n']} conversions")


# ---------------------------------------------------------------------------
# PART CHAINED
# ---------------------------------------------------------------------------
def _chained(name, make, slots, host_side, ticks, cov_label):
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
    k = [s.label for s in slots].index(cov_label)
    acted = sum(1 for covs in ch.coverage if int(covs[k][0]) > 0)
    print(f"{name}: {n} ticks, the chained twin equal to the reference every tick "
          f"({ch.runs} chain runs; [{cov_label}] changed rows on {acted} ticks; "
          f"{time.perf_counter() - t0:.0f} s)")
    if ch.runs != n:
        fail(f"{name}: the chain ran {ch.runs} times in {n} ticks")
    if acted == 0:
        fail(f"{name}: [{cov_label}] never acted (vacuous)")


def _orders_host_side(sim, pre, post):
    sim._order_requests.clear()


def _conversion_host_side(sim, pre, post):
    lay = unit_store.layout()
    z = lay.column("is_zombie").offset
    for r, u in enumerate(sim.units):
        if not pre[r, z] and post[r, z]:
            u.name = f"Z-{u.name}"
            u.impulse_books = [0] * 8


def part_chained():
    _chained("chained 1b' + 1c (ship, orders)", lambda: ShipDriver(seed=1),
             [SLOT_1B2, SLOT_1C], _orders_host_side, SHIP_TICKS, SLOT_1B2.label)
    _chained("chained 10' + conversion (box)", lambda: BoxDriver(),
             [SLOT_10E, SLOT_CONV], _conversion_host_side, BOX_TICKS, SLOT_CONV.label)
    _chained("chained 10' + conversion (ship, orders)", lambda: ShipDriver(seed=1),
             [SLOT_10E, SLOT_CONV], _conversion_host_side, SHIP_TICKS, SLOT_10E.label)


# ---------------------------------------------------------------------------
# main / bench
# ---------------------------------------------------------------------------
PARTS = ("tables", "lib", "ship", "doorway", "batch", "box", "chained")


def main(parts=None) -> int:
    if not getattr(bp, "HAS_CUDA", False):
        print("UNIT_KERNELS_RESULT: FAIL (not the CUDA build)")
        return 1
    parts = parts or os.environ.get("UNIT_KERNELS_PARTS", ",".join(PARTS)).split(",")
    t0 = time.perf_counter()
    ship_sim = None
    for p in parts:
        try:
            if p == "tables":
                ship_sim = part_tables()
            elif p == "lib":
                if ship_sim is None:
                    ship_sim = ShipDriver(orders=False, hooks=False).sim
                part_lib(unit_tables.values(ship_sim))
            elif p == "ship":
                part_ship()
            elif p == "doorway":
                part_doorway()
            elif p == "batch":
                part_batch()
            elif p == "box":
                part_box_twin()
            elif p == "chained":
                part_chained()
            else:
                fail(f"no part {p!r}")
        except Exception as ex:          # a part's crash fails it, the rest run
            import traceback
            traceback.print_exc()
            fail(f"part {p} raised {type(ex).__name__}: {ex}")
    print(f"total {time.perf_counter() - t0:.0f} s")
    ok = not FAILS
    print("UNIT_KERNELS_RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def bench(reps: int = 200) -> int:
    """The TIMING table: per kernel, upload + kernel + download and the kernel
    call alone (launch + kernel + the counts' D2H and sync), medians over
    ``reps``, at N = 1, 8, 64 (env rows replicated from a live ship_demo
    match after 300 ticks with the order schedule)."""
    import cupy as cp
    from cuda_unit_lockstep import capture, gather
    drv = ShipDriver(seed=1)
    for _ in range(300):
        drv.tick()
    sim = drv.sim
    rows1 = unit_store.pack(sim.units)
    U = rows1.shape[0]
    req = unit_kernels.requests_of(sim)
    for n in (1, 8, 64):
        rows = np.ascontiguousarray(np.broadcast_to(rows1, (n,) + rows1.shape))
        st = unit_store.DeviceStore(n, U, tables=np.tile(unit_tables.values(sim), (n, 1)))
        inp = gather(tuple("field:" + f for f in unit_kernels.WIND_FIELDS),
                     [capture(tuple("field:" + f for f in unit_kernels.WIND_FIELDS),
                              sim)] * n)
        reqs = unit_kernels.pad_requests([req] * n)
        ticks = np.full(n, int(sim.tick), dtype=np.int64)
        kern = {
            "1a integrate_turn": lambda: unit_kernels.integrate_turn(st),
            "9c2 wind_impulse": lambda: unit_kernels.wind_impulse(st, inp.fields),
            "1b' apply_orders": lambda: unit_kernels.apply_orders(st, reqs),
            "1c exposure_decay": lambda: unit_kernels.exposure_decay(st),
            "10' expire_orders": lambda: unit_kernels.expire_orders(st, ticks),
            "10 convert_zombies": lambda: unit_kernels.convert_zombies(st),
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
                t3 = time.perf_counter()
                full.append(t3 - t0)
                alone.append(t2 - t1)
            print(f"TIMING N={n:3d} U={U} {name:20s} upload+kernel+download "
                  f"{1e3 * float(np.median(full)):.3f} ms  kernel call "
                  f"{1e3 * float(np.median(alone)):.3f} ms")
        del st
        cp.get_default_memory_pool().free_all_blocks()
    return 0


if __name__ == "__main__":
    sys.exit(main())
