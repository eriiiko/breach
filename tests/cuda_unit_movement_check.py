"""#29 P6.2m -- THE SLOT-3 GATE: movement (the continuous path's
``Simulation._update_player_movement``) as a unit kernel on the device store,
against its Python law at tol 0.

Runs inside the GPU subprocess (tests/cuda_harness.run_cuda_script; wrapper
tests/test_cuda_unit_movement.py). The kernel is ``simulation.unit_kernels.
movement`` (C++ ``cpp/src/cuda_unit_movement.cu`` over the walk's functions in
``cpp/src/cuda_unit_lib.cuh``); its field input is ``field:material`` (the
resident ``GameMap.material`` planes, int8) and its tables the speed table,
the effort edge and the materials' mobility column
(``simulation.unit_tables``); the harness is ``tests/cuda_unit_lockstep.py``.

PART LIB -- the walk's library functions (default_speed's reduction, the
direction rule, the effort scale, the facing cosine, the block test and the
footprint walk) on synthetic rows over their whole proved range, against
``movement.default_speed`` / ``move_speed_q16`` / ``facing_cos_q16``,
``unit_impulse.effort_scale_q16`` and ``GameMap.is_passable_block`` /
``footprint_mobility``; out-of-range operands are refused (bad), never wrapped.

PART SHIP -- the SHADOW LOCKSTEP of slot 3 on ship_demo's bots match (seed 1,
500 ticks) with non-neutral WALK / SPRINT / TURN hooks on two thirds of the
marines; the hooked match equals the same match with no harness every tick.

PART BOX -- a movement box built for this gate (hull, a steel wall pair, a
furniture patch -- mobility 400 --, an idle horde) where every branch of the
law acts: walkers at throttle 0, ONE and between, sideways / backward against a
held aim, walkers with no aim (their facing target written), across the
furniture, a walk ban after a push, statuses that take the walk away, pushed
velocities alone into the walls on each axis, multi-substep pushes,
knockdowns, zombies displaced by a push, a dead unit, WALK / SPRINT hooks;
N = 1 with its harness-free twin, then N = 2 and 8 (env e: layout variant e %
3 with its own tile size, so its material plane and speed table differ; unit
counts differ, so padding rows ride along); env i of N == env i alone. Every
branch is COUNTED (the census) and must be above 0.

PART DOORWAY -- the P6.2 doorway jet (prone bodies in wind, pushed against
the walls) at N = 1 with its twin, and N = 2.

PART CHAINED -- slot 3 replaced by the device in a twin match (ship_demo bots,
500 ticks; the box, 400 ticks): pack at the slot's entry, the kernel,
download, unpack, then the host-only side (the momentum books' WALL entries,
the legacy WEGO path step's path_tick_offset); the twin's per-tick unit hash
and field digest equal an untouched reference's on every tick.

PROPERTY: the slot-3 kernel is the continuous path's Python law on every
valid row, integer for integer, in a batch as alone; it acts in the scenes
(coverage == the rows the host changed, every branch counted above 0); slot 3
on the device keeps a match on the reference's path. BREAKS IF the walk's
arithmetic differs from the law (a hook ignored, a rounding, a slide branch,
the wall's zeroing of pv, the substep count), the kernel touches a padding row
or another env's rows or reads env 0's plane for every env, or a table value
is not the reference's own. Each validated by breaking it (the patch's report).

Prints per-part lines and ``UNIT_MOVEMENT_RESULT: PASS`` / ``FAIL``; exits 0 /
1. ``bench()`` (a separate entry) prints the TIMING table.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import breach_physics as bp   # the CUDA build (cuda_harness bootstrap): FIRST

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "tools") not in sys.path:
    sys.path.append(str(_ROOT / "tools"))   # appended: never ahead of the CUDA build

import level_loader  # noqa: E402
from cuda_unit_kernels_check import (DoorwayDriver, ShipDriver,  # noqa: E402
                                     _same)
from cuda_unit_lockstep import (ChainedSlots, DeviceSlot,  # noqa: E402
                                ShadowLockstep)
from simulation import (movement, unit_impulse, unit_kernels,  # noqa: E402
                        unit_store, unit_tables, unit_traits)
from simulation.unit_fixed import cos_q16, sin_q16  # noqa: E402

FAILS: list = []
SHIP_TICKS = int(os.environ.get("UNIT_MOVEMENT_SHIP_TICKS", "500"))
BOX_TICKS = int(os.environ.get("UNIT_MOVEMENT_BOX_TICKS", "400"))
BATCH_TICKS = int(os.environ.get("UNIT_MOVEMENT_BATCH_TICKS", "240"))
DOOR_TICKS = int(os.environ.get("UNIT_MOVEMENT_DOOR_TICKS", "160"))
BATCH_NS = [int(x) for x in os.environ.get("UNIT_MOVEMENT_BATCH_NS", "2,8").split(",")]
CHAINS = [int(x) for x in os.environ.get("UNIT_MOVEMENT_CHAINS", "0,1").split(",")]
ONE = unit_traits.ONE
TWO_PI_Q16 = 2 * 205887


def fail(msg):
    FAILS.append(msg)
    print("FAIL:", msg)


SLOT_3 = DeviceSlot(label="3 movement",
                    run=lambda st, inp: unit_kernels.movement(st, inp.fields),
                    inputs=tuple("field:" + f for f in unit_kernels.MOVEMENT_FIELDS))

#: Hook sets dealt to the marines (index i % 3): WALK / SPRINT non-neutral on
#: two thirds, TURN on one (so the travel's facing target meets a hooked turn).
HOOK_SETS = (
    None,
    {"WALK": ONE * 5 // 4, "SPRINT": ONE * 3 // 4, "TURN": ONE * 5 // 4},
    {"WALK": ONE * 2 // 3, "SPRINT": ONE * 3 // 2},
)


def _hook(sim, units):
    for i, u in enumerate(units):
        h = HOOK_SETS[i % 3]
        if h is not None:
            unit_traits.compile_unit(sim, u, h)


# ---------------------------------------------------------------------------
# The census: which branches of the law acted (from the slot's rows)
# ---------------------------------------------------------------------------
BRANCHES = (
    "walk_t0", "walk_tone", "walk_tmid",          # throttle 0 / ONE / between
    "walk_forward", "walk_side", "walk_back",     # against a held aim
    "target_written",                             # a walker with no aim
    "terrain",                                    # mobility < 1 under foot
    "effort_slowed",                              # effort scale < ONE
    "hooked_walk",                                # WALK / SPRINT != ONE
    "walk_slide",                                 # a diagonal walk sliding
    "walk_ban",                                   # react_ticks > 0, MOVE_DIR
    "cant_move_pv", "cant_move_still",            # no can_move, with / no pv
    "pushed_alone",                               # pv only, no MOVE_DIR
    "zombie_displaced",
    "multi_substep",                              # |pv| >= one tile a tick
    "wall_x", "wall_y",                           # pv zeroed into a wall
    "wall_x_alone", "wall_y_alone",               # ... with no walk of its own
    "dead_rows",
)


class Census:
    """Counts, per branch of the law, the rows the slot met in it (and, for
    the acting branches, changed)."""

    def __init__(self):
        self.n = {b: 0 for b in BRANCHES}
        self.lay = unit_store.layout()
        self._flags = None

    def _col(self, name):
        c = self.lay.column(name)
        return c.offset, c.width

    def _can_move(self, row, tab):
        so, sw = self._col("statuses")
        flags = unit_tables.read(tab, "status_flags")
        for i in range(sw // 4):
            k = int(row[so + 4 * i])
            if k < 0:
                break
            if flags[k] & unit_tables.FLAG_NO_MOVE:
                return False
        return True

    def __call__(self, reps, ls):
        k = [s.label for s in ls.slots].index(SLOT_3.label)
        g = lambda n: self.lay.column(n).offset  # noqa: E731
        c_valid, c_life, c_z = self.lay.valid, g("life_state"), g("is_zombie")
        c_x, c_y, c_pvx, c_pvy = g("x_q"), g("y_q"), g("pvx"), g("pvy")
        c_md, c_aim, c_react = g("live_move_dir"), g("live_aim"), g("react_ticks")
        c_eff, c_fac, c_tgt = g("effort"), g("facing_q"), g("facing_target_q")
        c_fp, c_tr = g("footprint"), g("traits")
        for e in range(ls.N):
            pre, post = ls._pre[k][e], ls._post[k][e]
            tab = ls.tables[e]
            gmap = ls.drivers[e].sim.gmap
            edge = unit_tables.read(tab, "impulse_effort_edge_q")
            for r in range(pre.shape[0]):
                a, b = pre[r], post[r]
                if not a[c_valid]:
                    continue
                if a[c_life] != 0:
                    self.n["dead_rows"] += 1
                    continue
                moved = (a[c_x], a[c_y]) != (b[c_x], b[c_y])
                pv = bool(a[c_pvx] or a[c_pvy])
                alone = bool(a[c_z]) or not a[c_md]
                if a[c_pvx] and not b[c_pvx]:
                    self.n["wall_x"] += 1
                    self.n["wall_x_alone"] += int(alone)
                if a[c_pvy] and not b[c_pvy]:
                    self.n["wall_y"] += 1
                    self.n["wall_y_alone"] += int(alone)
                if max(abs(int(a[c_pvx])), abs(int(a[c_pvy]))) >= unit_impulse.SUBSTEP_Q \
                        and moved:
                    self.n["multi_substep"] += 1
                if a[c_z]:
                    if pv and moved:
                        self.n["zombie_displaced"] += 1
                    continue
                has_md = bool(a[c_md])
                can_move = self._can_move(a, tab)
                if not has_md:
                    if pv and moved:
                        self.n["pushed_alone"] += 1
                    continue
                if not can_move:
                    self.n["cant_move_pv" if pv else "cant_move_still"] += 1
                    continue
                if a[c_react] > 0:
                    self.n["walk_ban"] += 1
                    continue
                dx, dy, t = int(a[c_md + 1]), int(a[c_md + 2]), int(a[c_md + 3])
                if not moved:
                    continue
                self.n["walk_t0" if t == 0 else
                       ("walk_tone" if t == ONE else "walk_tmid")] += 1
                if a[c_aim]:
                    cq = movement.facing_cos_q16(dx, dy, int(a[c_fac]))
                    if cq > ONE * 7 // 10:
                        self.n["walk_forward"] += 1
                    elif abs(cq) < ONE * 3 // 10:
                        self.n["walk_side"] += 1
                    elif cq < -ONE * 7 // 10:
                        self.n["walk_back"] += 1
                elif (dx or dy) and a[c_tgt] != b[c_tgt]:
                    self.n["target_written"] += 1
                fp = int(a[c_fp])
                ty, tx = int(a[c_y]) >> 16, int(a[c_x]) >> 16
                mob = gmap.footprint_mobility(ty, tx, fp)
                if sum(int(m) for m in mob) < 1000 * len(mob):
                    self.n["terrain"] += 1
                if int(a[c_eff]) > edge:
                    self.n["effort_slowed"] += 1
                tr = a[c_tr:c_tr + unit_traits.N_HOOKS]
                if tr[unit_traits.WALK] != ONE or tr[unit_traits.SPRINT] != ONE:
                    self.n["hooked_walk"] += 1
                if not pv and dx and dy and ((a[c_x] == b[c_x]) != (a[c_y] == b[c_y])):
                    self.n["walk_slide"] += 1

    def line(self):
        return ", ".join(f"{b} {self.n[b]}" for b in BRANCHES)

    def require(self, name, keys):
        missing = [b for b in keys if self.n[b] == 0]
        if missing:
            fail(f"{name}: branches never met (vacuous scene): {missing}")


# ---------------------------------------------------------------------------
# The movement box
# ---------------------------------------------------------------------------
_BOX_CACHE: dict = {}
BOX_H, BOX_W = 30, 48
#: Per layout variant: the tile size (a per-map speed / impulse table) and the
#: shift of the walls and the furniture (a per-env material plane).
VARIANTS = ((1.0, 0), (0.5, 2), (1.0 / 3.0, 4))


def _box_level(variant, n0, n1, n_z):
    key = (variant, n0, n1, n_z)
    if key in _BOX_CACHE:
        return _BOX_CACHE[key]
    import level_lib
    from PIL import Image
    from simulation.factions import ZOMBIE_TEAM
    from simulation.match import MatchSpec, MatchTeam, SpawnRoom
    from simulation.materials import MAT_AIR, MAT_FURNITURE, MAT_HULL, MAT_STEEL
    tile, s = VARIANTS[variant]
    idle = lambda n: MatchTeam(n=n, loadout="rifleman", env_profile="default",  # noqa: E731
                               controller="idle")
    rooms = [SpawnRoom(team=0, x=3, y=3, w=15, h=10, max_per_team=6),
             SpawnRoom(team=1, x=3, y=17, w=15, h=10, max_per_team=6),
             SpawnRoom(team=ZOMBIE_TEAM, x=34, y=19, w=12, h=8, max_per_team=2)]
    spec = MatchSpec(teams=(idle(n0), idle(n1)), seed=1, zombies=n_z,
                     time_limit_s=600.0, zombie_controller="idle")
    tmp = Path(tempfile.mkdtemp(prefix="p62m_box_"))
    lvl = tmp / "movebox"
    lvl.mkdir()
    level_lib.write_level_header(lvl, name="movebox", tile_size_m=tile)
    tm = np.full((BOX_H, BOX_W), MAT_HULL, dtype=np.int32)
    tm[1:BOX_H - 1, 1:BOX_W - 1] = MAT_AIR
    tm[2:15, 21 + s] = MAT_STEEL                     # a vertical wall
    tm[15, 21 + s:33 + s] = MAT_STEEL                # and its foot
    tm[3:11, 25 + s:32 + s] = MAT_FURNITURE          # a furniture patch
    tm[13:16, 2 + s:20] = MAT_FURNITURE              # a strip between the rooms
    level_lib.write_tilemap_csv(lvl, tm, csv_bak=False)
    Image.new("RGB", (BOX_W, BOX_H)).save(lvl / "diffuse.png")
    level_lib.write_spawn_rooms(lvl / "level.toml", rooms)
    level_lib.write_match(lvl, spec)
    level = level_loader.load("movebox", str(tmp))
    _BOX_CACHE[key] = level
    return level


def _dir(a_q):
    """A Q16 unit vector at Q16 angle ``a_q`` (math-style, world y down)."""
    return int(cos_q16(a_q)), -int(sin_q16(a_q))


class BoxDriver:
    """The movement box (variant ``variant``): team 0 walks a schedule of
    compass directions and throttles (aims held east / sweeping / none, one
    walker immobilized from time to time), team 1 is pushed by scripted
    impulses (half of it also walking, one immobilized), the idle horde is
    pushed, one marine is killed at tick 60; WALK / SPRINT hooks dealt."""

    def __init__(self, variant=0, n0=6, n1=4, n_z=2, seed=1, hooks=True):
        from simulation.match_session import MatchSession
        self.session = MatchSession(_box_level(variant, n0, n1, n_z), seed=seed,
                                    breach_physics=bp)
        self.sim = self.session.sim
        self.team0 = [u for u in self.sim.units if u.team == 0]
        self.team1 = [u for u in self.sim.units if u.team == 1]
        self.horde = [u for u in self.sim.units if u.is_zombie]
        if hooks:
            _hook(self.sim, self.team0 + self.team1)
        self.k = 0
        self.seed = seed
        # Push sizes (Q16 tiles a tick) from the map's own impulse table:
        # absorbed by the sway, overflowing it (the reaction delay: the walk
        # ban), near the fall limit, past it (a knockdown), and well over a
        # tile a tick (several substeps).
        it = self.sim.impulse_table
        self.mags = (it.v_inplace_q[1] * 4 // 5, it.v_inplace_q[0] * 5 // 2,
                     it.v_fall_q[2] * 9 // 10, it.v_fall_q[0] * 4, ONE * 3 // 2)

    def _push(self, u, dv_q, a_q):
        dx, dy = _dir(a_q)
        m = int(u.mass_g)
        unit_impulse.add(self.sim.impulse_table, u, (dx * dv_q >> 16) * m,
                         (dy * dv_q >> 16) * m)

    def tick(self):
        from simulation import status
        from simulation.damage import KINETIC, DamagePacket, apply_packet
        sim, k = self.sim, self.k
        quarter = TWO_PI_Q16 // 16
        for i, u in enumerate(self.team0):
            if not u.alive:
                continue
            d = (k // 45 * 5 + i * 3 + self.seed) % 16
            if i == 3:                       # to and fro across the furniture strip
                d = 12 if k % 160 < 80 else 4
            dx, dy = _dir(d * quarter)
            t = (0, ONE // 3, ONE // 2 + 77, ONE)[(k // 40 + i) % 4]
            sim.set_move_dir(u.id, dx, dy, t)
            if i % 3 == 0:
                sim.set_aim(u.id, ONE, 0)                       # held east
            elif i % 3 == 1:
                ax, ay = _dir((k * 3000 + i * 50000) % TWO_PI_Q16)
                sim.set_aim(u.id, ax, ay)                       # sweeping
            if i == 4 and k % 70 == 10:
                status.apply_status(u, status.IMMOBILIZED, 0.0, 20)
            if i in (2, 5) and k % 9 == 4:
                # absorbed by the sway: the effort rises, the walk slows
                self._push(u, self.mags[0], (k * 7919 + i) % TWO_PI_Q16)
        for j, u in enumerate(self.team1):
            if not u.alive:
                continue
            if (k + 5 * j) % 20 == 0:
                a = ((k // 20 + j) % 8) * (TWO_PI_Q16 // 8)
                self._push(u, self.mags[(k // 20 + j) % len(self.mags)], a)
            if j % 2 == 0:
                d = (k // 25 + j) % 16
                dx, dy = _dir(d * quarter)
                sim.set_move_dir(u.id, dx, dy, ONE // 4)
            if j == 1 and k % 45 == 3:
                status.apply_status(u, status.IMMOBILIZED, 0.0, 15)
        for z, u in enumerate(self.horde):
            if u.alive and (k + 11 * z) % 30 == 0:
                a = ((k // 30 + 2 * z) % 4) * (TWO_PI_Q16 // 4)
                self._push(u, (self.mags[3], ONE * 6 // 5)[(k // 30) % 2], a)
        if k == 60 and len(self.team1) > 2 and self.team1[-1].alive:
            apply_packet(self.team1[-1], DamagePacket(amount=10000, dtype=KINETIC),
                         None, source="bullet")
        self.session.tick()
        self.k += 1

    def is_terminal(self):
        return self.session.is_terminal()


# ---------------------------------------------------------------------------
# The lockstep runner
# ---------------------------------------------------------------------------
def run_lockstep(name, drivers, ticks, twins=None, min_share=0.0, census=None):
    """Shadow slot 3 on ``drivers`` for up to ``ticks`` ticks: every tick
    equal (every column) and the coverage == the host's changed rows;
    ``twins`` (harness-free copies) stay equal; returns the share of
    env-ticks the kernel changed a row in."""
    t0 = time.perf_counter()
    cov = total = n = 0
    ok = True
    with ShadowLockstep(drivers, [SLOT_3]) as ls:
        for _ in range(ticks):
            if any(getattr(d, "is_terminal", lambda: False)() for d in drivers):
                break
            rep = ls.tick()[SLOT_3.label]
            n += 1
            total += len(drivers)
            if not rep.equal:
                fail(f"{name}: {ls.divergences[-1]}")
                break
            if not np.array_equal(rep.coverage, rep.host_changed):
                fail(f"{name} tick {rep.tick}: coverage {rep.coverage.tolist()} != "
                     f"host-changed {rep.host_changed.tolist()}")
                break
            cov += int((rep.coverage > 0).sum())
            if census is not None:
                census({SLOT_3.label: rep}, ls)
            if twins is not None:
                for d, tw in zip(drivers, twins):
                    tw.tick()
                    if not _same(d.sim, tw.sim):
                        fail(f"{name}: the hooked match diverged from the "
                             f"harness-free one at tick {d.sim.tick}")
                        ok = False
                if not ok:
                    break
    share = cov / max(1, total)
    counts = [len(d.sim.units) for d in drivers]
    print(f"{name}: N={len(drivers)} units per env {counts} (U {ls.U}), {n} ticks"
          f"{' (twin equal every tick)' if twins is not None and ok else ''}"
          f"{', env i == env i alone' if len(drivers) > 1 and ok else ''}; "
          f"coverage > 0 on {100 * share:.1f} % of env-ticks "
          f"({time.perf_counter() - t0:.0f} s)")
    if census is not None:
        print(f"{name}: census: {census.line()}")
    if share < min_share:
        fail(f"{name}: coverage > 0 on {100 * share:.1f} % of env-ticks, below "
             f"the stated {100 * min_share:.1f} % (vacuous scene)")
    return share


# ---------------------------------------------------------------------------
# PART LIB
# ---------------------------------------------------------------------------
def _probe(op, rows, material=None, mobility=None):
    return unit_kernels.movement_probe(bp, op, np.array(rows, dtype=np.int64).reshape(
        -1, 6), material, mobility)


def part_lib():
    from simulation.movement import FootprintSamples
    rng = np.random.default_rng(62)
    bad_cnt = {}

    def check(name, op, rows, ref):
        out = _probe(op, rows)
        n_bad = 0
        for row, o in zip(rows, out):
            try:
                want = ref(*row)
                wbad = False
            except (ValueError, OverflowError):
                want, wbad = None, True
            if wbad or o[3]:
                n_bad += int(bool(o[3]))
                if not (wbad and o[3]) and not (o[3] and not wbad and
                                                _out_of_range(name, row)):
                    fail(f"lib {name}{tuple(row)}: python "
                         f"{'raises' if wbad else want}, device "
                         f"{'bad' if o[3] else int(o[0])}")
                    return
                continue
            if int(o[0]) != want:
                fail(f"lib {name}{tuple(row)}: python {want}, device {int(o[0])}")
                return
        bad_cnt[name] = (len(rows), n_bad)

    # default_speed: (n, total, class, hook, scale)
    def ref_ds(n, total, cls, hook, scale, _):
        mob = [int(total)] + [0] * (int(n) - 1) if n > 0 else []
        return movement.default_speed(FootprintSamples(
            mobility=mob, effort_scale_q16=int(scale), speed_hook_q16=int(hook)), int(cls))
    rows = []
    for _ in range(4000):
        n = int(rng.integers(0, 65))
        total = int(rng.choice([rng.integers(-5, 6), rng.integers(1, 1000 * max(1, n) + 1),
                                rng.integers(1, 64 << 20)]))
        cls = int(rng.choice([rng.integers(ONE + 1, 1 << 26), 18 * ONE, 374154]))
        hook = int(rng.choice([ONE, ONE // 8, 8 * ONE, rng.integers(ONE // 8, 8 * ONE + 1)]))
        scale = int(rng.choice([ONE, ONE // 2, rng.integers(ONE // 2, ONE + 1)]))
        rows.append((n, total, cls, hook, scale, 0))
    rows += [(9, 9000, 374154, ONE, ONE, 0), (9, 3600, 374154, ONE, ONE // 2, 0),
             (9, 9000, 116390, ONE * 2 // 3, ONE, 0), (0, 0, 374154, ONE, ONE, 0),
             (9, 0, 374154, ONE, ONE, 0), (9, 9000, 374154, 0, ONE, 0)]
    check("default_speed", 0, rows, ref_ds)

    # move_speed_q16: (cw, cf, side, back, t, cos)
    def ref_ms(cw, cf, side, back, t, c):
        return movement.move_speed_q16(cw, cf, side, back, t, c)
    rows = []
    for _ in range(4000):
        cw = int(rng.integers(1, 1 << 29))
        cf = int(rng.choice([rng.integers(1, cw + 1), rng.integers(1, 1 << 29)]))
        side = int(rng.integers(1, ONE + 1))
        back = int(rng.integers(1, side + 1))
        t = int(rng.choice([0, ONE, rng.integers(0, ONE + 1), -1, ONE + 1]))
        c = int(rng.choice([ONE, -ONE, 0, rng.integers(-ONE, ONE + 1)]))
        rows.append((cw, cf, side, back, t, c))
    rows += [(374154, 116390, 52429, 39322, 0, ONE), (374154, 116390, 52429, 39322, ONE, ONE),
             (374154, 116390, 52429, 39322, ONE // 2, -ONE), (0, 5, ONE, ONE, 0, 0)]
    check("move_speed_q16", 1, rows, ref_ms)

    # effort_scale_q16: (e, e0)
    class _U:
        pass

    class _Tab:
        def __init__(self, e0):
            self.effort_edge_q = e0

    def ref_es(e, e0, *_):
        u = _U()
        u.effort = int(e)
        return unit_impulse.effort_scale_q16(_Tab(int(e0)), u)
    rows = [(int(rng.integers(-ONE, 3 * ONE)), int(rng.integers(0, ONE)), 0, 0, 0, 0)
            for _ in range(3000)] + [(0, 0, 0, 0, 0, 0), (ONE, 0, 0, 0, 0, 0),
                                    (ONE - 1, ONE - 2, 0, 0, 0, 0), (2 * ONE, ONE, 0, 0, 0, 0)]
    check("effort_scale_q16", 2, rows, ref_es)

    # facing_cos_q16: (dx, dy, facing)
    def ref_fc(dx, dy, f, *_):
        return movement.facing_cos_q16(dx, dy, f)
    rows = [(int(rng.integers(-ONE, ONE + 1)), int(rng.integers(-ONE, ONE + 1)),
             int(rng.integers(-4 * 205887, 4 * 205887)), 0, 0, 0) for _ in range(3000)]
    rows += [(ONE, 0, 0, 0, 0, 0), (-ONE, 0, 0, 0, 0, 0), (0, -ONE, 102944, 0, 0, 0),
             (0, 0, 0, 0, 0, 0)]
    check("facing_cos_q16", 3, rows, ref_fc)

    # passable_block / footprint_mobility over a synthetic map
    from simulation.gamemap import GameMap
    h, w = 13, 17
    mob = np.array([1000, 0, 0, 1000, 0, 0, 400, 0, 400, 1000, 0, 2500], dtype=np.int64)
    mat = rng.choice([0, 0, 0, 0, 3, 6, 8, 9, 11, 1, 4], size=(h, w)).astype(np.int8)

    class _Mats:
        mobility = mob

    gm = GameMap.__new__(GameMap)
    gm.material, gm.materials, gm._h, gm._w = mat, _Mats(), h, w
    rows = [(int(rng.integers(-4, h + 2)), int(rng.integers(-4, w + 2)),
             int(rng.integers(1, 6)), 0, 0, 0) for _ in range(3000)]
    out = _probe(4, rows, mat, mob)
    for row, o in zip(rows, out):
        fy, fx, fp = row[:3]
        y1, x1 = min(h, fy + fp), min(w, fx + fp)
        if y1 < 0 or x1 < 0:                       # numpy would count from the end
            if not o[3]:
                fail(f"lib footprint{row[:3]}: a negative stop not refused")
            continue
        want_p = int(GameMap.is_passable_block(gm, fy, fx, fp))
        fm = GameMap.footprint_mobility(gm, fy, fx, fp)
        if o[3] or (int(o[0]), int(o[1]), int(o[2])) != (want_p, len(fm), sum(fm)):
            fail(f"lib footprint{row[:3]}: python ({want_p}, {len(fm)}, {sum(fm)}), "
                 f"device {tuple(int(v) for v in o)}")
            break
    bad_cnt["passable/footprint"] = (len(rows), int(out[:, 3].sum()))
    print("lib: " + ", ".join(f"{k} {v[0]} rows ({v[1]} refused)"
                              for k, v in bad_cnt.items()))


def _out_of_range(name, row):
    """Rows the device refuses although Python computes them: operands past
    the proved ranges (cuda_unit_lib.cuh's P6.2m block)."""
    if name == "default_speed":
        n, total, cls, hook, scale, _ = row
        return (n > 64 or cls > (1 << 29) or hook > (1 << 20)
                or _ds_cost_over(n, total, cls, hook, scale))
    if name == "move_speed_q16":
        cw, cf = row[0], row[1]
        return cw > (1 << 29) or cf > (1 << 29)
    return False


def _ds_cost_over(n, total, cls, hook, scale):
    from simulation.movement import half_up
    if total <= 0:
        return False
    c = (half_up(cls * n * 1000, total) if hook == ONE
         else half_up(cls * n * 1000 * ONE, total * hook))
    if c > (1 << 29):
        return True
    if scale != ONE and scale > 0:
        c = half_up(c * ONE, scale)
    return c > (1 << 29)


# ---------------------------------------------------------------------------
# PART SHIP / BOX / DOORWAY / BATCH
# ---------------------------------------------------------------------------
def _ship(hooks=True):
    d = ShipDriver(seed=1, orders=False, hooks=False)
    if hooks:
        _hook(d.sim, [u for u in d.sim.units if not u.is_zombie])
    return d


def part_ship():
    cen = Census()
    run_lockstep("ship (bots, WALK / SPRINT / TURN hooks)", [_ship()], SHIP_TICKS,
                 twins=[_ship()], min_share=0.5, census=cen)
    cen.require("ship", ("walk_t0", "walk_forward", "walk_back", "hooked_walk"))


BOX_REQUIRED = tuple(b for b in BRANCHES)


def part_box():
    cen = Census()
    run_lockstep("box", [BoxDriver()], BOX_TICKS, twins=[BoxDriver()],
                 min_share=0.5, census=cen)
    cen.require("box", BOX_REQUIRED)


def part_batch():
    for n in BATCH_NS:
        drivers = [BoxDriver(variant=e % 3, n0=4 + e % 3, n1=3 + (e // 3) % 2,
                             seed=1 + e) for e in range(n)]
        cen = Census()
        run_lockstep(f"box batch N={n}", drivers, BATCH_TICKS, min_share=0.5, census=cen)
        cen.require(f"box batch N={n}", ("walk_t0", "walk_tone", "walk_tmid", "terrain",
                                         "wall_x", "wall_y", "zombie_displaced"))
        if len({len(d.sim.units) for d in drivers}) < 2:
            fail(f"box batch N={n}: every env has the same unit count (no padding)")
        tabs = {tuple(unit_tables.values(d.sim)) for d in drivers}
        if len(tabs) < 2:
            fail(f"box batch N={n}: every env has the same tables")


DOOR_REQUIRED = ("pushed_alone",)


def part_doorway():
    cen = Census()
    run_lockstep("doorway N=1", [DoorwayDriver(n_squad=3)], DOOR_TICKS,
                 twins=[DoorwayDriver(n_squad=3)], min_share=0.3, census=cen)
    cen.require("doorway", DOOR_REQUIRED)
    cen2 = Census()
    run_lockstep("doorway N=2", [DoorwayDriver(n_squad=2), DoorwayDriver(n_squad=4)],
                 DOOR_TICKS, min_share=0.3, census=cen2)


# ---------------------------------------------------------------------------
# PART CHAINED
# ---------------------------------------------------------------------------
def _movement_host_side(sim, pre, post):
    """What slot 3 does that is not unit rows: the momentum books' WALL
    entries (``unit_impulse.move`` books the pv it zeroes; REFERENCE_ONLY)
    and the legacy WEGO path step's ``path_tick_offset`` bump (a held
    non-zombie with no MOVE_DIR; on the continuous path ``move_path`` is
    empty, so nothing else)."""
    from simulation.status import composed_flags
    lay = unit_store.layout()
    cx, cy = lay.column("pvx").offset, lay.column("pvy").offset
    for r, u in enumerate(sim.units):
        if pre[r, cx] and not post[r, cx]:
            u.impulse_books[unit_impulse.WALL_X] += int(pre[r, cx])
        if pre[r, cy] and not post[r, cy]:
            u.impulse_books[unit_impulse.WALL_Y] += int(pre[r, cy])
        if not u.alive or u.is_zombie or u.live_move_dir is not None:
            continue
        if u.move_path:
            raise RuntimeError("chained slot 3: a WEGO path on the continuous path")
        if not composed_flags(u).can_move or not unit_impulse.walk_allowed(u):
            u.path_tick_offset += 1


def _chained(name, make, ticks):
    twin = make()
    ref = make()
    t0 = time.perf_counter()
    n = 0
    with ChainedSlots(twin, [SLOT_3], host_side=_movement_host_side) as ch:
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
            books = [list(u.impulse_books) for u in twin.sim.units]
            if books != [list(u.impulse_books) for u in ref.sim.units]:
                fail(f"{name}: the momentum books differ at tick {twin.sim.tick}")
                break
    acted = sum(1 for covs in ch.coverage if int(covs[0][0]) > 0)
    print(f"{name}: {n} ticks, the chained twin equal to the reference every tick "
          f"({ch.runs} chain runs; slot 3 changed rows on {acted} ticks; "
          f"{time.perf_counter() - t0:.0f} s)")
    if ch.runs != n:
        fail(f"{name}: the chain ran {ch.runs} times in {n} ticks")
    if acted == 0:
        fail(f"{name}: slot 3 never acted (vacuous)")


def part_chained():
    if 0 in CHAINS:
        _chained("chained 3 (ship, bots, hooks)", _ship, SHIP_TICKS)
    if 1 in CHAINS:
        _chained("chained 3 (box: pushes, walls)", BoxDriver, BOX_TICKS)


# ---------------------------------------------------------------------------
# main / bench
# ---------------------------------------------------------------------------
PARTS = ("lib", "ship", "box", "batch", "doorway", "chained")


def main(parts=None) -> int:
    if not getattr(bp, "HAS_CUDA", False):
        print("UNIT_MOVEMENT_RESULT: FAIL (not the CUDA build)")
        return 1
    parts = parts or os.environ.get("UNIT_MOVEMENT_PARTS", ",".join(PARTS)).split(",")
    t0 = time.perf_counter()
    fns = {"lib": part_lib, "ship": part_ship, "box": part_box, "batch": part_batch,
           "doorway": part_doorway, "chained": part_chained}
    for p in parts:
        try:
            if p not in fns:
                fail(f"no part {p!r}")
                continue
            fns[p]()
        except Exception as ex:          # a part's crash fails it, the rest run
            import traceback
            traceback.print_exc()
            fail(f"part {p} raised {type(ex).__name__}: {ex}")
    print(f"total {time.perf_counter() - t0:.0f} s")
    ok = not FAILS
    print("UNIT_MOVEMENT_RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def bench(reps: int = 200) -> int:
    """The TIMING table: upload + kernel + download and the kernel call alone
    (launch + kernel + the counts' D2H and sync), medians over ``reps``, at
    N = 1, 8, 64 (env rows replicated from ship_demo's bots match after 300
    ticks)."""
    import cupy as cp
    from cuda_unit_lockstep import capture, gather
    drv = _ship()
    for _ in range(300):
        drv.tick()
    sim = drv.sim
    rows1 = unit_store.pack(sim.units)
    U = rows1.shape[0]
    names = tuple("field:" + f for f in unit_kernels.MOVEMENT_FIELDS)
    for n in (1, 8, 64):
        rows = np.ascontiguousarray(np.broadcast_to(rows1, (n,) + rows1.shape))
        st = unit_store.DeviceStore(n, U, tables=np.tile(unit_tables.values(sim), (n, 1)))
        inp = gather(names, [capture(names, sim)] * n)
        full, alone = [], []
        for _ in range(reps):
            t0 = time.perf_counter()
            st.upload(rows)
            t1 = time.perf_counter()
            unit_kernels.movement(st, inp.fields)
            t2 = time.perf_counter()
            st.download()
            t3 = time.perf_counter()
            full.append(t3 - t0)
            alone.append(t2 - t1)
        print(f"TIMING N={n:3d} U={U} 3 movement           upload+kernel+download "
              f"{1e3 * float(np.median(full)):.3f} ms  kernel call "
              f"{1e3 * float(np.median(alone)):.3f} ms")
        del st
        cp.get_default_memory_pool().free_all_blocks()
    return 0


if __name__ == "__main__":
    sys.exit(main())
