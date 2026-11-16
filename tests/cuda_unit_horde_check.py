"""#29 P7d -- THE HORDE GATE: slot 5 (``ai_zombie.update_zombies_tick`` through
``Simulation._run_zombie_slot``) as the applier's HORDE stage
(``cpp/src/cuda_unit_horde.cuh``: the drain of the noise list, the request
latch, the field cache with section 4.9's refresh rule, passes 1 and 2 wide,
pass 3 serial with the bite and the bash applied at once) and THE FLOOD (the
stage's block-wide BFS) against the C++ reference, each at tol 0 (design
4.4c, the cut table's P7d row).

Runs inside the GPU subprocess (tests/cuda_harness.run_cuda_script; wrapper
tests/test_cuda_unit_horde.py). The stage is ``unit_kernels.horde``; the cache
``simulation.horde_cache.DeviceHordeCache`` (the harness lists ``horde`` --
staged from the host every tick -- and ``horde_live`` -- the device's own
cache across ticks, never restaged); the flood's TEST-ONLY probe
``unit_kernels.horde_flood_probe``.

PART TABLES -- every P7d device table row equals an independent read.
PART FLOOD -- the device flood equals ``horde.flood`` (the C++ reference,
``PhysicsEngine.horde_field``) cell for cell: every shipped level with every
mask the cache builds (walkable, plus the bashable doors, minus a feared band)
and footprints 1 to 4, single and many goals, goals outside the grid, on a
wall, repeated; random maps of every density with unreachable pockets and the
map edge; a map changed between two floods (a wall destroyed, a door closed);
in a batch of differing maps.
PART SCENE -- the SHADOW LOCKSTEP of slot 5 on the HORDE SCENES (ids not row
indices): the unit rows, the ring, the noise list after the drain, the cache
(header, keys, every distance plane and the masks by digest), every plane, the
books and the write set equal every tick; a CENSUS of the Python law's
branches in passes 1, 2, 2b and 3, each required, the unreachable named;
coverage == the rows the host changed; the hooked match equal to an unhooked
twin.
PART CACHE -- the device's OWN cache (``horde_live``: staged once, never
again) carried across every tick of the scenes and compared with the host's
every tick: each refresh cause (the tick, a topology change), each reuse, the
early returns (no prey; an empty request set) leaving it standing.
PART LEVELS -- ship_demo's bots match (4 zombies, the zombie controller's
latch) and playground's (door entities, 5 zombies) shadowed.
PART BATCH -- N = 2 and 8 of scenes that differ in layout, zombie count,
seed and fields (env 0 not the largest, a zombie-free env beside many).
PART PADDING -- garbage rows past the store's count and in the cache's unused
slots neither read nor written.
PART OVERFLOW -- the cache's capacity, the noise list's and the ring's refused
by name.
PART CHAINED -- slot 5 replaced by the device's in a running match (the
scenes, ship_demo's bots match 500 ticks): unit hash, field digest, planes,
books, noise list and cache equal an untouched reference every tick, the host
zombie law run 0 times in the twin; then chained WITH 4a and 4c around it.

Prints per-part lines and ``UNIT_HORDE_RESULT: PASS`` / ``FAIL``; exits 0 /
1. ``bench()`` prints the TIMING table.
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
from simulation import (applier, horde, horde_cache, unit_kernels,  # noqa: E402
                        unit_schema, unit_store, unit_tables)
from simulation.materials import MAT_AIR, MAT_HULL  # noqa: E402

FAILS: list = []
ONE = 65536
SCENE_TICKS = int(os.environ.get("UNIT_HORDE_SCENE_TICKS", "420"))
BATCH_TICKS = int(os.environ.get("UNIT_HORDE_BATCH_TICKS", "160"))
SHIP_TICKS = int(os.environ.get("UNIT_HORDE_SHIP_TICKS", "500"))
CHAIN_TICKS = int(os.environ.get("UNIT_HORDE_CHAIN_TICKS", "420"))
BATCH_NS = [int(x) for x in os.environ.get("UNIT_HORDE_BATCH_NS", "2,8").split(",")]


def fail(msg):
    FAILS.append(msg)
    print("FAIL:", msg)


# ---------------------------------------------------------------------------
# The slot
# ---------------------------------------------------------------------------
PLANE_OUTPUTS = tuple("field:" + n for n in applier.PLANE_NAMES)
HORDE_BOOK_NAMES = ("n_destruction_seed", "ge_retire", "ge_mint", "ge_destroy_seed",
                    "solid_destroy_wall", "tiles_changed")
HORDE_BOOKS = tuple("book:" + b for b in HORDE_BOOK_NAMES)
HOST_ONLY = (
    ("render_dirty", "the wall seam's render mark (the renderer drains it; no device form)"),
    ("_sky_mask_dirty", "the sky exchange's lazy-rebuild flag (no device form)"),
    ("_gas_t_amb_raw_cache", "a lazily filled cache of a config constant (the device reads "
     "the table row topo_t_amb_q)"),
    ("smoke", "GameMap.smoke IS a view of gas[SMOKE]: compared through field:gas"),
    ("noise_log", "THE noise list: compared through list:noise"),
)
_APPLIERS: dict = {}


def _applier_for(store):
    n = store.shape[0]
    if n not in _APPLIERS:
        _APPLIERS[n] = applier.DeviceApplier(n, 4, bp=bp)
    return _APPLIERS[n]


def _run_5(cache_list):
    def run(st, inp, ring, lists):
        return unit_kernels.horde(st, _applier_for(st), inp.fields, ring, lists["noise"],
                                  lists["doors"], lists["door_tiles"], lists[cache_list],
                                  inp.tables["horde_latch"])
    return run


def _slot(cache_list="horde"):
    ins = ("horde_latch", "list:noise", "list:doors", "list:door_tiles")
    if cache_list == "horde":
        ins = ins + ("list:horde",)
    return DeviceSlot(label="5 zombie AI", run=_run_5(cache_list), events=True, inputs=ins,
                      outputs=PLANE_OUTPUTS + HORDE_BOOKS + ("list:noise",
                                                             f"list:{cache_list}"),
                      host_only=HOST_ONLY)


SLOT_5 = _slot("horde")
SLOT_5_LIVE = _slot("horde_live")


# ---------------------------------------------------------------------------
# The reference values of the P7d table rows (cuda_unit_kernels_check's
# every-row gate reads this too)
# ---------------------------------------------------------------------------
def table_reference(sim) -> dict:
    """An INDEPENDENT read of every P7d table row: [zombie] through the
    settings door re-run on the sim's tile, the config's own values, the
    speed table's door re-run."""
    from config import CFG
    from simulation import wall_fixed
    from simulation.gas_fixed import quantize_q16_scalar
    from simulation.speeds import speed_table
    hs = horde.horde_settings(CFG, float(sim.gmap.tile_size_m))
    sp = speed_table(CFG, float(sim.gmap.tile_size_m))
    z = CFG.zombie
    return {
        "horde_trigger_r2": int(z.trigger_radius) ** 2,
        "horde_propagation_r2": int(z.propagation_radius) ** 2,
        "horde_refresh_ticks": int(z.field_refresh_ticks),
        "horde_wander": int(bool(z.wander)),
        "horde_wander_period_ticks": hs.wander_period_ticks,
        "horde_wander_leash_r2": hs.wander_leash_r2,
        "horde_hearing": int(bool(z.hearing)),
        "horde_noise_timeout_ticks": hs.noise_timeout_ticks,
        "horde_noise_fields_max": int(z.noise_fields_max),
        "horde_noise_arrive_tiles": hs.noise_arrive_tiles,
        "horde_fear_fire": int(bool(z.fear_fire)),
        "horde_fear_temp_q16": quantize_q16_scalar(float(z.fear_temp)),
        "horde_bash_doors": int(bool(z.bash_doors)),
        "horde_bash_damage_q": wall_fixed.quantize_scalar(float(z.door_bash_damage)),
        "horde_bash_loudness_tiles": hs.bash_loudness_tiles,
        "horde_sprint": int(bool(z.sprint)),
        "horde_sprint_r2": hs.sprint_r2,
        "zombie_attack_cooldown_ticks": int(z.attack_cooldown_ticks),
        "zombie_bite_class_q": int(round(float(z.melee_damage) * ONE)),   # traits-class: the reference read of the class value the table uploads
        "zombie_shamble_ticks": int(sp.zombie_shamble_ticks),   # traits-class: as above
        "zombie_sprint_ticks": int(sp.zombie_sprint_ticks),     # traits-class: as above
    }


def part_tables():
    sims = [HordeScene(0).sim, HordeScene(1).sim]
    tab = unit_tables.values_batch(sims)
    n = 0
    for e, sim in enumerate(sims):
        for k, v in table_reference(sim).items():
            dv = unit_tables.read(tab[e], k)
            n += 1
            if (tuple(dv) if isinstance(dv, (tuple, list)) else (dv,)) != (v,):
                fail(f"tables env {e} row {k}: device {dv} != reference {v}")
    print(f"tables: {n} horde values over {len(sims)} envs equal the reference "
          f"(tile sizes {[s.gmap.tile_size_m for s in sims]})")


# ---------------------------------------------------------------------------
# PART FLOOD
# ---------------------------------------------------------------------------
def _ref_flood(gmap, ok, fp, seeds):
    if getattr(gmap, "_physics_engine", None) is None:
        raise RuntimeError("flood: the reference map has no engine bound (the C++ flood)")
    return horde.flood(gmap, ok, fp, seeds)


def _flood_cases(sim, rng):
    """(name, tile_ok, fp, seeds) over sim's map: the cache's masks and goal
    sets, footprints 1..4, the corner cases."""
    g = sim.gmap
    h, w = g.material.shape
    walk = horde.walkable_tiles(g)
    bash = horde.bashable_door_tiles(g, sim._doors)
    band = np.zeros_like(walk)
    band[:, w // 2:w // 2 + 2] = True               # a feared band down the middle
    masks = {"walkable": walk, "walkable+doors": walk | bash,
             "walkable+doors-fear": (walk | bash) & ~band}
    free = np.argwhere(walk)
    out = []
    for mname, ok in masks.items():
        for fp in (1, 2, 3, 4):
            one = [tuple(free[rng.integers(len(free))])] if len(free) else [(0, 0)]
            many = [tuple(free[i]) for i in rng.integers(0, max(1, len(free)), 9)] \
                if len(free) else [(0, 0)]
            odd = [(-3, 2), (h + 1, w // 2), (h - 1, w - 1), (0, 0)] + many[:2] + many[:2]
            for sname, sd in (("one goal", one), ("many goals", many),
                              ("odd goals (outside, a wall, the edge, repeated)", odd)):
                out.append((f"{mname} fp {fp} {sname}", ok, fp, sd))
    return out


def _random_map(rng, h, w, density):
    tm = np.full((h, w), MAT_AIR, dtype=np.int32)
    tm[rng.random((h, w)) < density] = MAT_HULL
    tm[h // 3, :] = MAT_HULL                        # a wall across: pockets
    tm[h // 3, rng.integers(0, w)] = MAT_AIR
    return tm


def part_flood():
    from simulation import Simulation
    from simulation.ruleset import ContinuousRealtime
    rng = np.random.default_rng(7)
    n = 0
    t0 = time.perf_counter()
    # the shipped levels, each map with its own masks
    for name in ("ship_demo", "playground", "door_test", "airlock_demo", "test_level",
                 "planetside_demo", "wind_studio"):
        try:
            level = level_loader.load(name, str(_ROOT / "levels"))
        except Exception as ex:                      # noqa: BLE001
            fail(f"flood: level {name} did not load: {ex}")
            continue
        sim = Simulation(level, seed=1, breach_physics=bp, enable_recorder=False,
                         ruleset=ContinuousRealtime())
        n += _flood_compare(name, sim.gmap, _flood_cases(sim, rng))
    # random maps (every density; pockets; the edge), and a map changed between
    # two floods (a wall destroyed, a door closed)
    import cupy as cp
    for k, density in enumerate((0.0, 0.1, 0.3, 0.5, 0.7)):
        h, w = 23 + 3 * k, 31 + 5 * k
        sim = _bare_sim(_random_map(rng, h, w, density))
        g = sim.gmap
        ok = horde.walkable_tiles(g)
        free = np.argwhere(ok)
        seeds = [tuple(free[i]) for i in rng.integers(0, max(1, len(free)), 5)] if len(free) \
            else [(0, 0)]
        n += _flood_compare(f"random {density}", g,
                            [(f"random {density} fp {fp}", ok, fp, seeds) for fp in (1, 2, 3)])
    sim = _bare_sim(_random_map(rng, 30, 40, 0.25))
    g = sim.gmap
    seeds = [(1, 1), (28, 38)]
    before = _flood_compare("changed (before)", g,
                            [("before", horde.walkable_tiles(g), 2, seeds)])
    walls = np.argwhere(g.material == MAT_HULL)[:40]
    for (y, x) in walls:
        g.destroy_wall(int(y), int(x))
    after = _flood_compare("changed (walls destroyed)", g,
                           [("after", horde.walkable_tiles(g), 2, seeds)])
    tm2, ents2, _ = _scene_map(0)                    # a door's tile closes: a mask changes
    sim2 = _bare_sim(tm2, TILE, ents2, physics=True)
    sim2.gmap.material[sim2._doors[0].span[0]] = MAT_HULL
    sim2.gmap._update_caches()
    n += before + after + _flood_compare("changed (a door's tile closed)", sim2.gmap, [
        ("door closed", horde.walkable_tiles(sim2.gmap), 3, [(5, 5), (5, 40)])])
    # a batch of differing maps in one launch (env i == the reference)
    maps = [_bare_sim(_random_map(rng, 26, 33, d)).gmap for d in (0.05, 0.3, 0.6)]
    oks = np.stack([horde.walkable_tiles(m) for m in maps]).astype(np.uint8)
    fps = [1, 3, 2]
    sds = [[(2, 2)], [(20, 30), (3, 3), (40, 40)], []]
    dev = unit_kernels.horde_flood_probe(bp, cp.asarray(oks), fps, sds)
    for e, m in enumerate(maps):
        want = _ref_flood(m, oks[e].astype(bool), fps[e], sds[e])
        if not np.array_equal(dev[e], want):
            fail(f"flood: batch env {e} differs from the reference at "
                 f"{tuple(np.argwhere(dev[e] != want)[0])}")
        n += 1
    print(f"flood: {n} floods equal the C++ reference cell for cell (7 shipped levels x 3 "
          f"masks x fp 1..4 x 3 goal sets, 5 random densities, a changed map, a batch of 3) "
          f"({time.perf_counter() - t0:.0f} s)")


def _flood_compare(name, g, cases) -> int:
    import cupy as cp
    n = 0
    reached = 0
    for cname, ok, fp, seeds in cases:
        want = _ref_flood(g, ok, fp, seeds)
        got = unit_kernels.horde_flood_probe(bp, cp.asarray(ok.astype(np.uint8)[None]), [fp],
                                             [seeds])[0]
        n += 1
        reached += int((want != horde.UNREACHED).sum())
        if not np.array_equal(got, want):
            y, x = (int(v) for v in np.argwhere(got != want)[0])
            fail(f"flood {name} / {cname}: device {got[y, x]} != reference {want[y, x]} at "
                 f"({y}, {x})")
            return n
    if reached == 0 and cases:
        fail(f"flood {name}: no case reached a cell (vacuous)")
    return n


def _bare_sim(tm, tile=1.0, entities=(), physics=True):
    from level_loader import LevelData
    from simulation import Simulation
    from simulation.ruleset import ContinuousRealtime
    lvl = LevelData(name="p7d", version="3", path=Path("."), tilemap=tm,
                    tile_size_m=tile, diffuse_path=Path("."), entities=list(entities))
    sim = Simulation(lvl, seed=1, breach_physics=bp if physics else None,
                     enable_recorder=False, ruleset=ContinuousRealtime())
    sim.set_paused(False)
    return sim


# ---------------------------------------------------------------------------
# The horde scenes
# ---------------------------------------------------------------------------
def _door_entity(eid, ordinal, x, y, orient, length, state="closed"):
    from level_loader import EntityInstance
    from simulation.entities import door as door_mod
    fields = {f.name: f.default for f in door_mod.door.FIELDS}
    fields.update(x=x, y=y, orientation=orient, length_m=float(length), initial_state=state)
    return EntityInstance(id=eid, class_name="door", ordinal=ordinal, fields=fields)


TILE = 0.5
H, W = 54, 99


def _scene_map(variant):
    """1/2 m tiles, H x W. Room A (x 1..44) | the wall x 45 with the closed
    door D0 (3 tiles at y d0y) | room B (x 46..97, y 1..27). Below B, behind
    an S-shaped corridor (the wall y 28 open at x 47..49, the corridor y
    29..31, the wall y 32 open at x 94..96: no line of sight from B), room C
    (x 46..97, y 33..52) with a pillar (x 55, y 40..52) that hides its west
    end. A sealed closet in A's south-west corner (no way out). A hull door D1
    in the east hull to space (never bashed). Variants move D0 and add a wall
    across B with an OPEN door D2 (another layout, another door count)."""
    tm = np.full((H, W), MAT_HULL, dtype=np.int32)
    tm[1:H - 1, 1:W - 1] = MAT_AIR
    tm[:, W - 1] = level_loader.SPACE_CODE
    d0y = 10 + 3 * (variant % 3)
    tm[1:H - 1, 45] = MAT_HULL
    tm[d0y:d0y + 3, 45] = MAT_AIR
    tm[28, 46:98] = MAT_HULL                         # B | the corridor
    tm[28, 47:50] = MAT_AIR
    tm[32, 46:98] = MAT_HULL                         # the corridor | C
    tm[32, 94:97] = MAT_AIR
    tm[40:53, 55] = MAT_HULL                         # C's pillar
    tm[44:53, 1:9] = MAT_HULL                        # the closet's walls
    tm[45:52, 2:8] = MAT_AIR                         # its inside (sealed)
    tm[10:13, W - 2] = MAT_AIR                       # the hull door's span
    ents = [_door_entity("d0", 0, 45, d0y, "v", 1.5),
            _door_entity("d1", 1, W - 2, 10, "v", 1.5)]
    if variant % 2 == 1:                             # another layout: a wall across B
        tm[1:28, 80] = MAT_HULL
        tm[20:23, 80] = MAT_AIR
        ents.append(_door_entity("d2", 2, 80, 20, "v", 1.5, "open"))
    return tm, ents, d0y


def _loudest(rows):
    best = 0
    for r in rows.values():
        best = max(best, int(getattr(r, "loudness_tiles", 0) or 0))
    return best


class HordeScene:
    """The horde scene (no physics: the fire is a mask input written as the
    existing horde tests write it; slot 5 reads the planes as they stand).
    Units are added after three decoys are dropped, so no id equals its row.
    ``variant`` shifts the layout, the seed, the zombie count and the
    schedule; ``controller`` hands the horde to a zombie controller (the
    latch: SET for most ticks, a strict subset on some, EMPTY on every 9th).

    The cast: Mfar (room B, by D0's east side: big hp, the prey that stays),
    Mlow (room A, 1 hp: the first bite kills, the second bites the corpse,
    it turns at tick end and walks as a zombie next tick). Za / Zb awake and
    ready by Mlow; Zc and Zd awake ON ONE TILE before D0 (two zombies wanting
    one tile; they bash D0 until it breaks, the prey within the sprint
    radius but out of sight until then); Ze dormant in the sealed closet (it
    hears, but has no path); Zf awake inside a ring of fire (its feared field
    is unreached: the fearless field takes it to the ring, every step into it
    refused); Zh, Zi, Zj, Zk dormant in room C (they wander, hear the shots
    and the blast, walk the noise fields, settle; Zk hides behind the pillar:
    when Mfar is moved into C the others see it and the chain wakes Zk)."""

    def __init__(self, variant=0, controller=None, n_zombies=None):
        from simulation.factions import ZOMBIE_TEAM
        from simulation.unit import Unit
        self.v, self.k = variant, 0
        tm, ents, self.d0y = _scene_map(variant)
        self.sim = sim = _bare_sim(tm, TILE, ents, physics=False)
        sim.seed = 9100 + variant
        g = sim.gmap
        # D0 weakened (a fixture on its tiles' hp): three strikes break it
        for (y, x) in sim._doors[0].span:
            g.wall_hp[y, x] = 5 * ONE
        # the fire (mask inputs): a ring round Zf, and a band across room A
        g.fire[4:15, 6] = ONE
        g.fire[4:15, 16] = ONE
        g.fire[4, 6:17] = ONE
        g.fire[14, 6:17] = ONE
        if variant % 3 != 2:
            g.fire[20:40, 30:32] = ONE
            g.temperature[20:40, 32] = sim.horde_settings.fear_temp_q16
        y0 = self.d0y
        self.zombies, self.marines = [], []
        add = self._add
        for i in range(3):
            sim.add_unit(Unit(f"decoy{i}", x=90.0, y=45.0, team=0))
        sim.units[:] = []
        nz = 10 if n_zombies is None else int(n_zombies)
        self.m_far = add("Mfar", 52, y0, 0, hp=1.0e6)
        self.m_low = add("Mlow", 24, 44, 0, hp=1.0)
        zs = [("Za", 21, 44, True), ("Zb", 27, 44, True),
              ("Zc", 40, y0, True), ("Zd", 40, y0, True),
              ("Zh", 60, 36, False), ("Ze", 4, 47, False),
              ("Zf", 10, 8, True), ("Zi", 75, 45, False),
              ("Zj", 88, 36, False), ("Zk", 50, 46, False)]
        for i, (n, x, y, awake) in enumerate(zs[:nz]):
            z = add(n, x, y, ZOMBIE_TEAM)
            z.zombie_activated = awake
            z.last_melee_tick = -1000
        # NON-NEUTRAL HOOKS on every hook the slot reads (STRIKE: the bite;
        # WALK / SPRINT: the cadences), and one zombie on its own cadence
        from simulation import unit_traits
        hooks = {"Za": {"STRIKE": ONE * 3 // 2, "WALK": ONE * 3 // 4},
                 "Zb": {"STRIKE": ONE // 2, "SPRINT": ONE * 3 // 4},
                 "Zc": {"WALK": ONE * 5 // 4, "SPRINT": ONE * 3 // 2},
                 "Zh": {"WALK": ONE * 7 // 8}}
        for z in self.zombies:
            if z.name in hooks:
                unit_traits.compile_unit(sim, z, hooks[z.name])
            if z.name == "Zi":
                z.speed_ticks_per_tile = 9
        if controller is not None:
            sim.hand_zombies_to_controller()
        self.controller = controller
        self.shot_r = _loudest(sim.weapons_tables.weapons.by_name)
        self.blast_r = _loudest(sim.weapons_tables.payloads.by_name)

    def _add(self, name, x, y, team, hp=None):
        from simulation.unit import Unit
        u = Unit(name, x=float(x), y=float(y), team=team)
        self.sim.add_unit(u)
        if hp is not None:
            u.current_hp = hp
        (self.zombies if u.is_zombie else self.marines).append(u)
        return u

    def _ignite_first_candidate(self):
        """Fire on a tile of an awake walker's BEST candidate block that its
        second candidate's block does not hold (its cached prey field, between
        refreshes): its step refuses the first and takes the second."""
        g = self.sim.gmap
        st = self.sim.horde_state
        for z in self.zombies:
            if not (z.alive and z.zombie_activated):
                continue
            f = st.fields.get(("prey", z.footprint))
            if f is None:
                continue
            c = horde.candidates(f, z.tile_y, z.tile_x)
            if len(c) < 2 or c[0][0] != c[1][0]:
                continue
            fp = z.footprint
            a = {(c[0][2] + i, c[0][3] + j) for i in range(fp) for j in range(fp)}
            b = {(c[1][2] + i, c[1][3] + j) for i in range(fp) for j in range(fp)}
            for (yy, xx) in sorted(a - b):
                if g.material[yy, xx] == MAT_AIR:
                    g.fire[yy, xx] = ONE
                    return

    def _emit(self, y, x, r):
        from simulation import noise
        noise.emit(self.sim.gmap, y, x, r)

    def tick(self):
        sim, k = self.sim, self.k - (self.v % 4)
        if k == 4:
            self._emit(40, 70, self.shot_r)     # a gunshot's noise (the loudest weapon row)
        if k == 120:
            self._emit(48, 80, self.blast_r)    # a blast's noise (the loudest payload row)
        if 200 <= k < 240:                      # fire lit between refreshes on the best step
            self._ignite_first_candidate()
        if k == 2:                              # each dormant zombie its own target: five
            dorm = [z for z in self.zombies if z.alive and not z.zombie_activated]
            for z in dorm:
                self._emit(z.center_tile_y() + 1, z.center_tile_x(), 2)
        if k == 125:
            for z in self.zombies:              # a timeout on its next tick
                if z.alive and not z.zombie_activated and z.zombie_noise_ticks > 0:
                    z.zombie_noise_ticks = 1
                    break
        if k == 130:
            self._emit(20, 20, 300)             # heard by every sleeper; the closet: no path
        if k == 110:                            # Mfar into room C: a sight wake, the chain
            self.m_far.set_pos_q(90 * ONE, 46 * ONE)
        if k == 60 and len(self.zombies) > 3:   # a STUNNED basher: walks, cannot strike
            from simulation.status import STUNNED, apply_status
            apply_status(self.zombies[2], STUNNED, 1.0, 60)
        if k == 170 and len(self.zombies) > 3:  # a PARALYZED awake zombie: neither
            from simulation.status import PARALYZED, apply_status
            apply_status(self.zombies[0], PARALYZED, 1.0, 40)
        if k == 380:                            # every prey falls: no prey from here on
            from simulation.damage import DamagePacket, apply_packet
            for m in list(sim.units):
                if m.alive and not m.is_zombie:
                    apply_packet(m, DamagePacket(amount=1.0e7, dtype=0, attacker_id=-1),
                                 [], source="melee")
        if self.controller is not None:
            ids = [u.id for u in sim.units if u.alive and u.is_zombie]
            if k % 9 != 8:
                if k % 5 == 3:
                    ids = ids[1:]
                if ids:
                    sim.drive_zombies(ids)
        sim.step()
        sim.set_paused(False)
        self.k += 1


# ---------------------------------------------------------------------------
# The census: the Python law's branches, counted on the host's own decisions
# ---------------------------------------------------------------------------
class Census:
    """Wraps (reads only) the zombie law's functions and counts each branch."""

    def __init__(self):
        self.c = Counter()
        self._orig = []

    def _wrap(self, cont, attr, make):
        orig = getattr(cont, attr)
        w = make(orig)
        w.__wrapped__ = orig
        setattr(cont, attr, w)
        self._orig.append((cont, attr, orig))

    def install(self):
        import sys as _s
        from simulation import ai_zombie, simulation as simmod
        from simulation.simulation import Simulation
        c = self.c

        def run_slot(o):
            def f(sim, *a, **k):
                if sim._zombie_requests is not None and not sim._zombie_requests:
                    c["latch: an empty request set (returns, the cache standing)"] += 1
                elif sim._zombie_requests is not None:
                    c["latch: a request set"] += 1
                else:
                    c["latch: the conductor"] += 1
                if sim.gmap.noise_log:
                    c["drain: noises drained"] += 1
                return o(sim, *a, **k)
            return f
        self._wrap(Simulation, "_run_zombie_slot", run_slot)

        def upd(o):
            def f(gmap, units, tick, *a, **k):
                players = [u for u in units if not u.is_zombie and u.alive]
                zs = [u for u in units if u.is_zombie and u.alive]
                if not players:
                    c["no prey (returns, the cache standing)"] += 1
                    return o(gmap, units, tick, *a, **k)
                before = {u.id for u in zs if u.zombie_activated}
                c["_trig"] = 0
                out = o(gmap, units, tick, *a, **k)
                woke = {u.id for u in zs if u.zombie_activated} - before
                if woke:
                    c["pass 1: woken by sight"] += min(len(woke), c["_trig"])
                    if len(woke) > c["_trig"]:
                        c["pass 2: woken by the chain"] += 1
                return out
            return f
        self._wrap(simmod, "update_zombies_tick", upd)

        def los(o):
            def f(gmap, *a, **k):
                r = o(gmap, *a, **k)
                fr = _s._getframe(1)
                if fr.f_code.co_name == "update_zombies_tick":
                    c["pass 1: sight check " + ("clear" if r else "blocked")] += 1
                    if r:
                        c["_trig"] += 1
                elif fr.f_code.co_name == "_sees_prey":
                    c["sprint: sight " + ("clear" if r else "blocked")] += 1
                return r
            return f
        from simulation.gamemap import GameMap
        self._wrap(GameMap, "has_los", los)

        def begin(o):
            def f(st, gmap, tick, settings, doors=()):
                serial = getattr(gmap, "_tile_serial", 0)
                if st.serial is not None and int(tick) % settings.field_refresh_ticks == 0:
                    c["cache: dropped at a refresh tick"] += 1
                elif st.serial is not None and serial != st.serial:
                    c["cache: dropped by a topology change"] += 1
                elif st.serial is None:
                    c["cache: first drop"] += 1
                else:
                    c["cache: kept"] += 1
                return o(st, gmap, tick, settings, doors)
            return f
        self._wrap(horde.HordeState, "begin_tick", begin)

        def field(o):
            def f(st, gmap, key, fp, seeds, fearless=False):
                hit = key in st.fields
                tag = key[0] + (" fearless" if len(key) > 2 and key[2] == "fearless" else "")
                c[f"field: {tag} {'reused' if hit else 'flooded'}"] += 1
                if not hit and st.tile_ok_fearless is None:
                    c["mask: built"] += 1
                return o(st, gmap, key, fp, seeds, fearless)
            return f
        self._wrap(horde.HordeState, "_field", field)

        def walk(o):
            def f(*a, **k):
                r = o(*a, **k)
                c[f"walk: {r}"] += 1
                gait = a[-1] if len(a) >= 10 else k.get("gait")
                from simulation import unit_traits
                if gait == unit_traits.SPRINT:
                    c["walk: at the sprint"] += 1
                return r
            return f
        self._wrap(ai_zombie, "_walk", walk)

        def desc(o):
            def f(gmap, field, y, x, fp, state=None):
                cands = horde.candidates(field, y, x)
                if len(cands) >= 2 and cands[0][0] == cands[1][0]:
                    c["descent: a tie broken by the parity order"] += 1
                    if cands[0][2] != y and cands[1][2] != y or                             cands[0][3] != x and cands[1][3] != x:
                        c["descent: a tie between opposite neighbours"] += 1
                r = o(gmap, field, y, x, fp, state)
                if r is not None and r[0] == "step" and cands and (r[1], r[2]) != \
                        (cands[0][2], cands[0][3]):
                    c["descent: the first candidate refused (fear or a body)"] += 1
                return r
            return f
        self._wrap(horde, "descend", desc)

        def feared(o):
            def f(*a, **k):
                r = o(*a, **k)
                if r:
                    c["fear: a feared block refused"] += 1
                return r
            return f
        self._wrap(horde, "block_feared", feared)

        def bash(o):
            def f(gmap, z, ny, nx, tile, tick, flags, settings, events):
                from config import CFG
                from simulation.materials import MAT_DOOR_CLOSED
                if not flags.can_act:
                    c["bash: cannot act"] += 1
                elif tick - z.last_melee_tick < CFG.zombie.attack_cooldown_ticks:
                    c["bash: the cadence holds it"] += 1
                else:
                    c["bash: a strike"] += 1
                r = o(gmap, z, ny, nx, tile, tick, flags, settings, events)
                if int(gmap.material[tile]) != MAT_DOOR_CLOSED:
                    c["bash: the door tile broke"] += 1
                return r
            return f
        self._wrap(ai_zombie, "_bash", bash)

        def packet(o):
            def f(unit, pk, *a, **k):
                was = unit.alive
                r = o(unit, pk, *a, **k)
                c["bite: " + ("a kill" if was and not unit.alive else
                              "a corpse bitten" if not was else "a hit")] += 1
                return r
            return f
        self._wrap(ai_zombie, "apply_packet", packet)

        import inspect
        src, first = inspect.getsourcelines(ai_zombie.update_zombies_tick)
        why = {}
        for i, line in enumerate(src):
            for tag, name in (("# timed out", "settle: timed out"),
                              ("# arrived", "settle: arrived"),
                              ("# no path", "settle: no path to the noise")):
                if tag in line:
                    why[first + i] = name

        def settle(o):
            def f(u, settings):
                fr = _s._getframe(1)
                if fr.f_code.co_name == "_hear":
                    c["settle: dropped (noise_fields_max)"] += 1
                else:
                    c[why.get(fr.f_lineno, f"settle: at line {fr.f_lineno}")] += 1
                return o(u, settings)
            return f
        self._wrap(horde, "settle", settle)

        def hear(o):
            def f(zombies, noises, settings):
                before = {z.id: (z.zombie_noise_y, z.zombie_noise_x) for z in zombies}
                r = o(zombies, noises, settings)
                if any((z.zombie_noise_y, z.zombie_noise_x) != before[z.id] and
                       z.zombie_noise_ticks > 0 for z in zombies):
                    c["hearing: a target taken"] += 1
                return r
            return f
        self._wrap(ai_zombie, "_hear", hear)

        def wstep(o):
            def f(gmap, y, x, ay, ax, leash, uid, tick):
                cur = (y - ay) ** 2 + (x - ax) ** 2
                r = o(gmap, y, x, ay, ax, leash, uid, tick)
                c["wander: " + ("back toward the anchor" if cur > leash else
                                "a hashed step" if r is not None else
                                "the leash refuses")] += 1
                return r
            return f
        self._wrap(horde, "wander_step", wstep)

        def wdue(o):
            def f(uid, tick, period):
                r = o(uid, tick, period)
                c["wander: " + ("due" if r else "not due (idle)")] += 1
                return r
            return f
        self._wrap(horde, "wander_due", wdue)

        def clear(o):
            def f(u):
                if u.is_zombie and u.alive:
                    c["pass 2: an awake zombie's noise / anchor cleared"] += 1
                return o(u)
            return f
        self._wrap(horde, "clear", clear)
        return self

    def uninstall(self):
        for cont, attr, orig in reversed(self._orig):
            setattr(cont, attr, orig)
        self._orig.clear()


NEED = (
    "latch: the conductor", "latch: a request set",
    "latch: an empty request set (returns, the cache standing)",
    "no prey (returns, the cache standing)", "drain: noises drained",
    "pass 1: sight check clear", "pass 1: sight check blocked", "pass 1: woken by sight",
    "pass 2: woken by the chain", "pass 2: an awake zombie's noise / anchor cleared",
    "hearing: a target taken", "settle: dropped (noise_fields_max)",
    "settle: timed out", "settle: arrived", "settle: no path to the noise",
    "cache: first drop", "cache: dropped at a refresh tick",
    "cache: dropped by a topology change", "cache: kept",
    "field: prey flooded", "field: prey reused", "field: prey fearless flooded",
    "field: noise flooded", "field: noise reused", "mask: built",
    "walk: clock", "walk: step", "walk: wait", "walk: bash", "walk: at the sprint",
    "sprint: sight clear", "sprint: sight blocked",
    "descent: a tie broken by the parity order",
    "descent: a tie between opposite neighbours",
    "descent: the first candidate refused (fear or a body)",
    "fear: a feared block refused",
    "bash: a strike", "bash: the cadence holds it", "bash: the door tile broke",
    "bash: cannot act",
    "bite: a kill", "bite: a corpse bitten", "bite: a hit",
    "wander: due", "wander: not due (idle)", "wander: a hashed step",
)
UNREACHABLE = {
    "pass 1 / 3: a fault": "no shipped map or unit leaves the proved range (the device "
                           "refuses a run past it by name; PART OVERFLOW takes the cache's)",
}


# ---------------------------------------------------------------------------
# The shadow runner
# ---------------------------------------------------------------------------
def run_shadow(name, drivers, ticks, slots=None, twins=None, n_units=None, need_act=True):
    slots = slots or [SLOT_5]
    acts = Counter()
    n = 0
    t0 = time.perf_counter()
    with ShadowLockstep(drivers, list(slots), n_units=n_units) as ls:
        for _ in range(ticks):
            if any(getattr(d, "is_terminal", lambda: False)() for d in drivers):
                break
            reps = ls.tick()
            n += 1
            for lab, rep in reps.items():
                if not rep.equal:
                    fail(f"{name}: {ls.divergences[-1]}")
                    return acts, n
                # coverage: the rows the stage wrote, flagged once (a row changed
                # and changed back counts -- a noise heard and settled at once --,
                # the P7a.1 semantics): never fewer than the host changed
                if (rep.coverage < rep.host_changed).any():
                    fail(f"{name} tick {rep.tick}: coverage {rep.coverage.tolist()} < the "
                         f"rows the host changed {rep.host_changed.tolist()}")
                    return acts, n
                if (rep.coverage != rep.host_changed).any():
                    acts["ticks with a row changed and changed back"] += 1
                for e in range(len(drivers)):
                    if int(rep.host_changed[e]):
                        acts["rows changed"] += 1
                    if int(rep.books["horde_floods"][e]):
                        acts["floods"] += 1
                    if int(rep.books["horde_bites"][e]):
                        acts["bites"] += 1
                    if int(rep.books["horde_bashes"][e]):
                        acts["bashes"] += 1
            if twins is not None:
                from cuda_unit_verbs_check import _same
                for d, tw in zip(drivers, twins):
                    tw.tick()
                    if not _same(d.sim, tw.sim):
                        fail(f"{name}: the hooked match diverged from the unhooked one at "
                             f"tick {d.sim.tick}")
                        return acts, n
    print(f"{name}: N={len(drivers)}, {n} ticks, slot 5 equal every tick (rows, ring, noise "
          f"list, cache, planes, books, write set; coverage >= the rows changed)"
          f"{' (env i == env i alone)' if len(drivers) > 1 else ''}"
          f"{' (the unhooked twin equal)' if twins is not None else ''}; env-ticks: "
          f"{dict(acts)} ({time.perf_counter() - t0:.0f} s)")
    if need_act and not acts["rows changed"]:
        fail(f"{name}: slot 5 changed no row (vacuous)")
    return acts, n


class RidgeScene:
    """Two awake zombies each on a RIDGE of the prey field: Zv between a prey
    north and a prey south at equal path distance (an even cell: N and S tie,
    the vertical-first order decides), Zh between a prey west and a prey east
    (an odd cell: W and E tie). 1/2 m tiles, an open room."""

    def __init__(self):
        from simulation.factions import ZOMBIE_TEAM
        from simulation.unit import Unit
        tm = np.full((40, 60), MAT_HULL, dtype=np.int32)
        tm[1:39, 1:59] = MAT_AIR
        self.sim = sim = _bare_sim(tm, TILE, (), physics=False)
        for i in range(2):
            sim.add_unit(Unit(f"decoy{i}", x=5.0, y=5.0, team=0))
        sim.units[:] = []
        for n, x, y in (("Mn", 10, 2), ("Ms", 10, 34), ("Mw", 30, 20), ("Me", 55, 20)):
            u = Unit(n, x=float(x), y=float(y), team=0)
            sim.add_unit(u)
            u.current_hp = 1.0e9
        for n, x, y in (("Zv", 10, 18), ("Zh", 42, 19)):
            z = Unit(n, x=float(x), y=float(y), team=ZOMBIE_TEAM)
            sim.add_unit(z)
            z.zombie_activated = True

    def tick(self):
        self.sim.step()
        self.sim.set_paused(False)


def part_scene():
    census = Census().install()
    try:
        run_shadow("ridge scene", [RidgeScene()], 120)
        run_shadow("scene v0 (conductor)", [HordeScene(0)], SCENE_TICKS,
                   twins=[HordeScene(0)])
        run_shadow("scene v1 (controller)", [HordeScene(1, controller=True)], SCENE_TICKS)
        run_shadow("scene v2 (no fire band)", [HordeScene(2)], SCENE_TICKS)
    finally:
        census.uninstall()
    c = census.c
    for b in NEED:
        if not c[b]:
            fail(f"scene: the census branch {b!r} was never taken")
    print("scene census: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())
                                       if not k.startswith("_")))
    print("scene unreachable: " + "; ".join(f"{k} -- {v}" for k, v in UNREACHABLE.items()))


def part_cache():
    """The device's own cache across ticks (staged once at the first tick's
    begin, never again), compared with the host's every tick."""
    for v, ctl in ((0, None), (1, True), (3, None)):
        acts, n = run_shadow(f"cache v{v}{' (controller)' if ctl else ''}",
                             [HordeScene(v, controller=ctl)], SCENE_TICKS,
                             slots=[SLOT_5_LIVE])


# ---------------------------------------------------------------------------
# PART LEVELS
# ---------------------------------------------------------------------------
class ShipDriver:
    def __init__(self, name="ship_demo", seed=1):
        from simulation.match_session import MatchSession
        level = level_loader.load(name, str(_ROOT / "levels"))
        self.session = MatchSession(level, seed=seed, breach_physics=bp)
        self.sim = self.session.sim

    def tick(self):
        self.session.tick()

    def is_terminal(self):
        return self.session.is_terminal()


def part_levels():
    acts, n = run_shadow("ship_demo bots (4 zombies)", [ShipDriver()], SHIP_TICKS,
                         twins=[ShipDriver()])
    run_shadow("ship_demo bots, the device's own cache", [ShipDriver()], SHIP_TICKS,
               slots=[SLOT_5_LIVE])
    run_shadow("playground bots (door entities, 5 zombies)", [ShipDriver("playground")],
               BATCH_TICKS)


# ---------------------------------------------------------------------------
# PART BATCH
# ---------------------------------------------------------------------------
def _batch_drivers(n):
    """One grid (H x W) for every env: the scene's variants, differing in
    layout (the door, the inner wall), the fire band, the zombie count (one
    env with none) and the latch; env 0 NOT the largest."""
    if n == 2:
        return [HordeScene(2, n_zombies=4), HordeScene(1, controller=True)]
    out = [HordeScene(0, n_zombies=3), HordeScene(1, controller=True),
           HordeScene(2, n_zombies=0), HordeScene(3)]
    for v in range(4, n):
        out.append(HordeScene(v, n_zombies=5 + v % 4, controller=(v % 2 == 1) or None))
    return out


def part_batch():
    for n in BATCH_NS:
        drivers = _batch_drivers(n)
        counts = [sum(u.is_zombie for u in d.sim.units) for d in drivers]
        if counts[0] == max(counts) or len(set(counts)) < 2:
            fail(f"batch N={n}: zombie counts {counts} (env 0 the largest, or all equal)")
        run_shadow(f"batch N={n} (zombies {counts})", drivers, BATCH_TICKS)


# ---------------------------------------------------------------------------
# PART PADDING
# ---------------------------------------------------------------------------
def part_padding():
    """Two envs of different unit counts in one store (the short env's rows
    past its count are padding): garbage written there and into the unused
    cache slots changes no valid row, plane, list or cache record, and the
    padding is not written."""
    import cupy as cp
    from simulation import object_tables as ot
    scs = [HordeScene(0, n_zombies=2), HordeScene(1, n_zombies=9)]
    for _ in range(6):
        for s in scs:
            s.tick()
    sims = [s.sim for s in scs]
    U = max(len(s.units) for s in sims)
    lay = unit_store.layout()

    def run(dirty):
        st = unit_store.DeviceStore(2, U, bp=bp, tables=unit_tables.values_batch(sims))
        rows = np.stack([unit_store.pack(s.units, U) for s in sims])
        if dirty:
            junk = rows[1, 0].copy()
            junk[lay.valid] = 0
            for r in range(len(sims[0].units), U):
                rows[0, r] = junk
        st.upload(rows)
        ring = __import__("simulation.unit_events", fromlist=["x"]).EventRing(2, U, bp=bp)
        ring.begin()
        noise = ot.DeviceTable(unit_schema.NOISE_TABLE, 2, bp=bp)
        noise.stage_lists([unit_kernels.noise_of(s) for s in sims])
        doors = ot.DeviceTable(unit_schema.DOOR_TABLE, 2, bp=bp)
        tiles = ot.DeviceTable(unit_schema.DOOR_TILE_TABLE, 2, bp=bp)
        doors.stage_lists([unit_kernels.doors_of(s) for s in sims])
        tiles.stage_lists([unit_kernels.door_tiles_of(s) for s in sims])
        cache = horde_cache.DeviceHordeCache(2, U)
        cache.stage_lists([horde_cache.cache_records(s) for s in sims])
        if dirty:
            cache.keys[:, -1] = 99
            cache.dist[:, -1] = 7
        fields = {n: cp.asarray(np.stack([np.array(getattr(s.gmap, n)) for s in sims]))
                  for n in applier.PLANE_NAMES}
        latch = [horde_cache.latch_of(s, U) for s in sims]
        unit_kernels.horde(st, _applier_for(st), fields, ring, noise, doors, tiles, cache,
                           latch)
        return (st.download(), {k: v.get() for k, v in fields.items()}, ring.records(),
                noise.records(), cache.records(), rows, cache)
    clean, dirty = run(False), run(True)
    nv = len(sims[0].units)
    if not np.array_equal(clean[0][1], dirty[0][1]) or \
            not np.array_equal(clean[0][0, :nv], dirty[0][0, :nv]):
        fail("padding: garbage rows changed a valid row")
    if not np.array_equal(dirty[0][0, nv:], dirty[5][0, nv:]):
        fail("padding: a padding row was written")
    for k in applier.PLANE_NAMES:
        if not np.array_equal(clean[1][k], dirty[1][k]):
            fail(f"padding: garbage changed plane {k}")
            break
    if clean[2] != dirty[2] or clean[3] != dirty[3] or clean[4] != dirty[4]:
        fail("padding: garbage changed the ring, the noise list or the cache records")
    if int(dirty[6].keys[0, -1, 0].get()) != 99:
        fail("padding: an unused cache slot was written")
    if nv == len(sims[1].units):
        fail("padding: both envs hold the same unit count (vacuous)")
    print(f"padding: garbage rows past env 0's {nv} units and in the unused cache slots "
          f"neither read nor written")


# ---------------------------------------------------------------------------
# PART OVERFLOW
# ---------------------------------------------------------------------------
def part_overflow():
    import cupy as cp
    from simulation import object_tables as ot
    from simulation.unit_events import EventRing, UnitEventOverflow
    seen = Counter()
    sc = HordeScene(0)
    for _ in range(3):
        sc.tick()
    sim = sc.sim
    U = len(sim.units)

    def run(fcap=applier.HORDE_FIELD_CAP, e_max=None, noise_rows=None):
        st = unit_store.DeviceStore(1, U, bp=bp, tables=unit_tables.values_batch([sim]))
        st.upload(unit_store.pack(sim.units, U)[None])
        ring = EventRing(1, U, e_max, bp=bp) if e_max else EventRing(1, U, bp=bp)
        ring.begin()
        noise = ot.DeviceTable(unit_schema.NOISE_TABLE, 1, bp=bp,
                               cap=1 if noise_rows is not None else None)
        noise.stage_lists([noise_rows if noise_rows is not None else unit_kernels.noise_of(sim)])
        doors = ot.DeviceTable(unit_schema.DOOR_TABLE, 1, bp=bp)
        tiles = ot.DeviceTable(unit_schema.DOOR_TILE_TABLE, 1, bp=bp)
        doors.stage_lists([unit_kernels.doors_of(sim)])
        tiles.stage_lists([unit_kernels.door_tiles_of(sim)])
        cache = horde_cache.DeviceHordeCache(1, U, fcap=fcap)
        cache.begin()
        fields = {n: cp.asarray(np.array(getattr(sim.gmap, n))[None])
                  for n in applier.PLANE_NAMES}
        unit_kernels.horde(st, _applier_for(st), fields, ring, noise, doors, tiles, cache,
                           [horde_cache.latch_of(sim, U)])
    for _t in range(120):                     # a tick that floods two fields at once
        try:
            run(fcap=1)
        except applier.HordeCacheOverflow as ex:
            if "capacity 1" not in str(ex) or "[0]" not in str(ex):
                fail(f"overflow: the cache refusal does not name the env and the capacity: "
                     f"{ex}")
            seen["cache"] += 1
            break
        sc.tick()
    else:
        fail("overflow: a cache of one field took every tick's fields")
    try:
        cache = horde_cache.DeviceHordeCache(1, U, fcap=1)
        h, w = sim.gmap.material.shape
        hdr = np.zeros(len(applier.HORDE_HEADER), dtype=np.int64)
        hdr[applier.HORDE_HEADER.index("n_fields")] = 2
        rc = horde_cache.CacheRecords(hdr, [(0, 3, 0, 0), (2, 3, 4, 5)],
                                      np.zeros((2, h, w), dtype=np.int32),
                                      np.zeros((3, h, w), dtype=np.uint8))
        cache.stage_lists([rc])
        fail("overflow: a cache of one slot was staged with more fields")
    except applier.HordeCacheOverflow as ex:
        if "env 0" not in str(ex):
            fail(f"overflow: the staging refusal does not name the env: {ex}")
        seen["cache staging"] += 1
    except RuntimeError:
        fail("overflow: the scene's cache held fewer than two fields (vacuous)")
    print(f"overflow: past their capacities refused by name {dict(seen)} (the noise list's and "
          f"the ring's own refusals are the table's / ring's: UNIT_HORDE_PARTS=refuse)")


def part_refuse():
    """The noise list full at a bash and the ring full at a bite refuse the run
    by name (the record list's own overflow)."""
    import cupy as cp
    from simulation import object_tables as ot
    from simulation.unit_events import EventRing, UnitEventOverflow
    seen = Counter()
    # a bite with no room on the ring: the scene's first tick bites (Za / Zb on Mlow)
    sc = HordeScene(0)
    sim = sc.sim
    U = len(sim.units)
    st = unit_store.DeviceStore(1, U, bp=bp, tables=unit_tables.values_batch([sim]))
    st.upload(unit_store.pack(sim.units, U)[None])
    ring = EventRing(1, U, 1, bp=bp)
    ring.begin()
    noise = ot.DeviceTable(unit_schema.NOISE_TABLE, 1, bp=bp)
    noise.begin()
    doors = ot.DeviceTable(unit_schema.DOOR_TABLE, 1, bp=bp)
    tiles = ot.DeviceTable(unit_schema.DOOR_TILE_TABLE, 1, bp=bp)
    doors.stage_lists([unit_kernels.doors_of(sim)])
    tiles.stage_lists([unit_kernels.door_tiles_of(sim)])
    cache = horde_cache.DeviceHordeCache(1, U)
    cache.begin()
    fields = {n: cp.asarray(np.array(getattr(sim.gmap, n))[None]) for n in applier.PLANE_NAMES}
    try:
        unit_kernels.horde(st, _applier_for(st), fields, ring, noise, doors, tiles, cache,
                           [horde_cache.latch_of(sim, U)])
        fail("refuse: a ring of one event took two bites' events")
    except UnitEventOverflow as ex:
        if "[0]" not in str(ex):
            fail(f"refuse: the ring refusal does not name the env: {ex}")
        seen["ring"] += 1
    # a noise list of one row at a tick where two zombies strike D0 at once
    sc = HordeScene(0)
    for _t in range(300):
        sim = sc.sim
        U = len(sim.units)
        st = unit_store.DeviceStore(1, U, bp=bp, tables=unit_tables.values_batch([sim]))
        st.upload(unit_store.pack(sim.units, U)[None])
        ring = EventRing(1, U, bp=bp)
        ring.begin()
        noise = ot.DeviceTable(unit_schema.NOISE_TABLE, 1, bp=bp, cap=1)
        noise.begin()
        doors.stage_lists([unit_kernels.doors_of(sim)])
        tiles.stage_lists([unit_kernels.door_tiles_of(sim)])
        cache = horde_cache.DeviceHordeCache(1, U)
        cache.stage_lists([horde_cache.cache_records(sim)])
        fields = {n: cp.asarray(np.array(getattr(sim.gmap, n))[None])
                  for n in applier.PLANE_NAMES}
        try:
            unit_kernels.horde(st, _applier_for(st), fields, ring, noise, doors, tiles, cache,
                               [horde_cache.latch_of(sim, U)])
        except ot.ObjectTableOverflow as ex:
            if "'noise'" not in str(ex) or "env 0" not in str(ex):
                fail(f"refuse: the noise refusal does not name the table and the env: {ex}")
            seen["noise list"] += 1
            break
        sc.tick()
    else:
        fail("refuse: no tick struck twice (the noise list's refusal vacuous)")
    print(f"refuse: refused by name {dict(seen)}")


# ---------------------------------------------------------------------------
# PART CHAINED
# ---------------------------------------------------------------------------
def _host_side_5(ch, sim_of):
    """Slot 5's host-only side after the device ran it: the books landed (a
    bash that broke a tile), the render marks and the sky flag of a changed
    tile, the serial, the request set consumed (reqs.clear())."""
    def hs(sim, _pre, _post):
        g = sim.gmap
        b = ch.books
        g.n_destruction_seed_sum += b["n_destruction_seed"]
        for name, key in (("ge_retire", "retire"), ("ge_mint", "mint"),
                          ("ge_destroy_seed", "destroy_seed")):
            g._gas_energy_book(key, b[name])
        g._solid_energy_book("destroy_wall", b["solid_destroy_wall"])
        mat0 = ch.inputs._host_fields["material"][0]
        changed = np.argwhere(mat0 != g.material)
        for (y, x) in changed.tolist():
            g.render_dirty.add((int(y), int(x)))
        if len(changed):
            g._sky_mask_dirty = True
        g._tile_serial += b["tiles_changed"]
        if sim._zombie_requests is not None:
            sim._zombie_requests.clear()
    return hs


def _chained(name, make, ticks, slots=None, extra=None):
    from cuda_unit_topology_check import _same_twin
    from simulation import simulation as simmod
    twin, ref = make(), make()
    host_calls = Counter()
    o_upd = simmod.update_zombies_tick

    def upd(gmap, *a, __o=o_upd, **k):
        host_calls["twin" if gmap is twin.sim.gmap else "reference"] += 1
        return __o(gmap, *a, **k)
    simmod.update_zombies_tick = upd
    holder = {}
    slots = slots or [SLOT_5]
    ch = ChainedSlots(twin, slots, bp=bp, host_side=lambda s, a, b: holder["hs"](s, a, b))
    holder["hs"] = _host_side_5(ch, None)
    n = 0
    t0 = time.perf_counter()
    try:
        with ch:
            for _ in range(ticks):
                if getattr(ref, "is_terminal", lambda: False)():
                    break
                twin.tick()
                ref.tick()
                n += 1
                why = _same_twin(twin.sim, ref.sim)
                if why is None and twin.sim.gmap.noise_log != ref.sim.gmap.noise_log:
                    why = "the noise log"
                if why is None and list(horde_cache.cache_records(twin.sim)) != \
                        list(horde_cache.cache_records(ref.sim)):
                    why = "the horde cache"
                if why is not None:
                    fail(f"chained {name}: the twin left the reference at tick "
                         f"{twin.sim.tick} ({why})")
                    break
    finally:
        simmod.update_zombies_tick = o_upd
    if host_calls["twin"] or not host_calls["reference"]:
        fail(f"chained {name}: the host zombie law ran in the twin {dict(host_calls)}")
    print(f"chained {name}: {n} ticks, the twin's unit hash, field digest, planes, books, "
          f"noise log and horde cache equal the reference every tick ({ch.runs} chain runs; "
          f"host law calls {dict(host_calls)}; {time.perf_counter() - t0:.0f} s)")
    if ch.runs == 0:
        fail(f"chained {name}: the chain never ran")


def part_chained():
    _chained("scene v0", lambda: HordeScene(0), CHAIN_TICKS)
    _chained("scene v1 (controller)", lambda: HordeScene(1, controller=True), CHAIN_TICKS)
    _chained("ship_demo bots", ShipDriver, SHIP_TICKS)


def part_chained_all():
    """Slot 5 chained WITH the device slots around it: ship_demo's bots match
    with slot 2' + its landing in one chain and 4 + 4a + 4c + 5 in another
    (the shooting's noise list continued into slot 5's drain on the device);
    the horde scene with 4 + 4a + 4c + 5 and 9e(d)'s door sweep (the bashed
    door dies whole at the sweep). Unit hash, field digest, planes, books,
    noise log and cache equal the untouched reference every tick; the host
    zombie, shooting and door laws run 0 times in the twin."""
    from cuda_unit_doors_check import SLOT_9ED, _host_side_9ed
    from cuda_unit_shooting_check import (SLOT_2L, SLOT_2R, SLOT_4, SLOT_4A, SLOT_4C,
                                          _HostCalls, _host_side_of)
    from cuda_unit_topology_check import _same_twin
    from simulation import door_system, simulation as simmod
    for name, make, ticks, doors in (("ship_demo bots", ShipDriver, SHIP_TICKS, False),
                                     ("scene v0", lambda: HordeScene(0), CHAIN_TICKS, True)):
        twin, ref = make(), make()
        host = Counter()
        o_upd, o_sw = simmod.update_zombies_tick, simmod.sweep_doors

        def upd(gmap, *a, __o=o_upd, **k):
            host["zombie law (twin)" if gmap is twin.sim.gmap else "zombie law (ref)"] += 1
            return __o(gmap, *a, **k)

        def sw(sim, __o=o_sw):
            host["door sweep (twin)" if sim is twin.sim else "door sweep (ref)"] += 1
            return __o(sim)
        simmod.update_zombies_tick, simmod.sweep_doors = upd, sw
        h2, h4 = {}, {}
        holder = {}
        t0 = time.perf_counter()
        n = 0
        try:
            c2 = ChainedSlots(twin, [SLOT_2R, SLOT_2L], host_side=_host_side_of(h2), bp=bp)
            c4 = ChainedSlots(twin, [SLOT_4, SLOT_4A, SLOT_4C, SLOT_5], bp=bp,
                              host_side=lambda s, a, b: holder["h4"](s, a, b))
            hs_5 = _host_side_5(c4, None)

            def h4f(s, a, b):           # 4c's impulse books, then the chain's books once
                from cuda_unit_verbs_check import _books_host_side
                _books_host_side(s, a, b)
                hs_5(s, a, b)
            holder["h4"] = h4f
            h2["ch"], h4["ch"] = c2, c4
            chains = [c2, c4]
            if doors:
                c9 = ChainedSlots(twin, [SLOT_9ED], bp=bp,
                                  host_side=lambda s, a, b: holder["h9"](s, a, b))
                holder["h9"] = _host_side_9ed(c9)
                chains.append(c9)
            from contextlib import ExitStack
            with ExitStack() as stack, _HostCalls(twin.sim) as hc:
                for c in chains:
                    stack.enter_context(c)
                for _ in range(ticks):
                    if getattr(ref, "is_terminal", lambda: False)():
                        break
                    twin.tick()
                    ref.tick()
                    n += 1
                    why = _same_twin(twin.sim, ref.sim)
                    if why is None and twin.sim.gmap.noise_log != ref.sim.gmap.noise_log:
                        why = "the noise log"
                    if why is None and list(horde_cache.cache_records(twin.sim)) !=                             list(horde_cache.cache_records(ref.sim)):
                        why = "the horde cache"
                    if why is not None:
                        fail(f"chained all {name}: the twin left the reference at tick "
                             f"{twin.sim.tick} ({why})")
                        break
        finally:
            simmod.update_zombies_tick, simmod.sweep_doors = o_upd, o_sw
        twin_calls = {k: v for k, v in host.items() if "twin" in k}
        twin_calls.update({k: v for k, v in hc.n.items() if v})
        print(f"chained all {name}: {n} ticks, chains {[c.runs for c in chains]} "
              f"(2' + landing | 4 + 4a + 4c + 5{' | 9e(d)' if doors else ''}), the twin equal "
              f"to the reference every tick; host law calls in the twin {twin_calls or 0}, "
              f"in the reference {dict((k, v) for k, v in host.items() if 'ref' in k)} "
              f"({time.perf_counter() - t0:.0f} s)")
        if twin_calls:
            fail(f"chained all {name}: the host law ran in the twin {twin_calls}")
        if any(c.runs != n for c in chains[1:]):
            fail(f"chained all {name}: a chain skipped a tick")


# ---------------------------------------------------------------------------
PARTS = ("tables", "flood", "scene", "cache", "levels", "batch", "padding", "overflow",
         "refuse", "chained", "chained_all")


def main(parts=None) -> int:
    if not getattr(bp, "HAS_CUDA", False):
        print("UNIT_HORDE_RESULT: FAIL (not the CUDA build)")
        return 1
    parts = parts or os.environ.get("UNIT_HORDE_PARTS", ",".join(PARTS)).split(",")
    t0 = time.perf_counter()
    for p in parts:
        tp = time.perf_counter()
        try:
            globals()[f"part_{p}"]()
        except Exception as ex:                      # noqa: BLE001
            import traceback
            traceback.print_exc()
            fail(f"part {p} raised {type(ex).__name__}: {ex}")
        print(f"[part {p}: {time.perf_counter() - tp:.0f} s]")
    ok = not FAILS
    print(f"UNIT_HORDE_RESULT: {'PASS' if ok else 'FAIL'} ({len(FAILS)} failures, "
          f"{time.perf_counter() - t0:.0f} s)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1].split(",") if len(sys.argv) > 1 else None))


# ---------------------------------------------------------------------------
# bench
# ---------------------------------------------------------------------------
def _bench_sim(n_zombies):
    """ship_demo's map (no physics, no match) with 4 marines and ``n_zombies``
    zombies on free tiles spread over the map."""
    from simulation import Simulation
    from simulation.factions import ZOMBIE_TEAM
    from simulation.ruleset import ContinuousRealtime
    from simulation.unit import Unit
    level = level_loader.load("ship_demo", str(_ROOT / "levels"))
    sim = Simulation(level, seed=1, breach_physics=None, enable_recorder=False,
                     ruleset=ContinuousRealtime())
    g = sim.gmap
    free = [(int(y), int(x)) for y, x in np.argwhere(horde.block_passable(
        horde.walkable_tiles(g), 3))]
    rng = np.random.default_rng(3)
    pick = rng.choice(len(free), size=4 + n_zombies, replace=False)
    for i, k in enumerate(pick):
        y, x = free[k]
        u = Unit(f"U{i}", x=float(x), y=float(y), team=0 if i < 4 else ZOMBIE_TEAM)
        sim.add_unit(u)
        if i < 4:
            u.current_hp = 1.0e9
    return sim


def bench(reps: int = 50) -> int:
    """TIMING (median ms of the HORDE stage's launch, its books' download and
    the Python face; inputs re-staged per rep, not timed), N copies of
    ship_demo's map: FLOOR (an empty request set: the drain only); PASSES 1 +
    2 (every zombie dormant, the hearing and the wander off in the tables:
    pass 3 holds the woken on their stride clock); PASS 3 (every zombie awake and due, its prey field
    in the staged cache: the bites and the descent, no flood); PASS 3 + FLOOD
    (the same with a fresh cache: the mask and one prey flood per env)."""
    import cupy as cp
    from simulation import object_tables as ot
    from simulation.unit_events import EventRing
    rows = []
    for nz in (4, 32, 256):
        sim = _bench_sim(nz)
        U = len(sim.units)
        # the host's own cache after one slot with every zombie awake (its prey field)
        for u in sim.units:
            if u.is_zombie:
                u.zombie_activated = True
                u.zombie_move_accumulator = 10 ** 6
        sim._run_zombie_slot()
        for u in sim.units:
            if u.is_zombie:
                u.zombie_move_accumulator = 10 ** 6
                u.last_melee_tick = -10 ** 6
        awake_rows = unit_store.pack(sim.units, U)
        cache_rc = horde_cache.cache_records(sim)
        for u in sim.units:
            if u.is_zombie:
                u.zombie_activated = False
                u.zombie_move_accumulator = 0       # the woken stay on their stride clock
        dorm_rows = unit_store.pack(sim.units, U)
        for n in (1, 8, 64):
            tab = unit_tables.values_batch([sim] * n)
            quiet = tab.copy()
            lay = unit_tables.layout()
            for k in ("horde_wander", "horde_hearing"):
                quiet[:, lay.column(k).offset] = 0
            fields = {k: cp.asarray(np.repeat(np.array(getattr(sim.gmap, k))[None], n, 0))
                      for k in applier.PLANE_NAMES}
            ring = EventRing(n, U, bp=bp)
            noise = ot.DeviceTable(unit_schema.NOISE_TABLE, n, bp=bp)
            doors = ot.DeviceTable(unit_schema.DOOR_TABLE, n, bp=bp)
            tiles = ot.DeviceTable(unit_schema.DOOR_TILE_TABLE, n, bp=bp)
            doors.begin()
            tiles.begin()
            cache = horde_cache.DeviceHordeCache(n, U)
            latch = horde_cache.latch_of(sim, U)
            empty = latch.copy()
            empty[2] = 0
            timings = {}
            for name, tb, rws, lt, staged in (
                    ("floor (drain only)", tab, awake_rows, empty, True),
                    ("passes 1 + 2", quiet, dorm_rows, latch, True),
                    ("pass 3", tab, awake_rows, latch, True),
                    ("pass 3 + one flood", tab, awake_rows, latch, False)):
                st = unit_store.DeviceStore(n, U, bp=bp, tables=tb)
                ts = []
                for _ in range(reps):
                    st.upload(np.repeat(rws[None], n, 0))
                    ring.begin()
                    noise.begin()
                    if staged:
                        cache.stage_lists([cache_rc] * n)
                    else:
                        cache.begin()
                    t0 = time.perf_counter()
                    unit_kernels.horde(st, _applier_for(st), fields, ring, noise, doors, tiles,
                                       cache, [lt] * n)
                    ts.append((time.perf_counter() - t0) * 1e3)
                timings[name] = float(np.median(ts))
            rows.append((nz, n, timings))
            print(f"zombies {nz:3d} N {n:2d}: " + ", ".join(f"{k} {v:.3f}"
                                                           for k, v in timings.items()))
    print("TIMING table (median ms):")
    for nz in (4, 32, 256):
        for name in ("floor (drain only)", "passes 1 + 2", "pass 3", "pass 3 + one flood"):
            vals = [t[name] for z, n, t in rows if z == nz]
            print(f"| {name} | {nz} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    return 0
