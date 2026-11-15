"""#29 P7.0a -- THE KIT AND SLOT 9d GATE: the device draws, the device sight
function and the falloffs against their Python on synthetic inputs over their
whole range, and slot 9d (ignition from temperature) against
``combat.apply_temperature_ignition`` in the shadow lockstep, all at tol 0.

Runs inside the GPU subprocess (tests/cuda_harness.run_cuda_script; wrapper
tests/test_cuda_unit_kit.py). The kit is ``cpp/src/cuda_unit_draws.cuh``,
``cpp/src/cuda_unit_sight.cuh`` and ``fixed_point.h``'s falloff twins, probed
through the TEST-ONLY ``cpp/src/cuda_unit_kit_probe.cu``; slot 9d is
``cpp/src/cuda_unit_ignition.cu``, its Python face ``simulation.unit_kernels.
ignition``; the harness ``tests/cuda_unit_lockstep.py`` (FIELD OUTPUTS and the
write-set gate, the chained form).

PART TABLES -- the ign_* rows read back from the device equal the law's own
objects (the map's materials and gas tables, ``[physics.fire]``,
``combat.IGNITION_X_N_FLOOR``) per env of two maps.

PART DRAWS -- ``MatchDraws.at`` / ``.next``, ``cone_offset_q16``, ``roll_q16``
on a batch of envs whose keys differ (seed 0, 2^32 + k, 2^64 - 1, a tick past
2^32, a WEGO tick_base) over thousands of (salt, id, index) each -- every Salt
member and raw words, ids -1 / 0 / 2^32 - 2, indices 0 / 2^32 - 1, cones 0 /
pi / CONE_MAX, probabilities 0 / 1 / negative -- and two .next sites per row
interleaved (each site's own count); env i of the batch == env i alone; the
out-of-domain operands (an id of -2 or 2^32 - 1, an index of -1 or 2^32, a
cone past CONE_MAX, a key word past 2^32) flagged bad.

PART SIGHT -- ``GameMap.has_los`` in both forms on random pairs (short and
long lines, out-of-grid ends, start == end, solid starts) over four real maps
(ship_demo, playground, my_ship, test_level) with grille tiles stamped through
``on_tile_changed`` (no shipped level holds a grille), in ONE batch whose envs
hold different maps (padded to the largest with open tiles, which stop
nothing, as an out-of-grid tile does not): both answers often in both forms,
sight != physical on some pairs.

PART FALLOFF -- ``falloff_floor`` / ``falloff_round`` on the device and the
CPU twin (``bp.fp_falloff_*``) against ``unit_fixed`` over the callers'
ranges and the range ends (v 2^31 - 1, d2 = r2 - 1, r2 up to 2^62); outside
the proved range (v >= 2^31, d2 < 0) the twin refuses (-1).

PART LAW -- slot 9d's kernel on random synthetic planes (8 envs x 40 trials)
against ``apply_temperature_ignition`` on a stand-in map holding the same
planes: temperatures AT the threshold and one either side, fuel -1 / 0 / 1,
gas in vacuum tiles, open-neighbour sums below the fraction's floor, O2
fractions a count either side of X_ext -- the edges no scene reaches.

PART SCENES -- the SHADOW LOCKSTEP of slot 9d on an ignition scene built for
the gate (wood heated on a schedule so tiles ignite on many ticks; a burning
tile cooled so it re-arms; a hot armed tile lit by hand; a hot fuel-less tile;
a hot tile walled in with no open neighbour; a hot tile beside a pocket pumped
to near vacuum; a tile beside space; a tile on the grid edge) with a CENSUS
of the law's eleven branches, each required; and ship_demo's bots match, where ignition never happens (0
cells changed in 300 ticks). Coverage == the cells the host changed.

PART BATCH -- N = 2 and 8 ignition scenes that differ in layout (partition
columns, pockets), tile size, boundary, seed and heat schedule: env i == env
i alone, fields and write set.

PART CHAINED -- slots 9c, 9c2, 9c3 and 9d replaced by the device's in a twin
ignition match: per-tick unit hash and field digest equal an untouched
reference (fire and ignition_armed written back), 9d acting on many ticks,
the host law run 0 times in the twin and every tick in the reference.

PROPERTY: the device draws are MatchDraws / cone_offset_q16 / roll_q16 word
for word per env; the device has_los is GameMap.has_los in both forms; the
falloff twins are unit_fixed's integer for integer; slot 9d equals
apply_temperature_ignition on fire and ignition_armed with no other write.
BREAKS IF a counter word's order, the id word, the .next count's locality,
the cone's product or offset, the roll's shift, the walk's step rule, the
start / end test, the screen's use in either form, the falloff's rounding or
its 128-bit division, the re-arm's condition, the fuel gate, the open-
neighbour mask, the fraction's floor or division, the seed or the disarm
differs from the reference. Each validated by breaking it (the patch's
report).

Prints per-part lines and ``UNIT_KIT_RESULT: PASS`` / ``FAIL``; exits 0 / 1.
``bench()`` prints the TIMING table of slot 9d.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

import breach_physics as bp   # the CUDA build (cuda_harness bootstrap): FIRST

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "tools") not in sys.path:
    sys.path.append(str(_ROOT / "tools"))   # appended: never ahead of the CUDA build

import level_loader  # noqa: E402
from cuda_unit_lockstep import (ChainedSlots, DeviceSlot,  # noqa: E402
                                ShadowLockstep)
from simulation import unit_draws, unit_fixed, unit_kernels, unit_store, unit_tables  # noqa: E402
from simulation.draws import Draw, MatchDraws  # noqa: E402
from simulation.materials import MAT_AIR, MAT_GRILLE, MAT_HULL, MAT_WOOD  # noqa: E402
from simulation.rng_salts import Salt  # noqa: E402

FAILS: list = []
SCENE_TICKS = int(os.environ.get("UNIT_KIT_SCENE_TICKS", "120"))
SHIP_TICKS = int(os.environ.get("UNIT_KIT_SHIP_TICKS", "300"))
BATCH_TICKS = int(os.environ.get("UNIT_KIT_BATCH_TICKS", "60"))
CHAIN_TICKS = int(os.environ.get("UNIT_KIT_CHAIN_TICKS", "120"))
BATCH_NS = [int(x) for x in os.environ.get("UNIT_KIT_BATCH_NS", "2,8").split(",")]
DRAW_ROWS = int(os.environ.get("UNIT_KIT_DRAW_ROWS", "4096"))
SIGHT_ROWS = int(os.environ.get("UNIT_KIT_SIGHT_ROWS", "12000"))
M32 = (1 << 32) - 1
CONE_MAX = (1 << 31) - 1


def fail(msg):
    FAILS.append(msg)
    print("FAIL:", msg)


# ---------------------------------------------------------------------------
# The slot
# ---------------------------------------------------------------------------
IGN_INPUTS = tuple("field:" + n for n, _ in unit_kernels.IGNITION_FIELDS
                   if n not in ("fire", "ignition_armed"))
IGN_OUTPUTS = ("field:fire", "field:ignition_armed")


def _run9d(st, inp):
    return unit_kernels.ignition(st, inp.fields)


SLOT_9D = DeviceSlot(label="9d ignition", run=_run9d, inputs=IGN_INPUTS,
                     outputs=IGN_OUTPUTS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SPACE = level_loader.SPACE_CODE


def _level(tm, name, ts, boundary):
    from level_loader import LevelData
    return LevelData(name=name, version="3", path=Path("."), tilemap=tm,
                     tile_size_m=ts, diffuse_path=Path("."), boundary=boundary)


def _sim(level, seed=7):
    from simulation import Simulation
    from simulation.ruleset import ContinuousRealtime
    sim = Simulation(level, seed=seed, breach_physics=bp, enable_recorder=False,
                     ruleset=ContinuousRealtime())
    sim.set_paused(False)
    return sim


def _add_unit(sim, x, y, name="U0"):
    from simulation.unit import Unit
    u = Unit(name, x=float(x), y=float(y), team=0)
    sim.add_unit(u)
    return u


def table_reference(sim) -> dict:
    """The P7.0a device-table rows read from the law's own objects."""
    from config import CFG
    from simulation import combat, fire_fixed
    fc = getattr(CFG.physics, "fire", None)
    g = sim.gmap
    return {
        "ign_temp_q": tuple(int(x) for x in g.materials.ignition_temp_q16),
        "ign_x_ext_q": int(fire_fixed.quantize_scalar(float(getattr(fc, "o2_frac_ext", 0.13)))),
        "ign_seed_q": int(fire_fixed.quantize_scalar(float(getattr(fc, "ignition_seed", 0.1)))),
        "ign_x_n_floor_q": int(fire_fixed.quantize_scalar(combat.IGNITION_X_N_FLOOR)),
        "ign_gas_o2": int(g.gases.name_to_id["o2"]),
        "ign_gas_n2": int(g.gases.name_to_id["inert_n2"]),
    }


def part_tables():
    sims = [IgnitionScene(0).sim, ShipDriver().sim]
    st = unit_store.DeviceStore(2, 2, bp=bp, tables=unit_tables.values_batch(sims))
    got = st.tables.download()
    n = 0
    as_t = lambda x: tuple(x) if isinstance(x, (tuple, list)) else (x,)  # noqa: E731
    for e, sim in enumerate(sims):
        for k, v in table_reference(sim).items():
            dv = unit_tables.read(got[e], k)
            n += 1
            if as_t(dv) != as_t(v):
                fail(f"tables env {e} row {k}: device {dv} != reference {v}")
    if not any(unit_tables.read(got[0], "ign_temp_q")):
        fail("tables: no material has an ignition threshold (vacuous)")
    print(f"tables: {n} P7.0a rows over 2 envs equal the law's objects")


# ---------------------------------------------------------------------------
# PART DRAWS
# ---------------------------------------------------------------------------
class _Holder:
    def __init__(self, draws):
        self.draws = draws


def _env_draws():
    """Four envs whose keys and ticks differ (range ends included)."""
    envs = []
    for seed, tick, base in ((0, 0, 0), ((1 << 32) + 77, 123456, 0),
                             ((1 << 64) - 1, (1 << 32) + 5, 0), (31337, 900, 4500)):
        d = MatchDraws(seed)
        d.set_tick(tick)
        d.tick_base = base
        envs.append(d)
    return envs


def _draw_args(rng, R):
    salts = [int(s) for s in Salt]
    a = np.zeros((R, 6), dtype=np.int64)
    for r in range(R):
        if r % 4 == 3:                         # a raw word, not a member
            s, s2 = (int(x) for x in rng.integers(0, 1 << 32, size=2))
            if s2 == s:
                s2 = (s + 1) & M32
        else:
            i, j = rng.choice(len(salts), size=2, replace=False)
            s, s2 = salts[i], salts[j]
        ident = int(rng.choice([-1, 0, M32 - 1, int(rng.integers(0, 1 << 20)),
                                int(rng.integers(0, M32 - 1))]))
        index = int(rng.choice([0, M32, int(rng.integers(0, 1 << 16)),
                                int(rng.integers(0, 1 << 32))]))
        cone = int(rng.choice([0, 1, 205887, CONE_MAX, int(rng.integers(0, 1 << 18)),
                               int(rng.integers(0, CONE_MAX))]))
        p = int(rng.choice([0, 65536, -5, 1, int(rng.integers(0, 65537)),
                            int(rng.integers(-(1 << 40), 1 << 40))]))
        a[r] = (s, ident, index, cone, p, s2)
    return a


def _draw_want(d, row):
    s, ident, index, cone, p, s2 = (int(x) for x in row)
    w = Draw(d, d.tick, index, s, ident).word()
    out = [w, ((w * (2 * cone + 1)) >> 32) - cone, int((w >> 16) < p), 0]
    # two .next sites, interleaved, each with its own count (a fresh count per row)
    ka = kb = 0
    for _ in range(3):
        out.append(Draw(d, d.tick, ka, s, ident).word())
        out.append(Draw(d, d.tick, kb, s2, ident).word())
        ka += 1
        kb += 1
    return out


def part_draws():
    rng = np.random.default_rng(29070)
    envs = _env_draws()
    N, R = len(envs), DRAW_ROWS
    keys = unit_draws.draw_keys([_Holder(d) for d in envs])
    args = np.stack([_draw_args(rng, R) for _ in range(N)])
    got = unit_kernels.kit_draws_probe(bp, keys, args)
    bad = 0
    stats = dict(roll_true=0, roll_false=0, cone_neg=0, cone_pos=0, cone_zero=0)
    for e, d in enumerate(envs):
        for r in range(R):
            want = _draw_want(d, args[e, r])
            if list(int(x) for x in got[e, r]) != want:
                bad += 1
                if bad <= 3:
                    fail(f"draws env {e} row {r} args {args[e, r].tolist()}: device "
                         f"{got[e, r].tolist()} != python {want}")
            stats["roll_true" if want[2] else "roll_false"] += 1
            c = want[1]
            stats["cone_neg" if c < 0 else "cone_pos" if c > 0 else "cone_zero"] += 1
    # the .next form against MatchDraws.next itself (the host dict's count),
    # on the member-salt rows: a fresh tick per row so the count restarts
    n_next = 0
    for e, d0 in enumerate(envs):
        for r in range(0, R, 7):
            s, ident, _i, _c, _p, s2 = (int(x) for x in args[e, r])
            if r % 4 == 3:
                continue
            d = MatchDraws(d0.seed)
            d.set_tick(d0._fixed_tick)
            d.tick_base = d0.tick_base
            seq = []
            for _ in range(3):
                seq.append(d.next(Salt(s), ident).word())
                seq.append(d.next(Salt(s2), ident).word())
            n_next += 1
            if seq != [int(x) for x in got[e, r, 4:10]]:
                fail(f"draws env {e} row {r}: device .next {got[e, r, 4:10].tolist()} "
                     f"!= MatchDraws.next {seq}")
                break
    # env i of the batch == env i alone
    for e in range(N):
        one = unit_kernels.kit_draws_probe(bp, keys[e:e + 1], args[e:e + 1])
        if not np.array_equal(one[0], got[e]):
            fail(f"draws: env {e} alone != env {e} of the batch")
    # different keys give different words (the key reaches the kernel per env)
    if np.array_equal(got[0, :, 0], got[1, :, 0]):
        fail("draws: two envs' keys gave the same words (the key is not per env)")
    # out of the domain: flagged bad (Python raises / the cone's product)
    oob = np.array([[[int(Salt.SHOT_CONE), -2, 0, 0, 0, int(Salt.BEAM_CONE)],
                     [int(Salt.SHOT_CONE), M32, 0, 0, 0, int(Salt.BEAM_CONE)],
                     [int(Salt.SHOT_CONE), 3, -1, 0, 0, int(Salt.BEAM_CONE)],
                     [int(Salt.SHOT_CONE), 3, 1 << 32, 0, 0, int(Salt.BEAM_CONE)],
                     [int(Salt.SHOT_CONE), 3, 0, CONE_MAX + 1, 0, int(Salt.BEAM_CONE)],
                     [int(Salt.SHOT_CONE), 3, 0, -1, 0, int(Salt.BEAM_CONE)],
                     [int(Salt.SHOT_CONE), 3, 0, 0, 0, int(Salt.BEAM_CONE)]]],
                   dtype=np.int64)
    ob = unit_kernels.kit_draws_probe(bp, keys[:1], oob)
    if [int(x) for x in ob[0, :, 3]] != [1, 1, 1, 1, 1, 1, 0]:
        fail(f"draws: the out-of-domain flags {ob[0, :, 3].tolist()} != [1]*6 + [0]")
    badkey = keys[:1].copy()
    badkey[0, 2] = 1 << 32
    if int(unit_kernels.kit_draws_probe(bp, badkey, oob[:, -1:])[0, 0, 3]) != 1:
        fail("draws: a key word past 2^32 was not flagged")
    for k, v in stats.items():
        if v == 0:
            fail(f"draws: no {k} outcome (vacuous)")
    print(f"draws: {N} envs x {R} rows (+ 6 .next words each), {bad} differences; "
          f"{n_next} .next sequences equal MatchDraws.next; {stats}; batch == alone; "
          f"7 domain checks")


# ---------------------------------------------------------------------------
# PART SIGHT
# ---------------------------------------------------------------------------
SIGHT_MAPS = ("ship_demo", "playground", "my_ship", "test_level")


def _sight_sim(name, k):
    from simulation import Simulation
    lv = level_loader.load(name, str(_ROOT / "levels"))
    sim = Simulation(lv, seed=1, breach_physics=bp, enable_recorder=False)
    g = sim.gmap
    rng = np.random.default_rng(700 + k)
    air = np.argwhere(g.material == MAT_AIR)
    pick = air[rng.choice(len(air), size=min(len(air), 60 + 20 * k), replace=False)]
    for y, x in pick:                           # grilles, through the seam
        g.material[y, x] = MAT_GRILLE
        g.on_tile_changed(int(y), int(x))
    return sim


def _pairs(rng, h, w, R):
    p = np.zeros((R, 4), dtype=np.int64)
    for r in range(R):
        y1, x1 = int(rng.integers(-3, h + 3)), int(rng.integers(-3, w + 3))
        kind = r % 5
        if kind == 0:                           # long
            y2, x2 = int(rng.integers(-3, h + 3)), int(rng.integers(-3, w + 3))
        elif kind == 4:                         # start == end
            y2, x2 = y1, x1
        else:                                   # short (both answers often)
            s = (3, 8, 16)[kind - 1]
            y2, x2 = y1 + int(rng.integers(-s, s + 1)), x1 + int(rng.integers(-s, s + 1))
        p[r] = (y1, x1, y2, x2)
    return p


def part_sight():
    import cupy as cp
    sims = [_sight_sim(n, k) for k, n in enumerate(SIGHT_MAPS)]
    H = max(s.gmap.solid.shape[0] for s in sims)
    W = max(s.gmap.solid.shape[1] for s in sims)
    N = len(sims)
    solid = np.zeros((N, H, W), dtype=bool)
    screen = np.zeros((N, H, W), dtype=bool)
    for e, s in enumerate(sims):
        h, w = s.gmap.solid.shape
        solid[e, :h, :w] = s.gmap.solid
        screen[e, :h, :w] = s.gmap.sight_screen
    rng = np.random.default_rng(2907)
    pairs = np.stack([_pairs(rng, *s.gmap.solid.shape, SIGHT_ROWS) for s in sims])
    d_solid, d_screen = cp.asarray(solid), cp.asarray(screen)
    got = unit_kernels.kit_sight_probe(bp, d_solid, d_screen, pairs)
    tally = dict(sight_true=0, sight_false=0, phys_true=0, phys_false=0, differ=0)
    bad = 0
    for e, s in enumerate(sims):
        g = s.gmap
        for r in range(SIGHT_ROWS):
            y1, x1, y2, x2 = (int(v) for v in pairs[e, r])
            a = int(g.has_los(y1, x1, y2, x2))
            b = int(g.has_los(y1, x1, y2, x2, physical=True))
            if (int(got[e, r, 0]), int(got[e, r, 1]), int(got[e, r, 2])) != (a, b, 0):
                bad += 1
                if bad <= 3:
                    fail(f"sight env {e} ({SIGHT_MAPS[e]}) pair {(y1, x1, y2, x2)}: device "
                         f"{got[e, r].tolist()} != host ({a}, {b}, 0)")
            tally["sight_true" if a else "sight_false"] += 1
            tally["phys_true" if b else "phys_false"] += 1
            tally["differ"] += int(a != b)
    for e in range(N):
        one = unit_kernels.kit_sight_probe(bp, cp.ascontiguousarray(d_solid[e:e + 1]),
                                           cp.ascontiguousarray(d_screen[e:e + 1]),
                                           pairs[e:e + 1])
        if not np.array_equal(one[0], got[e]):
            fail(f"sight: env {e} alone != env {e} of the batch")
    far = np.array([[[0, 0, 1 << 31, 0]]], dtype=np.int64)
    if int(unit_kernels.kit_sight_probe(bp, cp.ascontiguousarray(d_solid[:1]),
                                        cp.ascontiguousarray(d_screen[:1]), far)[0, 0, 2]) != 1:
        fail("sight: an endpoint past COORD_MAX was not flagged")
    total = N * SIGHT_ROWS
    for k in ("sight_true", "sight_false", "phys_true", "phys_false"):
        if tally[k] < total // 20:
            fail(f"sight: {k} only {tally[k]} of {total} (not often)")
    if tally["differ"] == 0:
        fail("sight: the screen never made sight differ from physical (vacuous)")
    print(f"sight: {N} maps {SIGHT_MAPS} in one batch padded to ({H}, {W}), "
          f"{total} pairs x 2 forms, {bad} differences; {tally}; batch == alone")


# ---------------------------------------------------------------------------
# PART FALLOFF
# ---------------------------------------------------------------------------
def _falloff_args(rng, n):
    rows = []
    V_MAX = (1 << 31) - 1
    for _ in range(n):
        kind = int(rng.integers(0, 6))
        if kind == 0:                            # the deposit weights' shape
            v = 65536
            r = int(rng.integers(1, 40))
            r2 = (r * r) << 32
            d2 = int(rng.integers(0, r2 + 3))
        elif kind == 1:                          # blast damage, Q32 distances
            v = int(rng.integers(0, 1 << 30))
            r2 = int(rng.integers(1, 1 << 44))
            d2 = int(rng.integers(0, r2 + 2))
        elif kind == 2:                          # the range ends
            v = int(rng.choice([1, 2, V_MAX, V_MAX - 1]))
            r2 = int(rng.choice([1, 2, 1 << 62, int(rng.integers(1, 1 << 62))]))
            d2 = int(rng.choice([0, max(0, r2 - 1), r2 // 2, r2]))
        elif kind == 3:                          # anything in range
            v = int(rng.integers(0, V_MAX + 1))
            r2 = int(rng.integers(1, 1 << 62))
            d2 = int(rng.integers(0, r2))
        elif kind == 4:                          # trivial zeros (any operands)
            v = int(rng.integers(-(1 << 40), 1))
            r2 = int(rng.integers(-(1 << 40), 1 << 40))
            d2 = int(rng.integers(-(1 << 40), 1 << 40))
        else:                                    # outside the proved range
            v = int(rng.choice([V_MAX + 1, 1 << 40]))
            r2 = int(rng.integers(2, 1 << 40))
            d2 = int(rng.choice([-1, int(rng.integers(0, r2))]))
        rows.append((v, d2, r2))
    return np.array(rows, dtype=np.int64)


def _falloff_want(v, d2, r2):
    if v <= 0 or d2 >= r2:
        return (0, 0)
    if d2 < 0 or v > (1 << 31) - 1:
        return (-1, -1)
    return (unit_fixed.falloff_floor(v, d2, r2), unit_fixed.falloff_round(v, d2, r2))


def part_falloff():
    rng = np.random.default_rng(7029)
    args = _falloff_args(rng, 60000)
    got = unit_kernels.kit_falloff_probe(bp, args)
    bad = refused = nonzero = 0
    for i, (v, d2, r2) in enumerate(args.tolist()):
        want = _falloff_want(v, d2, r2)
        cpu = (int(bp.fp_falloff_floor(v, d2, r2)), int(bp.fp_falloff_round(v, d2, r2)))
        if (int(got[i, 0]), int(got[i, 1])) != want or cpu != want:
            bad += 1
            if bad <= 3:
                fail(f"falloff {(v, d2, r2)}: device {got[i].tolist()} cpu {cpu} != {want}")
        refused += int(want[0] == -1)
        nonzero += int(want[0] > 0)
    if refused == 0 or nonzero == 0:
        fail(f"falloff: refused {refused} / nonzero {nonzero} (vacuous)")
    print(f"falloff: {len(args)} (v, d2, r2) on the device and the CPU twin, {bad} "
          f"differences; {nonzero} non-zero, {refused} refused outside the proved range")


# ---------------------------------------------------------------------------
# The ignition scene (slot 9d)
# ---------------------------------------------------------------------------
FH, FW = 30, 40


class IgnitionScene:
    """Ignition on many ticks: a wood partition heated two tiles a tick on a
    schedule (the heat spreads on its own after that); a burning tile cooled
    by hand so it re-arms; a hot armed tile lit by hand (disarmed without a
    seed); a hot tile with no fuel left; a hot wood tile walled in by hull and
    space (no open neighbour); a hot wood tile beside a one-tile pocket pumped
    to near vacuum each tick (the fraction's floor: no O2); wood in the top
    hull beside space (a vacuum neighbour dropped); a hot wood tile on the grid
    edge (an out-of-grid neighbour). ``variant`` moves the
    partition, the pockets, the seed and the schedule; ``boundary`` / ``ts``
    the map. The fixtures write solid wood tiles' temperature (never a gas
    cell's) and pump gas only through ``GameMap.extract_gas_n``."""

    def __init__(self, variant=0, boundary="space", ts=1.0 / 3.0):
        from simulation import atmosphere_fixed as afx
        v = int(variant)
        self.v = v
        c1 = 12 + v % 5
        tm = np.full((FH, FW), SPACE, dtype=np.int32)
        tm[1:FH - 1, 1:FW - 1] = MAT_HULL
        tm[2:FH - 2, 2:FW - 2] = MAT_AIR
        tm[3:FH - 3, c1] = MAT_WOOD
        tm[1, 24:28] = MAT_WOOD                   # top hull wood beside space
        self.walled = (1, 30 + v % 3)             # walled in: hull below, space above
        tm[self.walled] = MAT_WOOD
        tm[2, self.walled[1]] = MAT_HULL
        self.edge = (0, 20 + v % 3)               # on the grid edge, in space
        tm[self.edge] = MAT_WOOD
        py, px = 6 + v % 4, 30 + v % 3            # the pocket and its wood
        tm[py - 1:py + 2, px - 1:px + 3] = MAT_HULL
        tm[py, px] = MAT_AIR
        tm[py, px + 1] = MAT_WOOD
        self.pocket, self.pocket_wood = (py, px), (py, px + 1)
        self.lit_by_hand = (FH - 6, c1 + 6)
        self.fuel_less = (FH - 8, c1 + 9)
        tm[self.lit_by_hand] = MAT_WOOD
        tm[self.fuel_less] = MAT_WOOD
        self.sim = _sim(_level(tm, f"p70a_ign_{v}", ts, boundary), seed=700 + v)
        _add_unit(self.sim, 4.0, 20.0)
        self.T_hot = int(afx.quantize_scalar(420.0 + 15.0 * (v % 3)))
        self.T_cold = 0
        rows = list(range(3, FH - 3))
        rng = np.random.default_rng(55 + v)
        rng.shuffle(rows)
        t0 = 2 + v % 3
        self.heat = {t0 + 3 * k: [(rows[2 * k], c1), (rows[2 * k + 1], c1)]
                     for k in range(len(rows) // 2)}
        self.c1 = c1

    def tick(self):
        sim, g = self.sim, self.sim.gmap
        t = sim.tick
        for tile in self.heat.get(t, ()):
            if int(g.material[tile]) == MAT_WOOD:
                g.temperature[tile] = self.T_hot
        # the standing fixtures, every tick
        for tile in (self.walled, self.pocket_wood, self.fuel_less, (1, 25), self.edge):
            if int(g.material[tile]) == MAT_WOOD:
                g.temperature[tile] = self.T_hot
        if int(g.material[self.fuel_less]) == MAT_WOOD:
            g.wall_hp[self.fuel_less] = 0
        g.extract_gas_n(self.pocket[0], self.pocket[1], 1 << 30)
        if int(g.material[self.lit_by_hand]) == MAT_WOOD:
            if t % 9 == 4:                           # lit by hand while hot and armed
                g.temperature[self.lit_by_hand] = self.T_hot
                g.fire[self.lit_by_hand] = 30000
            elif t % 9 == 6:                         # put out and cooled: it re-arms
                g.temperature[self.lit_by_hand] = self.T_cold
                g.fire[self.lit_by_hand] = 0
        if t % 11 == 7:                              # cool a disarmed burning tile
            dis = np.argwhere((g.material == MAT_WOOD) & ~g.ignition_armed)
            if len(dis):
                y, x = (int(q) for q in dis[(t // 11) % len(dis)])
                if (y, x) not in (self.walled, self.pocket_wood, self.lit_by_hand,
                                  self.fuel_less, (1, 25), self.edge):
                    g.temperature[y, x] = self.T_cold
        sim.step()


class ShipDriver:
    def __init__(self, seed=1):
        from simulation.match_session import MatchSession
        level = level_loader.load("ship_demo", str(_ROOT / "levels"))
        self.session = MatchSession(level, seed=seed, breach_physics=bp)
        self.sim = self.session.sim

    def tick(self):
        self.session.tick()

    def is_terminal(self):
        return self.session.is_terminal()


BRANCHES = ("rearm", "seed", "hot_unarmed", "hot_armed_lit", "no_o2_open",
            "no_open_neighbour", "floor_bound", "vacuum_dropped", "no_fuel",
            "cold_armed", "grid_edge")
def _census(fields, tables, e, counts):
    """Count the law's branches in env ``e``'s entry planes (numpy over the
    captured host fields), against its tables row."""
    g = {k: v[e] for k, v in fields.items()}
    thr = np.asarray(unit_tables.read(tables[e], "ign_temp_q"), dtype=np.int64)[
        g["material"].astype(np.int64)]
    t = g["temperature"].astype(np.int64)
    flam = g["flammable"]
    armed = g["ignition_armed"]
    fire = g["fire"]
    hot = flam & (g["wall_hp"] > 0) & (t >= thr)
    o2i, n2i = (int(unit_tables.read(tables[e], k)) for k in ("ign_gas_o2", "ign_gas_n2"))
    o2 = g["gas"][o2i].astype(np.int64)
    tot = o2 + g["gas"][n2i].astype(np.int64)
    opn = ~g["solid"] & ~g["is_vacuum"]
    h, w = t.shape
    so2 = np.zeros((h, w), np.int64)
    stot = np.zeros((h, w), np.int64)
    cnt = np.zeros((h, w), np.int64)
    vac = np.zeros((h, w), bool)
    for dy, dx in ((-1, 0), (1, 0), (0, 1), (0, -1)):
        ys0, ys1 = max(0, -dy), h - max(0, dy)
        xs0, xs1 = max(0, -dx), w - max(0, dx)
        sl = (slice(ys0 + dy, ys1 + dy), slice(xs0 + dx, xs1 + dx))
        o = opn[sl]
        so2[ys0:ys1, xs0:xs1] += np.where(o, o2[sl], 0)
        stot[ys0:ys1, xs0:xs1] += np.where(o, tot[sl], 0)
        cnt[ys0:ys1, xs0:xs1] += o
        vac[ys0:ys1, xs0:xs1] |= g["is_vacuum"][sl]
    fl = int(unit_tables.read(tables[e], "ign_x_n_floor_q"))
    x = (so2 << 16) // np.maximum(stot, fl)
    has = (cnt > 0) & (x > int(unit_tables.read(tables[e], "ign_x_ext_q")))
    ev = hot & armed & (fire == 0)
    c = dict(rearm=flam & (t < thr) & ~armed, seed=ev & has, hot_unarmed=hot & ~armed,
             hot_armed_lit=hot & armed & (fire > 0), no_o2_open=ev & (cnt > 0) & ~has,
             no_open_neighbour=ev & (cnt == 0), floor_bound=ev & (cnt > 0) & (stot < fl),
             vacuum_dropped=ev & vac, no_fuel=flam & (g["wall_hp"] <= 0) & (t >= thr),
             cold_armed=flam & (t < thr) & armed)
    edge = np.zeros((h, w), bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    c["grid_edge"] = ev & edge
    for k, m in c.items():
        counts[k] += int(m.sum())
    return int(c["seed"].sum())


def _shadow(name, drivers, ticks, need_branches=True, expect_none=False):
    N = len(drivers)
    counts = {b: 0 for b in BRANCHES}
    tally = dict(ticks=0, changed=0, ticks_seeding=0, empty_beside_full=0)
    t0 = time.perf_counter()
    with ShadowLockstep(drivers, SLOT_9D, bp=bp) as ls:
        for _ in range(ticks):
            if any(getattr(d, "is_terminal", lambda: False)() for d in drivers):
                break
            rep = ls.tick()
            tally["ticks"] += 1
            if not rep.equal:
                fail(f"{name}: {ls.divergences[-1]}")
                break
            inp = ls.last_inputs[SLOT_9D.label]
            pre = inp._host_fields
            post = {f: inp.fields[f].get() for f in ("fire", "ignition_armed")}
            changed = np.array([int(((pre["fire"][e] != post["fire"][e])
                                     | (pre["ignition_armed"][e]
                                        != post["ignition_armed"][e])).sum())
                                for e in range(N)], dtype=np.int64)
            if not np.array_equal(rep.coverage, changed):
                fail(f"{name}: tick {rep.tick}: coverage {rep.coverage.tolist()} != the "
                     f"cells changed {changed.tolist()}")
                break
            seeded = 0
            for e in range(N):
                seeded += _census(pre, ls.tables, e, counts)
            tally["changed"] += int(changed.sum())
            tally["ticks_seeding"] += int(seeded > 0)
            if (changed > 0).any() and (changed == 0).any():
                tally["empty_beside_full"] += 1
    tally["seconds"] = round(time.perf_counter() - t0)
    print(f"{name}: N {N}, {tally}; census {counts}")
    if need_branches:
        for b in BRANCHES:
            if counts[b] == 0:
                fail(f"{name}: branch {b!r} never taken")
        if tally["ticks_seeding"] < 5:
            fail(f"{name}: ignition on only {tally['ticks_seeding']} ticks")
    if expect_none and tally["changed"] != 0:
        fail(f"{name}: expected no ignition, the slot changed {tally['changed']} cells")
    return tally


def part_scenes():
    _shadow("scene ignition (space, 1/3 m)", [IgnitionScene(0)], SCENE_TICKS)
    _shadow("scene ignition (planetside, 1/2 m)", [IgnitionScene(1, "ambient", 0.5)],
            SCENE_TICKS, need_branches=False)
    t = _shadow("scene ship (bots, seed 1)", [ShipDriver(1)], SHIP_TICKS,
                need_branches=False, expect_none=True)
    print(f"scene ship: ignition changed {t['changed']} cells in {t['ticks']} ticks "
          f"(never: nothing on ship_demo gets hot)")


def _batch_envs(n):
    out = []
    for i in range(n):
        boundary = "ambient" if i % 2 else "space"
        ts = (1.0 / 3.0, 0.5, 1.0)[i % 3]
        out.append(IgnitionScene(i + 2, boundary, ts))
    return out


def part_batch():
    for n in BATCH_NS:
        t = _shadow(f"batch ignition N={n}", _batch_envs(n), BATCH_TICKS,
                    need_branches=False)
        if t["empty_beside_full"] == 0:
            fail(f"batch N={n}: no tick had an env that changed nothing beside one "
                 f"that changed cells")
        if t["ticks_seeding"] == 0:
            fail(f"batch N={n}: no env ignited (vacuous)")


def part_chained():
    from cuda_unit_damage_check import SLOT_9C, SLOT_9C2, SLOT_9C3
    from simulation import simulation as sim_mod
    twin, ref = IgnitionScene(4), IgnitionScene(4)
    slots = [SLOT_9C, SLOT_9C2, SLOT_9C3, SLOT_9D]
    t0 = time.perf_counter()
    # the host law must NOT run in the twin (the device replaced it) and must
    # run in the reference: count the original's calls per map underneath
    calls = {"twin": 0, "ref": 0}
    orig = sim_mod.apply_temperature_ignition

    def counted(gmap, *a, **kw):
        calls["twin" if gmap is twin.sim.gmap else "ref"] += 1
        return orig(gmap, *a, **kw)
    sim_mod.apply_temperature_ignition = counted
    try:
        _chain_run(twin, ref, slots, calls, t0)
    finally:
        sim_mod.apply_temperature_ignition = orig


def _chain_run(twin, ref, slots, calls, t0):
    from cuda_unit_damage_check import _same
    n = 0
    with ChainedSlots(twin, slots, bp=bp) as ch:
        for _ in range(CHAIN_TICKS):
            twin.tick()
            ref.tick()
            n += 1
            if not _same(twin.sim, ref.sim):
                fail(f"chained 9c..9d: the twin left the reference at tick {twin.sim.tick}")
                break
            for f in ("fire", "ignition_armed"):
                if not np.array_equal(getattr(twin.sim.gmap, f), getattr(ref.sim.gmap, f)):
                    fail(f"chained 9c..9d: plane {f} differs at tick {twin.sim.tick}")
                    break
    acted = sum(1 for covs in ch.coverage if int(covs[3][0]) > 0)
    print(f"chained 9c + 9c2 + 9c3 + 9d (ignition scene): {n} ticks, the twin equal to "
          f"the reference every tick ({ch.runs} chain runs; [9d] changed cells on "
          f"{acted} ticks; {time.perf_counter() - t0:.0f} s)")
    if ch.runs != n:
        fail(f"chained: the chain ran {ch.runs} times in {n} ticks")
    if acted < 5:
        fail(f"chained: [9d] acted on only {acted} ticks (vacuous)")
    if calls["twin"] != 0 or calls["ref"] != n:
        fail(f"chained: the host law ran {calls['twin']} times in the twin (want 0) and "
             f"{calls['ref']} in the reference (want {n})")


# ---------------------------------------------------------------------------
# PART LAW: slot 9d's kernel against the Python law on synthetic planes
# ---------------------------------------------------------------------------
class _Gases:
    def __init__(self, names):
        self.name_to_id = {n: i for i, n in enumerate(names)}


class _Mats:
    def __init__(self, thr):
        self.ignition_temp_q16 = thr


class _LawMap:
    """A GameMap stand-in holding exactly what apply_temperature_ignition
    reads and writes (the test_continuous_o2_law precedent)."""

    def __init__(self, planes, thr, gases):
        for k, v in planes.items():
            setattr(self, k, v)
        self.materials = _Mats(thr)
        self.gases = gases


def _law_planes(rng, h, w, n_gases, thr, o2i, n2i, x_ext_q, floor_q):
    """Random planes straddling every edge of the law: temperatures at the
    threshold and one either side, fuel -1 / 0 / 1, armed / unarmed, lit /
    unlit, gas in VACUUM tiles, open-neighbour sums below the floor, O2
    fractions one count either side of X_ext."""
    n_mat = len(thr)
    mat = rng.integers(0, n_mat, size=(h, w)).astype(np.int8)
    t_thr = thr[mat.astype(np.int64)]
    dt = rng.choice([-1, 0, 1, -5000, 5000, -(1 << 20), 1 << 20], size=(h, w))
    temp = np.clip(t_thr + dt, -(1 << 31), (1 << 31) - 1).astype(np.int32)
    flam = rng.random((h, w)) < 0.6
    wall_hp = rng.choice([-1, 0, 1, 70000], size=(h, w)).astype(np.int32)
    solid = rng.random((h, w)) < 0.25
    vac = (~solid) & (rng.random((h, w)) < 0.15)
    fire = np.where(rng.random((h, w)) < 0.3, rng.integers(1, 90000, size=(h, w)),
                    0).astype(np.int32)
    armed = rng.random((h, w)) < 0.7
    gas = rng.integers(0, 4000, size=(n_gases, h, w)).astype(np.int32)
    tot = rng.choice([0, 100, floor_q - 1, floor_q, floor_q + 1, 30000, 70000], size=(h, w))
    # an O2 share giving X one count either side of X_ext (or anywhere)
    share = np.where(rng.random((h, w)) < 0.5, (x_ext_q + rng.integers(-2, 3, size=(h, w))),
                     rng.integers(0, 65537, size=(h, w)))
    o2 = (tot * share) // 65536
    gas[o2i] = o2.astype(np.int32)
    gas[n2i] = (tot - o2).astype(np.int32)
    return dict(temperature=temp, material=mat, flammable=flam, wall_hp=wall_hp,
                solid=solid, is_vacuum=vac, gas=gas, fire=fire, ignition_armed=armed)


def part_law():
    """Slot 9d's kernel on random synthetic planes (8 envs x 40 trials of 24 x
    32, every edge of the law straddled) against apply_temperature_ignition
    run on a stand-in map holding the same planes; the env's tables are the
    ignition scene's (the real dials)."""
    import cupy as cp
    from config import CFG
    from simulation import combat
    sim = IgnitionScene(0).sim
    tab = unit_tables.values(sim)
    thr = np.asarray(unit_tables.read(tab, "ign_temp_q"), dtype=np.int64)
    o2i, n2i = (int(unit_tables.read(tab, k)) for k in ("ign_gas_o2", "ign_gas_n2"))
    x_ext_q = int(unit_tables.read(tab, "ign_x_ext_q"))
    floor_q = int(unit_tables.read(tab, "ign_x_n_floor_q"))
    names = [None] * sim.gmap.gas.shape[0]
    for k, i in sim.gmap.gases.name_to_id.items():
        names[i] = k
    gases = _Gases(names)
    o2_frac_ext, seed = combat.ignition_params(CFG)
    rng = np.random.default_rng(9070)
    N, h, w, G = 8, 24, 32, sim.gmap.gas.shape[0]
    st = unit_store.DeviceStore(N, 1, bp=bp, tables=np.tile(tab, (N, 1)))
    trials = int(os.environ.get("UNIT_KIT_LAW_TRIALS", "40"))
    seen = dict(changed=0, seeded=0, rearmed=0, disarmed=0, at_threshold=0,
                vac_with_gas=0)
    bad = 0
    for trial in range(trials):
        envs = [_law_planes(rng, h, w, G, thr, o2i, n2i, x_ext_q, floor_q) for _ in range(N)]
        dev = {k: cp.asarray(np.ascontiguousarray(np.stack([e[k] for e in envs])))
               for k, _ in unit_kernels.IGNITION_FIELDS}
        cov = unit_kernels.ignition(st, dev)
        got_f, got_a = dev["fire"].get(), dev["ignition_armed"].get()
        for e, pl in enumerate(envs):
            m = _LawMap({k: v.copy() for k, v in pl.items()}, thr, gases)
            combat.apply_temperature_ignition(m, o2_frac_ext, seed)
            ch = (m.fire != pl["fire"]) | (m.ignition_armed != pl["ignition_armed"])
            if not (np.array_equal(got_f[e], m.fire)
                    and np.array_equal(got_a[e], m.ignition_armed)) or int(cov[e]) != int(ch.sum()):
                bad += 1
                if bad <= 3:
                    d = np.argwhere((got_f[e] != m.fire) | (got_a[e] != m.ignition_armed))
                    fail(f"law trial {trial} env {e}: device != python at {d[:3].tolist()} "
                         f"(coverage {int(cov[e])} vs {int(ch.sum())})")
                continue
            seen["changed"] += int(ch.sum())
            seen["seeded"] += int(((m.fire != pl["fire"])).sum())
            seen["rearmed"] += int((m.ignition_armed & ~pl["ignition_armed"]).sum())
            seen["disarmed"] += int((~m.ignition_armed & pl["ignition_armed"]).sum())
            seen["at_threshold"] += int((pl["flammable"]
                                         & (pl["temperature"].astype(np.int64)
                                            == thr[pl["material"].astype(np.int64)])).sum())
            seen["vac_with_gas"] += int((pl["is_vacuum"] & (pl["gas"][o2i] > 0)).sum())
    for k, v in seen.items():
        if v == 0:
            fail(f"law: no {k} (vacuous)")
    print(f"law: {trials} trials x {N} envs of {h} x {w} random planes (every edge "
          f"straddled), {bad} differences; {seen}")


# ---------------------------------------------------------------------------
# main / bench
# ---------------------------------------------------------------------------
PARTS = ("tables", "draws", "sight", "falloff", "law", "scenes", "batch", "chained")


def main(parts=None) -> int:
    if not getattr(bp, "HAS_CUDA", False):
        print("UNIT_KIT_RESULT: FAIL (not the CUDA build)")
        return 1
    parts = parts or os.environ.get("UNIT_KIT_PARTS", ",".join(PARTS)).split(",")
    t0 = time.perf_counter()
    for p in parts:
        try:
            {"tables": part_tables, "draws": part_draws, "sight": part_sight,
             "falloff": part_falloff, "law": part_law, "scenes": part_scenes, "batch": part_batch,
             "chained": part_chained}[p]()
        except Exception as ex:          # a part's crash fails it, the rest run
            import traceback
            traceback.print_exc()
            fail(f"part {p} raised {type(ex).__name__}: {ex}")
    print(f"total {time.perf_counter() - t0:.0f} s")
    ok = not FAILS
    print("UNIT_KIT_RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def bench(reps: int = 100) -> int:
    """The TIMING table of slot 9d: the Python face (launch + the counts' D2H
    and sync) on ship_demo's planes (70 x 100) and the ignition scene's (30 x
    40, mid-burn) at N = 1, 8, 64; medians over ``reps``."""
    import cupy as cp
    scene = IgnitionScene(0)
    for _ in range(30):
        scene.tick()
    sims = {"ship_demo 70x100": ShipDriver().sim, "ignition 30x40": scene.sim}
    print("TIMING (median ms): slot 9d = launch + counts D2H + sync")
    print(f"{'planes':>18} {'N':>4} {'ms':>9}")
    for label, sim in sims.items():
        g = sim.gmap
        for n in (1, 8, 64):
            st = unit_store.DeviceStore(n, 1, bp=bp,
                                        tables=np.tile(unit_tables.values(sim), (n, 1)))
            fields = {k: cp.asarray(np.ascontiguousarray(np.stack([getattr(g, k)] * n)))
                      for k, _ in unit_kernels.IGNITION_FIELDS}
            ts = []
            for _ in range(reps):
                cp.cuda.runtime.deviceSynchronize()
                t0 = time.perf_counter()
                unit_kernels.ignition(st, fields)
                ts.append(time.perf_counter() - t0)
            print(f"{label:>18} {n:>4} {1e3 * float(np.median(ts)):>9.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
