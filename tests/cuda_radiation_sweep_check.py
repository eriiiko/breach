"""P4 — THE RADIATION SWEEP'S CUDA TWIN, held to the CPU sweep at tol 0
(ray-engine-v2, design v3 §2.9 gate 7). Runs INSIDE the GPU subprocess; the
pytest wrapper is tests/test_cuda_radiation_sweep.py.

PROPERTY: `breach_cuda::radiation_sweep_step` (cpp/src/cuda_radiation_sweep.cu)
and its launch core compute the SAME integers as `RadiationSweep::run` on every
input the engine can hand them — all four output planes (rad_net, rad_flux,
rad_amb, rad_fluence), the Fleck plane, and the stream telemetry (min_stream,
max_stream). Gate 0 holds the CPU sweep to the integer reference
(`sweep_ref_q.py`) bit for bit, so this chains the GPU to the reference too.

BREAKS IF: any term of design §2.3 is transcribed differently in the .cu (the
remainder split, the gather offsets, the ring's books, the body's ambient
re-emission, the Q24 Fleck product, the pre-pass's L_q chain, the ambient's
per-cell derivation); a wavefront launch stops being a topological order of
some ordinate's dependency DAG (a cell gathers an upwind cell before it is
written); the per-cell books stop being order-free integer sums; or the
(N, h, w) indexing mixes one env's cells, scalars or counters into another's.

CPU reference = the bound `RadiationSweep` (the engine's own sweep, gate 0's
subject). GPU = `bp.cuda_radiation_sweep_run` (the per-call path the engine
dispatches) and `bp.cuda_radiation_sweep_resident` (the (N, h, w) launch core).
Both live in the one CUDA-build module, so every comparison is one process,
one binary, one scene.

  PART 1  gate 0's scene matrix — 4 seeds x S16/S12 x leak off/on x shear/step
          x three ambients (uniform, a cold half-plane, a random multi-level
          plane), and the undamped (fleck off) configuration.
  PART 1L #78: the same matrix on the LIVE table in its FINE currency (2^11
          per heat count), thin rows mixed in so the Fleck damping engages at
          the live scale and the pre-pass's conversion is compared, with and
          without the smoke term; and PART 8 (the launch core) runs on both
          tables -- the device table's currency is a launch argument.
  PART 2  the DERIVED ambient (derive_ambient's twin, the live path's door):
          random vacuum masks at five vacuum levels, and THE VACUUM RING — the
          grid's border as space under a cold sky, so the virtual ambient ring
          returns a non-E0 level at every boundary read.
  PART 3  thin rows: `heat_inv_shift` bands reaching -16 (M1's exponents).
  PART 4  a HOT FIRE TILE on the LIVE calibration (rad_scale_derived, the
          shipped k_leak) — a burning crate at the 1263-game plateau and one at
          the table top — and the same room on the fitted table, where the
          Fleck damping engages on the fire tile.
  PART 5  dyn_heat_atten_q STAMPS: bodies on air and on solids, including a
          body standing beside the fire and one in its shadow.
  PART 6  shapes: degenerate strips and squares, odd sizes, the live 128x256
          and the full 256x512 (wavefronts longer than one block).
  PART 7  overwrite + ingress: garbage-filled outputs come back exact; every
          scene the CPU rejects the GPU rejects too (ValueError), and leaves
          the caller's planes untouched.
  PART 8  the launch core at N = 3 (RL-batch habits §A): three different envs
          with different per-env scalars in ONE launch sequence, each equal to
          its own CPU run; then one illegal env, masked alone.
  PART 9  the live conductor: Simulation.step on the playground level with a
          burning 1263-game wood tile, ONLY the radiation backend flipping —
          every array the GameMap holds (the four sweep planes included), the
          temperature solver's counters and the engine's own sweep telemetry,
          per tick, tol 0; the same under a cold sky; and the resident tick
          with every backend on but combustion (whose own twin diverges on
          this scene — a P4 finding, see part9_live).
  PART 10 THE SMOKE TERM (P5a, design v3 §6.3): gate 0's smoke matrix (random
          gas planes, heat_absorb tables with zero, thin, saturating and
          maximal coefficients, bulk counts below / at / above the N_EPS
          floor, negative densities); the one scene carrying every feature;
          a burning tile in a smoke cloud on the LIVE calibration; the
          engine's 7-plane gas layout on full-size grids; the launch core at
          N = 3 with per-env gas planes and a shared table (and an illegal
          table flagged in every env); the gas ingress rejections; and the
          live conductor with a TEST-FIXTURE smoke coefficient, only the
          radiation backend flipping. Parts 1-9 carry no smoke (the dormant
          path of the smoke term); this is the live one. SINCE P5c the shipped
          smoke absorbs (config.toml [gases.smoke] heat_absorb, derived), so the
          live worlds of part 9 run the smoke term wherever combustion made smoke.
          SINCE P5b (design v3 §2.8 / §6.3) the Fleck plane every one of
          those compares carries the GAS ARM too -- damped gas cells asserted
          present in the smoke matrix, the features scene, the full-size
          grids and the batch -- and three legs are the arm's own: P10h, one
          scene carrying every case of it (hot smoke at ambient density, at
          ambient pressure, at / below the n_floor density, at the N_EPS
          edge, a mix, a body in it; beside the cells it must leave at 2^24)
          on the LIVE table and the resolving one; the currency's ingress
          (refused alike on both backends, planes untouched); and P10i, the
          live conductor with HOT smoke (seeded through the gas-energy seam)
          and a fixture coefficient, where the engine's own gas_capacity_q
          prices the damped gas cells on both backends.

NON-VACUITY is asserted per part: planes non-zero, Fleck factors below 2^24
where a part claims damping, bodies present, the GPU dispatch counter moving.

Prints ``RS_RESULT: PASS``/``FAIL`` and exits 0/1.
"""
from __future__ import annotations

import random
import sys

import numpy as np

# Import the CUDA build FIRST so it is the cached `breach_physics` (the
# harness below inserts the CPU build dir on sys.path, which must not win).
import breach_physics as bp

from _radiation_sweep_harness import (  # noqa: E402
    F_ONE, ONE, R, T_AMB_Q, TRANSPORTS, as_i32, engine_currency, gas_arrays,
    gas_fleck_scene, live_table, random_gas, random_scene, reference_table,
    smoke_feature_scene)

Q = R.quant
K_LEAK = Q(0.10)
GARBAGE = -0x5A5A5A5A5A5A5A5            # what the GPU's output planes start as
F_GARBAGE = -7                           # ... and its Fleck out-plane
PLANES = ("rad_net", "rad_flux", "rad_amb", "rad_fluence")
_FAILS: list[str] = []


def _fail(msg: str) -> None:
    _FAILS.append(msg)
    print("  FAIL: " + msg)


# ---------------------------------------------------------------------------
# the two drivers — one scene, both backends
# ---------------------------------------------------------------------------
class Scene:
    """One sweep input, as the contiguous numpy dtypes both bindings require."""

    def __init__(self, a, d, T, his, ts):
        self.a = as_i32(a)
        self.d = as_i32(d)
        self.T = as_i32(T)
        self.his = as_i32(his)
        self.ts = np.ascontiguousarray(np.asarray(ts) != 0)
        self.h, self.w = self.a.shape


def _currency_kw(gas, currency):
    """The gas arm's currency (P5b) as the two engine keywords -- only with the
    gas group; `currency` is (n_floor_q, recip_cv), the shipped one by default."""
    if gas is None:
        return {}
    nf, rcv = currency if currency is not None else engine_currency()
    return dict(n_floor_q=int(nf), recip_cv=int(rcv))


def cpu_run(sc, table, *, transport="shear", n_ord=16, k_q=0, t_amb=int(T_AMB_Q),
            amb=None, is_vacuum=None, vac_level=-1, fleck=True, gas=None,
            currency=None):
    """RadiationSweep.run — with derive_ambient() first when the ambient is to be
    DERIVED (amb None, is_vacuum given): exactly what step_tail does. `gas` is
    the P5a smoke group, a (gas, heat_absorb_q16, n_bulk) tuple of int32 arrays;
    `currency` its gas arm's (n_floor_q, recip_cv), P5b (shipped by default)."""
    sw = bp.RadiationSweep()
    amb_arr = None
    if amb is not None:
        amb_arr = np.ascontiguousarray(np.broadcast_to(
            np.asarray(amb, dtype=np.int64), (sc.h, sc.w)))
    elif is_vacuum is not None:
        amb_arr = sw.derive_ambient(np.ascontiguousarray(is_vacuum), table,
                                    int(vac_level))
    out = [np.zeros((sc.h, sc.w), dtype=np.int64) for _ in range(4)]
    g_a, hq_a, nb_a = gas if gas is not None else (None, None, None)
    sw.run(sc.T, sc.a, sc.d, sc.his, sc.ts, table, amb_arr, int(t_amb), int(k_q),
           TRANSPORTS[transport], int(n_ord), *out, fleck_enabled=bool(fleck),
           gas=g_a, heat_absorb_q16=hq_a, n_bulk=nb_a, **_currency_kw(gas, currency))
    return out, np.asarray(sw.fleck_plane(), dtype=np.int32), \
        int(sw.min_stream), int(sw.max_stream)


def gpu_run(sc, table, *, transport="shear", n_ord=16, k_q=0, t_amb=int(T_AMB_Q),
            amb=None, is_vacuum=None, vac_level=-1, fleck=True, gas=None,
            currency=None):
    """cuda_radiation_sweep_run — the per-call path. The outputs start as
    GARBAGE, so every comparison is also the overwrite property."""
    amb_arr = None
    if amb is not None:
        amb_arr = np.ascontiguousarray(np.broadcast_to(
            np.asarray(amb, dtype=np.int64), (sc.h, sc.w)))
    vac = None if is_vacuum is None else np.ascontiguousarray(is_vacuum)
    out = [np.full((sc.h, sc.w), GARBAGE, dtype=np.int64) for _ in range(4)]
    fo = np.full((sc.h, sc.w), F_GARBAGE, dtype=np.int32)
    g_a, hq_a, nb_a = gas if gas is not None else (None, None, None)
    mn, mx, launches = bp.cuda_radiation_sweep_run(
        sc.T, sc.a, sc.d, sc.his, sc.ts, table, amb_arr, int(t_amb), int(k_q),
        TRANSPORTS[transport], int(n_ord), *out, fleck_enabled=bool(fleck),
        is_vacuum=vac, vac_level=int(vac_level), fleck_out=fo,
        gas=g_a, heat_absorb_q16=hq_a, n_bulk=nb_a, **_currency_kw(gas, currency))
    want = bp.cuda_radiation_sweep_launch_count(TRANSPORTS[transport], int(n_ord),
                                                sc.h, sc.w)
    if launches != want:
        _fail(f"launch count {launches} != the shape's {want}")
    return out, fo, int(mn), int(mx)


def same(tag, cpu, gpu) -> bool:
    """Tol 0 on the four planes, the Fleck plane and both telemetry values."""
    ok = True
    (c_out, c_f, c_mn, c_mx), (g_out, g_f, g_mn, g_mx) = cpu, gpu
    for name, c, g in zip(PLANES, c_out, g_out):
        if not np.array_equal(c, g):
            bad = np.argwhere(c != g)
            y, x = bad[0]
            _fail(f"{tag}: {name} differs at {len(bad)} cells; first ({y},{x}) "
                  f"cpu={int(c[y, x])} gpu={int(g[y, x])}")
            ok = False
    if not np.array_equal(c_f, g_f):
        bad = np.argwhere(c_f != g_f)
        _fail(f"{tag}: the Fleck plane differs at {len(bad)} cells")
        ok = False
    if (c_mn, c_mx) != (g_mn, g_mx):
        _fail(f"{tag}: stream telemetry cpu=({c_mn}, {c_mx}) gpu=({g_mn}, {g_mx})")
        ok = False
    return ok


def both(tag, sc, table, **kw):
    """Run the scene on both backends; assert tol 0; return the CPU result."""
    c = cpu_run(sc, table, **kw)
    g = gpu_run(sc, table, **kw)
    same(tag, c, g)
    return c


def nonzero_everywhere(tag, res) -> None:
    out, f, _mn, _mx = res
    for name, p in zip(PLANES, out):
        if not np.any(p != 0):
            _fail(f"{tag}: {name} is identically zero — the comparison is vacuous there")


# ---------------------------------------------------------------------------
# PART 1 — gate 0's matrix
# ---------------------------------------------------------------------------
def _ambient(kind, rng, h, w):
    """test_radiation_sweep_reference.py's AMBIENT axis, verbatim."""
    if kind == "uniform":
        return None
    e0 = R.E0
    if kind == "cold-half":
        return [[0 if x < w // 2 else e0 for x in range(w)] for _ in range(h)]
    return [[rng.choice([0, 1, e0 // 4, e0 // 2, e0 - 1, e0]) for _ in range(w)]
            for _ in range(h)]


def part1_gate0_matrix(table) -> None:
    print("PART 1 — gate 0's scene matrix, GPU vs CPU, tol 0")
    n = 0
    damped = bodies = 0
    for seed, h, w in ((1, 7, 9), (2, 9, 11), (3, 12, 8), (4, 5, 5)):
        for n_ord in (16, 12):
            for k_q in (0, K_LEAK):
                for transport in ("shear", "step"):
                    for kind in ("uniform", "cold-half", "random"):
                        rng = random.Random(20260916 + seed)
                        a, d, T, his, ts = random_scene(rng, h, w)
                        amb = _ambient(kind, rng, h, w)
                        sc = Scene(a, d, T, his, ts)
                        tag = f"P1 seed={seed} S{n_ord} k={k_q} {transport} {kind}"
                        res = both(tag, sc, table, transport=transport,
                                   n_ord=n_ord, k_q=k_q, amb=amb)
                        nonzero_everywhere(tag, res)
                        damped += int(np.any(res[1] < F_ONE))
                        bodies += int(np.any(sc.d > sc.a))
                        n += 1
                        # the undamped configuration (the reference's
                        # f_plane=None), once per scene/ordinate set
                        if k_q == 0 and kind == "uniform":
                            both(tag + " fleck=off", sc, table, transport=transport,
                                 n_ord=n_ord, k_q=k_q, fleck=False)
                            n += 1
    if damped == 0:
        _fail("P1: no configuration had a Fleck factor below 2^24 — the damped "
              "product was never compared")
    if bodies == 0:
        _fail("P1: no configuration carried a body share")
    print(f"  {n} configurations; {damped} with Fleck damping engaged, "
          f"{bodies} with bodies")


def part1_live_matrix() -> None:
    """#78 (brief 5.6): CPU == CUDA at tol 0 on the LIVE table in its fine
    currency -- the table the game runs."""
    print("PART 1L — #78: gate 0's matrix on the FINE live table, GPU vs CPU, tol 0")
    table = live_table()
    n = damped = smoky = 0
    for seed, h, w in ((21, 7, 9), (22, 9, 11), (23, 12, 8)):
        for n_ord in (16, 12):
            for k_q in (0, K_LEAK):
                for transport in ("shear", "step"):
                    for kind in ("uniform", "random"):
                        for smoke in (False, True):
                            rng = random.Random(20260925 + seed)
                            a, d, T, _his, ts = random_scene(rng, h, w)
                            his = [[rng.choice([-3, -2, 3, 5]) for _ in range(w)]
                                   for _ in range(h)]
                            amb = None
                            if kind == "random":
                                e0 = int(R.E_LIVE[0])
                                amb = [[rng.choice([0, 1, e0 // 4, e0 // 2, e0 - 1, e0])
                                        for _ in range(w)] for _ in range(h)]
                            gas = None
                            if smoke:
                                gas = gas_arrays(*random_gas(rng, h, w))
                                smoky += 1
                            sc = Scene(a, d, T, his, ts)
                            tag = (f"P1L seed={seed} S{n_ord} k={k_q} {transport} "
                                   f"{kind} smoke={smoke}")
                            res = both(tag, sc, table, transport=transport, n_ord=n_ord,
                                       k_q=k_q, amb=amb, gas=gas)
                            nonzero_everywhere(tag, res)
                            damped += int(np.any(res[1] < F_ONE))
                            n += 1
    if damped == 0:
        _fail("P1L: no live-table configuration had a Fleck factor below 2^24 -- "
              "the fine currency's conversion in the pre-pass was never compared")
    print(f"  {n} configurations on the fine live table ({smoky} with smoke); "
          f"{damped} with Fleck damping engaged")


# ---------------------------------------------------------------------------
# PART 2 — the derived ambient, and the vacuum ring
# ---------------------------------------------------------------------------
def _vacuum_ring_scene(h=14, w=18):
    """A hull in space: the grid's outer ring is VACUUM (space around the ship),
    the next ring a hull wall (a = 0.91 thermal solid), the interior air with a
    burning crate and two cold ones."""
    a = np.zeros((h, w), dtype=np.int64)
    ts = np.zeros((h, w), dtype=bool)
    T = np.zeros((h, w), dtype=np.int64)
    vac = np.zeros((h, w), dtype=bool)
    vac[0, :] = vac[-1, :] = vac[:, 0] = vac[:, -1] = True
    hull = np.zeros((h, w), dtype=bool)
    hull[1, 1:-1] = hull[-2, 1:-1] = hull[1:-1, 1] = hull[1:-1, -2] = True
    a[hull] = Q(0.91)
    for (y, x, t) in ((6, 7, 1263 << 16), (4, 12, 0), (9, 5, 20 << 16)):
        a[y, x] = Q(0.9)
        T[y, x] = t
    ts[a > 0] = True
    his = np.full((h, w), 3, dtype=np.int64)
    return Scene(a, a.copy(), T, his, ts), vac


def part2_derived_ambient(table) -> None:
    print("PART 2 — the DERIVED ambient (derive_ambient's twin) and the vacuum ring")
    e0 = int(np.asarray(table.table())[0])
    levels = (-1, 0, e0 // 3, e0 - 1, e0)
    n = 0
    for seed in (11, 12, 13):
        rng = random.Random(20260924 + seed)
        h, w = 9 + seed % 3, 12 - seed % 4
        a, d, T, his, ts = random_scene(rng, h, w)
        sc = Scene(a, d, T, his, ts)
        vac = np.asarray([[rng.random() < 0.35 for _ in range(w)] for _ in range(h)])
        for level in levels:
            for transport in ("shear", "step"):
                tag = f"P2 seed={seed} vac_level={level} {transport}"
                both(tag, sc, table, transport=transport, k_q=K_LEAK,
                     is_vacuum=vac, vac_level=level)
                n += 1
    # THE VACUUM RING under a cold sky, against its own R4 control.
    sc, vac = _vacuum_ring_scene()
    for transport in ("shear", "step"):
        for n_ord in (16, 12):
            tag = f"P2 vacuum-ring {transport} S{n_ord}"
            cold = both(tag + " cold", sc, table, transport=transport, n_ord=n_ord,
                        k_q=K_LEAK, is_vacuum=vac, vac_level=0)
            room = both(tag + " R4", sc, table, transport=transport, n_ord=n_ord,
                        k_q=K_LEAK, is_vacuum=vac, vac_level=-1)
            n += 2
            # NON-VACUITY: the cold sky is visible in the books — the ring cells'
            # ambient ledger and the hull's material ledger both move.
            if np.array_equal(cold[0][2], room[0][2]) or \
                    np.array_equal(cold[0][0], room[0][0]):
                _fail(f"{tag}: the cold sky changed nothing — the ring's ambient "
                      "is not being read per cell (vacuous)")
            # and the no-mask door is the room-temperature answer
            flat = both(tag + " no-mask", sc, table, transport=transport,
                        n_ord=n_ord, k_q=K_LEAK)
            n += 1
            same(tag + " R4 == uniform door", room, flat)
    print(f"  {n} configurations (5 vacuum levels x 3 masks x 2 transports; the "
          f"ring cold/R4/uniform x S16/S12 x 2 transports)")


# ---------------------------------------------------------------------------
# PART 3 — thin rows (negative exponents)
# ---------------------------------------------------------------------------
def _thin_scene(rng, h, w, his_lo, his_hi):
    """test_m1_extra_consumer_sites.py's vocabulary: gate 0's scene with the
    `heat_inv_shift` plane drawn from a band that includes NEGATIVE exponents."""
    ts = [[1 if rng.random() < 0.5 else 0 for _ in range(w)] for _ in range(h)]
    a = [[(rng.choice([0, Q(0.3), Q(0.37), Q(0.5), Q(0.91), ONE]) if ts[y][x] else 0)
          for x in range(w)] for y in range(h)]
    d = [[min(ONE, a[y][x] + rng.choice([0, 0, 0, Q(0.5), ONE - a[y][x]]))
          for x in range(w)] for y in range(h)]
    T = [[rng.choice([-(200 << 16), 0, 0, 300 << 16, 1263 << 16, 5000 << 16,
                      15999 << 16, 16000 << 16, 20000 << 16, (32767 << 16) + 65535])
          for _ in range(w)] for _ in range(h)]
    his = [[rng.randint(his_lo, his_hi) for _ in range(w)] for _ in range(h)]
    return a, d, T, his, ts


def part3_thin_rows(table) -> None:
    print("PART 3 — thin rows: negative heat_inv_shift")
    n = 0
    neg_damped = 0
    for seed, h, w in ((1, 7, 9), (2, 9, 11), (3, 5, 5)):
        for band in ((-16, -1), (-6, 6)):
            for transport in ("shear", "step"):
                rng = random.Random(seed)
                sc = Scene(*_thin_scene(rng, h, w, *band))
                tag = f"P3 seed={seed} his={band} {transport}"
                res = both(tag, sc, table, transport=transport, k_q=K_LEAK)
                n += 1
                neg_damped += int(np.any((res[1] < F_ONE) & (sc.his < 0) & sc.ts))
    if neg_damped == 0:
        _fail("P3: no thin (negative-exponent) thermal solid was ever damped — "
              "the signed-shift branch of L_q was not compared")
    print(f"  {n} configurations; {neg_damped} with a damped negative-exponent cell")


# ---------------------------------------------------------------------------
# PART 4 — a hot fire tile, live calibration and fitted
# ---------------------------------------------------------------------------
def _live_table_and_leak():
    from config import CFG
    import temperature_scale
    ts = temperature_scale.load(CFG)
    tbl = bp.EmissiveTable()
    tbl.rad_scale = float(CFG.physics.radiation.rad_scale_derived)
    tbl.kelvin_ambient = float(ts.kelvin_ambient)
    tbl.k_temp_to_kelvin = float(ts.k_temp_to_kelvin)
    tbl.bake()
    from simulation import optics_fixed
    k_q = int(optics_fixed.quantize_scalar(float(CFG.physics.radiation.k_leak)))
    t_amb = int(round(float(ts.kelvin_ambient) * ONE))
    return tbl, k_q, t_amb


def _fire_room(T_fire, h=20, w=26):
    """A walled room (a = 0.9 walls), a burning crate at T_fire in the middle,
    two cold crates at 2 and 4 tiles, open air between."""
    a = np.zeros((h, w), dtype=np.int64)
    a[0, :] = a[-1, :] = a[:, 0] = a[:, -1] = Q(0.9)
    fire = (h // 2, w // 2)
    for (y, x) in (fire, (h // 2, w // 2 + 2), (h // 2 + 4, w // 2)):
        a[y, x] = Q(0.9)
    T = np.zeros((h, w), dtype=np.int64)
    T[fire] = T_fire
    ts = a > 0
    his = np.full((h, w), 3, dtype=np.int64)
    return Scene(a, a.copy(), T, his, ts), fire


def part4_fire_tile(table_fitted) -> None:
    print("PART 4 — a hot fire tile: the LIVE calibration, and the fitted table")
    live, k_live, t_amb_live = _live_table_and_leak()
    n = 0
    for T_fire in (1263 << 16, 15999 << 16):
        sc, fire = _fire_room(T_fire)
        for transport in ("shear", "step"):
            tag = f"P4 live T={T_fire >> 16} {transport}"
            res = both(tag, sc, live, transport=transport, k_q=k_live,
                       t_amb=t_amb_live)
            n += 1
            rn = res[0][0]
            if not (rn[fire] < 0 and int(rn.max()) > 0):
                _fail(f"{tag}: the fire tile does not radiate (rad_net={int(rn[fire])}) "
                      f"or nothing absorbs (max={int(rn.max())}) — vacuous")
    # the fitted table: the Fleck damping engages ON the fire tile
    engaged = 0
    for T_fire in (5000 << 16, 15999 << 16):
        sc, fire = _fire_room(T_fire)
        for transport in ("shear", "step"):
            tag = f"P4 fitted T={T_fire >> 16} {transport}"
            res = both(tag, sc, table_fitted, transport=transport, k_q=K_LEAK)
            n += 1
            engaged += int(res[1][fire] < F_ONE)
    if engaged == 0:
        _fail("P4: the Fleck factor never dropped below 2^24 on a fire tile")
    print(f"  {n} configurations; Fleck engaged on the fire tile in {engaged}")


# ---------------------------------------------------------------------------
# PART 5 — dyn_heat_atten_q stamps
# ---------------------------------------------------------------------------
def part5_bodies(table) -> None:
    print("PART 5 — dyn_heat_atten_q stamps (bodies on air and on solids)")
    n = 0
    sc, fire = _fire_room(1263 << 16)
    d = sc.d.astype(np.int64)
    fy, fx = fire
    # a marine beside the fire, one in its shadow behind it, a partial body on
    # a cold crate, and a few scattered opaque bodies on air
    d[fy, fx - 1] = ONE
    d[fy, fx - 2] = ONE
    d[fy + 4, fx] = min(ONE, int(d[fy + 4, fx]) + Q(0.05))
    rng = np.random.default_rng(5)
    air = np.argwhere(sc.a == 0)
    for y, x in air[rng.choice(len(air), size=6, replace=False)]:
        d[y, x] = ONE if rng.random() < 0.5 else Q(0.5)
    sc_b = Scene(sc.a, d, sc.T, sc.his, sc.ts)
    for transport in ("shear", "step"):
        for k_q in (0, K_LEAK):
            tag = f"P5 bodies {transport} k={k_q}"
            res = both(tag, sc_b, table, transport=transport, k_q=k_q)
            n += 1
            rf = res[0][1]
            if not np.any(rf != 0):
                _fail(f"{tag}: rad_flux identically zero — the body share was "
                      "never compared")
            if not rf[fy, fx - 1] > 0:
                _fail(f"{tag}: the marine beside the fire absorbs nothing "
                      f"(rad_flux={int(rf[fy, fx - 1])}) — the hot stream never "
                      "met a body")
    # and under a cold sky, where the body sensor goes NEGATIVE in places
    vac = np.zeros((sc.h, sc.w), dtype=bool)
    vac[:, :3] = True
    for transport in ("shear", "step"):
        tag = f"P5 bodies cold-sky {transport}"
        res = both(tag, sc_b, table, transport=transport, k_q=K_LEAK,
                   is_vacuum=vac, vac_level=0)
        n += 1
    print(f"  {n} configurations")


# ---------------------------------------------------------------------------
# PART 6 — shapes
# ---------------------------------------------------------------------------
def _np_scene(rng, h, w):
    """A fast numpy scene for big grids (gate 0's vocabulary, vectorised)."""
    ts = rng.random((h, w)) < 0.45
    a = np.where(ts, rng.choice([0, Q(0.3), Q(0.5), Q(0.91), ONE], size=(h, w)), 0)
    body = rng.random((h, w)) < 0.05
    d = np.minimum(ONE, a + np.where(body, rng.choice([Q(0.5), ONE], size=(h, w)), 0))
    T = rng.choice(np.asarray([-(200 << 16), 0, 0, 0, 300 << 16, 1263 << 16,
                               5000 << 16, 15999 << 16, (32767 << 16) + 65535],
                              dtype=np.int64), size=(h, w))
    his = rng.choice([-3, 0, 3, 4, 5], size=(h, w))
    return Scene(a, d, T, his, ts)


def part6_shapes(table) -> None:
    print("PART 6 — shapes (degenerate, odd, live-size, full-size)")
    rng = np.random.default_rng(20260924)
    n = 0
    shapes = ((1, 1), (1, 9), (9, 1), (2, 2), (3, 40), (40, 3), (33, 65),
              (7, 300), (300, 7), (128, 256), (256, 512))
    for (h, w) in shapes:
        sc = _np_scene(rng, h, w)
        vac = rng.random((h, w)) < 0.1
        for transport in ("shear", "step"):
            n_ords = (16, 12) if h * w <= 128 * 256 else (16,)
            for n_ord in n_ords:
                tag = f"P6 {h}x{w} {transport} S{n_ord}"
                both(tag, sc, table, transport=transport, n_ord=n_ord,
                     k_q=K_LEAK, is_vacuum=vac, vac_level=0)
                n += 1
    print(f"  {n} configurations over {len(shapes)} shapes (the largest "
          f"wavefront is 512 cells, two blocks)")


# ---------------------------------------------------------------------------
# PART 7 — overwrite + ingress
# ---------------------------------------------------------------------------
def _raises_on_both(tag, sc, table, **kw):
    try:
        cpu_run(sc, table, **kw)
    except ValueError:
        cpu_raised = True
    else:
        cpu_raised = False
    out = [np.full((sc.h, sc.w), 777, dtype=np.int64) for _ in range(4)]
    fo = np.full((sc.h, sc.w), 777, dtype=np.int32)
    amb = kw.get("amb")
    amb_arr = None if amb is None else np.ascontiguousarray(
        np.broadcast_to(np.asarray(amb, dtype=np.int64), (sc.h, sc.w)))
    g_a, hq_a, nb_a = kw["gas"] if kw.get("gas") is not None else (None, None, None)
    try:
        bp.cuda_radiation_sweep_run(
            sc.T, sc.a, sc.d, sc.his, sc.ts, table, amb_arr, int(T_AMB_Q),
            int(kw.get("k_q", 0)), TRANSPORTS[kw.get("transport", "shear")],
            int(kw.get("n_ord", 16)), *out,
            is_vacuum=kw.get("is_vacuum"), vac_level=int(kw.get("vac_level", -1)),
            fleck_out=fo, gas=g_a, heat_absorb_q16=hq_a, n_bulk=nb_a,
            **_currency_kw(kw.get("gas"), kw.get("currency")))
    except ValueError:
        gpu_raised = True
    else:
        gpu_raised = False
    if not cpu_raised:
        _fail(f"{tag}: the CPU sweep did NOT reject this scene — the control is wrong")
    if not gpu_raised:
        _fail(f"{tag}: the GPU twin measured a scene the CPU rejects")
    if any(np.any(p != 777) for p in out) or np.any(fo != 777):
        _fail(f"{tag}: a rejected scene touched the caller's planes")


def part7_ingress(table) -> None:
    print("PART 7 — overwrite + ingress rejection")
    e0 = int(np.asarray(table.table())[0])
    rng = random.Random(7)
    base = Scene(*random_scene(rng, 6, 7))
    # overwrite: two consecutive GPU runs into the SAME planes equal one CPU run
    c = cpu_run(base, table, k_q=K_LEAK)
    out = [np.full((6, 7), GARBAGE, dtype=np.int64) for _ in range(4)]
    for _ in range(2):
        bp.cuda_radiation_sweep_run(base.T, base.a, base.d, base.his, base.ts, table,
                                    None, int(T_AMB_Q), K_LEAK,
                                    TRANSPORTS["shear"], 16, *out)
    for name, cp_, gp in zip(PLANES, c[0], out):
        if not np.array_equal(cp_, gp):
            _fail(f"P7 overwrite: {name} accumulated across two GPU runs")
    # the rejections
    a_gt_d = Scene(base.a, base.d, base.T, base.his, base.ts)
    a_gt_d.a = a_gt_d.a.copy(); a_gt_d.d = a_gt_d.d.copy()
    a_gt_d.a[2, 3] = Q(0.9); a_gt_d.d[2, 3] = Q(0.5)
    _raises_on_both("P7 a > d", a_gt_d, table)
    d_gt_one = Scene(base.a, base.d, base.T, base.his, base.ts)
    d_gt_one.d = d_gt_one.d.copy(); d_gt_one.d[1, 1] = ONE + 1
    _raises_on_both("P7 d > ONE", d_gt_one, table)
    a_neg = Scene(base.a, base.d, base.T, base.his, base.ts)
    a_neg.a = a_neg.a.copy(); a_neg.a[0, 0] = -1
    _raises_on_both("P7 a < 0", a_neg, table)
    _raises_on_both("P7 k_leak > ONE", base, table, k_q=ONE + 1)
    _raises_on_both("P7 k_leak < 0", base, table, k_q=-1)
    vac = np.zeros((6, 7), dtype=bool); vac[0, 0] = True
    _raises_on_both("P7 vac_level > E0", base, table, is_vacuum=vac, vac_level=e0 + 1)
    amb_hi = np.full((6, 7), e0, dtype=np.int64); amb_hi[3, 3] = e0 + 1
    _raises_on_both("P7 amb > E0", base, table, amb=amb_hi)
    amb_lo = np.full((6, 7), e0, dtype=np.int64); amb_lo[4, 4] = -1
    _raises_on_both("P7 amb < 0", base, table, amb=amb_lo)
    # an unsupported ordinate count: both refuse before touching anything
    try:
        cpu_run(base, table, n_ord=8)
        _fail("P7 S8: the CPU accepted n_ordinates = 8")
    except ValueError:
        pass
    try:
        gpu_run(base, table, n_ord=8)
        _fail("P7 S8: the GPU accepted n_ordinates = 8")
    except ValueError:
        pass
    print("  overwrite exact; 8 illegal scenes rejected by both, planes untouched")


# ---------------------------------------------------------------------------
# PART 8 — the launch core at N = 3
# ---------------------------------------------------------------------------
def run_batch(table, scs, transport, n_ord, vacs, vac_levels, k_leaks, t_ambs,
              amb_planes=None, gas_planes=None, hq=None, nb_planes=None,
              currency=None):
    """The (N, h, w) LAUNCH CORE on N scenes in one launch sequence. `gas_planes`
    (N arrays of (n_gases, h, w)), `hq` (one shared (n_gases,) table) and
    `nb_planes` (N (h, w) bulk planes) are the P5a smoke group — all or none;
    with it `currency` is the gas arm's shared (n_floor_q, recip_cv), P5b."""
    import cupy as cp
    SLOTS = int(bp.RADIATION_SWEEP_CNT_SLOTS)
    N = len(scs)
    h, w = scs[0].h, scs[0].w
    stack = lambda key, dt: cp.asarray(np.stack([getattr(s, key) for s in scs]).astype(dt))  # noqa: E731
    d_T, d_a, d_d = stack("T", np.int32), stack("a", np.int32), stack("d", np.int32)
    d_his, d_ts = stack("his", np.int32), stack("ts", np.bool_)
    d_vac = cp.asarray(np.stack(vacs[:N]))
    d_amb = None if amb_planes is None else cp.asarray(np.stack(amb_planes).astype(np.int64))
    d_etab = cp.asarray(np.asarray(table.table(), dtype=np.int64))
    d_vl = cp.asarray(np.asarray(vac_levels[:N], dtype=np.int64))
    d_kl = cp.asarray(np.asarray(k_leaks[:N], dtype=np.int32))
    d_ta = cp.asarray(np.asarray(t_ambs[:N], dtype=np.int32))
    d_gas = d_hq = d_nb = None
    n_gases = 0
    nf = rcv = 0                                   # no group: the currency is not read
    if gas_planes is not None:
        d_gas = cp.asarray(np.stack(gas_planes).astype(np.int32))      # (N, n_gases, h, w)
        d_hq = cp.asarray(np.asarray(hq, dtype=np.int32))
        d_nb = cp.asarray(np.stack(nb_planes).astype(np.int32))
        n_gases = int(d_gas.shape[1])
        nf, rcv = currency if currency is not None else engine_currency()
    d_out = cp.empty((N, n_ord, h, w), dtype=cp.int64)
    d_ambm = cp.empty((N, h, w), dtype=cp.int64)
    d_ex = cp.empty((N, h, w), dtype=cp.int64)
    d_f = cp.full((N, h, w), F_GARBAGE, dtype=cp.int32)
    d_ae = cp.empty((N, h, w), dtype=cp.int32)
    d_de = cp.empty((N, h, w), dtype=cp.int32)
    d_rad = [cp.full((N, h, w), GARBAGE, dtype=cp.int64) for _ in range(4)]
    d_cnt = cp.zeros((N, SLOTS), dtype=cp.int64)
    ptr = lambda a: 0 if a is None else a.data.ptr                    # noqa: E731
    launches = bp.cuda_radiation_sweep_resident(
        N, h, w, d_T.data.ptr, d_a.data.ptr, d_d.data.ptr, d_his.data.ptr,
        d_ts.data.ptr, ptr(d_amb), d_vac.data.ptr,
        d_etab.data.ptr, int(table.fine_bits),                # #78: its currency
        d_vl.data.ptr, d_kl.data.ptr, d_ta.data.ptr,
        ptr(d_gas), n_gases, ptr(d_hq), ptr(d_nb), int(nf), int(rcv),
        TRANSPORTS[transport], n_ord, True,
        d_out.data.ptr, d_ambm.data.ptr, d_ex.data.ptr, d_f.data.ptr,
        d_ae.data.ptr, d_de.data.ptr,
        *[p.data.ptr for p in d_rad], d_cnt.data.ptr)
    cp.cuda.Device().synchronize()
    want = bp.cuda_radiation_sweep_launch_count(TRANSPORTS[transport], n_ord, h, w)
    if launches != want:
        _fail(f"batch: the core issued {launches} launches, the shape says {want}")
    return ([p.get() for p in d_rad], d_f.get(), d_cnt.get())


def part8_batch(table) -> None:
    print("PART 8 — the launch core, N = 3 envs in one launch sequence")
    BAD_EXTINCTION = int(bp.RS_SLOT_BAD_EXTINCTION)
    S_MIN, S_MAX = int(bp.RS_SLOT_MIN_STREAM), int(bp.RS_SLOT_MAX_STREAM)
    e0 = int(np.asarray(table.table())[0])
    h, w = 13, 21
    rng = np.random.default_rng(88)
    scenes = [_np_scene(rng, h, w) for _ in range(3)]
    vacs = [rng.random((h, w)) < p for p in (0.0, 0.3, 0.6)]
    vac_levels = [-1, 0, e0 // 2]
    k_leaks = [0, K_LEAK, Q(0.37)]
    t_ambs = [int(T_AMB_Q), int(T_AMB_Q), int(T_AMB_Q) + (7 << 16)]

    def run_batch_(scs, transport, n_ord, amb_planes=None):
        return run_batch(table, scs, transport, n_ord, vacs, vac_levels, k_leaks,
                         t_ambs, amb_planes=amb_planes)

    n = 0
    for transport, n_ord in (("shear", 16), ("step", 12)):
        rad, f, cnt = run_batch_(scenes, transport, n_ord)
        for e, sc in enumerate(scenes):
            c = cpu_run(sc, table, transport=transport, n_ord=n_ord, k_q=k_leaks[e],
                        t_amb=t_ambs[e], is_vacuum=vacs[e], vac_level=vac_levels[e])
            g = ([rad[i][e] for i in range(4)], f[e], int(cnt[e][S_MIN]), int(cnt[e][S_MAX]))
            same(f"P8 env {e} {transport} S{n_ord}", c, g)
            n += 1
        # the envs really differ — so a cross-env mix-up could not hide
        if np.array_equal(rad[0][0], rad[0][1]) or np.array_equal(rad[0][1], rad[0][2]):
            _fail("P8: two envs produced the same rad_net — the batch is vacuous")
    # an explicit ambient plane per env (run()'s door) at N = 3
    amb_planes = [np.where(rng.random((h, w)) < 0.3, 0, e0).astype(np.int64)
                  for _ in range(3)]
    rad, f, cnt = run_batch_(scenes, "shear", 16, amb_planes=amb_planes)
    for e, sc in enumerate(scenes):
        c = cpu_run(sc, table, k_q=k_leaks[e], t_amb=t_ambs[e], amb=amb_planes[e])
        same(f"P8 env {e} amb-plane", c,
             ([rad[i][e] for i in range(4)], f[e], int(cnt[e][S_MIN]), int(cnt[e][S_MAX])))
        n += 1
    # one ILLEGAL env: masked alone, its planes left as the caller had them
    bad = [scenes[0], Scene(scenes[1].a, scenes[1].d, scenes[1].T, scenes[1].his,
                            scenes[1].ts), scenes[2]]
    bad[1].a = bad[1].a.copy(); bad[1].d = bad[1].d.copy()
    bad[1].a[5, 5] = Q(0.9); bad[1].d[5, 5] = Q(0.4)
    rad, f, cnt = run_batch_(bad, "shear", 16)
    if int(cnt[1][BAD_EXTINCTION]) != 1:
        _fail(f"P8 illegal env: counted {int(cnt[1][BAD_EXTINCTION])} bad cells, want 1")
    if int(cnt[0][BAD_EXTINCTION]) or int(cnt[2][BAD_EXTINCTION]):
        _fail("P8 illegal env: a LEGAL env was charged a violation")
    if any(np.any(rad[i][1] != GARBAGE) for i in range(4)):
        _fail("P8 illegal env: the rejected env's planes were touched")
    for e in (0, 2):
        c = cpu_run(bad[e], table, k_q=k_leaks[e], t_amb=t_ambs[e],
                    is_vacuum=vacs[e], vac_level=vac_levels[e])
        same(f"P8 legal env {e} beside an illegal one", c,
             ([rad[i][e] for i in range(4)], f[e], int(cnt[e][S_MIN]), int(cnt[e][S_MAX])))
        n += 1
    print(f"  {n} env comparisons (3 envs x 2 ordinate sets, the plane door, "
          f"and a masked illegal env)")


# ---------------------------------------------------------------------------
# PART 9 — the live conductor
# ---------------------------------------------------------------------------
_ALL_BACKENDS = (
    "set_temperature_backend", "set_water_backend", "set_smoke_backend",
    "set_fire_backend", "set_radiation_backend",
    "set_bulk_flux_backend", "set_sl_advection_backend",
    "set_mg_solve_backend", "set_kick_compression_backend",
    "set_combustion_backend",
)
_TEMP_COUNTERS = ("t_max_phys_hits", "t_low_rail_hits", "rad_clamp_hits",
                  "e_rad_clamp_drop_sum",                       # P5c
                  "e_rad_boundary_export_sum", "e_rad_floor_drop_sum",  # P5c follow-up
                  "e_cond_trunc_sum", "e_cond_cap_sum", "cond_limit_hits",
                  "e_vac_wipe_sum", "e_ring_pin_sum", "e_deposit_drop_sum",
                  "e_gas_deposit_sum", "e_gas_cond_sum", "e_gas_rail_sum",
                  "e_solid_deposit_sum", "e_solid_cond_sum",
                  "solid_energy_books_sum")


def _burning_playground():
    """tests/test_radiation_sweep_shadow_wiring.py's live scene: the playground
    level with a wood tile HEATED to the 1263-game plateau and lit — a fire is
    started by delivering heat (CLAUDE.md "Starting a fire")."""
    from level_loader import load as load_level
    from simulation import Simulation, fire_fixed
    from simulation.materials import MAT_WOOD
    sim = Simulation(load_level("playground"), seed=1, breach_physics=bp,
                     enable_recorder=False)
    g = sim.gmap
    ys, xs = np.where((g.material == MAT_WOOD) & g.thermal_solid)
    pick = None
    for y, x in zip(ys, xs):
        if 0 < y < g.material.shape[0] - 1 and 0 < x < g.material.shape[1] - 1:
            if not g.thermal_solid[y, x + 1] or not g.thermal_solid[y + 1, x]:
                pick = (int(y), int(x))
                break
    assert pick is not None, "the playground carries no wood tile beside air"
    g.temperature[pick] = 1263 << 16
    g.fire[pick] = fire_fixed.quantize_scalar(0.8)
    # A marine on open air near the fire, so the live stamp puts a BODY share
    # (d > a) into dyn_heat_atten_q and rad_flux_sweep is exercised live.
    from simulation.unit import Unit
    y0, x0 = pick
    spot = None
    for r in (2, 3, 1, 4):
        for dy, dx in ((0, -r), (0, r), (r, 0), (-r, 0), (r, r), (-r, -r)):
            y, x = y0 + dy, x0 + dx
            if (0 < y < g.material.shape[0] - 1 and 0 < x < g.material.shape[1] - 1
                    and not g.solid[y, x] and not g.thermal_solid[y, x]
                    and not g.is_vacuum[y, x]):
                spot = (y, x)
                break
        if spot:
            break
    assert spot is not None, "no open air cell near the fire for the marine"
    sim.add_unit(Unit("M1", x=spot[1], y=spot[0], team=0))
    return sim, pick


# The resident leg compares the SYNCED set (the cuda_s8a_check list) plus the
# sweep's four planes and the conserved energy field: in residency mode the
# mirror is authoritative for exactly those, and a non-synced scratch array is
# not a claim the resident tick makes.
_RESIDENT_FIELDS = ("atmosphere", "wave_p", "wind_x", "wind_y", "temperature",
                    "heat", "fire", "wall_hp", "water_depth", "flow_vx",
                    "flow_vy", "gas", "ripple", "ripple_v", "gas_energy",
                    "rad_net_sweep", "rad_flux_sweep", "rad_amb_sweep",
                    "rad_fluence")


def _arrays(g, fields=None):
    if fields is not None:
        return {k: getattr(g, k) for k in fields}
    return {k: v for k, v in vars(g).items() if isinstance(v, np.ndarray)}


def _compare_worlds(tag, t, s_cpu, s_gpu, fields=None) -> int:
    bad = 0
    a_cpu, a_gpu = _arrays(s_cpu.gmap, fields), _arrays(s_gpu.gmap, fields)
    if set(a_cpu) != set(a_gpu):
        _fail(f"{tag} tick {t}: the two worlds hold different array sets")
        return 1
    for k in sorted(a_cpu):
        x, y = a_cpu[k], a_gpu[k]
        eq = (np.array_equal(x, y, equal_nan=True) if x.dtype.kind == "f"
              else np.array_equal(x, y))
        if not eq:
            bad += 1
            _fail(f"{tag} tick {t}: gmap.{k} differs "
                  f"({int(np.count_nonzero(x != y))} cells)")
    tc, tg = s_cpu.physics_runner.engine.temperature, s_gpu.physics_runner.engine.temperature
    for c in _TEMP_COUNTERS:
        if int(getattr(tc, c)) != int(getattr(tg, c)):
            bad += 1
            _fail(f"{tag} tick {t}: temperature.{c} cpu={int(getattr(tc, c))} "
                  f"gpu={int(getattr(tg, c))}")
    rc, rg = s_cpu.physics_runner.engine.radiation, s_gpu.physics_runner.engine.radiation
    if (int(rc.min_stream), int(rc.max_stream)) != (int(rg.min_stream), int(rg.max_stream)):
        bad += 1
        _fail(f"{tag} tick {t}: engine sweep telemetry cpu=({rc.min_stream}, "
              f"{rc.max_stream}) gpu=({rg.min_stream}, {rg.max_stream})")
    if not np.array_equal(np.asarray(rc.fleck_plane()), np.asarray(rg.fleck_plane())):
        bad += 1
        _fail(f"{tag} tick {t}: the engine's Fleck plane differs")
    return bad


def _run_live(tag, n_ticks, gpu_on, gpu_off, cold_sky=False, fields=None,
              builder=None):
    fails_before = len(_FAILS)
    builder = builder if builder is not None else _burning_playground
    s_cpu, pick = builder()
    s_gpu, _ = builder()
    if cold_sky:
        for s in (s_cpu, s_gpu):
            s.physics_runner.rad_amb_vacuum_q = 0      # Erik's cold sky, both worlds
    if _compare_worlds(tag, "init", s_cpu, s_gpu, fields):
        return
    act = dict(rn=0, rf=0, phi=0, fire=0, calls=0)
    e0 = int(np.asarray(s_cpu.physics_runner.engine.emissive.table())[0])
    amb_phi = 16 * ((e0 * 4096) >> 16)
    for t in range(n_ticks):
        gpu_off()
        s_cpu.set_paused(False)
        s_cpu.step()
        c0 = int(bp.radiation_sweep_cuda_calls())
        gpu_on()
        s_gpu.set_paused(False)
        s_gpu.step()
        act["calls"] += int(bp.radiation_sweep_cuda_calls()) - c0
        gpu_off()
        g = s_cpu.gmap
        act["rn"] += int(np.any(g.rad_net_sweep != 0))
        act["rf"] += int(np.any(g.rad_flux_sweep != 0))
        act["phi"] += int(int(g.rad_fluence.max()) > amb_phi)
        act["fire"] += int(np.any(g.fire > 0))
        if _compare_worlds(tag, t, s_cpu, s_gpu, fields) >= 4:
            print("  aborting after 4 divergent ticks")
            break
    if act["calls"] != n_ticks:
        _fail(f"{tag}: the GPU sweep ran {act['calls']} times in {n_ticks} ticks — "
              "the dispatch did not fire every tick (a CPU-only 'GPU run' is vacuous)")
    for key, what in (("rn", "rad_net_sweep non-zero"), ("phi", "a fluence above ambient"),
                      ("fire", "a burning tile"), ("rf", "a body share (rad_flux_sweep != 0)")):
        if act[key] == 0:
            _fail(f"{tag}: never saw {what} — the live scene is vacuous")
    what = ("every GameMap array" if fields is None
            else f"the {len(fields)} synced + sweep planes")
    verdict = "tol 0" if len(_FAILS) == fails_before else "FAILED"
    print(f"  {tag}: {n_ticks} ticks, {verdict} on {what}, the temperature "
          f"counters and the engine's sweep telemetry; GPU sweep calls "
          f"{act['calls']}; ticks with rad_net_sweep != 0: {act['rn']}, with "
          f"rad_flux_sweep != 0: {act['rf']}, fluence above ambient: {act['phi']}; "
          f"fire at ({pick[0]},{pick[1]})")


def part9_live() -> None:
    print("PART 9 — the live conductor (Simulation.step, playground, a burning tile)")
    from simulation import physics_runner

    def only_radiation(on):
        physics_runner.set_residency(False)
        for name in _ALL_BACKENDS:
            getattr(bp, name)(False)
        bp.set_radiation_backend(bool(on))

    _run_live("9a step path, radiation backend only", 30,
              lambda: only_radiation(True), lambda: only_radiation(False))
    _run_live("9b step path, cold sky", 20,
              lambda: only_radiation(True), lambda: only_radiation(False),
              cold_sky=True)

    # The resident tick with every backend on EXCEPT combustion. On this scene
    # the combustion twin (cuda_combustion.cu) is NOT bit-identical to
    # CombustionSolver — 6 cells around the fire in temperature/gas_energy on
    # tick 0, whether the sweep runs on the CPU or the GPU and whether the tick
    # is resident or not (measured at P4; a finding in another system, recorded
    # and not fixed here). Excluding it keeps this leg a statement about the
    # sweep riding the resident tick's host bracket, which is what it is for.
    def all_gpu_resident(on):
        physics_runner.set_residency(bool(on))
        for name in _ALL_BACKENDS:
            getattr(bp, name)(bool(on) and name != "set_combustion_backend")

    try:
        import cupy  # noqa: F401
    except Exception as exc:      # pragma: no cover — reported, never skipped
        _fail(f"9c: cupy unavailable ({exc}) — the resident leg cannot be gated")
        return
    _run_live("9c resident tick, every backend but combustion", 15,
              lambda: all_gpu_resident(True), lambda: all_gpu_resident(False),
              fields=_RESIDENT_FIELDS)
    all_gpu_resident(False)


# ---------------------------------------------------------------------------
# PART 10 — THE SMOKE TERM (P5a, design v3 §6.3)
# ---------------------------------------------------------------------------
def _gas(gas, hq, n_bulk):
    """A reference-format gas group as the (gas, table, bulk) int32 triple."""
    return gas_arrays(gas, hq, n_bulk)


def _np_gas(rng, h, w, *, active=(Q(5.0), Q(0.9))):
    """The ENGINE's gas layout on a big grid: 7 planes (steam .. fuel_gas, o2,
    inert_n2), two trace gases absorbing, the rest at coefficient 0 (the per-call
    path must upload only the active two), trace densities in the tracer range,
    and a bulk count with vacuum-like and thin cells."""
    gas = np.zeros((7, h, w), dtype=np.int32)
    for gi in range(5):
        gas[gi] = np.where(rng.random((h, w)) < 0.4,
                           rng.choice([1, Q(0.02), Q(0.06), Q(0.27), ONE], size=(h, w)), 0)
    bulk = np.where(rng.random((h, w)) < 0.1, 0,
                    np.where(rng.random((h, w)) < 0.2, Q(0.2), ONE))
    gas[5] = (bulk * 13763) // ONE
    gas[6] = bulk - gas[5]
    hq = np.zeros(7, dtype=np.int32)
    hq[1], hq[3] = active                      # smoke, teargas absorb; others 0
    n_bulk = (gas[5].astype(np.int64) + gas[6]).astype(np.int32)
    return np.ascontiguousarray(gas), hq, np.ascontiguousarray(n_bulk)


def part10_smoke(table) -> None:
    print("PART 10 — THE SMOKE TERM (P5a): gas extinction on both backends, tol 0")
    n = 0
    # (a) gate 0's smoke matrix, verbatim
    for seed, h, w in ((11, 7, 9), (12, 9, 11), (13, 12, 8), (14, 5, 5)):
        for n_ord in (16, 12):
            for k_q in (0, K_LEAK):
                for transport in ("shear", "step"):
                    for kind in ("uniform", "random"):
                        rng = random.Random(20260924 + seed)
                        a, d, T, his, ts = random_scene(rng, h, w)
                        amb = _ambient(kind, rng, h, w)
                        gas = _gas(*random_gas(rng, h, w))
                        sc = Scene(a, d, T, his, ts)
                        tag = f"P10a seed={seed} S{n_ord} k={k_q} {transport} {kind}"
                        res = both(tag, sc, table, transport=transport, n_ord=n_ord,
                                   k_q=k_q, amb=amb, gas=gas)
                        clear = cpu_run(sc, table, transport=transport, n_ord=n_ord,
                                        k_q=k_q, amb=amb)
                        if np.array_equal(res[0][0], clear[0][0]):
                            _fail(f"{tag}: the smoke term moved nothing — vacuous")
                        # P5b: the gas arm damped a gas cell here, so the
                        # compared Fleck plane carries it
                        if not np.any((res[1] < F_ONE) & ~sc.ts):
                            _fail(f"{tag}: no gas cell damped — the gas arm was "
                                  "never compared")
                        n += 1
    # (b) the one scene carrying every feature
    a, d, T, ts, gas_l, hq_l, nb_l, cells = smoke_feature_scene()
    sc_f = Scene(a, d, T, R.plane(len(a), len(a[0]), 3), ts)
    for transport in ("shear", "step"):
        for k_q in (0, K_LEAK):
            res = both(f"P10b features {transport} k={k_q}", sc_f, table,
                       transport=transport, k_q=k_q, gas=_gas(gas_l, hq_l, nb_l))
            if not res[1][cells["hot"]] < F_ONE:
                _fail(f"P10b features {transport} k={k_q}: the hot smoke cell is "
                      "undamped — the gas arm was never compared")
            n += 1
    # (c) a burning tile in a smoke cloud, on the LIVE calibration
    live, k_live, t_amb_live = _live_table_and_leak()
    sc_r, fire = _fire_room(1263 << 16)
    fy, fx = fire
    h, w = sc_r.h, sc_r.w
    smoke = np.zeros((h, w), dtype=np.int32)
    yy, xx = np.mgrid[0:h, 0:w]
    ring = (np.abs(yy - fy) <= 3) & (np.abs(xx - fx) <= 3) & ~sc_r.ts
    smoke[ring] = np.where(((yy + xx) % 2 == 0)[ring], Q(0.27), Q(0.06))
    o2 = np.full((h, w), 13763, dtype=np.int32)
    n2 = np.full((h, w), 51773, dtype=np.int32)
    gas_c = (np.ascontiguousarray(np.stack([smoke, o2, n2])),
             np.asarray([Q(5.0), 0, 0], dtype=np.int32),
             np.ascontiguousarray(o2 + n2))
    for transport in ("shear", "step"):
        tag = f"P10c live smoky fire {transport}"
        res = both(tag, sc_r, live, transport=transport, k_q=k_live, t_amb=t_amb_live,
                   gas=gas_c)
        clear = cpu_run(sc_r, live, transport=transport, k_q=k_live, t_amb=t_amb_live)
        rn, rn0 = res[0][0], clear[0][0]
        if not (int(rn[ring].sum()) > 0 and int(rn[ring].max()) > 0):
            _fail(f"{tag}: the smoke ring does not absorb on the live table")
        if not (rn[fy, fx + 2] < rn0[fy, fx + 2]):
            _fail(f"{tag}: the crate behind the smoke is not shaded "
                  f"({int(rn[fy, fx + 2])} vs clear {int(rn0[fy, fx + 2])})")
        n += 1
    # (d) the engine's 7-plane layout on full-size grids (wavefronts > one block)
    rng = np.random.default_rng(20260925)
    for (h, w) in ((33, 65), (128, 256), (256, 512)):
        sc = _np_scene(rng, h, w)
        gas = _np_gas(rng, h, w)
        vac = rng.random((h, w)) < 0.05
        for transport in ("shear", "step"):
            res = both(f"P10d {h}x{w} {transport}", sc, table, transport=transport,
                       k_q=K_LEAK, is_vacuum=vac, vac_level=0, gas=gas)
            if not np.any((res[1] < F_ONE) & ~sc.ts):
                _fail(f"P10d {h}x{w} {transport}: no gas cell damped (P5b arm)")
            n += 1
    # an all-zero table on the SAME planes is the gas-free sweep, both backends
    sc = _np_scene(rng, 33, 65)
    gas_z = _np_gas(rng, 33, 65, active=(0, 0))
    z = both("P10d all-zero table", sc, table, k_q=K_LEAK, gas=gas_z)
    same("P10d all-zero table == no gas group", z, cpu_run(sc, table, k_q=K_LEAK))
    n += 1
    # (e) the launch core at N = 3: per-env gas planes, one shared table
    part10_batch(table)
    # (f) ingress: what the CPU rejects the GPU rejects, planes untouched
    base = Scene(*random_scene(random.Random(77), 6, 7))
    g_ok = _gas(*random_gas(random.Random(78), 6, 7))
    bad = (
        ("P10f heat_absorb < 0", (g_ok[0], np.where(np.arange(len(g_ok[1])) == 1, -1,
                                                    g_ok[1]).astype(np.int32), g_ok[2])),
        ("P10f heat_absorb > 2^28", (g_ok[0], np.full_like(g_ok[1], R.HEAT_ABSORB_Q_MAX + 1),
                                     g_ok[2])),
        ("P10f 17 gas planes", (np.zeros((17, 6, 7), dtype=np.int32),
                                np.zeros(17, dtype=np.int32), g_ok[2])),
    )
    for tag, g in bad:
        _raises_on_both(tag, base, table, gas=g)
    for tag, kw in (("P10f partial group", dict(gas=g_ok[0], heat_absorb_q16=g_ok[1])),):
        for fn in (_cpu_partial, _gpu_partial):
            try:
                fn(base, table, **kw)
                _fail(f"{tag}: accepted by {fn.__name__}")
            except ValueError:
                pass
    # P5b: the gas arm's currency — a non-positive one (a forgotten currency, the
    # bindings' 0 default) is refused alike, planes untouched
    nf, rcv = engine_currency()
    for tag, cur in (("P10f currency n_floor_q = 0", (0, rcv)),
                     ("P10f currency recip_cv = 0", (nf, 0)),
                     ("P10f currency n_floor_q < 0", (-nf, rcv)),
                     ("P10f currency recip_cv < 0", (nf, -rcv))):
        _raises_on_both(tag, base, table, gas=g_ok, currency=cur)
    print(f"  {n} configurations + the N = 3 batch + 4 illegal gas inputs + 4 "
          f"illegal currencies")
    # (h) the gas arm's own scene, both tables, two currencies
    part10h_gas_arm(table)


def _cpu_partial(sc, table, **kw):
    out = [np.zeros((sc.h, sc.w), dtype=np.int64) for _ in range(4)]
    bp.RadiationSweep().run(sc.T, sc.a, sc.d, sc.his, sc.ts, table, None, int(T_AMB_Q),
                            0, TRANSPORTS["shear"], 16, *out, **kw)


def _gpu_partial(sc, table, **kw):
    out = [np.zeros((sc.h, sc.w), dtype=np.int64) for _ in range(4)]
    bp.cuda_radiation_sweep_run(sc.T, sc.a, sc.d, sc.his, sc.ts, table, None,
                                int(T_AMB_Q), 0, TRANSPORTS["shear"], 16, *out, **kw)


def part10_batch(table) -> None:
    S_MIN, S_MAX = int(bp.RS_SLOT_MIN_STREAM), int(bp.RS_SLOT_MAX_STREAM)
    SCALARS = int(bp.RS_SLOT_BAD_SCALARS)
    e0 = int(np.asarray(table.table())[0])
    h, w = 13, 21
    rng = np.random.default_rng(99)
    scenes = [_np_scene(rng, h, w) for _ in range(3)]
    gases = [_np_gas(rng, h, w) for _ in range(3)]
    hq = gases[0][1]                                      # ONE table, shared by all envs
    vacs = [rng.random((h, w)) < p for p in (0.0, 0.3, 0.6)]
    vac_levels = [-1, 0, e0 // 2]
    k_leaks = [0, K_LEAK, Q(0.37)]
    t_ambs = [int(T_AMB_Q), int(T_AMB_Q), int(T_AMB_Q) + (7 << 16)]
    damped_envs = 0
    for transport, n_ord in (("shear", 16), ("step", 12)):
        rad, f, cnt = run_batch(table, scenes, transport, n_ord, vacs, vac_levels,
                                k_leaks, t_ambs, gas_planes=[g[0] for g in gases], hq=hq,
                                nb_planes=[g[2] for g in gases])
        for e, sc in enumerate(scenes):
            c = cpu_run(sc, table, transport=transport, n_ord=n_ord, k_q=k_leaks[e],
                        t_amb=t_ambs[e], is_vacuum=vacs[e], vac_level=vac_levels[e],
                        gas=(gases[e][0], hq, gases[e][2]))
            g = ([rad[i][e] for i in range(4)], f[e], int(cnt[e][S_MIN]), int(cnt[e][S_MAX]))
            same(f"P10e env {e} {transport} S{n_ord}", c, g)
            damped_envs += int(np.any((f[e] < F_ONE) & ~sc.ts))    # P5b: the gas arm
    if damped_envs == 0:
        _fail("P10e: no env had a damped gas cell — the batch never compared the gas arm")
    # P5b: a NON-SHIPPED currency rides the core by value into every env alike
    cur = engine_currency(ONE, Q(0.05))
    rad, f, cnt = run_batch(table, scenes, "shear", 16, vacs, vac_levels, k_leaks, t_ambs,
                            gas_planes=[g[0] for g in gases], hq=hq,
                            nb_planes=[g[2] for g in gases], currency=cur)
    for e, sc in enumerate(scenes):
        c = cpu_run(sc, table, k_q=k_leaks[e], t_amb=t_ambs[e], is_vacuum=vacs[e],
                    vac_level=vac_levels[e], gas=(gases[e][0], hq, gases[e][2]),
                    currency=cur)
        same(f"P10e env {e} currency c_v = 1, floor 0.05", c,
             ([rad[i][e] for i in range(4)], f[e], int(cnt[e][S_MIN]), int(cnt[e][S_MAX])))
    # an ILLEGAL shared table: every env flagged, every env's planes untouched
    bad_hq = hq.copy()
    bad_hq[2] = R.HEAT_ABSORB_Q_MAX + 1
    rad, f, cnt = run_batch(table, scenes, "shear", 16, vacs, vac_levels, k_leaks, t_ambs,
                            gas_planes=[g[0] for g in gases], hq=bad_hq,
                            nb_planes=[g[2] for g in gases])
    flag = int(bp.RS_BAD_HEAT_ABSORB)
    for e in range(3):
        if not int(cnt[e][SCALARS]) & flag:
            _fail(f"P10e illegal table: env {e} not flagged RS_BAD_HEAT_ABSORB")
        if any(np.any(rad[i][e] != GARBAGE) for i in range(4)):
            _fail(f"P10e illegal table: env {e}'s planes were touched")
    print("  P10e: N = 3 with per-env gas planes and one shared table, each env == its "
          "own CPU run (the shipped currency and a non-shipped one); an illegal table "
          "flagged in every env, planes untouched")


# The gas_fleck_scene cells the arm damps on both tables (gate 0's own list).
_GAS_DAMPED = ("hot_N1", "hot_top", "hot_isobaric", "hot_floor", "hot_below_floor",
               "hot_eps", "mix_hot", "body_hot")
_GAS_UNDAMPED = ("sub_ambient", "hot_no_bulk", "hot_inert_only")


def part10h_gas_arm(table) -> None:
    """P10h (P5b): the GAS ARM's own scene -- every case of it at once, on the
    LIVE table (where the game runs) and the resolving one, in the shipped
    currency and a non-shipped one (so a device that ignored the currency it was
    handed could not pass), both transports, S16 and S12, leak on and off."""
    print("PART 10h — the gas Fleck arm (P5b): every case, both tables, two currencies")
    a, d, T, ts, gas_l, hq_l, nb_l, cells = gas_fleck_scene()
    sc = Scene(a, d, T, R.plane(len(a), len(a[0]), 3), ts)
    gas = _gas(gas_l, hq_l, nb_l)
    live = live_table()
    shipped, other = engine_currency(), engine_currency(ONE, Q(0.05))
    n = 0
    for tname, tbl in (("live", live), ("resolving", table)):
        for transport in ("shear", "step"):
            for n_ord in (16, 12):
                for k_q in (0, K_LEAK):
                    tag = f"P10h {tname} {transport} S{n_ord} k={k_q}"
                    res = both(tag + " shipped currency", sc, tbl, transport=transport,
                               n_ord=n_ord, k_q=k_q, gas=gas, currency=shipped)
                    f = res[1]
                    bad = [c for c in _GAS_DAMPED if not f[cells[c]] < F_ONE]
                    bad += [c for c in _GAS_UNDAMPED if f[cells[c]] != F_ONE]
                    if bad:
                        _fail(f"{tag}: the gas arm's cases went wrong at {bad}")
                    # c_v = 1 with a 0.05 floor: 130x less stiff, so it damps less
                    # -- and a device that ignored the currency it was handed
                    # (priced every gas cell in the shipped one) would fail here
                    res_o = both(tag + " c_v = 1, floor 0.05", sc, tbl, transport=transport,
                                 n_ord=n_ord, k_q=k_q, gas=gas, currency=other)
                    if np.array_equal(res_o[1], f):
                        _fail(f"{tag}: the currency moved nothing — the second one "
                              "cannot tell a device that ignores it")
                    n += 2
    print(f"  {n} configurations; in the shipped currency every hot absorbing gas cell "
          f"is damped on both tables and every non-absorbing or sub-ambient one is at "
          f"2^24; the second currency moves the Fleck plane on both")


def _hot_smoky_burning_playground():
    """10g's smoky world with the smoke cloud HOT -- seeded at 3000 game through
    the gas-energy seam (gamemap.seed_gas_temperature, never a bare temperature
    write: CLAUDE.md 'Gas temperature is a mirror') -- so on the LIVE table the
    gas arm damps it (soot at ambient density is damped from 1676 game)."""
    sim, pick = _smoky_burning_playground()
    g = sim.gmap
    from simulation.gases import SMOKE
    cloud = (g.gas[SMOKE] > 0) & ~g.thermal_solid & ~g.solid & ~g.is_vacuum
    g.seed_gas_temperature(cloud, 3000 << 16)
    return sim, pick


def part10i_live_hot_smoke() -> None:
    print("PART 10i — the live conductor with HOT smoke: the gas arm on the live path "
          "(P5b)")
    from simulation import physics_runner

    def only_radiation(on):
        physics_runner.set_residency(False)
        for name in _ALL_BACKENDS:
            getattr(bp, name)(False)
        bp.set_radiation_backend(bool(on))

    # NON-VACUITY first: one GPU tick of the hot smoky world damps gas cells in
    # the engine's own Fleck plane, priced in its own gas_capacity_q.
    s, _ = _hot_smoky_burning_playground()
    only_radiation(True)
    s.set_paused(False)
    s.step()
    only_radiation(False)
    eng = s.physics_runner.engine
    fl = np.asarray(eng.radiation.fleck_plane())
    g = s.gmap
    damped = int(np.count_nonzero((fl < F_ONE) & ~g.thermal_solid))
    if damped == 0:
        _fail("10i: no gas cell damped on the live GPU path — the arm is vacuous there")
    _run_live("10i step path, HOT smoke absorbing (the gas arm live)", 15,
              lambda: only_radiation(True), lambda: only_radiation(False),
              builder=_hot_smoky_burning_playground)
    print(f"  one GPU tick damped {damped} gas cells in the engine's own Fleck plane "
          f"(currency {tuple(int(v) for v in eng.gas_capacity_q())})")


def _smoky_burning_playground():
    """part 9's scene plus a smoke cloud around the fire and a TEST-FIXTURE smoke
    coefficient (5.0; the SHIPPED smoke is 25.36 since P5c): the smoke term on
    the live path."""
    from simulation import gas_fixed, unit_fixed
    from simulation.gases import SMOKE
    sim, pick = _burning_playground()
    g = sim.gmap
    y0, x0 = pick
    air = (~g.thermal_solid) & (~g.solid) & (~g.is_vacuum)
    yy, xx = np.mgrid[0:g.material.shape[0], 0:g.material.shape[1]]
    cloud = air & (np.abs(yy - y0) <= 4) & (np.abs(xx - x0) <= 4)
    dens = gas_fixed.quantize(np.where((yy + xx) % 3 == 0, 0.27, 0.06))
    g.gas[SMOKE][cloud] = dens[cloud]
    hq = g.gases.heat_absorb_q16.copy()
    hq[SMOKE] = unit_fixed.quantize_scalar(5.0)
    g.gases.heat_absorb_q16 = np.ascontiguousarray(hq)
    return sim, pick


def part10_live() -> None:
    print("PART 10g — the live conductor with the smoke term on (a fixture coefficient)")
    from simulation import physics_runner

    def only_radiation(on):
        physics_runner.set_residency(False)
        for name in _ALL_BACKENDS:
            getattr(bp, name)(False)
        bp.set_radiation_backend(bool(on))

    # NON-VACUITY first: on the GPU backend, one tick of the smoky world and of
    # the same world with the coefficient zeroed must book different sweeps --
    # so the smoke term really crossed the bus on the live per-call path.
    s_on, _ = _smoky_burning_playground()
    s_off, _ = _smoky_burning_playground()
    s_off.gmap.gases.heat_absorb_q16 = np.zeros_like(s_off.gmap.gases.heat_absorb_q16)
    only_radiation(True)
    for s in (s_on, s_off):
        s.set_paused(False)
        s.step()
    only_radiation(False)
    if np.array_equal(s_on.gmap.rad_net_sweep, s_off.gmap.rad_net_sweep):
        _fail("10g: the smoke coefficient changed nothing on the GPU path — vacuous")
    _run_live("10g step path, smoke absorbing", 20,
              lambda: only_radiation(True), lambda: only_radiation(False),
              builder=_smoky_burning_playground)


# ---------------------------------------------------------------------------
def main() -> int:
    if not getattr(bp, "HAS_CUDA", False) or not bp.cuda_available():
        print("RS_RESULT: FAIL (no CUDA build / device)")
        return 1
    print("device:", bp.cuda_device_info())
    table = reference_table()
    for part in (part1_gate0_matrix, part2_derived_ambient, part3_thin_rows,
                 part4_fire_tile, part5_bodies, part6_shapes, part7_ingress,
                 part8_batch, part10_smoke):
        part(table)
    part1_live_matrix()                          # #78: the fine live table
    part8_batch(live_table())                    # #78: the launch core on it too
    part9_live()
    part10_live()
    part10i_live_hot_smoke()
    if _FAILS:
        print(f"RS_RESULT: FAIL ({len(_FAILS)} failures)")
        return 1
    print("RS_RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
