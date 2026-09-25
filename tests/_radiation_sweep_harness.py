"""Shared drivers for the ray-engine-v2 P1 gates (underscore = harness, not
collected).

Two things every sweep gate needs:

  * THE INTEGER REFERENCE, imported by path from the study folder the way
    tests/test_ray_engine_v2_integer_reference.py does (it is not engine code
    and nothing in src/ or cpp/ may import it — CLAUDE.md "Integer reference").
  * ONE way to drive the C++ sweep and the reference on the SAME scene, so
    gate 0 (bit for bit) and gates 1-6 (properties on the engine's output)
    share a scene vocabulary: `a` (material extinction, Q16), `d >= a` (stamped
    extinction, Q16), `k` (leak, Q16), `T` (Q16.16 temperature), `his`
    (log2 thermal mass per cell), `ts` (the thermal-solid mask).

Scenes are Python lists (the reference's format); `cpp_sweep` converts to the
contiguous numpy dtypes the binding requires (int32 planes, int64 outputs).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_STUDY = ROOT / "docs" / "ray_engine_v2_scheme_study_2026-09-13"
if str(_STUDY) not in sys.path:
    sys.path.insert(0, str(_STUDY))

import breach_physics as bp          # noqa: E402
import sweep_ref_q as R              # noqa: E402  (the executable spec)

ONE = R.ONE
F_ONE = R.F_ONE
T_AMB_Q = R.K_AMB << 16              # 293 game in Q16.16 (G12: slope 1)
TRANSPORTS = {"step": bp.RadiationSweep.STEP, "shear": bp.RadiationSweep.SHEAR}


def engine_currency(c_v_q=R.C_V_Q_LIVE, n_floor_q=R.N_FLOOR_Q_LIVE):
    """The gas arm's currency (P5b) as the ENGINE's sweep takes it: (n_floor_q,
    recip_cv), from the integer form of c_v the reference takes -- recip_cv =
    make_recip(c_v_q / 65536), the fold's own derivation (sweep_ref_q.make_recip
    transcribes the kit's one double divide). The default is the shipped
    currency, the reference's C_V_Q_LIVE / N_FLOOR_Q_LIVE; the live engine's own
    derivation of it (PhysicsEngine.gas_capacity_q) is held to this by
    tests/test_radiation_sweep_gas_fleck.py."""
    return int(n_floor_q), int(R.make_recip(c_v_q / 65536.0))


def reference_table():
    """An EmissiveTable baked at the reference's own dials (== config.toml's,
    which test_ray_engine_v2_integer_reference.py guards) -- the engine's twin
    of sweep_ref_q.E, the RESOLVING table, in its COARSE currency (#78:
    fine_bits = 0 -- the reference's own; that physics does not fit the fine
    currency inside int64, and every gate exercising it keeps its integers)."""
    tbl = bp.EmissiveTable()
    tbl.rad_scale = R.RAD_SCALE
    tbl.kelvin_ambient = float(R.K_AMB)
    tbl.k_temp_to_kelvin = float(R.K_SLOPE)
    tbl.fine_bits = R.fine_bits_of(R.E)
    tbl.bake()
    return tbl


def live_table():
    """An EmissiveTable baked at the LIVE calibration, the sweep's own
    `rad_scale_derived` -- the engine's twin of sweep_ref_q.E_LIVE (which
    config_dials_match() pins to config.toml). The table the game runs on, in
    the currency the game runs it in (#78: the reference's fine bits, which
    tests/test_sweep_fine_currency.py holds equal to the engine's E_FINE_BITS)."""
    tbl = bp.EmissiveTable()
    tbl.rad_scale = R.RAD_SCALE_LIVE
    tbl.kelvin_ambient = float(R.K_AMB)
    tbl.k_temp_to_kelvin = float(R.K_SLOPE)
    tbl.fine_bits = R.fine_bits_of(R.E_LIVE)
    tbl.bake()
    return tbl


def random_scene(rng, h, w):
    """THE randomised scene vocabulary the sweep gates share (gate 0's matrix,
    and the ambient-plane gates that must be measured on the SAME scenes).

    * `ts` — a random thermal-solid mask (about half the cells).
    * `a`  — random material extinction in (0, 1] Q16 on thermal solids
             (sometimes 0 there too, so the mask and `a > 0` are not the same
             set), 0 elsewhere (the materials ingress rule).
    * `d`  — `a` plus a body share on a handful of cells (on solids AND on air:
             a marine stands on air), capped at ONE.
    * `T`  — sub-ambient, ambient, the fire range, the plasma range, the table
             top and ABOVE it (e_bucket_of saturates), mixed per cell.
    * `his`— a per-cell log2(thermal_mass) plane in {3, 4, 5}.

    The rng call ORDER is part of the contract: the scalar-era digest in
    tests/test_radiation_sweep_ambient_plane.py was captured through it.
    """
    Q = R.quant
    ts = [[1 if rng.random() < 0.5 else 0 for _ in range(w)] for _ in range(h)]
    a = [[(rng.choice([0, Q(0.3), Q(0.37), Q(0.5), Q(0.91), ONE]) if ts[y][x] else 0)
          for x in range(w)] for y in range(h)]
    d = [[min(ONE, a[y][x] + rng.choice([0, 0, 0, Q(0.5), ONE - a[y][x]]))
          for x in range(w)] for y in range(h)]
    T = [[rng.choice([-(200 << 16), 0, 0, 300 << 16, 1263 << 16, 5000 << 16,
                      15999 << 16, 16000 << 16, 20000 << 16, (32767 << 16) + 65535])
          for _ in range(w)] for _ in range(h)]
    his = [[rng.choice([3, 4, 5]) for _ in range(w)] for _ in range(h)]
    return a, d, T, his, ts


def random_gas(rng, h, w, n_trace=3):
    """THE randomised GAS vocabulary (P5a, design v3 §6.3) gate 0 and the CUDA
    gate share: `n_trace` trace planes plus the bulk pair (O2, N2), in the
    reference's list format.

    * trace densities span none, a wisp, the measured smoke p90 / p99
      (docs/smoke_tau_diagnostic_2026-08-25.md) and the tracer cap ONE — and a
      few NEGATIVE values, which an int32 plane can hold and which must absorb
      nothing on both sides;
    * `hq` mixes a zero (the CPU skips it, the reference sums it: the two must
      agree), thin coefficients, a saturating one and the door's maximum; the
      bulk pair's coefficients are 0 (air does not absorb in the IR);
    * the bulk count is ambient on most cells, THIN on some, ZERO (below the
      N_EPS floor) on a tenth, and exactly N_EPS_RAW on one.
    Returns (gas planes, hq list, n_bulk plane)."""
    Q = R.quant
    dens = [0, 0, 1, Q(0.001), Q(0.06), Q(0.27), ONE, -Q(0.05)]
    trace = [[[rng.choice(dens) for _ in range(w)] for _ in range(h)]
             for _ in range(n_trace)]
    hq_pool = [0, Q(0.3), Q(0.9), Q(5.0), Q(37.5), R.HEAT_ABSORB_Q_MAX]
    hq = [0] + [rng.choice(hq_pool[1:]) for _ in range(n_trace - 1)] + [0, 0]
    o2 = [[13763] * w for _ in range(h)]
    n2 = [[51773] * w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            r = rng.random()
            if r < 0.10:
                o2[y][x] = n2[y][x] = 0                      # below the floor
            elif r < 0.25:
                o2[y][x], n2[y][x] = Q(0.04), Q(0.16)       # a thin, hot-ish cell
    y, x = rng.randrange(h), rng.randrange(w)
    o2[y][x], n2[y][x] = 0, R.N_EPS_RAW                      # the floor's edge
    n_bulk = [[o2[y][x] + n2[y][x] for x in range(w)] for y in range(h)]
    return trace + [o2, n2], hq, n_bulk


def smoke_feature_scene():
    """ONE scene carrying every feature of the smoke term on purpose (gate 0's
    random matrix may or may not draw each): a hot source, a thin absorbing
    cell, a SATURATED one (a_gas == ONE), a HOT one, a cell full of smoke but
    below the N_EPS floor, one exactly AT the floor, a body standing in thin
    smoke and one in opaque smoke, a thermal solid holding absorbing gas, a
    negative density, and a zero-coefficient gas with density everywhere."""
    Q = R.quant
    h, w = 7, 11
    a = R.plane(h, w, 0)
    d = R.plane(h, w, 0)
    T = R.plane(h, w, 0)
    for y in range(h):
        a[y][0] = d[y][0] = ONE
        T[y][0] = 1263 << 16                            # the source column
    a[3][10] = d[3][10] = Q(0.9)                        # a cold receiver
    a[5][6] = d[5][6] = Q(0.5)                          # a thermal solid ...
    ts = ts_from_a(a)
    smoke = R.plane(h, w, 0)
    zero_g = R.plane(h, w, Q(0.5))                      # coefficient 0: never read
    o2 = R.plane(h, w, 13763)
    n2 = R.plane(h, w, 51773)
    cells = {"thin": (1, 3), "saturated": (2, 4), "hot": (3, 5), "floored": (4, 3),
             "at_floor": (4, 4), "body_thin": (1, 7), "body_opaque": (2, 7),
             "negative": (6, 2)}
    for key, (y, x) in cells.items():
        smoke[y][x] = {"thin": Q(0.06), "negative": -Q(0.4)}.get(key, ONE)
    smoke[5][6] = ONE                                   # ... full of smoke it ignores
    T[3][5] = 5000 << 16                                # hot smoke radiates
    y, x = cells["floored"]
    o2[y][x] = n2[y][x] = 0
    y, x = cells["at_floor"]
    o2[y][x], n2[y][x] = 0, R.N_EPS_RAW
    for key in ("body_thin", "body_opaque"):
        y, x = cells[key]
        d[y][x] = ONE
    smoke[1][7] = Q(0.06)
    gas = [zero_g, smoke, o2, n2]
    hq = [0, Q(5.0), 0, 0]
    n_bulk = [[o2[y][x] + n2[y][x] for x in range(w)] for y in range(h)]
    return a, d, T, ts, gas, hq, n_bulk, cells


def isobaric_n_q(T_game):
    """The bulk N a cell at T holds at AMBIENT PRESSURE (p = C N T_abs):
    N = N_amb * T_amb / T_abs -- a hot cell is a thin cell."""
    return (ONE * R.K_AMB) // (T_game + R.K_AMB)


def gas_fleck_scene():
    """ONE scene carrying every case of the GAS FLECK ARM on purpose (P5b), damped
    on the LIVE table as well as the resolving one: hot absorbing smoke in each of
    P5a's stiffness regimes -- at ambient bulk density (and at the table top), at
    ambient PRESSURE (a hot cell's thin N), at the n_floor_heat density, BELOW it
    (the floor, not N, is then the capacity), at the N_EPS edge -- and a hot
    a = 0.2 mix; beside the cells the arm must leave at 2^24: cool absorbing smoke
    (g < 1), sub-ambient smoke (no excess), hot smoke with NO bulk (below N_EPS:
    a_gas = 0) and a hot cell whose only gas has coefficient 0; plus a hot
    THERMAL SOLID full of smoke (the solid arm, not the gas one), a body standing
    in hot smoke, and a burning wall column whose stream the smoke absorbs.
    Returns (a, d, T, ts, gas, hq, n_bulk, cells), cells = {name: (y, x)}."""
    Q = R.quant
    h, w = 8, 12
    a = R.plane(h, w, 0)
    d = R.plane(h, w, 0)
    T = R.plane(h, w, 0)
    for y in range(h):
        a[y][0] = d[y][0] = ONE
        T[y][0] = 1263 << 16                            # the burning wall column
    a[6][9] = d[6][9] = Q(0.5)                          # a hot thermal solid ...
    T[6][9] = 5000 << 16
    ts = ts_from_a(a)
    smoke = R.plane(h, w, 0)
    inert = R.plane(h, w, 0)                            # coefficient 0: never absorbs
    n2 = R.plane(h, w, ONE)                             # the bulk count, one plane
    smoke[6][9] = ONE                                   # ... full of smoke it ignores
    # name: (y, x, smoke density, bulk N, T game)
    spec = {
        "hot_N1":          (1, 3, ONE, ONE, 3000),
        "hot_top":         (1, 5, ONE, ONE, 15996),
        "hot_isobaric":    (1, 7, ONE, isobaric_n_q(5000), 5000),
        "hot_floor":       (2, 3, ONE, R.N_FLOOR_Q_LIVE, 1263),
        "hot_below_floor": (2, 5, ONE, 200, 800),
        "hot_eps":         (2, 7, ONE, R.N_EPS_RAW, 400),
        "mix_hot":         (3, 3, Q(0.04), ONE, 8000),   # a_gas = 5.0 x 0.04 = 0.2
        "cool":            (3, 5, ONE, ONE, 150),
        "sub_ambient":     (3, 7, ONE, ONE, -100),
        "hot_no_bulk":     (4, 3, ONE, 0, 5000),
        "hot_inert_only":  (4, 5, 0, ONE, 5000),
        "body_hot":        (4, 7, ONE, ONE, 5000),
    }
    cells = {}
    for name, (y, x, dens, n_q, T_game) in spec.items():
        smoke[y][x] = dens
        n2[y][x] = n_q
        T[y][x] = T_game << 16
        cells[name] = (y, x)
    inert[4][5] = ONE
    y, x = cells["body_hot"]
    d[y][x] = ONE                                       # a marine standing in hot smoke
    gas = [inert, smoke, n2]
    hq = [0, Q(5.0), 0]
    n_bulk = [row[:] for row in n2]
    return a, d, T, ts, gas, hq, n_bulk, cells


def as_i32(plane):
    return np.ascontiguousarray(np.asarray(plane, dtype=np.int64).astype(np.int32))


def his_plane(his, h, w):
    """The reference accepts `his` as an int or a plane; the C++ needs a plane."""
    if isinstance(his, int):
        return [[his] * w for _ in range(h)]
    return his


def ts_from_a(a):
    """The default thermal-solid mask the reference's Scene uses: a > 0."""
    return [[1 if v > 0 else 0 for v in row] for row in a]


def gas_arrays(gas, hq, n_bulk):
    """The gas group as the contiguous int32 arrays the bindings take (or three
    Nones)."""
    if gas is None:
        return None, None, None
    return (np.ascontiguousarray(np.asarray(gas, dtype=np.int64).astype(np.int32)),
            np.ascontiguousarray(np.asarray(hq, dtype=np.int64).astype(np.int32)),
            as_i32(n_bulk))


def cpp_sweep(a, d, k_q, T, his, ts, *, transport="shear", n_ord=16,
              table=None, sweep=None, fleck=True, amb=None,
              gas=None, hq=None, n_bulk=None,
              c_v_q=R.C_V_Q_LIVE, n_floor_q=R.N_FLOOR_Q_LIVE):
    """Run the C++ sweep on a reference-format scene. `k_q` is the UNIFORM leak
    (an int, Q16). `fleck=False` is the reference's `f_plane=None` (undamped)
    configuration. `amb` is the ambient LEVEL the same way the reference takes
    it (thermal v2 R3): None -> E°[0] everywhere, an int -> broadcast, a plane
    -> as given. `gas`/`hq`/`n_bulk` (P5a, all or none) are the smoke term, in
    the reference's list format; with them the sweep's gas Fleck arm (P5b)
    prices every absorbing gas cell in the currency (`c_v_q`, `n_floor_q`,
    the reference's integer forms, handed to the engine through
    engine_currency). Returns (rad_net, rad_flux, rad_amb, rad_fluence, fleck,
    sweep) as int64/int32 numpy arrays plus the sweep object (its min_stream /
    max_stream telemetry and effective planes)."""
    h, w = len(a), len(a[0])
    table = table if table is not None else reference_table()
    sweep = sweep if sweep is not None else bp.RadiationSweep()
    T_a = as_i32(T)
    a_a = as_i32(a)
    d_a = as_i32(d)
    his_a = as_i32(his_plane(his, h, w))
    ts_a = np.ascontiguousarray(np.asarray(ts, dtype=np.int64) != 0)
    rn = np.zeros((h, w), dtype=np.int64)
    rf = np.zeros((h, w), dtype=np.int64)
    ra = np.zeros((h, w), dtype=np.int64)
    rl = np.zeros((h, w), dtype=np.int64)
    amb_a = None if amb is None else np.ascontiguousarray(
        np.asarray(amb_plane(amb, h, w), dtype=np.int64))
    g_a, hq_a, nb_a = gas_arrays(gas, hq, n_bulk)
    cur = {}
    if gas is not None:
        nf, rcv = engine_currency(c_v_q, n_floor_q)
        cur = dict(n_floor_q=nf, recip_cv=rcv)
    sweep.run(T_a, a_a, d_a, his_a, ts_a, table, amb_a, int(T_AMB_Q), int(k_q),
              TRANSPORTS[transport], int(n_ord), rn, rf, ra, rl,
              fleck_enabled=bool(fleck), gas=g_a, heat_absorb_q16=hq_a, n_bulk=nb_a,
              **cur)
    return rn, rf, ra, rl, sweep.fleck_plane(), sweep


def amb_plane(amb, h, w):
    """The ambient LEVEL as a plane: an int broadcasts, a plane passes through.
    The SAME door sweep_ref_q.ambient_plane() is, so a scene written once drives
    both sides of gate 0."""
    if isinstance(amb, int):
        return [[amb] * w for _ in range(h)]
    return amb


def ref_sweep(a, d, k_q, T, his, *, transport="shear", n_ord=16, amb=None,
              ts=None, gas=None, hq=None, n_bulk=None,
              c_v_q=R.C_V_Q_LIVE, n_floor_q=R.N_FLOOR_Q_LIVE, table=None,
              gas_fleck=True):
    """The reference on the SAME scene: its Fleck pre-pass then its sweep.
    `ts` (the thermal-solid mask) selects the pre-pass branch exactly as the
    engine does; `gas`/`hq`/`n_bulk` (P5a, all or none, with `ts`) are the
    smoke term, and the pre-pass's GAS ARM (P5b) reads them in the currency
    (`c_v_q`, `n_floor_q`). `gas_fleck=False` hands the pre-pass no gas group --
    the arm P5a shipped (L = 0 on gas) -- so a gate can show the arm moved
    something. `table` defaults to the reference's own (the resolving scale,
    which reference_table() bakes for the engine). Returns (rad_net, rad_flux,
    rad_amb, rad_fluence, f_plane) as int64 arrays (the reference computes in
    Python ints; every value fits int64 by G11), plus the SweepResult (its
    telemetry and effective planes)."""
    h, w = len(a), len(a[0])
    tbl = R.E if table is None else table
    gas_kw = {}
    if gas is not None:
        gas_kw = dict(gas=gas, heat_absorb_q=hq, n_bulk=n_bulk)
    f = R.fleck_prepass(T, a, his_plane(his, h, w), e_ref=amb, ts=ts, table=tbl,
                        c_v_q=c_v_q, n_floor_q=n_floor_q,
                        **(gas_kw if gas_fleck else {}))
    k = R.plane(h, w, int(k_q))
    res = R.sweep_q(a, d, k, T, n_ord=n_ord, transport=transport, f_plane=f,
                    e_ref=amb, table=tbl, ts=ts, **gas_kw)
    to64 = lambda p: np.asarray(p, dtype=np.int64)   # noqa: E731
    return (to64(res.rad_net), to64(res.rad_flux), to64(res.rad_amb),
            to64(res.rad_fluence), np.asarray(f, dtype=np.int64), res)
