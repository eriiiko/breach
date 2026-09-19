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


def reference_table():
    """An EmissiveTable baked at the reference's own dials (== config.toml's,
    which test_ray_engine_v2_integer_reference.py guards)."""
    tbl = bp.EmissiveTable()
    tbl.rad_scale = R.RAD_SCALE
    tbl.kelvin_ambient = float(R.K_AMB)
    tbl.k_temp_to_kelvin = float(R.K_SLOPE)
    tbl.bake()
    return tbl


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


def cpp_sweep(a, d, k_q, T, his, ts, *, transport="shear", n_ord=16,
              table=None, sweep=None, fleck=True, amb=None):
    """Run the C++ sweep on a reference-format scene. `k_q` is the UNIFORM leak
    (an int, Q16). `fleck=False` is the reference's `f_plane=None` (undamped)
    configuration. `amb` is the ambient LEVEL the same way the reference takes
    it (thermal v2 R3): None -> E°[0] everywhere, an int -> broadcast, a plane
    -> as given. Returns (rad_net, rad_flux, rad_amb, rad_fluence, fleck,
    sweep) as int64/int32 numpy arrays plus the sweep object (its min_stream /
    max_stream telemetry)."""
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
    sweep.run(T_a, a_a, d_a, his_a, ts_a, table, amb_a, int(T_AMB_Q), int(k_q),
              TRANSPORTS[transport], int(n_ord), rn, rf, ra, rl,
              fleck_enabled=bool(fleck))
    return rn, rf, ra, rl, sweep.fleck_plane(), sweep


def amb_plane(amb, h, w):
    """The ambient LEVEL as a plane: an int broadcasts, a plane passes through.
    The SAME door sweep_ref_q.ambient_plane() is, so a scene written once drives
    both sides of gate 0."""
    if isinstance(amb, int):
        return [[amb] * w for _ in range(h)]
    return amb


def ref_sweep(a, d, k_q, T, his, *, transport="shear", n_ord=16, amb=None):
    """The reference on the SAME scene: its Fleck pre-pass then its sweep.
    Returns (rad_net, rad_flux, rad_amb, rad_fluence, f_plane) as int64 arrays
    (the reference computes in Python ints; every value fits int64 by G11)."""
    h, w = len(a), len(a[0])
    f = R.fleck_prepass(T, a, his_plane(his, h, w), e_ref=amb)
    k = R.plane(h, w, int(k_q))
    res = R.sweep_q(a, d, k, T, n_ord=n_ord, transport=transport, f_plane=f,
                    e_ref=amb)
    to64 = lambda p: np.asarray(p, dtype=np.int64)   # noqa: E731
    return (to64(res.rad_net), to64(res.rad_flux), to64(res.rad_amb),
            to64(res.rad_fluence), np.asarray(f, dtype=np.int64), res)
