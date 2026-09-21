"""M1 — THE THREE CONSUMERS OF `heat_inv_shift` THE DESIGN'S §5 LIST MISSES.

The thin-rows design (docs/thin_material_rows_design_2026-09-20.md §5) names
four sites that must learn a negative exponent. There are SEVEN. The three it
does not name are all live, all on the sim path, and all would have been
undefined behaviour the day M2 authors a row below one thermal_mass unit:

  * `radiation_sweep.h::fleck_L_solid_q`  — the Fleck emission damping's `L_q`,
    run once per cell per tick inside the sweep;
  * `combustion.cpp` + `cuda_combustion.cu` — the fuel-bed deposit's OBJECT-SITE
    conversion, which is how a burning crate heats itself;
  * `raycaster.h::rad_pair_budget_s` — the old cast's flux limiter. The old cast
    still runs every tick after T5b's flip (it fills `rad_net` and the light
    channels; only the FOLD moved to `rad_net_sweep`), so this is hot code.

Each is gated here against an oracle that is not the engine: the committed
integer reference for the sweep, and exact arithmetic for the other two.

The shipped material table reaches none of these exponents, so NOTHING ELSE IN
THE SUITE COVERS THEM — which is precisely why they were missed.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp                                       # noqa: E402
from _radiation_sweep_harness import (                            # noqa: E402
    F_ONE, ONE, R, cpp_sweep, ref_sweep, ts_from_a)

FP_ONE = 65536


# ===========================================================================
# 1. The radiation sweep's Fleck emission — gate 0's property, at a NEGATIVE
#    exponent
# ===========================================================================
def _thin_scene(rng, h, w, his_lo, his_hi):
    """Gate 0's scene vocabulary, with the `heat_inv_shift` plane drawn from a
    band that includes NEGATIVE exponents. Deliberately NOT `random_scene`:
    that helper's rng call order is a frozen contract (a committed digest was
    captured through it) and its `his` band is {3, 4, 5}."""
    Q = R.quant
    ts = [[1 if rng.random() < 0.5 else 0 for _ in range(w)] for _ in range(h)]
    a = [[(rng.choice([0, Q(0.3), Q(0.37), Q(0.5), Q(0.91), ONE]) if ts[y][x] else 0)
          for x in range(w)] for y in range(h)]
    d = [[min(ONE, a[y][x] + rng.choice([0, 0, 0, Q(0.5), ONE - a[y][x]]))
          for x in range(w)] for y in range(h)]
    T = [[rng.choice([-(200 << 16), 0, 0, 300 << 16, 1263 << 16, 5000 << 16,
                      15999 << 16, 16000 << 16, 20000 << 16,
                      (32767 << 16) + 65535])
          for _ in range(w)] for _ in range(h)]
    his = [[rng.randint(his_lo, his_hi) for _ in range(w)] for _ in range(h)]
    return a, d, T, his, ts


@pytest.mark.parametrize("seed,h,w", [(1, 7, 9), (2, 9, 11), (3, 5, 5)])
@pytest.mark.parametrize("transport", ["shear", "step"])
@pytest.mark.parametrize("band", [(-16, -1), (-6, 6)])
def test_the_sweep_reproduces_the_reference_at_negative_exponents(
        seed, h, w, transport, band):
    """PROPERTY: the C++ sweep equals the integer reference INTEGER FOR INTEGER
    — all four planes plus the Fleck plane — on scenes whose `heat_inv_shift`
    is negative. This is gate 0's own property, extended over the exponent axis
    that M1 opens.

    The reference is the specification (CLAUDE.md "Integer reference"): it was
    changed FIRST here and its own gates re-run before the C++ was written, so
    it is an oracle independent of the engine's behaviour.

    BREAKS IF: `fleck_L_solid_q` keeps `shr_round0_i64` (a negative shift — UB;
    on x86 a shift by `s & 63`, which yields 0, i.e. NO emission damping at all
    on exactly the thin rows the design is adding); or the sweep's transport
    picks up a second, unsigned copy of the shift.
    """
    rng = random.Random(20260921 + seed)
    a, d, T, his, ts = _thin_scene(rng, h, w, *band)
    got = cpp_sweep(a, d, R.quant(0.10), T, his, ts, transport=transport)
    exp = ref_sweep(a, d, R.quant(0.10), T, his, transport=transport)
    for name, g, e in zip(("rad_net", "rad_flux", "rad_amb", "rad_fluence",
                           "fleck"), got[:5], exp[:5]):
        g64 = np.asarray(g, dtype=np.int64)
        if not np.array_equal(g64, e):
            bad = np.argwhere(g64 != e)
            y, x = bad[0]
            pytest.fail(f"{transport} band={band} seed={seed}: {name} differs at "
                        f"{len(bad)} cells; first ({y},{x}) cpp={g64[y, x]} "
                        f"ref={e[y, x]}")
    rn, rf, ra, rl, fl = (np.asarray(p, dtype=np.int64) for p in got[:5])
    assert np.any(rn != 0) and np.any(rf != 0) and np.any(rl != 0)
    # Non-vacuity on the axis under test: the damping must actually ENGAGE.
    # A negative exponent makes L_q larger, so `f` should be BELOW F_ONE
    # somewhere — if it never is, this scene never exercised the branch.
    assert np.any(fl < F_ONE), "the Fleck damping never engaged"


def test_a_negative_exponent_damps_emission_harder_than_a_positive_one():
    """PROPERTY: the Fleck factor is MONOTONE DECREASING in the cell's capacity
    — a thinner object (more negative `heat_inv_shift`) radiates a larger share
    of its own heat per tick, so it needs MORE damping, so `f` is SMALLER.

    This is the physical meaning of the sign, stated so a future reader cannot
    conclude that the exponent's sign is a formatting detail. It also catches a
    break that a pure C++-vs-reference gate cannot: one where BOTH sides are
    changed the same wrong way.

    BREAKS IF: `fleck_L_solid_q` stops scaling with the exponent (e.g. a clamp
    at 0 sneaks back in, which would make every negative exponent identical).
    """
    tbl = bp.RadiationSweep  # the static entry lives on the class
    table = __import__("_radiation_sweep_harness").reference_table()
    T_q = 1200 << 16
    a_q = R.quant(0.9)
    t_amb_q = 293 << 16
    fs = [bp.RadiationSweep.fleck_f_solid_q24(table, T_q, a_q, his, t_amb_q)
          for his in (5, 3, 0, -2, -4, -8)]
    assert all(b <= a for a, b in zip(fs, fs[1:])), fs
    assert fs[0] > fs[-1], "the exponent had no effect on the damping at all"
    del tbl


# ===========================================================================
# 2. The old raycaster's flux limiter
# ===========================================================================
@pytest.mark.parametrize("his", list(range(-16, 6)))
@pytest.mark.parametrize("shift", [1, 2, 4])
@pytest.mark.parametrize("x", [0, 1, 65535, 1_048_576_000, 2 ** 40 + 7])
def test_the_raycaster_pair_budget_is_the_exact_signed_scaling(his, shift, x):
    """PROPERTY: `rad_pair_budget_s(x, his, shift) == floor(x * 2**his / 2**shift)`
    for a non-negative `x` and EITHER sign of `his` — the limiter budget scales
    with the emitter's own thermal mass, which is the whole point of the term.

    Asserted against Python integers, so the oracle is not the engine.

    BREAKS IF: the site keeps `(x << his) >> shift` — undefined behaviour at a
    negative `his`. On x86-64 that compiles to a shift by `his & 63`, i.e. 63
    for `his = -1`, so an even `x` yields a budget of ZERO: the limiter clamps
    every radiative exchange to nothing and a thin object silently stops
    exchanging heat with the old cast at all.
    """
    assert x >= 0
    assert bp.rad_pair_budget_s(x, his, shift) == (x * 2 ** his) // 2 ** shift


# ===========================================================================
# 3. Combustion's OBJECT-SITE fuel-bed deposit
# ===========================================================================
_COMB_DIALS = dict(
    burn_rate=0.6, o2_thresh_burn=0.05, H_FUEL_M=4.0, H_FUEL_SHIFT=0,
    soot_yield=0.03, fuel_per_o2=0.02, o2_frac_ext=0.10, o2_frac_full=0.21,
    T_MAX_PHYS=16000.0,
)
_O2, _N2, _SMOKE = 0, 1, 2


def _comb_solver():
    c = bp.CombustionSolver()
    for k, v in _COMB_DIALS.items():
        setattr(c, k, int(v) if k == "H_FUEL_SHIFT" else v)
    return c


def _burning_crate(his: int):
    """A burning flammable cell surrounded by PERMEABLE THERMAL SOLIDS —
    furniture's own shape: gas seeps through (`solid` is False) but the tile is
    thermally an object, so combustion's re-sited deposit lands on an
    OBJECT SITE and converts through `heat_inv_shift` instead of through N*c_v.

    `heat_inv_shift = his` on those neighbours is the ONLY thing that varies
    between runs, so the burn and the deposit are identical and the landing is a
    pure function of the exponent.
    """
    h = w = 3
    n_gas = 3
    gas = np.zeros((n_gas, h, w), dtype=np.int32)
    gas[_O2] = int(0.21 * FP_ONE)
    gas[_N2] = int(0.79 * FP_ONE)
    temperature = np.full((h, w), 600 << 16, dtype=np.int32)
    wall_hp = np.full((h, w), 60 * FP_ONE, dtype=np.int32)
    fire = np.zeros((h, w), dtype=np.int32)
    flammable = np.zeros((h, w), dtype=bool)
    solid = np.zeros((h, w), dtype=bool)
    is_vacuum = np.zeros((h, w), dtype=bool)
    ign = np.full((h, w), 300 << 16, dtype=np.int32)
    ts = np.zeros((h, w), dtype=bool)
    shift = np.zeros((h, w), dtype=np.int32)
    c = (1, 1)
    fire[c] = int(0.5 * FP_ONE)
    flammable[c] = True
    ts[:] = True             # every deposit site is an OBJECT site
    shift[:] = his
    return dict(
        gas=np.ascontiguousarray(gas),
        temperature=np.ascontiguousarray(temperature),
        wall_hp=np.ascontiguousarray(wall_hp),
        fire=np.ascontiguousarray(fire),
        flammable=np.ascontiguousarray(flammable),
        solid=np.ascontiguousarray(solid),
        is_vacuum=np.ascontiguousarray(is_vacuum),
        ign=np.ascontiguousarray(ign),
        ts=np.ascontiguousarray(ts),
        shift=np.ascontiguousarray(shift),
    ), c


# The deposit is RE-SITED to the four neighbours (P-O2b, design v5.2), so the
# object-site conversion is observed there, not at the burning cell.
_DEPOSIT_SITE = (0, 1)


def _run_combustion(his: int):
    s = _comb_solver()
    st, c = _burning_crate(his)
    t0 = int(st["temperature"][_DEPOSIT_SITE])
    s.step(st["gas"], _O2, _N2, _SMOKE, st["temperature"], st["wall_hp"],
           st["fire"], st["flammable"], st["solid"], st["is_vacuum"],
           st["ign"], 1.0 / 60.0, 0.0076849, 0.01,
           thermal_solid=st["ts"], heat_inv_shift=st["shift"])
    return int(st["temperature"][_DEPOSIT_SITE]) - t0, int(s.t_max_phys_hits)


@pytest.mark.parametrize("his", [-1, -2, -3, -4])
def test_the_combustion_object_site_deposit_scales_with_the_signed_exponent(his):
    """PROPERTY: the fuel-bed deposit a burning object writes into ITSELF is
    `deposit >> heat_inv_shift`, for either sign — so halving the object's
    thermal mass EXACTLY doubles the temperature rise from the same burn.

    The burn is identical across runs (same O2, same fire, same dials), so the
    ratio is an exact integer relation and needs no model of the deposit's
    magnitude. That makes this an oracle, not a snapshot: it cannot be satisfied
    by recording whatever the engine happened to do.

    THIS PATH IS HOW A BURNING CRATE HEATS ITSELF. It is the mechanism M2's thin
    rows exist to make work, and the design's §5 site list does not mention it.

    BREAKS IF: `combustion.cpp`'s object-site branch keeps `deposit >> shift`
    (UB at a negative exponent; on x86 a shift by `shift & 31`, which yields 0 —
    a burning crate that deposits nothing into itself and goes out).
    """
    base, base_rails = _run_combustion(0)
    got, rails = _run_combustion(his)
    assert base > 0, "the control run must actually deposit heat"
    assert base_rails == 0 and rails == 0, "the rail must not be what bound"
    assert got == base * 2 ** (-his), (
        f"his={his}: rise {got}, expected {base} * {2 ** (-his)} = "
        f"{base * 2 ** (-his)}")
