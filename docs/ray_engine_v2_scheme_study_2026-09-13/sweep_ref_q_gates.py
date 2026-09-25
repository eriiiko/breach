"""P0 GATES for the integer reference (`sweep_ref_q.py`), 2026-09-15 (+ P0b).

Runs the twelve gates of `docs/ray_engine_v2_design_v3_2026-09-15.md` sections
2.8-2.9 and of critique 3 sections 1-4 -- and since P5a / P5b / P5c the gas
term's three (G13, design 6.3's smoke extinction; G14, the gas arm of the Fleck
pre-pass; G15, the temperature fold's gas branch and design 8.4's boundary), and
since P5d the clamp's ceiling (G16, one bucket of headroom) -- prints the
MEASURED numbers (never a bare boolean), and exits non-zero if any fails.

    C:/Users/steen/anaconda3/python.exe sweep_ref_q_gates.py [--fast]

`--fast` shrinks grids and tick counts, never coverage; it is what
`tests/test_ray_engine_v2_integer_reference.py` runs.

EVERY GATE IS PAIRED. A gate that asserts "X is zero" also measures a variant in
which the mechanism is absent and X is NOT zero, so the gate cannot pass because
nothing happened (CLAUDE.md 2026-09-08).
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sweep_ref_q as R  # noqa: E402

ONE = R.ONE
F_ONE = R.F_ONE
Q = R.quant
QF = R.quant_f            # the Q24 door, for an f a gate FORCES

# the scene constants the whole arc uses
T_SRC_GAME = 1263          # the measured crate plateau (K = 1556)
T_IGN_GAME = 280           # furniture ignition_temp  (K = 573)
A_FURNITURE = Q(0.5)       # furniture-class absorptivity
HIS_FURNITURE = 3          # log2(thermal_mass)
K_LEAK = Q(0.10)           # the derived 2.5 m deck leak (dormant in the engine)
# P5d: the over-driven scene's RECEIVER is the furniture row as it SHIPS (a 0.5,
# thermal_mass 0.25 since M2's thin rows), not the P0-era HIS_FURNITURE = 3. At
# alpha floor 0 (P2a, row 39) that P0 crate is UNDAMPED below 1424 game and the
# one-tile 16000-game source holds it near 1110, so the scene stopped over-driving
# at P2a: its clamp hits since then were the bucket-edge pinning P5d removes, and
# under the headroom ceiling it is never clamped at all (G4 measures it). The
# shipped row is Fleck-damped there -- f ~ 0.06 -- which is what over-driven means.
HIS_OVERDRIVEN = -2


def _shipped_pairs():
    """Every DISTINCT (heat_atten, log2(thermal_mass)) pair the shipped material
    table carries, READ FROM config.toml (rows with heat_atten == 0 -- air,
    foliage -- never emit or absorb and are skipped). Several materials share one
    pair, so the names are joined; the arithmetic only ever sees the pair.

    Read, not hardcoded: gate 12's property is about the rows the game SHIPS, so
    a new material row must be able to fail it.
    """
    by_pair = {}
    for name, a_q, his, _atten, _tm in R.shipped_absorbing_rows():
        by_pair.setdefault((a_q, his), []).append(name)
    return tuple((a_q, his, " / ".join(names)) for (a_q, his), names in by_pair.items())


SHIPPED_ROWS = _shipped_pairs()
# The corner NO material ships: thermal_mass = 1 (his = 0) with full absorptivity.
# It is the mechanism gate 12 names -- the wobble is f_q's own resolution, so it
# grows as the thermal mass falls -- and it is why P0's proposed ingress rule
# (heat_atten > 0 => thermal_mass >= 8) became optional once f went to Q24.
PATHOLOGICAL_ROW = (ONE, 0, "a = 1, thermal_mass = 1 (no material ships this)")


def _rand_scene(rng, h, w, *, bodies=True, hot=True):
    a = [[rng.choice([0, Q(0.37), Q(0.5), Q(0.91), ONE]) for _ in range(w)]
         for _ in range(h)]
    if bodies:
        d = [[min(ONE, a[y][x] + rng.choice([0, 0, 0, Q(0.5), ONE - a[y][x]]))
              for x in range(w)] for y in range(h)]
    else:
        d = [row[:] for row in a]
    T = [[rng.choice([0, 0, 300 << 16, T_SRC_GAME << 16, 5000 << 16, 15999 << 16]
                     if hot else [0]) for _ in range(w)] for _ in range(h)]
    f = [[rng.choice([F_ONE, QF(0.7), QF(0.05)]) for _ in range(w)] for _ in range(h)]
    return a, d, T, f


# --------------------------------------------------------------------------- #
# P5a: gas cells that absorb (design 6.3). The scene vocabulary every gas gate
# shares: four gas planes in the engine's roles -- a SMOKE-like absorber, a
# STEAM-like weak one, and the BULK pair (O2, N2) that carries no optics -- the
# bulk sum as n_bulk (the engine's own derivation), and the thermal-solid mask
# as the cells whose material extinction is non-zero.
# --------------------------------------------------------------------------- #
O2_AMB, N2_AMB = 13763, 51773          # the P1 calibration split of one ambient cell
GAS_HQ = [Q(5.0), Q(0.3), 0, 0]        # smoke, steam, o2, n2 -- smoke SATURATES at 0.2


def _gas_planes(rng, h, w, ts):
    """Random gas planes over a scene: smoke from none to thick (so a_gas spans 0,
    thin, and the ONE cap), steam, the ambient bulk pair -- and some cells with NO
    bulk (below N_EPS_RAW: a breached cell) and one at EXACTLY N_EPS_RAW (the floor's
    edge, which must still absorb)."""
    smoke = [[rng.choice([0, 0, Q(0.001), Q(0.02), Q(0.06), Q(0.27), ONE, 3 * ONE])
              for _ in range(w)] for _ in range(h)]
    steam = [[rng.choice([0, 0, Q(0.5)]) for _ in range(w)] for _ in range(h)]
    o2 = [[O2_AMB] * w for _ in range(h)]
    n2 = [[N2_AMB] * w for _ in range(h)]
    for _ in range(max(1, (h * w) // 10)):
        y, x = rng.randrange(h), rng.randrange(w)
        o2[y][x] = n2[y][x] = 0                       # vacuum-like: no bulk
    y, x = rng.randrange(h), rng.randrange(w)
    o2[y][x], n2[y][x] = 0, R.N_EPS_RAW               # the floor's edge
    gas = [smoke, steam, o2, n2]
    n_bulk = [[o2[y][x] + n2[y][x] for x in range(w)] for y in range(h)]
    return gas, list(GAS_HQ), n_bulk


def _gas_kw(rng, a):
    """The four sweep_q gas arguments for a scene whose thermal solids are a > 0."""
    h, w = len(a), len(a[0])
    ts = [[1 if a[y][x] > 0 else 0 for x in range(w)] for y in range(h)]
    gas, hq, n_bulk = _gas_planes(rng, h, w, ts)
    return dict(gas=gas, heat_absorb_q=hq, n_bulk=n_bulk, ts=ts)


def _absorbing_gas_cells(res, ts):
    """Cells that are GAS (not thermal solid) and absorb through the smoke term."""
    return [(y, x) for y in range(len(ts)) for x in range(len(ts[0]))
            if not ts[y][x] and res.a_eff[y][x] > 0]


# --------------------------------------------------------------------------- #
def gate1_conservation(fast=False):
    """G1. Sum(rad_net) + Sum(rad_flux) + Sum(rad_amb) == 0, exactly, in int64.

    Breaks if: any writer books one side of a transfer without the other (the
    push->gather translation, the remainder split, the body's re-emission, the
    virtual ring's two books).
    """
    lines, ok = [], True
    h, w = (7, 8) if fast else (9, 11)
    rng = random.Random(20260915)
    for transport in ("shear", "step"):
        for kleak in (0, K_LEAK):
            for n_ord in (16, 12):
                a, d, T, f = _rand_scene(rng, h, w)
                k = R.plane(h, w, kleak)
                res = R.sweep_q(a, d, k, T, n_ord=n_ord, transport=transport,
                                f_plane=f)
                sn, sf, sa = res.sums()
                ident = sn + sf + sa
                nonzero = (sn != 0) and (sf != 0) and (sa != 0)
                good = (ident == 0) and nonzero
                ok &= good
                lines.append(
                    f"  {transport:5s} k={kleak:5d} S{n_ord:2d}: "
                    f"net={sn:>16d} flux={sf:>15d} amb={sa:>15d} "
                    f"identity={ident}  each-nonzero={nonzero}  "
                    f"{'OK' if good else 'FAIL'}")
    # PER-CELL AMBIENT (thermal v2 R3). The identity is structural -- every
    # integer that leaves the stream is booked once with each sign -- so it must
    # survive an ambient that varies cell to cell, WITH the leak live. A cold
    # patch (a hull breach: ambient level 0) beside a room-temperature interior.
    for transport in ("shear", "step"):
        for n_ord in (16, 12):
            a, d, T, f = _rand_scene(rng, h, w)
            k = R.plane(h, w, K_LEAK)
            amb = [[0 if (x < w // 3) else R.E0 for x in range(w)]
                   for _ in range(h)]
            res = R.sweep_q(a, d, k, T, n_ord=n_ord, transport=transport,
                            f_plane=f, e_ref=amb)
            sn, sf, sa = res.sums()
            ident = sn + sf + sa
            nonzero = (sn != 0) and (sf != 0) and (sa != 0)
            good = (ident == 0) and nonzero
            ok &= good
            lines.append(
                f"  {transport:5s} k={K_LEAK:5d} S{n_ord:2d} COLD-SKY PATCH: "
                f"net={sn:>16d} flux={sf:>15d} amb={sa:>15d} "
                f"identity={ident}  each-nonzero={nonzero}  "
                f"{'OK' if good else 'FAIL'}")
    # P5a: GAS CELLS THAT ABSORB (design 6.3). The identity is structural -- the
    # smoke term only changes WHICH a and d a cell reads -- so it must survive
    # absorbing and emitting gas cells, bodies standing in smoke, cells below the
    # N_EPS floor, the ONE cap, the leak on and off. NON-VACUOUS: a gas cell
    # absorbs somewhere and books a non-zero rad_net.
    for transport in ("shear", "step"):
        for kleak in (0, K_LEAK):
            for n_ord in (16, 12):
                a, d, T, f = _rand_scene(rng, h, w)
                gkw = _gas_kw(rng, a)
                k = R.plane(h, w, kleak)
                res = R.sweep_q(a, d, k, T, n_ord=n_ord, transport=transport,
                                f_plane=f, **gkw)
                sn, sf, sa = res.sums()
                ident = sn + sf + sa
                cells = _absorbing_gas_cells(res, gkw["ts"])
                gas_booked = sum(1 for (y, x) in cells if res.rad_net[y][x] != 0)
                nonzero = (sn != 0) and (sf != 0) and (sa != 0)
                good = (ident == 0) and nonzero and gas_booked > 0
                ok &= good
                lines.append(
                    f"  {transport:5s} k={kleak:5d} S{n_ord:2d} WITH SMOKE: "
                    f"net={sn:>16d} flux={sf:>15d} amb={sa:>15d} "
                    f"identity={ident}  absorbing gas cells={len(cells):>2} "
                    f"(booking rad_net: {gas_booked:>2})  {'OK' if good else 'FAIL'}")
    return ok, lines


def gate2a_uniform_ambient(fast=False):
    """G2a. A uniform ambient field is a PER-CELL exact fixed point of all three
    planes -- with bodies present, the leak on, non-dyadic a, both transports,
    S16 and S12. Holds for any f BY CONSTRUCTION (the excess is zero at ambient),
    so forcing f < 1 here exercises nothing and is not claimed as the gate.

    Breaks if: the body stops re-emitting at ambient (row 25), the ceiling return
    loses its `amb_m`-first association, the ring stops returning amb_m, or the
    Fleck factor multiplies the whole emission instead of the excess (row 22).
    """
    lines, ok = [], True
    h, w = (7, 8) if fast else (8, 10)
    rng = random.Random(4)
    a = [[rng.choice([0, Q(0.37), Q(0.91), ONE]) for _ in range(w)] for _ in range(h)]
    d = [[min(ONE, a[y][x] + rng.choice([0, Q(0.5), ONE - a[y][x]]))
          for x in range(w)] for y in range(h)]
    n_bodies = sum(1 for y in range(h) for x in range(w) if d[y][x] > a[y][x])
    k = R.plane(h, w, K_LEAK)
    T = R.plane(h, w, 0)
    f = R.plane(h, w, QF(0.3))
    for transport in ("shear", "step"):
        for n_ord in (16, 12):
            res = R.sweep_q(a, d, k, T, n_ord=n_ord, transport=transport, f_plane=f)
            nz = sum(1 for p in (res.rad_net, res.rad_flux, res.rad_amb)
                     for row in p for v in row if v != 0)
            ok &= (nz == 0)
            lines.append(f"  {transport:5s} S{n_ord:2d}: nonzero cells over the "
                         f"three planes = {nz}  (bodies on {n_bodies} cells, "
                         f"k=0.10, f=0.3)  {'OK' if nz == 0 else 'FAIL'}")
    # NON-VACUITY (a): the pre-ruling pure-sink body breaks the same scene.
    res = R.sweep_q(a, d, k, T, transport="shear", f_plane=f, body_mode="sink")
    nz_sink = sum(1 for p in (res.rad_net, res.rad_flux, res.rad_amb)
                  for row in p for v in row if v != 0)
    ok &= (nz_sink > 0)
    lines.append(f"  non-vacuity: body_mode='sink' (pre-ruling) on the same scene "
                 f"-> {nz_sink} nonzero cells, min rad_net = "
                 f"{min(v for row in res.rad_net for v in row)}")
    # NON-VACUITY (b): the v2 whole-emission Fleck form breaks it too (row 22).
    amb_m = (R.E0 * (ONE // 16)) >> 16
    src_v2 = (amb_m * QF(0.995)) >> R.F_SHIFT
    net_v2 = ((amb_m * Q(0.91)) >> 16) - ((src_v2 * Q(0.91)) >> 16)
    ok &= (net_v2 != 0)
    lines.append(f"  non-vacuity: v2 whole-emission Fleck (f=0.995, a=0.91) gives "
                 f"rad_net = {net_v2} counts per ordinate on an ambient cell "
                 f"(the excess form gives 0)")
    # NON-VACUITY (c), thermal v2 R3: the fixed point above is a fixed point OF
    # THE UNIFORM AMBIENT. Give a third of the grid a cold sky (ambient level 0,
    # every temperature left at ambient) and the SAME scene must stop being a
    # fixed point -- a 293 K wall facing 0 K genuinely radiates. A hoisted
    # `amb_m` reproduces the uniform answer here (zeros everywhere), which is
    # the bug this gate exists to make visible.
    amb_cold = [[0 if (x < w // 3) else R.E0 for x in range(w)] for _ in range(h)]
    res_c = R.sweep_q(a, d, k, T, transport="shear", f_plane=f, e_ref=amb_cold)
    nz_cold = sum(1 for p in (res_c.rad_net, res_c.rad_flux, res_c.rad_amb)
                  for row in p for v in row if v != 0)
    cold_net = min(v for row in res_c.rad_net for v in row)
    ok &= (nz_cold > 0 and cold_net < 0)
    lines.append(f"  non-vacuity: a cold-sky patch (ambient level 0 on x < {w // 3}, "
                 f"every T still at ambient) -> {nz_cold} nonzero cells, "
                 f"min rad_net = {cold_net} (a 293 K wall facing 0 K radiates; "
                 f"a HOISTED amb_m gives 0 everywhere)")
    # P5a: SMOKE AT AMBIENT is still a per-cell exact fixed point -- an absorbing
    # gas cell at ambient absorbs exactly the ambient stream it re-emits, by the
    # same integers (its excess is zero), with bodies standing in the smoke. Holds
    # BY CONSTRUCTION of the excess form, as the solid case does.
    rng_g = random.Random(40)
    gkw = _gas_kw(rng_g, a)
    for transport in ("shear", "step"):
        for n_ord in (16, 12):
            res = R.sweep_q(a, d, k, T, n_ord=n_ord, transport=transport,
                            f_plane=f, **gkw)
            nz = sum(1 for p in (res.rad_net, res.rad_flux, res.rad_amb)
                     for row in p for v in row if v != 0)
            cells = _absorbing_gas_cells(res, gkw["ts"])
            body_in_smoke = sum(1 for (y, x) in cells if res.d_eff[y][x] > res.a_eff[y][x])
            good = (nz == 0) and len(cells) > 0 and body_in_smoke > 0
            ok &= good
            lines.append(f"  {transport:5s} S{n_ord:2d} WITH SMOKE: nonzero cells = {nz}  "
                         f"(absorbing gas cells {len(cells)}, bodies standing in smoke "
                         f"{body_in_smoke}, k=0.10, f=0.3)  {'OK' if good else 'FAIL'}")
    # non-vacuity (d): ONE hot smoke cell in the same ambient scene breaks it --
    # so the smoke term really is in the arithmetic above
    (gy, gx) = _absorbing_gas_cells(
        R.sweep_q(a, d, k, T, f_plane=f, **gkw), gkw["ts"])[0]
    T_hot = [row[:] for row in T]
    T_hot[gy][gx] = T_SRC_GAME << 16
    res_h = R.sweep_q(a, d, k, T_hot, transport="shear", f_plane=f, **gkw)
    nz_h = sum(1 for p in (res_h.rad_net, res_h.rad_flux, res_h.rad_amb)
               for row in p for v in row if v != 0)
    ok &= (nz_h > 0 and res_h.rad_net[gy][gx] < 0)
    lines.append(f"  non-vacuity: one smoke cell at {T_SRC_GAME} game in the same scene -> "
                 f"{nz_h} nonzero cells, its own rad_net = {res_h.rad_net[gy][gx]} "
                 f"(hot smoke radiates)")
    return ok, lines


def gate2b_isothermal_box(fast=False):
    """G2b. An enclosed isothermal box: 2-cell-thick opaque walls at T0 = 1263,
    transparent interior. The INNER wall layer is a per-cell exact zero with
    f < 1 FORCED (0.3), both transports, S16 and S12. The outer layer legitimately
    cools to the ambient sky; its minimum is reported.

    Breaks if: emission and absorption stop being the same arithmetic (the
    `(E*w)*a` association, critique 1 fix 4), or the split stops carrying the
    remainder, or f is applied per-ordinate inconsistently.
    """
    lines, ok = [], True
    n = 10 if fast else 12
    a = R.plane(n, n, 0)
    d = R.plane(n, n, 0)
    k = R.plane(n, n, 0)
    T = R.plane(n, n, 0)
    for y in range(n):
        for x in range(n):
            if y < 2 or y >= n - 2 or x < 2 or x >= n - 2:
                a[y][x] = ONE
                d[y][x] = ONE
                T[y][x] = T_SRC_GAME << 16
    f = R.plane(n, n, QF(0.3))
    for transport in ("shear", "step"):
        for n_ord in (16, 12):
            res = R.sweep_q(a, d, k, T, n_ord=n_ord, transport=transport, f_plane=f)
            rn = res.rad_net
            inner = [rn[y][x] for y in range(n) for x in range(n)
                     if (y in (1, n - 2) and 1 <= x <= n - 2)
                     or (x in (1, n - 2) and 1 <= y <= n - 2)]
            outer = [rn[y][x] for y in range(n) for x in range(n)
                     if y in (0, n - 1) or x in (0, n - 1)]
            interior = [rn[y][x] for y in range(2, n - 2) for x in range(2, n - 2)]
            mi = max(abs(v) for v in inner)
            ok &= (mi == 0) and (max(abs(v) for v in interior) == 0) and (min(outer) < 0)
            lines.append(f"  {transport:5s} S{n_ord:2d}: inner layer max|rad_net| = "
                         f"{mi}   outer layer min = {min(outer)}   interior air "
                         f"max|rad_net| = {max(abs(v) for v in interior)}  "
                         f"{'OK' if mi == 0 else 'FAIL'}")
    # P5a: the SAME box FILLED WITH SMOKE at T0 -- thin, thick and saturated cells,
    # one below the N_EPS floor. A smoke cell in an isothermal cavity receives the
    # walls' damped source on every ordinate and re-emits it by the same
    # integers, so every INNER cell -- wall layer and smoke alike -- stays an exact
    # zero with f < 1 forced on all of them.
    ts = [[1 if a[y][x] > 0 else 0 for x in range(n)] for y in range(n)]
    smoke = R.plane(n, n, 0)
    o2 = R.plane(n, n, 0)
    n2 = R.plane(n, n, 0)
    T_f = [row[:] for row in T]
    dens = (Q(0.02), Q(0.06), Q(0.27), ONE)
    for y in range(2, n - 2):
        for x in range(2, n - 2):
            smoke[y][x] = dens[(y + x) % len(dens)]
            o2[y][x], n2[y][x] = O2_AMB, N2_AMB
            T_f[y][x] = T_SRC_GAME << 16
    o2[2][2] = n2[2][2] = 0                         # one breached cell: no bulk
    gkw = dict(gas=[smoke, R.plane(n, n, 0), o2, n2], heat_absorb_q=list(GAS_HQ),
               n_bulk=[[o2[y][x] + n2[y][x] for x in range(n)] for y in range(n)],
               ts=ts)
    for transport in ("shear", "step"):
        for n_ord in (16, 12):
            res = R.sweep_q(a, d, k, T_f, n_ord=n_ord, transport=transport,
                            f_plane=f, **gkw)
            rn = res.rad_net
            inner = [rn[y][x] for y in range(1, n - 1) for x in range(1, n - 1)]
            outer = [rn[y][x] for y in range(n) for x in range(n)
                     if y in (0, n - 1) or x in (0, n - 1)]
            smoky = sum(1 for y in range(2, n - 2) for x in range(2, n - 2)
                        if res.a_eff[y][x] > 0)
            mi = max(abs(v) for v in inner)
            good = (mi == 0) and (min(outer) < 0) and smoky > 0
            ok &= good
            lines.append(f"  {transport:5s} S{n_ord:2d} FILLED WITH SMOKE at {T_SRC_GAME}: "
                         f"every inner cell max|rad_net| = {mi} over {smoky} absorbing "
                         f"smoke cells   outer layer min = {min(outer)}  "
                         f"{'OK' if good else 'FAIL'}")
    return ok, lines


def gate3_positivity(fast=False):
    """G3. No negative stream anywhere; and the reference REJECTS a > d, d > ONE
    and k > ONE (the design's ingress invariants) with a raise.

    Breaks if: absorption stops being bounded by a + b <= ONE, or a scene with
    an illegal extinction is allowed to be measured at all.
    """
    lines, ok = [], True
    h, w = (7, 8) if fast else (9, 11)
    rng = random.Random(7)
    worst_min = None
    max_stream = 0
    for transport in ("shear", "step"):
        for kleak in (0, K_LEAK):
            a, d, T, f = _rand_scene(rng, h, w)
            k = R.plane(h, w, kleak)
            res = R.sweep_q(a, d, k, T, transport=transport, f_plane=f)
            worst_min = res.min_stream if worst_min is None else min(worst_min, res.min_stream)
            max_stream = max(max_stream, res.max_stream)
    ok &= (worst_min >= 0) and (max_stream > 0)
    lines.append(f"  min stream over all four runs = {worst_min}  "
                 f"(max stream = {max_stream}, so the scenes are not empty)  "
                 f"{'OK' if worst_min >= 0 else 'FAIL'}")
    # the rejections, each paired with the legal scene that must be accepted
    good = R.plane(2, 2, Q(0.5))
    R.validate_planes(good, R.plane(2, 2, ONE), R.plane(2, 2, 0))   # must not raise
    cases = [("a > d", R.plane(2, 2, Q(0.9)), R.plane(2, 2, Q(0.5)), R.plane(2, 2, 0)),
             ("d > ONE", good, R.plane(2, 2, ONE + 1), R.plane(2, 2, 0)),
             ("k > ONE", good, R.plane(2, 2, ONE), R.plane(2, 2, ONE + 1))]
    for name, aa, dd, kk in cases:
        try:
            R.validate_planes(aa, dd, kk)
            raised = False
        except ValueError:
            raised = True
        ok &= raised
        lines.append(f"  ingress rejects {name}: {'raised' if raised else 'ACCEPTED -- FAIL'}")
    # P5a: positivity WITH SMOKE -- a_gas <= ONE by its cap and d = max(d, a) keep
    # a + b <= ONE on every gas cell, so the stream stays non-negative.
    worst_g = None
    n_cells = 0
    for transport in ("shear", "step"):
        for kleak in (0, K_LEAK):
            a, d, T, f = _rand_scene(rng, h, w)
            gkw = _gas_kw(rng, a)
            res = R.sweep_q(a, d, R.plane(h, w, kleak), T, transport=transport,
                            f_plane=f, **gkw)
            worst_g = res.min_stream if worst_g is None else min(worst_g, res.min_stream)
            n_cells += len(_absorbing_gas_cells(res, gkw["ts"]))
    ok &= (worst_g >= 0) and n_cells > 0
    lines.append(f"  WITH SMOKE: min stream over four runs = {worst_g} "
                 f"({n_cells} absorbing gas cells)  {'OK' if worst_g >= 0 else 'FAIL'}")
    # ...and the gas extinction's own ingress rejections (validate_gas), each paired
    # with the legal scene that must be accepted.
    a2 = R.plane(2, 2, 0)
    d2 = R.plane(2, 2, 0)
    k2 = R.plane(2, 2, 0)
    T2 = R.plane(2, 2, 0)
    ts2 = R.plane(2, 2, 0)
    g2 = [R.plane(2, 2, Q(0.5)), R.plane(2, 2, ONE)]
    nb2 = R.plane(2, 2, ONE)
    R.sweep_q(a2, d2, k2, T2, gas=g2, heat_absorb_q=[Q(1.0), R.HEAT_ABSORB_Q_MAX],
              n_bulk=nb2, ts=ts2)                                    # must not raise
    bad = [("heat_absorb < 0", dict(gas=g2, heat_absorb_q=[-1, 0], n_bulk=nb2, ts=ts2)),
           ("heat_absorb > max", dict(gas=g2, heat_absorb_q=[R.HEAT_ABSORB_Q_MAX + 1, 0],
                                      n_bulk=nb2, ts=ts2)),
           ("one heat_absorb per plane", dict(gas=g2, heat_absorb_q=[0], n_bulk=nb2, ts=ts2)),
           ("more planes than the headroom covers",
            dict(gas=[R.plane(2, 2, 0)] * (R.N_GAS_PLANES_MAX + 1),
                 heat_absorb_q=[0] * (R.N_GAS_PLANES_MAX + 1), n_bulk=nb2, ts=ts2)),
           ("a gas plane of the wrong shape", dict(gas=[R.plane(2, 3, 0), g2[1]],
                                                   heat_absorb_q=[0, 0], n_bulk=nb2, ts=ts2)),
           ("gas without the thermal-solid mask", dict(gas=g2, heat_absorb_q=[0, 0],
                                                       n_bulk=nb2))]
    for name, kw in bad:
        try:
            R.sweep_q(a2, d2, k2, T2, **kw)
            raised = False
        except ValueError:
            raised = True
        ok &= raised
        lines.append(f"  ingress rejects {name}: {'raised' if raised else 'ACCEPTED -- FAIL'}")
    return ok, lines


def _marine_scene(body_mode):
    """The scene critique 3 section 4f found: a 7x7 ambient room, a wall column at
    a = 0.9, one body (d = ONE on air) one cell from the wall."""
    h = w = 7
    a = R.plane(h, w, 0)
    d = R.plane(h, w, 0)
    k = R.plane(h, w, 0)
    T = R.plane(h, w, 0)
    for y in range(h):
        a[y][6] = Q(0.9)
        d[y][6] = Q(0.9)
    d[3][5] = ONE                       # the marine, on air, beside the wall
    return R.Scene(a=a, d=d, k=k, T=T, his=6, body_mode=body_mode)


def _overdriven_scene(src_game=16000, gap=1, n=9, crate_his=HIS_OVERDRIVEN):
    """A held source one tile from a cold crate (design gate 4's over-drive).

    P5d: the crate is the SHIPPED furniture row (a 0.5, his -2 -- HIS_OVERDRIVEN
    has the why); the source keeps HIS_FURNITURE, which sets its own Fleck
    damping and so the fluence the crate sees -- unchanged from P0.
    `crate_his=HIS_FURNITURE` rebuilds the P0-era scene, which G4 measures."""
    a = R.plane(n, n, 0)
    d = R.plane(n, n, 0)
    k = R.plane(n, n, 0)
    T = R.plane(n, n, 0)
    sy, sx = n // 2, n // 3
    a[sy][sx] = ONE
    d[sy][sx] = ONE
    T[sy][sx] = src_game << 16
    cy, cx = sy, sx + gap + 1
    a[cy][cx] = A_FURNITURE
    d[cy][cx] = A_FURNITURE
    held = R.plane(n, n, 0)
    held[sy][sx] = 1
    his = R.plane(n, n, HIS_FURNITURE)
    his[cy][cx] = crate_his
    sc = R.Scene(a=a, d=d, k=k, T=T, his=his, held=held)
    return sc, (cy, cx)


def gate4_counters(fast=False):
    """G4. Per counter (design row 31). On the marine-beside-an-ambient-wall scene
    the low rail and the clamp are BOTH silent over 24 ticks; on the over-driven
    scene the clamp counter is non-zero and the T_MAX_PHYS rail stays silent
    (the clamp's ceiling saturates one LSB below 16000, below the rail).

    P5d: the over-driven crate is the SHIPPED furniture row, which the source
    genuinely over-drives (Fleck-damped, f ~ 0.06); the P0-era crate
    (HIS_FURNITURE = 3) is undamped there and is MEASURED beside it -- under the
    headroom ceiling it is not clamped at all, which is the ceiling working as
    ruled (an undamped cell needs no clamp), and why the receiver changed.

    Breaks if: the body stops re-emitting at ambient (the pure-sink variant, run
    below, trips the low rail every tick on every wall cell), or the clamp stops
    engaging on an over-driven scene (its tick count is set from the MEASURED
    first-hit tick, see the comment there -- this gate must never report a silent
    clamp as a pass).
    """
    lines, ok = [], True
    ticks = 8 if fast else 24
    sc = _marine_scene("reemit")
    res = sc.run(ticks)
    c = sc.counters
    wall = [res.rad_net[y][6] for y in range(7)]
    silent = c.t_low_rail_hits == 0 and c.rad_clamp_hits == 0 and c.t_max_phys_hits == 0
    ok &= silent
    lines.append(f"  marine + ambient wall, {ticks} ticks, body re-emits: "
                 f"low_rail={c.t_low_rail_hits} clamp={c.rad_clamp_hits} "
                 f"t_max={c.t_max_phys_hits}   wall rad_net = {wall}  "
                 f"{'OK' if silent else 'FAIL'}")
    sc_s = _marine_scene("sink")
    res_s = sc_s.run(ticks)
    cs = sc_s.counters
    ok &= (cs.t_low_rail_hits > 0)
    lines.append(f"  non-vacuity: the pre-ruling PURE SINK body on the same scene "
                 f"-> low_rail={cs.t_low_rail_hits} in {ticks} ticks, wall rad_net = "
                 f"{[res_s.rad_net[y][6] for y in range(7)]}")
    # positive assertion: the re-emitting body still SENSES a fire
    h = w = 7
    a = R.plane(h, w, 0)
    d = R.plane(h, w, 0)
    k = R.plane(h, w, 0)
    T = R.plane(h, w, 0)
    a[3][1] = ONE
    d[3][1] = ONE
    T[3][1] = T_SRC_GAME << 16
    d[3][3] = ONE
    res_f = R.sweep_q(a, d, k, T)
    flux = res_f.rad_flux[3][3]
    ok &= (flux > 0)
    lines.append(f"  positive: a re-emitting body one tile from a {T_SRC_GAME}-game "
                 f"fire books rad_flux = {flux} counts (> 0, so the sensor lives)")
    # Over-driven: the clamp must engage, the T_MAX_PHYS rail must not. The crate
    # starts cold and CLIMBS to its cap, so this sub-check has its own tick count.
    # History, measured: with the P0-era crate (his 3) at alpha floor 0 (P2a, row
    # 39) the first hit fell at tick 11 (the fast mode's 8 ticks would have passed
    # VACUOUSLY -- the reason this count is separate). P5d: that crate is undamped
    # and the headroom ceiling never clamps it (measured below); the shipped
    # furniture crate is over-driven from its first tick -- one explicit step from
    # cold would carry it to ~6100 game -- and is held at the ceiling.
    o_ticks = 12 if fast else 24
    sc_o, crate = _overdriven_scene()
    sc_o.run(o_ticks)
    co = sc_o.counters
    good = co.rad_clamp_hits > 0 and co.t_max_phys_hits == 0 and co.t_low_rail_hits == 0
    ok &= good
    lines.append(f"  over-driven (16000-game source one tile from a cold crate, the "
                 f"shipped furniture row), {o_ticks} ticks: clamp={co.rad_clamp_hits} "
                 f"t_max={co.t_max_phys_hits} low_rail={co.t_low_rail_hits}, crate T = "
                 f"{sc_o.T[crate[0]][crate[1]] / 65536:.1f} game  "
                 f"{'OK' if good else 'FAIL'}")
    sc_p, crate_p = _overdriven_scene(crate_his=HIS_FURNITURE)
    sc_p.run(o_ticks)
    f_p = R.fleck_f_solid_q(sc_p.T[crate_p[0]][crate_p[1]], A_FURNITURE, HIS_FURNITURE)[0]
    lines.append(f"  measured beside it: the P0-era crate (his {HIS_FURNITURE}) on the same "
                 f"scene is undamped (f = {f_p / F_ONE:.4f}) and settles at "
                 f"{sc_p.T[crate_p[0]][crate_p[1]] / 65536:.1f} game with "
                 f"{sc_p.counters.rad_clamp_hits} clamp hits -- the headroom ceiling does not "
                 f"touch a cell radiation cannot over-drive")
    return ok, lines


def gate5_maximum_principle(fast=False):
    """G5. Immediately after the radiative sub-step, on that sub-step alone:
    T_new <= max(T_before, e_ceiling_q(Phi)) for every cell -- the clamp's
    ceiling since P5d (the top of the first bucket out-emitting Phi; it was
    E_inv(Phi), that bucket's predecessor's low edge) -- and a COOLING step is
    never clipped: wherever the unclamped step does not raise the cell, T_new is
    that step exactly.

    Non-vacuity: the clamp engages on the over-driven scene (hits > 0, asserted);
    (a) with the clamp disabled the same cell climbs past the ceiling
    (reported); (b) a burning crate (T_before far above its ceiling) is NOT
    clamped -- the bare v2 form would have slammed it down to the ceiling, and
    that number is reported beside it.
    """
    lines, ok = [], True
    ticks = 8 if fast else 24
    sc, crate = _overdriven_scene()
    worst_excess = None
    cooling_steps = cooling_clipped = 0
    for _ in range(ticks):
        f = R.fleck_prepass(sc.T, sc.a, sc.his)
        res = R.sweep_q(sc.a, sc.d, sc.k, sc.T, transport=sc.transport, f_plane=f)
        before = [row[:] for row in sc.T]
        R.fold_pass1_solid(sc.T, res.rad_net, res.rad_fluence, sc.his, sc.ts,
                           sc.counters, held=sc.held)
        for y in range(len(sc.T)):
            for x in range(len(sc.T[0])):
                if not sc.ts[y][x] or sc.held[y][x]:
                    continue
                cap = max(before[y][x], R.e_ceiling_q(res.rad_fluence[y][x]))
                ex = sc.T[y][x] - cap
                worst_excess = ex if worst_excess is None else max(worst_excess, ex)
                t_after = R.sat_add_q16(before[y][x], R.shr_round0_signed(
                    res.rad_net[y][x], sc.his[y][x]))
                if t_after <= before[y][x]:
                    cooling_steps += 1
                    cooling_clipped += int(sc.T[y][x] != t_after)
    good = (worst_excess is not None and worst_excess <= 0
            and sc.counters.rad_clamp_hits > 0 and cooling_clipped == 0)
    ok &= good
    lines.append(f"  clamped, {ticks} ticks: worst (T_new - max(T_before, e_ceiling(Phi))) "
                 f"= {worst_excess} counts  (<= 0 required)  clamp hits = "
                 f"{sc.counters.rad_clamp_hits} (> 0 required)  {'OK' if good else 'FAIL'}")
    t_clamped = sc.T[crate[0]][crate[1]]
    # (a) the same scene with the clamp disabled
    sc2, crate2 = _overdriven_scene()
    sc2.run(ticks * 4, clamp_enabled=False)
    t_unclamped = sc2.T[crate2[0]][crate2[1]]
    ok &= (t_unclamped > t_clamped)
    # the crate's Phi as the fold saw it: the sweep WITH the pre-pass's Fleck
    # plane (P0's line read an undamped sweep here, whose "cap" was 8724 game)
    phi_c = R.sweep_q(sc.a, sc.d, sc.k, sc.T, f_plane=R.fleck_prepass(
        sc.T, sc.a, sc.his)).rad_fluence[crate[0]][crate[1]]
    lines.append(f"  non-vacuity (a): clamp_enabled=False, {ticks*4} ticks -> crate "
                 f"{t_unclamped / 65536:.1f} game vs clamped {t_clamped / 65536:.1f} "
                 f"game (ceiling = {R.e_ceiling_q(phi_c) / 65536:.2f}, E_inv(Phi) = "
                 f"{R.e_inv_q(phi_c) >> 16})")
    # the 0-D equilibrium the design quotes, where the geometric factor is 0.25
    phi = int(0.25 * R.E[R.e_bucket_of(16000 << 16)])
    t_run, _ = R.cell_march(0, phi, A_FURNITURE, HIS_FURNITURE, 400 * 24,
                            clamp_enabled=False, rails_enabled=False, int32_sat=False)
    t_true = R.e_inv_q(phi) >> 16
    ok &= (t_run / 65536 > 1e6)
    lines.append(f"  non-vacuity (a'): the 0-D G=0.25 model the design quotes -- "
                 f"Fleck-only equilibrium = {t_run / 65536:,.0f} game vs the radiation "
                 f"temperature E_inv(Phi) = {t_true} game (the clamp holds it at or below "
                 f"{R.e_ceiling_q(phi) / 65536:.2f})")
    # (b) a burning crate must NOT be clamped
    n = 7
    a = R.plane(n, n, A_FURNITURE)
    d = [row[:] for row in a]
    k = R.plane(n, n, 0)
    T = R.plane(n, n, 0)
    T[3][3] = T_SRC_GAME << 16          # a crate held hot by COMBUSTION, not radiation
    sc3 = R.Scene(a=a, d=d, k=k, T=T, his=HIS_FURNITURE)
    f = R.fleck_prepass(sc3.T, sc3.a, sc3.his)
    res3 = R.sweep_q(sc3.a, sc3.d, sc3.k, sc3.T, f_plane=f)
    before = T[3][3]
    before3 = [row[:] for row in sc3.T]
    R.fold_pass1_solid(sc3.T, res3.rad_net, res3.rad_fluence, sc3.his, sc3.ts,
                       sc3.counters)
    for y in range(n):
        for x in range(n):
            t_after = R.sat_add_q16(before3[y][x], R.shr_round0_signed(
                res3.rad_net[y][x], sc3.his))
            if t_after <= before3[y][x]:
                cooling_steps += 1
                cooling_clipped += int(sc3.T[y][x] != t_after)
    cap_bare = R.e_ceiling_q(res3.rad_fluence[3][3])
    bound = ((sc3.counters.rad_clamp_hits == 0) and (sc3.T[3][3] < before)
             and cap_bare < before and cooling_steps > 0 and cooling_clipped == 0)
    ok &= bound
    lines.append(f"  non-vacuity (b): a burning crate at {before >> 16} game, "
                 f"ceiling(Phi) = {cap_bare / 65536:.2f} game -> corrected clamp does not "
                 f"bind (hits={sc3.counters.rad_clamp_hits}, T {before / 65536:.1f} -> "
                 f"{sc3.T[3][3] / 65536:.1f}); the bare v2 form would have set it to "
                 f"{cap_bare / 65536:.2f} game in one tick. Over both scenes, cooling "
                 f"steps clipped: {cooling_clipped} of {cooling_steps}  "
                 f"{'OK' if bound else 'FAIL'}")
    return ok, lines


def _point_fluence(transport, half, n_grid, kleak=K_LEAK, e_ref=None):
    """The fluence field of a square fire of half-width `half` at T_SRC_GAME."""
    a = R.plane(n_grid, n_grid, 0)
    d = R.plane(n_grid, n_grid, 0)
    k = R.plane(n_grid, n_grid, kleak)
    T = R.plane(n_grid, n_grid, 0)
    c = n_grid // 2
    for y in range(c - half, c + half + 1):
        for x in range(c - half, c + half + 1):
            a[y][x] = ONE
            d[y][x] = ONE
            T[y][x] = T_SRC_GAME << 16
    return R.sweep_q(a, d, k, T, transport=transport, e_ref=e_ref).rad_fluence, c


def footprint(fl, c, phi_ign, n_dirs=64, rmax=None):
    """The ignition-footprint radius in `n_dirs` directions: where the fluence
    crosses E°[ignition_temp]. Returns (mean, max/min, radii)."""
    n = len(fl)
    if rmax is None:
        rmax = c - 1
    out = []
    for kdir in range(n_dirs):
        th = 2 * math.pi * kdir / n_dirs
        dx, dy = math.cos(th), math.sin(th)
        pr = pv = None
        r = 0.5
        while r < rmax:
            yy = int(round(c + dy * r))
            xx = int(round(c + dx * r))
            if not (0 <= yy < n and 0 <= xx < n):
                break
            v = fl[yy][xx]
            if pv is not None and pv >= phi_ign > v:
                t = (pv - phi_ign) / (pv - v)
                out.append(pr + t * (r - pr))
                break
            pr, pv = r, v
            r += 0.25
    if not out:
        return float("nan"), float("nan"), out
    return sum(out) / len(out), max(out) / min(out), out


def ring_stats(fl, c, radii):
    """max/min of the fluence around each ring (the ripple statistic)."""
    n = len(fl)
    out = {}
    for rr in radii:
        vals = [fl[y][x] for y in range(n) for x in range(n)
                if abs(math.hypot(y - c, x - c) - rr) <= 0.5]
        out[rr] = max(vals) / max(min(vals), 1)
    return out


def gate6_isotropy(fast=False):
    """G6. Ring MAX/MIN and the 64-direction ignition footprint spread, shear vs
    step, for a 1-tile / 3x3 / 5x5 source, leak on at k = 0.10.

    The asserted property is that SHEAR IS ROUNDER THAN STEP for a single burning
    tile and for a 5x5 fire -- the reason the design gives for "heat takes shear".
    It is NOT asserted at 3x3, because it is not true there under the design's own
    ambient ring (see report_p0.md; the design's 1.22-vs-1.32 was measured with
    the ambient inflow off, and the ordering flips when it is on).

    Breaks if: the transport constants s_m change, or the half-offset ordinate
    set is replaced by one with ordinates on an axis (step's four-point cross
    becomes a pencil and its spread jumps).
    """
    lines, ok = [], True
    n_grid = 41 if fast else 81
    phi_ign = R.E[R.e_bucket_of(T_IGN_GAME << 16)]
    lines.append(f"  grid {n_grid}x{n_grid}, source {T_SRC_GAME} game, ignition "
                 f"fluence E°[{T_IGN_GAME}] = {phi_ign}, leak k = 0.10, "
                 f"ambient ring ON (the design's configuration)")
    got = {}
    for transport in ("shear", "step"):
        for half, nm in ((0, "1 tile"), (1, "3x3"), (2, "5x5")):
            fl, c = _point_fluence(transport, half, n_grid)
            mean_r, spread, _ = footprint(fl, c, phi_ign)
            rings = ring_stats(fl, c, (1, 2, 3, 5, 8))
            got[(transport, half)] = (mean_r, spread)
            ok &= math.isfinite(spread) and spread >= 1.0
            lines.append(f"  {transport:5s} {nm:>6}: mean r_ign = {mean_r:6.2f} tiles, "
                         f"footprint spread (max/min) = {spread:5.3f}, ring max/min "
                         + " ".join(f"r{r}={v:5.2f}" for r, v in rings.items()))
    rounder = all(got[("shear", hf)][1] < got[("step", hf)][1] for hf in (0, 2))
    ok &= rounder
    lines.append(f"  shear rounder than step at 1 tile "
                 f"({got[('shear', 0)][1]:.3f} < {got[('step', 0)][1]:.3f}) and at 5x5 "
                 f"({got[('shear', 2)][1]:.3f} < {got[('step', 2)][1]:.3f}): {rounder}")
    lines.append(f"  NOT asserted at 3x3: shear {got[('shear', 1)][1]:.3f} vs step "
                 f"{got[('step', 1)][1]:.3f} -- the ordering the design quotes holds "
                 f"only with the ambient ring OFF (report_p0.md section 6)")
    # NON-VACUITY: the integer gather reproduces the FLOAT PUSH footprint, live,
    # in the float study's own configuration (no ambient, relative threshold).
    import numpy as np
    from sweep_ref import sweep as push_step, sweep_shear as push_shear
    g_ign = (573.0 / 1556.0) ** 4
    e_src = float(R.E[R.e_bucket_of(T_SRC_GAME << 16)])
    for transport, fn in (("shear", push_shear), ("step", push_step)):
        fl_i, c = _point_fluence(transport, 0, n_grid, e_ref=0)
        m_i, s_i, _ = footprint(fl_i, c, g_ign * e_src)
        af = np.zeros((n_grid, n_grid))
        ef = np.zeros((n_grid, n_grid))
        af[c, c] = 1.0
        ef[c, c] = 1.0
        fl_f = fn(af, ef, kleak=K_LEAK / ONE, E_amb=0.0, n=16)[1]
        m_f, s_f, _ = footprint(fl_f.tolist(), c, g_ign)
        close = abs(m_i - m_f) < 0.02 and abs(s_i - s_f) < 0.02
        ok &= close
        lines.append(f"  vs the float push reference, 1 tile, no ambient: integer "
                     f"r={m_i:.3f} spread={s_i:.3f}  float r={m_f:.3f} spread={s_f:.3f}"
                     f"  {'OK' if close else 'FAIL'}")
    return ok, lines


def gate7_float_agreement(fast=False):
    """G7. The integer GATHER form agrees with the float PUSH reference.

    Two comparisons: (a) against the committed `sweep_ref.py` itself, on a scene
    with no bodies and f = 1, where the two are algebraically identical; (b)
    against a push port carrying the body terms, on a scene with bodies.

    Breaks if: the gather form's recomputed split stops reproducing the push
    form's faces (the remainder convention), or the virtual ring stops
    reproducing sweep_ref.py's boundary seeding.
    """
    import numpy as np
    from sweep_ref import sweep as push_step, sweep_shear as push_shear

    lines, ok = [], True
    h, w = (7, 7) if fast else (9, 9)
    rng = random.Random(11)
    for transport in ("shear", "step"):
        a, d, T, _ = _rand_scene(rng, h, w, bodies=False)
        k = R.plane(h, w, K_LEAK)
        res = R.sweep_q(a, d, k, T, transport=transport)
        af = np.array([[a[y][x] / ONE for x in range(w)] for y in range(h)])
        Ef = np.array([[float(R.E[R.e_bucket_of(T[y][x])]) for x in range(w)]
                       for y in range(h)])
        fn = push_shear if transport == "shear" else push_step
        dE = fn(af, Ef, kleak=K_LEAK / ONE, E_amb=float(R.E0), n=16)[0]
        diff = max(abs(res.rad_net[y][x] - dE[y][x]) for y in range(h) for x in range(w))
        mag = res.max_abs_net
        ratio = diff / mag
        ok &= (ratio < 1e-4)
        lines.append(f"  {transport:5s} vs sweep_ref.py (no bodies): "
                     f"max|int - float| = {diff:12.1f} counts, max|rad_net| = {mag}, "
                     f"ratio = {ratio:.2e}  {'OK' if ratio < 1e-4 else 'FAIL'}")
    for transport in ("shear", "step"):
        a, d, T, _ = _rand_scene(rng, h, w, bodies=True)
        k = R.plane(h, w, K_LEAK)
        res = R.sweep_q(a, d, k, T, transport=transport)
        dE = R.sweep_push_float(a, d, k, T, transport=transport)
        diff = max(abs(res.rad_net[y][x] - dE[y][x]) for y in range(h) for x in range(w))
        mag = res.max_abs_net
        ratio = diff / mag
        ok &= (ratio < 1e-4)
        lines.append(f"  {transport:5s} vs the push port WITH bodies: "
                     f"max|int - float| = {diff:12.1f} counts, max|rad_net| = {mag}, "
                     f"ratio = {ratio:.2e}  {'OK' if ratio < 1e-4 else 'FAIL'}")
    return ok, lines


def gate8_fleck_form(fast=False):
    """G8. f_q24 = floordiv((T_abs<<24), max(T_abs, 4L)) equals the float
    1/(1 + alpha*g) with alpha = max(0, 1 - 1/g) to within one Q24 count over
    the WHOLE table, and f_q24 == 2^24 EXACTLY whenever 4*L_q <= T_abs_q.

    ALPHA FLOOR 0 (design row 39, RULED 2026-09-16; P2a). The iff is the whole
    content of the ruling: wherever the explicit material update is provably
    monotone (g <= 1, i.e. 4L <= T_abs) the emission is UNDAMPED, exactly, so a
    burning tile held at its plateau by combustion radiates full black body
    there. The undamped ceiling per shipped row is printed below (1068 game for
    wood, 1424 for the a = 0.5 rows -- P0b, report_p0.md section 0.8a). Above
    the branch both floors are the SAME arm (1 + alpha*g == g) and bit-identical.

    Q24, not Q16 (design row 32, P0b): the damped source `f*(E°[T] - E°[0])` is not
    monotone in T when f carries only 38 counts at the table top -- gate 12 is that
    property, and this gate is its accuracy half. The Q16 form is measured below
    beside the Q24 one, so "one count" cannot silently mean the wider one.

    Breaks if: alpha's max() is re-expanded wrongly (the superseded floor of one
    half puts `T_abs + 2L` in the denominator and makes f < 2^24 at EVERY L > 0 --
    measured below, so the iff is not vacuous), the denominator moves off the
    ABSOLUTE temperature, the division stops being the kit's exact floordiv, or f
    goes back to Q16 (the per-count error grows 256x and blows the bound).
    """
    lines, ok = [], True
    worst = 0.0
    worst_at = None
    stride = 1          # the whole table either way: 20 000 probes is free
    pairs = [(a, h) for a, h, _ in SHIPPED_ROWS] + [PATHOLOGICAL_ROW[:2]]
    iff_ok = True
    n_undamped = n_damped = 0
    for a_q, his in pairs:
        for b in range(0, R.E_TABLE_SIZE, stride):
            T_game = 4 * b
            T_q = T_game << 16
            f_q, L_q = R.fleck_f_solid_q(T_q, a_q, his)
            f_f = R.fleck_f_float(float(T_game), L_q / 65536.0)
            err = abs(f_q / F_ONE - f_f)
            if err > worst:
                worst, worst_at = err, (T_game, a_q, his)
            if not (0 < f_q <= F_ONE):
                ok = False
            # the RULED floor's iff: undamped exactly on the stable side of g = 1
            stable = (4 * L_q <= T_q + (R.K_AMB << 16))
            iff_ok &= ((f_q == F_ONE) == stable)
            n_undamped += int(stable)
            n_damped += int(not stable)
    ok &= (worst <= 1.0 / F_ONE) and iff_ok
    # non-vacuity of the iff: the probe must contain BOTH sides of g = 1, and the
    # superseded floor must disagree on the undamped side (it damps everywhere).
    half_damps_stable = all(
        R.fleck_f_q((4 * b) << 16, R.fleck_f_solid_q((4 * b) << 16, a_q, his)[1],
                    alpha_floor=R.ALPHA_FLOOR_HALF) < F_ONE
        for a_q, his in pairs for b in range(1, R.E_TABLE_SIZE, 97)
        if R.fleck_f_solid_q((4 * b) << 16, a_q, his)[1] > 0)
    ok &= (n_undamped > 0) and (n_damped > 0) and half_damps_stable
    lines.append(f"  worst |f_q24/2^24 - f_float| over {R.E_TABLE_SIZE // stride} "
                 f"buckets x {len(pairs)} (a, his) pairs = {worst:.3e}  (one Q24 count "
                 f"= {1/F_ONE:.3e}) at T={worst_at[0]} a={worst_at[1]} his={worst_at[2]}  "
                 f"{'OK' if worst <= 1/F_ONE else 'FAIL'}")
    lines.append(f"  f_q24 == 2^24 iff 4L <= T_abs (RULED floor 0, row 39): {iff_ok} "
                 f"-- {n_undamped} undamped and {n_damped} damped probes, both sides "
                 f"present; the superseded floor of one half damps every L > 0 probe "
                 f"on the same rows: {half_damps_stable}")
    # where each shipped row stops being undamped -- g = 1, the row-39 number
    for a_q, his, nm in SHIPPED_ROWS:
        top = max((4 * b for b in range(R.E_TABLE_SIZE)
                   if R.fleck_f_solid_q((4 * b) << 16, a_q, his)[0] == F_ONE),
                  default=None)
        lines.append(f"    undamped (f == 2^24) up to {top:>6} game: {nm}")
    # the SAME measurement on the rejected Q16 form, so the bound above is not a
    # bound on nothing: one Q16 count is 256x coarser and the same probe says so.
    worst16 = max(abs(R.fleck_f_solid_q((4 * b) << 16, a_q, his, shift=16)[0] / ONE
                      - R.fleck_f_float(float(4 * b),
                                        R.fleck_f_solid_q((4 * b) << 16, a_q, his)[1] / 65536.0))
                  for a_q, his in pairs for b in range(0, R.E_TABLE_SIZE, 8))
    ok &= (worst16 > worst)
    lines.append(f"  the rejected Q16 form on the same probe: worst = {worst16:.3e} "
                 f"(one Q16 count = {1/ONE:.3e}) -- {worst16/max(worst, 1e-18):.0f}x this "
                 f"gate's Q24 error, which is what row 32 bought")
    for T_game in (0, 280, 1263, 1800, 15996):
        f_q, L_q = R.fleck_f_solid_q(T_game << 16, Q(0.5), 3)
        g = 4.0 * (L_q / 65536.0) / (T_game + R.K_AMB)
        lines.append(f"    T={T_game:6d} a=0.5 his=3: L_q={L_q / 65536:12.2f} game  "
                     f"g={g:10.3f}  f_q24={f_q:9d} ({f_q / F_ONE:.7f})  "
                     f"float={R.fleck_f_float(float(T_game), L_q / 65536.0):.7f}")
    return ok, lines


def gate9_e_inv(fast=False):
    """G9. E_inv is idempotent (E°[E_inv(Phi)] <= Phi) including at the table's
    edges, returns 0 below E°[0], and saturates at 15996 game -- BELOW
    T_MAX_PHYS = 16000, so the clamp binds first on the radiative sub-step.

    Breaks if: the search returns a bucket's high edge, or the 12-trip count stops
    covering 4000 buckets, or the sub-E°[0] case is left undefined.
    """
    lines, ok = [], True
    probes = [R.E[0], R.E[0] + 1, R.E[17] - 1, R.E[17], R.E[2000] + 12345,
              R.E[3999] - 1, R.E[3999], R.E[3999] * 3]
    idem = True
    for phi in probes:
        Tc = R.e_inv_q(phi)
        idem &= (R.E[R.e_bucket_of(Tc)] <= phi)
    ok &= idem
    lines.append(f"  E°[E_inv(Phi)] <= Phi over {len(probes)} boundary probes "
                 f"(incl. E°[3999] and 3xE°[3999]): {idem}")
    below = R.e_inv_q(R.E[0] - 1)
    ok &= (below == 0)
    lines.append(f"  E_inv(E°[0] - 1) = {below}  (design row 31: 0, no radiative "
                 f"warming above the ambient floor)")
    sat = R.e_inv_q(R.E[3999] * 3) >> 16
    ok &= (sat == R.T_TABLE_TOP_GAME) and (sat < (R.T_MAX_PHYS_Q >> 16))
    lines.append(f"  E_inv(3 x E°[3999]) = {sat} game  (table top {R.T_TABLE_TOP_GAME}, "
                 f"T_MAX_PHYS {R.T_MAX_PHYS_Q >> 16}; the clamp saturates BELOW the rail)")
    # non-vacuity: the inverse is not a constant -- it tracks the table
    step = 1            # every bucket either way
    mono = all(R.e_inv_q(R.E[b]) <= R.e_inv_q(R.E[b + step])
               for b in range(0, R.E_TABLE_SIZE - step, step))
    exact = all(R.e_inv_q(R.E[b]) == (4 * b) << 16 for b in range(0, R.E_TABLE_SIZE, step))
    ok &= mono and exact
    lines.append(f"  E_inv(E°[b]) == 4b for every probed bucket: {exact}; monotone: {mono}")
    return ok, lines


def gate10_stability(fast=False):
    """G10. The stability and equilibrium tables re-run on the NEW forms (the
    excess-form Fleck factor and the corrected clamp) at the RULED alpha floor 0.

    THE PROPERTY, in two halves (design row 39, P2a):
      (a) BELOW g = 1 the damping is OFF -- the trajectory from 1263 game is the
          explicit one to the LAST COUNT, and its cost is plain forward Euler's
          -5.7 % against the analytic bath solution (P0b, report_p0.md 0.8b).
          The superseded floor of one half is measured on the same march (+0.23 %)
          so "equals explicit" is a statement about the floor, not about a probe
          that cannot tell two laws apart.
      (b) ABOVE it the damping still does its whole job: from 2500 and 16 000 game
          the explicit update rails to zero and the damped one does not, and the
          clamp still pins every equilibrium within [T_cont - 1 bucket, T_cont + 2
          buckets] of the continuous one, T_cont -- where Fleck alone (the damped
          5000- and 16 000-game rows) runs far outside that band. P5d: the
          ceiling is the top of the first bucket out-emitting Phi, so a held cell
          sits up to two buckets above E_inv(Phi) where P0 pinned it exactly ON
          E_inv(Phi); the "true (E_inv)" column is printed as the measured
          radiation temperature, never as a pin.

    Breaks if: the Fleck factor stops damping above g = 1 (explicit overshoots and
    the low rail fires), or alpha's floor moves back off 0 (the fire-range
    trajectory stops matching explicit and the un-clamped 1263-game equilibrium
    jumps from +0.15 % to +4.6 %), or the clamp is dropped from the fold (the
    damped rows leave the band -- the Fleck-only column, asserted outside it).
    """
    lines, ok = [], True
    A, HIS = A_FURNITURE, HIS_FURNITURE
    phi_amb = R.E0
    n = int(0.5 * R.TICK_HZ)
    # --- cooling, the OLD law (zero sky, whole emission) as an instrument check.
    # At floor 0 this start is still below g = 1 (the ambient part of the emission
    # moves g by 0.13 %), so the two columns coincide here too.
    e_old, _ = R.cell_march(T_SRC_GAME << 16, 0, A, HIS, n, e_ref=0, fleck=False,
                            clamp_enabled=False)
    f_old, _ = R.cell_march(T_SRC_GAME << 16, 0, A, HIS, n, e_ref=0, fleck=True,
                            clamp_enabled=False)
    x_old = R.cool_exact_zero_sky(float(T_SRC_GAME), 0.5, 0.5, HIS)
    lines.append(f"  cooling 1263 game for 0.5 s, OLD law (zero sky, whole emission, "
                 f"the instrument check): explicit {e_old / 65536:7.1f} "
                 f"({(e_old / 65536 - x_old) / x_old * 100:+5.2f}%)  Fleck "
                 f"{f_old / 65536:7.1f} ({(f_old / 65536 - x_old) / x_old * 100:+5.2f}%)"
                 f"  analytic {x_old:7.1f}")
    # --- cooling, the NEW law (ambient bath, excess form), floor 0 vs the
    # superseded floor of one half.
    e_new, ce = R.cell_march(T_SRC_GAME << 16, phi_amb, A, HIS, n, fleck=False,
                             clamp_enabled=False)
    f_new, cf = R.cell_march(T_SRC_GAME << 16, phi_amb, A, HIS, n, fleck=True,
                             clamp_enabled=False)
    h_new, _ch = R.cell_march(T_SRC_GAME << 16, phi_amb, A, HIS, n, fleck=True,
                              clamp_enabled=False, alpha_floor=R.ALPHA_FLOOR_HALF)
    x_new = R.cool_exact_bath(float(T_SRC_GAME), 0.5, 0.5, HIS)
    err_e = (e_new / 65536 - x_new) / x_new * 100
    err_f = (f_new / 65536 - x_new) / x_new * 100
    err_h = (h_new / 65536 - x_new) / x_new * 100
    is_euler = (f_new == e_new)
    half_differs = (h_new != e_new)
    ok &= is_euler and half_differs
    lines.append(f"  cooling 1263 game for 0.5 s, NEW law (ambient bath, excess form): "
                 f"explicit {e_new / 65536:7.1f} ({err_e:+5.2f}%)  Fleck@floor0 "
                 f"{f_new / 65536:7.1f} ({err_f:+5.2f}%)  analytic {x_new:7.1f}   "
                 f"{'OK (g = 0.74 < 1: floor 0 IS forward Euler, count for count)' if is_euler else 'FAIL (floor 0 must not damp below g = 1)'}")
    lines.append(f"    the superseded floor of one half on the SAME march: "
                 f"{h_new / 65536:7.1f} ({err_h:+5.2f}%) -- different from explicit: "
                 f"{half_differs}; that difference IS what floor 0 gives up on free "
                 f"cooling, and what it buys a DRIVEN source is gate 8's iff")
    # --- explicit really is unstable above ~1800 game, and Fleck really is not
    starts = (2500, 16000) if fast else (1800, 2500, 5000, 16000)
    for T0 in starts:
        e, cce = R.cell_march(T0 << 16, phi_amb, A, HIS, n, fleck=False,
                              clamp_enabled=False)
        f, ccf = R.cell_march(T0 << 16, phi_amb, A, HIS, n, fleck=True,
                              clamp_enabled=False)
        x = R.cool_exact_bath(float(T0), 0.5, 0.5, HIS)
        lines.append(f"    from {T0:6d} game, 0.5 s: explicit {e / 65536:9.1f} "
                     f"(low-rail hits {cce.t_low_rail_hits})  Fleck {f / 65536:9.1f} "
                     f"(hits {ccf.t_low_rail_hits})  analytic {x:8.1f}")
        if T0 >= 2500:
            ok &= (cce.t_low_rail_hits > 0) and (ccf.t_low_rail_hits == 0)
    # --- monotone, positive, finite from every start up to the table top
    stride = 200 if fast else 100
    bad = []
    for b in range(0, R.E_TABLE_SIZE, stride):
        T0 = 4 * b
        tr, cc = R.cell_march(T0 << 16, phi_amb, A, HIS, 48, fleck=True, trace=True)
        if any(v < 0 for v in tr) or any(tr[i + 1] > tr[i] for i in range(len(tr) - 1)):
            bad.append(T0)
    ok &= not bad
    lines.append(f"  monotone + positive + finite from every start in "
                 f"0..{R.T_TABLE_TOP_GAME} game (stride {4*stride}), 48 ticks: "
                 f"{'all clean' if not bad else 'VIOLATIONS at ' + str(bad)}")
    # --- what the damped source COSTS. Its shape (monotone in T) is gate 12's
    # property since P0b; here only the rate cost at the table top is quoted,
    # because that is the number the cooling story above is paid in.
    top = R.E[R.E_TABLE_SIZE - 1]
    damped_top = R.damped_source_q(R.T_TABLE_TOP_GAME << 16, A, HIS)
    lines.append(f"  at the table top the damped source is {damped_top:,} vs the "
                 f"undamped {top:,} counts (x{top / damped_top:.0f} less) -- the RATE "
                 f"cost of the stability fix, paid by a cell that keeps the energy and "
                 f"cools slower, not by energy going missing (its MONOTONICITY in T is "
                 f"gate 12)")
    # --- the equilibrium table (the 0-D model stability_study.py section 7 used)
    iters = (40 if fast else 400) * 24
    lines.append(f"  equilibrium under a held fluence Phi = G x E°[T_src], G = 0.25 "
                 f"({iters} ticks). The un-clamped column is the SCHEME'S BIAS, not a "
                 f"sweep prediction (design row 34):")
    lines.append(f"    {'T_src':>8}{'E_inv(Phi)':>14}{'T_cont':>14}"
                 f"{'Fleck only, 0':>16}{'Fleck only, 1/2':>17}"
                 f"{'Fleck+clamp':>13}{'clamp hits':>12}   (clamped within "
                 f"[T_cont - 4, T_cont + 8] required)")
    bias_0 = bias_h = None
    for T_src in (1263, 5000, 16000):
        phi = int(0.25 * R.E[R.e_bucket_of(T_src << 16)])
        t_true = R.e_inv_q(phi) >> 16
        t_true_f = R.e_inv_float(phi)
        t_f, _ = R.cell_march(0, phi, A, HIS, iters, clamp_enabled=False,
                              rails_enabled=False, int32_sat=False)
        t_h, _ = R.cell_march(0, phi, A, HIS, iters, clamp_enabled=False,
                              rails_enabled=False, int32_sat=False,
                              alpha_floor=R.ALPHA_FLOOR_HALF)
        t_c, cc = R.cell_march(0, phi, A, HIS, iters, clamp_enabled=True)
        # P5d: a PROPERTY, not P0's snapshot `(t_c >> 16) == E_inv(Phi)`: the
        # clamped equilibrium lies within one bucket below and two above the
        # continuous one -- and where Fleck damps (the two hotter rows), Fleck
        # alone does not (the clamp is what holds it there)
        in_band = (t_true_f - 4.0) <= t_c / 65536 <= (t_true_f + 8.0)
        runaway = t_f / 65536 > t_true_f + 8.0
        good = in_band and (runaway if T_src > T_SRC_GAME else True)
        ok &= good
        if T_src == T_SRC_GAME:
            bias_0 = (t_f / 65536 - t_true_f) / t_true_f * 100
            bias_h = (t_h / 65536 - t_true_f) / t_true_f * 100
        lines.append(f"    {T_src:>8}{t_true:>14}{t_true_f:>14.1f}"
                     f"{t_f / 65536:>16,.1f}{t_h / 65536:>17,.1f}"
                     f"{t_c / 65536:>13.1f}{cc.rad_clamp_hits:>12}"
                     f"  {'OK' if good else 'FAIL'}")
    # The result row 39 leans on: in the FIRE RANGE floor 0's own un-clamped bias
    # is nearly gone, where the superseded floor sits ~30x further out. Above the
    # fire range the two floors are bit-identical and both need the clamp equally,
    # so floor 0 weakens the clamp's case nowhere.
    fire_range_better = (abs(bias_0) < 1.0) and (abs(bias_h) > 4.0)
    ok &= fire_range_better
    lines.append(f"  un-clamped bias at the 1263-game source: floor 0 {bias_0:+.2f} % vs "
                 f"floor 1/2 {bias_h:+.2f} % of the continuous truth  "
                 f"{'OK (floor 0 removes the maximum-principle violation in the one regime the clamp was not already carrying)' if fire_range_better else 'FAIL'}")
    return ok, lines


def gate11_headroom(fast=False):
    """G11. Max stream and max |rad_net| on a scene seeded at the table top stay
    below 2^46 (the design's per-cell bound), and no intermediate product
    approaches 2^63 -- INCLUDING the Q24 Fleck product `ex_m * f_q24`, which is
    the widest one now that f is Q24 (design row 32, P0b): it is measured at both
    S16 and S12, because the ordinate weight (and so `ex_m`) grows as S falls.

    Breaks if: the ordinate weight, the table's top, the number of ordinates or
    the Fleck factor's fixed point grows enough to need more than int64 -- or
    (#78) the live table's FINE_BITS grows past 11 (at 12 the pre-pass's a * ex
    crosses 2^58; at 13 the Fleck product crosses 2^63), or the gas chain's
    conversion moves off its final narrow (measured: it would then overflow).
    """
    lines, ok = [], True
    h, w = (7, 8) if fast else (9, 11)
    a = R.plane(h, w, ONE)
    d = R.plane(h, w, ONE)
    k = R.plane(h, w, K_LEAK)
    T = R.plane(h, w, R.T_TABLE_TOP_GAME << 16)
    for transport in ("shear", "step"):
        for n_ord in (16, 12):
            res = R.sweep_q(a, d, k, T, transport=transport, n_ord=n_ord)
            lg = lambda v: math.log2(max(abs(v), 1))  # noqa: E731
            good = (res.max_abs_net < 2 ** 46 and res.max_fluence < 2 ** 46
                    and res.max_product < 2 ** 63
                    and res.max_fleck_product < 2 ** 63)
            ok &= good
            lines.append(f"  {transport:5s} S{n_ord:2d} at the table top "
                         f"({R.T_TABLE_TOP_GAME} game, a = 1, f = 2^24 undamped -- the "
                         f"worst case): max stream = 2^{lg(res.max_stream):.1f}, "
                         f"max|rad_net| = 2^{lg(res.max_abs_net):.1f}, max fluence = "
                         f"2^{lg(res.max_fluence):.1f}, max product = "
                         f"2^{lg(res.max_product):.1f}, max (ex_m * f_q24) = "
                         f"2^{lg(res.max_fleck_product):.2f}  (bounds: sums < 2^46, "
                         f"products < 2^63)  {'OK' if good else 'FAIL'}")
    # the case where the stream TRAVELS: a transparent room with table-top emitters
    a2 = R.plane(h, w, 0)
    d2 = R.plane(h, w, 0)
    T2 = R.plane(h, w, 0)
    for (y, x) in ((1, 1), (h - 2, w - 2), (h // 2, 1)):
        a2[y][x] = ONE
        d2[y][x] = ONE
        T2[y][x] = R.T_TABLE_TOP_GAME << 16
    for transport in ("shear", "step"):
        res = R.sweep_q(a2, d2, R.plane(h, w, 0), T2, transport=transport)
        lg = lambda v: math.log2(max(abs(v), 1))  # noqa: E731
        good = res.max_abs_net < 2 ** 46 and res.max_fluence < 2 ** 46
        ok &= good
        lines.append(f"  {transport:5s} three table-top emitters in a TRANSPARENT room "
                     f"(the stream travels): max stream = 2^{lg(res.max_stream):.1f}, "
                     f"max|rad_net| = 2^{lg(res.max_abs_net):.1f}, max fluence = "
                     f"2^{lg(res.max_fluence):.1f}  {'OK' if good else 'FAIL'}")
    lines.append(f"  E°[3999] = {R.E[3999]} = 2^{math.log2(R.E[3999]):.1f}; "
                 f"amb_m(S16) = {(R.E0 * (ONE // 16)) >> 16}; the stream cannot exceed "
                 f"the largest upstream source by more than one count per cell "
                 f"(critique 3 section 2e)")
    # P5a: THE GAS DENSITY SUM -- the one new product. Its worst case is every
    # plane the headroom argument covers at the door's maximum heat_absorb and the
    # largest int32 density; the engine accumulates it in plain int64.
    worst = R.gas_density_sum([R.INT32_MAX] * R.N_GAS_PLANES_MAX,
                              [R.HEAT_ABSORB_Q_MAX] * R.N_GAS_PLANES_MAX)
    lg = lambda v: math.log2(max(abs(v), 1))  # noqa: E731
    good = worst < 2 ** 63
    ok &= good
    # ...and the bound is the right one: ONE more doubling of it would NOT fit, so
    # the door's 4096 is the arithmetic's own limit, not a guess with slack in it.
    doubled = R.gas_density_sum([R.INT32_MAX] * R.N_GAS_PLANES_MAX,
                                [2 * R.HEAT_ABSORB_Q_MAX] * R.N_GAS_PLANES_MAX)
    good = good and doubled >= 2 ** 63
    ok &= good
    lines.append(f"  gas density sum, {R.N_GAS_PLANES_MAX} planes x heat_absorb_q = "
                 f"2^{lg(R.HEAT_ABSORB_Q_MAX):.0f} x N = INT32_MAX: 2^{lg(worst):.4f} "
                 f"(< 2^63 required); at twice the door's bound it would be "
                 f"2^{lg(doubled):.4f} (does not fit)  {'OK' if good else 'FAIL'}")
    # and a smoke-filled room at the table top: a gas cell at a_gas = ONE is, to the
    # stream, an a = 1 solid, so the per-cell bounds above must hold unchanged.
    ts = R.plane(h, w, 0)
    smoke = R.plane(h, w, 3 * ONE)
    o2 = R.plane(h, w, O2_AMB)
    n2 = R.plane(h, w, N2_AMB)
    n_bulk = [[o2[y][x] + n2[y][x] for x in range(w)] for y in range(h)]
    for transport in ("shear", "step"):
        res = R.sweep_q(R.plane(h, w, 0), R.plane(h, w, 0), R.plane(h, w, K_LEAK), T,
                        transport=transport, gas=[smoke, o2, n2],
                        heat_absorb_q=[R.HEAT_ABSORB_Q_MAX, 0, 0], n_bulk=n_bulk, ts=ts)
        opaque = all(v == ONE for row in res.a_eff for v in row)
        good = (opaque and res.max_abs_net < 2 ** 46 and res.max_fluence < 2 ** 46
                and res.max_product < 2 ** 63)
        ok &= good
        lines.append(f"  {transport:5s} a room of OPAQUE SMOKE at the table top (a_gas = ONE "
                     f"everywhere: {opaque}): max|rad_net| = 2^{lg(res.max_abs_net):.1f}, "
                     f"max fluence = 2^{lg(res.max_fluence):.1f}, max product = "
                     f"2^{lg(res.max_product):.1f}  {'OK' if good else 'FAIL'}")
    # P5b: THE GAS ARM'S CHAIN, the pre-pass's one new product chain. The engine
    # forms 4L as `L << 2` for the Fleck denominator, so L must stay below 2^61.
    # Its widest value is at a_gas = ONE, the table top under a 0-K sky (the
    # excess is then all of E°) and N at the floor; stage 1 of the staged chain
    # (mul128_shr(deposit, recip_N, 16)) must fit int64 too.
    top = R.T_TABLE_TOP_GAME << 16
    for tname, tbl in (("resolving", R.E), ("live", R.E_LIVE)):
        L = R.fleck_L_gas_q(top, ONE, R.N_FLOOR_Q_LIVE, table=tbl, e_ref=0)
        rn, _rc = R.gas_capacity_recips(R.N_FLOOR_Q_LIVE, R.C_V_Q_LIVE, R.N_FLOOR_Q_LIVE)
        stage1 = (tbl[-1] * rn) >> 16
        good = (L << 2) < 2 ** 63 and stage1 < 2 ** 63
        ok &= good
        lines.append(f"  the gas arm's L at its widest on the {tname} table, the fold's "
                     f"currency (c_v_q = {R.C_V_Q_LIVE}, n_floor_q = {R.N_FLOOR_Q_LIVE}): "
                     f"L = 2^{lg(L):.2f}, 4L = 2^{lg(4 * L):.2f}, stage 1 = "
                     f"2^{lg(stage1):.2f}  (< 2^63 required)  {'OK' if good else 'FAIL'}")
    # ...and on the LIVE table (the one the engine bakes) no positive integer
    # currency can overflow it: the most extreme dials there are, c_v_q = 1 and
    # n_floor_q = 1 (one raw count each), still leave 4L inside int64.
    L_x = R.fleck_L_gas_q(top, ONE, 1, c_v_q=1, n_floor_q=1, table=R.E_LIVE, e_ref=0)
    good = (L_x << 2) < 2 ** 63
    ok &= good
    lines.append(f"  live table, the most extreme currency (c_v_q = n_floor_q = 1): "
                 f"4L = 2^{lg(4 * L_x):.2f} (< 2^63: every positive dial fits)  "
                 f"{'OK' if good else 'FAIL'}")
    # ---- #78: THE FINE HEAT CURRENCY'S HEADROOM (brief section 4) ----------
    # The live table is baked 2^FINE_BITS finer than one heat count (E°[3999] ~
    # 2^41.1, ~0.66x the resolving table's 2^41.7), so design v3's int64 bounds
    # must be restated on IT: over-driven at the table top (a = 1 everywhere,
    # f = 2^24 undamped, S16 and S12, both transports, the leak on and off), and
    # a travelling stream under a 0-K sky. The per-cell sums stay below 2^46; every
    # PLAIN int64 product of the sweep loop stays below 2^58, and so does the
    # pre-pass's a * ex (the widest plain product there is, 2^16 x the table top);
    # the one product above that is ex_m * f_q24, which the engine forms in 128
    # bits (mul128_shr, design 2.8: "no headroom argument is load-bearing").
    live = R.E_LIVE
    fb = R.fine_bits_of(live)
    worst = dict(stream=0, net=0, flu=0, amb=0, plain=0, fleck=0)
    for transport in ("shear", "step"):
        for n_ord in (16, 12):
            for kleak in (0, K_LEAK):
                res = R.sweep_q(R.plane(h, w, ONE), R.plane(h, w, ONE),
                                R.plane(h, w, kleak), T, transport=transport,
                                n_ord=n_ord, table=live)
                res2 = R.sweep_q(a2, d2, R.plane(h, w, kleak), T2, transport=transport,
                                 n_ord=n_ord, table=live, e_ref=0)
                for r in (res, res2):
                    worst["stream"] = max(worst["stream"], r.max_stream)
                    worst["net"] = max(worst["net"], r.max_abs_net)
                    worst["flu"] = max(worst["flu"], r.max_fluence)
                    worst["amb"] = max(worst["amb"], R.plane_max_abs(r.rad_amb))
                    worst["plain"] = max(worst["plain"], r.max_plain_product)
                    worst["fleck"] = max(worst["fleck"], r.max_fleck_product)
    a_ex = ONE * live[-1]                     # the pre-pass's a * ex, zero sky
    sums_ok = max(worst["net"], worst["flu"], worst["amb"]) < 2 ** 46
    plain_ok = worst["plain"] < 2 ** 58 and a_ex < 2 ** 58
    good = sums_ok and plain_ok and worst["fleck"] < 2 ** 63
    ok &= good
    lines.append(f"  #78 the FINE live table (k = {fb}, E°[3999] = 2^{lg(live[-1]):.2f}), "
                 f"over-driven at the top, S16/S12 x shear/step x leak on/off, and "
                 f"the travelling stream under a 0-K sky: max stream = "
                 f"2^{lg(worst['stream']):.2f}, per-cell |rad_net| = 2^{lg(worst['net']):.2f}, "
                 f"fluence = 2^{lg(worst['flu']):.2f}, |rad_amb| = 2^{lg(worst['amb']):.2f} "
                 f"(< 2^46); plain int64 products = 2^{lg(worst['plain']):.2f}, the "
                 f"pre-pass's a * ex = 2^{lg(a_ex):.2f} (< 2^58: >= 2^5 of int64 margin); "
                 f"the 128-bit ex_m * f_q24 = 2^{lg(worst['fleck']):.2f} (< 2^63)  "
                 f"{'OK' if good else 'FAIL'}")
    # ...and the GAS chain converts at its FINAL narrow, which is what keeps its
    # int64 results where they were at k = 0: at the most extreme currency the
    # stage-1 result and L stay inside int64 on the fine table, while a chain that
    # narrowed to int64 at 32 and converted AFTER would overflow (measured, the
    # pair). The fold's widest other conversion is an exact LEFT shift of a
    # per-cell sum, fine_heat_shr(rn, -16, k) = rn << (16 - k): the counters'
    # currency and the thinnest representable row's step (his = -16) alike, < 2^47.
    rN_x, rc_x = R.gas_capacity_recips(1, 1, 1)
    s1_x = (live[-1] * rN_x) >> 16
    L_last = (s1_x * rc_x) >> (32 + fb)
    s2_at_32 = (s1_x * rc_x) >> 32
    left = R.fine_heat_shr(worst["net"], -16, fb)
    # the pair is only meaningful on a FINE table (at k = 0 the two orders are one)
    pair = (fb == 0) or s2_at_32 >= 2 ** 63
    good = s1_x < 2 ** 63 and (L_last << 2) < 2 ** 63 and pair and left < 2 ** 47
    ok &= good
    lines.append(f"  #78 the gas chain on the fine table, most extreme currency: stage 1 = "
                 f"2^{lg(s1_x):.2f}, L (converted at the final narrow, 32 + {fb}) = "
                 f"2^{lg(L_last):.2f}, 4L < 2^63; the SAME chain narrowed to int64 at 32 "
                 f"then converted would carry 2^{lg(s2_at_32):.2f} (does not fit -- why "
                 f"the conversion is the final narrow); the fold's left shifts of the "
                 f"widest rad_net = 2^{lg(left):.2f} (< 2^47)  {'OK' if good else 'FAIL'}")
    return ok, lines


def _backward_steps(a_q, his, shift, table):
    """The damped source E°[0] + f*(E°[T] - E°[0]) over the WHOLE `table`, at the
    bucket LOW EDGE (T_q = (4b) << 16, so e_bucket_of returns exactly b), and the
    places where it FALLS as T rises.

    Returns (n_backward, worst_fraction, first_T_game, all_positive_and_sane).
    """
    vals = [R.damped_source_q((4 * b) << 16, a_q, his, table, shift=shift)
            for b in range(R.E_TABLE_SIZE)]
    sane = all(0 < v <= table[b] for b, v in enumerate(vals))
    drops = [(4 * i, (vals[i] - vals[i + 1]) / vals[i])
             for i in range(len(vals) - 1) if vals[i + 1] < vals[i]]
    worst = max((d[1] for d in drops), default=0.0)
    return len(drops), worst, (drops[0][0] if drops else None), sane


# The bound gate 12 asserts on the pathological row. Measured 0.047 % in Q24
# against 19.89 % in Q16, so this is ~10x headroom on the measurement and ~400x
# below the form the design rejected: a regression to Q16 cannot slip through.
PATHOLOGICAL_WOBBLE_BOUND = 0.005          # 0.5 % in power


def gate12_damped_source_is_monotone(fast=False):
    """G12 (P0b, design row 32). What a cell EMITS -- E°[0] + f*(E°[T] - E°[0]),
    formed exactly as the sweep forms it -- is NON-DECREASING in T over the whole
    4000-bucket table, for every absorbing material row config.toml ships.

    A hotter body must not radiate less. In Q16 that failed above the fire range
    (P0 section 0.4): f_q carried only 38 counts at the table top for wood, so one
    count was 2.6 % of it and the emission stepped backwards by up to 2.47 %. In
    Q24 every shipped row is exactly monotone.

    RE-CHECKED AT ALPHA FLOOR 0 (P2a, design row 39) and unchanged: below g = 1
    the source is the undamped E°[T] itself, which is monotone by the bake, and
    above it both floors take the same `4L` arm -- so the row-by-row counts below
    are bit-identical to P0b's. The floor is not free of this property, though:
    it is measured here, not assumed.

    Breaks if: f returns to Q16 (measured below, on the same probe, and it is not
    monotone); the E° bake or rad_scale is retuned so far that a bucket's rise no
    longer clears one count of f; or a NEW material ships `heat_atten > 0` with a
    thermal mass small enough to quantize its own plasma emission -- which is
    P0's proposed ingress rule (`heat_atten > 0` => `thermal_mass >= 8`) and is
    exactly the failure this gate should raise rather than hide.

    WHICH TABLE (M3, 2026-09-23). The property is about the rows the game SHIPS
    at the scale the game RUNS, so the shipped rows -- and the Q16 control that
    proves the probe can see a backward step on that same table -- are measured
    on `R.E_LIVE`, the sweep's own `rad_scale_derived`. Until M3 they ran on the
    reference's default table, the retired cast's fitted scale 3110x above it,
    and the three thin rows M2 authored failed there (542 / 365 / 517 steps) on
    a combination of rows and scale the game never runs (M2b). The pathological
    corner stays on the default RESOLVING table, where P0b designed it: it shows
    the MECHANISM (f's own resolution), which needs a table on which that corner
    damps hard -- on the live table a = 1, thermal_mass = 1 damps so little
    that neither Q16 nor Q24 steps backward and its `Q24 < Q16` control is 0 < 0.
    """
    lines, ok = [], True
    live, res = R.E_LIVE, R.E
    lines.append(f"  the damped source at every bucket's LOW EDGE, T = 0..{4 * (R.E_TABLE_SIZE - 1)} "
                 f"game, {len(SHIPPED_ROWS)} distinct shipped (a, his) pairs read from "
                 f"config.toml on the LIVE table (rad_scale_derived = "
                 f"{R.RAD_SCALE_LIVE:g}, E°[3999] = {live[-1]}) + the pathological "
                 f"corner on the RESOLVING table (rad_scale = {R.RAD_SCALE:g}), at "
                 f"the RULED alpha floor {R.ALPHA_FLOOR_DEFAULT!r} (row 39)")
    lines.append(f"    {'row':>34}{'Q16 steps':>11}{'Q16 worst':>11}"
                 f"{'Q24 steps':>11}{'Q24 worst':>11}{'f_q24 @ top':>13}")
    q16_nonmono = 0
    for a_q, his, nm in SHIPPED_ROWS:
        n16, w16, _f16, _s16 = _backward_steps(a_q, his, 16, live)
        n24, w24, first24, sane24 = _backward_steps(a_q, his, R.F_SHIFT, live)
        q16_nonmono += n16
        good = (n24 == 0) and sane24
        ok &= good
        f_top = R.fleck_f_solid_q(R.T_TABLE_TOP_GAME << 16, a_q, his, live)[0]
        lines.append(f"    {nm:>34}{n16:>11}{w16 * 100:>10.3f}%{n24:>11}"
                     f"{w24 * 100:>10.3f}%{f_top:>13}  "
                     + ("OK (monotone; positive and <= E°[T] everywhere)" if good
                        else f"FAIL (first backward step at {first24} game, "
                             f"sane={sane24})"))
    # NON-VACUITY: the same probe on the form the design rejected is NOT monotone,
    # so "0 backward steps" above is a property of Q24 and not of the measurement.
    ok &= (q16_nonmono > 0)
    lines.append(f"  non-vacuity: the rejected Q16 form has {q16_nonmono} backward steps "
                 f"over the same rows ON THE SAME LIVE TABLE (the Q16 columns above) -- "
                 f"the probe detects non-monotonicity when it is there")
    # THE MECHANISM: the wobble is f_q's own resolution, so it grows as the thermal
    # mass falls. No material ships thermal_mass = 1; if one ever does, the row
    # above fails and P1 owes the ingress rule rather than a wider f.
    a_p, his_p, nm_p = PATHOLOGICAL_ROW
    n16p, w16p, _, _ = _backward_steps(a_p, his_p, 16, res)
    n24p, w24p, first_p, sane_p = _backward_steps(a_p, his_p, R.F_SHIFT, res)
    bounded = (w24p < PATHOLOGICAL_WOBBLE_BOUND) and sane_p and (w24p < w16p)
    ok &= bounded
    lines.append(f"  the mechanism (RESOLVING table) -- {nm_p}: f_q24 at the table top is only "
                 f"{R.fleck_f_solid_q(R.T_TABLE_TOP_GAME << 16, a_p, his_p, res)[0]} counts, so "
                 f"{n24p} backward steps survive, worst {w24p * 100:.3f}% in power "
                 f"(= {(1 + w24p) ** 0.25 * 100 - 100:.3f}% in a receiver's equilibrium T), "
                 f"first at {first_p} game; in Q16 the same row gave {n16p} steps of "
                 f"{w16p * 100:.2f}%. Asserted bound: no backward step above "
                 f"{PATHOLOGICAL_WOBBLE_BOUND * 100:.1f}% for ANY row, shipped or not  "
                 f"{'OK' if bounded else 'FAIL'}")
    return ok, lines


def _smoke_row_scene(smoke_q, *, bulk=None, T_smoke=0, h=5, w=9):
    """A hot source (a = 1, 1263 game) at x = 1, a full COLUMN of smoke cells at
    x = 4 (every ordinate from the source to the right half crosses it), a cold
    absorber (a = 1) at (h//2, 7). Everything else is transparent air. Returns
    (a, d, k, T, gas kwargs, receiver)."""
    a = R.plane(h, w, 0)
    T = R.plane(h, w, 0)
    for y in range(h):
        a[y][1] = ONE
        T[y][1] = T_SRC_GAME << 16
    rcv = (h // 2, 7)
    a[rcv[0]][rcv[1]] = ONE
    ts = [[1 if a[y][x] > 0 else 0 for x in range(w)] for y in range(h)]
    smoke = R.plane(h, w, 0)
    nb = R.plane(h, w, ONE)
    for y in range(h):
        smoke[y][4] = smoke_q
        T[y][4] = T_smoke << 16
        if bulk is not None:
            nb[y][4] = bulk
    gkw = dict(gas=[smoke], heat_absorb_q=[Q(0.9)], n_bulk=nb, ts=ts)
    return a, [row[:] for row in a], R.plane(h, w, 0), T, gkw, rcv


def gate13_gas_extinction(fast=False):
    """G13 (P5a, design 6.3): THE DENSITY LAW, THE N_EPS FLOOR, AND WHAT A GAS
    CELL DOES TO THE STREAM.

      (a) a_gas counts absorbers: k x the density is k x a_gas (within the k - 1
          counts truncation can lose) up to the ONE cap, monotone in every plane,
          and a NEGATIVE density absorbs nothing.
      (b) TRANSMISSION: what thin smoke does not absorb continues down the stream.
          A cold absorber behind a smoke column receives strictly less as the
          smoke thickens, OPAQUE cold smoke is a perfect shield (the absorber books
          exactly 0), and the smoke books what the absorber lost.
      (c) THE N_EPS FLOOR: a cell with bulk N below N_EPS_RAW is INVISIBLE -- a
          thick, HOT smoke column with no bulk neither absorbs nor emits (its
          rad_net is 0, the absorber sees exactly the no-smoke stream); at
          N_EPS_RAW exactly it absorbs again.
      (d) a THERMAL SOLID ignores the gas in its pores: every plane identical with
          and without smoke on a furniture cell (and the same smoke on air moves
          them, so the comparison is not vacuous).
      (e) the stamped total is a MAX: a body standing in OPAQUE smoke books no
          rad_flux (the smoke took the stream), in thin smoke it books some.
      (f) THE GAS L_q CHAIN (fleck_L_gas_q -- the gas arm's, wired at P5b: G14): at
          unit capacity (c_v = 1, N_bulk = 1) it IS the solid chain at
          thermal_mass 1, bucket for bucket; and while the cell is thin DENSITY
          CANCELS -- L at k x (soot, bulk) equals L at (soot, bulk) to the chain's
          truncation.

    Breaks if: a second density factor is applied (the v2.4 min(N, N_AMB)/N_AMB
    on top of the extinction -- design 6.3's double debit), the floor moves off
    N_EPS_RAW or off the BULK count, thermal solids start taking the smoke term,
    the body share is summed instead of MAXed, or the staged chain's order or its
    reciprocals change.
    """
    lines, ok = [], True
    hq = [Q(0.9), Q(0.3)]
    # (a) the density law, as arithmetic
    base = Q(0.013)
    a1 = R.gas_extinction_q([base, 0], hq, ONE)
    prop = True
    for kk in (2, 3, 5, 8):
        ak = R.gas_extinction_q([kk * base, 0], hq, ONE)
        prop &= (0 <= ak - kk * a1 <= kk - 1)
    mono = True
    prev = -1
    for n_s in range(0, 2 * ONE, ONE // 37):
        v = R.gas_extinction_q([n_s, Q(0.4)], hq, ONE)
        mono &= (v >= prev)
        prev = v
    capped = R.gas_extinction_q([10 * ONE, 10 * ONE], hq, ONE) == ONE
    neg = (R.gas_extinction_q([-ONE, Q(0.4)], hq, ONE)
           == R.gas_extinction_q([0, Q(0.4)], hq, ONE))
    good = prop and mono and capped and neg and a1 > 0
    ok &= good
    lines.append(f"  (a) a_gas(k x N) - k x a_gas(N) in [0, k-1] for k = 2,3,5,8 (a_gas(N) = "
                 f"{a1}): {prop};  monotone in density: {mono};  capped at ONE: {capped};  "
                 f"a negative density absorbs nothing: {neg}  {'OK' if good else 'FAIL'}")
    # (b) transmission through a smoke column
    got = []
    for dens in (0, Q(0.05), Q(0.2), Q(0.5), 2 * ONE):
        a, d, k, T, gkw, rcv = _smoke_row_scene(dens)
        res = R.sweep_q(a, d, k, T, **gkw)
        smoke_net = sum(res.rad_net[y][4] for y in range(len(a)))
        got.append((dens, res.a_eff[0][4], res.rad_net[rcv[0]][rcv[1]], smoke_net,
                    res.identity()))
    rcv_nets = [g[2] for g in got]
    strictly = all(rcv_nets[i + 1] < rcv_nets[i] for i in range(len(rcv_nets) - 1))
    shield = rcv_nets[-1] == 0 and got[-1][1] == ONE
    books = all(g[3] >= 0 for g in got) and got[0][3] == 0 and all(g[3] > 0 for g in got[1:])
    exact = all(g[4] == 0 for g in got)
    good = strictly and shield and books and exact
    ok &= good
    lines.append(f"  (b) a cold absorber behind a smoke column, source {T_SRC_GAME} game:")
    for dens, a_g, rn, sn, _i in got:
        lines.append(f"        smoke density {dens / ONE:5.2f} (a_gas {a_g / ONE:.4f}): "
                     f"absorber rad_net {rn:>9d}   smoke column books {sn:>9d}")
    lines.append(f"      strictly less behind thicker smoke: {strictly};  opaque cold smoke "
                 f"is a perfect shield (absorber books 0): {shield};  the smoke books a "
                 f"positive absorption: {books};  identity exact: {exact}  "
                 f"{'OK' if good else 'FAIL'}")
    # (c) the N_EPS floor
    a, d, k, T, gkw0, rcv = _smoke_row_scene(0)
    clear = R.sweep_q(a, d, k, T, **gkw0)
    a, d, k, T_hot, gkw_v, rcv = _smoke_row_scene(2 * ONE, bulk=R.N_EPS_RAW - 1,
                                                  T_smoke=5000)
    vac = R.sweep_q(a, d, k, T_hot, **gkw_v)
    silent = all(vac.rad_net[y][4] == 0 for y in range(len(a)))
    invisible = all(getattr(vac, p) == getattr(clear, p)
                    for p in ("rad_net", "rad_flux", "rad_amb", "rad_fluence"))
    a, d, k, T_hot, gkw_e, rcv = _smoke_row_scene(2 * ONE, bulk=R.N_EPS_RAW, T_smoke=5000)
    edge = R.sweep_q(a, d, k, T_hot, **gkw_e)
    edge_live = edge.a_eff[0][4] == ONE and edge.rad_net[0][4] != 0
    good = invisible and silent and edge_live
    ok &= good
    lines.append(f"  (c) a thick smoke column at 5000 game with bulk N = N_EPS_RAW - 1: its "
                 f"rad_net is 0 on every cell ({silent}) and all four planes equal the "
                 f"clear-air run's ({invisible}); at N = N_EPS_RAW it absorbs and radiates "
                 f"again (rad_net {edge.rad_net[0][4]}): {edge_live}  "
                 f"{'OK' if good else 'FAIL'}")
    # (d) thermal solids ignore their gas
    rng = random.Random(1313)
    a, d, T, f = _rand_scene(rng, 7, 8)
    for y in range(7):
        for x in range(8):
            if a[y][x] == 0:
                a[y][x] = Q(0.5)                   # every cell a thermal solid...
                d[y][x] = max(d[y][x], a[y][x])
    ts = [[1] * 8 for _ in range(7)]
    k = R.plane(7, 8, K_LEAK)
    smoke = R.plane(7, 8, 2 * ONE)
    nb = R.plane(7, 8, ONE)
    plain = R.sweep_q(a, d, k, T, f_plane=f)
    pores = R.sweep_q(a, d, k, T, f_plane=f, gas=[smoke], heat_absorb_q=[Q(0.9)],
                      n_bulk=nb, ts=ts)
    same = all(getattr(plain, p) == getattr(pores, p)
               for p in ("rad_net", "rad_flux", "rad_amb", "rad_fluence"))
    ts_air = [row[:] for row in ts]
    ts_air[3][4] = 0                              # ... but ONE of them is air
    a_air = [row[:] for row in a]
    a_air[3][4] = 0
    d_air = [row[:] for row in d]
    plain_air = R.sweep_q(a_air, d_air, k, T, f_plane=f)
    smoky_air = R.sweep_q(a_air, d_air, k, T, f_plane=f, gas=[smoke],
                          heat_absorb_q=[Q(0.9)], n_bulk=nb, ts=ts_air)
    moved = plain_air.rad_net != smoky_air.rad_net
    good = same and moved
    ok &= good
    lines.append(f"  (d) smoke in the pores of 56 thermal solids moves nothing: {same};  the "
                 f"same smoke on ONE air cell moves the books: {moved}  "
                 f"{'OK' if good else 'FAIL'}")
    # (e) a body in smoke: the stamped total is a MAX
    fl = []
    for dens in (Q(0.05), 2 * ONE):
        a, d, k, T, gkw, rcv = _smoke_row_scene(dens)
        d[2][4] = ONE                               # a marine standing in the smoke
        res = R.sweep_q(a, d, k, T, **gkw)
        fl.append((dens, res.a_eff[2][4], res.d_eff[2][4], res.rad_flux[2][4]))
    thin_ok = fl[0][3] > 0 and fl[0][2] == ONE and fl[0][1] < ONE
    opaque_ok = fl[1][3] == 0 and fl[1][1] == ONE and fl[1][2] == ONE
    # a PARTIAL body (d = 0.5, a small drone) in thin smoke is where a MAX and a
    # capped SUM part ways: the stamped total stays 0.5, the body keeps 0.5 - a_gas
    a, d, k, T, gkw, rcv = _smoke_row_scene(Q(0.05))
    d[2][4] = Q(0.5)
    res = R.sweep_q(a, d, k, T, **gkw)
    part_ok = (res.d_eff[2][4] == Q(0.5) and res.a_eff[2][4] == fl[0][1]
               and res.rad_flux[2][4] > 0)
    good = thin_ok and opaque_ok and part_ok
    ok &= good
    lines.append(f"  (e) a marine (d = ONE) in thin smoke (a_gas {fl[0][1] / ONE:.4f}) books "
                 f"rad_flux {fl[0][3]}; in opaque smoke (a_gas 1) it books {fl[1][3]}; a "
                 f"half-opaque body in the thin smoke keeps d = {res.d_eff[2][4] / ONE:.4f} "
                 f"(a MAX, not {(Q(0.5) + fl[0][1]) / ONE:.4f}): {part_ok}  "
                 f"{'OK' if good else 'FAIL'}")
    # (f) the gas L_q chain
    stride = 7 if fast else 1
    unit = all(R.fleck_L_gas_q((4 * b) << 16, a_q, ONE, c_v_q=ONE, n_floor_q=655)
               == R.fleck_L_solid_q((4 * b) << 16, a_q, 0)
               for a_q in (ONE, Q(0.37), Q(0.9)) for b in range(0, R.E_TABLE_SIZE, stride))
    c_v_q = R.quant(R.C_V_LIVE)
    n_floor_q = R.quant(R.N_FLOOR_HEAT_LIVE)
    # The tolerance is the arithmetic's own: a_gas truncates to a whole count, so
    # the thinner cell's L may differ from its denser twin's by up to one count in
    # a_k (relative 1/a_k); the two reciprocals and two floors add < 1e-5. Twice
    # that is the bound -- a change of LAW (a second density factor) moves L by
    # the density ratio itself, 2x or 4x, so the gate cannot mistake one for it.
    worst_rel, worst_tol, cancels = 0.0, 0.0, True
    for T_game in (300, 1263, 5000, 15996):
        for n_b in (ONE, Q(0.5), Q(0.2)):
            soot = Q(0.2) * n_b // ONE
            L1 = R.fleck_L_gas_q(T_game << 16, R.gas_extinction_q([soot], [Q(0.9)], n_b),
                                 n_b, c_v_q=c_v_q, n_floor_q=n_floor_q)
            for kk in (2, 4):
                a_k = R.gas_extinction_q([soot // kk], [Q(0.9)], n_b // kk)
                Lk = R.fleck_L_gas_q(T_game << 16, a_k, n_b // kk, c_v_q=c_v_q,
                                     n_floor_q=n_floor_q)
                rel, tol = abs(Lk - L1) / L1, 2.0 / a_k
                cancels &= rel <= tol
                if rel > worst_rel:
                    worst_rel, worst_tol = rel, tol
    # ...and THE CAPACITY LAW itself, L = a_gas * ex / (N_bulk * c_v), against the
    # float formula at the LIVE c_v (its one integer form c_v_q), over bulk counts
    # above and BELOW the n_floor_heat floor (where N stops shrinking). Pins what
    # the two comparisons above cannot see: both cancel c_v.
    worst_law = 0.0
    for T_game in (300, 1263, 5000, 15996):
        ex = R.E[R.e_bucket_of(T_game << 16)] - R.E[0]
        for a_q in (ONE, Q(0.2)):
            for n_b in (ONE, Q(0.3), n_floor_q, n_floor_q // 4):
                L = R.fleck_L_gas_q(T_game << 16, a_q, n_b, c_v_q=c_v_q,
                                    n_floor_q=n_floor_q)
                n_eff = max(n_b, n_floor_q) / ONE
                want = (a_q / ONE) * ex / (n_eff * (c_v_q / ONE))
                worst_law = max(worst_law, abs(L - want) / want)
    law = worst_law < 1e-4
    good = unit and cancels and law
    ok &= good
    lines.append(f"  (f) the gas chain at unit capacity IS the solid chain at his = 0, bucket "
                 f"for bucket, three absorptivities: {unit};  thin cell (soot fraction 0.2, "
                 f"h = 0.9), bulk N from 1.0 down to 0.05: L's worst relative change "
                 f"{worst_rel:.2e} against its truncation bound 2/a_gas = "
                 f"{worst_tol:.2e} (density cancels): {cancels};  L against a*ex/(max(N, "
                 f"n_floor) * c_v) at the live c_v, above and below the floor: worst "
                 f"relative {worst_law:.2e} (< 1e-4): {law}  {'OK' if good else 'FAIL'}")
    return ok, lines


# --------------------------------------------------------------------------- #
# P5b: THE GAS ARM OF THE FLECK PRE-PASS (design 2.8 / 6.3). The cells of
# p5a_gas_stiffness_study.py's table: pure soot (a = 1, "soot fraction 1.0") and
# a typical smoke mixture (a = 0.2 per tile), each at ambient bulk density, at
# the density a cell at T holds at AMBIENT PRESSURE, and at the n_floor_heat
# density -- the STIFFEST any gas cell can be, since g scales as
# a_gas / max(N, n_floor) with a_gas <= ONE.
# --------------------------------------------------------------------------- #
GAS_A_SOOT = ONE
GAS_A_MIX = Q(0.2)


def _isobaric_n_q(T_game):
    """The bulk N a cell at T holds at ambient pressure (p = C N T_abs):
    N = N_amb * T_amb / T_abs -- a hot cell is a thin cell."""
    return (ONE * R.K_AMB) // (T_game + R.K_AMB)


GAS_ARM_CASES = (
    ("pure soot, N = 1", GAS_A_SOOT, lambda T: ONE),
    ("typical mix (a = 0.2), N = 1", GAS_A_MIX, lambda T: ONE),
    ("pure soot, isobaric", GAS_A_SOOT, _isobaric_n_q),
    ("typical mix, isobaric", GAS_A_MIX, _isobaric_n_q),
    ("pure soot at the n_floor density", GAS_A_SOOT, lambda T: R.N_FLOOR_Q_LIVE),
    ("typical mix at the n_floor density", GAS_A_MIX, lambda T: R.N_FLOOR_Q_LIVE),
)
# 2 s. Above g = 1 the damped step is a quarter of T_abs per tick, so even a
# start at the table top is inside the undamped range within ~8 ticks; the rest
# of the march is the explicit tail the property must also hold on.
GAS_MARCH_TICKS = 48


def gate14_gas_fleck_arm(fast=False):
    """G14 (P5b, design 2.8 / 6.3): THE GAS ARM OF THE FLECK PRE-PASS.

      (a) THE PROPERTY IT EXISTS FOR, on the LIVE table (rad_scale_derived) in the
          fold's gas currency: a hot absorbing gas cell radiating into an ambient
          room (a held Phi = E°[0]) cools MONOTONICALLY and NEVER BELOW AMBIENT,
          from every start temperature up to the table top, over P5a's whole
          stiffness range (GAS_ARM_CASES) -- and it COOLS: every start ends below
          where it began, so the march is not vacuous.
      (b) THE BREAK THAT PROVES IT: the same march with the arm at L = 0 (what
          P5a shipped) overshoots below ambient in ONE step, and the set of starts
          where it does is EXACTLY {T : g > 4T/T_abs} = {T : L_q > T_q} (the
          undamped step removes more than the cell's whole excess) -- start for
          start, in every case, and it is not empty.
      (c) THE PRE-PASS CALLS IT: on randomised smoky scenes, both tables, every
          absorbing gas cell's f is fleck_f_gas_q's -- some damped (f < 2^24), some
          not; every thermal solid keeps the solid arm's f; a gas cell that
          absorbs nothing keeps 2^24; without the gas group every gas cell is back
          at 2^24 (the P5a arm) and the sweep's rad_net moves.
      (d) THE INGRESS: the arm refuses a non-positive c_v or n_floor (a silently
          undamped arm) and a partial gas group.
      (e) WHAT IT EMITS (gate 12's property, on the gas arm): on the live table at
          a held N, the damped source a gas cell emits never steps backward in T
          by more than gate 12's pathological bound -- Q24 is resolved finely
          enough for the stiffest gas cell there is. (At ambient PRESSURE the
          damped emission of a stiff cell is FLAT in T -- its loss is T_abs/4 of
          a capacity N c_v ~ 1/T_abs, a constant -- so the isobaric cells are
          reported, not held to it.)

    Breaks if: the arm loses or reorders a reciprocal, prices the cell in any
    currency but the fold's, reads anything but a_gas and the BULK count, or is
    disconnected -- (a) then fails, and (b) shows it must fail exactly where the
    undamped step exceeds the cell's own excess.
    """
    lines, ok = [], True
    live = R.E_LIVE
    phi = live[0]
    stride = 7 if fast else 1
    ticks = GAS_MARCH_TICKS
    lines.append(f"  (a)/(b) the 0-D gas cell, LIVE table (E°[0] = {live[0]}), currency "
                 f"c_v_q = {R.C_V_Q_LIVE}, n_floor_q = {R.N_FLOOR_Q_LIVE}, a held Phi = "
                 f"E°[0], {ticks} ticks from every {4 * stride}-game start in "
                 f"4..{R.T_TABLE_TOP_GAME}:")
    lines.append(f"    {'cell':<36}{'damped from':>12}{'wired: T<0':>11}{'non-mono':>9}"
                 f"{'worst step':>11}{'L = 0: T<0':>11}{'= g > 4T/T_abs':>15}")
    for name, a_q, n_of in GAS_ARM_CASES:
        undershoot = nonmono = stalled = 0
        worst_share = 0.0
        broke, predicted = set(), set()
        onset = None
        n_starts = 0
        for b in range(1, R.E_TABLE_SIZE, stride):
            T0g = 4 * b
            T0 = T0g << 16
            n_q = n_of(T0g)
            n_starts += 1
            tr = R.gas_cell_march(T0, phi, a_q, n_q, ticks, table=live, trace=True)
            undershoot += int(min(tr) < 0)
            nonmono += int(any(tr[i + 1] > tr[i] for i in range(len(tr) - 1)))
            stalled += int(not tr[-1] < T0)
            worst_share = max(worst_share, (tr[0] - tr[1]) / T0)
            f_q, L_q = R.fleck_f_gas_q(T0, a_q, n_q, table=live)
            if f_q < F_ONE and onset is None:
                onset = T0g
            if L_q > T0:                               # g > 4T/T_abs, exactly
                predicted.add(T0g)
            if R.gas_cell_march(T0, phi, a_q, n_q, 1, table=live, fleck=False) < 0:
                broke.add(T0g)
        exact = (broke == predicted)
        good = (undershoot == 0 and nonmono == 0 and stalled == 0 and exact
                and len(broke) > 0)
        ok &= good
        lines.append(f"    {name:<36}{str(onset) + ' game':>12}{undershoot:>11}{nonmono:>9}"
                     f"{worst_share * 100:>10.1f}%{len(broke):>11}"
                     f"{('yes, ' + str(len(predicted))) if exact else 'NO':>15}"
                     f"  {'OK' if good else 'FAIL'}")
    lines.append(f"    over {n_starts} starts per cell: the wired arm never crossed ambient, "
                 f"never rose, always cooled; 'worst step' is its largest first step as a "
                 f"share of the cell's own excess (< 100 % is the margin to ambient). The "
                 f"L = 0 arm crossed below ambient in ONE step at exactly the starts where "
                 f"L_q > T_q, i.e. g > 4T/T_abs")
    # (c) the pre-pass calls it
    rng = random.Random(20260924)
    h, w = (7, 8) if fast else (9, 11)
    n_damped = n_undamped = n_solid = n_silent = 0
    arm_ok = solids_ok = silent_ok = p5a_ok = moved_ok = True
    for tname, table in (("resolving", R.E), ("live", R.E_LIVE)):
        for _trial in range(2 if fast else 4):
            a, d, T, _f = _rand_scene(rng, h, w)      # T up to 15999 game: the live
            gkw = _gas_kw(rng, a)                     # table damps gas there too
            ts = gkw["ts"]
            k = R.plane(h, w, K_LEAK)
            f = R.fleck_prepass(T, a, 3, table=table, **gkw)
            f0 = R.fleck_prepass(T, a, 3, table=table, ts=ts)
            a_gas = R.gas_extinction_plane(gkw["gas"], gkw["heat_absorb_q"],
                                           gkw["n_bulk"], ts)
            for y in range(h):
                for x in range(w):
                    if ts[y][x]:
                        n_solid += 1
                        solids_ok &= (f[y][x] == f0[y][x] == R.fleck_f_solid_q(
                            T[y][x], a[y][x], 3, table)[0])
                        continue
                    p5a_ok &= (f0[y][x] == F_ONE)
                    if a_gas[y][x] == 0:
                        n_silent += 1
                        silent_ok &= (f[y][x] == F_ONE)
                        continue
                    want = R.fleck_f_gas_q(T[y][x], a_gas[y][x], gkw["n_bulk"][y][x],
                                           table=table)[0]
                    arm_ok &= (f[y][x] == want)
                    n_damped += int(want < F_ONE)
                    n_undamped += int(want == F_ONE)
            if any(f[y][x] < F_ONE for y in range(h) for x in range(w) if not ts[y][x]):
                r1 = R.sweep_q(a, d, k, T, f_plane=f, table=table, **gkw)
                r0 = R.sweep_q(a, d, k, T, f_plane=f0, table=table, **gkw)
                moved_ok &= (r1.rad_net != r0.rad_net) and (r1.identity() == 0)
    good = (arm_ok and solids_ok and silent_ok and p5a_ok and moved_ok
            and n_damped > 0 and n_undamped > 0 and n_solid > 0 and n_silent > 0)
    ok &= good
    lines.append(f"  (c) randomised smoky scenes, both tables: every absorbing gas cell's f is "
                 f"fleck_f_gas_q's: {arm_ok} ({n_damped} damped, {n_undamped} undamped); "
                 f"{n_solid} thermal solids keep the solid arm: {solids_ok}; {n_silent} "
                 f"non-absorbing gas cells keep 2^24: {silent_ok}; without the gas group "
                 f"every gas cell is 2^24 (the P5a arm): {p5a_ok}, and the arm moves "
                 f"rad_net with the identity exact: {moved_ok}  {'OK' if good else 'FAIL'}")
    # (d) the ingress
    a2, T2 = R.plane(2, 2, 0), R.plane(2, 2, 300 << 16)
    ts2 = R.plane(2, 2, 0)
    grp = dict(gas=[R.plane(2, 2, ONE)], heat_absorb_q=[Q(0.9)], n_bulk=R.plane(2, 2, ONE))
    R.fleck_prepass(T2, a2, 3, ts=ts2, **grp)                        # legal: no raise
    rejected = []
    for bad_name, kw in (("c_v_q = 0", dict(c_v_q=0)), ("c_v_q < 0", dict(c_v_q=-504)),
                         ("n_floor_q = 0", dict(n_floor_q=0)),
                         ("n_floor_q < 0", dict(n_floor_q=-655)),
                         ("no n_bulk", dict(gas=grp["gas"], heat_absorb_q=grp["heat_absorb_q"],
                                            n_bulk=None)),
                         ("no ts", dict(ts=None))):
        call = dict(ts=ts2, **grp)
        call.update(kw)
        try:
            R.fleck_prepass(T2, a2, 3, **call)
            rejected.append((bad_name, False))
        except ValueError:
            rejected.append((bad_name, True))
    good = all(r for _n, r in rejected)
    ok &= good
    lines.append(f"  (d) the arm's ingress refuses " + ", ".join(
        f"{n}: {'raised' if r else 'ACCEPTED -- FAIL'}" for n, r in rejected)
        + f"  {'OK' if good else 'FAIL'}")
    # (e) what the arm's cell emits, over the whole live table
    lines.append(f"  (e) the damped source E°[0] + f (E°[T] - E°[0]) of a gas cell, every "
                 f"bucket of the LIVE table (gate 12's property; its pathological bound "
                 f"{PATHOLOGICAL_WOBBLE_BOUND * 100:.1f} %):")
    for name, a_q, n_of in GAS_ARM_CASES:
        vals = []
        for b in range(R.E_TABLE_SIZE):
            f_q = R.fleck_f_gas_q((4 * b) << 16, a_q, n_of(4 * b), table=live)[0]
            vals.append(live[0] + (((live[b] - live[0]) * f_q) >> R.F_SHIFT))
        drops = [(vals[i] - vals[i + 1]) / vals[i] for i in range(len(vals) - 1)
                 if vals[i + 1] < vals[i]]
        worst = max(drops, default=0.0)
        f_top = R.fleck_f_gas_q(R.T_TABLE_TOP_GAME << 16, a_q, n_of(R.T_TABLE_TOP_GAME),
                                table=live)[0]
        held_n = "isobaric" not in name
        good = (worst < PATHOLOGICAL_WOBBLE_BOUND) if held_n else True
        ok &= good
        lines.append(f"    {name:<36} backward steps {len(drops):>5}, worst "
                     f"{worst * 100:.4f} %, f at the table top {f_top:>7} counts  "
                     + (("OK" if good else "FAIL") if held_n
                        else "(reported: flat at ambient pressure)"))
    return ok, lines


# --------------------------------------------------------------------------- #
# P5c: THE FOLD'S GAS BRANCH (design 2.8 / 6.3 / 8.4). fold_pass1_gas is what
# temperature_solver.cpp's gas branch and its CUDA twin transcribe; this gate is
# its property sheet, and tests/test_temperature_gas_radiation.py holds the
# engine's fold to it bit for bit.
# --------------------------------------------------------------------------- #
def _gas_fold_cells(rng, n, table):
    """n random ACCOUNTABLE gas cells as 1 x n planes: stored energy with a random
    residual E mod N, bulk from the N_EPS edge through the n_floor density to 3
    atm, rad_net of both signs from one count to 2^44, and a fluence that puts
    the cell's cap both above and below it (so the clamp binds on some and not
    on others, and some cells sit ABOVE their cap -- a combustion-held flame)."""
    T, Eg, rn, phi, nb = [[]], [[]], [[]], [[]], [[]]
    n_pool = [R.N_EPS_RAW, 200, R.N_FLOOR_Q_LIVE, Q(0.2), ONE, 3 * ONE]
    for _ in range(n):
        N = rng.choice(n_pool)
        t_game = rng.choice([-150, 0, 5, 300, 804, 1263, 3000, 9000])
        t_q = (t_game << 16) + rng.randrange(0, 1 << 16)
        e = N * (t_q + R.T_AMB_Q) + rng.randrange(0, N)      # a real residual
        mag = rng.choice([1, 77, 1 << 12, 1 << 20, 1 << 30, 1 << 44])
        r = mag if rng.random() < 0.6 else -mag
        cap_game = rng.choice([0, 4, 290, 804, 1263, 5000, 15996])
        f = table[min(R.E_TABLE_SIZE - 1, cap_game // 4)] + rng.choice([0, 1, 999])
        T[0].append(R.gas_mirror_q(e, N))
        Eg[0].append(e)
        rn[0].append(r)
        phi[0].append(f)
        nb[0].append(N)
    return T, Eg, rn, phi, nb


def _sealed_smoky_room(h, w, *, smoke_q, hq, T_smoke_game, wall_a=Q(0.85), wall_his=5):
    """A sealed room: one ring of opaque ambient walls (a thermal solid, a =
    wall_a, thermal_mass 2^wall_his), the interior full of smoke at density
    smoke_q (one absorbing gas, coefficient hq) and T_smoke_game, at ambient bulk
    density. Returns a reference Scene on the LIVE table."""
    a = R.plane(h, w, 0)
    T = R.plane(h, w, 0)
    for y in range(h):
        for x in range(w):
            if y in (0, h - 1) or x in (0, w - 1):
                a[y][x] = wall_a
            else:
                T[y][x] = T_smoke_game << 16
    ts = [[1 if a[y][x] > 0 else 0 for x in range(w)] for y in range(h)]
    smoke = [[0 if ts[y][x] else smoke_q for x in range(w)] for y in range(h)]
    nb = R.plane(h, w, ONE)
    return R.Scene(a=a, d=[r[:] for r in a], k=R.plane(h, w, 0), T=T, his=wall_his,
                   ts=ts, gas=[smoke], heat_absorb_q=[hq], n_bulk=nb, table=R.E_LIVE)


def _shield_scene(smoke_q, *, h=7, w=13, hq=Q(5.0)):
    """A held hot wall column (a = 1, 1263 game) at x = 0, a cold target column
    held at ambient at x = w - 1 (a = 0.9), smoke of density smoke_q filling
    x = 3..w-4 at ambient bulk density; everything else clear air. Returns the
    Scene (LIVE table) and the target cells."""
    a = R.plane(h, w, 0)
    T = R.plane(h, w, 0)
    held = R.plane(h, w, 0)
    for y in range(h):
        a[y][0] = ONE
        T[y][0] = T_SRC_GAME << 16
        held[y][0] = 1
        a[y][w - 1] = Q(0.9)
        held[y][w - 1] = 1
    ts = [[1 if a[y][x] > 0 else 0 for x in range(w)] for y in range(h)]
    smoke = [[smoke_q if (3 <= x <= w - 4) else 0 for x in range(w)] for y in range(h)]
    sc = R.Scene(a=a, d=[r[:] for r in a], k=R.plane(h, w, 0), T=T, his=3, ts=ts,
                 held=held, gas=[smoke], heat_absorb_q=[hq], n_bulk=R.plane(h, w, ONE),
                 table=R.E_LIVE)
    return sc, [(y, w - 1) for y in range(h)]


def _chain_bound(rn, N, cap, dT, fine_bits=0, c_v_q=R.C_V_Q_LIVE):
    """The staged chain's own precision, per gas cell, in Q32 heat currency
    (design 2.8's 'a DIFFERENT rounding, declared'): |rn_q16 - dT * cap_real| <=
    |rn| * N_q / 2^(16 + k) (recip_N is floor(2^32 / N_q), relative error <
    N_q / 2^32) + (ceil(recip_cv / 2^(32 + k)) + 1) * cap_real (stage 1's floor,
    amplified by recip_cv / 2^(32 + k), plus stage 2's own) + |dT| + 1
    (cap_real's own floor of N * c_v_q / 2^16). rn_q16 is rn in the Q16 heat
    currency, fine_heat_shr(rn, -16, k) (#78: k = the table's fine bits).
    Valid for N >= n_floor (below it the floor DILUTES the landing by N / n_floor,
    which is not a truncation -- measured separately).

    #78 IS THIS BOUND'S SECOND TERM: at k = 0 it is 132 * cap_real (stage 1's
    floor alone costs ~130 temperature LSBs at the shipped c_v); at the live
    k = 11 it is 2 * cap_real -- the temperature field's own resolution."""
    rc = R.make_recip(c_v_q / 65536.0)
    amp = -(-rc // (1 << (32 + fine_bits)))          # ceil(recip_cv / 2^(32+k))
    return abs(rn) * N // (1 << (16 + fine_bits)) + (amp + 1) * cap + abs(dT) + 1


def gate15_gas_fold(fast=False):
    """G15 (P5c, design 2.8 / 6.3 / 8.4): THE TEMPERATURE FOLD'S GAS BRANCH.

      (a) THE CLAMP'S ENERGY FORM HITS ITS TARGET AND KEEPS THE RESIDUAL. On
          randomised accountable gas cells (a stored residual E mod N; bulk from
          the N_EPS edge to 3 atm; rad_net of both signs up to 2^44; a fluence
          whose cap sits above and below the cell), every clamped cell's mirror
          lands EXACTLY on max(T_before, e_ceiling_q(Phi)) -- the clamp's
          ceiling since P5d -- and every other on
          sat(T_before + dT), with E mod N unchanged on every cell. The PAIR:
          the design's letter N * (T_target + t_amb) - E lands on the same
          mirror but DRAINS the residual -- counted, and non-zero.
      (b) THE BOOKS, and the withheld energy, exactly. sum(Eg) moves by
          e_gas_deposit_sum + e_gas_rail_sum to the count (group 1 -- no new
          group); per cell the landing plus what the clamp withheld is the
          unclamped step, and e_rad_clamp_drop_sum is that withheld step priced
          at cap_real, summed -- on the gas cells AND on thermal solids.
      (c) THE MAXIMUM PRINCIPLE ON GAS: on the radiative sub-step alone, every
          gas cell ends at or below max(T_before, e_ceiling_q(Phi)); with the
          clamp off some cell does not (the non-vacuity pair).
      (d) HOT SMOKE COOLS, THROUGH THE WHOLE TICK, ON THE LIVE TABLE: a sealed
          room of ambient walls full of hot absorbing smoke -- every smoke cell's
          T never rises, never goes below ambient, and ends lower; the walls warm.
          With the fold's gas branch off (the P5b state) the smoke does not move.
      (e) SMOKE SHIELDS, WITH THE FOLD LIVE: a held 1263-game wall and a target
          held at ambient across clear air; a smoke layer between them cuts the
          target's absorbed flux on EVERY tick against the clear-air control --
          also after the smoke has heated and re-radiates -- and thin smoke cuts
          less than thick.
      (f) THE SWEEP->FOLD BOUNDARY (design 8.4), over (d)'s and (e)'s runs: on
          every touched cell-tick, the sweep's rad_net and what the fold landed
          plus what the clamp withheld differ by at most the conversion's own
          truncation -- < one temperature LSB x C on a thermal solid (shr_round0),
          the staged chain's declared precision on gas (_chain_bound) -- so
          sum(rad_net) - sum(landed) - e_rad_clamp_drop_sum is bounded by the sum
          of those, and nothing else leaves the boundary uncounted.
      (g) THE BOUNDARY'S OTHER TWO EXITS, COUNTED (P5c follow-up): below
          n_floor the chain lands only ~N / n_floor of rn -- per cell, rn << 16
          is exactly the landing + the clamp's drop + e_rad_floor_drop_sum's
          share, and that share exceeds the chain's declared precision (a
          dilution, not a rounding); a gas cell OUTSIDE the accountable set is
          untouched and its whole rn << 16 goes to e_rad_boundary_export_sum.

    Breaks if: the gas branch scales dE instead of stepping N * (T_target -
    T_before) (a misses the target), drains or mints the residual (a), books a
    new group or skips e_gas_deposit_sum (b), clamps a cooling step or clamps
    before the conversion (c), converts through anything but the staged chain in
    the fold's currency, or is disconnected (d, f); or a floored cell's
    remainder or a boundary cell's rad_net leaves the fold uncounted (g).
    """
    lines, ok = [], True
    live = R.E_LIVE
    rng = random.Random(20260925)
    # (a) + (b) + (c): one-shot randomised cells, through fold_pass1_gas
    n = 400 if fast else 3000
    T, Eg, rn, phi, nb = _gas_fold_cells(rng, n, live)
    T0, E0 = [r[:] for r in T], [r[:] for r in Eg]
    ts = R.plane(1, n, 0)
    c = R.FoldCounters()
    R.fold_pass1_gas(T, Eg, rn, phi, nb, ts, c, table=live)
    target_ok = resid_ok = mp_ok = step_ok = True
    n_clamped = n_free = n_letter_drain = 0
    letter_drained = 0
    drop_sum = 0
    fb = R.fine_bits_of(live)                  # #78: rn is in the table's currency
    for i in range(n):
        N, e0, t0, r = nb[0][i], E0[0][i], T0[0][i], rn[0][i]
        dT = R.gas_rad_dT_q(r, N, fine_bits=fb)
        t_after = R.sat_add_q16(t0, dT)
        ceiling = max(R.e_ceiling_q(phi[0][i], live), t0)
        if t_after > ceiling:
            n_clamped += 1
            target_ok &= (T[0][i] == ceiling)
            drop_sum += (t_after - ceiling) * R.cap_real_q(False, 0, N)
            # the design's letter lands on the same mirror with a ZERO residual,
            # i.e. it drains the cell's E mod N; this form keeps it
            target_ok &= (R.gas_mirror_q(N * (ceiling + R.T_AMB_Q), N) == ceiling)
            drained = e0 - N * (t0 + R.T_AMB_Q)          # == e0 mod N
            if drained > 0:
                n_letter_drain += 1
                letter_drained += drained
        else:
            n_free += 1
            step_ok &= (T[0][i] == t_after)
        resid_ok &= (Eg[0][i] % N == e0 % N)
        mp_ok &= (T[0][i] <= ceiling)
    # every start is below 9001 game and every cap below the 15996 table top, so
    # no landing reaches T_MAX_PHYS: the rail is not what any assertion rests on
    books_ok = (sum(Eg[0]) - sum(E0[0]) == c.e_gas_deposit_sum + c.e_gas_rail_sum
                and c.t_max_phys_hits == 0)
    drop_ok = (c.e_rad_clamp_drop_sum == drop_sum) and (c.rad_clamp_hits == n_clamped)
    # ...and the same two books on THERMAL SOLIDS (fold_pass1_solid): the landing
    # priced at cap plus the withheld energy is the unclamped step, to the count
    n_s = n // 4
    Ts = [[rng.choice([0, 290 << 16, 804 << 16, 1263 << 16]) for _ in range(n_s)]]
    hs = [[rng.choice([-4, 0, 3, 5]) for _ in range(n_s)]]
    rs = [[rng.choice([1, -1]) * rng.choice([7, 1 << 14, 1 << 24, 1 << 34])
           for _ in range(n_s)]]
    ps = [[live[rng.choice([0, 72, 201, 315])] for _ in range(n_s)]]
    Ts0 = [r[:] for r in Ts]
    cs = R.FoldCounters()
    R.fold_pass1_solid(Ts, rs, ps, hs, R.plane(1, n_s, 1), cs, table=live)
    solid_books = cs.e_solid_deposit_sum + cs.e_rad_clamp_drop_sum
    solid_want = 0
    for i in range(n_s):
        cap = 1 << (hs[0][i] + 16)
        t_after = R.sat_add_q16(Ts0[0][i], R.fine_heat_shr(rs[0][i], hs[0][i], fb))
        ceiling = max(R.e_ceiling_q(ps[0][i], live), Ts0[0][i])
        t_cl = min(t_after, ceiling)
        t_railed = min(max(t_cl, 0), R.T_MAX_PHYS_Q)
        # the landing (post clamp, post rails -- the rails are counted by HITS,
        # the landing books what they allowed) plus the clamp's withheld step
        solid_want += (t_railed - Ts0[0][i]) * cap + (t_after - t_cl) * cap
    solid_ok = (solid_books == solid_want and cs.rad_clamp_hits > 0
                and cs.e_rad_clamp_drop_sum > 0)
    drop_ok &= solid_ok
    c_off = R.FoldCounters()
    T_off, E_off = [r[:] for r in T0], [r[:] for r in E0]
    R.fold_pass1_gas(T_off, E_off, rn, phi, nb, ts, c_off, table=live, clamp_enabled=False)
    over = sum(1 for i in range(n)
               if T_off[0][i] > max(R.e_ceiling_q(phi[0][i], live), T0[0][i]))
    good = (target_ok and step_ok and resid_ok and books_ok and drop_ok and mp_ok
            and n_clamped > 0 and n_free > 0 and n_letter_drain > 0 and over > 0
            and c_off.rad_clamp_hits == 0 and c_off.e_rad_clamp_drop_sum == 0)
    ok &= good
    # (g) P5c follow-up: the two other ways out of the boundary, counted -- on
    # (a)'s cells. BELOW THE FLOOR, per cell, the sweep's rn << 16 is exactly the
    # landing plus the clamp's drop plus the floor's remainder, and the remainder
    # is NOT a rounding (it exceeds the chain's declared precision); at or above
    # it nothing is booked there. OUTSIDE THE BOOKS (an acct mask), a cell is
    # untouched and its rn << 16 is exported, while every other cell folds
    # exactly as without the mask.
    floor_want = 0
    floor_exact = True
    n_floored = n_undiluted = 0
    for i in range(n):
        N, t0, r = nb[0][i], T0[0][i], rn[0][i]
        if N >= R.N_FLOOR_Q_LIVE:
            continue
        n_floored += 1
        cap = R.cap_real_q(False, 0, N)
        dT = R.gas_rad_dT_q(r, N, fine_bits=fb)
        t_after = R.sat_add_q16(t0, dT)
        r_q16 = R.fine_heat_shr(r, -16, fb)       # rn in the Q16 heat currency
        rem = r_q16 - (t_after - t0) * cap
        floor_want += rem
        ceiling = max(R.e_ceiling_q(phi[0][i], live), t0)
        t_land = min(t_after, ceiling)
        floor_exact &= (r_q16 == (t_land - t0) * cap + (t_after - t_land) * cap + rem)
        if abs(rem) > _chain_bound(r, N, cap, dT, fb):
            n_undiluted += 1
    floor_ok = (c.e_rad_floor_drop_sum == floor_want and floor_exact and n_floored > 0
                and n_undiluted > 0 and c_off.e_rad_floor_drop_sum == floor_want)
    mask_rng = random.Random(20260926)
    acct = [[1 if mask_rng.random() < 0.7 else 0 for _ in range(n)]]
    T_m, E_m = [r[:] for r in T0], [r[:] for r in E0]
    c_m = R.FoldCounters()
    R.fold_pass1_gas(T_m, E_m, rn, phi, nb, ts, c_m, acct=acct, table=live)
    out_cells = [i for i in range(n) if not acct[0][i]]
    export_want = sum(R.fine_heat_shr(rn[0][i], -16, fb) for i in out_cells)
    export_ok = (c_m.e_rad_boundary_export_sum == export_want and len(out_cells) > 0
                 and all(T_m[0][i] == T0[0][i] and E_m[0][i] == E0[0][i] for i in out_cells)
                 and all(T_m[0][i] == T[0][i] and E_m[0][i] == Eg[0][i]
                         for i in range(n) if acct[0][i])
                 and c.e_rad_boundary_export_sum == 0)
    good_g = floor_ok and export_ok
    ok &= good_g
    lines.append(f"  (a) {n} random accountable gas cells, LIVE table: {n_clamped} clamped, "
                 f"{n_free} not; every clamped mirror lands on max(T_before, e_ceiling(Phi)): "
                 f"{target_ok}; every other on sat(T_before + dT): {step_ok}; E mod N "
                 f"unchanged on every cell: {resid_ok}. The design's letter N*(T_target + "
                 f"t_amb) - E hits the same mirror but drains the residual on "
                 f"{n_letter_drain} clamped cells ({letter_drained} raw counts)")
    lines.append(f"  (b) sum(Eg) moved by e_gas_deposit_sum + e_gas_rail_sum exactly: "
                 f"{books_ok}; e_rad_clamp_drop_sum = sum (T_after - T_target) * cap_real "
                 f"= {c.e_rad_clamp_drop_sum} and rad_clamp_hits = {c.rad_clamp_hits}; on "
                 f"{n_s} thermal solids e_solid_deposit_sum + e_rad_clamp_drop_sum is the "
                 f"unclamped step priced at cap ({cs.rad_clamp_hits} clamped): {drop_ok}")
    lines.append(f"  (c) T_new <= max(T_before, e_ceiling(Phi)) on every gas cell: {mp_ok}; with "
                 f"the clamp off {over} cells exceed it (no hits, no drop booked)  "
                 f"{'OK' if good else 'FAIL'}")
    lines.append(f"  (g) the boundary's other two exits, counted: on the {n_floored} cells "
                 f"below n_floor, rn << 16 == landed + clamp drop + floor remainder per "
                 f"cell: {floor_exact}; e_rad_floor_drop_sum = {c.e_rad_floor_drop_sum} "
                 f"== the sum: {c.e_rad_floor_drop_sum == floor_want}, the clamp's switch "
                 f"leaves it alone: {c_off.e_rad_floor_drop_sum == floor_want}; the "
                 f"remainder exceeds the chain's declared precision on {n_undiluted} of "
                 f"them (a dilution, not a rounding); {len(out_cells)} cells OUTSIDE the "
                 f"books are untouched and exported whole (e_rad_boundary_export_sum = "
                 f"{c_m.e_rad_boundary_export_sum}), every other cell folds as unmasked: "
                 f"{export_ok}  {'OK' if good_g else 'FAIL'}")
    # (d) hot smoke cools in a sealed room, the whole tick, live table
    hh, ww = (7, 9) if fast else (9, 12)
    ticks = 24 if fast else 72
    sc = _sealed_smoky_room(hh, ww, smoke_q=Q(0.06), hq=Q(5.0), T_smoke_game=T_SRC_GAME)
    off = _sealed_smoky_room(hh, ww, smoke_q=Q(0.06), hq=Q(5.0), T_smoke_game=T_SRC_GAME)
    off.gas_fold = False
    interior = [(y, x) for y in range(hh) for x in range(ww) if not sc.ts[y][x]]
    walls = [(y, x) for y in range(hh) for x in range(ww) if sc.ts[y][x]]
    start = {p: sc.T[p[0]][p[1]] for p in interior}
    rises = below = 0
    bound_ok = True
    worst_ratio = 0.0
    n_touch = 0
    for _t in range(ticks):
        before_T = [r[:] for r in sc.T]
        before_E = [r[:] for r in sc.Eg]
        cdrop0 = sc.counters.e_rad_clamp_drop_sum
        rails0 = (sc.counters.t_max_phys_hits, sc.counters.t_low_rail_hits,
                  sc.counters.e_gas_rail_sum)
        res = sc.tick()
        off.tick()
        # (f) the boundary, per touched cell-tick
        landed = 0
        bound = 0
        rn_sum = 0
        for y in range(hh):
            for x in range(ww):
                r = res.rad_net[y][x]
                if r == 0:
                    continue
                n_touch += 1
                rn_sum += R.fine_heat_shr(r, -16, fb)
                if sc.ts[y][x]:
                    cap = R.cap_real_q(True, sc.his, 0)
                    landed += (sc.T[y][x] - before_T[y][x]) * cap
                    bound += cap
                else:
                    N = sc.n_bulk[y][x]
                    cap = R.cap_real_q(False, 0, N)
                    dE = sc.Eg[y][x] - before_E[y][x]
                    landed += (dE // N) * cap          # dE is a whole multiple of N
                    dT = R.gas_rad_dT_q(r, N, fine_bits=fb)
                    bound += _chain_bound(r, N, cap, dT, fb)
        cdrop = sc.counters.e_rad_clamp_drop_sum - cdrop0
        resid = rn_sum - landed - cdrop
        rails_now = (sc.counters.t_max_phys_hits, sc.counters.t_low_rail_hits,
                     sc.counters.e_gas_rail_sum)
        bound_ok &= (abs(resid) <= bound) and rails_now == rails0
        if bound:
            worst_ratio = max(worst_ratio, abs(resid) / bound)
        for (y, x) in interior:
            if sc.T[y][x] > before_T[y][x]:
                rises += 1
            if sc.T[y][x] < 0:
                below += 1
    cooled = all(sc.T[y][x] < start[(y, x)] for (y, x) in interior)
    walls_warm = sum(sc.T[y][x] for (y, x) in walls) > 0
    frozen = all(off.T[y][x] == start[(y, x)] for (y, x) in interior)
    good = rises == 0 and below == 0 and cooled and walls_warm and frozen
    ok &= good
    t_mean = sum(sc.T[y][x] for (y, x) in interior) / len(interior) / 65536.0
    t_min = min(sc.T[y][x] for (y, x) in interior) / 65536.0
    lines.append(f"  (d) a sealed {hh}x{ww} room of ambient walls full of {T_SRC_GAME}-game "
                 f"smoke (a_gas = 0.3), {ticks} ticks through the whole tick, LIVE table: "
                 f"cell-ticks where smoke ROSE {rises}, went BELOW ambient {below}; every "
                 f"cell cooled: {cooled} (mean {t_mean:.1f}, min {t_min:.1f} game); walls "
                 f"warmed: {walls_warm}; with the gas branch off the smoke never moved: "
                 f"{frozen}  {'OK' if good else 'FAIL'}")
    # (e) smoke shields, with the fold live
    ticks_e = 24 if fast else 96
    runs = {}
    for name, dens in (("clear", 0), ("thin", Q(0.02)), ("thick", Q(0.2))):
        s, tgt = _shield_scene(dens)
        series = []
        for _t in range(ticks_e):
            res = s.tick()
            series.append(sum(res.rad_net[y][x] for (y, x) in tgt))
        runs[name] = (series, s)
    clear, thin, thick = runs["clear"][0], runs["thin"][0], runs["thick"][0]
    every_tick = all(th < cl for th, cl in zip(thick, clear))
    ordered = all(tk < tn < cl for tk, tn, cl in zip(thick, thin, clear))
    s_thick = runs["thick"][1]
    smoke_cells = [(y, x) for y in range(len(s_thick.T)) for x in range(len(s_thick.T[0]))
                   if not s_thick.ts[y][x] and s_thick.gas[0][y][x] > 0]
    hot_smoke = max(s_thick.T[y][x] for (y, x) in smoke_cells) / 65536.0
    good = every_tick and ordered and hot_smoke > 0
    ok &= good
    lines.append(f"  (e) the target's absorbed flux (its rad_net, held at ambient), {ticks_e} "
                 f"ticks, LIVE table: clear {clear[0]} -> {clear[-1]}, thin smoke "
                 f"{thin[0]} -> {thin[-1]}, thick {thick[0]} -> {thick[-1]}; thick below "
                 f"clear on every tick: {every_tick}; thick < thin < clear on every tick: "
                 f"{ordered}; the smoke layer heated to {hot_smoke:.1f} game and still "
                 f"shields  {'OK' if good else 'FAIL'}")
    # (f) the boundary, measured over (d)
    good = bound_ok and n_touch > 0
    ok &= good
    lines.append(f"  (f) over (d)'s {ticks} ticks ({n_touch} touched cell-ticks, walls and "
                 f"smoke), sum(rad_net) - sum(landed) - e_rad_clamp_drop_sum stays within "
                 f"the conversions' own truncation every tick (worst |resid| / bound = "
                 f"{worst_ratio:.3g}), no rail engaged: {bound_ok}  {'OK' if good else 'FAIL'}")
    return ok, lines


# --------------------------------------------------------------------------- #
# P5d: THE CLAMP'S CEILING -- ONE BUCKET OF HEADROOM (Erik's ruling, 2026-09-24;
# docs/ray_engine_v2_p5d_clamp_headroom_brief_2026-09-24.md sections 5.1, 5.3,
# 5.4). The fold clamps at e_ceiling_q, the top of the first bucket whose E°
# exceeds Phi; this gate is its property sheet, and the two ceilings the ruling
# rejected are measured beside it so none of its properties can pass vacuously.
# --------------------------------------------------------------------------- #
ROW_WOOD = ("wood", Q(0.9), -3)                     # the shipped wood row (M2 thin)
ROW_FURNITURE = ("furniture", A_FURNITURE, HIS_OVERDRIVEN)   # the shipped furniture row


def _ceiling_low_edge(phi, table):
    """The PRE-P5d ceiling: E_inv(Phi), the low edge of Phi's own bucket."""
    return R.e_inv_q(phi, table)


def _ceiling_exact_inverse(phi, table):
    """The CONTINUOUS inverse of the staircase -- linear between the bucket
    MIDPOINTS the table is baked at (E°[b] sits at 4b + 2) -- i.e. what a patch
    that made E_inv 'exact' would ship. Not shipped; measured."""
    if phi < table[0]:
        return 0
    b = R.e_inv_q(phi, table) >> R.E_INDEX_SHIFT
    if b >= R.E_TABLE_SIZE - 1:
        return R.T_TABLE_TOP_GAME << 16
    lo, hi = table[b], table[b + 1]
    return ((4 * b + 2) << 16) + ((phi - lo) * (4 << 16)) // (hi - lo)


def _held_phi_march(phi, a_q, his, *, ticks, window, table, ceiling_fn=None,
                    clamp_enabled=True, t0_q=0):
    """One cell under a held Phi (cell_march, the fold's own arithmetic), marched
    FROM COLD: `ticks - window` ticks, then `window` more with fresh counters.
    Returns (trace of the last window, its counters). A cell_march's state is its
    T alone, so the split is exact; the approach is not what is measured."""
    t_mid, _ = R.cell_march(t0_q, phi, a_q, his, ticks - window, table=table,
                            ceiling_fn=ceiling_fn, clamp_enabled=clamp_enabled)
    tr, c = R.cell_march(t_mid, phi, a_q, his, window, table=table, trace=True,
                         ceiling_fn=ceiling_fn, clamp_enabled=clamp_enabled)
    return tr[1:], c


def gate16_clamp_ceiling(fast=False):
    """G16 (P5d, Erik's ruling of 2026-09-24): THE CLAMP'S CEILING -- ONE BUCKET
    OF HEADROOM. The fold clamps the radiative sub-step at max(T_before,
    e_ceiling_q(Phi)), the TOP of the first bucket whose E° exceeds Phi.

      (a) THE LOOKUP, over EVERY bucket of BOTH tables: for Phi in [E°[0],
          E°[3999]) -- each bucket's two edges and a point inside -- the ceiling
          c is the LAST Q16 value of its bucket, and that bucket is the FIRST
          whose E° exceeds Phi: E°[bucket(c) - 1] <= Phi < E°[bucket(c)]. c >=
          E_inv(Phi); c is monotone in Phi; Phi < E°[0] -> 0 (design row 31, the
          one deliberate exception); Phi >= E°[3999] -> (16000 << 16) - 1, below
          T_MAX_PHYS. The pair: E_inv, the pre-P5d ceiling, fails "first bucket
          above Phi" on every probe.
      (b) AN UNDAMPED CELL BALANCES WITH NOTHING WITHHELD (0-D, the LIVE table):
          the shipped wood and furniture rows under a held Phi whose continuous
          equilibrium T_cont spans the fire range, Phi in BOTH halves of its
          bucket: over the last W ticks the clamp withholds NOTHING, the mean T
          is within one bucket of T_cont, and the cell is undamped there (f ==
          2^24, else the case is not the one claimed). The pairs: the low edge
          (E_inv) withholds on every Phi; the exact inverse (linear between
          bucket midpoints) withholds wherever T_cont falls in Phi's own bucket
          -- the lower half -- which is why both halves are sampled.
      (c) THE CLAMP'S REAL JOB (0-D, the LIVE table): the wood row under a Phi far
          above the fire range (T_cont ~ 5 000 and ~ 11 000 game), where Fleck
          damps it (f < 2^24): the clamped march ends within [T_cont - 1 bucket,
          T_cont + 2 buckets], and the same march with the clamp OFF runs past
          that band -- Fleck alone does not hold it.

    Breaks if: the clamp's ceiling is reverted to e_inv_q (a: its bucket is
    Phi's own, not the first above it; b: the low edge withholds every tick), is
    made the exact inverse (b: the lower half withholds), loses its Phi < E°[0]
    -> 0 case or saturates at or above T_MAX_PHYS (a), or the clamp is removed
    (c: the damped cell runs away).
    """
    lines, ok = [], True
    # ---- (a) the lookup, every bucket, both tables -------------------------
    n_probe = bad = low_edge_fails = 0
    mono = True
    for _tname, tbl in (("resolving", R.E), ("live", R.E_LIVE)):
        last = -1
        for b in range(R.E_TABLE_SIZE - 1):
            lo, hi = tbl[b], tbl[b + 1]
            for phi in sorted({lo, lo + (hi - lo) // 2, hi - 1}):
                n_probe += 1
                c = R.e_ceiling_q(phi, tbl)
                bc = c >> R.E_INDEX_SHIFT
                top = ((c + 1) >> R.E_INDEX_SHIFT) == bc + 1
                first = 1 <= bc < R.E_TABLE_SIZE and tbl[bc - 1] <= phi < tbl[bc]
                if not (top and first and bc == R.e_bucket_of(c)
                        and c >= R.e_inv_q(phi, tbl)):
                    bad += 1
                mono &= c >= last
                last = c
                ci = R.e_inv_q(phi, tbl) >> R.E_INDEX_SHIFT
                low_edge_fails += int(not (1 <= ci and tbl[ci - 1] <= phi < tbl[ci]))
        below = [R.e_ceiling_q(v, tbl) for v in (0, tbl[0] - 1)]
        sat = [R.e_ceiling_q(v, tbl) for v in (tbl[R.E_TABLE_SIZE - 1],
                                              3 * tbl[R.E_TABLE_SIZE - 1])]
        edges = (below == [0, 0] and sat == [R.T_CEILING_TOP_Q] * 2
                 and R.T_CEILING_TOP_Q < R.T_MAX_PHYS_Q)
        bad += int(not edges)
    good = bad == 0 and mono and low_edge_fails == n_probe
    ok &= good
    lines.append(f"  (a) the lookup on {n_probe} probes (every bucket's two edges and "
                 f"midpoint, both tables): the ceiling is its bucket's top and that "
                 f"bucket is the first above Phi, >= E_inv, edge cases (0 below E°[0]; "
                 f"{R.T_CEILING_TOP_Q / 65536:.5f} game at the top, < T_MAX_PHYS): "
                 f"{bad} violations; monotone: {mono}. E_inv as the ceiling fails "
                 f"'first bucket above Phi' on {low_edge_fails} of {n_probe}  "
                 f"{'OK' if good else 'FAIL'}")
    # ---- (b) an undamped cell balances with nothing withheld ---------------
    live = R.E_LIVE
    # From cold the slowest case is the last bucket before the edge, where the
    # staircase's drive is only u x (E°[b+1] - E°[b]): the furniture row at 400
    # game and u = 0.15 needs ~13 000 ticks to cross it. `settled` below asserts
    # every shipped-ceiling case reached its balance inside the window.
    ticks, window = (8000, 2000) if fast else (20000, 2000)
    t_eqs = (800,) if fast else (400, 800, 1500)
    us = (0.2, 0.8) if fast else (0.15, 0.4, 0.6, 0.85)
    forms = (("headroom (shipped)", None), ("low edge (E_inv)", _ceiling_low_edge),
             ("exact inverse", _ceiling_exact_inverse))
    stats = {name: dict(n=0, withheld_cases=0, withheld=0, gross=0, dev=[], lower=0,
                        lower_withheld=0) for name, _ in forms}
    undamped = settled = True
    for _row, a_q, his in (ROW_WOOD, ROW_FURNITURE):
        for t_eq in t_eqs:
            b = t_eq // 4
            for u in us:
                phi = live[b] + int(u * (live[b + 1] - live[b]))
                t_cont = R.e_inv_float(phi, R.RAD_SCALE_LIVE, R.fine_bits_of(live))
                lower_half = t_cont < 4 * b + 4          # T_cont inside Phi's own bucket
                for name, fn in forms:
                    tr, c = _held_phi_march(phi, a_q, his, ticks=ticks, window=window,
                                            table=live, ceiling_fn=fn)
                    st = stats[name]
                    st["n"] += 1
                    st["withheld"] += c.e_rad_clamp_drop_sum
                    st["gross"] += window * R.fine_heat_shr((phi * a_q) >> 16, -16,
                                                            R.fine_bits_of(live))
                    st["withheld_cases"] += int(c.e_rad_clamp_drop_sum > 0)
                    st["dev"].append(sum(tr) / len(tr) / 65536.0 - t_cont)
                    st["lower"] += int(lower_half)
                    st["lower_withheld"] += int(lower_half and c.e_rad_clamp_drop_sum > 0)
                    if fn is None:
                        undamped &= all(R.fleck_f_solid_q(t, a_q, his, table=live)[0] == F_ONE
                                        for t in (min(tr), max(tr)))
                        # balanced, not still approaching: the window straddles the
                        # edge 4(b+1) where the bucket's E° crosses Phi
                        settled &= min(tr) < ((4 * (b + 1)) << 16) <= max(tr)
    head, low, exact = (stats[n] for n, _ in forms)
    good = (head["withheld_cases"] == 0 and all(abs(d) <= 4.0 for d in head["dev"])
            and undamped and settled
            and low["withheld_cases"] == low["n"]
            and exact["lower"] > 0 and exact["lower_withheld"] == exact["lower"]
            and head["lower"] > 0 and head["lower"] < head["n"])
    ok &= good
    lines.append(f"  (b) undamped rows (wood a 0.9 his -3, furniture a 0.5 his -2) under "
                 f"a held Phi, LIVE table, T_cont ~ {'/'.join(map(str, t_eqs))} game, Phi at "
                 f"{us} of its bucket ({head['lower']} of {head['n']} cases with T_cont in "
                 f"the lower half), {ticks} ticks from cold, stats over the last "
                 f"{window}; undamped there: {undamped}; every shipped-ceiling cell "
                 f"flickers across the edge where its bucket's E° crosses Phi: {settled}:")
    for name, _fn in forms:
        st = stats[name]
        lines.append(f"      {name:>20}: withheld / gross absorbed "
                     f"{100.0 * st['withheld'] / st['gross']:7.4f} %, on {st['withheld_cases']:2d}"
                     f" of {st['n']} cases (lower half {st['lower_withheld']} of "
                     f"{st['lower']}); mean T - T_cont in "
                     f"[{min(st['dev']):+.2f}, {max(st['dev']):+.2f}] game")
    lines.append(f"      the shipped ceiling withholds nothing and sits within one bucket; "
                 f"the low edge withholds on every case, the exact inverse on every "
                 f"lower-half case  {'OK' if good else 'FAIL'}")
    # ---- (c) the clamp's real job: a Fleck-damped cell -----------------------
    _row, a_q, his = ROW_WOOD
    ticks_c = 200 if fast else 600
    good_c = True
    for t_eq in (5000, 11000):
        b = t_eq // 4
        phi = live[b] + (live[b + 1] - live[b]) // 2
        t_cont = R.e_inv_float(phi, R.RAD_SCALE_LIVE, R.fine_bits_of(live))
        t_c, cc = R.cell_march(0, phi, a_q, his, ticks_c, table=live)
        t_u, _cu = R.cell_march(0, phi, a_q, his, ticks_c, table=live, clamp_enabled=False)
        f_c = R.fleck_f_solid_q(t_c, a_q, his, table=live)[0]
        held = (t_cont - 4.0) <= t_c / 65536 <= (t_cont + 8.0)
        runaway = t_u / 65536 > t_cont + 8.0
        g = held and runaway and f_c < F_ONE and cc.rad_clamp_hits > 0
        good_c &= g
        lines.append(f"  (c) wood under Phi with T_cont = {t_cont:8.1f} game, {ticks_c} ticks, "
                     f"LIVE: clamped -> {t_c / 65536:8.2f} (f = {f_c / F_ONE:.4f}, "
                     f"{cc.rad_clamp_hits} hits; band [{t_cont - 4:.1f}, {t_cont + 8:.1f}]), "
                     f"clamp OFF -> {t_u / 65536:9.1f} game (Fleck alone, "
                     f"{t_u / 65536 - t_cont:+.1f})  {'OK' if g else 'FAIL'}")
    ok &= good_c
    return ok, lines


# --------------------------------------------------------------------------- #
# #78: THE SWEEP'S FINE HEAT CURRENCY (docs/sweep_fine_heat_currency_brief_78_
# 2026-09-25.md sections 5.2 / 5.3). The live table is baked 2^FINE_BITS finer
# than a heat count; this gate is what that buys near ambient -- every property
# PAIRED with the pre-#78 live table: the same scale, in whole heat counts.
# --------------------------------------------------------------------------- #
E_LIVE_COARSE = R.bake_e_table(rad_scale=R.RAD_SCALE_LIVE)   # k = 0: the pre-#78 live table


def _lone_cell_scene(n, a_q, T_q, kleak):
    """One absorbing cell (a_q) at T_q in the middle of an n x n transparent room
    of ambient air: every stream it sees is the ambient ring's, unmodified."""
    a = R.plane(n, n, 0)
    T = R.plane(n, n, 0)
    c = n // 2
    a[c][c] = a_q
    T[c][c] = T_q
    return a, [r[:] for r in a], R.plane(n, n, kleak), T, (c, c)


def _near_ambient_scene(rng, h, w):
    """Shipped rows, air and bodies at temperatures from 8 game below to 20 above
    ambient -- where the per-ordinate floors decided the exchange before #78."""
    rows = [(a_q, his) for a_q, his, _nm in SHIPPED_ROWS]
    a = R.plane(h, w, 0)
    his = R.plane(h, w, 3)
    T = R.plane(h, w, 0)
    for y in range(h):
        for x in range(w):
            if rng.random() < 0.6:
                a[y][x], his[y][x] = rng.choice(rows)
            T[y][x] = rng.choice([-8, -4, 0, 0, 0, 2, 4, 6, 8, 12, 20]) << 16
    d = [[min(ONE, a[y][x] + rng.choice([0, 0, 0, 0, ONE - a[y][x]]))
          for x in range(w)] for y in range(h)]
    return a, d, T, his


def gate17_fine_heat_currency(fast=False):
    """G17 (#78): THE FINE HEAT CURRENCY RESOLVES THE EXCHANGE NEAR AMBIENT.

      (a) RADIATIVE COOLING NEAR AMBIENT (brief 5.2): on the LIVE table (fine,
          k = FINE_BITS) one cell of every shipped absorbing row at a temperature
          in bucket 1 or 2 (4 .. 12 game), alone in a transparent room of ambient
          air, books rad_net < 0 -- both transports, S16 and S12, the leak on and
          off -- and the same cell AT ambient (0) or below it (-4 game: the table
          floors at bucket 0) books exactly 0, on every plane of every cell. The
          PAIR: the pre-#78 live table (the same scale in whole heat counts) books
          exactly 0 on every S16 case -- the excess (7 and 14 counts) times
          w_m >> 16 floors to 0 in every ordinate, so such a cell could not shed it
          (at S12, w_m = 5461, bucket 2's excess reaches one count per ordinate).
      (b) THE UNIFORM AMBIENT FIXED POINT and THE IDENTITY survive the currency
          (brief 5.3): gate 2a's scene (bodies, the leak, non-dyadic a, f = 0.3
          forced) on the fine table is a per-cell exact zero on all three planes,
          S16 and S12, both transports, with absorbing smoke too; and Sum(rad_net)
          + Sum(rad_flux) + Sum(rad_amb) == 0 exactly on randomised fine scenes
          (gate 1's vocabulary), each sum non-zero.
      (c) WHAT ROW 31 STILL CATCHES (brief section 3; measured): the ambient field
          reads Phi = n x (E°_k[0] w_m >> 16), a few FINE counts below E°_k[0], so
          a cell at ambient still takes row 31's 0 ceiling. What a net absorber
          with Phi < E°[0] can book there -- on randomised near-ambient scenes of
          shipped rows, bodies and air at -8 .. 20 game -- is at most
          (E°[0] - n amb_m) + n counts of the table's currency, which on the fine
          table is under 1/32 of a heat count; the pre-#78 table books WHOLE heat
          counts there (the P5d bench's "shadow class", 11-13 on a thin panel).
      (d) STAGNATION IS GONE, through the whole tick (pre-pass, sweep, fold): a
          thin shipped row (his < 0) at 10 game alone in the ambient room cools on
          every tick and never below ambient on the fine table; on the pre-#78
          table it does not move at all (rad_net == 0: Croci & Giles' stagnation).
          The heavy rows' landing is reported: near ambient it is under one
          temperature LSB per tick, the temperature FIELD's own resolution (the
          fold's shr_round0, #73), not the sweep's.

    Breaks if: the live table is baked coarse (k = 0: (a), (c) and (d) fail,
    measured by their pairs), a reader converts absorption and emission in two
    currencies (b's fixed point breaks), or the excess stops being formed over the
    cell's own ambient (b).
    """
    lines, ok = [], True
    live, coarse = R.E_LIVE, E_LIVE_COARSE
    fb = R.fine_bits_of(live)
    n_room = 5 if fast else 7
    # ---- (a) near-ambient cooling, every shipped row, buckets 1 and 2 ----------
    temps = [4 << 16, 8 << 16, (12 << 16) - 1] if fast else \
        [4 << 16, 6 << 16, 8 << 16, 10 << 16, (12 << 16) - 1]
    n_case = n_neg = n_coarse_zero = n_s16 = n_s16_zero = 0
    ambient_zero = True
    least = None                                 # the least negative live rad_net
    for a_q, his, _nm in SHIPPED_ROWS:
        for transport in ("shear", "step"):
            for n_ord in (16, 12):
                for kleak in (0, K_LEAK):
                    for T_q in temps:
                        a, d, k, T, (cy, cx) = _lone_cell_scene(n_room, a_q, T_q, kleak)
                        rn = R.sweep_q(a, d, k, T, transport=transport, n_ord=n_ord,
                                       f_plane=R.fleck_prepass(T, a, his, table=live),
                                       table=live).rad_net[cy][cx]
                        rc = R.sweep_q(a, d, k, T, transport=transport, n_ord=n_ord,
                                       f_plane=R.fleck_prepass(T, a, his, table=coarse),
                                       table=coarse).rad_net[cy][cx]
                        n_case += 1
                        n_neg += int(rn < 0)
                        n_coarse_zero += int(rc == 0)
                        if n_ord == 16:
                            n_s16 += 1
                            n_s16_zero += int(rc == 0)
                        least = rn if least is None else max(least, rn)
                    for T_q in (0, -(4 << 16)):
                        a, d, k, T, _c = _lone_cell_scene(n_room, a_q, T_q, kleak)
                        for tbl in (live, coarse):
                            r = R.sweep_q(a, d, k, T, transport=transport, n_ord=n_ord,
                                          f_plane=R.fleck_prepass(T, a, his, table=tbl),
                                          table=tbl)
                            ambient_zero &= all(v == 0 for p in (r.rad_net, r.rad_flux,
                                                                 r.rad_amb)
                                                for row in p for v in row)
    good = (n_neg == n_case and ambient_zero and n_s16_zero == n_s16 and n_s16 > 0)
    ok &= good
    lines.append(f"  (a) {n_case} cases ({len(SHIPPED_ROWS)} shipped (a, his) pairs x "
                 f"shear/step x S16/S12 x leak on/off x T in buckets 1-2): rad_net < 0 "
                 f"on {n_neg} on the FINE live table (k = {fb}; the least negative "
                 f"{least} fine = {least / (1 << fb):.2f} heat counts per tick); the "
                 f"cell at ambient or below books exactly 0 everywhere, both tables: "
                 f"{ambient_zero}. PAIR: the pre-#78 live table books exactly 0 -- no "
                 f"radiative cooling at all -- on {n_coarse_zero} of {n_case}, every one "
                 f"of the {n_s16} S16 cases among them (its excess floors away in every "
                 f"ordinate)  {'OK' if good else 'FAIL'}")
    # ---- (b) the fixed point and the identity, on the fine table ---------------
    h, w = (7, 8) if fast else (8, 10)
    rng = random.Random(4)                       # gate 2a's scene, draw for draw
    a = [[rng.choice([0, Q(0.37), Q(0.91), ONE]) for _ in range(w)] for _ in range(h)]
    d = [[min(ONE, a[y][x] + rng.choice([0, Q(0.5), ONE - a[y][x]]))
          for x in range(w)] for y in range(h)]
    k = R.plane(h, w, K_LEAK)
    T0 = R.plane(h, w, 0)
    f03 = R.plane(h, w, QF(0.3))
    gkw = _gas_kw(random.Random(40), a)
    nz_all = 0
    for transport in ("shear", "step"):
        for n_ord in (16, 12):
            for kw in ({}, gkw):
                r = R.sweep_q(a, d, k, T0, n_ord=n_ord, transport=transport,
                              f_plane=f03, table=live, **kw)
                nz_all += sum(1 for p in (r.rad_net, r.rad_flux, r.rad_amb)
                              for row in p for v in row if v != 0)
    rng1 = random.Random(20260925)
    ident_ok = True
    for transport in ("shear", "step"):
        for n_ord in (16, 12):
            aa, dd, TT, ff = _rand_scene(rng1, h, w)
            for kw in ({}, _gas_kw(rng1, aa)):
                r = R.sweep_q(aa, dd, R.plane(h, w, K_LEAK), TT, n_ord=n_ord,
                              transport=transport, f_plane=ff, table=live, **kw)
                sn, sf, sa = r.sums()
                ident_ok &= (sn + sf + sa == 0) and sn != 0 and sf != 0 and sa != 0
    good = nz_all == 0 and ident_ok
    ok &= good
    lines.append(f"  (b) gate 2a's ambient scene (bodies, k = 0.10, f = 0.3, with and "
                 f"without smoke) on the fine table: {nz_all} non-zero cells over the "
                 f"three planes; the identity exact with every sum non-zero on "
                 f"randomised fine scenes: {ident_ok}  {'OK' if good else 'FAIL'}")
    # ---- (c) what row 31 still catches ----------------------------------------
    rng2 = random.Random(78)
    caught = {"fine": [0, 0, 0], "coarse": [0, 0, 0]}   # n cell-ticks, max, bound viol.
    trials = 3 if fast else 8
    hh, ww = (7, 9) if fast else (9, 11)
    for _trial in range(trials):
        aa, dd, TT, hs = _near_ambient_scene(rng2, hh, ww)
        for transport in ("shear", "step"):
            for n_ord in (16, 12):
                for kleak in (0, K_LEAK):
                    for name, tbl in (("fine", live), ("coarse", coarse)):
                        amb_m = (tbl[0] * (ONE // n_ord)) >> 16
                        bound = (tbl[0] - n_ord * amb_m) + n_ord
                        r = R.sweep_q(aa, dd, R.plane(hh, ww, kleak), TT, n_ord=n_ord,
                                      transport=transport, table=tbl,
                                      f_plane=R.fleck_prepass(TT, aa, hs, table=tbl))
                        for y in range(hh):
                            for x in range(ww):
                                rn = r.rad_net[y][x]
                                if r.rad_fluence[y][x] < tbl[0] and rn > 0:
                                    c = caught[name]
                                    c[0] += 1
                                    c[1] = max(c[1], rn)
                                    c[2] += int(rn > bound)
    f_n, f_max, f_viol = caught["fine"]
    c_n, c_max, c_viol = caught["coarse"]
    amb16 = 16 * ((live[0] * 4096) >> 16)
    good = (f_viol == 0 and c_viol == 0 and f_max < (1 << fb) // 32 and c_max >= 1
            and c_n > 0)
    ok &= good
    lines.append(f"  (c) row 31 still catches: the ambient field reads Phi = {amb16} = "
                 f"E°[0] - {live[0] - amb16} fine counts (S16), so an ambient cell takes "
                 f"the 0 ceiling. On near-ambient random scenes the Phi < E°[0] net "
                 f"absorbers book at most {f_max} fine counts = {f_max / (1 << fb):.4f} "
                 f"heat counts per cell-tick ({f_n} cell-ticks; the derived bound "
                 f"(E°[0] - n amb_m) + n never exceeded: {f_viol == 0}); the pre-#78 "
                 f"table books up to {c_max} WHOLE heat counts there ({c_n} cell-ticks)  "
                 f"{'OK' if good else 'FAIL'}")
    # ---- (d) stagnation is gone, through the whole tick ------------------------
    ticks = 48 if fast else 240
    thin_ok = True
    notes = []
    for a_q, his, nm in SHIPPED_ROWS:
        runs = {}
        for name, tbl in (("fine", live), ("coarse", coarse)):
            a, d, k, T, (cy, cx) = _lone_cell_scene(n_room, a_q, 10 << 16, K_LEAK)
            sc = R.Scene(a=a, d=d, k=k, T=T, his=his, table=tbl)
            tr = [sc.T[cy][cx]]
            for _t in range(ticks):
                sc.tick()
                tr.append(sc.T[cy][cx])
            runs[name] = tr
        fine, crs = runs["fine"], runs["coarse"]
        mono = all(fine[i + 1] <= fine[i] for i in range(ticks)) and min(fine) >= 0
        if his < 0:
            thin_ok &= mono and fine[-1] < fine[0] and all(v == crs[0] for v in crs)
        notes.append(f"{nm}: {fine[0] - fine[-1]} LSB fine vs {crs[0] - crs[-1]} coarse")
    ok &= thin_ok
    lines.append(f"  (d) one cell at 10 game in the ambient room, {ticks} full ticks, the "
                 f"temperature drop in Q16 LSBs (thin rows asserted: cools monotonically, "
                 f"never below ambient, where the pre-#78 table does not move): "
                 + "; ".join(notes) + f"  {'OK' if thin_ok else 'FAIL'}")
    return ok, lines


GATES = [
    ("G1  conservation", gate1_conservation),
    ("G2a uniform ambient fixed point", gate2a_uniform_ambient),
    ("G2b enclosed isothermal box", gate2b_isothermal_box),
    ("G3  positivity + ingress rejection", gate3_positivity),
    ("G4  counters, per counter", gate4_counters),
    ("G5  maximum principle", gate5_maximum_principle),
    ("G6  isotropy", gate6_isotropy),
    ("G7  agreement with the float push reference", gate7_float_agreement),
    ("G8  the Fleck integer form", gate8_fleck_form),
    ("G9  E_inv", gate9_e_inv),
    ("G10 stability and equilibrium on the new forms", gate10_stability),
    ("G11 headroom", gate11_headroom),
    ("G12 the damped source is monotone in T", gate12_damped_source_is_monotone),
    ("G13 the gas extinction: density law, N_EPS floor, transmission",
     gate13_gas_extinction),
    ("G14 the gas arm of the Fleck pre-pass: monotone cooling, never below ambient",
     gate14_gas_fleck_arm),
    ("G15 the fold's gas branch: the clamp's energy form, the books, cooling, "
     "shielding, the 8.4 boundary", gate15_gas_fold),
    ("G16 the clamp's ceiling: one bucket of headroom", gate16_clamp_ceiling),
    ("G17 the fine heat currency: near-ambient exchange resolved (#78)",
     gate17_fine_heat_currency),
]


def main(argv):
    fast = "--fast" in argv
    only = [a for a in argv[1:] if not a.startswith("--")]
    ok_cfg, detail = R.config_dials_match()
    print(f"config.toml dials {'MATCH' if ok_cfg else 'DRIFTED'}: {detail}")
    print(f"E°[0] = {R.E0}   E°[3999] = {R.E[3999]}   "
          f"amb_m(S16) = {(R.E0 * (ONE // 16)) >> 16}   "
          f"mode: {'fast' if fast else 'full'}")
    print(f"  (the RESOLVING default table, rad_scale = {R.RAD_SCALE:g}, whole heat "
          f"counts; the LIVE table, rad_scale_derived = {R.RAD_SCALE_LIVE:g}, in the "
          f"FINE currency 2^{R.fine_bits_of(R.E_LIVE)} per heat count (#78): E°[0] = "
          f"{R.E_LIVE[0]}, E°[3999] = {R.E_LIVE[3999]} -- G12's shipped rows are "
          f"measured on it)")
    all_ok = ok_cfg
    for name, fn in GATES:
        if only and not any(name.lower().startswith(o.lower()) for o in only):
            continue
        print(f"\n{name}\n{'-' * len(name)}")
        try:
            ok, lines = fn(fast)
        except Exception as exc:                      # noqa: BLE001
            ok, lines = False, [f"  RAISED: {type(exc).__name__}: {exc}"]
        for ln in lines:
            print(ln)
        print(f"  => {'PASS' if ok else 'FAIL'}")
        all_ok &= ok
    print(f"\n{'ALL GATES PASS' if all_ok else 'SOME GATES FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
