"""P0 GATES for the integer reference (`sweep_ref_q.py`), 2026-09-15 (+ P0b).

Runs the twelve gates of `docs/ray_engine_v2_design_v3_2026-09-15.md` sections
2.8-2.9 and of critique 3 sections 1-4, prints the MEASURED numbers (never a bare
boolean), and exits non-zero if any fails.

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


def _overdriven_scene(src_game=16000, gap=1, n=9):
    """A held source one tile from a cold crate (design gate 4's over-drive)."""
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
    sc = R.Scene(a=a, d=d, k=k, T=T, his=HIS_FURNITURE, held=held)
    return sc, (cy, cx)


def gate4_counters(fast=False):
    """G4. Per counter (design row 31). On the marine-beside-an-ambient-wall scene
    the low rail and the clamp are BOTH silent over 24 ticks; on the over-driven
    scene the clamp counter is non-zero and the T_MAX_PHYS rail stays silent
    (E_inv saturates at 15996, below the rail).

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
    # starts cold and CLIMBS to its cap, so this sub-check needs its own tick
    # count: at alpha floor 0 (P2a, row 39) the crate's own emission is undamped
    # below g = 1, so it climbs more slowly and the first clamp hit falls at tick
    # 11, where the superseded floor of one half had it at tick 8. The fast mode's
    # 8 ticks would therefore pass VACUOUSLY (clamp = 0, crate at 1074.6 game) --
    # measured, and the reason for the separate number below. 24 ticks: 14 hits at
    # floor 0, 17 at floor 1/2; the CLAMPED answer is 1108.0 game at both.
    o_ticks = 12 if fast else 24
    sc_o, crate = _overdriven_scene()
    sc_o.run(o_ticks)
    co = sc_o.counters
    good = co.rad_clamp_hits > 0 and co.t_max_phys_hits == 0 and co.t_low_rail_hits == 0
    ok &= good
    lines.append(f"  over-driven (16000-game source one tile from a cold crate), "
                 f"{o_ticks} ticks: clamp={co.rad_clamp_hits} t_max={co.t_max_phys_hits} "
                 f"low_rail={co.t_low_rail_hits}, crate T = "
                 f"{sc_o.T[crate[0]][crate[1]] / 65536:.1f} game  "
                 f"{'OK' if good else 'FAIL'}")
    return ok, lines


def gate5_maximum_principle(fast=False):
    """G5. Immediately after the radiative sub-step, on that sub-step alone:
    T_new <= max(T_before, E_inv(Phi)) for every cell.

    Non-vacuity: (a) with the clamp disabled the same cell climbs past the cap
    (reported); (b) a burning crate (T_before far above E_inv(Phi)) is NOT
    clamped -- the bare v2 form would have slammed it down, and that number is
    reported beside it.
    """
    lines, ok = [], True
    ticks = 8 if fast else 24
    sc, crate = _overdriven_scene()
    worst_excess = None
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
                cap = max(before[y][x], R.e_inv_q(res.rad_fluence[y][x]))
                ex = sc.T[y][x] - cap
                worst_excess = ex if worst_excess is None else max(worst_excess, ex)
    ok &= (worst_excess is not None and worst_excess <= 0)
    lines.append(f"  clamped, {ticks} ticks: worst (T_new - max(T_before, E_inv(Phi))) "
                 f"= {worst_excess} counts  (<= 0 required)  clamp hits = "
                 f"{sc.counters.rad_clamp_hits}  "
                 f"{'OK' if worst_excess is not None and worst_excess <= 0 else 'FAIL'}")
    t_clamped = sc.T[crate[0]][crate[1]]
    # (a) the same scene with the clamp disabled
    sc2, crate2 = _overdriven_scene()
    sc2.run(ticks * 4, clamp_enabled=False)
    t_unclamped = sc2.T[crate2[0]][crate2[1]]
    ok &= (t_unclamped > t_clamped)
    lines.append(f"  non-vacuity (a): clamp_enabled=False, {ticks*4} ticks -> crate "
                 f"{t_unclamped / 65536:.1f} game vs clamped {t_clamped / 65536:.1f} "
                 f"game (cap = {R.e_inv_q(R.sweep_q(sc.a, sc.d, sc.k, sc.T).rad_fluence[crate[0]][crate[1]]) >> 16})")
    # the 0-D equilibrium the design quotes, where the geometric factor is 0.25
    phi = int(0.25 * R.E[R.e_bucket_of(16000 << 16)])
    t_run, _ = R.cell_march(0, phi, A_FURNITURE, HIS_FURNITURE, 400 * 24,
                            clamp_enabled=False, rails_enabled=False, int32_sat=False)
    t_true = R.e_inv_q(phi) >> 16
    ok &= (t_run / 65536 > 1e6)
    lines.append(f"  non-vacuity (a'): the 0-D G=0.25 model the design quotes -- "
                 f"Fleck-only equilibrium = {t_run / 65536:,.0f} game vs the true "
                 f"{t_true} game (clamped)")
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
    R.fold_pass1_solid(sc3.T, res3.rad_net, res3.rad_fluence, sc3.his, sc3.ts,
                       sc3.counters)
    cap_bare = R.e_inv_q(res3.rad_fluence[3][3])
    bound = (sc3.counters.rad_clamp_hits == 0) and (sc3.T[3][3] < before)
    ok &= bound
    lines.append(f"  non-vacuity (b): a burning crate at {before >> 16} game, "
                 f"E_inv(Phi) = {cap_bare >> 16} game -> corrected clamp does not bind "
                 f"(hits={sc3.counters.rad_clamp_hits}, T {before / 65536:.1f} -> "
                 f"{sc3.T[3][3] / 65536:.1f}); the bare v2 form would have set it to "
                 f"{cap_bare >> 16} game in one tick  {'OK' if bound else 'FAIL'}")
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
          clamp still pins every equilibrium to E°inv(Phi) exactly.

    Breaks if: the Fleck factor stops damping above g = 1 (explicit overshoots and
    the low rail fires), or alpha's floor moves back off 0 (the fire-range
    trajectory stops matching explicit and the un-clamped 1263-game equilibrium
    jumps from +0.15 % to +4.6 %).
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
    lines.append(f"    {'T_src':>8}{'true (E_inv)':>14}{'true (float)':>14}"
                 f"{'Fleck only, 0':>16}{'Fleck only, 1/2':>17}"
                 f"{'Fleck+clamp':>13}{'clamp hits':>12}")
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
        good = (t_c >> 16) == t_true
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
    the Fleck factor's fixed point grows enough to need more than int64.
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
    return ok, lines


def _backward_steps(a_q, his, shift):
    """The damped source E°[0] + f*(E°[T] - E°[0]) over the WHOLE table, at the
    bucket LOW EDGE (T_q = (4b) << 16, so e_bucket_of returns exactly b), and the
    places where it FALLS as T rises.

    Returns (n_backward, worst_fraction, first_T_game, all_positive_and_sane).
    """
    vals = [R.damped_source_q((4 * b) << 16, a_q, his, shift=shift)
            for b in range(R.E_TABLE_SIZE)]
    sane = all(0 < v <= R.E[b] for b, v in enumerate(vals))
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
    """
    lines, ok = [], True
    lines.append(f"  the damped source at every bucket's LOW EDGE, T = 0..{4 * (R.E_TABLE_SIZE - 1)} "
                 f"game, {len(SHIPPED_ROWS)} distinct shipped (a, his) pairs read from "
                 f"config.toml + the pathological corner, at the RULED alpha floor "
                 f"{R.ALPHA_FLOOR_DEFAULT!r} (row 39)")
    lines.append(f"    {'row':>34}{'Q16 steps':>11}{'Q16 worst':>11}"
                 f"{'Q24 steps':>11}{'Q24 worst':>11}{'f_q24 @ top':>13}")
    q16_nonmono = 0
    for a_q, his, nm in SHIPPED_ROWS:
        n16, w16, _f16, _s16 = _backward_steps(a_q, his, 16)
        n24, w24, first24, sane24 = _backward_steps(a_q, his, R.F_SHIFT)
        q16_nonmono += n16
        good = (n24 == 0) and sane24
        ok &= good
        f_top = R.fleck_f_solid_q(R.T_TABLE_TOP_GAME << 16, a_q, his)[0]
        lines.append(f"    {nm:>34}{n16:>11}{w16 * 100:>10.3f}%{n24:>11}"
                     f"{w24 * 100:>10.3f}%{f_top:>13}  "
                     + ("OK (monotone; positive and <= E°[T] everywhere)" if good
                        else f"FAIL (first backward step at {first24} game, "
                             f"sane={sane24})"))
    # NON-VACUITY: the same probe on the form the design rejected is NOT monotone,
    # so "0 backward steps" above is a property of Q24 and not of the measurement.
    ok &= (q16_nonmono > 0)
    lines.append(f"  non-vacuity: the rejected Q16 form has {q16_nonmono} backward steps "
                 f"over the same rows (the Q16 columns above) -- the probe detects "
                 f"non-monotonicity when it is there")
    # THE MECHANISM: the wobble is f_q's own resolution, so it grows as the thermal
    # mass falls. No material ships thermal_mass = 1; if one ever does, the row
    # above fails and P1 owes the ingress rule rather than a wider f.
    a_p, his_p, nm_p = PATHOLOGICAL_ROW
    n16p, w16p, _, _ = _backward_steps(a_p, his_p, 16)
    n24p, w24p, first_p, sane_p = _backward_steps(a_p, his_p, R.F_SHIFT)
    bounded = (w24p < PATHOLOGICAL_WOBBLE_BOUND) and sane_p and (w24p < w16p)
    ok &= bounded
    lines.append(f"  the mechanism -- {nm_p}: f_q24 at the table top is only "
                 f"{R.fleck_f_solid_q(R.T_TABLE_TOP_GAME << 16, a_p, his_p)[0]} counts, so "
                 f"{n24p} backward steps survive, worst {w24p * 100:.3f}% in power "
                 f"(= {(1 + w24p) ** 0.25 * 100 - 100:.3f}% in a receiver's equilibrium T), "
                 f"first at {first_p} game; in Q16 the same row gave {n16p} steps of "
                 f"{w16p * 100:.2f}%. Asserted bound: no backward step above "
                 f"{PATHOLOGICAL_WOBBLE_BOUND * 100:.1f}% for ANY row, shipped or not  "
                 f"{'OK' if bounded else 'FAIL'}")
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
]


def main(argv):
    fast = "--fast" in argv
    only = [a for a in argv[1:] if not a.startswith("--")]
    ok_cfg, detail = R.config_dials_match()
    print(f"config.toml dials {'MATCH' if ok_cfg else 'DRIFTED'}: {detail}")
    print(f"E°[0] = {R.E0}   E°[3999] = {R.E[3999]}   "
          f"amb_m(S16) = {(R.E0 * (ONE // 16)) >> 16}   "
          f"mode: {'fast' if fast else 'full'}")
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
