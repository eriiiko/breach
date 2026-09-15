"""P0 GATES for the integer reference (`sweep_ref_q.py`), 2026-09-15.

Runs the eleven gates of `docs/ray_engine_v2_design_v3_2026-09-15.md` sections
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
Q = R.quant

# the scene constants the whole arc uses
T_SRC_GAME = 1263          # the measured crate plateau (K = 1556)
T_IGN_GAME = 280           # furniture ignition_temp  (K = 573)
A_FURNITURE = Q(0.5)       # furniture-class absorptivity
HIS_FURNITURE = 3          # log2(thermal_mass)
K_LEAK = Q(0.10)           # the derived 2.5 m deck leak (dormant in the engine)

# every (heat_atten, log2(thermal_mass)) pair the shipped material table carries
# (config.toml [materials.*]; air and foliage have heat_atten = 0 and never emit)
SHIPPED_ROWS = ((ONE, 3, "wood / door"),
                (ONE, 5, "hull / steel"),
                (Q(0.5), 3, "furniture / kindling"),
                (Q(0.3), 4, "glass"))


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
    f = [[rng.choice([ONE, Q(0.7), Q(0.05)]) for _ in range(w)] for _ in range(h)]
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
    f = R.plane(h, w, Q(0.3))
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
    src_v2 = (amb_m * Q(0.995)) >> 16
    net_v2 = ((amb_m * Q(0.91)) >> 16) - ((src_v2 * Q(0.91)) >> 16)
    ok &= (net_v2 != 0)
    lines.append(f"  non-vacuity: v2 whole-emission Fleck (f=0.995, a=0.91) gives "
                 f"rad_net = {net_v2} counts per ordinate on an ambient cell "
                 f"(the excess form gives 0)")
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
    f = R.plane(n, n, Q(0.3))
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
    below, trips the low rail every tick on every wall cell).
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
    # over-driven: the clamp must engage, the T_MAX_PHYS rail must not
    sc_o, crate = _overdriven_scene()
    sc_o.run(ticks)
    co = sc_o.counters
    good = co.rad_clamp_hits > 0 and co.t_max_phys_hits == 0 and co.t_low_rail_hits == 0
    ok &= good
    lines.append(f"  over-driven (16000-game source one tile from a cold crate), "
                 f"{ticks} ticks: clamp={co.rad_clamp_hits} t_max={co.t_max_phys_hits} "
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
    """G8. f_q = floordiv((T_abs<<16), max(T_abs+2L, 4L)) equals the float
    1/(1 + alpha*g) with alpha = max(1/2, 1 - 1/g) to within one count over the
    WHOLE table, and f_q == ONE exactly when L_q == 0.

    Breaks if: alpha's max() is re-expanded wrongly, the denominator moves off the
    ABSOLUTE temperature, or the division stops being the kit's exact floordiv.
    """
    lines, ok = [], True
    worst = 0.0
    worst_at = None
    stride = 1          # the whole table either way: 20 000 probes is free
    pairs = [(a, h) for a, h, _ in SHIPPED_ROWS] + [(ONE, 0)]  # + the his=0 corner
    iff_ok = True
    for a_q, his in pairs:
        for b in range(0, R.E_TABLE_SIZE, stride):
            T_game = 4 * b
            T_q = T_game << 16
            f_q, L_q = R.fleck_f_solid_q(T_q, a_q, his)
            f_f = R.fleck_f_float(float(T_game), L_q / 65536.0)
            err = abs(f_q / ONE - f_f)
            if err > worst:
                worst, worst_at = err, (T_game, a_q, his)
            if not (0 < f_q <= ONE):
                ok = False
            iff_ok &= ((f_q == ONE) == (L_q == 0))
    ok &= (worst <= 1.0 / ONE) and iff_ok
    lines.append(f"  worst |f_q/ONE - f_float| over {R.E_TABLE_SIZE // stride} buckets "
                 f"x {len(pairs)} (a, his) pairs = {worst:.2e}  (one count = {1/ONE:.2e}) at "
                 f"T={worst_at[0]} a={worst_at[1]} his={worst_at[2]}  "
                 f"{'OK' if worst <= 1/ONE else 'FAIL'}")
    lines.append(f"  f_q == ONE iff L_q == 0: {iff_ok}")
    for T_game in (0, 280, 1263, 1800, 15996):
        f_q, L_q = R.fleck_f_solid_q(T_game << 16, Q(0.5), 3)
        g = 4.0 * (L_q / 65536.0) / (T_game + R.K_AMB)
        lines.append(f"    T={T_game:6d} a=0.5 his=3: L_q={L_q / 65536:12.2f} game  "
                     f"g={g:10.3f}  f_q={f_q:6d} ({f_q / ONE:.5f})  "
                     f"float={R.fleck_f_float(float(T_game), L_q / 65536.0):.5f}")
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
    excess-form Fleck factor and the corrected clamp).

    Breaks if: the Fleck factor stops damping (explicit overshoots and the low
    rail fires), or alpha's branch changes (the Fleck-only equilibrium moves).
    """
    lines, ok = [], True
    A, HIS = A_FURNITURE, HIS_FURNITURE
    phi_amb = R.E0
    n = int(0.5 * R.TICK_HZ)
    # --- cooling, the OLD law (zero sky, whole emission) as an instrument check
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
    # --- cooling, the NEW law (ambient bath, excess form)
    e_new, ce = R.cell_march(T_SRC_GAME << 16, phi_amb, A, HIS, n, fleck=False,
                             clamp_enabled=False)
    f_new, cf = R.cell_march(T_SRC_GAME << 16, phi_amb, A, HIS, n, fleck=True,
                             clamp_enabled=False)
    x_new = R.cool_exact_bath(float(T_SRC_GAME), 0.5, 0.5, HIS)
    err_e = (e_new / 65536 - x_new) / x_new * 100
    err_f = (f_new / 65536 - x_new) / x_new * 100
    ok &= abs(err_f) < abs(err_e)
    lines.append(f"  cooling 1263 game for 0.5 s, NEW law (ambient bath, excess form): "
                 f"explicit {e_new / 65536:7.1f} ({err_e:+5.2f}%)  Fleck "
                 f"{f_new / 65536:7.1f} ({err_f:+5.2f}%)  analytic {x_new:7.1f}   "
                 f"{'OK (Fleck is closer)' if abs(err_f) < abs(err_e) else 'FAIL'}")
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
    # --- the damped source: what a cell actually radiates after the Fleck factor.
    # It is NOT globally monotone -- the integer floor-divisions wobble above the
    # fire range -- so the gate asserts what is true: positive, never above the
    # black body, strictly rising through the fire range, and the wobble bounded.
    top = R.E[R.E_TABLE_SIZE - 1]
    worst_shipped = 0.0
    for a_q, his, nm in SHIPPED_ROWS:
        def src_of(Tg, a_q=a_q, his=his):
            ex = R.E[R.e_bucket_of(Tg << 16)] - R.E0
            return R.E0 + ((ex * R.fleck_f_solid_q(Tg << 16, a_q, his)[0]) >> 16)
        vals = [(4 * b, src_of(4 * b)) for b in range(R.E_TABLE_SIZE)]
        sane = all(0 < v <= R.E[R.e_bucket_of(t << 16)] for t, v in vals)
        drops = [(vals[i][0], (vals[i][1] - vals[i + 1][1]) / vals[i][1])
                 for i in range(len(vals) - 1) if vals[i + 1][1] < vals[i][1]]
        first = drops[0][0] if drops else None
        worst = max((d[1] for d in drops), default=0.0)
        worst_shipped = max(worst_shipped, worst)
        fire_mono = (first is None) or (first > 2500)
        good = sane and fire_mono and worst < 0.05
        ok &= good
        f_top = R.fleck_f_solid_q(R.T_TABLE_TOP_GAME << 16, a_q, his)[0]
        lines.append(f"  damped source, {nm:22s} (a={a_q / ONE:.2f} his={his}): positive "
                     f"and <= E°[T] everywhere: {sane}; strictly rising to "
                     f"{'the table top' if first is None else str(first) + ' game'}; "
                     f"above that {len(drops)} backward steps of <= {worst * 100:5.2f}% "
                     f"in power (= {(1 + worst) ** 0.25 * 100 - 100:4.2f}% in a receiver's "
                     f"equilibrium T), f_q at the top = {f_top}  "
                     f"{'OK' if good else 'FAIL'}")
    # NON-VACUITY + the mechanism: the wobble IS f_q's resolution, so it grows as
    # the thermal mass falls. thermal_mass = 1 ships on no material today.
    corner_vals = [R.E0 + (((R.E[R.e_bucket_of((4 * b) << 16)] - R.E0)
                            * R.fleck_f_solid_q((4 * b) << 16, ONE, 0)[0]) >> 16)
                   for b in range(R.E_TABLE_SIZE)]
    corner_worst = max(((corner_vals[i] - corner_vals[i + 1]) / corner_vals[i]
                        for i in range(len(corner_vals) - 1)
                        if corner_vals[i + 1] < corner_vals[i]), default=0.0)
    ok &= (corner_worst > worst_shipped)
    lines.append(f"  the wobble is f_q's own Q16 resolution: at a=1 thermal_mass=1 "
                 f"(his=0, which NO material ships) f_q at the table top is only "
                 f"{R.fleck_f_solid_q(R.T_TABLE_TOP_GAME << 16, ONE, 0)[0]} counts and "
                 f"the wobble reaches {corner_worst * 100:.2f}% -- worse than every "
                 f"shipped row ({worst_shipped * 100:.2f}%), which is the mechanism, "
                 f"not noise. P1 should carry the ingress rule heat_atten > 0 => "
                 f"thermal_mass >= 8")
    lines.append(f"  at the table top the damped source is "
                 f"{R.E0 + (((top - R.E0) * R.fleck_f_solid_q(R.T_TABLE_TOP_GAME << 16, A, HIS)[0]) >> 16):,} "
                 f"vs the undamped {top:,} counts (x851 less) -- the RATE cost of the "
                 f"stability fix, paid by a cell that keeps the energy and cools "
                 f"slower, not by energy going missing")
    # --- the equilibrium table (the 0-D model stability_study.py section 7 used)
    iters = (40 if fast else 400) * 24
    lines.append(f"  equilibrium under a held fluence Phi = G x E°[T_src], G = 0.25 "
                 f"({iters} ticks):")
    lines.append(f"    {'T_src':>8}{'true (E_inv)':>14}{'true (float)':>14}"
                 f"{'Fleck only':>16}{'Fleck+clamp':>13}{'clamp hits':>12}")
    for T_src in (1263, 5000, 16000):
        phi = int(0.25 * R.E[R.e_bucket_of(T_src << 16)])
        t_true = R.e_inv_q(phi) >> 16
        t_true_f = R.e_inv_float(phi)
        t_f, _ = R.cell_march(0, phi, A, HIS, iters, clamp_enabled=False,
                              rails_enabled=False, int32_sat=False)
        t_c, cc = R.cell_march(0, phi, A, HIS, iters, clamp_enabled=True)
        good = (t_c >> 16) == t_true
        ok &= good
        lines.append(f"    {T_src:>8}{t_true:>14}{t_true_f:>14.1f}"
                     f"{t_f / 65536:>16,.1f}{t_c / 65536:>13.1f}{cc.rad_clamp_hits:>12}"
                     f"  {'OK' if good else 'FAIL'}")
    return ok, lines


def gate11_headroom(fast=False):
    """G11. Max stream and max |rad_net| on a scene seeded at the table top stay
    below 2^46 (the design's per-cell bound), and no intermediate product
    approaches 2^63.

    Breaks if: the ordinate weight, the table's top or the number of ordinates
    grows enough to need a wider accumulator than int64.
    """
    lines, ok = [], True
    h, w = (7, 8) if fast else (9, 11)
    a = R.plane(h, w, ONE)
    d = R.plane(h, w, ONE)
    k = R.plane(h, w, K_LEAK)
    T = R.plane(h, w, R.T_TABLE_TOP_GAME << 16)
    for transport in ("shear", "step"):
        res = R.sweep_q(a, d, k, T, transport=transport)
        lg = lambda v: math.log2(max(abs(v), 1))  # noqa: E731
        good = (res.max_abs_net < 2 ** 46 and res.max_fluence < 2 ** 46
                and res.max_product < 2 ** 62)
        ok &= good
        lines.append(f"  {transport:5s} at the table top ({R.T_TABLE_TOP_GAME} game, "
                     f"a = 1): max stream = 2^{lg(res.max_stream):.1f}, max|rad_net| = "
                     f"2^{lg(res.max_abs_net):.1f}, max fluence = "
                     f"2^{lg(res.max_fluence):.1f}, max product = "
                     f"2^{lg(res.max_product):.1f}  (bounds: sums < 2^46, products "
                     f"< 2^63)  {'OK' if good else 'FAIL'}")
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
