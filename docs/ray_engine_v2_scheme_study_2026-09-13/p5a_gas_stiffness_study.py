"""P5a -- THE GAS STIFFNESS, RE-DERIVED FROM LIVE PREMISES (2026-09-24).

Design v3 section 6.3 says gas stiffness is "bounded and modest, and density
cancels": pure soot reaches only g = 4.2 against a solid's 848, so Fleck and
the clamp "are not holding back a runaway". That paragraph cites
`smoke_heat_capacity_study.py` section 4, and that study rests on two premises
the tree has since retired:

  * the FITTED emission scale RAD_SCALE = 5.1427e-5 -- the live sweep bakes at
    `[physics.radiation] rad_scale_derived = 1.6533e-08`, 3110x weaker;
  * a gas cell's heat capacity = its particle count, i.e. c_v = 1 -- the live
    `[physics.thermal] c_v = 0.0076849` makes an air cell 130x lighter.

The two corrections pull in opposite directions; this instrument measures the
net instead of arguing it, with the INTEGER REFERENCE's own arithmetic: the E°
table (`E_LIVE`), design 2.8's gas L_q through the staged deposit chain
(`fleck_L_gas_q` -- recip_N, recip_cv, the n_floor_heat floor), and the engine's
Fleck factor f = T_abs / max(T_abs, 4L). g = 4 L / T_abs is the ratio the
Fleck branch keys on: f == 2^24 (undamped) exactly when g <= 1.

THE ONE VARIABLE. With the density law a_gas = min(1, sum_g h_g N_g) and the
capacity N_bulk * c_v, a gas cell's L is its absorptivity PER UNIT CAPACITY,
r = a_gas / max(N_bulk, n_floor), times the temperature's own factor:

    g(T, r) = r * G(T),     G(T) = 4 (E°[T] - E°[0]) / (c_v * T_abs)   [Q16 units]

G is "pure soot at ambient density": a_gas = 1 in a cell holding one ambient
air cell's worth of bulk gas -- the old study's own definition of a soot
fraction of 1.0. Every row below is computed through the integer chain, not
this formula; the formula is what they reduce to.

Run:
    C:/Users/steen/anaconda3/python.exe docs/ray_engine_v2_scheme_study_2026-09-13/p5a_gas_stiffness_study.py

Nothing here is engine code. The 0-D gas-cell march in section 5 is a MODEL of
design 6.3's Pass-1 gas branch (P5b's, not built), used only to measure where
the clamp would bind.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sweep_ref_q as R  # noqa: E402

ONE = R.ONE
F_ONE = R.F_ONE
LIVE = R.E_LIVE                               # rad_scale_derived, the sweep's table
STALE = R.E                                   # the retired fitted scale
C_V_Q_LIVE = R.quant(R.C_V_LIVE)              # 504: the ONE integer form of c_v
C_V_Q_STALE = ONE                             # c_v = 1, the old study's premise
N_FLOOR_Q = R.quant(R.N_FLOOR_HEAT_LIVE)      # 655: n_floor_heat = 0.01


def t_abs_q(T_game: int) -> int:
    return (T_game + R.K_AMB) << 16


def gas_g(T_game: int, a_q: int, n_q: int, *, table=LIVE, c_v_q=C_V_Q_LIVE,
          e_ref=None):
    """(g, f_q24, L_q) for one gas cell, through the reference's integer chain."""
    T_q = T_game << 16
    L = R.fleck_L_gas_q(T_q, a_q, n_q, c_v_q=c_v_q, n_floor_q=N_FLOOR_Q,
                        table=table, e_ref=e_ref)
    f = R.fleck_f_q(T_q, L)
    return 4.0 * L / t_abs_q(T_game), f, L


def solid_g(T_game: int, a_q: int, his: int, table=LIVE):
    L = R.fleck_L_solid_q(T_game << 16, a_q, his, table)
    return 4.0 * L / t_abs_q(T_game)


def isobaric_n_q(T_game: int) -> int:
    """The bulk N a cell at T holds at AMBIENT PRESSURE (p = C N T_abs, eos_solver):
    N = N_amb * T_amb / T_abs. A hot cell is a thin cell."""
    return (ONE * R.K_AMB) // (T_game + R.K_AMB)


def first_damped_T(a_q_of_T, n_q_of_T, **kw):
    """The lowest bucket low edge at which the Fleck factor leaves 2^24 (g > 1),
    or None if it never does below the table top."""
    for b in range(R.E_TABLE_SIZE):
        T = 4 * b
        g, f, _L = gas_g(T, a_q_of_T(T), n_q_of_T(T), **kw)
        if f < F_ONE:
            return T, g
    return None


A_PURE = ONE                     # pure soot: opaque in one tile (the old study's
                                 # "soot fraction 1.0": a = 1 at one ambient cell's
                                 # capacity)
A_TYPICAL = R.quant(0.2)         # a typical smoke mixture: 20 % per 0.333 m tile.
                                 # A dense smoke layer carries soot at a volume
                                 # fraction f_v ~ 1e-6; soot's Planck-mean
                                 # absorption is kappa_P = C f_v T with C = 1264
                                 # (Dalzell-Sarofim optical constants) to 1862
                                 # (Hubbard & Tien) per m.K, so at 400-600 K
                                 # kappa_P ~ 0.5-1.1 /m and a tile absorbs
                                 # 1 - exp(-kappa * 0.333) = 0.15-0.31. (It is also
                                 # the old study's thinner soot row.) How the
                                 # engine's smoke DENSITY maps onto f_v is the open
                                 # P-S2 question, so this is a value of a_gas, not
                                 # of heat_absorb.
T_ROWS = (0, 50, 100, 150, 200, 290, 500, 804, 1000, 1263, 1674, 2000, 3000,
          5000, 8000, 12000, 15996)


def section1_the_two_corrections():
    print("=" * 96)
    print("1. THE OLD NUMBER, REPRODUCED -- and the two corrections, one at a time")
    print("=" * 96)
    # the old study's scene: a smoke cell one tile from a 1263-game fire, G = 0.25,
    # zero sky, whole emission; its equilibrium temperature is E_inv(Phi)
    phi = int(0.25 * STALE[R.e_bucket_of(1263 << 16)])
    T_eq = R.e_inv_q(phi, STALE) >> 16
    g_old, _f, _L = gas_g(T_eq, A_PURE, ONE, table=STALE, c_v_q=C_V_Q_STALE, e_ref=0)
    g_scale, _f, _L = gas_g(T_eq, A_PURE, ONE, table=LIVE, c_v_q=C_V_Q_STALE, e_ref=0)
    g_cv, _f, _L = gas_g(T_eq, A_PURE, ONE, table=STALE, c_v_q=C_V_Q_LIVE, e_ref=0)
    g_live, _f, _L = gas_g(T_eq, A_PURE, ONE)
    print(f"   pure soot (a = 1, N_bulk = 1) at the old study's equilibrium point, "
          f"T = {T_eq} game:")
    print(f"     stale premises (fitted scale, c_v = 1, whole emission) g = {g_old:9.4f}"
          f"   <- the design's '4.2'")
    print(f"     live scale only                                       g = {g_scale:9.4f}"
          f"   (x{g_scale / g_old:.2e})")
    print(f"     live c_v only                                         g = {g_cv:9.4f}"
          f"   (x{g_cv / g_old:.1f})")
    print(f"     LIVE premises (derived scale, live c_v, excess form)  g = {g_live:9.4f}"
          f"   (x1/{g_old / g_live:.1f})")
    print("   -> at the SAME temperature the back-of-envelope holds: the scale takes "
          "3110x, c_v gives 130x back,")
    print(f"      and gas is {g_old / g_live:.1f}x LESS stiff than section 6.3 said.")
    print()


def section2_the_table():
    print("=" * 96)
    print("2. g AGAINST TEMPERATURE, LIVE PREMISES (rad_scale_derived, c_v_q = "
          f"{C_V_Q_LIVE}, n_floor_q = {N_FLOOR_Q})")
    print("=" * 96)
    rows = R.shipped_absorbing_rows()
    print("   columns: pure soot (a = 1) and a typical smoke mixture (a = 0.2), each at "
          "ambient bulk density")
    print("   (N_bulk = 1) and ISOBARIC (N_bulk = 293/T_abs, what a cell at T holds at "
          "ambient pressure);")
    print("   an opaque cell at the n_floor_heat density (N_bulk <= 0.01); and the "
          "stiffest SHIPPED solid row.")
    print(f"   f = 2^24 exactly while g <= 1; '*' marks a damped (f < 2^24) cell.\n")
    hdr = (f"   {'T game':>7}{'soot N=1':>12}{'mix N=1':>12}{'soot isob.':>12}"
           f"{'mix isob.':>12}{'soot floor':>12}{'solid max':>11}  {'(row)':<22}")
    print(hdr)
    for T in T_ROWS:
        n_iso = isobaric_n_q(T)
        cells = [gas_g(T, A_PURE, ONE), gas_g(T, A_TYPICAL, ONE),
                 gas_g(T, A_PURE, n_iso), gas_g(T, A_TYPICAL, n_iso),
                 gas_g(T, A_PURE, N_FLOOR_Q)]
        best = max(rows, key=lambda r: solid_g(T, r[1], r[2]))
        gs = solid_g(T, best[1], best[2])
        txt = "".join(f"{g:>11.4g}{'*' if f < F_ONE else ' '}" for g, f, _L in cells)
        print(f"   {T:>7}{txt}{gs:>11.4g}  ({best[0]})")
    print()
    return rows


def section3_where_fleck_engages(rows):
    print("=" * 96)
    print("3. WHERE FLECK ENGAGES ON GAS -- the lowest temperature with f < 2^24 (g > 1)")
    print("=" * 96)
    cases = (
        ("pure soot, N_bulk = 1", lambda T: A_PURE, lambda T: ONE),
        ("typical mix (a = 0.2), N_bulk = 1", lambda T: A_TYPICAL, lambda T: ONE),
        ("pure soot, isobaric", lambda T: A_PURE, isobaric_n_q),
        ("typical mix, isobaric", lambda T: A_TYPICAL, isobaric_n_q),
        ("pure soot at the n_floor density", lambda T: A_PURE, lambda T: N_FLOOR_Q),
        ("typical mix at the n_floor density", lambda T: A_TYPICAL, lambda T: N_FLOOR_Q),
    )
    for name, a_of, n_of in cases:
        hit = first_damped_T(a_of, n_of)
        where = (f"from {hit[0]:>5} game (g = {hit[1]:.3f} there)" if hit
                 else "never below the table top")
        print(f"   {name:<38} {where}")
    print("   for comparison, the SHIPPED solid rows on the same live table (design row 42 "
          "said 'never'):")
    seen = set()
    for name, a_q, his, _atten, _tm in rows:
        if (a_q, his) in seen:
            continue
        seen.add((a_q, his))
        onset = next((4 * b for b in range(R.E_TABLE_SIZE)
                      if R.fleck_f_solid_q((4 * b) << 16, a_q, his, LIVE)[0] < F_ONE), None)
        top = solid_g(R.T_TABLE_TOP_GAME, a_q, his)
        where = f"damped from {onset:>5} game" if onset is not None else "never damped"
        print(f"     {name:<12} a = {a_q / ONE:.3f} his = {his:>3}: {where}; "
              f"g at the table top = {top:.4g}")
    print()


def section4_density_does_and_does_not_cancel():
    print("=" * 96)
    print("4. DENSITY CANCELS ONLY WHILE THE CELL IS OPTICALLY THIN")
    print("=" * 96)
    T = 1263
    print(f"   T = {T} game; a soot fraction phi = N_soot / N_bulk held at 0.2, heat_absorb h "
          f"per unit soot density:")
    print(f"   {'N_bulk':>8}{'h = 1: a_gas':>14}{'g':>10}{'h = 20: a_gas':>15}{'g':>10}")
    for n_bulk in (1.0, 0.5, 0.2, 0.05, 0.01, 0.002):
        n_q = R.quant(n_bulk)
        soot_q = R.quant(0.2 * n_bulk)
        out = []
        for h in (1.0, 20.0):
            a_gas = R.gas_extinction_q([soot_q], [R.quant(h)], n_q)
            g, _f, _L = gas_g(T, a_gas, n_q)
            out.append((a_gas / ONE, g))
        print(f"   {n_bulk:>8.3f}{out[0][0]:>14.4f}{out[0][1]:>10.4f}"
              f"{out[1][0]:>15.4f}{out[1][1]:>10.4f}")
    print("   -> while THIN (h phi N_bulk < 1) g = G(T) h phi, FLAT in N_bulk -- the old "
          "study's 'density cancels' -- until")
    print("      N_bulk drops below n_floor_heat (0.01), where the capacity stops "
          "shrinking and g falls. While OPAQUE")
    print("      (h = 20, N_bulk >= 1/(h phi) = 0.25 here) g = G(T)/N_bulk, RISING as the "
          "cell thins, up to the same ceiling.")
    print("      So g <= G(T) * sum_g h_g phi_g always: density cancels in the BOUND, not "
          "in the value, and the bound")
    print("      scales with heat_absorb -- a soot 20x the old study's is 20x stiffer at "
          "the same soot fraction.")
    print("      And phi itself grows in a HOT cell: at ambient pressure N_bulk ~ 1/T_abs "
          "(the isobaric columns above),")
    print("      while a trace plane is a semi-Lagrangian sample of the VALUE with 'NO flux "
          "form' (smoke_dynamics.cpp):")
    print("      nothing dilutes the soot as the heated bulk expands out of the cell.")
    print()


T_MIN_Q = -292 << 16                          # config.toml:787, the gas recovery's floor


def march_gas_cell(T0_game, phi, a_q, n_q, ticks, *, fleck=True, clamp=True):
    """A 0-D MODEL of design 6.3's Pass-1 gas branch under a HELD fluence -- the
    branch P5b builds, NOT built here. rad_net in the excess form; dT through the
    staged chain, magnitude then sign; the clamp min(T_after, max(T_before,
    E_inv(Phi))); the T_MAX_PHYS rail and the recovery's T_MIN floor (which the
    engine applies once per tick in eos_solver step 7). Returns (trace, clamp
    hits, clamp drop in Q16 temperature, T_MIN rail hits)."""
    T = T0_game << 16
    recip_n, recip_cv = R.gas_capacity_recips(n_q, C_V_Q_LIVE, N_FLOOR_Q)
    t_cap = R.e_inv_q(phi, LIVE)
    hits = drop = low = 0
    tr = [T]
    for _ in range(ticks):
        ex = LIVE[R.e_bucket_of(T)] - LIVE[0]
        f = F_ONE
        if fleck:
            L = R.fleck_L_gas_q(T, a_q, n_q, c_v_q=C_V_Q_LIVE, n_floor_q=N_FLOOR_Q,
                                table=LIVE)
            f = R.fleck_f_q(T, L)
        src = LIVE[0] + ((ex * f) >> R.F_SHIFT)
        rn = ((phi * a_q) >> 16) - ((src * a_q) >> 16)
        mag = R.deposit_dT_wide_i64(abs(rn), recip_n, recip_cv)
        dT = mag if rn >= 0 else -mag
        t_before = T
        t_after = R.sat_add_q16(T, dT)
        t_new = t_after
        if clamp:
            ceiling = t_cap if t_cap > t_before else t_before
            if t_new > ceiling:
                hits += 1
                drop += t_new - ceiling
                t_new = ceiling
        if t_new > R.T_MAX_PHYS_Q:
            t_new = R.T_MAX_PHYS_Q
        if t_new < T_MIN_Q:
            t_new = T_MIN_Q
            low += 1
        T = t_new
        tr.append(T)
    return tr, hits, drop, low


def _osc(tr):
    """Sign changes of the step, over the last half of the trace (an overshoot
    signature of an explicit update running above g = 1)."""
    half = tr[len(tr) // 2:]
    steps = [half[i + 1] - half[i] for i in range(len(half) - 1)]
    return sum(1 for i in range(len(steps) - 1) if steps[i] * steps[i + 1] < 0)


GAS_CELLS = (("soot N=1", A_PURE, ONE),
             ("mix N=1", A_TYPICAL, ONE),
             ("soot N=0.2", A_PURE, R.quant(0.2)))


def section5_where_the_clamp_binds():
    print("=" * 96)
    print("5. WHERE THE CLAMP BINDS -- a 0-D model of design 6.3's gas branch "
          "(P5b's, NOT built), held fluence")
    print("=" * 96)
    print("   'Fleck on' = the gas arm WIRED (fleck_L_gas_q); 'L = 0' = the arm as P5a "
          "ships it (f == 2^24).")
    print("   hits = ticks the clamp bound (whole run / its last 60); drop = the "
          "temperature it removed, total;")
    print("   low = T_MIN rail hits; osc = sign changes of the step over the last half.")
    # (a) HEATING from ambient toward a held field
    print("\n   (a) HEATING: a cell entering a source's field from ambient. Phi = E°[0] + "
          "G (E°[T_src] - E°[0]),")
    print("       G = 0.25 (one tile from the source); 240 ticks = 10 s. "
          "T_cap = E_inv(Phi).")
    print(f"   {'source':>7}{'cell':>12}{'T_cap':>7}{'g(T_cap)':>10}"
          f"{'Fleck on: T / hits / last60 / drop':>42}"
          f"{'L = 0: T / hits / drop / osc':>40}")
    eq_hits, stiff = [], []
    for T_src in (290, 1263, 5000, 15996):
        phi = LIVE[0] + (LIVE[R.e_bucket_of(T_src << 16)] - LIVE[0]) // 4
        t_cap = R.e_inv_q(phi, LIVE) >> 16
        for name, a_q, n_q in GAS_CELLS:
            g_cap, _f, _L = gas_g(t_cap, a_q, n_q)
            tr, hits, drop, _lo = march_gas_cell(0, phi, a_q, n_q, 240, fleck=True)
            _t, h180, _d, _l = march_gas_cell(0, phi, a_q, n_q, 180, fleck=True)
            last60 = hits - h180
            tr0, hits0, drop0, _lo0 = march_gas_cell(0, phi, a_q, n_q, 240, fleck=False)
            print(f"   {T_src:>7}{name:>12}{t_cap:>7}{g_cap:>10.3g}"
                  f"{tr[-1] / 65536:>11.1f} /{hits:>4} /{last60:>4} /{drop / 65536:>12.1f}"
                  f"{tr0[-1] / 65536:>13.1f} /{hits0:>4} /{drop0 / 65536:>10.1f} /{_osc(tr0):>3}")
            if (t_cap << 16) in tr and g_cap <= 1.0:
                # from the tick it ARRIVES at T_cap: how many of the remaining ticks bind?
                arrive = tr.index(t_cap << 16)
                _t, h_arr, _d, _l = march_gas_cell(0, phi, a_q, n_q, arrive, fleck=True)
                surplus = phi - LIVE[R.e_bucket_of(t_cap << 16)]
                eq_hits.append((surplus, hits - h_arr, 240 - arrive))
            if g_cap > 1.0:
                stiff.append((drop, drop0, _osc(tr0), tr0[-1] == t_cap << 16))
    every = all(b == n for s, b, n in eq_hits if s > 0)
    never = all(b <= 1 for s, b, n in eq_hits if s == 0)
    print(f"   MEASURED (below g = 1, from the tick a cell ARRIVES at T_cap): with a non-zero "
          f"surplus Phi - E°[T_cap] the clamp")
    print(f"   binds on EVERY remaining tick: {every} "
          f"({', '.join(f'{b}/{n}' for s, b, n in eq_hits if s > 0)}); where Phi lands "
          f"exactly on E°[T_cap] (surplus 0) it")
    print(f"   does not: {never}. E_inv returns the bucket's LOW edge (idempotence) while the "
          f"explicit fixed point lies INSIDE the")
    print("   bucket, so an equilibrium approached from below is held at the low edge by "
          "clipping the surplus every tick")
    print("   (Fleck on or off alike: f == 1 there).")
    ratio = min(d / d0 for d, d0, _o, _h in stiff)
    held = all(h and o == 0 for _d, _d0, o, h in stiff)
    print(f"   MEASURED: where g(T_cap) > 1, WIRING the gas arm makes the clamp remove at "
          f"least {ratio:.0f}x MORE -- the damped")
    print("   emission deficit, dropped every tick; with L = 0 the clamp alone holds T_cap "
          f"without oscillation in every such case: {held}.")
    # (b) COOLING from above: a combustion-heated gas cell radiating into a room
    print("\n   (b) COOLING: a gas cell heated by combustion, radiating into an ambient "
          "room (Phi = E°[0]); 48 ticks.")
    print("       The clamp never clips a cooling step; this is Fleck's own job.")
    print(f"   {'T0':>7}{'cell':>12}{'g(T0)':>10}{'Fleck on: T1 / min T / low / osc':>40}"
          f"{'L = 0: T1 / min T / low / osc':>40}")
    phi_amb = LIVE[0]
    fleck_clean, crash = True, []
    for T0 in (290, 1263, 2000, 3000, 5000, 15996):
        for name, a_q, n_q in GAS_CELLS:
            g0, _f, _L = gas_g(T0, a_q, n_q)
            tr, _h, _d, lo = march_gas_cell(T0, phi_amb, a_q, n_q, 48, fleck=True)
            tr0, _h0, _d0, lo0 = march_gas_cell(T0, phi_amb, a_q, n_q, 48, fleck=False)
            print(f"   {T0:>7}{name:>12}{g0:>10.3g}"
                  f"{tr[1] / 65536:>12.1f} /{min(tr) / 65536:>9.1f} /{lo:>4} /{_osc(tr):>3}"
                  f"{tr0[1] / 65536:>16.1f} /{min(tr0) / 65536:>9.1f} /{lo0:>4} /{_osc(tr0):>3}")
            fleck_clean &= (lo == 0 and _osc(tr) == 0 and min(tr) >= 0)
            if min(tr0) < 0:
                crash.append((g0, 4.0 * T0 / (T0 + R.K_AMB)))
    print(f"   MEASURED: Fleck on -- monotone, never below ambient, no rail, in every case: "
          f"{fleck_clean}. L = 0 -- the first")
    print(f"   step overshoots BELOW AMBIENT in {len(crash)} cases, each with g > 4T/T_abs "
          f"(the step removes more than the cell's")
    print(f"   whole excess): {all(g > lim for g, lim in crash)}; the smallest such g was "
          f"{min(g for g, _l in crash):.2f}.")
    # (c) the equilibrium effect on SHIPPED solid rows (the clamp is live for them
    # since P3): run to equilibrium, then count the clamp on the final 240 ticks.
    rows = R.shipped_absorbing_rows()
    phi = LIVE[0] + (LIVE[R.e_bucket_of(1263 << 16)] - LIVE[0]) // 4
    t_cap = R.e_inv_q(phi, LIVE)
    print(f"\n   (c) the SHIPPED solid rows under the same 1263-game field (G = 0.25, "
          f"T_cap = {t_cap >> 16} game), the")
    print("       reference's own solid fold (live since P3), after 24 000 ticks (1000 s): "
          "clamp hits in the LAST 240.")
    for name, a_q, his, _atten, _tm in rows:
        tr, c = R.cell_march(0, phi, a_q, his, 24000, table=LIVE, trace=True)
        c2 = R.FoldCounters()
        T = [[tr[-1]]]
        for _ in range(240):
            rn = R.cell_rad_net_q(T[0][0], phi, a_q, his, table=LIVE)
            R.fold_pass1_solid(T, [[rn]], [[phi]], his, [[1]], c2, table=LIVE)
        print(f"     {name:<12} a = {a_q / ONE:.3f} his = {his:>3}: T = "
              f"{T[0][0] / 65536:9.3f} game; clamp hits in the last 240 ticks = "
              f"{c2.rad_clamp_hits:>3}, removed {c2.clamp_drop_sum / 65536:.4f} game")
    print("   -> the bucket-edge pinning is not a gas effect: every solid row that has "
          "reached T_cap is clipped every tick too.")
    print()


def section6_physical_cross_check():
    print("=" * 96)
    print("6. PHYSICS OR ARTIFACT? -- a real sooty flame gas at the 24 Hz tick (float, "
          "outside the engine)")
    print("=" * 96)
    # The textbook optically-thin loss dT/dt = -4 kappa_P sigma T^4 / (rho c_v), with
    # soot's Planck-mean kappa_P = C f_v T and the ENGINE's own rho c_v (c_v's
    # derivation in config.toml: 864.548 J/(m3 K) at 293 K, x 293/T at ambient
    # pressure); g = 4 L / T_abs exactly as the gates use it.
    sigma, rho_cv_amb, dt, tile = 5.670e-8, 864.548, 1.0 / 24.0, 0.333
    print(f"   {'T K':>6}{'f_v':>8}{'C':>7}{'kappa /m':>10}{'a per tile':>12}"
          f"{'tau_rad s':>11}{'L K/tick':>10}{'g':>8}")
    for T in (1000.0, 1500.0):
        for f_v in (1e-7, 1e-6):
            for C in (1264.0, 1862.0):
                kappa = C * f_v * T
                rho_cv = rho_cv_amb * 293.0 / T
                rate = 4.0 * kappa * sigma * T ** 4 / rho_cv
                L = rate * dt
                a = 1.0 - math.exp(-kappa * tile)
                print(f"   {T:>6.0f}{f_v:>8.0e}{C:>7.0f}{kappa:>10.3f}{a:>12.3f}"
                      f"{T / rate:>11.3f}{L:>10.1f}{4.0 * L / T:>8.2f}")
    print("   -> a sooty flame (f_v ~ 1e-6) sheds its heat radiatively in ~0.1 s, a few "
          "ticks: g ~ 1-3 at 1500 K is the")
    print("      physics of a LIGHT gas, not a calibration artifact. On gas, Fleck is "
          "load-bearing.")
    print()


def main():
    print(__doc__.split("Run:")[0])
    section1_the_two_corrections()
    rows = section2_the_table()
    section3_where_fleck_engages(rows)
    section4_density_does_and_does_not_cancel()
    section5_where_the_clamp_binds()
    section6_physical_cross_check()
    return 0


if __name__ == "__main__":
    sys.exit(main())
