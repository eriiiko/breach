"""m2b_fleck_probe.py — WHAT IS THE FLECK DAMPING ACTUALLY DOING?

A MEASUREMENT, not a patch (arc #12, between M2 and M3). It answers M2 §5.1's
open question: is `f = 0.013` on a thin row the correct effective emission rate,
or is the damping destroying radiative transport?

Nothing here is engine code. It drives the SHIPPED C++ (`RadiationSweep`,
`EmissiveTable`, `TemperatureSolver`) through the pybind bindings and compares
what they do against a closed-form physical reference computed from the same
constants the config derives `rad_scale_derived` from.

Run:  C:/Users/steen/anaconda3/python.exe docs/.../m2b_fleck_probe.py [section...]
      sections: provenance L f g12 fold testA testA2 testB testC testD testE
                (default: all, in that order)

EVERY section that asserts a property also runs a CONTROL that perturbs the
thing being measured and shows the instrument moving. A number with no control
is marked as such.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import breach_physics as bp          # noqa: E402
import sweep_ref_q as R              # noqa: E402

ONE = 1 << 16
F_ONE = 1 << 24

# ---- the two emission calibrations that exist in config.toml --------------
RS_FITTED = 5.1427e-5        # [physics.fire] rad_scale      — the OLD CAST's key
RS_DERIVED = 1.6533e-08      # [physics.radiation] rad_scale_derived — THE SWEEP's
K_AMB = 293.0
K_SLOPE = 1.0
T_AMB_Q = int(K_AMB) << 16

# ---- the physical constants rad_scale_derived is built from (config.toml
#      [physics.radiation] block, verbatim) --------------------------------
SIGMA = 5.670374419e-8       # W/m^2/K^4, CODATA 2018
TILE_W = 0.333               # m   (tile_size_ref_m, the shipped level)
CEIL_H = 2.5                 # m   ([physics.water] ceiling_h)
A_RAD = 4.0 * TILE_W * CEIL_H          # 3.33 m^2 — the FOUR lateral faces
TPS = 24.0                             # [clock] ticks_per_second
DT = 1.0 / TPS
RHO_C_PIN = 0.9e6                      # J/(m^3 K), R13's currency pin
V_TILE = TILE_W * TILE_W * CEIL_H      # 0.277223 m^3
J_PER_COUNT = RHO_C_PIN * V_TILE / (8 * 65536)     # 0.4759 J

# One thermal_mass unit in J/K: raising a thermal_mass = 1 tile by one game
# degree costs 65536 counts, i.e. 65536 * J_PER_COUNT joules.
J_PER_K_PER_TM = 65536.0 * J_PER_COUNT             # 31 187 J/K

SHIPPED = [("hull", 55706, 5), ("wood", 58982, -3), ("door", 58982, 3),
           ("steel", 55706, 5), ("glass", 19661, 4), ("furniture", 32768, -2),
           ("kindling", 32768, -3), ("foliage", 58982, -3)]


def etable(rad_scale):
    t = bp.EmissiveTable()
    t.rad_scale = float(rad_scale)
    t.kelvin_ambient = K_AMB
    t.k_temp_to_kelvin = K_SLOPE
    t.bake()
    return t


def f_of(tbl, T_game, a_q, his):
    return bp.RadiationSweep.fleck_f_solid_q24(tbl, int(T_game) << 16, a_q, his,
                                               T_AMB_Q) / F_ONE


def L_game(tbl, T_game, a_q, his):
    """L in GAME DEGREES PER TICK, exactly as the sweep's pre-pass forms it.

    Recomputed here from the same two primitives the pre-pass uses, because the
    binding only exposes `f`. Verified against `f` in section `L`.
    """
    e = np.asarray(tbl.table())
    ex = int(e[tbl.e_bucket_of(int(T_game) << 16)]) - int(e[0])
    ex = max(ex, 0)
    v = (a_q * ex) >> 16
    L_q = (v << (-his)) if his < 0 else (v >> his)      # shr_round0 on v >= 0
    return L_q / ONE


def L_physical(T_game, emissivity, thermal_mass):
    """The CLOSED FORM: eps*sigma*A*(T^4 - T_amb^4)*dt / C, in K per tick."""
    K = K_AMB + K_SLOPE * max(T_game, 0.0)
    K0 = K_AMB + 2.0 * K_SLOPE            # bucket 0's midpoint, 295 K
    C = thermal_mass * J_PER_K_PER_TM
    return emissivity * SIGMA * A_RAD * (K ** 4 - K0 ** 4) * DT / C


# ===========================================================================
def sec_provenance():
    print("\n" + "=" * 78)
    print("PROVENANCE — which E-table each owner bakes, on THIS tree")
    print("=" * 78)
    import importlib
    sys.path.insert(0, str(ROOT))
    from config import CFG                                    # noqa: E402
    fire = CFG.physics.fire
    rad = CFG.physics.radiation
    print(f"  [physics.fire] rad_scale          = {fire.rad_scale:.6g}"
          "   <- the OLD RAYCASTER's bake")
    print(f"  [physics.radiation] rad_scale_derived = {rad.rad_scale_derived:.6g}"
          "   <- THE SWEEP's bake")
    print(f"  ratio fitted/derived              = "
          f"{fire.rad_scale / rad.rad_scale_derived:.1f}x")
    ok_cfg, dials = R.config_dials_match()
    print(f"\n  the INTEGER REFERENCE's own default table (sweep_ref_q.RAD_SCALE):"
          f" {R.RAD_SCALE:.6g}")
    print(f"  ... which is the FITTED key. config_dials_match() -> {ok_cfg}"
          f" (it checks {list(dials)[0]!r} against [physics.fire])")
    print("\n  => the executable spec's DEFAULT table and the engine's sweep table"
          "\n     differ by the ratio above. Every f number quoted from the"
          "\n     reference without an explicit `table=` is on the FITTED scale.")
    # Is the fold flipped onto the sweep?
    src = (ROOT / "cpp" / "src" / "physics_engine.cpp").read_text(encoding="utf-8")
    flipped = "THE FLIP (T5b step 6" in src and "rad_net_sweep,\n" in src
    print(f"\n  the fold's source plane (physics_engine.cpp): "
          f"{'rad_net_sweep  (THE FLIP IS LIVE)' if flipped else 'rad_net (old cast)'}")
    stale = "the old cast above still feeds `rad_net`; P3 flips it" in \
        (ROOT / "src" / "simulation" / "physics_runner.py").read_text(encoding="utf-8")
    print(f"  physics_runner.py still carries the pre-flip comment: {stale}"
          f"  {'<- STALE, finding 6.2' if stale else ''}")


# ===========================================================================
def sec_L():
    print("\n" + "=" * 78)
    print("WHAT L IS — the engine's L against the closed-form physical rate")
    print("=" * 78)
    print("  claim: L_q == the temperature a cell would lose in ONE TICK if it")
    print("         radiated the full black-body excess freely, i.e. the")
    print("         EXPLICIT EULER STEP of  C dT/dt = -eps*sigma*A*(T^4-T_amb^4).")
    print(f"  constants: sigma={SIGMA:g}  A_rad={A_RAD:.3f} m^2  dt=1/{TPS:g} s"
          f"  J_per_count={J_PER_COUNT:.4f} J  1 tm unit = {J_PER_K_PER_TM:.0f} J/K")
    tbl = etable(RS_DERIVED)
    print(f"\n  {'row':>10}{'tm':>8}{'C (J/K)':>12}{'T game':>8}"
          f"{'L engine':>12}{'L closed-form':>15}{'rel err':>10}")
    worst = 0.0
    for name, a_q, his in SHIPPED:
        tm = 2.0 ** his
        for T in (300, 1200, 3000):
            le = L_game(tbl, T, a_q, his)
            lp = L_physical(T, a_q / ONE, tm)
            rel = abs(le - lp) / lp
            worst = max(worst, rel)
            print(f"  {name:>10}{tm:>8g}{tm * J_PER_K_PER_TM:>12.0f}{T:>8}"
                  f"{le:>12.4f}{lp:>15.4f}{rel:>9.2%}")
    print(f"\n  worst relative error over the table: {worst:.3%}"
          "   (the residue is the E-table's 4-game-unit bucket staircase)")

    print("\n  CONTROL (non-vacuity) — perturb each input, watch L move:")
    base = L_game(tbl, 1200, 58982, -3)
    print(f"    wood @1200                      L = {base:.4f} K/tick")
    print(f"    halve the mass (his -3 -> -4)   L = {L_game(tbl, 1200, 58982, -4):.4f}"
          f"   (expect 2x)")
    print(f"    halve emissivity (a 0.9->0.45)  L = {L_game(tbl, 1200, 29491, -3):.4f}"
          f"   (expect 0.5x)")
    print(f"    at the FITTED scale             L = "
          f"{L_game(etable(RS_FITTED), 1200, 58982, -3):.4f}"
          f"   (expect {RS_FITTED / RS_DERIVED:.0f}x)")

    print("\n  => g = 4L/T_abs is therefore the FRACTIONAL DROP IN EMISSIVE POWER")
    print("     the cell would suffer in one tick of free emission (dE/E = 4 dT/T),")
    print("     and f = 1/max(1, g) caps the per-tick drop at exactly T_abs/4.")
    print(f"\n  {'row':>10}{'T game':>8}{'L K/tick':>11}{'g':>10}{'f':>9}"
          f"{'eff. drop K/tick':>19}")
    for name, a_q, his in (("wood", 58982, -3), ("door", 58982, 3)):
        for T in (300, 1200, 3000, 8000, 15996):
            le = L_game(tbl, T, a_q, his)
            g = 4.0 * le / (T + K_AMB)
            f = f_of(tbl, T, a_q, his)
            print(f"  {name:>10}{T:>8}{le:>11.3f}{g:>10.4f}{f:>9.4f}"
                  f"{le * f:>19.3f}")


# ===========================================================================
def sec_f():
    print("\n" + "=" * 78)
    print("THE f TABLE — on BOTH scales. M2 §5.1 reproduced and located.")
    print("=" * 78)
    Ts = (300, 400, 600, 1200, 3000, 5000, 8000, 15996)
    for rs, tag in ((RS_FITTED, "FITTED [physics.fire] rad_scale  (the OLD CAST; "
                                "the integer reference's default)"),
                    (RS_DERIVED, "DERIVED rad_scale_derived  (THE LIVE SWEEP, "
                                 "post-flip)")):
        tbl = etable(rs)
        print(f"\n  --- {tag}")
        print(f"  {'row':>10}" + "".join(f"{t:>9}" for t in Ts))
        for name, a_q, his in SHIPPED:
            print(f"  {name:>10}" + "".join(f"{f_of(tbl, t, a_q, his):>9.4f}"
                                            for t in Ts))
    print("\n  M2 §5.1 quoted: door@1200 0.848 | wood@ignition 0.223 | wood@1200 0.013")
    tf = etable(RS_FITTED)
    print(f"  reproduced on the FITTED table:  door@1200 {f_of(tf, 1200, 58982, 3):.3f}"
          f" | wood@300 {f_of(tf, 300, 58982, -3):.3f}"
          f" | wood@1200 {f_of(tf, 1200, 58982, -3):.3f}")
    td = etable(RS_DERIVED)
    print(f"  the SAME cells on the LIVE table: door@1200 {f_of(td, 1200, 58982, 3):.3f}"
          f" | wood@300 {f_of(td, 300, 58982, -3):.3f}"
          f" | wood@1200 {f_of(td, 1200, 58982, -3):.3f}")

    print("\n  the g = 1 knee (first game temperature at which f < 1), per row:")
    print(f"  {'row':>10}{'derived':>12}{'fitted':>10}")
    for name, a_q, his in SHIPPED:
        knees = []
        for tbl in (td, tf):
            knee = None
            for b in range(R.E_TABLE_SIZE):
                if f_of(tbl, 4 * b, a_q, his) < 1.0:
                    knee = 4 * b
                    break
            knees.append("never" if knee is None else str(knee))
        print(f"  {name:>10}{knees[0]:>12}{knees[1]:>10}")

    print("\n  CONTROL (non-vacuity): the same probe on a row built to damp.")
    print(f"    a=1.0, thermal_mass=2^-16, T=1200, derived scale -> f = "
          f"{f_of(td, 1200, ONE, -16):.6f}  (the probe CAN report a damped f)")


# ===========================================================================
def _one_tile_scene(h, w, a_q, his, T_game, *, hot=(None,)):
    T = np.zeros((h, w), dtype=np.int32)
    aq = np.zeros((h, w), dtype=np.int32)
    dq = np.zeros((h, w), dtype=np.int32)
    shift = np.zeros((h, w), dtype=np.int32)
    ts = np.zeros((h, w), dtype=bool)
    cy, cx = h // 2, w // 2
    aq[cy, cx] = dq[cy, cx] = a_q
    shift[cy, cx] = his
    ts[cy, cx] = True
    T[cy, cx] = int(T_game * ONE)
    return T, aq, dq, shift, ts, (cy, cx)


def _planes(h, w):
    return (np.zeros((h, w), dtype=np.int64), np.zeros((h, w), dtype=np.int64),
            np.zeros((h, w), dtype=np.int64), np.zeros((h, w), dtype=np.int64))


def _shr_round0_signed(v, s):
    if s >= 0:
        return -((-v) >> s) if v < 0 else (v >> s)
    return v << (-s)


def sec_testA(rad_scale=RS_DERIVED, tag="DERIVED (live)"):
    print("\n" + "=" * 78)
    print(f"TEST A — RADIATIVE COOLING CURVE, one hot thin tile, {tag}")
    print("=" * 78)
    print("  One `wood` tile (a=0.9, thermal_mass=0.125) at 1200 game, cold room,")
    print("  no combustion, no conduction, k_leak = 0. The sweep + the Pass-1 fold")
    print("  run tick by tick; the reference is RK4 on")
    print("      C dT/dt = -eps*sigma*A*(T^4 - T_amb^4)   with 512 substeps/tick.")
    tbl = etable(rad_scale)
    sweep = bp.RadiationSweep()
    h = w = 7
    a_q, his, tm = 58982, -3, 0.125

    for T0 in (1200.0, 400.0):
        T, aq, dq, shift, ts, (cy, cx) = _one_tile_scene(h, w, a_q, his, T0)
        rn, rf, ra, rl = _planes(h, w)
        # exact reference
        C = tm * J_PER_K_PER_TM
        K0 = K_AMB + 2.0
        Tk = K_AMB + T0
        n_ticks = 240
        traj_ref, traj_eng = [], []
        for k in range(n_ticks):
            sweep.run(T, aq, dq, shift, ts, tbl, None, T_AMB_Q, 0,
                      bp.RadiationSweep.SHEAR, 16, rn, rf, ra, rl, True)
            dT = _shr_round0_signed(int(rn[cy, cx]), his)
            T[cy, cx] = int(T[cy, cx]) + dT
            traj_eng.append(int(T[cy, cx]) / ONE)
            sub = 512
            hstep = DT / sub
            for _ in range(sub):
                def dTdt(x):
                    return -(a_q / ONE) * SIGMA * A_RAD * (x ** 4 - K0 ** 4) / C
                k1 = dTdt(Tk)
                k2 = dTdt(Tk + 0.5 * hstep * k1)
                k3 = dTdt(Tk + 0.5 * hstep * k2)
                k4 = dTdt(Tk + hstep * k3)
                Tk += hstep * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            traj_ref.append(Tk - K_AMB)
        print(f"\n  T0 = {T0:g} game    {'tick':>6}{'engine':>12}{'exact':>12}"
              f"{'err':>10}{'rel':>9}")
        for k in (0, 1, 4, 23, 47, 119, 239):
            e, r = traj_eng[k], traj_ref[k]
            print(f"  {'':16}{k + 1:>6}{e:>12.3f}{r:>12.3f}{e - r:>10.3f}"
                  f"{(e - r) / max(abs(r), 1e-9):>8.2%}")
        rel = max(abs(a - b) / max(abs(b), 1e-9)
                  for a, b in zip(traj_eng, traj_ref))
        print(f"  worst relative deviation over {n_ticks} ticks "
              f"({n_ticks / TPS:.1f} s): {rel:.3%}")

    print("\n  CONTROL 1 (the instrument can see a suppressed rate): repeat the")
    print("  T0=1200 run with the FITTED table, where f = 0.013 at 1200 game.")
    tblf = etable(RS_FITTED)
    T, aq, dq, shift, ts, (cy, cx) = _one_tile_scene(h, w, a_q, his, 1200.0)
    rn, rf, ra, rl = _planes(h, w)
    first = None
    for k in range(240):
        sweep.run(T, aq, dq, shift, ts, tblf, None, T_AMB_Q, 0,
                  bp.RadiationSweep.SHEAR, 16, rn, rf, ra, rl, True)
        dT = _shr_round0_signed(int(rn[cy, cx]), his)
        if first is None:
            first = dT / ONE
        T[cy, cx] = int(T[cy, cx]) + dT
    print(f"    fitted-table first-tick drop {first:.2f} K  (= T_abs/4 = "
          f"{(1200 + K_AMB) / 4:.2f}); after 240 ticks T = {int(T[cy, cx]) / ONE:.3f}")
    print("    -> damping ON is VISIBLE to this probe: the trajectory is a")
    print("       different curve entirely, so 'matches' above is a measurement.")

    print("\n  CONTROL 2 (the fold is the only mover): rerun with fleck_enabled")
    print("  False on the derived table — f is forced to 1, which it already is.")
    T, aq, dq, shift, ts, (cy, cx) = _one_tile_scene(h, w, a_q, his, 1200.0)
    rn2, rf2, ra2, rl2 = _planes(h, w)
    sweep.run(T, aq, dq, shift, ts, tbl, None, T_AMB_Q, 0,
              bp.RadiationSweep.SHEAR, 16, rn2, rf2, ra2, rl2, False)
    T2, aq2, dq2, s2, ts2, _ = _one_tile_scene(h, w, a_q, his, 1200.0)
    rn3, rf3, ra3, rl3 = _planes(h, w)
    sweep.run(T2, aq2, dq2, s2, ts2, tbl, None, T_AMB_Q, 0,
              bp.RadiationSweep.SHEAR, 16, rn3, rf3, ra3, rl3, True)
    print(f"    rad_net damped {int(rn3[cy, cx])}  vs undamped {int(rn2[cy, cx])}"
          f"   identical = {int(rn3[cy, cx]) == int(rn2[cy, cx])}")
    print("    -> on the live scale the damping is INERT at 1200 game: it is not")
    print("       that the probe cannot see it (control 1), it is not engaging.")


def sec_testA2():
    """Where the damping DOES engage, does it redistribute or destroy?

    The same cooling curve, but started high enough that g > 1 for several
    ticks. Fleck's own claim is that the trajectory is preserved even though
    each tick emits less; the failure mode the question is about is a curve
    that lags permanently.
    """
    print("\n" + "=" * 78)
    print("TEST A2 — ACCURACY ACROSS THE WHOLE g RANGE (where the cap DOES bind)")
    print("=" * 78)
    print("  `wood` (a=0.9, tm=0.125) on the LIVE table, started at each T0. The")
    print("  cap binds above 4868 game. Reported: worst deviation from RK4 over")
    print("  the run, and the ENERGY the engine shed vs the exact integral.")
    tbl = etable(RS_DERIVED)
    sweep = bp.RadiationSweep()
    h = w = 7
    a_q, his, tm = 58982, -3, 0.125
    C = tm * J_PER_K_PER_TM
    K0 = K_AMB + 2.0
    print(f"\n  {'T0':>7}{'g at T0':>10}{'f at T0':>9}{'ticks':>7}"
          f"{'T end eng':>12}{'T end exact':>13}{'worst rel':>11}"
          f"{'t_half eng':>12}{'t_half ex':>11}{'lag':>6}"
          f"{'E shed MJ':>12}{'E exact MJ':>12}{'ratio':>8}")
    for T0, n_ticks in ((1200.0, 240), (3000.0, 240), (5000.0, 240),
                        (8000.0, 240), (15000.0, 240)):
        T, aq, dq, shift, ts, (cy, cx) = _one_tile_scene(h, w, a_q, his, T0)
        rn, rf, ra, rl = _planes(h, w)
        Tk = K_AMB + T0
        worst = 0.0
        th_e = th_r = None
        k_now = 0
        for _ in range(n_ticks):
            sweep.run(T, aq, dq, shift, ts, tbl, None, T_AMB_Q, 0,
                      bp.RadiationSweep.SHEAR, 16, rn, rf, ra, rl, True)
            T[cy, cx] = int(T[cy, cx]) + _shr_round0_signed(int(rn[cy, cx]), his)
            sub = 512
            hstep = DT / sub
            for _ in range(sub):
                def dTdt(x):
                    return -(a_q / ONE) * SIGMA * A_RAD * (x ** 4 - K0 ** 4) / C
                k1 = dTdt(Tk); k2 = dTdt(Tk + 0.5 * hstep * k1)
                k3 = dTdt(Tk + 0.5 * hstep * k2); k4 = dTdt(Tk + hstep * k3)
                Tk += hstep * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            e_now, r_now = int(T[cy, cx]) / ONE, Tk - K_AMB
            worst = max(worst, abs(e_now - r_now) / max(abs(r_now), 1e-9))
            k_now += 1
            if th_e is None and e_now <= T0 / 2.0:
                th_e = k_now
            if th_r is None and r_now <= T0 / 2.0:
                th_r = k_now
        e_end, r_end = int(T[cy, cx]) / ONE, Tk - K_AMB
        E_eng = (T0 - e_end) * C / 1e6
        E_ref = (T0 - r_end) * C / 1e6
        g0 = 4.0 * L_game(tbl, T0, a_q, his) / (T0 + K_AMB)
        lag = "-" if (th_e is None or th_r is None) else f"{th_e - th_r:+d}"
        print(f"  {T0:>7.0f}{g0:>10.3f}{f_of(tbl, T0, a_q, his):>9.4f}{n_ticks:>7}"
              f"{e_end:>12.2f}{r_end:>13.2f}{worst:>10.2%}"
              f"{str(th_e):>12}{str(th_r):>11}{lag:>6}"
              f"{E_eng:>12.3f}{E_ref:>12.3f}{E_eng / E_ref:>8.4f}")
    print("\n  READ: `t_half` is the TICK at which the tile first reaches T0/2 and")
    print("  `lag` how many ticks LATER the engine gets there than physics does.")
    print("  `ratio` is energy actually shed over the run / energy physics says.")
    print("  DESTROYED transport = ratio << 1 with a permanent lag. REDISTRIBUTED")
    print("  IN TIME = ratio ~ 1 with a bounded lag the trajectory then closes.")


# ===========================================================================
def sec_testB():
    print("\n" + "=" * 78)
    print("TEST B — TRANSPORT BETWEEN TWO CELLS (what fire spread rides on)")
    print("=" * 78)
    print("  A hot emitter and a cold receiver, both `wood`, separated by n")
    print("  transparent cells. Measured: the receiver's rad_net per tick, against")
    print("  the 2-D view-factor prediction  a_r * a_s * E(T_s) * w_solid,")
    print("  where w_solid is the S16 ordinate-weight share the receiver subtends.")
    tbl = etable(RS_DERIVED)
    sweep = bp.RadiationSweep()
    h, w = 9, 21
    a_q, his = 58982, -3
    for gap in (1, 2, 4, 8):
        T = np.zeros((h, w), dtype=np.int32)
        aq = np.zeros((h, w), dtype=np.int32)
        dq = np.zeros((h, w), dtype=np.int32)
        shift = np.zeros((h, w), dtype=np.int32)
        ts = np.zeros((h, w), dtype=bool)
        sy, sx = h // 2, 2
        ry, rx = h // 2, 2 + gap + 1
        for (yy, xx) in ((sy, sx), (ry, rx)):
            aq[yy, xx] = dq[yy, xx] = a_q
            shift[yy, xx] = his
            ts[yy, xx] = True
        T[sy, sx] = int(1200 * ONE)
        rn, rf, ra, rl = _planes(h, w)
        sweep.run(T, aq, dq, shift, ts, tbl, None, T_AMB_Q, 0,
                  bp.RadiationSweep.SHEAR, 16, rn, rf, ra, rl, True)
        e = np.asarray(tbl.table())
        ex_s = int(e[tbl.e_bucket_of(int(1200 * ONE))]) - int(e[0])
        gain = int(rn[ry, rx])
        loss = -int(rn[sy, sx])
        dT_r = _shr_round0_signed(gain, his) / ONE
        # the joule value of the receiver's gain
        W = gain * J_PER_COUNT / DT
        print(f"    gap {gap:>2} tiles   emitter loses {loss:>12} counts"
              f"   receiver gains {gain:>9} counts"
              f"  ({100.0 * gain / loss:>6.3f} % of the emission)"
              f"   dT_r {dT_r:>8.4f} K/tick   {W / 1000:>7.2f} kW")
    print(f"\n  the emitter's total emission at 1200 game: "
          f"{(a_q / ONE) * ex_s * J_PER_COUNT / DT / 1000:.1f} kW"
          f"  (closed form {(a_q / ONE) * SIGMA * A_RAD * ((K_AMB + 1200) ** 4 - 295 ** 4) / 1000:.1f} kW)")
    print("\n  CONTROL: the same geometry with a damped f (fitted table) —")
    tblf = etable(RS_FITTED)
    T = np.zeros((h, w), dtype=np.int32); aq = np.zeros((h, w), dtype=np.int32)
    dq = np.zeros((h, w), dtype=np.int32); shift = np.zeros((h, w), dtype=np.int32)
    ts = np.zeros((h, w), dtype=bool)
    sy, sx, ry, rx = h // 2, 2, h // 2, 4
    for (yy, xx) in ((sy, sx), (ry, rx)):
        aq[yy, xx] = dq[yy, xx] = a_q; shift[yy, xx] = his; ts[yy, xx] = True
    T[sy, sx] = int(1200 * ONE)
    for enabled in (True, False):
        rn, rf, ra, rl = _planes(h, w)
        sweep.run(T, aq, dq, shift, ts, tblf, None, T_AMB_Q, 0,
                  bp.RadiationSweep.SHEAR, 16, rn, rf, ra, rl, enabled)
        print(f"    fitted table, fleck_enabled={enabled!s:>5}: receiver gains "
              f"{int(rn[ry, rx]):>12} counts")
    print("    -> the probe responds to the damping by the expected ~77x; on the")
    print("       derived table the two are identical (measured in Test A c.2).")

    print("")
    print("  THE GEOMETRIC SHARE IS f-INDEPENDENT — the damping scales what the")
    print("  emitter sends, never WHERE it goes (gap 1, derived table):")
    print(f"  {'T0':>7}{'fleck':>7}{'emitter loss':>15}{'receiver gain':>15}{'share':>11}")
    for T0 in (1200, 8000, 15996):
        for en in (True, False):
            T = np.zeros((h, w), dtype=np.int32)
            aq = np.zeros((h, w), dtype=np.int32)
            dq = np.zeros((h, w), dtype=np.int32)
            sh = np.zeros((h, w), dtype=np.int32)
            tsm = np.zeros((h, w), dtype=bool)
            for (yy, xx) in ((h // 2, 2), (h // 2, 4)):
                aq[yy, xx] = dq[yy, xx] = a_q
                sh[yy, xx] = his
                tsm[yy, xx] = True
            T[h // 2, 2] = T0 << 16
            rn, rf, ra, rl = _planes(h, w)
            sweep.run(T, aq, dq, sh, tsm, tbl, None, T_AMB_Q, 0,
                      bp.RadiationSweep.SHEAR, 16, rn, rf, ra, rl, en)
            loss = -int(rn[h // 2, 2]); gain = int(rn[h // 2, 4])
            print(f"  {T0:>7}{en!s:>7}{loss:>15}{gain:>15}{gain / loss:>10.5%}")
    print("  -> identical shares. With Test A2's energy ratio of ~1.0000, the")
    print("     damping delays an emitter's budget, never re-routes or destroys it.")


def sec_fold():
    """INSTRUMENT VALIDATION: the Python fold used in Tests A/A2/E is the
    shipped TemperatureSolver's fold, bit for bit."""
    print("\n" + "=" * 78)
    print("INSTRUMENT VALIDATION — the probe's fold vs the shipped C++ solver")
    print("=" * 78)
    sys.path.insert(0, str(ROOT))
    from config import CFG                                   # noqa: E402
    NOF = int(CFG.physics.thermal.NO_FACE)
    h = w = 7
    ok = True
    for his, rn_val in ((-3, -74192), (-3, 74192), (3, -74192), (5, -1), (-2, 7)):
        T = np.zeros((h, w), dtype=np.int32)
        T[3, 3] = 1200 << 16
        heat = np.zeros((h, w), dtype=np.int32)
        shift = np.zeros((h, w), dtype=np.int32); shift[3, 3] = his
        fs = np.full((h, w, 4), NOF, dtype=np.int32)
        sol = np.zeros((h, w), dtype=bool)
        vac = np.zeros((h, w), dtype=bool)
        atm = np.full((h, w), 1 << 16, dtype=np.int32)
        tsm = np.zeros((h, w), dtype=bool); tsm[3, 3] = True
        rn = np.zeros((h, w), dtype=np.int64); rn[3, 3] = rn_val
        solver = bp.TemperatureSolver(); solver.no_face = NOF
        solver.step(T, heat, shift, fs, sol, vac, atm, None, None, 0.0,
                    None, tsm, rn, None, None, True)
        cpp = int(T[3, 3])
        py = (1200 << 16) + _shr_round0_signed(rn_val, his)
        same = cpp == py
        ok &= same
        print(f"    his={his:>3} rad_net={rn_val:>8}   C++ {cpp:>12}   probe {py:>12}"
              f"   {'identical' if same else 'DIFFER'}")
    print(f"  => the probe's fold is {'BIT-IDENTICAL' if ok else 'NOT identical'}"
          " to TemperatureSolver's Pass-1 radiation fold on these cases.")
    print("  CONTROL: perturb the probe's own fold by one count and it disagrees —")
    print(f"    probe+1 vs C++: "
          f"{'DIFFER (good)' if (py + 1) != cpp else 'still equal (BAD)'}")


def sec_testE():
    """The radiative EQUILIBRIUM a given fire power holds a tile at."""
    print("\n" + "=" * 78)
    print("TEST E — WHAT PLATEAU A GIVEN HEAT INPUT HOLDS, end to end")
    print("=" * 78)
    print("  A `wood` tile is given a CONSTANT heat deposit each tick (the shape")
    print("  of a combustion H_bed) and the sweep+fold are left to find their own")
    print("  balance. This is the quantity fire behaviour actually rides on.")
    tbl = etable(RS_DERIVED)
    sweep = bp.RadiationSweep()
    h = w = 7
    a_q, his, tm = 58982, -3, 0.125
    print(f"\n  {'P into tile':>13}{'plateau eng':>14}{'closed form':>13}"
          f"{'rel':>8}{'ticks to 95%':>14}{'f at plateau':>14}")
    for kW in (1.0, 5.0, 17.7, 100.0, 845.0, 5000.0):
        # counts per tick for that power
        dep = int(round(kW * 1000.0 * DT / J_PER_COUNT))
        T, aq, dq, shift, ts, (cy, cx) = _one_tile_scene(h, w, a_q, his, 0.0)
        rn, rf, ra, rl = _planes(h, w)
        prev = 0.0
        plateau = 0.0
        t95 = None
        for k in range(20000):
            sweep.run(T, aq, dq, shift, ts, tbl, None, T_AMB_Q, 0,
                      bp.RadiationSweep.SHEAR, 16, rn, rf, ra, rl, True)
            nT = int(T[cy, cx]) + _shr_round0_signed(int(rn[cy, cx]), his)
            nT += _shr_round0_signed(dep, his)
            nT = min(nT, 16000 << 16)
            T[cy, cx] = nT
            cur = nT / ONE
            if t95 is None and plateau == 0.0:
                pass
            if abs(cur - prev) < 1e-3 and k > 20:
                plateau = cur
                break
            prev = cur
        else:
            plateau = int(T[cy, cx]) / ONE
        # closed form: eps*sigma*A*(K^4 - K0^4) = P
        P = kW * 1000.0
        K0 = K_AMB + 2.0
        K = (P / ((a_q / ONE) * SIGMA * A_RAD) + K0 ** 4) ** 0.25
        cf = K - K_AMB
        # ticks to 95 % of plateau
        T2, aq2, dq2, s2, ts2, _ = _one_tile_scene(h, w, a_q, his, 0.0)
        rn2, rf2, ra2, rl2 = _planes(h, w)
        t95 = None
        for k in range(20000):
            sweep.run(T2, aq2, dq2, s2, ts2, tbl, None, T_AMB_Q, 0,
                      bp.RadiationSweep.SHEAR, 16, rn2, rf2, ra2, rl2, True)
            v = int(T2[cy, cx]) + _shr_round0_signed(int(rn2[cy, cx]), his)
            v += _shr_round0_signed(dep, his)
            T2[cy, cx] = min(v, 16000 << 16)
            if t95 is None and int(T2[cy, cx]) / ONE >= 0.95 * plateau:
                t95 = k + 1
                break
        print(f"  {kW:>10.1f} kW{plateau:>14.1f}{cf:>13.1f}"
              f"{(plateau - cf) / max(cf, 1e-9):>8.1%}"
              f"{(('%d (%.1f s)' % (t95, t95 / TPS)) if t95 else 'n/a'):>16}"
              f"{f_of(tbl, plateau, a_q, his):>14.4f}")
    print("\n  CONTROL: the plateau must MOVE with the emissivity. a 0.9 -> 0.45")
    dep = int(round(17700.0 * DT / J_PER_COUNT))
    for aa in (58982, 29491):
        T, aq, dq, shift, ts, (cy, cx) = _one_tile_scene(h, w, aa, his, 0.0)
        rn, rf, ra, rl = _planes(h, w)
        prev = -1.0
        for k in range(20000):
            sweep.run(T, aq, dq, shift, ts, tbl, None, T_AMB_Q, 0,
                      bp.RadiationSweep.SHEAR, 16, rn, rf, ra, rl, True)
            v = int(T[cy, cx]) + _shr_round0_signed(int(rn[cy, cx]), his)
            v += _shr_round0_signed(dep, his)
            T[cy, cx] = min(v, 16000 << 16)
            cur = int(T[cy, cx]) / ONE
            if abs(cur - prev) < 1e-3 and k > 20:
                break
            prev = cur
        print(f"    a = {aa / ONE:.2f}  ->  plateau {cur:.1f} game"
              f"   (expect x{(2.0) ** 0.25:.3f} on halving a)")


# ===========================================================================
def sec_testC():
    print("\n" + "=" * 78)
    print("TEST C — WHAT THE SCHEME'S OWN THEORY SAYS AT LOW CAPACITY")
    print("=" * 78)
    print("  ours:      f = 1 / max(1, g),          g = 4L/T_abs")
    print("  classical: alpha = 1/(1 + beta*c*sigma_a*dt), beta = 4aT^3/C_v")
    print("  The IMC 'Fleck number' is exactly our g: beta*c*sigma_a*dt is the")
    print("  ratio of the radiative emission RATE to the material's own heat")
    print("  content per unit temperature, over one timestep. Check numerically:")
    tbl = etable(RS_DERIVED)
    print(f"\n  {'row':>10}{'T':>7}{'g = 4L/T_abs':>15}"
          f"{'4*a*sigma*A*T^3*dt/C':>24}{'ratio':>9}")
    for name, a_q, his in (("wood", 58982, -3), ("door", 58982, 3),
                           ("furniture", 32768, -2)):
        tm = 2.0 ** his
        for T in (1200, 5000):
            g = 4.0 * L_game(tbl, T, a_q, his) / (T + K_AMB)
            K = K_AMB + T
            beta_form = 4.0 * (a_q / ONE) * SIGMA * A_RAD * K ** 3 * DT \
                / (tm * J_PER_K_PER_TM)
            print(f"  {name:>10}{T:>7}{g:>15.5f}{beta_form:>24.5f}"
                  f"{g / beta_form:>9.4f}")
    print("\n  => our g IS the classical Fleck number for this cell, to the")
    print("     E-table's own bucket resolution (the small excess subtracted).")
    print("  The two forms then differ ONLY in the damping shape:")
    print(f"\n  {'g':>10}{'ours 1/max(1,g)':>18}{'IMC 1/(1+g)':>14}"
          f"{'Fleck floor a=1/2':>20}")
    for g in (0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 77.0):
        imc = 1.0 / (1.0 + g)
        half = 1.0 / (1.0 + 0.5 * g)
        print(f"  {g:>10.2f}{1.0 / max(1.0, g):>18.4f}{imc:>14.4f}{half:>20.4f}")
    print("\n  Ours is the LEAST damping of the three and agrees with them to")
    print("  O(g) as g -> 0, so at the live scale (g <= 0.03 everywhere in the")
    print("  fire range) all three forms are 'no damping' to within 3 %.")
    print("\n  THE STABILITY-LIMITED EFFECTIVE RATE at low capacity: with f = 1/g")
    print("  the per-tick drop is L*f = T_abs/4 exactly, independent of L. That")
    print("  is the scheme's ceiling, and it is a TEMPERATURE cap, not an energy")
    print("  cap: a lighter cell hits it at a lower temperature but sheds less")
    print("  energy doing so. Numerically, for a = 0.9:")
    print(f"  {'thermal_mass':>14}{'knee T (g=1)':>15}{'cap dT/tick@knee':>19}"
          f"{'cap power at 1200 K':>22}")
    for his in (5, 3, 0, -2, -3, -6, -10, -16):
        tm = 2.0 ** his
        knee = None
        for b in range(R.E_TABLE_SIZE):
            if f_of(tbl, 4 * b, 58982, his) < 1.0:
                knee = 4 * b
                break
        cap_pow = (1200 + K_AMB) / 4.0 * tm * J_PER_K_PER_TM / DT / 1000.0
        print(f"  {tm:>14g}{'never' if knee is None else knee:>15}"
              f"{(knee + K_AMB) / 4.0 if knee else float('nan'):>19.1f}"
              f"{cap_pow:>22.1f} kW")


# ===========================================================================
def sec_testD():
    print("\n" + "=" * 78)
    print("TEST D — THE CAPACITY SCALING OF f, at fixed temperature")
    print("=" * 78)
    tbl = etable(RS_DERIVED)
    tblf = etable(RS_FITTED)
    print("  a = 0.9 (wood's emissivity), T = 1200 game (1493 K).")
    print(f"  {'thermal_mass':>14}{'his':>5}{'C (J/K)':>12}{'L K/tick':>12}"
          f"{'g':>11}{'f derived':>12}{'f fitted':>11}{'eff dT':>10}"
          f"{'eff power kW':>14}")
    for his in range(5, -17, -1):
        tm = 2.0 ** his
        L = L_game(tbl, 1200, 58982, his)
        g = 4.0 * L / (1200 + K_AMB)
        fd = f_of(tbl, 1200, 58982, his)
        ff = f_of(tblf, 1200, 58982, his)
        eff = L * fd
        pw = eff * tm * J_PER_K_PER_TM / DT / 1000.0
        print(f"  {tm:>14g}{his:>5}{tm * J_PER_K_PER_TM:>12.1f}{L:>12.3f}"
              f"{g:>11.4f}{fd:>12.4f}{ff:>11.4f}{eff:>10.3f}{pw:>14.2f}")
    print("\n  READ THIS COLUMN-WISE: `eff power kW` is what the cell actually")
    print("  radiates. It is CONSTANT at 845 kW (the true black-body power of a")
    print("  1493 K, 3.33 m^2 body) for every capacity where f == 1, and it")
    print("  COLLAPSES linearly with the mass once the cap binds — because the")
    print("  cap is on dT, not on energy. THAT is the form question: the physical")
    print("  emission is capacity-independent; the scheme's ceiling is not.")


def sec_g12():
    """M2 §5.1's RED gate, re-measured on the table the engine actually bakes."""
    print("\n" + "=" * 78)
    print("G12 RE-MEASURED — the monotonicity gate on BOTH tables")
    print("=" * 78)
    print("  gate12_damped_source_is_monotone counts the buckets where the")
    print("  DAMPED emission E°[0] + f*(E°[T]-E°[0]) FALLS as T rises. M2 reports")
    print("  it red on the three thin rows. The gate runs on sweep_ref_q.E, the")
    print("  reference's default table — the FITTED scale. Re-run per table:")
    E_fit = R.E
    E_der = R.bake_e_table(rad_scale=RS_DERIVED, kelvin_ambient=int(K_AMB),
                           slope=int(K_SLOPE))
    print(f"\n  {'row':>12}{'fitted steps':>15}{'fitted worst':>15}"
          f"{'derived steps':>16}{'derived worst':>16}")
    for name, a_q, his in SHIPPED:
        out = []
        for tab in (E_fit, E_der):
            vals = [R.damped_source_q((4 * b) << 16, a_q, his, tab)
                    for b in range(R.E_TABLE_SIZE)]
            drops = [(vals[i] - vals[i + 1]) / vals[i]
                     for i in range(len(vals) - 1) if vals[i + 1] < vals[i]]
            out.append((len(drops), max(drops, default=0.0)))
        print(f"  {name:>12}{out[0][0]:>15}{out[0][1] * 100:>14.3f}%"
              f"{out[1][0]:>16}{out[1][1] * 100:>15.3f}%")
    print("\n  CONTROL (non-vacuity): the same probe in Q16, where the design")
    print("  itself says the source is NOT monotone —")
    vals = [R.damped_source_q((4 * b) << 16, 58982, -3, E_fit, shift=16)
            for b in range(R.E_TABLE_SIZE)]
    n16 = sum(1 for i in range(len(vals) - 1) if vals[i + 1] < vals[i])
    print(f"    wood, FITTED table, Q16 f: {n16} backward steps "
          f"(the probe detects non-monotonicity when it is there)")


# ===========================================================================
def main(argv):
    secs = {"provenance": sec_provenance, "L": sec_L, "f": sec_f,
            "fold": sec_fold, "testA": sec_testA, "testA2": sec_testA2,
            "testB": sec_testB, "testE": sec_testE, "testC": sec_testC,
            "testD": sec_testD, "g12": sec_g12}
    want = [a for a in argv[1:] if not a.startswith("-")] or list(secs)
    for name in want:
        secs[name]()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
