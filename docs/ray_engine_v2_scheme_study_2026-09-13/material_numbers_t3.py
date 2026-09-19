# -*- coding: utf-8 -*-
"""T3 -- the instrument behind `report_t3.md`.

Sibling of `currency_audit_t2.py`. It MEASURES and PRINTS; it asserts nothing,
edits nothing and is imported by nothing in `src/` or `cpp/`. Every engine
number quoted in the report comes out of here, so a reader can re-run it
instead of trusting the prose.

    C:/Users/steen/anaconda3/python.exe \
        docs/ray_engine_v2_scheme_study_2026-09-13/material_numbers_t3.py

Needs the CPU extension built (`cpp/build_cpu_home.bat` -> cpp/build/Release).

Sections, matching the report:
  M1  sect.1.1  the per-tick gap fraction is exactly 2^-s
  M2  sect.1.1  the rod converges to the analytic erfc slab at alpha = 2^-s dx^2/dt
  M3  sect.4.2  the solid<->gas face, and the h a given shift implements
  M4  sect.5.3  the vacuum capacity floor: a cell at N == 0 still conducts
  M5  sect.3.4  the conduction dead-band, bisected
  M6  sect.5.4  is_vacuum vs n_bulk == 0 on live levels
  D   the paper derivation (Churchill-Chu h, the face table, Huggett)
"""
from __future__ import annotations

import math
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for p in (ROOT, os.path.join(ROOT, "src"),
          os.path.join(ROOT, "cpp", "build", "Release")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                                            # noqa: E402
import breach_physics as bp                                   # noqa: E402

# ---- the pinned currency (R13; report_p2b sect.2, report_t2 sect.2) --------
FP       = 65536
NO_FACE  = 63
RHO_C_PIN = 0.9e6                      # J/(m3.K) -- wood at ~12 % MC (R13)
TM_PIN    = 8
UNIT      = RHO_C_PIN / TM_PIN         # 112 500 J/(m3.K) per thermal_mass unit
TILE      = 0.333                      # m
DECK      = 2.5                        # m
V_TILE    = TILE * TILE * DECK
A_FACE    = TILE * DECK
DT        = 1.0 / 24.0
JPC       = RHO_C_PIN * V_TILE / (TM_PIN * 65536)     # J per heat count
RHO_CV    = 864.548                    # air, p/((gamma-1)T) -- report_t2 sect.2.1
C_V       = RHO_CV / UNIT
H_CONV    = 6.0                        # W/(m2.K) -- sect.4.1


def _solver(c_v=1.0, n_floor=0.01):
    s = bp.TemperatureSolver()
    s.no_face = NO_FACE
    s.cool_shift = 31                  # cooling OFF: isolate conduction
    s.cool_shift_vacuum = 31
    s.c_v = c_v
    s.n_floor_heat = n_floor
    return s


def _blank(shape):
    return (np.ascontiguousarray(np.zeros(shape, dtype=np.int32)),
            np.ascontiguousarray(np.zeros(shape, dtype=bool)),
            np.ascontiguousarray(np.full(shape, FP, dtype=np.int32)))


# ---------------------------------------------------------------- M1 -------
def m1_gap_fraction():
    print("== M1 (sect.1.1): is the per-tick gap fraction exactly 2^-s? ==")
    print(f"{'s':>3} {'gap raw':>12} {'measured dT_cold':>17} {'gap >> s':>10} {'equal':>6}")
    for s_shift in (8, 10, 12, 16, 20, 24):
        temp, isv, atm = _blank((1, 2))
        temp[0, 0] = 1000 * FP
        his = np.ascontiguousarray(np.full((1, 2), 3, dtype=np.int32))
        face = np.ascontiguousarray(np.full((1, 2, 4), NO_FACE, dtype=np.int32))
        face[0, 0, 2] = s_shift
        face[0, 1, 3] = s_shift
        solid = np.ascontiguousarray(np.ones((1, 2), dtype=bool))
        heat = np.ascontiguousarray(np.zeros((1, 2), dtype=np.int32))
        gap0 = int(temp[0, 0]) - int(temp[0, 1])
        _solver().step(temp, heat, his, face, solid, isv, atm)
        got, pred = int(temp[0, 1]), gap0 >> s_shift
        print(f"{s_shift:>3} {gap0:>12} {got:>17} {pred:>10} {str(got == pred):>6}")


# ---------------------------------------------------------------- M2 -------
def m2_rod_vs_erfc():
    print()
    print("== M2 (sect.1.1): does the rod converge to erfc at alpha = 2^-s dx^2/dt? ==")
    print(f"{'ticks':>8} {'cells>2%T0':>11} {'sqrt(at)/dx':>12} {'max rel err':>12}")
    N, s_shift, T0 = 400, 8, 4000 * FP
    for ticks in (500, 2000, 8000, 32000, 128000):
        temp, isv, atm = _blank((1, N))
        temp[0, 0] = T0
        his = np.ascontiguousarray(np.full((1, N), 3, dtype=np.int32))
        face = np.ascontiguousarray(np.full((1, N, 4), NO_FACE, dtype=np.int32))
        face[0, :-1, 2] = s_shift
        face[0, 1:, 3] = s_shift
        solid = np.ascontiguousarray(np.ones((1, N), dtype=bool))
        heat = np.ascontiguousarray(np.zeros((1, N), dtype=np.int32))
        sv = _solver()
        for _ in range(ticks):
            temp[0, 0] = T0                       # Dirichlet hot end
            sv.step(temp, heat, his, face, solid, isv, atm)
        temp[0, 0] = T0
        alpha = (2.0 ** -s_shift) * TILE * TILE / DT
        t = ticks * DT
        ana = np.array([T0 * math.erfc(i * TILE / (2 * math.sqrt(alpha * t)))
                        for i in range(N)])
        m = ana > 0.02 * T0
        rel = np.abs(temp[0].astype(float)[m] - ana[m]) / ana[m]
        print(f"{ticks:>8} {int(m.sum()):>11} {math.sqrt(alpha*t)/TILE:>12.2f} "
              f"{rel.max()*100:>11.2f} %")


# ------------------------------------------------------------- M3 / M4 -----
def _pair(s_shift, c_v, n_gas, t_solid=800.0, tm_shift=3, n_floor=0.01):
    """SOLID | GAS, one live face, one tick -> (dT_gas, dT_solid, e_cond_cap)."""
    temp, isv, atm = _blank((1, 2))
    temp[0, 0] = int(t_solid * FP)
    his = np.ascontiguousarray(np.array([[tm_shift, 0]], dtype=np.int32))
    ts = np.ascontiguousarray(np.array([[True, False]], dtype=bool))
    solid = np.ascontiguousarray(np.array([[True, False]], dtype=bool))
    face = np.ascontiguousarray(np.full((1, 2, 4), NO_FACE, dtype=np.int32))
    face[0, 0, 2] = s_shift
    face[0, 1, 3] = s_shift
    n_bulk = np.ascontiguousarray(
        np.array([[0, int(round(n_gas * FP))]], dtype=np.int32))
    heat = np.ascontiguousarray(np.zeros((1, 2), dtype=np.int32))
    sv = _solver(c_v, n_floor)
    before = temp.copy()
    sv.step(temp, heat, his, face, solid, isv, atm, None, None, 0.0,
            n_bulk, ts, None, None, None, None, False)
    return (int(temp[0, 1]) - int(before[0, 1]),
            int(temp[0, 0]) - int(before[0, 0]),
            int(sv.e_cond_cap_sum))


def m3_gas_face():
    print()
    print("== M3 (sect.4.2): SOLID(800 game, tm 8) | GAS(N=1), one face, one tick ==")
    print(f"{'s':>3} {'c_v':>11} {'gas dT [K]':>11} {'solid dT raw':>13} {'implied h':>11}")
    for s_shift in (9, 10, 11, 13):
        for c_v in (1.0, C_V):
            dg, ds, _ = _pair(s_shift, c_v, 1.0)
            h = (2.0 ** -s_shift) * (c_v * UNIT) * TILE / DT
            print(f"{s_shift:>3} {c_v:>11.7f} {dg/FP:>11.4f} {ds:>13} {h:>11.3f}")
    print("  the gas dT is c_v-INDEPENDENT on the T-form path (report_t2 sect.3.2);")
    print("  what c_v moves is the ENERGY, i.e. what the solid loses.")


def m4_vacuum_floor():
    print()
    print("== M4 (sect.5.3): the capacity floor -- does a cell at N == 0 conduct? ==")
    print(f"{'N [atm]':>10} {'N raw':>8} {'cap_used':>9} {'cap_real':>9} "
          f"{'gas dT raw':>11} {'solid dT':>9} {'e_cond_cap':>12}")
    n_floor_q, c_v_q = int(round(0.01 * FP)), int(round(C_V * FP))
    for n in (1.0, 0.05, 0.01, 0.002, 1.0 / FP, 0.0):
        dg, ds, cap = _pair(10, C_V, n)
        n_raw = int(round(n * FP))
        cu = max((max(n_raw, n_floor_q) * c_v_q) >> 16, 1)
        cr = (n_raw * c_v_q) >> 16
        print(f"{n:>10.6f} {n_raw:>8} {cu:>9} {cr:>9} {dg:>11} {ds:>9} {cap:>12}")
    dg, ds, _ = _pair(10, C_V, 0.0)
    de = (800 * FP) * 5 / 1024.0                       # gap * cap_used >> s
    print(f"  a hard-vacuum neighbour still takes {de/FP*JPC*24:.1f} W per face at a "
          f"800 K excess,")
    print(f"  = {de/FP*JPC*24/(5.670374419e-8*4*TILE*DECK*(1093.15**4-293.15**4))*100:.3f} %"
          f" of the same tile's radiative loss.")


# ---------------------------------------------------------------- M5 -------
def _one_tick_same_mat(s_shift, gap_game, tm_shift=3):
    temp, isv, atm = _blank((1, 2))
    temp[0, 0] = int(gap_game * FP)
    his = np.ascontiguousarray(np.full((1, 2), tm_shift, dtype=np.int32))
    face = np.ascontiguousarray(np.full((1, 2, 4), NO_FACE, dtype=np.int32))
    face[0, 0, 2] = s_shift
    face[0, 1, 3] = s_shift
    solid = np.ascontiguousarray(np.ones((1, 2), dtype=bool))
    heat = np.ascontiguousarray(np.zeros((1, 2), dtype=np.int32))
    _solver().step(temp, heat, his, face, solid, isv, atm)
    return int(temp[0, 1]), int(temp[0, 0]) - int(gap_game * FP)


def m5_dead_band():
    print()
    print("== M5 (sect.3.4): the conduction dead-band, bisected ==")
    print(f"{'row':<12} {'s':>3} {'2^s/65536 [K]':>14} {'measured [K]':>13}")
    for row, s_shift in (("steel/hull", 18), ("glass", 22), ("wood", 24),
                         ("(ships)", 8)):
        lo, hi = 0.0, max(4096.0, 2.0 ** s_shift / FP * 4)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if _one_tick_same_mat(s_shift, mid)[0] >= 1:
                hi = mid
            else:
                lo = mid
        print(f"{row:<12} {s_shift:>3} {2.0**s_shift/FP:>14.3f} {hi:>13.3f}")
    print("  below the band the HOT cell still loses (floordiv toward -inf):")
    for gap in (1.0, 100.0, 255.0, 256.0, 1000.0):
        gain, loss = _one_tick_same_mat(24, gap)
        print(f"    wood s=24 gap={gap:>7.1f} K : cold +{gain} raw, hot {loss} raw")


# ---------------------------------------------------------------- M6 -------
def m6_live_vacuum():
    print()
    print("== M6 (sect.5.4): is_vacuum vs n_bulk == 0, on live levels ==")
    from simulation.simulation import Simulation
    from level_loader import load as load_level
    for name in ("airlock_demo", "fire_tuning"):
        try:
            sim = Simulation(load_level(name, levels_dir=os.path.join(ROOT, "levels")),
                             seed=1, breach_physics=bp, enable_recorder=False)
        except Exception as exc:                      # pragma: no cover
            print(f"  ({name}: {exc})")
            continue
        g = sim.gmap
        vac = g.is_vacuum & (~g.thermal_solid) & (~g.solid)
        n = g._gas_bulk_n_raw()
        print(f"  {name:<14} open-vacuum cells {int(vac.sum()):>4}  "
              f"n_bulk min={n[vac].min() if vac.any() else '-'} "
              f"max={n[vac].max() if vac.any() else '-'} "
              f"nonzero={int((n[vac] != 0).sum()) if vac.any() else 0}")
        for _ in range(120):
            sim.set_paused(False)
            sim.step()
        n = g._gas_bulk_n_raw()
        if vac.any():
            print(f"  {'':<14} after 120 ticks:      "
                  f"min={n[vac].min()} max={n[vac].max()} "
                  f"nonzero={int((n[vac] != 0).sum())}")


# ----------------------------------------------------------- derivation ----
_AIR_A4 = [  # Incropera & DeWitt Table A.4: T[K], nu, alpha, k, Pr
    (250, 11.44e-6, 15.9e-6, 22.3e-3, 0.720), (300, 15.89e-6, 22.5e-6, 26.3e-3, 0.707),
    (350, 20.92e-6, 29.9e-6, 30.0e-3, 0.700), (400, 26.41e-6, 38.3e-6, 33.8e-3, 0.690),
    (450, 32.39e-6, 47.2e-6, 37.3e-3, 0.686), (500, 38.79e-6, 56.7e-6, 40.7e-3, 0.684),
    (550, 45.57e-6, 66.7e-6, 43.9e-3, 0.683), (600, 52.69e-6, 76.9e-6, 46.9e-3, 0.685),
    (650, 60.21e-6, 87.3e-6, 49.7e-3, 0.690), (700, 68.10e-6, 98.0e-6, 52.4e-3, 0.695),
    (750, 76.37e-6, 109.0e-6, 54.9e-3, 0.702), (800, 84.93e-6, 120.0e-6, 57.3e-3, 0.709),
    (900, 102.9e-6, 143.0e-6, 62.0e-3, 0.720), (1000, 121.9e-6, 168.0e-6, 66.7e-3, 0.726),
]


def _air(T):
    T = max(250.0, min(1000.0, T))
    for i in range(len(_AIR_A4) - 1):
        if _AIR_A4[i][0] <= T <= _AIR_A4[i + 1][0]:
            a, b = _AIR_A4[i], _AIR_A4[i + 1]
            f = (T - a[0]) / (b[0] - a[0])
            return tuple(a[j] + f * (b[j] - a[j]) for j in (1, 2, 3, 4))
    return _AIR_A4[-1][1:]


def churchill_chu(dT, L=DECK, T_amb=293.15):
    """Churchill & Chu (1975); Incropera eq. 9.26 (all Ra) / 9.27 (Ra <= 1e9)."""
    nu, al, k, Pr = _air(0.5 * (T_amb + dT + T_amb))
    Ra = 9.81 * (1.0 / (T_amb + 0.5 * dT)) * dT * L ** 3 / (nu * al)
    if Ra <= 1e9:
        Nu = 0.68 + 0.670 * Ra ** 0.25 / (1 + (0.492 / Pr) ** (9 / 16)) ** (4 / 9)
        reg = "laminar 9.27"
    else:
        Nu = (0.825 + 0.387 * Ra ** (1 / 6)
              / (1 + (0.492 / Pr) ** (9 / 16)) ** (8 / 27)) ** 2
        reg = "turbulent 9.26"
    return Ra, Nu, Nu * k / L, reg


def d_derivation():
    print()
    print("== D (sect.4.1): natural convection on a 2.5 m vertical wall ==")
    print(f"{'dT [K]':>8} {'Ra_L':>10} {'Nu_L':>8} {'h [W/m2K]':>10}  regime")
    for dT in (5, 10, 20, 50, 100, 200, 300, 500, 700, 1000, 1300):
        Ra, Nu, h, reg = churchill_chu(dT)
        print(f"{dT:>8} {Ra:>10.2e} {Nu:>8.1f} {h:>10.2f}  {reg}")
    print("  L-independence (turbulent: Nu ~ Ra^1/3 ~ L, so h = Nu k/L is L-free):")
    for L in (0.333, 1.0, 2.5, 5.0):
        hs = [churchill_chu(d, L=L)[2] for d in (10, 100, 700)]
        print(f"    L={L:>5.3f} m : h(10)={hs[0]:5.2f}  h(100)={hs[1]:5.2f}  "
              f"h(700)={hs[2]:5.2f}")

    print()
    print("== D (sect.3.2/4.2): the derived face shifts ==")
    KAP = dict(air=0.0257, hull=50.0, steel=50.0, glass=1.40, wood=0.12,
               door=0.12, door_closed=0.12, furniture=0.12, kindling=0.12,
               foliage=0.12)
    RC = dict(air=RHO_CV, hull=3.6e6, steel=3.6e6, glass=1.95e6, wood=0.9e6,
              door=0.9e6, door_closed=0.9e6, furniture=0.9e6, kindling=0.9e6,
              foliage=0.9e6)

    def sh(U, rc):
        r = U * DT / (rc * TILE)
        return r, -math.log2(r), int(round(-math.log2(r)))

    print("  solid|solid self-faces (U = kappa/dx):")
    for n in ("hull", "steel", "glass", "wood", "door"):
        r, s, si = sh(KAP[n] / TILE, RC[n])
        print(f"    {n:<12} alpha={KAP[n]/RC[n]:.3e}  r={r:.3e}  s={s:6.2f} -> {si}"
              f"   dx2/alpha = {TILE*TILE/(KAP[n]/RC[n])/3600:8.1f} h")
    print("  solid|gas faces (U = 1/((dx/2)/kappa_solid + 1/h)):")
    for n in ("hull", "steel", "glass", "wood", "furniture"):
        U = 1.0 / ((TILE / 2) / KAP[n] + 1.0 / H_CONV)
        r, s, si = sh(U, RHO_CV)
        print(f"    air|{n:<11} U={U:7.3f} W/m2K  r={r:.3e}  s={s:6.2f} -> {si}"
              f"   gas e-fold {2.0**si/24:7.1f} s")

    print()
    print("== D (sect.5.1/5.2): the vacuum arithmetic ==")
    P_ATM, LAM = 101325.0, 68e-9
    p_kn = P_ATM * LAM / TILE
    print(f"  lambda(air, 1 atm, 293 K) = {LAM*1e9:.0f} nm; lambda == tile at "
          f"{p_kn:.4f} Pa = {p_kn/P_ATM:.3e} atm")
    print(f"  n_floor_heat = 0.01 = {0.01*P_ATM:.1f} Pa = {0.01*P_ATM/p_kn:.2e}x it "
          f"({math.log10(0.01*P_ATM/p_kn):.2f} orders)")
    print(f"  ONE Q16.16 LSB of N = {P_ATM/FP:.4f} Pa -> Kn = "
          f"{LAM*P_ATM/(P_ATM/FP)/TILE:.4f} (continuum)")
    print(f"  -> the Knudsen threshold is {p_kn/(P_ATM/FP):.3f} of one LSB: "
          f"SUB-REPRESENTABLE, so the mask is N_raw == 0")

    print()
    print("== D (sect.6.3): combustion, against Huggett 1980 ==")
    n_tile = 101325.0 * V_TILE / (8.314462 * 293.15)
    per_o2 = n_tile * 0.0319988 * 13.1e6
    h_bed = 18125.0 * 128 * JPC
    print(f"  one tile at 1 atm      = {n_tile:.4f} mol")
    print(f"  1 unit of N_O2         = {n_tile*0.0319988:.5f} kg O2 -> "
          f"{per_o2/1e6:.4f} MJ  (13.1 MJ/kg-O2)")
    print(f"  H_BED (18125 << 7)     = {18125*128:.0f} counts = {h_bed/1e6:.4f} MJ"
          f"   = {h_bed/per_o2*100:.1f} % of Huggett")
    print(f"  H_fuel (4.0)           = {4.0*JPC:.4f} J        "
          f"   H_BED/H_fuel = {18125*128/4.0:,.0f}")
    print(f"  H_BED_M for a 25 % feedback share = {0.25*per_o2/JPC/128:.1f}  (shift 7)")
    print(f"  H_fuel  for a 75 % plume share    = {0.75*per_o2/JPC:,.0f} counts "
          f"-> M={0.75*per_o2/JPC/512:.0f} at shift 9 (Q16.16 caps at 32768)")
    print(f"  furniture hp 30 / fuel_per_o2 0.7 = {30/0.7:.3f} N_O2 = "
          f"{30/0.7*n_tile*0.0319988:.2f} kg O2 -> {30/0.7*per_o2/1e6:.1f} MJ "
          f"= {30/0.7*per_o2/13e6:.1f} kg of wood")


if __name__ == "__main__":
    m1_gap_fraction()
    m2_rod_vs_erfc()
    m3_gas_face()
    m4_vacuum_floor()
    m5_dead_band()
    m6_live_vacuum()
    d_derivation()
