"""EOS P6.8 — re-derived GPU FireSimulation::step bit-identity check (runs inside
the GPU subprocess).

The fire pass changed in the EOS refactor and the S6 kernel was STALE:
  * the O2 gate now reads the REAL bulk O2 density plane `n_o2`, together with
    `n_total` (continuous-O2 law, docs/continuous_o2_law_design_2026-07-24.md):
    the sustain factor is LINEAR in the local O2 MOLE FRACTION
    X = Σn_o2/Σn_total over open neighbours, clamped between an extinction
    limit (o2_frac_ext) and the ambient fraction (o2_frac_amb) — NOT the old
    absolute-density smoothstep(P_min, P_full);
  * the own-tile plume deposit is the plume->T shim — a `temperature` deposit
    self-limited against T_FLAME_MAX (clamp01 taper + belt-and-suspenders
    headroom hard-cap + saturating add) — NOT the retired atmosphere overpressure
    write (`atmosphere` is now vestigial: read-only, unread).
This gate re-proves the re-derived kernel (cuda_fire.cu) bit-identical to the
CURRENT CPU FireSimulation::step.

PASS STRUCTURE / ORDERING (the port's correctness argument): five device passes
(P2 logistic feedback → P3 plume->T → P4 smoke scatter → P5 wall burn → P6 clamp),
one per CPU loop, launched as a barriered chain. EVERY pass is an own-index write
with read-only neighbour reads (P2's O2 mole-fraction sums over `n_o2`/`n_total`),
an order-free integer atomicAdd scatter (P4 smoke), or an order-free device
counter (P5 destroyed). NO cell reads another cell's within-pass-written
fire/temperature — there is NO combustion-style Gauss-Seidel coupling — so the
parallel schedule reproduces the CPU sequential result bit-for-bit. The
P2-reads-T / P3-writes-T dependency is separated by the launch barrier (P2 sees
tick-entry T, matching the CPU's fully-sequential logistic-then-plume order).

Three gates:

  PART 1 — ISOLATED (synthetic, all branches + both T_FLAME_MAX self-limiter
  paths FORCED): random fuzz over sizes/regimes PLUS deterministic forcers —
  no-O2 (starve), full-O2 (grow), the plume sat-clamp-to-zero (T >= T_FLAME_MAX),
  the plume headroom HARD-CAP (uncapped dT overshoots -> result pinned exactly at
  T_FLAME_MAX, asserted), below-ambient (T<0) sat-clamp-to-one, wind fan/strip,
  wall burn-through (destroyed set + no drops/dupes), snap-extinguish, degenerate
  1xN / Nx1, all-solid + all-vacuum (empty O2-fraction sum), x_degenerate (the
  o2_frac_ext >= o2_frac_amb misconfig -> a step, mirroring the old P_degenerate
  smoothstep-step branch), non-identity temp_scale (recip_temp_scale path), a
  dense fire block (overlapping smoke atomicAdd), and the host max early-exit
  (fields untouched). CPU FireSimulation.step vs GPU cuda_fire_step on identical
  copies, byte-for-byte on fire/temperature/smoke/wall_hp + SET-equal destroyed.

  PART 2 — TRAJECTORY (the review's §4 P6.8 digest gate): ignition in an O2-rich
  room, plume heating (a cluster pinned at the T_FLAME_MAX ceiling), fire
  self-starving as the O2 MOLE FRACTION depletes (n_total held fixed while n_o2
  is drawn down — the same "O2 converts to inert gas, total roughly conserved"
  shape the real combustion pass produces), and wall burn-through — driven for
  130 ticks. TWO lockstep states (CPU-backend copy vs GPU-backend copy) stepped
  on identical evolving inputs, asserting per-tick byte-identity on fire /
  temperature / smoke / wall_hp + SET-equal destroyed over the WHOLE trajectory.
  The scenario is asserted to actually drive the pass hard (walls burned through,
  fire starved out as O2 depleted, temperature climbed under the plume, the
  T_FLAME_MAX ceiling reached and held — never exceeded).

  PART 3 — the CUDA build's CPU path still reproduces the committed default-
  scenario golden (proves the P6.8 C++ changes altered no CPU trajectory).

Prints ``P68_RESULT: PASS``/``FAIL`` and exits 0/1.
"""
from __future__ import annotations

import sys

import numpy as np

# Import the CUDA build FIRST so it is the cached `breach_physics`.
import breach_physics as bp

FP_ONE = 65536

# The mutated fields compared byte-for-byte (fire/temperature/smoke/wall_hp).
# `atmosphere` is vestigial (unread) and never written; `n_o2`/`n_total`/wind/
# masks are read-only inputs.
MUT = ("fire", "temperature", "smoke", "wall_hp")

# GPU dial kwargs, matching bindings.cpp cuda_fire_step's py::arg names EXACTLY
# (P_min/P_full RETIRED from the O2 gate, REPLACED by o2_frac_ext/o2_frac_amb —
# the continuous-O2 law's mole-fraction span; temp_gain_scale + T_FLAME_MAX
# still present from the earlier plume->T shim).
DIALS = ("k_grow", "k_die", "fire_T_ext", "fire_T_span", "fuel_ref",
         "o2_frac_ext", "o2_frac_amb", "I_min", "k_wind_fan", "k_wind_strip",
         "fire_pressure_gain", "smoke_emission", "wall_damage", "temp_scale",
         "temp_gain_scale", "T_FLAME_MAX")

# The full FireParams surface (incl. the vestigial p_expand_ref/P_min/P_full,
# set on the CPU object but not passed to the GPU — all three are unread by
# both paths now, kept only so old configs/bindings don't hard-error).
_PARAM_DEFAULTS = dict(
    k_grow=4.0, k_die=2.0, fire_T_ext=350.0, fire_T_span=150.0, fuel_ref=60.0,
    o2_frac_ext=0.13, o2_frac_amb=0.21, I_min=0.02, k_wind_fan=0.5,
    k_wind_strip=0.5, fire_pressure_gain=0.15, p_expand_ref=1.30,
    smoke_emission=0.8, wall_damage=0.4, temp_scale=float(FP_ONE),
    temp_gain_scale=50.0, T_FLAME_MAX=2000.0,
)


def _quantize(x):
    """Round-to-nearest Q16.16 (matches fixedpoint::quantize)."""
    x = np.asarray(x, dtype=np.float64)
    return np.int32(np.where(x >= 0, np.floor(x * FP_ONE + 0.5),
                             np.ceil(x * FP_ONE - 0.5)))


def make_params(**over):
    """Return (FireParams object for the CPU sim, dials dict for the GPU call)
    from ONE source of truth so the two paths cannot disagree on a dial."""
    base = dict(_PARAM_DEFAULTS)
    base.update(over)
    fp = bp.FireParams()
    for k, v in base.items():
        setattr(fp, k, float(v))
    dials = {k: float(base[k]) for k in DIALS}
    return fp, dials


def _contig(state):
    return {k: np.ascontiguousarray(v) for k, v in state.items()}


def run_pair(state, fp, dials, dt):
    """Run CPU FireSimulation.step + GPU cuda_fire_step on identical copies."""
    sim = bp.FireSimulation()
    sim.params = fp
    c = {k: state[k].copy() for k in state}
    d_cpu = sim.step(c["fire"], c["atmosphere"], c["n_o2"], c["n_total"],
                     c["smoke"], c["wall_hp"], c["temperature"], c["wind_x"],
                     c["wind_y"], c["is_wall"], c["is_vacuum"], c["flammable"],
                     dt)
    g = {k: state[k].copy() for k in state}
    d_gpu = bp.cuda_fire_step(
        g["fire"], g["atmosphere"], g["n_o2"], g["n_total"], g["smoke"],
        g["wall_hp"], g["temperature"], g["wind_x"], g["wind_y"], g["is_wall"],
        g["is_vacuum"], g["flammable"], dt, **dials)
    return c, list(d_cpu), g, list(d_gpu)


def compare(tag, c, d_cpu, g, d_gpu):
    ok = True
    for k in MUT:
        if not np.array_equal(c[k], g[k]):
            ok = False
            n_mis = int(np.count_nonzero(c[k] != g[k]))
            idx = int(np.argmax(c[k] != g[k]))
            print(f"  {tag}: {k} {n_mis} MISMATCH (first @ {idx}: "
                  f"cpu={c[k].flat[idx]} gpu={g[k].flat[idx]})")
    s_cpu = [tuple(t) for t in d_cpu]
    s_gpu = [tuple(t) for t in d_gpu]
    if set(s_cpu) != set(s_gpu):
        ok = False
        print(f"  {tag}: destroyed SET mismatch cpu={sorted(s_cpu)} "
              f"gpu={sorted(s_gpu)}")
    if len(s_gpu) != len(set(s_gpu)):
        ok = False
        print(f"  {tag}: GPU destroyed list has DUPLICATES ({s_gpu})")
    if len(s_cpu) != len(s_gpu):
        ok = False
        print(f"  {tag}: destroyed length cpu={len(s_cpu)} gpu={len(s_gpu)}")
    return ok


def _make_random_state(rng, h, w, wind_mag=0.0, hot=True):
    """Rich random fire state exercising every P2/P3/P4/P5 branch.

    n_o2/n_total are constructed as a MOLE FRACTION numerator/denominator pair
    (continuous-O2 law): n_total is a per-cell "total gas density" baseline and
    n_o2 = frac * n_total, with `frac` swept across the full range incl. 0
    (starve), the extinction limit (~0.13), ambient (~0.21), and above-ambient
    (full richness) — the SAME construction PART 2's trajectory uses, and the
    same one both the CPU and GPU sides read (apples-to-apples).
    """
    n = h * w
    flammable = (rng.random(n) < 0.6).reshape(h, w)
    is_wall = (rng.random(n) < 0.5).reshape(h, w)
    is_vacuum = (rng.random(n) < 0.10).reshape(h, w)

    # fire: mostly lit on flammable tiles (some 0, a few below thresh, some near 1)
    fire = np.zeros(n)
    lit = (rng.random(n) < 0.55)
    fire[lit] = rng.random(int(lit.sum())) * 1.0
    fire[rng.random(n) < 0.05] = 0.0005          # below the 0.001 early-exit thresh
    fire = _quantize(fire).reshape(h, w)

    # n_total: the mole-fraction denominator (a per-cell "total gas density"
    # baseline) — mostly a plausible ambient-ish range, with a slice forced
    # tiny to exercise the X_N_FLOOR guard.
    total = rng.random(n) * 1.0 + 0.3            # [0.3, 1.3)
    total[rng.random(n) < 0.08] = 0.0005         # near-vacuum -> X_N_FLOOR guard
    n_total = _quantize(total).reshape(h, w)

    # n_o2: the mole-fraction numerator, frac * total — full range incl. 0
    # (starve), the extinction limit, ambient, and > ambient (full richness).
    frac = rng.random(n) * 0.35
    frac[rng.random(n) < 0.15] = 0.0
    n_o2 = _quantize(frac * total).reshape(h, w)

    # temperature (dT above ambient): hot enough to grow, some negative, some
    # near/above T_FLAME_MAX (2000) to exercise the plume ceiling.
    lo = 300.0 if hot else -200.0
    t = rng.random(n) * 900.0 + lo
    t[rng.random(n) < 0.08] = -150.0             # below ambient (sat clamp to 1)
    t[rng.random(n) < 0.05] = 1990.0             # just under the ceiling
    t[rng.random(n) < 0.03] = 2100.0             # over the ceiling (sat -> 0)
    temperature = _quantize(t).reshape(h, w)

    # wall_hp: some near 0 for burn-through, some high for sustained fuel
    whp = rng.random(n) * 60.0
    whp[rng.random(n) < 0.12] = 0.02             # burn through this tick
    wall_hp = _quantize(whp).reshape(h, w)

    smoke = _quantize(rng.random(n) * 0.3).reshape(h, w)
    atmosphere = _quantize(rng.random(n) * 1.2).reshape(h, w)   # vestigial

    wx = _quantize((rng.random(n) * 2 - 1) * wind_mag).reshape(h, w)
    wy = _quantize((rng.random(n) * 2 - 1) * wind_mag).reshape(h, w)

    return _contig(dict(
        fire=fire.astype(np.int32), atmosphere=atmosphere.astype(np.int32),
        n_o2=n_o2.astype(np.int32), n_total=n_total.astype(np.int32),
        smoke=smoke.astype(np.int32),
        wall_hp=wall_hp.astype(np.int32), temperature=temperature.astype(np.int32),
        wind_x=wx.astype(np.int32), wind_y=wy.astype(np.int32),
        is_wall=is_wall, is_vacuum=is_vacuum, flammable=flammable))


def _blank(h, w):
    """A zeroed state (all air, no fire) to layer deterministic forcers onto."""
    z = lambda: np.zeros((h, w), dtype=np.int32)
    b = lambda: np.zeros((h, w), dtype=bool)
    return dict(fire=z(), atmosphere=z(), n_o2=z(), n_total=z(), smoke=z(),
                wall_hp=z(), temperature=z(), wind_x=z(), wind_y=z(),
                is_wall=b(), is_vacuum=b(), flammable=b())


def part1_isolated() -> bool:
    print("PART 1 — isolated GPU vs CPU (synthetic + deterministic forcers):")
    ok = True
    rng = np.random.default_rng(20260711)
    fp, dials = make_params()

    # (a) random fuzz over sizes / wind regimes (incl. degenerate 1xN, Nx1).
    n_cfg = 0
    for (h, w, wind_mag) in [(16, 16, 0.0), (16, 16, 3.0), (24, 32, 12.0),
                             (31, 17, 40.0), (40, 40, 0.5), (1, 50, 5.0),
                             (50, 1, 5.0), (8, 8, 100.0)]:
        for _ in range(4):
            n_cfg += 1
            st = _make_random_state(rng, h, w, wind_mag)
            c, dc, g, dg = run_pair(st, fp, dials, 1.0 / 24.0)
            ok &= compare(f"rand {h}x{w} wind={wind_mag}", c, dc, g, dg)

    # (b) no-O2 starve vs full-O2 grow (same hot lit tile). n_total is a fixed
    #     ambient baseline (1.0); n_o2 is the fraction NUMERATOR — 0.0 (X=0,
    #     below extinction) vs 0.30 (X=0.30, above ambient 0.21 -> full o2f).
    for tag, o2_frac, want in (("no-O2 (starve)", 0.0, "die"),
                               ("full-O2 (grow)", 0.30, "grow")):
        st = _blank(5, 5)
        st["flammable"][2, 2] = True
        st["is_wall"][2, 2] = True
        st["fire"][2, 2] = _quantize(0.5)
        st["temperature"][2, 2] = _quantize(700.0)     # hot
        st["wall_hp"][2, 2] = _quantize(55.0)          # fuel present
        st["n_total"][:] = _quantize(1.0)              # ambient baseline denom
        st["n_o2"][:] = _quantize(o2_frac)              # the driver (numerator)
        st["is_wall"][2, 2] = True
        st = _contig(st)
        c, dc, g, dg = run_pair(st, fp, dials, 1.0 / 24.0)
        ok &= compare(tag, c, dc, g, dg)
        i0 = _quantize(0.5)[()]
        f1 = int(c["fire"][2, 2])
        if want == "die" and not (f1 < i0):
            ok = False
            print(f"  {tag}: expected decay, got fire {i0}->{f1}")
        if want == "grow" and not (f1 > i0):
            ok = False
            print(f"  {tag}: expected growth, got fire {i0}->{f1}")

    # (c) plume sat-clamp-to-zero: T >= T_FLAME_MAX -> no deposit (T unchanged).
    #     (O2 gate irrelevant here — the plume deposit reads only fire/T.)
    st = _blank(4, 4)
    st["fire"][1, 1] = _quantize(0.99)
    st["temperature"][1, 1] = _quantize(2100.0)        # above the 2000 ceiling
    st["wall_hp"][1, 1] = _quantize(50.0)
    st = _contig(st)
    c, dc, g, dg = run_pair(st, fp, dials, 1.0)
    ok &= compare("plume sat->0 (T>ceiling)", c, dc, g, dg)
    if int(c["temperature"][1, 1]) != int(_quantize(2100.0)[()]):
        ok = False
        print(f"  sat->0: T changed ({int(c['temperature'][1,1])}) — deposit "
              f"was NOT gated to zero")

    # (d) plume headroom HARD-CAP: extreme gain/scale so the uncapped deposit
    #     overshoots the ceiling; assert the result is pinned EXACTLY at
    #     T_FLAME_MAX (proves the min(dT, headroom) branch executed) + GPU==CPU.
    fp_cap, dials_cap = make_params(fire_pressure_gain=1000.0, temp_gain_scale=10.0)
    st = _blank(4, 4)
    st["fire"][2, 2] = _quantize(0.99)
    st["temperature"][2, 2] = _quantize(1000.0)        # headroom = 1000, sat = 0.5
    st["wall_hp"][2, 2] = _quantize(50.0)
    st = _contig(st)
    c, dc, g, dg = run_pair(st, fp_cap, dials_cap, 1.0)
    ok &= compare("plume headroom cap", c, dc, g, dg)
    t_ceiling = int(_quantize(2000.0)[()])
    if int(c["temperature"][2, 2]) != t_ceiling:
        ok = False
        print(f"  headroom cap: T={int(c['temperature'][2,2])} != ceiling "
              f"{t_ceiling} — the hard-cap branch did not pin at T_FLAME_MAX")

    # (e) below-ambient T<0 -> sat clamps to 1 (deposit not amplified past 1x).
    st = _blank(4, 4)
    st["fire"][1, 2] = _quantize(0.6)
    st["temperature"][1, 2] = _quantize(-120.0)
    st["wall_hp"][1, 2] = _quantize(40.0)
    st = _contig(st)
    c, dc, g, dg = run_pair(st, fp, dials, 1.0 / 24.0)
    ok &= compare("plume T<0 sat->1", c, dc, g, dg)

    # (f) all-solid + all-vacuum: the O2 mole-fraction sum has no open
    #     neighbour -> both Σn_o2/Σn_total are 0 -> den floors -> X = 0.
    for tag, key in (("all-solid (empty O2-frac sum)", "is_wall"),
                     ("all-vacuum (empty O2-frac sum)", "is_vacuum")):
        st = _blank(6, 6)
        st[key][:] = True
        st["flammable"][:] = True
        st["fire"][3, 3] = _quantize(0.4)
        st["is_wall"][3, 3] = True
        st["temperature"][3, 3] = _quantize(600.0)
        st["wall_hp"][3, 3] = _quantize(30.0)
        st["n_total"][:] = _quantize(1.0)
        st["n_o2"][:] = _quantize(0.8)
        st = _contig(st)
        c, dc, g, dg = run_pair(st, fp, dials, 1.0 / 24.0)
        ok &= compare(tag, c, dc, g, dg)

    # (g) x_degenerate: o2_frac_amb <= o2_frac_ext -> the linear-law STEP branch
    #     (mirrors the old P_degenerate smoothstep-step branch).
    fp_deg, dials_deg = make_params(o2_frac_ext=0.21, o2_frac_amb=0.21)
    st = _make_random_state(rng, 20, 20, 2.0)
    c, dc, g, dg = run_pair(st, fp_deg, dials_deg, 1.0 / 24.0)
    ok &= compare("x_degenerate (linear-law step)", c, dc, g, dg)

    # (h) non-identity temp_scale (recip_temp_scale divide path).
    fp_ts, dials_ts = make_params(temp_scale=32768.0)
    st = _make_random_state(rng, 18, 22, 6.0)
    c, dc, g, dg = run_pair(st, fp_ts, dials_ts, 1.0 / 24.0)
    ok &= compare("non-identity temp_scale", c, dc, g, dg)

    # (i) dense fire block -> overlapping smoke atomicAdd (order-free proof).
    st = _blank(10, 10)
    st["flammable"][3:7, 3:7] = True
    st["is_wall"][3:7, 3:7] = True
    st["fire"][3:7, 3:7] = _quantize(0.7)
    st["temperature"][3:7, 3:7] = _quantize(650.0)
    st["wall_hp"][3:7, 3:7] = _quantize(45.0)
    st["n_total"][:] = _quantize(1.0)
    st["n_o2"][:] = _quantize(0.9)
    st = _contig(st)
    c, dc, g, dg = run_pair(st, fp, dials, 1.0 / 24.0)
    ok &= compare("dense-block overlapping smoke", c, dc, g, dg)
    if not (c["smoke"] > 0).any():
        ok = False
        print("  dense block: no smoke was deposited")

    # (j) burn-through forcer: a low-HP flammable wall tile burns through. dt is
    #     SMALL (1/24) so the fire survives the logistic pass (a large dt starves
    #     it to 0 there before P5 can burn the wall); wall_hp is below one tick's
    #     depletion so P5 crosses 0 and collects the destroyed cell.
    st = _blank(5, 5)
    st["flammable"][2, 2] = True
    st["is_wall"][2, 2] = True
    st["fire"][2, 2] = _quantize(0.9)
    st["temperature"][2, 2] = _quantize(700.0)
    st["wall_hp"][2, 2] = _quantize(0.005)             # < one tick's wall_damage
    st["n_total"][:] = _quantize(1.0)
    st["n_o2"][:] = _quantize(0.9)
    st = _contig(st)
    c, dc, g, dg = run_pair(st, fp, dials, 1.0 / 24.0)
    ok &= compare("burn-through", c, dc, g, dg)
    if (2, 2) not in [tuple(t) for t in dc]:
        ok = False
        print(f"  burn-through: (2,2) not destroyed (dc={list(dc)})")

    # (k) host max early-exit: all fire below thresh -> fields UNTOUCHED.
    st = _blank(6, 6)
    st["fire"][:] = _quantize(0.0005)                  # < 0.001 thresh
    st["temperature"][:] = _quantize(500.0)
    st["smoke"][:] = _quantize(0.1)
    st["wall_hp"][:] = _quantize(10.0)
    st["flammable"][:] = True
    st["n_total"][:] = _quantize(1.0)
    st["n_o2"][:] = _quantize(0.9)
    st = _contig(st)
    before = {k: st[k].copy() for k in MUT}
    c, dc, g, dg = run_pair(st, fp, dials, 1.0 / 24.0)
    ok &= compare("early-exit", c, dc, g, dg)
    for k in MUT:
        if not np.array_equal(c[k], before[k]) or not np.array_equal(g[k], before[k]):
            ok = False
            print(f"  early-exit: {k} was modified (should be untouched)")
    if len(dc) != 0 or len(dg) != 0:
        ok = False
        print(f"  early-exit: destroyed not empty (cpu={list(dc)} gpu={list(dg)})")

    if ok:
        print(f"  all {n_cfg} fuzz + 11 deterministic forcers bit-identical on "
              f"fire/temperature/smoke/wall_hp + SET-equal destroyed; T_FLAME_MAX "
              f"sat-clamp AND headroom hard-cap paths proven, burn-through + "
              f"early-exit + overlapping-smoke covered.")
    return ok


def part2_trajectory() -> bool:
    print("PART 2 — O2-rich room ignition trajectory (130 ticks, CPU vs GPU "
          "lockstep):")
    H = W = 20
    # Boosted plume so temperature visibly climbs and the ceiling engages.
    fp, dials = make_params(fire_pressure_gain=0.8, temp_gain_scale=600.0,
                            wall_damage=1.2)
    dt = 1.0 / 24.0

    st = _blank(H, W)
    # Air room full of O2: n_total is a FIXED ambient baseline (1.0, never
    # touched by the external driver below — the "O2 converts to inert gas,
    # total roughly conserved" shape); n_o2 = 0.30, i.e. X = 0.30 > o2_frac_amb
    # (0.21) -> the room starts genuinely O2-RICH (full o2f) under the
    # continuous-O2 law. A central block of flammable-wall fire tiles.
    st["n_total"][:] = _quantize(1.0)
    st["n_o2"][:] = _quantize(0.30)
    st["flammable"][7:13, 7:13] = True
    st["is_wall"][7:13, 7:13] = True
    st["fire"][7:13, 7:13] = _quantize(0.45)
    st["temperature"][7:13, 7:13] = _quantize(700.0)   # hot -> grows
    st["wall_hp"][7:13, 7:13] = _quantize(50.0)        # sustained fuel
    # A cell seeded AT the ceiling: sat = 0 there every tick, so the plume
    # deposit is gated to zero and T holds EXACTLY at T_FLAME_MAX (the
    # self-limiter's active engagement — proven bit-identically). The plume
    # can never CLIMB to the ceiling (sat -> 0 asymptotes it below), so a
    # seeded holder is how the ceiling value is reached-and-held in-trajectory.
    st["temperature"][7, 7] = _quantize(2000.0)
    # A plain 700-seed climber cell whose T we track to prove plume heating.
    CLIMB = (11, 11)
    st["temperature"][CLIMB] = _quantize(700.0)
    # A ring of low-HP "fuse" tiles that burn through early.
    for (yy, xx) in [(6, 9), (13, 10), (9, 6), (10, 13)]:
        st["flammable"][yy, xx] = True
        st["is_wall"][yy, xx] = True
        st["fire"][yy, xx] = _quantize(0.9)
        st["temperature"][yy, xx] = _quantize(700.0)
        st["wall_hp"][yy, xx] = _quantize(0.05)
    st = _contig(st)

    # Two lockstep copies (one CPU-backend, one GPU-backend).
    cpu = {k: st[k].copy() for k in st}
    gpu = {k: st[k].copy() for k in st}
    sim = bp.FireSimulation()
    sim.params = fp
    t_ceiling = int(_quantize(2000.0)[()])
    init_fire_total = int(cpu["fire"].sum())
    climb_seed = int(cpu["temperature"][CLIMB])
    climb_max = climb_seed

    n_ticks = 130
    bad = 0
    destroyed_total = 0
    max_T = -(1 << 62)
    for tick in range(n_ticks):
        d_cpu = sim.step(cpu["fire"], cpu["atmosphere"], cpu["n_o2"],
                         cpu["n_total"], cpu["smoke"],
                         cpu["wall_hp"], cpu["temperature"], cpu["wind_x"],
                         cpu["wind_y"], cpu["is_wall"], cpu["is_vacuum"],
                         cpu["flammable"], dt)
        d_gpu = bp.cuda_fire_step(
            gpu["fire"], gpu["atmosphere"], gpu["n_o2"], gpu["n_total"],
            gpu["smoke"],
            gpu["wall_hp"], gpu["temperature"], gpu["wind_x"], gpu["wind_y"],
            gpu["is_wall"], gpu["is_vacuum"], gpu["flammable"], dt, **dials)

        if not compare(f"tick {tick}", cpu, list(d_cpu), gpu, list(d_gpu)):
            bad += 1
            if bad >= 8:
                print("  aborting after 8 divergences")
                break

        destroyed_total += len(list(d_cpu))
        tick_max_T = int(cpu["temperature"].max())
        max_T = max(max_T, tick_max_T)
        climb_max = max(climb_max, int(cpu["temperature"][CLIMB]))
        if tick_max_T > t_ceiling:
            bad += 1
            print(f"  tick {tick}: temperature {tick_max_T} EXCEEDED the "
                  f"T_FLAME_MAX ceiling {t_ceiling} — self-limiter failed")

        # External driver (identical to BOTH copies): deplete the O2 NUMERATOR
        # in the room (n_total held fixed) to force the fire to self-starve as
        # its mole fraction falls, and rebuild destroyed walls into air.
        for s in (cpu, gpu):
            s["n_o2"][:] = np.maximum(s["n_o2"] - _quantize(0.004), 0)

    ok = (bad == 0)
    final_fire_total = int(cpu["fire"].sum())
    # The scenario must actually exercise the pass hard.
    if destroyed_total == 0:
        ok = False
        print("  scenario too tame: no wall burned through")
    if climb_max <= climb_seed:
        ok = False
        print(f"  scenario too tame: plume never raised the climber cell "
              f"(T {climb_seed} -> max {climb_max})")
    if max_T != t_ceiling:
        ok = False
        print(f"  T_FLAME_MAX ceiling not held-at-ceiling: peak T {max_T} "
              f"(ceiling {t_ceiling}) — the sat-clamp gate did not hold it")
    if final_fire_total >= init_fire_total:
        ok = False
        print(f"  scenario too tame: fire did not self-starve "
              f"(fire sum {init_fire_total} -> {final_fire_total})")
    if ok:
        print(f"  {n_ticks} ticks bit-identical (fire/temperature/smoke/wall_hp "
              f"+ SET-equal destroyed, CPU==GPU every tick); walls destroyed="
              f"{destroyed_total}, ceiling held at {max_T}=={t_ceiling}, climber "
              f"T {climb_seed}->{climb_max} (plume heating), fire self-starved "
              f"{init_fire_total}->{final_fire_total}.")
    return ok


def part3_golden() -> bool:
    print("PART 3 — CUDA build's CPU path vs the committed golden:")
    from field_ab_harness import capture_trajectory
    from field_digest import trajectory_digest

    GOLDEN = "98d3dd7eaf3d574d6e562513cd95f3b5ac077b7c69b1d0b024db931261735473"
    base = capture_trajectory(n_steps=30)
    dig = trajectory_digest(base)
    if dig != GOLDEN:
        print(f"  GOLDEN MISMATCH: {dig[:16]}... != {GOLDEN[:16]}...")
        return False
    print(f"  CUDA build CPU path reproduces the golden ({dig[:12]}...).")
    return True


def main() -> int:
    if not getattr(bp, "HAS_CUDA", False) or not bp.cuda_available():
        print("P68_RESULT: FAIL (no CUDA build / device)")
        return 1
    print("device:", bp.cuda_device_info())
    p1 = part1_isolated()
    p2 = part2_trajectory()
    p3 = part3_golden()
    if p1 and p2 and p3:
        print("P68_RESULT: PASS")
        return 0
    print("P68_RESULT: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
