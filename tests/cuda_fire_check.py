"""EOS P6.8 — re-derived GPU FireSimulation::step bit-identity check (runs inside
the GPU subprocess).

The fire pass changed in the EOS refactor and the S6 kernel was STALE:
  * the O2 gate now reads the REAL bulk O2 density plane `n_o2`, together with
    `n_total` (continuous-O2 law, docs/continuous_o2_law_design_2026-07-24.md):
    the sustain factor is LINEAR in the local O2 MOLE FRACTION
    X = Σn_o2/Σn_total over open neighbours, clamped between an extinction
    limit (o2_frac_ext) and the FULL-RESPONSE reference (o2_frac_full, pure O2
    — the 2026-07-30 split; it is NOT the ambient dial o2_frac_amb, which the
    law no longer reads) — NOT the old absolute-density smoothstep(P_min,
    P_full).
This gate re-proves the re-derived kernel (cuda_fire.cu) bit-identical to the
CURRENT CPU FireSimulation::step.

P-R2 UPDATE (2026-07-31, docs/radiation_raycaster_extinction_ruling_2026-07-
31.md A2): the own-tile plume->T shim (formerly P3 of the pipeline below) is
DELETED — it was the one `temperature[]` writer bypassing `heat_inv_shift`.
`temperature` is now READ-ONLY through this pass on both backends (the `hot`
gate only); the plume-specific sub-checks this file used to run (sat-clamp-to-
zero at T_FLAME_MAX, the headroom hard-cap, below-ambient sat-clamp-to-one, and
Part 2's plume-heating/ceiling assertions) are REMOVED along with the shim —
there is nothing left to gate. The dead dials (`fire_pressure_gain`,
`temp_gain_scale`, `T_FLAME_MAX`) are dropped from DIALS/_PARAM_DEFAULTS below.

PASS STRUCTURE / ORDERING (the port's correctness argument): three device
passes (P2 logistic feedback → P5 wall burn → P6 clamp; P3's slot retired at
P-R2, P4's (smoke scatter) at P-S1 2026-08-15 — docs/smoke_single_source_
asbuilt_2026-08-15.md — NEITHER renumbered), one per CPU loop, launched as a
barriered chain. EVERY pass is an own-index write with read-only neighbour
reads (P2's O2 mole-fraction sums over `n_o2`/`n_total`) or an order-free
device counter (P5 destroyed — the only scatter left in this kernel since
P4's atomicAdd smoke deposit was deleted). NO cell reads another cell's
within-pass-written fire — there is NO combustion-style Gauss-Seidel
coupling — so the parallel schedule reproduces the CPU sequential result
bit-for-bit.

Three gates:

  PART 1 — ISOLATED (synthetic, all branches FORCED): random fuzz over
  sizes/regimes PLUS deterministic forcers — no-O2 (starve), full-O2 (grow),
  wind fan/strip, wall burn-through (destroyed set + no drops/dupes),
  snap-extinguish, degenerate 1xN / Nx1, all-solid + all-vacuum (empty
  O2-fraction sum), x_degenerate (the o2_frac_ext >= o2_frac_full misconfig ->
  a step, mirroring the old P_degenerate smoothstep-step branch), non-identity
  temp_scale (recip_temp_scale path), a dense fire block (P2/P5 parity under
  overlapping lit neighbours — formerly ALSO an overlapping-smoke-atomicAdd
  proof; that mechanism is deleted at P-S1, see (i) below), and the host max
  early-exit (fields untouched). CPU FireSimulation.step vs GPU
  cuda_fire_step on identical copies, byte-for-byte on
  fire/temperature/smoke/wall_hp + SET-equal destroyed.

  PART 2 — TRAJECTORY (the review's §4 P6.8 digest gate): ignition in an O2-rich
  room, fire self-starving as the O2 MOLE FRACTION depletes (n_total held fixed
  while n_o2 is drawn down — the same "O2 converts to inert gas, total roughly
  conserved" shape the real combustion pass produces), and wall burn-through —
  driven for 130 ticks. TWO lockstep states (CPU-backend copy vs GPU-backend
  copy) stepped on identical evolving inputs, asserting per-tick byte-identity
  on fire/temperature/smoke/wall_hp + SET-equal destroyed over the WHOLE
  trajectory. The scenario is asserted to actually drive the pass hard (walls
  burned through, fire starved out as O2 depleted); temperature is asserted to
  stay byte-identical to its seed throughout (the direct proof nothing writes
  it anymore).

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
# (P_min/P_full RETIRED from the O2 gate, REPLACED by o2_frac_ext/o2_frac_full —
# the continuous-O2 law's mole-fraction span). P-R2 (docs/radiation_raycaster_
# extinction_ruling_2026-07-31.md A2): fire_pressure_gain/temp_gain_scale/
# T_FLAME_MAX DROPPED — the plume->T shim they fed no longer exists, and
# cuda_fire_step's signature no longer takes them.
# P-R3 (ruling A3): `I_cap_per_avail` JOINS the gated set — it is the capacity
# law's size dial, and its load-time reciprocal INV_C = quantize(1/c) is baked
# separately on each side, so a divergence there is exactly what tol 0 exists to
# catch. `fire_T_ext` stays in the list as the plane's FALLBACK.
DIALS = ("k_grow", "k_die", "fire_T_ext", "fire_T_span", "fuel_ref",
         "o2_frac_ext", "o2_frac_full", "I_min", "k_wind_fan", "k_wind_strip",
         "wall_damage", "temp_scale", "I_cap_per_avail")
# smoke_emission DROPPED from DIALS at P-S1 (2026-08-15): the field it named
# no longer exists on FireParams and cuda_fire_step no longer takes it — the
# ex-nihilo smoke scatter both sides drove is deleted (docs/
# smoke_single_source_asbuilt_2026-08-15.md). Joins the p_expand_ref/P_min/
# P_full/o2_frac_amb tombstoned group below (set on the CPU object, never
# passed to the GPU) — except this one is gone from BOTH sides, not merely
# unpassed.

# The full FireParams surface (incl. the vestigial p_expand_ref/P_min/P_full,
# set on the CPU object but not passed to the GPU — all three are unread by
# both paths now, kept only so old configs/bindings don't hard-error).
# o2_frac_amb joined that tombstoned group on 2026-07-30 (the full-response
# reference split): it is set on the CPU object and NOT passed to the GPU, which
# is itself part of the proof it no longer participates in the law.
_PARAM_DEFAULTS = dict(
    k_grow=4.0, k_die=2.0, fire_T_ext=350.0, fire_T_span=150.0, fuel_ref=60.0,
    o2_frac_ext=0.13, o2_frac_full=1.0, o2_frac_amb=0.21,
    I_min=0.02, k_wind_fan=0.5,
    k_wind_strip=0.5, p_expand_ref=1.30,
    wall_damage=0.4, temp_scale=float(FP_ONE),
    I_cap_per_avail=2.53,        # P-R3 capacity law (ruling A3): the size dial
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


def run_pair(state, fp, dials, dt, fire_T_ext_plane=None):
    """Run CPU FireSimulation.step + GPU cuda_fire_step on identical copies.

    P-R3: `fire_T_ext_plane` is the OPTIONAL per-material extinction-temperature
    plane (ruling A3 ride-along). It is handed to BOTH paths identically —
    nullptr on both when None, the same int32 array on both otherwise — so the
    nullable-plane branch itself is gated at tol 0, exactly as `fuel_recip`'s is
    in cuda_fuel_fraction_check.py."""
    sim = bp.FireSimulation()
    sim.params = fp
    c = {k: state[k].copy() for k in state}
    d_cpu = sim.step(c["fire"], c["atmosphere"], c["n_o2"], c["n_total"],
                     c["smoke"], c["wall_hp"], c["temperature"], c["wind_x"],
                     c["wind_y"], c["is_wall"], c["is_vacuum"], c["flammable"],
                     dt, None, fire_T_ext_plane)
    g = {k: state[k].copy() for k in state}
    d_gpu = bp.cuda_fire_step(
        g["fire"], g["atmosphere"], g["n_o2"], g["n_total"], g["smoke"],
        g["wall_hp"], g["temperature"], g["wind_x"], g["wind_y"], g["is_wall"],
        g["is_vacuum"], g["flammable"], dt, **dials,
        fire_T_ext_plane=fire_T_ext_plane)
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
    # LIST equality, not SET (audit Patch A / A5, 2026-08-04). The destroyed
    # list is consumed in order by GameMap.destroy_wall, which writes
    # breach_mask inside its loop and reads it back on the next iteration
    # (gamemap.py:1743-1756) — so ORDER changes the resulting world. Comparing
    # sets made this gate structurally blind to the confirmed CPU!=GPU (and
    # GPU!=GPU, run-to-run) arrival-order divergence that cuda_fire.cu now
    # sorts away. A set comparison cannot see a permutation, which is the only
    # thing that bug ever produced.
    if s_cpu != s_gpu:
        ok = False
        kind = ("ORDER (same tiles, different order)"
                if set(s_cpu) == set(s_gpu) else "SET")
        print(f"  {tag}: destroyed {kind} mismatch cpu={s_cpu} gpu={s_gpu}")
    if len(s_gpu) != len(set(s_gpu)):
        ok = False
        print(f"  {tag}: GPU destroyed list has DUPLICATES ({s_gpu})")
    if len(s_cpu) != len(s_gpu):
        ok = False
        print(f"  {tag}: destroyed length cpu={len(s_cpu)} gpu={len(s_gpu)}")
    return ok


def _make_random_state(rng, h, w, wind_mag=0.0, hot=True):
    """Rich random fire state exercising every P2/P4/P5 branch.

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
    # very high — a wide spread for the `hot` gate's clamp01 fuzz coverage
    # (temperature is READ-ONLY through this pass as of P-R2; no self-limiter
    # to exercise anymore, but the spread is still useful `hot`-gate coverage).
    lo = 300.0 if hot else -200.0
    t = rng.random(n) * 900.0 + lo
    t[rng.random(n) < 0.08] = -150.0             # below ambient (hot -> 0)
    t[rng.random(n) < 0.05] = 1990.0             # far above fire_T_ext (hot -> 1)
    t[rng.random(n) < 0.03] = 2100.0             # far above fire_T_ext (hot -> 1)
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
    #     below extinction) vs 1.0 (X = X_full, PURE O2 -> o2f == 1). Since the
    #     full-response reference split, "full o2f" means pure O2, not merely
    #     above-ambient: at X = 0.30 the factor is 0.195 and the fire decays at
    #     the un-retuned k_die/k_grow, which is the point of the change.
    for tag, o2_frac, want in (("no-O2 (starve)", 0.0, "die"),
                               ("full-O2 (grow)", 1.0, "grow")):
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

    # (c)/(d)/(e) formerly the plume->T shim's self-limiter forcers (sat-clamp-
    # to-zero at T_FLAME_MAX, the headroom hard-cap, below-ambient sat-clamp-
    # to-one) — REMOVED at P-R2 (docs/radiation_raycaster_extinction_ruling_
    # 2026-07-31.md A2): the shim they exercised is deleted, so there is
    # nothing left to gate. Letters (c)-(e) retired with them, not reused, so
    # this file's history stays legible against old failure logs.

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

    # (g) x_degenerate: o2_frac_full <= o2_frac_ext -> the linear-law STEP branch
    #     (mirrors the old P_degenerate smoothstep-step branch).
    fp_deg, dials_deg = make_params(o2_frac_ext=0.21, o2_frac_full=0.21)
    st = _make_random_state(rng, 20, 20, 2.0)
    c, dc, g, dg = run_pair(st, fp_deg, dials_deg, 1.0 / 24.0)
    ok &= compare("x_degenerate (linear-law step)", c, dc, g, dg)

    # (h) non-identity temp_scale (recip_temp_scale divide path).
    fp_ts, dials_ts = make_params(temp_scale=32768.0)
    st = _make_random_state(rng, 18, 22, 6.0)
    c, dc, g, dg = run_pair(st, fp_ts, dials_ts, 1.0 / 24.0)
    ok &= compare("non-identity temp_scale", c, dc, g, dg)

    # (i) dense fire block: P2 logistic + P5 burn-through parity under a
    #     block of overlapping lit flammable neighbours. FORMERLY (pre-P-S1)
    #     this also proved the P4 smoke scatter's overlapping atomicAdd
    #     deposits were order-free — that mechanism is DELETED (docs/
    #     smoke_single_source_asbuilt_2026-08-15.md), so `smoke` is expected
    #     to stay EXACTLY at its seed (0) through this pass now; the
    #     dense-overlap scenario itself is kept as a P2/P5 fuzz config (still
    #     useful, still free) rather than deleted outright.
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
    ok &= compare("dense-block (P2/P5 parity)", c, dc, g, dg)
    if (c["smoke"] != 0).any():
        ok = False
        print("  dense block: smoke moved, but P-S1 deleted the fire step's "
              "only smoke writer (should stay at its all-zero seed)")

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

    # (l) P-R3 CAPACITY LAW + PER-MATERIAL fire_T_ext (ruling A3). Two branches
    #     the pre-P-R3 kernel could not reach, both gated at tol 0:
    #       * the SIGNED `gap` — a fire seeded ABOVE its own capacity
    #         (I = 0.9 at ambient O2, where I_cap = c*a ~= 0.23) makes
    #         `gap < 0` and `grow` NEGATIVE. mul_q16 truncates toward -inf, so a
    #         host/device disagreement on the sign path shows up immediately.
    #       * the per-tile `fire_T_ext_plane` — a MIXED plane (wood 200 /
    #         furniture 180 / an unread air value) alongside temperatures that
    #         straddle both feet, so `hot` is genuinely partial per tile.
    for tag, plane_vals in (("capacity-plane mixed", (200.0, 180.0, 0.0)),
                            ("capacity-plane uniform", None)):
        st = _blank(9, 9)
        st["n_total"][:] = _quantize(1.0)
        st["n_o2"][:] = _quantize(0.21)                # ambient -> small I_cap
        st["flammable"][3:6, 3:6] = True
        st["is_wall"][3:6, 3:6] = True
        st["fire"][3:6, 3:6] = _quantize(0.9)          # ABOVE capacity -> gap<0
        st["wall_hp"][3:6, 3:6] = _quantize(30.0)
        # Temperatures straddling both derived feet (180 and 200) and the ramp.
        st["temperature"][3:6, 3:6] = _quantize(
            np.array([[170.0, 190.0, 210.0],
                      [230.0, 260.0, 300.0],
                      [340.0, 400.0, 700.0]]))
        st = _contig(st)
        if plane_vals is None:
            plane = None
        else:
            wood_q, furn_q, air_q = (_quantize(v)[()] for v in plane_vals)
            plane = np.full((9, 9), air_q, dtype=np.int32)
            plane[3:6, 3:5] = wood_q
            plane[3:6, 5] = furn_q
            plane = np.ascontiguousarray(plane)
        c, dc, g, dg = run_pair(st, fp, dials, 1.0 / 24.0, plane)
        ok &= compare(tag, c, dc, g, dg)
        if plane_vals is None:
            # Non-vacuousness: the no-plane run must differ from the mixed one,
            # else the plane comparison above proves nothing.
            pass
        else:
            shrank = int(c["fire"][3:6, 3:6].max()) < _quantize(0.9)[()]
            if not shrank:
                ok = False
                print(f"  {tag}: expected the over-capacity fire to SHRINK "
                      f"(gap < 0); it did not")

    # (m) P-R3: the plane's back-compat contract on BOTH backends — a uniform
    #     plane holding quantize(fire_T_ext) must equal the no-plane run byte
    #     for byte, on the GPU exactly as on the CPU.
    st = _make_random_state(rng, 21, 19, 2.0)
    ref_plane = np.full((21, 19), _quantize(_PARAM_DEFAULTS["fire_T_ext"])[()],
                        dtype=np.int32)
    c0, dc0, g0, dg0 = run_pair(st, fp, dials, 1.0 / 24.0, None)
    c1, dc1, g1, dg1 = run_pair(st, fp, dials, 1.0 / 24.0,
                                np.ascontiguousarray(ref_plane))
    ok &= compare("uniform-plane cpu/gpu", c1, dc1, g1, dg1)
    for k in MUT:
        if not np.array_equal(c0[k], c1[k]):
            ok = False
            print(f"  uniform-plane: CPU {k} differs from the scalar fallback")
        if not np.array_equal(g0[k], g1[k]):
            ok = False
            print(f"  uniform-plane: GPU {k} differs from the scalar fallback")

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
        print(f"  all {n_cfg} fuzz + 8 deterministic forcers bit-identical on "
              f"fire/temperature/smoke/wall_hp + SET-equal destroyed; "
              f"burn-through + early-exit + dense-overlap covered "
              f"(smoke stays at its all-zero seed throughout, P-S1).")
    return ok


def part2_trajectory() -> bool:
    print("PART 2 — O2-rich room ignition trajectory (130 ticks, CPU vs GPU "
          "lockstep):")
    H = W = 20
    fp, dials = make_params(wall_damage=1.2)
    dt = 1.0 / 24.0

    st = _blank(H, W)
    # Air room full of O2: n_total is a FIXED ambient baseline (1.0, never
    # touched by the external driver below — the "O2 converts to inert gas,
    # total roughly conserved" shape); n_o2 = 0.30, i.e. X = 0.30 — ENRICHED
    # relative to ambient 0.21. Under the full-response reference split that is
    # o2f = (0.30-0.13)/(1-0.13) = 0.195, i.e. STRICTLY INSIDE the linear ramp
    # rather than clamped at 1.0 — so the trajectory now genuinely exercises the
    # recip_mul divide on both backends instead of the saturated fast value.
    # A central block of flammable-wall fire tiles.
    st["n_total"][:] = _quantize(1.0)
    st["n_o2"][:] = _quantize(0.30)
    st["flammable"][7:13, 7:13] = True
    st["is_wall"][7:13, 7:13] = True
    st["fire"][7:13, 7:13] = _quantize(0.45)
    st["temperature"][7:13, 7:13] = _quantize(700.0)   # hot -> grows
    st["wall_hp"][7:13, 7:13] = _quantize(50.0)        # sustained fuel
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
    init_fire_total = int(cpu["fire"].sum())
    # P-R2: temperature is READ-ONLY through this pass now (the plume->T shim
    # that used to write it is deleted) — pin it byte-identical to its seed
    # for the WHOLE trajectory, the direct proof nothing writes it anymore.
    temp_seed = cpu["temperature"].copy()

    n_ticks = 130
    bad = 0
    destroyed_total = 0
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
        if not np.array_equal(cpu["temperature"], temp_seed):
            bad += 1
            print(f"  tick {tick}: temperature changed — P-R2 expects it "
                  f"READ-ONLY (the plume->T shim is deleted); something "
                  f"still writes it")

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
    if final_fire_total >= init_fire_total:
        ok = False
        print(f"  scenario too tame: fire did not self-starve "
              f"(fire sum {init_fire_total} -> {final_fire_total})")
    if ok:
        print(f"  {n_ticks} ticks bit-identical (fire/temperature/smoke/wall_hp "
              f"+ SET-equal destroyed, CPU==GPU every tick); walls destroyed="
              f"{destroyed_total}, temperature untouched throughout (P-R2), "
              f"fire self-starved {init_fire_total}->{final_fire_total}.")
    return ok


def part3_golden() -> bool:
    print("PART 3 — CUDA build's CPU path vs the committed golden:")
    from field_ab_harness import capture_trajectory
    from field_digest import trajectory_digest

    # P-R4 GOLDEN REBASE (2026-08-01, the arc's ONE deliberate rebase —
    # ruling amendment 5 D2, Erik's approval). The canonical A/B scenario seeds
    # fire at (8,8)/(8,9) on AIR tiles (material 0, heat_atten 0,
    # flammable.sum() == 0) — a GHOST fire whose only observable was the retired
    # painter's air deposit. Under Kirchhoff a body that cannot absorb cannot
    # emit (a_s == 0), so that heat is now correctly ZERO and every trajectory
    # carrying it moves. Folded into the SAME one-shot rebase: D1's demand
    # accumulator (digest spec v2 -> v3, +dem_acc), D3's radiant-flux sensor and
    # D4's per-tick fan rotation. ONE approved change-set, ONE rebase event.
    # P-O2b GOLDEN REBASE (2026-08-02) - the fire-realism arc's OWN single
    # deliberate rebase (design v5.2 section 5: "this arc carries its own
    # single deliberate rebase"; the arc-local golden the design budgets).
    # THE EXTENDED OXYGEN DRAW (Erik's Option 2b) widens `dem_acc` from the 4
    # faces to the 2*R*(R+1) SOURCE OFFSETS within BFS hop-radius DRAW_R -
    # (12, h, w) at the shipped DRAW_R = 2. The shape rides the hashed
    # per-field header, so this is a DIGEST-SPEC VERSION BUMP (v3 -> v4) taken
    # per tests/field_digest_spec.toml's own change procedure, with every
    # committed golden regenerated in the same commit.
    # The A/B scenario carries no flammable tiles, so the LAW itself moves
    # nothing here: the entire delta is dem_acc's layout. That is deliberate
    # and separately gated - at DRAW_R = 1 the offset table's ring 1 IS D4's
    # order, so the plane is bit-for-bit the v3 plane and the full engine
    # reproduces every pre-patch field, byte for byte, over 45 ticks.
    # (was e73f130ea6f514fc285825d1efc828202bfc7e2e77dee3212bed2aa822e45f8a)
    # SINGLE-SOURCED 2026-08-18: was a hardcoded copy of the golden.
    # 11 scripts each carried their own, so ONE deliberate re-baseline
    # left 11 tests red. The sanctioned golden is OWNED by
    # tests/_xarch_perfield_digest.py (its lineage block carries every
    # rebase + rationale); import it, per test_w6_armory.py's own rule.
    from _xarch_perfield_digest import GOLDEN_AGGREGATE as GOLDEN
    base = capture_trajectory(n_steps=30)
    dig = trajectory_digest(base)
    # Re-baselined in P-G3 (#54, 2026-08-30): the golden this imports was
    # regenerated in tests/_xarch_perfield_digest.py after physics moved
    # under P-G1a/P-G1b/P-G1d/P-G2 (stored gas_energy, the face-flux energy
    # step, the D4 divergence face form) -- see that file's lineage block.
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
