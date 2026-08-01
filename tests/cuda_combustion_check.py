"""EOS P6.9b — combustion GPU bit-identity check (runs inside the GPU subprocess).

Proves cuda_combustion.cu (the two-gather reformulation) byte-for-byte identical
to the CPU CombustionSolver::step over (1) a battery of synthetic edge configs +
random fuzz, (2) a hard 120-tick fire trajectory, and (3) the committed
default-scenario golden (the CUDA build's CPU path unmoved).

Continuous-O2 law (docs/continuous_o2_law_design_2026-07-24.md §2.3): the
per-claimant O2 DEMAND is now demand_k = burn_cap*I_k*o2f_j — `fire` (I_k) is
READ again (a choked/flameless source with fire[i]==0 draws nothing), and
o2f_j is LINEAR in the air cell's O2 MOLE FRACTION X_j = O2[j]/(O2[j]+N2[j])
between o2_frac_ext (extinction) and o2_frac_full (the FULL-RESPONSE reference).
The scene builders below seed `fire` at every source (default I=1.0 exactly) and
keep the local mole fraction comfortably above the span's top, so o2f_j clamps to
exactly 1.0 and demand_k reduces to the OLD uniform burn_cap_q bit-for-bit — the
fixtures' hand-tuned O2 values (the zero/nonzero-remainder split points, the
D<=O2j uncontested threshold, etc.) still hold unchanged under the new law.

FULL-RESPONSE REFERENCE SPLIT (2026-07-30): the span's upper end used to be the
ambient dial o2_frac_amb (0.21); it is now the separate o2_frac_full, shipped at
1.0 (pure O2), so ambient air no longer saturates o2f. This file's DIALS pin
o2_frac_full BACK to 0.21 so the whole hand-tuned fixture battery keeps its
o2f==1 arithmetic; the shipped default (1.0), where o2f sits strictly inside the
linear ramp and the recip_mul divide is genuinely exercised, is covered by the
extra PART 1 (k2) sweep and PART 2's second trajectory — both of which also
assert the two spans produce DIFFERENT results (the non-vacuousness control).

WHY bit-identical (docs/eos_p6_9_combustion_design.md §4): both gather passes are
per-cell functions of frozen inputs (Tsnap snapshot, pass-entry O2 read only at
the OWN cell, pre-payment wall_hp, masks) and the gas/temperature planes each
cell alone writes. The proportional split is plain int64 `/`,`%` — a single
portable answer, identical on CPU and CUDA. The GPU realizes the design's
`snapshot(T) -> Pass A -> face buffers -> Pass B` barrier chain as a launch-
barriered kernel chain. So the digest gate is a formality the design guarantees.

COMPARED per step (tol 0): the three mutated gas planes (O2 / inert_N2 /
smoke), `temperature`, `wall_hp`, AND the two PER-CELL rail counters
(heat_floor_hits, t_max_phys_hits — design §3: their ABSOLUTE value is not
asserted, only CPU==GPU equality).

  PART 1 — ISOLATED edge configs + fuzz: uncontested (D<=O2, full demand),
  heavily-contested 4-claimant cells with ZERO remainder (O2 divisible by 4) AND
  NON-ZERO remainder (the tiebreak + full-drain path), no-O2 (starve), char-out
  at FUEL_FLOOR (total-then-floor-once pins wall_hp at 1 LSB), the n_floor_heat
  rail forced, the T_MAX_PHYS rail forced, degenerate 1xN / Nx1, all-solid +
  all-vacuum, plus random fuzz over sizes/regimes. Each: CPU CombustionSolver.step
  vs GPU cuda_combustion_step on identical copies.

  PART 2 — TRAJECTORY: an O2-rich room igniting a flammable-wood cluster, spreading
  through contested multi-source air cells (exercising the proportional split +
  full-drain + aggregate deposit every tick), self-starving as O2 depletes, and
  charring walls toward FUEL_FLOOR — driven 120 ticks. TWO lockstep states (CPU vs
  GPU) stepped on identical evolving inputs; per-tick byte-identity on all five
  fields + both rail counters. Asserted to actually drive the pass hard (contested
  cells occurred, O2 drained fully, heat deposited, walls charred down).

  PART 3 — the CUDA build's CPU path still reproduces the committed golden.

Prints ``P69_RESULT: PASS``/``FAIL`` and exits 0/1.
"""
from __future__ import annotations

import sys

import numpy as np

# Import the CUDA build FIRST so it is the cached `breach_physics`.
import breach_physics as bp

from simulation.gases import O2, INERT_N2, SMOKE, N_GASES

FP_ONE = 65536
FUEL_FLOOR = 1

# Combustion dials (one source of truth for BOTH the CPU solver + the GPU call).
# Continuous-O2 law (docs/continuous_o2_law_design_2026-07-24.md §2.3):
# o2_frac_ext/o2_frac_full are the SAME mole-fraction span dial the fire
# logistic uses (one law, shared constants). o2_frac_full is pinned to 0.21 here
# — the PRE-SPLIT span — so the hand-tuned fixtures below keep o2f_j == 1.0
# exactly; the shipped 1.0 default is swept separately (see the module docstring).
# o2_frac_amb is carried only to be SET on the CPU solver: the law no longer
# reads it, and it is not passed to the GPU at all.
DIALS = dict(burn_rate=1.0, o2_thresh_burn=0.03, H_fuel=4.0, soot_yield=0.3,
             fuel_per_o2=0.7, o2_frac_ext=0.13, o2_frac_full=0.21,
             o2_frac_amb=0.21, T_MAX_PHYS=16000.0)
C_V = 1.0
N_FLOOR_HEAT = 0.05

# The material ignition threshold used throughout (any positive Q16.16 works —
# CPU and GPU read the SAME array; the exact value is not load-bearing).
IGN_Q = int(round(500.0 * FP_ONE))


def _quantize(x):
    x = np.asarray(x, dtype=np.float64)
    return np.int32(np.where(x >= 0, np.floor(x * FP_ONE + 0.5),
                             np.ceil(x * FP_ONE - 0.5)))


def _mk_solver(**over):
    c = bp.CombustionSolver()
    d = dict(DIALS)
    d.update(over)
    c.burn_rate = d["burn_rate"]
    c.o2_thresh_burn = d["o2_thresh_burn"]
    c.H_fuel = d["H_fuel"]
    c.soot_yield = d["soot_yield"]
    c.fuel_per_o2 = d["fuel_per_o2"]
    c.o2_frac_ext = d["o2_frac_ext"]
    c.o2_frac_full = d["o2_frac_full"]
    c.o2_frac_amb = d["o2_frac_amb"]   # set but UNREAD by the law (split, 07-30)
    c.T_MAX_PHYS = d["T_MAX_PHYS"]
    return c, d


def _contig(state):
    return {k: (np.ascontiguousarray(v) if isinstance(v, np.ndarray) else v)
            for k, v in state.items()}


def run_pair(state, dt, dials_over=None, c_v=C_V, n_floor_heat=N_FLOOR_HEAT):
    """Run CPU CombustionSolver.step + GPU cuda_combustion_step on identical
    copies of `state`. Returns (cpu_dict, cpu_rails, gpu_dict, gpu_rails)."""
    dials_over = dials_over or {}
    comb, d = _mk_solver(**dials_over)

    c = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in state.items()}
    # CPU: fresh solver has zeroed rail counters -> post-step members == this
    # step's per-cell counts.
    comb.step(c["gas"], O2, INERT_N2, SMOKE, c["temperature"], c["wall_hp"],
              c["fire"], c["flammable"], c["solid"], c["is_vacuum"],
              c["ignition_temp_q16"], dt, c_v, n_floor_heat)
    cpu_rails = (int(comb.heat_floor_hits), int(comb.t_max_phys_hits))

    g = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in state.items()}
    hf, tm = bp.cuda_combustion_step(
        g["gas"], O2, INERT_N2, SMOKE, g["temperature"], g["wall_hp"], g["fire"],
        g["flammable"], g["solid"], g["is_vacuum"], g["ignition_temp_q16"],
        dt, c_v, n_floor_heat,
        d["burn_rate"], d["o2_thresh_burn"], d["H_fuel"], d["soot_yield"],
        d["fuel_per_o2"], d["o2_frac_ext"], d["o2_frac_full"], d["T_MAX_PHYS"])
    gpu_rails = (int(hf), int(tm))
    return c, cpu_rails, g, gpu_rails


def compare(tag, c, cpu_rails, g, gpu_rails, check_rails=True):
    ok = True
    # The three mutated planes are inside `gas`; compare the WHOLE gas array
    # (untouched planes are trivially equal) + temperature + wall_hp.
    for k in ("gas", "temperature", "wall_hp"):
        if not np.array_equal(c[k], g[k]):
            ok = False
            diff = c[k] != g[k]
            n_mis = int(np.count_nonzero(diff))
            idx = int(np.argmax(diff))
            print(f"  {tag}: {k} {n_mis} MISMATCH (first flat @ {idx}: "
                  f"cpu={c[k].flat[idx]} gpu={g[k].flat[idx]})")
    if check_rails and cpu_rails != gpu_rails:
        ok = False
        print(f"  {tag}: rail-counter MISMATCH cpu={cpu_rails} gpu={gpu_rails}")
    return ok


# ---------------------------------------------------------------------------
# Scene builders
# ---------------------------------------------------------------------------
def _blank(h, w, ambient_n2=0.79):
    gas = np.zeros((N_GASES, h, w), dtype=np.int32)
    solid = np.zeros((h, w), dtype=bool)
    is_vacuum = np.zeros((h, w), dtype=bool)
    flammable = np.zeros((h, w), dtype=bool)
    wall_hp = np.zeros((h, w), dtype=np.int32)
    fire = np.zeros((h, w), dtype=np.int32)
    ign = np.zeros((h, w), dtype=np.int32)
    temperature = np.zeros((h, w), dtype=np.int32)
    gas[INERT_N2][:] = _quantize(ambient_n2)   # ambient in every cell (solid too;
                                               # the pass only reads air-cell N2)
    return dict(gas=gas, solid=solid, is_vacuum=is_vacuum, flammable=flammable,
                wall_hp=wall_hp, fire=fire, ignition_temp_q16=ign,
                temperature=temperature)


def _add_source(st, y, x, hp=60.0, temp_q=None, fire_i=1.0):
    """A hot, fuelled, LIT flammable-wood wall tile (a burn source).

    Continuous-O2 law (design §2.3): demand_k = burn_cap*I_k*o2f_j — a source
    with fire[i] == 0 draws NOTHING (a choked/flameless ember), so every
    fixture that expects to actually burn must seed `fire` at the source.
    `fire_i` defaults to FULL intensity (I=1.0 exactly): with o2f_j also
    clamped to exactly 1.0 (the fixtures below keep the local O2 mole
    fraction comfortably above the span top), demand_k == burn_cap_q exactly
    (mul_q16(x, FP_ONE) == x, no rounding) — i.e. the OLD uniform-demand
    arithmetic these fixtures were built against is reproduced bit-for-bit.
    """
    st["solid"][y, x] = True
    st["flammable"][y, x] = True
    st["wall_hp"][y, x] = int(round(hp * FP_ONE))
    st["ignition_temp_q16"][y, x] = IGN_Q
    st["temperature"][y, x] = IGN_Q * 2 if temp_q is None else temp_q
    st["fire"][y, x] = _quantize(fire_i)[()]


def _plus_scene(center_o2_raw, outer_o2=0.21, hp=60.0, h=9, w=9):
    """The isotropy PLUS — 4 flammable sources around one contested air cell."""
    st = _blank(h, w)
    cy, cx = h // 2, w // 2
    for dy, dx, oy_d, ox_d in ((-1, 0, -2, 0), (1, 0, 2, 0),
                               (0, -1, 0, -2), (0, 1, 0, 2)):
        _add_source(st, cy + dy, cx + dx, hp=hp)
        st["gas"][O2][cy + oy_d, cx + ox_d] = _quantize(outer_o2)
    st["gas"][O2][cy, cx] = int(center_o2_raw)
    return _contig(st), (cy, cx)


def _random_state(rng, h, w):
    """Rich random combustion state exercising every branch."""
    st = _blank(h, w)
    n = h * w
    solid = (rng.random(n) < 0.35).reshape(h, w)
    flammable = solid & (rng.random(n) < 0.6).reshape(h, w)
    is_vacuum = (~solid) & (rng.random(n) < 0.10).reshape(h, w)
    st["solid"][:] = solid
    st["flammable"][:] = flammable
    st["is_vacuum"][:] = is_vacuum

    # flammable sources: fuel + ignition + temperature (some above, some below).
    hp = rng.random(n).reshape(h, w) * 80.0
    hp[rng.random(n).reshape(h, w) < 0.15] = 2.0 / FP_ONE   # near FUEL_FLOOR
    st["wall_hp"][flammable] = _quantize(hp)[flammable]
    st["ignition_temp_q16"][flammable] = IGN_Q
    t = rng.random(n).reshape(h, w) * 1400.0 - 100.0        # some below 500 ign
    t[rng.random(n).reshape(h, w) < 0.1] = 15900.0          # near T_MAX_PHYS
    st["temperature"][flammable] = _quantize(t)[flammable]

    # fire (continuous-O2 law §2.3): I_k, the per-claimant demand magnitude.
    # Full range incl. 0 (a choked/flameless ember that draws nothing).
    fire_i = rng.random(n).reshape(h, w)
    fire_i[rng.random(n).reshape(h, w) < 0.2] = 0.0         # unlit/choked
    st["fire"][flammable] = _quantize(fire_i)[flammable]

    # air cells: O2 across the full range incl. 0 and > D; ambient N2 already
    # set — together with O2 this sweeps the o2f_j mole-fraction span (0 to
    # ~0.6, straddling o2_frac_ext=0.13 and the span top) across every branch.
    air = (~solid) & (~is_vacuum)
    o2 = rng.random(n).reshape(h, w) * 1.2
    o2[rng.random(n).reshape(h, w) < 0.15] = 0.0            # starved
    st["gas"][O2][air] = _quantize(o2)[air]
    # a little pre-existing soot so SOOT += is exercised from nonzero
    st["gas"][SMOKE][air] = _quantize(rng.random(n).reshape(h, w) * 0.05)[air]
    return _contig(st)


# ---------------------------------------------------------------------------
def part1_isolated() -> bool:
    print("PART 1 — isolated GPU vs CPU (edge configs + fuzz):")
    ok = True
    rng = np.random.default_rng(20260711)
    dt = 0.25   # burn_cap = quantize(1.0*0.25) = 16384; D(4 claimants) = 65536

    # (a) uncontested: one source, plenty of O2 (D = burn_cap <= O2j).
    st = _blank(5, 5)
    _add_source(st, 2, 2, hp=60.0)
    st["gas"][O2][2, 3] = _quantize(1.0)          # 65536 >> burn_cap 16384
    st = _contig(st)
    c, cr, g, gr = run_pair(st, dt)
    ok &= compare("uncontested (full demand)", c, cr, g, gr)
    if int(c["gas"][O2][2, 3]) != int(_quantize(1.0)[()]) - 16384:
        ok = False
        print("  uncontested: expected exactly burn_cap drawn")

    # (b) contested 4-claimant, ZERO remainder (isotropy bit-exact fixture).
    st, ctr = _plus_scene(center_o2_raw=40000)
    c, cr, g, gr = run_pair(st, dt)
    ok &= compare("contested-4 zero-remainder", c, cr, g, gr)
    if int(c["gas"][O2][ctr]) != 0:
        ok = False
        print(f"  contested zero-rem: centre did not fully drain "
              f"(O2={int(c['gas'][O2][ctr])}) — full-drain path not exercised")

    # (c) contested 4-claimant, NON-ZERO remainder (tiebreak + full-drain).
    st, ctr = _plus_scene(center_o2_raw=40002)
    c, cr, g, gr = run_pair(st, dt)
    ok &= compare("contested-4 nonzero-remainder", c, cr, g, gr)
    if int(c["gas"][O2][ctr]) != 0:
        ok = False
        print("  contested nonzero-rem: centre did not fully drain")

    # (d) contested with 2 sources (aggregate heat deposit, delta delta).
    st = _blank(5, 5)
    _add_source(st, 1, 2, hp=60.0)
    _add_source(st, 3, 2, hp=60.0)
    st["gas"][O2][2, 2] = 20000                    # < D=2*16384=32768 -> contested
    st = _contig(st)
    c, cr, g, gr = run_pair(st, dt)
    ok &= compare("contested-2 aggregate-deposit", c, cr, g, gr)
    if int(c["temperature"][2, 2]) <= 0:
        ok = False
        print("  aggregate-deposit: no heat deposited at the 2-source cell")

    # (e) no-O2 (starve): air cell below o2_thresh -> nothing burns anywhere.
    st = _blank(5, 5)
    _add_source(st, 2, 2, hp=60.0)
    st["gas"][O2][2, 3] = _quantize(0.01)          # < o2_thresh 0.03
    st = _contig(st)
    before = st["wall_hp"].copy()
    c, cr, g, gr = run_pair(st, dt)
    ok &= compare("no-O2 (starve)", c, cr, g, gr)
    if not np.array_equal(c["wall_hp"], before):
        ok = False
        print("  no-O2: wall_hp changed (should be untouched)")

    # (f) char-out at FUEL_FLOOR: a source just above the floor, heavy fuel_per_o2,
    #     drained by a big burn -> wall_hp pinned at FUEL_FLOOR (never below).
    st = _blank(5, 5)
    _add_source(st, 2, 2, hp=0.0)
    st["wall_hp"][2, 2] = FUEL_FLOOR + 3           # just above the floor
    st["gas"][O2][2, 3] = _quantize(1.0)
    st = _contig(st)
    c, cr, g, gr = run_pair(st, dt, dials_over=dict(fuel_per_o2=5.0))
    ok &= compare("char-out at FUEL_FLOOR", c, cr, g, gr)
    if int(c["wall_hp"][2, 2]) != FUEL_FLOOR:
        ok = False
        print(f"  char-out: wall_hp={int(c['wall_hp'][2,2])} != FUEL_FLOOR "
              f"(the total-then-floor-once pin did not engage)")

    # (g) n_floor_heat rail forced: a thin-gas air cell (tiny N2) so post-burn
    #     N_total < n_floor -> heat_floor_hits fires (CPU==GPU counter).
    st = _blank(5, 5, ambient_n2=0.0)              # no ambient N2 anywhere
    _add_source(st, 2, 2, hp=60.0)
    st["gas"][O2][2, 3] = _quantize(0.04)          # burns, tiny residual N_total
    st = _contig(st)
    c, cr, g, gr = run_pair(st, dt)
    ok &= compare("n_floor_heat rail", c, cr, g, gr)
    if cr[0] == 0:
        ok = False
        print("  n_floor rail: heat_floor_hits was 0 (rail not exercised)")

    # (h) T_MAX_PHYS rail forced: a low ceiling + a hot big burn -> clamp fires.
    st = _blank(6, 6)
    _add_source(st, 2, 2, hp=60.0)
    st["gas"][O2][2, 3] = _quantize(1.2)
    st["temperature"][2, 3] = _quantize(90.0)      # already near the low ceiling
    st = _contig(st)
    c, cr, g, gr = run_pair(st, dt, dials_over=dict(T_MAX_PHYS=100.0, H_fuel=4000.0))
    ok &= compare("T_MAX_PHYS rail", c, cr, g, gr)
    if cr[1] == 0:
        ok = False
        print("  T_MAX rail: t_max_phys_hits was 0 (rail not exercised)")
    if int(c["temperature"][2, 3]) != int(_quantize(100.0)[()]):
        ok = False
        print("  T_MAX rail: temperature not pinned at the ceiling")

    # (i) degenerate 1xN and Nx1.
    for (h, w) in ((1, 12), (12, 1)):
        st = _blank(h, w)
        if w >= 3:
            _add_source(st, 0, 1, hp=60.0)
            st["gas"][O2][0, 2] = _quantize(0.5)
        else:
            _add_source(st, 1, 0, hp=60.0)
            st["gas"][O2][2, 0] = _quantize(0.5)
        st = _contig(st)
        c, cr, g, gr = run_pair(st, dt)
        ok &= compare(f"degenerate {h}x{w}", c, cr, g, gr)

    # (j) all-solid + all-vacuum: no air cell burns.
    for tag, key in (("all-solid", "solid"), ("all-vacuum", "is_vacuum")):
        st = _blank(6, 6)
        st[key][:] = True
        st["flammable"][:] = True
        st["wall_hp"][:] = _quantize(30.0)
        st["ignition_temp_q16"][:] = IGN_Q
        st["temperature"][:] = IGN_Q * 2
        st["gas"][O2][:] = _quantize(0.8)
        before = st["wall_hp"].copy()
        st = _contig(st)
        c, cr, g, gr = run_pair(st, dt)
        ok &= compare(tag, c, cr, g, gr)
        if not np.array_equal(c["wall_hp"], before):
            ok = False
            print(f"  {tag}: wall_hp changed (no air cell should burn)")

    # (k) random fuzz over sizes/regimes/dt.
    n_cfg = 0
    for (h, w) in ((16, 16), (24, 32), (31, 17), (40, 40), (8, 8)):
        for dtt in (0.25, 1.0 / 24.0, 0.6):
            n_cfg += 1
            st = _random_state(rng, h, w)
            c, cr, g, gr = run_pair(st, dtt)
            ok &= compare(f"fuzz {h}x{w} dt={dtt:.3f}", c, cr, g, gr)

    # (k2) THE SHIPPED DEFAULT SPAN (full-response reference split, 2026-07-30):
    #      re-run the same fuzz states with o2_frac_full = 1.0, where o2f_j sits
    #      strictly INSIDE the linear ramp (ambient 0.21 -> 0.092) instead of
    #      clamping to 1.0. This is the span the game actually ships, and it is
    #      the one that exercises the per-cell recip_mul divide on both backends.
    #      NON-VACUOUSNESS CONTROL: the default-span result must DIFFER from the
    #      pinned-span (0.21) result on the same state — otherwise the dial is
    #      inert and "CPU == GPU" would prove nothing about the new law.
    n_cfg2 = 0
    n_moved = 0
    for (h, w) in ((16, 16), (24, 32), (31, 17), (40, 40), (8, 8)):
        for dtt in (0.25, 1.0 / 24.0, 0.6):
            n_cfg2 += 1
            st = _random_state(rng, h, w)
            c0, _, _, _ = run_pair(st, dtt)                       # pinned 0.21 span
            c, cr, g, gr = run_pair(st, dtt,
                                    dials_over=dict(o2_frac_full=1.0))
            ok &= compare(f"fuzz(X_full=1.0) {h}x{w} dt={dtt:.3f}", c, cr, g, gr)
            if not np.array_equal(c0["gas"], c["gas"]):
                n_moved += 1
    if n_moved == 0:
        ok = False
        print("  VACUOUS: o2_frac_full made no difference on any fuzz state — "
              "the new reference dial is not reaching the law")

    if ok:
        print(f"  all edge configs + {n_cfg} fuzz cases (pinned 0.21 span) + "
              f"{n_cfg2} fuzz cases at the SHIPPED span (o2_frac_full=1.0, "
              f"{n_moved} of which moved vs the pinned span) bit-identical on "
              f"gas/temperature/wall_hp + rail counters; full-drain, aggregate "
              f"deposit, char-out, both rails, degenerate + all-solid/vacuum "
              f"covered.")
    return ok


def part2_trajectory(x_full=None, burn_rate=None) -> bool:
    """``x_full`` None -> the DIALS-pinned 0.21 span (the pre-split arithmetic the
    fixtures were built on); 1.0 -> the SHIPPED full-response reference, where
    o2f_j sits strictly inside the linear ramp and the per-cell divide is live.

    ``burn_rate`` compensates the demand scale for the second run: at the shipped
    reference this scene's o2f_j is ~0.48 instead of a clamped 1.0, so the
    per-claimant demand falls ~2x and the CONTENTION path (the whole point of the
    checkerboard) would never fire. Raising burn_rate restores contention without
    touching the law — the lockstep claim is unaffected either way."""
    over = {} if x_full is None else dict(o2_frac_full=float(x_full))
    if burn_rate is not None:
        over["burn_rate"] = float(burn_rate)
    print(f"PART 2 — contested checkerboard fire trajectory (120 ticks, CPU vs "
          f"GPU lockstep, o2_frac_full="
          f"{DIALS['o2_frac_full'] if x_full is None else x_full}):")
    H = W = 18
    dt = 0.25
    # A CHECKERBOARD of flammable-wood sources (parity even) interleaved with air
    # cells (parity odd) across an interior region: every INTERIOR air cell is
    # 4-connected to FOUR sources, so it is contested by up to 4 claimants EVERY
    # tick — the proportional split + full-drain + aggregate-heat-deposit path is
    # exercised across the whole region continuously. O2 is refilled each tick
    # (identically to both copies) to a NON-divisible level (raw 40002 < D=65536)
    # so the remainder tiebreak + full-drain fire every tick for 120 ticks.
    REFILL_O2 = 40002
    LO, HI = 4, 14   # the source/air region [LO, HI)
    CONTESTED = (8, 9)   # 8+9 odd -> air; its 4 neighbours are all even -> sources

    st = _blank(H, W, ambient_n2=0.5)
    for yy in range(LO, HI):
        for xx in range(LO, HI):
            if (yy + xx) % 2 == 0:
                _add_source(st, yy, xx, hp=200.0)   # deep fuel -> lasts 120 ticks
    air_mask = (~st["solid"]) & (~st["is_vacuum"])
    st = _contig(st)

    cpu = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in st.items()}
    gpu = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in st.items()}
    comb, d = _mk_solver(**over)

    init_wall_total = int(cpu["wall_hp"].sum())
    max_T = -(1 << 62)
    contested_drains = 0
    o2_burned_total = 0
    n_ticks = 120
    bad = 0
    for tick in range(n_ticks):
        # External driver, identical to BOTH copies: refill air-cell O2 so the
        # contested burn is sustained. (Combustion is isolated here — no transport
        # replenishes O2 — so we refill to keep the pass hot every tick.)
        for s in (cpu, gpu):
            s["gas"][O2][air_mask] = REFILL_O2
        pre_o2 = int(cpu["gas"][O2][air_mask].sum())

        hf0, tm0 = int(comb.heat_floor_hits), int(comb.t_max_phys_hits)
        comb.step(cpu["gas"], O2, INERT_N2, SMOKE, cpu["temperature"],
                  cpu["wall_hp"], cpu["fire"], cpu["flammable"], cpu["solid"],
                  cpu["is_vacuum"], cpu["ignition_temp_q16"], dt, C_V, N_FLOOR_HEAT)
        cpu_rails = (int(comb.heat_floor_hits) - hf0,
                     int(comb.t_max_phys_hits) - tm0)
        hf, tm = bp.cuda_combustion_step(
            gpu["gas"], O2, INERT_N2, SMOKE, gpu["temperature"],
            gpu["wall_hp"], gpu["fire"], gpu["flammable"], gpu["solid"],
            gpu["is_vacuum"], gpu["ignition_temp_q16"], dt, C_V, N_FLOOR_HEAT,
            d["burn_rate"], d["o2_thresh_burn"], d["H_fuel"], d["soot_yield"],
            d["fuel_per_o2"], d["o2_frac_ext"], d["o2_frac_full"], d["T_MAX_PHYS"])
        gpu_rails = (int(hf), int(tm))

        if not compare(f"tick {tick}", cpu, cpu_rails, gpu, gpu_rails):
            bad += 1
            if bad >= 6:
                print("  aborting after 6 divergences")
                break

        # The known 4-claimant contested air cell must fully DRAIN each tick.
        if int(cpu["gas"][O2][CONTESTED]) == 0:
            contested_drains += 1
        o2_burned_total += pre_o2 - int(cpu["gas"][O2][air_mask].sum())
        max_T = max(max_T, int(cpu["temperature"].max()))

    ok = (bad == 0)
    final_wall_total = int(cpu["wall_hp"].sum())
    if contested_drains == 0:
        ok = False
        print("  scenario too tame: the 4-claimant contested cell never fully "
              "drained (full-drain path never hit)")
    if o2_burned_total <= 0:
        ok = False
        print("  scenario too tame: no O2 consumed across the trajectory")
    if final_wall_total >= init_wall_total:
        ok = False
        print(f"  scenario too tame: no fuel charred ({init_wall_total} -> "
              f"{final_wall_total})")
    if max_T <= 0:
        ok = False
        print("  scenario too tame: no heat deposited (max T <= 0)")
    if ok:
        print(f"  {n_ticks} ticks bit-identical (gas/temperature/wall_hp + both "
              f"rail counters, CPU==GPU every tick); contested cell fully drained "
              f"on {contested_drains}/{n_ticks} ticks, O2 burned={o2_burned_total}, "
              f"wall {init_wall_total}->{final_wall_total} (charred), peak T={max_T}.")
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
    GOLDEN = "28678e9d6210533f63cc701bba8f93194e23df9ebbdfa5f75f5d26681e897040"
    base = capture_trajectory(n_steps=30)
    dig = trajectory_digest(base)
    if dig != GOLDEN:
        print(f"  GOLDEN MISMATCH: {dig[:16]}... != {GOLDEN[:16]}...")
        return False
    print(f"  CUDA build CPU path reproduces the golden ({dig[:12]}...).")
    return True


def main() -> int:
    if not getattr(bp, "HAS_CUDA", False) or not bp.cuda_available():
        print("P69_RESULT: FAIL (no CUDA build / device)")
        return 1
    print("device:", bp.cuda_device_info())
    p1 = part1_isolated()
    # Both spans: the DIALS-pinned pre-split 0.21 (the fixtures' arithmetic) AND
    # the SHIPPED full-response reference 1.0 (full-response reference split).
    p2 = part2_trajectory()
    p2b = part2_trajectory(x_full=1.0, burn_rate=5.0)
    p3 = part3_golden()
    if p1 and p2 and p2b and p3:
        print("P69_RESULT: PASS")
        return 0
    print("P69_RESULT: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
