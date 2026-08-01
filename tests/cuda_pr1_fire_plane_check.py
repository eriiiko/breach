"""P-R4 gates (d) + (g), CUDA half — the RADIATION-LAW lockstep + cost bench.

REWRITTEN AT P-R4 (documented re-anchor). This module was the P-R1 transition
witness: it pinned ``heat`` byte-identical across four casts (old-CPU /
old-CUDA / new-CPU / new-CUDA) with the PAINTER as its oracle. P-R4 retires the
painter (docs/radiation_raycaster_extinction_ruling_2026-07-31.md A1), so that
oracle no longer exists; the patch's bounded re-anchor authority turns this into
the witness for the law that replaced it.

What is gated here (runs inside the GPU subprocess, tests/cuda_harness.py):

  PART 1 — gate (d), scenario A: a 600-emitter synthetic firestorm. ``rad_net``
           bit-identical CPU vs CUDA at TOLERANCE ZERO, and the plane conserving
           EXACTLY (sum == 0) on both backends. The device scatter is a PLAIN
           signed ``atomicAdd`` and the CPU a plain signed add — integer
           addition is associative and commutative, so the accumulation is
           order-free and the two agree count for count however the warps
           interleave. (A SATURATING signed add would NOT be order-free; that
           is why radiation has its own plane and its own contract — A1.7.)
  PART 2 — gate (d), scenario B: the EQUAL-T PAIR. Two adjacent emitters at the
           same temperature exchange EXACTLY 0 on both backends — gate (e)(i),
           on the GPU too.
  PART 3 — gate (d), scenario C: a hot/cold pair with the FLUX LIMITER ENGAGED
           (a T_MAX_PHYS-scale gap, where the raw T⁴ net exceeds the per-ray
           budget and the clamp actually bites). Bit-identical, and the scenario
           asserts the clamp really fired rather than passing vacuously.
  PART 4 — gate (g): the COST. 600 emitters, one batched device cast, timed
           against a 3.0 ms budget (2x the 1.5 ms S8c painter baseline). The CPU
           reference cast is timed alongside for the record.

Prints ``PR4_RADIATION_RESULT: PASS``/``FAIL`` and exits 0/1.
"""
from __future__ import annotations

import sys
import time

import numpy as np

# Import the CUDA build FIRST (cuda_harness bootstrap put cpp/build_cuda on
# the path) so `breach_physics` is the GPU build.
import breach_physics as bp

FP_ONE = 1 << 16
RAD_SCALE = 1.0e-5
DIALS = dict(fire_ray_count=8, range_base=2.0, range_per_i=3.0,
             intensity_base=0.3, intensity_per_i=0.7, color=(1.0, 0.6, 0.2))
COST_BUDGET_MS = 3.0        # gate (g): 2x the 1.5 ms S8c painter baseline


def _make_raycaster():
    rc = bp.Raycaster()
    rc.light_cull = 0.01
    rc.heat_cull = 0.01
    rc.smoke_absorb_scale = 1.4
    rc.rad_scale = RAD_SCALE
    rc.T_emit_gate = 180.0
    # P-F1a / v7 rule 4: emission rays reach RADIATION_RANGE (>= the grid
    # diagonal). Pinned so both backends march the SAME long rays.
    rc.radiation_range = 320.0
    rc.bake_emissive_table()
    return rc


class Scene:
    """The plane set the radiation cast reads, as raw numpy arrays."""

    def __init__(self, h, w, n_gases=2, rng=None):
        self.h, self.w = h, w
        self.fire = np.zeros((h, w), np.int32)
        self.temperature = np.zeros((h, w), np.int32)
        self.heat_atten = np.zeros((h, w), np.float32)
        self.heat_inv_shift = np.zeros((h, w), np.int32)
        self.thermal_solid = np.zeros((h, w), bool)
        self.light_atten = np.zeros((h, w, 3), np.float32)
        if rng is None:
            self.gas = np.zeros((n_gases, h, w), np.float32)
            self.gas_abs = np.zeros((n_gases, 3), np.float32)
            self.gas_sca = np.zeros((n_gases, 3), np.float32)
        else:
            self.gas = (rng.random((n_gases, h, w)).astype(np.float32) * 0.6)
            self.gas_abs = np.array([[1.0, 1.0, 1.0], [0.9, 0.2, 0.9]], np.float32)
            self.gas_sca = np.array([[0.6, 0.6, 0.6], [0.1, 0.7, 0.1]], np.float32)

    def solid(self, y, x, atten=0.5, his=3, T_game=0.0):
        self.heat_atten[y, x] = atten
        self.heat_inv_shift[y, x] = his
        self.thermal_solid[y, x] = True
        self.temperature[y, x] = int(round(T_game * FP_ONE))

    def burn(self, y, x, I=0.21):
        v = float(I) * float(FP_ONE)
        self.fire[y, x] = int(np.floor(v + 0.5))   # round-half-away-from-zero

    def _bufs(self):
        return (np.zeros((self.h, self.w, 3), np.float32),
                np.zeros((self.h, self.w), np.float32),
                np.zeros((self.h, self.w), np.float32),
                np.zeros((self.h, self.w), np.int32),
                np.zeros((self.h, self.w), np.int32),
                np.zeros((self.h, self.w), np.int32))

    def cast_cpu(self, rc, tick=0, light=False):
        rgb, dx, dy, rad, amb, flux = self._bufs()
        if not light:
            rgb = dx = dy = None      # P-F1a: skip the visible-light cast
        rc.cast_from_fire_plane(
            self.fire, DIALS["fire_ray_count"], DIALS["range_base"],
            DIALS["range_per_i"], DIALS["intensity_base"],
            DIALS["intensity_per_i"], DIALS["color"], rgb, dx, dy,
            self.gas, self.gas_abs, self.gas_sca, self.light_atten,
            self.heat_atten, self.temperature, self.heat_inv_shift,
            self.thermal_solid, rad, amb, flux, tick)
        return rad, amb, flux

    def cast_cuda(self, rc, tick=0, light=False):
        rgb, dx, dy, rad, amb, flux = self._bufs()
        if not light:
            rgb = dx = dy = None      # P-F1a: skip the visible-light cast
        bp.cuda_raycaster_cast_from_fire_plane(
            rc, self.fire, DIALS["fire_ray_count"], DIALS["range_base"],
            DIALS["range_per_i"], DIALS["intensity_base"],
            DIALS["intensity_per_i"], DIALS["color"], rgb, dx, dy,
            self.gas, self.gas_abs, self.gas_sca, self.light_atten,
            self.heat_atten, self.temperature, self.heat_inv_shift,
            self.thermal_solid, rad, amb, flux, tick)
        return rad, amb, flux


def _compare(tag, cpu_t, cuda_t) -> bool:
    """All THREE P-F1a planes at tol 0, plus the rule-4 ledger identity.

    RE-ANCHORED AT P-F1a. The old comparison checked `rad_net.sum() == 0`, which
    was only ever true because emission rays expired in mid-air and were charged
    to nobody (the corridor leak v7 rule 4 closes). The conserving quantity is
    now `rad_net.sum() + rad_amb.sum()`, and the SKY LEDGER is a third synced
    plane that must itself be bit-identical across backends -- it is written by
    a plain int32 atomicAdd on the device and a plain signed add on the CPU,
    order-free on both.
    """
    cpu, cpu_amb, cpu_flux = cpu_t
    cuda, cuda_amb, cuda_flux = cuda_t
    for name, a, b in (("rad_net", cpu, cuda),
                       ("rad_amb", cpu_amb, cuda_amb),
                       ("rad_flux", cpu_flux, cuda_flux)):
        if not np.array_equal(a, b):
            mism = int(np.count_nonzero(a != b))
            idx = int(np.argmax(a != b))
            ry, rx = divmod(idx, a.shape[1])
            print(f"  {tag}: {name} CPU != CUDA — {mism} MISMATCH (first @ "
                  f"({ry},{rx}): cpu={a.flat[idx]} cuda={b.flat[idx]})")
            return False
    b_cpu = int(cpu.sum()) + int(cpu_amb.sum())
    b_cuda = int(cuda.sum()) + int(cuda_amb.sum())
    if b_cpu != 0 or b_cuda != 0:
        print(f"  {tag}: LEDGER IDENTITY BROKEN — rad_net+rad_amb cpu={b_cpu} "
              f"cuda={b_cuda}")
        return False
    if int(cpu_flux.min()) < 0:
        print(f"  {tag}: the positive-only flux sensor went NEGATIVE")
        return False
    if int(cpu_amb.min()) < 0:
        print(f"  {tag}: the sky ledger went NEGATIVE")
        return False
    nz = int(np.count_nonzero(cpu))
    nzf = int(np.count_nonzero(cpu_flux))
    print(f"  {tag}: rad_net + rad_amb + rad_flux bit-identical at tol 0 "
          f"({nz} cells exchanged, |max|={int(np.abs(cpu).max())}; sky="
          f"{int(cpu_amb.sum())}; {nzf} air cells lit, flux max="
          f"{int(cpu_flux.max())}); rad_net+rad_amb sums to EXACTLY 0.")
    return True


# ---------------------------------------------------------------------------
def _firestorm(h=128, w=128, nfire=600, seed=20260801):
    rng = np.random.default_rng(seed)
    sc = Scene(h, w, rng=rng)
    cells = set()
    while len(cells) < nfire:
        cells.add((int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1))))
    for (y, x) in cells:
        sc.solid(y, x, atten=0.5, his=3, T_game=float(rng.uniform(200.0, 900.0)))
        sc.burn(y, x, I=float(rng.uniform(0.3, 1.0)))
    # Absorbers: a scatter plus two bands, so occlusion, the survival cull and
    # the source-tile self-occlusion skip are all exercised.
    for _ in range(500):
        y, x = int(rng.integers(0, h)), int(rng.integers(0, w))
        if (y, x) in cells:
            continue
        sc.solid(y, x, atten=float(rng.uniform(0.1, 1.0)),
                 his=int(rng.integers(2, 6)), T_game=0.0)
    sc.heat_atten[h // 2, :] = 0.7
    sc.thermal_solid[h // 2, :] = True
    sc.heat_inv_shift[h // 2, :] = 3
    sc.heat_atten[:, w // 2] = 1.0
    sc.thermal_solid[:, w // 2] = True
    sc.heat_inv_shift[:, w // 2] = 3
    return sc


def part1_firestorm(rc) -> bool:
    print("PART 1 — gate (d) scenario A: 600-emitter firestorm, rad_net tol 0:")
    sc = _firestorm()
    cpu, cuda = sc.cast_cpu(rc), sc.cast_cuda(rc)
    if int(np.count_nonzero(cpu[0])) == 0:
        print("  SCENARIO WEAK: nothing exchanged — vacuous gate")
        return False
    return _compare("firestorm", cpu, cuda)


def _equal_T_lattice(T_game, atten=0.5, his=3):
    """A SEALED, isothermal, AIR-SEPARATED emitter lattice.

    Gate (ix) names this as one of its four required scenarios, and it is the
    sharpest one for the backends: rule 2's half-weight mutual branch fires on
    every pair, the emitter mask is consulted at every marched cell, and the
    correct answer is EXACTLY ZERO everywhere -- so any CPU/GPU disagreement in
    the mask, the half fold or the halved clamp shows up as a nonzero, not as a
    small numeric drift that could be argued away.
    """
    sc = Scene(21, 21)
    for x in range(21):
        sc.solid(0, x, atten=1.0, his=his, T_game=T_game)
        sc.solid(20, x, atten=1.0, his=his, T_game=T_game)
    for y in range(21):
        sc.solid(y, 0, atten=1.0, his=his, T_game=T_game)
        sc.solid(y, 20, atten=1.0, his=his, T_game=T_game)
    for y in range(3, 18, 2):
        for x in range(3, 18, 2):
            sc.solid(y, x, atten=atten, his=his, T_game=T_game)
    return sc


def part2_equal_pair(rc) -> bool:
    print("PART 2 — gate (ix) scenario B: the EQUAL-T lattice (must be exactly 0):")
    # RE-ANCHORED AT P-F1a: the pair is AIR-SEPARATED (adjacent is a CONTACT
    # face under rule 3 and would make the check vacuous), and grown to a full
    # sealed lattice so rule 2 fires on many pairs at once.
    ok = True
    for T in (180.0, 443.0, 1000.0, 2500.0):
        sc = _equal_T_lattice(T)
        cpu, cuda = sc.cast_cpu(rc), sc.cast_cuda(rc)
        inner_cpu = int(np.abs(cpu[0][1:-1, 1:-1]).sum())
        inner_cuda = int(np.abs(cuda[0][1:-1, 1:-1]).sum())
        if inner_cpu != 0 or inner_cuda != 0:
            print(f"  equal-T lattice @ {T}: RECIPROCITY BROKEN (cpu "
                  f"{inner_cpu}, cuda {inner_cuda})")
            ok = False
            continue
        ok = _compare(f"equal-T lattice @ {T:.0f} game", cpu, cuda) and ok
    return ok


def part5_live_burn(rc) -> bool:
    """Gate (ix)'s fourth scenario: a LIVE EVOLVING BURN, not a frozen snapshot.

    The frozen scenes above each probe one configuration. This one steps the
    state forward the way the engine does -- folding each tick's rad_net back
    into `temperature` (shr_round0 by heat_inv_shift, the Pass-1 fold) and
    advancing the D4 tick -- so the two backends have to agree not just on one
    cast but on a TRAJECTORY, where any single-count divergence compounds into
    a different emitter set on the next tick.
    """
    print("PART 5 — gate (ix) scenario D: a LIVE EVOLVING BURN (trajectory):")
    h = w = 48

    def fresh():
        # A FRESH seeded generator per call: the two scenes must start
        # bit-identical or the trajectory comparison is meaningless.
        rng = np.random.default_rng(31337)
        sc = Scene(h, w)
        cells = set()
        while len(cells) < 40:
            cells.add((int(rng.integers(3, h - 3)), int(rng.integers(3, w - 3))))
        for (y, x) in sorted(cells):
            sc.solid(y, x, atten=0.5, his=3, T_game=600.0)
            sc.burn(y, x, I=0.8)
        for k in range(300):
            y = (k * 7919) % h
            x = (k * 104729) % w
            if sc.thermal_solid[y, x]:
                continue
            sc.solid(y, x, atten=0.5 if k % 2 else 1.0, his=3 - (k % 2),
                     T_game=float(20 * (k % 9)))
        return sc

    a, b = fresh(), fresh()
    if not np.array_equal(a.temperature, b.temperature):
        print("  the two scenes did not start identical — harness bug")
        return False
    ok = True
    for tick in range(12):
        cpu = a.cast_cpu(rc, tick=tick)
        cuda = b.cast_cuda(rc, tick=tick)
        if not _compare(f"live burn tick {tick:>2d}", cpu, cuda):
            return False
        # The Pass-1 fold, per backend, on its OWN plane — so a divergence
        # would propagate rather than be washed out by a shared state.
        for sc, rad in ((a, cpu[0]), (b, cuda[0])):
            fold = np.where(rad >= 0, rad >> sc.heat_inv_shift,
                            -((-rad) >> sc.heat_inv_shift)).astype(np.int32)
            sc.temperature = np.clip(
                sc.temperature.astype(np.int64) + fold, 0,
                16000 * FP_ONE).astype(np.int32)
        if not np.array_equal(a.temperature, b.temperature):
            n = int(np.count_nonzero(a.temperature != b.temperature))
            print(f"  temperature DIVERGED after tick {tick}: {n} cells")
            return False
    print(f"  12 ticks, temperature bit-identical throughout "
          f"(max T = {int(a.temperature.max()) / FP_ONE:.1f} game)")
    return ok


def part3_limiter(rc) -> bool:
    print("PART 3 — gate (d) scenario C: hot/cold pair, FLUX LIMITER ENGAGED:")
    # WHERE THE LIMITER ACTUALLY BINDS (measured, and worth writing down):
    # the raw per-ray net is a_s*a_r*w*|dE|, and E° SATURATES at INT32_MAX above
    # T_game ~ 1768 at the shipped rad_scale — so the raw net is capped at
    # w*2^31, while the budget (|dT| << his) >> 4 keeps growing LINEARLY in the
    # gap. The clamp therefore bites in a BAND: above E°'s saturation knee and
    # below the gap where the linear budget overtakes the capped net. For an
    # opaque pair (atten 1.0) at wood/furniture thermal mass (his = 3) that band
    # is roughly T_game 1768 .. 8190. 3000 sits inside it. (In the 400-500 game
    # OPERATING band the limiter is INERT by ~25x — exactly as ruling A1.6 says
    # it should be: a rail against T³ steepening, not part of the felt law.)
    # RE-ANCHORED AT P-F1a. Two changes:
    #  * the absorber is a COLUMN one air tile away, not the first ring — rule 3
    #    makes a face-adjacent absorber non-participating, so the old scene
    #    would now pass VACUOUSLY (nothing transferred at all);
    #  * the "where the limiter binds" note above is void where it argues from
    #    E° saturating at INT32_MAX above T_game ~ 1768. That ceiling is GONE
    #    (L2-B3 widened the table to int64). The limiter now binds wherever the
    #    T⁴ term outruns the LINEAR budget (|dT| << his) >> RAD_LIM_SHIFT — at
    #    the shipped dials, above ~1300 game against a cold partner. 3000 game
    #    is comfortably inside that, and the check below proves it rather than
    #    assuming it.
    sc = Scene(21, 21)
    sc.solid(10, 10, atten=1.0, his=3, T_game=3000.0)
    for y in range(21):
        sc.solid(y, 12, atten=1.0, his=3, T_game=0.0)   # AIR-SEPARATED column
    sc.burn(10, 10, I=1.0)
    cpu, cuda = sc.cast_cpu(rc), sc.cast_cuda(rc)
    dT = abs(int(sc.temperature[10, 10]))
    budget = (dT << 3) >> 4                     # per-END budget, RAD_LIM_SHIFT 4
    tab = rc.emissive_table()
    raw_1ray = int(0.125 * float(int(tab[min(dT >> 18, tab.shape[0] - 1)])))
    print(f"  raw per-ray net ~{raw_1ray}, per-pair budget {budget} -> limiter "
          f"{'ENGAGED' if raw_1ray > budget else 'inert'}")
    if raw_1ray <= budget:
        print("  (the limiter did not engage — this gate would be vacuous)")
        return False
    moved = int(cpu[0][:, 12].sum())
    print(f"  absorber column gained {moved} counts; per-ray clamp ceiling "
          f"{budget}, 8-ray ceiling {8 * budget}")
    if moved > 8 * budget:
        print("  LIMITER FAILED: transfer exceeded the 8-ray budget ceiling")
        return False
    return _compare("hot/cold + limiter", cpu, cuda)


def _median_ms(fn, warm=3, reps=11):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return ts[len(ts) // 2]


def part4_cost(rc) -> bool:
    """Gate (viii), the CUDA half — and an ATTRIBUTION, not just a verdict.

    RE-ANCHORED AT P-F1a in methodology (median of 11 timed reps after 3
    warm-ups, replacing a mean of 5 after 1). The old form was noisy enough to
    swing ~0.5 ms run to run, which is the same size as the effect being gated.

    The number is also BROKEN DOWN, because the headline is otherwise
    misleading: v7 rule 4 lengthened emission rays from ~5 tiles to >= the grid
    diagonal, a ~60x increase, and the natural assumption is that this dominates
    the cost. Measured, it does not — the pure-radiation fast path makes a long
    ray almost free (an air cell is one DDA step, two float multiplies and a
    bounds test), and what actually dominates is the per-call H2D/D2H of the
    plane set plus the SHORT visible-light cast's gas optics. Reporting the
    split is what lets the next patch optimise the right thing.
    """
    print(f"PART 4 — gate (viii): 600-emitter batched cast, budget "
          f"{COST_BUDGET_MS:.1f} ms:")
    sc = _firestorm()
    long_ms = _median_ms(lambda: sc.cast_cuda(rc))

    # ATTRIBUTION 1: the same cast with the OLD ~5-tile reach. The difference
    # is exactly what v7 rule 4's long rays cost.
    rc_short = _make_raycaster()
    rc_short.radiation_range = 5.0
    short_ms = _median_ms(lambda: sc.cast_cuda(rc_short))

    # ATTRIBUTION 2: a single-emitter scene — the fixed per-call transfer and
    # allocation overhead, with essentially no marching at all.
    tiny = Scene(sc.h, sc.w)
    tiny.solid(sc.h // 2, sc.w // 2, atten=0.5, his=3, T_game=600.0)
    tiny.burn(sc.h // 2, sc.w // 2, I=1.0)
    fixed_ms = _median_ms(lambda: tiny.cast_cuda(rc))

    # ATTRIBUTION 3: the SECOND DEVICE ROUND-TRIP. Asking for the visible-light
    # buffers turns one upload/launch/download into two. This is the dominant
    # cost of v7 rule 4's split — NOT the long rays — which is why the shipped
    # sim path passes None (it discarded those buffers anyway).
    with_light_ms = _median_ms(lambda: sc.cast_cuda(rc, light=True))

    cpu_ms = _median_ms(lambda: sc.cast_cpu(rc), warm=1, reps=5)
    print(f"  CUDA batched cast : {long_ms:8.3f} ms   (budget {COST_BUDGET_MS} ms)")
    print(f"  CPU reference cast: {cpu_ms:8.3f} ms")
    print(f"  ATTRIBUTION: fixed per-call overhead {fixed_ms:.3f} ms; "
          f"same scene at the OLD 5-tile reach {short_ms:.3f} ms; "
          f"asking for light too {with_light_ms:.3f} ms")
    print(f"    -> v7 rule 4's >= grid-diagonal rays cost "
          f"{long_ms - short_ms:+.3f} ms "
          f"({100.0 * (long_ms - short_ms) / max(long_ms, 1e-9):.1f}% of the total)")
    print(f"    -> the SECOND device round-trip (light cast) would cost "
          f"{with_light_ms - long_ms:+.3f} ms — skipped on the shipped path")
    if long_ms > COST_BUDGET_MS:
        print(f"  COST GATE FAIL: {long_ms:.3f} ms > {COST_BUDGET_MS} ms")
        return False
    return True


def main() -> int:
    if not getattr(bp, "HAS_CUDA", False) or not bp.cuda_available():
        print("PR4_RADIATION_RESULT: FAIL (no CUDA build / device)")
        return 1
    print("device:", bp.cuda_device_info())
    rc = _make_raycaster()
    p1 = part1_firestorm(rc)
    p2 = part2_equal_pair(rc)
    p3 = part3_limiter(rc)
    p4 = part4_cost(rc)
    p5 = part5_live_burn(rc)
    ok = p1 and p2 and p3 and p4 and p5
    print("PR4_RADIATION_RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
