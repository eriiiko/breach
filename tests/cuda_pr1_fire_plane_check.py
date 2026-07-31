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
                np.zeros((self.h, self.w), np.int32))

    def cast_cpu(self, rc, tick=0):
        rgb, dx, dy, rad, flux = self._bufs()
        rc.cast_from_fire_plane(
            self.fire, DIALS["fire_ray_count"], DIALS["range_base"],
            DIALS["range_per_i"], DIALS["intensity_base"],
            DIALS["intensity_per_i"], DIALS["color"], rgb, dx, dy,
            self.gas, self.gas_abs, self.gas_sca, self.light_atten,
            self.heat_atten, self.temperature, self.heat_inv_shift,
            self.thermal_solid, rad, flux, tick)
        return rad, flux

    def cast_cuda(self, rc, tick=0):
        rgb, dx, dy, rad, flux = self._bufs()
        bp.cuda_raycaster_cast_from_fire_plane(
            rc, self.fire, DIALS["fire_ray_count"], DIALS["range_base"],
            DIALS["range_per_i"], DIALS["intensity_base"],
            DIALS["intensity_per_i"], DIALS["color"], rgb, dx, dy,
            self.gas, self.gas_abs, self.gas_sca, self.light_atten,
            self.heat_atten, self.temperature, self.heat_inv_shift,
            self.thermal_solid, rad, flux, tick)
        return rad, flux


def _compare(tag, cpu_pair, cuda_pair) -> bool:
    """Both P-R4 planes at tol 0: the SIGNED energy ledger (which must also
    conserve exactly) and D3's positive-only damage sensor."""
    cpu, cpu_flux = cpu_pair
    cuda, cuda_flux = cuda_pair
    if not np.array_equal(cpu, cuda):
        mism = int(np.count_nonzero(cpu != cuda))
        idx = int(np.argmax(cpu != cuda))
        ry, rx = divmod(idx, cpu.shape[1])
        print(f"  {tag}: rad_net CPU != CUDA — {mism} MISMATCH (first @ "
              f"({ry},{rx}): cpu={cpu.flat[idx]} cuda={cuda.flat[idx]})")
        return False
    if not np.array_equal(cpu_flux, cuda_flux):
        mism = int(np.count_nonzero(cpu_flux != cuda_flux))
        idx = int(np.argmax(cpu_flux != cuda_flux))
        ry, rx = divmod(idx, cpu.shape[1])
        print(f"  {tag}: rad_flux CPU != CUDA — {mism} MISMATCH (first @ "
              f"({ry},{rx}): cpu={cpu_flux.flat[idx]} cuda={cuda_flux.flat[idx]})")
        return False
    s_cpu, s_cuda = int(cpu.sum()), int(cuda.sum())
    if s_cpu != 0 or s_cuda != 0:
        print(f"  {tag}: CONSERVATION BROKEN — sum cpu={s_cpu} cuda={s_cuda}")
        return False
    if int(cpu_flux.min()) < 0:
        print(f"  {tag}: the positive-only flux sensor went NEGATIVE")
        return False
    nz = int(np.count_nonzero(cpu))
    nzf = int(np.count_nonzero(cpu_flux))
    print(f"  {tag}: rad_net + rad_flux bit-identical at tol 0 ({nz} cells "
          f"exchanged, |max|={int(np.abs(cpu).max())}; {nzf} air cells lit, "
          f"flux max={int(cpu_flux.max())}), rad_net sums to EXACTLY 0.")
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


def part2_equal_pair(rc) -> bool:
    print("PART 2 — gate (d) scenario B: the EQUAL-T pair (must be exactly 0):")
    ok = True
    for T in (180.0, 443.0, 1000.0):
        sc = Scene(21, 21)
        sc.solid(10, 10, T_game=T)
        sc.solid(10, 11, T_game=T)
        sc.burn(10, 10)
        sc.burn(10, 11)
        cpu, cuda = sc.cast_cpu(rc), sc.cast_cuda(rc)
        if int(np.abs(cpu[0]).sum()) != 0 or int(np.abs(cuda[0]).sum()) != 0:
            print(f"  equal-T @ {T}: ANTISYMMETRY BROKEN (cpu "
                  f"{int(np.abs(cpu[0]).sum())}, cuda {int(np.abs(cuda[0]).sum())})")
            ok = False
            continue
        ok = _compare(f"equal-T @ {T:.0f} game", cpu, cuda) and ok
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
    sc = Scene(21, 21)
    sc.solid(10, 10, atten=1.0, his=3, T_game=3000.0)
    sc.solid(10, 11, atten=1.0, his=3, T_game=0.0)
    sc.burn(10, 10, I=1.0)
    cpu, cuda = sc.cast_cpu(rc), sc.cast_cuda(rc)
    dT = abs(int(sc.temperature[10, 10]))
    budget = (dT << 3) >> 4                     # per-END budget, RAD_LIM_SHIFT 4
    raw_1ray = int(0.125 * float(np.iinfo(np.int32).max))   # a_s*a_r*w*E°max
    print(f"  raw per-ray net ~{raw_1ray}, per-pair budget {budget} -> limiter "
          f"{'ENGAGED' if raw_1ray > budget else 'inert'}")
    if raw_1ray <= budget:
        print("  (the limiter did not engage — this gate would be vacuous)")
        return False
    moved = abs(int(cpu[0][10, 11]))
    print(f"  first-ring absorber gained {moved} counts; 2-ray clamp ceiling "
          f"{2 * budget}")
    if moved > 2 * budget:
        print("  LIMITER FAILED: transfer exceeded the 2-ray budget ceiling")
        return False
    return _compare("hot/cold + limiter", cpu, cuda)


def part4_cost(rc) -> bool:
    print(f"PART 4 — gate (g): 600-emitter batched cast, budget "
          f"{COST_BUDGET_MS:.1f} ms (2x the 1.5 ms S8c painter baseline):")
    sc = _firestorm()
    sc.cast_cuda(rc)          # warm-up (context + first alloc)
    n = 5
    t0 = time.perf_counter()
    for _ in range(n):
        sc.cast_cuda(rc)
    cuda_ms = (time.perf_counter() - t0) * 1000.0 / n
    t0 = time.perf_counter()
    for _ in range(3):
        sc.cast_cpu(rc)
    cpu_ms = (time.perf_counter() - t0) * 1000.0 / 3
    print(f"  CUDA batched cast : {cuda_ms:8.3f} ms   (budget {COST_BUDGET_MS} ms)")
    print(f"  CPU reference cast: {cpu_ms:8.3f} ms")
    if cuda_ms > COST_BUDGET_MS:
        print(f"  COST GATE FAIL: {cuda_ms:.3f} ms > {COST_BUDGET_MS} ms")
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
    ok = p1 and p2 and p3 and p4
    print("PR4_RADIATION_RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
