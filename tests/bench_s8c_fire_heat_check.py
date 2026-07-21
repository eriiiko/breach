"""S8c item 1 — the fire-heat PAYOFF bench (per-source cast loop vs ONE batched
device cast). Run under the CUDA build via tests/cuda_harness (like the other
cuda_*_check scripts).

WHAT IT MEASURES. A many-fires scenario (hundreds of burning tiles) built from
the default scenario. It builds the SAME LightSource list cast_fire_heat builds
(row-major over fire>0, the (x*7+y*13)%ray_count phase, jitter=0, the runner's
fire params), then times, BACK-TO-BACK at the same GPU clock/thermal state:

  (A) the OLD per-source path: one bp.cuda_raycaster_cast PER source — each call
      uploads all inputs + the running heat plane, marches, downloads (the
      hundreds-of-round-trips tax that measured ~3 fps, 2026-07-20 B5).
  (B) the NEW batched path: one bp.cuda_raycaster_cast_batch over ALL sources —
      one H2D of the inputs + running heat, one march, one D2H.

Both deposit into the same gmap.heat and are byte-identical on heat (proven tol
0 by cuda_s2_check's batch witness + cuda_s2b's live A/B). This bench only times
them.

ASSERTIONS (design section 5.2). Laptop power/thermal state makes cross-run
ratios unreliable, so:
  * PRIMARY (throttle-robust): the ratio A/B is measured back-to-back in one
    process (same clock state) — assert batched is > 3x faster than the loop.
  * FLOOR (with headroom): batched best-of-N < 100 ms on the many-fires scenario
    (> 3x under the 3-fps ~333 ms pain, ~3x headroom under a 2x throttle so a
    good path never false-FAILs). 30 fps / 33 ms is the TARGET (printed), not the
    gate — asserting < 33 ms would spuriously fail under throttle.
A > 100 ms best-of-N means the round-trip tax is genuinely back (a real
regression); the ratio disambiguates that from a throttle slowdown.
"""
from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, ".")
import breach_physics as bp  # noqa: E402

RATIO_MIN = 3.0        # batched must be > 3x the per-source loop (throttle-robust)
FLOOR_MS = 100.0       # batched best-of-N hard ceiling (headroom under throttle)
TARGET_MS = 33.3       # 30 fps — printed as the aspiration, NOT asserted


def _time_best_of(fn, reps=3):
    """Best (min) wall-clock ms over `reps` runs, each fully synced."""
    best = float("inf")
    for _ in range(reps):
        bp.cuda_device_synchronize() if hasattr(bp, "cuda_device_synchronize") else None
        t0 = time.perf_counter()
        fn()
        # raycaster_cast_directional ends with cudaDeviceSynchronize, so the call
        # returning already means the march + D2H are done — no extra sync needed.
        dt = (time.perf_counter() - t0) * 1e3
        best = min(best, dt)
    return best


def _build_many_fires(seed, fire_fraction):
    """Default-scenario GameMap + runner, with `fire_fraction` of the interior
    tiles burning (varied intensity), smoke in the path, heat_atten occluders —
    the same shape _build_runner_and_map uses, scaled up for the bench."""
    from field_ab_harness import default_scenario_sim
    from simulation import fire_fixed, gas_fixed
    from simulation.physics_runner import PhysicsRunner

    sim = default_scenario_sim()
    g = sim.gmap
    rng = np.random.default_rng(seed)
    interior = (~g.solid) & (~g.is_vacuum)
    ys, xs = np.nonzero(interior)
    h, w = g.solid.shape

    g.fire[...] = 0
    n_fire = max(8, int(len(ys) * fire_fraction))
    pick = rng.choice(len(ys), size=min(n_fire, len(ys)), replace=False)
    for k in pick:
        yy, xx = int(ys[k]), int(xs[k])
        g.fire[yy, xx] = fire_fixed.quantize_scalar(float(rng.uniform(0.3, 1.0)))

    for gi in range(g.gas.shape[0]):
        if rng.random() < 0.6:
            plane = np.zeros((h, w), dtype=np.int32)
            blob = (rng.random((h, w)) * 0.8 * gas_fixed.FP_ONE_F).astype(np.int32)
            plane[interior] = blob[interior]
            g.gas[gi] = plane
    g.heat_atten[h // 2, :] = 0.7
    if w > 4:
        g.heat_atten[:, w // 2] = 1.0

    runner = PhysicsRunner(bp)
    return runner, g


def _build_sources(runner, g):
    """Replicate cast_fire_heat's source build EXACTLY (so the bench sources are
    the production sources): row-major over fire>0, the (x*7+y*13)%ray_count
    phase, jitter=0, the runner's fire params. Also returns the float scratch +
    dequantized gas the binding needs."""
    import math
    from simulation import fire_fixed, gas_fixed

    fire = g.fire
    h, w = fire.shape
    ys, xs = np.nonzero(fire > 0)
    two_pi = 2.0 * math.pi
    ray_count = runner.fire_ray_count
    sources = []
    for yy, xx in zip(ys.tolist(), xs.tolist()):
        intensity_fire = float(fire[yy, xx]) / fire_fixed.FP_ONE_F
        s = bp.LightSource()
        s.x = float(xx) + 0.5
        s.y = float(yy) + 0.5
        s.max_range = runner.fire_range_base + runner.fire_range_per_i * intensity_fire
        s.ray_count = ray_count
        s.angle_spread = two_pi
        s.angle_center = ((xx * 7 + yy * 13) % ray_count) * (two_pi / ray_count)
        s.intensity = runner.fire_intensity_base + runner.fire_intensity_per_i * intensity_fire
        s.heat = runner.k_fire_heat * intensity_fire
        s.jitter = 0.0
        s.color = runner.fire_color
        sources.append(s)

    rgb = np.zeros((h, w, 3), np.float32)
    dx = np.zeros((h, w), np.float32)
    dy = np.zeros((h, w), np.float32)
    gas_f = (g.gas.astype(np.float32) / gas_fixed.FP_ONE_F).astype(np.float32)
    return sources, rgb, dx, dy, gas_f


def _synth_firestorm(h, w, nfire, seed):
    """A LARGE synthetic firestorm built from raw arrays (no GameMap), so the
    source count can reach the hundreds the B5 feel-test hit (~3 fps). Returns
    the same tuple shape as _build_sources' outputs plus the material tables, so
    the bench can drive the bindings directly. Sources use the production phase
    formula and jitter=0; params mirror a mid-range fire."""
    import math
    rng = np.random.default_rng(seed)
    ray_count = 8
    two_pi = 2.0 * math.pi
    # Fire on distinct interior tiles (row-major, like cast_fire_heat).
    cells = set()
    while len(cells) < nfire:
        cells.add((int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1))))
    cells = sorted(cells)                       # row-major
    sources = []
    for (yy, xx) in cells:
        i = float(rng.uniform(0.3, 1.0))
        s = bp.LightSource()
        s.x = float(xx) + 0.5
        s.y = float(yy) + 0.5
        s.max_range = 6.0 + 10.0 * i
        s.ray_count = ray_count
        s.angle_spread = two_pi
        s.angle_center = ((xx * 7 + yy * 13) % ray_count) * (two_pi / ray_count)
        s.intensity = 0.5 + 1.5 * i
        s.heat = 800.0 * i
        s.jitter = 0.0
        s.color = [1.0, 0.6, 0.2]
        sources.append(s)
    rgb = np.zeros((h, w, 3), np.float32)
    dx = np.zeros((h, w), np.float32)
    dy = np.zeros((h, w), np.float32)
    n_gases = 2
    gas_f = (rng.random((n_gases, h, w)).astype(np.float32) * 0.6)
    abs00 = np.array([[1., 1., 1.], [0.9, 0.2, 0.9]], np.float32)
    sca = np.array([[0.6, 0.6, 0.6], [0.1, 0.7, 0.1]], np.float32)
    atten = np.zeros((h, w, 3), np.float32)
    heat = np.zeros((h, w), np.int32)
    hatten = np.zeros((h, w), np.float32)
    hatten[h // 2, :] = 0.7
    hatten[:, w // 2] = 1.0
    return sources, rgb, dx, dy, gas_f, abs00, sca, atten, heat, hatten


def _bench_synth(seed, h, w, nfire, raycaster):
    """Bench one synthetic firestorm; returns (nsrc, heated, t_loop, t_batch,
    identical)."""
    (sources, rgb, dx, dy, gas_f, abs00, sca,
     atten, heat, hatten) = _synth_firestorm(h, w, nfire, seed)

    def per_source():
        heat[...] = 0
        for s in sources:
            bp.cuda_raycaster_cast(raycaster, s, rgb, dx, dy, gas_f, abs00, sca,
                                   atten, heat=heat, smoke_glow=None,
                                   heat_atten=hatten)

    def batched():
        heat[...] = 0
        bp.cuda_raycaster_cast_batch(raycaster, sources, rgb, dx, dy, gas_f,
                                     abs00, sca, atten, heat=heat,
                                     smoke_glow=None, heat_atten=hatten)

    per_source(); batched()                     # warmup
    heat[...] = 0
    for s in sources:
        bp.cuda_raycaster_cast(raycaster, s, rgb, dx, dy, gas_f, abs00, sca,
                               atten, heat=heat, smoke_glow=None,
                               heat_atten=hatten)
    heat_loop = heat.copy()
    batched()
    identical = bool(np.array_equal(heat_loop, heat))
    t_loop = _time_best_of(per_source, reps=3)
    t_batch = _time_best_of(batched, reps=3)
    return len(sources), int(np.count_nonzero(heat_loop)), t_loop, t_batch, identical


def main() -> int:
    if hasattr(bp, "device_name"):
        try:
            print("device:", bp.device_name())
        except Exception:
            pass
    # Force the raycaster backend on (parity with the live gate wiring; the bench
    # calls the bindings directly, but keep the flag honest).
    if hasattr(bp, "set_raycaster_backend"):
        bp.set_raycaster_backend(True)

    ok = True
    print("S8c item 1 — fire-heat batched-cast payoff bench "
          "(per-source loop vs ONE batched device cast, tol-0-identical heat):")
    for seed, frac in ((20260628, 0.33), (7, 0.5)):
        runner, g = _build_many_fires(seed, frac)
        sources, rgb, dx, dy, gas_f = _build_sources(runner, g)
        nsrc = len(sources)
        abs00 = g.gases.absorption
        sca = g.gases.scatter_albedo
        atten = g.dyn_light_atten
        hatten = g.heat_atten

        def per_source():
            g.heat[...] = 0
            for s in sources:
                bp.cuda_raycaster_cast(
                    runner.raycaster, s, rgb, dx, dy, gas_f, abs00, sca, atten,
                    heat=g.heat, smoke_glow=None, heat_atten=hatten)

        def batched():
            g.heat[...] = 0
            bp.cuda_raycaster_cast_batch(
                runner.raycaster, sources, rgb, dx, dy, gas_f, abs00, sca, atten,
                heat=g.heat, smoke_glow=None, heat_atten=hatten)

        # Warm up both paths (first CUDA call pays context/alloc one-time costs).
        per_source(); batched()
        # Byte-identity re-check at bench scale (cheap; the gate already proves it).
        g.heat[...] = 0
        for s in sources:
            bp.cuda_raycaster_cast(runner.raycaster, s, rgb, dx, dy, gas_f,
                                   abs00, sca, atten, heat=g.heat,
                                   smoke_glow=None, heat_atten=hatten)
        heat_loop = g.heat.copy()
        batched()
        identical = bool(np.array_equal(heat_loop, g.heat))

        # Back-to-back best-of-3 at the same clock/thermal state.
        t_loop = _time_best_of(per_source, reps=3)
        t_batch = _time_best_of(batched, reps=3)
        ratio = t_loop / t_batch if t_batch > 0 else float("inf")
        heated = int(np.count_nonzero(heat_loop))

        print(f"  seed {seed}: {nsrc} fire sources -> {heated} heated tiles | "
              f"per-source loop {t_loop:7.1f} ms | batched {t_batch:6.1f} ms | "
              f"{ratio:5.1f}x faster | heat identical={identical}")
        print(f"           target 30fps={TARGET_MS:.0f} ms, floor={FLOOR_MS:.0f} ms")

        if not identical:
            ok = False
            print(f"  seed {seed}: HEAT MISMATCH at bench scale — FAIL")
        if ratio < RATIO_MIN:
            ok = False
            print(f"  seed {seed}: ratio {ratio:.1f}x < {RATIO_MIN}x — "
                  f"batching did not remove the round-trip tax — FAIL")
        if t_batch > FLOOR_MS:
            ok = False
            print(f"  seed {seed}: batched {t_batch:.1f} ms > {FLOOR_MS:.0f} ms "
                  f"floor — the tax may be back — FAIL")

    # LARGE synthetic firestorms — the hundreds-of-fires regime the B5 feel-test
    # hit (~3 fps). Demonstrates the per-source loop crossing into 3-fps territory
    # (~333 ms) while the batched cast stays playable. Uses raw arrays (no map).
    print("  --- large synthetic firestorms (the ~3-fps regime) ---")
    rc = bp.Raycaster()
    for seed, (hh, ww, nf) in ((11, (96, 96, 300)), (23, (128, 128, 600))):
        nsrc, heated, t_loop, t_batch, identical = _bench_synth(seed, hh, ww, nf, rc)
        ratio = t_loop / t_batch if t_batch > 0 else float("inf")
        fps_loop = 1000.0 / t_loop if t_loop > 0 else float("inf")
        print(f"  {hh}x{ww}, {nsrc} fires -> {heated} heated | "
              f"per-source loop {t_loop:7.1f} ms (~{fps_loop:4.1f} fps) | "
              f"batched {t_batch:6.1f} ms | {ratio:5.1f}x | identical={identical}")
        if not identical:
            ok = False
            print(f"  synth {hh}x{ww}: HEAT MISMATCH — FAIL")
        if ratio < RATIO_MIN:
            ok = False
            print(f"  synth {hh}x{ww}: ratio {ratio:.1f}x < {RATIO_MIN}x — FAIL")
        if t_batch > FLOOR_MS:
            ok = False
            print(f"  synth {hh}x{ww}: batched {t_batch:.1f} ms > {FLOOR_MS:.0f} ms "
                  f"floor — FAIL")

    print("S8C_FIRE_BENCH_RESULT: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
