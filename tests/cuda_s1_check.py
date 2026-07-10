"""CUDA-S1 temperature bit-identity check (runs inside the GPU subprocess).

Two gates:

  PART 1 — ISOLATED (the rigorous one): build rich synthetic inputs that hit
  every branch of the 3-pass solver (conversion + SATURATION, conduction with
  NO_FACE + steep gradients + the int64 accumulator, cooling on NEGATIVE temps +
  vacuum exposure), run BOTH the GPU solver (bp.cuda_temperature_step) and the
  shipped CPU solver (bp.TemperatureSolver().step) on identical copies, and
  assert byte-for-byte equality. Many seeds + sizes.

  PART 2 — INTEGRATION: run the canonical A/B scenario (plus a seeded temperature
  hotspot so conduction + cooling actually evolve through the real engine) under
  both PhysicsEngine backends via set_temperature_backend(), and assert the full
  per-tick field trajectory is bit-identical. Also confirms the CUDA build's CPU
  path reproduces the committed golden.

Prints ``S1_RESULT: PASS``/``FAIL`` and exits 0/1.
"""
from __future__ import annotations

import sys

import numpy as np

# Import the CUDA build FIRST so it is the cached `breach_physics` before
# field_ab_harness (which inserts cpp/build/Release on sys.path) imports it.
import breach_physics as bp

FP_ONE = 65536
INT32_MAX = (1 << 31) - 1
INT32_MIN = -(1 << 31)


def _make_inputs(rng, h, w, no_face):
    n = h * w
    # temperature: wide spread incl. negatives and zeros (cooling sign + dead-band)
    temperature = rng.integers(-8 * FP_ONE, 60 * FP_ONE, size=n, dtype=np.int64).astype(np.int32)
    temperature[rng.random(n) < 0.15] = 0
    # heat: non-negative saturating accumulator; some zero, some large
    heat = rng.integers(0, 40 * FP_ONE, size=n, dtype=np.int64).astype(np.int32)
    heat[rng.random(n) < 0.3] = 0
    heat_inv_shift = rng.integers(0, 25, size=n, dtype=np.int32)
    # face_shift (h,w,4): mostly small conduction shifts, ~35% NO_FACE
    face = rng.integers(1, 16, size=(h, w, 4)).astype(np.int32)
    face[rng.random((h, w, 4)) < 0.35] = no_face
    solid = (rng.random(n) < 0.8)
    is_vacuum = (rng.random(n) < 0.12)
    atmosphere = rng.integers(0, FP_ONE, size=n, dtype=np.int64).astype(np.int32)

    # --- explicitly seed the SATURATION branch on a few solid cells ---
    sat_idx = rng.choice(n, size=min(8, n), replace=False)
    solid[sat_idx] = True
    temperature[sat_idx] = INT32_MAX - rng.integers(0, 100, size=sat_idx.size)
    heat[sat_idx] = INT32_MAX - rng.integers(0, 100, size=sat_idx.size)
    heat_inv_shift[sat_idx] = 0   # gain == heat -> overflow -> clamp at INT32_MAX

    return {
        "temperature": temperature.reshape(h, w).copy(),
        "heat": heat.reshape(h, w),
        "heat_inv_shift": heat_inv_shift.reshape(h, w),
        "face_shift": np.ascontiguousarray(face),
        "solid": solid.reshape(h, w).copy(),
        "is_vacuum": is_vacuum.reshape(h, w).copy(),
        "atmosphere": atmosphere.reshape(h, w),
    }


def part1_isolated() -> bool:
    print("PART 1 — isolated GPU vs CPU (synthetic, all branches):")
    ok = True
    rng = np.random.default_rng(20260627)
    configs = [
        (16, 16, 63, 5, 3, 0.3),
        (24, 32, 63, 6, 2, 0.3),
        (31, 17, 63, 4, 1, 0.5),
        (40, 40, 63, 7, 3, 0.1),
        (8, 8, 63, 2, 1, 0.9),
        (1, 50, 63, 5, 3, 0.3),   # degenerate 1-row
        (50, 1, 63, 5, 3, 0.3),   # degenerate 1-col
    ]
    for (h, w, no_face, cs, csv, o2) in configs:
        for seed_bump in range(4):
            inp = _make_inputs(rng, h, w, no_face)

            # CPU reference (the shipped, trusted solver), in place.
            cpu = bp.TemperatureSolver()
            cpu.no_face = no_face
            cpu.cool_shift = cs
            cpu.cool_shift_vacuum = csv
            cpu.o2_vacuum_thresh = o2
            t_cpu = inp["temperature"].copy()
            cpu.step(t_cpu, inp["heat"], inp["heat_inv_shift"], inp["face_shift"],
                     inp["solid"], inp["is_vacuum"], inp["atmosphere"])

            # GPU, in place on an independent copy.
            t_gpu = inp["temperature"].copy()
            bp.cuda_temperature_step(
                t_gpu, inp["heat"], inp["heat_inv_shift"], inp["face_shift"],
                inp["solid"], inp["is_vacuum"], inp["atmosphere"],
                no_face, cs, csv, o2)

            mism = int(np.count_nonzero(t_cpu != t_gpu))
            if mism:
                ok = False
                idx = np.argmax(t_cpu != t_gpu)
                print(f"  {h}x{w} cs={cs}/{csv} o2={o2}: {mism} MISMATCH "
                      f"(first @ {idx}: cpu={t_cpu.flat[idx]} gpu={t_gpu.flat[idx]})")
    if ok:
        print(f"  all {len(configs)*4} configs bit-identical (incl. saturation, "
              f"negatives, NO_FACE, vacuum).")
    return ok


def part2_integration() -> bool:
    print("PART 2 — integration (PhysicsEngine backend switch, seeded hotspot):")
    from field_ab_harness import capture_trajectory, default_scenario_sim, diff_trajectories
    from field_digest import trajectory_digest

    # Re-baselined 2026-06-28 (CUDA-S2): the raycaster pure-density redesign +
    # raycaster.cpp -> /fp:strict + k_fire_heat 200->1600 changed the synced
    # heat/temperature fields (the default A/B scenario seeds fire). Erik
    # feel-check-blessed the new look. CPU<->GPU temperature stayed bit-identical
    # throughout — only this CPU reference moved. (was 542931c7...e875b)
    # Re-baselined 2026-07-04 (Q2-lift): pure-integer trig kit wired into the
    # raycaster ray dirs/cone cos + unit facing + Q16.16-snapped HP deltas —
    # the trajectory legitimately moved by quantization-scale deltas.
    # (was 60bd331faccc0b08c11e1ccad3ca75fa6f2aa26232b0b04c1a070b6c65c86ba1)
    # Re-baselined 2026-07-04 (spawn-stat pin): unit spawn stats switched from
    # rng.multivariate_normal (LAPACK/BLAS -- CPU-dispatch-dependent, caused the
    # Ada tick-0 __unit_hp__ cross-machine divergence, lenovo_dev_setup.md 8b)
    # to Q16.16-quantized species MEANS (ingress door 2): spawn hp now exactly
    # 100.0. Only __unit_hp__ moved, from tick 0; all field trajectories identical.
    # (was 453829a67a38d79e0befd01d591cb19bdeb19f49d9234fb4d27a5083d126501a)
    # Re-baselined 2026-07-05 (P3 statuses): the synced unit record grows the
    # status list (__unit_status__ sub-hash); no field trajectory moved.
    # (was ae1164ca163b4bf49a86694ba78ea5319f86cfff46301c6aa59190207e6c1a12)
    # Re-baselined 2026-07-05 (P4 wave-push): shockwaves displace units +
    # trigger KNOCKED_DOWN (exchange.apply_wave_push, step 9c2). The A/B wave
    # pulse sub-tile-nudges the marine (~0.04 tiles before its heat death),
    # so only __unit_pos__ moved; no tile crossing -> the occupancy stamp and
    # ALL field trajectories are byte-identical (and the pulse's dv ~2.3 is
    # below the knockdown threshold 6.0 -> __unit_status__ unmoved too).
    # (was 6d690fda8259b392be9029082013623fbef0fc0322ed3089107d5db220e1b441)
    GOLDEN = "2bab9702e098b30a2aeb290e9aeb19301c9de4379f64443966ea9f3074a91b7a"

    def make_hot():
        sim = default_scenario_sim()
        g = sim.gmap
        # Seed a temperature hotspot on solid tiles so conduction + cooling evolve
        # through the real engine (the default scenario has no heat source).
        solid = g.solid
        ys, xs = np.where(solid)
        if len(ys):
            for k in range(min(5, len(ys))):
                g.temperature[ys[k], xs[k]] = (50 + 10 * k) * FP_ONE
        return sim

    bp.set_temperature_backend(False)
    traj_cpu = capture_trajectory(make_sim=make_hot, n_steps=30)
    bp.set_temperature_backend(True)
    traj_gpu = capture_trajectory(make_sim=make_hot, n_steps=30)
    bp.set_temperature_backend(False)   # restore

    diffs = diff_trajectories(traj_cpu, traj_gpu, tol=0.0)
    ok = (len(diffs) == 0)
    if not ok:
        print(f"  {len(diffs)} field divergence(s); first: {diffs[0]}")
    else:
        # Confirm the hotspot actually produced non-trivial temperature dynamics.
        peak = max(int(np.abs(s["temperature"]).max()) for s in traj_cpu)
        print(f"  CPU vs GPU backend: bit-identical over 30 ticks "
              f"(peak |temperature| = {peak} counts).")

    # The default (un-seeded) scenario's CPU-backend digest must match the golden.
    bp.set_temperature_backend(False)
    base = capture_trajectory(n_steps=30)
    dig = trajectory_digest(base)
    if dig != GOLDEN:
        ok = False
        print(f"  GOLDEN MISMATCH: {dig[:16]}... != {GOLDEN[:16]}...")
    else:
        print(f"  CUDA build CPU path reproduces the golden ({dig[:12]}...).")
    return ok


def main() -> int:
    if not getattr(bp, "HAS_CUDA", False) or not bp.cuda_available():
        print("S1_RESULT: FAIL (no CUDA build / device)")
        return 1
    print("device:", bp.cuda_device_info())
    p1 = part1_isolated()
    p2 = part2_integration()
    if p1 and p2:
        print("S1_RESULT: PASS")
        return 0
    print("S1_RESULT: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
