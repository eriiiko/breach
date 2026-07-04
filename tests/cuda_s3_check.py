"""CUDA-S3 water bit-identity check (runs inside the GPU subprocess).

Two gates:

  PART 1 — ISOLATED (the rigorous one): build rich synthetic inputs that hit
  every branch of the 8-pass pipe-model solver (surface incl. the head FLOAT
  BRIDGE with k_p in {0, 0.5} + random float atm/wave_p, nonzero tilt exercising
  the tan poly + the per-tile DOUBLE tilt product, the damped velocity kick with
  Neumann mirror + the +-v_max clamp, donor-cell upwind flux, flux_to_dq, the
  per-cell OUTFLOW LIMITER forced by convergent-flow + shallow-depth patches
  where out_sum > depth, the scale_mag face scaling, divergence, and the
  max0/solid/eps clamps), run BOTH the GPU solver (bp.cuda_water_step) and the
  shipped CPU solver (bp.WaterSolver().step) on identical copies, and assert
  byte-for-byte equality on water_depth AND flow_vx AND flow_vy (tol 0). Many
  seeds + sizes incl. degenerate 1xN / Nx1, null + explicit floor.

  PART 2 — INTEGRATION: run a seeded-water A/B scenario (make_wet: a water blob +
  a ship tilt so transport actually evolves) under both PhysicsEngine backends
  via set_water_backend(), and assert the full per-tick synced field trajectory
  is bit-identical over N ticks. Also confirms the CUDA build's CPU path (water
  backend OFF) reproduces the committed golden.

Prints ``S3_RESULT: PASS``/``FAIL`` and exits 0/1.
"""
from __future__ import annotations

import sys

import numpy as np

# Import the CUDA build FIRST so it is the cached `breach_physics` before
# field_ab_harness (which inserts cpp/build/Release on sys.path) imports it.
import breach_physics as bp

FP_ONE = 65536


# WaterSolver default dials (water_solver.h) — the GPU free-function takes them
# explicitly; the CPU method reads them off the solver. We sweep over a couple of
# (k_p, dial) variants below; these are the base values.
def _set_solver(cpu, g, damping, dx, k_p, v_max, depth_eps):
    cpu.g = g
    cpu.damping = damping
    cpu.dx = dx
    cpu.k_p = k_p
    cpu.v_max = v_max
    cpu.depth_eps = depth_eps


def _quantize(x):
    """Round-to-nearest Q16.16 (matches fixedpoint::quantize)."""
    return np.int32(np.where(x >= 0, np.floor(x * FP_ONE + 0.5),
                             np.ceil(x * FP_ONE - 0.5)))


def _make_inputs(rng, h, w, v_max, dx, k_p, has_floor, atm_on):
    """Synthetic water state that exercises every branch."""
    n = h * w
    # depth: non-negative metres, wide spread incl. 0 (dry) and deep columns.
    depth_m = rng.random(n).astype(np.float64) * 2.5
    depth_m[rng.random(n) < 0.2] = 0.0          # dry cells
    water_depth = _quantize(depth_m).reshape(h, w)

    # velocity: both signs, up to ~+-v_max (so the clamp fires on some).
    vx_m = (rng.random(n) * 2.0 - 1.0) * (v_max * 1.3)
    vy_m = (rng.random(n) * 2.0 - 1.0) * (v_max * 1.3)
    flow_vx = _quantize(vx_m).reshape(h, w)
    flow_vy = _quantize(vy_m).reshape(h, w)

    solid = (rng.random(n) < 0.25).reshape(h, w)

    floor = None
    if has_floor:
        floor = _quantize(rng.random(n).astype(np.float64) * 0.5).reshape(h, w)

    atmosphere = None
    wave_p = None
    if k_p != 0.0 and atm_on:
        # Random float head fields (the FLOAT BRIDGE) — both signs, modest scale.
        atmosphere = (rng.random((h, w)).astype(np.float32) * 2.0 - 0.5)
        wave_p = (rng.random((h, w)).astype(np.float32) * 1.0 - 0.5)

    # --- FORCE the outflow limiter: a convergent-flow + shallow-depth patch.
    # Pick interior cells, give them tiny depth and strong OUTWARD velocity on
    # all faces so out_sum > depth and the per-cell scale fires. We push velocity
    # so the east/south donor faces leave the cell and the west/north faces pull
    # mass away (dq_e[i-1] < 0 etc.), driving a large out_sum on a near-dry cell.
    if h >= 4 and w >= 4:
        for _ in range(max(2, n // 40)):
            cy = int(rng.integers(1, h - 1))
            cx = int(rng.integers(1, w - 1))
            i = cy * w + cx
            solid.flat[i] = False
            # shallow but nonzero depth
            water_depth[cy, cx] = _quantize(np.array(0.02 + rng.random() * 0.03))
            # strong outward velocities at the cell and its neighbours so the
            # donor faces carry large outgoing flux from this cell.
            big = v_max * 0.95
            flow_vx[cy, cx] = _quantize(np.array(big))      # east outflow
            flow_vx[cy, cx - 1] = _quantize(np.array(-big))  # west outflow
            flow_vy[cy, cx] = _quantize(np.array(big))      # south outflow
            flow_vy[cy - 1, cx] = _quantize(np.array(-big))  # north outflow
            # neighbours wet (deep) so the donor depth is large -> big flux.
            for (ny, nx) in ((cy, cx + 1), (cy, cx - 1), (cy + 1, cx), (cy - 1, cx)):
                solid[ny, nx] = False
                water_depth[ny, nx] = _quantize(np.array(2.0))

    return {
        "water_depth": np.ascontiguousarray(water_depth.astype(np.int32)),
        "flow_vx": np.ascontiguousarray(flow_vx.astype(np.int32)),
        "flow_vy": np.ascontiguousarray(flow_vy.astype(np.int32)),
        "solid": np.ascontiguousarray(solid),
        "floor": None if floor is None else np.ascontiguousarray(floor.astype(np.int32)),
        "atmosphere": None if atmosphere is None else np.ascontiguousarray(atmosphere),
        "wave_p": None if wave_p is None else np.ascontiguousarray(wave_p),
    }


def part1_isolated() -> bool:
    print("PART 1 — isolated GPU vs CPU (synthetic, all branches):")
    ok = True
    rng = np.random.default_rng(20260628)
    g, damping, depth_eps = 9.81, 1.0, 1e-5
    # (h, w, dx, k_p, v_max, dt, tilt_x, tilt_y, has_floor, atm_on)
    configs = [
        (16, 16, 0.333, 0.0, 8.0, 0.02, 0.0, 0.0, False, False),
        (16, 16, 0.333, 0.5, 8.0, 0.02, 0.10, -0.07, True, True),
        (24, 32, 0.5, 0.5, 6.0, 0.03, 0.30, 0.20, True, True),
        (31, 17, 0.25, 0.5, 8.0, 0.015, -0.40, 0.35, False, True),
        (40, 40, 0.333, 0.0, 8.0, 0.02, 0.55, -0.55, True, False),
        (12, 20, 1.0, 0.5, 4.0, 0.04, 0.61, 0.61, True, True),  # tilt at clamp edge
        (1, 50, 0.333, 0.5, 8.0, 0.02, 0.2, 0.1, True, True),   # degenerate 1-row
        (50, 1, 0.333, 0.5, 8.0, 0.02, 0.1, 0.2, True, True),   # degenerate 1-col
        (8, 8, 0.333, 0.5, 8.0, 0.02, 0.0, 0.4, False, True),
    ]
    n_cfg = 0
    for (h, w, dx, k_p, v_max, dt, tx, ty, has_floor, atm_on) in configs:
        for seed_bump in range(5):
            n_cfg += 1
            inp = _make_inputs(rng, h, w, v_max, dx, k_p, has_floor, atm_on)

            cpu = bp.WaterSolver()
            _set_solver(cpu, g, damping, dx, k_p, v_max, depth_eps)
            d_cpu = inp["water_depth"].copy()
            vx_cpu = inp["flow_vx"].copy()
            vy_cpu = inp["flow_vy"].copy()
            cpu.step(d_cpu, vx_cpu, vy_cpu,
                     inp["floor"], inp["atmosphere"], inp["wave_p"],
                     inp["solid"], dt, tx, ty)

            d_gpu = inp["water_depth"].copy()
            vx_gpu = inp["flow_vx"].copy()
            vy_gpu = inp["flow_vy"].copy()
            bp.cuda_water_step(
                d_gpu, vx_gpu, vy_gpu,
                inp["floor"], inp["atmosphere"], inp["wave_p"],
                inp["solid"], dt, tx, ty,
                g, damping, dx, k_p, v_max, depth_eps)

            for name, a, b in (("water_depth", d_cpu, d_gpu),
                               ("flow_vx", vx_cpu, vx_gpu),
                               ("flow_vy", vy_cpu, vy_gpu)):
                if not np.array_equal(a, b):
                    ok = False
                    mism = int(np.count_nonzero(a != b))
                    idx = int(np.argmax(a != b))
                    print(f"  {h}x{w} dx={dx} k_p={k_p} tilt=({tx},{ty}): "
                          f"{name} {mism} MISMATCH (first @ {idx}: "
                          f"cpu={a.flat[idx]} gpu={b.flat[idx]})")
    if ok:
        print(f"  all {n_cfg} configs bit-identical on depth+vx+vy (incl. head "
              f"bridge, tilt poly, outflow limiter, dry/solid/eps clamps).")
    return ok


def part2_integration() -> bool:
    print("PART 2 — integration (PhysicsEngine backend switch, seeded water):")
    from field_ab_harness import (capture_trajectory, default_scenario_sim,
                                   diff_trajectories)
    from field_digest import trajectory_digest

    # The committed default-scenario golden (CUDA-S2 re-baseline, 2026-06-28).
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
    GOLDEN = "6d690fda8259b392be9029082013623fbef0fc0322ed3089107d5db220e1b441"

    def make_wet():
        sim = default_scenario_sim()
        g = sim.gmap
        # Seed a deeper water blob over the interior + a ship tilt so the pipe
        # model transports water across the room (the default scenario seeds only
        # two shallow cells with no tilt -> little water motion).
        interior = (~g.solid) & (~g.is_vacuum)
        ys, xs = np.where(interior)
        for k in range(len(ys)):
            # a square blob of ~0.4 m water in one corner of the interior
            if ys[k] < (g.solid.shape[0] // 2) and xs[k] < (g.solid.shape[1] // 2):
                g.water_depth[ys[k], xs[k]] = int(round(0.4 * FP_ONE))
        # Tilt the ship so water slides (the Titanic). |tilt| well inside 30 deg.
        g.tilt_x = 0.15
        g.tilt_y = -0.10
        return sim

    bp.set_water_backend(False)
    traj_cpu = capture_trajectory(make_sim=make_wet, n_steps=30)
    bp.set_water_backend(True)
    traj_gpu = capture_trajectory(make_sim=make_wet, n_steps=30)
    bp.set_water_backend(False)   # restore

    diffs = diff_trajectories(traj_cpu, traj_gpu, tol=0.0)
    ok = (len(diffs) == 0)
    if not ok:
        print(f"  {len(diffs)} field divergence(s); first: {diffs[0]}")
    else:
        peak = max(int(np.abs(s["water_depth"]).max()) for s in traj_cpu)
        mvx = max(int(np.abs(s["flow_vx"]).max()) for s in traj_cpu)
        print(f"  CPU vs GPU water backend: bit-identical over 30 ticks "
              f"(peak |water_depth| = {peak} counts, peak |flow_vx| = {mvx}).")

    # The default (un-seeded-tilt) scenario's CPU-backend digest must match the
    # golden — the water backend OFF changes nothing.
    bp.set_water_backend(False)
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
        print("S3_RESULT: FAIL (no CUDA build / device)")
        return 1
    print("device:", bp.cuda_device_info())
    p1 = part1_isolated()
    p2 = part2_integration()
    if p1 and p2:
        print("S3_RESULT: PASS")
        return 0
    print("S3_RESULT: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
