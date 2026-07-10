"""CUDA-S4a smoke bit-identity check (runs inside the GPU subprocess).

Two gates:

  PART 1 — ISOLATED (the rigorous one): build rich synthetic inputs that hit every
  branch of the 4-pass smoke solver — the permeability-weighted diffusion Laplacian
  (the per-face perm float bridge incl. sealed faces), the wind-coupled diffusion
  d_eff fold (wind_diffusion_scale > 0 so |wind|^2 actually matters), and above all
  the INTEGER semi-Lagrangian advection: BOTH-SIGN high-magnitude wind (multi-cell,
  NEGATIVE-displacement back-traces + the DDA wall-clip march), wall/vacuum/obstacle
  masks (zeroing + breach venting + sealed-corner exclusion), and a configuration
  forcing a reciprocal_q16 renorm with WSUM near the floor (mostly-sealed corners).
  Run BOTH the GPU solver (bp.cuda_smoke_step) and the shipped CPU solver
  (bp.SmokeDynamics().step) on identical copies and assert byte-for-byte equality
  on the gas plane (tol 0). Many seeds + sizes incl. degenerate 1xN / Nx1, several
  gas planes (each its own d_smoke).

  PART 2 — INTEGRATION: run a seeded smoke+wind scenario under both PhysicsEngine
  smoke backends via set_smoke_backend(), and assert the full per-tick `gas` field
  trajectory is bit-identical over 30 ticks. Also confirms the CUDA build's CPU path
  (smoke backend OFF) still reproduces the committed default-scenario golden.

Prints ``S4A_RESULT: PASS``/``FAIL`` and exits 0/1.
"""
from __future__ import annotations

import sys

import numpy as np

# Import the CUDA build FIRST so it is the cached `breach_physics` before
# field_ab_harness (which inserts cpp/build/Release on sys.path) imports it.
import breach_physics as bp

FP_ONE = 65536
SMOKE_MAX_Q = FP_ONE


def _quantize(x):
    """Round-to-nearest Q16.16 (matches fixedpoint::quantize)."""
    x = np.asarray(x, dtype=np.float64)
    return np.int32(np.where(x >= 0, np.floor(x * FP_ONE + 0.5),
                             np.ceil(x * FP_ONE - 0.5)))


def _make_inputs(rng, h, w, wind_mag, force_floor_wsum):
    """Synthetic smoke state exercising every advection + diffusion branch.

    wind_mag is the real-units peak |wind| (counts = wind_mag * FP_ONE); a large
    value drives multi-cell back-traces of BOTH signs (the negative-displacement
    floor-divide). force_floor_wsum seeds a mostly-sealed neighbourhood so the
    bilinear renorm's WSUM lands near WSUM_FLOOR_Q.
    """
    n = h * w
    # smoke: a [0,1] tracer (Q16.16). Wide spread incl. 0 (clear) and full opacity.
    sm = rng.random(n).astype(np.float64)
    sm[rng.random(n) < 0.25] = 0.0
    sm[rng.random(n) < 0.10] = 1.0
    smoke = _quantize(sm).reshape(h, w)

    # wind: BOTH signs, high magnitude (so the displacement spans several cells and
    # the upwind back-trace goes NEGATIVE for the +sign cells). Q16.16 int32.
    wx_m = (rng.random(n) * 2.0 - 1.0) * wind_mag
    wy_m = (rng.random(n) * 2.0 - 1.0) * wind_mag
    wind_x = _quantize(wx_m).reshape(h, w)
    wind_y = _quantize(wy_m).reshape(h, w)

    # masks: obstacles / walls / vacuum scattered so the march clips and corners
    # get excluded; a few vacuum cells act as breaches (vent targets / 0 corners).
    obstacles = (rng.random(n) < 0.12).reshape(h, w)
    is_wall = (rng.random(n) < 0.10).reshape(h, w)
    is_vacuum = (rng.random(n) < 0.08).reshape(h, w)

    # permeability: a float bridge in [0,1] with some sealed (0) faces + full (1).
    perm = rng.random(n).astype(np.float32)
    perm[rng.random(n) < 0.15] = 0.0      # sealed
    perm[rng.random(n) < 0.30] = 1.0      # fully open
    permeability = perm.reshape(h, w)

    if force_floor_wsum and h >= 5 and w >= 5:
        # Build a cell whose upwind sample lands among MOSTLY-SEALED corners: a
        # small fractional displacement into a neighbourhood where 3 of 4 bilinear
        # corners are obstacles/zero-perm, leaving a single small partial weight ->
        # WSUM near WSUM_FLOOR_Q (256). Place smoke at that one live corner.
        cy, cx = h // 2, w // 2
        is_wall[cy, cx] = False
        is_vacuum[cy, cx] = False
        obstacles[cy, cx] = False
        permeability[cy, cx] = 1.0
        smoke[cy, cx] = _quantize(np.array(0.8))
        # tiny wind at the cell -> a sub-cell back-trace into the 2x2 block to the
        # upper-left; seal 3 of those 4 corners so only one partial weight survives.
        wind_x[cy, cx] = _quantize(np.array(0.02))   # small +x -> back-trace -x
        wind_y[cy, cx] = _quantize(np.array(0.02))   # small +y -> back-trace -y
        # the 2x2 block sampled is around (cx-?, cy-?); seal the three far corners.
        for (sy, sx) in ((cy - 1, cx - 1), (cy - 1, cx), (cy, cx - 1)):
            obstacles[sy, sx] = True

    return {
        "smoke": np.ascontiguousarray(smoke.astype(np.int32)),
        "wind_x": np.ascontiguousarray(wind_x.astype(np.int32)),
        "wind_y": np.ascontiguousarray(wind_y.astype(np.int32)),
        "obstacles": np.ascontiguousarray(obstacles),
        "is_wall": np.ascontiguousarray(is_wall),
        "is_vacuum": np.ascontiguousarray(is_vacuum),
        "permeability": np.ascontiguousarray(permeability),
    }


def part1_isolated() -> bool:
    print("PART 1 — isolated GPU vs CPU (synthetic, all branches):")
    ok = True
    rng = np.random.default_rng(20260628)
    # (h, w, dt, wind_mag, wind_diffusion_scale, advection_rate, force_floor_wsum)
    configs = [
        (16, 16, 0.02, 0.0,  0.0,  225.0, False),  # zero wind (identity advection)
        (16, 16, 0.02, 0.5,  0.0,  225.0, False),  # moderate wind, no wind-diffusion
        (16, 16, 0.02, 1.2,  3.0,  225.0, True),    # wind-diffusion ON + floor wsum
        (24, 32, 0.03, 2.0,  1.5,  225.0, True),    # bigger, strong wind, multi-cell
        (31, 17, 0.015, 1.5, 5.0,  300.0, True),    # odd dims, high wind-diffusion
        (40, 40, 0.02, 3.0,  0.0,  225.0, False),   # very strong wind (deep march)
        (12, 20, 0.04, 0.8,  2.0,  150.0, True),
        (1, 50, 0.02, 1.5,  1.0,  225.0, False),    # degenerate 1-row
        (50, 1, 0.02, 1.5,  1.0,  225.0, False),    # degenerate 1-col
        (8, 8, 0.02, 1.0,  4.0,  225.0, True),
    ]
    # per-config gas-plane diffusion coefficients (sweep several d_smoke values —
    # the multi-gas dispatch sets d_smoke per plane).
    d_smoke_values = [0.1, 0.4, 0.0]
    n_cfg = 0
    for (h, w, dt, wmag, wds, adv, floor_w) in configs:
        for seed_bump in range(5):
            for d_smoke in d_smoke_values:
                n_cfg += 1
                inp = _make_inputs(rng, h, w, wmag, floor_w)

                cpu = bp.SmokeDynamics()
                cpu.d_smoke = d_smoke
                cpu.wind_diffusion_scale = wds
                cpu.advection_rate = adv
                sm_cpu = inp["smoke"].copy()
                cpu.step(sm_cpu, inp["wind_x"], inp["wind_y"],
                         inp["obstacles"], inp["is_wall"], inp["is_vacuum"],
                         inp["permeability"], dt)

                sm_gpu = inp["smoke"].copy()
                bp.cuda_smoke_step(
                    sm_gpu, inp["wind_x"], inp["wind_y"],
                    inp["obstacles"], inp["is_wall"], inp["is_vacuum"],
                    inp["permeability"], dt, d_smoke, wds, adv)

                if not np.array_equal(sm_cpu, sm_gpu):
                    ok = False
                    mism = int(np.count_nonzero(sm_cpu != sm_gpu))
                    idx = int(np.argmax(sm_cpu != sm_gpu))
                    print(f"  {h}x{w} dt={dt} wmag={wmag} wds={wds} adv={adv} "
                          f"d_smoke={d_smoke}: gas {mism} MISMATCH (first @ {idx}: "
                          f"cpu={sm_cpu.flat[idx]} gpu={sm_gpu.flat[idx]})")
    if ok:
        print(f"  all {n_cfg} configs bit-identical on the gas plane (incl. "
              f"negative-displacement advection, the wind^2 diffusion fold, the "
              f"permeability bridge, wall/vacuum zeroing, WSUM-near-floor renorm).")
    return ok


def part2_integration() -> bool:
    print("PART 2 — integration (PhysicsEngine smoke backend switch, seeded smoke):")
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
    # Re-baselined 2026-07-05 (P4 wave-push): shockwaves displace units +
    # trigger KNOCKED_DOWN (exchange.apply_wave_push, step 9c2). The A/B wave
    # pulse sub-tile-nudges the marine (~0.04 tiles before its heat death),
    # so only __unit_pos__ moved; no tile crossing -> the occupancy stamp and
    # ALL field trajectories are byte-identical (and the pulse's dv ~2.3 is
    # below the knockdown threshold 6.0 -> __unit_status__ unmoved too).
    # (was 6d690fda8259b392be9029082013623fbef0fc0322ed3089107d5db220e1b441)
    GOLDEN = "2bab9702e098b30a2aeb290e9aeb19301c9de4379f64443966ea9f3074a91b7a"

    def make_smoky():
        sim = default_scenario_sim()
        g = sim.gmap
        # Seed a dense smoke blob in the interior so the wind field (driven by the
        # default scenario's breach + atmosphere) transports it across the room —
        # the smoke channel is gas plane 0.
        interior = (~g.solid) & (~g.is_vacuum)
        ys, xs = np.where(interior)
        for k in range(len(ys)):
            if ys[k] < (g.solid.shape[0] // 2) and xs[k] < (g.solid.shape[1] // 2):
                g.gas[0, ys[k], xs[k]] = int(round(0.7 * FP_ONE))
        return sim

    bp.set_smoke_backend(False)
    traj_cpu = capture_trajectory(make_sim=make_smoky, n_steps=30)
    bp.set_smoke_backend(True)
    traj_gpu = capture_trajectory(make_sim=make_smoky, n_steps=30)
    bp.set_smoke_backend(False)   # restore

    diffs = diff_trajectories(traj_cpu, traj_gpu, tol=0.0)
    ok = (len(diffs) == 0)
    if not ok:
        print(f"  {len(diffs)} field divergence(s); first: {diffs[0]}")
    else:
        peak = max(int(np.abs(s["gas"]).max()) for s in traj_cpu)
        print(f"  CPU vs GPU smoke backend: bit-identical over 30 ticks "
              f"(peak |gas| = {peak} counts).")

    # The default scenario's CPU-backend digest must still match the golden — the
    # smoke backend OFF changes nothing.
    bp.set_smoke_backend(False)
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
        print("S4A_RESULT: FAIL (no CUDA build / device)")
        return 1
    print("device:", bp.cuda_device_info())
    p1 = part1_isolated()
    p2 = part2_integration()
    if p1 and p2:
        print("S4A_RESULT: PASS")
        return 0
    print("S4A_RESULT: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
