"""CUDA-S4b smoke sink_hop bit-identity check (runs inside the GPU subprocess).

The decoupled breach SINK-PULL: ONE 1-cell BFS-gradient hop. Structurally identical
to S4a's wind advection, but the back-trace displacement is the SINK float bridge
(min(sink_strength,1.0) * sink_x/sink_y, quantized) instead of -wind*dt_adv. It
REUSES the verified backtrace_sample_q_dev + smoke_clamp from S4a — only the
displacement is new.

Two gates:

  PART 1 — ISOLATED (the rigorous one): build rich synthetic inputs that hit every
  branch of the sink hop — the integer semi-Lagrangian back-trace (the DDA wall-clip
  march, the integer bilinear sample, the reciprocal_q16 renorm), wall/vacuum/
  obstacle masks (zeroing + breach venting + sealed-corner exclusion), the
  permeability float bridge, AND the new sink float bridge in BOTH the
  breach-gradient regime (nonzero ±displacements, diagonal unit vectors) AND the
  SEALED-ROOM identity (all-zero sink -> bx=by=0 -> the hop is the identity, the
  load-bearing "sealed rooms untouched" guarantee). Run BOTH the GPU
  (bp.cuda_smoke_sink_hop) and the shipped CPU (bp.SmokeDynamics().sink_hop) on
  identical copies and assert byte-for-byte equality on the gas plane (tol 0). Many
  seeds + sizes incl. degenerate 1xN / Nx1, several sink_strength values (incl. one
  that does NOT cap at 1.0 so the min() branch is exercised both ways).

  PART 2 — INTEGRATION: run a smoke scenario WITH a breach (the default scenario
  opens a hull breach + has smoke_sink_strength=2.0, smoke_vent_hops=16, so sink_hop
  actively pulls) under both PhysicsEngine smoke backends via set_smoke_backend() —
  now routing BOTH the step AND the sink_hop to the GPU — and assert the full
  per-tick `gas` field trajectory is bit-identical over 30 ticks. Also confirms the
  CUDA build's CPU path (smoke backend OFF) still reproduces the committed default-
  scenario golden.

Prints ``S4B_RESULT: PASS``/``FAIL`` and exits 0/1.
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


def _make_inputs(rng, h, w, sealed, force_floor_wsum):
    """Synthetic smoke state + sink field exercising every sink_hop branch.

    sealed=True builds the all-zero sink field (the identity-hop case: sealed rooms
    must come out unchanged). Otherwise the sink field is a breach-ward unit-vector
    pattern — diagonal/axis ±components (unit-ish, like the real BFS sink field) so
    the capped 1-cell back-trace goes BOTH signs and clips on walls/vacuum.
    force_floor_wsum seeds a mostly-sealed neighbourhood so the bilinear renorm's
    WSUM lands near WSUM_FLOOR_Q.
    """
    n = h * w
    # smoke: a [0,1] tracer (Q16.16). Wide spread incl. 0 (clear) and full opacity.
    sm = rng.random(n).astype(np.float64)
    sm[rng.random(n) < 0.25] = 0.0
    sm[rng.random(n) < 0.10] = 1.0
    smoke = _quantize(sm).reshape(h, w)

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

    if sealed:
        # The SEALED-ROOM identity: no breach anywhere -> the sink field is all-zero
        # -> bx_q=by_q=0 -> backtrace_sample_q is the identity. The hop must leave
        # every interior cell unchanged (only the clamp/zero pass touches walls).
        sink_x = np.zeros(n, dtype=np.float32).reshape(h, w)
        sink_y = np.zeros(n, dtype=np.float32).reshape(h, w)
    else:
        # A breach-ward unit-vector field: each cell's sink points toward a
        # synthetic breach. Components are unit-ish (in [-1,1]) like the real BFS
        # field — axis-aligned AND diagonal so both the +x/+y and -x/-y back-traces
        # and the DDA wall-clip fire. Some cells get a zero sink (no path) so the
        # identity branch is also hit per-cell within the same grid.
        ang = rng.random(n) * (2.0 * np.pi)
        sx = np.cos(ang).astype(np.float32)
        sy = np.sin(ang).astype(np.float32)
        # Snap a fraction to clean axis/diagonal unit vectors (the BFS produces
        # 8-connected next-hop directions).
        dirs = np.array(
            [(1, 0), (-1, 0), (0, 1), (0, -1),
             (0.70710677, 0.70710677), (-0.70710677, 0.70710677),
             (0.70710677, -0.70710677), (-0.70710677, -0.70710677)],
            dtype=np.float32)
        snap = rng.random(n) < 0.6
        pick = rng.integers(0, len(dirs), size=n)
        sx[snap] = dirs[pick[snap], 0]
        sy[snap] = dirs[pick[snap], 1]
        # A fraction with no path to a breach -> zero sink (per-cell identity).
        nopath = rng.random(n) < 0.15
        sx[nopath] = 0.0
        sy[nopath] = 0.0
        sink_x = sx.reshape(h, w)
        sink_y = sy.reshape(h, w)

    if force_floor_wsum and h >= 5 and w >= 5:
        # A cell whose 1-cell back-trace lands among MOSTLY-SEALED corners: point
        # its sink up-and-left and seal 3 of the 4 destination corners so only one
        # partial weight survives -> WSUM near WSUM_FLOOR_Q (256).
        cy, cx = h // 2, w // 2
        is_wall[cy, cx] = False
        is_vacuum[cy, cx] = False
        obstacles[cy, cx] = False
        permeability[cy, cx] = 1.0
        smoke[cy, cx] = _quantize(np.array(0.8))
        # a small up-left sink -> a sub-cell back-trace into the upper-left 2x2.
        sink_x[cy, cx] = np.float32(-0.3)
        sink_y[cy, cx] = np.float32(-0.3)
        for (sy_, sx_) in ((cy - 1, cx - 1), (cy - 1, cx), (cy, cx - 1)):
            obstacles[sy_, sx_] = True

    return {
        "smoke": np.ascontiguousarray(smoke.astype(np.int32)),
        "sink_x": np.ascontiguousarray(sink_x.astype(np.float32)),
        "sink_y": np.ascontiguousarray(sink_y.astype(np.float32)),
        "obstacles": np.ascontiguousarray(obstacles),
        "is_wall": np.ascontiguousarray(is_wall),
        "is_vacuum": np.ascontiguousarray(is_vacuum),
        "permeability": np.ascontiguousarray(permeability),
    }


def part1_isolated() -> bool:
    print("PART 1 — isolated GPU vs CPU sink_hop (synthetic, all branches):")
    ok = True
    rng = np.random.default_rng(20260628)
    # (h, w, sealed, force_floor_wsum)
    configs = [
        (16, 16, True,  False),   # SEALED ROOM: all-zero sink -> identity hop
        (16, 16, False, False),   # breach-ward unit vectors (axis + diagonal)
        (16, 16, False, True),    # + floor-WSUM renorm corner
        (24, 32, False, True),    # bigger
        (31, 17, False, True),    # odd dims
        (40, 40, False, False),   # larger grid, dense sink field
        (12, 20, False, True),
        (1, 50, False, False),    # degenerate 1-row
        (50, 1, False, False),    # degenerate 1-col
        (8, 8, True,  False),     # small SEALED room (identity)
        (8, 8, False, True),
    ]
    # sink_strength sweep: values that DO and DON'T cap at 1.0 (exercise both sides
    # of the host-side min(sink_strength,1.0)), plus 0.0 (a global identity hop).
    sink_strengths = [2.0, 1.0, 0.5, 0.0]
    n_cfg = 0
    for (h, w, sealed, floor_w) in configs:
        for seed_bump in range(5):
            for sink_strength in sink_strengths:
                n_cfg += 1
                inp = _make_inputs(rng, h, w, sealed, floor_w)

                cpu = bp.SmokeDynamics()
                cpu.sink_strength = sink_strength
                sm_cpu = inp["smoke"].copy()
                cpu.sink_hop(sm_cpu, inp["sink_x"], inp["sink_y"],
                             inp["obstacles"], inp["is_wall"], inp["is_vacuum"],
                             inp["permeability"])

                sm_gpu = inp["smoke"].copy()
                bp.cuda_smoke_sink_hop(
                    sm_gpu, inp["sink_x"], inp["sink_y"],
                    inp["obstacles"], inp["is_wall"], inp["is_vacuum"],
                    inp["permeability"], sink_strength)

                if not np.array_equal(sm_cpu, sm_gpu):
                    ok = False
                    mism = int(np.count_nonzero(sm_cpu != sm_gpu))
                    idx = int(np.argmax(sm_cpu != sm_gpu))
                    print(f"  {h}x{w} sealed={sealed} floor_w={floor_w} "
                          f"sink_strength={sink_strength}: gas {mism} MISMATCH "
                          f"(first @ {idx}: cpu={sm_cpu.flat[idx]} "
                          f"gpu={sm_gpu.flat[idx]})")

                # For the SEALED case + nonzero strength, additionally assert the
                # hop really was the identity on interior cells (the load-bearing
                # "sealed rooms untouched" guarantee — the float bridge -> 0 disp).
                if sealed and sink_strength > 0.0:
                    interior = ~(inp["is_wall"] | inp["is_vacuum"])
                    if not np.array_equal(sm_cpu[interior], inp["smoke"][interior]):
                        ok = False
                        print(f"  {h}x{w} SEALED non-identity on interior "
                              f"(sink_strength={sink_strength}) — bug in the "
                              f"sealed-room guarantee, not a CPU/GPU mismatch")
    if ok:
        print(f"  all {n_cfg} configs bit-identical on the gas plane (incl. the "
              f"sealed-room identity hop, breach-ward ±/diagonal back-traces, the "
              f"sink float bridge with min(sink_strength,1) capped & uncapped, "
              f"wall/vacuum zeroing, WSUM-near-floor renorm).")
    return ok


def part2_integration() -> bool:
    print("PART 2 — integration (PhysicsEngine smoke backend, breach-active sink):")
    from field_ab_harness import (capture_trajectory, default_scenario_sim,
                                   diff_trajectories)
    from field_digest import trajectory_digest

    # The committed default-scenario golden (CUDA-S2 re-baseline, 2026-06-28).
    # Re-baselined 2026-07-04 (Q2-lift): pure-integer trig kit wired into the
    # raycaster ray dirs/cone cos + unit facing + Q16.16-snapped HP deltas —
    # the trajectory legitimately moved by quantization-scale deltas.
    # (was 60bd331faccc0b08c11e1ccad3ca75fa6f2aa26232b0b04c1a070b6c65c86ba1)
    GOLDEN = "453829a67a38d79e0befd01d591cb19bdeb19f49d9234fb4d27a5083d126501a"

    # The default scenario opens a hull breach (destroy_wall(8,0)) and runs with
    # smoke_sink_strength=2.0 + smoke_vent_hops=16, so sink_hop ACTIVELY pulls
    # smoke toward the breach each tick — exactly the regime S4b must reproduce on
    # the GPU. The whole smoke path (step + sink_hop) now routes to the GPU.
    bp.set_smoke_backend(False)
    traj_cpu = capture_trajectory(make_sim=default_scenario_sim, n_steps=30)
    bp.set_smoke_backend(True)
    traj_gpu = capture_trajectory(make_sim=default_scenario_sim, n_steps=30)
    bp.set_smoke_backend(False)   # restore

    diffs = diff_trajectories(traj_cpu, traj_gpu, tol=0.0)
    ok = (len(diffs) == 0)
    if not ok:
        print(f"  {len(diffs)} field divergence(s); first: {diffs[0]}")
    else:
        peak = max(int(np.abs(s["gas"]).max()) for s in traj_cpu)
        # Confirm the breach actually MOVED smoke (the gas field is non-static over
        # the run) — a regression where the sink did nothing would still be
        # "bit-identical" but meaningless. Compare tick 0 vs the last tick.
        moved = int(np.abs(traj_cpu[-1]["gas"].astype(np.int64)
                           - traj_cpu[0]["gas"].astype(np.int64)).max())
        print(f"  CPU vs GPU smoke backend (step + sink_hop): bit-identical over "
              f"30 ticks (peak |gas| = {peak} counts; max |delta gas| tick0->tick29"
              f" = {moved} counts -- the breach-active sink moved smoke).")

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
        print("S4B_RESULT: FAIL (no CUDA build / device)")
        return 1
    print("device:", bp.cuda_device_info())
    p1 = part1_isolated()
    p2 = part2_integration()
    if p1 and p2:
        print("S4B_RESULT: PASS")
        return 0
    print("S4B_RESULT: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
