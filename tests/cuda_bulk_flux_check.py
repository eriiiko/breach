"""EOS P6.1 bulk donor-cell flux bit-identity check (runs inside the GPU subprocess).

The first per-kernel P6 digest gate (docs/eos_p6_gpu_alignment_review.md §4,
P6.1 row: "digest_bulk_flux trajectory + per-plane byte-compare"). Two gates:

  PART 1 — ISOLATED (all branches): rich synthetic inputs that hit every branch
  of the 5-stage donor-cell pass (both donor directions on both axes, sealed
  faces via perm 0 AND via solid neighbors, partial furniture permeability, the
  per-cell OUTFLOW LIMITER forced by shallow-N cells under strong divergent
  wind, the scale_mag face scaling, the divergence apply, the solid/vacuum/
  negative clamps, trace-plane skip, all-zero-conservative-plane skip,
  degenerate 1xN / Nx1 grids), run BOTH the GPU entry (bp.cuda_bulk_flux_
  transport) and the shipped CPU entry (bp.bulk_flux_transport) on identical
  copies, and assert byte-for-byte equality on EVERY gas plane (tol 0) —
  including the trace planes, which must additionally equal the untouched input.

  PART 2 — TRAJECTORY (breach venting + blast, closed loop): a sealed two-room
  scene with a wall breach opening onto hard vacuum and a central blast
  overdensity; each tick derives an integer pseudo-pressure-gradient wind from
  the CURRENT gas state (so air vents toward the breach and the blast expands —
  and so any single-bit divergence feeds back and amplifies), then applies the
  CPU flux to the CPU state and the GPU flux to the GPU state. Asserts (a)
  per-plane byte-identity EVERY tick over the full run, and (b) the per-tick
  sha256 digest trajectories are identical end-to-end. Scenario-strength guards:
  venting must actually remove mass, the blast must actually spread, and the
  outflow limiter must actually engage (all asserted, so the gate can never go
  vacuous).

Engine dispatch waits for P6.5, so there is no backend-switch integration part
here (the review's P6.1 row is explicit: kernel-gate only); the closed-loop
trajectory stands in for it by evolving state through the kernel itself.

Prints ``BULK_FLUX_RESULT: PASS``/``FAIL`` and exits 0/1.
"""
from __future__ import annotations

import hashlib
import sys

import numpy as np

import breach_physics as bp

FP_ONE = 65536


def _quantize(x):
    """Round-to-nearest Q16.16 (matches fixedpoint::quantize)."""
    x = np.asarray(x, dtype=np.float64)
    return np.int32(np.where(x >= 0, np.floor(x * FP_ONE + 0.5),
                             np.ceil(x * FP_ONE - 0.5)))


def _run_both(gas, gas_conservative, wind_x, wind_y, solid, is_vacuum,
              dyn_permeability, dt):
    """Apply the CPU and GPU entries to copies; return (gas_cpu, gas_gpu)."""
    gas_cpu = gas.copy()
    bp.bulk_flux_transport(gas_cpu, gas_conservative, wind_x, wind_y,
                           solid, is_vacuum, dyn_permeability, dt)
    gas_gpu = gas.copy()
    bp.cuda_bulk_flux_transport(gas_gpu, gas_conservative, wind_x, wind_y,
                                solid, is_vacuum, dyn_permeability, dt)
    return gas_cpu, gas_gpu


# ----------------------------------------------------------------------------
# PART 1 — isolated synthetic A/B over every branch.
# ----------------------------------------------------------------------------

def _make_inputs(rng, h, w, n_gases, cons_mask, zero_plane):
    n = h * w
    # gas: non-negative Q16.16 densities around ambient (O2 0.21 / N2 0.79 /
    # traces small), wide spread incl. exact 0 cells and dense blast-like cells.
    gas = np.zeros((n_gases, h, w), dtype=np.int32)
    for gi in range(n_gases):
        base = rng.random((h, w)) * (1.5 if cons_mask[gi] else 0.3)
        base[rng.random((h, w)) < 0.15] = 0.0            # empty cells
        base[rng.random((h, w)) < 0.05] *= 8.0           # dense spikes
        gas[gi] = _quantize(base)
    if zero_plane is not None:
        gas[zero_plane] = 0                              # all-zero plane skip

    # wind: both signs, up to ~8 m/s (the solver's velocity scale).
    wind_x = _quantize((rng.random((h, w)) * 2.0 - 1.0) * 8.0)
    wind_y = _quantize((rng.random((h, w)) * 2.0 - 1.0) * 8.0)

    solid = rng.random((h, w)) < 0.22
    is_vacuum = (~solid) & (rng.random((h, w)) < 0.08)

    # permeability: sealed (0), furniture (0.35/0.5), open (1.0), random floats.
    perm = rng.random((h, w)).astype(np.float32)
    perm[rng.random((h, w)) < 0.15] = 0.0
    perm[rng.random((h, w)) < 0.20] = 0.5
    perm[rng.random((h, w)) < 0.30] = 1.0

    # FORCE the outflow limiter: interior cells with FULL-v outflow on the east
    # AND south faces (wind set on BOTH cells of each face so v_face is exactly
    # `big`, no random dilution — donor is the cell itself on both faces, so
    # out_sum >= 2*big*dt*N; the dt=0.2 CFL-stress config makes that 3.0*N —
    # engagement guaranteed by construction, not by seed luck).
    if h >= 4 and w >= 4:
        for _ in range(max(2, n // 40)):
            cy = int(rng.integers(1, h - 1))
            cx = int(rng.integers(1, w - 1))
            solid[cy, cx] = False
            is_vacuum[cy, cx] = False
            perm[cy, cx] = 1.0
            for gi in range(n_gases):
                if cons_mask[gi] and gi != zero_plane:
                    gas[gi, cy, cx] = int(_quantize(0.01 + rng.random() * 0.02))
            big = 7.5
            wind_x[cy, cx] = int(_quantize(big))       # east face donor = (cy,cx)
            wind_x[cy, cx + 1] = int(_quantize(big))   # ... v_face exactly big
            wind_y[cy, cx] = int(_quantize(big))       # south face donor = (cy,cx)
            wind_y[cy + 1, cx] = int(_quantize(big))
            wind_x[cy, cx - 1] = int(_quantize(-big))  # west-side pull (flavor)
            wind_y[cy - 1, cx] = int(_quantize(-big))  # north-side pull (flavor)
            for (ny, nx) in ((cy, cx + 1), (cy, cx - 1), (cy + 1, cx), (cy - 1, cx)):
                solid[ny, nx] = False
                is_vacuum[ny, nx] = False
                perm[ny, nx] = 1.0

    return {
        "gas": np.ascontiguousarray(gas),
        "gas_conservative": np.ascontiguousarray(np.array(cons_mask, dtype=bool)),
        "wind_x": np.ascontiguousarray(wind_x.astype(np.int32)),
        "wind_y": np.ascontiguousarray(wind_y.astype(np.int32)),
        "solid": np.ascontiguousarray(solid),
        "is_vacuum": np.ascontiguousarray(is_vacuum),
        "dyn_permeability": np.ascontiguousarray(perm.astype(np.float32)),
    }


def part1_isolated() -> bool:
    print("PART 1 — isolated GPU vs CPU (synthetic, all branches):")
    ok = True
    rng = np.random.default_rng(20260711)
    # (h, w, n_gases, cons_mask, zero_plane, dt)
    # cons_mask mirrors the game's layout classes: bulk pair among traces;
    # zero_plane (when set) empties ONE conservative plane (the .any() skip).
    configs = [
        (24, 32, 7, [False, False, False, False, False, True, True], None, 1.0 / 30.0),
        (24, 32, 7, [False, False, False, False, False, True, True], 5, 1.0 / 30.0),
        (16, 16, 2, [True, True], None, 1.0 / 60.0),
        (31, 17, 3, [True, False, True], None, 1.0 / 30.0 / 8.0),   # substep-like dt
        (40, 40, 2, [True, True], None, 0.2),                        # CFL-stress dt
        (40, 40, 2, [True, True], None, 0.05),                       # CFL-hot dt
        (12, 20, 4, [False, True, False, True], 1, 1.0 / 30.0),
        (1, 50, 2, [True, True], None, 1.0 / 30.0),                  # degenerate 1-row
        (50, 1, 2, [True, True], None, 1.0 / 30.0),                  # degenerate 1-col
        (8, 8, 2, [False, False], None, 1.0 / 30.0),                 # no conservative plane
    ]
    n_cfg = 0
    for (h, w, n_gases, cons_mask, zero_plane, dt) in configs:
        for _seed_bump in range(4):
            n_cfg += 1
            inp = _make_inputs(rng, h, w, n_gases, cons_mask, zero_plane)
            gas_cpu, gas_gpu = _run_both(
                inp["gas"], inp["gas_conservative"], inp["wind_x"], inp["wind_y"],
                inp["solid"], inp["is_vacuum"], inp["dyn_permeability"], dt)
            for gi in range(n_gases):
                if not np.array_equal(gas_cpu[gi], gas_gpu[gi]):
                    ok = False
                    a, b = gas_cpu[gi], gas_gpu[gi]
                    mism = int(np.count_nonzero(a != b))
                    idx = int(np.argmax(a != b))
                    print(f"  {h}x{w} n_gases={n_gases} dt={dt} plane {gi} "
                          f"(cons={cons_mask[gi]}): {mism} MISMATCH (first @ {idx}: "
                          f"cpu={a.flat[idx]} gpu={b.flat[idx]})")
                if not inp["gas_conservative"][gi]:
                    # Trace planes must be UNTOUCHED by both entries.
                    if not np.array_equal(gas_gpu[gi], inp["gas"][gi]):
                        ok = False
                        print(f"  {h}x{w} plane {gi}: GPU touched a TRACE plane")
                    if not np.array_equal(gas_cpu[gi], inp["gas"][gi]):
                        ok = False
                        print(f"  {h}x{w} plane {gi}: CPU touched a TRACE plane")
    if ok:
        print(f"  all {n_cfg} configs bit-identical on every plane (donor both "
              f"signs, sealed/partial faces, outflow limiter, solid/vacuum/neg "
              f"clamps, trace + all-zero-plane skips, degenerate grids).")
    return ok


# ----------------------------------------------------------------------------
# PART 2 — closed-loop trajectory: breach venting + blast.
# ----------------------------------------------------------------------------

H2, W2 = 48, 64
N_TICKS = 200
DT2 = 0.15                # CFL-stress dt: v*dt up to 1.2 tiles/step at the wind
                          # clip, so the blast rim's full-v outflow faces push
                          # out_sum past N and the limiter MUST engage (probed)
WMAX_Q = 8 * FP_ONE       # wind clip: +-8 m/s, the solver's velocity scale


def _make_scene():
    """Two-room scene: left room ambient + central blast; right strip hard
    vacuum behind a breached wall (3-cell gap); furniture (perm 0.35) patches."""
    solid = np.zeros((H2, W2), dtype=bool)
    solid[0, :] = solid[-1, :] = True
    solid[:, 0] = solid[:, -1] = True
    wall_x = W2 - 12
    solid[:, wall_x] = True
    breach_lo, breach_hi = H2 // 2 - 1, H2 // 2 + 2
    solid[breach_lo:breach_hi, wall_x] = False          # the breach gap

    is_vacuum = np.zeros((H2, W2), dtype=bool)
    is_vacuum[1:-1, wall_x + 1:W2 - 1] = True           # hard vacuum (space)

    perm = np.ones((H2, W2), dtype=np.float32)
    perm[solid] = 0.0
    # deterministic furniture patches in the room (partial faces).
    for (fy, fx, fh, fw) in ((8, 10, 3, 6), (30, 20, 4, 4), (18, 34, 5, 3)):
        perm[fy:fy + fh, fx:fx + fw] = 0.35

    # bulk pair at ambient split 0.21 / 0.79; a hot 5x5 blast overdensity (x9).
    o2 = np.full((H2, W2), 0.21, dtype=np.float64)
    n2 = np.full((H2, W2), 0.79, dtype=np.float64)
    cy, cx = H2 // 2, (wall_x) // 2
    o2[cy - 2:cy + 3, cx - 2:cx + 3] *= 9.0
    n2[cy - 2:cy + 3, cx - 2:cx + 3] *= 9.0
    o2[solid | is_vacuum] = 0.0
    n2[solid | is_vacuum] = 0.0

    # 3 planes: [trace smoke, O2, N2] — the trace rides along untouched.
    gas = np.zeros((3, H2, W2), dtype=np.int32)
    gas[0] = _quantize(np.where(solid | is_vacuum, 0.0, 0.05))
    gas[1] = _quantize(o2)
    gas[2] = _quantize(n2)
    cons = np.array([False, True, True], dtype=bool)
    return (np.ascontiguousarray(gas), cons, np.ascontiguousarray(solid),
            np.ascontiguousarray(is_vacuum), np.ascontiguousarray(perm))


def _wind_from_state(gas):
    """Deterministic integer pseudo-pressure-gradient wind from the CURRENT
    bulk density (pure int64 numpy — identical on both paths by construction;
    any single-bit gas divergence therefore diverges the wind and amplifies).
    wind = clip((down-gradient central difference of N_total) * 2, +-8 m/s):
    ambient-vs-vacuum contrast (~1.0 unit) gives ~2 m/s venting flow at the
    breach; the x9 blast rim saturates the clip (full-v outflow faces)."""
    ntot = gas[1].astype(np.int64) + gas[2].astype(np.int64)
    wx = np.zeros_like(ntot)
    wy = np.zeros_like(ntot)
    wx[:, 1:-1] = ntot[:, :-2] - ntot[:, 2:]     # -dN/dx * 2
    wy[1:-1, :] = ntot[:-2, :] - ntot[2:, :]     # -dN/dy * 2
    wx *= 2
    wy *= 2
    np.clip(wx, -WMAX_Q, WMAX_Q, out=wx)
    np.clip(wy, -WMAX_Q, WMAX_Q, out=wy)
    return (np.ascontiguousarray(wx.astype(np.int32)),
            np.ascontiguousarray(wy.astype(np.int32)))


def part2_trajectory() -> bool:
    print(f"PART 2 — closed-loop breach-venting + blast trajectory "
          f"({H2}x{W2}, {N_TICKS} ticks, dt={DT2:.4f}):")
    gas0, cons, solid, is_vacuum, perm = _make_scene()
    gas_cpu = gas0.copy()
    gas_gpu = gas0.copy()

    mass0 = int(gas0[1].astype(np.int64).sum() + gas0[2].astype(np.int64).sum())
    peak0 = int(gas0[1].max())
    limiter_seen = False
    dig_cpu = []
    dig_gpu = []
    ok = True

    for t in range(N_TICKS):
        wx_c, wy_c = _wind_from_state(gas_cpu)
        wx_g, wy_g = _wind_from_state(gas_gpu)

        # Outflow-limiter engagement probe (CPU state, pre-flux): a cell whose
        # one-face upwind outflow already exceeds its own N must be limited.
        # (Conservative under-detection: real out_sum sums 4 faces — enough to
        # prove engagement, cheap enough to run every tick.)
        if not limiter_seen:
            n_o2 = gas_cpu[1].astype(np.int64)
            vf = (wx_c[:, :-1].astype(np.int64) + wx_c[:, 1:].astype(np.int64)) >> 1
            donor_e = np.where(vf > 0, n_o2[:, :-1], n_o2[:, 1:])
            flux_scale = int(np.floor(DT2 * FP_ONE + 0.5))   # == quantize(dt)
            out_e = (vf * donor_e * flux_scale) >> 32
            if np.any((out_e > 0) & (out_e > n_o2[:, :-1])):
                limiter_seen = True

        bp.bulk_flux_transport(gas_cpu, cons, wx_c, wy_c,
                               solid, is_vacuum, perm, DT2)
        bp.cuda_bulk_flux_transport(gas_gpu, cons, wx_g, wy_g,
                                    solid, is_vacuum, perm, DT2)

        for gi in range(3):
            if not np.array_equal(gas_cpu[gi], gas_gpu[gi]):
                a, b = gas_cpu[gi], gas_gpu[gi]
                mism = int(np.count_nonzero(a != b))
                idx = int(np.argmax(a != b))
                print(f"  DIVERGENCE tick {t} plane {gi}: {mism} cells "
                      f"(first @ {idx}: cpu={a.flat[idx]} gpu={b.flat[idx]})")
                ok = False
        dig_cpu.append(hashlib.sha256(gas_cpu.tobytes()).hexdigest())
        dig_gpu.append(hashlib.sha256(gas_gpu.tobytes()).hexdigest())
        if not ok:
            break

    if ok and dig_cpu != dig_gpu:
        ok = False   # unreachable if per-plane compares passed; belt+braces
        print("  digest trajectories differ despite per-plane equality (?)")

    # --- scenario-strength guards (never let the gate go vacuous) -----------
    if ok:
        mass_end = int(gas_cpu[1].astype(np.int64).sum()
                       + gas_cpu[2].astype(np.int64).sum())
        peak_end = int(gas_cpu[1].max())
        vented = mass0 - mass_end
        if vented <= 0:
            ok = False
            print(f"  SCENARIO WEAK: no mass vented (mass {mass0} -> {mass_end}).")
        if peak_end >= peak0:
            ok = False
            print(f"  SCENARIO WEAK: blast never spread (peak N_O2 {peak0} -> "
                  f"{peak_end}).")
        if not limiter_seen:
            ok = False
            print("  SCENARIO WEAK: outflow limiter never engaged.")
        if not np.array_equal(gas_gpu[0], gas0[0]):
            ok = False
            print("  TRACE plane moved over the trajectory (must be skipped).")
        if ok:
            print(f"  CPU vs GPU bit-identical over {N_TICKS} ticks x 3 planes "
                  f"(digest trail {dig_cpu[0][:10]}.. -> {dig_cpu[-1][:10]}..); "
                  f"vented {vented / FP_ONE:.1f} units through the breach "
                  f"({100.0 * vented / mass0:.1f}% of the room), blast peak "
                  f"{peak0 / FP_ONE:.2f} -> {peak_end / FP_ONE:.2f}, "
                  f"limiter engaged.")
    return ok


def main() -> int:
    if not getattr(bp, "HAS_CUDA", False) or not bp.cuda_available():
        print("BULK_FLUX_RESULT: FAIL (no CUDA build / device)")
        return 1
    print("device:", bp.cuda_device_info())
    p1 = part1_isolated()
    p2 = part2_trajectory()
    if p1 and p2:
        print("BULK_FLUX_RESULT: PASS")
        return 0
    print("BULK_FLUX_RESULT: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
