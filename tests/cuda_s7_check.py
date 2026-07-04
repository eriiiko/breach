"""CUDA-S7 diffuse_solve bit-identity check (runs inside the GPU subprocess).

diffuse_solve is the LAST + hardest solver of the CUDA arc: the once-per-tick
IMPLICIT atmosphere step — the Red-Black Gauss-Seidel pressure relaxation (residual
form, per-cell Dinv), the vacuum BFS + sponge boundary pass, and the wind gradient.

Two gates:

  PART 1 — ISOLATED (the rigorous one): build rich synthetic inputs that hit every
  pass of diffuse_solve —
    * the mu-GATE: vary mu = d_atm*dt so SOME configs skip the GS entirely (mu <=
      MU_EPS -> the diffusion operator is the identity; atmosphere is unchanged by
      the GS, but the sponge + wind still run) and others run STRONG diffusion;
    * the Red-Black GS CONVERGENCE: multi-iter (gs_iters=8) relaxation over a
      non-uniform atmosphere with the per-face permeability bridge (sealed + full +
      partial faces) so the residual-form increment + the per-cell Dinv are
      exercised across many cells/iters;
    * the vacuum BFS layers: exposed-vacuum SEEDS (is_vacuum & !obstacle & !wall) so
      the distance field takes values 0 (seed) / 1 (inner) / 2 (outer) / 255 (far),
      with walls BLOCKING propagation;
    * the sponge tiers: each vac_dist tier scales atmosphere (mul_q16) + wave_v
      (scale_mag, BOTH signs so magnitude-shrink-vs-toward-(-inf) bites) + zeros
      wave_p/wave_v + clamps wave_source;
    * the wind gradient: +/- pressure gradients (atmosphere + wave_p) so wind_x/wind_y
      take both signs via shr_round0 (sign-symmetric >>1).
  Run BOTH the GPU step (bp.cuda_diffuse_solve) and the shipped CPU step
  (bp.AtmosphereSolver().diffuse_solve) on identical copies and assert byte-for-byte
  equality on ALL SIX fields (atmosphere, wave_p, wave_v, wave_source, wind_x,
  wind_y) at tol 0. Many seeds + sizes incl. degenerate 1xN/Nx1 + a tiny 3x3.

  THE DRIFT-FREE CHECK: a UNIFORM atmosphere field (no vacuum, no walls, full
  permeability) must stay EXACTLY uniform after the GS — at the fixed point the
  residual is 0 so the round-to-nearest increment is 0 (no DC mass drift). A
  toward-(-inf) truncating increment would shave -1 LSB off every cell each sweep;
  this check catches that.

  PART 2 — INTEGRATION: a scenario with pressure gradients + a breach (so the GS,
  sponge, AND wind all engage) run through both PhysicsEngine atmos backends via
  set_atmos_backend(), and assert the full per-tick trajectory of the 6 fields is
  bit-identical over 30 ticks. Also confirms the CUDA build's CPU path (atmos
  backend OFF) still reproduces the committed default-scenario golden.

Prints ``S7_RESULT: PASS``/``FAIL`` and exits 0/1.
"""
from __future__ import annotations

import sys

import numpy as np

# Import the CUDA build FIRST so it is the cached `breach_physics` before
# field_ab_harness (which inserts cpp/build/Release on sys.path) imports it.
import breach_physics as bp

FP_ONE = 65536
FP_SHIFT = 16


def _quantize(x):
    """Round-to-nearest Q16.16 (matches fixedpoint::quantize)."""
    x = np.asarray(x, dtype=np.float64)
    return np.int32(np.where(x >= 0, np.floor(x * FP_ONE + 0.5),
                             np.ceil(x * FP_ONE - 0.5)))


def _make_inputs(rng, h, w, atm_mag, wave_mag, vac_frac):
    """Synthetic atmosphere/wave state + masks + perm exercising every pass.

    atm_mag    : real-units peak |atmosphere| (BOTH signs -> the GS relaxes a real
                 gradient; the sponge mul_q16 sink bites).
    wave_mag   : real-units peak |wave_p|/|wave_v| (BOTH signs -> the wind gradient
                 takes both signs; the sponge wave_v scale_mag magnitude-shrink bites).
    vac_frac   : fraction of cells seeded as EXPOSED vacuum (is_vacuum & !solid) so
                 the BFS distance field gets 0/1/2/255 tiers + the sponge tiers fire.
    """
    n = h * w

    atm = (rng.random(n) * 2.0 - 1.0) * atm_mag
    atm[rng.random(n) < 0.10] = 0.0
    atmosphere = _quantize(atm).reshape(h, w)

    wp = (rng.random(n) * 2.0 - 1.0) * wave_mag
    wp[rng.random(n) < 0.20] = 0.0
    wave_p = _quantize(wp).reshape(h, w)
    wv = (rng.random(n) * 2.0 - 1.0) * (wave_mag * 0.5)
    wv[rng.random(n) < 0.20] = 0.0
    wave_v = _quantize(wv).reshape(h, w)
    ws = rng.random(n) * wave_mag
    ws[rng.random(n) < 0.30] = 0.0
    wave_source = _quantize(ws).reshape(h, w)

    # wind buffers (overwritten by diffuse_solve; seed with garbage so a missed
    # write would show up).
    wind_x = _quantize((rng.random(n) * 2.0 - 1.0) * 0.5).reshape(h, w)
    wind_y = _quantize((rng.random(n) * 2.0 - 1.0) * 0.5).reshape(h, w)

    # masks: obstacles / walls scattered; vacuum EXPOSED (is_vacuum & !solid) at
    # vac_frac so the BFS seeds + the sponge tiers fire. Walls block propagation.
    obstacles = (rng.random(n) < 0.06).reshape(h, w)
    is_wall = (rng.random(n) < 0.08).reshape(h, w)
    is_vacuum = (rng.random(n) < vac_frac).reshape(h, w)

    # permeability: float bridge in [0,1] with sealed (0) + full (1) + partial faces.
    perm = rng.random(n).astype(np.float32)
    perm[rng.random(n) < 0.15] = 0.0
    perm[rng.random(n) < 0.30] = 1.0
    permeability = perm.reshape(h, w)

    return {
        "atmosphere": np.ascontiguousarray(atmosphere.astype(np.int32)),
        "wave_p": np.ascontiguousarray(wave_p.astype(np.int32)),
        "wave_v": np.ascontiguousarray(wave_v.astype(np.int32)),
        "wave_source": np.ascontiguousarray(wave_source.astype(np.int32)),
        "wind_x": np.ascontiguousarray(wind_x.astype(np.int32)),
        "wind_y": np.ascontiguousarray(wind_y.astype(np.int32)),
        "obstacles": np.ascontiguousarray(obstacles),
        "is_wall": np.ascontiguousarray(is_wall),
        "is_vacuum": np.ascontiguousarray(is_vacuum),
        "permeability": np.ascontiguousarray(permeability),
    }


SIX = ("atmosphere", "wave_p", "wave_v", "wave_source", "wind_x", "wind_y")


def _run_pair(inp, dials, dt):
    """Run the CPU diffuse_solve and the GPU cuda_diffuse_solve on identical copies.
    Returns (cpu_fields, gpu_fields) dicts of the 6 mutated fields."""
    cpu = bp.AtmosphereSolver()
    for k, v in dials.items():
        setattr(cpu, k, v)

    cpu_f = {f: inp[f].copy() for f in SIX}
    gpu_f = {f: inp[f].copy() for f in SIX}

    cpu.diffuse_solve(
        cpu_f["atmosphere"], cpu_f["wave_p"], cpu_f["wave_v"], cpu_f["wave_source"],
        cpu_f["wind_x"], cpu_f["wind_y"],
        inp["obstacles"], inp["is_wall"], inp["is_vacuum"], inp["permeability"], dt)
    bp.cuda_diffuse_solve(
        gpu_f["atmosphere"], gpu_f["wave_p"], gpu_f["wave_v"], gpu_f["wave_source"],
        gpu_f["wind_x"], gpu_f["wind_y"],
        inp["obstacles"], inp["is_wall"], inp["is_vacuum"], inp["permeability"], dt,
        dials["d_atm"], dials["breach_rate"], int(dials["gs_iters"]))
    return cpu_f, gpu_f


def _compare(cpu_f, gpu_f, label):
    detail = []
    ok = True
    for f in SIX:
        if not np.array_equal(cpu_f[f], gpu_f[f]):
            ok = False
            a, b = cpu_f[f], gpu_f[f]
            mism = int(np.count_nonzero(a != b))
            idx = int(np.argmax(a != b))
            detail.append(f"{label}: {f} {mism} MISMATCH (first @ {idx}: "
                          f"cpu={a.flat[idx]} gpu={b.flat[idx]})")
    return ok, detail


def part1_isolated() -> bool:
    print("PART 1 — isolated GPU vs CPU (synthetic, RB-GS + BFS/sponge + wind):")
    ok = True
    rng = np.random.default_rng(20260628)

    # AtmosphereSolver dials. Sweep d_atm so mu = d_atm*dt straddles MU_EPS (the
    # mu-gate skip path) and reaches strong diffusion; vary breach_rate so the
    # sponge factors vary; gs_iters fixed at the member default 8 (+ a couple others
    # to vary the sweep count).
    dial_sets = [
        dict(d_atm=50.0, breach_rate=5.0, gs_iters=8),     # canon: mu~2.08 (strong)
        dict(d_atm=200.0, breach_rate=8.0, gs_iters=8),    # very strong diffusion
        dict(d_atm=12.0, breach_rate=3.0, gs_iters=4),     # moderate, fewer sweeps
        dict(d_atm=0.05, breach_rate=5.0, gs_iters=8),     # mu <= MU_EPS -> GS SKIP
    ]

    # (h, w, dt, atm_mag, wave_mag, vac_frac)
    configs = [
        (16, 16, 1.0 / 24.0, 5.0, 2.0, 0.06),
        (16, 16, 1.0 / 24.0, 10.0, 5.0, 0.12),   # strong gradients + more vacuum
        (24, 32, 1.0 / 30.0, 8.0, 3.0, 0.04),    # bigger
        (31, 17, 1.0 / 20.0, 6.0, 4.0, 0.10),    # odd dims
        (40, 40, 1.0 / 24.0, 12.0, 8.0, 0.03),   # large, near-format-max
        (12, 20, 1.0 / 24.0, 3.0, 1.0, 0.20),    # lots of vacuum (BFS-heavy)
        (1, 50, 1.0 / 24.0, 5.0, 2.0, 0.08),     # degenerate 1-row
        (50, 1, 1.0 / 24.0, 5.0, 2.0, 0.08),     # degenerate 1-col
        (3, 3, 1.0 / 24.0, 4.0, 1.5, 0.10),      # tiny (mostly-edge gather)
        (8, 8, 1.0 / 24.0, 6.0, 3.0, 0.15),      # small, dense vacuum
    ]

    n_cfg = 0
    for (h, w, dt, am, wm, vf) in configs:
        for seed_bump in range(5):
            for dials in dial_sets:
                n_cfg += 1
                inp = _make_inputs(rng, h, w, am, wm, vf)
                cpu_f, gpu_f = _run_pair(inp, dials, dt)
                good, detail = _compare(cpu_f, gpu_f, f"{h}x{w} d_atm={dials['d_atm']}")
                if not good:
                    ok = False
                    for d in detail:
                        print(f"  {d}")

    if ok:
        print(f"  all {n_cfg} configs bit-identical on all SIX fields (atmosphere, "
              f"wave_p, wave_v, wave_source, wind_x, wind_y) — incl. the Red-Black "
              f"GS convergence (8 sweeps, residual-form increment + per-cell Dinv), "
              f"the mu-gate SKIP path (mu<=MU_EPS), the vacuum BFS layers (0/1/2/255 "
              f"+ wall-blocked), the sponge tiers (mul_q16 + scale_mag +/-), and the "
              f"+/- wind gradients (shr_round0).")
    return ok


def part1_drift_free() -> bool:
    """A UNIFORM atmosphere field must stay EXACTLY uniform after the GS (the
    residual is 0 at the fixed point -> the round-to-nearest increment rounds to 0).
    No vacuum, no walls, full permeability -> the sponge does NOT touch the
    atmosphere, so any change is the GS drifting. A toward-(-inf) truncating
    increment would shave -1 LSB off every cell each sweep — this catches it."""
    print("DRIFT-FREE — a uniform atmosphere stays uniform after the GS:")
    ok = True
    for (h, w, val) in [(16, 16, 1.0), (24, 24, 7.5), (8, 32, -3.0), (3, 3, 2.0)]:
        n = h * w
        atm_val = int(_quantize(val))
        atmosphere = np.full((h, w), atm_val, dtype=np.int32)
        zero = np.zeros((h, w), dtype=np.int32)
        wave_p = zero.copy()
        wave_v = zero.copy()
        wave_source = zero.copy()
        wind_x = zero.copy()
        wind_y = zero.copy()
        obstacles = np.zeros((h, w), dtype=bool)
        is_wall = np.zeros((h, w), dtype=bool)
        is_vacuum = np.zeros((h, w), dtype=bool)     # NO vacuum -> sponge leaves atm
        permeability = np.ones((h, w), dtype=np.float32)   # full -> max diffusion

        # strong diffusion so the GS definitely runs (mu well above MU_EPS).
        dials = dict(d_atm=200.0, breach_rate=5.0, gs_iters=8)
        inp = dict(atmosphere=np.ascontiguousarray(atmosphere),
                   wave_p=np.ascontiguousarray(wave_p),
                   wave_v=np.ascontiguousarray(wave_v),
                   wave_source=np.ascontiguousarray(wave_source),
                   wind_x=np.ascontiguousarray(wind_x),
                   wind_y=np.ascontiguousarray(wind_y),
                   obstacles=np.ascontiguousarray(obstacles),
                   is_wall=np.ascontiguousarray(is_wall),
                   is_vacuum=np.ascontiguousarray(is_vacuum),
                   permeability=np.ascontiguousarray(permeability))
        cpu_f, gpu_f = _run_pair(inp, dials, 1.0 / 24.0)
        # The GPU AND the CPU must both leave the uniform interior unchanged.
        cpu_uniform = np.all(cpu_f["atmosphere"] == atm_val)
        gpu_uniform = np.all(gpu_f["atmosphere"] == atm_val)
        # And the two must agree (already covered by part1, but be explicit).
        agree = np.array_equal(cpu_f["atmosphere"], gpu_f["atmosphere"])
        if not (cpu_uniform and gpu_uniform and agree):
            ok = False
            n_drift_cpu = int(np.count_nonzero(cpu_f["atmosphere"] != atm_val))
            n_drift_gpu = int(np.count_nonzero(gpu_f["atmosphere"] != atm_val))
            print(f"  {h}x{w} val={val}: DRIFT! cpu_drift={n_drift_cpu} cells, "
                  f"gpu_drift={n_drift_gpu} cells, agree={agree}")
        else:
            print(f"  {h}x{w} val={val} ({atm_val} counts): uniform preserved "
                  f"(GS inc->0 at the fixed point, no DC drift).")
    return ok


def part2_integration() -> bool:
    print("PART 2 — integration (PhysicsEngine atmos backend switch, gradient+breach):")
    from field_ab_harness import (capture_trajectory, default_scenario_sim,
                                   diff_trajectories)
    from field_digest import trajectory_digest

    # The committed default-scenario golden (CUDA-S2 re-baseline, 2026-06-28).
    # Re-baselined 2026-07-04 (Q2-lift): pure-integer trig kit wired into the
    # raycaster ray dirs/cone cos + unit facing + Q16.16-snapped HP deltas —
    # the trajectory legitimately moved by quantization-scale deltas.
    # (was 60bd331faccc0b08c11e1ccad3ca75fa6f2aa26232b0b04c1a070b6c65c86ba1)
    GOLDEN = "453829a67a38d79e0befd01d591cb19bdeb19f49d9234fb4d27a5083d126501a"

    def make_gradient_breach():
        """A pressure gradient + a hull breach so the GS (diffusion of the gradient),
        the sponge (the breach vacuum drains the interior), AND the wind (the
        gradient) are all exercised."""
        sim = default_scenario_sim()
        g = sim.gmap
        interior = (~g.solid) & (~g.is_vacuum)
        ys, xs = np.where(interior)
        h, w = g.solid.shape
        # impose a left-to-right atmosphere gradient on the interior (the GS diffuses
        # it; the wind = -grad picks it up).
        for k in range(len(ys)):
            yy, xx = int(ys[k]), int(xs[k])
            g.atmosphere[yy, xx] = int(round((1.0 + 2.0 * xx / max(1, w - 1)) * FP_ONE))
        # punch a breach on the right edge interior column (exposed vacuum) so the
        # sponge drains toward it.
        for yy in range(h):
            if not g.solid[yy, w - 2] and not g.is_vacuum[yy, w - 2]:
                g.is_vacuum[yy, w - 1] = True
        # also seed a wave_source blast so wave_p is non-trivial into diffuse_solve.
        cy, cx = h // 2, w // 2
        for k in range(len(ys)):
            if abs(int(ys[k]) - cy) <= 2 and abs(int(xs[k]) - cx) <= 2:
                g.wave_source[ys[k], xs[k]] = int(round(3.0 * FP_ONE))
        return sim

    bp.set_atmos_backend(False)
    traj_cpu = capture_trajectory(make_sim=make_gradient_breach, n_steps=30)
    bp.set_atmos_backend(True)
    traj_gpu = capture_trajectory(make_sim=make_gradient_breach, n_steps=30)
    bp.set_atmos_backend(False)   # restore

    diffs = diff_trajectories(traj_cpu, traj_gpu, tol=0.0)
    ok = (len(diffs) == 0)
    if not ok:
        print(f"  {len(diffs)} field divergence(s); first: {diffs[0]}")
    else:
        peak_a = max(int(np.abs(s["atmosphere"]).max()) for s in traj_cpu)
        peak_w = max(int(np.abs(s["wind_x"]).max() + np.abs(s["wind_y"]).max())
                     for s in traj_cpu)
        print(f"  CPU vs GPU atmos backend: bit-identical over 30 ticks "
              f"(peak |atmosphere| = {peak_a}, peak |wind| sum = {peak_w} counts).")

    # The default scenario's CPU-backend digest must still match the golden — the
    # atmos backend OFF changes nothing.
    bp.set_atmos_backend(False)
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
        print("S7_RESULT: FAIL (no CUDA build / device)")
        return 1
    print("device:", bp.cuda_device_info())
    p1 = part1_isolated()
    pd = part1_drift_free()
    p2 = part2_integration()
    if p1 and pd and p2:
        print("S7_RESULT: PASS")
        return 0
    print("S7_RESULT: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
