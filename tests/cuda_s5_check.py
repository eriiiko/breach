"""CUDA-S5 wave_substep bit-identity check (runs inside the GPU subprocess).

Two gates:

  PART 1 — ISOLATED (the rigorous one): build rich synthetic inputs that hit every
  pass of the explicit damped-wave substep — the rate-limited source feed, the
  permeability-weighted Laplacian gather (the per-face perm float bridge incl.
  sealed faces + OOB edges), the int64 velocity kick, the pressure update, the
  per-cell ABSORPTION (scale_mag magnitude shrink, incl. negative wave_v/wave_p so
  the magnitude-shrink-vs-toward-(-inf) distinction bites), the wall/vacuum/obstacle
  BCs, and ABOVE ALL the **mean_wp int64 reduction** (the determinism crux: varied
  interior masks + BOTH-sign wave_p, summed order-free) feeding the one-sided
  anomaly transfer into atmosphere. Run BOTH the GPU substep (bp.cuda_wave_substep)
  and the shipped CPU substep (bp.AtmosphereSolver().wave_substep) on identical
  copies, n times, and assert byte-for-byte equality on wave_p AND wave_v AND
  wave_source AND atmosphere (tol 0). Many seeds + sizes incl. degenerate 1xN/Nx1.

  THE REDUCTION PROOF: because the transfer is ONE-SIDED (wave_p is not drained),
  the final wave_p IS the array that K7 summed. So we reconstruct the CPU int64 sum
  and mean_round directly from the final wave_p over the interior mask, and verify
  the GPU's atmosphere delta equals round_nearest((wave_p - mean_wp)*xfer_q) for
  EXACTLY that sum-derived mean_wp — i.e. the GPU int64 atomicAdd sum == the CPU
  mean_sum to the LSB. We print the raw sum + mean on several configs.

  PART 2 — INTEGRATION: run a seeded shockwave scenario under both PhysicsEngine
  wave backends via set_wave_backend(), and assert the full per-tick trajectory of
  wave_p/wave_v/wave_source/atmosphere is bit-identical over 30 ticks. Also confirms
  the CUDA build's CPU path (wave backend OFF) still reproduces the committed
  default-scenario golden (diffuse_solve/GS is CPU in both paths).

Prints ``S5_RESULT: PASS``/``FAIL`` and exits 0/1.
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


def _quantize_scalar(v: float) -> int:
    s = float(v) * FP_ONE
    return int(np.floor(s + 0.5) if s >= 0 else np.ceil(s - 0.5))


def _mean_round(sum_i: int, count: int) -> int:
    """Exact CPU fixedpoint::mean_round (round-half-away-from-zero, no pre-shift)."""
    if count <= 0:
        return 0
    half = count // 2
    if sum_i >= 0:
        # C integer division truncates toward 0; for sum>=0, (sum+half)>=0.
        return (sum_i + half) // count
    # python // floors; emulate C trunc-toward-0 on a negative dividend.
    num = sum_i - half
    q = -((-num) // count)
    return q


def _round_nearest_q(prod: int) -> int:
    """Exact sign-symmetric round-to-nearest narrow (the anomaly-transfer round)."""
    HALF = 1 << (FP_SHIFT - 1)
    if prod >= 0:
        return (prod + HALF) >> FP_SHIFT
    return -(((-prod) + HALF) >> FP_SHIFT)


def _make_inputs(rng, h, w, wave_mag, source_frac):
    """Synthetic wave state exercising every wave_substep pass.

    wave_mag is the real-units peak |wave_p|/|wave_v| (BOTH signs so the absorb
    magnitude-shrink + the mean_wp +/- sum are exercised). source_frac controls how
    many cells carry an above-threshold wave_source (the rate-limited feed).
    """
    n = h * w
    # wave_p / wave_v: BOTH signs, high magnitude. Q16.16 int32. A chunk left at 0.
    wp = (rng.random(n) * 2.0 - 1.0) * wave_mag
    wp[rng.random(n) < 0.20] = 0.0
    wave_p = _quantize(wp).reshape(h, w)
    wv = (rng.random(n) * 2.0 - 1.0) * (wave_mag * 0.3)
    wv[rng.random(n) < 0.20] = 0.0
    wave_v = _quantize(wv).reshape(h, w)

    # wave_source: non-negative; a fraction above the 0.001 feed threshold, some
    # large (hit the max_source_per_step cap), some just under (no feed).
    ws = np.zeros(n, dtype=np.float64)
    feed_cells = rng.random(n) < source_frac
    ws[feed_cells] = rng.random(int(feed_cells.sum())) * 2.0   # up to 2.0 (caps at 0.5)
    ws[(rng.random(n) < 0.05)] = 0.0005   # below the 0.001 threshold (no feed)
    wave_source = _quantize(ws).reshape(h, w)

    # atmosphere: a bulk field of both signs (the transfer accumulates onto it).
    atm = (rng.random(n) * 2.0 - 1.0) * 5.0
    atmosphere = _quantize(atm).reshape(h, w)

    # masks: obstacles / walls / vacuum scattered so the interior mask varies and
    # the BCs zero a real set of cells (-> the reduction count differs per config).
    obstacles = (rng.random(n) < 0.10).reshape(h, w)
    is_wall = (rng.random(n) < 0.10).reshape(h, w)
    is_vacuum = (rng.random(n) < 0.08).reshape(h, w)

    # permeability: a float bridge in [0,1] with some sealed (0) + full (1) faces.
    perm = rng.random(n).astype(np.float32)
    perm[rng.random(n) < 0.15] = 0.0
    perm[rng.random(n) < 0.30] = 1.0
    permeability = perm.reshape(h, w)

    # wave_absorb: a float bridge in [0,1] — some 0 (no absorb), some high (strong
    # shrink, a < 1 -> k in (0,1)), a few > the inverse-of-absorb_str*dt so a >= 1
    # -> k == 0 (full kill). absorb_strength*dt scales it; pick a spread.
    wa = rng.random(n).astype(np.float32)
    wa[rng.random(n) < 0.20] = 0.0
    wave_absorb = wa.reshape(h, w)

    return {
        "wave_p": np.ascontiguousarray(wave_p.astype(np.int32)),
        "wave_v": np.ascontiguousarray(wave_v.astype(np.int32)),
        "wave_source": np.ascontiguousarray(wave_source.astype(np.int32)),
        "atmosphere": np.ascontiguousarray(atmosphere.astype(np.int32)),
        "obstacles": np.ascontiguousarray(obstacles),
        "is_wall": np.ascontiguousarray(is_wall),
        "is_vacuum": np.ascontiguousarray(is_vacuum),
        "permeability": np.ascontiguousarray(permeability),
        "wave_absorb": np.ascontiguousarray(wave_absorb),
    }


def _verify_reduction(wave_p_final, atm_before, atm_after, masks_interior,
                      xfer_q, label):
    """Reconstruct the int64 sum + mean_round from the FINAL wave_p (the transfer is
    one-sided, so wave_p was NOT drained -> the final array is exactly what K7
    summed), then verify the atmosphere delta == round_nearest((wp-mean)*xfer_q) for
    every interior cell. Returns (ok, sum, mean, count). This is the explicit
    GPU-sum==CPU-sum proof: if the GPU atmosphere matches this formula with the
    sum-derived mean, the GPU's int64 atomicAdd reproduced the CPU mean_sum exactly.
    """
    interior = masks_interior
    count = int(interior.sum())
    wp_flat = wave_p_final.astype(np.int64).ravel()
    int_flat = interior.ravel()
    sum_i = int(wp_flat[int_flat].sum())   # int64 order-free sum (numpy exact)
    mean_wp = _mean_round(sum_i, count)

    delta = (atm_after.astype(np.int64) - atm_before.astype(np.int64)).ravel()
    ok = True
    for idx in np.where(int_flat)[0]:
        anom = int(wp_flat[idx]) - mean_wp
        d = _round_nearest_q(anom * int(xfer_q))
        if int(delta[idx]) != d:
            ok = False
            break
    # non-interior cells must have zero atmosphere delta (one-sided, masked).
    if ok and np.any(delta[~int_flat] != 0):
        ok = False
    return ok, sum_i, mean_wp, count


def part1_isolated() -> bool:
    print("PART 1 — isolated GPU vs CPU (synthetic, all passes + the reduction):")
    ok = True
    rng = np.random.default_rng(20260628)

    # AtmosphereSolver dials (the wave-relevant subset). Sweep a couple so the
    # quantized step constants vary (feed_rate*dt, c_sq*dt, absorb_str*dt, ...).
    dial_sets = [
        dict(c=300.0, damping=3.0, absorb_strength=8.0, transfer=0.5,
             feed_rate=200.0, max_source_per_step=0.5),
        dict(c=66.0, damping=1.5, absorb_strength=12.0, transfer=0.8,
             feed_rate=120.0, max_source_per_step=0.3),
        dict(c=150.0, damping=5.0, absorb_strength=4.0, transfer=0.25,
             feed_rate=300.0, max_source_per_step=0.7),
    ]

    # (h, w, dt, wave_mag, source_frac, n_substeps)
    configs = [
        (16, 16, 0.0015, 1.0, 0.20, 1),
        (16, 16, 0.0015, 3.0, 0.40, 3),    # strong blast, multi-substep
        (24, 32, 0.0010, 5.0, 0.30, 2),    # bigger, very strong wave
        (31, 17, 0.0020, 2.0, 0.50, 1),    # odd dims, dense source
        (40, 40, 0.0008, 8.0, 0.15, 2),    # large grid, near-format-max wave
        (12, 20, 0.0030, 0.5, 0.60, 1),
        (1, 50, 0.0015, 2.0, 0.30, 1),     # degenerate 1-row
        (50, 1, 0.0015, 2.0, 0.30, 1),     # degenerate 1-col
        (8, 8, 0.0015, 4.0, 0.50, 4),      # small, many substeps
        (3, 3, 0.0015, 1.5, 0.50, 1),      # tiny (mostly-edge Laplacian)
    ]

    n_cfg = 0
    printed = 0
    for (h, w, dt, wmag, sfrac, nsub) in configs:
        for seed_bump in range(5):
            for dials in dial_sets:
                n_cfg += 1
                inp = _make_inputs(rng, h, w, wmag, sfrac)

                cpu = bp.AtmosphereSolver()
                for k, v in dials.items():
                    setattr(cpu, k, v)

                wp_c = inp["wave_p"].copy()
                wv_c = inp["wave_v"].copy()
                ws_c = inp["wave_source"].copy()
                atm_c = inp["atmosphere"].copy()

                wp_g = inp["wave_p"].copy()
                wv_g = inp["wave_v"].copy()
                ws_g = inp["wave_source"].copy()
                atm_g = inp["atmosphere"].copy()

                # Track the atmosphere BEFORE the final substep for the reduction
                # proof (the last substep's transfer delta is what we verify).
                interior = (~inp["obstacles"] & ~inp["is_wall"] & ~inp["is_vacuum"])

                for s in range(nsub):
                    if s == nsub - 1:
                        atm_c_pre = atm_c.copy()
                        atm_g_pre = atm_g.copy()
                    cpu.wave_substep(wp_c, wv_c, ws_c, atm_c,
                                     inp["obstacles"], inp["is_wall"],
                                     inp["is_vacuum"], inp["permeability"],
                                     inp["wave_absorb"], dt)
                    bp.cuda_wave_substep(
                        wp_g, wv_g, ws_g, atm_g,
                        inp["obstacles"], inp["is_wall"], inp["is_vacuum"],
                        inp["permeability"], inp["wave_absorb"], dt,
                        dials["c"], dials["damping"], dials["absorb_strength"],
                        dials["transfer"], dials["feed_rate"],
                        dials["max_source_per_step"])

                fields_ok = (np.array_equal(wp_c, wp_g) and
                             np.array_equal(wv_c, wv_g) and
                             np.array_equal(ws_c, ws_g) and
                             np.array_equal(atm_c, atm_g))
                if not fields_ok:
                    ok = False
                    for name, a, b in (("wave_p", wp_c, wp_g),
                                       ("wave_v", wv_c, wv_g),
                                       ("wave_source", ws_c, ws_g),
                                       ("atmosphere", atm_c, atm_g)):
                        if not np.array_equal(a, b):
                            mism = int(np.count_nonzero(a != b))
                            idx = int(np.argmax(a != b))
                            print(f"  {h}x{w} dt={dt} wmag={wmag} c={dials['c']}: "
                                  f"{name} {mism} MISMATCH (first @ {idx}: "
                                  f"cpu={a.flat[idx]} gpu={b.flat[idx]})")
                    continue

                # The reduction proof on the LAST substep (CPU + GPU each), using
                # the EXACT xfer_q = quantize(transfer*dt).
                xfer_q = _quantize_scalar(dials["transfer"] * dt)
                rc_ok, sum_c, mean_c, cnt = _verify_reduction(
                    wp_c, atm_c_pre, atm_c, interior, xfer_q, "cpu")
                rg_ok, sum_g, mean_g, _ = _verify_reduction(
                    wp_g, atm_g_pre, atm_g, interior, xfer_q, "gpu")
                if not (rc_ok and rg_ok and sum_c == sum_g and mean_c == mean_g):
                    ok = False
                    print(f"  {h}x{w}: REDUCTION mismatch cpu(ok={rc_ok} "
                          f"sum={sum_c} mean={mean_c}) gpu(ok={rg_ok} sum={sum_g} "
                          f"mean={mean_g}) count={cnt}")
                elif printed < 5:
                    printed += 1
                    print(f"  reduction[{h}x{w} c={dials['c']}]: interior count="
                          f"{cnt}, int64 sum={sum_c}, mean_wp={mean_c} "
                          f"(GPU sum==CPU sum, verified to the LSB).")

    if ok:
        print(f"  all {n_cfg} configs bit-identical on wave_p/wave_v/wave_source/"
              f"atmosphere (incl. the source feed, the perm-weighted Laplacian, the "
              f"int64 velocity kick, the scale_mag absorption, the wall/vacuum BCs, "
              f"the sign-symmetric anomaly transfer, AND the order-free mean_wp "
              f"int64 reduction with varied interior masks + +/- wave_p).")
    return ok


def part2_integration() -> bool:
    print("PART 2 — integration (PhysicsEngine wave backend switch, shockwave):")
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
    # Re-baselined 2026-07-10 (EOS refactor P4 — combustion on real O2): the O2 gate
    # re-point (FireSimulation + apply_temperature_ignition now read gas[O2],
    # not atmosphere/P — item 3) and the newly-applied trace decay->inert_N2
    # credit (item 2, decisions.md #12 v2.1) both touch the default scenario
    # (it seeds fire + smoke): fire[8,8]/[8,9]'s O2-gated intensity and
    # black_smoke's decay both move the trajectory. Combustion itself (item 1)
    # does NOT touch this scenario (no flammable/wood material in the level).
    # (was 493645d34b01d7ad55e5f0e6ae7254e94989dc1b6dce5c1b7ee5e53acaff3e63)
    # Re-baselined 2026-07-10 (eos-p3fix-thermal-ceiling, design v2.4):
    # the plume shim's T_FLAME_MAX self-limiter fix, the saturating T/u
    # writes, the T_MAX_PHYS/U_MAX rails, the absorption-proportional gas
    # radiant deposit (Pass 1), and the O2-gate hot-zone-equilibrium
    # rescale (P_min/P_full/o2_threshold) all touch the default scenario
    # (it seeds fire + smoke): the fire tiles' heat->T->wind->O2 chain
    # moves the trajectory. ONE re-baseline for the whole branch (the
    # gate-h rule). DIGEST_SPEC_VERSION unchanged (values moved; no field
    # added/removed/retyped).
    # (was 7eeb41d431a79ba01cbafef37416188bbf1ecb2a194d92af5f4ede279c9f2758)
    GOLDEN = "98d3dd7eaf3d574d6e562513cd95f3b5ac077b7c69b1d0b024db931261735473"

    def make_shockwave():
        sim = default_scenario_sim()
        g = sim.gmap
        # Seed a strong wave_source blast in the interior so the explicit wave
        # propagates (the wave_substep loop runs n_wave x per tick) — exercises the
        # GPU wave pass under a real shockwave.
        interior = (~g.solid) & (~g.is_vacuum)
        ys, xs = np.where(interior)
        cy, cx = g.solid.shape[0] // 2, g.solid.shape[1] // 2
        for k in range(len(ys)):
            if abs(int(ys[k]) - cy) <= 2 and abs(int(xs[k]) - cx) <= 2:
                g.wave_source[ys[k], xs[k]] = int(round(3.0 * FP_ONE))
        return sim

    bp.set_wave_backend(False)
    traj_cpu = capture_trajectory(make_sim=make_shockwave, n_steps=30)
    bp.set_wave_backend(True)
    traj_gpu = capture_trajectory(make_sim=make_shockwave, n_steps=30)
    bp.set_wave_backend(False)   # restore

    diffs = diff_trajectories(traj_cpu, traj_gpu, tol=0.0)
    ok = (len(diffs) == 0)
    if not ok:
        print(f"  {len(diffs)} field divergence(s); first: {diffs[0]}")
    else:
        peak = max(int(np.abs(s["wave_p"]).max()) for s in traj_cpu)
        print(f"  CPU vs GPU wave backend: bit-identical over 30 ticks "
              f"(peak |wave_p| = {peak} counts).")

    # The default scenario's CPU-backend digest must still match the golden — the
    # wave backend OFF changes nothing (and diffuse_solve is CPU in both paths).
    bp.set_wave_backend(False)
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
        print("S5_RESULT: FAIL (no CUDA build / device)")
        return 1
    print("device:", bp.cuda_device_info())
    p1 = part1_isolated()
    p2 = part2_integration()
    if p1 and p2:
        print("S5_RESULT: PASS")
        return 0
    print("S5_RESULT: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
