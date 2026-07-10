"""CUDA-S6 FireSimulation::step bit-identity check (runs inside the GPU subprocess).

Two gates:

  PART 1 — ISOLATED (the rigorous one): build rich synthetic inputs that hit every
  pass of FireSimulation::step —
    * P1 the HOST max early-exit (an all-zero / all-tiny fire grid -> the CPU
      returns {} and leaves EVERY field untouched; the GPU must do the same,
      byte-for-byte, with no kernel side-effects);
    * P2 the per-tile signed-logistic feedback: the PINNED left-fold mul_q16 tree,
      the sqrt_q16 wind magnitude (incl. zero AND large +/- wind so the floor-isqrt
      path bites), the 4-neighbour open-atmosphere mean_round, the hot/o2/avail
      gates, and the snap-extinguish below I_min (intensities chosen to straddle it);
    * P3 the own-tile plume overpressure deposit (incl. atmosphere > p_expand_ref so
      `sat` goes negative and the gain>0 guard drops it);
    * P4 the smoke emission SCATTER with OVERLAPPING neighbour deposits — adjacent
      fire cells emit into a SHARED neighbour, so the per-neighbour smoke sum is the
      order-free integer atomicAdd of two-or-more deposits (proves the GPU scatter ==
      the CPU sequential row-major adds to the LSB);
    * P5 the wall burn-through: wall_hp tuned tiny on some flammable WALL cells so
      they cross <=0 and are collected into the destroyed list (set-equality vs the
      CPU), with fire zeroed on exactly those cells;
    * P6 the final fire/smoke clamp.
  Run BOTH the GPU step (bp.cuda_fire_step) and the shipped CPU step
  (bp.FireSimulation().step) on identical copies and assert byte-for-byte equality
  on fire AND atmosphere AND smoke AND wall_hp (tol 0), AND
  set(destroyed_cpu) == set(destroyed_gpu) (order-free), AND len-equal (no dupes).
  Many seeds + sizes incl. degenerate 1xN/Nx1 + a tiny 3x3, plus a DEDICATED
  overlapping-scatter scenario.

  PART 2 — INTEGRATION: run the canonical default scenario (it seeds fire at (8,8)
  and (8,9), so the fire feedback + plume + smoke emission + wall burn-through all
  engage) under both PhysicsEngine fire backends via set_fire_backend(), and assert
  the full per-tick trajectory of fire/atmosphere/smoke/wall_hp (+ all SIM_FIELDS)
  is bit-identical over 30 ticks. Also confirms the CUDA build's CPU path (fire
  backend OFF) still reproduces the committed default-scenario golden.

Prints ``S6_RESULT: PASS``/``FAIL`` and exits 0/1.
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


def _fire_dials():
    """The exact FireParams defaults the CPU step uses, read off a fresh solver so
    the GPU free-function call gets byte-identical dials. (kwargs for cuda_fire_step.)"""
    p = bp.FireSimulation().params
    return dict(
        k_grow=p.k_grow, k_die=p.k_die, fire_T_ext=p.fire_T_ext,
        fire_T_span=p.fire_T_span, fuel_ref=p.fuel_ref, P_min=p.P_min,
        P_full=p.P_full, I_min=p.I_min, k_wind_fan=p.k_wind_fan,
        k_wind_strip=p.k_wind_strip, fire_pressure_gain=p.fire_pressure_gain,
        p_expand_ref=p.p_expand_ref, smoke_emission=p.smoke_emission,
        wall_damage=p.wall_damage, temp_scale=p.temp_scale,
    )


def _make_inputs(rng, h, w, fire_frac, fire_mag, tiny_wall_frac, wind_mag,
                 atm_overpressure=False, all_quiet=False):
    """Synthetic fire state exercising every step() pass.

    fire_frac/fire_mag    control the lit-cell density + peak intensity (both
                          straddling I_min so the snap-extinguish fires).
    tiny_wall_frac        fraction of flammable-wall cells with near-zero wall_hp
                          (so the burn-through crosses <=0 -> destroyed list).
    wind_mag              peak |wind| component (both signs + zeros -> sqrt path).
    atm_overpressure      seed some atmosphere above p_expand_ref (plume sat < 0).
    all_quiet             zero/sub-threshold fire everywhere (the P1 early-exit).
    """
    n = h * w

    # fire: a fraction lit, intensities in (0, fire_mag], some just below/above
    # I_min (0.02). all_quiet -> all sub-threshold (max < 0.001) -> P1 early-exit.
    fr = np.zeros(n, dtype=np.float64)
    if all_quiet:
        # a few cells with a TINY non-zero (< 0.001 real) so the field is not all 0
        # but the host max is still below the early-exit threshold.
        q = rng.random(n) < 0.3
        fr[q] = rng.random(int(q.sum())) * 0.0005
    else:
        lit = rng.random(n) < fire_frac
        fr[lit] = rng.random(int(lit.sum())) * fire_mag
        # sprinkle a band straddling I_min so the snap-extinguish triggers.
        near = rng.random(n) < 0.10
        fr[near] = 0.01 + rng.random(int(near.sum())) * 0.02   # ~[0.01, 0.03)
    fire = _quantize(fr).reshape(h, w)

    # temperature: both signs around fire_T_ext (350) so `hot` spans [0,1] and 0.
    temp = (rng.random(n) * 600.0 - 50.0)        # ~[-50, 550]
    temperature = _quantize(temp).reshape(h, w)

    # atmosphere: a bulk field; optionally push some cells above p_expand_ref
    # (1.30) so the plume `sat = 1 - atm/p_expand` goes negative (gain guarded off).
    atm = rng.random(n) * 1.2                     # ~[0, 1.2]
    if atm_overpressure:
        hi = rng.random(n) < 0.25
        atm[hi] = 1.3 + rng.random(int(hi.sum())) * 1.0   # > p_expand_ref
    atmosphere = _quantize(atm).reshape(h, w)

    # smoke: a pre-existing tracer field (the emission scatter ADDS to it).
    sm = rng.random(n) * 0.5
    smoke = _quantize(sm).reshape(h, w)

    # wall_hp: most cells healthy; a fraction near-zero so burn-through crosses 0.
    whp = rng.random(n) * 80.0 + 20.0             # healthy ~[20, 100]
    tiny = rng.random(n) < tiny_wall_frac
    whp[tiny] = rng.random(int(tiny.sum())) * 0.05   # ~0 HP -> destroyed this tick
    wall_hp = _quantize(whp).reshape(h, w)

    # wind: both signs + a chunk of exact zeros (sqrt(0) path). Q16.16.
    wx = (rng.random(n) * 2.0 - 1.0) * wind_mag
    wy = (rng.random(n) * 2.0 - 1.0) * wind_mag
    wx[rng.random(n) < 0.25] = 0.0
    wy[rng.random(n) < 0.25] = 0.0
    wind_x = _quantize(wx).reshape(h, w)
    wind_y = _quantize(wy).reshape(h, w)

    # masks. flammable drives whether a tile burns; is_wall + flammable + hp<=0
    # is the destroyed predicate, so make the tiny-hp cells likely flammable WALLS.
    is_wall = (rng.random(n) < 0.30).reshape(h, w)
    is_vacuum = (rng.random(n) < 0.08).reshape(h, w)
    flammable = (rng.random(n) < 0.55).reshape(h, w)
    # bias: ensure a healthy number of (flammable & wall & tiny-hp) destroyable cells.
    flammable.ravel()[tiny] = True
    is_wall.ravel()[tiny] = True

    return {
        "fire": np.ascontiguousarray(fire.astype(np.int32)),
        "atmosphere": np.ascontiguousarray(atmosphere.astype(np.int32)),
        "smoke": np.ascontiguousarray(smoke.astype(np.int32)),
        "wall_hp": np.ascontiguousarray(wall_hp.astype(np.int32)),
        "temperature": np.ascontiguousarray(temperature.astype(np.int32)),
        "wind_x": np.ascontiguousarray(wind_x.astype(np.int32)),
        "wind_y": np.ascontiguousarray(wind_y.astype(np.int32)),
        "is_wall": np.ascontiguousarray(is_wall),
        "is_vacuum": np.ascontiguousarray(is_vacuum),
        "flammable": np.ascontiguousarray(flammable),
    }


def _run_pair(inp, dials, dt):
    """Run the CPU step + the GPU step on independent copies; return (cpu, gpu)
    dicts of the 4 mutated fields + the destroyed list."""
    cpu = bp.FireSimulation()
    # CPU uses its own params (== the defaults we pass to the GPU). No setattr
    # needed — _fire_dials() read them off a fresh solver, so they already match.

    c = {k: inp[k].copy() for k in inp}
    g = {k: inp[k].copy() for k in inp}

    d_cpu = cpu.step(c["fire"], c["atmosphere"], c["smoke"], c["wall_hp"],
                     c["temperature"], c["wind_x"], c["wind_y"],
                     c["is_wall"], c["is_vacuum"], c["flammable"], dt)
    d_gpu = bp.cuda_fire_step(
        g["fire"], g["atmosphere"], g["smoke"], g["wall_hp"],
        g["temperature"], g["wind_x"], g["wind_y"],
        g["is_wall"], g["is_vacuum"], g["flammable"], dt, **dials)

    cpu_out = dict(fire=c["fire"], atmosphere=c["atmosphere"], smoke=c["smoke"],
                   wall_hp=c["wall_hp"], destroyed=[tuple(t) for t in d_cpu])
    gpu_out = dict(fire=g["fire"], atmosphere=g["atmosphere"], smoke=g["smoke"],
                   wall_hp=g["wall_hp"], destroyed=[tuple(t) for t in d_gpu])
    return cpu_out, gpu_out


def _compare(cpu_out, gpu_out, label):
    """Bit-identity (tol 0) on the 4 fields + set-equality + len-equality on the
    destroyed list. Returns (ok, detail)."""
    ok = True
    detail = []
    for name in ("fire", "atmosphere", "smoke", "wall_hp"):
        a, b = cpu_out[name], gpu_out[name]
        if not np.array_equal(a, b):
            ok = False
            mism = int(np.count_nonzero(a != b))
            idx = int(np.argmax(a != b))
            detail.append(f"{name}: {mism} MISMATCH (first @ {idx}: "
                          f"cpu={a.flat[idx]} gpu={b.flat[idx]})")
    dc, dg = cpu_out["destroyed"], gpu_out["destroyed"]
    set_c, set_g = set(dc), set(dg)
    if set_c != set_g:
        ok = False
        only_c = set_c - set_g
        only_g = set_g - set_c
        detail.append(f"destroyed SET mismatch: cpu-only={list(only_c)[:5]} "
                      f"gpu-only={list(only_g)[:5]} (|cpu|={len(dc)} |gpu|={len(dg)})")
    # length equality catches a GPU drop/dupe even if the set happened to match.
    if len(dc) != len(set_c):
        ok = False
        detail.append(f"CPU destroyed has DUPES: len={len(dc)} set={len(set_c)}")
    if len(dg) != len(set_g):
        ok = False
        detail.append(f"GPU destroyed has DUPES: len={len(dg)} set={len(set_g)}")
    if len(dc) != len(dg):
        ok = False
        detail.append(f"destroyed LEN differ: cpu={len(dc)} gpu={len(dg)}")
    return ok, detail


def _overlap_scatter_scenario():
    """A hand-built grid proving the P4 atomicAdd order-freedom: a ROW of adjacent
    fire cells so each interior air neighbour receives OVERLAPPING deposits from
    BOTH horizontal fire neighbours (and the verticals). All cells flammable air
    (no walls) so every deposit lands; the GPU per-neighbour smoke sum must equal
    the CPU's sequential adds to the LSB."""
    h, w = 5, 9
    n = h * w
    fire = np.zeros((h, w), dtype=np.int32)
    # a solid horizontal bar of fire on row 2, cols 1..7 -> every air cell on rows
    # 1 and 3 (and the row-2 gaps) gets deposits from multiple fire sources.
    for x in range(1, 8):
        fire[2, x] = _quantize(0.7 + 0.03 * x)   # varied intensities -> varied deposits
    atmosphere = _quantize(np.full((h, w), 0.8)).astype(np.int32)
    smoke = _quantize(np.full((h, w), 0.1)).astype(np.int32)
    wall_hp = _quantize(np.full((h, w), 50.0)).astype(np.int32)
    temperature = _quantize(np.full((h, w), 500.0)).astype(np.int32)  # hot -> grow
    wind_x = np.zeros((h, w), dtype=np.int32)
    wind_y = np.zeros((h, w), dtype=np.int32)
    is_wall = np.zeros((h, w), dtype=bool)        # NO walls -> every deposit lands
    is_vacuum = np.zeros((h, w), dtype=bool)
    flammable = np.ones((h, w), dtype=bool)
    return dict(
        fire=np.ascontiguousarray(fire), atmosphere=np.ascontiguousarray(atmosphere),
        smoke=np.ascontiguousarray(smoke), wall_hp=np.ascontiguousarray(wall_hp),
        temperature=np.ascontiguousarray(temperature),
        wind_x=np.ascontiguousarray(wind_x), wind_y=np.ascontiguousarray(wind_y),
        is_wall=np.ascontiguousarray(is_wall),
        is_vacuum=np.ascontiguousarray(is_vacuum),
        flammable=np.ascontiguousarray(flammable))


def part1_isolated() -> bool:
    print("PART 1 — isolated GPU vs CPU (synthetic, all passes + scatter + burn):")
    ok = True
    rng = np.random.default_rng(20260628)
    dials = _fire_dials()

    # (h, w, dt, fire_frac, fire_mag, tiny_wall_frac, wind_mag, overpressure, quiet)
    configs = [
        (16, 16, 0.05, 0.30, 1.0, 0.10, 2.0,  False, False),
        (16, 16, 0.05, 0.50, 1.0, 0.20, 5.0,  True,  False),   # overpressure + burn
        (24, 32, 0.10, 0.25, 0.9, 0.08, 0.0,  False, False),   # zero wind everywhere
        (31, 17, 0.02, 0.60, 1.0, 0.25, 8.0,  True,  False),   # odd dims, dense+strong
        (40, 40, 0.03, 0.15, 0.8, 0.05, 100.0, False, False),  # large + huge wind (sqrt)
        (12, 20, 0.20, 0.40, 1.0, 0.15, 3.0,  True,  False),   # large dt
        (1, 50, 0.05, 0.40, 1.0, 0.20, 2.0,  False, False),    # degenerate 1-row
        (50, 1, 0.05, 0.40, 1.0, 0.20, 2.0,  False, False),    # degenerate 1-col
        (3, 3, 0.05, 0.60, 1.0, 0.30, 1.5,  False, False),     # tiny (mostly-edge mean)
        (16, 16, 0.05, 0.00, 0.0, 0.00, 0.0,  False, True),    # P1 early-exit (all quiet)
    ]

    n_cfg = 0
    for (h, w, dt, ff, fm, twf, wm, ov, quiet) in configs:
        for seed_bump in range(5):
            n_cfg += 1
            inp = _make_inputs(rng, h, w, ff, fm, twf, wm,
                               atm_overpressure=ov, all_quiet=quiet)
            cpu_out, gpu_out = _run_pair(inp, dials, dt)
            good, detail = _compare(cpu_out, gpu_out, f"{h}x{w}")
            if not good:
                ok = False
                for d in detail:
                    print(f"  {h}x{w} dt={dt} ff={ff} wm={wm} quiet={quiet}: {d}")
            # For the all-quiet config, assert the fields are byte-UNCHANGED (the
            # P1 early-exit must not touch anything) on BOTH paths.
            if quiet:
                for name in ("fire", "atmosphere", "smoke", "wall_hp"):
                    if not np.array_equal(cpu_out[name], inp[name]):
                        ok = False
                        print(f"  P1 early-exit: CPU modified {name}!")
                    if not np.array_equal(gpu_out[name], inp[name]):
                        ok = False
                        print(f"  P1 early-exit: GPU modified {name}!")
                if cpu_out["destroyed"] or gpu_out["destroyed"]:
                    ok = False
                    print("  P1 early-exit: destroyed list non-empty!")

    # The DEDICATED overlapping-scatter scenario (the atomicAdd order-freedom proof).
    inp = _overlap_scatter_scenario()
    cpu_out, gpu_out = _run_pair(inp, dials, 0.1)
    good, detail = _compare(cpu_out, gpu_out, "overlap-scatter")
    if not good:
        ok = False
        for d in detail:
            print(f"  overlap-scatter: {d}")
    else:
        # Show the overlap actually happened: how much smoke accumulated where two
        # fire neighbours deposited into a shared air cell.
        smoke_before = inp["smoke"]
        delta = gpu_out["smoke"].astype(np.int64) - smoke_before.astype(np.int64)
        multi = int((delta > 0).sum())
        peak = int(delta.max())
        print(f"  overlap-scatter: {multi} air cells received emissions, peak smoke "
              f"delta = {peak} counts (overlapping atomicAdd order-free == CPU).")

    if ok:
        n_dest = len(gpu_out["destroyed"])
        print(f"  all {n_cfg} configs + overlap-scatter bit-identical on fire/"
              f"atmosphere/smoke/wall_hp (tol 0), destroyed lists set-equal "
              f"(incl. the P1 host early-exit, the P2 logistic + sqrt_q16 + "
              f"snap-extinguish, the P3 plume, the P4 OVERLAPPING smoke scatter, "
              f"the P5 burn-through collection).")
    return ok


def part2_integration() -> bool:
    print("PART 2 — integration (PhysicsEngine fire backend switch, default scenario):")
    from field_ab_harness import capture_trajectory, diff_trajectories
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

    N_TICKS = 30
    bp.set_fire_backend(False)
    traj_cpu = capture_trajectory(n_steps=N_TICKS)
    bp.set_fire_backend(True)
    traj_gpu = capture_trajectory(n_steps=N_TICKS)
    bp.set_fire_backend(False)   # restore

    diffs = diff_trajectories(traj_cpu, traj_gpu, tol=0.0)
    ok = (len(diffs) == 0)
    if not ok:
        print(f"  {len(diffs)} field divergence(s); first: {diffs[0]}")
    else:
        peak = max(int(np.abs(s["fire"]).max()) for s in traj_cpu)
        print(f"  CPU vs GPU fire backend: bit-identical over {N_TICKS} ticks "
              f"(peak |fire| = {peak} counts).")

    # The default scenario's CPU-backend digest must still match the golden — the
    # fire backend OFF changes nothing.
    bp.set_fire_backend(False)
    base = capture_trajectory(n_steps=N_TICKS)
    dig = trajectory_digest(base)
    if dig != GOLDEN:
        ok = False
        print(f"  GOLDEN MISMATCH: {dig[:16]}... != {GOLDEN[:16]}...")
    else:
        print(f"  CUDA build CPU path reproduces the golden ({dig[:12]}...).")
    return ok


def main() -> int:
    if not getattr(bp, "HAS_CUDA", False) or not bp.cuda_available():
        print("S6_RESULT: FAIL (no CUDA build / device)")
        return 1
    print("device:", bp.cuda_device_info())
    p1 = part1_isolated()
    p2 = part2_integration()
    if p1 and p2:
        print("S6_RESULT: PASS")
        return 0
    print("S6_RESULT: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
