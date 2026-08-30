"""EOS P6.2 — fused 3-field SL advection bit-identity check (runs inside the
GPU subprocess).

Three gates:

  PART 1 — ISOLATED (synthetic, all branches): rich random inputs that hit
  every branch of the fused backtrace — zero-displacement fast path, sub-cell
  fractional displacements, multi-cell BOTH-SIGN backtraces (the DDA wall-clip
  march + the NEGATIVE-displacement floor-divide), sealed/breach/live cmask
  corners (incl. vacuum-sealed combinations), the all-live-corner renorm skip,
  a forced WSUM-near-floor Newton renorm, negative and extreme T values,
  n_sub in {1, 2, 3, 8}, degenerate 1xN / Nx1 grids. Run BOTH the GPU chain
  (bp.cuda_eos_sl_advect) and the CPU reference (bp.eos_sl_advect_ref — the
  SAME file-local backtrace routine EOSSolver::step calls) on identical copies
  and assert byte-for-byte equality on all three fields + digest equality.

  PART 2 — TRAJECTORY (the review's §4 P6.2 digest gate): a blast + venting
  scenario (hot core + O2 overpressure in a hull-ringed room breached to
  vacuum) driven through the REAL engine path (PhysicsEngine.run_substeps →
  EOSSolver::step) for 80 ticks on the CPU. Per tick: snapshot the
  step-1-entry (wind, T) state, run the real tick, then replay the isolated
  advection on the snapshot with the solver's own schedule (dbg_last_n_sub)
  through BOTH the CPU reference and the GPU chain, asserting
      ref_digest == gpu_digest
  and byte-equality of the post-advect fields — a full per-tick digest
  trajectory, CPU vs GPU, over the whole run. The scenario is asserted to
  actually drive advection hard (n_sub pins at N_SUB_MAX, multi-tile
  displacements).

  P-E1 (energy-books arc, design §2.1.1/§2.1.6 — AUTHORIZED REWRITE): SL
  advection is **u-only** now. The `.t` slot is retired (temperature rides
  the conservative energy books in step 1d), so (a) the digests here are
  over (wy, wx) alone, (b) the `== EOSSolver.digest_advect` leg is gone —
  digest_advect now hashes T-after-recovery and is taken AFTER the flux
  call, which an isolated advection replay cannot reproduce — and (c) a NEW
  assertion replaces it: both twins must leave `temperature` byte-for-byte
  untouched, which is the retirement itself, gated.

  PART 3 — the CUDA build's CPU path still reproduces the committed
  default-scenario golden (the s4a-check idiom; proves the P6.2 additions
  changed no CPU trajectory).

Prints ``P62_RESULT: PASS``/``FAIL`` and exits 0/1.
"""
from __future__ import annotations

import sys

import numpy as np

# Import the CUDA build FIRST so it is the cached `breach_physics` before
# anything else (field_ab_harness inserts cpp/build/Release) imports it.
import breach_physics as bp

FP_ONE = 65536


def _quantize(x):
    """Round-to-nearest Q16.16 (matches fixedpoint::quantize)."""
    x = np.asarray(x, dtype=np.float64)
    return np.int32(np.where(x >= 0, np.floor(x * FP_ONE + 0.5),
                             np.ceil(x * FP_ONE - 0.5)))


def _make_inputs(rng, h, w, wind_mag, t_mag, force_floor_wsum):
    """Synthetic (u, T, masks, perm) exercising every backtrace branch."""
    n = h * w
    wx = _quantize(((rng.random(n) * 2.0 - 1.0) * wind_mag)).reshape(h, w)
    wy = _quantize(((rng.random(n) * 2.0 - 1.0) * wind_mag)).reshape(h, w)
    # exact zeros so the zero-displacement fast path is hit alongside marches
    zero_mask = rng.random((h, w)) < 0.15
    wx[zero_mask] = 0
    wy[zero_mask] = 0

    # T is a ΔT above ambient: include NEGATIVE values (down to the -289 K
    # floor regime) and hot-core extremes.
    t = ((rng.random(n) * 2.0 - 1.0) * t_mag)
    t[rng.random(n) < 0.10] = -289.0
    t[rng.random(n) < 0.05] = 9000.0
    temperature = _quantize(t).reshape(h, w)

    solid = (rng.random(n) < 0.10).reshape(h, w)
    is_vacuum = (rng.random(n) < 0.08).reshape(h, w)

    perm = rng.random(n).astype(np.float32)
    perm[rng.random(n) < 0.12] = 0.0      # sealed faces (cmask 0, even open)
    perm[rng.random(n) < 0.30] = 1.0
    perm = perm.reshape(h, w)

    if force_floor_wsum and h >= 5 and w >= 5:
        # A live cell with a tiny fractional backtrace into a 2x2 corner block
        # where 3 of 4 corners are sealed -> WSUM lands near WSUM_FLOOR_Q and
        # the Newton-reciprocal renorm fires.
        cy, cx = h // 2, w // 2
        solid[cy, cx] = False
        is_vacuum[cy, cx] = False
        perm[cy, cx] = 1.0
        wx[cy, cx] = _quantize(np.array(0.02))   # +x -> back-trace -x
        wy[cy, cx] = _quantize(np.array(0.02))
        temperature[cy, cx] = _quantize(np.array(120.0))
        for (sy, sx) in ((cy - 1, cx - 1), (cy - 1, cx), (cy, cx - 1)):
            solid[sy, sx] = True

    return {
        "wind_x": np.ascontiguousarray(wx.astype(np.int32)),
        "wind_y": np.ascontiguousarray(wy.astype(np.int32)),
        "temperature": np.ascontiguousarray(temperature.astype(np.int32)),
        "solid": np.ascontiguousarray(solid),
        "is_vacuum": np.ascontiguousarray(is_vacuum),
        "perm": np.ascontiguousarray(perm.astype(np.float32)),
    }


def _run_pair(inp, dt, n_sub):
    """Run CPU reference + GPU chain on identical copies; return everything."""
    f_ref = {k: inp[k].copy() for k in ("wind_x", "wind_y", "temperature")}
    dig_ref = bp.eos_sl_advect_ref(
        f_ref["wind_x"], f_ref["wind_y"], f_ref["temperature"],
        inp["solid"], inp["is_vacuum"], inp["perm"], dt, n_sub)
    f_gpu = {k: inp[k].copy() for k in ("wind_x", "wind_y", "temperature")}
    dig_gpu = bp.cuda_eos_sl_advect(
        f_gpu["wind_x"], f_gpu["wind_y"], f_gpu["temperature"],
        inp["solid"], inp["is_vacuum"], inp["perm"], dt, n_sub)
    return f_ref, dig_ref, f_gpu, dig_gpu


def part1_isolated() -> bool:
    print("PART 1 — isolated GPU vs CPU reference (synthetic, all branches):")
    ok = True
    rng = np.random.default_rng(20260711)
    # (h, w, dt, wind_mag [m/s-class], t_mag, n_sub, force_floor_wsum)
    configs = [
        (16, 16, 0.5,  0.0,   300.0, 1, False),   # zero wind (identity fast path)
        (16, 16, 0.5,  0.6,   300.0, 1, False),   # sub-cell fractional
        (16, 16, 0.5,  0.6,   300.0, 3, True),    # substepped + floor-wsum renorm
        (24, 32, 1.0,  3.0,   900.0, 2, True),    # multi-cell both-sign marches
        (31, 17, 0.75, 5.0,  2000.0, 8, True),    # odd dims, deep march, full cap
        (40, 40, 2.0,  4.0,  6000.0, 1, False),   # very deep march (8 tiles)
        (12, 20, 1.0,  1.5,   300.0, 8, True),
        (1, 50, 1.0,  2.0,   300.0, 2, False),    # degenerate 1-row
        (50, 1, 1.0,  2.0,   300.0, 2, False),    # degenerate 1-col
        (8, 8, 0.5,  1.0,   300.0, 3, True),
    ]
    n_cfg = 0
    for (h, w, dt, wmag, tmag, n_sub, floor_w) in configs:
        for seed_bump in range(5):
            n_cfg += 1
            inp = _make_inputs(rng, h, w, wmag, tmag, floor_w)
            f_ref, dig_ref, f_gpu, dig_gpu = _run_pair(inp, dt, n_sub)
            for k in ("wind_x", "wind_y", "temperature"):
                if not np.array_equal(f_ref[k], f_gpu[k]):
                    ok = False
                    mism = int(np.count_nonzero(f_ref[k] != f_gpu[k]))
                    idx = int(np.argmax(f_ref[k] != f_gpu[k]))
                    print(f"  {h}x{w} dt={dt} wmag={wmag} n_sub={n_sub}: "
                          f"{k} {mism} MISMATCH (first @ {idx}: "
                          f"cpu={f_ref[k].flat[idx]} gpu={f_gpu[k].flat[idx]})")
            if dig_ref != dig_gpu:
                ok = False
                print(f"  {h}x{w} dt={dt} n_sub={n_sub}: digest mismatch "
                      f"(ref={dig_ref:#018x} gpu={dig_gpu:#018x})")
    if ok:
        print(f"  all {n_cfg} configs bit-identical on (wind_x, wind_y, T), "
              f"digests equal (incl. negative-displacement DDA marches, "
              f"sealed/breach corners, WSUM-near-floor renorm, n_sub up to 8, "
              f"degenerate 1xN/Nx1).")
    return ok


def part2_trajectory() -> bool:
    print("PART 2 — blast + venting trajectory (real engine, per-tick digest):")
    from pathlib import Path

    from config import CFG
    from level_loader import LevelData
    from simulation import atmosphere_fixed
    from simulation.gamemap import GameMap
    from simulation.gases import O2
    from simulation.physics_runner import PhysicsRunner

    H = W = 48
    # v1 tilemap vocabulary: 0 = outer space (vacuum), 1 = hull wall,
    # 4 = interior air. A vacuum band, a hull ring, interior air, and a
    # 4-tile breach carved through the east hull — sustained venting.
    tm = np.zeros((H, W), dtype=np.int32)
    tm[2:46, 2:46] = 1
    tm[3:45, 3:45] = 4
    tm[22:26, 45] = 4          # the breach: hull ring opened to the vacuum band
    level = LevelData(name="eos_p62_blast_vent", version="1", path=Path("."),
                      tilemap=tm, tile_size_m=1.0 / 3.0,   # ship tile scale
                      diffuse_path=Path("."))
    g = GameMap(level)
    g.stamp_units([])
    assert g.is_vacuum.any(), "scenario must have vacuum to vent into"

    # THE BLAST: a hot core (raises p* hard -> outward shock through the room)
    # + an O2 overpressure pocket (density spike venting toward the breach).
    q = atmosphere_fixed.quantize_scalar
    g.temperature[10:16, 10:16] += q(5000.0)
    g.gas[O2, 11:14, 11:14] += q(4.0)

    runner = PhysicsRunner(bp)
    runner.eos.dx = float(g.tile_size_m)
    eos = runner.engine.eos
    inert_n2_idx = int(g.gases.name_to_id["inert_n2"])
    dt = 1.0 / float(CFG.clock.ticks_per_second)

    n_ticks = 80
    max_n_sub = 0
    max_u_counts = 0
    bad = 0
    for tick in range(n_ticks):
        # Snapshot the eos.step step-1-entry state (run_substeps calls
        # eos.step FIRST; step 0 copies P only — u/T enter advection as-is).
        wx0 = np.ascontiguousarray(g.wind_x.copy())
        wy0 = np.ascontiguousarray(g.wind_y.copy())
        t0 = np.ascontiguousarray(g.temperature.copy())

        runner.engine.run_substeps(
            g.wave_p, g.atmosphere,
            g.wind_x, g.wind_y,
            g.temperature, g.gas_energy,   # arc #54 §2.2 (MECHANICAL)
            g.obstacles, g.solid, g.is_vacuum,
            g.dyn_permeability, g.dyn_wave_absorb,
            g.gas, g.gases.diffusion, g.gases.conservative,
            g.gases.decay, inert_n2_idx,
            dt,
        )
        n_sub = int(eos.dbg_last_n_sub)
        max_n_sub = max(max_n_sub, n_sub)
        max_u_counts = max(max_u_counts,
                           int(np.abs(wx0).max()), int(np.abs(wy0).max()))

        inp = {"wind_x": wx0, "wind_y": wy0, "temperature": t0,
               "solid": g.solid, "is_vacuum": g.is_vacuum,
               "perm": g.dyn_permeability}
        f_ref, dig_ref, f_gpu, dig_gpu = _run_pair(inp, dt, n_sub)

        # P-E1 (energy-books design SS2.1.1/SS2.1.6, AUTHORIZED REWRITE —
        # Appendix A): the "ref_digest == EOSSolver.digest_advect" leg is
        # DELETED. The premise it rested on ("the interleaved bulk flux
        # neither reads nor writes u/T, so the advection substeps can be
        # replayed back to back") died with the law change: step 1d now
        # RECOVERS temperature from the energy books every substep, and
        # digest_advect moved across the flux call to hash that recovered T.
        # An isolated SL replay cannot reproduce it and must not pretend to.
        # What survives — and is what this gate was always really for — is
        # CPU-reference vs GPU-twin bit-identity of the U advection, on the
        # real trajectory's per-tick step-1-entry snapshots.
        if dig_gpu != dig_ref:
            bad += 1
            print(f"  tick {tick}: GPU != CPU digest "
                  f"(gpu={dig_gpu:#018x} ref={dig_ref:#018x} n_sub={n_sub})")
        for k in ("wind_x", "wind_y", "temperature"):
            if not np.array_equal(f_ref[k], f_gpu[k]):
                bad += 1
                mism = int(np.count_nonzero(f_ref[k] != f_gpu[k]))
                print(f"  tick {tick}: {k} {mism} byte mismatch(es)")
        # P-E1 positive assertion: SL advection is U-ONLY now. BOTH twins must
        # leave `temperature` byte-for-byte as handed in — if either ever
        # writes it again, the retired mint is back.
        for k, twin in (("ref", f_ref), ("gpu", f_gpu)):
            if not np.array_equal(twin["temperature"], t0):
                bad += 1
                mism = int(np.count_nonzero(twin["temperature"] != t0))
                print(f"  tick {tick}: {k} SL twin WROTE temperature "
                      f"({mism} cells) — the retired T-copy is back")
        if bad >= 10:
            print("  aborting after 10 divergences")
            break

    ok = (bad == 0)
    # The scenario must actually drive advection HARD — a quiescent trajectory
    # would make this gate vacuous.
    if max_n_sub < int(eos.N_SUB_MAX):
        ok = False
        print(f"  scenario too tame: max n_sub {max_n_sub} never hit "
              f"N_SUB_MAX={int(eos.N_SUB_MAX)}")
    if max_u_counts < 30 * FP_ONE:
        ok = False
        print(f"  scenario too tame: peak |u| {max_u_counts / FP_ONE:.1f} m/s "
              f"< 30 m/s")
    if ok:
        print(f"  {n_ticks} ticks bit-identical (per-tick u-only digest, CPU "
              f"ref == GPU, temperature untouched by both; peak |u| = "
              f"{max_u_counts / FP_ONE:.1f} m/s, n_sub pinned at {max_n_sub}).")
    return ok


def part3_golden() -> bool:
    print("PART 3 — CUDA build's CPU path vs the committed golden:")
    from field_ab_harness import capture_trajectory
    from field_digest import trajectory_digest

    # The committed default-scenario golden (see cuda_s4a_check.py history;
    # last re-baselined 2026-07-10, eos-p3fix-thermal-ceiling).
    # P-R4 GOLDEN REBASE (2026-08-01, the arc's ONE deliberate rebase —
    # ruling amendment 5 D2, Erik's approval). The canonical A/B scenario seeds
    # fire at (8,8)/(8,9) on AIR tiles (material 0, heat_atten 0,
    # flammable.sum() == 0) — a GHOST fire whose only observable was the retired
    # painter's air deposit. Under Kirchhoff a body that cannot absorb cannot
    # emit (a_s == 0), so that heat is now correctly ZERO and every trajectory
    # carrying it moves. Folded into the SAME one-shot rebase: D1's demand
    # accumulator (digest spec v2 -> v3, +dem_acc), D3's radiant-flux sensor and
    # D4's per-tick fan rotation. ONE approved change-set, ONE rebase event.
    # P-O2b GOLDEN REBASE (2026-08-02) - the fire-realism arc's OWN single
    # deliberate rebase (design v5.2 section 5: "this arc carries its own
    # single deliberate rebase"; the arc-local golden the design budgets).
    # THE EXTENDED OXYGEN DRAW (Erik's Option 2b) widens `dem_acc` from the 4
    # faces to the 2*R*(R+1) SOURCE OFFSETS within BFS hop-radius DRAW_R -
    # (12, h, w) at the shipped DRAW_R = 2. The shape rides the hashed
    # per-field header, so this is a DIGEST-SPEC VERSION BUMP (v3 -> v4) taken
    # per tests/field_digest_spec.toml's own change procedure, with every
    # committed golden regenerated in the same commit.
    # The A/B scenario carries no flammable tiles, so the LAW itself moves
    # nothing here: the entire delta is dem_acc's layout. That is deliberate
    # and separately gated - at DRAW_R = 1 the offset table's ring 1 IS D4's
    # order, so the plane is bit-for-bit the v3 plane and the full engine
    # reproduces every pre-patch field, byte for byte, over 45 ticks.
    # (was e73f130ea6f514fc285825d1efc828202bfc7e2e77dee3212bed2aa822e45f8a)
    # SINGLE-SOURCED 2026-08-18: was a hardcoded copy of the golden.
    # 11 scripts each carried their own, so ONE deliberate re-baseline
    # left 11 tests red. The sanctioned golden is OWNED by
    # tests/_xarch_perfield_digest.py (its lineage block carries every
    # rebase + rationale); import it, per test_w6_armory.py's own rule.
    from _xarch_perfield_digest import GOLDEN_AGGREGATE as GOLDEN
    base = capture_trajectory(n_steps=30)
    dig = trajectory_digest(base)
    # EXPECTED RED until P-G3 re-baseline (#54): physics moved under
    # P-G1a/P-G1b/P-G1d/P-G2 (stored gas_energy, the face-flux energy step,
    # the D4 divergence face form) — golden regen is P-G3's job, not this
    # patch's (P-G2b is test-tooling only). Left asserting, not loosened.
    if dig != GOLDEN:
        print(f"  GOLDEN MISMATCH: {dig[:16]}... != {GOLDEN[:16]}...")
        return False
    print(f"  CUDA build CPU path reproduces the golden ({dig[:12]}...).")
    return True


def main() -> int:
    if not getattr(bp, "HAS_CUDA", False) or not bp.cuda_available():
        print("P62_RESULT: FAIL (no CUDA build / device)")
        return 1
    print("device:", bp.cuda_device_info())
    p1 = part1_isolated()
    p2 = part2_trajectory()
    p3 = part3_golden()
    if p1 and p2 and p3:
        print("P62_RESULT: PASS")
        return 0
    print("P62_RESULT: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
