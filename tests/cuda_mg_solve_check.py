"""EOS P6.3 — multigrid Helmholtz pressure solve bit-identity check (runs
inside the GPU subprocess).

Three gates:

  PART 1 — ISOLATED (synthetic, all branches + edges + overflow stress):
  hand-crafted solve inputs (pstar, div_u, n_total, p_prev + masks/perm) that
  hit every branch of the port — the full 9-level 160x160 pyramid (fused-tail
  entry at level 3), deep-tail smaller grids, the flat RB-GS path (both
  use_multigrid=False and the n_levels==1 degenerate: 1x1, 1xN, Nx1), 1-cell
  rooms inside solid rings, all-vacuum and all-solid grids, near-N_FLOOR
  conductance bands, and the review §1.8 overflow-stress regime (blast-scale
  |P| ~6,500 atm against floored-N̂ faces — the int64-edge g x dP products the
  128-bit staging exists for). Run BOTH the GPU V-cycle (bp.cuda_eos_mg_solve
  — hierarchy built host-side through the SAME EOSSolver::mg_build_levels the
  CPU calls, ENTIRE iteration on device) and the CPU reference
  (bp.eos_mg_solve_ref — the SAME internal routines step() calls) on
  identical inputs and assert byte-for-byte equality of the solved P + digest
  equality. Also asserts the fused coarse tail actually collapses the launch
  count (levels <= 1024 cells run in ONE single-block kernel).

  PART 2 — TRAJECTORY (the review's §4 P6.3 digest gate): a breach-to-vacuum
  + blast scenario (hot core + O2 overpressure in a hull-ringed room breached
  to vacuum — steep N gradients at the breach, warm-start reuse across ticks,
  deep V-cycles) driven through the REAL engine path
  (PhysicsEngine.run_substeps -> EOSSolver::step) for 120 ticks on the CPU.
  Per tick: run the real tick, reconstruct the EXACT solve inputs
  (eos.dbg_mg_inputs() = the pstar/div_u/n_total caches as the solve consumed
  them; the engine's p_prev buffer = the warm start), replay the isolated
  solve through BOTH the CPU reference and the GPU V-cycle, asserting
      ref_digest == EOSSolver.digest_helmholtz == gpu_digest
  and byte-equality of the solved P against BOTH the GPU result and the
  engine's own post-tick atmosphere (P_new — exactly the field the step-4
  velocity kick reads). A full per-tick digest trajectory, CPU vs GPU, over
  the whole run; the scenario is asserted to actually stress the solve
  (vacuum venting present, blast-scale pressure excursions, the full pyramid
  depth).

  PART 3 — the CUDA build's CPU path still reproduces the committed
  default-scenario golden (the s4a-check idiom; proves the P6.3 additions
  changed no CPU trajectory).

Prints ``P63_RESULT: PASS``/``FAIL`` and exits 0/1.
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


def _solver(dx=1.0 / 3.0, use_multigrid=True):
    s = bp.EOSSolver()
    s.dx = dx
    s.use_multigrid = use_multigrid
    return s


def _run_pair(solver, inp, dt):
    """Run CPU reference + GPU V-cycle on identical inputs; return all."""
    h, w = inp["pstar"].shape
    p_ref = np.zeros((h, w), dtype=np.int32)
    p_gpu = np.zeros((h, w), dtype=np.int32)
    dig_ref = bp.eos_mg_solve_ref(
        solver, inp["pstar"], inp["div_u"], inp["n_total"], inp["p_prev"],
        inp["solid"], inp["is_vacuum"], inp["perm"], dt, p_ref)
    dig_gpu, la, ln = bp.cuda_eos_mg_solve(
        solver, inp["pstar"], inp["div_u"], inp["n_total"], inp["p_prev"],
        inp["solid"], inp["is_vacuum"], inp["perm"], dt, p_gpu)
    return p_ref, dig_ref, p_gpu, dig_gpu, la, ln


def _rand_inputs(rng, h, w, p_mag=2.0, div_mag=50.0, n_lo=0.2, n_hi=3.0,
                 solid_frac=0.10, vac_frac=0.08):
    """Random mixed solve inputs (blast-scale knobs via the magnitudes)."""
    n = h * w

    def field(mag, signed=True):
        v = rng.random(n) * mag
        if signed:
            v *= np.where(rng.random(n) < 0.5, -1.0, 1.0)
        return _quantize(v).reshape(h, w)

    pstar = np.abs(field(p_mag))                      # p* >= 0 (EOS floor)
    div_u = field(div_mag)
    n_total = _quantize(n_lo + rng.random(n) * (n_hi - n_lo)).reshape(h, w)
    p_prev = field(p_mag)                             # warm start, +/-
    solid = (rng.random(n) < solid_frac).reshape(h, w)
    is_vacuum = (rng.random(n) < vac_frac).reshape(h, w)
    perm = rng.random(n).astype(np.float32)
    perm[rng.random(n) < 0.10] = 0.0                  # sealed faces
    perm[rng.random(n) < 0.30] = 1.0
    return {
        "pstar": np.ascontiguousarray(pstar),
        "div_u": np.ascontiguousarray(div_u),
        "n_total": np.ascontiguousarray(n_total),
        "p_prev": np.ascontiguousarray(p_prev),
        "solid": np.ascontiguousarray(solid),
        "is_vacuum": np.ascontiguousarray(is_vacuum),
        "perm": np.ascontiguousarray(perm.reshape(h, w)),
    }


def _ambient_inputs(h, w):
    """Quiet ambient room: p* = P_prev = 1 atm, N = 1, no divergence."""
    one = np.full((h, w), FP_ONE, dtype=np.int32)
    return {
        "pstar": one.copy(), "div_u": np.zeros((h, w), np.int32),
        "n_total": one.copy(), "p_prev": one.copy(),
        "solid": np.zeros((h, w), bool), "is_vacuum": np.zeros((h, w), bool),
        "perm": np.ones((h, w), np.float32),
    }


def part1_isolated() -> bool:
    print("PART 1 — isolated GPU vs CPU reference (synthetic/edge/stress):")
    ok = True
    rng = np.random.default_rng(20260711)
    dt = 1.0 / 24.0
    cases = []

    # -- random mixed grids (three seeds each shape; odd dims included) ------
    for (h, w) in ((16, 16), (17, 31), (48, 48), (33, 20)):
        for _ in range(3):
            cases.append((f"random {h}x{w}", _solver(), _rand_inputs(rng, h, w)))

    # -- degenerate shapes: n_levels == 1 -> the flat RB-GS branch -----------
    cases.append(("1x1 open cell", _solver(), _ambient_inputs(1, 1)))
    cases.append(("1x40 row", _solver(), _rand_inputs(rng, 1, 40)))
    cases.append(("40x1 col", _solver(), _rand_inputs(rng, 40, 1)))

    # -- explicit flat path (use_multigrid=False, the A/B reference) ---------
    cases.append(("flat 32x32 (use_multigrid=False)",
                  _solver(use_multigrid=False), _rand_inputs(rng, 32, 32)))

    # -- 1-cell room inside a solid ring --------------------------------------
    inp = _ambient_inputs(3, 3)
    inp["solid"][:] = True
    inp["solid"][1, 1] = False
    inp["pstar"][1, 1] = _quantize(np.array(4.0))
    inp["p_prev"][1, 1] = _quantize(np.array(3.5))
    cases.append(("1-cell room in solid ring", _solver(), inp))

    # -- all-vacuum / all-solid ------------------------------------------------
    inp = _ambient_inputs(16, 16)
    inp["is_vacuum"][:] = True
    cases.append(("all-vacuum", _solver(), inp))
    inp = _ambient_inputs(12, 12)
    inp["solid"][:] = True
    cases.append(("all-solid", _solver(), inp))

    # -- near-N_FLOOR everywhere (max conductance faces) ----------------------
    inp = _rand_inputs(rng, 24, 24, p_mag=1.0, div_mag=10.0)
    inp["n_total"][:] = _quantize(np.array(1e-3))   # == the solver floor
    cases.append(("near-N_FLOOR field", _solver(), inp))

    # -- the full game-size pyramid (9 levels; fused tail from level 3) -------
    big = _ambient_inputs(160, 160)
    big["is_vacuum"][:, :4] = True                   # vacuum band (Dirichlet)
    big["solid"][:, 4] = True                        # hull line
    big["solid"][70:90, 4] = False                   # breach: room open to vacuum
    big["pstar"][60:100, 60:100] = _quantize(np.array(8.0))   # hot core
    big["p_prev"][60:100, 60:100] = _quantize(np.array(6.0))
    big["div_u"][70:90, 70:90] = _quantize(np.array(200.0))
    cases.append(("160x160 breach+core (9 levels)", _solver(), big))

    # -- overflow-stress (review §1.8 / §3.4 budget regime): blast-scale
    #    |P| ~6,500 atm against floored-N̂ faces — the deep-level g x dP
    #    int64-edge products the 128-bit staging exists for. ------------------
    st = _rand_inputs(rng, 160, 160, p_mag=6500.0, div_mag=3000.0,
                      n_lo=1e-3, n_hi=8.0, solid_frac=0.05, vac_frac=0.05)
    st["n_total"][40:120, :] = _quantize(np.array(1e-3))  # floored-N̂ band
    st["perm"][:] = 1.0                                    # max conductance
    cases.append(("overflow-stress 160x160", _solver(), st))

    la160 = ln160 = None
    n_case = 0
    for (name, solver, inp) in cases:
        n_case += 1
        p_ref, dig_ref, p_gpu, dig_gpu, la, ln = _run_pair(solver, inp, dt)
        if name.startswith("160x160 breach"):
            la160, ln160 = la, ln
        if dig_ref != dig_gpu:
            ok = False
            print(f"  {name}: digest mismatch "
                  f"(ref={dig_ref:#018x} gpu={dig_gpu:#018x})")
        if not np.array_equal(p_ref, p_gpu):
            ok = False
            mism = int(np.count_nonzero(p_ref != p_gpu))
            idx = int(np.argmax(p_ref != p_gpu))
            print(f"  {name}: P {mism} MISMATCH (first @ {idx}: "
                  f"cpu={p_ref.flat[idx]} gpu={p_gpu.flat[idx]})")

    # The fused coarse tail must actually collapse the launch count on the
    # full pyramid (review §2.2: ~238 of ~304 launches removed).
    if la160 is None or ln160 is None or not (la160 < ln160 - 200):
        ok = False
        print(f"  fused-tail launch collapse missing: actual={la160} "
              f"naive={ln160}")
    else:
        print(f"  fused coarse tail: {la160} launches/solve vs {ln160} naive "
              f"(160x160, 9 levels, V(2,2)x2 + 32 coarsest sweeps).")
    if ok:
        print(f"  all {n_case} configs bit-identical on solved P, digests "
              f"equal (incl. flat path, 1x1/1xN/Nx1, solid-ring 1-cell room, "
              f"all-vacuum/all-solid, near-N_FLOOR, 9-level pyramid, "
              f"overflow stress).")
    return ok


def part2_trajectory() -> bool:
    print("PART 2 — breach-to-vacuum + blast trajectory (real engine, "
          "per-tick digest):")
    from pathlib import Path

    from config import CFG
    from level_loader import LevelData
    from simulation import atmosphere_fixed
    from simulation.gamemap import GameMap
    from simulation.gases import O2
    from simulation.physics_runner import PhysicsRunner

    H = W = 96
    # v1 tilemap vocabulary: 0 = outer space (vacuum), 1 = hull wall,
    # 4 = interior air. A vacuum band, a hull ring, interior air, and a
    # 4-tile breach carved through the east hull — sustained venting into
    # vacuum (steep N gradients at the breach; 96x96 -> an 8-level pyramid,
    # fused-tail entry at level 2).
    tm = np.zeros((H, W), dtype=np.int32)
    tm[2:94, 2:94] = 1
    tm[3:93, 3:93] = 4
    tm[46:50, 93] = 4          # the breach: hull ring opened to the vacuum band
    level = LevelData(name="eos_p63_blast_vent", version="1", path=Path("."),
                      tilemap=tm, tile_size_m=1.0 / 3.0,   # ship tile scale
                      diffuse_path=Path("."))
    g = GameMap(level)
    g.stamp_units([])
    assert g.is_vacuum.any(), "scenario must have vacuum to vent into"

    # THE BLAST: a hot core (raises p* hard -> outward shock through the room)
    # + an O2 overpressure pocket (density spike venting toward the breach).
    q = atmosphere_fixed.quantize_scalar
    g.temperature[20:32, 20:32] += q(5000.0)
    g.gas[O2, 22:28, 22:28] += q(4.0)

    runner = PhysicsRunner(bp)
    runner.eos.dx = float(g.tile_size_m)
    eos = runner.engine.eos
    inert_n2_idx = int(g.gases.name_to_id["inert_n2"])
    dt = 1.0 / float(CFG.clock.ticks_per_second)

    n_ticks = 120
    max_p_dev = 0
    max_pstar = 0
    max_n_sub = 0
    bad = 0
    for tick in range(n_ticks):
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
        dig_cpu = int(eos.digest_helmholtz)

        # Reconstruct the EXACT solve inputs: the dbg caches as the solve
        # consumed them (nothing after the solve writes them) + the engine's
        # p_prev buffer (g.wave_p — step 0 copied the tick-entry P into it:
        # THE warm start, carrying real cross-tick reuse).
        ps, dv, nt = eos.dbg_mg_inputs()
        inp = {
            "pstar": np.ascontiguousarray(ps.reshape(H, W)),
            "div_u": np.ascontiguousarray(dv.reshape(H, W)),
            "n_total": np.ascontiguousarray(nt.reshape(H, W)),
            "p_prev": np.ascontiguousarray(g.wave_p.copy()),
            "solid": g.solid, "is_vacuum": g.is_vacuum,
            "perm": g.dyn_permeability,
        }
        p_ref, dig_ref, p_gpu, dig_gpu, _la, _ln = _run_pair(eos, inp, dt)
        max_p_dev = max(max_p_dev, int(np.abs(
            p_ref.astype(np.int64) - FP_ONE).max()))
        max_pstar = max(max_pstar, int(inp["pstar"].max()))
        max_n_sub = max(max_n_sub, int(eos.dbg_last_n_sub))

        # The CPU reference must reproduce the REAL solver's digest_helmholtz
        # — proves both the input reconstruction and the reference itself.
        if dig_ref != dig_cpu:
            bad += 1
            print(f"  tick {tick}: CPU ref != solver digest_helmholtz "
                  f"(ref={dig_ref:#018x} solver={dig_cpu:#018x})")
        # The GPU V-cycle must be bit-identical to the reference (and hence
        # to the solver's own solve bytes).
        if dig_gpu != dig_ref:
            bad += 1
            print(f"  tick {tick}: GPU != CPU digest "
                  f"(gpu={dig_gpu:#018x} ref={dig_ref:#018x})")
        if not np.array_equal(p_ref, p_gpu):
            bad += 1
            mism = int(np.count_nonzero(p_ref != p_gpu))
            print(f"  tick {tick}: solved P {mism} byte mismatch(es)")
        # And the replayed P must BE the engine's post-tick P_new (the
        # atmosphere materialization the step-4 velocity kick read).
        if not np.array_equal(p_ref, g.atmosphere):
            bad += 1
            print(f"  tick {tick}: replayed P != engine atmosphere")
        if bad >= 10:
            print("  aborting after 10 divergences")
            break

    ok = (bad == 0)
    # The scenario must actually stress the solve — a quiescent trajectory
    # would make this gate vacuous. NOTE the hardness is asserted on the
    # solve INPUTS: the solved P itself stays near ambient BY DESIGN (the
    # Kwatra projection's rhs subtracts γ·p*·dt·div(u*) — the blast's
    # expansion velocity cancels its own overpressure; the acoustic
    # equilibration is the method's whole point), so a blast shows up as a
    # huge p* against a near-ambient P_new, not as a P spike.
    if max_pstar < 20 * FP_ONE:
        ok = False
        print(f"  scenario too tame: peak p* = {max_pstar / FP_ONE:.1f} atm "
              f"< 20 atm")
    if max_n_sub < int(eos.N_SUB_MAX):
        ok = False
        print(f"  scenario too tame: max n_sub {max_n_sub} never hit "
              f"N_SUB_MAX={int(eos.N_SUB_MAX)}")
    if ok:
        print(f"  {n_ticks} ticks bit-identical (per-tick digest_helmholtz == "
              f"CPU ref == GPU; P_new == engine atmosphere; peak p* = "
              f"{max_pstar / FP_ONE:.1f} atm vs peak |P-1atm| = "
              f"{max_p_dev / FP_ONE:.2f}; n_sub pinned at {max_n_sub}; warm "
              f"start reused every tick).")
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
    if dig != GOLDEN:
        print(f"  GOLDEN MISMATCH: {dig[:16]}... != {GOLDEN[:16]}...")
        return False
    print(f"  CUDA build CPU path reproduces the golden ({dig[:12]}...).")
    return True


def main() -> int:
    if not getattr(bp, "HAS_CUDA", False) or not bp.cuda_available():
        print("P63_RESULT: FAIL (no CUDA build / device)")
        return 1
    print("device:", bp.cuda_device_info())
    p1 = part1_isolated()
    p2 = part2_trajectory()
    p3 = part3_golden()
    if p1 and p2 and p3:
        print("P63_RESULT: PASS")
        return 0
    print("P63_RESULT: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
