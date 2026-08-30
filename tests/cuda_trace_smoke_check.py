"""EOS P6.7 trace-smoke advection bit-identity check (runs inside the GPU subprocess).

The P6.7 per-kernel digest gate (docs/eos_p6_gpu_alignment_review.md §4, P6.7
row: trace-smoke re-port at the new once-per-tick cadence; resolves the P3
``physics_engine.cpp`` cadence assert). THE RE-DERIVATION FINDING: the EOS
refactor changed only the trace CADENCE — traces now advect ONCE per tick on the
solver's final corrected wind, instead of n_smoke-substepped on the old wave
loop's wind — while ``SmokeDynamics::step``'s per-pass arithmetic (the
permeability-weighted diffusion Laplacian, the post-diffusion src snapshot, the
INTEGER semi-Lagrangian back-trace + DDA wall-clip + Newton-reciprocal renorm,
the clamp/zero) is UNCHANGED. So cuda_smoke.cu's ``smoke_step`` (the verbatim S4a
device mirror) is already bit-identical at the new cadence; P6.7 wires the real
GPU dispatch at the moved call site (the water/smoke/fire/eos dispatch idiom) and
re-proves it here.

Three gates:

  PART 1 — ISOLATED (all branches): rich synthetic inputs hitting every branch of
  the 4-pass solver — the permeability-weighted diffusion Laplacian (per-face perm
  float bridge incl. sealed faces), the wind^2 diffusion fold (wind_diffusion_scale
  > 0), the INTEGER SL advection (BOTH-sign high-magnitude wind -> multi-cell +
  NEGATIVE-displacement back-traces + the DDA wall-clip march), wall/vacuum/obstacle
  masks (zeroing + breach venting + sealed-corner exclusion), the WSUM-near-floor
  reciprocal renorm — PLUS the P6.7-named degenerate configs: 1xN / Nx1 grids,
  ALL-SOLID and ALL-VACUUM grids (everything zeroed), and NEAR-EMPTY planes (a
  single non-zero cell). ``bp.cuda_smoke_step`` vs the shipped ``bp.SmokeDynamics()
  .step`` on identical copies, byte-for-byte equality on the gas plane (tol 0).

  PART 2 — TRAJECTORY (blast + venting, REAL engine, once-per-tick cadence): a
  hull-ringed two-room scene breached to hard vacuum, with a hot blast core + an
  O2 overpressure pocket driving a real corrected wind, and a non-uniform black-
  smoke cloud seeded to advect across the rooms and vent out the breach. Driven
  through the REAL engine path (``run_substeps``) for 120 ticks TWICE on two
  independently built worlds: once with the smoke backend OFF (CPU
  ``SmokeDynamics::step``) and once ON (GPU ``smoke_step``). All EOS fields evolve
  on the CPU in BOTH runs (only the smoke backend flips), so the once-per-tick
  trace advection is the surface under test. Per tick, asserts byte-identity of
  every gas plane (bulk AND traces) + wind + temperature. Scenario-hardness guards
  (so the gate can never go vacuous): the smoke must actually advect (move off its
  seed), and it must actually VENT (total trace mass strictly drops as it leaves
  through the breach).

  PART 3 — the CUDA build's CPU path (smoke backend off) still reproduces the
  committed default-scenario golden (the s4a-check idiom; proves the P6.7 dispatch
  is strictly additive — no CPU trajectory changed; the P4 decay->inert_N2 credit
  stays CPU in both paths).

Prints ``TRACE_SMOKE_RESULT: PASS``/``FAIL`` and exits 0/1.
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


# ---------------------------------------------------------------------------
# PART 1 — isolated all-branch synthetic A/B (bp.cuda_smoke_step vs CPU step)
# ---------------------------------------------------------------------------
def _make_inputs(rng, h, w, wind_mag, force_floor_wsum):
    """Synthetic smoke state exercising every advection + diffusion branch (the
    s4a generator: both-sign multi-cell wind, sealed/open faces, breach corners,
    WSUM-near-floor renorm)."""
    n = h * w
    sm = rng.random(n).astype(np.float64)
    sm[rng.random(n) < 0.25] = 0.0
    sm[rng.random(n) < 0.10] = 1.0
    smoke = _quantize(sm).reshape(h, w)

    wx_m = (rng.random(n) * 2.0 - 1.0) * wind_mag
    wy_m = (rng.random(n) * 2.0 - 1.0) * wind_mag
    wind_x = _quantize(wx_m).reshape(h, w)
    wind_y = _quantize(wy_m).reshape(h, w)

    obstacles = (rng.random(n) < 0.12).reshape(h, w)
    is_wall = (rng.random(n) < 0.10).reshape(h, w)
    is_vacuum = (rng.random(n) < 0.08).reshape(h, w)

    perm = rng.random(n).astype(np.float32)
    perm[rng.random(n) < 0.15] = 0.0
    perm[rng.random(n) < 0.30] = 1.0
    permeability = perm.reshape(h, w)

    if force_floor_wsum and h >= 5 and w >= 5:
        cy, cx = h // 2, w // 2
        is_wall[cy, cx] = False
        is_vacuum[cy, cx] = False
        obstacles[cy, cx] = False
        permeability[cy, cx] = 1.0
        smoke[cy, cx] = _quantize(np.array(0.8))
        wind_x[cy, cx] = _quantize(np.array(0.02))
        wind_y[cy, cx] = _quantize(np.array(0.02))
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


def _make_edge_inputs(kind, h, w):
    """The P6.7-named degenerate configs: all-solid, all-vacuum, near-empty."""
    n = h * w
    smoke = np.zeros((h, w), dtype=np.int32)
    wind_x = _quantize(np.full(n, 1.2).reshape(h, w))   # strong wind -> deep march
    wind_y = _quantize(np.full(n, -0.9).reshape(h, w))
    obstacles = np.zeros((h, w), dtype=bool)
    is_wall = np.zeros((h, w), dtype=bool)
    is_vacuum = np.zeros((h, w), dtype=bool)
    permeability = np.ones((h, w), dtype=np.float32)

    if kind == "all_solid":
        obstacles[:] = True
        is_wall[:] = True
        permeability[:] = 0.0
        smoke[:] = _quantize(np.array(0.5))     # will be zeroed by the clamp pass
    elif kind == "all_vacuum":
        is_vacuum[:] = True
        smoke[:] = _quantize(np.array(0.5))     # zeroed
    elif kind == "near_empty":
        # a single non-zero cell in an otherwise-clear open plane
        smoke[h // 2, w // 2] = _quantize(np.array(0.9))
    elif kind == "empty":
        pass                                    # all-zero plane
    else:
        raise ValueError(kind)

    return {
        "smoke": np.ascontiguousarray(smoke),
        "wind_x": np.ascontiguousarray(wind_x.astype(np.int32)),
        "wind_y": np.ascontiguousarray(wind_y.astype(np.int32)),
        "obstacles": np.ascontiguousarray(obstacles),
        "is_wall": np.ascontiguousarray(is_wall),
        "is_vacuum": np.ascontiguousarray(is_vacuum),
        "permeability": np.ascontiguousarray(permeability),
    }


def _ab_one(inp, dt, d_smoke, wds, adv):
    """Return (sm_cpu, sm_gpu) after the CPU and GPU single trace step."""
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
    return sm_cpu, sm_gpu


def part1_isolated() -> bool:
    print("PART 1 — isolated GPU vs CPU (synthetic, all branches + edge configs):")
    ok = True
    rng = np.random.default_rng(20260711)
    # (h, w, dt, wind_mag, wind_diffusion_scale, advection_rate, force_floor_wsum)
    # advection_rate here sweeps 1/dx-scale values (the engine sets adv = 1/dx at
    # the new cadence — e.g. 3.0 for the 1/3 m ship tile) plus larger ones.
    configs = [
        (16, 16, 0.02, 0.0,  0.0,  3.0,   False),  # zero wind (identity advection)
        (16, 16, 0.02, 0.5,  0.0,  3.0,   False),  # moderate wind, no wind-diffusion
        (16, 16, 0.02, 1.2,  3.0,  3.0,   True),    # wind-diffusion ON + floor wsum
        (24, 32, 0.03, 2.0,  1.5,  3.0,   True),    # bigger, strong wind, multi-cell
        (31, 17, 0.015, 1.5, 5.0,  6.0,   True),    # odd dims, high wind-diffusion
        (40, 40, 0.02, 3.0,  0.0,  3.0,   False),   # very strong wind (deep march)
        (12, 20, 0.04, 0.8,  2.0,  2.0,   True),
        (1, 50, 0.02, 1.5,  1.0,  3.0,   False),    # degenerate 1-row
        (50, 1, 0.02, 1.5,  1.0,  3.0,   False),    # degenerate 1-col
        (8, 8, 0.02, 1.0,  4.0,  3.0,   True),
    ]
    d_smoke_values = [0.1, 0.4, 0.0]
    n_cfg = 0
    for (h, w, dt, wmag, wds, adv, floor_w) in configs:
        for seed_bump in range(5):
            for d_smoke in d_smoke_values:
                n_cfg += 1
                inp = _make_inputs(rng, h, w, wmag, floor_w)
                sm_cpu, sm_gpu = _ab_one(inp, dt, d_smoke, wds, adv)
                if not np.array_equal(sm_cpu, sm_gpu):
                    ok = False
                    mism = int(np.count_nonzero(sm_cpu != sm_gpu))
                    idx = int(np.argmax(sm_cpu != sm_gpu))
                    print(f"  {h}x{w} dt={dt} wmag={wmag} wds={wds} adv={adv} "
                          f"d_smoke={d_smoke}: gas {mism} MISMATCH (first @ {idx}: "
                          f"cpu={sm_cpu.flat[idx]} gpu={sm_gpu.flat[idx]})")

    # The P6.7-named degenerate/edge configs.
    edge_configs = [
        ("all_solid",  12, 12),
        ("all_solid",  1, 40),
        ("all_vacuum", 12, 12),
        ("all_vacuum", 40, 1),
        ("near_empty", 20, 20),
        ("near_empty", 1, 30),
        ("near_empty", 30, 1),
        ("empty",      16, 16),
    ]
    n_edge = 0
    for (kind, h, w) in edge_configs:
        for d_smoke in (0.4, 0.0):
            n_edge += 1
            inp = _make_edge_inputs(kind, h, w)
            sm_cpu, sm_gpu = _ab_one(inp, 0.02, d_smoke, 0.0, 3.0)
            if not np.array_equal(sm_cpu, sm_gpu):
                ok = False
                mism = int(np.count_nonzero(sm_cpu != sm_gpu))
                print(f"  edge {kind} {h}x{w} d_smoke={d_smoke}: gas {mism} MISMATCH")

    if ok:
        print(f"  all {n_cfg} synthetic + {n_edge} edge configs bit-identical on "
              f"the gas plane (incl. negative-displacement advection, the wind^2 "
              f"diffusion fold, the permeability bridge, wall/vacuum zeroing, "
              f"WSUM-near-floor renorm, 1xN/Nx1, all-solid, all-vacuum, near-empty).")
    return ok


# ---------------------------------------------------------------------------
# PART 2 — real-engine blast + venting trajectory (once-per-tick cadence)
# ---------------------------------------------------------------------------
def _build_scenario():
    """One independently constructed runner + map: a hull-ringed two-room scene
    breached to vacuum, a hot blast core + O2 overpressure pocket (real corrected
    wind), and a NON-UNIFORM black-smoke cloud that advects across the rooms and
    vents out the breach."""
    from pathlib import Path

    from config import CFG
    from level_loader import LevelData
    from simulation import atmosphere_fixed
    from simulation.gamemap import GameMap
    from simulation.gases import O2
    from simulation.physics_runner import PhysicsRunner

    H = W = 96
    # v1 tilemap vocabulary: 0 = outer space (vacuum), 1 = hull wall, 4 = air.
    # A vacuum band, a hull ring, interior air split into two rooms by an inner
    # wall with a doorway, and a 4-tile breach through the east hull.
    tm = np.zeros((H, W), dtype=np.int32)
    tm[2:94, 2:94] = 1
    tm[3:93, 3:93] = 4
    tm[3:93, 48] = 1           # inner dividing wall
    tm[44:52, 48] = 4          # a doorway between the two rooms
    tm[46:50, 93] = 4          # the breach: east hull opened to the vacuum band
    level = LevelData(name="eos_p67_smoke_vent", version="1", path=Path("."),
                      tilemap=tm, tile_size_m=1.0 / 3.0,
                      diffuse_path=Path("."))
    g = GameMap(level)
    g.stamp_units([])
    assert g.is_vacuum.any(), "scenario must have vacuum to vent into"

    q = atmosphere_fixed.quantize_scalar
    # The blast: a hot core (outward shock) + an O2 overpressure pocket (density
    # spike venting toward the breach) — both in the WEST room, so the wind pushes
    # smoke east through the doorway toward the breach.
    g.temperature[20:32, 20:32] += q(5000.0)
    g.gas[O2, 22:28, 22:28] += q(4.0)

    # A NON-UNIFORM black-smoke cloud in the west room (a gradient blob), so the
    # trace advection has real structure to carry across rooms and out the breach.
    trace_ids = [gi for gi in range(g.gas.shape[0])
                 if not bool(g.gases.conservative[gi])]
    assert trace_ids, "scenario needs a trace plane"
    smoke_id = trace_ids[0]
    yy, xx = np.mgrid[0:H, 0:W]
    blob = np.exp(-(((yy - 30) / 10.0) ** 2 + ((xx - 30) / 10.0) ** 2))
    interior = (~g.solid) & (~g.is_vacuum)
    cloud = np.where(interior, _quantize(0.9 * blob), 0).astype(np.int32)
    g.gas[smoke_id] += cloud

    runner = PhysicsRunner(bp)
    runner.eos.dx = float(g.tile_size_m)
    inert_n2_idx = int(g.gases.name_to_id["inert_n2"])
    dt = 1.0 / float(CFG.clock.ticks_per_second)
    return runner, g, inert_n2_idx, smoke_id, trace_ids, dt


def _tick(runner, g, inert_n2_idx, dt):
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


def part2_trajectory() -> bool:
    print("PART 2 — blast+venting REAL-engine trajectory, CPU smoke backend vs "
          "GPU smoke backend (per-tick byte-compare, once-per-tick cadence):")
    bp.set_smoke_backend(False)
    runner_cpu, g_cpu, n2_cpu, smoke_id, trace_ids, dt = _build_scenario()
    runner_gpu, g_gpu, n2_gpu, smoke_id2, _t2, dt2 = _build_scenario()
    assert dt == dt2 and smoke_id == smoke_id2

    fields = ("gas", "wind_x", "wind_y", "temperature", "atmosphere")
    for f in fields:
        assert np.array_equal(getattr(g_cpu, f), getattr(g_gpu, f)), \
            f"scenario construction not deterministic on {f}"

    smoke0 = g_cpu.gas[smoke_id].copy()
    total0 = int(smoke0.sum(dtype=np.int64))
    moved = False
    min_total = total0

    n_ticks = 120
    bad = 0
    for tick in range(n_ticks):
        bp.set_smoke_backend(False)
        _tick(runner_cpu, g_cpu, n2_cpu, dt)
        bp.set_smoke_backend(True)
        assert bp.get_smoke_backend(), "smoke backend did not switch to GPU"
        _tick(runner_gpu, g_gpu, n2_gpu, dt)
        bp.set_smoke_backend(False)   # restore

        for f in fields:
            a, b = getattr(g_cpu, f), getattr(g_gpu, f)
            if not np.array_equal(a, b):
                bad += 1
                mism = int(np.count_nonzero(a != b))
                idx = int(np.argmax(a != b))
                print(f"  tick {tick}: field {f}: {mism} MISMATCH(es) "
                      f"(first flat @ {idx}: cpu={a.flat[idx]} gpu={b.flat[idx]})")
        # scenario-hardness telemetry (CPU side)
        cur = g_cpu.gas[smoke_id]
        if int(np.abs(cur.astype(np.int64) - smoke0.astype(np.int64)).max()) > FP_ONE // 100:
            moved = True
        min_total = min(min_total, int(cur.sum(dtype=np.int64)))
        if bad >= 10:
            print("  aborting after 10 divergences")
            break

    ok = (bad == 0)

    # Scenario-hardness guards: the gate must actually exercise trace transport.
    if not moved:
        ok = False
        print("  scenario too tame: the smoke cloud never advected off its seed")
    if not (min_total < total0):
        ok = False
        print(f"  scenario too tame: trace mass never vented "
              f"(min total {min_total} !< seed total {total0})")

    if ok:
        drop = 100.0 * (1.0 - min_total / max(total0, 1))
        print(f"  {n_ticks} ticks bit-identical on every gas plane (bulk+trace) + "
              f"wind + T; the black-smoke cloud advected across the rooms and "
              f"vented out the breach (peak trace-mass drop {drop:.1f}%).")
    return ok


# ---------------------------------------------------------------------------
# PART 3 — the CUDA build's CPU path still reproduces the committed golden
# ---------------------------------------------------------------------------
def part3_golden() -> bool:
    print("PART 3 — CUDA build's CPU path (smoke backend off) vs the committed golden:")
    bp.set_smoke_backend(False)
    from field_ab_harness import capture_trajectory
    from field_digest import trajectory_digest

    # The committed default-scenario golden (last re-baselined 2026-07-10,
    # eos-p3fix-thermal-ceiling; see cuda_s4a_check.py / cuda_eos_step_check.py).
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
        print("TRACE_SMOKE_RESULT: FAIL (no CUDA build / device)")
        return 1
    print("device:", bp.cuda_device_info())
    p1 = part1_isolated()
    p2 = part2_trajectory()
    p3 = part3_golden()
    if p1 and p2 and p3:
        print("TRACE_SMOKE_RESULT: PASS")
        return 0
    print("TRACE_SMOKE_RESULT: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
