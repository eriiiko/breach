"""CUDA-S2 LIVE raycaster bit-identity check (runs inside the GPU subprocess).

S2 proved the GPU directional march's `heat` == the CPU cast byte-for-byte in an
ISOLATED harness (cuda_s2_check). This gate proves the LIVE WIRING preserves that:
the fire->heat ray cast that actually runs each tick — PhysicsRunner.cast_fire_heat,
the per-burning-tile source loop that CLEARS nothing (the per-tick heat clear is at
the end of Simulation.step) and ACCUMULATES each source's deposit into gmap.heat —
produces a byte-identical `heat` field whether cast_fire_heat dispatches to the CPU
(Raycaster.cast_source_directional) or the GPU (bp.cuda_raycaster_cast). It also
proves the full all-backends-on (field solvers + raycaster; 5/5 after the EOS P6.0 wave/atmos retirement) is bit-identical end-to-end
over a real 30-tick trajectory, including the synced `heat`/`temperature` fields,
and still reproduces the committed default-scenario golden.

Two parts:

  PART 1 — LIVE cast_fire_heat, multi-source, heat tol 0. Build a real GameMap
  with MANY burning tiles (so cast_fire_heat enumerates many LightSources and
  ACCUMULATES their saturating-add heat deposits into one shared gmap.heat), plus
  smoke/gas in the path (the gas optics `expf` lives on the RGB survival only and
  must NEVER perturb the heat-touched set) and heat_atten occluders. Run the SAME
  PhysicsRunner.cast_fire_heat with the raycaster backend OFF then ON; assert the
  resulting gmap.heat is byte-for-byte equal. This drives the production dispatch
  site, not a re-implementation — so it catches a wiring bug (wrong clear/accumulate,
  a dropped source, a buffer-aliasing slip) that the isolated S2 gate cannot.

  PART 2 — ALL-BACKENDS-ON INTEGRATION. The default A/B scenario (fire seeded -> cast_fire_heat
  runs) stepped 30 ticks with ALL backends ON vs ALL OFF (CPU). Assert the full
  per-tick trajectory of EVERY synced field (incl. heat + temperature, which the
  raycaster feeds) is bit-identical (diff_trajectories tol 0), and the CPU path
  still reproduces the golden. This is the proof that --cuda is fully GPU-dispatched.

Prints ``S2_LIVE_RESULT: PASS``/``FAIL`` and exits 0/1, plus the headline
``RAYCASTER_LIVE_RESULT: PASS``/``FAIL`` the task asks for.
"""
from __future__ import annotations

import sys

import numpy as np

# Import the CUDA build FIRST so it is the cached `breach_physics` before
# field_ab_harness / level_loader (which insert cpp/build/Release on sys.path)
# import it.
import breach_physics as bp

INT32_MAX = (1 << 31) - 1

# The committed default-scenario golden (CUDA-S2 re-baseline, 2026-06-28) — the
# raycaster being live on the GPU must NOT change the CPU-path digest.
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
# smoke's decay both move the trajectory. Combustion itself (item 1)
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


# ----------------------------------------------------------------------------
# PART 1 — the LIVE cast_fire_heat, multi-source, CPU vs GPU heat (tol 0).
# ----------------------------------------------------------------------------
def _build_runner_and_map(seed):
    """A real PhysicsRunner + a default-scenario GameMap with MANY burning tiles,
    smoke/gas in the path, and heat_atten occluders — so cast_fire_heat builds a
    multi-source list and accumulates the saturating-add heat across overlapping
    rays (the multi-source accumulation the live gate must preserve)."""
    from field_ab_harness import default_scenario_sim
    from simulation import fire_fixed, gas_fixed
    from simulation.physics_runner import PhysicsRunner

    sim = default_scenario_sim()
    g = sim.gmap
    rng = np.random.default_rng(seed)

    interior = (~g.solid) & (~g.is_vacuum)
    ys, xs = np.nonzero(interior)
    h, w = g.solid.shape

    # Light MANY interior tiles on fire (varied intensity -> varied range/heat),
    # so cast_fire_heat enumerates a big source list. Quantize to the Q16.16 fire
    # field exactly as the sim does (a raw assignment would store ~0 counts).
    g.fire[...] = 0
    n_fire = max(8, len(ys) // 3)
    pick = rng.choice(len(ys), size=min(n_fire, len(ys)), replace=False)
    for k in pick:
        yy, xx = int(ys[k]), int(xs[k])
        g.fire[yy, xx] = fire_fixed.quantize_scalar(float(rng.uniform(0.3, 1.0)))

    # Smoke/gas across the beams' paths (so the gas `expf` runs on the RGB path —
    # it must not touch the heat-set). Fill a couple of int32 gas planes with a
    # random interior cloud (Q16.16 counts).
    for gi in range(g.gas.shape[0]):
        if rng.random() < 0.6:
            plane = np.zeros((h, w), dtype=np.int32)
            blob = (rng.random((h, w)) * 0.8 * gas_fixed.FP_ONE_F).astype(np.int32)
            plane[interior] = blob[interior]
            g.gas[gi] = plane

    # A few heat_atten occluders cutting across the room (partial + full), so the
    # heat survival decays and the occlusion branch is exercised live.
    g.heat_atten[h // 2, :] = 0.7
    if w > 4:
        g.heat_atten[:, w // 2] = 1.0

    runner = PhysicsRunner(bp)
    # Mirror the binds the live Simulation does for the heat-cast params (the
    # PhysicsRunner __init__ already binds them from config; nothing extra needed).
    return runner, g


def _cast_live(runner, g, tick=0):
    """Run the production cast once into freshly-zeroed output planes and return
    them (the per-tick clear is the sim's job; we zero here to isolate THIS
    cast's accumulation).

    P-R4 RE-ANCHOR (ruling amendment 5 D2): the cast's synced output is no
    longer `heat`. The PAINTER is retired — a fire does not deposit one-way
    energy into every cell its rays cross — so `cast_fire_heat` now produces
    `rad_net` (the SIGNED net-T^4 energy ledger, solids only) and `rad_flux`
    (D3's positive-only damage SENSOR, air only). BOTH are compared at tol 0:
    they exercise the two different device scatters this gate exists to pin —
    a PLAIN signed atomicAdd for the ledger (order-free because integer
    addition is associative; a saturating signed add would NOT be) and the
    SATURATING atomic for the sensor (the old heat contract, order-free for
    non-negative deltas). `heat` is still zeroed and returned so the
    multi-source accumulation check below reads the same shape as before."""
    g.heat[...] = 0
    g.rad_net[...] = 0
    g.rad_flux[...] = 0
    runner.cast_fire_heat(g, tick=tick)
    return g.rad_net.copy(), g.rad_flux.copy()


def part1_live_cast() -> bool:
    print("PART 1 — LIVE cast_fire_heat heat bit-identity (multi-source, tol 0):")
    ok = True
    n_scen = 0
    for seed in (20260628, 3, 17, 51, 88):
        # Rebuild a fresh runner+map per backend so neither run sees the other's
        # accumulated state (the runner caches scratch buffers; a fresh instance
        # mirrors the live game's single-runner-per-session but guarantees the A/B
        # starts from identical zeroed scratch).
        runner_cpu, g_cpu = _build_runner_and_map(seed)
        bp.set_raycaster_backend(False)
        heat_cpu, flux_cpu = _cast_live(runner_cpu, g_cpu)

        runner_gpu, g_gpu = _build_runner_and_map(seed)
        bp.set_raycaster_backend(True)
        heat_gpu, flux_gpu = _cast_live(runner_gpu, g_gpu)
        bp.set_raycaster_backend(False)   # restore
        n_scen += 1

        # Sanity: both maps must be the SAME scenario (same fire layout) — the A/B
        # is only meaningful if the inputs match. (default_scenario_sim + the same
        # seed make them identical; assert the fire fields agree.)
        if not np.array_equal(g_cpu.fire, g_gpu.fire):
            ok = False
            print(f"  seed {seed}: SCENARIO MISMATCH (fire layout differs) — "
                  f"the A/B inputs are not identical, gate invalid.")
            continue

        if not np.array_equal(heat_cpu, heat_gpu) or            not np.array_equal(flux_cpu, flux_gpu):
            ok = False
            bad, cpu_a, gpu_a = (("rad_net", heat_cpu, heat_gpu)
                                 if not np.array_equal(heat_cpu, heat_gpu)
                                 else ("rad_flux", flux_cpu, flux_gpu))
            mism = int(np.count_nonzero(cpu_a != gpu_a))
            idx = int(np.argmax(cpu_a != gpu_a))
            ry, rx = divmod(idx, cpu_a.shape[1])
            print(f"  seed {seed}: {mism} {bad.upper()} MISMATCH "
                  f"(first @ ({ry},{rx}): cpu={cpu_a.flat[idx]} "
                  f"gpu={gpu_a.flat[idx]})")
        else:
            nz = int(np.count_nonzero(heat_cpu))
            nzf = int(np.count_nonzero(flux_cpu))
            nfire = int(np.count_nonzero(g_cpu.fire))
            peak = int(np.abs(heat_cpu).max())
            print(f"  seed {seed}: bit-identical ({nfire} fire sources -> "
                  f"{nz} exchanging tiles / {nzf} lit air tiles, "
                  f"|rad_net|peak={peak}).")
            if int(heat_cpu.sum()) != 0:
                ok = False
                print(f"  seed {seed}: rad_net does NOT conserve "
                      f"(sum={int(heat_cpu.sum())}) — antisymmetry broken.")
            # P-R4: `heat` is no longer the observable, so the strength check is
            # on the two planes the cast actually writes. A scene of fires on
            # AIR would legitimately exchange nothing (Kirchhoff), so the
            # multi-source ACCUMULATION signal is the flux sensor.
            if nzf == 0 or nfire < 2:
                ok = False
                print(f"  seed {seed}: SCENARIO TOO WEAK (nfire={nfire}, "
                      f"lit={nzf}) — multi-source accumulation not exercised.")
    if ok:
        print(f"  all {n_scen} live multi-source casts: GPU rad_net + rad_flux "
              f"== CPU byte-for-byte through PhysicsRunner.cast_fire_heat "
              f"(signed plain atomic + saturating atomic, both order-free).")
    return ok


def part1b_multitick_live() -> bool:
    """The live cast_fire_heat across an EVOLVING fire field — the production
    tick is fire-solver -> cast_fire_heat each tick, so the cast sees a CHANGING
    source list (fire grows, decays, saturates). Step ONE sim with the real fire
    solver; each tick, before the sim clears heat, cast on BOTH backends into a
    scratch heat buffer and assert byte-identical. Proves the wiring holds tick
    after tick on real evolving state (not just one frozen frame)."""
    print("PART 1b — LIVE cast over EVOLVING fire, per tick, radiation tol 0:")
    from field_ab_harness import default_scenario_sim

    sim = default_scenario_sim()
    g = sim.gmap
    # P-R4 re-anchor (ruling amendment 5 D2): give the scenario REAL EMITTERS.
    # The canonical A/B scenario seeds fire at (8,8)/(8,9) on AIR tiles — a
    # GHOST fire whose only observable was the retired painter's air deposit.
    # Under Kirchhoff a body that cannot absorb cannot emit (a_s == 0), so a
    # ghost fire radiates NOTHING and this leg would be vacuous. Put wood under
    # the two seeded tiles (plus a wood absorber beside them) and hold them at
    # flame temperature, so the cast has a genuinely evolving emitter set to
    # exercise tick after tick — which is what this leg is FOR. The golden
    # scenario itself is untouched (this is a local mutation of one sim).
    from simulation.materials import MAT_WOOD
    from simulation import fire_fixed as _ff
    for (yy, xx) in ((8, 8), (8, 9), (9, 8)):
        g.material[yy, xx] = MAT_WOOD
    g._update_caches()
    for (yy, xx) in ((8, 8), (8, 9)):
        g.fire[yy, xx] = _ff.quantize_scalar(0.8)
        g.temperature[yy, xx] = _ff.quantize_scalar(443.0)
    runner = sim.physics_runner if sim.physics_runner is not None else None
    if runner is None:
        print("  no physics_runner on the sim — cannot drive the live cast.")
        return False

    ok = True
    n_tick = 0
    max_peak = 0
    for t in range(20):
        # Cast THIS tick's fire on both backends into a fresh scratch heat buffer
        # (don't disturb the sim's own gmap.heat — we use a private copy of the
        # field state for the A/B, casting the SAME source list both ways).
        # P-R4 re-anchor: the cast's synced outputs are `rad_net` (signed
        # ledger) and `rad_flux` (D3 damage sensor), not `heat`. Swap BOTH out
        # for private scratch so the sim's real buffers are untouched, and pass
        # the real tick so D4's fan rotation is the production one.
        saved_net, saved_flux = g.rad_net, g.rad_flux
        g.rad_net = np.zeros_like(saved_net)
        g.rad_flux = np.zeros_like(saved_flux)
        bp.set_raycaster_backend(False)
        runner.cast_fire_heat(g, tick=sim.tick)
        heat_cpu, flux_cpu = g.rad_net.copy(), g.rad_flux.copy()

        g.rad_net[...] = 0
        g.rad_flux[...] = 0
        bp.set_raycaster_backend(True)
        runner.cast_fire_heat(g, tick=sim.tick)
        heat_gpu, flux_gpu = g.rad_net.copy(), g.rad_flux.copy()
        bp.set_raycaster_backend(False)
        g.rad_net, g.rad_flux = saved_net, saved_flux   # restore the sim's own

        if not np.array_equal(heat_cpu, heat_gpu):
            ok = False
            mism = int(np.count_nonzero(heat_cpu != heat_gpu))
            print(f"  tick {t}: {mism} RAD_NET MISMATCH on the live evolving cast.")
            break
        if not np.array_equal(flux_cpu, flux_gpu):
            ok = False
            mism = int(np.count_nonzero(flux_cpu != flux_gpu))
            print(f"  tick {t}: {mism} RAD_FLUX MISMATCH on the live evolving cast.")
            break
        max_peak = max(max_peak, int(np.abs(heat_cpu).max()), int(flux_cpu.max()))
        n_tick += 1

        # Advance the sim one real tick so the fire field evolves for the next
        # cast, holding the two emitters lit AND hot (P-R4: an emitter radiates
        # against its own temperature).
        sim.set_paused(False)
        sim.step()
        for (yy, xx) in ((8, 8), (8, 9)):
            g.fire[yy, xx] = max(int(g.fire[yy, xx]), _ff.quantize_scalar(0.8))
            g.temperature[yy, xx] = max(int(g.temperature[yy, xx]),
                                        _ff.quantize_scalar(443.0))
    if ok:
        print(f"  {n_tick} ticks of the live evolving fire->radiation cast: GPU "
              f"rad_net + rad_flux == CPU byte-for-byte every tick (peak over "
              f"the run = {max_peak} counts).")
        if max_peak == 0:
            ok = False
            print("  SCENARIO WEAK: both radiation planes stayed zero — vacuous. "
                  "(NOTE: the canonical A/B scenario seeds fire on AIR tiles, "
                  "which by Kirchhoff neither absorb nor emit — see the golden "
                  "rebase note above. The flux sensor is what registers there.)")
    return ok


# ----------------------------------------------------------------------------
# PART 2 — all-backends-on 30-tick integration vs CPU (tol 0) + golden.
# ----------------------------------------------------------------------------
# EOS P6.0: wave/atmos backends retired (cuda_wave.cu / cuda_atmosphere.cu
# deleted with their CPU solvers); the all-on set is now 5.
# EOS P6.5: the four EOS kernel-surface flags (bulk_flux, sl_advection,
# mg_solve, kick_compression) are now LIVE-DISPATCHED — with all four on,
# run_substeps routes the whole eos.step tick to the chained GPU orchestration
# (cuda_eos_step.cu). All six EOS-era setters are in the all-on set.
_SETTERS = ("set_temperature_backend", "set_water_backend", "set_smoke_backend",
            "set_fire_backend", "set_raycaster_backend", "set_bulk_flux_backend",
            "set_sl_advection_backend", "set_mg_solve_backend",
            "set_kick_compression_backend")


def _set_all(on):
    for name in _SETTERS:
        getattr(bp, name)(bool(on))


def part2_integration() -> bool:
    print("PART 2 — all-backends-on 30-tick trajectory vs CPU (tol 0) + golden:")
    from field_ab_harness import capture_trajectory, diff_trajectories
    from field_digest import trajectory_digest

    # ALL backends ON (incl. raycaster) — the full GPU tick, fire seeded so
    # cast_fire_heat runs the GPU ray cast each tick.
    _set_all(True)
    traj_gpu = capture_trajectory(n_steps=30)
    # ALL OFF — the pure CPU reference.
    _set_all(False)
    traj_cpu = capture_trajectory(n_steps=30)

    diffs = diff_trajectories(traj_cpu, traj_gpu, tol=0.0)
    ok = (len(diffs) == 0)
    if not ok:
        print(f"  {len(diffs)} field divergence(s) over 30 ticks; "
              f"first 5:")
        for d in diffs[:5]:
            print(f"    {d}")
    else:
        # `heat` is a per-tick deposit buffer CLEARED at the end of Simulation.step
        # (so the post-step snapshot sees 0 — its bit-identity over the trajectory
        # is real but the NON-VACUOUS heat proof is PART 1, which reads heat before
        # the clear). The witness that the GPU raycaster actually RAN each tick is
        # that fire is present (cast_fire_heat enumerates burning tiles every tick).
        peak_fire = max(int(np.abs(s["fire"]).max()) for s in traj_cpu)
        nfields = len(set(traj_cpu[0]) | set(traj_gpu[0]))
        print(f"  CPU vs all-on-GPU: ALL {nfields} synced fields bit-identical over "
              f"30 ticks (incl. heat + temperature; fire present each tick -> the "
              f"GPU fire->heat cast ran, peak |fire|={peak_fire} counts).")
        if peak_fire == 0:
            ok = False
            print("  SCENARIO WEAK: no fire over the run -> cast_fire_heat never "
                  "cast on the GPU; the integration is vacuous for the raycaster.")

    # The CPU path (all backends OFF) must still reproduce the committed golden —
    # the raycaster being live-wired changes NOTHING when the flag is off.
    _set_all(False)
    base = capture_trajectory(n_steps=30)
    dig = trajectory_digest(base)
    # EXPECTED RED until P-G3 re-baseline (#54): physics moved under
    # P-G1a/P-G1b/P-G1d/P-G2 (stored gas_energy, the face-flux energy step,
    # the D4 divergence face form) — golden regen is P-G3's job, not this
    # patch's (P-G2b is test-tooling only). Left asserting, not loosened.
    if dig != GOLDEN:
        ok = False
        print(f"  GOLDEN MISMATCH: {dig[:16]}... != {GOLDEN[:16]}...")
    else:
        print(f"  CUDA build CPU path reproduces the golden ({dig[:12]}...).")
    return ok


def main() -> int:
    if not getattr(bp, "HAS_CUDA", False) or not bp.cuda_available():
        print("RAYCASTER_LIVE_RESULT: FAIL (no CUDA build / device)")
        print("S2_LIVE_RESULT: FAIL (no CUDA build / device)")
        return 1
    print("device:", bp.cuda_device_info())
    p1 = part1_live_cast()
    p1b = part1b_multitick_live()
    p2 = part2_integration()
    ok = p1 and p1b and p2
    print("RAYCASTER_LIVE_RESULT:", "PASS" if ok else "FAIL")
    print("S2_LIVE_RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
