"""Issue #7 — the O2 law's PRESSURE factor: CPU == CUDA at tolerance 0 on both
law sites (runs inside the GPU subprocess; the pytest wrapper is
tests/test_cuda_o2_pressure.py).

The law g(p) = clamp01((p - p_ext) / (p_full - p_ext)) lives ONCE, FP_HD, in
cpp/src/o2_pressure_factor.h; the CPU TUs and the CUDA kernels call the same
function and the host bakes the span reciprocal with the same integer kit call.
This gate proves the two backends agree to the bit anyway — on the inputs (the
pressure gather, the plane upload), the edges' plumbing and the fold into o2f.

  PART A — the fire ODE site: FireSimulation.step vs cuda_fire_step over random
    fields whose `atmosphere` spans all three regimes (below p_ext, the ramp,
    above p_full), plus exact-edge forcers (every neighbour at p_ext - 1,
    p_ext, p_ext + 1, the midpoint, p_full - 1, p_full, p_full + 1, 1 atm), a
    mixed field with solid and vacuum neighbours, for the live edges, a span
    whose bare ramp stops short at p_full, and the dormant 0 / 0 default.
    Byte-identity on fire / smoke / wall_hp / temperature + the destroyed LIST
    (cuda_fire_check.compare).
  PART B — the claim-gate site: CombustionSolver.step vs cuda_combustion_step
    on cuda_po2b_check's full-surface scenario (contested claimants, permeable
    crates, a vacuum pocket, thermal solids, the fuel-bed `heat` plane, the D1
    `dem_acc` plane), at draw_r 1 and 2, with random `atmosphere` planes over
    all three regimes and the same edge sets. Byte-identity on gas /
    temperature / wall_hp / heat / dem_acc + the rail counters.
  PART C — the LIVE fire backend: a breached burning room and a sealed one,
    stepped by PhysicsRunner with the fire backend on the CPU and on the GPU
    (combustion on the CPU in both — the shipped `main.py --cuda`
    configuration, tools/run_on_cuda.py), every synced field byte-identical
    every tick. This is the proof that physics_engine.cpp hands the bound
    edges and the `atmosphere` plane to the GPU fire step.

Prints ``O2P_RESULT: PASS`` / ``FAIL`` and exits 0 / 1.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Import the CUDA build FIRST so it is the cached `breach_physics`.
import breach_physics as bp

import cuda_fire_check as FC
import cuda_po2b_check as PC
from simulation import atmosphere_fixed
from simulation.gases import INERT_N2, O2, SMOKE

FP_ONE = atmosphere_fixed.FP_ONE
DT = 1.0 / 24.0


def _edge_sets():
    """The live (config-bound) edges, a span whose bare ramp stops one count
    short of FP_ONE at p_full (the case the explicit branch exists for), and
    the dormant default."""
    from simulation.physics_runner import PhysicsRunner
    pr = PhysicsRunner(bp)
    p_ext, p_full = int(pr.fire.params.p_ext_q), int(pr.fire.params.p_full_q)
    short = None
    for hundredths in range(30, 96):
        pf = atmosphere_fixed.quantize_scalar(hundredths / 100.0)
        span = pf - p_ext
        if span > 0 and (span * int(bp.fp_reciprocal_q16(span))) >> 16 < FP_ONE:
            short = (p_ext, pf)
            break
    assert short is not None, "no short-ramp span found — re-derive"
    return {"live": (p_ext, p_full), "short_ramp": short, "dormant": (0, 0)}


EDGES = _edge_sets()


# ---------------------------------------------------------------------------
# PART A — the fire ODE site
# ---------------------------------------------------------------------------
def _fire_pair(tag, state, edges, dt=DT):
    fp, dials = FC.make_params()
    fp.p_ext_q, fp.p_full_q = (int(e) for e in edges)
    sim = bp.FireSimulation()
    sim.params = fp
    c = {k: state[k].copy() for k in state}
    d_cpu = sim.step(c["fire"], c["atmosphere"], c["n_o2"], c["n_total"],
                     c["smoke"], c["wall_hp"], c["temperature"], c["wind_x"],
                     c["wind_y"], c["is_wall"], c["is_vacuum"], c["flammable"], dt)
    g = {k: state[k].copy() for k in state}
    d_gpu = bp.cuda_fire_step(
        g["fire"], g["atmosphere"], g["n_o2"], g["n_total"], g["smoke"],
        g["wall_hp"], g["temperature"], g["wind_x"], g["wind_y"], g["is_wall"],
        g["is_vacuum"], g["flammable"], dt, **dials,
        p_ext_q=int(edges[0]), p_full_q=int(edges[1]))
    return FC.compare(tag, c, list(d_cpu), g, list(d_gpu)), c


def _lit_tile(p_nbr_q, solid_nbrs=(), vacuum_nbrs=()):
    st = FC._blank(5, 5)
    st["flammable"][2, 2] = True
    st["is_wall"][2, 2] = True
    st["fire"][2, 2] = FC._quantize(0.5)[()]
    st["temperature"][2, 2] = FC._quantize(700.0)[()]
    st["wall_hp"][2, 2] = FC._quantize(55.0)[()]
    st["n_total"][:] = FC._quantize(1.0)
    st["n_o2"][:] = FC._quantize(0.21)
    st["atmosphere"][:] = int(p_nbr_q)
    for c in solid_nbrs:
        st["is_wall"][c] = True
        st["atmosphere"][c] = 0
    for c in vacuum_nbrs:
        st["is_vacuum"][c] = True
        st["atmosphere"][c] = 0
    st["atmosphere"][2, 2] = 0
    return FC._contig(st)


def part_a_fire_site() -> bool:
    print("PART A — fire ODE site, CPU FireSimulation.step vs GPU cuda_fire_step:")
    ok = True
    rng = np.random.default_rng(20260923)
    n_cases = 0
    for name, edges in EDGES.items():
        # Random fields: `atmosphere` is uniform over [0, 1.2) atm in
        # FC._make_random_state — below p_ext, the ramp and above p_full.
        for (h, w, wind) in [(16, 16, 0.0), (24, 32, 5.0), (31, 17, 40.0),
                             (1, 50, 5.0), (50, 1, 5.0)]:
            for _ in range(3):
                st = FC._make_random_state(rng, h, w, wind)
                good, _ = _fire_pair(f"{name} rand {h}x{w}", st, edges)
                ok &= good
                n_cases += 1
        p_ext, p_full = edges
        forcers = [0, max(p_ext - 1, 0), p_ext, p_ext + 1,
                   (p_ext + p_full) // 2, max(p_full - 1, 0), p_full,
                   p_full + 1, FP_ONE]
        for p in forcers:
            good, _ = _fire_pair(f"{name} edge p={p}", _lit_tile(p), edges)
            ok &= good
            n_cases += 1
        mixed = _lit_tile(p_full, solid_nbrs=((1, 2), (2, 1)),
                          vacuum_nbrs=((3, 2),))
        good, _ = _fire_pair(f"{name} solid+vacuum neighbours", mixed, edges)
        ok &= good
        n_cases += 1
    # Non-vacuity: below p_ext the live law must actually differ from the
    # dormant one on this backend pair, or PART A proved nothing about g.
    _, live = _fire_pair("nv live", _lit_tile(0), EDGES["live"])
    _, dorm = _fire_pair("nv dormant", _lit_tile(0), EDGES["dormant"])
    if np.array_equal(live["fire"], dorm["fire"]):
        ok = False
        print("  non-vacuity: below p_ext the live law equals the dormant one")
    if ok:
        print(f"  {n_cases} cases bit-identical (3 edge sets x random fields + "
              f"exact-edge forcers + solid/vacuum gather); the live law bites "
              f"below p_ext.")
    return ok


# ---------------------------------------------------------------------------
# PART B — the claim-gate site
# ---------------------------------------------------------------------------
def _comb_cpu(s, draw_r, edges, atm):
    comb = PC._mk_solver()
    comb.p_ext_q, comb.p_full_q = (int(e) for e in edges)
    comb.step(s["gas"], O2, INERT_N2, SMOKE, s["temperature"], s["wall_hp"],
              s["fire"], s["flammable"], s["solid"], s["is_vacuum"],
              s["ignition_temp_q16"], PC.DT, PC.C_V, PC.N_FLOOR_HEAT,
              s["thermal_solid"], s["heat_inv_shift"], s["heat"], s["dem_acc"],
              draw_r, s["dyn_permeability"], int(s["dem_acc"].shape[0]),
              atmosphere=atm)
    return (int(comb.heat_floor_hits), int(comb.t_max_phys_hits),
            int(comb.e_deposit_drop_sum))


def _comb_gpu(s, draw_r, edges, atm):
    D = PC.DIALS
    hf, tm, dd = bp.cuda_combustion_step(
        s["gas"], O2, INERT_N2, SMOKE, s["temperature"], s["wall_hp"], s["fire"],
        s["flammable"], s["solid"], s["is_vacuum"], s["ignition_temp_q16"],
        PC.DT, PC.C_V, PC.N_FLOOR_HEAT,
        D["burn_rate"], D["o2_thresh_burn"], D["H_FUEL_M"], int(D["H_FUEL_SHIFT"]),
        D["soot_yield"], D["fuel_per_o2"], D["o2_frac_ext"], D["o2_frac_full"],
        D["T_MAX_PHYS"],
        s["thermal_solid"], s["heat_inv_shift"], s["heat"], PC.H_BED_M,
        PC.H_BED_SHIFT, s["dem_acc"], draw_r, s["dyn_permeability"],
        int(s["dem_acc"].shape[0]),
        atmosphere=atm, p_ext_q=int(edges[0]), p_full_q=int(edges[1]))
    return (int(hf), int(tm), int(dd))


def part_b_claim_gate() -> bool:
    print("PART B — claim-gate site, CPU CombustionSolver.step vs GPU "
          "cuda_combustion_step (draw_r 1 and 2, heat + dem_acc planes):")
    ok = True
    rng = np.random.default_rng(7)
    n_ticks = 0
    top = atmosphere_fixed.quantize_scalar(1.2)
    for name, edges in EDGES.items():
        for draw_r in (1, 2):
            cpu = PC.build_state(draw_r=draw_r, seed=11)
            gpu = PC._copy(cpu)
            for tick in range(12):
                # A fresh random pressure field each tick, over all regimes,
                # identical on both sides; tick 0 pins exact edges in bands.
                atm = rng.integers(0, top, size=cpu["fire"].shape).astype(np.int32)
                if tick == 0:
                    p_ext, p_full = edges
                    atm[:, 0::4] = p_ext
                    atm[:, 1::4] = p_full
                    atm[:, 2::4] = max(p_full - 1, 0)
                atm = np.ascontiguousarray(atm)
                r_cpu = _comb_cpu(cpu, draw_r, edges, atm)
                r_gpu = _comb_gpu(gpu, draw_r, edges, atm)
                if not PC.compare(f"{name} R={draw_r} tick {tick}", cpu, gpu):
                    ok = False
                if r_cpu != r_gpu:
                    ok = False
                    print(f"  {name} R={draw_r} tick {tick}: rails "
                          f"cpu={r_cpu} gpu={r_gpu}")
                n_ticks += 1
    # Non-vacuity: the claim gate must draw LESS O2 under a vacuum plane than
    # under an ambient one, on the GPU (where the new code runs).
    a, b = PC.build_state(draw_r=2, seed=11), PC.build_state(draw_r=2, seed=11)
    o2_0 = int(a["gas"][O2].sum())
    _comb_gpu(a, 2, EDGES["live"], np.zeros(a["fire"].shape, dtype=np.int32))
    _comb_gpu(b, 2, EDGES["live"],
              np.full(b["fire"].shape, FP_ONE, dtype=np.int32))
    drawn_vac, drawn_amb = o2_0 - int(a["gas"][O2].sum()), o2_0 - int(b["gas"][O2].sum())
    if not (drawn_vac == 0 < drawn_amb):
        ok = False
        print(f"  non-vacuity: O2 drawn under vacuum {drawn_vac}, under "
              f"ambient {drawn_amb} (want 0 < ambient)")
    if ok:
        print(f"  {n_ticks} ticks bit-identical (3 edge sets x draw_r 1,2 x 12 "
              f"random pressure fields); under a vacuum plane the GPU claim "
              f"gate draws 0 O2 (ambient: {drawn_amb}).")
    return ok


# ---------------------------------------------------------------------------
# PART C — the live fire backend
# ---------------------------------------------------------------------------
def _live_trajectory(vent, use_gpu_fire, ticks=120):
    from level_loader import LevelData
    from simulation.gamemap import GameMap
    from simulation.materials import MAT_AIR, MAT_HULL, MAT_WOOD, MaterialTable
    from simulation.physics_runner import PhysicsRunner
    from simulation import fire_fixed
    ign = int(MaterialTable.from_config().ignition_temp_q16[MAT_WOOD])
    tm = np.full((9, 9), MAT_HULL, dtype=np.int32)
    tm[1:8, 1:8] = MAT_AIR
    tm[4, 4] = MAT_WOOD
    ld = LevelData(name="o2p_live", version="2", path=Path("."), tilemap=tm,
                   tile_size_m=1.0 / 3.0, diffuse_path=Path("."))
    g = GameMap(ld)
    g.fire[4, 4] = fire_fixed.quantize_scalar(0.6)
    g.temperature[4, 4] = int(ign * 1.5)
    if vent:
        g.destroy_wall(0, 4)
    bp.set_fire_backend(bool(use_gpu_fire))
    try:
        pr = PhysicsRunner(bp)
        snaps = []
        for _ in range(ticks):
            pr.step(g, DT)
            g.heat.fill(0)
            g.rad_net.fill(0)
            g.rad_flux.fill(0)
            snaps.append({k: getattr(g, k).copy() for k in
                          ("fire", "wall_hp", "temperature", "atmosphere",
                           "gas", "gas_energy", "dem_acc")})
    finally:
        bp.set_fire_backend(False)
    return snaps


def part_c_live_backend() -> bool:
    print("PART C — live PhysicsRunner, fire backend CPU vs GPU (combustion "
          "on the CPU in both, the shipped --cuda configuration):")
    ok = True
    for vent in (True, False):
        tag = "vented" if vent else "sealed"
        a = _live_trajectory(vent, use_gpu_fire=False)
        b = _live_trajectory(vent, use_gpu_fire=True)
        diverged = False
        for t, (sa, sb) in enumerate(zip(a, b)):
            bad = [k for k in sa if not np.array_equal(sa[k], sb[k])]
            if bad:
                ok = False
                diverged = True
                print(f"  {tag}: tick {t + 1} diverged on {bad}")
                break
        i_end = int(a[-1]["fire"][4, 4])
        if not diverged:
            print(f"  {tag}: {len(a)} ticks bit-identical; fire at the end "
                  f"I = {i_end / FP_ONE:.4f}")
        if vent and not i_end < int(a[0]["fire"][4, 4]):
            ok = False
            print("  vented: the fire did not weaken — the pressure factor "
                  "never engaged, PART C proved nothing")
    return ok


def main() -> int:
    if not getattr(bp, "HAS_CUDA", False) or not bp.cuda_available():
        print("O2P_RESULT: FAIL (no CUDA build / device)")
        return 1
    print("device:", bp.cuda_device_info())
    print("edge sets:", EDGES)
    pa = part_a_fire_site()
    pb = part_b_claim_gate()
    pc = part_c_live_backend()
    if pa and pb and pc:
        print("O2P_RESULT: PASS")
        return 0
    print("O2P_RESULT: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
