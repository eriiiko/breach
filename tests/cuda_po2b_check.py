"""P-O2b (THE EXTENDED OXYGEN DRAW) — CPU<->CUDA bit-identity check.

docs/fire_realism_design_2026-08-01.md v5.2 "F-O2b". This is the patch's
gate (d): "CPU<->CUDA tol 0 step+resident at R = 1 AND R = 2, including
dem_acc". The shipped ``cuda_combustion_check.py`` predates the extended draw
and calls the pass with its DEFAULTS (draw_r == 1, no dem_acc, no heat), so it
covers only the byte-identical legacy path; this file drives the parts P-O2b
actually added.

WHAT IT PROVES, part by part:

  PART 1 — the R = 1 IDENTITY, on both backends. With draw_r == 1 the
    extended law must reproduce the shipped 4-face draw BIT FOR BIT. Here that
    is checked the strong way: the same scenario is run at draw_r == 1 through
    the NEW code path (with the widened plumbing live: a 4-deep dem_acc, the
    permeability plane, the re-sited deposit) and against the pass called with
    its pre-P-O2b default arguments. Every mutated plane must match exactly.

  PART 2 — CPU == GPU at draw_r 1, 2 and 3, over a multi-tick trajectory, on a
    scenario built to exercise every branch the law has: contested air cells
    with several claimants, permeable crates that ATTENUATE the draw through
    themselves, solid walls that BLOCK it, a vacuum pocket that TERMINATES
    expansion, a burning furniture tile (an open, gas-holding source that is
    also a legal traversal cell and a legal deposit site), and grid-edge tiles
    where slots fall out of bounds. Compared with tol 0: gas (all planes),
    temperature, wall_hp, heat, and — the point of the gate — dem_acc.

  PART 3 — ORDER FREEDOM, empirically. The GPU runs cells in a grid-stride
    loop across concurrently scheduled blocks, so its per-cell execution order
    is arbitrary and varies between launches; the CPU walks strict row-major.
    A tol-0 match between them, repeated, IS the statement that no enumeration
    order (of air cells, of source tiles, or of claimants within a cell)
    changes the resulting planes. Part 3 repeats the GPU launch and requires
    every repeat to agree with the single CPU reference.

Run (needs the CUDA build + a device):
    C:/Users/steen/miniconda3/envs/data/python.exe tests/cuda_po2b_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build_cuda", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np

import breach_physics as bp

from simulation.gases import O2, INERT_N2, SMOKE, N_GASES

FP_ONE = 65536

DIALS = dict(burn_rate=1.0, o2_thresh_burn=0.03, H_fuel=4.0, soot_yield=0.3,
             fuel_per_o2=0.7, o2_frac_ext=0.13, o2_frac_full=0.21,
             o2_frac_amb=0.21, T_MAX_PHYS=16000.0)
C_V = 1.0
N_FLOOR_HEAT = 0.05
H_BED_M = 25290.0
H_BED_SHIFT = 3
IGN_Q = int(round(500.0 * FP_ONE))
DT = 1.0 / 24.0

# The planes the pass mutates — every one is compared at tol 0.
MUTATED = ("gas", "temperature", "wall_hp", "heat", "dem_acc")


def _slot_count(r):
    return 2 * r * (r + 1)


def _mk_solver():
    c = bp.CombustionSolver()
    for k, v in DIALS.items():
        setattr(c, k, v)
    c.H_BED_M = H_BED_M
    c.H_BED_SHIFT = H_BED_SHIFT
    return c


def build_state(h=24, w=28, draw_r=2, seed=7):
    """A scenario that exercises every branch of the extended draw."""
    rng = np.random.default_rng(seed)
    solid = np.zeros((h, w), dtype=bool)
    is_vacuum = np.zeros((h, w), dtype=bool)
    flammable = np.zeros((h, w), dtype=bool)
    perm = np.ones((h, w), dtype=np.float32)

    solid[0, :] = solid[-1, :] = True          # hull
    solid[:, 0] = solid[:, -1] = True
    solid[6:12, 14] = True                     # an interior BLOCKING wall
    perm[solid] = 0.0

    # A vacuum pocket — expansion must TERMINATE there.
    is_vacuum[3:5, 22:25] = True

    # Permeable crates (perm 0.5) — they ATTENUATE the draw through themselves
    # and are themselves open, gas-holding cells.
    crates = [(9, 6), (9, 7), (14, 18), (15, 18), (5, 9)]
    for (y, x) in crates:
        perm[y, x] = 0.5

    # Flammable sources: wood/door style SOLID tiles, plus burning FURNITURE
    # (open + gas-holding, so also a traversal cell and a deposit site).
    solid_fuel = [(8, 5), (8, 8), (10, 12), (16, 20), (1, 3), (12, 26)]
    for (y, x) in solid_fuel:
        solid[y, x] = True
        flammable[y, x] = True
        perm[y, x] = 0.0
    furniture_fuel = [(9, 6), (14, 18)]        # a subset of the crates
    for (y, x) in furniture_fuel:
        flammable[y, x] = True

    open_mask = (~solid) & (~is_vacuum)

    gas = np.zeros((N_GASES, h, w), dtype=np.int32)
    # Varied O2 so both the uncontended and the CONTESTED/full-drain branches
    # fire, including cells right at the epsilon skip-floor.
    o2 = rng.integers(0, 4000, size=(h, w))
    o2[6:9, 4:10] = rng.integers(0, 60, size=(3, 6))    # near-starved pocket
    gas[O2][open_mask] = o2[open_mask].astype(np.int32)
    gas[INERT_N2][open_mask] = np.int32(14000)

    fire = np.zeros((h, w), dtype=np.int32)
    for (y, x) in solid_fuel + furniture_fuel:
        fire[y, x] = int(0.35 * FP_ONE)
    fire[1, 3] = int(0.9 * FP_ONE)

    temperature = np.zeros((h, w), dtype=np.int32)
    temperature[flammable] = IGN_Q + 1000
    wall_hp = np.zeros((h, w), dtype=np.int32)
    wall_hp[flammable] = 60 * FP_ONE
    ign = np.zeros((h, w), dtype=np.int32)
    ign[flammable] = IGN_Q

    thermal_solid = np.zeros((h, w), dtype=bool)
    heat_inv_shift = np.zeros((h, w), dtype=np.int32)
    for (y, x) in crates:
        thermal_solid[y, x] = True
        heat_inv_shift[y, x] = 3

    return dict(
        gas=gas, temperature=temperature, wall_hp=wall_hp, fire=fire,
        flammable=flammable, solid=solid, is_vacuum=is_vacuum,
        ignition_temp_q16=ign, thermal_solid=thermal_solid,
        heat_inv_shift=heat_inv_shift,
        heat=np.zeros((h, w), dtype=np.int32),
        dem_acc=np.zeros((_slot_count(draw_r), h, w), dtype=np.int32),
        dyn_permeability=perm,
    )


def _copy(s):
    return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in s.items()}


def _cpu_step(s, draw_r, legacy=False):
    comb = _mk_solver()
    if legacy:
        # The pre-P-O2b call: default draw_r/permeability/max_claimants.
        comb.step(s["gas"], O2, INERT_N2, SMOKE, s["temperature"], s["wall_hp"],
                  s["fire"], s["flammable"], s["solid"], s["is_vacuum"],
                  s["ignition_temp_q16"], DT, C_V, N_FLOOR_HEAT,
                  s["thermal_solid"], s["heat_inv_shift"], s["heat"], s["dem_acc"])
    else:
        comb.step(s["gas"], O2, INERT_N2, SMOKE, s["temperature"], s["wall_hp"],
                  s["fire"], s["flammable"], s["solid"], s["is_vacuum"],
                  s["ignition_temp_q16"], DT, C_V, N_FLOOR_HEAT,
                  s["thermal_solid"], s["heat_inv_shift"], s["heat"], s["dem_acc"],
                  draw_r, s["dyn_permeability"], int(s["dem_acc"].shape[0]))
    return (int(comb.heat_floor_hits), int(comb.t_max_phys_hits),
            int(comb.e_deposit_drop_sum))


def _gpu_step(s, draw_r):
    hf, tm, dd = bp.cuda_combustion_step(
        s["gas"], O2, INERT_N2, SMOKE, s["temperature"], s["wall_hp"], s["fire"],
        s["flammable"], s["solid"], s["is_vacuum"], s["ignition_temp_q16"],
        DT, C_V, N_FLOOR_HEAT,
        DIALS["burn_rate"], DIALS["o2_thresh_burn"], DIALS["H_fuel"],
        DIALS["soot_yield"], DIALS["fuel_per_o2"], DIALS["o2_frac_ext"],
        DIALS["o2_frac_full"], DIALS["T_MAX_PHYS"],
        s["thermal_solid"], s["heat_inv_shift"], s["heat"], H_BED_M, H_BED_SHIFT,
        s["dem_acc"], draw_r, s["dyn_permeability"], int(s["dem_acc"].shape[0]))
    return (int(hf), int(tm), int(dd))


def compare(tag, a, b):
    ok = True
    for k in MUTATED:
        if not np.array_equal(a[k], b[k]):
            ok = False
            d = a[k] != b[k]
            idx = int(np.argmax(d))
            print(f"  {tag}: '{k}' {int(d.sum())} MISMATCH "
                  f"(first flat @ {idx}: a={a[k].flat[idx]} b={b[k].flat[idx]})")
    return ok


def part1_r1_identity() -> bool:
    print("PART 1 — draw_r == 1 reproduces the pre-P-O2b default call, tol 0:")
    ok = True
    s0 = build_state(draw_r=1)
    a, b = _copy(s0), _copy(s0)
    for tick in range(20):
        ra = _cpu_step(a, 1, legacy=False)     # the NEW path, radius 1
        rb = _cpu_step(b, 1, legacy=True)      # the pre-P-O2b default call
        if not compare(f"cpu tick {tick}", a, b):
            ok = False
            break
        if ra != rb:
            print(f"  tick {tick}: rail counters {ra} != {rb}")
            ok = False
            break
    print(f"  {'OK' if ok else 'FAIL'}: new law at R=1 == shipped 4-face law "
          f"(20 ticks, gas/temperature/wall_hp/heat/dem_acc + rails)")
    return ok


def part2_cpu_gpu(draw_r) -> bool:
    print(f"PART 2 — CPU == GPU at draw_r = {draw_r}, tol 0, 30 ticks:")
    s0 = build_state(draw_r=draw_r)
    cpu, gpu = _copy(s0), _copy(s0)
    ok = True
    for tick in range(30):
        rc = _cpu_step(cpu, draw_r)
        rg = _gpu_step(gpu, draw_r)
        if not compare(f"tick {tick}", cpu, gpu):
            ok = False
            break
        if rc != rg:
            print(f"  tick {tick}: rail counters cpu={rc} gpu={rg}")
            ok = False
            break
    drawn = int(s0["gas"][O2].astype(np.int64).sum()
                - cpu["gas"][O2].astype(np.int64).sum())
    print(f"  {'OK' if ok else 'FAIL'}: 30 ticks bit-identical "
          f"(incl. dem_acc, depth {cpu['dem_acc'].shape[0]}); "
          f"O2 drawn over the run = {drawn} counts")
    return ok


def part3_order_freedom(draw_r=2, repeats=6) -> bool:
    print(f"PART 3 — order freedom at draw_r = {draw_r}: "
          f"{repeats} GPU launches vs one CPU reference:")
    s0 = build_state(draw_r=draw_r)
    ref = _copy(s0)
    _cpu_step(ref, draw_r)
    ok = True
    for r in range(repeats):
        g = _copy(s0)
        _gpu_step(g, draw_r)
        if not compare(f"repeat {r}", ref, g):
            ok = False
    print(f"  {'OK' if ok else 'FAIL'}: every GPU launch (arbitrary, varying "
          f"per-cell scheduling order) equals the row-major CPU result exactly")
    return ok


def part4_no_oxygen_created(draw_r=2) -> bool:
    """The draw must never CREATE oxygen (gate (c)'s sum check, kernel-level)."""
    print(f"PART 4 — the draw creates no oxygen (draw_r = {draw_r}):")
    s0 = build_state(draw_r=draw_r)
    s = _copy(s0)
    ok = True
    for tick in range(30):
        before = s["gas"][O2].copy()
        _cpu_step(s, draw_r)
        after = s["gas"][O2]
        if (after > before).any():
            n = int((after > before).sum())
            print(f"  tick {tick}: {n} cell(s) GAINED O2 in the combustion pass")
            ok = False
            break
        if (after < 0).any():
            print(f"  tick {tick}: negative O2 after the draw")
            ok = False
            break
    print(f"  {'OK' if ok else 'FAIL'}: no cell ever gains O2 in the pass and "
          f"no cell goes negative (30 ticks)")
    return ok


def main() -> int:
    if not getattr(bp, "HAS_CUDA", False) or not bp.cuda_available():
        print("PO2B_RESULT: FAIL (no CUDA build / device)")
        return 1
    print("device:", bp.cuda_device_info())
    results = [
        part1_r1_identity(),
        part4_no_oxygen_created(1),
        part4_no_oxygen_created(2),
        part2_cpu_gpu(1),
        part2_cpu_gpu(2),
        part2_cpu_gpu(3),
        part3_order_freedom(2),
        part3_order_freedom(3),
    ]
    ok = all(results)
    print(f"PO2B_RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
