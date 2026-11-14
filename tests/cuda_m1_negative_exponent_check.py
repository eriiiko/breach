"""M1 — CPU/CUDA bit-identity at a NEGATIVE `heat_inv_shift` (runs inside the
GPU subprocess).

M1 lifts the `thermal_mass >= 1` guard, so `heat_inv_shift` can now be negative
and every `>> heat_inv_shift` in the engine becomes a potential negative shift.
The CPU and CUDA temperature solvers both took that shift, and both were
rewritten to the kit's signed-exponent twin — one `FP_HD` definition in
`fixed_point.h`, called from both. This check proves the two agree at tol 0 on
exactly the exponents the shipped table cannot yet reach.

CPU reference = the bound `TemperatureSolver` (the source of truth);
GPU = `bp.cuda_temperature_step`. Both live in the one CUDA-build module.

  PART 1 — the Pass-1 SOLID HEAT DEPOSIT at every exponent from CAP_SHIFT_MIN
  to +5, including deposits large enough that the int64 gain leaves int32 (the
  site design section 5 calls "the one real risk"). Byte-identity on
  `temperature` plus equality of the T_MAX_PHYS hit count.

  PART 2 — the Pass-1 SIGNED RADIATION FOLD at negative exponents, both signs
  of `rad_net`. Already int64 before M1, but not SIGNED in the exponent.

  PART 3 — Pass-2 CONDUCTION between cells whose capacities straddle 1 unit
  (a 2**-4 panel against a 2**5 steel wall), which is where `cell_capacity_q`'s
  lifted floor actually shows up in a flux.

  PART 4 — a 60-tick TRAJECTORY over a randomized scene of mixed positive and
  negative exponents, asserting per-tick byte-identity, so the parity is not an
  artifact of a single crafted frame.

NON-VACUITY is asserted, not assumed. Parts 1 and 2 state the EXACT integer
each config must land — computed here in Python, so the check is an oracle and
not just a CPU/GPU agreement — and count how many configs had a nonzero effect;
a config whose exact answer is 0 (a deposit of 1 at exponent 3) is legitimate
and is not mistaken for a dead one. Part 1 also asserts the T_MAX_PHYS rail
actually engages, which is what proves the int64 path was driven past int32.
Parts 3 and 4 assert the field moved.

Prints ``M1_RESULT: PASS``/``FAIL`` and exits 0/1.
"""
from __future__ import annotations

import sys

import numpy as np

# Import the CUDA build FIRST so it is the cached `breach_physics`.
import breach_physics as bp

FP_ONE = 65536

DIALS = dict(
    no_face=63, o2_vacuum_thresh=0.3, c_v=0.0076849, n_floor_heat=0.05,
    gas_advection_rate=900.0, t_max_phys=16000.0,
)

E_COUNTERS = ("e_cond_trunc_sum", "e_cond_cap_sum", "cond_limit_hits",
              "e_vac_wipe_sum", "e_ring_pin_sum", "e_deposit_drop_sum")


def _make_solver():
    s = bp.TemperatureSolver()
    s.no_face = DIALS["no_face"]
    s.o2_vacuum_thresh = DIALS["o2_vacuum_thresh"]
    s.c_v = DIALS["c_v"]
    s.n_floor_heat = DIALS["n_floor_heat"]
    s.gas_advection_rate = DIALS["gas_advection_rate"]
    s.T_MAX_PHYS = DIALS["t_max_phys"]
    return s


def _run_pair(solver, temp, heat, his, fs, solid, vac, atm, ts, rad_net=None):
    """CPU reference + GPU kernel on identical copies of the same scene.
    Returns (t_cpu, hits_cpu, t_gpu, hits_gpu)."""
    t_cpu = np.ascontiguousarray(temp.copy())
    h0 = int(solver.t_max_phys_hits)
    solver.step(t_cpu, heat, his, fs, solid, vac, atm,
                thermal_solid=ts, rad_net=rad_net, clamp_enabled=False)
    hits_cpu = int(solver.t_max_phys_hits) - h0

    t_gpu = np.ascontiguousarray(temp.copy())
    out = bp.cuda_temperature_step(t_gpu, heat, his, fs, solid, vac, atm,
                                   thermal_solid=ts, rad_net=rad_net, **DIALS)
    hits_gpu = int(out[0])
    return t_cpu, hits_cpu, t_gpu, hits_gpu


def _compare(tag, t_cpu, hits_cpu, t_gpu, hits_gpu):
    ok = True
    if not np.array_equal(t_cpu, t_gpu):
        ok = False
        idx = int(np.argmax(t_cpu != t_gpu))
        print(f"  {tag}: temperature MISMATCH "
              f"(first @ {idx}: cpu={t_cpu.flat[idx]} gpu={t_gpu.flat[idx]})")
    if hits_cpu != hits_gpu:
        ok = False
        print(f"  {tag}: t_max_phys_hits cpu={hits_cpu} gpu={hits_gpu}")
    return ok


def _blank(h, w):
    fs = np.full((h, w, 4), DIALS["no_face"], dtype=np.int32)
    return (np.zeros((h, w), dtype=np.int32),          # temperature
            np.zeros((h, w), dtype=np.int32),          # heat
            np.zeros((h, w), dtype=np.int32),          # heat_inv_shift
            np.ascontiguousarray(fs),
            np.zeros((h, w), dtype=bool),              # solid (flow)
            np.zeros((h, w), dtype=bool),              # is_vacuum
            np.full((h, w), FP_ONE, dtype=np.int32),   # atmosphere
            np.zeros((h, w), dtype=bool))              # thermal_solid


# ---------------------------------------------------------------- PART 1 ----
def part1_deposit(solver):
    print("PART 1 — Pass-1 solid heat deposit across the exponent band")
    ok = True
    nonzero = 0
    exps = list(range(bp.CAP_SHIFT_MIN, 6))
    # Deposits chosen so the int64 gain crosses int32 on the steep exponents.
    deposits = [1, 4097, 1_000_000, 299_685_456, 2 ** 31 - 1]
    for e in exps:
        for d in deposits:
            temp, heat, his, fs, solid, vac, atm, ts = _blank(3, 3)
            c = (1, 1)
            heat[c] = d
            his[c] = e
            solid[c] = True
            ts[c] = True
            r = _run_pair(solver, temp, heat, his, fs, solid, vac, atm, ts)
            if not _compare(f"deposit e={e} d={d}", *r):
                ok = False
            # Non-vacuity, stated EXACTLY rather than as "it moved": a
            # deposit whose exact gain is 0 (d=1 at e=3) legitimately moves
            # nothing, and demanding movement there would be wrong. Demand
            # instead that the landing IS the exact integer.
            want = d * 2 ** (-e) if e < 0 else d >> e
            rail = int(DIALS["t_max_phys"] * FP_ONE)
            want = min(want, rail)
            if int(r[0][c]) != want:
                print(f"  WRONG: deposit e={e} d={d} -> {int(r[0][c])}, want {want}")
                ok = False
            if want != 0:
                nonzero += 1
    # The rail must actually engage somewhere in the sweep, or the wide path
    # was never exercised past int32.
    temp, heat, his, fs, solid, vac, atm, ts = _blank(3, 3)
    heat[1, 1] = 299_685_456
    his[1, 1] = -4
    solid[1, 1] = True
    ts[1, 1] = True
    r = _run_pair(solver, temp, heat, his, fs, solid, vac, atm, ts)
    if r[1] != 1 or r[3] != 1:
        print(f"  VACUOUS: the T_MAX_PHYS rail did not engage "
              f"(cpu={r[1]} gpu={r[3]})")
        ok = False
    if nonzero < 40:
        print(f"  VACUOUS: only {nonzero} configs had a nonzero exact gain")
        ok = False
    print(f"  {'PASS' if ok else 'FAIL'} ({len(exps) * len(deposits)} configs, "
          f"{nonzero} with a nonzero exact gain)")
    return ok


# ---------------------------------------------------------------- PART 2 ----
def part2_radiation_fold(solver):
    print("PART 2 — Pass-1 signed radiation fold at negative exponents")
    ok = True
    n = 0
    nonzero = 0
    for e in (-1, -3, -8, -16, 0, 3):
        for rn in (-5_000_000, -1, 1, 7_000_000, 2 ** 34):
            temp, heat, his, fs, solid, vac, atm, ts = _blank(3, 3)
            c = (1, 1)
            his[c] = e
            solid[c] = True
            ts[c] = True
            temp[c] = 400_000_000
            rad = np.zeros((3, 3), dtype=np.int64)
            rad[c] = rn
            r = _run_pair(solver, temp, heat, his, fs, solid, vac, atm, ts,
                          rad_net=np.ascontiguousarray(rad))
            if not _compare(f"fold e={e} rad_net={rn}", *r):
                ok = False
            # Exact expectation, so a legitimately-zero fold (rad_net = 1 at
            # e = 3) is not mistaken for a vacuous config.
            want = rn * 2 ** (-e) if e < 0 else (
                -((-rn) >> e) if rn < 0 else (rn >> e))
            rail = int(DIALS["t_max_phys"] * FP_ONE)
            # Pass 1 order: fold, T_MAX_PHYS rail, then the low rail at 0.
            want = max(0, min(400_000_000 + want, rail))
            if int(r[0][c]) != want:
                print(f"  WRONG: fold e={e} rad_net={rn} -> {int(r[0][c])}, "
                      f"want {want}")
                ok = False
            if want != 400_000_000:
                nonzero += 1
            n += 1
    if nonzero < 20:
        print(f"  VACUOUS: only {nonzero} folds had a nonzero exact effect")
        ok = False
    print(f"  {'PASS' if ok else 'FAIL'} ({n} configs, {nonzero} non-trivial)")
    return ok


# ---------------------------------------------------------------- PART 3 ----
def part3_conduction(solver):
    print("PART 3 — Pass-2 conduction across a capacity straddling 1 unit")
    ok = True
    n = 0
    for e_thin in (-1, -4, -10, -16):
        temp, heat, his, fs, solid, vac, atm, ts = _blank(1, 4)
        solid[:] = True
        ts[:] = True
        his[0, :] = 5                      # steel-like, 32 units
        his[0, 1] = e_thin                 # the thin panel
        his[0, 2] = e_thin
        temp[0, 0] = 600_000_000
        # E/W faces conduct; N/S stay NO_FACE (a 1-row grid).
        fs[0, :, 2] = 4
        fs[0, :, 3] = 4
        fs[0, 3, 2] = DIALS["no_face"]
        fs[0, 0, 3] = DIALS["no_face"]
        r = _run_pair(solver, temp, heat, his, np.ascontiguousarray(fs),
                      solid, vac, atm, ts)
        if not _compare(f"conduct e_thin={e_thin}", *r):
            ok = False
        if np.array_equal(r[0], temp):
            print(f"  VACUOUS: conduct e_thin={e_thin} moved nothing")
            ok = False
        n += 1
    print(f"  {'PASS' if ok else 'FAIL'} ({n} configs)")
    return ok


# ---------------------------------------------------------------- PART 4 ----
def part4_trajectory(solver):
    print("PART 4 — 60-tick mixed-exponent trajectory, per-tick byte-identity")
    rng = np.random.default_rng(20260921)
    h, w = 12, 14
    temp, heat, his, fs, solid, vac, atm, ts = _blank(h, w)
    ts[:] = rng.random((h, w)) < 0.55
    solid[:] = ts
    his[:] = rng.integers(-6, 6, size=(h, w)).astype(np.int32)
    his[~ts] = 0
    temp[:] = (rng.random((h, w)) * 300_000_000).astype(np.int32)
    fs[:] = 4
    fs[~ts] = DIALS["no_face"]
    heat[:] = (rng.random((h, w)) * 50_000_000).astype(np.int32)
    heat[~ts] = 0
    fs = np.ascontiguousarray(fs)

    ok = True
    moved = 0
    t = np.ascontiguousarray(temp)
    for tick in range(60):
        r = _run_pair(solver, t, heat, his, fs, solid, vac, atm, ts)
        if not _compare(f"trajectory tick {tick}", *r):
            ok = False
            break
        if not np.array_equal(r[0], t):
            moved += 1
        t = r[0]
    if moved < 30:
        print(f"  VACUOUS: the field moved on only {moved}/60 ticks")
        ok = False
    print(f"  {'PASS' if ok else 'FAIL'} (moved on {moved}/60 ticks)")
    return ok


def main():
    if not bp.cuda_available():
        print("M1_RESULT: FAIL (no CUDA device)")
        return 1
    solver = _make_solver()
    ok = True
    ok &= part1_deposit(solver)
    ok &= part2_radiation_fold(solver)
    ok &= part3_conduction(solver)
    ok &= part4_trajectory(solver)
    print(f"M1_RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
