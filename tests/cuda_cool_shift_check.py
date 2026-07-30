"""COOL-SHIFT AXIS — CPU<->CUDA lockstep gate at tolerance ZERO (GPU subprocess).

Gate (d) for the cool-shift axis (2026-07-30): the ambient-decay shift moved
from ONE global (`[physics.thermal] COOL_SHIFT`) to a PER-MATERIAL column
projected onto the per-tile `GameMap.cool_shift` grid, and the vacuum-exposed
shift is now derived from that same per-tile value by the single global offset
`COOL_SHIFT - COOL_SHIFT_VACUUM`, floored at `SHIFT_MIN`. That is MEDIUM-TEST
SITE 6/6 on both backends (`temperature_solver.cpp` Pass 3 and
`cuda_temperature.cu`'s `temp_cool`), so the two must agree bit-for-bit on a
map whose per-tile shifts are NOT uniform — the case a stale kernel would miss.

  PART 1 — ISOLATED, shift-divergent: synthetic configs with a NON-UNIFORM
  per-tile `cool_shift` grid over every structural mask shape (including
  vacuum-exposed tiles, so the offset rule and its floor are exercised), wind on
  and off. CPU reference = the real bound `TemperatureSolver`; GPU =
  `bp.cuda_temperature_step`; both given the SAME grid. `temperature`
  byte-identity + T_MAX_PHYS rail parity, tol 0.
  NON-VACUOUSNESS CONTROL: the same GPU call with the grid OMITTED (nullptr ->
  the `cool_shift` SCALAR, i.e. the pre-axis single-global behaviour) MUST differ
  from the CPU reference. If that control ever passed, the gate proves nothing.

  PART 1b — the FLOOR and the OFFSET, checked against an independent Python
  reference implementation of the rule (not against the C++ that implements it).

  PART 2 — engine A/B, STEP path: a furniture-burn world whose materials carry
  DIFFERENT `cool_shift` values (furniture 12, wood 9, hull/steel 5) via runtime
  CFG overrides, driven 40 ticks through the real `PhysicsRunner.step` ->
  `step_tail` with ONLY the temperature backend flipped. All synced fields +
  rail counters byte-identical every tick. Plus a grid-is-live control on the
  real end-of-run state.

  PART 3 — engine A/B, RESIDENT path: the same scenario, CPU (residency off,
  backends off) vs residency ON + every backend ON, 40 ticks tol 0, and the
  resident device set asserted to carry `cool_shift`.

Prints ``COOL_SHIFT_RESULT: PASS``/``FAIL`` and exits 0/1.
"""
from __future__ import annotations

import sys

import numpy as np

import breach_physics as bp   # the CUDA build (sys.path[0] == cpp/build_cuda)

# Reuse the P2 harness's synthetic-field / mask / engine-A-B machinery verbatim,
# so the two gates cover the same structural shapes and any drift shows up in
# both at once.
import cuda_thermal_mass_check as TM

FP_ONE = 65536
DIALS = dict(TM.DIALS)                 # no_face 63, cool_shift 5, vacuum 3, ...
SHIFT_MIN = 2                          # config [physics.thermal] SHIFT_MIN
VAC_OFFSET = DIALS["cool_shift"] - DIALS["cool_shift_vacuum"]


# ===========================================================================
# PART 1 — isolated, per-tile shift divergent
# ===========================================================================
def _shift_grid(h, w, rng, kind):
    """A per-tile `cool_shift` grid. `uniform` reproduces the pre-axis global."""
    if kind == "uniform":
        return np.full((h, w), DIALS["cool_shift"], dtype=np.int32)
    if kind == "two_material":
        g = np.full((h, w), 5, dtype=np.int32)
        g[(np.arange(h)[:, None] + np.arange(w)[None, :]) % 3 == 0] = 12
        return np.ascontiguousarray(g)
    if kind == "floor_heavy":
        # Everything AT or just above the floor, so the vacuum offset clamps.
        return np.ascontiguousarray(
            rng.integers(SHIFT_MIN, SHIFT_MIN + 3, size=(h, w)).astype(np.int32))
    if kind == "full_range":
        return np.ascontiguousarray(
            rng.integers(SHIFT_MIN, 21, size=(h, w)).astype(np.int32))
    raise AssertionError(kind)


def _run_pair(solver, f, solid, is_vacuum, ts, csg):
    """CPU reference vs GPU kernel on identical copies, BOTH given `csg`; plus a
    GPU control with the grid OMITTED (the pre-axis scalar fallback)."""
    t_cpu = np.ascontiguousarray(f["temperature"].copy())
    c0 = int(solver.t_max_phys_hits)
    solver.step(t_cpu, f["heat"], f["his"], f["fs"], solid, is_vacuum,
                f["atmosphere"], wind_x=f["wind_x"], wind_y=f["wind_y"],
                dt=f["dt"], n_bulk=f["n_bulk"], thermal_solid=ts,
                cool_shift_grid=csg)
    hits_cpu = int(solver.t_max_phys_hits) - c0

    t_gpu = np.ascontiguousarray(f["temperature"].copy())
    hits_gpu = int(bp.cuda_temperature_step(
        t_gpu, f["heat"], f["his"], f["fs"], solid, is_vacuum, f["atmosphere"],
        n_bulk=f["n_bulk"], wind_x=f["wind_x"], wind_y=f["wind_y"], dt=f["dt"],
        thermal_solid=ts, cool_shift_grid=csg, cool_shift_floor=SHIFT_MIN,
        **DIALS))

    t_ctl = np.ascontiguousarray(f["temperature"].copy())
    bp.cuda_temperature_step(
        t_ctl, f["heat"], f["his"], f["fs"], solid, is_vacuum, f["atmosphere"],
        n_bulk=f["n_bulk"], wind_x=f["wind_x"], wind_y=f["wind_y"], dt=f["dt"],
        thermal_solid=ts, cool_shift_grid=None, cool_shift_floor=SHIFT_MIN,
        **DIALS)
    return t_cpu, hits_cpu, t_gpu, hits_gpu, t_ctl


def _solver():
    s = TM._make_solver()
    s.cool_shift_floor = SHIFT_MIN
    return s


def part1_isolated() -> bool:
    print("PART 1 — isolated CPU vs GPU with a NON-UNIFORM per-tile cool_shift "
          "grid (every mask shape, vacuum offset + floor exercised):")
    ok = True
    solver = _solver()
    rng = np.random.default_rng(20260731)
    n_cfg = 0
    total_hits = 0
    control_differed = 0
    divergent_grids = 0
    exposed_seen = 0

    sizes = [(16, 16), (24, 20), (1, 40), (40, 1), (13, 17), (8, 8)]
    kinds = ["all_solid", "all_vacuum", "all_air", "mixed", "hull"]
    grids = ["uniform", "two_material", "floor_heavy", "full_range"]
    for (h, w) in sizes:
        for kind in kinds:
            for gkind in grids:
                for wind in (False, True):
                    solid, is_vacuum = TM._mask_grid(h, w, rng, kind)
                    ts = TM._crate_mask(solid, rng, "scatter")
                    csg = _shift_grid(h, w, rng, gkind)
                    fld = TM._synth_fields(h, w, rng, solid, is_vacuum,
                                           thin=True, wind=wind)
                    if not (csg == DIALS["cool_shift"]).all():
                        divergent_grids += 1
                    if is_vacuum.any():
                        exposed_seen += 1
                    t_cpu, hc, t_gpu, hg, t_ctl = _run_pair(
                        solver, fld, solid, is_vacuum, ts, csg)
                    tag = f"{h}x{w}/{kind}/grid={gkind}/wind={wind}"
                    ok &= TM._compare(tag, t_cpu, hc, t_gpu, hg)
                    total_hits += hc
                    n_cfg += 1
                    if not np.array_equal(t_cpu, t_ctl):
                        control_differed += 1

    if total_hits == 0:
        ok = False
        print("  COVERAGE HOLE: the T_MAX_PHYS rail never engaged")
    if divergent_grids == 0:
        ok = False
        print("  COVERAGE HOLE: the shift grid was never non-uniform")
    if exposed_seen == 0:
        ok = False
        print("  COVERAGE HOLE: no vacuum-exposed configs — the offset rule was "
              "never taken")
    if control_differed == 0:
        ok = False
        print("  VACUOUS: the grid-omitted control never differed from the CPU "
              "reference — the per-tile grid does not reach the kernel")
    if ok:
        print(f"  all {n_cfg} configs bit-identical on temperature + rail hits "
              f"({divergent_grids} with a non-uniform grid, {exposed_seen} with "
              f"vacuum-exposed tiles, rail engagements {total_hits}); the "
              f"grid-omitted control diverged in {control_differed}/{n_cfg} "
              f"configs (proof the grid is live).")
    return ok


# ===========================================================================
# PART 1b — the offset + floor rule, vs an INDEPENDENT Python reference
# ===========================================================================
def _py_cool(temperature, thermal_solid, is_vacuum, atmosphere, csg):
    """The rule, re-implemented from the DESIGN, not transcribed from the C++:
        base    = csg[i]
        exposed = any in-bounds 4-neighbour is vacuum or atmosphere < thresh
        shift   = exposed ? max(SHIFT_MIN, base - VAC_OFFSET) : base
        T      -= sign-symmetric (|T| >> shift)
    """
    h, w = temperature.shape
    thresh = int(round(DIALS["o2_vacuum_thresh"] * FP_ONE))
    out = temperature.copy()
    for y in range(h):
        for x in range(w):
            if not thermal_solid[y, x]:
                continue
            t = int(temperature[y, x])
            if t == 0:
                continue
            exposed = False
            for dy, dx in ((-1, 0), (1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if is_vacuum[ny, nx] or int(atmosphere[ny, nx]) < thresh:
                        exposed = True
                        break
            base = int(csg[y, x])
            shift = max(SHIFT_MIN, base - VAC_OFFSET) if exposed else base
            loss = -((-t) >> shift) if t < 0 else (t >> shift)
            out[y, x] = t - loss
    return out


def part1b_offset_rule() -> bool:
    print("PART 1b — the vacuum OFFSET + FLOOR vs an independent Python "
          "reference (conduction disabled, cooling isolated):")
    solver = _solver()
    rng = np.random.default_rng(4242)
    ok = True
    clamped_any = 0
    exposed_any = 0
    for trial in range(24):
        h, w = 12, 15
        solid = np.ascontiguousarray(rng.random((h, w)) < 0.5)
        is_vacuum = np.ascontiguousarray((~solid) & (rng.random((h, w)) < 0.3))
        ts = np.ascontiguousarray(solid | (rng.random((h, w)) < 0.25))
        csg = np.ascontiguousarray(
            rng.integers(SHIFT_MIN, 14, size=(h, w)).astype(np.int32))
        temperature = np.ascontiguousarray(
            rng.integers(-(1 << 24), 1 << 26, size=(h, w)).astype(np.int32))
        heat = np.zeros((h, w), dtype=np.int32)
        his = np.full((h, w), 3, dtype=np.int32)
        fs = np.full((h, w, 4), DIALS["no_face"], dtype=np.int32)  # no conduction
        atmosphere = np.ascontiguousarray(
            np.where(is_vacuum, 0, FP_ONE).astype(np.int32))
        clamped_any += int(((csg - VAC_OFFSET) < SHIFT_MIN).sum())
        exposed_any += int(is_vacuum.sum())

        want = _py_cool(temperature, ts, is_vacuum, atmosphere, csg)

        got_cpu = np.ascontiguousarray(temperature.copy())
        solver.step(got_cpu, heat, his, fs, solid, is_vacuum, atmosphere,
                    thermal_solid=ts, cool_shift_grid=csg)
        got_gpu = np.ascontiguousarray(temperature.copy())
        bp.cuda_temperature_step(
            got_gpu, heat, his, fs, solid, is_vacuum, atmosphere,
            thermal_solid=ts, cool_shift_grid=csg, cool_shift_floor=SHIFT_MIN,
            **DIALS)

        # Pass 0a zeroes gas-T at OPEN vacuum cells; the reference does not model
        # that, so compare only where the medium is thermally solid.
        m = ts
        if not np.array_equal(got_cpu[m], want[m]):
            ok = False
            print(f"  trial {trial}: CPU disagrees with the Python reference on "
                  f"{int(np.count_nonzero(got_cpu[m] != want[m]))} cells")
        if not np.array_equal(got_gpu[m], want[m]):
            ok = False
            print(f"  trial {trial}: GPU disagrees with the Python reference on "
                  f"{int(np.count_nonzero(got_gpu[m] != want[m]))} cells")
    if clamped_any == 0:
        ok = False
        print("  COVERAGE HOLE: the floor never clamped")
    if exposed_any == 0:
        ok = False
        print("  COVERAGE HOLE: no vacuum-exposed tiles")
    if ok:
        print(f"  24 trials: CPU and GPU both match the independent reference "
              f"cell for cell ({clamped_any} floor-clamping tiles, "
              f"{exposed_any} vacuum cells).")
    return ok


# ===========================================================================
# PARTS 2 / 3 — the engine A/B, with a genuinely non-uniform grid
# ===========================================================================
_MAT_SHIFTS = {"furniture": 12, "wood": 9, "hull": 5, "steel": 5,
               "glass": 7, "door": 9, "door_closed": 9, "air": 5}


class _MaterialShiftOverride:
    """Runtime CFG override of the per-material `cool_shift` column, so the
    engine A/B runs on a genuinely NON-UNIFORM per-tile grid (a uniform grid
    would make the whole gate a re-run of the P2 gate)."""

    def __enter__(self):
        from config import CFG
        self._old = []
        for name, v in _MAT_SHIFTS.items():
            row = getattr(CFG.materials, name)
            self._old.append((row, int(getattr(row, "cool_shift"))))
            setattr(row, "cool_shift", v)
        return self

    def __exit__(self, *exc):
        for row, v in self._old:
            setattr(row, "cool_shift", v)
        return False


def _grid_is_live(g_cpu, g_gpu):
    """Direct proof on the REAL end-of-run state that the per-tile grid reaches
    the kernel: the same GPU call with the grid omitted (the pre-axis scalar)
    must give a different temperature field."""
    from config import CFG
    thermal = CFG.physics.thermal
    dials = dict(
        no_face=int(getattr(thermal, "NO_FACE", 63)),
        cool_shift=int(getattr(thermal, "COOL_SHIFT", 5)),
        cool_shift_vacuum=int(getattr(thermal, "COOL_SHIFT_VACUUM", 3)),
        o2_vacuum_thresh=float(getattr(thermal, "o2_vacuum_thresh", 0.3)),
        c_v=float(getattr(thermal, "c_v", 1.0)),
        n_floor_heat=float(getattr(thermal, "n_floor_heat", 0.05)),
        gas_advection_rate=float(getattr(thermal, "gas_advection_rate", 900.0)),
        t_max_phys=float(getattr(thermal, "T_MAX_PHYS", 16000.0)),
    )
    uniq = sorted(set(int(v) for v in np.unique(g_cpu.cool_shift)))
    if len(uniq) < 2:
        print(f"  VACUOUS: the per-tile cool_shift grid is uniform {uniq}")
        return False
    heat = np.ascontiguousarray(np.full(g_cpu.temperature.shape,
                                        TM._q(200.0), dtype=np.int32))
    out = {}
    for tag, grid in (("grid", g_cpu.cool_shift), ("fallback", None)):
        t = np.ascontiguousarray(g_cpu.temperature.copy())
        bp.cuda_temperature_step(
            t, heat, g_cpu.heat_inv_shift, g_cpu.face_shift,
            g_cpu.solid, g_cpu.is_vacuum, g_cpu.atmosphere,
            n_bulk=None, wind_x=g_cpu.wind_x, wind_y=g_cpu.wind_y,
            dt=1.0 / 24.0, thermal_solid=g_cpu.thermal_solid,
            cool_shift_grid=grid,
            cool_shift_floor=int(getattr(thermal, "SHIFT_MIN", 2)), **dials)
        out[tag] = t
    if np.array_equal(out["grid"], out["fallback"]):
        print("  VACUOUS: on the real state the GPU gave the SAME answer with "
              "and without the per-tile cool_shift grid")
        return False
    n = int(np.count_nonzero(out["grid"] != out["fallback"]))
    print(f"  grid-is-live control: per-tile shifts on this map are {uniq}; the "
          f"scalar-fallback GPU call differs from the grid one on {n} cells.")
    return True


def _resident_grid_present(g_cpu, g_gpu):
    good = True
    if not (bool(g_gpu.residency_on()) and hasattr(g_gpu, "_dev")):
        print("  the GPU world never entered residency mode — VACUOUS")
        good = False
    elif "cool_shift" not in g_gpu.device_ptrs():
        print("  `cool_shift` is NOT in the resident device set "
              "(GameMap._RESIDENT_MASKS) — the resident half is missing")
        good = False
    else:
        dev = g_gpu._dev["cool_shift"]
        host = g_gpu.cool_shift
        if tuple(dev.shape) != tuple(host.shape) or dev.dtype != host.dtype:
            print(f"  resident cool_shift {dev.dtype}{tuple(dev.shape)} != host "
                  f"{host.dtype}{tuple(host.shape)}")
            good = False
        else:
            print(f"  resident device set carries cool_shift ({dev.dtype}, "
                  f"{tuple(dev.shape)}); distinct shifts on the map: "
                  f"{sorted(set(int(v) for v in np.unique(host)))}.")
    return good


def part2_step_path() -> bool:
    print("PART 2 — engine A/B, STEP path, NON-UNIFORM per-tile cool_shift "
          "(furniture 12 / wood+doors 9 / glass 7 / hull+steel 5), 40 ticks:")
    def gpu_setup():
        TM._residency(False)
        TM._only_temperature_backend(True)

    def cpu_setup():
        TM._residency(False)
        TM._only_temperature_backend(False)

    with _MaterialShiftOverride():
        ok = TM._run_ab("PART 2 (step)", 40, gpu_setup, cpu_setup,
                        extra_check=_grid_is_live)
    TM._residency(False)
    TM._set_backends(False)
    return ok


def part3_resident_path() -> bool:
    print("PART 3 — engine A/B, RESIDENT path (residency + every backend ON vs "
          "CPU), NON-UNIFORM per-tile cool_shift, 40 ticks:")
    try:
        import cupy  # noqa: F401
    except Exception as exc:      # pragma: no cover — reported, never skipped
        print(f"  cupy unavailable ({exc}) — the resident leg CANNOT be gated")
        return False

    def gpu_setup():
        TM._residency(True)
        TM._set_backends(True)

    def cpu_setup():
        TM._residency(False)
        TM._set_backends(False)

    res0 = int(bp.eos_resident_calls())
    with _MaterialShiftOverride():
        ok = TM._run_ab("PART 3 (resident)", 40, gpu_setup, cpu_setup,
                        extra_check=_resident_grid_present)
    res_delta = int(bp.eos_resident_calls()) - res0
    if res_delta < 40 and ok:
        print(f"  the resident EOS ran only {res_delta}/40 ticks — the resident "
              f"path was not really exercised")
        ok = False
    elif ok:
        print(f"  resident EOS confirmed live ({res_delta} resident calls).")
    TM._residency(False)
    TM._set_backends(False)
    return ok


def main() -> int:
    if not getattr(bp, "HAS_CUDA", False) or not bp.cuda_available():
        print("COOL_SHIFT_RESULT: FAIL (no CUDA build / device)")
        return 1
    print("device:", bp.cuda_device_info())
    p1 = part1_isolated()
    p1b = part1b_offset_rule()
    p2 = part2_step_path()
    p3 = part3_resident_path()
    if p1 and p1b and p2 and p3:
        print("COOL_SHIFT_RESULT: PASS")
        return 0
    print("COOL_SHIFT_RESULT: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
