"""FUEL-FRACTION AXIS — CPU<->CUDA lockstep, tolerance ZERO (runs in the GPU subprocess).

Gate (c) of the fuel-fraction axis (2026-07-30). The fire logistic's fuel term
stopped dividing by the single global ``[physics.fire] fuel_ref`` (which is
WOOD's hp) and now divides by THIS tile's own material hp, via a per-tile
``make_recip`` reciprocal plane (``GameMap.fuel_recip``). The GPU fire kernel
(``cuda_fire.cu`` ``fire_logistic``) must read the same plane and produce the
same bits — the fire pass is one of the two deliberately bit-identical O2-law
twins, and a stale kernel here would silently desync the backends on exactly the
maps this patch was written for (anything with furniture).

  PART 1 — ISOLATED. Random fuzz over sizes / structural regimes / fuel-plane
  SHAPES (uniform-at-the-global, uniform-off-global, two-material split, fully
  per-cell random, and the plane OMITTED), CPU ``FireSimulation.step`` vs GPU
  ``cuda_fire_step`` on identical copies: byte-for-byte on fire / temperature /
  smoke / wall_hp plus a SET-equal destroyed list. NON-VACUOUSNESS CONTROL: the
  same GPU call with the plane OMITTED (the pre-axis scalar) must DIVERGE from
  the CPU reference whenever the plane is non-trivial — otherwise the plane is
  not reaching the kernel and the whole part proves nothing.

  PART 2 — STEP path. The thermal-mass arc's 48x48 FURNITURE-BURN world (wood
  partition hp 60 + crate block hp 30 -> a genuinely NON-UNIFORM fuel plane),
  40 ticks through the real ``PhysicsRunner`` with ONLY the FIRE backend
  flipped, asserting every synced field + the EOS rail counters byte-identical
  every tick. A scripted crate burn-out at t=14 exercises the
  ``on_tile_changed`` patch of the plane. Plus a plane-is-live control on the
  REAL end-of-run state.

  PART 3 — RESIDENT path. Same scenario, CPU vs residency ON + all backends ON,
  40 ticks bit-identical, with the resident device set asserted to carry
  ``fuel_recip``.

Prints ``FUEL_FRACTION_RESULT: PASS``/``FAIL`` and exits 0/1.
"""
from __future__ import annotations

import sys

import numpy as np

# Import the CUDA build FIRST so it is the cached `breach_physics`.
import breach_physics as bp

import cuda_fire_check as FC      # the fire kernel's own A/B machinery
import cuda_thermal_mass_check as TM   # the furniture-burn engine A/B machinery

FP_ONE = 65536


def _q(x):
    x = np.asarray(x, dtype=np.float64)
    return np.int32(np.where(x >= 0, np.floor(x * FP_ONE + 0.5),
                             np.ceil(x * FP_ONE - 0.5)))


# ===========================================================================
# PART 1 — isolated kernel A/B on non-uniform fuel planes
# ===========================================================================
def _fuel_plane(rng, h, w, shape, fuel_ref):
    """The five plane shapes the gate sweeps. ``None`` means "no plane" (the
    scalar-fallback path, which must ALSO stay bit-identical CPU<->GPU)."""
    if shape == "none":
        return None
    if shape == "uniform_ref":            # == the retired global, exactly
        v = np.full((h, w), bp.fp_make_recip(fuel_ref), dtype=np.int64)
    elif shape == "uniform_off":          # a single, different material
        v = np.full((h, w), bp.fp_make_recip(fuel_ref / 2.0), dtype=np.int64)
    elif shape == "two_material":         # the real shape: wood + crates
        v = np.full((h, w), bp.fp_make_recip(fuel_ref), dtype=np.int64)
        v[rng.random((h, w)) < 0.45] = bp.fp_make_recip(30.0)
    elif shape == "per_cell":             # adversarial: every tile different
        hps = rng.uniform(1.0, 400.0, size=(h, w))
        v = np.array([[bp.fp_make_recip(float(x)) for x in row] for row in hps],
                     dtype=np.int64)
    elif shape == "with_zero":            # the hp == 0 sentinel, on live tiles
        v = np.full((h, w), bp.fp_make_recip(fuel_ref), dtype=np.int64)
        v[rng.random((h, w)) < 0.2] = 0
    else:
        raise AssertionError(shape)
    return np.ascontiguousarray(v)


_SHAPES = ("none", "uniform_ref", "uniform_off", "two_material", "per_cell",
           "with_zero")


def part1_isolated() -> bool:
    print("PART 1 — ISOLATED fire kernel, CPU vs GPU on every fuel-plane shape:")
    rng = np.random.default_rng(20260730)
    ok = True
    n_cfg = 0
    nonuniform = 0
    control_differed = 0
    control_eligible = 0
    zero_seen = 0
    burned = 0
    for (h, w) in ((7, 5), (16, 16), (23, 17), (1, 40), (40, 1), (33, 31)):
        for shape in _SHAPES:
            for wind in (0.0, 1.5):
                state = FC._make_random_state(rng, h, w, wind_mag=wind)
                # Force burn-through coverage in half the configs: a strip of
                # nearly-spent fuel, so the P5 wall-burn pass (the one that
                # CONSUMES the quantity this patch normalises) is exercised and
                # the destroyed-set comparison is not vacuous.
                if n_cfg % 2 == 0:
                    state["wall_hp"] = np.ascontiguousarray(state["wall_hp"].copy())
                    state["wall_hp"][state["flammable"]] = _q(0.002)
                fp, dials = FC.make_params()
                plane = _fuel_plane(rng, h, w, shape, dials["fuel_ref"])
                n_cfg += 1
                if plane is not None and len(np.unique(plane)) > 1:
                    nonuniform += 1
                if plane is not None and int((plane == 0).sum()):
                    zero_seen += 1

                sim = bp.FireSimulation()
                sim.params = fp
                c = {k: v.copy() for k, v in state.items()}
                d_cpu = sim.step(c["fire"], c["atmosphere"], c["n_o2"],
                                 c["n_total"], c["smoke"], c["wall_hp"],
                                 c["temperature"], c["wind_x"], c["wind_y"],
                                 c["is_wall"], c["is_vacuum"], c["flammable"],
                                 1.0 / 24.0, plane)
                g = {k: v.copy() for k, v in state.items()}
                d_gpu = bp.cuda_fire_step(
                    g["fire"], g["atmosphere"], g["n_o2"], g["n_total"],
                    g["smoke"], g["wall_hp"], g["temperature"], g["wind_x"],
                    g["wind_y"], g["is_wall"], g["is_vacuum"], g["flammable"],
                    1.0 / 24.0, **dials, fuel_recip=plane)
                burned += len(list(d_cpu))
                if not FC.compare(f"{h}x{w}/{shape}/wind{wind}",
                                  c, list(d_cpu), g, list(d_gpu)):
                    ok = False

                # NON-VACUOUSNESS: with the plane OMITTED the GPU must give a
                # DIFFERENT answer than the CPU-with-plane reference, whenever
                # the plane is not just the global.
                if shape in ("uniform_off", "two_material", "per_cell", "with_zero"):
                    control_eligible += 1
                    gg = {k: v.copy() for k, v in state.items()}
                    bp.cuda_fire_step(
                        gg["fire"], gg["atmosphere"], gg["n_o2"], gg["n_total"],
                        gg["smoke"], gg["wall_hp"], gg["temperature"],
                        gg["wind_x"], gg["wind_y"], gg["is_wall"],
                        gg["is_vacuum"], gg["flammable"], 1.0 / 24.0, **dials)
                    if not np.array_equal(gg["fire"], c["fire"]):
                        control_differed += 1

    if nonuniform == 0:
        ok = False
        print("  COVERAGE HOLE: the fuel plane was never non-uniform")
    if zero_seen == 0:
        ok = False
        print("  COVERAGE HOLE: the hp==0 sentinel never appeared in a plane")
    if burned == 0:
        ok = False
        print("  COVERAGE HOLE: no wall ever burned through")
    if control_differed == 0:
        ok = False
        print("  VACUOUS: the plane-omitted control never differed — the "
              "per-tile plane does not reach the kernel")
    if ok:
        print(f"  all {n_cfg} configs bit-identical on fire/temperature/smoke/"
              f"wall_hp + set-equal destroyed ({nonuniform} with a non-uniform "
              f"plane, {zero_seen} carrying the hp==0 sentinel, {burned} "
              f"burn-throughs); the plane-omitted control diverged in "
              f"{control_differed}/{control_eligible} eligible configs "
              f"(proof the plane is live).")
    return ok


# ===========================================================================
# PARTS 2 / 3 — the engine A/B on the furniture-burn world
# ===========================================================================
def _only_fire_backend(on: bool) -> None:
    """PART 2 isolates the patched kernel: ONLY the fire backend flips, so any
    divergence is the fire pass and nothing else."""
    bp.set_fire_backend(bool(on))


def _plane_is_live(g_cpu, g_gpu):
    """Direct proof on the REAL end-of-run state that the per-tile plane reaches
    the kernel: the same GPU fire call with the plane omitted (the pre-axis
    scalar) must give a different fire field."""
    flam = g_cpu.flammable
    uniq = sorted(set(int(v) for v in np.unique(g_cpu.fuel_recip[flam]))) if flam.any() else []
    if len(uniq) < 2:
        print(f"  VACUOUS: the per-tile fuel plane is uniform over fuel {uniq}")
        return False

    # The live dials, taken from a FireParams built the way PhysicsRunner builds
    # it: solver defaults, overridden by whatever [physics.fire] actually
    # carries. Reading CFG keys directly would break on any dial that lives only
    # as a solver default (temp_gain_scale, T_FLAME_MAX, ...).
    from config import CFG
    fp = bp.FireParams()
    fire_cfg = CFG.physics.fire
    dials = {}
    for k in FC.DIALS:
        v = getattr(fire_cfg, k, None)
        dials[k] = float(v) if v is not None else float(getattr(fp, k))
    dials["temp_scale"] = float(FP_ONE)
    n_total = np.ascontiguousarray(
        g_cpu.gas[np.asarray(g_cpu.gases.conservative, dtype=bool)].sum(
            axis=0).astype(np.int32))
    from simulation.gases import O2
    out = {}
    for tag, plane in (("plane", g_cpu.fuel_recip), ("fallback", None)):
        st = dict(
            fire=np.ascontiguousarray(g_cpu.fire.copy()),
            atmosphere=np.ascontiguousarray(g_cpu.atmosphere.copy()),
            n_o2=np.ascontiguousarray(g_cpu.gas[O2].copy()),
            n_total=n_total.copy(),
            smoke=np.ascontiguousarray(g_cpu.smoke.copy()),
            wall_hp=np.ascontiguousarray(g_cpu.wall_hp.copy()),
            temperature=np.ascontiguousarray(g_cpu.temperature.copy()),
            wind_x=np.ascontiguousarray(g_cpu.wind_x.copy()),
            wind_y=np.ascontiguousarray(g_cpu.wind_y.copy()),
        )
        # Seed some fire on the fuel so the logistic definitely steps (the tail
        # of a 40-tick run may already have snapped out).
        st["fire"][g_cpu.flammable] = _q(0.5)
        st["temperature"][g_cpu.flammable] = _q(900.0)
        bp.cuda_fire_step(
            st["fire"], st["atmosphere"], st["n_o2"], st["n_total"],
            st["smoke"], st["wall_hp"], st["temperature"], st["wind_x"],
            st["wind_y"], np.ascontiguousarray(g_cpu.solid),
            np.ascontiguousarray(g_cpu.is_vacuum),
            np.ascontiguousarray(g_cpu.flammable), 1.0 / 24.0,
            **dials, fuel_recip=(np.ascontiguousarray(plane)
                                 if plane is not None else None))
        out[tag] = st["fire"]
    if np.array_equal(out["plane"], out["fallback"]):
        print("  VACUOUS: on the real state the GPU gave the SAME answer with "
              "and without the per-tile fuel plane")
        return False
    n = int(np.count_nonzero(out["plane"] != out["fallback"]))
    print(f"  plane-is-live control: distinct fuel reciprocals over fuel on this "
          f"map are {uniq}; the scalar-fallback GPU call differs from the "
          f"plane one on {n} cells.")
    return True


def _resident_plane_present(g_cpu, g_gpu):
    good = True
    if not (bool(g_gpu.residency_on()) and hasattr(g_gpu, "_dev")):
        print("  the GPU world never entered residency mode — VACUOUS")
        good = False
    elif "fuel_recip" not in g_gpu.device_ptrs():
        print("  `fuel_recip` is NOT in the resident device set "
              "(GameMap._RESIDENT_MASKS) — the resident half is missing")
        good = False
    else:
        dev = g_gpu._dev["fuel_recip"]
        host = g_gpu.fuel_recip
        if tuple(dev.shape) != tuple(host.shape) or dev.dtype != host.dtype:
            print(f"  resident fuel_recip {dev.dtype}{tuple(dev.shape)} != host "
                  f"{host.dtype}{tuple(host.shape)}")
            good = False
        else:
            uniq = sorted(set(int(v) for v in np.unique(host)))
            print(f"  resident device set carries fuel_recip ({dev.dtype}, "
                  f"{tuple(dev.shape)}); distinct reciprocals on the map: {uniq}.")
    return good


def part2_step_path() -> bool:
    print("PART 2 — FURNITURE-BURN engine A/B, STEP path (only the FIRE backend "
          "flips), 40 ticks, all synced fields tol 0:")
    _only_fire_backend(True)
    if not bp.get_fire_backend():
        print("  fire backend flag did not take — cannot gate")
        return False
    _only_fire_backend(False)

    def gpu_setup():
        TM._residency(False)
        _only_fire_backend(True)

    def cpu_setup():
        TM._residency(False)
        _only_fire_backend(False)

    ok = TM._run_ab("PART 2 (step)", 40, gpu_setup, cpu_setup,
                    extra_check=_plane_is_live)
    TM._residency(False)
    TM._set_backends(False)
    return ok


def part3_resident_path() -> bool:
    print("PART 3 — engine A/B, RESIDENT path (residency + every backend ON vs "
          "CPU), NON-UNIFORM per-tile fuel plane, 40 ticks:")
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
    ok = TM._run_ab("PART 3 (resident)", 40, gpu_setup, cpu_setup,
                    extra_check=_resident_plane_present)
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
        print("FUEL_FRACTION_RESULT: FAIL (no CUDA build / device)")
        return 1
    print("device:", bp.cuda_device_info())
    p1 = part1_isolated()
    p2 = part2_step_path()
    p3 = part3_resident_path()
    if p1 and p2 and p3:
        print("FUEL_FRACTION_RESULT: PASS")
        return 0
    print("FUEL_FRACTION_RESULT: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
