"""P6d — THE LIGHT CHANNELS' TRANSPORT SETTING, held GPU == CPU on the LIVE
conductor (ray-engine-v2; docs/ray_engine_v2_p6d_light_transport_setting_
brief_2026-09-29.md section 3, gate G2). Runs INSIDE the GPU subprocess; the
pytest wrapper is tests/test_cuda_p6d_light_transport.py.

PROPERTY: with ``PhysicsEngine.light_transport`` set to ``RadiationSweep.
SHEAR`` (the P6d setting; gate 18 and tests/test_radiation_sweep_light.py
already hold the sweep itself to the reference on both transports -- this
gate is the LIVE PATH, the one place P6d touched: the ONE sweep invocation,
``run_sweep_``, both call sites) the GPU twin computes the SAME integers as
the CPU sweep on a real ``Simulation``: every synced GameMap array (the light
planes included), the engine's light books and its light-stream telemetry,
per tick, tol 0 -- on the normal per-call tick, on the resident tick, and on
``relight()``.

Scene: the playground level with its own three emergency lamps (the level's
``[[light]]`` entities, through the real assembly ``renderer.frame_lights.
frame_light_specs`` -> ``cone_rows``, the same seam the game uses -- P6d
brief §3, "the playground with its lamps"), a burning wood tile (test_
radiation_sweep_shadow_wiring.py's live scene, ``_burning_playground``,
reused from cuda_radiation_sweep_check.py per the Benches canonical-system
rule), and a smoke puff around the fire (the SHIPPED [gases.smoke] heat_absorb
/ light_absorb columns -- P5c/P6a's live smoke term, so both the heat and the
light smoke arms ride this scene). Light is REQUESTED through the real
``Simulation.set_light`` facade, never by poking the engine directly.

BREAKS IF: run_sweep_ ignores PhysicsEngine.light_transport on either backend
(reverting cpp/src/physics_engine.cpp's ``lc.transport = this->light_transport``
to a literal turns this red -- validated by hand during P6d development), or
the CUDA twin's light walk is not in fact topological for TRANSPORT_SHEAR (it
already reads ``light->transport`` at every site -- cpp/src/cuda_radiation_
sweep.cu -- so this is confirmation, not new plumbing).

Prints ``P6D_RESULT: PASS``/``FAIL`` and exits 0/1.
"""
from __future__ import annotations

import sys

import numpy as np

# Import the CUDA build FIRST so it is the cached `breach_physics` (the
# harness below inserts the CPU build dir on sys.path, which must not win).
import breach_physics as bp

_FAILS: list[str] = []


def _fail(msg: str) -> None:
    _FAILS.append(msg)
    print("  FAIL: " + msg)


# ---------------------------------------------------------------------------
# the scene: the playground, its lamps, the fire, a smoke puff -- shear
# ---------------------------------------------------------------------------
def _lit_smoky_playground():
    """The playground with a burning wood tile, its own three level lamps
    (via the real frame_lights assembly), a smoke puff around the fire (the
    shipped smoke absorption/glow columns), and light REQUESTED with the
    transport set to SHEAR (P6d's config setting, poked directly here --
    PhysicsRunner's config binding itself is gate G5, a separate pure-Python
    door test)."""
    from cuda_radiation_sweep_check import _burning_playground
    from renderer import frame_lights as fl
    from simulation import gas_fixed
    from simulation.gases import SMOKE

    sim, pick = _burning_playground()
    g = sim.gmap
    y0, x0 = pick
    air = (~g.thermal_solid) & (~g.solid) & (~g.is_vacuum)
    yy, xx = np.mgrid[0:g.material.shape[0], 0:g.material.shape[1]]
    cloud = air & (np.abs(yy - y0) <= 4) & (np.abs(xx - x0) <= 4)
    dens = gas_fixed.quantize(np.where((yy + xx) % 3 == 0, 0.27, 0.06))
    g.gas[SMOKE][cloud] = dens[cloud]

    h, w = g.solid.shape
    specs = fl.frame_light_specs(sim.level.lights, total_tick=0,
                                  sim_time_per_tick=1.0 / 24.0).specs
    cones = fl.cone_rows(specs, w, h, int(bp.L_FINE_BITS))
    if cones.shape[0] == 0:
        raise AssertionError("the playground's lamp entities produced no cone "
                             "rows -- the scene would light through the fire "
                             "alone, not 'with its lamps'")
    from config import CFG
    sky = fl.sky_for_level(sim.level.boundary, CFG, bp)
    sim.set_light(True, cones, sky)
    sim.physics_runner.engine.light_transport = bp.RadiationSweep.SHEAR
    return sim, pick


_ALL_BACKENDS = (
    "set_temperature_backend", "set_water_backend", "set_smoke_backend",
    "set_fire_backend", "set_radiation_backend",
    "set_bulk_flux_backend", "set_sl_advection_backend",
    "set_mg_solve_backend", "set_kick_compression_backend",
    "set_combustion_backend",
)
_RESIDENT_FIELDS = ("atmosphere", "wave_p", "wind_x", "wind_y", "temperature",
                    "heat", "fire", "wall_hp", "water_depth", "flow_vx",
                    "flow_vy", "gas", "ripple", "ripple_v", "gas_energy",
                    "rad_net_sweep", "rad_flux_sweep", "rad_amb_sweep",
                    "rad_fluence", "light_q", "light_flux_q", "light_glow")


def _compare(tag, t, s_cpu, s_gpu, fields=None) -> int:
    bad = 0
    a_cpu = ({k: getattr(s_cpu.gmap, k) for k in fields} if fields is not None
             else {k: v for k, v in vars(s_cpu.gmap).items() if isinstance(v, np.ndarray)})
    a_gpu = ({k: getattr(s_gpu.gmap, k) for k in fields} if fields is not None
             else {k: v for k, v in vars(s_gpu.gmap).items() if isinstance(v, np.ndarray)})
    for k in sorted(a_cpu):
        x, y = a_cpu[k], a_gpu[k]
        eq = (np.array_equal(x, y, equal_nan=True) if x.dtype.kind == "f"
              else np.array_equal(x, y))
        if not eq:
            bad += 1
            _fail(f"{tag} tick {t}: gmap.{k} differs "
                  f"({int(np.count_nonzero(x != y))} cells)")
    bc = s_cpu.physics_runner.engine.radiation.light_books
    bg = s_gpu.physics_runner.engine.radiation.light_books
    if dict(bc) != dict(bg):
        bad += 1
        _fail(f"{tag} tick {t}: light books cpu={dict(bc)} gpu={dict(bg)}")
    return bad


def _run(tag, n_ticks, gpu_on, gpu_off, fields=None) -> None:
    fails_before = len(_FAILS)
    s_cpu, pick = _lit_smoky_playground()
    s_gpu, _ = _lit_smoky_playground()
    if _compare(tag, "init", s_cpu, s_gpu, fields):
        return
    lit, calls = 0, 0
    for t in range(n_ticks):
        gpu_off()
        s_cpu.set_paused(False)
        s_cpu.step()
        c0 = int(bp.radiation_sweep_cuda_calls())
        gpu_on()
        s_gpu.set_paused(False)
        s_gpu.step()
        calls += int(bp.radiation_sweep_cuda_calls()) - c0
        gpu_off()
        lit += int(np.any(s_cpu.gmap.light_q != 0))
        if _compare(tag, t, s_cpu, s_gpu, fields) >= 4:
            print("  aborting after 4 divergent ticks")
            break
    if calls != n_ticks:
        _fail(f"{tag}: the GPU sweep ran {calls} times in {n_ticks} ticks — "
              "the dispatch did not fire every tick")
    if lit == 0:
        _fail(f"{tag}: light_q never non-zero — the live comparison is vacuous")
    verdict = "tol 0" if len(_FAILS) == fails_before else "FAILED"
    what = "every GameMap array" if fields is None else f"the {len(fields)} synced + light planes"
    print(f"  {tag}: {n_ticks} ticks, {verdict} on {what} + the light books; "
          f"{lit} lit ticks; GPU sweep calls {calls}; fire at {pick}")


def part1_per_call() -> None:
    print("PART G2a — per-call tick, radiation backend only, transport=SHEAR")
    from simulation import physics_runner

    def only_radiation(on):
        physics_runner.set_residency(False)
        for name in _ALL_BACKENDS:
            getattr(bp, name)(False)
        bp.set_radiation_backend(bool(on))

    _run("G2a", 20, lambda: only_radiation(True), lambda: only_radiation(False))


def part2_resident() -> None:
    print("PART G2b — resident tick, every backend but combustion, transport=SHEAR")
    from simulation import physics_runner

    def all_gpu_resident(on):
        physics_runner.set_residency(bool(on))
        for name in _ALL_BACKENDS:
            getattr(bp, name)(bool(on) and name != "set_combustion_backend")

    try:
        import cupy  # noqa: F401
    except Exception as exc:            # pragma: no cover -- reported, not skipped
        _fail(f"G2b: cupy unavailable ({exc}) — the resident leg cannot be gated")
        return
    _run("G2b", 12, lambda: all_gpu_resident(True), lambda: all_gpu_resident(False),
         fields=_RESIDENT_FIELDS)
    all_gpu_resident(False)


def part3_relight() -> None:
    print("PART G2c — relight(), transport=SHEAR")
    from simulation import physics_runner

    def only_radiation(on):
        physics_runner.set_residency(False)
        for name in _ALL_BACKENDS:
            getattr(bp, name)(False)
        bp.set_radiation_backend(bool(on))

    s_cpu, pick = _lit_smoky_playground()
    s_gpu, _ = _lit_smoky_playground()
    # a few ticks first, same idiom as G2a, so relight runs on a non-trivial
    # (already-lit, already-hot) state rather than the freshly-loaded one
    for _ in range(3):
        only_radiation(False)
        s_cpu.set_paused(False)
        s_cpu.step()
        only_radiation(True)
        s_gpu.set_paused(False)
        s_gpu.step()
    only_radiation(False)
    snap = {k: v.copy() for k, v in vars(s_cpu.gmap).items() if isinstance(v, np.ndarray)}
    only_radiation(False)
    s_cpu.physics_runner.relight(s_cpu.gmap)
    only_radiation(True)
    s_gpu.physics_runner.relight(s_gpu.gmap)
    only_radiation(False)
    bad = 0
    for k in ("light_q", "light_flux_q", "light_glow"):
        if not np.array_equal(getattr(s_cpu.gmap, k), getattr(s_gpu.gmap, k)):
            bad += 1
            _fail(f"G2c relight: {k} cpu != gpu")
    bc = s_cpu.physics_runner.engine.radiation.light_books
    bg = s_gpu.physics_runner.engine.radiation.light_books
    if dict(bc) != dict(bg):
        bad += 1
        _fail(f"G2c relight: light books cpu={dict(bc)} gpu={dict(bg)}")
    for k, v in snap.items():
        if k in ("light_q", "light_flux_q", "light_glow"):
            continue
        eq = (np.array_equal(v, getattr(s_cpu.gmap, k), equal_nan=True)
              if v.dtype.kind == "f" else np.array_equal(v, getattr(s_cpu.gmap, k)))
        if not eq:
            bad += 1
            _fail(f"G2c relight moved gmap.{k} on the CPU side (a tick's plane)")
    if not np.any(s_cpu.gmap.light_q != 0):
        _fail("G2c: relight produced no light — vacuous")
    print(f"  {'tol 0' if bad == 0 else 'FAILED'} on the three light planes + the "
          f"light books; relight moved no other array; fire at {pick}")


def main() -> int:
    try:
        dev = bp.cuda_device_info() if hasattr(bp, "cuda_device_info") else ""
        print(f"device: {dev}")
    except Exception:                                   # noqa: BLE001
        pass
    for part in (part1_per_call, part2_resident, part3_relight):
        try:
            part()
        except Exception as exc:                        # noqa: BLE001
            import traceback
            traceback.print_exc()
            _fail(f"{part.__name__} raised {type(exc).__name__}: {exc}")
    print(f"\nP6D_RESULT: {'PASS' if not _FAILS else 'FAIL'} ({len(_FAILS)} failures)")
    return 0 if not _FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
