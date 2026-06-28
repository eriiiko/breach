"""CUDA-S2 raycaster HEAT bit-identity check (runs inside the GPU subprocess).

The directional ray march deposits ``heat`` (Q16.16 int32) — the only
sim-affecting ray output (it feeds the temperature pass -> unit damage). This
gate proves the GPU march (cuda_raycaster_cast) produces a heat buffer
BYTE-FOR-BYTE equal to the shipped CPU march (Raycaster.cast_source_directional)
on a firestorm-with-smoke scenario, tol 0. The render channels (light_rgb /
light_dx/dy / smoke_glow) are deterministic-EXEMPT (float atomics + device expf
on the gas optics) and are NOT gated — only ``heat`` is.

The scenario exercises every branch the heat channel can take:

  (a) FIRE sources sitting ON heat-opaque tiles (heat_atten 1.0) — the
      source-tile self-occlusion SKIP (distance==0): a fire must radiate out of
      its own opaque tile, not be killed by it.
  (b) heat_atten WALLS across the rays' paths — occlusion (heat survival decays
      multiplicatively; a wall drives it to ~0 and blocks heat beyond).
  (c) SMOKE / multi-gas in the path — proves the gas Beer-Lambert `exp` (which
      lives on the RGB survival ONLY) never perturbs the heat-touched tile set:
      heat deposits gate on heat_survival>heat_cull (material-only).
  (d) MANY OVERLAPPING omni sources depositing into shared tiles — the
      SATURATING integer atomic (order-free non-negative saturating add), incl.
      tiles driven to INT32_MAX.

CPU and GPU each accumulate the SAME per-source list into a separately-zeroed
Q16.16 buffer; we assert np.array_equal on the int32 arrays.

Prints ``S2_RESULT: PASS``/``FAIL`` and exits 0/1.
"""
from __future__ import annotations

import sys

import numpy as np

# Import the CUDA build FIRST (cuda_harness bootstrap put cpp/build_cuda on the
# path) so `breach_physics` is the GPU build.
import breach_physics as bp

HEAT_SCALE = 65536          # must match cpp/src/raycaster.h HEAT_SCALE
INT32_MAX = (1 << 31) - 1


# ----------------------------------------------------------------------------
# Scenario construction
# ----------------------------------------------------------------------------
def _make_raycaster():
    """One Raycaster, shared by the CPU and GPU casts so light_cull / heat_cull /
    smoke_absorb_scale are identical on both paths (the GPU binding reads these
    three scalars off the passed raycaster)."""
    rc = bp.Raycaster()
    # Defaults (light_cull=heat_cull=0.01, smoke_absorb_scale=1.4) are fine; pin
    # them explicitly so the gate does not silently drift if a default changes.
    rc.light_cull = 0.01
    rc.heat_cull = 0.01
    rc.smoke_absorb_scale = 1.4
    return rc


def _fire_source(x, y, intensity, heat, max_range, ray_count, phase):
    """An omni fire-like heat source (mirrors physics_runner.cast_fire_heat:
    fixed ray_count, omni angle_spread, deterministic per-tile phase, jitter=0 —
    heat is sim-affecting so the source NEVER dithers)."""
    two_pi = 2.0 * np.pi
    s = bp.LightSource()
    s.x = float(x) + 0.5
    s.y = float(y) + 0.5
    s.max_range = float(max_range)
    s.ray_count = int(ray_count)
    s.angle_spread = float(two_pi)
    s.angle_center = float(phase)
    s.intensity = float(intensity)
    s.heat = float(heat)
    s.jitter = 0.0
    s.color = (1.0, 0.6, 0.2)   # render-only fire tint (discarded by the heat gate)
    return s


def _build_scenario(seed):
    """Return (h, w, sources, gas, gas_absorption, gas_scatter, light_atten,
    heat_atten). A pocket firestorm: a cluster of overlapping omni fires, some
    sitting on heat-opaque (flammable-solid) tiles, heat_atten walls partitioning
    the room, and a smoke/poison-gas cloud across the beams' paths."""
    rng = np.random.default_rng(seed)
    h, w = 28, 36

    # --- materials: light_atten (h,w,3) and heat_atten (h,w) ---
    light_atten = np.zeros((h, w, 3), np.float32)
    heat_atten = np.zeros((h, w), np.float32)

    # A few heat_atten WALLS (partial + full) cutting across the room (occlusion).
    heat_atten[:, 12] = 1.0          # full heat wall (column)
    heat_atten[10, :] = 0.3          # partial heat wall (row, "glass")
    heat_atten[:, 24] = 0.7          # stronger partial wall
    # A light-opaque-but-heat-clear smoked-glass strip (heat sails through, light
    # dies) — proves the 4-channel cull keeps the ray marching for heat.
    light_atten[:, 18, :] = 1.0
    # A scatter of random partial heat occluders.
    for _ in range(20):
        ry, rx = int(rng.integers(0, h)), int(rng.integers(0, w))
        heat_atten[ry, rx] = float(rng.uniform(0.1, 1.0))

    # --- gas: two gases (black smoke + green poison), a cloud across the path ---
    n_gases = 2
    gas = np.zeros((n_gases, h, w), np.float32)
    # Smoke cloud (gas 0): a soft blob in the middle.
    cy, cx = 14, 16
    yy, xx = np.mgrid[0:h, 0:w]
    blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 40.0).astype(np.float32)
    gas[0] = blob * 3.0
    # Poison (gas 1): a second offset blob.
    blob2 = np.exp(-((yy - 8) ** 2 + (xx - 26) ** 2) / 30.0).astype(np.float32)
    gas[1] = blob2 * 2.5
    # Per-gas per-channel tables (n_gases, 3): smoke greys, poison greens.
    gas_absorption = np.array([[1.0, 1.0, 1.0],
                               [0.9, 0.2, 0.9]], np.float32)
    gas_scatter = np.array([[0.6, 0.6, 0.6],
                            [0.1, 0.7, 0.1]], np.float32)

    # --- sources: an overlapping cluster of fires (the firestorm) ---
    two_pi = 2.0 * np.pi
    sources = []
    # (a) fires ON heat-opaque tiles (source-tile skip): mark these tiles opaque.
    fire_cells = [(14, 14), (15, 15), (13, 16), (14, 16), (16, 14),
                  (20, 20), (20, 21), (21, 20), (6, 6), (22, 30)]
    for k, (fy, fx) in enumerate(fire_cells):
        heat_atten[fy, fx] = 1.0     # fire burns on a flammable solid -> heat-opaque
        intensity = float(rng.uniform(0.5, 1.5))
        heat = float(rng.uniform(4.0, 1600.0))   # spans into the saturating regime
        rc_rays = 8
        phase = ((fx * 7 + fy * 13) % rc_rays) * (two_pi / rc_rays)
        sources.append(_fire_source(fx, fy, intensity, heat,
                                    max_range=float(rng.uniform(10.0, 22.0)),
                                    ray_count=rc_rays, phase=phase))

    # (d) a stack of HIGH-heat overlapping omni sources on ONE tile to push the
    #     shared cells to INT32_MAX (the saturating atomic clamp).
    for j in range(6):
        sources.append(_fire_source(18, 18, intensity=1.0, heat=1.0e6,
                                    max_range=8.0, ray_count=16,
                                    phase=j * 0.3))

    return (h, w, sources, gas, gas_absorption, gas_scatter,
            light_atten, heat_atten)


# ----------------------------------------------------------------------------
# Cast the whole source list on one backend into a zeroed Q16.16 heat buffer.
# ----------------------------------------------------------------------------
def _cast_cpu(rc, h, w, sources, gas, gabs, gsca, light_atten, heat_atten):
    heat = np.zeros((h, w), np.int32)
    rgb = np.zeros((h, w, 3), np.float32)
    dx = np.zeros((h, w), np.float32)
    dy = np.zeros((h, w), np.float32)
    glow = np.zeros((h, w, 3), np.float32)
    for s in sources:
        rc.cast_source_directional(
            s, rgb, dx, dy, gas, gabs, gsca, light_atten,
            heat=heat, smoke_glow=glow, heat_atten=heat_atten)
    return heat


def _cast_gpu(rc, h, w, sources, gas, gabs, gsca, light_atten, heat_atten):
    heat = np.zeros((h, w), np.int32)
    rgb = np.zeros((h, w, 3), np.float32)
    dx = np.zeros((h, w), np.float32)
    dy = np.zeros((h, w), np.float32)
    glow = np.zeros((h, w, 3), np.float32)
    for s in sources:
        bp.cuda_raycaster_cast(
            rc, s, rgb, dx, dy, gas, gabs, gsca, light_atten,
            heat=heat, smoke_glow=glow, heat_atten=heat_atten)
    return heat


def run() -> bool:
    print("HEAT bit-identity — GPU march vs CPU cast_source_directional "
          "(firestorm + smoke, tol 0):")
    ok = True
    n_scen = 0
    for seed in (20260628, 1, 7, 42, 99):
        (h, w, sources, gas, gabs, gsca,
         light_atten, heat_atten) = _build_scenario(seed)
        rc = _make_raycaster()

        heat_cpu = _cast_cpu(rc, h, w, sources, gas, gabs, gsca,
                             light_atten, heat_atten)
        heat_gpu = _cast_gpu(rc, h, w, sources, gas, gabs, gsca,
                             light_atten, heat_atten)
        n_scen += 1

        if not np.array_equal(heat_cpu, heat_gpu):
            ok = False
            mism = int(np.count_nonzero(heat_cpu != heat_gpu))
            idx = int(np.argmax(heat_cpu != heat_gpu))
            ry, rx = divmod(idx, w)
            print(f"  seed {seed}: {mism} MISMATCH "
                  f"(first @ ({ry},{rx}): cpu={heat_cpu.flat[idx]} "
                  f"gpu={heat_gpu.flat[idx]})")
        else:
            # Confirm the scenario actually deposited meaningful, varied heat
            # (incl. saturation) so the gate is not vacuously passing on zeros.
            nz = int(np.count_nonzero(heat_cpu))
            sat = int(np.count_nonzero(heat_cpu == INT32_MAX))
            peak = int(heat_cpu.max())
            print(f"  seed {seed}: bit-identical "
                  f"({nz} heated tiles, {sat} saturated@INT32_MAX, peak={peak}).")
            if nz == 0 or sat == 0:
                ok = False
                print(f"  seed {seed}: SCENARIO TOO WEAK "
                      f"(nz={nz}, saturated={sat}) — gate would be vacuous.")
    if ok:
        print(f"  all {n_scen} firestorm scenarios: GPU heat == CPU heat "
              f"byte-for-byte (incl. saturation, occlusion, smoke, source-skip).")
    return ok


def main() -> int:
    if not getattr(bp, "HAS_CUDA", False) or not bp.cuda_available():
        print("S2_RESULT: FAIL (no CUDA build / device)")
        return 1
    print("device:", bp.cuda_device_info())
    ok = run()
    if ok:
        print("S2_RESULT: PASS")
        return 0
    print("S2_RESULT: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
