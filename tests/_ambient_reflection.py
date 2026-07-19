"""Robust reflection harness for the planetside AMBIENT ring — spec §6 gate 3
(boundary_conditions_spec_2026-07-19). Reused by the B3c u-damping calibration +
the committed gate test, and by B4's CUDA lockstep.

THE BIG-MAP-REFERENCE PROTOCOL (immune to phase/sign/corner aliasing): run the
SAME deterministic detonation, at the SAME interior location, in two maps —

  (a) the TEST map, whose ambient ring is `test_half` tiles from the detonation;
  (b) a REFERENCE map, whose ring is pushed `ref_half` tiles away, far enough
      that no reflection returns to the probe within the window — the reference
      run IS "the atmosphere continued forever."

Any difference between the two at an interior probe region, over the whole
window, is REFLECTION off the near ring by construction. The metric is

    reflection = max over (probe region × window) |P_test − P_ref|
               / max over (probe region × window) |P_ref − P_amb|

i.e. the reflected transient normalized by the incident amplitude. Fully
deterministic (fixed integer detonation, fixed geometry, no RNG) — Q16 sim path.
Front speed is ~c·dt/dx ≈ 37.5 tiles/tick (c=300 m/s, dt=1/24 s, dx=1/3 m), so
`ref_half` must exceed ~c·dt·window for the reference to stay reflection-free;
`assert_reference_clean` checks that empirically.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import level_loader
from simulation.ambient import derive_ambient
from simulation.gamemap import GameMap
from simulation.physics_runner import PhysicsRunner

DT = 1.0 / 24.0
FRONT_TILES_PER_TICK = 37.5   # c·dt/dx at c=300, dt=1/24, dx=1/3


def _ambient_map(half, cfg):
    """A square planetside map: a 1-cell SPACE ring border (v1 code 0) around an
    open-air interior (code 9), `half` tiles from centre to ring."""
    n = 2 * half + 1
    tm = np.full((n, n), 9, dtype=np.int32)
    tm[0, :] = tm[-1, :] = tm[:, 0] = tm[:, -1] = 0
    ld = level_loader.LevelData(
        name="refl", version="1", path=Path("."), tilemap=tm,
        tile_size_m=1.0 / 3.0, diffuse_path=Path("."),
        boundary="ambient", ambient=cfg)
    return GameMap(ld)


def _detonate(g, c, amp):
    """A fixed, deterministic 5×5 overpressure+heat core at (c, c) — the same
    integer state in every run (pure Q16, no RNG)."""
    o2 = g.gases.name_to_id["o2"]
    n2 = g.gases.name_to_id["inert_n2"]
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            g.temperature[c + dy, c + dx] = int(3000 * 65536)
            g.gas[o2][c + dy, c + dx] = int(g._ambient.n_o2_q * amp)
            g.gas[n2][c + dy, c + dx] = int(g._ambient.n_n2_q * amp)
            g.atmosphere[c + dy, c + dx] = int(g._ambient.pin_q * amp)


def reflection_ratio(bp, *, k_max=0, sponge_width=8, test_half=30,
                     ref_half=170, window=5, probe_r=3, det_amp=8):
    """Return ``(reflection_ratio, numerator_raw, incident_raw)`` for the given
    u-damping ``k_max`` (Q16). The probe is a centred (2·probe_r+1)² block (the
    detonation focus — the incident is identical in both maps there, so the
    difference is pure reflection). σ ships at 0 (measured-reflective); only the
    velocity band (``sponge_u_damp = k_max``) is exercised."""
    pin = derive_ambient().pin_q
    cfg = derive_ambient(sponge_width=sponge_width, sponge_strength=0,
                         sponge_u_damp=int(k_max))
    gt = _ambient_map(test_half, cfg)
    gr = _ambient_map(ref_half, cfg)
    rt = PhysicsRunner(bp)
    rr = PhysicsRunner(bp)
    _detonate(gt, test_half, det_amp)
    _detonate(gr, ref_half, det_amp)

    def probe(g, c):
        return g.atmosphere[c - probe_r:c + probe_r + 1,
                            c - probe_r:c + probe_r + 1].astype(np.int64)

    num = den = 0
    for _ in range(window):
        rt.step(gt, DT)
        rr.step(gr, DT)
        num = max(num, int(np.abs(probe(gt, test_half) - probe(gr, ref_half)).max()))
        den = max(den, int(np.abs(probe(gr, ref_half) - pin).max()))
    ratio = (num / den) if den else 0.0
    return ratio, num, den


def min_clean_ref_half(window, margin=1.25, probe_r=3):
    """The smallest ``ref_half`` for which the reference stays reflection-free
    over ``window`` ticks: the front travels ~FRONT·window one-way, and a
    reflection round-trips, so ref_half ≥ FRONT·window·margin is safe."""
    return int(FRONT_TILES_PER_TICK * window * margin) + probe_r + 5
