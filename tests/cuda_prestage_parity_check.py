"""#29 P5 -- the DEVICE pre-stage equals the HOST pre-stage at N = 1.

docs/residency_batching_design_2026-11-07.md §5.1, "P5's first check": the
resident EOS stage no longer folds its substep count on the host. Its three
reductions (the T_abs maximum behind c_LOCAL, max|u|², the K·|∇P|·dt/N̂ scan)
are per-env DEVICE reductions, its per-cell planes (the cap² plane, the
absorb plane, coeffE / coeffS) DEVICE kernels, and one thread per env folds
n_sub and picks its fold-table row. This gate runs the host pre-stage (the
per-call path's own eos_host_prestage, host scans on) and the device pre-stage
(device_prestage, the resident tick's own code path, on a poisoned private
scratch) on the SAME tick-entry planes and compares n_sub (the env's slot, its
fold-table row and the batch max), c_LOCAL, dt_s_q and the five planes
exactly (``bp.eos_prestage_parity``). #217 §8.6.7: the row is (k, n_sub) --
the sub-tick count k (its slot, its row, the batch max) is compared too, and
the synthetic sweep must reach every k in {1, 2, 4, 8} and every n_sub.

PROPERTY: for any tick-entry state, the device pre-stage's outputs equal the
host pre-stage's bit for bit. A change that must break it: a device fold that
re-derives a scalar differently (e.g. dropping the ``+ 1`` count epsilon of
u_est, or reading T_max without the t_amb floor), a cap² branch that skips the
thermal-solid ambient rule, a coeff kernel reading another env's dt.

The states: the s8a blast room at several ticks of a CPU run (hot core, venting
wind, a blast source, a breach), a synthetic plane set driven to every
substep count the cap allows (the fold's whole range, incl. the N_SUB_MAX
clip and the cap² saturation branch), and ship_demo's match after 60 ticks.
Run through tests/test_cuda_prestage_parity.py (the cuda_harness subprocess).
"""
from __future__ import annotations

import sys

import numpy as np

import breach_physics as bp

import cuda_s8a_check as s8a


def _parity(label, runner, g, dt, thermal_solid=None) -> tuple[bool, int, int]:
    ts = g.thermal_solid if thermal_solid is None else thermal_solid
    bad, report = bp.eos_prestage_parity(
        runner.engine.eos, g.atmosphere, g.wind_x, g.wind_y, g.temperature,
        g.gas, g.gases.conservative, g.solid, g.is_vacuum, ts,
        g.dyn_permeability, g.dyn_wave_absorb, float(dt))
    head = report.splitlines()[0] if report else ""
    words = head.split()
    # "k K n_sub N c_local_q .. dtdx_s_q .." (#217 §8: the row is (k, n_sub))
    k = int(words[words.index("k") + 1]) if "k" in words else -1
    n_sub = int(words[words.index("n_sub") + 1]) if "n_sub" in words else -1
    if bad != 0:
        print(f"  {label}: FAIL ({bad} mismatches)\n{report}")
        return False, n_sub, k
    print(f"  {label}: device == host ({head})")
    return True, n_sub, k


def part_blast_room() -> bool:
    print("PART A -- the s8a blast room, CPU-stepped; the pre-stage compared at "
          "ticks 0, 1, 5, 6, 12, 20, 30:")
    s8a._residency(False); s8a._set_backends(False)
    runner, g, dt = s8a._build_scenario(72, 72)
    ok = True
    checks = {0, 1, 5, 6, 12, 20, 30}
    for t in range(31):
        if t in checks:
            g.stamp_units([])
            good, _, _ = _parity(f"tick {t:2d}", runner, g, dt)
            ok = ok and good
        s8a._blast_tick(runner, g, dt, t)
    return ok


def part_synthetic_range() -> bool:
    """Drive n_sub over the fold's whole range: a uniform wind of growing
    magnitude (max|u| -> u_est), hot cells (c_LOCAL and the cap² saturation
    branch), and a pressure step (the K·|∇P| scan)."""
    print("PART B -- synthetic states over the whole substep range (wind sweep, "
          "hot cells, a pressure step); every n_sub the cap allows must appear:")
    from simulation import atmosphere_fixed
    s8a._residency(False); s8a._set_backends(False)
    runner, g, dt = s8a._build_scenario(72, 72)
    eos = runner.engine.eos
    eos.dx = float(g.tile_size_m)
    g.stamp_units([])
    q = atmosphere_fixed.quantize_scalar
    seen = set()
    seen_k = set()
    ok = True
    rng = np.random.default_rng(29)
    # A calm base: ambient temperature, a uniform pressure (the blast removed),
    # so the wind and the patches below set the substep count.
    g.temperature[:] = 0
    open_cells = ~g.solid
    g.atmosphere[open_cells] = int(np.median(g.atmosphere[open_cells]))
    base_t = g.temperature.copy()
    base_p = g.atmosphere.copy()
    # A thermal-solid (crate) patch inside the hot patch: the cap² fold gives a
    # thermal solid the AMBIENT cap however hot it is (VELOCITY-CLAMP D4).
    ts = g.thermal_solid.copy()
    ts[12:16, 12:20] = True
    ts &= ~g.solid
    for k, u in enumerate(np.linspace(0.0, 40.0, 64)):
        g.wind_x[:] = int(q(float(u)))
        g.wind_y[:] = int(q(float(u) * 0.3))
        g.temperature[:] = base_t
        g.atmosphere[:] = base_p
        if k % 3 == 0:   # a hot patch: c_LOCAL above ambient, cap² saturates
            g.temperature[10:20, 10:30] += int(q(float(rng.uniform(500.0, 12000.0))))
        if k % 4 == 1:   # a pressure step across the room
            g.atmosphere[:, :32] += int(q(float(rng.uniform(0.0005, 0.02))))
        good, n_sub, kt = _parity(f"u={u:7.2f}", runner, g, dt, thermal_solid=ts)
        ok = ok and good
        seen.add(n_sub)
        seen_k.add(kt)
    # #217 §8.6.3: at k > 1 the substep count is max(ceil(n_adv / k), F_k)
    # (F = 8, 4, 2 at k = 2, 4, 8), so the k = 8 rows carry n_sub 2 .. 8 only
    # when n_adv runs to 8 N_SUB_MAX: a fast uniform wind (synthetic, past the
    # ceiling) drives n_adv there and reaches every n_sub row at k = 8.
    for u in np.linspace(48.0, 256.0, 27):
        g.wind_x[:] = int(q(float(u)))
        g.wind_y[:] = 0
        g.temperature[:] = base_t
        g.atmosphere[:] = base_p
        good, n_sub, kt = _parity(f"u={u:7.2f}", runner, g, dt, thermal_solid=ts)
        ok = ok and good
        seen.add(n_sub)
        seen_k.add(kt)
    want = set(range(1, int(eos.N_SUB_MAX) + 1))
    want_k = {kk for kk in (1, 2, 4, 8) if kk <= int(eos.K_MAX)}
    print(f"  substep counts reached: {sorted(seen)} (cap {eos.N_SUB_MAX}); "
          f"sub-tick counts reached: {sorted(seen_k)} (K_MAX {eos.K_MAX})")
    if not want.issubset(seen) or not want_k.issubset(seen_k):
        print(f"  FAIL: the sweep did not reach every n_sub in {sorted(want)} and "
              f"every k in {sorted(want_k)} -- the gate would not exercise the "
              "fold's whole (k, n_sub) range")
        ok = False
    return ok


def part_ship_demo() -> bool:
    print("PART C -- ship_demo's match (seed 1, headless) after 60 ticks:")
    from level_loader import load as load_level
    from simulation.match_session import MatchSession
    s8a._residency(False); s8a._set_backends(False)
    ses = MatchSession(load_level("ship_demo"), seed=1, breach_physics=bp)
    for _ in range(60):
        ses.tick()
    sim = ses.sim
    dt = 1.0 / float(sim._tps)
    good, _, _ = _parity("ship_demo t=60", sim.physics_runner, sim.gmap, dt)
    return good


def main() -> int:
    print(f"device pre-stage parity (#29 P5), breach_physics HAS_CUDA="
          f"{getattr(bp, 'HAS_CUDA', False)}")
    ok = part_blast_room()
    ok = part_synthetic_range() and ok
    ok = part_ship_demo() and ok
    print("PRESTAGE_PARITY_RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
