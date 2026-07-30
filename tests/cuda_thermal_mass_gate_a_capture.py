"""Gate (a) capture — FURNITURE-FREE byte-identity across the P2 CUDA patch.

Runs in the GPU subprocess (cuda_harness bootstrap). Drives full engine
trajectories with the **GPU temperature backend ON** over furniture-free
scenarios and prints one canonical per-scenario trajectory digest each. The
thermal-mass axis's gate (a) says: wherever ``thermal_solid == solid`` (i.e. any
map with no furniture) every path stays byte-identical — so the digests printed
by the PRE-P2 build and the POST-P2 build must match exactly, tolerance zero.

This is a BEFORE/AFTER artifact, not a self-checking gate: run it against the
pre-patch ``cpp/build_cuda`` .pyd, keep the output, rebuild, run it again, diff.
Each scenario asserts its own furniture-free precondition (thermal_solid ==
solid elementwise) so the gate cannot silently become vacuous, and asserts the
CUDA temperature backend really is engaged.

Deliberately NOT a pinned golden (no ``test_`` prefix, so pytest does not
collect it): pinning these digests would create a NEW red the moment Erik's
joint fire re-tune moves a dial, and gate (b) of this arc forbids touching
goldens here. The recorded before/after pair lives in the P2 commit message.

Usage (from the repo root, inside the cuda_harness subprocess bootstrap):
    import cuda_thermal_mass_gate_a_capture as c; c.main()
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import breach_physics as bp   # the CUDA build (sys.path[0] == cpp/build_cuda)

FP_ONE = 65536
N_TICKS = 30


def _sim_default():
    """The canonical A/B scenario (field_ab_harness.default_scenario_sim): a
    16x16 hull room with fire + smoke + water + a wave pulse + a hull breach to
    vacuum. All-hull/air -> furniture-free."""
    from field_ab_harness import default_scenario_sim
    return default_scenario_sim()


def _sim_ambient():
    """A planetside AMBIENT map (sky ring) with a wood-wall fuel block and a
    fire seed — exercises the Pass-0a ambient wipe + the wood thermal solid."""
    from level_loader import LevelData
    from simulation import Simulation, fire_fixed, gas_fixed
    from simulation.unit import Unit

    H = W = 24
    tm = np.full((H, W), 9, dtype=np.int32)            # SPACE -> ambient ring/interior
    tm[1:H - 1, 1:W - 1] = 0                           # v2 code 0 == MAT_AIR
    tm[4:9, 4:9] = 2                                   # a wood block (thermal solid)
    tm[12, 4:16] = 1                                   # a hull wall stub
    level = LevelData(name="tm_gate_a_ambient", version="2", path=Path("."),
                      tilemap=tm, tile_size_m=1.0 / 3.0,
                      diffuse_path=Path("."), boundary="ambient")
    sim = Simulation(level, seed=20260730, breach_physics=bp,
                     enable_recorder=False)
    g = sim.gmap
    assert g.is_ambient.any(), "ambient routing expected"
    g.fire[10, 6] = fire_fixed.quantize_scalar(0.9)
    g.fire[10, 7] = fire_fixed.quantize_scalar(0.6)
    g.smoke[14:20, 6:18] = gas_fixed.quantize_scalar(0.4)
    g.temperature[5, 5] = 400 * FP_ONE                 # a warm wood tile
    sim.add_unit(Unit("M1", x=16, y=16, team=0))
    sim.set_paused(False)
    return sim


def _sim_space_breach():
    """A space map with a wood fuel wall, a glass pane (thermal_mass 16), and a
    hull breach to vacuum — covers the Pass-0a vacuum wipe + the space-exposed
    hull guard + the per-tile shift spread."""
    from level_loader import LevelData
    from simulation import Simulation, fire_fixed
    from simulation.unit import Unit

    H = W = 20
    tm = np.full((H, W), 9, dtype=np.int32)            # outer space
    tm[2:H - 2, 2:W - 2] = 1                           # hull shell
    tm[3:H - 3, 3:W - 3] = 0                           # interior air
    tm[6:11, 8] = 2                                    # wood partition (fuel)
    tm[13, 5:10] = 5                                   # glass pane
    level = LevelData(name="tm_gate_a_space", version="2", path=Path("."),
                      tilemap=tm, tile_size_m=1.0 / 3.0, diffuse_path=Path("."))
    sim = Simulation(level, seed=20260730, breach_physics=bp,
                     enable_recorder=False)
    g = sim.gmap
    g.fire[7, 7] = fire_fixed.quantize_scalar(0.9)
    g.fire[8, 7] = fire_fixed.quantize_scalar(0.7)
    g.temperature[13, 6] = 300 * FP_ONE                # warm glass
    sim.add_unit(Unit("M1", x=6, y=15, team=0))
    g.destroy_wall(2, 10)                              # breach the shell to vacuum
    sim.set_paused(False)
    return sim


_SCENARIOS = (
    ("default_16x16_space", _sim_default),
    ("ambient_24x24_sky", _sim_ambient),
    ("space_20x20_breach", _sim_space_breach),
)


def main() -> int:
    from field_ab_harness import capture_trajectory
    from field_digest import trajectory_digest

    if not getattr(bp, "HAS_CUDA", False) or not bp.cuda_available():
        print("GATE_A_RESULT: FAIL (no CUDA build / device)")
        return 1
    print("device:", bp.cuda_device_info())

    ok = True
    for name, make_sim in _SCENARIOS:
        # Furniture-free precondition (gate (a)'s own hypothesis, addendum D4).
        probe = make_sim()
        g = probe.gmap
        if not np.array_equal(g.thermal_solid, g.solid):
            ok = False
            print(f"  {name}: NOT furniture-free (thermal_solid != solid) — "
                  f"scenario invalid for gate (a)")
            continue
        n_ts = int(np.count_nonzero(g.thermal_solid))
        if n_ts == 0:
            ok = False
            print(f"  {name}: no thermal solids at all — vacuous")
            continue

        bp.set_temperature_backend(True)
        assert bp.get_temperature_backend(), "temperature backend flag did not take"
        traj = capture_trajectory(make_sim=make_sim, n_steps=N_TICKS)
        bp.set_temperature_backend(False)

        peak = max(int(np.abs(s["temperature"]).max()) for s in traj)
        dig = trajectory_digest(traj)
        print(f"  GATE_A_DIGEST {name} = {dig}   "
              f"(thermal solids {n_ts}, peak |T| {peak} counts)")
        if peak == 0:
            ok = False
            print(f"  {name}: temperature never moved — vacuous")

    print("GATE_A_RESULT: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
