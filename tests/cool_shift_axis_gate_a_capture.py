"""Gate (a) capture — WHOLE-ENGINE byte-identity across the cool-shift axis.

The cool-shift axis (2026-07-30) turns the single global ``[physics.thermal]
COOL_SHIFT`` into a per-material column projected to ``GameMap.cool_shift``,
and derives the vacuum-exposed shift from it by the global offset
``COOL_SHIFT - COOL_SHIFT_VACUUM`` (floored at ``SHIFT_MIN``). **Every material
is seeded at the old global 5**, so the claim this script gates is UNCONDITIONAL
and stronger than the thermal-mass arc's furniture-free gate:

    the whole engine must be byte-identical to the pre-patch build on EVERY
    map, furniture and vacuum breaches included, tolerance ZERO.

This is a BEFORE/AFTER artifact, not a self-checking gate (deliberately no
``test_`` prefix — pytest does not collect it, and pinning digests here would
manufacture a red the moment Erik's joint fire re-tune moves a dial, which gate
(b) of this arc forbids). It is VERSION-AGNOSTIC on purpose: it never touches
``gmap.cool_shift`` or any other new attribute, so the identical file runs
against a pristine PRE-patch tree and the patched tree.

Usage (from a repo root that has cpp/build/Release built):

    python tests/cool_shift_axis_gate_a_capture.py <outdir>

writes ``<outdir>/<scenario>.npz`` holding every captured field for every tick
(so the comparison is genuinely per-tick per-CELL, not just a hash), and prints
one canonical trajectory digest per scenario. Run it in the pristine tree and in
the patched tree with different outdirs, then diff the printed digests and
compare the npz files cell by cell.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp   # noqa: E402

FP_ONE = 65536
N_TICKS = 30


# ---------------------------------------------------------------------------
# Scenarios. Between them they must cover, at minimum:
#   * a FURNITURE-BURN map (furniture is the material whose ONLY loss channel
#     is the dial this patch made per-material), and
#   * a VACUUM / BREACH map (the patch rewrote the vacuum-exposed branch from
#     "use the other global" to "subtract the global offset, clamped").
# ---------------------------------------------------------------------------
def _sim_default():
    """The canonical A/B scenario (field_ab_harness.default_scenario_sim): a
    16x16 hull room with fire + smoke + water + a wave pulse + a hull breach to
    vacuum. Every solver live."""
    from field_ab_harness import default_scenario_sim
    return default_scenario_sim()


def _sim_ambient():
    """A planetside AMBIENT map (sky ring) with a wood fuel block and a fire
    seed — the ambient-ring wipe + the interior (non-exposed) cooling branch."""
    from level_loader import LevelData
    from simulation import Simulation, fire_fixed, gas_fixed
    from simulation.unit import Unit

    H = W = 24
    tm = np.full((H, W), 9, dtype=np.int32)            # SPACE -> ambient ring
    tm[1:H - 1, 1:W - 1] = 0                           # v2 code 0 == MAT_AIR
    tm[4:9, 4:9] = 2                                   # wood block (thermal solid)
    tm[12, 4:16] = 1                                   # hull wall stub
    level = LevelData(name="cs_gate_a_ambient", version="2", path=Path("."),
                      tilemap=tm, tile_size_m=1.0 / 3.0,
                      diffuse_path=Path("."), boundary="ambient")
    sim = Simulation(level, seed=20260730, breach_physics=bp,
                     enable_recorder=False)
    g = sim.gmap
    assert g.is_ambient.any(), "ambient routing expected"
    g.fire[10, 6] = fire_fixed.quantize_scalar(0.9)
    g.fire[10, 7] = fire_fixed.quantize_scalar(0.6)
    g.smoke[14:20, 6:18] = gas_fixed.quantize_scalar(0.4)
    g.temperature[5, 5] = 400 * FP_ONE                 # warm wood tile
    sim.add_unit(Unit("M1", x=16, y=16, team=0))
    sim.set_paused(False)
    return sim


def _sim_space_breach():
    """A space map with a wood fuel wall, a glass pane and a hull breach to
    vacuum — THE VACUUM-PATH scenario: hull tiles all round the breach take the
    space-exposed cooling branch every tick."""
    from level_loader import LevelData
    from simulation import Simulation, fire_fixed
    from simulation.unit import Unit

    H = W = 20
    tm = np.full((H, W), 9, dtype=np.int32)            # outer space
    tm[2:H - 2, 2:W - 2] = 1                           # hull shell
    tm[3:H - 3, 3:W - 3] = 0                           # interior air
    tm[6:11, 8] = 2                                    # wood partition (fuel)
    tm[13, 5:10] = 5                                   # glass pane
    level = LevelData(name="cs_gate_a_space", version="2", path=Path("."),
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


def _sim_furniture_burn():
    """THE FURNITURE-BURN scenario. A hull-shelled room with a breach to vacuum,
    a block of FURNITURE (conductivity 0 -> the ambient decay this patch made
    per-material is its ONE loss channel), a fire seeded beside it AND on it, a
    wood partition, a water pool and a trace cloud. Some crate tiles sit next to
    the breach, so the crate exercises the VACUUM-EXPOSED cooling branch too."""
    from level_loader import LevelData
    from simulation import (Simulation, atmosphere_fixed, fire_fixed,
                            water_fixed)
    from simulation.gases import O2
    from simulation.unit import Unit

    H = W = 36
    tm = np.full((H, W), 9, dtype=np.int32)            # outer space band
    tm[2:H - 2, 2:W - 2] = 1                           # hull shell
    tm[3:H - 3, 3:W - 3] = 0                           # interior air
    tm[9:14, 9:14] = 6                                 # MAT_FURNITURE crate block
    tm[24:28, W - 5:W - 3] = 6                         # crates AT the east wall
    tm[20, 6:24] = 2                                   # wood partition (fuel)
    tm[H // 2 - 2:H // 2 + 2, W - 3] = 0               # breach the east hull
    level = LevelData(name="cs_gate_a_furniture", version="2", path=Path("."),
                      tilemap=tm, tile_size_m=1.0 / 3.0, diffuse_path=Path("."))
    sim = Simulation(level, seed=20260730, breach_physics=bp,
                     enable_recorder=False)
    g = sim.gmap
    q = atmosphere_fixed.quantize_scalar
    g.temperature[6:12, 6:12] += q(3000.0)
    g.temperature[9:14, 9:14] += q(500.0)              # pre-warm the crate
    g.temperature[24:28, W - 5:W - 3] += q(800.0)      # warm the breach-side crates
    g.gas[O2, 7:11, 7:11] += q(3.0)
    g.fire[10, 8] = fire_fixed.quantize_scalar(0.9)
    g.fire[11, 11] = fire_fixed.quantize_scalar(0.8)   # fire ON a crate tile
    g.water_depth[H - 8:H - 5, 6:W // 2] = water_fixed.quantize_scalar(0.4)
    sim.add_unit(Unit("M1", x=6, y=28, team=0))
    sim.set_paused(False)
    return sim


def _sim_sealed_room():
    """A fully sealed hull room, no breach — the conservation/sealed-room shape:
    every thermal solid takes the INTERIOR cooling branch for all 30 ticks."""
    from level_loader import LevelData
    from simulation import Simulation, atmosphere_fixed, fire_fixed
    from simulation.unit import Unit

    H = W = 18
    tm = np.full((H, W), 1, dtype=np.int32)            # all hull
    tm[2:H - 2, 2:W - 2] = 0                           # interior air
    tm[8, 4:14] = 2                                    # wood shelf
    tm[5:8, 12:15] = 6                                 # a crate stack
    level = LevelData(name="cs_gate_a_sealed", version="2", path=Path("."),
                      tilemap=tm, tile_size_m=1.0 / 3.0, diffuse_path=Path("."))
    sim = Simulation(level, seed=20260730, breach_physics=bp,
                     enable_recorder=False)
    g = sim.gmap
    assert not g.is_vacuum.any(), "sealed scenario must have no vacuum"
    q = atmosphere_fixed.quantize_scalar
    g.fire[7, 6] = fire_fixed.quantize_scalar(0.85)
    g.temperature[8, 5] += q(900.0)
    g.temperature[5:8, 12:15] += q(600.0)
    sim.add_unit(Unit("M1", x=6, y=12, team=0))
    sim.set_paused(False)
    return sim


_SCENARIOS = (
    ("default_16x16_space", _sim_default),
    ("ambient_24x24_sky", _sim_ambient),
    ("space_20x20_breach", _sim_space_breach),
    ("furniture_burn_36x36", _sim_furniture_burn),
    ("sealed_18x18_room", _sim_sealed_room),
)


def main(outdir: str) -> int:
    from field_ab_harness import SIM_FIELDS, capture_trajectory
    from field_digest import trajectory_digest

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"HAS_CUDA={getattr(bp, 'HAS_CUDA', False)}  outdir={out}")

    ok = True
    for name, make_sim in _SCENARIOS:
        probe = make_sim()
        g = probe.gmap
        n_ts = int(np.count_nonzero(g.thermal_solid))
        n_furn = int(np.count_nonzero(g.thermal_solid & ~g.solid))
        n_vac = int(np.count_nonzero(g.is_vacuum))
        if n_ts == 0:
            ok = False
            print(f"  {name}: no thermal solids at all — vacuous")
            continue

        traj = capture_trajectory(make_sim=make_sim, n_steps=N_TICKS)
        peak = max(int(np.abs(s["temperature"]).max()) for s in traj)
        dig = trajectory_digest(traj)

        # Save EVERY field of EVERY tick so the A/B is per-cell, not per-hash.
        payload = {}
        for t, snap in enumerate(traj):
            for f in SIM_FIELDS:
                if f in snap:
                    payload[f"t{t:03d}__{f}"] = snap[f]
        np.savez_compressed(out / f"{name}.npz", **payload)

        print(f"  GATE_A_DIGEST {name} = {dig}   "
              f"(thermal solids {n_ts}, furniture {n_furn}, vacuum {n_vac}, "
              f"peak |T| {peak} counts, arrays {len(payload)})")
        if peak == 0:
            ok = False
            print(f"  {name}: temperature never moved — vacuous")

    print("GATE_A_CAPTURE: " + ("OK" if ok else "VACUOUS"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "_gate_a_out"))
