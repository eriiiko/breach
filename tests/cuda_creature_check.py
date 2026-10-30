"""#106 P6d1 -- a limp octopus is backend-blind (in the CUDA subprocess).

A flooded room (written through level_lib, like tests/test_creature_body.py)
with one limp octopus, run 48 ticks THREE ways in lockstep in this one
process (the CUDA build is the cached ``breach_physics``):

  cpu       every GPU backend OFF, residency OFF (the CPU solvers)
  cuda      every GPU backend ON, residency OFF   (== ``main.py --cuda``)
  resident  every GPU backend ON, residency ON    (== ``--resident``)

Per tick it asserts, at tol 0, that the FIELD digest (the frozen integer
planes), the full TICK digest (fields + units + the ``__creature__`` section)
and the ``__creature__`` section bytes are identical across the three. The
arm chain itself runs on the CPU in every mode (no CUDA twin yet, recorded
debt), so what this proves is that the creature's inputs (the map, the other
units) and its outputs (the stamp, the pose) cross the GPU paths untouched.

Vacuity guards: the GPU EOS dispatch fired every tick on the cuda run, the
resident world entered residency mode, and the octopus actually moved (its
section changed over the run).

Prints ``CREATURE_RESULT: PASS`` / ``FAIL`` (or ``SKIP <reason>`` when the
CUDA build predates the arm chain solver) and exits 0 / 1.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

import breach_physics as bp   # the CUDA build (cuda_harness put it first)

N_TICKS = 48

_BACKENDS = (
    "set_temperature_backend", "set_water_backend",
    "set_fire_backend", "set_radiation_backend",
    "set_bulk_flux_backend", "set_sl_advection_backend",
    "set_mg_solve_backend", "set_kick_compression_backend",
    "set_combustion_backend",
)


def _mode(backends: bool, resident: bool) -> None:
    from simulation import physics_runner
    for name in _BACKENDS:
        getattr(bp, name)(bool(backends))
    physics_runner.set_residency(bool(resident))


def _room():
    import level_lib
    import level_loader
    from PIL import Image
    from simulation import water_fixed
    from simulation.match import CreatureSpawn
    from simulation.materials import MAT_AIR, MAT_HULL
    root = Path(tempfile.mkdtemp(prefix="cuda_creature_"))
    lvl = root / "flood"
    lvl.mkdir()
    w = h = 30
    level_lib.write_level_header(lvl, name="flood", tile_size_m=1.0 / 3.0)
    tm = np.full((h, w), MAT_HULL, dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = MAT_AIR
    level_lib.write_tilemap_csv(lvl, tm, csv_bak=False)
    Image.new("RGB", (w, h)).save(lvl / "diffuse.png")
    depth = np.where(tm == MAT_AIR, water_fixed.quantize_scalar(2.0), 0).astype(np.int32)
    level_lib.write_water(lvl, depth)
    level_lib.write_creatures(lvl / "level.toml",
                              [CreatureSpawn("octopus", 13.5, 13.5, heading=30.0)])
    return level_loader.load("flood", str(root))


def main() -> int:
    if not hasattr(bp, "arm_chain_tick"):
        print("CREATURE_RESULT: SKIP the CUDA build predates the arm chain solver "
              "(#106 P6c): rebuild cpp/build_cuda")
        return 0
    if not getattr(bp, "HAS_CUDA", False) or not bp.cuda_available():
        print("CREATURE_RESULT: FAIL (no CUDA build / device)")
        return 1
    from field_ab_harness import capture_snapshot
    from field_digest import field_digest, tick_digest
    from simulation import Simulation, arm_chain as AC, creature_fixed as CF
    from simulation.creature_system import CREATURE_DIGEST_KEY, section_bytes
    from simulation.ruleset import ContinuousRealtime

    level = _room()
    modes = {"cpu": (False, False), "cuda": (True, False), "resident": (True, True)}
    sims = {}
    for name in modes:
        s = Simulation(level, seed=7, breach_physics=bp, enable_recorder=False,
                       ruleset=ContinuousRealtime())
        (u,) = s.spawn_creatures(level.creatures)
        # released 0.2 m above the deck, so the arms fall (the limp gate's set-up)
        b = s._creature_system.bodies[u.id]
        lift = CF.chain_from_m(0.2)
        b.bufs.p[0, ..., 2] += lift
        b.bufs.body[0, 0, AC.BS_Z] += lift
        sims[name] = s

    calls0 = bp.eos_step_cuda_calls()
    res0 = bp.eos_resident_calls()
    first_sect = None
    bad = 0
    for t in range(N_TICKS):
        out = {}
        for name, (backends, resident) in modes.items():
            _mode(backends, resident)
            sims[name].set_paused(False)
            sims[name].step()
            _mode(False, False)
            snap = capture_snapshot(sims[name])
            out[name] = (field_digest(snap), tick_digest(snap),
                         section_bytes(snap[CREATURE_DIGEST_KEY]))
        if first_sect is None:
            first_sect = out["cpu"][2]
        for name in ("cuda", "resident"):
            for k, label in enumerate(("field digest", "tick digest", "__creature__")):
                if out[name][k] != out["cpu"][k]:
                    bad += 1
                    print(f"  tick {t}: {label} {name} != cpu")
        if bad >= 10:
            print("  aborting after 10 divergences")
            break

    ok = bad == 0
    fired = bp.eos_step_cuda_calls() - calls0
    if fired < N_TICKS:
        ok = False
        print(f"  the GPU EOS dispatch fired {fired}/{N_TICKS} ticks (a silently-CPU run "
              f"cannot pass)")
    res = bp.eos_resident_calls() - res0
    if res < N_TICKS:
        ok = False
        print(f"  the resident EOS fired {res}/{N_TICKS} ticks -- VACUOUS")
    if not bool(sims["resident"].gmap.residency_on()):
        ok = False
        print("  the resident world never entered residency mode -- VACUOUS")
    if out["cpu"][2] == first_sect:
        ok = False
        print("  the octopus never moved -- VACUOUS")
    print(f"  {N_TICKS} ticks x cpu / cuda / resident: {'identical' if bad == 0 else f'{bad} divergence(s)'}"
          f"; GPU EOS calls {fired}, resident EOS calls {res}")
    print(f"CREATURE_RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
