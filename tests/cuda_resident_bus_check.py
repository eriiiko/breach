"""#29 P4a — THE BUS GATE: slot 7 of a resident tick moves planes across the
bus exactly twice, at its ends.

Runs inside the GPU subprocess (tests/cuda_harness.run_cuda_script; wrapper
tests/test_cuda_resident_bus.py). The residency design's P4a row: slot 7 is
ONE batched upload at entry, a device chain with ZERO plane transfers between
its stages, and ONE batched download at exit
(docs/residency_batching_design_2026-11-07.md §9). This script runs the real
match (ship_demo seed 1, bots + horde, a headless ``MatchSession``) on the
resident path with ``tools/tick_stage_profile.py``'s timers installed, and per
measured tick reads the slot-7 call sequence the profiler records (every
stage call in order, every ``GameMap.from_host`` / ``to_host`` with its field
count and bytes):

  * exactly ONE ``from_host`` and ONE ``to_host`` per tick;
  * the ``from_host`` precedes every device stage of the tick (the water
    stage, the EOS stage, the trace tail, combustion, the tail);
  * the ``to_host`` follows the last of them (the tail, which ends with the
    one small books D2H inside C++) -- so NO GameMap transfer sits between
    the entry upload and the exit download;
  * the device stages all ran (each tick's sequence carries the EOS stage,
    the trace tail, combustion and the tail).

#29 P4b adds the stamp: the five stamp planes (obstacles, dyn_permeability,
dyn_wave_absorb, dyn_heat_atten_q, dyn_light_atten_q) are built on the device
from the host stamp's rows (``"  7 stamp (resident)"``, a stage inside the
window), so on every tick the stamp stage ran the entry ``from_host`` carries
NONE of them (and on a tick it did not run -- a topology change between the
stamp and the tick, rare -- the entry carries the planes the device did not
build). The stamp ran on most ticks (the match's walls do not change most
ticks).

It prints the two remaining calls with their fields and bytes per tick.

BREAKS IF the entry upload carries a stamp plane on a tick the device stamp
ran (the device stamp quietly became an upload again), the device stamp stops
running (the stage is missing on most ticks), a resident stage regains a host
bracket (a ``to_host`` / ``from_host``
pair around it), the entry or exit split into several calls, or a stage is
moved outside the upload / download window. (Copies made inside the C++
drivers -- the EOS's 8-byte n_sub scan result and its counter blocks, the
trace tail's books, the tail's one books arena -- are not GameMap planes and
are not counted here.)

Prints ``RESIDENT_BUS_RESULT: PASS`` / ``FAIL`` and exits 0 / 1.
"""
from __future__ import annotations

import os
import sys

import breach_physics as bp   # the CUDA build (cuda_harness bootstrap)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "tools"))

H2D = "  7 H2D GameMap.from_host"
D2H = "  7 D2H GameMap.to_host"
STAMP_STAGE = "  7 stamp (resident)"
STAMP_PLANES = ("obstacles", "dyn_permeability", "dyn_wave_absorb",
                "dyn_heat_atten_q", "dyn_light_atten_q")
DEVICE_STAGES = (
    STAMP_STAGE,
    "  7 water substeps (resident)",
    "  7 water tail (resident)",
    "  7 EOS run_substeps_resident",
    "  7 trace tail (resident)",
    "  7 combustion (resident)",
    "  7 tail (resident: fire+sweep+temp+books)",
)
REQUIRED = (
    "  7 EOS run_substeps_resident",
    "  7 trace tail (resident)",
    "  7 combustion (resident)",
    "  7 tail (resident: fire+sweep+temp+books)",
)
N_TICKS = 40


def main() -> int:
    if not getattr(bp, "HAS_CUDA", False) or not bp.cuda_available():
        print("RESIDENT_BUS_RESULT: FAIL (no CUDA build / device)")
        return 1
    from level_loader import load as load_level
    from run_on_cuda import enable_all_backends
    from simulation import physics_runner
    from simulation.match_session import MatchSession
    from tick_stage_profile import TickStageProfiler

    enable_all_backends(bp)
    physics_runner.set_residency(True)
    ses = MatchSession(load_level("ship_demo"), seed=1, breach_physics=bp)
    prof = TickStageProfiler(skip=2)
    prof.install(bp, device=True)
    try:
        for _ in range(N_TICKS):
            ses.tick()
    finally:
        prof.uninstall()
        physics_runner.set_residency(False)

    fails = []
    stamp_ticks = fallback_ticks = 0
    entry = {}
    exit_ = {}
    for t, seq in enumerate(prof.seq):
        tick = t + prof.skip + 1
        labels = [e[0] for e in seq]
        ups = [i for i, e in enumerate(seq) if e[1] == "H2D"]
        downs = [i for i, e in enumerate(seq) if e[1] == "D2H"]
        dev = [i for i, lab in enumerate(labels) if lab in DEVICE_STAGES]
        if len(ups) != 1 or len(downs) != 1:
            fails.append(f"tick {tick}: {len(ups)} from_host / {len(downs)} "
                         f"to_host calls in slot 7 (expected 1 / 1): {labels}")
            continue
        if dev and not (ups[0] < dev[0] and downs[0] > dev[-1]):
            fails.append(f"tick {tick}: a device stage sits outside the "
                         f"upload / download window: {labels}")
        missing = [r for r in REQUIRED if r not in labels]
        if missing:
            fails.append(f"tick {tick}: device stages missing {missing} "
                         f"(the resident chain did not run)")
        entry[tick] = seq[ups[0]][2:]
        exit_[tick] = seq[downs[0]][2:]
        # #29 P4b: the device stamp's planes are not on the bus.
        names = set(seq[ups[0]][4])
        carried = sorted(names & set(STAMP_PLANES))
        if STAMP_STAGE in labels:
            stamp_ticks += 1
            if carried:
                fails.append(f"tick {tick}: the device stamp ran AND the entry "
                             f"upload carried {carried}")
        else:
            fallback_ticks += 1
            if not {"dyn_permeability", "dyn_heat_atten_q"} <= names:
                fails.append(f"tick {tick}: no device stamp and the entry did "
                             f"not upload the host's stamp planes")
    if len(prof.seq) != N_TICKS - prof.skip:
        fails.append(f"{len(prof.seq)} measured ticks of {N_TICKS - prof.skip}")
    if stamp_ticks * 2 <= len(prof.seq):
        fails.append(f"the device stamp ran on only {stamp_ticks} of "
                     f"{len(prof.seq)} ticks")
    for f in fails[:10]:
        print(f"  FAIL: {f}")
    if entry:
        n = len(entry)
        fe = sum(v[0] for v in entry.values()) / n
        be = sum(v[1] for v in entry.values()) / n
        fx = sum(v[0] for v in exit_.values()) / n
        bx = sum(v[1] for v in exit_.values()) / n
        print(f"  slot 7, {n} measured ticks (ship_demo seed 1, resident, headless):")
        print(f"    entry  GameMap.from_host: 1 call/tick, {fe:.1f} fields, "
              f"{be / 1024:.1f} KiB/tick")
        print(f"    exit   GameMap.to_host:   1 call/tick, {fx:.1f} fields, "
              f"{bx / 1024:.1f} KiB/tick")
        print(f"    between them: 0 GameMap transfers on every tick")
        print(f"    device stamp on {stamp_ticks} ticks (no stamp plane uploaded), "
              f"host planes uploaded on {fallback_ticks}")
        s = prof.summary()
        for key, v in s["bus"].items():
            print(f"      {key:34s} {v['calls']:.2f} calls/tick "
                  f"{v['bytes'] / 1024:8.1f} KiB/tick")
    if fails:
        print(f"RESIDENT_BUS_RESULT: FAIL ({len(fails)} failures)")
        return 1
    print("RESIDENT_BUS_RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
