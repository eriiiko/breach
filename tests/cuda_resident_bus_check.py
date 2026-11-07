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

It prints the two remaining calls with their fields and bytes per tick.

BREAKS IF a resident stage regains a host bracket (a ``to_host`` / ``from_host``
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
DEVICE_STAGES = (
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
    if len(prof.seq) != N_TICKS - prof.skip:
        fails.append(f"{len(prof.seq)} measured ticks of {N_TICKS - prof.skip}")
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
