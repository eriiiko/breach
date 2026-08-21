"""tools/tabs_pw2_venting_capture.py — deterministic venting-bench recorder
capture for the T_abs compression-work arc (P-W2, design
docs/tabs_compression_work_design_2026-08-20.md §3 B-F7 / D-1's mach-census
data need: "P-W2 measures min P / max |grad P| / u_clamp_hits before/after
on the venting bench").

Scenario choice (documented, per the P-W2 brief's instruction to pick and
justify one): TRANSCRIBES ``tests/cuda_kick_check.py``'s PART-2 "blast +
venting" scenario verbatim (48x48, hull ring, a 4-tile breach carved through
the east hull open to an outer vacuum band, a 5000 K hot core + O2
overpressure pocket, PLUS a near-ceiling 15500 K pocket so T_MAX_PHYS /
U_MAX rails are reachable). Considered and rejected:

  * cuda_s8a_check's ring-adjacent breach world — single-tick / short-horizon
    residency check, not built for a sustained multi-hundred-tick vent.
  * a fresh headless scripted destroy_wall on a sealed room — would need new
    geometry authored from scratch; the kick-check scenario already carries
    a permanent (not destructible-transient) breach straight to vacuum, so
    the vent never re-seals and runs the LONGEST of the three candidates.

This tool re-runs that exact geometry/seed through the live engine (not the
kick-check gate's CPU/CUDA differential machinery — this is analysis only),
recording snapshots via ``simulation.recorder.PhysicsRecorder`` every tick so
the dump is consumable by ``tools/analyze_blowup_dump.py --mach-census``, and
prints the eos/temperature rail counters (u_clamp_hits, u_max_hits,
work_clamp_hits, energy_floor_hits, t_max_phys_hits, e_vac_wipe_sum,
e_ring_pin_sum) needed to price the vac/ring creation channel (design §3:
"bounded <= 290*N_vented per tick").

READ-ONLY measurement tool: no cpp/, no sim-path changes; drives the shipped
engine exactly like tools/storm_ledger.py / tools/bench_two_room.py.

Usage:
    conda run -n data python tools/tabs_pw2_venting_capture.py [--ticks 600] [--out debug_manual_....npz]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tools",
           ROOT / "cpp" / "build" / "Release",
           ROOT / "cpp" / "build_cuda" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp                            # noqa: E402
from config import CFG                                  # noqa: E402
from level_loader import LevelData                      # noqa: E402
from simulation import atmosphere_fixed                 # noqa: E402
from simulation.gamemap import GameMap                  # noqa: E402
from simulation.gases import O2                         # noqa: E402
from simulation.physics_runner import PhysicsRunner     # noqa: E402
from simulation.recorder import PhysicsRecorder         # noqa: E402

H = W = 48
DEFAULT_TICKS = 600

COUNTER_SPECS = (
    ("eos", "u_clamp_hits"), ("eos", "u_max_hits"),
    ("eos", "work_clamp_hits"), ("eos", "energy_floor_hits"),
    ("eos", "t_max_phys_hits"),
    ("temperature", "e_vac_wipe_sum"), ("temperature", "e_ring_pin_sum"),
)


def build_scenario():
    """TRANSCRIBED verbatim from tests/cuda_kick_check.py::part2_trajectory
    (the blast + venting scenario) — kept as a transcription rather than an
    import: that test module is not a stable import surface for tools/."""
    tm = np.zeros((H, W), dtype=np.int32)
    tm[2:46, 2:46] = 1
    tm[3:45, 3:45] = 4
    tm[22:26, 45] = 4          # the breach: hull ring opened to the vacuum band
    level = LevelData(name="eos_p64_blast_vent", version="1", path=Path("."),
                      tilemap=tm, tile_size_m=1.0 / 3.0,
                      diffuse_path=Path("."))
    g = GameMap(level)
    g.stamp_units([])
    assert g.is_vacuum.any(), "scenario must have vacuum to vent into"

    q = atmosphere_fixed.quantize_scalar
    g.temperature[10:16, 10:16] += q(5000.0)
    g.gas[O2, 11:14, 11:14] += q(4.0)
    g.temperature[30:36, 30:36] += q(15500.0)
    return g


def run(ticks: int = DEFAULT_TICKS, out: str | None = None) -> dict:
    g = build_scenario()
    runner = PhysicsRunner(bp)
    runner.eos.dx = float(g.tile_size_m)
    dt = 1.0 / float(CFG.clock.ticks_per_second)

    rec = PhysicsRecorder(H, W, capacity=ticks + 1)  # DEFAULT_FIELDS: mach-census-ready
    rec.record(g, 0, 0.0, [])
    for k in range(1, ticks + 1):
        runner.step(g, dt)
        rec.record(g, k, k * dt, [])

    counters = {f"{h}.{n}": int(getattr(getattr(runner, h), n))
                for h, n in COUNTER_SPECS}
    dump_path = rec.dump("manual") if out is None else None
    if out is not None:
        # dump() always timestamps its own filename; write to the requested
        # path explicitly by re-running the same packing dump() does.
        n = min(rec.count, rec.capacity)
        data = {name: rec.buffers[name][:n] for name in rec.fields}
        data["tick_ids"] = rec.tick_ids[:n]
        data["tick_times"] = rec.tick_times[:n]
        np.savez_compressed(out, **data)
        dump_path = out
    return {"counters": counters, "dump_path": dump_path, "ticks": ticks,
            "n_open_vacuum": int(g.is_vacuum.sum())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ticks", type=int, default=DEFAULT_TICKS)
    ap.add_argument("--out", default=None, help="npz path (default: recorder's own timestamped debug_manual_*.npz)")
    a = ap.parse_args(argv)

    out = run(ticks=a.ticks, out=a.out)
    print(f"tabs_pw2_venting_capture: {H}x{W} blast+vent scenario "
          f"(cuda_kick_check PART-2 geometry), {a.ticks} ticks, "
          f"vacuum cells={out['n_open_vacuum']}")
    print(f"  dump -> {out['dump_path']}")
    for k, v in out["counters"].items():
        print(f"  {k:28s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
