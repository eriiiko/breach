"""bench_two_room.py — the committed two-room storming bench (storm audit 2026-08-14).

ONE command that runs a fire in the committed two-room-with-door fixture
(levels/bench_two_room) for N ticks and emits the storm metrics: kinetic
energy, max wind, dominant mode period, EOS rail counters, and per-field
digests (trajectory-folded + final-state) so two runs — or two machines — can
be compared bit-for-bit.

Why it exists: every fire bench before this one was single-room, a geometry
structurally BLIND to the door-neck Helmholtz mode (the storming Erik saw;
docs/fire_atmosphere_oscillation_analysis_2026-08-03.md §4, R4 of
docs/audit_lessons_and_rules_2026-08-04.md). This closes that gap in the
committed bench set. Gate: tests/test_bench_two_room.py (fixture structure +
2-run digest determinism).

ANALYSIS ONLY: drives the shipped engine; nothing in cpp/ or src/ changes.
Dials move through fire_timing_harness.apply_overrides and are restored.

BLIND (R4): tile_size_m 0.5 (probe-battery scale, not the shipped 0.333);
sealed hull (no ambient exchange/sponge); one crate fire; no units/entities.

Provenance (R6): every JSON artifact carries git commit + dirty flag, UTC
timestamp, fixture id, and the full override dict.

Usage:
  conda run -n data python tools/bench_two_room.py                 # shipped config
  conda run -n data python tools/bench_two_room.py --pf1b          # P-F1b dials
  conda run -n data python tools/bench_two_room.py --ticks 4800 --damp 0.02 \
      --set k_wind_strip=0.5 --json out.json
"""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tools",
           ROOT / "cpp" / "build" / "Release",
           ROOT / "cpp" / "build_cuda" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp                      # noqa: E402
import level_loader                              # noqa: E402
from config import CFG                           # noqa: E402
from simulation import Simulation                # noqa: E402
from simulation import fire_fixed                # noqa: E402

import storm_probe as sp                         # noqa: E402
from fire_timing_harness import (                # noqa: E402
    FP_ONE, apply_overrides, restore_overrides,
)

FIXTURE = "bench_two_room"
# The crate the fixture carries (left-room center), in (x, y) tile coords —
# must match levels/bench_two_room/tilemap.csv and storm_probe.build_tworoom.
CRATE_XY = (7, 7)

# The synced planes the digests cover. Int32 Q16.16 state — bytes are
# machine-independent, so equal digests mean equal trajectories.
DIGEST_FIELDS = ("atmosphere", "temperature", "wind_x", "wind_y",
                 "gas", "fire", "smoke")


def load_fixture():
    """Load the committed level and assert it IS the storm-probe geometry.

    R1 (gates must be able to go red): a hand-edit to the tilemap that silently
    removed the partition or the door would make every number this bench emits
    incomparable to the storm-audit record — so the equivalence is asserted,
    not assumed.
    """
    level = level_loader.load(FIXTURE)
    ref = sp.build_tworoom(12, 12, 0.5, door_h=1, crate_xy=CRATE_XY)
    if level.tilemap.shape != ref.tilemap.shape or \
            not np.array_equal(level.tilemap, ref.tilemap):
        raise AssertionError(
            f"levels/{FIXTURE}/tilemap.csv no longer matches "
            "storm_probe.build_tworoom(12, 12, 0.5, door_h=1, crate_xy=(7, 7)) "
            "— the bench's numbers would be incomparable to the storm-audit "
            "record. Regenerate the fixture or update this gate DELIBERATELY.")
    if float(level.tile_size_m) != 0.5:
        raise AssertionError("bench_two_room tile_size_m must stay 0.5 "
                             "(the storm-probe battery scale).")
    return level


def _plane_bytes(gmap, name):
    if name == "gas":
        return np.ascontiguousarray(gmap.gas).tobytes()
    if name == "smoke":
        return np.ascontiguousarray(gmap.smoke).tobytes()
    return np.ascontiguousarray(getattr(gmap, name)).tobytes()


def run_bench(ticks=4800, damp=0.0, dials=None, digest_every=1):
    """Run the fire bench; return dict(summary, counters, digests, rows)."""
    overrides = dict(dials or {})
    restore = apply_overrides(overrides) if overrides else []
    try:
        level = load_fixture()
        sim = Simulation(level, seed=12345, breach_physics=bp,
                         enable_recorder=False)
        gmap = sim.gmap
        if damp > 0.0:
            sp.stamp_air_damping(gmap, damp)

        open_mask = ~gmap.solid
        for attr in ("is_vacuum", "is_ambient"):
            m = getattr(gmap, attr, None)
            if m is not None:
                open_mask = open_mask & (~m)

        # Ignite the crate exactly as the storm probe does.
        fx, fy = CRATE_XY
        seed_i = float(getattr(CFG.physics.fire, "ignition_seed", 0.1))
        gmap.fire[fy, fx] = fire_fixed.quantize_scalar(seed_i)
        gmap.temperature[fy, fx] = fire_fixed.quantize_scalar(280.0)

        tps = float(CFG.clock.ticks_per_second)
        h, w = gmap.solid.shape
        probe = (h // 2, w // 4)

        traj = {name: hashlib.sha256() for name in DIGEST_FIELDS}
        rows = []
        p_prev = gmap.atmosphere[open_mask].astype(np.float64) / FP_ONE
        for k in range(1, ticks + 1):
            sim.set_paused(False)
            sim.step()
            m = sp.measure(gmap, open_mask, p_prev, probe)
            p_prev = gmap.atmosphere[open_mask].astype(np.float64) / FP_ONE
            m["t"] = k / tps
            m["tick"] = k
            m["I"] = int(gmap.fire[fy, fx]) / FP_ONE
            rows.append(m)
            if k % digest_every == 0:
                for name in DIGEST_FIELDS:
                    traj[name].update(_plane_bytes(gmap, name))

        digests = {
            "trajectory": {n: hsh.hexdigest() for n, hsh in traj.items()},
            "final": {n: hashlib.sha256(_plane_bytes(gmap, n)).hexdigest()
                      for n in DIGEST_FIELDS},
        }
        res = dict(rows=rows, counters=sp.eos_counters(sim), dt=1.0 / tps,
                   probe=probe, n_open=int(open_mask.sum()), mode="fire",
                   geom=FIXTURE, damp=damp, dT=0.0, tile=0.5, interior=12)
        return dict(res=res, summary=sp.analyse(res), digests=digests,
                    counters=res["counters"])
    finally:
        restore_overrides(restore)


def provenance(overrides, ticks, damp):
    def _git(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, text=True,
                                  capture_output=True, timeout=30).stdout.strip()
        except Exception:
            return "unavailable"
    return {
        "fixture": FIXTURE,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "utc": datetime.now(timezone.utc).isoformat(),
        "ticks": ticks, "damp": damp, "overrides": dict(overrides),
        "blind": "tile 0.5 m (not shipped 0.333); sealed hull; single crate; "
                 "no units/entities",
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ticks", type=int, default=4800, help="24 tps; 4800 = 200 s")
    ap.add_argument("--damp", type=float, default=0.0,
                    help="air wave_absorb stamped on the mirrored planes")
    ap.add_argument("--pf1b", action="store_true", help="apply the P-F1b dials")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                    help="config override via apply_overrides (repeatable)")
    ap.add_argument("--json", default=None, help="write full artifact here")
    ap.add_argument("--csv", default=None, help="write per-tick rows here")
    a = ap.parse_args(argv)

    dials = dict(sp.PF1B) if a.pf1b else {}
    for item in a.set:
        k, _, v = item.partition("=")
        dials[k] = v

    out = run_bench(ticks=a.ticks, damp=a.damp, dials=dials)
    summ, digs, cnt = out["summary"], out["digests"], out["counters"]

    print(f"bench_two_room ticks={a.ticks} damp={a.damp} "
          f"overrides={sorted(dials)}")
    for k in ("ke_peak", "ke_final", "ke_last_decile_mean", "ke_retention",
              "umax_peak", "umax_final", "dom_period_s", "ke_decay_rate_per_s"):
        if summ.get(k) is not None:
            print(f"  {k:24s} {summ[k]:.6g}")
    print(f"  counters {cnt}")
    for name in DIGEST_FIELDS:
        print(f"  digest {name:12s} traj={digs['trajectory'][name][:16]} "
              f"final={digs['final'][name][:16]}")

    if a.json:
        art = {"provenance": provenance(dials, a.ticks, a.damp),
               "summary": summ, "counters": cnt, "digests": digs}
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(art, f, indent=2)
    if a.csv:
        rows = out["res"]["rows"]
        keys = list(rows[0].keys())
        with open(a.csv, "w", encoding="utf-8") as f:
            f.write(",".join(keys) + "\n")
            for r in rows:
                f.write(",".join(f"{r[k]:.8g}" for k in keys) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
