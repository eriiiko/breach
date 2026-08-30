"""tools/quiet_room_drift.py — quiet-room T-drift capture (P-W0, arc
`tabs-compression-work`, docs/tabs_compression_work_design_2026-08-20.md §6).

Deterministic capture: a sealed 28x28 interior box (the SAME structure gate 1
of ``tests/test_air_boundary.py`` proves flat — an ambient-bounded planetside
map, 1-cell SPACE ring, boundary="ambient"), seeded with a +0.1 atm Gaussian
pressure bump at the centre, run 2000 ticks through the live C++ ambient path
(``simulation.physics_runner.PhysicsRunner`` — the same harness the
air-boundary gates use), recording per tick:

  eos_energy_books_sum     PhysicsRunner.energy_books_sum(gmap) — the P-M4b
                            binding (Sigma n_bulk*T over the accountable set,
                            raw Q16.16^2; test_destroy_wall_conserves_mass.py
                            gate 6's ``_books()`` reads the identical call).
  e_kick_ke_sum            runner.eos.e_kick_ke_sum — PER-TICK (reset
                           (arc #54 P-G1a: `eth_compression_delta` is
                           structurally 0 now — step 4c is gone and the
                           face-flux step telescopes, so that bracket can
                           no longer detect anything. The kick's KE debit
                           is the live per-tick energy channel to watch
                           in a QUIET room: it should net ~0.)
                            at every EOSSolver.step() entry; recorded raw,
                            never diffed — the storm_ledger.PER_TICK_COUNTERS
                            idiom).
  max|T_rel|                max abs raw-then-dequantized T over the open
                            interior cells (game-deg).
  mean_t_rel                SIGNED spatial mean T over the open interior
                            cells (game-deg) — the R-3 MINT guard's own
                            quantity (design §0b; test_quiet_room_drift_
                            smoke.py's ``_run_with_signed_mean`` duplicated
                            this at P-W1b because tools/ was off that
                            patch's edit surface; P-W2 folds it back into
                            the tool proper, tools/ being back on-surface).
  t_min_gas                 min T over the open interior cells (game-deg) —
                            the storm_ledger.measure_state() idiom.
  rail counters              work_clamp_hits, energy_floor_hits,
                            t_max_phys_hits, u_clamp_hits, u_max_hits — all
                            CUMULATIVE on ``runner.eos`` (recorded as their
                            running value each tick, not diffed).

This arc's whole point is the T=0 fixed point under the CURRENT (ambient-
relative) compression-work law: on HEAD, T participates in 4c only through
k*T_rel, so ambient air (T_rel=0) provably never moves under 4c and every
row above except the acoustic (P/u) fields is expected to read exact zero.
P-W2 re-runs this same tool AFTER the T_abs law lands and applies the bound
gate (max|T_rel| <= 10 game-deg / 2000 ticks, provisional) that this baseline
run cannot fail by construction.

Usage:
    conda run -n data python tools/quiet_room_drift.py [--ticks 2000] [--out drift.npz]
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

import breach_physics as bp                          # noqa: E402
import level_loader                                   # noqa: E402
from simulation.ambient import derive_ambient          # noqa: E402
from simulation.atmosphere_fixed import FP_ONE         # noqa: E402
from simulation.gamemap import GameMap                 # noqa: E402
from simulation.physics_runner import PhysicsRunner    # noqa: E402

DT_TICK = 1.0 / 24.0
H = W = 28
BUMP_ATM = 0.1
BUMP_SIGMA = 4.0            # tiles; keeps the ~3-sigma radius inside the
                             # 26x26 interior (centre-to-ring distance ~13)
DEFAULT_TICKS = 2000

COUNTER_NAMES = ("work_clamp_hits", "energy_floor_hits", "t_max_phys_hits",
                 "u_clamp_hits", "u_max_hits")


def _ambient_gmap(h, w, ambient_cfg=None):
    """TRANSCRIBED from ``tests/test_air_boundary.py:749`` (``_ambient_gmap``)
    — a planetside map: 1-cell SPACE ring border (v1 code 0) around an
    open-air interior (code 9). Hand-built LevelData, no level folder (the
    physics path only needs the tilemap + boundary + dials). Kept as a
    transcription rather than an import: the test module is not a stable
    import surface for tools/."""
    tm = np.full((h, w), 9, dtype=np.int32)
    tm[0, :] = tm[-1, :] = tm[:, 0] = tm[:, -1] = 0
    ld = level_loader.LevelData(
        name="quiet_room", version="1", path=Path("."), tilemap=tm,
        tile_size_m=1.0 / 3.0, diffuse_path=Path("."),
        boundary="ambient", ambient=ambient_cfg)
    return GameMap(ld)


def _seed_pressure_bump(gmap, interior):
    """+0.1 atm Gaussian bump at the grid centre, quantized to raw Q16.16.

    Deterministic by construction: a closed-form Gaussian over integer grid
    coordinates, no RNG anywhere. np.round (round-half-to-even) -> int64,
    added to the existing (flat, ambient-pinned) atmosphere plane. Ring
    cells are left untouched — the interior mask excludes them, and the
    ambient path re-pins the ring every tick regardless."""
    h, w = gmap.atmosphere.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    bump = BUMP_ATM * FP_ONE * np.exp(-r2 / (2.0 * BUMP_SIGMA ** 2))
    bump_q = np.round(bump).astype(np.int64)
    new_p = gmap.atmosphere.astype(np.int64)
    new_p[interior] += bump_q[interior]
    gmap.atmosphere[:] = new_p.astype(gmap.atmosphere.dtype)


def run(ticks: int = DEFAULT_TICKS) -> dict:
    """Run the quiet-room scenario; return dict(series, summary)."""
    g = _ambient_gmap(H, W, derive_ambient())
    runner = PhysicsRunner(bp)
    interior = (~g.solid) & (~g.is_ambient)
    _seed_pressure_bump(g, interior)

    cols = ("tick", "eos_energy_books_sum", "e_kick_ke_sum",
            "max_abs_t_rel", "mean_t_rel", "t_min_gas") + COUNTER_NAMES
    series = {c: [] for c in cols}

    for k in range(1, ticks + 1):
        runner.step(g, DT_TICK)
        eos = runner.eos
        t_interior = g.temperature[interior].astype(np.int64)
        series["tick"].append(k)
        series["eos_energy_books_sum"].append(int(runner.energy_books_sum(g)))
        series["e_kick_ke_sum"].append(int(eos.e_kick_ke_sum))
        series["max_abs_t_rel"].append(float(np.abs(t_interior).max()) / FP_ONE)
        series["mean_t_rel"].append(float(t_interior.mean()) / FP_ONE)
        series["t_min_gas"].append(float(t_interior.min()) / FP_ONE)
        for c in COUNTER_NAMES:
            series[c].append(int(getattr(eos, c)))

    series = {k: np.asarray(v) for k, v in series.items()}
    summary = {
        "ticks": ticks,
        "eos_energy_books_sum_start": int(series["eos_energy_books_sum"][0]),
        "eos_energy_books_sum_end": int(series["eos_energy_books_sum"][-1]),
        "e_kick_ke_sum_sum": int(series["e_kick_ke_sum"].sum()),
        "max_abs_t_rel_over_run": float(series["max_abs_t_rel"].max()),
        "mean_t_rel_final": float(series["mean_t_rel"][-1]),
        "t_min_gas_over_run": float(series["t_min_gas"].min()),
    }
    for c in COUNTER_NAMES:
        summary[c + "_final"] = int(series[c][-1])
    return {"series": series, "summary": summary}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ticks", type=int, default=DEFAULT_TICKS)
    ap.add_argument("--out", default=None, help="npz path for the full series")
    a = ap.parse_args(argv)

    out = run(ticks=a.ticks)
    s = out["summary"]

    print(f"quiet_room_drift: {H}x{W} ambient-bounded box, "
          f"+{BUMP_ATM} atm Gaussian bump (sigma={BUMP_SIGMA} tiles), "
          f"{a.ticks} ticks")
    print(f"\n  {'field':28s} {'value':>18s}")
    print(f"  {'eos_energy_books_sum start':28s} {s['eos_energy_books_sum_start']:18d}")
    print(f"  {'eos_energy_books_sum end':28s} {s['eos_energy_books_sum_end']:18d}")
    print(f"  {'e_kick_ke_sum sum':28s} {s['e_kick_ke_sum_sum']:18d}")
    print(f"  {'max|T_rel| over run (deg)':28s} {s['max_abs_t_rel_over_run']:18.6f}")
    print(f"  {'mean T_rel at run end (deg)':28s} {s['mean_t_rel_final']:18.6f}")
    print(f"  {'t_min_gas over run (deg)':28s} {s['t_min_gas_over_run']:18.6f}")
    for c in COUNTER_NAMES:
        print(f"  {c + ' (final)':28s} {s[c + '_final']:18d}")

    if a.out:
        np.savez_compressed(a.out, **{f"s_{k}": v for k, v in out["series"].items()})
        print(f"\n  series -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
