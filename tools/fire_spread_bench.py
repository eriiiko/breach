"""P-F1b — THE RADIATIVE SPREAD BENCH (one air gap, still-air reference arena).

docs/fire_realism_design_2026-08-01.md v4 ruling 3 splits the spread channels:
CONDUCTION leads (30-60 s per hop, crate-to-crate through a shared face) and
RADIATION is the flashover channel. Crate conductivity is still 0 on this
branch, so the ONLY spread channel that exists today is the radiative one, and
under v7 rule 3 a face-touching pair is radiatively INERT by construction (a ray
entering a solid-solid contact face terminates with no deposit and no charge —
contact is conduction's domain). The measurable geometry is therefore
ONE AIR GAP: emitter tile, one air tile, receiver tile.

WHAT IT MEASURES. An ESTABLISHED burner (pinned at a given intensity and tile
temperature — "a burning pile", not one still ramping) and a cold receiver of
the same material, two tiles apart on the still-air reference arena
(`fire_timing_harness.build_level`, planetside, sky refill on). Reports the
receiver's warming curve, its ceiling, and the second at which it crosses its
own `ignition_temp` and takes a seed (gmap.fire > 0).

THE PHYSICS THIS BENCH EXISTS TO WATCH (P-F1b's derivation, config.toml
[materials.kindling] cool_shift):
  * a one-gap receiver is crossed by ~1 of the emitter's 8 fan rays, so it
    absorbs a_s*a_r*w*(E[T_s] - E[T_r]) per tick;
  * below `T_emit_gate` it does not CAST, so its only loss is cool_shift and its
    ceiling is where absorption == T/2^cool_shift;
  * above the gate it starts paying its own sky in the ~7 directions that leave
    the world, and its ceiling collapses to E_r ~ E_s/15 -- far under any
    ignition_temp. That is the GATE WALL P-F1a measured (a plank stalling at
    183.7 game against the old gate of 180), and why the gate now sits above
    every flammable ignition_temp.

RUN:
    python tools/fire_spread_bench.py                 # kindling + furniture
    python tools/fire_spread_bench.py --material kindling --src-T 310
    python tools/fire_spread_bench.py --natural       # source ignites naturally

Deterministic: fixed seed, no RNG in the driven path. Headless.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src",
           ROOT / "cpp" / "build" / "Release",
           ROOT / "cpp" / "build_cuda" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp                       # noqa: E402
from config import CFG                            # noqa: E402
from simulation import Simulation                 # noqa: E402
from simulation import fire_fixed                 # noqa: E402

from fire_timing_harness import (                 # noqa: E402
    FP_ONE, FURN, KIND, build_level, apply_overrides, restore_overrides,
)

ARTIFACTS_DIR = ROOT / "_fire_tuning_artifacts"
NAMES = {"kindling": KIND, "furniture": FURN}


def run_spread(material, *, src_I=None, src_T=None, gap=1, natural=False,
               interior_w=40, interior_h=24, src_xy=(12, 12), tile_size_m=0.5,
               max_seconds=300.0, overrides=None, seed=12345, verbose=True):
    """One emitter/receiver pair, ``gap`` AIR tiles apart, same material.

    ``natural=False`` (default) pins the emitter at (``src_I``, ``src_T``) every
    tick — the "established burner" the spread ruling talks about, and the only
    way to time the RECEIVER's clock without the emitter's own ramp in it.
    ``natural=True`` ignites the emitter with ``ignition_seed`` and lets it run,
    which measures the whole chain (ignition -> ramp -> spread) end to end.
    """
    mat_id = NAMES[material] if isinstance(material, str) else int(material)
    restore = apply_overrides(overrides or {})
    try:
        sx, sy = src_xy
        rx = sx + gap + 1
        level = build_level(interior_w, interior_h, src_xy, tile_size_m,
                            sky_tau_s=60.0, sponge_width=8, material=mat_id)
        level.tilemap[sy, rx] = mat_id
        sim = Simulation(level, seed=seed, breach_physics=bp, enable_recorder=False)
        g = sim.gmap
        ign = float(g.materials.ignition_temp[mat_id])
        seed_I = float(getattr(CFG.physics.fire, "ignition_seed", 0.1))
        g.fire[sy, sx] = fire_fixed.quantize_scalar(src_I if (src_I and not natural)
                                                    else seed_I)
        g.temperature[sy, sx] = fire_fixed.quantize_scalar(
            src_T if (src_T and not natural) else ign)
        hp0 = int(g.wall_hp[sy, sx])
        Iq = fire_fixed.quantize_scalar(src_I) if src_I else None
        Tq = fire_fixed.quantize_scalar(src_T) if src_T else None

        n_max = int(round(max_seconds * float(CFG.clock.ticks_per_second)))
        dt = 1.0 / float(CFG.clock.ticks_per_second)
        rec = {k: [] for k in ("t", "T_src", "I_src", "T_rcv")}
        ignite_tick = None
        for k in range(1, n_max + 1):
            if not natural:
                g.fire[sy, sx] = Iq
                g.temperature[sy, sx] = Tq
                g.wall_hp[sy, sx] = hp0          # infinite fuel: an ESTABLISHED burner
            sim.set_paused(False)
            sim.step()
            rec["t"].append(k * dt)
            rec["T_src"].append(int(g.temperature[sy, sx]) / FP_ONE)
            rec["I_src"].append(int(g.fire[sy, sx]) / FP_ONE)
            rec["T_rcv"].append(int(g.temperature[sy, rx]) / FP_ONE)
            if ignite_tick is None and int(g.fire[sy, rx]) > 0:
                ignite_tick = k
                break
        for key in rec:
            rec[key] = np.asarray(rec[key], dtype=np.float64)
        m = dict(material=g.materials.names[mat_id], mat_id=mat_id, gap=gap,
                 natural=natural, src_I=src_I, src_T=src_T, ign_temp=ign,
                 cool_shift=int(g.cool_shift[sy, rx]),
                 T_emit_gate=float(getattr(CFG.physics.fire, "T_emit_gate", 180.0)),
                 rcv_T_max=float(rec["T_rcv"].max()) if rec["T_rcv"].size else 0.0,
                 ignite_tick=ignite_tick,
                 ignite_s=(ignite_tick * dt) if ignite_tick else float("nan"),
                 rec=rec, dt=dt, overrides=dict(overrides or {}))
        if verbose:
            _print(m)
        return m
    finally:
        restore_overrides(restore)


def _print(m):
    src = ("NATURAL (ignition_seed)" if m["natural"]
           else f"pinned I={m['src_I']} T={m['src_T']} game")
    print(f"  SPREAD  {m['material']:10s} gap={m['gap']} air tile(s)  source: {src}")
    print(f"          receiver cool_shift={m['cool_shift']}  T_emit_gate="
          f"{m['T_emit_gate']:.0f}  ignition_temp={m['ign_temp']:.0f}")
    print(f"          receiver T_max={m['rcv_T_max']:7.2f} game   -> IGNITES at "
          + (f"{m['ignite_s']:.1f} s" if m["ignite_tick"] else "NEVER (stalled)"))


def write_spread_csv(m, path):
    import csv
    rec = m["rec"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# fire_spread_bench one-air-gap radiative spread"])
        w.writerow([f"# material={m['material']} gap={m['gap']} natural={m['natural']} "
                    f"src_I={m['src_I']} src_T={m['src_T']}"])
        w.writerow([f"# cool_shift={m['cool_shift']} T_emit_gate={m['T_emit_gate']} "
                    f"ignition_temp={m['ign_temp']} ignite_s={m['ignite_s']}"])
        w.writerow(["t_s", "T_src_game", "I_src", "T_receiver_game"])
        for i in range(len(rec["t"])):
            w.writerow([f"{rec['t'][i]:.4f}", f"{rec['T_src'][i]:.3f}",
                        f"{rec['I_src'][i]:.6f}", f"{rec['T_rcv'][i]:.3f}"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--material", default=None, choices=sorted(NAMES))
    ap.add_argument("--src-I", type=float, default=0.45)
    ap.add_argument("--src-T", type=float, default=310.0)
    ap.add_argument("--gap", type=int, default=1)
    ap.add_argument("--natural", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=300.0)
    ap.add_argument("--set", dest="sets", action="append", default=[],
                    metavar="KEY=VALUE")
    ap.add_argument("--csv-dir", default=None)
    args = ap.parse_args(argv)

    overrides = {}
    for s in args.sets:
        key, val = s.split("=", 1)
        overrides[key.strip()] = val.strip()

    mats = [args.material] if args.material else ["kindling", "furniture"]
    for mat in mats:
        m = run_spread(mat, src_I=args.src_I, src_T=args.src_T, gap=args.gap,
                       natural=args.natural, max_seconds=args.max_seconds,
                       overrides=overrides, verbose=True)
        if args.csv_dir:
            d = Path(args.csv_dir)
            d.mkdir(exist_ok=True)
            tag = "natural" if args.natural else f"pinned{int(args.src_T)}"
            write_spread_csv(m, d / f"spread_{mat}_gap{args.gap}_{tag}.csv")


if __name__ == "__main__":
    main()
