"""tools/bench_sight_lines.py -- the cost of sight for many tiles (#195 1c).

Two measurements, each in the process that runs it (run once per row, in a
fresh process, so no row inherits another's caches):

  --raw     rays per second on a map: random lines through
            ``vision.lines_clear`` (the C++ routine), ``has_los`` in a Python
            loop, and the pure-Python reference; and a 25 x 25 window
            (``window_clear``) against 625 ``has_los`` calls.
  --match   the per-tick cost of ``vision.compute`` over a fixed number of
            ticks of the level's own match (its match.toml teams and horde,
            the sight rule from ``--rule``): ``--path batch`` is
            ``vision.compute`` (under all_round, one many-rays call),
            ``--path pairwise`` is ``vision.compute_pairwise`` (every pair
            through ``can_see``: the path before 1c). Under cone the two are
            one function.

    C:/Users/steen/anaconda3/envs/data/python.exe tools/bench_sight_lines.py \\
        --ship ship_demo --raw
    ... --ship freighter-medium --match --rule all_round --path batch

``--ship`` is a level folder name under levels/ or ``freighter-medium`` /
``freighter-large`` (generated into a temp folder, seed 1). ``--cuda`` runs
the CUDA module (it must be built; the routine is the same host call).
"""
from __future__ import annotations

import argparse
import dataclasses
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tools")]


def _load_module(cuda: bool):
    if cuda:
        sys.path.insert(0, str(ROOT / "tests"))
        import cuda_harness
        cuda_harness.register_cuda_runtime()
        sys.path.insert(0, str(cuda_harness.CUDA_BUILD_DIR))
    else:
        sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))
    import breach_physics
    return breach_physics


def _level(ship: str):
    import level_loader
    if ship.startswith("freighter-"):
        import gen_ship as G
        plan = G.generate(1, ship.split("-", 1)[1], recipe="freighter",
                          profile="standard", zombies=4)
        base = Path(tempfile.mkdtemp(prefix="bench_sight_"))
        G.write(plan, base / plan.name)
        return level_loader.load(plan.name, levels_dir=str(base))
    return level_loader.load(ship, str(ROOT / "levels"))


def raw(bp, ship: str, n: int) -> None:
    import numpy as np
    from simulation import Simulation, vision
    from simulation.ruleset import ContinuousRealtime
    sim = Simulation(_level(ship), seed=1, breach_physics=bp,
                     enable_recorder=False, ruleset=ContinuousRealtime())
    g = sim.gmap
    h, w = g.solid.shape
    rng = np.random.default_rng(1)
    pairs = np.stack([rng.integers(0, h, n), rng.integers(0, w, n),
                      rng.integers(0, h, n), rng.integers(0, w, n)],
                     axis=1).astype(np.int32)
    lens = (np.maximum(abs(pairs[:, 0] - pairs[:, 2]),
                       abs(pairs[:, 1] - pairs[:, 3]))).mean()
    t = time.perf_counter()
    for _ in range(5):
        vision.lines_clear(g, pairs)
    cpp = (time.perf_counter() - t) / 5
    m = min(n, 4000)
    t = time.perf_counter()
    for a, b, c, d in pairs[:m].tolist():
        g.has_los(a, b, c, d)
    py = time.perf_counter() - t
    solid, screen = vision._planes(g)
    t = time.perf_counter()
    vision.sight_lines_reference(solid, screen, pairs[:m])
    ref = time.perf_counter() - t
    t = time.perf_counter()
    for _ in range(50):
        vision.window_clear(sim, h // 2, w // 2, h // 2 - 12, w // 2 - 12,
                            25, 25)
    win = (time.perf_counter() - t) / 50
    t = time.perf_counter()
    for yy in range(h // 2 - 12, h // 2 + 13):
        for xx in range(w // 2 - 12, w // 2 + 13):
            g.has_los(h // 2, w // 2, yy, xx)
    win_py = time.perf_counter() - t
    print(f"RAW {ship} {h}x{w} mean line {lens:.1f} tiles")
    print(f"  C++ routine : {n / cpp:12,.0f} rays/s ({n} rays in "
          f"{cpp * 1e3:.2f} ms)")
    print(f"  has_los     : {m / py:12,.0f} rays/s")
    print(f"  reference   : {m / ref:12,.0f} rays/s")
    print(f"  window 25x25: {win * 1e3:.3f} ms (C++) vs {win_py * 1e3:.2f} ms "
          f"(625 has_los) -> {win_py / win:.0f}x")


def match(bp, ship: str, rule: str, path: str, ticks: int) -> None:
    from simulation import vision
    from simulation.match_session import MatchSession
    level = _level(ship)
    spec = dataclasses.replace(level.match, sight=rule)
    session = MatchSession(level, spec, breach_physics=bp)
    sim = session.sim
    fn = vision.compute if path == "batch" else vision.compute_pairwise
    for _ in range(3):                      # warm-up (not sampled)
        session.tick()
        fn(sim)
    samples, units = [], []
    for _ in range(ticks):
        session.tick()
        t = time.perf_counter()
        fn(sim)
        samples.append(time.perf_counter() - t)
        units.append(sum(1 for u in sim.units if u.alive))
    ms = [s * 1e3 for s in samples]
    print(f"MATCH {ship} rule={rule} path={path} ticks={ticks} "
          f"living units mean {statistics.mean(units):.1f}: compute "
          f"mean {statistics.mean(ms):.3f} ms  p50 {statistics.median(ms):.3f}"
          f"  p95 {sorted(ms)[int(0.95 * len(ms))]:.3f}  max {max(ms):.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ship", default="ship_demo")
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--match", action="store_true")
    ap.add_argument("--rule", choices=("cone", "all_round"), default="all_round")
    ap.add_argument("--path", choices=("batch", "pairwise"), default="batch")
    ap.add_argument("--ticks", type=int, default=120)
    ap.add_argument("--rays", type=int, default=20000)
    ap.add_argument("--cuda", action="store_true")
    a = ap.parse_args()
    bp = _load_module(a.cuda)
    if a.raw:
        raw(bp, a.ship, a.rays)
    if a.match:
        match(bp, a.ship, a.rule, a.path, a.ticks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
