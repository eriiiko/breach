"""tools/match_pin_check.py -- the pinned ship_demo match on one backend (#29 P10).

Plays THE pinned match -- ``levels/ship_demo``, its match.toml, seed 1 --
through the one headless runner (``tools/run_match.py::run_match``) on the
backend named, and compares outcome, team, ticks, ``field_digest`` and
``tick_digest`` with the pin ``tests/test_zombie_control.py`` holds
(``_PRE_122_SHIP_DEMO``, imported, never copied). Exit 0 on equal, 1 on any
difference, printing both.

    <breach py> tools/match_pin_check.py                     # CPU build
    <breach py> tools/match_pin_check.py --backend cuda      # every live backend on
    <breach py> tools/match_pin_check.py --backend resident  # + field residency

``cuda`` / ``resident`` import the CUDA build from ``cpp/build_cuda`` before
anything imports the CPU one and turn its backends on through
``tools/run_on_cuda.py`` (the GPU launch path's own helpers), ``resident``
adds ``physics_runner.set_residency(True)`` -- the same switches as
``main.py --cuda [--resident]`` and ``bench_bot_matches --cuda [--resident]``.
This is the Linux-vs-Windows digest gate of P10 and the container's
``%test`` (``containers/breach.def``): one command, the same answer on every
platform and backend, or a named difference.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "tools", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

KEYS = ("outcome", "team", "ticks", "field_digest", "tick_digest")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="match_pin_check")
    ap.add_argument("--backend", choices=("cpu", "cuda", "resident"),
                    default="cpu")
    args = ap.parse_args(argv)
    if args.backend != "cpu":
        # Before anything imports breach_physics: the CUDA build wins.
        from run_on_cuda import enable_all_backends, setup_cuda_import
        setup_cuda_import()
        import breach_physics as bp
        enable_all_backends(bp)
        if args.backend == "resident":
            from simulation import physics_runner
            physics_runner.set_residency(True)
    from test_zombie_control import _PRE_122_SHIP_DEMO as PIN
    import breach_physics as bp
    import level_loader
    from run_match import run_match

    print(f"match_pin_check: backend {args.backend}, module {bp.__file__}, "
          f"build {getattr(bp, 'build_id', lambda: '?')()}", flush=True)
    level = level_loader.load("ship_demo", str(ROOT / "levels"))
    o = run_match(level)
    got = dict(outcome=o.result.outcome, team=o.result.team,
               ticks=o.result.ticks, field_digest=o.field_digest,
               tick_digest=o.tick_digest)
    bad = [k for k in KEYS if got[k] != PIN[k]]
    for k in KEYS:
        print(f"  {k:13s} {got[k]}" + ("" if k not in bad
                                         else f"   PIN {PIN[k]}"))
    print("match_pin_check: " + ("EQUAL to the pin" if not bad
                                 else f"DIFFERS in {', '.join(bad)}"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
