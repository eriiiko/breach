"""
CLI runner for the EOS Phase-1.2 visual prototype (P0 scaffold).

    python prototypes/eos/run.py --scheme placeholder --scenario S1 --grid 160 --ticks 120 --out prototypes/eos/out/

Writes `<scenario>_<scheme>.gif` into --out and prints a timing summary
table (ms/tick, substeps/tick).

`--scheme` currently only has `placeholder` registered in SCHEMES below.
P1 (rung A) / P2 (rung B) / P-ctrl (control) add their entries to SCHEMES
-- that is the only run.py change those patches need; the CLI's `choices`
follow SCHEMES automatically.
"""

import argparse
from pathlib import Path

import numpy as np

from state import State
from scenarios import SCENARIOS, apply_event
from solver import PlaceholderSolver
from scheme_control import ControlSolver
from scheme_rung_a import RungASolver
from scheme_rung_b import RungBSolver
from shallow_water import ShallowWaterDriver
from render import render_frame, make_gif
from timing import TickTimer

DT = 0.083   # seconds; ~83 ms / 12 Hz engine tick (docs/eos_research_brief.md §2)

SCHEMES = {
    "placeholder": PlaceholderSolver,
    "control": ControlSolver,
    "rungA": RungASolver,
    "rungB": RungBSolver,
}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EOS Phase-1.2 visual prototype runner")
    p.add_argument("--scheme", choices=sorted(SCHEMES), default="placeholder")
    p.add_argument("--scenario", choices=sorted(SCENARIOS), default="S1")
    p.add_argument("--grid", type=int, default=160, help="square grid size, tiles per side")
    p.add_argument("--ticks", type=int, default=120)
    p.add_argument("--dt", type=float, default=DT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "out")
    p.add_argument("--fps", type=int, default=20, help="GIF playback fps")
    p.add_argument("--px-per-tile", type=int, default=4, help="render upscale factor")
    p.add_argument("--velocity-stride", type=int, default=8)
    p.add_argument("--no-velocity", action="store_true", help="skip the velocity-arrow overlay")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    np.random.seed(args.seed)   # belt-and-braces; scenario builders take an explicit seed too

    state, schedule = SCENARIOS[args.scenario](args.grid, args.grid, seed=args.seed)
    solver = SCHEMES[args.scheme]()

    # shallow_water.py is a fixed driver, not a --scheme option -- only S5 carries
    # nonzero tilt/water_depth, so it is the only scenario that steps it.
    water = ShallowWaterDriver(state) if args.scenario == "S5" else None

    pending = sorted(schedule, key=lambda e: e[1])
    event_i = 0
    timer = TickTimer()
    frames = []

    print(f"scenario={args.scenario}  scheme={args.scheme}  grid={args.grid}x{args.grid}  "
          f"ticks={args.ticks}  dt={args.dt}")

    for tick in range(args.ticks):
        while event_i < len(pending) and pending[event_i][1] == tick:
            kind = apply_event(state, pending[event_i])
            print(f"  tick {tick:4d}: event {kind}")
            event_i += 1

        timer.tick_start()
        if water is not None:
            water.step(state, args.dt)
        solver.step(state, args.dt)
        timer.tick_end(solver.last_substeps)

        frames.append(render_frame(
            state, px_per_tile=args.px_per_tile, show_velocity=not args.no_velocity,
            velocity_stride=args.velocity_stride,
        ))

    args.out.mkdir(parents=True, exist_ok=True)
    gif_path = args.out / f"{args.scenario}_{args.scheme}.gif"
    make_gif(frames, gif_path, fps=args.fps)
    print(f"wrote {gif_path}  ({len(frames)} frames)")

    timer.print_table(grid_w=args.grid, grid_h=args.grid, scenario=args.scenario, scheme=args.scheme)


if __name__ == "__main__":
    main()
