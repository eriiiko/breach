# EOS Phase-1.2 visual prototype

Throwaway numpy spike (not canon, touches no live engine path) built to answer one
question: in Breach's top-down setting, how much better is rung B (Kwatra
semi-implicit compressible) than rung A (Feldman-O'Brien prescribed-divergence
incompressible) -- and is rung B cheap enough to run in realtime? Judged by Erik's
eyes (GIFs) plus a cost table (ms/tick, substeps/tick). See `PLAN.md` for the full
plan and `docs/eos_research_brief.md` / `docs/eos_research_report.md` for the
physics background.

## Layout
- `state.py` -- grid + `State`: `smoke`, `T`, `vx`, `vy`, `P` (derived, unused until
  a real solver writes it), masks `solid`/`vacuum`/`door`, water fields
  `floor_height`/`tilt`/`water_depth`.
- `solver.py` -- the pluggable `Solver` interface every scheme implements, plus
  `PlaceholderSolver` (cheap swirl + semi-Lagrangian advection; not a candidate
  scheme, just exercises the pipeline).
- `shallow_water.py` -- ~40-line numpy port of the shipped pipe + damped-velocity
  water model (`docs/architecture/engine/07_fluid_and_water.md` §2), used only by
  S5, plus the W3 air-coupling helper (`free_air_height`, `FreeHeightTracker`).
- `scenarios.py` -- the 5 scenario builders (S1-S5, brief §8) + generic
  solver-agnostic event dispatch (`detonate`, `breach`, `ignite`, `release_water`,
  `open_door`).
- `render.py` -- `State` -> RGB frame (walls/vacuum/smoke/fire/water/velocity
  arrows), and GIF assembly.
- `timing.py` -- ms/tick + substeps/tick instrumentation, reusable by every scheme.
- `run.py` -- CLI entry point.

## Usage
```
C:/Users/steen/miniconda3/envs/data/python.exe prototypes/eos/run.py \
    --scheme placeholder --scenario S1 --grid 160 --ticks 120 --out prototypes/eos/out/
```

`--scheme` currently only has `placeholder` registered in the `SCHEMES` dict at the
top of `run.py`. P1 (rung A) / P2 (rung B) / P-ctrl (control) each add one entry to
that dict -- the only change those patches need to make in `run.py`; the CLI's
`--scheme` choices follow `SCHEMES` automatically.

Requires `imageio` in the data env (`pip install imageio`; already done as of P0).

## Solver interface

```python
class Solver(ABC):
    name: str
    last_substeps: int
    def step(self, state: State, dt: float) -> None: ...  # mutates state in place
```

`last_substeps` must be set before `step` returns -- the timing harness reads it
after every tick to print substeps/tick alongside ms/tick.

## Status

P0 scaffold only: grid/state, the 5 scenarios + events, the water driver, render,
timing, CLI. **No real fluid solver** -- `PlaceholderSolver` is a stand-in.
Rung A, rung B, and the control baseline are separate later patches that each drop
a new `Solver` subclass into `SCHEMES`.
