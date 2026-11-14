# Tick budget: the generated ship at three sizes, headless (#125)

2026-10-29. This is a measurement only. Nothing in the sim changed, and the golden did not move.
Erik rules on the v1 map size. This doc gives a recommendation and the numbers behind it.

## Machine and method

- **Machine:** Home Desktop `DESKTOP-0E98HUV`, AMD Ryzen 7 3700X (8 cores, 3.6 GHz base),
  ~16 GB RAM, NVIDIA GeForce RTX 3070 (sm_86, CUDA runtime 12.4, driver 13030). Specs are
  from `ClaudeSync/environment.md`, because `docs/dev_setup.md` names only the Python and
  build per machine. Python is anaconda base 3.11.
- **Code:** branch `125-tick-budget`, cut from main `d5e6c30`.
  - CPU: the worktree's own build, `cpp/build_cpu_home.bat`.
  - CUDA: the main checkout's `cpp/build_cuda` .pyd, built 07:37 today. That is after the last
    `cpp/` commit (07:32), so the build is current. It was imported read-only through
    `--cuda-build`.
  - All 8 live backends were ON, from `tools/run_on_cuda.py`'s list.
  - "cuda" is the per-call path. "cuda+resident" also turns on `physics_runner.set_residency(True)`,
    which is what `--resident` does.
- **Levels:** `tools/gen_ship.py --seed 1 --size {small,medium,large} --vents`, all at
  0.333 m tiles, with a space boundary. They were written to a scratch levels folder, not
  `levels/`.

  | size | grid | rooms | doors | fuel tiles |
  |---|---|---|---|---|
  | small | 60 × 40 | 15 | 16 | 32 |
  | medium | 100 × 70 | 44 | 60 | 86 |
  | large | 160 × 110 | 99 | 144 | 248 |

  The playground is the reference: 100 × 70, 0.333 m, no space boundary.
- **Match:** the default generated match, which is 4 v 4 riflemen and 4 zombies on seed 1. The
  time limit was replaced by 60 s. `[clock] ticks_per_second = 24`, so a full run is 1440 ticks.
  - Each cell was run twice (two repeats in one process).
  - The table shows the mean of the two repeats; the ticks/s column shows both.
- **Controller configs:**
  - **bots**: both teams on `bots`, the horde on `zombies`. This is the v1 match.
  - **idle+horde**: teams `idle`, the horde on `zombies`.
  - **all idle**: the horde on `idle` too. This is the physics-only baseline.
- **Instrument:** `tools/bench_bot_matches.py --timing`, which this branch adds. It uses the one
  match loop (`run_match` with `on_tick`).
  - Each tick is split into **ctrl** and **step**:
    - ctrl = every controller's `handle_frame` plus the runner's `is_terminal`, i.e. the bots' A*
      and vision and the horde's law.
    - step = `sim.step()`, i.e. physics and gameplay.
  - Tick 1 is left out of the sample (warm-up).
  - The bench's own watcher time is left out.
  - Example command: `tools/bench_bot_matches.py --level ship_s1_medium --levels-dir <scratch> --seeds 1 --timing --sim-seconds 60 --repeat 2 [--cuda --cuda-build <dir> [--resident]] [--controller idle [--zombies-idle]]`.
- **Determinism check, a free side result:** for each (level, config), the final `tick_digest`
  was identical on all three backends and both repeats.
  - The idle+horde and all-idle digests are also equal to each other. On these ships the horde
    never sees an idle marine, so it never moves, and **idle+horde measures physics only**.

## Results

"real-time ×" = sim seconds per wall second = (ticks/s) / 24. A value of 1.0 or more is faster
than real time. Times are in ms per tick.

### CPU

| level | controllers | ticks run | ticks/s (rep1 / rep2) | real-time × | total mean / p50 / p95 / max | step mean / p95 | ctrl mean / p95 / max | result |
|---|---|---|---|---|---|---|---|---|
| small | bots | 1114 | 141.8 / 142.1 | **5.91** | 7.0 / 6.8 / 8.1 / 24 | 6.4 / 6.9 | 0.63 / 1.23 / 17 | team 1 wins |
| small | idle+horde | 1440 | 167.1 / 166.8 | 6.96 | 6.0 / 6.5 / 6.9 / 10 | 6.0 / 6.9 | 0.01 / 0.02 / 0 | draw |
| small | all idle | 1440 | 169.8 / 169.8 | 7.08 | 5.9 / 6.4 / 6.8 / 10 | 5.9 / 6.8 | 0.01 / 0.01 / 0 | draw |
| medium | bots | 1440 | 46.7 / 47.5 | **1.96** | 21.2 / 20.7 / 21.7 / 79 | 19.9 / 21.0 | 1.29 / 1.13 / 59 | draw |
| medium | idle+horde | 1440 | 51.9 / 53.0 | 2.19 | 19.1 / 19.3 / 20.0 / 30 | 19.0 / 20.0 | 0.01 / 0.02 / 0 | draw |
| medium | all idle | 1440 | 52.2 / 53.2 | 2.20 | 19.0 / 19.2 / 19.8 / 27 | 19.0 / 19.8 | 0.01 / 0.01 / 0 | draw |
| large | bots | 1308 | 17.0 / 17.1 | **0.71** | 58.6 / 51.5 / 54.1 / 325 | 50.3 / 51.8 | 8.26 / 1.36 / 272 | team 1 wins |
| large | idle+horde | 1440 | 19.9 / 20.1 | 0.83 | 50.1 / 50.2 / 51.6 / 70 | 50.1 / 51.6 | 0.02 / 0.02 / 0 | draw |
| large | all idle | 1440 | 19.9 / 20.1 | 0.83 | 50.0 / 50.1 / 51.5 / 75 | 50.0 / 51.5 | 0.01 / 0.02 / 0 | draw |
| playground | bots | 15 | 77.4 / 77.0 | 3.22 | 13.0 / 12.0 / 15.8 / 16 | 11.8 / 14.4 | 1.17 / 1.63 / 2 | team 0 wins |
| playground | idle+horde | 1440 | 78.9 / 80.8 | 3.33 | 12.5 / 12.4 / 15.2 / 25 | 12.5 / 15.2 | 0.02 / 0.02 / 0 | draw |
| playground | all idle | 1440 | 78.8 / 81.2 | 3.33 | 12.5 / 12.4 / 15.2 / 24 | 12.5 / 15.1 | 0.01 / 0.01 / 0 | draw |

### CUDA, per-call path (`--cuda`)

| level | controllers | ticks run | ticks/s (rep1 / rep2) | real-time × | total mean / p50 / p95 / max | step mean / p95 | ctrl mean / p95 / max | result |
|---|---|---|---|---|---|---|---|---|
| small | bots | 1114 | 68.7 / 54.7 | **2.57** | 16.4 / 15.9 / 19.3 / 35 | 15.8 / 18.0 | 0.66 / 1.27 / 18 | team 1 wins |
| small | idle+horde | 1440 | 68.2 / 56.5 | 2.60 | 16.2 / 16.6 / 18.4 / 25 | 16.2 / 18.4 | 0.02 / 0.02 / 0 | draw |
| small | all idle | 1440 | 71.4 / 56.6 | 2.67 | 15.8 / 16.1 / 18.3 / 24 | 15.8 / 18.3 | 0.01 / 0.01 / 0 | draw |
| medium | bots | 1440 | 44.8 / 40.3 | **1.77** | 23.6 / 23.5 / 25.7 / 83 | 22.3 / 24.7 | 1.32 / 1.15 / 59 | draw |
| medium | idle+horde | 1440 | 49.7 / 43.9 | 1.95 | 21.4 / 22.0 / 24.4 / 31 | 21.4 / 24.4 | 0.02 / 0.02 / 0 | draw |
| medium | all idle | 1440 | 50.6 / 43.9 | 1.97 | 21.3 / 21.8 / 24.3 / 31 | 21.2 / 24.3 | 0.01 / 0.01 / 0 | draw |
| large | bots | 1308 | 24.7 / 23.6 | **1.01** | 41.4 / 34.8 / 37.8 / 308 | 33.1 / 35.6 | 8.37 / 1.37 / 279 | team 1 wins |
| large | idle+horde | 1440 | 32.8 / 30.6 | 1.32 | 31.6 / 32.4 / 34.1 / 42 | 31.6 / 34.1 | 0.02 / 0.02 / 0 | draw |
| large | all idle | 1440 | 33.5 / 30.7 | 1.34 | 31.2 / 32.0 / 33.9 / 46 | 31.2 / 33.9 | 0.01 / 0.02 / 0 | draw |
| playground | bots | 15 | 61.2 / 61.1 | 2.55 | 16.4 / 15.0 / 19.6 / 20 | 15.2 / 18.1 | 1.17 / 1.53 / 2 | team 0 wins |
| playground | idle+horde | 1440 | 54.5 / 46.1 | 2.10 | 20.0 / 20.6 / 23.4 / 28 | 20.0 / 23.4 | 0.02 / 0.02 / 0 | draw |
| playground | all idle | 1440 | 54.7 / 46.4 | 2.11 | 19.9 / 20.5 / 23.1 / 28 | 19.9 / 23.1 | 0.01 / 0.01 / 0 | draw |

### CUDA with field residency (`--cuda --resident`)

| level | controllers | ticks run | ticks/s (rep1 / rep2) | real-time × | total mean / p50 / p95 / max | step mean / p95 | ctrl mean / p95 / max | result |
|---|---|---|---|---|---|---|---|---|
| small | bots | 1114 | 124.7 / 107.9 | **4.84** | 8.6 / 8.1 / 10.8 / 28 | 8.0 / 9.0 | 0.67 / 1.27 / 17 | team 1 wins |
| small | idle+horde | 1440 | 130.8 / 115.0 | 5.12 | 8.2 / 8.3 / 9.0 / 13 | 8.2 / 9.0 | 0.02 / 0.02 / 0 | draw |
| small | all idle | 1440 | 132.5 / 117.2 | 5.20 | 8.0 / 8.0 / 9.0 / 14 | 8.0 / 9.0 | 0.01 / 0.01 / 0 | draw |
| medium | bots | 1440 | 75.0 / 67.8 | **2.98** | 14.0 / 12.9 / 15.6 / 76 | 12.7 / 14.5 | 1.31 / 1.13 / 61 | draw |
| medium | idle+horde | 1440 | 85.3 / 78.2 | 3.41 | 12.3 / 12.1 / 13.6 / 18 | 12.2 / 13.6 | 0.02 / 0.02 / 0 | draw |
| medium | all idle | 1440 | 87.4 / 78.9 | 3.47 | 12.1 / 11.9 / 13.3 / 18 | 12.0 / 13.3 | 0.01 / 0.01 / 0 | draw |
| large | bots | 1308 | 32.5 / 30.9 | **1.32** | 31.5 / 24.6 / 27.3 / 306 | 23.2 / 25.4 | 8.33 / 1.33 / 281 | team 1 wins |
| large | idle+horde | 1440 | 49.0 / 42.9 | 1.91 | 21.9 / 22.4 / 24.3 / 33 | 21.8 / 24.3 | 0.02 / 0.02 / 0 | draw |
| large | all idle | 1440 | 48.9 / 43.7 | 1.93 | 21.7 / 22.2 / 24.4 / 32 | 21.7 / 24.4 | 0.01 / 0.02 / 0 | draw |
| playground | bots | 15 | 86.3 / 87.3 | 3.62 | 11.5 / 10.3 / 14.7 / 15 | 10.4 / 13.2 | 1.16 / 1.54 / 2 | team 0 wins |
| playground | idle+horde | 1440 | 103.2 / 90.2 | 4.03 | 10.4 / 10.2 / 11.8 / 16 | 10.4 / 11.8 | 0.02 / 0.02 / 0 | draw |
| playground | all idle | 1440 | 105.3 / 89.5 | 4.06 | 10.3 / 10.1 / 11.9 / 16 | 10.3 / 11.9 | 0.01 / 0.01 / 0 | draw |

### Notes on the numbers

- **Spread:**
  - On CPU the two repeats agree within ~2 %.
  - On CUDA the **second repeat in the same process is consistently 10–20 % slower** than the
    first, on both paths and on every level.
    - That points to something that grows across matches in one process, such as device
      allocations or scratch caches.
    - It was not investigated. A training loop that runs many episodes per process would see the
      slower number, or worse.
- **Matches that ended before 60 s:**
  - Bots on small ended at tick 1114: team 1 won at 46 s.
  - Bots on large ended at tick 1308.
  - Bots on the playground ended in 15 ticks (the spawns see each other, as recorded under #121).
    That row is not a measurement of the bots.
- **Physics scales linearly in area on CPU.** Step time is ~2.5 / 2.7 / 2.85 µs per cell for small
  (2,400 cells) / medium (7,000) / large (17,600).
- **The playground is faster than the medium ship at the same grid.** On CPU it is 12.5 ms against
  19 ms.
  - The ship sits in a space boundary and has ~3.5 ms/tick of pure-Python `find_burst_walls` (see
    the profile).
  - It was not dug into further.
- **CUDA per-call is slower than CPU** on small and medium, and only wins on large. Residency is
  about 1.6× faster than CPU on medium and 2.4× on large, but still a little slower than CPU on
  small.

## What the bots cost

| size | bots' ctrl, mean | bots' ctrl, p95 | bots' ctrl, max | step, bots − idle (CPU) | total cost of the bots (CPU) |
|---|---|---|---|---|---|
| small | 0.6 ms | 1.2 ms | 17 ms | +0.4 ms | ~1.0 ms / tick (≈ 15 %) |
| medium | 1.3 ms | 1.1 ms | ~60 ms | +0.9 ms | ~2.1 ms / tick (≈ 11 %) |
| large | 8.3 ms | 1.4 ms | ~275 ms | +0.2 ms | ~8.5 ms / tick (≈ 17 %) |

- **The bots' cost is spikes, not a steady load.** The p95 stays near 1 ms on every size, but the
  mean on large is 8 ms because of replans that take 100–280 ms.
  - Those are pure-Python A* (`pathfinding.astar`) runs across a 160 × 110 grid.
  - A 280 ms spike is about 7 ticks of real time, so it shows as a visible hitch when watching.
- The step also grows a little with bots, from more moving units and events.
- **The horde costs nothing measurable** on these ships, because it never sees an idle marine.

## Profile: medium ship, CPU, bots, 600 ticks (cProfile)

The profiled run took 18.7 s against ~12.7 s unprofiled. cProfile inflates Python calls but not
the C++ call, so the real share of `run_substeps` is about 14 ms of the ~21 ms tick (≈ 70 %).

Top cumulative functions, below the bench / `run_match` frames:

| # | function | cumulative | per tick | what it is |
|---|---|---|---|---|
| 1 | `Simulation.step` | 13.77 s | 22.9 ms | the whole sim tick |
| 2 | `PhysicsRunner.step` | 9.96 s | 16.6 ms | physics orchestration |
| 3 | `breach_physics.run_substeps` (C++) | 8.53 s | 14.2 ms | **physics kernels: EOS substeps, trace transport** |
| 4 | `control_bots.BotTeam.handle_frame` | 4.46 s | 7.4 ms (2 teams) | the bots |
| 5 | `control_bots._drive` → `_advance` → `_walk_to` | 3.13 s | 5.2 ms | bot steering incl. replans |
| 6 | `control_bots._plan` | 2.83 s | 4.7 ms (200 calls) | the bots' replans |
| 7 | `pathfinding.astar` | 2.81 s | 14 ms per call, 204 calls | **pure-Python A*** (1.03 M `_in_bounds` + `blocked` calls) |
| 8 | `GameMap.find_burst_walls` | 2.08 s | 3.5 ms | **pure-Python wall-burst scan, every tick** (tottime 2.07 s, almost all its own) |
| 9 | `vision.state_for` / `VisionSystem.compute` | 1.30 s | 2.2 ms | the bots' vision (57,600 `can_see`, 52,296 `has_los`) |
| 10 | `GameMap.stamp_units` / `_stamp_units_cpp` | 0.65 s | 1.1 ms | unit stamping |

Next after these: `_run_combustion` (0.55 s, of which the C++ `breach_physics.step` is 0.52 s),
`optics_fixed.quantize_scalar` (0.41 s, 28,801 calls), `exchange.apply_wave_push` (0.30 s).

**Reading:**
- The C++ kernels are about two thirds of a real tick.
- The largest Python costs are:
  - the bots' A* (about 1 ms/tick averaged, but arriving as 14 ms+ bursts);
  - `find_burst_walls` (3.5 ms every tick, ~17 % of the medium tick, independent of the bots);
  - vision.
- `find_burst_walls` and A* are the two cheapest wins if the budget ever needs one. Both are
  records only; neither was touched here.

## Reading and recommendation

**Who runs faster than real time:**
- With the full v1 match (bots + horde):
  - **small** runs faster than real time on every backend (CPU 5.9×, resident 4.8×, per-call 2.6×).
  - **medium** runs faster than real time on every backend (CPU 2.0×, resident 3.0×, per-call 1.8×).
  - **large** is **slower than real time on CPU (0.71×)**, only just real time on per-call CUDA
    (1.0×), and 1.3× on resident CUDA.
- On large, the bots' 100–280 ms A* spikes come on top of that and would show as hitches in a
  window.
- The window's own rendering and the light sweep were not measured. A headless match requests no
  light, but a watched one does, so real-time headroom matters when watching.

**What the bots cost:** about 1 / 2 / 8.5 ms per tick on small / medium / large, which is 11–17 %
of a tick. The cost is spike-dominated on large. Physics dominates at every size.

**Recommendation (Erik rules):**

**(i) Watching a match: medium.** It has 2× headroom on CPU and 3× on resident CUDA, which leaves
room for the renderer and the light sweep. Its bot spikes (~60 ms max) are rare. Large is not
watchable on CPU and is marginal on GPU until the bots' A* is fixed.

**(ii) Headless training at N=1: small for the first agent work, with medium as the v1 target
map.**
- Throughput:
  - Small runs at ~140 ticks/s on CPU, so one 60 s match takes ~8 s of wall time.
  - Medium runs at ~47 ticks/s on CPU (~31 s) or ~70–75 on resident CUDA (~20 s).
  - Large runs at ~17 ticks/s on CPU (~85 s).
- At N=1 the **CPU is the best backend for small**, and on medium CPU and resident are close
  enough that CPU processes in parallel are likely the better use of the box. That is 8 cores; it
  is a guess and was not measured.
- So the recommendation is:
  - a v1 size of **medium** (100 × 70 at 0.333 m, about 33 × 23 m);
  - **small** as the training curriculum's first map;
  - **large** deferred until the batched GPU path (#29 / RL-batch habits) or a faster A*.

## Not measured, and why

- **The window:** rendering, the light sweep, and `main.py --warp`. The brief said not to run the
  game window, and a headless match requests no light. So every number here is lower than the cost
  of a watched match.
- **More than one seed per size.** Only seed 1 was run. A different seed gives a different room
  graph at the same grid size, and the physics cost is set mostly by area.
- **Multi-process CPU throughput** (several matches in parallel on 8 cores). It matters for
  training but is out of scope.
- **The CUDA repeat slowdown.** It was seen on every CUDA row but not investigated.
