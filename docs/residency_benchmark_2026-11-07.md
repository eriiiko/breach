# Residency + batching benchmark: where a headless tick's time goes, and how it scales (#29)

2026-11-07. This is a measurement only, the first step of the residency + batching arc
(`docs/residency_where_we_stand_2026-11-07.md` §8 item 1). Nothing in the sim, the config or the
levels changed, and the golden did not move. The design doc is planned against these numbers.

## Machine and method

- **Machine:** Home Desktop, AMD Ryzen 7 3700X (8 cores), 16 GB RAM, NVIDIA GeForce RTX 3070
  8 GB (sm_86, CUDA runtime 12.4, driver 610.88 = CUDA driver API 13030). Python is the `data`
  env, 3.12.14, CuPy 14.2.0.
- **Code:** branch `29-residency-batching` at `6b63aaa` (main `ea7ee92` + the orientation page).
- **Build:** ONE module for every mode, the main checkout's `cpp/build_cuda`
  `breach_physics.cp312-win_amd64.pyd`, built 09:10 today, after the last `cpp/` commit
  (`a459b2f`, 08:56), imported read-only.
  - **cpu** = that module with every backend OFF and residency off. Unlike the tick-budget doc
    (`docs/tick_budget_2026-10-29.md`), which used a separate CPU build, the CPU numbers here come
    from the CUDA-built module. The final tick digest says whether the paths agree, and it was
    identical on all three backends in every configuration below.
  - **cuda** = every live backend on (`tools/run_on_cuda.py`'s eight setters): the per-call path.
    Combustion is not among them (its twin is off, #71), so combustion runs on the host in every
    mode.
  - **resident** = those plus `physics_runner.set_residency(True)`, as `--resident` does.
- **Levels:** `tools/gen_ship.py --seed 1 --size {small,medium,large} --vents --out <scratch>`,
  written to a scratch folder outside `levels/`. They are the tick-budget doc's ships: 15 / 44 / 99
  rooms, 16 / 60 / 144 doors, 32 / 86 / 248 fuel tiles, at 60 × 40 / 100 × 70 / 160 × 110 tiles of
  0.333 m.
- **Every configuration ran in a fresh process.** The tick-budget doc found a second CUDA match in
  one process 10 to 20 % slower; that slowdown is not in these numbers.
- **Machine time:** about 6 minutes of runs in all (the match profiles 2 min, the stacked sweeps
  78 s, the batched sweep bench 38 s, smoke tests and one failed Nsight attempt the rest).

### The instruments (this branch)

- **`tools/tick_stage_profile.py`** (new, headless, pyray-free): wall-clock timers
  (`time.perf_counter`) patched at class or module level around the callables each numbered slot of
  `Simulation.step` invokes, and around the stages of slot 7. No sim state is read or written.
  - Per measured tick it keeps each label's seconds (0 when the stage did not run), so every row
    has a per-tick **mean** and **p95**.
  - A slot that is inline code in `step` (the event clear, the slot-9 burn-through loop, the
    end-of-tick plane wipes) lands in "(rest of step)". The host pre-physics of slot 7 (ambient
    args, the water pre-step, lazy binds, device pointers) lands in "rest of 7".
  - **The bus:** `GameMap.from_host` / `to_host` are wrapped to count calls and bytes per tick by
    field name (`arr.nbytes` of the host mirror).
  - **Attribution:** with `--profile-sync`, `cupy.cuda.Device().synchronize()` runs after every
    device stage. A resident launch core returns before the device finishes, so without a sync its
    device time lands on whichever later stage blocks first (`to_host`). With the sync, each stage
    carries the device work it launched, and the attributed sum can exceed the plain tick.
  - **The plain total** (the truth) is always from a run without the sync. Its timers are in place,
    but they cost about 1 µs per wrapped call (about 30 per tick).
- **`tools/bench_bot_matches.py --profile [--profile-sync] [--profile-skip K]`** and
  **`--build DIR`**. `--build DIR` imports `breach_physics` from any build; without `--cuda`, every
  backend stays off. Ticks 1 and 2 are skipped (warm-up: residency allocation, first-touch
  scratch). The existing `--timing` gives total, step and ctrl as in the tick-budget doc.
- **`tools/bench_resident_scaling.py`** (new): the stacked-grid probe (Deliverable 2, below).
- **`tools/bench_radiation_sweep.py --cuda --envs ...`** (extended): the batched sweep launch core
  over N envs (below).
- `--nvtx` (both benches) adds NVTX ranges per stage and a `cudaProfilerStart` / `Stop` window, for
  an Nsight Systems capture.

### Nsight Systems: installed, but it does not run here

`nsys` 2023.4.4 is installed (`C:\Program Files\NVIDIA Corporation\Nsight Systems 2023.4.4`). Every
`nsys profile` run, even `--trace=cuda --sample=none --cpuctxsw=none` on a ten-line CuPy script,
exits 1 with:

```
ReflexStatsTraceLoggingProvider provider requires administrator privileges
```

The session cannot elevate. So the copies made **inside the C++ per-call twins** were not traced.
They are **estimated from the code** below (which planes each `*_step` uploads and downloads),
and the estimate is labelled as one.

### Commands

```
# Deliverable 1: one fresh process per line. A synced (--profile-sync) twin of every --cuda line
# was run too.
python tools/bench_bot_matches.py --level ship_s1_medium --levels-dir <scratch>/levels --seeds 1 \
    --timing --sim-seconds 60 --build <main>/cpp/build_cuda --profile [--cuda [--resident]] [--profile-sync]
python tools/bench_bot_matches.py --level ship_s1_{small,large} ... --cuda --resident \
    --controller idle --zombies-idle --profile [--profile-sync]
# Deliverable 2
python tools/bench_resident_scaling.py --levels-dir <scratch>/levels --src ship_s1_small \
    --build <main>/cpp/build_cuda --sweep --ns 1,2,4,8,16,32,64,128 --backends cpu,cuda,resident
python tools/bench_resident_scaling.py ... --src ship_s1_medium --sweep --ns 1,4,16 --backends resident
python tools/bench_radiation_sweep.py --cuda --build <main>/cpp/build_cuda \
    --envs 1,2,4,8,16,32,64,128,256,512,1024 [--smoke]
python tools/bench_radiation_sweep.py --cuda --build ... --envs 1,4,16,64,256 --env-grid 70x100
```

## 1. The headless match, slot by slot (medium, bots)

The v1 match is 4 v 4 riflemen on `bots` plus 4 zombies on `zombies`, seed 1, with the time limit
set to 60 s (`[clock] ticks_per_second = 24`).

- **The match now ends at tick 722.** Team 1 wins at 30.1 s, the same on every backend. So 720
  ticks are profiled, not 1440.
- "ctrl" is the controllers plus `is_terminal`, outside `sim.step()`, measured as in #125.
- Times are ms per tick.

| backend | ticks run | ticks/s | real-time x | total mean / p95 | step mean / p95 | ctrl mean / p95 | digest |
|---|---|---|---|---|---|---|---|
| cpu | 722 | 48.1 | 2.00 | 20.78 / 21.67 | 19.28 / 20.54 | 1.50 / 1.18 | `f35ed2beadb7` |
| cuda | 722 | 56.5 | 2.35 | 17.70 / 21.44 | 16.18 / 20.02 | 1.52 / 1.21 | `f35ed2beadb7` |
| cuda (synced run) | 722 | 56.4 | 2.35 | 17.73 / 22.38 | 16.20 / 20.41 | 1.52 / 1.21 | `f35ed2beadb7` |
| resident | 722 | 86.7 | 3.61 | 11.54 / 12.47 | 10.02 / 10.89 | 1.51 / 1.22 | `f35ed2beadb7` |
| resident (synced run) | 722 | 85.7 | 3.57 | 11.67 / 12.77 | 10.16 / 11.02 | 1.52 / 1.23 | `f35ed2beadb7` |

### Per slot and per slot-7 stage

"plain" means no device sync; "attributed" means the synced run. Rows that are 0 on every backend
(water substeps: the ships carry no water) are left out. Values are mean / p95 in ms per tick.

| stage | cpu | cuda plain | cuda attributed | resident plain | resident attributed |
|---|---|---|---|---|---|
| step | 19.291 / 20.527 | 16.178 / 20.013 | 16.194 / 20.397 | 9.995 / 10.869 | 10.141 / 10.997 |
| 1 events + round head | 0.005 / 0.007 | 0.005 / 0.008 | 0.005 / 0.008 | 0.005 / 0.007 | 0.005 / 0.006 |
| 1b intents | 0.020 / 0.025 | 0.020 / 0.025 | 0.020 / 0.025 | 0.020 / 0.024 | 0.020 / 0.025 |
| 2 projectiles | 0.003 / 0.002 | 0.003 / 0.004 | 0.003 / 0.004 | 0.003 / 0.003 | 0.003 / 0.002 |
| 2b statuses | 0.008 / 0.010 | 0.009 / 0.010 | 0.009 / 0.010 | 0.008 / 0.009 | 0.008 / 0.009 |
| 3 movement | 0.282 / 0.335 | 0.285 / 0.348 | 0.287 / 0.355 | 0.302 / 0.349 | 0.282 / 0.343 |
| 4 shooting | 0.015 / 0.013 | 0.016 / 0.013 | 0.016 / 0.013 | 0.015 / 0.012 | 0.014 / 0.012 |
| 4b sprays | 0.003 / 0.003 | 0.003 / 0.003 | 0.003 / 0.003 | 0.003 / 0.003 | 0.003 / 0.003 |
| 4c resolve | 0.002 / 0.001 | 0.002 / 0.001 | 0.002 / 0.001 | 0.002 / 0.001 | 0.002 / 0.001 |
| 5 zombie AI | 0.154 / 0.200 | 0.155 / 0.195 | 0.155 / 0.194 | 0.154 / 0.192 | 0.153 / 0.193 |
| 6 stamp_units | 0.484 / 0.589 | 0.491 / 0.573 | 0.486 / 0.562 | 0.506 / 0.567 | 0.501 / 0.564 |
| 6b FieldEdit flush | 0.004 / 0.001 | 0.004 / 0.001 | 0.004 / 0.001 | 0.004 / 0.001 | 0.004 / 0.001 |
| **7 physics** | 17.441 / 19.110 | 14.273 / 18.247 | 14.288 / 18.333 | 8.040 / 8.909 | 8.226 / 9.009 |
| &nbsp;&nbsp;7 H2D `GameMap.from_host` | — | — | — | 0.243 / 0.300 | 0.473 / 0.866 |
| &nbsp;&nbsp;7 water (cpu / per-call `_step_water`) | 0.014 / 0.017 | 0.015 / 0.017 | 0.022 / 0.030 | — | — |
| &nbsp;&nbsp;7 EOS `run_substeps_resident` | — | — | — | 3.767 / 4.293 | 3.719 / 4.335 |
| &nbsp;&nbsp;7 EOS `run_substeps` (cpu / per-call) | 14.812 / 16.413 | 10.132 / 13.163 | 10.130 / 13.287 | — | — |
| &nbsp;&nbsp;7 trace tail (resident) | — | — | — | 0.115 / 0.149 | 0.118 / 0.165 |
| &nbsp;&nbsp;7 D2H `GameMap.to_host` | — | — | — | 0.329 / 0.384 | 0.330 / 0.399 |
| &nbsp;&nbsp;7 combustion (host) | 0.987 / 1.179 | 0.641 / 0.730 | 0.640 / 0.715 | 0.596 / 0.663 | 0.598 / 0.672 |
| &nbsp;&nbsp;7 sky exchange (host) | 0.001 / 0.002 | 0.001 / 0.002 | 0.001 / 0.002 | 0.001 / 0.001 | 0.001 / 0.001 |
| &nbsp;&nbsp;7 `step_tail` (ripple + fire + temperature + sweep) | 1.568 / 1.819 | 3.421 / 4.862 | 3.431 / 4.842 | 2.864 / 3.300 | 2.870 / 3.300 |
| &nbsp;&nbsp;rest of 7 (host pre-physics) | 0.058 / 0.082 | 0.063 / 0.081 | 0.064 / 0.080 | 0.124 / 0.156 | 0.116 / 0.134 |
| 9b over-pressure walls | 0.386 / 0.454 | 0.408 / 0.453 | 0.405 / 0.447 | 0.417 / 0.478 | 0.403 / 0.461 |
| 9c unit heat damage | 0.159 / 0.184 | 0.166 / 0.188 | 0.166 / 0.186 | 0.166 / 0.192 | 0.165 / 0.191 |
| 9c2 wind on units | 0.141 / 0.163 | 0.145 / 0.163 | 0.144 / 0.163 | 0.144 / 0.172 | 0.142 / 0.171 |
| 9c3 gas rows | 0.021 / 0.027 | 0.023 / 0.028 | 0.022 / 0.027 | 0.026 / 0.035 | 0.022 / 0.029 |
| 9d ignition | 0.047 / 0.064 | 0.047 / 0.061 | 0.047 / 0.062 | 0.058 / 0.070 | 0.059 / 0.073 |
| 9e entities | 0.060 / 0.070 | 0.062 / 0.067 | 0.062 / 0.068 | 0.061 / 0.071 | 0.060 / 0.068 |
| 10 tick end (ruleset) | 0.003 / 0.004 | 0.003 / 0.003 | 0.003 / 0.003 | 0.003 / 0.004 | 0.003 / 0.004 |
| (rest of step) | 0.055 / 0.069 | 0.060 / 0.070 | 0.066 / 0.080 | 0.060 / 0.075 | 0.068 / 0.085 |

Everything outside slot 7 (slots 1 to 6b and 9 to 10) costs 1.85 ms per tick on cpu and 1.96 ms on
resident (plain), of which stamp_units 0.5, 9b 0.4, movement 0.3, heat damage 0.16, zombie AI 0.15 and wind
0.14. The controllers add 1.5 ms outside the step.

### Slot 7 per size, resident (attributed)

- Small and large are physics-only ("all idle": teams and horde on `idle`), 1440 ticks.
- Medium is the bots run above.

| stage | small (all idle) mean / p95 | medium (bots) mean / p95 | large (all idle) mean / p95 |
|---|---|---|---|
| step | 7.141 / 7.985 | 10.141 / 10.997 | 14.248 / 15.790 |
| 6 stamp_units | 0.490 / 0.511 | 0.501 / 0.564 | 0.609 / 0.699 |
| 7 physics | 5.898 / 6.722 | 8.226 / 9.009 | 12.349 / 13.751 |
| 7 H2D `from_host` | 0.381 / 0.465 | 0.473 / 0.866 | 0.663 / 0.791 |
| 7 EOS `run_substeps_resident` | 2.848 / 3.656 | 3.719 / 4.335 | 5.027 / 5.548 |
| 7 trace tail (resident) | 0.112 / 0.151 | 0.118 / 0.165 | 0.131 / 0.194 |
| 7 D2H `to_host` | 0.274 / 0.323 | 0.330 / 0.399 | 0.471 / 0.613 |
| 7 combustion (host) | 0.199 / 0.235 | 0.598 / 0.672 | 1.932 / 2.211 |
| 7 sky exchange (host) | 0.001 / 0.001 | 0.001 / 0.001 | 0.001 / 0.003 |
| 7 `step_tail` (ripple + fire + temp + sweep) | 1.990 / 2.449 | 2.870 / 3.300 | 3.964 / 4.668 |
| rest of 7 (host pre-physics) | 0.093 / 0.111 | 0.116 / 0.134 | 0.160 / 0.221 |
| 9b over-pressure walls | 0.276 / 0.312 | 0.403 / 0.461 | 0.600 / 0.703 |
| plain total (unsynced run) | 7.075 / 7.905 | 11.535 / 12.470 | 13.012 / 13.688 |
| digest (plain / synced) | `3f9ffeaf0395` / `3f9ffeaf0395` | `f35ed2beadb7` / `f35ed2beadb7` | `9e075c5542af` / `9e075c5542af` |

### The bus per tick (resident)

**Seen from Python (measured):** `GameMap.from_host` / `to_host`, calls × bytes per tick. Each call
is one blocking copy.

| field | small | medium | large |
|---|---|---|---|
| H2D atmosphere | 1 x 9.4 KiB | 1 x 27.3 KiB | 1 x 68.8 KiB |
| H2D dyn_permeability | 1 x 9.4 KiB | 1 x 27.3 KiB | 1 x 68.8 KiB |
| H2D gas | 1 x 65.6 KiB | 1 x 191.4 KiB | 1 x 481.2 KiB |
| H2D gas_energy | 1 x 18.8 KiB | 1 x 54.7 KiB | 1 x 137.5 KiB |
| H2D is_ambient | 1 x 2.3 KiB | 1 x 6.8 KiB | 1 x 17.2 KiB |
| H2D is_vacuum | 1 x 2.3 KiB | 1 x 6.8 KiB | 1 x 17.2 KiB |
| H2D solid | 1 x 2.3 KiB | 1 x 6.8 KiB | 1 x 17.2 KiB |
| H2D temperature | 1 x 9.4 KiB | 1 x 27.3 KiB | 1 x 68.8 KiB |
| H2D thermal_solid | 1 x 2.3 KiB | 1 x 6.8 KiB | 1 x 17.2 KiB |
| H2D wind_x | 1 x 9.4 KiB | 1 x 27.3 KiB | 1 x 68.8 KiB |
| H2D wind_y | 1 x 9.4 KiB | 1 x 27.3 KiB | 1 x 68.8 KiB |
| D2H atmosphere | 1 x 9.4 KiB | 1 x 27.3 KiB | 1 x 68.8 KiB |
| D2H gas | 1 x 65.6 KiB | 1 x 191.4 KiB | 1 x 481.2 KiB |
| D2H gas_energy | 1 x 18.8 KiB | 1 x 54.7 KiB | 1 x 137.5 KiB |
| D2H temperature | 1 x 9.4 KiB | 1 x 27.3 KiB | 1 x 68.8 KiB |
| D2H wave_p | 1 x 9.4 KiB | 1 x 27.3 KiB | 1 x 68.8 KiB |
| D2H wind_x | 1 x 9.4 KiB | 1 x 27.3 KiB | 1 x 68.8 KiB |
| D2H wind_y | 1 x 9.4 KiB | 1 x 27.3 KiB | 1 x 68.8 KiB |
| **total H2D** | 11 copies, 140.6 KiB | 11 copies, 410.2 KiB | 11 copies, 1031.2 KiB |
| **total D2H** | 7 copies, 131.2 KiB | 7 copies, 382.8 KiB | 7 copies, 962.5 KiB |

`water_depth` / `flow_vx` / `flow_vy` do not move: water is dormant on these ships.

**Inside the C++ calls (ESTIMATE from the code, not traced):** the medium ship, `n` = 7,000 cells,
on a tick without fire.

| where | H2D copies | H2D bytes | D2H copies | D2H bytes | also per tick | source |
|---|---|---|---|---|---|---|
| `from_host` / `to_host` (measured, above) | 11 | 410 KiB | 7 | 383 KiB | — | Python counter |
| inside `run_substeps_resident` | 5 (coeffE, coeffS, absorb_q, cap2, cons_flag) + `div_source` on a blast tick | 20n B = 137 KiB | up to 5 counter blocks | < 1 KiB | D2D: 1 + 3 per EOS substep | `cuda_eos_resident.cu` 812-824, 854-881, 911, 987-1045 |
| temperature twin in `step_tail` (every tick) | 13 | 63n B + 32,000 B (E-table) = 462 KiB | 5 | 12n B + counters = 82 KiB | 20 `cudaMalloc` + 20 `cudaFree`, 3 memsets, 1 D2D, 5 launches, 1 device sync | `cuda_temperature.cu` 650-806 |
| radiation-sweep twin in `step_tail` (every tick) | 13 (6 planes, E-table, 3 scalars, the smoke plane, its coefficient, n_bulk) | 26n B + 32,020 B = 209 KiB | 6 (counters, 4 int64 planes, Fleck plane) | 36n B = 246 KiB | 1 arena malloc + free, 3 + max(h, w) = 103 launches (S16 shear), 1 device sync | `cuda_radiation_sweep.cu` 1050-1229, `physics_engine.cpp` 579-640 |
| fire twin in `step_tail` (only ticks with fire ≥ 0.001) | 10-13 | 31n B to 47n B | 4-5 | 12n B + counter | 15 malloc + free, 3 launches, 1 sync | `cuda_fire.cu` 449-560 |
| trace tail (resident) | small (folds) | — | small (books) | — | — | not counted |
| **total, non-fire tick (estimate)** | **about 42** | **about 1.2 MiB** | **about 23** | **about 0.7 MiB** | 21 malloc / free pairs, at least 2 explicit device syncs plus every blocking copy | |

## 2. The stacked-grid scaling probe

**The level:** N vertical copies of one generated ship in ONE grid of shape (N·h, w), written
through `level_lib` by `tools/bench_resident_scaling.py --make N`:
- the tilemap is tiled, and the ship's own space margin separates the copies;
- the boundary is `space`;
- each copy gets its own duct and two vents, with ids suffixed `_c<k>` and rows shifted (the ships
  have no `air_init.npy` / `temp_init.npy`);
- the diffuse picture is tiled, because the loader requires one;
- no lights, no spawns, no match.

**The run:** a `Simulation(level, seed=1, breach_physics=bp, enable_recorder=False,
ruleset=ContinuousRealtime())` stepped directly, physics only: no units, no controllers. Each
(N, backend) is a fresh subprocess:
- 2 warm-up ticks;
- 10 plain ticks (the totals);
- then 10 more ticks with the device sync (the per-stage split).

**Device memory** is `cudaMemGetInfo`'s used bytes after the timed ticks minus before the
`Simulation` was built, and CuPy's pool size.

**This is a COST proxy only.** The copies share one multigrid hierarchy, one radiation sweep and
one EOS substep count, so the physics is not N independent worlds. The per-cell cost is what is
wanted here.

### Small ship (60 x 40 per copy), totals (plain)

| N | cells | backend | ms / tick (mean / p95) | us / cell / tick | ms / env-tick | env-ticks / s | device MiB in use (after - before) | CuPy pool MiB | tick digest |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2400 | cpu | 5.11 / 5.59 | 2.127 | 5.106 | 196 | - | - | `f95daba133a6` |
| 1 | 2400 | cuda | 10.70 / 11.02 | 4.459 | 10.701 | 93 | 0 | 0 | `f95daba133a6` |
| 1 | 2400 | resident | 6.54 / 6.90 | 2.727 | 6.545 | 153 | 2 | 1 | `f95daba133a6` |
| 2 | 4800 | cpu | 9.41 / 10.12 | 1.960 | 4.705 | 213 | - | - | `03377517fec5` |
| 2 | 4800 | cuda | 13.97 / 15.59 | 2.911 | 6.986 | 143 | 0 | 0 | `03377517fec5` |
| 2 | 4800 | resident | 7.70 / 8.39 | 1.604 | 3.850 | 260 | 4 | 1 | `03377517fec5` |
| 4 | 9600 | cpu | 19.27 / 20.93 | 2.007 | 4.818 | 208 | - | - | `ff699dc6e733` |
| 4 | 9600 | cuda | 13.80 / 14.15 | 1.437 | 3.449 | 290 | 0 | 0 | `ff699dc6e733` |
| 4 | 9600 | resident | 9.59 / 10.18 | 0.999 | 2.398 | 417 | 6 | 2 | `ff699dc6e733` |
| 8 | 19200 | cpu | 37.50 / 40.29 | 1.953 | 4.688 | 213 | - | - | `017b0eca61fb` |
| 8 | 19200 | cuda | 19.17 / 19.55 | 0.998 | 2.396 | 417 | 0 | 0 | `017b0eca61fb` |
| 8 | 19200 | resident | 14.40 / 15.49 | 0.750 | 1.799 | 556 | 10 | 5 | `017b0eca61fb` |
| 16 | 38400 | cpu | 75.24 / 80.34 | 1.959 | 4.702 | 213 | - | - | `08c5a6d71380` |
| 16 | 38400 | cuda | 28.46 / 29.69 | 0.741 | 1.779 | 562 | 0 | 0 | `08c5a6d71380` |
| 16 | 38400 | resident | 22.37 / 25.22 | 0.582 | 1.398 | 715 | 18 | 10 | `08c5a6d71380` |
| 32 | 76800 | cpu | 150.93 / 163.33 | 1.965 | 4.717 | 212 | - | - | `0960d8adfb1a` |
| 32 | 76800 | cuda | 52.25 / 53.39 | 0.680 | 1.633 | 612 | 0 | 0 | `0960d8adfb1a` |
| 32 | 76800 | resident | 39.50 / 41.17 | 0.514 | 1.234 | 810 | 42 | 20 | `0960d8adfb1a` |
| 64 | 153600 | cpu | 301.81 / 325.01 | 1.965 | 4.716 | 212 | - | - | `59bdbb8ccc89` |
| 64 | 153600 | cuda | 128.10 / 130.82 | 0.834 | 2.002 | 500 | 0 | 0 | `59bdbb8ccc89` |
| 64 | 153600 | resident | 105.59 / 107.30 | 0.687 | 1.650 | 606 | 88 | 40 | `59bdbb8ccc89` |
| 128 | 307200 | cpu | 631.46 / 679.03 | 2.056 | 4.933 | 203 | - | - | `59ce30980ff0` |
| 128 | 307200 | cuda | 316.61 / 321.18 | 1.031 | 2.473 | 404 | 0 | 0 | `59ce30980ff0` |
| 128 | 307200 | resident | 282.57 / 291.51 | 0.920 | 2.208 | 453 | 204 | 79 | `59ce30980ff0` |

The per-call path frees what it allocates within each call, so its "in use after" is 0. Its peak
inside a call is not seen.

### Small ship, per-stage split (attributed, ms per tick)

"bus" is `from_host` + `to_host`. "rest of 7" includes the water and sky-exchange stages (both
dormant here). "other slots" is step − slot 7 − stamp_units − 9b.

| N | backend | step | EOS | bus | trace tail | combustion | step_tail | rest of 7 | 6 stamp_units | 9b walls | other slots |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | cpu | 5.46 | 4.20 | 0.00 | 0.00 | 0.21 | 0.52 | 0.05 | 0.04 | 0.28 | 0.17 |
| 1 | cuda | 10.65 | 7.28 | 0.00 | 0.00 | 0.22 | 2.56 | 0.06 | 0.04 | 0.30 | 0.19 |
| 1 | resident | 6.58 | 3.24 | 0.45 | 0.10 | 0.19 | 2.02 | 0.09 | 0.03 | 0.29 | 0.16 |
| 2 | cpu | 10.55 | 8.52 | 0.00 | 0.00 | 0.36 | 1.03 | 0.06 | 0.05 | 0.32 | 0.22 |
| 2 | cuda | 12.07 | 8.11 | 0.00 | 0.00 | 0.41 | 2.79 | 0.08 | 0.06 | 0.36 | 0.27 |
| 2 | resident | 7.59 | 3.71 | 0.49 | 0.11 | 0.35 | 2.22 | 0.10 | 0.04 | 0.35 | 0.22 |
| 4 | cpu | 20.79 | 16.80 | 0.00 | 0.00 | 1.09 | 2.02 | 0.07 | 0.06 | 0.40 | 0.35 |
| 4 | cuda | 13.99 | 8.75 | 0.00 | 0.00 | 0.75 | 3.50 | 0.09 | 0.08 | 0.44 | 0.38 |
| 4 | resident | 10.01 | 4.00 | 0.66 | 0.12 | 0.75 | 3.39 | 0.14 | 0.10 | 0.47 | 0.39 |
| 8 | cpu | 41.30 | 33.57 | 0.00 | 0.00 | 2.16 | 4.14 | 0.11 | 0.10 | 0.56 | 0.65 |
| 8 | cuda | 19.20 | 10.60 | 0.00 | 0.00 | 1.77 | 5.18 | 0.13 | 0.19 | 0.64 | 0.69 |
| 8 | resident | 13.71 | 4.42 | 0.78 | 0.12 | 1.73 | 5.11 | 0.17 | 0.17 | 0.60 | 0.62 |
| 16 | cpu | 82.15 | 67.31 | 0.00 | 0.00 | 3.89 | 8.68 | 0.17 | 0.20 | 0.82 | 1.07 |
| 16 | cuda | 28.43 | 13.93 | 0.00 | 0.00 | 3.93 | 8.25 | 0.19 | 0.20 | 0.84 | 1.09 |
| 16 | resident | 21.98 | 5.36 | 1.32 | 0.13 | 4.24 | 8.42 | 0.23 | 0.28 | 0.85 | 1.16 |
| 32 | cpu | 165.13 | 134.62 | 0.00 | 0.00 | 8.04 | 18.53 | 0.27 | 0.33 | 1.32 | 2.02 |
| 32 | cuda | 52.72 | 21.26 | 0.00 | 0.00 | 9.54 | 17.70 | 0.27 | 0.49 | 1.40 | 2.06 |
| 32 | resident | 40.68 | 7.47 | 2.18 | 0.14 | 8.93 | 17.81 | 0.35 | 0.44 | 1.34 | 2.03 |
| 64 | cpu | 334.33 | 270.19 | 0.00 | 0.00 | 16.49 | 40.14 | 0.45 | 0.53 | 2.28 | 4.26 |
| 64 | cuda | 127.54 | 36.81 | 0.00 | 0.00 | 17.05 | 65.81 | 0.45 | 0.67 | 2.28 | 4.46 |
| 64 | resident | 106.74 | 11.83 | 3.81 | 0.16 | 17.54 | 65.48 | 0.48 | 0.67 | 2.38 | 4.39 |
| 128 | cpu | 698.67 | 547.80 | 0.00 | 0.00 | 32.81 | 103.09 | 0.73 | 1.21 | 4.63 | 8.40 |
| 128 | cuda | 319.42 | 67.02 | 0.00 | 0.00 | 35.48 | 202.06 | 0.73 | 1.35 | 4.15 | 8.63 |
| 128 | resident | 281.91 | 21.38 | 7.46 | 0.23 | 35.55 | 201.98 | 0.82 | 1.34 | 4.51 | 8.63 |

### Small ship, per-stage cost per cell (attributed, ns per cell per tick)

| N | backend | EOS | bus | trace | combustion | step_tail | stamp | 9b | other | step |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | cpu | 1748 | 0 | 0 | 88 | 218 | 15 | 116 | 71 | 2276 |
| 1 | cuda | 3033 | 0 | 0 | 93 | 1068 | 16 | 125 | 77 | 4436 |
| 1 | resident | 1352 | 188 | 42 | 79 | 843 | 14 | 120 | 66 | 2740 |
| 2 | cpu | 1774 | 0 | 0 | 76 | 214 | 10 | 66 | 46 | 2198 |
| 2 | cuda | 1690 | 0 | 0 | 85 | 581 | 12 | 76 | 56 | 2515 |
| 2 | resident | 772 | 103 | 22 | 73 | 462 | 9 | 73 | 47 | 1582 |
| 4 | cpu | 1750 | 0 | 0 | 113 | 210 | 6 | 42 | 36 | 2166 |
| 4 | cuda | 912 | 0 | 0 | 78 | 365 | 8 | 46 | 40 | 1457 |
| 4 | resident | 417 | 68 | 13 | 78 | 353 | 10 | 48 | 41 | 1043 |
| 8 | cpu | 1749 | 0 | 0 | 112 | 216 | 5 | 29 | 34 | 2151 |
| 8 | cuda | 552 | 0 | 0 | 92 | 270 | 10 | 33 | 36 | 1000 |
| 8 | resident | 230 | 40 | 6 | 90 | 266 | 9 | 31 | 32 | 714 |
| 16 | cpu | 1753 | 0 | 0 | 101 | 226 | 5 | 21 | 28 | 2139 |
| 16 | cuda | 363 | 0 | 0 | 102 | 215 | 5 | 22 | 29 | 740 |
| 16 | resident | 140 | 34 | 3 | 110 | 219 | 7 | 22 | 30 | 572 |
| 32 | cpu | 1753 | 0 | 0 | 105 | 241 | 4 | 17 | 26 | 2150 |
| 32 | cuda | 277 | 0 | 0 | 124 | 231 | 6 | 18 | 27 | 686 |
| 32 | resident | 97 | 28 | 2 | 116 | 232 | 6 | 17 | 26 | 530 |
| 64 | cpu | 1759 | 0 | 0 | 107 | 261 | 3 | 15 | 28 | 2177 |
| 64 | cuda | 240 | 0 | 0 | 111 | 428 | 4 | 15 | 29 | 830 |
| 64 | resident | 77 | 25 | 1 | 114 | 426 | 4 | 16 | 29 | 695 |
| 128 | cpu | 1783 | 0 | 0 | 107 | 336 | 4 | 15 | 27 | 2274 |
| 128 | cuda | 218 | 0 | 0 | 115 | 658 | 4 | 13 | 28 | 1040 |
| 128 | resident | 70 | 24 | 1 | 116 | 657 | 4 | 15 | 28 | 918 |

### Medium ship (100 x 70 per copy), resident only (the cross-check)

| N | cells | ms / tick (mean / p95) | us / cell / tick | env-ticks / s | device MiB (after - before) | CuPy pool MiB | EOS | bus | trace | combustion | step_tail | stamp | 9b | other | digest |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 7000 | 7.93 / 8.36 | 1.133 | 126 | 4 | 2 | 3.32 | 0.51 | 0.10 | 0.63 | 2.60 | 0.08 | 0.39 | 0.19 | `e0d54e71b9d8` |
| 4 | 28000 | 15.99 / 16.21 | 0.571 | 250 | 14 | 7 | 4.94 | 0.95 | 0.12 | 3.18 | 5.34 | 0.23 | 0.79 | 0.52 | `9cf8c48945b6` |
| 16 | 112000 | 48.10 / 49.50 | 0.429 | 333 | 60 | 29 | 9.58 | 2.89 | 0.15 | 13.36 | 18.51 | 0.60 | 1.88 | 1.58 | `5e6189141eba` |

Stage columns are attributed ms per tick. In ns per cell at N = 16: EOS 86, bus 26, trace 1,
combustion 119, step_tail 165, stamp 5, 9b 17, other 14.

### The stacking artifact in `step_tail`, and the batched sweep instead

The per-call radiation-sweep twin issues 3 + K launches per tick. Under the S16 shear transport, K
is the largest wavefront axis over the ordinates, and the table has both x-major and y-major
ordinates (`radiation_sweep.cpp` `S16_SHEAR`, `cuda_radiation_sweep.cu::wavefront_shape`). So
K = max(N·h, w) on a stacked grid:
- 63 launches at N = 1;
- 5,123 launches at N = 128.

That is why `step_tail` per cell climbs from about 230 ns (N ≤ 32) to 657 ns at N = 128 on the GPU
backends while the CPU's stays at 210 to 340 ns. A batched tick does not have this: the resident
sweep core is already (N, h, w), env on the grid's z axis, and K stays max(h, w).

`tools/bench_radiation_sweep.py --cuda --envs` times that batched core directly:
- N copies of the bench's busy scene (12 % solids, a third of them hot, a few bodies);
- `cuda_radiation_sweep_resident` once per tick, launch + sync, best of 3 × 20;
- env 0 and env N−1 checked bit for bit against the CPU sweep of the scene first.

"dev MiB" is the device memory the N-env buffers took.

| N | cells | 40 x 60: core ms | ms / env | ns / cell-update | 40 x 60 + smoke fixture: core ms | ms / env | 70 x 100: core ms | ms / env | ns / cell-update | dev MiB (40 x 60) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2,400 / 7,000 | 0.395 | 0.3951 | 10.29 | 0.354 | 0.3538 | 0.585 | 0.5852 | 5.23 | 4 |
| 2 | 4,800 | 0.401 | 0.2007 | 5.23 | 0.406 | 0.2029 | | | | 0 |
| 4 | 9,600 / 28,000 | 0.517 | 0.1293 | 3.37 | 0.501 | 0.1253 | 0.905 | 0.2262 | 2.02 | 2 |
| 8 | 19,200 | 1.201 | 0.1501 | 3.91 | 1.188 | 0.1485 | | | | 4 |
| 16 | 38,400 / 112,000 | 2.171 | 0.1357 | 3.53 | 2.186 | 0.1366 | 3.995 | 0.2497 | 2.23 | 8 |
| 32 | 76,800 | 4.071 | 0.1272 | 3.31 | 4.116 | 0.1286 | | | | 14 |
| 64 | 153,600 / 448,000 | 7.927 | 0.1239 | 3.23 | 8.013 | 0.1252 | 15.170 | 0.2370 | 2.12 | 32 |
| 128 | 307,200 | 15.624 | 0.1221 | 3.18 | 15.847 | 0.1238 | | | | 72 |
| 256 | 614,400 / 1,792,000 | 31.140 | 0.1216 | 3.17 | 31.617 | 0.1235 | 60.632 | 0.2368 | 2.12 | 140 |
| 512 | 1,228,800 | 62.138 | 0.1214 | 3.16 | 63.072 | 0.1232 | | | | 254 |
| 1024 | 2,457,600 | 124.246 | 0.1213 | 3.16 | 126.107 | 0.1232 | | | | 498 |

A cell-update is one cell in one ordinate, so 16 per cell per tick. The sweep issues 63 launches
per tick at 40 x 60 and 103 at 70 x 100, whatever N. For comparison, the CPU sweep of one env takes
0.388 ms at 40 x 60 (10.1 ns per cell-update), 0.435 ms with smoke, and 1.184 ms at 70 x 100
(10.6 ns).

## Reading

Facts from the tables above; the design draws the conclusions.

**Where the time goes at N = 1 (medium, bots, ms per tick):**

| part | cpu | resident (attributed) |
|---|---|---|
| controllers (outside step) | 1.50 | 1.52 |
| gameplay slots (step − slot 7) | 1.85 | 1.92 |
| EOS stage | 14.81 | 3.72 (device chain + its host pre-stage) |
| trace tail | (inside EOS) | 0.12 |
| bus `from_host` + `to_host` | — | 0.80 |
| combustion (host in both) | 0.99 | 0.60 |
| `step_tail` | 1.57 (all host) | 2.87 (host ripple + per-call fire / temperature / sweep twins) |
| rest of 7 | 0.06 | 0.12 |

- On resident, the two device-resident stages (EOS + trace tail) are 3.8 ms of an 11.5 ms tick
  (33 %).
- The brackets, the bus and the per-call tail are 4.4 ms. Gameplay and controllers are 3.4 ms.
- On resident, `step_tail` costs 1.3 ms MORE than on the CPU: the per-call twins' allocations and
  transfers cost more than the work they move.

**The host stages scale with cells** (stacked, resident, ns per cell per tick, flat from N ≈ 16):

| stage | ns / cell |
|---|---|
| combustion | 110-116 |
| `step_tail` (N ≤ 32, before the stacking artifact) | 219-232 |
| bus | 24-28 |
| 9b over-pressure walls | 15-17 |
| other slots | 26-30 |
| stamp_units | 4-7 |
| **host floor (sum)** | **about 400 ns per cell**; the whole resident tick is 0.51 to 0.58 µs per cell at N = 16 to 32 |

**Where the device's per-cell cost flattens:**
- **The resident EOS stage**, which still includes its host pre-stage, falls from 1,352 ns per
  cell at N = 1 to 97 ns at N = 32, 77 at N = 64 and 70 at N = 128. Its marginal cost from N = 64
  to N = 128 is 62 ns per cell. On the CPU path the EOS is a constant 1,750 ns per cell.
- **The batched sweep core** falls from 10.3 ns per cell-update at N = 1 to 3.4 ns at N = 4 envs
  of 40 x 60 (9,600 cells) and stays between 3.2 and 3.9 ns from there (3.16 at N = 1,024). At
  70 x 100 it is 2.0 to 2.2 ns from N = 4 (28,000 cells). That is 51 and 34 ns per cell per tick:
  3 to 5 times faster than one CPU core's 10 ns per cell-update.
- **The trace tail** is about 0.1 ms at every N: launch-bound.

**Estimated env-ticks per second on one RTX 3070 if the host stages were gone (arithmetic):**

| | small (40 x 60) | medium (70 x 100) |
|---|---|---|
| (a) EOS + trace tail, resident, per env-tick | (21.38 + 0.23) / 128 = 0.169 ms (stacked N = 128) | (9.58 + 0.15) / 16 = 0.608 ms (stacked N = 16, not yet flat) |
| (b) radiation sweep, batched core, per env-tick | 15.62 / 128 = 0.122 ms | 15.17 / 64 = 0.237 ms |
| (a) + (b) | 0.291 ms → **3,440 env-ticks/s** (143× real time summed over envs) | 0.845 ms → **1,180 env-ticks/s** (49×) |
| with an allowance of one more sweep's cost (b) for the stages that have no resident core yet (temperature, fire, combustion, ripple, stamp, the gameplay slots) | 0.413 ms → **2,420 env-ticks/s** (101×) | 1.082 ms → **920 env-ticks/s** (38×) |

- (a) still contains the EOS host pre-stage (the reductions plus four coefficient planes built on
  the host and uploaded every tick). So (a) over-states the device time and the env-ticks/s are a
  floor for those stages.
- The allowance in the last row is an assumption, not a measurement.
- For scale: one CPU core runs the small ship physics-only at 196 env-ticks/s (5.1 ms per tick).
  Today's resident path peaks at 810 env-ticks/s (stacked N = 32), where the host floor and the
  sweep artifact set the slope.

**Device memory is not the constraint:**
- 204 MiB for 128 stacked small ships on resident (79 MiB of it CuPy's pool, the rest allocated
  outside the pool by the C++ cores);
- 498 MiB for 1,024 batched sweeps.

**Determinism:** the tick digest was identical on cpu, cuda and resident, synced and unsynced, in
every configuration (3 match sizes, 8 + 3 stacked sizes); the batched sweep was bit-identical to
the CPU sweep at every N.

**One-line surprises (not investigated):**
- The medium bots match now ends at tick 722 (team 1 wins at 30.1 s) on every backend. On 10-29 it
  was a 1440-tick draw.
- The code moved since #125:
  - medium resident is 3.6× real time (was 3.0×) and per-call 2.35× (was 1.77×, now faster than
    the CPU's 2.00×);
  - large resident all-idle is 13.0 ms (was 21.7 ms);
  - medium slot 9b `find_burst_walls` is 0.39 ms (the 10-29 profile had 3.5 ms).
- On resident, `step_tail` (2.87 ms) costs more than on the CPU (1.57 ms): the per-call twins
  allocate 21 buffers and move about 1 MiB a tick (estimate).
- Combustion runs on the host in every mode (#71) and is the largest host stage per cell
  (about 110 ns, more than four times the bus). Its dispatch also rebuilds the per-tile ignition
  plane in numpy every tick (`physics_runner.py:1158`).
- The stacked grid makes the per-call sweep twin launch-bound (K = N·h wavefronts), so the stacked
  `step_tail` numbers above N = 32 are not what a batched tick would pay.
- `from_host` doubles under the sync (0.24 → 0.47 ms on medium): the pageable `.set()` returns
  before its copy has landed.
- The attributed sum for large resident (14.2 ms) exceeds its plain tick (13.0 ms): the syncs cost
  their own time.

## Not measured, and why

- **Nsight Systems traces** (kernel launches, memcpys and syncs per tick, kernel time per stage).
  `nsys` 2023.4.4 refuses to run without administrator privileges here. So the C++ per-call copies
  are an estimate from the code. `--nvtx` is in both benches for an elevated run.
- **The device-only time of the EOS stage.** Its host pre-stage runs inside
  `run_substeps_resident`, so Python cannot split the two. The design's device-pre-stage question
  (where-we-stand §7 item 3) needs a trace or a C++ timer.
- **The device cost of temperature, fire, combustion and ripple as resident cores.** None exist
  yet; only their per-call twins, which are transfer- and allocation-bound.
- **The EOS substep count `n_sub` per tick.** Python cannot see it, and the resident D2D copies
  scale with it.
- **Bots on small and large** (both ran all idle, as allowed), more than one seed, and medium
  stacked beyond N = 16.
- **The second-CUDA-run-in-one-process slowdown:** every configuration here is a fresh process.
- **Multi-process CPU throughput** (8 cores, several matches at once).
- **The window, the light sweep and the renderer:** headless runs request no light.
- **CUDA graphs (#24) and the launch overhead they would remove.**

## P1a: the farm measured (2026-11-07)

The design's §9.1 protocol, run on the `cpu_farm` backend P1a built (`src/rl_env.py`), with its
random-action driver `tools/rl_smoke.py`. This is the yardstick every later farm or device reading
is compared with: same driver, same run files, same decimations.

### Method

- **Machine:** the Home Desktop above (Ryzen 7 3700X, 8 cores / 16 threads, 16 GB). Nothing else
  ran during the measurement (17:14 to 17:50).
- **Code:** branch `29-p1a-batched-env` at `85f4cc57`, its own CPU build (`cpp/build_cpu_home.bat`
  -> `cpp/build/Release`). Farm workers load only that build and start with one BLAS thread
  (`rl_env.WORKER_ENV`; see "Memory" below).
- **The farm:** `BatchedEnv` with N worker processes (spawn), ONE env per worker, plus the driver
  process (the trainer stub: it draws the random action rows, `control_policy.random_action_rows`,
  seed 0). Every marine team on the `policy` controller, the horde on `zombies`.
- **Placeholder rows ON:** an observation of all 11 crop planes (the 7 frozen channels,
  `material`, `units_own` / `units_enemy` / `units_zombie`) at `crop_radius = 8` (17 x 17 tiles) for
  every row, the 6 vector fields, 8 reward weights, the per-unit counters, and the checkpoint
  digests every 60 ticks.
- **The random rows (placeholder domain):** per decimation step each unit draws a uniform
  direction for move, aim and throw; the move is cleared with p = 0.25, the trigger held with
  p = 0.5, throw and use issued with p = 0.05 each; fuse uniform in [0.5, 3] s, reach in [0, 16]
  tiles (the sim clamps it).
- **Levels:** the small ship `tools/gen_ship.py --seed 1 --size small --vents --out
  <scratch>/levels/ship_s1_small` (60 x 40, outside `levels/`) and `levels/ship_demo` (medium,
  100 x 70); both 4 v 4 + 4 zombies.
- **Configurations:** decimation 1 and 4; 7 and 8 workers (plus 1, for scale); each in a fresh
  process, serially; every env runs until it has finished at least 3 episodes through the in-step
  reset; the first 20 steps are warm-up and not timed (worker start, imports, first match).
- **Read:** env-ticks/s = ticks simulated by all envs / wall time of the timed window, the
  driver's action generation included; step ms = `BatchedEnv.step` wall time per call (one
  decimation step for all envs); memory per worker at the end (psutil: the working set, its peak,
  the private bytes).

Commands (one per configuration; `measure.sh` loops them):

```
C:/Users/steen/anaconda3/envs/data/python.exe tools/rl_smoke.py <scratch>/run_<level>_d<D>_n<N>.toml \
    --workers <N> --episodes 3 --seed 0 --warmup-steps 20
```

The run files (`run_small_*` add `levels_dir = "<scratch>/levels"` and name `ship_s1_small`;
`n_envs` and `decimation` vary):

```toml
# #29 P1a farm measurement (design 9.1): placeholder obs + reward rows ON.
[run]
seed = 2026
n_envs = 8
decimation = 4
levels = ["ship_demo"]
policy = "policy"

[obs]
channels = ["pressure", "smoke", "water_depth", "temperature", "fire", "o2", "solid", "material", "units_own", "units_enemy", "units_zombie"]
crop_radius = 8
vector = ["hp", "team", "alive", "facing", "x", "y"]

[reward]
kills = 1.0
team_kills = -1.0
damage_dealt = 0.01
damage_taken = -0.01
deaths_bullet = -1.0
deaths_explosion = -1.0
deaths_melee = -1.0
deaths_heat = -1.0
```

### Small ship (60 x 40)

| decimation | workers | env-ticks/s | per env | step ms p50 / p95 / max | RSS / peak / private MB per worker | timed ticks | episodes per env |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 176.1 | 176.1 | 5.7 / 7.3 / 17.3 | 44 / 45 / 32 | 2 552 | 3 |
| 1 | 7 | **754.6** | 107.8 | 8.4 / 11.7 / 20.3 | 45 / 46 / 32 | 38 129 | 3 to 14 |
| 1 | 8 | 726.7 | 90.8 | 10.7 / 12.5 / 21.9 | 45 / 46 / 32 | 43 480 | 3 to 14 |
| 4 | 1 | 214.2 | 214.2 | 14.1 / 26.7 / 34.9 | 44 / 45 / 32 | 7 362 | 3 |
| 4 | 7 | 873.8 | 124.8 | 31.2 / 37.1 / 48.1 | 45 / 46 / 32 | 92 704 | 3 to 18 |
| 4 | 8 | **951.2** | 118.9 | 33.7 / 40.7 / 48.6 | 45 / 46 / 32 | 60 383 | 3 to 18 |

### Medium ship (ship_demo, 100 x 70)

| decimation | workers | env-ticks/s | per env | step ms p50 / p95 / max | RSS / peak / private MB per worker | timed ticks | episodes per env |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 52.6 | 52.6 | 18.9 / 20.1 / 33.6 | 46 / 51 / 34 | 14 439 | 3 |
| 1 | 7 | **277.7** | 39.7 | 24.1 / 30.9 / 48.9 | 49 / 50 / 37 | 87 500 | 3 to 26 |
| 1 | 8 | 277.1 | 34.6 | 28.5 / 33.5 / 50.1 | 49 / 51 / 36 | 116 088 | 3 to 8 |
| 4 | 1 | 78.3 | 78.3 | 35.6 / 80.3 / 87.8 | 46 / 51 / 34 | 7 384 | 3 |
| 4 | 7 | 318.0 | 45.4 | 88.0 / 100.4 / 117.5 | 48 / 50 / 36 | 53 473 | 3 to 12 |
| 4 | 8 | **336.5** | 42.1 | 94.4 / 107.7 / 125.7 | 49 / 51 / 37 | 172 094 | 3 to 13 |

The two single-worker rows at decimation 4 timed only ~7 400 ticks (three short episodes); read
them as ±15 %.

### Where one env's step goes (in-process, the same `EnvRunner` a worker runs, 3 000 ticks)

| level, decimation | ticks/s | match tick (controllers + `step`) | observation | checkpoint digests | counters, reset, rows |
|---|---|---|---|---|---|
| small, 1 | 175.8 | 5.23 ms (91.9 %) | 0.22 ms (3.8 %) | 0.05 ms (0.8 %) | 0.19 ms (3.3 %) |
| small, 4 | 210.4 | 4.60 ms (96.8 %) | 0.05 ms (1.1 %) | 0.04 ms (0.9 %) | 0.05 ms (1.1 %) |
| medium, 1 | 51.1 | 19.03 ms (97.2 %) | 0.28 ms (1.4 %) | 0.08 ms (0.4 %) | 0.20 ms (1.0 %) |
| medium, 4 | 56.4 | 17.51 ms (98.8 %) | 0.07 ms (0.4 %) | 0.08 ms (0.5 %) | 0.06 ms (0.3 %) |

(ms per simulated tick. The farm's own work is 1 to 8 % of a step; the rest is the game.)

### Reading

- **The yardstick:** the 8-core farm runs **755 to 951 env-ticks/s on the small ship** and
  **278 to 337 on the medium ship** with the placeholder observation and reward on. Decimation 4
  is the better case on both (and the design's planning decimation, §2.4).
- **Against the design's §2.3 estimate** (1 000 to 1 450 for the small-ship match, "P1 MEASURES
  it"): the farm reaches 0.75 to 0.95 of the low end. The estimate assumed 7 to 8 workers at an
  all-core factor of 0.9 and SMT 1.0 to 1.3, i.e. 6.3 to 9.4 single-core equivalents; measured, 7
  workers give **4.3** (small, d1) and **5.3** (medium, d1) single-core equivalents, 8 workers
  **4.4** (small, d4) and about **4.3** (medium, d4, against the short single-worker row). So
  per-core efficiency under load is 0.54 to 0.75, not 0.9; with this, the §2.4 farm rows for this
  PC stretch accordingly, and the device-to-farm ratio here (§2.3, 1.4 to 3.4x) rises by the same
  factor (1.05 to 1.3x).
- **The 8th worker:** at decimation 1 it buys nothing (small 755 -> 727, medium 278 -> 277): the
  driver steps every tick, and 8 workers plus the driver oversubscribe the 8 physical cores. At
  decimation 4 the driver is idle 3 ticks in 4 and the 8th worker adds 6 to 9 %. On a Berzelius
  slot (16 cores) the same rule applies: N_workers = cores - 1 at decimation 1.
- **Decimation 4 is faster per tick for two reasons:** the farm's per-step work (observation,
  driver, IPC) is paid once per 4 ticks, and the random rows throw and toggle doors 4x less often
  per tick, which makes the ticks themselves cheaper (small 5.23 -> 4.60 ms in-process). The
  second is a property of the random driver, not of the farm; a trained policy's tick cost is its
  own.
- **Step-time tails are the game's:** p95 / p50 is 1.1 to 1.4 at 7 to 8 workers. The step waits
  for the slowest worker, and a worker's tick cost follows its match (blasts, fires, many shots).
- **Memory:** a farm worker holds ~45 to 50 MB working set and ~32 to 37 MB private bytes. Before
  `WORKER_ENV`, OpenBLAS committed a buffer per hardware thread in every worker: ~1.9 GB of private
  bytes each on this 16-thread machine (8 workers ~ the machine's 16 GB of commit), resident
  ~45 MB. Measured on the first pass of this protocol and fixed before the numbers above (digests
  unchanged; G-a and G-b re-run).
- **Episodes:** random rows end most episodes in 100 to 200 ticks (team-kills and grenades in the
  spawn rooms), with the odd 7 200-tick draw at the time limit; so the measurement spends a few
  percent of its steps on resets, which cost ~25 ms each (a fresh `MatchSession`).

### Not measured

- 14 / 16 workers on a node (Berzelius); the protocol row is ready (`--workers 14` / `16`).
- A real trainer in the driver process: the stub only draws random rows; a policy forward pass
  shares the driver's core (at decimation 1 that core is already the limit).
- The large ship.

## P2: the CUDA second-run slowdown, found (2026-11-07)

**Cause: the NVIDIA driver's power policy, not our process.** On this box (RTX 3070, Windows,
driver 13030) a CUDA context runs at **P2, 1830 MHz SM / 6801 MHz memory for its first ~7 s of
load**; then, the GPU being ~30-40 % utilized by the per-call and resident mix, the driver holds it
at **P3, ~1150-1320 MHz SM / 5001 MHz memory**, with dips to P5 (~1100 / 810 MHz). The step lands at
the start of the second match of a `--repeat` run, which is why it looked like a second-run effect.
Every GPU-touching stage slows (kernels at the lower clocks, transfers and syncs with them); the
CPU-only stages (movement, `stamp_units`, 9b) do not move.

**Ruled out, measured** (`ship_demo`, seed 1, bots, `--sim-seconds 20`, resident, every backend
incl. combustion):

| check | result |
|---|---|
| fresh processes back to back (3 x 1 match) | step 10.91 / 10.90 / 10.85 ms, GPU 41 °C, 1830 MHz each time: not thermal |
| the previous match's `GameMap` after the next starts (weakref + `gc.get_objects`) | 0 alive: nothing retained |
| device memory used (`cudaMemGetInfo`) / CuPy pool, after each match | 1083.4 MiB / 2.0 MiB, flat: no device leak |
| freeing the CuPy pool between matches (`free_all_blocks`) | no effect |
| a fixed 196 KB H2D / D2H in a tight loop, and the second map's own `gas` mirror | 41-43 / 67-70 µs in every run (the driver re-boosts under a tight loop) |
| the C++ persistent scratch (`g_eos_res` etc.) | keyed (h, w), allocated once; the same dims every match |
| one long match in one process (720 ticks) | no ramp before its blasts: tick count alone does not trigger it |
| **15 s idle between matches, same process** | **the GPU falls to P8, then comes back at P5 -> P3, never P2**: the second and third match stay slow (16.5 / 16.3 ms) |
| GPU clock sampled every 0.5 s across 3 matches (`nvidia-smi -lms 500`) | P2 1830 / 6801 for ~7 s, then P3 ~1200 / 5001 for the rest of the process |

**The trend over 5 repeats** (`tools/bench_bot_matches.py --level ship_demo --seeds 1 --timing
--sim-seconds 20 --repeat 5 --cuda --resident --profile`), step mean per match:
**10.85, 12.68, 13.18, 13.22, 13.95 ms** (+17 % at match 2, +29 % at match 5). The growth is the
P3 clock dithering (1155-1320 MHz) and its P5 dips, not an accumulation: device memory, the pool
and the object count stay flat, and the per-stage growth from match 2 on is in the GPU stages only
(EOS 3.51 -> 5.82 ms, combustion 2.02 -> 2.53, step_tail 2.60 -> 2.73, D2H 0.32 -> 0.45).

**What it means.**
- A training process lives in the P3 regime: plan this mix at **~13-15 ms per medium tick, not
  10.9**. Every short fresh-process benchmark in this doc (§1, §2) measured the boosted window and
  is optimistic by 15-30 % for a long run on this box. A GPU benchmark that claims a steady state
  warms up for more than 10 s in-process first, or samples `nvidia-smi` beside it.
- Nothing in our code to fix. The lever is the driver: NVIDIA Control Panel -> Manage 3D settings
  -> Program settings -> `python.exe` -> Power management mode = **Prefer maximum performance**
  (Erik's call; not tried here), or `nvidia-smi -lgc` (administrator, not tried). A batched resident
  engine (P5 on) keeps the GPU busy enough that the light-load heuristic should not engage; the
  device farm's measurement should still sample the clocks. A data-center GPU on Linux
  (Berzelius) runs its application clocks under load; its farm run samples them too.
- Determinism is unaffected (every run above: digest `1af8c952fa72`).

**A cost P2 adds, measured:** with the combustion twin enabled (`tools/run_on_cuda.py`, so
`--cuda` and `--resident`), slot 7's combustion costs **2.0 ms per medium tick on the per-call GPU
twin against 0.6 ms on the host** (fresh process, P2 clocks; the twin allocates ~22 device buffers,
uploads ~15 and downloads ~8 per call). The resident core `combustion_launch_resident` is already
the launch body; P4a wires it to the persistent fields and removes the difference.

## P2: the resident tick under Nsight (2026-11-07)

Nsight capture pending when this section was written (since taken and READ: see "P2 follow-up: the Nsight capture, read" below; Erik runs C:\tmp\breach_nsys_capture.ps1 elevated; then
`tools/nsys_report.py C:/tmp/breach_resident_medium_stats`). The reader is in place
(`tools/nsys_report.py`, tested on hand-made tables, `tests/test_nsys_report.py`): per captured
tick, kernel launches, copies by direction (count, bytes, time), `cudaMalloc` / `cudaFree`,
explicit device syncs and the top kernels, from the `nsys stats --format csv` tables the script
writes. Its table replaces the code estimate in §1 "The bus per tick". The capture runs 15 s of
sim, so it sits mostly in the P3 regime above; read its kernel times with that in mind.

## P4a: the brackets gone (2026-11-07)

**What moved.** The resident tick's remaining host stages run on the device (#29 P4a, branch
`29-p4a-resident-brackets`): the water tail (W5 + W3 + the seal), combustion (P2's
`combustion_launch_resident`), fire (its P1 early exit a per-env device gate), the radiation
sweep (`radiation_sweep_launch_resident`) and temperature (the new `temperature_launch_resident`),
through the drivers in `cpp/src/cuda_resident_tick.cu`, on persistent scratch, with the static
tables uploaded only when their host value changes. Slot 7 is now ONE `GameMap.from_host` at
entry, the device chain, ONE `GameMap.to_host` at exit, and the host ripple (skipped headless).
The EOS stage's gas- and permeability-dependent pre-stage pieces (the Dalton + K|grad P|dt/N scan
and the coeffE / coeffS cache) run on the device too, so a device water tail before it needs no
mirror refresh.

**Method.** As §1 / P2: `tools/bench_bot_matches.py --level ship_demo --seeds 1 --timing
--sim-seconds 15 --cuda --cuda-build cpp/build_cuda --resident --profile`, one fresh process per
line (15 s of sim, so both runs sit in the driver's boosted P2 window -- see "the second-run
slowdown"), the same length before and after. Before = main `b560ea38` (this worktree's build at
the branch point); after = the branch head's build. A second plain after-run reproduced the step
mean to 0.01 ms. A `--profile-sync` after-run gives the attribution. The digest is the same on
both (`59719adc8953`, the 15 s match), and the full-length pinned match on `--cuda --resident`
reproduces the CPU pin of `tests/test_zombie_control.py` (win team 1 at tick 722, field
`da05242a…`, tick `f35ed2be…`).

**Per stage, ms per tick, mean / p95 (plain):**

| stage | before (main `b560ea38`) | after (P4a) |
|---|---|---|
| total (step + controllers) | 12.80 / 13.62 | 9.26 / 9.96 |
| step | 10.968 / 11.670 | 7.473 / 7.744 |
| 7 physics | 9.006 / 9.607 | 5.626 / 5.811 |
| &nbsp;&nbsp;7 H2D `GameMap.from_host` | 0.241 / 0.299 | 0.294 / 0.354 |
| &nbsp;&nbsp;7 EOS `run_substeps_resident` | 3.521 / 3.621 | 3.683 / 3.765 |
| &nbsp;&nbsp;7 trace tail (resident) | 0.111 / 0.123 | 0.105 / 0.118 |
| &nbsp;&nbsp;7 combustion | 2.035 / 2.329 (per-call GPU twin) | 0.029 / 0.035 (resident, launches only) |
| &nbsp;&nbsp;7 sky exchange (host) | 0.001 / 0.002 | — (a host decision only; no work on a space map) |
| &nbsp;&nbsp;7 `step_tail` (ripple + fire + temp + sweep, per-call twins) | 2.649 / 2.754 | — |
| &nbsp;&nbsp;7 tail (resident: fire + sweep + temp + the books D2H) | — | 0.811 / 0.833 |
| &nbsp;&nbsp;7 D2H `GameMap.to_host` | 0.326 / 0.381 | 0.585 / 0.648 |
| &nbsp;&nbsp;rest of 7 (host pre-physics) | 0.123 / 0.156 | 0.119 / 0.146 |
| real-time x | 3.26 | 4.50 |

- **Slot 7 fell 3.38 ms (−38 %), the step 3.50 ms (−32 %).** Combustion's per-call twin (2.0 ms,
  P2's measured cost) is gone, and the per-call fire / temperature / sweep twins' allocations and
  transfers with it: the tail costs 0.81 ms where `step_tail` cost 2.65. The estimate in the brief
  was "roughly −1.4 ms" against the host combustion; against the per-call GPU twin that `--cuda`
  ran since P2 it is −2.0 ms for combustion alone.
- **Attributed** (`--profile-sync`, after): H2D 0.599 (the pageable `.set()` returns before its
  copy lands, as §1 found), EOS 3.425, trace tail 0.108, combustion 0.084, tail 0.780 (of it the
  books D2H's wait), D2H 0.585. The device work of combustion + fire + sweep + temperature is
  under 0.9 ms a tick on the medium ship.
- The EOS stage grew ~0.16 ms (3.52 → 3.68 plain): the device pre-stage scan adds one 8-byte D2H
  (a sync) before the substep loop; its host scan and the two coefficient-plane uploads are gone.
- The D2H grew 0.26 ms: the exit download now carries `dem_acc` (328 KiB -- 12 claimant slots of
  int32 per cell, the largest plane on the bus), `fire`, `wall_hp`, `heat` and `rad_flux_sweep`,
  which the host stages used to write on the mirror.

**The bus per tick, after (measured, Python-visible, `tests/cuda_resident_bus_check.py`):**

| | calls | fields | bytes |
|---|---|---|---|
| entry `GameMap.from_host` | 1 | 17 (atmosphere, wind_x / y, temperature, gas, gas_energy, heat, fire, wall_hp, solid, is_vacuum, is_ambient, thermal_solid, flammable, material, dyn_permeability, dyn_heat_atten_q) | 533.2 KiB |
| between them | **0** | | |
| exit `GameMap.to_host` | 1 | 12 (atmosphere, wave_p, wind_x / y, temperature, gas, gas_energy, heat, fire, wall_hp, dem_acc, rad_flux_sweep) | 847.7 KiB |

Before: 11 H2D fields (410.2 KiB) and 7 D2H fields (382.8 KiB) visible to Python, plus, inside
the per-call twins, ~42 H2D / ~23 D2H copies (~1.9 MiB, the §1 estimate) and ~21 + ~22
`cudaMalloc` / `cudaFree` pairs a tick (temperature, sweep, fire, combustion). After, inside C++:
the EOS's per-tick `absorb_q` / `cap2` / `cons_flag` uploads (its `coeffE` / `coeffS` uploads are
gone), the 8-byte n_sub scan result, the EOS and trace-tail counter blocks, and ONE ~1 KiB books
arena (every tail stage's counters + the head of the destroyed-wall list); zero `cudaMalloc` in
steady state. Conditional, not on this match: the water planes (+ the W3-sealed
`dyn_permeability`) on a tick the water stage runs, the three render-only sweep planes when not
headless, the light planes and the light inputs while light is requested, the table-derived static
planes when `GameMap._tile_serial` moved, and the sky bracket's `gas` pair on an ambient map with
the sky exchange live.

**Not measured:** the P3-regime (long-run) numbers -- both runs are 15 s fresh processes; the
large ship; a wet or ambient match. The Nsight capture was still pending (read since: "P2 follow-up" below).

## P2 follow-up: the Nsight capture, read (2026-11-07, with P4b)

Erik's capture (`C:\tmp\breach_resident_medium.nsys-rep`, taken after P4a merged: `bench_bot_matches
--level ship_demo` medium, `--cuda --resident --nvtx`, 90 captured ticks, 15 s of sim so the P2 boosted
window), read with `tools/nsys_report.py C:/tmp/breach_resident_medium_stats --top 20` (the tool ran on
the file as it stands; no fix needed). Per captured tick (window total / 90):

| quantity | per tick |
|---|---|
| kernel launches (GPU side) | 537.9 |
| kernel time (GPU) | 2090 us |
| `cudaLaunchKernel` API time | 3.0 ms (537.9 calls, mean 5.6 us) |
| H2D copies | 18.7 x 560 KB (86 us GPU) |
| D2H copies | 16.9 x 773 KB (130 us GPU) |
| D2D copies | 67.1 x 2104 KB (75 us GPU) |
| memsets | 34.0 x 1604 KB (38 us GPU) |
| copy / memset API calls (all four) | 136.7 (2.28 ms of host API time) |
| explicit device syncs | 12.4 (0.63 ms; `cudaDeviceSynchronize` mean 345 us) |
| `cudaMalloc` / `cudaFree` | 0 / 0 (the zero-malloc steady state holds) |

Top kernels by GPU time (share of the 2.09 ms): `mg_fused_tail` 565 us (27.0 %, 7.2 launches),
`mg_smooth_color` 296 us (14.2 %, **114.9 launches**), `rs_wavefront` (the radiation sweep) 285 us (13.6 %,
88 launches), `trace_apply` 228 us (10.9 %, 35.8 launches), `bulk_e_apply_v2` 66 us (3.2 %),
`energy_flux_pass_b` 58 us (2.8 %), `trace_wipe` 53 us (2.5 %, 35.8 launches), `energy_flux_pass_a` 45 us (2.2 %),
`mg_residual` 36 us, `bulk_e_recover_v2` 34 us, `bulk_nb_accum` 33 us (28.6 launches), `bulk_flux_dq` 32 us.
Host NVTX wall (the capture's own overhead inflates it ~40 % over an unprofiled run: `step` 11.0 ms against
7.9 ms in section "P4a"): `7 physics` 8.41 ms, of it EOS 4.86, tail 1.03, D2H 0.81, H2D 0.65,
combustion 0.35, trace tail 0.16; `6 stamp_units` 0.53 (host C++), `9b` 0.41.

**Reading.**
1. **The resident tick is launch- and sync-bound, not kernel-bound.** The GPU does 2.09 ms of kernel work in a
   step that takes ~8 ms of host time (11 ms under capture): ~20 % busy at N = 1. Its 538 launches cost 3.0 ms of
   launch API time alone (more than the kernels they start), the 137 copy / memset API calls another 2.3 ms, the
   12 explicit syncs 0.6 ms; a mean kernel is 3.9 us of GPU time.
2. **The pressure solve is ~45 % of the GPU time and most of the launches** (`mg_fused_tail` + `mg_smooth_color`
   + residual / restrict / prolong): the smoother alone is 115 launches a tick. The sweep is 13.6 % in 88 launches,
   the trace planes 13.4 % in 72 launches (a launch per plane per substep).
3. **Copies are launch-count, not bandwidth:** the 101 D2D + memset calls a tick move 3.7 MB at ~1 us of GPU time
   each but cost ~1 us-class host API time each; the H2D / D2H pair is 560 / 773 KB a tick at ~100 us of GPU time
   each.
4. **What P5 and P9 should target.** P5 (N envs in one launch) is the direct fix for point 1: the launch, sync and
   copy-call overhead is per tick, not per env, so it amortizes across N and the kernel time is what is left (the
   2.09 ms here is the N = 1 floor of useful work). P9 attacks the per-tick fixed cost itself: fuse or graph the
   multigrid smoother's 115 launches and the per-plane trace launches (72), take the 12 syncs and the ~100
   scratch D2D / memset calls out of the substep loop (device-side scratch zeroing inside kernels), and move the
   `dem_acc` plane (328 KB D2H a tick, read by nothing but the digest) off the per-tick bus. Both are measurable
   against this table: launches per tick, API ms per tick, syncs per tick.
5. **Caveat:** 15 s captures sit in the driver's boosted P2 window (the P2 finding): the kernel times here are
   the optimistic end of a long run's.

## P4b: the stamp kernel (2026-11-07)

**What moved.** Slot 6's five stamp planes (`obstacles`, `dyn_permeability`, `dyn_wave_absorb`,
`dyn_heat_atten_q`, `dyn_light_atten_q`) are built on the device in the resident tick, by
`stamp_launch_resident` (`cpp/src/cuda_stamp.cu`, N-shaped, 3 launches: reset + the MAX stamps + the
last-row-wins permeability write), from a per-tick upload of the unit rows the host stamp already flattened
(`GameMap.take_stamp_rows`: one `(R, 9)` int32 matrix, 108 rows = 3.8 KiB a tick on this match, staged through
pinned memory so the copy is asynchronous). The CPU stamp stays THE oracle and still runs (gameplay reads its
planes); the device planes are a second copy built from the same rows, so the entry `from_host` no longer carries
`dyn_permeability` / `dyn_heat_atten_q` (and not `dyn_light_atten_q` while light is requested). The per-tile refill
(`_neighbor_mean`'s float division, atmosphere) stays on the host: its output rides the atmosphere upload that
slot 6b's FieldEdits keep anyway, and a device refill would run after that flush rather than before it.

**Method.** As P4a: `tools/bench_bot_matches.py --level ship_demo --seeds 1 --timing --sim-seconds 15 --cuda
--cuda-build cpp/build_cuda --resident --profile` (plus a `--profile-sync` run), one fresh process per line, the
same length before and after, two plain runs each. Before = this worktree's build at main `c833fb89`; after = the
branch head's. The digest is `59719adc8953` on all six runs.

| stage, ms per tick, mean (plain, run 1 / run 2) | before (`c833fb89`) | after (P4b) |
|---|---|---|
| total (step + controllers) | 9.78 / 9.89 | 9.70 / 9.73 |
| step | 7.926 / 8.057 | 7.863 / 7.890 |
| 6 stamp_units (host C++, the oracle) | 0.504 / 0.497 | 0.505 / 0.505 |
| 7 physics | 6.036 / 6.202 | 5.967 / 6.005 |
| &nbsp;&nbsp;7 H2D `GameMap.from_host` | 0.298 / 0.291 | 0.276 / 0.286 |
| &nbsp;&nbsp;7 stamp (resident) | -- | 0.047 / 0.050 |
| &nbsp;&nbsp;7 EOS `run_substeps_resident` | 3.983 / 4.135 | 3.915 / 3.930 |
| &nbsp;&nbsp;7 tail (resident) | 0.843 / 0.874 | 0.832 / 0.834 |
| &nbsp;&nbsp;7 D2H `GameMap.to_host` | 0.629 / 0.623 | 0.612 / 0.615 |
| real-time x | 4.26 / 4.21 | 4.30 / 4.28 |

Attributed (`--profile-sync`): the stamp stage 0.118 ms (3 launches + the rows copy); H2D 0.722 -> 0.588; EOS
3.747 -> 3.596.

**Bus per tick:** entry `from_host` 15 fields, **478.5 KiB** (was 17 fields, 533.2 KiB: the two 27.3 KiB stamp
planes left) plus the 3.8 KiB of rows inside C++; exit `to_host` unchanged, 12 fields, 847.7 KiB. The device stamp
ran on 360 / 360 ticks of the match (0 fallbacks).

**Reading.** A small win, as predicted: the step is ~0.1 ms (1.3 %) faster and real time 4.23x -> 4.29x, within
two runs' spread but in the same direction on every pair. The two planes are 10 % of the entry bytes and ~20 us of
PCIe time; the host stamp (0.5 ms) is untouched because it is still the oracle and gameplay still reads its
planes. A first version copied the rows with a pageable `cudaMemcpy` and the stamp stage cost 0.31 ms of host time
-- it waited for the entry upload still in flight (the wait moved out of the EOS stage, 3.98 -> 3.65, for no net
gain); the pinned asynchronous copy brought the stage to 0.05 ms. The patch's value is architectural: the device
no longer needs the host's stamp planes, which is the shape P5 (N envs) and the later patches that drop the host
mirror need.

## P5: the batched physics (2026-11-08)

**What moved.** Every slot-7 launch core runs over an `(N, h, w)` batch of homogeneous envs (`blockIdx.y` = the
env, pointers rebased per env, the kernel bodies unchanged): the water substeps and tail, the EOS substep chain, the
multigrid build and V-cycle (the fused coarse tail one block per env), kick, store, energy flux, joining the trace
tail, combustion, fire, the sweep, temperature and the stamp. The EOS pre-stage moved to the device (per-env
reductions; one thread per env folds n_sub and picks its row of a fold table built once per config; cap2 / absorb /
coeffE / coeffS built on the device). Both substep loops run to the batch's max n_sub with per-env masks. Every book
lands per env in one arena, read back in ONE D2H at the tail. One blocking tick stream. Host syncs per tick: the
8-byte max(n_sub) read and the books read (was: the 8-byte max_du read, five EOS book D2Hs with a device sync, the
trace tail's sync + D2H, the books D2H). The driver for N > 1 is `simulation/resident_batch.py`
(`ResidentBatch`); the played tick is the same driver at N = 1.

**N = 1, the medium bots match (ship_demo, seed 1).** Method as P4b: `tools/bench_bot_matches.py --level ship_demo
--seeds 1 --timing --sim-seconds 15 --cuda --cuda-build cpp/build_cuda --resident --profile`, one fresh process per
line, two plain runs each, interleaved before / after. Before = main `a3812a43` (a detached checkout with its own
CUDA build); after = the P5 branch. The digest is `59719adc8953` on all four runs.

| stage, ms per tick, mean (plain, run 1 / run 2) | before (`a3812a43`) | after (P5) |
|---|---|---|
| total (step + controllers) | 9.69 / 9.76 | 9.02 / 9.18 |
| step | 7.896 / 7.935 | 7.226 / 7.380 |
| 7 physics | 6.058 / 6.102 | 5.379 / 5.540 |
| &nbsp;&nbsp;7 H2D `GameMap.from_host` | 0.260 / 0.276 | 0.263 / 0.264 |
| &nbsp;&nbsp;7 EOS `run_substeps_resident` | 3.994 / 4.021 | 1.805 / 1.865 |
| &nbsp;&nbsp;7 trace tail (resident) | 0.115 / 0.116 | 0.031 / 0.031 |
| &nbsp;&nbsp;7 tail (resident, ends with the books D2H) | 0.838 / 0.856 | 2.436 / 2.516 |
| &nbsp;&nbsp;7 D2H `GameMap.to_host` | 0.611 / 0.620 | 0.589 / 0.610 |
| &nbsp;&nbsp;7 applier (P6, reserved) | -- | 0.000 / 0.000 |
| real-time x | 4.30 / 4.27 | 4.62 / 4.54 |

**Reading (N = 1).** Faster, not slower: the step is 0.6 ms (7.5 %) shorter and real time 4.28x -> 4.58x. The
EOS stage no longer blocks the host (its five book D2Hs and the device sync are gone, its only wait is the 8-byte
max(n_sub)), the trace tail no longer syncs, the per-tick H2D of cap2 and the absorb plane is gone, so the host
waits once, at the tail's books read -- which is why the tail's line grew by what the EOS line lost. The bus is
unchanged (478.5 KiB in, 847.7 KiB out).

**The batched physics, N envs.** `tools/bench_resident_scaling.py --batch-sweep --level <L> --levels-dir
<scratch>/levels --ns 1,2,8,16,32,64 --warmup 5 --ticks 60`, one fresh process per N: N envs of the level (seeds
1..N), `Simulation` under `ContinuousRealtime`, NO units, the level's entities ON (vents, ducts, doors), headless,
in one `ResidentBatch`. Small = `ship_s1_small` (60 x 40, 2 400 cells, `tools/gen_ship.py --seed 1 --size small
--vents`), medium = `ship_demo` (100 x 70, 7 000 cells). Every env ran n_sub = 8 (the cap) on these levels. "tick"
is one batched tick of all N envs; "slot 7" is the `run_resident_batch` call (the device chain PLUS every env's
entry / exit transfers); "host" is everything else per env (slots 1..6b and 9..10). The device-chain line is a
second, attributed run (`p5_split.py`-style timers on the same calls, 30 ticks): slot 7 minus the per-env
`from_host` / `to_host` and the host pre-physics.

| small (2 400 cells) | N = 1 | 2 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| batched tick, ms | 4.84 | 6.55 | 15.64 | 28.72 | 54.61 | 107.56 |
| env-ticks / s (whole tick) | 207 | 306 | 512 | 557 | 586 | 595 |
| slot 7, ns per cell per tick | 1 833 | 1 186 | 647 | 578 | 541 | 521 |
| host per env (slots 1..6b, 9..10), ms | 0.442 | 0.426 | 0.402 | 0.407 | 0.408 | 0.429 |
| per-env bus (from_host + to_host), ms (attributed) | 0.71 | -- | 0.84 | 0.92 | -- | 0.97 |
| device chain per env, ms (attributed) | 3.68 | -- | 0.69 | 0.44 | -- | 0.25 |
| device chain, ns per cell per tick | 1 533 | -- | 288 | 183 | -- | 104 |

| medium (7 000 cells) | N = 1 | 2 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| batched tick, ms | 5.80 | 7.62 | 20.36 | 38.01 | 74.13 | 143.78 |
| env-ticks / s (whole tick) | 172 | 262 | 393 | 421 | 432 | 445 |
| slot 7, ns per cell per tick | 747 | 464 | 282 | 261 | 250 | 242 |
| host per env (slots 1..6b, 9..10), ms | 0.575 | 0.566 | 0.572 | 0.552 | 0.570 | 0.550 |
| per-env bus (from_host + to_host), ms (attributed) | 0.86 | -- | 1.05 | -- | -- | 1.12 |
| device chain per env, ms (attributed) | 4.34 | -- | 0.98 | -- | -- | 0.54 |
| device chain, ns per cell per tick | 620 | -- | 140 | -- | -- | 77 |

Memory (`cudaMemGetInfo` used, after minus before the sims were built): small 6 / 22 / 166 MiB at N = 1 / 8 / 64
(~2.5 MiB per env), medium 10 / 62 / 456 MiB (~7 MiB per env); N = 64 medium uses 1.5 GiB of the 8 GB card in
total -- N is bounded here by time, not memory.

**Reading (N envs).** The device chain amortises as designed: from 4.3 ms per env at N = 1 to 0.54 ms (medium) /
0.25 ms (small) at N = 64 -- 77 / 104 ns per cell per tick, i.e. ~1 850 (medium) / ~4 000 (small) env-ticks/s of
device physics alone, the launch- and sync-bound regime of P4b's Nsight read paid once per batch instead of once
per env. The whole batched tick saturates near 600 (small) / 450 (medium) env-ticks/s because the per-env host
work stays per env: the gameplay slots (0.4-0.6 ms) and, larger, the per-env pageable CuPy transfers of the entry /
exit sets (~0.9-1.1 ms per env: ~27 `set` / `get` calls of 7-336 KiB each). At P5 that bus is the honest price of
host gameplay in the loop -- it is exactly what G1 must weigh (design §3: "the device side's host 9e cost reported
as its own line ... plus the bus"); it leaves with the authority flip (P6+), when the host stops reading the
fields every tick. Against P1a's farm (8 workers: ~755 small / ~278 medium env-ticks/s with units and controllers
on, a different scene), the batch's whole-tick numbers are not the G1 reading -- G1 runs the same scene on both
sides on a node; this section is its input.

**Gates.** `cuda_s8a_check` (N = 1 == CPU, every leg, tol 0), `cuda_prestage_parity_check` (device pre-stage ==
host, every n_sub 1..8; validated by dropping the cap2 thermal-solid rule: 32 cells caught -- a first break,
dropping the fold's `+ 1` count epsilon, was NOT caught: no scene put ceil_div on a boundary, so the gate cannot
see that change), `cuda_batch_invariance_check` (env i of N == alone, N in {1, 2, 8, 64}, n_sub 8 vs 1 on the same
tick on 30/30 ticks; validated twice: an ignored `sl_advect3` mask was caught at N = 8 and on the ship leg -- not at
N = 2, where the masked env was windless --, an un-rebased `K_pstar` temperature was caught at N = 2),
`cuda_tick_capture_check` (3 segments / ~690-720 graph nodes per tick, host syncs exactly `max_n_sub`, `books`;
validated by launching `K_store_atm` on the default stream: "operation would make the legacy stream depend on a
capturing blocking stream").

## G1: the reading (2026-11-10)

The design's first stop reading (§3): the same scene on both sides, the batched device path against
the CPU farm, on the Home Desktop. Measured numbers first, then the projection against the 3x gate,
then the transfer to an A100 slot. Every line below says whether it is measured or projected.

**In short.**
- Measured, same scene, today's code: the batched device path runs 597 (small) / 432 (medium)
  env-ticks/s; the 16-worker farm runs 1 756 / 552. The device path is at 0.34x / 0.78x of the farm
  today, because each env still pays 1.5 to 1.8 ms of host work per tick.
- Measured: the device chain alone costs 2.9 ms per batched tick plus 0.197 ms (small) / 0.486 ms
  (medium) per env, linear from N = 16 to 256.
- Projected, this PC, host work gone and the allowance charged: 2 950 / 1 335 env-ticks/s, which is
  2.2 to 2.6x (small) and 2.7 to 3.0x (medium) the full-game farm. With a farm on the slot's spare
  cores beside the device, as §3's metric counts it: 3.1 to 3.5x and 3.6 to 3.9x.
- Projected, one A100 slot: 2.5 to 3.3x (small) and 3.2 to 4.3x (medium) at the design's integer
  factor, more if a stage turns out bandwidth-bound; 3.4 to 4.2x and 4.1 to 5.2x with the farm on
  the spare cores.
- New since P1a: the farm scales past 8 workers. 16 workers give 45 to 54 % more than 8, so the
  yardstick moved up.
- **Erik's ruling (2026-11-10, on these numbers as they came in): the device route continues**
  (design §12.7).

### Method

- **Machine:** the Home Desktop (Ryzen 7 3700X, 8 cores / 16 threads, 16 GB; RTX 3070 8 GB, driver
  610.88, "Prefer maximum performance" set globally since 2026-11-07). Before timing: no heavy job;
  Spotify held about 0.4 of one core and VS Code about 0.1 to 0.8 throughout; the idle WSL VM was shut
  down. Everything ran serially, one configuration per fresh process.
- **Code:** main at `ad9cb343` (#183 G1p to G3p merged) on branch `29-g1-reading`, with its own CPU
  build (`cpp/build/Release`) and CUDA build (`cpp/build_cuda`), `bp.build_id()` == that commit on
  both. No `cpp/` file changed since P10.
- **The scene (design §3):** NO units, the level's entities ON (vents, ducts, doors),
  `ContinuousRealtime`, headless (no ripple), the small ship (`tools/gen_ship.py --seed 1 --size
  small --vents`, 60 x 40 = 2 400 tiles, outside `levels/`) and the medium ship (`levels/ship_demo`,
  100 x 70 = 7 000 tiles).
- **The scene is not stationary, so both sides time the same ticks of it.** A fresh empty ship runs
  the EOS at its substep cap (n_sub = 8) while its vents start; the small ship then settles (ticks
  0 to 1 000: n_sub mean 7.2 to 7.9; 1 000 to 1 200: 6.3; from 1 200: about 1), and its CPU tick
  falls from 5.3 ms to 2.8 ms. The medium ship holds 7.4 to 8.0 through tick 1 600. P5's batched
  numbers were ticks 6 to 65 of that transient. Here both sides time **ticks 51 to 400 after a
  fresh start** (n_sub mean 7.97 small / 8.00 medium on both sides), repeated from fresh starts
  until enough time is pooled. A full game was checked afterwards (2026-11-11,
  "Measured: substeps in a full game" below): it holds the cap for its first 1 200 (small) to
  2 400 (medium) ticks and runs at about half the substeps after that, so this window is a
  match's most expensive stretch, not its average.
- **Same scene, checked:** the tick digest of the first envs after 300 ticks is identical on the
  device side and on the farm side (`c56541526370` small, `e70e091a1672` medium), at every N and W.
- **Device side:** `tools/bench_resident_scaling.py --batch-run --window 50:400` -- N envs in ONE
  `ResidentBatch` (the P5 driver). The context is first loaded for 15 s on a throwaway state, then
  every env is replaced by a fresh simulation (`ResidentBatch.reset_env`) and ticks 51 to 400 are
  timed, pooled to at least 30 s. A second fresh start runs 100 ticks with a device sync after
  every device stage: the attributed split (`tick_stage_profile`'s timers plus the batch driver's
  own callables). `nvidia-smi` sampled the GPU every 0.5 s over the timed ticks: P2, 1 830 to
  2 010 MHz in every run, so the P2 section's clock drop did not occur.
- **Farm side, same scene:** `--farm-run --window 50:400` -- W processes on the CPU build, one env
  each, one BLAS thread (`rl_env.WORKER_ENV`), every worker timing ticks 51 to 400 of fresh start
  after fresh start until it has 25 s, and cycling on until all have (constant load). No driver
  and no IPC per step, so this is the farm's best case for the scene.
- **Farm side, the full game:** the §9.1 protocol unchanged (`tools/rl_smoke.py`, P1a's run files:
  4 v 4 + 4 zombies, random rows, the placeholder observation and reward rows ON, at least 3
  episodes per env, 20 warm-up steps), at 8 and 16 workers, decimation 1 and 4.
- **Machine time:** about 75 minutes (device 22, same-scene farm 9, full-game farm 44).

```
# device side, one fresh process per (level, N); N in 1 4 16 64 128 256
python tools/bench_resident_scaling.py --levels-dir <dir> --level <level> --batch-run --n <N> \
    --build cpp/build_cuda --warmup 5 --warmup-s 15 --window 50:400 --seconds 30 \
    --split-ticks 100 --digest-tick 300
# farm side, the same scene; W in 1 4 7 8 12 16
python tools/bench_resident_scaling.py --levels-dir <dir> --level <level> --farm-run --workers <W> \
    --cpu-build cpp/build/Release --window 50:400 --seconds 25 --digest-tick 300
# farm side, the full game (run files as in "P1a" above, n_envs = workers)
python tools/rl_smoke.py <scratch>/run_<level>_d<D>_n<W>.toml --workers <W> --episodes 3 --seed 0 \
    --warmup-steps 20
# the allowance: the batched sweep core
python tools/bench_radiation_sweep.py --cuda --build cpp/build_cuda --envs 16,64,256 [--env-grid 70x100]
```

### Measured: the device side

Plain = no stage syncs (the truth for the whole tick). Attributed = the synced pass; its sum is 3
to 6 % above the plain tick.

| small (2 400 cells) | N = 1 | 4 | 16 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|
| batched tick, ms (plain) | 5.51 | 11.04 | 30.84 | 116.4 | 214.3 | 436.1 |
| **env-ticks / s (whole tick)** | 182 | 363 | 519 | 550 | **597** | 587 |
| host slots per env, ms (plain: tick minus slot 7) | 0.50 | 0.44 | 0.41 | 0.45 | 0.41 | 0.41 |
| per-env bus (from_host + to_host), ms (attributed) | 1.22 | 1.02 | 1.19 | 1.17 | 1.04 | 1.06 |
| device chain per batched tick, ms (attributed) | 4.00 | 3.90 | 6.51 | 15.11 | 27.71 | 53.43 |
| **device chain per env, ms** | 4.00 | 0.98 | 0.41 | 0.236 | 0.217 | **0.209** |
| GPU utilisation over the timed ticks, % | 50 | 35 | 24 | 23 | 23 | 14 |

| medium (7 000 cells) | N = 1 | 4 | 16 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|
| batched tick, ms (plain) | 6.24 | 12.42 | 40.82 | 153.3 | 296.1 | 603.9 |
| **env-ticks / s (whole tick)** | 160 | 322 | 392 | 418 | **432** | 424 |
| host slots per env, ms (plain) | 0.61 | 0.55 | 0.55 | 0.57 | 0.57 | 0.64 |
| per-env bus, ms (attributed) | 1.16 | 1.26 | 1.23 | 1.31 | 1.21 | 1.21 |
| device chain per batched tick, ms (attributed) | 4.06 | 4.88 | 10.64 | 33.88 | 65.00 | 127.34 |
| **device chain per env, ms** | 4.06 | 1.22 | 0.67 | 0.529 | 0.508 | **0.497** |
| GPU utilisation over the timed ticks, % | 52 | 39 | 35 | 39 | 22 | 23 |

**The device chain is linear in N** (least squares over N = 16, 64, 128, 256; the four points sit
within 7 % of the line):

| | fixed per batched tick | per env | per cell per tick | the chain alone, N large |
|---|---|---|---|---|
| small | 2.90 ms | 0.197 ms | 82 ns | 5 090 env-ticks/s |
| medium | 2.80 ms | 0.486 ms | 69 ns | 2 060 env-ticks/s |

The fixed part is what does not grow with N (launches, the two host syncs); the per-env part is
what the card does per env once it is busy.

**Where one env-tick goes at N = 256 (attributed, ms per env-tick).** The device side's host cost,
line by line, as §3 asks:

| line | small | medium | after the flip |
|---|---|---|---|
| device chain: EOS | 0.081 | 0.248 | stays |
| device chain: tail (fire + the sweep + temperature + books) | 0.122 | 0.235 | stays |
| device chain: stamp + trace tail + combustion | 0.006 | 0.014 | stays |
| **device chain, sum** | **0.209** | **0.497** | |
| bus: entry upload (164 / 479 KiB per env) | 0.50 | 0.55 | leaves (P6+) |
| bus: exit download (291 / 848 KiB per env) | 0.56 | 0.66 | leaves (P6+) |
| host slot 9e, the entities (vents, ducts, doors) | 0.051 | 0.052 | becomes a kernel (P6b) |
| host slot 9b, the over-pressure wall scan | 0.248 | 0.335 | becomes a kernel (P6) |
| host slots, the rest (stamp rows 0.03 to 0.05, ignition, gas rows, the round head) | 0.12 | 0.28 | become kernels |
| slot 7's own host work (prepare, the physics head, the driver) | 0.07 | 0.08 | leaves |
| **host, sum** | **1.56** | **1.96** | |
| the attributed tick | 1.76 | 2.45 | |

- The per-env bus is 60 % (small) and 49 % (medium) of today's batched tick. It is about 27
  pageable CuPy `set` / `get` calls per env per tick; its cost barely differs between 455 KiB and
  1.3 MiB, so it is bound by the calls, not by the bytes.
- Device memory: 2.7 MiB per small env and 6.8 MiB per medium env (700 MiB / 1.7 GiB at N = 256).
  Host memory: 2.3 / 6 MB per env on top of ~320 MB.

**The batched sweep core** (the allowance below), re-measured at this commit: 0.130 ms per small
env and 0.252 ms per medium env at N = 64 to 256 (3.4 / 2.2 ns per cell-update; 63 / 103 launches
per tick). In the chain the whole tail costs 0.122 / 0.235 ms per env, so the sweep is 58 % of the
small chain and 47 % of the medium one.

### Measured: the farm

**The same scene** (no units, free-running workers, ticks 51 to 400):

| workers | small env-ticks/s | per worker | core-equivalents | medium env-ticks/s | per worker | core-equivalents |
|---|---|---|---|---|---|---|
| 1 | 184 | 184 | 1.00 | 57.1 | 57.1 | 1.00 |
| 4 | 700 | 175 | 3.80 | 216 | 53.9 | 3.77 |
| 7 | 1 122 | 160 | 6.09 | 319 | 45.6 | 5.58 |
| 8 | 1 210 | 151 | 6.57 | 368 | 46.0 | 6.44 |
| 12 | 1 478 | 123 | 8.02 | 448 | 37.3 | 7.83 |
| **16** | **1 756** | 110 | 9.53 | **552** | 34.5 | 9.66 |

**The full game** (§9.1: 4 v 4 + 4 zombies, random rows, observation and reward rows on, the
lock-step `BatchedEnv`):

| ship | decimation | workers | env-ticks/s | step ms p50 / p95 | timed ticks | / the same-scene farm at that W |
|---|---|---|---|---|---|---|
| small | 1 | 8 | 758 | 10.3 / 13.5 | 76 984 | 0.63 |
| small | 1 | 16 | **1 114** | 13.6 / 18.1 | 272 096 | 0.63 |
| small | 4 | 8 | 869 | 36.7 / 45.3 | 122 719 | 0.72 |
| small | 4 | 16 | **1 342** | 46.7 / 56.7 | 242 309 | 0.76 |
| medium | 1 | 8 | 320 | 23.9 / 36.7 | 172 640 | 0.87 |
| medium | 1 | 16 | **451** | 33.6 / 46.9 | 345 280 | 0.82 |
| medium | 4 | 8 | 326 | 99.9 / 120.8 | 172 115 | 0.89 |
| medium | 4 | 16 | **487** | 128.9 / 155.9 | 344 239 | 0.88 |

- **The farm scales past 8 workers.** Sixteen workers on the 8 cores / 16 threads give 9.5 to 9.7
  single-core equivalents on the same scene (8 workers: 6.4 to 6.6), and 41 to 54 % more full-game
  throughput than 8. P1a measured 7 and 8 only; its "the 8th worker buys nothing at decimation 1"
  holds for 7 against 8, not beyond. **The yardstick on this PC is the 16-worker farm: 1 114 to
  1 342 env-ticks/s small, 451 to 487 medium.**
- At 8 workers the full game is where P1a had it (758 / 869 against 727 to 755 / 951; 320 / 326
  against 277 / 337), so the gameplay pack did not move the farm's cost.
- **The physics share**, as §3 uses the word (the full-game farm over the same-scene farm): 0.63
  to 0.76 on small, 0.82 to 0.89 on medium. It holds everything the full game adds: units and
  controllers, the observation, the pipes, the lock-step wait for the slowest worker, resets. §3
  assumed about 0.8 on small.
- Random-row episodes are long at this commit: about 2 400 ticks per episode on average on small
  (P1a: mostly 100 to 200). With magazines and reloads (G2p) random teams no longer kill each other
  in the spawn rooms.
- Farm workers hold 45 to 49 MB each at any W.

### Measured: substeps in a full game (2026-11-11)

One in-process `EnvRunner` on the CPU build (the object a farm worker runs), the full game of §9.1
(4 v 4 + 4 zombies, random rows), decimation 1, `eos.dbg_last_n_sub` read every tick: 30 000 ticks
on small (14 episodes), 15 000 on medium (2 episodes, both to the 7 200-tick limit).

| ticks into the episode | small: mean n_sub | share at the cap of 8 | medium: mean n_sub | share at the cap of 8 |
|---|---|---|---|---|
| 1 to 400 | 7.89 | 0.98 | 7.96 | 0.99 |
| 401 to 1 200 | 7.30 | 0.86 | 7.88 | 0.98 |
| 1 201 to 2 400 | 2.58 | 0.21 | 7.60 | 0.91 |
| 2 401 to 4 800 | 2.73 | 0.25 | 3.69 | 0.34 |
| 4 801 to 7 200 | 4.37 | 0.48 | 2.95 | 0.24 |
| **all ticks** | **4.43** | **0.48** | **4.92** | **0.53** |

- A tick is at 1 substep or at the cap, rarely between: 49 % / 37 % of all ticks ran exactly 1,
  48 % / 53 % exactly 8.
- The start-up transient of the vents is the same as in the empty scene, so the timed window
  (ticks 51 to 400) matches the first part of a match and overstates the rest of it.
- What this does to the device side is NOT measured. A batch runs both substep loops to the
  largest n_sub among its envs, which at N = 256 is 8 on nearly every tick; an env past its own
  n_sub is masked and does no arithmetic, but its launches still run. The EOS is 39 % (small) /
  50 % (medium) of the chain, so the per-env chain in a full game lies between the measured number
  and roughly three quarters of it.
- Random-row episodes on small are short or long, rarely between (8 of 14 under 250 ticks, 2 at
  the limit; mean 2 100 ticks), so the per-bucket means of the later rows rest on few episodes.

### The reading

**1. Measured against measured (the same scene, today's code).**

| | small | medium |
|---|---|---|
| device, whole batched tick (N = 128) | 597 | 432 |
| farm, 16 workers | 1 756 | 552 |
| device / farm | **0.34x** | **0.78x** |
| device / the 8-worker farm | 0.49x | 1.18x |

Today the batched path is slower than the farm on both ships. P5 read "below on small, above on
medium" against the 8-worker farm; against 16 workers it is below on both. This is the state §3
calls R1: resident physics with the gameplay still on the host, never a shipped one.

**2. Projected against the 3x gate (this PC).** The device backend as the design completes it,
against the full-game farm. The assumptions, each stated:

| # | assumption | basis |
|---|---|---|
| A1 | After the flip the device side's per-env host work is zero: no bus, no host slots. | The design's goal (§1); not measured. The 1.56 / 1.96 ms above is what must vanish. |
| A2 | The gameplay kernels and applier stages that replace the host slots cost one more sweep per env-tick: 0.130 ms small, 0.252 ms medium. | §3's allowance, the sweep core measured at this commit. No such kernel exists; this is the least certain number here. |
| A3 | The device chain in a full game costs what it costs in this scene. | Both run the substep cap, which is the chain's worst case. Measured afterwards: a full game runs the cap on about half its ticks (mean n_sub 4.4 / 4.9). The farm's full-game rows already contain the cheap ticks and the device's chain numbers do not, so A3 errs AGAINST the device; by how much is not measured. |
| A4 | N = 256. | Measured. The fixed 2.9 ms is 5 % of the chain there. |
| A5 | The farm is the full game at 16 workers, as measured. | Measured. |
| A6 | Nothing is charged for the policy sharing the GPU, the observation export or the rollout buffers. | §3 names them; nothing of them exists at G1. A downside. |

| | small | medium |
|---|---|---|
| device chain per env, N = 256 (measured) | 0.209 ms | 0.497 ms |
| + the allowance (A2) | 0.130 ms | 0.252 ms |
| = projected device tick per env | 0.339 ms | 0.749 ms |
| **projected device env-ticks/s** | **2 950** | **1 335** |
| farm, full game, 16 workers: decimation 4 / 1 (measured) | 1 342 / 1 114 | 487 / 451 |
| **ratio, the device alone** | **2.2x / 2.6x** | **2.7x / 3.0x** |
| **ratio, the device plus a farm on the slot's spare cores** (§3's metric) | **3.1x / 3.5x** | **3.6x / 3.9x** |
| ratio, the device alone, against the 8-worker farm (P1a's yardstick) | 3.4x / 3.9x | 4.1x / 4.2x |

- The second ratio adds 0.9: 14 farm workers on the threads the device's one driver thread leaves
  free are about 0.9 of the 16-worker farm (the same-scene table, between its 12- and 16-worker
  rows). The two were not run together, and a trainer that consumes both backends at once is not
  built. §3 words the metric this way; the first ratio is the stricter one.
- Sensitivity to A2: with no allowance the device-alone ratios are 3.6 / 4.3 (small) and 4.1 / 4.5
  (medium); with twice the allowance 1.6 / 1.9 and 2.1 / 2.2.
- The design's own estimate for this PC was 1.4 to 3.4x (§2.3). The reading lands inside it.

**Against the working gate of 3x: on this PC the device alone projects below it on the small ship
(2.2 to 2.6x) and at its edge on the medium ship (2.7 to 3.0x); by §3's own metric it projects at or
above it on both (3.1 to 3.9x).**

**3. Transferred to one A100 slot (projected from projections).** §3's factors, applied by part:

| part | this PC | A100 slot | factor | why |
|---|---|---|---|---|
| device, fixed per batched tick | 2.9 ms | 3.6 ms | 0.8x | launch- and sync-bound: a slower host core and a lower GPU clock |
| device, per env (chain + allowance) | 0.327 / 0.738 ms | 0.172 / 0.389 ms | 1.9x | the integer factor: 2.35x the SMs at 0.8x the clock |
| the same, if a stage is bandwidth-bound | | 0.073 to 0.093 / 0.164 to 0.211 ms | 3.5 to 4.5x | the A100's DRAM bandwidth; today's chain runs far below the 3070's bandwidth limit, so this is upside, not the expectation |
| farm, full game, decimation 4 | 1 342 / 487 | 1 610 to 2 150 / 584 to 779 | 1.2 to 1.6x | 32 threads against 16, each Epyc 7742 core 0.6 to 0.8 of a 3700X core, the same scaling with workers assumed |

| one A100 slot, N = 256 | small | medium |
|---|---|---|
| projected device env-ticks/s, integer factor | 5 380 | 2 490 |
| projected device env-ticks/s, bandwidth factor | 9 300 to 11 500 | 4 450 to 5 630 |
| projected farm env-ticks/s | 1 610 to 2 150 | 584 to 779 |
| **ratio, the device alone, integer factor** | **2.5x to 3.3x** | **3.2x to 4.3x** |
| ratio, the device alone, bandwidth factor | 4.3x to 7.2x | 5.7x to 9.6x |
| ratio with the farm on the spare cores, integer factor | 3.4x to 4.2x | 4.1x to 5.2x |
| GPU-hours per 5e8 env-ticks (one §2.4 training run): farm / device at the integer factor | 65 to 86 / 26 | 178 to 238 / 56 |

- The integer factor is the one to expect. The per-env cost is constant from N = 16 to 256 on the
  3070, so the card is saturated in what the kernels ask of it, and the chain sits at about a sixth
  of the 3070's bandwidth limit (§2.3's ~6 KB per cell per tick against 448 GB/s), so bandwidth is
  not what saturates it.
- The transfer multiplies the PC ratio by 1.2 to 1.6 at the integer factor (1.9 over 1.2 to 1.6).
  It lowers the ratio only if the A100 gains less on today's kernels than the Epyc farm gains over
  this farm.
- A larger N on the A100 changes little: the fixed part is already under 10 % at N = 256.

**4. What is measured and what is projected.**

- Measured here: the device chain per env and its split, the fixed part, the bus, every host line,
  the farm on the same scene and on the full game at 8 and 16 workers, the sweep core, the GPU's
  clocks, digest equality of the scene.
- Projected: that the host lines vanish (A1), what the gameplay kernels cost (A2), that a full game
  costs the chain what this scene does (A3), every A100 and Epyc number.
- Not measured at all: a policy on the same GPU, the observation and rollout buffers, 14 or 32
  farm workers on a node, the device backend and a farm together.

**5. Can this PC reading stand as the G1 reading? (Erik's question; his to rule.)**

Recommendation: **yes.** A node run would replace two numbers, the A100's factor on today's
kernels and the Epyc farm's factor. It would not touch the two assumptions the projection leans on
most, A1 and A2, which no machine can measure before the gameplay kernels exist: they are what G2
measures. So a node reading at G1 narrows the smaller uncertainty and leaves the larger.

What could make that wrong:
- The A100 gains less than about 1.6x on today's kernels (a stage bound by per-thread latency
  rather than by the number of SMs). The slot ratio would then be below the PC's.
- The Epyc farm does better than 0.8 of a 3700X core per core, or scales better across 32 threads
  than this farm does across 16.
- The allowance (A2) is too small: at twice the allowance every ratio here falls by about a
  quarter.
- The policy shares the A100 with the device envs and slows them; the farm leaves the GPU to the
  learner.

The check is cheap when Berzelius is in use anyway: the two commands above on one slot, about 15
minutes of one GPU, with the container P10 built. It is a calibration of the transfer, worth
doing in the first session there, not a gate. G2 is the reading that belongs on a node.

**6. The ruling.** Erik, 2026-11-10, with the numbers above in front of him (the projection at
about 2 to 4x per slot at today's kernels): the device route continues, and the kernel throughput
work is wanted after it ("the whole point of the project"). Recorded in the design doc, §12.7.
Whether this PC reading stands as G1 without a node run is asked in item 5 and not yet ruled.

### Where the time is: input for the throughput work

Facts from the tables, in the order of what they cost today. None is acted on here.

1. **The per-env bus, 1.0 to 1.3 ms per env-tick,** is the largest line of today's batched tick. It
   leaves with the flip (P6+). Until then it is bound by the number of copy calls (about 27 per env
   per tick), not by bytes: one copy per field over the whole batch instead of one per env would
   remove most of it.
2. **Slot 9b, the over-pressure wall scan, 0.25 to 0.34 ms per env-tick,** is the largest host
   slot, five to six times the entities (9e, 0.05 ms).
3. **In the device chain the sweep is as large as the EOS:** the tail is 58 % of the small chain
   and 47 % of the medium one, at 3.4 / 2.2 ns per cell-update and 63 / 103 launches per tick. P9's
   single-launch sweep is the first kernel target by size.
4. **The fixed cost per batched tick is 2.9 ms** and already small at N = 128 (11 % of the small
   chain); the graph capture (P9) can only win that part back.
5. **The substep cap.** A fresh ship holds n_sub = 8 for its first 1 000 ticks or more with nothing
   happening in it, and a settled small ship ticks at half the CPU cost. How many substeps a match
   needs is a physics question (Erik's); each one less is cheaper on both backends.
   In a full game about half of all ticks sit at the cap and the first 1 200 to 2 400 ticks of
   every episode nearly all do (the table above): the vents' start-up, not the fight.
6. **The farm has headroom too:** it loses 24 to 37 % (small) between the free-running same scene
   and the lock-step full game, part of it the wait for the slowest worker.

### Not measured

- A Berzelius node: the A100, 14 to 32 farm workers on Epyc cores.
- The policy update in the loop, the observation and rollout buffers (§3's full metric); the device
  backend and a farm running together on one machine.
- The large ship; decimation 4 on the device side (the batch has no decimation yet).
- More than one repetition of the full-game farm rows (each ran once, 77 000 to 345 000 ticks).
