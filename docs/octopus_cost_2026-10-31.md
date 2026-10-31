# What an octopus costs per tick, by part (#106 P6e)

2026-10-31. This is a measurement only, and it sets no threshold. Erik's ruling is that cost is
not a design input until the final result is seen, so these are the numbers to judge by, not a
gate. Nothing in the sim's behaviour changed: the golden did not move, and slot 5b's new part timer
is off by default (see Method). This report sits beside the v1 match budget,
`docs/tick_budget_2026-10-29.md`.

## Machine and method

- **Machine:** Home Desktop `DESKTOP-0E98HUV` (`docs/dev_setup.md`'s table).
  - CPU: AMD Ryzen 7 3700X, 8 cores (from `wmic`).
  - GPU: NVIDIA GeForce RTX 3070, driver 610.62.
  - Python: anaconda base 3.11.7.
  - It is the same machine as the #125 tick budget.
- **Code:** branch `106-p6e-bench`, cut from `33-octopus-concept` `eba4fd6`.
  - CPU build: `cpp/build/Release`.
  - CUDA build: `cpp/build_cuda`, with all 8 live backends on (`tools/run_on_cuda.py`'s list,
    through `bench_bot_matches.setup_cuda`). "resident" adds `physics_runner.set_residency(True)`.
- **Instrument:** `tools/bench_octopus.py`. Each cell is one process with one scenario, 30 sim
  seconds (720 ticks at 24 tps) after 24 warm-up ticks:

      <breach-py> tools/bench_octopus.py --scenarios <s> [--octopi N] --sim-seconds 30 --json <out>
      <breach-py> tools/bench_octopus.py --cuda [--resident] --scenarios <s> [--octopi N] --sim-seconds 30 --no-baseline   --json <out>
      <breach-py> tools/bench_octopus.py --cuda [--resident] --scenarios <s> [--octopi N] --sim-seconds 30 --baseline-only --json <out>

  The CPU command runs the octopus run and its baseline in one process. On CUDA they run in two
  separate processes (see section D).

- **Part timer:** a timer hook on slot 5b, `creature_system.set_tick_laps`.
  - It is OFF by default. When off, each part costs one empty method call: no clock read and no
    state. It lives at module level, so it is never on a sim, a body or a digest.
  - When the bench attaches it, it reads `perf_counter_ns` between the real calls, in their real
    order. The parts are:

    | part | what it covers |
    |---|---|
    | **setup** | the unit lookup and the footprint rects, once per tick |
    | **hub** | `route_intents` + `hub_step` + `after_solver` (the Python integer state machines) |
    | **inputs** | `write_inputs` |
    | **chain** | `arm_chain.tick`, i.e. the C++ `arm_chain_tick` binding |
    | **unit** | `write_unit`'s pose write-back |
    | **hit_shape** | `build_hit_shape`, the P6d3 seam |

  - **slot 5b** is the sum of the parts.
  - **record** is `serialize_record`. It runs only when a recorder or a per-tick digest asks for
    it; a headless match pays it once, at the end.
- **The rest of the tick:**
  - **step** is the whole `sim.step()`. **rest** is step − slot 5b.
  - **ctrl** is the controllers' time in a match, taken from the `bench_bot_matches.TickTimer`
    split.
  - Match scenarios go through `run_match`, the one headless match loop. Room scenarios step the
    sim directly, the way `tests/_creature_scenarios.Scenario` does.
- **Baselines:** every scenario is also run without its octopus: same room, same match, no
  `[[creature]]`. On CPU each baseline runs immediately before its octopus run, in the same
  process. On CUDA the baseline runs in a fresh process of its own.
- **The timer is inert:** `--verify-inert` re-ran every scenario with the timer detached.
  - The final `tick_digest` was identical in 10/10 scenarios on CPU, 10/10 on CUDA and 10/10 on
    resident.
  - The digests were also identical across all three backends, scenario by scenario.
- **Scenarios:**

  | scenario | setup |
  |---|---|
  | **idle** | the gate room (flooded 30 × 30, 0.333 m tiles), one octopus, no intent |
  | **crawl** | the gate room; MOVE_DIR round a square, 3 s per side |
  | **swim** | the gate room; the umbrella swim (ORDER_MOVE_COVER) round the same square, from 8 m up |
  | **grab** | the gate room; the demo prop, 0.5 s of crawl, USE, then the square |
  | **hunt** | the gate room; the `creature` controller hunting two idle marines (one per team), through `run_match` |
  | **octopi ×N** | a flooded 56 × 36 room; N = 1, 2, 4 or 8 animals on a 2 × 4 grid (pitch 9 tiles), all crawling the square |
  | **ship** | `gen_ship --seed 1 --size medium --vents` (the #125 ship); its default match (4 v 4 bots, 4 zombies) plus one octopus through a `[[creature]]` match row, through `run_match` |

  Notes on the ship scenario:
  - The octopus stands in the zombies' room, because that is the only spot on this ship where its
    rest pose fits on open floor.
  - The horde STANDS (`zombie_controller = "idle"`), in both the octopus run and the baseline. On
    its own controller the horde bites the octopus dead by tick 43.
  - Even so, a team-1 bot shoots the octopus at tick 590. It was therefore alive for 564 of the
    720 timed ticks, and the per-octopus numbers are divided by living octopus-ticks.
  - The ship is not flooded, but the solver ignores the medium in v1, so this does not change the
    cost.

## Results

### A. One octopus, by part (CPU, ms per octopus per tick, mean)

| scenario | setup | hub | inputs | chain | unit | hit_shape | **slot 5b** | slot 5b p95 (all animals) | record | share of a 41.7 ms tick |
|---|---|---|---|---|---|---|---|---|---|---|
| idle | 0.008 | 0.154 | 0.004 | 1.98 | 0.012 | 0.340 | **2.50** | 2.71 | 0.024 | 6.0 % |
| crawl | 0.008 | 0.134 | 0.004 | 2.00 | 0.012 | 0.342 | **2.50** | 2.65 | 0.024 | 6.0 % |
| swim | 0.008 | 0.241 | 0.004 | 2.07 | 0.012 | 0.310 | **2.64** | 2.79 | 0.024 | 6.3 % |
| grab | 0.008 | 0.208 | 0.004 | 2.10 | 0.013 | 0.340 | **2.67** | 2.96 | 0.037 | 6.4 % |
| hunt | 0.017 | 0.136 | 0.006 | 2.00 | 0.013 | 0.348 | **2.52** | 2.77 | 0.025 | 6.1 % |
| octopi ×1 | 0.008 | 0.140 | 0.004 | 2.00 | 0.013 | 0.346 | **2.51** | 2.66 | 0.026 | 6.0 % |
| octopi ×2 | 0.007 | 0.136 | 0.005 | 2.00 | 0.012 | 0.340 | **2.49** | 5.27 | 0.020 | 6.0 % |
| octopi ×4 | 0.005 | 0.136 | 0.006 | 1.99 | 0.012 | 0.339 | **2.49** | 10.34 | 0.017 | 6.0 % |
| octopi ×8 | 0.005 | 0.135 | 0.008 | 1.99 | 0.012 | 0.340 | **2.49** | 20.48 | 0.015 | 6.0 % |
| ship | 0.073 | 0.149 | 0.011 | 2.08 | 0.015 | 0.354 | **2.68** | 2.87 | 0.040 | 6.4 % |

**An octopus costs 2.5–2.7 ms per tick, which is about 6 % of a 24 Hz tick.**
- **The chain is ~80 % of that** (2.0–2.1 ms).
- **The hit shape is ~13 %** (0.34 ms).
- **The hub is 5–9 %** (0.13–0.24 ms; the swim and the grab are the busiest).
- Everything else is under 0.1 ms.
- The tails are tight: p95 is within 10 % of the mean, and the worst single tick seen was 4.2 ms.

### B. The same split on every backend (ms per octopus per tick: slot 5b / chain)

| scenario | CPU | CUDA per-call | CUDA + resident |
|---|---|---|---|
| idle | 2.50 / 1.98 | 2.57 / 2.02 | 2.53 / 1.99 |
| crawl | 2.50 / 2.00 | 2.56 / 2.03 | 2.51 / 1.99 |
| swim | 2.64 / 2.07 | 2.72 / 2.11 | 2.67 / 2.08 |
| grab | 2.67 / 2.10 | 2.72 / 2.12 | 2.68 / 2.10 |
| hunt | 2.52 / 2.00 | 2.56 / 2.02 | 2.51 / 1.99 |
| octopi ×8 (per animal) | 2.49 / 1.99 | 2.54 / 2.02 | 2.51 / 2.00 |
| ship | 2.68 / 2.08 | 2.74 / 2.12 | 2.68 / 2.08 |

The chain runs on the host on every backend, so the octopus's own cost does not depend on the
backend. These GPU cells are from the same-process runs; the separate-process runs gave the same
slot 5b to within 0.05 ms. Section D shows what the octopus does to the rest of the tick.

### C. Several animals (crawling, 56 × 36 room)

The table shows ms per tick, mean, for all the animals together. The GPU columns come from
separate processes (see section D).

| N | slot 5b CPU | step CPU (baseline 2.25) | real-time × CPU | step resident (baseline 5.2) | real-time × resident | step per-call (baseline 12.3) | real-time × per-call |
|---|---|---|---|---|---|---|---|
| 1 | 2.51 | 4.92 | 8.46 | 7.96 | 5.24 | 16.34 | 2.55 |
| 2 | 4.99 | 7.48 | 5.57 | 10.97 | 3.80 | 20.51 | 2.03 |
| 4 | 9.96 | 12.63 | 3.30 | 18.66 | 2.23 | 28.92 | 1.44 |
| 8 | 19.89 | 22.93 | 1.82 | 34.36 | 1.21 | 44.18 | 0.94 |

- **The octopus's own cost is exactly linear in the number of animals:** 2.49 ms each at N = 1,
  2, 4 and 8, on every backend.
- The per-tick setup that the animals share is negligible.
- On CPU the rest of the step grows by ~0.1 ms per extra animal (one more unit to stamp, push and
  see).
- On the GPU the rest of the step grows superlinearly, because of the host-gap effect in
  section D: +0.24 / +0.61 / +3.4 / +9.1 ms on resident for N = 1 / 2 / 4 / 8.
- **Eight octopi take half a tick on CPU** (19.9 of 41.7 ms) and on top of that the room's own
  physics.
- With 8 octopi, resident is slower than CPU (1.21× against 1.82× real time).

### D. The whole tick, octopus vs baseline (ms, mean)

"rest − baseline" is what the octopus adds to the tick OUTSIDE slot 5b.

**How the GPU cells were run.** A CUDA process slows from run to run (#125's "repeat slowdown").
Here, per-call went 10.7 → 17.1 ms on the same baseline room by its third run, and in an earlier
same-process layout this was charged to the octopus as up to +7 ms. So on CUDA the octopus run and
its baseline each ran in a FRESH process (`--no-baseline` / `--baseline-only`). CPU does not
drift, so there they ran in one process, baseline first.

**CPU**

| scenario | slot 5b | step | baseline step | step − baseline | rest − baseline | real-time × (baseline) |
|---|---|---|---|---|---|---|
| idle | 2.50 | 3.75 | 1.11 | +2.65 | +0.15 | 11.1 (37.6) |
| crawl | 2.50 | 3.75 | 1.11 | +2.64 | +0.14 | 11.1 (37.6) |
| swim | 2.64 | 3.90 | 1.11 | +2.80 | +0.15 | 10.7 (37.7) |
| grab | 2.67 | 3.95 | 1.10 | +2.85 | +0.18 | 10.6 (37.8) |
| hunt | 2.52 | 3.96 | 1.29 | +2.68 | +0.15 | 10.4 (32.2) |
| ship | 2.10 | 22.22 | 20.36 | +1.86 | −0.24 | 1.69 (1.84) |

**CUDA + resident**

| scenario | slot 5b | step | baseline step | step − baseline | rest − baseline | real-time × (baseline) |
|---|---|---|---|---|---|---|
| idle | 2.54 | 7.37 | 4.66 | +2.70 | +0.16 | 5.66 (8.94) |
| crawl | 2.53 | 7.34 | 4.67 | +2.68 | +0.15 | 5.67 (8.93) |
| swim | 2.68 | 7.50 | 4.73 | +2.77 | +0.09 | 5.55 (8.81) |
| grab | 2.69 | 7.51 | 4.67 | +2.84 | +0.15 | 5.55 (8.93) |
| hunt | 2.53 | 7.48 | 4.84 | +2.64 | +0.11 | 5.54 (8.60) |
| ship | 2.11 | 15.74 | 11.73 | +4.01 | +1.89 | 2.29 (2.98) |

**CUDA per-call**

| scenario | slot 5b | step | baseline step | step − baseline | rest − baseline | real-time × (baseline) |
|---|---|---|---|---|---|---|
| idle | 2.58 | 15.42 | 11.10 | +4.32 | +1.75 | 2.70 (3.75) |
| crawl | 2.56 | 14.83 | 11.00 | +3.82 | +1.26 | 2.81 (3.79) |
| swim | 2.69 | 15.27 | 11.02 | +4.26 | +1.56 | 2.73 (3.78) |
| grab | 2.70 | 15.40 | 10.66 | +4.74 | +2.04 | 2.71 (3.91) |
| hunt | 2.55 | 15.34 | 11.26 | +4.08 | +1.53 | 2.71 (3.70) |
| ship | 2.14 | 23.35 | 17.94 | +5.41 | +3.27 | 1.61 (2.06) |

On the ship rows, slot 5b is averaged over all ticks, and the octopus was alive for 78 % of them.

**One octopus, on CPU and on resident, adds its own slot 5b and almost nothing else** (+0.1–0.2 ms
for one more unit).

**FINDING: on CUDA, host work between device calls slows the device share of the step.**
- **Per-call:** +1.3–2.0 ms beyond slot 5b with one octopus.
- **Resident:** the effect appears once host work is large:
  - N = 4 octopi: +3.4 ms;
  - N = 8 octopi: +9.1 ms;
  - the ship (bots + octopus): +1.9 ms.
- A cProfile of per-call shows where the time goes: it is inside the device calls
  (`run_substeps`, `step_water`), while the Python around them stays flat.
- **A plain CPU spin with no octopus partly reproduces it.** This is the baseline room's step,
  excluding the spin, with one process per value:

  | backend | no spin | 2.5 ms spin | 6 ms spin |
  |---|---|---|---|
  | CUDA per-call | 10.68 ms | 10.77 ms | 12.03 ms |
  | CUDA + resident | 4.68 ms | 4.75 ms | 4.74 ms |

  So a short gap is nearly free, and a long or busy one is not.
- The likely causes are the GPU dropping its clocks or power state, and launch or copy latency
  that is no longer hidden, perhaps with cache pressure from the solver's working set. I did not
  investigate further.
- It applies to any host work that grows: the bots' A*, a future brain, many animals.

### E. The ship beside the v1 match budget

The tick-budget ship is gen_ship seed 1 medium, 4 v 4 bots. The table shows ms per tick, mean;
GPU cells come from separate processes.

| backend | #125 (bots + horde, 60 s) total / real-time × | this bench, baseline (bots, standing horde, 30 s) total / real-time × | with 1 octopus: total / real-time × | octopus alive |
|---|---|---|---|---|
| CPU | 21.2 / 1.96 | 22.6 / 1.84 | 24.7 / 1.69 | 564 of 720 ticks |
| CUDA per-call | 23.6 / 1.77 | 20.2 / 2.06 | 25.9 / 1.61 | 564 of 720 |
| CUDA + resident | 14.0 / 2.98 | 14.0 / 2.98 | 18.2 / 2.29 | 564 of 720 |

- **One octopus costs the v1 match about 2.1 ms (CPU) to 4.2 ms (resident) per tick, averaged
  over the match.** It died at tick 590, so this is its slot 5b of 2.7 ms while alive, plus
  section D's host-gap effect on the GPU.
- The ship stays faster than real time on every backend.
- The bench's 30 s windows carry the bots' early-match planning, so their ctrl is ~2.2–2.5 ms
  against #125's 60 s mean of 1.3 ms.

### F. The chain against the P1 cost ceiling (reported, no verdict)

- **The assumption being checked:** P1 put a ceiling of 0.2 ms per octopus-tick at 1 ns per op.
  Erik has suspended that ceiling.
- **The P6c tally:** P6c's G7 counted **28,098 weighted ops per arm-substep**. That is ×16
  substeps × 8 arms = **3.60 M per octopus-tick** (P1's float estimate was 0.6 M).
- **Measured:**
  - The chain costs **1.98–2.10 ms per octopus-tick in the sim**.
  - That is **0.55–0.58 ns per weighted op**, so the C++ runs at about twice the speed of the
    "1 ns/op" assumption.
  - It is **0.0155–0.0164 ms per arm-substep**.
  - The whole octopus, 2.5 ms, is ~12× the suspended 0.2 ms ceiling. The integer chain does ~6×
    P1's op count, at about half P1's assumed cost per op.
- **The solver call alone** (`--chain-micro`) runs on a copy of a crawling octopus's buffers,
  200 calls each:

  | substeps | ms per call | ratio to 16 substeps |
  |---|---|---|
  | 16 (shipped) | 1.86 | 1.00 |
  | 12 | 1.40 | 0.75 |
  | 8 | 0.94 | 0.50 |

  The substep count was overridden for timing only; the other row constants keep the shipped dt,
  so this is not a valid 12- or 8-substep animal.
- **Batching does not help.** One (N, 8, n, 3) call against N single calls takes 1.871 / 3.711 /
  7.440 / 14.818 ms against 1.872 / 3.725 / 7.439 / 14.945 ms for N = 1 / 2 / 4 / 8. That is a
  ratio of 0.99–1.00, so the binding's per-call overhead is negligible.
- **The call costs ~6 % more inside the sim than alone** (1.98 against 1.86 ms), most likely
  because the rest of the tick evicts it from the cache.

## What dominates

1. **The chain solver, 80 %:** ~2.0 ms per octopus. It is pure C++ integer arithmetic and
   single-threaded. Its cost scales linearly with substeps and with animals.
2. **The hit shape, 13 %:** 0.34 ms of Python per tick. That is ~140 `segment_tiles` calls (one
   per arm segment) plus a dict build. It is rebuilt every tick, even when nothing shoots.
3. **The hub, 5–9 %:** 0.13–0.24 ms of Python integer state machines.
4. **On the GPU backends, a second-order effect:** host time slows the GPU's share of the tick
   (section D).
   - With one octopus it is +1.3–2 ms on per-call and ~0 on resident.
   - With 4–8 octopi, or with bots plus an octopus, it is several ms on resident too.

## Levers: options for Erik, with estimates (none is recommended; quality first)

The three that look cheapest come first.

1. **Hit shape, built lazily or in C++. Saves ≈0.3 ms per octopus per tick (~13 %).**
   - The shape is derived and never digested. Its only readers are the bullet march and the laser
     skewer.
   - **Lazily:** build it on the first `hit_tiles()` read of the tick, from the chain buffers as
     they stand after slot 5b. That saves ~0.34 ms on every tick without a shot in flight. The
     P6d3 gates (arm hit, near miss, skewer, frozen at death) would be re-run. The code cost is
     small.
   - **In C++:** emit the supercover tiles from the solver call. That costs ~0.01 ms but adds a
     reference twin.
2. **Substeps 16 → 12 or 8. Saves ≈0.5 or ≈1.0 ms per octopus (chain −25 % / −50 %).** This is
   one row value, `[creatures.octopus.solver] substeps`, with no code change.
   - **The gates to re-run:**
     - the P6b integer reference gates;
     - the P6c tol-0 C++ = reference check over 10 s reach / crawl / fall;
     - all P6d2 gates through `Simulation` (crawl speed and rhythm, umbrella, jet train, grab,
       idle);
     - CPU = CUDA = resident.
   - **Quality risk to judge:** P1 found that 8 × 1 leaves the arm behind the bend 6–7° curved,
     while 16 × 2 brings it to 2°. That is why the row says 16 × 2. Raising passes and lowering
     substeps are not interchangeable (P1).
   - This is a modelling call, Erik's.
   - About one session for 12 (re-derive, re-gate, look at the GIFs), and the same again for 8.
3. **Threads in the chain. Saves up to ~1.5 ms per octopus at N = 1, or ~85 % at N = 8.**
   - **Over animals:** one batched (N, 8, n, 3) call with the env loop on threads. The animals
     are uncoupled in the call, so this is trivially parallel and determinism-safe (integer, no
     shared writes). On 8 cores, 8 octopi's 16 ms becomes ~2–3 ms. Batching alone gains nothing
     (section F), so it pays only together with threads.
   - **Over the 8 arms of one animal:** the arms couple through the crown within each pass, which
     needs ~100 barriers per tick. Estimate ~3× on the chain, i.e. 2.0 → ~0.7 ms.
   - Medium engineering. The thread pool must not fight the engine's own threads.

Further options:

4. **SIMD over the 8 arms. Estimate 1.5–2.5× on the chain (≈ −0.8 to −1.2 ms).** The 8 arms map
   onto AVX2 lanes, but the kit's 128-bit products and symmetric roundings vectorise badly on
   int64. Bit-identity to the reference is kept by construction (same integer ops). High
   engineering cost.
5. **Hub in C++. Saves ≈0.1–0.2 ms per octopus.**
   - It means porting `creature_primitives` (the state machines, the xorshift and the decision
     comb) into the engine.
   - It needs a new twin, and P6d2's gates and the digest would have to stay bit-identical.
   - It has the highest cost per ms saved.
6. **The deferred CUDA twin of the solver.**
   - At N = 1 it would be launch-bound: 16 substeps × several dependent kernels over 144 nodes.
     It would likely be no faster than the C++, and it adds the host↔device copies.
   - It pays only for batched many-env RL (#29), with N ≫ 8 animals in one launch.
   - Section D's host-gap effect means that, on GPU backends, shrinking host work also gives
     back part of the device step. That counts for levers 1–5 as well.
7. **Multiple animals sharing one (N, 8, n, 3) call** is measured at 0–1 % on its own. It is only
   a carrier for lever 3 or 6.

## Findings to revisit

- **The P6d1 root-velocity ghost.** This is reference behaviour. While the crown idles or crawls
  at up to 0.55 m/s, the root nodes' stored velocity sits 2–7 mm/s off the crown's rigid velocity
  (0.02 mm/s when limp). No primitive reads node velocities, so it is harmless so far. A future
  reader of root velocity would inherit it. It has no cost impact and was not measured further
  here.
- **The GPU host-gap effect (section D).** On CUDA, long host work between device calls slows the
  device share of the step.
  - Per-call: +1.3–2 ms with one octopus.
  - Resident: +3.4 / +9.1 ms with 4 / 8 octopi.
  - Not investigated.
- **#125's open "CUDA repeat slowdown" reproduced, and it is large.** On the same baseline room, a
  per-call process's third run took 17.1 ms against 10.7 ms for its first. Any multi-episode
  process (RL training) sees it. Same-process A/B timings on CUDA are therefore invalid, and
  `bench_octopus` separates the processes for that reason.
- **The v1 match kills the octopus.**
  - On the ship the horde bites it dead by tick 43 when it shares their room: zombies attack a
    creature.
  - With the horde standing, a bot shoots it at tick 590.
  - Neither is a bug against any rule. They are recorded because a match bench needs a standing
    horde to keep an octopus alive.
- **The ship has exactly one octopus spawn spot.** On gen_ship seed 1 medium, only the zombies'
  18 × 19 room fits the rest pose (arms straight and radial, ~3 m around the crown, every node on
  open floor). A generated ship meant for octopi needs a room that size or a curled spawn pose
  (#120 / #131, the zone vocabulary).

## Not measured

- **The window:** the renderer, the light sweep, and P7's arm render.
- **Bigger animals.** The row is the mid-size animal; the cost per arm-substep should scale with
  nodes.
- **Multi-process CPU throughput.** Running several sims on 8 cores is the obvious answer to the
  N-animal scaling at RL time, and it was not measured.
