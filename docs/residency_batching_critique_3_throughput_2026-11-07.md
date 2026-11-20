# Design critique, lens 3: throughput honesty (#29)

2026-11-07. One critic, one lens, on `docs/residency_batching_design_2026-11-07.md` §0–§13
(design v1, branch `29-residency-batching` at `64db125`). §14 and the other lenses' files were
not read. Evidence read: the benchmark doc (whole), the where-we-stand page, the litsearch
§4–§7 and §10, `docs/tick_budget_2026-10-29.md`, `config.toml` (`[clock]`, the EOS MG keys),
`cpp/src/cuda_eos_resident.cu`, `cpp/src/cuda_eos_step.cu` (the host pre-stage),
`cpp/src/cuda_mg_solve.cu`, `cpp/src/cuda_radiation_sweep.cu`, `tools/bench_radiation_sweep.py`.
Nothing was run.

**VERDICT: SOUND WITH FIXES.** The table arithmetic re-derives. But the CPU yardstick sits on
the wrong basis, the A100 top end and every §3 range are not derivable from what the design
states, and the stop condition is read on the wrong machine, after the cost it should avoid, with
no anchor in what training needs.

**Checked and correct** (no finding): §2.1's rows against the benchmark's medium resident table
(they sum to 11.5 ms); every §2.2 cell against the per-cell table; the ~400 ns host floor
(110-116 + 219-232 + 24-28 + 15-17 + 26-30 + 4-7); 1,750 / 70 = 25x; (a) = 21.61 / 128 =
0.169 ms; (b) = 15.62 / 128 = 0.122 ms; 3,440 and 2,420 env-ticks/s; medium 1,180 / 920; the
1.6x here (2,420 / 1,500); the A100 factors as hardware ratios (108 vs 46 SMs at 1.41 vs ~1.73
GHz boost ≈ 1.9x on integer; 1,555-2,039 vs 448 GB/s = 3.5-4.5x on DRAM).

## Blockers

**B1. §3 stop condition, §12.2, §9 P9: the gate is read on the wrong slot and has no anchor.**

- *Evidence.* §1 defines throughput "per GPU slot (one A100 + 16 Epyc cores on Berzelius; one
  RTX 3070 + 8 cores here)". §2.3's own numbers give the device route ~1.6x the farm here and
  1.5-7x per Berzelius slot at today's kernels. A gate read on the PC can stop the arc where the
  target slot already clears 3x, or pass where it does not.
- *How badly the PC's ratio transfers.*
  - GPU: the A100 is ~1.9x a 3070 on integer-bound work and 3.5-4.5x on DRAM-bound work. On
    latency-bound work it is ~0.8x, because its boost clock is lower. The tick has such work:
    - the MG coarse tail is one block (`mg_fused_tail<<<1, tail_block>>>`,
      `cuda_mg_solve.cu:399`), with ~119 barrier phases per V-cycle at 40 × 60 and 8 V-cycles
      a tick (`config.toml:780-783`);
    - the sweep is 60 dependent wavefront launches (`cuda_radiation_sweep.cu:841-852`).
  - CPU: 16 Epyc cores are ~1.2-1.6x the 8 Ryzen cores (16 × 0.6-0.8 against 8; SMT on both
    sides).
  - So the device/farm ratio on Berzelius is the PC's ratio × ~0.5 to ~3.7. "3x here" means
    anything from ~1.5x to ~11x there.
- *What the allocation buys.* NSC allocates GPU-hours, with 16 cores bundled per A100 (litsearch
  §10). On Berzelius the farm keeps 16 cores busy and leaves the A100 nearly idle (the policy
  only). The device route does the opposite. The quantity the allocation buys is env-ticks per
  GPU-hour, and a hybrid (the device backend plus farm workers on the slot's spare cores, which
  litsearch §10 already notes) is a third option the gate ignores.
- *No anchor.* Nowhere in §0–§13 does the design state the throughput training needs, so "3x"
  measures against nothing. The table below estimates it: 1e9 agent decisions ≈ 5e8 env-ticks ≈
  11-19 h on one Berzelius node as a farm, or ~6-8 days as a farm on this PC.
- *Fix.*
  - Restate §3 / §12.2 as: env-ticks/s per Berzelius GPU slot, on the training map(s), with the
    policy update in the loop. Compare the device backend (plus the farm on the slot's spare
    cores) against the farm alone. Use the same driver, tables and decimation on both sides
    (M5).
  - Take the decisive reading on a Berzelius node, so P10 lands before P9's gate. The PC
    reading becomes a proxy reported beside it.
  - Derive the factor from GPU-hours per training run at the planned scale. Erik rules on the
    scale (decisions per run × runs) and on the factor; the design shows the table.

**B2. §3 + §9: the stop condition is read after the cost it is meant to avoid.**

- *Evidence.* The gate runs at P9 only. By then these have all merged:
  - P3: Philox, Q16 unit positions and the distance field. That is three golden re-baselines
    and two HUMAN-TESTs, all in the played game.
  - P5: the batched physics.
  - P6, P6b, P7 and P8: a device copy of gameplay, which carries twin debt on every later
    gameplay change (§6, §10).

  §3 promises that if the kernels do not deliver, "the arc stops honestly at R0 plus finished
  residency". As scheduled, it would stop at R0 + residency + batched physics + an indebted
  gameplay twin, so the promise cannot be kept. The readings that would decide the device
  route's future exist earlier: physics-only device against physics-only farm after P5, and
  equal game after P8.
- *Fix.* Three readings, all with B1's metric:
  - **G1, after P5** (batched physics, brackets gone): continue to P6 only if the G1 ratio,
    scaled by the farm's measured physics share of its env-tick, projects at or above the gate.
    On small that share is 5.1 of ~6.4 ms, about 0.8.
  - **G2, after P8** (equal game, measured). P9 runs only if G2 is below the gate and P2's
    Nsight capture names a bounded kernel.
  - **G3, after P9.**

  Move P3's position and zombie changes behind G1: only P6 onward needs them, and the P4 / P5
  gates compare the device to the CPU reference at the same commit. Then stopping at G1 really
  is "R0 + residency + batched physics", with no feel re-baseline sunk into it.

## Majors

**M1. §0, §2.3: the single-core yardstick is on the wrong basis.**

- *Evidence.*
  - 196 is the stacked probe's N = 1 CPU row (5.11 ms). That probe steps a `Simulation`
    "physics only: no units, no controllers" (benchmark line 231), and the benchmark itself
    calls the number "physics-only" (line 445).
  - The design labels it "the game as-is" and "(all idle)" (design lines 29 and 92), and says
    "the farm's gameplay is inside its 196" (line 100). That is false.
  - With the match's 12 units idle, small runs 5.9 ms = 170 ticks/s (tick budget line 64), or
    ~175 allowing for the ~3 % the code has gained since (medium cpu step 19.9 → 19.28 ms). The
    units alone add ~0.46 ms of `stamp_units`: 0.490 ms in the small all-idle match against
    0.03 ms with no units in the stacked split.
  - "~160 with bots" is the 10-29 step-only time, 6.4 ms (tick budget line 62); it is not in
    the 11-07 benchmark. It is the right per-core basis for a training farm, because the policy
    replaces the bots.
  - The 16-core range is nevertheless built on ~196: 16 × 196 × [0.6 × 1.0, 0.8 × 1.3] =
    1,880-3,260 ≈ "2,000 to 3,000". The basis of the 8-core range is not stated.
  - SMT is applied to the Epyc (1.0-1.3x) but not to the 3700X (8C/16T). Neither machine
    reserves a core for the trainer, the obs assembly or the IPC, and litsearch §10 itself says
    the 16 cores are shared.
- *Fix.*
  - Quote one core four ways: 196 physics-only with no units, ~175 for the match with units
    idle, ~155-160 for the match with active units (training-shaped, step only), and 142 with
    the bots' controllers.
  - 8 cores here: 7-8 workers × ~155 × 0.9 (all-core clock, shared L3) × SMT 1.0-1.3 ≈
    1,000-1,450. Physics-only it is ≈ 1,250-1,830.
  - A 16-core slot: 14 workers × ~155 × 0.6-0.8 × 1.0-1.3 ≈ 1,300-2,250.
  - Compare like with like: physics-only against physics-only, game against game.

**M2. §2.3, §0 claim (3): the A100 range cannot be reproduced from its factors, and it assumes
the whole tick scales.**

- *Evidence.*
  - 2,420 × 2 = 4,840 and 2,420 × 4.5 = 10,900. The top end of 14,000 matches 3,440 × 4,
    which is the benchmark's row WITHOUT the allowance (line 438). The top end silently drops
    the allowance.
  - The factor holds only for throughput-bound stages. These do not scale with it:
    - the latency-bound parts (the fused tail, the 60 dependent sweep launches, every launch
      floor) run at the A100's ~0.8x clock;
    - (a) contains the EOS host pre-stage, single-threaded CPU code that no GPU factor touches:
      several per-cell loops, one of them with an int64 division per cell (`cuda_eos_step.cu`
      216-260), and a per-cell `quantize((double)…)` (`cuda_eos_resident.cu:800`).
  - In the other direction, the A100's 40 MB L2 can hold a batch's level-0 MG working set
    (~127 KB per small env: five int64 arrays, an int32 and an int64 per cell, `MGLevelDev`,
    `cuda_mg_solve.cu:101-111`) up to ~300 envs, which can beat 4.5x. So the factor is bounded
    by neither 2 below nor 4.5 above.
- *Fix.* Quote 4,800-10,900 as the throughput-bound band and say it is conditional. Split the
  3070 tick into throughput-bound and latency-bound parts from P2's Nsight capture before
  quoting any A100 number. The decisive figure is the one measured on a Berzelius slot (B1).

**M3. §3: the route table's numbers are underived and contradict §2.3; the 10,000-50,000 needs
its roofline.**

- *Evidence: the table.* The R3 cell reads "2,000-5,000 (3070 now ~2,400)". It sits in the row
  whose R0 entry (2,000-3,000) is the Berzelius-slot number, while §2.3 puts R3 on an A100 at
  5,000-14,000. R2 repeats R3 with no basis. "10,000 to 50,000" appears nowhere in the
  benchmark or the litsearch.
- *Evidence: the DRAM traffic, counted from the code.* n_sub = 1, small env 40 × 60, per-env
  hierarchy:

  | part | bytes per cell per tick | source |
  |---|---|---|
  | MG solve, level 0 | ≈ 3.5 KB: 8 V-cycles × (4 RB-GS sweeps that each stream m, gE, gS, recip, b as int64 and P as int32, ≈ 92 B because colour-skipped cells share cache lines; + residual ≈ 44 B, restrict ≈ 11 B, prolong ≈ 9 B) | `cuda_mg_solve.cu:101-111`, `:380-410`; `config.toml:780-783` |
  | substep chain | ≈ 0.5 KB: 3 D2D copies, SL advect, bulk / energy transport over the trace planes | `cuda_eos_resident.cu:875-905` |
  | build, div / N / p*, kick, store, energy flux | ≈ 0.5 KB | |
  | the sweep | ≈ 1 KB: 16 ordinates × (two upwind int64 reads, one int64 store, four int64 per-cell books, five plane reads) | `cuda_radiation_sweep.cu:470-515` |
  | temperature, fire, combustion, trace tail | ≈ 0.5 KB | |
  | **total** | **≈ 6 KB (4-9 KB)** | |

- *Evidence: the roofline.* A small env-tick moves ≈ 14 MB. At 100 % of peak DRAM bandwidth
  that caps a 3070 at ≈ 31,000 env-ticks/s (21,000-47,000), an A100 40 GB at ≈ 108,000 and an
  A100 80 GB at ≈ 140,000.
  - Today's 2,420 is ≈ 8 % of the 3070's roofline.
  - The EOS chain runs at ≈ 18 % at N = 128 (≈ 5.7 KB × 307,200 cells in 21.4 ms), and that
    time includes its host pre-stage.
  - The batched sweep runs at ≈ 3 % (4.9 M cell-updates × ~40 B in 15.6 ms), at ≤ 60 of 256
    lanes busy per block (`grid = (ceil(L_max / 256), 16, N)`, `:842-844`).
- *What 10,000-50,000 means.* Per A100 it is 9-46 % of the roofline. So it is not made up. But
  the top end needs about half the roofline across the WHOLE tick, every stage 3-15x more
  efficient than today. On the 3070, the gate's own machine, 50,000 exceeds the roofline
  outright.
- *The latency floor (my estimate, unmeasured).* Per batch tick:
  - launches: ~250-300, of which 97 are MG solve launches (8 × 12 + 1), 22 the build and 63
    the sweep;
  - the fused tail's ~950 in-block barrier phases.

  That is ~1-2 ms, so 50,000/s needs N ≥ ~400-800 envs per batch to keep the floor under a
  quarter of a batch tick.
- *Fix.*
  - Derive every §3 cell or delete it, and name the GPU in each.
  - State 10,000-50,000 as a ceiling band, "10-45 % of the A100's DRAM roofline at today's pass
    count (~6 KB per cell per tick)".
  - If a planning number is wanted, ~20 % of the roofline after P9 gives ≈ 6,000 on a 3070 and
    ≈ 20,000 on an A100 40 GB.
  - Replace the bytes-per-cell count with Nsight's measured DRAM bytes at P2.

**M4. §0 claim (4), §3: "pays only after the authority flip AND kernel work" is not
established. The numbers allow either outcome.**

- *Evidence: what lifts the estimate.*
  - (a) includes the host pre-stage (benchmark "Not measured"; `cuda_eos_resident.cu:776-801`),
    so the device EOS costs less than 70 ns per cell.
  - The allowance is 0.122 ms per env-tick: 51 ns per cell, ≈ 16 KB per cell of DRAM traffic
    at 314 GB/s. Temperature, fire, combustion and gameplay need ≈ 0.5 KB per cell plus 12-16
    unit rows, ~25-30 µs per env-tick even at today's ~15 % efficiency. So the allowance is
    ~4x generous at today's efficiency and ~30x in raw bandwidth.
  - The flip alone therefore gives ~3,000-3,400 here: 2-3x the corrected farm (M1), at or near
    the gate without P9.
- *Evidence: what lowers it.*
  - The stacked probe ran identical, quiet copies with one shared n_sub. A batch of fighting
    envs runs to the batch maximum (`N_SUB_MAX = 8`, `config.toml:784`). Masks save bandwidth
    but not launches (~12 per extra substep, `cuda_eos_resident.cu:875-905`).
  - The CUDA second-run slowdown is 10-20 % (tick budget, "Notes").
  - The policy shares the GPU.
  - The 8 GB card limits N (M6).
- *Fix.* Quote the device estimate as 2,400-3,400 here and name these movers. Let G2 (B2)
  decide whether P9 is needed.

**M5. P1: the stop condition's denominator has no protocol.**

- *Evidence.* P1's gate is "N workers reproduce `run_match`'s digests per seed … the farm number
  in the benchmark doc". It does not say:
  - which driver steps the farm (with `run_match`'s bots, ctrl adds 0.63 ms per tick on small,
    tick budget line 62);
  - whether obs and reward are on, or at what decimation;
  - how many processes run on 8C/16T (8 or 16);
  - whether a core is reserved for the trainer;
  - how many episodes each worker runs through reset (the 10-20 % repeat effect exists on CUDA;
    the farm must show it has none);
  - which map.

  Each of these moves the denominator by 10-30 %.
- *Fix.* Specify the measurement:
  - `BatchedEnv` `cpu_farm` stepped by P1's random-action driver, with the `[rl.obs]` /
    `[rl.reward]` placeholder rows on;
  - decimation 1 and 4;
  - 7, 8, 14 and 16 processes, with one core running a trainer stub;
  - ≥ 3 episodes per worker through reset;
  - small and medium;
  - report env-ticks/s, p95 and per-worker RSS.

**M6. §2.3, §3, §12.2: the 8 GB card cannot hold the batch the estimates assume plus the
policy's rollout.**

- *Evidence.*
  - The estimate is taken at N = 128 (benchmark lines 436-439). The kernel-efficient regime
    after P9 needs several hundred envs (M3's latency floor).
  - The sim's own memory is small: 204 MiB for 128 stacked ships, 498 MiB for 1,024 batched
    sweeps.
  - §4.5's observations are per-unit crops: C ≈ 10-12 planes × r² floats, ≈ 21 KB per agent at
    r = 21 and 12 channels. A PPO rollout of 8 agents × 512 envs × 32 steps is ≈ 131 k crops,
    ≈ 2.8 GB (≈ 6.8 GB at r = 33), before torch's pool and the policy itself.
  - "Device memory is not the constraint" (benchmark) holds for the sim alone.
- *Fix.* Measure the gate at a stated N, with the obs buffers allocated and a rollout-sized
  buffer reserved, or report the largest N that fits on 8 GB at a stated rollout. Store
  observations as fp16 / uint8 where the table allows. On an A100 (40 / 80 GB) the limit goes
  away, which is one more reason for B1's Berzelius reading.

## Minors

1. §2.1's bus 0.8 ms is the synced (attributed) figure. The plain one is 0.57 ms, because
   `from_host` doubles under the sync (benchmark).
2. §0's "Python gameplay of about 3 ms per env-tick" includes the bots' 1.5 ms of controllers,
   which training replaces. The slots alone are 1.9 ms (medium, bots) and ~1.2 ms (small, units
   idle).
3. "25x one CPU core" compares the whole 3070, host pre-stage included, with one core. Against
   the whole 8-core machine it is ~3x.
4. The batched sweep is only 3.2x one CPU core (3.16 against 10.1 ns per cell-update), at ~1 %
   of integer issue and ~3 % of DRAM bandwidth. Say so: it is the strongest argument for P9's
   sweep item.
5. The stacked EOS is not the per-env hierarchy. By `run_schedule`'s count it makes 449 solve
   launches per tick at N = 128 (5 wide levels, 7 levels total) against 97 for a 40 × 60 env.
   The per-env tail moves a 600-cell level into one block per env. The net direction is
   unknown.
6. n_sub was unmeasured, and the probe's copies all shared one value. State the masked
   batch-max assumption in §4.2's cost terms.
7. The sweep bench reports best-of-3 × 20, not the mean (`bench_radiation_sweep.py:163-172`).
8. The 10-20 % CUDA second-run slowdown is not carried as a risk band on the device numbers.
9. Medium: device 920 (N = 16, "not yet flat") against a farm of ~400-490 is ~2x, against
   1.6x on small. A gate on small alone ignores the v1 target map (tick budget,
   recommendation (ii)).
10. Farm IPC: per-unit crops over pipes are ≈ 170-200 KB per env-step at decimation 1. Use
    shared memory, not pickled pipes.
11. Farm workers must never create a CUDA context, whether through the CUDA-built module (the
    benchmark's "cpu" mode imports it) or through torch: that is ~300 MB of device memory each,
    on an 8 GB card.
12. Per-worker memory: GameMap allocates ~61 planes (25 int32, 12 int64, 10 float32 by
    `np.zeros` / `full` / `empty` count), plus the gas planes and solver scratch, ≈ 2-5 MB on
    small. RSS is dominated by the Python runtime (~150-300 MB, unmeasured), so 16 workers fit
    in 16 GB.
13. Per joule today (spec-sheet power): the farm at ~150 W (3700X PPT 88 W + platform) against
    the device at ~260 W (3070 TBP 220 W + one core) is ~1-1.3x. That makes no case for a
    joule gate.
14. Decimation leaves env-ticks/s unchanged but divides the obs, IPC and policy cost per
    env-tick. P1 and every gate should report the decimation they measured at.

## What the design should say instead

| quantity | design says | should say | assumptions |
|---|---|---|---|
| one core, small, physics only | "~196, game as-is (all idle)" | 196, physics only, no units | stacked probe N = 1 cpu, 11-07 |
| one core, small, the v1 match | "~160 with bots" | ~175 with units idle; ~155-160 with active units (step only, training-shaped); 142 with the bots' controllers | tick budget 5.9 / 6.4 / 7.0 ms, code ~3 % faster since |
| 8-core farm here | 1,300-1,500 | game: 1,000-1,450; physics only: 1,250-1,830 | 7-8 workers, all-core 0.9, SMT 1.0-1.3, obs / IPC excluded |
| 16-core Berzelius slot farm | 2,000-3,000 | 1,300-2,250 | 14 workers (2 cores for the trainer), Epyc 0.6-0.8, SMT 1.0-1.3; whether the 16 are physical cores is unknown |
| 3070 device, R3 at today's kernels | ~2,400, physics only | 2,400-3,400 (2,000 with the downside movers) | the allowance is generous in bandwidth; pre-stage inside (a); n_sub, second-run and policy against |
| A100 device, today's kernels | 5,000-14,000 | 4,800-10,900, throughput-bound stages only; latency-bound stages ×0.8 | 14,000 = 3,440 × 4 drops the allowance |
| ratio here, equal scope | ~1.6x | ~1.4-3.4x | M1 + M4 |
| ratio per Berzelius slot | 1.5-7x | ~2-8x | the flip alone may clear 3x on the target |
| R3 + throughput | 10,000-50,000 per slot | DRAM roofline at ~6 KB per cell per tick: 3070 ≈ 31,000, A100 40 GB ≈ 108,000 at peak; 10,000-50,000 = 9-46 % of the A100's roofline; planning at ~20 % ≈ 6,000 (3070) / 20,000 (A100) | my pass count, ±50 %; Nsight at P2 replaces it |
| training need | not stated | 1e8-1e9 agent decisions = 5e7-5e8 env-ticks | 4 v 4, 24 tps, decision every 4 ticks, 8 learning marines (self-play) = 2 decisions per env-tick; one learning team doubles the env-ticks |

**Wall-clock per training run** (η = 0.7 sim share of wall clock on both routes, the Sample
Factory 45-85 % band; device at today's kernels):

| route | pure env-ticks/s | 5e7 env-ticks (1e8 decisions) | 5e8 env-ticks (1e9 decisions) | GPU-hours for 5e8 |
|---|---|---|---|---|
| farm, this PC | 1,000-1,450 | 14-20 h | 5.7-8.3 days | — |
| device, this PC | 2,000-3,400 | 5.8-9.9 h | 2.4-4.1 days | — |
| farm, one Berzelius node (8 slots) | 10,400-18,000 | 1.1-1.9 h | 11-19 h | 88-153 (all 8 GPUs charged for their cores) |
| device, one Berzelius node | 32,000-88,000 | 0.2-0.6 h | 2.3-6.2 h | 18-50 |

At the brief's 1e9 decisions, the farm alone delivers a run in under a day on one Berzelius node.
The device route earns its twin debt on this PC, at 1e10 decisions (farm ≈ 880-1,530 GPU-h,
device ≈ 180-500), or across many runs. That is the anchor B1 asks the design to state.

## Questions that could not be grounded

- In a Berzelius allocation, are the 16 "cores" per A100 physical cores or hardware threads?
  The SMT factor depends on it.
- What share of the 70 ns/cell EOS is the host pre-stage? It needs a C++ timer or Nsight (P2).
- Does the CUDA second-run slowdown persist under in-place masked resets, which allocate
  nothing?
- Why does the batched sweep reach only ~3 % of DRAM bandwidth? Candidates: occupancy (≤ 60 of
  256 lanes active per block, 16 × N blocks per launch), the four int64 per-cell books shared by
  ordinates whose wavefronts cover the same column, or something else. Only Nsight Compute can
  say.
- How much does a farm worker on the small ship actually use (RSS)?
- What training scale (decisions per run × runs) and what Berzelius allocation (GPU-hours per
  month) does Erik plan? B1's factor needs both.
