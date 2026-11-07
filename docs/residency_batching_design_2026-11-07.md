# GPU residency + batching — design (2026-11-07, v1)

Issue #29 (the umbrella; #77, #24, #25 and #71 ride it). Lead: Claude under
Erik's mandate of 2026-11-07. Inputs, in reading order:
`docs/residency_where_we_stand_2026-11-07.md` (the tick today),
`docs/residency_benchmark_2026-11-07.md` (the numbers),
`docs/research/batched_gpu_sims_litsearch_2026-11-07.md` (the literature),
`docs/rl_env_arc_proposal_2026-08-27.md` (B1–B6 and the §A habits, which this
design supersedes as the plan while keeping the habits),
`docs/architecture/engine/02_state_and_ownership.md` (canon: fields migrate
behind GameMap; this design revisits "CPU owns the actors").

## 0. Summary

The arc builds ONE batched environment API with two backends. The `cpu_farm`
backend runs the game exactly as played in N processes. The `device` backend
runs N games per launch with every per-tick computation on the GPU and the
host doing only per-episode and rare-event work. The API, the observation /
action / reward tables and the gates are shared, so RL work starts on the
farm the day the API exists and moves to the device backend stage by stage,
each stage bit-identical to the reference and measured against the farm.

What the benchmark says (RTX 3070, 2026-11-07): at N = 1 the resident tick is
bound by launches, copies and host brackets, not by device compute; batched,
the device EOS reaches 70 ns per cell per tick (25x one CPU core), but a host
floor of about 400 ns per cell (combustion, the tail's per-call twins, the
bus, the wall scan, the slots) and Python gameplay of about 3 ms per env-tick
cap the tick. With the host stages gone, physics alone gives about 2 400
env-ticks/s on the small ship per 3070; one CPU core already gives about 196
with the game as-is. Residency and batching are therefore necessary but not
sufficient: the GPU route pays only after the authority flip (gameplay on the
device) and kernel throughput work. Both are in this design, after the farm,
and the arc carries an explicit stop condition (§3).

Decided with Erik on 2026-11-07: snapshots with catch-up (training runs on a
tagged version in its own checkout and build; game development continues on
the reference path; device twins catch up in batches); zombie pathing may
become a distance field.

## 1. Goal and non-goals

**Goal.** `BatchedEnv.step(actions)` advances N games and returns
observations, rewards and terminals as tensors with a leading N axis. The
device backend does it with zero host round trips per tick; the farm with N
processes. Env i of N is bit-identical to env i alone and to the played game
at the snapshot tag. Throughput is measured in env-ticks per second per GPU
slot (one A100 + 16 Epyc cores on Berzelius; one RTX 3070 + 8 cores here).

**Non-goals (the agent era).** The observation design beyond the tables'
mechanism, the action space beyond the intent verbs, the reward, the
trainer, frame skip, per-marine vs per-team policies, policy placement on the
cluster. The design makes both agent shapes fit (`rl_ready_scope` §3) and
decides none of them; its tables carry PLACEHOLDER rows.

**Out of this arc (own issues).** A human control scheme (a ControlSource,
unconstrained here); the thermal overhaul #68; wildlife #63; the window and
the renderer (headless only here); a cheaper "training physics" (§12.3).

## 2. Inputs, in numbers

### 2.1 Where the tick's time goes (medium ship, bots, resident, ms per tick)

| part | ms | note |
|---|---|---|
| controllers (bots + horde) | 1.5 | Python; A* + vision; replaced by the policy in training, the horde by a kernel |
| gameplay slots (step minus slot 7) | 1.9 | Python; stamp_units 0.5 (C++), 9b walls 0.4, movement 0.3, 9c/9c2 0.3, zombies 0.15 |
| EOS stage | 3.7 | device chain + its host pre-stage, one C++ call |
| trace tail | 0.1 | device |
| bus | 0.8 | 11 uploads + 7 downloads, ~0.8 MiB (measured) |
| combustion | 0.6 | host (#71) |
| step_tail | 2.9 | host ripple + per-call fire / temperature / sweep twins: ~21 malloc/free, ~1 MiB of copies (estimate) |
| **total** | **11.5** | 3.6x real time |

### 2.2 How the cost scales (stacked small ship, resident, ns per cell per tick)

| stage | N = 1 | N = 32 | N = 128 | behaviour |
|---|---|---|---|---|
| EOS (device chain + host pre-stage) | 1352 | 97 | 70 | flattens: device-bound |
| bus | 188 | 28 | 24 | flattens |
| combustion (host) | 79 | 116 | 116 | linear: host |
| step_tail (host ripple + per-call twins) | 843 | 232 | 657 | linear, then the stacked sweep's launch artefact |
| 9b walls + other slots + stamp | 200 | 49 | 47 | linear: host |
| whole tick | 2740 | 530 | 918 | |

The batched radiation sweep core: 0.12 ms per small env-tick, 3.2 ns per
cell-update, 63 launches per tick whatever N, bit-identical to the CPU sweep.

### 2.3 The yardstick

| route | env-ticks/s, small ship | basis |
|---|---|---|
| one CPU core, game as-is | ~196 (all idle), ~160 with bots | measured |
| 8 cores (this PC), multiprocess | ~1 300 to 1 500 | extrapolated; P1 measures it |
| 16 cores per A100 slot (Berzelius thin node) | ~2 000 to 3 000 | extrapolated (Epyc 7742 core 0.6 to 0.8x a 3700X core, SMT 1.0 to 1.3x: assumptions) |
| one RTX 3070, host stages gone, physics only | ~2 400 (EOS + sweep + an allowance) | estimated from the batched stages |
| one A100, same kernels | ~5 000 to 14 000 | 2x on integer-bound kernels, 3.5 to 4.5x on bandwidth-bound; unmeasured |

Per Berzelius slot the GPU route at today's kernel efficiency is 1.5 to 7x
the farm, before the policy shares the GPU; per slot here it is about 1.6x.
Both are physics-only; the farm's gameplay is inside its 196.

### 2.4 What the literature adds (`docs/research/batched_gpu_sims_litsearch_2026-11-07.md`)

- Unit store as SoA columns with a fixed `(N, max_units)` capacity and a
  `valid` mask (Madrona's special case); each column is the tensor torch
  reads in place. Reset through a per-env mask read inside the step.
- Gameplay never as one thread per env (Madrona measured 1.9 % utilisation
  for that shape): one thread per unit row across all envs, one block per env
  where a slot needs in-env cooperation, separate kernels per stage.
- The step contract is Isaac Lab's: `step(actions)` runs `decimation` ticks,
  returns `terminated` and `truncated` separately, resets done envs inside the
  same step by mask and returns post-reset observations.
- At training batch sizes launches are a few percent of a tick (~150 kernels
  as a graph: ~80 us device floor); capture the tick as a graph anyway (#24's
  spike is bit-identical), with a WHILE node over the device max of `n_sub`;
  no megakernel, fuse only where data stays on chip (the MG coarse tail
  already is: `mg_fused_tail`).
- The sweep's 63 launches per tick become ONE: one block per (env, ordinate),
  the wavefront loop inside the block, the previous wavefront's outflow in
  shared memory (the KBA / Kripke shape); arithmetic unchanged, so gate 0
  holds.
- Multigrid: a per-env hierarchy on the `(N, h, w)` layout, fine levels as
  wide kernels, the fused coarse tail one block per env; the fixed 8 V-cycles
  keep envs in lockstep; never one tall grid.
- Integer throughput: the A100 is ~2x a 3070 on integer-bound kernels; the
  128-bit staging (`__mul64hi`) is the costly primitive; a narrowing is an
  arithmetic change that lands in the CPU reference first and only after
  Nsight Compute shows the kernel compute-bound.
- Distance fields: one block per (env, field), the frontier loop inside the
  block, int16 distances in shared memory (35 KB for 17 600 cells); fixed
  neighbour order; refresh every K ticks.
- Deterministic parallel logic: #169's two-phase shape is the device shape;
  the serial walk-order apply needs a device form; every draw a counter-based
  RNG keyed (seed, env, tick, slot, purpose). No surveyed system gates batch
  invariance; Breach builds that gate itself.
- Zero-copy: DLPack from persistent preallocated buffers; the sim on torch's
  stream (`Stream.from_external()`) or an explicit event wait; the two memory
  pools capped on the 8 GB card.

## 3. The routes and the choice

| | R0 CPU farm | R1 resident physics, Python gameplay | R2 resident + batched physics, C++ gameplay on the host | R3 everything per-tick on the device | R3 + kernel throughput |
|---|---|---|---|---|---|
| env-ticks/s per GPU slot, small | 2 000 to 3 000 | below R0 (Python 3 ms per env-tick) | 2 000 to 5 000 | 2 000 to 5 000 (3070 now ~2 400) | 10 000 to 50 000 |
| port cost | none | small | large: a second gameplay copy in C++ | large: gameplay kernels | plus kernel engineering |
| twin debt per gameplay change | none | none | yes | yes | yes |
| game dev continues in Python | yes | yes | yes | yes | yes |

**Choice.** R0 first, as the batched API's first backend: cheap, the
yardstick, and RL work can start. Then the device backend in stages:
residency finished (the brackets gone), the physics batched, the authority
flip (R3), then throughput. R2 is rejected: a C++ host copy of gameplay would
be a third copy once the kernels exist. R1 is a state the build passes
through, never a shipped one.

**Stop condition (Erik rules, §12.2).** The device backend is called done
only when, on this PC, it beats the MEASURED 8-process farm by at least 3x on
the small ship at equal game (P9's gate). If the kernels do not deliver, the
arc stops honestly at R0 plus finished residency, which is still the played
game's fastest single-instance path.

## 4. Architecture

### 4.1 The batched environment API (one API, two backends)

`src/rl_env.py` (provisional): `BatchedEnv(bank, n_envs, backend, tables)`.

- `reset(mask=None, seeds=None)` re-initialises the masked envs from the
  level template bank (§4.6) with the given episode seeds. Reset is by MASK,
  never by an id list.
- `step(actions)` runs `decimation` sim ticks (`[rl] decimation`, 1 until the
  agent era sets it) for every env; returns `obs`, `reward`, `terminated`,
  `truncated`, `info`, each with a leading N axis; envs that ended are reset
  inside the same call and return post-reset observations (the terminal
  observation is kept in `info`).
- Observations, rewards and terminals are table-driven (§4.5); the same
  config rows produce the same tensors on both backends.
- `backend = "cpu_farm"`: N worker processes (the GIL forces processes), each
  one `Simulation` on the reference path (`run_match`'s loop without the
  report), driven over pipes; obs assembled in the worker from the sensor
  accessor's channels and the unit list; stacked on the host and moved to
  torch once per step.
- `backend = "device"`: the batched resident engine (§4.2 onward); the
  tensors are persistent CuPy buffers exported zero-copy (§4.10).
- **Backend equality gate:** for the same bank, seeds and action stream, the
  two backends return the same obs and the same per-env tick digests. This
  is the oracle for the whole arc.

The API is Isaac-shaped so rl_games / skrl / PufferLib-style trainers consume
it; its first consumer is a smoke driver that steps random actions and runs
the equality gate. The trainer is the agent era's.

### 4.2 The batched resident engine

- Every plane is `(N, h, w)` (gas `(N, g, h, w)`, the light triples
  `(N, h, w, 3)`); every launch core takes `N`; per-env scalars are device
  arrays of length N read by env index. This applies the §A habit to the
  cores born `(h, w)`: water, the EOS substeps, MG, energy flux, kick /
  compression, temperature, fire, combustion. Scratch keyed `(N, h, w)`.
- A BATCH IS HOMOGENEOUS: one grid shape, one boundary type (space or
  ambient), one config (tick rate, material and gas tables); the level
  template may differ per env. Different shapes are different batches.
  v1 training batches are space maps; the sky exchange stays a host bracket
  on ambient maps, outside the device backend (accepted gap).
- The EOS pre-stage reductions (`t_max_abs`, `max_rad`, `max_du` → `n_sub`)
  become per-env device reductions (an integer max is order-free, so a
  parallel max is exact); the substep loop runs to the batch's maximum
  `n_sub` with per-env masks, a masked env doing exactly nothing. Under a
  CUDA graph the loop is a WHILE node over the device max (CUDA 12.4+).
- The MG hierarchy is per env on the `(N, h, w)` layout: fine levels as wide
  kernels over `(N, cells_at_level)`, the fused coarse tail one block per
  env. Never one tall grid (the stacked probe's coupling and launch
  artefacts).
- The radiation sweep's resident core is already `(N, h, w)`; its driver
  moves into the resident tick (P4) and its schedule becomes one block per
  (env, ordinate) with the wavefront loop in the block (P9).
- No host sync inside a tick. The tick is one launch sequence on one
  non-blocking stream per batch; the only waits are the events torch waits
  on when it reads (§4.10).
- `ripple` / `ripple_v` (render-only float, excluded from the digest) are not
  computed on the device backend (`headless`: `step_ripple` skipped). Nothing
  in the sim reads them (accepted gap, recorded).

### 4.3 Authority: what lives where

| runs every tick for every env | device |
|---|---|
| physics: water + its tail, EOS + traces, trace tail, combustion, temperature, fire, radiation sweep | P4, P5 |
| 6 stamp_units (5 planes), 9b over-pressure walls, 9c heat damage, 9c2 wind, 9c3 gas rows, 9d ignition, 2b statuses | P6 |
| 3 movement, 5 zombie AI, 4 / 4b / 4c shooting + sprays + resolve, 2 projectiles and grenades in flight | P6, P7 |
| 9e entities: vents / ducts / pumps, heat sources, doors, timed charges, sensors + the signal bus | P6b, per class as SoA rows |
| 6b FieldEdit flush | the device edit queue, P4 |

| runs rarely or once per episode | host |
|---|---|
| reset: template copy, spawn placement, seeding | the host decides, the device copies (P8) |
| blasts and other payload executions (`execute_payload`: wall damage, the reach walk, the deposits) | the host, on a windowed read of the planes the walk needs, writing through the edit queue (§4.8) |
| level topology changes (`destroy_wall`, `seal_tiles`: the blasts' and burn-through's events) | the host, as edits plus mask patches |
| match end | a `done` kernel per env (the MatchRules law); the host reads `done` |
| the recorder, logging | the host, from exported tensors |

Kernel mapping for the gameplay slots (lit §2): one thread per unit row
across all envs for per-unit work (movement, statuses, heat / wind / gas
rows, ignition reads); one block per env where units interact inside an env
(shooting's phase A, line of sight, the zombie descent); separate kernels per
stage. Rare host events are frequency-bounded: a batch with M envs holding a
blast this tick pays M windowed reads (medium bots: ~2 frag throws per
722-tick match). Accepted gap: a scenario that blasts every tick runs
host-bound; no v1 scenario does.

### 4.4 The device unit store

`(N, U, cols)` SoA with a `valid` column; U = the batch's `max_units` (the
sum of `max_per_team` over the teams plus the zombie cap), fixed at bank
build. Columns come from a SCHEMA (`src/simulation/unit_schema.py`,
provisional): every synced field of `Unit` and of a status row is a named
column with a dtype (int32 Q16 or int64) and a digest flag; the schema is the
one table the digest, the serializer, the obs gather and the kernels read.
Legacy WEGO fields (orders, plan, ap, overwatch, belt, action cooldowns) are
NOT columns: a match never uses them; the CPU reference keeps them.

Columns (v1): id, slot, team, is_zombie, valid / alive / life_state, x, y
(Q16, §7), facing (Q16), footprint id, hp, species / loadout / env-profile
ids, weapon id, last_fire_tick, magazine state, reload_done_tick, spray
ticks, zombie activated / descent cursor / move accumulator, last_melee_tick,
cadence, move accumulators, the status rows (kind, magnitude, remaining) as a
fixed-width block, the exposure state the coupling rows keep. Python `Unit`
objects do not exist on the device backend; the CPU backend keeps them.

### 4.5 Actions, observations, rewards, terminals

- **Actions** = the intent buffer `(N, U, A)` int32: move dx, dy (a Q16 unit
  vector), speed_mode, aim dx, dy, trigger, throw dx, dy, reach, use. These
  are `match_replay.INTENT_KINDS` minus `drive_zombies` (the horde is a
  kernel). A new verb is a column in the intent schema and a row in
  `INTENT_KINDS`, never a kernel rewrite. The CPU backend applies the same
  buffer through the intent facade, so the recording tap sees it and a
  training episode is replayable (§5.3).
- **Observations** = tables (`[rl.obs]`): a per-unit CROP `(N, U, C, r, r)`
  of C planes around the unit (the sensor accessor's seven channels plus
  material / solid / unit occupancy by team / light where requested), a
  per-unit VECTOR `(N, U, V)` of the unit columns the table names, a per-env
  GLOBAL map `(N, C_g, h/k, w/k)` for the team shape. The policy wrapper
  composes a team view as the union of its units' crops. The gather converts
  Q16 to float inside the kernel; sim state stays integer.
- **Rewards and terminals** = tables over device counters: per-unit event
  counters `(N, U, E)` (hits dealt / taken, kills, deaths by cause,
  team-kills, exposure ticks) written by the kernels that emit UnitHit /
  UnitKilled today, with the credited attacker; `[rl.reward]` rows weight
  the counters; `terminated` per env from the MatchRules law (fewer than two
  marine teams alive), `truncated` from the time limit.

### 4.6 Reset and the level template bank

A bank is built offline by `tools/build_level_bank.py` from level folders of
one shape: the static planes (material, solid, flammable, floor_height, the
masks, fuel, the optics planes, `air_init`, `temp_init`) as `(L, h, w)`
arrays, the spawn cells per team per level (the formation rule, computed on
the host once), the entity rows. `reset(mask, seeds)` copies template
`template_index[env]` (a device array, set by the host from the seeds) into
each masked env, places the units, zeroes the dynamic planes, re-derives the
derived planes (gas energy from the seeds, the mirrors) and sets the env's
Philox key. Level generation stays offline; the bank is a file that records
the build id of the generator that wrote it.

### 4.7 RNG

Every sim draw on the device is Philox-4x32-10 from the vendored kit
(`cpp/src/philox32.h` + `src/simulation/philox32.py`, chapter 17 §3's
layout): key = the env's episode seed, counter = (tick, draw_index,
stream_salt, unit_or_site_id). The CPU reference makes the SAME draws through
the Python twin, replacing `sim.rng` at the draw sites a match runs (combat
spread, FieldEdit noise, charges, heat sources, timed charges). `sim.rng`
stays for WEGO's timeline and the ruleset's legacy draws. A golden
re-baseline (§7).

### 4.8 The device edit queue and the rare-event path

`EditQueue.flush` is relocated: the host (or a kernel) enqueues records
`(env, field, region, coords, value, mode, policy)` into a pinned buffer; one
kernel applies them in the queue's stable sort order with the same per-field
policies and combine functions, ported as device functions. Semantics
unchanged; the host never reads a field to apply an edit. A blast's
`execute_payload` runs on the host on a windowed read (the open mask and the
two air planes over its reach, D2H for that env only), computes its deposits
and wall damage as edits, and enqueues them.

### 4.9 Pathing: the distance field

`pathfinding.astar` stays THE pathfinder for host callers (the bots, tools).
On the device, zombies walk a PER-ENV DISTANCE FIELD: integer BFS distance to
the nearest living marine over the passable mask, one block per (env,
field) with the frontier loop in the block, refreshed every `[zombie]
field_refresh_ticks`, descended by comparing the four neighbours in a fixed
order. Erik approved the behaviour change; it lands on the reference path
first as a HUMAN-TEST (§7). The bots keep A* on the host in v1 (they are not
in the device backend's hot loop); a device bot on per-goal fields is a
later patch.

### 4.10 Zero-copy export, streams, syncs

Persistent CuPy buffers; torch reads them through DLPack
(`__cuda_array_interface__` works today). The engine launches on torch's
current stream (`cupy.cuda.ExternalStream`) or the API records an event
after `step` that the torch side waits on; never a device-wide sync. The host
reads only `terminated` / `truncated` each step (a tiny D2H). Both memory
pools are capped (`[rl] device_pool_mib`).

### 4.11 CUDA graphs and kernel throughput (stage C, P9)

After batching, the tick is a fixed launch sequence per `(N, h, w)`: capture
once into a CUDA graph (#24; capture on the non-blocking stream, thread-local
capture mode, as the 2026-08 spike found), a WHILE node for the substeps,
replay per tick. Then the measured items, in this order: the sweep as one
block per (env, ordinate) (63 launches → 1); the EOS chain's pass count and
fusion where data stays on chip; the 128-bit multiplies only after Nsight
Compute shows a kernel compute-bound, as a CPU-reference-first arithmetic
change. Targets come from an elevated Nsight capture (P2), not from guesses.

## 5. Determinism and gates

1. **Reference bit-identity.** Every device stage equals the CPU reference
   at tol 0 on every synced field and every book (`cuda_*_check.py`;
   `cuda_s8a_check.py` extended per stage).
2. **Batch invariance.** Env i of N equals env i alone, bit for bit, for N
   in {1, 2, 8, 64} and for envs holding different templates, seeds and
   substep counts. Per-env masks make it true; integer atomics make the
   books order-free.
3. **Replay-driven lockstep (the gameplay twin oracle).** A match recording
   (#126) replays on the CPU reference and on the device backend; the
   per-tick `(field_digest, tick_digest)` agree at every checkpoint.
   Recordings from played matches (ship_demo seeds 1–8, the freighters) are
   the corpus; a drifted twin fails here.
4. **Backend equality** (§4.1).
5. **The two-phase apply.** Phase A decides in parallel against the
   start-of-tick state and records; conflicts resolve by integer reductions
   (`atomicMin` on (key, slot)); the walk-order apply runs as ONE thread per
   env looping over that env's recorded effects in walk order (a handful per
   tick), which is batch-invariant and matches `SimultaneousEffects.resolve`.
6. **#71** fixed first (P2): the CUDA combustion twin onto the #54 seam, a
   burning-scene gate at tol 0, validated by breaking it.
7. **Digest spec.** The unit store's synced columns enter the entity section
   (a version bump, goldens regenerated in the same commit); obs tensors are
   not digested (float, derived).

## 6. The snapshot system and the twin-debt rule

- A training snapshot is a git tag `train/<name>` on main (on the arc branch
  while the arc runs), checked out in its own folder (a worktree or a
  clone) with its own build. `BREACH_BUILD_ID` (`git describe` at configure)
  is compiled into the module and exposed as `bp.build_id()`.
- `rl_env.require_snapshot(tag)` REFUSES unless HEAD is exactly the tag, the
  tree is clean, `bp.build_id()` equals the tag's describe, and the bank's
  recorded build id matches; every training entry point calls it. Berzelius
  clones at the tag and builds in the container, so the same check holds.
- Twin debt: a reference-path change that breaks a device twin's gate is
  marked `xfail` naming the catch-up issue in the same patch, never deleted,
  never silently re-baselined; `tests/test_twin_debt.py` asserts every such
  mark names an open issue. The catch-up patch removes the marks.

## 7. Reference-side changes first (each a deliberate re-baseline)

| change | why | gate | feel |
|---|---|---|---|
| sim draws through Philox at the match draw sites | per-env keyed draws; no shared stream | goldens regenerated; the kit's KAT + layout tests | none |
| unit positions to Q16 (`Unit.x/y` integers; the slide / round math in integer) | the device store is integer; both paths must agree bit for bit | goldens regenerated once; movement A/B in dequantized units | HUMAN-TEST: Erik plays a match |
| zombies on the distance field | per-unit A* has no device shape | goldens; ship_demo seeds 1–8 before / after | HUMAN-TEST |
| the unit schema (synced columns) | the store's schema | digest version bump | none |

Order as listed; each merged on green with its re-baseline rationale written.

## 8. Linux, the container, Berzelius

- `cpp/CMakeLists.txt` already carries the gcc/clang strict flags
  (`-fno-fast-math -ffp-contract=off`) and `--fmad=false`; the CUDA arch
  list (75 / 86 / 89) gains 80 (A100) and 90 (the Hopper phase).
- `cpp/build_linux.sh` (CMake + Ninja; the conda env from `environment.yml`)
  and `containers/breach.def` (CUDA 12.4+ devel base for the conditional
  graph nodes, micromamba, the env, the build) are the patch; the suite on
  Linux must pass and the digests must equal Windows's.
- First Linux box: this PC once SVM is enabled in the BIOS and WSL2
  installed (Erik, 2026-11-07 evening), else the Lenovo (WSL2 + Ubuntu +
  RTX 1000 Ada), else the cluster's login node.
- Multi-GPU: one process per GPU, N envs each; the engine never assumes one
  GPU. Policy placement (one A100 shared vs split) is the agent era's.

## 9. Patch plan

Each patch: one worktree agent, commits as it goes, merged `--no-ff` on green
by the agent that ran the gate (HUMAN-TEST excepted). Tiers per the
autonomous-patch-workflow rule: oracle-gated → Sonnet, judgement → Opus.

| # | patch | gate | tier | HUMAN-TEST |
|---|---|---|---|---|
| P1 | `BatchedEnv` v1 with the `cpu_farm` backend; `[rl.*]` tables (placeholder rows); the random-action smoke driver; `require_snapshot`, `BREACH_BUILD_ID` / `bp.build_id()`; `test_twin_debt`; the 8-process farm MEASURED (the yardstick) | N workers reproduce `run_match`'s digests per seed; the smoke run; the farm number in the benchmark doc | Sonnet | no |
| P2 | determinism gaps: #71 (the combustion twin onto the #54 seam); the CUDA second-run slowdown measured; an elevated Nsight capture of the resident tick (Erik runs the command; the agent reads it) | burning scene tol 0, validated by breaking it | Opus | no |
| P3 | reference-side changes (§7), one commit each: Philox, Q16 positions, the distance field, the unit schema | goldens + A/B + replay corpus re-pinned with rationale | Opus | yes (positions, zombies) |
| P4 | finish residency: resident combustion / temperature / fire / sweep drivers, the water tail, stamp_units kernel, the device edit queue, `headless` ripple | `cuda_s8a_check` extended, tol 0; zero per-tick plane transfers (the profiler's bus counter reads 0) | Sonnet | no |
| P5 | batch the physics: `(N, h, w)` on every core, per-env scalars, per-env MG, per-env `n_sub` with masks, device pre-stage | N = 1 == resident; batch invariance | Opus | no |
| P6 | device unit store + the per-tick unit slots (movement, stamp, statuses, the coupling rows, 9b walls, 9d ignition) | replay lockstep | Opus | no |
| P6b | entities on the device (vents / pumps, heat sources, doors, charges, sensors + bus, the #25 gather) | replay lockstep | Sonnet | no |
| P7 | combat on the device (shooting phase A / B, sprays, projectiles, grenades; the host rare-event path for blasts), the horde on the field | replay lockstep | Opus | no |
| P8 | reset kernel + template bank + obs / reward / terminal kernels + zero-copy export = the `device` backend | backend equality | Opus | no |
| P9 | throughput: the graph, the sweep's single launch, EOS fusion | bit-identity; the stop condition (≥ 3x the measured farm on the small ship) | Opus | no |
| P10 | Linux build + container + the suite on Linux | digests equal to Windows | Sonnet | no |
| close | tag `train/v1`; CLAUDE.md rows (§13 b); skill amendments; #29 / #77 / #24 / #25 / #71 closed; Roadmap #46 | | | |

P1 and P2 are independent of each other and of P3; they run first and in
that order. P3 before P4 so every later gate compares equal games.

## 10. Rules for continued game development (after the port)

- A gameplay change is made on the REFERENCE path first (Python / C++), as
  today; the played game never waits for a kernel.
- A new per-unit field is a schema row; a new verb is an intent column; a new
  observation plane or reward term is a table row. None is a kernel edit.
- A new per-tick gameplay rule that the training path needs gets its kernel
  twin in the catch-up patch; until then its reference test is `xfail`
  naming the catch-up issue (§6). A rare event (a new payload, a topology
  change) is a host handler writing through the edit queue and needs no twin.
- Training runs on a `train/*` tag only; the launcher refuses anything else.

## 11. Accepted gaps

- Ambient (planetside) maps are outside the device backend (sky exchange).
- Ripple is not computed on the device backend.
- Blasts and topology changes run host-side per event on a windowed read.
- The rule-based bots stay host-side on A*; the device backend's baseline
  opponents are the horde and, later, a device bot; self-play needs neither.
- The field recorder does not run on the device backend; replay is
  re-simulation from the intents.
- WEGO and its timeline are untouched and absent from the device backend.

## 12. Rulings needed from Erik (scoped)

1. Unit positions to Q16 on the reference path (§7): a feel re-baseline he
   plays once.
2. The stop condition (§3): the device backend is "done" only at ≥ 3x the
   measured farm on the small ship; otherwise the arc stops at R0 plus
   residency.
3. Whether training may ever run a cheaper physics preset than the played
   game. Recommendation: no, by the snapshot principle; the curriculum uses
   the small map instead.

## 13. Systems

**(a) Existing canonical systems this design uses:** Simulation facade; tick
conductor (its slot order is THE order the kernels keep); GameMap (fields
migrate behind it); FieldEdit (the device queue is its flush relocated);
payload executor (the host rare-event path); gas pump primitives (the vent
kernels call the same law); PhysicsRunner / PhysicsEngine (the batched step
is engine orchestration, never Python glue); fixed-point kits (device unit
math); Q16 boundary modules; trace currency; config (the `[rl.*]` tables,
bound once); material / gas tables; entity system + serializer + sensor
accessor (the obs channels; #25's gather); recorder (unchanged); field digest
+ golden + A/B lockstep + CUDA harness (every gate); control seams (the
intents are the action schema; ControlSource unchanged); match config /
spawn rooms (the bank's spawn cells); match recording / replay (the twin
oracle); credited cause (the reward counters); simultaneous attacks (phase
A / B maps to the kernels); A* (host); unit / loadouts / env profiles (the
schema's source); headless match runner (the farm worker's loop); benches
(+ `tick_stage_profile.py`, `bench_resident_scaling.py`, the batched sweep
bench); radiation sweep (its batched core exists); gas energy seam + closure
identity (every device stage books as the CPU does); Philox (vendored from
the swarm branch, chapter 17 §3).

**(b) New systems and their draft rules (enter CLAUDE.md at implementation):**

1. **Batched environment** (`src/rl_env.py`): THE one env; two backends,
   one API, one table set; a trainer never touches a Simulation or a kernel.
2. **Batched resident engine**: every launch core `(N, h, w)`; per-env
   scalars as device arrays; a batch homogeneous in shape / boundary /
   config; batch invariance is a gate; never a tall grid.
3. **Device unit store + unit schema**: schema-driven SoA `(N, U, cols)`; a
   new unit field is a schema row (dtype, digest flag), never a kernel edit.
4. **Device edit queue**: the FieldEdit flush on the device; the host never
   reads a field to apply an edit; rare events enqueue.
5. **Observation / reward / terminal tables** (`[rl.obs]`, `[rl.reward]`,
   `[rl.done]`): a task is rows, never a bespoke kernel.
6. **Level template bank** (`tools/build_level_bank.py`): reset copies;
   level generation never runs mid-training.
7. **Distance-field pathing**: THE device pathing; one kernel shape; A*
   stays the host pathfinder.
8. **Philox draws**: the device RNG and the reference's match draws;
   `sim.rng` only for legacy paths.
9. **Training snapshot + `require_snapshot`**: tag + checkout + build id,
   refused on mismatch.
10. **Twin-debt rule** + `test_twin_debt`.
11. **Build id** (`BREACH_BUILD_ID`, `bp.build_id()`).
12. **Linux build + container** (`cpp/build_linux.sh`, `containers/breach.def`).
13. **Tick stage profiler** (`tools/tick_stage_profile.py`): THE per-slot /
    per-stage / bus instrument; a new per-tick cost question extends it.

## 14. Critique lenses (verdicts written here, one critic at a time)

| lens | verdict | blockers | resolution |
|---|---|---|---|
| determinism / residency: bit-identity, batch invariance, masks, atomics, the two-phase apply, RNG, syncs | **NOT SOUND as written** (critique `dd058b6`, `docs/residency_batching_critique_1_determinism_2026-11-07.md`): the architecture holds (batched cores, per-env masks, integer atomics, the two-phase shape, the replay oracle) but two named mechanisms cannot deliver §5's claims | **B1** §4.3/4.8/5.5: phase B carries host-only effects (`chew_wall`, `execute_payload`, `destroy_wall`) and device slots raise topology events every slot (burn-through, 9b bursts, doors, fuse-outs, charges, heat-source deaths), so "zero host sync" and the one-thread-per-env walk-order loop both fail; fix: `destroy_wall` / `seal_tiles` / `on_tile_changed` / the wall law as device functions with per-env atomics, and for payloads either a device blast kernel or a declared stop/resume split with a reworded claim; gates for a burst, a burn-through, a chew-to-zero, a door toggle. **B2** §4.5/5.3/5.7: the device cannot compute `tick_digest` (events hashed in emission order with float damage; WEGO `ap` / `n_orders` hashed; `round(v·1e9)`); fix: a per-env device event log in the reference's emission order, `_unit_record` redefined from the schema in P3's commit with the corpus re-pinned, ONE digest function over a D2H of store + log. **Majors** M1 `n_sub` drives two loops and four host double folds (fold table per batch, device-built planes, both loops masked); M2 interim substep control before P9 (8 masked iterations or one counted 4-byte D2H); M3 the books need a per-env `(N, slots)` int64 home with assign/accumulate semantics and tick-start reset; M4 the edit queue needs a deterministic composite `seq` key, serial-per-tile application, a decision on its float64 combines (integer FieldEdit at the door as a §7 re-baseline, preferred), seam atomics, a named Philox counter for the noise; M5 the payload path reads/writes far more than stated (wall_hp, units, gas through the pump primitive, `div_source`, `expansions`, TickCredit, `blast_log`); M6 many float64 decision sites in combat / damage / zombie AI and the kit has no double draw — inventory each: IEEE transcription or integer re-baseline; M7 no stream is threaded, ~200 sync memcpys + 18 device syncs, RAW/WAR hazards on action/obs buffers, scratch outside the pools; M8 in-step auto-reset needs per-env seeds as a pure function of (run seed, env, episode) on the device or batch invariance breaks from episode 2; M9 the bank must store the post-`spawn_match` tick-0 state (digest-verified), never re-derive; M10 three serial laws (bites, the vent ledger, sprays after shooting) need per-env serial stages. Minors: draw-site list corrections (charges never draw; exposure / crit / melee / beam cone do), per-(slot, site) salts, water dormancy as a per-env mask, per-env failure flags instead of throws, float32 schema dtypes for the permeability / absorb stamps, Q16 rounding spelled out, two host whole-plane rebuilds per tick, gate additions | pending v2 (synthesised after all four lenses) |
| systems reuse / scope: does §13 miss a canonical system; is anything a parallel copy; what belongs to another arc | **NOT SOUND as v1** (critique `214deed`, `docs/residency_batching_critique_2_systems_scope_2026-11-07.md`): the route holds; three things change on paper before any patch: the device form of the topology seam, an oracle the device can compute, the patch plan | **B1** §4.3/§11: topology changes on the device backend go AROUND the booked seam — `destroy_wall` carries the mass-books bracket, reservoir evacuation, `gas_energy_retire`, the ambient seed, the `on_tile_changed` diff; `seal_tiles` / `unseal_tiles` book solid energy; callers are per-tick (chew, doors / airlocks, slots 9 / 9b, `apply_explosion`); slot 9 unassigned; fix: ONE device form of the seam, gated by a device twin of `test_gas_energy_tile_flip` + the closure identity on a burst / door cycle / chew scene; amend the GameMap row; assign slot 9. **B2** §5.3/§5.7: the device cannot compute `tick_digest` (WEGO `ap` / `n_orders`, 1e-9 positions, an ORDERED event stream); units are NOT in the entity section, so §5.7's procedure is wrong; fix: a per-env per-tick event ring written by the walk-order apply (the digest's event list AND the reward counters derive from it), `_unit_record` from the schema with `ap` / `n_orders` settled, a tick_digest re-pin with rationale. **Majors** M1 §13(a) misses ~20 canonical systems (vent ledger, heat source + `blast_log`, timed charge, coupling table / wind load / footprint samples, vision / LOS, weapons, starting a fire, the five explosion rows + div_source, controller registry, light accessor, ingress lint / float ratchet, RL-batch habits, `sweep_ref_q`, extinction planes, prop stamps, level data layer + LevelSpec, `unit_fixed`); M2 a §13(c) AMENDMENTS list is missing (15 CLAUDE.md rows incl. facade "only tick entry", conductor "ONE driver", GameMap, FieldEdit, payload executor, A*, RL-batch habits, Unit, replay "ONE tap", runner "never a second loop", spawn "ONE match spawn", recorder, sensor accessor; canon engine/02, ml/01 §1 "no AI-mode fork", engine/10, mechanics/02, engine/14), each assigned to a patch; M3 the obs crop must be an accessor METHOD with #25's kernel as its device backing, and the GLOBAL map is an agent-era ruling (drop from v1); M4 a light channel pre-empts #81 (refuse at the obs door; ACCEPTED GAP); M5 `[rl.*]` in config.toml overlaps match.toml and ties every reward experiment to a snapshot — obs / reward rows go in a door-checked RUN FILE outside the snapshot, the bank carries each template's MatchSpec, the policy is a controller name; M6 the farm worker is a second match loop and the policy is not a controller — factor `run_match` into a steppable session, register a `policy` controller issuing action rows through the intent facade; M7 device episodes must export as `match_replay.Recording` v1 so a trained agent can be WATCHED (gate: replays under `run_match --replay`); M8 the action buffer must derive from INTENT_KINDS (fuse_seconds float → §7, `clear_move_dir` ≠ zero vector, hold vs latch semantics per decimation tick); M9 Q16 positions are not forced by the stated reason (float64 columns reproduce with `--fmad=false`; the real reason is the iron rule) and their consumers (renderer, bots, `_unit_record`, the recorder's frozen `unit_fx` key) are uninventoried — both options to §12.1; M10 the bank is the archetype doc §5.3 bank COMPILED (each row through `generate_from_spec` → `level_loader` → `Simulation` → `spawn_match`, the post-load state snapshotted), reset = copy + key, never a re-derivation of `refresh_gas_energy`; M11 the rare-event path has no write inventory, tick position or §13(b) rule; M12 the unit schema must collapse FOUR definitions (`Unit.__init__`, `_unit_record`, `serialize_statuses`, the recorder) + a property test that every Unit attribute is a schema row or on an explicit non-synced list; missing columns listed; M13–M16 patch splits (P1a env+farm+measure Opus / P1b snapshot+build id+twin-debt Sonnet; P3a schema / P3b Philox / P3c positions HT / P3d field HT; P4a drivers / P4b stamp / P4c edit queue Opus; P6 needs a schema-driven Unit ↔ store BRIDGE so mid-port gates compare equal games; P6b per-class scenes at Opus; P7 ×4; P8 ×3); M17 "bots stay on the host" is a ruling for Erik, not a gap; M18 the container installs through `ml_stack.py`, never a second version set; M19 one git-describe helper (`match_replay.build_id`), `bp.build_id()` only for the `.pyd`'s provenance, `--exact-match --tags` for lightweight tags. Minors: `tick_stage_profile` listed twice; §4.3 omits slots 1 / 1b / 9 / 10; store rows ordered (team, slot); reward axis from `events.Cause`; time limit is a DRAW (truncated = placeholder); the distance-field HUMAN-TEST needs a written spec (target choice, activation, cadence, footprint passability); no effort levels; P2 hides a human step; P9 re-measures the farm on the same tag; Philox cherry-picked from `63-swarm-units`; `BREACH_BUILD_ID` goes stale without reconfigure | pending v2 |
| throughput honesty: the numbers, the yardstick, the stop condition, what P9 can and cannot buy | **SOUND WITH FIXES** (critique `94c2d0a`, `docs/residency_batching_critique_3_throughput_2026-11-07.md`): the table arithmetic re-derives (§2.1, §2.2, the 400 ns floor, the 25x, 0.169 / 0.122 ms, 3 440 / 2 420, the A100 hardware factors) | **B1** §3/§12.2/P9: the stop condition is read on the wrong slot (this PC: 1.6x; a Berzelius slot: 1.5–7x, because the A100 is ~1.9x on integer, 3.5–4.5x on DRAM but ~0.8x on LATENCY-bound work, and the tick has latency-bound parts: the one-block MG tail x 8 V-cycles, 60 dependent sweep launches) and has no anchor in what training needs; fix: gate on env-ticks/s per Berzelius slot on the training maps with the policy in the loop, device + farm-on-spare-cores vs farm alone, the decisive reading on a Berzelius node (P10 before P9's gate; the PC reading a proxy); the factor derived from GPU-hours per run at the planned scale — Erik rules the scale. **B2** §3/§9: the condition is read AFTER the cost it should avoid (P3's three re-baselines + two HUMAN-TESTs, P5, P6–P8's twin debt are all merged by P9), so "stops honestly at R0 + residency" cannot be kept as scheduled; fix: THREE readings — G1 after P5 (physics-only device vs physics-only farm, scaled by the farm's ~0.8 physics share), G2 after P8 (equal game; P9 only if below the gate and Nsight names a bounded kernel), G3 after P9 — and P3's position / zombie changes move BEHIND G1 (only P6 onward needs them; P4 / P5 gate against the CPU reference at the same commit). **Majors** M1 the 196 is PHYSICS-ONLY, no units (the design calls it "game as-is"); game-vs-game: ~155 per core; 8-core farm 1 000–1 450; 16-core slot 1 300–2 250 (14 workers, Epyc 0.6–0.8, SMT 1.0–1.3); M2 the A100 range 5 000–14 000 cannot be reproduced (14 000 = 3 440 x 4 drops the allowance): quote 4 800–10 900 as the throughput-bound band, latency-bound parts x0.8, the host pre-stage inside (a) is single-threaded int64 division per cell; M3 the §3 route table is underived (R3 "2 000–5 000" on a Berzelius row, R2 baseless, 10 000–50 000 in no source): roofline from a ~6 KB/cell/tick DRAM count = ~31 000 (3070) / ~108 000 (A100 40 GB) at peak; today's 2 420 is ~8 % of the 3070's; planning number ~20 % of roofline = ~6 000 / ~20 000; latency floor ~1–2 ms per batch tick → 50 000/s needs N ≥ 400–800; M4 "pays only after the flip AND kernel work" is not established — the flip alone may give 2 400–3 400 here (2–3x the corrected farm; the allowance is ~4x what the unported stages need), lowered by batch-max `n_sub`, the 10–20 % second-run slowdown, the policy sharing the GPU — G2 decides P9; M5 P1's farm measurement needs a PROTOCOL (random-action driver, placeholder rows on, decimation 1 and 4, 7 / 8 / 14 / 16 processes + a trainer stub, ≥ 3 episodes per worker, small and medium; env-ticks/s, p95, RSS); M6 the 8 GB card: per-unit crops (12 ch x 21² ≈ 21 KB per agent; a rollout of 8 x 512 x 32 ≈ 2.8 GB) — store crops fp16 / uint8, measure at a stated N with the rollout buffers allocated. **Training need (to state):** 1e8–1e9 decisions = 5e7–5e8 env-ticks (4 v 4, 24 tps, a decision every 4 ticks, 8 learning marines); the farm on ONE Berzelius node (10 400–18 000) finishes 5e8 env-ticks in 11–19 h; the device route (32 000–88 000 per node, optimistic) in 2–6 h; the device route earns its twin debt at ~1e10 decisions or across many runs. Minors: bus 0.57 ms plain vs 0.8 synced; "~3 ms Python gameplay" includes 1.5 ms of bots that training replaces; "25x one core" = ~3x the 8-core machine; the batched sweep is only 3.2x one CPU core at ~1 % integer issue / ~3 % bandwidth (the case for P9's sweep item); n_sub unmeasured; best-of-3 x 20 in the sweep bench; medium gives ~2x vs 1.6x on small — gate on the v1 map too; farm IPC via shared memory; farm workers must never create a CUDA context; per joule the device is ~1–1.3x; every measurement reports its decimation | pending v2 |
| game-dev continuity: snapshots, twin debt, the rules of §10, what a gameplay change costs after the port | **SOUND WITH FIXES** (critique `3a1d1d5`, `docs/residency_batching_critique_4_continuity_2026-11-07.md`): the architecture holds; two things fixed before P1 or a training run at a new snapshot runs a device game that silently differs from the played game while main's GPU play modes rot | **B1** §6/§10: twin debt cannot be detected where gameplay is written — every CUDA gate SKIPS in a worktree without its own `cpp/build_cuda` (29 of 30 `test_cuda_*` carry the skip), the corpus never exercises what will change next (the only loadout's K5 has `mag_size = 0`, so magazines / reloads / sprays / pellets / melee are never fired and their twins stay GREEN whether wrong or missing), mag state is outside the digest, a stored corpus stales weekly, and #71 is the precedent of a twin "handled" by switching it off; fix: a COMPUTED debt ledger (a twin manifest: each device twin → the reference paths it mirrors + the commit it was last level at; `tools/twin_debt.py` diffs `level_at..HEAD`), refusing doors on the training side (`build_level_bank`, the run file and `require_snapshot(backend="device")` refuse a feature without a twin or with non-empty debt; the FARM never needs the ledger and trains at any tag), LIVE lockstep (record on the reference at HEAD, replay on the device in the same run; a pre-flight of K random-action episodes on the run's own bank across both backends), per-feature COVERAGE counters (shots per weapon row, reloads, statuses per kind, entity classes, coupling rows, payloads; refuse a feature in the run's config with zero exercise). **B2** §4.2/§6/§10: device code is not split into what the PLAYED game runs and what only training runs — after P4 / P5 the played `--resident` game runs the batched cores, the edit queue and the stamp kernel, so "the played game never waits for a kernel" + "any twin may be xfailed" would break `--cuda` / `--resident` on main (#71 generalised), against today's rule of reference → C++ → CUDA in the same change (32 CUDA commits beside 50 C++ since 2026-09-07); fix: **Class P** (physics cores, trace tail, sweep, combustion / temperature / fire, edit queue, stamp_units — kept level per patch on main as today, never xfailed) and **Class T** (unit store + slots, device entities, combat, horde field, reset, obs / reward / done — carry debt under the ledger); the played `--resident` keeps Python gameplay over device physics. **Majors** M1 P1 / P2 / P3 merge to MAIN as they pass (the two HUMAN-TESTs on main), the first `train/*` tag on main, the arc branch merges main at every patch start, only Class-T work may ride it; M2 provenance holes: ANNOTATED tags placed by Erik, SHAs compared (not describe strings; the repo mixes tag kinds), untracked files under `src/ levels/ config.toml cpp/` count as dirty (`air_init.npy` is found by presence), the bank records sha256 of every input + the tag SHA, assert `bp.__file__` under the snapshot's own build and `simulation.__file__` under its `src/` (worktrees routinely import main's mixed-build `cpp/build_cuda`), the build id generated at BUILD time into one generated TU, a run manifest with CuPy / torch / numpy / driver versions (numpy's Generator streams are not promised across versions), run outputs outside the checkout; M3 a training-episode RECORD (tag + SHA, bank sha256 + template index + source, episode seed / key, decimation, action rows) convertible to a `Recording` that replays on the reference; the fuse as an integer in `INTENT_KINDS`; M4 the cost claims ignore BEHAVIOUR (a reload verb does nothing until a kernel consumes it): a per-change cost table, a **device table upload** system (weapons, ammo, loadouts, species, the status registry, payloads, `[exchange]`, `[zombie]`, `[movement]` uploaded through a column schema), the rare path's unit access specified; M5 three §10 rules with CPU-side gates in the ordinary suite: match-path draws only through Philox with a registered salt (`spawn_match` installs a refusing `sim.rng` stand-in), new per-tick match-path arithmetic integer with a float ratchet over the match-path modules, every per-unit collection with a schema capacity + overflow rule (`STACK_STACK` capped); M6 the unit schema must REUSE the entity kinds and serializer under a `UNIT_SECT_V2` preamble, columns absence-transparent (a default hashes nothing, the `DIGEST_SECTIONS` precedent), `Unit.__slots__` = schema columns + a declared `REFERENCE_ONLY` list so an undeclared attribute raises where it is set; §5.7 corrected (the unit hash is `UNIT_DIGEST_KEY`, not the entity section); M7 Q16 positions: store synced `x_q` / `y_q`, keep `x` / `y` as read-only dequantized float properties so the ~15 float readers (renderer, UI, bots, tools) do not change; M8 the training checkout needs a row in the git-workflow skill: `git worktree add --detach ../breach-train-<name> train/<name>` outside `.claude/worktrees`, an in-use marker the audit honours, a hot-fix route (fix on main; if the run cannot move, a `<issue>-train-<name>-fix` branch from the tag, cherry-pick, Erik tags `train/<name>.1`). Minors: §10 bullet 3 must xfail the DEVICE gate, never a reference test; draw-site list corrected; `test_twin_debt` checks the issue-number format in pytest, the state in a tool; the field amends the A* row AND the zombies "WRAPPED, never rewritten" row (`test_zombie_control.py`); the reference field in C++ / vectorised, cost stated; `current_mag = None` needs an integer sentinel; doors via `seal_tiles` make the "rare" path frequent; each snapshot costs a full CPU + CUDA (+ container) build, never copy a `.pyd` | pending v2 |
