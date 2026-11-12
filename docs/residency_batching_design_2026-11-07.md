# GPU residency + batching — design (v3, 2026-11-07)

Issue #29 (the umbrella; #77, #24, #25 and #71 ride it). Lead: Claude under
Erik's mandate of 2026-11-07. **v3 after the four-lens critique round and the
v2 re-check** (§14; v1 is in git at `d6c0957`, v2 at `fae5955`). Inputs, in reading order:
`docs/residency_where_we_stand_2026-11-07.md` (the tick today),
`docs/residency_benchmark_2026-11-07.md` (the numbers),
`docs/research/batched_gpu_sims_litsearch_2026-11-07.md` (the literature),
`docs/rl_env_arc_proposal_2026-08-27.md` (B1–B6 and the §A habits, which this
design supersedes as the plan while keeping the habits),
`docs/architecture/engine/02_state_and_ownership.md` (canon this design
amends: "entity / logic state stays CPU" becomes false for the training twin).

## 0. Summary

The arc builds ONE batched environment API with two backends. The `cpu_farm`
backend runs the game exactly as played, in N processes, at any tag. The
`device` backend runs N games per launch with every per-tick computation on
the GPU; the host does nothing per tick. The API, the tables and the gates
are shared, so RL work starts on the farm the day the API exists and moves
to the device backend stage by stage, each stage bit-identical to the
reference and measured against the farm.

What the numbers say (RTX 3070, 2026-11-07, corrected in §2.3): one CPU core
runs the v1 match at about 155 env-ticks/s with the game as-is; the 8-core
farm here at 1 000 to 1 450; one Berzelius slot's 16 cores at 1 300 to
2 250. The device backend at today's kernel efficiency projects 2 400 to
3 400 per 3070 and 4 800 to 10 900 per A100 (throughput-bound parts only),
so 1.4 to 3.4x the farm here and 2 to 8x per Berzelius slot. A training run
of 1e9 decisions (5e8 env-ticks) finishes in under a day on ONE node's farm.
The device route earns its upkeep at ten times that scale or across many
runs. So: the farm is the first deliverable and the yardstick; the device
backend proceeds in gated stages with three readings (§3) and may stop
honestly; whether it is pushed to its ceiling is a question of the planned
training scale, which Erik rules (§12.2).

Two consequences of the critique shape everything else. Device code splits
into **Class P** (what the played `--cuda` / `--resident` game runs: the
physics cores, the edit queue, the stamp kernel; kept level per patch, never
in debt) and **Class T** (what only training runs: the unit store and its
slots, entities, combat, the horde, reset, observations; carries debt under a
computed ledger with refusing doors). And every order-dependent law on the
device runs as a **per-env serial applier**, one thread per env executing
that env's recorded effects in the reference's order through device
functions, so topology changes, blasts, bites and the vent ledger never
leave the device.

Decided with Erik on 2026-11-07: snapshots with catch-up; zombie pathing may
become a distance field.

## 1. Goal and non-goals

**Goal.** `BatchedEnv.step(actions)` advances N games and returns
observations, rewards and terminals as tensors with a leading N axis. The
device backend does it with zero host work per tick; the farm with N
processes. Env i of N is bit-identical to env i alone and to the played game
at the snapshot tag. Throughput is measured in env-ticks per second per GPU
slot (one A100 + 16 Epyc cores on Berzelius; one RTX 3070 + 8 cores here).

**Non-goals (the agent era).** The observation design beyond the crop
mechanism, the action space beyond the intent verbs, the reward, the
trainer, frame skip, per-marine vs per-team policies, policy placement on
the cluster, a full-map observation (§12.5). The tables carry PLACEHOLDER
rows.

**Out of this arc.** A human control scheme (a ControlSource; unconstrained
here); the thermal overhaul #68; wildlife #63; the window and the renderer;
a cheaper training physics (§12.4).

## 2. Inputs, in numbers

### 2.1 Where the tick's time goes (medium ship, bots, resident, ms per tick)

| part | ms | note |
|---|---|---|
| controllers (bots + horde) | 1.5 | Python; training replaces the bots with the policy and the horde with a kernel |
| gameplay slots (step minus slot 7) | 1.9 | Python; stamp_units 0.5 (C++), 9b walls 0.4, movement 0.3, 9c / 9c2 0.3, zombies 0.15 |
| EOS stage | 3.7 | device chain + its host pre-stage, one C++ call |
| trace tail | 0.1 | device |
| bus | 0.6 plain / 0.8 synced | 11 uploads + 7 downloads, ~0.8 MiB |
| combustion | 0.6 | host (#71) |
| step_tail | 2.9 | host ripple + per-call fire / temperature / sweep twins: ~21 malloc / free, ~1 MiB of copies (estimate) |
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
cell-update (3.2x ONE CPU core, at ~1 % of integer issue and ~3 % of
bandwidth: the case for P9's sweep item), 63 launches per tick whatever N,
bit-identical to the CPU sweep.

### 2.3 The yardstick (corrected by lens 3)

| quantity | value | basis |
|---|---|---|
| one core, physics only, no units | 196 | the stacked probe at N = 1 |
| one core, the v1 match, step only | ~175 idle units / ~155 to 160 active units; 142 with the bots' controllers | tick budget 5.9 / 6.4 / 7.0 ms, ~3 % faster since |
| 8-core farm here | 1 000 to 1 450 (game); 1 250 to 1 830 (physics only) | 7 to 8 workers, all-core 0.9, SMT 1.0 to 1.3; P1 MEASURES it |
| one Berzelius slot's 16 cores | 1 300 to 2 250 | 14 workers, Epyc 7742 core 0.6 to 0.8x a 3700X core, SMT 1.0 to 1.3 (assumptions) |
| 3070 device, today's kernels, host stages gone | 2 400 to 3 400; ~2 000 with the downside movers | EOS + sweep + an allowance; movers: batch-max `n_sub`, the 10 to 20 % second-run slowdown, the policy sharing the GPU |
| A100 device, same kernels | 4 800 to 10 900, throughput-bound parts only; latency-bound parts x0.8 | 1.9x on integer, 3.5 to 4.5x on DRAM, 0.8x on latency (lower clock); unmeasured |
| ratio here, equal scope | 1.4 to 3.4x | |
| ratio per Berzelius slot | 2 to 8x | |
| roofline (DRAM) | ~31 000 (3070) / ~108 000 (A100 40 GB) at peak; planning ~20 % = ~6 000 / ~20 000 | ~6 KB of DRAM traffic per cell per tick (MG level 0 ~3.5 KB, substeps ~0.5, rest of EOS ~0.5, sweep ~1, the rest ~0.5), ±50 %, replaced by Nsight's bytes at P2 |
| latency floor | ~1 to 2 ms per batch tick (~250 to 300 launches + ~950 fused-tail barrier phases) | so 50 000/s would need N ≥ 400 to 800 |

Today's 2 420 is ~8 % of the 3070's roofline; the EOS chain runs at ~18 %
(host pre-stage included), the batched sweep at ~3 %.

### 2.4 What training needs (the anchor)

| quantity | value | assumptions |
|---|---|---|
| one training run | 1e8 to 1e9 agent decisions = 5e7 to 5e8 env-ticks | 4 v 4, 24 ticks/s, a decision every 4 ticks, 8 learning marines |
| farm, this PC | 5.7 to 8.3 days per 5e8 env-ticks | 70 % of wall clock in the sim |
| device, this PC | 2.4 to 4.1 days | today's kernels |
| farm, one Berzelius node (8 slots) | 11 to 19 hours; 88 to 153 GPU-hours | |
| device, one node | 2.3 to 6.2 hours; 18 to 50 GPU-hours | optimistic: throughput-bound factors |

The farm on one node finishes a 1e9-decision run in under a day. The device
route pays for its upkeep at ~1e10 decisions per run, or across many runs
(a league, many reward variants). Erik rules the planned scale (§12.2).

### 2.5 What the literature adds (`docs/research/batched_gpu_sims_litsearch_2026-11-07.md`)

- Unit store as SoA with a fixed `(N, max_units)` capacity and a `valid`
  mask; each column is the tensor torch reads in place.
- Gameplay never as one thread per env (Madrona measured 1.9 % utilisation
  for that shape): one thread per unit row across all envs for per-unit
  work, one block per env where units interact, separate kernels per stage.
- The step contract is Isaac Lab's: `step(actions)` runs `decimation` ticks,
  returns `terminated` and `truncated` separately, resets done envs inside
  the same step by mask and returns post-reset observations.
- Launches are a few percent at batch sizes; capture the tick as a graph
  anyway (#24's spike is bit-identical) with a WHILE node over the device
  max of `n_sub` (CUDA 12.4+); no megakernel; fuse only where data stays on
  chip (`mg_fused_tail` already does).
- The sweep's 63 launches per tick become ONE: one block per (env,
  ordinate), the wavefront loop inside the block, the previous wavefront's
  outflow in shared memory; arithmetic unchanged.
- Multigrid: a per-env hierarchy on the `(N, h, w)` layout; never a tall
  grid (coupling at coarse levels, red-black parity, different coarsening).
- Integer throughput: the 128-bit staging (`__mul64hi`) is the costly
  primitive; a narrowing is an arithmetic change that lands in the CPU
  reference first, only after Nsight Compute shows a kernel compute-bound.
- Distance fields: one block per (env, field), the frontier loop inside the
  block, int16 distances in shared memory; fixed neighbour order.
- Deterministic parallel logic: #169's two-phase shape is the device shape;
  every draw a counter-based RNG keyed (seed, env, tick, slot, purpose); no
  surveyed system gates batch invariance, Breach builds that gate itself.
- Zero-copy: DLPack from persistent buffers, the sim on torch's stream or
  an explicit event wait, both memory pools capped.

## 3. The routes, the choice, the stop condition

| | R0 CPU farm | R1 resident physics, Python gameplay | R2 batched physics, C++ gameplay on the host | R3 everything per-tick on the device | R3 + kernel throughput |
|---|---|---|---|---|---|
| env-ticks/s per 3070 slot (8 cores), small | 1 000 to 1 450 | below R0 (Python 1.2 to 1.9 ms per env-tick on the host) | ~R3 (bounded by the same device cost) | 2 400 to 3 400 (today's kernels; §2.3) | ceiling ~31 000 at peak; planning ~6 000 |
| env-ticks/s per A100 slot (16 cores) | 1 300 to 2 250 | below R0 | ~R3 | 4 800 to 10 900 (throughput-bound parts; unmeasured) | ceiling ~108 000; planning ~20 000 |
| port cost | none | small | large: a second gameplay copy in C++ | large: gameplay kernels | plus kernel engineering |
| twin debt per gameplay change | none | none | yes | yes (Class T, §6) | yes |
| game dev continues in Python | yes | yes | yes | yes | yes |

**Choice.** R0 first, as the batched API's first backend. Then the device
backend in stages: Class P residency finished, the physics batched, the
authority flip (R3), then throughput. R2 is rejected: a C++ host copy of
gameplay would be a third copy once the kernels exist. R1 is a state the
build passes through, never a shipped one.

**Stop condition: three readings, per Berzelius slot (this PC a proxy).**
The metric is env-ticks per second per GPU slot on the training maps (small
AND medium), with the policy update in the loop, the device backend plus a
farm on the slot's spare cores against the farm alone, same driver, tables
and decimation, at a stated N with the observation and rollout buffers
allocated. The decisive readings are on a Berzelius node (so the Linux patch
P10 lands before them); the PC reading is a proxy whose transfer is stated
(the A100's 1.9x / 3.5 to 4.5x / 0.8x by part).

- **G1, after P5** (physics batched, gameplay still on the host): the SAME
  scene on both sides — NO units, the ship's entities ON (vents, ducts,
  doors, charges, heat sources), the small and the medium ship — device
  against farm; the device side's host 9e cost reported as its own line
  (it still runs on the host per env at G1, ~0.06 ms per env-tick plus the
  bus), the farm scaled by its physics share (~0.8 on small), and the
  device side charged an allowance for the gameplay launches and applier
  stages it does not yet have (one sweep's cost, as the benchmark did).
  Continue to P3 and the flip only if it projects at or above the gate
  factor.
- **G2, after P8** (equal game, the device backend complete): the real
  ratio. P9 runs only if G2 is below the gate AND the Nsight capture names a
  bounded kernel.
- **G3, after P9.**

The gate factor derives from GPU-hours per training run at the planned
scale (§2.4), which Erik rules (§12.2). Until he does, the working factor is
3x the farm per slot. If a reading fails, the arc stops at R0 plus Class P
residency, which is still the played game's fastest single-instance path
and leaves nothing in debt.

## 4. Architecture

### 4.1 The batched environment API (one API, two backends)

`src/rl_env.py` (provisional): `BatchedEnv(bank, n_envs, backend, run)`.

- `reset(mask=None)` re-initialises the masked envs from the bank (§4.6).
  Reset is by MASK, never by an id list. Each env's episode k uses seed and
  template `f(run_seed, env, k)`, a pure function evaluated where the reset
  happens (a Philox draw on the device; the same function on the farm), so
  an env's episode sequence never depends on when other envs end.
- `step(actions)` runs `decimation` ticks (`run.decimation`, default 1)
  for every env; returns `obs`, `reward`, `terminated`, `truncated`, `info`
  with a leading N axis; envs that ended are reset inside the call and
  return post-reset observations (the terminal observation in `info`).
  Latch verbs (trigger, use, throw) apply on the first tick of the
  decimation step; move and aim are held through it.
- The **run file** (`run.toml`, outside the snapshot tree, door-checked,
  its sha256 in the run manifest) carries the observation rows, the reward
  rows, `decimation`, `n_envs`, the bank path and the policy's name. Nothing
  about a reward experiment lives in the game snapshot (lens 2 M5).
- The match's parameters stay in match.toml / `MatchSpec`, carried per
  template by the bank; `terminated` is the MatchRules law; the time limit
  is a DRAW in v1 (the `truncated` row is PLACEHOLDER until the agent era).
- `backend = "cpu_farm"`: N worker processes over a **`MatchSession`**, the
  steppable session factored out of `tools/run_match.py` (setup,
  `spawn_match`, `create_match_controllers`, the recorder hook, `tick()`);
  `run_match` itself becomes its loop's one caller. The policy is a
  registered controller, `policy` (the `AgentPolicy` slot `control_source`
  reserves), that issues the action rows through the intent facade, so the
  recording tap sees them. Workers communicate over shared memory (crops
  are ~170 to 200 KB per env-step at decimation 1); a worker never creates
  a CUDA context. The farm trains at any tag; it needs no ledger.
- `backend = "device"`: the batched resident engine (§4.2 onward); the
  tensors are persistent CuPy buffers exported zero-copy (§4.10).
- **Backend equality gate:** for the same bank, run seed and action stream,
  the two backends return the same obs and the same per-env tick digests.

### 4.2 The batched resident engine

- Every plane is `(N, h, w)` (gas `(N, g, h, w)`, the light triples
  `(N, h, w, 3)`); every launch core takes `N`; per-env scalars are device
  arrays of length N read by env index. This applies the §A habit to the
  cores born `(h, w)`: water, the EOS substeps, MG, energy flux, kick /
  compression, temperature, fire, combustion. Scratch keyed `(N, h, w)`,
  preallocated at batch build inside the capped pool; no allocation inside
  a step.
- A BATCH IS HOMOGENEOUS: one grid shape, one boundary type, one config; the
  template may differ per env. v1 training batches are space maps; the sky
  exchange stays a host bracket on ambient maps, outside the device backend
  (ACCEPTED GAP).
- **`n_sub` per env.** The pre-stage reductions (`t_max_abs`, `max_rad`,
  `max_du`) become per-env device reductions (an integer max is order-free).
  The quantities `n_sub` feeds are a FOLD TABLE per batch: `N_SUB_MAX` rows
  of (`dt_s_q`, `k_flux_q`, `flux_pu_cap`, …) built on the host once per
  config with the range checks at build time; the per-cell planes
  (`coeffE` / `coeffS` over `dyn_permeability`, `absorb_q`, `cap2`) are
  built by device kernels from the resident planes. BOTH loops (the
  transport substeps and the energy-flux sub-cycles) run to the batch's
  maximum with per-env masks, a masked env doing exactly nothing. Interim
  (P5 to P8): ONE counted 4-byte D2H of `max(n_sub)` per tick drives the
  loop; P9 replaces it with a WHILE graph node.
- **Books.** Every counter and seam channel the CPU keeps (the EOS / thermal
  / combustion `e_*_sum`s, the rails, `boundary_flux_`, the trace books, the
  gas-energy seam channels, the solid-energy channels, `n_destruction_seed_sum`)
  lives in per-env `(N, slots)` int64 arrays under ONE channel enumeration
  shared by the seams and the closure gate; assign vs accumulate per counter
  as on the CPU; tick-start reset per env. The closure identity is checked
  per env at checkpoint cadence, never by a per-tick D2H. Host `throw`s
  become per-env failure flags (the sweep's pattern).
- The MG hierarchy is per env on the `(N, h, w)` layout: fine levels as wide
  kernels over `(N, cells_at_level)`, the fused coarse tail one block per
  env, the fixed 8 V-cycles keeping envs in lockstep. Never a tall grid.
- The radiation sweep's core is already `(N, h, w)`; its driver moves into
  the resident tick (P4) and its schedule becomes one block per (env,
  ordinate) (P9).
- **Streams.** One stream threaded through every launch core; every in-tick
  copy and memset asynchronous on it; from P4 a capture-mode test asserts
  the whole tick captures in global mode (any legacy-stream operation fails
  it). Observation and action buffers are double-buffered so torch's reads
  and writes never race the engine's; the host reads `terminated` only at
  checkpoint cadence through an asynchronous pinned copy (in-step reset
  needs no host read).
- `ripple` / `ripple_v` are not computed on the device backend (`headless`
  skips `step_ripple`; nothing in the sim reads them: ACCEPTED GAP).

### 4.3 Authority: the device runs the tick

Every slot of `Simulation.step` has a device form; the host does nothing
per tick. The device tick keeps the reference's ORDER slot by slot; where
the reference is serial, the device form is serial (the applier); where the
reference decides against the start-of-slot state (#169's two-phase laws:
4 / 4b → 4c, slot 2's rounds, slot 5's bites), phase A is wide and the
applier applies. The per-slot sequence (from the re-check's iteration-order
table, `docs/residency_batching_critique_5_v2_recheck_2026-11-07.md`):

| slot | reference order | device sequence | patch |
|---|---|---|---|
| 1 events, 1b intents | units in id order; a throw appends a projectile; USE toggles `want_open` | the ring's tick head; the action buffer consumed in id order (throws get a spawn ordinal) | P6 |
| 2 grenades | projectile-list order, each detonation IMMEDIATE and seen by the next | applier(grenades), serial, before the rounds | P7b / P7c |
| 2 rounds | bullet-list order; phase A marches against the post-grenade state; resolve in record order | wide(rounds phase A) → applier(rounds) | P7b |
| 2b statuses | id order; DoT on the own unit; events in id order | one thread per unit row; the ring appended in id order | P6 |
| 3 movement | id order; walls only | one thread per unit row (the integer law, §7) | P6 |
| 4 / 4b shooting, sprays: phase A | id order; sprays after shooting; own-row writes immediate | one block per env over its shooters (line of sight, the march), effects recorded; sprayers after shooters | P7a |
| 4c resolve | `SimultaneousEffects` record order: shooters by id x records, then sprayers; hit / status / face / world (`chew_wall`, `cover.chew`, `execute_payload`, spray) | the applier in record order | P7a |
| 5 zombie AI | passes 1–2 order-free; pass 3 in id order; ties to the lower id; bites immediate but decisions read stale prey | the distance field (§4.9) + one thread per zombie; bites recorded, applied by the applier in id order | P7d |
| 6 stamp_units | MAX stamps; per-tile refill (exact integer division) | one kernel over unit rows (the five planes) | P4b |
| 6b FieldEdit flush | stable sort `(field, source_id, region, seq)`, `seq` spanning the previous tick's 9e; tiles in region order; noise in that order | the device edit queue applied by the applier (§4.8) | P4c |
| 7 physics | | the batched cores (§4.2) | P4a, P5 |
| 9 burn-through | row-major; `destroy_wall` serial | the wall law per cell → applier(topology) | P6 |
| 9b over-pressure walls | after 9; stable sort by −spread, ties row-major, cap 16 | a per-env top-16 selection → applier(topology) | P6 |
| 9c heat damage, 9c2 wind, 9c3 gas rows, 9d ignition | id order; teargas loop then poison loop; events in id order | one thread per unit row / per cell; the ring in id order | P6 |
| 9e entities | emit / sensors / logic (ordinal) → pumps → vents (duct ordinal; returns then supplies) → doors (ordinal; span row-major) → charges → heat sources (read `blast_log`, including this tick's) → swap | ONE serial applier stage per env in that order (vents are not parallel per duct: two actuators may share a tile) | P6b |
| 10 tick end | heat / rad / TickCredit clears; `convert_marines_to_zombies` every tick in id order; MatchRules | the clears; conversion in id order; the `done` kernel; the ring sealed | P6 |

**The per-env serial applier** is THE device form of every order-dependent
law (lenses 1, 2, 4 and the re-check converge on it). It runs as ONE BLOCK
per env: thread 0 walks that env's recorded effects in the reference's
order and does the scalar, order-dependent work; an effect's disjoint tile
set (a deposit over a walked set, the reach walk's frontier, a disc's
tiles) fans out over the block's threads with a barrier between effects,
so order is kept and a C4's ~2 000 edit records do not run on one lane.
Each effect is a DEVICE FUNCTION with the same book writes into the per-env
book channels: `apply_packet`, a status, a facing snap, `chew_wall`,
`cover.chew`, `destroy_wall`, `seal_tiles` / `unseal_tiles`,
`on_tile_changed` (its `render_dirty` / sky-mask side is host-only and
dropped on the device), the wall law, `execute_payload` (the reach walk;
its deposits as edit records; the blast's unit damage on the store; the
`blast_log` row; the TickCredit claim; the `div_source` write and its
saturation counters; the `expansions` row; the ExplosionEvent into the
ring), a spray deposit, a bite, a door toggle with the door's per-tile hp
list, the edit application. Phase A stays wide. There is NO host rare-event
path: a blast, a chew, a door cycle and a burn-through all stay on the
device, in slot order, and "zero host work per tick" holds on every tick.

**Cost bound (stated, measured by the profiler's applier-ms row):** a frag
(radius 2) is ~160 tile operations; a `demolition_c4` (radius 8) ~2 000 to
2 500 edit records; the vent ledger ~0.3 to 1 ms per env per tick if
serial. A batch tick waits for its slowest env. The block form above is
what keeps a C4 tick under an agreed share (10 %) of the batch tick; if the
measured share exceeds it, the walk and the deposits move to one block per
blast. `apply_explosion`'s disc stays serial on thread 0.

The host keeps: the bank, logging, the recorder export (§4.5). Reset
happens inside `step` by mask (§4.1); the host reads `terminated` only at
checkpoint cadence. A terminal tick inside a decimation step ends that
env's step at the terminal tick (the remaining ticks of the step are not
run for it); the ring is kept PER TICK so a checkpoint inside a decimation
step is well-defined. The sky exchange (ambient maps only) is outside the
device backend.

### 4.4 The device unit store and the unit schema

`(N, U, cols)` SoA with a `valid` column; U = the batch's `max_units` (the
sum of `max_per_team` over the teams plus the zombie cap), fixed at bank
build; rows ordered (team, slot). Columns come from ONE schema
(`src/simulation/unit_schema.py`), which REUSES the entity system's field
kinds and `entities/serialize.py`'s `name|int64` encoding under its own
`UNIT_SECT_V2` preamble; a column at its default hashes nothing
(absence-transparent, the `DIGEST_SECTIONS` precedent), so a new column
moves no golden until a unit uses it. The schema is the one table the
digest (`_unit_record` reads it), the serializer, the recorder snapshot, the
obs gather, the Unit ↔ store bridge and the kernels read: today's four
definitions collapse to it (lens 2 M12, lens 4 M6). `Unit` gets
`__slots__` = schema columns + a declared `REFERENCE_ONLY` list (the WEGO
fields: orders, plan, ap, overwatch, belt, action cooldowns; the bots'
caches), so an undeclared attribute raises at the line that sets it.

Every per-unit collection has a schema CAPACITY and an overflow rule at its
door: the status block is fixed-width and `STACK_STACK` kinds are capped;
`current_mag = None` gets an integer sentinel. Positions are synced `x_q` /
`y_q` (Q16, §7) with `x` / `y` as read-only dequantized float properties, so
no reader changes (lens 4 M7). Columns (v1): id, slot, team, is_zombie,
valid / alive / life_state, x_q, y_q, facing_q, footprint id, hp (Q16),
species / loadout / env-profile ids, weapon id, has_grenade, has_explosive,
last_fire_tick, magazine state, reload_done_tick, spray state, zombie
activated / awakened / descent cursor / accumulator, last_melee_tick,
killed_by_zombie, cadence, move accumulators, the intent state (`set_trigger`
HOLDS a live trigger, it is not a latch; both backends apply the same
decimation semantics), the status block, the exposure state,
`_possessed_ids` (capped). `recent_attackers` is unread on the match path
and is `REFERENCE_ONLY`.

**The Unit ↔ store bridge** (schema-driven, both directions, at tick
boundaries) is a named system: during P6 to P7 the device backend runs with
the slots not yet ported executing on the host over bridged `Unit` objects,
so every mid-port gate compares equal games; after P7 the bridge serves the
recorder export and the tile inspector only.

#### 4.4a Amendment 2026-11-11: P3a re-planned against the code at `bac6e077`

The column list above was written on 2026-11-07. Since then the gameplay pack
(#183: turning, unit impulse, exposure, magazines, grenade kinds, melee,
flashlight, the horde field), #193 (the planted charge) and the agent design
(`docs/rl_agent_design_2026-11-11.md` §9) changed what a unit holds. The
inventory below was taken from the code, not from memory: `unit.py`, a grep
of every attribute write on a unit in `src/`, and a spy on `Unit.__setattr__`
over the whole suite (5 613 tests), which found 109 attribute names. Every
one is sorted into exactly one group. P3a's property test holds the sort.

**Group S, synced columns** (device columns; hashed from the schema, each
only while it differs from its schema default unless marked *always*):

| block | columns | note |
|---|---|---|
| identity | `id`, `slot`, `team`, `is_zombie`, `life_state` (*always*) | `alive` stays a property over `life_state` |
| kit (table indices on the device; strings on the host) | `species_id`, `loadout_name`, `env_profile_name`, `weapon_id`, `melee_id`, `charge_ammo` (`""` = none), `magazines_weapon`, `magazines`, `footprint`, `mass_g` | fixed per unit bar a weapon swap; NEW vs §4.4: melee row, charge row, mass |
| position | `x`, `y` (*always*, `round(v·1e9)` until P3c), `facing_q` (*always*), `facing_target_q` (hashed as `turn`), `omega_q` | the turning law's state is NEW vs §4.4 |
| motion | `speed_ticks_per_tile` (`None` → an integer sentinel), `zombie_move_accumulator` | |
| health | `current_hp` (*always*, hashed as a Q16 integer; P3a asserts it is an exact n/65536 at every write), `statuses` (fixed-width block, capacity below), `exposure` (fixed block) | |
| impulse | `imp_x`, `imp_y`, `imp_l`, `stand_x`, `stand_y`, `swx`, `swy`, `pvx`, `pvy`, `effort`, `react_ticks` | NEW vs §4.4 |
| magazine | `current_mag` (`None` → `MAG_UNBOUND`, an integer sentinel), `spare_mags`, `reload_done_tick` | |
| grenades and charges | `grenade_counts` (fixed block, one per hand-grenade row), `has_explosive`, `throw_release_tick`, `throw_kind`, `throw_reach_q`, `throw_fuse_ticks`, `plant_done_tick`, `plant_fuse_ticks` | throw and plant in progress NEW vs §4.4 |
| cool-downs | `last_fire_tick`, `last_blow_tick`, `last_shove_tick`, `last_melee_tick` | |
| spray | `spray_ticks_left`, `spray_target`, `spray_aim_angle` | the last two FLOAT until P3e (as positions until P3c) |
| zombie | `zombie_activated`, `killed_by_zombie`, `zombie_noise_y`, `zombie_noise_x`, `zombie_noise_ticks`, `zombie_anchor_y`, `zombie_anchor_x` | the horde's state NEW vs §4.4; §4.4's "descent cursor" does not exist (the field law keeps none) |
| flags | `flashlight_on` | NEW |
| intent state (the held verbs and the latches) | `live_move_dir`, `live_aim`, `live_trigger`, `pending_throw`, `pending_reload`, `pending_shove`, `pending_melee`, `pending_plant`, `pending_use` | each with an integer encoding in the schema (an absent intent = the default); the agent design's §9 needs the held verbs on the device (the team display reads them) |

**Group D, derived constants** (never written after spawn; rebuilt on the
device from group S's kit indices, so not columns and not hashed): `base_stats`,
`base_speed`, `mass` (the species' float kg; `mass_g` is the synced one),
`offsets`, `environment`, `mitigation`, `inventory` (a stub, always empty),
`_frag_slot`; and the four per-unit optics overrides the stamp reads by
`getattr` with a map default (`permeability`, `wave_absorb`, `heat_atten`,
`light_atten`: no game path sets them, tests do), declared as slots and left
unset. A property test checks every group-D value equals its derivation from
group S at every tick of a match, conversion included.

**Group R, `REFERENCE_ONLY`** (host only; never read on the continuous match
path, or a diagnostic): the WEGO / onephase state (`orders`,
`current_order_type`, `ap`, `plan`, `plan_round`, `busy_until_tick`,
`gcd_until_tick`, `swap_cd_until_tick`, `action_cd`, `loadout`, `belt`,
`active_slot`, `overwatch_facing`, `overwatch_half_deg`, `move_path`,
`path_tick_offset`, `spray_order`, `fire_target`), `recent_attackers`
(unread on the match path), `impulse_books` (the momentum books: a
diagnostic, as today), `name` (a label; conversion prefixes it), and
`awakened` (written only by `__init__`; read only by `stats.py`'s hidden
`imagination` stat -- P3a deletes it if nothing else reads it, else R).

**Not unit state, a table of its own: the planted charges**
(`Simulation.planted_charges`, #193). A `CHARGE_SCHEMA` in the same module:
tile x / y, `det_tick` (`None` = remote → `NO_TIMER`), the charge row as its
index among the ammo rows, `owner_id`, the approach tile (`None` →
`NO_TILE`), `live`; a fixed capacity per sim with a refusing door at the
plant (a plant into a full table is refused, counted, no event); hashed as
today, only while one lives.

**Capacities.** `statuses`: a fixed width, `STACK_STACK` kinds capped, the
overflow rule at `status.apply`'s door (refuse the new instance, counted) --
the width chosen so that no test, no pinned match and no shipped weapon row
reaches it, asserted. `grenade_counts`, `exposure`: already fixed.
`_possessed_ids` (§4.4) is a set on the Simulation, not unit state; it stays
there.

**Differences from §4.4's list**, in one line each: NEW columns -- the
turning law (3), the impulse state (11), `mass_g`, the melee row, the charge
row, `has_explosive`'s partner `charge_ammo`, the throw (4) and plant (2) in
progress, the shove and blow cool-downs, the horde's noise and anchor (5),
the flashlight bit, the nine intent fields named one by one; DROPPED --
"descent cursor" (no such state), "cadence" (it is `speed_ticks_per_tile`),
`awakened` (to R or deleted), "valid" (a store column, not a `Unit`
attribute; it arrives with the store in P6); MOVED -- `_possessed_ids` stays
on the Simulation. The order's four columns (the agent design's patch 1f,
#195) are NOT in P3a: 1f adds them as four schema rows, hashed only while
set.

**The digest change at P3a** (§5 item 9): the unit record is built from the
schema; `ap` and `n_orders` leave it; `current_hp` hashes as a Q16 integer;
every group-S column hashes while it differs from its default (so the
magazine, the grenades, the cool-downs, the kit and the intent state enter
the digest for the first time); the events and the charges keep their
records (the event ring is P6's). Proof that only the hash moved, not the
game: the new record is computed by the SAME schema code on the old `Unit`
(main before P3a, attributes read by `getattr`) and on the new one, tick by
tick over the pinned matches, and must be equal; every field digest is
unchanged.

**Tier.** §9 gives P3a to Sonnet; the list doubled since and the sort, the
sentinels and the derivations are judgement, and the re-pin is not a tol-0
gate, so P3a runs on Opus (the lead's call, 2026-11-11).

**As built (P3a, 2026-11-12).** The sort above held against the code; no
column changed group. Corrections and choices, one line each:
- `orders` and `move_path` are WEGO state the continuous path still READS --
  slot 4's `process_shooting` looks for a fire order in `orders`, slot 3's
  WEGO branch takes `len(move_path)` -- but both are always empty there, so
  they stay R (a REFERENCE_ONLY reason says so); `path_tick_offset`,
  `recent_attackers`, `impulse_books` and `name` are WRITTEN on the match
  path and never decided on. Property (e) holds it: a bots and a bots_v1
  match with every other R attribute poisoned (raises on any use) and those
  four given garbage play to the same result and digests.
- `awakened` stays R (read by `stats.is_stat_player_visible`).
- The status block is 16 rows (8 non-stacking kinds x 1 + POISONED x
  `STATUS_STACK_CAP` 8); the planted-charge table holds `CHARGE_CAPACITY` 32.
- `current_hp` is a property over the slot `_current_hp` whose setter refuses
  a value off the Q16 grid; no write in the suite or the pinned matches was
  off it.
- Sentinels on the HOST: `current_mag` `MAG_UNBOUND` = -1, `speed_ticks_per_tile`
  `SPEED_FROM_TABLE` = -1. Encoded only (host keeps `None`): `pending_plant`
  `NO_PLANT` -1, a throw's reach `THROW_NO_REACH`, a charge's remote
  `det_tick` `NO_TIMER` -1 and missing approach `NO_TILE` (-1, -1); a table
  column's "none" row -1; an intent column is a presence flag + its ints.
- The record's bytes are `name|int64` rows under `UNIT_SECT_V2`
  (`unit_schema.unit_section_bytes`), the charges `CHARGE_SECT_V1`; the events
  keep their repr. `facing_target_q` hashes as `turn` (the turn still to go),
  `facing_q` as `facing`, `omega_q` as `omega`; every other key is the column
  name.

**As built (P3c, 2026-11-12).** Positions are the synced Q16 ints `x_q` /
`y_q` (schema columns, *always*, hashed as plain ints under the keys `x_q` /
`y_q`); `x` / `y` are read-only float properties (`x_q / 65536`, exact), so
no reader changed. One writer, `Unit.set_pos_q(x_q, y_q)` (refuses a
non-integer); its float door `Unit.set_pos(x, y)` quantizes once, round half
away from zero (`unit_fixed.quantize_scalar`, the rule `unit_impulse.move`
already applied on entry). `tile_x` / `tile_y` truncate on the int, as
`int(x)` did (negative positions included: a test fixture reaches -1).
One line each:
- Every writer, and whether it was on the grid: spawn (`Unit.__init__`,
  `add_unit(position=)`, so `spawn_match`) through the float door -- every
  match spawn is a whole tile, an authored off-grid spawn is quantized (one
  legacy test spawns at 4.37: its assert now allows 2^-17);
  `unit_impulse.move` reads and writes its own ints (was `quantize(u.x)` in,
  `x / 65536.0` out: bit-identical); the zombie step and wander write whole
  tiles (`nx << 16`); the WEGO path and the one-phase timeline go through
  the float door (their `(st + 1) / step_ticks` interpolation is OFF the
  grid: 348 / 310 off-grid unit-ticks in the driver's two legacy scenarios);
  the WEGO round snap is `round(x) << 16` (the same `round`).
- The integer movement law: the only continuous-path step that left the
  integers was `move`'s float round trip, now gone; nothing else was
  changed (every float reader of a position is P3e's).
- Proof (`tests/_p3c_sole_cause.py`, main `229f7097` vs the branch, records
  normalised to `x_q := round(x * 65536)`): EQUAL records and field digests
  over ship_demo bots (2902 ticks), bots_v1 (264), the #122 box (92), the
  scripted arena (420) -- 0 off-grid positions on main -- AND over the
  one-phase (200) and TwoPhaseWEGO (260) scenarios: the legacy quantization
  moved no later decision there.
- The re-pin: unit_stream / tick_digest of nine pinned matches (outcome,
  ticks, field_digest unmoved) and `GOLDEN_AGGREGATE` `3646fd0d` ->
  `4a13fc72` (per-field lines identical bar `__unit_pos__` /
  `__unit_state__`). Recording format unchanged (positions are not
  recorded; P3a's precedent for a digest redefinition). ship_demo bots
  match: 51.66 s -> 51.24 s (medians of three, noise).

### 4.5 Actions, observations, rewards, terminals, the event ring

- **Actions** = the intent buffer `(N, U, A)` int32 whose columns are
  GENERATED from `match_replay.INTENT_KINDS` (minus `drive_zombies`): move
  dx, dy (Q16 unit vector), speed_mode, a `clear_move` flag (distinct from a
  zero vector), aim dx, dy, trigger, throw dx, dy, reach, fuse (INTEGER
  ticks: a §7 change that amends `INTENT_KINDS`'s `fhex` to `int`), use. A
  new verb is a row in `INTENT_KINDS` (its column follows) AND a kernel that
  consumes it: the plumbing is free, the behaviour is a Class-T twin (§10).
  The farm applies the same buffer through the intent facade.
- **Observations** = the sensor accessor's new `crop(...)` method, whose
  device backing is #25's gather kernel: a per-unit CROP `(N, U, C, r, r)`
  of C channels (the accessor's seven, plus material and unit occupancy by
  team; LIGHT IS REFUSED at the door until #81: ACCEPTED GAP) and a
  per-unit VECTOR `(N, U, V)` of schema columns; channels and r from the run
  file; stored fp16 (or uint8 for masks); Q16 → float inside the kernel. No
  global map in v1 (§12.5).
- **The event ring** `(N, E_max, fields)`, kept per tick: every UnitHit /
  UnitKilled / ExplosionEvent appended in the reference's EMISSION ORDER
  (the §4.3 table), with `_event_record`'s fields as integers — damage and
  hp are already on the Q16 grid (`quantize_hp_delta`), so the integer
  record is lossless and no P3e change; the ExplosionEvent's float
  `strength` gets an integer field. The digest's view of the ring filters to
  `_SYNCED_EVENT_TYPES` exactly as today (ExplosionEvent is not among them).
  Overflow is a counted per-env failure flag; so is overflow of every other
  per-env queue (edit records, per-shooter effects, `blast_log`, rounds,
  projectiles), each with a schema capacity. The ring is the source of BOTH
  the digest's event list and the reward counters (lens 2 B2).
- **Rewards and terminals**: `(N, U, E)` per-unit counters derived from the
  ring (the E axis from `events.Cause`: hits dealt / taken, kills, deaths by
  cause, team-kills) plus exposure ticks from the coupling rows; the run
  file weights them. `terminated` from the MatchRules law; `truncated`
  PLACEHOLDER (the time limit is a draw).
- **The training-episode record**: for selected envs the device backend
  writes (tag SHA, bank sha256, template index and source, episode seed,
  decimation, the action rows per step, the ring's checkpoints and digests);
  a converter turns it into a `match_replay.Recording` v1 that replays on
  the reference under `run_match --replay` through every checkpoint. This
  is how a trained agent is WATCHED (lens 2 M7, lens 4 M3).

### 4.6 Reset and the bank

The bank is the levelgen archetype doc's §5.3 bank (`levels/banks/<name>.toml`:
rows of LevelSpec + output digest; train / test-iid / test-ood / probe)
COMPILED by `tools/build_level_bank.py`: each row runs
`gen_ship.generate_from_spec` → `level_loader.load` → `Simulation` →
`spawn_match(LevelData.match)`, and the post-load TICK-0 STATE (every synced
plane, the store, the entity rows and signals, the books, the MatchSpec) is
snapshotted into the device bank file and digest-verified against
`run_match`'s tick-0 checkpoint. Nothing is re-derived at reset:
`reset(mask)` copies the template's state into each masked env, CLEARS
every pending per-env queue (edit records left by 9e, `div_source`,
`expansions`, `blast_log`, rounds, projectiles, the possessed set, the
ring) and sets its Philox key; the bank
records the sha256 of every input it read and the tag's SHA. Level
generation never runs mid-training; held-out sizes are separate banks (one
shape per batch). The bank's build refuses a level whose entity classes,
weapons, status kinds, payloads or coupling rows have no device twin (§6).

### 4.7 RNG

Every match-path draw is Philox-4x32-10 from the vendored kit (cherry-picked
from `63-swarm-units`: `cpp/src/philox32.h`, `src/simulation/philox32.py`,
chapter 17 §3's layout): key = the env's episode seed, counter = (tick,
draw_index, stream_salt, unit_or_site_id), one salt per (slot, site) in ONE
enum, `draw_index` local to the site, an in-flight round carrying its spawn
ordinal. The draw sites a match runs: the shot cone, exposure and crit rolls
(including melee), the beam cone; charges, timed charges and heat sources
never draw. The CPU reference makes the SAME draws through the Python twin;
`spawn_match` installs a refusing stand-in for `sim.rng` on the match path,
so a new `rng.uniform` raises in the ordinary suite (lens 4 M5). **The
FieldEdit noise draws Philox on EVERY path** (keyed (episode seed, tick,
edit key, tile)), not only in a match: the flush is Class P (the played
`--resident` game runs the device queue), the played game cannot ship
PCG64's state to the device, and PCG64 is never ported (re-check B2); this
is one re-baseline of every noisy blast in the played game (P3b), with no
feel change. `sim.rng` stays for WEGO's timeline and the ruleset's legacy
draws only. Golden re-baselines (§7).

### 4.8 The device edit queue

`EditQueue.flush` is relocated: records `(env, key, field ordinal, region,
coords, value, mode, policy)` go into a pinned buffer, the key a composite
`(enqueue phase, slot, walk row or effect index, k)` with phase 0 = the
PREVIOUS tick's 9e (timed charges, heat sources), so the reference's
cross-tick `seq` order is reproduced (re-check M3; an overlap scene gates
it), no device atomic counter; the field ordinal table keeps the string
fields' lexicographic order; the per-env applier applies them in that order
with the same per-field policies and combine functions as device functions,
booking into the per-env seam channels. FieldEdit goes INTEGER at the
enqueue door (§7 P3e): the dequantize → double → requantize bridges, the
`amount · weight · mult` product, `linear_weights` and the DISC weights
become integer arithmetic, so the device port is exact by construction; the
noise is Philox (§4.7). P4c therefore follows P3b and P3e's FieldEdit item
(§9). Semantics otherwise unchanged; the host never reads a field to apply
an edit.

### 4.9 Pathing: the distance field

`pathfinding.astar` stays the host pathfinder (the bots, tools). The horde
walks a PER-ENV DISTANCE FIELD on BOTH paths after P3d: integer BFS distance
to the nearest living marine (path-nearest, which replaces today's
Euclidean-nearest target choice at `ai_zombie.py`'s activation) over the
passable mask with the footprint's passability, recomputed every
`[zombie] field_refresh_ticks`, descended by comparing the four neighbours
in a fixed order; activation and melee unchanged; the law's `math.sqrt`
gates become integer compares of squared distances. The reference field is
C++ (a `PhysicsEngine` call) so the played game's zombie slot stays near
today's 0.15 ms. On the device: one block per (env, field), the frontier
loop in the block, int16 distances in shared memory. Erik approved the
behaviour change; P3d is a HUMAN-TEST on main with a written spec; it amends
the A* row AND the zombies "WRAPPED, never rewritten" row
(`test_zombie_control.py` re-pinned). The bots keep A* on the host (§12.3).

**Host half built in #183 G9p (2026-11-11).** The C++ reference flood
(`cpp/src/horde_field.cpp`, `PhysicsEngine.horde_field`, a host call in both
modules), the Python law on it (`src/simulation/horde.py`, `ai_zombie.py`),
`[zombie] field_refresh_ticks = 2`, and the behaviours designed onto the field
(wander, hearing, fear of fire, door bashing, the sprint;
`docs/bots_zombies_speeds_design_2026-11-10.md` §3 as built). Deviation: the
descent's tie order is per cell parity (vertical first on even x + y), not one
fixed order, so a horde walks the staircase toward a diagonal goal instead of
an L. What remains for P7d: the device kernel (one block per (env, field),
the frontier loop in the block, int16 distances; host sentinel is
0x7FFFFFFF), the masks (walkable, bashable doors, feared tiles) and the
noise fields on the device, the per-zombie descent as one thread per zombie,
and the bites through the applier; the field cache's refresh rule (drop on
`tick % N == 0` or a topology change, flood on first need) is the contract.

### 4.10 Zero-copy export, streams, syncs

Persistent CuPy buffers exported to torch through DLPack; the engine runs on
torch's current stream (`cupy.cuda.ExternalStream`) or records an event
after `step` that the torch side waits on; never a device-wide sync. Both
memory pools capped (`run.device_pool_mib`); the engine's own scratch sits
inside the capped pool. The 8 GB card bounds N with the observation and
rollout buffers counted (12 channels x 21² fp16 ≈ 10 KB per agent per step).

### 4.11 Device table upload

Every gameplay table a kernel reads (weapons, ammo, loadouts, species,
environment profiles, the status registry, payloads, `[exchange]`,
`[zombie]`, `[movement]`, materials, gases) is uploaded from the validated
Python table through a COLUMN SCHEMA at batch build; a new column is a
schema row, never a hand-edited device struct (lens 4 M4).

### 4.12 CUDA graphs and kernel throughput (P9)

After batching the tick is a fixed launch sequence per `(N, h, w)`: capture
once into a CUDA graph (#24's spike: non-blocking stream, thread-local
capture mode), a WHILE node for the substeps, replay per tick. Then the
measured items in this order: the sweep as one block per (env, ordinate);
the EOS chain's pass count and fusion where data stays on chip; the 128-bit
multiplies only after Nsight Compute shows a kernel compute-bound, as a
CPU-reference-first arithmetic change through `sweep_ref_q` where the sweep
is concerned. Targets come from P2's Nsight capture.

## 5. Determinism and gates

1. **Reference bit-identity.** Every device stage equals the CPU reference
   at tol 0 on every synced field and every book (`cuda_*_check.py`;
   `cuda_s8a_check.py` extended per stage). P5's first check: the device
   pre-stage equals the host pre-stage at N = 1 (`n_sub`, `c_local`, `cap2`).
2. **Batch invariance.** Env i of N equals env i alone, bit for bit, for N
   in {1, 2, 8, 64}, with envs holding different templates, seeds and
   substep counts, one env with a topology event while the others have
   none, and one env resetting mid-batch.
3. **Live replay lockstep (the gameplay twin oracle).** A match is RECORDED
   on the reference at HEAD and replayed on the device backend in the same
   run; the per-tick `(field_digest, tick_digest)` agree at every
   checkpoint. Nothing stored: a stored corpus stales weekly (lens 4 B1).
   `require_snapshot(backend="device")` runs a pre-flight of K random-action
   episodes on the run's own bank, loadouts and tables across both backends.
4. **Backend equality** (§4.1).
5. **Coverage counters.** Shots per weapon row, reloads, statuses per kind,
   entity classes, coupling rows, payloads; the pre-flight refuses a feature
   in the run's config with zero exercise.
6. **The two-phase apply.** Phase A decides against the start-of-tick state
   and records; the per-env serial applier applies in the reference's order
   (`SimultaneousEffects.resolve`'s walk order; the first packet to zero is
   the kill, as today — no conflict resolution exists to port).
7. **Topology seam gates.** A device twin of `test_gas_energy_tile_flip`
   plus the closure identity on scenes with a burst wall, a burn-through, a
   chew-to-zero, a door cycle, a detonation, a bite kill and a vent tick.
8. **#71** fixed first (P2): the CUDA combustion twin onto the #54 seam, a
   burning-scene gate at tol 0, validated by breaking it.
9. **The digest.** The unit hash (`UNIT_DIGEST_KEY`, folded by `tick_digest`,
   no version of its own) is redefined from the schema in P3a: WEGO `ap` /
   `n_orders` leave it, hp hashes as a Q16 integer, events as the ring's
   integer records; positions keep `round(v·1e9)` until P3c makes them Q16.
   Each P3 commit is its own deliberate re-pin (five re-pins: a, b, c, d, e)
   with written rationale; `GOLDEN_AGGREGATE` MOVES at P3a and P3e (the
   golden scenario has a unit, and `trajectory_digest` folds the unit hash;
   `_xarch_perfield_digest`'s unit-subfield reader changes with the record)
   while its per-field lines stay unmoved — re-baselined, not "checked
   unmoved". ONE digest function (Python) over a D2H of the planes, the
   store, the ring, the entity rows and the signals at checkpoint cadence
   (the device's entity SoA exports through `serialize.py`'s encoding, so
   the `__entity__` and `__signals__` sections hash as today); never a
   device digest. Obs tensors are not digested.
10. **Books.** The closure identity is checked per env at checkpoint
   cadence over the WINDOW since the previous checkpoint (the books
   accumulate; Σ `gas_energy` is read at both ends); every cumulative
   channel carries an int64 range argument in the manifest (or a counted
   wrap). `n_destruction_seed_sum`'s device form is the destroyed tile's own
   bulk delta.

## 6. Snapshots, the two classes of device code, twin debt

**Snapshot.** A training snapshot is an ANNOTATED tag `train/<name>` placed
by Erik on MAIN, pushed, never moved; checked out detached in its own
folder outside `.claude/worktrees` (`git worktree add --detach
../breach-train-<name> train/<name>`), with its own CPU and CUDA builds
(never a `.pyd` copied between checkouts; each snapshot costs a full build,
on Berzelius a container build from the tag's `containers/breach.def`), its
outputs (checkpoints, logs, episode records) outside the checkout, and an
in-use marker the git-workflow audit honours (the skill gets a
training-checkout row). A hot-fix lands on main as a normal patch; if a run
cannot move to HEAD, a `<issue>-train-<name>-fix` branch from the tag takes
the cherry-pick and Erik tags `train/<name>.1`.

**`rl_env.require_snapshot(tag, backend)`** refuses unless: HEAD's SHA is
the tag's commit SHA (`git rev-parse train/<name>^{}`); the tree is clean
INCLUDING untracked files under `src/`, `levels/`, `config.toml` and
`cpp/`; `bp.__file__` lies under this checkout's own `cpp/build*` and
`simulation.__file__` under its `src/`; `bp.build_id()` (generated at BUILD
time into one generated translation unit, so an incremental build cannot go
stale) matches HEAD; the bank's recorded SHA matches; and, for the device
backend, the debt ledger is empty for everything the run uses and the
pre-flight passed. The run manifest records the tag SHA, the bank sha256,
the run file's sha256, CuPy / torch / numpy / CUDA runtime and driver
versions and the GPU; a resumed run refuses on a change.
`match_replay.build_id()` stays THE describe helper (`--tags
--exact-match` for the tag check); `bp.build_id()` is the module's
provenance only. The FARM trains at any tag and needs no ledger.

**Class P and Class T — every device artefact has a class.**

| artefact | class | why |
|---|---|---|
| the physics cores (water, EOS, MG, kick, trace tail, sweep, combustion, temperature, fire), the fold table, the stream, the per-env books' physics channels, the batched engine at N = 1 | P | the played `--cuda` / `--resident` game runs them |
| the device edit queue + the applier's EDIT APPLICATION + the FieldEdit policies and combines + the gas-energy seam functions they call; the stamp kernel; the table upload of materials and gases | P | the played `--resident` game flushes through them from P4c |
| the applier's topology, payload, combat, bite, door and entity functions; the unit store, the schema's device side and the bridge; the horde field; the reset kernel and the bank; the ring; the obs / reward / `done` kernels; the table upload of weapons, ammo, loadouts, species, statuses, payloads, `[exchange]`, `[zombie]`, `[movement]` | T | only training runs them |
| a seam function called from both classes | P | a change to it moves the P twin in the same patch AND creates T debt for the T callers |

| class | discipline |
|---|---|
| **P** | kept level PER PATCH on main (reference → C++ → CUDA in the same change, bit-identity gates); never in debt. ENFORCED, not assumed (re-check B1): the twin manifest lists Class-P twins too (`class = "P"`); a green run of the CUDA suite writes an untracked stamp `.cuda_green` holding HEAD's SHA; an ordinary-suite test that never skips (`tests/test_class_p_level.py`) fails when the branch's diff from its merge-base touches a Class-P reference path and the stamp does not match HEAD ("build `cpp/build_cuda` and run the CUDA suite before merging"); the git-workflow merge step refuses on the same condition. A change to FieldEdit, `stamp_units`, the seam or a resident driver IS Class P and needs a CUDA build in that worktree. The cost is today's ratio of CUDA to C++ commits, now visible. |
| **T** | carries DEBT under the ledger; brought level in catch-up patches; refused at the training doors while in debt |

The played `--resident` game keeps Python gameplay over device physics;
whether it ever runs Class T is a ruling for Erik (§12.6).

**Twin debt is computed, never remembered.** The twin manifest
(`tests/twin_manifest.toml`) maps each twin to the reference FILE SET it
mirrors and the commit it was last level at (`level_at`). The file sets are
GENERATED, not hand-listed (re-check M6): `tools/twin_manifest_gen.py` runs
the live-lockstep corpus on the reference under a coverage tracer and
records, per slot, the files executed, so transitive helpers (`unit_fixed`,
`movement.default_speed`, `has_los`, `find_burst_walls`, `species.py`,
`weapons.py`'s normalisation, `blast_expansion`, `blast_air`, `events`,
`factions`) are in the set of every twin whose slot runs them; a scan test
asserts every module under `src/simulation/` reachable from the match path
is in some twin's set or listed `NOT_TWINNED` with a reason.
`tools/twin_debt.py` lists every twin whose set changed in
`git diff level_at..HEAD`; `tests/test_twin_debt.py` asserts the manifest
covers every twin (enumerated from the manifest itself, so no build is
needed) and that each `level_at` is an ancestor of HEAD. The training-side
doors (`build_level_bank`, the run file, `require_snapshot`) refuse a
feature without a twin or with non-empty debt. A **catch-up patch** re-ports
the drifted twin from the reference, its oracle the live lockstep on the
run's own bank with the coverage counters non-zero for the feature, and
moves `level_at`. Live lockstep runs the reference (the CUDA build's CPU
path) and the device in ONE `data` subprocess (the CUDA harness pattern).
Scattered `xfail` marks are NOT the mechanism (lens 4 B1): a gameplay patch
in a fresh worktree skips every CUDA gate and would never place one.

## 7. Reference-side changes (each a deliberate re-baseline of the played game, BEHIND G1)

| change | why | gate | feel | cost of not doing it |
|---|---|---|---|---|
| P3a the unit schema (+ `__slots__`, capacities, the digest redefinition) | one definition of a unit's synced state | digest re-pin with rationale; property test "every attribute is a schema row or REFERENCE_ONLY" | none | four drifting definitions; a new attribute never reaches the device |
| P3b Philox at the match draw sites + the refusing `sim.rng` stand-in | per-env keyed draws | goldens regenerated; the kit's KAT + layout tests | none | no device draws can match the reference's stream |
| P3c positions to Q16 (`x_q` / `y_q`; float properties kept) + the integer movement law (BUILT 2026-11-12, §4.4a's As built (P3c)) | the iron rule; integer digests | goldens regenerated once; movement A/B in dequantized units | none: Erik blessed Q16 without a play test (§12.1, 2026-11-07), so the HUMAN-TEST is waived | float64 columns on the device (IEEE, `--fmad=false`) and a standing exception to the iron rule — Erik's option B in §12.1 |
| P3d the distance field (both paths, C++ reference) | per-unit A* has no device shape | goldens; ship_demo seeds 1–8 before / after; the A* and zombies rows amended | HUMAN-TEST on main | the horde stays on the host: no device backend  **Built in #183 G9p (host half; §4.9)** |
| P3e the match path to integer (the table below) | no float in the ported slots; exact device ports | goldens; A/B of hit rates and blast outcomes in real units over many seeds | HUMAN-TEST on main (shot and blast feel) | an IEEE-double transcription per site (op order pinned, `--fmad=false`), gated by lockstep: doable, fragile |

P3e's site table (re-check M4; each row decided integer unless marked, and
named with the applier function or kernel it lands in):

| site | today | decision | lands in |
|---|---|---|---|
| the shot cone (`combat.py` ~1058–1095: `math.radians`, `rng.uniform`, float rx / ry) | float64 | integer: `philox angle_q16`, a Q16 line march through `unit_fixed` | phase A (4) |
| exposure / crit / melee rolls (`attack_resolver.py` 80, 138; `combat.py` 1714) | float compares | `uniform_q16` compares | phase A (4), bites (5) |
| mitigation and hp (`damage.py` 11–22, 82–90, 260; `unit.py` 216) | float then a Q16 snap | Q16 throughout | `apply_packet` |
| `int(dmg · bullet_damage_multiplier)` (`combat.py` 406) | float product | `mul_q16` | phase A (4) |
| `math.sqrt` range gates (`combat.py` 855, 1009; `ai_zombie.py` 77, 93, 123) | float | squared-distance integer compares | phase A (4), zombies (5) |
| `_neighbor_mean`'s true division (`gamemap.py` 2130, 2389) | float | integer floor division | stamp refill (6), `destroy_wall` |
| FieldEdit's bridges, `amount · weight · mult`, `linear_weights`, the DISC weights, `_iter_region`'s falloff | float64 | integer | the edit application (6b) |
| `apply_explosion`'s sqrt falloff, wall decrement, `wave_source` amounts; `apply_blast_damage`'s sqrt and `int(max_damage · falloff)` | float | integer (`sqrt_q16`, `mul_q16`) | `execute_payload` |
| the payload heat product; spray amounts; grenade flight | float | integer | `execute_payload`, spray deposit, slot 2 |
| 9c's heat chain (`exchange.py` 384–401), 9c3's poison dose (`:709`) | float | integer | the coupling-row kernels (9c, 9c3) |
| the grenade fuse (`fuse_seconds`) | float seconds | integer ticks at the door; `INTENT_KINDS` amended | 1b |

Order: P3a, P3c and P3e before P6 (P6 needs the schema, Q16 positions and
the integer hp / mitigation); P3b before P7a (the shot cone's draws) and
before P4c (the noise); P3d before P7d. Each merged to main on green with
its re-baseline rationale written; none before G1 says the device route
continues.

## 8. Linux, the container, Berzelius

- `cpp/CMakeLists.txt` already carries the gcc/clang strict flags
  (`-fno-fast-math -ffp-contract=off`) and `--fmad=false`; the CUDA arch
  list (75 / 86 / 89) gains 80 (A100) and 90 (the Hopper phase).
- `cpp/build_linux.sh` (CMake + Ninja) and `containers/breach.def` (CUDA
  12.4+ devel base for the conditional graph nodes; the environment
  installed through the ml-stack skill's `ml_stack.py`, ONE version set,
  never an `environment.yml`) are the patch; the suite on Linux must pass
  and the digests must equal Windows's.
- P10 lands early (after P5) because G1's decisive reading is on a node.
- First Linux box: this PC once SVM is enabled in the BIOS and WSL2
  installed (Erik, 2026-11-07 evening), else the Lenovo, else the login node.
- Multi-GPU: one process per GPU, N envs each; the engine never assumes one
  GPU. Policy placement (one A100 shared vs split) is the agent era's.

## 9. Patch plan

Each patch is its own branch off main, one worktree agent, commits as it
goes, merged `--no-ff` to main on green by the agent that ran the gate
(HUMAN-TEST excepted); no long-lived arc branch (lens 4 M1: a months-long
branch would turn all of main's interim changes into debt). The docs and
bench tools on `29-residency-batching` merge first. Tiers: oracle-gated →
Sonnet; judgement or no oracle → Opus; effort default unless stated.

| # | patch | gate | tier | HUMAN-TEST |
|---|---|---|---|---|
| P1a | `MatchSession` factored from `run_match`; the `policy` controller; `BatchedEnv` with the `cpu_farm` backend; the run file door; the random-action smoke driver; the farm MEASURED by the §9.1 protocol | N workers reproduce `run_match`'s digests per seed; the smoke run; the farm numbers in the benchmark doc | Opus | no |
| P1b | `require_snapshot` (farm form), `bp.build_id()` at build time, the run manifest, the twin manifest + `tools/twin_debt.py` + `test_twin_debt`, the git-workflow training-checkout row | the doors refuse each listed mismatch (tests) | Sonnet | no |
| P2 | #71 (the combustion twin onto the #54 seam); the CUDA second-run slowdown measured; an elevated Nsight capture of the resident tick (Erik runs the command once; the agent reads it) | burning scene tol 0, validated by breaking it; the capture's kernel / memcpy / bytes table in the benchmark doc | Opus | no (one human step: the capture) |
| P4a | resident drivers for combustion / temperature / fire / the sweep + the water tail, inside the resident tick; `headless` ripple | `cuda_s8a_check` extended, tol 0; zero plane transfers inside slot 7 | Sonnet | no |
| P4b | the stamp_units kernel (five planes) | tol 0 | Sonnet | no |
| P5 | batch the physics: `(N, h, w)` on every core, the fold table, per-env MG, per-env `n_sub` + both masked loops, the device pre-stage, per-env books, one stream, the capture test; the profiler's applier-ms row reserved | N = 1 == resident; batch invariance; pre-stage N = 1 check | Opus | no |
| P10 | Linux build + container + the suite on Linux; `cuda_harness.py`'s Windows-only paths ported | digests equal to Windows | Sonnet | no |
| **G1** | the §3 scene (no units, entities on) device vs farm, per slot, on a node | the gate factor (§3) | | Erik reads it |
| P3a, P3c, P3e | the schema, Q16 positions, the integer match path (§7), one branch each, on main | per row | P3a Sonnet; P3c, P3e Opus | P3e only (P3c's waived, §12.1) |
| P3b, P3d | Philox (+ FieldEdit's noise on every path), the distance field; P3b before P4c and P7a, P3d before P7d | per row | P3b Sonnet; P3d Opus | P3d |
| P4c | the device edit queue (Class P; after P3b and P3e: integer policies, Philox noise, the phased key) | one tol-0 scene per policy row + the cross-tick overlap scene | Opus | no |
| P6 | the device unit store + schema's device side, the Unit ↔ store bridge, slots 1 / 1b / 2b / 3 / 9 / 9b / 9c–9d / 10 as kernels, the per-env serial applier (block form) with the topology seam as device functions, the event ring, the entity export for the digest | live lockstep; the topology scene gates | Opus | no |
| P6b | entities on the device per class with per-class scenes (airlock_demo, a heat-source + heat-trigger studio, a vent studio); the #25 gather as the accessor's device backing | live lockstep per scene | Opus | no |
| P7a | shooting + sprays phase A / B on the device | live lockstep | Opus | no |
| P7b | projectiles and grenades in flight; fuse-outs | live lockstep | Opus | no |
| P7c | payloads through the applier (the walk, deposits, blast damage, div_source, credit) | live lockstep + the detonation scene | Opus | no |
| P7d | the horde on the field; bites through the applier | live lockstep | Opus | no |
| P8a | the bank compiler + reset (copy + key; per-episode seed function) | tick-0 digest equality per template; mid-batch reset invariance | Sonnet | no |
| P8b | the obs crop (accessor method + kernel), the reward counters, `terminated`, the table upload, zero-copy export, the pre-flight and the coverage counters = the `device` backend | backend equality | Opus | no |
| P8c | the training-episode record and its `Recording` export | replays under `run_match --replay` through every checkpoint | Sonnet | no |
| **G2** | equal game, per slot, on a node | the gate factor | | Erik reads it |
| P9 | the graph + WHILE node; the sweep's single launch; EOS fusion by Nsight's order | bit-identity; zero syncs on any tick; **G3** | Opus | no |
| close | the first `train/*` tag on main (Erik); CLAUDE.md rows + amendments (§13 b, c); skills amended; #29 / #77 / #24 / #25 / #71 closed; Roadmap #46 | | | |

### 9.1 The farm measurement protocol (P1a)

The `cpu_farm` backend with the random-action driver, placeholder
observation and reward rows ON, decimation 1 and 4, with 7 / 8 (here) and
14 / 16 (a node) worker processes plus a trainer stub, at least 3 episodes
per worker through reset, on the small and the medium ship; report
env-ticks/s, p95 step time and RSS per worker; every later reading uses
the same driver, tables and decimation.

## 10. Rules for continued game development (after the port)

| the change | what it costs |
|---|---|
| a gameplay change on the reference path (Python / C++) | as today; the played game never waits for a kernel |
| a new per-unit field | a schema row (free: digest, serializer, obs, bridge follow); the device kernel that USES it is a Class-T twin at catch-up |
| a new verb | a row in `INTENT_KINDS` (its column follows); the kernel that consumes it is a Class-T twin at catch-up; until then the device backend refuses a run that uses it |
| a new table column (weapon, status kind, payload row) | free if schema-uploaded; a new BRANCH in the law is a Class-T twin |
| a new entity class | the class + its runtime (as today) + a device SoA layout, a kernel and bank rows: about 2 to 4x today's cost; the bank refuses the class until its twin exists |
| a physics change in the shared C++ | Class P: reference → C++ → CUDA in the same patch, as today; a running training is untouched (its own checkout and build) |
| a new coupling row (`exchange.py`) | a Class-T twin at catch-up; its per-unit state obeys the schema's capacities |

Three rules with CPU-side gates in the ordinary suite keep the reference
twinnable: match-path draws only through Philox with a registered salt
(`spawn_match`'s refusing stand-in); new per-tick match-path arithmetic is
integer (a float ratchet over the match-path modules, counts only go down);
every per-unit collection has a schema capacity and an overflow rule. Class
P is never in debt; Class T's debt is computed by the ledger and refused at
the training doors; a catch-up patch re-ports from the reference against
live lockstep. Training runs on a `train/*` tag only.

## 11. Accepted gaps (each a decision)

- ACCEPTED GAP: ambient (planetside) maps are outside the device backend
  (the sky exchange stays a host bracket); v1 training is on space maps.
- ACCEPTED GAP: ripple is not computed on the device backend; nothing in
  the sim reads it.
- ACCEPTED GAP: no light channel in observations until #81 separates
  light's inputs; the obs door refuses it.
- ACCEPTED GAP: the field recorder (`recorder.py`) does not run on the
  device backend; episodes are re-simulated from the training-episode
  record.
- ACCEPTED GAP: WEGO and its timeline are absent from the device backend
  and lose their `ap` / `n_orders` digest coverage (WEGO is legacy).
- ACCEPTED GAP: a scenario whose envs detonate every tick runs the serial
  applier hard; no v1 scenario does, and it is measured.

## 12. Rulings from Erik (2026-11-07)

Recorded as given; the numbering is the question's.

1. **Positions: A, Q16 -- RULED.** `x_q` / `y_q` on the reference path, the
   iron rule, integer digests, readers unchanged through float properties.
   Erik blessed it without a play test, so P3c's HUMAN-TEST is waived (§7's
   feel cell and §9's P3c cell say so); option B (float64 columns on the
   device) is not taken.
2. **The planned training scale:** not set. Erik leaves it to the G readings;
   the working factor (3x the farm per slot, §3) stands.
3. **Bots in training: wanted.** Rule-based bots bootstrap early training,
   pure RL comes later (Erik will survey bootstrapping methods himself). How
   they run beside device envs -- the farm, a hybrid batch (host bots on the
   slot's cores), or a later device bot on per-goal fields -- is decided at
   G2 with numbers. v1 training on the device backend is self-play or
   against the horde.
4. **A cheaper training physics preset: no, not now.** Port first (the
   snapshot principle holds; the curriculum uses the small map). A lower
   resolution is held in reserve only if the G readings fall short.
5. **Observation: both shapes.** The per-unit crop AND a reveal-all full-map
   option, both offered by the observation door (P8b).
6. **Class T in the played `--resident` game: no.** Claude's recommendation;
   Erik had no objection. The played game keeps Python gameplay over device
   physics.
7. **G1 (2026-11-10): the device route continues -- RULED.** The reading is
   `docs/residency_benchmark_2026-11-07.md`, "G1: the reading": measured on
   the Home Desktop, the batched path is at 0.34x (small) / 0.78x (medium) of
   the 16-worker farm today; projected with the host work gone and the
   allowance charged, 2.2 to 3.0x the full-game farm for the device alone and
   3.1 to 3.9x with a farm on the slot's spare cores (§3's metric); per A100
   slot 2.5 to 4.3x and 3.4 to 5.2x at the integer factor. Erik, with those
   numbers in front of him: yes, continue, and the kernel throughput work
   (P9 and beyond) is wanted after the flip -- "this is the whole point of
   the project". So the arc proceeds to P3a once the gameplay pack (#183) has
   landed. The 3x working factor was not re-ruled: the ruling is on the
   route, at a projected 2 to 4x per slot before P9. Two consequences for §3:
   the yardstick on an 8-core / 16-thread box is the 16-worker farm (P1a
   measured 8), and G2's farm row runs one worker per hardware thread.
   RULED (Erik, 2026-11-11): the PC reading stands as G1 without a node run
   ("yes absolutely"); the 15-minute calibration on one slot stays worth
   doing the first time Berzelius is in use, as a calibration, not a gate.
   Also 2026-11-11: the over-pressure wall failure (slot 9b) is switched off
   (`[physics] burst_enabled = false`, #186: a redesign in its own session),
   so the port writes no kernel for it.

## 13. Systems

**(a) Existing canonical systems this design uses** (the twin or the host
path must keep each one's rule): Simulation facade; tick conductor (the
slot order is THE order the device tick keeps); GameMap (topology changes
on the device through device forms of the same functions); FieldEdit (the
device queue is ITS flush); payload executor (its device form in the
applier); generic explosion, divergence source, blast reach, blast air
read, fixed-total tile deposit (all written by the applier's device
functions, `div_source` included); starting a fire (no new igniter);
gas pump primitives (the vent kernels call the same law); vent / duct
system and its plenum ledger (a serial stage per duct); heat source +
`blast_log` (a per-env device list); timed charge (its trigger on the
device, its payload through the applier); PhysicsRunner / PhysicsEngine
(the batched step is engine orchestration); fixed-point kits and
`unit_fixed`; Q16 boundary modules; trace currency; config (the fold
table, the device table upload); material / gas tables; coupling table,
wind load, footprint samples builder (their rows as kernels; the builder's
`src/` scan gate kept); vision / LOS (`has_los`, `sight_screen` on the
device); weapons (no direct hp write on the device either); unit /
loadouts / env profiles (the schema's source); entity system + serializer
+ sensor accessor (the schema reuses its kinds and encoding; the `crop`
method; #25's gather); extinction planes and prop stamps (bank-time);
recorder (unchanged; the device backend does not run it); field digest +
GOLDEN_AGGREGATE + A/B lockstep + CUDA harness (every gate; the digest's
change procedure for the re-pins); ingress lint, float ratchet,
`/fp:strict` (the obs float conversion lives in a non-scanned TU);
RL-batch habits (absorbed into this design's rules); the integer reference
`sweep_ref_q` (P9's sweep arithmetic lands there first); control seams
(intents are the action schema; `ControlSource` unchanged); match
controller registry (`policy` joins it); match config / spawn rooms (the
bank compiles through `spawn_match`); match recording / replay (the twin
oracle; the episode export); credited cause (the ring's `Cause` axis);
simultaneous attacks (phase A / B maps to the kernels and the applier); A*
(the host pathfinder); headless match runner (the `MatchSession`); level
data layer + LevelSpec + the §5.3 bank format (the device bank compiles
them); light accessor (refused in obs until #81); benches (+
`tick_stage_profile.py`, `bench_resident_scaling.py`, the batched sweep
bench); radiation sweep (its batched core); gas energy seam + closure
identity (per-env books, the same channels); Philox (vendored from the
swarm branch, chapter 17 §3).

**(b) New systems and their draft rules (enter CLAUDE.md at implementation):**

1. **Batched environment** (`src/rl_env.py`): THE one env; two backends,
   one API, one run file; a trainer never touches a Simulation or a kernel.
2. **MatchSession + the `policy` controller** (`tools/run_match.py` →
   `src/simulation/match_session.py`): THE steppable match; `run_match` and
   the farm worker are its callers; a policy drives units only as this
   controller.
3. **Batched resident engine**: every launch core `(N, h, w)`; per-env
   scalars as device arrays; a batch homogeneous in shape / boundary /
   config; one stream, no allocation in a step; batch invariance is a gate;
   never a tall grid.
4. **Per-env serial applier**: THE device form of every order-dependent
   law; effects recorded in phase A, applied in the reference's order by one
   thread per env through device functions with the same books; no host
   rare-event path.
5. **Device unit store + unit schema + the Unit ↔ store bridge**: ONE
   schema (entity kinds, `UNIT_SECT_V2`, absence-transparent) that the
   digest, serializer, recorder, obs and kernels read; `Unit.__slots__`;
   capacities at the door.
6. **Device edit queue**: the FieldEdit flush on the device with a
   deterministic composite key; the host never reads a field to apply an
   edit.
7. **Event ring**: the one source of the digest's events and the reward
   counters; emission order; counted overflow.
8. **Per-env books**: `(N, slots)` int64 under one channel enumeration;
   closure checked per env.
9. **Device table upload**: every gameplay table a kernel reads, through a
   column schema.
10. **Bank compiler + reset** (`tools/build_level_bank.py`): the §5.3 bank
    compiled to tick-0 states; reset copies; refuses untwinned features.
11. **Distance-field pathing**: the horde's pathing on both paths; one
    kernel shape; A* stays the host pathfinder.
12. **Philox draws + the refusing `sim.rng` stand-in**: the match path's
    only draws; one salt enum.
13. **Training snapshot + `require_snapshot` + the run manifest**: tag SHA,
    clean tree incl. untracked, own build, own module, matching bank.
14. **Class P / Class T + the twin manifest + `tools/twin_debt.py` +
    `tools/twin_manifest_gen.py` + `.cuda_green` + `test_class_p_level`**:
    every device artefact has a class; computed debt from coverage-generated
    file sets; Class P enforced by the stamp test and the merge step;
    refusing training doors; catch-up by re-port.
15. **Training-episode record + export**: how a device episode becomes a
    `Recording`.
16. **Build id at build time** (`bp.build_id()`).
17. **Linux build + container** (`cpp/build_linux.sh`, `containers/breach.def`
    through `ml_stack.py`).
18. **Tick stage profiler** (`tools/tick_stage_profile.py`): THE headless
    per-slot / per-stage / bus instrument, the sibling of `frame_profile`.

**(c) Amendments (each assigned to the patch that makes it true; the canon
fold itself stays DEFERRED by ruling, so the chapters are listed as owed):**

| row or chapter | amendment | patch |
|---|---|---|
| Simulation facade "the only tick entry" | + the batched engine is the device backend's tick entry | P6 |
| Tick conductor "slot 5 ONE driver" | + the device tick mirrors the slots; the horde kernel is the device driver | P7d |
| GameMap "topology only via destroy_wall / seal_tiles" | + their device forms, called by the applier | P6 |
| FieldEdit | the device queue is its flush (amend, not a second row) | P4c |
| Payload executor | + its device form | P7c |
| A* "the only pathfinder"; zombies "WRAPPED, never rewritten" | the horde walks the field on both paths; A* is the host pathfinder | P3d |
| RL-batch habits | absorbed into the batched engine row | P5 |
| Unit "one class" | + the schema and `__slots__` | P3a |
| Weapons | + the device table upload | P8b |
| Match controller registry | + `policy` | P1a |
| Replay "ONE tap" | + the episode export | P8c |
| Headless runner "never a second loop" | the `MatchSession` is the loop | P1a |
| Spawn rooms "ONE match spawn" | the bank compiles through it | P8a |
| Recorder | unchanged; not on the device backend (stated) | P8b |
| Sensor accessor | + `crop`; #25 as its device backing | P8b |
| Benches | + `tick_stage_profile.py` | the docs merge |
| engine/02 "entity / logic state stays CPU" | false for the training twin (Class T) | owed |
| ml/01 §1 "the environment IS the Simulation facade; no AI-mode fork" | the device backend is a gated twin, accepted by the snapshot ruling | owed |
| engine/10 §3, mechanics/02, engine/14 | pathing, combat currency, the Philox door | owed |
| git-workflow skill | the training-checkout row | P1b |

## 14. Critique lenses (verdicts on v1, one critic at a time; resolutions in v2)

| lens | verdict | blockers | resolution |
|---|---|---|---|
| determinism / residency: bit-identity, batch invariance, masks, atomics, the two-phase apply, RNG, syncs | **NOT SOUND as written** (critique `dd058b6`, `docs/residency_batching_critique_1_determinism_2026-11-07.md`): the architecture holds (batched cores, per-env masks, integer atomics, the two-phase shape, the replay oracle) but two named mechanisms cannot deliver §5's claims | **B1** §4.3/4.8/5.5: phase B carries host-only effects (`chew_wall`, `execute_payload`, `destroy_wall`) and device slots raise topology events every slot (burn-through, 9b bursts, doors, fuse-outs, charges, heat-source deaths), so "zero host sync" and the one-thread-per-env walk-order loop both fail; fix: `destroy_wall` / `seal_tiles` / `on_tile_changed` / the wall law as device functions with per-env atomics, and for payloads either a device blast kernel or a declared stop/resume split with a reworded claim; gates for a burst, a burn-through, a chew-to-zero, a door toggle. **B2** §4.5/5.3/5.7: the device cannot compute `tick_digest` (events hashed in emission order with float damage; WEGO `ap` / `n_orders` hashed; `round(v·1e9)`); fix: a per-env device event log in the reference's emission order, `_unit_record` redefined from the schema in P3's commit with the corpus re-pinned, ONE digest function over a D2H of store + log. **Majors** M1 `n_sub` drives two loops and four host double folds (fold table per batch, device-built planes, both loops masked); M2 interim substep control before P9 (8 masked iterations or one counted 4-byte D2H); M3 the books need a per-env `(N, slots)` int64 home with assign/accumulate semantics and tick-start reset; M4 the edit queue needs a deterministic composite `seq` key, serial-per-tile application, a decision on its float64 combines (integer FieldEdit at the door as a §7 re-baseline, preferred), seam atomics, a named Philox counter for the noise; M5 the payload path reads/writes far more than stated (wall_hp, units, gas through the pump primitive, `div_source`, `expansions`, TickCredit, `blast_log`); M6 many float64 decision sites in combat / damage / zombie AI and the kit has no double draw — inventory each: IEEE transcription or integer re-baseline; M7 no stream is threaded, ~200 sync memcpys + 18 device syncs, RAW/WAR hazards on action/obs buffers, scratch outside the pools; M8 in-step auto-reset needs per-env seeds as a pure function of (run seed, env, episode) on the device or batch invariance breaks from episode 2; M9 the bank must store the post-`spawn_match` tick-0 state (digest-verified), never re-derive; M10 three serial laws (bites, the vent ledger, sprays after shooting) need per-env serial stages. Minors: draw-site list corrections (charges never draw; exposure / crit / melee / beam cone do), per-(slot, site) salts, water dormancy as a per-env mask, per-env failure flags instead of throws, float32 schema dtypes for the permeability / absorb stamps, Q16 rounding spelled out, two host whole-plane rebuilds per tick, gate additions | B1 → §4.3 the per-env serial applier with the topology seam and payloads as device functions, §5.7 gates; B2 → §4.5 the event ring, §5.9 the digest redefinition; M1/M2 → §4.2 fold table, both loops masked, the counted interim D2H; M3 → §4.2 per-env books; M4 → §4.8 composite key, applier-serial, integer FieldEdit (§7 P3e); M5 → §4.3 the applier's payload write set; M6 → §7 P3e; M7 → §4.2 streams, double buffers, capture test; M8 → §4.1 the per-episode seed function; M9 → §4.6 tick-0 state bank; M10 → §4.3 serial stages; minors → §4.7, §4.2, §4.4, §5.2 |
| systems reuse / scope: does §13 miss a canonical system; is anything a parallel copy; what belongs to another arc | **NOT SOUND as v1** (critique `214deed`, `docs/residency_batching_critique_2_systems_scope_2026-11-07.md`): the route holds; three things change on paper before any patch: the device form of the topology seam, an oracle the device can compute, the patch plan | **B1** §4.3/§11: topology changes on the device backend go AROUND the booked seam (`destroy_wall`'s mass-books bracket, reservoir evacuation, `gas_energy_retire`, the ambient seed, the `on_tile_changed` diff; `seal_tiles` / `unseal_tiles` book solid energy; per-tick callers; slot 9 unassigned). **B2** §5.3/§5.7: the device cannot compute `tick_digest`; units are not in the entity section. **Majors** M1 ~20 canonical systems missing from §13(a); M2 a §13(c) amendments list; M3 the obs crop as an accessor method, the GLOBAL map dropped; M4 light pre-empts #81; M5 `[rl.*]` in config.toml ties experiments to snapshots; M6 the farm worker is a second loop, the policy not a controller; M7 device episodes must export as Recordings; M8 the action buffer must derive from INTENT_KINDS (fuse float, clear vs zero, latch semantics); M9 Q16 positions not forced, consumers uninventoried; M10 the bank re-invents the §5.3 bank; M11 the rare-event path's write inventory; M12 the schema must collapse four definitions; M13–M16 patch splits + the bridge; M17 bots as a ruling; M18 the container through `ml_stack.py`; M19 one describe helper | B1 → §4.3 applier + §5.7; B2 → §4.5 ring + §5.9; M1 → §13(a); M2 → §13(c); M3/M4 → §4.5; M5 → §4.1 the run file; M6 → §4.1 `MatchSession` + `policy`; M7 → §4.5 the episode record; M8 → §4.5 actions; M9 → §7 P3c + §12.1; M10 → §4.6; M11 → §4.3; M12 → §4.4; M13–M16 → §9; M17 → §12.3; M18 → §8; M19 → §6 |
| throughput honesty: the numbers, the yardstick, the stop condition, what P9 can and cannot buy | **SOUND WITH FIXES** (critique `94c2d0a`, `docs/residency_batching_critique_3_throughput_2026-11-07.md`): the table arithmetic re-derives | **B1** the stop condition read on the wrong slot with no anchor in training need; **B2** read after the cost it should avoid. **Majors** M1 the 196 is physics-only (game-vs-game ~155); M2 the A100 range irreproducible (14 000 dropped the allowance); M3 the route table underived, the roofline missing; M4 "pays only after the flip AND kernel work" not established; M5 no farm measurement protocol; M6 the 8 GB card with rollout buffers | B1/B2 → §3 the three readings G1 / G2 / G3 per Berzelius slot, P10 early, P3 behind G1, Erik rules the scale (§12.2); M1–M3 → §2.3 corrected, §2.4 the anchor, §3 derived; M4 → §2.3 movers; M5 → §9.1; M6 → §4.10 |
| game-dev continuity: snapshots, twin debt, the rules of §10, what a gameplay change costs after the port | **SOUND WITH FIXES** (critique `3a1d1d5`, `docs/residency_batching_critique_4_continuity_2026-11-07.md`) | **B1** twin debt cannot be detected where gameplay is written (CUDA gates skip without a build; the corpus never exercises what changes next; stored corpora stale; #71 the precedent); **B2** device code not split into what the played game runs and what only training runs. **Majors** M1 P1–P3 merge to main as they pass, no long arc branch; M2 provenance holes; M3 the training-episode record and the integer fuse; M4 behaviour costs ignored, the device table upload; M5 three CPU-side rules; M6 the schema reuses the entity kinds, absence-transparent, `__slots__`; M7 `x_q` / `y_q` + float properties; M8 the training checkout in the git workflow | B1 → §6 the computed ledger, refusing doors, live lockstep, coverage counters; B2 → §6 Class P / Class T, §12.6; M1 → §9; M2 → §6; M3 → §4.5; M4 → §4.11 + §10; M5 → §10; M6 → §4.4; M7 → §4.4 + §7; M8 → §6 |

**v2 re-check** (critique `3a2f821`, `docs/residency_batching_critique_5_v2_recheck_2026-11-07.md`):
**SOUND WITH FIXES** — the applier, the ring and the P / T split are the
right shapes; nothing forces another route. B1 Class P had nothing
enforcing it → §6 the manifest lists Class P, the `.cuda_green` stamp, the
never-skipping `test_class_p_level`, the merge-step refusal. B2 P4c could
not pass before Philox and integer FieldEdit → §4.7 FieldEdit noise is
Philox on every path (PCG64 never ported), §9 P4c after P3b / P3e. M1 the
aggregate moves at P3a and P3e → §5.9 re-baselined. M2 slot 2 and 9e are
serial → §4.3 the per-slot sequence table. M3 the cross-tick key → §4.8 the
enqueue phase. M4 the float inventory → §7 P3e's site table. M5 the P / T
boundary per artefact → §6 the artefact table. M6 path lists are remembered
lists → §6 coverage-generated sets + the scan test. M7 G1's scene → §3.
M8 the oracle needs the entity export → §5.9. M9 the applier's work bound →
§4.3 block form, the 10 % share, the applier-ms row. Minors folded (slot
10's conversion and clears; trigger holds; ExplosionEvent's integer
strength and the digest filter; damage already Q16; five re-pins; 9b's
top-16; 9e serial; the write set; int64 ranges; the closure window;
`n_destruction_seed_sum`; reset clears queues; queue capacities; the
in-step terminal rule; P6 dedupe; P10's harness paths; one subprocess for
lockstep; `recent_attackers` REFERENCE_ONLY; P3 ordering by need; the
manifest enumerates twins without a build; the ring per tick). **The
design is built from v3; no further critique round.**
