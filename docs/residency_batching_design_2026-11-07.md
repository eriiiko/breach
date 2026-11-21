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

#### 4.4b Amendment 2026-11-14: P6 re-planned against the code at `43115fec`

Written by the lead after the unit design session
(`docs/unit_design_2026-11-14.md`) from a slot-by-slot inventory of the tick
as it stands (`docs/residency_p6_slot_inventory_2026-11-14.md`, every
statement with `file:line`). §4.3's table and §9's P6 row are from
2026-11-07; since then the tick gained slots 1a (impulse + turn), 1b'
(orders), 1c (exposure), 4a (melee), 10' (order expiry) and 11 (the team
picture), planted charges in 9e, and the unit design adds the `traits` block
and the poison law (slot 9c3). This amendment replaces the P6 row. The
design's "P6b" (entities on the device) keeps its name and follows.

**Decisions (the lead's; each is the simplest form the inventory allows).**

1. **The store is one array `(N, U, W)` of int64**, row-major: a unit is a
   row of W integers, the schema's group-S columns in schema order, each at
   its encoded width (the `encode_unit` / `decode_into` hooks P3a left for
   this), then `valid`. int64 throughout: the impulse accumulators exceed
   int32, and at U of tens the bytes do not matter (about 2 KB a unit). The
   column offsets reach C++ as a GENERATED, checked-in header
   (`cpp/src/unit_store_layout.h`, written by `tools/gen_unit_layout.py
   --write`, held equal to the schema by a test: the light table's and the
   edit ordinals' precedent). A new schema column is: regenerate, commit.
2. **Row r is the r-th unit of `sim.units`**, which is id (spawn) order: the
   order every reference loop walks. `(team, slot)` are columns, not the row
   order (§4.4's "(team, slot)" is withdrawn: the inventory could not show
   the two orders always agree, and the laws are written in id order). U is
   fixed when the store is built (a match spawns once; a unit is never
   removed); a batch pads to the largest U with `valid = 0`.
3. **The store is not part of the played resident tick.** The unit kernels
   are Class T (training only, §6; ruling §12.6), so `run_resident_batch`'s
   entry and exit sets, the bus gate and the played `--resident` game are
   untouched by P6. The store is owned by its own driver
   (`simulation/unit_store.py`: the layout, `pack`, `unpack`, the device
   buffer on the tick stream), which the gates use now and the `device`
   backend uses from P8b.
4. **Two gates carry the port, both on live matches, both tol 0:**
   - IDENTITY: every tick, `unpack(pack(units))` leaves every group-S
     attribute equal, the unit digest computed from the ROWS equals the
     reference's, and a device round trip returns the rows bit for bit; in a
     batch, env i's rows are env i's alone.
   - SHADOW LOCKSTEP, per slot: the reference plays the whole tick in Python
     as always. Around a ported slot k the harness packs the units (and
     uploads the fields slot k reads), runs the device kernel, lets the host
     run slot k, packs again and compares rows, events and written fields
     with the device's. The device never drives the game, so a divergence
     names its slot and its tick, and the played game cannot be affected.
     A third form, CHAINED (several device slots in a row, unpacked back and
     played on, digests equal to the reference match), arrives when two
     adjacent slots are ported.
   Each gate must see its slot DO something: per slot a coverage counter of
   the rows it changed, and scenes with non-neutral hooks for every slot
   that reads one (the unit design's critique 1.6).
5. **Every slot gets a profiler label** (`tools/tick_stage_profile.py` has
   none for 1a, 1b', 1c, 4a, 9, 9e(d'), the tick-end clears, 10', 11), since
   a twin-manifest row names its slot by label.

**The cut.** Patches after P6.1 touch different kernels over one harness and
may run side by side; each is re-checked against the inventory when it
starts.

| patch | builds | gate |
|---|---|---|
| P6.1 | the store (decisions 1 to 3), the bridge, the generated layout header, the device buffer and its upload / download on the tick stream, the shadow-lockstep harness, the missing profiler labels, and ONE pilot kernel end to end: slot 1c, exposure decay (a per-unit law with one table scalar, no field read, no event) with its Class-T twin row | IDENTITY on the pinned matches at N = 1 and in a batch; SHADOW LOCKSTEP of 1c with its coverage counter > 0; each validated by breaking it |
| P6.2 | the per-unit kernels that emit no synced event: 1a (impulse integration, the turn; the hooks' device accessors), 3 (movement: the footprint samples, the speed table, the walk ban), 1b' and 10' (orders: apply, expire), the tick-end conversion; the device table upload (§4.11) for the tables they read | SHADOW LOCKSTEP per slot, non-neutral hook scenes included |
| P6.3 | the device `apply_packet` (mitigation, the exposure scale and landing, the hit points, the kill) and the EVENT RING (§4.5); on them 2b (statuses), 9c (heat), 9c2 (wind), 9c3 (the gas rows and the poison law), with the tick's credit claims as a per-slot table | SHADOW LOCKSTEP per slot, events compared in emission order |
| P6.4 | the per-env serial applier (block form, §4.3) and the topology seam as device functions; 9 (burn-through), the tick-end clears, the `done` kernel; 1b's per-unit verbs (reload, shove, the latches) -- its appends to the projectile and charge tables go with P7b. (The entity export for the digest moved to P6b, where the entity SoA it exports first exists: the lead's ruling, 2026-11-15.) Cut in two patches run side by side: P6.4a (1b's verbs, the `done` kernel) and P6.4b (the field side) | SHADOW LOCKSTEP; the topology scenes |

**What each later patch must know** (the inventory's Part D and E):

- Order-free as the code stands: 1a, 1b', 1c, 2b, 3, the phase-A decisions
  of 4 / 4a / 4b, 9c, 9c2, 9c3, 10', the conversion. Order-dependent: 1b's
  appends, 2 (grenades serial and immediate), 4c, 5 pass 3, 6's refill, 9 /
  9b, 9e.
- Slot 9b (over-pressure walls) is switched off (§12.7) and gets no kernel.
- Reference debts to settle before the patch that meets them, each a small
  reference change of its own: on the continuous path `sim.projectiles` is
  never pruned (a detonated grenade stays in the list), so the list has no
  bound (before P7b); `melee_intent` does not mark its unit possessed while
  slot 1b consumes latches only for possessed units (before P6.4's 1b);
  a `Projectile` has no spawn ordinal, though §4.3 assumes one (P7b).
  **Settled 2026-11-14 (the reference-debts patch):** the continuous ruleset
  (`ContinuousRealtime.on_tick_end`, slot 10, so `MatchRules` too) drops a
  detonated grenade from `sim.projectiles`, so on the continuous path the list
  holds only grenades in flight (the two WEGO prunes unchanged). Possession
  carries no state: a living marine outside `_possessed_ids` holds nothing
  slot 1b consumes or drops (no held trigger, no throw / plant / use / reload
  / shove latch, no wind-up, no plant under way: every writer of those
  columns possesses its unit, `tests/test_possession_carries_nothing.py`),
  and `melee_intent` now possesses its unit like every other hand verb. So
  P6.4 may port 1b as "every living marine consumes its own columns" with no
  possession column; the drop branch only ever acts on the dead and the
  turned. The spawn ordinal stays open (P7b).
- Host structures without a schema or a capacity that need both before their
  slot is ported: projectiles, bullets in flight, the noise log, the blast
  log, the credit claims, the recorded effects of `SimultaneousEffects`
  (P7a to P7c); the planted charges have both.
- Slot 11 (the team picture) is observation-side and not synced; its device
  form belongs to P8b with the observation.

**As built (P6.1, 2026-11-14).** The store, the bridge, the header, the
buffer, the harness, the labels and the pilot kernel landed as decided; no
decision moved. One line each:
- The layout is `simulation/unit_store.layout()`: 79 columns, W = 180 on the
  shipped config after the poison patch's merge (76 / 174 before it: the
  patch added three columns and three hooks, and the header was regenerated
  at the merge, as the rule says; the widths read config tables: the status
  block 4 x 16, the exposure block 9, the traits block 12, the hand-grenade
  rows 6), then `valid`; a sha256 fingerprint of the layout's text rides in the generated
  header and the driver refuses a compiled module whose fingerprint is not
  the schema's (a header regenerated without a rebuild).
- `pack` / `unpack` are `encode_unit` / `decode_into` and nothing else;
  `unpack` lands a row on its own unit or on a BLANK one (`blank_units`: the
  schema defaults and no constructor, so the group-D attributes stay unset --
  the identity gate proves the rows alone carry group S). The digest read
  from a row (`record_from_row`) evaluates the one `digest_of` column (the
  turn) on a view of the decoded row.
- The device buffer is a C++ object (`UnitStoreDevice`, bound as
  `bp.UnitStoreBuffer`), not a CuPy array: the driver needs no CuPy, frees
  the memory when dropped, and stages through ONE pinned buffer guarded by an
  event (a second upload waits for the first copy to have read it). Its
  download and the kernel's coverage read are host syncs on the tick stream
  -- allowed because nothing in the played tick calls the store.
- The kernel `k_exposure_decay` reads no life state, because `decay_units`
  reads none (the brief said "skip dead rows as `decay_units` does": the code
  does not skip them; a dead unit's block is zeroed at its death, so the two
  agree in play and the kernel follows the code). Its product is the kit's
  `mul128_shr`, exact for every int64 exposure; `decay_q` is refused outside
  [0, 65536].
- The harness hooks the slot's `tick_stage_profile` callables (so a slot is
  named once, by the label a twin row uses), packs at the first entry and at
  every exit, runs the kernel once over the batch after every env ticked, and
  also runs each env alone through an N = 1 store; a slot that did not run in
  an env raises.
- Slot 9's burn-through loop and the tick-end clears were inline code in
  `_step_post_physics`; they became `Simulation._burn_through_walls` and
  `_clear_tick_buffers` (code motion, so the profiler can wrap them). The new
  labels: `1a integrate + turn`, `1b' orders`, `1c exposure decay`,
  `4a melee`, `9 burn-through walls`, `9e(d') planted charges`,
  `tick-end clears`, `10' order expiry`, `11 team picture`. The twin
  manifest's generated files moved for no existing twin.
- The scene for the shadow lockstep needs no test config: the shipped
  `[damage.exposure]` (k_expose 1, a 3-tick half life) raises exposure in
  ship_demo's bots match from tick ~437 (the kernel changes rows on 23 of 700
  ticks) and in the focus-fire bench's arena from the first volley (93 to
  98 % of env-ticks).
- The first Class-T row, `exposure_decay`: its generated files are
  `exposure.py` alone (the label's trace), which therefore left
  `[not_twinned]`; its debt can never block a merge (Class-T files are not
  Class-P paths) and blocks only the device training door.

**Re-check at P6.2's start (2026-11-14, the lead; it amends the cut above).**
"Patches after P6.1 ... may run side by side" does not hold for P6.2 and P6.3
as cut: read against P6.1 as built, both need the same shared layer, and two
agents would build each part twice. The shared layer, built ONCE, in P6.2,
each part with a real user in that patch:

1. **The device tables (§4.11)**: every table value a unit kernel reads, taken
   from the SAME validated objects the reference laws read (the impulse
   table, the speed table, the turning dials, the status registry, the
   species' class values, the environment profiles), through a column schema
   with generated offsets, uploaded when the store is built. Per env where
   the table is quantized per map.
2. **The harness's slot inputs**: `DeviceSlot.run(store)` sees only the rows.
   A slot also reads things that are not unit rows: field planes, the tick,
   a per-tick host table (the order requests). The harness captures them per
   env when the slot is entered and hands them to the kernel. A field reaches
   a kernel as a device pointer to an `(N, h, w)` plane in the resident
   field's own dtype and currency, so the device backend (P8b) can hand it
   `ResidentBatch`'s buffers unchanged.
3. **The device unit library**: the functions several slots share, as one
   `__device__` header over the generated layout: the status block (the
   composed flags, apply, the sweep), the hook accessor (`round(class value x
   hook)`, skipped at ONE), the life gate.
4. **Where a kernel lives**: `UnitStoreDevice` stays the buffer and its
   copies. A slot's kernel is a free function in its slot family's own TU
   over the store's pointer; the store class gains no method per slot.
   Kit-level integer helpers that `unit_fixed.py` holds (`mul_q16_round`,
   `div_round`) get their twin in `fixed_point.h` beside the three already
   there, held equal by `tests/test_unit_impulse_twins.py`; a law (the turn,
   the integration) is never kit code.

The cut from here:

| patch | builds | runs |
|---|---|---|
| P6.2 | the shared layer (1 to 4) and its first users: the unit impulse law's two slots, 1a (integrate + turn) and 9c2 (the wind's impulse; it emits no event, so it moves here from P6.3), 1b' and 10' (orders), the tick-end conversion; the CHAINED gate on the two adjacent pairs it creates (1b' + 1c, 10' + conversion) | first, alone among the kernel patches |
| P6.2m | slot 3, movement (the footprint samples, the speed table, the direction rule, the walk ban, the slide rule) | after P6.2, beside P6.3 |
| P6.3 | as above, without 9c2: the device `apply_packet`, the event ring, 2b, 9c, 9c3 (teargas, the poison law) | after P6.2, beside P6.2m |
| P6.4 | as above | after P6.3 |

The two reference debts (the projectile list never pruned; `melee_intent`'s
possession) and the stale docstrings of the inventory's Part E are one small
reference patch run beside P6.2 (`29-ref-debts`).

**As built (P6.2, 2026-11-14).** The shared layer and its five first users
landed as re-checked; no decision moved. One line each:
- THE DEVICE TABLES are `simulation/unit_tables.py`: `ROWS`, each a
  `TableRow(name, width, value(src), lo, hi)` whose `value` reads the
  reference's own object (`sim.impulse_table`, the sim's turning class dials,
  the status registry, `exposure.get_settings()`, `unit_traits.class_hp_q`,
  the species / env-profile stabilities, `ZOMBIE_TEAM`, the gas indices) and
  whose `lo` .. `hi` is the range the kernels were proved for (refused by
  name at `values`); 25 rows, T = 45 on the shipped config. One vector per env
  (`(N, T)`; the impulse table is per map); the generated
  `cpp/src/unit_tables_layout.h` (written by the same `gen_unit_layout.py`
  as the store's) carries the offsets, each hook's index and each status
  kind's id, all in a fingerprint the store checks. The tables are uploaded
  when the store is built (`DeviceStore(..., tables=)`), a separate
  `UnitTablesDevice`.
- THE SLOT INPUTS: `DeviceSlot(label, run, columns, inputs)`; an input is a
  row of `cuda_unit_lockstep.INPUTS` (`tick`, `order_requests`) or
  `field:<GameMap attribute>`, captured per env at the slot's FIRST entry in
  a tick and gathered into a `SlotInputs` (`.tick (N,)`, `.tables[name]
  (N, R, C)`, `.fields[name]` a CuPy `(N, ...)` array in the resident
  field's dtype); `run(store)` without inputs (P6.1's form) or `run(store,
  inputs)`. A field reaches a kernel as a device pointer (the binding takes
  `uintptr`s and the shape), so P8b passes `ResidentBatch.buffers` as they are.
  The per-tick host table is staged through the store's `stage_input` slots.
  `ShadowLockstep` takes one slot or a list (one match shadows all six).
- THE DEVICE UNIT LIBRARY is `cpp/src/cuda_unit_lib.cuh`; its gate is a
  TEST-ONLY probe TU (`cuda_unit_lib_probe.cu`) running each function on
  synthetic rows.
- WHERE A KERNEL LIVES: `UnitStoreDevice` keeps the buffer, the copies, the
  per-env COUNTS (coverage / faults / refused, a 3 x N device array every
  kernel writes) and the input staging; P6.1's `exposure_decay` method left
  it. Each slot is a free function in its family's TU (`cuda_unit_exposure.cu`
  1c, `cuda_unit_impulse.cu` 1a + 9c2, `cuda_unit_orders.cu` 1b' + 10',
  `cuda_unit_conversion.cu` the conversion), declared in `cuda_unit_kernels.h`,
  bound one `bp.unit_*` function each returning the three counts, and faced
  in Python by `simulation/unit_kernels.py` (which refuses a fault). The kit
  gained `isqrt_q` / `isqrt_u64_q` (exact over int64 / uint64: `sqrt_q16`
  clamps at int32), `isqrt_ceil_q`, `div_round_q`, `mul_q16_round_q` (a
  128-bit product through `umul64_wide`); every `.cu` is on the target-wide
  CUDA determinism floor (confirmed in CMakeLists: `--fmad=false`,
  `-Xcompiler /fp:strict`), no per-source list needed.
- WIDTHS: Python's unbounded ints meet int64 in three places. The wind load
  `W = N * v2` reaches 2^80 and is formed in 128 bits, its impulse
  `((W >> 16) * k) >> 16` too (a result past int64 FAULTS); the projection
  `j * v / n` is the exact split `j = q n + r` (`|v| <= n`, so `r v < 2^63`).
  The demand's components are clamped at 2^30 before their square, exactly
  (past it the effort saturates whatever the rest). Every other square is of
  components the tables' doors and the V_MAX clamp bound; a state operand past
  2^31 in a squared 2-vector (2^15 tiles a tick) faults the row instead of
  wrapping. No fault occurred in any scene.
- THE CONVERSION's slot is the label `10 tick end (ruleset)`
  (`ContinuousRealtime.on_tick_end`: `convert_marines_to_zombies`, then --
  since the reference-debts patch merged mid-P6.2 -- dropping the detonated
  grenades from `sim.projectiles`), so no nested label was added. Its
  host-only side (the name's "Z-" prefix, the momentum books, the grenade
  prune: a host table until P7b) has no device form; the chained harness does
  it after the unpack.
- THE GATES (`tests/cuda_unit_kernels_check.py`, wrapper
  `tests/test_cuda_unit_kernels.py`): the tables read back equal the
  reference objects over two maps of different tile size; the library on
  synthetic rows; the six slots shadowed together on ship_demo's bots match
  with TURN / HOLD / HP hooks on two thirds of the marines and an order
  schedule (500 ticks, twin equal every tick); slots 1a + 9c2 on the doorway
  jet scene (knockdowns, prone bodies in wind) at N = 1, 2, 8; 1b' / 10' / 1c
  on ship batches with extra idle units, the conversion / 10' / 1a on box
  matches where a horde bites idle marines, N = 2, 8; the chained pairs 1b' +
  1c and 10' + conversion. Every gate was broken once (the patch's report).
- The corpus traces no conversion (no zombie kills a marine in 300 ticks), so
  the conversion twin's generated files are `ai_zombie.py`, `ruleset.py` and
  `unit.py` only; `integrate_turn` and `wind_impulse` share `unit.py` /
  `gamemap.py` with Class-P twins (a change there needs the stamp through
  those). Eleven modules left `[not_twinned]`.

**As built (P6.2m, 2026-11-14).** Slot 3 landed as cut, beside P6.3; no
decision moved. One line each:
- THE CONTINUOUS PATH ONLY: the kernel is `Simulation._update_player_movement`
  as the continuous ruleset runs it (`cpp/src/cuda_unit_movement.cu`,
  `unit_kernels.movement`). OnePhaseWEGO's slot 3 (`_displace_pushed` + its
  timeline) and TwoPhaseWEGO's path branch are legacy and not ported; the WEGO
  path step after the loop is a no-op on the continuous path (`move_path` is
  empty) bar its `path_tick_offset` bump, which is REFERENCE_ONLY.
- THE FIELD INPUT is `field:material`, the GameMap's own int8 plane (already a
  resident entry field, so P6.4's topology seam and P8b feed it unchanged),
  projected through a new table row `material_mobility` -- the materials
  table's mobility column, the projection `is_passable_block` and
  `footprint_mobility` both make. No derived plane: both of slot 3's readers project
  `material`, and the GameMap holds no mobility plane.
- THE TABLE ROWS (one block, appended): `speed_walk_cost_q`, `speed_full_cost_q`
  (the WALK / SPRINT hooks' class values, from `sim.speed_table`),
  `speed_side_q`, `speed_back_q`, `impulse_effort_edge_q`, `material_mobility`
  (per map: two maps of different tile size carry different costs); T = 61.
- THE LIBRARY gained the walk (`tile_of`, `walk_allowed`, `effort_scale_q16`,
  `MobilityMap`, `passable_block`, `footprint_mobility`, `default_speed_q`,
  `move_speed_q16`): the zombie walk and the legacy walkers read the same
  Python functions. The displacement (`unit_impulse.move`) stays in the slot
  TU (slot 3 is the one displacement loop). The kit gained `half_up_q`
  (`movement.half_up`, floor divisions; `bp.fp_half_up`).
- WIDTHS: a footprint of at most 8 x 8, a Q16 cost of at most 2^29 after each
  reduction step and a hook operand of at most 2^20 keep the speed's numerator
  below 2^62; a move component or a walk speed past 2^31, a pushed velocity past
  2^40 or more than 64 substeps FAULTS the row. A material id outside the table
  faults (numpy would wrap a negative id); a footprint block whose clamped stop
  is negative faults (numpy would count from the end). No fault in any scene.
- THE HOST-ONLY SIDE in the chained form: the momentum books' WALL entries
  (the pv `move` zeroes, REFERENCE_ONLY) and the `path_tick_offset` bump.
- THE GATES (`tests/cuda_unit_movement_check.py`, wrapper
  `tests/test_cuda_unit_movement.py`, kept apart from P6.2's script so P6.3
  could append beside it): LIB (the walk's functions on synthetic rows over
  the proved range, a TEST-ONLY probe `cuda_unit_movement_probe.cu`); ship_demo's
  bots match with WALK / SPRINT / TURN hooks; a movement box built for the gate
  (walls, furniture, scripted pushes from the map's own impulse table,
  statuses, an idle horde, a dead unit) where a census counts every branch;
  batches N = 2 / 8 whose envs differ in map, tile size and tables; PADDING
  (the scenes' padding rows are all zeros, which no branch would move, so a
  part fills them with live rows); the doorway jet; CHAINED slot 3 on the ship
  and the box. Every gate broken once (the patch's report).

**As built (P6.3, 2026-11-14).** The device `apply_packet`, the event ring
and slots 2b, 9c, 9c3 landed as cut; no decision moved. One line each:
- THE DAMAGE PIPELINE is `cpp/src/cuda_unit_damage.cuh`, beside the unit
  library: `apply_packet(row, tab, packet, sink, changed, &killed, &bad)`
  returns the applied Q16 damage and is `damage.apply_packet` integer for
  integer (mitigation through `mul_q16_round_q`; the exposure scale in 128
  bits; the hp; the kill on alive -> dead only; the death clears of
  `exposure.on_death`, `unit_impulse.clear`, `horde.clear`,
  `unit_orders.clear`, `poison.clear` and the conversion flag when the packet
  marks it; the exposure raise at once -- these slots own no attack slot).
  It takes ANY row and any sink, so P6.4 / P7's serial applier hands it the
  target's row and the env's ring. A packet or row operand past its proved
  range (|amount| 2^46, exposure 2^40, expose / AP 2^40) writes nothing and
  faults the row. `recent_attackers` (REFERENCE_ONLY, WEGO's idle stance) has
  no device form. Its gate beyond the scenes: a TEST-ONLY probe kernel runs it
  on synthetic rows, packets and tables (non-dyadic resists, armor and AP,
  corpses, the conversion flag, heals, every cause) against the Python.
- THE EVENT RING (§4.5) is `simulation/unit_events.py` (the one field table:
  `kind, unit_id, damage_q, cause, attacker_id, dir_x_q, dir_y_q` -- the
  digest's `_event_record` plus the render-only direction, so a record is
  the host event bit for bit) with the generated `cpp/src/unit_events_layout.h`
  (which also carries the kinds, causes, damage types and the claims' channel
  codes) and `cpp/src/cuda_unit_events.{h,cu}` (`UnitEventRing`). ORDER: a
  per-row kernel writes each row's events into that row's staging (at most
  `row_cap()` = the status block + 1, 2b's worst case) and one thread per env
  then appends the staged rows in row order -- the reference's emission order
  (id order per slot; the law's order within a unit) whatever the thread
  schedule. CAPACITY `E_max` (default 1024 per env per tick); past it, or past
  a row's staging, nothing is written, the env's dropped count rises and the
  Python face refuses the run (`UnitEventOverflow`). ExplosionEvent is not in
  the ring yet: it joins with its producer (P7c).
- THE HARNESS: `DeviceSlot(..., events=True)` hands the kernel a ring the
  harness begins and compares it, record for record, with the events the host
  appended to `sim.tick_events` from the slot's first entry to its last exit
  (the first difference as tick, env, event index, field; the alone run's ring
  must equal the batch's env i); `ChainedSlots` appends the ring's records to
  `sim.tick_events` as host events where the first slot would have, and the
  chain now runs at the first CALLABLE of the first slot (a label may name
  several). The credit claims are the table input `credit_claims`
  (`unit_kernels.claims_of`: `(channel, y, x, attacker)`, the heat buffer -1,
  a gas slice its index; every key once, as the dict holds it).
- THE SLOTS: 2b is `cuda_unit_status.cu` (the sweep, each DoT status's packet
  while the unit lives, the countdown -- every status counts down even after
  a death mid-list, as the Python does); 9c and 9c3 are `cuda_unit_exchange.cu`.
  9c3's label names three callables (`apply_teargas_blind`,
  `apply_poison_dose`, `poison.tick`); each unit's three steps read only the
  planes and its own row, so ONE kernel runs them per row in that order (the
  doses clipped at the body's store are counted in `refused`). The rows'
  all-zero-plane early-outs need no device form: the thresholds are table
  rows refused below one count. The device runs the gas rows whether or not
  the sim has a physics runner (the host gates the teargas and dose rows on
  one; every scene has one). A kernel works on a LOCAL copy of its row and
  writes it back whole or not at all (`finish_row`).
- THE TABLES gained 33 rows (T 150): mitigation (zombie and per species),
  exposure (k, cap, resists), the status registry's DoT type and cause, the
  heat row's dials and per-profile / per-species band and rate, the tick rate
  and the sweep's fine bits (per map), the gas planes' indices, thresholds and
  the blind's ticks, the poison law's class values, `X_MAX`, `FLUSH_Q`, and
  `footprint_rowmajor` (refuses a species whose default footprint is not the
  row-major square the kernels walk).
- THE GATES (`tests/cuda_unit_damage_check.py`, wrapper
  `tests/test_cuda_unit_damage.py`): part APPLY; scenes ship (ship_demo bots,
  seed 1, 500 ticks), fire, gas, status (four-lane levels with fixtures
  written between ticks: units about to die carry an order, exposure, an
  impulse, poison and a status; DoT statuses through `status.apply_status`,
  since no shipped trigger applies one), each with hooks and an unhooked-twin
  check; batches N = 2 and 8 with padding; the overflow; chained 2b alone and
  9c + 9c2 + 9c3. Every gate was broken once (the patch's report).

**As built (P6.4a, 2026-11-15).** P6.4 was cut in two patches run side by
side: P6.4a the UNIT side (slot 1b's per-unit verbs, the `done` kernel), P6.4b
the serial applier, the topology seam, slot 9 and the tick-end clears. P6.4a
landed as cut; no decision moved. One line each:
- SLOT 1b is `cpp/src/cuda_unit_verbs.cu` (`unit_kernels.verbs`), the
  continuous path's `_consume_direct_intents` integer for integer: the drop
  branch (a dead or turned row releases a held trigger, a wind-up, a plant),
  then per living marine throw (latch, `can_throw`, the HANDLING wind-up, a
  0-tick wind-up releasing at once; the advance: dropped when the unit cannot
  act, released along the facing on its tick), plant (latch, `can_plant`, the
  advance, the charge on the last tick), use, reload (`can_reload`,
  `mag_settle`, `reload_begin`), shove (`can_shove`, `shove_target`, the PUSH
  impulse). NO POSSESSION COLUMN (§4.4b "Settled 2026-11-14"): every scene's
  rows are equal with the host still gating on `_possessed_ids`; no scene
  showed a difference a possession column would remove.
- THE GATES' DEVICE FORMS live in the unit library (`cuda_unit_lib.cuh`'s P6.4a
  block, so P8b's mask kernel calls them): `acts`, `can_act`, `handling_ticks`,
  `can_throw`, `can_plant`, `weapon_row`, `magazines_carried`, `mag_status`,
  `can_reload`, `shove_cooled`, `can_shove`. `magazines_carried` compares the
  encoded weapon indices: the schema decodes "no row" to `""`, never `None`
  (`Unit.__init__` writes `""` too), and `""` is not `None` in the Python's
  test, so a weapon in the hands with no loadout weapon carries 0 magazines.
- THE REQUEST LIST (§4.3's "a throw appends a projectile; USE toggles
  `want_open`"): what slot 1b does that is not a row write is a REQUEST record
  of `simulation/unit_requests.py`'s one field table (`kind, unit_id, tick,
  tile_x, tile_y, target_x_q, target_y_q, ticks, item, approach_y,
  approach_x, door, accepted`; kinds throw / plant / use; the generated
  `cpp/src/unit_requests_layout.h`, fingerprint-checked), every value the
  reference computes at that point (a throw: its start tile, Q16 target, the
  clamped fuse, thrown tick, kind, owner; a plant: its tile, ammo row, owner,
  `det_tick`, approach, accepted; a use: the door's ordinal in `sim._doors`,
  the door spans a per-tick table input `door_spans`). The list is the event
  ring's own class GENERALISED (`UnitRecordList`: any record width and row cap,
  the one staging and the one gather; `UnitEventRing` is now an instance), so
  per-row staging gathered per env in row (= id) order is the reference's
  order (slot 2 detonates in it; `charges.place` counts in it). Capacity
  `R_MAX` 256 per env per tick, row cap 3 (one of each kind); past it counted
  and refused by name (`UnitRequestOverflow`).
- THE ORDER-DEPENDENT DECISION (which plant a full table refuses) is ONE thread
  per env walking the gathered list against the live-charge count (the scalar
  input `charges_live`): an accepted plant spends its planter's `has_explosive`,
  a refused one is counted in `refused` (== the host's `charges_refused`).
- THE SHOVE writes another row: each shover stages (target row, jx, jy,
  angular) in a per-row scratch, a second kernel lets each target sum the
  pushes aimed at it in shover-row order (`unit_impulse.add`: integer sums of
  per-input terms, the lever clamped per input) and counts each row once.
  The angular term squares an impulse of up to 2^35 (ship_demo's 1/3 m tiles
  put the shove at 2^29.6, times the PUSH hook): the kit gained
  `isqrt_u128_q` (`bp.fp_isqrt_u128`, held equal to `unit_fixed.isqrt` by
  `tests/test_unit_impulse_twins.py`).
- THE DONE KERNEL is `cpp/src/cuda_unit_done.cu` (`unit_kernels.match_done`):
  per env (terminal, outcome none / win / draw, winner team, living marine
  teams) from the rows, the tick after the step and the table row
  `match_limit_ticks` (`MatchRules.limit_ticks`; 0 = no match). It is not a
  slot: its gate compares it at the harness's tick end with the session's
  `MatchRules`; its twin row names slot "10 tick end (ruleset)" (§4.3 places
  it there; the traced files carry `ruleset.py`). No profiler label was added.
- THE TABLES gained 13 rows (T 215): `weapon_mag_size`, `weapon_reload_ticks`
  (per weapons-table row, the schema's order, refused if the sim's table
  differs), `hand_windup_class_ticks`, `hand_max_reach_q`,
  `hand_fuse_lo_ticks` / `hi`, `plant_class_ticks`, `charge_capacity`,
  `shove_q`, `shove_cone_q`, `shove_cooldown_ticks`, `impulse_r_body_q`,
  `match_limit_ticks`.
- THE HARNESS: `DeviceSlot(..., requests=True)` hands the kernel a begun
  `RequestList` and records, while the host slot runs and in the host's order,
  each grenade `_release_grenade` appended, each charge offered to
  `charges.place` with its answer, each door `_direct_use` flipped (read through
  `verb_gates.use_door`); compared record for record, the alone run's list
  equal to the batch's env i. A new input kind "scalar" (`SlotInputs.scalars`).
  `ChainedSlots` turns the list into the host's objects
  (`unit_requests.apply_to_host`: the projectiles appended, each plant offered
  to `charges.place`, which must agree with the record, the doors flipped).
  The render-only `ThrowEvent` / `PlantEvent` / `ReloadEvent` / `ShoveEvent`
  have no device form (outside every digest); the chained 1b alone keeps the
  REFERENCE_ONLY momentum books' IN entries, the four-slot chain does not.
- THE GATES (`tests/cuda_unit_verbs_check.py`, wrapper
  `tests/test_cuda_unit_verbs.py`): TABLES; LIB (the gates on synthetic rows,
  both answers of each); scenes ship (ship_demo bots, seed 1, 500 ticks,
  HANDLING / PUSH hooks), verbs (a three-room level, an interior, a hull and a
  dead door, a scripted schedule; a census of the host's own decisions counts
  51 branches, any at 0 fails, and two planters at a full table on one
  tick) and zero (the class wind-up and plant patched to
  0 on the sim's rows); batches N = 2 / 8 (verb variants differing in layout,
  seed, unit count, tick offset and charge table, beside ship_demo); PADDING
  (copies of rows holding latches); OVERFLOW; CHAINED 1b alone and 1a + 1b +
  1b' + 1c on ship and verbs; DONE (elimination, a limit draw, a fallen team,
  a batch ending on five different ticks). Every gate broken once (the
  patch's report).

**As built (P6.4b, 2026-11-15).** P6.4 was cut in two patches run side by
side: P6.4a (slot 1b's verbs, the `done` kernel) and P6.4b, the FIELD side:
the per-env serial applier, the topology seam as device functions, slot 9
and the tick-end clears. No decision moved. One line each:
- THE APPLIER is `simulation/applier.py`, ONE table with generated offsets
  (`cpp/src/applier_layout.h`, the fourth header `gen_unit_layout.py` writes,
  fingerprint-checked by `DeviceApplier`): an effect RECORD `(kind, y, x,
  arg)`, its kinds (DESTROY_WALL today), the per-env BOOK channels (each naming
  the host quantity it twins: `n_destruction_seed_sum`, the gas energy seam's
  `retire` / `mint` / `destroy_seed`, the solid books' `destroy_wall`,
  `_tile_serial` as `tiles_changed`; plus `applied`, `dropped`, `faults`), the
  27 PLANES the topology seam touches (the launch's pointer table, each in the
  GameMap field's own dtype) and the 4 CLEAR_PLANES. `cpp/src/cuda_applier.cu`:
  `FieldApplier` (`bp.FieldApplierBuffer`: the list `(N, cap, F)` with each
  env's REQUESTED count, the books `(N, B)`, pinned staging), `k_applier` on a
  `(1, N)` grid of 32-thread blocks with thread 0 the walk. Past `cap` the
  first `cap` effects are applied and the rest counted; the Python face
  (`unit_kernels.apply_effects` / `burn_through`) refuses the run by name
  (`EffectOverflow`, `ApplierFault`). THE FAN-OUT of an effect's tile set
  over the block's threads is NOT built: it arrives with its first user (P7c's
  deposits), the block shape kept so it lands without moving the kernel. No
  device-side append either (the producers are P6b / P7): the slot-9 list is
  staged from the host input.
- THE TOPOLOGY SEAM is `cpp/src/cuda_topology.cuh` (namespace `topo`):
  `env_view` (one env's typed pointers, rebased as `bptr` does),
  `accountable`, `refresh_cell`, `solid_books_cell` (the engine's
  `conduction::cell_capacity_q`, already `FP_HD`), `patch_face_shift`,
  `on_tile_changed`, `destroy_wall`. Every operand is read before the first
  write, so an effect is all or nothing; a fault (a donor's negative gas, the
  composition's product past 64 bits, a solid book past int64 -- Python's ints
  would hold them) writes nothing. The gas energy seam's structural rows went
  into `gas_energy.h` (`mirror_sat_q`, `retire`, `born_at_ambient`) and the
  device edit queue's mirror refresh now calls `mirror_sat_q` (its private
  copy deleted) -- so the patch touched a Class-P file (cuda_green at HEAD).
- THE TABLE ROWS: 25 (T 564 with P6.4a's, after the merge): `topo_mat_air` / `_hull`, the boundary, the
  ambient seed's three values, `T_AMB`, and per material every value
  `on_tile_changed` patches (the float planes' values as float32 bit patterns,
  the face table M x M, `no_face`, the solid predicate on the float32
  permeability, `blocks_sight`), read from the env's own `GameMap.materials`.
- NOT RESIDENT TODAY (P8b must hold them): `sight_screen`, `heat_atten`,
  `light_atten` (the seam writes them) and `rad_net`, `rad_amb`, `rad_flux`
  (the clears zero them; nothing has written them since T6).
- DROPPED ON THE DEVICE (host-only, no device form): `render_dirty`,
  `_sky_mask_dirty`, `_tile_serial` itself (booked as `tiles_changed`), the
  solid books' sink into the solver's `e_solid_seam_sum` (the chained harness
  books through `_solid_energy_book`), the render-only Wall / Door
  DestroyedEvent, the lazily filled `_gas_t_amb_raw_cache`; the clears' credit
  claims (their device form is the claims table, P7) and the shot tracers'
  expiry (render).
- THE HARNESS: a slot declares FIELD OUTPUTS (`outputs=("field:..",
  "book:..")`, `host_only=((attr, reason), ...)`); the output fields are
  captured at the entry like inputs and written by the kernel in place; after
  the host slot every declared plane is compared bit for bit (floats by their
  bits) naming (tick, env, field, tile), every book against the host's delta
  over the slot, the alone run against the batch's env. THE WRITE-SET GATE
  diffs every GameMap array, scalar, dict key, set, list and object probe
  (`damage_credit`) across the host slot and fails on a change that is neither
  an output nor host-only; arrays that are the same object (`obstacles` IS
  `solid` until the first stamp) count as one. An input read from the slot
  callable's ARGUMENTS is an `ARG_INPUTS` row (`burn_walls`: slot 7's list, in
  its order). `ChainedSlots` writes the outputs back and hands `books` /
  `inputs` to the host side.
- WHAT THE REFERENCE SHOWED (nothing changed in it): `destroy_wall` never
  takes `on_tile_changed`'s membership RETIRE (air is never less accountable
  than what it replaced), so a TEST-ONLY probe (`cuda_topology_probe.cu`:
  `material = m; on_tile_changed`) drives both directions; the donor
  predicate's `not is_vacuum` can never bite (a non-solid vacuum or ambient
  neighbour makes the tile a breach first), so only `not solid` shows, and
  only on a solid tile holding gas (a seam fixture); the explicit `T = 0` is
  load-bearing only on a breach of a hot GAS tile (elsewhere the membership
  diff or the seed's refresh already wrote 0); no shipped door is flammable,
  so the fire scene appends its doors to slot 7's list.
- `seal_tiles` / `unseal_tiles` are NOT built: left to P6b with their only
  callers, the doors (their refusals -- SealBlocked, the int32 pre-check --
  need a device refusal path, and their write set reaches non-topology planes:
  `wave_p`, the wind and flow planes, the float `ripple` pair, the trace
  saturation's books).
- THE ENTITY EXPORT for the digest moved to P6b (the lead's ruling): its
  entity SoA first exists there.
- TWO CLASS-T TWINS, `burn_through` (slot 9) and `tick_end_clears`. The corpus
  burns no wall (ship_demo: 0 burn-throughs in 500 ticks), so
  `burn_through`'s traced files are `simulation.py` alone: a change to
  `gamemap.py`'s seam does NOT put it in debt (the P6.2 conversion twin's
  thin-trace gap again); the topology gate is the guard until a corpus burns.
- THE GATES (`tests/cuda_unit_topology_check.py`, wrapper
  `tests/test_cuda_unit_topology.py`; `tests/test_applier_layout.py` CPU):
  TABLES; SEAM (three fixture maps -- space 1/3 m, planetside 1/2 m, space
  1 m -- each list forward and reversed, the two orders required to differ;
  every branch counted non-zero; the energy, mass and solid books closing on
  the DEVICE's planes as on the host's; the probe); SCENES (two fire scenes,
  ship_demo's bots match: 0 burn-throughs in 500 ticks, the clears acting on
  one tick); BATCH N = 2 / 8 (layout, tile size, boundary, seed, schedule
  differ; an env with no event beside one with events on most ticks);
  PADDING (empty envs, the clears' mask); OVERFLOW; CHAINED 9 (space,
  planetside) and the clears, digests + every topology plane + the books
  equal every tick. Every gate broken once (the patch's report).

#### 4.4c Amendment 2026-11-15: P6b and P7 re-planned against the code at `ca2c85a9`

Written by the lead after P6 merged, from a second inventory
(`docs/residency_p7_inventory_2026-11-15.md`: the host state the remaining
slots own, the laws they share, the device side as it stands, a sharing
matrix; every statement with `file:line`). Version 2, after one critique
(`docs/residency_p7_replan_critique_2026-11-15.md`: SOUND WITH FIXES, nine
majors and fifteen minors, all folded in below). §9's rows P6b and P7a to P7d
are from 2026-11-07 and one line each. The inventory shows that none of them
is one patch and that they are not independent: `execute_payload` is called
from four places (grenades, payload rounds at the resolve, charges, a heat
source's death); the recorded effects of `SimultaneousEffects` are shared by
five slots; shooting marches its rounds on the tick they are fired, so it
shares the march and the table of rounds with slot 2; line of sight is read by
sprays, the horde, the motion sensors and every flash; the noise log is
written by three slots and drained by a fourth. This amendment replaces those
five rows.

**Scope: the continuous path only.** The device twins port what a continuous
match runs (WEGO is legacy, ruling 2026-10-31). Not ported, each named so the
device door can refuse it by name: slot 4's targeted-fire and auto-fire
branches (the `orders` branches, with their float range test),
`start_spray_burst` (the WEGO form), the FACE effect (its only producers are
those branches), physical cover and its chew (OnePhaseWEGO), the order path's
melee crit, and slot 9b (switched off, §12.7). The device door refuses a run
whose ruleset is not continuous, a unit holding WEGO `orders`, and
`burst_enabled`: the ruleset check alone proves neither of the last two.

**Decisions (the lead's; each the simplest form the inventory allows).**

1. **ONE effect list, ONE walker.** The device form of `SimultaneousEffects`
   is the per-env serial applier's own input, widened: a per-env record list
   built on `UnitRecordList` (per-row producers stage per row and ONE gather
   per env appends in row order, which is the reference's record order:
   shooters by id x rounds with each round's recoil before its own records,
   then melee by id, then sprayers by id; a serial producer appends
   directly), walked by the one applier kernel. The applier's reach grows from
   the 27 planes to the unit store, the device tables, the event ring (through
   `env_sink`, which has no caller today), the claims and the logs of decision
   4. The record's fields and kinds stay ONE Python table with a generated
   header, owned by P7a.1. Kinds: HIT, STATUS, PUSH, CHEW and PAYLOAD are
   defined by P7a.1 (PAYLOAD as a record only: the applier refuses to APPLY it
   by name until P7c.1 gives it its function, so P7a.2 and P7b can both record
   it); SPRAY_CELLS arrives with P7a.3 and carries a SPAN into a per-env cell
   buffer that phase A fills against the walls standing at the slot's start
   (its own capacity, counted and refused by name); the walk never recomputes
   a cone, because an earlier CHEW in the same resolve may have removed a
   wall. Never a second list and never a second walker: slot 2's landing and
   slot 4c are stages that walk the record; slot 9e and slot 5's pass 3 are
   stages of the SAME kernel that walk the entity tables and the zombie rows
   in the reference's order and apply AT ONCE, as the reference does (a bash
   changes the map the next zombie reads; a door's sweep changes the gas the
   next door reads).
2. **The exposure landing is a mode of the device `apply_packet`, chosen per
   (stage, kind).** A HIT inside a resolve stage holds its raises and zeroings
   per target in per-env scratch, and they land together at the end of the
   walk (`landing.land()`: raises skip a unit dead at the landing, then the
   dead are zeroed). Every other packet lands at once: the statuses, the
   couplings, the bite, and a PAYLOAD's blast packets in whichever stage they
   occur (`exchange.apply_blast_damage` takes no landing in the reference; a
   finding for Erik, not changed). One packet law, two modes; a caller never
   picks the mode.
3. **What flies and what is placed are TABLES with a schema and a capacity:**
   the planted charges (their schema exists), the grenades in flight, the
   rounds in flight, the multi-tick blast sources (`gmap.expansions`) and each
   entity class. Each is built with ONE table module over the unit store's
   machinery (schema rows, `tools/gen_unit_layout.py`, `pack` / `unpack`, an
   append, a stable compaction, an IDENTITY gate), never a second generator.
   P7.0 builds the module with the charge table as its pilot; every later
   table is a client. Row order is the host list's order, kept by the stable
   compaction (survivors, then this tick's appends: the order every reference
   loop walks); an object reference (shooter, weapon, ammo, payload, duct)
   becomes an id or a table row index; a variable-length member (a door's
   span and per-tile hp, a duct's slots) is a span into a side table, as
   `door_spans` already is. None of the host lists has a bound today except
   the charges: each table gets a capacity sized from the largest count its
   gate scenes and the bots' matches reach, and a full table is counted and
   refused by name, like the ring.
4. **The logs are per-env ordered lists, each with its own life.** The CREDIT
   CLAIMS and the BLAST LOG are per tick and go with the tick-end clears. The
   NOISE LOG is not: nothing clears it at tick end, it is drained at the head
   of slot 5, and a bash's or a 9e detonation's noise is heard on the NEXT
   tick. So the noise list persists across the tick boundary, is drained at
   the slot-5 stage's head and cleared only at env reset (added to §4.6's
   list). Its order is load-bearing (hearing ties to the earlier noise): slot
   order, within slot 4 the shooter's id order (the gather), within a stage
   the walk. The claims keep "first claim wins" by being read first match
   first (`credit_at` already scans so). Each list arrives with its first
   writer (noise: P7a.2; claims: P7a.3; blast log: P7c.1).
5. **Draws.** The device forms of `MatchDraws.at` and `.next`, `cone_offset_q16`
   and `roll_q16`, over the vendored Philox, with the `Salt` members reaching
   C++ as generated constants. `.next` counts per `(salt, id)` within a tick in
   a host dict; its device form is a counter local to the row's thread, which
   is exact while every `.next` user of one salt sits in ONE kernel per tick.
   That holds today (SHOT_CONE and BEAM_CONE: one shooter's draws in slot 4;
   MELEE_CRIT only on the order path). The kit's generated table carries the
   one site per `.next` salt, so a second site fails a test instead of
   silently restarting the count.
6. **Sight is a device library function in both of `has_los`'s forms:** the
   sight form over `solid` and `sight_screen`, and `physical=True` over
   `solid` alone (the spray cone's), each integer for integer and held equal
   on random pairs over real maps. `sight_screen` is a slot input until P7e
   makes it resident. The bulk form (`rays_clear`, the observation's) stays
   P8b's.
7. **The edit door moves to the device in three steps.** (a) P7a.3: device
   producers append EDIT and TILE rows through device forms of the door's
   quantization and the region walk for the regions they use. (b) P7c.1 and
   P7c.2 add their regions (DISC, TILES, the falloff weights) the same way;
   P6b.4 the heat source's. Pumps and vents enqueue no edit. (c) P7e: the
   per-env sort `(field, source, region, seq)` and K1 / K2 fed from
   device-built buffers. Three rules hold throughout:
   - `seq` is a per-env counter, plain because a device producer is always an
     effect function of the serial applier. It RESETS AT THE 6b FLUSH, never
     at tick end or the tick head (one window runs from the previous tick's
     9e to this tick's 6b), and is cleared at env reset. It is also the noise
     key, so for a noisy edit its absolute value matters, not only the order.
   - Until (c) a producer is gated AT THE DOOR: its rows equal
     `pack_for_device` of the edits the host producer enqueued, with `seq` and
     the edit-row indices rebased to the producer's first edit. That is exact
     for noise-free edits (a spray's carry none). A noisy edit's absolute key
     is compared only on ticks where the window's base is known (the producer
     is the window's first, or the base is a harness input); a tick on which
     host and device producers interleave inside one walk is refused by the
     gate by name, never compared loosely.
   - Until (c) a CHAINED run lands a device producer's edits through the
     host's own enqueue, replayed from the device's records, which must
     reproduce the device's rows (the request list's `apply_to_host`
     pattern). The played edit queue (Class P) is untouched until (c).
8. **The harness learns lists.** A host list a slot reads or writes (the
   effects, the tables of decision 3, the logs of decision 4, the edit rows) is
   declared like a field, as an input or an output, through ONE record function
   per list; never a hand-rolled compare. The ring keeps only synced events:
   the harness drops render-only events before comparing (slots 4 to 4c emit
   several). ExplosionEvent is render-only, read by no law, and does not fit
   the ring's record: it gets NO device form, and P7c.1 corrects the Event
   ring's rule, which still promises it one. `ChainedSlots` stays N = 1; batch
   invariance stays the shadow's.
9. **Every patch builds its own scenes and counts its branches.** The shipped
   levels and bots take few of these paths: no level holds a spray, a hitscan
   weapon, a payload round or a heat source; no bot plants or throws smoke;
   door bashing and the use verb appear in one level that no CUDA gate runs;
   the 300-tick corpus ends before the first exposure rise. So each gate is a
   hand-built scene (P6.4a's verb scene and P6.4b's fire scenes are the model)
   with a census of the Python law's branches, each required to be taken, and
   the branches a scene cannot reach named with the reason.

**The cut.** Thirteen patches, two side by side (the Home Desktop's limit). A
free slot takes the next patch whose needs are merged; the stages below are
the order that follows from the needs. Each patch is re-checked against the
inventory when it starts and merges main before its final gates. Two patches
side by side both regenerate the generated headers (`unit_tables_layout.h`,
`applier_layout.h`): the second to merge regenerates after merging main and
rebuilds, as P6.4a and P6.4b did.

| stage | patch | builds | needs |
|---|---|---|---|
| 1 | P7.0 the foundations | the kit, library only, each function held equal to its Python on synthetic inputs through test-only probes: the draws (decision 5), sight (6), the falloffs (`falloff_round` / `falloff_floor`), the squared-distance tests. The table module (decision 3) with the planted-charge table as its pilot. The reference's slot 2 split into two named calls (grenades, rounds) with two profiler labels and a sole-cause proof, so two later patches can each port one. Slot 9d (ignition from temperature: a per-cell law, order-free), which no row of the old plan named | |
| 1 | P7a.1 the effect list | decisions 1, 2 and 8; kinds HIT / STATUS / PUSH / CHEW (`chew_wall` over the topology seam) and the PAYLOAD record; slot 4a (melee: `can_melee`, the arc target, the melee rows) as the first producer; slot 4c as the first stage | |
| 2 | P7a.2 shooting | slot 4's `_directional_fire`: the burst, `start_spray_burst_directional` and its noise, and the beam as a stretch that may ship as its own patch; the cone draws; the weapon and ammo rows, the per-material cover exposure and the zombie bullet multiplier; THE MARCH as a library function; the table of rounds; recoil and push; the noise list; slot 2's rounds (the same march, a landing stage). A payload round's stop is recorded and refused at the apply until P7c.1 | P7.0, P7a.1 |
| 2 | P7b grenades and charges | the table of grenades (the throw request becomes an append), flight, the fuse-out as a PAYLOAD record; `charges.tick` over the charge table (the plant request becomes an append); the slot-10 prune. Detonations are RECORDS here, compared with the host's `execute_payload` calls | P7.0, P7a.1 |
| 3 | P7a.3 sprays | slot 4b (the cone over physical sight, the cell buffer), SPRAY_CELLS, the edit door step (a), the claims list | P7.0, P7a.1 |
| 3 | P6b.1 doors | the door table (the first entity table: ordinal order, the span and hp side tables, its rows of the entity section); `seal_tiles` / `unseal_tiles` / `can_seal_tiles` in the topology seam (the refusal path; the float planes as bit patterns); the gas-N primitives (`inject` / `extract` / `transfer`, over the gas energy seam's shared rows) and `_aperture_blocked` as device functions; the door sweep with its occupancy test and external destruction; the use request becomes a table write | P7.0, P7a.1 |
| 4 | P7c.1 payloads, the disc path | `execute_payload` as the PAYLOAD effect function for rows with `strength = 0`: the disc and its wall damage (serial on thread 0, row-major), the wave edits, blast damage, the flash, `emit_gas`, `ignite_ring`, `deposit_heat` (the DISC region at the edit door), the noise, the blast log, the claims | P7.0, P7a.2, P7a.3, P7b |
| 4 | P6b.2 vents | ducts and vents (bus-independent): the ledgers in int64 with floor division through the kit and a named refusal past the proved range | P6b.1 |
| 5 | P7c.2 payloads, the strength path | rows with `strength > 0` (the shipped frag is one): the reach walk, the air read, `start_source` / `start_expansion` with the blast-source table and the device form of `take_source_plane` (so `div_source` is no longer uploaded), `deposit_over_tiles` with linear weights, the fan-out over the block's threads, bulk gas; the detonation scene | P7c.1, P6b.1 |
| 5 | P6b.3 signals | the signal bus, the sensors (motion sensors over sight; field sensors through a device form of the accessor's raw `sample`), the logic nodes (predicates as codes), the pumps, and 9e(c) (the doors' inputs from the bus) | P7.0, P6b.1 |
| 6 | P7d the horde | the flood as a fan-out of the stage's block, run when the walk first needs a field, from the live map; the masks; the cache with §4.9's refresh rule, including the ticks that return before `begin_tick` (no prey, or an empty request set) and leave it standing; passes 1 and 2 wide; pass 3 serial in id order with the bite and the bash applied at once; the wander hash; the match's zombie request set as a host latch | P7.0, P7a.1, P7a.2, P6b.1 |
| 6 | P6b.4 timed charges and heat sources | the two entity classes that detonate; the heat source's chain within one tick (a loop over a growing list, bounded by the first discs plus the source count) and its per-tick deposit; slot 9e CHAINED whole; the entity and signal sections of the digest exported from the tables | P7c.2, P6b.2, P6b.3 |
| 7 | P7e the device tick | the conductor: every slot's kernel in the reference's order for N envs with no host slot between them (the host writes only the latches: actions and the zombie request set); every plane a device slot reads or writes resident between slots (`applier.non_resident_planes()` empty: `sight_screen`, `heat_atten`, `light_atten`, the radiation planes); the edit door step (c); the remaining tick-end pieces; profiler labels per sub-slot. Gate: recorded matches replayed through it, digests equal at every tick, N = 1 and in a batch | all |

P7e is the input of G2 and of P8b; it is re-cut when P7c.2 has merged,
together with the P8 re-plan (P8b grew when rung 1 of #192 built the
observation, the masks and the reward in host Python: that re-plan is with
Erik).

**P7e RE-CUT (2026-11-19; Erik: "seems like it can run without me").** Cut
from a read-only inventory of main `92f2bf3f`,
`docs/residency_p7e_inventory_2026-11-19.md` (the reference tick as a table,
the twenty host seams H1 to H20 between device slots, the refusals, the
swarm). What the inventory found that the row above did not say:

- nothing in `src/` drives a device unit kernel today: every face is called
  by the check scripts alone, at N = 1, re-staged from the host at each
  chain's entry; no chain spans slots 6 / 6b / 7 or a tick boundary;
- "the radiation planes" of the row are three dead planes that are only
  cleared (delete them); the float `heat_atten` / `light_atten` have no
  reader; `ignition_armed` and the flight plane are missing from the row;
- the host latches are five, not two: the actions, the zombie request set,
  the order table, the tick and draw keys, and the tile-change serial (which
  device slots move inside the tick, so it becomes a device counter);
- there is no device writer for the actions (only a whole-row pack), no
  device sort, no device form of slot 9b, and the stamp's rows and refill
  are host-built;
- every face downloads its counts and raises on the host (about 30 syncs a
  tick against the two the resident rule allows);
- no recorded match is checked in, and a recording holds a checkpoint every
  60 ticks: the gate records its own matches and compares every tick with a
  host re-simulation beside it.

P7e is therefore five patches, one implementer at a time, each gated at
tolerance 0 against the reference:

| Patch | Builds | Needs |
|---|---|---|
| P7e.1 planes and latches resident | the dead `rad_*` planes deleted; `sight_screen`, `ignition_armed`, `div_source` device buffers of the batch (the float atten planes dropped from the plane list if truly unread); the flight plane on the device; a per-env device TICK ROW (tick, draw key, tile-change count) read by every kernel in place of the per-launch arguments and the latch serial; 9c / 9c3 fed from the device claims list. Gate: every existing shadow and chained check unchanged with its inputs from the resident buffers; batch invariance | none |
| P7e.2 the device sorts and 9b | the burn-through list sorted row-major on the device and fed to TOPOLOGY; a device form of 9b (the scan, the top-16 with the host's tie order, DESTROY_WALL records); the stamp's rows built from the unit store in the host stamp's row order, and its refill. Gate: shadows of 6 / 9 / 9b on fire, breach and walking scenes | P7e.1 |
| P7e.3 the edit door, step (c) | the per-env stable sort `(field, source, region, seq)` on the device, the buffers K1 / K2 read, `pack_for_device`'s range checks as faults refused by name, `seq` reset at the 6b flush. Gate: K1 / K2 fed from device-built rows equal to the host pack every tick on the spray, payload, hazard and heat-source scenes; books equal | P7e.1 |
| P7e.4 the conductor at N = 1 | the store, tables, lists, door and cache resident ACROSS ticks; one function walking the reference's order; a latch writer for the actions (a column scatter) and the order table; every face's counts deferred to ONE books read per tick, raised there by name, and a refused tick is refused, never compared. Gate: recorded `ship_demo` matches (`bots`, `bots_v1`, full length) and the hand scenes replayed, `(field_digest, tick_digest)` equal at every tick to a host re-simulation | P7e.1 to .3 |
| P7e.5 the batch and the profile | N envs; profiler labels per sub-slot; the swarm's kernels in the same walk. Gate: env i of N equal to env i alone at every tick; the capture check (two named syncs) | P7e.4 |

P7e.2 and P7e.3 are independent of each other and may run side by side.
The three risks the implementers are told about: the actions latch (the
replay's intents must be shown to touch latch columns only, or the gate
diverges at tick 1); host-only side effects the reference reads later, some
in no digest (the tile serial, the solid-books sink, the swarm's
`high_water`, the claims dedupe); and ranges over a full-length match (the
device gates ran 500 ticks, `bots` runs 2721).

**What each patch must know** (the inventory's Parts B, D and F, checked by
the critique):

- Grenade flight reads no map state, so it is order-free; the detonations are
  what is serial. Slot 2's order stays: every grenade, detonations in list
  order, THEN the rounds' phase A against the state the grenades left, then
  the landing.
- The march's first hit is the first unit in LIST order whose footprint holds
  the tile, and the shooter is excluded by object identity: on the device,
  row order and the shooter's id (a converted shooter keeps its id).
- `apply_explosion`'s loop reads `solid` after earlier tiles of the same disc
  were destroyed: the disc stays serial.
- Slot 5's bash is applied at once and the zombies after it read the live
  map for their step, their own bash, their fear and their sight, and the
  lazy flood is built from the live map on first need. So pass 3 is never a
  wide decide with a deferred apply, and the flood is never moved to the
  stage's head.
- Float conversions on the payload path (`float(wall_damage)`, the heat and
  soot products, `_source_q`, the DISC radii, `blast_air_settings()` per call)
  and the heat source's float per-tick deposit are quantized per call in the
  reference: each becomes a Device tables row read from the same validated
  object, and a product that cannot be a row is reported before it is ported.
- Duct ledgers and the filter mean are unbounded Python integers with floor
  semantics on negatives: int64 on the device, a named refusal past the
  proved range.
- State that crosses ticks and is in no digest: rounds and grenades in
  flight, `gmap.expansions`, the edit queue's window, the noise log, the horde
  cache. The tables' IDENTITY and shadow gates cover the twins; adding any of
  them to the digest would move every golden and is not part of this port (a
  finding for Erik).

**As built (P7e.1, 2026-11-19).** Branch `29-p7e1-planes-latches`.
- The dead `rad_net` / `rad_amb` / `rad_flux` left `applier.CLEAR_PLANES`,
  the host clear and the device `tick_end_clears` (heat alone). Not digested,
  no golden moved. The GameMap attributes stay (tests fill them; `rad_net` is
  still the dead `step_tail` argument, recorded debt).
- `heat_atten` / `light_atten`: read by no law, but read by tests
  (`tests/test_unit_permeability.py` reads `g.light_atten`,
  `tests/test_a5_seal_evacuation.py` compares both), so they were kept and
  made RESIDENT with the other planes rather than dropped.
- `GameMap._RESIDENT_UNIT_PLANES` = sight_screen, heat_atten, light_atten,
  ignition_armed, div_source: batch buffers. Choice where the design was
  silent: while host gameplay is in the loop the played tick refreshes the
  first three with the `_tile_serial` upload and the last two every tick
  (`_RESIDENT_ENTRY_STATIC` / `_ALWAYS`); the downloads are unchanged (no
  device slot runs in the played tick). `applier.non_resident_planes()` now
  covers every plane a device unit slot reads or writes
  (`device_slot_planes()`: the seam's PLANES, the clears, each face's field
  tuple, the flight plane's two sources); the survivors are the ripple pair,
  host-computed by design (`applier.HOST_COMPUTED_PLANES`).
- The flight plane: a device builder (`unit_kernels.flight_plane`,
  `k_flight_plane` in `cpp/src/cuda_unit_objects.cu`, table row
  `mat_blocks_thrown`), not the walk reading `material`: simpler and the
  walk stays as proved. The harness builds it on the device from padded
  solid / material (padding solid, so blocked as before).
- THE TICK ROW: `unit_draws.TickRow`, `(N, 5)` int64 (tick, seed_lo,
  seed_hi, draw_tick, tiles_changed), generated `TICK_ROW_*` in
  `unit_draws_layout.h`; every unit kernel face takes it (expire_orders,
  verbs, match_done, melee, grenades, charges_tick, shooting, rounds_march,
  EMIT, HORDE; the faces refuse host ticks by TypeError); the seam's
  `on_tile_changed` bumps tiles_changed when the walk is handed the row
  (`walk_effects(..., tick_row=)` and its wrappers resolve / detonate /
  burn_through / apply_effects); the HORDE latch is the request set alone.
  The swarm's env words (slot 7, Class P) still take the host's tick: left
  to P7e.4 / P7e.5. The tick-end slots read the row after the advance: the
  latch writer moves the tick (P7e.4's conductor), never a kernel.
- 9c / 9c3 read `attack_lists.claims_list` in place (`credit_at` over the
  env's first count records); `unit_kernels.claims_of` / `pad_claims` were
  removed. In the shadow the list is staged from the TickCredit's first
  claims (`attack_lists.claim_rows`), the fold `credit_at` answers alike.
- Gate: `tests/cuda_unit_resident_check.py` (REFRESH, INPLACE, FLIGHT,
  TICKROW, CLAIMS), each part shown red by a deliberate break (a staged
  plane copy, the flight builder without the material, the walk without the
  row, 9c without the list).

**As built (P7e.2, 2026-11-20).** Branch `29-p7e2-sorts-9b`. One Class-T
TU, `cpp/src/cuda_unit_sorts.cu`, faces in `unit_kernels`; no physics, no
golden, no played-tick behaviour moved.
- THE BURN-THROUGH SORT: `unit_kernels.burn_through_device(store, applier,
  fields, burn, tick_row)` empties the effect list, ranks each env's list
  (`BurnList`: device addresses of the `(N,)` counter and `(N, S)`
  arrival-order indices) with a STABLE RANK in one block per env (each index
  written straight to its rank's slot after what the list holds, the rest
  counted dropped), then walks TOPOLOGY. The resident fire's own list is read
  in place through `bp.resident_burn_list()` (the arena's fire counter, the
  `dlist` buffer; a new accessor in `cuda_resident_tick.cu`, nothing else in
  the played tick changed). Choice where silent: a rank, not a sort network
  -- the list is small (capped by the effect list's 256) and the rank is
  exact with ties; a count or index outside the grid FAULTS the env.
- SLOT 9b: `Simulation._burst_walls()` is now a named call (the body moved
  unchanged; the profiler row `9b over-pressure walls` hooks it instead of
  the bare scan, so the shadow sees the destroys). `unit_kernels.burst_walls`
  scans in raw Q16 ints: the reference's float64 spread is `S / 2^16`
  exactly, so `spread > t` is `S > floor(t * 2^16)` (row `burst_thresh_q`,
  -1 never, clamped at 2^40 which no int32 spread reaches); the vacuum side
  0, the ambient side `burst_p_amb_q`; the top-K by `burst_cap` rounds of a
  block reduction under the total order (spread descending, then row-major),
  each pick strictly after the last. `burst_cap` is 0 while `burst_enabled`
  is off (the shipped config: 9b does nothing in game).
- THE STAMP FROM THE STORE: `unit_kernels.stamp_rows` / `stamp` build
  `take_stamp_rows`' 9-column rows from the store (per-row tile count, a
  one-thread prefix over (env, row), the fill; rows past the total env -1 so
  the launch takes the capacity N * U * 64 and the host reads no count), run
  `stamp_launch_resident` unchanged, and the refill (freed captured before
  the reset; one block per env walks the grid in 256-cell chunks, thread 0
  serially in row-major order only where a chunk holds a freed tile). The
  per-unit values are the map defaults (`stamp_*` rows); a sim whose unit
  sets an optics override is refused at the table build
  (`stamp_optics_default`). The played tick keeps the host rows and the host
  refill.
- Finding (reported, not changed): on the C++ stamp path `obstacles` IS
  `solid` (`_update_caches` aliases them and the C++ stamp writes in place),
  so `destroy_wall` opens `obstacles` too and the refill never fires there;
  the gate reaches the refill through the Python stamp path
  (`use_cpp_stamp = False`).
- Gate: `tests/cuda_unit_sorts_check.py` (BURN: slot 9 shadowed with the
  host's list SCRAMBLED, fire scenes N = 1 / 2; RESIDENT: the resident fire's
  arrival list sorted in place == the played tick's list, 7 of 40 ticks
  arrived unsorted; BURST: a built membrane scene with 35 candidates, the cap
  cutting a tie, later scans of distinct spreads, N = 1 / 2; STAMP: ship_demo's
  walk and fire scenes, rows == the host C++ stamp's rows every tick, the
  refill on the Python path), wrapped by `tests/test_cuda_unit_sorts.py`. Each
  property shown red by a break in the TU (the rank replaced by the arrival
  index, the tie order reversed, the footprint walked column-major).
  `tests/cuda_unit_lockstep.py`'s write-set compare learned tuples of arrays
  (the stamp's kept rows).

**As built (P7e.3, 2026-11-20).** Branch `29-p7e3-edit-door-c`. One Class-T
TU, `cpp/src/cuda_edit_flush.cu` (`EditFlush`, header `cuda_edit_flush.h`),
face `edit_door.DeviceEditFlush`; K1 / K2 unchanged; no physics, no golden,
no played-tick behaviour moved (the played tick keeps the host pack).
- THE SORT: `k_flush_edits`, one block per env, RANKS each door EDIT row by
  `(field ordinal, source, region)` with the row index as the tie. The door
  writes rows at consecutive seqs (checked: a gap or a seq that does not end
  the window refuses the env, `key`), so the index order is the seq order and
  the rank is `take_ordered`'s stable sort. Choice where silent: a rank, not a
  sort network or a radix sort -- exact with ties, one int64 of scratch per
  row, O(E^2 / 256) compares per env (E at most the door's 8192; the scenes'
  busiest window held a few hundred edits). The per-edit tile spans come from
  a binary search over the tiles' (ascending) edit column; thread 0 takes the
  tile prefix in applied order (serial, E steps).
- THE LAYOUT: fixed-stride edits (env e's at `e * edit_cap + rank`, E_ENV = e,
  plus one DUMMY row), tiles compacted per env in applied order with the edit
  index batch-global, env offsets from the totals; `k_flush_tiles` writes the
  scalars from THE TICK ROW (`draw_tick & M32`, seed lo, seed hi) and the
  tables' `topo_t_amb_q` (an existing row: no layout regenerated), and pads
  the tiles past the batch total with the dummy (amount 0, den 1) so K1 takes
  the capacity `N * tile_cap` and the host reads no count.
- THE REFUSALS (`edit_door.FLUSH_FAULTS`; the lowest code then the lowest row
  per env wins, so the name is stable): door overflow, an unknown field,
  water_depth, a gas slice outside [0, G), a CONSERVATIVE slice (the tables'
  `gas_trace` row), a mode, den outside (0, 2^62), a noisy low bound, the
  key, a tile, the product >= 2^125, the contribution >= 2^55 (exact 128-bit
  compares). A refused env writes nothing (its tile range empty); the others
  apply. The pack's "noisy edits without MatchDraws" has no device form: the
  tick row is built from `sim.draws`' key, so such a sim is refused at the
  latch. Finding: the product bound is implied by the other two (den < 2^62
  and contribution < 2^55 give a product < 2^117), so a row "just inside" it
  is refused as `contrib`; it stays as the pack's own check.
- THE WINDOW AND SEQ: nothing in the flush empties the door before its build;
  `run` ends with `EditDoor::begin` (rows empty, seq 0). `EditDoor.reset_env`
  resets one env (an env reset in a batch).
- THE BOOKS: K2's books and the fault words share one allocation and come
  down in ONE copy (`books()` / `fold()`), raised there by name
  (`EditFlushRefused`, then K2's failure slot as `EditQueueFailed`), folded by
  `physics_runner.fold_edit_queue_books` -- the resident tick's
  `_fold_edit_books` body moved to a module function both call.
- Gate: `tests/cuda_edit_flush_check.py`. SORT (seven fields, three regions, a
  noisy scaled DISC, ties, scrambled arrival, N = 1 / 2; the host's sort,
  pack and buffer made to raise during the device run); RANGE (each refusal
  by a row the host refuses or falls back on, the other env applied; den
  2^62 - 1, contribution 2^55 - 1 and the last trace slice accepted equal);
  WINDOW (rows across a tick-row advance); SEQ (an absolute base 5 noisy edit,
  the reset, `reset_env`); SCENES (spray, payload, hazard, N = 1 / 2, 70 ticks
  each: the host's every enqueue appended to the door at its seq, slots 7 to
  9e's after the flush so they cross the boundary; the door == the host
  window at each 6b, planes and books == the host-pack flush, door empty
  after; census 444 edit ticks, 257 out of order with ties, 29 noisy, 216
  spanning the boundary). Each property shown red by a break in the TU (the
  rank by index alone, the contribution check removed, the door emptied
  before the build, the reset removed). The producers themselves are not
  re-run in SCENES: the rows are `door_rows` of the host's edits, which the
  door's own gates hold equal to the device producers' rows.

**As built (P7e.4, 2026-11-20).** Branch `29-p7e4-conductor`. The device
conductor, `simulation.conductor.DeviceConductor` (Class T); no physics, no
golden, no played-tick behaviour moved (slot 7's chain is code motion).
- THE SHAPE (a choice where the design was silent): the walk is a Python
  function, one call line per slot of `Simulation.step()` over the
  `unit_kernels` faces (each one to three launches), not a C++ orchestration.
  Its cost on ship_demo is about 11 to 12 ms a conductor tick (all slots,
  slot 7 included, the latch included; host Python plus launches), measured
  by the gate on the pinned matches.
- STATE: the twin sim seeds the device ONCE at the build (its map made
  resident through `ResidentBatch([sim])`, the store packed and uploaded, every
  object and entity table staged, the horde cache staged persistent, the door
  at the host's seq); nothing is restaged between ticks (the gate's restage
  probe counts every store / table / cache / `from_host` upload inside a tick:
  zero; shown red with `CONDUCTOR_BREAK=restage`). The ripple pair, the only
  non-resident planes the seam writes, are conductor-owned device copies.
- THE LATCHES: `latch(intents)` takes the tap's `(kind, args)`; each intent
  method (`set_move_dir` .. `set_flashlight`, `plant_charge_intent`) writes ONE
  `Unit` attribute from its arguments alone (it reads the unit's existence and
  nothing else; `throttle` and the plant fuse are clamped BEFORE the tap, so
  the recorded args are the written value; `_possessed_ids` arms no device
  slot), so the latch is a CuPy column scatter of the schema's encodings into
  the store (`LATCH_COLUMNS`), last write per (unit, column) winning. The
  order table (`order_intent`, `until` from the conductor's tick) is staged to
  slot 1b' only on a tick that holds an order (the host's dormancy); the
  zombie request set is the HORDE latch. `order_intent`'s door
  (`unit_orders.check`) reads `u.alive` / `u.is_zombie`, but it refuses BEFORE
  the tap: a recorded order is one the reference accepted. The tick row is
  written once at the build and advanced on the device at slot 10 (tick and
  draw tick); the twin's host `tick` / `real_time` advance beside it (the
  swarm's env words read `sim.draws.tick`: left to P7e.5 as cut).
- SLOT 1b's two host inputs (H3) are built on the device:
  `unit_kernels.verbs_device` -> `unit_verbs_tables` (one thread per env walks
  the door table for the LIVE doors' span tiles, counts the charge table's
  live rows), the same three kernels after it.
- SLOT 7: `physics_runner._resident_chain` (the water stage to the tail's
  books D2H) is now ONE function both `run_resident_batch` and
  `run_resident_conductor` call; the conductor's slot 7 runs the head's host
  books (trace dt check, saturation books), the chain with the SOURCES stage's
  device hand-over, no stamp rows / edit buffer / entry upload / exit download,
  and brackets the solver backends ON (restoring them, so a host
  re-simulation in the same process keeps the CPU path). Slot 9 ranks the
  resident fire's own list (`BurnList.resident`); 9b, 6 and 6b are P7e.2's and
  P7e.3's device forms.
- THE TILE SERIAL: every walk that can change a tile is handed the tick row
  (`sweep_doors`, `timed_charges`, `heat_sources` gained the argument): without
  it the door sweep's seal / unseal left the serial behind and the horde
  cache's refresh rule diverged (found by the gate's cache compare).
- `obstacles` IS `solid` on the host's C++ stamp path (P7e.2's finding): the
  device mirrors the alias (the stamp writes the `solid` buffer for it);
  without it every opened door or burnt wall diverged in `obstacles`.
- ONE BOOKS READ (`cpp/src/cuda_unit_books.{h,cu}`): while a conductor tick
  runs, `UnitStoreDevice::end_counts`, a record list's counts-only download,
  `FieldApplier::download_books`, the door and object slots' counts and the
  flush's books copy D2D into one arena under the call line's label and hand
  the host zeros; `_books_read` raises by name at the tick's end (a fault, an
  overflow, each applier book that refuses, a shot / spray refusal, the
  flush's refusals; the flush's books are folded there). `DeviceTable.check`
  reads the counts alone (it downloaded the whole table); a source-free
  level empties the blast log by the record list's `begin` (no upload).
  Every unit-side host sync now goes through `counted_sync` /
  `counted_event_sync` and `tick_sync_point` notes its own: the S4 count.
- GATE `tests/cuda_conductor_check.py` (wrapper `tests/test_cuda_conductor.py`),
  the reference on the CPU path beside the conductor's twin,
  `match_replay.match_digests` compared every tick, then the state in no
  digest (rounds, grenades, noise, sources and their tiles, the edit door's
  window, the horde cache) against the reference's record functions:
  S1 the hand scenes (payloads, hazards 0 / 2, sprays, horde and horde with
  the controller, signals, fire 0 / 1, walking), between-tick fixtures hoisted
  into the setup or dropped, 160 ticks each, every tick equal; RESIDENT the
  restage probe, zero; S2 a 300-tick ship_demo bots match recorded through
  `run_match(record=True)` and replayed (the reference by the replay
  controller, the conductor from the intent rows), every tick equal; S3 the
  pinned `bots` (1349 ticks, 22867 intents) and `bots_v1` (272 ticks, 6566
  intents) recorded and replayed to their end, every tick equal, no refusal
  met (no beam in the loadout; vent ledgers, the horde cache, bulk gas, the
  list capacities and the edit door's caps held over the full length); S4 3
  host syncs per steady conductor tick on payloads, hazards and ship_demo
  (`max_n_sub`, the physics books, the books read; the first tick's scratch
  growth, 6 syncs, is warm-up) and nothing else, shown red by a face's download
  inside the tick (`CONDUCTOR_BREAK=sync`). The digest compare shown red by
  leaving slot 2b out (`CONDUCTOR_BREAK=slot`: a unit's statuses, tick 1).
- Twin rows `payload`, `sources` and `conductor` (Class T); the signal and
  hazard stages stay `[not_twinned]` (neither the corpus nor a played
  ship_demo match holds a bus, a timed charge or a heat source, so no tick
  traces them: a twin row would have no files), their reasons corrected.
- Host seams left open, by design (named): the twin's host clock beside the
  device row; the swarm's env words (host tick, P7e.5); the order table's H2D
  on a tick with an order; the team picture, the recorder and the render-only
  events (swarm deaths, Wall / DoorDestroyedEvent) not run; the `Z-` name and
  the momentum books (diagnostics read by no rule) set / not kept on the twin;
  the solid books' sink (`TemperatureSolver.book_solid_seam`: the closure
  identity's `e_solid_seam_sum` and the books snapshot, read by no law; the
  device's twin is the applier's solid book, kept in `last_books`).

**As built (P7e.5, 2026-11-20).** Branch `29-p7e5-batch`. The conductor over
N envs, `simulation.conductor.DeviceConductor(sims)` (Class T); no physics, no
golden, no played-tick behaviour moved (the played tail's fold is code motion).
- THE BATCH: one walk, every face launched once for the N envs (the store
  `(N, U, W)`, padded to the largest unit count; every record list, object /
  entity table, the edit door and flush, the horde cache and the tick row per
  env; slot 7 through `physics_runner.run_resident_conductor(runners, gmaps,
  ...)`, the one batched chain over `ResidentBatch(sims)`, which refuses a
  heterogeneous batch by name). N = 1 is one sim (the P7e.4 gate unchanged in
  what it asserts). `latch(intents, env=e)` per env; every env's column writes
  land in ONE CuPy scatter at the tick's head. The 9e slots run when ANY env
  holds the class (an env without rows is a no-op of the face; held by the
  gate on the hand scenes that share a level). Choice where silent: AN ENDED
  ENV keeps ticking with no intents (its physics, zombies and timers run on)
  until the caller replaces it. A per-env reset was NOT built: it needs each
  per-env table / list / cache / store row / tick row restaged for env e alone
  (the record lists stage all envs at once today; `EditDoor.reset_env` and
  `ResidentBatch.reset_env` exist), and a refusal-free proof that nothing of
  the other envs is touched -- named in the rule file as an accepted gap for
  P8.
- REFUSALS PER ENV: `_books_read` reads each deferred block in its per-env
  layout (the store's counts count-major `(3, N)`, `conductor.counts_by_env`;
  door / object counts env-major; applier books `(N, B)`; the flush's books
  `(N, slots)` then N fault words) and raises the first refusal in walk order,
  lowest env first, with `.env` and "env k" in the message. A refusal REFUSES
  THE BATCH (choice: no face was shown unable to let one env's fault reach
  another's state, and the refused env is partly written; letting the others
  continue needs that proof per face plus the per-env reset).
- THE SWARM: `swarm.engine_args(..., tick_row=)` hands the conductor's tick
  row address to the resident swarm stage (`ResidentTickArgs.swarm_tick_row`);
  `swarm.cu` writes each env's launch words (draw tick, seed lo / hi) from the
  row ON THE DEVICE after the launch block's upload (`k_launch_from_row`), the
  host env words zeros. The host `high_water` stays the launch extent (a load
  constant in a match: nothing on the host spawns during a tick).
- TWO SYNCS: the physics tail's fold is deferred (`bp.resident_tail_defer`):
  the tail enqueues its books D2H and returns; the conductor's ONE books read
  (`conductor_books_end`, one `cudaStreamSynchronize` on the tick stream)
  waits for it too, then `bp.resident_tail_finish` folds every env's books
  (`resident_tail_fold`, the played tail's fold moved to a function both
  call; the conductor's fold builds no burn-through list: slot 9 ranks the
  fire's own list on the device). A steady conductor tick syncs at
  `max_n_sub` and the books read: two, at N = 1, 2, 4 alike.
- THE PROFILE: `bp.conductor_profile_begin / _mark / _end`
  (`cpp/src/cuda_unit_books.cu`): while open every `conductor_label` and
  every mark records a CUDA event on the tick stream and the host's steady
  clock; slot 7's chain marks its stages (`physics_runner._PROFILE_MARK`: EOS
  substeps, trace tail, combustion) and the resident tail its own (bulk sum +
  fire, sweep, temperature, swarm, books D2H; a no-op unless open).
  `tick(profile=True)` lands `(label, GPU ms, host ms)` per call line in
  `last_profile`. The bench: `tools/bench_conductor.py`.
- REFUSALS VECTORISED: the books read checks each block whole (numpy) and
  walks envs only on a hit; `last_books` is `{book: (N,)}`. (The first,
  per-env Python loop cost about 40 ms a tick at N = 256: found by the
  bench, the fold is now about 2 ms flat.) The deferred arena is sized per
  env (`BOOKS_CAP_PER_ENV`, 1024 words; the flat 2^14 was crossed at N = 64,
  about 257 words an env): a capacity, raised.
- GATE `tests/cuda_conductor_batch_check.py` (wrapper
  `tests/test_cuda_conductor_batch.py`): every env played ALONE by the
  conductor at N = 1, then together; each env's `(field_digest,
  tick_digest)`, its no-digest state (content hashes of every array) and its
  swarm death events equal at every tick. SCENES: seven groups, N = 2 and 4,
  160 ticks each (payloads / sprays / signals / walking with per-env delays of
  the schedule; hazards 0 + 2, horde with and without its controller, fire 0
  + 1 side by side). MATCHES: ship_demo recorded at seeds 11 / 12 / 13 / 14
  (`bots`, `bots_v1`, `bots`, `bots_v1`; ends 1664 / 300 / 970 / 275 ticks),
  N = 2 and 4 replayed to the longest end, the ended envs ticking on without
  intents, every tick equal. SWARM: larva_studio seeds 1..4 (the broods'
  scatter differs), N = 2 and 4, 200 ticks, 600 live larvae an env; env 1
  alone == the CPU reference every tick (the launch words from the device
  row). SWAP: three envs in reversed and rotated order == alone (red with
  `CONDUCTOR_BATCH_BREAK=index`, the index in the draw key: an env differs at
  tick 5 / 9). SYNCS: 2 per steady tick at N = 1, 2, 4 (`max_n_sub`, the
  books read; red with `=sync`, a download per env: 3 / 4 / 6). FAULT: a
  beam trigger in env 2 of 4 refused as `ShotRefused` with `.env == 2` at
  tick 0, the batch refused (red with `=layout`, the counts read env-major:
  a `ConductorFault` naming env 0). P7e.4's gate unchanged and green (its S4
  now counts 2 syncs a tick: the physics books no longer sync).
- THE PROFILE (`tools/bench_conductor.py`, Home Desktop RTX 3070 8 GB, the
  machine otherwise idle; one recorded match's intents latched into every
  env, seeds 1..N; 10 warm-up, 60 timed, 20 profiled ticks; ms/tick is the
  wall time of `tick()` ending in the books read's stream sync, i.e.
  GPU-synchronised; GPU columns are CUDA-event time on the tick stream between
  call lines, host columns the steady clock; "slot 7" holds its head (the
  SOURCES stage), the EOS substeps, trace tail, combustion and the resident
  tail; ship_demo 70 x 100 with 12 units, range_hall 26 x 39 with 2):
| level | N | ms/tick | ms/env-tick | GPU unit side | GPU slot 7 | host unit side | host slot 7 | books read (host wait) | host fold | device MB (marginal per env) | top GPU sub-slots |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ship_demo | 1 | 10.23 | 10.233 | 3.29 | 4.95 | 4.54 | 3.70 | 0.38 | 1.80 | 1276 | 7 EOS substeps 40%, 7 tail: radiation sweep 8%, 7 physics head (sources) 7% |
| ship_demo | 2 | 10.86 | 5.430 | 2.76 | 4.63 | 4.12 | 2.97 | 0.68 | 1.78 | 1292 (16.0) | 7 EOS substeps 44%, 7 tail: radiation sweep 9%, 7 physics head (sources) 7% |
| ship_demo | 4 | 11.40 | 2.851 | 2.94 | 5.13 | 4.25 | 3.16 | 1.06 | 1.80 | 1318 (14.0) | 7 EOS substeps 43%, 7 tail: radiation sweep 11%, 7 physics head (sources) 6% |
| ship_demo | 8 | 13.08 | 1.635 | 2.89 | 6.97 | 4.32 | 3.07 | 2.91 | 1.83 | 1372 (13.7) | 7 EOS substeps 41%, 7 tail: radiation sweep 22%, 7 physics head (sources) 6% |
| ship_demo | 16 | 17.79 | 1.112 | 3.03 | 10.30 | 4.56 | 3.20 | 6.05 | 1.83 | 1480 (13.6) | 7 EOS substeps 41%, 7 tail: radiation sweep 29%, 7 physics head (sources) 4% |
| ship_demo | 32 | 27.39 | 0.856 | 3.23 | 18.34 | 4.58 | 3.23 | 14.32 | 1.87 | 1690 (13.4) | 7 EOS substeps 44%, 7 tail: radiation sweep 36%, 7 physics head (sources) 2% |
| ship_demo | 64 | 47.08 | 0.736 | 3.76 | 34.29 | 5.60 | 3.98 | 29.22 | 1.98 | 2106 (13.2) | 7 EOS substeps 46%, 7 tail: radiation sweep 40%, 7 combustion 2% |
| ship_demo | 128 | 87.62 | 0.685 | 5.20 | 66.25 | 7.40 | 5.39 | 59.96 | 2.26 | 2942 (13.1) | 7 EOS substeps 47%, 7 tail: radiation sweep 41%, 7 combustion 2% |
| ship_demo | 256 | 166.55 | 0.651 | 7.24 | 129.92 | 8.55 | 6.87 | 124.02 | 2.75 | 4532 (12.8) | 7 EOS substeps 47%, 7 tail: radiation sweep 43%, 7 combustion 2% |
| range_hall | 1 | 8.42 | 8.419 | 2.99 | 2.18 | 4.02 | 1.36 | 0.20 | 1.76 | 1268 | 7 EOS substeps 31%, 5 zombie AI 7%, 1b intents 7% |
| range_hall | 2 | 7.44 | 3.722 | 2.64 | 2.19 | 3.63 | 1.31 | 0.24 | 1.52 | 1278 (10.0) | 7 EOS substeps 33%, 6 stamp_units 7%, 7 tail: radiation sweep 6% |
| range_hall | 4 | 8.68 | 2.169 | 2.74 | 2.27 | 3.78 | 1.37 | 0.23 | 1.54 | 1288 (6.7) | 7 EOS substeps 31%, 7 tail: radiation sweep 7%, 6 stamp_units 7% |
| range_hall | 8 | 7.44 | 0.930 | 2.66 | 2.66 | 3.80 | 1.39 | 0.54 | 1.55 | 1310 (6.0) | 7 EOS substeps 29%, 7 tail: radiation sweep 14%, 6 stamp_units 6% |
| range_hall | 16 | 8.58 | 0.536 | 2.76 | 3.34 | 3.90 | 1.45 | 1.18 | 1.55 | 1352 (5.6) | 7 EOS substeps 24%, 7 tail: radiation sweep 24%, 1b intents 5% |
| range_hall | 32 | 10.18 | 0.318 | 3.40 | 4.89 | 5.17 | 2.08 | 1.57 | 1.63 | 1440 (5.5) | 7 tail: radiation sweep 31%, 7 EOS substeps 21%, 1b intents 5% |
| range_hall | 64 | 14.85 | 0.232 | 3.71 | 7.96 | 5.27 | 2.51 | 4.62 | 1.70 | 1616 (5.5) | 7 tail: radiation sweep 44%, 7 EOS substeps 17%, 7 physics head (sources) 4% |
| range_hall | 128 | 23.35 | 0.182 | 4.29 | 13.82 | 5.61 | 2.91 | 10.64 | 1.87 | 1974 (5.6) | 7 tail: radiation sweep 56%, 7 EOS substeps 14%, 7 physics head (sources) 4% |
| range_hall | 256 | 42.06 | 0.164 | 6.14 | 28.36 | 7.91 | 5.13 | 23.70 | 2.45 | 2664 (5.5) | 7 tail: radiation sweep 59%, 7 EOS substeps 15%, 7 physics head (sources) 6% |
  What dominates: at N = 1 the tick is HOST-bound (about 4.5 ms of Python and
  launches on the unit side, 3.7 in slot 7's call lines, 1.8 in the fold;
  the GPU idles between launches); ship_demo turns GPU-bound by N = 8,
  range_hall by N = 32, and from there slot 7 is 90 to 95 % of the GPU (the
  EOS substeps about 45 % and the radiation sweep about 40 % on ship_demo; the
  sweep 59 % on the small room, whose EOS is cheap) while the whole unit side
  stays 3 to 9 ms a tick for every N (every face is one launch for all envs).
  The per-env cost flattens at 0.65 ms (ship_demo) and 0.16 ms (range_hall)
  an env-tick. Device memory: about 1.27 GB fixed at N = 1 (both levels: the
  conductor's capacity-sized lists and scratch, not the grid; not split
  further) plus 12.8 to 13.6 MB an env (ship_demo) and 5.5 MB (range_hall).
  Largest N on 8 GB: ship_demo N = 384 fits (6.2 GB, 247 ms a tick, 0.643 ms
  an env-tick); N = 512 allocates (7.1 GB) but pages (10.8 s a tick: WDDM
  oversubscription), so 384 is the practical ceiling; range_hall N = 256 used
  2.7 GB. At these rates a 6 GB card holds about 300 ship_demo envs and a
  12 GB card about 800 (fixed 1.27 GB + 13 MB an env, the CUDA context
  aside).
- Host seams left open, by name: the latch (host intents -> one scatter per
  tick, the order table's H2D on an order tick); the swarm's `high_water`
  (host, a load constant in a match); the swarm's species rules and tables
  re-packed on the host per tick (constants per config); the EOS stage's
  `max_n_sub` read; the books read; the twins' host clocks; the team picture,
  the recorder and the render-only events (swarm deaths, wall / door events)
  not run; no per-env reset (above); the 1.27 GB fixed allocation not
  analysed.

**Systems (additions to §13).** New, each with its ONE index line and its
constraints in `.claude/rules/device-units.md` when it is built: the device
table module and each table on it (charges, grenades, rounds, blast sources,
the entity classes), the device draws, the device sight function, the device
edit door. Changed, each amending its rule section: the Per-env serial
applier (its reach, its producers, its stages), the Device unit library (the
march, the falloffs), the Device topology seam (seal / unseal, the gas-N
primitives), the Event ring (no ExplosionEvent), the Request list (throw,
plant and use become table writes and leave it). Used as they are: Match
draws, Verb gates, Weapons, Payload executor, FieldEdit, Horde field, Noise,
Entity system, Trace currency, Gas energy seam.

**As built (P7a.1, 2026-11-15).** Decisions 1, 2 and 8 landed as cut for the
kinds, slot 4a and slot 4c; no decision moved. One line each:
- THE EFFECT LIST is the applier's record widened: `kind` then ten arg slots
  whose meaning is the kind's own (`applier.KIND_FIELDS`, the header's
  `APPLIER_<KIND>_<FIELD>` columns): DESTROY_WALL `(y, x)`; HIT `(row,
  amount_q, dtype, attacker, indirect, expose, dir_x_q, dir_y_q, ap_q, cause)`
  (the packet at its door: `q()`, the AP quantized as `mitigate_q` does, the
  attacker normalized, the cause's index); STATUS `(row, status, magnitude_q,
  duration, source)`; PUSH `(row, jx_q, jy_q, lever_q, twist_q16)`; CHEW `(y,
  x, damage_q, live)` (`live` 0 for `wall_damage <= 0`, the damage quantized
  per call in the record); PAYLOAD `(y, x, payload, owner, approach_y,
  approach_x, has_approach, kind)` (`payload` the row's index in
  `sim.weapons_tables.payloads`, `kind` its index in `PAYLOAD_KINDS`).
  `ROW_CAP` 3 (4a's blow), `E_CAP` 256; the books gained `refused`.
- THE LIST IS A RECORD LIST: `FieldApplier` holds a `UnitRecordList` (width
  F, `n_units` rows of staging, `row_cap`), so a per-row producer stages per
  row and the one gather appends per env in row order AFTER what the list
  holds (an append continues it: the `append` gate stages two records and
  requires 4a's after them); `stage()` sets the list from host records (the
  first `cap`, the rest counted as dropped: the old semantics);
  `begin()` empties it. Python: `DeviceApplier(n_env, cap, n_units=,
  row_cap=)` with `stage_lists` / `begin` / `records`.
- ONE KERNEL, A STAGE ARGUMENT: `applier_run(applier, store, tables, planes,
  stage, ring)` (stages `TOPOLOGY` = slot 9, `RESOLVE` = slot 4c; a stage
  `LANDS` or not), returning the books, the coverage (the rows the walk
  changed, flagged per row and counted once; a row changed and changed back
  counts) and the ring's dropped count. Python: `unit_kernels.walk_effects`
  (THE refusals: `EffectOverflow`, `UnitEventOverflow`, `EffectRefused`,
  `ApplierFault`, in that order), `apply_effects` (host lists staged, default
  TOPOLOGY: slot 9 unchanged), `resolve` (the list as the producers left it).
- THE REACH: the store's rows (an effect's `row` outside the env's rows or on
  a padding row FAULTS), the env's device tables, the env's ring through
  `env_sink` (its first caller; a HIT with no ring faults). The planes stay
  required by every stage.
- THE LANDING is `ExposureLanding`'s own two LISTS per env (scratch `(N, cap,
  4)` beside the list: a raise `(row, dtype, add)` per slot, a death's row in
  the fourth column), not per-target sums: `land()` exactly (raises in record
  order, each `min(x_max, x + add)`, skipped for a unit dead at the landing;
  then the dead zeroed). `apply_packet` gained an overload taking an
  `ExposureLanding*` and the row index; the old seven-argument form IS the
  overload with null and is unchanged bit for bit (P6.3 / P6.4b gates green).
- THE KINDS' FUNCTIONS: HIT `apply_packet` (landing in RESOLVE); STATUS
  `apply_status` if alive (a stack-cap refusal changes nothing; a bad kind or
  duration faults); PUSH `impulse_add` if alive -- `unit_impulse.add` with its
  `angular_q` (the lever clamped per input, the magnitude's square in 128
  bits, the twist), MOVED to the unit library from the verbs TU's
  `stage_shove`, which now calls it; CHEW `chew_wall` (air and `live` 0
  no-ops; a tile outside the grid FAULTS where numpy would wrap or raise; the
  new hp outside int32 FAULTS; `destroy_wall` at `<= 0` through the seam);
  PAYLOAD counted in `refused`, never applied.
- SLOT 4a is `cpp/src/cuda_unit_melee.cu` (`unit_kernels.melee`): the latch
  consumed, `can_melee` (both moments in the library, the probe's columns
  13 / 14), `melee_arc_target` and `melee_adjacent` (the square footprints of
  `footprint_rowmajor`), `last_blow_tick` at once, the blow staged as HIT /
  STATUS / PUSH. A striker whose melee row has a crit FAULTS (the loadouts
  door refuses one; `melee_blow` would draw); a melee row without a dtype
  faults; the push is refused past 2^46. Eight table rows (T 716):
  `weapon_melee_damage_q`, `_dtype`, `_status_kind`, `_status_ticks`,
  `_arc_half_q`, `weapon_rof_ticks`, `weapon_melee_crit`,
  `weapon_melee_push_q` (per map: the impulse table's).
- THE ONE RECORD FUNCTION: `applier.record_of` / `records_of` (a world call
  recognised by its unwrapped function; FACE, a cover chew, a spray's cells,
  any other call, and an effect on another sim's map / table / events refused
  by name, `EffectRefused`); `applier.effects_of` finds a slot callable's
  `SimultaneousEffects`.
- THE HARNESS: `LISTS` (name -> host object, record function, consume,
  device list, field names); `inputs=` / `outputs=("list:effects",)`; an input
  list staged before the kernel, an output-only one begun; outputs compared
  record for record (`list[i].KIND.field`), the alone run's list too;
  `DeviceSlot.select(sim, args, kw, caller)` (4c is the conductor's resolve,
  `_step_pre_physics`); the ring compared on `unit_events.is_synced` events;
  `ChainedSlots` stages its lists at the chain's entry, empties the host's
  list after the kernels, and refuses by name a later chained callable that
  finds the host appended to it in between (4b's records would be lost).
- TWO CLASS-T TWINS, `melee` (slot 4a) and `resolve` (slot 4c). The corpus
  never strikes and its 4c resolves record no hit in 300 ticks, so `melee`'s
  traced files are `combat.py`, `simultaneous.py` and `resolve`'s
  `exposure.py`, `simultaneous.py`: a change to `damage.py` or
  `combat.chew_wall` does NOT put `resolve` in debt (the thin-trace gap
  again); the effect-list gate is the guard. `combat.py` and
  `simultaneous.py` left `[not_twinned]`.
- WHAT THE REFERENCE SHOWED (nothing changed in it): the profiler label "4c
  resolve" wraps `SimultaneousEffects.resolve` itself, so it also times slot
  2's landing resolve (`_update_projectiles`) -- the harness narrows by caller,
  the profiler still adds both under 4c; `melee_blow` computes the screen-
  convention `angle_q` on the verb path, where nothing reads it (no crit); no
  melee row a loadout may carry is without a status, so "a blow without a
  status" is unreachable; ship_demo's bots never strike (4a acted on 0 of 500
  ticks); melee packets carry no `expose`, so the landing's raises come from
  rounds (ship_demo: 4 raises landed, 4 PUSHes with a recoil lever); the
  landing's skip of a raise for a unit dead at the landing cannot change a
  result today (every unit dead there died inside the slot through a HIT and
  is zeroed after the raises: its deliberate break stays green) -- it bites
  first when a PAYLOAD's blast kills in immediate mode inside a resolve
  (P7c.1 must gate it); a HEAL packet heals only with a negative amount.
- THE GATES (`tests/cuda_unit_melee_check.py`, wrapper
  `tests/test_cuda_unit_melee.py`; CPU `tests/test_applier_layout.py`):
  TABLES; LIB (synthetic rows, both answers of each gate); KNIFE; SCENE (the
  melee scene, 4a + 4c shadowed, a census of 32 branches each required,
  four named unreachable); RESOLVE (the landing's result different from at
  once, CHEW to zero, forward / reversed differ in ring and planes, PAYLOAD
  refused, FACE refused); BATCH N = 2 / 8 (layout, seed, unit count, fields;
  4a also beside ship_demo); PADDING; OVERFLOW (append, list, staging, a
  staged list walked); CHAINED 4a + 4c; SHIP 500 ticks. Every gate broken once
  (the patch's report).

**As built (P7.0a, 2026-11-15).** P7.0 was cut in two side by side: P7.0a
(this: items 1 and 4, the kit and slot 9d) and P7.0b (the table module with
the charge pilot, slot 2's split). Decisions 5 and 6 landed as written; one
line each:
- THE DEVICE DRAWS are `cpp/src/cuda_unit_draws.cuh` (`unit_draws_dev`):
  `draw_key(keys, e)` reads env e's row of an `(N, 3)` int64 input (seed lo,
  seed hi, tick word: `unit_draws.draw_key(sim)`, i.e. `sim.draws.key` and
  `sim.draws.tick & 0xFFFFFFFF`, the WEGO base included, captured at the
  slot's entry; a later slot stages it like any table input); `draw_at(key,
  salt, ident, index)`; `DrawSite` + `draw_next(key, site)` (the count in the
  thread); `cone_offset_q16(word, cone_q)`, `roll_q16(word, p_q)`. Out of
  the Python's domain (id, index, a key word, a cone past 2^31 - 1, where
  the 64-bit product would overflow) is `bad`, the caller's fault. The salts
  are `unit_draws::SALT_*` in the generated `cpp/src/unit_draws_layout.h`
  (`src/simulation/unit_draws.py`, a sixth header of
  `tools/gen_unit_layout.py`, fingerprint `bp.unit_draws_layout()`), which
  also records `NEXT_SITES` (SHOT_CONE `fire_burst`, BEAM_CONE `fire_beam`,
  MELEE_CRIT `melee_blow`, all `combat.py`); an AST scan of `src/` fails
  `tests/test_unit_draws.py` on a second site.
- THE DEVICE SIGHT is `cpp/src/cuda_unit_sight.cuh`: `unit_sight::has_los(
  solid, screen, h, w, fy1, fx1, fy2, fx2, &bad)`, `screen == nullptr` the
  physical form; one env's `(h, w)` bool planes, the caller rebases. An end
  past +-2^30 is `bad`.
- THE FALLOFFS are `FP_HD` `falloff_floor_q` / `falloff_round_q` in
  `fixed_point.h` beside the P6.2 kit twins (a 128-bit numerator, a 128 / 64
  restoring ceil division, the uint64 ceil root), bound as
  `bp.fp_falloff_*`; proved for v < 2^31 and 0 <= d2 < r2 < 2^63 (the
  callers: v = 65536 with Q32 distances for the deposit weights, a blast's
  Q16 damage), -1 outside. The P7.0 row's "squared-distance tests" were left
  out on the lead's brief (no distance helper until a caller needs one).
- SLOT 9d is `cpp/src/cuda_unit_ignition.cu` (`unit_kernels.ignition(store,
  fields)`, the nine `IGNITION_FIELDS` planes, `fire` and `ignition_armed`
  in place, coverage = cells changed): one thread per cell, a non-flammable
  cell returns first. Six table rows (T 732): `ign_temp_q` per material
  (`materials.ignition_temp_q16`), `ign_x_ext_q` / `ign_seed_q` (through
  `combat.ignition_params`, now the conductor's one read of the two dials),
  `ign_x_n_floor_q` (`combat.IGNITION_X_N_FLOOR`, the literal named),
  `ign_gas_o2` / `ign_gas_n2` (the map's gas table). The store is taken for
  the tables and the counts only; the kernel reads no row.
- THE HARNESS: the chained form's `_is_for` also recognises a callable
  handed the sim's GameMap (9d's only sim argument); without it the host law
  ran beside the device's, unseen because 9d is idempotent on its own
  output, so the chained gate now counts the host law's calls.
- WHAT THE REFERENCE SHOWED (nothing changed in it): 9d reads NO pressure
  factor -- `apply_temperature_ignition` gates on the mole fraction alone,
  while the fire's sustain multiplies `o2_pressure_factor.h`'s g(p) -- so its
  twin does not either (the brief asked for the factor; porting it would
  change the game); the fraction's floor was a literal in the law (named
  now, value unchanged); ship_demo never ignites (0 cells in 300 ticks:
  nothing gets hot); the ignition scene's no-O2 cases all come through the
  fraction's floor (a pumped pocket): vitiated air at full density is not
  built.
- THE GATES (`tests/cuda_unit_kit_check.py`, wrapper
  `tests/test_cuda_unit_kit.py`; CPU `tests/test_unit_draws.py`,
  `tests/test_unit_kit_twins.py`): TABLES; DRAWS (4 envs: seed 0, 2^32 + k,
  2^64 - 1, a tick past 2^32, a WEGO base; 4096 rows each over every salt
  and raw words, the range ends; two .next sites per row; batch == alone;
  7 domain flags); SIGHT (ship_demo, playground, my_ship, test_level with
  grilles stamped through `on_tile_changed`, one batch padded to the largest
  map; 48 000 pairs x 2 forms); FALLOFF (60 000 triples, device and CPU);
  LAW (9d's kernel on 320 random synthetic map pairs against the law on a
  stand-in map: four deliberate breaks -- the threshold's equality, a
  vacuum neighbour counted, the floor, the fraction's scale -- stayed green
  on every scene, which never puts a tile exactly at its threshold, gas in
  a vacuum tile or a fraction a count from X_ext); SCENES (the ignition scene, space and planetside, an 11-branch census;
  ship_demo 300 ticks with none); BATCH N = 2 / 8; CHAINED 9c + 9c2 + 9c3 +
  9d. Every gate broken once (the patch's report).

**As built (P7.0b, 2026-11-15).** Items 2 and 3 of the P7.0 row (the table
module with the charge table as its pilot; the reference's slot 2 split); the
kit and slot 9d are P7.0a's. No decision moved. One line each:
- THE TABLE MODULE is `src/simulation/object_tables.py`; every table fact is a
  schema in `unit_schema.OBJECT_TABLES` (`ObjectTable(name, columns, capacity,
  live, build)`, columns `TableColumn(name, encode, width, decode)`, the old
  `ChargeColumn` its alias; `CHARGE_TABLE` the pilot, `CHARGE_SCHEMA` unchanged
  in content so `charge_record` and the digest are byte-identical). A new table
  is a schema and a row of that tuple, then `tools/gen_unit_layout.py --write`
  (the sixth header, `cpp/src/object_tables_layout.h`: `OBJ_<T>_W / _CAP /
  _LIVE / _COL_<C>`, one fingerprint over every table).
- THE BRIDGE: `records(table, objs)` (the one record function), `pack(table,
  objs) -> (rows (cap, W), count)` (refuses past the capacity naming the table,
  and the env in `pack_batch`), `unpack(table, rows, count) -> objects` (the
  schema's decode and constructor; rows past the count never read); the host
  reference of the two list operations `append_rows` / `compact_rows`.
- THE DEVICE BUFFER: `DeviceTable(table, n_env, n_units=1, row_cap=1)` over
  `breach_physics.ObjectTableBuffer`, C++ `ObjectTable : UnitRecordList`
  (`cpp/src/cuda_object_table.{h,cu}`): the rows ARE the record list's
  records, the count its count, the overflow its dropped count, so a serial
  producer appends through `emit_record` and a per-row kernel through the
  list's staging and its one gather, and pybind registers it as a
  `UnitRecordListBuffer`. Its own operations: `upload(rows, counts)`,
  `append(recs, n)` (host-staged, one thread per env, then the overflow check:
  `ObjectTableOverflow` naming the table, the env and the rows lost),
  `compact()` (one thread per env, stable, the freed rows zeroed), `download()`.
- CHOICES where this section was silent: the table's capacity is STORAGE
  (`CHARGE_CAPACITY` rows); `charges.place`'s refusal at that many LIVE charges
  stays the law's own; the compaction is serial per env (W 8, cap 32: a scan
  is cheaper than a prefix sum); the live column is a schema fact (the charge
  table's own `live`), the grenades will declare theirs as an encoded column.
- SLOT 2 is two conductor calls, `Simulation._update_grenades` (label "2
  grenades") then `Simulation._update_rounds` (slot 2', label "2' rounds"),
  each one call line with its ordering comment; the harness wraps slots
  through the same `SLOT_WRAPS`. Sole cause EQUAL on all four
  `tests/_p3a_sole_cause.py` scenarios (ship_demo 2570 ticks, bots_v1 7200,
  the box 231, the scripted arena 420 with every grenade kind and a held
  trigger), every unit record and field digest identical.
- WHAT THE REFERENCE SHOWED (nothing changed): the profiler label "4c resolve"
  still also times the rounds' landing resolve (now called from
  `_update_rounds`; the harness's `select` by caller is unaffected); the
  planted charges are in no digest-visible order other than list order, and
  the list holds dead charges only between `charges.fire` and the prune inside
  the same `charges.tick` call on the continuous path (`detonate_owned`, WEGO
  only, can leave them to the next tick).
- THE GATES (`tests/cuda_object_table_check.py`, wrapper
  `tests/test_cuda_object_table.py`; CPU `tests/test_object_tables.py`,
  `tests/test_slot2_split.py`): TABLES (fingerprint, a mismatched module
  refused); TRIP (random rows bit for bit); APPEND (three ordered appends, an
  overflow refused and counted, nothing written); COMPACT (eight patterns);
  SCENE (P6.4a's verb scene variants 0 and 3 and a charge scene, N = 4, 160
  ticks: identity, round trip, the tick's append, the compaction, env i alone;
  an eleven-branch census). Every gate broken once (the patch's report).

**As built (P7b, 2026-11-15).** The cut table's P7b row; no decision moved.
One line each:
- THE GRENADE TABLE is `unit_schema.GRENADE_TABLE` (capacity 64, storage):
  `proj_type, owner_id, start_x_q, start_y_q, target_x_q, target_y_q,
  pos_x_q, pos_y_q, fuse_seconds` (an exact Q16, refused otherwise),
  `fuse_ticks` (`NO_TIMER` for the WEGO order path's None; a real fuse equal
  to the sentinel refused), `thrown_tick, ammo_name` (the ammo table's row),
  `travel_speed_q`, and the live column `detonated` ENCODED (1 while it
  flies). `build` is the `Projectile` constructor plus the position and the
  flag, refusing a row whose speed is not its round's.
- SIX TABLE ROWS (T 756): `clock_tps` (CFG, as `update_position` reads it),
  `grenade_flight_eps_q`, `order_grenade`, `hand_grenade_ammo_row` and
  `hand_grenade_speed_q` per throw kind, `ammo_payload_row` per ammo row
  (`payload_for_ammo`'s index, -1 none).
- THE KERNELS, `cpp/src/cuda_unit_objects.cu`, faces in `unit_kernels`:
  `table_appends(store, requests, grenades, charges)` (after `verbs`: one
  thread per env walks the request list; a THROW appends the
  `_release_grenade` Projectile, an ACCEPTED plant the `charges.place`
  charge); `grenades(store, table, effects, ticks)` (the flight one thread per
  row; the fuse-outs one thread per env in row order, each lob's detonation a
  PAYLOAD record `(target tile, ammo_payload_row[ammo], owner, the start tile
  as the approach, has_approach 1, kind grenade)`); `charges_tick(store,
  table, effects, ticks)` (serial in row order, PAYLOAD `(y, x, payload,
  owner, approach or -1, has_approach, kind door_explosive)`, then the
  compaction); `prune_grenades(table)` (slot 10: the compaction). Counts per
  env: changed, faults, refused (a WEGO grenade), dropped (effect records).
  The device door `refuse_wego(sims)` refuses a ruleset that `drives_units`
  (`detonate_owned`) and a grenade with no integer fuse, by name.
- CHOICES where this section was silent: the throw / plant records STAY in
  the request list as the transport (the plant's capacity pass and the
  shadow read it) and the append is a per-env pass after it, not a second
  staging; the fuse-outs are flagged by the wide kernel in a scratch and
  recorded by a serial pass (the effect list's row staging is unit rows);
  the charge kernel compacts inside the same call (`charges.tick`'s prune).
- THE HARNESS: `LISTS` rows `grenades` / `charges` (an object table over the
  host list: staged, compared, LANDED back in place by the chained form,
  never consumed) and `payloads` (`execute_payload` at its two P7b call sites
  through a tap; `ListDef.mark` empties the log at the slot's entry; the
  chained form replays each device record through the host's call, which
  must reproduce it); output lists kept PER NAME (one slot may output
  several; until now the last overwrote the others); `DeviceSlot(skip_ok=
  True)` for a slot the conductor gates off (`charges.tick` runs only while a
  charge lies).
- WHAT P7c.1 INHERITS: the PAYLOAD record of both call sites as above; the
  host-only writes the shadow saw inside slot 2 and 9e(d') (see the patch
  report), every one the payload law's; the blast's own events (UnitHit /
  UnitKilled, the flash's blindings) in those slots, which no ring compares
  yet; the noise, claims and blast log of each call.
- THE GATES (`tests/cuda_unit_objects_check.py`, wrapper
  `tests/test_cuda_unit_objects.py`; CPU `tests/test_grenade_table.py`):
  TABLES; FLIGHT (synthetic full tables against `_update_grenades`, its
  `execute_payload` recorded); SCENE (the grenade scene: every kind, six
  fuse-outs of different kinds on one tick, a release on a fuse-out tick, a
  zero flight, a 0-tick fuse, two charges firing on one tick, a full charge
  table refusing a plant; 1b + appends, 2, 9e(d'), 10 shadowed; a census);
  BATCH N = 2 / 8; PADDING; OVERFLOW (each table and the effect list);
  REFUSE; CHAINED (1b + 1b' + 1c + 2, and 9e(d')); SHIP 500 ticks.
**As built (P7a.2, 2026-11-15).** Slot 4's shooting on the continuous path,
slot 2''s rounds, THE MARCH they share, the table of rounds and the noise list.
Decisions 1, 3, 4, 5 and 8 landed as cut; the beam was NOT built (below). One
line each:
- THE MARCH is `cpp/src/cuda_unit_march.cuh`: `unit_march::march(const
  MarchMap& m, int64_t* rd, RecordSink& fx, bool* bad) -> still_flying`, `m`
  the env's rows, tables, `solid` / `material` planes, draw key and tick, `rd`
  ONE round-table row read and written in place, `fx` the sink its records go
  to (CHEW; or HIT then PUSH; then a payload round's PAYLOAD, kind `shell`).
  Beside it `round_draw` (`draw_at(salt, shooter id, round_index(tick -
  spawn_tick, ordinal))`), `arc_multiplier`, `list_env_sink`. The first hit is
  the first valid row in row order holding the tile, the shooter excluded by
  id, the dead skipped. Proved: positions within 2^40, steps within 2^24, a
  speed in (0, 2^30], an age / ordinal within 16 bits; outside, a fault.
- THE TABLE OF ROUNDS is `unit_schema.ROUND_TABLE` (W 21, capacity 128):
  `shooter_id`, `weapon`, `ammo` (schema rows), `payload` (the payload table's
  row, -1 none), `angle_q`, `rx_q`, `ry_q`, `prev_ix`, `prev_iy`,
  `remaining_steps`, `speed_q16`, `budget_q16`, `ox_q`, `oy_q`, `sx_q`, `sy_q`,
  `spawn_tick`, `ordinal`, `march_cover` / `march_facing_defense` (always 0:
  the encoder refuses a WEGO round by name), `live` (1 on the host).
  `unit_kernels.rounds_of` is its record function (behind `shooting_door`),
  `rounds_to_host` its way back (the shooter bound by id). The schema's
  `TableColumn` gained `get` (how a value is read when it is no attribute).
- THE NOISE LIST is `unit_schema.NOISE_TABLE` (W 4: `fy`, `fx`, `radius`,
  `live`; capacity 64) on the same machinery, `unit_kernels.noise_of`; it is
  staged from the host's log at the slot's entry and appended to (decision 4:
  it is drained at slot 5, P7d's stage).
- SLOT 4 is `cpp/src/cuda_unit_shooting.cu`'s `unit_shooting`
  (`unit_kernels.shooting(store, effects, rounds, noise, ticks, keys,
  fields)`): one thread per row; the gates (alive, not a zombie, can act, a
  weapon, the trigger, the cadence, `mag_settle`); melee / lobbed / placed
  rows end there; a spray arms its burst on the row and makes its noise; a
  projectile makes its noise and fires its burst (per round the SHOT_CONE
  `.next` draw, the kit step, the recoil PUSH, THE MARCH, the round appended
  when it flies); `mag_spend` and `last_fire_tick` at once. Effects, rounds
  and noises each staged per row and gathered per env in row order onto what
  the list holds. A hitscan trigger is REFUSED (`unit_kernels.ShotRefused`).
- SLOT 2' is `unit_rounds` (`unit_kernels.rounds_march(..., compact=True)`):
  ONE thread per env marches the table in order, its records appended to the
  effect list directly; finished rounds compacted. The LANDING is the
  applier's RESOLVE stage (`unit_kernels.resolve`): the same law and landing
  as 4c, so no new stage; `applier.STAGES` says so.
- THE REFERENCE, touched as a pure refactor (pins and golden unchanged):
  `Simulation._update_rounds` is now `_march_rounds(landing)` (phase A, called
  every tick, `self.bullets = survivors` set before the resolve, which reads no
  list) then `landing.resolve()`; the profiler label "2' rounds" moved to
  `_march_rounds` (the landing is timed under "4c resolve", as it already was).
- THE ROWS (32, T 1211): per weapons-table row `weapon_archetype` (codes
  `unit_tables.ARCHETYPE_CODES`, in the header and the fingerprint),
  `_shots_per_trigger`, `_range_tiles`, `_cone_aim_q` / `_cone_snap_q` (the
  class cones; SPREAD applied in the kernel), `_ammo`, `_crit_on`,
  `_crit_chance_q`, `_crit_mult_q`, `_loudness_tiles`, `_burst_ticks`; per ammo
  row `ammo_speed_q16`, `_damage`, `_dtype`, `_ap_q`, `_expose_q`,
  `_wall_live`, `_wall_damage_q`, `_payload`, `_push_q`, `_recoil_q`; scalars
  `impulse_twist_q16`, `zombie_bullet_mult_q`, `crit_front / flank /
  behind_mult_q`; the arc halves per profile and per species; per material
  `mat_cover_exposure_q`. Every per-call float door of the reference (the
  zombie multiplier, `crit_chance`, `crit_mult`, `wall_damage`, `ap`) is one
  row quantized as the law quantizes it; no product needed reporting.
- THE LIBRARY gained the magazine law (`mag_settle`, `reload_begin`), moved
  from the verbs TU, which now calls it. `applier.ROW_CAP` went 3 -> 32 (four
  records a round, eight pellets).
- THE HARNESS: the input `draw_key`; the lists `rounds` and `noise` (object
  tables; `ListDef.land` writes a device list back onto the host in the
  chained form); `DeviceSlot.name` (two slots on one label: 4c and 2''s
  landing); the caller test skips the harness's own wrappers; an output list's
  host records are kept per list (a slot with two output lists overwrote the
  first: fixed).
- THE BEAM was not built: it is not the march (no budget, no first-hit stop:
  every unit on the line is skewered once, a per-beam set of rows; the energy
  attenuated per tile by the trace-gas Beer-Lambert over the `gas` planes, a new
  plane input with `gases.beam_absorb_q16` rows and the trace currency's shift;
  HITs of cause laser, no push; the BEAM_CONE `.next` site). It is a second
  device function beside the march with its own scene; the shipped beams are
  `lance_3` and `lance_5`, which the device refuses by name.
- WHAT THE REFERENCE SHOWED (nothing changed in it): `combat` reads the
  PROCESS-WIDE weapon tables (`weapons.get_tables()`, rebuilt by every new
  `Simulation` at its own tile size) while each sim keeps its own
  `weapons_tables`: two sims of two tile sizes in one process shoot with the
  last one's ranges (a finding for Erik; the device reads each sim's own, so a
  batch must share a tile size); slot 1b drops a dead unit's trigger, so a
  corpse holding the trigger reaches slot 4 only when it died after 1b (a round
  landing at 2', a grenade at 2, a DoT at 2b); a payload round's PAYLOAD
  record cannot be applied until P7c.1, so the payload scene shadows the march
  slots and not the landing; ship_demo's bots pulled the trigger 6 times in
  500 ticks (6 hits, no kill, no round outliving its tick).
- THE GATES (`tests/cuda_unit_shooting_check.py`, wrapper
  `tests/test_cuda_unit_shooting.py`): TABLES (3 envs, two tile sizes); MARCH
  (24 x 16 envs x 12 synthetic rounds over random maps of every material, a
  23-branch census); SCENE (1/3 m tiles, 2' + landing + 4 + 4c shadowed, a
  46-branch census); SCENEB (payload rounds, a spray, a budget-starved bolt, 7
  branches); BATCH N = 2 / 8; PADDING; OVERFLOW (the effect list, a row's
  staging, the round table, the noise list); REFUSE (5 named refusals);
  CHAINED (2' + landing; 4 + 4a + 4c; the host law run 0 times); SHIP 500
  ticks. Every gate broken once (the patch's report).

**As built (P7a.3, 2026-11-16).** The cut table's P7a.3 row; decisions 1, 4
and 7(a) landed as written. One line each:
- SLOT 4b is `cpp/src/cuda_unit_sprays.cu` (`unit_kernels.sprays(store,
  effects, cells, {"solid": ...})`): ONE block per env, thread 0 walking the
  rows in id order (a dead or interrupted burst cleared; a live one's cone,
  then the countdown at once), the block's threads splitting each cone's box
  (`decide_box`), thread 0 emitting it in row-major order (`emit_box`). Counts:
  changed, faults, refused (a WEGO burst: no aim angle), dropped (records +
  cells).
- THE CONE is `cpp/src/cuda_unit_spray.cuh` (`cone_from` / `cone_params`,
  `cone_tile`, `box_size`): `spray_cone_tiles` integer for integer over the
  PHYSICAL `unit_sight::has_los`; proved for a range <= 2^14 and an apex
  within +-2^20. The launch's scratch is one box per env, `min((2 r_max +
  1)^2, h w)` (`unit_kernels.spray_box_max`). TEST-ONLY probe
  `unit_kernels.spray_cone_probe` (the same decide / emit pair).
- THE CELL BUFFER is a record list of `(y, x, div)` (`attack_lists.cell_buffer`,
  cap 4096 per env per tick, storage); SPRAY_CELLS `(owner, heat_q, gas_id,
  gas_q, start, count)` holds a span into it (`applier.record_of` recognises
  `_enqueue_spray_cells` and appends its cells: `records_and_cells_of`).
- SPRAY_CELLS AT THE RESOLVE is `unit_spray::apply_spray_cells`, a branch of
  the one `k_applier` (any stage; all operands checked before the first
  write). The walk's further reach is ONE optional argument, `ApplierLists` /
  `unit_kernels.WalkLists(cells, door, claims)`; a SPRAY_CELLS without them
  faults.
- THE DEVICE EDIT DOOR (step (a)) is `cpp/src/cuda_edit_door.{h,cuh,cu}` +
  `edit_door.DeviceEditDoor`: per env the EDIT rows (pack_for_device's 16
  columns, then `source_id` and `region`, the sort key step (c) needs), the
  TILE rows (edit index local to the env's list) and `seq` (reset by `begin`,
  set by `set_seq`). `door_tile_edit` is `_edit_door` + `_iter_region`'s TILE
  for an edit with `amount_q`, FLAT, no noise, no scale, the policy clamp (the
  tables' `edit_clamp_*` rows per field ordinal). Host reference
  `edit_door.door_rows`; chained landing `edit_door.replay` (each row through
  `EditQueue.enqueue` at its key, then re-packed and compared). E_ENV is 0, as
  in the pack.
- THE CLAIMS LIST is a record list of `(channel, y, x, attacker)`
  (`attack_lists.claims_list`, cap 8192), every claim in order (duplicates
  kept, the environment's dropped): `credit_at`'s table as it stands. Host
  side: `claim_rows`, `first_claims`, `replay_claims`.
- ELEVEN TABLE ROWS (T 772 -> 912): `spray_range_tiles`, `spray_cone_half_q`
  (`cone_q16_of_deg`), `spray_has_ammo`, `spray_heat_q`, `spray_gas_q`
  (`quantize_trace_scalar`), `spray_gas_id` (per map; a conservative slice
  refused at the read), `spray_src_heat`, `spray_src_gas`, `edit_clamp_flags`,
  `edit_clamp_lo`, `edit_clamp_hi`. `burst_ticks` is NOT a row: 4b never reads
  it (the burst's start, slot 4, is P7a.2's).
- THE RECORD LIST gained `upload` (moved up from the object tables, which
  inherit it) and `counts()` (counts and dropped without the records), and the
  unit library `list_env_sink` (a serial producer's sink; P7b's two private
  copies may fold onto it).
- THE HARNESS: `LISTS` rows `spray_cells` (the effect list's cells),
  `edits` (the queue's edits since the slot's mark, through `door_rows`, an
  edit of another source refused as an interleave; landed by `replay`),
  `claims` (every `TickCredit.claim` of the slot through a tap; landed by
  `replay_claims`); the scalar input `edit_seq` (the window's base: the
  device's keys are ABSOLUTE in the shadow and the chained run).
- CHOICES where this section was silent: 4b is one serial block per env with a
  block-wide box (not a per-row producer: its cells are variable-length and the
  record must follow them); the device door writes E_ENV 0 like the pack; the
  claims list keeps every claim (the reader's first-match is the reference's
  first-claim-wins, so the writer needs no lookup); the CELL / CLAIM / EDIT
  capacities are storage sized from the Dragon-9 heavy at 60 tiles.
- WHAT THE REFERENCE SHOWED (nothing changed in it): `_enqueue_spray_cells`
  reads no map state, so an earlier CHEW of the walk changes no edit or claim
  (gated: the scene's carbine destroys a glass tile a later spray's cells
  hold); `process_sprays` reads the process-wide `combat.weapon_tables()`
  while the rows read `sim.weapons_tables` (one object per match; two sims of
  different tile sizes in one process would see another sim's range -- an
  accepted gap, the batch gates keep one tile size); every shipped spray cone
  yields at least a wall face (zero cells unreachable on the shipped rows);
  ship_demo's bots never spray (0 deposits in 500 ticks).
- THE GATES (`tests/cuda_unit_sprays_check.py`, wrapper
  `tests/test_cuda_unit_sprays.py`; CPU `tests/test_spray_device_records.py`):
  TABLES; CONE (6000 synthetic cones over four maps, set, order and divisors);
  SCENE (every shipped spray, a census of 20 branches each required, three
  named unreachable; 4b + 4c shadowed: rows, records, cells, ring, planes,
  books, door rows, claims); BATCH N = 2 / 8; PADDING; OVERFLOW (cells, effect
  list, claims, door); REFUSE (WEGO burst, interleave, a span outside the
  buffer, a walk without its lists); SEQ (a window base of 1234: the keys are
  absolute); CHAINED 4b + 4c, and 4 + 4a + 4b + 4c (P7a.2's slot 4 arming the
  bursts; merged in at 97098c7a); SHIP 500 ticks. Every
  gate broken once (the patch's report).
**As built (P6b.1, 2026-11-16).** The cut table's P6b.1 row: the door table, seal /
unseal / can_seal and the gas-N primitives in the topology seam, the door sweep, the
use write. No decision moved. One line each:
- THE DOOR TABLE is `unit_schema.DOOR_TABLE` (W 7, capacity 32): `ordinal`, `alive`,
  `state`, `want_open`, `tile_start`, `tile_count`, `live` (always 1: a destroyed door
  stays a row, dead by `alive`; never compacted), row k the k-th door of
  `Simulation._doors` (ordinal order). Its span and per-tile hp are ONE side table,
  `DOOR_TILE_TABLE` (W 4, capacity 256: `ty`, `tx`, `hp`, `live`), the doors' spans
  concatenated in door order, each row-major -- one table, not two, because `hp[i]`
  belongs to `span[i]` (the S4 pin). Record functions `unit_kernels.doors_of` /
  `door_tiles_of`; `door_objects` unpacks to `unit_schema.DoorRow`; `doors_to_host`
  lands on the sim's OWN runtimes (refusing another count, ordinal or span:
  `DoorTableMismatch`); `door_entity_rows` reads a door's ENTITY_SECT_V1 rows (alive,
  state, want_open, hp_i) from the table, held equal to the serializer (P6b.4's export).
- THE OBJECT TABLES gained SIDE SPANS in the schema's terms: `unit_schema.SideSpan(name,
  side, start, count, items, max_count)` on `ObjectTable.spans`; `object_tables.records`
  fills `start` / `count` from the list position (`table_record(..., given=)`),
  `side_records` is the side table's record function, `pack_sides` packs both,
  `unpack(..., sides=)` hands `build` each object's side objects. A member past its
  bound is refused at the encode, a list past a capacity at the pack, by name.
- SEAL / UNSEAL / CAN_SEAL and THE GAS-N PRIMITIVES are `cpp/src/cuda_topology.cuh`'s
  `topo::seal_tiles`, `unseal_tiles`, `can_seal`, `inject_gas_n`, `extract_gas_n_vec`
  (its bulk sum is `extract_gas_n`), `transfer_gas_n`, `aperture_blocked`, over the
  shared seam rows (`energy_move` is `gas_energy_move` with the call's channel and its
  two `diag_*` twins; the counted trace saturation is the kit's `trace_add_sat`). A
  span is `(y, x)` pairs handed row-major (an unsorted one is a fault). THE REFUSALS are
  `applier.SEAM_REFUSALS` (generated `APPLIER_SEAM_*`): out_of_bounds, already_solid,
  unknown_material, material_not_solid, standing_water, sealed_pocket, overflow,
  duplicate_tile, not_solid, span_too_long (device only: past 256), fault; each writes
  nothing. Proved range (pre-checked on every cell a call touches): gas planes >= 0,
  |gas_energy| < 2^61, a donor's |T_abs| < 2^28, T_AMB < 2^28, a solid book in int64.
  The ripple pair is written as 0u, the seal's `permeability > 0.0` is the
  `seal_mat_ok` row: no float arithmetic.
- THE PLANES grew by eight (`wave_p`, `wind_x`, `wind_y`, `flow_vx`, `flow_vy`, the
  float `ripple` / `ripple_v`, the read-only `water_depth`); THE BOOKS by 20 (`ge_seal`,
  `ge_unseal`, `ge_pump_inject` / `_extract` / `_transfer`, their six `diag_*` twins,
  `solid_seal`, `solid_unseal`, `trace_clip_0..6` -- `applier.host_book` reads a LIST
  host by index, `trace_clip_books:g`); five table rows (T 1274): `door_mat_closed`,
  `seal_mat_ok` per material, `gas_trace` per slice, `inject_o2_frac_q` (the reference's
  0.21 literal named `GameMap.INJECT_O2_FRACTION`, a pure refactor), `door_states`.
- THE SWEEP is the applier's third stage, DOORS (`applier.STAGES`; `cpp/src/cuda_unit_
  doors.cuh`'s `doors::sweep`, called by `k_applier` on the door table and the tile
  table instead of the effect list): per door in row order, a dead door skipped; a
  CLOSED door with a tile no longer `MAT_DOOR_CLOSED` dies whole (alive 0, DESTROYED,
  hp 0) and its remaining closed tiles go through `topo::destroy_wall` in span order;
  CLOSED and `want_open`: the hp copied from `wall_hp`, `topo::unseal_tiles`, OPEN; OPEN
  and not `want_open`: no LIVING valid row's footprint square on the span (corpses
  never block) and `topo::can_seal`, then `topo::seal_tiles(span, MAT_DOOR_CLOSED)` and
  the hp restamped, CLOSED. Python face `unit_kernels.sweep_doors(store, applier,
  fields, doors, tiles) -> (transitions, books)`; `applier_run` takes the two tables
  for that stage. Its profiler label is its own, "9e(d) doors" (moved out of "9e
  entities").
- THE USE WRITE is a sibling of P7b's appends in the same form, `unit_kernels.door_uses`
  (`cpp/src/cuda_unit_doors.cu`, after `verbs`): one thread per env walks the request
  list and flips `want_open` on the row each USE names (two uses in a tick flip twice).
- CHOICES where the design was silent: the span and hp in one side table; the use
  write a sibling kernel, not a fourth argument of `table_appends` (its tables are not
  the door's); a span's bound is the tile table's capacity (the seal's close-T
  snapshot lives in a local array of that size); the unseal's donors' live T_abs is
  checked against the range before the first write only (a donor shared by two span
  tiles needs a bent span: no door has one); `_aperture_blocked` out of the grid is
  refused (numpy would wrap or raise); `inject_gas_n_vec` (the vents' deposit with its
  rails) not built: it arrives with its first device caller (P6b.2).
- ONE CLASS-T TWIN, `doors`, slots "1b intents" and "9e(d) doors": the corpus has no
  door entity, so the sweep alone would trace no file (the P7b precedent); its traced
  files are 1b's, `door_system.py` stays `[not_twinned]` and the doors gate is the guard.
- THE HARNESS: lists `doors` / `door_tiles` (the tile table lands first into a stash,
  the door table then lands both; a chain naming only the doors lands over the host's
  own tiles); the write-set gate covers a LIST attribute by its declared element books.
- WHAT THE REFERENCE SHOWED (nothing changed in it): `transfer_gas_n` drains its source
  and THEN raises on a destination past int32 (not atomic; the device refuses before
  any write); `seal_tiles` would raise mid-mutation on a negative trace count at a
  receiver (the device's pre-pass refuses it as a fault); a door's unseal never moves
  the `mint` channel (a solid tile holds no gas, so born-at-ambient books 0); a seal
  of a gas-free tile with no receiver is allowed (the pocket rule refuses only gas);
  playground's bots match ends near tick 33 (not investigated); ship_demo's doors are
  openings (no entity), so the sweep never runs in the corpus.
- THE GATES (`tests/cuda_unit_doors_check.py`, wrapper `tests/test_cuda_unit_doors.py`;
  CPU `tests/test_door_table.py`): TABLES; PROBE (1 613 ops on five shipped levels
  and both door scenes, every refusal taken, planes / books / closure equal on both
  sides); SCENE (the door scene, space and planetside, 1b + 9e(d) shadowed, a
  23-branch census, two named unreachable, the identity and entity rows every
  tick); BATCH N = 2 / 8 (door-scene variants: layout, door count, boundary, seed,
  schedule); LEVELS (playground's match, airlock_demo, test_level, door_test);
  PADDING; OVERFLOW; CHAINED (1b + use write and 9e(d), both variants, the host law run
  0 times in the twin); SHIP 500 ticks (the sweep gated off). Every gate broken once
  (the patch's report: 29 single-site breaks, all red).

**As built (P7c.1, 2026-11-16).** The cut table's P7c.1 row, the disc path of
`execute_payload`; decisions 1, 2, 4, 7(b) and 8 landed as written. One line
each:
- THE PAYLOAD FUNCTION is `cpp/src/cuda_unit_payload.cuh`
  (`unit_payload::apply_payload(rec, TopoEnv, tab, e, PayloadReach, books)`),
  the PAYLOAD branch of the one `k_applier`, thread 0, in the reference's
  sub-step order: `apply_explosion` (the atmosphere DISC, the smoke-clear
  DISC, the disc d2 <= r^2 walked serially in row-major order -- a blast-wall
  material's hp less `falloff_round(wall_q, d2, r^2)`, `topo::destroy_wall`
  at once at <= 0, then the 3x3 wave stencil's TILE edits on the planes as
  they stand -- and the blast log's record), the noise, `apply_blast_damage`
  (the IMMEDIATE `apply_packet` per valid living row in row order),
  `apply_flash_blind` (`unit_sight::has_los`'s sight form, `isqrt_u64_q`,
  `atan2_q16`, never shortened, `apply_status`), `emit_gas`, `ignite_ring`,
  `deposit_heat` (claims, then DISC edits). ExplosionEvent has no device form.
  Results APPLIED / FAULT / REFUSED into the `applied` / `faults` / `refused`
  books.
- WHICH ROWS: a row is applied when `applier.payload_refusal` is 0 (no
  strength, no bulk gas, no explosive, no expansion) -- shipped:
  `studio_smoke`, `smoke_screen`, `tear_burst`, `poison_cloud`,
  `incendiary_splash`, `studio_ignite`, `flashbang`, `hazmat_release`,
  `plasma_splash_small`, `plasma_splash_large` (10). Refused by name, nothing
  written (`applier.PAYLOAD_REFUSALS`): strength (`frag_standard`,
  `breach_focus`, `demolition_c4`, `magazine_cookoff`, `engine_death`), bulk
  gas (`o2_store_rupture`, `halon_dump`), explosive -- the air read -- (the 19
  studio / comparison rows `frag_heat`, `frag_hot_mass`, `quad_*`, `div_*`,
  `quad_src_*`, `studio_small` / `_medium` / `_large` / `_hull`). The three
  call sites can deliver only the first ten and the strength rows.
- THE ROWS (33, T 2206): per payload row `payload_refuse`, `_radius`
  (`int(radius)`, refused unless an integer), `_radius_q`, `_atm_q` (the bulk
  currency), `_press_q`, `_wall_q`, `_inner_q` (2/5 of the radius on the Q16
  grid), `_blast_live`, `_unit_damage`, `_loudness_tiles`, `_blind_radius_q`,
  `_blind_ticks`, `_blind_away_q`, `_gas_id` (a conservative slice refused at
  the read), `_gas_q`, `_gas_radius_q`, `_ignite_live`, `_ignite_radius_q`,
  `_ignite_q`, `_heat_live`, `_heat_q`, `_heat_radius_q`; scalars the four
  source ids, `payload_smoke_slice`, `payload_smoke_clear_q`,
  `payload_ignite_heat_q`, `payload_ignite_clamp` (flags, lo, hi),
  `blast_damage_threshold` (the ceiling of the CFG value), `flash_front_q16`;
  per material `blast_wall_mat`. No product needed reporting: on the disc path
  every float the law converts per call is a row; the soot and `heat_mj`
  products belong to the explosive / strength rows (P7c.2).
- THE DEVICE EDIT DOOR, step (b): `edit_door::door_disc_edit` (a DISC, FLAT
  or LINEAR, the clamp given) and `disc_claims` (`_claim_disc`) in
  `cpp/src/cuda_edit_door.cuh`; the EDIT row grew four REGION ARGUMENTS
  (`edit_door.ARG_COL`: a TILE's tile, a DISC's centre, Q16 radius and
  falloff), so the chained landing (`edit_door.edit_of` / `replay`) rebuilds a
  clipped disc it could not from its TILE rows; `door_rows` writes them.
- THE BLAST LOG is `unit_schema.BLAST_LOG_TABLE` (W 5, capacity 64) on the
  object tables, `unit_kernels.blast_log_of`; THE NOISE LIST gained its second
  producer. Both reach the walk as `WalkLists(noise=, blast_log=)` / C++
  `ApplierLists`; a PAYLOAD without its lists or the ring FAULTS.
- WHERE IT IS WALKED: the RESOLVE stage (slot 4c and slot 2''s landing: the
  payload round's stop at its place in the record order) and a new stage
  DETONATE (`applier.STAGE_DETONATE`, no landing): `unit_kernels.detonate`
  right after `grenades` (slot 2) and `charges_tick` (slot 9e(d')). Grenade
  flight reads no map, so all flights then every detonation in row order is
  the reference's interleaving.
- THE CORPSE SKIP bites for the first time: in a resolve a blast kills a unit
  whose exposure raise is pending, and the landing skips the raise (gated: the
  scene's set V, K's carbine and P's plasma on one victim; its break is red).
- CHOICES where this section was silent: the explosive rows are refused (the
  brief's "the air read" is P7c.2's), which leaves `add_explosion_smoke`
  unbuilt: every row that reaches it carries an explosive (the row door
  requires one for `emit_blast_smoke`) and its scale IS the air read; the
  fan-out stays P7c.2's (the disc path is serial on thread 0); a fault after
  the first write stops the effect as CHEW's does (the record's operands and
  every disc wall's new hp are checked first).
- THE HARNESS: `LISTS` rows `blast_log` and `detonations` (the payload calls
  consumed, never replayed, so the twin's host `execute_payload` runs 0 times;
  a refused row replayed alone only when `REPLAY_REFUSED`); the edits list
  became one record per EDIT row then one per TILE row (a DISC has many);
  the door's owned sources include the payload's; FIXED: the claims tap
  consumed a generator of tiles (`_claim_disc`) so the HOOKED host claimed
  nothing -- a hooked match differed from an unhooked one (the spray's lists
  hid it).
- WHAT THE REFERENCE SHOWED (nothing changed in it): a payload round carries
  the PROCESS table's PayloadDef (`combat.weapon_tables()`) while
  `applier.payload_index` names the sim's own row by identity, so sims built
  one after another fail the record function unless they share one table (the
  gates share it; one tile size); no shipped row the call sites deliver has
  `unit_damage > 0` on the disc path (the frags are strength rows), so blast
  damage is reached only through the gates' raised rows; no planted-charge
  ammo carries a disc-path row (both are strength rows); `blast_damage_threshold`
  is read from CFG per call (now a row read once per table build); ship_demo's
  bots throw one frag (refused) in 500 ticks, so the shadow stops by name at
  that tick (456) and the chained ship replays it on the host.
- THE GATES (`tests/cuda_unit_payloads_check.py`, wrapper
  `tests/test_cuda_unit_payloads.py`; CPU `tests/test_payload_device_rows.py`):
  TABLES (1 m and 1/3 m); PROBE (120 synthetic lists on random maps, DETONATE
  and RESOLVE, shipped then raised rows, a 32-branch census); DISC (every DISC
  form at 8 radii and 4 centres); SCENE (five row sets: all 40 (call site,
  shipped row) pairs, slots 2, 2''s landing, 4c and 9e(d') shadowed: rows,
  ring, planes, books, door rows, claims, noise, blast log, payload records;
  a 10-branch census); BATCH N = 2 / 8; PADDING; OVERFLOW (edits, tiles,
  claims, noise, blast log, ring); REFUSE (26 rows by name, nothing written; a
  walk without its door faults); CHAINED (2; 2' + landing; 4 + 4a + 4b + 4c;
  9e(d'): the host `execute_payload` run 0 times in the chained slot); SHIP 500
  ticks. Every gate broken once (the patch's report).
**As built (P6b.2, 2026-11-16).** The cut table's P6b.2 row: the duct and vent tables,
`inject_gas_n_vec` in the topology seam, the ledgers in int64 with a named refusal, the
sweep. No decision moved. One line each:
- THE DUCT TABLE is `unit_schema.DUCT_TABLE` (W 20, capacity 16): `ordinal`, `filter` (the
  `[filters.*]` row's index in `unit_schema.filter_names()`, config's rows sorted by name),
  `o2_raw`, `n2_raw`, `trace_raw[5]`, `e_plenum`, `sink[5]`, `e_wipe`, `rail_lo_hits`,
  `rail_hi_hits`, `alive`, `live` (1). THE VENT TABLE is `VENT_TABLE` (W 11, capacity 64):
  `ordinal`, `duct_ordinal` (the duct's ENTITY ORDINAL, -1 unwired or dangling), `role` (0
  supply / 1 return: the law's `role == "return"`), `aperture_y` / `_x`, the load constants
  `rate_raw_per_s` and `tps`, `accum`, `guard_skips`, `alive`, `live`. Record functions
  `unit_kernels.ducts_of` / `vents_of`; `duct_objects` / `vent_objects` unpack to
  `unit_schema.DuctRow` / `VentRow`; `ducts_to_host` / `vents_to_host` land on the sim's OWN
  runtimes (another count, ordinal, filter, duct, role or constant: `VentTableMismatch`);
  `duct_entity_rows` / `vent_entity_rows` read ENTITY_SECT_V1's rows from a table row, held
  equal to the serializer (P6b.4's export).
- FIVE TABLE ROWS (T 1432): `vent_filter_eff_q` (per filter row x trace slot, through the
  reference's own `_duct_filter_efficiency_q`), `vent_trace_ids` (`_TRACE_IDS`),
  `vent_t_min_q` / `vent_t_max_q` (`_t_rails_q()`, the sim's cached pair when it has one),
  `vent_n_eps` (`N_EPS`). THREE BOOKS: `ge_pump_inject_vec`, `ge_pump_rail` (the deposit's two
  seam channels) and `ledger_refused`.
- `topo::inject_gas_n_vec(E, tab, y, x, add[], t_dep, t_min, t_max, &t_new, &rail, B,
  InjectVecBooks)` is `GameMap.inject_gas_n_vec` integer for integer (the core's overflow
  refusal before any write, the conversion seam on an accountable tile with Python's floor,
  the rails and the rail's write-back, the T-mix on a non-accountable one, the bulk-free and
  zero-vector returns); the caller's book pair (the vents': `vent_inject_books()`). Its
  proved range: every slice in [0, 2^54), the cell's gas >= 0, |E| < 2^61, |t_dep|, |t_min|,
  |t_max| < 2^30 and on a non-accountable tile |T_old| < 2^30 (a fault past it, nothing
  written).
- THE KIT gained `muldiv_floor_q(a, b, c)` (`(a * b) // c` for a, b >= 0, c > 0: a 128-bit
  product, a 128 / 64 restoring floor division `udiv128_floor_q`; -1 past a 2^63 quotient),
  bound with `floordiv_q` as `bp.fp_muldiv_floor` / `bp.fp_floordiv`.
- THE PROVED RANGE (the ledgers are unbounded ints in Python, packed as int64 only by the
  serializer). A tile's bulk is < 2^32 (two int32 planes, >= 0), a trace slice < 2^31, and
  the extraction's own range keeps a tile's |T_abs| < 2^28, so its relative T < 2^29. ONE
  return of quantum q therefore moves the bulk pair by at most min(q, 2^32), a trace slot and
  its sink by < 2^31 each, and e_plenum by at most min(q, 2^32) * 2^29 in magnitude; a supply
  only drains (the floor split and floor subadditivity never hand out more than the plenum
  holds), adds 1 to a rail count, and leaves |e| <= max(|e|, N_bulk) (with q_dep = floor(e /
  N): e - D q_dep lies in [0, e] for e >= 0 and in [e, D] for e < 0, D <= N the bulk
  handed out); the wipe moves e into e_wipe once. With B_N the sum over a duct's returns of
  min(q_r, 2^32), B_E = B_N * 2^29, R / S its return / supply counts, a duct whose ledgers
  hold AT THE HEAD OF ITS SWEEP 0 <= o2_raw, n2_raw <= 2^61 - B_N; 0 <= trace_raw[i] <=
  2^54 - R 2^31; 0 <= sink[i] <= 2^62 - R 2^31; |e_plenum| <= 2^61 - B_E; |e_wipe| <=
  2^62; 0 <= rail hits <= 2^62 - S; 0 <= each vent's guard_skips <= 2^62 - 1 keeps every
  value of the tick in int64 (N_bulk <= 2^62, the wipe <= 2^62 + 2^61, the debit sum <= |e| +
  N_bulk < 2^63, the three floor products exact through `muldiv_floor_q` with quotients <=
  their first operand, a deposit's trace <= 2^54 inside `inject_gas_n_vec`'s range). The
  head refuses a duct outside it BY NAME (`ledger_refused`; `unit_kernels.sweep_vents`
  raises `applier.LedgerRangeRefused` naming the env, the duct's ordinal, the ledger, its
  value and its bounds, through `vent_ledger_range`, the host twin) before any write of that
  duct: its vents, its ledgers, the map. The vent rows' constants are a FAULT outside tps in
  [1, 2^16], accum in [0, tps), the rate in [0, 2^39] (a quantum < 2^40: the extraction's
  range), a role 0 / 1, the aperture in the grid. Separately the deposit FAULTS on a t_dep
  past 2^30 (about 16 400 K above ambient): a plenum's mean temperature passes it only by a
  fixture (the extraction refuses tiles past 2^28 absolute, and the floor's upward drift is at
  most N / 64 counts a draining tick).
- THE SWEEP is the applier's stage VENTS (2; at the merge with P7c.1 DOORS became 3 and
  DETONATE 4: the 9e(d) numbers follow the conductor's pumps -> vents -> doors), `cpp/src/cuda_unit_vents.cuh`'s `vents::sweep` called
  by `k_applier` on the two tables (`ApplierLists.duct_table` / `vent_table`, `WalkTables`):
  per duct in row order the head check, the returns (dead skipped; `topo::aperture_blocked` a
  counted guard skip with the accumulator frozen; the accrual; `topo::extract_gas_n_vec` with
  T read before it; the bulk pair, the filter's truncating scrub per trace slot, the bulk-only
  energy credit), the supplies' weights, the wipe below N_EPS, the distribution
  (`floordiv_q` for t_dep, `muldiv_floor_q` for the share, the split over the REMAINING pair
  with its O2-first cascade and the trace against the ORIGINAL holdings), the debit, each
  deposit through `topo::inject_gas_n_vec`, the rail counts. Python face
  `unit_kernels.sweep_vents(store, applier, fields, ducts, vents) -> (coverage, books)`
  (coverage: extractions that took gas plus deposits that landed). Profiler label "9e(d)
  vents" (moved out of "9e entities").
- CHOICES where this section was silent: two independent tables, not a side table (the vents
  are the sim's own ordinal list, unwired ones included; the sweep scans the vent table per
  duct, cap 64); a vent's duct by ordinal (an entity table's key), not by row; the filter
  efficiencies a device tables row with an index in the duct row (not five duct columns);
  the range checked at the duct's HEAD from worst-case per-tick steps (so "nothing written"
  holds for the whole duct), not per update; a seam fault inside a duct abandons it (the run
  refused) as the doors do.
- ONE CLASS-T TWIN, `vents`, slot "9e(d) vents": ship_demo's duct (1 duct, 2 vents) runs in
  the corpus, so the twin traces `vent_system.py` (it left `[not_twinned]`).
- WHAT THE REFERENCE SHOWED (nothing changed in it): with N_EPS = 64 "every share floored to
  0" needs more than 64 ready supplies on one duct (unreachable on the device's capacity);
  an airless cell's trace is wiped by the tick's transport before 9e (a vent depositing smoke
  into an empty pocket lands it on a clean cell); `GameMap.smoke` is a view of `gas[SMOKE]`,
  which the doors gate's write-set list had not met (the vent scene seals a smoky tile: its
  HOST_ONLY row added); the guard reads the water depth, so a fixture flood blocks a vent;
  ship_demo, playground and fire_smoke_studio move gas through their ducts on every tick
  (airlock_demo has no duct).
- THE GATES (`tests/cuda_unit_vents_check.py`, wrapper `tests/test_cuda_unit_vents.py`; CPU
  `tests/test_vent_table.py`, `tests/test_unit_kit_twins.py`): TABLES; KIT (3 072 floor
  divisions, 3 006 floor products); PROBE (522 `inject_gas_n_vec` ops on six maps, both
  rails, both floors of both branches, the clip, every refusal); SCENE (the vent scene: six
  ducts, sixteen wired vents and two unwired, a door over a vent, two ducts' vents on one
  tile; 1b + 9e(d) vents + 9e(d) doors shadowed 130 ticks, both boundaries; a 31-branch
  census, three named unreachable; the ledgers from 0 to 10^13); LEVELS (ship_demo's bots,
  playground, fire_smoke_studio, airlock_demo); REFUSAL (12 ledger cases one past their
  range refused by name, nothing of the duct written; at the bound equal to the host); BATCH
  N = 2 / 8; PADDING; OVERFLOW; CHAINED (vents + doors on the vent scene, both boundaries,
  and on the door scene; ship_demo's bots 500 ticks; the host laws run 0 times in the twin).
  Every gate broken once (the patch's report).

**As built (P7c.2, 2026-11-16).** The cut table's P7c.2 row: every payload row on the device.
No decision moved. One line each:
- THE PAYLOAD FUNCTION applies EVERY row: `applier.PAYLOAD_REFUSALS` is empty (the
  mechanism kept for a future row; the gate asserts "every row applied" over the table).
  After `apply_flash_blind` and before any deposit, a strength row walks its heat tiles
  (`reach_tiles` over `heat_r2`, the approach seed rule), reads the air over them, walks its
  source tiles and adds its one-tick source; a row with an explosive reads the air over
  `disc_tiles`; a comparison row walks `expansion_r2` (no approach), adds its first tick and
  appends its later ticks to the blast-source table. Then the blast smoke (a strength row's
  TILES deposit; else `add_explosion_smoke` with `blast_smoke_peak_q`'s peak, noisy, scaled
  by the disc read), `emit_gas`, bulk gas, a strength row's claims + TILES heat (else
  `ignite_ring`), `deposit_heat` (times the heat factor when the row read the air).
- THE BLAST WALK is `cpp/src/cuda_unit_blast.cuh` (`unit_blast::reach`, `walk_seeds`,
  `air_read`, `air_smoke_q`, `disc_tiles`, `add_source`, `take_sources`): the walk marks a
  per-env scratch (`FieldApplier::walk_scratch`: a mark byte per cell, the BFS queue, the heat
  and source lists, grown to h * w) and reads the box around the centre row-major, so its
  output is `sorted(...)` whatever the visiting order. Test-only probe
  `unit_kernels.blast_probe` (binding `applier_blast_probe`).
- THE AIR READ is taken once, at the reference's point; its per-tile smoke factor is
  recomputed from the planes at the smoke edit (nothing between the read and that edit writes
  a plane the read reads: a later sub-step there needs the factors stored).
- THE DEVICE EDIT DOOR, step (b) for TILES: `edit_door::door_tiles_edit` (the fixed-total
  deposit: weights over their sum, the noise's low bound, the per-tile scale),
  `door_disc_edit_ns` (a DISC with noise and scale: the blast smoke), `tiles_claims`; the
  EDIT row's region arguments for TILES are (tile count, scaled, 0, 0); `edit_door.edit_of`
  rebuilds a TILES edit and a noisy / scaled DISC or TILES edit (the noise from the low
  bound, the scale as a plane of ONE with the TILE rows' factors), `door_rows` packs a noisy
  edit beside a fixed keyed-draws object (the rows never read the draws).
- THE BLAST-SOURCE TABLE is `unit_schema.SOURCE_TABLE` (W 5, capacity 32: `s_q`, `ticks`,
  `tile_start`, `tile_count`, `live` = ticks > 0) with its side table `SOURCE_TILE_TABLE`
  (W 3, capacity 2048: `y`, `x`, `live`); record functions `unit_kernels.sources_of` /
  `source_tiles_of`, the way back `sources_to_host` (in place), the lockstep LISTS rows
  `sources` / `source_tiles` (the tiles land first). A comparison row's append is all or
  nothing (both tables' room checked first; a full table counted and refused by name).
- THE SOURCES STAGE (applier stage 5, `unit_kernels.take_sources(store, applier, fields,
  lists, source_out)`): `take_source_plane` -- the pending plane handed out into a device
  (N, h, w) int32 array (zeros when None; the `source_taken` book says which) and zeroed,
  each source added in row order (saturating, counted) and counted down, then the SPAN
  COMPACTION (survivors in order, their tiles moved down to a running cursor, starts
  rewritten) -- the first compaction of a table with a side span.
- THE HAND-OVER: `ResidentFields.div_source` (the dict key "div_source"),
  `eos_step_resident(..., d_div_source)`, `run_resident_batch(..., div_source_dev=)`,
  `ResidentBatch.step(take_sources=)`: given, the EOS copies the device-taken source device to
  device and does not read the engines' `eos.div_source`; the played tick passes none and
  uploads as before (Class P, the digest gates equal).
- BULK GAS: `topo::inject_gas_n_vec` with the pair `ge_payload_bulk_gas` / `_rail`; the
  tiles a FLAT disc minus `aperture_blocked`; a deposit whose slice would pass int32 on any
  tile is checked first (`inject_core_fits`) and refused whole by name
  (`applier.BulkGasOverflow`, the `bulk_refused` book).
- ROWS (20, T 2798): per payload row `payload_generic`, `_air_mode`, `_air_disc_r`,
  `_heat_r2`, `_source_r2`, `_source_total_q`, `_expansion_live`, `_expansion_ticks`,
  `_smoke_live`, `_soot_units_q`, `_heat_base_q`, `_bulk_slice`, `_bulk_q`, `_bulk_radius_q`,
  `_bulk_temp_q`; scalars `payload_src_explosion_smoke`, `blast_smoke_noise_q`,
  `blast_air_q` (6: `blast_air_settings()`, read once per table build, not per call),
  `blast_s_max_q`, `bulk_t_rails`. Each through a door function of the law, added as a pure
  refactor (pins and goldens equal): `blast_expansion.source_total_q` / `expansion_r2`,
  `payloads.heat_base_q` / `soot_units_q`, `blast_air.disc_reach`. No product needed reporting.
- BOOKS: `div_source_sat_hits`, `div_source_cut_q`, `ge_payload_bulk_gas`,
  `ge_payload_bulk_gas_rail`, `bulk_refused`, `source_taken`; PLANE `div_source`.
- THE FAN-OUT was left out: the walks and deposits are serial on thread 0 (measured in the
  benchmark doc's P7c.2 section: 1.0 ms the frag, 3.9 ms the C4, flat in N).
- CHOICES where the design was silent: the walk marks then reads its box row-major (order-free,
  so the sorted result needs no sort); the smoke factors recomputed at the smoke edit, not
  stored; one SOURCES stage handing out a device plane (the `source_taken` book for None);
  the hand-over a device-to-device copy into the EOS scratch, the played tick untouched; bulk
  gas refused whole on overflow; the span compaction written inside the stage (its host twin
  is `take_source_plane` itself); `door_rows` packs noisy edits beside a fixed MatchDraws.
- TWIN LEDGER: the corpus (300 ticks) detonates nothing, so no twin traces `payloads.py`,
  `physics.py` or `blast_air.py` (their `[not_twinned]` reasons name the device form and the
  gate); ship_demo's frag (tick 456 of 500) is detonated on the card by the gate's SHIP part.
- WHAT THE REFERENCE SHOWED (nothing changed in it): `blast_air.disc_tiles` raises
  (numpy broadcast) on a centre more than its radius above or left of the grid -- an
  explosive row detonated there crashes `execute_payload` (the probe keeps such centres
  within the served range; the device serves them); `inject_bulk_gas` writes its first
  tiles and then raises OverflowError on a tile past int32 (not atomic; the device refuses
  the whole deposit first); a strength blast in vacuum walks nothing (vacuum is outside the
  open mask), so it deposits no heat, no smoke and no source at all, and its air read is the
  empty read (heat factor 1); the air read's per-tile scale reads only walked (non-vacuum)
  tiles, so vacuum lowers a strength blast's heat only through the thin air beside it.
- THE GATES (`tests/cuda_unit_payloads_check.py`, wrapper `tests/test_cuda_unit_payloads.py`;
  CPU `tests/test_payload_device_rows.py`): TABLES (every new row, 1 m and 1/3 m); WALK
  (1 080 walks and disc reads on 24 random maps and ship_demo / airlock_demo / playground:
  the tile set and its order, the heat factor, all-one, every smoke factor; an 18-branch
  census: every seed rule, vacuum and off-grid centres, a clipped box, gaps, walls, grilles,
  thin / starved / airless air); PROBE (300 synthetic lists, all 36 rows, DETONATE and
  RESOLVE, a 54-branch census incl. the walk's seeds, the air read, a saturation, a later-ticks
  source, bulk gas' guard, TILES noisy and scaled, a noisy scaled DISC); RACE (N = 8 envs
  walking strength blasts in one launch); DISC; SOURCES (288 env-ticks of the SOURCES stage
  == `take_source_plane`, N = 1 / 2 / 3: the hand-out, plane, table, tile table, books; an
  8-branch census: none pending, overlap, last tick, a survivor moved down, saturation);
  BULK (a deposit past int32: the reference raises, the device refuses by name, nothing
  written); SCENE (14 row sets, nine of them the P7c.2 rows: all 144 (call site, row) pairs;
  a 22-branch census incl. the shipped frag thrown and killing, a wall stopping a walk and
  a walk round through a gap, a blast walking a tile it destroyed, thin and starved air,
  two sources alive and overlapping, a source alive 30 ticks after its blast, bulk gas into
  a sealed room and against a blocked aperture); BATCH N = 2 / 8 (the disc set and a
  strength set); PADDING; OVERFLOW (8 lists incl. the two source tables); REFUSE (every row
  applied; a PAYLOAD without its door or a multi-tick source without its tables faults,
  nothing written); CHAINED (the four call-site groups in three row sets, 90 ticks, sources
  alive up to 45 ticks, the host `execute_payload` run 0 times); SHIP (ship_demo's bots 500
  ticks shadowed to the end with the frag on the card; chained, no replay, host calls 0);
  HANDOVER (resident ship_demo N = 1 and 2, 40 ticks: the twin's slot 7 reads only the
  device-taken source -- its host head takes nothing -- equal to the uploaded reference
  every tick). Every gate broken once (the patch's report: 28 single-site breaks, 27 red; the
  one green, every env walking env 0's scratch list, is a race no gate can force: each
  block reads its own writes back through its L1, so the outputs never mixed).

**As built (P6b.3, 2026-11-16).** The cut table's P6b.3 row: the signal bus, the sensors, the logic
nodes, 9e(c) and the pumps on the device. No decision moved. One line each:
- THE SIGNAL TABLES (eight, on the object tables): `unit_schema.SIGNAL_TABLE` (the bus, W 8, cap
  256: `ordinal`, `name` (`signal_names()` index), `pub`, `stg`, `node`, `emit`, `emit_ref`, `live`,
  slot order), `SENSOR_TABLE` (W 17, cap 32, ordinal order) with its disc side table
  `SENSOR_TILE_TABLE` (cap 1024), `NODE_TABLE` (W 37, cap 64: kind, `out`, the `in` span into
  `NODE_IN_TABLE` (cap 256), `cmp` / `threshold` / `require_alive` / `alive_slot`, `k` / `ema`, the
  airlock's `a_in[10]` / `a_out[9]`, `far_is_outer`, `equalize_cmd`, `interior`, `band`, `state`,
  `dwell`), `PUMP_TABLE` (W 16, cap 16: the latch, `at_slot`, port, source, target, bands, quantum,
  the span into `PUMP_IN_TABLE` (cap 64: inject rows then extract rows)) and `DOOR_DRIVE_TABLE` (W 4,
  cap 64: door-table row, close flag, slot). A slot is a bus ROW index; row objects
  `SignalRow` / `SensorRow` / `NodeRow` / `PumpRow` / `DriveRow`. Record functions
  `unit_kernels.SIGNAL_RECORDS` (`signals_of`, `sensors_of`, `sensor_tiles_of`, `nodes_of`,
  `node_ins_of`, `pumps_of`, `pump_ins_of`, `door_drives_of`), the way back `signals_to_host`,
  `nodes_to_host` (a filter's ema, an airlock's state / dwell), `pumps_to_host` (at_target); the
  load-constant tables land by being equal (`sensors_to_host`, `drives_to_host`, refused by name
  otherwise: `SignalTableMismatch`). P6b.4's export rows: `signal_digest_rows` (==
  `SignalBus.digest_rows`), `node_entity_rows`, `pump_entity_rows`.
- THE CODES are ONE table, `applier._signal_codes()` (generated `SIGNAL_EMIT_*`, `SIGNAL_SENSOR_*`,
  `SIGNAL_CHANNEL_*`, `SIGNAL_CMP_*`, `SIGNAL_NODE_*`, `SIGNAL_AIRLOCK_*`), from `unit_schema`'s
  `SIGNAL_EMITS`, `SENSOR_KINDS`, `sensor_channels()` (the accessor's frozen `CHANNEL_ORDER`),
  `comparators()`, `NODE_KINDS` and the airlock's state names.
- THE STAGES (`applier.STAGES`, renumbered in 9e's order): TOPOLOGY 0, RESOLVE 1, EMIT 2, LOGIC 3,
  DOOR_INPUTS 4, PUMPS 5, VENTS 6 (was 2), DOORS 7 (was 3), DETONATE 8 (was 4), SWAP 9, SOURCES 10
  (was 5); `cpp/src/cuda_unit_signals.cuh` (`signals::emit`, `logic`, `door_inputs`, `pumps`, `swap`,
  `sample_raw`, `field_value`, `motion_value`, `next_state`, `airlock`), dispatched by `k_applier` on
  the tables (`ApplierLists.signal_table` ... `door_drives`, `ticks`; binding kwargs
  `signal_tables=` (the eight in `uk.SIGNAL_LISTS` order) and `ticks=`). LOGIC, DOOR_INPUTS and SWAP
  read no plane (their pointer table may be null). Python faces `unit_kernels.signals_emit(store,
  applier, fields, lists, ticks)`, `signals_logic(store, applier, lists)`,
  `signals_door_inputs(store, applier, lists)`, `signals_pumps(store, applier, fields, lists)`,
  `signals_swap(store, applier, lists)` -> `(coverage, books)`, coverage the `signal_writes` book
  (writes that changed a value). Profiler labels "9e(a) signals", "9e(b) logic", "9e(c) door
  inputs", "9e(d) pumps", "9e(e) swap" (out of "9e entities").
- THE SAMPLE'S DEVICE FORM is `signals::sample_raw` (the accessor's `_channel_array` cell by cell:
  smoke through `floordiv_q` by 2^TRACE_FINE_BITS per sampled cell, never the whole plane) with
  `field_value` (the single tile, or the area's floored mean over the currently non-solid disc
  tiles, 0 when none). Motion plates walk the unit rows (a count, so the rows' order does not
  change it) with `unit_sight::has_los`.
- THE FILTER'S PROVED RANGE: |ema|, |in| <= 2^60 and 0 <= k <= 62 keep every intermediate in int64
  (proof in the header); the LOGIC stage's head checks every filter of the env (and every slot,
  span, interior pressure) and past it refuses the env by name (`applier.SignalRangeRefused`, book
  `signal_refused`, host twin `unit_kernels.filter_range`), nothing written.
- BOOKS: `signal_refused`, `signal_writes`; the pumps book through the seam's existing
  `ge_pump_inject` / `_extract` / `_transfer` and their `diag_*` twins.
- A PURE REFACTOR of the reference: `logic_nodes._DeciderEval` keeps its comparator TOKEN
  (`comparator`) beside its lambda (pins and goldens equal).
- CHOICES where this section was silent: five stages, not one (each sub-step has its own label,
  shadow and chain); the area disc a side table (the house's variable-length rule), not a radius
  recomputed on the device; one input side table per class with a flag for the pump's extract wires;
  the door drives a flat table keyed by door row; a non-door `alive` emitted as a constant packed
  at the stage's entry, and an `alive` that moves in the tick on the host only (heat source, cover)
  or a non-door `is_open` with a `state` refused by name (`SignalTableRefused`); the clock's tick a
  per-env launch input copied through the applier's pinned buffer; a filter range breach refuses
  the whole env's LOGIC stage.
- NO TWIN ROW: the corpus (ship_demo) has no bus, so no twin would trace a file (the P7c.2
  precedent); `signal_bus.py`'s `[not_twinned]` reason names the device form and the gate.
- WHAT THE REFERENCE SHOWED (nothing changed in it): a motion plate's count is order-free, so the
  per-tick `sorted(sim.units, key=id)` is not load-bearing; an EntityInstance has no `alive`, so a
  sensor's alive is True unless a fixture sets it; airlock_demo's `[[spawn]]` builds no unit on a
  plain `Simulation` (the gate adds the marine); `or2`-style ORs of a constant `alive` never fall.
- THE GATES (`tests/cuda_unit_signals_check.py`, wrapper `tests/test_cuda_unit_signals.py`; CPU
  `tests/test_signal_tables.py`): TABLES (identity, digest and entity rows, landings); SAMPLE (3 072
  field / clock samples on four maps with randomized planes, every channel, both sides of the trace
  floor, empty discs; 2 560 motion plates on random placements); LOGIC (3 000 steps, 21 nodes, every
  predicate both ways at its threshold, filters k 0..40 with negative emas); AIRLOCK (6 000 steps,
  eight controllers, all 15 transitions); REFUSAL (one past each bound refused by name, nothing
  written; at the bound equal); SCENE (the signal scene 320 ticks in both boundaries, airlock_demo 260
  ticks, 1b + 9e(a) (b) (c) (d) pumps + vents + doors + (e) shadowed every tick, the tables' identity
  every tick, an unhooked twin equal; a 77-branch census, three named unreachable); BATCH N = 2 / 8
  (an env with no bus); PADDING; OVERFLOW; CHAINED (the scene both ways and airlock_demo with every
  9e signal sub-step, the vents and the doors on the device, the host laws run 0 times in the twin;
  ship_demo's bots 500 ticks, the signal chains idle). Every gate broken once (the patch's report).

**As built (P7d, 2026-11-16).** The cut table's P7d row: slot 5, the horde, on the device. No
decision moved. One line each:
- THE STAGE is the applier's HORDE stage (`applier.STAGES`, merged after P6b.4: TOPOLOGY 0,
  RESOLVE 1, HORDE 2, then 9e's EMIT 3, LOGIC 4, DOOR_INPUTS 5, PUMPS 6, VENTS 7, DOORS 8,
  TIMED_CHARGES 9, DETONATE 10, HEAT_SOURCES 11, SWAP 12, SOURCES 13 -- slot 5 sits between 4c and 9e
  in the conductor, so the stage sits between RESOLVE and EMIT and the 9e stages moved up by one, by
  name). `cpp/src/cuda_unit_horde.cuh`
  (`horde_dev::stage`, called by `k_applier` with the WHOLE block, 256 threads; every other stage keeps
  thread 0 and 32 threads): the DRAIN (the env's noise list copied to the block and emptied), the LATCH
  and the prey (an EMPTY request set or no living prey ends the stage before `begin_tick`: the cache
  stands), `begin_tick` (drop at `tick % field_refresh_ticks == 0`, at a serial unlike the stored one,
  or with none stored), pass 1 (trigger, one thread per row), pass 2 (the chain iterated to its
  fixpoint: the closure the host's sweep reaches whatever its order; then `horde.clear` of the awake),
  pass 2b (hearing per row; the `noise_fields_max` keep and its settles on thread 0), pass 3 on thread
  0 in row order. Python face `unit_kernels.horde(store, applier, fields, ring, noise, doors, tiles,
  cache, latch) -> (coverage, books)`.
- PASS 3 is a resumable walk (`horde_dev::advance` / `zombie_step`, phases START / PREY /
  PREY_FEARLESS / NOISE): thread 0 runs the zombie until it needs a field the cache lacks, appends the
  key (refused by name past the capacity: `applier.HordeCacheOverflow`, book `horde_refused`), the
  block floods it, thread 0 resumes the zombie where it stopped -- the reference's lazy, live-map,
  first-need flood. The bite is the immediate `apply_packet` (cause melee, the biter credited, the
  conversion flag), the bash `topo::chew_wall` (moved out of the applier's CHEW, a pure refactor: both
  call it) with its noise appended to the noise list, the wander `unit_draws_dev::seedless_word` (the
  vendored block keyed (0, 0) on (tick, 0, SALT_WANDER, id)) and `below`.
- THE FLOOD is `horde_dev::flood_block`: the anchor grid's passability per cell, the seeds on thread
  0, then the frontier processed in parallel (an atomic CAS claims a cell, an atomic counter appends it
  to the next frontier). BFS distances do not depend on the visiting order, so it equals
  `horde_field.cpp` cell for cell; distances are int32 in the cache's global planes (not int16 in
  shared memory: a ship map does not fit a block's shared memory, and the cache must outlive the
  launch). The TEST-ONLY `horde_flood_probe` (`cpp/src/cuda_unit_horde.cu`) runs the same function.
- THE CACHE (`simulation.horde_cache`, layout `applier.HORDE_*` / generated `APPLIER_HORDE_*`) is
  device state across ticks: per env a header (`mask_built`, `has_serial`, `serial`, `floods`,
  `n_fields`), up to `HORDE_FIELD_CAP` = 16 keys in flood order (kind PREY / PREY_FEARLESS / NOISE, fp,
  sy, sx), an int32 distance plane per key and the three mask planes. `DeviceHordeCache` (CuPy;
  planes allocated for the map on first use; `persistent` never resets after its first `begin`),
  `cache_records(sim)` its ONE record function (`CacheRecords`: a header record and a record per field,
  each with a 56-bit blake2b digest of its plane(s), the planes riding along), `to_host` the landing.
- THE LATCH (`horde_cache.latch_of`): per env `(tick, GameMap._tile_serial, mode)` then one column per
  unit row, mode CONDUCTOR (-1), EMPTY (0), SET (1): what the host writes; the request set never runs a
  host slot in the tick. The device reads the topology serial from it (no device serial: the
  `tiles_changed` book is its increment, landed on the host by a chain).
- 21 TABLE ROWS (`horde_*`: the sim's own `HordeSettings`; `zombie_attack_cooldown_ticks`,
  `zombie_bite_class_q` (the STRIKE hook applied in the kernel), `zombie_shamble_ticks` /
  `zombie_sprint_ticks` (the hooks WALK / SPRINT applied inside `default_speed_q`)); BOOKS
  `horde_floods`, `horde_bites`, `horde_bashes`, `horde_refused`.
- THE HARNESS: lists `horde` (the cache staged from the host every call) and `horde_live` (the
  device's own cache, never restaged), input `horde_latch`; `ChainedSlots` sums a book over its slots
  (a 4c chew and a slot-5 bash in one chain; P6b.4 made the same change, kept from main).
- CHOICES where the design was silent: the stage, not a separate kernel, floods (the row's "a fan-out
  of the stage's block"), so one block serves both the walk and its floods (no per-(env, field) launch:
  the fields are needed one at a time in the walk's order); the coverage counts a row the stage wrote,
  so a row changed and changed back (a noise heard and settled in one slot) counts (the P7a.1
  semantics; the gate holds coverage >= the rows the host changed); a fault stops the env's pass 3 and
  refuses the run by name; the bench's "largest zombie count" is 256 (the store has no row cap).
- WHAT THE REFERENCE SHOWED (nothing changed in it): there is no movement claim -- `is_passable_block`
  reads terrain only, so two zombies wanting one tile both enter it, in row order; the prey list is
  frozen at the slot's start, so a second bite in the tick lands on a corpse (its UnitHit, no second
  kill) and a corpse still seeds the prey field and the sprint's sight; the slot writes no door runtime
  (the bash is a `wall_hp` write; the door dies whole at 9e(d)'s sweep); the stride clock's `+= 1` then a
  step's reset can leave a row unchanged across the slot; playground's bots match ends at tick 33
  before any zombie reaches a door (P6b.1's note).
- ONE CLASS-T TWIN, `horde` (slot "5 zombie AI"; traced: `ai_zombie.py`, `horde.py`, `noise.py`,
  `philox32.py`, `turning.py`, `status.py`, `unit_impulse.py`, ...); `horde.py`, `noise.py` and
  `philox32.py` left `[not_twinned]`.
- THE GATES (`tests/cuda_unit_horde_check.py`, wrapper `tests/test_cuda_unit_horde.py`): TABLES; FLOOD
  (273 floods: seven shipped levels x three masks x fp 1..4 x three goal sets, random maps of five
  densities, a map changed between floods, a batch of three maps); SCENE (the horde scenes, three
  variants, 420 ticks each, conductor and controller latches, shadowed every tick with an unhooked twin;
  a 45-branch census, one named unreachable); CACHE (the device's own cache across 420 ticks, three
  scenes; ship_demo's 500); LEVELS (ship_demo bots 500 ticks with an unhooked twin, playground);
  BATCH N = 2 / 8 (a zombie-free env, env 0 not the largest); PADDING; OVERFLOW; REFUSE (the ring, the
  noise list); CHAINED (two scenes, ship_demo 500 ticks; the host zombie law run 0 times); CHAINED ALL
  (2' + landing | 4 + 4a + 4c + 5 | 9e(d) on the scene, the first two on ship_demo, 500 ticks). Every
  gate broken once (the patch's report).
**As built (P6b.4, 2026-11-17).** The cut table's P6b.4 row: the timed charges and the heat
sources on the device, slot 9e chained whole, the entity and signal sections exported from the
tables. One decision moved (below: the detonations of 9e are applied AT ONCE, not recorded for
the DETONATE walk). One line each:
- THE TABLES (on the object tables): `unit_schema.TIMED_CHARGE_TABLE` (W 13, cap 32: `ordinal`,
  `fy`, `fx`, `payload` (the row's index), `period_ticks`, `ticks_to_fire`, `heat`,
  `trigger_temp_q16`, `fuse_ticks`, `enabled`, `fired`, `alive`, `live`) and `HEAT_SOURCE_TABLE`
  (W 11, cap 32: `ordinal`, `fy`, `fx`, `heat_q16`, `heat_radius`, `enabled`, `hp_q16`,
  `death_payload` (-1 none), `max_temp_q16`, `alive`, `live`), row objects `TimedRow` /
  `HeatRow`. Record functions `unit_kernels.timed_charges_of` / `heat_sources_of`, the way back
  `timed_charges_to_host` / `heat_sources_to_host` (onto the sim's own runtimes: the countdown,
  latch and enabled; hp, alive and enabled; another row refused by name,
  `HazardTableMismatch`), entity rows `timed_entity_rows` / `heat_entity_rows` (and
  `sensor_entity_rows`). No Device tables row: every value is a column read from the runtime
  (the per-tick deposit's float is the runtime's once-quantized `heat_q16`, which the edit door
  re-quantizes exactly).
- THE STAGES (`applier.STAGES`, renumbered in 9e's order): TOPOLOGY 0, RESOLVE 1, EMIT 2,
  LOGIC 3, DOOR_INPUTS 4, PUMPS 5, VENTS 6, DOORS 7, TIMED_CHARGES 8 (new), DETONATE 9 (was 8),
  HEAT_SOURCES 10 (new), SWAP 11 (was 9), SOURCES 12 (was 10); `cpp/src/cuda_unit_hazards.cuh`
  (`hazards::timed_sweep`, `heat_sweep`, `blast_damage`, `payload_record`), dispatched by
  `k_applier` with ONE `apply_payload_rec` lambda that the effect list's PAYLOAD branch, both
  stages' detonations share (one payload law, one booking). Python faces
  `unit_kernels.timed_charges(store, applier, fields, ring, walk_lists, table)` and
  `heat_sources(...)`; bindings `applier_run(..., timed_table=, heat_table=)`. Each stage appends
  the PAYLOAD records it applied to its own effect list (compared with the host's calls, never
  walked). BOOKS `timed_fired`, `heat_deaths`, `heat_deposits`, `heat_refused`. Profiler labels
  "9e(d') timed charges" and "9e(d'') heat sources" (the last of "9e entities").
- THE DECISION THAT MOVED: the brief asked for the timed charges' detonations as PAYLOAD records
  walked by DETONATE. The reference reads a heat charge's tile temperature BETWEEN detonations
  (`destroy_wall` sets a destroyed tile's T to 0; bulk gas refreshes a cell's mirror), so a
  record-then-walk form differs when an earlier charge of the sweep destroys or gasses a later
  heat charge's tile; the scene takes that case (a heat charge on a hot wood wall that an
  earlier charge's blast destroys in the same sweep: the reference, and the device, read it
  cold). So both stages apply at once through the one payload function; the planted charges
  keep P7b's records + DETONATE (`charges.tick` reads no map).
- THE CHAIN'S BOUND (proved; the head refuses past it): each source dies at most once (a dead
  source is skipped) and a death applies one payload, which appends at most one disc (only a row
  with radius > 0 appends). So the discs read number at most D0 + S_d (D0 the log's count at the
  head, S_d the living sources whose death payload's row has radius > 0) and the log never holds
  more. The head refuses D0 + S_d > the log's capacity (64) BY NAME (`applier.HeatChainRefused`,
  host twin `unit_kernels.heat_chain_bound`), nothing written.
- THE EXPORT is `unit_kernels.EntityExport(sim)`: `entity_section(tables)` /
  `signal_section(bus_records)` / `digest(section)`; each record is `serialize.record_head` (made
  once) + `record_tail` of the table's entity rows (doors with their tile table, ducts, vents,
  nodes, pumps, sensors, timed charges, heat sources); a class with no device table (button,
  terminal, light, prop, zones, cover) carries a tail taken at build (nothing moves it on the
  continuous path; the gates hold it equal every tick). `serialize.entity_records` split into
  `record_head` / `record_tail` (a pure refactor: pins and goldens equal).
- THE `alive` REFUSAL: the heat source's half is lifted, not by a device read of the table: the
  reference's emit reads `_entity_by_ordinal`, built BEFORE `build_heat_sources` replaced the
  instance with its runtime, so a wired heat-source `alive` reads 1 even after the source died
  (a finding, not changed); the device emits that constant, and `signal_rows` refuses by name a
  map that holds the runtime. What remains refused: a cover row's `alive` (OnePhaseWEGO's chew)
  and a non-door `is_open` with a `state`.
- THE HARNESS: `LISTS` rows `timed_charges`, `heat_sources` (landed) and `timed_payloads`,
  `heat_payloads` (logged PAYLOAD calls; the tap's sites gained `timed_charge_system` and
  `heat_source_system`; `PAYLOAD_LOGS`); FIXED: `ChainedSlots` SUMS the books of a chain's slots
  (it kept the last slot's: a whole-9e chain books one channel in several stages);
  `ChainedSlots` on a unit-free level pads one row.
- CHOICES where the design was silent: at-once application (above); the own effect list as the
  stages' record of what they applied; the chain's bound from the log's capacity; the export's
  static tails for classes with no table; a chained 9e starts at the first sub-step whose host
  gate is open on the level (the conductor runs nothing before it); the shipped level with a
  heat source does not exist (no shipped level holds one), so the scene and the batch carry them.
- WHAT THE REFERENCE SHOWED (nothing changed in it): a wired heat source's `alive` never falls
  (above); a heat source's blast damage is the disc law, which a wall does not stop (the scene's
  source behind an intact wood wall takes it); the brief's 9e(d') order "planted, then timed" is
  the reverse of the conductor's (`sweep_timed_charges` runs before `charges.tick`): the device
  follows the code; the grenade, charge, round and blast-source tables are in no digest (adding
  them is a membership change: a finding for Erik).
- THE GATES (`tests/cuda_unit_hazards_check.py`, wrapper `tests/test_cuda_unit_hazards.py`; CPU
  `tests/test_hazard_tables.py`; the scene `tests/_hazard_scene.py`): TABLES; SCENE (the hazard
  scene both boundaries 160 ticks and fire_smoke_studio 120 ticks: 9e(d') timed, 9e(d') planted
  and 9e(d'') heat shadowed, the tables' identity every tick, an unhooked twin equal; a 32-branch
  census, three named unreachable); BOUND (at the bound equal, one past refused, nothing
  written); BATCH N = 2 / 8; PADDING; OVERFLOW; CHAINED (slot 9e whole, ten slots, on the scene
  both ways, airlock_demo, fire_smoke_studio and ship_demo's bots 500 ticks; the export from the
  device's tables equal to the serializer every tick; the host laws run 0 times in the twin).
  Measured in the benchmark doc's P6b.4 section. Every gate broken once (the patch's report: 30
  single-site breaks, all red).

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

**P8 RE-CUT (2026-11-20; Erik: "observation - i think we just move it
directly").** Cut from `docs/residency_p8_observation_inventory_2026-11-20.md`
(parts A to K; read it before any P8 patch). It replaces the P8a / P8b rows of
§9 for everything below; P8c (the training-episode record) stays after it.
Why now: the batched tick does a ship_demo decision in about 2.6 ms per env
while the host observation costs about 23 ms per env, so a host observation
caps the gain at about 4x (#29, 2026-11-20). No training run starts before
the gas solver's instability (#217) is fixed; this port does not wait for it.

*What does not change.* Observation spec v1, the full view, action spec v1 and
reward profile v1 stay what they are, value for value. The host builders
(`obs_spec_v1`, `team_picture`, `reward_v1`, `control_policy`) stay THE
reference; the device forms are twins of them, as the unit kernels are twins
of the CPU tick. The env keeps ONE API (`BatchedEnv`), a `device` backend
beside `cpu_farm`.

*The gate every patch shares.* A reference sim on the CPU path replays a
recorded match beside the conductor; at every decision tick (per tick for
state that lives across ticks) the host arrays are compared with the device's,
block by block, value for value, each gate shown red once by a break switch.
Corpus: (a) the pinned `bots` / `bots_v1` matches, (b) a training-room match,
(c) one recorded "v1" episode driven by random decisions through
`AgentPolicy`, (d) small hand matches for cases the others never reach.

*Lead's decisions (computer science; each vetoable).*
1. **Floats: exact, by one shared function, never a tolerance.** Every float
   column is built on the card in float64 in the reference's own operation
   order, FMA contraction off, cast ONCE to the served dtype. The four places
   that call libm (`squash`'s log, `d_c`'s log1p and hypot, `_hit_from`, the
   wind direction) move, on BOTH sides, to one small module of functions
   written in IEEE basic operations only (`hypot` as `sqrt(dx*dx + dy*dy)`;
   one `log` as a fixed sequence of +, -, *, / on the split mantissa), so the
   host and the card run the same operations and agree bit for bit, and so do
   Windows and a Linux node. This moves the HOST observation by at most the
   last bit of a float64 before the cast (no network is trained on it yet;
   the rung-2 copy records are re-made anyway). It lands as its own patch,
   P8.0b, with a gate that bounds the move against the old libm values.
2. **Sight rule: both.** `all_round` (what training uses) and `cone` (what
   the pinned matches use, IEEE-only float64) are both built in P8.2, so
   corpus (a) serves every later gate.
3. **Orders: the observation reads the device order state; who writes it is
   decided by measurement.** The order is game state already on the card
   (the ORDER table). The first form keeps the host commander, fed by one
   download on its own period (every 6 ticks), measured at N = 64 and 256 in
   P8.6. If that download or the Python team layer per env is the new
   bottleneck, the options (port the team layer; a run without a commander)
   go to Erik: the second changes what a soldier is shown.
4. **Reset: host build, device restage, first.** Env e's next episode is
   built through `MatchSession` on the host (equal to the reference by
   construction) and its slice of every device structure restaged, the
   others untouched. The bank file of §4.6 is built only if the measured
   cost says so.
5. **No host copy between the card and the trainer.** The obs buffers are
   CuPy arrays owned by the env and handed to torch by DLPack; `agents/`
   changes to hold its batch as device tensors (P8.10).
6. **Opponents on the device backend:** self-play and the horde; any other
   controller of a team is refused by name (§12.3 stands).

*The patches* (one implementer at a time; a green patch merges; the played
game never changes):

| patch | what | gate | tier |
|---|---|---|---|
| P8.0 | the whole path at its smallest: CuPy obs buffers, the (team, slot) to store-row map, `valid` / `in_match`, ONE integer block (near wall / door, far wall share / door) built inside the conductor tick at the decision cadence, handed to torch by DLPack | the block EXACT on (a), (b); the tensor's pointer is the buffer's; still 2 syncs per tick | Opus |
| P8.0b | the shared float functions on the host (decision 1) | every observation value within the stated last-bit bound of the old one on (a), (b); the functions' own property tests (monotone, exact at powers of two, equal on host and card over a dense sweep once the card has them) | Opus |
| P8.1 | the obs table and every vector column whose sources are on the card | vector EXACT on (a), (b) | Opus |
| P8.2 | sight in bulk on the card and who sees whom, both rules; the refusal of a level with live cover | pairs and windows equal to `rays_clear` / `window_clear`; `seen_by` / `visible_by_team` equal every tick | Opus |
| P8.3 | the device team picture (records, motion, damage window, bars, the object tracker with a stable serial), a last stage of the walk | the picture's whole state equal EVERY TICK on (a), (b), (d) | Opus |
| P8.4 | the list and the four masks (the `use_door` geometry lifted into the unit library) | `list`, `list_mask`, `list_uid`, masks EXACT | Opus |
| P8.5 | the windows: physics planes, air, felt temperature, the two sight planes, route and place | near / far planes EXACT | Opus |
| P8.6 | the decoder and the action latch on the card; the commander's download measured | latch columns EXACT every tick against `AgentPolicy` on (c) | Opus |
| P8.7 | reward books, `terminated` / `truncated`, the result | reward groups, counters, terminals EXACT per step | Opus |
| P8.8 | the full view | EXACT | Opus |
| P8.9 | per-env reset (decision 4) | env e reset mid-batch == env e built alone; the other envs unchanged at the reset tick | Opus |
| P8.10 | `BatchedEnv(backend="device")` and `agents/` on device tensors | BACKEND EQUALITY: same run file, seed and actions give the same obs, rewards, terminals and checkpoints on both backends, N = 1 and a batch | Opus |

Order: P8.0, P8.0b, P8.1, P8.2, P8.3, then P8.4 and P8.7 (both need only
P8.3), P8.5, P8.6, P8.8, P8.9, P8.10. After P8.5 the bench is re-run with the
observation on the card: that number says whether the remaining host work
(the commander, checkpoints) matters.

*Risks carried* (inventory J): the host controllers that run every tick; the
team picture as state across ticks with Python-identity semantics; the reset
having no form yet. *Open by name* (inventory K): numpy's summation order in
the reward fold (checked in P8.7), a direct double to half conversion on the
card (checked in P8.0b / P8.1), a continuous-path level with live cover (P8.2
refuses it), the cost of a checkpoint download at N = 256 (measured in P8.10).

*Systems.* (a) Used as they are: Device conductor, Unit store, Device tables,
Object tables, Tick row, Event ring, Device sight, Device unit library,
Batched resident engine, Observation spec, Full view, Action spec, Reward
profile, Team picture, Batched environment, Verb gates, Order, Room graph,
Class P / Class T. (b) New, each ONE index line at its patch: *Device
observation* ("THE device form of observation spec v1 and the full view,
built in the conductor tick into env-owned buffers; the host builders stay
the reference; never a second observation"); *Observation arithmetic* ("the
float functions an observation calls beyond IEEE basic operations are ONE
module on host and card; never libm in an observation"); *Device team
picture*; *Device decoder*; *Device books* (reward and terminals).

**P8 RE-CUT, VERSION 2 (2026-11-20, after the critique:
`docs/residency_p8_recut_critique_2026-11-20.md`, 1 blocker, 10 major, 12
minor, all folded in here; where this block and the one above disagree, this
one holds).**

*Decisions changed.*
- **Decision 1 is withdrawn in its strong form** (critique 2). Every log
  result is only cast to float32 / float16 and never branched on; a 1-ulp
  move of the float64 changed the served value 0 times in 5e6 tries. So: NO
  shared log module. `hypot` becomes `sqrt(dx*dx + dy*dy)` on both sides
  (IEEE, exact); the card uses CUDA's own `log` / `log1p`; the gate compares
  the SERVED arrays as raw bits (so -0.0 and NaN count), and a single
  differing value FAILS and is printed. The shared module is built only if
  that gate ever goes red.
- **`action_spec_v1.onto_reach_q`'s `float(d2) ** 0.5` becomes `math.sqrt`**
  (critique 1): it is `pow`, it feeds the throw latch (game state), and it
  differs from sqrt before quantizing on 8011 of 4e6 integers, on none after.
  P8.0b proves "no quantized value moves" by an exhaustive sweep over every
  reachable d2; no re-pin.
- **The "counters" reward profile is refused by name on the device backend**
  (critique 3): numpy sums its fold pairwise, which a serial kernel cannot
  equal; the named profiles' fold is sequential and stays exact.
- **Cover is ported, not refused** (critique 7, the blocker): the training
  rooms place cover crates and only the observation reads them
  (`line_of_fire.solid_in_way`). A per-env cover table and a device
  `blocks_segment` (an IEEE slab test); only `blocks_los = true` cover is
  refused, by name.
- **Decision 3 is sharper** (critique 17): the host commander needs a named
  host view of what it reads (units, the picture, vision, fire lanes, the
  horde flood, the room goals) built from ONE download, with the device's
  vision, never a recomputation; after the first resets some env is due on
  every even tick, so the download is per tick, not per 6 ticks; estimated
  20 ms of host Python per tick at N = 256 with two policy teams. Built and
  measured in P8.6b with staggered resets; the choice that follows (keep it,
  port the team layer, or a longer commander period) goes to Erik with the
  number.
- **Decision 4 stands** (critique 18): a `MatchSession` build is 18 to 21 ms;
  the next episode is pre-built while the current one runs.

*Rules every gate follows* (critique 4, 8, 12, 22).
- A COLUMN-TO-PATCH table (filled by P8.0 from the inventory's part A; it
  lives as `tests/obs_column_owners.toml`, machine-readable, read by the
  shared check): every column and plane of the observation, the full
  view, the masks and the reward has ONE owning patch; a gate compares what
  it and earlier patches own.
- A gate counts the non-default values of every column it owns and FAILS on
  an owned column that is constant over its corpus (the pinned bots matches
  issue no order and no `aims`; a training room has no zombies).
- Corpus (c) is generated live inside the CUDA subprocess: an in-process
  `EnvRunner` with seeded masked-random decisions (a Recording replay swaps
  the policy controller out, so it cannot serve). Orders reach a corpus
  through a commander team or the door bots.
- THE WALL has a device gate: perturbing an unseen enemy's store row changes
  no policy observation value.
- One shared CUDA observation check script, so each pinned match is recorded
  once per subprocess; each device builder adds its Class T twin row.

*The patches, version 2.*

| patch | what | gate | tier |
|---|---|---|---|
| P8.0 | the path at its smallest: CuPy obs buffers (double for the `final_` keys), the (team, slot) to store-row map, `valid` / `in_match`, a stub ended-env latch, ONE integer block (near wall / door, far wall share / door) built in the conductor tick at the decision cadence, handed to torch by DLPack with the stream ordering asserted; the shared check script; the column-to-patch table; spec constants from a generated layout header | the block EXACT (raw bits) on (a), (b); the tensor's pointer is the buffer's; 2 syncs per tick | Opus |
| P8.0b | host arithmetic made portable: `hypot` to sqrt, `onto_reach_q` to sqrt | the exhaustive d2 sweep; pinned matches' digests unchanged; observation values move only where hypot was not correctly rounded (counted) | Sonnet |
| P8.1 | the obs table (from the reference's own objects, floats as their own bit patterns, `pressure_min` / `max`), the hand double-to-half conversion with its exhaustive test, the vector columns that need only the store and tables | owned columns EXACT | Opus |
| P8.2 | bulk sight on the card and who sees whom, both rules; the cover table and `blocks_segment`; observation double code in its own header, not the unit library | pairs / windows equal to `rays_clear` / `window_clear`; `seen_by` / `visible_by_team` equal every tick | Opus |
| P8.3 | the device team picture (records with a refused overflow, motion, damage window, bars, the object tracker with a stable serial), the last stage of the walk; the vector columns it owns | the picture's whole state equal EVERY TICK on (a), (b), (d) | Opus |
| P8.4a | lanes (with a floor-division helper), `aim_of`, line of fire | owned columns EXACT | Opus |
| P8.4b | the list's five blocks, sorting, `list_uid` | EXACT | Opus |
| P8.4c | the four masks; `use_door`'s geometry lifted into the unit library | EXACT | Opus |
| P8.5a | the physics planes, air, felt temperature, the two sight planes, the far unit planes | EXACT | Opus |
| P8.7 | the reward books (with `RewardBooksMismatch` as a counted per-env fault), `terminated` / `truncated`, the result, the per-env "ended at tick t" latch that freezes the books | EXACT per step | Opus |
| P8.6a | decisions as a device tensor, `check_actions` / `masked_choices`, pointer to uid, the per-tick decoder writing the latch columns, the slot state, the zombie latch (its equivalence a gate) | latch columns EXACT every tick on (c) | Opus |
| P8.6b | the commander's host view from one download; the order block, route and place (P8.5b) | owned columns EXACT with orders issued; the cost at N = 64 / 256 with staggered resets | Opus |
| P8.8 | the full view | EXACT | Opus |
| P8.8b | the terminal observation: the `final_` keys built at the ending tick | EXACT at every ended env | Opus |
| P8.9a | per-env reset of the sim side (store slice, tables, lists, caches, tick row, broods) | env e reset mid-batch == env e built alone; others unchanged | Opus |
| P8.9b | per-env reset of the observation side (picture, slot state, books, per-template rows); the pre-built next episode | the same, on the observation | Opus |
| P8.10a | `BatchedEnv(backend="device")`: coverage counters, the pre-flight, `require_snapshot`'s device branch, the checkpoint download; the stale "P8b" lines in rules and headers | BACKEND EQUALITY on both backends, N = 1 and a batch | Opus |
| P8.10b | `agents/` on device tensors | the trainer's own suite; a collect step with no host copy of an observation | Opus |

Order (real edges): P8.0 -> P8.0b -> P8.1 -> P8.2 -> P8.3 -> P8.4a ->
P8.4b -> P8.4c -> P8.5a -> P8.7 -> P8.6a -> P8.6b -> P8.8 -> P8.8b ->
P8.9a -> P8.9b -> P8.10a -> P8.10b. The bench is re-run after P8.5a and
after P8.6b.

**P8.0 AS-BUILT (2026-11-20, branch `29-p80-obs-path`).** The device
observation is `simulation.device_obs.DeviceObservation(conductor, specs,
n_slots, decimation, n_teams=None)`: CuPy buffers of every served key
(`served_shapes`: `obs_spec_v1.shapes` + `action_spec_v1.mask_shapes` +
`obs_spec_v1.full_shapes`) and a `final_` set, a roster row map `(N, T*S)`
int32 (store row or -1, from `Simulation.roster`, checked against the store's
`slot` column at the build), a per-env parameter row (episode start tick,
decimation) and an `ended` `(N,)` uint8 stub latch; `torch()` (DLPack,
pointer-identical), `team_rows(t)`, `host(keys)` (the gate's download). The
conductor gained `attach_observation(obs)` and ONE call line at the walk's
end (`11' observation`, after slot 10's tail): `build_if_due()` launches when
some env is on a decision tick (host clocks, no sync); the kernel checks each
env's own cadence on the tick row and skips an ended env. The block is
`bp.obs_block` (`cpp/src/cuda_obs.cu`): an atomic-max door plane from the
door / door-tile tables, then one thread per (env, row, window cell) writing
near `wall` / `door`, far `wall_share` (float16 bit patterns of k/16 from a
generated table) / `door`, and `valid` / `in_match` (cell 0). Constants: the
generated `cpp/src/obs_layout.h` (`tools/gen_obs_layout.py`, fingerprint
checked at the build). The column-to-patch table is
`tests/obs_column_owners.toml` (owner choices where the inventory is silent
are written at its head); `device_obs.BUILT` must equal the rows up to the
check's `PATCH`. The shared check `tests/cuda_obs_check.py` plays each
corpus match ONCE on the host with its own controllers, tapping the intents
and the owned host observation per decision (no recording, no second
replay); parts gate / batch / handover / syncs / ended; breaks `anchor`,
`stale`, `constant`. Gates: raw-bit equal at every decision tick on bots
(338 decisions), bots_v1 (69), range_door (51); batches N = 2 / 4 of
ship_demo (seeds 11-14, ends 1664 / 300 / 970 / 275) and range_door (seeds
21-24); the three breaks red at their first decision; torch pointers equal,
tick stream flags 0 (blocking), CuPy / torch on stream 0; 2.00 syncs per
steady tick at N = 1 and 2 with blocks built. Cost (`tools/bench_conductor.py
--obs`, ship_demo, GPU time of one decision's blocks for all envs): 0.014 ms
at N = 1, 0.017 ms at N = 64, 0.039 ms at N = 256. The twin row
`observation` (Class T) sits on slot `11 team picture / observation`, so the
team picture and vision became its traced files (their `[not_twinned]` rows
retired); `obs_spec_v1.py` / `rl_env.py` are not traced by the twin corpus
(named in the row as an accepted gap).

**P8.0b AS-BUILT (2026-11-20, branch `29-p80b-host-arith`).** The host
reference's roots are IEEE. `action_spec_v1.onto_reach_q` takes
`math.sqrt(float(d2))` (was `float(d2) ** 0.5`, the C runtime's `pow`); the
observation's four `hypot` calls became ONE scalar `obs_spec_v1.length(dx,
dy)` (`float()` of each, two products, one add, `math.sqrt`) for
`_hit_from` and the two `d_c` callers, and `np.sqrt(vx*vx + vy*vy)` on the
float64 wind plane (the same operations element for element); their
`ingress-exempt` comments went with them (`sqrt` is door 3). The other files
the critique named (`team_picture`, `reward_v1`, `control_policy`,
`sensor_accessor`, `rl_env`) held no `hypot` and no float `**`
(`control_policy`'s aim norm was already `np.sqrt`); `log` / `log1p` stay.
The sweep (`tools/reach_sqrt_sweep.py`, run once, 132 s): the bound is
every integer d2 in [0, 2**31), since the caller's two points are tile
coordinates of Q16.16 int32 positions (|dx|, |dy| <= 32767, d2 <=
2 147 352 578). The old form is reproduced in bulk by `np.power` with an
ARRAY exponent (checked equal to CPython's `**` on 6e6 values; a scalar 0.5
exponent takes numpy's sqrt shortcut and would test nothing). The roots
differ before quantizing on 3 988 877 integers; the quantized reach on 3
(1455325460, 1716802626, 1878698752), none of them a sum of two squares
with both terms <= 32767 and all with roots past 38 000 tiles (beyond the
int32 Q16 reach column, d2 < 2**30): 0 reachable values moved, no re-pin.
`tests/test_portable_host_roots.py` keeps a dense sample (every d2 < 2**22,
every offset within 256 tiles against the exactly rounded integer root, a
seeded sample to 32767) and an AST check that the observation, action
spec, decoder, picture, reward, accessor and env files call no `hypot` and
no float `**`. Served values that moved (old module from `main` beside the
new one, each decision observed new / old / new, the new pair the control):
0 of 132 043 950 values (obs keys and the full view, raw bits, 267
columns) over bots (338 decisions), bots_v1 (69) and range_door (51). The
order column `vector[91]` is not exercised by that corpus (no orders).

**P8.1 AS-BUILT (2026-11-21, branch `29-p81-obs-vector`).** The 58 vector
columns the column-to-patch table gives P8.1 (body, the nine status
clocks, kit and grenades, throw / plant / melee / shove, class, weapon and
its three ratios, trigger, the clock, `mates_alive`, `team_ammo`) are
built by `bp.obs_vector` (`cpp/src/cuda_obs.cu`, one thread per (env, row),
the same cadence and ended latch as P8.0's block, launched beside it from
`DeviceObservation.build`); no row moved to another patch (every one needs
only the store and the tables). THE OBS TABLE is `device_obs.table_values`
(per env, `(OBS_TAB_W,)` int64, 149 wide on today's config: the
`ObsContext` scalars, the sim's turn-rate class value, the hand wind-up and
plant ticks, the nine status scales, the three plane densities as float64
bits; per loadout row magazines / explosives / charge / weapon row / the six
grenade counts; per weapon row mag size / range / rof / reload / the cone
door of `spread_deg`; per `[env_profiles.*]` row and per species profile
`temperature_max` / `pressure_min` / `pressure_max` as float64 bits) and
`team_values` (`(T, 2)`: n, loadout row), offsets generated into
`obs_layout.h` (in the fingerprint; the block widths are config table
sizes). Choices where the design is silent: the integer reads the vector
shares with the unit kernels (`mag_status`, `composed_flags`,
`handling_ticks`, `scaled`, `wrap_q16`, the kit's `cos_q16` / `sin_q16`)
are the unit library's, called with the device tables, the divisors the
obs table's (the weapon rows sit in both, each read from the sim's own row
object; a CPU test holds the obs table to those objects); the species'
own environment profiles are carried too (a unit without `env_profile`
reads its species'); the mitigation `resist_mult[POISON]` and
`heat_dials` are left to P8.5a, which reads them. THE FLOAT KIT is
`cpp/src/cuda_obs_float.cuh`: `__dadd_rn` / `__dmul_rn` / `__ddiv_rn`
(never fused, whatever `--fmad`), `ratio`, `squash` with CUDA's `log`, the
cast to float32 (`__double2float_rn`, a NaN given numpy's quieted pattern)
and the hand double -> float16 (integer bits, nearest even once,
subnormals, overflow, signed zero, numpy's NaN rule). The vector is
float32; the float16 keys (`near_f`, `far_f`) take the hand conversion
from P8.5a on (`device_obs.FLOAT16_KEYS` / `FLOAT32_KEYS`). Gates
(`tests/cuda_obs_check.py`, `PATCH = "P8.1"`): raw-bit equal at every
decision tick on bots (338 decisions), bots_v1 (69), range_door (51) and
corpus (d) (81: a ship_demo bots match, seed 31, cut at 320 ticks, with a
hand state at tick 0 for the reference and the twin alike -- the
`[marine]` loadout row in the live tables replaced by one carrying all six
hand-grenade kinds, a marine without loadout or melee row, two marines with
other weapons, a blow just struck, status kinds 1..8 -- and hand intents
through the tap: a plant, throws of kinds 1..5, a reload, a shove); batches
N = 2 / 4 of ship_demo (493 / 805 env-decisions) and of range_door (112 /
220); every owned column non-constant over (a)+(b)+(d) (31 were constant
over (a)+(b) alone: the other grenade kinds, statuses 1..8, class, weapon
and its ratios, the plant, the melee cooldown, `ammo_untracked`,
`charges`, `can_aim`, `has_melee`, `has_charge_kind`); `log` moved 0 served
values on the corpus. Part FLOATKIT: double -> float16 equal to numpy on
446 538 values (every half, every midpoint and its two float64 neighbours,
the edges, NaN payloads), double -> float32 on 1 246 538, a*b + c never
fused on 100 000 triples where the fused form differs on all, the squash's
served float32 equal to the host's on 414 101 values (its float64 differs
on 92: CUDA's log is not correctly rounded, the cast hides it). Breaks, each
red: `divisor` (the weapon block shifted one row), `half_f32`, `fused`,
and P8.0's `anchor`, `stale`, `constant`. 2.00 syncs per steady tick at
N = 1 and 2. Cost (`tools/bench_conductor.py --obs`, ship_demo, GPU time
of one decision's blocks for all envs, P8.0's block and the vector
together): 0.018 ms at N = 1, 0.018 ms at N = 64, 0.055 ms at N = 256 (P8.0's
block alone measured 0.014 / 0.017 / 0.039).

**P8.2 AS-BUILT (2026-11-21, branch `29-p82-obs-sight`).** No served
column; the sight every later block reads. THE DEVICE FUNCTIONS are
`cpp/src/cuda_obs_sight.cuh` (namespace `obs_sight`, the observation's own
header, including `cuda_unit_sight.cuh`'s `has_los` and never walking a line
itself): `line_clear(solid, screen, h, w, physical, y0, x0, y1, x1, bad)`
(`rays_clear` for one pair), `window_cell(..., to_window, y, x, wy, wx, ww,
k, bad)` (`window_clear`'s cell k, row-major), `blocks_segment(cover_row,
x0, y0, x1, y1)` (`CoverRuntime.blocks_segment`, float64 through the kit:
`1.0 / d`, the two products, swaps and clips in the reference's order),
`solid_in_way(solid, h, w, covers, n, ax, ay, bx, by, bad)`
(`line_of_fire.solid_in_way`: the physical walk, then every live cover),
`within_cone(facing_q, dx, dy, thr)` and `can_see(o, t, rule, cone_cos,
radius, ...)` (`vision.can_see` for two living units of different teams,
both rules), plus the one `floordiv` / `tile_of` / centre (P8.0's copies in
`cuda_obs.cu` now use these). THE LAUNCHES (`cpp/src/cuda_obs_sight.cu`,
every env in one launch on the tick stream, a `(N,)` fault-bit buffer, no
sync): `bp.obs_vision` (one thread per (env, observer row, target row)),
`bp.obs_rays` (pairs `(N, P, 4)`), `bp.obs_windows` (queries `(N, Q, 4)` =
(y, x, wy, wx), out `(N, Q, wh, ww)`), `bp.obs_lof_pairs` (every (row,
row) of the store over the cover table); faces
`device_obs.rays_clear_device` / `window_clear_device` (CuPy planes and
pairs) and `DeviceObservation.vision` / `lof_pairs`. WHO SEES WHOM is
`DeviceObservation.sees` `(N, U, U)` uint8 (`sees[e, o, t]`: store row o
is among `seen_by[row t]`) and `visible` `(N, T + 1, U)` uint8 (slot s < T
the marine team s, slot T the zombie team; a unit on any other team is
refused at the build, `check_teams`); `device_obs.vision_state` reads one
env's tables back as `VisionState`'s two dicts. It runs EVERY tick at the
walk's `11'` line BEFORE the blocks (`DeviceObservation.stage`, which the
conductor's call line now calls) and at the build (tick 0): the place the
reference's slot 11 (`team_picture.update`) first asks `vision.state_for`,
after the advance, order expiry and the ruleset's tick end (the
conversion); every later ask in the reference's tick (the bots at the next
tick's start, the observation) reads that cache. Every tick, not only on
decision ticks: P8.3's picture reads it every tick. Ended envs are not
skipped (cheap, and the state is defined). THE RULE AND ITS DIALS are obs
table rows (`device_obs.sight_values`): `SIGHT_RULE` (`match.SIGHT_RULES`'
index of `Simulation.sight`), `CONE_COS_BITS` (`vision.cos_threshold` of the
live half-angle) and `AWARENESS_R_BITS` (`float(awareness_radius_tiles)`);
`OBS_SIGHT_*` and `OBS_ZOMBIE_TEAM` are generated constants. THE CONE needed
nothing beyond the kit: `Unit.facing` is `facing_q / 65536.0` exactly, so
`unit_fixed.cos_rad` / `sin_rad` quantize it back to `facing_q` and the
facing vector is `cos_q16(f) / 65536.0`, `-(sin_q16(f) / 65536.0)` (the
integer kit, already on the card); `math.radians` and the threshold's
cosine are taken on the host once per table build (the table row is the
reference function's own float); the per-pair test is int squares, two
products, an add, `__dsqrt_rn` (IEEE, correctly rounded like `math.sqrt`)
and a divide, never fused; the awareness test compares the exact int `d2`
with the float64 product `r * r` as Python compares int with float. THE
COVER TABLE (`device_obs.cover_values`, `(N, C, OBS_COVER_W)` int64 + counts:
the rectangle as the CoverRuntime's own float64 bits, alive, blocks_los) is
staged ONCE at the build. K7, traced: the continuous shooting path never
reads `sim.cover` (`process_shooting`, `_directional_fire` and the auto-fire
branch call `_dispatch_trigger` without `cover`, so `march_cover` stays
None and a round passes a crate); the only writer of a cover's `alive` is
`CoverRuntime.chew`, reached only through a round marched WITH cover, which
only OnePhaseWEGO's `timeline._fire_at` passes (refused by the conductor);
payloads and blasts do not touch cover. So nothing changes a cover on the
continuous path, the game cannot destroy one there, and the check asserts
every match's covers keep their `alive`. A cover with `blocks_los = true`
is refused by name at the build: the host applies it inside every
`ray_clear` / `rays_clear` / `window_clear` (who sees whom, every window),
which the device lines do not; no level places one. Choices where the
design was silent: the visible table is per team SLOT (marine teams of the
layout plus the zombie team), not per team value; faults are a per-env bit
buffer the gate reads (no refusal path is reachable on a served level:
team values are checked at the build, coordinates are map-bounded), not a
books row; the cover table rides `DeviceObservation`, not the conductor's
object tables (nothing in the sim reads it). Finding: the check's docstring
said `range_door` carries cover; the crates are `range_hall`'s (and placed
OFF the line between the spawns: no pair of that room's match is ever
blocked by one), and the training rooms are 1 v 1, so no team there ever has
both a seen and an unseen enemy. Hence two P8.2 corpus matches: the cover
hand match `hall_cover` (`range_hall`, crate 0 made a 3 x 1 box on the line
between the spawns at tick 0, reference and twin alike) and
`ship_all_round` (the level's own ship_demo bots match with `sight =
"all_round"`, cut at 480 ticks). Gates (`tests/cuda_obs_check.py`,
`PATCH = "P8.2"`): WHO SEES WHOM equal to the reference's `VisionState`
(`seen_by`, `visible_by_team`, every pair) at EVERY tick of bots (1349
ticks), bots_v1 (272), range_door (200), the hand match (320), range_hall
(32), hall_cover (26), ship_all_round (480), and of every env of the
batches N = 2 / 4 (ship_demo, the room); census: cone 256 608 pair-ticks,
164 (match, observer, target) pairs both seen and unseen, 396 ever unseen;
all_round 64 014 pair-ticks, 45 both, 138 ever unseen; a rule without a
both-pair fails. THE LINE OF FIRE equal on every (unit, unit) pair at every
decision tick of the same matches (48 672 pairs on bots, 17 424 on
ship_all_round, ...), a cover decided 2 pairs (hall_cover, tick 0; the
room's own crates decide none), every cover's alive unchanged over every
match. PART SIGHT: 30 000 pairs per env in both forms and 120 windows x
both directions x both forms against `rays_clear` / `window_clear` on
ship_demo (doors shut / 30 of 60 opened / 178 grille tiles: 773 pairs the
doors turn, 465 the two forms answer apart; ~20 % of pairs with an end off
the map, a third on one tile), range_door (its door flipped: 1155 pairs
turned) and range_hall (crates); every value equal. PART WALL: 7 (team,
tick) cases on bots (cone) and 6 on ship_all_round: an unseen enemy's row
perturbed (facing a quarter turn, hp, turn rate, effort, exposure) leaves
the team's visible set unchanged, the same on a seen enemy too, killing the
seen enemy removes it. Breaks, each red: `cone` (half-angle + 1 degree: bots
tick 419, pair 8 -> 2), `early` (vision before the conversion: bots_v1 tick
271, 6 pairs), `cover` (boxes one tile down: hall_cover tick 0), `forms`
(part SIGHT: 465 pairs on the grille map). Unchanged: every P8.0 / P8.1
part (raw-bit equal decisions as before, 2.00 syncs per steady tick at N =
1 and 2 with vision every tick). Cost (`tools/bench_conductor.py --obs`,
ship_demo, GPU time): vision 0.015 ms per tick at N = 1, 0.013 at N = 64,
0.030 at N = 256; a representative window call per decision (144 windows
of 25 x 25 per env: every roster row's window from every unit's centre)
0.039 / 0.491 / 1.919 ms; the P8.0 + P8.1 blocks 0.030 / 0.018 / 0.045 ms.
No golden re-baseline; the played game unchanged (no host value moved).

**P8.3 AS-BUILT (2026-11-21, branch `29-p83-obs-picture`).** THE DEVICE TEAM
PICTURE is `simulation.device_picture.DevicePicture` (`DeviceObservation.picture`)
over `cpp/src/cuda_obs_picture.{cuh,cu}`: `TeamPicture.update` line for line,
every env in two launches on the tick stream at the walk's `11'` line BETWEEN
device vision and the blocks (`DeviceObservation.stage`: vision, picture,
blocks), i.e. the reference's slot 11 itself, after the advance, the order
expiry and the conversion (and after `prune_grenades`). `k_picture_units`
(one thread per env): the tick check (a skipped or doubled update is a
fault), the object serials, then `_track_units` over the store rows (=
`sim.units` order: bars at the first update a unit exists and again at the
first that sees it a zombie, death tick, motion), then `last_hit` over the
ring's hits with damage > 0 in emission order (the last wins).
`k_picture_teams` (one thread per (env, marine team)): the damage window
against the records BEFORE the update, the seen deaths (a kill whose
victim's record is of tick - 1), the memory, the sightings from the device
`visible` table and the live row, the object tracker (the dead forgotten, the
sight gate over every (mate centre, object tile) line through
`obs_sight::line_clear`, a first sighting drawing `OBJECT_UID_BASE + k`). The
first update (start_team_picture's) reads no events. THE BLOCK: one int64 row
of `PIC_W` words per env (`device_picture.layout(T, U)` and the kernels'
`pic_layout`, compared at every build): `REC (T, U, 15)` (a team's record of
store row r; the loadout as its table row), `UNIT (U, 15)` (bar as Q16 `hp_q`,
turned, death tick, motion, last hit with the cause's index), `DMG (T, 1024, 3)`
+ `DMG_HD (T, 2)` (each team's damage window as ONE FIFO ring of (victim uid,
tick, Q16 damage) in append order: every entry is appended at the current
tick and dropped by age, so the FIFO's prefix is what every per-victim deque
drops and victim v's deque is its subsequence in order), `OBJ` / `SEEN (T,
96, 2)` + counts and `OBJ_NEXT`, `SERIAL_NEXT`, the side arrays `SIDE_G (64)` /
`SIDE_C (32)` with their `KNOWN` counts, `PARAM` (memory_ticks, window_ticks,
the env's team count, from the reference's own picture object), `TICK`; 7565
words (60 KB) per env at T = 2, U = 12. CAPACITIES (refusals, never clamps):
records one per (team, store row) and the tracker live objects only (96 =
the two tables' capacities), both bounded by the law, so neither can
overflow; the damage window 1024 per (env, team), no law bounds it (peak 75
over the corpus); a fault word `(N,)` (unknown uid, damage / tracker overflow,
skipped tick, sight range, serial, teams) rides the conductor's ONE books read
(`books_defer("picture_faults")`, raised by name with the env:
`ConductorOverflow` for an overflow, else `ConductorFault`), read directly
only at the build. THE SERIAL: the host tracker is keyed by object identity
and the tables compact, so every grenade / charge row carries a serial in a
SIDE ARRAY of the picture's block that the tables' ONE stable compaction
carries (`ObjectTable::set_side`; `k_table_compact` moves side[i] with row i
and recounts KNOWN, the rows already numbered); rows only ever append at the
end, so the update numbers rows `KNOWN .. count` from `SERIAL_NEXT`, grenades
first. No table column changed (no digest moved). On the host the matching
notion is test-side only: an object's POSITION in `projectiles +
planted_charges` (`device_picture.host_form`); the gate compares trackers by
position, identity-free. THE VECTOR: the 13 columns the table gives P8.3
(`hp`, `vel_x` / `vel_y`, the five `hit_*`, the four record counts) by
`bp.obs_picture_vector` beside P8.1's block (`hp` as `float(hp_q) / 65536`
over the bar's same view; `_hit_from` with `__dsqrt_rn`). THE SEAM for P8.9b:
`DevicePicture.restage(e, picture | form)` through ONE conversion
(`block_from_form`, whose `device_form` is the form again: a CPU test holds
the round trip every tick of a ship_demo match). ORDER DEPENDENCES, each
reproduced: `sim.units` = store row order (per-unit trackers, independent
anyway); `tick_events` = the ring's emission order (last hit, damage-window
append order); `killed` a sorted set and the memory loop over `sorted(recs)`
(deletions, order-free); `visible_by_team` (per uid, order-free);
`live_objects` = projectiles then charges in list order = grenade rows then
charge rows (the id draws and `seen_now`); `_objects` / `_damage` dict
orders reach no value. No order the device cannot know. Choices where the
design was silent: the build takes the device's own first update, except an
env whose hand state comes after the reference's first update (the check's
hand matches), which is restaged from its twin's host picture; the faults
are a books-read entry, not a separate sync; the picture runs for an ended
env too (cheap, defined). Findings: (1) the damage window's float sums are
EXACT (every damage is on the Q16 grid and the sums stay below 2^37), so the
order the reference adds them in cannot move a value: the `dmg_sum` break
(the device sums reversed) stays green, and the order is held instead on the
window's STATE (`dmg_order`, the tick's hits fed in reverse, goes red); (2)
`team_picture`'s docstring says a dealt hit needs "a living non-zombie unit
of the team", the code does not test `alive` (the device twins the code);
(3) the corpus never re-saw a lost grenade, so the door hand match
(`range_door`, both teams idle, the units placed at tick 0 either side of the
open door: a frag then a long-fuse smoke, the frag's detonation compacting the
smoke down a row while both teams track it, the door shut and opened by
hand); a hand state at tick 0 must retake the spawn's vision cache and
picture (`vision.state_for` caches per tick). Gates (`tests/cuda_obs_check.py`,
`PATCH = "P8.3"`): the picture's WHOLE state equal to the reference's at
EVERY tick, field by field, and `damage_last_s` raw-bit equal for every
(team, unit), on bots (1350 ticks), bots_v1 (273), range_door (201), the
hand match (321), range_hall (33), hall_cover (27), ship_all_round (481), the
door match (151), the batches N = 2 / 4 of ship_demo (to tick 1664) and the
room, and part RESTAGE (env 1 of 2 scrambled and restaged at tick 60, both
equal over 321 ticks); coverage: records created 53, refreshed after a gap
13, expired 4, deaths seen 27 / unseen 1, conversions with a record 2,
grenade-ticks tracked 1554, lost 5, re-seen 1, carried through a compaction
37, charge-ticks tracked 294, damage entries added 767 / dropped 617. The 13
columns raw-bit equal at every decision tick and non-constant over
(a)+(b)+(d). Breaks, each red: `early_picture` (bots_v1 tick 271, the turned
unit's zombie record), `lifetime` (bots tick 663, a record kept a tick),
`dmg_order` (bots tick 439, a victim's two hits of one tick), `serial_row`
(at the first compaction under a tracked object: door match tick 30, bots
439, hand 74), and P8.0 to P8.2's unchanged. 2.00 syncs per steady tick at
N = 1 and 2. Cost (`tools/bench_conductor.py --obs`, ship_demo, GPU time of
one update for all envs): 0.021 ms per tick at N = 1, 0.021 at N = 64, 0.023 at
N = 256; the blocks with the picture's columns 0.026 / 0.036 / 0.046 ms per
decision (P8.2's 0.030 / 0.018 / 0.045). No golden re-baseline; no host value moved;
no table column added.

**P8.4 AS-BUILT (2026-11-21, branch `29-p84-obs-list`, three parts by three
agents in turn).** THE LIST AND THE FOUR MASKS, each block every env on its
decision tick on the tick stream, no sync, the host builders the reference.
PART A (P8.4a) -- `DeviceObservation.aims` over `bp.obs_aims`
(`cpp/src/cuda_obs.cu`, five launches): the fire lanes of every marine team
(`lane_grid (N, T, GH, GW)` int32, draw_lanes' codes: -1 nobody, an owner's
uid, -2 several; `GH, GW = h // cell + 1, w // cell + 1`, the cell and half
width obs-table rows `LANE_CELL` / `LANE_HALFWIDTH` from
`TeamKnowledge.lane_dials`), each owner's own lane (`lane_own (N, U, GH,
GW)`), the served `near_u` plane `lanes`; `aim_of` per (team view, store row)
(`aim (N, T, U)`: a uid, -1 nobody, -2 not asked: the hit does not depend on
the observing soldier); per (soldier row, unit its team sees with a record)
the line of fire (`lof`: ASKED | SOLID through P8.2's `solid_in_way` to the
RECORDED centre | MATE through `mate_near_segment`) and `ticks_to_within`
(`ttw`, iterating `unit_lib::turn_law_q16`). The device helpers live in
`cpp/src/cuda_obs_sight.cuh` (`lane_div` = Python's flooring `//` through
`floordiv`, `lane_segment`, `lane_steps`, `lane_cell`, `lane_claim` (the
atomic several-owners rule), `AimKey` / `aim_less` = aim_of's (d2, uid, cx,
cy) order). The answers are compared with the reference's OWN intermediate
values at every decision tick through a test-side tap (`AimTap` wraps
`build_list`, `aim_of`, `solid_in_way` / `mate_in_way`, `_enemy_geometry` +
`ticks_to_within`, `TeamKnowledge.lanes` / `lane_of` for one `observe_rows`
and restores them; the builder unchanged). PART B (P8.4b) --
`DeviceObservation.build_list` over `bp.obs_list` (`cpp/src/cuda_obs_list.cu`,
one thread per (env, row, block)): the five blocks at fixed places (mates S - 1,
seen 16, records, objects, doors), every entry column the table gives P8.4b,
`list_mask`, `list_uid`; the later patches' columns (`on_next_tick`,
`prev_aim`, the order block 47-51) written 0. THE SORT: every block's key is
(squared centre distance, id) with an id unique within the block (unit id,
object uid, door ordinal), so the order is TOTAL and the card takes it by
successive minima (`next_min`, `key_less`) up to the block's capacity = the
host's sort then `[:cap]`; no float, no tie left. The row predicates
(`env_due`, `is_mate`, `is_seen`) moved into ONE header,
`cpp/src/cuda_obs_rows.cuh` (P8.4a's and the picture's copies retired); the
obs table gained an AMMO block (OBJ_KIND, RADIUS, TRAVEL_SPEED_BITS per
`[ammo.*]` row); the wired flags are a static `(N, wired_cap)` per door table
row (`device_obs.wired_values`). PART C (P8.4c) -- `DeviceObservation.
build_masks` over `bp.obs_masks` (`k_masks` in `cuda_obs_list.cu`, one thread
per (env, row), after the list of the same tick): `mask_feet` /
`mask_trigger` all legal, `mask_aim` the wind-up's KEEP alone
(`throw_release_tick >= tick + decimation - 1`, the decimation read from the
env's parameter row) or the 16 tokens + "walk" + the list's entries with
mask set and type in `AIMABLE_TYPES` (a generated bit set), `mask_hand` ONE
unit-library gate per bit (`can_reload`, `can_melee(.., at_tick_start =
true)`, `can_shove`, `can_plant`, `can_throw` per kind; the flashlight 0),
a row that is not valid `default_masks`; faults (`mask_faults`: a gate's
index past its table, a live door's span off the tile table) read by the gate.
THE LIFT: `verb_gates._reach_key`'s geometry was a function local to
`cpp/src/cuda_unit_verbs.cu::use_door` and restated in the list's
`door_entry`; it is now ONE library function, `unit_lib::reach_key(row, ty,
tx)` (`cpp/src/cuda_unit_lib.cuh`: a span tile's row-major index over the
footprint plus its one-tile ring, -1 outside), called by `use_door`'s two
passes, the list's `in_reach` and the use mask (`door_in_reach_any`: acts()
and a live door row with a span tile at `reach_key >= 0`, which is exactly
`can_use`: `use_door` returns a door iff one is in reach). Choices where the
design was silent: the per-tile function is the lift (the verb kernel's
per-tick span rows and the observation's door / tile tables are different
layouts; the tile is what both share); the masks live in the list's TU (they
read its buffers; one more launch); the mask faults are their own word, not
on the books read (like the list's and the aims'); the mask coverage pools
the aim's entry positions (the mate / object / door places are never
aimable by the spec) and exempts by name the bits the spec fixes (stand, the
first trigger option and "nothing" always legal, the flashlight always
masked). Findings: (1) `tools/bench_conductor.py --obs` timed the blocks on
the tick it stopped at (70, off the decision cadence of 4), where every kernel
exits at `env_due`: the P8.1 to P8.3 "obs blocks per decision" figures timed
early exits; the bench now times every block with each env due (the
parameter row's decimation set to 1 for the timing and put back) and reports
`obs_due`; (2) the host computes the per-enemy columns only for the first 16
seen enemies after the sort, the device buffers hold every seen enemy (the
served values agree); (3) no corpus level has a hull door or a wired door,
the bots never set throttle > 0, never take an order, never plant a remote
charge and never put a seen enemy within the frag radius of a mate, and no
played match puts a seen enemy past the ticks_to_within cap: two hand matches
(corpus (d)) carry them: `slow_turn` (range_door, idle, unit 1 facing away
with hook TURN = ONE/8) and `list_hand` (ship_demo all_round, idle, 24
ticks: a door made hull and wired, a remote charge, an order with a priority
target, half throttle, the frag radius). Gates (`tests/cuda_obs_check.py`,
`PATCH = "P8.4c"`, raw bits at every decision tick, every earlier column
still equal): part GATE N = 1 on bots 338, bots_v1 69, range_door 51, hand
81, range_hall 9, hall_cover 7, ship_all_round 121, door_grenade 38,
slow_turn 11, list_hand 7 env-decisions, the masks equal on their first
build (as were part a's and b's blocks); the intermediates equal to the tap
(bots 676 team lanes / 2126 aim_of / 1098 lines of fire / 1100 ticks;
bots_v1 137 / 850 / 918 / 1168; ship_all_round 242 / 1155 / 748 / 748);
coverage: lanes crossing a mate 108, cells of several owners 272, a line of
fire blocked by a wall 792 / a cover crate 2 / a mate 1983 / clear 1889,
aim_of nobody 3532 / a mate 786 / another side 705, ticks 0 1033 / between
2094 / at the cap 11; list entries mate 11 180, seen 3 222, record 855,
grenade 1 043, charge 352, door in reach 1 656 / out of reach 15 211, a
block over its capacity 4 207 (doors 4 167, objects 40; the host truncates
after the sort and so does the card); every mask bit the spec does not fix
both set and clear over the corpus (e.g. reload 578 / 7 046, use 1 656 /
5 968, an aimable entry 3 956 / 292 220 entry places), the KEEP-only aim 31,
a default row 3 227. Census over (a)+(b)+(d): every owned column, plane and
mask non-constant. Batches ship_demo N = 2 / 4 (493 / 805 env-decisions) and
the room N = 2 / 4 (112 / 220) all equal; 2.00 syncs per steady tick at N = 1
and 2. THE WALL extended at each part: the unseen enemy's row perturbed (its
trigger flipped too) moves none of the team's lanes, lanes plane, aims, lines
of fire, turns, list entries, list mask, uids or four masks (cone 7,
all_round 6 (team, tick) cases). Breaks, each red at its first decision:
`lane_trunc` (C's truncation for the floor division: bots tick 420, team 1's
grid cell (5, 15)), `lane_wide` (the lane one tile wide: bots tick 424),
`list_tie` (the sort ties by the higher id: bots tick 4, list_uid 1 vs 2),
`list_cap` (the door block one short: bots tick 4, entry 38), `mask_swap`
(the reload and shove gates swapped: tick 4 of every match, `mask_hand` row 0
bit 1), `mask_reach` (the use reach one tile wider: bots tick 4, row 3's use
bit; range_door tick 152); P8.0 to P8.3's unchanged. The lift is a Class-P
change to the verb kernel: `tests/test_cuda_unit_verbs.py` (the two-door
scene, the door in reach / none / dead / hull census), `tests/
test_cuda_unit_doors.py` and `tests/test_cuda_conductor.py` green unchanged.
Cost (`tools/bench_conductor.py --obs`, ship_demo, every env due, GPU time
for all envs): the list + masks 0.265 / 0.452 / 0.877 ms per decision at
N = 1 / 64 / 256 (the list's successive minima are serial per (row, block),
one thread each; not yet profiled further); the whole
decision's blocks 0.357 / 0.610 / 1.237 ms; vision 0.012 / 0.015 / 0.030 and
the team picture 0.020 / 0.021 / 0.021 ms per tick; 2.0 syncs per tick at
every N. No golden re-baseline; no host value moved; no table column
added.

**P8.5a AS-BUILT (2026-11-21, branch `29-p85a-obs-windows`, stacked on
P8.4's head `39024b43` while P8.4's card gate ran).** THE WINDOWS: the 31
rows the column-to-patch table gives P8.5a, no row moved to another patch.
`DeviceObservation.build_windows` over `bp.obs_planes`
(`cpp/src/cuda_obs_windows.cu`, four launches on the tick stream, after the
list and masks of the same tick, every env on its decision tick with its
ended latch clear, no sync): `k_far_raw` (one thread per (env, block):
FieldPlanes.far_raw into `far_raw (N, 8, hb, wb)` float64 -- the block MAX of
the first six planes, the wind pair of the block's FIRST largest load, tiles
row-major, the pad 0.0), `k_planes` (one thread per (env, row, cell): the
eight near planes at the tile and the eight far planes at the cell's block),
`k_sight_units` (the same grid: `watched`, `sees_target`, far `mates` /
`enemies` / `records`) and `k_air` (one thread per (env, row): the vector's
`air[0..7]` and `felt_temp`). THE SOURCES: every plane the windows read is
RESIDENT (`temperature`, `fire`, `gas`, `atmosphere`, `wind_x` / `wind_y`,
`heat`, `rad_flux_sweep`, `solid`, `sight_screen`: `device_obs.WINDOW_PLANES`,
dtype checked at the build); none had to be made resident or derived. The raw
values are the host's dequantizes (a float64 divide by the `*_fixed` module's
own currency, generated as `OBS_Q16_ONE` / `OBS_TRACE_ONE` / `OBS_FIRE_ONE` /
`OBS_ATM_ONE`), the wind `wind_load_plane`'s W (the square in int64 `>> 16`,
the product float64, 0 on solid), `floor(floor(W / 65536) * wind_k / 65536)`,
the direction `vx / sqrt(vx*vx + vy*vy)` with `__dsqrt_rn`, all through the
kit unfused; then `physics_value` per soldier (its environment profile row,
`[env_profiles.*]` else its species', its `resist_mult[POISON]`, the three
densities, `float(mass_g) * float(cap_q)`; `_div`'s zero rule), the squash
(CUDA's `log`) and ONE cast (the hand double -> float16 for the planes, float32
for the air block). THE OBS TABLE gained what P8.1 left: `ZOMBIE_RESIST_POISON_BITS`,
`WIND_K_Q` (the sim's impulse table's) and a `SPECIES_RESIST_POISON` block
(each species' mitigation's `resist_mult[POISON]` bits). THE FELT TEMPERATURE
is the unit library's `felt_temperature_q` over the device tables (the heat
row's own law, already on the card); the build refuses a batch whose device
tables' `heat_gain_q` / `heat_ambient_q` are not `exchange.heat_dials_q` of the
episode's `ObsContext` dials (`check_window_sources`), so no Q16 dial is
carried twice. THE REDUCTIONS: no float sum or mean over a block exists in
P8.5a's planes (the max and the argmax are order-free comparisons; the counts
integer; `wall_share`'s count / 16 is P8.0's), so numpy's summation order
never arises; the argmax's FIRST maximum is load-bearing (ties at W = 0 on
solid / still tiles carry different directions: `win_argmax` goes red at tick
4). SEES_TARGET needs the decoder's aim: `DeviceObservation.aimed (N, rows)`
int32 (-1 none), read through the list just built exactly as `build_list`'s
1h tail finds the entry (the first unit-block place with that uid, a place
past the mates' block, the team's record of it); P8.6a's decoder will write
that buffer. Until then the check's `aim_rule` (row i aims at choice
`(tick // 4 + i) % n` of its team's record uids plus one mate, from the second
decision) is handed to the reference's `observe_rows` and written into
`aimed` before the tick that builds. `watched` is `obs_sight::window_cell`
FROM each present record's centre (the same call P8.8's `full_watched` makes
over its own sources). Choices where the design was silent: the aim stand-in
above (the row stays P8.5a's); a fault word `win_faults` (sight range, the
felt range, a profile row past its block), read by the gate, not on the books
read; the full view's reuse is the device function, not a shared kernel.
Gates (`tests/cuda_obs_check.py`, `PATCH = "P8.5a"`, raw bits at every
decision tick, every earlier column still equal): part GATE N = 1 on bots 338,
bots_v1 69, range_door 51, hand 81, range_hall 9, hall_cover 7, ship_all_round
121, door_grenade 38, slow_turn 11, list_hand 7 and the new FIRE HAND MATCH 16
env-decisions (`fire_hand`: range_door, both teams idle, a fire started under
unit 0 at tick 0 through `debug_keys.debug_ignite` -- no played match lights a
fire near a soldier, so `fire` and `air[1]` were constant without it), every
window plane equal on its first build; batches ship_demo N = 2 / 4 and the room
N = 2 / 4 all equal; 2 syncs per steady tick at N = 1 and 2. Census over
(a)+(b)+(d): every owned plane and column non-constant. Coverage: watched tiles
186 146, tiles seeing the aimed target 115 337, near windows over the left /
right / top / bottom edge 1 099 / 690 / 133 / 114, far cells on a block
straddling the map's edge 65 192, far cells off the block grid 1 471 031, an
aimed entry found 806 / not found 3 320. Cells served past the squash's knee
(the log path): near temperature 250 964, poison 9 453, pressure 22 871,
teargas 1 791, wind_x 6 390, wind_y 6 087; far temperature 155 876, poison
2 136, pressure 2 888, teargas 556, wind_x 1 908, wind_y 1 479; fire (at most
1) and smoke never pass it. `log` moved 0 served values; no reduction ever
differed. THE WALL extended: the unseen enemy's row perturbed moves none of the
team's window planes, air block or felt temperature. Breaks, each red at its
first decision: `win_anchor` (bots tick 4, far temperature), `win_half` (bots
tick 72, far pressure: 56 cells, 0x3249 vs 0x324a), `win_divisor` (each gas
plane over the neighbouring density row: bots tick 440, air[4]), `win_argmax`
(bots tick 4, far wind_x); `win_profile` (the profile one row further) CANNOT
go red: today's config holds one env profile (`default`) and one species
(`human`) with the same three floats, so every neighbouring row a soldier can
read is equal -- `win_divisor` carries the "wrong divisor row" property on
rows that differ. Cost (`tools/bench_conductor.py --obs`, ship_demo, every env
due, GPU time for all envs, the card shared with another run): the windows
0.234 / 2.157 / 7.207 ms per decision at N = 1 / 64 / 256; all the decision's
blocks 0.706 / 3.031 / 8.354 ms; vision 0.021 / 0.021 / 0.040 and the team
picture 0.034 / 0.045 / 0.065 ms per tick; the whole observation 0.927 / 3.295
/ 8.774 ms per decision (blocks + 4 ticks of vision and picture), i.e. 0.034 ms
per env per decision at N = 256 against 0.70 ms per env-tick of the
conductor's tick; the windows dominate the observation (86 % at N = 256),
most likely `watched`'s walk per (record, cell), not yet split per kernel.
No golden re-baseline; no host value moved.

**P8.7 AS-BUILT (2026-11-21, branch `29-p87-obs-reward`, stacked on P8.5a's
head `d470269e` while P8.4 / P8.5a's card gates ran).** THE DEVICE BOOKS are
`simulation.device_books.DeviceBooks` (`DeviceObservation.books`) over
`cpp/src/cuda_obs_books.cu` (`bp.obs_books`, ONE THREAD PER ENV, one launch on
the tick stream): `rl_env.EnvRunner.step`'s loop body line for line -- the
step's accumulators zeroed on a step's first tick (`start_step`), the old
counters over the event ring (`count_events`, the rows ended before the tick
frozen), `RewardBooks.end` (the walk over the ring in emission order through
`_settle`, the turnings and the mismatch check per store row, the snapshot,
the team part from the standing in store row order, the eliminations, the
flush), then `MatchRules`' end (fewer than two marine teams alive or the limit
tick), the result and `finish`, and on a decision tick or the ending tick the
SERVED arrays (`reward` (N, U, 5), `counters` (N, U, K) and the three team
blocks float32, `rows_ended`, `terminated`, `truncated` uint8: the shared
memory's own dtypes and casts) in `DeviceBooks.served`. WHERE: the walk's `11'`
line after the device team picture (whose bars they read) and after the tick's
blocks (`DeviceObservation.stage`: vision, picture, blocks, books), so the
ending tick's blocks are built as the reference builds its last observation.
THE BLOCK: one int64 row per env (`device_books.layout`, compared with the
compiled `obs_books_layout` at the build): the unit snapshot (`_units`), the
standing (H, A per team, the zombies' H, the teams' dict order kept as the
order a team first appears among the living non-zombies: `balance` and
`sum(H.values())` add in that order), living / eliminated teams as bit sets,
the rows' ended / pending / step-ended flags, the tick's scratch, the step's
counters and raw, the old counters (float32 bits), rest / d_balance / outcome,
the walk's working hp and life, the scalars (finished, the ENDING TICK, the
result, terminated / truncated, the tick, the env's team count). THE WEIGHTS:
obs-table rows `BOOK_TERMS` (the profile's eight terms) and `BOOK_G`
(`reward_v1.group_weights` over `rl_env.COUNTER_NAMES`, (K, 5)), float64 bits
read from the reference's own objects (`device_books.table_rows`);
`DeviceObservation(..., reward_profile=, reward_overrides=)`, default
"fireteam"; "counters" REFUSED by name (`device_books.profile_terms`). NUMPY'S
ORDER (inventory K5, the experiment on numpy 2.5.3, kept as
`tests/test_device_obs.py` properties): the named fold
`(counters[:, :, None] * G[None]).sum(axis=1)` is SEQUENTIAL over the K
counters from +0.0 (0 mismatches on 900 random (U, K) cases of 12 / 64 / 6 rows
by 32 / 40 counters; all -0.0 products sum to +0.0, so the reduction starts
from the identity, not the first term); the old counters' `a[r, c] += dmg` on
float32 is NEP 50 (the damage cast to float32 once, a float32 add: 0
mismatches in 200 000, 46 219 against float64-then-cast); the "counters"
profile's last-axis float32 sums are pairwise (11 635 of 24 000 rows differ
from a serial fold at K = 18): refused, as the design's version 2 decided.
EVENTS: the ring carries every event the books read (it is `apply_packet`'s
record of every `UnitHitEvent` / `UnitKilledEvent`, heals included: 250 heal
hits over the corpus walked equal); the claims are already folded into each
event's attacker; no event was missing, no ring record added, no unit kernel
touched. THE MISMATCH (`RewardBooksMismatch`): a per-env count in the books'
words, raised BY NAME with the env at the tick's one books read (kind
`reward_books`); the fault bits (a skipped tick, a team off the layout) as
`ConductorFault`. TERMINATION REACHES THE HOST with no third sync: the same
words carry each env's "ended this tick" flag (`DeviceConductor.ended_now`
after every tick); the served `terminated` stays a device array for the
trainer. THE ENDED LATCH: set on the card at the ending tick (after that
tick's blocks), `ENDED_TICK` kept for P8.8b's terminal observation; a set
latch freezes the env's books (and skips every later block); the host stops
stepping an ended episode, so the frozen books ARE its final books. Choices
where the design was silent: `begin` is the identity on the card (nothing
changes a unit between two conductor ticks, so the snapshot of the previous
end is the tick's start and no team can be eliminated between ticks; a reset
or a hand state restages the books with it -- P8.9b); the books keep every
named profile through the obs table, the gate runs "fireteam" on the whole
corpus and "range" on bots, the room and the clock match; the served arrays
live beside the observation buffers (`DeviceBooks.served`), not in them; the
standing holds at most 8 marine teams (`T_MAX`, refused past it). Gates
(`tests/cuda_obs_check.py`, `PATCH = "P8.7"`; every earlier part unchanged and
green): the books' WHOLE state equal to the reference's (`RefBooks`:
`EnvRunner.step`'s loop over `RewardBooks` and `count_events`, every named
profile) at EVERY tick as raw bits, and the served arrays at every step's end:
bots 1350 ticks / 338 steps, bots_v1 273 / 68, range_door 201 / 50, hand 321 /
80, range_hall 33 / 8, hall_cover 27 / 7, ship_all_round 481 / 120,
door_grenade 151 / 37, slow_turn 41 / 10, list_hand 25 / 6, fire_hand 61 / 15
and the new CLOCK HAND MATCH 49 / 12 (`clock_hand`: range_door, both teams idle,
the time limit cut to 2 s: no played match ends by the clock); "range" on bots
(338 decisions), range_door (51) and clock_hand (13) equal; batches ship_demo N
= 2 / 4 (493 / 805 env-decisions, envs ending at 1664 / 300 / 970 / 275: the
ended envs' books, served arrays and latch compared FROZEN every tick after
their end) and the room N = 2 / 4 (112 / 220) equal. Coverage (fireteam, served
counters non-zero cells): won 11, eliminated 11, clock 2, bars_enemy 51,
bars_lost 356, kills_enemy 9, died 16, bars_zombie 44, share_bars 618,
share_enemy_bars 810, share_alive 47, share_enemy_alive 47, share_zombie_bars
384, bars_team 31, team_kills 4; ends by elimination 5, by the clock 1; results
5 wins, 1 draw; the walk met 23 overkill hits, 250 heals, 1 corpse hit, 2
turnings, 3 posthumous credits, 1757 uncredited hits. Census: every served key
non-constant but `truncated` (0 under every named profile by design 6.5,
exempt by name). Breaks, each red: `bk_order` (the tick's events walked in
reverse: bots tick 452, `rest[0]` one ulp; bots_v1 tick 204, a row's
`bars_lost` -- the order matters through the clip at zero health and the
float sums of non-dyadic bar ratios, and the gate holds it), `bk_weight` (the
terms read one row further: bots tick 419, `d_balance[0]`; clock_hand tick
48, `outcome[0]`), `bk_late` (termination one tick late: the `won` / `clock`
counters missing at every ending tick), `bk_freeze` (the books not frozen:
batch ship_demo N = 2 env 1 at tick 301, N = 4 env 3 at 276, the room N = 4
env 3 at 209). 2.00 syncs per steady tick at N = 1 and 2 (`max_n_sub`, the
books read). Cost (`tools/bench_conductor.py --obs`, ship_demo, GPU time for
all envs, the card shared with another run): the books 0.063 / 0.141 / 0.152 ms
per tick at N = 1 / 64 / 256 and 0.143 / 0.343 / 0.396 ms on a tick that serves
a step (the reward fold and the casts), i.e. about 0.33 / 0.77 / 0.85 ms per
decision of 4 ticks against the whole observation's 0.698 / 2.835 / 8.527 ms;
the serial walk per env dominates at small N. No golden re-baseline; no host
value moved; no table column, ring record or unit kernel added.

**P8.6a AS-BUILT (2026-11-21, branch `29-p86-obs-decoder`, stacked on P8.7's
head `ec380021` while P8.5a / P8.7's card gates ran).** THE DEVICE DECODER is
`simulation.device_decoder.DeviceDecoder(obs, specs)` over
`cpp/src/cuda_obs_decoder.cu`, attached to the conductor
(`DeviceConductor.attach_decoder`). THE DECISION: `set_decisions(actions,
envs=None, list_uid=None)` -- the trainer's ONE device write per decision
(CuPy, a torch tensor by DLPack, or numpy) copied into the decoder's `(N,
rows, 5)` int32 buffer, then `bp.obs_dec_check` (one thread per env, the
rows serially): `check_actions` as per-env fault words (the count, per head
the first row and its value), `masked_choices` into `DeviceDecoder.masked`
`(N, rows)` uint8 against the mask buffers AS THEY ARE AT THE CALL (the
served masks of the observation the decision answers), the pointer to a
game id against `list_uid` at the call, `set_decisions`' fallbacks
(flashlight -> nothing, a throw onto a non-unit aim -> nothing, a plant
"onto" -> nothing). The pointer is resolved BEFORE any tick, so the next
observation (built at the decision's last tick) cannot overwrite what it
reads: no copy of `list_uid` is kept (the `list_uid=` argument is P8.10a's
double buffer's door and the check's break). An out-of-range decision is
raised at the next tick's ONE books read (kind `decoder_faults`) as the
host's `ValueError` with `check_actions`' own message (the conductor's slot
prefix in front), `.env` set. THE DECODE: `bp.obs_decode`, one thread per
(env, row) of a decoder-driven team, at the walk's HEAD (call line `0
decode`, inside the books deferral, before slot 1; the reference decodes in
`handle_frame` before `sim.step`): `AgentPolicy._decode` line for line over
a per-(env, row) int64 STATE RECORD (`device_decoder.STATE_FIELDS`: the
decision -- feet, the aim as kind / value / index with its `last_point`,
trigger, hand after the fallbacks, arg, HAVE, FIRST -- and the slot -- the
held throw aim, which may ALIAS the decision's aim object (the host's
`st.throw_aim = d.aim` shares one `_Aim`, so a `last_point` refreshed
through either is the other's; the check un-aliases it, copying the
decision aim's fields, when the next decision replaces the object),
`last_vec`, `aimed_uid`, `prev`, the uid). The aimed unit's point is the
DEVICE team picture's record of it (live while seen, last seen otherwise),
`aim_vector` and `onto_reach_q` in float64 through the kit (`__dsqrt_rn`, the
reciprocal, two products, `quantize_scalar`'s half-away rounding). Each
issued intent is written straight into its `LATCH_COLUMNS` column in the
store's encoding (`live_move_dir` (set, dx, dy, throttle) with the
throttle clamped as `set_move_dir` clamps, `live_aim` (1, dx, dy) every
tick, `live_trigger`, `flashlight_on` = 1 every tick (the OFF sense of a zero
column), and on the decision's FIRST decoded tick only the latch kinds:
`pending_reload` / `melee` / `shove` / `use`, `pending_plant` (the nonzero
gate, then the plant row's clamp), `pending_throw` (1, the facing vector,
the fuse, the reach, the kind) behind the vector gate); the FIRST flag is
consumed on the policy's next decoded tick whichever of its units live, as
the host's `_first`. `DeviceObservation.aimed` is written every decoded
tick. No verb lacked a device path: every `VERB_RULES` kind is a column
write (the commander's `order_intent` is not the decoder's: P8.6b).
`device_decoder.constants` refuses a `VERB_RULES` the kernel does not issue.
THE OBSERVATION'S COLUMNS: `bp.obs_dec_obs` after the windows on a decision
tick (`DeviceObservation.build_decoder_obs`): the vector's `prev[0..5]` (-1
before a slot's first decision) and, on the first unit-block entry holding
`aimed`, `prev_aim` and `on_next_tick` (the record's centre, `aim_vector`,
`turning.predict` through `unit_lib::turn_law_q16` with the soldier's own a
/ w / pending spin -- one copy, `obs_rows::turn_dials`, now also P8.4a's
`ticks_to_within`'s -- within its shot cone); with no decoder attached a zero
state (`NullDecoderView`) and the check's stand-in `aimed`. The owners table
gives P8.6a eight rows (prev[0..5], on_next_tick, prev_aim), not nine. THE
HORDE: `bp.obs_dec_horde` at the same tick head writes the zombie
CONTROLLER's latch (SET with a flag per living zombie row, EMPTY when none
lives; an `idle` horde EMPTY) into `DeviceDecoder.horde_rows`, handed to slot
5 device to device (`unit_kernels.horde(latch=None)`: no host list, no
upload). K8 ANSWERED: the controller's latch is NOT the conductor's own
CONDUCTOR call on a tick with no living zombie (EMPTY returns before
`HordeState.begin_tick`; the conductor's call still drops the field cache at
a refresh tick or a topology change), so a horde emptied and refilled by a
rise off a refresh tick can walk a field the conductor's call would have
dropped; `Simulation.drive_zombies`' docstring ("reproduces the conductor's
call exactly") holds only while some zombie lives. The card serves the
controller's latch (the reference the training path plays), gated equal to
`horde_cache.latch_of` taken after the controllers' `handle_frame` every
tick; corpus (c) holds 455 such empty-horde ticks. THE TWO WAYS OF DRIVING:
with a decoder attached every marine team must be `policy` (`DecoderRefused`
by name at the build) and the horde `zombies` / `idle`;
`DeviceConductor.latch` refuses a `LATCH_COLUMNS` intent naming a
decoder-driven unit and `drive_zombies` for a card-served horde by name; the
tap still carries the commander's orders (the order table). Without a
decoder the conductor is as before (the tap-driven corpora). THE RECORD (old
P8c): the decisions buffer and the state record are what a device episode's
record keeps (`DeviceDecoder.decisions`, `.state`). CORPUS (c), GENERATED
LIVE in the CUDA subprocess (`tests/cuda_obs_check.py` `PolicyDriver`): an
in-process `rl_env.EnvRunner` (run file parsed from a dict: every team
`policy`, v1 observation and actions, "fireteam", the commander on), reset
once, its session played tick by tick as `EnvRunner.step` drives it (each
policy handed the masked-random decisions with the `list_uid` they were
drawn on; the controllers' `handle_frame`, the reference's latch columns /
slot states / horde latch taken, `sim.step`, the digests); decisions are
`tools/rl_smoke.py`'s drawer (each head uniform among its legal options) plus,
per row with p 0.04, one head forced to a random in-range index and the
three fallbacks forced now and then (masked choices: 253 counted). A hand
kit at tick 0 (every marine with 3 of each of the six hand-grenade kinds,
the loadout row to match) and, on five ship_demo runs, a CROWD (every unit,
both teams and the horde, placed two tiles apart around team 0's first
marine, vision and picture retaken), so enemies are seen and aimed at and
zombies bite, die and rise from the first decisions. Gates (`PATCH =
"P8.6a"`; part GATE runs corpus (c) with every earlier corpus, part DEC_GATE
it alone): the store's latch columns after the decode, every slot state, the
horde latch and `(field_digest, tick_digest)` equal to the reference at
EVERY tick, `masked_choices` equal at every decision, every owned
observation column raw-bit equal at every decision (the earlier patches'
too, now under real decisions): crowd 102 / 104 / 105 / 110 / 116 (97 / 145
/ 105 / 109 / 78 ticks), ship_demo 133 (308), range_door 83 / 87 (209 /
246): 1297 ticks, 330 env-decisions, 7 462 slot-ticks. Coverage (each > 0):
all 49 feet options; aims at a seen enemy 119, a record 33, a token 899,
walk 55, keep 251; trigger 697 / 663; reload 63, melee 53, shove 179, use 85;
every throw kind with every reach (arg 4 "onto" 1 to 3 per kind); plants of
each timer 6 to 10; the fallbacks (flashlight 26, onto without a unit aim 29,
plant onto 37); a throw aim held through the wind-up 1 523 slot-ticks; an
aimed unit lost 9 / dead 50 slot-ticks; zombies dying in a tick 7, rising 4.
Part DECODER: a bots team refused at the build; an out-of-range hand
raised at the books read with the host's message; a tap `set_aim` for a
decoder unit refused; 2.00 syncs per steady tick at N = 1 and 2 with the
decoder (`max_n_sub`, the books read) and NO host-to-device upload from
Python in a steady tick (the decisions are one write per decision; the
horde latch's per-tick upload is gone for a card-served horde); the card's
`aim_vector` / `onto_reach_q` equal the host's on every offset |dx|, |dy| <=
400. Breaks, each red but one: `dec_every` (the hand latch on every tick of a
decision: tick 1 of every run, `pending_throw` of row 0 after the decode),
`dec_late` (the decode after the walk: tick 1, the digests), `dec_horde_late`
(the horde latch at the previous tick's end, before the conversion: red
exactly at the rises, crowd 104 tick 9 and crowd 116 tick 56, a risen zombie
not driven), `dec_next_uid` (the pointer against the NEXT observation's
`list_uid`: tick 0 of the crowd runs, `live_aim` of row 3); `dec_div`
(`dx / mag` in place of `dx * (1 / mag)`) CANNOT turn red: on every offset
|dx|, |dy| <= 400 (641 601) the two forms quantize alike (the sweep is part
DECODER's), and the fused form of the squares is exact (integers below
2^26), so no input in reach orders the arithmetic differently in the
result. FINDING (pre-existing, not the decoder's): the
device DOORS stage (`cpp/src/cuda_unit_doors.cuh` over the topology seam)
REFUSES (ApplierFault at '9e(d) doors') a door opened or closed beside gas a
blast heated past the seam's proved range |T_abs| < 2^28 (e.g. ship_demo,
run seed 81, tick 32: door 33 opened by a use, its neighbour (17, 7) at T_abs
324 247 500 after five explosions the tick before), on the TAP path as on
the decoder's: 11 of 30 crowd / ship_demo seeds scanned hit it (the corpus's
seeds are chosen clear of it); a policy run will meet it. Choices where the
design was silent: the pointer resolved at the call (no `list_uid` copy);
the decoder's columns read `aimed` and the state, so the tap-driven corpora
keep the stand-in; one thread per env for the check (rows are few); the
fault words on the books read (no new sync); the decoder rides the
observation twin row (its decode at the walk's head, its columns with the
blocks). Cost (`tools/bench_conductor.py --obs`, ship_demo, GPU time for all
envs, the card shared with another run): the decode at the tick head (every row,
with the horde latch) 0.017 / 0.013 / 0.015 ms per tick at N = 1 / 64 / 256,
a decision's hand-over (the copy, the check, the pointers) 0.134 / 0.155 /
0.157 ms, the decoder's columns 0.008 / 0.007 / 0.007 ms per decision --
against the whole observation's 0.698 / 3.099 / 8.626 ms per decision and the
conductor's 13.2 / 52.1 / 180.8 ms per tick; 2.0 syncs per tick at every N. No golden re-baseline; no
host value moved; no table column, ring record or unit kernel added
(`unit_kernels.horde` takes `latch=None`).

**P8.8 AS-BUILT (2026-11-21, branch `29-p88-obs-fullview`, stacked on P8.6a's
head `f5a8531f`, unmerged).** THE FULL VIEW (`full_list`, `full_mask`,
`full_uid`, `full_match`, `full_watched`, the table's five whole keys) is
`DeviceObservation.build_full` over `bp.obs_full` (`cpp/src/cuda_obs_list.cu`,
beside the entry builders it calls; four launches on the tick stream at the end
of every decision build, no sync): `k_full_aim` (one thread per (env, ray, store
row): `FullView.aim` -- every other living unit at its live centre as a
candidate, (d2, uid, cx, cy) order, the cone of `_cone_of`, the first clear sight
line -- along the facing (0) and the facing target (1) into `full_aim (N, 2,
U)`), `k_full_list` (one thread per (env, row, block): the soldier itself (alive:
its mate entry; dead or turned: the corpse of 5.6, `age` from the picture's death
tick), the other living marines by (team, slot, id), the zombies, the live
objects (grenade rows then charge rows: `live_objects`' order; the uid the
object's place in the KEPT block) and the living doors by (d2, id); each block's
place follows from the counts, the room cut objects first, then zombies, then
doors, never a marine (a marine overflow is a fault)), `k_full_watched` (one
thread per (env, row, cell): `obs_sight::window_cell` FROM every living enemy's
live centre, zombies included, OR-ed) and `k_full_match` (one thread per (env,
row): the living counts, the standing H(t) summed in the full view's marine order
(slot, id) through `obs_pic::bar_ratio`, the horde awake / dormant, the clock).
WHAT IS SHARED, by calling: the list's `entry_common`, `put_angle`,
`unit_columns`, `mate_entry` (now taking the caller's aim answer and whether the
lane pair is written), `enemy_tail` (cols 27, 29-38, split out of `seen_entry`
with the mates' aims as a functor: the list passes `aim_of`, the full view its
every-unit aim), `object_entry`, `door_entry`, `next_min` / `key_less`; lifted
into `cpp/src/cuda_obs_rows.cuh` from `cuda_obs.cu` (one copy, the list and the
vector now call them): `aim_cone`, `bearing_q16`, `ticks_to_within`,
`ticks_toward` (`_enemy_geometry`'s ticks), `status_left`, `kit_of`
(`_kit_values`: the vector's kit columns and the full view's extra columns read
ONE function); `bar_ratio` lifted from the books' TU into
`cpp/src/cuda_obs_picture.cuh`. The full view's own: its every-unit aim, live
centres and motion, the line of fire to the live centre and `mate_in_way` per
entry, the extra columns (`full_extra`: knowledge, kit, the four statuses of
`FULL_STATUS_KINDS`, the order's kind) and the mate entries' order block (cols
47-51, `order_tail` over a static PLACE TABLE, `device_obs.place_values`: the room
graph's rooms by id and openings by `ORDER_OPENING_BASE + id` with their centres,
`_place_centre`'s first match). `DeviceObservation.team_rows` (the policy feed)
hands no full key. Choices where the design was silent: the full view lives in
the list's TU (its builders are that TU's); the order block of a full-view mate
entry is built here (the full list is one whole key the table gives P8.8), the
place table rides `DeviceObservation`; `tests/obs_column_owners.toml`'s patch
order is the LANDING order, so P8.8 / P8.8b now precede P8.6b (whose columns stay
unowned); a fault word `full_faults` read by the gate. Gates (`tests/
cuda_obs_check.py`, `PATCH = "P8.8"`, raw bits at every decision tick, every
earlier column still equal): part GATE N = 1 on bots 338, bots_v1 69, range_door
51, hand 81, range_hall 9, hall_cover 7, ship_all_round 121, door_grenade 38,
slow_turn 11, list_hand 7, fire_hand 16, clock_hand 13 and corpus (c) 330
env-decisions, equal on the first run; batches ship_demo N = 2 / 4 (493 / 805)
and the room N = 2 / 4 (112 / 220) equal; coverage: mate entries 15 894, enemies
the team sees 7 530 / holds a record of 2 154 / does not know of 31 210, zombie
entries 19 184, objects seen 10 871 / known 377 / unknown 3 200, door entries
349 580, corpse rows 1 352, rows past the cap 6 110, watched cells 728 703, order
blocks 4 582, mate aims on a unit 4 461; every `full_match` column moved (5 to 35
distinct values). THE WALL the other way round (part WALL): the unseen enemy's row
perturbed moves the team's full list in every case (cone 7, all_round 6) while it
moves no policy value. Breaks, each red at bots tick 4: `full_vis` (the full view
from the team's visible set: 2 586 values), `full_tie` (ties by the higher id:
door entry 53), `full_cap` (the room one short: entry 63). Cost: see P8.8b.

**P8.8b AS-BUILT (2026-11-21, the same branch).** THE TERMINAL OBSERVATION. The
reference (`rl_env._worker_main` "step"): when `EnvRunner.step` ends the match
(its tick loop breaks at the ending tick, on or off the decimation's phase), the
worker calls `r.observe` into the shared `final_` arrays of that env -- every
observation key, the masks, `valid` / `in_match` and the full view -- on the
session's state right after the ending tick, before the reset; the shared arrays
keep the last ending's values, and `BatchedEnv._final` serves them for the envs
that ended in the step, zeros on every other row. THE DEVICE FORM:
`DeviceObservation.terminal(ended_now)`, ONE call line in `DeviceConductor.tick`
right after THE ONE BOOKS READ (whose words already carry the books' ended-now
flags: no sync of its own), calls `build_terminal(envs)`: the very blocks of a
decision (`build`) over the `final_` buffers, with a parameter row whose episode
tick is each env's own tick (the cadence test passes off the phase; the
decimation kept, the masks read it) and a skip row clear for exactly the ending
envs (the latch the books just set is not read), both uploaded only on such a
tick; the live buffers untouched (the reference fills them with the next
episode's tick 0: P8.9's). An env's `final_` buffers hold zeros before its first
ending and its last terminal observation after it (the shared arrays' semantics;
the per-step zeroing of the other rows is `BatchedEnv`'s, P8.10a). The tick the
device serves is the reference's: the books end at the walk's end of tick k with
the tick row at k, `sim.tick` = k when the worker observes. Gates (`PATCH =
"P8.8b"`, at every decision tick and every ending tick of every GATE / BATCH run):
every ended env's owned `final_` columns raw-bit equal to its terminal reference
(the check's `host_observation` right after the ending tick; a tap-driven match
takes the aims the twin holds then, the stand-in of the last decision) and
unchanged at every later check, every env not ended all zeros, and at an ending
off the cadence every live buffer of every env unchanged by the tick. Endings
compared: part GATE 14 (5 on the cadence, 9 off; 13 by elimination, 1 by the
clock: clock_hand), zero checks 1 086, live buffers held 2; batches 12 endings (5
on, 7 off: ship_demo ends 1664 / 300 / 970 / 275, the room 222 / 222 / 218 /
208), frozen checks 1 213, zero checks 1 631, live held 5. Breaks, each red:
`term_early` (every tick's state built for the envs not ended: the final_ set
written before the ending, clock_hand / bots tick 4), `term_next` (built at the
next decision tick instead: clock_hand tick 48, bots tick 1349, `final_valid`
zeros against the reference), `term_other` (written for the next env too: batch
ship_demo N = 2 tick 300, env 0's final_ set not zeros). Unchanged: every earlier
part; 2.00 syncs per steady tick at N = 1 and 2 (parts SYNCS and DECODER).
Finding (the check's, fixed): the sync counts could pick up the DESTRUCTORS of
earlier parts' dead conductors (`cuda_unit_store.cu` / `cuda_applier.cu` /
... stream syncs) when Python's cycle collector happened to run inside a counted
tick -- the observation's extra allocations moved that moment into one (part
DECODER at N = 1, 3.77 per tick); both counting loops now collect first and
hold the collector off while they count. Cost (`tools/bench_conductor.py --obs`,
ship_demo, GPU time for all envs, every env due, the card shared with another
agent): the full view 1.956 / 2.434 / 6.010 ms per decision at N = 1 / 64 / 256
(inside the decision's blocks: 2.535 / 5.117 / 14.362 ms); a terminal build of
every env at once 2.630 / 5.220 / 14.475 ms (an ending tick only); THE WHOLE per
decision (blocks with the full view, 4 ticks of vision and picture, the books,
the decoder) 3.176 / 6.196 / 15.606 ms against the conductor's 11.4 / 49.9 /
173.3 ms per tick; the full list's serial blocks (one thread per (row, block),
60 doors by successive minima, a sight walk per enemy entry) dominate it, not yet
profiled further. No golden re-baseline; no host value moved; no table column,
ring record or unit kernel added.
**TOPOLOGY RANGE AS-BUILT (2026-11-21, branch `29-topology-range`, stacked on
P8.6a's).** P8.6a's finding closed: the device topology seam
(`cpp/src/cuda_topology.cuh`) refused a seal or an unseal beside gas past a
moving cell's |T_abs| < 2^28 (about 4096 game degrees), where the host's
Python ints carry on; blast-heated gas is legally hotter (reproduced at the
pre-patch build: corpus (c)'s crowd 100 at tick 112 and ship_demo 100 at
tick 48, ApplierFault at '9e(d) doors'; ship_demo 81 as named in the finding
did not reproduce on this stack). THE BOUND: the refusal moved from the
temperature to the product a moved parcel is priced with: a moving cell (a
sealed tile; an unseal's donor) must have N * (|T_abs| + 1) < 2^60
(`TERM_BOUND`), computed exactly in uint64 (N < 2^32, two non-negative
int32 planes; |T_abs| + 1 <= 2^31 + 2^28). Temperature itself is not
bounded at all: T is whatever the int32 plane holds (a seam write refreshes
the mirror without the rails, so between recoveries a mirror can sit above
T_MAX_PHYS; only the dtype's range holds unconditionally), |T_AMB| < 2^28
as before. Where the physics sits: after the recovery's rails |T_abs| <=
(T_MAX_PHYS + T_AMB_K) raw = (16000 + 290) * 2^16 < 2^30, and the engine's
own per-cell range is n_bulk < 2^30 (`bulk_transport.h`), so a cell at the
ceiling with the largest bulk is inside the bound (`T_ABS_PHYS_MAX`, a
static_assert at the current, PROVISIONAL dial); a mirror past the ceiling
up to the int32 top passes with N < 2^28.8 (hundreds of times the ambient
bulk). THE PROOF'S SHAPE: every term a cell gives in the call is < 2^60, a
cell takes at most four terms in and its own out, |E| < 2^61 (pre-checked
as before), a born-at-ambient E is < 2^60, so every store and sum fits
int64 and is the same integer as Python's. NOTHING WAS RESTATED: no
expression needed a 128-bit intermediate or a reordering; only the
pre-pass's admission test changed. THE RE-READ DONOR (a hole of the old
proof, found while re-deriving it): unseal reads a donor's T_abs live at
each span tile it borders and refreshes its mirror after each drain, which
can RAISE it (by the remainder of E / N over the smaller N); when the
donor's mirror is its energy (the iron rule's state) every later term is <
N0 * (|T_abs0| + 1), so the one product covers it; an accountable donor that
borders two or more span tiles and is NOT a mirror of its energy is bounded
by the plane's range, N0 * (2^31 + |T_AMB| + 1) < 2^60 (the old check let
such a donor through with any N < 2^32, where a later read could overflow:
red at the pre-patch build, the device wrote where it must refuse). The
gas-N primitives (inject / extract / transfer) keep `T_BOUND` = 2^28 on
T_abs: the vents' proof (`cuda_unit_vents.cuh`, the sibling branch's) leans
on the extraction's; destroy_wall and on_tile_changed never had a
temperature bound (no T_abs product; the solid books' product is checked in
128 bits). The host is pure Python (`GameMap`): no C++ CPU twin carries the
bound. GATE: `tests/cuda_unit_doors_check.py` part RANGE (both door scenes,
every door): seal and unseal with the moving cells heated through the energy
field at the old edge, just past it, T_MAX_PHYS, twice it, the int32 top
and bottom and six random draws over the plane's range, each op and its way
back equal to the host on every plane and book; bulk at its maximum on the
term edge (hot and cold) equal, just past it refused by name with nothing
written while the host carries on; the L span's re-read donor (a mirror
with the largest remainder at the plane's top and bottom and at max bulk:
equal; not a mirror: small bulk equal, bulk 2^30 refused); destroy_wall +
on_tile_changed beside cells at the extremes equal. Red at the pre-patch
build (the old admission test: dozens of refusals the host does not make,
and the re-read donor's write). `tests/cuda_obs_check.py` part RANGE_SCENES
(its own pytest test) plays crowd 100 and ship_demo 100 through the gate
(decisions and digests equal every tick). They are NOT in `DEC_RUNS`:
appending runs there moved part DECODER's sync count at N = 1 (64 syncs on
one steady tick, allocation sites; red with the two runs, green without, the
seam's build unchanged): that measurement depends on what the process played
before it, a fragility of the check, reported, not fixed here. No golden
re-baselined; no host value moved;
`.claude/rules/device-units.md` never stated the bound (unchanged).

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

As built (P4c, 2026-11-13, branch `29-p4c-edit-queue`): the composite key is
realised as the record ORDER -- the host sorts exactly as the CPU flush does
(`EditQueue.take_ordered`; `seq` already spans the previous tick's 9e because
the queue is cleared only at a flush) and packs that order, so the device
needs no sort and no counter; each edit row carries its `seq` as the noise
key. Records are two-level (`field_edit.pack_for_device`): an EDIT row per
edit (env, field ordinal, gas slice, mode, `amount_q`, den, noise low bound,
key, clamp flags + six bounds) through the CPU's own door, and a TILE row per
tile of its region (edit row, `r*w+c`, Q16 weight, scale) from the one region
walk, `_iter_region` -- so the device never re-derives a region. The device
(`cpp/src/cuda_edit_queue.cu`) runs K1 in parallel over every tile (the
contribution in 128-bit magnitude arithmetic, the Philox noise) and K2 one
thread per env over its tiles in order (the skip masks, combines and the
gas-energy seam: deposit, book, mirror refresh). Books: per env in the arena,
folded at the tail's one D2H into `edit_queue_books`, added by the runner into
the map's trace books, gas-energy channels and `MatchDraws.drawn`. Fallback
(the host flush at 6b, bit-identical): a `water_depth` edit (the resident
pre-step reads it on the host) or an edit past the device's integer range.
The entry upload set is unchanged: every field an edit touches has another
host writer between ticks (pumps, vents, topology, ignition, the heat clear),
so the device applies the edits on top of the uploaded planes.

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

**P3d CLOSED 2026-11-13 with no patch of its own.** Re-checked against main
`0e166d91`: the horde walks the field on every path (no A* call in the
zombie law, gated by `tests/test_horde_field.py`), the C++ flood equals the
Python reference flood, the law's `math.sqrt` gates are integer squared
compares (`ai_zombie._d2`), and the systems rows this section asked to amend
are amended (CLAUDE.md's A* row; the Match controller registry row names the
field law "the approved rewrite of the pre-G9p A* law, residency P3d";
`tests/test_zombie_control.py` re-pinned at G9p). The HUMAN-TEST was G9p's,
played by Erik with the gameplay pack. `test_horde_field`,
`test_horde_behaviours`, `test_zombie_control` and `test_pathfinding_parity`:
67 passed at `0e166d91`. Everything that remains is P7d's (above).

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

#### §7 P3e site table, re-inventoried 2026-11-12 (main `24b87481`)

The table above was written against the code of 2026-11-07. Re-taken by
reading every slot of `Simulation.step` on the CONTINUOUS ruleset (a
`MatchSession` with the `bots` controller and the horde, and the human's
direct control), function by function, at main `24b87481` (after P3a, P3c,
the gameplay pack #183 and #193). A "site" is a float operation whose result
can change a synced outcome (hp, a status, a position, a wall, a field, a
draw). Status: **DONE-already** (integer before P3e), **TODO** (P3e does
it; the group letter is the commit group), **OUT** (out of scope, with the
reason). "Exact" = the integer form gives the same value as today's float
on every input (no game change); otherwise the change is a deliberate
re-baseline measured by the A/B.

Doors (a config or table constant quantized ONCE, at load or at the edge)
are not sites: a float→Q16 conversion of an authored number at a door is
allowed by the iron rule's four doors. Where a constant was re-quantized
per tick from config, P3e moves it to the door unchanged in value.

| # | slot / site (file :: function) | today | decision (kit) | lands in | status |
|---|---|---|---|---|---|
| 1 | 1a impulse + turning (`unit_impulse.integrate_all`, `turning.step`) | integer (G5p-a1/a2) | -- | 1a | DONE-already |
| 2 | 1b throw release target (`Simulation._release_grenade`: `cx + dequantize(dx_q) * reach + 0.5`, `int(target_fx)` at detonation) | float (exact rationals) | Q16 target `(cx << 16) + (dx_q * reach_q >> 16) + HALF`; detonation tile `shr_trunc` (exact) | 1b, 2 | TODO (e) |
| 3 | 1b wind-up release direction (`_advance_throw`: `quantize_scalar(u.facing)`) | float round trip of an exact value | `u.facing_q` (exact) | 1b | TODO (a) |
| 4 | 1b shove, plant, the fuse in ticks | integer (#183, #193) | -- | 1b | DONE-already |
| 5 | 1c exposure decay | integer (G4p) | -- | 1c | DONE-already |
| 6 | 2 grenade flight (`Projectile.update_position`: `sqrt`, `/ travel_speed`, `elapsed / travel`) | float | Q16 positions, `isqrt` distance, one rational per tick; `fx` / `fy` float views | 2 | TODO (e). Read by the observation and the renderer only: the detonation reads the target |
| 7 | 2 WEGO fuse `int(fuse_seconds * tps)` (`Projectile.get_detonate_tick`) | float | -- | -- | OUT: WEGO order path; the continuous throw carries `fuse_ticks` |
| 8 | 2 / 4 the shot cone (`fire_burst`, `fire_beam`: `math.radians(spread)`, `rng.uniform(-cone, cone)`, float angle) | float draw | integer draw `draws.cone_offset_q16(rng, cone_q)` (one draw per former draw, same order), `cone_q` at the weapons door; angle Q16 | 4 phase A, 2 | TODO (a). Re-baseline: new random values |
| 9 | 2 / 4 the line march (`BulletInFlight.advance`: `rx += step_x`, `int(rx)`) | float, exact on the Q16 grid | Q16 ints, `shr_trunc` tiling (exact) | 4 phase A, 2 | TODO (a) |
| 10 | 4 free-aim bearing (`_directional_fire`: `-u.facing`) | float (exact) | `-u.facing_q` | 4 | TODO (a) |
| 11 | 4 tile-target bearing (`atan2_rad(fy2 - fy1, ...)`) | float in / kit / float out | `atan2_q16` of Q16 deltas (exact) | 4 | TODO (a) |
| 12 | 2 / 4 exposure roll (`attack_resolver.roll_exposure`: `uniform(0, 1) < exposure`) | float draw | `draws.roll_q16(rng, p_q)` -- `rng.integers(0, ONE) < p_q`, `p_q` the materials door's Q16 `cover_exposure` | 4 phase A | TODO (b). Live on the match path (furniture 0.55). Re-baseline |
| 13 | crit roll (`roll_crit`: `uniform < chance * arc_mult`), `arc_multiplier` (float angles, `math.pi` wrap) | float | `roll_q16` against `mul_q16(chance_q, mult_q)`; arc on `facing_q` with `turning.wrap_q16`, half-arcs Q16 at the profile door | 4 phase A | TODO (b). Dormant on the match path (the loadouts door refuses a melee crit; no ranged row has one) |
| 14 | `scale_half_away(amount, mult)` (crit / facing-defense amount) | float product | `amount * mult_q` rounded half away (exact for grid multipliers) | 4 | TODO (b) |
| 15 | `int(dmg * bullet_damage_multiplier)` (zombie hit, `BulletInFlight.advance`) | float product | `(dmg * mult_q) >> 16` (truncation of a non-negative, exact for a grid multiplier: 0.25) | 4 phase A | TODO (c) |
| 16 | mitigation + the hp write (`damage.mitigate`, `apply_packet`: float `max(0, a - armor) * resist`, then the Q2-lift snap; `Unit.current_hp` a float store) | float then Q16 | `DamagePacket.amount_q` (Q16; the float `amount` quantized once at the packet door when no `amount_q`), armor / resist Q16 at the species door, `mul_q16_round` (half away), hp stored as the int `Unit._hp_q`, `current_hp` a read-only-grid float view | `apply_packet` | TODO (c). Exact for every amount already on the grid (every shipped packet but heat / poison) |
| 17 | 2b status DoT (`tick_statuses`: `dequantize(magnitude_q16)`) | float of an exact value | `amount_q = magnitude_q16` | 2b | TODO (c), exact |
| 18 | 4c `chew_wall`: `quantize_scalar(float(wall_damage))` | float of an int | `wall_damage << 16` (exact) | 4c | TODO (e) |
| 19 | 4 range gates `math.sqrt(...) > range_tiles` (`process_shooting`, `auto_fire`) | float | -- | -- | OUT: WEGO fire orders / Move & Attack only; the continuous path fires directionally and the march bounds range in integer steps |
| 20 | 4 melee verb (`process_melee_intents`, `melee_arc_target`) | integer (G6p) | -- | 4a | DONE-already |
| 21 | 4b spray cone (`spray_cone_tiles`: `atan2_rad`, `cos_rad(radians(half))`), amounts `heat / div`, `gas / div`; `spray_target`, `spray_aim_angle` float columns | float | bearing `atan2_q16`, cone cosine at the weapons door, amounts through FieldEdit's integer door (`amount_q // div`, rounded), the two columns Q16 ints (`spray_target` a Q16 point, `spray_aim_angle` a Q16 angle) | 4b | TODO (e). No shipped match loadout sprays |
| 22 | 5 the horde (`horde.py`, `ai_zombie.py`: field, wander, activation, bite) | integer (G9p) | -- | 5 | DONE-already |
| 23 | 5 `face_towards(float(nx), float(ny))` | float in, quantized at the kit (exact on whole tiles and Q16 positions) | -- | 5 | DONE-already (exact; the device reads `x_q`) |
| 24 | 3 the walk and the push (`move_speed_q16`, `unit_impulse.move`) | integer (G5p-a2, P3c) | -- | 3 | DONE-already |
| 25 | 6 the stamp refill (`GameMap._neighbor_mean`: `total / count` into int32) | float division, truncated by the int32 store | `total // count` (exact: non-negative ints) | 6 | TODO (f) |
| 26 | 6b FieldEdit: the bridges (dequantize → float combine → requantize per field), `amount * weight * mult * scale/ONE`, the DISC / BEAM `sqrt` weights, `_falloff_weight`, TILES float weights, the noise `uniform(low, 1)`, the atmosphere 21 / 79 split | float64 | the amount quantized ONCE per edit into the field's currency (or handed in as `amount_q`); integer weights (`linear_weight_q16`, exact rounding of `1 - sqrt(d2)/r`), TILES integer weights over a `weight_den`, the noise an integer draw `draws.noise_mult_q16`, the combine an integer add / sub / max / clamp in the currency, the split `mul_q16_round` by the Q16 fractions | 6b | TODO (f). Re-baseline: every blast's deposits move by rounding |
| 27 | 2 / 9e `apply_explosion` (`sqrt` falloff, `quantize(wall_damage * falloff)`, the 3x3 `wave_source` amounts, the inner smoke clear `radius * 0.4`) | float | exact integer falloff (`unit_fixed.falloff_round` / `falloff_floor`, from `isqrt` of the scaled square), wave amounts as `amount_q`, inner radius Q16 | `execute_payload` | TODO (e) |
| 28 | `apply_blast_damage` (`sqrt`, `int(max_damage * falloff)`) | float | `falloff_floor` (exact: `M - ceil(isqrt_ceil(M^2 d2) / r)`), range `d2 <= r^2` | `execute_payload` | TODO (e), exact bar float rounding on boundaries |
| 29 | the generic explosion's deposits (`linear_weights` `sqrt`, `deposit_over_tiles` `w / norm`, `total / (1 - noise/2)`, `heat_mj * units_per_mj * scale/ONE`, smoke units of soot) | float | integer weights + `weight_den`, totals quantized once per (payload, map) at a door, scaled by the Q16 air read in integers | `execute_payload` | TODO (e) |
| 30 | `blast_expansion._source_q` (`expansion / (ticks * n * V * dt)`) | float with the walk's tile count | per-tick total quantized once, `div_round(total_q, n)` | `execute_payload` | TODO (e). Class-P twin file (div_source) |
| 31 | the heat source's blast wall law (`heat_source_system._blast_damage_q`: `sqrt` falloff) | float | the same `falloff_round` as `apply_explosion` | 9e | TODO (e) |
| 32 | non-strength payload rows (`blast_smoke_peak`'s float weight sum, `ignite_ring`, `deposit_heat`, `start_expansion`, `add_explosion_smoke`) | float amounts | amounts reach FieldEdit's integer door; the peak's weight sum through the same integer weights | `execute_payload` | TODO (e) for the weights; the authored float amounts are doors. No shipped match loadout carries these rows (smoke_screen is a gas emit, the flashbang has none) |
| 33 | `apply_flash_blind`, noise radii | integer (G3p, G9p) | -- | 2 / 9e | DONE-already |
| 34 | 9c heat chain (`felt_temperature`: `peak / HEAT_SCALE`, `phi * abs * (1 - refl)`, `amb + k * phi`; `apply_environmental_damage`: `rate * (1 + k * over) * dt`) | float | `t_felt_q = amb_q + mul_q16_round(A_q, peak)` with `A = flux_to_temp * abs * (1 - refl)` at the door; `dmg_q = div_round(rate_q * (ONE + mul_q16_round(k_q, over_q)), ONE * tps)` | 9c | TODO (g). Re-baseline by rounding |
| 35 | 9c2 wind (`apply_wind_force`, `wind_impulse_q`) | integer (G5p-b) | -- | 9c2 | DONE-already (`wind_q_pa` is a diagnostic) |
| 36 | 9c3 teargas | integer | -- | 9c3 | DONE-already |
| 37 | 9c3 poison dose (`dps * dequantize_trace(d) * dt`) | float | `div_round(dps_q * d, TRACE_ONE * tps)` | 9c3 | TODO (g) |
| 38 | 9d ignition (`quantize_scalar(o2_frac_ext)`, `ignition_seed` per tick) | config floats quantized at the call | -- | 9d | DONE-already (door) |
| 39 | 9e vents, pumps, doors, timed charges, `charges.tick`, heat-source deposit (`heat_q16` at build) | integer / door | -- | 9e | DONE-already |
| 40 | 7 the physics core | integer C++ | -- | 7 | DONE-already |
| 41 | the bots (`control_bots*.py`, `pathfinding.astar`) | float decision code | -- | -- | OUT: host controllers, not ported (§12.3); they write only intents, which are integer at the facade |
| 42 | `ExplosionEvent.strength` / `flash*`, `ShotFiredEvent` tiles, `Shot` tracers | float | -- | -- | OUT: render-only, outside the digest |
| 43 | onephase / WEGO (`timeline.py`, the round snap, `defense_multiplier`, `cover_at`) | float | -- | -- | OUT: legacy (§11) |
| 44 | the stamp's float32 `permeability` / `wave_absorb` per-unit values | float32 bits | -- | -- | OUT: physics-plane dtypes already stamped bit-exact on the device (P4b) |

Counts: 44 rows -- DONE-already 13, TODO 25 (in seven groups), OUT 6.
Changes from the 2026-11-07 table: the range gates (row 19) are WEGO-only
on today's code (OUT); the zombie `math.sqrt` gates were done by G9p; the
fuse was done by G3p; NEW sites: the throw target (2), the spray columns
(21), the generic explosion's deposits (29), `_source_q` (30), the heat
source's wall law (31), grenade flight (6).

**As built (P3e, 2026-11-12, branch `29-p3e-integer-match`, HUMAN-TEST
pending).** Every TODO row done; none deferred. One line each:
- Groups a + b (one commit): the cone `draws.cone_offset_q16` on a Q16
  bearing (`aim_q`, the free aim `-facing_q`), the march `rx_q` / `ry_q`
  (bit-identical: the float march was exact on the grid), the rolls
  `draws.roll_q16` against the materials table's new `cover_exposure_q16`
  and a Q16 crit product, `arc_multiplier` on `wrap_q16` returning Q16.
- Group c: `DamagePacket.amount_q` + the door `q()`, `damage.mitigate_q`
  (one rounding, half away; exact for grid amounts), the hp STORED as the
  int `Unit._hp_q` (`hp_q`), `current_hp` its float view, the schema column
  keeping its name and key (read through `hp_q`); the zombie bullet
  multiplier a Q16 product truncated as `int()` was; DoT hands its Q16.
- Group g: `exchange.felt_temperature_q` (one gain quantized at the door),
  the heat row and the poison dose as single rationals rounded once; within
  1e-4 relative of the float model (the dials' Q16 quantization).
- Groups f + e (one commit, coupled): FieldEdit integer from the door to the
  field (`_CURRENCY`, Q16 weights, `falloff_round`, BEAM as an exact
  rational, TILES over `weight_den`, the noise draw, one rounding per tile,
  the 21 / 79 split per hundred); `_neighbor_mean` `div_trunc`
  (bit-identical); `apply_explosion` / the heat source's wall law
  `falloff_round`; `apply_blast_damage` `falloff_floor` (exactly
  `int(M * falloff)`); the generic explosion's weights and totals integer;
  `_source_q` divided in integers; sprays, the throw target and the
  grenade flight Q16.
- Choices where the design was silent: hp as an int store (not a float with a
  grid assert); the draw = ONE raw step of the bit generator's high 32 bits
  (`uniform` was one step too, so the stream position per draw is as
  before); LINEAR weights Q16 (2^-17 relative of the weight), not exact
  through the product; atmosphere split as the exact 21 / 100; a policy with
  no integer currency refused (the float combine deleted: no shipped field
  used it); the throw target rounded to Q16 (its tile can differ from the
  float's only within 2^-17 of a tile edge).
- Not as the design expected: `GOLDEN_AGGREGATE` did NOT move (4a13fc72:
  its scenario reaches no P3e site that changes a value); the recording
  format was NOT bumped (no recorded intent's meaning changed; the
  precedent of G9p / P3c: a game change alone diverges at a checkpoint).
- The A/B: `docs/p3e_ab_2026-11-12.md` — within seed noise; the firefight's
  heat deaths (+21 %, ~1.6 s.e.) the one number to watch.

Draws: every match-path draw P3e touches goes through ONE module,
`src/simulation/draws.py` (`cone_offset_q16`, `roll_q16`,
`noise_mult_q16`), each ONE `rng.integers` call per former `rng.uniform`
call, in the same order -- so P3b swaps the generator in one file.

As built (P3b, 2026-11-12, branch `29-p3b-philox`):
- THE salt enum: `src/simulation/rng_salts.py` (`Salt`: EPISODE and WANDER
  moved in with their values, so the farm's episode draws and the wander are
  bit-identical; SHOT_CONE, BEAM_CONE, EXPOSURE, SHOT_CRIT, MELEE_CRIT,
  FIELD_NOISE -- four ASCII letters each). No salt literal in `src/`
  (`tests/test_match_draws.py` scans the block calls).
- The keyed draws: `draws.MatchDraws` on every sim as `sim.draws` -- key =
  the seed (64 bits; an UNSEEDED sim keys from its entropy-seeded
  generator, so it stays unseeded: CHOICE), counter = (tick, index, salt,
  id), the block's `out[0]`, one block per draw. `next(salt, id)` is the
  site-local index (the k-th draw of that (salt, id) this tick, the count
  cleared when the tick changes); `at(salt, id, index)` names one. The
  tick is the sim's own clock plus `tick_base`, which legacy TwoPhaseWEGO's
  `_end_round` advances by the round's length before rewinding `sim.tick`
  (CHOICE: otherwise every round would replay the last one's draws). An id
  of -1 ("no unit") is the word 2^32 - 1.
- Sites: the shot cone is the shooter's `next(SHOT_CONE, id)`; the round
  carries that draw's tick and index (`spawn_tick`, `ordinal`), and its
  later exposure / crit rolls are `at(EXPOSURE | SHOT_CRIT, shooter,
  round_index(age, ordinal))`, `(age << 16) | ordinal` (refused past 16
  bits each); the beam cone `next(BEAM_CONE, id)`; the order path's melee
  crit `next(MELEE_CRIT, attacker)` (the melee VERB never draws: its rows
  carry no crit). FieldEdit noise, on EVERY path: `at(FIELD_NOISE, r * w +
  c, edit key)`, the EDIT KEY being the queue's `seq` in this flush (its
  enqueue index, the existing deterministic tie-break; CHOICE: P4c's
  composite key orders records exactly as `seq` does, so the device can
  rebuild it as the rank); `apply_field_edit` refuses a noisy edit
  without a key. Charges, timed charges, heat sources and the zombie bite
  draw nothing (`MatchDraws.drawn`, a word counter, proves it in tests).
- `sim.rng` is drawn from by nothing in `src/` (every former caller --
  the slots, the ruleset's door explosives, the onephase timeline, the
  charge / timed-charge / heat-source detonations -- passes `sim.draws`);
  `simulation.match.spawn_match` replaces it with `draws.RefusingRng`,
  which raises on any attribute use, naming it. The helpers refuse a numpy
  Generator by name.
- The C++ header `cpp/src/philox32.h` vendored verbatim (as on
  `63-swarm-units`, upstream 6c33a3d0, recorded sha256) with its bindings,
  so `tests/test_philox32.py`'s C++ KAT, Python == C++ sweep and Q16
  mappings run; no C++ sim code draws yet (P4c / P7a). The paper is
  archived (`docs/papers/salmon2011_random123_philox_sc11.pdf`).
- SOLE CAUSE: `tests/_p3b_sole_cause.py` (a pytest plugin feeding main's
  numpy words through the keyed sites) makes all eight pinned matches pass
  at main's values on the branch. Re-pinned once (ship_demo seed 1: win 0
  at 2966 -> win 0 at 2708); `GOLDEN_AGGREGATE` unmoved (its scenario draws
  nothing). No recording format bump (no intent's meaning changed: a
  recording re-simulates from its seed). A/B: `docs/p3b_ab_2026-11-12.md`,
  everything within 1.6 s.e.

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
| P3b, P3d | Philox (+ FieldEdit's noise on every path), the distance field; P3b before P4c and P7a, P3d before P7d | per row | P3b Sonnet; P3d Opus | P3d (**both DONE: P3b merged `f55a5b89`; P3d built by #183 G9p and closed 2026-11-13 with no patch, §4.9**) |
| P4c | the device edit queue (Class P; after P3b and P3e: integer policies, Philox noise, the phased key) | one tol-0 scene per policy row + the cross-tick overlap scene | Opus | no (**DONE 2026-11-13, §4.8 as built**) |
| P6 | the device unit store + schema's device side, the Unit ↔ store bridge, slots 1 / 1b / 2b / 3 / 9 / 9b / 9c–9d / 10 as kernels, the per-env serial applier (block form) with the topology seam as device functions, the event ring, the entity export for the digest | live lockstep; the topology scene gates | Opus | no |
| | **RE-PLANNED 2026-11-14 into P6.1 to P6.4: §4.4b replaces the row above** (the unit design `docs/unit_design_2026-11-14.md` first: U1 the hooks, PX the poison law) | per patch, §4.4b | Opus | no |
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

**Re-planned 2026-11-15:** the rows P6b and P7a to P7d above are replaced by
§4.4c's cut (thirteen patches, P7.0 to P7e), written from
`docs/residency_p7_inventory_2026-11-15.md` after P6 merged.

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
| A* "the only pathfinder"; zombies "WRAPPED, never rewritten" | the horde walks the field on both paths; A* is the host pathfinder | P3d (done at #183 G9p) |
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
