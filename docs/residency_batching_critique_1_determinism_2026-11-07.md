# Critique 1: determinism and residency (2026-11-07)

Design under review: `docs/residency_batching_design_2026-11-07.md` (v1, `d6c0957`).
Lens: whether the design keeps the iron rule (synced state integer, CPU == device at
tol 0) and gets batch invariance (env i of N == env i alone) from the mechanisms it
names, and whether the residency claim ("zero host round trips per tick") holds.
Each finding names the file and line it rests on. All line numbers are from this
worktree at `d6c0957`.

**VERDICT: NOT SOUND as written.** The architecture holds: batched `(N, h, w)` cores,
per-env masks, integer atomics, the two-phase shape and the replay oracle. But two of
the mechanisms the design names cannot deliver what §5 claims (B1, B2). Fix those two
and fold in the majors, and the design becomes SOUND WITH FIXES. Nothing here forces
another route.

---

## BLOCKERS

### B1. §4.3 / §4.8 / §5.5: the device's walk-order apply calls effects that only the host can run, and device slots raise topology events that are routed to the host. Ordering and "zero host sync" both fail.

**Evidence.**
- Phase B is not just packets. The march records `fx.world(chew_wall, …)`
  (`combat.py:335, 382, 1228`) and `fx.world(execute_payload, …)` (`combat.py:480`).
  `SimultaneousEffects.resolve` runs these in recording order, interleaved with
  `apply_packet` (`simultaneous.py:118-136`). `chew_wall` writes `wall_hp` directly
  and calls `gmap.destroy_wall` at zero (`combat.py:190-192`).
- Order inside phase B is observable:
  - A unit dies once. The packet that takes hp to zero carries the kill credit
    (`damage.py:361`), and `UnitKilled.attacker_id` is hashed in `tick_digest`
    (`field_ab_harness.py:250`).
  - A second chew on a destroyed tile is a no-op (`combat.py:188-189`).
  - `destroy_wall` seeds the opened tile from its four neighbours' gas and the
    breach mask (`gamemap.py:2590-2700`). So two adjacent destroys in one phase B
    depend on their order.
- §4.3 puts "blasts and other payload executions" and "level topology changes
  (`destroy_wall`, `seal_tiles`)" on the host. §5.5 says phase B is "ONE thread per
  env looping … in walk order". These cannot both hold: the loop would have to stop
  at a host-only entry, wait for the host, and resume. Neither the stop, the resume
  point nor the sync is specified.
- Topology events are also raised by stages that §4.3 puts on the device:
  - slot 9 burn-through: `step_tail` returns `destroyed`, then host `destroy_wall`
    (`simulation.py:1653-1670`);
  - 9b over-pressure walls: `find_burst_walls` → `destroy_wall`
    (`simulation.py:1682-1689`). §4.3 lists 9b as device (P6) and its consequence as
    host.
  - 9e door structural sweeps: `seal_tiles` / `unseal_tiles`.
  - slot 2 grenade fuse-outs: immediate `execute_payload`, `simulation.py:1938`.
  - 9e timed charges and heat-source deaths: `execute_payload` through `sim.rng`.

  For each of these the host must poll a per-env flag, which is a D2H and a sync at
  every such slot of every tick. That contradicts §1 ("zero host round trips per
  tick") and §4.2 ("No host sync inside a tick"). It also breaks §4.11's
  "capture once … replay per tick": the graph would have to be split at slots
  2, 4, 8, 9, 9b and 9e.
- The frequency argument (§4.3: "~2 frag throws per 722-tick match") counts only
  frags. It leaves out chew-to-zero, burn-through, bursts, door toggles, payload
  rounds, charges and heat-source deaths. At N = 64, "some env has an event this
  tick" is the common case, not the rare one.

**Fix.**
1. Make the topology seams DEVICE functions, with books through per-env atomics:
   `destroy_wall`, `seal_tiles`, `unseal_tiles`, `on_tile_changed`, and the wall law
   (`chew_wall`, `apply_explosion`'s wall damage). They can be called from the
   per-env phase-B loop and from the 9 / 9b / 9e kernels.
   - `destroy_wall` is tile-local. It reads four neighbours, writes one tile's
     material, caches, breach mask, gas, `gas_energy` (through the seam) and
     atmosphere.
   - Its only non-local read is the mass bracket
     `n_destruction_seed_sum += _bulk_n_sum() - _n_before`
     (`gamemap.py:2571, 2762`). Only tile (fy, fx)'s gas changes in
     between, so that delta equals the tile's own change. The device form is the
     local delta.
2. For `execute_payload`, choose one of these explicitly:
   - (a) A device payload kernel: one block per blast; the reach walk as an
     in-block BFS (the distance-field shape); deposits as device edit records.
     `apply_blast_damage` runs in the env's serial phase-B thread at its turn.
   - (b) A declared split. Phase B stops at the env's first detonation record; the
     host runs the payload on a D2H of that env's unit store and planes; the device
     resumes at the next record. State the sync per blast-slot, the graph segments
     it implies, and that §1's claim becomes "zero host sync on ticks where no env
     has a host event".
3. Restate §4.2's sync rule accordingly. Add an equivalence gate: a burst wall, a
   burn-through, a chew-to-zero and a door toggle must each be bit-identical between
   the device and the CPU (not covered by the replay corpus unless its recordings
   contain them).

### B2. §4.5 / §5.3 / §5.7: the device backend cannot compute the oracle it is gated on (`tick_digest`). Counters are not the hashed event stream, and the hashed unit record is not the schema.

**Evidence.**
- Gate 3 compares the per-tick `(field_digest, tick_digest)` at every checkpoint
  (`match_replay.py:149-161`).
- `tick_digest` is sha256 over `repr({units: [_unit_record(u) sorted by id],
  events: [_event_record(ev) for UnitHit/UnitKilled in EMISSION ORDER]})`
  (`field_ab_harness.py:255-270`):
  - The event record is `kind, unit_id, damage, source, killed_by, attacker_id`
    (`:220-232`). `damage` is the float `quantize_hp_delta` value
    (`damage.py:336-338`).
  - The unit record (`:179-215`) hashes `ap` and `n_orders`, which are WEGO fields
    that §4.4 says are "NOT columns". It hashes `x`, `y`, `current_hp` and
    `facing` through `q(v) = round(v·1e9)`, plus `offsets`, `faction` and the
    `life_state` string.
- §4.5 gives the device only per-unit event COUNTERS `(N, U, E)`. Counters lose the
  order, the damage amounts and the per-event cause.
- §5.7 only mentions moving the store's columns into the field digest's entity
  section, not this record.
- So the "gameplay twin oracle" for P6, P6b, P7 and P8 cannot run on the device as
  specified.

**Fix.**
1. Add a per-env device EVENT LOG `(N, E_max)` of `(kind, unit_id, damage_q, cause,
   killed_by, attacker)`, appended in the reference's emission order:
   - slot order;
   - unit-id order inside 2b (`status.py:309` sorts by id) and inside 9c / 9c3;
   - phase-B order inside 2 / 4;
   - zombie walk order inside 5.

   Append only from serial per-env loops, or into per-`(slot, unit, k)` slots
   compacted in that order. Overflow is a counted per-env failure. Rewards read
   the counters derived from this log.
2. In P3's schema commit, redefine `_unit_record` from the schema: raw Q16 ints,
   WEGO fields dropped or explicit constants. Re-pin the replay corpus in the same
   commit with its rationale.
3. The device's checkpoint digest is the same Python function applied to a
   checkpoint D2H of the store and the log. Never a second digest.

---

## MAJORS

### M1. §4.2: per-env `n_sub` drives more than one loop and many host-double folds. The design names one loop and no folds.

**Evidence.** In `EOSSolver::step`, `n_sub` (`eos_solver.cpp:726-730`) sets
`dt_s_d = dt_d / n_sub`. The following are all host double folds from it:
- `k_flux_q = quantize(k_work_d·dt_s_d/(2·dx_d))` with its range `throw`
  (`:742-756`);
- `flux_pu_cap = 2^60 / k_flux_q` (`:763`);
- the per-cell donor coefficients
  `coeffE/S = mul_q16(quantize((double)ff), quantize(dt_s_d))` over the float32
  `dyn_permeability` (`:786-799`);
- `dt_s_q` (`:807`).

There are TWO n_sub loops:
- the transport substeps (`:805`);
- the energy-flux sub-cycles (`:1500`). On the resident path this is
  `energy_flux_scalar_folds(…, pre.n_sub)` (`cuda_eos_resident.cu:965`;
  `cuda_kick_compression.cu:606-628`, its own `throw`) and the loop at
  `cuda_kick_compression.cu:649`, with the `d_e0` snapshot before it.

The resident chain builds `coeffE`, `coeffS`, `absorb_q` (from the float
`dyn_wave_absorb`) and `cap2` on the host and uploads them every tick
(`cuda_eos_resident.cu:776-824`).

**Fix.**
- Build an n_sub FOLD TABLE once per batch on the host: `N_SUB_MAX` = 8 rows
  (`eos_solver.h:179`) of `(dt_s_q, k_flux_q, flux_pu_cap, dts_q_c)`. Config and dt
  are homogeneous, so the table is per batch. Run both range checks over all 8 rows
  at table build. Device code indexes the table by the env's `n_sub`.
- Build `coeffE/S`, `absorb_q` and `cap2` with device kernels after the reduction:
  float32 → double is exact and the one double add is the
  `quantize((double)pf)` idiom `cuda_eos_resident.cu` already argues bit-exact.
- Mask both loops per env. Give the graph two WHILE nodes, or one body with both
  loops.

### M2. §4.2 vs §9: no host sync for substep control is only true once P9 exists. P5 needs a decision.

**Evidence.** The host drives the substep loop today
(`for (int s = 0; s < pre.n_sub; ++s)`, `cuda_eos_resident.cu:875`; `:649` for
K3). A device-side `n_sub` needs either a host read of the batch max or a device
loop. The WHILE node arrives at P9; P5 (batching) and P6–P8 run without it.

**Fix.** State the P5–P8 interim:
- either always run `N_SUB_MAX` iterations with per-env masks (a cost, measured in
  P5),
- or one 4-byte D2H of `max(n_sub)` per tick (a sync, counted by the profiler).

Make P9's gate "the profiler's per-tick sync count reads 0 on a no-event tick".

### M3. §5.1 / §5.2 / §13a: the books have no batched home. Today they are host singletons and Python dicts filled every tick.

**Evidence.**
- The resident chain zeroes its counters, then on every tick runs
  `cudaDeviceSynchronize` and D2H's `boundary_flux_`, the rail, and the kick, flux,
  ecnt and trace counters into `EOSSolver` fields (`cuda_eos_resident.cu:826-1053`).
- Some counters are ASSIGNED per tick and some are ACCUMULATED: for example
  `trace_clip_sum_ +=` (`:992`).
- The seam books are Python-int dicts with string channels:
  - `gas_energy_books[channel]` (`gamemap.py:1082-1086`);
  - `solid_energy_books` (`:1104-1110`);
  - `trace_deposit_books` / `trace_clip_books` (`field_edit.py:727, 730`);
  - `n_destruction_seed_sum`.
- `reset_trace_books()` runs at tick start (`simulation.py:1559`).
- Gate 1 says "every book" and gate 2 says "integer atomics make the books
  order-free". That is true of a per-env slot, but no per-env layout is specified.

**Fix.**
- Define per-env book arrays `(N, slots)` int64 as part of the batched engine:
  - a fixed channel enumeration replacing the string keys, a registry the seams
    and the closure gate share;
  - per-counter assign or accumulate semantics as on the CPU;
  - per-env tick-start reset.
- Run the closure identity per env from those arrays at checkpoint cadence, never
  by a per-tick D2H.
- State the int64 range argument against Python's unbounded ints over the longest
  training episode, or count a wrap.

### M4. §4.8: the device edit queue's three claims (same order, same policies, same semantics) need four things the design does not say.

**Evidence and fix, item by item.**

1. **`seq`.** The flush sorts on `(field, source_id, region, seq)`
   (`field_edit.py:805-807`). `seq` is the global enqueue index (`:795-797`), and
   `source_id`s are per emitter TYPE, not per emitter:
   - `_SRC_SPRAY_HEAT` / `_SRC_SPRAY_GAS` at `combat.py:1451, 1460`;
   - `_SRC_PAYLOAD_*` at `payloads.py:123, 193, 224, 459, 477`;
   - `_SRC_EXPLOSION*` at `physics.py:96-189`.

   So two sprayers' or two blasts' edits tie on the first three keys and are
   ordered by `seq` alone. A device `atomicAdd` counter for `seq` is
   nondeterministic. *Fix:* a composite deterministic key that reproduces the
   reference's enqueue order, `(slot, phase-B effect index or emitter row in walk
   order, local k)`. The string `field` key becomes an ordinal table that preserves
   its lexicographic order.
2. **Serial per tile.** The combines are not commutative: clamps, the int32
   saturation `trace_add_sat` (`field_edit.py:728`), MAX mixed with ADD. *Fix:* parallel
   only within one edit's tile list, with a barrier between edits whose regions can
   overlap, or one serial thread per env. Name which.
3. **float64.** Every combine is a dequantize → double → requantize bridge
   (`_combine_water/_gas/_fire/_temperature/_atmosphere_to_N`,
   `field_edit.py:428-540`). Also `contribution = amount·weight·mult` and
   `*= int(scale)/_Q16_ONE` (`:676-678`). "Ported as device functions" therefore
   means float64 device code in the sim path, held to Python's operation order, no
   FMA, `quantize_scalar`'s `floor(v+0.5)` / `ceil(v-0.5)`
   (`unit_fixed.py:58-69`). *Fix:* either say that and gate it, or (preferred) add
   "FieldEdit integer at the enqueue door" to §7 as a re-baseline.
4. **The seam inside the applier.** `gas_energy_deposit` writes into
   `field_edit_wave`, `field_edit` and `field_edit_atm` (`:693-768`), alongside
   `gas_is_accountable`, `gas_t_abs_at` and the trace books. *Fix:* these become
   per-env atomics in the device applier (M3), with the closure identity checked
   on a burning, blasting scene.

**Noise draws.** Today they are one stream over the surviving tiles in flush order
(`:670-673`). Under Philox the counter must name either the flush ordinal of the
surviving tile (a prefix sum over skip masks across all edits) or `(edit key,
tile)`. Pick one and list it in §7.

### M5. §4.8: the rare-event path's data set is not "the open mask and the two air planes, writing through the edit queue".

**Evidence.** If any part of `execute_payload` stays on the host (B1 option b), it
reads and writes much more (`payloads.py:365-520`):
- `apply_explosion` writes `wall_hp` / material IMMEDIATELY, and can call
  `destroy_wall` (`:422`);
- `apply_blast_damage` iterates the UNIT LIST and emits hashed events (`:432`), so
  it needs the env's unit store;
- `inject_bulk_gas` writes gas and `gas_energy` IMMEDIATELY through the pump
  primitive, not the queue (`:251-290, 471`);
- `start_source` writes `div_source` directly (`:446`). That is not a FieldEdit
  field (`field_edit.py:227-278`); it is consumed by `take_source_plane`'s
  whole-plane `.any()` on the host every tick (`blast_expansion.py:210-222`;
  `physics_runner.py:890`) and uploaded by `cuda_eos_resident.cu:911`;
- `gmap.expansions` is multi-tick host state;
- `_claim_tiles` writes the TickCredit claims (`:473`) that the device 9c / 9c3
  rows read;
- `blast_log` is read by the device heat sources (9e).

**Fix.** List the full read and write set of the host rare-event path and the H2D /
D2H it implies, per env. Or take B1 option (a) and keep it all on the device.

### M6. §7: the only currency change listed is Q16 positions. The gameplay slots that §4.3 ports are float64 at many decision points, and the vendored Philox kit has no double draw.

**Evidence.**
- The bullet march accumulates `rx`, `ry` as floats (`combat.py:225-232, 322-326`).
- The cone is `base_angle + float(rng.uniform(-cone, cone))` with
  `cone = math.radians(...)` (`combat.py:1058, 1089, 1172, 1195`).
- Exposure and crit rolls are `float(rng.uniform(0,1)) < p`
  (`attack_resolver.py:80, 138`).
- Mitigation is `max(0.0, amount − armor)·resist_mult` (`damage.py:260`), and
  `current_hp` is a float.
- The zombie multiplier is `int(actual_dmg·CFG.zombie.bullet_damage_multiplier)`
  (`combat.py:406`).
- Range and activation tests use `math.sqrt` (`combat.py:855, 1009`;
  `ai_zombie.py:77, 93, 123`).
- The `stamp_units` freed-tile refill is a true division `_neighbor_mean`
  (`gamemap.py:2130, 2377-2389`).
- The kit vendored at `bd7557d` exposes only `philox32_uniform_q16`, `_angle_q16`
  and `_below` (`src/simulation/philox32.py`). §4.7's "the SAME draws through the
  Python twin" therefore either changes every consumer to integer (re-baselines not
  in §7) or needs a double construction the kit does not have.

**Fix.**
- Add a float inventory of every synced-feeding site in the slots of §4.3. For each
  site decide one of:
  - (i) a device double transcription (IEEE, `--fmad=false`, Python's operation
    order, `__dsqrt_rn`, the same quantize rounding), gated by replay lockstep;
  - (ii) an integer port as a P3 re-baseline.
- Define the draw type of each Philox site: Q16, or a double from two words, with
  the formula fixed in the kit and its twin.

### M7. §4.10: streams. Every existing core runs on the legacy default stream with synchronous copies, and the torch handshake is one-directional.

**Evidence.**
- No launch in `cpp/src/*.cu` passes a stream: zero `<<<g, b, s, stream>>>`.
- There are about 200 synchronous `cudaMemcpy` calls and 18 `cudaDeviceSynchronize`
  calls, including inside the resident tick (`cuda_eos_resident.cu:812-834, 854-881,
  974-1053`).
- Torch writes the ACTION buffer on its stream, and the engine reads it. That is a
  RAW hazard the other way from the one §4.10 handles.
- The next `step` overwrites the obs buffers that torch may still be reading
  asynchronously (a WAR hazard).
- §4.10's per-step host read of `terminated` / `truncated` is a host sync per step
  (per tick at `decimation` = 1), and it is not needed: reset happens in-step by
  mask.
- The C++ engine's own `cudaMalloc` scratch (`g_eos_res.ensure`) is outside both
  capped pools.

**Fix.**
- Thread one stream through every launch core and replace in-tick `cudaMemcpy` /
  `cudaMemset` with `cudaMemcpyAsync` / `cudaMemsetAsync`.
- Gate from P4 onward, not P9: capturing the whole tick in global capture mode
  fails on any legacy-stream operation.
- Use a two-way event protocol:
  - the engine waits on an event torch records after writing the actions and after
    its last read of the obs buffers;
  - torch waits on the engine's post-step event;
  - or double-buffer obs and actions.
- Make the `terminated` read an async pinned copy, or drop it.
- State "no allocation inside a step".

### M8. §4.1 / §4.6: auto-reset inside `step` needs a seed and template per env that the host does not have to decide.

**Evidence.** `template_index[env]` is "a device array, set by the host from the
seeds" (§4.6), yet envs reset inside `step` by mask (§4.1).
- The host would have to read which envs ended: a sync.
- If seeds are dealt in done order, env i's episode sequence depends on when the
  other envs end. That breaks batch invariance from the second episode on.

**Fix.**
- Env i's k-th episode seed and template are a pure function of
  `(run seed, env index, k)`, computed on the device (Philox, a reserved salt).
- Host `seeds` only at an explicit `reset(mask, seeds)`.
- The farm uses the same function, so backend equality holds across resets.
- Add "an env resetting mid-batch" to gate 2's cases.

### M9. §4.6: reset must copy the reference's loaded state, not re-derive it.

**Evidence.** §4.6 says reset "zeroes the dynamic planes, re-derives the derived
planes (gas energy from the seeds, the mirrors)". On the reference, that state comes
from:
- `GameMap.__init__`;
- `refresh_gas_energy` (`gamemap.py:1028-1051`);
- `seed_gas_temperature`;
- the loader's `temp_init` solids write;
- the door and prop stamps;
- the entity build;
- `spawn_match`'s `add_unit` (`simulation.py:475-511`).

A device re-derivation is a second implementation of each of these.

**Fix.**
- The bank stores the reference's full post-`spawn_match` tick-0 state: every
  synced and derived plane, the store, entity rows, the signal bus and books.
- `build_level_bank` verifies each entry's digest equals `run_match`'s tick-0
  checkpoint.
- Reset is a copy plus the Philox key. Nothing at load draws a random number
  (`simulation.py:487-490`), so a copy is exact.

### M10. §4.3 kernel mapping: some laws inside slots are serial, not two-phase, and need per-env serial loops in the reference's order.

**Evidence.**
- Zombie bites apply IMMEDIATELY in zombie walk order (`ai_zombie.py:147`). The
  `players` list is not refreshed (`:62`), so a later zombie can bite a marine an
  earlier one killed. Kill credit (hashed) depends on the order.
- The vent sweep is a serial per-duct ledger in entity-ordinal order
  (`vent_system.py:378-551`).
- Sprays' phase A reads unit state that shooting's phase A wrote in the same tick
  (`start_spray_burst`, `combat.py:876`; `process_sprays`,
  `simulation.py:1629`).
- §4.3 maps per-unit work to "one thread per unit row" and interaction to "one
  block per env", without naming these.

**Fix.**
- List the serial laws: zombie bites, phase B, the edit flush, the duct ledger,
  `destroy_wall` sequences.
- Implement each as a per-env serial loop in the reference's order, as separate
  stages.
- State that a kernel never reorders them.

---

## MINORS

- **§4.7 site list.** Charges, timed charges and heat sources never draw
  (`payloads.py:403`, `timed_charge_system.py:12`). Missing from the list: exposure
  (`attack_resolver.py:80`), crit including melee (`combat.py:414, 1714`) and the
  beam cone (`combat.py:1195`).
- **§5.5 atomicMin.** "Conflicts resolve by integer reductions (atomicMin on (key,
  slot))" contradicts the reference. Nothing is resolved: every packet lands in walk
  order, and the first to zero is the kill. Delete it or define it.
- **Philox counter.** In-flight rounds (slot 2, `simulation.py:1955`) and new bursts
  (slot 4) share `(tick, shooter id)`, so `stream_salt` must be per (slot, site).
  `draw_index` is local per (site, salt, tick) and never a cross-unit prefix; a
  bullet in flight needs its own spawn ordinal.
- **Water dormancy.** `water_depth.any()` (`physics_runner.py:1349`) is a per-env
  branch. Prove the water stage on a dry env is bit-identical to skipping it, or
  mask it.
- **Per-env failure.** Host `throw`s (`k_flux_q` range, combustion's claimant
  hard-check, FieldEdit's conservative-slice refusal) cannot throw in a batch.
  Generalise the sweep's count-and-mask (`cuda_radiation_sweep.cu:231`) into
  per-env failure flags that truncate the env and fail the gate.
- **Schema dtypes.** `{int32 Q16, int64}` lacks float32. Per-unit `permeability` /
  `wave_absorb` feed the float32 `dyn_permeability` / `dyn_wave_absorb`, which the
  EOS reads (`eos_solver.cpp:795`) and the digest excludes (`field_digest_spec.toml:85`).
- **Q16 step rounding.** Specify magnitude-then-sign for `dx_q·tpt_q`: a plain
  `>>` floors negative steps, so left and right moves differ by 1 LSB per tick,
  which biases one side of a mirror-symmetric generated ship. `tpt_q` should be the
  kit reciprocal of the integer `tick_cost` (`movement.py:146-153`), and
  `int(nx)` → `x_q >> 16` holds only for x ≥ 0.
- **Ignition plane.** `ignition_temp_q16[gmap.material]` is rebuilt on the host
  every tick (`physics_runner.py:1158`), and material changes with topology. It
  becomes a device gather.
- **Freed-tile refill.** `stamp_units`' refill (`gamemap.py:2126-2131`) is a host
  whole-plane compare every tick. Port it as an integer floor division (equal to
  the float `/` then int32 truncation at these magnitudes).
- **MG (§4.2).** Sound as stated: `mg_cycles` is a fixed loop with no residual
  test (`cuda_mg_solve.cu:380`), and the tail level depends only on shape (`:371`).
  A naive tall grid would couple envs three ways: `mirror_idx` at the seam rows,
  red-black parity flipping when h is odd, and a different coarsening and level
  count. The per-env tail must keep the every-cell-write rule
  (`cuda_eos_resident.cu:13-18`).
- **Gate 2 cases.** Add one env with a host or device topology event while the
  others have none, and one env resetting mid-batch.
- **P5 gate.** Also gate "device pre-stage == host pre-stage" at N = 1 on `n_sub`,
  `c_local` and `cap2`, before batching.

---

## QUESTIONS (not grounded)

- Do v1 training matches toggle doors (interior airlocks, wired doors) often enough
  that `seal_tiles` / `unseal_tiles` must be device functions even under B1
  option (b)?
- Does any non-test per-tick consumer read the EOS telemetry between ticks? None was
  found in `src/` (`grep boundary_flux` hits only comments in `gamemap.py`). If
  there is none, checkpoint-cadence books are safe.
- What is the bound on `gas_energy_books` channel magnitudes over a 10^4-tick
  episode? This is the int64 argument M3 asks for.
