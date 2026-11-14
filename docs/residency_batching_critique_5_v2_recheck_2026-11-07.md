# Critique 5: re-check of design v2 (applier, ring, classes, plan), 2026-11-07

Design under review: `docs/residency_batching_design_2026-11-07.md` v2 (`fae5955`).
Scope: the three mechanisms new in v2 (the per-env serial applier §4.3 / §13 b.4, the
event ring §4.5 / §5.9, Class P / Class T with the computed ledger §6 / §10 / §13 b.14)
and the patch plan (§3, §9), checked against the code in this worktree. Findings the
four earlier critiques raised and v2 answers correctly are not repeated. Line numbers
are from this worktree at `fae5955`.

**VERDICT: SOUND WITH FIXES.** The applier, the ring and the two classes are the right
shapes, and nothing here forces another route. But two items must be fixed on paper
before the patches they govern start: the Class-P "never in debt" claim has no
mechanism where code is written (B1), and P4c cannot meet its own gate as ordered (B2).
The majors are wrong statements or missing sequencing in the three mechanisms. Each has
a concrete fix.

---

## BLOCKERS

### B1. §6 / §10: Class P is "kept level per patch, as today", and today's mechanism is the one that produced #71. v2 also grows Class P into files that gameplay patches edit.

**Evidence.**
- Class P's discipline is "kept level PER PATCH on main, as today (reference → C++ →
  CUDA in the same change, bit-identity gates in the suite); never in debt" (§6 table).
  The gates are the `test_cuda_*` wrappers. Every one of them skips when the checkout
  has no `cpp/build_cuda/breach_physics*.pyd` (`tests/cuda_harness.py:42`
  `cuda_pyd`, `:74` `cuda_available`).
- 1 of the 39 folders under `.claude/worktrees/` holds a CUDA build today
  (`140-wind-q-threshold`, counted 2026-11-07). The master rule puts all code in a fresh
  worktree. So a Class-P change in an ordinary patch passes `pytest tests -q` with every
  Class-P gate SKIPPED.
- #71 is exactly this: a Class-P twin (CUDA combustion) left behind by arc #54. Lens 4's
  B1 said so (`critique_4 …:59-64`). v2 answers B1 for Class T (the ledger, the doors) and
  leaves Class P on "as today". The manifest maps only Class-T twins (§6, "maps each
  Class-T twin").
- v2 widens Class P. After P4a the step_tail drivers are in it. After P4b, `stamp_units`
  is (`gamemap.py:2126-2131` refill included). After P4c, every `FieldEdit` policy and
  combine is (`field_edit.py:428-540, 640-768`), and so is the gas-energy seam those
  policies call (`gas_energy_deposit` from `field_edit.py:705-760`). A gameplay patch
  that adds a payload field, a FieldEdit policy or a stamp output then breaks
  `--resident` unless its author builds CUDA. §10's first row ("a gameplay change on
  the reference path: as today; the played game never waits for a kernel") becomes
  false for these files, and §10 has no row for them.

**Fix.**
1. Class-P twins enter the same manifest, with `class = "P"` and their reference paths:
   `field_edit.py`, `gamemap.py::stamp_units` and the gas-energy seam functions, the
   physics `.cpp` files each CUDA core mirrors.
2. Running the CUDA suite green writes an untracked stamp in the worktree: the HEAD SHA
   plus the gates that passed.
3. Add an ordinary-suite test that does NOT skip: when `git diff $(git merge-base main
   HEAD)..HEAD` touches a Class-P reference path, it requires a stamp at HEAD. The
   git-workflow merge step refuses the `--no-ff` merge on the same condition, so a
   Class-P patch without its twin cannot merge from a no-CUDA worktree.
4. Add one §10 row: "a change to FieldEdit, `stamp_units`, the gas-energy seam or a
   step_tail driver is Class P after P4a–c: it needs a CUDA build in the worktree".

### B2. §9 P4c: the device edit queue cannot pass "one tol-0 scene per policy row" before P3b / P3e, and its float port is thrown away at P3e.

**Evidence.**
- Every noisy edit draws at the flush from `sim.rng`:
  `mult = float(rng.uniform(low, 1.0))` (`field_edit.py:673`).
- `sim.rng` is numpy's PCG64, `np.random.default_rng(seed)` (`simulation.py:281`). Host
  phase A draws from the same stream earlier in the tick (the shot cone, the exposure
  and crit rolls), and draws again in the next tick.
- Every blast is noisy:
  - `deposit_over_tiles(..., noise=_blast_noise())` on a strength row
    (`payloads.py` in `execute_payload`);
  - `add_explosion_smoke` with `explosion_smoke_noise` 0.85 (`physics.py:150-200`).
- A device flush at P4c would therefore need numpy's PCG64 and its `uniform` double
  construction ported to the device, plus the generator state shipped H2D and D2H around
  6b every tick a noisy edit exists. §4.8 says neither.
- P4c is Class P, so this is also the played single-player game's path. That path keeps
  `sim.rng`: §4.7 moves only "match-path" draws to Philox.
- P4c's "exact IEEE port of the bridges until P3e" is a Class-P float64 device copy of
  `_combine_*`, the `_iter_region` DISC weights (`np.sqrt`, `field_edit.py:329`) and
  `amount·weight·mult` (`:676-678`). It is gated per policy row, then rewritten as
  integer at P3e (also Class P, so re-ported a second time).
- G1 does not need P4c: the physics-only scene has no units and few edits.

**Fix.**
- Move P4c after P3b and after P3e's FieldEdit item, into the P6 group. The device queue
  is then integer-only and draws Philox keyed (edit key, tile), and the throwaway port
  disappears.
- Decide, in §4.7, which draw the PLAYED non-match game's FieldEdit noise uses once the
  queue is on the device:
  - Philox on every path: one re-baseline of the single-player game's smoke texture;
    or
  - host-pre-drawn multipliers: the host sorts the records, draws one multiplier per
    surviving tile against its own `solid` / `is_vacuum`, and uploads them with the
    records. This is kept for the non-match path permanently.
- Never port PCG64 to the device.

---

## MAJORS

### M1. §5.9: "`GOLDEN_AGGREGATE` checked unmoved" at P3a is impossible.

**Evidence.**
- `GOLDEN_AGGREGATE` is `trajectory_digest` (`tests/_xarch_perfield_digest.py:591`). It
  folds `tick_digest` per tick (`tests/field_digest.py:242-250`), and `tick_digest`
  folds the unit hash (`:223-228`).
- The golden scenario has a unit: `sim.add_unit(Unit("M1", x=7, y=7, team=0))`
  (`tests/field_ab_harness.py:152`).
- The unit hash is sha256 over `repr` of the records (`:255-270`).
- P3a removes the `ap` and `n_orders` keys and changes the encoding of `x`, `y`,
  `current_hp` and `facing`. So the repr, the hash and the aggregate all move at P3a.
- The aggregate moves again at P3e: the golden marine takes heat damage (the
  re-baseline note at `_xarch_perfield_digest.py:73-78`, "before its heat death") through
  the 9c float chain that P3e makes integer.
- `_unit_subfield_payloads` reads record keys by name (`r["x"]`, `r["current_hp"]`,
  `:619-632`), so it must change in P3a too.

**Fix.**
- P3a re-baselines `GOLDEN_AGGREGATE` with written rationale.
- The check that stays unmoved is the per-field lines of the perfield dump (every
  `DIGEST_FIELDS` row per tick); the `__unit_*__` lines move.
- Say "re-baseline", not "unmoved".

### M2. §4.3: slot 2 and slot 9e are serial in the reference, so "recorded effects" applied after a wide pass gives the wrong order.

**Evidence.**
- **Slot 2.** Grenades detonate IMMEDIATELY, one after another in `self.projectiles`
  list order: `execute_payload` is called inline (`simulation.py:1918-1942`). The
  rounds' phase A then marches against the post-grenade state
  (`simulation.py:1951-1962`); `landing.resolve()` comes after. The march reads
  `gmap.solid` and `e.alive` (`combat.py` round `advance`, `:333-356`). So a wall a
  grenade destroyed this tick, or a unit it killed, changes what a round hits. Grenade 2
  also reads grenade 1's results: `open_mask` and `read_blast_air` after its wall damage,
  and the hp `apply_blast_damage` reduced.

  v2's row "one thread per round; fuse-outs recorded as effects" reads as one applier
  pass after the rounds' phase A. That is the wrong order.
- **Slot 9e.** The sub-order is serial (`simulation.py:1780-1821`):
  1. pumps;
  2. vents (immediate primitive writes);
  3. doors (`seal_tiles` / `unseal_tiles` / `destroy_wall`; reads the units through
     `_occupancy_clear`, `door_system.py:256-259`);
  4. timed charges (`execute_payload`);
  5. heat sources. These read `gmap.blast_log`, including the blasts the charges just
     appended (`payloads.py` `execute_payload` appends at its step 1). They then run the
     death payload and enqueue their heat.

  v2's row records "heat sources (deaths recorded), doors (toggles recorded), timed
  charges (triggers recorded)". A death decided in a wide kernel before the applier runs
  this tick's charges misses those blasts.

**Fix.** Add a per-slot SEQUENCE table to §4.3 (the iteration-order table below is a
start). The rules it states:
- Two-phase (wide decide, then applier) only where the reference is two-phase:
  - slot 4 / 4b → 4c;
  - the rounds of slot 2;
  - slot 5's bites. These are exact because a zombie's pass-3 decision reads only stale
    `players` positions and walls (`ai_zombie.py:62, 112-200`;
    `is_passable_block` is terrain only, `gamemap.py:2273-2285`).
- Slot 2 is `applier(grenades, list order)` → `wide(rounds phase A)` →
  `applier(rounds)`.
- Slot 9e (d) is ONE applier stage in the reference sub-order. The heat-source damage
  test runs inside it, after the charges.
- Slot 9b's spread scan runs after the slot-9 applier (see minor 6).

### M3. §4.8: the composite key `(slot, walk row or effect index, k)` omits the enqueues that cross the tick boundary.

**Evidence.**
- The flush sorts on `(field, source_id, region, seq)` and then resets `seq`
  (`field_edit.py:793, 806, 812-829`).
- The flush runs at 6b (`simulation.py:1649`). Edits enqueued at 9e of tick t therefore
  sit in the queue with the LOWEST `seq` when tick t+1's slots 2 / 4c enqueue:
  - timed charges: "Its queued FieldEdits land at the next step's 6b flush"
    (`simulation.py:1805-1809`);
  - heat sources, every tick, `heat` / `_SRC_PAYLOAD_HEAT` / DISC
    (`heat_source_system.py:243`, `payloads.py:221-224`).
- A plasma splash or a payload heat deposit at slot 2 / 4c of t+1 ties with them on the
  first three keys.
- The combines requantize after every edit (`_combine_*`), and heat saturates. Order is
  therefore observable on overlapping tiles.
- A key that sorts by slot number puts 9e after 2 and 4c. That is the reverse of the
  reference.

**Fix.**
- Key = `(enqueue phase, slot, effect index, k)`, where the enqueue phase is 0 for the
  previous tick's 9e and 1 for this tick.
- Add a gate scene: a heat source plus a payload heat deposit on overlapping tiles in
  one flush.
- Reset clears the pending phase-0 records (minor 12).

### M4. §7 P3e: the float inventory is a partial list, so "no float in the ported slots; exact device ports" is false.

**Evidence.** These synced-feeding float64 sites run in the ported slots and are not in
P3e's list:
- `apply_explosion`'s disc: `math.sqrt` falloff, the `wall_hp -= quantize(wall_damage ·
  falloff)` decrement and the `pressure · falloff · kw / 16` `wave_source` amounts
  (`physics.py:110-145`);
- `apply_blast_damage`: `math.sqrt` distance and `int(max_damage · falloff)`
  (`exchange.py:427-462`);
- `linear_weights`: `1 − sqrt(d²)/sqrt(r2)` and their float normalisation
  (`payloads.py:295-316`);
- the payload heat product `heat_mj · heat_units_per_mj · scale / ONE`
  (`execute_payload`);
- 9c heat: `phi·absorption·(1−reflectivity)`, `t_felt`,
  `env_rate·(1+k·over)·dt_tick` (`exchange.py:384-401`);
- 9c3 poison: `dps · density · dt_tick` (`exchange.py:709`);
- the spray cone's float amounts and target (`combat.py:1391-1500`);
- grenade flight and its target (`simulation.py:1056-1078`, `Projectile`);
- the DISC weights of `_iter_region` (`field_edit.py:329`).

"mitigation and hp" covers only the step after these chains.

**Fix.** A per-site table in §7, each row deciding one of:
- integer (a P3e re-baseline); or
- IEEE transcription (correctly rounded `+ − × ÷ sqrt`, Python's operation order,
  `--fmad=false`, serial sums in reference order), gated by lockstep.

Every IEEE row names the applier device function it lands in. Drop "no float in the
ported slots" unless every row reads integer.

### M5. §6: the P / T boundary is undefined for the applier and the device functions shared between the classes.

**Evidence.**
- Class P lists "the device edit queue". Class T lists the slots, entities and combat.
- The applier executes both. It applies the edit queue at 6b (§4.3 slot 6b, §4.8), which
  is Class P because the played `--resident` game runs it from P4c. It also runs
  `destroy_wall` / `seal_tiles` / `execute_payload`, which only training runs.
- Some device functions are shared:
  - the gas-energy seam device functions (`gas_energy_deposit` / `_mint` / `_retire` /
    `_refresh_cell`, called from the FieldEdit path and from `destroy_wall`,
    `gamemap.py:2638, 2705-2707`);
  - the per-env books array (written by Class-P cores and by Class-T applier functions);
  - the table upload (materials and gases are already uploaded for Class P; weapons and
    the rest are Class T).
- The ring is Class T. It is read only by the digest and the rewards, and the played
  game never uses it. That is correct, but it is never stated.

**Fix.**
- Add one table to §6 listing every device artefact with its class:
  - **P:** the fold table, the stream, the books array and the channel enumeration, the
    edit-application path of the applier, the seam device functions, the materials and
    gases upload;
  - **T:** the applier's topology, payload, combat, bite and entity functions, the ring,
    the store, the bridge, the gameplay table upload.
- A physics patch that changes a shared seam function moves the Class-P twin in the same
  patch (B1). It creates Class-T debt for every T function that calls the seam, and the
  ledger lists that debt.

### M6. §6: a PATH-based `git diff level_at..HEAD` misses transitive helpers and new modules.

**Evidence.** Lens 4's drift cases:

| drift | caught by |
|---|---|
| a `Unit.__init__` attribute | P3a's `__slots__`, if `unit_schema.py` / `unit.py` are mapped |
| a `combat.py` branch | the path diff, if mapped |
| a new status kind | the path diff (`status.py` tuple), plus the bank door |
| a new entity class | the bank door: new files map to no existing twin, so the diff is silent |
| a config column | the uploaded schema, plus the diff of the code that reads it |

What escapes is a change inside an unmapped helper that a ported law calls:
- `unit_fixed.py` (quantize, `atan2_rad`);
- `movement.py::default_speed` / `footprint_samples`;
- `gamemap.is_passable_block` / `has_los` / `find_burst_walls`;
- `species.py` (Python mitigation constants: `ZOMBIE_MITIGATION`);
- `weapons.py`'s load-time normalisation;
- `blast_expansion.py` / `blast_air.py`;
- `events.py` / `factions.py`;
- code defaults in `getattr(CFG.x, "y", default)`.

A hand-written path list per twin is a remembered list, which is what lens 4's B1
wanted gone.

**Fix.**
- Derive each twin's reference set mechanically: the `src/` import closure of the
  slot's entry functions, or the files executed by that slot under the lockstep scene.
- Add an ordinary-suite scan test: every `src/simulation/*.py` module the match path
  imports is in some twin's set or in a `NOT_TWINNED` list with a reason (the repo's
  "every X is registered" pattern, `test_trace_reader_invariance.py`). A new module then
  fails the suite until it is classified.

### M7. §3 G1: "physics-only device vs physics-only farm" does not say which scene.

**Evidence.** The benchmark doc uses the phrase for two different scenes:
- the stacked probe: "physics only: no units, no controllers", each copy with its own
  duct and two vents (`residency_benchmark …:225-231`);
- the match profiles: "physics-only ('all idle': teams and horde on `idle`)" (`:156`),
  which is units present.

They differ by `stamp_units`, slots 2–6 and 9c on both sides. Other open points:
- At G1 the device backend still runs 9e (vents, doors, heat sources) on the host per env
  (P6b is later), at ~0.06 ms per env-tick (`:146`), plus the bus for the planes the
  vents write.
- "Scaled by the farm's physics share (~0.8)" scales only the farm. The device's own
  gameplay cost (P6–P8: ~10 more slots of launches, the applier's serial stages)
  is assumed zero.

**Fix.** State G1's scene as no units, the v1 ship's entities ON, on both sides. The
device side's host 9e and its bus are counted and reported as a separate line. Scale the
projection on BOTH sides:
- the farm by its measured physics share;
- the device by an allowance for its gameplay launches and applier stages (taken from
  the P5 launch count and the §2.3 latency floor).

Say that idle units are excluded because they make the G1 device side R1 (host Python
per env), which §3 already places below R0.

### M8. §4.4 / §5.9: the device oracle has no entity export.

**Evidence.**
- `tick_digest` folds the `__entity__` and `__signals__` sections whenever
  `n_entities > 0` (`tests/field_digest.py:229-240`).
- Every v1 ship has vents and ducts, whose plenum ledger is ENTITY_SECT-digested
  (CLAUDE.md vent row). Doors, timed charges, heat sources and the airlock controller
  carry `runtime_digest_rows` (`entities/actuators.py:198-206`, `door.py`, `charge.py`,
  `heat_source.py`).
- §5.9's digest input is "a D2H of the store and the ring". §4.4 names a Unit ↔ store
  bridge but no entity bridge. From P6b the device cannot produce the digest it is gated
  on.

**Fix.** Name a schema-driven entity SoA → serializer export, the sibling of the unit
bridge. It reads the entity registry's field kinds and runtime rows and emits
`entities/serialize.py`'s `name|int64` rows and the signal payload, at checkpoint
cadence. §5.9's D2H becomes: planes + store + ring + entity rows + signals.

### M9. §4.3: "effect counts per tick are tens, so one thread per env is cheap" counts effects, not the work inside them.

**Evidence.**
- One `execute_payload` is O(r²) tile work, and so is the flush after it.
- For `[payloads.frag_standard]` (strength 1, radius 2; `r_s = 1`, `heat_r2 = 4`,
  `weapons.py:591-598`):
  - `apply_explosion`'s disc is 13 tiles with up to 9 `wave_source` TILE edits each
    (`physics.py:132-145`);
  - plus the heat and smoke TILES deposits over a 13-tile walk;
  - about 160 tile operations.
- For `[payloads.demolition_c4]` (strength 12, radius 8, `heat_r2 = 36`):
  - about 200 disc tiles, so about 1 800 `wave_source` records;
  - plus about 113-tile heat and smoke deposits;
  - about 2 000 to 2 500 records to sort and apply.
- One GPU thread is latency-bound at roughly 0.3–0.5 µs per dependent global access.
  That is ~0.1 ms per frag tick and ~1–3 ms per C4 or cook-off tick.
- The vent ledger runs every tick: about 30–60 dependent accesses per vent, so 0.3–1 ms
  per env for a 20–40-vent ship.
- Every batch tick waits for the slowest env. Against a 3070 batch tick of ~20–27 ms
  (N = 64, §2.3) that is a few percent. Against the A100's ~6–13 ms, the C4 ticks
  approach 20–40 %.

**Fix.**
- State this bound in §4.3.
- Add an "applier ms per batch tick (max over envs)" row to `tick_stage_profile`.
- Keep the ORDER serial per env, but run each effect's tile set as a block. The tiles of
  one edit are disjoint, the combine is per tile, the books are integer atomics, and the
  noise is Philox keyed per tile. The same holds for the reach walk as an in-block
  frontier.
- Switch to the block form when P7c measures the applier above an agreed share. Not
  every pass parallelises: `apply_explosion`'s disc stays serial, because its row-major
  `destroy_wall` changes `solid` for later tiles' stencil checks (`physics.py:118-140`).

---

## Per-slot iteration order (check 1a)

| slot | law (code) | reference order | order-dependent? | v2 device form | knowable on device without host? |
|---|---|---|---|---|---|
| 1 | `tick_events.clear`, `reset_trace_books` (`simulation.py:1554-1559`) | — | — | ring head, books reset | yes |
| 1b | `_consume_direct_intents` (`:1011-1046`) | `units` list = id order (ids assigned in `add_unit` order, `:502-510`); throw appends to `projectiles`; USE toggles `want_open` | throw spawn order: yes; USE is XOR, so commutative | per-unit kernel | yes: spawn ordinal = (tick, unit id) |
| 2 grenades | `_update_projectiles` (`:1918-1942`) | `projectiles` list (spawn) order; `execute_payload` IMMEDIATE, each sees the previous; BEFORE rounds | yes | "fuse-outs recorded as effects" (ambiguous, M2) | yes, with a persistent spawn ordinal |
| 2 rounds | `b.advance` + `landing.resolve()` (`:1951-1962`) | `bullets` list (spawn) order for phase A against post-grenade state; resolve in record order | yes | one thread per round + applier | yes: spawn ordinal + stable compaction |
| 2b | `tick_statuses` (`status.py:309`) | `sorted(units, key=id)`; statuses in list order; DoT packets immediate on own unit | no across units; events in id order | one thread per unit | yes: compaction by id |
| 3 | `_update_player_movement` (`:1964-2016`) | id order; own state; terrain-only passability | no | one thread per unit | yes |
| 4 phase A | `process_shooting` (`combat.py:766`) | id order; RNG in walk order (numpy shared stream until P3b); own-row writes immediate; effects recorded | RNG order until P3b, else no | one block per env | yes after P3b |
| 4b phase A | `process_sprays` (`combat.py:1545`) | id order; `spray_ticks_left` written immediately; deposit recorded | no | same block, after 4 | yes |
| 4c | `SimultaneousEffects.resolve` (`simultaneous.py:118-136`) | record order: shooters by id × their records, then sprayers by id; hit / status-if-alive / face / world (`chew_wall`, `cover.chew`, `execute_payload`, spray deposit) | yes (first packet to zero is the kill; a second chew on a destroyed tile is a no-op) | applier | yes: key (phase, shooter id, k) |
| 5 | `update_zombies_tick` (`ai_zombie.py:41-200`) | pass 1 and pass 2 (fixed point) order-free; pass 3 in id order over requested ids; nearest-target ties to the lower id; bites IMMEDIATE, decisions independent of earlier bites | bites only | wide + applier | yes |
| 6 | `stamp_units` | MAX stamps; per-tile refill | no (refill = integer floor division, exact at these magnitudes) | stamp kernel | yes |
| 6b | `EditQueue.flush` (`field_edit.py:812-829`) | stable sort `(field string, source_id, region, seq)`; `seq` spans the previous tick's 9e; tiles in `_iter_region` order; noise per surviving tile in that order | yes (requantize per edit, saturation, clamps) | applier | only with M3's key; noise needs Philox (B2) |
| 9 | burn-through (`simulation.py:1667-1674`) | row-major (C++ and CUDA sort, `cuda_fire.cu:560`); `destroy_wall` serial | yes (seed reads neighbours) | wall law + applier | yes: row-major compaction |
| 9b | `find_burst_walls` (`gamemap.py:2405-2502`) | post-9 state; stable sort by −spread, ties row-major; cap `burst_max_per_tick` (16); destroy serial | yes | "wall law per cell" (incomplete) | yes, with a per-env top-k stage (minor 6) |
| 9c | `apply_environmental_damage` (`exchange.py:350-421`) | id order; own unit; reads TickCredit | no; events in id order | one thread per unit | yes |
| 9c2 | `apply_wind_knockdown` | id order; status only | no | one thread per unit | yes |
| 9c3 | `apply_teargas_blind`, then `apply_poison_dose` | teargas over all units, then poison over all units, each in id order | no; poison events in id order | one thread per unit | yes |
| 9d | `apply_temperature_ignition` | per cell | no | per cell | yes |
| 9e | `simulation.py:1780-1821` | (a) emit, sensors (ordinal); (b) logic (ordinal); (c) door inputs; (d) pumps (ordinal) → vents (duct ordinal; returns then supplies, ordinal; immediate primitive writes, `vent_system.py:539-551`) → doors (ordinal; span row-major) → charges (ordinal) → heat sources (ordinal; read `blast_log`); (e) swap | yes (shared tiles, `blast_log`) | "per class … recorded" (M2) | yes, as one serial stage |
| end / 10 | heat / rad fills; `tick_credit.clear()` (`:1893`); `MatchRules.on_tick_end` → `convert_marines_to_zombies` (`ruleset.py:275-280`, `ai_zombie.py:215-225`) | conversion in id order, per unit | no | "done kernel; ring sealed" (conversion missing, minor 1) | yes |

---

## MINORS

1. §4.3 slot 10 omits `convert_marines_to_zombies`. It runs every tick on the match path (team → 255, `is_zombie`, `alive`, hp = `zombie.hp`, activated, name) and is a per-unit kernel. The row also omits the end-of-tick heat / rad / TickCredit clears.
2. §4.5 calls trigger a latch. `set_trigger` HOLDS (`live_trigger` persists, `simulation.py:948-958`). Fix its decimation semantics identically on both backends, or backend equality breaks.
3. ExplosionEvent is not in `_SYNCED_EVENT_TYPES` (`field_ab_harness.py:250`). Say whether the digest's view of the ring filters to UnitHit / UnitKilled or deliberately adds explosions (a re-pin). ExplosionEvent's float `strength` needs an integer field.
4. Check 2: damage is already on the Q16 grid (`quantize_hp_delta` = dequantize(quantize()), `unit_fixed.py:105-116`), and so is hp. Integer records for them are a lossless re-encoding, not a P3e behaviour change. Positions are not on the grid before P3c, so P3a keeps `round(v·1e9)` for `x` / `y` until P3c, or the oracle goes lossy.
5. §5.9's "a deliberate re-pin" is five re-pins: P3a (record shape), P3b (draws), P3c (positions), P3d (horde), P3e (combat integers). Each needs its own rationale.
6. §4.3 9b: the order is a per-env stable top-16 by spread (an exact integer compare after dequantize), ties row-major, computed after the slot-9 applier. That is a selection stage, not "the wall law per cell".
7. §4.3 9e "a serial stage per duct": running ducts in parallel is exact only if no tile is touched by two actuators (pump transfer source and port, vent apertures). Either make the whole 9e (d) chain serial, or add a load-time disjointness door.
8. The applier's write set misses:
   - `cover.chew` (CoverRuntime objects, `combat.py:349, 1240`; no v1 ship has cover, and the bank door covers it);
   - `render_dirty` and `_sky_mask_dirty`, written by every `on_tile_changed` (`gamemap.py:1897, 1903`; dropped on the device);
   - `div_source_sat_hits` / `div_source_cut_q` (books);
   - the door's per-tile `d.hp` list (capacity = max span).
9. §4.2 books: the int64 range argument critique 1 M3 asked for is absent. `gas_energy_books` and `solid_energy_books` are cumulative Python ints since map build.
10. §4.2 "closure at checkpoint cadence": the EOS counters are assigned per tick. Say whether the device keeps window accumulators or checks only the checkpoint tick, which needs Σ `gas_energy` at that tick's start.
11. `n_destruction_seed_sum`'s device form is the destroyed tile's own bulk delta, because it is the only cell that changes inside the whole-grid bracket (`gamemap.py:2571, 2762`). State it.
12. Reset must clear the per-env pending state. List it as part of the copied tick-0 state: the 9e-enqueued edit records, `div_source`, `expansions`, `blast_log`, rounds, projectiles, the possessed set.
13. Per-env capacity with a counted failure flag is stated only for the ring. It is also needed for the edit-record buffer, the per-shooter effect buffer (pellets × records), `blast_log` and projectiles.
14. §4.3 "the host keeps reset's decision (mask from done at checkpoint cadence)" contradicts §4.1 / §4.2's in-step auto-reset by the device's done mask. A terminal tick inside a decimation step needs one rule that both backends follow.
15. §9 P6 lists slot 6, which is already P4b. Say that P6 re-points the stamp kernel at the device store.
16. P10: `tests/cuda_harness.py` hardcodes Windows paths (`anaconda3\envs\data\python.exe`, `*.pyd`, `cudart64_*.dll`). Name its port in P10.
17. §5.3 live lockstep in pytest is a `cuda_*_check.py` subprocess that holds BOTH the reference (the CUDA build's CPU path) and the device: one build, one process, the recording in memory. Say so, and say the pre-flight does the same inside the training process.
18. `recent_attackers` is read by nothing on the match path (it serves the OnePhaseWEGO stance) and is not hashed. Make it `REFERENCE_ONLY`, not a capped column.
19. P6 needs P3a, P3c and P3e's mitigation / hp part only. P3b gates P7a (and the noise, B2) and P3d gates only P7d. The plan puts two HUMAN-TESTs ahead of P6 that it does not need.
20. `test_twin_debt` "covers every Class-T twin": say how twins are enumerated without a build (a source-marker scan of `cpp/src/cuda_*`).
21. Mid-port (P6–P7) the host slots also need the planes they write (`chew_wall` → `destroy_wall`, `execute_payload`'s immediate writes). So today's resident bus persists beside the unit bridge until P7. State it.
22. The ring and decimation: the digest reads one tick's records and the rewards accumulate over the step. A checkpoint tick inside a decimation step needs the ring kept per tick, or decimation 1 under lockstep.

---

## QUESTIONS (not grounded)

- Does any v1 loadout or bank template carry a strength ≥ 9 row (`demolition_c4`, `magazine_cookoff`)? That decides whether M9's block form is needed before G2.
- Does `gen_ship` ever put two actuator ports (a vent aperture, a pump source or port) on one tile? That decides minor 7.
- Does `cuda_s8a_check`'s book comparison read only the host dicts? After P4c the played game's `field_edit*` seam channels are booked on the device and must reach the host for its closure gates.
- Is `[payloads.*] expansion_m3` on a non-strength row (`start_expansion`, the multi-tick `gmap.expansions`) refused by the bank door? No shipped training row uses it (`blast_expansion.py:34`).
