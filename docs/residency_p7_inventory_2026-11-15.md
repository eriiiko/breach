# P7 / P6b inventory: the remaining gameplay slots as they are (2026-11-15)

Read-only inventory for re-planning residency P6b, P7a, P7b, P7c and P7d
(#29). Source: worktree `29-p7-replan` at main `ca2c85a9`. Every statement
cites `file:line` on that commit; paths are relative to the repo root.
"Not determined" says why where the code does not settle a point; an
inference is marked "(inferred)". No plan, no judgement of the design.

It continues `docs/residency_p6_slot_inventory_2026-11-14.md` ("the P6
inventory", at `43115fec`) and does not repeat it. Conventions are that
document's: "apply_packet (the set)" is its line 18; "S" / "D" / "R" are
schema groups; "the design" is `docs/residency_batching_design_2026-11-07.md`.
Parts C, E and the entity half of Parts A and D were gathered by three
read-only sub-investigations and spot-checked here (the applier kernel
signature, `env_sink`'s callers, `non_resident_planes`, `ChainedSlots`' N,
the ring's kind refusal).

## 0. What changed since the P6 inventory (slots 2, 4, 4a, 4b, 4c, 5, 9e)

The laws of these slots are unchanged in shape. Every `simulation.py` line
number moved; the conductor today: slot 2 `simulation.py:2111` ->
`_update_projectiles` `2520-2569`; 4 `2155-2158`; 4a `2165-2167`; 4b
`2177-2178`; 4c `2180`; 5 `2184` -> `_run_zombie_slot` `682-711`; 9e
`2349-2396`; the clears `_clear_tick_buffers` `2458-2515`; 10 `2428-2435`.
Commits that touched these laws since `43115fec`:

| commit | what moved | effect on the P6 inventory |
|---|---|---|
| `e68d5317` U1 hooks | the cone through `unit_traits.cone_q16` (`combat.py:1333`, `1452`); the blow through `strike_damage_q` (`combat.py:2043`); the melee push through `push_impulse_q` (`combat.py:2129`); the bite through `strike_damage_q(z)` (`ai_zombie.py:312`); the zombie walk through `hook(z, gait)` (`ai_zombie.py:149-154`) | slots 4, 4a, 5 now read the `traits` block (S columns); a device form needs the hook accessors (they exist in the library, `cuda_unit_lib.cuh:106-116`) |
| `11dade1d` PX poison | 9c3 only | none for these slots |
| `9f1f4be4` ref debts | a detonated grenade leaves `sim.projectiles` at `ContinuousRealtime.on_tick_end` (`ruleset.py:289-290`); `melee_intent` possesses its unit | the P6 inventory's Part E items "projectiles never pruned" and "melee_intent not possessed" are SETTLED |
| `c64cb4d9` docstrings | `Simulation.step`, `charges.tick`, the spray columns, `process_shooting`'s draws, the ignition path, slot 9c | six of the P6 inventory's Part E stale-docstring lines are fixed |

Still true from the P6 inventory: slot 2's grenades are serial and immediate,
its rounds two-phase (`simulation.py:2559-2569`); 4 / 4a / 4b record on one
`SimultaneousEffects` resolved at 4c (`simulation.py:2133`, `2180`); slot 5's
pass 3 bites at once against a `players` list built at the slot head
(`ai_zombie.py:221`, `290-323`); 9e's sub-order (`simulation.py:2349-2396`).

---

## Part A: the host state these slots own

Columns: type / currency | writer (slot) | reader (slot) | lifetime | bound |
digest (T = tick digest, F = field digest, E = entity serializer) | schema.

### A1. `sim.bullets` / `BulletInFlight` (`combat.py:266-337`)

List on the Simulation (`simulation.py:401`), replaced by the survivors each
tick (`simulation.py:2569`); appended in slot 4 by `fire_burst`
(`combat.py:1384-1385`) in shooter-id x round order. NOT in any digest
(`tests/field_ab_harness.py` captures units, events, charges only, `:222-249`);
no schema, no capacity.

| attr | type | written | read | note |
|---|---|---|---|---|
| `shooter` | Unit REFERENCE | ctor `:297` | march `:481` (`e is self.shooter`) | identity exclusion from hits |
| `shooter_id` | int | `:298` | packets, draws `:345` | |
| `weapon`, `ammo` | WeaponDef / AmmoDef REFERENCES | `:299-300` | `range_tiles`, `crit_*`, `damage`, `dtype_id`, `ap`, `expose_q`, `wall_damage`, `glow`, `name` | rows, not ids |
| `payload` | PayloadDef REFERENCE or None | `:316-318` | detonation `:611-625` | resolved once at spawn |
| `angle_q` | Q16 rad | `:304` | crit arc `:543` | |
| `rx_q`, `ry_q` | Q16 tiles | `:305-306`, `:452-453`, rollback `:476`, `:512` | march | |
| `prev_ix`, `prev_iy` | int tile | `:307-308`, `:516` | cover tile `:504`, payload fallback `:621-625` | |
| `remaining_steps` | int | `:309`, `:518` | `:445`, `:439` (launch test) | starts at `weapon.range_tiles` |
| `speed_q16`, `budget_q16` | Q16 | `:310-311`, `:442-444` | `:443` | fraction carries |
| `march_cover` | list of `CoverRuntime` or None | `fire_burst :1378` | `:431-432` | float rectangles (`cover_system.py:50-53`); None on the continuous path |
| `march_facing_defense` | bool | `:1379` | `:433`, `:555-560` | float `defense_multiplier` (OnePhaseWEGO only) |
| `ox_q`, `oy_q`, `sx_q`, `sy_q` | Q16 | `:328-331` | push lever `:376-378`, hit direction `:574` | |
| `spawn_tick`, `ordinal` | int | `fire_burst :1373` (from the cone draw) | `_draw :342-346` | the round's draw identity |

The float views `rx`, `ry`, `angle` (`:350-360`) feed only the tracer, the
render events and the legacy cover. Speed: K5 `speed_tiles_per_tick` 96, not
tile-scaled; range in tiles is tile-scaled (`weapons.py:575`, `:725`), so on
ship_demo's 1/3 m tiles a K5 round (90 m, about 270 tiles) can outlive its
firing tick; whether it does in the corpus is not determined (Part E).

### A2. `sim.projectiles` / `Projectile` (`combat.py:124-216`)

List (`simulation.py:394`); appended by slot 1b `_release_grenade`
(`simulation.py:1566`) in unit-id order, and by the legacy WEGO order path
(`simulation.py:2822`); detonated grenades dropped at slot 10
(`ruleset.py:289-290`). NOT in any digest; no schema, no capacity.

| attr | type | written | read |
|---|---|---|---|
| `proj_type` | int (ORDER_GRENADE) | ctor `:138` | slot 2 `simulation.py:2531` |
| `owner_id` | int | `:141` | `execute_payload(owner_id=)` `simulation.py:2547` |
| `start_x_q/_y_q`, `target_x_q/_y_q` | Q16 (quantized from FLOAT arguments at the ctor door `:148-151`) | ctor | flight, `target_tile` `:182-186`, approach `simulation.py:2548-2549` |
| `pos_x_q/_y_q` | Q16 | `update_position :210-216` | renderer, observation (the detonation reads the target) |
| `fuse_seconds` | float | ctor `:154` | `get_detonate_tick :193` only when `fuse_ticks` is None (WEGO) |
| `fuse_ticks`, `thrown_tick` | int | ctor | `get_detonate_tick :190-191` |
| `detonated` | bool | slot 2 `simulation.py:2530` | slot 2 skip, slot 10 prune |
| `ammo_name` | str | ctor | `payload_for_ammo` `simulation.py:2542-2543` |
| `travel_speed` | float; `travel_speed_q` Q16 | ctor `:165-168` (door from the row) | `update_position :208-216` |

`update_position` reads `CFG.clock.ticks_per_second` per call (`:203`).
The 1b request record already carries every ctor value as integers
(`unit_requests.py:59-72`: start tile, Q16 target, fuse ticks, kind index,
owner, tick).

### A3. `sim.planted_charges` / `PlantedCharge` (`charges.py:35-60`)

List (`simulation.py:521`), appended only through THE door `charges.place`
(`charges.py:63-75`, capacity `CHARGE_CAPACITY` 32, `unit_schema.py:103`;
refusals in `sim.charges_refused`, `simulation.py:524`); fired and rebuilt by
`charges.tick` at 9e(d') (`charges.py:121-137`). In the TICK digest as
`CHARGE_SECT_V1` rows while one exists (`tests/field_ab_harness.py:179-186`,
`242-245`); schema `CHARGE_SCHEMA` (`unit_schema.py:797-805`): `x`, `y`,
`det_tick` (sentinel `NO_TIMER`), `ammo_name` (encoded as an ammo index),
`owner_id`, `approach` (2 wide, `NO_TILE`), `live`. All integers once encoded.
NOT in `sim.entities`, so not in ENTITY_SECT.

### A4. `sim.shots` / `Shot` (`combat.py:222-231`)

Render tracer list (`simulation.py:402`): `fx1..fy2` FLOAT, `time` FLOAT
(`real_time`), `duration` FLOAT from config. Appended by the march
(`combat.py:582`) and the beam (`:1553`); expired at the tick-end clears by a
float compare (`simulation.py:2513-2515`). Never digested; no sim reader.

### A5. Per-tick logs on the GameMap

| store | type | writers | readers | lifetime | digest |
|---|---|---|---|---|---|
| `gmap.blast_log` (`gamemap.py:341-351`) | list of `(fy, fx, radius, wall_damage_q)` ints | `execute_payload` when `radius > 0` (`payloads.py:458-470`; the Q16 quantized from `float(wall_damage)` each call) | 9e(d'') heat sources (`heat_source_system.py:221-239`), else cleared (`simulation.py:2393-2394`) | one tick (slots 2, 4c, 9e) | none |
| `gmap.noise_log` (`gamemap.py:352-357`) | list of `(fy, fx, r_tiles)` | `noise.emit` (`noise.py:43-51`): every trigger pull (`combat.py:1153-1154`), each spray start (`:1123-1124`, `:1210-1211`), EVERY `execute_payload` even radius 0 (`payloads.py:473-474`), each bash (`ai_zombie.py:137`) | slot 5 drain `noise.take` (`simulation.py:692`, `noise.py:54-63`) | crosses one tick boundary for 9e / 5-bash noises (ACCEPTED GAP, `noise.py:21-27`) | none |
| `gmap.damage_credit` / `TickCredit` (`damage.py:219-275`) | dict `(channel, y, x) -> attacker`; channel `"heat"` (str) or a gas slice int; first claim wins (`setdefault`, `:258`) | `_claim_tiles` / `_claim_disc` in payloads (`payloads.py:79-94`, `123`, `200`, `224`, `527`), the spray cells at 4c (`combat.py:1750`, `1752`) | 9c, 9c3 (`exchange.py`; device: `credit_at`, a linear scan of the `credit_claims` input, `cuda_unit_damage.cuh:308`) | cleared at the tick-end clears (`simulation.py:2510`) | none |
| `gmap.div_source` (`gamemap.py:363`) | int32 Q16 1/s plane | `blast_expansion.add_source` (`blast_expansion.py:182-193`) from `start_source` / `start_expansion` / `take_source_plane` | slot 7 (`physics_runner.py:995-998`) | one tick (a 9e blast leaves it to the next) | F: `DIV_SOURCE_SECT_V1` while non-zero (`tests/field_digest.py:149`) |
| `div_source_sat_hits`, `div_source_cut_q` | Python ints | `add_source :190-191` | diagnostics | persistent | none |
| `gmap.expansions` (`gamemap.py:366`) | list of `[tiles list, s_q, ticks_left]` | `start_expansion :212-213` (rows WITHOUT `strength` but with `expansion_m3`) | `take_source_plane :224-228` at slot 7 | MULTI-TICK | none (`gamemap.py:359-362`: "none of the three is digested") |

### A6. The FieldEdit queue (`sim.edit_queue`)

`FieldEdit` (`field_edit.py:175-194`): `field` str, `region` enum, `coords`
(a DISC radius is a FLOAT, e.g. `payloads.py:125`, `physics.py:96`), `amount`
FLOAT, `amount_q` int or None, `mode`, `falloff`, `channel`, `clamp` FLOAT
pair, `noise` FLOAT, `scale` an (h, w) int64 PLANE REFERENCE or None (the
blast air read, `blast_air.py:128-130`, `payloads.py:509-520`), `source_id`,
`weight_den`. Writers in the remaining slots: slot 2 (payloads), 4c (payload
rounds, spray cells), 9e (timed / planted charges, heat sources). Ordered at
the flush by `(field, source_id, region, seq)` (`field_edit.py:826-838`);
9e's edits wait for the next tick's 6b (seq spans). Not in any digest
(no reference in `tests/field_digest.py` or `tests/field_ab_harness.py`).
source_id namespace: explosion 1, explosion smoke 2, payload gas 3, ignite 4,
spray heat 5, spray gas 6, payload heat 7 (`physics.py:49-50`,
`payloads.py:74-76`, `combat.py:1594-1595`).

### A7. `SimultaneousEffects` and `ExposureLanding`

`SimultaneousEffects._pending` (`simultaneous.py:92-96`): a list of tuples
holding Unit REFERENCES, `DamagePacket` objects, the events list, the impulse
table object, and for `_WORLD` a Python CALLABLE with its args
(`simultaneous.py:141-148`): `chew_wall`, `CoverRuntime.chew`,
`execute_payload` (with the units list, the queue, the payload row),
`_enqueue_spray_cells` (with a list of `(y, x, div)`). One per attack slot
(slot 2's rounds `simulation.py:2561`; slots 4 / 4a / 4b `:2133`), applied in
recording order (`:151-182`). `ExposureLanding` (`exposure.py:286-314`): lists
of `(unit, dtype, add)` and dead units, landed together after the last
effect; raises skip a unit dead at landing (`:311`). Not digested; their
effects are.

Effect kinds and what each writes at resolve: `_HIT` -> `apply_packet(...,
exposure_landing=landing)` (`:165-168`); `_STATUS` -> `apply_status` if alive
(`:169-171`, `188-191`); `_FACE` -> `turning.set_target` (`:172-174`);
`_PUSH` -> `unit_impulse.add` if alive (`:175-178`); `_WORLD` -> the call
(`:179-181`).

### A8. The horde's per-env state

| store | type | writer | reader | lifetime | digest |
|---|---|---|---|---|---|
| per-zombie S columns `zombie_activated`, `zombie_noise_y/x/ticks`, `zombie_anchor_y/x`, `zombie_move_accumulator`, `last_melee_tick` | int | slot 5 | slot 5, obs | persistent | T (unit section) |
| `sim.horde_state` / `HordeState` (`horde.py:345-423`) | `fields` dict keyed `("prey", fp[, "fearless"])` / `("noise", fp, y, x)` -> int32 (h, w); `tile_ok`, `tile_ok_fearless`, `bashable` bool planes; `serial`; `settings`; `doors` tuple; `floods` | `begin_tick` (drop at `tick % field_refresh_ticks == 0` or a `_tile_serial` move, `:362-375`); lazily at first need (`mask :377-391`, `_field :393-399`) | descent, bash tile, `field_at` | MULTI-TICK between drops (`field_refresh_ticks = 2`, design §4.9) | none ("never synced, never digested", `:346-350`) |
| `sim.horde_settings` | `HordeSettings` (`horde.py:109`), map-quantized | reset | slot 5 | per map | none |
| `sim._zombie_requests` | set or None | `drive_zombies` | slot 5 filter (`simulation.py:693-701`) | one tick | none |

The prey field is flooded from the `players` list of the tick it was first
needed (`horde.py:401-413`, seeds = prey centres) and SURVIVES into the next
tick when that tick is not a refresh tick and no tile changed (`:369-375`);
the mask and the bashable doors are built from the live map and door states
at that first need (`:381-391`) (inferred: the field a zombie descends at
tick T+1 can hold tick T's prey positions, depending on which tick first
needed it).

### A9. Entity runtimes (9e; P6b)

Holders and order: `sim._doors` (sorted by ordinal, `door_system.py:179-200`,
always built), `sim._ducts` / `sim._vents` (ordinal, `vent_system.py:258-332`),
`sim._signal_bus` (only if wires / logic / sensors, `signal_bus.py:102-160`,
slots sorted `(ordinal, name)` with names as strings `:158`),
`sim._logic_nodes`, `sim._sensors`, `sim._pumps` (built ONLY when the bus
exists, `simulation.py:352-353`), `sim._timed_charges`, `sim._heat_sources`
(ordinal). Nothing spawns at runtime (`level_loader.py:309`); every runtime is
rebuilt at `_reset_internal`. No count caps (only `CHARGE_CAPACITY` exists).
E = in ENTITY_SECT_V1 (`serialize.py:121-171`): declared synced fields from
`e.fields` (load constants: int, q16, bool, enum index, entity_ref ordinal,
`serialize.py:77`, `141-147`), a free `alive|` row (`:148`), then
`runtime_digest_rows`.

| class (file) | runtime state (type; writer -> readers) | E rows | values an int64 table cannot hold as is |
|---|---|---|---|
| Door `DoorRuntime` (`door_system.py:123-176`) | `alive` bool (9e(d) `:236`); `state` 0/1/2 (`:237`, `254`, `264`); `want_open` bool (1b `_direct_use` `simulation.py:1576`, 9e(c) `:1943-1945`, `unit_requests.apply_to_host :286`); `hp` list of Q16 (copied from `wall_hp` at open `:238`, `252`; restamped `:262-263`); `span` list of tiles (row-major, load); `_span_set_cache` set | `state`, `want_open`, `hp_0..n-1`, `alive` | `span`, `hp` variable-length lists; `inst` reference |
| Pump `PumpRuntime` (`pump_system.py:43-149`) | `at_target` 0/1 (9e(d) `:142-149`); `alive` never cleared | `at_target` | `inject_slots` / `extract_slots` lists; `source` tuple or None |
| Duct `DuctRuntime` (`vent_system.py:136-177`) | `o2_raw`, `n2_raw`, `trace_raw[5]`, `sink[5]`, `e_plenum`, `e_wipe` (UNBOUNDED Python ints, `e_plenum` may be negative), `rail_lo_hits`, `rail_hi_hits` (9e(d) `_duct_sweep :378-534`) | 6 named + `trace_0..4`, `sink_0..4` (`vents.py:105-121`); int64 checked only at serialization (`serialize.py:88-95`) | unbounded ints |
| Vent `VentRuntime` (`:180-237`) | `accum` int in [0, tps) (9e(d) `:227-237`); `guard_skips` diagnostic | `accum` | `role` str; `duct` reference |
| Timed charge (`timed_charge_system.py:63-111`) | `ticks_to_fire` int (9e(d') `:194-203`); `fired` 0/1 (`:193`) | `ticks_to_fire`; `fired` only for heat charges (`entities/charge.py:103-104`) | `payload` PayloadDef reference |
| Heat source (`heat_source_system.py:76-115`) | `hp_q16` Q16 (9e(d'') `:235`); `alive`, `enabled` (set False in `_die :206-208`; the digest keeps `fields["enabled"]`) | `hp` | `death_payload` reference |
| Sensors (`sensor_system.py:70-150`) | none (load geometry; output is a bus `pub` slot) | none (SIGNAL_SECT) | `channel` Enum; `disc` list |
| Logic nodes (`logic_nodes.py`) | `FilterRuntime.ema` unbounded int (`:142-195`); `AirlockControllerRuntime.state` 0..6, `dwell` int (`:378-379`); evaluators hold `predicate` LAMBDAS (`:35-42`) | `ema`; `state`, `dwell` | lambdas, None slot ints |
| SignalBus (`signal_bus.py:46-99`) | `pub`, `stg` numpy int64 (9e(a), 9e(d) pump, 9e(e) swap; 9e(b)) | `pub` non-`alive` slots in SIGNAL_SECT_V1 (`serialize.py:174-197`) | `index` dict (lookup only) |
| Cover (`cover_system.py:31-57`, not in 9e) | `hp_now`, `alive` via `chew :122-135` (called from the march `combat.py:475`, beam `:1528`) | `alive`, `hp_now` (a second `alive|` row beside the free one) | x0..y1 FLOATS |

Entity digest fold: `tick_digest` folds blake2b of ENTITY_SECT and
SIGNAL_SECT only when `n_entities > 0` (`tests/field_digest.py:229-238`);
`registry_hash` rides the carrier but is not hashed.

---

## Part B: the shared laws

"Callers" are slots among the remaining ones (2g = grenades, 2r = rounds,
4, 4a, 4b, 4c, 5, 9e). "Float" lists arithmetic in floats on the call path
(a `p3e-door` quantization per call counts: the device would read a table
row instead).

| law | callers | inputs | writes | draws | float / identity / data-dependent loops |
|---|---|---|---|---|---|
| `execute_payload` (`payloads.py:396-558`) | 2g (`simulation.py:2544`), 4c `_WORLD` from a payload round's stop (`combat.py:622-625`; the round may be fired in 4 or still flying in 2r, whose landing is the same call), 9e(d') timed (`timed_charge_system.py:198`) and planted (`charges.py:101-104`), 9e(d'') heat-source death (`heat_source_system.py:210`) | map, queue, units LIST, tile, PayloadDef, `rng` (unused, `:439-441`), owner, approach | everything in its sub-steps below; `ExplosionEvent` (render, FLOAT fields `:549-558`) | none | `float(wall_damage)` per call (`:469-470`); `heat_mj * heat_units_per_mj(gmap)` (`:531-532`) and `blast_base_soot_g` (`blast_air.py:133-143`) float products per call; units walked in LIST order |
| `apply_explosion` (`physics.py:60-158`) | via `execute_payload` (radius > 0) | radius, pressure, wall_damage (FLOATS quantized per call `:121-122`) | `wall_hp` and `destroy_wall` IMMEDIATE inside the row-major disc loop (`:123-134`); enqueues atmosphere DISC (`:94-99`), smoke-clear DISC (`:106-113`), 9 `wave_source` TILE edits per open tile (`:140-158`) | none | the wave loop reads `solid` AFTER earlier tiles' `destroy_wall` in the same loop (order-dependent within one blast); only `_BLAST_WALL_MATERIALS` take damage (`:38-39`) |
| `exchange.apply_blast_damage` (`exchange.py:495-545`) | via `execute_payload` (unit_damage > 0) | units list, centre, radius (FLOAT quantized `:517`), max damage | apply_packet (the set) per unit in list order, `indirect=True`, NO `exposure_landing` (`:540-545`) | none | `CFG.combat.blast_damage_threshold` per call (`:530`) |
| `flash_blind.apply_flash_blind` (`flash_blind.py:87-118`) | via `execute_payload` (`blind_radius_q > 0`) | units, tile, payload `blind_*` | `statuses` (BLINDED) through `apply_status`, never shortened (`:113-118`) | none | `has_los` per unit; `quantize_scalar(u.facing)` (a float view round trip, exact) `:110` |
| reach walk `blast_expansion._walk` / `reach_tiles` (`blast_expansion.py:115-141`) | via `execute_payload` (strength rows, twice: heat r2 and source r2, `payloads.py:492-496`) and `start_expansion` | `open_mask` (`:86-90`), centre, integer r2, approach | none (returns a sorted list) | none | BFS frontier (deque), a dict of visited tiles, length set by geometry; `walk_seeds` `min(..., key=)` tie by order (`:103-112`) |
| `start_source` / `add_source` / `start_expansion` (`blast_expansion.py:182-213`) | via `execute_payload` | walked tiles, `expansion_m3` | `div_source` (saturating, counted), `gmap.expansions` (multi-tick) | none | `_source_q` divides FLOATS per call (`:176-178`); `r2 = ceil(float(expansion_radius)**2)` (`:205`) |
| `read_blast_air` (`blast_air.py:106-130`) | via `execute_payload` | walked or disc tiles | none; returns `heat_scale_q16` and an (h, w) int64 smoke-scale PLANE | none | `blast_air_settings()` re-quantizes config floats per call (`:61-84`) |
| `deposit_over_tiles` / `linear_weights` (`payloads.py:299-337`) | via `execute_payload` (strength rows: smoke, heat) | tile list, int weights, `total_q` | one TILES FieldEdit (`weight_den`) | noise at the flush (FIELD_NOISE) | `noise` float quantized per call (`:330`) |
| `emit_gas`, `ignite_ring`, `deposit_heat`, `add_explosion_smoke` (`payloads.py:97-229`, `physics.py:161-202`) | via `execute_payload`; `deposit_heat` also 9e(d'') each living heat source (`heat_source_system.py:246-247`, FLOAT amount, no `amount_q`) | | DISC FieldEdits with FLOAT radii and amounts; credit claims | smoke noise at the flush | float amounts quantized at the queue door |
| `inject_bulk_gas` (`payloads.py:255-296`) | via `execute_payload` (`bulk_gas` rows) | disc minus `vent_system._aperture_blocked` | `gas`, `gas_energy` + mirror IMMEDIATE through `inject_gas_n_vec`, books `payload_bulk_gas*` | none | per open tile; raises `OverflowError` past int32 (`gamemap.py:3387-3390`) |
| gas-N pump primitives `inject_gas_n(_vec)`, `extract_gas_n(_vec)`, `transfer_gas_n`, `_air_withdrawal` (`gamemap.py:3336-3560`) | 9e(d) pumps and vents; via `inject_bulk_gas` (2g / 4c / 9e) | tile, N vector, rails | `gas` all slices, `gas_energy`, mirror, books | none | Python ints; NO device form (Part C) |
| `BulletInFlight.advance` (the march, `combat.py:382-627`) | 4 (`fire_burst :1380-1383`), 2r (`simulation.py:2563-2566`) | map `solid`, `material` (cover table), units LIST, own state | records `_WORLD chew_wall` / `cover.chew`, `_HIT`, `_PUSH`, `_WORLD execute_payload`; appends `sim.shots` and render events at once | `Salt.EXPOSURE`, `Salt.SHOT_CRIT` via `rng.at(salt, shooter_id, round_index(age, ordinal))` (`:339-346`), lazy | step loop up to `n_steps` (data); first hit = first unit in LIST order whose footprint holds the tile (`:480-486`); shooter excluded by IDENTITY; `defense_multiplier` float on the legacy path only |
| `chew_wall` (`combat.py:238-259`) | `_WORLD` at 4c and 2r's landing (march, beam), slot 5 bash IMMEDIATE (`ai_zombie.py:134`) | tile, `wall_damage` | `wall_hp`, `destroy_wall` at <= 0 | none | `wall_fixed.quantize_scalar(float(wall_damage))` per call (`:257`); no-op on air (`:255`) |
| cover chew `CoverRuntime.chew` (`cover_system.py:122-135`) | `_WORLD` (march `:475`, beam `:1528`), only under physical cover (OnePhaseWEGO) | | cover `hp_now`, `alive` | none | FLOAT rectangles; not on the continuous path (`march_cover` None) |
| `spray_cone_tiles` / `deposit_spray_cone` / `_enqueue_spray_cells` (`combat.py:1598-1760`) | 4b (decide), 4c (`_WORLD` enqueue) | apex, Q16 aim, range, cone | heat / gas TILE FieldEdits with `amount_q`; claims | none | per tile `has_los(physical=True)` (`:1679`); `math.isqrt` exact; amounts quantized from FLOAT row values per call (`:1721-1723`); `own` set for membership only |
| `has_los` (`gamemap.py:2459`) / `rays_clear` / line of fire | 4 (WEGO targeted `:1104`, auto-fire `:1269`), 4b (cone), 2g/4c/9e via flash, 5 (trigger `ai_zombie.py:244`, sprint `:79`), 9e(a) motion sensors | `solid`, `sight_screen` | none | none | Bresenham walk, length = ray length; NO device form |
| `noise.emit` (`noise.py:43-51`) | 4, 4b-start (in 4), 2g/4c/9e via `execute_payload`, 5 (bash) | tile, radius | `noise_log` append | none | append ORDER is load-bearing: `_hear` ties to the earlier noise (`ai_zombie.py:96-99`) |
| credit claims `TickCredit.claim` (`damage.py:249-258`) | via payloads (2g, 4c, 9e), 4c spray cells | channel, tiles, owner | dict, first claim wins | none | dict keyed by a str or int channel |
| `turning.set_target` | 4c `_FACE` (WEGO paths only: tile-target shots, sprays, melee order), 5 (`Unit.face_towards`, `ai_zombie.py:132`, `165`, `408`) | | `facing_target_q` | none | `face_towards(float(nx), float(ny))` quantizes whole tiles (exact) |
| `unit_impulse.add` (`unit_impulse.py:488-503`) | 4c `_PUSH` (round push `combat.py:362-380`, recoil `:1284-1296`, melee push `:2137`) | table, unit, j, lever, twist | `imp_x/y/l` | none | integer sums commute |
| `apply_status` | 4c `_STATUS` (melee stun), flash (2g / 4c / 9e) | | `statuses` | none | |
| exposure landing | 4c and 2r's landing only (`simultaneous.py:162-182`) | | `exposure` | none | blast packets inside a resolve bypass it (`exchange.py:540-545`) |
| horde field reads `HordeState.prey_field` / `noise_field` / `flood` (`horde.py:263-423`) | 5 | mask (`walkable_tiles`, `bashable_door_tiles`, `feared_tiles`), seeds | the cache | none (WANDER is a seedless hash, `horde.py:510-517`) | C++ BFS when an engine is bound (`horde.py:268-274`); lazily built per tick through a per-zombie CLOSURE (`ai_zombie.py:346-354`) |
| the bite `apply_packet(..., mark_killed_by_zombie=True)` (`ai_zombie.py:313-323`) | 5, IMMEDIATE | | apply_packet (the set) + `killed_by_zombie` | none | |
| `destroy_wall` (`gamemap.py:2635`) | via `apply_explosion` and `chew_wall` (2g, 4c, 5, 9e), 9e(d) door external destruction (`door_system.py`) | | topology (device seam exists, Part C) | none | float planes written by `on_tile_changed` |
| `seal_tiles` / `unseal_tiles` (`gamemap.py:3040-3312`) | 9e(d) doors only (`door_system.py:253`, `258`, `259`) | span | `gas` (all slices, trace saturation counted), `gas_energy_move`, `material`, `on_tile_changed`, zeroes `atmosphere`, `wave_p`, `wind_x/y`, `flow_vx/vy`, FLOAT `ripple` / `ripple_v` set 0.0, `temperature` (neighbour floor mean), solid books; `unseal` may write `is_vacuum` / `is_ambient` (`:3226-3239`) | none | refusals `ValueError` / `SealBlocked` / `OverflowError` all before mutation (`_seal_blockers :2935-3021`, a FLOAT `permeability > 0.0` compare `:2966`); NO device form |

**Re-entry.** `execute_payload` never calls itself, but a heat source's death
payload appends a new `blast_log` disc that the same 9e(d'') loop then reads
(the `discs` list grows while iterating, `heat_source_system.py:226-239`), so
one blast can chain through several sources in one tick (`tests/test_heat_source.py`
"chained engines"). A blast's `destroy_wall` of a door tile is reconciled by
the door sweep only at 9e(d) (whole-door rule), i.e. the SAME tick for a slot
2 / 4c blast and the NEXT tick for a 9e(d') charge (inferred from the sub-order,
`simulation.py:2366-2390`).

---

## Part C: the device side today

All Class T except the edit queue and the stamp (Class P).

**C1. Per-env serial applier** (`src/simulation/applier.py`, header
`cpp/src/applier_layout.h`, `cpp/src/cuda_applier.{h,cu}`). Record int64
`(kind, y, x, arg)` (`applier.py:56`); ONE kind, `DESTROY_WALL` = 1
(`:59-60`; any other code counts a fault, `cuda_applier.cu:66-67`); `E_CAP`
256 (`applier.py:62`). Books (9, `:75-96`): `n_destruction_seed`,
`ge_retire`, `ge_mint`, `ge_destroy_seed`, `solid_destroy_wall`,
`tiles_changed`, `applied`, `dropped`, `faults`. 27 PLANES (`:103-131`) and 4
CLEAR_PLANES (`heat`, `rad_net`, `rad_amb`, `rad_flux`, `:138-143`). Adding a
kind: a row, `tools/gen_unit_layout.py --write`, rebuild, a branch in
`k_applier`'s if-chain (`cuda_applier.cu:59-68`). Kernel `<<<dim3(1, N), 32>>>`,
thread 0 walks (`cuda_applier.cu:44-47`, `157`); the fan-out is not built.
REACH of an effect function: the 27 planes (`TopoEnv`), the env's device
tables, its book row (`cuda_applier.cu:44-46`). It CANNOT reach the unit
store, the event ring, the request list, the claims table, or the clear planes
(separate `k_tick_end_clears`, `:72-81`). Records are staged only from the host
(`FieldApplier::stage`, `:121-132`; `applier.pack`, `applier.py:261-276`); no
device producer appends. Overflow: first `cap` applied, rest counted,
`EffectOverflow`; an operand out of range writes nothing, `ApplierFault`
(`unit_kernels.py:527-538`). `non_resident_planes()` (`applier.py:155-160`)
evaluates to `sight_screen`, `heat_atten`, `light_atten`, `rad_net`,
`rad_amb`, `rad_flux` (sub-investigation's evaluation, not run).

**C2. Topology seam** (`cpp/src/cuda_topology.cuh`, namespace `topo`):
`env_view` `:106`, `accountable` `:155`, `refresh_cell` `:160`,
`solid_books_cell` `:168`, `patch_face_shift` `:182`, `on_tile_changed`
`:207-245`, `destroy_wall` `:250-338` (all operands read before any write).
Host-only: `render_dirty`, `_sky_mask_dirty`, `_tile_serial` (booked as
`tiles_changed`), the solid-book sink, Wall / DoorDestroyedEvent
(`cuda_topology.cuh:32-35`). `seal_tiles` / `unseal_tiles` absent (rule:
`.claude/rules/device-units.md` "Device topology seam").

**C3. Device `apply_packet`** (`cpp/src/cuda_unit_damage.cuh:194-196`):
`apply_packet(row, tab, Packet, EventSink&, changed, killed*, bad*)`; `Packet`
= `amount_q, dtype, attacker, indirect, expose, dir_x_q, dir_y_q, ap_q, cause,
mark_killed_by_zombie` (`:111-121`). Exposure is raised AT ONCE (`:192-193`,
`256-266`): no landing / deferral exists, so START-OF-SLOT exposure (4c, 2r's
landing; `.claude/rules/combat.md` "Exposure") has no device form. Sinks:
`row_sink` (`:77`, per-row staging, every caller), `env_sink` (`:83`, direct
per-env append: NO caller in `cpp/src`).

**C4. Event ring** (`unit_events.py`, `cuda_unit_events.*`): fields `kind,
unit_id, damage_q, cause, attacker_id, dir_x_q, dir_y_q` (`unit_events.py:60-62`);
kinds UnitHit 1, UnitKilled 2 only (`:64-65`); `record_of` REFUSES every other
event type (`:190-192`), so a slot emitting ShotFired / Explosion / DoorDestroyed
/ Melee / SprayJet events cannot be shadowed with `events=True` as is
(inferred from the refusal; the harness compares against `sim.tick_events`).
`E_MAX` 1024, `ROW_CAP` 17. 10 causes incl. bullet, laser, melee, explosion
(`unit_events_layout.h:36-46`).

**C5. Request list** (`unit_requests.py:59-86`): `kind, unit_id, tick,
tile_x, tile_y, target_x_q, target_y_q, ticks, item, approach_y, approach_x,
door, accepted`; kinds throw 1 / plant 2 / use 3; `ROW_CAP` 3, `R_MAX` 256;
`apply_to_host` (`:252-290`) builds a `Projectile`, offers each plant to
`charges.place` (must agree), flips `want_open`. Plant capacity by
`k_plant_accept`, one thread per env after the gather (`cuda_unit_verbs.cu:494-498`).

**C6. Device unit library** (`cuda_unit_lib.cuh`): row / table rebasing
(`unit_row :71`, `env_tables :75`), `row_valid`, `row_alive`, the counters,
`put`; hooks `hook` / `scaled_by` / `scaled` (`:106-116`); statuses
`composed_flags :127`, `apply_status :155`, `sweep_statuses :195`; turning
`wrap_q16`, `normalize_q16`, `turn_law_q16`, `top_speed_within`,
`facing_cos_q16`; walk `tile_of`, `walk_allowed`, `effort_scale_q16`,
`mobility_at`, `passable_block :391`, `footprint_mobility :404`,
`default_speed_q :428`, `move_speed_q16`; records `list_view`,
`list_row_sink`, `emit_record`; verb gates `acts`, `can_act`,
`handling_ticks`, `can_throw`, `can_plant`, `weapon_row`, `magazines_carried`,
`mag_status`, `can_reload`, `shove_cooled`, `can_shove` (`:555-678`).
`cuda_unit_damage.cuh`: `attacker_value`, `mitigation_of`, `mitigate_q`,
`expose_resist_of`, `scale_q`, `apply_packet`, `footprint_of`, `reduce_max`,
`credit_at` (linear scan of the claims input, `:308`), `felt_temperature_q`,
`finish_row`. ABSENT: any falloff (`falloff_round` / `falloff_floor`; rule
`.claude/rules/fields-gas.md` "Fixed-point kits": "The falloffs have no device
twin yet (P7)"), any line of sight, any Philox draw, any distance / range test,
the march, `atan2` users (the kit's `atan2_q16` / `sin_q16` / `cos_q16` are
`FP_HD`, `fixed_point.h:1123-1167`).

**C7. Device tables** (`unit_tables.py:335`, header T = 564,
`unit_tables_layout.h:18`): 102 rows: impulse 13, shove 3, turning 2,
stability 3, status 5, exposure 6, hp class / identities 3, gas indices 4,
speed (marine) 4, material mobility 1, mitigation 4, heat 9, teargas 2,
poison 8, footprint 1, weapons 2 (`weapon_mag_size`, `weapon_reload_ticks`,
`unit_tables.py:522-528`), hand / plant 6, match 1, topology scalars 7,
per-material 18. NO ROW for: weapon cone, damage, range, speed, expose, AP,
dtype, loudness, `bullet_g` / `muzzle_mps` push / recoil, crit, melee damage
/ push / arc, `rof_interval_ticks`, archetype, `burst_ticks`, ammo rows; any
`[payloads.*]` column, `[blast]`, `[blast_air]`, flash / `blind_*`; any
`[zombie]` AI value, the horde settings, zombie speeds; noise radii; salts.

**C8. Device edit queue** (Class P; `serve_edit_queue`
`physics_runner.py:935-955` -> `pack_for_device` `field_edit.py:927-1007` ->
`cuda_edit_queue.cu`). The HOST still: enqueues and stable-sorts
(`field_edit.py:816`, `826-838`); quantizes every edit once at `_edit_door`
(`:962-981`); WALKS every region in `_iter_region` (TILE, DISC with a Q16
radius, BEAM, RECT, TILES; LINEAR weights via `unit_fixed.falloff_round`,
`:318-418`); reads the `scale` plane per tile (`:992`); bounds products
(`:904-906`, `993-996`); sets the noise key (`seq`, `:978`; Philox key and tick
from `MatchDraws`, `:1002-1005`). Device: K1 per tile (contribution + keyed
Philox), K2 per env in row order (skip masks, combine, gas-energy seam)
(`cuda_edit_queue.cu:154-176`, `240`, `401`). A device PRODUCER of edits
would lack: the region walk, the falloff, the `_edit_door` quantization, the
sort, and any device append into the edit / tile buffers (host-built
`_edit_buffer`, `physics_runner.py:2099`). Fallback reasons: a `water_depth`
edit, a slice out of range, an integer past 64 bits, a product past range,
noisy edits without `MatchDraws` (`field_edit.py:890-999`). Field ordinals
(lexicographic): atmosphere 0, fire 1, gas 2, gas_energy 3, heat 4, smoke 5,
water_depth 6, wave_source 7; modes ADD 0 / REMOVE 1 / MAX 2; regions TILE 0,
DISC 1, BEAM 2, RECT 3, TILES 4 (`field_edit.py:149-160`, `886`).

**C9. Device Philox.** `cpp/src/philox32.h` (`philox32_4x32_10`, `_draw`,
`_uniform_q16`, `_angle_q16`, `_below`, `_make_key`, `_make_counter`,
`:113-182`), used only by `cuda_edit_queue.cu` (FIELD_NOISE, `:53`, `171-176`).
No unit kernel includes it. No device form of `MatchDraws.at` / `.next`
(`draws.py:142-156`; `.next` counts per `(salt, id)` per tick in a host dict)
nor of `cone_offset_q16` / `roll_q16` (`draws.py:186-196`). No C++ `Salt`
enum (Python `rng_salts.py:38-47`; the edit queue gets the salt as a launch
argument, `physics_runner.py:2100-2102`).

**C10. Device forms of the other host pieces.**

| piece | where | device form |
|---|---|---|
| sight lines | `cpp/src/sight_lines.{h,cpp}` | none (host TU in both modules, `sight_lines.h:28-30`) |
| horde flood | `cpp/src/horde_field.{h,cpp}` | none ("its device kernel is residency P7d", `horde_field.h:20-21`) |
| gas-N pump primitives | `gamemap.py:3398-3560` | none (Python only) |
| `div_source` | writer `blast_expansion.py:182` | consumer only, `K_div_source` (`cuda_eos_resident.cu:144-152`); the plane is uploaded every tick (`:1171-1180`) |
| stamp | `cpp/src/cuda_stamp.cu` | yes, Class P |

**C11. Resident planes.** `_RESIDENT_SYNCED` (`gamemap.py:125-158`) and
`_RESIDENT_MASKS` (`:159-262`) hold, among others, `wall_hp`, `material`,
`solid`, `gas`, `gas_energy`, `temperature`, `fire`, `atmosphere`, `wave_p`,
`wind_x/y`, `flow_vx/vy`, `is_vacuum`, `is_ambient`. Not resident and needed
by the remaining slots: `sight_screen` (sight, the seam), `heat_atten`,
`light_atten` (seam floats), `div_source` (uploaded per tick), `wave_source`
(an edit-queue field; residency not determined here), `ripple` / `ripple_v`
(seal), `rad_*` clears.

**C12. Shadow lockstep** (`tests/cuda_unit_lockstep.py`). `DeviceSlot(label,
run, columns, inputs, events, requests, outputs, host_only)` (`:390-448`),
called `run(store[, inputs][, ring][, req])` (`:421-429`). `INPUTS`: `tick`,
`order_requests`, `credit_claims`, `door_spans`, `charges_live` (`:162-168`);
`ARG_INPUTS`: `burn_walls` (`:184-186`); `OBJECT_PROBES`: `damage_credit`
(`:214-216`); any `field:<GameMap attr>` (`:346-387`). `ShadowLockstep`
(`:513-926`) wraps the label's `SLOT_WRAPS` callables, compares rows tol 0,
env i of N vs alone, coverage, ring vs `sim.tick_events`, requests vs the
hooked `_release_grenade` / `charges.place` / `_direct_use`, field outputs bit
for bit, books vs host deltas; the WRITE-SET gate traces `vars(gmap)` only
(`:219-291`, `227`). So `blast_log` / `noise_log` (GameMap lists) are seen by
the write-set gate but have no INPUT or OUTPUT kind; `sim.bullets`,
`sim.projectiles`, `sim.planted_charges`, entity runtimes and
`SimultaneousEffects` are Simulation / host objects the harness neither
captures nor sees. `ChainedSlots` (`:937-1037`) is N = 1 only
(`build_store([driver.sim])`, `:951`). Scenes are hand `LevelData` tilemaps or
shipped levels (e.g. `tests/cuda_unit_topology_check.py:178-197`).

**C13. Twin manifest** (`tests/twin_manifest.toml`): 12 Class-P rows
(water, eos, trace_tail, div_source, energy_flux, radiation_sweep, temperature,
fire, combustion, stamp, edit_queue, water_tail; `:126-508`) and 14 Class-T
(exposure_decay, integrate_turn, wind_impulse, order_apply, order_expiry,
zombie_conversion, movement, status_tick, heat_damage, gas_rows, verbs,
match_done, burn_through, tick_end_clears; `:535-826`). `[not_twinned]`
(`:77-124`) names for P7a `attack_resolver`, `combat`, `draws`, `events`,
`simultaneous`, `vision`, `weapons`; for P7c `blast_air`, `payloads`,
`physics`; for P7d `horde`, `noise`; for P6b `vent_system`,
`entities/registry`, `entities/schema`, `entities/vents`; "setup only on the
corpus" `door_system`, `heat_source_system`, `signal_bus`,
`timed_charge_system` (P6b); never a twin: `line_of_fire`, `obs_spec_v1`,
`action_spec_v1`, `sensor_accessor`, `team_picture`, `match_session`, `stats`.

**C14. Profiler labels** (`tools/tick_stage_profile.py:60-109`). Unported
labels: `2 projectiles` (`_update_projectiles`), `4 shooting`, `4a melee`,
`4b sprays`, `4c resolve`, `5 zombie AI`, `9e entities` (nine callables, not
`charges.tick`), `9e(d') planted charges`, `9b`, `9d`. Reserved
`APPLIER_LABEL = "  7 applier (per-env serial, P6 -- reserved)"`, printed as
zeros (`:52-55`, `349`, `374`). Slot 2 is ONE label over grenades and rounds;
9e is one label over nine sub-steps.

---

## Part D: per remaining slot

Notation: "needs" = does not exist on the device today.

**2g — grenades** (`simulation.py:2525-2549`). Tables: A2 (read / mutate),
A5 (all five via `execute_payload`), A6, units (apply_packet, BLINDED).
Laws: `update_position`, `execute_payload` and every sub-step (Part B).
Usable: the request list's throw record (every ctor value), apply_packet (no
landing needed: grenades detonate immediately, not in a resolve), the topology
seam (`destroy_wall` of `apply_explosion`), `apply_status`. Needs: a
projectile table with a capacity (none exists; the list was unbounded until
the prune), a detonation path through the applier that reaches unit rows and a
ring (C1 cannot), the falloffs, the reach walk, line of sight (flash), payload
/ blast / blast-air table rows, the div_source writer, the pump primitive
(bulk gas), a device edit producer (region walk, sort, door), the blast_log /
noise_log / claims as device tables, ExplosionEvent in the ring.

**2r — rounds in flight** (`simulation.py:2559-2569`). Tables: A1, a slot-local
A7, A4 (render), A5 via a payload round. Laws: the march, `chew_wall`,
`push_on`, payload at the stop. Phase A order-free (keyed draws per round);
landing in list order. Needs: a bullet table (no schema, no capacity; refs to
shooter / weapon / ammo / payload to become ids), the march and its draws
(`Salt.EXPOSURE` / `SHOT_CRIT` via `.at` with `round_index`), a deferred
exposure landing, weapon / ammo rows, cover exposure per material (a
`cover_exposure_q16` column; not a device table row today). Shares everything
with 4 / 4c.

**4 — shooting phase A** (`combat.py:963-1135`). On the continuous path only
`_directional_fire` runs (orders empty; the WEGO targeted / auto-fire branches
with their float `math.sqrt` range test, `:1099`, `:1267`, are dead there).
Tables: shooter S columns (P6 inventory slot 4 list) incl. `traits`, A1
(append), A4, A5 `noise_log` (IMMEDIATE), A7 (records). Laws: `mag_gate` /
`mag_spend` / `mag_settle`, `unit_traits.cone_q16`, `rng.next(SHOT_CONE |
BEAM_CONE)`, the march or the beam (trace-gas Beer-Lambert over `gas`), recoil,
`noise.emit`, `start_spray_burst_directional`. Usable: `mag_status`,
`magazines_carried`, `weapon_row`, `can_act`, `composed_flags`, `hook`.
Needs: Philox `.next` semantics (index local per (salt, shooter) per tick),
`cone_offset_q16`, the march, the beam, weapon / ammo rows, a bullet table, a
device noise log, a recorded-effect list (the applier has one kind).

**4a — melee** (`combat.py:2095-2141`). Order-free decisions; records `_HIT`,
`_STATUS` (stun), `_PUSH`. Laws: `can_melee` (device form exists? only the
P6.4a gates listed in C6; `can_melee` is NOT among them), `melee_arc_target`
(atan2, `wrap_q16`), `strike_damage_q`, `push_impulse_q`. Needs: `can_melee`,
melee rows (damage, arc, push, dtype, status kind, rof), the effect list.

**4b — sprays** (`combat.py:1808-1900`). Own-row writes at once; records one
`_WORLD _enqueue_spray_cells` per sprayer. Laws: `spray_cone_tiles` (`has_los`
per tile), credit claims, FieldEdit TILE ADDs. Needs: line of sight, a device
edit producer (TILE edits; no region walk needed), claims table writes, spray
rows (`heat_deposit`, `gas_species`, `gas_amount`, `range_tiles`, half-angle,
`burst_ticks`). No draw, no noise (the noise is 4's).

**4c — resolve** (`simultaneous.py:151-182`). Applies A7 in record order:
apply_packet WITH the landing, status, face, push, and `_WORLD` (chew,
payload, spray cells). Usable: apply_packet (immediate exposure only),
`apply_status`, the topology seam for `destroy_wall`. Needs: a deferred
exposure landing; an applier whose effects reach unit rows, the ring and the
claims; effect kinds hit / status / face / push / chew / payload / spray-cells;
`set_target` and `unit_impulse.add` as effect functions (the shove's
per-target sum is the precedent, `cuda_unit_verbs.cu:371`, `463-470`).

**5 — zombie AI** (`ai_zombie.py:176-410`). Tables: A8, A5 `noise_log`
(drain, append), units (prey hp), entity doors (bashable set: `state`,
`alive`, `span`). Laws: `has_los` (trigger, sprint), the flood + mask + cache,
`descend`, `bash_tile`, `chew_wall` (IMMEDIATE), the bite (IMMEDIATE
apply_packet), `footprint_samples` / `default_speed` (device forms exist:
`footprint_mobility`, `default_speed_q`), wander hash (Philox block, seedless).
Order-dependent pass 3 (P6 inventory slot 5). Needs: the flood on the device,
the masks (walkable, bashable from door runtimes, feared), the multi-tick
cache semantics (A8), line of sight, the noise log, `[zombie]` rows, zombie
speed rows, the bite and bash through an applier that reaches rows, Philox
for the wander hash.

**9e — entities** (sub-steps, `simulation.py:2349-2396`):

| sub-step | reads / writes | laws | device gap |
|---|---|---|---|
| (a) `_signal_emit` (`simulation.py:1909-1924`), `sample_sensors` (`sensor_system.py:225-232`) | bus `pub`; field sensors read atmosphere, smoke (converted with `gas_fixed.trace_to_q16` over the WHOLE plane per call, `sensor_accessor.py:233`), water, temperature, fire, O2, solid; motion sensors read units in `sorted(..., key=id)` (`sensor_system.py:137`) | `has_los` | bus as a table, sensors, sight |
| (b) `sweep_logic_nodes` (`logic_nodes.py:554-560`) | `pub` -> `stg`; `ema` (floor shift of negatives), airlock `state` / `dwell` | lambdas per comparator | node table, the predicates as codes |
| (c) `_resolve_door_inputs` (`simulation.py:1926-1946`) | `want_open` from `pub` (close beats open) | dict lookup `_door_drives` | door table |
| (d) pumps (`pump_system.py:265`), vents (`vent_system.py:537`) | `gas`, `gas_energy`, mirror, books; duct ledgers (unbounded ints); `atmosphere` at the port | pump primitives (C10: none on device); vent floor division of a negative `e_plenum` (`:448`) | pump primitives, duct / vent tables, int64 bounds on the ledgers |
| (d) doors (`door_system.py:215-265`) | `material`, `wall_hp`, units' footprints (occupancy); `destroy_wall`, `unseal_tiles`, `can_seal_tiles` + `seal_tiles` | seal / unseal (C2: absent) | seal / unseal device functions with a refusal path; door table with variable spans |
| (d') timed charges (`timed_charge_system.py:180-203`), planted (`charges.py:121-137`) | `temperature` mirror (heat trigger), countdown; `execute_payload` | all of 2g's | as 2g, plus a charge table (CHARGE_SCHEMA exists) |
| (d'') heat sources (`heat_source_system.py:215-247`) | drains `blast_log` (growing list), source hp (`falloff_round`), `_die` -> `execute_payload`; `deposit_heat` with a FLOAT amount, no `amount_q` (`:246-247`) | as 2g + DISC heat edit | as 2g; the float deposit |
| (e) `swap_node_signals` (`signal_bus.py:85-90`) | `pub[node] = stg[node]` | | bus table |

**Tick-end pieces not yet ported.** The clears are ported except the credit
claims (`damage_credit` host-only in `SLOT_CLEARS`, "device form is the
claims table (P7)", `tests/cuda_unit_topology_check.py:134-137`) and the
render tracer expiry. Slot 10's grenade prune is host-only ("a host table
until P7b", design 4.4b P6.2 as built). 9b and 9d have no kernel (9b is
switched off by ruling, design §12.7 per 4.4b).

**The entity part of the digest export.** Moved to P6b (design 4.4b P6.4b
as built). The bytes are A9's ENTITY_SECT_V1 and SIGNAL_SECT_V1; variable
row counts per class (door `hp_0..n-1`, timed charge `fired` only for heat
charges, cover's duplicate `alive`).

**The observation gather.** `EntityFieldAccessor` (`sensor_accessor.py:209`):
`sample` / `sample_many` / `area` raw integers; `crop` (`:328`) float32,
dequantized per plane; `observe` (`:287`) dispatches to
`obs_spec_v1.observe_rows` / `observe_full_rows`, which read `sim._doors`
(`state`, `span`, `alive`; e.g. `obs_spec_v1.py:574`, `677`, `959`, `1557`,
`1805`) and emit floats. Callers: `sensor_system.py:88` (in the tick),
`rl_env.py:786`, `957`, `982`, `998`, `control_bots.py:606-609`, `1512`,
`control_bots_door.py:709` (read side). Its device form is P8b's.

### D-matrix: which remaining slots share a table (T) or a law (L)

Rows and columns: 2g, 2r, 4, 4a, 4b, 4c, 5, 9e-act (pumps / vents / doors /
logic / sensors), 9e-chg (timed / planted charges, heat sources), obs.

| | 2g | 2r | 4 | 4a | 4b | 4c | 5 | 9e-act | 9e-chg | obs |
|---|---|---|---|---|---|---|---|---|---|---|
| **2g** | | T(A5,A6) L(payload) | T(noise) | | T(claims, A6) | L(payload, apply_packet, flash, destroy_wall) T(A5,A6) | T(noise) L(destroy_wall) | L(destroy_wall, pump prim.) | L(payload) T(A5,A6) | T(A2) |
| **2r** | | | T(A1, A7-kinds) L(march, draws, chew) | T(A7) | | L(resolve, chew, payload) | L(chew) | | L(payload) | |
| **4** | | | | T(A7) | T(A7, spray cols) | T(A7) | T(noise) L(sight†) | L(sight) | T(noise) | T(A1) |
| **4a** | | | | | T(A7) | T(A7) L(push, status) | | | | |
| **4b** | | | | | | T(A7) L(edit producer, claims) | | L(sight) | L(edit producer) | |
| **4c** | | | | | | | L(chew, apply_packet) T(noise) | L(destroy_wall) | L(payload) T(A5,A6) | |
| **5** | | | | | | | | T(door runtimes) L(sight, chew/destroy_wall) | T(noise) | T(door runtimes) |
| **9e-act** | | | | | | | | | T(door runtimes, blast_log order) L(destroy_wall, pump prim.) | T(door runtimes) |
| **9e-chg** | | | | | | | | | | |

† slot 4's sight use is on the WEGO branches only; the continuous path's shot
uses none (`combat.py:1168-1223`).

Reading it (facts only): `execute_payload` and its sub-steps are shared by
2g, 2r (via 4c), 4c and 9e-chg; the A7 effect list by 2r, 4, 4a, 4b, 4c; line
of sight by 4b, 5, 9e-act and the flash inside every payload; the noise log
by 2g, 4, 4c, 5, 9e-chg; `chew_wall` / `destroy_wall` by 2g, 2r, 4c, 5,
9e-act, 9e-chg; door runtimes by 1b (use), 5, 9e-act and obs; the pump
primitives by 9e-act and every bulk-gas payload.

---

## Part E: gates and scenes that exist

**E1. Levels** (entity classes from each `level.toml`).

| level | holds | remaining slots it exercises |
|---|---|---|
| `levels/ship_demo` (match: 4 v 4 bots, `rifleman`, 4 zombies, seed 1, `match.toml:4-19`) | 1 duct, 2 vents (`level.toml:409-432`); 60 `[[opening]] kind="door"` are room-graph edges stamped `MAT_DOOR`, NOT door entities (`tools/gen_ship.py:310-313`) | 2, 4 (K5 rounds; frags / flashbangs when bots throw), 5 (wander, hearing, fear of fire on: `config.toml:4995`, `5004`, `5012`), 9e vents. No door bashing (needs door entities, `horde.py:320-334`), no use, no charges, no heat sources |
| `levels/playground` (match: 4 v 4, 5 zombies, 60 s) | 6 doors, 2 ducts, 4 vents, 3 lights | the ONLY shipped match with door entities (bots' use, zombie bashing); no CUDA unit gate runs its match |
| `levels/airlock_demo` | 2 doors, pump, `sensor_motion`, `pressure`, 2 deciders, `airlock_controller`, 15 wires, 1 marine | 9e doors, pump, sensors, logic, bus (the only shipped sensor / logic level) |
| `door_test`, `test_level`, `wind_studio`, `fire_studio`, `fire_tuning` | 4 / 5 / 2 / 1 / 1 doors | 9e doors |
| `explosion_studio` | 4 `timed_charge` (`studio_small/medium/large/hull`) | 9e(d') -> payloads |
| `fire_smoke_studio` | 6 `timed_charge` `studio_ignite`, a duct + 2 vents, 3 marines | 9e(d'), vents |
| `flashbang_studio` | 3 `timed_charge` (flashbang, smoke) | flash; no units; loaded by no test |
| `blast_div` / `blast_final` / `blast_quad` / `blast_smoke` | 4 / 4 / 9 / 4 `timed_charge` | payloads; only `blast_smoke` is loaded by a test |
| `smoke_light_studio`, `unhcr_vessel`, `aquarium_demo`, `wego_test` | charges / zombies / legacy | |

No shipped level holds a `heat_source`, a planted charge, or a spray / GL-6 /
hitscan / pellet user. The only loadout is `[loadouts.rifleman]`: `k5_carbine`,
frag x2, smoke x1, flash x1, `explosives = 1`, `magazines = 3`,
`melee = "rifle_butt"` (`config.toml:5177-5184`). Bot actions: shoot,
hold_fire, frag, flash, shove, melee, move, retreat, stand
(`control_bots.py:802-820`): no plant, no smoke grenade, no manual reload.

**E2. CUDA gate scenes that touch these slots (host side only today).**
Verbs scene (`tests/cuda_unit_verbs_check.py:201-436`: 3 rooms, interior /
dead / hull doors, every grenade kind thrown, two planters at a full table,
census of 51 branches); ship scenes (500 ticks, every unit gate); damage lanes
(`cuda_unit_damage_check.py:37-53`: a flamethrower jet on a marine = 4b with
credited heat, teargas, a claimed poison cloud); box horde bites
(`cuda_unit_kernels_check.py:290-334`); topology fire scene
(`cuda_unit_topology_check.py:591-660`, no door entity, no 9e); batch
invariance leg B (`cuda_batch_invariance_check.py:353-391`: a raw `add_source`
and a raw `destroy_wall`, NOT `execute_payload`); an in-memory heat source
(`cuda_edit_queue_check.py:387-389`); `fire_smoke_studio` timed charges
(`cuda_s8a_check.py:736`).

**E3. Property tests on these laws (pytest).** Payloads: `test_payloads.py`
(the replica gate `:137`, rows, falloff), `test_explosion.py` (books close, a
wall stops and a gap passes, approach side, totals in room vs corridor,
saturation), `test_reach_depth.py`, `test_blast_air.py`,
`test_blast_expansion.py`, `test_eos_div_face_form.py`,
`test_cuda_div_source.py`, `test_blast_smoke.py`, `test_bulk_gas_payload.py`,
`test_heat_ignition_proxy_payloads.py`. Flash and grenades:
`test_grenade_kinds.py`, `test_projectile_list_bound.py`. Rounds:
`test_weapons_march.py` (K5 resolves in its tick, slow rounds persist, chew),
`test_onephase_cover.py`, `test_w6_armory.py`. Simultaneous / exposure:
`test_simultaneous_shots.py`, `test_exposure.py`. Sprays:
`test_spray_weapons.py`. Melee: `test_melee_verb.py`, `test_melee_weapons.py`.
Charges: `test_plant_charge.py`, `test_unit_requests.py`, `test_timed_charge.py`
(incl. `explosion_studio` determinism `:221`), `test_timed_charge_heat.py`,
`test_heat_source.py` (chained engines). Doors / seal: `test_a6_doors.py`,
`test_a5_seal_evacuation.py`, `test_seal_clears_vacuum.py`,
`test_wave_push_door.py`. Pumps / vents / airlock: `test_b4_pump.py`, four
`test_vent_*.py`, `test_b5_airlock.py`, `test_airlock_interior_mode.py`.
Bus / logic / sensors: `test_b1_signal_bus.py`, `test_b2_nodes.py`,
`test_b3_sensors.py`, `test_b6_logic_golden.py`. Horde / noise:
`test_horde_behaviours.py` (the only test reading `noise`),
`test_horde_field.py`, `test_zombie_control.py` (pinned ship_demo and box
outcomes, `:667`, `:681`). Credit: `test_credited_events.py` (first claim
wins `:371`). Entity digest: `test_entity_digest.py`,
`test_entities_core.py`. Drivers: `tests/_p3a_sole_cause.py` (ship_demo bots,
bots_v1, the conversion box, a scripted arena with every grenade kind, a C4
planted and fired, melee, shove, reload; `:16-21`, `140-209`), its
`_p3b/_p3c/_px/_u1` siblings, `_1e_lockstep.py` (adds an interior-door
scene). GOLDEN_AGGREGATE holds no payload or grenade
(`tests/field_ab_harness.py:139-146`).

**E4. What the P6 gates recorded about the corpus** (design 4.4b as built).
P6.1: exposure rises in ship_demo's bots match "from tick ~437 (the kernel
changes rows on 23 of 700 ticks)" (design `:694-698`). P6.2: "no zombie kills
a marine in 300 ticks" (`:815-819`). P6.4a: the verbs census "counts 51
branches, any at 0 fails" (`:1013-1015`); the ship scene needs at least one
grenade in 500 ticks (`tests/cuda_unit_verbs_check.py:1032-1034`). P6.4b:
"ship_demo: 0 burn-throughs in 500 ticks" (`:1102-1106`); "no shipped door is
flammable" (`:1093`). Manifest: the corpus (ship_demo, seed 1, 300 ticks,
`twin_manifest.toml:42-46`) "consumes none [no latch] in its ticks" (`:121`);
doors, heat sources, timed charges "setup only on the corpus" (`:89`, `96`,
`118`); the first 40 ticks throw no grenade (`cuda_resident_bus_check.py:38`).
Bot tournament: 5.1 frags and 0.8 flashbangs per ship_demo match against an
idle team plus the horde (`docs/bots_zombies_speeds_design_2026-11-10.md:239-245`).
No document counts shots, bashes or in-flight rounds per corpus; whether the
300-tick corpus fires any shot or detonates anything is not determined
(`payloads.py` appears as executed in `[not_twinned]`, but the tracer records
imports too, `tools/twin_manifest_gen.py:177-180`).

---

## Part F: surprises

- In-flight grenades (`sim.projectiles`) and rounds (`sim.bullets`) are in NO digest (`tests/field_ab_harness.py:222-249`): a device twin that diverges in flight is seen only at its landing.
- The horde's field cache is cross-tick hidden state that changes outcomes: a prey field flooded at tick T from that tick's prey survives into T+1 unless T+1 is a refresh tick or a tile changed (`horde.py:369-375`, `401-413`); which tick floods it depends on which zombie first needed it (lazy closure, `ai_zombie.py:346-354`). Never digested (`horde.py:346`).
- `_run_zombie_slot` returns before `begin_tick` when there is no prey (`ai_zombie.py:221-223`) or an empty request set (`simulation.py:697-698`), so the cache is not dropped on those ticks (inferred: a later slot may reuse a field older than `field_refresh_ticks`).
- `gmap.expansions` (multi-tick source rows) and the FieldEdit queue (9e edits crossing into the next tick) are cross-tick state in no digest (`gamemap.py:359-366`).
- `apply_blast_damage` takes no exposure landing (`exchange.py:540-545`): a payload resolved inside 4c or 2r's landing zeroes a victim's exposure at once instead of at the landing.
- A planted or timed charge's credit claims (made at 9e) are cleared at the tick-end clears before its heat edits land at the next 6b, so no 9c / 9c3 reader ever sees them (inferred from the order `simulation.py:2389-2390`, `2510`; `damage.py:228-235`).
- Device `apply_packet` raises exposure at once (`cuda_unit_damage.cuh:192-193`): START-OF-SLOT exposure has no device form.
- The applier reaches only planes, tables and books (`cuda_applier.cu:44-46`); no effect can hurt a unit, emit an event or claim credit; `env_sink` has no caller (`cuda_unit_damage.cuh:83`).
- The ring refuses every event but UnitHit / UnitKilled (`unit_events.py:190-192`); slots 2 / 4 / 4a / 4b / 5 / 9e append render events into `sim.tick_events` beside them.
- Per-call float doors on the payload path: `float(wall_damage)` (`payloads.py:469-470`, `physics.py:121`, `combat.py:257`), `heat_mj * heat_units_per_mj(gmap)` (`payloads.py:531-532`), `_source_q`'s float divide (`blast_expansion.py:176-178`), `ceil(float(expansion_radius)**2)` (`:205`), `blast_base_soot_g` (`blast_air.py:133-143`), `blast_air_settings()` (`:61-84`), FLOAT DISC radii into `_iter_region` (`payloads.py:92`, `125`, `194`, `226`; `physics.py:96`, `198`).
- The heat source's per-tick deposit passes a FLOAT amount with no `amount_q` (`heat_source_system.py:246-247`).
- `apply_explosion` reads `solid` for the wave stencil after earlier tiles of the same disc were destroyed (`physics.py:123-158`): intra-blast order rests on the row-major loop.
- The march's first hit is the first unit in LIST order holding the tile and excludes the shooter by object identity (`combat.py:480-486`); the beam's `skewered` list is identity-based (`:1486`, `1536`).
- `BulletInFlight` holds object references (shooter, weapon, ammo, payload, cover list) (`combat.py:297-323`); `Projectile` reads `CFG.clock.ticks_per_second` per call (`:203`) and quantizes FLOAT ctor arguments (`:148-151`).
- The WEGO shooting branches use a float `math.sqrt` range test (`combat.py:1099`, `1267`); dead on the continuous path (orders empty), live under the WEGO rulesets.
- `start_spray_burst` (WEGO) forms its face target from floats through `atan2_q16_of` (`combat.py:1781-1783`); the directional burst is integer.
- Sensors' smoke sample converts the whole trace plane per call (`sensor_accessor.py:233`); motion sensors re-sort the units every tick (`sensor_system.py:137`).
- Duct ledgers and the filter EMA are unbounded Python ints, int64-checked only at serialization (`vent_system.py:67-72`, `serialize.py:88-95`); the vent's `e_plenum // n_bulk` and the filter's `ema >> k` rely on Python floor semantics for negatives (`vent_system.py:448`, `logic_nodes.py:193-195`).
- `seal_tiles` writes FLOAT `ripple` / `ripple_v` and decides by a FLOAT `permeability > 0.0` (`gamemap.py:2966`, `3160-3161`); `on_tile_changed` derives `solid` from a float compare.
- The heat-source sweep's `discs` list grows while it is iterated (`heat_source_system.py:226-239`): a data-dependent chain inside 9e(d'').
- Pumps, logic nodes and sensors exist only when a bus exists (`simulation.py:352-353`); an unbuilt pump hashes `at_target` 0 through a tolerant `getattr` (`entities/nodes.py:267`).
- Cover's record carries two `alive|` rows (its runtime row beside the free one, `entities/cover.py:90`, `serialize.py:148-158`), though `entities/schema.py:209-210` says names must not collide.
- Door `hp` rows are stale while a door is closed (live damage sits in `wall_hp`, `door_system.py:252`); heat-source and timed-charge runtime `enabled` is not what is hashed (`serialize.py:145`).
- `tests/cuda_unit_verbs_check.py:32` says the ship scene has doors and `tests/cuda_batch_invariance_check.py:31` says leg B has doors and a blast; ship_demo has no door entity and leg B's "blast" is a raw `add_source`.
- The corpus (300 ticks) ends before the first exposure rise (~437) and consumes no latch: twins traced from it for 2 / 4 / 4a / 4c / 5-bite will see few or none of their branches (the conversion / burn-through thin-trace gap).
- No bot plants or throws smoke (`control_bots.py:802-820`); no shipped level holds a heat source; door bashing and bots' use have one shipped match level (`playground`), which no CUDA gate runs.
- `APPLIER_LABEL` still reads "P6 -- reserved" and sits under slot 7 (`tools/tick_stage_profile.py:52-55`); the applier merged in P6.4b and serves slot 9.
- `ChainedSlots` is N = 1 only (`tests/cuda_unit_lockstep.py:951`); the write-set gate cannot see Simulation attributes (`:227`).
- Design §4.3 row "4c" lists `cover.chew` and omits `_PUSH` and the exposure landing; row "2 grenades" omits the immediate bulk-gas pump write, the flash, the noise and the `blast_log`; row "9e" names no planted charges and no heat-source chain; row "5" is silent on the field cache's cross-tick life (P6 inventory's notes still stand).
- `docs/residency_batching_critique_4_continuity_2026-11-07.md:43-45` says the K5 has `mag_size = 0`; it is 30 (`config.toml:3525`).
