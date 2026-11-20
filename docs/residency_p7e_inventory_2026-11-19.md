# P7e inventory -- the code as it stands (read-only)

main at `92f2bf3f` (2026-11-19, after the #63 swarm merge). Read-only: nothing built, run or changed.
"NOT FOUND" means I searched and did not find it. File:line are on this commit.

Key files: reference tick `src/simulation/simulation.py:2053-2510`; device faces
`src/simulation/unit_kernels.py` (2737 lines); the applier table `src/simulation/applier.py`;
the played resident tick `src/simulation/physics_runner.py:2051-2215` (`run_resident_batch`);
the batch `src/simulation/resident_batch.py`; the harness `tests/cuda_unit_lockstep.py`
(`ShadowLockstep`, `ChainedSlots` :1910).

---------------------------------------------------------------------------------------------

## A. The reference tick as a table

`Simulation.step()` (simulation.py:2053) = `_step_pre_physics` (:2098, slots 1..6b) ->
`_step_physics` (:2252, slot 7) -> `_step_post_physics` (:2268, slots 9..11). Device form =
the `unit_kernels` face (Class T unless noted). "Patch" from `tests/twin_manifest.toml`
headers and the design's As-built paragraphs.

| slot | call (file:line) | device form (face) | patch | reads / writes |
|---|---|---|---|---|
| 1 | `tick_events.clear()` :2104 | ring `begin()` (harness) | P6.3 | host event list |
| 1 | `gmap.reset_trace_books()` :2109 | NONE (host book dict) | -- | trace books (host) |
| 1 | `ruleset.on_round_start(sim)` :2115 | NONE (ContinuousRealtime: no tick work found; WEGO only) | -- | -- |
| 1a | `unit_impulse.integrate_all(sim)` :2128 | `integrate_turn(store)` uk:87 | P6.2 | unit rows |
| 1b | `_consume_direct_intents()` :2135 | `verbs(store, requests, ticks, door_spans, charges_live, fields{flight})` uk:409; then `table_appends(store, requests, grenades, charges)` uk:1225; `door_uses(store, requests, doors)` uk:1482 | P6.4a / P7b / P6b.1 | rows; request list; grenade + charge tables; door table `want_open`; FLIGHT plane (derived, see C) |
| 1b' | `unit_orders.apply_requests(sim)` :2141 | `apply_orders(store, requests)` uk:152 (requests = HOST table `requests_of(sim)` uk:130) | P6.2 | rows |
| 1c | `exposure.decay_units` :2146 | `exposure_decay(store)` uk:80 | P6.1 | rows |
| 2 | `_update_grenades()` :2151 | `grenades(store, table, applier, ticks, {flight})` uk:1245 -> DETONATE stage `detonate(...)` uk:779 | P7b + P7c.1/.2 | grenade table; PAYLOAD records; planes (payload fn); noise list; blast log; source table; edit door; claims; ring |
| 2' | `_update_rounds()` :2155 (= `_march_rounds` + `landing.resolve()`) | `rounds_march(store, effects, rounds, ticks, keys, {solid, material})` uk:1080 -> RESOLVE stage `resolve(...)` uk:768 | P7a.2 | round table; effect list; rows; planes (CHEW / PAYLOAD) |
| 2b | `tick_statuses(units, events)` :2171 | `status_tick(store, ring)` uk:296 | P6.3 | rows; ring |
| 3 | `_update_player_movement()` :2192 (or `ruleset.drive_units` -- WEGO, refused) | `movement(store, {material})` uk:181 | P6.2m | rows |
| 4 | `process_shooting(...)` :2199 | `shooting(store, effects, rounds, noise, ticks, keys, {solid, material})` uk:1046 | P7a.2 (beam NOT built) | rows; effect list; round table; noise list |
| 4a | `process_melee_intents(...)` :2209 | `melee(store, applier, ticks)` uk:791 | P7a.1 | rows; effect list |
| 4b | `process_sprays(...)` :2221 | `sprays(store, applier, cells, {solid})` uk:1331 | P7a.3 | rows; cell buffer; effect list (SPRAY_CELLS) |
| 4c | `attack_effects.resolve()` :2224 | `resolve(store, applier, fields, ring, lists)` uk:768 (RESOLVE stage) | P7a.1 | rows (landing); planes; edit door (step a); claims list; ring |
| 5 | `_run_zombie_slot()` :2228 / :710 | `horde(store, applier, fields, ring, noise, doors, tiles, cache, latch)` uk:2342 (HORDE stage) | P7d | rows; noise list (drained); door table; horde cache (cross-tick device state); planes (bash via topology seam) |
| 6 | `gmap.stamp_units(units)` :2231 | `cuda_stamp.cu` (Class P, inside slot 7) fed by HOST-flattened rows `GameMap.take_stamp_rows` gamemap.py:2118 | P4b | dyn_permeability, dyn_wave_absorb, dyn_heat_atten_q, dyn_light_atten_q, obstacles |
| 6b | `runner.serve_edit_queue` :2247 / `edit_queue.flush` :2249 | device edit queue K1 `edq_contrib` / K2 `edq_apply` (`cpp/src/cuda_edit_queue.cu:154/240`, Class P) fed by a HOST pack (`field_edit.pack_for_device` field_edit.py:927) | P4c | the edited planes + seam books |
| 7 head | `_physics_head` physics_runner.py:1007 (`take_source_plane`, `set_div_source`, `_check_trace_dt`, `reset_trace_sat_books`, solid-books sink) | SOURCES stage `take_sources(store, applier, fields, lists, source_out)` uk:669 -> `run_resident_batch(div_source_dev=)` | P7c.2 (Class T hand-over; played path uploads) | div_source; source table |
| 7 | `_step_physics` :2252 -> `run_resident_batch` pr:2051 | resident chain (Class P): stamp, edits, water, EOS, trace tail, combustion, [sky, N=1], tail (fire, sweep, temperature, SWARM last), books D2H | P4a-P5, #63 P3 | the resident planes; swarm store; bite / count planes; burn-through list |
| (9 head) | `tick_events.extend(swarm_death_events(...))` :2273 | NONE (rows come down at the books D2H; host events) | #63 P3 | host events (not in synced digest) |
| 9 | `_burn_through_walls(destroyed)` :2277 / :2512 | `burn_through(store, applier, fields, walls)` uk:819 (TOPOLOGY stage, DESTROY_WALL) | P6.4b | planes; books. Host side: render marks, Wall/DoorDestroyedEvent (render-only) |
| 9b | `find_burst_walls(max_pops=16)` + `destroy_wall` loop :2285-2293 | NOT FOUND (no scan kernel; `atmosphere_fixed.py` is a `[not_twinned]` row) | -- | atmosphere, solid, is_vacuum (scan); planes (destroy) |
| 9c | `apply_environmental_damage` :2312 | `heat_damage(store, ring, {heat, rad_flux_sweep, atmosphere, temperature, solid}, claims)` uk:313 (claims = HOST table `claims_of(sim)` uk:225) | P6.3 (+#215 body) | rows; ring |
| 9c2 | `apply_wind_force` :2329 | `wind_impulse(store, {gas, wind_x, wind_y, solid})` uk:98 | P6.2 | rows |
| 9c3 / 9c3' / 9c3'' | teargas, poison dose, bite dose, corpse :2342-2356; `poison.tick` :2362; `stress.tick` :2367 | `gas_rows(store, ring, {gas, swarm_bite_dose, swarm_count}, claims)` uk:338 -- ONE fused kernel for all of them | P6.3 (+#63 P5a/b, #215) | rows; ring |
| 9d | `apply_temperature_ignition` :2391 | `ignition(store, IGNITION_FIELDS)` uk:895 | P7.0a | fire, ignition_armed (in place) |
| 9e(a) | `_signal_emit`, `sample_sensors` :2417-2419 | `signals_emit` uk:2259 (EMIT) | P6b.3 | bus, sensor tables; planes read |
| 9e(b) | `sweep_logic_nodes` :2421 | `signals_logic` uk:2272 (LOGIC) | P6b.3 | node tables |
| 9e(c) | `_resolve_door_inputs` :2422 | `signals_door_inputs` uk:2300 (DOOR_INPUTS) | P6b.3 | door table want_open |
| 9e(d) pumps | `sweep_pumps` :2424 | `signals_pumps` uk:2309 (PUMPS) | P6b.3 | planes via seam |
| 9e(d) vents | `sweep_vents` :2432 | `sweep_vents` uk:1745 (VENTS) | P6b.2 | duct/vent tables; planes |
| 9e(d) doors | `sweep_doors` :2434 | `sweep_doors` uk:1499 (DOORS) | P6b.1 | door table; planes (seal/unseal/destroy) |
| 9e(d') timed | `sweep_timed_charges` :2441 | `timed_charges` uk:2561 (TIMED_CHARGES, payload fn at once) | P6b.4 | timed table; planes; blast log |
| 9e(d') planted | `charges.tick(sim)` :2457 | `charges_tick(store, table, applier, ticks)` uk:1265 -> DETONATE | P7b / P7c | charge table; PAYLOAD |
| 9e(d'') | `sweep_heat_sources` :2459 / `blast_log.clear()` :2461 | `heat_sources` uk:2579 (HEAT_SOURCES; empties the blast log) | P6b.4 | heat table; blast log; edit door (deposit) |
| 9e(e) | `swap_node_signals` :2463 | `signals_swap` uk:2318 (SWAP) | P6b.3 | bus |
| rec | `recorder.record(...)` :2481 | NONE (host; headless runs have no recorder) | -- | host |
| clears | `_clear_tick_buffers()` :2489 / :2526 | `tick_end_clears(bp, {heat, rad_net, rad_amb, rad_flux})` uk:834; claims list "lives one tick" (no clear face found -- `begin()` of the list); shot tracers = render | P6.4b | heat (+3 dead planes, see C) |
| 10 | `tick += 1; real_time += dt` :2496 | NONE (host scalars; every kernel takes `ticks (N,)` from the host) | -- | -- |
| 10' | `unit_orders.expire` :2501 | `expire_orders(store, ticks)` uk:160 | P6.2 | rows |
| 10 tail | `ruleset.on_tick_end` :2503 (ContinuousRealtime ruleset.py:~275: `convert_marines_to_zombies`, prune detonated grenades) | `convert_zombies(store)` uk:168 + `prune_grenades(table)` uk:1277; `match_done(store, ticks)` uk:449 (the DONE kernel) | P6.2 / P7b / P6.4a | rows; grenade table. Host side: "Z-" name prefix, momentum books |
| 11 | `team_picture.update(sim)` :2510 | NONE (host; not synced, not hashed) | -- | host |

Twin-manifest note: the signal stages (P6b.3), the hazards stages (P6b.4), the SOURCES stage and the payload
function (P7c) have NO `[twins.*]` row -- only `[not_twinned]` reasons (twin_manifest.toml:86-119:
"setup only on the corpus"). 9b, the recorder, slot 1's book reset and slot 11 have no device form at all.

---------------------------------------------------------------------------------------------

## B. How the device forms are driven TODAY

**Who calls them.** No module under `src/` calls any `unit_kernels` face, `DeviceApplier`,
`DeviceStore`, `DeviceTable`, `DeviceHordeCache` or `DeviceEditDoor` (grep over `src/simulation/*.py`:
only their own modules and docstrings). The ONLY drivers are the CUDA check scripts
(`tests/cuda_unit_*_check.py`) through `tests/cuda_unit_lockstep.py`:
- `ShadowLockstep` (:1363): the host plays every slot; around a slot it packs the rows and captures inputs
  at entry, repacks at exit, then uploads, runs the kernel over N envs, downloads, compares. The device
  never drives the game.
- `ChainedSlots` (:1910, N = 1 ONLY: `build_store([driver.sim])` :1924): a contiguous run of slots replaced
  in one twin sim. Per tick `_run_chain` (:1945): capture inputs from the host, `unit_store.pack` +
  `store.upload` (:1952-1953), stage every input list from the HOST's records (:1962-1967), run the
  kernels, `store.download` + `unpack` onto `sim.units` (:1980-1981), copy every field output back into the
  GameMap (:1985-1987), ring -> `sim.tick_events` (:1989-1991), requests -> `apply_to_host` (:1996), every
  list LANDED back onto the host list or consumed (:2001-2007), then the caller's `host_side` (:2011).
- The longest chains that exist: `[4, 4a, 4b, 4c]` (sprays_check:1097), `[4, 4a, 4c, 5]`
  (horde_check:1271), `[2', landing]`, `[1b, 1b', 1c, 2]` (objects_check), 9e whole as a plan
  (hazards_check:899), `[9e(d) doors]`, `[VENTS]`. NO chain spans slot 7, slot 6/6b, or a tick boundary;
  units, tables and lists are re-packed from the host EVERY tick (the persistent horde cache is the only
  device state that crosses a tick in a gate, `horde_cache.DeviceHordeCache(persistent=...)`).

**The played resident tick** (`run_resident_batch` pr:2051, Class P) touches no unit kernel; its own host
steps per tick, per env:
1. `_step_pre_physics` (all of slots 1..6b on the host) -- resident_batch.py:132-134.
2. `_resident_prepare` pr:1554: water pre-step + dormancy decision (host, reads `water_depth`), ambient args,
   `take_stamp_rows` (host-flattened unit footprints, gamemap.py:2118).
3. `_physics_head` pr:1007 per env: `take_source_plane(gmap)` (host), `eos.set_div_source`,
   `_check_trace_dt`, `reset_trace_sat_books`, solid-books sink wiring.
4. Entry upload `from_host(_resident_entry_names)` pr:1591: EVERY tick `atmosphere, wind_x, wind_y,
   temperature, gas, gas_energy, heat, fire, wall_hp, solid, is_vacuum, is_ambient, thermal_solid,
   flammable, material` (pr:1630); the 3 stamp planes on a tick the device stamp did not serve; the 8
   static planes + `light_atten_q` when `_tile_serial` moved (pr:1676); water planes when not dormant;
   dirty swarm stores.
5. Stamp rows H2D (pr:2124-2133); edit buffer H2D built from each env's host pack (`_edit_buffer`
   pr:2232, from `serve_edit_queue` pr:955: `queue.take_ordered()` host sort + `pack_for_device`).
6. Device chain; host sync `tick_sync_point("max_n_sub")` inside the EOS stage; the sky bracket
   (ambient N = 1: `to_host(["gas"])` / `from_host(["gas"])` pr:2186-2190).
7. Books D2H + destroyed-wall heads D2H, `tick_sync_point("books")` (cuda_resident_tick.cu:1127-1134);
   the burn-through list SORTED ROW-MAJOR ON THE HOST (`fire_destroyed_row_major`, :1380-1397; a tail
   past 64 walls is a second D2H :1392).
8. `_fold_edit_books` pr:985 (edit-queue books into host dicts); exit download
   `to_host(_resident_exit_names)` pr:1604 (`atmosphere, wave_p, wind_x, wind_y, temperature, gas,
   gas_energy, heat, fire, wall_hp, dem_acc, rad_flux_sweep` + render planes when not headless + swarm
   columns and bite/count planes while a swarm is present); `water_depth_before` copy; W6a ripple
   (host, skipped headless).
9. `_step_post_physics` on the host (slots 9..11).

**Every host step between two device slots in a device-ordered tick, as the code stands** (what P7e
must remove or keep as a latch). Counting each distinct seam once:

| # | between | host step today | where |
|---|---|---|---|
| H1 | tick head -> 1a | `tick_events.clear`, `reset_trace_books`, `on_round_start` | sim:2104-2115 |
| H2 | every unit kernel | pack / upload rows, download / unpack (`ChainedSlots` only; no resident unit store in the played tick) | lockstep:1952,1980 |
| H3 | 1b | `ticks (N,)`, `door_spans` (HOST table `unit_requests.pad_door_spans`), `charges_live (N,)` HOST scalar, the FLIGHT plane computed on the host (`combat.flight_plane` combat.py:140, a fresh copy) | uk:409, lockstep:182-260 |
| H4 | 1b' | the order request table from the host dict `sim._order_requests` (`requests_of` uk:130) | uk:130-157 |
| H5 | 2 / 2' / 4 | `ticks`, the draw key per env (`unit_draws.draw_key` from `sim.draws`), grenade / round / noise tables staged from host lists | uk:1046-1098, 1245 |
| H6 | 4 / 4b | `shooting_door(sim)` (a host check over `sim.units`' WEGO `orders` and the ruleset) inside `rounds_of` | uk:992 |
| H7 | 5 | THE LATCH `horde_cache.latch_of(sim, U)`: tick, `GameMap._tile_serial` (HOST counter), the zombie request set | horde_cache.py:161-168 |
| H8 | 5 -> 6 | slot 6 stamp rows are flattened by the HOST C++ stamp from `Unit` objects (`take_stamp_rows`); NO device builder from the unit store found | gamemap.py:2118-2143 |
| H9 | 6 -> 6b | host sort `EditQueue.take_ordered()` + `pack_for_device` (+ its range checks and the noisy-edit / water_depth fallbacks) | pr:955-975, field_edit.py:927-1005 |
| H10 | 6b -> 7 | `_resident_prepare` (water pre-step / dormancy on the host), `_physics_head` (host `take_source_plane` on the played path), the entry upload of 15+ planes | pr:1554, 1007, 1591 |
| H11 | inside 7 | `tick_sync_point("max_n_sub")`; sky bracket (ambient only) | cuda_resident_tick.cu:96 |
| H12 | 7 -> 9 | books D2H + `fold`, burn-through list sorted on the host, swarm death rows -> host events, the exit download | cuda_resident_tick.cu:1127-1397, pr:2200-2214 |
| H13 | 9 | `walls_of` / `pad_walls` build the TOPOLOGY records from the host list; the host side (render marks, `_sky_mask_dirty`, `_tile_serial`, solid-books sink, render events) | uk:486-499, 819 |
| H14 | 9b | the whole slot is host (scan + `destroy_wall`) | sim:2285-2293 |
| H15 | 9c / 9c3 | the claims table built on the host from `TickCredit`'s dict (`claims_of` uk:225: deduped + sorted), not from the device claims list | uk:225-257 |
| H16 | 9e | signal / door / duct / vent / pump / hazard tables staged from host runtimes each run; EMIT's `ticks`; the `alive` constants packed at the stage's entry (`signal_rows` refuses moving ones) | uk:1841-2052, 2259 |
| H17 | 9e -> tick end | recorder (host), `tick_end_clears` (device face exists), shot tracers (render) | sim:2481-2489 |
| H18 | tick end | `tick += 1`, `real_time`, `expire_orders(ticks)`, `on_tick_end` host side ("Z-" name, momentum books), `match_done(ticks)` | sim:2496-2503 |
| H19 | tick end | team picture (host, match only) | sim:2510 |
| H20 | every face | every `unit_kernels` face DOWNLOADS its counts (coverage, faults, refused, dropped) and raises on the host: one implicit D2H sync per launch (`_done` uk:68, `walk_effects` uk:586-641, `_object_done` uk:1192). The resident rule allows two syncs per tick ("max_n_sub", "books"). | uk:68 ff |

So: 20 seams, of which the per-launch count download (H20) recurs at every one of the ~30 launches.
Inputs come from the previous device kernel's outputs only INSIDE one `ChainedSlots` chain (and the
noise list / effect list continued between producers of one chain); everything else comes from the host
reference.

---------------------------------------------------------------------------------------------
## C. Non-resident planes

`applier.non_resident_planes()` (applier.py:567) = `PLANES + CLEAR_PLANES` (applier.py:431-506) minus
`GameMap._RESIDENT_FIELD_NAMES` (gamemap.py:130-281, incl. the swarm columns and planes). By reading,
it returns today: **`sight_screen`, `heat_atten`, `light_atten`, `ripple`, `ripple_v`, `div_source`,
`rad_net`, `rad_amb`, `rad_flux`** (9 names; I did not run it).

| plane | producer | consumers | on the device today? |
|---|---|---|---|
| `sight_screen` (bool) | `_update_caches` gamemap.py:1767; `on_tile_changed` :1958 (and its device twin in `cuda_topology.cuh`) | device sight (`unit_sight::has_los`) in HORDE, the PAYLOAD flash blind, EMIT's motion plates; host `vision.py:157` (bots, the observation's bulk sight) | NO: a slot input per launch |
| `heat_atten` (float32) | `_update_caches` :1655, `on_tile_changed` :1900 (+ device seam, as float bits) | NO reader found in `src/`, `renderer/`, `ui/`, `tools/`, `main.py` (the sweep reads `heat_atten_q`): write-only | NO |
| `light_atten` (float32 x3) | :1649, :1897 (+ device seam) | NO reader found (the sweep reads `light_atten_q`): write-only | NO |
| `ripple`, `ripple_v` (float32) | render-only W6a ripple on the host mirror (pr:1612); the seal writes 0.0 bits | render; `EXCLUDED_FLOAT_FIELDS` (not digested) | NO (host-computed by design) |
| `div_source` (int32) | `blast_expansion` (start_source / take) on the host; device payload fn + SOURCES stage | slot 7's EOS; digest SECTION `DIV_SOURCE_SECT_V1` (field_digest.py:150) | partly: the Class-T hand-over `run_resident_batch(div_source_dev=)`; the played path uploads the engine's `eos.div_source` |
| `rad_net`, `rad_amb`, `rad_flux` (int64) | NOTHING since T6 (gamemap.py:586-602: "DEAD ... nothing reads or writes it"; `rad_net` is a dead `step_tail` argument, pr:1239-1244) | NOTHING; only `_clear_tick_buffers` zeroes them (sim:2547-2560) | NO |

Planes a device slot reads that are not resident but are NOT in that list:
- `ignition_armed` (bool): written in place by slot 9d's kernel (`IGNITION_FIELDS` uk:888), DIGESTED
  (`DIGEST_FIELDS`, field_digest.py:118 ff, v2), not in `_RESIDENT_*`. `non_resident_planes()` does not
  name it because it is not an applier PLANE.
- THE FLIGHT plane (`combat.flight_plane` combat.py:140: `solid | blocks_thrown[material]`): no GameMap
  attribute; computed fresh on the host per call (slots 1b and 2). A device tick needs a device builder,
  or the walk reading `material` + a Device tables row.
- `wave_v`, `wave_source`: digested, never written after `__init__` (gamemap.py:407-408, retired), not
  resident; a device tick need not hold them, but the gate's snapshot must still read zeros.

Already on the device in the resident engine (slot 6/7): every `PLANES` name except the six above,
`heat` (CLEAR), the sweep planes, `dem_acc`, `obstacles`, the swarm columns + bite / count planes.
Caveat: "resident" today means a device copy REFRESHED FROM THE HOST MIRROR at every entry: the entry
upload rewrites 15 of them every tick (pr:1630), the static ones on a `_tile_serial` move (pr:1676), the
exit download brings 12+ back (pr:1663). In a device tick the device copy is the truth and both
transfers must go (a gate downloads for its digest only).

**Contradiction with the P7e row:** the row names "the radiation planes" as planes to make resident;
the only radiation planes `non_resident_planes()` names are the three DEAD legacy planes. The honest
change is deleting them from `CLEAR_PLANES` and their clears (sim:2547-2560), not making them resident.
Likewise `heat_atten` / `light_atten` have no reader: a device buffer the seam writes is enough (or they
leave `PLANES` with the host twin; neither is digested, so no golden moves).

---------------------------------------------------------------------------------------------

## D. The edit door, step (c)

Built:
- Steps (a) P7a.3, (b) P7c.1 / P7c.2 / P6b.4: `cpp/src/cuda_edit_door.cuh` (forms TILE, DISC
  FLAT / LINEAR, TILES; the noise's low bound; the per-tile scale; `disc_claims` / `tiles_claims`), the
  buffer `cpp/src/cuda_edit_door.h` (`EditDoor`: two record lists + per-env `seq`), the face
  `edit_door.DeviceEditDoor` (edit_door.py:255; `begin()` :282 = the 6b reset; `set_seq()` :286).
  Device producers: SPRAY_CELLS (4c), the PAYLOAD function at every call site (2, the 2' landing, 4c,
  9e(d') timed and planted, 9e(d'') death payloads) and the heat source's per-tick `deposit_heat`. Pumps
  and vents enqueue no edit. Every continuous-path host producer of a FieldEdit (`combat.py:1966/1975`
  sprays, `payloads.py`, `physics.py`, `heat_source_system.py`, `charges.py`, `timed_charge_system.py`)
  has a device form; `ruleset.py:147-180` (WEGO door explosives) and `Simulation.edit` (debug) do not.
- The door's EDIT row = `pack_for_device`'s 16 columns + `source_id` + `region` + 4 region args
  (`EDIT_COLS = 22`, edit_door.py:49-54); TILE rows `(edit index LOCAL to the env, cell, weight,
  scale)`; caps `EDIT_CAP = TILE_CAP = 8192` per env per window.
- Host references: `door_rows` (:93), `edit_of` (:162, the inverse), `replay` (:232).

How the EDIT / TILE rows reach K1 / K2 today: THEY DO NOT. The device door's rows are compared with
`door_rows` of the host's edits in the shadow, or, in a chained run, REPLAYED through the host's own
`EditQueue.enqueue` (`edit_door.replay`, which must reproduce them). The played queue then flushes on the
host, or packs AGAIN on the host: `serve_edit_queue` pr:955 -> `take_ordered` field_edit.py:830 ->
`pack_for_device` :927 -> `_edit_buffer` pr:2232 -> `apply_edits_resident` pr:2145 -> K1
`edq_contrib` / K2 `edq_apply` (`cpp/src/cuda_edit_queue.cu:154 / 240`).

What step (c) still needs, read from the code:
1. A per-env STABLE SORT of the door's EDIT rows by `(E_FIELD ordinal, SRC_COL, REGION_COL, E_KEY = seq)`
   (the host key is `(edit.field STRING, source_id, region, seq)` field_edit.py:826-828; the device
   ordinal is the lexicographic rank, `DEVICE_FIELD_ORDINALS` :886, so the two orders agree), the TILE
   rows carried with their edit, the edit index made batch-global, the env tile offsets and the per-env
   scalars `(tick_word, key lo, key hi, t_amb_raw)` built ON THE DEVICE: today all host (`_edit_buffer`).
   K1 / K2 need no change if fed that layout (the 16 packed columns; the 6 extra are dropped). NOT FOUND:
   any device sort; the only device reorder primitive is the object tables' stable compaction.
2. The range checks `pack_for_device` makes on the host (the row fits int64, `den < 2^62`, the product
   `< 2^125`, the contribution `< 2^55`; field_edit.py:979-996) and its fallbacks (a `water_depth` edit,
   a gas slice outside [0, G), noisy edits without `MatchDraws`). In the device door I found the DISC
   radius range and the weight's kit range (`cuda_edit_door.cuh` :62, :118, :159, :204) but NOT the
   product / contribution bound. A device-built row needs that check as a fault, because no host fallback
   exists inside a device tick.
3. The window: rows written from the previous tick's 9e (timed charges, planted charges, heat sources)
   through this tick's 5 are applied at this tick's 6b, so the door buffer persists across the tick
   boundary as per-env device state (nothing empties it but `begin()`).
4. `seq` as coded: resets ONLY at the 6b flush (`DeviceEditDoor.begin`) and at env reset, never at the tick
   head or end; it is the noise key, so its absolute value matters. The host twin `EditQueue._seq` resets
   in `clear()` (field_edit.py:821), called by `take_ordered` at 6b: the same rule.
5. The edit-queue books fold (`_fold_edit_books` pr:985: trace deposit / clip books, the gas-energy seam
   channels, `rng.drawn += noise count`) is host; in a device tick they stay in the books arena until the
   one books D2H (or a gate's download). K2's failure slot already throws at the books fold
   (cuda_edit_queue.cu:36-41): the one-sync pattern exists.

---------------------------------------------------------------------------------------------

## E. Host latches and tick-end pieces

What the design lets the host write per tick: the actions and the zombie request set. What enters today:

| latch | today | device entry found? |
|---|---|---|
| ACTIONS (move dir, throttle, aim, trigger, throw / plant / use / reload / shove / melee latches, flashlight) | the intent methods write `Unit` attributes on the host (the replay controller re-issues recorded intents through them); the device sees them only through a whole-row `unit_store.pack` + `upload` | NOT FOUND: no partial latch-column writer into `UnitStoreDevice` (unit_store.py:359-440 has whole-store `upload(rows)` / `download()` only) |
| commander orders | `sim._order_requests` dict -> `requests_of` (uk:130) -> `apply_orders` | a host table per tick: a THIRD latch the P7e row does not name |
| zombie request set | `sim._zombie_requests` -> `horde_cache.latch_of` (horde_cache.py:161) | yes, the HORDE latch row; but the row ALSO carries `sim.tick` and `GameMap._tile_serial`, which device slots earlier in the same tick move (2, 2', 4c CHEW, the bash itself next tick, 9e doors): in a device tick the serial must be a device counter (the `tiles_changed` book), not a host latch |
| tick / draw key | `ticks (N,)` and `unit_draws.draw_key(sim)` handed to 1b, 2, 2', 4, 4a, 9e(a), 10', done; the swarm's env words (`swarm._env_words` swarm.py:1206) | host scalars per launch; a device tick counter would remove them |
| derived host inputs | `door_spans`, `charges_live`, the flight plane | derivable on the device (door table spans, charge table count, `material` + `blocks_thrown` row) |

Tick-end and other pieces, by side:
- IN THE TICK, NO device form: slot 9b (`find_burst_walls` gamemap.py:2549: a scan, the `max_pops = 16`
  worst differentials SORTED DESCENDING, then `destroy_wall` + render events); the burn-through list's
  row-major sort (host, cuda_resident_tick.cu:1380-1397); slot 6's stamp-row builder (host C++ stamp);
  the edit sort (D.1); the physics head (`take_source_plane` on the played path, `_check_trace_dt`,
  `reset_trace_sat_books`, the solid-books sink); `gmap.reset_trace_books` (slot 1); the water pre-step
  and dormancy decision (pr:1564, reads `water_depth` on the host); the sky bracket (ambient, N = 1).
- IN THE TICK, device form EXISTS: events (the ring; render-only events have none, by rule); the claims list
  (`attack_lists.claims_list`), BUT 9c / 9c3 still take the HOST `claims_of` table (deduped, sorted:
  uk:225-257) -- the switch to the list "from P7e" (device-units.md, Attack lists) is not built;
  `tick_end_clears`; `expire_orders`; `convert_zombies` (+ the corpse rise); `prune_grenades`;
  `match_done`; the entity / signal digest export (`EntityExport` uk:2634); the unit digest from rows
  (`unit_store.section_bytes` :331 / `record_from_row` :312).
- HOST BY DESIGN (outside the tick): the recorder (absent headless), the team picture (slot 11, match only,
  not hashed), reward books, the observation (P8b), render-only events, shot tracers, the ripple, the
  conversion's "Z-" name prefix and momentum books (`convert_zombies`' host side), `render_dirty` /
  `_sky_mask_dirty` / the solid-books sink (device-units.md, topology seam "Host-only").
- Prune / compaction already on the device: the grenade table (slot 10), the round table (2'), the charge
  table (9e(d')), the source table's span compaction (SOURCES), the noise list drained at HORDE's head.
  The blast log is emptied by HEAT_SOURCES; the claims list and the cell buffer are per-tick `begin()`s.

---------------------------------------------------------------------------------------------

## F. Refusals and gaps

Refused by name in the device code (Python face raises after the launch; the kernel writes nothing for
the refused item unless noted):

| exception / code | where | condition |
|---|---|---|
| `ShotRefused` | uk:1072 (kernel count), `shooting_door` uk:1002 / 1006 / 1009 | a HITSCAN trigger (the beam, `lance_3` / `lance_5`); a ruleset that is not `ContinuousRealtime` or `drives_units`; a unit holding WEGO `orders`; a draws `tick_base != 0` |
| round encoder `ValueError` | unit_schema.py:1095 / 1104 / 1111 | a round with no spawn tick, under physical cover, under the facing defense |
| `ObjectSlotRefused` | `refuse_wego` uk:1181 / 1187; `_object_done` uk:1202 | a `drives_units` ruleset; a grenade with no integer fuse (WEGO order path) |
| grenade encoder `ValueError` | unit_schema.py:954 / 1012 | fuse out of range; travel speed not exact Q16 |
| `SpraySlotRefused` | uk:1365 | a WEGO burst (no aim angle) |
| `unit_tables` `ValueError` | unit_tables.py:1005, 1384, 1639, 1338, 1404, 1418, 1665, 2057-2065 | a spray or payload of a CONSERVATIVE gas; a non-conservative bulk gas row; a payload radius / threshold / ramp the kernels were not proved for; any row value outside its proved range (at store build) |
| `UnitKernelFault` | `_done` uk:73, `_object_done` uk:1196, sprays uk:1360 | a row operand past its proved range (e.g. the march: positions within 2^40, steps 2^24, speed (0, 2^30], age / ordinal 16 bits; a bounced leg past +-2^25) |
| `EffectOverflow` | uk:605, 812, 1033, 1209, 1372 | effect list past `E_CAP = 256` per env per launch or `ROW_CAP = 32` per row |
| `EffectRefused` | uk:617 | a record of a refused kind (`UNAPPLIED_KINDS = ()`) or refused payload row (`PAYLOAD_REFUSALS = ()`, both EMPTY); a host effect with no device kind (FACE, a cover chew) at `record_of` |
| `ApplierFault` | uk:630, 2251, 2394 | an effect operand past its range; a slot / door row / span outside its table; a sample tile off the grid; a clock period < 1; a pump seam call the reference would raise on |
| `SEAM_REFUSALS` codes | applier.py:482-494 | seal / unseal / gas-N: out of bounds, already solid, unknown material, not solid, standing water, sealed pocket, int32 overflow, duplicate tile, span past `DOOR_TILE_CAPACITY`, fault |
| `BulkGasOverflow` | uk:623 | a bulk-gas deposit whose slice would pass int32 (the reference raises mid-loop: a finding) |
| `LedgerRangeRefused` | uk:1782 | a duct ledger outside its proved range (`vent_ledger_range` uk:1705) |
| `SignalTableRefused` | uk:1864 / 1870 / 1876 / 1911 / 1954 | an `is_open` of a non-door with a state; an `alive` that moves on the host only (`_ALIVE_MOVES = ("cover",)`); a sensor / node evaluator with no device code |
| `SignalRangeRefused` | uk:2293 | a filter node's ema / input past 2^60 or k outside [0, 62] |
| `HordeCacheOverflow` | uk:2389 | a zombie needs a field past `HORDE_FIELD_CAP = 16` |
| `HeatChainRefused` | uk:2605 | the heat chain's bound past the blast log's capacity (64) |
| `UnitEventOverflow` | uk:612, 266, 2385 | ring past `E_MAX = 1024` per env per tick, or a row's staging |
| `UnitRequestOverflow` | uk:437 | request list past `R_MAX = 256` |
| `ObjectTableOverflow` | object_tables.py:224 / 425 / 472, uk:1041 | grenades 64, charges 32, rounds 128, noise 64, blast log 64, sources 32 / 2048 tiles, doors 128 / 1024 tiles, ducts 16, vents 64, signals 256, sensors 32, nodes 64, pumps 16, timed charges 32 (unit_schema.py:103-1875) |
| `ListOverflow` | attack_lists.py:146 / 167 | cell buffer 4096, claims 8192 |
| `DoorOverflow` / `DoorInterleave` | edit_door.py:307 / 105, 114, 177-244 | edit door rows past 8192; a host + device interleave in one walk; a row the door does not write (until step (c)) |
| `*TableMismatch` | uk:1447, 1633, 1654, 2099-2153, 2478, 2699-2717 | the landing's rows are not the sim's runtimes (gate-side) |
| resident batch `ValueError` | pr:2079-2091, resident_batch.py:74-85 | N > 1 with an ambient map, with light, a different grid / gas table / tilt / tick rate |
| edit-pack fallback (not a refusal: a host flush) | field_edit.py:947 / 960 / 980 / 996 / 999 | a `water_depth` edit; a gas slice outside [0, G); an integer past the device's range; noisy edits without `MatchDraws` -- INSIDE A DEVICE TICK THIS BECOMES A REFUSAL, there is no host to fall back to |

Accepted gaps in the rule files P7e's replay gate could trip over:
- The beam has no device form (device-units.md, The march): any level / loadout with `lance_3` / `lance_5`.
- Process-wide weapon tables vs each sim's own (two places, device-units.md): a batch of sims built one
  after another must share one tile size, or the device and the host read different rows.
- Physical cover and the facing defense not ported (WEGO); grenade / charge / round / source tables, the
  horde cache and the noise list are in NO digest: a replay gate that compares digests every tick does NOT
  see a divergence in them until it reaches a digested plane or unit (could be many ticks later).
- The sky bracket (ambient maps, N = 1 only); "the books come down every tick while host gameplay is in
  the loop" (device-residency.md): a device tick changes that.
- A fault met after the first write stops a payload effect where it is (device-units.md, Payload fn);
  `transfer_gas_n` non-atomic in the reference vs atomic on the device (topology seam): equal only while
  no fault / refusal happens.
- A heat source's wired `alive` is the constant the reference emits (a finding, not changed).
- The stamp's refill (`_neighbor_mean`, atmosphere) "is still host Python in both stamp paths"
  (device-residency.md, Stamp kernel, DEBT): a host step inside slot 6 not listed in the P7e row.
- The status OVERFLOW (`status.OVERFLOW_REFUSED`, tracked by `_pinned_matches.PinnedPlay.overflow_refused`)
  -- I did not check whether the device status kernel refuses or counts it the same way.

Which pinned match / level would hit a refusal, from reading:
- Both pinned matches are ship_demo (`levels/ship_demo`: boundary "space", so not ambient; 60 door
  entities, 1 duct, 2 vents, no bus, no sensor / node / pump, no timed charge, no heat source, 4 zombies;
  `match.toml`: two teams of 4 `rifleman` = `k5_carbine` (archetype `projectile`, 1 shot per trigger) +
  2 frags, a smoke, a flash, one `demo_c4`, 3 magazines, `rifle_butt` melee). `bots`: 2721 ticks;
  `bots_v1`: 264 ticks.
- NOT hit by construction: `ShotRefused` (no hitscan weapon), `SpraySlotRefused` (no spray),
  `ObjectSlotRefused` (continuous ruleset, integer fuses), the signal / hazard refusals (no bus, no
  charges, no sources), the payload-row refusals (empty table; the frag is a strength row served since
  P7c.2), the batch's ambient / light refusals (space map, headless).
- CANNOT be told by reading, need a run: `LedgerRangeRefused` past tick 500 (the existing ship_demo legs
  run 500 ticks; `bots` runs 2721), `HordeCacheOverflow` (noise fields per drop), `BulkGasOverflow`
  (a frag in a small sealed room), `EffectOverflow` / `UnitEventOverflow` in a busy tick, the edit pack's
  product bound on device-built rows, the flash / smoke grenades' PAYLOAD rows if a bot throws them (the
  design text says no bot throws smoke; I did not verify flash). To know: play each pinned match on the
  host under a `ShadowLockstep` of every ported slot for its full length (or run the P7e conductor in
  shadow) and count refusals and overflows per kind.
- No recorded match file is checked in (`git ls-files` finds no recording): the gate must record its own
  through `match_replay` (FORMAT_VERSION 8, checkpoints every `[replay] checkpoint_every_ticks`, 60 in
  the format docstring) -- and "digests equal at every tick" needs a host re-simulation beside it, since a
  recording holds only periodic checkpoints.

---------------------------------------------------------------------------------------------

## G. The swarm (#63)

In the reference tick:
- Load only: `swarm_brood.spawn_broods(sim)` once in `_reset_internal` (simulation.py:449; the facade
  `spawn` sets `swarm_host_dirty`, swarm.py:1107). `reclaim` (swarm.py:1149) has NO caller in `src/`.
  So in a match nothing on the host writes the store during a tick.
- Slot 7: the swarm step is the resident tail's LAST stage (after temperature, before the books D2H,
  `step_tail_resident(swarm=...)` pr:2193-2199; host path `PhysicsEngine.step_swarm` pr:1277). It reads
  map planes (temperature, solid, the slot-6 occupancy stamp) and writes the store, the bite-dose and
  count planes, and the death rows.
- Slot 9 head: `tick_events.extend(swarm_death_events(rows))` (sim:2273): host events, NOT in the
  synced event digest (swarm.md, Swarm death events).
- Slot 9c3: `apply_bite_dose`, `apply_corpse_infestation` (sim:2349-2356) read the bite / count planes;
  device form = the fused `gas_rows` kernel (`GAS_FIELDS` uk:310 takes `swarm_bite_dose`,
  `swarm_count`).
- Slot 10: the corpse rise in `convert_marines_to_zombies`; device form = `convert_zombies`.
- Digest: the `__swarm__` carrier from the host mirror (`swarm_carrier` swarm.py:825,
  `swarm_section_bytes` :856), folded in `tick_digest` (field_digest.py:242-249).

On the device and what a conductor must do:
- The store is `GameMap._RESIDENT_SYNCED` (`SWARM_RESIDENT_NAMES`, gamemap.py:280), stacked (N, max_units)
  by `ResidentBatch`; a batch-of-swarms gate exists (`tests/cuda_swarm_batch_check.py`,
  `cuda_batch_invariance_check.py` LEG B).
- Host steps the swarm still has per tick, all in `swarm.engine_args` (swarm.py:1219): (1) the launch
  extent is the HOST `high_water` (host-authoritative, D16; never read from the device), packed per env;
  (2) the env words (`_env_words` :1206: `draws.tick` and the seed key) per tick; (3) `rule_violations`
  checked on the host at every launch (:1252); (4) the species tables packed (`swarm_tables.values_batch`)
  every launch; (5) the dirty-store upload at entry (pr:1590-1598); (6) the death rows come down at the
  books D2H and become host events; (7) the exit download of every present species' columns + both
  planes (`resident_exit_names` :1321) because the digest, the recorder and host slot 9c3 read the mirror.
- For a device tick: (1) stays valid as a constant latch while nothing spawns in a tick (true today);
  (2) is the same tick / key latch every unit kernel needs; (3) / (4) are per-config constants that could
  be packed once per config; (5) happens only after a host write (load / reset); (6) may stay host-side
  (render + facts, not digested) or ride the ring; (7) goes away: 9c3's `gas_rows` reads the device planes.
- I found NO #63 device code that assumes a host step between two device slots inside the tick: the
  swarm sees units only through the slot-6 stamp planes and units see the swarm only through the two
  planes (the boundary rule), all resident. No applier effect kind exists for the swarm (`KIND_FIELDS`
  applier.py:97-107: no swarm kind), so no ordered swarm law yet.

---------------------------------------------------------------------------------------------

## H. A proposed cut (PROPOSAL, from A-G only)

Each step gated at tol 0 against the reference, N = 1 and in a batch where it applies.

1. **P7e.1 planes and latches resident** (needs: nothing). Delete the three dead `rad_*` planes from
   `CLEAR_PLANES` and their clears; make `sight_screen`, `ignition_armed`, `div_source` (and the write-only
   `heat_atten` / `light_atten`, or drop them from `PLANES`) device buffers of the batch; a device flight
   plane (or the walk reading `material` + `blocks_thrown`); a per-env device TICK ROW (tick, draw key,
   `_tile_serial` as the `tiles_changed` count) read by every kernel instead of `ticks` / `keys` / the
   latch serial; 9c / 9c3 fed from the device claims list (not `claims_of`). Gate: every existing
   shadow and chained check unchanged with inputs from the resident buffers; `cuda_s8a_check`,
   batch invariance (Class P: the resident set changes).
2. **P7e.2 the device sorts and 9b** (needs 1). Burn-through list sorted row-major on the device and fed
   to TOPOLOGY; a 9b device form (the scan, the top-16 by differential with the host's tie order, then
   DESTROY_WALL records); stamp rows built from the unit store in the host C++ stamp's row order.
   Gate: shadow of 6 / 9 / 9b on fire, breach and walking scenes.
3. **P7e.3 the edit door step (c)** (needs 1). The device per-env stable sort, the buffer layout K1 / K2
   read, the device range checks of `pack_for_device` as faults, the window persisting across ticks,
   `begin()` at 6b. Gate: K1 / K2 fed from device-built rows == fed from the host pack, every tick, on the
   spray, payload, hazard and heat-source scenes; books equal; the noise key absolute.
4. **P7e.4 the conductor at N = 1** (needs 1-3). The unit store, tables, lists, door and cache resident
   ACROSS ticks; one function walking A's order; a latch writer for the actions (a column scatter into the
   store, from the replay controller's intents) and the order table; every face's counts deferred to the
   tick's books read (one sync), raised there by name; the gate's download for the digest. Gate:
   recorded ship_demo matches (`bots`, `bots_v1`) + the hand scenes (payloads, signals, hazards, sprays,
   horde with bashing) replayed, `(field_digest, tick_digest)` equal every tick to a host re-simulation.
5. **P7e.5 the batch and the profile** (needs 4). N envs, profiler labels per sub-slot. Gate: env i of N ==
   env i alone, digests per tick; the capture check (two named syncs).

Three biggest risks:
1. THE ACTIONS LATCH. No device latch writer exists; today every intent writes a host `Unit` and the
   device sees it only through a whole-row pack. The replay's intents, their normalisation at the tap, and
   anything that reads unit state while issuing them must be shown to touch only latch columns, or the
   device and the host diverge at tick 1 of the gate.
2. HOST-ONLY SIDE EFFECTS THE REFERENCE READS LATER: `_tile_serial` (the horde latch, the stamp's
   validity, the static-plane upload), the solid-books sink, the swarm's host `high_water`, the claims
   dedupe, the books folds. A device tick that drops one reads stale state silently; several are in no
   digest (tables, cache, noise list), so the per-tick digest catches a divergence late.
3. RANGES AND REFUSALS OVER LONG MATCHES. The device gates ran ship_demo for 500 ticks; `bots` is 2721
   ticks. Vent ledgers, horde cache fields, bulk gas, list capacities and the edit pack's product bound are
   unproven there; and once counts are deferred to one sync, a refusal mid-tick means the later slots ran
   on a partially written state before the run is refused (the gate must refuse the tick, never compare it).

---------------------------------------------------------------------------------------------

## I. What I could not determine by reading

- The exact tuple `applier.non_resident_planes()` returns (derived by reading the two lists; not run).
- Whether any bot throws the flash grenade in the pinned matches, and whether the flash / smoke PAYLOAD rows
  pass the device tables' ranges at ship_demo's tile size in a 2721-tick play.
- Whether `LedgerRangeRefused`, `HordeCacheOverflow`, `BulkGasOverflow` or any capacity is hit in the
  `bots` match past tick 500.
- Whether the device edit door enforces `pack_for_device`'s product (2^125) / contribution (2^55) bounds
  (I found only the radius and weight ranges in `cuda_edit_door.cuh`).
- Whether the intents' normalisation at the replay tap reads host unit state (risk 1); I did not read
  `src/match_replay.py`'s tap or `Simulation`'s intent methods in full.
- Whether the device status kernel treats `status.OVERFLOW_REFUSED` as the host does.
- Whether `ContinuousRealtime.on_round_start` does any per-tick work (I read only `on_tick_end`).
- The stamp refill debt's exact host call site inside slot 6 (named in device-residency.md; not located).
- Whether `heat_atten` / `light_atten` float planes are read by a renderer path outside the directories
  I grepped (`src/`, `renderer/`, `ui/`, `tools/`, `main.py`).
- The current cost of the per-launch count downloads (H20); measured nowhere I found.
