# P6 slot inventory: the tick as it is on main (2026-11-14)

Read-only inventory for re-planning residency P6 (#29). Source: the main
checkout at `43115fec`. Every statement cites `file:line` on that commit;
paths are relative to the repo root. Where something could not be settled
from the code it says "not determined" and why. No plan, no judgement of the
design: Part A item 9 only says where the residency design
(`docs/residency_batching_design_2026-11-07.md`, "the design") is stale or
silent.

Conventions used below:

- "S" = a `unit_schema.COLUMNS` column (group S, synced, `src/simulation/unit_schema.py:452-560`, 75 columns); "D" = `DERIVED` / `OPTICS_OVERRIDES` (`unit_schema.py:611-624`); "R" = `REFERENCE_ONLY` (`unit_schema.py:631-665`). A column name is given as the schema name (`current_hp` is held as the Q16 int `hp_q`, `unit_schema.py:488-491`; `x_q` / `y_q` back the float views `x` / `y`; `life_state` backs `alive`).
- "statuses" read through `status.composed_flags(u)` means the column `statuses` is read.
- `tile_x` / `tile_y` / `center_tile_*()` / `occupied_tiles()` are views over `x_q`, `y_q`, `footprint`.
- `sim.units` is in id order: ids are assigned monotonically at the only append (`simulation.py:580-588`).
- Synced events = `UnitHitEvent`, `UnitKilledEvent` only (`tests/field_ab_harness.py:219`); every other event type is render-only.
- `apply_packet` (`src/simulation/damage.py:363-469`) is reached from many slots. What it touches, once, here: reads `mitigation` / `environment` (D), `exposure`, `id`, `life_state`; writes `current_hp`, and on the alive -> dead transition `life_state`, the 11 impulse columns (`unit_impulse.clear`, `damage.py:450-452`), `zombie_noise_y/x/ticks`, `zombie_anchor_y/x` (`horde.clear`, `damage.py:454-456`, `horde.py:547-554`), the four `order_*` columns (`damage.py:458-460`), `killed_by_zombie` when asked (`damage.py:461-462`); `exposure` raise or zero (`damage.py:447`, `467-468`; deferred to `ExposureLanding` inside an attack slot); R `recent_attackers` (`damage.py:434-438`). Appends `UnitHitEvent` and, on a kill, `UnitKilledEvent` (`damage.py:439-466`). Below this is written "apply_packet (the set above)".

---

## Part A: the tick today, slot by slot

`Simulation.step` (`src/simulation/simulation.py:1983-2009`) is three halves:
`_step_pre_physics` (slots 1 .. 6b, `simulation.py:2014-2160`),
`_step_physics` (slot 7, `simulation.py:2162-2176`), `_step_post_physics`
(slots 9 .. 11, `simulation.py:2178-2447`). `ResidentBatch.step` runs the
same halves per env around one batched slot 7
(`src/simulation/resident_batch.py:117-138`). The match loop is
`MatchSession.tick`: every controller's `handle_frame(sim, None)`, then
`sim.step()` (`src/simulation/match_session.py:97-105`); it does not use
`ResidentBatch`.

The docstring of `step` (`simulation.py:1991-2005`) still lists the
pre-gameplay-pack order (it has an "8. Door explosives" and no 1a / 1b / 1b'
/ 4a / 9c2 / 9c3 / 9e / 11); the code below is the order that runs.

### Slot 1: fresh event buffer + round head

1. `simulation.py:2016-2031`: `paused` check (2016), `tick_events.clear()` (2020), `gmap.reset_trace_books()` (2025), `ruleset.on_round_start(self)` (2031).
2. Clears the per-tick event list and the Python trace books. Under `ContinuousRealtime` the round head is a no-op (`src/simulation/ruleset.py:270-273`).
3. Unit columns: none.
4. Writes `sim.tick_events` (cleared), `gmap` trace books (reset).
5. Order: no per-unit work.
6. No events, draws or noise.
7. TwoPhaseWEGO's `on_round_start` does tick-0 work (`ruleset.py:141`); not on the continuous path.
8. Pure Python.
9. Design 4.3 row "1 events": "the ring's tick head". Consistent; the trace-books reset is not mentioned.

### Slot 1a: integrate + turn (`unit_impulse.integrate_all`)

1. `simulation.py:2044` -> `unit_impulse.integrate_all` (`src/simulation/unit_impulse.py:732-745`) -> `integrate` (`unit_impulse.py:618-729`) then `turning.step` (`src/simulation/turning.py:227-237`).
2. For every living unit: the impulse accumulators become dv / dw (remainders kept), the effort low-pass, the in-place sway and its overflow into the pushed velocity, one tick of hold resisting, then the knockdown checks (`apply_status(KNOCKED_DOWN)`, `unit_impulse.py:612-615`). Then the live aim (if held) becomes the facing target and the turning law advances `facing_q` / `omega_q` with the outside spin dw. A unit "at rest" (`unit_impulse.py:436-441`) skips the linear part (`unit_impulse.py:624-625`) but still turns.
3. Reads: `life_state`, `mass_g`, `imp_x`, `imp_y`, `imp_l`, `stand_x`, `stand_y`, `swx`, `swy`, `pvx`, `pvy`, `effort`, `react_ticks`, `statuses` (`is_prone`, `unit_impulse.py:660`), `is_zombie` + D `environment` (stability, `src/simulation/exchange.py:554-570`), `facing_q`, `facing_target_q`, `omega_q`, `live_aim` (`turning.py:233`). Writes: the 11 impulse columns, `statuses` (knockdown), `facing_q`, `facing_target_q`, `omega_q`; R `impulse_books`. 20 read / 15 written.
4. Other state: `sim.impulse_table` (map-quantized constants, `unit_impulse.py:191-232`); the turning dials `sim.unit_turn_accel_q16` / `unit_turn_rate_q16` (one per sim, `simulation.py:1060-1072`); the status registry.
5. ORDER-FREE per unit: "Each unit reads only its own state, so the order changes nothing" (`unit_impulse.py:736-738`); confirmed: `integrate` and `turning.step` read and write only `u` and constants.
6. `StaggerEvent` (render-only, `unit_impulse.py:692-696`), appended in id order. No draw, no noise.
7. Runs under every ruleset (before the `drives_units` branch).
8. Pure Python. C twins exist for its integer helpers only: `fixed_point.h` `div_trunc_q` / `shr_trunc_q` / `effort_step_q` (`cpp/src/fixed_point.h:801-825`), held equal by `tests/test_unit_impulse_twins.py:42-70`; `turning.law_q16` has no C twin found.
9. Design 4.3: SILENT (slot 1a did not exist on 2026-11-07). Unit design 8 puts it in P6.2.

### Slot 1b: direct-control intents (`_consume_direct_intents`)

1. `simulation.py:2051` -> `_consume_direct_intents` (`simulation.py:1319-1403`), calling `_start_throw` (1405-1431), `_advance_throw` (1433-1452), `_release_grenade` (1495-1541), `_start_plant` (1454-1470), `_advance_plant` (1472-1493), `_direct_use` (1543-1550), `_reload` (1301-1317), `unit_impulse.shove` (`unit_impulse.py:896-919`); gates `verb_gates.can_throw` / `can_plant` / `use_door` / `can_reload` / `can_shove` (`src/simulation/verb_gates.py:61-154`).
2. Returns at once when `_possessed_ids` is empty (1338). Otherwise, per unit in list order: a unit not possessed, dead or a zombie has a held trigger, a wind-up and a plant dropped (1341-1354); a possessed living marine consumes its latches in a fixed order: throw (start, then advance; at release a `Projectile` is appended), plant (start, then advance; at the last tick a `PlantedCharge` is placed through `charges.place`), use (toggles `door.want_open`), reload, shove (adds an impulse to the adjacent unit nearest the facing). Trigger, aim and move are not consumed here (1397-1403). A unit becomes possessed by any intent call except `melee_intent`, `set_flashlight`, `clear_move_dir`, `order_intent` (`simulation.py:1033, 1058, 1102, 1142, 1179, 1190, 1208, 1229`; `melee_intent` 1231-1254 has no `add`).
3. Own row reads: `id`, `life_state`, `is_zombie`, `live_trigger`, `pending_throw`, `throw_release_tick`, `throw_kind`, `throw_fuse_ticks`, `throw_reach_q`, `pending_plant`, `plant_done_tick`, `plant_fuse_ticks`, `pending_use`, `pending_reload`, `pending_shove`, `statuses`, `grenade_counts`, `has_explosive`, `charge_ammo`, `facing_q`, `x_q`, `y_q`, `footprint`, `weapon_id`, `current_mag`, `spare_mags`, `reload_done_tick`, `magazines`, `magazines_weapon` (via `mag_settle` -> `mag_bind`, `src/simulation/combat.py:835-863`), `last_shove_tick`. Own row writes: `live_trigger`, `pending_throw`, `throw_release_tick`, `throw_kind`, `throw_fuse_ticks`, `throw_reach_q`, `pending_plant`, `plant_done_tick`, `plant_fuse_ticks`, `pending_use`, `pending_reload`, `pending_shove`, `grenade_counts`, `has_explosive`, `current_mag`, `spare_mags`, `reload_done_tick`, `last_shove_tick`. OTHER units (shove target, `unit_impulse.py:858-885`): reads `life_state`, `x_q`, `y_q`, `footprint`, `id`; writes `imp_x`, `imp_y`, `imp_l` (+ R `impulse_books`) through `unit_impulse.add` (`unit_impulse.py:488-503`). About 30 read / 18 written.
4. Other state: `sim._possessed_ids` (set); `sim.projectiles` (append, 1540); `sim.planted_charges` (append through `charges.place`, `src/simulation/charges.py:63-75`, capacity `CHARGE_CAPACITY` 32, refusal counted in `sim.charges_refused`); `sim._doors` (`want_open` toggled, 1550; door scan `verb_gates.py:110-124`); `sim.weapons_tables` (hand grenade row, `PLANT_WEAPON` row); `sim.impulse_table`; `sim.tick`.
6. Events (all render-only): `ThrowEvent` (1420), `PlantEvent` (1467), `ReloadEvent` (`combat.py:913`), `ShoveEvent` (`unit_impulse.py:918`). No draw, no noise.
5. ORDER: per-unit decisions read only own state, other units' positions / facing / `alive` (unchanged during 1b) and constants. Shared writes: (a) shove impulses into other units' accumulators: integer adds, commutative (`unit_impulse.py:494-495`); (b) door `want_open` toggles: two toggles of one door commute; (c) `sim.projectiles` append order = unit id order, and slot 2 detonates grenades in that list order (`simulation.py:2457`), so the APPEND ORDER is load-bearing; (d) `charges.place` counts live charges first (`charges.py:70-73`): which of two plants on one tick is refused at capacity depends on id order; (e) event emission order is id order.
7. Dormant under TwoPhaseWEGO (no possessed units, 1323-1325). Runs under OnePhaseWEGO too (before the `drives_units` branch).
8. Pure Python.
9. Design 4.3 row "1b intents": "the action buffer consumed in id order (throws get a spawn ordinal)". Stale: a `Projectile` carries no ordinal (`combat.py:134-167`); its order is its place in `sim.projectiles`. The table is silent on plant, reload, shove, the wind-up / plant advance, and the possession gate. The design's 4.4 intent columns (`set_trigger` HOLDS, `simulation.py:1092-1102`) match.

### Slot 1b': the commander's orders (`unit_orders.apply_requests`)

1. `simulation.py:2057` -> `src/simulation/unit_orders.py:173-194`.
2. Writes this tick's latched `order_intent` requests onto their units (living non-zombies only), then clears every request.
3. Reads `life_state`, `is_zombie`; writes `order_what`, `order_where`, `order_until`, `order_target`. 2 / 4.
4. `sim._order_requests` (dict uid -> tuple; written by `order_intent`, `simulation.py:1299`); `sim.get_unit` (linear scan, `simulation.py:705-710`).
5. ORDER-FREE: one request per unit, applied in sorted uid order (`unit_orders.py:182`).
6. None.
7. Every ruleset. Dormant on an empty dict (`unit_orders.py:180-181`).
8. Pure Python.
9. Design 4.3: SILENT (1f added it on 2026-11-12).

### Slot 1c: exposure decays (`exposure.decay_units`)

1. `simulation.py:2062` -> `src/simulation/exposure.py:223-231`.
2. Every unit's non-zero exposure block decays one tick (`decay_step`).
3. Reads / writes `exposure`. Note: it does NOT check `life_state` (a corpse's block is zeroed at death, `damage.py:447`). 1 / 1.
4. `exposure.get_settings().decay_q` (config).
5. ORDER-FREE per unit.
6. None.
7. Every ruleset.
8. Pure Python.
9. Design 4.3: SILENT (G4p).

### Slot 2: projectiles (grenades, then rounds)

1. `simulation.py:2065` -> `_update_projectiles` (`simulation.py:2452-2501`); `Projectile.update_position` (`combat.py:194-215`); `execute_payload` (`src/simulation/payloads.py:396-558`); `BulletInFlight.advance` (`combat.py:381-626`); a slot-local `SimultaneousEffects` (`simulation.py:2493`, `src/simulation/simultaneous.py:151-182`).
2. Grenades, in `sim.projectiles` order: each not-detonated projectile moves (integers); at its fuse tick it detonates IMMEDIATELY through `execute_payload` (wall damage, blast damage to units, flash blinding, field edits enqueued, a bulk-gas injection applied at once, the blast log, a noise, the expansion source). Detonated projectiles are flagged and stay in the list (see Part E). Then in-flight rounds, in `sim.bullets` order: each marches its integer budget against the state left by the grenades, RECORDING its hit, push, chew and payload on the slot's effects; `landing.resolve()` applies them in list order; survivors replace the list.
3. Grenades (via `execute_payload` -> `exchange.apply_blast_damage`, `exchange.py:494-544`, and `flash_blind.apply_flash_blind`, `payloads.py:482-483`): read every unit's `life_state`, `x_q`, `y_q`, `footprint`, and for the flash `facing_q`; write apply_packet (the set above) and `statuses` (BLINDED). Rounds: read every unit's `life_state`, `x_q`, `y_q`, `footprint` (march hit test, `combat.py:479-485`), `is_zombie` (bullet multiplier, `combat.py:527-535`), `facing_q` (crit arc, only if `crit_chance > 0`, `combat.py:541-546`); recorded writes on the hit unit: apply_packet (the set above) and `imp_x`, `imp_y`, `imp_l` (push, `combat.py:361-379`).
4. Map: `material`, `solid`, `wall_hp` (+ topology through `destroy_wall` from `apply_explosion` and `chew_wall`, `combat.py:237-258`), `gmap.blast_log` (append, `payloads.py:464-470`), `gmap.noise_log` (`payloads.py:474`), `gmap.damage_credit` (claims, `payloads.py:527`, first claim wins: `damage.py:254-255` `setdefault`), `div_source` / expansion (`payloads.py:495-502`), `gas` (bulk injection, immediate, `payloads.py:524-525`), the edit queue (enqueued deposits). Host: `sim.projectiles` (each `Projectile` mutated), `sim.bullets` (replaced), `sim.shots` (tracer list, append, `combat.py:581`), the weapons / payload tables, `CFG.clock.ticks_per_second` (read inside `update_position`, `combat.py:202`), `CFG.combat.blast_damage_threshold`.
5. ORDER-DEPENDENT. Grenades are serial and IMMEDIATE: each detonation changes walls, unit hp / life and fields that the next grenade and every round then read; order = list (append) order. Rounds: phase A per round reads the post-grenade state and its own state only (draws keyed per round, below), so phase A is order-free; phase B applies hits / chews / payloads in list order (a wall chewed to zero by one round is absent for the next round's `chew_wall`, `combat.py:249-255`; credit claims first-wins).
6. Events: synced `UnitHitEvent` / `UnitKilledEvent` (blast, bullet); render-only `ShotFiredEvent`, `ProjectileGlowEvent` (`combat.py:584-597`), `ExplosionEvent` (`payloads.py:548-558`). Draws: `Salt.EXPOSURE` (`combat.py:506`) and `Salt.SHOT_CRIT` (`combat.py:544`), each `rng.at(salt, shooter_id, round_index(age, ordinal))` (`combat.py:338-345`), lazy (only on cover / crit). Grenade detonations draw nothing (`payloads.py:439-441`). Noise: one per detonation.
7. Every ruleset. TwoPhaseWEGO prunes detonated projectiles at round end (`simulation.py:2701`), OnePhaseWEGO in its on-tick-end (`ruleset.py:477`); the continuous path never prunes them.
8. Pure Python (the blast's C++ parts are slot 7's: `div_source` is consumed there).
9. Design 4.3 rows "2 grenades" (applier, serial, before the rounds; P7b / P7c) and "2 rounds" (wide phase A -> applier in record order; P7b): consistent with the code. Silent on: the bulk-gas injection being an immediate field write; the flash blinding; the round's push; detonated projectiles never leaving the list.

### Slot 2b: statuses tick (`tick_statuses`)

1. `simulation.py:2081` -> `src/simulation/status.py:324-363`.
2. Per living unit in explicitly sorted id order: sweep expired statuses, then each status emits its damage-over-time / heal packet (if its row has a dtype and a non-zero magnitude) and counts down.
3. Reads `id`, `life_state`, `statuses`; writes `statuses`; plus apply_packet (the set above) on the SAME unit only. About 4 / 12 (with the death clears).
4. `STATUS_REGISTRY` (config-built table).
5. ORDER-FREE per unit (each unit's packets land on itself). Events are appended in id order (the sort, `status.py:342`).
6. Synced `UnitHitEvent` / `UnitKilledEvent` (DoT kills never convert, `status.py:336-337`). No draw, no noise.
7. Every ruleset.
8. Pure Python.
9. Design 4.3 row "2b statuses": "one thread per unit row; the ring appended in id order" (P6). Consistent.

### Slot 3: movement (the displacement loop)

1. `simulation.py:2095-2102`: under `ruleset.drives_units` (OnePhaseWEGO) `_displace_pushed` (2582-2590) then `ruleset.drive_units`; otherwise `_update_player_movement` (`simulation.py:2503-2580`) -> `_step_move_dir` (2592-2652) -> `unit_impulse.move` (`unit_impulse.py:778-837`).
2. Per living unit in id order: a non-zombie with a live move direction walks (terrain-, effort- and direction-scaled speed from the Speed table, throttle) plus its pushed velocity as one displacement on the Q16 grid, with the slide rule against `is_passable_block`; a unit holding no aim makes its travel its facing target. Any other unit with a pushed velocity moves by it alone. Then (non-zombies) the legacy WEGO path step, which is a no-op on the continuous path (`move_path` always empty, `unit_schema.py:648-653`).
3. Reads: `life_state`, `is_zombie`, `live_move_dir`, `live_aim`, `statuses` (`can_move`), `react_ticks`, `effort`, `x_q`, `y_q`, `footprint`, `facing_q`, `pvx`, `pvy`; R `move_path`, `path_tick_offset`. Writes: `x_q`, `y_q`, `pvx`, `pvy` (zeroed into a wall), `facing_target_q`; R `impulse_books`, `path_tick_offset` (bumped for a downed unit, 2554-2557). 13 / 5.
4. Map: `material` + the materials table's `mobility` (via `is_passable_block`, `gamemap.py:2353`, and `footprint_mobility`, `src/simulation/movement.py:79-100`). `sim.speed_table`, `sim.impulse_table`.
5. ORDER-FREE: "Each move reads only the map and the unit's own state, so the loop is order-free (units pass through each other, #142)" (`simulation.py:2518-2519`); confirmed: no read of another unit.
6. None.
7. OnePhaseWEGO replaces it (2095-2099); TwoPhaseWEGO uses the `move_path` branch.
8. Pure Python.
9. Design 4.3 row "3 movement": "id order; walls only; one thread per unit row" (P6). Consistent, but silent on the pushed velocity (G5p-a2) and the direction rule (G1p-c) that the slot now includes.

### Slot 4: shooting, phase A (`process_shooting`)

1. `simulation.py:2087` (the slot's `SimultaneousEffects`), `simulation.py:2109-2112` -> `combat.process_shooting` (`combat.py:958-1126`) -> `_directional_fire` (`combat.py:1160-1215`) / `auto_fire` (1218-1273) / `_dispatch_trigger` (1129-1157) -> `fire_burst` (1291-1375) / `fire_beam` (1389ff) / `start_spray_burst[_directional]` (1752-1793) / `melee_strike` (1946-1998).
2. Per living non-zombie with a weapon row and `can_act`, in id order: on the continuous path a held `live_trigger` fires directionally along `-facing_q` through the cadence and magazine gates (the `orders` branches are dead there: `orders` always empty, `unit_schema.py:632-634`). Each pull emits a noise, draws one cone offset per bullet, constructs `BulletInFlight`s and marches them at once; a hit, push, recoil, chew or payload is RECORDED, not applied; a round still flying is appended to `sim.bullets`. The shooter's own magazine, cadence and spray state are written at once.
3. Shooter reads: `id`, `is_zombie`, `life_state`, `statuses` (`can_act`, `can_aim`), `weapon_id`, `live_trigger`, `last_fire_tick`, `current_mag`, `spare_mags`, `reload_done_tick`, `magazines`, `magazines_weapon`, `facing_q`, `x_q`, `y_q`, `footprint`, `spray_ticks_left`, `team` (auto-fire only); R `orders`. Shooter immediate writes: `last_fire_tick`, `current_mag`, `spare_mags`, `reload_done_tick`, `spray_ticks_left`, `spray_target`, `spray_aim_angle`; R `spray_order`. Other units (march): `life_state`, `x_q`, `y_q`, `footprint`, `is_zombie`, `facing_q` (crit). Recorded writes: on hit units apply_packet (the set above), `imp_x/y/l` (push); on the shooter `imp_x/y` (recoil, `combat.py:1276-1288`); `facing_target_q` (face effect, only on the tile-target paths, `combat.py:1338-1345`). About 20 read / 9 immediate + recorded.
4. Map: `material`, `solid`, `wall_hp` (read in the march; chew recorded), `gas` (the beam's attenuation, not traced line by line), `gmap.noise_log` (append, IMMEDIATE in phase A, `combat.py:1145-1146`, `1202-1203`, `1115-1116`); the impulse table cached on the map (`unit_impulse.table_for`). Host: `sim.bullets` (append, `combat.py:1374-1375`), `sim.shots` (append), `sim.draws`, the weapons / ammo tables, the edit queue (recorded payload deposits), `CFG.clock.ticks_per_phase` (`combat.py:998`).
5. Phase A decisions are ORDER-FREE: every decision reads slot-start state (only recorded effects change others), the shooter's immediate writes are to columns no other shooter reads, and the draws are keyed per shooter. Order-bearing shared writes: the record order on the effects (shooter by id, round by round), the append order of `sim.bullets` (next tick's slot-2 order) and of `gmap.noise_log` (`_hear` ties to the earlier noise, `src/simulation/ai_zombie.py:83-96`).
6. Events: synced UnitHit / UnitKilled come at 4c (or slot 2), not here; render-only `ShotFiredEvent`, `LaserFiredEvent`, `ReloadEvent` here. Draws: `Salt.SHOT_CONE` (`combat.py:1356`, `rng.next(salt, shooter_id)`: index local to the shooter this tick), `Salt.BEAM_CONE` (`combat.py:1470`), `Salt.EXPOSURE`, `Salt.SHOT_CRIT` (lazy). Noise per trigger pull.
7. OnePhaseWEGO replaces slots 3 and 4 with its timeline (2095-2099); TwoPhaseWEGO uses the `orders` branches.
8. Pure Python.
9. Design 4.3 row "4 / 4b phase A": "one block per env over its shooters; sprayers after shooters" (P7a). Consistent. Silent on the noise emitted in phase A and on the recoil / push effects.

### Slot 4a: the melee verb (`process_melee_intents`)

1. `simulation.py:2119-2121` -> `combat.process_melee_intents` (`combat.py:2078-2122`) -> `verb_gates.can_melee` (`verb_gates.py:157-181`), `melee_arc_target` (`combat.py:2046-2075`), `melee_blow` (`combat.py:2001-2043`).
2. Every unit's `pending_melee` latch is consumed; a living marine with a melee row, able to act, with no wind-up or plant and its row's cadence run strikes the adjacent enemy nearest its facing within the arc (ties to the lower id). The blow (packet, delivery-site status, push) is RECORDED on the same effects as slot 4.
3. Own row reads: `pending_melee`, `life_state`, `is_zombie`, `melee_id`, `statuses`, `throw_release_tick`, `plant_done_tick`, `last_blow_tick`, `x_q`, `y_q`, `footprint`, `facing_q`, `team`, `id`; writes `pending_melee`, `last_blow_tick` (immediate). Other units: `life_state`, `team`, `x_q`, `y_q`, `footprint`, `id`; recorded writes on the target: apply_packet (the set above), `statuses` (STUNNED), `imp_x/y`. 14 read / 2 immediate + recorded.
4. `sim.weapons_tables`, `sim.impulse_table` (`melee_push_q`).
5. ORDER-FREE decisions (slot-start state: slot 4 wrote only shooter-own columns not read here). Record order: after all of slot 4's records, in id order.
6. Render-only `MeleeEvent` (`combat.py:2119-2122`). No draw: the loadouts door refuses a melee row with a crit (`combat.py:2094-2095`); `melee_blow` draws `Salt.MELEE_CRIT` only on the WEGO order path (`combat.py:2024-2028`). No noise.
7. Every ruleset (2114-2118: the latch is consumed even where slot 4 is the timeline's).
8. Pure Python.
9. Design 4.3: SILENT (G6p).

### Slot 4b: spray deposits (`process_sprays`)

1. `simulation.py:2131-2132` -> `combat.process_sprays` (`combat.py:1796-1888`) -> `deposit_spray_cone` (`combat.py:1673-1723`) -> recorded `_enqueue_spray_cells` (`combat.py:1726-1749`).
2. Per unit with a burst running, in id order: a dead unit's burst is cleared; a unit that cannot act is interrupted (burst cleared, its WEGO order removed); otherwise one tick's cone (tiles occluded by the slot-start walls) is computed and its heat / gas FieldEdits and credit claims are RECORDED; the burst counts down.
3. Reads `spray_ticks_left`, `life_state`, `statuses`, `weapon_id`, `spray_target`, `spray_aim_angle`, `x_q`, `y_q`, `footprint`, `id`; R `spray_order`, `orders`. Writes (immediate, own) `spray_ticks_left`, `spray_target`, `spray_aim_angle`; R `spray_order`, `orders`. 10 / 3.
4. Map: `material` / `solid` (cone occlusion, `spray_cone_tiles`, `combat.py:1588`), `gmap.gases.name_to_id`; recorded: the edit queue, `gmap.damage_credit` claims.
5. ORDER-FREE per unit (own writes; world writes recorded). Record order: after slot 4 and 4a, in id order.
6. Render-only `SprayJetEvent` (`combat.py:1874-1881`). No draw (`combat.py:1824-1826`), no noise (the noise is the trigger's, slot 4).
7. Every ruleset.
8. Pure Python.
9. Design 4.3 groups it with row "4 / 4b phase A". Consistent.

### Slot 4c: phase B (`SimultaneousEffects.resolve`)

1. `simulation.py:2134` -> `simultaneous.py:151-182`.
2. Applies the recorded effects in recording order: `_HIT` -> `apply_packet` with an `ExposureLanding`; `_STATUS` -> `apply_status` if alive; `_FACE` -> `turning.set_target`; `_PUSH` -> `unit_impulse.add` if alive; `_WORLD` -> the recorded call (`chew_wall`, `execute_payload`, `_enqueue_spray_cells`). Then the exposure raises / zeroings land together (`landing.land()`).
3. Writes on targets: apply_packet (the set above), `statuses`, `facing_target_q`, `imp_x`, `imp_y`, `imp_l`; a `_WORLD` `execute_payload` (a payload round) writes apply_packet + `statuses` on every unit in its radius (as slot 2). Reads `life_state` (alive gates).
4. Map through `_WORLD`: `wall_hp`, `material` + topology (`destroy_wall`), the edit queue, `gmap.damage_credit`, `gmap.blast_log`, `gmap.noise_log` (a payload's noise is emitted at 4c, inside `execute_payload`), `div_source`, `gas` (bulk injection).
5. ORDER-DEPENDENT by construction: serial in record order (shooters by id x rounds, then melee by id, then sprayers by id). The first packet to zero is the kill (`simultaneous.py:32-37`); credit claims are first-wins; a later chew finds an earlier-destroyed tile gone.
6. Synced UnitHit / UnitKilled (emitted by `apply_packet` at its turn, so the synced event order is record order); render-only `ExplosionEvent` from a payload. No draw ("fn must draw no RNG", `simultaneous.py:143-144`).
7. Every ruleset (OnePhaseWEGO's timeline uses `IMMEDIATE` inside its own call, `simultaneous.py:45-47`).
8. Pure Python.
9. Design 4.3 row "4c resolve": consistent (lists hit / status / face / world; silent on `_PUSH`, added at G5p-a2, and on the exposure landing).

### Slot 5: zombie AI (`_run_zombie_slot`)

1. `simulation.py:2138` -> `_run_zombie_slot` (`simulation.py:672-698`) -> `ai_zombie.update_zombies_tick` (`src/simulation/ai_zombie.py:170-394`), `_hear` (81-112), `_walk` (137-164), `_bash` (115-134), `horde.HordeState` (`src/simulation/horde.py:362-422`), `horde.descend` (476-498), `horde.flood` (C++ when an engine is bound, `horde.py:264-272`).
2. Drains `gmap.noise_log` every tick (682). In a match only the zombies the controller named through `drive_zombies` run, plus all non-zombies as prey (`simulation.py:689-691`); an empty request set returns after the drain (687-688). Passes: (1) trigger: a dormant zombie with LOS to prey strictly inside the trigger radius wakes; (2) chain activation to a fixed point; waking clears noise / anchor; (2b) hearing; (3) per zombie in list order: awake: bite the nearest prey within `footprint + 1` (IMMEDIATE `apply_packet`, converting kill), else walk the prey field (sprinting with LOS to prey in range); hearing: walk the noise's field, settling on arrival / timeout / no path; dormant: the wander (one hashed step per period within the leash). A blocked step may bash a closed door (`chew_wall` at the bite's cadence, a noise per strike).
3. Zombie reads: `id`, `life_state`, `is_zombie`, `zombie_activated`, `zombie_noise_y/x/ticks`, `zombie_anchor_y/x`, `x_q`, `y_q`, `footprint`, `statuses`, `last_melee_tick`, `speed_ticks_per_tile`, `zombie_move_accumulator`, `react_ticks`, `effort`. Zombie writes: `zombie_activated`, `zombie_noise_y/x/ticks`, `zombie_anchor_y/x`, `last_melee_tick`, `zombie_move_accumulator`, `x_q`, `y_q` (whole tiles, `ai_zombie.py:160`, `394`), `facing_target_q` (`Unit.face_towards`, `src/simulation/unit.py:541-555`). Prey reads: `life_state`, `is_zombie`, `x_q`, `y_q`, `footprint`; prey writes: apply_packet (the set above) with `killed_by_zombie`. About 18 read / 12 written.
4. Map: `gmap.noise_log` (drained; appended by bashes), `material`, `mobility` (via `is_passable_block`), `solid` / `sight_screen` (`has_los`), `temperature` / `fire` (fear of fire, `horde.block_feared`), `wall_hp` + topology (bash -> `chew_wall` -> `destroy_wall`), `gmap._tile_serial` (field cache drop, `horde.py:368-375`). Host: `sim.horde_state` (the field cache: fields per key, masks; not synced), `sim.horde_settings`, `sim.speed_table`, `sim._doors` (bashable doors), `sim._zombie_requests`, `CFG.zombie.attack_cooldown_ticks` / `melee_damage`.
5. ORDER-DEPENDENT in pass 3 (list order). Exactly: (a) a bite is applied at once (`ai_zombie.py:304-312`) while `players` was built once at the slot head (`ai_zombie.py:215`) and the nearest-prey loop does not re-check `alive` (`ai_zombie.py:284-288`): a later zombie can choose and bite a prey an earlier zombie killed this slot (a UnitHit with no second UnitKilled); (b) a bash destroys a door tile mid-slot: later zombies' `is_passable_block` and `bash_tile` read the live `material`, while the field cache (`begin_tick` runs once at the slot head) may hold a mask or field built before the bash, or build one after it on first need (`horde.py:377-399`); (c) the hearing cap (`ai_zombie.py:101-112`) is a global selection (ties by tile), not per zombie. Passes 1 and 2 yield the same activated set in any order (pass 2 runs to a fixed point). Zombie moves do not read other zombies' positions (descent reads fields seeded from prey, and the map).
6. Synced UnitHit / UnitKilled (source `melee`, credited to the biter); render-only `DoorDestroyedEvent` (`ai_zombie.py:132-133`). Draws: none from `sim.draws`; the wander direction is a Philox block used as a seedless hash with key (0, 0) and `Salt.WANDER` (`horde.py:510-517`), and the stagger is arithmetic (`horde.py:504-507`). Noise: one per bash (`ai_zombie.py:134`).
7. One driver per tick: the conductor (game paths) or the zombie controller's requests (a match; `simulation.py:672-698`). Every ruleset calls the slot.
8. Pure Python law; the BFS flood is host C++ `cpp/src/horde_field.cpp` (`PhysicsEngine.horde_field`) with a Python reference `horde.flood_reference` (`horde.py:237`); no device kernel.
9. Design 4.3 row "5 zombie AI": "passes 1-2 order-free; pass 3 in id order; ties to the lower id; bites immediate but decisions read stale prey" -> device "distance field + one thread per zombie; bites recorded, applied by the applier in id order" (P7d). The reference facts match the row; it is silent on the field cache's refresh-and-bash interplay, hearing, wander, fear of fire, the sprint and the match's request filter. 4.9 records P3d as closed.

### Slot 6: re-stamp (`GameMap.stamp_units`)

1. `simulation.py:2141` -> `gamemap.py:2071-2090` -> `_stamp_units_cpp` (`gamemap.py:2126-2211`, C++ oracle `PhysicsEngine::stamp_units`) or `_stamp_units_python` (`gamemap.py:2213-2335`).
2. Rebuilds `obstacles` (walls only), `dyn_permeability`, `dyn_wave_absorb`, `dyn_heat_atten_q`, `dyn_light_atten_q` from the static baselines and every living unit's footprint (MAX stamps; the permeability `min(unit, static)` written per tile, so the last unit wins on overlap). The C++ path caches the flattened rows for the device stamp (`gamemap.py:2200-2201`). Then the atmosphere refill: each wall tile freed since the previous stamp gets the neighbour mean of `atmosphere` (`gamemap.py:2203-2211`, `2330-2335`).
3. Reads `life_state`, `x_q`, `y_q`, `footprint`; D optics overrides `permeability`, `wave_absorb`, `heat_atten`, `light_atten` by `getattr` (no game path sets them, `unit_schema.py:621-624`). Writes no unit column. 4 / 0.
4. Map writes: the five planes, `atmosphere` (refill), `gmap._stamp_pending`. Reads: `permeability`, `wave_absorb`, `heat_atten_q`, `light_atten_q`, `is_vacuum`, `_tile_serial`; `CFG.physics.unit_permeability` / `unit_wave_absorb`.
5. Per unit: MAX is commutative; the permeability write is last-wins (identical results unless per-unit overrides differ). The refill loop is row-major and each freed tile reads neighbours' `atmosphere` while earlier freed tiles have just been written (`gamemap.py:2208-2211`): order-dependent between adjacent freed tiles (row-major order fixed).
6. None.
7. Every ruleset.
8. Host C++ (`cpp/src/physics_engine.cpp:1249`) and the device kernel `cpp/src/cuda_stamp.cu` (P4b, served on the resident path through `take_stamp_rows`, see Part B); the refill is Python on every path.
9. Design 4.3 row "6": "MAX stamps; per-tile refill (exact integer division) -> one kernel over unit rows (P4b)". The kernel exists; the refill stays on the host (CLAUDE.md Stamp kernel row: "The refill ... stays on the host until P3e"; P3e has merged and the refill is still host Python).

### Slot 6b: FieldEdit flush

1. `simulation.py:2155-2159`: `runner.serve_edit_queue(...)` when `runner.serves_edit_queue()` (resident + device queue on), else `edit_queue.flush(gmap, draws)`.
2. Applies every queued FieldEdit (this tick's and the previous tick's slot-9e ones) in one stable-sorted pass; on the resident path the host sorts and packs and the device applies at slot 7's start (P4c), with a bit-identical host fallback.
3. Unit columns: none.
4. Map: every edited field (`heat`, `gas`, `atmosphere`, `wave_source`, `fire`, `gas_energy` + mirror, trace books); `sim.edit_queue`; `sim.draws` (noise).
5. Fixed order: the stable sort on `(field, source_id, region, seq)` (CLAUDE.md FieldEdit row; `field_edit.py:830-838` per the Part B report).
6. Draws `Salt.FIELD_NOISE` keyed `(tick, seq, FIELD_NOISE, tile)` (`src/simulation/field_edit.py:585`, `689`). No events.
7. Every ruleset.
8. Python flush + device queue `cpp/src/cuda_edit_queue.cu` (Class P).
9. Design 4.3 row "6b" (P4c): done; consistent.

### Slot 7: physics

1. `simulation.py:2162-2176` -> `PhysicsRunner.step` (`src/simulation/physics_runner.py:906`), which on the resident path is `run_resident_batch([self], [gmap], sim_time)` (`physics_runner.py:1525`, `2013-2164`); `ResidentBatch.step` calls `_physics_head` then `run_resident_batch` for N envs (`resident_batch.py:131-135`).
2. The physics solvers (water, EOS, trace, combustion, fire, temperature, the radiation sweep). Returns the burn-through walls (row-major on the resident path, `physics_runner.py:2024`).
3. Unit columns: none (units reach physics only through slot 6's planes).
4. Every physics plane; outputs read later by unit slots: `heat`, `rad_flux_sweep`, `temperature`, `gas`, `wind_x` / `wind_y`, `atmosphere`.
5. Not a unit slot.
6. Combustion's draw `r0._draw_r` (`physics_runner.py:2134`); not traced further.
7. Skipped when `physics_runner is None` (`simulation.py:2167`).
8. C++ / CUDA (P4a, P5).
9. Design row "7": done (P4a, P5).

### Slot 9: fire burn-through walls

1. `simulation.py:2184-2191`.
2. For each wall slot 7 reports burnt through: `destroy_wall`, and a `DoorDestroyedEvent` or `WallDestroyedEvent`.
3. Unit columns: none.
4. Map: `material`, topology via `destroy_wall` (`gamemap.py:2635ff`: `on_tile_changed`, the ambient seed into `gas`, books brackets, `render_dirty`, `_tile_serial`).
5. Serial in list order; `destroy_wall`'s "exposes the boundary" test reads neighbours an earlier destroy may have changed (`gamemap.py:2714-2722`).
6. Render-only `DoorDestroyedEvent` / `WallDestroyedEvent`. No draw, no noise.
7. Every ruleset; the list is empty without a runner.
8. Python (`destroy_wall`). On the resident path the device returns the first 64 indices per env inside the books D2H and copies any further ones with a separate blocking `cudaMemcpy` (`cpp/src/cuda_resident_tick.cu:1061-1065`, `1284-1288`).
9. Design row "9": "row-major; destroy_wall serial -> the wall law per cell -> applier(topology)" (P6). Consistent. `tools/tick_stage_profile.py` has no label for this slot (its `SLOT_WRAPS`, lines 60-95, wrap no callable here).

### Slot 9b: over-pressure walls

1. `simulation.py:2199-2207` -> `GameMap.find_burst_walls` (`gamemap.py:2523-2621`), then `destroy_wall` per tile.
2. Vectorised scan of solid tiles with a burst threshold whose open-side pressure spread exceeds it; stable sort by descending spread over the row-major scan; cap `burst_max_per_tick` (16); each tile destroyed in that order.
3. Unit columns: none.
4. Map: `solid`, `material`, `is_vacuum`, `is_ambient`, `atmosphere` (read, float64 dequantize, `gamemap.py:2605-2608`); topology writes as slot 9. `CFG.physics.burst_enabled` / `burst_max_per_tick`.
5. The selection is computed once from the post-physics state; the destroys are serial in its order (exposure test as slot 9).
6. Render-only wall / door events.
7. Every ruleset.
8. Python + numpy, float64 comparisons (`gamemap.py:2576-2581`); frozen loop oracle `tests/_reference_scans.py`.
9. Design row "9b": "stable sort by -spread, ties row-major, cap 16 -> per-env top-16 -> applier(topology)" (P6). Consistent (the float64 compare is not mentioned).

### Slot 9c: unit heat damage (`apply_environmental_damage`)

1. `simulation.py:2220-2224` -> `exchange.py:337-487`, `felt_temperature_q` (`exchange.py:294-334`).
2. Per living unit in list order: the felt temperature from the hottest footprint tile of `max(heat, rad_flux_sweep)`; above the profile's `temperature_max` a HEAT packet (credited to the tile's `heat` claimant when the exposure is the `heat` buffer).
3. Reads `life_state`, `x_q`, `y_q`, `footprint`, D `environment` (band and rate), plus apply_packet's reads; writes apply_packet (the set above) on the same unit. About 5 / 12.
4. Map: `heat`, `rad_flux_sweep` (+ its fine-bit currency), `gmap.damage_credit` (read). `CFG.combat`.
5. ORDER-FREE per unit (own packet only). Events in id order.
6. Synced UnitHit / UnitKilled (source `heat`). No draw, no noise.
7. Skipped without a runner (`simulation.py:2220`).
8. Pure Python.
9. Design row "9c ... 9d": "id order; teargas loop then poison loop; events in id order -> one thread per unit row / per cell; the ring in id order" (P6). Consistent.

### Slot 9c2: wind on units (`apply_wind_force`)

1. `simulation.py:2237-2238` -> `exchange.py:588-625` -> `GameMap.footprint_wind` (`gamemap.py:2405`), `unit_impulse.add_wind` (`unit_impulse.py:530-556`).
2. Per living unit: the max dynamic-pressure load over the footprint's gas tiles and that tile's wind become one tick of impulse (prone area while prone, the standing surplus kept in `stand_x/y`). Returns at once when both wind constants are 0 (`exchange.py:616-617`).
3. Reads `life_state`, `x_q`, `y_q`, `footprint`, `statuses` (`is_prone`); writes `imp_x`, `imp_y`, `stand_x`, `stand_y` (+ R `impulse_books`). 5 / 4.
4. Map: `gas` bulk N, `wind_x`, `wind_y`; `sim.impulse_table`.
5. ORDER-FREE per unit.
6. None.
7. Skipped without a runner.
8. Pure Python.
9. Design 4.3: the row "9c heat damage, 9c2 wind, ..." names 9c2 but predates G5p-b (the wind is now an impulse, not #140's cadence scale); "one thread per unit row" fits.

### Slot 9c3: gas rows (`apply_teargas_blind`, `apply_poison_dose`)

1. `simulation.py:2250-2253` -> `exchange.py:639-669`, `exchange.py:672-737`.
2. Teargas: footprint-max teargas density at or above a threshold applies BLINDED (refresh). Poison: footprint-max poison density at or above a threshold emits one POISON packet per tick, skipped for a unit with zero poison resist (the zombie); credited to the claimant of the first densest footprint tile. Each returns at once on an all-zero plane.
3. Teargas reads `life_state`, `x_q`, `y_q`, `footprint`; writes `statuses`. Poison reads the same + D `mitigation`; writes apply_packet (the set above). About 5 / 13.
4. Map: `gas[GAS_TEARGAS]`, `gas[GAS_POISON]`, `gmap.damage_credit` (read). `CFG.exchange`.
5. ORDER-FREE per unit. Teargas loop, then poison loop.
6. Synced UnitHit / UnitKilled (source `poison_gas`). No draw, no noise.
7. Skipped without a runner.
8. Pure Python.
9. Design row "9c ... 9d": consistent today. The unit design (section 6) replaces `apply_poison_dose`'s direct damage with a two-compartment poison law in this slot (patch PX, not on main).

### Slot 9d: ignition from temperature (`apply_temperature_ignition`)

1. `simulation.py:2268-2278` -> `combat.py:632-809`.
2. Vectorised over the grid: re-arm cooled flammable tiles; seed `fire` on armed, hot, oxygenated, fuelled, unlit tiles; disarm hot burning tiles.
3. Unit columns: none.
4. Map: reads `flammable`, `temperature`, `material`, `wall_hp`, `solid`, `is_vacuum`, `gas` (O2, N2); writes `fire`, `ignition_armed`. `CFG.physics.fire`.
5. Order-free (per tile from frozen inputs, `combat.py:695-696`).
6. None.
7. Skipped without a runner.
8. Python + numpy.
9. Design row "9c ... 9d" says "one thread per ... cell". Consistent.

### Slot 9e: entities (sub-steps in order)

`simulation.py:2303-2350`. Sub-steps:

- (a) `_signal_emit` (2304) and `sample_sensors` (2306): only when a SignalBus exists. Motion sensors read every unit's `life_state`, `x_q`, `y_q`, `team`, `footprint` in id order, with `has_los` (`src/simulation/sensor_system.py:134-150`); read only.
- (b) `sweep_logic_nodes` (2308): logic nodes in ordinal order; no unit columns found.
- (c) `_resolve_door_inputs` (2309): wired doors' `want_open`.
- (d) `sweep_pumps` (2311, bus only; `pump_system.py:265`), `sweep_vents` (2319, gated on ducts; `vent_system.py:537`): gas moved through the pump primitives (`transfer_gas_n`, `extract_gas_n_vec`, `inject_gas_n_vec`), immediate field writes with energy booked; then `sweep_doors` (2321; `src/simulation/door_system.py:215-265`): per door in ordinal order, external destruction (whole-door rule, `destroy_wall` per remaining span tile) and the `want_open` latch (`unseal_tiles`; a close only if no living unit footprint is on the span, `door_system.py:203-212`, `257-259`).
- (d') `sweep_timed_charges` (2327-2328; `timed_charge_system.py:180`, `execute_payload` at 198) and planted charges `charges.tick` (2343-2344; `charges.py:121-134`): due charges fire through `execute_payload` (blast damage to units in list order, flash, deposits enqueued for the next 6b, wall damage, `blast_log`, noise); spent charges are dropped from `sim.planted_charges`.
- (d'') `sweep_heat_sources` (2345-2346; `heat_source_system.py:215ff`): reads `gmap.blast_log` (`heat_source_system.py:221`), applies the wall law to each source's hp, a dying source runs its death payload (`execute_payload`, 210), each living source enqueues heat (`deposit_heat`, 246); else the log is cleared (2347-2348).
- (e) `swap_node_signals` (2350).

Per item: 3. unit columns: read by sensors and the door occupancy test (`life_state`, `x_q`, `y_q`, `footprint`, `team`, `id`); written only through `execute_payload` (apply_packet + BLINDED, as slot 2). 4. Entity runtimes (`sim._doors`, `_pumps`, `_ducts`, `_timed_charges`, `_heat_sources`, `_sensors`, `_logic_nodes`, `_signal_bus`), `sim.planted_charges`, `gmap.blast_log`, the edit queue, `gas` / `gas_energy` + mirror, `material` / topology, `wall_hp`. 5. ORDER-DEPENDENT: serial by sub-step and entity ordinal (CLAUDE.md Vent row: duct ordinal; the design notes two actuators may share a tile); a charge's blast damage is immediate in unit list order. 6. Synced UnitHit / UnitKilled from blasts; render-only `DoorDestroyedEvent`, `ExplosionEvent`; noise per detonation; no draws (`payloads.py:439-441`). 7. Planted charges are skipped here when `ruleset.drives_units` (OnePhaseWEGO ticks them in its timeline, 2343); everything else runs on every ruleset and without a runner. 8. Pure Python. 9. Design row "9e": "emit / sensors / logic (ordinal) -> pumps -> vents -> doors -> charges -> heat sources (read blast_log) -> swap; ONE serial applier stage per env" (P6b). The order matches, with planted charges (#193) between the timed charges and the heat sources, which the row does not name separately.

### Recorder snapshot (no number)

`simulation.py:2368-2371`: `recorder.record(...)` when a recorder is attached. No recorder in `run_match` or `MatchSession` (CLAUDE.md Headless match runner row: "no recorder"). Reads every unit (the recorder's own contract); writes none. Not a design row (4.3 "The host keeps ... the recorder export").

### Tick-end clears (no number)

1. `simulation.py:2382-2403` (with a runner): `heat`, `rad_net`, `rad_amb`, `rad_flux` filled with 0 in place. `simulation.py:2421`: `tick_credit(gmap).clear()` (every path). `simulation.py:2424-2426`: `sim.shots` filtered by `real_time` (a float clock).
2. Per-tick buffers wiped after their readers.
3. Unit columns: none.
5. Not per unit.
8. Python.
9. Design row "10": "heat / rad / TickCredit clears" (P6). Consistent.

### Slot 10: tick advance

`simulation.py:2433-2434`: `tick += 1`, `real_time += 1/tps`. No unit columns. Design row "10": consistent.

### Slot 10': order expiry (`unit_orders.expire`)

1. `simulation.py:2438` -> `unit_orders.py:197-206`.
2. Clears every order whose `order_until <= tick`.
3. Reads `order_what`, `order_until`; writes the four `order_*` columns. 2 / 4. Note: no `life_state` check (a dead unit's order was cleared at death).
5. ORDER-FREE per unit.
6. None.
7. Every ruleset.
8. Pure Python.
9. Design 4.3: SILENT (1f).

### Slot 10 (ruleset tail): `on_tick_end` -> zombie conversion

1. `simulation.py:2440` -> `ContinuousRealtime.on_tick_end` (`ruleset.py:275-280`; `MatchRules` inherits it, `ruleset.py:309`) -> `ai_zombie.convert_marines_to_zombies` (`ai_zombie.py:400-434`).
2. Every dead non-zombie with `killed_by_zombie` becomes a living zombie on `ZOMBIE_TEAM` with `[zombie] hp`, its exposure, turn, impulse, horde state and order cleared, `zombie_activated` set.
3. Reads `is_zombie`, `life_state`, `killed_by_zombie`; writes `team`, `is_zombie`, `life_state`, `current_hp` (`= float(CFG.zombie.hp)` through the property setter, `ai_zombie.py:417`), `exposure`, `facing_target_q`, `omega_q` (`turning.settle`, `turning.py:183-184`), the 11 impulse columns, `zombie_noise_y/x/ticks`, `zombie_anchor_y/x`, the four `order_*`, `zombie_activated`, `killed_by_zombie`; R `name`, `impulse_books`. 3 / 27.
4. `CFG.zombie.hp`, `factions.ZOMBIE_TEAM`.
5. ORDER-FREE per unit.
6. None (no conversion event).
7. TwoPhaseWEGO converts at round end (`simulation.py:2662`); OnePhaseWEGO in its `on_tick_end` (`ruleset.py:451-459`). Match termination (`MatchRules.is_terminal`, `ruleset.py:340-342`) is read by the caller, not run in the tick.
8. Pure Python.
9. Design row "10": "convert_marines_to_zombies every tick in id order; the done kernel" (P6). Consistent. (Since `on_tick_end` runs after the tick advance, conversion happens at the new tick number.)

### Slot 11: the team picture (`TeamPicture.update`)

1. `simulation.py:2446-2447` -> `src/simulation/team_picture.py:212ff`.
2. In a match only: each marine team's records from this tick's visibility (`vision.state_for`), the trackers (motion, last hit, damage window, bars, death tick, the object tracker over grenades and charges), seen deaths dropped, memory expiry.
3. Reads every unit's `id`, `life_state`, `is_zombie`, `team`, `x_q`, `y_q`, `footprint`, `facing_q`, `facing_target_q` (mates' aims), `current_hp`, `loadout_name`, `live_trigger` and weapon range (fire lanes), and `sim.tick_events` (`team_picture.py:228-232`); exact set per CLAUDE.md Team picture row, not re-traced line by line. Writes no unit column.
4. `sim.team_picture` (host object, not synced, in no digest); reads `sim.projectiles`, `sim.planted_charges`, `solid` / `sight_screen` (sight lines; the all-round rule calls the C++ `sight_lines` routine through `vision.rays_clear`).
5. Not a unit-state writer; reads all units after every write of the tick.
6. None.
7. Only when `sim.team_picture` is set (`spawn_match`).
8. Python; the sight test in bulk is host C++ (`cpp/src/sight_lines.cpp`).
9. Design 4.3: SILENT (#195 1d). The picture feeds the observation (P8b's domain), not the synced state.

---

## Part B: what exists on the device side to build on

Facts gathered by a read-only sub-investigation (spot-checked: the field-list
lines, `DESTROYED_HEAD`, the profiler labels).

**B1. Per-game field stacking.** Resident fields are `GameMap` attributes listed in `_RESIDENT_SYNCED` (outputs, `src/simulation/gamemap.py:125-158`) or `_RESIDENT_MASKS` (inputs, `gamemap.py:159-262`); `_RESIDENT_FIELD_NAMES` is the two (`gamemap.py:263`). Reassigning a resident field while residency is on raises (`gamemap.py:265-278`). A lone map gets one CuPy copy per name (`enable_residency`, `gamemap.py:2019-2042`; `device_ptrs` `2048-2053`). `ResidentBatch` allocates one `cp.empty((N,) + shape)` per name and binds `gmap._dev[name] = buf[e]`, seeding with `.set` (`resident_batch.py:59-63`, `87-101`); `reset_env` rebinds one env (`107-115`). `from_host` / `to_host` are Python per-name CuPy copies, one call per env per field (`gamemap.py:2055-2069`); in the tick a defaulted (all-names) call is forbidden (`physics_runner.py:1519-1520`). The C++ side gets base addresses through the fixed `ResidentFields` struct (`cpp/src/physics_engine.h:37-60`) filled by name in `resident_fields_from` (`cpp/src/bindings.cpp:70-103`). Per tick (`run_resident_batch`, `physics_runner.py:2013-2164`): host prepare per env (also takes the stamp rows), the entry upload per env (`_resident_entry_names`, `physics_runner.py:1638-1651`: an always set, the stamp planes only when the device stamp did not serve, the static planes only when `_res_tile_serial != _tile_serial`, water when active), device stamp, device edits, water, EOS, trace tail, combustion, the sky bracket (ambient only), the tail with the books D2H, the edit books fold, the exit download per env (`_resident_exit_names`, `physics_runner.py:1653-1661`). To join, a new per-game array must be a `GameMap` attribute in one of the two lists, be named in the entry / exit tuples (`physics_runner.py:1599-1661`), and, if C++ reads it, get a `ResidentFields` member and a `resident_fields_from` line. Reusable as is: `ResidentBatch` allocates `(N,) + a0.shape` for any fixed shape, so a `(U, C)` array would get an `(N, U, C)` buffer and per-env slices; its homogeneity check covers grid, gas table, tilt, tick rate, ambient (`resident_batch.py:70-85`), not unit counts.

**B2. Host-to-device unit rows (the stamp).** `_stamp_units_cpp` (`gamemap.py:2126-2201`) builds one row per living unit's in-bounds footprint tile from `alive`, `occupied_tiles()` and the four optics overrides by `getattr` (quantized with `optics_fixed.quantize_scalar`), calls the C++ oracle and caches arrays + `_tile_serial` in `_stamp_pending`. `take_stamp_rows` (`gamemap.py:2092-2124`) packs `(R, 9)` int32 `[env, y, x, perm bits, wabsorb bits, heat_q, l0, l1, l2]` and clears the cache; it returns None (upload the host planes instead) when no C++ stamp ran since the last take or `_tile_serial` moved since the stamp. The batch rewrites the env column and concatenates; the device stamp serves only if every env has rows (`physics_runner.py:2050`, `2080-2094`). The binding checks `(n, 9)` (`bindings.cpp:4893-4908`); `PhysicsEngine::stamp_units_resident` -> `breach_cuda::resident_stamp` (`physics_engine.cpp:892-903`; driver `cuda_resident_tick.cu:488-528`: host env-range check, pinned `PinBuf` staging, one `cudaMemcpyAsync` on `tstream()`, stage counter 11). Kernels `cuda_stamp.cu`: K1 reset to baselines (75-93), K2 atomic maxes + last row per cell (96-113), K3 the winning row writes permeability (116-131); a flat 1-D grid over N·h·w, not the `bgrid` convention (146-160). Reusable: the host-pack + pinned async upload + host range check idiom; the rows are rebuilt every tick and discarded; they carry no unit identity (no unit id column).

**B3. Books arena and `tick_sync_point`.** `ArenaLayout(N)` lays out regions of `n * <STAGE>_SLOTS` unsigned 64-bit slots: wt, comb, temp, sweep, kick, flux, ecnt, rail, tcnt, ttail, pre, nsub, fire_cnt, fire_gate, edq (`cuda_resident_tick.cu:200-228`); env e's block is `region + e*slots`; a driver passes `arena_dev() + L.<region>`. `resident_tail` does one async D2H of the whole arena plus `DESTROYED_HEAD` (64) destroyed-wall entries per env (`cuda_resident_tick.cu:1057-1066`), `tick_sync_point("books")`, then folds per env into `engs[e]`, assign vs accumulate as on the CPU (`1074-1271`). Exactly two sync names exist: `"max_n_sub"` (`cuda_eos_resident.cu:1097`) and `"books"` (`cuda_resident_tick.cu:1066`); `tick_sync_point` is a stream sync, or under the capture probe closes / runs / restarts the graph segment (`cuda_resident_tick.cu:95-135`; `cuda_resident_tick.h:54-72`). Buffers: `DevBuf` (grow, never freed per tick), `PinBuf`, `upload_if_changed`, per-env int32 scalars `ENV_*` (`cuda_resident_tick.cu:151-198`, `233-240`, `309-318`). How the wt / comb / temp / sweep cores zero their regions: not determined (not read). Reusable: a new stage's books are a new region with `<X>_SLOTS`, folded at the same D2H.

**B4. Tick stream helpers (`cpp/src/cuda_batch.cuh`).** `tstream()` the one blocking stream (38-40; created `cuda_resident_tick.cu:49-58`); `benv()` = `blockIdx.y` (45); `bptr(p, stride)` rebase per env, nullptr stays nullptr (48-51); `sub_off(es, s)` masked substep (55-57); `env_off(active)` per-env byte mask (60-62); `bgrid(n_cells, n_env, block=256)` -> `dim3(ceil(n_cells/block), N)` (65-67); rule text at 7-22 (2-D launch, y = env, body verbatim behind the rebase, never a tall grid). No macros beyond these. Exceptions already in tree: the stamp kernels and edit-queue K1 / K2 are 1-D (`cuda_stamp.cu:146-160`, `cuda_edit_queue.cu:404-413`). Reusable as is for a `(N, U)` kernel (`bgrid(U, N)`).

**B5. Device edit queue.** `field_edit.pack_for_device(gmap, ordered, rng)` (`field_edit.py:927-1007`) takes `EditQueue.take_ordered()`'s order (830-838) and returns `(DevicePack, None)` or `(None, reason)`: EDIT rows int64 x 16 (env, field ordinal, slice, mode, amount, den, noisy, low, key = seq, clamp flags, six bounds) and TILE rows int64 x 4 `(edit idx, r*w+c, weight_q, scale)` from `_iter_region`; fallback reasons: a `water_depth` edit, slice out of range, out-of-range rows or products, noisy edits without `MatchDraws`. Runner: `serves_edit_queue` / `serve_edit_queue` / `_flush_pack_on_host` / `_fold_edit_books` (`physics_runner.py:928-985`), `_edit_buffer` one int64 buffer `[env tile offsets][env scalars][edit rows][tile rows]` (2181-2211), `_check_edit_tables` (2167-2178). Device `resident_edits` (`cuda_resident_tick.cu:534-624`): host row checks, arena memset, pinned staging, one async copy; K1 `edq_contrib` one thread per tile (128-bit magnitude contribution, Philox noise) (`cuda_edit_queue.cu:160-204`), K2 `edq_apply` one thread per env serial in row order (249ff). Reusable: the "host sorts and packs the reference order, device applies serially per env" pattern and the two-level row format.

**B6. Horde field and sight (host C++ only).** `horde::flood(tile_ok, h, w, fp, seeds_yx, n_seeds, dist) -> int64` (integral image + multi-source FIFO BFS, `UNREACHED = 0x7FFFFFFF`; `cpp/src/horde_field.cpp:8-77`, `horde_field.h:31,37`); binding `horde_field(tile_ok, fp, seeds, out)` (`bindings.cpp:4147-4172`). `sight::line_clear`, `sight::rays(solid, screen, h, w, physical, pairs, n, out)`, `sight::window(...)` (`sight_lines.h:42-56`, `sight_lines.cpp:6-55`); bindings `sight_rays` / `sight_window` (`bindings.cpp:4179-4235`). Both are bound outside the CUDA `#ifdef` (`bindings.cpp:1948-1950`, `4139`) and compiled into both modules (`cpp/CMakeLists.txt:99-100`); both release the GIL and read no engine state. No `.cu` / `.cuh` mentions either: no device kernel exists.

**B7. Device fixed-point kit and Philox.** `cpp/src/cuda_fixedpoint_device.cuh` (device-only): `mul128_shr_signed` (54), `flux_to_dq_dev` (69), `heat_saturating_add_dev` (85), `recip_mul_dev` (100), `deposit_dT_wide_q16_dev` (114), `drag_dT_wide_q16_dev` (132), `reciprocal_q16_dev` (159), `scale_mag_dev` (197), `round_nearest_q_dev` (213), `shr_round0_dev` (226), `sqrt_q16_dev` (244). The unit-law helpers are in `fixed_point.h` as `FP_HD` (`__host__ __device__` under nvcc, `fixed_point.h:55-59`): `sin_q16` (1123), `cos_q16` (1137), `atan2_q16` (1167, documented device-safe, 977-991), `mul128_shr` with a `__CUDA_ARCH__` branch (185-193), `sat_add_q16` (440), `sat_add_q16_i64` (458), `floordiv_q` (795), `div_trunc_q` (812), `shr_trunc_q` (818), `effort_step_q` (825), `sqrt_q16` (914), `mul_q16` (115), `mul_wide` (123), `narrow_round_signed` (148), `ceil_div` (837). `recip_mul` has no device branch (242-267). No `.cu` calls `atan2_q16` / `sin_q16` / `cos_q16` yet; a half-away `div_round` exists only file-local in `cuda_edit_queue.cu` (`div_round_i64` :112, `div_128_64` :99). `philox32.h`: `PHILOX32_HD` = `__host__ __device__` (78-84); `philox32_4x32_10` (113), `philox32_draw` (175), `philox32_uniform_q16` (131), `philox32_angle_q16` (139), `philox32_below` (152), `philox32_make_key` / `_counter` (158-171); already used in a kernel (`cuda_edit_queue.cu:52,182`).

**B8. Other per-unit device state.** None. `ResidentFields` has no unit member (`physics_engine.h:37-60`). The only unit data reaching the device is the per-tick stamp rows. The per-unit read of `rad_flux_sweep` (slot 9c) is on the host; the plane is in the always-downloaded exit set (`physics_runner.py:1632-1635`).

---

## Part C: the gates that exist

Facts gathered by a read-only sub-investigation (spot-checked: the profiler
labels). No CUDA gate compares unit records today.

**C1. `tests/cuda_harness.py`.** CUDA build `cpp/build_cuda/breach_physics*.pyd|.so` (:33, 53, 56-59); interpreter `CUDA_PYTHON` (the `data` env; `BREACH_CUDA_PYTHON` overrides, :43-51); `cuda_available()` a presence check (:107-119); `_bootstrap()` adds DLL dirs and puts `cpp/build_cuda`, `src`, repo root, `tests` first on `sys.path` (:125-135); `run_cuda_script(body, timeout)` runs `[CUDA_PYTHON, "-c", bootstrap + body]` (:138-145). Wrapper convention (e.g. `tests/test_cuda_s8a_residency.py:25-40`): skip if unavailable, run `import X; sys.exit(X.main())`, assert `"<NAME>_RESULT: PASS"` and return code 0. `tests/test_cuda_check_scripts_are_wired.py:29-51` fails when a `cuda_*_check.py` is not named in a `test_*.py`.

**C2. `tests/cuda_s8a_check.py`** (wrapper timeout 1800, `S8A_RESULT`). Two identically built worlds, CPU vs resident (`_set_backends` :109-120, `_residency` :123-125). `_run_leg(label, world_cpu, world_gpu, n_ticks, step_fn, evidence, headless_render, post_tick)` (:485-569): compare at init, then per tick ledger snapshot, CPU step, poison the stamp planes and resident step, stamp-plane check, closure residual on both (equal; 0 from tick 1, :520-531), `_compare_all`, `post_tick`; stop at 8 divergences; vacuousness guards (resident calls advanced, per-call counters unmoved, a combustion shim saw 0 calls, stage-counter evidence minimums; :542-562). `_compare_all` (:398-422): every ndarray in `vars(gmap)` with `np.array_equal`, plus `_all_books` (every `e_*`, `*_hits`, `*_sum` int on the solvers and engine, trace books, n_sub, radiation stream, seam nets; :302-352), tol 0. Legs: 1a synthetic room 40 ticks; 1b ambient room 40; 1c MG parity; 1d water 60; 1e `fire_smoke_studio` Simulation 60; 1f `MatchSession(ship_demo, seed=1)` bots + horde 120 ticks (:772-788); 1g ignition gather probe; 2 benches. Unit records are not compared even in leg 1f.

**C3. `tests/cuda_stamp_check.py`** (900 s, `STAMP_RESULT`). Oracle `g.stamp_units(units)`, rows `take_stamp_rows()`, `from_host` static planes, poison the five device planes, `stamp_units_resident`, compare `_dev[name].get()` vs host with `array_equal` (:147-195). 48x64 synthetic room; units built directly with per-unit optics overrides (:87-98); scenes empty, crowd (11 units), edits (walls at ticks 6, 9), dense (34 units) (:204-207); motion with a death and a rebirth (:117-130); row-count and counter guards (:189-192, 216-221).

**C4. `tests/cuda_batch_invariance_check.py`** (1800 s). Env i inside a `ResidentBatch` of N equals env i alone, every tick, by `_fingerprint` (sha1 per gmap array + sha1 of the books; :243-248), no unit records. Leg A: N in (1, 2, 8, 64) (:56), 30 ticks, four templates, seeds 1000+i, a perturbation, topology events only in env 0, env 1 reset at tick 15, per-env edits, n_sub must differ on some tick (:283-347). Leg B: three ship_demo Simulations (entities on, no match), 20 ticks, a blast in env 1, a breach in env 2 (:402-444). Reusable: `RawWorld` (:62-120), `make_world`, `_fingerprint`, `_diff`, `_free` (:275-280).

**C5. `tests/cuda_edit_queue_check.py`** (1800 s). Three Simulations per scene (cpu, resident, a no-edit reference that proves the edits landed, :309-375) through `s8a._run_leg` with `edit_queue` evidence and `edit_device_ticks == N_TICKS`, `edit_host_ticks == 0`; scenes trace, atmosphere, wave, fire, heat, gas_energy, tiles, ambient (:283-292); fallback, the cross-tick overlap (wraps `_run_zombie_slot` / `_step_physics`, :394-445), refusal, noise probe (:496-558).

**C6. `tests/cuda_tick_capture_check.py`** (900 s). Sets `physics_runner._TICK_CAPTURE` to a probe over `bp.resident_capture_begin/_end/_abort` (:51-73); a captured `ResidentBatch` beside an uncaptured twin (3 warm-up + 3 captured ticks, :76-112); asserts `sync_points == ["max_n_sub", "books"]` (:48, 96-101) and captured == uncaptured by fingerprint; scenes N = 1 and N = 3 (:117-118).

**C7. `tests/cuda_resident_bus_check.py`** (900 s). `MatchSession(ship_demo, seed=1)` resident, 40 ticks, a smoke disc and a heat disc per tick, `TickStageProfiler(skip=2, device=True)` (:98-134); asserts exactly 1 H2D and 1 D2H per tick, every device stage between them, required stages present, the stamp planes absent from the H2D when the device stamp ran, the edits stage present, the stamp ran on more than half the ticks (:140-179); label constants :69-90.

**C8. `tests/field_ab_harness.py`** (CPU build). `SIM_FIELDS` (:80-107). `capture_snapshot(sim, fields, capture_units=True)` (:300-316): gmap copies, `__unit_state__` (`_capture_unit_state`), the entity carrier with signals. `_unit_record(u)` = `unit_schema.digest_record(u)` (:168-176; schema `unit_schema.py:722-733`): every column in schema order under its key, always for `id`, `life_state`, `x_q`, `y_q`, `facing`, `current_hp`, else only while it differs from its default. `_charge_records` (:179-186) as `CHARGE_SCHEMA` rows (`unit_schema.py:771-789`). `_event_record` (:189-204): `{kind, unit_id, damage, source, killed_by, attacker_id}`. `_SYNCED_EVENT_TYPES = ("UnitHitEvent", "UnitKilledEvent")` (:219). `_capture_unit_state` (:222-249): units sorted by id as `unit_section_bytes` (`UNIT_SECT_V2`), `repr` of the synced events in emission order, the charge section if any live charge; `{"units", "events", "hash"}`. `tick_digest` is in `tests/field_digest.py:200-239` (blake2b of `field_digest` + the unit hash + entity / signal hashes). Diff helpers `diff_trajectories` (:390-426), `_diff_unit_state` (:319-355). The bridge hooks already in the schema: `encode_unit` / `decode_into` ("the bridge's direction store -> unit; P6's", `unit_schema.py:710-719`).

**C9. Sole-cause drivers.** `tests/_p3a_sole_cause.py`: `<py> tests/_p3a_sole_cause.py OUT.pkl [scenarios]` on two trees, then `--compare A B` (:13-14, 279-282); not pytest-wrapped; `_Tape` records per tick (tick, `digest_record` per unit, charge rows, synced events, `field_digest`) + coverage counters (:52-82); scenarios ship_demo bots seed 1, ship_demo `bots_v1`, box_conversions, a scripted 64x64 arena of 420 ticks driving every verb (:90-214); `compare` reports the first differing tick and part (:230-276). `tests/_p3c_sole_cause.py` reuses it with a normalising patch and adds the two legacy WEGO scenarios (:37-135). `tests/_p3b_sole_cause.py` is a pytest plugin (:1-35). No `_p3e*` driver exists.

**C10. `tools/match_pin_check.py`.** `--backend cpu|cuda|resident` (:10-12, 38-50); plays `run_match(ship_demo)` and compares `(outcome, team, ticks, field_digest, tick_digest)` with `test_zombie_control._PRE_122_SHIP_DEMO` (:34, 51, 58-63). The digests are END STATE only (`tools/run_match.py:111-124`, `src/match_replay.py:185-198`); the pin's per-tick `unit_stream` (`tests/test_zombie_control.py:85-95`, 694-699) is not compared here.

**C11. `tools/cuda_green.py`, `.cuda_green`, `tests/test_class_p_level.py`.** `cuda_green.py` (:56-94): delete the stamp, refuse uncommitted tracked changes, require CUDA and `bp.build_id()` == HEAD, run `CUDA_PYTHON -m pytest tests -q -k cuda`, require exit 0 and >= 1 passed, write `.cuda_green` = HEAD + summary. `test_class_p_level` (:31-39) asserts `twin_debt.class_p_problems(ROOT) == []` (`tools/twin_debt.py:139-156`: a branch touching a Class-P path since the merge base needs a clean tree and the stamp at HEAD; Class-P paths = Class-P twins' `files` and `device` + `[class_p] build` globs, :96-108).

**C12. `tests/twin_manifest.toml`.** Row `[twins.<name>]`: `class` P | T, `kind` (per-call | resident launch core | driver), `device` (hand-kept), `slots` (tick_stage_profile labels with no indent, or `"<label> / <part>"`), `level`, `level_at`, `files` GENERATED (key docs :9-34). Tables `[corpus]` (ship_demo, seed 1, 300 ticks; :41-45), `[class_p] build` (:47-59), `[cpp_stages]` (:61-74), `[not_twinned]` (every executed `src/simulation` module outside a twin with its reason, :76-135; `unit_impulse.py`, `turning.py`, `combat.py`, `verb_gates.py`, `unit_orders.py` marked "Class T ... built from P6", and `simulation.py` "the conductor: its slots are Class T, built from P6"). `tools/twin_manifest_gen.py` traces `MatchSession` on the corpus (CPU build only) under `sys.settrace` with a label stack from `SLOT_WRAPS` / `PHYSICS_WRAPS` (:150-260), unions traced files per slot plus the `[cpp_stages]` include closure (:95-114, 266-287), refuses an unknown label or a part slot without a `[cpp_stages]` row (:278-283), rewrites only `files` (:311-341), `--check` (:369-373). Existing rows (all Class P, level true): water, eos, trace_tail, div_source, energy_flux, radiation_sweep, temperature, fire, combustion, stamp (`slots = "6 stamp_units"`, :439), edit_queue (`"6b FieldEdit flush"`, :485), water_tail. A new row: add the table, run the generator, `tools/twin_debt.py --mark-level X` in the patch's last commit (manifest :23-29; `twin_debt.py:171-212`).

**C13. `tools/tick_stage_profile.py`.** Monkeypatches callables with `perf_counter` wrappers (`_patch`, :225-270); `Simulation.step` is the tick boundary (:199). Slot labels (`SLOT_WRAPS`, :60-95): "1 events + round head" (`reset_trace_books`, `on_round_start`), "1b intents", "2 projectiles", "2b statuses", "3 movement", "4 shooting", "4b sprays", "4c resolve", "5 zombie AI", "6 stamp_units", "6b FieldEdit flush" (`EditQueue.flush`, `serve_edit_queue`), "7 physics" (:51), "9b over-pressure walls", "9c unit heat damage", "9c2 wind on units", "9c3 gas rows", "9d ignition", "9e entities" (the nine 9e callables, not `charges.tick`), "10 tick end (ruleset)". Slot-7 stage labels (`PHYSICS_WRAPS`, :99-124, two leading spaces). Derived / reserved: "step", "(rest of step)", "  (rest of 7: host pre-physics)", "  7 applier (per-env serial, P6 -- reserved)" printed as zeros (:55, 335). Slots with NO label today: 1a, 1b', 1c, 4a, 9 (burn-through loop), 9e(d') `charges.tick`, the tick-end clears, 10', 11; their time lands in "(rest of step)". A twin row cannot name them until a label exists (`twin_manifest_gen.py:278-280`).

**C14. Other two-implementation gates.** `tests/test_unit_impulse_twins.py:42-70` (`unit_fixed` helpers == `bp.fp_*`); `tests/_1e_lockstep.py` (two trees, per-tick `tick_digest`; not pytest-wrapped); `tests/_1h_forced.py` (two in-process `EnvRunner`s, digests equal every tick; wrapped by `tests/test_action_spec_v1.py:920-936`); `tests/_reference_scans.py` (frozen `astar`, `find_burst_walls`; `tests/test_pathfinding_parity.py:102,199`); `tests/test_horde_field.py:120-134` (C++ flood == Python reference); `tests/test_sight_lines.py:145-200`; `tests/test_philox32.py:121-207`; per-tick `unit_stream` pins `tests/test_zombie_control.py:681-699` and `tests/test_bots_v1.py`. Not read: `tests/test_twin_debt.py`, `cuda_prestage_parity_check.py`, `cuda_sight_lines_check.py`, the `DIGEST_FIELDS` spec table.

---

## Part D: summary

Columns: per-unit independent | needs other units' state | writes map topology / fields | synced events | draws (`sim.draws`) | pure Python today | unit columns read / written (approximate; apply_packet's set counted once where it applies).

| slot | per-unit indep. | other units' state | map topology / fields written | synced events | draws | pure Python | cols R / W |
|---|---|---|---|---|---|---|---|
| 1 head | n/a | no | trace books reset | no | no | yes | 0 / 0 |
| 1a integrate + turn | yes | no | no | no | no | yes | 20 / 15 |
| 1b intents | partly | positions, alive (shove target); writes targets' `imp_*` | `door.want_open`; projectile, charge appends | no | no | yes | ~30 / ~18 |
| 1b' orders | yes | no | no | no | no | yes | 2 / 4 |
| 1c exposure | yes | no | no | no | no | yes | 1 / 1 |
| 2 projectiles | no (grenades); partly (rounds phase A) | every unit's position / alive / facing | topology, `wall_hp`, `gas` (bulk, immediate), `blast_log`, `noise_log`, credit, `div_source`, edit queue | yes | yes (EXPOSURE, SHOT_CRIT) | yes | ~8 / ~20 |
| 2b statuses | yes | no | no | yes | no | yes | ~4 / ~12 |
| 3 movement | yes | no | no | no | no | yes | 13 / 5 |
| 4 shooting (A) | yes (decisions) | targets' position / alive / zombie / facing | `noise_log` (immediate); rest recorded | no (at 4c) | yes (SHOT_CONE, BEAM_CONE, EXPOSURE, SHOT_CRIT) | yes | ~20 / 9 + recorded |
| 4a melee (A) | yes (decisions) | enemies' position / alive / team | recorded only | no (at 4c) | no | yes | 14 / 2 + recorded |
| 4b sprays (A) | yes | no | recorded edits + claims | no | no | yes | 10 / 3 |
| 4c resolve | no | n/a (applies to targets) | topology, `wall_hp`, edit queue, credit, payloads | yes | no | yes | ~3 / ~20 |
| 5 zombie AI | no (pass 3) | prey positions / alive; prey hp writes | topology (bash), `noise_log`, horde cache | yes | no (wander is a seedless hash) | yes (flood C++) | ~18 / ~12 + prey |
| 6 stamp | yes (commutative) | no | five stamp planes, `atmosphere` refill | no | no | no (C++ + CUDA; refill Python) | 4 / 0 |
| 6b flush | n/a | no | every edited field | no | yes (FIELD_NOISE) | no (CUDA queue) | 0 / 0 |
| 7 physics | n/a | no | all physics planes | no | combustion's | no | 0 / 0 |
| 9 burn-through | n/a | no | topology | no | no | yes | 0 / 0 |
| 9b burst walls | n/a | no | topology | no | no | yes (numpy) | 0 / 0 |
| 9c heat damage | yes | no | no | yes | no | yes | ~5 / ~12 |
| 9c2 wind | yes | no | no | no | no | yes | 5 / 4 |
| 9c3 gas rows | yes | no | no | yes (poison) | no | yes | ~5 / ~13 |
| 9d ignition | per tile | no | `fire`, `ignition_armed` | no | no | yes (numpy) | 0 / 0 |
| 9e entities | no | positions / alive (sensors, door occupancy); blast hp writes | `gas` / `gas_energy`, topology, `blast_log`, edit queue | yes (blasts) | no | yes | ~6 / ~13 (via blasts) |
| tick-end clears | n/a | no | `heat`, `rad_net`, `rad_amb`, `rad_flux`, credit | no | no | yes | 0 / 0 |
| 10 advance | n/a | no | no | no | no | yes | 0 / 0 |
| 10' order expiry | yes | no | no | no | no | yes | 2 / 4 |
| 10 conversion | yes | no | no | no | no | yes | 3 / 27 |
| 11 team picture | n/a (reader) | all units | no (host picture) | no | no | yes (sight C++) | ~12 / 0 |

**Slots whose per-unit work is order-free as the code stands:** 1a; 1b' (sorted uid); 1c; 2b (sorted id; events in id order); 3; 4 / 4a / 4b phase-A decisions (with their appends and records in id order); 6's unit stamps (MAX; permeability last-wins equal unless overrides differ); 9c; 9c2; 9c3; 10'; conversion. 1b is order-free in its per-unit decisions and commutative in its shove / toggle writes, with the exceptions listed next.

**Order-dependent slots:**
- 1b: projectile append order (= id order) is slot 2's detonation order; `charges.place` refuses by count at capacity, so which plant is refused depends on id order.
- 2 grenades: serial, IMMEDIATE detonations, each seen by the next and by the rounds.
- 2 rounds phase B / 4c: record order (shooters, melee, sprayers by id; rounds by list); first packet to zero kills; credit claims first-wins; chew of an already-destroyed tile is a no-op.
- 5 pass 3: immediate bites with a `players` list fixed at the slot head (a corpse can be bitten); a bash changes `material` mid-slot under a field cache built at first need; the hearing cap is a global selection.
- 6 refill: row-major neighbour mean over tiles written in the same loop.
- 9 / 9b: serial `destroy_wall` whose boundary test reads earlier destroys; 9b's selection is a stable descending sort.
- 9e: entity ordinal order per sub-step; blast damage from a charge is applied in unit list order.
- 11: reads every unit after the tick (no writes).

**Host structures a device unit store would also need on the device** (as the code stands):
- `sim.projectiles`: list of `Projectile` (about 15 attributes, `combat.py:134-167`); no schema, no capacity; on the continuous path detonated entries are never removed.
- `sim.bullets`: list of `BulletInFlight` (about 20 attributes, `combat.py:290-336`, holding references to the shooter, weapon and ammo rows); no schema, no capacity; replaced by survivors each tick (`simulation.py:2491-2501`).
- `sim.planted_charges`: `PlantedCharge` (`charges.py:41-52`); schema `CHARGE_SCHEMA`, capacity `CHARGE_CAPACITY` 32 at `charges.place`.
- `sim._possessed_ids` (set, gates slot 1b), `sim._order_requests` (dict), `sim._zombie_requests` (set or None): no schema.
- `gmap.noise_log` (list of `(y, x, r)`, drained at slot 5), `gmap.blast_log` (list of `(y, x, r, wall_damage_q)`, cleared at 9e), `gmap.damage_credit` (`TickCredit`, a dict `(channel, y, x) -> attacker`, first claim wins, cleared at tick end; `damage.py:218-275`): no schema, no capacity.
- `SimultaneousEffects._pending` (per slot, a list of tuples holding unit and table references and callables, `simultaneous.py:92-148`): no schema.
- `sim.horde_state` (`HordeState`: per-key field arrays and masks, not synced; `horde.py:362-422`).
- Entity runtimes (`_doors` with their per-tile `hp` lists, `_pumps`, `_ducts` and plenum ledgers, `_timed_charges`, `_heat_sources`, `_sensors`, `_logic_nodes`, the `SignalBus`): entity serializer schema exists (CLAUDE.md Entity system row); the device form is design P6b.
- The edit queue: device form exists (P4c).
- `sim.tick_events` (the design's event ring); `sim.shots` (render tracer list); `sim.team_picture` (observation-side, not synced).
- Tables read per unit: weapons / ammo / payloads, `impulse_table`, `speed_table`, `horde_settings`, the status registry, the turning dials, `CFG.zombie` / `CFG.combat` / `CFG.exchange`. A status block is fixed at 16 rows (`unit_schema.py` status capacity; design 4.4a as built).

---

## Part E: surprises

- `Simulation.step`'s docstring order (`simulation.py:1991-2005`) lists an "8. Door explosives" slot and omits 1a, 1b, 1b', 4a, 9c2, 9c3, 9e, 10', 11.
- On the continuous path `sim.projectiles` is never pruned: detonated grenades stay in the list and are skipped each tick (`simulation.py:2457-2459`; pruning only at `simulation.py:2701` and `ruleset.py:477`, both WEGO). The design's 4.5 gives "projectiles" a per-env capacity with a counted overflow; the reference list grows without bound.
- Slot 5's nearest-prey search uses `players` built at the slot head and does not re-check `alive` (`ai_zombie.py:215`, `284-288`), so a zombie can bite a prey killed earlier in the same pass. The design's row says "decisions read stale prey" and so agrees; CLAUDE.md's Horde field row says nothing of it.
- `charges.tick`'s docstring says it "Runs at the head of the ruleset's unit simulation" (`charges.py:125-128`); on the continuous path it runs in slot 9e(d') (`simulation.py:2343-2344`).
- `start_spray_burst` / `start_spray_burst_directional` docstrings say the spray state is "deliberately NOT in the synced digest surface" (`combat.py:1757-1761`, `1779-1781`); `spray_ticks_left`, `spray_target`, `spray_aim_angle` are schema columns hashed while non-default (`unit_schema.py:521-525`, `722-733`).
- `process_shooting`'s docstring calls `rng` "the simulation's numpy.random.Generator" (`combat.py:977-979`); it is `sim.draws` (`MatchDraws`, keyed Philox).
- `apply_temperature_ignition`'s docstring says it runs "ALONGSIDE the existing cellular fire spread" (`combat.py:691-694`); the conductor comment says the cellular spread is DELETED and this path is the sole spread (`simulation.py:2261-2264`).
- The slot-9c conductor comment says the unit samples "the already-occluded `heat` buffer" (`simulation.py:2209-2217`); the row reads `max(heat, rad_flux_sweep)` (`exchange.py:312-330`).
- CLAUDE.md's Stamp kernel row says the atmosphere refill "stays on the host until P3e"; P3e has merged and the refill is still host Python in both stamp paths (`gamemap.py:2203-2211`, `2330-2335`).
- The resident burn-through list is copied in two parts: the first 64 per env inside the "books" D2H, any further entries with a separate blocking `cudaMemcpy` per env (`cuda_resident_tick.cu:1061-1065`, `1284-1288`), a host transfer beyond the two named sync points; not determined whether `cuda_tick_capture_check` ever reaches it (its scenes were not traced to that count).
- `melee_intent` does not add the unit to `_possessed_ids` (`simulation.py:1231-1254`), unlike every other hand verb; slot 4a processes every unit's latch regardless of possession, while slot 1b processes only possessed units.
- Design 4.4 orders store rows (team, slot); every reference slot walks `sim.units` in id (spawn) order (`simulation.py:580-588`), and slot 2b and 1b' sort explicitly by id. Whether (team, slot) order equals id order for every match spawn plus conversions was not determined (the spawn order in `simulation.match.spawn_match` was not read).
- Design 4.3 row "1b" says "throws get a spawn ordinal"; a `Projectile` has no ordinal field (`combat.py:134-167`); `BulletInFlight` does (`combat.py:331-336`).
- `tick_stage_profile` has no label for slot 9's burn-through loop, nor for 1a, 1b', 1c, 4a, 9e(d'), 10', 11 (`tools/tick_stage_profile.py:60-95`), so the twin manifest cannot name those slots yet (`twin_manifest_gen.py:278-280`).
- No CUDA gate compares unit records: `cuda_s8a_check._compare_all` and the batch fingerprint read GameMap arrays and books only (`tests/cuda_s8a_check.py:398-422`, `tests/cuda_batch_invariance_check.py:243-248`), and `match_pin_check` compares end-state digests only (`tools/run_match.py:111-124`).
