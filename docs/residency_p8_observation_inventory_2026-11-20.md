# P8 observation inventory -- the observation, masks, reward and reset as they stand (read-only)

main at `dde25e03` (2026-11-20, after P7e.5 and the #97 water close). Read-only: nothing built, run
or changed. "NOT FOUND" means I searched and did not find it. Anchors are file + function (line
numbers only as a help, on this commit).

Key files: the spec `src/simulation/obs_spec_v1.py` (2331 lines: `observe_rows`, `TeamKnowledge.build`,
`FieldPlanes`, `build_vector`, `build_list`, `build_windows`, `FullView`, `observe_full_rows`); the door
`src/simulation/sensor_accessor.py` (`EntityFieldAccessor.observe`); the env `src/rl_env.py`
(`EnvRunner.observe` / `.step`, `observe_v1`, `count_events`, `BatchedEnv`); the masks
`src/simulation/action_spec_v1.py` (`build_masks`); the decoder `src/control_policy.py`
(`AgentPolicy.set_decisions` / `_decode_frame` / `_decode`); the gates `src/simulation/verb_gates.py`;
the reward `src/simulation/reward_v1.py` (`RewardBooks`); the picture `src/simulation/team_picture.py`;
sight `src/simulation/vision.py`, `cpp/src/sight_lines.cpp`, `cpp/src/cuda_unit_sight.cuh`; the device
`src/simulation/conductor.py` (`DeviceConductor`), `src/simulation/unit_store.py`,
`src/simulation/unit_tables.py`, `cpp/src/cuda_unit_lib.cuh`, `cpp/src/cuda_unit_damage.cuh`,
`cpp/src/cuda_unit_horde.cuh`.

How to read "resident?": **R** = already on the card in the conductor, named (store column
`col:x`, unit-table row `tab:x`, object table, resident plane `plane:x`); **R-fn** = a device
function already computes it (named); **H** = host only today, with what would have to become
resident. "Float" = the first place a float appears in the reference arithmetic.

The obs is built per DECISION (every `decimation` = 4 ticks, design 13.1) at the start of the decision
tick, for every row `t*S + s` of the env; the team picture, the reward books and the decoder run EVERY
TICK. That split matters for the whole port (sections C, E, F).

---------------------------------------------------------------------------------------------

## A. The observation as a table

### A.0 The keys and shapes (spec v1 + masks + full view; `obs_spec_v1.shapes`, `full_shapes`, `action_spec_v1.mask_shapes`)

U = T x S rows per env (T = most marine teams of any level, S = largest `max_per_team`); E = S + 33
list entries; K = hand-grenade rows (6 today).

| key | shape | dtype | built by | per |
|---|---|---|---|---|
| `vector` | (N, U, 97) | float32 | `build_vector` | living row |
| `list` | (N, U, E, 67) | float32 | `build_list` | living row |
| `list_mask` | (N, U, E) | uint8 | `build_list` | living row |
| `list_uid` | (N, U, E) | int32 | `build_list` (an equality key, never a network input) | living row |
| `near_f` | (N, U, 8, 25, 25) | float16 | `build_windows` | living row |
| `near_u` | (N, U, 5, 25, 25) | uint8 | `build_windows` | living row |
| `far_f` | (N, U, 10, 25, 25) | float16 | `build_windows` | living row |
| `far_u` | (N, U, 5, 25, 25) | uint8 | `build_windows` | living row |
| `valid`, `in_match` | (N, U) | uint8 | `EnvRunner.row_flags` | row |
| `mask_feet` / `_aim` / `_trigger` / `_hand` | (N, U, 49 / 18+E / 2 / 7+K) | uint8 | `build_masks` | row (invalid rows get `default_masks`) |
| `full_list` | (N, U, 64, 85) | float32 | `build_full_list` | in_match row (dead / turned too) |
| `full_mask`, `full_uid` | (N, U, 64) | uint8 / int32 | `build_full_list` | in_match row |
| `full_match` | (N, U, 2T+3) | float32 | `FullView.match` (one per env, copied per row) | in_match row |
| `full_watched` | (N, U, 25, 25) | uint8 | `build_full_watched` | in_match row |

Every key has a `final_` twin (the terminal observation of an env that ended inside the step), written
by the same builders before the reset (`_worker_main` "step").

### A.1 What a builder reads, once per env per decision (`observe_rows`)

| read | what | resident? |
|---|---|---|
| `FieldPlanes(gmap, doors, impulse_table)` | the 8 physics planes of the WHOLE map in float64, their 4x4 block reduction, `wall`, `wall_share`, `door`, `far_door` | sources R: `plane:temperature`, `plane:fire`, `plane:gas` (poison, teargas, smoke slices), `plane:atmosphere`, `plane:wind_x/y`, `plane:solid`; doors R (`DOOR_TABLE` + `DOOR_TILE_TABLE`, state + span tiles). The float planes themselves: H (built per decision) |
| `TeamKnowledge.build(sim, picture, team)` one per marine team | `mates` (living, non-zombie units of the team), `roster_ids`, `records` (the picture's), `seen` (records in `vision.state_for(sim).visible_by_team[team]`), `objects` (`picture.seen_now[team]`), `doors` (living), the trackers | units R (store); picture H; vision H (section B); doors R |
| `ObsWorld` | `gmap`, `tick`, weapons / speed / impulse tables, `graph` (room graph), `wires`, `cover`, `doors`, the turn-rate / accel / after-turn functions | tables mostly R (A.6); graph H (static); wires H (static); cover: the device has none (B.5) |
| `ObsContext.build` once per EPISODE | every denominator (A.6) | partly R, partly H |
| `aims` (the decoder's view, 1h) | per unit `(aimed_uid, prev)` from `AgentPolicy.aim_view` | H (controller state, section F) |

### A.2 `vector` (97 columns, `build_vector`; float32; ratio = float64 `float(x) / float(d)`, 0 when d = 0)

| col | name | reads | arithmetic | resident? |
|---|---|---|---|---|
| 0 | hp | `current_hp`, picture `bars[uid]` | squash(ratio) | `col:current_hp` R; bar H (picture) |
| 1-2 | facing cos, sin | `facing_q` | `cos_q16(f) / 65536` (exact) | `col:facing_q` R |
| 3 | omega | `omega_q` / unit turn rate | squash(ratio) | `col:omega_q` R; turn rate = `scaled(row, UNIT_HOOK_TURN, tab:turn_rate_class_q16)` R-fn (inside `cuda_unit_impulse.cu`; not a lib function) |
| 4 | turn_left | `wrap_q16(facing_target_q - facing_q) / PI_Q16` | int, then one float division | `col:facing_target_q` R |
| 5-6 | vel x, y | picture `motion[uid]` (pos now - pos prev, Q16) / `v_ref_q` | squash(ratio) | motion H (picture tracker; = this tick's minus last tick's `x_q`/`y_q`) |
| 7-8 | pushed x, y | `pvx`, `pvy` / `v_fall_q` | squash(ratio) | `col:pvx/pvy` R, `tab:impulse_v_fall_q` R |
| 9 | effort | `effort` / FP_ONE | squash(ratio) | `col:effort` R |
| 10 | react | `react_ticks` / table react_ticks | ratio | `col:react_ticks`, `tab:impulse_react_ticks` R |
| 11-14 | prone, can_move, can_act, can_aim | `status.composed_flags(u)` | flags | R-fn `unit_lib::composed_flags` |
| 15-23 | status_left[k] | max `remaining_ticks` of kind k / `STATUS_SCALE_TICKS[k]` | squash(ratio) | `col:statuses` R; scale H (A.6) |
| 24-28 | rounds, mags, ammo_untracked, reloading, reload_left | `combat.mag_status` (`_kit_values`), loadout `magazines`, `unit_traits.reload_ticks` | ratio; squash on mags | R-fn `unit_lib::mag_status`; reload ticks = `handling_ticks(row, tab:weapon_reload_ticks)` R-fn; loadout magazines: `col:magazines` R |
| 29-40 | grenade[k], grenade_kind_ok[k] (k < 6) | `grenade_counts[k]` / the LOADOUT's count of row k | ratio, flag | `col:grenade_counts` R; the loadout's per-kind count H (no loadout table on the card) |
| 41-43 | throw_windup, throw_left, throw_kind | `throw_release_tick`, `unit_traits.throw_windup_ticks`, `throw_kind` | ratio | R cols; windup = `handling_ticks(row, tab:hand_windup_class_ticks)` R-fn |
| 44-45 | charges, has_charge_kind | `has_explosive` / loadout `explosives`; loadout `charge` | ratio, flag | `col:has_explosive` R; loadout explosives / charge H |
| 46-48 | planting, plant_left, plant_timer | `plant_done_tick`, `unit_traits.plant_ticks`, `plant_fuse_ticks` / `c4_fuse_max_ticks` | ratio | R cols; `tab:plant_class_ticks` R; c4 fuse max H (A.6) |
| 49-51 | melee_cool, has_melee, shove_cool | `last_blow_tick` + melee row rof; `melee_id`; `last_shove_tick` + shove cooldown | ratio | R cols; `tab:weapon_rof_ticks`, `tab:shove_cooldown_ticks` R |
| 52-53 | class, weapon | `unit_schema.table_row("loadouts"/"weapons", name)` | kind (row index) | `col:loadout_name`, `col:weapon_id` R (they ARE the table indices) |
| 54-56 | range, rof, spread | weapon `range_tiles` / `L_ref`; rof / ref rof; `unit_traits.shot_cone_q16` / ref cone | squash(ratio) | `tab:weapon_range_tiles`, `tab:weapon_rof_ticks`, `tab:weapon_cone_aim_q` R; the SPREAD hook `scaled(loc, UNIT_HOOK_SPREAD, cone)` R-fn (inside `cuda_unit_shooting.cu`) |
| 57 | trigger | `live_trigger` | flag | R |
| 58-65 | air[8] | the near window's raw value at the centre tile, `physics_value` | float64 (A.4), squash | planes R; the float law H |
| 66-70 | hit_recent, hit_age, hit_from cos/sin, hit_cause | picture `last_hit[uid]` = (tick, dir_x_q, dir_y_q, cause) | `math.hypot` in `_hit_from` (FLOAT, transcendental-class) | H (picture tracker; its source is the event ring: B/C) |
| 71-76 | prev[6] | the decoder's last decision | kinds | H (controller state) |
| 77 | time_left | `limit_ticks - tick` | ratio | `tab:match_limit_ticks` R; tick R (tick row) |
| 78 | mates_alive | count of `know.mates` / n_t | ratio | store R; n_t H (spec) |
| 79 | team_ammo | every mate's `mag_status` rounds + spares x mag_size / (n_t x loadout magazines x mag_size) | squash(ratio) | R-fn `mag_status`; team loadout H |
| 80-83 | enemies_known, enemies_seen, zombies_known, zombies_near | the team's records (marines / zombies), `seen`, records within `zombie_near_tiles` of any mate | ratio / squash | records H (picture) |
| 84-95 | order_what .. has_target | `order_what/where/until/target`, `room_graph.place_at`, `_place_centre`, the ROUTE field (`route` -> `horde.flood`), `horde.candidates`, `horde.field_at` | `math.hypot` + `d_c` (log1p) at 91: FLOAT, transcendental-class | order cols R; graph H (static); route H (B.6) |
| 96 | felt_temp | `exchange.felt_temperature(gmap, u, heat_dials)` / env `temperature_max` | the law is Q16 (`felt_temperature_q`), dequantized, ratio, squash | R-fn `cuda_unit_damage.cuh::felt_temperature_q` (its header says "from P8b, the observation's ONE read"); `plane:heat`, `plane:rad_flux_sweep` R; env `temperature_max` float H |

### A.3 `list` (E entries x 67 columns, `build_list`; float32)

Five blocks at fixed places: mates (cap S-1), seen enemies (16), records (8), objects (6), doors (4);
each block sorted by squared centre distance, ties the lower id (`_candidates`). `list_uid`: the unit id,
`OBJECT_UID_BASE + k` (the team's own object counter), `DOOR_UID_BASE + ordinal`.

| cols | name | applies to | reads | arithmetic | resident? |
|---|---|---|---|---|---|
| 0 | type | all | the block | kind | -- |
| 1-3 | dx, dy, dist_c | all | centre offsets (mates live, enemies / records RECORDED centre, objects their tile, doors span middle) | squash(ratio over `L_ref`); `d_c(hypot)` = `log1p(hypot)/log1p(L_ref)` FLOAT, transcendental-class | units R; records H; objects R (tables); doors R |
| 4-5 | cos, sin | units | mate `facing_q`, seen enemy LIVE `facing_q`, record `facing_q` | exact | R / H (record) |
| 6-7 | vel x, y | units, grenades | mate: picture motion; enemy / record: `rec.vel_q`; grenade: (target - start) x speed / tps / dist | squash(ratio); grenade: `math.sqrt` (IEEE, exact-class) | H (picture) / R (`GRENADE_TABLE`: start/target/pos Q16, `travel_speed_q`) |
| 8-9 | side, class | units (-1 for objects / doors) | team, loadout row (zombie_class for a zombie) | kind | R / H (record) |
| 10 | hp | mates, seen | `current_hp` / picture bar | squash(ratio) | R + H (bar) |
| 11-23 | exposure, firing, reloading, reload_left, winding, throw_left, throw_kind, planting, plant_left, plant_timer, stunned, blinded, down | mates, seen (`_unit_columns`, LIVE unit) | `exposure[KINETIC]` / `x_max_q`, `live_trigger`, mag_status, throw / plant columns, composed_flags | ratio, flags | all R (`col:exposure`, `tab:expose_x_max_q`, R-fn) |
| 24 | seen_by_me | seen | `vision.observers_of(euid)` contains me | flag | H (vision state, B.2) |
| 25 | lof_solid | seen | `line_of_fire.solid_in_way` (me -> enemy, `rays_clear(physical=True)` + cover) | flag | H (sight, B) |
| 26 | lof_mate | seen | `line_of_fire.mate_in_way` (integer geometry over the other mates' centres) | flag | inputs R; function H (pure integer, trivially portable) |
| 27 | ticks_to_on | seen | `turning.ticks_to_within` (iterates `law_q16` up to `4 x half_turn_ticks`) / half turn | squash(ratio) | R-fn `unit_lib::turn_law_q16`; `unit_impulse.pending_dw` (imp_l / inertia) R cols; the iteration H |
| 28 | on_next_tick | the aimed entry | `aim_vector` (FLOAT: `quantize_stick_direction`, IEEE sqrt) then `facing_after_turn_q16` | flag | H (decoder state + float seam) |
| 29-30 | in_range, in_throw_range | seen | d2 vs `range_tiles^2`, `max_throw_range^2` | flags | R tables |
| 31-32 | facing_on_me, aims_at | seen | `aim_of(know, it)` (B.3) | flag, kind | H (sight) |
| 33 | zombie_state | seen zombie | `zombie_activated`, `zombie_noise_ticks` | kind | R |
| 34-35 | mates_aiming, mates_firing | seen | every mate's `aim_of` answer and trigger | ratio over n_t | H (sight) |
| 36 | dmg_last_s | seen | picture damage window / bar | squash(ratio) | H (picture) |
| 37 | priority | seen | `order_target == euid` | flag | R |
| 38 | frag_catch | seen | mates within `frag_radius` of the enemy | ratio | R + H (frag radius in ctx) |
| 39-46 | aim cos/sin, aim_is_unit, feet dx/dy, throttle, land dx/dy | mates (`_mate_tail`) | `facing_target_q`; `aim_of(m)`; `live_move_dir`; `throw_reach_q` x cos/sin(facing_target) | exact / ratio; land: float products (IEEE) | R; aim_is_unit H (sight) |
| 47-51 | order_what, place dx/dy, same_place, order_left | mates | order columns, `_place_centre` | ratio | R + graph H |
| 52-53 | me_in_its_lane, it_in_my_lane | mates | `know.lane_of(uid)` (the rasteriser over ONE mate) | flag | H (lanes, B.4) |
| 54 | age | records | `tick - rec.tick` / `memory_ticks` | ratio | H (picture) |
| 55-61 | obj_kind, whose, time_left, remote, blast_r, in_blast, on_floor | objects (`_object_entry`) | ammo row index in hand+charge rows, owner vs roster, detonate tick, charge `det_tick is None`, payload radius, landed | ratio, squash | R (`GRENADE_TABLE`, `CHARGE_TABLE`, `tab:payload_radius`, `tab:ammo_payload`); roster H (static per episode) |
| 62-65 | open, in_reach, hull, wired | doors (`_door_entry`) | door state; `verb_gates.door_in_reach` (span vs footprint ring); `horde.is_hull_door`; wired set | flags | R (door table); R-fn `cuda_unit_horde.cuh` is_hull_door; in_reach: geometry inside `cuda_unit_verbs.cu::use_door` (not a lib function); wired H (static) |
| 66 | prev_aim | the aimed entry | decoder `aimed_uid` | flag | H (decoder) |

Objects reach the list ONLY through `picture.seen_now[team]` (the team's object tracker) and units
only through the team's records and its mates (rule file "THE WALL").

### A.4 The windows (`build_windows`; near tiles 25x25 centred, far 4x4 blocks 25x25 centred, block grid anchored at (0,0))

The physics law (`physics_value`, also the vector's `air` block), per plane, all FLOAT64 then squash, then
written into float16:

| plane | raw (`FieldPlanes.raw`) | divided by | float first appears | exactness class |
|---|---|---|---|---|
| temperature | `gas_fixed.dequantize_q16(temperature)` (/65536 exact) | the unit's env profile `temperature_max` (float) | the dequantize (exact) | IEEE division: exact-class |
| fire | `fire_fixed.dequantize(fire)` | nothing | exact | exact |
| poison | `dequantize_trace(gas[POISON])` (/2^TRACE exact) | `poison_min_density` (config float) x `mitigation_for(u).resist_mult[POISON]` (float) | two IEEE ops | exact-class |
| teargas | `dequantize_trace(gas[TEARGAS])` | `teargas_blind_density` | IEEE | exact-class |
| smoke | `dequantize_trace(gas[SMOKE])` | `[bots] smoke_dense` | IEEE | exact-class |
| pressure | `atmosphere_fixed.dequantize(atmosphere)` | `(raw - pressure_min) / (pressure_max - pressure_min)` (env profile floats) | IEEE | exact-class |
| wind_x / wind_y | `GameMap.wind_load_plane`: W = float64(N_O2 + N_N2) x float64((vx^2 + vy^2) >> 16), 0 on solid; j = floor(floor(W / 65536) x wind_k_q / 65536); direction = vx / `np.hypot(vx, vy)` | `mass_g x cap_q` | `np.hypot` | TRANSCENDENTAL-class (hypot is not correctly rounded on every libm; CUDA's has up to 2 ulp) |

Then the squash `sign(x)(1 + ln|x|)` beyond 1, capped at 4 (`squash`, `squash_array`): `log` is
TRANSCENDENTAL-class for every value with |x| > 1.

| key | plane | source | arithmetic | resident? |
|---|---|---|---|---|
| near_f[0..7] | the 8 physics planes, cropped at the soldier's centre, outside the map 0 | `FieldPlanes.raw` | the law above, float16 | planes R; law H |
| near_u[0] | wall | `solid` crop, outside 1 | uint8 | R |
| near_u[1] | door | per living door span tile: 1 open, 2 closed (max) | uint8 | R (door table + tiles) |
| near_u[2] | watched | OR over the team's RECORDS of `vision.window_clear(from rec.centre to every window tile)` | uint8 | H (B.1) |
| near_u[3] | sees_target | `window_clear(to_window=True)`: from every window tile TO the aimed entry's record centre (0 without one) | uint8 | H (B.1) |
| near_u[4] | lanes | the team's `fire_lanes` grid sampled at `(y // cell, x // cell)`: -2, or an owner other than me | uint8 | H (B.4) |
| far_f[0] | wall_share | block mean of `solid` (pad 1) = count / 16 | float16 (exact: k/16) | R |
| far_f[1..8] | physics | block MAX of the raw plane (pad 0); wind: the raw wind of the block's ARGMAX-W tile (first max) | the law, float16 | planes R |
| far_f[9] | route | the order's route field (`horde.flood`) block MIN over 16 tiles, / `L_ref`, squash; unreached -1; no order -1 | float16 | H (B.6) |
| far_u[0] | door | block max of the door plane | uint8 | R |
| far_u[1..3] | mates, enemies, records | counts per block: other living mates at live centres; records at RECORDED centres split seen / not seen | uint8 counts | store R; records H |
| far_u[4] | place | blocks holding a tile of the order's place (`place_blocks`) | uint8 | H (static per graph and place) |

### A.5 The full view (`full_v1`, the other side of the wall; `FullView`, `build_full_list`, `build_full_watched`, `FullView.match`)

| key / cols | what | reads | resident? |
|---|---|---|---|
| full_list entry 0 | the soldier itself (alive: its mate entry at offset 0; dead / turned: `type` 0, hp 0, `age` = (tick - `picture.death_tick`) / limit) | as A.3 + `death_tick` | death_tick H (picture) |
| full_list entries 1.. | EVERY other living marine (team, slot order) as mate (own team) or type-1 entry (other team) with the LIVE unit's columns, then zombies nearest first, then live objects, then doors; over 64 drop farthest doors, then zombies, then objects | store; `live_objects(sim)` (grenades not detonated, then live charges, in table order); doors | units R; objects R; door R |
| cols 0-66 | 2.2's entry against the observing soldier; `aims_at` over EVERY living unit (`FullView.aim`), lanes per team (`FullView.lane_of`), `lof` batched (`line_of_fire.solid_in_way_many`) | as A.3 but live | sight H |
| col 67 knowledge | 0 seen / 1 record / 2 unknown (objects: seen now / ever seen by the team) | picture records, vision, `object_ids`, `seen_now` | H |
| cols 68-84 | kit (`_kit_values`: rounds, mags, untracked, grenades[6], charges, melee_cool, shove_cool), 4 statuses (knocked_down, stunned, burning, poisoned), order_what | store + loadout | R + loadout H |
| full_match | per team: living count / n_t, `reward_v1.standing` H(t) / n_t squashed; zombies awake / dormant / spec.zombies squashed; time_left | store, picture bars | R + bars H |
| full_watched | OR over EVERY living enemy (zombies included, dormant or not) at its LIVE centre of `window_clear(from it, to the window)` | sight | H (B.1) |

### A.6 The per-episode table values (`ObsContext.build`; read once)

| value | source | on the card today? |
|---|---|---|
| `l_ref` (ref weapon `range_tiles`), `ref_rof_ticks`, `ref_cone_q` (`cone_q16_of_deg(spread_deg)`), `ref_melee_arc_q` | the `[marine] loadout`'s weapon and melee rows | the per-row values are (`tab:weapon_range_tiles`, `tab:weapon_rof_ticks`, `tab:weapon_cone_aim_q`, `tab:weapon_melee_arc_half_q`); "which row is the reference kit" is H |
| `v_ref_q` = 2^32 // `speed_table.full_cost_q` | speed table | `tab:speed_full_cost_q` R |
| `limit_ticks`, `memory_ticks`, `tps` | MatchRules, `[team_picture] memory_s`, sim | `tab:match_limit_ticks`, `tab:clock_tps` R; memory_ticks H |
| `hand_rows`, `charge_rows`, `frag_radius`, `max_throw_range`, `c4_fuse_max_ticks`, `hand_fuse_max_ticks` | weapons tables | partly (`tab:hand_grenade_ammo_row`, `tab:hand_fuse_hi_ticks`, `tab:payload_radius`); c4 fuse max and the charge rows list H |
| `status_scale[k]` | `status_scale_ticks` (max over payloads / weapons) | H |
| `turn_rate_q`, `half_turn_ticks`, `line_clear_tiles`, `zombie_near_tiles`, lane dials (`[bots] lane_cell`, `lane_halfwidth`) | turning, `[bots]` | turn class R; the rest H |
| `x_max_q`, `v_fall_q`, `react_ticks`, `shove_cooldown_ticks`, `cap_q` | exposure / impulse tables | R (`tab:expose_x_max_q`, `tab:impulse_v_fall_q`, `tab:impulse_react_ticks`, `tab:shove_cooldown_ticks`, `tab:impulse_cap_q`) |
| `zombie_class` | number of loadout rows | H |
| `poison_min_density`, `teargas_blind_density`, `smoke_dense` | config FLOATS | H (the card has `tab:poison_thresh_q`, `tab:teargas_thresh_q` in Q16: equal as values only if the float is a Q16 multiple: K) |
| `heat_dials` | `exchange.heat_dials` | R as `tab:heat_gain_q`, `tab:heat_ambient_q` (the Q16 law) |
| env profile `temperature_max`, `pressure_min`, `pressure_max` (floats, per unit through `u.environment`) | `[env_profiles.*]` | `tab:heat_temp_max_profile_q` is Q16; pressure_min / max NOT FOUND on the card |
| `mitigation_for(u).resist_mult[POISON]` (float) | species / zombie mitigation | `tab:mitig_resist_zombie_q` / `_species_q` are Q16 (equality to the float: K) |
| per LOADOUT: grenades per kind, `magazines`, `explosives`, `charge`, `weapon`; per TEAM: n_t, team loadout; `spec.zombies`, `n_teams` | `[loadouts.*]`, MatchSpec | NOT FOUND on the card (only per-unit copies: `col:magazines`, `col:charge_ammo`) |

What has to become resident for A: one OBS TABLE per batch (the values above as int64 / float64
bit patterns, read from the reference's own `ObsContext` and loadout objects, the "Device tables" rule)
plus the per-episode roster / team rows.

---------------------------------------------------------------------------------------------
