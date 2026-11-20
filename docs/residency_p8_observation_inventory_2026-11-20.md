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

## B. Sight

Every line the observation, the picture and the full view ask is `GameMap.has_los`'s Bresenham walk
(int64, no float), through `vision.rays_clear` (many pairs) or `vision.window_clear` (one tile to / from
every tile of a rectangle), host C++ (`cpp/src/sight_lines.cpp`, `PhysicsEngine.sight_rays` /
`sight_window`) plus cover in Python. The DEVICE has the per-line function in both forms
(`cpp/src/cuda_unit_sight.cuh::unit_sight::has_los(solid, screen | nullptr, ...)`, held equal to
`has_los` by `tests/cuda_unit_kit_check.py` part SIGHT) and BOTH planes resident (`plane:solid`,
`plane:sight_screen`, resident since P7e). Its header says "The bulk form (sight_lines.cpp's rays_clear,
the observation's) stays P8b's": NO bulk device kernel exists (no many-rays, no window kernel).
`.claude/rules/sight.md` says "No device kernel: the CUDA module runs the same host TU": that line
becomes stale the day a bulk device kernel lands (to amend in that patch).

Cost shapes: W = 625 (a 25 x 25 window); R_t = the team's records (every enemy marine AND every zombie
it saw within `memory_s`, uncapped); L_e = living units of the env; M_t = the team's living mates.

| # | question | who asks, when | today's call | rays (per env) | direction | device mapping |
|---|---|---|---|---|---|---|
| B.1a | `watched`: which window tiles a known enemy sees | `build_windows`, per living row, per decision | `window_clear(rec.centre -> window)` per record, OR-ed | sum over rows of W x R_t | FROM the record centre | one thread per (env, row, window tile), a loop over the team's records calling `has_los(solid, screen)`; needs the records on the card (C) |
| B.1b | `sees_target`: from which window tiles the aimed entry is visible | `build_windows`, rows with an aimed entry | `window_clear(..., to_window=True)` | W per aiming row | FROM each window tile TO the centre (the walk is not symmetric) | the same kernel, the other argument order |
| B.1c | `full_watched` | `build_full_watched`, per in_match row (dead rows too) | `FullView.watched_from` per living enemy at its LIVE centre (cached per (centre, window origin): shared only between rows with the same window) | sum over in_match rows of W x (living enemies incl. zombies) | FROM the enemy | the same kernel over the store's live centres |
| B.2 | who sees whom (`VisionState`: `seen_by`, `visible_by_team`) | `vision.state_for(sim)`, once per TICK (cached on `sim._vision_cache`), read by the picture's update, the obs (`seen`, col 24) and the full view | `all_round`: `_compute_all_round`, ONE `rays_clear` over every (non-zombie observer, other-team living target) pair. `cone` (the DEFAULT: ship_demo's match.toml names no sight, so both pinned matches run it): `compute_pairwise`, per pair `can_see` = awareness radius 4 OR `within_cone` (FLOAT64 `dot / math.sqrt(d2) >= cos_rad(radians(55.0))`: IEEE ops only, `cos_rad` = the integer kit) then `ray_clear` | ~L_e^2 per tick | observer -> target | one thread per (env, observer, target): the cone test in float64 with FMA contraction OFF (`dx*fx + dy*fy` must not fuse), then `has_los`; output a per-env bit matrix (U x U); the reference's sorted tuples become row (= id) order |
| B.3 | `aim_of(u)`: the nearest known unit inside u's cone with a clear line (cols 31-32, 34-35, 41; the full view: over every living unit) | per unit per decision, cached per (uid, ray, cone, centre); the full view keeps its own | `ray_clear` per candidate, nearest first (squared distance, then uid), until one is clear | <= candidates per unit | u's centre -> candidate | a serial loop per (env, view, unit): candidates (<= U + R_t) sorted by (d2, uid), the cone test in Q16 (`atan2_q16`, `wrap_q16`: integer), `has_los` until clear |
| B.4 | the fire lanes (near_u[4], cols 52-53) | `TeamKnowledge.lanes` per team, `lane_of` per mate, per decision; the full view its own | `team_picture.fire_lanes_of`: a per-tile walk on `solid` (NOT `has_los`) from each triggering mate's centre along its facing to its weapon range, then `draw_lanes` onto a coarse grid (cell 3, half-width 1): -1 nobody, uid, -2 several (order-independent by its docstring) | one walk <= range tiles per triggering mate | -- | per (env, mate) a walk, then the rasteriser with an atomic compare-and-swap per cell (-1 -> uid, another uid -> -2): order-independent, so parallel is exact |
| B.5 | lof_solid (col 25) | per (row, seen enemy); full view per (row, every enemy), batched per row | `rays_clear(physical=True)` + every LIVE cover's `blocks_segment` | sum over rows of seen enemies | me -> enemy | `has_los(solid, nullptr)` |
| B.6 | the route (far_f[9], cols 92-94) | per row holding an order, cached per (graph, where, footprint) on `gmap._tile_serial` + the openings' states | not sight: `horde.flood` over walkable tiles OR the tiles of every OPENABLE opening (`room_graph.opening_states`: closed living doors, not wired, not hull doors), seeded at the passable anchors whose block centre lies in the place | one flood per (env, place, footprint) when the map or a door changed | -- | the device flood exists (`cuda_unit_horde.cuh::flood_block(tile_ok, h, w, fp, out)`, P7d's) and `is_hull_door` has a device form there; `opening_states`, the seeds, `horde.candidates` / `field_at` and the block-min are H |
| B.7 | the picture's object tracker | slot 11, per team per TICK | ONE `rays_clear` over every (mate centre, object tile) pair | M_t x live objects per tick | mate -> object | a pair kernel |

COVER. `vision.ray_clear` / `rays_clear` / `window_clear` apply every live cover entity with
`blocks_los` (`sim.cover`, a float slab test `blocks_segment`) in Python; `line_of_fire.solid_in_way`
applies EVERY live cover. The device has no cover entities. The conductor refuses only WEGO's
`march_cover` (`unit_schema._enc_no_cover`); I found NO refusal of a continuous-path level whose
`sim.cover` is non-empty (ship_demo's level.toml has no cover). A device observation needs that refusal
by name (or cover on the card).

Cost seen by the lead (ship_demo, CPU, per decision): windows 5.1 ms (B.1a / B.1b plus the float planes),
the full view 11.7 ms (B.1c over every enemy per in_match row, B.3 over every living unit).

---------------------------------------------------------------------------------------------

## C. The team picture and the last-seen records

`src/simulation/team_picture.py::TeamPicture`, started by `Simulation.start_team_picture` (from
`spawn_match`; it takes the FIRST update at once, at tick 0), then `update(sim)` ONCE per tick at slot
11 (`Simulation._step_post_physics`, after the tick advance and `on_tick_end`'s conversion). `update`
refuses a skipped or doubled tick; `records(team)` refuses a stale picture. Host memory: not synced,
not hashed, in no recording; rebuilt only by re-simulation.

| state | per | written by `update` from | device source today |
|---|---|---|---|
| `_records[team][uid]` = `Record(uid, team, zombie, loadout, tile, centre, pos_q, vel_q, tick, facing_q)` | marine team x seen unit | (1) delete when the unit's `UnitKilledEvent` is in this step's events AND the record was written at `tick - 1` (a SEEN death); (2) delete when `tick - rec.tick > memory_ticks`; (3) for every uid of `vision.visible_by_team[team]`: a new record from the LIVE unit, `vel_q` = pos minus the previous record's pos when that was `tick - 1`, else (0, 0) | units R (store); kill events R (ring, `KIND_KILLED`); vision H (B.2) |
| `motion[uid]` = (pos prev, pos now) | every living unit | live `x_q`, `y_q` | R (store) |
| `last_hit[uid]` = (tick, dir_x_q, dir_y_q, cause) | unit | the last `UnitHitEvent` with damage > 0, emission order | R: the ring carries `dir_x_q`, `dir_y_q`, `cause`, `damage_q` (`unit_events.FIELDS`) |
| `_damage[team][uid]`: deque of (tick, damage float) | team x unit | hits the team DEALT (attacker a living non-zombie of the team) or SAW (the victim's record at `tick - 1`, read BEFORE this update's record writes), kept `window_ticks` (= tps) | ring R (attacker_id, damage_q) |
| `bars[uid]` (float, exact n/65536) | unit | `current_hp` at the first update the unit exists, again at the first update that sees it `is_zombie` | R (store) |
| `death_tick[uid]` | unit | the first update seeing it dead or turned | R (store) |
| `_objects[team][obj] -> OBJECT_UID_BASE + k`, `_obj_next[team]`, `seen_now[team]` | team x live object | ONE `rays_clear` over (mate centre, object tile) pairs; a first sighting draws the team's next id; a dead object is forgotten; KEYED BY PYTHON OBJECT IDENTITY | grenades / charges R (`GRENADE_TABLE`, `CHARGE_TABLE`), but NO stable per-object identity: the grenade table is compacted by `prune_grenades`, and neither schema has a serial (`GRENADE_SCHEMA`: proj_type .. detonated; `CHARGE_SCHEMA`: x, y, det_tick, ammo_name, owner_id, approach, live) |

What the conductor does NOT run: slot 11 at all (`DeviceConductor._walk` ends with "11. the team
picture: host, a match only, in no digest (not run)"). The ring is begun at every tick's head
(`ring.begin()`), so a device picture (and the reward books) must consume each tick's events in that
tick, before the next `begin`.

What a device form needs (STATE ACROSS TICKS, not a function of the tick's state):
- per env, per marine team, a record table with a capacity (records are uncapped today: one per enemy
  unit, zombies included, so the cap is U) and `Record`'s columns (the loadout as its row index);
- the trackers as per-env arrays: motion (2 x 2 int64 per unit), last_hit (4 per unit), bars (Q16 per
  unit), death_tick, the damage window (per team per unit, `window_ticks` slots; the reference sums
  floats in deque order, so the sum's order must be the deque's), the object id map;
- a stable object identity for grenades and charges (a serial written at the append and carried by the
  compaction), so "the same object re-seen after a gap keeps its id" holds; a new column in
  `CHARGE_SCHEMA` reaches the digest's charge rows (hashed while a charge lives), so a side array beside
  the tables is the change that moves no digest;
- the update's ORDER inside the tick: the damage window (against the records BEFORE the update),
  seen-death deletion, memory expiry, new sightings, then objects; and its PLACE: after slot 10 and the
  conversion, i.e. a new last stage of `_walk`;
- the tick-0 update at episode start (no events).

It is per-team serial work over a few dozen units: one thread (or one warp) per (env, team) walking
units in id order is the natural exact form.

---------------------------------------------------------------------------------------------

## D. Masks

`action_spec_v1.build_masks(sim, units, list_type, list_mask, list_uid, tables, decimation, out)`, at the
START of the decision tick T (after tick T-1's step), ONE gate call per hand bit. Invalid rows
(`units[i] is None`) get `default_masks` (stand, "where I walk", release, nothing).

| mask | size | rule | gate (`verb_gates`) | device form |
|---|---|---|---|---|
| mask_feet | 49 | all 1 | -- | trivial |
| mask_trigger | 2 | all 1 | -- | trivial |
| mask_aim | 18 + E | `throw_release_tick >= T + decimation - 1`: only AIM_KEEP; else the 16 tokens and "walk" 1, KEEP 0, entry i = `list_mask[i]` and type in (1, 2) | -- (reads the list just built) | trivial once the list is on the card |
| hand[0] nothing | | always | -- | -- |
| hand[1] reload | | `can_reload(sim, u)` | `combat.mag_status` | R-fn `unit_lib::can_reload` |
| hand[2] melee | | `can_melee(u, T, wt, at_tick_start=True)` | | R-fn `unit_lib::can_melee(..., at_tick_start=true)` |
| hand[3] shove | | `can_shove(sim, u)` | `unit_impulse.shove_cooled` | R-fn `unit_lib::can_shove` |
| hand[4] use | | `can_use(sim, u)` = a live door whose span touches the footprint or its one-tile ring (`_reach_key`); WHICH door (`use_door`, #215) is not the mask's question | | the geometry lives inside `cpp/src/cuda_unit_verbs.cu::use_door`, a TU-local `__device__` function, NOT in the library: to lift into a header (the door-tile table is resident) |
| hand[5] plant | | `can_plant(sim, u)` | | R-fn `unit_lib::can_plant` |
| hand[6] flashlight | | always 0 (reserved) | -- | -- |
| hand[7 + k] throw kind k | K | `can_throw(sim, u, k)` | | R-fn `unit_lib::can_throw` |
| arg mask | 5 | `arg_mask(hand, aim)`, a pure function of the DRAWN hand and aim | -- | the trainer's (`agents/choosing.py`), not served |

All gates are integer. The library's forms take `(row, tab, tick, &bad)`; the mask kernel is one thread
per (env, row) at the decision tick's start. The masks also feed `masked_choices` and `check_actions`
on the host in `BatchedEnv.step` (they read the served masks; on the card they become CuPy reductions).

---------------------------------------------------------------------------------------------

## E. Reward and termination

`reward_v1.RewardBooks` (one per episode; `begin(sim)` before each tick, `end(sim)` after it,
`finish(result)` once), plus the env's OLD counters (`rl_env.count_events`), plus `MatchRules`.

### E.1 What the books read, per TICK

| read | source on the host | on the card? |
|---|---|---|
| per unit `(hp_q, alive, team, zombie, bar_q)` before and after the tick (`_snapshot`) | units + `picture.bars` | store R; bars H (C) |
| the tick's events in emission order: `UnitHitEvent(unit_id, attacker_id, damage float)` (`d_q = round(damage x 65536)`), `UnitKilledEvent(unit_id, attacker_id, killed_by)` | `sim.tick_events` | R: the ring holds exactly these, `damage_q` already Q16, `cause` as its index |
| the standing (`standing(units, bars)`: per team H, A; zombie H) before and after | units + bars | store R; bars H |
| the living marine teams (`MatchRules.living_teams`) | units | R (a per-env reduction over the store) |
| the RESULT at the end (`MatchRules.result`) | units, tick | R |

The RewardBooksMismatch check (the events must reproduce every living unit's hp and life) reads the
same; on the card it is a counted per-env fault raised at the books read.

### E.2 The terms of profile v1 (`PROFILES["fireteam"]`; "range" has the same terms, zombie / ff at 0)

| counter (`BOOK_COUNTERS`, after the old ones) | group, term, sign (`_COUNTER_TERMS`) | built from |
|---|---|---|
| won / eliminated / clock | outcome: win +10, eliminated -10, clock -3 | `finish(result)`; `_eliminations` (a team's last living marine gone) |
| bars_enemy, kills_enemy | own: marine_bar +1, marine_alive +0.5 | hits / kills credited to the row's unit on an enemy MARINE (`_settle`) |
| bars_lost, died | own: marine_bar -1, marine_alive -0.5 | the row's own unit hit / killed |
| bars_zombie | own: zombie_bar +0.25 | hits credited on a zombie |
| share_bars, share_alive, share_enemy_bars, share_enemy_alive, share_zombie_bars | team (signs per term), each `/ n_t` | the CHANGE of the standing read from state (`end`) |
| bars_team, team_kills (an old counter) | friendly_fire: ff_bar -2, ff_kill -3 | hits / kills on a team-mate |
| (order) | all zero (6.7) | -- |

Per team also `d_balance`, `rest` (the uncredited rest, BUILT from its sources), `outcome`; per row
`rows_ended`. The OLD counters (`count_events`): hits_dealt, damage_dealt (float32 accumulation of
`float(ev.damage)`), team_hits, kills, team_kills, hits_taken, damage_taken, deaths_<cause> per
`events.CAUSE_NAMES`: the same two event kinds and the team-at-tick-start map.

Arithmetic: float64 in the books (`bar_ratio = min(max(hp_q, 0), bar_q) / bar_q`, sums, products by the
term weights), accumulated in EVENT ORDER within a tick and tick by tick over the decision; the env's
group numbers are `(counters[:, :, None] * G[None]).sum(axis=1)` (a numpy reduction over K counters)
cast to float32. Every operation is IEEE (+, -, *, /, min, max): exact on the card IF every sum keeps the
reference's order (numpy's pairwise summation in that last reduction included: K is small, so its order
is to be checked, K) and FMA contraction is off.

### E.3 Do the events exist on the card?

Yes, for the two kinds the books and the picture read: the event ring (`unit_events.EventRing`, `(N,
E_max, F)`, in the reference's emission order, hits and kills, with attacker, cause, Q16 damage and the
direction). The claims list (`attack_lists.claims_list`) is the CREDIT input of `apply_packet`, already
folded into each event's `attacker_id`: the books need no claims. Not on the ring: the swarm's death
events (`SwarmUnitDiedEvent`; no reward or picture rule reads them) and render-only events.

### E.4 `terminated`, truncation, the winner

| output | rule | where | device |
|---|---|---|---|
| `terminated` (N,) | named profile: every end (`len(living_teams) < 2` OR `tick >= limit_ticks`); `"counters"`: the elimination only | `EnvRunner.step` after `MatchRules.is_terminal` | a per-env reduction over the store + the tick row; `tab:match_limit_ticks` R |
| `truncated` | False under a named profile; the limit under "counters" | same | same |
| winner / `result` | `MatchRules.result` | same | same |
| `rows_ended` | a team eliminated, or the episode over | `RewardBooks._flush` | part of the books |

A decimation step stops early when the match ends (`if s.is_terminal(): break` in `EnvRunner.step`). On
the card an env that ended mid-step must stop its observation-side state (the books finish, no more
counters, the terminal observation taken at that tick) while the conductor keeps ticking it (P7e.5: an
ended env ticks on until replaced). The books and the picture need a per-env "ended at tick t" latch,
and the terminal observation must be built AT the ending tick, not at the step's end.

---------------------------------------------------------------------------------------------
