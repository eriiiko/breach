# Lens 3: systems and buildability (critic, 2026-11-11)

Read against `docs/rl_agent_design_2026-11-11.md` at `5c422d05`, the roadmap's
sections 3, 4 and 7, `CLAUDE.md`'s canonical-systems tables and the code below.
Lenses 1 and 2 were not read.

## Verdict

**Survives with fixes. 0 blockers, 7 major, 11 minor.** Every verb, gate and
field the design names exists, and the #193 reading in sections 2, 3 and 9 is
correct, with two exceptions noted under M1 and M6 (the plant's own gate is
clean and can be lifted as it stands). Nothing a network computes reaches
synced state, because every choice goes out as an existing verb through the one tap.
The faults are in five places. (a) The masks of 2.4 paraphrase the verbs'
gates, and in places the paraphrase is wrong. (b) Two things are written as if
they already lived somewhere: the room graph that is "kept current", and the
sight setting per match. Neither has a home or a rung-1 patch. (c) The order
verb has consequences for the policy schema, the replay format and the digest
that the plan does not state. (d) Section 9 is short of what P3a needs. (e) The
cost claims of 3.7 hold only if two pieces are built in C++. Each fix is a
paragraph on paper. None needs new machinery.

## Findings

### MAJOR

**M1. The masks of 2.4 do not match the verbs' real gates, and the gates are not all pure.**
2.4 gives one general rule for every hand verb ("can act, not knocked down,
not winding up a throw") and then adds a condition per verb. The code gates each
verb differently:
- reload: `Simulation._reload` (simulation.py:1224) checks only the magazine. It does not check `can_act` or a wind-up.
- use: `_direct_use` (:1463) has no gate at all. It toggles the first live door in the footprint's one-tile ring, so a stunned unit can open a door.
- shove: `can_act` and its cooldown only (`_consume_direct_intents` :1314, `unit_impulse.shove` :896).
- melee: `can_act`, no wind-up, no plant, and the row's cadence (`combat.process_melee_intents` :1971-1977). It also needs a target in the arc, judged at slot 4 against the facing after the turn and the positions after the move (`melee_arc_target` :1913).
- throw and plant: the gates in `_start_throw` (:1335-1342) and `_start_plant` (:1387-1392).

Three more problems:
- The reload gate calls `mag_settle` (combat.py:747), which writes the unit's magazine state. A mask that calls it would mutate the game from the environment, against the facade's rule ("outside code reads `get_state()`").
- Every gate runs at consumption: slot 1b, after slot 1a, which may knock the unit down; or slot 4. A mask made at decision time is therefore an estimate one slot early. A choice the mask allows can still do nothing.
- A latch goes out on a decision's first tick only (`control_policy.py:263`). A cooldown that ends on tick 2 of a decision loses that verb for the whole decision.

*Fix:* replace the bullets of 2.4 with a table, one row per verb, copied from
the code above. Lift each verb's gate as a pure predicate taking `(sim, u,
tick)`. For reload, use the non-mutating `mag_status` form (combat.py:834):
the verb first settles the magazine, then calls the predicate. The melee and
shove predicates cover eligibility and cooldown only. "No target in reach" is
not masked; it is an ACCEPTED GAP, and a whiff costs nothing (:1961, :894).
Narrow the gate of 2.7 / rung 1 to "a masked choice issued by force does
nothing". Record "an unmasked choice may do nothing" as accepted.

**M2. A throw's wind-up spans two decisions, and the second decision steers it.**
- The wind-up is 7 ticks (`throw_windup_seconds` 0.3). The grenade leaves along the facing the unit has at release (`_advance_throw` :1373-1377).
- With a decision every 4 ticks, the facing at release belongs to the next decision's aim choice. "Onto my aim target" fixes its reach on the issuing tick (`throw_reach_q`, :1355), while the target keeps moving.
- The bots solve this by holding their aim on the lob through the wind-up (CLAUDE.md, Grenade kinds row). 2.1's line "a throw goes where the soldier aims" hides the problem. It also splits the credit for where the grenade lands across two choices.

*Fix (decide one on paper):*
- (a) While `throw_release_tick >= 0`, the aim head is masked to a single "keep the throw's aim" option. The decoder keeps aiming at the throw's target or direction, as the bots do. Recommended.
- (b) Accept the split and state it.

The same question arises for the plant (24 ticks). It does no harm there, because the plant lands at the feet.

**M3. Sight per match has no home, and changing it "outright" would break a frozen pin.**
- The cone is global config: `CFG.onephase.vision_cone_half_deg` and `awareness_radius_tiles` (vision.py:156-160, config.toml:55-56).
- `control_bots_v1` reads the cone (control_bots_v1.py:212), and its match is pinned "never re-pinned for a bot change" (tests/test_bots_v1.py:352-375).
- Erik allowed the outright change in 3.6 because no earlier match needs preserving. He did not consider this frozen pin, and the outright change would break it.
- The `Match config` row says match.toml is "the ONLY way a match is parameterised". Section 11 puts the sight setting in rung 1, but none of patches 1a to 1e builds it.
- `VisionState` is cached by tick alone (vision.py:197), so the rule must be part of the simulation, not of a caller.

*Fix:* a match.toml key such as `sight = "cone" | "all_round"`, defaulting to
today's cone, written by `level_lib.write_match` and read through
`parse_match`. The simulation holds it, and `vision.can_see` reads it. Training
run files choose all-round sight. Name it in patch 1a. Its gate: every existing
match, including bots_v1's pin, stays bit for bit.

**M4. The order verb (5.7, patch 1d) has consequences the plan does not state.**
- `INTENT_KINDS` (match_replay.py:104) feeds the policy's generated action row (`control_policy.POLICY_KINDS`, :79). A new order kind therefore becomes an action column at once: every soldier could give itself an order, and the row's width changes. The order kind must join `EXCLUDED_KINDS` (:77), as `drive_zombies` does.
- `FORMAT_VERSION` goes to 8.
- `test_bots.py::test_the_bots_change_nothing_but_intents` (:524) requires that a controller's `handle_frame` changes no digest. So the order verb must latch and be applied inside `step`. It must not write unit state directly, as `set_flashlight` does (:1222).
- If the order is hashed (it is unit state, 5.7), then from patch 1d on every match in which the bots' team layer writes orders hashes differently. "These patches change no behaviour" (roadmap rung 1) holds for play, but not for digests.

*Fix:* write all four points into 5.7 and 1d. Make 1d's gate a deliberate
re-pin with written rationale (or the order hashed absence-transparently, and
written only for teams that have a policy). Both satisfy the digest rule.

**M5. The room graph: never saved, nowhere to be "kept current", and missing from Systems.**
- gen_ship's `ShipGraph` rooms are rectangles: `Room` with `x, y, w, h`, kind and tags, and `Edge` with `a, b, kind, at` (tools/gen_ship.py:696-729).
- Today `level_lib` writes only `[[spawn_room]]` (level_lib.py:101-170). `[levelgen]` holds counts only (level_loader.py:1488). Version-1 levels and hand-made levels have no graph.
- 5.7's "keep the graph current when a wall is destroyed or a door sealed" names no place. The topology already has one change counter, `GameMap._tile_serial`. The bots key off it (`control_bots.py:1478`), and so does the horde field.

*Fix:*
- The graph is static level data: `[[room]]` and `[[opening]]` managed families plus a room-id grid file, written only by gen_ship through `level_lib` and read through a loader door. Only an opening whose kind can be walked is something an order may name.
- The live edges are derived, not stored: from the static grid, the current `solid` mask and the doors' states, cached by `_tile_serial`. It lives on the environment's side and is neither synced nor unit state. An opening's id is derived (room a, room b, first tile in row-major order), never a counter, so a recorded order replays.
- A level that has no graph is refused at the training doors.
- Add it as its own rung-1 patch, before 1b and 1d. Add a new canonical system "Room graph" to section 10 and roadmap 7, with the draft rule: one map from tile to room; orders, the commander, and the bots' room goals (`room_goal`, :1539) all read it.

**M6. Section 9 is incomplete for P3a.**
P3a's property test makes every attribute a schema row or REFERENCE_ONLY
(residency design, the P3a row). The observation (3.2, 3.3, 4a, 7.1), the team
display (4.2) and the masks read many attributes that exist today but are not
in the unit digest (`field_ab_harness._unit_record`, :180-279). If P3a marks any
of them host-only, the device observation is cut off. Section 9 lists none of
them. The full list is in the table below. Section 9 also has one row wrong:
last-seen records and "damage in the last second" cannot be derived from
"state plus the tick's events". They are memory carried by the environment over
the last 6 s (`TeamBrain.observe`, control_bots.py:325-337). They must be updated
every tick, not every decision, and they cannot be rebuilt from a snapshot or
after a seek within a replay.

**M7. The cost claims of 3.7 hold only with two pieces built in C++.**
Estimates, not measurements:
- The farm's budget: about 1 400 ticks a second across 16 workers (UNVERIFIED) is about 11 ms per tick per worker.
- `vision.compute` is O(U²). Each test is a pure-Python Bresenham (`GameMap.has_los`, gamemap.py:2436), and the zombies count as observers (vision.py:175-185).
- All-round sight removes the cone's pruning. On `ship_demo` (12 units) that is about 96 rays a tick at about 40 µs each, so about 4 ms a tick. That is roughly 3 times today's cost, and it is paid every tick.
- The "watched" plane in Python would cost 8 enemies × about 1 000 tiles × about 30 µs, so about 240 ms per decision. That is unusable.
- The line-of-fire flags (about 32 rays per team per decision) and the turn-time of 3.3 for about 16 × 4 candidates are a few ms. Acceptable.
- Building the far window per soldier means about 101 × 101 × 18 values each. Reducing the map to blocks once and then cropping costs about 1/16 of that.

*Fix:*
- 3.7 states three requirements:
  - The sight routine is a C++ batch form of `vision.ray_clear`: the same Bresenham, in the same direction (observer to tile), including `sight_screen` and the cover that `blocks_los`. A different algorithm (shadowcasting) would be a second sight model.
  - `vision.compute` skips observers that nobody reads (the zombies), or moves to that same batch routine.
  - `EntityFieldAccessor.crop` gains a block-reduction argument.
- 1b measures each of these against today's farm.

### MINOR

- **m1. The fire-lane plane (4.5) is built from controller memory.** `TeamBrain._lanes` rasterises `self.firing` (control_bots.py:445-472), which contradicts 4.1. *Fix:* lift the rasteriser so that it takes (shooter, target) pairs, and feed it the pairs that 4.1's rule produces.
- **m2. Where the decoder lives.** `AgentPolicy` issues one fixed row for a whole decision (`issue_row`, :257). Calling `set_rows` on every tick would re-fire the latches (:307). Following an entity on each tick therefore needs a per-tick aim hook inside `AgentPolicy`. It must not be a second controller, because the MatchSession row says a policy drives units only as this controller. Build the aim vector with `control_source.quantize_point_aim`. The throw's latch gate is `vector` (:88), so a throw needs a non-zero `dx_q, dy_q` even when the aim is "where I walk". Pass the facing.
- **m3. Wind planes (3.4).** Read them through the Wind-load row (`footprint_wind_load` / `footprint_wind`: "never read the planes"), not from `wind_x` / `wind_y`. Scale "the push that knocks this unit down" by the `ImpulseTable` limits × `stability`.
- **m4. Blast reach (3.3 "whether I stand inside that reach", 4.3 "a frag thrown at it would catch").** Use `blast_expansion.reach_tiles`, or declare a radius approximation as an ACCEPTED GAP.
- **m5. "How long until my body could be on it" (3.3).** It becomes a pure function in `turning.py` that iterates `law_q16` (:140). It must not be a copy of the law.
- **m6. Wording.**
  - The "kind's default fuse" (2.2) is the single `[weapons.hand_grenade] fuse_default_seconds` (config.toml:3298), not one fuse per kind.
  - The plant's four timers come from `[weapons.c4_satchel]` (`PLANT_WEAPON`, simulation.py:1118), not from "the charge's table row".
  - The reach is a fraction of the one `max_throw_range` (:1444).
- **m7. Systems lists.**
  - Roadmap 7 names "Timed charge". The plant is the **Planted charge** row, which says "never a timed_charge entity".
  - Section 10 / roadmap 7 omit several systems the design must use: Match config (M3), Level data layer and gen_ship / LevelSpec (M5), Vision `ray_clear` (M7), `horde.flood` by name (3.5), Blast reach (m4), Wind load (m3), Throttle / Speed table (the speed levels), and `weapons.fuse_ticks_from_seconds`.
  - The new canonical systems the design creates are not flagged: the Room graph (M5), the sight routine (M7) and the verb-gate predicates (M1).
- **m8. The order of the rung-1 patches.** 1b puts the order and the compass into the observation before 1d defines them. Either put 1d (and M5) first, or let 1b reserve a block of zeros. 1a's gate speaks of "the bots' pinned matches", but `bots` has none today; only `bots_v1` has a frozen pin. Take an A/B pin of `bots` before the patch. Lifting the gates (M1) is a refactor of the simulation and needs the A/B harness's digest lockstep as its proof.
- **m9. The planted-charge table for P3a.** `PlantedCharge` holds `ammo_name` as a string, `det_tick` possibly `None` (OnePhaseWEGO's remote charge), and `approach` possibly `None` (charges.py:37-51). The table needs integer encodings: an ammo-row index, sentinel values, and a capacity.
- **m10. "Whether it is firing" (4.2).** It must read `live_trigger`. `ShotFiredEvent` is render-only and outside the digest (Frame lights row).
- **m11. 3.7's "the per-tick sight cache the game already keeps".** The cache is lazy (vision.py:193-202), so in a match between policies it is built only when the environment asks for it. Say that the environment builds it every tick (M6, M7).

## Section 9 as it should read for P3a

| state | where today | in the digest today | P3a |
|---|---|---|---|
| charges carried `has_explosive`, `charge_ammo` | unit.py:294-301 | no | column; `charge_ammo` as an ammo-row index |
| a plant under way `plant_done_tick`, `plant_fuse_ticks` | :303-304 | while planting | column (as written) |
| planted charges | `sim.planted_charges` | while one lives | own table (m9) |
| magazine `current_mag`, `spare_mags`, `reload_done_tick` | :344-346 | no (mag_gate's note: "P3a hashes it") | columns |
| grenades by kind `grenade_counts` | :283 | no | fixed block |
| throw wind-up `throw_release_tick`, `_kind`, `_fuse_ticks`, `_reach_q` | :290-293 | no | columns |
| cooldowns `last_blow_tick`, `last_shove_tick`, `last_fire_tick` | :358, unit_impulse, :~318 | no | columns |
| held intents `live_move_dir` (dx, dy, throttle), `live_trigger`, `live_aim` | intents | no | columns (the team display reads them) |
| zombie awake `zombie_activated` | ai_zombie.py:14,237 | **no** | column (4a: dormant / hunting) |
| kit `weapon_id`, `melee_id`, `loadout_name`, class | unit.py | no (static per unit) | table indices |
| the order: what, where id, until tick, target id | — (1d) | — | four columns |
| last-seen records, last hit, damage over the last second, velocity | env memory | — | not unit state: carried by the environment and updated every tick |
| room graph live edges | — (M5) | — | not unit state: derived from `_tile_serial` |
| sight rule | — (M3) | — | match state, a scalar per env |

## Checked and sound

- The plant verb's clamp, its latch and the `nonzero` gate (simulation.py:1094-1129, control_policy.py:98). The plant cancels throws and blows in both directions (:1337, combat.py:1972). The plant cancels on stun or death (:1408, :1273). The charge lands on the centre tile at the end and is credited to the owner (:1414-1419). `charges.tick`. `_charge_records`.
- The hold and latch semantics and `decimation` (rl_env.py:544). Per-tick events reach `_count` (:548), so the team picture and the reward can read every tick.
- `facing_after_turn_q16` is the bots' predictor (:1024). The throttle clamp. `set_aim` sets a target only.
- `MatchRules.result` and `living_teams` (ruleset.py:335-344), and `stats.effective_vitality` (stats.py:55), exist.
- The crop dequantises each plane once per call (sensor_accessor.py:292). The far and near windows can extend it.
- The reward, the full view and the networks stay outside the synced game. Replays record the verbs, so no float a decoder uses can reach a digest.

## Not checked

- The farm's breakdown per worker. The cost of `reach_tiles`. Whether the carbine's round really crosses its range within one tick (2.2).
- The environment profile's poison, tear-gas and fuel-vapour thresholds (lens 2's ground).
