# CLAUDE.md restructure — ledger (#206, 2026-11-15)

Every row of the old CLAUDE.md (`git show deade9d2:CLAUDE.md`), in its order: where its
constraints went and what was dropped. Bytes: the old row; the new index line; the rule
section (0 = index line only). Dropped kinds: **H** provenance (patch labels, "since", as-built),
**B** inventory of what is built, **G** gate results / measurements / timings, **C** what the
code states (names, signatures, values, formulas, limits), **T** test-file lists beyond the gate,
**D** another row's rule restated (kept once, in its owner), **P** PLACEHOLDER note (checked: in
config.toml beside the value), **N** nothing but wording. Kept everywhere: who must use it,
never-build-a-second, change-together, rulings with dates, accepted gaps, debts, incidents.

| # | Row | Old | Index | Rule file | Rule | Dropped → lives in |
|---|---|---|---|---|---|---|
| 1 | Simulation facade | 158 | 158 | — | 0 | N |
| 2 | Tick conductor | 1078 | 174 | sim-core.md | 894 | C (slot-by-slot call names); the slot detail of 1a/3/5/9e/11 kept in sim-core.md |
| 3 | GameMap | 122 | 122 | — | 0 | N |
| 4 | FieldEdit | 1657 | 214 | fields-gas.md | 974 | H (#29 P3e/P4c labels); C (weight/rounding formulas, `take_ordered`/`pack_for_device` call chain) -> field_edit.py, cuda_edit_queue.cu |
| 5 | Payload executor | 733 | 180 | explosions-fire.md | 377 | C (unit of bulk amount, flat disc); P (`o2_store_rupture`, `halon_dump`: config.toml) |
| 6 | Generic explosion | 1513 | 206 | explosions-fire.md | 825 | H; C (flash column semantics) -> payload door; P (strengths: config.toml `[blast]`); D (Flash blinding columns: its own row) |
| 7 | Divergence source | 946 | 172 | explosions-fire.md | 788 | N beyond wording |
| 8 | Blast reach | 746 | 117 | explosions-fire.md | 552 | C (`open_mask`, `walk_seeds` names) |
| 9 | Blast air read | 351 | 157 | — | 0 | P (`[blast_air] smoke_scale` look dial: config.toml says it, with Erik's date) |
| 10 | Fixed-total tile deposit | 611 | 190 | explosions-fire.md | 388 | H (#29 P3e) |
| 11 | Starting a fire | 1371 | 231 | explosions-fire.md | 904 | C (the `hotf` formula, `k_die` e-fold, `IGNITE_HEAT`) -> payloads.py, the fire law |
| 12 | O2 pressure factor | 582 | 162 | fields-gas.md | 406 | C (edges `p_ext_atm`/`p_full_atm`, the 0 / 1 behaviour) -> o2_pressure_factor.h |
| 13 | Gas pump primitives | 1122 | 205 | fields-gas.md | 953 | N beyond wording |
| 14 | Vent / duct system | 346 | 183 | fields-gas.md | 210 | N |
| 15 | Trace transport | 1306 | 205 | fields-gas.md | 976 | N beyond wording |
| 16 | PhysicsRunner / PhysicsEngine | 189 | 189 | sim-core.md | 148 | N |
| 17 | Fixed-point kits | 1098 | 203 | fields-gas.md | 526 | H (P6.2/P6.2m/P6.4a); B, C (function list `isqrt_q`, `div_round_q`, `half_up_q`, `isqrt_u128_q` ...) -> fixed_point.h, unit_fixed.py |
| 18 | Q16 boundary modules | 121 | 109 | — | 0 | N |
| 19 | Trace currency | 933 | 218 | fields-gas.md | 765 | N beyond wording |
| 20 | Config | 120 | 120 | — | 0 | N |
| 21 | Material / gas tables | 635 | 160 | fields-gas.md | 545 | N |
| 22 | Filter table | 196 | 121 | — | 0 | N |
| 23 | Temperature scale | 118 | 118 | — | 0 | N |
| 24 | Coupling table | 598 | 159 | units.md | 304 | H (#195 1g (b), G5p-b); G (`match_pin_check` EQUAL across the extraction) |
| 25 | Wind load on a body | 2364 | 219 | units.md | 1143 | H (#140, G5p-b); C (drag area, k_slow, 39 / 245 Pa, ~2.6 kPa prone) -> unit_impulse.py, config.toml `[impulse]`; T (test_wave_push, test_wind_doorway_e2e) |
| 26 | Footprint samples builder | 1156 | 142 | units.md | 633 | H (G5p-a2, U1); C (the member list) |
| 27 | Recorder | 303 | 109 | sim-core.md | 340 | N |
| 28 | Entity system | 235 | 214 | — | 0 | N |
| 29 | Prop tile stamps | 369 | 232 | — | 0 | N (fits its index line) |
| 30 | Level data layer | 1427 | 144 | levels.md | 1026 | H (#133 E.2, #195 1b); C (`LevelData.rooms` shape, `[[room]]`/`[[opening]]` columns) -> level_lib.py, level_loader.py |
| 31 | Interior drag | 568 | 195 | fields-gas.md | 418 | N beyond wording |
| 32 | Gas energy field | 600 | 154 | energy-books.md | 466 | N beyond wording |
| 33 | Gas energy seam | 589 | 145 | energy-books.md | 453 | N beyond wording |
| 34 | Gas temperature is a mirror | 529 | 171 | energy-books.md | 422 | N |
| 35 | Face-flux energy step | 471 | 150 | energy-books.md | 382 | N |
| 36 | Gas energy recovery | 301 | 133 | energy-books.md | 271 | N |
| 37 | Energy closure identity | 2790 | 198 | energy-books.md | 1521 | H (P-G1b, P-G5, P5c, T5b); C (CUDA slot numbers 12-16, per-counter definitions) -> temperature_solver.*, cuda_temperature.cu; report_t5b.md |
| 38 | Temperature solver | 2579 | 207 | energy-books.md | 1535 | H (T5b, P5c, P5d); C (CUDA slot numbers, the clamp formula) -> temperature_solver.cpp |
| 39 | Radiation sweep | 7968 | 212 | radiation-light.md | 3161 | H (T5b, P4, P5b-P6d); C (Fleck expression, light absorb/emit/glow, cone row layout, door limits with #173's per-ordinate charge and the 16-ONE headroom argument) -> radiation_sweep.h; G (shipped-scale f == 2^24, held by its test); T |
| 40 | Emissive table | 2585 | 165 | radiation-light.md | 1127 | H (P5d, #78, P6c); G (#78's dormancy measurement); C (12-trip inverse, 15996 saturation, `E_TABLE_TOP_MAX`, `L_FINE_BITS` = 37 and its derivation) -> emissive_table.h |
| 41 | Extinction planes | 2756 | 207 | radiation-light.md | 1518 | H (P5a, P5c, P6a, P6c, T5b); C (the a_gas formula) -> the sweep pre-pass |
| 42 | Integer reference | 1426 | 170 | radiation-light.md | 570 | B (gate list per phase: gates 18, 19, #88's) |
| 43 | Light accessor | 585 | 132 | radiation-light.md | 351 | N |
| 44 | Stamp kernel | 1728 | 160 | device-residency.md | 870 | H (#29 P4b, P5); C (the `(R, 9)` row layout, pinned async upload, baseline uploads) -> cuda_stamp.cu |
| 45 | Batched resident engine | 2225 | 238 | device-residency.md | 1618 | H (#29 P5, design 4.2); C (`EnvSub`, `bptr` details) |
| 46 | Unit store | 5472 | 261 | device-units.md | 2024 | H (P6.1-P6.4a); B (built and chained slots: the twin manifest); C (INPUTS vocabulary, `run(store, inputs)`, count semantics) -> cuda_unit_lockstep.py, unit_store.py; T |
| 47 | Device tables | 1744 | 227 | device-units.md | 761 | H (P6.2); C (`ROWS`, `TableRow`, `values_batch`, header macro names) |
| 48 | Device unit library | 2970 | 199 | device-units.md | 968 | H (P6.2-P6.4a); B, C (function inventory `unit_row`, `row_alive`, `sq2`, `MobilityMap`, the gate list ...) -> cuda_unit_lib.cuh |
| 49 | Event ring | 2540 | 160 | device-units.md | 1037 | C (record fields, `row_cap()`, `E_max` 1024) -> unit_events.py; H |
| 50 | Request list | 2501 | 166 | device-units.md | 976 | C (record fields, `R_MAX` 256) -> unit_requests.py; H |
| 51 | Field digest | 435 | 150 | determinism-gates.md | 358 | N |
| 52 | GOLDEN_AGGREGATE | 131 | 131 | — | 0 | N |
| 53 | A/B lockstep harness | 124 | 121 | — | 0 | N |
| 54 | CUDA harness | 166 | 166 | — | 0 | N |
| 55 | Build id | 523 | 173 | device-residency.md | 297 | C (git SHA + `-dirty`, the generated file path) |
| 56 | Class P / Class T + the twin ledger | 2743 | 211 | device-residency.md | 1417 | H, B (the Class-T row list by patch: the manifest holds it) |
| 57 | Ingress lint + float ratchet | 185 | 191 | — | 0 | N |
| 58 | Test conventions | 346 | 123 | determinism-gates.md | 538 | N; the fixture lines (temperature mirror, heat-first fire) added as cross-references |
| 59 | GameRenderer + WorldComposite | 227 | 174 | — | 0 | N |
| 60 | One tone-map | 1489 | 208 | render-core.md | 971 | C (`aces_curve`'s other caller `glow_share`); G (F3's +51..+142 % luminance, 17 deg hue: design 13.8) |
| 61 | Look shots | 368 | 157 | render-core.md | 269 | N |
| 62 | Wall-destruction render seam | 1034 | 167 | render-core.md | 676 | C (drain chain `render_dirty` -> `TileArtSeam` -> `_apply_tile_patches`) -> tile_art.py |
| 63 | LightingPass | 566 | 140 | render-core.md | 291 | H (P6b, P6c); C (`consume_light_view`, `pack_light_view`) |
| 64 | Gas medium | 556 | 123 | render-effects.md | 432 | N beyond wording |
| 65 | No BLAS in the frame path | 656 | 215 | render-core.md | 427 | N (incident kept) |
| 66 | Smoke relief | 948 | 152 | render-effects.md | 630 | C (the look's term list: coverage, height, N.L, AO ...) |
| 67 | Billow noise bake | 386 | 134 | — | 0 | N (fits its index line) |
| 68 | Smoke cast product | 1390 | 173 | render-effects.md | 818 | C (RG/BA layout, 8 px/tile, `u_mode` 1, occlusion slot); H |
| 69 | Tamed wind (render) | 1254 | 139 | render-effects.md | 744 | C (3x3 smoothing, `v_ref`/`v_sens`) |
| 70 | Blackbody | 366 | 115 | render-core.md | 182 | N |
| 71 | Blackbody GPU LUT | 422 | 141 | render-core.md | 190 | C (1024 entries, sampling formula) |
| 72 | Ping-pong helper | 875 | 165 | render-core.md | 492 | G (~5 / ~3 us vs ~40 / ~10-30 per pass: kept as 'far cheaper') |
| 73 | FireVFX pass | 2500 | 164 | render-effects.md | 1083 | C (the loop's stage list, bicubic reconstruction, flow-map details, debug views); H (F3) |
| 74 | Particle primitive | 3263 | 143 | render-effects.md | 956 | B (consumer numbering); C (spawn rates, `(n, 8)` layout, substep rule, seeding) -> particles.py |
| 75 | Resolve distortion block | 1320 | 150 | render-effects.md | 470 | C (shimmer's fBm / flow-map composition, `t0_k`, `zoom_zero`); G (0.05 / ~0.21 ms) |
| 76 | Shock front | 1358 | 132 | render-effects.md | 475 | C (RGBA plane packing, Gaussian band, `UNREACHED`) |
| 77 | Muzzle flame | 2103 | 166 | render-effects.md | 762 | C (jet generator seeding, `size`, `flame_seen`) |
| 78 | Tracers | 1970 | 127 | render-effects.md | 678 | C (`shutter_s`, halo, afterglow, `stopped_on_wall`, `impacts_last_frame`) |
| 79 | Fire room light (gain + flicker) | 1256 | 181 | render-effects.md | 635 | C (`1 + a*clip(z/2)`, `puff_share`, the |rho| < 0.3 gate) -> fire_flicker.py, test_fire_flicker.py |
| 80 | Frame lights | 1967 | 111 | render-core.md | 670 | C (`muzzle_flash_lights`, `blast_flash_lights` formulas, 1 % drop, the 1024 value) |
| 81 | Light curve | 1052 | 135 | render-core.md | 528 | C (256 floats, sqrt-encoded axes, cubic with corners; Ctrl+R; `write_settings`) |
| 82 | Dequantize convention | 164 | 164 | — | 0 | N |
| 83 | Tile inspector | 552 | 151 | render-core.md | 285 | N beyond wording |
| 84 | 3D units | 3446 | 197 | render-units.md | 1682 | C (shading key list, `HandAssigner`'s dealing formula, shadow keys); H |
| 85 | Held props | 1937 | 143 | render-units.md | 717 | C (row keys `length_m`, `grip`, `gloss`, `grips.*`; the mirrored-grip formula) |
| 86 | Pose layer | 7744 | 180 | render-units.md | 2532 | H (M13, #162-#187, D1); C (per-skeleton table keys in detail, twist / pivot / gait mechanics, SOMA clip names, rate 0.51); state ('ON since 2026-11-07, unplayed until #165': the issue holds it) |
| 87 | Lit-3D seam | 428 | 135 | — | 0 | N |
| 88 | Prop generator | 256 | 132 | — | 0 | N |
| 89 | Static props | 529 | 124 | render-units.md | 358 | C (owned-memory model cache, shared `begin_mode_3d` pass) |
| 90 | UI split | 421 | 109 | render-core.md | 282 | N |
| 91 | Sound layer | 1677 | 164 | audio.md | 939 | C (`choke = "emitter"`, `Start`/`Stop` details) |
| 92 | Sound samples | 558 | 183 | audio.md | 294 | C (sound_clips.py function list) |
| 93 | Unit | 2160 | 127 | units.md | 1333 | H (#29 P3a/P3c, U1, #195 1f); D (impulse, order, traits details: their rows) |
| 94 | Unit schema | 1634 | 143 | units.md | 816 | C (sentinel and capacity names, `UNIT_SECT_V2` format, P3e column notes); D (traits block: Unit traits row) |
| 95 | Unit traits | 2464 | 153 | units.md | 1296 | C (`HOOKS` order list, accessor list); H (#29 PX TOX hooks) |
| 96 | Weapons | 1801 | 203 | combat.md | 817 | D (magazine, grenade, exposure, impulse, melee details: their rows; hooks: Unit traits) |
| 97 | Match draws | 1915 | 176 | combat.md | 915 | C (counter per site: shot cone, exposure / crit rolls, beam, melee crit, field noise; `tick_base`) -> draws.py |
| 98 | Exposure | 1784 | 129 | combat.md | 503 | C (raise / decay / scale formulas, the bisection of `decay`); P (config.toml `[damage.exposure]`) |
| 99 | Poison | 2737 | 126 | combat.md | 1098 | C (per-tick fractions, `X_MAX`, the 0.6 x 10 HP equivalence); P (4 s / 20 s / 10 HP: config.toml `[damage.poison]`) |
| 100 | Magazines | 2616 | 170 | hand-verbs.md | 857 | H (G2p, G1p-c, #195 1e); C (`SPARES_UNTRACKED`, `MAG_UNBOUND`, the event's fields) |
| 101 | Grenade kinds | 2583 | 164 | hand-verbs.md | 789 | C (count syntax, `throw_grenade_intent` signature, key 4 / pad B); P (0.3 s wind-up: config.toml) |
| 102 | Flash blinding | 1114 | 174 | hand-verbs.md | 459 | C (the f / g formulas, 60 deg); P (12 tiles, 4 s, 0.25: config.toml) |
| 103 | Melee verb | 2566 | 154 | hand-verbs.md | 753 | C (keys, `MeleeEvent` fields, cadence columns); P (rifle_butt's five values: config.toml) |
| 104 | Flashlight bit | 1157 | 152 | hand-verbs.md | 377 | C (key Q / D-pad, `DirectControl.flashlight_off`) |
| 105 | Simultaneous attacks | 1548 | 157 | combat.md | 837 | N beyond wording |
| 106 | Credited cause | 1856 | 183 | combat.md | 949 | N beyond wording |
| 107 | Loadouts / suits | 1069 | 163 | combat.md | 263 | C (column list) -> config.toml `[loadouts.*]`, loadouts.py |
| 108 | Control seams | 892 | 230 | match.md | 454 | N |
| 109 | Direct control | 2142 | 165 | match.md | 662 | B (which verbs went live in which patch, G1p-#193); C (the key per action) -> config.toml `[controls.keys]` |
| 110 | Camera mode | 569 | 167 | — | 0 | N (fits its index line) |
| 111 | Throttle | 965 | 179 | units.md | 412 | C (pad deflection, keyboard 0 / ONE, bots at 0) |
| 112 | Speed table | 2069 | 186 | units.md | 984 | C (the direction-rule formula, sprint cadence); G (the 1/18-tile re-pin proof); P (speeds: config.toml) |
| 113 | Turning law | 2784 | 117 | units.md | 1027 | C (the law's formula, hash names); P (360 deg/s, 30 rad/s^2: config.toml); G (90 deg in 11 ticks) |
| 114 | Unit impulse | 4872 | 147 | units.md | 1455 | C (slot-1a algorithm: sway, effort, knockdown conditions; shove geometry); P (config.toml `[impulse]`); G (1.5 profile slides ~6 m in 12 s) |
| 115 | Horde field | 2832 | 170 | horde.md | 1062 | C (anchor grid, UNREACHED value, behaviour dials, sprint rule) -> horde.py; H (G9p, residency P3d) |
| 116 | Noise | 1149 | 127 | horde.md | 539 | C (the three source sites) |
| 117 | Match config | 1605 | 186 | match.md | 772 | C (spec fields and defaults, writer / replay byte details) |
| 118 | Spawn rooms | 1231 | 144 | match.md | 539 | C (formation grid: 3x3 footprints at pitch 5) |
| 119 | Match controller registry | 3219 | 198 | match.md | 1245 | H (#119-#195 1j); C (`IdleControl`, registry names, `bots_v1`'s seven keys) |
| 120 | Bot layers | 2730 | 137 | bots.md | 1184 | C (plays list, focus key, staggering details); H |
| 121 | Position features | 932 | 102 | bots.md | 461 | C (the feature list: design 2.2) |
| 122 | Team picture | 3458 | 151 | match.md | 1074 | C (`Record` fields, tracker definitions, the saw rule in detail) -> team_picture.py; H |
| 123 | Match recording / replay | 4775 | 181 | match.md | 1295 | H, B (FORMAT_VERSION history v2-v8: git and the module); C (follow-camera mechanics, keys, `[replay]` key list) |
| 124 | MatchSession + the `policy` controller | 1578 | 174 | match.md | 478 | C (per-verb column names) |
| 125 | Batched environment | 4643 | 106 | rl-agent.md | 1389 | G (farm readings 1h / 1i, env-ticks/s) -> rung-1 notes 1h, 1i; C (obs / mask / reward shapes and key lists) -> rl_env.py |
| 126 | Observation spec | 6301 | 155 | rl-agent.md | 1606 | C (window shapes, plane definitions, denominators, the CHOICES list: marked `CHOICE` in the module); G (imprint ratios: spec 3.2); H |
| 127 | Full view | 3722 | 99 | rl-agent.md | 745 | C (column counts and layout, per-block rules); G (6.6 ms; budget 417.3 / 385.3 / 358.2 env-ticks/s: the spec doc) |
| 128 | Action spec | 3541 | 177 | rl-agent.md | 1245 | C (head sizes 49 / 18+E / 2 / 7+K / 5, argument steps, `stride`) |
| 129 | Reward profile | 2609 | 157 | rl-agent.md | 867 | C (balance formula, group list, book identities) |
| 130 | Bots through the door | 2058 | 150 | bots.md | 748 | C (record format 1 fields, `meta` contents) |
| 131 | Training snapshot | 1193 | 145 | rl-agent.md | 645 | C (manifest contents list) |
| 132 | Vision / engagements / cover | 1823 | 215 | sight.md | 1058 | H (#195 1a, 1g) |
| 133 | Sight for many tiles | 2402 | 171 | sight.md | 729 | G (~2.7x at 12 units); B (first / second user) |
| 134 | Room graph | 2338 | 144 | match.md | 728 | C (`opening_states`' openable definition, cache keys) |
| 135 | A* | 665 | 74 | horde.md | 309 | N beyond wording |
| 136 | Planted charge | 2361 | 162 | hand-verbs.md | 1032 | C (keys X / 5, `PlantEvent`, `det_tick` formula); P (`plant_seconds` 1.0: config.toml) |
| 137 | Verb gates | 2382 | 164 | hand-verbs.md | 1056 | N beyond wording |
| 138 | Order | 2728 | 149 | match.md | 881 | C (`order_intent` signature, `ORDER_OPENING_BASE`); D (PLAY_ORDER: Bot layers row) |
| 139 | Timed charge | 1264 | 217 | explosions-fire.md | 866 | P (cookoff / hazmat rows: config.toml) |
| 140 | Heat source | 1166 | 207 | explosions-fire.md | 560 | C (runtime digest rows); P (`engine_death`: config.toml) |
| 141 | main.py | 337 | 115 | match.md | 221 | N |
| 142 | Map editor pattern | 126 | 126 | — | 0 | N |
| 143 | Undo | 83 | 83 | — | 0 | N |
| 144 | Editor UI from registry | 87 | 87 | — | 0 | N |
| 145 | Baker / airtight / play-scratch | 188 | 200 | — | 0 | N |
| 146 | Level generators | 8167 | 215 | levels.md | 3618 | H (#120, #133, #134, #136, #177); C (EdgeKind details, vent rate formula, lamp span rules, interior airlock mechanics, bypass geometry) -> gen_ship.py, levelgen_zone_vocabulary_2026-10-29.md |
| 147 | Recipe library + resolver | 2361 | 151 | levels.md | 1052 | C (attach / exit / hull_door placement details, the named config tables) |
| 148 | Archetype recipe + grammar | 2206 | 174 | levels.md | 898 | C (G_MAX = 16, archetype file contents, the seven moves' names, profile operators) |
| 149 | Layout stage | 2378 | 128 | levels.md | 961 | C (skeleton geometry: slab, rows A / B, stubs, end zone, ring, hull strip) |
| 150 | LevelSpec | 1317 | 160 | levels.md | 638 | C (the record's key list) |
| 151 | Declared-edge contact | 680 | 163 | levels.md | 281 | C (`at` placement rules) |
| 152 | Axis parity | 387 | 105 | levels.md | 282 | C (`DOOR_W` 4, `GLASS_W` 2, `AIRLOCK_W` 5) |
| 153 | Character kit | 667 | 140 | characters.md | 377 | N |
| 154 | Game-ready step | 1158 | 145 | characters.md | 786 | C (`gloss_signal`, the preview tool) |
| 155 | Toon look | 555 | 108 | characters.md | 186 | C (EEVEE technique list); state (Erik's 2026-10-22 praise) |
| 156 | Worker builder | 617 | 168 | characters.md | 339 | C (the table names) |
| 157 | Making a character | 429 | 126 | — | 0 | N (the skill holds its provisional note) |
| 158 | Tuning the smoke look | 409 | 94 | — | 0 | N (the skill holds the procedure) |
| 159 | Tuning the light curve | 361 | 102 | — | 0 | N (the skill holds the procedure) |
| 160 | Tuning fire + smoke | 556 | 203 | — | 0 | N (the skill holds the procedure) |
| 161 | Benches | 712 | 155 | determinism-gates.md | 69 | C (per-bench descriptions) |
| 162 | GPU launch | 89 | 89 | — | 0 | N |
| 163 | Headless match runner | 634 | 141 | match.md | 137 | C (printed outputs, CLI flags) |
| 164 | Bot tournament | 1320 | 108 | bots.md | 282 | C (CLI flags, per-match record fields, pin folder) |
| 165 | E2E driver | 973 | 118 | determinism-gates.md | 260 | C (flag list) |

**Totals:** 165 rows; 249921 bytes of rows became 26274 bytes of index lines + 106760 bytes of rule sections; 28 rows need no rule section. Plus the dated header log (~12 KB), dropped whole (git log holds it).
