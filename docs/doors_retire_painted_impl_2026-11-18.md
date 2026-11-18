# Retire the painted doors; use the door you face (#215) -- as built

Branch `215-doors`, cut from `215-grenade-bounce` at `30bd61ca`. Erik's
rulings (2026-11-18): retire the legacy painted doors completely; doors are
manually operated (use toggles, the door stays until used again); closed,
with bashing on; more grilles if air does not move as wanted; use picks the
door the unit FACES when several are in reach, reach alone when one is.

## 1. Census BEFORE anything changed (base `30bd61ca`)

Measured by a read-only script over every `levels/*/level.toml` (the
loader's own parse, `entities.door.runtime_span` for the spans; a "run" is a
4-connected component of bare `MAT_DOOR` tiles).

| level | `MAT_DOOR` tiles | under a door span | bare | bare runs (h x w : count) | door entities |
|---|---|---|---|---|---|
| `ship_demo` | 240 | 0 | 240 | 4x1 : 30, 1x4 : 30 (60 runs, all straight, 4 tiles) | 0 |
| `unhcr_vessel` | 21 | 0 | 21 | 1x3 : 3, 1x6 : 2 (5 runs, straight) | 0 |
| `bake_demo` | 1 | 0 | 1 | 1x1 : 1 | 0 |
| every other level (20) | 0 | 0 | 0 | -- | airlock_demo 2, door_test 4, fire_studio 1, fire_tuning 1, playground 6, test_level 5, wego_test 1, wind_studio 2, the rest 0 |

Every bare run is a straight line, so every one is representable as ONE
door entity (a straight `h` / `v` span of 1..6 tiles); nothing had to be
refused for shape.

**Generators that emit bare paint**: `tools/gen_ship.py`, the `door` and
`vented_door` rows of `EDGE_KINDS` (stamp `MAT_DOOR`, no entity). v1 ships
use `door` for every room joint (`ship_demo` is one); the recipe library
uses `door` in five cycle files and four department files and `vented_door`
in `crew_quarters`. `tools/gen_training_rooms.py` uses `entity_door` only
(no paint). No other generator writes a level with `MAT_DOOR`
(`tools/eos_p5_bake.py` builds in-memory maps for old-checkout comparisons
and writes no level).

**Tests that place `MAT_DOOR` by hand** (synthetic grids or `LevelData`,
never a level file; 87 lines in 18 files): `cuda_unit_kernels_check.py`,
`cuda_unit_payloads_check.py`, `cuda_unit_topology_check.py`,
`test_a5_seal_evacuation.py`, `test_atmosphere_conservation.py` (and it
reads unhcr_vessel's doors), `test_bake_level_art.py` (and it asserts
bake_demo holds a `MAT_DOOR`), `test_eos_p2_sealed_room_energy.py`,
`test_gen_ship.py`, `test_gen_ship_grille.py`, `test_grille.py`,
`test_level_editor_tool.py`, `test_level_format_v2.py`,
`test_map_editor_tool.py`, `test_materials.py`, `test_migration.py`,
`test_playground_level.py`, `test_wave_push_door.py`,
`test_wind_doorway_e2e.py`.

**Tools that place or read `MAT_DOOR`**: `tools/gen_ship.py` (stamps it,
above); `tools/align_level_art.py` (paint key 3); `tools/map_editor.py`
(`door_check`, the legacy gate kept for API stability; its DOOR tool places
an entity + `MAT_DOOR_CLOSED`); `tools/migrate_level_entities.py` (A7's
painted -> entity migration, never run on the three levels above);
`tools/migrate_tilemap_v2.py` (v1 -> v2 code map); `tools/obs_body_imprint.py`
and `tools/wind_load_probe.py` (find doorways by `MAT_DOOR`); the map
editors' palettes derive from `MATERIAL_NAMES` (so "door" is paintable).

**What reads `MAT_DOOR` in the code**: `src/simulation/door_system.py` (a
span may sit on it), `src/simulation/physics.py` (the blast wall tuple),
`src/simulation/simulation.py` (slots 9 / 9b: a destroyed door tile emits
`DoorDestroyedEvent`), `src/simulation/gamemap.py` (import only),
`level_loader.py` (the v2 decode). `cpp/` and `renderer/` read no door id
(the device's blast set is a table row built from the Python tuple).

## 2. What was built

**The loader's refusal.** `level_loader.refuse_bare_painted_doors`, called by
`load` after the entities are parsed: a `MAT_DOOR` tile under no door
entity's span (base resolution) is refused by name, with the migration tool
named. The `MAT_DOOR` id itself stays in the table (ids never move); a door's
span may still sit on it (the door's load stamp overrides it), only the BARE
use is refused. Nothing more was retired: the id keeps its `[materials.door]`
row (synthetic tests and the v2 decode still use it), and the three readers in
`src/` (span materials, the blast tuple, the slot-9/9b door event) are inert
on every level and were left as they are. `level_lib.open_level` and `load`
take `allow_bare_painted_doors` (the migration tool's read only).

**The generator.** `gen_ship`'s `door` and `vented_door` rows now stamp
`MAT_DOOR_CLOSED` and name a door entity (`entity="door"`); `_build_doors`
emits one per edge (`door_e<i>`), its initial state READ from the row's stamp
(`door_initial`: closed for the two door rows, open for `entity_door`, whose
`ENTITY_DOOR_INITIAL` is unchanged). `validate`'s reach gate walks through
every interior door edge's opening (a unit opens it; the horde bashes it),
exactly as it already walked through an airlock's closed outer door. The
walk, solid and structure masks are bit-identical to the painted door's (the
painted tile was walkable and solid), so no plan, sub-seed or draw moves.

**The migration tool.** `tools/migrate_level_entities.py` (A7's, never run
on a shipped level until now) gained `--doors-only`: the painted doors alone,
`[[light]]` left as it is (unhcr_vessel still carries five legacy lights; the
light migration is A7's other half and moves digests on its own, so it stays
out of this patch). KEPT: it is THE way a level that still carries painted
doors (an old branch, a hand edit) comes to entity form, since such a level no
longer loads.

**The levels.**

| level | how | what changed |
|---|---|---|
| `ship_demo` | regenerated by its own header's command (`gen_ship --seed 1 --size medium --max-per-team 6 --zombies 4 --vents`) | `tilemap.csv`: exactly the 240 door tiles, 3 -> 7; `level.toml`: 60 `door` entities appended after the vents (no existing line or ordinal moved); `diffuse.png`: exactly the 240 door tiles' 16x16 pixel blocks (61 440 px, the generator's palette colour of a closed door); `match.toml`, `rooms.npy` byte-identical |
| `unhcr_vessel` | `migrate_level_entities.py --doors-only` | 21 tiles 3 -> 7; 5 closed doors (`door_1..5`: three of 3 tiles, two of 6) appended; art untouched (hand-painted) |
| `bake_demo` | the same, then re-baked through `tools/bake_level_art.py` | 1 tile 3 -> 7; 1 closed door; the baked diffuse / normal moved only on the door tile and its two autotiled jambs (column 11, rows 6..8) |

**The bash.** `[zombie] door_bash_damage` 2.0 -> 4.0 (bash_doors was already
on). As built: `[materials.door_closed] hp` 40 / 4 = 10 strikes; the first at
once, then one per bite cadence (2.2 s -> 53 ticks at 24 tps): ONE zombie
breaks an interior door 9 x 53 = 477 ticks = 19.9 s after its first strike
(the slot-9e sweep brings the whole door down the tick its first tile reaches
0). Several zombies add up only on the SAME tile (each strikes the first
bashable tile its blocked step touches, row-major): k zombies on one tile
~20 / k s; zombies on different tiles of one door do not add (the door falls
with its first tile). Hull doors stay un-bashable (`is_hull_door`, unchanged).

**Use by facing.** `verb_gates.use_door`: among the live doors in reach, the
one with the smallest `facing_off` (|`wrap_q16(atan2_q16(-dy, dx) -
facing_q)`| from the unit's footprint centre to the centre of the door's
nearest span tile, squared Q16 distance, ties to the earlier span tile; the
shove's own bearing form), ties to the reach key (today's scan order). One
door in reach: that door, the facing never read. `can_use` and the mask are
unchanged ("a door in reach"). Device: `use_door` in
`cpp/src/cuda_unit_verbs.cu`, two passes over the per-door span rows (the
candidates and their reach keys; then, only with two or more, the offsets),
an operand past int32 a fault. The door table's capacity rose to 128 doors /
1024 span tiles (ship_demo alone has 60 / 240; was 32 / 256).

**The bots.** `BotTeam._open_door` goes through the verb (`use_intent`), so
it now issues the use only when `verb_gates.use_door(sim, u)` IS the door on
its path; otherwise it waits a tick (it turns toward its waypoint as it
walks). A turn in slot 1a between the bot's read and the verb could still
change the choice when two doors lie almost equally off its facing
(reported, not closed).

**Grilles.** None added.

**Also migrated: `prototypes/unhcr_vessel_2`** (the one tracked level outside
`levels/`, the F2 art level; missed by the levels-only census): 30 bare
painted tiles, 8 closed doors, `--doors-only`. Its two tracked `.bak` files
were overwritten by the tool's backup and restored from git (a finding: the
tool's `.bak` step clobbers a tracked `.bak`).

**The walk gates.** Two more gates walked the painted door's walkable tile:
`gen_ship.validate` without a graph (the tiles alone: now every closed door
tile is walked through, since the tiles cannot tell an airlock's from an
edge's and a hull door leads only to vacuum) and `ship_layout.bot_path` (the
recipe path's `bot_walkable` / `spawn_distance`: now it walks through the
span of every door an edge emitted, `gen_ship.edge_door_tiles`, where a
closed door still stands). Without them no recipe ship passed its gates.

## 3. Measurements

**Air and smoke do not move.** Each level's physics alone (no units; a
smoke puff of 2.0 seeded in a 5x5 box), 240 ticks, the base tree's level vs
this branch's, per-field hashes at ticks 60/120/180/240 of `atmosphere`,
`temperature`, `gas_energy`, `smoke`, `wind_x`, `wind_y`, `fire`, `wall_hp`,
`solid`: **bit-identical** on `ship_demo` and `unhcr_vessel` (the migrated
levels: only the material plane differs, exactly the doorway tiles 3 -> 7)
and on `playground` (an unchanged level). `[materials.door]` and
`[materials.door_closed]` differ only in `mobility` (1000 vs 0), which no gas
law reads. No grille was needed.

## 4. The pinned matches (NOT re-baselined)

Both pinned ship_demo matches move: every one of ship_demo's 60 doorways is
now a closed door. Read-only hooks around `run_match` (`_direct_use` and the
slot-9e sweep):

| match | pin (main) | base `30bd61ca` | this branch |
|---|---|---|---|
| `bots` | win team 0 at 2570 | win team 1 at 2121 | **win team 0 at 608** (field `7ea139a1e506...`) |
| `bots_v1` | draw at 7200 | draw at 7200, field `14ae2600...` | **draw at 7200**, field `fae949e041c9...` |

- `bots`: 6 uses, all opens, 3 per team (team 1 at ticks 45, 257, 406 on
  `door_e59`, `door_e33`, `door_e55`; team 0 at 69, 269, 416 on their mirror
  twins `door_e58`, `door_e32`, `door_e54`), all by the same unit of each team
  (the lead walker); no door closed again, none destroyed. Team 1 is wiped
  out at 608; the four zombies are dead by then, none moved more than 25
  tiles from where it spawned, and no door was bashed.
- `bots_v1`: **0 uses, 0 doors opened, 0 bashed**: bots_v1 never uses a door
  (its law is pinned), so both teams stay shut in their spawn rooms and the
  dormant zombies in theirs (nothing hunts, so nothing bashes): a draw at the
  time limit with all 12 units alive. A finding for the lead / Erik: the
  `bots_v1` ladder rung no longer plays a match on ship_demo.

The six pin tests (base = `30bd61ca`, from the base's own doc; branch = the
full suite at `822a4be1`):

| test | pin (main) | base | this branch |
|---|---|---|---|
| `test_zombie_control::..._pinned_ship_demo_match` (`bots`) | win 0 at 2570 | win 1 at 2121 | **win 0 at 608** |
| `test_bots_v1::..._pre_g8p_pinned_match` (`bots_v1`) | draw at 7200 | same | same outcome, field `fae949e0...` |
| `test_exposure::..._pre_g4p_match` | draw at 7200 | same | same outcome, field `fae949e0...` |
| `test_grenade_kinds::..._pre_g3p_match` | win 1 at 240 | same | **draw at 7200** (shut in) |
| `test_horde_behaviours::..._stage_1_law` | win 1 at 255 | same | **draw at 7200** |
| `test_unit_impulse::..._re_pin_1s` | draw at 7200 | same | same outcome, field `fae949e0...` |

No GOLDEN_AGGREGATE, field-digest golden, replica gate or baked-art golden
moved (the A/B scenario is a synthetic 16x16 room; `test_bake_level_art`'s
pinned golden bake is a synthetic map); nothing was re-baselined.

## 5. Tests and breaks

New tests (each names its property and the change that breaks it):
`tests/test_doors_retired.py` (no level in the repo holds a bare painted door,
every level folder plus `prototypes/unhcr_vessel_2`; the loader's refusal by
name, a span's own paint allowed; gen_ship's door rows emit closed entities;
a migrated ship_demo doorway blocks rounds, grenades, gas, sight and movement
closed, none of them open, all again closed); `tests/test_use_door_facing.py`
(the faced door at a sweep of 64 facings, the exact tie to the scan order,
one door ignores facing, one use toggles the faced door and only it, the gate
stays "a door in reach"); `tests/test_horde_behaviours.py::test_one_hunting_
zombie_breaks_an_interior_door_in_about_20_seconds` (18..22 s; the hull door
test was already there); `tests/test_bots.py::test_a_bot_between_two_doors_
opens_the_one_on_its_path` and `::test_the_bot_uses_a_door_only_when_the_
verb_would_flip_that_door`; host equals device: `tests/cuda_unit_verbs_check.py`
gained a two-door wall in its verb scene (a unit swept over ten facings,
three new census branches that must be reached: the tie, the faced door
first in table order, the faced door not first).

Every new gate broken once (a text patch, the named tests run, restored):

| break | red in |
|---|---|
| use ignores facing (host: the scan order only) | `test_use_door_facing` |
| a tie goes to the LATER scan key | `test_use_door_facing` |
| facing enters the single-door case | `test_use_door_facing` |
| the bot skips the verb's choice | `test_bots::test_the_bot_uses_a_door_only_...` (the walking two-door test stayed GREEN: in that geometry the bot already faces its door; hence the second test) |
| the loader accepts a bare painted door | `test_doors_retired` |
| gen_ship's `door` row paints `MAT_DOOR` again | `test_doors_retired` |
| `door_bash_damage` back at 2.0 | the 20 s bash test |
| a closed door passes gas (`permeability = 1.0` on `door_closed`) | the migrated-doorway test |
| the DEVICE use ignores facing (`use_door` always the first candidate; built, run, restored, rebuilt) | `test_cuda_unit_verbs` (the chained verbs scene left the reference at tick 82, the facing sweep) |

Pre-existing tests changed, each because it pinned the painted door itself
(its id 3, its walkable tile, or a scene that depended on ship_demo's open
doorways), its property kept: `test_bake_level_art` (bake_demo's door is
`MAT_DOOR_CLOSED`), `test_level_loader` / `test_level_format_v2` /
`test_materials` (unhcr_vessel's doors are id 7), `test_atmosphere_
conservation` (the stamp guard on a closed door tile), `test_migration` (its
fixture is read with the migration's allowance), `test_grille` (the vented
door's contrast is an open doorway; the gas contrast a closed door),
`test_gen_ship` / `test_gen_ship_grille` / `test_gen_ship_vents_doors` (the
door rows' stamp; the entity-door and airlock tests count their own doors
among the new edge doors), `test_room_graph` (two tests open the `door`
edges' doors first so only the annex door is in play), `test_levelspec`
(`bot_path` is handed the level's entities), `test_bots_door::test_a_masked_
latch_is_never_sent_on_a_whole_match` (its scene moves from seed 4 to 6:
on the new ship seeds 1-5 mask no latch), `cuda_unit_horde_check` (its scene
drops the cache by a topology change off a refresh tick with two pillar
tiles, since the faster bash now breaks the door on a refresh tick; its
ship_demo chain takes the 9e(d) door sweep), `cuda_s8a_check` (below).

## 6. Findings in the reference (reported, not changed)

1. **The resident stamp's `obstacles` is one tick stale after a slot-9e
   topology change.** The device stamp builds `obstacles` at slot 6 from the
   uploaded permeability; a door the 9e sweep opens or closes changes the
   host's `obstacles` through `on_tile_changed` the same tick, the device's
   only at the next tick's stamp. No device stage reads `obstacles` (the
   comment in `physics_runner` says so), so nothing diverges, but the S8a
   gate compared it at the tick's end and went red on ship_demo's first door
   opening (ticks 45 and 69, 4 cells each). The gate now compares
   `obstacles` on the next tick when a tile changed after the stamp's rows
   were taken (`cuda_s8a_check._track_stamp_serial`). If a device stage
   ever starts reading `obstacles`, this is a real bug.
2. **`bots_v1` no longer plays a match on ship_demo**: it never uses a door
   (its law is pinned), so every team and the dormant horde stay shut in
   (section 4).
3. **Team 0 wins every bots match on the new ship_demo** (the pinned `bots`
   at 608; the door-team scene's seeds 1-6 all `win 0` in 751..1195 ticks).
   On a mirrored map the first doors each team opens are mirror twins, team 1
   at tick 45 and team 0 at 69, so something in the play is not mirror-
   symmetric; the use's tie rule (row-major scan order) is not mirror-
   symmetric by construction, and is one candidate. Not investigated.
4. **The migration tool's `.bak` step overwrites a TRACKED `.bak`**
   (`prototypes/unhcr_vessel_2` carries both in git); restored by hand.
5. **The map editors' material palette still offers `door`** (`MAT_DOOR`);
   a level painted with it is refused at load (accepted gap in the rule).
6. **The full view drops every zombie on ship_demo** (section 7: 60 doors
   outrank the zombies in the 64-entry list); `test_obs_spec_v1::test_the_
   full_view_equals_an_independent_reconstruction` is left RED for the
   game-AI lead.
7. **A bot's use can still pick the other door if slot 1a turns it** between
   its read of `use_door` and the verb at 1b, when the two doors lie almost
   equally off its facing.

## 7. Gates

- Both builds (`cpp\build_cpu_home.bat`, `cpp\build_cuda.bat`): exit 0.
- `tools/twin_debt.py`: the verbs, table_appends and doors twins were in debt
  only for `verb_gates.py` (this patch moved it with its device form,
  `cpp/src/cuda_unit_verbs.cu`); marked level after the device gates passed
  (`--mark-level`), so the ledger is empty.
- The device gates on this tree: `test_cuda_unit_verbs`, `test_cuda_unit_doors`,
  `test_cuda_unit_horde`, `test_cuda_unit_vents`, `test_cuda_s8a_residency`
  green after the gate fixes above; `tools/cuda_green.py` is run last, at the
  final HEAD (its stamp is HEAD's; see the report).
- **The full suite** (`pytest tests -q`, one run at `822a4be1`, 1 h 52 min):
  6485 passed, 20 failed. Of the 20:
  - the SIX pin names (RED by design, not re-pinned): `test_zombie_control::
    test_the_controller_plays_the_pinned_ship_demo_match`, `test_bots_v1::
    test_bots_v1_play_the_pre_g8p_pinned_match`, `test_exposure::test_with_
    the_law_dormant_the_pinned_match_is_the_pre_g4p_match`, `test_grenade_
    kinds::test_with_no_wind_up_the_pinned_match_is_the_pre_g3p_match`,
    `test_horde_behaviours::test_with_every_behaviour_off_the_horde_plays_
    the_stage_1_law`, `test_unit_impulse::test_g1_with_every_input_off_
    nothing_is_pushed_and_the_pins_are_re_pin_1s`;
  - red only because they share those pins (each plays or watches the same
    ship_demo match): `test_turn_rate::test_at_half_a_turn_per_tick_the_
    pinned_match_is_the_pre_g1p_match` and `test_turning_law::test_g3_at_
    unlimited_acceleration_the_pins_are_re_pin_1s` (both compare a law-off
    play of the `bots_v1` match with a pin: now the shut-in draw);
    `test_unit_schema::test_derived_constants_and_round_trips_hold_at_every_
    tick[bots]` (it watches the pinned `bots` play and requires a conversion
    in it: the new play, won at 608, has none) and `test_unit_store::test_
    identity_ship_demo_every_tick` (requires the ship_demo bots match to run
    650 ticks: it now ends at 608). These two are scene-length / non-vacuity
    bounds on the pinned play: the re-pin will not turn them green; the lead
    picks their scene with the re-pin;
  - `test_class_p_level` (no stamp yet at that HEAD: `cuda_green` runs last);
  - FIXED after the run (each re-run green on its own): `test_cuda_unit_vents`
    (its ship_demo chain takes the door sweep), `test_twin_debt` x2 (the
    manifest regenerated, the door modules classified), `test_wave_push_door`
    (its painted door built in memory), `test_wind_doorway_e2e` x4 (the
    closed door entity's id);
  - **LEFT RED, a finding for the game-AI lead:** `test_obs_spec_v1::test_the_
    full_view_equals_an_independent_reconstruction`. The full view (spec 5.3,
    `obs_spec_v1._full_entries`) keeps 64 entries and, past that, drops the
    farthest zombies first, then objects, and doors only last ("a CHOICE for
    an absurd map"). ship_demo now has 60 doors, so 1 self + 7 marines + 56
    doors fill the list and the four living zombies are dropped from every
    soldier's full view; the test's "every living unit is in the list" fails.
    The priority (doors over zombies) is a spec decision, so it is not
    changed here.
- `python -m pytest agents/tests -q`: see the report.

## 8. Follow-up 1: the full view's drop order (the game-AI lead's ruling, 2026-11-18)

Over `FULL_ENTRIES` the full view (`obs_spec_v1._full_entries`) now drops the
FARTHEST DOORS first, then the farthest zombies, then the farthest objects,
never a marine: the room after the marines goes to the objects, then the
zombies, and the doors (nearest first) fill what is left. The entry order in
the list is unchanged (marines, zombies, objects, doors). The reference
(`tests/_obs_reference.py::full_list`) counts what each kind keeps, written
apart from the builder's slicing; the docstrings and
`.claude/rules/rl-agent.md` say the new order.

NOT a new spec version: no stored view changes, because no level a network
has been trained on overflows. The training rooms (`gen_training_rooms.
HAND_ROOMS`: `range_hall` 0 doors, `range_door` 1, `range_corner` 2; the
largest door count is **2**) hold a handful of entries against 64.

New test `test_obs_spec_v1::test_over_the_cap_the_farthest_doors_go_first_
never_a_unit` (ship_demo after one tick with three live charges placed: 60
doors against 49 free entries; every living row's list holds every zombie
and every object, and its kept doors are exactly its nearest ones). Broken
once (the builder's old order restored): it went RED, and so did `test_the_
full_view_equals_an_independent_reconstruction`; restored, both green.

Gates run: `tests/test_obs_spec_v1.py`, `tests/test_reward_v1.py`,
`tests/test_training_rooms.py`, `tests/test_bots_door.py` (the files outside
agents/ that read the full view), `tests/test_claude_rules_paths.py`: 170
passed; `python -m pytest agents/tests -q -k "full or value or sizes"`: 6
passed, 113 deselected. `test_the_full_view_equals_an_independent_
reconstruction` is green (it was the one test left red in section 7).
