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
