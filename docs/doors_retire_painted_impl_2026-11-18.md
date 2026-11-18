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
