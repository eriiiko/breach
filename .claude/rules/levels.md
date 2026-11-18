---
paths:
  - "level_lib.py"
  - "level_loader.py"
  - "levels/**"
  - "tools/gen_*.py"
  - "tools/ship_grammar.py"
  - "tools/ship_layout.py"
  - "tools/levelgen_recipes.py"
  - "tools/map_editor.py"
  - "tools/undo_log.py"
  - "tools/entity_editor_ui.py"
  - "tools/bake_level_art.py"
  - "tools/level_airtight.py"
  - "tools/play_scratch.py"
  - "tools/migrate_tilemap_v3.py"
---

# Levels: the data layer, the editors, the generators

Fires on contact with these files. CLAUDE.md's index says what each system is
and who must use it; this file holds the constraints the code cannot say. The
`level-design-lead` skill is the role's continuity.

## Level data layer
- One writer ever (`level_lib.py`): every tool is a client; never hand-write level.toml.
- NO BARE PAINTED DOOR (#215, Erik 2026-11-18: the legacy painted door is retired): `level_loader.refuse_bare_painted_doors` refuses a `MAT_DOOR` tile under no door entity's span, by name. Every doorway is a `door` entity; no generator or door tool stamps `MAT_DOOR` (they stamp `MAT_DOOR_CLOSED` / air under the entity); the id stays in the table and may sit under a span. ACCEPTED GAP: the map editors' material palette (`tools/level_edit_common.py`) still lists it, so a hand-painted one is caught only at load. A level that still carries painted doors is converted by `tools/migrate_level_entities.py --doors-only` (its read alone passes `allow_bare_painted_doors=True`); never a second door form. `tests/test_doors_retired.py` reads every level.
- Optional per-tile seeds beside level.toml, found by presence, dormant when absent: `air_init.npy` and `temp_init.npy` (#133 E.2: game degrees above ambient; refused below 0, since sub-ambient is a parked physics ruling, above `T_MAX_PHYS`, and non-zero on vacuum, the ring or a wall without thermal mass), written only by `write_air_init_npy` / `write_temp_init_npy` (an all-zero temp grid DELETES the file). The loader seeds gas through `GameMap.seed_gas_temperature` and thermal solids before the first books snapshot: never a bare `temperature[...] =` elsewhere.
- The ROOM GRAPH (`[[room]]`, `[[opening]]`, `rooms.npy`) is written only by `level_lib.room_graph_blocks` / `write_rooms_npy` and read through the refusing door `level_loader.parse_room_graph` (the three parts together or none).
- DEBT: `tools/play_scratch.py` and the map editor do not carry `temp_init.npy`; the editor has no temperature paint.

## Level generators
- A generated level is written through `level_lib` (`write_level_header`, `write_boundary_field`, `write_managed_blocks`, `write_tilemap_csv`): never a hand-written level.toml or CSV (`tools/gen_fire_studio.py`'s `np.savetxt` + literal toml is the anti-pattern, #67).
- `tools/gen_ship.py`: stage 1 a room graph, stage 2 its embedding (`embed`), then the gates (`validate`: airtight via `level_airtight.find_leaks`, A* spawn reachability, mirror symmetry, formation cells on floor) inside the generator before any write; a rejected plan retries the next sub-seed. A new generator replaces stage 1 and reuses `embed` + `validate`.
- Family B (#136): `generate(seed, size, recipe=, profile=)` draws grammar attempt g ONLY from `SeedSequence([seed, 0, g])` and layout attempt a from `SeedSequence([seed, 1, g, a])`, through `validate` and then `ship_layout.acceptance`; `ship_layout.check_request` refuses before any draw. Without `recipe` it is v1, stream and output unchanged.
- Every ship, on both paths, is written with its room graph: `gen_ship.room_graph(graph)` is the one derivation.
- Every room adjacency is a row of `gen_ship.EDGE_KINDS`: a new way two rooms join is a new row there, never an inline stamp. Per-room tags are the `ROOM_TAGS` schema, refused by name when unknown. A grille tile is STRUCTURE for walking and for `compartments()`, so "one volume of air" is a flood over the loaded level's permeable tiles, never the compartment count. An unreached compartment passes only if every room in it is `secret` and a secret-path edge joins it.
- THE AIR-VOLUME VENT RULE (vocabulary §0.6, #124): `gen_ship.vent_air_rule` is the ONE statement, called by `ShipGraph.check`, the recipe door and the grammar's `air_volume` invariant on EVERY application, never a copy: every air volume a duct has a vent in carries a supply AND a return on it. `ShipGraph.check` also refuses a duct spanning two cells, a `vent_only` edge off a shared duct, and a cross-cell edge outside `CELL_CROSSING_KINDS`. A room pair carries ONE edge, bar the airlock bypass (`ShipGraph._check_bypass`).
- A recipe ship's vent rates are DESIGNED (`gen_ship.vent_rates`, from `[levelgen.vents] ach` through `gen_ship.vent_ach`, folded into the recipe hash): a new vent writer on the recipe path takes its rate from `vent_rates`, never a literal.
- `gen_ship.fuel_materials()` is THE fuel / contents set: never a literal; a mix naming another row is refused.
- Lamps: a new roof-light shape is a `[levelgen.lamp_shapes.*]` row, never code. A ship keeps its lamps within `gen_ship.LAMP_LIGHT_BUDGET` (the sweep's per-ordinate cone door, less the muzzle flashes' and beams' shares, #173); a new light source a generator emits counts against this budget.
- An airlock edge's mode is DERIVED by `gen_ship.airlock_mode` (Erik, 2026-10-31): `hull` when a tile past the outer door is space, else `interior` (each leg equalises the vestibule to the side about to open; nothing leaves the ship; both doors never open at once). A `hull_door` tag is a door entity, CLOSED by default (`state = "open"` refused until the breach archetype): a door to space is a valve a wire opens, not a path. An `entity_door` edge is emitted OPEN (`ENTITY_DOOR_INITIAL`); a `door` / `vented_door` edge is a door entity emitted CLOSED (#215). An entity row's initial state is READ from its stamp (`gen_ship.door_initial`), never a second table. `validate`'s reach gate walks through every interior door edge (a unit uses it, the horde bashes it).
- DEBTS: the v1 string vent shorthand is EXEMPT from the air-volume rule so `levels/ship_demo` stays byte-identical (fixing it is a deliberate re-baseline); the reachability gate counts an airlock's closed outer door as passable although bots cannot path through it (#142); a unit starts an airlock cycle only from the near side; no generated airlock runs in hull mode yet.

## Training rooms
- `tools/gen_training_rooms.py` writes rung 2's levels (design `docs/rl_agent_rung2_design_2026-11-14.md` 5.1): its `Draft` writer functions (room, door, open edge, cover crate, spawn rooms, match) and `HAND_ROOMS` are the one place a training room is drawn; 2g's seeded shapes are added there on the same functions, never a second tool, and never on the ship generator (hull, mirror and ship rules a single room does not meet).
- Gated BEFORE any write by the ship generator's own `gen_ship.validate` (airtight, spawn floor, A* reach) with `mirrored=False`: never a copy of a gate. Doors and opening stamps come from `gen_ship.EDGE_KINDS`; cover is the game's `cover` entity, never the ship generator's fuel crates.
- The rooms live OUTSIDE the repository (the tool refuses a folder inside it) and are made again from the tool: two writes are byte-equal (`tests/test_training_rooms.py`). Every room has a match file with `sight = "all_round"`, no zombies and its task's clock; each layout is also written `<name>_swapped` (the two spawn rooms exchanged, nothing else).

## Recipe library + resolver
- The family-B grammar's data (`docs/levelgen_cycle_types_and_recipes_2026-10-31.md`), loaded through a DOOR that refuses by name (file + field): unknown fields, a physics NUMBER where a preset NAME belongs, a missing named config row, an edge kind outside `EDGE_KINDS`, the air-volume rule at every pattern of optional-room absence, the triangle rule, the layout's packing twin at the minima (`ship_layout.department_feasibility`, never a second bound), the galley fuse, a malformed `second_leg`.
- Where a room goes is its `attach`, and a declared edge joins a room only to what it attaches to.
- A recipe never carries a physics number: it names config rows.
- The RESOLVER is the one step from those names to `ROOM_TAGS` values. A key whose stage-2 mechanism is not built stays in `ResolvedRoom.pending`, named with what it waits for, and `stamp_tags()` refuses it: never a silent drop.
- The size and count ranges are tuned against the layout; every moved number is a dated line in the recipe doc's as-built section.

## Archetype recipe + grammar
- A ship kind ON AN EXISTING SKELETON is one archetype file (`docs/levelgen_archetype_recipe_2026-10-31.md`); a new skeleton or anchor class is `tools/ship_layout.py` code first. The door refuses by file and field, walks the closure, and checks the closed-form skeleton conditions per (size, profile).
- THE GRAMMAR is the only stage 1 of a recipe ship: every application goes through one of the SEVEN moves and the phase's invariants, refused BY NAME and logged. Its product, the `QuotientGraph`, is the layout stage's ONLY input; the grammar never places a rectangle. A new shape is a cycle file on these moves, never an eighth move without a design line.
- Layout feasibility is the ONE call `ship_layout.layout_feasibility` (exact with the graph at the end of every attempt; by the door per (size, profile) at the archetype's minima). NOT BUILT: banks, `mutate`.

## Layout stage
- A recipe ship's geometry is produced ONLY here (`docs/levelgen_layout_stage_2026-10-31.md`). It emits the `ShipGraph` that `embed` consumes, adds no edge, picks no room, drops no application and never stamps a tile: a miss is `_Reject`.
- EVERY DRAW IS CONDITIONED ON EXACT SETS, and `layout_feasibility(size, graph=)` IS whether that set is empty, so the grammar's feasibility and the layout can never disagree. A change to how a room is placed changes `_config` / `_u_options` AND `pack_department` together (`tests/test_ship_layout.py::test_the_feasible_sets_are_exactly_what_packs`, `::test_feasibility_is_exact_on_drawn_graphs`).
- Steel voids are four kinds, each at most `VOID_MAX` and never adjacent: no 5x5 of steel inside the hull.
- A new way to place a room is a primitive here: never a search, a repair pass or a second layouter. NOT BUILT: the wall-slide repair, a hull of several rectangles, `mutate`'s per-department streams.

## LevelSpec
- A recipe ship records every generator input and its realised factors in ONE `[levelgen]` table, written only by `gen_ship.write` on the recipe path (v1 writes none: `levelgen = None`). The door refuses an unknown, missing or mistyped key by name; the writer runs it first.
- A factor is measured the way its gate measures it, never a second way; `generator_version` is `match_replay.build_id()`.
- A ship is a pure function of its record: `generate_from_spec` regenerates it and refuses a stale `recipe_hash` (`StaleRecipeError`) unless `allow_stale`. The level's seed is its own; match.toml's `seed` stays the episode's.

## Declared-edge contact
- The layout realises every DECLARED quotient edge as a shared wall of its opening + `CLEAR` each side, with its twin at the reflected tiles. An undeclared touch is a blank wall and is never forbidden; the grammar's invariants run on declared edges only.

## Axis parity
- An opening between two axis rooms is centred on the mirror axis and of EVEN width; an odd one is `_Reject`ed, so an axis-axis cell crossing is a bulkhead or a breakable wall, never an airlock. Engineering is entered only through the lobbies' mirrored airlock pair.
