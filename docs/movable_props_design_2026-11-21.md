# Movable props: furniture as simulation state (#226)

2026-11-21. **Status: PROPOSED, v1.** Paper only: no simulation code, no
digest, no golden, no spec and no device table changes with this document.
Track B step 2 of the November push (#238). The build starts when the agent
(#29 P8) and physics (#217) tracks are quiet.

How to read it. Sections 2 to 7 each state a DECISION, the ALTERNATIVE it beat
and why, and its COST. Claims about existing code carry `file:function`.
What was inferred and not confirmed in the code is marked UNVERIFIED. Gaps
the design accepts are marked `ACCEPTED GAP:` and collected in section 12.
Erik's rulings to make are in section 14, one line each.

Reading this as a critic. The determinism lens should start at sections 3
(the stamp rule and its check), 4.5 (the move and the refusal pass) and 10
(the gates). The physics and books lens at 4 (the push law and its momentum
books), 6 (fire, O2 and heat) and 12. The scope lens at 2 (where it lives),
8 (cost), 10 (the patch list) and 13 (systems).

---

## 1. Erik's notes and rulings, and what this arc is

### 1.1 The notes the design rests on (issue #226, quoted)

- **Erik, 2026-11-21 (issue body, voice):** "I want them, part of them at
  least, the big ones, to be some kind of CUDA swarm unit. I want them in the
  same state. The reason being that the octopus and other units should be
  able to use furniture as shields. So the first step is to get some kind of
  furniture into the state... and I want them to move with the wind and
  explosions."
- **Erik, 2026-11-21 (issue body, rulings already made):** small debris and
  small props are render-only and never touch the state ("they could be just
  rendered"); "not every object has to be a unit".
- **Erik, 2026-11-21, later the same day (comment 1):** "Perhaps the swarm
  unit is not the right unit for the furniture. Perhaps they need their own,
  some kind of middle ground. I think they need orientation, at least 3
  numbers or 6 numbers, and some kind of size, and some kind of stamping I
  would assume they would need. I also want them to be able to burn... some
  of them should have the property how easily they can burn, so that burning
  stuff can move around." On cost: "It's a meaty patch for sure, but I'm 100%
  sure it's doable. It's much less computation all in all compared to the
  solvers, which touch every cell 8 times per tick with their 8 substeps. So
  it should be kind of light." This replaces "swarm unit" in the 2026-11-15
  ruling on #207 for furniture.
- **Erik, 2026-11-21, third note (comment 2), burning RULED:** "I would assume
  that it's enough that it heats whatever is beside and underneath it. When
  it's stamped, it also moves the heat source." So a burning prop is a heat
  source that moves with its stamp; the flame on screen is render-only (#235).
- **Same note, to investigate:** "if we can allow a table to burn on one side
  and not the other side." (answered in section 6.2)
- **Same note, new columns:** friction ("these furniture things will need some
  kind of friction coefficient. That could depend on the weight and
  everything, and it could be unique for each item"); centre of mass and
  momentum ("feels like small additions when we are designing this");
  density ("Perhaps we need to give all items a density so we know which ones
  float and which ones don't").
- **Same note, water:** props are pushed by water as well as wind, floating
  ones follow the flow; that is its own issue (#236), done at the same time.
- **Same note, the scene it all aims at:** "the aquarium bursts in mid-fight.
  Someone shoots the wall to the aquarium. Water is coming out, and also the
  big octopus, and it starts to fight and throw things around." (#111)

### 1.2 What the arc is

Milestone 26, in Erik's order (issue body): (1) furniture in the state,
pushed by wind and blasts, able to burn (#226, this design); (2) it shields
wind, flamethrower, heat, then rounds and sight (#227); (3) units pick it up,
carry it and throw it (#228); (4) damage from impacts (#229, "the arc
after?"). This document designs step 1 completely and leaves explicit hooks
for 2 to 4, so their designs extend this one instead of reopening it.

### 1.3 What the arc is not

- **Small debris stays render-only** (#230): cups, papers, cables. The line
  is drawn by size in section 3.6: a thing whose smaller side is under
  1.5 tiles (0.5 m at 0.333 m tiles) cannot hold a stamp and is debris.
- **Damage to props and to thrown bodies** is #229. This arc records every
  hard contact (with its impact speed) as an event nothing reads, and gives
  the prop no hp column yet (an integrity column is #229's to add; see 12).
- **Water on bodies** is #236. This arc stores density (so #236 has it) and
  makes water do nothing to a prop.
- **The flame picture** is #235 (render-only, the FireVFX pass).
- **The octopus** (#111, #107) is what this prepares: a body that throws
  things needs things with mass, a pose and a push law.

---

## 2. The object: its state, and where it lives

### 2.1 The state, column by column

A movable prop is a rectangle on the floor with a pose, a mass and a material.
Every value is an integer. Two kinds of column: LOAD constants (fixed after
the level loads, derived once from the class row and the level's tile size)
and SYNCED columns (moved by the laws, hashed).

**Load constants** (from the prop's class row and its instance overrides,
quantized once per map at the door, like `unit_impulse.impulse_table`):

| column | integer type and currency | source |
|---|---|---|
| `cls` | int, the class row's index | the level's entity |
| `mat` | int8, a `[materials.*]` id | the class row names a material row (furniture, wood, steel ...) |
| `half_a_q`, `half_b_q` | int32, Q16 tiles | half length along the heading, half depth across it; refused below 0.75 tiles each (section 3.6) |
| `height_q` | int32, Q16 tiles | class row (read by the wind area now, by sight and rounds in #227) |
| `mass_g` | int64, grams | class row or instance override |
| `inertia_g` | int64, gram-tile^2 | DERIVED: `m (L^2 + D^2) / 12 + m |o|^2` (rectangle about its centre plus the parallel-axis term for the mass offset), integer, floored, at least 1 |
| `com_u_q`, `com_v_q` | int32, Q16 tiles | centre of mass offset in the prop's own frame (u along the heading); refused outside the rectangle |
| `mu_q16` | int32, Q16 | friction coefficient, class row or instance override (Erik: "unique for each item") |
| `cap_lin_q` | int32, Q16 tiles per tick | DERIVED: one tick of Coulomb friction, `mu g / (tile_m tps^2)` (the unit law's `cap_prone_q` form) |
| `cap_ang_q` | int32, Q16 radians per tick | DERIVED: one tick of the floor's friction torque on a uniform rectangle over its inertia |
| `density_kgm3` | int32, kg/m^3 | DERIVED: mass over the box volume (`L D H`); stored for #236 (floats or sinks); read by nothing in this arc |
| `wind_kw_q` | int64, Q16 | DERIVED: the wind impulse per unit of wind load per tile of projected width, `rho_air Cd H 1000 / (2 tile tps^2)` per tile (section 4.3) |
| `fuel_cap_q` | int32, Q16 hp units | DERIVED per burn cell: a quarter of the prop's burnable mass in the material row's hp currency (section 6.3); 0 for a row that is not flammable |
| `cell_heat_q` | int32 | DERIVED per burn cell: heat capacity from the material row's `density x specific_heat` and a quarter of the box (section 6.3) |
| stamp tables | 17 entries each | DERIVED: the coverage-to-plane tables of section 3.3, built once per class per map in integers |

**Synced columns** (written only by the prop laws, hashed every tick in the
prop's entity record, section 2.3):

| column | integer type and currency | written by |
|---|---|---|
| `x_q`, `y_q` | int32, Q16 tiles | the centre of MASS (not the rectangle's centre); the move (slot 3p) |
| `th_q` | int32, Q16 radians in `(-PI_Q16, PI_Q16]` | the heading; the move, wrapped through `turning.normalize_q16` (the unit side's one wrap) |
| `vx_q`, `vy_q` | int32, Q16 tiles per tick | the integration (slot 1a) and the move's wall and contact rules |
| `w_q` | int32, Q16 radians per tick | the integration and the move |
| `imp_x`, `imp_y` | int64, Q16 gram-tiles per tick | every pusher through the one `add` (section 4.1); the integration takes them |
| `imp_l` | int64, Q16 gram-tile^2 per tick | the angular part of every push |
| `t_q[4]` | int32, Q16 game degrees (the temperature solver's currency) | per burn cell; the prop thermal row (slot 9c) |
| `fuel_q[4]` | int32, Q16 hp units | per burn cell; the burn (slot 9e(d'')) only |
| `burning[4]` | 0/1 | per burn cell; the ignition (9d) and the burn (9e(d'')) |
| `alive` | 0/1 | 1 in this arc (no prop is ever destroyed before #229) |

Host-only (not hashed, the unit precedent "the momentum books" in
`.claude/rules/device-units.md`, Device conductor): the momentum books per
axis `(in, resisted, wall, clamped, contact_out, contact_lost)` and the
angular books.

Not stored at all: "asleep". It is DERIVED, section 5: a prop is asleep when
its impulse state is all zero, the `unit_impulse.at_rest` idea.

Why the centre of mass is the stored position and not the rectangle's
centre: a rotation then never moves the stored position, and every lever is
measured from it (`unit_impulse.lever_q` takes a centre). The rectangle's
centre `G = C - R(th) (com_u, com_v)` is derived each tick it is needed, from
the same `c, s` the stamp uses (section 3.2).

Cost: 17 synced integers per prop plus 4 x 3 per prop for the burn cells;
about 30 load constants. A few hundred bytes per prop.

### 2.2 Where it lives

**DECISION: a level entity class `movable_prop` with a host runtime store and
a device object table; its state hashed through the entity section.**

Concretely:
- Authored in the level as an entity (`src/simulation/entities/movable_prop.py`,
  schema only, import-light like `entities/heat_source.py`), placed by tools
  through `level_lib` (the Level data layer), its class rows in config
  (`[movable_props.<class>]`, a table, never a hardcoded class).
- Host form: ONE store object on the sim, `Simulation.props`
  (`src/simulation/movable_props.py`), structure of arrays in entity ordinal
  order, numpy int columns, so host loops vectorize. It is the only Python
  writer of the columns (the Swarm facade discipline: "the only Python-side
  writer of the store"). Built once at reset (the `build_heat_sources`
  pattern, `heat_source_system.build_heat_sources`).
- Device form: ONE `ObjectTable` schema, `unit_schema.PROP_TABLE`, on the
  Object tables mechanism (`src/simulation/object_tables.py`), rows in
  ordinal order, with its record function `unit_kernels.props_of`, its way
  back `props_to_host` and its entity-row function `prop_entity_rows`, as the
  door and heat-source tables do (`.claude/rules/device-units.md`, Timed
  charges and heat sources). The burn cells are four fixed columns each
  (every prop has exactly four cells, section 6.2), so no variable-length
  member and no side table.
- Digest: the entity section. Each prop's record is
  `entities/serialize.record_head` (the load-constant header) plus
  `record_tail` of its synced columns, the rows read by the entity-row
  function (`.claude/rules/device-units.md`, Entity section export: "A class
  whose state starts moving gets a table and an entity-row function in the
  same patch").

**Alternatives it beat:**

1. *Rows of the unit store / a `Unit`.* Rejected. The Unit is "one class;
   subclassing is an anti-goal" and the Unit schema is "THE one table of what
   a unit holds" (`.claude/rules/units.md`): a prop row there would put
   health, statuses, loadouts, turning, the observation's unit block and
   every unit law's loop over a table. Each of those laws would need a
   "not a prop" branch, which is the per-kind if the canon forbids.
2. *A swarm species with a footprint* (Erik's first idea, withdrawn by his
   second note). Rejected on three grounds read in the rules: a swarm unit is
   a point and a species law is "ONE FP_HD per-slot law" run in slot 7's tail
   (`.claude/rules/swarm.md`, Swarm step), while a prop's laws are unit-side
   slots (1a, 3, 6, 9c2); THE BOUNDARY says a swarm law "reads map fields,
   never a unit column" and the unit side sees the swarm only through planes,
   while a prop must be shoved by a unit, hand a unit momentum and later be
   carried (#228); and the swarm store has no pose, no heading rate, no mass.
3. *Its own SoA store on the map beside the swarm store* (`gmap.prop_*`,
   resident, its own `__props__` digest carrier). Considered seriously: it is
   the best host layout and it is how the swarm reached the device. Rejected
   because the map-resident store exists for laws that run INSIDE slot 7's
   device chain (`.claude/rules/device-residency.md`, Batched resident engine);
   a prop runs no law in slot 7. On the device conductor its laws are unit-side
   kernels, and "a new table is a schema there, never a second table
   mechanism" (Object tables). A third store would be a second mechanism with
   its own carrier, upload, dirty flag and gate. The host keeps the SoA layout
   anyway (inside `Simulation.props`), which is what made option 3 attractive.
4. *Static props that are re-stamped as materials every tick* (#220's "movable
   solid": the prop writes its material id into `material` each tick).
   Rejected in section 3.5: every move would be a topology change through
   `GameMap.on_tile_changed`, with gas retired and born, the tile serial
   bumped every tick (which makes `GameMap.take_stamp_rows` return None and
   the played resident tick fall back to uploading planes every tick).

**Cost of the decision:** a new entity class and its schema; a config table;
the host store and its facade; one `ObjectTable` schema plus the generated
layout (`tools/gen_unit_layout.py --write`); the record, way-back and
entity-row functions; Class-T kernels in a new family TU
`cpp/src/cuda_unit_props.cu` (on CMakeLists' CUDA list and the `/fp:strict`
list); a Class-T twin row. The entity serializer's per-record cost is Python
per prop per digest; at tens of props that is small (UNVERIFIED at hundreds).

### 2.3 Digest and versioning

- The entity section format is unchanged (`ENTITY_SECT_V1`): a new class adds
  records only on a level that holds one. No level in the golden corpus holds
  a movable prop, so **no golden moves** when the class lands. A level that
  gains movable props later (levelgen, #171 job 1) moves its own goldens in
  that patch, with written rationale.
- The prop's synced columns are hashed always (not "while non-zero"): they
  are new, so no existing digest depends on their absence inside a record.
- New per-tile planes (section 3.3: the prop mobility cap and the prop owner
  plane) are NOT digested: each is a pure function of the hashed prop rows
  and the static map, rebuilt at slot 6 every tick, like the swarm's count
  plane ("not digested (a function of the store)", `.claude/rules/swarm.md`).
  The existing digested stamp planes (`obstacles`, `dyn_heat_atten_q`,
  `dyn_light_atten_q` per `tests/field_digest_spec.toml`) will change on a
  prop-bearing level, which is correct and moves no golden for the reason
  above. `dyn_permeability` and `dyn_wave_absorb` are float planes excluded
  from the digest (`tests/field_digest_spec.toml` `excluded_float_fields`).
- The recorder (`src/simulation/recorder.py`) gains the prop columns
  additively in the patch that wants them in recordings (not required here).

---

## 3. The stamp: an integer rule for a turned rectangle

### 3.1 What the stamp must do

Every tick, each prop's rectangle at its pose becomes tile coverage that the
tick's readers see: gas permeability, blast absorption, heat and light
extinction (the four dynamic planes a unit already stamps,
`GameMap._stamp_units_python`), and movement. The rule must be exact
integers, a pure function of the pose (so order-free), the same bits on the
CPU and the device, and cheap.

### 3.2 The rule

Notation: `ONE = 2^16`. The prop's rectangle centre `G = (X, Y)` in Q16
tiles, its half sides `A` (along the heading) and `B` (across) in Q16 tiles,
its heading's cosine and sine `c = cos_q16(th)`, `s = sin_q16(th)` from the
trig kit (`cpp/src/fixed_point.h` `cos_q16` / `sin_q16`, Python
`unit_fixed.cos_q16` / `sin_q16`), each a Q16 integer in `[-ONE, ONE]`.

1. **The sample lattice.** Each tile `(i, j)` (column, row) holds `S x S`
   sample points, `S = 4`, at offsets `o_p = (2p + 1) ONE / (2S)` for
   `p = 0..3`, i.e. 8192, 24576, 40960, 57344 (exact integers). A sample's
   position is `P = (i ONE + o_p, j ONE + o_q)`. The lattice is GLOBAL: every
   sample belongs to exactly one tile.
2. **The frame.** `dx = Px - X`, `dy = Py - Y` (Q16), then
   `u = dx c + dy s` and `v = -dx s + dy c` (Q32, int64).
3. **Inside.** A sample is inside iff `|u| <= A ONE` and `|v| <= B ONE`
   (closed on both ends, so the rule is symmetric under `(c, s) -> (-c, -s)`).
4. **Coverage.** `k(i, j)` = the number of inside samples of tile `(i, j)`,
   an integer in `0..16`.
5. **The tiles visited.** Only the bounding box:
   `Rx = ((A |c| + B |s|) >> 16) + 2`, `Ry = ((A |s| + B |c|) >> 16) + 2`
   (Q16 tiles, the `+2` a pad for the kit's `c^2 + s^2 != ONE^2`), tiles
   `(X - Rx) >> 16 .. (X + Rx) >> 16` by `(Y - Ry) >> 16 .. (Y + Ry) >> 16`.
6. **The burn cells.** The same samples split half-open by the signs of `u`
   and `v`: front `u >= 0`, back `u < 0`, right `v >= 0`, left `v < 0`. So
   `k = k_FR + k_FL + k_BR + k_BL` exactly, per tile.
7. **The blocking set.** Tiles with `2 k >= S^2` (at least half the samples).

Nothing else: no division, no rounding, no float, no trig inside the rule
(the kit is called once per tick per awake prop, for `c` and `s`).

Integer range: `|dx| <= 2^28` on a 4096-tile map, `|c| <= 2^16`, so
`|u|, |v| <= 2^45`, far inside int64 (the device's `int64_t` arithmetic,
`cpp/src/fixed_point.h`'s house types). The largest intermediate seen in the
check below is 2^34.4 on 60-tile maps.

### 3.3 What coverage means for each reader

Each class row names ONE `[materials.*]` row, and the prop stamps that row's
values, scaled by coverage where a reader can take a fraction. This is the
"Prop tile stamps" rule's spirit ("a prop's blocking + fuel are ONE
`[materials.*]` row ... never a per-prop if or a new per-tile flag",
CLAUDE.md), applied to the dynamic planes instead of `material`.

The 17 values per plane per class (`k = 0..16`) are built ONCE per map at
the class door, in integers, and handed to every stamp site as table rows:

| plane | value at coverage k | combine with the plane | why graded or binary |
|---|---|---|---|
| `dyn_permeability` (float32) | `p_k = (ONE - ((ONE - p_row_q) k) // 16) / ONE`: an exact integer over 2^16, so the float32 is exact on every machine; refused below the floor `P_PROP_MIN` (section 3.5) | MIN with the current value (after the units' write) | graded: gas sees a smooth porous body as it slides; no flicker at a tile edge |
| `dyn_wave_absorb` (float32) | `a_k = (a_row_q k // 16) / ONE`, snapped to 0 inside any band the solver forbids (`config.toml` notes a forbidden `(0, 0.02)` band on air; UNVERIFIED whether it binds a dynamic stamp) | MAX | graded |
| `dyn_heat_atten_q` (int32 Q16) | `h_row_q k // 16` | MAX | graded: a half-covered tile is half opaque |
| `dyn_light_atten_q` (int32 Q16, x3) | `l_row_q[c] k // 16` | per-channel MAX | graded |
| `prop_mob_cap` (int32, NEW) | the row's `mobility + 1` where `2k >= 16`, else 0 (no cap) | MIN over nonzero caps | binary: a walker either enters a tile or not |
| `prop_owner` (int32, NEW) | the prop's row + 1 where `2k >= 16` | written; two props never share a blocking tile (section 4.5) | binary |

Readers of the new planes:
- **Movement.** A tile's effective mobility is the static one, or
  `min(static, cap - 1)` where `prop_mob_cap != 0`. Read by
  `GameMap.is_passable_block` (the step) and `GameMap.footprint_mobility`
  (the speed through `movement.footprint_samples`), so a walker climbs a
  table at its material's penalty (furniture: mobility 400, "climbable at a
  penalty: 40% speed", `config.toml` `[materials.furniture]`) or is stopped
  by a row with mobility 0. With no prop on the map the plane is all zero and
  both readers take an early-out, bit-identical to today.
- **Planning (A*, the horde field).** Read terrain only, as today. A path
  may run through a prop; the step is blocked and the walker slides or waits.
  ACCEPTED GAP: planners do not see props (a plane that changes every tick
  would invalidate the horde field cache every tick).
- **Doors.** A door does not close on a prop: `door_system._occupancy_clear`
  also asks `prop_owner` over the span (and its device twin in the applier's
  DOORS stage, same patch, `.claude/rules/sim-core.md` Door table).
- **Sight, rounds, grenade flight, the spray cone:** NOT in #226. They read
  `solid`, `sight_screen` and the flight plane today
  (`.claude/rules/sight.md`, `.claude/rules/device-units.md` Grenade and charge
  slots). #227 decides each against the material row's own columns
  (`blocks_sight`, `blocks_thrown`, `cover_exposure`) read through the
  blocking set. ACCEPTED GAP until #227.
- **`obstacles` stays static.** A prop is a porous body, like a unit, never a
  hard wall: the solvers' hard-zero boundaries key off `obstacles`
  (`GameMap._stamp_units_python` docstring), and a prop must not create them.

Order on the stamp: the units' rows first, exactly as today (the unit
permeability WRITE whose last row wins, `cpp/src/cuda_stamp.cu` header), then
the props' rows, each a MIN or MAX against the plane as it stands. MIN and
MAX commute, so the props' result does not depend on their order, and the
units' arithmetic is untouched, so a prop-free tick is byte-identical.

### 3.4 Sleep and the stamp

A sleeping prop (section 5) does not recompute its coverage: its last rows
(tile, k, quadrant split) are cached and re-emitted. The cache is not synced
state: it is a pure function of the pose, recomputed on wake, and a gate
holds cached == fresh (section 11).

### 3.5 The gas in a tile a prop moves into or out of

**DECISION: nothing moves.** A prop, like a unit, is a porous body on the
dynamic planes. The gas in a tile stays in it when a prop arrives or leaves:
no N, no energy, no trace moves, so the mass and energy books close
trivially (nothing is written to them). This is the units' rule today
("units do NOT push atmosphere as they walk", `GameMap.stamp_units`
docstring).

To keep that honest a prop never seals a tile: the class door refuses a
material row whose permeability is below `P_PROP_MIN` (proposed 0.05; the
exact floor is the patch's measurement). A zero-permeability dynamic tile
over a gas cell would leave a cell the solvers still count but no face
reaches (the inverse of the door-stamp leak named in
`GameMap._stamp_units_python`); refusing it avoids the question instead of
answering it. UNVERIFIED that a zero would misbehave; the refusal costs
nothing either way.

**Alternative it beat:** the prop displaces its volume of air (#220's
question). A furniture row is 3.24% of its tile's volume at 10.8 mm board
(`config.toml` `[materials.furniture]`, M2 note), so the displaced air is
small, and doing it exactly means moving N and energy through the gas energy
seam every tick a prop moves (`GameMap.gas_energy_move`, booked), with a
device twin. ACCEPTED GAP: a moving prop displaces no air. A solid heavy prop
(a fridge) is where the gap is largest; it is still a porous body to the gas.

### 3.6 The check (run, pure Python)

Script: `C:/tmp/props226/stamp_check.py` (not in the repo); its output is
`C:/tmp/props226/stamp_check_result.txt`. It implements 3.2 verbatim. The
trig values `c, s` are `round(cos x 2^16)` stand-ins on the harness side
only: the claims are about the rule for ANY integer `(c, s)`, which never
calls trig. 400 random poses, sides 1 to 8 tiles, any heading:

```
random poses: 400 (sides 1..8 tiles, any heading, S = 4)
  bbox         failures: 0    a scan 3 tiles wider finds nothing more
  order        failures: 0    tiles visited in a shuffled order: same map
  translate    failures: 0    an integer-tile shift shifts the stamp exactly
  quadrant     failures: 0    the four burn cells sum to k on every tile
  pi_symmetry  failures: 0    (c, s) -> (-c, -s): identical stamp
  mirror       failures: 0    x mirrored about a tile line, s negated: mirrored stamp
  empty_block  failures: 1    (a side near 1 tile; see below)
  block_4conn  failures: 6    (sides near 1 tile; see below)
  block_8conn  failures: 0
  any_4conn    failures: 0    the set of tiles with k > 0 is always 4-connected
  worst |coverage area - true area| / perimeter: 0.0602 tiles
  worst relative area error: 0.1238 (the smallest props)
  largest |intermediate| seen: 21988618672 (= 2^34.4)
  point tests per prop (bbox x S^2): min 64, median 768, max 2304
axis-aligned 4x2 block on tile lines exact: True
bound on a 4096-tile map: |u| <= 2*268435456*65536 = 2^45 < 2^63
min side (tiles) -> poses of 300 with an EMPTY blocking set:
  {0.5: 128, 0.75: 29, 1.0: 2, 1.5: 0}
  min side 1.0: blocking set not 4-connected in 113 of 300
  min side 1.5: blocking set not 4-connected in 0 of 300
  min side 2.0: blocking set not 4-connected in 0 of 300
  min side 3.0: blocking set not 4-connected in 0 of 300
```

What it shows:
- The rule is exact, order-free, translation-covariant, symmetric under a
  half turn and under mirroring, and its bounding box is complete. These are
  the properties the CPU / device identity rests on: each tile is computed
  alone from integers, so any thread order gives the same bits.
- The burn cells partition the coverage exactly (no sample counted twice or
  lost), so a per-cell deposit weighted by `k_q` sums to the whole prop's.
- Coverage area is within 0.06 tiles per tile of perimeter of the true area.
- **The size floor.** At a smaller side of 1 tile the half rule leaves the
  blocking set empty in 2 of 300 poses and not 4-connected in 113 of 300 (a
  wall a unit could slip through diagonally, though no 3-tile walker fits).
  At 1.5 tiles it never fails. So the class door refuses a smaller side
  below 1.5 tiles (0.5 m at 0.333 m tiles). That is the line between a
  movable prop and debris (#230): a ruling for Erik (section 14, R1).

### 3.7 Worked example

A 4.8 x 2.4 tile table (1.6 x 0.8 m) at heading 30 degrees, centre
(12.4, 7.7) tiles: `X = 812646`, `Y = 504627`, `A = 157286`, `B = 78643`,
`c = 56756`, `s = 32768`. Coverage `k` per tile (columns 9 to 14, rows 5 to 9):

```
        9  10  11  12  13  14
    5   0   2   4   0   0   0
    6   0  11  16  10   1   0
    7   2  15  16  16  15   8
    8   0   1   9  16  16  12
    9   0   0   0   2  12   2
sum k / 16 = 11.6250 tiles^2, true area = 11.5199
```

Tile (12, 7) is fully covered, `k = 16`, split front-right 2, front-left 6,
back-right 2, back-left 6. Its first sample: `dx = -18022`, `dy = -37683`,
`u = -2257653176`, `v = -1548191452`, both inside the limits
`A ONE = 10307895296` and `B ONE = 5153947648`. The blocking set is the 11
tiles with `k >= 8`. On those tiles a furniture row stamps `prop_mob_cap =
401` (climbable at 40% speed) and permeability `0.5`; on tile (10, 5) with
`k = 2` it stamps permeability `1 - 0.5 x 2/16 = 0.9375` and no mobility
cap.

### 3.8 How it meets the existing stamp and the topology seam

- **`stamp_prop_tiles` (static props) is untouched.** It writes `material`
  once at load and never ticks (CLAUDE.md, Prop tile stamps). A movable prop
  never writes `material`, so it never calls `GameMap.on_tile_changed`, never
  bumps the tile serial, and never touches the topology seam or its device
  twin (`cpp/src/cuda_topology.cuh`). The one-material-row rule is kept: the
  movable prop also stamps exactly one row's values.
- **The stamp kernel.** A prop's rows join slot 6's rows. Today's row is
  `[env, y, x, perm bits, wabsorb bits, heat_q, l0, l1, l2]`
  (`GameMap.take_stamp_rows`); a prop row adds a MODE column (unit write vs
  prop min) and the mobility cap and owner. On the device a fourth kernel
  after `stamp_perm_write` applies the props' permeability as an `atomicMin`
  on the float's bit pattern (exact, because the permeabilities are
  non-negative floats and their bit patterns order as the values do), and the
  owner and cap planes as plain writes (one prop per blocking tile). "A new
  stamp output is a column of the rows, the kernel and the CPU stamp in ONE
  change" (`.claude/rules/device-residency.md`, Stamp kernel). The stamp
  kernel is Class P (the played resident tick runs it), so that patch merges
  only with `.cuda_green` at HEAD.
- **Where a tile changes under a prop** (a wall burned through next to it):
  the static baselines move through the seam as today; the prop's next stamp
  combines with the new baseline. A wall tile is never under a prop's
  blocking set (section 4.5), so a destroyed wall does not need to know
  about props.

---

## 4. The push law

### 4.1 What is shared with units, and what a prop adds

**DECISION: one impulse door for every body.** Everything that pushes a prop
hands it an impulse through the same function units use,
`unit_impulse.add(table, body, jx_q, jy_q, lever, twist)` (Unit impulse:
"THE way anything pushes a unit: an impulse through `add`, never a position
or velocity write"). `add` already works on any object with `imp_x`, `imp_y`,
`imp_l` and `impulse_books`; the prop store offers a row view with those
names. One change to share it: `angular_q` clamps the lever to the TABLE's
`r_body_q` (`unit_impulse.angular_q`); it gains the clamp from the body (a
unit passes `table.r_body_q`, so units stay bit-identical; a prop passes its
half diagonal).

Shared, called as they are:
- `unit_impulse.add`, `angular_q`, `lever_q`, `along_q` (the door and the
  lever arithmetic);
- `div_trunc` and the remainder discipline: `dv = J / m` truncated, the
  remainder stays in the accumulator, so momentum closes exactly
  (`unit_impulse.integrate` step 1, `books_identity`);
- `_shrink` and `_clamp_vmax` (made public in the patch, unchanged);
- the prone branch's arithmetic: "every input meets the prone friction at
  once and what is left moves the body" (`unit_impulse.integrate`, the
  `is_prone` branch) IS Coulomb friction. A prop is a body that is always
  prone, with its own friction cap;
- the wind read `GameMap.footprint_wind` (section 4.3) and the wind impulse
  form of `unit_impulse.wind_impulse_q`.

What a prop adds (its own module, `src/simulation/prop_motion.py`):
- the integration without sway, effort, reaction delay or knockdown;
- rotation about the centre of mass, with an angular friction cap;
- its own move: a turned rectangle against walls, units and props
  (section 4.5);
- the projected wind area (section 4.3);
- the contact rule that hands an obstacle body momentum (section 4.6).

**Alternative it beat:** a separate rigid-body solver for props (velocity
iterations, contact manifolds, restitution). Rejected: it would be a second
push law beside the unit's and a second set of collision rules, it is
iterative (order-dependent unless run to a fixed count with a fixed order),
and nothing in the milestone needs stacking or resting contact between
props.

### 4.2 The integration (slot 1a, after the units)

For each prop that is not asleep, in row order (each row reads only itself,
so the order changes nothing):

1. `dvx = div_trunc(imp_x, m)`, `imp_x -= dvx m` (and y); `dw =
   div_trunc(imp_l, I)`, `imp_l -= dw I`. The remainders stay.
2. `v += dv`, clamped to `V_MAX_PROP` (proposed equal to the unit law's
   `unit_impulse.V_MAX_Q`, 4 tiles a tick, so every square fits int64 as it
   does for units; booked `clamped`); `w += dw`, clamped
   below half a turn per tick (the turning law's `OMEGA_MAX_Q16` bound).
3. Friction, linear: `(vx, vy) = shrink((vx, vy), cap_lin_q)` (booked
   `resisted`). Angular: `|w|` shortened toward zero by `cap_ang_q`.
4. **Held this tick?** If after step 3 `v = 0` and `w = 0`, the remainders
   still in the accumulators are below one tick of friction by construction
   (each is below one count of velocity times the mass). They are booked as
   `resisted` and zeroed. So a prop that friction holds ends the tick with
   ALL impulse state zero, exactly, and is asleep (section 5). Without this
   step a remainder below `m` would sit in the accumulator forever and the
   prop would never sleep. The books still close: the remainder moves from
   the accumulator term to the resisted term of the identity.

Coulomb friction falls out: a steady force below `mu m g` adds `dv` below
`cap_lin` every tick and is shrunk to zero every tick (it never moves the
prop); above it, the prop gains `(F/m - mu g) dt` a tick and slides; once the
force stops it decelerates by `mu g` a tick and stops exactly at zero (the
shrink never overshoots). The hold scales with mass because the cap is a
velocity and the impulse is divided by the mass (Erik: friction "could
depend on the weight": it does, as `mu m g`).

ACCEPTED GAP: one coefficient for starting and for sliding (static =
kinetic); friction on the linear and the angular motion are applied
separately (a sliding box in reality turns more easily than a resting one).
Two coefficients is a ruling (section 14, R4).

### 4.3 Wind and blasts (slot 9c2, with the units)

- **The read.** THE read of what the air does to a body is
  `GameMap.footprint_wind` (CLAUDE.md, Wind load on a body: "never a second
  reader"). It reads a square block today. The patch splits it into the
  per-tile walk over a TILE LIST and the square form that calls it with its
  row-major tile list, so the unit read is the same function with the same
  tie rule ("the first such tile in row-major order on a tie") and stays
  bit-identical (a property test, section 11). A prop calls the tile-list
  form over its covered tiles (`k > 0`) in row-major order.
- **The force.** The impulse is the wind's dynamic pressure on the area the
  prop shows the wind: height times projected width, `w_proj = 2A |vx s -
  vy c| + 2B |vx c + vy s|` over `|v|` (Q16 tiles; `|v|` the kit's `isqrt`;
  one truncated division). `J = ((W >> 16) wind_kw_q w_proj) >> 32`, along
  the wind, through `add` with the lever from the centre of mass to the
  rectangle's centre `G` (the centre of pressure). So the wind turns a prop
  whose mass sits off centre (Erik: "an off-centre mass makes a push at one
  end turn the prop"), and does not turn a balanced one.
- **The blast.** "A blast pushes only through its own wind" (Generic
  explosion, `.claude/rules/explosions-fire.md`). A detonation moves props
  through this row and nothing else: no blast impulse is added for props, so
  nothing is counted twice.
- **Self-shielding.** The prop's own stamp lowers permeability on its tiles,
  so the wind it reads on them is the wind inside a porous body. Units have
  the same property today (they read their own stamped footprint). ACCEPTED
  GAP; the size of the effect is UNVERIFIED (a measurement for the patch:
  the wind on a prop's tiles against the wind on the same tiles with the
  prop absent).

### 4.4 A unit shoves a prop (slot 1b)

The shove verb (`unit_impulse.shove`, `shove_target`) chooses among units
today. It also considers props whose rectangle is adjacent within the cone,
by the same bearing rule and a deterministic tie (units before props at an
equal bearing, then the lower id). The shove's impulse goes through `add`
with the lever from the prop's centre of mass to the line of the shove, so
shoving one end of a table turns it. ACCEPTED GAP (inherited, Unit impulse):
the shover feels no reaction. Walking into a prop does not push it in #226
(only the shove does; carrying is #228): a ruling (section 14, R2).

### 4.5 The move (slot 3p, after the units' slot 3)

The prop's displacement is one call line after the units' displacement loop
(the Tick conductor: "a new system = one call line"). Everything it tests
against is the state at the START of the slot, so the result does not depend
on the order of the props.

1. **Substeps.** `k = ceil(max(|vx|, |vy|, R_max |w|) / (ONE - 1))`, so no
   corner moves a whole tile in a substep (the `unit_impulse.move` rule,
   `SUBSTEP_Q`, extended to rotation; `R_max` the half diagonal).
2. **Each substep** tries the full step (translate and turn), else the
   translation alone (the turn is dropped: `w = 0`, booked `wall_ang`), else
   x alone, else y alone, else stop; the axis not taken is INTO THE OBSTACLE:
   its velocity component is zeroed and booked (`wall`, or `contact` when the
   obstacle is a body, 4.6). The unit move's slide rule, verbatim in shape.
3. **The test** of a candidate pose: its blocking set (3.2 step 7) must hold
   only tiles that are
   - prop-enterable in the static map: open floor (the static prop stamp's
     own base rule, `prop_system._STAMP_OK_BASE_MATERIALS`, today `MAT_AIR`),
     not vacuum, not the ambient ring, inside the grid (so a closed door,
     a wall, a grille, a static crate tile all block);
   - `prop_owner` 0 or this prop (the start-of-slot plane: no other prop's
     blocking set);
   - not newly inside a living unit's footprint: a tile in the candidate set,
     not in the prop's own start set, inside a living unit's
     `occupied_tiles()` after slot 3 blocks (so a unit standing ON a
     climbable table does not pin it forever, and a unit walked into by a
     prop stops it).
4. **The refusal pass.** Two props may each move into the same free tiles in
   one tick, because each tested only the start plane. After every prop has
   moved, each moved prop adds 1 to a scratch claim plane over its new
   blocking set (an integer sum: order-free; `atomicAdd` on the device). A
   moved prop with any new tile claimed twice is REFUSED: it returns to its
   start pose and its velocity is zeroed and booked (`contact_lost`). A
   refused prop goes back to tiles it held at the start, which no other prop
   could have entered (they tested the start plane), so one pass suffices.
   Everything is decided against the start state and applied together: the
   simultaneous-attacks principle ("an attack slot decides against its START
   state and applies together", `.claude/rules/combat.md`) applied to props.

Cost: the move recomputes coverage per substep for the moving props only.
ACCEPTED GAPS: no bounce (a prop that hits a wall stops on that axis, the
grenade bounce is not borrowed); two fast props whose paths cross inside one
tick's substeps but whose end poses do not overlap pass through each other;
a prop cannot enter space (the hull stops it: a breach does not blow
furniture out, section 9 "not decided").

**Alternative it beat:** moving props one after another, each seeing the ones
before it. Deterministic (fixed row order) but order-DEPENDENT: the device
would need it serial per env (the per-env serial applier), and a prop's fate
would depend on its ordinal. The start-state test plus one refusal pass is
order-free and runs one thread per prop.

### 4.6 What a prop hits: walls, units, props

- **A wall** (or any static non-enterable tile): the blocked component is
  zeroed and booked `wall`. Momentum leaves the books' bodies there, as for
  a unit (`unit_impulse.move`).
- **A unit** (the unit test of 4.5 step 3): the prop hands the unit the
  momentum a perfectly inelastic collision would share along the blocked
  axis, `J = div_trunc(m_p m_u v_axis, m_p + m_u)`, through
  `unit_impulse.add` (an integer sum, order-free, integrated in the unit's
  next slot 1a), with the lever from the unit's centre to the contact line.
  What the unit then does is the unit law's: sway, stagger, knockdown. The
  prop's blocked component is zeroed; the part of its momentum not handed on
  is booked `contact_lost` (the inelastic loss). So a sliding fridge shoves
  or floors a marine and does not hurt it (damage is #229). A ruling
  (section 14, R5).
- **A resting or slower prop** (blocked by the start plane): the same rule,
  `J` handed to that prop through `add`. So a crate slid into a crate pushes
  it on the next tick (a push chain moves one link a tick).
- **Two props that both moved into each other** (the refusal pass): both stop,
  no transfer. ACCEPTED GAP.
- **A contact event.** Every hard contact (wall, unit, prop) with the impact
  speed above a floor is recorded as a render-only event
  (`PropContactEvent`: prop id, the other's kind and id, the axis, the speed
  in Q16 tiles per tick), the hook #229 asked for. Not in the synced event
  digest; if a synced law ever reads it, it joins the digest in that patch
  (the swarm death event's rule, `.claude/rules/swarm.md`).

### 4.7 The momentum books

Per prop per axis, exactly, after every integration and move:

```
sum of J added == m (v + resisted + wall + clamped + contact_out + contact_lost) + imp_residual
```

`contact_out` is what this prop handed another body; the receiving body books
it as `J added`. Over every body in a contact, the handed momentum appears
once as out and once as in, so the sum over all props and units of
`(in - contact_in) - (out terms)` closes per tick. The angular books are the
same identity on `imp_l`, `w`, `I`. Gate in section 11.

---

## 5. Sleep

**DECISION: asleep is derived, not stored.** A prop is asleep when `imp_x =
imp_y = imp_l = 0` and `vx = vy = w = 0` (the `unit_impulse.at_rest`
predicate on the prop's columns). Section 4.2 step 4 makes a prop that
friction holds land there exactly.

What an asleep prop costs per tick:
- nothing in slot 1a (the integration's fast path) and nothing in slot 3p;
- its stamp: its cached rows re-emitted, no coverage recomputed (3.4);
- the wind read over its covered tiles at 9c2, always: the read is what
  decides whether it wakes, and a gate on the read itself would lose sums
  (two sub-threshold pushes from different sources must still add up); on
  the host it is one integer pass over the wind-load terms of the map plus a
  segmented maximum over the cached tile lists (section 8), not a Python
  loop per prop;
- for a flammable prop, the thermal read over its tiles at 9c (section 6).
  A row that is not flammable has no thermal state and no read.

What wakes it: any `add` that leaves a non-zero accumulator. If the push is
below friction, the next integration shrinks it to zero and books it, and
the prop is asleep again that same tick. Nothing else wakes a prop.

Why it is deterministic: the asleep predicate is a pure function of synced
columns, so every backend and every replay agrees on it tick by tick; the
cache it enables is a pure function of the pose, recomputed on wake, never
read across a pose change (section 11 holds cached == fresh).

**Alternative it beat:** baking sleeping props into the static baseline
planes (`permeability`, `wave_absorb`, the static extinction planes and a
static mobility) so a sleeping prop costs exactly zero in the stamp.
Rejected: those baselines change only through the topology seam and the
tile serial, so every wake and every fall asleep would be a topology event
with its device twin, and the device stamp rows would be refused on that
tick (`GameMap.take_stamp_rows` returns None when a tile changed).

**Dormancy of the whole system:** a level with no movable prop builds no
store, launches nothing, stamps no row and reads no plane: every path is
byte-identical to today (the presence gating of the swarm and the entity
sections).

---

## 6. Fire

### 6.1 The ruling, and the decision

Erik ruled (1.1): a burning prop heats what is beside and underneath it, and
the heat source moves with the stamp. So:

**DECISION: a prop owns its fuel and its burn state; tiles never hold a
prop's fuel or a prop's fire.** A burning prop is a heat source carried by
the prop: each tick each burning cell deposits heat into the tiles under it
and around it through THE heat deposit path, draws oxygen from those tiles,
and makes smoke there. When the prop moves, the deposit moves with it. The
floor keeps the heat it already received, so a sliding burning table leaves
a hot trail, not a fire.

**Alternative it beat:** the prop lends its fuel to the tiles it stands on
(writes `flammable`, `wall_hp` and `fuel_recip` into them each tick so the
tile combustion law burns them) and takes back what is left when it moves
on. Rejected: the tile `fire` intensity and `temperature` stay with the tile
when the prop leaves (a burning table would leave its fire on the floor,
exactly what the issue's second comment warned of), the lend and take-back
are topology-like writes every tick, and Erik's ruling is the heat source.

### 6.2 One side burning: four cells

**DECISION: every prop has exactly four burn cells**, the quadrants of its
own frame (front-right, front-left, back-right, back-left; 3.2 step 6), each
with its own temperature, fuel and burning flag. That is the smallest form
that lets "a table burn on one side and not the other side" along either
axis and one corner alone, and it gives #235 the per-part burn amounts it
asks for ("a burn amount per part, read from the state").

Fire spreads from cell to cell through the tiles, not through a conduction
law inside the prop: the four cells' tile sets overlap along the prop's
midlines (a tile covered by two quadrants), so a burning cell heats tiles
its neighbour also feels. The furniture row has conductivity 0 ("fire eats
it via flammability instead", `config.toml`), so no internal conduction is
the material row's own answer.

**Alternatives:** one cell (cheapest; the whole table burns or none: fails
Erik's question); two halves along the long axis (half the state; cannot burn
one long side); one cell per covered tile in the prop's frame (looks best,
but a variable-length member, a side table, and up to 30 cells per prop).
Four is the default; two or per-tile is a ruling (section 14, R7).

### 6.3 The laws, per burn cell

Each law reuses a row or a function that exists; none is a second fuel law
or a second O2 read.

1. **Heat capacity and fuel (load).** The cell's share is a quarter of the
   prop. Heat capacity from the material row's `density` and
   `specific_heat` (the row's own numbers; `[materials.furniture]` 555 kg/m^3
   and 1620 J/(kg K)) over a quarter of the prop's burnable mass. Fuel in the
   material row's own currency: a tile's fuel is `wall_hp` paid at
   `fuel_per_o2` hp per unit of O2 burned (`config.toml`
   `[physics.combustion]`, "P5.1 stoichiometric fuel consumption"; the
   per-tile `GameMap.fuel_per_o2_plane` from the table's `fuel_per_o2_q16`).
   A cell's fuel capacity is the hp a quarter of the prop's mass is worth at
   the row's own mass-to-hp rule (`materials.py`: `fuel_per_o2[mat] =
   hp[mat] / O2_UNITS_PER_TILE[mat]`, `O2_UNITS_PER_TILE = density V_tile /
   KG_FUEL_PER_N_O2`). So a 20 kg table holds the fuel of 20 kg of its
   material, not of its tile count. A row that is not flammable has fuel 0
   and no thermal state.
2. **Warming (slot 9c, with the units' heat row).** The cell's temperature
   moves toward what its tiles deliver: the felt temperature of its tile set
   (THE felt function, `exchange.felt_temperature_q`, generalized from a
   unit's footprint to a tile list the same way as the wind read, the unit
   path bit-identical) and the air over those tiles, by the same two-inflow
   step as a body's shell temperature (`body_temp.advance`; UNVERIFIED that
   its signature takes a heat capacity and a tile list without change; if
   not, the patch lifts the step into a shared function both call, never a
   copy). The cell's capacity sets its time constant, so a thin heavy prop
   warms slowly.
3. **Ignition (slot 9d, with the tiles' ignition).** A cell lights when its
   temperature reaches the material row's `ignition_temp` AND the oxygen
   over its tiles passes the same test a tile must pass to light
   (`combat.apply_temperature_ignition`'s mole-fraction gate against
   `o2_frac_ext`, read once through `combat.ignition_params`), AND it has
   fuel. Heat starts it, as every fire must ("A fire is started by
   DELIVERING HEAT", `.claude/rules/explosions-fire.md`): no writer sets
   `burning` without the temperature. Erik's "how easily it catches" is the
   material row's `ignition_temp` and its heat capacity, both existing
   columns: never a new per-prop flag.
4. **Burning (slot 9e(d''), right after the heat sources, its own call
   line).** A burning cell draws oxygen at the tile combustion law's own
   rate, `[physics.combustion] burn_rate` (N_O2 per second per burn site at
   full intensity), over each of its blocking tiles weighted by `k_q / 16`,
   capped by the O2 there (UNVERIFIED that the per-site rate at full
   intensity is the right single number to borrow; the patch derives it from
   the tile law's demand at intensity 1 and states the derivation). With the
   O2 drawn, `dO2`:
   - **fuel**: `fuel_q -= fuel_per_o2[mat] dO2` (the row's own exchange
     rate), floored at the tile law's `FUEL_FLOOR` idea, never below zero;
   - **oxygen**: `dO2` of O2 becomes inert N2 in the same tile (the tile
     combustion's own products: "converting consumed O2 into smoke
     (soot_yield) + inert_N2 (the remainder -- N_total is CONSERVED)",
     `config.toml` `[physics.combustion]`). Through a NEW gas seam
     primitive, `GameMap.swap_o2_to_inert(fy, fx, dn)`: bulk N is unchanged
     (bulk is O2 + N2, the gas energy field's denomination), so the cell's
     energy and temperature are unchanged, exactly; it books its O2 and N2
     movement in a named channel and asserts `dE = 0`. Its device twin lives
     in `cpp/src/cuda_topology.cuh` with the other gas-N primitives, in the
     same patch (`.claude/rules/fields-gas.md`, GameMap);
   - **heat**: `dQ = H_fuel x o2_potency x dO2` (the tile law's heat per
     unit O2, `[physics.combustion]` `H_fuel = H_FUEL_M 2^H_FUEL_SHIFT`;
     UNVERIFIED whether `H_bed` also belongs, the patch reads the tile
     law's deliver path), landed as ONE fixed-total tile deposit on the
     `heat` field (`field_edit.Region.TILES` + `payloads.deposit_over_tiles`,
     CLAUDE.md Fixed-total tile deposit) over the cell's tiles with weights
     `k_q`, plus the one-tile ring around the prop's blocking set on the
     cell's side with a fixed ring weight: "beside and underneath it";
   - **smoke**: `soot_yield x dO2` of smoke trace (`[physics.combustion]
     soot_yield`) as a second fixed-total deposit on the smoke plane over
     the same tiles (the trace path's booked deposit, `trace_deposit_books`).
     The swap above converts ALL of `dO2` to N2, so bulk N and energy are
     untouched; the smoke is MINTED on the trace plane, not taken out of the
     bulk. The tile law instead takes the soot share out of the bulk and
     sheds its energy with it (`cpp/src/combustion.cpp`, the Dalton split and
     `e_soot_shed_sum`). ACCEPTED GAP: a burning prop's products hold
     `soot_yield x dO2` more bulk than a burning tile's; the price is a swap
     with `dE = 0` exactly, instead of a second energy-shedding path.
5. **Going out.** A cell stops burning when its temperature falls below the
   material's `fire_T_ext` (`materials.fire_T_ext_from_ignition`:
   `ignition_temp - ignition_to_ext_delta`, the tile law's own extinction
   point), when the O2 test fails, or when its fuel reaches the floor.
6. **Burnt out.** When all four cells are at the floor the prop is spent: it
   stays where it is, still blocks, still can be pushed, its mass unchanged
   (ACCEPTED GAP: burning does not lighten it), drawn charred (#235). What a
   burnt-out prop becomes (ash, broken, removed) is #229 / #130: a ruling
   (section 14, R8).

### 6.4 The books

- **Energy.** Every joule a burning prop gives the world enters through the
  `heat` FieldEdit, which the temperature solver converts and books in its
  own group, exactly as every heat source's deposit is (CLAUDE.md, Heat
  source; `.claude/rules/energy-books.md`, the four groups; UNVERIFIED which
  counter a `heat` deposit on a gas tile lands in, the thermal solver's gas
  deposit or the solid one: the same as for a heat source on a floor tile
  today). The O2 swap moves no energy (`dE = 0`, asserted and booked). So the
  closure identity needs no fifth term: a burning prop is, to the books, a
  heat source whose output is set by its own fuel and the O2 it draws.
- **Chemical energy.** The prop's fuel is not an energy column, exactly as a
  tile's `wall_hp` is not. The identity checked instead is per tick and
  exact: `sum of fuel spent == fuel_per_o2 x sum of O2 swapped`, and `heat
  deposited == H x sum of O2 swapped` to the deposit's rounding (n/2 LSB over
  n tiles, the fixed-total deposit's own bound).
- **Mass.** O2 out equals N2 in, per tile, through the primitive's channel;
  the smoke is a booked trace mint (`trace_deposit_books`), not bulk (the
  ACCEPTED GAP in step 4).
- **When the prop moves.** Nothing is transferred: the fuel is the prop's,
  the heat already deposited is the tiles', and the next deposit lands where
  the prop now is. No mass, energy or fuel crosses a tile boundary because a
  prop moved, so nothing needs booking.
- **What the prop takes from the room.** Nothing. Warming reads the felt
  temperature and the air; it does not cool them. ACCEPTED GAP, the body
  temperature's own gap (`.claude/rules/units.md`, Body temperature: what
  warms a body changes the felt temperature or the air, never the body; here
  the reverse direction is missing too). A prop is not a heat sink.

**Alternative it beat (books):** the prop's thermal node as a booked member
of the closure identity (warming takes `dE` from the gas cells through the
seam, the prop holds int64 energy, a fifth reservoir with its own identity
like `solid_energy_books_sum`). Exact, and the physics lens may prefer it.
Rejected for v1 on cost: a new books reservoir, an exchange term in the
identity, its device twin, and the gates for all three, for a node whose
heat capacity is a few kJ/K (the furniture row: 7.8 kJ/K a tile,
`config.toml`). It can be added later without changing anything above.

### 6.5 Oxygen, if Erik says no

If Erik rules that a burning prop does not use the room's oxygen (section
14, R6), step 4's swap is dropped: no new seam primitive, no device twin.
The ignition and the going-out still read O2, so a prop still will not light
or keep burning without air, but a fire in a sealed room starves only by its
tiles' own fires.

---

## 7. Who it blocks

| what | in #226 | through | later |
|---|---|---|---|
| wind and gas flow | yes, graded by coverage | `dyn_permeability` (the solvers' face weighting, `GameMap._stamp_units_python` docstring: "slowed by the face = min(perm) flux weighting") | #227 judges whether it is enough of a shield |
| the blast wave | yes, graded | `dyn_wave_absorb` (MAX); the blast also pushes the prop through its wind | #227 |
| radiant heat | yes, graded | `dyn_heat_atten_q` (MAX), seen by the radiation sweep (Extinction planes) | |
| light | yes, graded | `dyn_light_atten_q` (MAX): props cast shadows | |
| walking | yes, binary | `prop_mob_cap`: the material row's mobility (furniture climbable at 40%, a mobility-0 row a wall) | R3 |
| doors closing | yes | `prop_owner` over the span | |
| the flamethrower | NO (passes through) | the spray cone stops on solid only (Spray slot) | #227 |
| rounds | NO | the march stops on `solid` (Sight rules) | #227, the row's `cover_exposure` |
| sight | NO | `has_los` reads `solid` and `sight_screen` | #227, the row's `blocks_sight` and the class height |
| thrown grenades | NO | the flight plane (`combat.flight_plane`) | #227, the row's `blocks_thrown` |
| other props | yes | the move's start plane and the refusal pass (4.5) | |
| carried as a shield | NO | | #228 |

The four NO rows are deliberate: each is a canonical system with its own
device twin and its own Class-P or Class-T gate, and #227 exists to decide
them with Erik. Until then a table is cover against wind, heat and light
but not against bullets. ACCEPTED GAP.

---

## 8. Cost

All figures here are ESTIMATES argued from measured numbers in the docs;
none was measured for this design.

**The scale.** The medium generated ship is 100 x 70 tiles with 86 fuel tiles
(`docs/tick_budget_2026-10-29.md`). Assume 20 to 60 movable props on such a
ship (UNVERIFIED: levelgen places none yet). A typical prop's bounding box is
48 tiles (768 point tests at S = 4, the check's median; 2304 at the largest
8-tile prop) and it covers 12 to 30 tiles.

**CPU, played tick.** The medium ship's tick is 19 to 21 ms, of which the
physics kernels are about 14 ms and the unit stamp 1.1 ms
(`docs/tick_budget_2026-10-29.md`, profile rows 3 and 10).
- Asleep (the normal case): stamp = re-emit cached rows (~20 rows a prop,
  appended to the unit rows: ~1,000 rows for 50 props, the same order as the
  unit stamp's ~100 rows today x 10, in numpy); wind = one int64 pass over
  the map's wind-load terms (7,000 cells) plus a segmented max over the
  cached tile lists; thermal for flammable props the same shape. Estimate
  0.3 to 0.6 ms a tick for 50 sleeping props, IF vectorized. A Python loop
  per prop per tick would cost several ms and is not acceptable: the host
  store is SoA for this reason.
- Awake (a blast just passed): coverage per awake prop per substep, ~768
  integer point tests, vectorized per prop (~30 to 60 us, UNVERIFIED);
  50 awake props for the few ticks after a blast: +2 to 3 ms on those ticks.
- Fire: per burning cell a deposit over ~10 to 20 tiles through the edit
  queue, the same order as one heat source.

**Device, batched training.** The resident EOS + trace tail costs about
0.169 ms per env-tick at N = 128 and the batched sweep 0.122 ms
(`docs/residency_benchmark_2026-11-07.md`, the per-env-tick table). Props at
N = 128 envs x 50 props x 768 point tests = 4.9 M integer point tests per
tick if every prop were awake, a few multiply-adds each: well under 0.1 ms
on the card (estimate), against about 37 ms per batched tick for those two
stages alone. Asleep props cost their cached rows. The kernels are one thread
per prop (integration, move) and one thread per (prop, tile) (coverage), all
order-free; the refusal pass is one integer `atomicAdd` per claimed tile.
The cost that matters on the device is not arithmetic but plumbing: the
stamp rows grow, the conductor gains call lines, and the books read gains
the props' counts (no new host sync: they ride the one books read,
`.claude/rules/device-units.md`, Device conductor).

**Erik's estimate holds**: the arithmetic is small beside the solvers. The
weight of the arc is the new synced state, its CPU and device twins, and the
laws that must see it (stamp, books, movement, doors), as Claude's first
reading said on the issue.

**The observation.** Observation spec v1 does not change, ever ("a new idea
is a new spec, never an edit", `.claude/rules/rl-agent.md`). On v1 a prop
reaches an agent only through what v1 already reads: the physics planes it
windows (`device_obs.WINDOW_PLANES`; UNVERIFIED which stamp planes are among
them) and the movement it experiences. A trained soldier cannot see a table
until a NEW spec (v2) adds a prop block (position, heading, size, the four
burn cells' state, the class) and a full-view entry. ACCEPTED GAP: no
training level should hold movable props until that spec exists, or the
agents learn in a world they cannot see.

---

## 9. The prop in the interaction table (#237)

Each cell: the rule and where it lives, "none, on purpose" with why, or "not
decided" with its issue. Read "prop to X" as what the prop does to X, "X to
prop" as what X does to the prop.

| kind | prop to X | X to prop |
|---|---|---|
| air and gases | Slows flow through its tiles: `dyn_permeability` graded by coverage, MIN (3.3). Displaces no air (3.5, ACCEPTED GAP). A burning cell turns O2 into N2 in its tiles (6.3, `swap_o2_to_inert`; R6). | Pushes it: the wind load over its covered tiles through THE read and `add` (4.3). |
| smoke (trace gases) | Slows smoke as it slows air (the trace rides the bulk flux, Trace transport). A burning cell makes smoke (6.3). | None, on purpose: smoke has no momentum or effect on a body beyond the air that carries it. |
| heat (temperature, radiation) | Shadows radiant heat: `dyn_heat_atten_q` (3.3). A burning cell deposits heat beside and under it (6.3). | Warms its four cells through the felt temperature and the air (6.3 step 2). A prop does not cool the room (ACCEPTED GAP). |
| fire (tile fire) | None directly: the prop never writes `fire`; its heat can light a flammable tile through the ordinary ignition. | A tile fire under or beside it heats its cells, which may light (6.3). |
| water | None in this arc. | Not decided: #236 (density is stored now; pushing, floating, a soaked prop that will not light). |
| walls (solid tiles) | None: a wall is never under a blocking set. | Stops it on the blocked axis, booked `wall` (4.6); no bounce (ACCEPTED GAP). |
| doors | Holds a door open: a door does not close on a prop's blocking set (3.3, `door_system._occupancy_clear` + its device stage). | A closed door is not enterable (static `MAT_DOOR_CLOSED`); an open doorway is (air). A door closing does not push a prop (none, on purpose: it refuses to close instead). |
| glass | None: glass is a wall-like row to the move (not prop-enterable). | Stops it like a wall. A prop thrown through glass breaking it: not decided (#229). |
| units | Blocks or slows walking through the mobility cap (3.3, R3). Hands a unit it hits momentum (4.6, R5). Contact event recorded (#229 reads it later). | A unit's footprint blocks the prop's move (4.5); the shove verb pushes it (4.4); carrying: not decided (#228); walking into it: does not push (R2). |
| corpses | None, on purpose: corpses are not stamped and do not block (`door_system._occupancy_clear` "corpses never block"; Corpse law ACCEPTED GAP), so a prop slides over a corpse. | None, on purpose: a corpse does not block or push a prop. |
| swarm units | None, on purpose in this arc: swarm laws read fields and probe `solid` (Swarm shared kit); a larva crawls under and over furniture. A prop crushing larvae: not decided (#229). | None, on purpose: a swarm unit has no push (the swarm meets the unit side only through planes). |
| bullets (rounds) | None in #226: rounds pass through (ACCEPTED GAP). Not decided: #227 (the row's `cover_exposure`). | None in #226. A round pushing a prop (as a hit pushes a unit, `ammo_push_q`): not decided (#227). Damage: #229. |
| grenades and charges | None in #226: a grenade flies through (ACCEPTED GAP); #227 (`blocks_thrown`). | Their blast pushes it through the blast's wind (4.3). A charge planted on a prop: not decided (#227/#229). |
| blasts | Damps the wave behind it: `dyn_wave_absorb` (3.3). | Pushes it through its own wind only (4.3, Generic explosion). Its heat may light it (6.3). Breaking it: #229. |
| light | Casts shadows: `dyn_light_atten_q` (3.3). A burning prop's room light: render-only, follows it (#235, the fire room light). | None, on purpose: light carries no momentum and no heat beyond the radiation sweep's heat channel (covered under heat). |
| noise | Not decided: a hard contact making a noise through `noise.emit` (loudness by mass and speed) is #229's natural partner; the contact event exists for it. | None, on purpose: noise moves nothing. |
| vents and other entities | None, on purpose: a vent's flux is the gas primitives', which the prop's permeability already slows; a prop on a vent's tile does not seal it. A prop over a sensor or a pressure plate: not decided (no such sensor reads bodies today). | A vent pushes it only through the wind it makes (4.3). A heat source heats it (6.3). |
| other props | Blocks their moves (start plane), hands a resting or slower prop momentum (4.6), both stop on a mutual overlap (ACCEPTED GAP). Its heat can light a neighbour. | The same, by symmetry. |
| space / vacuum | None. | Not decided: a hull breach blowing furniture out. In this arc the hull stops it (4.5). |

---

## 10. The build (not now)

Patches in order, each green on main by itself, each presence-gated so a
level without movable props is byte-identical to today. "Bit-identical"
below means the field digest, the entity section and the unit stream of the
pinned matches and the golden aggregate are unchanged, held by the existing
gates.

| # | patch | gates and oracle | goldens |
|---|---|---|---|
| P0 | **The shared reads, refactored with no behaviour change.** `GameMap.footprint_wind` split into a tile-list walk and the square form over it; `exchange.felt_temperature_q` the same; `unit_impulse.angular_q` takes its lever clamp from the body; `_shrink` / `_clamp_vmax` public. | Property tests: square form == tile-list form on every footprint of a random map (units); the full suite; the pinned matches' unit streams. CUDA: the device twins of the wind and felt reads (`unit_lib`) unchanged or changed in the same patch (UNVERIFIED which device functions mirror these two reads). | none |
| P1 | **The stamp rule as an executable reference.** `src/simulation/prop_stamp.py` (Python) and `cpp/src/prop_stamp.h` (FP_HD, on `SIM_HEADERS` in `tests/test_no_float_in_sim_tu.py`), the integer reference of section 3.2 that any later change lands in FIRST (the Integer reference precedent). | Section 11 properties 1 to 8 on both, and C++ == Python on random poses. | none |
| P2 | **The class, the entity, the store, the stamp, no motion.** `[movable_props.*]` rows with their refusing door; the `movable_prop` entity; `Simulation.props`; the entity record tail; slot 6 stamps props (Python and C++ stamp paths) into the four dynamic planes plus `prop_mob_cap` / `prop_owner`; movement reads the cap; the door occupancy reads the owner (host). A new test level with props (written by a generator through `level_lib`, never in a golden corpus). | Prop-free: byte-identical (the pinned matches, the golden aggregate, `tests/test_stamp_units_cpp_ab.py`). Prop-bearing: Python stamp == C++ stamp byte for byte (extend `test_stamp_units_cpp_ab`); movement and door properties (11). | none (no golden level holds a prop) |
| P3 | **The device stamp rows (Class P).** The row's mode, cap and owner columns; the fourth stamp kernel (`atomicMin` on bits, the plane writes); `take_stamp_rows` packs prop rows; the door stage's prop read on the device. | `tests/cuda_stamp_check.py` on prop scenes, every `cuda_s8a_check` leg (N = 1 == CPU, tol 0), `cuda_batch_invariance_check` (env i of N == alone, props in some envs only). `.cuda_green` == HEAD. | none |
| P4 | **Motion.** Slot 1a integration, slot 3p move with the refusal pass, 9c2 wind, the shove verb on props, the contact rule into `unit_impulse.add`, sleep, `PropContactEvent`. Host only. | Section 11 properties 9 to 16; the momentum books every tick in a scene suite (a blast in a furnished room, a shove, a slide into a wall, into a unit, into a prop, a crossing pair). Prop-free byte-identical. | none |
| P5 | **Fire.** The four cells' warming, ignition and burn; `GameMap.swap_o2_to_inert` with its device twin in `cpp/src/cuda_topology.cuh` (same patch, `tests/cuda_unit_topology_check.py` extended); the deposits through the edit queue. | Section 11 properties 17 to 21; the energy closure benches with a burning prop (`tests/test_e1_hot_rail.py`, `tests/_quiet_books_bench.py` scene added); prop-free byte-identical. | none |
| P6 | **The device conductor (Class T).** `unit_schema.PROP_TABLE`, its record, way-back and entity-row functions, `tools/gen_unit_layout.py --write`; kernels in `cpp/src/cuda_unit_props.cu` for 1a, 3p (with the refusal pass), 9c2, 9c, 9d and the burn (the burn's deposits through the device edit door, its swap through the topology seam); the conductor's call lines; the stamp rows from the table (`unit_kernels.stamp`). | The shadow lockstep (`tests/cuda_unit_lockstep.py`: rows tol 0, env i of N == alone, declared field outputs and books), the table's identity gate (`tests/cuda_object_table_check.py` pattern), `tests/cuda_conductor_check.py` and `cuda_conductor_batch_check.py` on prop scenes. A `[twins.movable_props]` Class-T row. | none |
| P7 | **Levels.** Levelgen places movable props (#171 job 1); the first golden level that holds them. | The level lint; the bot tournament for balance. | YES: that level's goldens, once, with written rationale. Not part of #226; listed so nobody slips it into P2. |

Order rationale: P0 and P1 change nothing and de-risk the rest; P2 before P3
because the CPU stamp is THE oracle (`.claude/rules/device-residency.md`);
P3 before P4 so the played resident tick never sees host prop planes it
cannot build; P5 after P4 because fire needs the cells' tiles from a pose
that can move; P6 last (Class T carries debt and blocks only the device
training door, `.claude/rules/device-residency.md` Class P / Class T). #227
starts after P4; #228 after P4 and the verb gates.

The CUDA lockstep: P3 is Class P (the stamp kernel serves the played resident
tick), so it needs `.cuda_green` at HEAD; P5's topology primitive changes
`cuda_topology.cuh`, a Class-T device form whose gate is
`tests/cuda_unit_topology_check.py`; P6 is Class T.

---

## 11. The verification plan

Properties, never snapshots (CLAUDE.md, Test conventions). Each names what it
protects and the change that must break it.

The stamp (P1):
1. **Pure and order-free:** coverage of a tile is the same whether tiles are
   visited in row-major, shuffled or bounding-box order. Breaks if the rule
   gains running state.
2. **Bounding box complete:** a scan three tiles wider finds no tile with
   `k > 0`. Breaks if the pad is cut below the kit's `c^2 + s^2` slack.
3. **Translation covariance:** an integer-tile shift shifts the map exactly.
4. **Half-turn and mirror symmetry** (as in 3.6). Breaks if a bound becomes
   half-open on the whole prop.
5. **Burn cells partition the coverage** exactly per tile. Breaks if a
   quadrant bound becomes closed on both sides.
6. **Area bound:** `|sum k / 16 - 4AB| <= 0.07 x perimeter` over random
   poses (measured 0.0602). Breaks if the lattice or the inside test drifts.
7. **The size floor:** for every smaller side at or above 1.5 tiles the
   blocking set is non-empty and 4-connected over random poses; the class
   door refuses below it. Breaks if the floor is lowered without a new rule.
8. **CPU == device == Python** on random poses, bit for bit.

The stamp on the map (P2, P3):
9. **No prop, no change:** every plane and every movement answer equals
   today's on every level of the suite.
10. **Order-free combine:** permuting the props' rows leaves every plane
    unchanged.
11. **Cached == fresh:** a sleeping prop's re-emitted rows equal a fresh
    computation at its pose, every tick of a scene.
12. **Never seals:** no prop-stamped permeability is below `P_PROP_MIN`.

Motion (P4):
13. **Momentum books close** per prop per axis and angularly, every tick, and
    the contact transfers balance across bodies (4.7).
14. **Friction holds below the threshold:** a steady force below `mu m g`
    never moves a prop and leaves it asleep (all impulse state zero) at the
    end of every tick. A force above it moves the prop, and when it stops the
    prop comes to rest exactly. Breaks if step 4 of 4.2 is removed.
15. **Order-free move:** permuting the props' rows leaves every pose
    unchanged, including scenes where two props race for the same tiles.
16. **Never overlaps:** after every move no two props share a blocking tile,
    no blocking tile is a wall, a closed door or out of the map, and no prop
    newly overlaps a living unit's footprint.
17. **Mirror physics:** a mirrored wind on a mirrored furnished room gives the
    mirrored poses (the truncation is sign-symmetric). Breaks if a rounding
    becomes a floor.
18. **The wind turns only an off-centre prop:** zero spin with the centre of
    mass at the rectangle's centre, a spin of the expected sign otherwise.

Fire (P5):
19. **Heat starts it:** no cell burns without its temperature at its row's
    ignition point and the O2 test passing; a fixture lights a prop only by
    delivering heat.
20. **Fuel and heat identities** per tick (6.4): fuel spent equals
    `fuel_per_o2` times O2 swapped; heat deposited equals `H` times O2
    swapped within the deposit's rounding; the swap moves no energy.
21. **The energy closure identity holds** with burning props (the existing
    benches with a prop scene added).
22. **One side burns:** heating one end of a table lights that end's cells
    first and the far cells later or never (with the heat cut).
23. **Moving moves nothing:** a burning prop slid across a room transfers no
    fuel, mass or energy at the tile boundaries it crosses (only its
    deposits land), and the floor it left keeps its heat.

The device (P3, P6): the existing harnesses' properties (tol 0, env i of N ==
alone, the write-set gate, the declared outputs and books), never a
hand-rolled compare.

---

## 12. Accepted gaps

- `ACCEPTED GAP:` a moving prop displaces no air; gas stays in the tiles it
  enters and leaves (3.5).
- `ACCEPTED GAP:` sight, rounds, grenade flight and the flamethrower's cone
  pass through props until #227 (7).
- `ACCEPTED GAP:` planners (A*, the horde field) do not see props; walkers
  are stopped at the step (3.3).
- `ACCEPTED GAP:` one friction coefficient for starting and sliding; linear
  and angular friction applied separately (4.2).
- `ACCEPTED GAP:` no bounce off walls; a blocked axis stops (4.5).
- `ACCEPTED GAP:` two props that both moved into each other stop with no
  momentum exchanged; two fast props whose paths cross inside one tick may
  pass through each other (4.5, 4.6).
- `ACCEPTED GAP:` the hull stops a prop; nothing is blown into space (4.5).
- `ACCEPTED GAP:` the wind a prop reads is the wind inside its own porous
  stamp (4.3), as for units.
- `ACCEPTED GAP:` the shover feels no reaction (inherited, Unit impulse).
- `ACCEPTED GAP:` a prop warms from the room without cooling it; its thermal
  node is outside the closure identity (6.4); what it gives the room when
  burning is booked.
- `ACCEPTED GAP:` burning does not lighten a prop; a burnt-out prop stays
  (6.3, R8).
- `ACCEPTED GAP:` a burning prop's smoke is minted on the trace plane while
  all its O2 becomes N2, so its products hold more bulk than a burning
  tile's by the soot share (6.3 step 4).
- `ACCEPTED GAP:` no damage from impacts, and no integrity column, until
  #229; the contact event is recorded and unread (4.6).
- `ACCEPTED GAP:` water does nothing to a prop until #236 (9).
- `ACCEPTED GAP:` the trained agent cannot see props until a new observation
  spec; no training level holds props before it (8).
- `ACCEPTED GAP:` swarm units ignore props (9).
- `ACCEPTED GAP:` a level's set of movable props is fixed at load (no spawn,
  no break-up into smaller props) until #229 brings stable-slot spawning.

---

## 13. Systems

### 13.a Existing canonical systems the design must use (and must not
parallel)

| system (CLAUDE.md index) | how this design uses it | the parallel it must not build |
|---|---|---|
| Simulation facade, Tick conductor | the prop laws are call lines in slots 1a, 1b, 3p, 6, 9c, 9c2, 9d, 9e(d'') | no logic block in `step()`; no second tick entry |
| GameMap | the new planes `prop_mob_cap`, `prop_owner` live there | no plane outside the map |
| FieldEdit, Fixed-total tile deposit | a burning cell's heat and smoke are TILES deposits through the edit queue | never `gmap.heat[...] +=`, never a twin of `deposit_heat` / `_claim_disc` |
| Gas energy seam, Gas pump primitives, Device topology seam | the O2 swap is a new seam primitive beside `inject_gas_n` / `extract_gas_n`, twinned in `cuda_topology.cuh` | never a bulk write outside the seam |
| Energy closure identity, Temperature solver | a prop's heat enters as a booked `heat` deposit | no fifth group, no `temperature[...] =` |
| Starting a fire, O2 pressure factor | a cell lights by heat, under the tile's O2 test | no `burning = 1` without heat; no new O2 read |
| Material / gas tables | a class names ONE material row; every blocking, thermal and fuel value is that row's | no per-prop material if, no per-class copy of a row value |
| Prop tile stamps | static props unchanged; the movable prop extends its one-row principle to the dynamic planes | movable props never write `material` |
| Unit impulse, Coupling table, Wind load on a body | THE `add`, the wind read, the shove; a push on a prop is an impulse | no second push law, no second wind reader, no position write from a force |
| Footprint samples builder | the mobility cap reaches the walk's speed through `footprint_mobility` | no second speed reduction for props |
| Turning law (its wrap) | the prop heading is wrapped by `turning.normalize_q16` | no second wrap |
| Fixed-point kits, Q16 boundary modules | `cos_q16` / `sin_q16`, `div_trunc`, `isqrt`, quantize once at the class door | no re-derived shift, round or reciprocal |
| Stamp kernel | prop rows are rows of THE stamp; a fourth kernel in the same TU | no second stamp kernel |
| Object tables, Device tables, Device unit library, Device conductor, Per-env serial applier | `PROP_TABLE`, its rows; the props' values as table rows; the conductor's call lines | no second table mechanism, no law constant as a kernel argument |
| Entity system, Entity section export | `movable_prop` is an entity class; its record tail through the one serializer | no second serializer, no `__props__` carrier |
| Felt temperature (Coupling table), Body temperature | the cells warm through THE felt function and the body step | no copy of the felt chain |
| Heat source | a burning prop is a heat source in the books' eyes; its burn runs right after the heat sources | no second heat-deposit path |
| Field digest, Class P / Class T, CUDA harness, A/B lockstep | as in section 10 | no hand-rolled compare |
| Level data layer, Level generators | props are written through `level_lib`; a generator places them (P7) | no hand-written level.toml |
| Observation spec | v1 unchanged; props need a new spec | never a key added to v1 |
| FireVFX pass, Static props, Lit-3D seam (render) | #235 draws the flames on the moving prop through FireVFX; the prop's model through the static prop path | no second flame renderer, no prop render state on the sim store |

### 13.b New systems, each with a draft one-line rule and its rule file

1. **Movable prop store** (`src/simulation/movable_props.py`, `Simulation.props`,
   `entities/movable_prop.py`, `[movable_props.*]`): "THE state of movable
   props, a structure of arrays in entity ordinal order written only through
   this facade; a prop is never a `Unit`, never a swarm unit, never a
   material stamp; a new column is a schema row with its entity record row
   and its `PROP_TABLE` column in the same patch." Rule file: a new
   `.claude/rules/props.md` (paths: `src/simulation/movable_props.py`,
   `prop_motion.py`, `prop_stamp.py`, `entities/movable_prop.py`,
   `cpp/src/prop_stamp.h`, `cpp/src/cuda_unit_props.cu`, `tests/*prop*`), with
   a row in CLAUDE.md's region table.
2. **Prop stamp rule** (`src/simulation/prop_stamp.py` + `cpp/src/prop_stamp.h`):
   "THE integer rule that turns a turned rectangle into tile coverage (a 4 x 4
   global sample lattice, closed inside test, quadrants half-open); its Python
   form is the executable spec, any change lands there first, the C++ FP_HD
   form held equal; nothing else rasterizes a prop." Rule file: `props.md`.
3. **Prop motion** (`src/simulation/prop_motion.py`): "A prop moves only by its
   own integration (an always-prone body under Coulomb friction, its
   remainders booked when friction holds it) and its own move (start-state
   tests plus one refusal pass); every push reaches it through
   `unit_impulse.add`; contact hands momentum on through `add`, never a
   position write." Rule file: `props.md`, with a line added to
   `.claude/rules/units.md` (Unit impulse: `add` serves props; the lever clamp
   comes from the body).
4. **Prop planes** (`gmap.prop_mob_cap`, `gmap.prop_owner`): "Built by slot 6's
   stamp alone, every tick, never written elsewhere, not digested; movement
   reads the cap only through `is_passable_block` / `footprint_mobility`,
   doors read the owner only through `_occupancy_clear`; all zero on a level
   without props." Rule file: `.claude/rules/fields-gas.md` (GameMap) and the
   Stamp kernel section of `.claude/rules/device-residency.md`.
5. **O2-to-inert swap** (`GameMap.swap_o2_to_inert` + its twin in
   `cpp/src/cuda_topology.cuh`): "THE in-place conversion of O2 to burnt
   products outside the tile combustion pass; bulk N unchanged, `dE = 0`
   asserted and booked in its channel; a change lands in the topology seam in
   the same patch." Rule file: `.claude/rules/fields-gas.md` (GameMap, the
   gas-N primitives list) and `.claude/rules/device-units.md` (Device topology
   seam). Dropped if R6 is no.
6. **Prop burn cells** (in `movable_props.py` / `prop_motion.py`, slots 9c,
   9d, 9e(d'')): "A prop's fuel and fire live in its four cells, never in
   tiles; a cell lights only by heat under the tile O2 test, burns at the tile
   law's own rate and exchange rates, and gives the world heat and smoke only
   through fixed-total tile deposits." Rule file: `props.md`, with a line in
   `.claude/rules/explosions-fire.md` (Starting a fire: a prop cell is an
   igniter under the same rule).
7. **Prop table (device)** (`unit_schema.PROP_TABLE`, `cpp/src/cuda_unit_props.cu`):
   "THE device form of the prop store on the object tables; record function
   `unit_kernels.props_of`, way back `props_to_host`, entity rows
   `prop_entity_rows`; a store column that crosses a tick is a table column
   in the same patch." Rule file: `.claude/rules/device-units.md`.

Existing rules to amend when built: CLAUDE.md's "Prop tile stamps" line
("props never tick") becomes "STATIC props never tick; movable props are the
Movable prop store"; `.claude/rules/units.md` Wind load on a body (the
tile-list form) and Coupling table (felt temperature over a tile list);
`.claude/rules/sim-core.md` Door table (the prop owner read).

---

## 14. Rulings Erik must make (draft for #226)

One line each: the default this design assumes, and what changes otherwise.

- **R1, size line:** a movable prop's smaller side is at least 0.5 m (1.5
  tiles at 0.333 m); anything smaller is render-only debris (#230). Lower it
  and thin props need a different blocking rule that over-blocks.
- **R2, walking into a prop:** does not push it; only the shove verb pushes
  in #226 (carrying is #228). If yes: a walk-contact push, light props drift
  away from walkers, and bots must handle it.
- **R3, climb or wall:** the prop's material row decides (furniture is
  climbable at 40% speed like today's crate tiles; a mobility-0 row is a
  wall). If every movable prop must block walking: one class column forces
  mobility 0.
- **R4, friction:** one coefficient per prop, the same to start and to keep
  sliding. If "hard to get going, then slides": two coefficients and a little
  more state.
- **R5, a prop hits a unit:** it hands the unit momentum (a sliding fridge
  can shove or floor a marine) and does no damage until #229. If no: the prop
  just stops at the unit.
- **R6, oxygen:** a burning prop uses up the room's oxygen like a burning
  floor tile (sealed rooms starve it). If no: cheaper (no new gas
  primitive), but it burns to the end of its fuel in a sealed room once lit.
- **R7, one side burning:** four parts (front-left, front-right, back-left,
  back-right), each with its own temperature, fuel and flame. If two halves
  are enough: half the fire state; if per-tile: a larger state and a variable
  part count.
- **R8, burnt out:** it stays where it is, charred, still blocking and
  pushable, until #229 decides breaking. If it should vanish or turn to ash
  now: a removal path in this arc.
- **R9, training levels:** no training level holds movable props until an
  observation spec shows them to the agents. If Erik wants them earlier, the
  agents train beside things they cannot see.
