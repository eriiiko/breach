# Movable props: furniture as simulation state (#226)

2026-11-21. **Status: v2, three critiques folded; PROPOSED; nothing built;
arc 2 (fire) needs its own critique before its build.** Paper only: no
simulation code, no digest, no golden, no spec and no device table changes
with this document. Track B step 2 of the November push (#238). v1 is commit
`a647e771`; the three critics' reports are in
`docs/movable_props_critique_2026-11-21.md`; section 16 lists what v2 did with
every finding.

How to read it.
- Part A (sections 2 to 8) is ARC 1: a prop that exists, stamps, blocks,
  sleeps, and is pushed by wind, blasts and the shove, on the host AND the
  device in the same patches.
- Part B (section 9) is ARC 2: fire on props. Fully designed here, with a
  worked plateau calculation, but it is the least verified part and it gets
  its own critique round before anything of it is built.
- Part C (sections 10 to 17) is common: the interaction row and column, the
  build, the verification plan, the accepted gaps, the systems, the borrowed
  dials, the critique ledger and, LAST, the rulings list Erik answers.
- Each design section states a DECISION, the ALTERNATIVE it beat and why, and
  its COST. Claims about existing code carry `file:function`. What was
  inferred and not confirmed in the code is marked UNVERIFIED. Gaps the design
  accepts are marked `ACCEPTED GAP:` and collected in section 13. Sizes are in
  metres, kilograms and seconds; tiles appear only where the integer rule
  needs them, at the shipped tile of 0.333 m.
- Erik's rulings are R0 to R18 in section 17. The design assumes the default
  of each; the text says where a ruling changes it.

---

## 1. Erik's notes, and what the arcs are

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
  and not the other side." (answered in section 9.2)
- **Same note, new columns:** friction ("these furniture things will need some
  kind of friction coefficient. That could depend on the weight and
  everything, and it could be unique for each item"); centre of mass and
  momentum ("feels like small additions when we are designing this");
  density ("Perhaps we need to give all items a density so we know which ones
  float and which ones don't"). The sentence "an off-centre mass makes a push
  at one end turn the prop", which v1 quoted as Erik's, is Claude's gloss in
  the same comment, not Erik's words.
- **Same note, water:** props are pushed by water as well as wind, floating
  ones follow the flow; its own issue (#236), "done at the same time as this
  arc". Section 4.3 names the wind read this design builds as the read #236
  extends to water ("a fluid pushes a body", air and water as two rows), so
  the two arcs share one law instead of growing two.
- **Same note, the scene it all aims at:** "the aquarium bursts in mid-fight.
  Someone shoots the wall to the aquarium. Water is coming out, and also the
  big octopus, and it starts to fight and throw things around." (#111)

### 1.2 What the arcs are

Milestone 26, in Erik's order (issue body): (1) furniture in the state, pushed
by wind and blasts, able to burn (#226, this design); (2) it shields wind,
flamethrower, heat, then rounds and sight (#227); (3) units pick it up, carry
it and throw it (#228); (4) damage from impacts (#229).

**DECISION (pending R0): step (1) is built as two arcs.**
- **Arc 1** (Part A): a movable prop exists in the synced state, stamps the
  four dynamic planes and a walking cap, blocks walkers, doors and other
  props, sleeps at rest, and is pushed by wind, by blasts (through their
  wind) and by a unit's shove, with momentum kept in every contact. Host and
  device land together, patch by patch (section 11).
- **Arc 2** (Part B): a prop burns, one side at a time, as a heat source that
  moves with its stamp, with its fuel and its heat booked.

Erik put burning into step 1 himself (comment 1), so the split is HIS call:
ruling R0. What he gets under each answer:
- **Two arcs (the default):** the first build gives him tables and crates that
  slide in a blast, block wind, heat and light, hold a door open and can be
  shoved, on the played game and on the training device. Burning comes in a
  second build after arc 2's own critique. Arc 1 is about half the work, and
  it does not wait on the fire law's open questions (section 9.8).
- **One arc:** nothing lands until fire is designed, critiqued and built too;
  the first build is about twice as long, and the riskiest part (the fire's
  heat reaching the gas, section 9.6) gates everything else.

Every arc-1 decision leaves the hooks arc 2 needs (the burn cells' tile sets
are the stamp's quadrants, section 3.2 step 6; the state record has room for
the fire columns, section 2.1), and both leave hooks for #227 to #229.

### 1.3 What the arcs are not

- **Small debris stays render-only** (#230): cups, papers, cables. The line is
  drawn by size in section 3.6 and ruling R14: a thing whose smaller side is
  under 0.5 m (1.5 tiles at 0.333 m) cannot hold a stamp and is debris.
- **Damage to props and to thrown bodies** is #229. Arc 1 records every hard
  contact as a render-only event nothing reads, and gives the prop no hp
  column (R17).
- **Water on bodies** is #236. The prop stores its mass and box volume (so
  #236 has a density, R16); water does nothing to a prop in either arc.
- **The flame picture** is #235 (render-only, the FireVFX pass).
- **The octopus** (#111, #107) is what this prepares: a body that throws
  things needs things with mass, a pose and a push law.
- **Shields** (#227): sight, rounds, grenade flight and the flamethrower's cone
  pass through props in both arcs (section 7).

---

# Part A. Arc 1: a prop that exists, blocks and is pushed

## 2. The object: its state, and where it lives

### 2.1 The state, column by column

A movable prop is a rectangle on the floor with a pose, a mass and a material.
Its pose is planar: x, y and a heading, with their three rates (Erik's "3 or 6
numbers": the six of a top-down body; tipping is ruling R11). Every value is an
integer. Two kinds of column: LOAD constants (fixed after the level loads,
derived once from the class row and the level's tile size) and SYNCED columns
(moved by the laws, hashed).

**Load constants** (from the prop's class row and its instance overrides,
quantized once per map at the class door, like `unit_impulse.impulse_table`):

| column | integer type and currency | source and range (refused by name outside it) |
|---|---|---|
| `cls` | int, the class row's index | the level's entity |
| `mat` | int8, a `[materials.*]` id | the class row names ONE material row |
| `half_a_q`, `half_b_q` | int32, Q16 tiles | half length along the heading, half depth across it, from metres; each side at least `max(0.5 m, 1.5 tiles)` (section 3.6, R14), at most 8 tiles |
| `height_q` | int32, Q16 tiles | class row, metres; read by the wind area now, by sight and rounds in #227 |
| `mass_g` | int64, grams | class row or instance override; 1 kg to 2000 kg (section 4.8) |
| `com_u_q`, `com_v_q` | int32, Q16 tiles | centre of mass offset in the prop's frame (u along the heading); refused where `inertia_g` would fall below a tenth of the centred value (section 4.2) |
| `inertia_g` | int64, gram tile^2 | DERIVED about the centre of MASS: `m (L^2 + D^2) / 12 - m |o|^2` (the parallel-axis theorem, MINUS: v1 had plus), floored, at least 1 |
| `r_lever_q` | int32, Q16 tiles | DERIVED: the largest distance from the centre of mass to a corner; the prop's own lever clamp in `unit_impulse.add` (section 4.1) |
| `mu_q16` | int32, Q16 | friction coefficient, class row or instance override (Erik: "unique for each item"); 0.05 to 1.5 |
| `cap_lin_q` | int32, Q16 tiles per tick | DERIVED: one tick of Coulomb friction, `mu g / (tile_m tps^2)` (the unit law's prone-cap form) |
| `cap_ang_q` | int32, Q16 radians per tick | DERIVED: one tick of the floor's friction torque on a uniform rectangle over `inertia_g` |
| `vol_mm3` | int64, cubic millimetres | DERIVED: the box volume `L D H`; with `mass_g` it gives the enclosed density #236 may read (R16); read by nothing in arc 1 |
| `k_wind_q` | int64, Q16 | DERIVED: `unit_impulse._wind_k(rho_air, Cd x H_m x tile_m, wind_push_mult, tile, tps)`, the wind's impulse per unit of wind load per TILE of projected width; the area is in m^2, so the force in newtons does not depend on the tile size (section 4.3) |

Per-CLASS constants that are not per-instance (the 17-entry coverage-to-plane
tables of section 3.3, the class's `cd`) are Device tables rows on the device
(`unit_tables.ROWS`, read from the same validated class object the host law
reads), never `PROP_TABLE` columns and never kernel arguments
(`.claude/rules/device-units.md`, Device tables).

**Synced columns** (written only by the prop laws, hashed every tick in the
prop's entity record, section 2.3). Record row names are tokens (the
serializer's `_NAME_RE`, `entities/serialize.py`): `x_q`, never `x[0]`.

| column | integer type and currency | written by |
|---|---|---|
| `x_q`, `y_q` | int32, Q16 tiles | the centre of MASS (not the rectangle's centre); the move (slot 3p) |
| `th_q` | int32, Q16 radians in `(-PI_Q16, PI_Q16]` | the heading; the move, wrapped through `turning.normalize_q16` (the unit side's one wrap) |
| `vx_q`, `vy_q` | int32, Q16 tiles per tick | the integration (slot 1a) and the move's contact rules |
| `w_q` | int32, Q16 radians per tick | the integration and the move |
| `imp_x`, `imp_y` | int64, Q16 gram-tiles per tick | every pusher through the one `add` (section 4.1); the move's kept momentum (4.6); the integration takes them |
| `imp_l` | int64, Q16 gram-tile^2 per tick | the angular part of every push |
| `alive` | 0/1 | 1 in both arcs (no prop is destroyed before #229) |

Arc 2 adds per burn cell, named `<col>_fr`, `<col>_fl`, `<col>_br`, `<col>_bl`:
`e_cell` (int64 heat counts), `fuel_q` (int64, Q16 units of N_O2), `burning`
(0/1) and `o2_acc` (int64, the sub-count demand accumulator: synced because it
crosses ticks, section 9.4). They are record-tail columns of the same record,
added in arc 2's first patch.

Host-only (not hashed; the unit precedent, "the momentum books" in
`.claude/rules/device-units.md`, Device conductor): the momentum books per
axis `(in, resisted, wall, clamped, contact_out)` and the angular books
`(in_l, resisted_l, wall_l, clamped_l)` (section 4.7).

Not stored at all: "asleep". It is DERIVED (section 5): all impulse state
zero, the `unit_impulse.at_rest` idea.

Why the centre of mass is the stored position: a rotation then never moves
it, and every lever is measured from it (`unit_impulse.lever_q` takes a
centre). The rectangle's centre `G = C - R(th) (com_u, com_v)` is derived when
needed, from the same `c, s` the stamp uses (section 3.2).

Cost: 11 synced integers per prop in arc 1, 16 more in arc 2; about 20 load
constants. A few hundred bytes per prop.

### 2.2 Where it lives

**DECISION: a level entity class `movable_prop` with a host runtime store and
a device object table, from the first patch that holds state; its state hashed
through the entity section.**

- **Authored** in the level as an entity (`src/simulation/entities/movable_prop.py`,
  schema only, import-light like `entities/heat_source.py`), placed through
  `level_lib` (the Level data layer), its class rows in config
  (`[movable_props.<class>]`, a table, never a hardcoded class). The class
  rows bind at reset: they are NOT hot-reloadable (a Ctrl+R that changes one
  takes effect at the next level load), because they are quantized at the
  class door into synced load constants (the Config reload seam is for live
  tables).
- **A new class, not an extension of `prop`.** The existing `prop` entity is a
  load-time tile stamp that "never ticks" (CLAUDE.md, Prop tile stamps) and
  whose entity tail is taken once at build (`.claude/rules/device-units.md`,
  Entity section export). Extending it would break both. The look fields of a
  movable prop (model, scale, tint) live on the `[movable_props.<class>]` row,
  never synced (section 11, patch A5).
- **Host form:** ONE store object on the sim, `Simulation.props`
  (`src/simulation/movable_props.py`), a structure of arrays in entity ordinal
  order, numpy int64 columns. It is the only Python writer of the columns (the
  Swarm facade discipline). `sim.entities` holds row VIEWS of the store, never
  copies, so the entity serializer reads live columns (a property: moving a
  prop changes its entity bytes, section 12). The reference laws compute in
  Python ints; a vectorized host path is either proved in range at the class
  door or split into exact 16-bit halves (section 4.8), and a property holds
  it equal to the Python-int reference.
- **Device form:** ONE `ObjectTable` schema, `unit_schema.PROP_TABLE`, on the
  Object tables mechanism (`src/simulation/object_tables.py`), rows in ordinal
  order, with its record function `unit_kernels.props_of`, its way back
  `props_to_host` and its entity-row function `prop_entity_rows`, as the door
  and heat-source tables do. It lands in patch A2, the patch where the state
  first exists, so the entity-export rule ("A class whose state starts moving
  gets a table and an entity-row function in the same patch",
  `.claude/rules/device-units.md`, Entity section export) holds from the first
  tick a prop can move (patch A3).
- **Digest:** the entity section. Each prop's record is
  `entities/serialize.record_head` (the load-constant header) plus
  `record_tail` of its synced columns.

**Alternatives it beat:**

1. *Rows of the unit store / a `Unit`.* Rejected. The Unit is "one class;
   subclassing is an anti-goal" and the Unit schema is "THE one table of what
   a unit holds" (`.claude/rules/units.md`): a prop row there would put
   health, statuses, loadouts, turning and the observation's unit block on
   it, and every unit law would need a "not a prop" branch.
2. *A swarm species with a footprint* (Erik's first idea, withdrawn by his
   second note). A swarm unit is a point run by "ONE FP_HD per-slot law" in
   slot 7's tail (`.claude/rules/swarm.md`); THE BOUNDARY says a swarm law
   never reads a unit column, while a prop is shoved by a unit and later
   carried (#228); the swarm store has no pose, heading rate or mass.
3. *Its own SoA store on the map beside the swarm store* (`gmap.prop_*`, its
   own `__props__` carrier). Rejected: the map-resident store is for laws that
   run inside slot 7's device chain (`.claude/rules/device-residency.md`,
   Batched resident engine); a prop runs no law in slot 7, and "a new table is
   a schema there, never a second table mechanism" (Object tables). The host
   keeps the SoA layout anyway.
4. *Re-stamping the prop's material into `material` every tick* (#220's
   "movable solid"). Rejected in section 3.5: every move would be a topology
   change through `GameMap.on_tile_changed`, the tile serial bumped every tick
   (which makes `GameMap.take_stamp_rows` return None).

**Cost of the decision:** a new entity class and its schema; a config table;
the host store and its facade; one `ObjectTable` schema plus the generated
layout (`tools/gen_unit_layout.py --write`); the record, way-back and
entity-row functions; Class-T kernels in a new family TU
`cpp/src/cuda_unit_props.cu` (on CMakeLists' CUDA list and the `/fp:strict`
list); new twin rows (section 11).

### 2.3 Digest and versioning

- The entity section format is unchanged (`ENTITY_SECT_V1`): a new class adds
  records only on a level that holds one. No level in the golden corpus holds
  a movable prop, so **no golden moves** when the class lands; adding a class
  changes only `registry_content_hash`, which is provenance and not hashed
  into goldens (scope critic, checked). A level that gains movable props later
  (levelgen, #171 job 1) moves its own goldens in that patch, with written
  rationale.
- The prop's synced columns are hashed always (not "while non-zero"): they are
  new, so no existing digest depends on their absence.
- The two new per-tile planes (`prop_mob_cap`, `prop_owner`, section 3.3) are
  NOT digested: each is a pure function of the hashed prop rows. Because they
  are read across ticks (built at slot 6, read at the next tick's slots 3 and
  3p), they are rebuilt at reset and on EVERY state landing (a twin's
  `props_to_host`, the gate's download, a chained run's host replay), they
  join `GameMap._RESIDENT_UNIT_PLANES` (`src/simulation/gamemap.py`), and the
  lockstep declares them field outputs compared at tol 0 (section 12).
- Of the existing stamp planes, `tests/field_digest_spec.toml` digests
  `dyn_heat_atten_q` and `obstacles` only (v1 wrongly listed
  `dyn_light_atten_q`). `dyn_heat_atten_q` changes on a prop-bearing level,
  which is correct and moves no golden for the reason above; `obstacles` does
  not change (section 3.3). `dyn_permeability` and `dyn_wave_absorb` are float
  planes excluded from the digest (`excluded_float_fields`).
- The recorder (`src/simulation/recorder.py`) gains the prop columns
  additively in the patch that wants them in recordings (not required here).

---

## 3. The stamp: an integer rule for a turned rectangle

### 3.1 What the stamp must do

Every tick, each prop's rectangle at its pose becomes tile coverage that the
tick's readers see: gas permeability, blast absorption, heat and light
extinction (the four dynamic planes a unit already stamps,
`GameMap._stamp_units_python`), and walking. The rule must be exact integers,
a pure function of the pose (so order-free), the same bits on the CPU and the
device, and cheap.

### 3.2 The rule

Unchanged from v1 (the determinism critic checked it: exact, the same in any
visiting order, CPU and device trig the same code). Notation: `ONE = 2^16`.
The rectangle centre `G = (X, Y)` in Q16 tiles, half sides `A` (along the
heading) and `B` (across) in Q16 tiles, `c = cos_q16(th)`, `s = sin_q16(th)`
from the trig kit (`cpp/src/fixed_point.h`; Python `unit_fixed.cos_q16` /
`sin_q16` call the C++ kit), each a Q16 integer in `[-ONE, ONE]`.

1. **The sample lattice.** Each tile `(i, j)` holds `S x S` sample points,
   `S = 4`, at offsets `o_p = (2p + 1) ONE / (2S)` for `p = 0..3` (8192,
   24576, 40960, 57344). A sample's position is `P = (i ONE + o_p, j ONE +
   o_q)`. The lattice is global: every sample belongs to exactly one tile.
2. **The frame.** `dx = Px - X`, `dy = Py - Y` (Q16), then `u = dx c + dy s`
   and `v = -dx s + dy c` (Q32, int64).
3. **Inside.** A sample is inside iff `|u| <= A ONE` and `|v| <= B ONE`
   (closed on both ends, so the rule is symmetric under `(c, s) -> (-c, -s)`).
4. **Coverage.** `k(i, j)` = the number of inside samples of tile `(i, j)`,
   `0..16`.
5. **The tiles visited.** Only the bounding box:
   `Rx = ((A |c| + B |s|) >> 16) + 2`, `Ry = ((A |s| + B |c|) >> 16) + 2`,
   tiles `(X - Rx) >> 16 .. (X + Rx) >> 16` by `(Y - Ry) >> 16 .. (Y + Ry) >> 16`.
   What keeps the box complete is the half-sample offset (8192 counts: no
   sample sits within 8192 of a tile edge), not the `+2` pad; the kit's
   `c^2 + s^2` deviates from `ONE^2` by up to about 2.1e-5, about 11 counts at
   an 8-tile prop, far inside 8192 (determinism critic, finding 11). The pad
   stays as margin.
6. **The burn cells** (read only by arc 2). The same samples split half-open
   by the signs of `u` and `v`: front `u >= 0`, back `u < 0`, right `v >= 0`,
   left `v < 0`. So `k = k_FR + k_FL + k_BR + k_BL` exactly, per tile.
7. **The blocking set.** Tiles with `2 k >= S^2` (at least half the samples).

No division, no rounding, no float, no trig inside the rule (the kit is called
once per tick per moving prop, for `c` and `s`).

Integer range: `|dx| <= 2^28` on a 4096-tile map, `|c| <= 2^16`, so `|u|,
|v| <= 2^45`, inside int64 (checked by the determinism critic).

### 3.3 What coverage means for each reader

Each class row names ONE `[materials.*]` row, and the prop stamps that row's
values, scaled by coverage where a reader can take a fraction (the "Prop tile
stamps" principle of one material row, extended to the dynamic planes; the
index line's amendment is in section 14.2).

The 17 values per plane per class (`k = 0..16`) are built ONCE per map at the
class door, in integers, and are Device tables rows on the device:

| plane | value at coverage k | combine with the plane | why graded or binary |
|---|---|---|---|
| `dyn_permeability` (float32) | `p_k = (ONE - ((ONE - p_row_q) k) // 16) / ONE`, an exact integer over 2^16, exact in float32 on every machine; the class door refuses a row below `P_PROP_MIN` (section 3.5, R1) | MIN with the value after the units' write | graded: gas sees a smooth porous body as it slides |
| `dyn_wave_absorb` (float32) | `a_k = (a_row_q k // 16) / ONE`, snapped to 0 inside any band the solver forbids (UNVERIFIED whether `config.toml`'s forbidden `(0, 0.02)` band on air binds a dynamic stamp) | MAX | graded |
| `dyn_heat_atten_q` (int32 Q16) | `h_row_q k // 16` | MAX | graded |
| `dyn_light_atten_q` (int32 Q16, x3) | `l_row_q[c] k // 16` | per-channel MAX | graded |
| `prop_mob_cap` (int32, NEW) | the row's `mobility + 1` on the blocking set, else 0 (no cap) | a plain WRITE (blocking sets never overlap, below) | binary: a walker enters a tile or not |
| `prop_owner` (int32, NEW) | the prop's row + 1 on the blocking set, else 0 | a plain WRITE | binary |

**Blocking sets never overlap, at load and after every move.** The load door
refuses a level whose props' blocking sets overlap each other, sit on a tile
that is not prop-enterable (section 4.5 step 3) or leave the grid (determinism
critic, finding 4); the move keeps them disjoint (section 4.5, property 16).
So each blocking tile has exactly one writer and `prop_mob_cap` /
`prop_owner` are plain writes on both backends. As a check, not a rule, the
device counts writers per tile in a scratch plane (`atomicAdd`) and refuses
the run by name (`PropOverlapRefused`) if any tile has two; the host asserts
the same.

Readers of the new planes:
- **Walking.** A tile's effective mobility is the static one, or
  `min(static, cap - 1)` where `prop_mob_cap != 0`. Read by
  `GameMap.is_passable_block` (the step) and `GameMap.footprint_mobility`
  (the speed through `movement.footprint_samples`), and on the device by the
  slot-3 movement kernel (`cpp/src/cuda_unit_movement.cu`) through the same
  resident plane, in the same patch (A2). So a walker climbs a table at its
  material's penalty (furniture: mobility 400, "climbable at a penalty: 40%
  speed", `config.toml` `[materials.furniture]`) or is stopped by a row with
  mobility 0 (R6). With no prop on the map the plane is all zero and both
  readers take an early-out, bit-identical to today.
- **Planning (A*, the horde field)** reads terrain only, as today. A path may
  run through a prop; the step is blocked and the walker slides or waits.
  ACCEPTED GAP: planners do not see props (a plane that changes every tick
  would invalidate the horde field cache every tick). With a wall-like prop
  (mobility 0) in a corridor, bots and zombies can jam there: part of R6.
- **Doors.** A door does not close on a prop: `door_system._occupancy_clear`
  also asks `prop_owner` over the span, and so does its device twin in the
  applier's DOORS stage (`cpp/src/cuda_unit_doors.cuh`), same patch
  (`.claude/rules/sim-core.md`, the door sweep's twin rule). R8.
- **Sight, rounds, grenade flight, the spray cone:** not in either arc
  (section 7), #227.
- **`obstacles` stays static.** A prop is a porous body, like a unit, never a
  hard wall: the solvers' hard-zero boundaries key off `obstacles`
  (`GameMap._stamp_units_python` docstring).

Order on the stamp: the units' rows first, exactly as today (the unit
permeability WRITE whose last row wins, `cpp/src/cuda_stamp.cu`), then the
props' rows, each a MIN or MAX against the plane as it stands. MIN and MAX
commute, so the props' result does not depend on their order, and the units'
arithmetic is untouched, so a prop-free tick is byte-identical.

### 3.4 The device stamp kernel (fixes the determinism critic's finding 3)

`cpp/src/cuda_stamp.cu` today: K1 resets the planes, K2 picks a winner per
tile by `atomicMax(&winner[idx], r)` over every row, K3 lets only the winner
write the permeability, then the MAXes. Appending prop rows after the unit
rows would let a prop row win K2 and erase the units' write. v2:
- **K1 also zeroes `prop_owner` and `prop_mob_cap`** (no stale owner survives
  a prop leaving).
- **K2 picks its winner among UNIT-mode rows only** (a row's MODE column says
  unit write or prop combine). K3 is unchanged.
- **K4 (new, after K3):** each prop row does `atomicMin` on the permeability
  float's bit pattern (exact: the values are non-negative floats, whose bit
  patterns order as the values do; checked by the determinism critic), the
  MAXes as the units' rows do, and the two plain writes.
- **The oracle** includes a unit and a prop on one tile, and two props whose
  partly covered (non-blocking) tiles overlap.

"A new stamp output is a column of the rows, the kernel and the CPU stamp in
ONE change" (`.claude/rules/device-residency.md`, Stamp kernel). The stamp
kernel is Class P, so that patch (A2) merges only with `.cuda_green` at HEAD.

### 3.5 The gas in a tile a prop moves into or out of

**DECISION (R1): nothing moves, and no prop seals.** A prop, like a unit, is a
porous body on the dynamic planes. The gas in a tile stays in it when a prop
arrives or leaves: no N, energy or trace moves, so the mass and energy books
close trivially. This is the units' rule ("units do NOT push atmosphere as
they walk", `GameMap.stamp_units` docstring).

The class door refuses a material row whose permeability is below
`P_PROP_MIN` (proposed 0.05; the exact floor is the patch's measurement).
A zero-permeability dynamic tile over a gas cell would leave a cell the
solvers still count but no face reaches.

This is a physics and gameplay decision, so it is ruling R1, with the real
numbers (physics critic, finding 11): a thin-board furniture row fills 3.24%
of its tile's air column (`config.toml` `[materials.furniture]`, M2 note),
but a solid fridge fills about 72% of the air over its tiles (0.88 m^3 of
1.22 m^3). Under the default a barricade of furniture slows a decompression
but can never stop it, and a steel fridge must name a porous material row.

**Alternative it beat:** the prop displaces its volume of air, or seals.
Doing either exactly means moving N and energy through the gas energy seam
every tick a prop moves (`GameMap.gas_energy_move`, booked), or making a prop
a moving wall to the gas (a topology change every move), each with a device
twin. ACCEPTED GAP: a moving prop displaces no air.

### 3.6 The check (run, pure Python)

Script `C:/tmp/props226/stamp_check.py` (not in the repo); output
`C:/tmp/props226/stamp_check_result.txt`, reproduced byte for byte by the
determinism critic. 400 random poses, sides 1 to 8 tiles, any heading:

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

What it shows: the rule is exact, order-free, translation-covariant,
symmetric under a half turn and under mirroring, and its bounding box is
complete; the burn cells partition the coverage exactly; coverage area is
within 0.06 tiles per tile of perimeter of the true area.

What it cannot show (determinism critic, finding 10): random poses never land
a sample exactly on the rectangle's edge, so a closed inside test and an open
one differed 0 times in 300 random poses, while on one axis-aligned fixture
closed gives `k = 35` and open 15. Property 8 (section 12) therefore adds
axis-aligned fixtures on tile and sample lines and the exact headings 0,
+-102944 and 205887 (where the kit returns exact (+-ONE, 0) and (0, +-ONE)).

**The size floor (R14).** At a smaller side of 1 tile the half rule leaves the
blocking set empty in 2 of 300 poses and not 4-connected in 113 of 300; at
1.5 tiles it never fails. The floor is stated in metres and tiles both:
`max(0.5 m, 1.5 tiles)`. At the shipped 0.333 m tile that is 0.5 m. On a
level with 1.0 m tiles (airlock_demo) it is 1.5 m, and a 0.8 m table is
refused there (physics critic, finding 9). ACCEPTED GAP: coarse-tile levels
hold only large movable props.

### 3.7 Worked example

A 1.6 x 0.8 m table (4.8 x 2.4 tiles) at heading 30 degrees, centre
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
`u = -2257653176`, `v = -1548191452`, inside `A ONE = 10307895296` and
`B ONE = 5153947648`. The blocking set is the 11 tiles with `k >= 8`. On
those a furniture row stamps `prop_mob_cap = 401` and permeability `0.5`; on
tile (10, 5) with `k = 2` it stamps permeability `1 - 0.5 x 2/16 = 0.9375` and
no cap.

### 3.8 How it meets the existing stamp and the topology seam

- **`stamp_prop_tiles` (static props) is untouched.** It writes `material` once
  at load and never ticks. A movable prop never writes `material`, so it never
  calls `GameMap.on_tile_changed`, never bumps the tile serial, and never
  touches the topology seam or its device twin.
- **The stamp rows.** Today's row is `[env, y, x, perm bits, wabsorb bits,
  heat_q, l0, l1, l2]` (`GameMap.take_stamp_rows`); a prop row adds the MODE
  column, the cap and the owner. On the device the prop rows come from
  `PROP_TABLE` (`unit_kernels.stamp`, the Device sorts and 9b system), never a
  second stamp kernel.
- **Where a tile changes under a prop** (a wall burned through next to it): a
  wall tile is never under a blocking set (load door, move test), so a
  destroyed wall does not need to know about props; the static baselines move
  through the seam as today and the prop's next stamp combines with them.

---

## 4. The push law

### 4.1 What is shared with units, and what a prop adds

**DECISION: one impulse door for every body.** Everything that pushes a prop
hands it an impulse through the function units use,
`unit_impulse.add(table, body, jx_q, jy_q, lever, twist_q16)` (Unit impulse:
"THE way anything pushes a unit: an impulse through `add`, never a position
or velocity write").

How `add` serves a prop, exactly (scope critic, finding 10a):
- `add` writes `body.imp_x`, `imp_y`, `imp_l` and `body.impulse_books[IN_X]`
  / `[IN_Y]` (`unit_impulse.add`). The prop store hands out a ROW VIEW
  (`movable_props.PropRow`) whose `imp_*` are the store's synced columns and
  whose `impulse_books` is the prop's host-only books row, indexed by the
  same `IN_X` / `IN_Y` constants.
- `add` and `angular_q` clamp the lever to `table.r_body_q`
  (`unit_impulse.angular_q`). They gain ONE optional argument,
  `r_clamp_q=None`: `None` means `table.r_body_q`, so every unit caller is
  bit-identical; a prop passes its `r_lever_q` (section 2.1). The device
  `impulse_add` / `angular_q` (`cpp/src/cuda_unit_lib.cuh`) gain the same
  argument in the same patch (A0).

Shared, called as they are: `unit_impulse.add`, `angular_q`, `lever_q`,
`along_q`; `div_trunc` and the remainder discipline (`dv = J / m` truncated,
the remainder stays in the accumulator, `unit_impulse.integrate` step 1);
`_shrink` and `_clamp_vmax` (made public in A0, unchanged); the prone
branch's arithmetic, "every input meets the prone friction at once and what
is left moves the body" (`unit_impulse.integrate`, the `is_prone` branch),
which IS Coulomb friction: a prop is a body that is always prone, with its
own friction cap; the per-tile wind term of `GameMap.footprint_wind`
(section 4.3).

What a prop adds (`src/simulation/prop_motion.py`): the integration without
sway, effort, reaction delay or knockdown; rotation about the centre of mass
with an angular friction cap; its own move (a turned rectangle against walls,
units and props, section 4.5); the wind read for a many-tile body (4.3); the
contact rule (4.6).

**Alternative it beat:** a separate rigid-body solver (velocity iterations,
contact manifolds, restitution). It would be a second push law and a second
set of collision rules, iterative and order-dependent unless run to a fixed
count in a fixed order; nothing in the milestone needs stacking or resting
contact between props.

### 4.2 The integration (slot 1a, after the units)

For each prop that is not asleep, in row order (each row reads only itself):

1. `dvx = div_trunc(imp_x, m)`, `imp_x -= dvx m` (and y); `dw =
   div_trunc(imp_l, I)`, `imp_l -= dw I`. The remainders stay.
2. `v += dv`, clamped to `V_MAX_PROP = unit_impulse.V_MAX_Q` (4 tiles a tick:
   32 m/s at 0.333 m; booked `clamped`); `w += dw`, clamped below half a turn
   per tick (`turning.OMEGA_MAX_Q16`; booked `clamped_l`, physics critic 14).
3. Friction, linear: `(vx, vy) = shrink((vx, vy), cap_lin_q)` (booked
   `resisted`). Angular: `|w|` shortened toward zero by `cap_ang_q` (booked
   `resisted_l`).
4. **Held this tick?** If after step 3 `v = 0` and `w = 0`, the remainders in
   the accumulators are below one tick of friction by construction (each is
   below one count of velocity times the mass). They are booked `resisted`
   and zeroed, so a prop friction holds ends the tick with ALL impulse state
   zero, exactly, and is asleep (section 5). The physics critic checked this
   step: exact.

Coulomb friction falls out: a steady force below `mu m g` never moves the
prop; above it the prop gains `(F/m - mu g) dt` a tick; when the force stops
it stops exactly at zero, the same in m/s at both tile sizes (physics critic,
checked). The hold scales with mass (Erik: friction "could depend on the
weight": it does, as `mu m g`).

ACCEPTED GAP (R13): one coefficient for starting and for sliding; linear and
angular friction applied separately, which overestimates the dissipation of a
prop that slides and turns at once by up to about 2x (physics critic 14).

**The inertia** is about the centre of mass: `I_cm = m (L^2 + D^2)/12 - m
|o|^2` with `o` the mass offset from the rectangle's centre. Worked (scratch
`C:/tmp/props226/v2/push.py`), a 20 kg table 1.6 x 0.8 m:

```
offset 0.0 m: I_G 5.33, v1 (plus) 5.33, v2 I_cm 5.33 kg m2
offset 0.2 m: I_G 5.33, v1 (plus) 6.13, v2 I_cm 4.53 kg m2
offset 0.4 m: I_G 5.33, v1 (plus) 8.53, v2 I_cm 2.13 kg m2
offset 0.6 m: I_G 5.33, v1 (plus) 12.53, v2 I_cm -1.87 kg m2 REFUSED
the class door refuses an offset with I_cm below I_G / 10: |o| > 0.490 m for this table
```

The uniform-rectangle-plus-point-offset model is the design's; a class that
needs another mass distribution sets `inertia_g` directly (a class column,
checked against the same floor).

### 4.3 Wind and blasts (slot 9c2, with the units)

**The read (DECISION, R9).** The read of what the air does to a unit takes the
MAX of the per-tile wind term over its 3 x 3 footprint and applies that one
tile's pressure to the whole body (`GameMap.footprint_wind`). Over a prop's
12 to 30 tiles that max picks noise peaks (the #217 checkerboard), reads
shielded air on tiles the prop's own stamp slows, and puts the centre of
pressure at one fixed point, so a blast on one end never spins a balanced
table (physics critic, finding 5). v2's model, for a many-tile body:

1. **The windward ring.** The ring is every gas tile inside the grid with
   coverage `k = 0` that is 4-adjacent to at least one tile with `k > 0`.
   Its air is just outside the prop and unslowed by the prop's own stamp.
   Each ring tile `i` carries a weight `omega_i` = the number of its four
   neighbours with `k > 0` (1 to 4): its contact with the prop, the
   coverage weight. It is windward when its wind points toward the
   rectangle's centre: `a_i = vx_i (Gx - Px_i) + vy_i (Gy - Py_i) > 0`.
2. **The per-tile term** is THE term the unit read uses, `W_i = N_raw_i *
   ((vx_i^2 + vy_i^2) >> 16)` (`GameMap.footprint_wind`'s loop body), lifted
   in patch A0 into one function both reads call.
3. **The pressure:** a coverage-weighted mean over the windward ring tiles,
   `qbar = sum(omega_i W_i) / sum(omega_i)`, truncated. The direction:
   `d = sum(omega_i (W_i >> 32) v_i)` (the load-weighted wind; the shift
   keeps the sum in 128 bits), `|d|` by the kit's `isqrt`; no windward tile or
   `d = 0` returns at once (no division by zero, determinism critic 9).
4. **The force:** the area the prop shows `d`, its projected width times its
   height. `w_proj = (2A |dx s - dy c| + 2B |dx c + dy s|) / |d|` (Q16 tiles;
   the formula the physics critic derived independently). The magnitude is
   `J = ((((qbar >> 16) * k_wind_q) >> 16) * w_proj) >> 16`, taken in two
   128-bit steps on the device as `add_wind` takes its product
   (`cpp/src/cuda_unit_impulse.cu`), Python ints on the host. `k_wind_q` is
   built from an area in m^2 per tile of width (section 2.1), so the force in
   newtons is independent of the tile size (worked below).
5. **The torque from the per-tile forces.** `J` is shared among the windward
   ring tiles in proportion to `omega_i W_i`: `F_i = div_trunc(J omega_i W_i,
   sum(omega_j W_j))`, each along `d`, applied at the tile's contact point
   (the mean centre of the faces it shares with the covered set). Each share
   goes through `add(table, prop, F_ix, F_iy, lever_i, r_clamp_q=r_lever_q)`
   with `lever_i = lever_q(COM, contact_i, d_hat)`. The linear impulse the
   prop receives is `sum F_i` (the truncation residue of the shares never
   enters), and the torque is the sum of `r_i x F_i` through the lever
   arithmetic `add` already has. So a gust on one end turns even a balanced
   table, and an even wind turns only an off-centre one.

Worked (`C:/tmp/props226/v2/push.py`; `Cd` 1.1, a 0.75 m tall table 1.6 m
wide, head on at 100 Pa; `k_face` is `k_wind_q`, the constant per tile of
width):

```
tile 0.333: k_face 56541 (0.8627); faces across 1.6 m: 4.80; at q 100 Pa the table takes 132.0 N (q Cd H w = 132.0 N)
tile 1.0: k_face 56541 (0.8627); faces across 1.6 m: 1.60; at q 100 Pa the table takes 132.0 N (q Cd H w = 132.0 N)
```

And the wind at which a prop starts to slide head on (`mu` 0.4):

```
chair      10 kg  face 0.5x0.9 m: slides above    79 Pa (wind 11.5 m/s); marine floored at 245 Pa
table      20 kg  face 1.6x0.75 m: slides above    59 Pa (wind  9.9 m/s); marine floored at 245 Pa
fridge    100 kg  face 0.7x1.8 m: slides above   283 Pa (wind 21.7 m/s); marine floored at 245 Pa
bookcase   40 kg  face 0.9x1.8 m: slides above    88 Pa (wind 12.1 m/s); marine floored at 245 Pa
```

So a gust that floors a marine slides every piece of furniture lighter than a
fridge. `Cd` is a new class column with no source beyond the flat-plate value
(section 15). UNVERIFIED: the magnitude of a real blast's push through this
read (the wind is sampled once per tick, not integrated over a shock that
passes in milliseconds), measured in patch A3's scenes.

**The read is shared, not second.** "THE read of what the air does to a unit
... never a second reader" (CLAUDE.md, Wind load on a body). The per-tile
term is one function; the unit read is its footprint max (unchanged, so
units stay bit-identical); the prop read is its windward-ring mean. The index
line and `.claude/rules/units.md` (Wind load on a body) are amended to say
so (section 14.2). This ring read is the read #236's "a fluid pushes a body"
extends: water is a second row of the same law (the water's depth and flow
planes in place of `N` and the wind, its density in place of air's), never a
second reader.

**The blast.** "A blast pushes only through its own wind" (Generic explosion,
`.claude/rules/explosions-fire.md`). A detonation moves props through this
row and nothing else.

**What #217 changes here.** #217 (the EOS checkerboard) is changing the wind
the read sees. Its measured idle wind on generated ships (20 to 30 in the
planes' units, scope critic 4) would push props around on idle ships if it is
above the slide thresholds above. The read is built on today's wind; patch A6
(levels) waits for #217 to land, and section 15 lists the dials tuned under
the bug.

### 4.4 A unit shoves a prop (slot 1b)

The shove verb (`unit_impulse.shove`, `shove_target`) chooses among units
today. It also considers props whose rectangle is adjacent within the cone, by
the same bearing rule and a deterministic tie (units before props at an equal
bearing, then the lower id). Its impulse is limited by the hand's speed
(physics critic, finding 8; R7):

```
J = min(J_shove, m_prop x v_hand)        v_hand = 2.0 m/s (proposed)
```

`J_shove` is the unit's shove (`[impulse] impulse_ns = 100`, through
`unit_traits.push_impulse_q`); `v_hand` is a new `[movable_props]` dial and a
Device tables row. It goes through `add` with the lever from the prop's
centre of mass to the line of the shove, so shoving one end turns it. Worked
(`push.py`, friction applied at once in the same tick):

```
v_hand 2.0: chair   J  20.0 N s -> 2.00 m/s, after this tick's friction 1.84, slides 0.43 m
v_hand 2.0: table   J  40.0 N s -> 2.00 m/s, after this tick's friction 1.84, slides 0.43 m
v_hand 2.0: fridge  J 100.0 N s -> 1.00 m/s, after this tick's friction 0.84, slides 0.09 m
v1 (no limit): chair 10 m/s slides 12.3 m
```

On the device the shove of a prop is a PUSH request (section 6), the target
chosen in `cpp/src/cuda_unit_verbs.cu` by the same rule. The bots
(`src/control_bots.py` `try_shove`, which reads `t.team`) keep choosing units
only: one filter in `try_shove`, ACCEPTED GAP until a bot has a reason to
shove furniture. ACCEPTED GAP (inherited, Unit impulse): the shover feels no
reaction. Walking into a prop does not push it (R7; carrying is #228).

### 4.5 The move (slot 3p, after the units' slot 3)

The prop's displacement is one call line after the units' displacement loop
(the Tick conductor: "a new system = one call line"). sim-core.md's "Slot 3 is
the ONE displacement loop" is amended: slot 3 is the one loop of units, slot
3p the one of props (section 14.2).

**Everything is double-buffered** (determinism critic, finding 5). The move
reads every prop's pose and velocity from the START-of-slot buffer, writes
candidate poses and velocities to a scratch buffer, and commits after the
refusal pass. No prop reads another prop's state as it changes within the
pass, so the result is the same in any row order on the CPU and race-free
with one thread per prop on the device.

1. **Substeps.** `k = ceil(max(|vx|, |vy|, (R_max |w|) >> 16) / (ONE - 1))`,
   `R_max` the half diagonal (Q16 tiles), `w` Q16 radians per tick, the
   product Q32 shifted back to Q16 (determinism critic 13). No corner moves a
   whole tile in a substep (the `unit_impulse.move` rule, `SUBSTEP_Q`,
   extended to rotation).
2. **How v and w split.** Substep `i` (0 to k - 1) moves
   `div_trunc(vx (i + 1), k) - div_trunc(vx i, k)` (and y, and w), so the k
   parts sum to `v` and `w` exactly, with no remainder.
3. **Each substep** tries the full step (translate and turn), else the
   translation alone (the turn is dropped: `w = 0`, booked `wall_l`), else x
   alone, else y alone, else stop; the axis not taken is INTO THE OBSTACLE and
   meets the contact rule (4.6). The unit move's slide rule, in shape.
4. **The test** of a candidate pose: its blocking set must hold only tiles
   that are
   - prop-enterable in the static map: open floor (the static prop stamp's own
     base rule, `prop_system._STAMP_OK_BASE_MATERIALS`, today `MAT_AIR`), not
     vacuum, not the ambient ring, inside the grid (so a closed door, a wall, a
     grille, a static crate tile all block; an open doorway is air);
   - `prop_owner` 0 or this prop (the start-of-slot plane: no other prop's
     blocking set);
   - not newly inside a living unit's footprint: a tile in the candidate set,
     not in the prop's own start set, inside a living unit's `occupied_tiles()`
     after slot 3 blocks (so a unit standing on a climbable table does not pin
     it, and a unit a prop slides into stops it). On the device this reads the
     env's unit rows in row order (a pure read).
5. **The refusal pass.** Two props may each move into the same free tiles in
   one tick. After every prop has moved into scratch, each moved prop adds 1
   to a claim plane over its NEW blocking tiles (an integer sum, order-free;
   `atomicAdd` on the device) and records `atomicMin` / `atomicMax` of its row
   on the same tiles. A moved prop with any new tile claimed twice or more is
   REFUSED: it returns to its start pose. Its momentum is not deleted (physics
   critic 14):
   - a tile claimed by exactly TWO props names the pair (the min and max row).
     A refused pair takes its common velocity: both get `v_c = (m_1 v_1 + m_2
     v_2) / (m_1 + m_2)` (per axis, `div_trunc`, the remainders into each
     prop's accumulator), so the pair's momentum is kept exactly; next tick
     they move together and the front one leads;
   - a tile claimed by THREE or more props stops every claimer as on a wall
     (velocity zeroed, booked `wall`). ACCEPTED GAP: a three-way pile-up in one
     tick (rare) loses momentum to the wall book.
   A refused prop goes back to tiles it held at the start, which no other prop
   could have entered (they tested the start plane), so one pass suffices
   (determinism critic, checked: chains, cycles and ties resolve
   symmetrically). A unit and a prop wanting one tile: the unit wins, because
   slot 3 runs before 3p.

Cost: the move recomputes coverage per substep for the moving props only.
ACCEPTED GAPS: no bounce (a blocked axis stops; the grenade bounce is not
borrowed); two fast props whose paths cross inside one tick but whose end
poses do not overlap pass through each other.

**The hull (R2).** A prop cannot enter vacuum or the ambient ring, so a hull
breach never blows furniture into space; the wind of a decompression slides
props to the breach and they stop at its edge. In a game called Breach this
is a headline feel call, so it is a ruling, not only a gap: under "yes, blow
them out", a prop that crosses the hull is removed from the state, which
needs the stable-slot removal #229 brings (section 13).

**Alternative it beat:** moving props one after another, each seeing the ones
before it. Deterministic but order-DEPENDENT: a prop's fate would depend on
its ordinal, and the device would need it serial per env.

### 4.6 What a prop hits: walls, units, props (the contact rule)

**DECISION (R3): momentum is kept in a contact.** v1 zeroed the prop's
blocked component and booked what it did not hand on as lost, which destroys
momentum (a 100 kg fridge at 3 m/s into a 100 kg marine handed over 150 N s
and deleted the other 150; physics critic, finding 3). v2 treats a contact as
a perfectly inelastic collision along the blocked axis, which loses kinetic
energy (body kinetic energy is in no group of the closure identity,
energy-books.md) and keeps momentum.

**The hit set and the partner** (determinism critic, finding 5). When a
substep's axis is blocked, the HIT SET is a pure function of the candidate
pose and the start-of-slot state: every static non-enterable tile, every
prop whose start blocking set the candidate enters, and every living unit
whose footprint it newly enters.
- **A wall in the hit set wins:** the blocked component is zeroed and booked
  `wall` (momentum leaves the bodies there, as for a unit, `unit_impulse.move`).
- **Otherwise the partners are the hit bodies**, sorted (units before props,
  then the lower id). With `M = sum m_b` and `vbar` their mass-weighted mean
  velocity along the axis (start of slot), the prop closes on them if
  `v_p > vbar` toward them. Then the common velocity is
  `v_c = (m_p v_p + M vbar) / (m_p + M)`, the impulse handed on is
  `J = m_p (v_p - v_c) = m_p M (v_p - vbar) / (m_p + M)`, computed with the
  kit's 128-bit multiply-divide (section 4.8). Each partner gets
  `J_b = floor(J m_b / M)`, the remainder to the first partner, through
  `add` with the lever from the partner's centre to the contact line.
- **The prop keeps the rest:** its blocked velocity component is zeroed for
  the move and `m_p v_c` goes back into its own accumulator (booked
  `contact_kept`, an internal move, not a book term), so at the next
  integration it carries on at `v_c` less friction. `J + m_p v_c = m_p v_p`
  exactly. A sliding fridge thus carries a marine along with it until the
  marine's feet (the unit law's friction) drain the push.
- **A partner that is not slower** (`v_p <= vbar`): nothing is handed on and
  the prop keeps all of `m_p v_p` (its partner is moving away as fast; it
  tries again next tick).
- **What the unit then does** is the unit law's: sway, stagger, knockdown. No
  damage until #229 (R17).
- **On the device** every `J_b` is a PUSH request (section 6), applied after
  the refusal pass in prop row order. If the prop is refused in the refusal
  pass, its requests of this tick are dropped and the refusal rule (4.5 step
  5) decides its momentum instead.
- **A unit flung into a prop** (by a blast or a shove) stops as on a wall in
  the unit move and hands the prop nothing (scope critic 10d). ACCEPTED GAP
  until #228 makes units throw props, which is where the reverse contact
  matters.
- **A contact event.** Every hard contact (wall, unit, prop) with the impact
  speed above a floor is recorded as a render-only `PropContactEvent` (prop
  id, the other's kind and id, the axis, the speed), the hook #229 asked for.
  Not in the synced event digest; if a synced law ever reads it, it joins the
  digest in that patch (the swarm death event's rule).

### 4.7 The momentum books

Per prop per axis, exactly, after every integration and move:

```
sum of J added (all callers of add, contacts in included)
    == m v + resisted + wall + clamped + contact_out + imp_accumulator
```

`contact_kept` moves momentum from the velocity term to the accumulator term
inside one body, so it is not a term. `contact_out` is what this prop handed
another body; the receiving body books the same integers as `J added`. Over
every body of a tick, `sum contact_out == sum of contact J added` exactly, so
the identity summed over all props and units closes with only the wall,
resisted and clamped sinks. The angular books are the same identity on
`imp_l`, `w`, `I` with `resisted_l`, `wall_l` and `clamped_l`. The books are
host-only (not hashed), like the units'.

### 4.8 Integer ranges (determinism critic, finding 9)

Proved at the class door, or refused by name there; scratch `push.py`:

```
contact numerator at 2 t x 2 t x 8 tiles/tick: 2.097e+18 = 2^60.9 (int64 max 2^63)
momentum m v at 2 t, 4 tiles/tick Q16: 5.243e+11 = 2^38.9
torque term r x J at r 6 tiles, J = 2 t x 4 tiles/tick: 2^57.5 (Q16 x Q16 -> 128-bit, >> 16 after)
face wind W < 2^80 (device bound, cuda_unit_impulse.cu): (W >> 16) * k < 2^81.1: 128-bit, as add_wind
```

- **Mass** 1 kg to 2000 kg, sides at most 8 tiles, `inertia_g` at least a
  tenth of the centred value, `k_wind_q` below 2^24: refused by name outside.
- **The contact impulse** `m_p M (v_p - vbar) / (m_p + M)` fits int64 at the
  mass ceiling, and is taken in the kit's 128-bit multiply-divide anyway
  (the trunc form beside `muldiv_floor_q`), so a raised ceiling cannot wrap
  silently.
- **The wind** is taken in two 128-bit steps on the device, mirroring
  `add_wind` (`cpp/src/cuda_unit_impulse.cu`), Python ints in the reference.
  The device faults by name (the slot's fault count, refused at the books
  read) where a sum of shares or a torque passes int64.
- **The host vectorized paths** never wrap where the reference does not. The
  per-tile term `W >> 16` is computed in int64 numpy by the exact split
  `W >> 16 = N (v2 >> 16) + ((N (v2 & 0xFFFF)) >> 16)` (both products below
  2^62), and `((W >> 16) k) >> 16 = (a >> 16) k + (((a & 0xFFFF) k) >> 16)`
  the same way; a property holds the numpy form equal to the Python-int
  reference over the full range (section 12). Sums that can pass int64 (the
  ring's `sum(omega_i W_i)`) are taken per prop in Python ints, which only
  props that pass the sleep gate of section 5 reach.

---

## 5. Sleep

**DECISION: asleep is derived, not stored.** A prop is asleep when `imp_x =
imp_y = imp_l = 0` and `vx = vy = w = 0` (the `unit_impulse.at_rest`
predicate on the prop's columns). Section 4.2 step 4 makes a prop that
friction holds land there exactly.

**The stamp cache is keyed on the POSE, not on waking** (determinism critic,
finding 6). A pose can change with all velocity and impulse zero on every
path that LANDS state: a twin's `props_to_host`, the gate's download, a
chained run's host replay, #228's carrying. So:
- on the host, each prop's cached rows (tiles, `k`, quadrant split, ring list)
  carry the pose `(x_q, y_q, th_q)` they were computed at; every tick the
  store compares the poses (one vectorized compare) and recomputes the rows of
  any prop whose pose differs;
- the sleeping props' rows are combined once into a SLEEPER LAYER per dynamic
  plane (their MIN / MAX, which commute), rebuilt only when the set of
  sleepers or any sleeper's pose changes; each tick the stamp combines the
  layer onto the planes in one vectorized pass and adds the awake props' rows;
- on the device, the same layer lives per env, rebuilt by the same pose-keyed
  test (a per-prop compare reduced to one flag per env, order-free), and the
  awake props' coverage is recomputed every tick.

A property writes a new pose with zero velocity (as a landing does) and holds
the rows and the layer equal to a fresh computation (section 12).

What an asleep prop costs per tick:
- nothing in slot 1a (the integration's fast path) and nothing in slot 3p;
- its share of the sleeper layer (nothing unless the layer changes);
- the wind read over its ring at 9c2. Always, because the read is what wakes
  it, and two sub-threshold pushes from different sources must still add up.
  But the host has an exact fast path: at 9c2 a sleeper's accumulator holds
  everything slots 1b and 3p added this tick, and nothing else is added before
  the next integration. If that accumulator is zero and an integer UPPER bound
  of the wind impulse (from the ring's largest `W_i` and the largest projected
  width `2(A + B)`, linear and angular) is at most one tick of the prop's
  friction (`m cap_lin` and `I cap_ang`), the next integration would shrink
  it to zero and leave the prop asleep with every synced column unchanged. The
  host then skips the exact read for that prop and counts it in a host-only
  book. Only the host-only momentum books differ (the skipped impulse would
  have been booked in and resisted); a property runs a breeze scene with and
  without the fast path and holds every synced column equal (section 12). The
  device always runs the exact read (it is cheap there).

What wakes it: any `add` that leaves a non-zero accumulator. Below friction,
the next integration shrinks it to zero, books it, and the prop is asleep
again that tick.

Why it is deterministic: the asleep predicate is a pure function of synced
columns; the caches are pure functions of the pose, checked against the pose
every tick.

**Alternative it beat:** baking sleeping props into the static baseline
planes so a sleeper costs exactly nothing in the stamp. Those baselines change
only through the topology seam and the tile serial, so every wake and every
fall asleep would be a topology event with its device twin.

**Dormancy of the whole system:** a level with no movable prop builds no
store, launches nothing, stamps no row and reads no plane: every path is
byte-identical to today (the presence gating of the swarm and the entity
sections).

---

## 6. The device path: who writes what, in what order

The determinism critic's findings 2, 5, 6 and 7 are about writers sharing a
row or a tile and about state landed by paths other than stepping. In one
place:

| law | host | device | why it is safe |
|---|---|---|---|
| stamp (slot 6) | `GameMap.stamp_units` + prop rows | `cuda_stamp.cu` K1 to K4 (3.4) | MIN / MAX commute; owner and cap have one writer (3.3) |
| integration (1a) | per prop, row order | one thread per prop | each row reads and writes only itself |
| shove (1b) | `unit_impulse.shove` on a prop | `cuda_unit_verbs.cu` emits a PUSH request | the device `impulse_add` (`cuda_unit_lib.cuh`) is a plain read-modify-write built for one writer; device-units.md: "a slot that does what is not a row write [goes] through the Request list" |
| move (3p) | double-buffered, then the refusal pass | one thread per prop into scratch; the claim plane by `atomicAdd` / `atomicMin` / `atomicMax`; commit | every decision reads the start buffer |
| contacts (3p) | `add` on partners after the refusal pass, prop row order | PUSH requests, applied by a new PROP_PUSH stage of the per-env serial applier in prop row order | one writer per row at a time; integer sums, the same totals as the host |
| wind (9c2) | per prop | one thread per prop | each prop writes only its own row |
| sleeper layer, planes | rebuilt on pose change and at every landing | the same, per env | pure functions of hashed rows; declared field outputs at tol 0 |

The PUSH request carries `(target kind, target row, jx, jy, lever, r_clamp)`;
it is one kind of the Request list (`src/simulation/unit_requests.py`), and
the PROP_PUSH stage is a stage of the one applier (`applier.STAGES`,
renumbered by name), never a second walker. The device conductor refuses a
level with movable props by name (`DeviceConductor._refuse`, reason "movable
props") from patch A2 until patch A4 lands the last arc-1 twin and its
whole-tick gate (section 11).

---

## 7. Who it blocks

| what | arc 1 | through | later |
|---|---|---|---|
| wind and gas flow | yes, graded by coverage | `dyn_permeability` (the solvers' face weighting) | #227 judges whether it is enough of a shield |
| the blast wave | yes, graded | `dyn_wave_absorb` (MAX); the blast also pushes the prop through its wind | #227 |
| radiant heat | yes, graded | `dyn_heat_atten_q` (MAX), seen by the radiation sweep (Extinction planes) | |
| light | yes, graded | `dyn_light_atten_q` (MAX): props cast shadows | |
| walking | yes, binary | `prop_mob_cap`: the material row's mobility (furniture climbable at 40%, a mobility-0 row a wall) | R6 |
| doors closing | yes | `prop_owner` over the span | R8 |
| other props | yes | the move's start plane and the refusal pass (4.5) | |
| the flamethrower | NO (passes through) | the spray cone stops on solid only (Spray slot) | #227 |
| rounds | NO | the march stops on `solid` (Sight rules) | #227, the row's `cover_exposure` |
| sight | NO | `has_los` reads `solid` and `sight_screen` | #227, the row's `blocks_sight` and the class height |
| thrown grenades | NO | the flight plane (`combat.flight_plane`) | #227, the row's `blocks_thrown` |
| carried as a shield | NO | | #228 |

The four NO rows are deliberate: each is a canonical system with its own
device twin and gate, and #227 exists to decide them with Erik. Until then a
table is cover against wind, heat and light but not against bullets.
ACCEPTED GAP.

---

## 8. Cost

All figures are ESTIMATES argued from measured numbers in the docs; none was
measured for this design.

**The scale.** The medium generated ship is 100 x 70 tiles with 86 fuel tiles
(`docs/tick_budget_2026-10-29.md`). Assume 20 to 60 movable props on such a
ship (UNVERIFIED: levelgen places none yet), and price #226's own aim,
"thousands of props that never move should cost nothing", at 1000 sleepers.
A typical prop's bounding box is 48 tiles (768 point tests at S = 4) and it
covers 12 to 30 tiles; its ring is about 20 to 30 tiles.

**CPU, played tick.** The medium ship's tick is 19 to 21 ms, of which the
physics kernels are about 14 ms and the unit stamp 1.1 ms
(`docs/tick_budget_2026-10-29.md`, profile rows 3 and 10).
- **50 sleepers:** the sleeper layer combined onto six planes (7,000 cells
  each, vectorized), the pose compare, and the wind fast path over about
  1,500 ring tiles: estimate 0.1 to 0.3 ms a tick.
- **1000 sleepers:** the same layer combine (it does not grow with the prop
  count), a pose compare over 1000 rows, and the fast path over about 30,000
  ring tiles in int64 numpy: estimate 0.3 to 0.6 ms in still air. In a breeze
  above the fast path's bound every sleeper takes the exact read in Python
  ints: several ms, which is why the bound exists; #217's idle wind decides
  how often that happens (UNVERIFIED). A layer rebuild (after a blast wakes and
  moves some) is about 20,000 cached rows in numpy, about 1 ms, on those ticks
  only.
- **Awake (a blast just passed):** coverage per awake prop per substep, about
  768 integer point tests, vectorized per prop (about 30 to 60 us,
  UNVERIFIED); 50 awake props add 2 to 3 ms on the few ticks after a blast.

**Device, batched training.** The resident EOS + trace tail costs about 0.169
ms per env-tick at N = 128 and the batched sweep 0.122 ms
(`docs/residency_benchmark_2026-11-07.md`). At N = 128 x 50 props all awake,
4.9 M integer point tests a tick: well under 0.1 ms (estimate). At 1000
sleepers per env the ring reads are 128 x 1000 x 30 = 3.8 M reads and the
layer combine one pass per plane per env: about 0.1 ms (estimate). The
kernels are one thread per prop (integration, move, wind) and one per (prop,
tile) (coverage), all order-free; the contacts and the shove go through the
serial applier's PROP_PUSH stage, a handful of records a tick. No new host
sync: the counts ride the one books read (Device conductor).

**Erik's estimate holds**: the arithmetic is small beside the solvers. The
weight of the arc is the new synced state, its CPU and device twins, and the
laws that must see it (stamp, books, movement, doors).

**The observation.** Observation spec v1 does not change ("a new idea is a
new spec, never an edit", `.claude/rules/rl-agent.md`). Its `WINDOW_PLANES`
(`device_obs.py`) hold no stamp plane (scope critic, checked), so a prop
reaches an agent only through the wind, heat and movement it changes. A
trained soldier cannot see a table until a new spec adds a prop block. Ruling
R15: no training level holds movable props until that spec exists.

