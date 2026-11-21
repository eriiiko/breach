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
that is not prop-enterable (section 4.5 step 4) or leave the grid (determinism
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
   `d = sum(omega_i (W_i >> sh) v_i)`, the load-weighted wind, with
   `sh = max(0, bitlen(max_i W_i) - 31)` taken from the ring's largest term
   (a pure function of the read, the same on both backends), so the weights
   keep 31 bits whatever the wind's strength and the sum stays in 128 bits;
   `|d|` by the kit's `isqrt`; no windward tile or `d = 0` returns at once (no
   division by zero, determinism critic 9).
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

---

# Part B. Arc 2: fire on props

**Read this part as a design, not a build order.** v1's fire law was UNSOUND
(physics critic, finding 1: it neither held its flame nor closed the books).
v2 redesigns it the way that critic asked, puts a worked plateau calculation
in, and lists what remains UNVERIFIED (9.8). Arc 2 gets its own critique round
(physics and books, determinism) before any of its patches is built.

## 9. Fire

### 9.1 The ruling, and the decision

Erik ruled (1.1): a burning prop heats what is beside and underneath it, and
the heat source moves with the stamp.

**DECISION: a prop owns its fuel, its burn state and its heat; tiles never
hold a prop's fuel or a prop's fire. Each burn cell is a BOOKED energy
reservoir.** The tile law (`cpp/src/combustion.cpp`) lands most of a fire's
heat, `H_bed`, in the burning SOLID's own tile: the fuel bed that holds the
flame plateau (combustion.cpp, the bed deposit; `config.toml`
`[physics.combustion]`, the P-R4 block). For a prop the fuel bed is the prop's
own cell. So:
- the bed heat lands in the cell and is booked as the tile law's solid heat is
  (`CombustionSolver.e_comb_solid_heat_sum`);
- the cell exchanges heat with its tiles through a booked seam term, both
  ways (so a prop warms from a hot room and takes that heat out of the room:
  v1's unbooked warming, which created about 9.1 MJ for a 20 kg table warmed
  280 K, is gone; physics critic, finding 7);
- the gas-side parcel `H_fuel` goes to the gas as in the tile law;
- the O2 demand takes the tile law's form (9.4);
- the products follow the tile law's split (9.5, R5).

When the prop moves, its reservoir moves with it (the heat is the prop's);
what it already gave the room stays in the room, so a sliding burning table
leaves a hot trail, not a fire.

**Alternative it beat (v1):** the cell's temperature as an unbooked node
warmed by the body-temperature step, the burn's heat paid at `H_fuel` alone
into the tiles. It gave a burning prop about 9% of a burning tile's heat per
unit of O2, so a prop fire went out unless a tile fire held it up, and its
warming minted energy (physics critic, findings 1 and 7).

**Alternative it beat (the lend):** the prop lends its fuel to the tiles under
it each tick and takes back what is left. Rejected: the tile `fire` and
`temperature` stay with the tile when the prop leaves (a burning table leaves
its fire on the floor), and the lend and take-back are topology-like writes
every tick.

### 9.2 One side burning: four cells

**DECISION (R10): every prop has exactly four burn cells**, the quadrants of
its own frame (front-right, front-left, back-right, back-left; 3.2 step 6),
each with its own reservoir, fuel and burning flag. That is the smallest form
that lets "a table burn on one side and not the other side" along either axis
and one corner alone, and it gives #235 the per-part burn amounts it asks for.

Fire spreads from cell to cell through the tiles: a cell's seam warms the gas
of its tiles, and the cells' tile sets meet along the prop's midlines, so a
burning cell warms air its neighbour also exchanges with. The furniture row
has conductivity 0 ("fire eats it via flammability instead", `config.toml`),
so no conduction inside the prop is the material row's own answer.

**Alternatives:** one cell (fails Erik's question); two halves (cannot burn one
long side); one cell per covered tile (a variable-length member, a side table,
up to 30 cells).

### 9.3 The cell's state and its heat

Per cell (synced, section 2.1): `e_cell` (int64 heat counts above ambient, the
currency every solid capacity is denominated in, `J_per_count` = 0.4759 J,
`materials.py`), `fuel_q` (Q16 units of N_O2 left to burn), `burning`, and
`o2_acc` (the sub-count demand accumulator).

Load constants per cell: the cell's mass `m/4`; its heat capacity `C_cell =
(m/4) x specific_heat` (the material row's own `specific_heat`, 1620
J/(kg K) for furniture), quantized in heat counts per game degree; its seam
area `A_cell` = a quarter of the box's surface (`(2LD + 2LH + 2DH) / 4`);
its fuel `(m/4) / KG_FUEL_PER_N_O2` units of N_O2 (`materials.py`, the tile
law's own fuel per unit of O2, 0.3716 kg).

The cell temperature is `T_cell = e_cell / C_cell` in game degrees (Q16), the
temperature solver's own currency, so it compares directly with the material
row's `ignition_temp` (280 game degrees for furniture, 573 K). v1 mixed game
degrees and the body law's degrees C (physics critic 7); v2 never touches the
body law.

### 9.4 The laws, per cell, per tick

1. **The seam (slot 9c).** `Q_seam = h_conv A_cell (T_cell - T_air) +
   eps sigma A_cell (T_cell_K^4 - T_air_K^4)`, with `h_conv` the solid|gas
   convection the thermal model already uses (`[physics.thermal] h_conv` =
   6.0 W/(m^2 K), `materials._build_conduction_tables`), `eps` the material
   row's `heat_atten` (0.5 for furniture), `T_air` the mean gas temperature
   over the cell's blocking tiles. Its two parts land in different places:
   - the CONVECTIVE part, signed, goes to or from the gas of the cell's
     blocking tiles through the gas energy seam (`GameMap.gas_energy_deposit`,
     channel `prop_seam`, split over the tiles with the fixed-total weights
     `k_q`), debited from or credited to `e_cell`;
   - the RADIATIVE part, out only, is emitted into the radiation sweep as a
     source on the cell's tiles (a new emission channel of the one traversal,
     `dyn_emit_q`, stamped like `dyn_heat_atten_q`), debited from `e_cell`
     and booked by the sweep like any emitter's output. The sweep lands it on
     walls, solids and smoke the way it lands a burning crate's radiation.
     Incoming radiation reaches the cell through the gas: the sweep deposits
     what the prop's tiles absorb into their gas (`e_gas_deposit_sum`,
     energy-books.md), and the convective part carries it on.
   The emission uses the emissive table (`cpp/src/emissive_table.*`, THE
   E°(T) map), never a second T^4. Why the radiative part: without it the
   seam is linear in `T` against a gain that is linear in `T` (through
   `hotf`), and a linear balance has no stable plateau (9.6 (d)). The tile law
   holds its plateau by radiation for the same reason.
2. **Ignition (slot 9d, with the tiles' ignition).** A cell lights when
   `T_cell >= ignition_temp` of its material row, it has fuel, and the oxygen
   test passes: the O2 mole fraction over the cell's blocking tiles, defined
   in integers as `X = sum(N_O2) / sum(N_O2 + N_N2)` over those tiles (the
   tile law defines its test over one tile's open 4-neighbours,
   `combat.apply_temperature_ignition`; determinism critic 13), scaled by the
   O2 pressure factor (`cpp/src/o2_pressure_factor.h`: "a new O2 read goes
   through it"), against `o2_frac_ext`. Heat starts it ("A fire is started by
   DELIVERING HEAT", `.claude/rules/explosions-fire.md`). How easily a class
   catches is its material row's `ignition_temp` and heat capacity, never a
   per-prop flag (R12).
3. **The demand (slot 9e(d'''), right after the heat sources).** The tile
   law's form (`[physics.combustion]` P-O2b and R3 blocks):

   ```
   demand_j = burn_rate x I_cls x hotf(T_cell) x o2f_j x W_hop[d_j] x w_path_j
   ```

   over the cell's reach: hop 1 its own blocking tiles (`w_path` 1), hop 2
   their open 4-neighbours outside the cell's set (`w_path` = the stamped
   permeability of the hop-1 tile it is reached through, the tile law's
   multiplicative path rule); `W_hop` the tile law's table (1, then 2/3);
   `o2f_j` the demand-side linear O2 fraction of tile `j` (`(X_j - X_ext) /
   (X_full - X_ext)`); `hotf = (T_cell - T_ext) / fire_T_span`, capped at
   `hotf_cap` 10, with `T_ext = ignition_temp - ignition_to_ext_delta`.
   `I_cls` is the class's burn calibration (R4), the one number the tile law
   gets from its intensity logistic and a prop does not have. A demand below
   one count accumulates in `o2_acc` (synced: it crosses ticks) as the tile
   law's `dem_acc` does.
4. **The draw, in a fixed order (determinism critic, finding 1).** The draws
   are a stage of the per-env serial applier, PROP_BURN, right after
   HEAT_SOURCES: props in row order, cells FR, FL, BR, BL, the reach in
   row-major order, each draw capped by the O2 STILL LEFT in its tile. The CPU
   runs the identical loop. **Why serial and not an integer proportional
   split:** the burn's writes (the products through the topology seam, the
   gas parcel and the seam through the gas energy seam) are exactly the kind
   of writes the HEAT_SOURCES stage already makes serially per env
   (`cpp/src/cuda_unit_hazards.cuh`); a proportional split would make the
   grant order-free but its writes would still need ordering through those
   seams, so it would add a parallel grant pass without removing the serial
   one. ACCEPTED GAP: where two burning props reach one tile, the lower row
   drinks first (the tile law splits contested O2 evenly); blocking sets never
   overlap, so this happens only in hop-2 tiles. The tile combustion (slot 7)
   runs before 9e, so tiles drink before props, in a fixed order.
5. **What a draw `dO2` does:**
   - **bed heat** `H_bed x dO2` into `e_cell` (booked `e_prop_bed_sum`, the
     prop ledger's source term, as `e_comb_solid_heat_sum` is the solid
     ledger's);
   - **gas parcel** `H_fuel x dO2` into the gas of the tile it was drawn from,
     through the gas energy seam (channel `prop_burn_gas`), as the tile law's
     gas-side yield lands in its claimed air cell;
   - **products** through the shared products function (9.5);
   - **fuel** `fuel_q -= dO2`, floored at zero.
6. **Going out.** A cell stops burning when `T_cell < T_ext`, when the O2 test
   fails, or when its fuel is zero. Burnt out (all four cells empty): the prop
   stays, still blocks and can be pushed, its mass unchanged (ACCEPTED GAP),
   drawn charred (#235); R18.

### 9.5 The products (R5)

The tile law takes the soot share (`soot_yield` 0.5) of the consumed O2 out
of the bulk with its energy and makes the rest inert N2, so total N is
conserved (`cpp/src/combustion.cpp`, the Dalton split and
`e_soot_shed_sum`). v1's `swap_o2_to_inert` kept all of `dO2` as N2 and also
minted the soot, so gas plus trace grew by `0.5 dO2` (about 3.5% pressure in a
room that smothers; physics critic 12). v2 lifts the tile law's per-claim
product step into ONE shared function (patch B0: the tile law calls it
bit-identically; its device form beside `cuda_combustion.cu`), and the prop
burn calls the same function through the topology seam. No second product
law, no `swap_o2_to_inert`.

Why not `extract_gas_n` + `inject_gas_n`: those move bulk with its energy out
of and into a cell as two booked moves at two temperatures with two
roundings; the products step is an in-place conversion whose energy rule
(the soot share sheds its energy, the N2 keeps the cell's) is the tile law's
own. The amendment to the Gas pump primitives line says so (section 14.2).

### 9.6 The steady-state plateau, worked

Does a burning table hold its flame, at what temperature, how many kW reach
the room, and how long does its fuel last? Scratch script
`C:/tmp/props226/v2/plateau.py` (not in the repo), from `config.toml`
(`burn_rate` 0.018, `o2_frac_ext` 0.13, `o2_frac_full` 1.0, `ignition_to_ext_delta`
200, `fire_T_span` 180, `hotf_cap` 10, `H_BED_M` 38.73, `H_FUEL_M` 4.0,
`h_conv` 6.0, the furniture row's 280 / 0.5 / 1620) and `materials.py`
(`J_per_count` 0.4759, `KG_FUEL_PER_N_O2` 0.3716). One cell of a 20 kg table,
1.6 x 0.8 x 0.75 m, at 0.333 m tiles. Its output, verbatim:

```
== the cell
table 20.0 kg 1.6x0.8x0.75 m; cell mass 5.0 kg, C_cell 8.1 kJ/K, A_cell 1.54 m2, own tiles 2.89, hop-2 cells 7.21
H_bed 1.208 MJ/unit, H_fuel 124.8 kJ/unit; o2f(0.21) 0.0920
fuel per cell 13.45 units of N_O2 (5.0 kg / 0.3716)

== (a) I = 0.152, the tile law's measured crate operating point
  Ta     0 game: net at ignition (280)    -5187 W; fixed points []
  Ta   100 game: net at ignition (280)    -3543 W; fixed points [(115.5, 'stable')]

== (b) I that puts the plateau at the tile law's measured crate plateau, 917 K = 624 game
  Ta     0: I_cls 1.137; fixed points [(106.7, 'unstable'), (624.0, 'stable')]
    at T* 624 game: draw 30.07 mN_O2-units/s per cell, bed 36.32 kW, gas parcel 3.75 kW per cell
    whole table (4 cells): 160.3 kW into the room; burnout 448 s (7.5 min)
    net at ignition 280: 6380 W  (>0: a lit cell climbs on its own; lower fixed point [106.7])
    stability d(net)/dT at T*: -77.2 W/K (<0 stable); relaxation time C/|slope| 105 s
    seam split at T*: convection 5.77 kW, radiation 30.55 kW
  Ta   100: I_cls 1.085; fixed points [(624.0, 'stable')]
    at T* 624 game: draw 28.70 mN_O2-units/s per cell, bed 34.67 kW, gas parcel 3.58 kW per cell
    whole table (4 cells): 153.0 kW into the room; burnout 469 s (7.8 min)
    net at ignition 280: 7419 W  (>0: a lit cell climbs on its own; lower fixed point [])
    stability d(net)/dT at T*: -80.2 W/K (<0 stable); relaxation time C/|slope| 101 s
    seam split at T*: convection 4.84 kW, radiation 29.83 kW

== (c) the gas load: where the table's heat lands, per tile, against a burning crate tile
  table: 160.3 kW over 30 tiles = 5350 W of GAS heating per tile
  crate tile (tile law, I 0.152, o2f 0.0825, same hotf 3.02): gas parcel 794 W over its own tile (+ bed 7.69 kW into the SOLID, which radiates)
  ratio prop gas heating per tile / crate's gas parcel: 6.7x  (test_e1_hot_rail trips at 2x H_fuel)

== (d) without the radiative term (convection only): linear against linear
  gain slope per unit I 58.7 W/K; loss slope 9.2 W/K; I below 0.157 always dies, above it runs to hotf_cap (T 1880 game)

== (e) oxygen: the cell's draw against the O2 in reach
  O2 in reach of one cell 2.12 units; draw 0.0301/s -> 70 s to empty it with NO resupply

== (f) the design's split: convection + gas parcel to the gas, radiation into the sweep as an emitter
  per cell at T*: convection 5.77 kW + gas parcel 3.75 kW to the GAS; radiation 30.55 kW into the sweep
  table: 38.1 kW to the gas over 30 tiles = 1271 W per tile = 1.60x a crate tile's gas parcel; 122.2 kW radiated
```

What it says:
- **(a)** At the burn rate a lone crate actually runs at (I 0.152), a table
  cell cannot hold a flame: at its ignition point it loses 5.2 kW more than it
  makes, and it settles near the room's temperature. A prop has far more
  surface per tile of floor than a crate tile, so the crate's number does not
  carry over.
- **(b)** With `I_cls` about 1.1 the cell holds a STABLE plateau at 624 game
  degrees (917 K, the tile law's measured crate plateau), with an unstable
  point at 107 game degrees below the ignition point: once lit by heat, a cell
  climbs to the plateau on its own, and a cell cooled below 107 dies. At the
  plateau the table gives the room about 160 kW and burns its 20 kg out in
  about 7.5 minutes. The plateau relaxes with a time constant of about 100 s.
  That is the default of R4.
- **(c)** If all of that heat went into the gas of the table's tiles, they
  would take 6.7 times the gas-side heat of a burning crate tile; doubling
  the crate's gas-side yield already trips `test_e1_hot_rail` (`config.toml`,
  the `H_FUEL_M` block). That is why the radiative part goes into the sweep.
- **(d)** A seam with no radiative term cannot plateau: the gain and the loss
  are both linear in temperature, so the fire either always dies or runs to
  `hotf_cap` (1880 game degrees).
- **(e)** One cell's reach holds 2.1 units of O2; at the plateau it would
  empty it in about 70 s with no resupply. The `o2f` factor throttles the
  draw as the O2 falls, so starvation sags the plateau instead of a cliff;
  whether resupply holds it at 917 K is UNVERIFIED.
- **(f)** With the design's split the gas takes 38 kW over about 30 tiles,
  1.6 times a crate tile's gas parcel per tile, under the 2x that is known to
  trip the rail but above 1x: marginal, and UNVERIFIED (9.8).

### 9.7 The books

- **The prop ledger.** A parallel identity like the solid ledger
  (`TemperatureSolver.solid_energy_books_sum`, energy-books.md):
  `sum e_cell == e_prop_bed_sum - e_prop_seam_sum - e_prop_emit_sum`, exact in
  int64 per tick, where `e_prop_seam_sum` is the signed convective transfer
  and `e_prop_emit_sum` the radiative emission.
- **The gas identity keeps its four groups.** The prop's writes into the gas
  enter through existing groups: the convective seam and the gas parcel
  through the gas energy seam (the seams group, channels `prop_seam` and
  `prop_burn_gas`), the radiation through the sweep's gas deposit (the
  thermal-solver group), the products through the shared products function's
  booking (the combustion group's soot-shed counter). No fifth group.
- **The sweep's books** gain the prop emission as a source term (its own
  counter), the way a solid's emission is booked; the integer reference
  (`docs/ray_engine_v2_scheme_study_2026-09-13/sweep_ref_q.py`) gets it FIRST
  ("any change lands THERE FIRST").
- **Fuel.** Not an energy column, as a tile's `wall_hp` is not. Per tick,
  exact: `fuel spent == O2 drawn`, `e_prop_bed_sum` delta `== H_bed x O2
  drawn` to the multiply's truncation, the gas parcel `== H_fuel x O2 drawn`
  likewise.
- **When the prop moves.** Nothing is transferred: the reservoir is the
  prop's, the heat already given is the room's, and the next exchange happens
  where the prop now is.

### 9.8 What remains UNVERIFIED, and why arc 2 needs its own critique

- **The emission channel.** Whether the radiation sweep can take a per-tile
  external emission source as a channel of the one traversal, and book it,
  without a second traversal. If it cannot, the radiative part has nowhere to
  go but the gas, and (c) says that trips the rail. This is the first question
  arc 2's critique must settle; the fallback is to wait for P5's gas
  radiative loss channel.
- **The gas load at 1.6x a crate tile.** Below the measured 2x that trips
  `test_e1_hot_rail`, above 1x; only a bench with a burning table says.
- **The plateau target.** 917 K is the crate plateau measured at the slow-burn
  dials before M3 re-derived `H_bed` (`config.toml`, the `H_BED_M` history);
  whether today's crate still sits there is UNVERIFIED, and `I_cls` is
  calibrated against it.
- **`I_cls` above 1.** The tile law's intensity is a logistic state; whether
  its range reaches 1.1 is UNVERIFIED. For the prop it is only a calibration
  number, so the name may change.
- **The seam area and `eps`.** A quarter of the box's surface and the row's
  `heat_atten` are the design's choices, not measured.
- **O2 resupply** through the prop's own permeability (9.6 (e)).
- **The signed gas energy deposit on the device.** `GameMap.gas_energy_deposit`
  adds a signed integer on the host; its device twin in `gas_energy.h` with a
  negative argument, and a floor at zero energy, are UNVERIFIED.
- **Whether the played resident tick re-uploads host gas written at 9e** (the
  pumps are the precedent; not traced, determinism critic).
- **The inherited tile-size dependence.** `KG_FUEL_PER_N_O2` is fixed at the
  0.333 m tile while a unit of N_O2 scales with the tile's volume, 9x off at
  1.0 m (physics critic 15); a prop's fuel inherits it.

The critique round for arc 2 should be a physics-and-books critic and a
determinism critic, on this section and patches B0 to B2 of section 11.

---

# Part C. Common

## 10. The movable prop in the interaction table (#237)

In the merged table's language (`docs/interaction_table_2026-11-21.md` on
main, sections 1 and 6). The table is not edited here; the patch that builds
each arc updates it (its section 6 step 6).

**The new kind (section 1 row):** `PROP` | Movable props | The movable prop
store (`Simulation.props`, `src/simulation/movable_props.py`), its device form
`unit_schema.PROP_TABLE`, its class rows `[movable_props.*]` | It has a pose,
a mass, a stamp every tick and (arc 2) its own fuel: Air, Water, Blasts,
Marines and Fire each need a rule that no static furnishing has. Static props
stay in FURN.

Marks as the table's: **R** a rule (where it lives); **0** none, on purpose
(why); **?** not decided (issue, or "no issue"); **.** not applicable. "Arc 2"
marks what arc 2 adds; the mark given is the one after both arcs.

### 10.1 The row: what a movable prop does to each kind

| PROP acts on | mark | the cell |
|---|---|---|
| AIR | R | Slows flow through its tiles: `dyn_permeability` graded by coverage, MIN after the units' write (3.3; the stamp, `GameMap.stamp_units` + `cuda_stamp.cu` K4; the material row's `permeability`). Displaces and seals nothing (3.5, R1). Arc 2: a burning cell draws O2 from its reach and turns it into the tile law's products (9.4, 9.5; PROP_BURN stage). |
| GAS | R | Slows smoke as it slows air (the trace rides the bulk face flux, Trace transport). Arc 2: a burning cell makes smoke through the shared products function (9.5; `soot_yield`). |
| HEAT | R | Shadows radiant heat: `dyn_heat_atten_q` graded, MAX (3.3). Arc 2: a burning cell gives heat to its tiles' gas through the seam and to the room through the sweep's emission channel, and a cooler cell takes heat from hotter air, booked (9.4, 9.7). |
| FIRE | R | Through heat only (arc 2): the prop never writes `fire`; its heat can light a flammable tile through `combat.apply_temperature_ignition`. A cold prop acts on fire only through the air it slows (through AIR). |
| WAT | ? | None in either arc. #236 (pushing water aside, a floating prop damming a flow). |
| WALL | ? | None: a wall tile is never under a blocking set, and a prop sliding into a wall or glass does no damage. Breaking glass or a wall by impact: #229. |
| FURN | 0 | None, on purpose: a static crate, kindling, grille or foliage tile is not prop-enterable, so a movable prop never enters or moves it (4.5; `prop_system._STAMP_OK_BASE_MATERIALS`). Arc 2: its heat can light a crate through HEAT (named there). |
| DOOR | R | Holds a door open: the door does not close while `prop_owner` is non-zero over its span (`door_system._occupancy_clear` + the DOORS stage, `cuda_unit_doors.cuh`; R8). |
| DEV | ? | None: the motion plate reads units only (`.claude/rules/device-units.md`, signal slots), so a prop on it is not seen; arc 2: the fire sensor (`sensor_system.sample_sensors`, `Channel.FIRE`) reads tile fire, so a burning prop does not trip it. No issue. |
| HAZ | R | Through heat only (arc 2): a burning prop's heat reaches a heat-triggered timed charge's tile (HEAT acts on Hazards). Arc 1: none, on purpose (a hazard reads only temperature and blasts). |
| MAR | R | Slows or stops its walk: `prop_mob_cap` read by `GameMap.is_passable_block` / `footprint_mobility` and the device movement kernel (3.3; the row's `mobility`; R6). Blocks its step. Hands it momentum in a contact (4.6, R3) through `unit_impulse.add`; what follows is the unit law's (sway, stagger, knockdown). A render-only `PropContactEvent`. Arc 2: heats it through heat. |
| ZOM | R | As for marines. Arc 2: `horde.block_feared` (`[zombie] fear_fire`) reads tile fire, so a zombie that fears fire does not fear a burning prop's own tiles (whether it avoids the hot tiles around it depends on what `block_feared` reads, UNVERIFIED). Not decided: no issue. |
| CORP | 0 | None, on purpose: corpses are not stamped and do not block (`door_system._occupancy_clear`: "corpses never block"; Corpse law gap), so a prop slides over a corpse. |
| SWM | 0 | None, on purpose: swarm laws read fields and probe `solid` (Swarm shared kit), and a prop is not solid, so a larva crawls under and over furniture. A prop crushing larvae: #229. |
| RND | ? | None: rounds pass through (7; ACCEPTED GAP). Cover from the row's `cover_exposure`: #227. |
| SPR | ? | None: the spray cone passes through (7). #227. |
| GRN | ? | None: a grenade flies through (7). Bouncing off a prop by the row's `blocks_thrown`: #227. |
| BLST | R | Damps the wave behind it: `dyn_wave_absorb`, MAX (3.3). |
| LIT | R | Casts shadows: `dyn_light_atten_q`, per channel MAX (3.3). Arc 2: a burning prop's room light is render-only and follows it (#235). |
| NOI | ? | None: a hard contact makes no noise. A contact noise through `noise.emit` (loudness by mass and speed) is #229's natural partner; the contact event exists for it. #229. |
| SEE | ? | None: a prop does not block sight (`GameMap.has_los` reads `solid` and `sight_screen`). By the row's `blocks_sight` and the class height: #227. |
| PROP | R | Blocks another prop's move (the start-of-slot `prop_owner` plane); shares momentum in a contact by the common velocity (4.6); a pair that claims one tile in one tick is refused together at its common velocity, three or more stop as on a wall (4.5). Arc 2: its heat warms a neighbour through the air. |

### 10.2 The column: what each kind does to a movable prop

| X acts on PROP | mark | the cell |
|---|---|---|
| AIR | R | Pushes it: the windward-ring read of THE per-tile wind term, a coverage-weighted mean, the force on the projected area, the torque from the per-tile shares, all through `unit_impulse.add` at slot 9c2 (4.3; `[movable_props.<class>] cd`, `[materials.air]` density; R9). Arc 2: the air's O2 feeds and starves a burning cell (9.4). |
| GAS | 0 | None, on purpose: smoke carries no momentum beyond the air that carries it, and only O2 matters to a burning cell. |
| HEAT | R | Arc 2: the air's temperature warms or cools each cell through the seam, booked (9.4); radiation reaches the cell through the gas its tiles absorb into. Arc 1: none (a prop has no thermal state). |
| FIRE | R | Through heat (arc 2): a tile fire beside or under it heats the air its cells exchange with, which may light them; the tile fire also drinks first from shared O2 (slot 7 before 9e). |
| WAT | ? | None in either arc: #236 (pushed by the flow, floating by density, a soaked prop that will not light). |
| WALL | R | Stops it on the blocked axis, booked `wall` (4.6); no bounce (ACCEPTED GAP). |
| FURN | R | A static crate, kindling, grille or foliage tile is not prop-enterable: it stops a prop as a wall does (4.5). |
| DOOR | R | A closed door is not enterable (static `door_closed`); an open doorway is air. A door closing does not push a prop: it refuses to close instead (DOOR row above). |
| DEV | R | Through AIR only: a vent or a pump pushes a prop only through the wind it makes. |
| HAZ | R | Through BLST (a timed charge's blast) and, arc 2, through HEAT (a heat source warms the air its cells exchange with). |
| MAR | R | The shove verb pushes it, limited by hand speed (4.4; `[impulse] impulse_ns`, `[movable_props] v_hand`; R7). A marine's footprint blocks its move (4.5). Walking into it does not push it (R7). Carrying: #228. |
| ZOM | R | A zombie's footprint blocks its move (4.5). Zombies do not shove (UNVERIFIED that the zombie controller never issues the verb). Throwing: #228. |
| CORP | 0 | None, on purpose: a corpse does not block or push a prop. |
| SWM | 0 | None, on purpose: a swarm unit has no push (the swarm meets the unit side only through planes). |
| RND | ? | None: a round passes through. A round pushing a prop (as a hit pushes a unit, `ammo_push_q`): #227. Damage: #229. |
| SPR | ? | None: the cone passes through; its heat reaches a prop only through HEAT (arc 2). #227. |
| GRN | R | Through BLST: their blast pushes it. A charge planted on a prop, a grenade bouncing off one: #227 / #229. |
| BLST | R | Pushes it through its own wind only (4.3; Generic explosion). Arc 2: its heat may light it through HEAT. Breaking it: #229. |
| LIT | 0 | None, on purpose: light carries no momentum and no heat beyond the sweep's heat channel (under HEAT). |
| NOI | 0 | None, on purpose: noise moves nothing. |
| SEE | . | Not applicable: seeing a prop does nothing to it. |

43 cells: 22 in the row (its own diagonal included), 21 in the column. Every
cell has a mark.

### 10.3 Neighbouring cells that change (the table's section 6, step 4)

- **DOOR acts on MAR / ZOM** ("never close on a living unit"): the same
  occupancy check gains props; the unit cells read the same, and DOOR acts on
  PROP says so.
- **MAR / ZOM act on AIR** (the units' stamp): unchanged in behaviour; the
  cell should say that the unit write is now the last-row-wins write among
  UNIT rows, with props combined after it (3.3, 3.4).
- **WAT acts on FURN** stays "?" for static crates; its pointer "floating by
  density is the movable-props work (#226, #236)" moves to WAT acts on PROP.
- **AIR acts on MAR / ZOM**: a prop upwind slows the wind a unit reads, through
  AIR (its permeability). No rule change; the cells could name it.
- **FIRE acts on DEV and on ZOM** (the fire sensor, `fear_fire`): correct for
  tile fire, incomplete once arc 2 lands (new hole 2 below).
- **Section 4 hole 15** ("bodies pass through each other"): still true of
  units; props never overlap each other or a unit's footprint.

### 10.4 New holes (the table's section 4, step 5), most consequential first

1. **A table is no cover** against rounds, sight, sprays or grenades until
   #227: a marine crouched behind a table is hit as in the open.
2. **Arc 2: a burning prop is invisible to every reader of tile fire**: the
   fire sensor does not trip, a zombie that fears fire walks onto its tiles,
   and nothing flames on screen until #235 reads the prop. No issue.
3. **Water does nothing to props** (#236): in the aquarium scene the flood
   flows through the furniture and moves none of it.
4. **Planners do not see props**: bots and zombies path into furniture and
   slide or wait at the step; a wall-like prop in a corridor jams them (R6).
5. **Nothing leaves through a hull breach** (R2): furniture slides to the
   breach and piles at its edge.
6. **A unit flung into a prop gives it nothing** (until #228).
7. **The trained agent cannot see props** (R15): no training level holds them.
8. **Swarm units ignore props**: larvae crawl through a fridge's tiles.

---

## 11. The build (not now)

Rules that hold at EVERY patch boundary:
- **Presence gating.** Every patch is byte-identical to today on a level
  without movable props: the field digest, the entity section and the unit
  stream of the pinned matches and the golden aggregate are unchanged, held by
  the existing gates.
- **The entity-export rule.** `PROP_TABLE` and `prop_entity_rows` land in A2,
  the patch where the state first exists; props first move in A3.
- **The twin ledger** (`tests/twin_manifest.toml`, `tools/twin_debt.py`). A
  patch that touches a Class-P twin's files is Class P: it merges only with
  `.cuda_green` at HEAD and re-marks those twins level
  (`tools/twin_debt.py --mark-level <twin>`) in its last commit. For every
  Class-T twin whose files it touches, it runs that twin's shadow check (the
  change-based tier `tools/affected_cuda.py` selects them) and re-marks it
  level in the same patch; no patch leaves debt for a later one, so the
  device training door (`rl_env.require_snapshot`) never closes across the
  arc. Each patch regenerates the manifest's `files`
  (`tools/twin_manifest_gen.py`). The corpus (ship_demo, 300 ticks) holds no
  movable prop, so the prop laws are in no twin's TRACED files, as the
  grenade laws are not: a change to a prop law lands in its device TU in the
  same patch, and `tests/cuda_unit_props_check.py` is the guard (the grenade
  precedent, `.claude/rules/device-units.md`, Grenade and charge slots).
- **The conductor refusal.** From A2 the device conductor refuses a level
  with movable props by name (`DeviceConductor._refuse`, "movable props");
  A4 lifts it with the whole-tick gate. Each patch's own device twin is
  proved by its own per-slot shadow lockstep on prop scenes, which does not
  need the conductor.

### 11.1 Arc 1

**A0. The shared reads, no behaviour change.**
- Content: the per-tile wind term lifted out of `GameMap.footprint_wind` into
  one function the unit read and `wind_load_plane` call; `unit_impulse.add` /
  `angular_q` take the optional `r_clamp_q` (None = the table's), and the
  device `impulse_add` / `angular_q` in `cpp/src/cuda_unit_lib.cuh` the same;
  `_shrink` / `_clamp_vmax` public.
- Class: **P** (`gamemap.py` is in the files of the Class-P twins `eos`,
  `div_source`, `stamp`): `.cuda_green` at HEAD, those three re-marked level.
- Twins touched, and what it does: Class T through `gamemap.py`
  (`wind_impulse`, `movement`, `stamp_rows`, `tick_end_clears`, `doors`,
  `vents`, `horde`, `sources`, `conductor`, `observation`), through
  `unit_impulse.py` (`integrate_turn`, `wind_impulse`, `movement`, `horde`)
  and through `cuda_unit_lib.cuh` (every Class-T twin listing it): shadow
  checks green, re-marked level. No new twin.
- Refusal: none (no prop exists).
- Oracle: the unit footprint read equals its old form on every footprint of
  random maps (property); the pinned matches' unit streams; the device
  library probe (`cpp/src/cuda_unit_lib_probe.cu`) holds `impulse_add` with
  `r_clamp_q = None` equal to today's over the range.
- Collisions: #29 (`cuda_unit_lib.cuh` is in every device unit twin, and #29's
  P8 work is in those files). May fold into A3 if #29 is busy.
- Rule-file paths: none new.

**A1. The stamp rule as an executable reference.**
- Content: `src/simulation/prop_stamp.py` (Python) and `cpp/src/prop_stamp.h`
  (FP_HD, on `SIM_HEADERS` in `tests/test_no_float_in_sim_tu.py`), the integer
  reference of section 3.2 that any later change lands in FIRST; a test
  binding for C++ == Python.
- Class: **P** by build (`cpp/src/bindings.cpp` is on the manifest's
  `[class_p] build` list): `.cuda_green` at HEAD. No twin's files.
- Refusal: none.
- Oracle: properties 1 to 8 (section 12), including the aligned fixtures and
  the exact headings.
- Collisions: #217 also edits `bindings.cpp` (a merge conflict, not a design
  one).
- Rule-file paths: CREATES `.claude/rules/props.md` with `paths:`
  `src/simulation/prop_stamp.py`, `cpp/src/prop_stamp.h`,
  `tests/test_prop_stamp.py`, and its section `## Prop stamp rule`; adds the
  region-table row and the index row (section 14.3).

**A2. The class, the entity, the store, PROP_TABLE, the stamp on both
backends, walking and doors.**
- Content: `[movable_props.*]` rows and their refusing class door (sizes,
  mass, offset, `mu`, `cd`, permeability floor); the load door's overlap and
  enterability refusals; the `movable_prop` entity; `Simulation.props` with
  its row views in `sim.entities`; the entity record tail; `PROP_TABLE` with
  `props_of`, `props_to_host`, `prop_entity_rows` and the generated layout;
  slot 6 stamps props on the Python and C++ CPU paths AND the device stamp
  kernel (K1 to K4, section 3.4), `take_stamp_rows` packing prop rows and
  `unit_kernels.stamp` building them from the table; the sleeper layer, keyed
  on pose; `prop_mob_cap` and `prop_owner` in `_RESIDENT_UNIT_PLANES`,
  rebuilt at reset and every landing; walking reads the cap on the host
  (`is_passable_block`, `footprint_mobility`) and the device
  (`cuda_unit_movement.cu`, through a resident plane); doors read the owner
  on the host (`_occupancy_clear`) and the device (the DOORS stage,
  `cuda_unit_doors.cuh`); the conductor refusal "movable props"; the
  per-class stamp tables as Device tables rows.
- Class: **P** (`cuda_stamp.cu` and `gamemap.py`: `stamp`, `eos`,
  `div_source`): `.cuda_green` at HEAD; `stamp` changes its device side in
  this patch and is re-marked level with it.
- Twins touched, and what it does: Class T with device changes in this patch
  (`stamp_rows`, `movement`, `doors`): their shadow locksteps gain prop scenes
  (a prop-bearing env beside prop-free ones), rows at tol 0, the two planes
  declared field outputs; re-marked level. Class T touched only by file
  (`gamemap.py`, `simulation.py`, `unit_schema.py`, `object_tables.py`,
  `unit_kernels.py`, `unit_tables.py`, the regenerated layouts: in practice
  every Class-T twin): shadow checks green prop-free, re-marked level.
  `PROP_TABLE` joins the object tables' identity gate
  (`tests/cuda_object_table_check.py` pattern).
- Refusal: the conductor refuses a level with movable props.
- Oracle: prop-free byte-identical (the pinned matches, the golden aggregate,
  `tests/test_stamp_units_cpp_ab.py`); Python stamp == C++ stamp byte for byte
  on prop scenes; `tests/cuda_stamp_check.py` on prop scenes with a unit and a
  prop on one tile and two props' partial tiles overlapping; every
  `cuda_s8a_check` leg (N = 1 == CPU, tol 0); `cuda_batch_invariance_check`
  with props in some envs only; the entity section from the device tables ==
  from the host; properties 9 to 16.
- Collisions: #29 (`conductor.py`, `unit_kernels.py`, `unit_schema.py`,
  `cuda_unit_movement.cu`, `cuda_unit_doors.cuh`, the generated layouts);
  #217 (the stamp's resident wiring in `cuda_resident.h`,
  `cuda_resident_tick.cu/.h`, which #217 edits).
- Rule-file paths: `props.md` gains `src/simulation/movable_props.py`,
  `src/simulation/entities/movable_prop.py`, `tests/test_movable_props.py`
  and its section `## Movable prop store`; `device-units.md` gains its
  section `## Prop table` (its `paths:` already match `unit_kernels.py`,
  `object_tables.py` and `cpp/src/cuda_unit_*`).

**A3. Motion: integration, move, contacts, wind, sleep.**
- Content: `src/simulation/prop_motion.py`: slot 1a integration, slot 3p
  move (double buffer, substeps, refusal pass with the pair rule), the
  contact rule (common velocity, PUSH requests), slot 9c2 windward-ring read,
  the sleep fast path, `PropContactEvent`; the device kernels in a new TU
  `cpp/src/cuda_unit_props.cu` (CMakeLists' CUDA list and `/fp:strict` list)
  for 1a, 3p, its claim pass and 9c2; the PUSH request kind
  (`unit_requests.py`) and the PROP_PUSH stage of the applier
  (`applier.py`, stages renumbered by name); the call lines in
  `Simulation.step()` and `DeviceConductor`.
- Class: **P** if the ring read's helper lives in `gamemap.py` (it does: THE
  per-tile term), so `.cuda_green` at HEAD and `eos`, `div_source`, `stamp`
  re-marked; the prop kernels themselves are Class T.
- Twins touched, and what it does: a NEW Class-T row `[twins.props]` (slots
  1a, 3p, 9c2 of props; device `cuda_unit_props.cu`), born level; Class T by
  file: `simulation.py` (`integrate_turn`, `movement`, `verbs`,
  `burn_through`, `burst_walls`, `tick_end_clears`, `table_appends`,
  `grenades_and_charges`, `rounds`, `doors`, `horde`, `payload`),
  `unit_impulse.py`, `applier.py` (`burn_through`, `tick_end_clears`,
  `melee`, `resolve`, `grenades_and_charges`, `sprays`, `shooting`, `doors`,
  `vents`, `horde`, `payload`, `sources`), `unit_requests.py` (`verbs`):
  shadow checks green, re-marked level.
- Refusal: the conductor still refuses movable props (the whole-tick gate is
  A4's).
- Oracle: the per-slot shadow lockstep of 1a, 3p (with the PUSH requests and
  the PROP_PUSH stage compared as a record list) and 9c2 on prop scenes, rows
  at tol 0, env i of N == alone, the write-set gate; their chained form;
  properties 17 to 27; the momentum books every tick in a scene suite (a blast
  in a furnished room, a slide into a wall, into a unit, into a prop, a pair
  racing for one tile, a three-way pile-up, a breeze over sleepers).
- Collisions: #29 heavily (`applier.py` stages, `unit_requests.py`,
  `conductor.py`, `cuda_unit_lib.cuh`); #217's wind (the scenes' thresholds
  are measured on main's wind and re-measured after #217).
- Rule-file paths: `props.md` gains `src/simulation/prop_motion.py`,
  `tests/test_prop_motion.py`, `cpp/src/cuda_unit_props.cu`,
  `tests/cuda_unit_props_check.py`.

**A4. The shove on props, and the device conductor serves props.**
- Content: `unit_impulse.shove_target` / `shove` consider props, the impulse
  `min(J_shove, m v_hand)`; the device target choice in
  `cpp/src/cuda_unit_verbs.cu` and its PUSH request; `v_hand` as a Device
  tables row; `src/control_bots.py` `try_shove` filters to units; the
  conductor refusal removed.
- Class: **T** (no Class-P file).
- Twins touched, and what it does: `verbs` (device change, shadow lockstep
  with prop targets), `conductor` (refusal removed; whole-tick gate),
  `integrate_turn`, `wind_impulse`, `movement`, `horde` (through
  `unit_impulse.py`): shadow checks green, re-marked level.
- Refusal: none left for arc 1.
- Oracle: `tests/cuda_conductor_check.py` (hand scenes with props, N = 1,
  equal to a host re-simulation every tick) and
  `tests/cuda_conductor_batch_check.py` (env i of N == alone, props in some
  envs, the order swapped); property 25.
- Collisions: #29 (`conductor.py`, `cuda_unit_verbs.cu`); the bots
  (`.claude/rules/bots.md`).

**A5. Drawing them (render-only).** Can start after A2.
- Content: the renderer draws a movable prop's model at its pose every frame
  through the static prop path (`renderer/static_props.py`; UNVERIFIED that
  it takes a per-frame transform today; if not, this patch adds one there,
  never a second prop renderer); the look fields on the class row; a
  studio level, `tools/gen_props_studio.py` writing `levels/props_studio`
  through `level_lib` (a furnished room, a blast, a breach, a shove), with
  its rows in `tools/look_shots.toml`, never used by a test that pins play.
- Class: none (render and tools). No twins.
- Rule-file paths: none in `props.md` (render rules live in
  `.claude/rules/render-units.md`, Static props, amended).

**A6 (outside #226). Levels.** Levelgen places movable props (#171 job 1); the
first golden level that holds them, its goldens moved once with written
rationale; the bot tournament for balance. Waits for #217 to land (its idle
wind would push props on idle ships) and for R15 (no training level holds
props before an observation spec shows them).

Order rationale: A0 and A1 change nothing and de-risk the rest; A2 lands the
stamp on both backends at once, so the played `--cuda --resident` tick never
builds slot 6 without props on a prop level (scope critic 3); A3 before A4
because the shove needs the PUSH path; the conductor's whole-tick gate lands
with the last arc-1 law. #227 can start after A3; #228 after A4.

### 11.2 Arc 2 (after its own critique; after arc 1 and #217 have landed)

**B0. The tile law's products as one shared function.**
- Content: the per-claim product step of `cpp/src/combustion.cpp` (O2 to
  inert N2, the soot share out of the bulk with its energy) lifted into one
  function the tile law calls bit-identically, its device form beside
  `cuda_combustion.cu`, and a device-unit form the applier can call.
- Class: **P** (`combustion`): `.cuda_green` at HEAD, re-marked level.
- Oracle: the tile law byte-identical (the pinned matches, the energy benches,
  the combustion twin's checks).
- Collisions: #217 (`cuda_topology.cuh`, the gas primitives its finding 2
  proposes to change).

**B1. The cells' reservoirs, the seam and ignition.**
- Content: the four cells' synced columns (record tail, `PROP_TABLE`
  columns, layout); the seam at 9c (convection through
  `GameMap.gas_energy_deposit`, signed; the emission channel into the sweep,
  landed FIRST in the sweep's integer reference); ignition at 9d with the
  tile-list O2 test through `o2_pressure_factor.h`; the prop ledger identity;
  host and device.
- Class: **P** (`radiation_sweep`: the emission channel; `gamemap.py`).
- Twins touched: `radiation_sweep` (P, device change), `ignition` (T: the
  9d face reads prop cells too), the `[twins.props]` row (device change),
  the by-file Class-T set as in A2: shadow checks, re-marked level.
- Refusal: the conductor refuses a level holding a FLAMMABLE movable prop
  ("burning props") until B2.
- Oracle: properties 29 and 30 (the prop ledger exact every tick); the sweep's
  integer reference == the CPU sweep == the device sweep with prop emitters.
- Rule-file paths: `props.md` gains `src/simulation/prop_fire.py` (the burn
  cells' laws) and `tests/test_prop_fire.py`.

**B2. The burn.**
- Content: the PROP_BURN applier stage after HEAT_SOURCES (9e(d''')),
  serial; the demand, the draw, the bed heat, the gas parcel, the products
  through B0's function, the fuel; going out; the refusal lifted.
- Class: **P** (`gamemap.py`, the seam's host side) with the Class-T stage.
- Twins touched: `[twins.props]`, `burn_through` and the other applier twins
  (stage renumbering), `doors` / `vents` / `horde` / `payload` (through
  `gas_energy.h` and `cuda_topology.cuh` if the products' device form lands
  there): shadow checks, re-marked level.
- Oracle: properties 31 to 37; the energy closure benches with a burning prop
  (`tests/test_e1_hot_rail.py`, `tests/_quiet_books_bench.py`, a table scene
  added); `tests/cuda_conductor_check.py` with a burning table.
- Collisions: #217 (gas primitives, topology seam).

---

## 12. The verification plan

Properties, never snapshots (CLAUDE.md, Test conventions). Each names what it
protects and the change that must break it. Numbers are referenced from
section 11.

The stamp rule (A1):
1. **Pure and order-free:** a tile's coverage is the same in row-major,
   shuffled or bounding-box order. Breaks if the rule gains running state.
2. **Bounding box complete with pad 0:** with the `+2` pad removed, a scan
   three tiles wider still finds no tile with `k > 0`, because the half-sample
   offset (8192 counts) exceeds the kit's `c^2 + s^2` slack (about 11 counts at
   8 tiles). Breaks if the sample offsets move toward the tile edge or the
   bound drops a term. (v1 named the pad, which is not what keeps it
   complete: determinism critic 11.)
3. **Translation covariance:** an integer-tile shift shifts the map exactly.
4. **Half-turn symmetry and the y-mirror:** `(c, s) -> (-c, -s)` gives the
   identical stamp; mirroring y about a tile line with `th -> -th` (exact in
   the kit) gives the mirrored stamp. The x-mirror (`th -> pi - th`) is NOT a
   property: the kit's values differ there on 39,152 of 82,355 headings
   (determinism critic 12). Breaks if a bound becomes half-open on the whole
   prop.
5. **Burn cells partition the coverage** exactly per tile. Breaks if a
   quadrant bound becomes closed on both sides.
6. **Area bound:** `|sum k / 16 - 4AB| <= 0.07 x perimeter` over random poses
   (measured 0.0602). Breaks if the lattice or the inside test drifts.
7. **The size floor:** at or above `max(0.5 m, 1.5 tiles)` the blocking set is
   non-empty and 4-connected over random poses; the class door refuses below.
8. **CPU == device == Python, bit for bit,** on random poses AND on
   axis-aligned fixtures with edges on tile and sample lines AND at the exact
   headings 0, +-102944 and 205887. Breaks if the inside test becomes open
   (the aligned fixture's `k` goes from 35 to 15; random poses cannot see it:
   determinism critic 10).

The stamp on the map (A2):
9. **No prop, no change:** every plane and every movement answer equals
   today's on every level of the suite.
10. **Order-free combine:** permuting the props' rows leaves every plane
    unchanged.
11. **Cached == fresh, keyed on pose:** a sleeping prop's rows and the sleeper
    layer equal a fresh computation every tick of a scene, INCLUDING after a
    new pose is written with zero velocity and zero impulse (as
    `props_to_host` or the gate's download does). Breaks if the cache is
    invalidated by waking instead of by pose (determinism critic 6).
12. **Never seals:** no prop-stamped permeability is below `P_PROP_MIN`.
13. **The units' write survives:** a unit and a prop on one tile give the
    unit's permeability write, then the prop's MIN, on the CPU and the device
    alike; two props' partly covered tiles overlapping give the MIN of both.
    Breaks if the device K2 lets a prop row win (determinism critic 3).
14. **One writer per blocking tile:** the load door refuses overlapping
    blocking sets, blocking sets on non-enterable tiles and off-grid ones; the
    device writer count is 1 on every blocking tile of every scene.
15. **The planes after a landing:** after `props_to_host`, the gate's
    download and a chained replay, `prop_mob_cap` and `prop_owner` equal a
    fresh stamp of the landed poses (determinism critic 7).
16. **The entity bytes follow the store:** moving a prop changes its entity
    record (`sim.entities` holds views, determinism critic 13); the entity
    section built from the device tables equals the host's.

Motion (A3, A4):
17. **Momentum books close** per prop per axis and angularly every tick, the
    contact transfers balance across all bodies, and no term deletes momentum
    in a contact (the identity of 4.7 has no `contact_lost`).
18. **Friction holds below the threshold:** a steady force below `mu m g`
    never moves a prop and leaves it asleep (all impulse state zero) at the
    end of every tick; above it the prop moves, and when the force stops it
    comes to rest exactly. Breaks if step 4 of 4.2 is removed.
19. **Order-free move:** permuting the props' rows leaves every pose
    unchanged, including two props racing for one tile and a three-way
    pile-up; the device equals the host.
20. **Never overlaps:** after every move no two props share a blocking tile,
    no blocking tile is a wall, a closed door or off the map, and no prop
    newly overlaps a living unit's footprint.
21. **y-mirror physics:** a y-mirrored furnished room in a y-mirrored wind
    gives y-mirrored poses (`th -> -th`). The ring read has no tie rule, so
    no tie breaks the symmetry. Breaks if a truncation becomes a floor.
22. **The wind turns the right props:** an even wind on a balanced prop gives
    zero spin; a gust on one end of a balanced table spins it with the right
    sign; an even wind on an off-centre prop spins it with the right sign.
23. **Tile-size independence:** the same scene built at 0.333 m and at 1.0 m
    tiles gives the same force in newtons within the stamp's grain. Breaks if
    `k_wind_q` is built from a tile count instead of an area (physics
    critic 4).
24. **Contacts keep momentum:** in a prop-into-unit and a prop-into-prop
    contact, `J handed on + m_p v_c == m_p v_p` exactly; a refused pair keeps
    its summed momentum. Breaks on v1's rule (physics critic 3).
25. **The shove is limited:** a shove of a prop of mass `m` gives at most `m
    v_hand` (and at most `J_shove`).
26. **The host's int64 paths equal the Python-int reference** over the whole
    range of `N`, the wind and `k_wind_q` (the 16-bit splits of 4.8). Breaks
    if a vectorized product is taken unsplit (determinism critic 9).
27. **The sleep fast path changes nothing synced:** a breeze over 1000
    sleepers run with and without the fast path gives equal synced columns
    every tick.
28. **The device**, by the existing harnesses only: the shadow lockstep (rows
    at tol 0, env i of N == alone, the write-set gate, the PUSH requests as a
    record list, the prop planes as declared field outputs at tol 0), and from
    A4 the conductor checks with props. Never a hand-rolled compare.

Fire (B0 to B2):
29. **Heat starts it:** no cell burns without `T_cell` at its row's ignition
    point and the O2 test passing; a fixture lights a prop only by delivering
    heat.
30. **The prop ledger is exact** every tick (9.7), and the gas closure
    identity holds with burning props (the existing benches, a table scene
    added).
31. **Fuel identity:** fuel spent equals O2 drawn, per cell per tick.
32. **The products are the tile law's:** B0 leaves the tile law
    byte-identical; a prop burn conserves total N the way a tile burn does
    (gas plus the soot share's accounting). Breaks on v1's swap.
33. **The draw order:** two burning props reaching one tile give equal results
    on the CPU and the device, and no tile's O2 goes below zero (determinism
    critic 1).
34. **One side burns:** heating one end of a table lights that end's cells
    first and the far cells later or never (with the heat cut).
35. **Moving moves nothing:** a burning prop slid across a room transfers no
    fuel, mass or energy at the tile boundaries it crosses, and the floor it
    left keeps its heat.
36. **The plateau band:** a lone burning table in a large room settles within
    a band around R4's target temperature and burns out within a band around
    its time. A property with bands, not a snapshot of a trace.
37. **No rail:** a room with a burning table runs a `test_e1_hot_rail`-style
    window with no rail hit.

---

## 13. Accepted gaps

- `ACCEPTED GAP:` a moving prop displaces no air, and no prop seals (3.5, R1).
- `ACCEPTED GAP:` sight, rounds, grenade flight and the flamethrower's cone
  pass through props until #227 (7).
- `ACCEPTED GAP:` planners (A*, the horde field) do not see props (3.3, R6).
- `ACCEPTED GAP:` one friction coefficient for starting and sliding; linear
  and angular friction separate, up to about 2x too much dissipation when a
  prop slides and turns (4.2, R13).
- `ACCEPTED GAP:` planar pose: tall props slide, never tip (R11).
- `ACCEPTED GAP:` no bounce off walls; a blocked axis stops (4.5).
- `ACCEPTED GAP:` two fast props whose paths cross inside one tick may pass
  through each other; a three-way pile-up in one tick stops all three as on a
  wall (4.5).
- `ACCEPTED GAP:` the hull stops a prop; nothing is blown into space (4.5,
  R2).
- `ACCEPTED GAP:` a unit flung into a prop stops as on a wall and hands it
  nothing (4.6), until #228.
- `ACCEPTED GAP:` the shover feels no reaction (inherited, Unit impulse); bots
  never shove props (4.4).
- `ACCEPTED GAP:` the wind is sampled once per tick; a shock that passes in
  milliseconds is felt by its one-tick sample (4.3, UNVERIFIED size).
- `ACCEPTED GAP:` coarse-tile levels (1.0 m) hold only props at least 1.5 m
  across (3.6, R14).
- `ACCEPTED GAP:` no damage from impacts, no integrity column, no impact
  noise until #229; the contact event is recorded and unread (4.6, R17).
- `ACCEPTED GAP:` water does nothing to a prop until #236 (10).
- `ACCEPTED GAP:` the trained agent cannot see props; no training level holds
  them before a new observation spec (8, R15).
- `ACCEPTED GAP:` swarm units ignore props (10).
- `ACCEPTED GAP:` a level's set of movable props is fixed at load (no spawn, no
  removal, no break-up) until #229 brings stable-slot spawning.
- Arc 2: `ACCEPTED GAP:` where two burning props reach one tile the lower row
  drinks first (9.4); burning does not lighten a prop (9.4); a burnt-out prop
  stays (R18); a burning prop is invisible to readers of tile fire (10.4);
  the fuel per unit of O2 inherits the tile-size dependence of
  `KG_FUEL_PER_N_O2` (9.8).

---

## 14. Systems

### 14.1 Existing canonical systems the design must use

| system (CLAUDE.md index) | how this design uses it | the parallel it must not build |
|---|---|---|
| Simulation facade, Tick conductor | the prop laws are call lines in slots 1a, 1b, 3p, 6, 9c2 (arc 2: 9c, 9d, 9e(d''')) | no logic block in `step()` |
| GameMap | the planes `prop_mob_cap`, `prop_owner`; THE per-tile wind term | no plane outside the map |
| Material / gas tables | a class names ONE material row; every blocking, thermal and fuel value is that row's | no per-prop material if |
| Prop tile stamps | static props unchanged; the one-row principle extended to the dynamic planes | movable props never write `material` |
| Unit impulse, Coupling table, Wind load on a body | THE `add` (with the body's lever clamp), THE per-tile wind term, the shove | no second push law, no second wind reader |
| Footprint samples builder | the cap reaches the walk's speed through `footprint_mobility` | no second speed reduction |
| Turning law (its wrap) | the heading wrapped by `turning.normalize_q16` | no second wrap |
| Fixed-point kits, Q16 boundary modules | `cos_q16` / `sin_q16`, `div_trunc`, `isqrt`, the 128-bit multiply-divide | no re-derived shift or reciprocal |
| Stamp kernel, Device sorts and 9b | prop rows are rows of THE stamp; K4 in the same TU; prop rows built from `PROP_TABLE` by `unit_kernels.stamp` | no second stamp kernel |
| Object tables, Device tables, Device unit library, Request list, Per-env serial applier, Device conductor | `PROP_TABLE`; the class tables as rows; `impulse_add` with the clamp; PUSH requests; the PROP_PUSH (and arc 2 PROP_BURN) stages; the conductor's call lines and refusal | no second table mechanism, walker or request list |
| Entity system, Entity section export | `movable_prop` is an entity class; its record tail through the one serializer; `prop_entity_rows` | no second serializer, no `__props__` carrier |
| Door table | the DOORS stage reads `prop_owner` | no second door sweep |
| Level data layer, Level generators | the props studio and later levelgen write through `level_lib` | no hand-written level.toml |
| Observation spec | v1 unchanged | never a key added to v1 |
| Static props (render), FireVFX pass | the model drawn at its pose; #235 draws the flames | no second prop renderer or flame renderer |
| Arc 2: Gas energy seam, Energy closure identity, Temperature solver | the seam's convective part and the gas parcel through `gas_energy_deposit`; the prop ledger beside the solid ledger | no fifth group, no `temperature[...] =` |
| Arc 2: Radiation sweep, Emissive table, Integer reference | the emission channel on the one traversal, E°(T) from the table, the reference first | no second traversal, no second T^4 |
| Arc 2: Starting a fire, O2 pressure factor | a cell lights by heat; its O2 test goes through the factor | no `burning = 1` without heat; no new O2 read |
| Arc 2: Timed charges and heat sources | PROP_BURN follows HEAT_SOURCES as a stage of the same applier | no second hazard walker |

### 14.2 Existing rules the design amends, with the amended wording

Each amendment lands in the patch named.

1. **CLAUDE.md, Prop tile stamps** (A2). Today: "A prop's blocking + fuel are
   ONE `[materials.*]` row stamped at load, never a per-prop if or a new
   per-tile flag; props never tick; look fields never synced". Amended: "A
   STATIC prop's blocking + fuel are ONE `[materials.*]` row stamped at load,
   never a per-prop if or a new per-tile flag; it never ticks; look fields
   never synced". (`prop_mob_cap` / `prop_owner` are stamp outputs of the
   movable prop store, not flags of static props.)
2. **CLAUDE.md, Wind load on a body** (A0). "THE read of what the air does to
   a unit" becomes "THE read of what the air does to a body".
   `.claude/rules/units.md`, Wind load on a body, gains: "The per-tile term
   `W = N_raw * (v^2 >> 16)` is ONE function: a unit reads its footprint max
   (no lever, v1); a movable prop reads its windward ring's coverage-weighted
   mean with a lever per ring tile (`prop_motion`); a fluid pushing a body
   (#236) is a row of the same read."
3. **`.claude/rules/units.md`, Unit impulse** (A0). "Anything that pushes a
   unit hands it an IMPULSE through `add(...)`" becomes "Anything that pushes
   a unit or a movable prop hands it an IMPULSE through `add(...)`"; new
   bullet: "`add` / `angular_q` clamp the lever to `r_clamp_q`, `None` meaning
   the table's `r_body_q`; a prop passes its own `r_lever_q`."
4. **`.claude/rules/sim-core.md`, Tick conductor** (A3). "Slot 3 is the ONE
   displacement loop (Unit impulse)" becomes "Slot 3 is the ONE displacement
   loop of units (Unit impulse); slot 3p, right after it, is the ONE of
   movable props (Movable prop store), each deciding against the start of its
   slot."
5. **`.claude/rules/device-residency.md`, Stamp kernel** (A2). "reset, then
   MAXes, the permeability a WRITE (the LAST row wins where bodies overlap,
   not a running min)" becomes "reset (`prop_owner` and `prop_mob_cap`
   included), then MAXes, the UNITS' permeability a WRITE (the LAST unit row
   wins where bodies overlap, not a running min; the winner is picked among
   unit rows only), then the movable props' rows as MIN / MAX over it and
   their owner and cap as plain writes (one writer per tile)".
6. **`GameMap.is_passable_block`'s docstring** ("Terrain only") and
   **`cpp/src/cuda_unit_movement.cu`'s header** ("no derived plane") (A2):
   "terrain, capped by the movable props' `prop_mob_cap` (the one derived plane
   both read)".
7. **`.claude/rules/device-units.md`, Entity section export** (A2): in "(button,
   terminal, light, prop, the zones, cover)", `prop` becomes "the static
   `prop`; a `movable_prop` has the Prop table".
8. **`.claude/rules/sim-core.md`, the door sweep's twin rule** (A2): "a change
   to the sweep, `_occupancy_clear` ..." gains "(which also reads
   `prop_owner`)".
9. **`.claude/rules/render-units.md`, Static props** (A5): a movable prop is
   drawn by the same path at its pose every frame; its look fields live on its
   class row.
10. **CLAUDE.md, Gas pump primitives** (B2). "never a parallel flux path or
    per-tick FieldEdits" gains, in `.claude/rules/fields-gas.md`: "A movable
    prop's burn products are the tile law's products function applied in
    place through the topology seam, not a flux: no N crosses a tile."
11. **CLAUDE.md, O2 pressure factor** (B1). "read by both law sites on both
    backends" becomes "read by every law site (the tile sustain, the tile
    demand, the movable prop's burn cell) on both backends".
12. **CLAUDE.md, Heat source** (B2). "A thing hot every tick is a
    `heat_source` entity" becomes "A FIXED thing hot every tick is a
    `heat_source` entity (a burning movable prop is its own burn cells)".
13. **`.claude/rules/explosions-fire.md`, Starting a fire** (B1): "a movable
    prop's burn cell lights under the same rule: by heat at its row's ignition
    point, never by a write of `burning` alone".
14. **`.claude/rules/energy-books.md`, Energy closure identity** (B1): beside
    the solid ledger, "the movable props' ledger: `sum e_cell` closes against
    `e_prop_bed_sum - e_prop_seam_sum - e_prop_emit_sum`; its gas writes
    enter through the seams group, its emission through the sweep".
15. **`.claude/rules/radiation-light.md`, Radiation sweep** (B1): the prop
    emission is a channel of the one traversal, landed in the integer
    reference first.

The index lines amended in 1, 2, 11 and 12 keep their length to within a few
words ("a patch that changes a system amends the rule file, never this line's
length").

### 14.3 New index rows (three systems, one tool)

| row (index section) | where | draft rule | rule file, section |
|---|---|---|---|
| **Movable prop store** (Sim core, after Prop tile stamps) | `src/simulation/movable_props.py` (`Simulation.props`) + `entities/movable_prop.py` + `[movable_props.*]` | THE state of movable props, SoA in entity ordinal order, written only through this facade and its laws (`prop_motion`, arc 2 `prop_fire`); never a `Unit`, a swarm unit or a material stamp; a new column is a schema row, its entity record row and its `PROP_TABLE` column in one patch | `.claude/rules/props.md`, `## Movable prop store` (A2) |
| **Prop stamp rule** (Sim core) | `src/simulation/prop_stamp.py` + `cpp/src/prop_stamp.h` | THE integer rule that turns a turned rectangle into tile coverage (a 4 x 4 global sample lattice, a closed inside test, half-open quadrants); the Python form is the executable spec and any change lands there first, the FP_HD form held equal; nothing else rasterizes a prop | `.claude/rules/props.md`, `## Prop stamp rule` (A1) |
| **Prop table** (Device, after Door table) | `unit_schema.PROP_TABLE` + `unit_kernels.props_of` / `props_to_host` / `prop_entity_rows` + `cpp/src/cuda_unit_props.cu` | THE device form of the movable prop store on the object tables, and of its slots 1a, 3p and 9c2 (contacts and shoves as PUSH requests through the applier's PROP_PUSH stage); a store column that crosses a tick is a table column in the same patch | `.claude/rules/device-units.md`, `## Prop table` (A2) |
| **Props studio** (Tools) | `tools/gen_props_studio.py` -> `levels/props_studio` + its rows in `tools/look_shots.toml` | THE level for judging movable props in wind, blasts and shoves, written by its generator through `level_lib`; never used by a test that pins play | none (a tool row; A5) |

Not systems (scope critic 9): the prop planes are stamp outputs (rule lines in
the Stamp kernel section and in `props.md`); the motion and the burn cells are
laws of the store (bullets of `## Movable prop store`); v1's O2 swap is gone
(9.5), and the products function is an amendment of the combustion twin's
rules, not a row.

### 14.4 The new rule file `.claude/rules/props.md`

`test_claude_rules_paths.py` fails a `paths:` glob that matches no tracked
file and a section with no index row of the same name, so the file is created
by the first patch that creates its first path, and each path is added by the
patch that creates it. Exact paths, no wildcards that catch other tests
(v1's `tests/*prop*` matched `test_velocity_clamp_property.py` and others):

| patch | `paths:` added | section added |
|---|---|---|
| A1 | `src/simulation/prop_stamp.py`, `cpp/src/prop_stamp.h`, `tests/test_prop_stamp.py` | `## Prop stamp rule` |
| A2 | `src/simulation/movable_props.py`, `src/simulation/entities/movable_prop.py`, `tests/test_movable_props.py` | `## Movable prop store` |
| A3 | `src/simulation/prop_motion.py`, `tests/test_prop_motion.py`, `cpp/src/cuda_unit_props.cu`, `tests/cuda_unit_props_check.py` | (bullets of the store section) |
| B1 | `src/simulation/prop_fire.py`, `tests/test_prop_fire.py` | (bullets of the store section) |

The region table in CLAUDE.md gains `| props.md | movable prop store, prop
stamp rule |` in A1 (with only the stamp rule) and is completed in A2.

---

## 15. Borrowed dials: tuned for other objects or under the #217 bug

For Erik (physics critic, finding 16): the numbers this design reads that
were set for something else, or tuned while transport ran at a third of its
speed under the #217 bug. None is changed by this design; each is a place a
prop may feel wrong for a reason that is not the prop's.

| dial | where | set for | what it does to a prop |
|---|---|---|---|
| `Cd` (new) | `[movable_props.<class>] cd`, proposed 1.1 | the flat-plate value, no source | scales every wind push |
| frag rows, `expansion_m3_per_strength` | `[payloads.*]`, `[blast]` | grenades against units, tuned under the bug | how far a blast's wind throws furniture |
| `k_drag2` | `[physics]` drag block | tuned while transport ran at 1/3 | the wind a prop reads after a blast |
| the wind sampled once per tick | slot 9c2 | units, which react slowly | a shock passing in ms is felt as one sample (UNVERIFIED size) |
| `impulse_ns` 100 | `[impulse]` shove | a two-handed shove on a person | the shove's ceiling; `v_hand` limits it for light props |
| `V_MAX_Q` 4 tiles a tick | `unit_impulse.V_MAX_Q` | units at 0.333 m (32 m/s; 96 m/s at 1.0 m) | the prop's top speed |
| `mu_slide` 0.4 | `[impulse]` | a prone body on deck | the default `mu` of a class without one |
| `burn_rate` 0.018 | `[physics.combustion]` | tile fires, "re-tune pending" | arc 2's demand scale |
| `H_FUEL_M` 4.0 | `[physics.combustion]` | a gas-side placeholder at the rail's ceiling | arc 2's gas parcel |
| `H_BED_M` 38.73 | `[physics.combustion]` | a calibrated lumped constant for a crate's bed | arc 2's bed heat |
| `soot_yield` 0.5 | `[physics.combustion]` | provisional, for smoke visibility | arc 2's smoke |
| `h_conv` 6.0 | `[physics.thermal]` | natural convection on a deck-high wall | arc 2's seam (a fire plume is forced convection, 10 to 100) |
| `KG_FUEL_PER_N_O2` | `materials.py` | fixed at the 0.333 m tile | arc 2's fuel, 9x off at 1.0 m |
| `heat_flux_to_temp`, `unit_absorption`, `unit_reflectivity`, the 159 W/K body loss | the body law | human skin and a person's shell | NOT used any more (v1 borrowed them for the cells; v2 does not) |

---

## 16. Critique

Three critics read v1 (commit `a647e771`), one lens each, none seeing
another's report (`docs/movable_props_critique_2026-11-21.md`).

| lens | verdict on v1 | blockers |
|---|---|---|
| scope, systems and rules | SOUND WITH FIXES | 1 (S.1: host-only motion breaks the entity-export rule and opens Class-T debt) |
| physics and the books | UNSOUND (the fire law); the push law fixable in place | 1 (P.1: the fire neither holds its flame nor closes the books) |
| determinism and the device path | SOUND WITH FIXES | 1 (D.1: several writers draw O2 from one tile) |

Where two critics touched one thing, v2 gives one resolution: the fire law
(P.1, P.2, P.7, P.12, D.1, D.13: section 9), the contact rule (P.3, P.14, D.2,
D.5: sections 4.5, 4.6, 6), the patch plan (S.1, S.2, S.3, S.4, S.8, D.8:
section 11).

What v2 did with each finding:

- **S.1** Taken. `PROP_TABLE` and `prop_entity_rows` in A2, where the state
  first exists; every patch re-proves and re-marks the twins it touches; no
  debt carried (11).
- **S.2** Taken. Every patch touching `gamemap.py` (A0, A2, A3, B1, B2) or
  `bindings.cpp` (A1) is Class P with `.cuda_green` (11).
- **S.3** Taken: the host and device stamps land together in A2 (11).
- **S.4** Taken: two arcs (R0), the smallest arc's patch list (A0 to A4); the
  collisions with #29 and #217 named per patch. Partly not taken: the unit
  hand-off and the door read stay in arc 1 (the PUSH path is built for prop to
  prop anyway; without the door read a door would close on a prop), and the
  felt-temperature refactor is gone entirely (arc 2 no longer uses it).
- **S.5** Taken: fifteen amendments with wording (14.2).
- **S.6** Taken: planar pose (R11), how easily it burns (R12), the integrity
  column (R17), the door held open (R8), no blowing out to space (R2),
  wall-like props while planners are blind (R6), fire as a second arc (R0).
- **S.7** Taken: the row and the column in the table's kinds and codes, every
  cell closed, the neighbouring cells and the new holes (10).
- **S.8** Taken: exact paths, created patch by patch (14.4).
- **S.9** Taken: three index rows plus a tool row (14.3).
- **S.10** Taken: (a) the row view and `r_clamp_q` (4.1); (b) per-instance
  constants as `PROP_TABLE` columns, per-class ones as Device tables rows
  (2.1); (c) the device shove in A4, the bots filtered (4.4); (d) a flung unit
  stops as on a wall, a gap (4.6); (e) a new class, look fields on its row,
  A5 draws it (2.2, 11); (f) not hot-reloadable (2.2); (g) the props studio
  (11, 14.3); (h) the device walker reads the cap in A2 (3.3).
- **S.11** Taken: the quote is attributed to Claude (1.1).
- **S.12** Taken: only `dyn_heat_atten_q` and `obstacles` are digested (2.3).
- **S.13** Taken: 1000 sleepers priced, the sleeper layer and the wind fast
  path (5, 8).
- **S.14** Taken: the ring read is named as the read #236 extends (1.1, 4.3).
- **P.1** Taken as the orchestrator decided: the cell a booked reservoir, the
  bed heat in it, a booked seam, the gas parcel to the gas, the plateau
  calculation (9). Added beyond the finding: the seam's radiative part goes
  into the sweep, because (9.6 c, d) a linear seam cannot plateau and an
  all-gas seam trips the rail.
- **P.2** Taken: the tile law's demand form with `hotf`, `o2f`, `W_hop`,
  `w_path` over a two-hop reach, and a per-class calibration (9.4, R4).
- **P.3** Taken: common velocity, momentum kept (4.6, R3).
- **P.4** Taken: `k_wind_q` from an area in m^2 through `_wind_k`; `Cd` a
  declared class column (2.1, 4.3).
- **P.5** Taken: the windward ring outside the covered set, the
  coverage-weighted mean (weight = contact faces), the torque from the
  per-tile shares (4.3, R9). The weight is the ring tile's contact count, the
  one coverage measure that does not depend on where the prop sits within its
  edge tiles.
- **P.6** Taken: minus, and the refusal floor (2.1, 4.2).
- **P.7** Taken by removal: the cells no longer use the body law; one currency
  (game degrees); the room-cooling gap is closed by the booked seam (9.3,
  9.4). Its numbers no longer need a ruling.
- **P.8** Taken: `J = min(J_shove, m v_hand)` (4.4, R7).
- **P.9** Taken: the floor in metres and tiles, the coarse-tile gap stated
  (3.6, R14).
- **P.10** Taken: R11, default "slides".
- **P.11** Taken: R1 with the 72% fridge figure.
- **P.12** Taken: the tile law's products through one shared function (9.5,
  R5); the ACCEPTED GAP is gone.
- **P.13** Taken: the box volume is stored and the meaning of density is
  #236's, as R16.
- **P.14** Taken: `clamped_l` booked; the 2x friction overestimate in R13's
  text; the refusal pass keeps momentum (4.2, 4.5).
- **P.15** Noted, inherited: in 9.8, 13 and 15; not fixed here (it is the tile
  law's).
- **P.16** Taken: the dials table (15).
- **P.17** Taken: the rulings list rebuilt (17). R15 (training) stays on the
  list as a scope ruling, marked as one.
- **D.1** Taken: the draws as a serial applier stage, PROP_BURN. The integer
  proportional split NOT taken, because its writes would still need ordering
  through the seams the serial stage already orders (9.4).
- **D.2** Taken: contacts and shoves as PUSH requests through the applier
  (6).
- **D.3** Taken: K1 resets both planes, K2 picks among unit rows, K4 for
  props; the oracle scenes (3.4).
- **D.4** Taken as the load refusal plus plain writes, checked by a writer
  count. `atomicMin` on owner and cap NOT taken: one writer per tile is an
  invariant the move already keeps, and a MIN would hide its violation instead
  of refusing it (3.3).
- **D.5** Taken: the double buffer and the hit-set rule with its partners
  (4.5, 4.6).
- **D.6** Taken as a pose-keyed cache on both backends with a sleeper layer.
  "Recompute every tick on the device" taken for awake props only: at 1000
  sleepers per env the layer is cheaper and the pose key makes it as safe
  (5).
- **D.7** Taken: the planes rebuilt at reset and every landing,
  `_RESIDENT_UNIT_PLANES`, declared outputs at tol 0 (2.3, property 15).
- **D.8** Taken: the four device twins in A2 to A4, the refusal by name from
  A2 (11).
- **D.9** Taken: ranges at the class door, the 128-bit steps, the 16-bit
  splits on the host, the zero-wind return (4.8).
- **D.10** Taken: aligned fixtures and exact headings in property 8.
- **D.11** Taken: property 2 restated.
- **D.12** Taken: the y-mirror only; the ring read has no tie rule (properties
  4, 21).
- **D.13** Taken: the substep shift and split (4.5), token row names (2.1),
  row views in `sim.entities` (2.2, property 16), the synced O2 accumulator
  (2.1, 9.4), the integer tile-list O2 test (9.4).

Not taken, in one place: D.1's proportional split (serial chosen), D.4's
`atomicMin` (refusal plus writer count chosen), the part of S.4 that would cut
the unit hand-off and the door read from arc 1, and P.15 (inherited, the tile
law's). Arc 2 has had no critique of its v2 form; it needs one.

---

## 17. Rulings for Erik

One line each, most game-changing first. Each states the default this design
assumes, then what changes if Erik says otherwise. **[arc 1]** = needed before
arc 1 starts; **[arc 2]** = needed before arc 2 starts; **[later]** = can wait.

- **R0 [arc 1] Two builds:** first props that exist, block and get pushed
  (wind, blasts, shove); fire on props in a second build after its own review.
  If one build: nothing lands until fire is done too, about twice as long.
- **R1 [arc 1] No prop seals air:** furniture slows gas but a barricade never
  stops a decompression (a solid fridge's 72% of the air column is ignored).
  If props may seal: a prop becomes a moving wall to the gas, a much larger
  change.
- **R2 [arc 1] Props stay inside the hull:** a breach slides furniture to the
  hole and it stops there. If they should blow out into space: props need a
  removal path, which waits for #229.
- **R3 [arc 1] A sliding prop shares its speed with what it hits:** a fridge
  sliding into a marine carries him along (he may stagger or fall), no damage
  until #229. If no: the prop just stops at a person.
- **R4 [arc 2] A burning table settles at about 645 C (917 K, like a burning
  crate), gives the room about 160 kW and burns out in about 7.5 minutes.** If
  hotter, cooler, faster or slower: one number per kind of prop.
- **R5 [arc 2] A burning prop uses the room's oxygen and makes the same smoke
  as a burning crate** (a sealed room starves it). If no: it is cheaper, but
  once lit it burns to the end even in a sealed room.
- **R6 [arc 1] Climb or wall, by material:** furniture is climbable at 40%
  speed like a crate today; a wall-like material blocks, and then bots and
  zombies can jam on it because their path planning does not see props. If
  every prop must be climbable in this build: no jams, no barricades.
- **R7 [arc 1] People push props only with the shove,** at most as fast as a
  hand moves (2 m/s: a chair or table slides about 0.4 m, a fridge 0.1 m);
  walking into one does not push it. If walking should push: light props
  drift away from walkers and the bots must cope.
- **R8 [arc 1] A prop in a doorway holds the door open:** an airlock cannot
  seal on a table. If doors should close anyway: a rule for what the door
  does to the prop (push it, crush it).
- **R9 [arc 1] Wind pushes on the side of a prop that faces it, read in the
  air just outside it:** a gust on one end can spin a table. If the simpler
  strongest-single-spot read is wanted: cheaper, but jumpy, and a balanced
  table never spins.
- **R10 [arc 2] Four burning parts per prop** (front-left, front-right,
  back-left, back-right), so one side can burn alone. If two halves: half
  the fire state, one long side cannot burn alone; if per tile: more state.
- **R11 [arc 1] Props slide, never tip:** a tall bookcase slides like a crate
  (in reality it would tip first). If tipping: a fallen state, a new
  footprint, and more work.
- **R12 [arc 2] How easily a prop catches fire comes from its material only**
  (its ignition temperature and heat capacity), not a per-prop setting. If
  per prop: one column per kind of prop.
- **R13 [arc 1] One friction value per prop, the same to start and to keep
  sliding** (default 0.4); sliding and turning friction counted separately
  (up to twice too much drag when a prop slides and turns). If "hard to get
  going, then slides": two values.
- **R14 [arc 1] A movable prop is at least 0.5 m across its smaller side;**
  anything smaller is drawn only (debris). If smaller ones should move:
  thin props need another blocking rule.
- **R15 [later] No training level holds movable props** until a new
  observation spec lets the agents see them. If earlier: agents train beside
  things they cannot see.
- **R16 [later] Density is mass over the box size** (stored now; then a table
  and a fridge both float); #236 decides whether floating uses that or the
  material's own density. If material density: decided in #236.
- **R17 [later] Props have no hit points** until #229. If now: an integrity
  column and a damage rule in this build.
- **R18 [later] A burnt-out prop stays where it is, charred, still blocking
  and pushable,** until #229 decides breaking. If it should vanish or turn to
  ash: a removal path, which waits for #229.
