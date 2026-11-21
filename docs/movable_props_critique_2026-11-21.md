# Movable props: the critics' verdicts

Companion to `docs/movable_props_design_2026-11-21.md`. Three critics read
design v1 (commit `a647e771`), one lens each, in parallel, none seeing
another's findings. Their reports as returned, shortened only in wording. The
synthesis is in the design doc (v2).

## Critic: scope, systems and rules (Opus, 2026-11-21)

**VERDICT: SOUND WITH FIXES.** The architecture holds (its own entity class,
an SoA host store, a table on the object tables, the existing impulse door,
fire as a moving heat source). What breaks: the patch plan against the twin
ledger and the entity-export rule, the list of rule amendments, scope, and
missing rulings.

### Findings, most severe first

1. **P4 makes props move on the host only: it breaks the entity-export rule
   and leaves Class-T debt open from P4 to P6.** BLOCKER. device-units.md:236:
   "A class whose state starts moving gets a table and an entity-row function
   in the same patch"; the design quotes it (2.2) and then puts `PROP_TABLE`
   and `prop_entity_rows` in P6. `tools/twin_debt.py` counts a twin in debt
   when any of its listed files changed since its `level_at`:
   `unit_impulse.py` (T twins integrate_turn, movement, wind_impulse, horde),
   `exchange.py` (gas_rows, heat_damage, wind_impulse), `gamemap.py`
   (conductor, observation, doors, movement, vents ...). P0, P2, P4, P5 all
   edit them. `require_snapshot(tag, "device")` refuses while any debt is
   open: the device training door (#29's product) stays shut for the whole
   window. Fix: (a) the `PROP_TABLE` schema, its record / way-back functions
   and `prop_entity_rows` move into the patch where state first moves; (b) for
   every patch, list the twins whose files it touches and either run their
   shadow checks plus `--mark-level`, or name the debt and its catch-up patch.
2. **Only P3 is named as needing `.cuda_green`, but `gamemap.py` belongs to
   Class-P twins** (eos, stamp, div_source). FIX. Every patch that edits it is
   Class P by file: P0 (the `footprint_wind` split), P2 (new planes,
   `is_passable_block`, `footprint_mobility`), P5 (`swap_o2_to_inert`). Mark
   them, with `.cuda_green` at HEAD.
3. **Main is not green at the P2 / P3 boundary on a prop level.** FIX. P2
   stamps props into the host planes; until P3 the played `--cuda --resident`
   tick builds the slot-6 planes on the device from unit rows only. Fix: merge
   P2 and P3, or P2 makes `take_stamp_rows` return None whenever props exist
   (the host planes are uploaded), stated and gated.
4. **The arc nests other arcs; the smallest first arc is much smaller.** FIX.
   Nested: fire (P5: four cells, a new gas primitive, three slots, a
   felt-temperature refactor); the shove verb's new target (hand-verbs: the
   device shove in `cuda_unit_verbs.cu`, the bots' `try_shove`); momentum from
   a prop to a unit (R5); the door occupancy read; contact events (#229).
   Smallest arc that gives a table a blast pushes and that blocks wind: P0 cut
   to the wind tile-list split (plus the lever clamp only if off-centre
   turning stays); P1; P2 and P3 merged (store, entity, stamp into the four
   dynamic planes, the mobility cap); P4 cut to integration, move, wind,
   sleep, walls and props as obstacles; the P6 subset for those laws. Fire is
   the second arc. Erik put burning into step 1 himself (comment 1), so the
   split is HIS call: a ruling, not a unilateral cut. Collisions with the
   running tracks: #217 (P5 adds a primitive beside `inject_gas_n` /
   `extract_gas_n` in gamemap.py and cuda_topology.cuh, the primitives #217's
   finding 2 proposes to change; #217's idle wind of 20 to 30 on generated
   ships would push props around on idle ships, so P7 depends on #217
   landing); #29 (P0's felt generalisation touches
   `unit_lib::felt_temperature_q`, which `cuda_obs_windows.cu:60` includes and
   labels "used, never added to"). Say all of this in section 10.
5. **The list of rules to amend is incomplete.** FIX. Changed but not amended:
   Prop tile stamps ("never ... a new per-tile flag": `prop_owner`,
   `prop_mob_cap` are new per-tile planes); the Stamp kernel law ("the
   permeability a WRITE (the LAST row wins)" becomes write-then-MIN); Heat
   source ("A thing hot every tick is a `heat_source` entity"); Unit impulse /
   Wind load index lines ("a unit"; units.md: wind has "no lever");
   sim-core.md ("Slot 3 is the ONE displacement loop": slot 3p is a second);
   `is_passable_block`'s "Terrain only" and device movement's "no derived
   plane" (`cuda_unit_movement.cu:38-40`); device-units.md:236 lists `prop`
   among the classes that never move; Gas pump primitives ("never a parallel
   flux path") for the swap; O2 pressure factor ("a new O2 read goes through
   it": the burn's O2 draw is a third O2 sustain site not routed through
   `o2_pressure_factor.h`). List each, and why the swap cannot be
   `extract_gas_n` + `inject_gas_n`.
6. **Rulings missing from section 14** (Erik's calls the design makes itself).
   FIX. R10 planar pose (x, y, heading only, no tipping; Erik said "at least 3
   numbers or 6 numbers"); R11 how easily it burns (per material row only,
   against Erik's "some of them should have the property how easily they can
   burn"); R12 the integrity column (#229 asked for it "from day one"; 1.3
   declines); R13 a prop holds a door open (3.3: a table blown into an airlock
   or hull doorway keeps it from sealing); R14 no blowing out to space (4.5: a
   hull breach never ejects furniture; in a game called Breach a headline feel
   call, not an accepted gap); R15 mobility-0 classes while planners cannot
   see props (a fridge in a corridor jams the horde and the bots); R0 fire as
   a second arc.
7. **The interaction row (section 9) does not follow the table's own kinds**
   (`docs/interaction_table_2026-11-21.md`, sections 1 and 6). FIX. Missing
   rows: Sprays and Sight (only in section 7); Zombies, merged into units,
   which hides `fear_fire` (`horde.block_feared` reads tile fire, which a
   burning prop never writes); Furnishings (static crate, kindling, foliage
   tiles against a movable prop); Hazards. Devices: the fire sensor
   (`Channel.FIRE`) is blind to a burning prop; "no such sensor reads bodies
   today" is wrong (the motion plate reads units, device-units.md:223). "Not
   decided" with no issue: space / vacuum, a prop on a sensor. The table's
   section 6 steps 4 (neighbouring cells that change) and 5 (new holes) are
   not done. Use the table's codes and close every cell.
8. **The new rule file `props.md` would fail its own test and fire in the
   wrong places.** FIX. `tests/*prop*` matches
   `test_velocity_clamp_property.py`, `test_drag2_dead_zone_property.py`,
   `test_thermal_v2_properties.py` and others; `prop_motion.py`,
   `prop_stamp.py` lack their `src/simulation/` prefix;
   `cpp/src/cuda_unit_props.cu` does not exist until P6 (CLAUDE.md: "files
   that exist"). Exact paths; the file created by the patch that creates its
   first path; each path added by the patch that creates it.
9. **The seven "new systems" are three or four index rows.** FIX. Real: the
   store (motion and the burn cells as its laws in `props.md`), the stamp rule
   (an integer reference with an FP_HD twin), the device table. Not systems:
   the prop planes (stamp outputs: rule lines in fields-gas.md and
   device-residency.md); the O2 swap (an extension of the gas energy seam and
   the gas primitives: an amendment). Say which index section each row joins.
10. **Where two implementers would build different things.** FIX. (a) The
    "row view" that lets `add` work (`add` writes `unit.impulse_books[IN_X]`
    and calls `angular_q(table, ...)`, unit_impulse.py:488-503): where the
    body's clamp comes from, what the prop's books look like. (b) Where class
    load constants and the 17-entry stamp tables live on the device
    (`PROP_TABLE` columns, or rows of the Device tables: "never a law constant
    as a kernel argument"). (c) `shove_target` returning a prop:
    `control_bots.try_shove` reads `t.team` (control_bots.py:1177); the device
    shove is in `cuda_unit_verbs.cu`; neither is in any patch. (d) A unit
    flung by a blast into a prop stops as on a wall and hands the prop no
    momentum: not stated. (e) Who draws a movable prop: the existing `prop`
    entity keeps its look fields on the entity; do they go on `movable_prop`
    or `[movable_props.*]`; is a new class better than extending `prop`; how
    does `static_props.py` get a pose that changes every frame. (f) Whether
    `[movable_props.*]` is hot-reloadable (the config reload seam). (g) Which
    generator writes P2's test level (a new studio tool is a Tools index row).
    (h) The device twin of the walker reading `prop_mob_cap`
    (`cuda_unit_movement.cu`) is in no patch.
11. NOTE. One quote in 4.3 attributed to Erik is Claude's gloss in comment 2
    ("an off-centre mass makes a push at one end turn the prop"): drop
    "Erik:".
12. NOTE. 2.3 misreads the digest spec: only `dyn_heat_atten_q` and
    `obstacles` are digested, not `dyn_light_atten_q`. The conclusion (no
    golden moves) holds.
13. NOTE. Cost (section 8) is labelled as estimates and argued at N = 128, but
    the always-on wind and thermal reads of SLEEPING props on the device are
    not costed, and #226 asks for "Thousands of props that never move should
    cost nothing" while the design assumes 20 to 60. One paragraph at 1000
    sleeping props.
14. NOTE. Water: Erik's "done at the same time" is bent (1.3 defers to #236).
    #236 asks for one law "a fluid pushes a body" with air and water as two
    rows: name P0's tile-list read as that read, so #236 extends it.

### The owner's notes against the design

Honoured: furniture in the state moved by wind and explosions; a kind of its
own, a size, a stamp, burning, the cost view; debris render-only; burning as a
moving heat source; one side burning; friction unique per item; centre of
mass; density stored; shields deferred to #227 as his order says; the swarm
idea replaced by his own second note. Bent with no ruling: orientation "3 or 6
numbers"; "how easily they can burn"; water "at the same time"; #229's
integrity column. Section 14: every line states its default and can be
answered in one line; none is a false decision.

The brief's list (#238 track B step 2) is all present; the table row is
incomplete (finding 7).

### Checked and found true

- `add` and `angular_q` clamp the lever to `table.r_body_q`; `add` is an
  integer sum into `imp_*` and the books (unit_impulse.py:458-503).
- `footprint_wind` takes the max over gas tiles, ties to the first in
  row-major order; `footprint_wind_load` delegates to it
  (gamemap.py:2423-2473).
- The stamp kernel's permeability is a write, the last row wins.
- `_STAMP_OK_BASE_MATERIALS = (MAT_AIR,)` (prop_system.py:47);
  `_occupancy_clear` skips corpses and takes units only (door_system.py:203).
- `dyn_permeability` and `dyn_wave_absorb` are excluded float fields;
  `dyn_heat_atten_q` is digested.
- Adding an entity class changes only `registry_content_hash` (provenance, not
  hashed into goldens): "no golden moves" holds.
- Observation v1's `WINDOW_PLANES` (device_obs.py:677) hold no stamp plane: a
  prop reaches an agent only through wind, heat and `solid` (supports R9).
- The device twins of the wind and felt reads exist
  (`cuda_unit_impulse.cu`, `cuda_unit_lib.cuh`, `cuda_unit_damage.cuh`).

UNVERIFIED by the critic: that P0's refactor could re-mark twin levels without
touching a kernel; whether `paths:` globs that match nothing fail
`test_claude_rules_paths.py`.

## Critic: determinism and the device path (Opus, 2026-11-21)

**VERDICT: SOUND WITH FIXES.** The integer stamp rule holds: exact, the same
in any visiting order, CPU and device trig agree bit for bit. The holes are in
what the design leaves unspecified: how several writers share one tile or one
row, what happens when state is LANDED on a sim by a path other than stepping,
and the int64 ranges. Scratch: `C:/tmp/props226/critic_det/` (`kit.py`,
`exp1.py` to `exp4.py`, `rerun.txt`).

### Findings, most severe first

1. **The prop burn's O2 draw is "capped by the O2 there", but several writers
   draw from the same tile.** BLOCKER. The four quadrant cells split one
   tile's samples (3.7's tile (12,7): FR 2 / FL 6 / BR 2 / BL 6), and a tile
   partly covered by one prop can be covered by another. In parallel (section
   8: "one thread per prop, order-free") each cell caps against the full O2
   and the tile is over-drawn into negative gas; sequentially the result
   depends on cell and prop order and CPU and device disagree unless the order
   is fixed. The deposits through the device edit door must arrive in the
   reference's order too. Fix: the burn is a stage of the per-env serial
   applier, like HEAT_SOURCES (`cuda_unit_hazards.cuh`): props in row order,
   cells FR, FL, BR, BL, tiles row-major, each draw capped by the O2 still
   left; the CPU runs the identical loop. Or an integer proportional split per
   tile. Warming and ignition are per-prop pure reads and can stay parallel.
2. **On the device, the contact and shove pushes write other bodies' rows,
   and the device `add` is not atomic.** FIX. `cuda_unit_lib.cuh:798`
   `impulse_add` is a plain read-modify-write built for one writer. Two props
   hitting one unit race. device-units.md:42: "a slot that does what is not a
   row write [goes] through the Request list". Fix: the 3p contact rule emits
   PUSH requests the serial applier applies in prop row order; the 1b shove on
   a prop the same. Wind (each prop writes its own row) stays per prop.
3. **The device stamp kernel as described lets prop rows erase the units'
   permeability write.** FIX. `cuda_stamp.cu` K2 does `atomicMax(&winner[idx],
   r)` over every row and K3 lets only the winner write: prop rows appended
   after unit rows always win. `prop_owner` and `prop_mob_cap` have no reset
   in K1 (stale owners survive a prop leaving). Fix: K2 picks a winner among
   unit-mode rows only; K1 zeroes both planes; P3's oracle includes a unit and
   a prop on one tile and two props' partly covered tiles overlapping.
4. **Nothing refuses overlapping blocking sets at load, so the device's "plain
   write" of owner and cap is a race.** FIX. 4.5 only shows the MOVE keeps the
   sets disjoint. 3.3's "MIN over nonzero caps" contradicts "plain write".
   Fix: the load door refuses blocking sets that overlap, sit on a
   non-enterable tile or leave the grid; or owner and cap are atomicMin writes
   on both backends.
5. **The contact rule reads another prop's state in the middle of the pass,
   and its partner is undefined.** FIX. "Resting or slower", the lever and "J
   handed to that prop" read the other prop while it moves in the same pass:
   row-order dependent on the CPU, a race on the device. When one candidate
   hits several props, a unit and a wall, "the obstacle body" is singular and
   unspecified. Fix: double-buffer pose and velocity (read start-of-slot,
   write scratch, commit after the refusal pass); the partner a pure function
   of the hit set (all hit bodies with an integer split of J, or a stated id
   rule); how momentum is booked when the other prop is not slower.
6. **The sleep cache is invalidated by "wake", not by pose.** FIX. A pose can
   change with all velocity and impulse zero on every path that LANDS state:
   `props_to_host` into a twin, the gate's download, a chained run's host
   replay, #228's carrying. Each would re-emit stale rows. Fix: key the cache
   on (x_q, y_q, th_q) compared every tick, or recompute every tick on the
   device (section 8 prices it under 0.1 ms). Property: write a new pose with
   zero velocity; the rows equal a fresh computation.
7. **The two undigested planes are read across ticks.** FIX. Built at slot 6,
   read at the next tick's slot 3 and 3p. A pure function of end-of-tick
   poses, so equal digests agree only if every reset or state-landing path
   rebuilds them; the swarm precedent is same-tick (swarm.md:94). Fix: rebuild
   both at reset and on every state landing; put them in
   `_RESIDENT_UNIT_PLANES`; declared field outputs at tol 0 in the lockstep.
8. **The patch plan is missing device twins and a refusal.** FIX. P6 omits:
   the device slot-3 movement reading `prop_mob_cap`; the 1b shove target
   choosing props; the DOORS stage's `prop_owner` read (in the Class-T
   applier, not the played tick); the device `angular_q` taking its lever
   clamp from the body. P2 to P5 never make `DeviceConductor` refuse a level
   with props. Fix: the refusal by name in P2; the four twins in P6's list and
   lockstep scenes.
9. **int64 ranges are not proved; the host plan would wrap where the reference
   does not.** FIX. The wind load W is below 2^80 and the device takes it in
   128 bits (`cuda_unit_impulse.cu:28-30`); the host "one int64 pass over the
   wind-load terms" overflows. The prop wind impulse `((W>>16) * wind_kw_q *
   w_proj) >> 32` is a three-factor product beyond 128 bits. The host store is
   numpy int64 where the reference laws use Python ints. No mass or inertia
   bound: the contact numerator `m_p * m_u * v` overflows at about 10 t.
   `w_proj` divides by |wind|, which can be 0. Fix: ranges on mass, inertia,
   `wind_kw_q` and the sides at the class door; the wind in two 128-bit steps
   mirroring `add_wind`; the host reference in Python ints or overflow-checked
   where vectorised; device faults by name; an early return on zero wind.
10. NOTE. P1's oracle cannot see the boundary conventions: over 300 random
    poses a closed inside test versus an open one differs 0 times; on one
    aligned fixture closed gives k = 35, open 15. Add aligned fixtures and the
    exact headings 0, +-102944, 205887 to property 8.
11. NOTE. Property 2's stated failure condition is wrong: c^2 + s^2 deviates
    by +-2.1e-5 (up to about 11 counts at an 8-tile prop, more than the "+2"
    pad); what keeps the bounding box complete is the 8192-count half-sample
    offset. Restate it (completeness with pad 0).
12. NOTE. Property 17 (mirror physics) will fail for reasons that are not
    bugs: mirroring x maps the heading to pi - th, not exact in the kit (39 152
    of 82 355 headings differ); mirroring y maps to -th, exact. The wind
    read's row-major tie rule is not mirror-symmetric. Specify the y-mirror
    and scenes without wind ties.
13. NOTE. Smaller gaps: the substep formula's `R_max * |w|` is Q32 and needs
    `>> 16`, and how v, w and their remainders split across substeps is not
    stated; record row names must be tokens (`t_q_fr`, not `t_q[4]`,
    serialize.py `_NAME_RE`); `sim.entities` must hold the store's row views
    so `runtime_digest_rows` reads live columns (property: moving a prop
    changes the entity bytes); a sub-count O2 accumulator would be synced
    state and join the record tail; the O2 ignition test over a tile list
    needs an integer definition (combat.py:968+ defines it over one tile's
    open 4-neighbours).

### Checked and found true

- CPU and device trig are the same code (`unit_fixed.cos_q16` / `sin_q16`
  call the C++ kit). Checked over every heading: 0 gives (65536, 0), +-102944
  gives (0, +-65536), 205887 gives (-65536, 0); |c|, |s| never exceed 65536.
- The 2^45 bound is right; `stamp_check.py` re-run reproduces its result byte
  for byte, and with the kit's c, s the bounding box is still complete.
- One LSB of heading changed coverage in 0 of 60 poses of an 8 x 4 prop.
- The float planes are safe: p_k = n / 2^16 is exact in float32, the solvers
  take it through an exact quantize, MIN and MAX commute, atomicMin on the
  bits of a non-negative float is exact; `obstacles` stays static.
- No golden moves: the digest's field list is explicit; the entity section's
  preamble is a constant.
- The refusal pass is order-free and one pass suffices, given the start-plane
  owner test (chains, cycles and ties resolve symmetrically once finding 5 is
  fixed). A unit and a prop wanting one tile: the unit wins (slot 3 before
  3p).
- Sleep is an exact identity; no order flips a prop between asleep and awake.
- Nothing needs randomness.
- `cuda_topology.cuh` is Class T: P5 needs no `.cuda_green`. The tile
  combustion (slot 7) and the swap (9e) run in a fixed order.
- No new host sync per tick.

UNVERIFIED by the critic: that the played resident tick re-uploads host gas
written at 9e (pumps are the precedent; not traced).

## Critic: physics and the books (Opus, 2026-11-21)

**VERDICT: UNSOUND.** The push law can be fixed in place. The fire law
cannot, as designed: it neither sustains itself nor closes the books.
Scratch arithmetic: `C:/tmp/props226/critic_phys/check.py`.

### Findings, most severe first

1. **A burning prop gives off the wrong heat, and no existing constant both
   keeps it lit and keeps the gas off its rail.** BLOCKER. The design pays
   `dQ = H_fuel x dO2` (6.3 step 4). `H_fuel = 4.0` (config.toml:1018) is the
   tile law's gas-side PLACEHOLDER: 125 kJ per unit of N_O2, 2.6% of Huggett's
   4.83 MJ and about 9% of what a burning tile releases per unit of O2. Most
   of a tile's heat is `H_bed` (1.21 MJ per unit, config.toml:1138-1152),
   which the tile law lands in the burning SOLID's own tile
   (combustion.cpp:725-735): the fuel bed that holds the flame plateau. For a
   prop the fuel bed is the prop's own cell, which the design keeps outside
   the books. Worked: each burning cell deposits about 0.018/24 units per tile
   per tick; with `H_fuel` that is 2.2 kW per tile, which the felt read turns
   into +11 C; the fire goes out at about 100 C: a prop fire goes out unless a
   tile fire holds it up. With `H_bed` the excess is +104 C (marginal) and
   1.2 MJ per unit would go into GAS cells; config.toml:983-996 records that
   even 2x `H_fuel` tripped `test_e1_hot_rail`. UNVERIFIED: that the felt read
   sees the prop's own deposit next tick; the exact rail outcome. Smallest
   fix: adopt the alternative 6.4 rejected. Book the cell as an energy
   reservoir: `H_bed` into the cell, booked like `e_comb_solid_heat_sum`; the
   cell exchanges heat with its tiles through a booked seam term; the `H_fuel`
   parcel goes to the gas as in the tile law. A steady-state plateau
   calculation in the design before the build.
2. **The O2 draw is not the tile law's.** FIX. The design draws `burn_rate`
   per blocking tile "at full intensity", leaving out I, `o2f`, `hotf`, the
   hop weights and `w_path` (config.toml:940-967, 1062-1083). At the tile
   law's measured operating point a tile draws 2.3e-3 units/s; the prop's
   0.018 units/s per tile is 7.7x that. Without `o2f` it burns at full rate
   until the extinction gate: a cliff, not a throttle. It draws only from the
   tiles under it and resupply crosses its own 0.5 permeability: a tile's 0.21
   units of O2 empty in about 12 s, the single-tile starvation (T2) that
   `draw_r = 2` exists to cure. Fix: the tile law's demand form (`hotf` from
   the cell's temperature, `o2f` from the tile's O2 fraction, a stated
   per-class I as a calibration for Erik), drawn over the blocking set plus
   the ring with the existing hop weights.
3. **The contact hand-off destroys momentum.** FIX. 4.6 zeroes the prop's
   blocked component and books `m_p v - J` as `contact_lost`. A perfectly
   inelastic collision loses KINETIC ENERGY, not momentum: a 100 kg fridge at
   3 m/s into a 100 kg marine hands over 150 N s and the design deletes the
   other 150. Fix: the prop's axis velocity becomes
   `v' = m_p v / (m_p + m_u)`; it keeps its share, the next tick's contact
   repeats, the push drains into the unit's feet (`resisted`). The same prop
   to prop. Lost KE needs no energy term (body KE is in no group of the
   closure identity, energy-books.md:43-45).
4. **The wind constant depends on tile size.** FIX. As written
   (`rho Cd H 1000 / (2 tile tps^2)` per tile of projected width) it is right
   only at 1.0 m: 3x too large at 0.333 m if H is metres, 9x if H is the
   stored `height_q` in tiles. Fix: build the impulse constant from an area in
   m^2 through `_wind_k` (unit_impulse.py:242-256), area = Cd x H_m x w_m.
   `Cd` is a new dial with no source: declare it, or a drag-area column per
   class as units have `drag_area_m2`.
5. **The wind read is wrong for a body that spans many tiles.** FIX.
   `footprint_wind` returns the MAX over the tiles (gamemap.py:2444-2478) and
   the design applies that one tile's q to the whole projected width: over 12
   to 30 tiles the max picks noise peaks (the #217 checkerboard). With graded
   permeability, barely covered edge tiles sit near the free stream while
   fully covered tiles are shielded, so the force depends on where the prop
   sits within a tile (an axis-aligned prop on tile lines reads only shielded
   air). The centre of pressure is fixed at G: a blast on one end never spins
   a balanced table. Fix: read the air on the windward ring just outside the
   blocking set, a coverage-weighted mean q, torque as the sum of r_i x F_i.
   A model choice for Erik; magnitude UNVERIFIED.
6. **The parallel-axis sign is wrong in `inertia_g`.** FIX. Rotation is about
   the centre of mass: `I_cm = I_G - m |o|^2`, not plus. A 20 kg table 1.6 x
   0.8 m with the mass 0.4 m off centre: the design gives 8.53 kg m^2, correct
   is 2.13; at 0.6 m the correct formula goes negative. Fix: minus, and refuse
   an offset where it reaches zero, or a radius-of-gyration column per class.
7. **The thermal node mixes currencies and borrows a human's calibration.**
   FIX. `t_q` is declared in game degrees but `body_temp.step_q` works in
   degrees C (game is about C minus 20): comparing to `ignition_temp` without
   converting is 20 degrees off. The 159 W/K loss is a human shell's and
   scales with tile volume (body_temp.py:85-90; 1434 W/K at 1.0 m): a cell's
   time constant is 51 s at 0.333 m and 5.6 s at 1.0 m. The felt gain uses the
   marine's absorption and reflectivity. A 20 kg table is 32.4 kJ/K, more than
   the air of a 100-tile room (23.9 kJ/K): warming it 280 K creates about
   9.1 MJ the room never loses. The world's books still close (the node only
   reads), so the closure rule is not broken in letter, but the gap is far
   larger than stated: Erik should rule on it with these numbers.
8. **The shove ignores the target's mass.** FIX. The 100 N s unit shove
   (config.toml:5371) launches a 10 kg chair at 10 m/s (sliding 12.7 m at mu
   0.4), a 20 kg table at 5 m/s. Use `J = min(J_shove, m x v_hand)`, `v_hand`
   a dial for Erik.
9. **R1 is in tiles, so it depends on tile size.** FIX. At 1.0 m tiles the
   floor is 1.5 m and a 0.8 m table is refused. State it, or set the floor in
   metres and accept over-blocking on coarse tiles.
10. **Ruling missing: tall props tip before they slide.** FIX. A box tips
    first when (d/2)/(h/2) < mu: fridge 0.39 against mu 0.4, a bookcase 0.17.
    The top-down model always slides. Section 14, default "slides".
11. **Ruling missing: "no prop seals" is a physics and gameplay decision.**
    FIX. It decides a barricade can never stop a decompression, and forces a
    steel fridge to name a porous row. 3.5's "3.24% of the tile" is the
    thin-board furniture row; a solid fridge fills about 72% of the air column
    over its tiles (0.88 m^3 of 1.22 m^3). The mass books close trivially. Put
    the sealing question in section 14 with the real displacement figure.
12. **Ruling missing: the products differ from the tile law.** FIX. The tile
    law takes the soot share (`soot_yield` 0.5) out of the bulk with its
    energy, so total N is conserved (combustion.cpp:999-1046); the prop keeps
    all of dO2 as N2 and also mints the soot: gas plus trace grows by 0.5 dO2,
    about 3.5% pressure in a room that smothers. Listed as an ACCEPTED GAP but
    not as a ruling.
13. NOTE. Density from the box volume makes everything float (table 21,
    fridge 113, steel cabinet 82 kg/m^3). #236 must choose between material
    density and enclosed density; the design chose silently.
14. NOTE. Smaller gaps: the `OMEGA_MAX` clamp on w is not stated as booked;
    separate linear and angular friction overestimates dissipation by up to
    about 2x (in section 12, missing from R4's text); the refusal pass deletes
    the pair's net momentum (the common velocity would keep it).
15. NOTE (inherited). Fuel per unit of O2 depends on tile size:
    `KG_FUEL_PER_N_O2` is fixed at the 0.333 m tile (materials.py:349-354)
    while a unit of N_O2 scales with tile volume: 9x off at 1.0 m.
16. NOTE. Dials borrowed from other objects or tuned under the #217 bug, for
    Erik: the frag row, `expansion_m3_per_strength`, `k_drag2` (tuned while
    transport ran at 1/3); the wind sampled once per tick, not integrated over
    a shock that passes in ms (UNVERIFIED size); `burn_rate` 0.018 ("re-tune
    pending"); `H_fuel` 4.0 (placeholder); `H_bed` (a calibrated lumped
    constant); `soot_yield` 0.5 (provisional); `heat_flux_to_temp`,
    `unit_absorption`, `unit_reflectivity` (human skin); the 159 W/K body
    loss; the 100 N s shove; `V_MAX_Q` (4 tiles a tick: 32 m/s at 0.333 m, 96
    at 1.0 m); the new `Cd`.
17. On the rulings list. Missing: the heat constant and where the bed heat
    lands (1), the burn-rate law (2), how the wind is read and the centre of
    pressure (5), the shove limit (8), tipping (10), sealing (11), the product
    split (12), the meaning of density (13), the size of the room-cooling gap
    (7). R5 asks whether a prop pushes a unit, not whether momentum is kept
    (3). R9 is a scope and training rule, not physics.

### Checked and found true

- Coulomb friction through `_shrink` (unit_impulse.py:578-586, 664-676): a
  force below mu m g never moves the prop, a sliding prop stops exactly at
  zero, the same m/s at both tile sizes.
- Step 4 of 4.2 is safe: each remainder is below one count of velocity times
  m, so booking it as resisted when friction holds is exact.
- The projected-width formula, derived independently.
- The angular units: impulse and `inertia_g` both scale as 1/tile^2, so dw is
  the same at both tile sizes.
- Slide thresholds are plausible with a correct constant (Cd 1.1, mu 0.4):
  table 59 Pa, chair 79 Pa, fridge 283 Pa, against the marine floored at 245.
- `swap_o2_to_inert` keeps the energy exactly (gas energy is N x T_abs with
  one heat capacity per bulk count).
- Wall stops book `wall` as the unit move does; lost KE needs no term.
- Graded permeability falls linearly with the covered fraction: the right
  open-area scaling.
- Ignition by temperature only agrees with "start a fire by delivering heat"
  in letter (the heat comes from an unbooked node, finding 7).
- The burn heat goes through the same `heat` FieldEdit and the same solver
  groups as a heat source, by `payloads.deposit_over_tiles`: a second caller
  of one seam, not a second deposit path.
- The O2 swap has a home: a new seam primitive with a device twin.
