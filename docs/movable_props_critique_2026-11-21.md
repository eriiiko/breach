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

## Critic: determinism and the device path

Pending.

## Critic: physics and the books

Pending.
