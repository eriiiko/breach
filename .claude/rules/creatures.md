---
paths:
  - "src/simulation/creature*.py"
  - "src/simulation/arm_chain.py"
  - "src/simulation/fixedpoint_twins.py"
  - "src/control_creature.py"
  - "cpp/src/arm_chain_solver.*"
  - "cpp/src/activation_table.inc"
  - "docs/octopus_integer_reference/*.py"
  - "renderer/creature_*.py"
  - "tools/bench_octopus.py"
  - "tools/gen_octopus_tank.py"
  - "tools/gen_activation_table.py"
  - "tools/octopus_clips.py"
  - "tests/*creature*.py"
  - "tests/test_hit_seam.py"
  - "tests/test_chain_accessor.py"
  - "tests/test_octopus_*.py"
  - "tests/test_activation_table.py"
  - "tests/test_fixedpoint_twins.py"
---

# Creatures: the row, the chain, the body, the hub, the hit seam, the picture (#104-#113)

Fires on contact with these files. CLAUDE.md's index says what each system is
and who must use it; this file holds the constraints the code cannot say. The
octopus is an ASSET (a versioned `[creatures.*]` row on shared systems), never
a class; a creature is a `Unit(species_id=...)` whose species names a creature
row (Erik, 2026-10-29 / 10-30). Lead: `.claude/skills/octopus-lead/SKILL.md`.

## Creature table
- A creature's body plan, tissue, solver resolution, primitive dials and rules are ONE versioned `[creatures.*]` row, read through ONE door (`creatures.build_creature`) that refuses a missing, unknown, mistyped or out-of-range key and a size over the chain's headroom (each failing bound named), quantized once; its `version` + `row_hash` are hashed in every creature record.
- THE creature key is `species.creature is not None` (`creature_system.is_creature`): never a creature name in a branch, never a per-creature constant in code, never a second table. `SpeciesDef.armed` / `.can_become_zombie` are species DATA: an unarmed species binds no loadout; a non-convertible one never rises (a branch of `ai_zombie.rises_as_zombie`, the one conversion predicate).
- Creatures live on `factions.CREATURE_TEAM`, which `MatchRules` does not count (Erik, 2026-10-30: hostile to all, never the zombie team).
- The swarm's species table (`swarm.SPECIES_FIELDS`) is a DIFFERENT thing: swarm units are not `Unit`s. A creature is never a swarm species and a swarm species never a creature row.

## Chain currency
- Chain positions and velocities are int32 Q(16+8) metres and m/s (+/-128 m); masses Q16 kg, inverse masses Q32 (`2^48 // m_q16`, never from a float), stiffness Q16. Every read or write converts ONCE through `creature_fixed` (or the kit's twin), never an inline shift; every conversion and every signed correction rounds half away from zero (the no-slide guarantee).
- The pose <-> unit converts only through `pose_to_unit_anchor` / `unit_anchor_to_pose` and `heading_q24_to_facing_q16` (the Q24 heading rounded once to the unit's Q16 radians).
- `fixedpoint_twins.py` is the bit-exact Python twin of `fixed_point.h`'s chain kit (`shr_round_away`, `div_round_away`, `mul_shr_away`, `CHAIN_FINE_BITS`), held equal by `tests/test_fixedpoint_twins.py`. Never a private rounding helper.
- The headroom bounds are the ones the arithmetic forms (`dist_impulse_fit` caps the arm at ~4.2 m on the shipped row; `bend_dlam_fit`); the bend skips a plane below the door's `l_min` (`bend_plane_min_q32`).

## Activation table
- Yekutieli's sigmoid S(x) = (1 + tanh x)/2 is a CHECKED-IN Q16 table (`tools/gen_activation_table.py --write`), never a tanh at load or run time; the engine compiles the .inc, the integer reference parses the same file; the lookup law is `chain_ref_q.activation_s_q16`, held equal to the compiled one.

## Chain integer reference
- `docs/octopus_integer_reference/chain_ref_q.py` is THE executable spec of the arm chain's integer arithmetic: a change lands THERE FIRST, its gates re-run, then the C++. Not engine code: nothing in `src/` or `cpp/` imports it.
- Its velocity rebuild carries the predict's rounding remainder and the corrections' FINE (Q32) remainder: never erase a sub-count velocity. Two-body corrections move each participant by its own inverse mass times ONE shared impulse: never a per-node rounded factor (it slid the free animal).

## Arm chain solver
- THE articulated-body solver for every creature with arms: STATELESS over caller-owned buffers (`arm_chain.allocate`, born `(N, A, n, 3)`); walls from `GameMap.solid` (material + door state, never `obstacles`), other units' footprints only for the crown's disc. Pure integer, `/fp:strict`, 0/0/0 on the float ratchet; held at tol 0 to the reference after every tick (`tests/test_octopus_arm_chain_solver.py`).
- NO arm-unit contact (Erik, 2026-10-30: arms pass through marines until that design; ACCEPTED GAP). Never a second chain, never a float in its state.
- HOST-SIDE ON EVERY PATH: no CUDA twin (recorded debt, a `(N, A, n, 3)` transcription). The played `--cuda` / `--resident` paths run slot 5b on the host; the device conductor (Class T) REFUSES a creature by name (`conductor.ConductorRefused`); the device unit store carries no creature body.

## Creature bodies
- One `CreatureBody` per creature unit owns its solver buffers and the hub's integer state; every timer is an ELAPSED-tick counter, never compared to `sim.tick`. Slot 5b (`CreatureSystem.tick`, one call line) makes ONE solver call per living creature in unit-id order and re-reads the walls when `GameMap._tile_serial` moves.
- Slot 5b is the ONLY per-tick writer of a creature unit's anchor and facing, FROM the integer pose: the anchor through `Unit.set_pos` (the unit's float door, quantized once), the facing through `turning.place` (target and facing at rest). Every other writer SKIPS a creature: slot 1a (`unit_impulse.integrate_all`: no impulse integration, no turn), slot 3, `OnePhaseWEGO.drive_units`, the round-end snap; the AIM is the hub's reach goal, never a facing target.
- The wind reaches a creature as a BODY impulse inside the ONE wind row (`exchange.apply_wind_force(..., creatures=)` -> `_creature_wind`: THE read and THE price, divided by the unit's mass, added to the crown velocity): never the unit impulse law, never a second reader, no knockdown. VETOABLE (Erik): `creatures=None` at 9c2 leaves a creature unpushed.
- Arms never block, stamp, jam a door, sense heat or gas, or count for melee: only the body footprint does.
- ONE serializer (`serialize_record` / `decode_record`) for the `__creature__` digest section and the recorder's `creature_snapshots` ring; `CREATURE_SECT_V<n>` bumps only on an encoding change; the section's carrier `n_creatures` is ALWAYS written and folded only when > 0, so a creature-free sim hashes as before (no golden moves; `DIGEST_SPEC_VERSION` unmoved).
- Spawned only by `Simulation.spawn_creatures` (`C-<slot>`), refused without `breach_physics.arm_chain_tick` (no Python fallback). `[[creature]]` rows are written only by `level_lib.write_creatures` / `write_match`; a match never reads the level's own rows.
- OPEN (Erik, the merge of 2026-11-20): the body draws from its OWN xorshift64 seeded from the sim seed and unit id, which predates main's Match draws rule (every random number a keyed Philox block, a `Salt` member). Not redesigned at the merge.

- ONE module, `creature_primitives.py`: idle (the default), crawl, swim (umbrella, jet train), reach, grab (a creature-owned PROP; a grabbed UNIT is #107's). A primitive writes ONLY solver inputs and reads the solver's report, never re-deriving a contact; a sucker grips only a node the report put on the FLOOR. Every dial is the row's (`PrimDials`, quantized once); all state integer and hashed. A new primitive is a state machine there over row dials.
- THE SPEED LAW (Erik, 2026-10-31: "control the speed from 0 to its max speed, as if we had an Xbox controller"): the creature reads main's MOVE_DIR THROTTLE (`throttle_q`) as its analog speed through `creature_speed_q16` (the throttle itself; main's WALK, throttle 0, = `crawl_until`, an OPEN question for Erik at the merge) and picks the gait by band (`speed_band`); each band's top is the P6d2 gait exactly; a band change waits `gait_hold_ticks`. Never a second speed field on MOVE_DIR, never a speed-mode mapping.
- AIM -> reach; USE -> grab (only with a prop; the conductor's USE slot skips a creature, the hub consumes it). `--play creature` puts `CREATURE_TEAM` under main's direct scheme (`control_source.select_play`, `DirectControl.drives_one_unit`): never a second keyboard scheme.

## Hit seam
- Bullet and beam hits test `Unit.hit_tiles()` / `in_hit_tiles` (the bullet march, the laser skewer: ONLY those two); stamps, doors, exchange rows, melee, the bite, vision and click-picking test `occupied_tiles()`. A human's two sets are equal. Never a third occupancy set.
- A creature's set is the footprint plus every in-map tile an arm SEGMENT crosses (top-down, z ignored, the prop excluded), an exact-integer supercover walk in the solver's tile law (`creature_fixed.segment_tiles`), rebuilt with every pose write: DERIVED (`Unit.hit_shape`, a `REFERENCE_ONLY` schema entry), never digested. The struck (arm, node) goes to the body's hashed `last_hit`, never onto `UnitHitEvent`.

## Chain accessor
- THE read of a creature's chain for the renderer, tools and rules (`read_chain` / `Simulation.creature_chains`): float32 copies dequantized ONCE, in the FACING convention, dead creatures included. Never the buffers, never written back: a reader cannot move a digest (`tests/test_chain_accessor.py`).

## Creature render
- A creature is drawn ONLY from its chain, its unit taken out of the marine / zombie draws; its look is a `[render.creature_looks]` row (species -> looks), never a species name in renderer code. Its bones are posed by `creature_pose.pose_from_chain` on the art scripts' ONE copy of `octopus_math` (imported by file) behind the SIZE DOOR (`match_size`) and the BIND DOOR (`creature_model.check_bind_pose`); drawn in the units' 3D pass with the ONE lit shader, the unit's one light read, its projected and smoke shadows.
- OVERLAP (finding at the merge of 2026-11-20): main's pose layer (`renderer/pose_layer.py`) is THE place a 3D unit's pose is composed, clip-driven and skeleton-agnostic; a creature's pose is FROM ITS SIM CHAIN and does not route through it. Recorded, not resolved: Erik decides whether the chain pose becomes a pose-layer source.
- Render wiggle is `[render.creature]` (read every frame, refused missing or mistyped, 0 = off). No model or animation state on `Unit` or the sim.

## Benches
- A creature's cost is `tools/bench_octopus.py` (by part of slot 5b, through `creature_system.set_tick_laps`, OFF by default, never on a sim, body or digest): REPORTED, never gated. On CUDA take an octopus run and its baseline in separate processes.
