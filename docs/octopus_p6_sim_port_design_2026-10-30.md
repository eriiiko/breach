# Octopus P6 — the port into the shared sim (design)

**Status**: draft v1, 2026-10-30, written in the P6 design session with Erik.
Author: Claude (octopus + creatures lead). Study brief:
`docs/octopus_animation_study_2026-10-29.md` §5 (P6). Prototypes P1–P5:
`prototypes/octopus_motion/` (their README holds every measured number this
doc reuses). Issue #105 (study), #104 (the asset ruling), #106 (the sim side).

## 1. Rulings taken in the session (Erik, 2026-10-30)

1. **Cost is not a design input.** Build the best octopus first; measure;
   optimise only after the final result has been seen. The brief's 0.2 ms /
   octopus / tick ceiling is suspended for P6 — the port MEASURES and reports,
   it never trades fidelity for the number. ("Rather one fantastic octopus
   than ten meh ones.")
2. **z is in the chain.** Nodes are 3-D; the floor is a contact; lifting a
   marine and swimming are real.
3. **The integer trig is the kit's**: `atan2_q16`, `sin_q16`, `cos_q16`,
   `sqrt_q16` in `cpp/src/fixed_point.h`. A look-up table with a few percent
   of error is blessed if `atan2` ever becomes the hot spot; not expected.
4. **The octopus is a Unit** — the one class, no subclass. It gets HP, a
   team, hits, kills, events, match counting and the digest's unit hash for
   free; the arms are the only new thing.
5. **Claude is creatures lead** as well as octopus lead: how creature data is
   organised is Claude's call. Decision: a `[creatures.octopus]` row in
   `config.toml` (where materials, gases, filters, weapons and loadouts all
   live; splitting the file is not this arc's question), read by one door.

Open for Erik later, not now: gameplay speeds (brief §6 q1) — tuned at P8 on
the clips; every speed below is the prototype's biological value scaled to
the mid animal and is a row value, not code.

## 2. Shape of the thing: four layers, two owners

```
  [creatures.octopus]  ──door──►  CreatureDef  (creatures.py)        DATA
          │
  Unit(species_id="octopus")      HP, team, events, digest unit hash  SIM (exists)
          │ id
  CreatureBody (creature_system.py, one per octopus unit)             SIM, synced
     ├── hub + primitives   (Python: lottery, states, timers)         "brain stem"
     └── ArmChainSolver     (C++ Q16, PhysicsEngine-owned)            "muscle"
          │ node positions (Q16, 8 × 18 × 3)
  render: chain → skin  (P7; reads nodes through one accessor)         RENDER, float
```

- **Data**: the row (§3). Nothing in code says "if octopus".
- **Sim, Python**: `creature_system.py` — the hub and the primitive state
  machines, ported from `crawl_sim` / `swim_sim` / `grab_sim` / `idle_sim`.
  Decisions a few times a second; integer-only state; seeded through the
  sim's one RNG hand-off (never its own generator).
- **Sim, C++**: `ArmChainSolver` — the P1/P2 chain (`arm_chain.py` /
  `arm_bank.py`) in Q16.16, owned by `PhysicsEngine`, called once per tick
  per octopus with that tick's activation / rest-scale / anchor / grab
  inputs. The heavy part; `/fp:strict`; no float.
- **Render**: P7. Reads nodes through one accessor, never the solver.

### Why the hub stays in Python (vetoable)
The hub runs at decision rate (0.1 s), touches the unit table, the payload
table and the RNG hand-off, and is where the brain (#108) will plug in. The
chain runs 16 × 2 passes × 8 arms × 18 nodes per tick and touches nothing but
its own arrays. Same split as vents (`vents.py` schema + `vent_system.py`
runtime over the C++ primitives). If a profile later shows the hub mattering,
it moves; its state is already integer.

## 3. The creature row and its door

`config.toml` `[creatures.octopus]`, read by `src/simulation/creatures.py`
into a frozen `CreatureDef`. Groups, each a sub-table, each value a
prototype-measured number (the README's):

| group | rows (first version) |
|---|---|
| `body` | `arms = 8`, `nodes_per_arm = 18`, `arm_length_m`, `radius_base_m`, `radius_tip_m`, `tissue_density`, `body_mass_kg`, `body_radius_m`, `footprint` (tiles), `arm_angles_deg` (8), `front_pairs` / `back_pairs` |
| `tissue` | `k_bend`, `k_passive_frac`, `k_dist`, `bend_damp_s`, `c_drag_perp`, `c_drag_tan`, `g_eff` (buoyancy-reduced gravity), `mu_floor` |
| `solver` | `substeps = 16`, `passes = 2` |
| `reach` | `a_ca`, `t_act_s`, `front_width`, `stretch`, `aim_max_deg` |
| `crawl` | `anchor_nodes`, `elongation_max`, `push_s` + jitter, `recover_s` + jitter, `decision_s`, `p_recruit`, `max_stance`, `s_min`, `back_bias`, `neighbour_bonus`, `grip_limit_n`, `stance_activation`, `steer_gain`, `steer_window_s` |
| `swim` | `umbrella_open_deg`, `umbrella_closed_deg`, `bundled_deg`, `open_s`, `close_s`, `activation`, `web_factor`, `jet_dv`, `jet_s`, `body_c_drag` |
| `grab` | `grab_nodes`, `sucker_reach_m`, `wrap_activation`, `hold_activation`, `pull_in_scale`, `pull_in_s`, `grip_limit_n`, `hold_s` |
| `idle` | `max_active`, `p_start`, `hold_s` + jitter, `wave_s` + jitter, `stretch` + jitter, `activation`, `touch_s` + jitter, `limp_s` + jitter, `aim_deg`, `front_bias` |

Door rules (the four number-doors apply): every length in metres, every time
in seconds, quantized ONCE at load (`ticks_from_seconds`; Q16 through a new
`creature_fixed.py` beside the other `*_fixed.py` modules); a missing or
mistyped key is refused, never defaulted; a value the physics cannot take
(`nodes_per_arm < 3`, `substeps < 1`, a negative stiffness, `0 ≤ fractions ≤
1`) is refused at the door. The species table gets an `"octopus"`
`SpeciesDef` row whose new optional field `creature: str | None` names the
creature row — the one join between the two tables; a human has `None`.

**Versioning** (the asset ruling, #104): the row carries `version = 1`.
A later octopus is the row changed and the version bumped in the same
commit, tagged `octopus-v<N>`; the digest spec bumps with it (§5).

## 4. The octopus as a Unit

- `Unit(name, x, y, team, footprint=<row>, species_id="octopus")`. Position
  is the body (crown) centre; `footprint` the mantle's tiles (mid size
  0.93 m → 3 tiles at 0.333 m). The body blocks like any unit
  (`stamp_units`); **arms never block and never stamp** — their contacts are
  §6's node-vs-tile and node-vs-unit tests.
- HP, death, conversion, `DamagePacket`, events, `attacker_id`, the match's
  team counting: unchanged code. A dead octopus keeps its body (the dead-arm
  signature of P1: no activation, drag only) until the corpse rule, #118.
- Movement: an octopus is never moved by `_update_player_movement`'s tile
  stepping. Its position is WRITTEN by the creature body each tick from the
  solver's rigid body (the P2 `Body2D`, now with z): the arms move the
  animal, nothing else does. The unit's `facing` is the body heading.
- **Hitbox: the arms can be shot** (Erik, 2026-10-30: "it would be
  disappointing if we couldn't"). Today a bullet's march stops at the first
  tile inside a unit's footprint rectangle (`combat.py`, the march and the
  laser's skewer) and melee walks `occupied_tiles()`. The octopus's
  `occupied_tiles()` becomes its body tiles PLUS the tiles its arm nodes
  lie in (8 x 18 nodes -> a tile set, rebuilt from the synced chain once per
  tick after slot 5b, so it is deterministic and costs nothing), and the two
  march sites test occupancy through that one seam instead of the rectangle
  (a human's set is still its rectangle: no behaviour change, golden
  untouched). HIT tiles and BLOCKING tiles are different sets: an arm can be
  shot but walked over. The hit event records WHICH node was struck (the
  packet's existing position); v1 damage goes to the unit's one HP pool;
  per-arm damage (a shot-off or dead arm feeding the chain's `alive`, the
  P1 dead-arm signature) is #118's row on the same record, not a new seam.
  Bullet fate stays geometry (the Vision / engagements / cover rule).
- Orders: the hub consumes the same intents a unit gets (MOVE_DIR → travel
  direction, a target → reach/grab); v1 ships the intent-to-goal map for
  travel, grab, swim, idle; a bot or the brain (#108) issues the intents.

## 5. Synced state and the digest

Per octopus, all integer, all hashed:

| state | shape / type | note |
|---|---|---|
| node positions | (8, 18, 3) int32 Q16 m | world frame |
| node velocities | (8, 18, 3) int32 Q16 m/s | |
| body pose | x, y, z, heading, vx, vy, vz, yaw rate — int32 Q16 | Q16 turns for the heading, as the sweep's cones |
| anchors | (8, 18) uint8 mask + anchor positions (8, 18, 3) | sucker seals to the floor / a wall tile |
| grabs | list of (arm, node, unit_id, local offset Q16) | sealed to a unit (the P4 prop becomes a unit, #107) |
| hub | per-arm state enum, timers in TICKS, the push / recover / hold lengths drawn this cycle, stance count, the travel goal, the steering window | the lottery's RNG is the sim's hand-off |

Derived each tick and NOT stored: activation, rest-scale, bend targets,
the root frame (from body pose + row angles). The render reads positions.

**Digest**: a new ENTITY-section record kind, `CREATURE_SECT_V1`, absence
transparent exactly as the vents and charges are (an octopus-free level's
digest is byte-identical to today's). The record hashes every row above in
a fixed order. The field-plane spec version does not move (no grid field
changes). A row-version bump (§3) bumps `CREATURE_SECT`'s version.

**Recorder**: positions are added to `DEFAULT_FIELDS` additively as a
per-tick (8, 18, 3) int32 block when an octopus exists (P7 needs them for
the clips; the contract's int32 branch covers it).

## 6. The chain in Q16 — what changes from the prototype

Units: positions in metres Q16.16 (a 2.5 m arm's segment is 0.147 m ≈ 9.6 k
counts; a constraint correction of 0.1 mm is 6–7 counts; a 1 m/s node moves
170 counts per substep at 24 Hz × 16). Velocities in m/s Q16. Mass in kg Q16
(the mid arm's tip node is tens of grams: 2 k counts — fine; at the real
0.24 m scale it would not be, which is one reason the port is at game scale
from the first commit). Angles in radians Q16 via `atan2_q16`.

- **XPBD compliance** α̃ = 1 / (k · dt²) and the Lagrange multiplier update
  Δλ = (−C − α̃ λ) / (Σ w |∇C|² + α̃): the divide is the kit's `floordiv_q` /
  `reciprocal_q16`, the products `mul_wide` in int64, exactly the sweep's
  staged-chain discipline; the per-constraint λ is int32 Q16 and lives
  only inside the tick (reset each substep, as the prototype does).
- **Bend constraint**: angle between successive segments via `atan2_q16` of
  the 2-D cross / dot in the segment's bending plane; in 3-D the bend target
  is a direction (the hub's `bend_dir`), so the constraint acts on the
  component of the turn in that plane and a passive bend stiffness on the
  rest. Rayleigh damping as the prototype: `γ = bend_damp_s / dt`.
- **Drag**: anisotropic quadratic, decomposed on the local tangent; the
  square and the sign through `mul_wide` + a saturating narrow; `c_tan /
  c_perp` row values.
- **Gravity and floor**: `g_eff` (buoyancy-reduced) on z; the deck is `z = 0`
  with a one-sided contact (project up, kill the normal velocity, Coulomb
  friction `mu_floor` on the tangential one — the P2 anchor friction
  generalised). A **vacuum tile** (space) has no water: `c_drag` and
  buoyancy read the tile's medium — v1 ships air = water for the arm's
  drag (ACCEPTED GAP: the octopus is a water animal in a flooded ship until
  #107 asks otherwise) and simply notes the tile.
- **Walls**: a node inside a blocked tile is projected to the nearest face
  (PBD collision, one pass per substep after the chain passes), with the
  floor friction law. A sucker may ANCHOR on a wall tile's face as well as
  the deck (`anchors` carry a position, not a tile).
- **Units**: a node inside a living unit's footprint is pushed out the same
  way; a GRAB is the P4 rigid attachment to that unit's position (the unit
  is a rigid body of its mass for the attachment; its own movement sees the
  arm as an external displacement — the real coupling is #107).
- **Substeps × passes**: the row says 16 × 2; the P1 finding stands (8 × 1
  fails straightness at any stiffness; 32 × 1 whips the tip).

**Reference first, C++ second** (the sweep's own rule): a Python integer
reference `docs/octopus_chain_ref_q.py` (ints only, the kit's twins from
`gas_fixed` / `sweep_ref_q` style helpers) with its gates — the P1 gates
re-run on the integer chain at game scale (reach reproduced, amplitude
invariance, drag load-bearing, dead arm) plus a bit-identity gate of the
C++ against it. The C++ `ArmChainSolver` (`cpp/src/arm_chain_solver.{h,cpp}`,
on the `/fp:strict` list) is written against the reference and held to it
at tol 0 by `tests/test_arm_chain_reference.py`.

**CUDA twin**: deferred, recorded as debt, not P6 — units and zombies are
host-side today and the octopus joins them; the chain goes to the device
when units do (the resident path, RL-batch §A, where it is born
`(N, 8, 18, 3)`). Vetoable: if Erik wants the twin now, it is one more
patch after P6b.

## 7. The conductor slot

`Simulation.step()` **slot 5b — creature bodies**, after zombie AI (5) and
before the obstacle re-stamp (6): hub decisions → solver inputs → one
`ArmChainSolver.step` per octopus → body pose → `unit.x/y/facing` → anchors
and grabs updated. One call line, `self._creature_system.tick(...)`, the
logic in `creature_system.py`. Dormancy: no octopus → no `CreatureBody`, no
call, no record (the charges' pattern).

## 8. Patches

| patch | content | gate |
|---|---|---|
| P6a | the creature row + door (`creatures.py`, `creature_fixed.py`), the `"octopus"` species row with `creature=` | door tests (refusals, quantization once), `test_ingress_lint` |
| P6b | the integer reference `octopus_chain_ref_q.py` at game scale + its gates (P1's seven on ints; z + floor + wall gates) | the gates pass; `tests/test_octopus_chain_reference.py` guards the file |
| P6c | `ArmChainSolver` C++ + binding; bit-identity to the reference | tol 0 over 10 s of reach / crawl inputs; float ratchet unchanged |
| P6d | `creature_system.py`: hub + primitives ported (crawl, swim, grab, idle, reach) over the solver; `CreatureBody`; slot 5b; `CREATURE_SECT_V1`; recorder block; `occupied_tiles()` with the arm tiles and the two march sites reading occupancy through it | P2–P5 gates re-run through `Simulation` on a flooded test room (property gates, dequantized units); absence-transparency (octopus-free digest unchanged, golden untouched); determinism (two runs, one digest); a shot across an outstretched arm hits the octopus, across its empty crown-side tiles misses |
| P6e | cost measured (`tools/bench_octopus.py`: ms / octopus / tick, CPU, by part: hub, chain, contacts) and REPORTED; no optimisation | a number in the report, no threshold |

Each patch one worktree agent, one writer; design-gated patches: P6c and
P6d. HUMAN-TEST before merge: none here (nothing shows until P7), but P6d's
GIF harness from the prototypes is re-pointed at the sim so Erik can see the
ported animal move before P7 if he wants.

## 9. Systems

**Existing canonical systems this design uses** (CLAUDE.md inventory):
Unit (one class, species row); Config (`CFG`, one door per table); Material /
gas tables pattern (a row, never an if); Fixed-point kits (`fixed_point.h`,
`atan2_q16` & co); Q16 boundary modules (`*_fixed.py`); PhysicsRunner /
PhysicsEngine (the solver's owner and only caller); Tick conductor (one call
line); Entity-section digest (absence-transparent records); Recorder
(additive `DEFAULT_FIELDS`); Ingress lint + float ratchet; Integer-reference
pattern (`sweep_ref_q`: reference first, C++ held to it at tol 0); Control
seams (intents → goals); Test conventions (property gates).

**New systems this design creates, with draft rules**:

| system | where | draft rule |
|---|---|---|
| Creature table | `config.toml` `[creatures.*]` + `src/simulation/creatures.py` (`CreatureDef`) + `creature_fixed.py` | A creature's body plan, tissue, solver resolution and primitive dials are ONE `[creatures.*]` row, versioned (`version`, tag `<name>-v<N>`), quantized once at the door, refused when missing or out of range; a species joins it by `SpeciesDef.creature`. Never a per-creature constant in code, never a second table (the asset ruling, #104) |
| Arm chain solver | `cpp/src/arm_chain_solver.*` (PhysicsEngine-owned) + the integer reference `docs/octopus_chain_ref_q.py` | THE articulated-body solver for every creature with arms/tentacles: XPBD chain, Yekutieli stiffening wave, anisotropic quadratic drag, floor/wall/unit contacts, in Q16 with the kit's arithmetic; a change is made in the reference FIRST and the C++ held to it at tol 0. Never a second chain, never a float rig in sim state |
| Creature bodies | `src/simulation/creature_system.py` (`CreatureBody`, slot 5b) | The ONE runtime that turns a creature unit's intents into primitives (reach, crawl, swim, grab, idle) and drives the solver; it is the only writer of a creature unit's position; its state is integer and hashed in `CREATURE_SECT`; a new primitive is a state machine here over row dials, never a special case in a controller |
| Chain accessor | `src/simulation/creature_system.py::read_chain` (P7 fills it) | The render, tools and rules read a creature's nodes only through it (dequantized float32 views), never the solver or the arrays |
