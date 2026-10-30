# Octopus P6 — the port into the shared sim (design)

**Status**: v2, 2026-10-30 — v1 written in the P6 design session with Erik,
then revised against the adversarial critique
(`docs/octopus_p6_critique_2026-10-30.md`, 23 findings; disposition in §10).
Author: Claude (octopus + creatures lead). Study brief:
`docs/octopus_animation_study_2026-10-29.md` §5 (P6). Prototypes P1–P5:
`prototypes/octopus_motion/` (README = every measured number). Issues #105
(study), #104 (asset ruling), #106 (sim side).

**Base**: this branch is ~95 commits behind `main` (#119–#125 landed there:
any-int teams and `ZOMBIE_TEAM`, loadouts, the match registry, slot 5 as
`_run_zombie_slot`, credited causes). `main` is merged INTO
`33-octopus-concept` before P6a and §4/§7 are re-read against it.

## 1. Rulings taken in the session (Erik, 2026-10-30)

1. **Cost is not a design input.** Build the best octopus first; measure;
   optimise only after the result is seen. The brief's 0.2 ms ceiling is
   suspended: P6 MEASURES and reports, never trades fidelity for the number.
2. **z is in the chain.** 3-D nodes; the floor is a contact; lifting and
   swimming are real.
3. **The integer trig is the kit's** (`atan2_q16`, `sin_q16`, `cos_q16`,
   `sqrt_q16`); a LUT with a few percent of error is blessed if `atan2` ever
   binds.
4. **The octopus is a Unit** — the one class, no subclass.
5. **Claude is creatures lead**; creature data is organised by Claude:
   a `[creatures.octopus]` row in `config.toml`, one door.
6. **The arms are hittable** — "it would be disappointing if we couldn't".

Open for Erik, asked after this revision: **which team / faction the octopus
is on and how a match treats it** (§4.6). Later, at P8: gameplay speeds.

## 2. Shape: four layers, two owners

```
  [creatures.octopus] ──door──► CreatureDef (creatures.py)           DATA
          │
  Unit(species_id="octopus")   HP, team, events, unit hash          SIM (exists)
          │ id
  CreatureBody (creature_system.py, one per creature unit)          SIM, synced
     ├── hub + primitives  (Python, decision rate, int state, own xorshift)
     ├── int arrays        (nodes, velocities, anchors, grabs, body pose; owned HERE)
     └── ArmChainSolver    (C++ Q, STATELESS over those arrays; PhysicsEngine binding)
          │ nodes
  render: chain → skin (P7) through read_chain()                   RENDER, float
```

- **Data**: §3. Nothing in code says "if octopus"; every creature-specific
  branch keys on `species.creature is not None`, which is data.
- **Sim, Python** — `creature_system.py`: the hub and primitive state
  machines ported from `crawl_sim` / `swim_sim` / `grab_sim` / `idle_sim`.
  All state integer; every timer an ELAPSED-TICK counter (never compared to
  `sim.tick`, which TwoPhaseWEGO rewinds); the lottery's randomness a
  **per-creature integer xorshift** seeded from the sim seed and the unit id
  (the study's choice, §3.C) — never `sim.rng`, whose draw count would couple
  an arm's decision to how many shots were fired that tick; its state is
  hashed (§5). The decision comb is `decision_s` quantized once (0.1 s → 2
  ticks, 12 Hz); the P2 rhythm gate is re-run after quantization.
- **Sim, C++** — `ArmChainSolver`: the chain substep, stateless, over
  int32/int64 numpy buffers owned by `CreatureBody` (the pybind buffer
  pattern), born `(N, 8, nodes, 3)` with N = 1 (RL-batch §A). Owning no state
  keeps the sim picklable (AI training serializes the sim) and leaves a
  Python-reference fallback when no engine is bound. **Per-tick contract**,
  fixed: IN — positions, velocities, per-node activation, rest-scale, bend
  targets (two signed per interior node, §6.4), anchors (mask + position),
  grabs (mask + target pose), root frames, body pose, the static topology
  view (`material`, door state), the living units' footprints, the medium
  per tile; OUT — positions, velocities, body pose, **per-node contact
  report** (floor / wall / unit id, this tick), **anchor grip force and pops**,
  **grab force**, the activation clock advanced (per-arm elapsed SUBSTEPS).
  The hub reads the report (P4's stop-on-contact, seal-what-touches; P2's
  release-on-saturation) and never re-derives it.
- **Render**: P7, through one accessor.

### Why the hub stays in Python (vetoable)
It runs at decision rate, touches the unit table and intents, and is where
the brain (#108) plugs in. The chain runs 16 × 2 × 8 × 18 per tick and
touches only its arrays. Same split as vents. The new host-side per-tick
logic is recorded as RL-batch debt (§9).

## 3. The creature row and its door

`config.toml` `[creatures.octopus]`, read by `src/simulation/creatures.py`
into a frozen `CreatureDef`; Q16 conversion through `creature_fixed.py`
(registered with `test_ingress_lint`); seconds through `ticks_from_seconds`
(at least one tick); **quantized ONCE at load**; construction-bound like the
weapon tables (an F5 reload never retunes a live octopus). Groups:

| group | rows (v1) |
|---|---|
| `version` | `1` — hashed INTO every creature record (§5), never the section version |
| `body` | `arms`, `nodes_per_arm`, `arm_length_m`, `radius_base_m`, `radius_tip_m`, `tissue_density`, `body_mass_kg` (THE one mass: it sets the species' mass mean; the stat draw's mass is unused for creatures), `body_radius_m`, `footprint`, `arm_angles_deg`, `front_pairs`, `back_pairs` |
| `tissue` | `k_bend`, `k_passive_frac`, `k_dist`, `bend_damp_s`, `c_drag_perp`, `c_drag_tan`, `g_eff`, `mu_floor` |
| `solver` | `substeps = 16`, `passes = 2` — bound in `PhysicsRunner` (the Config rule) |
| `reach` / `crawl` / `swim` / `grab` / `idle` | the P1–P5 dials, as v1 listed them (`prototypes/octopus_motion/README.md`), each a measured number at GAME scale from P5z (§8) |
| `rules` | `can_become_zombie = false`, `armed = false` (no loadout row is bound; slot 4 never fires for it), `blocks = "body"` |

**Scaling law** (P5z measures, the row records): same-duration similarity
from the 0.24 m animal to the mid one (L × ~10.4): mass ∝ L³, `k_dist` ∝ L³,
`k_bend` ∝ L⁵, grip limit ∝ L⁴, quadratic drag self-similar, `g_eff`
unscaled (Froude) — so durations stay biological and speeds scale with L,
the brief's §6 q1 premise.

**Door refusals** (the four number-doors): missing or mistyped key; a value
the physics cannot take (`nodes_per_arm < 3`, `substeps < 1`, a negative
stiffness, a fraction outside [0, 1]); and the **headroom check**: from the
row the door computes the smallest node mass in counts, the largest
`k·W` and `W` (§6.2) and the largest `rest·stretch` against the `atan2` door
(§6.6), and REFUSES a size that overflows — this is how #112's small and big
sizes are accepted or refused honestly, not by luck.

**Species join**: `SpeciesDef` gets `creature: str | None` (`None` for a
human); `species.py` gets the `"octopus"` row. `can_become_zombie` is
honoured by `convert_marines_to_zombies` (it only checks `killed_by_zombie`
today — a bug fixed in P6a, gated).

## 4. The octopus as a Unit

### 4.1 Construction and position convention
`Unit(name, x, y, team, footprint=<row>, species_id="octopus")`. `Unit.x/y`
are float physics-tile coordinates of the footprint's TOP-LEFT anchor;
`facing` is float radians, 0 = East, CCW on the y-down grid. The creature
body's pose is Q metres of the crown centre and a Q16 **radian** heading.
ONE conversion each way, in `creature_fixed.py` (`pose_to_unit_anchor`,
`unit_anchor_to_pose`), handedness pinned by a test. The body footprint
blocks and stamps like any unit (`stamp_units`); **arms never block, stamp,
jam a door, sense heat or gas, or count for melee adjacency** — see 4.2.

### 4.2 Hit seam: `hit_tiles()` ≠ `occupied_tiles()`
`occupied_tiles()` is the FOOTPRINT seam (stamps, doors, exchange rows,
melee adjacency, `occupies()`) and stays the body rectangle. A new
`Unit.hit_tiles()` defaults to `occupied_tiles()` and, for a creature unit,
returns the body tiles plus every tile an arm SEGMENT crosses (segments are
rasterised, not just node tiles: at 0.147 m segments on 0.333 m tiles there
are ~2.3 nodes per tile, but a segment can still clip a tile corner between
two nodes; rebuilt once per tick after slot 5b from the synced chain; sorted;
stored on the body, not digested — it is derived). Erik's observation of
2026-10-30: more than one node per tile is expected and harmless. ONLY the bullet march's footprint test and the
laser skewer read `hit_tiles()`. Melee and every exchange row keep
`occupied_tiles()`, by explicit choice: a bite lands on the body, heat is
felt by the body (an arm in a fire is #118's row). Gates: humans are
byte-identical through the new seam (golden + the ship_demo pin unchanged);
arm tiles leave `obstacles` and `dyn_heat_atten_q` unchanged; a shot across
an outstretched arm hits, across the empty crown-side tiles misses. The
struck node is NOT put on `UnitHitEvent` (hashed record; would move the pin):
it goes on the creature record as "last hit node" for P7/#118. New creature
damage sources (crush, strike) add an `events.Cause` row in their patch.

### 4.3 Who writes the position
"The arms move the animal" means: for a unit whose species has a creature,
- slot 3 movement and `OnePhaseWEGO.drive_units` SKIP it (keyed on the
  species field); a `MOVE_DIR` intent or a target intent is routed to the
  hub as a goal (travel direction / reach-grab target);
- `add_unit(position=)` sets the initial pose (the one exception, load-time);
- the **wave push** (slot 9c2) becomes a body IMPULSE on the `CreatureBody`
  through one coupling row in `exchange.py` (the push is physics the animal
  should feel; the next 5b integrates it instead of overwriting it) —
  vetoable: Erik may prefer the push ignored as an accepted gap;
- zombie conversion never happens (`can_become_zombie = false`);
- a GRAB does **not** move the grabbed unit in v1: the arm follows the unit
  (the attachment target is the unit's pose); the real two-way coupling
  (dragging, wading, struggling) is #107's seam. The P4 "carry" is therefore
  demonstrated on a creature-owned prop body until #107.

**Arms vs marines is UNDESIGNED by Erik's ruling (2026-10-30)**: neither
tile-blocking (wrong) nor a design decided tonight. v1 ships NO arm–marine
interaction (arms pass through marines, a NAMED ACCEPTED GAP), and the
coordinate-based collision Erik wants ("calculated from their coordinates",
capsule-vs-capsule between arm segments and a marine's body) is its own
design session after P6, with #107's coupling. P6c therefore collides nodes
with the static topology only; the unit-footprint pushout of v2's first
draft is withdrawn.

### 4.4 Collisions
Nodes collide with the STATIC topology (`material` + door state), never with
`gmap.obstacles` (which holds every unit's last stamp, the octopus's own
crown included). Wall face chosen toward the open neighbour with a fixed
integer tie-break. Nodes do NOT interact with units in v1 (4.3). The BODY collides too: disc vs blocked tiles and vs other units'
footprints, projected in the solver before the pose is written, so the stamp
never overlaps a solid.

### 4.5 Spawn and driver
A level declares a creature as `[[creature]]` rows in level.toml (`species`,
`x`, `y`, `team`, `heading`), written through `level_lib.write_creatures`,
read by the loader; `Simulation.spawn_creatures` is the one spawn (and
`spawn_match` calls it for a match.toml `[[creature]]` list). In a match
every unit has one driver: a `creature` controller row in the match
registry (`register_controller`) drives creature units through intents;
default `idle`-equivalent is the hub's own idle primitive (no intent = idle).

### 4.6 Team, faction, match end — ERIK'S CALL
`MatchRules.living_teams` counts every non-zombie team as a marine team, so
an octopus on its own team keeps a match open while it lives. Options:
(a) its own `CREATURE_TEAM` (like `ZOMBIE_TEAM`), hostile to all, no team in
the match count — the match ends on marines alone, the octopus a hazard;
(b) a match-config choice per creature row (`counts_for_victory`); (c) a
regular team member. Lead's recommendation: (a) for v1 with (b)'s key
added when a mode needs it. Not blocking (Erik: "it's a shame to block"):
P6 proceeds on (a) as the stated default; Erik vetoes at any point before
P6d1 merges.

## 5. Synced state, digest, recorder

Per creature unit, integer, hashed, in this record order (keyed by unit id):

| state | type | note |
|---|---|---|
| row provenance | version + content hash of the `[creatures.*]` row | finding 17 |
| node positions | (8, nodes, 3) int32, Q24 m (`CHAIN_FINE_BITS = 8`) | §6.1 |
| node velocities | (8, nodes, 3) int32, Q24 m/s | |
| body pose | x, y, z (Q24), heading (Q16 rad), vx, vy, vz (Q24), yaw rate (Q16) | |
| anchors | (8, nodes) uint8 mask + (8, nodes, 3) positions | floor or wall face |
| grabs | count-prefixed list sorted by (arm, node): unit_id, local offset (3 × Q24) | |
| hub | per-arm state (uint8), elapsed-tick counters, the drawn lengths, stance count, travel goal, steering window, last hit node | |
| rng | the xorshift state (uint64) | finding 13 |
| activation clock | per-arm elapsed substeps | |

Derived, NOT stored: activation, rest-scale, bend targets, root frames,
`hit_tiles`.

**Digest**: creatures get their OWN section and snapshot key `__creature__`
with preamble `CREATURE_SECT_V1` and an ALWAYS-WRITTEN carrier
`n_creatures` (the entity section's carrier gates on `n_entities > 0` and an
octopus is not a level entity, so a creature could not ride it). Folded
after the entity and signal sections only when `n_creatures > 0`; a
`require_creature_carrier` twin of the strict entity rule. One serializer,
`creature_system.serialize_record`, shared by digest and recorder. The
field-plane spec version does not move; `CREATURE_SECT_V<n>` bumps only when
the ENCODING changes (a tuning change alters trajectories, not bytes). The
A/B harness's `diff_trajectories` learns to locate a creature divergence per
arm and node. Absence transparency: no creature → no section → every
octopus-free digest byte-identical to today's; the golden untouched.

**Recorder**: a presence-gated `creature_snapshots` ring beside
`entity_snapshots`, holding the section bytes per tick, emitted in `dump()`
only when non-empty — a contract extension (the Recorder rule), not a cast.

## 6. The chain in integers — the arithmetic that must survive the port

### 6.1 Currency
Positions and velocities are int32 in a **fine chain currency**, Q24
(`CHAIN_FINE_BITS = 8`, the `TRACE_FINE_BITS` precedent): 60 nm per count,
±128 m of map (the door refuses a larger level), products in int64 via
`mul_wide`. Why not Q16: with Q16 metres and the PBD velocity rebuild
`v = (p − p_prev)/dt`, every acceleration below one count per substep is
erased — buoyancy-reduced gravity (0.188 m/s², 0.08 Q16 counts per substep)
would never sink a dead arm. In Q24 it is 20 counts. Every reader converts
ONCE through `creature_fixed` (`chain_from_m`, `chain_to_f32`) or the kit's
twin; no inline shift. Masses in kg Q16 (the mid tip node is tens of grams,
~2 k counts; the door's headroom check guards the small size). Angles in
**radians Q16** everywhere in sim state (the kit's trig is radians); turns
only where the light cones need them.

### 6.2 Constraint projection: stiffness form, no α̃, no stored λ
The prototype validated **λ reset every pass** (`dlam = −C/(W + α̃)`,
`arm_chain.py::_project_distance` / `_project_bend`); the integer reference
reproduces THAT, not accumulated XPBD — the "16 × 2" finding belongs to it.
With dt = 1/(tps·n_sub) = 1/384 s, `1/dt² = 147456` is an exact integer, so:

    Δλ = −(k·C) / (k·W·(1+γ) + 147456)        [bend: numerator −(k·C + k·D)]

W = Σ wᵢ|∇Cᵢ|², evaluated in int64, ONE `floordiv` of a pre-shifted
numerator, **never** `reciprocal_q16` (it self-guards `denom ≤ 0` by
returning 0, which would make the passive tip rigid; and it keeps ~2 counts
of a 2.8e4 denominator) and never a narrowed W. No λ is stored, so no λ
currency question. The damping term `D = γ·(∇C · (p − p_prev))` as the
prototype, `γ = bend_damp_s · 384`.

### 6.3 Symmetric rounding — the "no slide" guarantee
In reals the projection's impulses sum to zero (Σ mᵢΔpᵢ = Δλ Σ∇C = 0); that
is why an unanchored arm cannot push the body (study §3.C). A floor toward
−∞ on each `wᵢ·Δλ·gᵢ` would bias every correction and let the animal crawl
on rounding alone. Rule: every signed correction and every signed narrow in
the chain rounds **symmetrically** (round-half-away-from-zero, the kit's
signed-round helper; never a bare `>>` or `floordiv_q` on a signed
correction). Where two nodes share a correction the magnitude is computed
once and applied ± (the face-flux discipline). Gate: a free animal (no
anchors, idle waves on, symmetric drag) moves its centre of mass < ε over
60 s; a dead arm released 0.2 m above the deck sinks to it and stays.

### 6.4 The bend in 3-D
A centre-line chain has no material frame, and the unsigned 3-D angle is
singular at straight — the very state a reach drives to. So: a **Bishop
frame parallel-transported from the root** along each arm (transport in the
reference, integer rotation by the kit's trig), and **two signed bend
constraints per interior node** (about the frame's two normals), each the
2-D constraint of the prototype in its plane (`atan2_q16` of the in-plane
cross / dot). A primitive's `bend_dir` is a direction in that frame; the
stiffening wave scales both constraints' k; the passive fraction applies
to both. Twist is render-only (CONTROLS `twist`), never a constraint.

### 6.5 The activation wave
Yekutieli's `0.5·(1 + tanh((s_front − s)/w))` is not in the kit. It becomes
a **checked-in Q16 table** over a fixed argument grid (`activation_table.inc`,
generated by `tools/gen_activation_table.py`, the L° precedent, with its
twin test), read at the per-arm elapsed-substep clock. The reach gate is
re-run on the table in P6b. Spawn-pose powers (`w ** 1.5`) are load-time
poses computed once into the row's rest pose, not runtime.

### 6.6 Trig headroom
`atan2_q16` takes int32: the Q48 cross and dot of two Q24 segment vectors are
narrowed by the SAME shift (the angle is scale-free) to fit 30 bits — the
shift derived from the row's `rest · stretch_max` at the door (§3 headroom).
Drag squares and the sign through `mul_wide` + a saturating narrow;
`sqrt_q16` on the Q48 sum of squares as the kit documents.

### 6.7 Medium
Buoyancy and drag read the tile's medium; v1 ships water everywhere for the
arm (ACCEPTED GAP until #107 asks what a dry corridor does) and notes the
tile.

### 6.8 Reference first, C++ second, float before both
Order: (1) **P5z**, a float game-scale 3-D step in `prototypes/octopus_motion/`
(the P1–P5 gates re-run at the mid size in 3-D with z, floor and the Bishop
frame, row values by §3's law — the first 3-D chain must not also be the
first integer one); (2) the **integer reference** `docs/octopus_chain_ref_q.py`
(ints only; bit-exact Python twins of `atan2_q16`/`sin_q16`/`cos_q16`/
`sqrt_q16` added beside `floordiv_q`/`reciprocal_q16`'s and held equal to
the C++ by a twin test), its gates = P1's seven on ints + the §6.3 gates + a
float-vs-int TOLERANCE gate against P5z's traces; (3) `ArmChainSolver`
(`cpp/src/arm_chain_solver.{h,cpp}`, `/fp:strict`, author + paper header:
Macklin 2016/2019, Yekutieli 2005 I/II, Levy 2015, Levy & Hochner 2017,
Katija 2025 — all in `docs/papers/octopus/`) held to the reference at tol 0.

**CUDA twin**: deferred, recorded debt — the chain goes to the device when
units do, born `(N, 8, nodes, 3)`. Vetoable.

## 7. The conductor slot
`Simulation.step()` **slot 5b — creature bodies**, after `_run_zombie_slot`
(5) and before the obstacle re-stamp (6): intents → hub → solver inputs →
ONE solver call per creature (loop ordered by unit id) → body pose → unit
anchor + facing → anchors, grabs, `hit_tiles` → record. One call line,
`self._creature_system.tick(...)`. Dormancy: no creature → no body, no call,
no section.

## 8. Patches (each one worktree agent, one writer; design-gated: P6b, P6c, P6d1)

| patch | content | gate |
|---|---|---|
| pre | merge `main` into the branch; re-read §4/§7 | suite green |
| P5z | float game-scale 3-D prototype step (z, floor, Bishop frame, scaling law → row values, traces) | P1–P5 gates at the mid size in 3-D |
| P6a | row + door (`creatures.py`, `creature_fixed.py`, headroom check), species join, `can_become_zombie` honoured, `armed = false` path | door refusals incl. an over-headroom size; ingress lint; conversion gate |
| P6b | integer reference + kit twins + activation table + gates | P1 seven on ints; dead arm sinks; no slide; float-vs-int tolerance; twin test |
| P6c | `ArmChainSolver` C++ (stateless, buffer contract, static-topology contacts, body collision; no arm–unit contact) | tol 0 vs reference over 10 s reach / crawl / fall inputs; ratchet unchanged |
| P6d1 | `CreatureBody` + slot 5b + `[[creature]]` spawn + `__creature__` section + recorder ring + writer routing (4.3) + body collision | a LIMP octopus (dead-arm signature) first; A/B harness finds a planted divergence; digest identity CPU / `--cuda` / `--resident`; WEGO round boundary; pickle round trip; golden untouched |
| P6d2 | primitives: reach, crawl, swim, grab (prop body), idle; the `creature` controller row | P2–P5 gates through `Simulation` on a flooded room, dequantized units; P2 rhythm after the 2-tick comb |
| P6d3 | `hit_tiles()` seam + the two march sites | humans byte-identical (golden + ship_demo pin); arm hit / crown-side miss; stamps unchanged |
| P6e | `tools/bench_octopus.py`: ms / octopus / tick by part (hub, chain, contacts) — REPORTED, no threshold | a number in the report |

HUMAN-TEST before merge: none (nothing shows until P7); P6d2 re-points the
prototype GIF harness at the sim so Erik can see the ported animal early.

## 9. Systems

**Existing canonical systems used**: Unit (one class, species row); Config
(one door per table; solver params bound in PhysicsRunner); Material / gas
tables pattern; Fixed-point kits; Q16 boundary modules; Trace currency
(the fine-currency precedent); PhysicsRunner / PhysicsEngine; Tick
conductor; Field digest + A/B harness (a new section, its carrier rule);
Recorder (additive, presence-gated ring); Ingress lint + float ratchet;
Integer-reference pattern; Control seams (intents → goals); Match
controller registry; Level data layer (`level_lib` writes `[[creature]]`);
Coupling table (the wave-push row); Credited cause; Emissive/L° table
precedent (a checked-in generated table); Test conventions.

**New systems, with draft rules** (written into CLAUDE.md at implementation):

| system | where | draft rule |
|---|---|---|
| Creature table | `config.toml` `[creatures.*]` + `src/simulation/creatures.py` (`CreatureDef`) + `creature_fixed.py` | A creature's body plan, tissue, solver resolution, primitive dials and rules are ONE versioned `[creatures.*]` row, quantized once at the door, refused when missing, out of range or over the integer headroom; joined to a species by `SpeciesDef.creature`; its version + content hashed in every creature record. Never a per-creature constant in code, never a second table (#104) |
| Chain currency | `CHAIN_FINE_BITS` in `fixed_point.h` + `creature_fixed.chain_*` twins | Chain positions and velocities are int32 Q(16+8); every read or write converts ONCE through the kit or `creature_fixed`; symmetric rounding on every signed correction; held equal by a twins test |
| Arm chain solver | `cpp/src/arm_chain_solver.*` + reference `docs/octopus_chain_ref_q.py` | THE articulated-body solver for every creature with arms: stateless over `CreatureBody`-owned buffers, fixed in/out contract (contacts, grip, pops), stiffness-form projection with λ reset per pass, Bishop-frame bends, kit trig, static-topology collision; a change lands in the reference FIRST, the C++ held at tol 0. Never a second chain, never float in sim state |
| Creature bodies | `src/simulation/creature_system.py` (`CreatureBody`, slot 5b, `serialize_record`) | THE runtime turning a creature unit's intents into primitives and driving the solver; the only per-tick writer of a creature unit's anchor and facing (load-time `add_unit` and the wave-push coupling row excepted); state integer, own xorshift, hashed in `__creature__`; a new primitive is a state machine here over row dials |
| Hit seam | `Unit.hit_tiles()` | Bullet and beam hits test `hit_tiles()`; stamps, doors, exchange rows and melee test `occupied_tiles()`; a human's two sets are equal. Never a third occupancy set |
| Chain accessor | `creature_system.read_chain` (P7) | Render, tools and rules read a creature's nodes only through it (float32 views), never the buffers |

## 10. Critique disposition (v1 → v2)
Blockers 1–5: fixed (§4.2 hit seam; §5 own section; §6.2 stiffness form;
§6.1 Q24; §6.3 rounding + gates). Majors 6–15: fixed (§6.2 prototype's
variant; §8 P5z; §6.4–6.6; §4.3 writers; §3 `armed`/`can_become_zombie`,
§4.1 conventions, §4.5 spawn + driver, §4.6 → Erik; §5 recorder ring; §4.4
collisions; §2 xorshift; §2 contract + stateless solver; pre-patch merge).
Minors 16–23: fixed (§4.2 last-hit node on the record; §5 provenance +
encoding; §8 split + gates; §3/§6.8/§9 rules named; §2 counters + comb;
§6.6; §6.1 radians; §3 one mass). Nothing in the critique's "keep" list was
lost.
