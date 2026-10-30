# Critique of `docs/octopus_p6_sim_port_design_2026-10-30.md` (draft v1)

Adversarial pass, 2026-10-30. Read against `main` (932eafa), not only this
branch, which is 56 commits behind it (see finding 15). Erik's rulings (cost not
a design input, z in the chain, kit trig, octopus is a Unit, row in config.toml,
arms hittable) are taken as given. Severity: BLOCKER, MAJOR or MINOR.

## Findings

### 1. BLOCKER: putting the arm tiles into `occupied_tiles()` turns arms into blockers, stampers, door-jammers and heat/gas sensors
§4: "The octopus's `occupied_tiles()` becomes its body tiles PLUS the tiles its arm nodes lie in … HIT tiles and BLOCKING tiles are different sets."
`occupied_tiles()` is not the hit seam. It is the footprint seam for:
- `stamp_units` (gamemap.py:2020, 2148 on main): obstacles, `dyn_permeability`, `dyn_wave_absorb`, `dyn_heat_atten_q` (digested, spec v6) and `dyn_light_atten_q`.
- `door_system.py:209`: a door cannot close on a unit tile.
- `exchange.py:357/579/679/725`: heat, poison and pressure exposure, and the `apply_wave_push` gradient.
- `combat.py:1379` (the shooter's own tiles) and `combat.py:1566-1567` (melee adjacency, which would let a zombie bite at 2.5 m and let the octopus strike at 2.5 m).
- `Unit.occupies()`, which reads `offsets` and would silently disagree with the new set.

So the change contradicts the design's own "arms never block and never stamp" and moves the radiation and digest planes whenever an arm moves.
**Fix:** leave `occupied_tiles()` as the body footprint. Add a separate hit seam, e.g. `Unit.hit_tiles()`, which defaults to `occupied_tiles()` and for a creature returns body plus node tiles. Route only the bullet march (combat.py:341) and the laser skewer (combat.py:1202) through it. Melee and the exchange rows each make an explicit choice of seam, with the choice written down. Add a gate that the arm tiles leave `obstacles` and `dyn_heat_atten_q` unchanged.

### 2. BLOCKER: `CREATURE_SECT_V1` cannot be "an ENTITY-section record kind"
§5: "a new ENTITY-section record kind … absence transparent exactly as the vents and charges are."
Vents and charges are level entities. Their records are framed `{ordinal}|{id}|{class_name}`, with schema-declared FIELDS from the registry and provenance from `registry_content_hash` (entities/serialize.py:121-171). An octopus Unit has no ordinal, no entity id and no registry class.

Worse, the fold is gated on `carrier["n_entities"] > 0` (field_digest.py `tick_digest`). On a level with an octopus but no entities, which is the P6d flooded test room, the creature state would not be hashed at all. That is a silent unhashed synced state, the failure the A4 strict-carrier rule exists to prevent.
**Fix:** give creatures their own section and snapshot key, `__creature__`, with its own preamble `CREATURE_SECT_V1` and a carrier that is always written (`n_creatures`). It folds after the entity and signal sections only when `n_creatures > 0`, and has a `require_creature_carrier` twin. Records are keyed by unit id and use one serializer, shared by digest and recorder. Teach `field_ab_harness.diff_trajectories` to locate a creature divergence per arm and node. Hash the lottery's RNG state in the record (finding 13).

### 3. BLOCKER: the XPBD arithmetic as written does not fit Q16
§6: "α̃ = 1/(k·dt²) … the divide is the kit's `floordiv_q` / `reciprocal_q16` … the per-constraint λ is int32 Q16."

Worked numbers, at dt = 1/384 s:
- **dt² is not representable.** dt² = 6.8e-6 s² is 0.44 counts, so it is 0. dt itself is 170.67 counts and rounds to 171 (0.2 % off).
- **α̃ overflows or collapses at the passive tip.** The prototype's own passive tip k is 0.01·0.1·(0.912/7.5)³ ≈ 1.8e-6 N·m/rad, giving α̃ ≈ 8e10. Scaled to the game arm (k ∝ L⁵, L × 10.4) k is ≈ 0.22, giving α̃ ≈ 6.7e5. Both exceed int32 Q16's 32767. If it is computed as `reciprocal_q16(k·dt²)`, then k·dt² quantizes to 0, and `reciprocal_q16` self-guards `denom_q <= 0` by returning 0 (fixed_point.h:687). So α̃ becomes 0 and the limp tip turns rigid. That inverts the dead-arm gate.
- **λ in Q16 is a dead band.** Near the tip at game scale (node ≈ 13 g, w ≈ 74/kg, |∇C| ≈ 1/0.147 m), Σw|∇C|²·(1+γ) ≈ 74·(46+185+46)·1.38 ≈ 2.8e4. So Δλ = C/2.8e4 is 0 counts for any C below about 0.4 rad. The distance constraint's tip Δλ = C/148 is 0 for C below about 2 mm. Small errors are never corrected, and P1's ≤ 2° straightness gate will measure the dead band, not the muscle.
- **`reciprocal_q16` loses its bits on large denominators.** For a denominator of 2.8e4 it returns about 2 counts, a 40 % error. The same denominator sits about 15 % under the int32 Q16 ceiling at the mid size, and overflows at #112's small size (w ∝ L⁻³, |∇C|² ∝ L⁻²).

**Fix:** write the constraint in stiffness form. dt is exactly 1/(tps·n_sub), so 1/dt² = 147456 is an exact integer:
Δλ' = −(k·C + 147456·λ') / (k·W + 147456)
evaluated in int64 with one `floordiv_q` of a pre-shifted numerator. Never form α̃, never use `reciprocal_q16` here, and never narrow W to int32. Hold λ (if it is kept at all; see finding 6) in a wide currency such as Q32 int64. The door computes the worst-case magnitudes of the row (min node-mass counts, max W, max k·W) and refuses a size whose headroom fails. That door is what lets #112's sizes be accepted or refused honestly.

### 4. BLOCKER: Q16 positions with the PBD velocity update erase every small acceleration
§6: "Velocities in m/s Q16", plus the prototype's `v = (p − p_prev)/dt`.
With positions in Q16 metres, Δp per substep is a whole number of counts, so a derived velocity is a multiple of 384 counts, i.e. 5.9 mm/s.
- Buoyancy-reduced gravity, g_eff = 0.188 m/s², adds 0.49 mm/s per substep. That is 0.08 counts of displacement, which floors to 0. The next substep's v is rebuilt from Δp = 0, so gravity is erased every substep: a dead arm never sinks, and the floor contact never engages from rest.
- At 9.81 m/s² (if air ever means air) it is 4.36 counts per substep. `floor` gives −5 going down and +4 going up, so gravity is 15 % stronger on falling than on rising.

**Fix:** carry positions in a finer currency, as `TRACE_FINE_BITS` does for smoke (e.g. Q16+8 in int32, which bounds the map to ±128 m at the door, or Q32 in int64 with `mul128_shr`). Alternatively keep v as the state and integrate p with a carried remainder. Either way, say which in §6 and add a gate: a dead arm released 0.2 m above the deck reaches it and stays there.

### 5. BLOCKER: integer rounding breaks the "no slide" guarantee the crawl depends on
Study §3.C: "bend forces sum to zero inside the chain so the arms cannot push the body without an anchor or water — that is the 'no slide' guarantee, structural." The design drops this property silently.
In reals, Σ mᵢΔpᵢ = Δλ·Σ∇C = 0. In integers, each `wᵢ·Δλ·gᵢ` is rounded separately, and the kit's `>>` and `floordiv_q` floor toward −inf. Every projection therefore creates a biased impulse. Coherently, that is (17 + 16) constraints × 2 passes × 16 substeps × 8 arms per tick. An octopus could crawl by rounding alone, and P5's "body drift 0 mm in 60 s" becomes a rounding artefact in either direction.
**Fix:** round symmetrically (`shr_round0_signed_i64` or round-half-even, never a bare `>>` or `floordiv_q` on a signed correction). Where the masses allow, compute one correction and apply it ± (the face-flux discipline). Add an integer gate: a free animal with no anchors, activated idle waves and symmetric drag moves its centre of mass < ε over 60 s.

### 6. MAJOR: the formula in §6 is not the algorithm P1 validated
§6 writes Δλ = (−C − α̃λ)/(Σw|∇C|² + α̃) with λ "reset each substep, as the prototype does".
The prototype keeps no λ at all. `arm_chain.py::_project_distance` computes `dlam = -C/(wsum+alpha)`, and `_project_bend` computes `dlam = (-C - damp)/denom`; `arm_bank.py:185,210` is the same. That is λ reset every PASS. With 2 passes, true XPBD accumulation is stiffer and behaves differently, and "16 × 2 stands (8 × 1 fails, 32 × 1 whips)" is a finding about the prototype's variant only.
**Fix:** the integer reference reproduces the prototype exactly (no α̃λ term). Alternatively, re-run P1's gates in float with accumulation before choosing it. Either way, state it.

### 7. MAJOR: there is no game-scale baseline; every prototype ran at 0.24 m
§3: "each value a prototype-measured number (the README's)". §6: "the port is at game scale from the first commit". §8 P6b: "P1's seven on ints at game scale".
Every prototype constructs `ArmGeometry()` at its 0.24 m default (crawl_sim.py:244, grab_sim.py:178, idle_sim.py:143, swim_sim.py:150, reach_gates.py:42). No game-scale value has been measured.

Scaling is not uniform. Under same-durations similarity, mass goes as L³, quadratic drag stays self-similar, but bend stiffness must go as L⁵, k_dist as L³ and the grip limit as L⁴, while g_eff does not scale (Froude). So if P6b's integer gates fail, nobody will know whether the cause is scale or quantization.

z was also never prototyped: the README's P4 section says "'Lift' is out of this plane … The z of the chain is the brief's open question 2". P6b would be the first 3-D chain ever and the first integer chain at once.
**Fix:** insert a float step before P6b: the P1–P5 gates re-run in 3-D at the mid size, with the row's values derived by a stated scaling law, in the prototype folder. Its output is the row and the float traces the integer reference is compared to (a tolerance gate, float vs int, beside the tol-0 C++ vs reference gate).

### 8. MAJOR: the 3-D bend constraint and the activation wave are not specified, and the reference has no kit twins
§6: "angle … via `atan2_q16` of the 2-D cross / dot in the segment's bending plane; in 3-D the bend target is a direction (the hub's `bend_dir`), so the constraint acts on the component of the turn in that plane and a passive bend stiffness on the rest."
Several pieces are missing:
- **The 3-D gradient.** The 2-D gradient `perp(d)/l²` has no 3-D form until a plane normal is fixed, and the unsigned 3-D angle is non-differentiable at straight, which is exactly the state a reach drives to. A plane built from `bend_dir × tangent` flips when the tangent passes `bend_dir`, which happens in every dorsal curl and wrap. A centre-line chain has no material frame to carry `bend_dir`.
  **Fix:** two signed bend constraints per interior node about a frame parallel-transported from the root (or a curvature-vector constraint), with the frame transport in the reference.
- **The activation wave.** Yekutieli's sigmoid is `0.5·(1+tanh(...))` (arm_chain.py `set_activation_wave`). The kit has no tanh or exp. The spawn poses use `w ** 1.5`, and spawn is runtime (`spawn_match`), not load.
  **Fix:** a checked-in table regenerated by a tool (the L° precedent) or an integer rational, plus a re-run of the reach gate on it.
- **Python twins.** The kit's Python twins exist only for `floordiv_q` and `reciprocal_q16` (sweep_ref_q.py:630, 702). There is no Python `atan2_q16`, `sin_q16`, `cos_q16` or `sqrt_q16` anywhere, and `gas_fixed` has none either.
  **Fix:** P6b must write bit-exact twins, held equal to the C++ by a twin test (the trace-currency four-twins precedent).

### 9. MAJOR: "the only writer of a creature unit's position" is false on main
§4, §9: "Its position is WRITTEN by the creature body each tick … the arms move the animal, nothing else does." Other writers, all ignored by the design:
- `apply_wave_push` at slot 9c2 (exchange.py:481-499, 600-640) moves every living unit. The next 5b overwrites the push, so a blast never moves an octopus, and the KNOCKED_DOWN status rides on it.
- `Simulation.add_unit(position=)` (simulation.py:455-468).
- Slot 3 `_update_player_movement` and `MOVE_DIR` intents, which §4 says the hub "consumes".
- `OnePhaseWEGO.drive_units` (simulation.py, the `drives_units` branch).
- Zombie conversion: `ai_zombie.convert_marines_to_zombies` ignores `SpeciesDef.can_become_zombie` (it checks only `killed_by_zombie`) and revives the unit. A zombie-killed octopus becomes a zombie driven by slot 5 as well as 5b.

The grab also makes the creature system a second writer of a MARINE's x/y (§6 "Units").
**Fix:** state the routing for each writer.
- Exclude creature units from slot 3 and from `drive_units`, keyed on `species.creature is not None`, which is data, not "if octopus".
- Route `MOVE_DIR` and targets for creature units to the hub.
- Make the wave push a body impulse on the `CreatureBody` (one coupling row in exchange.py), or exclude creatures as a named ACCEPTED GAP.
- Set the octopus row `can_become_zombie = False` and make conversion honour it (that is a bug today).
- For v1 the grab does not move the grabbed unit (the arm follows it), until #107 designs the coupling seam.

### 10. MAJOR: `Unit.__init__` and the match code assume a marine
§4: "HP, death, conversion, … the match's team counting: unchanged code."
- **It would be armed.** On main, any non-zombie gets `_loadout_tables().loadout(None)`, the marine default rifle and grenades (unit.py:237-243), and `[loadouts.*]` refuses a row without a weapon (loadouts.py:82). Slot 4 would shoot with it. Needs an unarmed path (a species field or a weaponless loadout row via the door).
- **It would hold a match open.** `MatchRules.living_teams` counts every non-zombie team as a marine team (ruleset.py:334-337), so the octopus's team keeps a match from ending. Which team or faction an octopus is on is a game-design question. It is Erik's call, so ask him.
- **It has no driver in a match.** "In a match every unit has ONE driver and it is a controller" (CLAUDE.md, match controller registry). The design has no controller row, no `match.toml` key and no registry entry.
- **It has no spawn path.** `spawn_match` spawns teams and zombies only, `[[spawn]]` is the authored path, and nothing says how a level declares an octopus. P6d's test room needs it, and the declaration must be written through `level_lib`.
- **Position convention.** `Unit.x/y` are float physics-tile coordinates of the footprint's TOP-LEFT anchor (`tile_x = int(x)`, footprint rectangle `[tile_x, tile_x+footprint)`, `center_tile_x = tile_x + footprint//2`). They are not the crown centre in metres. Writing the centre shifts the hitbox and stamp by one tile.
- **Facing convention.** `facing` is float radians, 0 = East, CCW, on a y-down tile grid. The design has heading in "Q16 turns".

**Fix:** pin one conversion function (Q16 m → anchor tiles; heading → facing), one handedness, and one angle unit (the kit's trig is radians; see finding 22).

### 11. MAJOR: the recorder claim is wrong three ways
§5: "positions are added to `DEFAULT_FIELDS` additively as a per-tick (8, 18, 3) int32 block … the contract's int32 branch covers it."
- `DEFAULT_FIELDS` names gmap planes with ring buffers `(capacity, fh, fw)` (recorder.py:81-121).
- There is no int32 branch. int32 Q16 fields are dequantized into a float32 ring; only `_BOOL_FIELDS` and `_INT64_FIELDS` keep their dtype.
- The number of octopuses is variable.

**Fix:** add a presence-gated `creature_snapshots` ring beside `entity_snapshots`/`signal_snapshots`, storing the section bytes from finding 2's serializer, emitted in `dump()` only when non-empty. Per the Recorder rule, this is a contract extension, not a cast.

### 12. MAJOR: the body has no collision, and the nodes would collide with the wrong map
§6 covers nodes against walls and units only. The body, a rigid disc driven by jets and anchor reactions, can be driven into a wall or into a marine's footprint. Its stamped footprint would then overlap solids and other units, which `stamp_units`, wave push `_free` and pathing all assume cannot happen.

"A node inside a blocked tile" is also ambiguous. `gmap.obstacles` contains every unit's stamp from the previous tick, including the octopus's own body. Collide against it and the arms collide with their own crown, and push out of other units twice.
**Fix:**
- Add body-vs-wall and body-vs-footprint projection in the solver.
- Make nodes read the static topology (material and door state), never `obstacles`.
- Choose wall faces only toward open neighbours, with a fixed integer tie-break.

### 13. MAJOR: the RNG is changed from the study without saying so
§2: "seeded through the sim's one RNG hand-off (never its own generator)". The study §3.C says "seeded `xorshift` for the lottery", and its game-state split lists "the lottery's RNG state" as digested.
Drawing from `sim.rng` couples every arm decision to how many draws slots 1–5 made that tick: one extra shot reshuffles the octopus. It also makes P2–P5 gates through `Simulation` scenario-dependent, and the state is not in any digest.
**Fix:** a per-creature integer xorshift seeded from the sim seed and unit id, hashed in the creature record. Draws use integer ranges (jitter in ticks), and lottery weights are integers.

### 14. MAJOR: the solver/hub contract and solver-state ownership are undefined
§2 lists solver inputs (activation, rest scale, anchor, grab) but no outputs.
- Anchor pops happen per substep on a grip-force test (arm_bank.py:215-225).
- P4's "stop on contact" and the wrap's "seal what touches" need contact reports per substep.
- Activation is evaluated per substep from a clock (`set_activation_wave(cmd, self.time)`).

"ArmChainSolver (PhysicsEngine-owned)" holding state also blocks `pickle`/`deepcopy` of a sim (simulation.py:150-163: "AI training serializes via pickle"), and leaves no path when no engine is bound.
**Fix:** make the solver stateless over int32 or int64 numpy arrays owned by `CreatureBody` (the pybind buffer pattern). Born `(N, 8, 18, 3)`, per RL-batch §A. Define a fixed per-tick in/out record including pops, contacts and grip forces, computed by the solver, not the hub. Activation's clock is a per-arm elapsed-substep counter.

### 15. MAJOR: the design was written on a stale base
This branch is 56 commits behind main. Since the fork, main has added:
- #119: `ZOMBIE_TEAM = 255`, any-int teams, `unit.slot`, loadouts and suits.
- #122: slot 5 is `_run_zombie_slot` with `drive_zombies`, and the one-driver-per-unit rule.
- #123: `events.Cause` and `attacker_id` hashed in the unit digest.
- #121/#125: bots that shoot on sight.

The worktree's `simulation.py:1445` still shows `update_zombies_tick` in slot 5. Findings 9, 10 and 16 come from main.
**Fix:** merge main into `33-octopus-concept` before P6a and re-read §4 and §7 against it.

### 16. MINOR: hit attribution and causes
§4: "The hit event records WHICH node was struck (the packet's existing position)."
`DamagePacket` has no position (damage.py:139-170). Adding one to `UnitHitEvent` changes a record hashed in the per-tick unit digest (`_SYNCED_EVENT_TYPES`), which moves the ship_demo `tick_digest` pin even with no octopus present.
**Fix:** keep the struck node in a render-only event or the creature record. Any new octopus damage (grab crush, arm strike) adds an `events.Cause` row in the same patch (the Credited cause rule).

### 17. MINOR: a row-version bump is not a section-format bump
§3/§5: "A row-version bump bumps `CREATURE_SECT`'s version." A tuning change alters trajectories, not the byte format.
**Fix:** hash the row's content or version into each creature record (as `registry_content_hash` provenance does for entities). Bump `CREATURE_SECT_V<n>` only when the encoding changes. Specify the variable-length `grabs` encoding (count prefix, sorted by arm and node) and the record order (unit id).

### 18. MINOR: the patch plan has an oversized patch and missing gates
P6d bundles hub, five primitives, slot, digest section, recorder, hit seam and two combat sites. It cannot be bisected.
**Fix:** split it.
- P6d1: `CreatureBody` + slot 5b + spawn + `__creature__` section + recorder, with a limp octopus (the dead-arm signature) as the first gate.
- P6d2: the primitives.
- P6d3: the hit seam.

Missing gates:
- the A/B harness locates a planted creature divergence;
- digest identity with an octopus on CPU, `--cuda` and `--resident` (units are host-side, but a physics backend switch must not move it);
- a TwoPhaseWEGO round boundary;
- a pickle/deepcopy round trip;
- the door refusing a size beyond headroom (finding 3);
- the no-slide gate (finding 5);
- the dead-arm-sinks gate (finding 4);
- humans byte-identical through the new hit seam (golden plus the ship_demo pin);
- a float-vs-int tolerance gate (finding 7).

### 19. MINOR: CLAUDE.md rules not named
- **Credit the source.** `arm_chain_solver.{h,cpp}`, the reference and `creature_system.py` carry author and paper headers: Macklin 2016 / 2019, Yekutieli 2005 I/II, Levy 2015, Levy & Hochner 2017, Katija 2025, all already in `docs/papers/octopus/`.
- **Config rule.** "solver params bound in PhysicsRunner only", so the tissue and solver group is bound there, not by `creature_system`.
- **F5 reload.** `CreatureDef` is construction-bound like the weapon tables. A mid-run reload must not retune live octopuses on one peer.
- **Door registration.** `creature_fixed.py` is registered with `test_ingress_lint`.
- **RL-batch §A.** The new per-tick host logic is recorded as debt against "no new host-side tick logic".

### 20. MINOR: timers and clock quantization
The prototype's wave is keyed on absolute time (`t − cmd.t_start`, `self.time`). It must become per-arm elapsed counters, never compared to `sim.tick`, because TwoPhaseWEGO rewinds `sim.tick`.

`decision_s = 0.1` becomes 2 ticks (0.083 s, a 12 Hz decision comb). That is fine but should be stated, and the P2 spectral gate re-checked after quantization.

### 21. MINOR: atan2 inputs and overflow edges
Pass the int64 Q32 cross and dot to `atan2_q16` un-narrowed. Its args are int32, and Q32 fits only while |d| ≤ 46341 counts (0.707 m), so put a door on rest·stretch, or narrow with a stated precision. Order the multi-octopus loop by unit id. Use sorted tiles or a mask, not iteration over a Python set, for the hit tiles.

### 22. MINOR: two angle units
§5 has heading in "Q16 turns, as the sweep's cones". §6 has "Angles in radians Q16". The kit's `sin_q16`/`cos_q16` take radians, so every heading-to-root-frame rotation would need a turns→radians product with its own rounding.
**Fix:** use radians in sim state, and convert to turns only where the light cones need them.

### 23. MINOR: two masses
`Unit.mass` comes from the species stat draw (add_unit), and the row has `body_mass_kg`.
**Fix:** one owner. The row sets the species mean, or the unit's mass is ignored for creatures, stated.

## What the design gets right (keep it)
- Reference first, C++ held to it at tol 0, on `/fp:strict`. The sweep's discipline is the right one, and the only way findings 3–5 will be caught.
- One `[creatures.*]` row behind one door, quantized once, refusing what is missing, joined to species by `SpeciesDef.creature`. No "if octopus".
- The octopus as one `Unit` with the arms in a separate, hashed `CreatureBody`: no pose state on `Unit`.
- The intent that hit and block are different sets. Finding 1 only says it is not delivered.
- Dormancy: no octopus means no body, no call, no record, and an unchanged golden.
- Slot 5b before the re-stamp, so the body footprint is stamped from this tick's pose.
- Cost measured, never gated, by part. The CUDA twin deferred as recorded debt with the `(N, 8, 18, 3)` layout named.
- Arms do not stamp. The vacuum/air medium is an explicit ACCEPTED GAP.

## Verdict
Redesign §5 (the digest carrier and recorder) and §6 (the Q16 chain: stiffness form, wide λ, finer positions, symmetric rounding, the 3-D bend, the tanh). Add a float game-scale 3-D step before P6b. §3, §4, §7 and §8 are ready once findings 1, 9, 10 and 12–18 are fixed.
