# Breach — project instructions

Breach is an ML/RL project wearing a game's clothes: a deterministic GPU
physics engine as a state space, then agents trained in it for emergent
strategy. Every rule below serves that.

## Where things live

- **Rules**, in three homes (master CLAUDE.md, "Where knowledge lives"):
  this file holds the iron rules and ONE index line per canonical system;
  `.claude/rules/<region>.md` holds each system's constraints, firing on
  contact with its region's files (`paths:`; `.claude/hooks/path_rules.py`
  makes them fire in worktrees too); procedures live in skills. A rule holds
  only what the code cannot say (who must use it, what must never be built
  beside it, what changes together, Erik's rulings, accepted gaps,
  incidents), never history. The long form this file had until #206:
  `git show deade9d2:CLAUDE.md`.
- **Procedures**: project skills in `.claude/skills/` (tracked in this repo),
  provisional until proven (see Working style).
- **Tasks & ordering**: GitHub issues at `eriiiko/breach`. The pinned
  **Roadmap issue (#46)** holds the arc sequencing: work top-down unless Erik
  reorders.
- **Narrative**: `docs/architecture/` is canon (canon-fold DEFERRED by ruling:
  code + git history are the source of truth while systems still land
  weekly). Everything else in `docs/` is append-only capture (dated docs);
  arcs archive their working docs to `docs/archive/` at close. As-built notes,
  gate results and measurements go in the design doc, never in a rule.

## Environment (specifics in docs/dev_setup.md + docs/lenovo_dev_setup.md)

- **Python: always this machine's breach env** (`docs/dev_setup.md` names
  each one); bare `python` may be another install and fails breach imports
  with a misleading ModuleNotFoundError.
- **pytest: always `pytest tests -q`**, never bare `pytest` from repo root.
- C++/CUDA: `cpp/` via CMake; per-machine scripts `cpp/build_*.bat`, on Linux
  `cpp/build_linux.sh` (both modules; `containers/breach.def` builds with it).
  CPU build → `cpp/build/Release`; CUDA build → `cpp/build_cuda/`.
- Do NOT add machine specs to this file; they belong in the dev-setup docs.

## Iron rules

- **Determinism is a hard requirement.** Synced sim state is Q16.16 integer
  only: no floats, no libm transcendentals in the sim path; use
  `cpp/src/fixed_point.h` (incl. `atan2_q16`/`sin_q16`/`cos_q16`).
  Digest/golden gates guard this; goldens are re-baselined only deliberately,
  once per approved behavioral change, with written rationale. Render-layer
  code is exempt (dequantize at the render read, never write back).
- **Check the canonical-systems index below before building anything new.**
  If a system covers your need, use it; if it almost fits, extend it; never
  build a parallel copy. **Before editing a file, read the rule file whose
  region covers it** (the table below): a shell read (`cat`, `sed`) fires no
  rule, and a design that will land in a region touches no file, so it reads
  the rule file deliberately. New C++ sim TUs go on the `/fp:strict` list in
  `cpp/CMakeLists.txt`, always.
- **Never `git add -A`**: the tree carries untracked art, notes and
  prototypes on purpose. Stage explicit paths.
- **A green patch merges; Erik decides when he wants a test first.** A patch
  whose gates are green merges on green, gameplay and look changes included
  (re-baselines with written rationale as ever). A HUMAN-TEST (built, gated,
  pushed, Erik plays or looks before merge) happens only when Erik asks for
  one (Erik, 2026-11-13: "I'll decide from now on when I want a game test").
- **Credit the source**: any file implementing a published technique carries
  an author + paper citation in its header; archive the paper under
  `docs/papers/`.

## Rule regions (`.claude/rules/`)

| File | Covers |
|---|---|
| `sim-core.md` | facade, tick conductor, PhysicsRunner, recorder, entities |
| `fields-gas.md` | GameMap, FieldEdit, pumps, vents, trace transport + currency, O2 factor, drag, kits, materials |
| `energy-books.md` | gas energy field / seam / mirror, face flux, recovery, closure identity, temperature solver |
| `radiation-light.md` | radiation sweep, emissive table, extinction planes, integer reference, light accessor |
| `explosions-fire.md` | payloads, generic explosion, divergence, blast reach, tile deposit, fire, timed charge, heat source |
| `device-residency.md` | stamp kernel, batched resident engine, build id, Class P / T |
| `device-units.md` | unit store, device tables, unit library, event ring, request list, per-env serial applier, device topology seam |
| `units.md` | unit, schema, traits, turning, impulse, speeds, throttle, footprint samples, coupling, wind load |
| `horde.md` | horde field, noise, A* |
| `combat.md` | weapons, draws, exposure, poison, simultaneous attacks, credit, loadouts |
| `hand-verbs.md` | magazines, grenades, flash blinding, melee, flashlight, planted charge, verb gates |
| `sight.md` | vision / cover / line of fire, sight for many tiles |
| `match.md` | match config, spawn, controllers, replay, MatchSession, control, team picture, order, room graph, main.py |
| `rl-agent.md` | batched env, observation spec, full view, action spec, reward, training snapshot |
| `agents.md` | the trainer's side in `agents/`: network spec, choosing |
| `bots.md` | bot layers, position features, door bots, tournament |
| `render-core.md` | tone-map, look shots, wall seam, lighting, light curve, frame lights, inspector, ping-pong, BLAS, UI, blackbody |
| `render-effects.md` | gas medium, smoke relief + cast, tamed wind, FireVFX, particles, distortion, shock, muzzle flame, tracers, fire light |
| `render-units.md` | 3D units, held props, pose layer, static props |
| `audio.md` | sound layer, sound samples |
| `levels.md` | level data layer, generators, recipes, archetype grammar, layout stage, LevelSpec |
| `determinism-gates.md` | tests, field digest, test conventions, E2E driver, benches |
| `characters.md` | character kit, game-ready step, toon look, worker builder |

## Canonical systems — check before building anything new

One line per system built for reuse: what it is for and who must use it. Its
constraints are in its region's rule file. A new system adds ONE line here and
its constraints to the rule file of its region; a patch that changes a system
amends the rule file, never this line's length.

### Sim core

| System | Where | Rule |
|---|---|---|
| Simulation facade | `src/simulation/simulation.py` | The only tick entry + state mutation seam; outside code reads `get_state()`, writes `apply_action()` |
| Tick conductor | `Simulation.step()` numbered slots | A new system = one call line + ordering comment, never a logic block (god-file policy); one driver per law per tick |
| GameMap | `src/simulation/gamemap.py` | The single field store; topology changes only via `destroy_wall`/`seal_tiles` |
| FieldEdit | `src/simulation/field_edit.py` | The only way to write a continuous field, never `gmap.<field>[...] +=`; integer, quantized once at the door; on the resident path the device edit queue is its flush |
| Payload executor | `src/simulation/payloads.py` + `src/simulation/physics.py` | The one entry for gameplay events perturbing fields; bulk gas only through its `bulk_*` columns |
| Generic explosion | `[payloads.*] strength` + `[blast]` | A payload row with `strength > 0` is THE explosion; a new explosive is a row, never a new blast shape; a blast pushes only through its own wind |
| Divergence source | `src/simulation/blast_expansion.py` → `EOSSolver.div_source` | THE way to make air move outward with no mass, written only by the blast expansion |
| Blast reach | `blast_expansion.reach_tiles` | THE walk for what a blast touches; a wall stops it, a gap does not |
| Blast air read | `src/simulation/blast_air.py` | A charge reads P and O2 ONCE over the tiles it reached, after its wall damage and before its edits land |
| Fixed-total tile deposit | `field_edit.Region.TILES` + `payloads.deposit_over_tiles` | THE way to land a total over a walked set of tiles; never a twin of `deposit_heat` / `_claim_disc` |
| Starting a fire | `payloads.ignite_ring` · `combat.apply_temperature_ignition` · `src/debug_keys.py` | A fire is started by DELIVERING HEAT, never by writing `fire` alone; a blast has no fire-starting rule (Erik, 2026-11-06) |
| O2 pressure factor | `cpp/src/o2_pressure_factor.h` | THE pressure gate on O2 sustain, read by both law sites on both backends; a new O2 read goes through it |
| Gas pump primitives | `src/simulation/pump_system.py` + `GameMap.inject_gas_n`/`extract_gas_n` | THE per-tick gas mass flux path for entity actuators; never a parallel flux path or per-tick FieldEdits |
| Vent / duct system | `src/simulation/entities/vents.py` + `src/simulation/vent_system.py` | The only ambient-airflow mechanism; flux only via the gas-N `_vec` primitives at 9e(d) |
| Trace transport | `cpp/src/bulk_transport.cpp` (+ CUDA twin) + `cpp/src/trace_decay.h` | A trace plane moves ONLY by riding the bulk face flux + the per-tick face diffusion; every loss and gain booked |
| PhysicsRunner / PhysicsEngine | `src/simulation/physics_runner.py` / `cpp/src/physics_engine.*` | The only solver callers; new C++ orchestration lands in PhysicsEngine, not Python glue |
| Fixed-point kits | `cpp/src/fixed_point.h` + `cpp/src/cuda_fixedpoint_device.cuh` + `src/simulation/unit_fixed.py` | The only sim arithmetic, CPU and device; never re-derive a shift/round/reciprocal |
| Q16 boundary modules | `src/simulation/*_fixed.py` | All Python↔field conversion; never hardcode 65536 |
| Trace currency | `cpp/src/fixed_point.h` `TRACE_FINE_BITS` + `src/simulation/gas_fixed.py` | Trace planes are int32 in the 2^24 trace currency; every read/write converts ONCE through the kit, never an inline shift |
| Config | `config.toml` via `CFG` (`config.py`), F5 reload | All tunables; solver params bound in PhysicsRunner only |
| Material / gas tables | `src/simulation/materials.py` / `src/simulation/gases.py` | New material/gas = a table row, never a hardcoded id or per-material if |
| Filter table | `config.toml` `[filters.*]` | A filter is a table row; ducts reference it by name, never a per-gas if |
| Temperature scale | `src/temperature_scale.py` | The single T_game→Kelvin map for bake, render, readouts, tools |
| Coupling table | `src/simulation/exchange.py` | A physics→unit coupling is one row, not plumbing; a push is an impulse; felt temperature is ONE function |
| Wind load on a body | `GameMap.footprint_wind_load` + `exchange.apply_wind_force` + `unit_impulse.add_wind` | THE read of what the air does to a unit; its effect is a force on the impulse law, never a second reader |
| Footprint samples builder | `movement.footprint_samples` | THE `FootprintSamples` builder; every `default_speed` caller in `src/` uses it |
| Recorder | `src/simulation/recorder.py` | Frozen .npz contract, extended additively; not the replay path |
| Entity system | `src/simulation/entities/` + `src/simulation/signal_bus.py` + `src/simulation/sensor_accessor.py` | Registry-driven; one serializer (`entities/serialize.py`); sensors read only via the accessor |
| Prop tile stamps | `src/simulation/prop_system.py::stamp_prop_tiles` | A prop's blocking + fuel are ONE `[materials.*]` row stamped at load, never a per-prop if or a new per-tile flag; props never tick; look fields never synced |
| Level data layer | `level_lib.py` (write) / `level_loader.py` (read) | One writer ever; every tool is a client; never hand-write level.toml |
| Interior drag | `cpp/src/eos_solver.cpp` kick loops + `cpp/src/cuda_kick_compression.cu` | Momentum drag lives only in the staged drag block; extend its stages, never a parallel damping site |
| Gas energy field | `GameMap.gas_energy` | THE conserved truth for gas thermal energy (int64); `refresh_gas_energy` is the level-load initialiser only |
| Gas energy seam | `GameMap.gas_energy_*` + `cpp/src/gas_energy.h` | EVERY change of gas N or gas heat goes through the seam and books itself |
| Gas temperature is a mirror | `GameMap.seed_gas_temperature` / `EOSSolver::step` step 7 | NOTHING writes a gas cell's `temperature` directly, tests and tools included |
| Face-flux energy step | `cpp/src/eos_solver.cpp` step 6 | Gas-gas energy exchange is a per-face flux evaluated once, never a per-cell source term |
| Gas energy recovery | `cpp/src/eos_solver.cpp` step 7 | The ONE place gas `temperature` is derived from `gas_energy`, with rails |
| Energy closure identity | the solvers' `e_*_sum` counters + `GameMap.gas_energy_seam_net()` | Every energy writer adds a counter AND a term in one of the four groups; nothing relaxes to ambient |
| Temperature solver | `cpp/src/temperature_solver.*` + `cpp/src/cuda_temperature.cu` | Solids' temperature is derived here alone; gas temperature is the energy field's mirror; the Pass-1 clamp lives here |
| Radiation sweep | `cpp/src/radiation_sweep.*` + `cpp/src/cuda_radiation_sweep.*` | THE radiative transport, heat and light: one traversal per ordinate, every payload a channel on it, never a second traversal |
| Emissive table | `cpp/src/emissive_table.*` (owner `engine.emissive`) | THE E°(T) map, its inverse and the clamp's ceiling; one owner; L° checked in beside it |
| Extinction planes | `GameMap.heat_atten_q` / `dyn_heat_atten_q` + `src/simulation/optics_fixed.py` | Every optical interaction the sweep sees is a Q16 coefficient on these planes; dynamic stamps are MAX |
| Integer reference | `docs/ray_engine_v2_scheme_study_2026-09-13/sweep_ref_q.py` | THE executable spec of the sweep's integer arithmetic; any change lands THERE FIRST |
| Light accessor | `src/simulation/light_field.py` | THE light read for renderer, rules and RL; never a sweep plane or a producer |
| Stamp kernel | `cpp/src/cuda_stamp.cu` + `GameMap.take_stamp_rows` | The resident tick builds slot 6's planes on the device; the CPU stamp stays THE oracle |
| Batched resident engine | `cpp/src/cuda_batch.cuh` + `cpp/src/cuda_resident_tick.*` + `src/simulation/resident_batch.py` | Slot 7 of N homogeneous envs in ONE device chain; the played resident tick is N = 1 of it, never a second copy |
| Unit store | `src/simulation/unit_store.py` + generated `cpp/src/unit_store_layout.h` + `cpp/src/cuda_unit_store.{h,cu}` | THE device form of the units, one `(N, U, W)` int64 array built only through `pack` / `unpack`; not part of the played tick (Class T) |
| Device tables | `src/simulation/unit_tables.py` + generated `cpp/src/unit_tables_layout.h` | Every table value a unit kernel reads is ONE row, read from the reference's own object; never a law constant as a kernel argument |
| Device unit library | `cpp/src/cuda_unit_lib.cuh` + `cpp/src/cuda_unit_damage.cuh` | THE `__device__` functions unit kernels share, incl. the one device `apply_packet`; never a copy in a slot TU |
| Event ring | `src/simulation/unit_events.py` + `cpp/src/cuda_unit_events.*` | What a device unit kernel emits, as records in the reference's emission order |
| Request list | `src/simulation/unit_requests.py` + `UnitRecordList` | What a device unit kernel does that is not a row write, as records in the reference's order |
| Per-env serial applier | `src/simulation/applier.py` + generated `cpp/src/applier_layout.h` + `cpp/src/cuda_applier.{h,cu}` | THE device form of every order-dependent law that writes the map: one block per env walking its effect records in the reference's order; never a second walker |
| Device draws | `cpp/src/cuda_unit_draws.cuh` + `src/simulation/unit_draws.py` + generated `cpp/src/unit_draws_layout.h` | THE device `MatchDraws.at` / `.next` and the outcome maps; every device draw goes through it, its salt a generated constant |
| Device sight | `cpp/src/cuda_unit_sight.cuh` | THE device `GameMap.has_los` in both forms; every device kernel that asks a line goes through it, never its own walk |
| The march (device) | `cpp/src/cuda_unit_march.cuh` | THE device `BulletInFlight.advance`: slot 4's shooting and slot 2''s rounds both call it on a round-table row, never a copy |
| Noise list (device) | `unit_schema.NOISE_TABLE` on the object tables | THE device form of `GameMap.noise_log`: a device producer of a noise appends to it in the reference's emission order, never a second log |
| Object tables | `src/simulation/object_tables.py` + `unit_schema.OBJECT_TABLES` + generated `cpp/src/object_tables_layout.h` + `cpp/src/cuda_object_table.{h,cu}` | What flies and what is placed (charges first; grenades, rounds, blast sources, entity classes next) as device rows: a new table is a schema there, never a second table mechanism |
| Grenade and charge slots | `unit_schema.GRENADE_TABLE` + `cpp/src/cuda_unit_objects.cu` + `unit_kernels.table_appends` / `grenades` / `charges_tick` / `prune_grenades` | THE device form of the grenades in flight and the planted charges: slot 1b's throws and plants become table rows, slot 2 and `charges.tick` record every detonation as a PAYLOAD effect; never a second table or detonation path |
| Device topology seam | `cpp/src/cuda_topology.cuh` | `GameMap.on_tile_changed` / `destroy_wall` as device functions; every device caller that changes a tile's material calls them, never a copy |

### Determinism gates

| System | Where | Rule |
|---|---|---|
| Field digest | `tests/field_digest.py` + spec toml | Membership/dtype/currency change = version bump + all goldens regenerated in the same commit |
| GOLDEN_AGGREGATE | `tests/_xarch_perfield_digest.py` | THE golden; re-baseline once per approved change with written rationale |
| A/B lockstep harness | `tests/field_ab_harness.py` | The refactor gate; never prove a refactor with whole-grid means |
| CUDA harness | `tests/cuda_harness.py` | GPU tests via `run_cuda_script` subprocess; never import the CUDA .pyd into pytest; `cuda_*_check.py` + `test_*` wrapper |
| Build id | `cpp/gen_build_id.cmake` → `bp.build_id()` | The compiled module's provenance, generated at build time; `match_replay.build_id()` stays THE describe helper |
| Class P / Class T + the twin ledger | `tests/twin_manifest.toml` + `tools/twin_debt.py` + `tools/cuda_green.py` | Every device twin is a manifest row; a Class-P change merges only with `.cuda_green` == HEAD |
| Ingress lint + float ratchet | `tests/test_ingress_lint.py`, `tests/test_no_float_in_sim_tu.py` | The four number-doors + counts-only-go-down; exceptions carry `ingress-exempt:` with why |
| Test conventions | `tests/` | `test_*` collected, `_*` harness; properties, never snapshots; GL checks in a subprocess |
| Pinned match plays | `tests/_pinned_matches.py` (`watch`, `pinned_play`, `WHICH`) | The two pinned ship_demo matches (`bots`, `bots_v1`) are played ONCE per pytest process and shared (#205): a test that only WATCHES one of them hangs its observer on `watch(which, on_tick)`, never a fresh `run_match`; a test that changes the match in any way (monkeypatch, spec, seed, level, a mutated unit) plays its own. Observers never write to a view. Self-checking: the bots_v1 G8p pin and the ship_demo controller pin compare the shared play's outcome and per-tick unit stream with their pins |

### Render / UI

| System | Where | Rule |
|---|---|---|
| GameRenderer + WorldComposite | `renderer/game_renderer.py`, `renderer/world_composite.py` | The only renderer; world drawing inside `compose_world()` into the world RTs |
| One tone-map | `WorldComposite.resolve` + `shaders/world_resolve.fs` | Every world layer writes linear HDR; `resolve()` is the ONLY tone-map, gamma and dither; one named exception (the legacy heat layer) |
| Look shots | `tools/look_shots.toml` + `tools/lighting_demo.py --shots` | THE named shot set; every graphics patch renders it before and after and looks |
| Wall-destruction render seam | `GameMap.on_tile_changed` → `renderer/tile_art.py` | A topology change reaches the picture through this ONE seam, for every cause |
| LightingPass | `renderer/lighting.py` | A consumer of the light accessor; new lit passes sample `light_tex_a/b`, never a second raycast |
| Gas medium | `renderer/gas_medium.py` | ONE smoke tau and ONE per-gas optics binding that every smoke integrator reads |
| No BLAS in the frame path | `renderer/gas_medium.py::species_sum` | Per-frame / per-tick Python (`renderer/`, `src/simulation/`, `ui/`, `main.py`) never contracts through BLAS; load-time uses say `blas-exempt:` |
| Smoke relief | `renderer/gas_relief.py` + `shaders/gas_relief.fs` | THE smoke look; a new smoke term goes in this shader, never a second smoke pass |
| Billow noise bake | `renderer/noise_bake.py` | THE smoke billow noise; a new noise shape is a function here, never an inline bake |
| Smoke cast product | `GasReliefPass.render_product` + `lit3d._SMOKE_SHADOW_GLSL` | THE smoke's occlusion as ONE texture; every receiver reads it through the lit3d block |
| Tamed wind (render) | `renderer/gas_detail.py::tame_wind` | THE render-side wind, computed ONCE per frame and shared by every consumer |
| Blackbody | `renderer/blackbody.py` | The single ΔT→colour map and the source of the sweep's checked-in L° |
| Blackbody GPU LUT | `renderer/fire_vfx.py::blackbody_lut_data` | The GPU face of the one ramp; shaders sample it, never re-derive Planck |
| Ping-pong helper | `renderer/core.py::PingPong` (+ `GLDirect`, `DirectPasses`) | THE double-buffered RT helper for iterative GPU passes; never a second FBO pair |
| FireVFX pass | `renderer/fire_vfx.py` + `shaders/fire_glow.fs` | THE flame look and the fire's ONE draw; its numpy twin `renderer/fire_vfx_ref.py` owns the law |
| Particle primitive | `renderer/particles.py` | THE one particle system (embers, sparks, debris); consumers extend it, never build a second |
| Resolve distortion block | `shaders/world_resolve.fs::distortion_px` | Every screen-space distortion is a term of this block, never a second pass |
| Shock front | `renderer/shockwave.py` | THE blast's screen ring, a term of the distortion block; never from the pressure planes |
| Muzzle flame | `renderer/muzzle_flame.py` + `shaders/muzzle_flame.fs` | THE procedural fire at a gun's barrel; a new flash is a `kinds` row, never a sprite sheet |
| Tracers | `renderer/tracers.py` | THE picture of a round in flight, at the round's own speed; sprites of the particle draw |
| Fire room light (gain + flicker) | `renderer/fire_flicker.py` + `[render.fire_light]` | THE fire's room light: its own gain × a light-only flicker into the sweep's `emit_mult` |
| Frame lights | `renderer/frame_lights.py` + `src/level_lights.py` | THE one light assembly; never a second |
| Light curve | `renderer/light_curve.py` + `lit3d._LIGHT_CURVE_GLSL` | THE brightness map on the light, one GLSL copy; Erik's curve |
| Dequantize convention | per-field `*_fixed.dequantize_f32` at `upload_state` | Fresh float copies at the render read, never written back, never `/65536` inline |
| Tile inspector | `renderer/hover_readout.py::pack_hover_readout` | THE per-tile debug readout; tools and HUD read it, never a parallel field probe |
| 3D units | `renderer/unit_model_renderer.py` + `renderer/unit_looks.py` + `renderer/marine_shader.py` | No model/anim state ever on `Unit`; a new look is a table entry, never a second renderer |
| Held props | `renderer/held_props.py` + `[render.held_props]` | A held prop is a row on a hand socket, drawn through the units' one shader |
| Pose layer | `renderer/pose_layer.py` (+ `pose_clips.py`, `pose_skinner.py`) | THE only place a 3D unit's pose is composed; skeleton-agnostic; reactions and carriages are rows |
| Lit-3D seam | `renderer/lit3d.py` | THE shared light-field GLSL, `LightFieldCtx` and top-down `Camera3D`; a second copy is the bug |
| Prop generator | `renderer/propgen.py` | THE procedural prop / vegetation geometry; new flora = a generator fn + `PALETTES` row |
| Static props | `renderer/static_props.py` | The ONLY path drawing placed 3D props; no prop render state on a sim entity |
| UI split | `ui/model.py` (pure) / `ui/draw.py` (dumb) | Decisions in model as data; draw decides nothing |
| Sound layer | `audio/model.py` / `audio/device.py` / `audio/game.py` + `[sound]` | THE only audio path, presentation only; a sound is a cue row + a binding row; loops play from state |
| Sound samples | `tools/sound_audition.py --export` → `assets/sounds/` | Every sample is exported from a generated clip and sourced in `assets/sounds/SOURCES.md`; never hand-cut |

### Gameplay

| System | Where | Rule |
|---|---|---|
| Unit | `src/simulation/unit.py` | One class; `is_zombie` is a state and the only zombie check; subclassing is an anti-goal |
| Unit schema | `src/simulation/unit_schema.py` | THE one table of what a unit holds; a new unit attribute is a schema row in the same patch |
| Unit traits | `src/simulation/unit_traits.py` | How one unit differs from its class is a hook read through ONE accessor per value; never the raw row |
| Weapons | `src/simulation/weapons.py` + `combat.py` + `attack_resolver.py` + `damage.py` + `status.py` | A weapon is a data row; every hit is a DamagePacket; nothing writes the hp but `apply_packet` |
| Match draws | `src/simulation/draws.py` + `src/simulation/rng_salts.py` | EVERY random number is one keyed Philox block; a new draw site is a `Salt` member, never a stream |
| Exposure | `src/simulation/exposure.py` + `[damage.exposure]` | THE focus-fire law; a new debuff-under-fire is a column here |
| Poison | `src/simulation/poison.py` + `[damage.poison]` | Poison is ONE conserving law entered only through `poison.dose` |
| Magazines | `combat.mag_*` / `reload_begin` + `[weapons.*] mag_size` | Ammunition is magazines; a reload discards the one in the gun; the law is these functions only |
| Grenade kinds | `[loadouts.*] grenades` + `WeaponsTables.hand_grenade_rows` | A throwable is a hand-grenade `[ammo.*]` row appended to the family, never a verb |
| Flash blinding | `[payloads.*] blind_*` + `src/simulation/flash_blind.py` | A payload blinds by line of sight and facing through these columns; never a per-grenade check |
| Melee verb | `Simulation.melee_intent` + `combat.process_melee_intents` | A blow is a melee-archetype weapon row named by the loadout, never a branch |
| Flashlight bit | `Unit.flashlight_on` + `Simulation.set_flashlight` | One synced bit; nothing in the sim reads it; the light assembly draws from it |
| Simultaneous attacks | `src/simulation/simultaneous.py` | An attack slot decides against its start state and applies together; a new attack path records |
| Credited cause | `src/simulation/events.py` `Cause` + `damage.TickCredit` | Every hit and kill carries a typed cause and its credited unit; a new damage source adds a `Cause` row |
| Loadouts / suits | `src/simulation/loadouts.py` over `[loadouts.*]` / `[env_profiles.*]` | What a unit carries and survives are rows; never a per-unit literal |
| Control seams | `ruleset.py`, `src/control_source.py`, `orders.py`/`intents.py`, `action_registry.py`, `timeline.py` | Phase decisions via Ruleset; schemes are ControlSource subclasses; verbs are registry rows; WEGO is legacy |
| Direct control | `src/control_direct.py` + `src/control_gamepad.py` + `[controls.keys]` | THE one human direct-control scheme; a device is a half; keys are data |
| Camera mode | `[display] camera_mode` (`renderer/follow_camera.py`) | How the view follows a possessed unit is a mode VALUE, never camera code in a control source |
| Throttle | `Simulation.set_move_dir(..., throttle_q)` + `movement.move_speed_q16` | The commanded speed is a Q16 throttle between walk and full speed, formed in integers once |
| Speed table | `[speeds]` (m/s) + `src/simulation/speeds.py` | Every unit speed is a `[speeds]` row quantized once per map; the direction rule is the one place facing changes a speed |
| Turning law | `src/simulation/turning.py` | THE only writer of a unit's facing; a new reason to face is a target |
| Unit impulse | `src/simulation/unit_impulse.py` | THE way anything pushes a unit: an impulse through `add`, never a position or velocity write |
| Horde field | `src/simulation/horde.py` + `cpp/src/horde_field.cpp` | The horde paths only by the distance field; a new behaviour is a mask, goal set or descent rule |
| Noise | `src/simulation/noise.py` | What can be heard is a noise through `noise.emit`; hearing reads noises, never sources |
| Match config | `levels/<level>/match.toml` (`level_lib.write_match` / `simulation/match.py::parse_match`) | The ONLY way a match is parameterised; never argv scans or per-test dicts |
| Spawn rooms | `[[spawn_room]]` + `Simulation.spawn_match` | Where teams stand is the level's decision; `spawn_match` is the one match spawn |
| Match controller registry | `control_source.register_controller` / `create_match_controllers` | Every match driver is a registry row; in a match every unit has ONE driver and it is a controller |
| Bot layers | `src/control_bots.py` + `[bots]` | A new bot behaviour is a scored action, a play or a position feature, never a branch |
| Position features | `control_bots.FEATURES` | A tile is judged only through named yes/no features |
| Team picture | `src/simulation/team_picture.py` | What a team knows of the enemy lives in ONE place, updated once per tick; never a second picture |
| Match recording / replay | `src/match_replay.py` + `Simulation.attach_intent_log` | Replay is re-simulation through ONE intent tap; `main.py --replay` is the only playback path |
| MatchSession + the `policy` controller | `src/simulation/match_session.py` + `src/control_policy.py` | THE steppable match; a policy drives units only as this controller |
| Batched environment | `src/rl_env.py` | THE one env; a trainer never touches a Simulation or a kernel |
| Observation spec | `src/simulation/obs_spec_v1.py` + `EntityFieldAccessor.observe` | A named, versioned spec behind THE WALL; a new idea is a new spec |
| Full view | `obs_spec_v1.FullView` | The value network's privileged view, never a key of `obs` |
| Action spec | `src/simulation/action_spec_v1.py` + the decoder in `src/control_policy.py` | THE one layer from a network's choices to verbs; masks from the game's own gates |
| Reward profile | `src/simulation/reward_v1.py` + `[reward] profile` | A reward is a named table of term weights; the env returns groups, never their sum |
| Bots through the door | `src/control_bots_door.py` | The layered bots driven through action spec v1 (the imitation teacher); never a second table |
| Training snapshot | `rl_env.require_snapshot` + `write_run_manifest` | A training run's code is an annotated tag at HEAD in its own checkout |
| Network spec | `agents/spec.py` + `agents/network.py` + `agents/sizes.py` | A network is a named spec in `agents/`, meeting the game only through an observation and an action spec, at any team size; its record names all three and any other layout refuses it |
| Choosing | `agents/choosing.py` | The chances under the masks, the draw, the probability of given choices and which heads counted are ONE module, used when acting, learning from reward and copying |
| Stored decision and batch | `agents/batch.py` | Only rows of a learning team are stored; what is kept per decision is ONE layout, copied from the environment's arrays, time first |
| Trainer | `agents/trainer.py` + `agents/collector.py` + `agents/learner.py` + `agents/grading.py` + `agents/value_scale.py` | Touches the game only through `BatchedEnv`; a collector and a learner with one hand-over each way; every round runs the live check |
| Vision / engagements / cover | `src/simulation/vision.py`, `engagement.py`, `cover_system.py`, `line_of_fire.py` | One sight oracle (`has_los`, never `solid` alone); the match's sight rule decides who sees whom |
| Sight for many tiles | `cpp/src/sight_lines.cpp` + `vision.rays_clear` / `window_clear` | THE sight test in bulk, held equal to `has_los`; nothing walks its own lines |
| Room graph | `level_loader.RoomGraph` + `src/simulation/room_graph.py` | ONE map from tile to room and ONE list of openings; never a second |
| A* | `pathfinding.py::astar` | THE host pathfinder; never the horde's |
| Planted charge | `Simulation.plant_charge_intent` + `src/simulation/charges.py` | A planted charge is a `PlantedCharge` and nothing else; one driver per path |
| Verb gates | `src/simulation/verb_gates.py` | Whether a hand verb would act is ONE pure read, called by the verb and the masks; a new hand verb brings its gate |
| Order | `src/simulation/unit_orders.py` + `Simulation.order_intent` | An order is game state set through ONE verb; information only until rung 4 |
| Timed charge | `src/simulation/entities/charge.py` + `src/simulation/timed_charge_system.py` | A scripted, periodic or heat-triggered detonation is a `timed_charge` row; never a scheduler or a second payload path |
| Heat source | `src/simulation/entities/heat_source.py` + `src/simulation/heat_source_system.py` | A thing hot every tick is a `heat_source` entity; `GameMap.blast_log` is THE seam for reacting to blasts |
| main.py | `main.py` `_parse_*` flags | New launch flags are `_parse_*` helpers, never argv scans in subsystems |

### Tools

| System | Where | Rule |
|---|---|---|
| Map editor pattern | `tools/map_editor.py` shell + pure cores | Every editor feature = pure raylib-free core + thin shell |
| Undo | `tools/undo_log.py` | ONE transaction history; ops via the builder seam |
| Editor UI from registry | `tools/entity_editor_ui.py` | The registry IS the editor |
| Baker / airtight / play-scratch | `tools/bake_level_art.py`, `tools/level_airtight.py`, `tools/play_scratch.py` | The only art path (golden-gated) / the level lint / the only editor→game bridge |
| Level generators | `tools/gen_ship.py` (+ `tools/gen_explosion_studio.py`, the precedent) | A generated level is written through `level_lib`; `gen_ship` is THE ship generator, new generators replace its stage 1 |
| Recipe library + resolver | `levels/recipes/` + `tools/levelgen_recipes.py` | The family-B grammar's data; a recipe never carries a physics number |
| Archetype recipe + grammar | `levels/recipes/archetypes/` + `tools/ship_grammar.py` | A ship kind is one archetype file; the grammar is the only stage 1 of a recipe ship |
| Layout stage | `tools/ship_layout.py` | A recipe ship's geometry is produced ONLY here; never a search or a second layouter |
| LevelSpec | level.toml `[levelgen]` + `gen_ship.generate_from_spec` | A recipe ship records every generator input in one table and is a pure function of it |
| Declared-edge contact | `ship_layout.layout` openings loop | Every declared quotient edge is a shared wall of its opening; an undeclared touch is a blank wall |
| Axis parity | `ship_layout._opening_at` | An axis-axis opening is centred and even; never an airlock |
| Character kit | `art/characters/charkit/` | The scripted-Blender kit for the existing scripted characters; not the default for new ones |
| Game-ready step | `art/characters/charkit/gameready.py` + `rig.py` → `assets/models/` | The ONLY way a scripted character reaches the game |
| Toon look | `art/characters/charkit/toonlook.py` | THE toon look for high-res renders, render-time only |
| Worker builder | `art/characters/charkit/workwear.py` + `workerbuild.py` + `workergame.py` | A coverall-clad human is a set of tables on this builder, never a copy |
| Making a character | the `scripted-character` skill | THE procedure for a new character; load it before writing the brief |
| Tuning the smoke look | the `smoke-tuning` skill | Load it before changing any smoke dial |
| Tuning the light curve | the `light-curve` skill | Load it before changing `[render.light_curve]` |
| Tuning fire + smoke | the `fire-smoke-tuning` skill + `levels/fire_smoke_studio` + `tools/fire_smoke_probe.py` | Load it before changing `[render.blackbody]`, soot `decay`/`soot_yield` or the studio |
| Benches | `tools/bench_*.py`, `tools/bot_tournament.py`, `tools/lighting_demo.py --perf`, … | Reuse the existing instrument before writing a new one |
| GPU launch | `tools/run_on_cuda.py` (== `main.py --cuda`) | The only GPU launch path |
| Headless match runner | `tools/run_match.py` | THE one way a match runs to a result without a window; tests and benches call `run_match` |
| Bot tournament | `tools/bot_tournament.py` | THE measure of a bot change: tuning seeds, held-out report |
| E2E driver | `tools/e2e_drive.py` | THE headless end-to-end reproduction and frame profiler; bug fixes start here |

## Working style

- Big changes run as arcs: design doc (ending with its Systems section) →
  adversarial critique → patches with gates → CUDA lockstep → close (push,
  archive, tag; canon fold deferred).
- A patch that builds a reusable system adds its ONE index line above and its
  constraints to its region's rule file (a new rule file needs `paths:`, a row
  in the region table and files that exist). A patch that changes a system
  amends its rule file. As-built notes, gate results and dates go in the
  design doc and the commit, never in a rule; `tests/test_claude_rules_paths.py`
  fails on a named path that does not exist.
- Arc close also updates/closes the arc's GitHub issues and re-orders the
  pinned Roadmap issue (#46) if the sequencing changed.
- Repo hygiene: delete merged branches and finished worktrees (local +
  remote); never touch Erik's parked branches (check the Roadmap issue).
- Commit design docs to the branch BEFORE spawning worktree agents that
  depend on them: agents can't see your uncommitted working tree.
- **Project skills are refined by the runs that use them** (Erik, 2026-10-04;
  this repo only: breach is where workflows are still being explored). A
  skill in `.claude/skills/` records how the last good run went, not how every
  run must go:
  - every run that used a skill amends it in its own branch before merging
    (what was skipped, what failed, what was learned, a cost that moved; one
    dated line is enough);
  - a step is a MUST only with its incident written beside it, and is demoted
    the same way when the incident no longer applies;
  - any other step may be done differently when the brief or the report says
    which step and why; if the other route was better, it replaces the step;
  - a skill is marked PROVISIONAL until three runs in a row needed no
    amendment.
