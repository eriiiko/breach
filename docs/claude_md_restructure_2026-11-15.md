# CLAUDE.md restructure — proposal (#206, 2026-11-15)

Applies the master file's "Where knowledge lives" (amended 2026-11-15: *Three
homes*, *What a rule may hold*, *Rules lifecycle*) to Breach. Proposal for
Erik's yes before any row is converted.

## Where we start

`CLAUDE.md`: 489 lines, 269 685 bytes (~65 000 tokens) in every session and
every subagent. The dated header is ~12 KB; 165 canonical-system rows are
~250 KB (Sim core 50, Determinism gates 8, Render / UI 34, Gameplay 51,
Tools 22). The biggest: Level generators 8.2 KB, Radiation sweep 8.0, Pose
layer 7.7, Observation spec 6.3, Unit store 5.5.

## The shape after

`CLAUDE.md` (~30 KB): purpose, where things live, environment, iron rules, a
short REGION TABLE (rule file → what it covers), then the index — today's five
sections and today's 165 row names, ONE line each: name | where | what it is
for and who must use it, plus any "never build a second X" an agent must know
before it opens a file. One line saying how to read the old long form
(`git show deade9d2:CLAUDE.md`).

`.claude/rules/<region>.md`, each with `paths:`: the constraints of the systems
in that region, one `## <row name>` section per system that has any (a row
whose whole content fits its index line gets no section).

`.claude/hooks/path_rules.py` + `.claude/settings.json` (the `path-rules-hook`
skill's canonical copies) so the rules also fire in worktrees;
`tests/test_path_rules_hook.py` (perfusio's, HOOK path adjusted);
`tests/test_claude_rules_paths.py` (every file path named in `CLAUDE.md` and
`.claude/rules/` exists; globs must match at least one file; untracked build
outputs on an explicit list with their reason). `.gitignore` gains
`!.claude/rules/`, `!.claude/hooks/`, `!.claude/settings.json`.

Because auto mode reads through the shell (which fires no rule), CLAUDE.md
says in one line: before editing a file, read the rule file whose region
covers it; a design that will land in a region reads it deliberately.

## Regions (19 files)

| file | `paths:` | rows (examples) |
|---|---|---|
| `sim-core.md` | `src/simulation/simulation.py`, `src/simulation/physics_runner.py`, `cpp/src/physics_engine.*`, `config.py`, `src/simulation/recorder.py`, `src/simulation/entities/**`, `src/simulation/sensor_accessor.py`, `src/simulation/signal_bus.py` | Simulation facade, Tick conductor, PhysicsRunner, Config, Recorder, Entity system |
| `fields-gas.md` | `src/simulation/{gamemap,field_edit,pump_system,vent_system,materials,gases,prop_system}.py`, `src/simulation/*_fixed.py`, `cpp/src/{bulk_transport,eos_solver,trace_decay,o2_pressure_factor}.*`, `cpp/src/cuda_{bulk_transport,kick_compression,edit_queue}.*`, `cpp/src/fixed_point.h`, `cpp/src/cuda_fixedpoint_device.cuh` | GameMap, FieldEdit, Gas pump primitives, Vent, Trace transport, Trace currency, O2 pressure factor, Interior drag, Fixed-point kits, Q16 boundary, Material / gas tables, Filter table, Prop tile stamps |
| `energy-books.md` | `cpp/src/{eos_solver,temperature_solver,combustion,gas_energy}.*`, `cpp/src/cuda_{temperature,combustion,eos_step}.*`, `src/simulation/gamemap.py`, `tests/test_*energy*.py`, `tests/_*_bench.py` | Gas energy field / seam / mirror, Face-flux step, Recovery, Energy closure identity, Temperature solver |
| `radiation-light.md` | `cpp/src/{radiation_sweep,emissive_table}.*`, `cpp/src/cuda_radiation_sweep.*`, `cpp/src/light_emission_table.inc`, `docs/ray_engine_v2_scheme_study_2026-09-13/sweep_ref_q*.py`, `src/simulation/{optics_fixed,light_field}.py`, `tools/gen_light_table.py` | Radiation sweep, Emissive table, Extinction planes, Integer reference, Light accessor |
| `explosions-fire.md` | `src/simulation/{payloads,physics,blast_expansion,blast_air,timed_charge_system,heat_source_system,combat}.py`, `src/simulation/entities/{charge,heat_source}.py`, `cpp/src/{combustion,fire_simulation}.*` | Payload executor, Generic explosion, Divergence source, Blast reach, Blast air read, Fixed-total tile deposit, Starting a fire, Timed charge, Heat source |
| `device-residency.md` | `cpp/src/cuda_*` (non-unit), `cpp/src/cuda_batch.cuh`, `cpp/CMakeLists.txt`, `cpp/gen_build_id.cmake`, `src/simulation/resident_batch.py`, `tests/cuda_*.py`, `tests/twin_manifest.toml`, `tools/{cuda_green,twin_manifest_gen,twin_debt}.py` | Stamp kernel, Batched resident engine, CUDA harness, Build id, Class P / Class T |
| `device-units.md` | `cpp/src/cuda_unit_*`, `cpp/src/unit_*_layout.h`, `src/simulation/unit_{store,tables,events,requests,kernels}.py`, `tests/cuda_unit_*.py`, `tools/gen_unit_layout.py` | Unit store, Device tables, Device unit library, Event ring, Request list |
| `units.md` | `src/simulation/{unit,unit_schema,unit_traits,turning,unit_impulse,movement,speeds,exchange,horde,noise,ai_zombie,unit_fixed}.py`, `cpp/src/horde_field.*`, `pathfinding.py` | Unit, Unit schema, Unit traits, Turning law, Unit impulse, Speed table, Throttle, Footprint samples, Coupling table, Wind load on a body, Horde field, Noise, A* |
| `combat.md` | `src/simulation/{weapons,combat,attack_resolver,damage,status,exposure,poison,simultaneous,events,loadouts,charges,flash_blind,verb_gates,draws,rng_salts,philox32,vision,engagement,cover_system,line_of_fire}.py`, `cpp/src/{sight_lines,philox32}.*` | Weapons, Match draws, Exposure, Poison, Magazines, Grenade kinds, Flash blinding, Melee verb, Flashlight bit, Simultaneous attacks, Credited cause, Loadouts, Planted charge, Verb gates, Vision, Sight for many tiles |
| `match.md` | `src/simulation/{match,match_session,ruleset,team_picture,unit_orders,room_graph}.py`, `src/{control_source,control_policy,control_direct,control_gamepad,control_match,control_zombies,match_replay}.py`, `main.py`, `tools/run_match.py`, `renderer/follow_camera.py` | Match config, Spawn rooms, Match controller registry, Match recording / replay, MatchSession, Control seams, Direct control, Camera mode, Team picture, Order, Room graph, main.py, Headless match runner |
| `rl-agent.md` | `src/rl_env.py`, `src/simulation/{obs_spec_v1,action_spec_v1,reward_v1}.py`, `tests/_obs_reference.py`, `tools/rl_smoke.py` | Batched environment, Observation spec, Full view, Action spec, Reward profile, Training snapshot |
| `bots.md` | `src/control_bots*.py`, `tools/bot_tournament.py`, `tools/bench_bot_matches.py` | Bot layers, Position features, Bots through the door, Bot tournament |
| `render-core.md` | `renderer/{game_renderer,world_composite,core,lighting,lit3d,frame_lights,light_curve,blackbody,hover_readout,tile_art}.py`, `shaders/{world_resolve,lighting}.fs`, `src/level_lights.py`, `ui/**`, `tools/look_shots*` | GameRenderer, One tone-map, Look shots, Wall seam, LightingPass, Light curve, Frame lights, Lit-3D, Dequantize, Tile inspector, Ping-pong, No BLAS, UI split, Blackbody (+ LUT) |
| `render-effects.md` | `renderer/{gas_*,noise_bake,advected_noise,fire_vfx*,fire_flicker,particles,heat_shimmer,shockwave,muzzle_flame,tracers}.py`, `shaders/{gas_*,fire_*,particle_*,muzzle_*}` | Gas medium, Smoke relief, Billow noise, Smoke cast product, Tamed wind, FireVFX, Particle primitive, Resolve distortion, Shock front, Muzzle flame, Tracers, Fire room light |
| `render-units.md` | `renderer/{unit_model_renderer,unit_looks,marine_shader,unit_shadow,held_props,pose_*,static_props,propgen}.py` | 3D units, Held props, Pose layer, Static props, Prop generator |
| `audio.md` | `audio/**`, `tools/sound_*.py`, `assets/sounds/**` | Sound layer, Sound samples |
| `levels.md` | `level_lib.py`, `level_loader.py`, `levels/**`, `tools/gen_*.py`, `tools/ship_*.py`, `tools/levelgen_recipes.py`, `tools/{map_editor,undo_log,entity_editor_ui,bake_level_art,level_airtight,play_scratch}.py` | Level data layer, Level generators, Recipe library, Archetype grammar, Layout stage, LevelSpec, Declared-edge contact, Axis parity, Map editor, Undo, Editor UI, Baker |
| `determinism-gates.md` | `tests/**`, `cpp/src/bindings.cpp` | Field digest, GOLDEN_AGGREGATE, A/B harness, Ingress lint, Test conventions, Benches |
| `characters.md` | `art/characters/**`, `assets/models/**` | Character kit, Game-ready step, Toon look, Worker builder, Making a character |

(Brace sets above are shorthand for this table; the files list each path
plainly. Exact globs are fixed during the conversion and gated by the
path-existence test. A file may match several rules: a unit kernel's TU fires
`device-units.md` and `device-residency.md`.)

## Three rows converted

### Radiation sweep (7 967 bytes → index 230 + rule ~2 900)

Index line:

> | Radiation sweep | `cpp/src/radiation_sweep.*` + `cpp/src/cuda_radiation_sweep.*` | THE radiative transport, heat and light: one traversal per ordinate, every payload a channel on it, never a second traversal; an arithmetic change lands in the Integer reference first |

`radiation-light.md`, section:

> ## Radiation sweep
> - Readers: the temperature fold reads `rad_net_sweep`, the Pass-1 clamp `rad_fluence`, units `rad_flux_sweep` (`src/simulation/exchange.py`). The old `gmap.rad_net` / `rad_flux` / `rad_amb` planes carry nothing: never read them (`rad_net` survives as a dead `step_tail` argument, recorded debt).
> - Exact-integer conservative (`Σ rad_net_sweep + Σ rad_flux_sweep + Σ rad_amb_sweep ≡ 0` in int64) and pure integer: the float ratchet pins it at 0.
> - The transport step is a parameter, never a fork. Light's is `[light] transport`: `"shear"` shipped (Erik, 2026-09-30), `"step"` the fallback for an absent key, anything else refused at load; bound once in PhysicsRunner.
> - Constants are checked in, never computed at load: the per-ordinate constants (`RadiationSweep::ordinate_table`, the CUDA twin's source too, never `std::cos`), the flux direction cosines, and L° (`cpp/src/light_emission_table.inc`, regenerated by `tools/gen_light_table.py --write`).
> - The heat currency is `E_FINE_BITS` finer than a heat count (#78): every reader of a sweep plane or an E° value converts ONCE through `fixedpoint::fine_heat_shr` (Python `sweep_ref_q.fine_heat_shr` / `optics_fixed.fine_heat_shr`; display `optics_fixed.dequantize_heat`), never an inline shift.
> - Order of change: the Integer reference, then `cpp/src/radiation_sweep.cpp`, then `cpp/src/cuda_radiation_sweep.cu`; `tests/cuda_radiation_sweep_check.py` holds GPU == CPU at tol 0 on every plane and counter. Device books use the unsigned-long-long `atomicAdd` idiom, never float; the `(N, h, w)` launch core counts and masks an illegal env, never throws.
> - Fleck damping is ONE exact kit floor-division per cell per tick on the excess only, alpha floored at 0 (Erik, 2026-09-16): never α = ½, never a re-derived reciprocal. The gas arm of L works in the temperature fold's own gas currency (`PhysicsEngine::gas_capacity_q()`), never a second one, and keys on `a_gas`.
> - `run()` zeroes its four int64 planes itself and the conductor never wipes them (they hold the last run for the tile inspector); a new output plane joins that zeroing, never a caller's `fill(0)`.
> - ONE invocation per path: `PhysicsEngine::run_sweep_` (step 2b, and `relight()`, whose heat outputs land in engine scratch, never a tick's plane); the resident tick's `breach_cuda::resident_tail`.
> - Light rides the same traversal as three RGB channels, only when requested (`PhysicsEngine.light_requested`, default off): Kirchhoff per channel with a dark ambient, a body emits nothing, no Fleck, never in the heat books; the per-channel books close exactly and heat is bit-identical with light on or off.
> - `RadiationSweep::cone_emission` is THE cone projection (the CUDA twin and the sky's door use it). The sweep's door on cones, sky and `emit_mult` is the only check: no light assembly re-checks it. `emit_mult` scales THERMAL light emission only (the fire's room light), never heat, never a cone; `build_emit_mult` is the one scatter.
> - `light_q` is NOT sim-pure until #81 (Light as Game State): its inputs are render state (`renderer/frame_lights.py` `EMITTER_INPUTS`). No game rule may read light before then.

Dropped: "LIVE since T5b", the P4 / P5b / P5c / P6a / P6b / P6d labels, the
formulas the code states (the Fleck expression, the light channel's absorb /
emit / glow, the cone row layout), the door limits and their headroom
arguments (`LIGHT_CONE_ROW_MAX`, `LIGHT_CONE_BUDGET`, `LIGHT_SKY_MAX`,
`LIGHT_EMIT_MULT_MAX`: all in `cpp/src/radiation_sweep.h` with their reasons,
#173's per-ordinate charge included), the shipped-scale measurement
(`f == 2²⁴`, held by its test), the test-file list beyond the one gate a change
must pass.

### Unit store (5 471 bytes → index 240 + rule ~2 300)

Index line:

> | Unit store | `src/simulation/unit_store.py` + generated `cpp/src/unit_store_layout.h` + `cpp/src/cuda_unit_store.{h,cu}` | THE device form of the units, one `(N, U, W)` int64 array built only through `pack` / `unpack`; not part of the played tick (Class T) |

`device-units.md`, section:

> ## Unit store
> - A unit is a ROW: the schema's group-S columns in schema order, then `valid`. Row r is the r-th unit of `sim.units` (id order, the laws' order); `(team, slot)` are columns, never the row order; a batch pads to the largest U with all-zero rows. Built only through `pack` / `unpack` over the schema's own `encode_unit` / `decode_into`.
> - C++ reads offsets ONLY from the generated header. A new, removed or re-widened schema column, a config row that moves a block's width included (a hand-grenade row, a status kind, a damage type, a hook), means `tools/gen_unit_layout.py --write` and commit in the same patch, then a rebuild of `cpp/build_cuda`.
> - NOT PART OF THE PLAYED TICK: no slot, `ResidentBatch` or `run_resident_batch` touches it, and the device never drives the game; the `device` backend uses it from P8b.
> - `UnitStoreDevice` is the buffer, its copies, the per-env counts and the input staging: NEVER a method per slot.
> - A new slot kernel is all of: its table values as Device tables rows; a free function in its family's TU `cpp/src/cuda_unit_<family>.cu` (declared in `cpp/src/cuda_unit_kernels.h`, on CMakeLists' CUDA list) over the Device unit library; a binding returning the three counts (coverage, faults, refused); a function in `src/simulation/unit_kernels.py`; a Class-T twin row; its shadow lockstep. A slot that emits goes through the Event ring, a slot that does what is not a row write through the Request list.
> - A unit kernel is born `(N, U)` (one thread per row, padding skipped) on `tstream`, reads no law constant from its arguments, copies through pinned staging, and is integer for integer the Python law: an arithmetic change lands in the law first.
> - A ported slot is gated by the SHADOW LOCKSTEP (`tests/cuda_unit_lockstep.py`), never a hand-rolled compare: rows at tol 0, env i of N == env i alone, the coverage counter == the rows the host changed, the hooked match equal to an unhooked one, scenes with non-neutral hooks for every hook the slot reads; adjacent ported slots also by the chained form. Which slots are ported is the twin manifest's to say.

Dropped: the list of built and chained slots (the manifest and git say it),
the INPUTS vocabulary and `run(store, inputs)` signature (code), the patch
labels, the test-file list beyond the gate.

### Blast air read (350 bytes → index 190, no rule section)

> | Blast air read | `src/simulation/blast_air.py` (`read_blast_air`) | A charge reads P and O2 ONCE, over the tiles it reached, after its wall damage and before its field edits land; never a second read of the blast's air |

Dropped: "`[blast_air] smoke_scale` is a look dial on top of the cited
`soot_yield`" — `config.toml` says exactly that beside the value, with Erik's
date.

## What leaves

1. The dated header log (12 KB): git log holds it.
2. Provenance: patch labels, "since X", "LIVE since", "as built", which patch
   added which member.
3. Inventories of what is built (ported slots, consumer numbering, rows of a
   table), which the code, the manifest or git already list.
4. Gate results, measurements, timings, farm readings, costs.
5. What the code says: function and attribute lists, signatures, numeric
   values, door limits and their headroom arguments, formulas — kept only where
   the rule IS about the number and the code does not carry its reason.
6. Test-file lists, beyond the one gate a change must pass.
7. Restatements of another row's rule (the hooks' accessor note repeated in
   nine rows lives once, in Unit traits; the impulse entry repeated in five
   lives once, in Unit impulse).
8. PLACEHOLDER notes, AFTER checking `config.toml` carries them beside the
   value (155 already do). Where one is missing I keep the note in the rule
   file rather than edit `config.toml`, which other sessions are tuning.

Kept, shortened: who must use a system, "never build a second X", what must
change together in one patch, Erik's rulings with their dates, accepted gaps,
debts, incidents (one line each with their date or issue).

Every row is accounted for in a ledger,
`docs/claude_md_restructure_ledger_2026-11-15.md`: row → index line, rule
file, what was dropped and where it lives instead.

## Expected size

- `CLAUDE.md` ~30 KB (index 165 × ~150 B ≈ 25 KB, iron rules + the rest ~5 KB):
  ~7 500 tokens in every session, from ~65 000.
- `.claude/rules/`: 19 files, ~70–90 KB in all, most 3–6 KB; a session pays
  only for the regions it touches.

## Also in this patch

- CLAUDE.md's "Where things live" and "Working style" rewritten for the new
  homes; the project skills that tell a run to grow a CLAUDE.md row amended
  (a new system = one index line + its region's rule section).
- `docs/canonical_systems_survey_2026-08-22.md` becomes a dated historical
  capture: a header note says it is superseded by the index + `.claude/rules/`
  and CLAUDE.md stops calling it the long form.

## As built (2026-11-15, after Erik's yes)

- 22 rule files, not 19: `combat.md` came out at 13.5 KB and `units.md` at
  12 KB, so they split by file region into `combat.md`, `hand-verbs.md`,
  `sight.md` and `units.md`, `horde.md` (touching `vision.py` now loads 2.5 KB,
  not 13.5).
- `CLAUDE.md` 269 685 → ~34 700 bytes (index 26 KB for 165 lines; the region
  table 2.6 KB). Above the 30 KB aim: the index lines average 160 bytes, and
  cutting further would drop "who must use it" clauses.
- `.claude/rules/` ~120 KB in all, 1.5–10 KB per file. The rows' 250 KB became
  26 KB of index + 107 KB of rule sections; 28 rows need no section.
- The ledger: `docs/claude_md_restructure_ledger_2026-11-15.md`.
- Tests: `tests/test_claude_rules_paths.py` (every named path exists; every
  `paths:` glob matches a tracked file; every rule section is an index row;
  the region table equals the rule files; each broken once by hand and seen
  red), `tests/test_path_rules_hook.py` (perfusio's, HOOK path adjusted).
- Probes: the hook's pipe test (injects once, then silent); `claude -p`
  rooted in the main checkout with the hook reading a worktree file
  (PRESENT, quoted the Unit store bullet); `claude -p` rooted in the worktree
  (native injection, quoted `horde.md`).
