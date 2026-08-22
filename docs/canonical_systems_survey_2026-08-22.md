# Canonical-systems survey — 2026-08-22

Two read-only survey agents walked the repo (sim/engine side + render/tools/
gameplay side) to build the canonical-systems inventory for the rules-first
CLAUDE.md restructure. This capture doc holds the FULL survey; CLAUDE.md
carries the compact table distilled from it. Chapter references locate design
intent, not as-built status (canon-fold is deferred by ruling).

## A. Simulation / engine

| # | System | Where (entry points) | Reuse rule |
|---|---|---|---|
| 1 | Simulation facade | `src/simulation/simulation.py` — `Simulation` (`step`, `apply_action`, `get_state`, `reset`, `edit`) | The only place game state is mutated and the only tick entry point — never step solvers or mutate `gmap` from main/renderer/tools |
| 2 | Tick conductor | `Simulation.step()` — numbered slots 1…10 | The slot sequence IS the tick documentation — a new system adds one call line + ordering comment, never a logic block (god-file policy) |
| 3 | GameMap | `src/simulation/gamemap.py` — field arrays, `destroy_wall`, `seal/unseal_tiles`, `stamp_units`, `inject/extract_gas_n` | The single field store — never allocate a parallel field array or mutate topology outside `destroy_wall`/`seal_tiles` |
| 4 | FieldEdit | `src/simulation/field_edit.py` — `EditQueue`, `FIELD_POLICY` | The only way to write a continuous physics field (enqueue → 6b flush) — never `gmap.smoke[...] +=` inline. Carve-out: structural topology |
| 5 | Discrete physics events | `src/simulation/physics.py` — `apply_explosion`; `payloads.py::execute_payload` | The one entry for "gameplay event perturbs fields" — combat never touches pressure/fire/smoke directly |
| 6 | PhysicsRunner | `src/simulation/physics_runner.py` — `step`/`_step_resident`, `set_residency`, config→solver binding | The only caller of the C++ solvers and home of the IMEX substep loop — never construct a `bp.*Solver` yourself |
| 7 | PhysicsEngine (C++) | `cpp/src/physics_engine.{h,cpp}` — owns all solvers; `run_substeps`, `run_substeps_resident` | New C++ orchestration lands inside this class, never as new Python glue |
| 8 | Fixed-point kit | `cpp/src/fixed_point.h` — `mul`, `recip_mul`, `sqrt_q16`, `atan2/sin/cos_q16`, `sat_add`, `mean_round`, … | The only source of sim arithmetic; no libm in synced state; trig deliberately has no Python fallback |
| 9 | CUDA device kit | `cpp/src/cuda_fixedpoint_device.cuh` — `mul128_shr_signed`, `sqrt_q16_dev`, `recip_mul_dev`, … | Every `.cu` uses these device twins — never re-derive shift/round/reciprocal inline in a kernel |
| 10 | Q16 boundary modules | `src/simulation/{water,wave,gas,fire,wall,atmosphere,unit}_fixed.py` | Python↔field conversion goes through the field's own module — never hardcode 65536 |
| 11 | Build (`breach_physics`) | `cpp/CMakeLists.txt` (`BREACH_CUDA`, `BREACH_FP_STRICT` list), `bindings.cpp`, `build_*.bat` | CPU → `cpp/build/Release`, CUDA → `cpp/build_cuda`. New sim TU MUST be added to the `/fp:strict` list |
| 12 | Backend flags + residency | `bindings.cpp` `set_*_backend`; `set_residency`; `GameMap.enable_residency`; `tools/run_on_cuda.py` | GPU is a flag, never a fork; CuPy imported only inside `enable_residency()` |
| 13 | Field digest | `tests/field_digest.py` (`DIGEST_SPEC_VERSION`, `DIGEST_FIELDS`) + `field_digest_spec.toml` | The one serialization contract; membership/dtype change = version bump + regenerate every golden in the same commit |
| 14 | A/B lockstep harness | `tests/field_ab_harness.py` — `capture_trajectory`, `assert_trajectories_match` | The per-cell per-tick refactor gate — never "prove" a refactor with whole-grid means |
| 15 | GOLDEN_AGGREGATE | `tests/_xarch_perfield_digest.py`; runner `tests/xarch_digest.py` | THE golden (~13 importers). Re-baseline once per approved behavioral change, rationale written |
| 16 | CUDA gate harness | `tests/cuda_harness.py` — `run_cuda_script`; pattern `cuda_*_check.py` + `test_cuda_*.py` wrapper | GPU tests run in an isolated subprocess — never import the CUDA .pyd into pytest |
| 17 | Ingress lint | `tests/test_ingress_lint.py` — AST scan of `src/simulation/`; `ingress-exempt:` pragma | The four number-doors are law; exceptions carry an inline why-safe comment |
| 18 | Float ratchet | `tests/test_no_float_in_sim_tu.py` — per-TU baselines | Counts only go down |
| 19 | EOS solver | `cpp/src/eos_solver.{h,cpp}` + `cuda_eos_*.cu`, `cuda_mg_solve.cu`, `cuda_kick_compression.cu` | Pressure is derived (`p* = C·N·T_abs`) — never store a second pressure state |
| 20 | Bulk/trace transport | `cpp/src/bulk_transport.*`, `smoke_dynamics.*`, `sky_exchange.*` + CUDA twins | Add a gas = a `[gases.*]` config row, never a bespoke advection loop |
| 21 | Fire + combustion | `cpp/src/fire_simulation.*`, `combustion.*`; ignition twin in `combat` | Spread is radiation→heat→ignition; there is no cellular spread rule |
| 22 | Temperature solver | `cpp/src/temperature_solver.*` + `cuda_temperature.cu` | Heat deposits are Q16.16 saturating adds; temperature derived here alone |
| 23 | Temperature scale | `src/temperature_scale.py` — `load(cfg)`/`from_toml`; `[physics.temperature_scale]` | The single T_game→Kelvin answer for bake, render, readouts, tools |
| 24 | Water solver | `cpp/src/water_solver.*` + `cuda_water.cu`; `[physics.water]` | Depth/flow are Q16.16 metres via `water_fixed`; `dx` lazy-bound from `tile_size_m` |
| 25 | Raycaster | `cpp/src/raycaster.*` + `cuda_raycaster.cu`; `PhysicsRunner.cast_fire_heat` | One DDA march, two consumers (light+heat) — never a second marcher |
| 26 | Material table | `src/simulation/materials.py` — `MAT_*`, `MaterialTable`; `[materials.*]` | New material = config row + id + CSV mapping; per-tile constants are column lookups, never per-material ifs |
| 27 | Gas table | `src/simulation/gases.py` — `GAS_*`, `GasTable` | The single source of gas indices |
| 28 | Config / hot-reload | `config.py` — `CFG`, `CFG.reload()` (F5); `config.toml` | All tunables in config.toml through CFG; solver params bound in PhysicsRunner only |
| 29 | Ambient derivation | `src/simulation/ambient.py` | Loader and GameMap must agree to the LSB, so it lives here once |
| 30 | Coupling table | `src/simulation/exchange.py` — `COUPLING_TABLE`, `apply_wave_push`, … | A physics→unit coupling is one row, not plumbing |
| 31 | Recorder | `src/simulation/recorder.py` — frozen `.npz` schema; `tools/analyze_blowup_dump.py` | The npz contract is frozen offline — extend `DEFAULT_FIELDS` additively, never rename |
| 32 | Entity schema | `src/simulation/entities/` — `REGISTRY`, import-light (stdlib only, CI-gated) | Entity classes are declarations; tuning numbers in `entities.toml`, never in code |
| 33 | Entity serializer | `entities/serialize.py` — digest + recorder share the same bytes recipe | Never a second serializer |
| 34 | SignalBus | `src/simulation/signal_bus.py` — dormant (`None`) when no wires | The only dataflow substrate between entities |
| 35 | Entity runtimes | `logic_nodes.py`, `sensor_system.py`, `door_system.py`, `pump_system.py` | Schema in `entities/`, runtime in `simulation/` — the import-light package never touches numpy/gmap |
| 36 | Sensor accessor | `src/simulation/sensor_accessor.py` — frozen `Channel` enum | Sensors read physics through this one accessor (the future-GPU-gather seam) |
| 37 | Level data layer | `level_lib.py` (write, atomic managed blocks) + `level_loader.py` (read) | One writer implementation ever — editor, baker, migrations, ML gen are all clients; never hand-write level.toml |
| 38 | Test bootstrap | `tests/conftest.py` — pinned 1.0 m/tile world autouse | Tests inherit the pinned world; rebind inside the test if needed |

## B. Renderer

| # | System | Where | Reuse rule |
|---|---|---|---|
| R1 | GameRenderer | `renderer/game_renderer.py` — frame protocol `upload_state → … → end_frame` | The only renderer; world drawing goes inside `compose_world()` into the single world RT |
| R2 | WorldComposite | `renderer/world_composite.py` | The only world-sized RT; a new layer is a draw call inside `compose()`, never a new RT |
| R3 | Camera + coords | `renderer/camera.py`, `renderer/coords.py` (`_tile/_wpx/_spx` suffix rule) | One `DrawTexturePro` blit; import the converters, never inline the multiply |
| R4 | LightingPass | `renderer/lighting.py` — `light_tex_a/b`; `shaders/lighting.*` | Any new lit pass samples the light textures — never runs its own raycast |
| R5 | Gas medium | `renderer/gas_medium.py` + `gas_detail.py` + `advected_noise.py` + `shaders/gas_medium.fs` | The only smoke/gas look (legacy pair only behind the F9 A/B toggle) |
| R6 | Blackbody | `renderer/blackbody.py` — `BlackbodyRamp`, `pack_emissive_rgba` | The single ΔT→colour map — never hand-roll a temperature ramp |
| R7 | Heat/cold overlays | `renderer/overlays.py::HeatFieldOverlay` + `renderer/cold_overlay.py` | Hot = additive, cold = premultiplied blue before it, same T toggle |
| R8 | Frame lights | `renderer/frame_lights.py::build_frame_light_sources` + `fire_lights.py` + `src/level_lights.py` | The ONLY per-frame light-list assembly (main + lighting_demo both call it) |
| R9 | WaterPass | `renderer/water.py` + `shaders/water.fs` | The shipped water look; `WaterFieldOverlay` is retired, don't extend |
| R10 | 3D units | `renderer/unit_model_renderer.py` + `marine_shader.py` | No model/anim state may ever land on `Unit` (digest!) |
| R11 | Hover readout | `renderer/hover_readout.py::pack_hover_readout` | The one gmap→display packer, headless-testable |
| R12 | Renderer core | `renderer/core.py` — window + RGBA16F texture helpers | The only pyray wrapper for uploads |
| R13 | Dequantize convention | per-field `*_fixed.dequantize_f32` at `upload_state` | Dequantize at the render read into fresh float copies never written back; never `/65536` inline |

## C. Gameplay / mechanics

| # | System | Where | Reuse rule |
|---|---|---|---|
| G1 | Unit | `src/simulation/unit.py` (+ stats/species/inventory/environment) | One Unit class; `is_zombie` is a state — subclassing is an anti-goal |
| G2 | Weapons tables | `src/simulation/weapons.py` — `[weapons/ammo/payloads.*]`, `get_tables`/`rebuild_tables`; archetypes in `combat.py` | A weapon is a row of data, not a system; `range_m` converts to tiles once at load |
| G3 | Payload executor | `src/simulation/payloads.py::execute_payload` | The one place a payload row becomes world effects |
| G4 | Attack resolver | `src/simulation/attack_resolver.py` | To-hit/crit decided here in front of the pipeline — never inline a roll in a weapon branch |
| G5 | DamagePacket | `src/simulation/damage.py` — packet → mitigation → quantized apply | Every damage source emits a packet; nothing writes `unit.current_hp` directly |
| G6 | Status system | `src/simulation/status.py` — `apply_status`, digest-hashed | Anything temporarily true of a unit is a status row, never a bespoke bool on Unit |
| G7 | Ruleset seam | `src/simulation/ruleset.py` — `TwoPhaseWEGO`, `ContinuousRealtime` | Every phase/AP decision routes through the Ruleset |
| G8 | ControlSource | `src/control_source.py` (+ input_handler/control_gamepad/control_onephase) | A control scheme is a subclass + factory name, writing only via `apply_action`/intents |
| G9 | Orders vs Intents | `orders.py` (WEGO, queued) / `intents.py` (per-tick, Q16.16) | Never subclass Order; never a float angle in an Intent |
| G10 | Action registry | `src/simulation/action_registry.py` | A player verb = a row; UI/timeline/executor all query this table |
| G11 | Timeline | `src/simulation/timeline.py` — `Plan` compiler + executor | The ONE place that knows how long things take |
| G12 | Charges | `src/simulation/charges.py` | Detonation is an absolute tick on the monotonic clock |
| G13 | Vision | `src/simulation/vision.py` | The single "what can this unit see" oracle |
| G14 | Engagements | `src/simulation/engagement.py` | Standing engagements share one shape; never a bespoke firing loop |
| G15 | Cover | `cover_system.py` + `entities/cover.py` | A bullet's fate is geometry; no cover bonus bookkeeping |
| G16 | A* | `pathfinding.py::astar` | The only pathfinder (temporal_astar currently unused — see flags) |
| G17 | Tick events | `src/simulation/events.py` | Transient → tick event; persistent → sim entity. Never blur |
| G18 | UI model/draw split | `ui/model.py` (pure, headless) / `ui/draw.py` (dumb raylib) | Every UI decision is answered in model.py as data; draw.py decides nothing |
| G19 | main.py entry | `main.py` — `_parse_*` helpers for `--level/--res/--control/--debug/--cuda/--resident` | A new launch flag is a `_parse_*` helper, never an ad-hoc argv scan in a subsystem |

## D. Tools + test conventions

| # | System | Where | Reuse rule |
|---|---|---|---|
| T1 | pytest invocation | `pytest tests -q`, conda env `data` | Collection pinned to tests/; never bare `pytest` |
| T2 | Test naming | `test_*.py` collected; `_*.py` harness/not collected; `cuda_*_check.py` + wrapper | New non-gate harness gets a leading underscore |
| T3 | Property gates | e.g. `test_velocity_clamp_property.py`, `test_destroy_wall_conserves_mass.py` | Prefer constructed-state property assertions over goldens while systems land |
| T4 | Feel regressions | `tests/_s*_feel_*.py` + committed `.pkl` baselines | capture/compare in REAL dequantized units |
| T5 | Map editor pattern | `tools/map_editor.py` shell + pure cores (`editor_layout`, `undo_log`, `entity_editor_ui`, …) | Every editor feature = a pure raylib-free core + a thin shell |
| T6 | Transaction-log undo | `tools/undo_log.py` | ONE global history; register ops via the builder seam, never a per-domain ring |
| T7 | Registry-driven editor UI | `tools/entity_editor_ui.py` | The registry IS the editor — never hand-author a palette row |
| T8 | Play-from-editor | `tools/play_scratch.py` | Reuses the level_lib writers; never a bespoke scratch serializer |
| T9 | Baker | `tools/bake_level_art.py` — golden-image gated | The only tilemap→art path; editor preview calls `bake_region` |
| T10 | Airtight lint | `tools/level_airtight.py` | Uses real GameMap decode so it can't disagree with the sim |
| T11 | Benches | `tools/bench_two_room.py`, `storm_probe.py`, `analyze_blowup_dump.py`, … | Reuse the existing instrument before writing a new one |
| T12 | Lighting demo | `tools/lighting_demo.py` + presets | The look-tuning harness (see flags — carries duplicated loop/HUD) |
| T13 | GPU launcher | `tools/run_on_cuda.py` | The only GPU launch path; hands off to `main.main()` |

## E. Parallel-implementation flags (candidate cleanup issues)

Deliberate-by-design pairs (CPU/GPU solver twins under bit-identity gates,
per-call vs resident orchestration, painted vs tiled editor) are NOT listed.

1. **`tests/water_q16.py` is a 7th copy** of the Q16 rounding helpers ("mirrors
   water_fixed") — a test-only duplicate of a sim-path rounding rule; should
   import `water_fixed`.
2. **`tools/run_on_cuda.py` duplicates `tests/cuda_harness.py`** CUDA-build
   discovery (`_cuda_dll_dir`/`_cuda_pyd`, pinned CUDA v12.4 path) — should
   import the harness.
3. **`TEMP_SCALE = 65536.0` re-declared in four render modules**
   (`blackbody.py`, `cold_overlay.py`, `fire_lights.py`, `hover_readout.py`,
   each with a "MUST match" comment; `gas_detail.py` has a fifth) — one import,
   not a comment. Related: confirm `blackbody.py`/`cold_overlay.py` read
   `src/temperature_scale.py` rather than local constants.
4. **`pathfinding.py` sits at repo root**, outside the ingress lint's
   `src/simulation/` scan, yet is on the sim path (3 sim importers). Also
   carries `temporal_astar` + `ReservationTable` with zero callers.
5. **`tools/lighting_demo.py` (83 KB) carries its own game loop, HUD, panel,
   level-arg parser and perf bench** mirroring `main.py` + GameRenderer.
   Highest-value convergence target.
6. **Six per-field Q16 boundary modules** are byte-identical ~60-line copies of
   one rounding rule (deliberate, per `wave_fixed`'s docstring — but a rounding
   fix lands in six places; consider a shared `_q16_base` with re-exports).
   Symptom: `temperature_scale.py` imports `quantize_scalar` from `gas_fixed`.
7. **Legacy render paths pending deletion once their A/Bs settle:** legacy
   smoke pair (`FieldOverlay`+`GlowOverlay` behind F9), retired
   `WaterFieldOverlay` (constructed "so nothing breaks"), `UnitSprites` vs the
   3D unit path (M toggle — retirement already on TODO).
8. **`AtmosphereSolver` retained-but-superseded** (asserted unreachable in
   `run_substeps`; EOS replaced it) + tombstoned `cuda_wave.cu`/
   `cuda_atmosphere.cu` bindings.
9. **Superseded-but-still-bound fire params** (`FIRE_T_EXT`, `FIRE_FUEL_REF`
   fallbacks vs per-tile planes) + tombstoned config keys.
10. **Dangling reference:** `levels/playground/level.toml` cites
    `tools/gen_playground_level.py` — the file does not exist; playground is
    not reproducible from a tool.
11. **Two "metres→tiles" quantizers** (`entities/door.py::quantize_span_tiles`
    vs `entities/cover.py::quantize_extent_tiles`) — one helper.
12. **Three TOML serializers** (`config.py` read, `level_lib.py` managed-block
    write, `lighting_demo.py` preset save/load).
13. **Two HUD layers with no shared primitives** (`GameRenderer.draw_panel`/
    `draw_debug_hud` dev HUD vs `ui/draw.py` game HUD) — acceptable for now,
    flagged for when UI grows.
14. **Coupling table registry vs execution** — `COUPLING_TABLE` is the formal
    registry but rows still execute at legacy tick positions (9c/9c2/9c3);
    the consolidated EXCHANGE-READ slot is explicitly deferred. Registry and
    schedule must currently be kept in agreement by hand.
