# Breach — project instructions

> **Last updated**: 2026-08-22 — rules-first restructure. Tasks moved to
> GitHub issues; canonical-systems inventory added.

Breach is an ML/RL project wearing a game's clothes: a deterministic GPU
physics engine as a state space, then agents trained in it for emergent
strategy. Every rule below serves that.

## Where things live

- **Rules** (invariants + canonical systems): this file. Keep it lean.
- **Tasks & ordering**: GitHub issues at `eriiiko/breach`. The pinned
  **Roadmap issue (#46)** holds the arc sequencing — work top-down unless
  Erik reorders. (`docs/TODO.md` + `docs/priority_ledger.md` are retired →
  `docs/archive/`.)
- **Narrative**: `docs/architecture/` is canon (live-edited; canon-fold
  currently DEFERRED by ruling — code + git history are the source of truth
  while systems still land weekly). Everything else in `docs/` is append-only
  capture (dated docs); arcs archive their working docs to `docs/archive/` at
  close.
- **Full canonical-systems survey** (this table's long form + known
  parallel-implementation flags): `docs/canonical_systems_survey_2026-08-22.md`.

## Environment (machine-agnostic — specifics in docs/dev_setup.md + docs/lenovo_dev_setup.md)

- **Python: always the conda env `data`** — bare `python` is a different
  install and fails breach imports with a misleading ModuleNotFoundError.
- **pytest: always `pytest tests -q`** — never bare `pytest` from repo root.
- C++/CUDA: `cpp/` via CMake; per-machine build scripts `cpp/build_cuda*.bat`.
  CPU build → `cpp/build/Release`; CUDA build → `cpp/build_cuda/`.
- Do NOT add machine specs to this file — they belong in the dev-setup docs.

## Iron rules

- **Determinism is a hard requirement.** Synced sim state is Q16.16 integer
  only: no floats, no libm transcendentals in the sim path — use
  `cpp/src/fixed_point.h` (incl. `atan2_q16`/`sin_q16`/`cos_q16`).
  Digest/golden gates guard this; goldens are re-baselined only deliberately,
  once per approved behavioral change, with written rationale. Render-layer
  code is exempt (dequantize at the render read, never write back).
- **Check the canonical-systems table below before building anything new.**
  If a system covers your need, use it; if it almost fits, extend it — never
  build a parallel copy. New C++ sim TUs go on the `/fp:strict` list in
  `cpp/CMakeLists.txt`, always.
- **Never `git add -A`** — the tree carries untracked art, notes, and
  prototypes on purpose. Stage explicit paths.
- **Feel-adjacent changes never auto-merge.** Anything touching game feel
  gets a HUMAN-TEST gate: built, gated, pushed, Erik plays it before merge.
  Mechanical digest-gated changes may auto-merge on green only when
  pre-authorized.
- **Credit the source**: any file implementing a published technique carries
  an author + paper citation in its header; archive the paper under
  `docs/papers/`.

## Canonical systems — check before building anything new

One line per system built for reuse. Long form + entry points:
`docs/canonical_systems_survey_2026-08-22.md`.

### Sim core

| System | Where | Rule |
|---|---|---|
| Simulation facade | `src/simulation/simulation.py` | The only tick entry + state mutation seam; outside code reads `get_state()`, writes `apply_action()` |
| Tick conductor | `Simulation.step()` numbered slots | A new system = one call line + ordering comment, never a logic block (god-file policy) |
| GameMap | `src/simulation/gamemap.py` | The single field store; topology changes only via `destroy_wall`/`seal_tiles` |
| FieldEdit | `src/simulation/field_edit.py` | The only way to write a continuous field — never `gmap.<field>[...] +=` inline |
| Payload executor | `src/simulation/payloads.py` + `physics.py::apply_explosion` | The one entry for gameplay events perturbing fields |
| Gas pump primitives | `src/simulation/pump_system.py` + `gamemap.py::inject_gas_n`/`extract_gas_n` (+ `_vec` variants) | THE per-tick gas mass flux path for entity actuators (slot 9e sweep, integer-native) — never a parallel flux path, never per-tick FieldEdits (incident: vent design v1 nearly rebuilt it, 2026-08-23) |
| Vent / duct system | `src/simulation/entities/vents.py` + `vent_system.py` | The only ambient-airflow mechanism; flux only via the gas-N `_vec` primitives at 9e(d); plenum ledger R3-counted (bulk pair + int64 raw-e), measured-delta booked, ENTITY_SECT-digested; plenum bulk = circulating credit ONLY (a future reserve is a separate row pair) |
| PhysicsRunner / PhysicsEngine | `src/simulation/physics_runner.py` / `cpp/src/physics_engine.*` | The only solver callers; new C++ orchestration lands in PhysicsEngine, not Python glue |
| Fixed-point kits | `cpp/src/fixed_point.h` + `cpp/src/cuda_fixedpoint_device.cuh` | The only sim arithmetic, CPU and device — never re-derive a shift/round/reciprocal |
| Q16 boundary modules | `src/simulation/*_fixed.py` (per field) | All Python↔field conversion; never hardcode 65536 |
| Config | `config.toml` via `CFG` (`config.py`), F5 reload | All tunables; solver params bound in PhysicsRunner only |
| Material / gas tables | `src/simulation/materials.py` / `gases.py` | New material/gas = a table row, never a hardcoded id or per-material if |
| Filter table | `config.toml` `[filters.*]` (read in `vent_system.py`) | A filter is a table row (per-gas efficiency, validated [0,1]); ducts reference by name — never a hardcoded per-gas if |
| Temperature scale | `src/temperature_scale.py` | The single T_game→Kelvin map for bake, render, readouts, tools |
| Coupling table | `src/simulation/exchange.py` | A physics→unit coupling is one row, not plumbing |
| Recorder | `src/simulation/recorder.py` | Frozen .npz contract; extend `DEFAULT_FIELDS` additively |
| Entity system | `src/simulation/entities/` (schema, import-light) + runtimes in `simulation/` + `signal_bus.py` + `sensor_accessor.py` | Registry-driven; one serializer (`entities/serialize.py`); sensors read only via the accessor |
| Level data layer | `level_lib.py` (write) / `level_loader.py` (read) | One writer ever — every tool is a client; never hand-write level.toml |
| Interior drag | `eos_solver.cpp` kick loops + `cuda_kick_compression.cu` | Momentum drag (the storm sink) lives only in the staged drag block; extend its stages (e.g. `k_drag`/`k_drag2`'s linear+quadratic terms), never add a parallel damping site — the `dyn_wave_absorb` chain and the B3c space-sponge band are separately-scoped neighbours in the same loop, NOT under this rule |
| RL-batch habits | `docs/rl_env_arc_proposal_2026-08-27.md` §A | New resident-path code follows §A (Erik's ruling 2026-08-27, ahead of arc #29): new `*_launch_resident` born `(N, h, w)`-shaped (N=1 today), no new host-side tick logic, no new mirror-only fields, per-env masks over host gating, scratch keyed `(N, h, w)` when touched |

### Determinism gates

| System | Where | Rule |
|---|---|---|
| Field digest | `tests/field_digest.py` + spec toml | Membership/dtype change = version bump + regenerate all goldens same commit |
| GOLDEN_AGGREGATE | `tests/_xarch_perfield_digest.py` | THE golden; re-baseline once per approved change with written rationale |
| A/B lockstep harness | `tests/field_ab_harness.py` | The refactor gate — never prove a refactor with whole-grid means |
| CUDA harness | `tests/cuda_harness.py` | GPU tests via `run_cuda_script` subprocess; never import the CUDA .pyd into pytest; `cuda_*_check.py` + `test_*` wrapper |
| Ingress lint + float ratchet | `tests/test_ingress_lint.py`, `test_no_float_in_sim_tu.py` | The four number-doors + counts-only-go-down; exceptions carry `ingress-exempt:` with why |
| Test conventions | `tests/` | `test_*` collected, `_*` harness, property gates preferred over goldens while systems land; feel regressions compare in real dequantized units |

### Render / UI

| System | Where | Rule |
|---|---|---|
| GameRenderer + WorldComposite | `renderer/game_renderer.py`, `world_composite.py` | The only renderer; world drawing inside `compose_world()` into the one world RT |
| LightingPass | `renderer/lighting.py` | New lit passes sample `light_tex_a/b` — never a second raycast |
| Gas medium | `renderer/gas_medium.py` (+ detail/noise/shader) | The only smoke/gas look |
| Blackbody | `renderer/blackbody.py` | The single ΔT→colour map |
| Frame lights | `renderer/frame_lights.py` | The only per-frame light-list assembly |
| Dequantize convention | per-field `*_fixed.dequantize_f32` at `upload_state` | Fresh float copies at the render read, never written back, never `/65536` inline |
| 3D units | `renderer/unit_model_renderer.py` + `marine_shader.py` | No model/anim state ever lands on `Unit` (digest!) |
| UI split | `ui/model.py` (pure) / `ui/draw.py` (dumb) | Decisions in model as data; draw decides nothing |

### Gameplay

| System | Where | Rule |
|---|---|---|
| Unit | `src/simulation/unit.py` | One class; `is_zombie` is a state; subclassing is an anti-goal |
| Weapons | `weapons.py` tables + `combat.py` archetypes + `attack_resolver.py` + `damage.py` + `status.py` | A weapon is a data row; every hit is a DamagePacket; nothing writes `current_hp` directly; temporary truths are status rows |
| Control seams | `ruleset.py`, `control_source.py`, `orders.py`/`intents.py`, `action_registry.py`, `timeline.py` | Phase decisions via Ruleset; schemes are ControlSource subclasses; verbs are registry rows; durations live in Timeline only |
| Vision / engagements / cover | `vision.py`, `engagement.py`, `cover_system.py` | One sight oracle; engagements share one shape; bullet fate is geometry |
| A* | `pathfinding.py::astar` | The only pathfinder |
| main.py | `main.py` `_parse_*` flags (`--level/--res/--control/--cuda/--resident/--debug`) | New launch flags are `_parse_*` helpers, never argv scans in subsystems |

### Tools

| System | Where | Rule |
|---|---|---|
| Map editor pattern | `tools/map_editor.py` shell + pure cores | Every editor feature = pure raylib-free core + thin shell |
| Undo | `tools/undo_log.py` | ONE transaction history; ops via the builder seam |
| Editor UI from registry | `tools/entity_editor_ui.py` | The registry IS the editor |
| Baker / airtight / play-scratch | `tools/bake_level_art.py`, `level_airtight.py`, `play_scratch.py` | The only art path (golden-gated) / the level lint / the only editor→game bridge |
| Benches | `tools/bench_*.py`, `storm_probe.py`, `analyze_blowup_dump.py`, … | Reuse the existing instrument before writing a new one |
| GPU launch | `tools/run_on_cuda.py` (== `main.py --cuda`) | The only GPU launch path |

## Working style

- Big changes run as arcs: design doc → adversarial critique → patches with
  gates → CUDA lockstep → close (push, archive, tag; canon fold deferred).
- Arc close also updates/closes the arc's GitHub issues and re-orders the
  pinned Roadmap issue (#46) if the sequencing changed.
- Repo hygiene: delete merged branches and finished worktrees (local +
  remote); never touch Erik's parked branches (check the Roadmap issue).
- Commit design docs to the branch BEFORE spawning worktree agents that
  depend on them — agents can't see your uncommitted working tree.
