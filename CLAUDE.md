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
| FieldEdit | `src/simulation/field_edit.py` | The only way to write a continuous field — never `gmap.<field>[...] +=` inline. The `gas_energy` row is a DUAL WRITE (store + mirror, through the seam) vetoed by the accountable set; the `gas` policy REFUSES a conservative slice (it is a float bridge, and bulk N is what `gas_energy` is denominated against — arc #54) |
| Payload executor | `src/simulation/payloads.py` + `physics.py::apply_explosion` | The one entry for gameplay events perturbing fields |
| Gas pump primitives | `src/simulation/pump_system.py` + `gamemap.py::inject_gas_n`/`extract_gas_n` (+ `_vec` variants) | THE per-tick gas mass flux path for entity actuators (slot 9e sweep, integer-native) — never a parallel flux path, never per-tick FieldEdits (incident: vent design v1 nearly rebuilt it, 2026-08-23). **ENERGY-AWARE (arc #54): the primitives carry `ΔN·T_abs` through the gas energy seam — extract removes it at the cell's own T_abs, inject credits it at the deposit's; the vent plenum ledger stays RELATIVE and converts here (`E = e + n·T_AMB`)** |
| Vent / duct system | `src/simulation/entities/vents.py` + `vent_system.py` | The only ambient-airflow mechanism; flux only via the gas-N `_vec` primitives at 9e(d); plenum ledger R3-counted (bulk pair + int64 raw-e), measured-delta booked, ENTITY_SECT-digested; plenum bulk = circulating credit ONLY (a future reserve is a separate row pair) |
| PhysicsRunner / PhysicsEngine | `src/simulation/physics_runner.py` / `cpp/src/physics_engine.*` | The only solver callers; new C++ orchestration lands in PhysicsEngine, not Python glue |
| Fixed-point kits | `cpp/src/fixed_point.h` + `cpp/src/cuda_fixedpoint_device.cuh` | The only sim arithmetic, CPU and device — never re-derive a shift/round/reciprocal |
| Q16 boundary modules | `src/simulation/*_fixed.py` (per field) | All Python↔field conversion; never hardcode 65536 |
| Config | `config.toml` via `CFG` (`config.py`), F5 reload | All tunables; solver params bound in PhysicsRunner only |
| Material / gas tables | `src/simulation/materials.py` / `gases.py` | New material/gas = a table row, never a hardcoded id or per-material if. **ID BUDGET FULL (arc #60): id 9 (foliage) numerically == the CSV SPACE_CODE — harmless, loader-intercepted, but an 11th material needs Erik's SPACE_CODE-migration ruling FIRST** |
| Filter table | `config.toml` `[filters.*]` (read in `vent_system.py`) | A filter is a table row (per-gas efficiency, validated [0,1]); ducts reference by name — never a hardcoded per-gas if |
| Temperature scale | `src/temperature_scale.py` | The single T_game→Kelvin map for bake, render, readouts, tools |
| Coupling table | `src/simulation/exchange.py` | A physics→unit coupling is one row, not plumbing |
| Recorder | `src/simulation/recorder.py` | Frozen .npz contract; extend `DEFAULT_FIELDS` additively — and a new DTYPE CLASS (int64) is a contract extension with its own ring branch, never a cast (a float64 ring is exact only to 2^53 and would drop the LSBs the energy gates assert on; arc #54 D13) |
| Entity system | `src/simulation/entities/` (schema, import-light) + runtimes in `simulation/` + `signal_bus.py` + `sensor_accessor.py` | Registry-driven; one serializer (`entities/serialize.py`); sensors read only via the accessor |
| Prop tile stamps | `src/simulation/prop_system.py::stamp_prop_tiles` + `entities/prop.py` | A prop's blocking + fuel are ONE `[materials.*]` row stamped load-time in the door-stamp `GameMap.__init__` slot (after doors) — never a per-prop if, never a new per-tile flag; props never tick; look fields are NON-synced kinds so art edits never move a digest (arc #60) |
| Level data layer | `level_lib.py` (write) / `level_loader.py` (read) | One writer ever — every tool is a client; never hand-write level.toml |
| Interior drag | `eos_solver.cpp` kick loops + `cuda_kick_compression.cu` | Momentum drag (the storm sink) lives only in the staged drag block; extend its stages (e.g. `k_drag`/`k_drag2`'s linear+quadratic terms), never add a parallel damping site — the `dyn_wave_absorb` chain and the B3c space-sponge band are separately-scoped neighbours in the same loop, NOT under this rule. **Drag L/Q deposits its removed KE as heat through the DERIVED `k_ke` (arc #54 §2.1); the absorb / sponge / cap stages EXPORT or DESTROY theirs, counted (D6) — never a heat dial** |
| Gas energy field | `gamemap.py::gas_energy` (+ `_gas_energy_accountable`) / `EOSSolver::step`'s `gas_energy` arg | THE conserved truth for gas thermal energy (int64, the exact unshifted `N_raw·T_abs_raw`, on the one canonical accountable set), and the CROSS-TICK truth since arc #54 P-G1b. `refresh_gas_energy` is the LEVEL-LOAD INITIALISER only (`reseed_gas_energy(sel)` for a scenario builder that just wrote bulk `gas` directly) — re-deriving it from the mirror mid-run DESTROYS state: `N·floordiv(E,N) ≤ E` drains up to N−1 raw counts per cell, plus any sub-count energy a seam wrote |
| Gas energy seam | `gamemap.py::gas_energy_move`/`_deposit`/`_mint`/`_retire`/`_born_at_ambient` + `cpp/src/gas_energy.h` | EVERY change of gas N or gas heat goes through the seam — MOVED mass carries its source's `T_abs`, MINTED mass (no gas donor) is born at ambient. Each call BOOKS its net effect on `Σ_accountable gas_energy` into a named `gas_energy_books` channel (`diag_*` = energy crossing the books' boundary, excluded from `gas_energy_seam_net()`); a writer that skips the seam is invisible to the closure identity, which is the whole failure mode arc #54 exists to close |
| Gas temperature is a mirror | `gamemap.py::seed_gas_temperature` / `gas_energy_refresh_cell` / `EOSSolver::step` step 7 | NOTHING writes a gas cell's `temperature` directly — tests, tools and benches included. The once-per-tick recovery derives it with the rails; a seam write refreshes the MIRROR ONLY (no rails, no write-back into E). A bare `temperature[...] =` now moves no books at all, so it does not fail loudly — it goes VACUOUS (incident: three test fixtures at P-G1b, one of which had stopped driving any wind) |
| Face-flux energy step | `eos_solver.cpp` step 6 (`face_flux` + the two-pass gather) | Energy exchange between gas cells is a per-face flux evaluated ONCE in canonical orientation (`i` = lower linear index) and applied with opposite signs to its two cells — never a per-cell source term. That is what makes `Σ_region ΔE ≡ 0` exact in int64, which is the whole of arc #54. Wall AND thermal_solid faces carry no face; vacuum/ambient-ring faces export to a counter |
| Gas energy recovery | `eos_solver.cpp` step 7 | The ONE place `temperature` is derived from `gas_energy` with RAILS, once per tick over the whole accountable set. Rails write `gas_energy` back ONLY when one binds (`e_rail_sum`) — an unconditional write-back drains `N−1` raw per cell per tick |
| Energy closure identity | `EOSSolver` / `TemperatureSolver` / `CombustionSolver` `e_*_sum` counters + `GameMap.gas_energy_seam_net()` | Every channel that moves `gas_energy` books itself, and since P-G1b the identity holds ACROSS WHOLE TICKS in int64: `Δ Σ_accountable gas_energy == EOS + thermal-solver-gas-side + combustion + seams`. A new writer adds a counter AND a term in ONE of those four groups — there is no fifth. Gated by `test_e1_hot_rail.py::test_no_transport_mint`, `test_gas_energy_tile_flip.py`, and the §6 benches (`_sealedbox_bisect`, `_fire`, `_vent`, `_blast`, `_quiet_books`). **P-G5 (arc #54, 2026-08-30)**: the ledger now spans gas AND solids — `TemperatureSolver.solid_energy_books_sum` (Σ thermal_mass_raw·T_raw, a snapshot) closes against `e_solid_deposit_sum` + `e_solid_cond_sum` + `e_thermostat_sum` + `CombustionSolver.e_comb_solid_heat_sum`, with the two-way ambient thermostat (Pass 3 relax-to-ambient, ERIK'S RULING: a deliberate modelling boundary, not a bug) counted by name; see `docs/gas_energy_thermostat_ledger_2026-08-30.md` and `test_thermostat_books.py` |
| Temperature solver | `cpp/src/temperature_solver.*` + `cuda_temperature.cu` | Heat deposits are Q16.16 saturating adds; **SOLIDS' temperature is derived here alone — GAS temperature is the energy field's mirror** (arc #54: the gas side's Pass-1 deposit and Pass-2 conduction sum go through the gas energy seam and the endpoint divide is deleted; `gas_energy == nullptr` keeps the whole pre-#54 T-form law bit-identical for direct-binding callers) |
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
| Lit-3D seam | `renderer/lit3d.py` | THE shared light-field GLSL, `LightFieldCtx`, and top-down `Camera3D` for everything 3D drawn in the world RT (marines, props, future 3D walls) — a second copy of any of the three is the bug; the marine byte-identity test (`test_lit3d_extraction.py`) gates it (arc #60) |
| Prop generator | `renderer/propgen.py` | THE procedural prop/vegetation geometry source (pure numpy, seeded, render-only float — never imports/reaches sim); new flora = a generator fn + `PALETTES` row here, never inline mesh code elsewhere (arc #60) |
| Static props | `renderer/static_props.py` | The ONLY path drawing placed 3D props: owned-memory model cache + `draw_props` in the units' shared `begin_mode_3d` pass; no prop render state ever lands on a sim entity (digest!). Prop assets live `assets/models/props/<pack>/` with a license file per pack; OBJ preferred (raylib 5.5 cgltf rejects 2020-era GLBs) (arc #60) |
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
