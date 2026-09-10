# Cudaunit canon brief — constraints the design doc must honor

Read-only survey of `c:\Users\steen\projects\breach` (branch `fire-12`, 2026-09-06), anchored on
issue #63 and `docs/fauna_design_inputs_2026-09-07.md`. Cudaunits = GPU-resident simplified units
(larvae pilot: Q16.16 position, hp, 3-state thermal/foraging AI, deterministic RNG, thousands+).

Key framing found in the anchors themselves: `docs/rl_env_arc_proposal_2026-08-27.md` **§B3
"Device unit store"** is ALREADY the sketched cudaunit architecture — SoA `[N, max_units, ...]`
with a `valid` mask, Q16 where synced, fixed capacity, no mid-episode allocation, Python objects
as views or absent. The larvae design doc is effectively a pilot of B3 and should say so; §B5
even names "reactive zombie / fauna reflex" as the first agent tier.

---

## 1. GPU-residency canon (engine/02) — where cudaunit state lives

Source: `docs/architecture/engine/02_state_and_ownership.md`.

- **Rule of thumb (line ~32): "the GPU owns the world's *fields*; the CPU owns the world's
  *actors*."** Units/orders/RNG are named CPU-side "entity/logic state" (lines 29–35, 261–269).
  A cudaunit deliberately CROSSES this line — the design doc must amend engine/02 (or state the
  amendment) rather than silently violate it. The licensed escape hatch is the array shape: the
  canon's real argument is *branchy-serial-Python vs dense-array* — a fixed-capacity SoA per-unit
  array IS field-shaped state and fits the "arrays, not objects" doctrine (lines 42–75: no tile
  objects; arrays are what the sim, C++, and the neural net all want). B3 already makes this move.
- **GameMap stays the interface**: callers reach state only via `gmap.<field>` and must never
  assume where a field lives (lines 253–259). Physical home is per-field migratable; residency is
  internal to GameMap (`GameMap.enable_residency`, CuPy-owned buffers, `device_ptrs()` /
  `from_host()` / `to_host()` — see `src/simulation/physics_runner.py:1189` `_step_resident`).
  If cudaunit arrays live on GameMap, they inherit this contract for free; if they live in a new
  store, that store must replicate it (mirror + residency + digest access).
- **In-place discipline** (lines 179–183): per-tick buffers are allocated once and written in
  place, never reassigned — device views bind to memory once. Cudaunit SoA arrays must obey this.
- **Freshness split** (lines 285–290): render may read one-tick-stale; **the sim never does** —
  sim-side reactions read current-tick values on-GPU before staleness. A larva kernel reading
  temperature must read the device copy the resident EOS just wrote (physics_runner.py:1322–1336
  shows the precedent: trace_smoke reads device gas/wind because they're FRESHER than the mirror).
- **stamp_units** (lines 226–249): units are projected onto `dyn_permeability`, `dyn_wave_absorb`,
  `dyn_light_atten` once per tick via `unit.occupied_tiles()`. Decide: do larvae stamp? (Thousands
  of 1-tile soft bodies would be a per-tick scatter kernel, not a Python loop — §B2 already plans
  device `stamp_units`.)
- The old GPU-migration canon: mirror-currency invariant, targeted H2D lists, and the rule
  **"a DEFAULTED to_host() is forbidden in the resident tick"** (physics_runner.py:1338–1345).

## 2. §A resident-path habits (Erik-ruled 2026-08-27, binding on new code)

`docs/rl_env_arc_proposal_2026-08-27.md` §A (lines 58–84), near-verbatim:

1. **Every new `*_launch_resident` takes `(N, h, w)` even if N is always 1.** Index
   `env = i/(h*w); cell = i%(h*w)`; per-env scalars become `const T* d_scalar` arrays length N.
   For cudaunits: the unit table should be born `[N, max_units, ...]`, not `[max_units, ...]`.
2. **No new host-side tick logic.** Orchestration goes in `PhysicsEngine` (C++), not
   `physics_runner.py`.
3. **No new fields that only exist on the numpy mirror.** Any new field is allocated through the
   residency path (`GameMap.enable_residency` / CuPy-owned) from day one, even if a mirror copy
   exists for render.
4. **Prefer masked kernels over early-return-on-host.** Per-env `active` masks inside the kernel,
   not host gating. (Larvae dormancy/death = a `valid`/`state` mask, never a host skip.)
5. **Scratch keyed by `(N, h, w)`** when touched.
6. **Golden discipline unchanged** — N=1 batched must be bit-identical to today's resident path.

Also binding from the CLAUDE.md RL-batch row: these were adopted as a lean CLAUDE.md rule ahead of
arc #29 — they are not optional for the pilot.

## 3. What a unit/entity IS today — what the SoA table must mirror

**Unit** (`src/simulation/unit.py`, 480 lines):
- One class for all species; `is_zombie` is a *state*; **subclassing is an anti-goal** (CLAUDE.md
  G1). Larvae-as-units must not fork a Larva subclass on the CPU side; a cudaunit species is a
  *table*, mirroring the SpeciesDef/data-driven pattern.
- Position: float `x`/`y` in physics tiles, `tile_x = int(x)` (unit.py:124, 419–427). NOTE: CPU
  unit position is still float (memory says Q16 int-backed attrs are OWED with the stats
  redesign); `facing` is float but computed via the integer kit (`unit_fixed.atan2_rad`,
  unit.py:390 — the Q2-lift precedent for "quantize at boundary, integer trig, dequantize").
  A cudaunit position should be Q16.16 int from birth (issue #63 says so) — this is *ahead* of
  the CPU Unit, not inconsistent with it, and matches §B3 ("Q16 where synced").
- State roster relevant to a minimal per-unit row: `x, y, facing, current_hp, life_state
  (ALIVE|DEAD enum), team/faction_id, species_id, footprint offsets, statuses`. Everything
  temporarily true is a **status row** (mechanics/06), never a bespoke bool (G6) — a larva's
  3-state AI (thermal/foraging/flee?) is a *state machine field*, which is fine (like
  `life_state`), but burning/stunned-style conditions should map to a status concept or be
  deliberately scoped out.
- **Nothing writes `current_hp` directly** — every hit is a DamagePacket through
  `damage.apply_packet` (G5). A larva taking heat damage on-GPU either (a) routes through a
  device analogue with the same law, or (b) the doc rules an explicit exception; silent parallel
  damage math is the named anti-pattern.
- **No model/anim state ever lands on Unit** (R10, digest!). The generated larva body/crawl-wave
  is render-only (fauna doc already says: "generated body stays render-only, like the marine
  model").
- Synced-unit digest surface: `tests/field_ab_harness.py:58–73` — `UNIT_DIGEST_KEY =
  "__unit_state__"`, `SYNCED_UNIT_FIELDS = (tile_x, tile_y, x, y, current_hp, alive, life_state,
  faction, offsets, facing, ap, n_orders, statuses)`. Cudaunit synced state needs an equivalent
  hashed section (see §6).

**Entities** (`src/simulation/entities/`): units are explicitly **NOT entities** (design §3e,
zones.py:19, schema.py:70–76 — an entity_ref to a spawn unit is a hard error "until the stack-2
convergence"). So the cudaunit table is a *unit-system* artifact, not an entity class. But the
entity machinery sets the digest/serialization pattern to copy:
- **One serializer** (`entities/serialize.py`): section-local versioned byte stream
  (`ENTITY_SECT_V1`), ordinal order, `name|int64` rows, presence-gated fold into the tick digest
  (absence-transparent — entity-free digests byte-identical). A `CUDAUNIT_SECT_V1` (or an SoA
  bytes-dump keyed like a field) should follow this exact shape: presence-gated, section-local
  version, one serializer shared by digest + recorder.
- **Schema in code, numbers in TOML** (`registry.py`): tuning overlay may override numeric
  defaults only, hard error on unknown keys; `registry_content_hash()` is match-setup material
  like the seed. A larva species table (mass, hp, thermal thresholds, speeds) should be a
  materials.py/entities.toml-style table row, never hardcoded per-species ifs.
- **Sensors read only via `sensor_accessor.py`** — the frozen `Channel` enum
  (PRESSURE/SMOKE/WATER_DEPTH/TEMPERATURE/FIRE/O2/SOLID, sensor_accessor.py:43–66) is explicitly
  designed as "the future-GPU-gather seam": the resident backing is an `(n_sites × n_channels)`
  int32 gather buffer with frozen ordering. A larva kernel sampling fields per-unit is the
  *device-side sibling* of this gather — the doc should either reuse the channel vocabulary or
  say why not.

## 4. Tick conductor — Simulation.step() slots and where cudaunits go

`src/simulation/simulation.py::step` (line 1295). Slot order (comment-numbered):

- 1 fresh event buffer · 1b direct-control intents · 2 projectiles · 2b unit statuses/conditions
- 3 player movement · 4 shooting · 4b spray deposits · 5 zombie AI
- 6 re-stamp obstacles (`stamp_units`) · 6b FieldEdit queue flush (single RNG consumer for
  noise edits, simulation.py:1401)
- 7 physics (`physics_runner.step` — the whole EOS/smoke/water/fire/temperature chain, resident
  or per-call; line 1405)
- 9 fire burn-through → `destroy_wall` · 9b over-pressure wall failure
- 9c unit heat damage (coupling row, max-over-footprint of `heat`) · 9c2 wave impulse push +
  knockdown · 9c3 gas coupling rows (teargas/poison)
- 9d ignition from temperature · 9e entities split gate (sample/emit → logic nodes → input
  resolve → actuator sweep: pumps then **vents at 9e(d)**) · 9f retired
- ~9g temperature/heat consumer + end-of-tick heat clear, rad_net/sky-ledger lifetime
- 10 advance tick.

**Conductor rule** (CLAUDE.md): a new system = ONE call line + ordering comment in step(), never
a logic block. A cudaunit behavior step is one line.

**Natural slot**: the larva AI reads post-physics fields (temperature, O2, fire) and moves/eats —
that is the same family as 9c (unit heat damage) and 9d (ignition): **after slot 7 physics,
beside the 9c/9d consumers**. Two sub-decisions the doc must make:
- If larvae *consume* food/gas via a per-tick flux, that consumption must ride the gas-pump
  primitives at the **9e(d) actuator sweep** (`inject_gas_n`/`extract_gas_n` `_vec` — THE
  per-tick gas mass-flux path; incident rule: vent design v1 nearly rebuilt it) and the
  **gas energy seam** (moved mass carries source T_abs; every ΔN books a named
  `gas_energy_books` channel — a writer that skips the seam is invisible to the arc-#54 closure
  identity, the named failure mode).
- If the larva step runs *inside* the resident physics tick (a kernel launched by
  `PhysicsEngine`), the "one call line" is inside `run_substeps_resident`'s chain instead — but
  ordering vs the D2H at step 6 of `_step_resident` (physics_runner.py:1338) must be settled:
  before D2H (device reads fresh fields, results ride the same snapshot) is the §A-consistent
  choice.
- Heat damage ordering contract (9c comment, simulation.py:1463–1479): heat damage FIRST, then
  push — larva damage must slot into that documented order, not around it.

## 5. C++/CUDA orchestration — how a new kernel family lands

- **PhysicsRunner** (`src/simulation/physics_runner.py:147`) is the only caller of C++ solvers;
  `step()` (line 784) dispatches to `_step_resident` (line 1189) when residency is on. New C++
  orchestration lands in **PhysicsEngine** (`cpp/src/physics_engine.h:32` — owns solver
  instances: atmos, smoke, fire, temperature, raycaster, water, eos, combustion), never Python
  glue (§A rule 2, survey row 7). A `CudaUnitSolver`/larva kernel = a new member + a call in the
  engine's orchestration, exposed via `bindings.cpp` (backend flags row 12: "GPU is a flag,
  never a fork").
- Engine does NOT cache field pointers — solvers re-fetch numpy pointers per step
  (physics_engine.h:12–16); robust to GameMap realloc on reset. A device unit table must define
  its own lifetime/rebind story (CuPy-owned via enable_residency fits).
- **CPU-twin pattern**: every GPU kernel has a bit-identical CPU twin (deliberate pairs,
  survey line 114 — "CPU/GPU solver twins under bit-identity gates" are the one licensed
  parallel-implementation form). A larva kernel needs a CPU `.cpp` twin (or a Python/​numpy twin?
  — no precedent for that; the precedent is `.cpp` twin + `.cu`), and the `.cpp` TU **goes on
  the `/fp:strict` list** in `cpp/CMakeLists.txt:176–188` (`set_source_files_properties(...
  PROPERTIES COMPILE_OPTIONS "${BREACH_FP_STRICT}")` — currently 11 TUs). CUDA side compiles
  with `--fmad=false -prec-div=true -prec-sqrt=true -ftz=false` + host `/fp:strict`
  (CMakeLists:125–135).
- **Fixed-point kit** (`cpp/src/fixed_point.h`, all `FP_HD` = host+device): `quantize/
  dequantize(_f)`, `mul_q16`, `mul_wide`, `narrow`, `narrow_round(_signed)`, `mul128_shr`,
  `make_recip`/`recip_mul`, `deposit_dT_wide_q16`, `drag_dT_wide_q16`, `sat_add_q16`,
  `shr_round0`, `reciprocal_q16`, `scale_mag`, `work_fade_clamp01_q`, `floordiv_q`, `ceil_div`,
  `mean_sum`/`mean_round`, `sqrt_q16`, trig kit `sin_q16`/`cos_q16` (fixed_point.h:991/1005)
  and `atan2_q16` (line 1035, Q.30 internals, pinned 9e-6). Never re-derive a shift/round/
  reciprocal.
- **Device kit** (`cpp/src/cuda_fixedpoint_device.cuh` — include in every .cu):
  `mul128_shr_signed` (no device __int128 under MSVC-host nvcc — uses `__mul64hi`),
  `flux_to_dq_dev`, `heat_saturating_add_dev`, `recip_mul_dev`, `deposit_dT_wide_q16_dev`,
  `drag_dT_wide_q16_dev`, `reciprocal_q16_dev`, `scale_mag_dev`, `round_nearest_q_dev`,
  `shr_round0_dev`, `sqrt_q16_dev`.
- Known order-free patterns from the CUDA arc (memory): global reductions = int64 `atomicAdd`;
  scatter = integer `atomicAdd`, source-only deposit; variable-length GPU→host output =
  atomicAdd counter + index array with a SET-equal gate. A larva-deposits-into-tile write
  (eating, heat, permeability stamp) must be one of these order-free forms.

## 6. Determinism gates — how cudaunit state joins

- **Field digest** (`tests/field_digest.py`): frozen `(name, dtype)` tuple `DIGEST_FIELDS`
  (20 fields, spec v5) + `field_digest_spec.toml`. **Membership/shape/dtype change = bump
  `DIGEST_SPEC_VERSION` + regenerate EVERY golden in the same commit** (lines 40–71 document v2–v5
  precedents, incl. v5's new-int64-field bump for `gas_energy`). A new per-tile field (e.g. food
  density) that is synced state = a version bump.
- **Per-unit state does NOT go in DIGEST_FIELDS** — the precedent is the fold: `tick_digest`
  (field_digest.py:137) folds `__unit_state__` (field_ab_harness hash) and the presence-gated
  `__entity__`/`__signals__` sections, each with its own section-local version. **Cudaunit synced
  state should be a new absence-transparent section** (e.g. `CUDAUNIT_SECT_V1`): levels without
  larvae hash byte-identically, existing goldens survive, and the section carries its own version.
- **GOLDEN_AGGREGATE** (`tests/_xarch_perfield_digest.py`): 30-tick trajectory over the canonical
  A/B scenario; per-(field, tick) hashes + per-attribute unit sub-hashes so a divergence NAMES the
  field; re-baselined once per approved behavioral change with written rationale. Cross-machine
  workflow = run on both boxes, diff.
- **A/B lockstep harness** (`tests/field_ab_harness.py`): per-cell diff over `SIM_FIELDS` + unit
  digest — the refactor gate; never prove a refactor with whole-grid means.
- **CUDA gate pattern** (fire example, `tests/cuda_fire_check.py` + `test_cuda_p68_fire.py`
  wrapper, run via `tests/cuda_harness.py::run_cuda_script` in an isolated subprocess — never
  import the CUDA .pyd into pytest): PART 1 isolated fuzz with every branch FORCED (synthetic
  regimes, degenerate shapes, all-solid/all-vacuum) asserting byte-identity CPU vs GPU on all
  mutated fields + SET-equal variable-length outputs; PART 2 a 130-tick lockstep trajectory
  (two states, identical evolving inputs, per-tick byte-identity), asserted to actually drive
  the mechanism hard; PART 3 the CUDA build's CPU path still reproduces the committed golden.
  Prints `RESULT: PASS/FAIL`, exits 0/1. **A larva kernel gate should clone this 3-part shape**,
  with per-unit arrays byte-compared and any spawn/death list SET-equal. Also note the fire
  check's pass-structure argument (own-index writes / order-free counters ⇒ parallel schedule
  reproduces sequential CPU) — the larva kernel needs the same written argument, and unit-vs-unit
  interactions (collision, shared food cell) are where it gets hard.

## 7. Recorder (.npz ring) — adding per-unit arrays

`src/simulation/recorder.py`:
- Frozen .npz schema; `DEFAULT_FIELDS` extended **additively** (line 81). Per-tick unit arrays
  already exist as precedent: `unit_fx/unit_fy/unit_hp (int32), unit_alive (bool)` shaped
  `(capacity, n_units)` (lines 262–280) — cudaunit arrays would be the same idiom
  (`larva_x`, `larva_state`, …), presence-gated like `entity_state` (additive keys, dumps from
  larvae-free levels byte-identical).
- **Dtype-class lesson** (lines 88–94 + CLAUDE.md Recorder row): a new dtype CLASS is a contract
  extension with its own ring branch (`_BOOL_FIELDS`, `_INT64_FIELDS`), never a cast — float64
  is exact only to 2^53 and drops the LSBs energy gates assert on. Q16.16 per-unit ints should
  ride an int32 branch raw or be dequantized explicitly at the recorder boundary (the field
  precedent at line 190: int32 Q16.16 fields dequantize /65536 into the float32 ring, "render/
  debug only — not part of the synced state"). Pick one and say which.

## 8. Field access pattern for a larva kernel + the "room"/food question

- **Temperature + walls, the combustion pattern** (`cpp/src/combustion.h:522–552`): a kernel takes
  raw pointers `int32_t* temperature (h,w Q16.16)`, `int32_t* wall_hp`, `const bool* solid`,
  `const bool* flammable`, `const bool* is_vacuum`, per-material Q16.16 threshold planes
  (`ignition_temp` via the same table `apply_temperature_ignition` uses), plus nullable axis
  planes (`thermal_solid`, `heat_inv_shift`). Gates are integer compares against Q16.16-quantized
  table thresholds (`temperature[i] >= ign[i]`). A larva thermal-AI kernel reuses exactly this:
  gather own-tile/neighbour `temperature`, mask by `solid`, compare Q16 thresholds from a species
  table row.
- **GAS temperature is a mirror** (arc #54): a larva must READ `temperature` freely but any heat
  it deposits/removes goes through the temperature solver's deposit path or the gas energy seam —
  never a bare `temperature[...] =` (goes VACUOUS, the P-G1b incident).
- Walls for movement: `solid` (= `permeability <= 0`) is the flow/LoS boundary; **walkability is
  `mobility > 0`, a CPU-only query today** (engine/02 "never goes GPU-side"). A GPU larva that
  pathfinds needs a device-readable mobility/passability plane — a canon amendment, or reuse
  `solid` and accept larvae ignore furniture rules. The A* rule (one pathfinder,
  `pathfinding.py::astar`) doesn't fit thousands of GPU units; the doc must rule that cudaunit
  movement is reflex/gradient (no A*) or open the fork explicitly.
- Unit heat-damage precedent on CPU: coupling row `heat | max` over footprint
  (`src/simulation/exchange.py:693–704`, runs at 9c). Larva heat damage on-GPU is the device twin
  of this row — CLAUDE.md: "a physics→unit coupling is one row, not plumbing"; §B3 says the
  coupling rows "become kernels reading device fields". Frame larva damage as a device coupling
  row, not new physics.
- **Rooms: no room concept exists.** The closest is **zones**: `zones.npy` — a uint8 paint-id
  grid (0 = unpainted, ids 1..255) discovered by presence beside level.toml, with each zone an
  `[[entity]]` instance carrying `zone_id` (`src/simulation/entities/zones.py`,
  `level_lib.py:409` `write_zones_npy`; level_lib is the ONE writer, every tool a client).
  A coarse food-density layer has two lawful shapes: (a) a painted sidecar `.npy` (the zones/
  water precedent — "the file IS the field", level_lib.py:449) initializing (b) a real per-tile
  GameMap field (int32, digest-joining, residency-allocated per §A-3). If larvae EAT (mutate
  food), it must be (b): a synced field, digest version bump, seam-reviewed writers.
- Spawn path today: `[[spawn]]` rows in level.toml (level_lib.py:80,120–129 manage the block) →
  `Simulation.add_unit` (assigns stable integer id, re-applies deterministic predefined
  attributes, simulation.py:426–436). `breach_site.roster` (zones.py:51) already anticipates
  seeded in-zone spawn randomization "the ML variation hook" — larvae nests spawning broods fits
  that slot; §B4's pre-baked level template bank is the batched endpoint.

## 9. Canonical-survey rows the short table elides

`docs/canonical_systems_survey_2026-08-22.md`, relevant long-form rows:
- Row 12 **Backend flags + residency**: `bindings.cpp set_*_backend`; `set_residency`;
  `GameMap.enable_residency`; "GPU is a flag, never a fork; CuPy imported only inside
  `enable_residency()`" — a cudaunit must work (or degrade defined) on the CPU backend.
- Row 18/19: PhysicsRunner is the only C++ caller + home of the IMEX substep loop ("never
  construct a `bp.*Solver` yourself"); new C++ orchestration inside PhysicsEngine.
- Row 22 **Q16 boundary modules**: per-field `*_fixed.py` incl. `unit_fixed.py` — a cudaunit
  boundary module (`cudaunit_fixed.py`?) is the lawful place for Python↔table conversion; never
  hardcode 65536.
- Row 36 **Sensor accessor** = "the future-GPU-gather seam" (see §3).
- Row 30 coupling table, G5 DamagePacket, G6 status, G17 tick events ("transient → tick event;
  persistent → sim entity — never blur"): larva deaths/spawns visible to gameplay should emit
  tick events.
- Row R10/G1 as in §3. Row 16 CUDA gate harness as in §6.
- Line 114: deliberate CPU/GPU twin pairs are exempt from the parallel-implementation flags —
  the larva CPU twin is licensed by this clause.
- T7: "the registry IS the editor" — if larvae/nests get editor placement, it should ride the
  entity registry UI (nests are #60 props/entities) rather than new editor code.

## 10. Deterministic RNG precedent — what exists

- **`sim.rng`**: ONE seeded `np.random.Generator` on the Simulation
  (simulation.py:246–263), plumbed through known sites; consumers: FieldEdit flush noise
  (single RNG consumer, sorted-order draws — simulation.py:1401, field_edit.py:293),
  attack_resolver (PCG64 stream, "uniform is affine on the bitstream: exact",
  attack_resolver.py:38), payload/explosion noise. Cross-machine determinism holds because
  PCG64's integer bitstream is platform-stable and draws happen in a deterministic order.
- **The owed sampler does NOT exist yet**: `src/simulation/generation.py` (lines 11–21) documents
  the incident — `rng.multivariate_normal` went through LAPACK/BLAS and was cross-machine
  NON-deterministic (O(sigma) divergence, the 2026-07-04 Ada tick-0 `__unit_hp__` finding); the
  replacement is deterministic *predefined* (mean-vector, no RNG) attributes, with "seeded
  integer RNG stream → pure algebraic transform → Q16.16 snap" promised for the stats redesign.
- **No counter-based/per-thread device RNG exists anywhere** (no splitmix/xorshift/pcg/philox in
  cpp/ or .cu — repo grep). A per-larva GPU RNG is NEW canon: it must be (a) integer-only
  (no libm), (b) order-free (counter-based, keyed on `(seed, unit_id, tick)` — Philox/Squares/
  splitmix64-style hash, never a shared mutable stream, so thread schedule can't reorder draws),
  (c) FP_HD so the CPU twin draws identical bits, (d) sequenced independently of `sim.rng`
  (device draws must not desync host draw counts). This becomes a new canonical-kit entry
  (fixed_point.h or a sibling header) + a CLAUDE.md row — and it can double as the owed stats
  sampler's core, which the design doc should note.

---

## Open forks the design doc must settle

1. **Where the per-unit SoA arrays live**: on GameMap (inherits gmap.<field> contract, residency,
   from_host/to_host, digest access — but GameMap is "the single FIELD store" and these are
   actors) vs a new device unit store (§B3's framing — then it must replicate mirror/residency/
   serializer contracts and get its own canonical-systems row). Engine/02's field/actor split
   needs an explicit amendment either way.
2. **Digest membership**: new `CUDAUNIT_SECT_V1` presence-gated section (recommended by the A4
   precedent) vs folding into `__unit_state__`; who serializes (one serializer rule); registry/
   species-table hash as match-setup material.
3. **Tick slot**: host-conductor line at ~9c/9d (post-physics, before 9e actuators) vs inside the
   resident chain in PhysicsEngine; ordering vs the once-per-tick D2H; interaction with the 9c
   heat-damage-then-push contract.
4. **Spawn/death path**: fixed capacity + `valid` mask (§B3: "no allocation mid-episode") —
   who spawns (nest entity at 9e? breach_site-roster idiom? host enqueue into a pinned buffer
   like §B2's FieldEdit queue?); death = mask flip + tick event; SET-equal gates for spawn/death
   lists.
5. **Food representation**: painted sidecar .npy (init) + real synced int32 per-tile field
   (mutable, digest v-bump) vs read-only render/lore layer; if larvae eat gas/O2 instead, the
   flux MUST ride `inject_gas_n`/`extract_gas_n` + the gas energy seam with a named books channel.
6. **CPU-twin strategy**: full `.cpp` twin on the /fp:strict list with the 3-part gate (the only
   precedented shape) vs Python reference (no precedent, weaker); and whether the CPU game
   backend *runs* larvae via the twin (GPU-is-a-flag row says it must).
7. **World coupling scope for v1**: do larvae stamp `dyn_permeability`/`dyn_light_atten`/
   `dyn_wave_absorb` (thousands ⇒ needs the device stamp kernel, §B2), take DamagePacket-lawful
   damage, block movement, get seen by `vision.py`? Each is a canon seam; v1 should enumerate
   in/out explicitly.
8. **Movement law**: no A* on device — gradient/reflex movement (heat/O2/food fields as the
   potential) + which passability plane a kernel may read (`solid` vs a new device `mobility`
   plane, amending "walkability never goes GPU-side").
9. **Device RNG kit**: algorithm choice (counter-based, integer-only, FP_HD), key structure
   `(seed, unit_id, tick, draw_idx)`, relationship to `sim.rng` and to the owed stats sampler.
10. **Batch shape now**: born `[N, max_units, ...]` with N=1 (§A-1) — and whether `max_units` is
    config or level data.
11. **Recorder**: which per-larva arrays join the .npz (additive, presence-gated), raw-int vs
    dequantized ring.
12. **Segmented body**: sim = single-tile anchor (head) with render-only spine (the cheap,
    R10-consistent v1) vs per-segment sim state (per-segment fire/damage — multiplies digest
    and kernel cost); the fauna doc leaves this open, the design doc must close it.
