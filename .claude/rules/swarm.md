---
paths:
  - "src/simulation/swarm.py"
  - "src/simulation/swarm_fixed.py"
  - "src/simulation/swarm_*.py"
  - "cpp/src/swarm*"
  - "tools/gen_swarm_layout.py"
  - "tests/*swarm*.py"
  - "src/simulation/entities/zones.py"
  - "renderer/swarm_dots.py"
  - "shaders/swarm_dots.*"
  - "tools/gen_larva_studio.py"
---

# Swarm: the store of swarm units, its digest, its facade, its table, its heading, the brood, the dot view

Fires on contact with these files. CLAUDE.md's index says what each system is
and who must use it; this file holds the constraints the code cannot say.
Erik's P6 ruling (2026-11-14, `docs/unit_design_2026-11-14.md` §7): TWO
stores (units and swarm units), ONE batch axis, ONE source of draws, and the
two sides meet only through FIELDS.

## Swarm store
- A swarm unit is NOT a `Unit`: it never touches `src/simulation/unit_schema.py`, `Unit.__slots__` or the device unit store (`src/simulation/unit_store.py`), and the unit side never gains swarm columns. Two stores (Erik, 2026-11-14).
- THE BOUNDARY (Erik, 2026-11-14): a swarm law reads map fields, never a unit column; a unit law sees the swarm only through a per-tile plane the swarm writes (the bite-dose plane, the count field), read by a coupling row. Never a record or queue of swarm events read by a unit law (a record channel is a design line first); a swarm kernel sees units only through the fields the unit side already writes (the slot-6 occupancy stamp).
- The batch axis is `ResidentBatch`'s: the store is per map, born in `GameMap._RESIDENT_SYNCED`; never a second N (Erik, 2026-11-14).
- A species is a `SWARM_SPECIES` entry + a species-table row + a kernel: never a class, never a per-species if-chain. A new column is a `ROSTER` row + a `SWARM_SECT` version bump, never a stray gmap array or a `DIGEST_FIELDS` row.
- Stable slots: never compaction, never atomic-ticket spawns (thread order would decide a slot). `unit_id` is identity; the slot is not, and a freed slot is the all-zero form.
- Every random number a swarm law draws is a keyed `sim.draws` block with its own `rng_salts.Salt` member (the Match draws row): never a second salt enum, a second key derivation from the seed or a second tick clock (Erik, 2026-11-14: one source of draws).
- A new `rng_salts.Salt` member changes in the same patch: `tools/gen_unit_layout.py --write` (the device draw layout carries every salt), `tools/gen_swarm_layout.py --write` where a swarm kernel reads it, and a rebuild of both modules. Incident (the #63 merge of main 824405bd, 2026-11-17): the swarm's TUMB / WALT salts without the regenerated `cpp/src/unit_draws_layout.h` made `test_cuda_unit_kit` refuse the stale fingerprint.
- A swarm law whose result depends on which unit came first (eating a corpse or a wall) is an effect KIND of the per-env serial applier (`src/simulation/applier.py`), never a second walker or a queue of its own; anything that changes a tile's material calls the device topology seam (`cpp/src/cuda_topology.cuh`).

## Swarm digest section
- Per-unit swarm state enters the digest and every snapshot ONLY through the presence-gated `__swarm__` carrier: absent on a swarm-free run, so no golden moves without a swarm. Every capture path writes it (`field_ab_harness.capture_snapshot`); a snapshot without the key reads as absent.
- The digest hashes the ONE encoder's bytes (`swarm_section_bytes`); the recorder stores the same columns raw, held equal by `tests/test_swarm_recorder.py`. Capacity and the species table never enter hashed bytes.

## Swarm facade
- The only Python-side writer of the store; the sim-side writers are the swarm kernels and their CPU twins. Never index `gmap.swarm_*` to write from gameplay, tools or tests.
- Spawn takes explicit Q16 values and draws nothing: callers own randomness. Overflow is `SwarmFull`, never a capacity clamp or a partial fill.
- Every host write sets the species' host-dirty flag in the same call, so the device upload sees it: a new writer that skips it leaves the device stale. The swarm step SETS it after every host-path launch (the host store moved) and the resident entry upload CONSUMES it (`physics_runner._resident_upload`).

## Swarm species table
- A swarm tunable is a `Field` row + a `SPECIES_RELOAD` class + a TOML value under `[swarm.<species>]`, validated strictly at load and on every reload (an unknown key or species is an error) and quantized once at ingress through `swarm_fixed`.
- Installed only at GameMap construction and by the Config reload seam; every consumer reads `gmap.swarm_species` and never reloads. An invalid reload is rejected loudly and the prior table stays in force. The table hash is provenance, never digested.
- Every value a swarm kernel reads is a row of this table, read from the validated table object (the Device tables discipline): never a law constant passed as a kernel argument.
- A temperature row is authored in KELVIN (`SPECIES_UNITS`) and converted through `temperature_scale` at the table's load only: never a game-dT literal, never a conversion in a reader. A cross-field invariant the kernel relies on is a `SPECIES_RULES` entry (never tuning taste), checked at load (hard error), at reload (rejected loudly) and at every launch on the exact launch ints.
- The field set is one tuple shared by every species (ACCEPTED GAP until a second species splits it into shared + per-species sets).

## Swarm step
- `PhysicsEngine::step_swarm` (host path) and `resident_swarm` (the resident tail's LAST stage, after temperature, before the books D2H) are THE only orchestration of swarm species: dormancy (nothing present in any env: no copy, no launch, no counter), the launch, the (species, env, slot) death compaction. Python only gathers (`swarm.engine_args`: arrays, pointers, env words; it transfers and launches nothing) and turns rows into events: never host-side swarm tick logic.
- The extent of every loop, launch and compaction is the HOST `high_water` in the launch block; the device copies of `high_water` / `next_unit_id` are never read, so they are neither uploaded nor downloaded.
- The draws reach the runner by `PhysicsRunner.bind_draws(sim.draws)` at reset, never a step argument (both tick paths must see them). The death events enter the tick as the first statement of `Simulation._step_post_physics`.
- A new swarm stage adds no third host sync: its records ride the tail's one books read, its counters are a region of the books arena.

## Swarm species laws
- A species is a `SWARM_SPECIES` entry + a species row + ONE FP_HD per-slot law (`cpp/src/swarm_<sp>.h`) + its entries + ONE `SPECIES_LAWS` line: the CPU twin, the per-call kernel and the resident core are loops around that one law, never a copy. There is no Python copy of a law: the C++ CPU entry is the reference.
- A law writes only its own slot's columns and its own record; every field and table row is read-only; no atomics in a law (the resident counters are order-free sums outside it). A law whose result depends on order is an applier effect kind (Swarm store), never a walk here.
- A law's draws are the draw table in its header: one block per (salt, unit) per tick, `draw_index` 0, a new purpose a new `Salt` member.
- The CPU loop counts every branch of a law (`bp.swarm_branch_tally`); every scene that runs the CPU twin declares the branches it must take, and a new branch arrives with a scene that takes it.
- Gate: `tests/cuda_swarm_check.py` (the three loops byte-identical with live-looking padding, the facade lockstep, the three sim paths) and `tests/cuda_batch_invariance_check.py` LEG B (swarm-bearing envs). The twin is `[twins.swarm_larva]`, Class P.

## Swarm shared kit
- Every species senses through `head_sweep` with a field reader and moves through `probe_move`: out of the grid and solid are BLOCKED, never clamped, never a reading; no sliding; a step stays under one tile (the no-tunnelling rule) so a move never skips a tile.
- No `std::` helper in an FP_HD function (nvcc runs without relaxed constexpr). The sim-law headers and the swarm TUs hold no `float` / `double` line, not even a comment (`tests/test_no_float_in_sim_tu.py`). `bindings.cpp` reaches the law only through out-of-line entries in strict TUs.

## Swarm kernel tables
- A new value a swarm kernel reads is ONE row of `swarm_tables.ROWS` (read from the species row, the launch context or `Salt`, with the range the law's arithmetic was proved for), then `tools/gen_swarm_layout.py --write`, commit, rebuild both modules. C++ reads offsets ONLY from the generated header; `bind_draws` refuses a module whose fingerprint differs. Never `tools/gen_unit_layout.py` (the unit store's).

## Swarm death events
- A death is emitted from the slot's PRE-ZERO values in (species, env, slot) order. It is not in the synced event digest: a sim-side consumer acts at the emission point, and if one ever reads the event the type joins the digest in that patch.

## Swarm heading law
- Every heading write, kit-derived or drawn, passes `wrap_heading_q16`. Its period is `2·PI_Q16`, deliberately NOT `philox32.TWO_PI_Q16` (the draw range): draw, then wrap; never unify the two constants.
- A device twin floor-mods in int64 and is identity-tested against the Python wrap over the whole input range.

## Brood entity
- Where swarm units start is a `swarm_brood` zone (zone_id, species, count) in the zone paint namespace; its randomness is the match draws (`Salt.SWARM_BROOD`), never a seed column (Erik: one source of draws).
- A brood keeps no runtime state and has no device table; it passes `unit_kernels.EntityExport` as a load constant. If a brood ever ticks (nests, births), it gets a table and an entity-row function in that patch.

## Brood spawn
- `swarm_brood.spawn_broods` is called ONCE, in `Simulation._reset_internal` after the door stamps with the clock at tick 0: every match path (the game, `tools/run_match.py`, `MatchSession`, each batched env's reset) constructs there. Never a second spawn site; a brood-free level draws nothing.
- The key is (seed, brood ordinal, k), draw `4 k + j`: a brood's scatter depends on nothing else in the level. Every check (no free tile, unknown species, `SwarmFull` over the broods' total) runs before the first write; the level lint (`tools/level_airtight.py`) prints the same `brood_problems` list.

## Swarm dot view
- Render-only: reads the store's host mirror and `SwarmUnitDiedEvent`, never the freed slot; nothing under `src/` reads it or `[debug] swarm_dots` (`tests/test_swarm_dots.py`). Switched by the config row, never a key.
- Linear HDR, unlit, premultiplied, drawn once after the water and before the floor smoke; its vertex data is numpy in one pass (no Python loop per unit in the frame path). Its flecks are render state (capped, sim-clock fade), never a decal layer: corpses are the world decal layer's when it exists (engine/17 section 9).
