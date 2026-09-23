# Swarm units P2 — critique round (2026-09-24)

Critiques of `docs/swarm_P2_impl.md` v1 (commit `c13b964`, the
pre-critique version). One lens per critic, one critic at a time, each verdict
written here before the next spawns. Critics run independently: none sees
another's findings. Synthesis happens once, after the round, into v2.

## Lens 1 — Determinism & wire contracts (Opus) — **SOUND WITH FIXES**

Absence-transparency design is correct (a swarm-free run hashes zero extra
bytes), but one BLOCKER makes the build's "no new reds, zero golden edits"
gate unmeetable as written; 2 MAJOR, 10 MINOR.

### BLOCKER 1 — a tracked pre-P2 trajectory pickle goes red
After P2 `capture_trajectory` always writes `__swarm__`; `diff_trajectories`
checks `k not in sa or k not in sb` at `tests/field_ab_harness.py:391-393`
BEFORE any special-case (the doc places the swarm special-case "after
:397-399"). `tests/_pg5_base_trajectory_45050f3.pkl` is tracked, has no
`__swarm__` key, and `test_thermostat_books.py::test_thermostat_books_byte_identical_to_base`
(`:153-167`) diffs a fresh capture against it (passes today). Probe: adding a
`present=False` `__swarm__` to a fresh capture → 30 mismatches ("field
'__swarm__' present in only one run"). Same for any old-build vs new-build A/B
via `save_trajectory`/`load_trajectory` across the P2 boundary. Regenerating
the pickle = a golden edit requiring a build of 45050f3.
**Fix:** handle `SWARM_DIGEST_KEY` before the one-sided check; a missing key
reads as an absent carrier (pre-P2 snapshots are swarm-free by construction —
§5.2's own `tick_digest` argument); report only if the other side's carrier is
`present`. Add `test_thermostat_books.py` to §11's must-stay-green list.

### MAJOR 1 — `str((hw,))` in the section header is numpy-version-dependent
§5.3 grammar `column := col "|" dtype.str "|" str((hw,)) "\n" raw`; with `hw`
an `np.int32` scalar, NumPy 2 (NEP 51) gives `'(np.int32(5),)'`, NumPy 1.26
gives `'(5,)'` (1.26.4 is in cross-machine use, `docs/lenovo_dev_setup.md:147`).
Identical state → different hash across machines; contradicts §5.3's "integers
are ASCII decimal". `_field_bytes` is safe only because it uses `a.shape`
(Python ints, `field_digest.py:114`). Same hazard for the §6 `__swarm__.scalars`
xarch line if built by `repr` over numpy values.
**Fix:** header from `arr[e, :hw].shape` or `f"({int(hw)},)"`; rule: every
integer in any swarm hash payload is `int()`-converted; test: section bytes
from numpy-scalar inputs == from Python-int inputs, and `b"np."` never appears.

### MAJOR 2 — hot-reload swap runs inside the recorder path
`carrier()` runs the lazy `species` reload check (§3/§4.3) and the recorder
line calls `self.swarm.carrier()` every tick; the doc never pins that spawn
(and P3's H2D) read through `SwarmUnits.species` rather than
`gmap.swarm_species`. If spawn reads `gmap.swarm_species` directly: recorder-on
runs swap the table next tick, recorder-off runs (harness/RL/tests) keep the
old table forever → a debug tool decides a digested column's value. T18 hides
it by touching `sim.swarm.species` before spawning.
**Fix:** `carrier()` read-only (reads `gmap.swarm_species.hash`, never
reloads); rule: every consumer of species numbers reads through
`SwarmUnits.species`, the one reload point; rewrite T18 to spawn without
touching `.species`; add a test: recorder-on vs recorder-off with the same
mid-run `CFG.swarm` swap → identical section bytes.

### MINOR
3. **`import_state` accepts impossible (and digest-invisible) states** — only
   `n_env` and `hw ≤ capacity` are validated; e.g. all `hw == 0` with
   `next_unit_id = 500` imports, `present` is False so the fold is skipped,
   yet it shifts every later id / P3 Philox stream. Also accepts `nid ≤ 0`,
   live `unit_id ≥ nid`, duplicate ids, `valid ∉ {0,1}`, non-zero freed slots,
   non-canonical headings; and a present→absent import breaks §5.1's
   monotone-presence claim (stale ring rows). **Fix:** decoder enforces per env:
   `nid == FIRST_UNIT_ID` iff all `hw == 0`; valid ids unique and `< nid`; freed
   slots all-zero; `valid ∈ {0,1}`; canonical headings; note the import
   exception in §5.1; recorder writes every tick once rings exist.
4. **Integer-range guards off by one / missing** — `next_unit_id + k - 1 <=
   INT32_MAX` lets the counter reach `INT32_MAX + 1` (array `+=` wraps silently
   to `INT32_MIN`; a Python-int assignment raises after column writes →
   non-atomic refusal). `faction` unbounded (`col[:] = np.array([2**33+5])`
   stores 5). **Fix:** guard `next_unit_id + k <= INT32_MAX` in int64;
   range-check `faction` (and pre-wrap heading) before any write.
5. **Kit angle sources produce non-canonical headings** — heading law verified
   correct (period 2·PI_Q16 = 411774, idempotent, `wrap(INT32_MIN) = −82238`),
   but `atan2_q16(-1, -6553600) = -205887` (= −P, outside (−P, P]) and
   `philox32_angle_q16` spans 0..411774 (`philox32.h:136-142`), 411774 ≡ 0 →
   direction 0 double-weighted after wrap. **Fix:** §9 hand-off: every
   kit-derived heading passes `wrap_heading_q16`; record the double weight on 0
   as an accepted bias. NumPy 2 `PI_Q16 - h` on int32 overflows silently —
   §2.4's "in int64" is load-bearing (T15 catches it).
6. **Residency hand-off names only spawn** — `reclaim` and `import_state` are
   host writers too; a P3 uploading only after spawns lets a stale device
   `valid=1` resurrect a reclaimed unit at the slot-7 D2H. `_RESIDENT_SYNCED`
   is the default D2H set (`gamemap.py:1737-1743`) → a future defaulted
   `to_host()` would wipe host spawns (no caller today). **Fix:** every facade
   write sets a host-dirty flag P3's upload consumes; §9 says "any host store
   write"; caveat in the comment on the added line.
7. **`free_count()` best-effort pattern reintroduces capacity dependence** that
   §2.2's atomic refusal exists to prevent. **Fix:** synced callers never clamp
   by capacity; overflow = `SwarmFull` (authoring error); a P4 clamp uses an
   authored species-row cap.
8. **Name collisions** — `swarm_<sp>_<col>` vs `swarm_next_unit_id` (species
   `next`), future column `high_water` vs `swarm_<sp>_high_water`; silent
   aliasing + ambiguous `.npz` keys (section itself safe, `|` separators).
   **Fix:** import-time uniqueness assertion; species-name regex without `_`;
   reserved column names.
9. **`get_state()` omitted as a capture path** unlike the A4 precedent
   (`simulation.py:1124-1133`) the chapter says to copy. **Fix:** defaulted
   `swarm_state` field on `SimState` (additive), or state explicitly that
   `get_state` is not a capture path.
10. **Recorder provenance/truncation underspecified** — one 0-d
    `swarm_species_hash` cannot represent a hot reload inside the ring window;
    truncation at `hw > swarm_units_cap` is silent. **Fix:** per-tick hash (or
    reload tick); print once on first truncation.
11. **Harness locator crashes if it copies the entity fast path** — `if ea ==
    eb` (`field_ab_harness.py:353`) on ndarray-valued dicts raises "truth value
    ambiguous". **Fix:** compare carriers via `swarm_section_bytes` (+ `present`,
    `species_hash`).
12. **Shared `next_unit_id` weakens §5.3's §9b claim** — once species B spawns,
    the shared counter shifts species A's later ids. **Fix:** document as the
    cost of the shared-id decision.

### Claims verified correct
Fold transparency (`field_digest.py:137-176`, A4 precedent `:150-153`, no spec
bump); `GOLDEN_AGGREGATE` byte-identical (`test_w6_armory` canonical scenario,
xarch dumpers, B6 golden, gate_a scripts all key on `SIM_FIELDS`/`tick_digest`;
nothing pins snapshot key counts, `_RESIDENT_SYNCED` or spec toml); no leak of
new arrays (no GameMap attribute enumeration/deepcopy/pickle; `sim.gmap`
reassigned only at `simulation.py:262`); residency inert for swarm-free runs
(all transfers use explicit name lists — `physics_runner.py:1187, 1197, 1242,
1299`); no CPU/CUDA difference in P2; recorder additive + presence-gated,
no-cast honoured, ≈77 MiB right; section canonicality (species/ROSTER order,
`[0,hw)`, hw-0 omission, `dtype.str`, big-endian refusal) sound apart from
MAJOR 1; spawn slot selection deterministic; save round-trip exact; reload
writes no gmap array; no RNG draw added.

## Lens 2 — Systems reuse & scope (Opus) — **SOUND WITH FIXES**

No blocker; 3 MAJOR, 6 MINOR. Residency, digest and harness reuse are well
done; the weak spots are hot reload, save, and the species-table schema, each
sitting beside an existing mechanism the doc never confronts. (Critic did not
open lens 1.)

### MAJOR 1 — hot reload is a private lazy side mechanism with no stated apply point
§4.3 detects reload by object identity (`CFG.swarm is gmap.swarm_species.source`)
hidden in the `SwarmUnits.species` property and `carrier()`. Chapter 12's
canonical direction is explicit rebuild hooks on the Ctrl+R path
(`GameMap.reload_material_table`, gamemap.py:1652, "the correct hook", 12 §5/§7).
The recorder calls `self.swarm.carrier()` only when on (simulation.py:1580-1584);
the A/B harness calls free `swarm_carrier(sim.gmap)` (no check) and runs with
`enable_recorder=False` → taking a snapshot changes which table is in force. P3's
kernel reads `gmap.swarm_species` from PhysicsRunner (which holds `gmap`, not
`sim.swarm`): recorder-on → dials apply one tick late; recorder-off with no
spawns → never. §9's "H2D when `species.hash` changes" names no trigger. "First
genuinely live table" is overstated (exchange.py:302/:495, ai_zombie.py,
simulation.py:1436 read CFG live per call).
**Fix:** explicit `reload_species()` (keeps validation + loud rejection) called
from ONE seam, e.g. `Simulation.on_config_reload()`, invoked in
`debug_keys.handle_debug_keys` right after `CFG.reload()` (the only reload call
site, debug_keys.py:55, `sim` in hand); applies from the next tick; chapter 12's
material/physics re-binds can join the seam later. `carrier()`, `swarm_carrier`
and `spawn` become pure readers; tests call the seam. debug_keys.py:55 is
outside fire-12's hunks (22-33, 92+).

### MAJOR 2 — §8 builds a parallel save format; "one encoder for digest + recorder + save" is false
Chapter 02's save contract is arrays ("np.save the field set",
02_state_and_ownership.md:325-327; `get_state()` pickle-friendly, 02:397-398,
simulation.py:53-57) — fork 2a put the swarm on GameMap precisely so it rides
along. `export_state`/`import_state`/`SwarmSave`/`decode_swarm_section`/T21 have
no consumer and would canonize "save = SWARM_SECT bytes". The recorder (§7)
stores raw columns, not encoder bytes (only T22 re-encodes).
**Fix:** cut §8's build (decoder, import, export, T21); one-line hand-off to
chapter 02 (arrays = `SWARM_RESIDENT_NAMES` on gmap, saved with the field set,
`species_hash` as provenance); P2's only save duty = keep it picklable
(`SwarmSpeciesRow` a module-level class, not runtime-generated); reword the rule
("the digest hashes the one encoder's bytes; the recorder stores the same
columns raw; T22 gates them equal").

### MAJOR 3 — the species schema duplicates the entity registry's vocabulary
`SpeciesField(name, kind, reload, lo, hi, doc)` mirrors
`entities/schema.py::Field(name, kind, default, minimum, maximum, choices, doc)`;
the §4.2 validator mirrors `entities/registry.py:54-111 apply_tuning_overlay`
(unknown class/field = hard error, int kinds reject floats, inclusive bounds,
no schema in TOML); `.hash` mirrors `registry_content_hash` (registry.py:171-181)
with a different recipe; §15(a) never mentions it.
**Fix:** reuse `schema.Field` (stdlib-only) with `KIND_INT` for counts plus one
documented quantize-at-load kind for authored reals (the `KIND_LENGTH_M`
precedent), reload class as a separate mapping — or justify in §15(a); either
way flag the validator as the candidate shared strict-config-table helper
(chapter 12 records a "no schema validation" gap).

### MINOR
4. **Multi-env structure baked into hashed bytes** — `n_env|N` preamble and
   `env|e` tokens make env k of a batch never equal an N=1 run on the same seed
   — exactly the RL arc's B1 golden. **Fix:** per-env self-contained section, or
   encode env 0 only with `assert N_ENV == 1`, multi-env fold left to B1.
5. **Systems (a) omissions / wrong precedents** — the Deterministic RNG row
   (`unit_id` = Philox counter lane; `philox32.TWO_PI_Q16 = 411775` beside
   `HEADING_PERIOD_Q16 = 411774` — state they combine as draw-then-wrap so P3
   does not "unify" them; add 411774 to T15); `FACTION_NONE = -1` belongs in
   `factions.py`; one sentence distinguishing `simulation/species.py`
   (`SpeciesDef`); the wire-order tuple precedent is
   `entities/prop.py:32-35 PROP_MATERIAL_CHOICES` (append-only), not
   `WEAPON_ARCHETYPES` (a frozenset, weapons.py:45-51) — and P4's stdlib-only
   `swarm_brood` schema will duplicate + load-validate the species tuple (the
   prop_system route): state the hand-off in §9; state that P2 leaves `SimState`
   alone (the entity carrier also rides `SimState.entity_state`, simulation.py:1132).
6. **Draft-rule precision** — facade = "the only Python-side writer" (P3's C++
   twin reclaims host-side); swarm-store rule regains "a species is a table row +
   kernel, never a class / per-species if-chain" and "no atomic-ticket spawns";
   flag `N_ENV` as the codebase's first batch-axis constant (the one N until B1).
7. **Scope trims** — `SPECIES_RULES` empty → P3 adds it; cut
   `Simulation.spawn_swarm`/`reclaim_swarm` (no non-test caller; keep
   `sim.swarm`); defer the xarch additions to P3 (runners only capture the
   swarm-free canonical scenario, xarch_digest.py:58); `swarm_fixed.py` would be
   a 7th hand-copy of the rounding helpers (survey §E flag 6) — re-export from an
   existing boundary module, write only `PI_Q16`, the heading period and the wrap
   law.
8. **Merge-friendliness** — §10 omits CLAUDE.md (fire-12 edits rows at
   merge-base 67-91, 113-121; append after the Deterministic RNG row); do not fix
   the stale "F5 reload" Config row in P2 (conflicts with fire-12 line 76 —
   queue for after the merge); the cited `cool_shift` caveat (gamemap.py:170-179)
   is deleted on fire-12 — cite fire-12's `rad_*_sweep` resident planes instead.
9. **Where a Sonnet implementer would guess** — how `SwarmSpeciesRow` is built
   (must be picklable); whether spawn triggers the reload check (moot after
   MAJOR 1); T19's "prints once" (per reload or per value?); bool arrays at
   spawn; the read accessor — P4/P5/PhysicsRunner hold `gmap`, not `sim.swarm`,
   so make `swarm_column(gmap, sp, col)` a free function the facade delegates to.

### Reuse decisions verified correct
Residency (`_RESIDENT_SYNCED + SWARM_RESIDENT_NAMES` = fire-12's own host-written
resident-plane precedent; no defaulted `to_host()`/`from_host()` caller);
digest (presence-gated fold after the entity fold, no `DIGEST_FIELDS` row, no
spec bump, section-local version = the A4 precedent); encoder in `swarm.py` not
`serialize.py` (stdlib-only by rule; swarm units are not entities); A/B harness
(carrier, strict-carrier rule, locator, special-case key); recorder (additive,
presence-gated, dtypes verbatim); config (`[swarm.larva]` via CFG, `from_config`
at GameMap construction = material/gas precedent); `SwarmFull` = `SealBlocked`
precedent; no C++ in P2 (the philox twin precedent); no conductor /
physics_runner / CMake line; allocation at the end of `GameMap.__init__` clear
of fire-12; RL-batch §A rules 1 and 3 honoured.

## Synthesis — resolutions (orchestrator, 2026-09-24; binding on v2)

Both lenses are SOUND WITH FIXES; the core (absence-transparent digest,
residency, harness, recorder reuse) survives. Every finding is ACCEPTED; the
choices below settle each "or" and merge overlapping findings. None needs an
Erik ruling (all implement his rulings; none changes scope or a lasting name).

| # | Finding(s) | Resolution for v2 |
|---|---|---|
| R1 | L1-B1 | `diff_trajectories` handles `SWARM_DIGEST_KEY` BEFORE the one-sided-key check; a missing key = an absent carrier; mismatch only if the other side is `present`. `test_thermostat_books.py` joins §11's must-stay-green list; a test pins "pre-P2 trajectory (no key) == fresh swarm-free capture". |
| R2 | L1-M1 | Every integer in any swarm hash payload is `int()`-converted; header uses `f"({int(hw)},)"` (or `.shape`). Test: numpy-scalar inputs and Python-int inputs give identical bytes, and `b"np."` never appears. |
| R3 | L1-M2 + L2-M1 | ONE explicit reload seam: `Simulation.on_config_reload()`, called in `src/debug_keys.py` right after `CFG.reload()` (line 55); it calls `swarm.reload_species(gmap)` (strict validation; on rejection print loudly ONCE PER RELOAD EVENT and keep the old table). The seam is the only writer of `gmap.swarm_species` after init; `carrier()`, `swarm_carrier`, `spawn`, and every future consumer (P3 H2D) are pure readers of `gmap.swarm_species`. Applies from the next tick. Drop "first genuinely live table". Tests call the seam; add recorder-on vs recorder-off with the same mid-run reload → identical section bytes; T18 spawns without touching any species accessor first. |
| R4 | L2-M2 + L1-m3 | Cut §8's build (`export_state`, `import_state`, `SwarmSave`, `decode_swarm_section`, T21). One-line hand-off: save rides chapter 02's field-set contract (arrays on gmap; `species_hash` provenance). P2's save duty: the swarm arrays and `SwarmSpeciesRow` (a module-level frozen dataclass) are picklable — pinned by a test that pickles those objects (not all of GameMap). Keep L1-m3's invariant list as a pure reader `swarm.check_invariants(gmap)` (nid == FIRST_UNIT_ID iff all hw == 0; valid ids unique and < nid; freed slots all-zero; valid ∈ {0,1}; canonical headings; hw ≤ capacity), used by the lifecycle tests after every operation (P3's gates will reuse it). Reword the encoder rule: "the digest hashes the one encoder's bytes; the recorder stores the same columns raw; T22 gates them equal". |
| R5 | L2-M3 | Reuse `entities/schema.py::Field` as the species field descriptor. Extend schema.py with ONE kind, `KIND_REAL_Q16` ("authored real, quantized to Q16.16 at load" — the `KIND_LENGTH_M` precedent), unused by entities today. Reload class is a separate mapping in `swarm.py`. The validator stays in `swarm.py` mirroring `apply_tuning_overlay`'s strictness; §15(b) flags it as the candidate shared strict-config-table helper (chapter 12's gap) — not extracted in P2. The species `.hash` recipe states why it differs from `registry_content_hash`, or reuses it if it fits. |
| R6 | L2-m4 | Section encodes env 0 only, with `assert N_ENV == 1`; no `n_env`/`env` tokens in the bytes; multi-env fold left to the RL arc's B1 (env k with seed s == N=1 with seed s stays reachable). |
| R7 | L1-m4 | Id guard `next_unit_id + k <= INT32_MAX` computed in int64, checked before any write; `faction` range-checked to int32 and heading pre-wrap to int64 before any write (refusal stays atomic). Strict dtypes at spawn: bool arrays are a TypeError. |
| R8 | L1-m5 + L2-m5 (RNG) | §2.4/§9 state: `philox32.TWO_PI_Q16 = 411775` (the draw range) and `HEADING_PERIOD_Q16 = 411774` (the wrap period) are distinct on purpose; every kit-derived heading (atan2 output, Philox draw, `draw − PI_Q16`) passes `wrap_heading_q16`; the double weight on direction 0 (2 of 411775 buckets) is an ACCEPTED bias. 411774 and 411775 join T15's inputs. Reuse an existing Python `PI_Q16` if one exists; else define it once in `swarm_fixed.py`. |
| R9 | L1-m6 | Every facade write (spawn, reclaim) sets a per-species host-dirty flag (not synced, not digested) that P3's upload consumes; §9 says "any host store write"; the `_RESIDENT_SYNCED` line's comment carries the defaulted-`to_host()` caveat. |
| R10 | L1-m7 | Remove the `free_count()` best-effort recommendation: synced callers never clamp by capacity; overflow = `SwarmFull` (authoring error); a P4 clamp, if ever needed, is an authored species-row cap. |
| R11 | L1-m8 | Species-name regex `^[a-z][a-z0-9]*$` (no underscore); reserved column names; import-time assertion that all attribute and `.npz` key names are unique. |
| R12 | L1-m9 + L2-m5 (SimState) + L2-m9 (accessor) | P2 leaves `SimState` alone and says so explicitly (`get_state` is not a swarm capture path in P2). Read access for P3/P4/P5 is a free function `swarm_column(gmap, species, col)` the facade delegates to. |
| R13 | L1-m10 | Recorder: per-tick species-hash ring (int64 = first 8 bytes of the hash; int64 is an existing dtype class) instead of one 0-d value; print once on first truncation at `hw > swarm_units_cap`. |
| R14 | L1-m11 | The harness locator compares swarm carriers via `swarm_section_bytes` (+ `present`, `species_hash`), never `==` on ndarray dicts. |
| R15 | L1-m12 | Document the shared-`next_unit_id` cost against §9b's per-species byte-stability claim. |
| R16 | L2-m5 (rest) | `FACTION_NONE = -1` defined in `src/simulation/factions.py` (not in fire-12's diff) and imported by swarm; one sentence distinguishing `simulation/species.py` (`SpeciesDef`); wire-order tuple precedent = `PROP_MATERIAL_CHOICES` (append-only), not `WEAPON_ARCHETYPES`; §9 states P4's `swarm_brood` hand-off (stdlib-only schema duplicates + load-validates the species tuple, the prop_system route). |
| R17 | L2-m6 | Draft rules: facade = "the only Python-side writer"; swarm-store rule regains "a species is a table row + kernel, never a class / per-species if-chain" and "no atomic-ticket spawns"; flag `N_ENV` as the first batch-axis constant (the one N until B1). |
| R18 | L2-m7 | Trims: `SPECIES_RULES` deferred to P3; cut `Simulation.spawn_swarm`/`reclaim_swarm` (keep `sim.swarm`); xarch additions deferred to P3; `swarm_fixed.py` re-exports the rounding helpers from an existing boundary module and adds only `PI_Q16` (if absent), the heading period and the wrap law. |
| R19 | L2-m8 | §10 adds CLAUDE.md: append rows after the Deterministic RNG row; do NOT touch the stale "F5 reload" Config row (queued for after fire-12 merges); cite fire-12's `rad_*_sweep` resident planes instead of the deleted `cool_shift` caveat. |
