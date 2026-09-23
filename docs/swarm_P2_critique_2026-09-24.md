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

## Lens 2 — Systems reuse & scope (Opus) — *pending*
