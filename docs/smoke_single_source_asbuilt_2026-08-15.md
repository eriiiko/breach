# Smoke single-source — as-built record (P-S1, 2026-08-15, Claude)

**Status: BUILT.** Executes Erik's ruling in
`docs/smoke_single_source_design_2026-07-24.md` ("DELETE source A ... duplicates
B's purpose") and kills the smoke→N₂ pressure pump the storm audit measured
(`docs/storm_audit_2026-08-14.md` §4.2). Scope was deliberately narrow — delete
the ex-nihilo fire-smoke scatter only. `soot_yield` is untouched; the design
doc's Q6 starvation-dependent yield law (§2.1) is still unbuilt, still stacked
on the unbuilt `o2-continuous-law` line, and out of scope here.

Branch `storm-damping` @ base `4dca575`.

---

## 1. What was deleted

**Source A** — the fire step's ex-nihilo smoke scatter: on every lit fire
tile, every tick, `smoke[neighbour] += smoke_emission·dt·I` into the 4
open air neighbours, with nothing debited anywhere.

| Site | Before | After |
|---|---|---|
| CPU mechanism | `cpp/src/fire_simulation.cpp:289-309` (the scatter loop) + the `emission_q` load-time precompute (was line 124) | deleted, replaced by a tombstone comment |
| CUDA mechanism | `cpp/src/cuda_fire.cu`'s `fire_smoke_emit` kernel (was ~243-272) + its launch + `emission_q` precompute | deleted, replaced by a tombstone comment |
| `FireParams::smoke_emission` | `cpp/src/fire_simulation.h` struct member | deleted (not left wired — see §2) |
| `config.toml` | `[physics.fire] smoke_emission = 0.8` | key removed |
| Python binding | `PhysicsRunner.__init__` (`src/simulation/physics_runner.py`) bound it from config | binding removed |
| pybind11 | `bindings.cpp`'s `FireParams` class + `cuda_fire_step` free function both carried `smoke_emission` | both dropped |
| GPU dispatch | `physics_engine.cpp`'s `step_tail` passed `this->fire.params.smoke_emission` to `breach_cuda::fire_step` | argument removed |

**Where the scatter actually executed** (the plan asked this to be verified,
since the storm ledger's seam attributes it to the "tail" pass, not
"fire_cast"): `fire_cast` in the ledger is `PhysicsRunner.cast_fire_heat`
(radiation → heat/rad_net) — an unrelated pass. The smoke scatter lived
entirely inside `FireSimulation::step()` (CPU) / `breach_cuda::fire_step()`
(GPU), both of which are called from `PhysicsEngine::step_tail()`
(`cpp/src/physics_engine.cpp`) — a single C++ call the ledger seams as one
pass named "tail". There was no second, hidden site and no dead half: one
CPU site, one CUDA mirror, one call site, all under `step_tail`.

**What was NOT touched:** combustion's soot channel (`cpp/src/combustion.cpp`
:770-772, `soot_yield`), the P4 decay→inert_N₂ credit
(`physics_engine.cpp`'s `run_substeps` trace loop), any damping dial, any
rail guard, and every other trace source (steam, grenade gas, explosion
smoke — see §5).

**Loud stale-key guard.** Following the `src/temperature_scale.py` migration-
guard idiom: `PhysicsRunner.__init__` now raises `RuntimeError` at load if
`[physics.fire]` still carries `smoke_emission`, naming this doc and the
07-24 ruling. An old config left un-migrated fails loudly instead of silently
doing nothing.

**Test/tool updates** (grepped `smoke_emission` across `cpp/`, `tests/`,
`tools/` — every hit fixed): `tests/test_pr3_capacity_law.py`,
`tests/test_continuous_o2_law.py`, `tests/test_eos_p4_combustion.py` (dead
`= 0.0` isolation assignments removed — nothing left to isolate, the
mechanism is gone, not just zeroed), `tests/cuda_fire_check.py` (dropped from
`DIALS`/`_PARAM_DEFAULTS`; the "dense fire block" sub-test's
overlapping-smoke-atomicAdd assertion is now an assertion that smoke stays at
its all-zero seed — the mechanism it used to prove order-free no longer
exists), `tests/test_lighting_demo_studio.py` + `tools/lighting_demo.py` (the
demo's soot-handover pair is now soot_yield alone), `tests/test_no_float_in_sim_tu.py`
(the ratchet baseline for `fire_simulation.cpp` tightened `double` 17→16 for
the deleted `emission_q` precompute — see that file's own changelog for why
`float` was NOT touched). Canon touched: `docs/architecture/engine/05_smoke.md`
§1/§4/Implementation-status and `docs/architecture/engine/06_temperature_and_fire.md`
stage 3 + the fire-parameter table (both now state combustion soot as the ONE
fire-smoke source; stage numbering kept, not renumbered, per this codebase's
own "retire the slot" convention).

---

## 2. Gate 0 (hypothesis reproduction — done by the orchestrator, reconfirmed here)

Two-room bench, 4800 ticks (200 s), P-F1b dials, shipped config:

- Sealed-room bulk-N creation inside the EOS pass: **+125.2 counts** (the
  P4 decay credit converting unbacked trace mass to full-pressure-weight
  N₂).
- Attribution: fire-step scatter minted **+200.3** smoke in the "tail" pass;
  combustion minted **+0.244** in the same run — source A outweighed source
  B roughly 800×.
- One-dial kill (`gases.smoke.decay=0`): bulk-N creation **0 to the LSB**
  (control — proves the pump is exactly the composition of the scatter with
  the decay credit, nothing else).

---

## 3. Gates

### B — round-trip conservation test (red → green)

New test: `tests/test_ps1_smoke_roundtrip.py`. Sealed 9×9 room, one wood
tile ignited, 300 ticks, `Σ` over ALL 7 gas planes (O₂ + inert_N₂ + steam +
smoke + poison + teargas + fuel_gas) in raw Q16.16 counts.

**Judgment call, flagged for Erik.** The plan's wording was "conserved to
the LSB." Measured reality (both before and after the fix) is that the
trace-gas semi-Lagrangian transport is *documented* as lossy — `cpp/src/
smoke_dynamics.h`: "NON-CONSERVATIVE by design (the `>>16` truncation is a
gentle built-in decay) ... accepted (Q-S2-1) ... NO flux form, NO limiter,
NO outflow clamp." That truncation only ever *removes* a fractional count
(never adds one), and it is pre-existing and unrelated to P-S1. So bit-exact
equality every tick is not achievable once the fire is alive and generating
real wind (verified empirically — see below) — the honest, correct claim is
**bounded-above** (nothing may ever exceed its start), the same idiom
`test_eos_p4_combustion.py`'s `test_e2e_1_sealed_room_fire_self_starves`
already uses for the O₂+N₂ pair. The test asserts `total <= total0` every
tick.

Measured on this scenario (standalone probe, not the committed test's exact
numbers but the same shape): with the scatter alive, total rose **+5244**
counts on the very first tick and **+617,761** by tick 120. With the scatter
deleted, total only ever drifted **down**, by **≤2400** counts over 300
ticks (out of a ~3.1M-count room) — three orders of magnitude apart, and the
opposite sign.

- **RED on HEAD:** `assert 3150972 <= 3145728` fails at **tick 0**
  (`+5244` counts).
- **GREEN after the fix:** 1 passed.

### C — ledger re-run (`tools/storm_ledger.py --ticks 4800 --damp 0.0 --pf1b`)

| pass | n_bulk (before) | n_bulk (after) | n_smoke (before) | n_smoke (after) |
|---|---:|---:|---:|---:|
| eos | **+125.2** | **0** | −125.9 | −0.7183 |
| combustion | −0.244 | −0.7193 | **+0.244** | **+0.7193** |
| tail | 0 | 0 | **+200.3** | **0** |

EOS bulk-N creation dropped from +125.2 to **exactly 0**. The fire step's
own smoke deposit ("tail" pass) dropped from +200.3 to **exactly 0**.
Combustion is now the only positive `n_smoke` contributor anywhere in the
ledger, at the same order of magnitude as before (+0.244 → +0.7193 — the
fire itself now burns differently, see §6, so the absolute combustion figure
moved too, but the *shape* — combustion-only, no other source — is exact).
Full per-pass tables recorded above (both runs; `ke`/`eth_gas`/etc. columns
also moved because the fire's own trajectory changed post-fix — see §6).

### D — CPU↔CUDA fire-path lockstep

`pytest tests/test_cuda_p68_fire.py -q -s` (wired to `tests/cuda_fire_check.py`
via `cuda_harness`):

- **PART 1** (32 fuzz configs + 8 deterministic forcers, incl. the
  dense-overlap block whose smoke assertion is now "stays at its all-zero
  seed"): **bit-identical**, `fire`/`temperature`/`smoke`/`wall_hp`, SET-equal
  destroyed.
- **PART 2** (130-tick O₂-rich ignition trajectory, CPU-backend vs
  GPU-backend lockstep): **bit-identical** every tick; 4 walls destroyed;
  fire self-starved 1297604→0.
- **PART 3** (CPU path vs the committed golden): **MISMATCH** —
  `aacc539bb832c978ae4588bc033c79da303bd3aa4960d778a9e82934d2a8e4cc` vs
  committed `28678e9d6210533f63cc701bba8f93194e23df9ebbdfa5f75f5d26681e897040`.

Every *parity* assertion (the thing this gate exists to catch) is clean.
The only failure is the shared committed golden, which is **expected**: the
canonical A/B scenario burns a fire, and P-S1 is a real, intended behavioral
change (deleting a mass-minting mechanism necessarily moves any trajectory
that includes it). Goldens are deferred by standing ruling during this
retuning phase (not this patch's call to re-bless).

**This same golden mismatch (identical new digest) also newly surfaced in 11
other CUDA gate files** once CUDA was built in this worktree (it had never
been built on this tree/branch before — the whole storming-audit arc ran
CPU-only): `test_cuda_eos_step`, `test_cuda_mg_solve`,
`test_cuda_p62_sl_advection`, `test_cuda_p64_kick_compression`,
`test_cuda_p66_conduction`, `test_cuda_p68_fire`, `test_cuda_p69_combustion`,
`test_cuda_s2b_raycaster_live`, `test_cuda_s3_water`, `test_cuda_s4a_smoke`,
`test_cuda_trace_smoke`. Every one of the 11 was individually verified
(full output inspected, not assumed): **every actual CPU/GPU parity check in
every file is bit-identical**; the sole failure in each is the same shared
`GOLDEN MISMATCH: aacc539bb832c978... != 28678e9d6210533f...`. None of these
files touch fire or smoke code — they moved because they all check the same
one canonical scenario's aggregate digest, and that scenario burns a fire.
`tests/test_w6_armory.py::test_canonical_scenario_golden_and_untouched_rng`
checks the identical golden directly and was **already failing on HEAD**
before this patch (pre-patch digest `9dbb9cd24bb1...`, unrelated pre-existing
reason, part of the tree's 36 known reds) — after this patch it still fails,
now with `aacc539bb832c978...` (the same new digest as everywhere else). Same
test name, still failed, both before and after — not a set change, just
a different wrong value for an already-red test.

### E — full suite (`pytest tests -q`)

- **Before** (HEAD, unmodified): 37 failed, 2150 passed, 29 skipped. 37 = the
  tree's 36 known reds + this patch's new test (expected red pre-fix).
- **After**: 47 failed, 2164 passed, 5 skipped.

Diff, explained in full:

- **−1**: `tests/test_ps1_smoke_roundtrip.py::test_ps1_roundtrip_all_gas_planes_never_exceed_start`
  — the new test, now green (as required).
- **+11**: the CUDA golden-mismatch tests named in §D above. All 11 were
  **SKIPPED before** (no CUDA build existed anywhere in this worktree until
  this patch's Gate A required building one) and are **newly visible, not
  newly broken** — every one's actual parity assertions pass; only the
  pre-attributed, standing-deferred golden fails, for the reason in §D.
- **36 unchanged**: every pre-existing red stayed red, under the same name,
  for the same class of reason (mostly stale `FireSimulation.step()` call
  signatures in `test_fire_feedback.py`/`test_s3b_fire_determinism.py`
  predating the continuous-O₂ law, and other already-broken goldens/digests
  in `test_b*`/`test_eos_p5_1_stoich`/`test_eos_p6_9_isotropy`/
  `test_cool_shift_axis` unrelated to smoke). None of these tests reference
  `smoke_emission`; none changed failure mode.

37 − 1 + 11 = 47. ✓.

### F — bench scorecard (`tools/fire_tune_loop.py`, P-F1b dials, 900 s window)

| metric | before | after | verdict shift |
|---|---:|---:|---|
| peak I | 0.488 | 0.714 | PASS → MISS (now above the 0.4–0.6 band) |
| peak time | 1.55 min | 1.97 min | MISS → MISS |
| fire death | never (900 s window) | never (900 s window) | unchanged |
| flame plateau T (game) | 309 | 384 | MISS → MISS, but **75 game closer** to the 400–500 target band |
| far-field T rise | 0.0 | 0.1 | PASS → PASS |
| room N_total min | 1.000 | 1.000 | PASS → PASS |
| far-field X min | 0.1990 | 0.2065 | PASS → PASS |
| wall_hp at end | 17.51 | 12.39 | more fuel consumed |

Matches the plan's prediction (X-fraction rises, flame T hotter) — by more
than "slightly." See §6.

---

## 4. Measured smoke-density change + suggested `soot_yield`

Two-room bench, 4800 ticks, P-F1b dials (same fixture/dials as gates C/F).
"Fire room" = the crate's own room (left interior, excluding the door/
partition/right room).

| | before | after | ratio |
|---|---:|---:|---:|
| total smoke minted (ledger, combustion+tail) | ≈200.5 counts | 0.7193 counts | **≈279×** |
| peak smoke density in the fire room (0–1 scale) | **1.000** (saturated) | 0.00125 | **≈800×** |
| mean smoke density in the fire room (time-avg, 0–1) | 0.4114 | 0.000013 | ≈31,600× (inflated — see below) |

The peak-density ratio (≈800×) lands almost exactly on the order of
magnitude the storm audit already flagged ("source A outweighs source B
~800×") — a good cross-check between two independent measurements. The mean-
density ratio is much larger only because the *before* state spent a large
fraction of the run pinned at the density ceiling (1.0, fully opaque) rather
than because the underlying mint rate varies that much tick to tick; peak
and the ledger's minted-total (≈279×) are the more trustworthy comparators.

**Suggested `soot_yield` for equal-look, scaled by the minted-ratio:**
`0.3 × 279 ≈ 84`. This is **not physically meaningful** — `soot_yield` is
the fraction of *burned mass* converted to soot and cannot sensibly exceed
1.0 (currently a headroom of only ≈3.3×, `0.3 → 1.0`). At the physical
ceiling (`soot_yield = 1.0`) the new smoke would still sit roughly **three
orders of magnitude thinner** than the old (buggy) look (peak density
≈0.004 vs the old 1.0). Matching the old visual density is not achievable
through `soot_yield` alone, because the old density was never physically
bounded in the first place — it was an ex-nihilo leak, not a combustion
product. **Flagged for Erik's HUMAN-TEST**: evaluate the new (much thinner,
honest) smoke look directly rather than trying to recreate the old one;
`soot_yield` (currently 0.3, config `[physics.combustion]`) is the dial to
turn if the new look reads as starved. This is a `soot_yield`-VALUE decision
only — this patch changed nothing about the dial itself.

---

## 5. Queued decision for Erik — class-level ruling on the OTHER ex-nihilo sources

P-S1 fixed fire smoke only ("one bug at a time," per the audit's own
practice). The same ex-nihilo pattern — a trace deposit with nothing debited
anywhere, which the P4 decay→N₂ credit can turn into a bulk-N pump exactly
the way it did for fire smoke — still exists in:

- **Steam** (`[gases.steam]`) — water→gas boil/evaporation sources. Possibly
  **defensible**: steam genuinely has a physical mass source (liquid water
  converting to vapour), unlike fire's smoke scatter, which conjured mass
  from nothing. Whether the current mechanism actually debits the water
  plane it should is not verified here — flagged, not investigated.
- **Grenade gas** (teargas/poison puffs on throw/detonate).
- **Explosion smoke** (`add_explosion_smoke`, the noisy disc deposit) — the
  design's own `docs/smoke_single_source_design_2026-07-24.md` explicitly
  called this out as "a separate legitimate source, UNTOUCHED" at design
  time, deliberately not part of this ruling's scope.

None of these were touched, measured, or exercised differently by this
patch. Erik's call: whether each is "physically defensible ex nihilo"
(steam, arguably) vs. "another instance of the same bug" (grenade/explosion
smoke, unclear) is a class-level question the storm audit did not answer and
this patch does not answer either.

---

## 6. Side effect flagged for HUMAN-TEST: the fire itself burns differently

Removing the smoke→N₂ pump doesn't only change how smoke *looks* — it
changes how the fire *behaves*, measurably:

- Two-room bench, 4800 ticks: final fire intensity **I=0.033** (nearly out)
  before vs **I=0.739** (still roaring) after, at otherwise-identical dials.
- `fire_tune_loop` bench (§3 gate F): peak intensity +46% (0.488→0.714),
  flame plateau temperature +75 game/+24% (309→384), more fuel consumed
  (wall_hp 17.51→12.39).

**Why, mechanistically:** the old pump was a real (if unintended) mass
injection, which is a real pressure injection (`p* = C·N·T`) localized in and
around the fire. That spurious overpressure was pushing wind outward from
the fire room, which plausibly diluted or displaced the fresh-O₂ inflow the
fire's own sustain law (`o2f`, the local O₂ mole fraction) depends on. Remove
the fake pressure source and the fire draws oxygen more effectively — a
*second-order* consequence of fixing a mass-conservation bug, not a
combustion-law change (nothing in `combustion.cpp` or the fire logistic
moved). This is exactly the kind of feel-adjacent shift the project's
HUMAN-TEST gate exists for: the numbers above are informational for Erik's
anchor re-check, not a claim that the new behavior is better or worse.

---

## 7. Pointers

- Blessed ruling: `docs/smoke_single_source_design_2026-07-24.md`.
- Audit: `docs/storm_audit_2026-08-14.md` §4.2 (mechanism), §4.5 (conservation
  verdict table), §5 item E (the ruling's origin).
- Canon: `docs/architecture/engine/05_smoke.md` §1/§4/Implementation status;
  `docs/architecture/engine/06_temperature_and_fire.md` §5 stage 3 + fire
  parameter table.
- New test: `tests/test_ps1_smoke_roundtrip.py`.
