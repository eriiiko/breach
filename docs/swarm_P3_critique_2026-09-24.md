# P3 critique — arc #63 larva behaviour kernel

Critique of `docs/swarm_P3_impl.md` v1 (commit `4528571`). One section per
lens, filled one critic at a time.

## Lens 1 — determinism, twin spec & residency (Opus)

**Verdict: SOUND WITH FIXES** (1 blocker, 3 major, 8 minor). The integer
sequence is twin-safe as written, and so are both E1 readers. The defects are
in the gate spec, the resident-branch selection, the Philox key derivation,
and the N>1 paths that nothing exercises yet.

**Checked and holds** (so later lenses need not re-derive these):

- **V5.** TwoPhase rewinds `tick` at `simulation.py:1846`; the base ruleset
  routes the round clock (`ruleset.py:111-121`); OnePhase's clock is
  free-running (`ruleset.py:372-378`, and the boundary at `:405` does not
  rewind); ContinuousRealtime's `turn_number` stays at 1. The absolute tick is
  gap-free under all three rulesets.
- **F5.** Step 6's D2H (`physics_runner.py:1299`) comes before the host brackets
  (`:1303-1336`), and step 4 re-uploads `temperature` every tick (`:1242`). So
  the extra upload in `_run_swarm` is idempotent for the physics.
- **`solid` is const through the tail** (`physics_engine.h:104`).
- **Heading wrap.** The C++ truncating-`%` wrap equals `swarm_fixed.wrap_heading_q16`
  on 2×10⁵ values in [−2³³, 2³³] plus the edge values (scratch probe).
- **§5.3 numbers.** 13107/196805/2731/39360/1640/273/2731 reproduce, and so do the
  k=3 Kelvin ints (221730/658637/−215177/−171486).
- **Displacement bound.** `|narrow_round_signed(cos·step)| ≤ step`.
- **FOOTPRINT rounding.** `floordiv_q(s, 4)` equals `s >> 2` (floor) on both
  backends.
- **Philox.** The argument order matches `philox32_draw` (`philox32.h:175`).
- **Events.** The digest is an allowlist (`field_ab_harness.py:258`), and the
  renderer ignores unknown event types (`game_renderer.py:1178`).
- **V3.** No salt enum exists anywhere in `src/` or `cpp/src/`.

---

### L1-B1 — PART 1 as specified goes red on a correct kernel (§11 "PART 1")

**(a) A non-canonical heading reaches the output.** PART 1 draws "headings
incl. ±PI_Q16" into hand-built arrays, which bypass `spawn` and its wrap
(`swarm.py:775`). It then asserts "after every case the output satisfies the P2
invariants".

- `−PI_Q16` is outside the canonical range: `check_invariants` rejects
  `live_heading <= -PI_Q16` (`swarm.py:467-468`).
- L8 writes the stored heading back unchanged on every tick without a
  tumble/wall turn/death (§3.5 "canonical: stored value or a wrap() output").
- So any such slot that neither turns nor dies fails invariant 5, and the gate
  is red on a correct build.
- Non-canonical valid-slot headings are also outside the kernel's own
  contract: the §3.3 int32 add `heading + side*angle_q` is overflow-free only
  for canonical headings.

**(b) A zero-extent CUDA launch is invalid.** The "hw = 0 (nothing touched)"
forcer against `cuda_swarm_larva_step` gives `launch_hw = 0`.

- That means a 0-block grid, which returns `cudaErrorInvalidConfiguration`.
- The file-local `cuda_check` the build brief specifies (§13 step 5) turns that
  into an exception.
- §3.6/§4.4/§10 specify no zero-extent early return. `_run_swarm` skips hw 0,
  but PART 1 calls the binding directly.

**Why blocker.** A Sonnet implementer greens a red gate by bending something:
a wrap on every store in the kernel, or a weakened checker. That is the pattern
the test rule forbids.

**Fix.**

- PART 1's precondition is "inputs satisfy `check_invariants`". Draw headings
  from (−PI_Q16, PI_Q16], with extremes `−PI_Q16+1` and `PI_Q16`. Draw
  positions in-grid, and `unit_id < next_unit_id ≤ INT32_MAX`.
- Both CUDA entries do `if (n_env == 0 || launch_hw == 0) return 0;` before any
  allocation, transfer or launch.

### L1-M1 — the resident branch is picked by a sticky flag and leans on implicit cross-method invariants (§4.3 steps 5-7, §4.4)

**The sticky flag.** §4.3 step 5 sets
`resident = gmap.residency_on() and on_cuda`.

- `residency_on()` is set by `enable_residency()` (`gamemap.py:1736`) and is
  never cleared.
- The tick dispatch keys on the module global `_RESIDENCY_ENABLED`
  (`physics_runner.py:771`).
- **Failure.** After `set_residency(False)` on a map that has ever run resident
  (`set_residency` is a public toggle, `physics_runner.py:42`), `step()` runs
  the non-resident path. `_run_swarm` then takes the resident branch and
  launches on device `solid`, which only `_step_resident` uploads
  (`:1187-1188`, `:1242-1244`), so it is stale since the last resident tick.
- Any wall destroyed or sealed since then gives GPU larvae the old walls while
  a CPU run has the new ones.
- PART 2b cannot see this: its CPU world never enables residency.

**Implicit invariants.** Even on the true resident path the kernel depends on
two things:

- **`solid` freshness.** It relies on step 4's upload and on nothing writing
  `solid` before the end of slot 7. That is true today, but F5 itself says RL-B2
  moves these brackets.
- **Two sources for `high_water`.**
  - The kernel masks by device `d_high_water[e]`.
  - `launch_hw`, the CPU loop and `death_events` use the host array.
  - If the dirty protocol ever misses (a future host writer that forgets the
    flag), slots in [dev_hw, host_hw) keep last tick's host death records.
    Those become phantom `SwarmUnitDiedEvent`s.

**Fix.**

- Use `_run_swarm(gmap, sim_time, philox, *, resident)`, with `resident=True`
  passed only from `_step_resident` (still `and on_cuda`).
- The resident branch uploads `["temperature", "solid"]` (h·w bytes more), so
  the kernel's inputs are self-contained.
- Carry `hw[e]` as an env-block word and mask with it. It is a per-env scalar,
  so it belongs in the env block under §A rule 1, and the host array becomes the
  one source for the mask, the extent and the compaction.
- PART 2b adds a scripted mid-run `destroy_wall` next to larvae (the S8a
  `_EDIT_TICK` idiom, `cuda_s8a_check.py:134-146`).

### L1-M2 — `match_key` rejects numpy-int seeds and collides for spawned SeedSequences (D13, §6)

**Numpy-int seeds.** The probe (NumPy 2.4.6) gives
`default_rng(np.int64(5)).bit_generator.seed_seq.entropy` as a `numpy.int64`,
not an `int`.

- D13 says "TypeError if the entropy is not an int". Written as
  `isinstance(e, int)`, every `Simulation(seed=<numpy int>)` raises in
  `_reset_internal`.
- That includes swarm-free runs, because the key is computed unconditionally
  (§6). Dormancy breaks for the RL arc's most likely seed source.
- Today every caller passes a literal int, so no gate sees it.

**Spawned SeedSequences.** `SeedSequence(s).spawn(n)` children all carry
`entropy == s` (probe: `[42, 42, 42]`).

- Two sims built from sibling SeedSequences are the standard numpy multi-env
  idiom.
- They draw different `sim.rng` streams but get identical Philox keys, so their
  larvae draw identical words.
- The "key = the match seed" identity (CLAUDE.md "Deterministic RNG") fails
  silently.

**Fix.**

- `e = operator.index(ss.entropy)`: this accepts numpy ints and rejects lists
  and floats.
- Raise `ValueError` when `ss.spawn_key != ()`, or define the key over
  `(entropy, spawn_key)` and record it as a chapter-14 amendment. That is
  Erik's call.
- T-P3-27 adds a `seed=np.int64(k)` case and a spawned-child case.

### L1-M3 — N>1 (§A) is never exercised, and the GPU record-copy extent is unpinned (§3.6, §4.4, §10, §11 PART 1)

**N>1 is untested.**

- Every gmap-level gate runs `N_ENV == 1`, and PART 1 lists no `n_env > 1` case.
- So these run for the first time in the RL arc:
  - the env-offset arithmetic (`k = e*max_units + s`, `T + e*h*w`,
    `e = i / launch_hw`);
  - the in-kernel per-env mask, which is the §A rule-4 claim.

**The record-copy extent is unpinned.**

- §4.4 D2H's records `[0, launch_hw)` per env.
- The CPU twin writes `[0, hw[e])` and leaves the rest untouched (T-P3-8).
- For N>1 with unequal hw, env e's records `[hw[e], launch_hw)` come back as
  stale device scratch. That scratch is file-static and grown on demand, so it
  is reused across calls and fuzz cases.
- A full-array byte compare therefore differs on a correct kernel.
- §10's per-call "D2H the 9 columns + records" never states the extent.

**Fix.**

- Both CUDA entries copy records `[0, hw[e])` per env, and the host
  death-count scan uses the same range.
- PART 1 adds `n_env ∈ {2, 3}` with unequal hw, including 0.
- Pre-fill columns and records with identical garbage, then compare the full
  arrays, so the untouched regions are asserted too.

### Minor

**L1-m1 — the float ratchet guards the loop, not the law** (F6, §9, §16 "Swarm kernel law" draft rule).

- `swarm_larva.cpp` at 0/0/0 is a loop. The law (`larva_step_slot`,
  `head_sweep`, `turn_heading`, `probe_move`) lives in `swarm.h` and
  `swarm_larva.h`.
- The ratchet never reads those headers: it scans only `SIM_TUS` entries
  (`test_no_float_in_sim_tu.py:49-56, 366, 378-396`).
- The ingress lint is Python-only (`test_ingress_lint.py:30`).
- Parts of the inline law (`wrap_heading_q16`, `larva_derive`) are also
  instantiated in `bindings.cpp`. That file is off the `/fp:strict` list
  (`CMakeLists.txt:176-188`) and gets global `-ffast-math` on gcc/clang
  (`:29-33`). This is harmless only while the law stays integer.
- **Fix:** add `swarm.h`, `swarm_larva.h` and `philox_streams.h` to the scan at
  0/0/0, and reword the §16 rule to name the headers.

**L1-m2 — no `std::` helpers in `FP_HD` code.**

- nvcc runs without `--expt-relaxed-constexpr` (`CMakeLists.txt:125-135`).
- So `std::min/max/clamp` and `<cstdlib>` `std::abs` cannot be called from
  device code.
- The spec writes `clamp(...)` (FOOTPRINT, §3.2) and `|…|` (L4).
- **Fix:** say "hand-written ternaries, as `fixed_point.h` does" in §3.

**L1-m3 — own-position reads are unguarded and their invariant is unchecked.**

- L2 and L7 call `R(px, py)` without `tile_of`. RAW is "called only for
  in-grid points" (§3.2).
- In-grid is guaranteed only by spawn's bounds check (`swarm.py:743-750`) and by
  the kernel itself.
- `check_invariants` (`swarm.py:370-482`) has no position rule, so neither the
  PART 1/2 post-conditions nor T-P3-30 can flag an escape.
- An out-of-grid own position is an out-of-bounds device read (garbage or a
  fault), which differs from the CPU's.
- **Fix:** add "live pos ∈ [0, w<<16) × [0, h<<16)" to `check_invariants`. It is
  a pure-reader extension with no digest effect.

**L1-m4 — the Philox clock's `turn_number` is in no snapshot, and two absolute-tick definitions exist** (V5, §4.2).

- `SimState` (`simulation.py:1136-1146`), the recorder (`recorder.py:214`) and
  the digests carry the round-local `tick` only.
- The chapter-02 save contract (ch.17 §6, "designed, unbuilt") must therefore
  carry `turn_number` or the absolute tick. Otherwise a restored TwoPhase match
  replays round 0's Philox counters.
- `level_lights.monotonic_total_tick` = `(turn_number−1)·tpr + tick`
  (`level_lights.py:55-65`, used at `main.py:533`) is a second, ruleset-blind
  definition. It double-counts under OnePhase, whose boundary bumps
  `turn_number` (`ruleset.py:435`) while its tick is already absolute.
- **Fix:** one `Simulation.abs_tick` property (the doc's ruleset-routed formula)
  feeds `PhiloxClock`. Record the save-contract requirement in the §16 rule and
  ch.17 §6.

**L1-m5 — `swarm_present` is env-0-only** (`swarm.py:493-505`).

- D14 says the host "decides only whether there is anything at all". With
  env 0 empty and env k populated, the host skips the launch.
- **Fix:** gate `_run_swarm` on any species' `hw.max() > 0` (the step-7 test)
  instead of `swarm_present`.

**L1-m6 — reload rejection misses the assert branch** (§5.2).

- Only a `RuntimeError` from `temperature_scale.load` is turned into the seam's
  `ValueError`.
- But `load` → `_build` → `_assert_invariants` raises `AssertionError` when
  `phi_exp·k ≠ 1` (`temperature_scale.py:102-113`). It is silent under `-O`.
- So a Ctrl+R with a broken scale escapes `reload_species` as a crash, not a
  "RELOAD REJECTED" line.
- **Fix:** catch `(RuntimeError, AssertionError)`.

**L1-m7 — two test oracles contradict the accepted twin spec.**

- **T-P3-1.** The "diagonal staircase" wall must be 4-connected (edge-sharing).
  §3.3 accepts corner-cutting between two orthogonally solid tiles, so a
  corner-touching staircase is crossed by a correct kernel.
- **FOOTPRINT oracles.** Any Python oracle or display of the FOOTPRINT reader
  (law-test expectations, §12's felt-T map) must floor: `s >> 2` or
  `np.floor_divide`. `int(s/4)` or a cast float mean truncates instead. The
  probe shows the two differ for negative sums (−3, −1 → floor −1, trunc 0).
  That is exactly the sub-ambient cells near the coma thresholds.

**L1-m8 — PART 2b cannot see two of the paths it certifies.**

- **(i) Per-call CUDA through `_run_swarm` is exercised nowhere.** That is the
  `fn = bp.<kern.cuda>` host branch, and the `run_on_cuda.py` default. PART 1/2
  call bindings directly, and 2b is CPU vs resident.
- **(ii) "Upload happened" has no observable.** A spawn into a death-freed slot
  does not isolate R9's actual hazard: a reclaim of a live unit resurrected by a
  stale device `valid=1`.
- **Fix:**
  - Make 2b three-way (CPU / per-call through the sim / resident), and assert
    `swarm_cuda_calls()` advanced on the per-call world.
  - Add a runner counter `swarm_uploads`, asserted to tick exactly on dirty
    ticks.
  - Script a mid-run `sim.swarm.reclaim` of a live larva on both worlds.

### Findings in other systems (outside P3)

- **Monotonic-tick helper.** `level_lights.monotonic_total_tick`
  (`level_lights.py:55-65`) is not ruleset-aware and double-counts under
  OnePhaseWEGO (render-only today; see L1-m4).
- **`check_invariants`.** It has no position-in-grid rule, although spawn
  enforces one (`swarm.py:370-482` vs `:743-750`); see L1-m3.
- **`residency_on()` is sticky.** `GameMap.residency_on()` is never cleared,
  while `physics_runner.set_residency(False)` is a public toggle. Any future
  code keying on the map flag inherits L1-M1's hazard.

## Lens 2 — systems reuse, scope & verification (Opus)

**Verdict: SOUND WITH FIXES** — 2 BLOCKER, 6 MAJOR, 7 minor. What holds: one
FP_HD law under two loops, a field-generic shared kit (`head_sweep<R>`,
`side_preference`, `turn_heading`, `probe_move` take no larva knowledge — a
moth reader or a fish passability mask plugs in unedited), `SPECIES_KERNELS`
as a table, `SPECIES_UNITS` routed through `temperature_scale`, the salt enum
as the ONE enum chapter 14 requires (none exists; `philox32.h:55` asks for one
per project), and §9's fire-12 claims — every overlap row re-checked with
`git diff -U0 $(git merge-base HEAD origin/fire-12) origin/fire-12` and each
P3 hunk is ≥ 1 unchanged line from fire-12's. The two blockers: a parallel
coupling registry justified partly by a snapshot test, and a sensing-test pair
that passes on a sensing-dead kernel and goes red on a correct one. (Critic did
not open Lens 1.)

### L2-B1 — `SWARM_COUPLING_TABLE` is a parallel coupling registry, partly designed around a snapshot test (§8, F8, §13 "Do NOT touch")

- **Evidence.** CLAUDE.md "Coupling table": "a physics→unit coupling is one
  row". mechanics/05 §6 (`05_physics_unit_exchange.md:193`): "cudaunit swarms
  … plug in *behind this same interface* as another consumer". Chapter 17
  fork 7b: "the coupling table documents both rows side by side".
  `CouplingRow` already carries `reduction: Optional[str]` plus a `note` for a
  row that does its own sampling (`exchange.py:682`, the blast row). It lacks
  only an optional `response` and a consumer discriminator. The doc instead
  adds a second row type and a second table. One of its two reasons (F8) is
  that the unit table's tests pin its count. Only
  `test_exchange_reductions.py:266-267, 275, 291` pin the exact field list and
  a 5-tuple unpack. `test_wave_push.py:426-430` pins index 2 only, which
  survives an append, so F8 misstates it. Those tests say "the table GROWING is
  the design's point" (`:258`) and pin it anyway. That is the finding: the
  master test rule makes it something to report, not a design input. The other
  reason, the executor, is prospective. The EXCHANGE-READ slot does not exist
  (`exchange.py:28-34`, mechanics/05 `:231`), and it can filter.
- **Consequence.**
  - One concept gets two registries and two row types. The next species
    (moth → light) and the future executor must both know about both.
  - The design bends to protect a snapshot test, the perfusio #70 class.
- **Fix.**
  - Extend `CouplingRow` additively. fire-12 touches `exchange.py` only at
    `:296-317`. Add `response: Optional[Callable] = None`,
    `consumer: str = "unit"` (`"swarm:<species>"` for swarm rows) and
    `reads: tuple = ()`.
  - Append the two larva rows to `COUPLING_TABLE` with `response=None`. The
    future executor iterates only the rows with `consumer == "unit"`.
  - Rewrite the three pinned tests as properties:
    - the five shipped unit rows are registered with their reductions and
      responses;
    - they keep the relative order heat < push < teargas < poison;
    - every row's `reduction` is in `REDUCTIONS`, or is `None` with a note;
    - every swarm row has `response is None` and a `consumer` that names a
      `SWARM_SPECIES`.
  - Tighten T-P3-22: `reads` (as stored `<name>_q`) must be a subset of
    `SPECIES_KERNELS[sp].params`. That, not the names merely existing, is what
    makes §8's "cannot drift from what the kernel reads" true.
  - Drop the "Swarm coupling rows" draft rule. Amend the "Coupling table" row
    to: "… one row; a swarm row names its consumer and is executed in its
    species kernel".
  - Take `COUPLING_TABLE and its tests` off §13's do-not-touch list. The 9c
    comment points at `COUPLING_TABLE`.

### L2-B2 — the sensing tests pass on a sensing-dead kernel, and the ablation control is red on a correct one (§11 T-P3-15/16/17)

- **Evidence.**
  - T-P3-15 and T-P3-17 place the two groups at `x ∈ [4, 20]` and
    `[44, 60]` on a 64-wide walled grid. They measure mean Δx after 2400
    ticks against "4 SE of the flat-field control".
  - A random walk that starts near a wall drifts away from it with no sensing
    at all.
  - A float model of the ruled walk measures this (scratch probe, not
    committed). Default dials: step 1640 raw/tick on 0.333 m tiles, `p_base`
    273, turns 30–120°, a blocked move takes a wall turn.
  - In a flat field the left group's mean Δx is **+7.5 tiles, SE 0.72
    (z ≈ 10)**. On 1.0 m tiles it is +1.15, SE 0.31 (z ≈ 3.7).
- **Consequence.**
  - T-P3-15's asserted direction (left group +x, right group −x) is exactly
    the wall drift. It therefore passes on a kernel whose sensing is dead, or
    sign-flipped but weak.
  - T-P3-17 asserts "ablated drift within 4 SE of 0". It is red on a correct
    kernel at the default dials and flaky on 1.0 m tiles.
  - The builder's only routes to green are retuning the test dials until the
    wall drift hides, or bending the law.
- **Fix.**
  - Make the control PAIRED: identical placements, headings, uids and seeds.
    The asserted quantity is Δx(law, gradient) − Δx(ablated, gradient). It
    must be > 0 for the left group and < 0 for the right, beyond 4 SE of the
    paired difference.
  - T-P3-17 becomes: ablated law in the gradient vs ablated law in a flat
    field, Δx difference within 4 SE. The gradient response is gone and the
    wall drift cancels.
  - Alternative: state an RMS-displacement bound derived from the dials and
    place the groups ≥ 4 RMS from any wall.
  - T-P3-16 must state its placement: symmetric about the grid centre, so the
    wall drift cancels in expectation.
  - The tumble-frequency measurements (T-P3-9, T-P3-16) must be single-tick
    independent trials: many uids, one fixed configuration, probe unblocked,
    counting the `W[0]` decision. "Over many ticks" is ill-posed, because the
    wall turn overrides the heading (D6) and changes the configuration.

### L2-M1 — V1 contradicts a CLAUDE.md row and an Erik ruling, and is filed as "FYI (non-blocking)" (V1, §4.3, §14)

- **Evidence.**
  - CLAUDE.md:72: "new C++ orchestration lands in PhysicsEngine, not Python
    glue".
  - CLAUDE.md:92 and `rl_env_arc_proposal_2026-08-27.md:68-70` (Erik's
    ruling, 2026-08-27): "No new host-side tick logic … put it in
    PhysicsEngine (C++), not physics_runner.py".
  - `_run_swarm` (§4.3) is all Python: presence gating, a species loop with
    per-species `hw.max()` gating, dirty-flag upload and clear, kernel
    dispatch, `to_host`, death-record compaction and event building.
  - The cited precedent, `_run_combustion` (`physics_runner.py:979`), is a
    backend choice plus one binding call. §4.3's "orchestration is C++ except
    the backend choice" is not true of `_run_swarm`.
  - §9's routing list omits all the canon V1 amends: CLAUDE.md:72, :92, and
    chapter 17 §5's "last stage of PhysicsEngine's orchestration".
- **Consequence.** This is a silent exception to a ruled constraint. RL-arc B2
  (port the tail to resident) will not know `_run_swarm` is on its list.
- **Fix.**
  - Move V1 to §14 as a ruling for Erik (E2), not an FYI.
  - Cost the in-rule alternative: a C++ `swarm_step(...)` free function in the
    NEW swarm TU owns the species loop, the launch and the death count. That
    TU has zero fire-12 overlap, which is V1's real reason. Python keeps only
    the backend choice and the residency brackets the tail already uses.
  - If Erik accepts V1:
    - amend CLAUDE.md:72 and :92 with a named, scoped exception
      (`PhysicsRunner._run_swarm`, deleted at RL B2);
    - add `_run_swarm` to `rl_env_arc_proposal` §B2's port list;
    - route V1 into chapter 17 §5.

### L2-M2 — shared machinery sits in the per-species files, and larva-only rules sit in the shared module (§0, §3.6, §4.4, §10, §13 steps 3–5 and 8, vs chapter 17 §9b)

- **Evidence.**
  - §9b (`17_swarm_units.md:359-371`) puts the shared foundation in
    `cpp/src/swarm.h + .cu/.cpp`. Only "the state-machine kernel" goes in
    `swarm_larva.cu/.cpp`.
  - §13 step 5 nevertheless puts species-agnostic code in `swarm_larva.cu`:
    - `set_swarm_backend` and `get_swarm_backend`;
    - `swarm_cuda_calls` and `swarm_resident_calls`;
    - `SwarmResidentScratch` (env and death buffers);
    - the env-block H2D and the record D2H.
  - The `(e, s < hw[e])` loop and the `i → (e, s)` launch mapping are also
    written per species.
  - In the other direction, `swarm.py` (the shared module) gains
    `larva_derived`, `LARVA_SWEEP_DIST_Q16` and rules R2–R4. Those are
    larva-law invariants applied to EVERY species' row, since §5.3 runs
    `check_species_rules(table, …)` over all rows.
  - §5.1's "Known limit" names `SPECIES_FIELDS` but not `SPECIES_RULES`.
- **Consequence.** Species 2 either `extern`s larva's backend flag and scratch
  (a cross-species dependency) or copies them (a parallel system). Its row
  then fails R2–R4 with an AttributeError.
- **Fix.**
  - Species-agnostic device code goes in `swarm.cu`: the backend flag, the
    counters, the scratch, the env and record transfers, and a
    `template <class Law>` launch core in a shared `.cuh`.
  - The `(e, s)` loop template goes in `swarm.cpp`.
  - `swarm_larva.*` holds only `larva_step_slot`, its params POD and its
    derive.
  - `SPECIES_RULES` becomes per species: `dict[species, tuple[SpeciesRule]]`,
    with R1 shared and R2–R4 belonging to larva. Add an import-time assertion
    `set(SPECIES_RULES) == set(SWARM_SPECIES)`, the `SPECIES_KERNELS` idiom.
  - Extend §5.1's known limit to name the rules.
  - §16(b)'s "Swarm shared kit" row then points at `swarm.h` +
    `swarm.cu`/`.cpp`.

### L2-M3 — the absolute tick already has a helper, and the doc adds a second formula inline in the conductor (V5, §4.2, §13 step 12)

- **Evidence.**
  - `src/level_lights.py:55-65 monotonic_total_tick(turn_number,
    ticks_per_round, tick)` is the existing "monotonic across rounds" helper.
    Its callers are main.py:532, renderer/frame_lights.py:35 and
    tools/lighting_demo.py:943.
  - It is WRONG under `OnePhaseWEGO`. There `sim.tick` runs freely
    (`ruleset.py:372-378`) while `turn_number` still increments
    (`ruleset.py:435`), so the helper double-counts every round.
  - The doc's ruleset-derived formula,
    `round_index * ticks_per_round + round_tick`, is correct under both
    rulesets. But it sits inline in the `physics_runner.step(...)` call
    (§13 step 12). That puts a logic block in the tick conductor, whose
    CLAUDE.md rule is one call line.
  - §16(b) says "the counter tick is the ABSOLUTE tick" but never says where a
    consumer gets it.
  - fire-12's D4 fan still keys on `sim.tick`.
- **Consequence.** Three notions of "the tick" coexist: round-local,
  render-monotonic (wrong under OnePhase), and P3's inline formula. The next
  Philox consumer will derive it yet again.
- **Fix.**
  - Add one `Simulation.total_tick` property beside
    `round_index`/`round_tick` (`simulation.py:1097-1106`, outside fire-12's
    hunk at 1616): `self.round_index * self.ticks_per_round +
    self.round_tick`.
  - The conductor passes `PhiloxClock(*self._philox_key, self.total_tick)`,
    one line.
  - §16(b) gets a row: "the absolute tick is `sim.total_tick`; every
    tick-keyed determinism input (Philox counters, phase rotations) reads it,
    never `sim.tick`".
  - T-P3-27 asserts on `sim.total_tick`.
  - Report `monotonic_total_tick`'s OnePhase double-count (see "Findings in
    other systems" below). Its callers should move to `sim.total_tick` after
    P3.

### L2-M4 — the Kelvin conversion makes P2's strict-schema test T22 vacuous (§5.2 two-pass `from_config`, F7, §11 testkit)

- **Evidence.**
  - T22's `expect_reject` builds `cfg = Namespace({}); cfg.swarm = …`, with
    no `physics` (`test_swarm_species.py:76-80`). It asserts only
    `pytest.raises(ValueError)`, without matching the key.
  - Under §5.2, any config that PASSES validation reaches pass 2. Pass 2's
    first Kelvin field calls `temperature_scale.load(cfg)`.
  - That call raises `RuntimeError` when `[physics.temperature_scale]` is
    missing (`temperature_scale.py:136-139`), and §5.2 re-raises it as
    `ValueError`.
  - The Kelvin rows are REQUIRED, so every validation mutant still raises
    `ValueError`: a lax `KIND_REAL_Q16` branch, a dropped bounds check, a
    Namespace pass-through.
  - T22's "must break under …" clauses therefore no longer hold. F7 closed one
    vacuity (missing keys) and opened this one.
- **Consequence.** The strict-schema gate silently stops killing mutants the
  day P3 lands.
- **Fix.**
  - T22's rejection cases assert that the message names the offending key:
    `match=re.escape(f"[swarm.{sp0}].{f.name}")`, and each structural case
    matches its own token.
  - The testkit's `swarm_cfg()` returns a full cfg wrapper that carries
    `physics.temperature_scale` from CFG. A rejection can then only come from
    the validator.
  - Add one positive T22 case: the testkit's full valid cfg passes
    `from_config`.
  - The four P2 helpers keep their current explicit values
    (`max_units=64/8192`, `hp_max=1.0`) as overrides. P2 tests then do not
    start depending on PLACEHOLDER tunables that a P4 retune will move.

### L2-M5 — the float ratchet's "hard 0/0/0" covers the loop, not the law (F6, §9, §16(b) "Swarm kernel law")

- **Evidence.**
  - `test_no_float_in_sim_tu.py:49` (`SIM_TUS`) and `:378-395` scan only the
    `.cpp` files named in `SIM_TUS`.
  - Per §3 and §13, the whole law is `FP_HD inline` in
    `swarm.h`/`swarm_larva.h`: `larva_step_slot`, `larva_derive`,
    `head_sweep`, `side_preference`, `turn_heading`, `probe_move` and
    `wrap_heading_q16`. `swarm_larva.cpp` is a two-line loop.
  - The same headers are also compiled into `bindings.cpp`, because §10
    includes `swarm_larva.h` to bind `larva_derive` and the wrap.
    `bindings.cpp` is NOT on the strict list (`CMakeLists.txt:176-188`).
  - That is harmless only because the law is integer-only, and nothing
    enforces that.
- **Consequence.** A `float` or `double` in the law passes every gate that
  §16(b) calls hard.
- **Fix.**
  - Extend the ratchet with a header list held at 0/0/0: `swarm.h`,
    `swarm_larva.h`, `philox_streams.h`. It is the same line-count test run
    over headers.
  - §16(b)'s rule says "the law headers and the loop TU are on the ratchet at
    0/0/0".
  - Place a new `SIM_HEADERS` tuple after fire-12's BASELINE hunk, like §9's
    `SIM_TUS` extension.

### L2-M6 — the first-look protocol is predicted by its own probe to show a comatose population, not thermotaxis (§12, Appendix A3)

- **Evidence.** A3 (`swarm_P3_impl.md:1463-1469`): with the 0 °C cold hold
  on, the room bulk time-means about 7 °C, below `T_ctmin` = 10 °C. Every
  forcing tried leaves a spatial sd of about 15 K. §12 notes the cold bulk but
  only reports it in the caption.
- **Consequence.**
  - Under either E1 reader, most larvae spend the run in COMA.
    - RAW: about half the bulk cells sit below 10 °C, so larvae flap.
    - FOOTPRINT: nearly all of them do.
  - Panel (c) (mean |T_felt − 30 °C| of RUN larvae, law vs ablation) averages
    only a handful of units.
  - P3's stated exit is Erik's first look at the behaviour (plan row P3). It
    would show chill coma, not "toward T_prefer from both sides".
- **Fix.**
  - Calibrate the forcing so the bulk sits inside (`T_coma_exit`,
    `T_lethal_hot`) with margin. Options: a smaller cold patch, or clamping
    the cold hold at `T_ctmin − 5 K` rather than 0 °C. Probe it in the 24×36
    room, as A3 did.
  - After the 10 s warm-up, the tool ASSERTS two conditions: the bulk
    time-mean is in [15, 25] °C, and fewer than 20 % of interior felt-T cells
    are below `T_ctmin`. Otherwise it exits non-zero and prints the numbers.
    It never plots a coma picture silently.
  - Panel (c)'s felt-T comes from a numpy copy of the E1 reader (§12).
    Cross-check it once per run against the kernel's own decisions (e.g.
    every COMA entry has numpy felt-T < `T_ctmin`). The headline metric is
    then not a second, unchecked implementation.

### Minor

- **L2-m1 — FOOTPRINT's "mean" is not `REDUCTIONS["mean"]` (§3.2, §8).**
  - If E1 = FOOTPRINT, the reader is `floordiv_q(sum, 4)` with edge tiles
    CLAMPED, i.e. double-counted.
  - The row it is labelled with, `REDUCTIONS["mean"]`
    (`exchange.py:141-155`), is the round-half-away `mean_round` (twin of
    `fixed_point.h:747`), and it EXCLUDES off-grid tiles.
  - D12's own sign-symmetry rationale also argues against the floor.
  - Fix: the FOOTPRINT reader uses `fixedpoint::mean_round` over the in-grid
    tiles (count 1–4), so the row's label is true. Otherwise the row must not
    claim `mean`. This takes no position on RAW vs FOOTPRINT.
- **L2-m2 — derive once on the host and drop the test-only twins (§3.4, §4.3
  step 4, §10, T-P3-20).**
  - `larva_derive` exists in both C++ and Python so that R1/R2 (Python) and
    the kernel (C++) agree. Proving it takes a binding and an identity test.
  - `pack_env_params` already runs R1/R2 on the exact launch ints, on the
    host. Pass the derived `step_q`, `p_base_q` and `p_alarm_q` as the
    kernel's per-env ints. The rules then check exactly what the kernel
    consumes, by construction.
  - That removes the C++ derive, the `swarm_larva_derived` binding, T-P3-20,
    and the `ENV_DT_Q`/`ENV_INV_TILE_Q` words.
  - The same applies to `swarm.LARVA_SWEEP_DIST_Q16`, a Python twin with no
    Python consumer.
- **L2-m3 — reuse P2's scenario and the existing xarch line builders (§11
  xarch, T-P3-30, PART 2b).**
  - `field_ab_harness.swarm_scenario_sim` (`:146-174`) is P2's "ready
    scenario for P3's lockstep". It has a fire at (8,8) on the larvae's
    diagonal and a hull breach.
  - It goes kernel-active in P3 without mention, and P2's T7/T9 now exercise
    the kernel. List them in "must stay green", and say why a second thermal
    scenario is needed (or extend the first).
  - Build the swarm xarch run from
    `capture_trajectory(make_sim=swarm_thermal_scenario_sim)` plus the
    existing `build_perfield_lines` and `first_divergence`
    (`_xarch_perfield_digest.py:280-315`). Add only a new `build_swarm_lines`.
    A FIELD divergence upstream of the swarm is then localized before its
    swarm consequence.
- **L2-m4 — route every deviation into the passage it amends, and point rules
  at code (§9, §16(b)).**
  - §9 routes only one "As built (P3)" line under the §4 banner and one
    chapter 14 sentence. Name the passages:
    - V4 → chapter 17 §4.1 step 3's "clamped to the grid";
    - V5 → chapter 17 §3 and the chapter 14 amendment's
      `counter = (tick, …)`, and the CLAUDE.md "Deterministic RNG" row →
      "absolute tick (`sim.total_tick`)";
    - V1 → chapter 17 §5 (with L2-M1).
  - The "Swarm kernel law" draft rule says "every draw is a row of the doc's
    draw table", which points at a dated capture doc. Move the draw table into
    `swarm_larva.h`'s header comment and point the rule there.
- **L2-m5 — the death-event rule pre-commits a sim consumer to an undigested
  event (§7, §16(b)).**
  - The draft rule says "renderers and later food consumers read the event".
    But tick events are renderer signals (`events.py:1-30`), and
    `SwarmUnitDiedEvent` is outside `_SYNCED_EVENT_TYPES`
    (`field_ab_harness.py:258`).
  - Reword: renderers read the event. A sim-side consumer (P6 food) acts at
    the emission point. If it ever reads the event, the type joins the synced
    event digest in that patch.
- **L2-m6 — test hygiene (§11).**
  - T-P3-3 "check_invariants-style checks on the arrays": reuse
    `swarm.check_invariants` on a duck-typed store (a `SimpleNamespace` with
    the gmap attribute names; the function only `getattr`s). Never a
    re-derived copy.
  - PART 2's "scripted synthetic T field and interior walls" must be
    standalone arrays passed to the kernel. They must not be writes to
    `gmap.temperature` or `gmap.solid` ("Gas temperature is a mirror"; GameMap
    topology changes only via `destroy_wall`/`seal_tiles`).
  - T-P3-1's "diagonal staircase" must be 4-connected: an 8-connected diagonal
    is exactly what the ACCEPTED corner-cutting crosses. Add a positive case
    proving the corner-cut IS allowed, so no builder "fixes" it.
  - T-P3-7's geometry must work under either reader: a lethal region ≥ 3×3,
    with positions chosen against the reader's footprint.
  - T-P3-23 divides by a literal 65536. Use `swarm_fixed.dequantize`/`FP_ONE`.
- **L2-m7 — resident dormancy is untested (D14, T-P3-26).**
  - T-P3-26 proves dormancy on the CPU path only.
  - On the resident path, a non-dormant `from_host(["temperature"])`
    re-uploads identical values, so no digest can catch it.
  - Add a swarm-free resident leg to PART 2b. It asserts that
    `swarm_resident_calls()` did not advance and (spy) that no swarm upload
    happened.

### Findings in other systems (outside P3)

- `level_lights.monotonic_total_tick` double-counts under `OnePhaseWEGO`,
  because `sim.tick` runs freely while `turn_number` increments. Under
  `--control onephase`, beacons jump `ticks_per_round` ticks at every round
  seam (main.py:532, tools/lighting_demo.py:943/1612).
- `test_exchange_reductions.py:260-300` pins `COUPLING_TABLE`'s exact field
  list and a 5-tuple unpack, while its own comment says the table is designed
  to grow. L2-B1's fix closes it, if adopted.
- Backend-setter lists are duplicated per script (`tools/run_on_cuda.py:92-108`,
  `tests/cuda_s8a_check.py:64`, `tests/cuda_sky_exchange_check.py:38`, …). F9
  (combustion missing from `run_on_cuda`) is a symptom; one canonical list
  would close it.
- The float ratchet scans no header anywhere. Every FP_HD-inline sim law in
  `cpp/src/*.h` (e.g. `gas_energy.h:56-75`) is outside it, not only P3's.
