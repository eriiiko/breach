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
