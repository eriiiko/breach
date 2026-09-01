# Phase 3c design brief — fire death-side & ignition redesign (session #12)

> **Status: SESSION OPENER — prepared 2026-09-01 (Home Desktop) for seamless
> pickup on the work PC.** This is the brief for the 3c DESIGN SESSION
> (Erik + Claude, decisions together). Read in order: issue #12's comments
> (the session log), `docs/fire_phase3a_measurements_2026-09-01.md` (the
> evidence), then this brief. Working mode: design doc → adversarial
> critique → patches with gates (arc standard).

## 0. Rulings already locked (do not re-litigate)

- **3b (Erik, 2026-09-01): NO model change.** The July "decouple sustain
  heat from displayed T" question is closed — 3a showed k_grow is simply
  the wrong knob for G1's magnitude; H_bed/cool_shift are the temperature
  levers and move in Phase 4. 3c is about the DEATH SIDE + IGNITION, not
  the heat model.
- G12 one map shipped + blessed; Phase 2 HUD + fire_tuning level shipped +
  blessed; 3a measurements stand (verdict table in the memo).
- Erik on 3a's transient shape: peak at ~2 min / 80 s ramp is "very very
  nice" — the RAMP is not on trial, the DEATH is.

## 1. The problem, in 3a's numbers

The reference fire self-extinguishes at 27.4 min with **26.7% fuel
unburned** — death by heat/availability collapse (T falls through
`fire_T_ext` as `avail = F·o2f` erodes), never by fuel exhaustion, and
never by O2 (X at death 0.184–0.203, gate is 0.13). Sealed rooms only
*shorten* the same collapse (18.9 min). G3 wants: fuel-governed burnout in
5–10 min; G5 wants: sealed rooms genuinely starve fires.

## 2. Scope — one coherent design, these items

1. **Die-term review (G3/G11)**: full death-side pass — `die =
   k_die·(1−avail·hot)·I + k_wind_strip·W·(1−I)·I`, the `I_min=0.02`
   snap, the claim gate (`combustion.cpp:511`), FUEL_FLOOR/charred-at-1-LSB,
   the never-destroys invariant. Goal: fuel exhaustion (and, sealed, O2)
   becomes the thing that actually kills fires.
2. **NEW (Erik, 2026-09-01): the hot-burns-faster gap.** `hot` clamps at 1
   above `T_ext + fire_T_span` (≈573 K honest) — beyond that, extra heat
   buys ZERO extra burn rate, so "really hot + O2-rich races to ashes" is
   not in the engine. Design question: should burn/consumption rate keep a
   T-dependence past saturation (e.g. widen fire_T_span, or a second
   T-factor on fuel_cost/wall_damage)? Interacts with item 1's death goal
   — a hot fire that eats fuel faster reaches fuel-death faster, which may
   close much of G3 on its own.
3. **Exposure-integral ignition (G8)**: dwell-time law replacing the
   instantaneous threshold (walls must not flash-ignite off transient
   shockwave heat, but sustained heating eventually ignites them —
   Erik's 2026-08-25 comment). The instantaneous compare becomes its
   limiting case. If adopted this is a NEW canonical system (rules row at
   implementation).
4. **Ember + auto-reignite + wind-strip = ONE hysteresis design
   (G6/G7/G11)**: today's edge-trigger re-arms only after cooling below
   threshold — a still-hot stripped tile cannot reignite (structurally
   confirmed). k_wind_strip revival rides this (⚠ forbidden band:
   any material `wave_absorb ∈ (0, 0.02)` while strip > 0 —
   `physics_runner.py` hard-errors; re-check wave_absorb values).
5. **o2f vacuum amendment (G5/#7)**: o2f is mole-fraction only — add the
   absolute-density factor (Erik's amendment on record) so fires cannot
   burn in near-vacuum. Also revisit the 0.13 gate vs 3a's finding that
   fires die of heat long before X reaches it.
6. **Config algebra rewrite (small, rides along)**: the T* equilibrium
   comment (config.toml ~:855) overshoots reality 105× (constant-I
   assumption never holds) — rewrite as an explicit transient-regime
   note so nobody tunes against it again.

## 3. Benches to add/run during 3c (design informs, then verifies)

- **Infinite-fuel variant (Erik's M7 suggestion)**: pin `wall_hp` (debug
  write each tick, or a huge-hp material row in the harness) so the O2
  reservoir effect is isolated from fuel erosion → does a sealed infinite-
  fuel fire ACTUALLY hit the 0.13 O2 wall? This cleanly separates item 1's
  two death channels.
- **Cluster coupling**: 1 crate vs the 2×2 bonfire stage
  (levels/fire_tuning station 1) — measure the mutual radiation feeding
  (burning tiles are pairwise net-T⁴ emitters; a cluster should hold
  `hot`=1 collectively). Baseline exists: M1 single-crate curves in
  `tests/_phase3a_artifacts/` (untracked, Home Desktop) + the memo.
- Reuse `tools/fire_timing_harness.py` + `tests/_phase3a_driver.py` —
  extend the driver, never a parallel bench.
- M7's material confound (wood conductivity 0.15 vs furniture 0) — keep
  the supplementary furniture-sealed scenario as the sealed reference.

## 4. Open design questions for Erik (the session's decision list)

1. Death priority: when fuel is plentiful but O2 falls — how sharp should
   the O2 death be (current linear o2f fade vs a harder wall near the gate)?
2. Item 2's shape: T-dependent burn rate — cap it (how high does "burns
   faster" scale before it's just I_cap again)?
3. Ember: a real state (T-sustained, low-I, low-consumption) or keep
   emergent-but-reachable via the hysteresis fix? Smolder rationale
   recorded at `02cf4ca`.
4. Auto-reignite arming: pure T-hysteresis, or exposure-integral (item 3's
   law) doing double duty as the re-arm?
5. Dwell law memory: per-tile accumulator is a new SYNCED field (digest
   membership → spec version bump + golden re-baseline) — accept that
   cost, or design a stateless approximation?
6. k_wind_strip revival target behavior (July parked it deliberately).

## 5. Machine/handoff notes

- Branch `fire-12`, everything through 3a is pushed (`a9d4681`). Work PC:
  `git fetch && git checkout fire-12`; conda env `data` exists THERE
  (`conda run -n data python -m pytest tests -q`); Lenovo build scripts
  `cpp/build_cpu_data.bat` / `cpp/build_cuda_lenovo.bat`. Rebuild both
  builds after checkout (G12 changed no C++, but the working tree moves).
- Raw 3a artifacts are untracked on the Home Desktop only; every number
  that matters is in the committed memo (+ regeneration commands in its
  appendix — the driver is committed, so curves are reproducible anywhere).
- Lenovo housekeeping still owed there: velocity-clamp worktree removal +
  VS Code re-point (issue #12 comment 2026-08-31).
- Known suite reds everywhere: 2× cool_shift (parked → Phase 4). The
  test_bench_two_room scipy red is Home-Desktop-only (env, not repo).
