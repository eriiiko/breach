# Storm ledger audit — measured report + decision sheet (2026-08-14)

**Status: investigation record, analysis only. No sim-path code was changed;
config.toml was not touched; every dial move went through
`fire_timing_harness.apply_overrides` and was restored.** The deliverable is
this report and its decision sheet — the decisions are Erik's. Written by
Claude on branch `storm-damping` (base `ee97f61`, post
temperature-scale-unification merge), per Erik's standing ruling: AUDIT BEFORE
ANY DAMPING DIAL; nothing feel-adjacent ships from this arc.

**For a reader with no session context.** Erik reported (2026-08-03): fire
perturbs the atmosphere so much that it oscillates unnaturally ("storming").
The overnight analysis `docs/fire_atmosphere_oscillation_analysis_2026-08-03.md`
attributed it to a two-room door-neck Helmholtz mode with zero momentum
dissipation, and reported (its §5, self-flagged as unverified) that the
shipped air-damping lever `wave_absorb` opens a violent instability window at
0.002–0.01 with a live fire. This audit (1) independently re-ran that
battery on today's tree, actively trying to refute it, (2) committed the
two-room bench the fire arc never had, and (3) built an exact per-pass
momentum/energy ledger over the Q16.16 state and ran it on the bench and on
Erik's captured in-game blowup (`debug_blowup_20260814_015714.npz`, 555
recorder snapshots, 70×100 level, ticks spanning several rounds).

New apparatus (committed this arc):
- `levels/bench_two_room/` + `tools/bench_two_room.py` +
  `tests/test_bench_two_room.py` — commit `2f60c23`.
- `tools/storm_ledger.py` — commit `12e6d8f`.

Baseline gate: the suite's known-red set (36 FAILED, name-sorted) was captured
before the first commit and is byte-identical after the last; the additions
contribute 3 new PASSING tests (2 bench gates + the level round-trip
parametrization picking up the new fixture).

---

## 0. The five headlines

1. **The 08-03 §5 instability window did NOT reproduce on today's tree — and
   the reason is a config change, not a fix.** P-K0 (2026-08-13) promoted
   `k_wind_strip = 0.0` into config.toml ("plume self-blow-out off",
   2026-07-23). The 08-03 battery ran at the then-shipped `k_wind_strip = 0.5`,
   which the P-F1b override set does not touch. Restoring 0.5 by override
   reproduces the doc's window quantitatively (§2). **The window is real but
   currently dormant**; every other 08-03 claim reproduced, several to the
   third digit.
2. **The instability's engine is compression-work compounding, not the 1/N
   combustion amplifier.** In the unstable bench runs the amplifier's measured
   gain never exceeds **1.10×** and its 0.05 floor never engages
   (`heat_floor_hits = 0`); the runaway is the multiplicative
   `T ← T·(1−k)` step-4c update with its ±0.5 rate rail and no value bound —
   a supercooling spiral to the T_MIN floor (T_abs → ~1 K), exactly the
   hazard `eos_solver.h:168-186` warns about (§4.3).
3. **Erik's real in-game blowup is the SAME engine on the opposite rail** —
   a hot compression runaway: a starved, smoke-saturated, fully evacuated fire
   cell (O2 = 0.000, N ≈ 0) has its T multiplied by **×1.4957 per tick for
   ~20 straight ticks** (the ×1.5 = 1+T_WORK_CLAMP rail signature, measured
   from the dump) until the T_MAX_PHYS ceiling (16000) catches it; pressure
   spikes of 47–65 atm follow when gas slams back into the pocket, tripping
   the recorder. **Here the 1/N amplifier IS engaged at its full 20× cap**
   (the 08-03 doc's suspicion — confirmed for the in-game case, refuted for
   the bench window) (§4.4).
4. **The ledger found a real, unintended mass/pressure pump:** fire's ex
   nihilo smoke plume (`fire_simulation.cpp:289-310`, a deliberate legacy
   visual) decays into the **full-pressure-weight** inert-N₂ plane
   (`physics_engine.cpp:509-516`, the deliberate P4 "decay is oxidation into
   inert bulk, not deletion" rule). Two defensible decisions compose into:
   a sealed burning room gains **+42% of its entire bulk gas inventory in
   200 s** (289 → 411 counts). With `gases.smoke.decay = 0` the creation is
   **0 to the LSB** (§4.2). Smoke sat as trace mass at pressure weight 0.02;
   after decay it weighs 1.0 — so the pump inflates p* where smoke is dense,
   i.e. exactly in and around the fire room.
5. **Momentum dissipation is genuinely absent, and the O2-throttling objection
   to fixing it has (today) evaporated.** The kick is the only momentum
   source; interior sinks are none at shipped dials. And because
   `k_wind_strip` is now 0.0, air damping no longer kills the fire: at every
   `wave_absorb` in 0→0.1 the bench fire lived **100%** of a 200 s run
   (the 08-03 "fire died faster under damping" side effect was mediated
   entirely by the wind-strip death term, not by O2 starvation) (§2.4, §5).

Ledger verdict in one line: **both** — one true injection bug-by-composition
(the smoke→N₂ pump, headline 4) and one rail-design gap that manufactures
energy through the ambient reservoir once a runaway starts (headline 2/3);
but the storming Erik sees at normal operation is still, as the 08-03 doc
said, **physically-sourced fire work with dissipation genuinely absent** —
the pump and the rails set how bad the tail gets, not whether the air rings.

---

## 1. Method

- Worktree `storm-damping` @ `ee97f61`; CPU module rebuilt from this tree
  (`cpp/build_cpu_data.bat`); all runs CPU, `conda run -n data`.
- Probe: the committed `tools/storm_probe.py` / `tools/batch_storm.py`
  apparatus (the 08-03 instrument), P-F1b dial set via overrides, geometry
  two 12×12 rooms, 0.5 m tiles, 1-tile door, 24 tps, 200 s unless stated.
- Ledger: `tools/storm_ledger.py` — seams `PhysicsRunner.step()` at its
  Python call sites (fire_cast / water / eos / combustion / sky / tail;
  `run_substeps`/`step_tail` via an engine proxy) and takes int64-exact sums
  of the Q16.16 planes before/after each pass. The unattributed residual
  ("other") closes to float epsilon every tick; O2 closes to the LSB.
- Erik's dump: `debug_blowup_20260814_015714.npz` (recorder default fields:
  P, T, O2, smoke, fire, obstacles — **no wind planes**, so the dump supports
  energy/state analysis only; a momentum ledger needs a recorder session that
  adds `wind_x`/`wind_y` to `fields`).

## 2. Phase 1 — reproduction vs the 08-03 analysis, claim by claim

| # | 08-03 claim | 08-03 numbers | today (shipped config + P-F1b overrides) | verdict |
|---|---|---|---|---|
| a | sealed room calm vs two-room still growing at 200 s | room: KE final 0.48, umax 0.12 m/s; two-room: KE 225, umax 6.25, growing | room: KE settled 0.216, umax final 0.092; two-room: KE settled 243.6, umax final 4.91, **still growing** (fit −0.001/s) | **CONFIRMED** |
| b | storming peaks at 2-tile door, collapses at 6 | KE settled 212 / 912 / 46.6 / 1.5 (door 1/2/3/6) | 243.6 / **1220** / 33.7 / 1.63; umax final 4.9 / 6.1 / 0.67 / 0.27 | **CONFIRMED** (peak-at-2 sharper today) |
| c | steady pinned ΔT does not pump (settled KE flat across 12× drive) | 0.365/0.370/0.361/0.350/0.336/0.346 (ΔT 50→600) | 0.3648/0.3697/0.3611/0.3497/0.3355/0.3459 | **CONFIRMED to the 3rd digit** (also proves the temperature-arc's EOS byte-identity held) |
| — | two-room thermal mode period | 0.646 s, no decay | 0.646 s (thermal impulse ring-down) | **CONFIRMED exactly** |
| d | wave_absorb window: 0 clean; 0.002–0.01 violently unstable WITH fire; ≥0.02 clean | KE settled 212/918/1950/2893/111/76/28; umax peak up to 117 m/s; work-clamp 645–898; T-floor 3157–4512 | **at shipped k_wind_strip = 0.0: NO instability at any level.** KE settled 243.6/351.7/288.8/184.0/116.6/68.8/23.4; umax peak ≤ 11.7 m/s; **all rails zero** | **REFUTED on today's config** |
| d′ | same battery with `k_wind_strip = 0.5` restored (the 08-03 value) | — | damp 0.005: KE settled 1568, KE peak 19150, umax peak **76.9**, work-clamp 649, T-floor **4324**; damp 0.01: KE settled 2024, peak 34140, umax **106**, work-clamp 772, T-floor 3629; damp 0.0 and 0.02: clean (KE 201 / 120, zero rails) | **CONFIRMED — window is strip-gated** |
| — | rad_scale retune as alternative cause of the delta | — | `rad_scale = 1.0e-5` restored alone at damp 0.005: clean (KE settled 269, zero rails) | ruled out |
| e | damping throttles O2 supply / kills the fire (package-A re-opens) | fire alive 8.6% at damp 0.01 (with strip 0.5) | with strip 0.5: alive 9.8% at 0.005, 15.1% at 0.02 — reproduced. **With shipped strip 0.0: alive 100% at EVERY damp 0→0.1** | **CONFIRMED then / MOOT today** |
| f | heat-deposit amplifier "up to 20× gain" drives the §5 instability | suspected, unmeasured | bench window runs: measured gain ≤ **1.10×**, `heat_floor_hits = 0`; any bench operating point ≤ 2.2× | **REFUTED for the bench window** |
| f′ | same amplifier in the real game | — | Erik's dump: N at burning cells → 0.000, divisor floored ⇒ **20.0× engaged** across ~100 starved fire cells | **CONFIRMED in-game** |

Residual quantitative drift vs 08-03 (KE 243.6 vs 212 undamped; door-2 1220
vs 912) is consistent with the P-K0 dial promotions that P-F1b overrides do
not cover (notably the radiation re-anchor changing solid cooling); direction
and structure are identical everywhere.

## 3. Phase 2 — the committed two-room bench

`levels/bench_two_room/` is byte-identical to
`storm_probe.build_tworoom(12, 12, 0.5, door_h=1, crate_xy=(7, 7))` — the
geometry every number in the 08-03 analysis and this audit was measured on —
and `tools/bench_two_room.py` runs fire-in-two-rooms in one command, emitting
KE / max wind / mode period / rails + per-field sha256 digests (trajectory and
final state), with R6 provenance and an R4 BLIND header (tile 0.5 m not the
shipped 0.333; sealed hull; single crate; no units). Gates
(`tests/test_bench_two_room.py`, both PASS):
- fixture structure is non-vacuous (two air regions disjoint with the door
  sealed; exactly one door tile; one crate) — R1 applied to the scenario;
- two 120-tick runs are digest-identical, both layers.

This closes the "every fire bench is single-room" blindness (audit rules R4)
that let gate (f) of the thermal-mass ruling stay green while Erik's eye
caught the storming in a real level.

## 4. Phase 3 — the ledger

Scenario battery (bench, 4800 ticks = 200 s each; P-F1b dials):

| run | damp | strip | KE inj. by EOS | tail→gas eth | eos eth | N₂ pump | T_min gas | rails |
|---|---|---|---|---|---|---|---|---|
| baseline | 0 | 0.0 | 146.2 | −605 | +38 250 | **+125.2** | −0.02 | none |
| damped | 0.005 | 0.0 | 286.1 | −3 283 | +46 580 | **+121.9** | −0.03 | none |
| **window** | 0.005 | 0.5 | **637.6** | **+25 150** | **−35 860** | +0.25 | **−288.65 (floor)** | work-clamp 649, T-floor 4324 |
| damped-safe | 0.02 | 0.5 | 81.0 | −22 | +257 | +0.67 | −0.53 | none |
| pump-off | 0 | 0.0 (+`gases.smoke.decay=0`) | 267.2 | −1 656 | +7 805 | **0 to the LSB** | −0.02 | none |

("eth" = Σ N_bulk·T_abs over gas cells, c_v = 1, T_abs = T + 290; "KE" =
½·Σ N_bulk·|u|²; N₂ pump = bulk-N created inside the EOS pass = the trace-
decay credit. The full per-tick series are reproducible via §7.)

### 4.1 Momentum

Only the EOS pass writes velocity — measured zero for every other pass, every
tick. Sources/sinks inside it: the pressure kick (source), `wave_absorb`
(inert at shipped air 0.0), the B3c sponge band (ambient maps only — inert
here), and the |u| cap (never hit in any bench run). **At shipped dials a
sealed doored level has NO interior momentum sink at all** — B2's 2026-07
finding, still true, `k_drag` (`eos_refactor_decisions.md:163-169`,
sized ~0.02–0.05) still unbuilt, zero repo hits.

### 4.2 Mass — the smoke→N₂ pressure pump (new finding)

O2 is conserved by the EOS transport to the LSB over 4800 ticks (measured 0).
Bulk N₂ is not: the EOS pass created +125.2 counts (baseline) / +121.9
(damped) — **+42–43% of the room's entire 289-count inventory in 200 s**.
Chain, all sites verified:
1. `fire_simulation.cpp:289-310` — fire deposits smoke into its 4 neighbours
   at `smoke_emission = 0.8`/s·I, **ex nihilo** (the file's own header:
   plume+smoke+wall-burn are "fire SOURCES/sinks, not cancelling flux pairs").
2. Smoke rides p* at `trace_mass_scale = 0.02` — nearly pressure-invisible.
3. `physics_engine.cpp:498-516` — the P4 rule credits trace decay
   (`[gases.smoke] decay = 0.008`/s) to the inert-N₂ plane in the same cell,
   **at full pressure weight** — a 50× weight jump per decayed count.
Control: `gases.smoke.decay = 0` ⇒ EOS bulk-N creation exactly 0.
Consequence: a standing, fire-localized p* source (measured p_sum +149 over
the damped run) that keeps the door flow fed after the thermal transient; in
Erik's dump the same pump is visible as level-wide smoke_sum growing 0 → 1277
across the recording. Not the cause of the ringing (the pump-off run still
storms and still grows: KE final 523, +1.24/s) — but a genuine unintended
injection that inflates sealed burning rooms indefinitely.

### 4.3 The §5 window's anatomy (strip 0.5, damp 0.005)

- Wind-strip kills the fire at t ≈ 19.6 s (alive 9.8%). KE injected by the
  EOS **before** fire death: 91.6; **after**: 546 — the doc's "energy with no
  source", now attributed.
- With the fire dead and the mode over-damped (no recompression half-cycle),
  the hot zone's expansion pocket becomes persistent: T_min descends
  monotonically ~600 ticks (−91 → −288.65 game = T_abs ≈ 1.35 K), driven by
  step-4c compression work whose ±0.5 rail bounds the RATE, never the value.
- First work-clamp hit and first T-floor hit land the **same tick (1341)**;
  within 20 ticks KE bursts 92 → 2520 (umax 26 m/s, run peak 76.9) as the
  floored pocket's p* ≈ 0 makes a standing pressure deficit the kick converts
  to wind.
- The energy books after fire death: the tail injects **+25 150** eth into
  gas — Pass-2 conduction (air↔air AND solid↔air) pulling heat from the
  ambient-pinned hull ring (Pass-3's cooling law re-pins the hull toward
  ΔT=0 for free), i.e. **the ambient reservoir back-feeds the supercooled
  pocket every tick**; the EOS removes −36 020 as compounding expansion work
  and converts a slice into kinetic energy. That loop is why the "settled" KE
  in the window (1568–2024) exceeds anything the fire itself ever drove.
- Why only 0.002–0.01: at 0 the mode re-compresses each half-period (work
  alternates sign — no net spiral; T_min −0.02); at ≥0.02 flows die before
  cumulative work matters (T_min −0.53); in between, damping suppresses the
  restoring half-cycle while leaving enough flow to keep the pocket expanding.
  Fire death (strip) matters because a live fire re-heats the pocket
  (no-strip damp 0.005: T_min −0.03, stable at every level).

### 4.4 Erik's in-game blowup (the npz)

Different rail, same engine. Timeline (recorder snapshots; tick ids wrap per
round): fire grows 9 → 124 cells across rounds; burning cells exhaust their
O2 to exactly 0.000 while smoke saturates (1.0); flame cells sit nearly
evacuated (N_est = 290·P/(T+290) ≈ 0 — the reconstruction is exact where
smoke is thin). Then, at cell (y=27, x=61) from snap 469: T = 22 → 33 → 49 →
74 → … → 9310 → 16000, a clean geometric **×1.4957/tick for ~20 ticks** —
the 1 + T_WORK_CLAMP (0.5) rate rail, i.e. step-4c compression work on a
persistent CONVERGENT pocket, value-unbounded until the T_MAX_PHYS ceiling
(16000) catches it. 33 cells sit above T 10000 at the last snap (all gas
cells, none obstacle). P spikes 47.7 / 65.3 atm arrive when a gas slug
(N_est 48–65, i.e. ~50× ambient piled into one cell) meets the ceiling-hot
pocket — `p* = C·N·T_abs` — tripping the recorder (|ΔP| 62.3 > 50).
**The 1/N heat-deposit amplifier (combustion.cpp:798-803) is engaged at its
full 20× cap here** (divisor floored at n_floor_heat 0.05 across ~100 starved
fire cells) and plausibly seeds the hot pockets while O2 lasts — but the
geometric signature says the runaway multiplier itself is compression work.
Zero cells near the T_MIN floor in the whole dump: the in-game failure is the
HOT rail, and it happens at shipped dials (strip 0.0, air damping 0.0) — the
bench window is dormant, the hot runaway is live.

### 4.5 Conservation verdict (the ledger's headline answer)

| channel | verdict | evidence |
|---|---|---|
| momentum | **no injection bug; dissipation genuinely absent** | only the EOS writes u; no interior sink at shipped dials; sum_u drifts freely |
| O2 / bulk transport | **exactly conservative** | 0 LSB over 4800 ticks, every run |
| bulk N₂ | **unintended pump** (two deliberate rules composing) | +42%/200 s sealed; 0 with decay=0 |
| gas thermal energy | open by design (cool laws, conduction, books) — but the ambient reservoir **back-feeds runaways** | +25k eth into a supercooled pocket after the fire died |
| compression work | **rail-design gap**: rate-clamped (±0.5/tick), value-unbounded | ×1.4957/tick geometric climb to T_MAX_PHYS in-game; T_MIN spiral in the bench window; both counter-tracked but only after the fact |
| combustion heat deposit | correct per spec; 1/N gain real but bounded by its 0.05 floor at **20×**, reached only when a cell is fully evacuated | bench ≤ 2.2×; in-game 20× at O2-starved cells |

## 5. Phase 4 — candidate mechanisms (analysis only; decisions are Erik's)

**A. `wave_absorb ≥ 0.02` on air, as-is (zero new code).**
- Changes: `u *= 1 − wave_absorb·8·dt` every open cell (config
  `[materials.air] wave_absorb`, today 0.0). At 0.02 ⇒ KE e-fold ≈ 1.5 s.
- Evidence for: kills the ring-down (two-room KE settled 243.6 → 116.6 at
  0.02, 23.4 at 0.1); the §5 danger window is (i) below 0.02 anyway and
  (ii) gated on `k_wind_strip > 0`, which is now 0.0; clean rails at 0.02–0.1
  in every run, strip on or off.
- Side effects: **the O2-throttling worry has no measured teeth today** —
  fire alive 100% at every damp with strip 0.0 (the 08-03 fire-death effect
  was the strip term, not O2 supply). Residual risks: the intensity plateau
  could still shift (not measured here — compare `fire_tune_loop` curves at
  0 vs 0.02); damping is uniform, so quiet rooms lose their faint currents
  too; and IF wind-strip ever returns, the 0.002–0.01 band must be treated
  as forbidden (guard: config load-warn, or a test pinning air wave_absorb
  ∉ (0, 0.02)).
- Feel-test: two-room fire (this bench in-game), a breach vent, an explosion
  shockwave (absorb acts on all winds, not just fire's), smoke drift look.

**B. `k_drag` interior momentum decay (B2's named remedy, ~0.02–0.05/tick).**
- Changes: new dial + ~5 lines in the EOS kick loop (one Q16.16
  magnitude-first multiply, the absorb idiom at a fixed rate) — a *sim-path
  change*, CPU+CUDA, lockstep-gated.
- Evidence for: mathematically the same sink as A at k_drag ≈
  wave_absorb·8·dt (A@0.02 ≡ k_drag 0.0067); a separate named dial decouples
  "air viscosity for feel" from the material wave_absorb axis (walls use
  wave_absorb for shockwave absorption — A rides a dial that also has combat
  meaning).
- Against: strictly a superset of A's effort for the same physics; A already
  exists and is gated off by one config value. Build B only if Erik wants
  the axis separation or per-context tuning A can't express.
- Feel-test: same battery as A.

**C. Boundary/door-local friction (wall-adjacent or neck-only drag).**
- Changes: per-cell damping keyed off wall adjacency (the physical location
  of viscous loss in a real duct); sim-path change with a topology pass.
- Evidence for: targets the Helmholtz neck where KE concentrates (~100× per
  cell), leaves room interiors lively; physically defensible (no-slip walls).
- Against: the ledger shows the mode's energy lives in BOTH rooms' sloshing,
  not only the neck; more code, more CUDA surface, new calibration axis; no
  measurement here says uniform damping harms feel. Try only if A/B feel dead.
- Feel-test: door-flow look under venting; whistling-door scenarios.

**D. Fix the amplifier (combustion.cpp:798-803).**
- What it would change: cap the deposit's effective 1/N gain (e.g. divisor
  floor at a fraction of ambient rather than 0.05 ⇒ 20× → ~2-4×), or scale
  the deposit by N like the EOS step-1b deposit deliberately does.
- Evidence: NOT the bench-window driver (≤1.1× there) — fixing it does not
  buy damping. But in-game it multiplies heat 20× exactly at starved,
  evacuated fire cells, seeding the hot-rail runaway (§4.4). A physical
  reading: depositing combustion heat into a cell that holds almost no gas
  should heat the *neighbourhood's* gas, not multiply a phantom.
- Side effects: retunes flame-zone temperatures at low O2 → plateau/ignition
  chains move; must re-run the P-R3/P-F1b anchors. Gate: bench + goldens
  planned post-arc.
- Feel-test: big multi-cell fires in tight rooms (the blowup level), flame
  temperature look at smother.

**E. (New, from headline 3/4 — flagged for its own decision, not sized here.)**
Two rail gaps the audit exposes are upstream of ALL of the above and are the
difference between "storms" and "explodes":
- **step-4c compression work needs a VALUE guard, not only a rate guard**
  (or a div-u-consistency clamp at near-empty cells): it is the measured
  runaway multiplier on both rails.
- **the smoke→N₂ pump** wants a decision: either smoke stops decaying into
  full-weight bulk (decay to nothing above a pressure-neutrality budget), or
  the fire's smoke emission debits something real. `gases.smoke.decay = 0`
  is a one-dial experiment that removes the pump today (visual cost: soot
  never settles).
Fixing E does not damp the Helmholtz mode — A/B/C still decide the feel
question — but E is what turns Erik's screenshot-grade blowups into bounded
events.

## 6. Refuted/confirmed summary vs 2026-08-03

Confirmed: the storming mechanism (geometry + zero dissipation), the door-
width resonance, drive-independence of the settled state, the mode period
(0.646 s exactly), the §5 window's existence *under the 08-03 config*, the
compression-work compounding hazard (now measured on both rails), the
amplifier's existence and its 20× ceiling (in-game).
Refuted / corrected: the §5 window on TODAY'S config (dormant — strip-gated,
strip now 0.0); the amplifier as the window's driver (≤1.1× there); damping's
fire-killing side effect at today's dials (100% alive at every level);
"energy appearing with no source" (attributed: ambient reservoir via hull
conduction + cooling law, converted by the kick against a floored-T pressure
deficit).
New since 08-03: the smoke→N₂ pressure pump (+42%/200 s, controlled to 0 by
one dial); the in-game hot-rail blowup anatomy (×1.4957/tick, T_MAX_PHYS,
50× gas pile-up, 65 atm spikes).

## 7. Reproduction (all read-only; overrides restored on exit)

```
# build (this tree):        cpp\build_cpu_data.bat
# bench smoke + digests:    conda run -n data python tools/bench_two_room.py --ticks 240 --pf1b
# suite gate:               conda run -n data python -m pytest tests -q      # 36 known reds, set-unchanged

# Phase 1 battery (numbers in §2):
conda run -n data python tools/storm_probe.py --mode fire --geom room    --seconds 200 --pf1b
conda run -n data python tools/storm_probe.py --mode fire --geom tworoom --seconds 200 --pf1b
conda run -n data python tools/storm_probe.py --mode hotplate --geom room --seconds 40 --dT 300
# window, dormant vs restored:
conda run -n data python tools/bench_two_room.py --ticks 4800 --damp 0.005 --pf1b
conda run -n data python tools/bench_two_room.py --ticks 4800 --damp 0.005 --pf1b --set k_wind_strip=0.5

# Phase 3 ledger (per-pass totals + per-tick npz series):
conda run -n data python tools/storm_ledger.py --ticks 4800 --damp 0.0   --pf1b --out ledger_d0.npz
conda run -n data python tools/storm_ledger.py --ticks 4800 --damp 0.005 --pf1b --set k_wind_strip=0.5 --out ledger_window.npz
conda run -n data python tools/storm_ledger.py --ticks 4800 --damp 0.0   --pf1b --set gases.smoke.decay=0.0 --out ledger_pump_off.npz
```

Erik's dump stays uncommitted input data
(`debug_blowup_20260814_015714.npz`, repo root of the `thermal-mass-axis`
worktree). Momentum analysis of a future in-game blowup needs a recorder
session constructed with `fields` including `wind_x`, `wind_y`.

## 8. Source index

Window reproduction: §2 battery above · kick/absorb/sponge/caps:
`cpp/src/eos_solver.cpp:618-724` · compression work + rate-rail:
`eos_solver.cpp:726-784`, warning `eos_solver.h:168-186`, `T_WORK_CLAMP=0.5`
`eos_solver.h:180` · amplifier: `cpp/src/combustion.cpp:774-808`
(`n_floor_heat=0.05`, `c_v=1` config.toml:154-157; `H_fuel=4.0` :622) ·
smoke ex nihilo: `cpp/src/fire_simulation.cpp:289-310`
(`smoke_emission=0.8` config.toml:328) · decay→N₂ credit:
`cpp/src/physics_engine.cpp:498-516` (`[gases.smoke] decay=0.008`
config.toml:747) · trace weight: `trace_mass_scale=0.02` (eos_solver.h) ·
conduction/cooling passes: `cpp/src/temperature_solver.h:43-67,239-249` ·
strip term retired: config.toml:325 (`k_wind_strip=0.0`, P-K0) vs 08-03's
0.5 · k_drag decision: `docs/eos_refactor_decisions.md:163-169` · recorder
format: `src/simulation/recorder.py` · 08-03 analysis:
`docs/fire_atmosphere_oscillation_analysis_2026-08-03.md` · audit rules:
`docs/audit_lessons_and_rules_2026-08-04.md`.
