# The storming atmosphere — analysis + measurements (2026-08-03)

**Status: analysis only. No simulation code was changed.** Written by Claude
overnight 2026-08-03 in answer to Erik's seed (`docs/breach_todo_2026-08-03.md`,
on main). Every number below was measured tonight with a harness-level probe
(scratchpad `storm_probe.py`) that drives the shipped engine and changes
nothing; dials move through the existing `apply_overrides`/`restore_overrides`
path. The plan is made together after Erik reads this; nothing here is a ruling.

Erik's report: *"fire kind of works, but fire now perturbs the atmosphere so
much that it oscillates — it's not looking natural."* His theory: we pin a whole
1/3 × 1/3 × 2.5 m tile at ~1000 K when a real flame fills only a fraction of
that volume, so the cell-average temperature — and with it the expansion drive —
is overestimated.

---

## THE ANSWER IN FIVE LINES

The storming is **not** caused by the fire being too hot, and **not** by the
flicker. It is caused by **connected geometry with no momentum dissipation**: a
door between two rooms creates a Helmholtz mode that the engine damps not at all,
so the fire's *startup transient* rings forever. Measured: the same P-F1b crate
fire leaves a single sealed room at **0.12 m/s** after 200 s, and a two-room
level at **6.25 m/s with kinetic energy still growing**. Erik diagnosed this
himself during the EOS refactor (B2, "rooms shouldn't oscillate forever") and the
fix he queued then — a velocity-drag dial — was never built.

---

## 0. State facts (read first)

- **The build Erik played is P-F1b**, `origin/pf1b-recalibration` (`4133512`),
  **unmerged**, FEEL-ADJACENT / HUMAN-TEST pending. Its doc
  (`fire_recalibration_2026-08-02.md`) lives only on that branch. This worktree
  (`thermal-mass-axis`, HEAD `fcfcd01` = P-F1a) still carries the frozen,
  fire-dead config. P-F1b is **config + tools + tests only — no engine code** —
  so its 13 dials can be reproduced anywhere via overrides. All measurements
  below use them.
- Dials used: `k_grow 2.0`, `k_die 0.008` (r = 0.004), `I_cap_per_avail 14.0`,
  `ignition_seed 0.12`, `ignition_to_ext_delta 200`, `fire_T_span 180`,
  `wall_damage 0.03`, `T_emit_gate 310`, H_bed 2.900e5 (`H_BED_M 18125`,
  `H_BED_SHIFT 4`), `cool_shift 13` on wood/furniture/kindling.
- Probe geometry: hull-sealed rooms, 0.5 m tiles, 12×12 interior; "two rooms"
  = two 12×12 rooms sharing a partition with a 1-tile door. 24 tps.
- Metric v0 (per tick): kinetic energy `Σ(u_x² + u_y²)` over open cells in
  (m/s)², max |u|, the pressure transient |P − P_prev| (mean/max), and the five
  EOS rail counters.

## 1. The mechanism as built

Rung B is live: the Kwatra semi-implicit compressible ideal gas
(`cpp/src/eos_solver.h:1-14`). Per tick (dt = 1/24 s): fire heat and combustion
deposits raise `T` → `p* = C·N_total·(T + 290)` (`eos_solver.cpp:565`) → an
implicit multigrid pressure solve (acoustic-CFL-free, V(2,2)×2, frozen) → the
kick `u -= dt·K·∇P/N̂` (K ≈ 64,286) → semi-Lagrangian advection + donor-cell N
flux → compression work `T ← T − (γ−1)·T·div(u)·dt`.

**Erik's coupling note, corrected on one point.** Erik wrote: *"they also consume
O2 but I don't think that changes the pressure anywhere."* It does, locally and
by design. The O2 leaves the **donor** cells (the R=2 draw ring) and the
combustion products appear at the **flame** cell, so `N_total` is conserved
*globally but not per cell* — `combustion.cpp:755-760` states the intent
outright: *"the deficit they leave behind is what draws fresh air in."* Since
`p* = C·N·T_abs`, that is a per-tick pressure dipole at the fire, modulated by
intensity. Soot is additionally weighted at `trace_mass_scale = 0.02` while the
O2 removed carries full weight, so combustion is also mildly deflationary in
pressure terms. Not a large effect next to the thermal one, but not zero, and it
is a second time-varying pump.

## 2. ★ FINDING 1 — the engine runs two different Kelvin maps

| Consumer | Map | Where |
|---|---|---|
| Radiation books E°(T)⁴, renderer, hover readout | **K = 293 + 2·T_game** | hardcoded as the literals `297 + 8t`, `raycaster.cpp:59`; separately as config keys at `config.toml:844-845`; reimplemented again in 4 test files |
| **EOS pressure — i.e. the expansion** | **T_abs = T_game + 290** (T₀ = 290, slope **1**) | `eos_solver.cpp:542,565`; `config.toml:469-470` |

The same field means **893 K to the radiation code and 590 K to the gas**. The
independent C++ audit found this too and called it "a physics decision for Erik,
not a cleanup."

Consequence for Erik's theory: measured against the books' own Kelvin story, the
EOS **already applies exactly half the temperature excess** — a de-facto flame
volume fraction of φ = 0.5, undocumented and unintentional. An honest
column-average for a 12.7 kW fire in a 0.111 m² × 2.5 m tile (flame height
≈ 0.27 m by Heskestad, plume decaying as z^−5/3 above) is **≈ 150–250 K of
excess**; the EOS applies ≈ 300 K; the books would say 600 K. **So Erik's
direction is right, but the residual overdrive is only ~1.2–2×, not 3×** — and
§3 shows that overdrive is not what makes the air oscillate.

Note also that the sizing ruling already names a flame fraction — *"φ ≡ 1,
flame_lift ≡ 0"* — but that φ rides **E_emit (radiation)**, not the EOS. Erik's
tile-average idea, formalised, is a *new* expansion-side fraction, and it is
free of book fallout precisely because the two maps are already separate.

## 3. ★ FINDING 2 — steady drive does not storm; the flicker does not pump

**T1, hotplate:** one tile pinned at ΔT, no fire, sealed 12×12, 40 s.

| ΔT (game) | 50 | 100 | 200 | 300 | 400 | 600 |
|---|---|---|---|---|---|---|
| KE, initial transient | 1.26 | 2.00 | 5.97 | 10.0 | 15.8 | 30.2 |
| **KE, settled (last decile)** | **0.365** | **0.370** | **0.361** | **0.350** | **0.336** | **0.346** |
| max \|u\| final (m/s) | 0.17 | 0.11 | 0.12 | 0.09 | 0.12 | 0.13 |

The settled disturbance is **flat across a 12× change in drive**. Only the
*transient* scales. A hot tile, however hot, does not keep the air moving.

**F3, flicker amplitude** (two rooms, 10 s period, 120 s): amplitude 0 → KE 430;
amplitude 15 → 467; amplitude 45 → 430; amplitude 90 → 452. **A flicker of ±90
game units storms exactly as much as no flicker at all.** Frequency response
(1 s to 60 s periods) is likewise flat.

This is decisive: **Erik's φ fix, on its own, will not stop the oscillation.** It
reduces the size of each kick, not the fact that kicks never die. It remains
worth doing for realism and for peak-event violence — but it is not the cure.

## 4. ★ FINDING 3 — the cause: connected geometry with zero dissipation

**T2, ring-down.** An impulse, then no drive at all:

| case | KE retained after 30 s | e-fold | mode period |
|---|---|---|---|
| velocity impulse, one sealed room | 3.8 % | 12.4 s | 0.81 s |
| thermal impulse, one sealed room | 0.4 % | — | (no clear mode) |
| **thermal impulse, two rooms + door** | **58.6 %** | **∞ (no decay)** | **0.646 s** |

Erik measured this mode during the EOS refactor at **15 ticks = 0.625 s**
(`eos_refactor_decisions.md:163-169`). Tonight it measures **0.646 s** — the same
mode, one tick apart, still undamped. His words then: *"rooms shouldn't oscillate
forever… unphysically undamped because the momentum update has no viscosity or
drag."* The remedy he queued, a `k_drag` velocity-decay dial (~0.02–0.05), **was
never built** — zero code hits repo-wide.

**T0, the real fire.** Same P-F1b crate, 200 s, no wind forcing:

| | one sealed room | two rooms + door |
|---|---|---|
| KE peak | 10.1 | 371 |
| KE final | 0.48 | **225** |
| KE retention | 2.4 % | **57 %** |
| max \|u\| peak | 1.41 m/s | 9.22 m/s |
| **max \|u\| final** | **0.12 m/s** | **6.25 m/s** |
| KE trend at 200 s | decayed | **still growing** (decay rate −0.0004/s) |
| EOS rails | all zero | all zero |

A campfire-sized fire produces a **permanent ~6 m/s indoor wind** as soon as the
room has a door to somewhere. The single-room benches the whole fire arc was
tuned on — the still-air arena, the sealed room bench — **cannot show this**.
That is why gate (f) of the thermal-mass ruling did not fire (bench report §6, 12
operating points, std(P) 1–3×10⁻⁵) while Erik's eye caught it immediately in a
real level. The gate was armed on the wrong geometry.

Why the door matters: KE ∝ u², and continuity forces the whole room's volume flux
through one tile, so neck velocity is ~10× the room velocity and neck KE ~100×
per cell. The mode is a genuine Helmholtz resonator (gas inertia in the neck,
compressibility either side) — not an acoustic mode, which the implicit solve
filters (a 6 m room's acoustic period is ~0.04 s, sub-tick).

**And it is worst at exactly the geometry the game uses.** Same fire, same rooms,
varying only the door width:

| door width (tiles) | 1 | **2** | 3 | 6 |
|---|---|---|---|---|
| KE (settled) | 212 | **912** | 46.6 | 1.5 |
| max \|u\| peak (m/s) | 9.2 | **11.0** | 3.1 | 1.5 |
| max \|u\| final (m/s) | 6.25 | **10.34** | 0.79 | **0.13** |

The response peaks at a **2-tile door** and collapses once the opening stops
being a constriction — at width 6 the two rooms are effectively one space and the
air is as calm as the single-room case. This is the Helmholtz signature (a
resonator needs a neck), and it confirms the mechanism independently of the
particular level: **a realistic ship door is the worst case, and a wide-open
archway is nearly free.**

## 5. ★ FINDING 4 — the obvious fix is not safe as-is

The damping lever Erik half-remembered **does exist and needs no new code**:
`u *= (1 − wave_absorb·absorb_strength·dt)` runs on every open cell every tick
(`eos_solver.cpp:632-642`), inert in rooms only because `[materials.air]
wave_absorb = 0.0` (`config.toml:979`, `absorb_strength = 8.0`).

Without fire it behaves beautifully — monotonic, clean, zero rails at every level
(two rooms, 120 s): pulse drive KE 430 → 405 → 367 → 286 → 200 → 107 → 40 across
wave_absorb 0 → 0.1; jet impulse 225 → 212 → 170 → 123 → 125 → 79 → 24.

**With a real fire it opens an instability window:**

| air wave_absorb | 0.000 | 0.002 | 0.005 | 0.010 | 0.020 | 0.050 | 0.100 |
|---|---|---|---|---|---|---|---|
| KE (settled) | 212 | 918 | 1950 | **2893** | 111 | 76 | 28 |
| max \|u\| peak (m/s) | 9.2 | **92.4** | **117.5** | **110.0** | 8.9 | 7.7 | 5.5 |
| work-clamp hits | 0 | 898 | 843 | 645 | 0 | 0 | 0 |
| **T-floor hits** | 0 | **4512** | **4359** | **3157** | 0 | 0 | 0 |

At 0.002–0.01 the sim reaches 110 m/s winds and drives gas onto the `T_MIN` floor
(T_abs → 1 K) thousands of times. In the 0.01 run **the fire is out by t ≈ 17 s
and the kinetic energy still grows 30× over the following 180 s** — energy
appearing with no source.

The mechanism is consistent with the failure mode the EOS header already warns
about (`eos_solver.h:168-186`): compression work `T ← T·(1 − k)` is multiplicative
and the `T_WORK_CLAMP` rail bounds the per-tick *rate*, never the *value*, so a
persistent expansion pocket compounds. Once T reaches the floor, `p*` collapses
to ~0 and the resulting gradient produces the 110 m/s kick.

Two consequences, both important:
1. **This requires combustion.** The no-fire controls above are clean at every
   damping level. So it is a fire↔atmosphere coupling, not an EOS defect —
   H-F confirmed, and the likely amplifier is the gas heat deposit
   `ΔT = deposit/(max(N, 0.05)·c_v)` (`combustion.cpp:770-796`), which *divides
   by a density the fire itself is driving down* — up to 20× gain, with no
   absorption scaling (unlike the EOS step-1b deposit, which deliberately
   cancels N).
2. **Air damping is not a free visual dial.** It also throttles convective O2
   supply: in the damped runs the fire *died faster* (alive 8.6 % of the run at
   0.01 vs 57 % undamped). The entire package-A sizing calibration — 12.7 kW,
   890 K — was measured at zero damping. Turning damping on re-opens it.

**Caveat, stated plainly:** this instability is reproducible in my probe but has
not been independently reproduced. It should be verified before anyone acts on
it. It is reported because it changes what the obvious next step should be.

## 6. Verdicts on Erik's six threads

1. **"Why fix the map in advance? Fit k afterwards."** For the *books*, k is now
   load-bearing (E° is baked as `297 + 8t`, `rad_scale` absorbs the K⁴ scale, four
   test files pin it), and a 2× map change moves T⁴ by ~16× — the wrong knob.
   But the **EOS slope was never calibrated against anything** and is already a
   separate map. Fitting *the expansion scale* a posteriori — exactly Erik's
   proposal — is free of book fallout. That is the φ_exp dial.
2. **Tile-average overestimate.** Right in spirit; quantitatively softened
   (de-facto φ is already 0.5; residual ~1.2–2×). Worth doing for honesty and
   for event violence — but §3 shows it will not stop the oscillation.
3. **Air damping.** Found; it is Erik's own unbuilt `k_drag`, and a zero-code
   equivalent already ships. But §5 says do not just turn it on.
4. **O2 radius 1 vs 2.** Keep 2. R=2 costs little (40→70 registers, ~50 %
   occupancy, zero spills) and delivers 2.26–2.48× the power; the whole
   package-A calibration rests on it. R=3 is the expensive step, already
   declined.
5. **Drop gridsearch; tune one at a time.** There was no gridsearch. The 100–250×
   `k_grow`/`k_die` gap is a *derived* point from the P-R3 capacity law
   (`fire_tune_loop.py:449-454`; P-F1b §3.1: the old 0.5 ratio "demanded a =
   0.333 — 3.6× the maximum the atmosphere can supply… arithmetically fire-dead").
   So the ratio's *placement* is defensible. What was never bench-gated is its
   *dynamics* — and note `k_wind_strip·W` is still 4× `k_die` at the operating
   point, i.e. the atmosphere holds the dominant term in the fire's death
   equation. That is the coupling §5 just bit us with.
6. **Diagnostics.** Mostly already built — and `fire_tune_loop.py`, the tool that
   draws exactly the intensity/temperature/room-O2 panels Erik asked for, **is
   currently broken**: it still passes the retired `k_fire_heat` dial and dies
   with a `KeyError` before running (see the codebase audit). One-line fix.

## 7. What I recommend we decide together

1. **Ratify the two-scale doctrine.** Books-Kelvin (k = 2, calibrated, untouched)
   vs expansion-Kelvin (EOS, today de-facto φ = 0.5). Name it, write it into
   canon, and decide whether the expansion fraction becomes an explicit `φ_exp`
   dial. This is a physics decision, and it is Erik's.
2. **Treat the undamped-room problem as its own arc, not a fire dial.** It
   predates the fire work (B2, 2026-07), it is geometric, and every fire number
   we have was measured on a geometry that cannot show it. Suggested order:
   verify the §5 instability independently → understand the 1/N deposit
   amplifier → then choose a dissipation mechanism.
3. **Add a two-room fixture to the bench set before tuning anything further.**
   Single-room arenas are structurally blind to the phenomenon Erik is objecting
   to. This is cheap and it is the prerequisite for judging any fix.
4. **Do not merge P-F1b on the strength of the single-room curves alone** — or
   merge it knowing the storming is a separate, older problem it did not cause.
   Erik's call; the evidence says P-F1b is not the culprit.

## 8. Reproduction

Probe + batches (scratchpad, not committed):
`storm_probe.py` (modes: thermal / jet / hotplate / pulse / fire; geometries:
room / tworoom / arena), `batch_storm.py`, `batch_pulse.py`, `batch_fire.py`,
`batch_rails.py`, `inspect_fire.py`. The CPU module was rebuilt from this branch
(`cpp/build_cpu_data.bat`) because the checked-out `.pyd` predated P-F1a.

## 9. Source index

Erik's seed: main `docs/breach_todo_2026-08-03.md` · P-F1b:
`origin/pf1b-recalibration` `docs/fire_recalibration_2026-08-02.md` · EOS chain:
`cpp/src/eos_solver.cpp:436-786`, `eos_solver.h:136-244` · the two maps:
`raycaster.cpp:55-68` vs `eos_solver.cpp:542,565` + `config.toml:469-470,844-845`
· damping: `eos_solver.cpp:632-663`, `config.toml:979,483` · **B2, the original
finding: `docs/eos_refactor_decisions.md:163-169`** · fire→gas couplings:
`combustion.cpp:307-420,613-617,755-798` · compression-work compounding warning:
`eos_solver.h:168-186`, `eos_solver.cpp:746-759` · flicker record:
`fire_realism_design_plain_2026-08-02.md` §Problem 2, P-F1b §4(b,d) · strip term:
P-F1b §2.3, `fire_tuning_plan_2026-07-22.md` §5.3 · gate (f) that did not fire:
`thermal_mass_eos_ruling_2026-07-30.md` §1-A3 + `thermal_mass_axis_bench_report_2026-07-30.md` §6.
