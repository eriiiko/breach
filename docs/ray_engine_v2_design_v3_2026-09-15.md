# Ray engine v2 — design v3 (2026-09-15)

> **Status:** design, critiqued three times (physics, engine integration,
> determinism/CUDA). **Critique 3 is folded in** (2026-09-15 evening; every
> change is in §0 rows 25–31 and Appendix C, and the affected sections are
> edited in place). Nothing built. **Next: P0.**
> This supersedes `ray_engine_v2_design_v2_2026-09-13.md` wherever they disagree;
> §0 lists every disagreement and its reason. Every file:line below was verified
> against the tree at `fire-12` `5a252f5` on 2026-09-15; critique 3 re-verified
> them at `968f62a`.
>
> **Depends on:** `ray_engine_v2_survey_2026-09-12.md` (rulings R-A..R-S) ·
> `ray_engine_v2_critique_1_physics_2026-09-12.md` (12 changes, resolved in v2,
> not re-litigated) · `ray_engine_v2_critique_2_engine_2026-09-13.md` (30
> entries, resolved here — the map is Appendix A) ·
> `ray_engine_v2_reach_and_papers_2026-09-13.md` · `ray_engine_v2_scheme_study_2026-09-13/report.md` ·
> `fire_radiation_assumptions_2026-09-11.md` · `docs/papers/README_ray_engine_v2_2026-09-13.md` ·
> Erik's five answers of 2026-09-15 to the handoff's §5 (recorded in §0 rows 1–6).
>
> **Instruments**, all in `docs/ray_engine_v2_scheme_study_2026-09-13/`:
> `sweep_ref.py` (float reference of the paired scheme, both transport steps,
> leak, ambient inflow), `reach_and_leak_study.py`, `flashover_study.py`,
> `contact_faces_study.py`, `stability_study.py`, `smoke_heat_capacity_study.py`
> (§4 corrects §§1–2 of the same file), `cascade_fix_study.py`,
> `reflections_study.py`, `smoke_study.py`, and the real-scene renders.
> **There is no integer reference yet** — that is P0 (§11), and it is first.

---

## 0. What changed since v2, and why

Rows 1–6 are Erik's rulings of 2026-09-15. Rows 7–20 resolve critique 2. Rows
21–24 are corrections I found while doing the code archaeology the critique
asked for; rows 22 and 23 changed the arithmetic and were marked derived, not
measured — **critique 3 then measured them with its own integer probe and they
hold** (`ray_engine_v2_critique_3_determinism_cuda_2026-09-15.md`, "What I
could not break"). Rows 25–31 are the fold of critique 3 and Erik's three
rulings on it, the same evening. Rows 32–34 are P0's findings, the same night
(`report_p0.md`, merged at `40f2479`): the integer reference is built and every
gate passes, and it corrected three of this document's numbers and one of its
number types. P0b, the same night, measured the Q24 form and both α floors; its
corrections are folded in place (rows 32 and 34, §2.8, §12 item 4). Rows
36–38 are P1's findings (2026-09-16, merged `2471fc1`): the sweep is built,
in shadow, and holds the reference integer for integer. Rows 41–43 are P2b's
(2026-09-16, merged): the calibration is derived, and it cancels out of the
reach — it sets the rate, not the range.

| # | v2 said | v3 says | why |
|---|---|---|---|
| 1 | "the replacement lands *here*" and, two lines later, a new `radiation_sweep.*`; the old raycaster survives for beams and the legacy march | **A new TU, `cpp/src/radiation_sweep.*`, and the old `raycaster.*` + `cuda_raycaster.*` are ARCHIVED** — deleted from the tree at P6 (the last patch that reads them), kept in git history, with a pointer in `docs/archive/` and a supersession banner on `docs/architecture/engine/08_ray_engine.md`. Nothing survives beside the new engine | **Erik, 2026-09-15**: *"archive the old raycaster.cpp and replace it with the new design … I am not against retiring the current iteration — if we break anything perhaps we just fix it."* Verified: the weapon beam never used it (`src/simulation/combat.py:1057-1200` is its own integer tile march over `gmap.has_los`), and neither does vision (`vision.py:88-103`, Bresenham). Nothing outside `raycaster.cpp` calls `march_ray*`. The name `radiation_sweep` Erik blessed: light and heat are both radiation |
| 2 | (nothing) | **§4 — the producer contract**: what any radiation producer exposes, what the old raycaster exposed, and where each of its 31 members goes | Erik: *"perhaps we need to formalize what raycasters should expose going forward … this part of the architecture I'm happy to leave with you"* |
| 3 | `rad_flux` recalibration is a feel ruling, P5 HUMAN-TEST | **Derived, not felt.** `rad_flux` becomes absorbed energy in the fold's own currency; the `[combat]` flux→damage dials are re-derived from physical units on a bench; a HUMAN-TEST remains the *merge gate* of the flip, not the calibration method | Erik: *"we want radiation to radiate physically — which is possible now, it will just fall out — then the tuning will be more in the heat capacity of items"* (§9) |
| 4 | Gas density double-count: unruled | **Radiation into gas rides the density-proportional law — in the extinction coefficient**, the one place the stream is debited, so what thin smoke fails to absorb is *transmitted*, not destroyed. The Pass-1 v2.4 factor is therefore not applied a second time to `rad_net` (§6.3) | Erik: *"it must ride the density proportional law"*. The two candidate sites are not equivalent: the v2.4 site would over-debit the stream and destroy the difference (counted, but wrong). If Erik meant the *code site* rather than the *law*, §6.3 states the alternative and its cost |
| 5 | The 16-light cap deleted at P3 | **Deleted at P6**, the patch where every emitting cell becomes free. Deleting it earlier would put every burning tile on the old per-source CPU march (34 758 DDA steps per frame *at* the cap, survey §5.1) | Erik: *"16 light cap is totally irrelevant now — each tile is an emitter and cost free … this cap should be removed!"* The cap was a cost decision; the sweep dissolves the cost, so removal rides the sweep |
| 6 | Foliage `heat_atten = 0.0` unruled; #61 "fix the property" | **Out of scope.** Trees are not ready-made things; P2 calibrates ONE standard burnable furniture row and nothing else. Foliage stays `0.0` — a tree tile is radiation-transparent, neither absorbs nor emits, and can only ignite by conduction or a direct deposit — and that is accepted until the tree rows are authored | Erik: *"consider trees as not ready made things, so we don't expect them to behave. I want to tune one piece of standard burnable furniture first"* |
| 7 | `(stream * a) >> 16` with `a` undeclared; `heat_atten` is float | **The integer extinction planes are declared** (§3): `heat_atten_q` (static, Q16) and `dyn_heat_atten_q` (per-tick MAX stamp), a new boundary module `optics_fixed.py`, quantized at the material-table door | critique BLOCKING 1 |
| 8 | Clamp: `after the material update: if E°[T] > Φ: T ← E°⁻¹(Φ)`, home unstated | **Inside the temperature solver's Pass-1 fold**, solids in the block that already books `e_solid_deposit_sum`, gas through the seam — and in its correct form `T ← min(T_after, max(T_before, E°⁻¹(Φ)))` (§2.8) | critique BLOCKING 2 — and the bare v2 form would have clamped a burning crate (row 21) |
| 9 | Φ undeclared | **`rad_fluence`**, int64, per-tick, born through the residency path, one clear line (§3) | critique BLOCKING 3 |
| 10 | P1's gate list omits the maximum principle | In P1, with the evaluation point (immediately after the clamp, on the radiative sub-step alone) and a non-vacuous over-driven scenario (§2.9) | critique BLOCKING 4 |
| 11 | §5.3 "the sweep is a booked writer on the seam" (fixed in v2, verify) | Verified and restated: the sweep writes `rad_net`; the Pass-1 `ts[i]` mask is what opens; the counters are the existing group-1 ones (§6.3) | critique BLOCKING 5 |
| 12 | Identity without a unit exit | **The body share.** Every cell carries `a_mat` (emits and absorbs) and `a_dyn ≥ a_mat` (the stamped total); the difference absorbs into `rad_flux` and never emits. The identity is `Σ rad_net + Σ rad_flux + Σ rad_amb ≡ 0`, gated at P1 with a seeded body (§2.3) | critique BLOCKING 6 |
| 13 | `ret` with two Q16 factors and one shift | Already fixed in v2's listing (`amb_m` first); kept, and the leak channel is now booked into `rad_amb` (§2.7) | critique item 7 |
| 14 | "Verified in `sweep_ref.py`" for an int64 claim | `sweep_ref.py` is float. **P0 = the integer reference**, and it is the arc's first deliverable | critique item 8 |
| 15 | "One divide per cell … exact if the operands are pinned" | **One exact kit floor-division** (`fixed_point.h:562 floordiv_q`), door 1, and α never needs computing: `f = T_abs / max(T_abs + 2L, 4L)` (§2.8). "Doors 1 and 2 only" is now true | critique items 9, 5a |
| 16 | New TU not on `/fp:strict`, not on the ratchet; call site unstated; resident path unaddressed; E° owner unstated | All stated (§8.1, §8.2, §11 P1): the TU joins both lists at P1; the sweep is **step 2b of `PhysicsEngine::step`** (after fire, before the temperature pass) and both `cast_fire_heat` call sites die at P3; §A residency habits apply; E° moves to `cpp/src/emissive_table.h`, owned by `PhysicsEngine`, read by the sweep and by Pass 1 | critique items 10–13 |
| 17 | "four closure groups, no fifth" | **Six** groups (`tests/test_thermostat_books.py:70-92`); the CLAUDE.md row is amended at arc close (§6.3) | critique item 15 |
| 18 | "a Recorder DTYPE-class contract extension"; "digested? yes" | The Recorder never records these planes. The int64 widening touches `GameMap` dtypes, four pybind signatures, the CUDA header and four test harnesses (§3); `E°` is already int64; per-tick planes are not digested (§8.3) | critique items 18, 25 |
| 19 | `rad_amb` "a global scalar or a plane" | **A plane — the ambient ledger**: net per cell (out − in), sky faces at boundary cells, the ceiling channel at every cell. Engages `gamemap.py:487-494`'s order-freedom argument; the consumer list is §11's table | critique items 19, 6d |
| 20 | Light: float, transport unstated, clock unstated; the rules channel reads a float plane | **Light is an integer payload on the same sweep, on the sim clock, in C++/CUDA.** The eye field is the *dequantized* integer field plus render-only post; the rules channel reads the same integer extinction planes. **This amends R-P's "light in GLSL" half and retires the R-S build question for light** — Erik to confirm (§7, §12) | critique items 23, 24 |
| 21 | Clamp `if E°[T] > Φ: T ← E°⁻¹(Φ)` | `T ← min(T_after, max(T_before, E°⁻¹(Φ)))`. The v2 form is what `stability_study.py:271` measured, where radiation was the *only* heater. In the engine a burning crate's T comes from combustion and sits far above the radiation it receives; the bare form would cool it toward ambient every tick | found in archaeology; §2.8 |
| 22 | Fleck factor multiplies the whole emission `E°[T]` | **It multiplies the excess `E°[T] − E°[0]`.** With `f < 1` at ambient (it is 0.995 at the shipped scale) a cell at ambient in an ambient field gains 0.5 % of its exchange every tick and gate 2's *exact* per-cell fixed point is impossible. Damping the excess restores it exactly — `x·f + x·(ONE−f)` is not needed; the ambient part is simply not damped — and it is the correct linearisation: the Fleck derivation damps `∂E°/∂T·ΔT`, and a constant has no derivative | **derived, not yet measured** — P0 confirms; §2.6, §2.8 |
| 23 | Push form (each cell writes two downwind faces) | **Gather form**: each cell reads its upwind neighbours' stored outflow and recomputes the split from the same integers; out-of-grid reads return the ambient outflow constant. Algebraically identical to `sweep_ref.py`'s push, and it makes the CUDA twin's bit-identity *structural* — one formulation, two backends, no atomics inside an ordinate | §2.3, §8.2 — **derived**; P0 cross-checks against the float push reference |
| 24 | P3 = flip + all deletions; P5 = units absorb | **Units absorb at the flip.** `rad_flux` has one consumer (marine burn damage) and its only writer dies with the old law; a flip without the body share leaves marines un-burnable. Render deletions move to P6; the old-law test surface (46 tests, four files) is *deleted* at P3, because its replacement gates are written at P1 against the shadow sweep | critique items 21, 22, 30; §11 |
| 25 | The body share "absorbs into `rad_flux`, never emits" | **A body re-emits at the ambient level**, with the material's own arithmetic: `emitted_body = (amb_m · b_i) >> 16`; `rad_flux` books the **net**, `abs_body − emitted_body`, and is signed | **Erik, 2026-09-15 evening**, on critique 3's probe: a pure sink shadows the *sky*, so a marine standing beside an ambient wall cooled the wall by 775 counts a tick and tripped the low rail (critique 3 §4f). With ambient re-emission the ambient fixed point is exact with bodies present and a body shadows only the excess above ambient — the fire, not the room |
| 26 | §3's widening list: two bindings and one CUDA header "die at P3" | **Five more 32-bit readers of `rad_net` survive P3** — the temperature solver's signature, `step_tail`, two bindings, and the CUDA *temperature* twin — and pybind's default `forcecast` makes a missed one silent (a truncated int32 copy, no error). All widened at P1, the two surviving bindings without forcecast so a stale caller fails loudly | critique 3 BLOCKING §6a |
| 27 | The clamp in Pass 1; P4 ports only the sweep | **The clamp's GPU twin lands in the same patch that makes it live**: solids at P3, gas at P5, inside `cuda_temperature.cu`, with `rad_fluence` and the `E°` table uploaded, an `FP_HD` inverse, a third hit counter and pinned energy slots | critique 3 BLOCKING §7b; **Erik**: twin inside P3, not P4 before P3 — a solver and its twin change together |
| 28 | `dyn_heat_atten_q` not digested ("a pure function of digested state") | **Enters `DIGEST_FIELDS` at P3**, spec v6, in the same commit as P3's re-baseline (zero marginal cost then); `heat_atten_q` stays out on the static-projection precedent | **Erik**; the "pure function" rule is not the spec's rule — `obstacles` is a digested stamp output (critique 3 §6b) |
| 29 | "Ordinates may run sequentially or concurrently"; scratch `(N, h, w)`; launches uncounted | **Ordinates run concurrently**, scratch `(N, 16, h, w)`, one launch per wavefront *index* covering all 16 ordinates; per-cell sums by the tree's int64 atomic idiom; **launch counts in §10** (256 shear / 383 step per tick at 128×256) | critique 3 §2b–2d |
| 30 | `E°` bake "verbatim into a header"; `s_m` "quantized at load"; `L°` from the blackbody ramp | The bake body lives in `emissive_table.cpp` on the `/fp:strict` list (a header-inline bake would compile under `bindings.cpp`'s `/fp:fast` and could FMA-contract `k4·scale + 0.5`); `s_m` are **checked-in integer literals** with a recompute test, never `std::cos` at load; `L°` at P6 by the `E°` algebraic pattern or checked-in constants, never `np.power`/`np.log` | critique 3 §5a–5c |
| 31 | Gate 2 "with `f < 1` forced on"; gate 4 one scenario; gate 5 "compiled out"; `E°⁻¹` undefined below `E°[0]`, saturation "owned by the rail" | Gate 2 = (a) uniform ambient by construction + (b) the **enclosed isothermal box**; gate 4 **per counter**; gate 5 via a `clamp_enabled` binding keyword; `E°⁻¹(Φ < E°[0]) = 0`; the table top is **15 996**, below `T_MAX_PHYS`, so the clamp binds first on the radiative sub-step and the rail is reachable only through the `heat` branch | critique 3 §4b, §4c, §4e |
| 32 | `f_q` in Q16 | **`f` in Q24**: `f_q24 = floordiv_q(T_abs_q << 24, D)`, `src = amb_m + mul128_shr(ex_m, f_q24, 24)`. In Q16 the damped source `f·(E°[T] − E°[0])` is **not monotone in T** above the fire range — `f_q` has only 38 counts at the table top for wood, so one count is 2.6 % and the emission wobbles backwards by up to 2.47 % (wood), 19.9 % at `thermal_mass = 1`; in Q24 **every shipped row is exactly monotone** (0 backward steps; the pathological `thermal_mass = 1` row 0.047 %; gate G12, which finds 2033 backward steps under Q16) and no extra ingress rule is needed | P0 §0.4 (orchestrator's call: arithmetic detail, no physics change); **P0b measured it** |
| 33 | §2.8's "Fleck alone" column 845 / 18 982 / 1 727 794; "+2.2 % above the analytic cooling curve"; the isotropy table quoted without its configuration | The Fleck-alone column at v3's **own** α is **844 / 38 310 / 3 454 227** (the old numbers were `stability_study.py`'s fixed α = ½; the clamp buys ×300, not ×150); the Fleck cooling error is **+0.23 %** (the 2.2 % existed only in the study's docstring; its own §5 prints +0.2 %); the isotropy table was the float study's no-inflow configuration — with the ambient ring on, P0 measures shear 1.46 / 1.28 / 1.12 vs step 1.63 / 1.26 / 1.25, so the 3×3 ordering flips and the 1-tile and 5×5 cases decide. **Shear stays** | P0 §0.1–0.3 |
| 35 | §3: "all of the 'never' rows are one P1 commit" — the live planes widened at P1 | **The live planes `rad_net`/`rad_amb`/`rad_flux` stay int32 through P2; the fold's parameter, the two surviving bindings and the CUDA temperature twin widen at P3, with the flip.** At P1 the old cast still writes the live planes every tick through by-value `py::array_t<int32_t>` bindings, so widening them under it would hand the old cast a discarded forcecast temporary — fire heat silently vanishing, goldens moving — the very trap critique 3 §6a describes. The sweep's outputs at P1 are the four NEW int64 planes; the kit's int64 twins land at P1 unused; everything that changes a live type changes at the flip | orchestrator, 2026-09-15 night, at the P0b→P1 boundary (an execution-sequencing call; the inventory is unchanged) |
| 34 | The 0-D equilibrium table read as a sweep prediction | In the 2-D sweep a cold crate one tile from a held 16 000-game source settles at **1108** clamped vs **1220** unclamped under floor ½ (Q24; P0's Q16 pair was 1104 / 1212) — and at floor 0 (P2a) **1108.0 vs 1113.6**, 0.5 %: the clamp still fires (14 hits in 24 ticks) but is no longer what holds the fire-range answer down, the 2-D confirmation of P0b §0.8c: the geometric factor at one tile is 0.14 (shear) / 0.25 (step), and the Fleck factor damps a table-top source **×851**. The 0-D table is the *scheme's bias*, correct as such. Two consequences: **P2 calibrates against the damped emission**, and the damping's effect on a *driven* source (a burning tile held at 1263 game emits 73 % of black body under α = ½) is an **open question for Erik** (§12 item 4) | P0 §0.6; §9; §12 |
| 36 | Critique 3 §6a: make the surviving bindings "int64 without forcecast so a stale caller fails loudly" | **`.noconvert()` is what makes it loud.** Dropping `forcecast` alone is not: pybind11's second overload pass still performs the *safe* int32→int64 cast into a discarded temporary. P1 marks every plane argument of the sweep and the step-tail bindings `.noconvert()`, and `tests/test_radiation_sweep_shadow_wiring.py` proves both an omitted plane and an int32 plane raise `TypeError`. The P3 widening uses the same word | P1 finding 1 |
| 37 | §2.4's isotropy table (Fleck off, the reference's configuration) | **The damped ordering is floor-dependent, not a fact about the transports.** Under floor ½ (P1) the 1-tile ordering flipped (shear 1.716 vs step 1.688); under floor 0 (P2a) it does not (1.520 vs 1.762) and the 5×5 case flips by ≈1 % instead (1.261 vs 1.249), 3×3 favouring shear both times. The ruling stands on the undamped configuration, which the engine gate asserts and which reproduces P0's table to three decimals; the damped numbers are reported, never asserted | P1 finding 4; P2a finding 2 |
| 38 | §3: the four per-tick planes "wiped beside the other three" at the end of the tick | **The wipe blinds the readout**: `rad_fluence` is zero by the time the renderer's tile inspector reads it. **P2 moves the zeroing into the sweep's own start** — it accumulates over ordinates, so it must clear its four outputs before the first one anyway — and deletes the four conductor lines; the planes then hold the last tick's values at snapshot time, which changes nothing digested (they are in neither `DIGEST_FIELDS` nor `SIM_FIELDS`). Also from P1: the A/B default scenario is **radiatively inert** (its seeded fire is a decorative one, CLAUDE.md "Starting a fire"), so the `max|rad_net| < 2³¹` check lives on the playground level with a 1263-game wood tile (old cast 150 148 108, 7 % of 2³¹); the un-clamped 2-D crate **settles** at 1219.8 game at P1 because the int32 live plane cannot carry the 0-D runaway, so the rail is proven separately with an `INT32_MAX` deposit; the dynamic extinction plane's resting state is the static plane, so `a ≤ d` holds for direct runner callers that skip the stamp; the q16 `shr_round0` wraps at `INT32_MIN` under MSVC (pinned; the P3 widening removes the edge); **measured cost 5.3 ms per tick** shear at 128×256 (§10) | P1 findings 2, 3, 5, 6, 7, 9 |
| 39 | `α = max(½, 1 − 1/g)`, Fleck's IMC bound as the floor | **Floor 0 — `α = max(0, 1 − 1/g)`, `D = max(T_abs_q, 4·L_q)`.** No damping wherever the explicit material update is provably monotone (`g ≤ 1`: 1424 game for the `a = 0.5` rows, 1068 for wood), the same `T_abs/4`-per-tick cap above it. A burning tile held at its plateau by combustion radiates at full black body where the tick allows; free cooling becomes forward Euler (−5.7 % at 0.5 s from 1263). Implemented at P2a, **reference first**, gate 0 re-run | **Erik, 2026-09-16: "Let's go floor 0 then."** P0b measured both floors (`report_p0.md` §0.8); §12 item 4 carries the case as it was put |
| 40 | §11.5: Fable for P3, P5, P6 | **Every remaining patch is split into small Opus steps** (§11.5): P2 → P2a/P2b, P3 → P3a/P3b/P3c, P5 → P5a/P5b, P6 → P6a/P6b/P6c. No Fable implementer this week, and every implementer writes its report file **incrementally** so a kill costs no resume | Erik, 2026-09-16: cutting the patches is where the leverage is; the Fable weekly limit stood at 86 % after one day (issue #66) |
| 41 | §9: "The reach curve then falls out" of the derived `rad_scale` — the calibration is what fixes the reach | **`rad_scale` cancels out of the reach.** The irradiance a receiver sees is `q = Φ·J_per_count/(A_rad·Δt) = σ·Φ/rad_scale`, and `E°⁻¹` inverts the same table `Φ` was written in, so the calibration divides out of both — *except* through the Fleck factor, the one scale-dependent term in the scheme. What the calibration sets is the **rate**: degrees per tick for a given heat capacity. Reach falls out of the **geometry** (the sweep's `G(d)`) and `k_leak`, and was already correct before P2b. **This reframes what P3's flip changes: not how far fire reaches, but how fast things heat** | P2b §5.1, §14.1 — **measured**, and visible as two indistinguishable curves in the bench figure's panel (a). Verified independently by the orchestrator |
| 42 | Row 34: "P2 calibrates against the damped emission"; §12 item 4: the driven-source damping question | **At the derived scale the Fleck damping never engages** — `f == 2²⁴` on all 4000 buckets for all eight shipped absorbing rows, worst case `g = 0.70` (wood at the table top, against `g = 1696` at the fitted scale). So **row 34 is vacuous here** and **§12 item 4 is moot**: both were written against a 2419×-oversized emission. Read backwards, the §2.8 stability apparatus existed to tame an over-hot dial; it stays (it is correct, cheap, and guards future rows and the `a = 1` materials at the fitted scale) but it is dormant at physical calibration. Row 34 also never bit for the row it was written for: floor 0 leaves `furniture` undamped through 1424 game, so the 1263-game plateau has `f = 1` at *either* scale; only the `a = 1` rows (undamped through 1068) ever saw it | P2b §4, §14.2–3. The orchestrator re-derived `g` independently (0.7009 for wood at the table top) and recovered P0b's fitted-scale thresholds from the same formula |
| 43 | §9: what is tuned is `thermal_mass` and `cool_shift`, per material row | **The reach curve is an upper bound that `cool_shift` takes back.** Solving the row's own ambient decay against the radiative gain at the derived scale, a furniture tile one tile from a plateau fire settles at **67 game** (360 K) and radiative ignition happens **at no distance at all**; it would need `cool_shift ≥ 16` (8× today's e-fold) to reach 280. And the model is **lumped** — one temperature per tile, no surface/bulk gradient — so the pinned 139 kg crate heats 20–45× slower than real piloted ignition, where only a ~2.6 mm skin must reach `ignition_temp`. The two readings of `thermal_mass` (bulk inertia vs effective surface-layer capacity) differ by **~32×**, want opposite values, and are the same key today; the surface reading needs `thermal_mass = 0.25`, which the power-of-two table cannot express. **The honest long-run answer is a two-node surface/bulk thermal solid — a design addition, not a dial.** This is the first thing P3's HUMAN-TEST will run into | P2b §9, §14.5. Findings, not changes (P2b applied none); §12 items 6–7 carry the rulings |

**Not changed, on purpose:** the leak channel stays designed, dormant (`k = 0`)
and unruled — Erik set the reach question aside (handoff §0). Everything in the
handoff's §2 "settled" list stands.

**Two critique findings are wrong and are not applied** (Appendix B): the
`config.toml` furniture comment *is* present (`678d9b6`), and the "1 failed / 351
passed" figure was a truncated `-x` run — the branch is green at `82b11a5`.

---

## 1. What we are building, in one paragraph

Radiation stops being a per-emitter ray cast and becomes a **field solve**, like
pressure and conduction already are. **One** directional sweep visits every cell
once per ordinate and carries several integer payloads: an exactly conservative
heat channel inside the sim, a light channel for the rules and for RL, and the
same light channel dequantized for the eye. Heat and light differ in their
transport step and their per-cell coefficients — not in their number type and
not in their machinery. Cost is set by the grid, so reach is free and the number
of hot or bright things does not matter, which is the entire point. The old
ray engine is retired, not kept beside it.

---

## 2. The sweep

### 2.1 The law

The radiative transfer equation, non-scattering (R-L: clear air is transparent;
smoke absorbs but does not scatter in the heat channel):

```
    ŝ · ∇I(x, ŝ)  =  −κ(x) I(x, ŝ)  +  κ(x) B(T(x))
```

Discretise `ŝ` into `N` ordinates and the grid into cells; each ordinate becomes
an independent transport problem.

### 2.2 Why ONE sweep is exact — and exactly what would break it

In a non-scattering medium the ordinates do not couple, so one upwind sweep per
ordinate is the exact discrete solution of the frozen-source problem — and the
source *is* frozen: temperature is fixed for the whole cast (the fold consumes
`rad_net` afterwards), and light crosses a 40 m compartment in 0.13 µs against a
41.67 ms tick.

Two things would break it, both excluded from P1: **diffuse reflection** (a
scattering term; measured, one extra full solve per bounce,
`reflections_study.py`) and **angular diffusion** (Camminady's "solve the
modified equation directly", same cost). Neither is needed for heat — R-K gives
walls a re-emission channel through their own temperature. For light, bounce is
available later and can be amortised across ticks (§7.4).

**Upwind order** is a topological order of the per-ordinate dependency DAG. For
shear the dependency is a **whole column** (both outputs of a cell land in the
next column); for step it is an **anti-diagonal** (the KBA skew). §8.2 says what
that means for the CUDA twin.

### 2.3 The integer scheme, in gather form, and why it conserves exactly

**Inputs per cell `i`** (all Q16.16 integers, `ONE = 65536`):

| symbol | plane | meaning |
|---|---|---|
| `a_i` | `heat_atten_q` (static) or the smoke term (§6.3) | the **material** extinction: absorbs *and emits* (Kirchhoff) |
| `d_i` | `dyn_heat_atten_q` | the **stamped** extinction, `d_i ≥ a_i`: the material plus any body standing on the cell |
| `b_i = d_i − a_i` | derived | the **body** share: absorbs the stream and **re-emits the ambient level** (a grey body at room temperature; Erik's ruling, row 25); its net, the excess above ambient, lands in `rad_flux` |
| `k_i` | `[physics.radiation] k_leak`, uniform today | the out-of-plane leak coefficient, `0` until ruled |
| `T_i` | `temperature` | the cell's temperature, both media |
| `f_i` | computed pre-sweep, §2.8 | the Fleck factor |

**Per-ordinate constants** (quantized ONCE at load, door 2):

| symbol | value | note |
|---|---|---|
| `w_m` | `ONE / 16 = 4096` | the ordinate weight; `Σ w_m = ONE` so the baked total-power `E°` drops in unchanged |
| `s_m` | shear: `ONE − f_m`, `f_m = abs(minor/major)`; step: `abs(μ) / (abs(μ) + abs(η))` | the **major share** of the downwind split, a Q16 constant per ordinate |
| `amb_m` | `(E°[0] · w_m) >> 16` | the ambient stream per ordinate — the *same integer* seeds the sky and every cell's ambient emission |

**The step**, per ordinate `m`, in wavefront order, for cell `i` (int64 throughout):

```
    i_in      =  gather of the upwind outflow(s):  fa from the straight upwind cell,
                 fb from the diagonal upwind cell, recomputed from THEIR stored i_out:
                     fa = (i_out_up * s_m) >> 16 ;  fb = i_out_up − fa
                 an out-of-grid upwind cell reads as i_out = amb_m   (the virtual ambient ring, §2.7)
    leaked    =  (i_in  * k_i) >> 16                 // out-of-plane, dormant at k = 0
    ret       =  (amb_m * k_i) >> 16                 // the ceiling radiates back at ambient
    stream    =  i_in − leaked + ret
    abs_mat   =  (stream * a_i) >> 16                // the material's share
    abs_body  =  (stream * b_i) >> 16                // the body's share
    ex_m      =  ((E°[T_i] − E°[0]) * w_m) >> 16     // the EXCESS emission over ambient, ≥ 0
    src       =  amb_m + mul128_shr(ex_m, f_i, 24)   // Fleck damps the excess only (row 22); f_i is Q24 (row 32)
    emitted   =  (src * a_i) >> 16                   // the SAME arithmetic as abs_mat
    emit_body =  (amb_m * b_i) >> 16                 // the body re-emits AMBIENT (row 25) — no excess, no Fleck
    i_out     =  stream − abs_mat − abs_body + emitted + emit_body
    rad_net[i]     +=  abs_mat − emitted             // the material ledger, signed
    rad_flux[i]    +=  abs_body − emit_body          // the body sensor: NET above ambient, signed
    rad_amb[i]     +=  leaked − ret                  // the ambient ledger's ceiling part
    rad_fluence[i] +=  stream                        // Φ, for the clamp
    store i_out for the next wavefront's gather
```

A boundary cell additionally books, into its own `rad_amb[i]`, `+` every share
of its `i_out` that leaves the grid and `−` every `fa`/`fb` it gathered from a
virtual cell. So `rad_amb` is **net ambient exchange per cell**: sky out minus
sky in at boundary cells, ceiling out minus ceiling return everywhere.

**Why the body emits ambient** (row 25). At exact ambient `abs_body ==
emit_body` by the same integers, so a body disturbs nothing in an ambient room
and the per-cell fixed point (gate 2) holds with bodies present; in a hot stream
the body passes `amb_m` on, so it shadows the fire and not the sky. `rad_flux`
is therefore **signed** — negative only in another body's shadow — and its one
consumer, `exchange.py:299-340`, takes `max(heat, rad_flux)` with `heat ≥ 0`,
which floors it at zero. Critique 3's probe (§4f there) measured the pure-sink
alternative: a marine beside an ambient wall drained the wall by 775 counts per
tick and tripped `t_low_rail_hits` every tick.

**Conservation is structural.** For every cell, `i_in − i_out = (leaked − ret) +
abs_mat + abs_body − emitted − emit_body`, i.e. the sum of what that cell booked. Summing
over the grid, every interior gather is some other cell's stored outflow, so the
interior telescopes and only the virtual-ring terms remain — which the boundary
cells booked. Hence

```
    Σ_cells rad_net  +  Σ_cells rad_flux  +  Σ_cells rad_amb  ≡  0        exactly, in int64
```

by the `eos_solver.cpp::face_flux` idiom: one integer, applied twice with
opposite signs. There is no `sky_in` scalar, no `ceiling` scalar and no emitter
attribution; those were v1/v2 artefacts of the push form and of the old
per-emitter engine.

**Why the gather form** (row 23). In the push form a cell writes two downwind
faces and two upwind cells write into one downwind cell, which on a GPU is an
atomic or an ordering. In the gather form a cell reads the two stored outflows
and recomputes `fa`/`fb` from them with the same shift, so it produces the same
integers the push would have, with **no atomics inside an ordinate**. The float
reference (`sweep_ref.py`) is written in push form; P0's integer reference is
written in gather form and cross-checked against it (§11).

**Invariants, each an ingress check with a test:**

- `0 ≤ a_i ≤ d_i ≤ ONE`. Positivity depends on it: `abs_mat + abs_body ≤ stream`
  because truncation only decreases, and `(stream·a)>>16 + (stream·b)>>16 ≤
  (stream·(a+b))>>16 ≤ stream` when `a + b ≤ ONE`. Every dynamic stamp is a
  **MAX**, never a sum. The site is `src/simulation/materials.py` (inside the
  ingress lint's scope, with the `ingress-exempt:` precedent at `:640`, `:659`)
  and the validated-range precedent is the filter table's `[0,1]` rule.
- `0 ≤ k_i ≤ ONE`, same site.
- `heat_atten > 0 ⇒ thermal_mass > 0` at table build. Today every absorbing
  material is a thermal solid (`temperature_solver.cpp:243-246` says so, belt and
  braces); the check makes it loud, because an absorbing cell the fold ignores
  is an uncounted sink.
- `E°[T] ≥ E°[0]` for every bucket (true by construction of the bake; asserted).

**Four corrections inherited from v2**, each of which leaks or overflows if
omitted (critique 1, changes 3–5): the remainder split `fb = i_out − fa`
(measured, two shifts leak −5 626 counts per tick on a 24×24 grid); one Q16
factor per shift on `emitted` and on `ret`; `a ≤ ONE` as an invariant; **int64
for the stream and the four planes** — `E°` at `T_MAX_PHYS` is ≈ 3.6·10¹² at the
shipped `rad_scale`, and v2 measured 1.9·10⁷ on a small fire scene. `E°` was
already int64 (`cpp/src/raycaster.cpp:62-97`); what widens is `rad_net`,
`rad_amb`, `rad_flux` and the new `rad_fluence` (§3).

> **2026-09-25, #78 (Erik's request of 2026-09-24):** every integer in this section is in the heat channel's FINE currency, `2^k` per heat count (`E_FINE_BITS = 11`, `emissive_table.h`): the `E°` table, and with it `amb_m`, `ex_m`, Φ and the four planes, is baked at `rad_scale_derived · 2^k`, and every reader converts once through `fixedpoint::fine_heat_shr` (the gas chain at `deposit_dT_wide_i64`'s final narrow). The per-ordinate floors lose at most `2^−11` of a heat count; the identity is unchanged — `sweep_fine_heat_currency_brief_78_2026-09-25.md`.

### 2.4 The transport step: one parameter, two settings

```
    step  (θ = 0):  push to the side neighbour and the down neighbour, split by
                    the direction cosines.  Rebuilt from two moves at every cell,
                    so the beam smears like √distance.  Axis-aligned ordinates
                    become pencils that never spread: the four-point cross.
    shear (θ = 1):  advance a FULL cell along the dominant axis, split only
                    transversally by f = |minor/major|.  The beam keeps its
                    heading, so it barely smears.  With 16 ordinates you see 16
                    of them: the star.
```

In the gather form the two differ only in *which* two upwind cells are read and
in the constant `s_m`: shear reads `(y, x−sx)` and `(y−sy, x−sx)` for an x-major
ordinate; step reads `(y, x−sx)` and `(y−sy, x)`. One code path, one table.

Davis et al. 2012 names this trade as fundamental: a sharper step buys shadow
fidelity and pays in *"fan-shaped spokes"* at modest angular resolution, and
*"a greater degree of diffusion in the intensity can mitigate unphysical
effects"*.

**Heat takes shear. Light takes step.** Heat is judged by the ignition
footprint: measured spread (max ÷ min radius over 64 directions, leak on) shear
1.37 / 1.22 / 1.17 for a 1-tile, 3×3 and 5×5 fire, against step's 1.64 / 1.32 /
1.25 — **in the float study's configuration, no ambient inflow** (row 33). With
v3's own ambient ring on, P0's integer reference measures shear **1.46 / 1.28 /
1.12** against step **1.63 / 1.26 / 1.25**: the 3×3 ordering flips (the ambient
floor moves the ignition crossing outward into a ring where shear's 16 lobes are
stronger), the 1-tile and 5×5 cases decide, and the picture (`p0_isotropy.png`)
shows step's single-tile spikes are *longer* on the same waist. And with the
engine's **own Fleck damping on** (P1, row 37) the 1-tile ordering flips —
shear 1.716 against step 1.688 — while 3×3 (1.236 vs 1.298) and 5×5 (1.165 vs
1.267) still favour shear; a single damped tile is a weaker, rounder-by-step
source, and a fire is never a single tile for long. Light is judged by eye,
and Erik chose step on the real-scene renders.

Both are exactly conservative in the paired form, both hold the uniform-ambient
fixed point, both are positive. The scheme study's 26 % shear leak was two
absorption laws in one step (`κ·I·w` charged, `exp(−κ·path)` advanced); in the
paired form the residual is zero. **Conservation does not discriminate.**

The shear boundary seeding that first read as a scheme defect — a shear step
gathers from **two** upwind cells, so the transverse inlet edge needs seeding as
well as the major-axis edge — is handled by the virtual ambient ring (§2.7) with
no special case: the diagonal upwind read simply falls outside the grid and
returns `amb_m`.

### 2.5 Angular: S16, half-offset, no rotation

- **N = 16**, evenly spaced, **half-offset** so no ordinate lies on an axis or a
  45°. `w_m = 1/16`, `Σ w_m = 1`, so the baked `E°[T]` — already the cell's
  *total* per-tick emissive power — drops in unchanged.
- **No per-tick rotation.** rSN carries and interpolates an angular flux; we
  carry none, and Camminady's §7 says it *"is not directly compatible with
  sweeping"*. Measured, our jitter made ring ripple worse at every radius. Dropping
  it also removes a `tick mod N` dependence from a synced law.
- **S16 is a sweet spot.** More ordinates make shear worse and do nothing for
  step's near field.
- `w_m = 4096` is dyadic. Gate 2 (§2.9) therefore also runs at a non-dyadic
  ordinate count (S12), because every configuration that caught the v1
  emission-form defect used a non-power-of-two `w`.

### 2.6 Emission: the temperature map IS the emission map (R-J)

No emitter list, no `T_emit_gate`. Every cell contributes `a_i · [E°[0] + f_i ·
(E°[T_i] − E°[0])]`, per ordinate `w_m` of it.

**The `E°` table moves to `cpp/src/emissive_table.{h,cpp}`** — the bake from
`raycaster.cpp:62-97` (int64, 4000 buckets of 4 game units, `K⁴` by repeated
integer multiplication, the integer-valued Kelvin-map precondition), the lookup
`e_bucket_of` from `raycaster.h:203-213`, and the new inverse. **The bake's body
is out of line in `emissive_table.cpp`, which joins the `/fp:strict` list**
(row 30): the chain `(double)k4 · scale + 0.5` is a multiply feeding an add
that MSVC may contract into a fused multiply-add under `/fp:fast`, and a
header-inline bake would be compiled by `bindings.cpp`, which is deliberately
`/fp:fast` (`CMakeLists.txt:23-25`). The header carries only the integer
lookups (`e_bucket_of`, `E°⁻¹`, both `FP_HD`, the inverse a **fixed 12-trip**
loop in the `sqrt_q16_dev` idiom so host and device are one function) and the
table pointer. One instance, owned by `PhysicsEngine`, passed to the sweep *and*
to the temperature solver's Pass 1 (the clamp needs it) and its CUDA twin
(uploaded as `cuda_raycaster.cu` uploads it today). `rad_scale` stays the
bake's input through P1–P2 and becomes a derived constant at P2 (§9).

**`E°⁻¹` and saturation** (critique item 28). The table is strictly increasing
in `T` (K⁴ is), so `E°⁻¹(Φ)` is a binary search: the largest bucket `b` with
`E°[b] ≤ Φ`, mapped to the bucket's **low edge**, `T = 4b` game units, so that
re-applying the clamp is idempotent (`E°[E°⁻¹(Φ)] ≤ Φ`; critique 3 probe [7]).
Twelve compares, pure integer, identical on every backend. Two edge cases,
both defined (row 31):

- **`Φ < E°[0]`** has no bucket. It is reachable (a cell in another body's
  shadow, or any cell while the leak is on and `ret` is truncated). `E°⁻¹` returns
  `T_cap = 0`: no radiative warming above the ambient floor, which is consistent
  because such a cell's `rad_net ≤ 0` anyway.
- **Saturation is at 15 996, not 16 000.** The last bucket's low edge is
  `4 · 3999`, so for `Φ ≥ E°[3999]` the clamp returns 15 996 game and binds
  *below* `T_MAX_PHYS = 16000` (`config.toml:186`, `temperature_solver.h:394`).
  On the radiative sub-step the clamp therefore always engages first and the
  `T_MAX_PHYS` rail never does; the rail stays reachable through the `heat`
  deposit branch only, which is why gate 4 is stated per counter (§2.9). The
  60 000-game rows in v2's §2.8 table describe states the engine clamps away
  and are struck.

> **2026-09-24, P5d (Erik's ruling):** the clamp no longer reads `E°⁻¹`; its ceiling is `e_ceiling_q`, the top of the first bucket out-emitting Φ (saturating one LSB below 16 000) — `ray_engine_v2_p5d_clamp_headroom_brief_2026-09-24.md`. `E°⁻¹` itself is unchanged.

> **2026-09-25, #78:** in the fine currency the ambient field's Φ = `n·(E°[0]·w_m >> 16)` sits up to 15 fine counts below `E°[0]`, so near-ambient cells still take the `Φ < E°[0]` branch (about 2.4 M cell-ticks on the P5d bench's 180 s playground), but what they book there is at most 0.0063 heat counts a tick and none of it converts to a temperature LSB: the clamp never binds on that branch (0 hits, 0 W, against P5d's 152 988 landing cell-ticks and ~1 kW). The branch stays as ruled — `sweep_fine_heat_currency_brief_78_2026-09-25.md`.

**The sub-ambient floor** is inherited, not new: `e_bucket_of(T ≤ 0) = 0`, so a
cell below ambient emits at the ambient level. Its excess is zero, so it neither
warms nor cools radiatively in an ambient bath. A modelling boundary, recorded.

### 2.7 Boundaries

**The virtual ambient ring.** Every upwind read that falls outside the grid
returns `i_out = amb_m`. That single rule gives the ambient blackbody inflow at
every boundary face for *both* transport steps, seeds the shear transverse edge
correctly (the diagonal read falls outside too), and needs no per-edge code.
Boundary cells book what they take from the ring and what they push into it,
signed, into their own `rad_amb[i]`.

v1 had no inflow — a 0 K sky, measured at **−4.96 game/s** on the worst exposed
cell with a permanently-firing low-rail counter, which the temperature solver
documents as a RED (`temperature_solver.cpp:283-290`).

**`rad_amb` is the ambient ledger, a plane, not a scalar** (critique items 19,
6d). `gamemap.py:487-494` records why a plane: a global counter is a contended,
order-dependent-if-saturated atomic; a per-tile plane with plain adds is
order-free and the host reduces it. That argument survives the change of meaning
— what changes is the key (exit cell, not emitter) and the sign (net, not
non-negative), and int64 with a headroom argument replaces "wrapping" (§8.2). The
two attribution-dependent assertions
(`tests/test_pr1_fire_plane_cast.py:255-263` "sky booked to exactly one tile",
`tests/test_pf1a_radiation_books.py:756` `rad_net[15,15] == -rad_amb.sum()`) die
with the old law; the full disposition is §11.

**The sky beyond the grid is ambient, even in space.** This matches the
thermostat ruling (P-G5: the two-way ambient boundary is a modelling choice) and
the EOS. A space-cold sky is a per-boundary-face value on the same machinery,
later, if wanted.

**The leak channel** is dormant (`k = 0`), derived (`k ≈ 0.10` on a 2.5 m deck,
`reach_and_papers` §2) and unruled. It is booked into `rad_amb` because
physically it *is* an ambient exchange (the deck's floor and ceiling at
ambient). Its fixed point is exact: at `i_in = amb_m`, `leaked == ret` as the
same integer.

**Inflow, light** — a directional sky BC replacing today's flat shader constant
`u_ambient = (0.18, 0.18, 0.22)` (`renderer/lighting.py:105`, `shaders/lighting.fs:133`):
per-ordinate values on the virtual ring, so overcast is uniform, a sun is a few
ordinates, and a sealed compartment goes properly black. A small flat floor stays
a separate dial. §7.

### 2.8 Stability: the Fleck factor in exact integers, and the clamp in its correct form and home

**The problem.** The material update is explicit and the loss goes as T⁴, so it
is monotone only while `4 × (per-tick radiative loss) < T_absolute`. At the
shipped scale that holds to ≈1800 game and fails hard above: at `T_MAX_PHYS` the
cell overshoots by ×864 in one tick. No calibration constant fixes it.

**The remedy, from Fleck & Cummings 1971** (Wollaber's review, eqs. 17–21): scale
the emission by

```
    f = 1 / (1 + α · g),      g = 4 · loss_per_tick / T_absolute,      α = max(½, 1 − 1/g)
```

α follows Fleck's own condition: `α = ½` is the unconditional-stability bound and
the coefficient `[1 − (1−α)g]/[1 + αg]` turns negative above `g = 1/(1−α)`, so
`α = ½` admits an oscillatory mode above `g = 2` — which is 1800 game, exactly the
plasma range this scheme exists to survive. `α = max(½, 1 − 1/g)` is identically ½
through the fire range (measured unchanged to five figures) and rises toward 1
where Fleck says it must; cost +0.2 % against +6.0 % for a fixed α = 1.

**RULED 2026-09-16 (row 39): the floor is 0, not ½ — `α = max(0, 1 − 1/g)`.**
Fleck's `α ≥ ½` is the bound for IMC's own linearised equations with effective
scattering; for *our* material update the monotone-stability condition is that
the fixed-point multiplier `x = g·(1 + αg/4)/(1 + αg)²` stays in `(0, 1]`, and
`α = max(0, 1 − 1/g)` satisfies it for every `g` (`x = g` below `g = 1`,
`x = (g + 3)/4g ≤ 1` above; P0b measured monotone, positive and finite from
every start to the table top). The reason it matters: the Fleck factor assumes
the cell will cool during the tick, which is wrong for a *driven* source — a
burning tile that combustion re-heats every tick. Under floor ½ such a tile at
the 1263-game plateau radiated 73 % of black body (furniture) or 58 % (wood);
under floor 0 it radiates at full black body wherever `g ≤ 1` (1424 game for
the `a = 0.5` rows, 1068 for wood — a wood crate at 1263 still radiates 68 %,
because with full emissivity and a small heat capacity it genuinely would shed
more than a quarter of its absolute temperature in one tick, which no explicit
scheme at 24 Hz can do stably; P2's heat-capacity tuning moves that cap). The
cost is on *free* cooling — plain forward Euler, −5.7 % at 0.5 s from 1263
against floor ½'s +0.23 % — which is the cool-down after a fire has died. One
result favoured floor 0 outright: at a fire-range source the un-clamped
equilibrium bias is 0.15 % against +4.6 % under ½. Above ≈1800 game the two
floors are bit-identical.

**The integer form** (critique items 9, 5a). Substituting α:

```
    1 + α·g  =  max(1,  g)              ⇒        f  =  T_abs / max(T_abs,  4L)        // floor 0 (row 39)
    (the ruled-½ form, built at P1 and replaced at P2a, was  max(1 + g/2, g)  ⇒  T_abs / max(T_abs + 2L, 4L))
```

so α is never computed and there is **one exact kit floor-division per cell per
tick**, door 1:

```
    L_q     =  the cell's free excess-emission loss this tick, in Q16.16 temperature:
               solid:  shr_round0( (a_i · (E°[T_i] − E°[0])) >> 16,  heat_inv_shift[i] )
               gas:    the STAGED wide chain on the same excess (row 30, critique 3 §1c):
                       mul128_shr( mul128_shr(e64, recip_N, FP_SHIFT), recip_cv, RECIP_SHIFT )
                       — two narrows, because the existing deposit_dT_wide_q16 forms its first
                       product as a plain int64 (sound for an int32 deposit, 2^64–2^72 for E° at
                       the table top); ≤ 1 LSB from the heat deposit's one-narrow chain, a
                       DIFFERENT rounding, declared; FP_HD mul128_shr on both backends; the same
                       chain converts the P5 radiative gas deposit itself (magnitude, then sign)
    T_abs_q =  temperature[i] + t_amb_q            // 293 game units in Q16.16 (slope 1 since G12)
    D       =  max(T_abs_q,  4·L_q)                 // floor 0 (row 39); shifts, exact
    f_q24   =  floordiv_q((int64)T_abs_q << 24, D)  // fixed_point.h:562 — exact, identical on every backend; Q24 (row 32)
```

`T_abs_q > 0` always (`T_MIN = −292` game keeps `T_abs ≥ 1`), `D ≥ T_abs_q`, so
`0 < f_q24 ≤ 2²⁴`, and `f_q24 == 2²⁴` exactly when `4·L_q ≤ T_abs_q`, i.e.
throughout the explicitly-stable range `g ≤ 1` (under the old floor ½ it was
`2²⁴` only at `L_q == 0`). **Q24, not Q16**
(row 32): at the table top `f` is ≈ 1/851, which in Q16 is 38 counts for wood
(77 for furniture, 4 at `thermal_mass = 1`), so the damped source
`f·(E°[T] − E°[0])` steps backwards by up to 2.47 % as `T` rises — bounded, never
inverting the cooling map, but a real artefact P0 measured across every shipped
material row. Q24 divides it by 256. `T_abs_q << 24 < 2⁵⁴` and the product
`ex_m · f_q24 < 2⁶³` (P0b measured 2^62.13 at S12, 2^61.72 at S16); the engine uses the kit's
`mul128_shr(ex_m, f_q24, 24)` so no headroom argument is load-bearing, and the
reference asserts the bound anyway. The denominator is on
the **absolute** temperature because `β = 4aT³/c_v` is defined on Kelvin; a
delta-above-ambient denominator diverges at ambient. Applied **before the
sweep**, so conservation is untouched and the books need no counter — a smaller
source is not a leak (critique 2g, granted).

`reciprocal_q16` (`fixed_point.h:458`, Newton, truncated) is *not* used: it
returns a Q16 reciprocal of a Q16 operand and would need a second multiply and
a second truncation; `floordiv_q` is the kit's exact division and the value is
consumed once.

**Why the excess** (row 22). With `f` multiplying the whole `E°[T]`, `f = 0.995`
at ambient at the shipped scale, so an ambient cell in an ambient field gains
0.5 % of its exchange every tick and drifts ≈0.4 game units — sub-bucket, so the
clamp cannot see it, and gate 2's exact per-cell zero is unattainable. Damping
only `E°[T] − E°[0]` makes the ambient fixed point exact for any `a`, any `w` and
any `f`, and is the correct linearisation. **P0 measured it on this form** (`report_p0.md` §7): stable, monotone, positive
and finite from every start to the table top; a 1263-game cell cooling for
0.5 s lands **+0.23 %** above the analytic solution of `dT/dt = −c[(T+K)⁴ − K⁴]`
where explicit lands −5.70 % below and rails to zero from 2500 game up. (v2's
"+2.2 %" was never reproducible — it lived only in `stability_study.py`'s
docstring; that file's own §5 prints +0.2 %.)

**But emission-only damping is not enough.** Damping the loss but not the gain
moves equilibrium from `Φ = E°(T)` to `Φ = E°(0) + f·(E°(T) − E°(0))`, and since
`f` falls as `T⁻³` the equilibrium goes linearly in `Φ` instead of as its fourth
root: measured in the 0-D model under a held fluence `Φ = 0.25·E°[T_src]`, a
cell settles at **3.45 million** game units at v3's own α (1.73 million at a
fixed α = ½, the number v2 quoted) instead of 11 224 — the maximum-principle violation
Wollaber calls *"arguably the most serious deficiency of the IMC equations"*.
Real IMC scales absorption by `f` too and makes the remainder effective
scattering; we cannot (scattering couples the ordinates). Fleck's other route —
the local nonlinear material solve — *"does not conserve energy, and energy
checks in typical problems may run as high as 20 %"*. So:

**The maximum-principle clamp — its correct form** (row 21). A body cannot be
carried *by radiation* above the black body in equilibrium with the radiation it
absorbs. Radiation is not the only heater — combustion deposits directly into a
burning tile, and that tile is legitimately far hotter than its surroundings —
so the clamp bounds the radiative sub-step, not the temperature:

```
    T_after  =  T_before + ΔT_rad                       // the fold's radiative sub-step
    T_cap    =  E°⁻¹(Φ_i)                               // rad_fluence[i], §3
    T_new    =  min(T_after, max(T_before, T_cap))
```

> **2026-09-24, P5d (Erik's ruling):** `T_cap` is now `e_ceiling_q(Φ_i)`, the top of the first bucket out-emitting Φ_i (up to two buckets above `E°⁻¹(Φ_i)`) — `ray_engine_v2_p5d_clamp_headroom_brief_2026-09-24.md`.

It binds only when radiation would *raise* a cell above `T_cap`; a cell that was
already above it (a fire) keeps `T_before` as its ceiling and may only cool; a
cooling step is never clipped. On the study's scenario (radiation the only
heater, `T_before ≤ T_cap` always) this reduces to the measured form:

| source | true (`E°⁻¹`, bucket low edge) | true (continuous) | Fleck alone, v3's α | Fleck + clamp |
|---|---|---|---|---|
| 1263 game | 804 | 806.6 | 844 | **804** |
| 5000 game | 3448 | 3451.1 | 38 310 | **3448** |
| 16 000 game | 11 224 | 11 226.5 | 3 454 227 | **11 224** |

(P0 §7, 9600 ticks, `G = 0.25`. v2's 845 / 18 982 / 1 727 794 were at a fixed
α = ½; under the adaptive α the runaway is twice as far. The integer answer sits
one bucket below the continuous one by construction, because `E°⁻¹` returns the
bucket's low edge so that re-applying it is idempotent.)

**The 0-D table is the scheme's bias, not a sweep prediction** (row 34). In the
2-D sweep the same 16 000-game source one tile from a cold crate gives **1108**
game clamped against **1220** unclamped — 10 %, not ×300 — because the sweep's
geometric factor at one tile is 0.14 (shear) / 0.25 (step) and the source's own
emission is damped ×851 by its Fleck factor. The clamp still engages (17 hits in
24 ticks), so the gate is not vacuous. And the best single argument that the
scheme is physical: a cell fully ringed by 1263-game walls receives `G = 0.9987`
and `E°⁻¹(Φ) = 1256` game — a cavity reaches its wall temperature to within one
bucket.

**Its home** (critique BLOCKING 2): **inside the temperature solver's Pass-1
radiation fold**, `cpp/src/temperature_solver.cpp:247-299`, never in the sweep.

- **Solids.** Between the `sat_add_q16` and the two rails, with `t_before_rad`
  already in hand. The block books the *actual applied* ΔT × `cap_real_` into
  `e_solid_deposit_sum` (`:297-299`), so a clamp there is **booked automatically**
  and the P-G5 solid ledger keeps closing. Engagement is counted in the
  `t_max_phys_hits` / `t_low_rail_hits` idiom (`temperature_solver.h:395`,
  `:404`): a new `rad_clamp_hits`.
- **Gas** (from the patch that opens the mask, §6.3). Never `temperature[i] =` —
  the "Gas temperature is a mirror" rule records that failure as *silent*
  ("a bare `temperature[...] =` now moves no books at all … it goes VACUOUS").
  The radiative `ΔE` is computed, the clamp is applied to the **deposit**: the
  mirror is a floor-division of `E` by `N`, so the target is hit exactly by
  `ΔE_new = N · (T_target + t_amb_q) − E` with `T_target = min(T_after,
  max(T_before, T_cap))` — never a scaled `ΔE` (critique 3 §4d) — and the
  reduced `ΔE` goes through `gas_energy::deposit_railed` (`gas_energy.h:98`) with
  `e_gas_deposit_sum += nb·dT` as the existing group-1 branch does
  (`temperature_solver.cpp:373-388`). The reduction is a counted drop,
  `e_rad_clamp_drop_sum`, so the sweep→fold boundary (§8.4) stays accountable.

**Both must cover gas** (§6.3): a solid's heat capacity is its `thermal_mass`; a
gas cell's is its particle count, and the clamp is the protection that does not
care about heat capacity at all.

**The Fleck factor cannot live in Pass 1** — it is an input to the sweep, so it
is computed in a pre-pass inside `RadiationSweep` from `temperature`, the
extinction plane and the capacity planes the fold uses (`heat_inv_shift`; for gas
the `n_bulk`/`c_v` chain). It is never stored across ticks.

### 2.9 Gates the scheme must pass

Every gate names the property and the change that breaks it (CLAUDE.md
2026-09-08 rule); no gate may pass vacuously.

0. **The reference is the spec** (from P1 on): the engine reproduces
   `docs/ray_engine_v2_scheme_study_2026-09-13/sweep_ref_q.py` **bit for bit** —
   all four planes, the Fleck pre-pass and the fold's clamp — on randomised
   grids, both transports, leak on and off, bodies present, S16 and S12. Any
   change to the arithmetic is made in the reference first, its gates re-run,
   and the C++ written against it (P0's draft rule, Systems). This is the
   oracle that makes P1 an oracle-gated patch.
1. **Conservation**: `Σ rad_net + Σ rad_flux + Σ rad_amb ≡ 0`, exactly in int64,
   on randomised grids with randomised `a`, `d ≥ a`, `k`, `T`, every tick, both
   transport steps, leak on and off. **Non-vacuous**: the test seeds `d > a` on
   some cells (a body) so the `rad_flux` term is exercised at P1 (BLOCKING 6),
   and asserts each of the three sums is individually non-zero.
2. **Second law**, two halves (row 31). **(a)** A uniform field at ambient is a
   **per-cell exact** fixed point — every `rad_net[i] == 0`, `rad_flux[i] == 0`,
   `rad_amb[i] == 0` — with non-dyadic `a`, bodies present, the leak on, at S16
   *and* S12, over a transparent-interior sealed room and a mixed-material
   scene. This holds for any `f` **by construction** (the excess is zero at
   ambient), so forcing `f < 1` here exercises nothing and is not claimed.
   **(b)** The **enclosed isothermal box**: opaque walls two cells thick at
   `T0 > 0`, transparent interior; the inner wall layer is a per-cell exact
   zero **with `f < 1` forced on**, S16 and S12, because every gather it
   receives is another `T0` cell's `src` split and re-summed under the same
   shift. The outer layer legitimately cools to the sky. This is the successor
   of `test_pf1a_radiation_books.py:399-443`'s isothermal-lattice property;
   critique 3 probe [4] measured it at `T0 = 1263`, `f = 0.3`.
3. **Positivity**: no negative stream anywhere, with `d ≤ ONE` enforced; and the
   ingress checks reject `a > d`, `d > ONE`, `k > ONE`, and `heat_atten > 0` with
   `thermal_mass = 0`.
4. **Stability, per counter** (row 31): `t_low_rail_hits` and `rad_clamp_hits`
   are zero in every normal scenario — including a marine beside an ambient
   wall, the scene that fails under a pure-sink body — and `rad_clamp_hits` is
   **non-zero in a deliberately over-driven one** (a 16 000-game source one tile
   from a cold crate). `t_max_phys_hits` is not reachable through the radiative
   sub-step (the clamp saturates at 15 996 first, §2.6); it is asserted non-zero
   only in the `heat`-branch over-drive that already exists for it.
5. **Maximum principle** (BLOCKING 4). **Evaluation point:** immediately after
   the Pass-1 radiative sub-step, on that sub-step alone — `T_new ≤ max(T_before,
   E°⁻¹(Φ))` for every cell, asserted from a direct-binding call of
   `TemperatureSolver.step` (the `tests/test_temperature_convert.py:92-102`
   precedent) with `rad_net`/`rad_fluence` planes the test constructs. Measured at
   end of tick it would be falsely red (conduction, the thermostat and
   combustion's direct write all legitimately run after). **Non-vacuous:** the
   same test with `clamp_enabled=False`, a keyword on the `TemperatureSolver.step`
   binding in the `rad_net = py::none()` idiom (`bindings.cpp:2081`) — never a
   global static a test flips, which is hidden state that forks a digest —
   reproduces the unclamped runaway — in the engine's int32 Q16.16 field that
   is not 3.45 million game but `sat_add_q16`'s ceiling of 32 768 and then the
   `T_MAX_PHYS` rail at 16 000 with `t_max_phys_hits` firing every tick (P0
   §0.5): the clamp-off run **expects that counter**, it does not assert it
   zero; and a burning-crate scenario
   asserts the clamp does **not** bind on a cell whose `T_before > T_cap`.
6. **Isotropy**: ring MAX/MIN and the 64-direction ignition footprint, in P1.
7. **CPU↔GPU bit-identity**, tol 0 (P4), through `tests/cuda_harness.py`, on
   every plane **and every counter**. The check script calls the engine
   directly and reads the planes **before the conductor's wipe** — the
   `cuda_pr1_fire_plane_check.py` shape — because an A/B snapshot and the digest
   are both taken after `simulation.py:1603-1616` and see the four per-tick
   planes as zero. Gates 1–3 run against a direct call for the same reason.
8. **The A/B lockstep harness** (`tests/field_ab_harness.py`) at every wiring
   patch (P3, P6): never prove a refactor with whole-grid means. Stated
   honestly (critique 3 §7a): across a *law change* the harness proves
   run-to-run determinism of one build and **localises** the expected
   before/after difference to the fields the law touches; it is not a 0-ULP
   gate across P3. The 0-ULP gate is the twin's, gate 7.
9. **Goldens unmoved at P1** (asserted, since P1 is a shadow computation, §11)
   and **one deliberate re-baseline at P3** with written rationale.

---

## 3. Planes, number types, digest — the contract per plane

| plane | dtype | lifetime | allocation | `DIGEST_FIELDS` | `SIM_FIELDS` (A/B) | resident |
|---|---|---|---|---|---|---|
| `heat_atten_q` | int32 Q16 (h,w) | static; rebuilt in `_update_caches`, patched in `on_tile_changed` — the `heat_atten` seam (`gamemap.py:1356`, `:1578`) | quantized through **`src/simulation/optics_fixed.py`** (new; reuse-or-new: none of the seven `*_fixed.py` modules is about optics, and the light RGB planes join it at P6) | no — the static-projection precedent: `heat_inv_shift`, `face_shift`, `thermal_solid`, `cool_shift`, `heat_atten` are none of them digested | yes | static mask, one upload at `enable_residency` |
| `dyn_heat_atten_q` | int32 Q16 (h,w) | per tick, from `stamp_units` | fourth dynamic output of `PhysicsEngine::stamp_units` (`physics_engine.cpp:937`, header `:496`): copy of `heat_atten_q`, then per-unit **MAX** from a new per-row `heat_q` value; Python wrapper `gamemap.py:1749-1826` gains one row array, and **`_stamp_units_python` (`gamemap.py:1839`) grows identically** because `tests/test_stamp_units_cpp_ab.py` compares the two; unit attribute `heat_atten` (default `1.0`, opaque body), quantized by `optics_fixed.quantize_scalar` in `_stamp_units_cpp` (`:1785-1805`), the boundary | **yes, at P3** (Erik, row 28): `DIGEST_SPEC_VERSION` v6 in P3's own re-baseline commit, so a stamp desync is named one tick before it surfaces as `temperature`; the precedent is `obstacles`, a digested stamp output (`field_digest.py:75-78`) | yes, beside `dyn_light_atten` | always-upload set, `gamemap.py:137` |
| `rad_net` | **int64** (was int32) (h,w) | per tick, wiped at `simulation.py:1603` | same site | no (wiped at every snapshot, `field_digest.py:50-52`) | no | **joins `_RESIDENT_SYNCED` at P1** (int64 precedent `gas_energy`, `gamemap.py:130`). None of the three existing planes is resident today: the old cast fills them on the host mirror (`physics_runner.py:1207`) and `step_tail` reads the mirror (`:1356-1372`) — the design must not imply otherwise |
| `rad_amb` | **int64** (was int32) (h,w) | per tick, wiped at `:1610` | same | no | no | same, at P1 |
| `rad_flux` | **int64** (was int32) (h,w), **signed** (row 25) | per tick, wiped at `:1616` | same | no | no | same, at P1 |
| `rad_fluence` (Φ) | **int64**, new (h,w) | per tick; written by the sweep, read by Pass 1, wiped beside the other three | next to `rad_net` (`gamemap.py:479`); **allocated through the residency path from day one** (RL-batch habits §A rule 3: no mirror-only fields) | no — same argument as `rad_net` | no | same, at P1 |
| **P1 shadow planes** `rad_net_sweep`, `rad_flux_sweep`, `rad_amb_sweep` | int64 (h,w) | P1–P2 only: the shadow sweep's outputs while the old cast still fills the live three; wiped beside them (four `fill(0)` lines at P1, three fewer at P3 when the old writer dies and the sweep writes the live planes) | next to the live planes | no | no | resident, as the live ones |
| sweep scratch (stored outflow) | int64 **(N, 16, h, w)** — one plane per ordinate, ≈4 MB at 128×256 | inside one sweep call | owned by `RadiationSweep`; the ordinate axis exists because the ordinates run **concurrently** (§8.2, row 29) | — | — | device scratch |

**What the int64 widening actually touches — the full inventory** (critique 2
item 18 corrected the Recorder claim; critique 3 §6a found the list still
incomplete, row 26). Every site that types the three planes as 32-bit, from a
`grep` over `src tests tools renderer cpp/src`, and when it dies:

| site | dies at | action |
|---|---|---|
| `gamemap.py:479`, `:499`, `:514` allocations | — | widen at P1 |
| `temperature_solver.h:646` `const int32_t* rad_net` (the solver's own signature); `.cpp:248` the read, `:255` `shr_round0` | **never** | widen at P1; the kit gains an `FP_HD` int64 `shr_round0` (the operand is *signed*, so a plain `>>` is not the same function) — one function in `fixed_point.h`, never re-derived in the `.cu` |
| `physics_engine.cpp:77` `step_tail(... const int32_t* rad_net ...)` | never | widen at P1 |
| `bindings.cpp:2062-2081` the `TemperatureSolver.step` direct binding (gate 5's entry) | never | widen at P1, **`py::array_t<int64_t, py::array::c_style>` without `forcecast`** |
| `bindings.cpp:3176-3281` the `step_tail` binding | never | same |
| `cuda_temperature.h:110`; `cuda_temperature.cu:202`, `:220`, `:487`, `:568` (`cudaMalloc nb`), `:583-585` (`cudaMemcpy nb = n·4` bytes) | never | widen pointer, malloc size and memcpy size at P1 |
| `bindings.cpp:609-611` (CUDA cast), `:2276-2278` (CPU cast) | P3 | die |
| `cuda_raycaster.h:115-121`, `.cu:193`, `:376`, `:402-404`, `:420-424` | P4 | die |
| `tests/test_pf1a_radiation_books.py:100-102`, `test_pr1_fire_plane_cast.py:146-148`, `test_fire_heat_source.py:85-89`, `cuda_pr1_fire_plane_check.py:76` (`np.int32` harness planes) | P3/P4 | die |
| the gas chain's `deposit_dT_wide_q16` (`fixed_point.h:307`), int32 first operand | — | the staged wide chain of §2.8, both backends |

**Why the omissions were silent.** pybind11's `py::array_t<T>` carries
`forcecast` by default, so an int64 array handed to an int32 parameter (by
value at `:609`, `:2276`, or through `.cast<py::array_t<int32_t>>()` at
`:2068`, `:3241`) is converted into a *temporary truncated copy* with no error;
an un-widened reader folds wrapped garbage, an un-widened *output* parameter
writes into the temporary and is discarded, and the CUDA temperature twin
copies half the plane. The CPU goldens would still pass at P1 (the old cast's
values are in int32 range, so a truncated copy is exact) while
`tests/cuda_thermal_mass_check.py` and its siblings go red — which is exactly
the class of failure the two surviving bindings are made **loud** against by
dropping `forcecast`. All of the "never" rows are one P1 commit.

**Goldens unmoved at P1** holds under four conditions, each asserted rather
than believed (critique 3 §6c): (i) the old cast's arithmetic is unchanged —
but its documented out-of-band *wrap* contract (`raycaster.h rad_signed_add`;
`tests/test_pf1a_radiation_books.py:322-340`, a firestorm at "96 % of
`INT32_MAX`") changes meaning the day the accumulator is int64, so that test
is **re-dispositioned at P1**, not P3 (an int64 accumulator makes the scene
exact and the assertion says so), and the A/B default scenario is asserted
wrap-free once (`max|rad_net| < 2³¹`); (ii) the fourth `stamp_units` output
leaves the three existing outputs' bytes unchanged; (iii) the clamp is
dormant with `rad_fluence == nullptr` and the int64 `shr_round0` equals the
`q16` one on int32-range input, asserted in the kit's test; (iv) step 2b
writes only the four wiped planes. The Recorder's `_INT64_FIELDS` rule
(`recorder.py:94`) applies only if a future session records these planes;
none is in `DEFAULT_FIELDS` and nothing passes them in `fields=`.

**Headroom.** `E°[3999] ≈ 3.6·10¹²`; a cell can gather at most 16 ordinates' worth
of a full-power neighbour plus the ambient inflow, so `|rad_net| < 2⁵⁰` by a wide
margin; the per-tile sums never approach `2⁶³` and no arithmetic relies on
wrapping (§8.2).

---

## 4. The producer contract — what a radiation producer exposes

Erik's ask: formalise what "raycasters" expose going forward, so the old engine
can be retired without losing a capability nobody remembers. This section is
that contract. It is deliberately about *what*, not *how*, so cascades (§7.4)
or any future producer can satisfy it.

### 4.1 Inputs a producer consumes

| input | heat | light |
|---|---|---|
| material extinction, per cell | `heat_atten_q` | `light_atten_q` (h,w,3) — P6 |
| stamped extinction, per cell (bodies) | `dyn_heat_atten_q` | `dyn_light_atten_q` (h,w,3) — P6, the integer twin that replaces the float plane |
| medium extinction, per cell (gas) | the smoke term from `[gases.*] heat_absorb` × density (§6.3) | the per-gas RGB `absorption` × density, quantized — P6 |
| thermal emission | `E°[T]` (`emissive_table.h`) | `L°[T]`, a second baked table (visible-band power, from the blackbody ramp `renderer/blackbody.py`, quantized at load, door 2) — P6 |
| cone emitters | none | a list of `(cell, rgb_q, angle_center, angle_spread)` — lamps, beacons, flashlights, transient weapon glow. **No range, no ray count, no jitter, no heat** |
| boundary | the ambient ring at `E°[0]` | the directional sky BC: per-ordinate RGB on the virtual ring, plus a flat floor dial |
| leak | `k_leak` | none |
| stability | the Fleck pre-pass (from `temperature` + capacities) | none (light has no material feedback) |

### 4.2 Outputs a producer delivers

| output | meaning | consumer |
|---|---|---|
| `rad_net` | signed material ledger, heat counts | Pass-1 fold |
| `rad_flux` | body sensor, **signed**: net absorbed above ambient, heat counts (negative only in another body's shadow; the consumer floors it at 0) | `exchange.py` unit heat damage |
| `rad_amb` | net ambient ledger | the conservation gate; diagnostics |
| `rad_fluence` | Φ per cell | the Pass-1 clamp; `pack_hover_readout` |
| `light_q` (h,w,3) | integer RGB irradiance per cell | the rules (stealth), RL observation — P7 |
| `light_flux_q` (h,w,2) | the net flux vector `Σ_m I_m ŝ_m`, integer | the renderer's `light_dir` (normal-mapped shading) — free: two more multiply-adds per cell-ordinate |
| `light_glow` (h,w,3) | in-scattered glow at gas cells (`stream × scatter_albedo × density`), render-only | the god-ray / gas-medium pass |
| counters | `rad_clamp_hits` etc. live on the solver that owns the clamp, not on the producer | tests, telemetry |

### 4.3 The one accessor

`src/simulation/light_field.py` (new, P6): `read_light(gmap)` for the renderer
(dequantized views + the flux vector), `light_at(gmap, y, x)` for rules and RL.
**Nothing outside may know how the light field was produced.** `LightingPass`
(`renderer/lighting.py:293`) stops being a solver caller and becomes a consumer
of the accessor; the canonical "LightingPass … never a second raycast" and
"Frame lights" rows survive with that one change of input (critique 1i).
`gmap.light_map` (`gamemap.py:380`, "legacy") becomes a derived scalar from the
accessor at P6 and is deleted when its two render consumers
(`game_renderer.py:823`, `lighting.py:304`) read the accessor directly.

### 4.4 What the old raycaster exposed, and where each thing goes

`cpp/src/raycaster.h` (933 lines) + `.cpp` (1235) + `cuda_raycaster.cu` (446) +
`.h` (127), bound at `bindings.cpp:399-730` and `:2083-2135`.

| old member | fate |
|---|---|
| `LightSource {x, y, max_range, ray_count, angle_center, angle_spread, intensity, heat, jitter, color}` | → the cone-emitter row `(cell, rgb_q, angle_center, angle_spread)`. `max_range` and `ray_count` have no meaning in a sweep; `heat` was already hard-pinned 0 (`fire_lights.py` header); `jitter` was the only RNG door and is gone (door 4 stays untouched). `src/level_lights.py` and `renderer/frame_lights.py` keep their inputs and change their output row |
| `cast_source`, `cast_source_directional`, `march_ray`, `march_ray_directional`, `march_ray_radiation` (the DDA) | **deleted at P6.** No caller outside the file. Erik: *"ray marching might be something we want to keep — for now I can't say."* It costs nothing to keep in history and nothing to lose: the live sub-tile marcher for weapons is the beam's own integer march in `combat.py` |
| `cast_from_fire_plane`, `build_fire_sources`, `build_fire_ray_list`, `build_ray_list`, `RadSource`, `RadRay`, `RadCtx` | deleted at P3 with the heat law |
| `bake_emissive_table`, `emissive_table`, `E_TABLE_SIZE`, `E_BUCKET_SHIFT`, `e_bucket_of`, `rad_scale`, `kelvin_ambient`, `k_temp_to_kelvin` | → `emissive_table.h`, owned by `PhysicsEngine`; the three dials keep their `[physics]` config homes, assigned to the engine instead of to a `Raycaster` (`physics_runner.py:383-410`) |
| `T_emit_gate`, `radiation_range` / `RADIATION_RANGE_MIN`, `RAD_LIM_SHIFT`, `rad_pair_budget_s`, `Falloff` | deleted at P3 (`config.toml:542`, `:569`) |
| `light_cull`, `heat_cull` | deleted at P6 — a sweep has no rays to cull (critique 3c: `light_cull` is the render march's, so not before P6) |
| `smoke_absorption`, `smoke_absorption_rgb`, `smoke_scatter_albedo`, `smoke_absorb_scale` | the render `gas_medium.py:120`, `:300` reads `smoke_absorb_scale` off the `Raycaster` as "the shared base scale" (`game_renderer.py:513`). At P6 the light channel's smoke extinction is a per-cell integer term (§4.1) and the medium pass reads `[smoke]` (`config.toml:614`) directly — same dial, one owner, no `Raycaster` |
| `set_raycaster_backend` / `get_raycaster_backend` (`bindings.cpp:721`, `:726`) | → `set_radiation_backend` at P4, in the per-solver backend idiom. Callers: `physics_runner.py:169-175`, `tests/bench_s8c_fire_heat_check.py`, `tests/cuda_s2b_raycaster_live_check.py`, `tests/cuda_s8a_check.py`, `tests/cuda_sky_exchange_check.py`, `tests/cuda_thermal_mass_check.py`, `tests/cuda_thermal_mass_eos_check.py`, `tests/_run_cuda_smoke.py`, `tools/run_on_cuda.py` — one-line edits, listed in §11 |
| `normalize_directions` | → the flux vector is normalised at the render read, in `light_field.py` |
| `HEAT_SCALE`, `heat_quantize`, `heat_saturating_add` (raycaster.h) | used by `exchange.py:230`, `field_edit.py:64-81`, the temperature solver. **Move to `fixed_point.h`** at P3 (they are kit functions living in the wrong header) |

---

## 5. How every current ray interaction maps

| Interaction today | Under v3 |
|---|---|
| **Units block light** (`dyn_light_atten`, per-tick MAX) | Unchanged in kind; the plane becomes integer at P6 (§7) |
| **Units block heat** — they do not | **New, at P3**: `dyn_heat_atten_q`, the MAX stamp; the body share absorbs into `rad_flux` (§6.2) |
| Per-material `light_atten` RGB, `heat_atten` | Unchanged as table rows; quantized to the Q16 planes through `optics_fixed.py` |
| Glass light-clear / heat-opaque | Unchanged and structural: two coefficients read by two channels |
| Smoke attenuating light | The render `exp` stays render-side until P6; at P6 it becomes the light channel's per-cell integer extinction |
| **Smoke attenuating heat** — does not exist | **New (R-M), its own patch** (§6.3) |
| Kirchhoff (ε == a) | Unchanged and now exact: `a_i` appears identically in `abs_mat` and `emitted` |
| Rule 3, contact faces radiation-inert | **Deleted deliberately** (§6.1) |
| Rule 2, mutual pairs at half weight | Subsumed — there is no pair term |
| The flux limiter | **Replaced** by Fleck + clamp (§2.8) |
| `rad_flux` → unit heat damage | Kept; the quantity changes from incident-per-ray to absorbed-per-tick and the dials are re-derived (§6.2) |
| Weapon beams | **Unchanged, out of scope** — their own integer Beer-Lambert (`combat.py:1057-1200`, `GasTable.beam_absorb_q16`, `gases.py:148`); never used the raycaster |
| Line of sight, cover | **Unchanged** — `vision.ray_clear` (`vision.py:88-103`) walks `gmap.has_los` (Bresenham) plus `cover_system.blocks_segment` (`cover_system.py:84-100`); neither touches the ray engine. Survey §10.2's disposition, closed |
| Flashlights, lamps, beacons | Cone emitters (§4.1); cost stops scaling with light count |
| Fire's own light | Every burning cell emits through `L°[T]`; **`renderer/fire_lights.py` (the 16-light cap, its NMS, `t_light_min`, `[render.fire_lights]` `config.toml:1254-1260`) is deleted at P6** |
| Ambient light | The directional sky BC (§2.7) |
| `gmap.light_map` | Derived from the accessor, then deleted (§4.3) |
| `range_base`, `range_per_intensity`, `intensity_base`, `intensity_per_intensity` (`config.toml:576-578`) | The first two are `rad_flux`'s legacy reach guard (assumption A7; `test_pr1_fire_plane_cast.py:401`) and die **at P3 with the `rad_flux` rework**, not before; the second two are the fire's *render* light intensity and die at P6 |

---

## 6. Contact, units, and gas

### 6.1 Touching solids exchange radiation, and we keep it

An opaque cell **emits**. A's emission lands entirely in B, B is hotter next tick,
and it chains: measured with radiation only, a held 1263-game face drives heat
ten cells into a wall in 17 seconds. Radiation and conduction are **parallel
channels, summed** — the textbook combined conduction–radiation model
(`k_rad = 16σT³/(3β)` in the optically thick limit, added to the molecular
conductivity), which is Erik's ruling in that formulation.

Measured, radiation against the conduction face at the same gap: hull/steel
0.0× / 0.2× / 1.5× (20-unit gap / fire gap / 3000→300), glass 0.3× / 0.9× / 12×,
wood **7.1× / 41.7× / 392×**.

Two honest statements. **This is teleportation error** (Wollaber §5.2: a
histogram temperature field is documented as inadequate against it) used
deliberately as the contact-spread channel. **The magnitude is a choice**: a
one-tile mean free path inside a solid, so at 40:1 the radiative term largely
erases wood's low conductivity.

The `config.toml` furniture row is a placeholder test crate whose
`conductivity = 0.0` exists so a lone crate burns in isolation during tuning —
now commented in the table (`config.toml:1551-1553`, commit `678d9b6`). Do not
generalise from it.

### 6.2 Units block heat, and absorb into `rad_flux`

**Blocking without absorbing is not available** in a conservative scheme; scatter
needs a second sweep and reflection is excluded by R-A. So a body absorbs, and
its absorbed integer needs a home: `rad_flux`, whose one consumer is
`exchange.py:299-340`.

**The mechanism is the body share** (§2.3): the unit's `heat_atten` (default 1.0)
is MAX-stamped into `dyn_heat_atten_q`; the material keeps `a_i` of the stream
and the body takes `d_i − a_i`, **and re-emits the ambient level** (row 25). A
marine on a crate: the crate absorbs 0.5, the marine the other 0.5 (stamped to
1.0). A marine on air in a fire's stream: the marine keeps the excess above
ambient and passes ambient on, so what stands behind it is in the fire's shadow
but not the room's — a wall behind a marine is neither warmed by the fire nor
drained by the marine. Ordering needs no change:
units move at slot 3, obstacles re-stamp at slot 6 (`simulation.py:1392`),
physics at slot 7 (`:1405`). **The stamp is frozen before the sweep touches it**
(critique 4a, verified).

**`rad_flux` changes meaning, and "consumer unchanged" is withdrawn** (critique
3f). Today it carries an undebited *incident* estimate `τ·w·a_s·E°[T_s]` per ray
at air cells (`gamemap.py:501-512`); under v3 it carries the **absorbed energy
per tick in the fold's own currency** — the same currency as `heat`. Consequences:

- It is a ledger **exit** now (debited from the stream), so it is in the identity
  at P1 (BLOCKING 6), not "an accepted named leak".
- **`max(heat, rad_flux)` stays.** Both are now energy-per-tick at the tile
  (combustion's deposit at a burn site; radiation absorbed by the body), so "the
  larger of two exposures in one currency" is *more* coherent than before, not
  less. Ruled here; Erik may overrule.
- **The five `[combat]` dials are re-derived, not felt** (row 3):
  `unit_absorption`, `unit_reflectivity` (kept as the body's own optics — and
  `unit_absorption × (1 − reflectivity)` should become the unit's stamped
  `heat_atten`, one number instead of three), `heat_flux_to_temp`,
  `heat_ambient_ref`, `heat_overtemp_scale`. With a physical currency (§9) the
  absorbed power per tick converts to W/m² on the body, and the burn
  thresholds come from the literature (the survey's 10–12 kW/m² wood ignition
  is the same table) rather than from play. The HUMAN-TEST on P3 checks the
  result; it does not set it.
- **Known limitation, accepted**: flux→damage has no thermal inertia. The
  additive upgrade is a scalar unit temperature heated by `rad_flux` and cooled
  toward ambient — and once units have a temperature they **emit for free**,
  which is infravision. `exchange.py` already reads a per-tile plane with
  per-unit absorption applied (critique 3f is right that v2 mis-described this as
  hardcoded); the upgrade is additive against it.

### 6.3 Gas: glowing smoke, the density law, and where the stability problem lives

Under R-M smoke absorbs heat. The emission half is **free**: a gas cell has a
temperature and the temperature map is the emission map, so warmed soot glows
with no extra machinery — the honest version of the black-body smoke idea, which
was a render-only advected field only because the old engine could not afford
emitters. It self-limits correctly: soot in the flame glows, smoke a metre away
is cool and dark.

**The writer** (BLOCKING 5, verified). The sweep writes `rad_net`, exactly as
today. The only thing stopping gas from receiving radiation is the `ts[i]` mask
on the Pass-1 fold (`temperature_solver.cpp:247`). Its gas branch two dozen lines
down already deposits through the seam and books itself —
`gas_energy::deposit_railed` then `e_gas_deposit_sum += nb·dT` (`:373-388`), with
`e_gas_rail_sum` for the rail. So the patch **opens a mask and reuses a booked
channel**: radiation-into-gas is **group 1** of the closure identity with the
counters that already exist. The sweep never calls `gas_energy_*`, and there is
no seventh group.

**Six groups, not four** (critique item 15). The shipped identity
(`tests/test_thermostat_books.py:70-92`) has EOS · thermal-solver gas · combustion
· the Python seams · the water-evacuation export · the P-G5 solid group. The
CLAUDE.md row still says four; it is amended at arc close.

**The density law lives in the extinction** (row 4, Erik: *"it must ride the
density proportional law"*). The gas cell's material extinction is

```
    a_gas,i  =  min(ONE,  Σ_g (heat_absorb_q16[g] · N_g,raw[i]) >> 16)       and  0  if  N_bulk,raw[i] < N_EPS_RAW
```

— a new per-gas scalar column `heat_absorb` in `[gases.*]` (`gases.py:92-99`
`_SCALAR_COLUMNS`), quantized at table build in the `beam_absorb_q16` idiom
(`:148-165`), default `0.0` so every gas is dormant until the patch that opens
the mask. Absorption is proportional to the number of absorbers, which *is* the
density-proportional law, and what thin smoke does not absorb **continues down
the stream** to the next cell. The Pass-1 v2.4 factor `min(N, N_AMB)/N_AMB`
(`temperature_solver.cpp:344-358`) is therefore **not applied to `rad_net`** — it
exists for combustion's `heat` deposit, which is computed as a full-cell quantity;
`rad_net` is already the absorbed amount. Applying both would debit the stream
in full and destroy the un-absorbed remainder into `e_deposit_drop_sum`: counted,
but a thin smoke would then shadow like a thick one. If Erik's ruling meant that
code site specifically, that is the alternative and that is its cost.

**The `N_EPS_RAW` floor** (`gas_energy.h:48`, "ONE value, every file"): a
sub-`N_EPS` cell is *defined* to read ambient, so it must not emit at any other
temperature; `a_gas = 0` there makes both its absorption and its emission vanish
together, using the canonical value rather than a new threshold.

**The Pass-1 gas radiation branch** (the new code, at the patch): for an
accountable gas cell with `rad_net[i] ≠ 0`, convert the signed heat counts to a
signed `dT` through the same wide chain the deposit uses (magnitude then sign,
the `shr_round0` symmetry idiom; the int32-first-operand limit is lifted, §3),
apply the clamp to the deposit (§2.8), then `deposit_railed` + the group-1
counter. Cooling below ambient rides the once-per-tick recovery rails
(`eos_solver.cpp` step 7), which run before the fold. Radiation alone cannot
drive a gas cell below ambient: every stream is at least `amb_m` (the sky
supplies it, a cell at or above ambient passes at least it on, and — since row
25 — a body re-emits it), so a cell at ambient absorbs at least what it emits.
Critique 3 §4f showed this was *false* under the pure-sink body: shadowed smoke
cooled below ambient. Erik's ruling restored it.

**Gas stiffness is bounded and modest, and density cancels**
(`smoke_heat_capacity_study.py` §4, correcting §§1–2): `g ~ φ_soot · E°/T_abs`.
A decompressing room does not get stiffer as it empties; even pure soot reaches
only `g = 4.2` against a solid's 848 at `T_MAX_PHYS`. Fleck + clamp are still
required on gas; they are not holding back a runaway. The gas `L_q` for the
Fleck pre-pass is the excess `a_gas·(E°[T] − E°[0])` through the deposit chain
(`recip_N`, `recip_cv`).

**Second-order effect, predicted**: radiation heats gas → pressure → wind. A new
EOS coupling, exactly what R-O wanted; stabilising (hot gas expands, thins,
absorbs less).

---

## 7. Light

### 7.1 The same sweep, integer, on the sim clock (row 20)

Light rides the same traversal with step transport and three RGB channels plus
the flux vector (§4.2). It is **integer** — the sweep stays all-integer, the new
TU stays at 0/0/0 on the float ratchet, and the rules channel (P7) reads the
same integer extinction planes as the eye channel, which closes critique 3h/24
(the float `dyn_light_atten`, deliberately outside the cross-GPU contract at
`field_digest.py:104`, can no longer feed synced state). The eye field is the
integer field **dequantized at the render read** (`*_fixed.dequantize_f32`, the
existing convention), with exposure, gamma and the flat floor applied in the
shader as today. §8.5's "monotone function" gate becomes trivially true.

**Clock and language** (critique 7c): the sweep runs **once per sim tick at
24 Hz, in C++ with a CUDA twin**, and the renderer uploads the two RGBA16F
textures once per tick and samples them per frame — which is what `LightingPass`
already does with `light_tex_a/b`, today at up to 60 Hz; only the producer and
the cadence change. Per frame the cost is zero sweep work. Per tick it is two
texture uploads (≈0.5 MB at 128×256, ≈2 MB at 256×512 — a fraction of a
millisecond against the bus) plus three channels and a two-component vector on a
traversal already running (§10). 3D models are drawn at their per-frame
positions and sample the field as it stood at the last tick, at most 41 ms old,
which is how units already relate to the world.

**The upload is not the end state, and the seam knows it** (Erik, 2026-09-15:
*"if we're computing it on the gpu, why can't it just stay there for the
rendering too?"*). The end state is CUDA–OpenGL interop — the twin writes the
light field straight into a GL texture — which is the S8 residency item
(`docs/architecture/engine/02`, render interop) and the architecture Erik wants
to learn in the loop rather than have dropped in. It is not in P6 for two
reasons: the CPU-only build has no device compute and needs the host path
regardless, and interop belongs to residency. The accessor (§4.3) is the seam:
the renderer reads light only through it, so replacing the host upload with an
interop-mapped texture later changes nothing outside `light_field.py`.

**This amends R-P** ("light in GLSL, render-only") and **retires R-S for light**:
no fragment-pass sweep, no compute shader, no raylib GL 4.3 build. R-P's reasons
— no CUDA dependency for a drawing machine, headless training skips RGB — are
preserved differently: the C++ path draws on any machine, and headless training
simply does not request the light channels. Erik blessed the transport collapse
in v2; the clock and language are the part he has not ruled on (§12).

### 7.2 What step costs, and cone emitters

Step's four-point cross is visible as a halo around an *isolated* small source
seen through smoke; more ordinates will not remove it (a stencil artifact); it is
averaged away by clusters, which is why a 117-tile fire rendered cleanly.
Measured ring ripple around an isotropic lamp in a plume: step 1.75×, shear
1.75×, cascades 1.24×.

A flashlight emits only into the ordinates inside its beam — native to a
directional solve, and it gives the hard-edged cone Erik prefers. Its tell at S16
is banding inside the beam (a cone quantised into ordinates).

### 7.3 Fire's light and the deleted cap

Every burning cell emits through `L°[T]`; the intensity is the blackbody ramp's
`T⁴` half (`renderer/blackbody.py`), quantized into a second 4000-bucket table
at load. `renderer/fire_lights.py` (`:66` `max_lights = 16`, `:82-91` NMS), its
15 tests, `[render.fire_lights]` and the "K / n" HUD count die at P6 (row 5).

### 7.4 Diffuse reflection and the cascades seam

Not in P1. Bounce is source iteration for either method; for render-only light
the previous tick's bounce can be fed back so it converges across ticks — not
available to heat. Shear's low diffusion makes bounced light spoky; step's
smoothness is right here, another reason light takes step.

Cascades are not built; they remain a measured, de-risked upgrade for volumetric
smoke (1.24× against 1.75×). The seam that keeps the option open is §4: **the
producer contract and the one accessor**, not a pluggable solver. If ever built
they need the bilinear fix (Osborne & Sannikov §2.5; measured, light in a room
that should be dark drops from 1.05 % to 0.00 %), and breach's one-tile walls
violate the penumbra criterion by construction, which makes sub-tile light
resolution a correctness question for cascades and only a look question for the
sweep.

---

## 8. Determinism and the books

### 8.1 Number ingress — doors 1 and 2 only, and this time it is true

Integer sweep. `E°` is baked once at load in a strict TU (§2.6; the one double
multiply is pre-existing and audited). `w_m` and `amb_m` are integer, door 1.
**`s_m` are checked-in integer literals** for S16 and S12 with a test that
recomputes them within one count (row 30): the obvious route, `cos((m + ½)·2π/16)`
at load, is a libm transcendental and therefore engine/14's case 3, not door 2
— door 2 is a config constant snapped once — and `std::cos` in a sim TU is the
banned list in spirit even where the ratchet cannot see it. The extinction
planes and the gas coefficients are quantized at the table boundary (door 2;
`heat_absorb_q16` in the `beam_absorb_q16` idiom is one float divide then
`quantize_scalar`, door 3 then 2, as that idiom already is). The Fleck factor
by the kit's exact `floordiv_q` (door 1); `E°⁻¹` by a fixed-trip integer binary
search. **`L°` at P6** is baked by the `E°` algebraic pattern (integer `K⁴`, one
strict double multiply, round) or checked in as 4000 constants with a
regeneration script under `tools/` — never from `renderer/blackbody.py`'s
`np.power`/`np.log` at load, which would be a determinism hole the day P7 puts
`light_q` in the digest. **No `exp`, no `pow`, no float divide, no RNG** — door
3 is not opened and door 4 is untouched (Philox stays a swarm-units concern).

The three guards, extended at P1 (critique items 10, 5b), as P1 deliverables:

- `cpp/src/radiation_sweep.cpp` **and `emissive_table.cpp`** join the
  `/fp:strict` list at `cpp/CMakeLists.txt:177-189` (iron rule).
- Both join `tests/test_no_float_in_sim_tu.py:49-56` `SIM_TUS` with a
  `BASELINE` of `{0, 0, 0}` for the sweep, and that file's stale comment
  (`:47-48`, "raycaster.cpp is render-only") is corrected — it has written the
  synced `rad_net` since P-R4. Two honest limits of that ratchet: it scans
  `.cpp` files only, never headers, so 0/0/0 is a statement about the sweep TU
  and not about `emissive_table.h`; and it counts *lines containing the word*
  `float`/`double`, comments included, so the baseline forbids the word in
  comments too. Pre-existing and not v3's: `eos_solver.cpp`, `combustion.cpp`,
  `bulk_transport.cpp`, `sky_exchange.cpp` are on the strict list but not on
  `SIM_TUS`, so "every sim TU" is already false there.
- The ingress lint is a Python AST scan over `src/simulation/**` and cannot see
  C++; `optics_fixed.py` is covered automatically, and the extinction
  invariants (§2.3) are asserted where the lint *does* look, in `materials.py`
  and `optics_fixed.py`.

**The god-file policy is not violated** (critique 4e): the sweep is step 2b
inside physics slot 7; `Simulation.step` gains one `fill(0)`.

### 8.2 CPU↔GPU: one formulation, two backends

- **Within an ordinate**: the gather form (§2.3). Shear's wavefront is a column
  (all cells of column `x` depend only on column `x − sx`), step's is an
  anti-diagonal (the KBA skew). The CUDA twin launches one wavefront at a time;
  every thread reads two stored int64 outflows and recomputes `fa`/`fb` with the
  same shift. **No atomics, no ordering inside an ordinate**, so the integers are
  the CPU's by construction.
- **Across ordinates — the chosen mechanics** (row 29): the 16 ordinates run
  **concurrently**, each with its own stored-outflow plane (scratch
  `(N, 16, h, w)`), and one launch per wavefront *index* covers all 16
  (`blockIdx.y = m`; x-major ordinates take the column index, y-major the row
  index, so the index runs to `max(h, w)`). The per-cell sums into `rad_net`,
  `rad_flux`, `rad_amb`, `rad_fluence` are then integer atomics in this tree's
  idiom — CUDA has no signed 64-bit `atomicAdd`, so it is
  `atomicAdd((unsigned long long*)p, (unsigned long long)v)` as at
  `cuda_temperature.cu:120`, `cuda_bulk_transport.cu:313`,
  `cuda_eos_resident.cu:940`; unsigned wrap *is* the two's-complement signed
  add, so the reinterpreted int64 is exact while the true sum is in range, and
  the headroom argument (§3; critique 3 measured `< 2⁴⁶` per cell, no product
  above `2⁵⁸`) guarantees that. The claim is exactness, not defined overflow.
  **Launch count** at 128×256: shear 256 launches per tick, step 383 (the
  anti-diagonals, `h + w − 1`); at 3–10 µs each on Windows WDDM that is 1–4 ms
  and it is in §10. The route that removes it while keeping the gather
  argument intact — a cooperative persistent kernel with `grid.sync()` per
  wavefront, or a captured CUDA graph — is the S8 residency direction, not P4.
  Sequential ordinates with `(N, h, w)` scratch and no atomics at all is the
  other consistent choice; it costs 16× the launches and is not taken.
- **The one division** is an int64 `/`, exactly specified on every backend.
- **Residency** (RL-batch habits §A, critique item 12): the twin follows the
  `cuda_resident.h` precedent — a `radiation_sweep_launch_resident(...)` core
  born `(N, h, w)` (N = 1 today) that takes device pointers and only launches,
  wrapped by a per-call `*_step` that mallocs, copies in, launches, syncs,
  copies out and frees, which is what `step_tail` dispatches for temperature
  today (`cuda_temperature.cu:535-595`). At P4 the sweep is dispatched that way
  from step 2b on the host mirror on both paths, so per tick it pays H2D of
  `temperature`, the two extinction planes (and the gas planes at P5) and D2H
  of four int64 planes — about 2 MB at 128×256, the same tax the temperature
  twin pays, in §10. No host-side tick logic: the sweep is inside
  `PhysicsEngine::step_tail`, which the resident path already brackets on the
  mirror (`physics_runner.py:1356`). All four rad planes join
  `_RESIDENT_SYNCED` at P1 (§3). Scratch keyed `(N, 16, h, w)`.
- **No dormancy skip.** The old cast early-outs on the host when no emitter
  exists (`physics_runner.py:1557-1560`). The sweep has no such skip: a
  uniform-ambient cell costs the same as any other and computes exact zeros
  (gate 2a), and a "device-side flag" could only skip a *host* launch after a
  readback and a sync, which is host gating with a stall — the thing §A rule 4
  forbids. Cost is set by the grid; that is the thesis.

> **2026-09-25, #78:** the headroom is restated at the fine live scale (`k = 11`, measured by the integer reference at the table top, over-driven): per-cell sums ≤ `2^41.1` (< `2^46`), the sweep loop's plain products ≤ `2^53.5`, the Fleck pre-pass's `a·ex` `2^57.1` (< `2^58`, `2^5.9` below int64), `ex_m·f` `2^61.5` (formed in 128 bits, as before) — 0.6 bits under the resolving scale the gates already exercise. The bake refuses a table top at or above `2^44` (`E_TABLE_TOP_MAX`); `k = 12` would break the `2^58` line (`a·ex` `2^58.1`) — `sweep_fine_heat_currency_brief_78_2026-09-25.md` §4.

### 8.3 What enters the digest (critique items 6b, 6e)

Per §3: **no new `DIGEST_FIELDS` entry** in P0–P6. The heat channel reaches the
digest through `temperature` and `heat` as today; the per-tick planes are wiped
before every snapshot (`field_digest.py:50-52`); the extinction planes are pure
functions of digested state. The A/B lockstep harness carries `heat_atten_q`
and `dyn_heat_atten_q` (and at P6 the integer light planes), beside
`dyn_light_atten` in `SIM_FIELDS` (`field_ab_harness.py:87`).

**P7 is a `DIGEST_SPEC_VERSION` bump plus regeneration of every committed
golden in the same commit** (`field_digest.py:39-44`; the CLAUDE.md "Field
digest" row) — not a re-baseline — on the day a rule reads `light_q`. Until
then the rules-side field is computed and not digested: a field nothing consumes
cannot change behaviour, and entering the digest is a one-way door.

### 8.4 The sweep→fold boundary (critique item 17)

§2.3's identity holds on the sweep's own books. The channel into temperature is
**not** exact: the fold truncates (`shr_round0(rn, his)`, `:255`; the gas wide
chain), the rails clamp (`:262-294`), and the maximum-principle clamp clips.
The solid ledger closes because it books the *applied* ΔT, so what the fold
dropped is the difference between two ledgers — a pre-existing property of the
fold, not a v3 defect. v3 states the boundary and gates it: **`Σ rad_net −
Σ applied` is bounded by one truncation LSB per touched cell plus the counted
rail and clamp drops** (`t_max_phys_hits`, `t_low_rail_hits`, `rad_clamp_hits`,
`e_rad_clamp_drop_sum`, `e_gas_rail_sum`). "Exactly conservative inside the
sim" is withdrawn as a phrase; "exactly conservative on the sweep's books,
bounded and counted across the fold" is what is claimed.

> **2026-09-25, #78:** the fold converts out of the sweep's fine currency ONCE per touched cell — a solid's one shift by `his + k`, the gas chain at its final narrow (its stage-1 floor now costs `recip_cv / 2^(32+k)` of an LSB: the declared bound fell from 132·C to 2·C at the shipped currency) — so the unlanded remainder is under one temperature LSB × capacity per cell, the temperature field's own resolution; gated per tick on the smoke scenes and the breached rooms by `test_temperature_gas_radiation.py` — `sweep_fine_heat_currency_brief_78_2026-09-25.md`.

### 8.5 The gate that keeps the two light fields honest

What looks dark to the player must be dark to the rules. With one integer field
and a dequantized view, agreement is structural; the test asserts the render
field is a monotone function of `light_q` on a real scene, and it exists so a
future render-side post-process cannot silently break it. Binary
can-A-see-B stays symmetric shadowcasting over `has_los`, a separate primitive.

---

## 9. Calibration — derive, do not fit (P2)

Erik, 2026-09-15: *"we want radiation to radiate physically — which is possible
now, it will just fall out — then the tuning will be more in the heat capacity
of items, which will determine how fast they will heat up and cool down."*

That is a change of what "calibration" means. Today `rad_scale`
(`config.toml:526`) is a fitted dial. Under v3:

1. **The currency is defined once.** One heat count is the energy that raises a
   `thermal_mass = 1` tile by `1/65536` game degree. Pin it to joules by giving
   **one** material — the standard burnable furniture row, per row 6 — its real
   volumetric heat capacity × tile volume. Everything else (every `thermal_mass`)
   is then in joules per kelvin by definition.
2. **`rad_scale` becomes derived**: `σ · A_face · Δt / J_per_count`, with the
   tile face `0.333 m × 2.5 m` and `Δt = 1/24 s`. No fit. It is written into
   `[physics.radiation]` with its derivation, in place of the fitted key.
3. **`a_i` are emissivities**, from the literature per material row.
4. **The reach curve then falls out** — survey §9.3 derived ~3 tiles for one
   crate and ~8 for a room from view factors and 10–12 kW/m²; the bench
   (`tools/fire_tuning_lab.py`, which today carries no radiation dial at all and
   a single-station `IGNITE_TILES` at `:60` — so P2 is a real multi-probe
   extension, not a dial) measures the engine against that curve.
5. **What is tuned is `thermal_mass`** (how fast things heat and cool) and
   `cool_shift`, per material row, starting with the furniture crate and only
   then wood, doors, and later the tree rows. #61 ("wood cannot sustain fire") is
   read as a heat-capacity problem, as Erik said, and is fixed as a property.

**Calibrate against the damped emission, not against `E°[T]`** (row 34, P0
§0.6). What a cell radiates is `E°[0] + f·(E°[T] − E°[0])`; at the 1263-game
plateau `f = 0.73` under α = ½, at the table top `f ≈ 1/851`. The reach curve
that "falls out" is the one the damped law produces, and §12 item 4 asks Erik
whether the fire range should be damped at all.

The old `k_fire_heat`-style constants and the survey's "calibration risk" are
therefore closed by derivation. The HUMAN-TEST on P3 confirms feel; it is not
where the numbers come from.

---

## 10. Cost

| | per tick | scales with |
|---|---|---|
| Heat sweep, 128×256, S16 | 524 k cell-updates — **measured (P1, one CPU core): 5.3 ms shear, 4.9 ms step, ≈10 ns per cell-update** | grid × ordinates |
| Heat sweep, 256×512, S16 | 2.1 M cell-updates — **measured: 39.0 ms shear, 20.2 ms step**; shear's x-major wavefront walks a column (stride `w`), cache-hostile at `w = 512` (18.6 ns per update). The CUDA twin is the path for this size; a transposed traversal for x-major ordinates is a later CPU optimisation | grid × ordinates |
| Light on the same traversal | + 3 channels + 2 vector components per cell-update | grid × ordinates |
| Fleck pre-pass + clamp | 1 division + 1 binary search per cell | grid |
| CUDA twin: launches per tick (§8.2) | shear 256, step 383 at 128×256 (one per wavefront index, all 16 ordinates per launch); ≈1–4 ms at 3–10 µs per launch on Windows WDDM | max(h, w) or h + w |
| CUDA twin: host↔device copies per tick (P4, pre-residency) | H2D `temperature` + two extinction planes (+ gas at P5), D2H four int64 planes: ≈2 MB at 128×256 | grid |
| Per frame (renderer) | two RGBA16F uploads per tick; sampling per pixel as today | — |

Nothing scales with the number of burning tiles or light sources. **The budget
is 41.67 ms at 24 Hz** (R-H); at shipped scale `tests/_eos_p3_bench.py` reports
the whole `Simulation.step()` at p50 1.6 ms / p99 9.78 ms, so about **32 ms of
headroom**. The survey's "18.97 ms atmosphere group" was that same bench timing
the whole step at 25 600 cells under hostile load — a mischaracterisation the
survey should correct, not this document. The per-system gate in the older perf
docs (`p99 <= 0.25 * 83.0`, `_eos_p3_bench.py:370`) is stale documentation, to
be restated against 41.67 ms.

**The number that is not known is a single cell-update's cost**; estimating ~5 ns
puts the four-channel sweep near 10 ms per tick at 128×256 on one CPU core and
four times that at 256×512, i.e. the CUDA twin is the path for the full-size
map. **P1 carries a timing bench and prints the picture**, per the arc's rule
that a scalar summary is not trusted until the profile has been looked at.

---

## 11. Migration and patch plan

**Sequence and why**: the integer reference first (the piece most likely to
force a redesign); then the sweep in shadow, so its gates run against the engine
without moving a golden; calibration on the shadow output; then **one** flip
patch in which the old heat law, its tests and its tools die together and units
start absorbing (`rad_flux` must not lose its writer, row 24); the CUDA twin;
smoke heat, which completes the heat physics; light, which is the patch that
finally lets the old raycaster be archived; then the rules-side payload.

| # | patch | gate | risk | human |
|---|---|---|---|---|
| **P0** — **DONE**, merged `40f2479` (2026-09-15 night; 13 tests, 1.4 s; `report_p0.md`) | **The integer reference**, `docs/ray_engine_v2_scheme_study_2026-09-13/sweep_ref_q.py`: a numpy int64 transcription of §2.3 in **gather form**, both transport steps, leak, virtual ambient ring, body share, the excess-form Fleck factor and the corrected clamp (rows 21–23). Re-runs the stability and equilibrium tables on the excess form | gates 1–5 in the reference; agreement with the float push reference (`sweep_ref.py`) to the shift truncation; the stability tables reproduced | the arc's design risk, retired first | — |
| **P0b** — **DONE**, merged 2026-09-15 night (14 tests 1.5 s; also pinned LF in `tools/gen_fire_studio.py`, which was byte-deterministic per platform only) | **The reference goes to Q24 `f`** (row 32) with a new gate G12: the damped source is monotone in `T` over the whole table for every shipped material row; the driven-source table for both α floors (§12 item 4), a measurement for Erik, not a change; the `.gitattributes` `*.csv text eol=lf` fix so a fresh worktree does not rewrite `levels/fire_tuning/tilemap.csv` to CRLF and fail `test_fire_tuning_level.py::test_generator_is_byte_deterministic` (P0 found it; every worktree agent on this machine hits it); `report_p0.md` and the pytest wrapper updated | all P0 gates re-run green; G12; the byte-determinism test green in a fresh worktree | small | — |
| **P1** — **DONE**, merged `2471fc1` (2026-09-16; gate 0: 34 configurations bit for bit against the reference; gates 1–6; suite 2552 passed / 0 failed with the CUDA gates; goldens untouched; 5.3 ms per tick at 128×256) | **The sweep, CPU, heat, in shadow.** `radiation_sweep.{h,cpp}` (step 2b of `PhysicsEngine::step`, `/fp:strict`, ratchet 0/0/0) · `emissive_table.h` with `E°⁻¹` · `optics_fixed.py` + `heat_atten_q` + `dyn_heat_atten_q` (the `stamp_units` signature grows by one input, one output, one row array; `tests/test_stamp_units_cpp_ab.py` extended) · `rad_fluence` + the three int64 widenings · the kit's int64 `shr_round0` and wide deposit twin · the Fleck pre-pass · the clamp code in Pass 1 (solids), **dormant on the live path** (`rad_fluence == nullptr` → no clamp) and exercised by direct-binding tests · `pack_hover_readout` rows for Φ, `a`/`d`, `f`, `E°⁻¹(Φ)` (critique 1j) · a timing bench. **Not wired**: the old cast still feeds the live planes; the sweep writes the shadow planes `rad_net_sweep` / `rad_flux_sweep` / `rad_amb_sweep` + `rad_fluence` (§3). **Also in P1** (critique 3): the complete int64 inventory of §3 in one commit, the two surviving bindings without `forcecast`, the CUDA temperature twin's pointer/malloc/memcpy widened, the `FP_HD` int64 `shr_round0`, all four rad planes into `_RESIDENT_SYNCED`, the `emissive_table.cpp` strict TU, the `s_m` literals + recompute test, the re-disposition of `test_pf1a_radiation_books.py:322-340`'s wrap-contract scene, the one-time `max\|rad_net\| < 2³¹` assertion on the A/B scenario, `_stamp_units_python` grown with the C++ stamp | **gate 0** (bit for bit against `sweep_ref_q.py`, the executable spec — this makes P1 oracle-gated) and gates 1–6 and 9 (goldens unmoved, asserted under §3's four conditions); the eight new property gates written here are the ones P3 keeps; the CUDA temperature gates stay green (the widening is complete) | the whole design | — |
| **P2a** — **DONE**, merged 2026-09-16 evening (Opus, 277k tokens; gate 0 34 equal; suite 2529/0 CPU; goldens untouched; gate 4's fast mode found vacuous at floor 0 and fixed) | **The prelude, mechanical** (rows 38, 39): the sweep zeroes its own four outputs at its start and the conductor's four `fill(0)` lines go, so the tile inspector shows Φ, `f` and `E°⁻¹(Φ)` at render time; **floor 0** in `sweep_ref_q.py` first (its gates re-run; G8 against `α = max(0, 1 − 1/g)`, G10's cooling figure becomes −5.7 %, G12 re-checked), then in `radiation_sweep.cpp`'s pre-pass, gate 0 re-run bit for bit; the CLAUDE.md sweep row's formula updated | all P1 gates green; gate 0; goldens unmoved (nothing live changes) | small | — |
| **P2b** — **DONE**, merged 2026-09-16 (Opus, 310k tokens; suite 2529 passed / 0 failed, gate 0 unmoved, goldens untouched; `report_p2b.md`, 825 lines) | **Calibration by derivation** (§9): the currency pinned on the furniture row's real heat capacity with stated assumptions; the derived scale written as `[physics.radiation] rad_scale_derived` and fed to the **sweep's** table only — the old cast keeps its fitted `rad_scale` until P3c, so the live game and the goldens do not move; per-row emissivities proposed in the report, not applied (the material rows are Erik's); the reach bench on the shadow planes (`E°⁻¹(Φ)` crossing `ignition_temp`, per source size, fitted vs derived scale, plotted) extending `fire_tuning_lab.py`; `report_p2b.md`, written incrementally | the survey §9.3 curve, stated against the engine's own temperature-ignition criterion, **against the damped emission** (row 34); goldens unmoved | medium | — |
| **P3** | **The flip.** The fold reads the sweep's `rad_net`; units absorb (§6.2: per-unit `heat_atten`, the body share live, `exchange.py` dials re-derived, `max()` kept); **delete** `cast_fire_heat` (both call sites `physics_runner.py:819`, `:1207`), `cast_from_fire_plane` and its CUDA twin + bindings, `T_emit_gate`, `RADIATION_RANGE`, `fire_ray_count`, `range_base`/`range_per_intensity`, `RAD_LIM_SHIFT`, the pair budget, `set_raycaster_backend`; `rad_scale` re-homed on the engine; `HEAT_SCALE` & co. to `fixed_point.h`. **The clamp's GPU twin lands here** (Erik: inside P3, not P4-before-P3): in `cuda_temperature.cu::temp_convert_unified` (`:194-240`) — `rad_fluence` H2D, the `E°` table H2D, the `FP_HD` `E°⁻¹`, a third hit counter beside `hits`/`low_hits` (`:226`, `:231`), the `TEMPERATURE_ENERGY_SLOTS` enum (`cuda_temperature.h:147`, 13 today) extended with pinned slots — so the temperature backend's existing tol-0 gates stay green on the day the clamp goes live. **`DIGEST_SPEC_VERSION` v6** with `dyn_heat_atten_q` in the same commit as the re-baseline (row 28). **Test surface**: the 46 old-law tests in four files are **deleted with rationale** (table below) — their properties no longer exist and their replacements shipped in P1. **Tools**: dispositions below | full suite; A/B lockstep harness; **one deliberate golden re-baseline** with written rationale (the first golden move of the arc) | HUMAN-TEST | **yes** — fire spread and marine burn |
| **P4** | **CUDA twin** `cuda_radiation_sweep.{cu,h}`, `(N,h,w)`-shaped, wavefront gather; `set_radiation_backend`; `tests/cuda_radiation_sweep_check.py` + `test_cuda_radiation_sweep.py` through `cuda_harness`; delete `cuda_raycaster.{cu,h}` (heat side; the render march never had a GPU path), `cuda_pr1_fire_plane_check.py`, `cuda_s2_check.py`, `cuda_s2b_raycaster_live_check.py`, `bench_s8c_fire_heat_check.py` + `test_s8c_fire_heat_bench.py`; the backend-flag one-liners in the other CUDA checks and `run_on_cuda.py`. The wavefront mechanics of §8.2 (concurrent ordinates, `(N, 16, h, w)` scratch, one launch per wavefront index, the unsigned-long-long atomic idiom); the `*_step` wrapper + `*_launch_resident` core split of `cuda_resident.h` | tol-0 lockstep on every plane **and every counter**, by a check script that calls the engine directly and reads the planes before the wipe (gate 7); the §10 launch and copy costs measured | mechanical, pattern known | — |
| **P5** | **Smoke absorbs heat** (§6.3): `[gases.*] heat_absorb`, the smoke term in `a_i`, the Pass-1 gas radiation branch with the clamp on gas (`ΔE_new = N·T_target_abs − E`, §2.8), the staged wide chain (§2.8), `rad_clamp_hits` / `e_rad_clamp_drop_sum`; **the gas clamp in `cuda_temperature.cu` in the same patch**, with the drop counter in a pinned `TEMPERATURE_ENERGY_SLOTS` slot | the #54 closure identity still closes with six groups (`test_thermostat_books.py`, the §6 benches); the sweep→fold boundary bound (§8.4); a smoke-shielding bench | **books** | — |
| **P6** | **Light on the sweep** (§7): RGB + flux vector + glow channels, `L°[T]` (baked by the `E°` algebraic pattern or checked in as constants — never from `np.power`/`np.log` at load, §8.1), cone emitters, the directional sky BC, `light_atten_q`/`dyn_light_atten_q` (the float `dyn_light_atten` leaves `EXCLUDED_FLOAT_FIELDS`), `light_field.py`, `LightingPass` as a consumer; **archive `raycaster.{h,cpp}`** (git history + `docs/archive/` pointer + banner on engine/08); delete `fire_lights.py` + its 15 tests + `[render.fire_lights]`, `light_cull`/`heat_cull`, `intensity_base`/`per_intensity`, the smoke dials off the `Raycaster`; `light_map` derived then deleted. **Render-side test surface** disposed (table below) | timing on a real map; the look; the A/B harness; §8.5's monotone gate | HUMAN-TEST | **yes** — the look |
| **P7** | **`light_q` for the rules**: the stealth query, the RL observation hook, `DIGEST_SPEC_VERSION` bump + all goldens regenerated | agreement on a real scene; the spec bump procedure | design-complete | — |

**Is P1 wired?** No (critique item 30). The shadow sweep is computed every tick
into planes nothing consumes; gates 1–6 run against the engine's own output,
and the goldens are asserted unmoved. **P3 moves the golden**, once.

### 11.1 Test dispositions — the old heat law (deleted at P3)

Per CLAUDE.md, a failing pre-existing test is a finding, never a target to bend
the design to; these tests pin a law that is being deleted on purpose, and each
line names the property that replaces it.

| file | tests | what it pins | disposition |
|---|---|---|---|
| `tests/test_pf1a_radiation_books.py` | 11 | rule-1 pairs, rule-2 half weight, rule-3 contact faces, rule-4 sky by emitter, the emitter gate's knife edge, `rad_net.sum() + rad_amb.sum() == 0` | **delete.** Pairs, half weight, contact inertness and the gate no longer exist; the identity is gate 1 with three terms; the isothermal-lattice zero (`:399-443`) is gate 2 |
| `tests/test_pr1_fire_plane_cast.py` | 11 | the ledger identity, air inertness, the `rad_flux` reach guard, fan rotation as a function of the tick, the 8-ray fan's touched set, the exact-integer bake | **delete.** Air inertness holds by `a = 0` (gate 2's mixed scene); the reach guard dies with A7; there is no fan; the bake test moves to `emissive_table.h`'s own test (kept as a property: "the bake is the exact int64 chain and saturates only above the table top") |
| `tests/test_fire_heat_source.py` | 14 | the 8-ray fan, `range_base + range_per_intensity`, occlusion, the full chain to ignition, unit burn, determinism, "wired into `Simulation.step`" | **delete 10, rewrite 4**: `test_full_chain_heat_ignites_air_separated_wood`, `test_unit_next_to_fire_loses_hp_and_zombie_takes_4x`, `test_determinism_bit_identical_temperature`, `test_fire_heat_is_wired_into_simulation_step` become end-to-end properties of the sweep (ignition across a gap at the derived reach; a marine burns and a zombie takes 4×; bit-identity; the sweep is step 2b) |
| `tests/test_heat_attenuation.py` | 10 | `heat_atten` as the 4th channel of the render march; air / glass / wall ordering; light-vs-heat independence | **delete 8, rewrite 2**: the ordering (`air < glass < wall`) and the heat-transparent/light-opaque independence become gate-3 scenes on the sweep's extinction planes |
| `tests/test_temperature_convert.py::test_air_tile_receives_radiation_deposit` | 1 | the gas branch's deposit | **survives** (it exercises `heat`, not `rad_net`); the P5 gas radiation branch gets its own direct-binding test beside it |

### 11.2 Test dispositions — the render march (disposed at P6)

| file | tests | what it pins | disposition |
|---|---|---|---|
| `tests/test_heat_smoke_glow.py` | 14 | the march's `heat` deposit + `smoke_glow` | **delete** (the heat side is dead code after P3; the glow becomes the light channel's `light_glow`, gated by a new property test: glow only where gas is, proportional to `stream × albedo × density`) |
| `tests/test_multigas_colour.py` | 8 | density-weighted per-channel `exp` transmission across gases | **rewrite 4, delete 4**: tint, neutral dimming, density-weighted mixing and "empty gas leaves light untouched" become properties of the integer per-gas extinction term; the Beer-Lambert `exp` identities go |
| `tests/test_dyn_light_atten.py` | 5 | `stamp_units`'s dynamic plane (4) + one downstream ray test | **4 survive on the integer twin, 1 rewritten** to "the sweep is occluded downstream of a stamped body" |
| `tests/test_rgb_light_atten.py` | 4 | per-channel material attenuation in the march | **rewrite** as gate-3 scenes on the light extinction planes (opaque blocks, glass ~90 %, asymmetric tint); "aggregate termination" dies (no rays) |
| `tests/test_rgb_light_pack.py` | 3 | the RGB deposit + the 16F pack layout | **1 survives** (pack layout), 2 rewritten against the accessor |
| `tests/test_fire_lights.py` | 15 | the 16-light cap, NMS, `t_light_min` | **delete** (row 5) |
| `tests/test_cuda_s2_raycaster.py` + `cuda_s2_check.py`, `cuda_s2b_raycaster_live_check.py`, `bench_s8c_fire_heat_check.py` + `test_s8c_fire_heat_bench.py` | 2 (+3 scripts) | the old march's GPU bit-identity and batched-cast payoff | **delete at P4**, replaced by `cuda_radiation_sweep_check.py` |

### 11.3 Tool and bench dispositions (critique items 26, 3b)

| consumer | today | disposition |
|---|---|---|
| `tools/storm_ledger.py:291-293` | wraps `runner.cast_fire_heat` **by name** through `getattr` | at P3 the wrapped name becomes the engine step (the sweep is inside `step_tail`); the ledger's "fire_cast" pass label maps to the sweep's planes. Edited in P3, run before merge |
| `tests/bench_s8c_fire_heat_check.py:108`, `:115` | reads `fire_ray_count`, `range_base + range_per_i` | deleted at P4 (its payoff question — batched vs per-source — has no meaning for a sweep) |
| `tools/fire_tune_loop.py:515`, `tools/storm_probe.py:66` | `T_emit_gate = 310.0` as a dial override | the key is removed at P3; both tools drop the row (they read config keys by name, so a stale row would raise) |
| `tools/fire_timing_harness.py:259` | documents the tick order by function name | comment updated at P3 |
| `tools/fire_tuning_lab.py` | no radiation dial, one ignition station | extended at P2 (§9) |
| `tools/lighting_demo.py`, `main.py:446-590`, `renderer/frame_lights.py`, `src/level_lights.py` | build `bp.LightSource` lists | at P6 the row type changes (§4.4); the assembly seam survives |
| `renderer/gas_medium.py:120`, `:300`, `game_renderer.py:513` | `smoke_absorb_scale` off the `Raycaster` | read `[smoke]` directly at P6 |
| `tools/run_on_cuda.py`, `tests/_run_cuda_smoke.py`, `cuda_s8a_check.py`, `cuda_sky_exchange_check.py`, `cuda_thermal_mass_check.py`, `cuda_thermal_mass_eos_check.py` | `set_raycaster_backend(True)` | one-line rename at P4 |
| `src/simulation/vision.py`, `cover_system.py`, `combat.py` beams | never used the raycaster | **no change** |
| `gmap.light_map` | legacy scalar (`gamemap.py:380`) | derived then deleted at P6 |

### 11.4 The archive step (P6)

`git rm cpp/src/raycaster.h cpp/src/raycaster.cpp` (the CUDA pair already went
at P4), the `bindings.cpp` blocks, the CMake entries. Then: a line in
`docs/archive/README` (or the arc's archived working docs) naming the last
commit that carried the files; a supersession banner at the top of
`docs/architecture/engine/08_ray_engine.md` pointing here (the chapter itself is
rewritten at the deferred canon fold, and until then it is the description of
record for the *technique* Erik may want back); the CLAUDE.md "Ray engine"
canonical row replaced by the rows in the Systems section below.

---

### 11.5 Implementation agents, per patch

Erik's rule (2026-09-15): a core patch that touches three or more systems at
once warrants a **Fable** implementer; a normal patch is **Opus**; purely
mechanical work is **Sonnet**. One implementation agent at a time, in its own
worktree, per the master rules; critics one at a time, Opus unless the
document's complexity warrants more.

**Amended 2026-09-16 (row 40, issue #66).** P1 on Fable was killed twice by the
Fable session limit; a single resume for its report cost 687 000 tokens. Erik:
*"cutting the patches is where our leverage is"*, and the Fable weekly limit
stood at 86 % after one day. So: **no Fable implementer this week**; every
remaining patch is cut into steps small enough that a kill is cheaper to redo
than to resume (one system, a handful of files, one gate); every implementer
writes its `report_pX.md` **incrementally** as it goes, so a kill leaves the
findings on disk and no resume is ever needed; the orchestrator session moves
to **Opus** for the execution phase. The table below is the cut plan; the
original tiers are kept in the rows' history.

| patch | systems touched | agent | mechanical sub-tasks for Sonnet, split out |
|---|---|---|---|
| P0 | one study script | **Opus** (the arithmetic must be exactly right, but it is one file against a float reference) | — |
| P0b | the reference, one test file, `.gitattributes` | **Opus** | — |
| P1 | the sweep TU, the emissive table, `optics_fixed.py`, `stamp_units` (C++ and Python), `GameMap` planes and residency sets, the Pass-1 clamp, the kit twins, four bindings, the CUDA temperature twin's widening, `pack_hover_readout`, the build and ratchet lists | **Fable** | — (the widening inventory is one commit and must be done by the hand that understands `forcecast`) |
| **P2a** | the sweep's self-zeroing (row 38), floor 0 in the reference then the engine (row 39), gate 0 re-run, the CLAUDE.md row text | **Opus** | — |
| **P2b** | the calibration derivation (§9): the currency, the derived scale as `[physics.radiation] rad_scale_derived` feeding the sweep's table only (the old cast keeps `rad_scale` until it dies), emissivities proposed per row (not applied — Erik's rows), the reach bench on the shadow planes (`E°⁻¹(Φ)` against `ignition_temp`), `report_p2.md` | **Opus** | — |
| **P3a** | the flip's core: widen the live planes, the fold reads the sweep's planes, the clamp live (CPU and the CUDA temperature twin, one patch), `dyn_heat_atten_q` into the digest (spec v6), the one golden re-baseline | **Opus** (was Fable) | — |
| **P3b** | units absorb: per-unit `heat_atten`, the body share live, `exchange.py`'s dials re-derived, `max()` kept — **HUMAN-TEST** | **Opus** | — |
| **P3c** | delete the old heat law: `cast_fire_heat` and both call sites, the dials, `RAD_LIM_SHIFT`, the pair budget, the CUDA cast bindings; the 46 old-law tests deleted with rationale; the five tool edits; the `HEAT_SCALE` move | **Sonnet** (mechanical, suite-gated) | — |
| P4 | one new `.cu` + header, the backend flag, one check script | **Opus** (the pattern is known; the wavefront shape is the only new thing) | the backend-flag renames in six CUDA checks and `run_on_cuda.py`; deleting the three old check scripts |
| **P5a** | the gas table column `heat_absorb`, the sweep's smoke term, gate 0 extended to gas cells in the reference first | **Opus** (was Fable) | — |
| **P5b** | the Pass-1 gas radiation branch with the clamp on gas, its CUDA twin, the closure books (six groups still close) | **Opus** | — |
| **P6a** | the light channels on the sweep (RGB, flux vector, glow), `L°`, the integer light planes, reference first | **Opus** (was Fable) | — |
| **P6b** | the accessor, `LightingPass` as a consumer, cone emitters, the directional sky BC — **HUMAN-TEST** (the look) | **Opus** | — |
| **P6c** | archive the raycaster, delete `fire_lights`, the render-side test dispositions, the digest exclusions | **Sonnet** (mechanical) | the archive step (file removals, `docs/archive/` pointer, the engine/08 banner) |
| P7 | `light_q`, the stealth query, the RL hook, the spec bump | **Opus** | the golden regeneration run |

---

## 12. Still open — Erik's

1. **`max(heat, rad_flux)`** kept, both now energy-per-tick (§6.2).
2. **The leak channel**: designed, dormant, derived (`k ≈ 0.10`), unruled — and
   if adopted, whether `k` is per level or per cell from the outset.
3. **`unit_absorption × (1 − reflectivity)` folded into one per-unit
   `heat_atten`** (§6.2) — a naming call that lasts.
4. **The Fleck α floor for DRIVEN sources — RULED by Erik, 2026-09-16: floor
   0** (*"Let's go floor 0 then."*), implemented at P2a, reference first (row
   39, §2.8). The case as it was put to him (P0 §0.6; the orchestrator's
   finding, 2026-09-15 night): The Fleck factor assumes the cell will cool
   during the tick, so a cell *held* at its temperature by a deposit — a burning
   tile, fed by combustion every tick — emits `f·(E°[T] − E°[0])`: **73 % of
   black body at the 1263-game plateau** under the ruled `α = max(½, 1 − 1/g)`,
   0.12 % at the table top. The Fleck bound `α ≥ ½` is IMC's (for its own
   linearised equations with effective scattering); for *our* material update
   the monotone-stability condition is that the fixed-point multiplier
   `x = g·(1 + αg/4)/(1 + αg)²` stays in `(0, 1]`, and `α = max(0, 1 − 1/g)`
   satisfies it for every `g` (`x = g` below `g = 1`, `x = (g + 3)/4g ≤ 1`
   above). That choice leaves the fire range **undamped below `g = 1`** — a
   burning tile radiates physically there, which is what Erik asked for in
   row 3 — and caps the per-tick loss at `T_abs/4` above it, exactly where the
   explicit scheme fails. **P0b measured both floors** (`report_p0.md` §0.8,
   `p0b_alpha_floor.png`), and two things the prose above did not anticipate:
   `g = 1` is reached at 1424 game for the `a = 0.5` rows (furniture, kindling)
   but at only **1068 game for wood** (`a = 1`), so a wood crate at the 1263
   plateau still radiates 68 % under floor 0 (58 % under ½); and below `g = 1`
   floor 0 *is* forward Euler to the last count, so its free-cooling error is
   exactly explicit's −5.7 % where `α = ½` gives +0.23 %. Above ≈1800 game the
   two floors are bit-identical. One result favours floor 0 outright: at a
   fire-range source the *un-clamped* equilibrium bias is 0.15 % under floor 0
   against +4.6 % under ½. It is one line in the reference and the engine
   (`D = max(T_abs_q, 4·L_q)`). **P1 builds `α = ½` as ruled unless Erik says
   otherwise.** The orchestrator's recommendation is still the α = 0 floor:
   fires are the case that matters, and the loss of accuracy is on the
   cool-down after they die.
5. **The currency pin value: 0.7 or 0.9 MJ/m³/K?** (P2b §7, §13.1.) The shipped
   derivation uses **0.7**, the number the `furniture` row's own comment names.
   But **0.9** — wood at 12 % moisture content, which is what furniture aboard a
   ship is — lands *every* material row inside 15 % of literature instead of
   20–30 % light, and it is one character in `config.toml`: `rad_scale_derived`
   becomes **1.6533e-8**. A one-line ruling that improves the whole table at
   once. Nothing else in the arc depends on which is chosen.
6. **`cool_shift` versus radiative ignition** (row 43). At the derived scale the
   shipped `cool_shift = 13` holds a 1-tile receiver at 67 game and radiative
   ignition never happens. Either `cool_shift ≥ 16` on the furniture row, or a
   smaller effective heat capacity, or the surface model of item 7. **P3 cannot
   pass its HUMAN-TEST without one of them.**
7. **Is `thermal_mass` bulk or surface-layer heat capacity?** (Row 43.) ~32×
   apart, opposite values wanted, one key today, and the surface reading needs
   `thermal_mass = 0.25` which the power-of-two table cannot express (0 is taken
   — it means the gas branch). The two-node surface/bulk thermal solid is the
   design-scale answer; whichever reading is adopted meanwhile must be adopted
   **consistently**, because the same key sets ignition timing *and* cool-down
   inertia.
8. **`k_leak` should be reopened** (amends item 2). P2b measures it as **the
   largest single lever on reach** — far larger than the calibration: the
   1-tile furniture reach goes 5.75 → 3.92 tiles and the 11 kW/m² crossing
   3.15 → 2.37 — and at its derived 0.10 it lands the engine's own ignition
   criterion almost exactly on survey §9.3's independently derived view-factor
   number. Erik set the reach question aside before P1; reach is precisely what
   §9 asked P2b to calibrate, and this moves it more than anything P2b changed.
9. **`rad_scale_derived` is a global literal, but `A_rad` is per level** (P2b
   §13.7). It assumes `tile_size_m = 0.333` and a 2.5 m deck; `levels/airlock_demo`
   is at 1.0 m and `levels/bench_two_room` at 0.5, where the emission is off by
   `tile_size/0.333` (3× on the former). The precedent is already in the tree and
   is the opposite of a literal: the water solver takes its `dx` from the level,
   never assumed (`config.toml:600-604`). If this matters, `A_rad` should derive
   from the level's `tile_size_m` and `ceiling_h` at bind time. Recorded, not
   acted on.
10. **~~`heat_atten` does two jobs at once~~ — WITHDRAWN, 2026-09-16 (Erik).**
   The orchestrator relayed P2b §13.5–6 as a scheme question: that `heat_atten`
   conflates emissivity with opacity and cannot be one number. **That is wrong,
   and the correction is worth recording because the arc nearly acted on it.**
   The sim carries **one thermal band**; `heat_atten` is that band's
   absorptivity, and the sweep uses the *same integer* for absorbing and
   emitting (`abs_mat = (stream·a_i)>>16`, `emitted = (src·a_i)>>16`), so
   **Kirchhoff holds by construction, as an identity of the code, not an
   approximation**. A real material whose absorptivity and emissivity differ
   does so *across wavelengths*; a one-band model chooses one number and accepts
   the compromise, which is a fidelity limit shared with every grey-body engine,
   not an inconsistency. Nor is opacity a third quantity here: there is **no
   reflection channel** — un-absorbed stream is *transmitted* — so incoming =
   absorbed + transmitted is a complete accounting, and every opaque row is at
   `heat_atten = 1.0` anyway. And for a geometrically partly-filled tile a single
   number is *exactly* right: such a tile absorbs the fill fraction of the
   crossing stream and emits the same fraction of black body — **Kirchhoff
   survives geometric dilution**. Erik, 2026-09-16: *"heat is carrying every
   energy that we model, for all materials… this doesn't break Kirchhoff at
   all."* Closed; do not reopen. What *does* survive from P2b §13.5–6 is a **row**
   observation, not a scheme one, and it moves to item 11.
11. **The `furniture` row does not line up with itself** (P2b §13.6, and the
   thermal-solid prep doc §6). It carries `heat_atten = 0.5` beside
   `light_atten = 0.55` (*"partial occlusion: a crate stack leaks some light"*)
   and `permeability = 0.5` (*"smoke/air drift past crates"*) — three columns
   describing a **half-empty** tile — and `thermal_mass = 8`, which is **solid
   wood's**, with a comment reading *"wood-like (real rho·c ~0.7)"*. A tile smoke
   drifts through is not 139 kg of solid wood. Either `heat_atten = 0.5` is a
   geometric fill fraction (and `thermal_mass` is ≈2× too high) or it is an
   emissivity (wrong for wood at ε ≈ 0.9, *and* inconsistent with the other two
   columns). The same ambiguity is P2b's `V_tile` pin, so the two should be
   settled together — **in the two-node session, issue #68**.
12. **Conduction across a material boundary is correct as built** — recorded so
   it is not re-examined. The face rate is the **harmonic mean** of the two
   conductivities (`materials.py::_build_conduction_tables`), which is the
   correct series-resistance combination, and `face_energy_q` is antisymmetric
   through `min(cap_i, cap_j)`: one integer, both signs, exact conservation
   across any boundary. Radiation and conduction are separate passes, each booked
   into the arc #54 ledger, and they do not interfere.
13. **Should `rad_flux` have a damage ceiling at all?** (P3a-1 finding 3,
   `report_p3a1.md` §6.) `rad_flux` saturated at `INT32_MAX`, and widening the
   plane to int64 made it obvious that **this number was never an overflow
   guard**: `rad_flux` is the D3 *sensor*, outside the energy ledger, read by no
   solver, and consumed by exactly one thing — **unit heat damage**
   (`exchange.py:325-340`, `max(heat, rad_flux)` → `phi` → the burn band). So
   `INT32_MAX` is a cap on **how hard a fire can burn a marine**, and it **binds
   in ordinary play**: on the shipped playground under the full conductor it
   engages from tick **12** (with the level only at 2740 game), on **38 of 120**
   ticks, on as many as **249 cells at once**, and the un-capped peak reaches
   **459 %** of it. The orchestrator's own run of the tripwire sees 157 capped
   cell-ticks in 24 ticks. **P3a-1 therefore did NOT lift it** — saturating at
   `INT64_MAX` "because the plane is wide now" would have been a silent feel
   change, and **no golden could ever catch it** (`rad_flux` is deliberately in
   neither `DIGEST_FIELDS` nor `SIM_FIELDS`). It is now the explicit
   `raycaster.h::RAD_FLUX_CEILING`, read by both backends, with the measurement
   beside it and a non-vacuous tripwire
   (`test_the_flux_sensor_ceiling_did_not_move_with_the_width`) so that lifting
   it must be deliberate. **The question for Erik**: nobody appears to have
   chosen 32 768 game as a burn-damage cap — it is the width of an int32 doing
   gameplay design, and it is limiting fire lethality in the game as it ships
   today. Keep it (and re-home it as a named `[combat]` dial with a rationale),
   raise it, or delete it? **Due before P3b**, which is where units start
   absorbing from the sweep's body share and `rad_flux` gets its new writer.
14. **`.noconvert()` is inert on every nullable argument in this codebase**
   (P3a-1 finding 1) — recorded as a standing trap, not a question. A plane that
   is optional must be a `py::object` for the `None` idiom, so there is no
   overload pass to annotate and `.cast<>()` runs `py::array_t`'s default
   forcecast *directly*; the annotation compiles, reads as hardening, and does
   nothing. The only loud form is explicit `py::isinstance` + `py::type_error`
   (the house pattern, `rad_fluence`, P1). **Row 36's wording is therefore
   incomplete**: `.noconvert()` is what makes a *by-value* argument loud, and
   nothing makes a nullable one loud except the explicit check. This applies
   again at P3a-2 and P5.

**Ruled by Erik on 2026-09-15, after v3 was written:**

- **R-P amended** (§7.1): light on the C++/CUDA sweep once per sim tick, the
  renderer sampling the result per frame; R-S retired for light. Erik: *"24 hz is
  enough for the light to be cast … I absolutely agree that light is on the sim
  tick sweep — if you by light mean the resulting map of the rays."* The
  CUDA–GL interop end state is recorded in §7.1.
- **The density law's site** (§6.3): in the extinction coefficient, no second
  discount in the fold. Erik: *"yes I agree with that."*

**Ruled by Erik on 2026-09-15, after critique 3:**

- **A body radiates.** The body share re-emits at the ambient level; `rad_flux`
  is the signed net above ambient (row 25).
- **`dyn_heat_atten_q` enters the digest at P3**, spec v6, with the re-baseline
  (row 28).
- **The clamp's GPU twin lands inside P3** (and P5 for gas), not P4 before P3
  (row 27).

Closed since v2: the tick budget (24 Hz / 41.67 ms; only stale perf docs to
restate), the `cool_shift` reds (rewritten to assert the property, `82b11a5`),
foliage (row 6), the 16-light cap (row 5), where the sweep lives (row 1),
`rad_flux` calibration (row 3).

---

## Systems

**Existing canonical systems this design must use.** `PhysicsEngine` — the sweep
is a numbered step in it, never Python glue · the temperature solver's Pass-1
fold — the only place radiation becomes temperature, and now the clamp's home ·
the gas energy seam (`deposit_railed`) — the gas branch's only writer, at P5 ·
the energy closure identity — six groups; radiation-into-gas is group 1 with
existing counters; there is no seventh · the face-flux energy step — the
one-integer-two-signs idiom copied in §2.3 · the fixed-point kit — `floordiv_q`,
`shr_round0`, `mul128_shr`, extended (int64 twins), never re-derived · the
material table and the gas table — every new coefficient is a column · the Q16
boundary modules — `optics_fixed.py` joins them · the field digest and the A/B
lockstep harness — per §3 · the CUDA harness — P4's gate · the Recorder — untouched
(§3) · `pack_hover_readout` — the readout rows land in P1 · `tools/fire_tuning_lab.py`
— extended at P2 · GOLDEN_AGGREGATE discipline — one re-baseline, at P3 ·
"Starting a fire" — a fire is started by delivering heat · RL-batch habits §A —
the twin's shape · the Frame-lights and LightingPass rows — kept, with the
accessor as their input.

**New systems this design creates**, with draft rules for CLAUDE.md once the
code exists (scoped per critique 1i):

- *The radiation sweep* (`cpp/src/radiation_sweep.*`, `cuda_radiation_sweep.*`).
  THE radiative transport path for heat and light, replacing the ray engine.
  ONE traversal carries every payload; a new payload is a channel on the sweep,
  never a second traversal; the transport step is a per-channel parameter
  (shear for heat, step for light), never a fork. Written in gather form; the
  CPU and CUDA twins are one formulation.
- *The emissive table* (`cpp/src/emissive_table.{h,cpp}`, the `.cpp` on the
  `/fp:strict` list, the header lookups only). THE `E°(T)` map, its inverse and
  its bake; one instance owned by `PhysicsEngine`, read by the sweep, by Pass 1
  and by Pass 1's CUDA twin; `L°(T)` for light lives beside it, baked the same
  algebraic way. Never a second table, never `pow`, never a bake in a header.
- *The exact-integer conservation idiom for transport.* An integer leaving the
  stream is added to a ledger as the same integer; the downwind split carries the
  remainder, never a second shift; the only escapes are booked planes
  (`rad_net`, `rad_flux`, `rad_amb`), and `Σ` of the three is identically zero.
- *The extinction planes* (`heat_atten_q`/`dyn_heat_atten_q`,
  `light_atten_q`/`dyn_light_atten_q`, `optics_fixed.py`). Every optical
  interaction **the sweep's channels** see — material, body, gas — is a Q16
  coefficient on these planes; `a ≤ d ≤ ONE` is an ingress invariant with a test;
  dynamic stamps are MAX, never sums; the material share emits, the body share
  only absorbs. (Weapon beams keep their own gas table and are out of scope.)
- *The emission stability pair* (Fleck factor + maximum-principle clamp). The
  excess emission is damped before the sweep by `f = T_abs / max(T_abs + 2L, 4L)`;
  no radiative sub-step may carry a cell above `max(T_before, E°⁻¹(Φ))`; the
  clamp lives in Pass 1, booked by the applied-ΔT idiom on solids and as a
  counted deposit reduction on gas. Cite Fleck & Cummings 1971 in the header.
- *The light-field accessor* (`src/simulation/light_field.py`). Gameplay, RL and
  the renderer read light through it, never by calling a producer; the eye field
  is the dequantized integer field; the integer field is authoritative; the
  rules-side field enters the digest only when a rule reads it, by spec bump.
- *The directional sky boundary condition.* Ambient light and heat enter on the
  virtual ring and are transported like any other radiation; the flat shader
  floor is a separate, smaller dial and never a substitute.
- *The producer contract* (§4). Any future light producer (cascades) satisfies
  §4.1–4.3 or it is not a producer.
- *The integer reference* (`docs/ray_engine_v2_scheme_study_2026-09-13/sweep_ref_q.py`
  + `sweep_ref_q_gates.py`, guarded by `tests/test_ray_engine_v2_integer_reference.py`;
  **exists since P0, `40f2479`**). THE executable specification of the sweep's
  integer arithmetic. Any change to the sweep, the Fleck factor, `E°⁻¹` or the
  radiative fold is made here first, its gates re-run, and the C++ written
  against it — never the other way round. Not engine code; nothing in `src/` or
  `cpp/` may import it. Goes into the project CLAUDE.md beside the
  `radiation_sweep` row at P1.

---

## Appendix A — where each critique-2 entry is resolved

| entry | where |
|---|---|
| 1 extinction plane | §3, §2.3 invariants |
| 2 clamp home + medium split | §2.8 |
| 3 Φ | §3 `rad_fluence` |
| 4 gate 5 in P1, evaluation point | §2.9 gate 5; §11 P1 |
| 5 §5.3 writer | §6.3 |
| 6 unit-absorption exit | §2.3 body share; §2.9 gate 1 |
| 7 `ret` shift | §2.3 listing (`amb_m` first) |
| 8 integer reference | §11 P0 |
| 9 `reciprocal_q16` / door 3 | §2.8 integer form (`floordiv_q`), §8.1 |
| 10 `/fp:strict` + ratchet + stale comment | §8.1, §11 P1 |
| 11 call site + both `cast_fire_heat` sites | §0 row 16, §11 P3 |
| 12 resident path | §8.2 |
| 13 TU name vs "lands here"; E° owner | §0 row 1; §2.6 |
| 14 `dyn_heat_atten` dtype, A/B, digest, `stamp_units` | §3 |
| 15 six groups | §6.3 |
| 16 density double-count | §6.3 (Erik's ruling, realised) |
| 17 sweep→fold boundary | §8.4 |
| 18 Recorder claim; "digested? yes"; E° already int64 | §3, §8.3 |
| 19 `rad_amb` plane vs scalar; consumer list | §2.7; §11.1 |
| 20 P8 spec bump | §8.3 |
| 21 split P3 | §11 (render deletions at P6; `range_*` at P3 with `rad_flux`; test dispositions §11.1–11.3) |
| 22 `rad_flux` consumer + `max()` ruling | §6.2 |
| 23 render light: language, clock, per-frame number | §7.1, §10 |
| 24 rules-side extinction | §7.1 |
| 25 furniture comment (present); E° int64 | Appendix B; §2.3 |
| 26 disposition lines | §5, §11.3 |
| 27 "§9.3 curve" cross-reference | §9 item 4 (survey §9.3) |
| 28 `E°⁻¹` saturation; 60 000 rows | §2.6 |
| 29 A/B harness in gates; draft rules scoped; hover readout | §2.9 gate 8; Systems; §11 P1 |
| 30 P1 wired? | §11 |

## Appendix B — critique-2 findings not applied

- **8a "the furniture comment is absent"**: it is present, committed at
  `678d9b6` (`config.toml:1551-1553`, "comment only, no value changed"). The
  critique ran against `5155894`, before that commit.
- **The suite figure "1 failed / 351 passed"**: a truncated `-x` run; the
  critique's own later figure was 2 failed / 2451 passed, both in
  `test_cool_shift_axis.py`, since rewritten to assert the property (`82b11a5`,
  2453 passed).

## Appendix C — where each critique-3 entry is resolved

`ray_engine_v2_critique_3_determinism_cuda_2026-09-15.md`: 3 BLOCKING, 12
REQUIRED, 10 NOTE; its integer probe confirmed conservation, the excess-form
fixed point, the virtual ring, the Fleck integer form, `E°⁻¹` idempotence and
the headroom. Its REQUIRED CHANGES list, entry by entry:

| entry | where |
|---|---|
| 1 the int64 inventory (BLOCKING) | §3 inventory table + forcecast; §11 P1 |
| 2 the clamp's GPU twin (BLOCKING) | §11 P3 (solids), P5 (gas); §0 row 27 — Erik: inside P3 |
| 3 the body share's ambient emission (BLOCKING) | §2.3 listing, §6.2, §6.3; §0 row 25 — Erik: the body radiates |
| 4 the P1 shadow planes | §3 table |
| 5 the gas `L_q` / deposit chain | §2.8 integer form (staged `mul128_shr`) |
| 6 the `E°` bake in a strict TU; the ratchet's limits | §2.6, §8.1 |
| 7 `s_m` as integer literals | §8.1 |
| 8 wavefront mechanics, scratch shape, launch count | §8.2, §10, §3 |
| 9 `E°⁻¹` below `E°[0]`; gate 4 per counter | §2.6, §2.9 gate 4 |
| 10 gate 2 restated | §2.9 gate 2 (a)/(b) |
| 11 `dyn_heat_atten_q` in the digest | §3; §0 row 28 — Erik: yes, at P3 |
| 12 the wrap-contract scene at P1; the A/B scenario wrap-free | §3 "goldens unmoved" conditions; §11 P1 |
| 13 all four rad planes resident at P1 | §3, §8.2 |
| 14 the P4 check script before the wipe; the A/B harness's role at P3 | §2.9 gates 7–8; §11 P4 |
| 15 `L°` by algebraic bake or constants | §8.1, §11 P6 |
| 16 the gas clamp's exact `ΔE`; the `clamp_enabled` keyword; `_stamp_units_python`; the dormancy flag deleted | §2.8, §2.9 gate 5, §3, §8.2 |

Its three open questions were all ruled by Erik the same evening (§12). Its
"checked and FALSE" table is absorbed: the two widening claims by §3, the
saturation claim by §2.6, the gas-below-ambient claim by §6.3 under row 25.
