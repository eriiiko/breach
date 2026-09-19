# Thermal model v2 — design (2026-09-19)

> **Supersedes `thermal_model_v1_design_2026-09-19.md`**, which survives as
> history together with its three critique verdicts (L1 physics, L2
> determinism/CUDA, L3 scope/regression/reuse — **7 BLOCKING, 32 REQUIRED, 22
> NOTE**). §9 below records where each blocker is resolved.
>
> **Depends on**: `ray_engine_v2_design_v3_2026-09-15.md` (the sweep);
> `report_p2b.md`, `report_p3a1.md`; `thermal_solid_two_node_design_prep_2026-09-16.md`
> (issue #68, deferred).
>
> **This document supersedes** design v3 §12 items 5–8 and 13, and design v3
> **row 6**'s "foliage stays `heat_atten = 0.0`" — see R12.

---

## 1. The principle this model is built on

Erik, 2026-09-19: *"I'd like the logic to be based off of first principles — and
those principles are inspired by physics, and designed by us to find a good
middle ground between being able to compute everything in real time and have at
least an OK believable physics simulation."*

That is the standard every decision below is held to. Concretely it means:

- **A quantity is derived, not dialled.** If a number cannot be traced to a
  physical constant or a stated geometric argument, it is a defect, not a
  tuning opportunity.
- **A channel is computed and booked, or it does not exist.** The failure this
  arc keeps finding is a hand-rolled stand-in for a real process (`cool_shift`
  for radiation; a fitted `rad_scale` for emission).
- **Simplifications are named, not hidden.** An `ACCEPTED GAP:` is a deliberate
  middle-ground choice with its cost written down.
- **A question the model cannot pose cleanly is deleted, not answered.** See R11.

---

## 2. Rulings

R1–R9 are carried from v1 unchanged. R10–R13 are new (Erik, 2026-09-19).

| # | Ruling | Reason |
|---|---|---|
| R1 | **`cool_shift` is deleted.** | A hand-rolled radiative loss (`temperature_solver.cpp:136` says so). The sweep computes the real one; keeping both counts it twice. |
| R2 | **`k_leak` goes live.** **v2: UNIFORM, not per-tile.** | R1 makes the out-of-plane path load-bearing — with `k_leak = 0` a sealed interior room is radiatively adiabatic. Per-tile authored leak needs level data and editor support (L3-B3); it is deferred to the extension points (§7). Uniform at the derived value delivers R1's substance now. |
| R3 | **Ambient becomes per-tile, DERIVED from state the engine already has** (vacuum ring vs interior), not authored. | Erik wants *"room temp in ship, 0 K outside"* expressible. The engine already knows which cells are vacuum/ambient-ring; deriving from that needs no new level data. |
| R4 | **Space is at room temperature for v1.** | *"Most of the playing will take place inside the space ships."* |
| R5 | **`thermal_mass` stays a power of two**, real `ρc` snapped. | It rides a bit-shift. Cost is 6–10 % at the ruled 0.9 pin (R13); it would have been −26 %/+40 % at 0.7 (L1-R2c). |
| R6 | **No two-node solid.** Walls honest, furniture tailored. | *"I don't want to increase the complexity in the model before we have a working simulation."* Deferred: #68. |
| R7 | **Real numbers throughout; nothing tuned to produce flashover.** | Retune after the engine works. |
| R8 | **Dirichlet tiles: SOLIDS ONLY**, not built in v1. | A fixed-temperature gas tile is an infinite source/sink inside the EOS. |
| R9 | **The `rad_flux` ceiling stays.** | It stops binding on its own after the flip (0.2 % of the ceiling). |
| **R10** | **Conduction runs at REAL physical rates.** The absolute rate stops being a CFL stability anchor and becomes `α·Δt/Δx²` per material. | Erik, 2026-09-19: *"Let's stick with trying to stay close to physics… wood may be fully insulating, it's OK — we WILL have lots of other means for temp to travel. 36 h across a tile of wood, I don't mind if you set it to 0."* Resolves L1-B2. **Consequence, accepted**: conduction becomes a slow background process (steel 4.4e-6 per tick, wood 8.1e-8, glass 1.9e-7), and fire spread is carried by radiation and hot gas — which is what actually carries it in a real compartment fire. **Very likely closes #61**: the ~3 s e-fold draining wood's ignition heat *is* the ~43 000× overspeed. |
| **R11** | **The load-time fire-seed sustain check (`materials.py:796-805`) is deleted**, and a sweep for tests of its kind follows as its own patch. | Erik: *"doesn't that also ask an ill-posed question? … it's kind of obvious that if the material is hot enough it will sustain, but otherwise it won't — this type of test is exactly the ones I think we ought to look for and remove."* Resolves L3-B2 by deletion rather than by re-derivation. It is a **load-time** check whose gain is literally `bed_per_I · 2^(cool_shift − heat_inv_shift)`, so R1 guts it anyway. |
| **R12** | **`foliage` gets its real emissivity** (`heat_atten ≈ 0.9`, cellulosic), superseding design v3 row 6. | L1-B1: with `heat_atten = 0.0` **and** `conductivity = 0.0`, and `cool_shift` gone, a flammable foliage tile is **strictly adiabatic** — combustion writes into it and nothing can remove the energy, so it ratchets to `T_MAX_PHYS` and re-ignites neighbours forever, silently. Under R10 a small conductivity would **not** save it (real conduction is negligible), so the fix must be radiative. Row 6's `0.0` was an acceptance of a gap, not a physical claim, and R7 supersedes it: 0.9 is the real number. **This does not author tree behaviour** — it stops foliage being radiatively invisible. |
| **R13** | **The currency pin is 0.9 MJ/(m³·K)** — wood at ~12 % moisture content. `J_per_count = 0.4759 J`; `rad_scale_derived` becomes **1.6533e-08**. | **Erik, 2026-09-19: *"0.9 it is."*** The pin is a UNIT DEFINITION, not a measurement: it fixes what one heat count is worth, and therefore what every other row's `thermal_mass` must be (its real `ρc` ÷ the unit, snapped to a power of two). The criterion is which unit makes the whole table land kindly. At **0.9**: steel 33.8→32 (−5.3 %), glass 17.8→16 (−10.0 %), wood **8.0→8, exact** — and all three reproduce the SHIPPED values, so the table has been encoding this pin all along. At 0.7 every row is 22–26 % light AND glass would have to double to 32. 0.7 is roughly oven-dry wood; timber indoors equilibrates near 12 % MC, and ship furniture is not oven-dry. Nothing live reads `rad_scale_derived` yet, so the change is free. |

### ACCEPTED GAPs

Decisions, not oversights. Do not build machinery to close them; do not report
them as findings.

- `ACCEPTED GAP:` A breach will not make a room cold (R4).
- `ACCEPTED GAP:` Floors and ceilings have no temperature of their own — they are
  a radiative boundary (R2), not state. A tile is gas *or* solid.
- `ACCEPTED GAP:` Floors cannot burn.
- `ACCEPTED GAP:` Thick structure barely auto-ignites (R6); flamethrowers and
  direct deposits still light it.
- `ACCEPTED GAP:` `thermal_mass` quantisation, 6–10 % at the 0.9 pin (R5).
- `ACCEPTED GAP:` `c_p` constant per material (~30 % drift for wood over our
  range).
- **CLOSED, not a gap:** `c_v` vs `c_p` for gas. Erik, 2026-09-19: heating air makes
  it expand, which takes the temperature back down. That is right, and the engine
  already charges it — `eos_solver.cpp`'s `k_work = (γ−1)·T_AMB_K` removes book energy
  in proportion to `div u`. So `c_v` is the correct constant for the FIELD
  (constant-volume internal energy, `N·T_abs`) and the expansion is charged SEPARATELY
  as work; using `c_p` in the capacity would double-count the same physics. Locally the
  spike evens out, globally the room keeps the energy and warms at `c_v`.
- `ACCEPTED GAP:` **Conduction is negligible at tile scale** (R10). Heat moves by
  radiation and gas advection. A material wanting fast conduction (a composite,
  a heat pipe) is authored as its own row later, with its `κ` chosen rather than
  measured — and *labelled as such*.
- `ACCEPTED GAP:` **`k_leak` is uniform** (R2). Every tile leaks at the same
  out-of-plane rate regardless of what is above and below it.

---

## 3. What falls out

### 3.1 The loss channels after `cool_shift`

A solid loses heat by exactly three computed routes — **and L1 showed the count
is a per-material claim, not a global one**:

1. **In-plane radiation** (the sweep) — requires `heat_atten > 0`.
2. **Out-of-plane radiation** (`k_leak` → `rad_amb`) — acts on the *stream*, so
   it also requires `heat_atten > 0` to reach a given cell's stored energy.
3. **Conduction into adjacent gas**, then advection — requires
   `conductivity > 0`, and under R10 is *negligible in magnitude* even when
   present. **T2 sharpened this**: under R10 combined with the `c_v` correction
   this channel may land at **identically zero** in integers, i.e. structurally
   absent rather than merely small — see §4 item 7 and T2's §8. If so, radiation
   (routes 1 and 2) is the ONLY loss path a solid has, and gas is heated only by
   combustion until P5 gives it radiative absorption.

> **Therefore: for a flammable thermal solid, radiation is the only meaningful
> loss channel, and `heat_atten > 0` is an INVARIANT, not a preference.**
> `furniture` and `kindling` already satisfy it (0.5); `foliage` does not, which
> is R12. The invariant is enforced at the material-table door (§6 item 1)
> alongside the existing `heat_atten > 0 ⇒ thermal_mass > 0`.

`k_leak` is **not a time constant**: it is the fraction of each ordinate's stream
leaving the deck plane per cell crossing, with the ceiling returning ambient
radiation. A cell at ambient is in exact balance. Its value ~0.10 comes from
fitting the exact slab law `S(r) = a/√(a² + r²)` (max 8.2 % over r = 1–12); v1's
"~6 % of radiating area" was a *different and worse* derivation presented as
corroboration (L1) and is struck.

### 3.2 The currency

`cell_capacity_q` already produces solid and gas capacities in **one unit** —
which is why conduction can take `min(cap_i, cap_j)` across a solid–gas face.
Solid capacity is `thermal_mass`; gas is `N · c_v`. But **`c_v = 1.0` is a
placeholder** by its own config comment, so an ambient air tile behaves as
`thermal_mass = 1.0` where real air is **0.0076849**: air is **130.13× too heavy
thermally**. **CORRECTED by T2**: the 0.0098 / 101.56× pair in this row was computed
at the superseded **0.7** pin; at R13's ruled **0.9** it is 0.0076849 / 130.13×. T2
also derived `ρc_v` better than this document did — not from tabulated air, but from
the engine's OWN EOS constants, `ρc_v = p/((γ−1)T) = 864.548 J/(m³·K)`, the residue of
`eos_solver.cpp`'s `k_work` derivation (it agrees with the tabulated value to 0.34 %).
`V_tile` cancels, so unlike `rad_scale` this number is free of P2b's two ambiguous
factors and of tile size, and it is independent of molar mass. Where `c_v` belongs — the
`gas_energy = N·T_abs` representation or the conversion seam — is T2's
measurement.

### 3.3 The ambient quantity is `amb_m`, not `t_amb_q`

**v1 named the wrong scalar** (L2-B1). In `radiation_sweep.cpp`, `t_amb_q` is the
**absolute-zero offset** of the temperature scale (`T_abs = temperature[i] +
t_amb`, feeding the Fleck denominator), bound from the same `293 << 16` the
gas-energy seam runs on. The ambient the sweep **radiates** at is
`amb_m = (e_table[0] · w_m) >> 16`, from `E°[0]`, and it never reads `t_amb_q`.

So R3's per-tile ambient is a per-tile **`amb_m`**, and `t_amb_q` stays the one
global scale offset it has always been. This also removes v1's draft-rule
contradiction with the gas-energy seam's born-at-ambient rule (L3).

---

## 4. The calibration ledger

| # | Phenomenon | Settled by |
|---|---|---|
| 1 | Radiation transport | **Done** — gate 0 |
| 2 | Radiative emission scale | **Done and now determined** — P2b's derivation at R13's pin: **`rad_scale_derived = 1.6533e-08`** (was 2.1256e-08 at the provisional 0.7). Applied to the sweep's table at T3/T5 |
| 3 | Solid heat capacity | **Derivation, pin RULED (R13 = 0.9)**. T3 derives every row against it: one `thermal_mass` unit = 0.1125 MJ/(m³·K) |
| 4 | Gas heat capacity (`c_v`) | **Test** — §3.2, T2 |
| 5 | Energy → temperature | Derivation — verified; it is the definition of heat capacity |
| 6 | Conduction solid↔solid | **RULED (R10)** — real `α·Δt/Δx²`; derivation only, no stability anchor |
| 7 | Conduction solid↔gas | **Same law, same ruling — plus TWO corrections found 2026-09-19 with Erik.** **(a) The law is wrong at a wall.** Our face uses pure conduction, `κ/dx = 0.072 W/(m²·K)`; the real process is CONVECTION through a sub-tile boundary layer, `h ≈ 2–10` natural and `10–100` in a fire plume — we are **28–139× too weak**. Using `h` is not a fudge: pure conduction is the WRONG LAW for a solid–gas interface and `h` is a measured quantity, so this sits inside R7. **T3 derives it, T5 applies it.** **(b) Vacuum must not conduct.** Kinetic theory: `κ` is density-INDEPENDENT (`n` and `λ` cancel) until `λ` reaches the gap, which for a 0.333 m tile is **0.02 Pa = 2×10⁻⁷ atm**. But `n_floor_heat = 0.01` is 1013 Pa — **five orders above that** — so the floor invents a conducting medium in vacuum and a breached cell keeps conducting at a floored capacity. The fix is a **MASK** (a vacuum cell does not conduct), never a density law for `κ`. **T3 states the threshold, T5 applies the mask.** |
| 8 | Out-of-plane loss (`k_leak`) | Derivation from the exact slab law (~0.10) |
| 9 | Emissivity | Derivation from literature; includes R12's foliage |
| 10 | Ignition temperature | Literature |
| 11 | Unit heat damage | Test — a marine's time-to-death against a stated survivability target |
| 12 | Combustion | Derivation (heat of combustion) + test (burn duration) |

---

## 5. Patch plan

**The flip is atomic.** v1's cut split it and scheduled every test deletion
*after* the patch that invalidated it (L3-B1). The fix is design v3's own logic,
which v1 had broken: the old-law test surface is retired **first**, safely,
because its replacement gates were written at P1 against the shadow sweep and are
already green.

| # | Patch | Mode | Tier | Gate | Human |
|---|---|---|---|---|---|
| **T1** | **Per-tile `amb_m` (derived from vacuum/interior state) + uniform `k_leak` live in the sweep.** Reference first (`sweep_ref_q.py`), then the engine. Still shadow | subagent, worktree | **Sonnet 5** | **gate 0** bit-for-bit; the uniform-ambient case reproduces today's output exactly; goldens unmoved | — |
| **T2** | **The currency audit** (ledger 4). What `c_v` must be, and where it belongs given `gas_energy = N·T_abs`. Report + proposed value; **applied at T5** | subagent, worktree | **Opus 5** | #54 identity still closes; goldens unmoved | — |
| **T3** | **Derive the real material table** (ledger 3, 6, 9, 10, 12): `thermal_mass` snapped, `κ` at **real rates (R10)**, emissivities **including foliage (R12)**, ignition temps, heat of combustion, **the solid–gas convection coefficient `h` and the vacuum-mask threshold** (ledger 7a, 7b). **A report, not an edit** | subagent, worktree | **Opus 5** | every number carries a citation; the 0.7-vs-0.9 pin recommended with its table | **review** |
| **T4** | **Retire the dead test surface, before it can go red.** The 46 old-law tests (design v3 §11.1), `test_cool_shift_axis.py` (25), `test_temperature_cooling.py` (9), the CUDA check pair, the gate-A capture, `test_thermal_mass_axis.py`'s exact-value pin (rewritten to a property), and the R11 sustain check. Each deletion names the property that replaces it | subagent, worktree | **Haiku 4.5** | **suite green** — the old law is still live, so this only removes tests whose replacements already pass | — |
| **T5** | **THE FLIP — atomic.** Fold reads the sweep; clamp live on CPU **and** the CUDA temperature twin; units absorb (body share); `cool_shift` deleted across all 43 files **including the CUDA Pass 3 and the two pinned positional enum slots** (L2 — removing entries renumbers survivors silently); `k_leak` live; the real material table applied; `c_v` corrected; **`dyn_heat_atten_q` into the digest, spec v6, and the arc's ONE golden re-baseline, in this commit** (L2-B2) | subagent, worktree | **Opus 5** | full suite; #54 identity closes with the thermostat *term* removed; CUDA tol-0 | **HUMAN-TEST** — built, gated, pushed, **NOT merged** |
| **T6** | **Delete the old-law code**: `cast_fire_heat` + both call sites, the dead dials, `RAD_LIM_SHIFT`, the pair budget, the CUDA cast bindings, the five tool edits | subagent, worktree | **Haiku 4.5** | suite-gated, mechanical | — |
| **T7** | **CLAUDE.md rules walkthrough** — every canonical row re-read against code after this arc removed systems | inline | **Opus 5** | every row points at code that exists | **review** |
| **T8** | **The ill-posed-test sweep** (R11's generalisation): find and remove tests that pose questions the model cannot answer cleanly. **Its own patch, after the rest** | subagent, worktree | **Opus 5** | each removal names why the question was ill-posed | **review** |

---

## 6. Verification plan — properties to pin

1. **A flammable thermal solid has a loss channel.** The material-table door
   refuses `flammable && thermal_solid && heat_atten == 0`. *Breaks if* a row is
   authored that could ratchet to `T_MAX_PHYS`. (T1/T5)
2. **The uniform case is unchanged.** With ambient uniform, the sweep reproduces
   the scalar-era output integer for integer. *Breaks if* the uniform path moves
   **at all**. (T1 — **CORRECTED**: this row first read *"breaks if `amb_m` is
   mis-indexed"*, which T1 measured to be FALSE. A uniform-ambient test is
   **structurally blind** to an in-plane mis-index — every cell holds the same
   level, so reading the wrong cell changes nothing. T1 proved it by injecting
   four bugs including v1's exact hoist: **item 2 passes all four.** Mis-indexing
   is caught by gate 0's non-uniform axis and item 3's localisation leg.)
3. **Per-tile ambient actually varies.** A vacuum-ring cell and an interior cell
   radiate against different ambients. *Breaks if* `amb_m` is hoisted out of the
   cell loop — **which is exactly the bug v1 would have shipped**. (T1)
   **This is the ONLY test standing between R3 and a silent no-op.** Item 2 is
   blind to a hoist by construction, and **so is item 4** — a conservation
   identity is *structural*, so it holds just as exactly for a sweep doing the
   wrong thing. T1 built item 3 to PROVE hoist-impossibility rather than assert
   it: two mirror-image compartments with `is_vacuum` as the ONE asymmetry, so
   collapsing the ambient to any scalar forces a symmetric output while the real
   per-cell run is not (the port wall loses 42× its starboard twin). Validated
   against four injected bugs; item 3 catches all four. **Any future patch on
   this axis keeps item 3 or replaces it with something equally adversarial.**
4. **Conservation survives.** `Σ rad_net + Σ rad_flux + Σ rad_amb ≡ 0` in int64
   with non-uniform ambient and `k_leak > 0`. (T1)
5. **One gas joule is one solid joule.** A known energy across a solid–gas face
   raises the gas by its real heat capacity's prediction. (T2/T5)
5b. **A face loses exactly what the gas gains.** Erik's own assertion, 2026-09-19,
   and it is **not currently tested**: the solid's loss and the gas's gain must be the
   SAME integer across a solid–gas face. It holds today only because `c_v = 1` makes
   it hold by accident — T2 measured received/delivered `= c_v`. *Breaks if* the
   energy→temperature conversion on the gas side and the face flux disagree about the
   capacity. (T5)
5c. **A gas cell never ends hotter than the solid heating it.** The maximum principle,
   also Erik's. Structurally guaranteed today by `min(cap_i, cap_j)` — `full` is
   exactly the energy that brings the smaller-capacity side to the other's temperature
   — and the `c_v` fix must preserve it. (T5)
6. **A sealed room is no longer adiabatic.** With `k_leak > 0` a hot interior
   room's total energy falls; at 0 it does not. (T5)
7. **Nothing relaxes to ambient.** No solid's temperature changes except through
   a booked channel. *Breaks if* a `cool_shift` remnant survives any of 43 files.
   (T5)
8. **Conduction is physical.** A two-tile slab's e-fold matches `α·Δt/Δx²` for
   its material, to the shift quantisation. *Breaks if* a stability anchor
   returns. (T5)
9. **The books close** with the thermostat term gone and `rad_amb` grown. (T5)
10. **A marine burns, a zombie takes 4×.** (T5)
11. **Contents burn, structure resists.** A furniture tile beside a plateau fire
    ignites within the bench window; a wood wall does not. R6's accepted
    consequence, asserted so it is deliberate. (T5)

---

## 7. Systems

### 7.1 Existing canonical systems this design must use

Integer reference (`sweep_ref_q.py` — every arithmetic change lands **there
first**) · Radiation sweep · **Extinction planes** (`k_leak_q` already routes
through `optics_fixed`, `physics_runner.py:465`, so it **extends** this system —
L3) · Temperature solver · Gas energy seam + the closure identity · Material /
gas tables (**id budget FULL**, arc #60 — v2 adds no material) · Config ·
Q16 boundary modules · Field digest / GOLDEN_AGGREGATE / A/B lockstep / CUDA
harness · `tools/fire_tuning_lab.py`.

**Note on gating**: the A/B lockstep harness proves a refactor moved nothing; it
**cannot gate a deliberate behaviour change** (L3), so T5 is gated on the suite,
the closure identity and the human test — not on A/B.

### 7.2 New systems, with draft rules

- **Per-tile ambient emissive level (`amb_m` per cell)** — *draft rule*: THE
  per-tile ambient the sweep radiates against, **derived** from vacuum/interior
  state, never authored and never a global. `t_amb_q` remains the single global
  temperature-scale offset and is not per-tile.
- **The heat-count currency** — *draft rule*: one heat count is a fixed number of
  joules, pinned once on the standard furniture row; **every** thermal capacity,
  solid and gas, is denominated in it. A new medium states `ρc` in SI and
  converts here.
- **The loss-channel invariant** — *draft rule*: a flammable thermal solid must
  have `heat_atten > 0`. Radiation is the only meaningful loss channel under
  R10; without it the tile is an energy ratchet.

### 7.3 Systems retired

- **`cool_shift` / `cool_shift_vacuum`** — the dials, the per-row column, the
  `COOL_SHIFT`/`COOL_SHIFT_VACUUM` globals, Pass 3 **and its CUDA twin**, the two
  **pinned positional** counter slots (`C_COOL`, `C_THERMOSTAT` — folded by index
  at `physics_engine.cpp:377-388`), `e_cool_sum` and its four test consumers plus
  `storm_ledger.py`. **43 non-doc files** (L3, verified).
- **The ambient thermostat's ledger term** — `e_thermostat_sum`. The identity
  must close with the term **removed**, not zeroed.
- **The load-time fire-seed sustain check** (R11).
- **The CFL stability anchor** for conduction (R10) — `SHIFT_AT_REF`/`KAPPA_REF`
  stop setting the absolute rate.

### 7.4 Named extension points — designed, not built

- **Authored per-tile `k_leak`** (floor grate / deck above / open sky). Needs
  `level_lib` + `level_loader` + editor support.
- **Dirichlet fixed-temperature tiles** (R8) — AC, radiators, heated
  compartments. **Solids only.**
- **Two-node skin/core solids** — issue #68.

---

## 8. Where each blocker is resolved

| blocker | resolution |
|---|---|
| L1-B1 — `foliage` strictly adiabatic | **R12** (real emissivity) + the §6.1 invariant. A conductivity fix would **not** work under R10 |
| L1-B2 — conduction not derivable | **R10** — Erik ruled real rates |
| L2-B1 — `t_amb_q` is the wrong scalar | **§3.3** — the per-tile quantity is `amb_m`; `t_amb_q` stays global |
| L2-B2 — spec bump moves goldens in its own commit | **T5** carries the bump *and* the re-baseline together |
| L3-B1 — test deletions scheduled after invalidation | **T4** retires the surface first, while the old law is still live and green |
| L3-B2 — orphaned load-time sustain check | **R11** — deleted, not re-derived |
| L3-B3 — no source for the per-tile planes | **R2/R3 scope-down** — uniform `k_leak`, ambient derived from existing state; authored leak is §7.4 |
