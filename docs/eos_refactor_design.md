# EOS refactor — design document **v2** (rung B, compressible ideal gas)

> **Status:** v2 DRAFT for round-2 critique, then Erik. Supersedes v1 in place; the round-1
> critique lives at `docs/eos_refactor_design_critique.md`. Companions:
> `eos_refactor_decisions.md` (LOCKED decisions incl. the three v2 calls below),
> `eos_refactor_interaction_map.md`, `eos_research_report.md` §4. The float prototype
> (`prototypes/eos/scheme_rung_b.py`, branch `eos-prototype`) is demoted to a **shape
> reference only** — v2 deliberately does *not* carry its numerics forward (see §3).
>
> **Nothing is built until this doc survives critique** (decision 9).

## v1 → v2 changelog (what changed, and which finding each change kills)

| Change | Kills |
|---|---|
| **Solver reformulated to true Kwatra pressure-evolution** — the Helmholtz RHS carries the advected absolute pressure `p* = C·N·T`, not a divergence-only correction. Venting, thermal expansion, and water push become *native*; the `div_target` heuristic and its tuned gains (`K_EXPAND`, `W_DISPLACE_GAIN=60`) are **deleted**. | **B1** (native venting), D6 partly, retires two hacks |
| **One velocity.** `wind_x/wind_y` become views of the solver's `(vx,vy)`; smoke and fire-fan advect on the *same* velocity that transports O2. No more cosmetic-vs-physical split. | **B1-sub** (smoke vents while O2 stays) |
| **Bulk O2/N2 transport = donor-cell conservative flux** (the shipped water-solver pattern, Q16.16 + outflow limiter, CUDA precedent `cuda_water.cu`). Global mass-renormalization is **deleted** from the design. Flux form *is* the continuity equation — dilation included. | **B2** (missing `−N∇·u`), **D1** (sealed-room leak, unfittable divide, cadence ambiguity), renorm div-by-zero minor |
| **Fixed-point spec for the pressure solve**: all Helmholtz coefficient/diagonal/neighbor-sum arithmetic in **wide int64 (Q16.16-scaled)** intermediates, narrowed only at the final per-cell quotient; a **solver-local `N_floor`** independent of gameplay; an **overflow/saturation stress sweep** in the gate. | **B3**, **B4** |
| **Patch graph re-derived** — every patch leaves the game runnable + testable; `wave_p` retirement and its consumer migration happen **in the same patch**; `atmosphere` survives as a zero-copy alias of `P`; unified temperature lands *before* the solver that needs gas-T. | **B5**, **B6** |
| **Cost honesty**: p99/max (not mean) acceptance gates; explicit sweep-count re-derivation at the patch-3 gate; per-species cost model; combined-system bake-off gate as its own HUMAN-TEST patch. | **B7**, **D8** |
| **Combustion products are conservative**: the non-soot fraction of consumed O2 is credited to `inert_N2` ("burnt products"). Sealed-room baseline pressure no longer fake-drains. | **D2** |
| Compression-work term **substepped** with advection + the `T` floor **named as an accounted sink** (debug telemetry counter). | **D3** |
| Unit shockwave-shielding placed concretely: **per-cell velocity damping in the correction step** (`u *= 1 − absorb·dt`, from `dyn_wave_absorb`). | **D4** |
| Substep count via the proven **integer-ceil discipline** (`smoke_cliff_count` class); `max|u|` via `sqrt_q16`/`sqrt_q16_dev`. | **D5** |
| Ripple splash reads the **per-tick pressure transient `|P − P_prev|`** (P_prev is a kept copy of last tick's materialized P). | **D6** |
| **GPU backends for touched systems pinned to CPU** for the migration window (patches 3–5). | **D7** |
| Patch-6 CUDA surface **fully enumerated** (incl. the new velocity-advection and bulk-flux kernels; `cuda_wave.cu`/`cuda_atmosphere.cu` retirement). | **D9** |
| Per-sub-kernel bit-identity checkpoints in the solver patch (not one end-of-tick digest). | minor |
| Floors: combustion's `N_total` divisor floored independently of `o2_thresh`; burst-walls furniture-`N` note; no-shock-capturing fidelity limit named. | minors |

---

## 1. Goal & scope

**Adopt rung B** — a genuine compressible ideal gas — replacing the two-field
`atmosphere + wave_p` IMEX scheme. Two primary fields carry the physics:

- **`T`** — gas temperature, Q16.16 Kelvin, **one unified field** across gas + solid
  (decision 7 / A1).
- **`N_i`** — per-species particle density, Q16.16, the existing `(N_species, h, w)`
  array grown by two **bulk** species (O2, inert_N2).

Pressure is derived: **`P = C · N_total · T`**, materialized **once per tick** into a
stored Q16.16 field before any consumer reads (decision 5). Downstream consumers read
`P`; upstream writers feed `N` or `T` — "feeding pressure" ceases to exist as a concept.

**Fixed-point discipline:** every touched field is already Q16.16 (decision 8). This
refactor *rearranges* fixed-point state; the only float bridges it meets, it **removes**
(water head, the `atm_f_`/`wave_p_f_` mirrors). A patch reaching for a new float
intermediate is a red flag.

**Out of scope** (unchanged from v1): 2.5D z-layers; molar-mass buoyancy (returns with
z); realistic combustion kinetics / species diffusion (no lit search — structure now,
constants by eye); through-wall wave transmission; the fire-intensity logistic's shape
(only its O2 *input* changes).

**Fidelity limit, named:** fixed-sweep GS on the Helmholtz operator smooths
discontinuities — blasts are soft compression waves, not Rankine–Hugoniot shocks, by
construction. That is the intended game aesthetic, not an accident.

---

## 2. Field architecture

### 2.1 Species

| Species | Class | Transport | Conserved? |
|---|---|---|---|
| **O2** | bulk | **donor-cell flux** (§2.2) | exactly (transport); consumed only by combustion |
| **inert_N2** | bulk | **donor-cell flux** | exactly; receives combustion's "burnt products" (§5) |
| `white_smoke, black_smoke, poison, teargas, fuel_gas` | trace | semi-Lagrangian (as today) | no (decay-permitted, unchanged) |

`N_total = Σ N_i` (Dalton). Traces contribute their (small) real mass; their
non-conservation is an already-blessed simplification — the bulk pair carries ~99 % of
`N_total`, so `P` integrity rests on the conserved species.

**Initialization / calibration (patch 1):** split today's ambient (`atmosphere = 1.0`)
as `N_O2 = 0.21·N_amb`, `N_N2 = 0.79·N_amb`, and choose `C` such that
`quantize(C · N_amb · T_amb) == quantize(1.0)` — ambient `P` preserves today's scale
**exactly at the Q16.16 level** (assert within 1 count), so every downstream consumer
tuned against "1.0 = 1 atm" keeps its calibration at rest.

### 2.2 Bulk transport: donor-cell conservative flux (v2 call #1 — LOCKED)

The two bulk species move by **first-order upwind donor-cell flux** on the solver
velocity `u`:

```
for each face (i → j), face permeability-gated as today:
    flux = u_face · N_donor · dt / dx        # donor = the upwind cell
    N_i -= flux ; N_j += flux                # exact integer transfer, one writer per face pass
```

with the **per-cell outflow limiter** pattern lifted verbatim from
`water_solver.cpp` (scale down a cell's total outflow so it never exports more than it
holds — the same conservation-critical block that keeps `water_depth ≥ 0` honest).

Why this over global renormalization (v1 §2.2): flux form **is** the continuity
equation — `∂N/∂t + ∇·(Nu) = 0` — so *convergence raises local density* (the B2 dilation
physics) automatically; conservation is **local and exact in integer arithmetic** (every
subtraction has a matching addition), so **a sealed room is airtight by construction**
— no cross-room leakage artifact, no grid-wide divide, no cadence ambiguity, no
`Σ→0` edge case. In-house precedent end-to-end: the pattern, its Q16.16 arithmetic, its
limiter, and its **CUDA port** (`cuda_water.cu`) all ship today.

Traces stay semi-Lagrangian (visual tracers; their decay is a feature).

### 2.3 Deleted / merged / aliased

- **`wave_p`, `wave_v`, `wave_source` retired** — subsumed by the solver state
  (`u`, `P`). Their consumers migrate **in the same patch** (§8, patch 3): unit push →
  `grad(P)`; ripple splash → `|P − P_prev|`; recorder field list updated;
  `test_wave_absorption` reworked against the new absorption placement (§6).
  FieldEdit's `wave_source` policy remaps to an **energy deposit** (a `T` spike via
  §4.3's reciprocal) — explosions inject energy, not phantom acoustic staging.
- **`atmosphere` becomes a zero-copy alias of `P`.** With §2.1's calibration, `P` lives
  on the same "1.0 = 1 atm" Q16.16 scale, so `gmap.atmosphere` *is* the stored `P`
  buffer under its old name. Every legacy reader (renderer, field-edit tooling,
  recorder, `atmosphere_fixed` helpers, temperature's vacuum-cool compare, fire's O2
  gate until patch 4) keeps working unmodified through the whole migration. Formal
  rename/deprecation happens in cleanup (patch 7), not before.
- **`wind_x`/`wind_y` become views of the solver's `(vx, vy)`.** One velocity: what
  pushes smoke is what carries O2. (Consumers: smoke advection, fire wind-fan/strip —
  read the same arrays as before, now solver-owned.)
- **`sink_hop` + its BFS machinery deleted** (decision 3) — venting is native (§3).
  Breach→vacuum **generalized** beyond edge-hull (any destroyed tile exposing vacuum).
- **Dead `wave_solver.{cpp,h}` deleted.**

---

## 3. The compressible solver — true Kwatra pressure evolution

### 3.1 The v1→v2 correction (why this is the load-bearing change)

The prototype (and v1 §3.2) solved a **pressure-correction/projection** system: RHS
`= dt·c²·(div u* − div_target)` — it responds only to *existing motion* plus a bolted-on
source heuristic. Round-1 critique B1 proved the consequence: a quiescent breached room
produces exactly zero flow ("native venting" was not delivered).

True Kwatra solves the **pressure evolution equation** implicitly. Discretized (their
eq. 15-17 shape, adapted to our fields):

```
p*    = C · N_total · T                     # advected-state ABSOLUTE pressure (post step-1)
solve (I − dt²·∇·( c²/ρ̂ )∇) P_new  =  p* − dt·ρ̂c²·div(u*)      # fixed-sweep RB-GS
u    -= dt · grad(P_new) / ρ̂                # momentum kick from the ABSOLUTE field
```

where `ρ̂` is the mass density from `N` under a **solver-local floor** (§3.4) and `c` is
the capped sound-speed dial. The identity term keeps the operator **strictly diagonally
dominant** (round-1 "sound" column) — fixed-sweep convergence stays guaranteed.

What the absolute-`p*` RHS buys, with no source heuristics at all:

- **Breach**: vacuum cell holds `P = 0` (Dirichlet); the room's `p*` is ~1 atm; the
  solve produces a steep `P_new` gradient at the hole; step-3 kicks `u` outward; flux
  transport (§2.2) carries N out; as `N → 0`, `p* → 0` and the flow **stops by itself**.
  Venting is the equation, not a mechanism. *(Gate: the quiescent-cold-breach E2E, §8 P3.)*
- **Explosion/fire**: `T` spike ⇒ `p*` spike ⇒ outward kick ⇒ expansion ⇒ §4's
  compression-work cools the parcel — the fireball arc, natively.
- **Water rise (W3)**: water shrinks the free column ⇒ `N` (per free volume) rises ⇒
  `p*` rises ⇒ air pushed — no ×60 gain.

`div_target`, `K_EXPAND`, `W_DISPLACE_GAIN` are **gone**. The buffet-vs-dome distinction
consumers rely on is P's own time evolution: the acoustic transient rides `P_new`'s fast
relaxation; the standing dome is its slow component. *(§6 verifies the `apply_wave_push`
fast path: `∇P ≡ 0` over any uniform region regardless of baseline — integer-exact.)*

### 3.2 Tick order

```
0. P_prev := P (kept copy — ripple transient + debug)     [P was materialized last tick]
1. ADVECTION SUBSTEPS — n = ceil_int(dt / dt_adv), integer-ceil discipline
   (smoke_cliff_count class); dt_adv = CFL_ADV·dx / (max|u| + eps), max|u| via
   sqrt_q16 over vx²+vy² (int64 sums); n capped at N_SUB_MAX (cap value: patch-3 gate).
   per substep (dt_s = dt/n):
     a. u  ← semi-Lagrangian self-advection            (§3.3 simplification, flagged)
     b. T  ← semi-Lagrangian advection (gas mask)
     c. traces ← per-slice SL (unchanged, decay-permitted)
     d. bulk N_O2, N_N2 ← donor-cell flux on u          (§2.2 — conservative)
     e. T -= (γ−1)·T·div(u)·dt_s   (compression work, SUBSTEPPED — D3;
        per-substep |(γ−1)·div(u)·dt_s| bounded by the advection CFL itself;
        T floored at T_MIN — every floor hit increments a debug "energy-floor" counter:
        the named 4th sink, visible in the sealed-room energy gate)
     f. zero u,N on solid; masks as today
2. p* := C · N_total · T            (wide mul, §3.4)
3. HELMHOLTZ SOLVE (once per tick, fixed sweeps, RB-GS, red-black — §3.4 numerics):
     BCs: Neumann mirror at solid; Dirichlet P=0 at vacuum
4. u -= dt·grad(P_new)/ρ̂
   u *= (1 − absorb·dt)  per cell   (unit/material shockwave absorption — D4; reads
                                     dyn_wave_absorb exactly as the old wave kick did)
   zero u outside open-air
5. P := P_new  — materialized ONCE, stored (aliased as `atmosphere`), BEFORE any consumer
6. combustion pass (§5, patch 4+) — reads settled P/N/T, feeds next tick
7. consumers (§6): smoke/fire advection on u(=wind), water head, burst walls, unit push
```

### 3.3 Momentum representation — carried simplification, now with a stress gate

Velocity self-advection (not conservative `N·u`) is carried from the prototype for the
same stability reason (no divide-by-small-N at breach fronts). It remains a **named
physics simplification**, and patch 3's gate now includes the specific stress probes
round-1 asked for: multi-grenade stacks, O2-tank-rupture fireball, and an
interacting-blast (Woodward–Colella-flavored) scenario. If it visibly misbehaves there,
the fallback is conservative momentum *with the same wide-intermediate discipline* —
a scoped pivot, not a redesign.

### 3.4 Fixed-point numerics (B3/B4 — the load-bearing spec)

The Helmholtz face coefficient is `k_f = dt²·c²/ρ̂_f`. With `N → N_floor` this is
**unrepresentable in Q16.16** (round-1 measured ~450× over ceiling at the prototype's
constants). The v2 rules:

1. **Wide arithmetic end-to-end in the solve.** `k_f`, the diagonal
   `d = 1 + Σ k_f`, the neighbor sum `Σ k_f·P_nb`, and the RHS are all computed and
   held in **int64 at Q16.16 scale** (the `mul_wide`/`narrow` idiom in
   `fixed_point.h`). Narrowing to int32 happens **exactly once** per cell per sweep — at
   the final quotient `P_new = wide_num / d` (via a widened `reciprocal`/long-division
   path, NOT the q16-input `reciprocal_q16`, whose validated input range this divisor
   exceeds — round-1 finding honored).
2. **Solver-local density floor** `ρ̂ = max(ρ, RHO_FLOOR_SOLVER)`, chosen so
   `max k_f = dt²c²/RHO_FLOOR_SOLVER` (plus the ×4 face sum, plus the max representable
   `P` in the neighbor products) fits int64 with ≥ 8 bits of headroom — the concrete
   inequality with numbers is patch 3's **design-gate deliverable #1**. This floor is
   solver-internal only: gameplay N (suffocation, combustion) reads the *real* unfloored
   field.
3. **The gate must include an overflow stress sweep** (B4): drive `N` to floor across a
   breach/blast scenario and **assert no intermediate exceeds its container** —
   explicitly *in addition to* accuracy-vs-double-reference, because an overflow is
   bit-identical on both platforms and invisible to the digest gate.
4. **Sweep count re-derived, not inherited.** The prototype's `PRESSURE_SWEEPS = 40` vs
   the shipped kernel's `gs_iters = 8` is a 5× cost gap (B7). Patch 3's gate measures
   convergence-vs-sweeps on the real Q16.16 operator across the stress scenarios and
   pins the count as a **solver constant** (fixed forever after — never adaptive), with
   the p99 tick-cost target (§8 P3) as the binding constraint.
5. Substep count + `max|u|`: integer-ceil + `sqrt_q16` (D5), per §3.2.
6. **Per-sub-kernel digest checkpoints** in patch 3: advection, bulk flux, `p*`
   materialization, Helmholtz, velocity correction, compression-work — six digests, not
   one end-of-tick hash (a compensating-error pair must not be able to hide).

---

## 4. Unified temperature field

Unchanged from v1 in architecture (decision 7 / A1: one Q16.16-Kelvin array; gas rules
on the open-air mask — advect + compression-work + sources, **no decay**; solid rules on
the solid mask — the shipped convert/conduct/cool pipeline untouched; **conduction is
one whole-grid pass** keyed on `conductivity`, with air given a small nonzero value —
the free solid↔gas interface, the sealed-room energy sink).

v2 refinements:

- **Compression work is substepped** (§3.2 step e) — D3's stability bound honored by
  construction (per-substep `div·dt_s` is CFL-bounded).
- **The `T_MIN` floor is a named, counted energy sink** (debug telemetry), and the
  patch-2 sealed-room energy-balance E2E asserts the counter stays at zero in
  non-abusive scenarios.
- Energy exits (unchanged): vent to vacuum (advect/flux out), conduct into structure,
  hull radiates via `cool_shift_vacuum` (reinterpreted as radiate-to-space). Interior
  solids conduct to gas; the phantom ambient-decay is retired for gas and interior
  solids alike.
- The gas heat deposit `ΔT = ΔE/(N_total·c_v)`: per-tile reciprocal, **with
  `N_total` floored by the combustion/deposit floor `N_FLOOR_HEAT`** — independent of
  the tunable `o2_thresh` (round-1 minor), and distinct from §3.4's solver floor.
- `heat` (the ray-deposit buffer) keeps its per-tick-clear contract — unaffected.

---

## 5. Combustion on real O2 — conservative products (v2 call #2 — LOCKED)

Structure per v1 (own pass, once per tick, after P materialization — cadence matches
today's `FireSimulation` and stays the lowest-risk choice; revisit only if the fireball
feel demands sub-tick burn), with the mass-bookkeeping fix:

```
if N_O2 > o2_thresh_burn AND T ≥ ignition_temp AND fuel > 0:
    burn        = burn_rate · dt                          (clamped by available N_O2)
    N_O2       -= burn                                    (integer, saturating ≥ 0)
    N_smoke    += burn · soot_yield                       (visible product, decay-permitted)
    N_N2       += burn · (1 − soot_yield)                 (burnt products → inert bulk;
                                                           N_total CONSERVED — D2)
    T          += burn · H_fuel / (c_v · max(N_total, N_FLOOR_HEAT))   (§4.3 reciprocal)
```

A sealed room that burns now behaves physically: pressure **rises** with T during the
fire, relaxes as heat conducts away, and the baseline never fake-drains from
bookkeeping. O2 depletion still starves the fire (the intended effect), and the O2→N2
conversion means "burnt air" is exactly that — unbreathable but pressure-bearing.

Emergent payoffs (unchanged, now actually delivered by §3's native physics):
self-starving fires, breach-kills-fire, **O2-tank rupture → fireball** (a local `N_O2`
spike — patch-1 range check that a tank's spike fits Q16.16 headroom), inert-flood
smothering. Suffocation reads real `N_O2` (unit-side mechanics arc, enabled here, wired
later). `o2_thresh_burn` and `o2_thresh_breathe` are **separate constants** (§9).

---

## 6. Downstream consumers on derived P

| Consumer | Change | v2 note |
|---|---|---|
| wind / smoke advection / fire fan | **none** (reads `wind` = view of solver `u`) | one-velocity unification (§2.3) |
| water pressure-head (W4) | read integer `P` via `mul_q16(kp_q, P)` — float bridge **removed** | `k_p` recalibration in §9; `water_solver.cpp` structure untouched |
| ripple splash (W6a) | reads **`|P − P_prev|`** (per-tick transient) instead of raw `wave_p` | D6: no standing-baseline splash; P_prev kept per §3.2 step 0 |
| `find_burst_walls` | none (reads `P` spread via the `atmosphere` alias) | furniture note: furniture tiles are open-air for gas ⇒ they carry real `N`, so "solid contributes 0" only applies to true solids — one assert in the patch-3 gate |
| `apply_wave_push` (units) | reads `grad(P)`; `k_push` recalibrated (§9) | the zero-mean fast path survives: `∇P ≡ 0` over uniform P, integer-exact, baseline-independent — verified by a dedicated test, not assumed |
| unit shockwave shielding | **placed**: per-cell `u`-damping in §3.2 step 4, driven by `dyn_wave_absorb` | same coefficient field, same feel intent; `test_wave_absorption` reworked to assert through-body attenuation of a passing P transient |
| fire O2 gate + `apply_temperature_ignition` | reads `N_O2` (patch 4) | until patch 4 they read the `atmosphere` alias — behavior unchanged through the migration window |
| temperature vacuum-cool compare | none (alias) | |
| FieldEdit `atmosphere`/`wave_source` policies | remap to **N/T deposits** (explosion = energy) in patch 3 | payload `pressure` param becomes an energy scale; one table edit + doc |
| recorder | field list updated in patch 3 (`wave_p` → `P`; add `N_O2`) | blowup trigger re-keyed on `|P − P_prev|` max |

Units: partial obstacles/absorbers to pressure (kept, via the damping placement);
transparent to water (kept).

---

## 7. Determinism / fixed-point plan

- All fields Q16.16; the solve's internals wide-int64 per §3.4 (a *representation*
  discipline, not a new format class).
- **No global reductions remain in the sim path** (renorm deleted) except `max|u|` —
  an order-free integer max. Donor-cell flux is per-face integer transfers (exact,
  associative-free by construction: sequential per-face pass on CPU; on GPU, the
  red-black / face-coloring pattern `cuda_water.cu` already uses).
- Fixed GS sweep counts (never adaptive); fixed substep cap; integer-ceil counts;
  `sqrt_q16` everywhere a magnitude is needed.
- **CPU + CUDA lockstep** (decision 9): every new/changed kernel double-implemented and
  digest-gated. **GPU backends for atmosphere/wave/smoke/temperature pinned to CPU from
  patch 3 until their patch-6 port lands** (D7) — the stale kernels must be
  unreachable, not just unused.
- Per-sub-kernel digests in patch 3 (§3.4.6).
- Float bridges: water-head bridge and the `atm_f_`/`wave_p_f_` mirrors **removed** (their
  sole consumer is migrated in patch 3); render-only channels stay float as ever.

---

## 8. Patch decomposition (re-derived — every patch leaves the game runnable)

Ordering fixes B5/B6: temperature lands **before** the solver needs it; `wave_p`
retirement and its reader migration are **atomic** in one patch; `atmosphere` is
alias-preserved throughout, so no patch strands a legacy reader.

1. **P1 — species + conservative transport (additive).** O2/N2 ids (`N_GASES` 5→7);
   donor-cell flux transport for the bulk pair **riding today's wind field** (no solver
   change — purely additive; nothing consumes the new species yet); §2.1 calibration
   (ambient P preserved to ≤1 count; O2-tank spike range check). *Gate:* legacy 5
   species bit-identical; sealed-room bulk conservation **exact** over 1000 ticks;
   save/load + field-edit round-trip.
2. **P2 — unified temperature (additive).** Gas-T on the unified field (advect on
   today's wind; conduction unified via air conductivity; radiation deposits via the
   §4.3 reciprocal; no compression work yet — that term belongs to the new solver's
   `div u`). Old solver untouched; ignition still on the legacy path. *Gate:*
   sealed-room energy-balance E2E (conduct→hull-radiate, floor-counter = 0); existing
   temperature tests green (solid path unchanged).
3. **P3 — the compressible solver + atomic consumer migration.** *Design-gate first*
   (§3.4 deliverables: overflow inequality with numbers, sweep count, substep cap).
   Then: true-Kwatra solve replaces `wave_substep`+`diffuse_solve`; compression work
   moves into the substep loop; `u` becomes the one velocity (wind views);
   `atmosphere` re-pointed as the P alias; **in the same patch**: `apply_wave_push` →
   `grad(P)`, water head → integer P (bridge removed), ripple → `|P−P_prev|`, FieldEdit
   remap, recorder update, absorption placement + `test_wave_absorption` rework,
   `sink_hop` + `wave_solver.*` deleted, breach→vacuum generalized, GPU backends pinned.
   *Gates:* 6 sub-kernel digests; overflow stress sweep; the quiescent-cold-breach
   native-venting E2E; §3.3 stress probes; behavioral-parity bakes for push/head;
   **p99 ms/tick ≤ 25 % of the 83 ms budget at 160² on the dev desktop** (hard number,
   worst scenario, not mean).
4. **P4 — combustion on real O2.** §5 wholesale (conservative products, floors,
   both thresholds); ignition + fire O2 gate re-pointed to `N_O2`. *Gate:* the four
   emergent payoffs as E2E scenarios (mechanism visible; constants still TBD).
5. **P5 — combined-system bake-off (HUMAN-TEST).** S1–S5 (+ the venting scenario)
   baked on the assembled stack vs the pre-refactor engine; cost table (p99);
   **Erik's eyes are the gate.** First feel-tuning pass of §9 happens here.
6. **P6 — CUDA ports**, one gated sub-patch per kernel, full surface: Helmholtz solve;
   velocity self-advection; T advection + compression work; bulk donor-cell flux
   (precedent `cuda_water.cu`); unified conduction (extend `cuda_temperature.cu`);
   combustion pass; **retire** `cuda_wave.cu`/`cuda_atmosphere.cu`; unpin backends.
   *Gate:* bit-identical digests per kernel (the `cuda-breached` harness), non-negotiable.
7. **P7 — cleanup + canon.** Formal `atmosphere`→`P` deprecation decision; float-mirror
   removal confirmation; stale-doc fixes (interaction map §0's six spots); fold the
   as-built design into engine/04 + 06 chapters and archive the brainstorms (the
   post-EOS consolidation commitment).

Dependencies: P1, P2 independent (parallel worktrees fine). P3 needs both. P4 needs P3.
P5 needs P4. P6 per-kernel after the corresponding CPU code stabilizes (≥ P3). P7 last.

---

## 9. TBD / tuning (feel-gated; first pass at P5)

`c_max` (sound-speed dial — trades Helmholtz conditioning, *not* substep count);
`k_push` + knockdown thresholds (vs the new transient-∇P scale); `k_p` (water head);
air conductivity ("small": big enough that the interface sink fires, small enough that
air doesn't become the rejected heat-advecting field); `cool_shift_vacuum` rate under
the real energy path; combustion `burn_rate / H_fuel / soot_yield / o2_thresh_burn`;
`o2_thresh_breathe` (separate); `CFL_ADV` + `N_SUB_MAX`; sweep count (pinned at the P3
gate, then frozen). **Gone from this list vs v1:** `K_EXPAND`, `W_DISPLACE_GAIN` (deleted
with `div_target`).

## 10. Remaining open items (honest residue)

1. **Velocity self-advection under abuse** — carried simplification; P3's stress probes
   are the decision point; conservative-momentum fallback scoped (§3.3).
2. **Sweep count / cost** — unknowable on paper for the Q16.16 operator; P3's gate
   measures and pins it against the hard p99 target.
3. **Feel of the merged transient** (buffet-vs-dome from one field's evolution) —
   physically sound, but the *game feel* of knockback/ripple under the new P dynamics is
   exactly what P5's HUMAN-TEST exists to judge.
4. **Combustion cadence** — once-per-tick chosen (matches today); flagged for revisit
   only if P5's fireball feel wants sub-tick burning.
