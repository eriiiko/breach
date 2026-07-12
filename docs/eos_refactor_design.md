# EOS refactor — design document **v2** (rung B, compressible ideal gas)

> **Status:** v2 DRAFT for round-2 critique, then Erik. Supersedes v1 in place; the round-1
> critique lives at `docs/archive/eos_refactor_design_critique.md` (archived P7 — folded into v2).
> Companions:
> `eos_refactor_decisions.md` (LOCKED decisions incl. the three v2 calls below),
> `docs/archive/eos_refactor_interaction_map.md` (archived P7 — design-prep, absorbed here),
> `eos_research_report.md` §4. The float prototype
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

### v2.1 — round-2 fixes (same day; round-2 verdicts: 14/16 round-1 findings FIXED, 2 partial → closed below)

- **Occupancy-transition mass rule (round-2 physics BLOCKER):** a cell leaving the open-air
  mask (flooding water, closing door, spawned wall) **evacuates its `N` conservatively** into
  open neighbors via the same donor-cell/limiter machinery — never the zero branch (§2.2).
  Zeroing is for vacuum drain only. This is *also* the real W3 wiring: water displacement
  **is** that evacuation (§3.1) — no field multiply at all.
- **Writer migration completed (round-2 determinism BLOCKER):** the three direct
  `atmosphere` writers are now explicit P3 tasks — W3's `atmosphere *= ratio`
  (`physics_engine.cpp:599`) → replaced by the evacuation rule; fire's plume
  `atmosphere[i] += gain` → a minimal plume→T shim **in P3** (the "pop" never goes inert);
  `destroy_wall`'s neighbor-mean refill → seeds the new open tile's `N` by neighbor-mean
  (same anti-vacuum-pulse intent, now on the real state). FieldEdit `atmosphere` policy →
  bulk-`N` deposit (21/79 split); `wave_source` policy → `T` energy deposit (§6).
- **Trace decay returns mass to `inert_N2`** (round-2: D2 was only *partially* fixed —
  decaying soot re-opened the slow pressure drain). Decay = settling/oxidation into inert
  bulk; `N_total` now conserved through the *full* burn-decay cycle (§5).
- **Permeability also scales the Helmholtz face coefficient `k_f`** — a throttled face
  throttles pressure coupling coherently with species flux (no knockback through a shut
  door that visibly blocks smoke) (§3.4).
- **`ρ̂ := N_total`** stated (unit particle mass — consistent with molar mass dropped);
  the changelog's "N_floor" and `RHO_FLOOR_SOLVER` are the same constant (§3.1).
- **The wide divide is amortized:** the Helmholtz diagonal `d` is constant across sweeps
  within a tick ⇒ its wide reciprocal is precomputed **once per cell per tick**, reused
  every sweep — the per-cell divide cost worry collapses (§3.4).
- **`CFL_ADV ≤ 0.5` pinned as a constraint** (not free TBD) ⇒ the compression-work bound
  `(γ−1)·2·CFL_ADV ≤ 0.4 < 1` is guaranteed, closing D3's caveat (§3.2).
- **P3 design-gate deliverable #0: a napkin cost model** reconciling the 85%-of-budget
  worst-case baseline with the p99 ≤ 25% target *before* build; **deliverable #4: verify
  the operator's ρ̂ placement line-by-line against Kwatra eq. 15–17** (self-adjointness /
  diagonal dominance) (§3.4, §8).
- Minors: `cuda_water.cu`'s determinism pattern correctly named (**precompute-then-gather**,
  not face-coloring) (§7); furniture gates donor-cell flux via the same face permeability as
  today's diffusion and carries real `N` (§6); named fidelity note — bulk `N` (donor-cell)
  and `T` (semi-Lagrangian) use different schemes and can decorrelate at sharp fronts (§1);
  `dyn_wave_absorb` now also locally damps the smoke-carrying wind (named for the P5
  feel-check, §10); empty-sealed-room acoustic ringing is damped only by numerical
  diffusion/RB-GS smoothing — acceptable, noted (§10).

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

**Fidelity limits, named:** fixed-sweep GS on the Helmholtz operator smooths
discontinuities — blasts are soft compression waves, not Rankine–Hugoniot shocks, by
construction. And bulk `N` (donor-cell) vs `T` (semi-Lagrangian) use different advection
schemes, so the two can mildly decorrelate at sharp fronts (a fireball edge), feeding
`p*` a slightly inconsistent state there. Both are intended game-scale aesthetics, not
accidents.

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

Traces stay semi-Lagrangian (visual tracers; their decay is a feature — and their decayed
mass is credited to `inert_N2`, §5, so the Dalton sum stays whole).

**Occupancy-transition rule (v2.1, load-bearing):** when a cell leaves the open-air mask —
water floods it, a door closes onto it, a wall is spawned — its `N_i` is **evacuated
conservatively** into adjacent open cells via the same donor-cell/limiter machinery
*before* the cell is masked; it is never zeroed. (Zeroing remains correct only for vacuum
cells, where mass genuinely leaves the system.) This rule *is* the water-displacement
mechanism: rising water pushes its cell's air into the neighbors, `N` rises there,
`p* = C·N·T` rises, and the push falls out of §3 — no field multiply, no gain constant.
Conversely a cell *joining* open-air (`destroy_wall`) is seeded by neighbor-mean `N`
(the same anti-vacuum-pulse smoothing the old code applied to `atmosphere`).

**Furniture:** open-air for gas (carries real `N`); its partial permeability gates
donor-cell face flux exactly as it gates today's diffusion stencil.

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
solve [I − (γ·p*)_cell·dt²·K·∇·( (1/N̂_face)·∇ )] P_new
        =  p* − (γ·p*)_cell·dt·div(û*)      # Kwatra eq.(14) with ρc² = γp (exact ideal-gas
                                            #   identity — the paper's own EOS-generality);
                                            #   solved by FIXED-SCHEDULE MULTIGRID (v2.2)
u    -= dt · K · grad(P_new) / N̂_face       # momentum kick; K = the ONE unit-bridge constant
```

**v2.2 (D-A, adopted by Erik 2026-07-10): the coefficient is `γ·p*`, not `N·c²`.** P3's gate
measured the transplant's `N(game-units)·c²(SI)` coefficient at ~64,000× the pressure
field's scale — a unit/impedance inconsistency that saturated the field at ANY sweep count.
The exact identity `ρc² = γP` eliminates the mixed-unit coefficient entirely: `γ·p*` is in
P's own units by construction. One unit-bridge constant **`K`** survives, in the momentum
update only, **calibrated once so that ambient sound speed `c = √(γP·K/ρ̂)` equals 300 m/s
at (P=1, N=1)**. Consequences: **`c` is now state-derived — `c ∝ √T`** (hot blast cores get
proportionally faster, sharper acoustics; physically correct), and Erik's dial survives as
`K` (scales ambient c uniformly; the 66→300 range is a graceful-degradation continuum if
ever needed). Solver complexity unchanged — the operator already carried a per-cell
coefficient; it now reads the `p*` array instead of a constant. Overflow budget re-derived
at the MG gate (the face-coefficient product is preserved — P3's measurement).

**Operator placement — VERIFIED against the paper (deliverable #4, 2026-07-10, from
`docs/papers/ADA492343.pdf` eq. 9–15), restated in v2.2 units:** the paper's `ρc²` is an
**outer, per-CELL multiplier evaluated pre-solve** on both sides — in v2.2 form that
multiplier is **`(γ·p*)_cell`** (the exact identity ρc² = γP; γ = 1.4, compile-time
constant). `1/N̂` sits **inside** the divergence-gradient sandwich, **per-FACE**, with
`N̂_face = (N_i + N_j)/2` from the *post-advection* densities, floored by `N_FLOOR_SOLVER`.
(v2's `∇·((c²/ρ̂)∇)` form — density inside, no outer factor — was uniform-density-only;
corrected per round-2. The v2.2 unit fix replaces the outer `N·c²` with `γ·p*` —
**every formula in this doc now uses the γ·p* form; any surviving `N·c²` is an error.**)

**Three properties, stated explicitly (round-3 critique demanded them, they were implicit):**
1. **The per-tick system is LINEAR.** `p*` is computed once (step 2), and the outer
   coefficient `(γ·p*)_cell` is FROZEN at that advected value through the whole solve —
   it is data, never a function of the unknown `P_new`.
2. **Near-vacuum degeneracy is the correct physics, for free:** as `p*_i → 0` both the
   outer coefficient and the RHS vanish → the row collapses to `diag = 1, RHS = 0 ⇒
   P_new = 0` — the intended Dirichlet behavior emerges. Correspondingly,
   `N_FLOOR_SOLVER` applies ONLY to the face `1/N̂` divide (where zero is dangerous),
   NEVER to the outer multiplier (where zero is the desired physics).
3. **Why D-A cannot fix D-B (inline, since it's the crux):** the pressure↔velocity
   round-trip coupling is `(γ·p*)·K = p*·(c_amb²·…) ≈ p*·c²` at ambient — D-A only
   *relocates* the c² factor (operator → momentum constant); the product that sets the
   solve's difficulty is unchanged. Hence two independent fixes.
Diagonal dominance still holds row-wise (diag = `1 + (γp*)_i·K·dt²/dx²·Σ_f(perm_f/N̂_f)` >
off-diagonal sum, by the identity term) — but per the P3 measurements this guarantees
only ASYMPTOTIC convergence, NOT small-fixed-schedule convergence; that is the multigrid
section's problem (§3.2 step 3), no longer claimed solved by dominance alone. **Named deviation from the paper:** Kwatra advects pressure itself (`p_a`); we
derive `p* = C·N_adv·T_adv` from the advected state — a consistent O(dt) choice that
guarantees P can never drift from (N, T); the paper itself notes the method is
EOS-agnostic. (Energy: the paper updates E conservatively; we carry T with the explicit
compression-work term — the §3.3-class named simplification.)

where **`ρ̂ := N_total`** (unit particle mass — the one consistent choice given molar mass
is deliberately dropped; a future implementer must NOT invent a real molecular-mass
conversion here while `P = C·N·T` ignores it) under a **solver-local floor**
`N_FLOOR_SOLVER` (§3.4 — the changelog's "N_floor" and "RHO_FLOOR_SOLVER" are this one
constant), and `c` is the capped sound-speed dial. The identity term keeps the operator **strictly diagonally
dominant** (round-1 "sound" column) — fixed-sweep convergence stays guaranteed.

What the absolute-`p*` RHS buys, with no source heuristics at all:

- **Breach**: vacuum cell holds `P = 0` (Dirichlet); the room's `p*` is ~1 atm; the
  solve produces a steep `P_new` gradient at the hole; step-3 kicks `u` outward; flux
  transport (§2.2) carries N out; as `N → 0`, `p* → 0` and the flow **stops by itself**.
  Venting is the equation, not a mechanism. *(Gate: the quiescent-cold-breach E2E, §8 P3.)*
- **Explosion/fire**: `T` spike ⇒ `p*` spike ⇒ outward kick ⇒ expansion ⇒ §4's
  compression-work cools the parcel — the fireball arc, natively.
- **Water rise (W3)**: a flooding cell **evacuates its air conservatively** into its
  neighbors (§2.2's occupancy-transition rule) ⇒ their `N` rises ⇒ `p*` rises ⇒ air
  pushed — no ×60 gain, no field multiply, and no mass ever deleted at the waterline.
  (Replaces `physics_engine.cpp:599`'s `atmosphere *= ratio` — an explicit P3 task.)

`div_target`, `K_EXPAND`, `W_DISPLACE_GAIN` are **gone**. The buffet-vs-dome distinction
consumers rely on is P's own time evolution: the acoustic transient rides `P_new`'s fast
relaxation; the standing dome is its slow component. *(§6 verifies the `apply_wave_push`
fast path: `∇P ≡ 0` over any uniform region regardless of baseline — integer-exact.)*

### 3.2 Tick order

```
0. P_prev := P (kept copy — ripple transient + debug)     [P was materialized last tick]
1. ADVECTION SUBSTEPS — n = ceil_int(dt / dt_adv), integer-ceil discipline
   (smoke_cliff_count class); dt_adv = CFL_ADV·dx / (u_est + eps) with
   **u_est = max|u| + (max K·|∇P|/N̂)·dt** — the paper's own velocity estimate (its §3:
   `max|u|` alone under-substeps a quiescent field about to be kicked, e.g. the Sod tube
   or OUR cold-breach tick-0); note the **K** (v2.2 — the ∇P term converts pressure to
   acceleration through the same unit bridge as the momentum kick). max|u| via sqrt_q16
   over vx²+vy² (int64 sums); the ∇P term from last tick's stored P (integer ops only);
   **u_est capped at c_LOCAL = c_amb·sqrt(T_max_abs/T_AMB)** (v2.2: c is state-derived —
   a 9000 K core's sound speed is ~5.5× ambient; a stale ambient cap would re-create the
   under-substep failure this term exists to prevent; T_max_abs = per-tick max over
   open-air, one sqrt_q16); **N_SUB_MAX = 16** (microbench 2026-07-10:
   the stress tail reached n=159, but our substeps exist for ACCURACY not stability — SL
   is unconditionally stable and the donor-cell limiter rate-caps gracefully — so a low
   cap costs only slight front-resolution on a blast's wildest 1-2 ticks, invisible at
   1/3 m tiles, instead of a designed frame stall).
   per substep (dt_s = dt/n):
     a. u  ← semi-Lagrangian self-advection            (§3.3 simplification, flagged)
     b. T  ← semi-Lagrangian advection (gas mask)
     c. (traces do NOT substep — see below)
     d. bulk N_O2, N_N2 ← donor-cell flux on u          (§2.2 — conservative)
     e. (COMPRESSION WORK DOES NOT HAPPEN HERE — corrected 2026-07-10, see step 4c.
        v2.1 had it in this loop, which DOUBLE-COUNTS the compression physics: the
        advected T would already carry this tick's compression response into p*,
        while the Helmholtz RHS's −(Nc²)·dt·div(û*) term carries the SAME physics
        into the solve — an ≈(2γ−1)-vs-γ over-response per tick ⇒ a growing pressure
        oscillation INDEPENDENT of sweep count. Found by P3's gate; root-caused on
        paper. In the paper, p_a is PURELY advected and eq.(3)'s energy update runs
        post-solve on the corrected state.)
     f. zero u on solid; N follows the occupancy-transition rule (§2.2 — evacuate,
        never delete); masks as today. CFL_ADV ≤ 0.5 (pinned constraint, not TBD):
        guarantees the step-e bound (γ−1)·2·CFL_ADV ≤ 0.4 < 1 by construction.
2. p* := C · N_total · T            (wide mul, §3.4)
3. PRESSURE SOLVE (once per tick) — **FIXED-SCHEDULE MULTIGRID V-CYCLES** (v2.2, D-B,
   adopted by Erik 2026-07-10), RB-GS as the smoother at every level:
     BCs: Neumann mirror at solid; Dirichlet P=0 at vacuum
     WHY (P3's gate measurement): the implicit solve must propagate influence
     c·dt/dx ≈ 75 tiles per tick at ambient c=300, but point-GS moves information
     ~1-2 cells/sweep — the room-scale solution is STRUCTURALLY unreachable at small
     fixed S; the un-solved residual behaves like the explicit scheme we escaped
     (CFL≈37 ⇒ the measured ×700/tick blow-up), and S≈128 (46 ms) is unaffordable.
     Multigrid solves on a grid pyramid (160²→80²→…→~10²) — coarse levels carry
     influence across the whole ship in one cycle, ~4/3 the fine-level cost,
     ~10×-error-reduction per V-cycle independent of grid size.
     DETERMINISM: fixed cycle count × fixed sweeps/level × fixed integer transfer
     stencils = a fixed integer-op sequence, exactly as reproducible as fixed-S GS.
     TRUTH LIVES ON THE FINE GRID: coarse levels are accelerators only — an imperfect
     coarse mask degrades convergence RATE near complex geometry, never correctness.
     Schedule (V(2,2), level count L, cycle count C) pinned at the MG design-gate,
     frozen thereafter.
4. u -= dt·K·grad(P_new)/N̂_face      (v2.2: K is MANDATORY — without it this line IS the
   64,000× unit bug. K = c_amb²/γ ≈ 64,286: does NOT fit a plain Q16.16 value — stored
   as a WIDE int64 Q16.16-scaled constant; the whole kick chain runs int64
   (K·ΔP → /N̂ → ·dt), clamps |u| ≤ c_LOCAL, and narrows to q16 velocity ONCE at store)
   u *= (1 − absorb·dt)  per cell   (unit/material shockwave absorption — D4; reads
                                     dyn_wave_absorb exactly as the old wave kick did)
   zero u outside open-air
4b. traces ← per-slice SL, ONCE per tick on the final velocity (microbench amendment
    2026-07-10: they are visual, non-conservative, and run once/tick today — substepping
    them multiplies 5 field passes × n for zero gameplay effect)
4c. **compression work, ONCE per tick, POST-correction (the paper's eq.(3) analog,
    T-carrier form):** T -= (γ−1)·T·div(u_new)·dt, using the CORRECTED velocity — the
    energy bookkeeping consistent with the flow the solve actually produced. It feeds
    NEXT tick's p*, never this tick's solve (no double count, no phase lag). T floored
    at T_MIN with the debug energy-floor counter (the named 4th sink). Stability: the
    per-tick |(γ−1)·div(u_new)·dt| can exceed the old substepped bound — clamp the
    factor to [T_WORK_CLAMP_LO, hi] (e.g. ±0.5) as a safety rail, counter-tracked like
    the floor.
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

The face coefficient (v2.2 form) is `k_f = (γ·p*)_cell·K·dt²/(dx²·N̂_f)` — ambient value
≈ 5,580 (the pre-v2.2 product, preserved by construction). It is **unrepresentable in
Q16.16** near floors and spikes. The rules:

1. **Wide arithmetic end-to-end in the solve.** `k_f`, the diagonal
   `d = 1 + Σ k_f`, the neighbor sum `Σ k_f·P_nb`, and the RHS are all computed and
   held in **int64 at Q16.16 scale** (the `mul_wide`/`narrow` idiom in
   `fixed_point.h`). Narrowing to int32 happens **exactly once** per cell per sweep — at
   the final quotient `P_new = wide_num / d` (NOT via the q16-input `reciprocal_q16`,
   whose validated input range this divisor exceeds — round-1 finding honored).
   **Cost amortization (v2.1):** `d = 1 + Σ k_f` is constant across sweeps within a tick,
   so its widened reciprocal is precomputed **once per cell per tick** and reused every
   sweep — the per-cell wide divide is paid once, not `sweeps×`.
   **Permeability scales `k_f` (v2.1):** each face's Helmholtz coefficient is multiplied
   by the same face permeability that gates species flux — pressure coupling and mass
   exchange throttle *coherently* (no full-strength knockback through a shut-but-leaky
   door that visibly blocks smoke).
   **Budget recomputed for the verified operator (2026-07-10):** worst face coefficient
   `k_f = (N_cell·c²·dt²/dx²)/N̂_f ≈ 894·N_cell/N̂_f` — even an O2-tank cell (200× ambient)
   venting against the 10⁻³ floor peaks the wide products at ~2.4×10¹⁵, ≈12 bits under
   int64. `N_FLOOR_SOLVER = 10⁻³` stands.
   **Sharper bound (c=300 update, same day):** because `N̂_face` is the arithmetic mean of
   its two cells, `N_cell/N̂_face ≤ 2` ALWAYS — so `k_f ≤ 2·c²·dt²/dx²` regardless of any
   density configuration (`≈ 11,180` at c=300, inside Q16.16's value range), and the
   "tank-spike cell against a floored face" pathological case is structurally impossible
   (the face mean is at least half the spike). The floor only governs faces where BOTH
   neighbors are near-vacuum — where P is also near zero. The budget closes at c=300
   with wide margin.
   **v2.2 JOINT-CASE CAVEAT (round-3 critique — deliverable #1 RE-OPENED, closes at the
   MG gate):** the bound above was for `N` alone. Under γ·p*, a **joint** spike —
   O2-tank `N≈200×` AND fireball `T≈9000 K` — gives `p* ≈ 6,200` (fits Q16.16, ~5×
   headroom) but `k_f ≈ 3.5×10⁵` (int64-only) and worst products `k_raw·P_raw ≈ 9×10¹⁸ —
   AT the int64 edge.** The MG gate must re-derive the full inequality under γ·p*·K with
   explicit operation ORDERING (e.g. divide by N̂ before multiplying by P_nb) and/or
   coefficient caps, per level. `K = c_amb²/γ ≈ 64,286` itself is a **wide int64
   constant** (does not fit q16 — see §3.2 step 4).
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
4. **(v2.2 — supersedes the fixed-S plan, which P3's gate killed):** the solver is a
   fixed-schedule multigrid V-cycle (§3.2 step 3). **The MG design-gate is a MEASUREMENT
   gate, hardened per round-3 critique** (this project has now twice been burned by
   "asymptotically convergent ≠ convergent in the fixed schedule"):
   - **Convergence is MEASURED at the real coupling (~10³, variable-coefficient, masked
     domain, fixed-point), not assumed from textbook Poisson folklore.** V(2,2)×C=2 is
     the STARTING guess; the gate escalates (V(3,3), W-cycles, more cycles) until the
     venting + water E2Es are durably stable, then freezes the schedule at measured cost.
   - **Coarse operators: RE-DISCRETIZED with HARMONIC face-coefficient averaging** (the
     `face_shift` idiom this codebase already trusts; robust to coefficient jumps).
     Galerkin RAP is rejected: sparse-matrix assembly is alien to the integer-stencil
     codebase. If measured convergence near walls/doorways is inadequate, THAT is the
     trigger to revisit — measured, not assumed.
   - **Coarse Dirichlet (vacuum) rule, explicit:** a coarse cell is Dirichlet iff ALL its
     children are vacuum; straddlers remain regular cells (their fine-informed face
     coefficients carry the boundary's effect). Gated by a **dedicated
     breach-adjacent-to-coarse-boundary test** — the exact geometry of the original
     blow-up — NOT by an appeal to asymptotic MG theory (fine-grid-defines-truth is
     asymptotic; at fixed C it must be measured).
   - **Transfer operators: fixed integer stencils with a NAMED rounding rule**
     (full-weighting restriction via fixed `>>` shifts; bilinear prolongation with one
     stated round direction), identical CPU/CUDA — determinism is per-op specified, not
     asserted.
   - **Per-LEVEL overflow budgets** — coarsened coefficients aggregate; each level gets
     its own §3.4-style headroom derivation (incl. the v2.2 joint-case above).
   **§3.4's earlier "diagonal dominance ⇒ fixed-sweep guarantee" is retracted as an
   overclaim** (asymptotic convergence ≠ 8-sweep convergence at game coupling — the
   conflation P3's measurements exposed).

   **v2.3 — AS-BUILT amendments (P3's MG measurement gate, blessed by Erik 2026-07-10;
   full data `docs/eos_p3_gate_measurements.md`):**
   - **Transfers are VARIATIONAL/GALERKIN, not the re-discretized/bilinear pick above** —
     that pick was MEASURABLY DIVERGENT on deep pyramids (error ×7/cycle at a breach;
     round-3's Galerkin instinct vindicated by data). As built: the row divided by its
     diagonal factor gives an SPD mass+face-Laplacian form; masses/conductances/residuals
     SUM under coarsening and prolongation is the exact transpose — coarse corrections
     are energy-norm projections and CANNOT amplify at any depth. Straddler coarse cells
     fold their regular-child→vacuum conductances into the diagonal (the Galerkin
     Dirichlet anchor — omitting it reproduces the amplification).
   - **Frozen schedule: V(2,2)×C=2, full pyramid, coarsest-level 32 sweeps, WARM-STARTED
     from the previous tick's P** (deterministic — P_prev is state; bought ~2 cycles).
     Measured convergence ×0.55/cycle at real coupling; 300-tick durability: water
     worst-dev 0.0066 atm, vent overshoot 0.0005 atm.
   - **`trace_mass_scale = 0.02`**: traces are [0,1] OPACITY tracers, not molar
     densities — unweighted Dalton made a 0.6-opacity teargas cloud a +60% pressure bomb.
     Traces contribute to N_total through this scale.
   - **Trace advection is engine-owned `u·dt/dx`** (physical units); the legacy
     `advection_rate=900` config is dead; `wind_diffusion_scale` disabled pending P5 feel.
   - **N_SUB_MAX re-pinned 16 → 8** (measured equally stable; the cost driver is
     SUSTAINED sonic venting pinning the cap for the whole post-breach regime, not rare
     spikes) + a targeted micro-opt pass on the substep inner loop, gate re-measured.
4b. **Widen the GS flux-narrow (D-C, trivial):** `narrow(Σ mul_wide(face_k, ΔP))` wrapped
   int32 past ~23 atm neighbor differences — keep the accumulator wide until the final
   per-cell store.
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

### v2.4 as-built amendment — thermal/velocity rails (BLESSED by Erik at the P5 review, 2026-07-11)

*(branch `eos-p3fix-thermal-ceiling`, 2026-07-10/11 — the decisions.md #16 "thermal
spike" investigation. Same class as the blessed T_MIN/work-clamp rails, NOT a solver
redesign; every rail is counted telemetry, never silent.)*

**Corrected diagnosis of the flagged spike** (root-caused by per-term instrumentation,
not the flagged hypothesis): the P3 plume→T shim was a MINOR term (<1% of the measured
peak — though its self-limiter WAS structurally dead: it gated on `atmosphere[i]` at
its own solid tile, which the solver force-zeroes; fixed, now gates on T against
`T_FLAME_MAX≈2000`). The measured drivers are (a) **Pass-1's `ΔT=ΔE/(N·c_v)`
reciprocal** dividing the fire's radiant deposit by a collapsing local N — the hot
zone's own pressure evacuates its gas, so the same deposit heats the thinning remainder
ever harder — coupled with (b) **step 4c's multiplicative compression work**, whose
±T_WORK_CLAMP rail bounds the per-tick RATE but never the VALUE (compounds ~1.5×/tick
at the rail). Two int32 WRAP bugs rode on top (T past the Q16.16 ceiling; `u` past
int64 in the kick's magnitude test) — both fixed with saturating arithmetic
independent of any rail.

The rails:

- **`T_MAX_PHYS`** (config `[physics.thermal]`, default 16000 K-relative ≈ 2× the
  design's stated 9000 K extreme): a counted saturating clamp at every T write path —
  EOS step 4c, thermal Pass 1 (both branches), combustion's deposit (each with its own
  hit counter), plus FieldEdit's T deposit (Python authored-edit path — clamped,
  uncounted). Physically honest story: a near-vacuum
  cell's T is thermodynamically ill-defined; real gas would equilibrate the spike away
  instantly — the cap stands in for that missing fast equilibration. Bounds the
  runaway regardless of driving term. (Conduction needs no rail — convex combination;
  cooling only shrinks; SL advection is interpolation.)
- **`U_MAX`** (config `[physics.eos]`, default 1000 m/s): defense-in-depth velocity
  rail — the step-4 store clamp caps |u| at `min(c_LOCAL, U_MAX)`, counted; plus an
  overflow guard (±2^30 raw component pre-clamp) so the magnitude test's `u²` sum can
  never overflow int64 again (the measured chaotic-wind wrap).
- **`N_FLOOR_HEAT` checked, kept at 0.05**: the single-tick criterion
  `N_floor ≥ heat_tick_max/(T_MAX_PHYS·c_v)` holds at 0.05 for the measured worst
  deposit (~330/tick ⇒ 6,600 K < 16,000); the stacked-firestorm case is bounded by
  the counted T_MAX_PHYS rail (its job). A trial raise to 0.2 perturbed marginal
  ignition timings suite-wide for no correctness gain.

**Measured outcome (B4/B7/B6 re-baked, game-faithful loop — heat cleared per tick):**
the intended story, rails untouched (all counters 0): B4 T_fire rises to ~1240 (flame
scale), fire self-starves t≈39, temps decay, no pin/wrap, winds ≤ 5 m/s; B7 peak
excursion 11.9 kK decays to 5.1 kK, no pin; B6 smooth post-peak. In the
pr.step-only harness (heat never cleared — `tools/eos_p5_bake.py` and the P4 E2E
loops), stale heat re-radiates every tick and T pins at the rail — a HARNESS
fidelity artifact, flagged.

**The O2-gate second rescale (adopted with this block):** with T bounded, fire
scenarios lose the chaotic near-ceiling dynamics several shipped E2E calibrations
silently depended on — a fire's heat (measured ~330 units/tick/cell at
`k_fire_heat=9`) drives adjacent air to a few kK, whose pressure evacuates local O2
*thermally* in every scenario (at ideal-gas pressure equilibrium
`N_local ≈ N_amb·T_amb_abs/T_air_abs`; the P4-era gates at 0.12/0.126 implicitly
assumed near-ambient density at the flame edge — an atmosphere-proxy-era assumption
`P=C·N·T` revokes; real fires keep their flame edge oxygenated by buoyant
entrainment, which this 2D no-gravity model lacks, so the gate scale compensates).
Adopted (BLESSED by Erik at the P5 review, 2026-07-11): `P_min 0.126→0.01`, `P_full 0.21→0.03`, ignition
`o2_threshold 0.12→0.01` — the second half of the exact rescale P4 already performed
on these constants (1.0-scale → 0.21-scale → hot-zone-equilibrium scale). Measured:
restores STRONG O2 differentiation (sealed 172 / vented 49 / flooded 39 ticks in the
e2e trio) and the original ignition budgets (flamethrower dist-3 t=28) through
genuine oxygen physics, perturbation-stable — a 1e-5 dial perturbation no longer
moves timings (gated by `test_payoff_orderings_perturbation_robust`).

**Remaining P5 flags:** (1) a sealed room whose whole gas mass ends up at flame-scale
temperatures enters a *fuel-free smolder*: hot gas conducts the wood back above
`ignition_temp` indefinitely and `CombustionSolver` — which by P4 design consumes no
`wall_hp` ("wall_damage stays the sole fuel-consumption brake", combustion.h) — burns
O2 without consuming fuel for thousands of ticks (physically a sealed oven, but
fuel-free; surfaced by the e2e_1 re-pin, test_eos_p4_combustion.py). (2) The
pr.step-only bake harness (`tools/eos_p5_bake.py`, eos-p5-bake branch) lacks the game
loop's per-tick `heat` clear — one-line fidelity fix at merge time. (3)
`k_fire_heat`'s scale (gas near a wood fire reaches a few kK) is untouched — Erik's
feel dial.

**P5 flag status (2026-07-11, Erik's review):** (1) resolved by the **v2.5 amendment**
(§5 — P5.1 stoichiometric fuel consumption, Erik's clamp-at-1-LSB rule); (2) landed on
main (`cae2d13` + `f646056`); (3) remains Erik's §9 feel dial. The v2.4 rails, the
absorption-∝-density deposit, the O2-gate rescale, and the T_FLAME_MAX shim were all
**blessed as shipped** (decisions log #17).

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

**Trace decay routes to `inert_N2` (v2.1):** when any trace species decays (soot settling,
teargas dispersing), its lost mass is credited to `inert_N2` in the same cell — decay is
settling/oxidation into inert bulk, not deletion. With this, `N_total` is conserved through
the **full** burn-then-decay cycle, closing round-2's "soot decays and the pressure drains
anyway" residual of D2.

A sealed room that burns now behaves physically: pressure **rises** with T during the
fire, relaxes as heat conducts away, and the baseline never fake-drains from
bookkeeping — not at burn time, and not later as the smoke settles. O2 depletion still starves the fire (the intended effect), and the O2→N2
conversion means "burnt air" is exactly that — unbreathable but pressure-bearing.

Emergent payoffs (unchanged, now actually delivered by §3's native physics):
self-starving fires, breach-kills-fire, **O2-tank rupture → fireball** (a local `N_O2`
spike — patch-1 range check that a tank's spike fits Q16.16 headroom), inert-flood
smothering. Suffocation reads real `N_O2` (unit-side mechanics arc, enabled here, wired
later). `o2_thresh_burn` and `o2_thresh_breathe` are **separate constants** (§9).

### v2.5 amendment — P5.1 stoichiometric fuel consumption (Erik, 2026-07-11)

Closes v2.4's P5 flag #1 (the fuel-free smolder) and completes the fire lifecycle Erik
originally wanted: fires die (starved or blown out) → hot tiles ember → wind-borne O2
re-ignites them → they burn out or starve again. **No new state**: the ember is emergent
from the existing fields — `(fire I = 0, T ≥ ignition_temp, wall_hp > floor)`.

The §5 pseudo-code above already gates on `fuel > 0`, but the as-built P4
`CombustionSolver` takes `wall_hp` **const** — it reads the gate and never draws down
the store, so an I=0 smolder burns O2 and radiates heat from fuel it never consumes
(a perpetual ember). `FireSimulation`'s `wall_damage` pass remains the FLAME-scale
consumption (I>0, unchanged); combustion gains the EMBER-scale consumption:

- `CombustionSolver` takes `wall_hp` **mutable**; per neighbour burn the source tile pays
  `fuel_cost = narrow_round(mul_wide(fuel_per_o2_q, burn))` — round-to-nearest, the same
  unbiased-sink idiom `fire_simulation.cpp`'s wall-damage depletion already uses.
- **Clamp at 1 LSB — smolder NEVER destroys (Erik's call, 2026-07-11):** this pass floors
  `wall_hp` at 1 (one Q16.16 LSB) and structural destruction remains exclusively
  `FireSimulation`'s I>0 path. Emergent rule, by design: a long-smoldered wall survives
  as charred tissue paper at 1 LSB — easy prey for almost any other damage source
  (and for a real flame, whose damage pass CAN still take it to 0 and destroy it).
- **The no-fuel gate moves from `wall_hp ≤ 0` to `wall_hp ≤ FUEL_FLOOR (= 1 LSB)`**
  (combustion.cpp line-62 class): a fully-charred tile's ember goes OUT — no O2 draw, no
  heat deposit — instead of burning its final LSB forever (which would just re-open the
  perpetual-ember hole one LSB lower). Its T then decays via the normal conduction/
  cooling paths, and it can never re-ignite (F ≈ 0 starves any flame the ignition check
  might light).
- **`fuel_per_o2`** — new `[physics.combustion]` dial (§9), Q16.16 at load time,
  default **0.7** (wood stoichiometry burns ≈0.7 mass-units of fuel per unit of O2).
  Physically honest default, but in play it is THE ember-lifetime dial: smaller → embers
  glow for minutes awaiting oxygen; larger → they char out fast.

Expected B4 story after this patch: burn → O2-starve (t≈39) → smolder flicker →
**char-out → quiet** (today's flicker persists indefinitely; timing depends on
`fuel_per_o2`). The O2-differentiation trio timings (172/49/39) WILL move — behavioral
by design, re-measured + perturbation-gated at the patch gate, golden re-baselined ONCE.

*Gates (P5.1):* unit tests — fuel decrement exact/deterministic, the 1-LSB floor never
crossed by this pass, no destruction ever originates from combustion; a **lifecycle E2E**
— ignite → O2-starve → ember persists (T ≥ ignition, I = 0, fuel draining) → O2 inflow
re-ignites a proper flame (I > 0, FLAME-scale consumption resumes) → sealed again →
char-out at the floor, ember extinguishes, wall stands at 1 LSB and one hit destroys it;
O2-differentiation trio re-measured + `test_payoff_orderings_perturbation_robust` green;
suite green; golden re-baseline once with rationale.

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
  **precompute-then-gather** pattern `cuda_water.cu` actually uses — one kernel writes
  per-face flux buffers, a second applies them; round-2 corrected the earlier
  "face-coloring" mislabel).
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
3. **P3 — the compressible solver + atomic consumer AND writer migration.** *Design-gate
   status (2026-07-10, Erik + Fable):* **#0 DONE** — napkin model: the old wave core's
   ~50 explicit substeps/tick (~150 field passes) are REPLACED by ~2-3 substeps (~20-30
   passes) + one S-sweep solve (3·S pass-equivalents) ⇒ at S=8-16 the new solver is
   ~2-3× CHEAPER than what it deletes; the 85%-of-budget scare was numpy overhead ×
   ungained constants. Verdict: the 25% gate is plausibly passed with headroom,
   contingent on the two microbenchmarks below. **#1 DONE** — overflow inequality closed
   with numbers (§3.4; N_FLOOR 10⁻³, ≈12 bits int64 headroom incl. the tank-spike case).
   **#4 DONE** — operator verified against the paper (§3.1: outer per-cell (Nc²), inner
   per-face 1/N̂; one placement CORRECTION applied + the CFL velocity-estimate
   augmentation adopted; paper archived at `docs/papers/ADA492343.pdf`). **#2 and #3
   remain** — the first two commits of P3 itself: microbenchmark (a) per-sweep cost of
   the existing RB-GS at 160² (pins S + confirms #0), (b) real substep counts under
   blast/breach with the augmented estimate (pins N_SUB_MAX).
   **Citation requirement (Erik, 2026-07-10): the solver file carries a header comment
   crediting the technique's authors** — N. Kwatra, J. Su, J.T. Grétarsson, R. Fedkiw,
   "A Method for Avoiding the Acoustic Time Step Restriction in Compressible Flow", J.
   Comput. Phys. 228 (2009) 4146–4161 — and every future file implementing a published
   technique does likewise (project convention).
   Then: true-Kwatra solve replaces `wave_substep`+`diffuse_solve`; compression work
   moves into the substep loop; `u` becomes the one velocity (wind views);
   `atmosphere` re-pointed as the P alias; **readers, in the same patch**:
   `apply_wave_push` → `grad(P)`, water head → integer P (bridge removed), ripple →
   `|P−P_prev|`, recorder update, absorption placement + `test_wave_absorption` rework;
   **writers, in the same patch (v2.1 — round-2 blocker)**: W3 displacement →
   the §2.2 evacuation rule (replaces `physics_engine.cpp:599`), fire plume → a minimal
   plume→T shim (the "pop" never goes inert during P3→P4), `destroy_wall` refill →
   neighbor-mean `N` seeding, FieldEdit `atmosphere`→N-deposit / `wave_source`→T-deposit;
   plus `sink_hop` + `wave_solver.*` deleted, breach→vacuum generalized, GPU backends pinned.
   *Gates:* 6 sub-kernel digests; overflow stress sweep; the quiescent-cold-breach
   native-venting E2E; **a water-rise displacement E2E** (mass conserved at the waterline,
   push visible — the v2.1 occupancy rule under test); §3.3 stress probes;
   behavioral-parity bakes for push/head; **p99 ms/tick ≤ 25 % of the 83 ms budget at
   160² on the dev desktop** (hard number, worst scenario, not mean).
4. **P4 — combustion on real O2.** §5 wholesale (conservative products, floors,
   both thresholds); ignition + fire O2 gate re-pointed to `N_O2`. *Gate:* the four
   emergent payoffs as E2E scenarios (mechanism visible; constants still TBD).
5. **P5 — combined-system bake-off (HUMAN-TEST).** S1–S5 (+ the venting scenario)
   baked on the assembled stack vs the pre-refactor engine; cost table (p99);
   **Erik's eyes are the gate.** First feel-tuning pass of §9 happens here.
   **P5.1 — stoichiometric fuel consumption** (v2.5 amendment, from Erik's P5 review
   2026-07-11): combustion consumes `wall_hp` at ember scale, 1-LSB floor, `fuel_per_o2`
   dial. *Gate:* the v2.5 gate block (lifecycle E2E + trio re-measure + golden once).
6. **P6 — CUDA ports**, one gated sub-patch per kernel, full surface: Helmholtz solve;
   velocity self-advection; T advection + compression work; bulk donor-cell flux
   (precedent `cuda_water.cu`); unified conduction (extend `cuda_temperature.cu`);
   combustion pass; **retire** `cuda_wave.cu`/`cuda_atmosphere.cu`; unpin backends.
   *Gate:* bit-identical digests per kernel (the `cuda-breached` harness), non-negotiable.
   *P6.0 LANDED:* per-kernel unpinning mechanism (`cuda_harness.EOS_P6_PENDING_KERNELS`
   pending set + `cuda_available(kernel=...)`, one key per sub-patch) and the
   `cuda_wave.cu`/`cuda_atmosphere.cu` retirement (files + bindings + CMake + s5/s7
   gates deleted, caller-free verified) — per `docs/eos_p6_gpu_alignment_review.md`
   §2.1 / §1.11 / §4.
7. **P7 — cleanup + canon.** Formal `atmosphere`→`P` deprecation decision; float-mirror
   removal confirmation; stale-doc fixes (interaction map §0's six spots); fold the
   as-built design into engine/04 + 06 chapters and archive the brainstorms (the
   post-EOS consolidation commitment).

Dependencies: P1, P2 independent (parallel worktrees fine). P3 needs both. P4 needs P3.
P5 needs P4. P6 per-kernel after the corresponding CPU code stabilizes (≥ P3). P7 last.

---

## 9. TBD / tuning (feel-gated; first pass at P5)

`γ = 1.4` — adiabatic index, compile-time constant (the ideal-gas identity ρc²=γP).
`K = c_amb²/γ ≈ 64,286` — the ONE unit-bridge constant (v2.2), **wide int64 storage**
(exceeds q16's value range); THE ambient-sound-speed dial: scales ambient c uniformly,
c's T-dependence rides on top. Erik's graceful-degradation fallback lives here (ambient-c
anywhere in the 66→300 continuum is one constant).
`c_amb` — **SET: 300 m/s at ambient (Erik, 2026-07-10).** The shipped `wave_c=66` was never a design
choice, only a performance compromise; under Kwatra `c` costs no substeps (empirically
proven), so the compromise is retired. Honest baseline correction: the OLD engine at the
*desired* c=300 needs ~50 wave substeps ≈ 12 ms — the new solve at S=8 (2.9 ms) is ~4×
cheaper *for the physics Erik actually wanted*, and even S=16 (5.7 ms) is 2× cheaper.
**S: start 8; the P3 gate measures Helmholtz convergence at c=300 and may pin up to 16**
(budget-supported vs the honest baseline), frozen thereafter;
`k_push` + knockdown thresholds (vs the new transient-∇P scale); `k_p` (water head);
air conductivity ("small": big enough that the interface sink fires, small enough that
air doesn't become the rejected heat-advecting field); `cool_shift_vacuum` rate under
the real energy path; combustion `burn_rate / H_fuel / soot_yield / o2_thresh_burn / fuel_per_o2` (v2.5 —
the ember-lifetime dial, default 0.7);
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
   exactly what P5's HUMAN-TEST exists to judge. Two named mechanisms for that checklist
   (round-2): `dyn_wave_absorb` now also locally damps the smoke-carrying wind around
   units (an emergent "bodies slow the breeze" effect — plausibly nice, possibly a
   smoke-hugging artifact); and an empty sealed room's acoustic ringing is damped only by
   numerical diffusion + RB-GS smoothing (likely fine; listen for it).
4. **Combustion cadence** — once-per-tick chosen (matches today); flagged for revisit
   only if P5's fireball feel wants sub-tick burning.
