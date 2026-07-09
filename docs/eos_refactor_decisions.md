# EOS refactor — decisions log & resume notes

> **Status:** living decisions record for the pressure-model refactor — adopt **rung B**
> (genuine compressible ideal gas, `P = C·N·T`). Companions: `eos_research_report.md` (the
> A/B/CFL evidence), `eos_refactor_interaction_map.md` (code-grounded interaction inventory).
> Decisions taken 2026-07-08 → 07-09 (Erik + Claude).
>
> **RESUME PROTOCOL (Erik's request):** next session, *quickly re-confirm the LOCKED decisions
> below* (he was a little distracted when they were made), then continue with the OPEN items.

## LOCKED / working decisions

1. **Adopt rung B** — genuine compressible ideal gas (Kwatra semi-implicit), `P = C·N·T`. Defer is
   off the table; A-vs-B resolved to **B**. Why: real ideal-gas fidelity (rung A's `N` is
   static/decorative — it advects everything *but* N, so `P ∝ T`; rung B advects a genuinely
   *conserved* `N`); **native breach-venting**; real acoustics (feeds sound-ML) + a chemistry
   substrate; determinism path credible (its hardest kernel is the already-proven spike0b GS
   Poisson); Q16.16 range fine; cost only ~1.3× rung A and comfortably realtime (18% of budget).
   *(Quick explicit re-confirm next session.)*
2. **Merge `wave_p` into one unified `P`.** It existed *only* as a numerical workaround — the IMEX
   split ran implicit-diffusion (bulk) and explicit-wave (acoustic) on two fields because those two
   PDEs fight on one field. A compressible solver produces bulk equilibration AND acoustic waves
   from ONE pressure, so the split is obsolete — **the merge is implied by choosing rung B** (rung A
   would keep it hybrid; rung B unifies). The transient-buffet(wave_p)/sustained-dome(atmosphere)
   distinction consumers rely on survives as the single field's *time evolution* (front passes,
   dome lingers) — more physical. **Work item:** rewrite `apply_wave_push` to read `grad(P)` +
   recalibrate `k_push`.
3. **Delete `sink_hop` + the geometric venting hacks.** Venting emerges natively from `−∇P` toward a
   true-vacuum (N=0) cell under rung B. *(Erik's blessing given.)* Also generalize breach→vacuum
   beyond today's edge-hull-only rule.
4. **O2 gates on real `N`**, not `atmosphere`-as-a-pressure-proxy. Combustion reads gas
   quantity/species density, not P.
5. **`P` is materialized once per tick** into a stored Q16.16 field — right after the (N,T) update
   and *before any consumer* — for determinism (all consumers see an identical P) + CUDA
   cleanliness. *(Claude's recommendation; Erik to confirm.)*
6. **Purify the water pressure-head float bridge.** Today it dequantizes `atmosphere+wave_p` → float
   → `×k_p` → requantize. No reason for float (k_p is a fractional coeff → a `mul_q16` by a quantized
   k_p). **Fold into the refactor** (the head term is rewritten to read the derived integer P anyway).
   No urgency (already deterministic via the quantize-pin), but do it — pure integer > pinned bridge.
7. **Temperature: shared conduction, separate energy rules per medium** (SHAPE locked; DETAILS open,
   see OPEN-A). Conduction unifies for free (it's `conductivity`-keyed, no solid/air branch — give air
   conductivity and it conducts via the existing mechanism). Gas `T` needs its *own* energy rules
   (advect + compression work + radiation), NOT the solid path's bit-shift-convert (needs a real
   divide for dynamic N heat-capacity) or its decay-to-ambient (gas energy must conserve, not decay).
   Keep the working solid `temperature`; add gas `T`; couple at interfaces via conduction + the
   one-way `heat` radiation channel. *(Bonus: the old `temperature_design_proposal.md` §5 rejected an
   air-temperature field as "one consumer, not worth it" — the EOS dissolves that; gas T co-derives P.)*
8. **Everything involved is ALREADY Q16.16 fixed-point** (gas, atmosphere, wave, heat, temperature,
   water, fire). The refactor **rearranges** fixed-point fields (N+T→P), it does not convert
   float→fixed. Big de-risk. (A few float bridges remain at boundaries — item 6 is one; purify where
   the refactor touches them.)
9. **Process = the design-gate practice.** design doc → independent adversarial critique (distinct
   lenses) → resolve blockers on paper (iterate v2/v3) → patch plan → autonomous build (**CPU + CUDA
   in lockstep** — both wave & diffusion already have bit-identical GPU mirrors, so it's a double
   implementation) with the determinism/bit-identity gates. **Nothing is built until the doc survives
   critique.**

## OPEN — to decide next (needs Erik's loop-time)

- **A. Temperature medium-split DETAILS** — the exact gas-`T` energy discretization (the divide for
  dynamic-N heat capacity; the compression-work term; interface conduction/radiation coupling). Walk
  carefully.
- **B. What `N` actually is + multi-gas / chemistry — a DEDICATED design session** (Erik's request).
  Investigate: per-gas molar mass + gas constants, individual species tracking, emergent chemistry
  (two gases mix → explode, etc.). **Claude's complexity read:** the multi-gas *core largely FALLS
  OUT* of existing foundations (gas is already `(N_gases,h,w)` Q16.16; Dalton `P = C·T·ΣN_i` is a sum;
  molar mass is a new table column) — a *design* task, not research. Chemistry = a cheap emergent
  tier (threshold reactions on per-gas density + T — falls out) + a deep realistic-kinetics tier
  (a lit search *if* we want that — decide within the session). Sub-question: does bulk air become an
  explicit N-species, or stay implicit? **Parked post-birthday.**
- **C. Remaining `§D` items** in the interaction map not yet locked (P-materialization contract
  details; whether to add an artificial acoustic-damping-for-feel term; etc.).

## Resume plan (next session)

1. **Quickly re-confirm LOCKED 1–9** (Erik wanted a brief review — he was distracted).
2. Walk **OPEN-A** (temperature details), then schedule **OPEN-B** (the N / multi-gas dedicated session).
3. Then: full **design doc → adversarial critique → resolve → patch plan → CPU+CUDA build**.
4. Housekeeping: consolidate EOS design docs onto the refactor branch when cut; **delete dead
   `wave_solver.*`**; fix the 6 stale doc spots (interaction map §0).
