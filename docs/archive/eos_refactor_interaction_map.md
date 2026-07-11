# EOS refactor — pressure-interaction map (design-prep)

> **Status:** analysis / design-prep, NOT canon. Foundation for the human+Claude design
> session that gates the EOS (rung-B) refactor. Synthesized 2026-07-08 from five grounded,
> code-cited subsystem audits (atmosphere/wave, smoke/gas, water, fire/temperature/heat,
> gameplay-exchange). The reframe: two primary fields — gas **T** and particle density **N** —
> derive pressure **P = C·N·T**; everything downstream *reads* the derived P; everything
> upstream *feeds N or T*. Companion: `docs/eos_research_report.md`, `..._brief.md`,
> `blackbody_smoke_and_rendering_brainstorm.md` §0.

---

## 0. Staleness verdict (Erik: "I think the docs aren't stale — but double-check")

**Mostly right — the interaction map and chapters are substantively accurate, with pinpoint drift.**
`physics_field_interaction_map.md` (2026-06-14, "built from code") is still a faithful read/write
map; its main drift is **attribution** (some Python→C++ moves). Per-chapter, the fixes to make
before/with the refactor:

| Doc | Drift |
|---|---|
| engine/04 atmosphere | §2.5 "orchestration" **stale** — it's one `run_substeps` C++ call now, not the shown Python loop; §5/§6 forward-list stale (water coupling, CUDA S5/S7, and the `wave_p` unit-push all shipped but listed unbuilt); silent on fixed-point entirely. |
| engine/05 smoke | **dtype stale** — says `float32`, code is Q16.16 int; §2 advection form stale (says gradient-dot, code is semi-Lagrangian). Status section is accurate. |
| engine/06 temp/fire | The ambient-**cooling pass ("STEP C")** is *absent from the chapter entirely* (lives only in `temperature_design_proposal.md` + code). Minor: furniture missing; "cross-machine test pending" now done (`cuda-breached`). |
| engine/07 water | Mostly current; **`ceiling_h=2.5` value undocumented**; §7 reads like a pre-port TODO — trust §8 only. |
| engine/13 field_edit | `gas` omitted from the §3 policy table (though live); stale "per-gas pending" comment. |
| mechanics/05 exchange | Blast-damage row labelled as a field read; it is **geometric** (no field). |

`wave_solver.cpp/.h` is **dead code** — no caller, no binding, superseded by `AtmosphereSolver`. Delete during the refactor.

---

## A. DOWNSTREAM — everything that READS pressure today (can it ride derived P?)

| Consumer | Reads | Rides derived P? |
|---|---|---|
| **wind** `= −½·grad(atmosphere + wave_p)` → smoke advection, fire fan/strip | atmosphere, wave_p | **Yes**, if P is materialized *before* the wind-gradient step (a scheduling constraint). |
| **water pressure-head (W4)** `surface += k_p·(atmosphere+wave_p)` | atmosphere, wave_p (const float bridge) | **Yes — near-drop-in**: `water_solver.cpp` needs *zero* changes if the existing float bridge instead writes `C·N·T`. But `k_p` is calibrated to today's scale → **recalibrate**. |
| **find_burst_walls** (wall bursts on pressure spread) | atmosphere spread vs `burst_threshold` | **Yes** — reads a derived P differential fine. |
| **apply_wave_push** (units — impulse + knockdown) | `grad(wave_p)` per footprint | **Needs a decision** — see §D-1 (transient vs dome) + `k_push` recalibration. One hardcoded line (`gmap.wave_p`) to repoint. |
| **apply_temperature_ignition** / **fire O2 gate** | mean `atmosphere` as an **O2 proxy** | **Does NOT map cleanly** — see §D-4. Should gate on `N` (gas quantity/species), not P. |
| temperature cooling (vacuum-exposure) | `atmosphere < thresh` (incidental) | Yes — read-only compare, works on derived P. |
| EnvironmentProfile.pressure_min/max | (unbuilt stub) | N/A yet. |

**Takeaway:** the *bulk* of downstream (wind, water, burst) rides a derived P unchanged — the low-risk
half of your dream holds. The two that need real thought are the **unit impulse-push** (transient vs
dome) and the **O2 proxy** (should become N/species, not P).

## B. UPSTREAM — everything that FEEDS pressure today (remap to N or T)

| Writer | Today | Remaps to |
|---|---|---|
| **wave→atmosphere transfer** (`wave_substep`) | one-way, DC-neutral deposit of `(wave_p−mean_wp)` into atmosphere | The acoustic↔bulk coupling. **Decision** (§D-1): does acoustic become a real N/T perturbation, or stay a summed overlay? A conservative ±-pair version existed and was *deliberately reverted* ("Erik's call"). |
| **fire own-tile plume** (`fire_simulation.cpp:206`) | `atmosphere += fire_pressure_gain·I·sat·dt` | **feed-T** — combustion heat; the "pop" becomes `P=C·N·T` rising off the T spike. Clean. |
| **apply_explosion** (`physics.py`) | atmosphere disc-add + wave_source add (+fire +smoke) | **feed-T (energy) + feed-N (gas)** — a heat/energy dump, not phantom pressure. Clean in principle. |
| **water W3 displacement** (`physics_engine.cpp:599`) | `atmosphere *= ratio` (rising water shrinks free-air column) | **feed-N** — less volume ⇒ higher density ⇒ higher P. Clean. |
| **destroy_wall / stamp refill** | atmosphere `= neighbour_mean` on every destruction | **N redistribution** — neighbour-mean refill (never a hard-zero pulse). |
| **breach → vacuum** | `is_vacuum=True` **only for edge-hull** destruction; interior destruction just refills | **N drains to 0 at the breach**; under rung B, venting *emerges from −∇P* (see §D-3). Note the edge-hull-specificity — the EOS would generalize it. |

**The reframe simplifies this:** "feeding pressure" stops existing — every site above becomes a
**heat/energy feed (T)** or a **gas-mass feed (N)**, and P follows from physics. No more phantom
pressure injection. **Three independent writers (explosion, fire, water) + the wave transfer** — a
wider set than the two we'd flagged, but all map to N or T except the acoustic-coupling decision.

## C. Temperature today, and the "unify solid `temperature` with gas `T`?" answer

**Today = two quantities, never merged:** `heat` (per-tick radiation deposit, Q16.16, cleared each
tick, one-way) and `temperature` (persistent, **solids only**: convert via bit-shift by
`log2(thermal_mass)` → conduct via `face_shift` → cool toward ambient=0). Air has no temperature by
foundational design (engine/06 §1).

**The unify verdict (grounded):**
- **Conduction unifies for free** — the conduct pass has no solid/air branch; it's keyed on
  `conductivity==0`. Give air a nonzero conductivity and it conducts through the *existing* mechanism.
- **Conversion does NOT carry over** — the bit-shift trick needs `thermal_mass` as a fixed power-of-two
  constant; gas "mass" under EOS is `N` (dynamic, per-tile) → needs a real divide / different
  discretization.
- **Cooling is physically WRONG for gas** — solids decay toward ambient=0 (implicit external sink);
  gas energy should *conserve/advect/expand*, not exponentially decay.
- **⇒ shared conduction, separate energy rules per medium.** (Erik's "different rules for air vs
  massive materials" instinct, made precise.) Recommend: **keep the working solid path as-is**, give
  gas `T` its own EOS energy rules, couple at interfaces via conduction (already conductivity-keyed) +
  radiation (the existing `heat` channel, one-way per the blackbody decision).
- **Bonus:** `temperature_design_proposal.md` §5 *rejected* an air-temperature field because it would
  serve only one consumer (pressure) — the EOS makes gas `T` central (it co-derives P), so **that
  rejection is dissolved; the reframe resolves a tension the design already had.**
- Caveat: `furniture` (permeability 0.5) is already neither clean-solid nor gas — a live precedent that the solid/air split is coarse; weigh it when designing the medium switch.

---

## D. The design decisions / risk list (the load-bearing part — bring to the session)

1. **`wave_p` (transient acoustic) vs `atmosphere` (sustained dome) — merge or hybrid?** These are two
   *qualitatively different* signals with different unit behaviors (buffet-that-cancels vs sustained
   throw), and the wave→atmosphere transfer is **one-way by deliberate design**. Even if both collapse
   into one `P=C·N·T`, consumers (esp. `apply_wave_push`) still need the transient-vs-bulk distinction.
   **Decision:** does the acoustic become a genuine N/T perturbation (real, unifies, feeds sound-ML) or
   stay a bolted-on overlay summed at the wind step (as today)? *(Erik's call — the biggest architectural one.)*
2. **Gas as *conserved* N is a real fight, not a coefficient.** Smoke's semi-Lagrangian advection is
   **deliberately non-conservative** (per-step truncation decay, "accepted Q-S2-1"). A conserved mass
   field can't silently decay. This is the single largest structural change to make `N` real.
3. **Breach venting should become native (delete `sink_hop`).** Today venting is a **geometric BFS
   hack** (`sink_hop`, K passes/tick) precisely *because* the current wind dies as pressure equalizes.
   Rung B's real `−∇P` to a true-vacuum (N=0) cell vents natively — retiring the hack (a genuine
   improvement + your signature decompression mechanic done right). Also generalize breach→vacuum
   beyond the current edge-hull-only rule.
4. **O2 is a bulk-pressure proxy today.** `apply_temperature_ignition` and the fire O2 gate read
   `atmosphere` as a stand-in for oxygen *quantity*. Under EOS, real `N` (and eventually per-species
   fractions) makes O2 a genuine density read — combustion should gate on **N, not P**. A
   "doesn't-map-cleanly" item *and* an opportunity (real oxygen, real chemistry hook).
5. **Materialization scheduling.** P becomes *derived* (computed on demand). Decide **where/when**
   `P=C·N·T` is materialized each tick so every reader (wind, water head, burst, unit push) sees a
   consistent P at the right tick-point. A scheduling contract, not a solver change.
6. **Determinism/CUDA = a *double* implementation.** wave+diffusion already have bit-identical CUDA
   mirrors (S5/S7). Any EOS redesign of these equations must be built **twice (CPU + CUDA in lockstep)**
   — a larger migration surface than the docs imply. Upside: the hard kernel (RB-GS Poisson) is already
   proven bit-identical (spike0b/S7), and Q16.16 range is fine (Erik confirmed).
7. **Inherited float bridges.** `atm_f_`/`wave_p_f_` float mirrors are rebuilt every tick for water's
   still-float head term — an existing purity boundary to either collapse (full integer) or preserve.

## E. What's free vs what's a fight (the one-glance summary)

- **Free / near-free:** downstream wind, water head, burst all ride derived P (scheduling only);
  conduction unifies by giving air conductivity; per-gas slice structure already `(N,h,w)` and
  Dalton-summable; the design philosophy *welcomes* gas T (old rejection dissolved).
- **Real work:** making gas the conserved `N` (kill the non-conservative decay); gas energy rules
  (advect + compression work, no ambient decay); the acoustic merge-or-hybrid decision; native breach
  venting (delete `sink_hop`); recalibrating `k_p`/`k_push`; O2 → N/species; and the CPU+CUDA double build.
- **Cleanup riders:** delete dead `wave_solver.*`; fix the six stale doc spots in §0.

---

*Next: this map + the design decisions in §D feed the design-doc draft. Erik locks the §D calls
(one at a time), then adversarial critique → resolve on paper → patch plan → autonomous build with
the determinism gates. Per the design-gate practice: nothing is built until the design doc survives critique.*
