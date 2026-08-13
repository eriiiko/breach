# EOS / ideal-gas reframe — research report (Phase 1.1 output)

> **Provenance & status (read this first).** Input brief: `docs/eos_research_brief.md`.
> The deep-research harness (run `wf_e4895702-568`, 2026-07-08) completed its expensive
> phases — 5 search angles → 23 primary sources → 112 extracted claims → top-25 selected —
> but **rate/credit limits killed most of the adversarial-verification votes and the automated
> synthesis**. Crucially: of the 25 claims sent to verification, **2 were CONFIRMED, 0 were
> REFUTED, and 23 errored** (infrastructure, not evidence). This report is the **synthesis I
> (Opus) completed in-thread** from the cached primary-source claims + the brief, with every
> load-bearing statement confidence-labelled and the CFL arithmetic recomputed from scratch.
> **Not canon** — it is the evidence base for the Phase-1.3 adopt/defer decision.
>
> Confidence key:
> - **[✓ HARNESS]** — passed the adversarial 3-vote (or 2-vote) verification.
> - **[✓ LIT]** — extracted from a primary source and consistent with the established literature
>   (Opus-checked); harness vote errored on limits, not on content.
> - **[⚠ GAP]** — the search did **not** find supporting evidence, or the claim is unproven at
>   Breach's regime. These are the honest holes the Phase-1.2 prototype must fill.

---

## 0. Headline

**The acoustic-CFL problem — the brief's #1 fear (~250 explicit substeps/tick from a 2500 K
core) — is a known, *solved* problem in graphics, with three independent escape routes, each
with published or shipped-production precedent. It is not a blocker for any rung.** The decision
therefore is *not* "can we afford it" but "how much real compressible physics (momentum-driven
curl, shock steepening) is worth the extra machinery, given the visual payoff at Breach's coarse
grid is the one thing nobody has measured."

The three escape routes, cheapest → richest:

1. **Prescribed-divergence incompressible (Feldman–O'Brien 2003).** No acoustics in the solver at
   all — explosion expansion is a source term `∇·u = φ` on the RHS of an incompressible pressure
   solve. Zero acoustic substeps *by construction*. **This is what Autodesk Bifrost ships in
   production.** → **Rung A.**
2. **Semi-implicit compressible (Kwatra 2009).** Real momentum + compressibility, but the acoustic
   part is solved implicitly, so the timestep is limited by `|u|` not `|u|+c`. The implicit solve
   *is a Poisson equation* → reuses Breach's RB-GS kernel. → **Rung B.**
3. **Reduced/capped sound speed (RSST, Hotta 2012; weakly-compressible).** Keep everything
   explicit, just dial `c` down by a constant. Directly answers "is c a tunable knob": **yes, it's
   a published standard technique.** → the cheap dial-based fallback.

---

## 1. Scored comparison table

Weights per brief §6: realtime **30%**, visuals-at-coarse-grids **25%**, stability **20%**,
integration-fit **15%**, determinism-fit **10%**. Hard gates: **realtime** and **stability** (a
scheme scoring ≤2 on either is out regardless of total). Scores are 1–5, my judgement, evidence
cited in §7/§8.

| Scheme | Realtime ×.30 | Visual@coarse ×.25 | Stability ×.20 | Integration ×.15 | Determinism ×.10 | **Total** | Gate |
|---|---|---|---|---|---|---|---|
| **Feldman–O'Brien divergence-controlled** (rung A) | 5 | 4 | 5 | 5 | 4 | **4.65** | ✅ |
| **Kwatra semi-implicit compressible** (rung B) | 4 | 4 ⚠ | 4 | 5 | 4 | **4.15** | ✅ |
| Stable-Fluids-compressible *(control)* | 5 | 2 | 5 | 5 | 4 | **4.15** | ✅ |
| RSST / weakly-compressible capped-c | 4 | 3 | 3 | 3 | 5 | **3.50** | ✅ |
| Explicit Euler + artificial viscosity + positivity | 2 | 4 | 3 | 3 | 4 | **3.05** | ⚠ realtime |
| Thermal Lattice-Boltzmann | 4 | 2 | 2 | 2 | 3 | **2.70** | ❌ stability |

The table tells one clean story: the **control (semi-Lagrangian Stable-Fluids) is cheap, stable,
already-built — and visually the thing to beat** (super-diffusive → dissolves curl to grey mush,
scores 2 on visuals). **Feldman–O'Brien wins because it is nearly as cheap/stable/integrated as the
control but carries production-grade fireballs.** Kwatra ties the control on total but for the
opposite reason — it *maxes the visual ceiling* (real baroclinic curl + shocks) at some realtime
and (unproven) coarse-grid-payoff cost.

---

## 2. Recommendation, per rung

### Rung A → **Feldman–O'Brien prescribed-divergence incompressible**, EOS-driven à la Bifrost.
- **Why:** production-proven (Bifrost); zero acoustic substeps; reuses Breach's incompressible-style
  pressure-Poisson (RB-GS); and the divergence source is derived *exactly the way Breach's reframe
  is built* — `∇·u = −(1/ρ)Dρ/Dt` with ρ from the ideal gas law at held pressure **[✓ LIT]**, i.e.
  `P = C·N·T` drives the expansion. Erik's "everything downstream stays the same" intuition is
  literally this scheme.
- **Runner-up:** RSST / capped-c weakly-compressible — if you want *some* real momentum/curl without
  a Poisson solve at all, at the cost of distorted acoustics.
- **The hybrid (brief Q2) is native here:** Feldman **decouples fireball-expansion from blast-wave
  impulse by construction** — they model detonation as a scheduled `φ` and *ignore the wave as
  "largely invisible"* **[✓ LIT]**. So Breach **keeps its tuned `wave_p` as the separate blast-impulse
  channel** and adds a divergence-driven thermal-expansion fireball on top. Do **not** collapse
  atmosphere+wave_p for rung A.

### Rung B → **Kwatra semi-implicit compressible** (the confirmed scheme).
- **Why:** the *only* route that gives genuine momentum-carrying compressibility — real baroclinic
  vorticity (mushroom caps that are physics, not curl-noise garnish) and shock steepening —
  **without** the acoustic-CFL penalty **[✓ HARNESS]**, on a solve that **reduces to the
  incompressible Poisson Breach already runs** **[✓ HARNESS]**. This is the "one level up = could
  make money" ambition realized on infrastructure that already exists and is already CUDA-bit-identical.
- **Runner-up:** explicit compressible Euler + a **Zhang–Shu positivity-preserving limiter** — the
  honest-shocks path if the implicit solve proves troublesome, but it re-incurs substeps and flirts
  with the realtime gate.
- **For rung B, atmosphere+wave_p *can* unify** into one `(N,u,T)` solver spanning both regimes
  (Kwatra → incompressible Poisson as `c→∞` **[✓ HARNESS]**). That unification is a *reward* of rung
  B, not a prerequisite.

### What would flip A→B or B→A (the decision hinges on ONE measurement)
**[⚠ GAP] The single biggest evidence hole: nobody has shown baroclinic curl / mushroom caps /
the cooling arc at Breach's ≤256² 2D resolution.** The Kwatra demos are offline **3D at 1.5M–67M
cells taking 30 min–hours** **[✓ LIT]**; Feldman's are 3D ~55–84k cells but particle-based. So:
- If, at ≤256², rung B's real curl **visibly** beats rung A's expansion-only fireball → **go B.**
- If the delta is marginal at that coarseness (semi-Lagrangian advection may smear it out anyway) →
  **rung A wins outright** (cheaper, simpler, production-proven).
- **This is exactly what the Phase-1.2 A/B/current GIF bake-off must settle by eye** — the report
  cannot, and honest scoring says it should not pretend to.

---

## 3. Worked CFL / substep math (recomputed from scratch — brief Q1)

Constants: tile `dx = 1/3 m ≈ 0.333 m`; tick `Δt_tick = 83 ms`; air `γ = 1.4`, so
`c = 20.05·√T` m/s.

| Regime | Sound speed c | Explicit acoustic dt = dx/(|u|+c) | **Substeps / 83 ms tick** |
|---|---|---|---|
| Ambient 290 K | 341 m/s | 0.98 ms | **~85** |
| Hot core 2500 K | 20.05·√2500 = **1002 m/s** | 0.33 ms | **~250** |

Both reproduce the brief's figures independently — the brief's arithmetic is **correct**.
Now the three escapes at the hot core:

- **Kwatra semi-implicit** — dt limited by `|u|` only. Cap game outflow at `|u| ≤ 100 m/s` ⇒
  dt = 3.3 ms ⇒ **~25 substeps**, *each doing one RB-GS Poisson solve*. Net vs explicit:
  a corroborating multiphase semi-implicit paper measured **+12% cost/substep but 4.3× larger dt ⇒
  ~4× net win** **[✓ LIT]**; the graphics regime (larger c/|u|) wins more (~10×). **250 → ~25.**
- **Feldman–O'Brien** — **no acoustic substeps at all**: one (or a few) pressure solve per tick,
  advection-limited. **250 → ~1–4.** Cheapest.
- **RSST capped-c** — set `c_max = 100 m/s` (a dial): even the hot core gives dt = 3.3 ms ⇒
  **~25 fully-explicit substeps, no Poisson solve.** The cost is that acoustics/shocks now travel
  at the capped speed (fine for a game; wrong for physics).

**Answer to Q1 (is c a tunable dial?): yes, unambiguously** — RSST does it explicitly by construction
**[✓ LIT]**, and Kwatra is stable all the way to `c→∞` **[✓ LIT]**. Both directions are safe.

---

## 4. Prototype pseudocode (numpy-shaped, Breach field names)

### Rung A — Feldman–O'Brien / Bifrost divergence-controlled (per tick)
```python
# state: N_i (per-gas density), T (shared gas temp), u (vx,vy), P derived
# sources: combustion, breach(vacuum), door(topology)

# 1. combustion source terms (structure from Feldman Eq.8-10; constants tuned by eye)
burn      = clip(fuel_gas, 0, k_burn*dt) * (T > T_ignite)     # fuel fraction consumed
fuel_gas -= burn
heat_dep += burn * H_fuel                # -> existing Q16.16 heat channel (one-way, decided)
N_soot   += burn * soot_yield
T        += burn * H_fuel / (c_v * N_total)                   # local temperature spike

# 2. thermal-expansion divergence source (the EOS coupling; Bifrost recipe)
#    div_target = -(1/rho) D rho/Dt, with rho from P=C*N*T at HELD pressure
rho          = M_air * N_total
div_target   = -(1.0/rho) * material_deriv(rho, u, dt)       # thermal expand -> outflow
div_target  += detonation_phi(t)          # scheduled blast pulse: peak->decay->slight-neg

# 3. incompressible pressure solve with prescribed divergence (REUSE RB-GS kernel)
#    solve  laplacian(p) = (rho/dt)*(div(u*) - div_target)   [fixed sweep count!]
p  = rb_gauss_seidel(rhs=(rho/dt)*(divergence(u_star) - div_target),
                     sweeps=FIXED_N,      # <- determinism: NOT a convergence tol
                     obstacles=solid, vacuum=is_vacuum)
u  = u_star - (dt/rho) * grad(p)

# 4. advect N_i, T on u (existing semi-Lagrangian); wave_p stays SEPARATE (blast impulse)
N_i = advect_semilagrangian(N_i, u, dt)
T   = advect_semilagrangian(T,   u, dt)
# blast impulse on units still rides the existing tuned wave_p channel (hybrid)
```

### Rung B — Kwatra semi-implicit compressible (per tick)
```python
# 1. EXPLICIT advection of conserved (N_i, N*u, N*E) at the |u|-limited dt (NOT |u|+c)
#    ~25 substeps at hot core instead of ~250
for _ in range(n_advect):                 # n_advect from |u| CFL only
    N_i, mom, E = advect_conservative(N_i, mom, E, dt_adv)

# 2. IMPLICIT acoustic solve -> Helmholtz/Poisson (REUSE RB-GS, fixed sweeps)
#    (I - dt^2 * div( (c^2/rho) grad )) p = rhs      ->  identity + Poisson term
#    reduces to incompressible Poisson as c->inf  [HARNESS-CONFIRMED]
p = rb_gauss_seidel_helmholtz(rhs, c2_over_rho, sweeps=FIXED_N,
                              obstacles=solid, vacuum=is_vacuum)
mom -= dt * grad(p)                        # acoustic velocity correction
E   -= dt * div(p * u)                     # energy update
# 3. derive P = C * N_total * T for the emission LUT (blackbody) + gameplay crossings
T   = temperature_from(E, N_total, u)
P   = C * N_total * T
```

Both feed the **one-way heat channel** and the **thin authoritative crossings** (ignition flag,
`wave_p` impulse, HP delta) exactly as the brief's fixed constraints require; the `(N,u,T,P)` fluid
itself stays **float/local/non-authoritative**.

---

## 5. Risk register (top 5)

| # | Risk | Likelihood | Mitigation |
|---|---|---|---|
| 1 | **Rung-B curl doesn't read at ≤256²** — the whole reason to pay for B evaporates. **[⚠ GAP]** | Medium | The Phase-1.2 GIF bake-off *is* the test. If marginal, ship rung A. Do NOT commit to B before this measurement. |
| 2 | **Hot-core CFL still bites** if `|u|` (blast outflow) itself approaches c. | Medium | Cap `|u|` (a game dial), and/or cap `c` (RSST). Both are precedented, both safe. |
| 3 | **Positivity blow-up** — extreme single-cell energy dumps drive density/pressure negative → NaN. | Medium-High | Feldman/Kwatra largely dodge it (incompressible / conservative-implicit); if explicit Euler is used, the **Zhang–Shu positivity-preserving limiter** is the canonical fix **[✓ LIT]**. |
| 4 | **Determinism trap: adaptive Poisson iteration counts** break same-box replay (brief Q10). | Low (if disciplined) | **Fixed sweep count, never a convergence tolerance** — Breach's RB-GS already does this. Bake it into the spec as a hard rule. |
| 5 | **Scope creep: unifying atmosphere+wave_p** turns a prototype into a rewrite. | Medium | Keep them **separate** for rung A (Feldman decouples them anyway). Unification is a *rung-B reward*, deferred until B is chosen and working. |

---

## 6. Brief Q1–Q10, answered explicitly

- **Q1 — c as a tunable dial?** **Yes.** RSST reduces c by a free constant by design **[✓ LIT]**;
  Kwatra is stable to `c→∞` **[✓ LIT]**. Both directions safe.
- **Q2 — atmosphere+wave_p: unify or hybrid?** **Rung A: keep hybrid** (Feldman decouples
  expansion from blast wave **[✓ LIT]**). **Rung B: may unify** (Kwatra→incompressible Poisson as
  `c→∞` **[✓ HARNESS]**) — a reward, not a requirement.
- **Q3 — combustion source-term numbers?** Structure, not physical constants: fuel burns above
  `T_ignite` at rate `z`, releasing heat `Ḣ=b_h·z`, gas volume `δφ=b_g·z/V`, soot `s=b_s·z`
  **[✓ LIT, Feldman Eq.8–10]** — *the paper says the constants were chosen for looks*, so mine the
  **structure**, tune the numbers by eye. Kwatra's abuse benchmark deposits **10×T_atm (2900 K) +
  1000×p_atm** in one cell and survives **[✓ LIT]** — direct evidence Breach's single-tile dumps are safe.
- **Q4 — stability under abuse?** Kwatra's paper passes strong shock tube (10¹⁰:0.1 pressure jump),
  Mach 240, Woodward–Colella interacting blasts with solid walls, near-vacuum rarefactions
  **[✓ LIT]**; incompressible schemes (Feldman) sidestep the failure mode entirely. Explicit Euler
  needs the positivity limiter (risk #3).
- **Q5 — contact vs radiation ignition (precedent)?** **Split, and it validates Breach's default.**
  High-end production (Bifrost TOG-2022) uses a **P-1 radiative model to ignite fuel at a distance
  *without heating the intervening air*** **[✓ LIT]** — exactly Breach's radiation-only heat
  channel. The *cheap game* path (Noita) uses pure **contact** ignition (burning pixel ignites a
  flammable neighbour) **[✓ LIT]**. Breach already owns the ray/heat channel → **stay radiation-only**;
  it has the better physical pedigree.
- **Q6 — tracers become conserved mass?** The move is exactly Bifrost's: density from the ideal gas
  law drives a divergence/continuity source **[✓ LIT]**. Main new obligation: momentum-at-emission
  (a flamethrower nozzle injects `N·u`, not just `N`) and genuine mass conservation in sealed rooms
  (a fixed-sweep solve conserves to solver tolerance).
- **Q7 — vacuum-breach BC + does EOS fix the lingering-haze venting weakness?** **Plausibly yes,
  and this is a real upgrade.** Today's haze lingers because wind dies as interior pressure → 0 (the
  gradient vanishes). Under `P=C·N·T`, a breach drains **N to true vacuum** and `−∇P` *is* the
  sustained outrush — the venting becomes physics, not the current sponge/sink-hop hack. **[⚠ GAP:
  no source measured this for a 2D game specifically; flag as a prototype success-criterion, not a
  guarantee.]**
- **Q8 — CUDA shape + ×K for 2.5D?** Both recommendations' hot loop is the **RB-GS Poisson/Helmholtz
  solve Breach already ported bit-identically** + semi-Lagrangian advect. Per-cell, stencil-local,
  atomics-free → the existing kernel patterns carry over. 2.5D = the same solver ×K layers +
  a thin per-column vertical exchange, cost ≈ ×K **[✓ LIT — Feldman/Kwatra are already the
  3D generalisation]**.
- **Q9 — coarse-grid visual evidence?** **The central gap [⚠ GAP].** All primary demos are offline
  3D at ≥1.5M cells (Kwatra) or particle-based (Feldman). No ≤256² 2D real-time evidence exists →
  Phase-1.2 must generate it.
- **Q10 — scheme-internal nondeterminism?** The one real trap is **adaptive iteration counts /
  convergence-tolerance Poisson solves**. Mitigation is a hard rule (risk #4): fixed sweeps only.
  Fully-explicit schemes (RSST, explicit Euler) have no iterative solve → cleanest on this axis
  (why RSST scores 5 on determinism-fit).

---

## 7. Annotated reading list (the ~10 that matter)

1. **Kwatra, Su, Grétarsson, Fedkiw 2009 — "A Method for Avoiding the Acoustic Time-Step Restriction
   in Compressible Flow"** (JCP 228). *The rung-B scheme.* Semi-implicit split; Poisson-with-identity
   solve; `|u|` not `|u|+c`. `apps.dtic.mil/sti/pdfs/ADA492343.pdf` **[✓ HARNESS source]**.
2. **Feldman, O'Brien, Arikan 2003 — "Animating Suspended Particle Explosions"** (SIGGRAPH). *The
   rung-A scheme.* Prescribed-divergence incompressible; detonation as scheduled `φ`; source-term
   structure. `graphics.berkeley.edu/papers/Feldman-ASP-2003-08/`.
3. **Bifrost combustion — TOG 2022 (`dl.acm.org/doi/10.1145/3526213`) + SIGGRAPH-2019 talk
   (`…/3306307.3328149`).** *Production precedent* that fireballs/subsonic explosions ship WITHOUT a
   compressible solver, with **P-1 radiative ignition-at-a-distance** — validates rung A *and*
   Breach's radiation-only heat.
4. **Nguyen, Fedkiw, Jensen 2002 — "Physically Based Modeling and Animation of Fire."** Explicitly
   *rejects* compressible flow for the acoustic-CFL reason and uses incompressible + a thin reaction
   front — the canonical "route around the CFL" precedent. `physbam.stanford.edu/papers/stanford2002-02.pdf`.
5. **Hotta et al. 2012 — Reduced Speed of Sound Technique (RSST)** (A&A). *The "c is a dial" proof*:
   artificially lower c, stay explicit + local. `aanda.org/…/aa18268-11`.
6. **Zhang & Shu 2011 — positivity-preserving / maximum-principle limiters** (Proc. R. Soc. A). The
   stability-gate answer for any explicit-Euler path: keeps density/pressure positive under blast +
   vacuum. `royalsocietypublishing.org/doi/10.1098/rspa.2011.0153`.
7. **Semi-implicit pressure-correction (multiphase), OSTI 1765291.** Independent corroboration of the
   CFL removal with hard numbers: +12% cost/step, 4.3× dt, ~4× net. `osti.gov/servlets/purl/1765291`.
8. **Bridson — *Fluid Simulation for Computer Graphics* (2nd ed.).** The baseline every candidate is
   scored against; semi-Lagrangian, projection, curl-noise, MacCormack.
9. **Noita — GDC "Exploring the Tech and Design"** (Purho). Shipped falling-everything precedent;
   pure contact ignition; the Q5 counter-example. `gdcvault.com/play/1025695`.
10. **GPU Gems 3 ch.30 — "Real-Time Simulation and Rendering of 3D Fluids."** The realtime-GPU-fluid
    baseline (grid sizes, kernel shapes) to sanity-check §Q8 cost claims.

---

## 8. The one-paragraph brief for Phase 1.3

Build **both** prototypes in Phase 1.2 (float numpy, real ship geometry — the S1–S4 scenarios):
**rung A = Feldman–O'Brien divergence-controlled with `P=C·N·T` driving the expansion and `wave_p`
kept as the separate blast channel**; **rung B = Kwatra semi-implicit** on the RB-GS kernel. Bake
the A / B / current-engine GIFs on the *same* seeds. The decision reduces to **one measurement**:
does rung B's real baroclinic curl visibly beat rung A's expansion-only fireball at ≤256²? If yes,
B is the "could-make-money" ceiling and it runs on kernels you already own bit-identically. If the
delta is marginal at that coarseness, A wins — production-proven, cheapest, and already the shape of
your instinct. Either way the acoustic-CFL fear is retired, determinism stays intact (fixed-sweep
Poisson, fluid render-only), and the heat channel stays the one-way absorber you decided on.

---

**Appended 2026-08-14 (supersession note).** Any `T + 290`-only EOS ambient
description above is superseded by the unified canonical map in
`[physics.temperature_scale]`: the sim-wide Kelvin map is now `K = 293 +
3·T_game`, with the EOS pressure calibration keeping a named, deliberate
exception at `eos_t_amb_k = 290` (unchanged value, now a documented exception
rather than the only convention). See
`docs/temperature_scale_unification_design_2026-08-13.md`.
