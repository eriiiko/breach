# S2 — Smoke/Gas Advection in Q16.16 — deterministic conservative advection research

**Status:** research synthesis, informing the S2b migration step. Sibling to
`docs/s1_water_fixed_point_plan.md`. Synthesizes 5 research briefs (conservative-SL / flux-form /
error-compensated / fixed-point-determinism-fit / internal-cost-and-feel) into one recommendation.
Awaiting Erik's decision (§9) before any implementation — per the "discuss approach first" rule.

**Depends on:** `docs/fixed_point_migration_plan.md` (§2.3 R2-CONS edge-flux idiom, §4 reductions,
§5.2 smoke transcendental/divide audit, §6.3 S2b), `docs/s1_water_fixed_point_plan.md` (the shipped
conservative-flux template), `docs/architecture/engine/05_smoke.md` (smoke v2 design rationale).

**Implementation status:** none. This is the language-first design pass for S2b.

---

## 0. TL;DR for the impatient

**Switch smoke + the 5 gas planes from semi-Lagrangian back-trace to a conservative flux-form scheme.**
Do not try to make the current back-trace conserve to the LSB — the only cheap fix is global (a forbidden
cross-GPU reduction), and every local fix re-derives a flux scheme while keeping the back-trace's
determinism baggage (per-cell `sqrt`, three rounding modes, the `acc/wsum` divide).

The flux switch is the **same conservation skeleton S1 water already shipped** — gather each face flux
once as a wide `int64`, apply `±` the identical value to the two cells, narrow at one shared point →
conserved to the LSB by construction, no global reduction, no per-cell transcendental.

The one real decision is **how much crispness you want back** (§9):
- **Default: donor-cell upwind + a divide-free two-slope MC/minmod limiter** — conservative, deterministic,
  second-order in smooth air (≈ visual parity with today), zero per-cell divides and zero transcendentals.
- **Floor fallback: bare donor-cell upwind** — the literal water solver, near-zero risk, but first-order
  diffusive (mushier). Acceptable only because today's scheme is *already* heavily diffused by a separate
  wind-coupled Laplacian and is *already* non-conservative.

---

## 1. The problem statement (the four axes)

S2b must give smoke + 5 gas planes an advection scheme that is simultaneously:

- **(a) conservation** — total `Σ smoke` and `Σ gasₖ` constant to the integer LSB in a sealed room (the
  **P2** contract). Not "within 0.1%" — to the bit.
- **(b) cross-GPU bit-determinism** — bit-identical integer trajectory across GPU architectures. No global
  reductions (the `mean_wp`-class hazard, plan §4), no per-cell IEEE transcendentals that differ across
  arch, no value-dependent control-flow cliffs that a 1-LSB slip can flip.
- **(c) diffusion / visual feel** — keep smoke reading as *crisp wisps* rather than smearing to uniform
  grey. This is the stated gameplay/visual value and the original reason smoke v2 chose semi-Lagrangian
  (`05_smoke.md` §2).
- **(d) migration effort** — a tractable port that reuses the S1 toolkit (`fixed_point.h`:
  `mul_wide`/`narrow`/`scale_mag`/`recip_mul`/`shr_round0`) and the shipped water flux idiom.

### 1.1 Why this is hard — the current scheme's structure

The current smoke advection (`cpp/src/smoke_dynamics.cpp`) is **semi-Lagrangian back-trace**: each
*destination* cell independently samples upstream via a wall-clipped ray-march (`sqrt` march length at
`:63`, `ceil`/`floor(x+0.5)`/`floor` rounding at `:64`/`:72-73`/`:95-96`) + bilinear interpolation, then
**renormalizes per destination** by `acc / wsum` (`:122`) so sealed/breach corners don't dilute the sample.

Two structural facts make this the wrong base for the four axes:

1. **It is a gather, not a flux pair.** Conservation rides entirely on the per-destination renorm divide,
   which normalizes *interpolation weights*, not *mass*. Unlike S1 water — where `flux` is gathered once
   and applied `±` to two cells so the `>>16` narrow can never create/destroy mass — there is no shared
   value cancelling between a source and a sink. So the S1 "conserve by construction" pattern does **not**
   directly apply.

2. **It is not conservative even in float, today.** Two destinations can back-trace into the same upstream
   region (mass duplicated) or a region can be sampled by no cell (mass dropped). The code's own test
   `tests/test_smoke_semilagrangian.py:181` (`test_closed_domain_no_mass_growth`) only asserts
   `total1 <= total0 + 1e-3` and `total1 > 0.9·total0` — i.e. it **explicitly tolerates losing up to 10%
   of the mass**. The doc agrees (`05_smoke.md`: "Semi-Lagrangian is not strictly conservative"). **P2 is a
   brand-new requirement smoke has never met** — there is no conservation regression to protect, only a
   conservation property to gain.

### 1.2 The determinism filter (the decisive lens)

Integer `+ − * >>` are exact and associative on every GPU architecture. Everything else is a hazard:

| Hazard class | What it is | Where it bites smoke today |
|---|---|---|
| **T — per-cell transcendental** | `sqrt`/`sin`/`exp` per cell; device `sqrtf` may differ across arch | the march length `sqrt` (`:63`) |
| **D — per-cell dynamic divide** | a runtime, data-dependent, possibly-zero-crossing divisor (the S1 toolkit's `make_recip` only handles *loop-invariant* divisors) | the renorm `acc/wsum` (`:122`); a flux-limiter ratio `r` |
| **R — global reduction** | a domain-wide sum (non-associative across parallel arch) | not present today; the trap a global mass-fixer would import |
| **S/O — signed/order dependence** | sign-asymmetric truncation, read-after-write ordering, value-dependent branches | the wall-clip early-`break` march; three rounding modes |

The back-trace hits **T + D + S/O** at once and is non-conservative. It is the *worst* family for the
four axes. The smoke-v2 decision (SL for low diffusion) **inverts under the determinism lens**: the thing
chosen for visual quality is the hardest thing to take to bit-exact cross-GPU integer.

### 1.3 Two facts the literature settles (these decide the shape of the answer)

- **Fact A — low diffusion is a property of reconstruction order, NOT of SL-vs-flux.** First-order/linear
  (bilinear, donor-cell) interpolation is diffusive whether back-trace or flux-form; van Leer/MUSCL/PPM
  reconstruction is sharp whether back-trace or flux-form. The current scheme uses **bilinear** interp —
  it is already at the *diffusive* end of its family. Switching to flux-form at the *same* (linear)
  reconstruction order costs essentially **no extra diffusion**; adding a limited high-order reconstruction
  makes it *sharper* than today, conservatively. The "we use SL for low diffusion" premise is carried by
  the *interpolation order*, which flux-form schemes keep. (SIAM, *On Shape-Preserving Interpolation and
  Semi-Lagrangian Transport*; CFD VL/PPM primer.)

- **Fact B — the cheapest "make-SL-conservative" trick is the exact determinism trap the migration exists
  to kill.** A global multiplicative mass-fixer (scale the whole field by `M_before/M_after`) conserves to
  machine precision but needs a **global sum of total mass each step** — the same non-associative reduction
  as `mean_wp` (plan §4, "the deepest latent desync"). **Disqualified on determinism grounds**, even
  though it is trivial to code. Any acceptable scheme must conserve **locally** (per-face), no domain-wide
  reduction. (ECMWF global mass fixers; Kaas 2008 LMCSL.)

Together: the answer is **not** "keep the back-trace and bolt on conservation." It is "adopt a local
flux-form scheme; recover crispness through reconstruction order, not through SL."

---

## 2. The families, rated on the four axes

Ratings are **relative to today's bilinear SL** and to the S1 water idiom already in the repo.

### Family 1 — Semi-Lagrangian back-trace (the current scheme)
- **Conservation:** ✗ none by construction; rides the per-destination renorm divide; not conservative even
  in float (10% loss tolerated by its own test).
- **Diffusion/feel:** low *in calm air* (its selling point) — but see §4: in the wind regimes the player
  actually looks at, a separate Laplacian already dominates.
- **FP-determinism fit:** ✗ worst — hits T (`sqrt`) + D (`acc/wsum`) + S/O (3 rounding modes, branch march)
  and is non-conservative, so making it conserve means importing R (global fixer). Four hazard classes.
- **Effort to make it meet (a)+(b):** highest — must migrate the integer sqrt, the dynamic reciprocal,
  three rounding modes, *and* retrofit conservation. **Reject as the base.**

### Family 2 — Conservative semi-Lagrangian (Lentine–Grétarsson–Fedkiw 2011; CSLAM; SLICE)
The only SL-family schemes that genuinely conserve, and they do it by **turning SL into a mass-transfer /
flux scheme**:
- **Lentine 2011** (graphics, *SCA '11*): reinterpret the bilinear weights as **mass fluxes**, enforce that
  the **sum of weights each *source* node gives away = 1**, locally per node (scale down where >1; a second
  forward pass redistributes where <1). This is the *scatter*-side constraint today's code is missing —
  today's `acc/wsum` normalizes the *gather* (per destination), conservation needs the *scatter* pinned
  (per source). The current scheme is "one loop away" from conservative, but the missing constraint is on
  the **other** loop, so it cannot be fixed by tightening the existing divide.
  - **Conservation:** ✓ exact, local. **Diffusion:** = today (still bilinear SL). **FP fit:** keeps the
    back-trace `sqrt`; the `<1` remainder forward pass is a **scatter write** (multiple sources → one cell)
    = a GPU race unless done as an additive flux into a per-face accumulator. **Effort:** medium-low
    (closest to existing code) but lands on a forward-scatter pattern that is less GPU/integer-clean than a
    pure gather of faces.
- **CSLAM** (Lauritzen 2010): integrate the upstream departure *polygon* against a sub-cell reconstruction.
  Exact, low-diffusion — but needs **polygon clipping + Gauss-Green line integrals + quadrature** per cell,
  heavy per-cell geometry/branching that is painful to make bit-identical in Q16.16, and **overkill on a
  regular axis-aligned grid**. **Effort: high. Not recommended** — it solves curved/unstructured-mesh
  problems Breach does not have.
- **SLICE** (Zerroukat–Wood–Staniforth): cascade of **1D conservative remaps** (PPM/PSM). Exact, local,
  low-diffusion, dimensionally split → easy to make bit-exact. In practice on a regular grid **converges to
  the same thing as flux-form SL** (Family 3) — a cousin, not a distinct endpoint.

**Verdict on Family 2:** if you *must* keep an SL flavour, Lentine is the cheapest conservative path and a
fine **first prototype** (reuses the back-trace), but its forward-scatter and retained `sqrt` make it a
worse fixed-point/GPU *endpoint* than Family 3. CSLAM is out.

### Family 3 — Flux-form / Eulerian-flux schemes (donor-cell, FFSL, MUSCL-limited) — RECOMMENDED
Evolve the **cell average** by the **divergence of face fluxes**: each face flux computed once, applied `+`
to one cell and `−` to its neighbour. This is *identically* the S1 water idiom. Conservation is **by
construction**, local, no global sum.

- **3a — Donor-cell upwind (first order):** the literal water solver. `v_face = (wind[i]+wind[n])>>1`,
  `donor = upwind smoke`, `flux = mul_wide(v_face, donor)`, then water's `flux_to_dq` + outflow limiter +
  `±` apply.
  - **Conservation:** ✓ LSB by construction (= water). **FP fit:** ✓ best — pure int adds/`mul_wide`/`>>`,
    **no transcendental, no per-cell dynamic divide** (the outflow limiter's `depth/out_sum` is a bounded
    exact integer divide, already shipped in water). **Effort:** lowest — port the water flux section,
    swap water's `v_face` for the wind-derived face velocity; smoke is *easier* than water (no own velocity
    field, no surface potential). **Diffusion:** ✗ first-order upwind is the *most* diffusive usable scheme
    (truncation error ≈ a Laplacian ∝ `|u|·dx·(1−Courant)/2`). This is the one real regression — mushier
    than today's SL in calm air. The **diffusivity floor.**
- **3b — Flux-form SL (FFSL, Lin–Rood 1996; SWIFT 2024):** split the displacement into an **integer
  cell-shift + a fractional sub-cell flux**. The integer shift is exact integer arithmetic (perfect for
  fixed-point); only the fractional flux needs a 1D polynomial reconstruction.
  - **Conservation:** ✓ LSB by construction. **FP fit:** ✓ excellent — the `c_int`/`c_frac` split is
    `floor` + fraction (top-16/bottom-16 bits of Q16.16), which **replaces the SL march `sqrt` entirely**;
    no per-cell transcendental, no dynamic divide. Handles large Courant (blast wind) natively. **Diffusion:**
    low with a van Leer/PSM 1D reconstruction (this is *the* operational climate-model low-diffusion
    conservative scheme). **Effort:** medium — a genuinely different solver, but the *same conservation
    skeleton S1 shipped*, generalized; dimensional splitting (X sweep then Y sweep) keeps it 1D-at-a-time.
- **3c — Donor-cell + MUSCL/TVD slope limiter:** donor-cell *plus* a limited anti-diffusive correction flux
  per face. The limiter `φ(r)` switches the high-order correction off at sharp gradients (monotone, no new
  extrema) and on in smooth regions (second order).
  - **Conservation:** ✓ LSB — the limited correction is *added to the face flux*, still gathered once,
    applied `±`. **Diffusion:** low — second-order in smooth air, ≈ parity with or better than today's SL,
    TVD-monotone at edges (no checkerboard). **FP fit:** ✓ good *if* the limiter is chosen well — see §3,
    the limiter ratio is the one hazard. **Effort:** moderate — "donor-cell (= water) + one limited
    correction flux," reusing `scale_mag` as the monotone outflow limiter.
- **3d — Graphics flux-interpolated SL (Hirasawa–Kanai–Ando 2021):** reformulate the SL interpolation as a
  **cell-face flux** so "the sum of flux exactly counteracts on cell faces" — a **plug-and-play extension
  to conventional SL** that "inherits all the benefits of SL" while becoming conservative + lower-diffusion.
  Same face-cancellation as S1. The most *directly on-point single citation* (game/film smoke, minimal
  extension to the SL code you have), but one paper, less battle-tested than Lin–Rood — treat it as 3b's
  mechanism in graphics terms.

**Verdict on Family 3:** this is the recommended structure. **3c (donor + limited correction)** is the
sweet spot for Breach — conservative, crisp, and the limiter can be made divide-free (§3). **3b (FFSL)** is
the cleaner endpoint if large-Courant blast winds prove troublesome or you want O(Δx⁴) sharpness. **3a
(bare donor)** is the zero-risk floor.

### Family 4 — Error-compensated SL (MacCormack / BFECC)
Forward + backward SL passes, estimate the round-trip error, subtract it → second-order, **dispersive not
diffusive** error → the *crispest* of the SL family. MacCormack preferred over BFECC (2 passes vs 3).
- **Conservation:** ✗ — these are *de-diffusers, not conservers*. They sit on the same non-conservative SL
  base and gain/lose mass exactly as plain SL does. The mandatory **extrema clamp** (clamp the corrected
  value to the min/max of the SL stencil nodes) is an L∞ *stability* limiter, not a *mass* budget — and the
  clamp-vs-revert-to-SL branch actively *breaks* conservation at the seam (one cell reverts, its neighbour
  keeps the MacCormack value, no flux pairing between them → mass appears/disappears). To conserve you must
  bolt on a *separate* global renormalization = the forbidden R reduction.
- **Diffusion/feel:** ✓ best of the SL family — crisper than today. **FP fit:** ✗ poor — inherits *all* of
  the back-trace's T+D+S/O hazards and **multiplies them 2–3×** (two-to-three back-traces), adds a
  value-dependent clamp-vs-revert branch (a new discrete desync vector of the class that flips fire's
  igniting↔extinguishing). **Effort:** heaviest SL lift, and still owes conservation. **Reject** — crisp
  but anti-conservative, the heaviest fixed-point surface area.
- The *salvageable* idea: apply a MacCormack-style correction **to the flux** (limit the anti-diffusive
  flux, keep the `±` face pairing) on a conservative base = exactly Family 3c. The error-compensated *flux*
  is viable; the error-compensated *cell value* (classic MacCormack/BFECC) is not.

### Family 5 — WENO
Flux-form (conservative by construction), crispest of all (3rd/5th order, holds sharp fronts). **FP fit:**
✗ poor — smoothness weights are **nonlinear rational functions** (`w_k ∝ 1/(ε+β_k)²`, then normalized by a
sum) → multiple per-cell divides + a per-cell normalizing division + a precision-budget headache in Q16.16.
**Effort: high.** Overkill: 5th-order shock-capturing for a `[0,1]` smoke tracer buys crispness you cannot
perceive at the cost of the heaviest per-cell divide load. **Not recommended.**

### Summary table

| Family | Conservation (P2) | Diffusion (feel, vs today) | FP/determinism fit | Effort |
|---|---|---|---|---|
| 1. SL back-trace (current) | ✗ not by construction (≤10% loss) | low (calm air) | ✗ worst: T+D+S/O, +R to conserve | highest to fix |
| 2. Conservative-SL (Lentine) | ✓ exact local | = today (bilinear) | ~ keeps `sqrt`; forward-scatter race | medium-low (proto) |
| 2. CSLAM | ✓ exact local | low | ✗ per-cell polygon geometry | high (reject) |
| **3a. Donor-cell upwind** | **✓ LSB by construction** | ✗ most diffusive | **✓ best: no T, no dynamic D** | **lowest** |
| **3b. FFSL (Lin–Rood)** | **✓ LSB by construction** | **low (crisp)** | **✓ no T, no dynamic D; replaces sqrt** | medium |
| **3c. Donor + MUSCL limiter** | **✓ LSB by construction** | **low (≈ parity / crisper)** | ✓ good (limiter ratio = the one D, avoidable) | moderate |
| 3d. Flux-interpolated SL (graphics) | ✓ LSB by construction | low (crisp) | ✓ good | low-medium |
| 4. MacCormack / BFECC | ✗ (anti-conservative clamp) | ✓ crispest SL | ✗ T+D+S/O ×2–3, +R to conserve | heaviest |
| 5. WENO | ✓ by construction | ✓ crispest | ✗ per-cell divide storm | high |

---

## 3. The crispness ↔ determinism crux — the limiter divide

The cheapest crisp conservative path (3c) hinges on **one** integer object: the flux-limiter ratio of
consecutive gradients, `r = (s_i − s_{i−1}) / (s_{i+1} − s_i)`. Its denominator is a **per-cell, runtime,
sign-changing** quantity that **passes through zero at every smooth extremum** — exactly where smoke wisps
peak. This is a class-D hazard the S1 toolkit's `make_recip` cannot launder (`make_recip` is for a single
loop-invariant divisor computed once in double at load; `r`'s divisor is per-cell dynamic).

There are two routes, and the choice is the heart of §9:

**Route (a) — divide-free two-slope limiters (RECOMMENDED).** minmod, MC (monotonized-central), superbee
and Koren can be written **without ever forming `r`** — directly as `min`/`max`/`minmod` of the two
one-sided differences `(s_i − s_{i−1})` and `(s_{i+1} − s_i)` in Q16.16. `min`, `max`, and sign-compares
are **exact and bit-identical in integer** (no IEEE, no divide). This route has **zero per-cell divides**
— strictly fewer determinism hazards than today's SL (which has a per-cell divide *and* a `sqrt`). The
division-by-zero of `r` is moot because `r` is never formed.

**Route (b) — explicit-`r` limiters (van Leer, van Albada).** Slightly smoother look, but need a genuine
per-cell integer divide (deterministic — integer `/` is bit-exact, truncating toward zero — but the worst
GPU op, and the migration plan flags per-cell divides as the thing to avoid, §3). Not worth a per-cell
divide when MC gets ~95% of van Leer's look divide-free.

**Limiter choice within Route (a):**
- **minmod** — most diffusive limiter, safest/simplest, a bit soft.
- **MC (monotonized-central)** — second-order, smooth look, **the right default for soft smoke.**
- **superbee** — least diffusive, but over-compresses smooth gradients into artificial stair-steps
  ("squaring-off") — *wrong aesthetic for soft smoke.*

→ **MC (two-slope, divide-free) is the default limiter.** minmod is the safe fallback. superbee is the
wrong look; van Leer only if MC's look disappoints and a per-cell divide is accepted.

### Crispness ranking (honest read, for *this* smoke use-case)
`WENO ≳ MacCormack/BFECC ≳ MUSCL(superbee) > MUSCL(van Leer/MC) ≈ current SL > MUSCL(minmod) > donor-cell.`

The load-bearing point: **a van-Leer/MC flux-limited scheme is ≈ visual parity with today's SL** — you are
*not* giving up crispness by switching to the conservative flux form, *provided* you use a limiter and not
bare donor-cell. The smear cost is real **only if you stop at first-order donor-cell.**

---

## 4. Why the visual cost is smaller than it looks (the "internal reality" check)

The "crisp wisps / low diffusion" premise does not fully survive contact with the shipped config:

- The smoke `step()` runs a **wind-coupled explicit diffusion pass *before* the SL advection**
  (`smoke_dynamics.cpp:155-171`): `smoke[i] += d_eff·dt·lap[i]` with
  `d_eff = d_smoke·(1 + wind_diffusion_scale·wind²)`, at `wind_diffusion_scale=50.0`. Wherever wind is
  strong — breaches, blasts, the exact moments the player looks at smoke — the field is **already
  aggressively smeared by a Laplacian** before the SL pass touches it. The crispness lives in *calm* air
  (`wind²≈0`), which donor-cell handles fine (near-zero wind → near-identity, no flux).
- A first-order donor-cell scheme adds numerical diffusion ∝ wind speed — i.e. it smears *most* exactly
  where the existing wind-diffusion *already* smears most. The two diffusivities **overlap heavily**; you
  would retune `d_smoke` / `wind_diffusion_scale` **down** to compensate and likely land at a similar or
  *crisper* calm-air look. (A `d_smoke` retune is already owed — the `dt_scale²` removal made diffusion
  ~9× weaker, `05_smoke.md`.)
- **Render-time crispness is the shader's job, by design.** Canon already plans the visual sharpness from
  an advected normal-map / curl-noise layer (`05_smoke.md` §6.1), explicitly so "the coarse tile grid
  [is never] visible." The *transport grid* does not have to carry the wisp detail.
- The gameplay couplings (ray attenuation / god-rays, fire plumes, breach venting) depend on smoke being
  *present in the right cells* and *venting correctly* — not on wisp sharpness. Venting is a flux behaviour;
  flux-form does it more honestly than the SL + `sink_hop` bolt-on.

So the diffusivity regression of even the *bare* donor-cell floor is partly pre-paid; with a limiter (3c)
it is ≈ neutral.

---

## 5. Fixed-point hazards of the recommended path (Family 3c) and how to handle them

The recommended path **deletes** the current file's three worst hazards and introduces a small, bounded set:

**Deleted by the switch (no longer a problem):**
- The per-cell march **`sqrt`** (`:63`) — gone; there is no ray-march in flux form. (T eliminated.)
- The **three rounding modes** (`ceil` `:64`, `floor(x+0.5)` `:72-73`, `floor` `:95-96`) — gone. (S/O reduced.)
- The **`acc/wsum` renorm divide** (`:122`) — gone; conservation is structural, not via a divide. (D eliminated.)
- The **wall-clip anti-tunnelling march** (`:62-87`) — gone; a single-cell donor flux *cannot* tunnel a
  one-cell wall by construction (a solid face carries no flux, like water's `!solid[i] && !solid[i+1]`
  guard). The complexity disappears rather than needing a fixed-point reimplementation.

**Introduced / retained (the new hazard surface):**

1. **The flux pair / shared-narrow conservation point (P2-critical).** Reuse the S1 idiom verbatim: gather
   the face flux once as `mul_wide` (Q32.32 int64), apply the *same* value `+dq`/`−dq` to the two cells,
   narrow at **one** shared truncation (`flux_to_dq` in `water_solver.cpp:208-230`). **Watch:** the MSVC
   vs clang/gcc 128-bit narrow must be bit-identical — water already solved this (`tests/_s1_flux_truncation_check.py`)
   and the same `flux_to_dq` lambda must be reused, not re-derived.

2. **The limiter (the one new D hazard) → use Route (a) divide-free MC/minmod (§3).** Express the limiter as
   `min`/`max`/`minmod` of the two one-sided Q16.16 differences. Zero per-cell divide. The limited
   correction flux is then folded into the face flux *before* the single narrow, so any limiter truncation
   still cancels in the `±` pair and **cannot break conservation.** (`scale_mag`'s shrink-only magnitude
   scaling is the correct monotone primitive for clamping the correction.)

3. **The outflow limiter** — reuse water's exactly: `out_sum = Σ outgoing dq magnitudes`, if
   `out_sum > smoke[i]` scale that cell's outgoing dq by `smoke[i]/out_sum` via **`scale_mag`** (shrink-only,
   toward 0). This is the bounded exact integer divide water already ships — *not* a transcendental, *not*
   a zero-crossing divisor. It is what lets the `clamp(smoke,0,1)` never inject mass.

4. **Signed wind / sign-symmetric truncation (S/O).** The donor pick is a sign test on `v_face`; the
   correction-flux and limiter must be sign-symmetric. Reuse `scale_mag`/`shr_round0` (not bare `mul_q16`)
   for magnitude operations, exactly as water documents the over-drain trap.

5. **No global reduction (R) anywhere.** Conservation is per-face local. **Do not** add a global mass-fixer
   under any circumstance — it is the `mean_wp`-class trap (plan §4). If P2 fails, the bug is in the
   shared-narrow flux pair (item 1), not a missing global fix.

6. **Q16.16 range/precision for a `[0,1]` tracer.** Smoke and each gas are normalized to `[0,1]`; Q16.16
   resolution is ≈ 1.5e-5, far below perceptual/gameplay thresholds, and the value range is tiny — no
   overflow risk in the flux `mul_wide`. (Same scale as heat/water; confirm the gas planes share it.)

### 5.1 The genuinely-new design task: `sink_hop` → a flux bias

This is the **one piece that does not drop out of the water template** and must be scoped explicitly.

`sink_hop` (`smoke_dynamics.cpp:225-277`) currently *is* a semi-Lagrangian gather: it runs
`backtrace_sample` with a one-cell pull toward the BFS breach-direction field (`sink_x/sink_y`), whose
breach corner samples 0 — deliberately **deleting** mass to vent the room. It is *supposed* to be
non-conservative (vacuum is a sink).

In a flux-form world, breach venting becomes natural and *does not need a separate pass*: the donor-cell
flux across a face into a vacuum cell carries smoke out, and the vacuum cell is zeroed each step — the mass
leaves through the flux and is deleted at the sink. The BFS direction field's *job* (pulling smoke along
the shortest air-path to a distant breach faster than raw diffusion) is **re-expressed as an extra
advective velocity** `= sink_strength · sink_dir` added into `v_face` for the same conservative flux gather,
run K×. This is a clean reformulation (just another velocity contribution to the same flux pair), but it
**is a behavioural rewrite of `sink_hop`, not a copy** — the deliberate mass-deletion now happens via the
flux into a zeroed vacuum cell rather than a corner-sampled-0 gather. In-scope for the feel-gated S2b step,
but it owes its own small design pass and a feel A/B (does the room still vent at the right rate?).

### 5.2 The wind-diffusion pass — migrate it too, and get conservation for free
The Pass-A wind-coupled Laplacian (`:155-171`) is independent of the advection choice and migrates cleanly
to the **R2-CONS edge-flux idiom** itself: compute each face's `flux = mul(d_eff_face, smoke[n]-smoke[i])`
once, apply `±`, share the narrow. That makes the diffusion pass *also* LSB-conservative — a bonus the
current `+= d_eff·dt·lap` form (an explicit Laplacian with truncation) does **not** give. Note `d_eff`
depends on `wind²` and feeds the `n_smoke` substep-count cliff (`physics_engine.cpp:188-203`); that cliff
is a **separate first-class determinism deliverable** (integer-max + fixed-point `ceil`, reuse
`fixed_point.h::ceil_div`) regardless of the advection scheme.

---

## 6. Migration shape for S2b (each its own gated commit)

Mirrors the S1 step structure (`s1_water_fixed_point_plan.md` §7). On an `s2b-smoke-fixedpoint` branch:

- **S2b-0** — representation: quantize `smoke` + 5 gas planes to int32 Q16.16 (the `[0,1]` tracers share
  the water/heat scale); float dequantize for the renderer + any remaining float bridges.
- **S2b-1** — **bare donor-cell upwind flux** (Family 3a) for smoke: port `water_solver.cpp:176-307`'s flux
  section, substitute the wind-derived `v_face` for water's, drop the surface-potential/velocity-kick
  passes. Gate **P1** (self-match `tol=0.0`) + **P2** (sealed-room `Σ smoke` LSB-constant). This is the
  conservation+determinism milestone, independent of feel.
- **S2b-2** — **the divide-free MC/minmod limited correction flux** (Family 3c): add the limited
  anti-diffusive face flux folded into the shared narrow. Gate P1+P2 (limiter must not break either) +
  the **feel A/B** (SSIM on rendered frames vs the float SL golden — confirm it still reads as crisp wisps).
- **S2b-3** — the **wind-diffusion pass** to R2-CONS edge-flux (§5.2) + the **`n_smoke` cliff** to
  fixed-point `ceil_div`.
- **S2b-4** — **`sink_hop` → sink-velocity flux bias** (§5.1): the genuinely-new piece. Feel-gated A/B
  (room vents at the right rate).
- **S2b-5** — **batch the 5 gas planes** through the identical kernel; they reuse everything.

Every step lands with the §6.3 gating contract: within-config self-match `tol=0.0`, no state-leak across
seeds, P2 green for the conserved fields, golden regenerated + version-bumped in the same commit,
feel-gated steps get a committed-artifact A/B (not a live eyeball).

---

## 7. What changes vs the smoke-v2 design doc

`05_smoke.md` §2 frames SL as *the* choice for low diffusion. This research **does not contradict the
visual goal** — it relocates where crispness comes from: from the *transport scheme* (SL) to the
*reconstruction order* (the MC limiter) + the *render layer* (the §6.1 noise shader). The conservative
flux form delivers the same calm-air crispness while *also* satisfying P2 + cross-GPU determinism, which SL
provably cannot without importing a global reduction. `05_smoke.md` §2/§3 will need an update note once
S2b lands (it is a feel-gated behaviour change vs the float golden) — to be folded into the architecture
chapter per the "design docs are canon chapters" rule.

---

## 8. Sources (grounding)

**Flux-form / conservative SL:**
- Lin & Rood 1996, *Multidimensional Flux-Form Semi-Lagrangian Transport*, Mon. Wea. Rev. 124:2046–2070 —
  https://journals.ametsoc.org/view/journals/mwre/124/9/1520-0493_1996_124_2046_mffslt_2_0_co_2.xml
- SWIFT (monotone FFSL, large Courant), 2024 — https://arxiv.org/abs/2405.20006
- Hirasawa, Kanai, Ando 2021, *A Flux-Interpolated Advection Scheme for Fluid Simulation*, The Visual
  Computer 37 — https://link.springer.com/article/10.1007/s00371-021-02187-2
- Lentine, Grétarsson, Fedkiw 2011, *An Unconditionally Stable Fully Conservative Semi-Lagrangian Method*,
  J. Comput. Phys. 230:2857–2879 — https://www.ulfhedinn.net/papers/stanford2010-01.pdf ; SCA '11 version —
  https://dl.acm.org/doi/10.1145/2019406.2019419
- Lauritzen, Nair, Ullrich 2010, *CSLAM on the cubed-sphere*, J. Comput. Phys. 229:1401 —
  https://www.sciencedirect.com/science/article/abs/pii/S002199910900597X
- Zerroukat, Wood, Staniforth — SLICE / PSM-SLICE —
  https://www.sciencedirect.com/science/article/abs/pii/S0021999107000113

**Flux limiters / MUSCL / upwind:**
- Sweby 1984, *High Resolution Schemes Using Flux Limiters*, SIAM J. Numer. Anal. — https://epubs.siam.org/doi/10.1137/0721062
- Flux limiter overview (minmod/superbee/van Leer/MC/Koren formulas + the min-max-vs-divide breakdown) —
  https://en.wikipedia.org/wiki/Flux_limiter
- MUSCL scheme — https://en.wikipedia.org/wiki/MUSCL_scheme
- Upwind differencing (first-order upwind: conservative, heavily diffusive) —
  https://en.wikipedia.org/wiki/Upwind_differencing_scheme_for_convection
- Dullemond, *Advection algorithms II: flux conservation & limiters* (lecture notes) —
  https://www.ita.uni-heidelberg.de/~dullemond/lectures/num_fluid_2012/Chapter_4.pdf

**Error-compensated (for contrast — not conservative):**
- Selle, Fedkiw, Kim, Liu, Rossignac 2008, *An Unconditionally Stable MacCormack Method*, J. Sci. Comput.
  35:350–371 — https://www.andyselle.com/papers/7/maccormack.pdf
- Kim, Liu, Llamas, Rossignac 2005, *FlowFixer: Using BFECC for Fluid Simulation* —
  https://faculty.cc.gatech.edu/~jarek/papers/FlowFixer.pdf
- GPU Gems 3 ch.30, *Real-Time Simulation and Rendering of 3D Fluids* —
  https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-30-real-time-simulation-and-rendering-3d-fluids
- Fedkiw, Stam, Jensen 2001, *Visual Simulation of Smoke* (the canonical SL-dissipation statement) —
  https://graphics.stanford.edu/papers/smoke-sig03/smoke.pdf

**Conservation requires explicit machinery / global-fixer trap:**
- *Efficient and conservative fluids using bidirectional mapping*, SIGGRAPH 2019 —
  https://dl.acm.org/doi/10.1145/3306346.3322945
- ECMWF global mass fixers (the global-reduction trap) —
  https://www.ecmwf.int/en/elibrary/74226-global-mass-fixer-algorithms-conservative-tracer-transport-ecmwf-model
- Kaas 2008, LMCSL (local mass-conserving SL), Tellus —
  https://tellusjournal.org/articles/10.1111/j.1600-0870.2007.00293.x

**Diffusion ∝ reconstruction order; fixed-point determinism:**
- SIAM, *On Shape-Preserving Interpolation and Semi-Lagrangian Transport* — https://epubs.siam.org/doi/abs/10.1137/0911039
- FixPointCS (bit-identical fixed-point sqrt/div/rcp via normalize + fixed-depth poly/LUT) —
  https://github.com/XMunkki/FixPointCS
- *Controlling Floating-Point Determinism in NVIDIA CCCL* (integer reductions deterministic; float needs
  RFA) — https://developer.nvidia.com/blog/controlling-floating-point-determinism-in-nvidia-cccl/
- Gaffer On Games, *Floating Point Determinism* — https://gafferongames.com/post/floating_point_determinism/

**Local code grounding:**
- `cpp/src/water_solver.cpp:176-307` — the shipped conservative donor-cell flux template (gather
  `mul_wide`, shared `flux_to_dq` narrow, `scale_mag` outflow limiter, `±` apply).
- `cpp/src/fixed_point.h` — the toolkit (`mul_wide`/`narrow` :84-98, `scale_mag` :164-179, `recip_mul`
  :121-154 [loop-invariant divisors only — the reason a per-cell `r` divide is a hazard], `ceil_div` :189).
- `cpp/src/smoke_dynamics.cpp` — the current SL scheme to replace (`backtrace_sample` :52-123: `sqrt` :63,
  rounding :64/:72-73/:95-96, renorm divide :122; `step` :125-219; `sink_hop` :225-277).
- `docs/fixed_point_migration_plan.md` — §2.3 R2-CONS, §4 reductions, §5.2 smoke audit (M8), §6.3 S2b.
- `docs/architecture/engine/05_smoke.md` — smoke-v2 rationale (§2), render-crispness plan (§6.1).
- `tests/test_smoke_semilagrangian.py:181` — `test_closed_domain_no_mass_growth` (tolerates ≤10% loss;
  proof today's scheme is not conservative).

---

## 9. The decision for Erik

**The question:** does S2 smoke/gas advection STAY semi-Lagrangian (made conservative + deterministic) or
SWITCH to a conservative flux-form scheme?

**Recommended default: SWITCH to a conservative flux-form scheme, specifically donor-cell upwind + a
divide-free two-slope MC/minmod limiter (Family 3c).**

Reasoning, mapped to the four axes:
- **(a) conservation:** free, by construction — reuses the shipped S1 flux pair (`mul_wide` → shared narrow
  → `±` apply). P2 LSB-conservation is met by the structure, not a per-cell divide or a global fixer. (And
  smoke has never been conservative before — there is no regression to protect.)
- **(b) determinism:** strictly *better* than SL — the switch **deletes** the per-cell `sqrt`, the three
  rounding modes, the wall-clip branch march, and the `acc/wsum` divide. The MC/minmod limiter is pure
  integer `min`/`max`/sign (exact, bit-identical cross-arch). **Zero per-cell transcendentals, zero per-cell
  dynamic divides, no global reduction.**
- **(c) feel:** ≈ visual parity with today's SL (a limited second-order flux is not mushier than bilinear
  SL); and the visual cost is partly pre-paid by the existing wind-diffusion pass, with render crispness
  owned by the §6.1 shader.
- **(d) effort:** the path of *most reuse* — it is the water solver's flux section + one limited correction
  flux + the gas batch. The only genuinely-new piece is reformulating `sink_hop` as a sink-velocity flux
  bias (§5.1), which is in-scope and feel-gated.

**The alternative (if Erik prefers minimum-change-first):** prototype **Lentine per-source-weight=1
conservative SL** (Family 2) — it reuses the existing back-trace + bilinear and only adds the local
scatter-normalization + a remainder pass, so it is the cheapest path that is *both* local and conservative.
But expect Family 3 to be the cleaner cross-GPU/fixed-point endpoint (a pure gather of faces, vs Lentine's
forward-scatter race and retained per-cell `sqrt`). Treat Lentine as a *de-risking prototype*, not the
shipped scheme.

**Explicitly rejected:**
- **Keep the renorm-divide back-trace + a global mass-fixer** — re-imports the `mean_wp`-class global
  reduction the whole migration exists to kill (plan §4). The "obvious cheap fix" is the determinism trap.
- **MacCormack / BFECC** — not conservative (the extrema clamp + revert-to-SL seam actively breaks it),
  heaviest fixed-point lift (2–3× the back-trace hazards), still owes a conservation mechanism.
- **WENO** — per-cell divide storm, imperceptible benefit for a `[0,1]` tracer.

**The one tuning knob to pin in §9's decision:** the limiter — **MC (two-slope, divide-free)** as the
default smooth-smoke look; **minmod** if you want maximum safety/simplicity at a small crispness cost;
**bare donor-cell (3a)** as the zero-risk floor if even the limiter feels like scope creep for the first
landing (ship 3a in S2b-1, add the limiter in S2b-2). Do **not** pick superbee (stair-steps — wrong look)
or van Leer (a per-cell divide for ~5% more smoothness than MC).

**Owed before implementation (per the discuss-first rule):** Erik's OK on (1) flux-form vs Lentine-proto,
(2) the MC-default limiter (vs minmod / bare-donor floor), (3) accepting the `sink_hop` rewrite as in-scope
for S2b, and (4) the `d_smoke`/`wind_diffusion_scale` retune that the diffusivity overlap (§4) will require.
