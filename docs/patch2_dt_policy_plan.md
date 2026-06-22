# Patch 2 — the coherent dt policy (plan)

**Status:** plan, post-design-discussion with Erik (2026-06-19). Builds on Patch 1 (the unified C++
`PhysicsEngine`). **Feel-gated, not bit-identical** — Patch 2 deliberately *changes behavior*, so the
0-ULP A/B harness does NOT apply; the acceptance test is conservation + the venting test + **Erik's eye**.

---

## 1. Why

The build finding behind it: **one number — the wave-CFL substep count `n` — is doing four jobs at
once**: (1) the explicit wave's CFL, (2) the implicit diffusion's step count (which wants 1), (3) the
smoke explicit-diffusion CFL, and (4) the smoke sink-pull's drain rate (capped 1 cell/substep, so it
drains `n` cells/tick). Patch 2 untangles them so each system runs on its own schedule.

"Tuning isn't sacred" (Erik): today's numbers aren't optimal anyway, so Patch 2's job is the right
*structure* with sensible defaults — the real feel-tuning is a later dedicated pass.

---

## 2. Scope (the reshaped, simpler version)

After the design discussion we **dropped** the wave-active gate, the dead-wave snap-to-zero, the epsilon
dial, and the SIMT/determinism caveats (see §5). What remains:

1. **Split the wave-substep loop from a single diffusion solve.** Today `atmos.step` fuses an explicit
   wave kick + the implicit (Gauss-Seidel) diffusion, and the orchestrator runs the whole thing `n`
   times. Restructure so the **wave substeps at its CFL `n`** but the **implicit diffusion solves ONCE**
   per tick (it's unconditionally stable — it never needed the substeps). *This is the headline win:
   the diffusion is the expensive part (~8 GS sweeps), and running it once instead of ~18× is roughly a
   3× cut to the atmosphere cost, single-game and farm alike.* Behavior change (the diffusion sees the
   accumulated wave transfers at once rather than incrementally) — validated by conservation + feel.
2. **Sink-pull → its own `K`-hop pass.** Pull the breach-pull *out* of smoke's advection back-trace.
   Keep the **one-cell-per-hop cap** (load-bearing: the sink direction is a BFS next-hop pointer down
   the shortest path to the breach, so smoke must walk it one cell at a time or it cuts a corner into a
   wall). Run it **K one-cell hops per tick**, where `K` is a real "vent cells/tick" dial, independent
   of the wave's `n`. Sensible default (~the old `n`, ≈16), **Erik tunes K by eye**.
3. **Smoke-CFL floor.** Smoke's *diffusion* is explicit (forward-Euler), so it has a stability speed
   limit that *tightens under high wind* (`d_eff = d_smoke·(1 + wind_diffusion_scale·|wind|²)`). The
   smoke substep count becomes `max(1, n_smoke)` derived from the **spatial-max** `d_eff` — auto-tightens
   under a shockwave, relaxes otherwise. Derived stability bound, no tuning knob.
4. **Remove `dt_scale`.** It's applied twice (≈9×) and scales smoke *diffusion and advection* both.
   Delete the fudge: smoke moves on the real dt, and "how hard smoke rides the wind" becomes the clean
   `advection_rate` coefficient. Changes the diffusion rate (the 9× went away) → **Erik re-tunes
   `d_smoke`** later. Bonus: removing it pushes smoke *back inside* the CFL limit (the fudge was
   inflating the timestep toward instability).
5. **GS-residual hook** (Claude builds, read-only). The instrumentation doesn't exist yet — add a
   measurement of the GS residual **after the sweeps but before the vacuum/sponge BC pass** (the BC pass
   mutates atmosphere post-solve, so a naive post-step residual is contaminated). Norm: Linf, normalized
   by the field scale. It answers "do 8 sweeps under-relax at this resolution?" — which decides whether
   the resolution patch ever needs a two-grid correction. No behavior change.

---

## 3. The reshaped per-tick order

```
cast_fire_heat                       (Python, unchanged — Q3)
[water Python-prep] → engine.step_water   (unchanged from Patch 1)
engine.run_substeps  →  becomes:
   for _ in range(n_wave):  wave_substep()        # explicit wave at its CFL
   diffusion_solve_once()   →  wind = -grad(atm + wave_p)   # ONCE, not n×
   for _ in range(n_smoke): smoke_advect_diffuse()          # rides the (quasi-static) wind
   for _ in range(K):       smoke_sink_hop()                # the decoupled vent drain
engine.step_tail                     (unchanged from Patch 1)
```

Each loop on its own count: `n_wave` (wave CFL), 1 (diffusion), `n_smoke` (smoke diffusion CFL), `K`
(vent rate). None secretly sets another's.

---

## 4. Implementation pieces

- **AtmosphereSolver** (`atmosphere_solver.cpp`): split `step()` into a `wave_substep()` (feed source +
  explicit kick + transfer→atmosphere) and a `diffuse_solve()` (the implicit GS + BCs + wind). The
  engine's `run_substeps` calls the wave loop then the single diffuse. Add the GS-residual accessor.
- **SmokeDynamics** (`smoke_dynamics.cpp`): remove the sink from the advection back-trace (advection
  becomes wind-only); add a `sink_hop()` method (one 1-cell BFS-gradient pull). Remove the `dt_scale`
  multiply; introduce/repurpose `advection_rate` as the wind-ride coefficient on the real dt.
- **`PhysicsEngine::run_substeps`** (`physics_engine.cpp`): the reshaped order above — derive `n_wave`,
  `n_smoke` (from spatial-max `d_eff`), drive the wave loop / single diffuse / smoke loop / `K` sink
  hops. `K` from config (`smoke_vent_hops` or a `drain_rate`·tps).
- **config.toml**: `dt_scale` removed; `advection_rate` (re-tuned), `K`/vent-rate dial, the smoke-CFL is
  derived (no key). All `[physics]`.

---

## 5. Dropped, with rationale (Erik's wave question)

The wave-active gate (snap small waves to zero, skip the wave's own substeps when calm) is **deferred,
not deleted**:
- It's a *single-game-only* win (calm-tick atmosphere ~5× cheaper) — on the parallel-sim farm it's
  washed (data-dependent substep count → SIMT divergence; one active blast forces the whole batch to
  the full count).
- It adds real complexity: the snap, an epsilon dial, the steel-hull ringdown feel risk, and a float-
  `max` branch that's a cross-machine determinism hazard (resolves under fixed-point, but still).
- The diffusion-decouple (§2.1) is the big win and is in regardless.
- **Revisit after profiling:** Erik's single-game-speed focus makes it a closer call than the farm
  argument implies — so once Patch 2 lands, profile; if the wave is a real single-game bottleneck, the
  gate is a small clean bolt-on then.

---

## 6. Gates (NOT 0-ULP — Patch 2 changes behavior by design)

- **Conservation:** `test_atmosphere_conservation` (sealed room conserves exactly) must stay green —
  the wave/diffusion split must not leak mass.
- **Venting:** `test_smoke_sink_pull::test_breached_room_clears` must still pass (a breached room clears)
  — at the new `K`. Update the test if the rate changes meaningfully; add a "venting half-life
  invariant as `n_wave` varies" test (proves the decouple worked).
- **Dormancy:** the water/smoke dormancy checks (no behavior when a system is off).
- **The 336 suite** green (excluding any test that encoded the old coupled rate — update deliberately).
- **Erik's feel-test (the real acceptance):** (1) calm room — air still settles right with the
  decoupled diffusion? (2) breach venting — smoke clears at a `K` you like? (3) smoke riding a shockwave
  wind — looks right, no checkerboard? (The steel-ring ringdown test is no longer needed — we dropped
  the snap.)

---

## 7. Open / deferred

- **`K` default + the `d_smoke` re-tune** — Erik tunes by eye (a tuning pass).
- **The wave-gate** — deferred, revisit after profiling (§5).
- **Implicit smoke diffusion** (fold smoke diffusion into the IMEX) — the fine-resolution fix; deferred
  to the resolution patch. Patch 2 keeps smoke diffusion explicit + the CFL floor.

---

## 8. Process

Patch 2 is smaller than the unification and feel-gated, so the validation is conservation + the feel-
test, not a 0-ULP gate or a full panel. Plan: a light sanity pass on this plan (optional), then
implement (the AtmosphereSolver split is the meatiest; do it behind the conservation test), then **Erik
feel-tests** the calm/venting/shockwave trio. Tuning of `K` and `d_smoke` is a follow-up pass.
