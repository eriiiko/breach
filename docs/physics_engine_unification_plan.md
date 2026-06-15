# PhysicsEngine unification — plan v2 (post-panel)

**Status:** v2, revised after the expert+adversarial panel (6 reviewers, code-grounded). The panel
verdict on v1 was **needs-rework / over-scoped**; this v2 acts on it. Grounded in
`docs/physics_field_interaction_map.md`, `docs/resolution_architecture_proposal.md`, the locked
decisions (uniform grid, full fixed-point cross-GPU — both *later*), and the panel findings.

**The one-line change from v1:** the "unification" was six subsystems in one patch resting on two false
claims (a 0-ULP gate that can't hold under `/fp:fast`, and a harness that doesn't exist). v2 **splits it
into three patches**, makes the missing harness Patch 0, and corrects five code-level errors v1 made.

---

## 0. The split (the panel's unanimous call)

| Patch | Contents | Gate |
|---|---|---|
| **0 — the harness** (prerequisite) | a field-level A/B determinism harness (per-cell, every sim field, old-path vs new-path, same seed) | builds + passes on the *current* code (identity) |
| **1 — unification (THIS plan)** | C++ `PhysicsEngine` skeleton + glue→C++ + `stamp_units` full-rebuild→C++ + GPU-prep hooks + trivial deletions | **pure structure**, A/B harness green per field per tick |
| **2 — coherent dt policy** (feel-gated) | subcycle-when-active + dead-wave snap + smoke-CFL floor + the sink-pull decouple + the `dt_scale` cleanup + the GS-residual hook | new venting/sealed/residual tests **+ Erik's eye** |
| **3+ — with the CUDA port** | the shared stencil; the `stamp_units` *delta* (refcounted + flooded-restore + flat buffer) | GPU occupancy / farm bandwidth |
| **separate** | the Q16.16 fixed-point migration + the cross-machine two-seeded harness | its own multi-week project |

Rationale: Patch 1 stays on a **hard determinism gate with no feel-tests**; the one behavior change
(dt policy) is isolated in Patch 2; both pay-later-on-GPU optimizations move to the patch that pays for
them. This directly fixes v1's self-flagged scope creep.

---

## 1. Patch 0 — the A/B determinism harness (must exist before Patch 1 touches a solver)

v1 cited a "0-ULP A/B harness" as the gate. **It does not exist.** Today's only determinism test
(`tests/test_simulation.py`) compares five whole-grid *means/maxes* in same-process replay — it cancels
per-cell sign-flipped errors and re-runs the same build, so it **cannot** detect the float-reorder
desync this patch risks. The one glue-refactor precedent accepted `atol=1e-5`; `_water_dormant_check` is
an untracked render-RT byte compare.

**Build it first:** snapshot every sim field (`atmosphere`, `wave_p/v/source`, `wind_x/y`, `gas[N]`,
`fire`, `water_depth`, `flow_vx/vy`, `heat`, `temperature`) each tick; run the old path and the new path
on the same seed+inputs; assert per-field per-tick equality. **The tolerance is itself a decision (§2).**

---

## 2. The `/fp:fast` decision (Erik's call — recommendation inside)

The build is `/O2 /fp:fast /arch:AVX2` (`cpp/CMakeLists.txt:14,16`). Under that, the compiler may
reassociate, contract to FMA, and flush denormals, and AVX2 reduces 8-wide — a *different* order than
strict-IEEE numpy. So a numpy→C++ glue port **will not** match to 0 ULP except by luck (the
`water_solver.cpp:31-35` `zeros()` scratch exists precisely because `+0.0f` folds differently under
`/fp:fast`). Two ways to make the gate honest, **pick one:**

- **(A, recommended) Compile the moved glue in an `/fp:precise` translation unit** and prove 0-ULP per
  expression. The glue (W3/W5/per-gas) is small and elementwise, so the perf cost is negligible and the
  determinism story stays *hard* — which the fixed-point future wants anyway.
- **(B) Drop the gate to a stated tolerance** (repo precedent `atol=1e-5`) **and add a separate
  same-build replay-determinism gate.** Cheaper, but the tolerance gate can't certify bit-identity, so
  it can mask a real desync.

My lean is **(A) for the glue.** This is the one genuinely-yours decision in Patch 1.

---

## 3. Patch 1 — scope & structure (pure, test-identical)

**Goal unchanged:** one C++ `PhysicsEngine` owning the (verified stateless/const) solver instances +
the per-tick order, exposing `step(dt)` — the CUDA migration boundary. `GameMap` stays the interface
(`gmap.<field>`); no caller changes. **Arithmetic is not touched** (Q16.16 is a separate patch).

### 3a. Corrected invariants (v1 errors the panel caught)
- **Re-fetch field pointers every `step()`** (today's per-call `get_2d` pattern) — **NOT** "one bind at
  construction." A construction-time bind is a dangling-pointer trap against `reset()`
  (`simulation.py:188` reallocates GameMap) and `_end_round` (`simulation.py:817` *reassigns*
  `obstacles`). The per-call re-fetch is zero-cost and already how it works.
- **`smoke == gas[BLACK_SMOKE]` aliasing is an invariant:** the engine binds ONE gas buffer; `smoke` is
  a view into `gas[BLACK_SMOKE]` (`gamemap.py:109`), never a second allocation — `fire.step` writes
  `smoke`, the per-gas loop writes `gas[gi]`, same backing store. Add to the gate list.
- **Per-gas diffusion is a `step()` argument, not a mutated member.** Today the orchestrator does
  `self.smoke.d_smoke = gas_diffusion[gi]` per slice (`physics_runner.py:369`) — a hidden per-tick write
  to a "config" member. Pass it as an arg; keep `SmokeDynamics` stateless (matches the GPU want: one
  batched stencil over `(n_gases,h,w)` with diffusion as a coefficient vector).

### 3b. Phase A — skeleton
C++ `PhysicsEngine` holds the const solver instances + the **outer** per-tick order:
`fire-heat → water(+W3/W5/seal) → substep-loop → ripple → fire → temperature`. `PhysicsRunner` becomes
a thin Python shim calling `engine.step(...)`. **No arithmetic change.** *(The substep loop is NOT pure
order — see 3c; Phase A owns the outer order, the loop moves as a unit in Phase B.)*

### 3c. Phase B — glue → C++, one coupling at a time, each behind the A/B harness
- **The IMEX substep loop + per-gas loop move as ONE entangled unit** (v1 wrongly split them). The loop
  body interleaves `n = ceil(sim_time / atmos.max_dt())`, `dt_actual = sim_time / n`, the per-gas
  `d_smoke` rebind + `.any()` empty-slice skip. **Pin `n` and `dt_actual`:** compute them in ONE
  language/precision and pass across the boundary — `n` is an integer cliff where a 1-ULP difference in
  `sim_time/n` flips `n` and desyncs everything downstream.
- **W3/W5/flooded-seal** (`physics_runner.py:606-651`): pure elementwise `np.where`/`clip`/`multiply` —
  the cleanest to port (1:1 to a per-cell kernel later), so do these first. **`/fp:precise` per §2(A).**
- **Keep `cast_fire_heat` in Python** (Q3): it feeds the raycaster, which is out of scope; moving only
  the enumeration buys nothing and adds a C++↔raycaster coupling. *(Doc fix: §4's old note "the
  `update_from_fire` C++ path exists" is wrong — the runner calls `cast_source_directional`, a different
  path; `update_from_fire` is the legacy scalar `light_map` updater.)*

### 3d. Phase E — `stamp_units` full-rebuild → C++ (NO delta)
Move the per-tick full-field rebuild (`gamemap.py:532-544`: `dyn_permeability[:] = permeability`,
`dyn_light_atten` ×3 RGB, `dyn_wave_absorb`, + the unit loop) into C++ **as an idempotent full rebuild**
— still a big farm win (no Python loop, no per-tick re-build cost) and bit-trivially identical. **The
delta is deferred** (Patch 3) because it breaks three things v1 missed: the W3 flood-seal auto-clear
(`physics_runner.py:648` is a *second* writer of `dyn_permeability` relying on the full reset to clear —
no test covers "water recedes, air flows again"), overlapping-footprint MIN/MAX combine
(`gamemap.py:572-581`, non-composable without per-cell occupancy refcounting), and the freed-wall
neighbour-mean atmosphere fill (`gamemap.py:583-588`).
**Pin the cross-tick order:** `stamp_units(rebuild) → W3/W5/flooded-seal → substep loop`. The seal lives
exactly one tick today only because `stamp_units` (Simulation step 6) runs *before* physics (step 7);
moving either into the engine without fixing the order clobbers or staleness-breaks the seal.

### 3e. GPU-prep hooks (behavior-neutral, fold in now — they're free and save a repaint)
- **A reusable scratch arena owned by the engine.** Every solver except temperature allocates fresh
  `std::vector`s inside `step()` (atmosphere `lap/rhs/vac_dist`, smoke `lap/src`, water
  `surface/fx/fy/scale`) — on CUDA that's a per-step `cudaMalloc` (fatal). `temperature_solver.h:126`'s
  `mutable scratch_` is the template; the `PhysicsEngine` owns the pool, solvers take from it.
  Bit-identical → fits the test-identical phase.
- **The `mean_wp` deterministic-reduction helper, with its summation ORDER pinned now**
  (`atmosphere_solver.cpp:108-116`). This is the deepest determinism trap (CPU-scalar/SIMD/CUDA-warp
  give different float sums). Isolating the call site without fixing the *order* defers the hardest work
  to a rewrite — so fix the order now (a deterministic reduction tree), even while still float.
- *(Deferred to Patch 3: the flat packed delta buffer for the eventual `stamp_units` delta.)*

### 3f. Trivial deletions (separate micro-commits)
- Remove the vestigial `AtmoDiffusion` (`PhysicsRunner` uses `AtmosphereSolver`, never it — confirmed).
- Remove the vestigial `is_wall` C++ param (fed `gmap.solid`; redundant since the retire).
- Fix `simulation.py:817` `obstacles = solid.copy()` → in-place `obstacles[:] = solid` (honor engine/02).

---

## 4. §10 rescoped (the panel: don't oversell it)

There is **no** existing typedef'd-scalar arithmetic template (temperature/raycaster use raw `int32` +
hand shifts + free functions). "float→fixed is a type change not a rewrite" is aspirational, and the
hard fixed-point sites (the GS per-cell divide `atmosphere_solver.cpp:196`, the gather `min`, `sqrt`) are
not free type-swaps. So **drop the "typedef-swappable arithmetic across all solvers" framing.** The
defensible, cheap, behavior-neutral fixed-point-*aware* core that stays in Patch 1 is exactly the three
hooks in §3e (scratch arena, `mean_wp` deterministic reduction, the deferred flat delta) **+ "write no
new gratuitously-float idioms in the moved glue."** The real arithmetic migration is its own patch.

---

## 5. What stays Python / out of scope (unchanged, confirmed correct by the panel)

FieldEdit enqueue, `destroy_wall`, burst, ignition, unit-heat-damage stay in `Simulation.step` (actor/
topology/threshold logic). Reading the `Unit` list to compute footprints stays Python (actors are CPU);
only the field-write half is C++. The split rule (engine/02): field→field → C++; actor→field → boundary
Python. The fixed-point migration, GPU residency, the resolution change: all separate later patches.

---

## 6. Gates (Patch 1)

- **Per-field A/B equality** (Patch 0 harness) every phase — at the tolerance chosen in §2.
- The existing 336 suite green incl. `test_atmosphere_conservation` (the door-stamp-leak guard),
  `test_smoke_sink_pull` (venting — unchanged in Patch 1), the water/smoke dormancy checks.
- Both `tests/test_main_smoke.py --auto` runs (default + `unhcr_vessel_2`) exit 0 each phase.
- The `smoke==gas[BLACK_SMOKE]` aliasing + in-place-write invariants in the bit-identity checklist.

---

## 7. Top risks (panel-sourced)

1. **Bit-identity under `/fp:fast`+AVX2** — the load-bearing one. Mitigated by §2's decision (lean: glue
   in `/fp:precise`, prove 0-ULP per expression).
2. **The harness doesn't exist** — Patch 0 builds it before any solver is touched.
3. **Dangling pointers** — solved by per-call pointer re-fetch (§3a) + the `:817` in-place fix.
4. **Phase A/B entanglement + the integer cliff at `n`** — move the substep loop as a unit, compute
   `n`/`dt_actual` once and pass across (§3c).
5. **`stamp_units` delta correctness** — avoided by shipping the C++ full-rebuild only (§3d).
6. **GPU-port repaint** if the scratch arena / per-gas-as-arg aren't done now — they are (§3e, §3a).

---

## 8. Carried into Patch 2 (the dt policy — recorded so it's not lost)

The coherent dt policy, **with the panel's corrections:**
- Subcycle-when-active + the **dead-wave snap-to-zero** (note the feel risk: snapping the sub-eps
  acoustic tail can deaden the deliberately-*ringy* steel hull — the feel-test must include a
  sealed-steel-room blast ringdown, not just calm air + venting). The gate is a float-`max > eps` branch
  → a cross-machine divergence hazard, so design it against the determinism harness.
- The **smoke-CFL floor** from the *actual* expression
  `dt < dx² / (4 · d_smoke · (1 + wind_diffusion_scale·|wind|²) · dt_scale²)` using the **spatial-max**
  `d_eff` (it tightens under shockwave wind — correct) and the corrected `dt_scale²` (=9× at the shipped
  3.0, double-applied).
- The **sink-pull decouple (Q2):** keep the 1-cell-per-hop BFS-gradient cap (it's a correctness
  requirement — an uncapped multi-cell hop flies off the BFS path into a wall) and run the sink sub-pass
  its **own K = round(drain_rate·dt) times per tick, independent of `n`**. No bit-identical baseline —
  the old drain *was* `min(sink_strength,1)·n`, an artifact of the wave CFL; Erik re-tunes K by eye,
  with `test_breached_room_clears` as a floor and a new "venting half-life invariant as n varies" test.
- The **`dt_scale` cleanup** moves here as a **flagged behavior change** (9×, scales both) with a
  diffusion re-tune under Erik's eye — not the "behavior-preserving fold" v1 claimed.
- The **GS-residual hook** must be *built* (it doesn't exist — fixed 8-sweep loop, no residual eval),
  measured AFTER the GS sweeps but BEFORE the vacuum/sponge BC pass (`atmosphere_solver.cpp:243-264`
  mutates atmosphere post-solve and would contaminate it), norm + normalization specified.
