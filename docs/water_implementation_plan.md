# Water Implementation Plan — pipe-model fluid layer

**Status:** build plan, review round 1 complete (2026-06-10, three lenses: numerics/physics,
codebase contracts, test/build/process — all findings folded in below). NOT canon. Canon
design is [engine/07_fluid_and_water.md](architecture/engine/07_fluid_and_water.md); on any
conflict, the chapter wins.

**Procedure:** each step is one implementer agent: implement → `cmake -B build` /
`cmake --build build --config Release` (from `cpp/`; game window must be CLOSED) →
full pytest (baseline: 228 collected, all green) → commit+push if green / revert+report
if red. Every step ships **dormant-safe**: with zero water on the map, a full physics tick
is behaviour-identical to before the step.

**Cross-system interface (agreed with the fire/temperature side, 2026-06-10):**
`gmap.water_depth` — float32 `(h, w)`, metres of standing water — is THE shared field.
The fire side will read it as a heat sink (heat landing on a wet tile spends itself on
evaporation; the tile stays cool; fire dies) and expects boil-off to emit `white_smoke`.
There is **no water-temperature field**. That heat-sink consumer is the fire side's lane —
this plan must NOT implement it; this plan's W5 owns only the pressure-keyed flash-boil.
`ice_depth` joins the contract later (canon §5.4, post-temperature-tuning).

---

## Step W1 — C++ `WaterSolver` (the pipe model, solver only — unwired)

**Files:** new `cpp/src/water_solver.h` + `cpp/src/water_solver.cpp`; append both to
`pybind11_add_module` (`cpp/CMakeLists.txt:23-30`); bind in `cpp/src/bindings.cpp`
(`py::class_` + `def_readwrite` per tunable + lambda `step`, pattern `bindings.cpp:44-71`).
**Nullable arrays** use the existing precedent (`cast_source_directional`,
`bindings.cpp:294-356`): parameters typed `py::object`, `is_none()` → `nullptr`, else cast
to `py::array_t<float>` kept alive in scope, default `py::arg(...) = py::none()`.

**Class shape** (stateless; params public members; raw-pointer step):

```cpp
struct WaterSolver {
    // tunables (bound from config [physics.water])
    float g         = 9.81f;   // m/s^2 (prototype-validated)
    float damping   = 1.0f;    // 1/s pipe friction — LIFTED from fluid_test.py:39 (the
                               // side-by-side run that justified the model); tune in [0.5, 1.0]
    float dx        = 0.333f;  // m — set from the level's tile_size_m, never assumed
    float k_p       = 0.0f;    // pressure head, m per pressure-unit; 0 == head OFF (W4 turns on)
    float v_max     = 8.0f;    // m/s safety clamp (safe WITH the outflow limiter below)
    float depth_eps = 1e-5f;   // m snap-to-zero (kills denormal creep)
    float h_ref     = 2.5f;    // m reference column for the CFL bound (= ceiling_h)

    // House pattern (AtmosphereSolver::max_dt, physics_runner.py:216): Python owns the
    // substep loop, the solver owns the bound. Plain wave CFL at the reference depth,
    // with a margin for the head-term stiffening (see W4):
    //   max_dt = 0.5 * dx / sqrt(g * h_ref * (1 + k_p * P_ref / head_ref))
    // with P_ref = 1.0 (atm), head_ref = 0.2 (m — the W4 documented worst-case free
    // column; round-2 review: 0.5 under-covered it by 27%). k_p = 0 ->
    // 0.5*dx/sqrt(g*h_ref) = 33.6 ms at dx=1/3, h_ref=2.5 -> 2 substeps at 24 tps.
    // k_p = 0.5 -> 18.0 ms -> 3 substeps, worst-case CFL 0.37 (covers c_eff = 8.9 m/s).
    // NOT covered: the flood_eps asymptote (free_h -> 0.05 m -> c_eff ~ 16 m/s) — the
    // clamps+limiter contain it; the W4 tuning session watches near-flooded cells.
    // This is a REAL CFL: linearised, the pipe model is a damped wave with
    // c = sqrt(g*depth); damping removes the wet/dry blow-up, NOT the wave CFL.
    float max_dt() const;

    void step(float* water_depth, float* flow_vx, float* flow_vy,
              const float* floor_height,          // nullable -> flat zero
              const float* atmosphere,            // nullable -> no head term
              const float* wave_p,                // nullable -> no head term
              const bool*  solid,                 // STATIC walls (gmap.solid) — units do NOT block water
              int h, int w, float dt,
              float tilt_x, float tilt_y) const;  // radians about grid centre; sane range |tilt| < ~30 deg
};
```

**Numerics** (canon §2.2; cell-centred velocities + averaged faces + donor-cell upwind —
verified IDENTICAL to the validated prototypes `fluid_test.py:92-155`,
`fluid_tilted_ship.py:39-49`, with ONE deliberate exception, below):

```
# 1. surface potential (per cell; metres throughout)
tilt(x,y)  = tan(tilt_x)*(x - cx)*dx + tan(tilt_y)*(y - cy)*dx     # cx,cy = grid centre (W/2, H/2)
surface[i] = floor[i] + tilt + depth[i]
if (k_p != 0)  surface[i] += k_p*(atm[i] + wave_p[i])    # GATED: k_p==0 must be bit-identical
                                                         # to passing no pressure fields at all

# 2. damped explicit velocity kick (central difference; Neumann MIRROR at solid)
dS/dx at (y,x) = (S[y,x+1] - S[y,x-1]) / (2*dx)          # a solid neighbour mirrors the centre value
vx += dt * (-g * dS/dx - damping*vx);   vy likewise
vx = vy = 0 on solid; clamp |component| <= v_max
# out-of-bounds neighbours are treated as solid (grid border = wall)

# 3. upwind face fluxes from PRE-UPDATE depth (gather), then the per-cell OUTFLOW LIMITER,
#    then apply divergence (one pass each — deterministic, no sweep-order dependence)
for x-face between (y,x) and (y,x+1):
    v_face = 0.5*(vx[y,x] + vx[y,x+1])
    flux   = v_face * (v_face > 0 ? depth[y,x] : depth[y,x+1])     # donor cell
    flux   = 0 if solid[y,x] or solid[y,x+1]
# OUTFLOW LIMITER (mass-exactness; numerics review): a cell can be donor on up to 4 faces,
# so worst-case outflow = 4 * (v_max*dt/dx) * depth = 2x its depth at defaults -> the
# non-negative clamp would CREATE mass. Per cell: out_sum = sum of its outgoing fluxes;
# if out_sum*dt/dx > depth: scale THAT CELL'S outgoing fluxes by depth*dx/(dt*out_sum).
depth[i] -= (dt/dx) * (flux_out - flux_in)
depth = max(depth, 0); depth = 0 on solid; depth[depth < depth_eps] = 0
```

**The one prototype divergence (deliberate, canon wins):** the prototypes fake walls as
tall terrain (`terrain[walls] = 0.5 / 1.0 / 2.0` — `fluid_test.py:51`,
`fluid_scenarios.py:83`, `fluid_tilted_ship.py:119`), which leaves a standing
away-from-wall push and a depressed rim whose size depends on an arbitrary constant. The
plan uses canon §2.2's Neumann mirror instead: true zero-gradient equilibrium at walls, no
magic constant. Test 2 below asserts the property the prototype scheme would fail.

**Units discipline:** depth/floor/surface m, velocity m/s, dx m, g m/s², damping 1/s;
`atmosphere`/`wave_p` in game pressure units (1.0 = 1 atm), `k_p` m/atm. No tile-unit
constants inside the solver.

**Determinism:** no RNG; gather-then-apply everywhere; fixed iteration order. Float now
(Level-1, same-machine reproducible); engine-wide fixed-point is a later cross-cutting pass.

**Tests** (`tests/test_water_solver.py`; sys.path header copied from
`tests/test_fire_feedback.py:29-49`; all totals summed with `dtype=np.float64`):
1. **Mass conservation** — sealed 32×32 box, deterministic lumpy init (e.g.
   `0.3 + 0.2*sin(row)*cos(col)` — no RNG), 1000 steps at dt=16 ms: total conserved to
   1e-4 relative. Non-vacuity guard: assert the field actually evolved
   (`std` changed) per the `test_fire_feedback.py:323-334` pattern.
2. **Levelling + wall flatness** — dam-break in a box settles: depth variance over WET
   cells → ~0; over a sloped `floor_height` the settled SURFACE (floor+depth) is flat on
   wet cells; and a settled pool's depth is flat **up to and including the wall-adjacent
   column** (the mirror-BC property; the prototype's tall-terrain scheme would fail this).
3. **Containment** — a 1-tile interior wall between a full and an empty chamber: empty
   side stays exactly 0 for 500 steps; no depth ever on solid.
4. **Tilt slide** — uniform depth, constant `tilt_x`: mass migrates low-side, total
   conserved; zero tilt on a settled flat pool → bit-stable (two more steps change nothing
   beyond depth_eps snaps).
5. **Stability hammer** — 64×64, checkerboard 0 / 2.0 m columns, dt = 0.1 s (≈ 3× the CFL
   bound — clamps+limiter must keep it sane, NOT an endorsement of that dt), 500 steps:
   all finite, depth ≥ 0, `depth.max() ≤ 2.2` (slack over the initial max: the limiter
   caps each donor's outflow, but a receiver fed by 4 donors can transiently pile up —
   bounded-not-monotone is the guarantee), total conserved to 1e-3 rel, and `depth.std()`
   at step 500 < at step 0 (it settles, not merely survives).
6. **Null fields** — `floor_height=None` ≡ explicit zeros (bit-identical run);
   `atmosphere=wave_p=None` ≡ zeros.
7. **Head gating** — `k_p=0` + wild-but-FINITE `wave_p` is **bit-identical** to
   `wave_p=None` (the C++ gate makes this exact). `k_p=0.5` + spatially-UNIFORM pressure ≈
   `k_p=0` to `atol=1e-5` over 100 steps (NOT bit-exact — float `(a+c)−(b+c) ≠ a−b`).
8. **Determinism** — two identical 200-step runs `np.array_equal`, with the non-vacuity
   guard.
9. **Outflow-limiter conservation** — single 2.5 m column on a dry plane (the worst-case
   4-face donor), 200 steps at dt = `max_dt()`, run with `depth_eps = 0` (this test
   targets the limiter; the snap is a designed eps-scale sink that would eat the budget):
   total conserved to 1e-5 rel, never negative anywhere.

---

## Step W2 — GameMap fields, config, tick insertion, sources, debug (W2a sim / W2b keys+overlay)

**GameMap** (`src/simulation/gamemap.py` `__init__`, in the allocation block at :81-241):

```python
self.water_depth  = np.zeros((h, w), dtype=np.float32)   # metres on floor — THE shared field
self.flow_vx      = np.zeros((h, w), dtype=np.float32)   # m/s
self.flow_vy      = np.zeros((h, w), dtype=np.float32)
self.floor_height = np.zeros((h, w), dtype=np.float32)   # OPTIONAL terrain, flat-zero default (canon §2.1/§3)
self.tilt_x       = 0.0                                  # radians — ship tilt input (gameplay writes)
self.tilt_y       = 0.0
self.tile_size_m  = float(level_data.tile_size_m)        # REQUIRED dataclass field (level_loader.py:52;
                                                         # loader default 0.333 — do NOT add a second default here)
self.water_sources = []                                  # [(y, x, level_m)] continuous pipe/breach holds
```

`destroy_wall` needs **no** water hook (verified: water has no analogue of the
atmosphere's anti-vacuum-pulse fill; a released tank just flows next step). `gmap.solid`
is static walls-only (`permeability <= 0`, `gamemap.py:306`; `stamp_units` never touches
it) — units do not block water, as intended.

**FieldEdit registration** (`field_edit.py:188-194` policy table):
`"water_depth": _FieldPolicy("float", (0.0, float("inf")), _skip_solid)` — the clamp is a
two-sided tuple (`field_edit.py:325`). Event-shaped water writes (tank dump, scripted
flood) go through the queue; the continuous pipe source is a per-tick hold in the runner
(`depth = max(depth, level_m)`), same architectural slot as `wave_source` feeding.

**Config** (`config.toml`, new section after `[physics.fire]` at :65). All keys are
**restart-bound** (engine/12 §5: PhysicsRunner binds once at construction; Ctrl+R does NOT
re-bind solver params — `input_handler.py:90-94`):

```toml
[physics.water]
g = 9.81          # m/s^2
damping = 1.0     # 1/s pipe friction (prototype-validated: fluid_test.py used 1.0)
v_max = 8.0       # m/s velocity clamp (paired with the C++ outflow limiter)
depth_eps = 1e-5  # m dry snap
k_p = 0.0         # pressure head m/atm — OFF until W4 (real 10.3 is numerically AND feel-wise too hot)
ceiling_h = 2.5   # m air column (W3 displacement + the solver's h_ref CFL reference)
ratio_cap = 1.5   # max per-tick compression ratio (W3)
flood_eps = 0.05  # m remaining air column below which a cell counts flooded (W3)
boil_rate = 0.02      # m/s flash-boil sink in near-vacuum (W5)
boil_p_thresh = 0.3   # atmosphere below this boils (W5; pressure-keyed — drained rooms boil too, intended)
steam_yield = 4.0     # white_smoke density per metre boiled (W5)
```

(No `substeps` key — the count is derived, house-style: `n = ceil(sim_time / water.max_dt())`,
which gives 2 at 24 tps. `boil_*`/`ratio_*` keys pre-declared here are unused until W3/W5 —
harmless.)

**PhysicsRunner** (`src/simulation/physics_runner.py`):
- `__init__`: `self.water = bp.WaterSolver()`; bind via a **`_bind_water_params()`**
  method (so a future reload hook can re-call it), using the `getattr(CFG.physics,
  "water", None)` + `_fp(key, default)` pattern (`physics_runner.py:102-127`) with
  module-level `WATER_*` fallbacks. `dx` lazy-binds on first step from `gmap.tile_size_m`.
  Scratch: `self._water_depth_before = None` (lazy, `_fire_scratch_*` pattern
  `physics_runner.py:181-183, 368-376`).
- `step()`: factor the whole water block into **`self._step_water(gmap, sim_time)`** —
  called once per tick right after `cast_fire_heat(gmap)` (:214), BEFORE
  `dt = self.atmos.max_dt()` (:216). Factoring it is the dormancy-test enabler
  (monkeypatchable). Inside:

```python
def _step_water(self, gmap, sim_time):
    if self._water_depth_before is None or shape mismatch:
        # first call: alloc AND seed with CURRENT depth -> level-painted water is
        # "pre-existing" (no tick-1 compression spike). A FieldEdit dump landing on the
        # very first physics tick is absorbed by this seed too — same semantics, intended.
        self._water_depth_before = gmap.water_depth.copy()
        # lazy dx/centre bind: self.water.dx = gmap.tile_size_m
    before = self._water_depth_before
    if not gmap.water_sources and not gmap.water_depth.any() and not before.any():
        return                                          # dormant early-out (dry ship costs ~one .any())
    for (y, x, lvl) in gmap.water_sources:              # continuous holds (counted vs `before`)
        gmap.water_depth[y, x] = max(gmap.water_depth[y, x], lvl)
    wdt_max = self.water.max_dt()
    n = max(1, math.ceil(sim_time / wdt_max))
    for _ in range(n):
        self.water.step(gmap.water_depth, gmap.flow_vx, gmap.flow_vy,
                        gmap.floor_height, gmap.atmosphere, gmap.wave_p,
                        gmap.solid, sim_time / n, gmap.tilt_x, gmap.tilt_y)
    # W5 boil sink goes HERE; W3 displacement accounting goes HERE (reads `before`),
    # then closes the loop:  np.copyto(before, gmap.water_depth)
```

  **Snapshot semantics (numerics-review fix):** `before` is the depth at the END of the
  previous tick's water accounting — NOT a copy taken this tick — so FieldEdit dumps
  (flushed at `simulation.py:626`, before physics) and source holds are all counted by
  W3's displacement exactly once. Verified tick order: `stamp_units` (:616) → FieldEdit
  flush (:626) → `physics_runner.step` (:632) → burn-through sweep → burst sweep; nothing
  rebuilds `dyn_permeability` after stamping until next tick, so in-runner mask edits
  (W3) survive the whole tick.

**W2b — debug keys + overlay** (`src/input_handler.py`, `renderer/overlays.py`,
`renderer/game_renderer.py`; free keys verified: C, L, M, N, O, P, U, V, X, Y, Z):
- **U** — pour 0.2 m under cursor. Write `gmap.water_depth` DIRECTLY (the I-ignite /
  J-gas precedent, `input_handler.py:102-104, 237`: direct writes land while paused; the
  FieldEdit queue only flushes in an unpaused step). The FieldEdit path is exercised by
  pytest instead (below).
- **O** — water overlay: depth-scaled blue tint via the `FieldOverlay` pattern
  (`overlays.py:23-35`, `HeatFieldOverlay` precedent :115, wiring
  `game_renderer.py:124-139, 294-326`).
- **P / Shift+P** — nudge `gmap.tilt_x` ±2° (feel the Titanic slide; clamp ±20°).
- Gate for W2b: full suite green + `tests/test_main_smoke.py` passes (the only automated
  net over `main.py`); visual check is Erik's morning eyeball.

**Tests** (`tests/test_water_integration.py`):
1. **Dormancy, house pattern** (per `test_temperature_ignition.py:201-223` +
   `test_multigas_structure.py:299-318`): (i) full `Simulation` on the test vessel, fixed
   seed, 5 dry ticks → `water_depth`/`flow_vx`/`flow_vy` stay exactly zero;
   (ii) monkeypatch `sim.physics_runner.water.step` to a raiser → 5 dry ticks raise
   nothing (the early-out is really taken); (iii) A/B: two same-seed 60-tick rollouts,
   one with `_step_water` monkeypatched to a no-op — signature tuple (atmosphere, wave_p,
   gas, fire, temperature, dyn_permeability) `np.array_equal`.
2. **Source spread** — 9×9 sealed room, source (4,4,0.5): after tick 1
   `depth[4,4] ≥ 0.4`; by tick 200, ≥ 90% of open tiles `> depth_eps`; total water rose
   then plateaus (Δtotal over the last 20 ticks < 1e-3).
3. **FieldEdit** — queued ADD then REMOVE on `water_depth` flushes correctly, clamps at 0,
   skips solid.
4. **dx binding** — after one step, `water.dx == gmap.tile_size_m`.
5. **Runner conservation** — sealed room, no sources, water painted in: total conserved
   (float64) across 100 ticks.

---

## Step W3 — volume displacement (water → atmosphere)

**Where:** inside `_step_water`, after the substeps (and after W5's boil, once that
lands), reading `before` (see W2 snapshot semantics), before the IMEX loop.

```python
free_before = np.maximum(ceiling_h - before,           flood_eps)
free_after  = np.maximum(ceiling_h - gmap.water_depth, flood_eps)
ratio = np.clip(free_before / free_after, 1.0/ratio_cap, ratio_cap)
np.multiply(gmap.atmosphere, ratio, out=gmap.atmosphere)    # isothermal P·V = const
flooded = free_after <= flood_eps
gmap.dyn_permeability[flooded] = 0.0
np.copyto(before, gmap.water_depth)                          # close the accounting loop
```

Dry cells get ratio == 1 automatically; receding water → ratio < 1 → inrush (canon §5.1
symmetry). A cell that stays fully flooded has both sides clamped to `flood_eps` →
ratio == 1 → its pressure is frozen under the water, correct. The cap deliberately leaks
P·V on ceiling-slams (1.5 per tick vs the true 50×) — stability-over-bookkeeping,
documented; diffusion re-equalises.

**Mechanism (contracts-review correction):** `dyn_permeability == 0` seals a cell by
**face-flux blocking** — the wave Laplacian, IMEX diffusion, and wind gradient all gather
faces as `min(perm[self], perm[n])` (`atmosphere_solver.cpp:62-65, 164-175, 271-279`).
The hard-zero wave BC keys off `obstacles`/`is_wall`/`is_vacuum` and does NOT see flooded
cells: trapped `wave_p` inside a flooded cell decays via damping, it is not zeroed. The
functional claim (flooded corridors block airflow and smoke) holds. Do NOT test
`wave_p == 0` on flooded tiles.

**Tests** (`tests/test_water_displacement.py`):
1. Sealed single-cell column, raise depth 0 → 0.5 in one tick (FieldEdit dump):
   pressure × exactly `2.5/(2.5−0.5) = 1.25` (cap doesn't bite below d = 0.833); lower it
   back → returns within float tolerance.
2. Ceiling-slam (depth → 2.5 in one tick): ratio capped at `ratio_cap`, no inf/NaN.
3. Flooded line across a corridor: smoke/gas stays one-sided over 50 ticks; wind across
   the flooded faces is zero.
4. **Wet-static exactness**: settled flat pool, sealed box: `free_before == free_after`
   → ratio is IEEE-exactly 1.0 → A/B tick (displacement no-op'd vs live) bit-identical
   atmosphere. Plus the dry case via W2 test 1(iii).

---

## Step W4 — pressure head ON (atmosphere → water)

Set `k_p` from config (start **0.5**; physical 10.3 is excluded both by feel — `wave_p`
rings ~1 s vs a real blast's ms — and by numerics). One plumbing change: `max_dt()`
already includes the head margin (W1). **The tuning session must know the numeric
ceiling** (numerics review): the head stiffens the restoring force,
`c_eff = sqrt(g·h·(1 + k_p·P/free_h))` — near-flooded cells are the worst case; at
k_p = 0.5, P = 1, free_h = 0.2 m, h = 2.3 m: c_eff ≈ 8.9 m/s, CFL ≈ 0.56 at the derived
substeps — marginal-but-clamped churn, watch it in the demo. Erik's research flag
(2026-06-09) = this tuning session.

**Tests** (`tests/test_water_pressure_head.py`):
1. Uniform pressure ≈ no-op (`atol 1e-5`, 100 steps — see W1 test 7 for why not bit-exact).
2. Gaussian `wave_p` bump on a pool: centre depth drops, an outward ring forms, total
   conserved (float64).
3. Sustained low pressure at one end of a flooded corridor: net flux toward it.

The `wave_source` "whoosh" stays deferred (canon §5.1).

---

## Step W5 — flash-boil vacuum sink + steam puff

**Where:** inside `_step_water`, after the substeps, BEFORE the displacement accounting
(so a boiled-off column reads as receding water → slight decompression — physically the
right sign, magnitude ~0.03%/tick, negligible either way).

```python
boiling = (gmap.atmosphere < boil_p_thresh) & (gmap.water_depth > 0)
boiled  = np.where(boiling, np.minimum(gmap.water_depth, boil_rate * sim_time), 0.0)
# (np.where, NOT np.minimum(where=...) — the where-without-out trap leaves garbage)
gmap.water_depth -= boiled
steam_idx = gmap.gases.name_to_id["white_smoke"]        # gases.py:95; never hardcode the index
gmap.gas[steam_idx] += (steam_yield * boiled).astype(np.float32)
```

Direct array rule in the runner (a continuous sim rule like vacuum drain, not an event —
no FieldEdit). The fire-side evaporative heat-sink (wet tiles stay cool) is the OTHER
side's consumer — explicitly out of scope here (see the interface note up top). Agreed
2026-06-10 (07_notes_from_claude.md Answers): their evaporation branch decrements
`water_depth` (a second own-tile writer, runs later in the tick — no conflict) and emits
white_smoke using the SAME `[physics.water] steam_yield` constant, so heat-boil and
vacuum-boil steam consistently.

**Tests** (`tests/test_water_boil.py`): single wet tile, `atmosphere = 0.0`,
depth 0.1 m, `boil_rate = 0.02`, tick = 1/24 s: after one tick depth
`== 0.1 − 0.02/24` (atol 1e-7); white_smoke gain `== 4.0 · 0.02/24` (atol 1e-6); after
121 ticks depth == 0.0 exactly; a twin tile at `atmosphere = 1.0` is bit-exact unchanged;
dormancy (no water → no gas writes).

---

## Step W6a — ripple field (sim, visual-only) … W6b — water rendering

**W6a sim** (`WaterSolver::step_ripple`, same class; fields `gmap.ripple`,
`gmap.ripple_v` float32; tunables `[physics.water] gamma_r = 2.0, h_cap = 0.25,
k_amp = 0.5, k_splash`):

```cpp
c2[i]     = g * min(depth[i], h_cap);                  // m^2/s^2 — deep-water splice (canon §6)
lap       = (r[N]+r[S]+r[E]+r[W] - 4*r[i]) / (dx*dx);  // the 1/dx^2 is REQUIRED: c2 is in
                                                        // SI units, unlike the atmosphere's
                                                        // tile-unit wave — do not copy-paste
ripple_v += dt * (c2 * lap - gamma_r * ripple_v);
ripple   += dt * ripple_v;
ripple    = clamp(ripple, -k_amp*depth[i], +k_amp*depth[i]);   // applied AFTER the drift;
                                                                // gamma_r eats clamp-injected energy
ripple = ripple_v = 0 where depth == 0 or solid;
```

Form: keep canon's `c²Δr` (the conservative `∇·(c²∇r)` is the upgrade if shoaling
amplitudes ever look wrong — same cost, face-sampled c²). Static
`ripple_max_dt = 0.5·dx/sqrt(g·h_cap)` = 106 ms ≫ tick → one call per tick, derived once.
**Splash source:** after the IMEX loop, `ripple_v += k_splash · wave_p` on wet tiles.
Runner placement: `step_ripple` AFTER the atmosphere substeps (it reads the fresh
`wave_p` splash; it feeds nothing back — locked visual-only).

**W6a tests** (`tests/test_water_ripple.py`): decay in a still pool; zero on dry/solid;
point splash front ≥ 3 tiles out after 1.0 s (c_cap = √(9.81·0.25) ≈ 1.57 m/s ≈ 4.7
tiles/s) and `|ripple| < 1e-7·splash_amp` beyond `c_cap·t + 2` tiles — NOT exact zero:
an explicit stencil's numerical domain of dependence grows 1 tile/step, so far-field
precursors are nonzero-but-negligible on wet tiles (exact zero holds only on dry/solid,
where the zeroing rule applies); clamp holds
(`|ripple| ≤ k_amp·depth` everywhere); **the visual-only guarantee**: 60-tick A/B rollout
with rippling on vs off — every transport field (`water_depth`, `flow_v*`, atmosphere,
gas, fire, temperature) `np.array_equal`.

**W6b render** (`renderer/overlays.py` water overlay v2 + ambient sines): depth-blue
tint + brightness from `∂ripple/∂x` (cheap pseudo-normal) + foam (white) where
`|∇ripple|` > threshold + shader-side ambient sines, amplitude = base + local ripple
energy. Gate: suite green + `test_main_smoke.py` + **Erik's eyeball** (morning). The real
optics pass stays the canon §6 research item.

---

## Out of scope (this plan)

Fire-side evaporative heat sink (their lane — interface note up top) · conduction (§5.2) ·
oil (§5.3) · ice ↔ water (§5.4, post-temperature-tuning) · rain/snow research · movement
penalty (§5.5) · the whoosh · per-pixel art-resolution refinement · CUDA.

## Build log (2026-06-10)

All steps shipped green, per-step commits: W1 `3ebf62b` (237 tests) → W2a `aea3956` (244)
→ W3 `194d383` (249) → W4 `00c67b3` (253) → W5 `1f104fa` (257) → W2b `8789567` (262) →
W6a `ae98284` (268) → W6b `cb229c7` (277). Honest in-flight corrections by the
implementer agents (each documented in the test files): the W2 spread plateau is at tick
~260–320 (the plan's 200 was a pre-run estimate, re-pinned again when W4's head shifted
it); the W6a far-field bound shipped as 1%-of-splash at `c_cap·t + 2` (the plan's
1e-7·splash_amp first holds ~+6 tiles — the stencil tail decays ~a decade per tile);
pybind methods are read-only, so dormancy stubs swap whole solver objects / runner
methods; the W3 flooded-corridor test uses a basin floor (a flat-floor full-depth line is
not an equilibrium — and with `k_p` on, a sustained pressure step measurably shoves a
near-flooded plug, the live specimen of the W4 churn-zone note). `k_p = 0.5` and
`k_splash = 2.0` are LIVE and await the feel session.

## Review log

- **R1 numerics (2026-06-10):** wall-BC prototype divergence documented + test; real CFL
  pinned (`max_dt()`, derived substeps); per-cell outflow limiter added (mass-exactness);
  k_p numeric ceiling formula recorded for the tuning session; damping default corrected
  2.0 → 1.0 (prototype-lifted); persistent `before` snapshot (FieldEdit dumps counted);
  boil rationale sign fixed; ripple laplacian 1/dx² pinned; `c²Δr` form kept.
- **R2 contracts (2026-06-10):** all integration points verified real (no blockers);
  flooded-seal mechanism corrected to face-flux blocking (no `wave_p == 0` test);
  FieldEdit two-sided clamp; pour key writes directly (paused-flush gotcha); restart-bound
  config documented (+`_bind_water_params` hook); `tile_size_m` loader default 0.333;
  free keys U/O/P; injection order verified against `simulation.py:616-658`.
- **R3 test/build (2026-06-10):** dormancy tests rewritten to the house pattern
  (`_step_water` factoring + inert/raiser/A-B trio); W3 dry-ratio test replaced
  (wet-static exactness); W6 split a/b (render eyeball-gated); weak tests tightened with
  concrete numbers; float64 sums; non-vacuity guards; `np.where` boil fix; uniform-head
  bit-exact claim demoted to allclose + C++ k_p gate.
