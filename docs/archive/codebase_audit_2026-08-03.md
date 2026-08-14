# Breach codebase audit — full overview (2026-08-03)

**Status: read-only audit. Nothing was changed.** Commissioned by Erik
2026-08-03 ("go over the codebase, see how well written it is… without changing
anything — just noting where we should put the effort"). Six parallel read-only
passes over ~122k lines: `src/` Python core, the C++ engine, CUDA + bindings,
`tools/` + `renderer/`, `tests/`, and one cross-cutting hunt for duplicated
constants and unit conventions. Findings are cited `file:line` throughout the
per-area reports quoted below; this document is the synthesis.

---

## THE HEADLINE

**Erik's hypothesis — "the Python files are probably worse than the C++/CUDA
ones" — is not supported.** The Python simulation core grades **B+** and the
renderer **A−**, against **A−** for the C++ engine. The two *weakest* areas are
`bindings.cpp` (**C+**) — which is C++ — and `tools/` (**B−**).

The real predictor is not language. **It is whether the code sits under a
digest/golden gate.** Everything the gates watch is excellent in any language.
Everything outside them — the pybind argument boundary, the measurement
harnesses, the places where a constant is transcribed rather than shared — is
where the rot is. That is the finding worth acting on.

| Area | Grade | Lines | Character |
|---|---|---|---|
| C++ engine (`cpp/src/*.cpp,*.h`) | **A−** | ~16k | Research-grade numerics; overflow budgets *derived*, not assumed; ~1:1 comment-to-code. Weakness is the perimeter, not the math. |
| CUDA kernels (`*.cu`) | **A−** | ~7.1k | Best-documented CUDA in the tree; every kernel names its CPU twin by file:line; determinism argued per kernel. |
| `renderer/` | **A−** | 6.3k | Cleanly layered; every primitive has a pyray-free, unit-tested core. |
| `src/` simulation core | **B+** | 18.5k | Reads like a numerics library by someone burned by nondeterminism. Fails by *archaeology*, not sloppiness. |
| `tests/` | **B+** | ~large | Determinism engineering of unusual craft — undermined by what it structurally cannot see. |
| `tools/` | **B−** | 16.9k | Strong methodology, weak reproducibility. Editor family A−; fire benches B; one file cannot execute. |
| `bindings.cpp` | **C+** | 3.1k | Organised, well-commented, and carrying total validation debt: zero shape, dtype, or contiguity checks in 3141 lines. |

---

## 1. The five cross-cutting themes

### T1 — Constants transcribed instead of shared (the biggest class)

Found independently by four of the six passes. The two-Kelvin-map problem that
started this is one instance of a systemic pattern.

- **The two Kelvin maps.** Radiation/render: `K = 293 + 2·T_game`, hardcoded as
  `297 + 8t` (`raycaster.cpp:59`), again as config keys
  (`config.toml:844-845`), and reimplemented in four test files and two tools.
  EOS: `T_abs = T_game + 290` (`eos_solver.cpp:542`) — different offset **and**
  different slope. *This is a physics decision for Erik, not a cleanup* — see
  the companion doc `fire_atmosphere_oscillation_analysis_2026-08-03.md`.
- **`T_MAX_PHYS = 16000`** in six places, three wired independently from config
  so they merely *happen* to agree — plus an implicit seventh: the E° table's
  `E_TABLE_SIZE = 4000 × 4` game units. Raise it in config and the radiation
  table silently saturates, **and** the int32 overflow proof at
  `raycaster.h:270` becomes false without a word of warning.
- **Ten constants re-declared across the CPU/CUDA seam** (`WSUM_FLOOR_Q`,
  `WSUM_EPS_Q`, `M_CAP`, `M_CAP_L`, `X_N_FLOOR`, `TILT_MAX`, the MG level cap…).
  The sharpest is the MG level cap: a bare literal `9` at `eos_solver.cpp:848`
  vs a named constant in two `.cu` files. Edit one, get a 10-level CPU pyramid
  and a 9-level GPU one, silently.
- **`FP_ONE = 65536` redefined ~15× in tests** and inconsistently (`1 << 16`
  here, `_q` there, `_quantize` elsewhere) — each a Python reimplementation of
  `fixedpoint::quantize`'s rounding. A divergence makes a *gate* quietly wrong
  rather than red. The test auditor rates this worse than the Kelvin one.
- **Ambient O2 `0.21` in six places**; the material-id table copy-pasted into two
  benches (and it has already grown once, `MAT_DOOR_CLOSED = 7` inserted mid-table).

**On the test side the same class has already produced a real defect.** An
exhaustive sweep of `tests/` for reimplemented constants found:

- **`test_thermal_mass_axis.py:646` uses `int(0.21 * FP_ONE)` = 13762 where
  `quantize_scalar(0.21)` = 13763** — truncation instead of round-half-away.
  13762 + 51773 = **65535**, so this fixture silently violates the exact
  `N_amb == FP_ONE` invariant that `test_eos_p1_calibration.py:55-62` exists to
  pin. Every other O2 fixture in the suite uses `quantize_scalar`. This is the
  concrete form of the "a divergent copy makes a *gate* quietly wrong rather
  than red" hazard.
- **`test_pr3_capacity_law.py:65-66` pins `k_grow = 3.5`, `k_die = 0.035`** —
  values config has *not* moved to (4.0 / 2.0). The comment calls 0.035 "P-R5's
  blessed-config move": the test is pinned to a future state.
- **`cuda_po2b_check.py:57` sets `burn_rate = 1.0` against a shipped 0.02** (50×,
  no comment) — in the gate that never runs (T3).
- **`dx` is spelled two different ways** across the suite — `1.0/3.0` (24 files)
  and `0.333` (11 files) — which are not the same float, sometimes in the same
  file.
- The radiation dial block is duplicated in **four** places and its `color`
  tuple has gone stale in all four (`(1.0, 0.6, 0.2)` vs config's
  `[1.0, 0.45, 0.12]`).

**The project already knows the answer** and applies it in the newest code:
`raycaster.h`'s `RC_HD` host/device shared helpers, and `cuda_combustion.cu`
uploading the canonical draw tables straight from `combustion.h` via
`cudaMemcpyToSymbol` *with the explicit rationale that transcription drift is
"structurally impossible."* The pattern exists; it just hasn't been applied
backwards.

**The cheapest structural fix already exists in the codebase.** For three of six
solvers, `physics_runner.py` uses the `_ep`/`_cp` idiom where the **C++ struct
member itself is the fallback** (`self.eos.<attr>`), so Python-vs-C++ drift is
*structurally impossible*. Every EOS/combustion/temperature dial matches. The
solvers using hand-written literal fallbacks (`_fp` — fire, water, atmosphere)
are exactly where all the drift is. Converting them is near-zero cost and
permanently deletes the whole drift class.

**Live-path drift worth knowing about now** (each is latent — the config key
exists, so the fallback only fires if a key is deleted or mistyped, and
`config.py` has **no schema, no validation, and no unknown-key rejection**, so a
typo is silent):

| key | config.toml | code fallback | if the key goes missing |
|---|---|---|---|
| `physics.water.k_p` | 0.5 | **0.0** (`physics_runner.py:118`) | **water pressure head silently OFF** |
| `physics.max_source_per_step` | 10.0 | **0.5** (`:181`) | blast source energy cut 20× |
| `physics.combustion.draw_r` / `max_claimants` | 2 / 12 | **1 / 4** (`fire_o2_supply_baseline.py:243-244`) | the supply bench silently measures the retired 4-face law |

The highest-leverage variant: a mistyped **section** name in
`getattr(CFG.physics, "<section>", None)` (10 sites) yields `None`, and every
dial inside it falls back at once.

**Two calibration-anchor gaps found in passing, both fire-relevant:**
- `rad_scale`, `H_BED_M/SHIFT`, `I_cap_per_avail` and `T_emit_gate` were all
  derived at **`cool_shift = 9`** (`config.toml:317,602`) — but every shipped
  painted material ships **5**. Since `T* ∝ 2^(cool_shift − his)`, the shipped
  game runs its fire calibration **16× off its own anchor**. It is flagged
  pending ("RE-TUNE AT P-R5"), but the anchor is not written where the value
  lives. (P-F1b moves these to 13, which is why it had to re-solve H_bed.)
- `levels/airlock_demo/level.toml:38` ships **`tile_size_m = 1.0`** while
  `rad_scale` (0.833 m² face), `burn_rate` (ceiling_h anchor) and
  `RADIATION_RANGE` (tiles) are frozen scalars assuming 0.333 m. Canon
  ch.01:55-58 explicitly forbids this: *"a solver that assumes 1/3 m is silently
  wrong at any other resolution."* Weapons rescale; fire does not.

### T2 — Load-bearing comments that are now false

The comments here carry real weight (that is a strength), which makes drift
expensive:

- `eos_solver.h:453` claims the P6.4 CPU reference "replays **EXACTLY**"
  `step()` — it has been missing the `sponge_udamp` band since B3c, so **the
  gate structurally cannot cover the planetside/ambient path**, and a lockstep
  failure there would blame the GPU.
- `cuda_smoke.h:17` says sink-hop "runs on the GPU" — the **CPU twin was
  deleted**; ~150 lines of orphaned kernel still compile and ship.
- `renderer/lighting.py:255` says nothing reads `gmap.heat` — false since P-R4;
  the temperature solver and `exchange.py:299` both do. This one guards the
  render→sim write invariant, so the comment a future patch reasons from is
  wrong about the one thing that matters.
- `physics_engine.cpp:596-603` describes a function that no longer exists;
  `:614` still claims `/fp:precise` where CMake says `/fp:strict`.
- `Simulation.reset()`'s docstring promises solver/grid reuse; it reallocates
  the world (and an ~80 MB recorder). That docstring is the contract an RL
  author will trust for the episode loop.

### T3 — Gates that cannot see what they claim to gate

- **The canonical golden scenario contains no flammable tiles.** Stated twice in
  the file that owns it (`_xarch_perfield_digest.py:92,106`). So *every*
  combustion, fuel, or ignition formula change preserves the golden **by
  construction** — during the month the fire arc has been changing exactly that.
- **Two CUDA parity gates are never run**: `cuda_po2b_check.py` (the shipped
  `draw_r = 2` extended O2 draw, incl. the persistent `dem_acc` plane) and
  `cuda_sky_exchange_check.py` have no pytest wrapper. They exist, they are
  correct, and they do nothing.
- **`cuda_fire_check.py:170` compares sets**, so it cannot see the confirmed
  destroyed-wall ordering divergence below.
- **The sanctioned golden lives in a file whose own docstring calls it a
  "THROWAWAY diagnostic"**, and the only CPU-side golden gate in the suite is
  buried in a *weapons* test file. The literal is duplicated 12×.
- **No CI exists.** Every CUDA gate runs only when a human runs `pytest tests -q`
  on a box with a CUDA build; on a CPU-only checkout all 22 silently skip.

### T4 — Validation debt at the boundaries

- `bindings.cpp`: **zero** shape checks, **zero** contiguity checks, **zero**
  dtype guards across 3141 lines. `h`/`w` are taken from the first array and
  every other array's extracted dimensions are discarded (~35 sites). A
  wrong-dtype *output* array is silently copied by pybind's `forcecast`, so all
  writes vanish with no error; a non-contiguous view is indexed as if it were
  contiguous. Thirteen raw device pointers are accepted as `uintptr_t`
  **defaulting to 0**.
- **Runtime shift counts are unclamped** in C++/CUDA (`temperature_solver.cpp:329,522`,
  `raycaster.h:235`). Python validates; the kernels don't. And x86 (`sar`, masks
  to `s & 31`) and PTX (`shr`, clamps) resolve out-of-range shifts *differently*
  — a CPU/GPU divergence waiting on one bad config value.
- **One reachable divide-by-zero**: `eos_solver.cpp:308` divides by `t_amb_q`
  with no floor, and `T_AMB_K` is writable from Python. Every other divide in
  that file is floored.
- **Load-bearing invariants asserted only in prose**: the radiation int32
  overflow proof depends on `heat_inv_shift ≤ 5` (2.4% margin) and nothing
  enforces it — add a `thermal_mass = 64` material and it wraps silently.

### T5 — Reproducibility of the science

`tools/` measures well and records poorly. **No bench stamps a git commit, a
timestamp, or a config hash into any artifact** — yet
`OPERATING_POINT_I = 0.192`, a load-bearing constant, is sourced from a CSV in
the untracked `_fire_tuning_artifacts/`. `fire_room_bench.write_room_csv` has the
dial set in hand (`m['overrides']`) and drops it. Against that, the methodology
itself is genuinely strong — the pin-I law-agnostic draw measurement
(re-running the real combustion pass on settled state and reverting every plane)
is the best instrument design in the tree, and the benches repeatedly **report
findings that contradict the design doc rather than smoothing them over**. That
culture is the most valuable thing here; it just needs provenance headers.

---

## 2. The things I would fix first

Ordered by (impact × certainty × cheapness). Everything in tier 1 is
behaviour-preserving and gate-safe.

### Tier 1 — cheap, certain, no digest movement

1. **`fire_tune_loop.py` cannot run** — it still passes the retired
   `k_fire_heat`, so `_resolve_key` raises `KeyError` before anything executes.
   Erik's primary manual tuning loop is a brick, and it is the tool that draws
   the intensity/temperature/room-O2 panels he asked for last night. Delete one
   dict entry. *(1 minute.)* While there: the scorecard also prints two
   confident predictions derived from dials nothing reads.
2. **Close the cross-toolchain determinism hole** — `cpp/CMakeLists.txt:139-153`
   wraps the `/fp:strict` per-source override in `if(MSVC)`, and the non-MSVC
   branch applies **`-ffast-math` globally**. The file's own comment says every
   sim TU is strict; that is true only on Windows. Every `quantize((double)…)`
   boundary in the live EOS path is exposed on a gcc/clang build. *(~1 h.)*
3. **Run the two gates that already exist** — add pytest wrappers for
   `cuda_po2b_check.py` and `cuda_sky_exchange_check.py`, plus a meta-test that
   every `cuda_*_check.py` is referenced by some `test_*.py`. This closes the
   largest genuine hole in the suite using gates already written. *(~30 min.)*
4. **Sort the destroyed-wall list** — `cuda_fire.cu:483` returns it in atomic
   arrival order; the CPU returns row-major; `destroy_wall` is order-dependent
   (it writes `breach_mask` that the next iteration reads). This is a
   **confirmed CPU≠GPU *and* GPU≠GPU divergence**. Three lines, plus changing
   the gate from set-comparison to list-comparison. *(~30 min.)*
5. **Fix the drifted P6.4 reference** (`eos_solver.cpp:1345`) so the ambient path
   is actually gated, and **poison the resident digests** so a stale read is
   loud instead of plausible. *(~1 h.)*
6. **Make `apply_overrides` atomic** (`fire_timing_harness.py:144`) — today a
   `KeyError` mid-loop leaves CFG half-patched with no handle to undo it, which
   is precisely how item 1 poisons a whole sweep process. *(15 min.)*

### Tier 2 — a day each, high value

7. **Vectorize `find_burst_walls`** (`gamemap.py:1595`) — a pure-Python loop over
   every solid tile × 4 neighbours, plus a full-grid float64 alloc, **every
   tick**. Measured **10–38 ms/tick** at shipping grid sizes and wall fractions
   (`unhcr_vessel` is 37% solid). The entire resident CUDA physics tick is
   27.1 ms. This one loop can more than double the tick and silently eat the
   whole S8a residency win. It is a pure gather; behaviour-preserving.
8. **Kill the weapon-table global** (`weapons.py:671`) — two sources of truth;
   `combat.py` reads the module global, `Simulation` reads its instance. Already
   caused one real cross-test bug (papered over by an autouse fixture in
   `conftest.py`). **It becomes load-bearing the moment two `Simulation`s
   coexist in one process — which is exactly what a vectorized RL env is.**
   Eight call-site edits now; a mystifying debugging session later.
9. **One validation helper in `bindings.cpp`** — `require_2d(arr, h, w, name)`
   asserting ndim, shape, and C-contiguity, wired into `get_2d`/`get_3d` and the
   ~35 discarded-shape sites. Retires an entire class of silent-corruption and
   segfault paths for a few hours.
10. **A fuel-bearing golden scenario** — the prerequisite for fire/combustion
    having any regression protection at all. Most of the work already exists in
    the four `*_gate_a_capture.py` scripts; promoting one to a pinned second
    golden is the job.

### Tier 3 — worth doing, needs its own arc

11. **The units-and-anchors canon page** (see §3).
12. **`p*` precision staging** (`eos_solver.cpp:565`) — the intermediate
    `mul_q16(C, N_total)` carries only ~8 bits, so at ambient a **0.44% change
    in N produces no change in `p*` at all**. A quantization plateau under the
    entire pressure solve. The fix is one wide chain and is overflow-safe — but
    it is digest-moving and needs a CUDA twin plus a written re-baseline. This
    is the most *interesting* finding in the audit and it belongs in an arc.
13. **RAII/`try-catch` for device allocations** — 8 of 14 `.cu` files leak every
    prior `cudaMalloc` on any CUDA error. The correct pattern already exists in
    this repo (`cuda_mg_solve.cu:494-529`); it just needs copying.

---

## 3. On Erik's question: "shall we have a code cleanup day?"

**My recommendation: no — not a line-by-line sweep.** Three reasons. The rot
rate is genuinely low (the Python audit verified ~120 dead lines in 18.5k, and
every substantial piece carries a comment explaining why it is still there). A
broad edit is expensive precisely because the digest/golden gates are good — you
would spend the day re-baselining. And a line-by-line pass optimises for the
wrong defect: what actually hurt us this week was not ugly code, it was **a
fact nobody had written down**.

**What I would do instead, in three named pieces:**

**(a) A "units and anchors" canon page** — `architecture/engine/`, live-edited.
One table: every physical constant, its unit, its real-world anchor (with
citation), its owner, and its blast radius. Add the rule: *a physical constant
without a written anchor is a bug*. The fire constants audit (2026-07-30) already
proved this recipe works — §7b of that doc is what surfaced the render half of
the Kelvin problem. This pass extends the same recipe to the EOS/atmosphere/water
paths. **This is the direct answer to "how shall we address the temperature
functions."** Note the outcome may well be *documentation*, not code: the EOS
slope is arguably a legitimate free parameter, and naming it (`φ_exp`) is the
fix.

**(b) A shared-constants pass, mechanical and gate-safe** — apply the patterns
the newest code already uses, backwards. Three concrete moves, in value order:
1. Convert the `_fp` (fire/water/atmosphere) config binds to the **`_ep` idiom**
   already used for EOS/combustion/temperature, where the C++ struct member *is*
   the fallback. Deletes the entire Python↔C++ drift class structurally.
2. A `[units]` block in `config.toml` owning the game→Kelvin map, referenced by
   both `[render.blackbody]` and the sim — plus a ~10-line gate test that
   recomputes `297 + 8t` from `CFG.units` and compares against the baked
   emissive table. The C++ literal must stay hardcoded (the bake needs exact
   integers), but the coupling becomes loud instead of silent. Note today the
   render dial is even labelled *"(TUNE BY EYE)"* — a designer is invited to
   move a number the radiation physics silently shares.
3. Share the ten re-declared CPU/CUDA constants via `RC_HD`-style headers and
   `cudaMemcpyToSymbol`; fix the test-side `FP_ONE` reimplementations and the
   bench-side material tables.

Plus a ~30-line schema check in `config.py` (reject unknown keys in the physics
sections; a `_REQUIRED` set so a deletion errors instead of silently falling
back). That single addition is what makes every remaining `getattr` safe.

Explicitly **not** worth centralising, and worth *documenting* as deliberate: the
seven `*_fixed.py` `FP_ONE` modules (format constants, deliberately independent,
already guarded by a `static_assert` and every digest gate) and the five
verbatim `K = c_max²/γ` transcriptions (the CPU/CUDA tol-0 contract *requires*
identical transcription; a shared cross-`.cu` helper would be the riskier
change).

**(c) Tier 1 above, as one afternoon.** Six items, all behaviour-preserving,
each independently valuable.

Total: roughly two focused days, versus an open-ended sweep — and it fixes the
class of defect that actually bit us.

---

## 4. What is genuinely good (do not "clean this up")

Stated plainly because it matters as much as the criticism, and because a
cleanup pass is exactly how good work gets damaged:

- **`fixed_point.h` in its entirety** — the rounding contract, the SAR
  `static_assert`, the Newton reciprocal's round-to-nearest narrow (and the
  analysis of why truncation would shave 1 ULP off every SL cell every step),
  the branch-identical isqrt, the exact-by-construction trig kit. The whole
  project rests on this file.
- **The conservation idiom**, held identically across four independent solvers:
  gather flux once as `int64`, apply the *same* narrowed integer ±.
- **The seven `*_fixed.py` boundary modules** and their *deliberate* non-DRY —
  each system names its own quantize helpers so a cross-import can never imply
  two fields share semantics. `unit_fixed.py`'s refusal to provide a Python trig
  fallback ("a second implementation could drift and silently desync") is the
  most mature determinism decision in the codebase.
- **The dormancy discipline** — every new subsystem ships behind a gate that
  makes a level without it byte-identical to before it existed. This is why the
  golden suite still works after ~15 arcs.
- **Non-vacuousness controls in the CUDA gates** — running the GPU *without* the
  new plane and **requiring the comparison to fail**. Rare, and it should be the
  template for every new gate.
- **The render→sim write invariant**, defended at three independent layers with
  the hazard named in prose. The single best piece of discipline in the tree.
- **`test_field_ab_harness.py:55`** — the gate that tests *itself* by injecting a
  mean-preserving perturbation and asserting the harness catches what the old
  signature missed.
- **The honest negative results** in `tools/` — invoking the task order's STOP
  clause, cross-checking against two independent benches, and explicitly
  refusing to hack around the finding. That is the best cultural property here.
- **The tombstone comments** recording what was deleted and why, at the site.
  They are the reason dead code could be told apart from deliberate scaffolding.

---

## 5. Reports

Each area's full report — with every `file:line`, the complete issue tables, the
dead-code inventories, and the per-area "3 things to do first" — was produced by
a separate read-only pass and is summarised above. The six areas: `src/` Python
core; C++ engine; CUDA + bindings; `tools/` + `renderer/`; `tests/`; and the
cross-cutting constants/units hunt (whose full tables land in the units-and-
anchors canon page proposed in §3a).
