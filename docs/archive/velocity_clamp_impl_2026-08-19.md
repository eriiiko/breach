# Velocity-clamp arc — implementation design (P-V1)

**v3, 2026-08-19** — v1 rewritten after the 3-lens adversarial critique
(arithmetic/lockstep, physics/failure-modes, scope/regression); v3 folds in
the focused buildability re-verification (ratio_umax int64 overflow, fold-site
count, prestage plumbing, Python bignum arithmetic). v1's per-cell "compute cap
in the kick from kick-time T" is replaced by a **cap² plane folded at tick
entry**; the rescale is replaced by an exact int64-divide form; the claims
section is re-scoped per the physics lens's dump re-measurement.

Fixes the two defects confirmed by `docs/velocity_clamp_audit_2026-08-19.md`:
(1) the kick's velocity ceiling is one per-tick global scalar from the hottest
cell on the map; (2) the Chebyshev component pre-test skips the magnitude clamp
for diagonal flow up to √2×cap.

## What this patch claims — and what it deliberately does NOT (re-scoped)

CLAIMS: after P-V1, no stored gas velocity exceeds its own cell's sound-speed
cap (±2 raw counts); the diagonal leak is closed exactly; CPU↔GPU lockstep
holds at tol 0; the P6.4 replay contract strengthens (the cap becomes fully
derivable from the replay's own inputs).

DOES NOT CLAIM: that the ≫100× pile-ups or the flashing tiles vanish. The
critique re-measured the seed dump: at the spike ticks the required advection
substep count is ~140 against `N_SUB_MAX = 8` (`eos_solver.h:159` rail binds;
any wind above ~32 m/s already needs n_sub > 8 at dt=1/24, dx=0.333,
CFL_ADV=0.5), and the donor-cell outflow limiter stays saturated even at
capped speeds (re-applying the proposed clamp to the recorded winds leaves
40–53% of cells near the sinks limiter-saturated, vs 49–62% today). The clamp
fix removes the *illegal* velocities; the transport still runs ~14× over its
resolvable Courant number during blasts. **That ceiling — N_SUB_MAX vs the
post-breach sonic regime — is the likely owner of the remaining flashing and
is a separate design decision (perf-coupled) that goes to Erik after P-V2's
measurements.** HUMAN-TEST expectation is set accordingly: spikes attenuated,
not necessarily gone.

Also named, not fixed here (accepted gaps, measured by P-V2):
- **Clamp energy is unbooked**: scale-to-cap removes kinetic energy with no
  ledger counterparty (only `u_clamp_hits` counts events). Pre-existing; the
  per-cell cap makes it bind more often at shock fronts. P-V2 reports the
  energy residual alongside the symptom table.
- **U_MAX becomes reachable per-cell** where hot cells carry fast flow
  (cap ≥ U_MAX needs T_abs/T_amb ≥ (1000/300)² ≈ 11.1, i.e. T ≳ 2930).
  `u_max_hits` — structurally zero today — is the exact detector; P-V2
  reports it.
- **D1's ambient floor** is a no-op today (measured: 0 sub-ambient open cells
  in 4.86M cell-snapshots — 4c cannot cool below ambient until TODO item 2
  lands). After item 2, `T_MIN` admits t_abs → ~1 K where true c ≈ 17.6 m/s
  vs the 300 floor — item 2 inherits this number.

## Erik's locked rulings (2026-08-19, this session)

- **Squares against squares — NO per-cell sqrt.** Compare `ux²+uy²` to `cap²`.
- **Quadratic drag (TODO item 3) is NOT in this patch.** Linear `k_drag = 0.5`
  stays; item 3 keeps its own design session.
- **`n_sub` keeps the global max** (the substep count must satisfy the worst
  cell). Only the velocity ceiling becomes per-cell.
- **HUMAN-TEST gates the merge** (feel-adjacent). Golden re-baseline happens
  once, at arc close, after Erik's PASS.

## Decisions (v2)

- **D1 — per-cell cap floors at ambient c** (`t_abs` floored at `t_amb_q`):
  preserves the old "never below ambient" invariant per cell; avoids
  velocity-freezing rarefaction pockets. Revisit at TODO item 2 (see above).
- **D2v2 — the cap derives from TICK-ENTRY temperature, folded as a per-cell
  `cap2` plane inside the existing per-tick scan (`eos_solver.cpp:401-406`),
  which already visits every cell.** The kick consumes the plane. Why entry-T
  and a plane, not kick-time T (v1): tick-entry T is exactly the `t0` state
  the P6.4 replay and `cuda_kick_check` PART 2 reconstruct from, so the wind
  ground-truth trajectory gate stays exact (kick-time T would have silently
  retired it — critique blocker); the cap shares the T-basis of the global
  `c_local` scan (one scan, one basis); and the property gates become exact
  (no T-drift tolerance). One-tick staleness of the cap is symmetric and
  bounded — the same staleness the global cap has always had.
- **D3 — counter semantics:** `u_clamp_hits` fires on `rad > cap²` (exact
  magnitude test; the old inner `umag > u_cap` re-test collapses into it).
  NOTE (critique): this is a slightly *stricter* trigger than the old
  floor-sqrt path near the boundary, and a cell parked at the cap can
  re-trigger each tick — P-V2 must not read the u_clamp_hits delta as pure
  signal. `cap_is_umax` := `cap2_plane[i] >= u_max2_q32` (see plane spec).
- **D4 — `ts` (thermal-solid) gas cells get the AMBIENT cap** in the plane
  fold: their `temperature[i]` is the *object's* T by ruling A1, not the
  gas's, and reading it would make furniture the one structural path to a
  U_MAX-scale cap (critique). Conservative, matches D1's direction, policy
  lives in ONE place (the scan).
- **D5 — the kick TRUSTS the plane verbatim** (no re-min against U_MAX in the
  kick): the scan owns floor/ts/min policy. This lets tests drive regimes
  directly (e.g. a 2300²-style plane to keep the clamp disengaged, the
  existing `test_p_e3_drag` idiom) and keeps the kernel branch-light.
- **D6 — exact rescale replaces the reciprocal chain** (critique blocker: the
  old `reciprocal_q16` scale lands up to ~0.8% above cap and its signed
  `mul128_shr` floors negative components toward −∞):

  ```c
  ux = (ux * (int64_t)u_cap_q) / (int64_t)umag;   // trunc toward 0 — shrinks
  uy = (uy * (int64_t)u_cap_q) / (int64_t)umag;   // magnitude on BOTH signs
  ```

  `|ux·u_cap| ≤ 2^30·2^26 = 2^56` int64-safe; C++ and CUDA integer division
  both truncate toward zero — bit-identical; overshoot ≤ ~2 raw counts
  (`umag = ⌊√rad⌋ ≥ |u| − 1`). Post-clamp invariant: `|u| ≤ u_cap + 2 counts`.
- **D7 — `u_est` clip widens to `max(c_local_q, u_max_q)`**
  (`eos_solver.cpp:486` + `cuda_eos_step.cu:228`): stored `|u|` may now reach
  U_MAX on hot cells while `c_local` is folded from entry-T — clipping at
  `c_local` alone would under-derive `n_sub` (critique). One line per side;
  still a global scalar (inside Erik's ruling).

## The change, exactly

### A. Plane fold — TWO transcription sites (verified: there is no third)

The fold lives in `eos_solver.cpp:401-406` (extend the existing scan loop)
and in the shared host pre-stage `eos_host_prestage`
(`cuda_eos_step.cu:112-175` — `cuda_eos_resident.cu:685` CALLS it; the
resident path only *uploads*, it does not re-fold). Do NOT create a third
transcription.

```c
// per-tick folds, next to the existing ones (eos_solver.cpp:407 has c_amb_q;
// cuda_eos_step.cu:176 already has it too — eos_host_prestage must ADD a
// u_max_q fold (~:159): const q16 u_max_q = quantize((double)solver.U_MAX);
const int64_t c_amb2_q32 = (int64_t)c_amb_q * (int64_t)c_amb_q;   // Q32.32
const int64_t u_max2_q32 = (int64_t)u_max_q * (int64_t)u_max_q;   // Q32.32
// smallest ratio that rails at U_MAX. HOST DOUBLE FOLD — the K_raw idiom
// (eos_solver.cpp:393 / cuda_eos_step.cu:166): a per-tick scalar, one
// transcription per side, no float in the per-cell path. The naive integer
// form ((u_max2_q32 << 16) / c_amb2_q32) OVERFLOWS int64 at the shipped
// dials (4.3e15 << 16 = 2^68) — do not use it. Value at shipped dials:
// 728178 (≈ 11.11 in Q16.16).
const double ru = (double)u_max_q / (double)c_amb_q;
const int64_t ratio_umax = (int64_t)(ru * ru * 65536.0) + 1;

// inside the scan loop. The kick's skip-set (solid||is_vacuum||ambient-ring)
// is a strict SUPERSET of the scan's (solid||is_vacuum), so no kick-processed
// cell ever reads the filler — u_max2_q32 is a safe defined value:
if (solid[i] || is_vacuum[i]) { cap2_plane_[i] = u_max2_q32; continue; }
int64_t t_abs = (((int64_t)s_eos_q * (int64_t)temperature[i]) >> 16)
                + (int64_t)t_amb_q;                    // bare >>16 — the
                                                       // existing twins' idiom
if (ts[i] || t_abs < (int64_t)t_amb_q) t_abs = (int64_t)t_amb_q;  // D4 + D1
const int64_t ratio = (t_abs << 16) / (int64_t)t_amb_q;   // int64, NO narrow
cap2_plane_[i] = (ratio >= ratio_umax)
    ? u_max2_q32                                       // rail; avoids the
    : mul128_shr(c_amb2_q32, ratio, 16);               // 2^65+ overflow path
```

(`ts` is in scope in step() — folded at `eos_solver.cpp:285`.
`eos_host_prestage` does NOT currently receive `thermal_solid`: add the param
with the `thermal_solid ? thermal_solid : solid` fallback; both callers
already hold it — `cuda_eos_step.cu:291`, `cuda_eos_resident.cu:656`. On the
`cuda_eos_step.cu` twin the shared-multiply helper is `mul128_shr_host`.)

The global `t_max_abs_raw` reduction, `c_local_q`, and `dbg_last_c_local_q`
are UNCHANGED (n_sub + telemetry). `cap2_plane_` is a per-solver int64 scratch
plane (the `n_total_` idiom). Overflow: `ratio < ratio_umax ≈ 7.3e5` in the
mul branch → product ≤ ~2^68, 128-bit intermediate via `mul128_shr`, result
≤ u_max2 ≈ 2^52 ✓ (the branch guard exists precisely because a clamp AFTER
the multiply would be unsafe at degenerate `T_AMB_K` dials, where the product
reaches ~2^95). The `>>16` on a possibly-negative product is arithmetic SAR
(compile-checked, `fixed_point.h:85`); D1's floor runs before the divide, so
the divide never sees a negative numerator — identical trunc-toward-zero
semantics host/device.

### B. Kick clamp — all three sites, identical text

Replace the `u_cap_q`/`cap_is_umax` derivation and the clamp block
(`eos_solver.cpp:766-803`, `eos_solver.cpp:1660-1679`,
`cuda_kick_compression.cu:189-213`); `sqrt_q16` ↔ `sqrt_q16_dev` per side:

```c
const int64_t cap2_q32 = cap2_plane[i];                     // D5: trusted
const bool cap_is_umax = (cap2_q32 >= u_max2_q32);
/* RAD_SAFE component pre-clamp: UNCHANGED, verbatim */
const int64_t rad = ux * ux + uy * uy;    // int64-safe (RAD_SAFE guard)
if (rad > cap2_q32) {
    ++u_clamp_hits;                       // atomicAdd on the GPU, as today
    if (cap_is_umax) ++u_max_hits;
    const q16 umag    = sqrt_q16(rad);        // Q.32 radicand → Q16.16
    const q16 u_cap_q = sqrt_q16(cap2_q32);   // same convention (verified:
                                              // all three call sites Q.32)
    ux = (ux * (int64_t)u_cap_q) / (int64_t)umag;   // D6 exact rescale
    uy = (uy * (int64_t)u_cap_q) / (int64_t)umag;
}
```

`u_max2_q32` derives from the existing `u_max_q` fold at each site
(`(int64_t)u_max_q * u_max_q` — host fold or kernel-local, identical result).
The old `ax`/`ay` component absolutes leave the clamp; delete them if nothing
else reads them.

Division safety under D5 (the kick trusts arbitrary test-supplied planes, so
"umag ≥ c_amb" is NOT an invariant): **`cap2_plane[i] ≥ 0` is a hard contract
of both bindings** (document it in the docstrings; a negative entry would make
`rad = 0 > cap2` reachable → divide by zero). With the contract, inside the
branch `rad > cap2 ≥ 0 ⇒ rad ≥ 1 ⇒ umag ≥ 1` — nonzero divisor. The D6
product bound that holds for ANY plane: `sqrt_q16` self-clamps at INT32_MAX
(`fixed_point.h:749`), so `|ux·u_cap_q| < 2^30·2^31 = 2^61` — int64-safe
unconditionally (the tighter 2^56 figure assumed `u_cap ≤ u_max`, which D5
does not guarantee).

**Sqrt census:** zero per-cell sqrts; two per *clamped* cell (engage branch
only) — today's shape. **No float in the per-cell path.**

## Change sites (corrected inventory — from the scope critique, verified)

C++ behavior:
1. `cpp/src/eos_solver.cpp:397-427` — scan grows the plane fold (A); keep
   global reduction; update comment `:431-433` (global is n_sub input ONLY).
2. `cpp/src/eos_solver.cpp:766-803` — live kick clamp (B); comments `:704`,
   `:766-769` (Chebyshev claim), `:830-832` (rename, now TRUE), `:876-877`
   (the √2 slack line — bound becomes `max(u_cap, RAD_SAFE)`).
3. `cpp/src/eos_solver.cpp:1520-1556` — P6.4 reference: contract block
   `:1535-1537` INVERTS (the cap is now derivable from the replay's inputs —
   say so); signature: `int32_t c_local_q` → `const int64_t* cap2_plane`;
   `:1567-1587` folds add `u_max2_q32`; `:1660-1679` clamp (B).
4. `cpp/src/eos_solver.h:611-682` (P6.4 contract + decl `:660`), `:28`,
   `:613-615`; new `cap2_plane_` scratch member near `n_total_`.
5. `cpp/src/cuda_kick_compression.cu`: kernel params `:111-128`
   (`c_local_q` → `const int64_t* cap2_plane`), clamp `:189-213` (B),
   `kick_scalar_folds` def `:340-370` (u_max2 fold), launch-core
   `kick_compression_launch_resident` `:374-399`, per-call entry
   `breach_cuda::eos_kick_compression` `:412-427` (NOT "run_kick_compression"
   — v1 misnamed it), folds call `:436-439`, core call `:517-519`; header
   comments `:5`, `:18`.
6. `cpp/src/cuda_kick_compression.h:11-12`, `:41-51` (the "c_local_q is the
   solver's per-tick cap" lie), decl `:57-89` (param `:65`).
7. `cpp/src/cuda_resident.h:114-134` (`KickScalarFolds`), `:135-139`
   (`kick_scalar_folds` decl), `:153-158` (launch-core decl).
8. `cpp/src/cuda_eos_step.cu`: `eos_host_prestage` (`:112-175`) gains a
   `thermal_solid` param (+ `ts` fallback fold) and a `u_max_q` fold (~`:159`),
   and fills the plane in its scan twin (`:171-175`); `EOSHostPrestage`
   carries the plane as a HOST `std::vector<int64_t> cap2` — the
   `coeffE`/`coeffS` idiom, NOT a device pointer (`cuda_eos_step.h:91-92`
   documents the struct as CUDA-type-free, "H2D'd once by the caller");
   `:228` D7 clip; `:291` caller passes `thermal_solid`; `:322` delete the
   now-unused local; `:600-616` call passes the plane. Per-call upload: the
   kick entry's own H2D block (`cuda_kick_compression.cu:469-513`, the
   `d_udamp`/`d_tsol` idiom).
9. `cpp/src/cuda_eos_step.h:87` (comment: c_local is n_sub/telemetry only);
   `:91-107` (`EOSHostPrestage` + `eos_host_prestage` decl grow as above).
10. `cpp/src/cuda_eos_resident.cu`: `:656`/`:685` caller passes
    `thermal_solid` into the shared prestage (NO re-fold here); plane upload
    joins the existing per-tick H2D block `:715-723` (`S.coeffE`/`S.coeffS`/
    `S.absorb_q` precedent) with an `int64_t* cap2` member in
    `EOSResidentScratch` via the `a64` allocator `:609-611` (P-E1's
    `e`/`nbulk` idiom); `:691-696` (folds call) + `:810-814` (launch). One
    ~56 KB H2D per tick at 70×100 — P-V1 notes the measured cost.
11. `cpp/src/bindings.cpp:1117-1182`, `:2394-2491` (both bindings: drop
    `c_local_q`, take an int64 cap2 plane array — copy the
    `py::array_t<int64_t>` marshalling idiom from `fuel_recip` at `:2825` /
    `:2871-2872`; document the `cap2_plane ≥ 0` contract in both docstrings),
    `:2269-2272` (telemetry docstring). No pybind defaults — all call sites
    updated explicitly.

Python:
12. `tests/cuda_kick_check.py` — `_run_pair` `:123-135` (the plane drops into
    the old `c_local_q` positional slot as an `(h,w)` int64 array — per-call,
    verified); **PART 1 REWRITTEN** (`:272-354`): block (a)'s two cap-regime
    configs become two constructed planes (`u_max2` cells → `u_max_hits != 0`
    regime — reachable, the forcer's RAD_SAFE-clamped seeds far exceed 1000
    m/s; `c_amb2` cells → `== 0` regime); blocks (b) `:304-323` and (c)
    `:330-337` ALSO pass c_local today and need per-config planes — use
    `u_max2` for (c) so the drag forcer's ~1272 m/s still engages the clamp
    (a 2300² plane would disengage it under D5 and lose that coverage);
    PART 2 `:456`/`:508` rebuilds the plane from `t0` with fold A verbatim
    (t0 IS tick-entry T, never mutated — the replay stays exact; the scenario
    has no ts-gas cells, so `ts = solid` on both sides); module docstring
    `:6-21`. **Python-side fold arithmetic: use plain Python ints, NOT numpy
    int64** — `c_amb2·ratio` reaches ~2^68 and wraps silently in np.int64
    (build per-cell with int(); vectorize only the final ≤-comparison).
13. `tests/cuda_thermal_mass_eos_check.py:177-192` (kwarg caller).
14. `tests/test_thermal_mass_axis.py:634-644`, `:681-691` (kwarg).
15. `tests/test_p_e3_drag.py:76,178,188,231` (positional, spelled `c_local`):
    pass a 2300²-scale plane to keep the clamp disengaged (D5 allows it);
    RE-MEASURE the `:148` `worst_frac < 1e-3` bound, do not assume it.
16. `tests/test_p_e4_reversible_work.py:112`: signature only. Its neighbours
    sit at T=0 → cap = c_amb = 300 = |u| exactly, and `rad > cap2` is strict
    → no clamp, same as today. DO NOT "fix" the 300.0.
17. New property tests (gates 1-2 below).
18. **Step 0, before any code:** record the pre-patch red-test NAME list from
    `pytest tests -q` into the patch doc (gate 5's baseline — a count hides
    swaps).

Downstream consumers of the global bound: exactly one (`u_est` clip — D7
handles it); nothing else in `cpp/src`, `src/`, `tools/` reads a velocity
bound (verified by the scope critique).

## Gates (P-V1 is oracle-gated)

1. **Property — diagonal leak closed (exact, via the P6.4 reference):** drive
   `eos_kick_compression_reference` with constructed states (strong diagonal
   winds, both-components-under-cap) + a constructed plane; assert every open
   cell ends with `wx² + wy² ≤ (⌊√cap²⌋ + 2)²` — squares computed in
   **int64** (`.astype(np.int64)`; int32 squares wrap silently), and any
   Python-side cap-plane folding in plain Python ints (the np.int64 wrap
   hazard above). Exact because the reference receives the very state the
   cap derives from.
2. **Property — cap locality (exact, same harness):** hot cells (fire-range
   T) in one region of the constructed state, blast-scale ∇P in a cool
   region; fold the plane from that T with formula A in the test; assert the
   cool region obeys the AMBIENT cap bound of gate 1's form — the remote hot
   cell must not raise it. Plus one full-engine smoke: playground blast
   scenario, post-tick snapshot, assert `wx²+wy² ≤ cap²(T_snap)·1.5` for all
   open cells (loose e2e wiring check only; T moves after the fold, hence
   the slack — the exact assertions live in gates 1-2).
3. **P6.4 reference parity:** reference and step() changed in lockstep; the
   existing parity gate green on the new behavior.
4. **CUDA lockstep:** `cuda_kick_check` PART 1 (rewritten) green at tol 0;
   PART 2's **wind-side** replay (plane from t0) green at tol 0 — this gate
   is *restored to full validity* by D2v2, do not let it regress; PART 2's
   `digest_compression` red is the KNOWN pre-existing defect — measure and
   report before/after, do not gate on it, do not claim a fix. Full-engine
   40-tick CPU-vs-CUDA A/B (s8a pattern) tol 0 on a blast scenario, both
   per-call and resident modes.
5. **Suite:** `pytest tests -q` — zero newly-red vs the step-0 name list,
   EXCEPT the pre-declared `GOLDEN_AGGREGATE` dependents: the sanctioned
   golden (`tests/_xarch_perfield_digest.py:155`) feeds ~12 currently-green
   tests; if the golden trajectory engages the new clamp (P-V1 measures
   `u_clamp_hits` over `capture_trajectory(n_steps=30)`), those flip
   together and are EXPECTED-red until the arc-close re-baseline — record
   the measured hits + the exact flipped-test list in the patch doc. Any
   OTHER new red is a gate failure.

Behavioral-change note: sim digests move (that is the point). No golden
re-baseline in this patch — once, at arc close, per Erik's standing ruling.

## Patch contract (autonomous-patch-workflow annotations)

| patch | content | mode | tier | gate |
|---|---|---|---|---|
| P-V0 | design v1 → 3-lens critique → this v2 → focused re-verify | inline (this session) | Fable, xhigh | doc survives critique |
| P-V1 | the fix per this doc, sites 1-18 | subagent | Sonnet 5, high | oracle (gates 1-5) → commit+push on green, NO merge |
| P-V2 | measurement: scripted blast + seed-dump comparison — supersonic-vs-own-cap count, peak pile-up cell-delta, P_min, n_sub_required vs the 8-rail, u_clamp_hits/u_max_hits, clamp-energy residual; capture doc | subagent | Sonnet 5, low | report only |
| P-V3 | **HUMAN-TEST** — Erik plays playground w/ pressure viz; expectation set: spikes attenuated, not necessarily gone | Erik | — | feel verdict → merge on PASS |
| close | re-baseline goldens ONCE w/ rationale (the six + any GOLDEN_AGGREGATE flips); archive; tag; N_SUB_MAX question to Erik with P-V2 numbers | subagent | Haiku 4.5, low | suite green post-baseline |

Build (Lenovo): `cmd /c "cpp\build_cuda_lenovo.bat"` (RTX 1000 Ada sm_89,
CUDA 12.9); python = conda env `data`; `pytest tests -q` only. Branch:
`velocity-clamp` (this worktree). Stage explicit paths, never `-A`.

## Non-goals

- Drag law (item 3); `n_sub`/substep derivation beyond D7's clip; N_SUB_MAX
  policy (goes to Erik with P-V2's numbers); grenade payload; aquarium;
  smoke saturation. No new config dials.
