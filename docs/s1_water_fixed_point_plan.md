# S1 — Water → integer (Q16.16) — plan

**Status:** plan, awaiting Erik's OK. The **first field migration** of the fixed-point arc (after S0's
prereq gate). Built from the water-solver map (Explore, 2026-06-24) and the idioms/gating in
`docs/fixed_point_migration_plan.md`. Water is a well-behaved first target: **no RNG, no global
reductions, no order-dependent sums, all transcendentals are scalar (none per-cell), and the conserved
field is already in conservative flux form.**

---

## 1. Scope — what converts, what doesn't

- **CONVERT to int32 Q16.16** (the synced shallow-water state): `water_depth` (conserved), `flow_vx`,
  `flow_vy`. Plus the load-time scalars they need.
- **STAYS FLOAT — render-only, never synced** (the synced/local boundary, per Q4): `ripple`, `ripple_v`
  (cosmetic surface waves) and the whole `step_ripple` path + `ripple_max_dt`. Determinism doesn't need
  cosmetics — leave them float and fast.
- **STAYS A FLOAT BRIDGE until S2:** the **W3** (water↔atmosphere displacement/seal) and **W5**
  (water↔smoke steam) couplings — they read/write `atmosphere` + `gas`, which are still float (the S2
  group). `water_depth` dequantizes to float at those boundaries. ⇒ **S1 water is self-reproducible
  (bit-identical, same config) now, and becomes cross-GPU-deterministic when S2 lands.** (Honest
  labelling per the plan's coupling-group order.)
- **`floor_height`** (read-only terrain): quantize to Q16.16 at load (it's added to `water_depth` in the
  surface potential).

## 2. The conserved field is already correct (the nice part)

`water_depth` changes **only** via donor-cell upwind flux divergence (`water_solver.cpp:108–179`):
`flux = v_face·depth[donor]`, `depth -= dt/dx·div`. That is mass-conserving **by construction** — it
*is* the edge-flux idiom the plan wants, already present. No re-formulation (unlike if it had been a
temperature-style difference-shift). The Q16.16 conversion just carries the flux multiply in int64 and
narrows:
```
flux_q16 = (int32)(((int64)v_face_q16 * depth_q16) >> 16)   // donor-cell, conservative
depth_q16 -= (int32)(((int64)dt_over_dx_q16 * div_q16) >> 16)
```
**P2 watch:** the `>>16` narrow must not leak mass — confirm the flux in/out of a face uses the *same*
rounded value on both sides (gather the face flux once, apply ±it to both cells) so the round is
conservative.

## 3. Divides → reciprocals / shifts

- **Constant divides** (`two_dx`, `dt_over_dx`): precompute reciprocals (Q16.16) at load. (`inv_dx2` is
  ripple-only → stays float.)
- **Outflow limiter** (`water_solver.cpp:154`, the one per-cell dynamic divide in the water core):
  `scale = depth·dx / (dt·out_sum)`. → precomputed-reciprocal multiply: reciprocal of `(dt·out_sum)`
  as an int64, multiply. This is the conservation clamp (keeps outflow ≤ depth), so it's load-bearing.
- **W3 ratio** (`physics_engine.cpp:381`): lives in the float-bridge → stays float until S2.

## 4. Transcendentals — all scalar, none per-cell (easy)

- **`tan(tilt_x)`, `tan(tilt_y)`** (`water_solver.cpp:56–57`): scalar, once per tick. → precompute as
  Q16.16 in the Python prep (`physics_runner`) before the engine call. **Q-S1-2 (below):** small
  committed Q16.16 `tan` LUT indexed by quantized tilt (recommended — tilt range is tiny) vs CORDIC.
- **`sqrt` in water `max_dt()`** (`water_solver.cpp:10`): the water CFL, feeding the synced substep
  count `n`. `max_dt` is a **constant** from config (`g, h_ref, k_p, P_REF, HEAD_REF`) → computed once
  at load. **Q-S1-3:** a reusable **deterministic integer-sqrt** helper (Newton, fixed iters — we need
  it for S2/S3 anyway) vs a single double-sqrt-at-load + quantize (also cross-platform-deterministic
  for a correctly-rounded sqrt of a constant). Recommend the reusable integer sqrt.
- **`sqrt` in `ripple_max_dt()`**: ripple is render-only → stays float.

## 5. The integer cliff — the substep count `n`

`physics_engine.cpp:286–288`: `n = ceil(sim_time / max_dt())` in **float64** → `int`. A 1-ULP slip
flips `n` → different peer trajectories. → make `max_dt` a Q16.16 constant; compute `n` in fixed point:
a deterministic integer `sim_time_q / max_dt_q` (or reciprocal-multiply) with a fixed-point `ceil`
(`(x + (1<<16) - 1) >> 16` form). **This is the cross-GPU determinism fix for water's substep count.**

## 6. Gating

- **P1 — within-config self-match at `tol=0.0`** (run twice, bit-identical) AND **cross-config**
  self-consistency (vary `tps` / `tile_size_m`(=dx) — the integer path stays internally consistent).
  Use the field A/B harness (now incl. the unit-state digest from S0).
- **P2 — conservation:** Σ`water_depth` over a sealed flood scenario constant to the LSB (verify the
  int64-flux narrow doesn't leak — §2).
- **Behavior change** (integer ≠ the old float exactly) → regenerate the water goldens; **feel-regression
  check** (pour water, flood a room — it flows the same; ripples are unchanged, still float) + **Erik's
  eye**.
- Full suite green + both `--auto` exit 0.

## 7. Migration steps (each its own gated commit, on an `s1-water-fixedpoint` branch)

- **S1a** — the Q16.16 toolkit + field representation: a reusable header (`fixed_point.h`?) with the
  int-sqrt, Q16.16 mul/narrow, reciprocal helpers; convert `water_depth`/`flow_vx`/`flow_vy` to int32
  (with a float dequantize for the renderer + the W3/W5 bridges). `floor_height` quantized at load.
- **S1b** — `WaterSolver::step()` core in integer: surface potential (tilt precompute), velocity kick
  (reciprocal `two_dx`), upwind flux (int64 mul), outflow limiter (reciprocal), divergence, clamp.
  Gated P1+P2.
- **S1c** — the substep cliff: fixed-point `max_dt` + integer `n`.
- **S1d** — the float bridges (W3/W5): explicit `water_depth` int→float dequantize at the
  atmosphere/smoke boundary; documented as bridges-until-S2.
- **ripple**: untouched (stays float).

## 8. Open questions for Erik

- **Q-S1-1** — confirm `ripple`/`ripple_v` stay **float** (render-only, never synced). *Recommend: yes.*
- **Q-S1-2** — `tan(tilt)`: small committed Q16.16 LUT (recommend) vs CORDIC vs double-at-tick-quantize.
- **Q-S1-3** — water `max_dt` sqrt: reusable **integer-sqrt** helper (recommend — reused in S2/S3) vs
  double-at-load-quantize (fine for a single constant).
- **Q-S1-4** — `water_depth` for the renderer + the W3/W5 bridges: **dequantize on demand** at the
  boundary (recommend — one source of truth, the int field) vs keep a parallel float mirror.
- **Q-S1-5** — Q16.16 enough precision for water? `water_depth` resolution at Q16.16 ≈ **15 µm**; flow
  ≈ 15 µm/s. Almost certainly yes (water is perceptual, depths are cm-to-m). Confirm we don't need a
  finer fractional format for any water field. *Recommend: Q16.16 is fine (the map's range table agrees).*
