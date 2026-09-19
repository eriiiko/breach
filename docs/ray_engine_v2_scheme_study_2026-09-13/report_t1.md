# T1 — per-tile ambient (`amb_m`) + uniform `k_leak` live in the sweep

> Issue #12, ray-engine-v2. Worktree `breach-t1`, branch
> `12-t1-ambient-plane-leak-live` off `fire-12`.
> Design: `docs/thermal_model_v2_design_2026-09-19.md` R2/R3, §3.3, §5 T1, §6 2-4.
> **Written incrementally** — every section below is filled as its work lands.

## 0. Status

| step | state |
|---|---|
| D1 reference (`sweep_ref_q.py`) | pending |
| D1 reference gates G1-G12 | pending |
| D2 engine (`radiation_sweep.{h,cpp}` + binding + runner) | pending |
| D3 gate 0 bit-for-bit | pending |
| D3 goldens unmoved | pending |
| D4 item 2 (uniform == scalar era) | pending |
| D4 item 3 (non-uniform genuinely varies) | pending |
| D4 item 4 (conservation with per-cell ambient AND k_leak > 0) | pending |
| full suite | pending |

## 1. The scalar that was named wrong, and the one that is right

`t_amb_q` is the **absolute-zero offset of the temperature scale** — the integer
that turns a game temperature (a ΔT above ambient) into Kelvin for the Fleck
denominator `g = 4L/T_abs` (`radiation_sweep.cpp:211`). It is bound from the same
`293 << 16` the arc #54 gas-energy seam runs on, and it stays a **global scalar**:
untouched by this patch (thermal v2 §3.3, L2-B1, L2-R1).

The ambient the sweep *radiates* at is `amb_m = (E°[0] · w_m) >> 16` — hoisted
once per run at `radiation_sweep.cpp:175` before this patch. **That** is what
becomes per-cell.

## 2. Decisions — which cell's ambient, at each of the four sites

*(filled at D1)*

## 3. The reference change (D1)

*(filled at D1)*

## 4. The engine change (D2)

*(filled at D2)*

## 5. Gate results (D3, D4)

*(filled at the end)*

## 6. Findings, surprises, and what T5 inherits

*(filled as they appear)*

## 7. Commits

*(filled as they land)*
