# report_m1.md — M1: negative `thermal_mass` exponents

> Implementer's report, written by the M1 agent; committed by the orchestrator
> because the session's write policy blocked the agent from creating it.
> **Design of record**: `docs/thin_material_rows_design_2026-09-20.md` §5.
> Branch `12-m1-negative-exponents`, commits `2e20c0f`, `d5e3ba4`.

---

## 1. What landed, per site

`thermal_mass` could not go below 1. That was a **guard, not an arithmetic
limit**: `heat_inv_shift` has always been `int32_t`, and `cell_capacity_q` has
always built a Q16.16 capacity `1 << (s + 16)`, where `s = -2` is
`1 << 14 == 16384 ==` exactly 0.25.

### The kit — one new primitive

`cpp/src/fixed_point.h:497` — `fixedpoint::shr_round0_signed_i64(x, s)`: divide
by `2^s` for a **signed** `s`.

- `s >= 0` **is** `shr_round0_i64(x, s)`, value for value — the identity that
  makes every rewritten site bit-identical on the shipped table.
- `s < 0` **multiplies** by `2^-s`. Exact. Computed as a multiply, **not**
  `x << (-s)`: left-shifting a negative `int64` is implementation-defined before
  C++20 and this is a determinism TU. Saturates at the int64 rails (signed
  overflow is UB); shift width clamped at 62 so a direct-binding caller cannot
  reach `1 << 64`.

`FP_HD`, so the CUDA twins call the same definition. **No device copy was added
to `cuda_fixedpoint_device.cuh`**, contrary to the brief's letter: `fixed_point.h`
is already included by `cuda_temperature.cu` / `cuda_combustion.cu`, and its
`FP_HD` siblings (`shr_round0_i64`, `sat_add_q16_i64`) are called from device code
there today. A second definition would be the parallel-implementation bug the
canonical-systems rule exists to prevent.

### The four sites the design names

| # | site | line | what |
|---|---|---|---|
| 1 | `src/simulation/materials.py::pow2_snap` | `:365` | returns the **signed exponent**; floor `THERMAL_MASS_EXP_MIN = -16` (`:362`), still a **named refusal** below it, never a clamp |
| 2 | `cpp/src/temperature_solver.h::cell_capacity_q` | `:223` | `if (s < 0) s = 0` → `if (s < CAP_SHIFT_MIN) s = CAP_SHIFT_MIN`, `CAP_SHIFT_MIN = -FP_SHIFT` at `:199`. At `s = -16`, `cap_used == 1` — one raw count, never zero, the exact twin of the gas branch's `cu < 1` divide guard |
| 3 | `cpp/src/temperature_solver.cpp` radiation fold | `:291` | `shr_round0_i64` → the signed twin. Already int64; **"already wide" is not "already signed"** |
| 4 | `cpp/src/temperature_solver.cpp` heat deposit | `:379` | **the overflow site.** `int32_t gain = deposit >> shift` → int64 through the signed twin, landed by `sat_add_q16_i64` |

CUDA twins, line for line: `cuda_temperature.cu:233` (fold), `:275` (deposit).

### `pow2_snap`'s knock-on inside `MaterialTable`

`derive_thermal_mass_exp` is now THE one place the exponent is computed, feeding
**both** `thermal_mass` and `heat_inv_shift`, so they cannot disagree;
`derive_thermal_mass` survives as a `math.ldexp(1.0, exp)` wrapper.

Two consequences that had to be handled or M2 would have broken silently:

- the old shape recovered the shift from the value via `int(round(tm))` +
  `bit_length()`. `int(round(0.125)) == 0` — a thin panel's shift would have come
  out as the **gas-regime placeholder**.
- `thermal_solid` was derived from those same rounded ints. It is now
  `thermal_mass > 0` on the **float32** column (the definition, not a proxy).
  Same for the two `heat_atten` ingress doors.

**On the gas sentinel, which the brief asked to verify rather than assume**: it
does *not* collide at the C++ boundary (`is_ts` is a separate `bool`, the value is
never tested) — but it **did** collide on the Python side, in two places.

### Three more sites — the design's §5 list is incomplete

`heat_inv_shift` has **seven** consumers, not four. All three extras are live on
the sim path:

| site | line | why it matters |
|---|---|---|
| `radiation_sweep.h::fleck_L_solid_q` | `:73` | Fleck emission damping's `L_q`, once per cell per tick. On x86 the UB shift by `s & 63` yields 0 — **no damping at all on exactly the thin rows the design is adding** |
| `combustion.cpp` object-site deposit (+ `cuda_combustion.cu:531`) | `:1084` | how a burning object heats itself. Also an **overflow site** — routed through int64 and clamped to `[0, INT32_MAX]`, mirroring the gas branch beside it |
| `raycaster.h::rad_pair_budget_s` | `:242` | the old cast's flux limiter. **The old cast still runs every tick** after T5b's flip (only the *fold* moved to `rad_net_sweep`). `(x << his) >> shift` collapses to one signed exponent: `shr_round0_signed_i64(x, shift - his)`, identical for the `x >= 0` every caller passes |

### The integer reference went first

Per the CLAUDE.md rule: `sweep_ref_q.py` gained `shr_round0_signed`;
`fleck_L_solid_q`, `fold_pass1_solid` and `cell_rad_net_q` were routed through it;
`_thermal_mass_ref` → `_thermal_mass_exp_ref` returning the signed exponent. Its
gates were re-run (112 passed) **before** any C++ was written.

### Bindings added (test surface only)

`fp_shr_round0_signed_i64`, `conduction_cell_capacity_q`, `rad_pair_budget_s`,
`CAP_SHIFT_MIN`/`CAP_SHIFT_MAX`, and an optional `rad_net` kwarg on
`cuda_temperature_step`. Same precedent as the existing `fp_*` block.

---

## 2. Gate results

Python `C:/Users/steen/anaconda3/python.exe`. Both builds produced, both green:
`cpp/build/Release` (VS 2022) and `cpp/build_cuda` (Ninja + nvcc 12.4, archs
75/86/89).

| | passed | failed | skipped | xfailed |
|---|---:|---:|---:|---:|
| **baseline** (`8dd9d5a`, both builds fresh) | 2563 | **7** | 4 | 4 |
| **M1 final** | **3390** | **7** | 4 | 4 |

**The 7 failures are identical before and after** — the pre-existing T5b runaway
reds (`test_eos_p4_combustion` ×3, `test_ps1_smoke_roundtrip`,
`test_s3c_unit_state_digest`, `test_unit_state_digest` ×2). A finding of T5b, not
a target. Nothing was bent to them.

CUDA gates run for real (verified by reading the subprocess transcript, not just
the green dot):

```
PART 1 — Pass-1 solid heat deposit across the exponent band
  PASS (110 configs, 105 with a nonzero exact gain)
PART 2 — Pass-1 signed radiation fold at negative exponents
  PASS (30 configs, 28 non-trivial)
PART 3 — Pass-2 conduction across a capacity straddling 1 unit
  PASS (4 configs)
PART 4 — 60-tick mixed-exponent trajectory, per-tick byte-identity
  PASS (moved on 60/60 ticks)
M1_RESULT: PASS
```

CPU/CUDA bit-identity is **tol 0** on `temperature` and the `T_MAX_PHYS` hit
count, at every exponent from `CAP_SHIFT_MIN` to +5. No new C++ TU, so the
`/fp:strict` list needs no entry; every file touched was already on it.

---

## 3. Golden re-baseline — there isn't one, and that is the evidence

**No golden re-baselined, no digest spec bump.** The brief and the design both
anticipated one. It would have been wrong:

- `heat_inv_shift` is **not a digest field** (`field_digest_spec.toml` lists it
  nowhere) → no membership/dtype change → no version bump.
- M1 authors no row, so every shipped exponent stays non-negative.
- On a non-negative exponent every rewritten site computes the *same integer*:
  `shr_round0_signed_i64(x,s) == shr_round0_i64(x,s)` for `s >= 0`;
  `sat_add_q16_i64(t,g) == heat_saturating_add(t,g)` for `g >= 0`.

So the golden set **staying put** is the neutrality evidence, and re-baselining
would have destroyed the very check this patch most needed. `GOLDEN_AGGREGATE`
untouched at `e369b616bb…`; all 14 golden-bound tests passed unedited.

**What was used as the oracle for the NEW behaviour instead:**

1. **The committed integer reference** — changed first, its gates re-run, then the
   C++ held to it integer-for-integer on negative-exponent scenes. It is a spec in
   Python big-integers; it cannot be satisfied by recording what the engine did.
2. **Exact arithmetic computed in the tests**: `2**(s+16)` for capacity,
   `x * 2**-s` for the shift, `deposit * 2**-s` for the deposit, and a **1:2:4
   ratio identity** for combustion (same burn, only the exponent varies) that
   needs no model of the deposit's magnitude.
3. **CPU↔CUDA agreement at tol 0** — an independent implementation.

---

## 4. Every gate validated by breaking it

Ten deliberate breaks; each applied, rebuilt, run, observed red, reverted.

| # | the break | gate that caught it | what it showed |
|---|---|---|---|
| **B1** | site 4 reverted to the *sign-only* int32 form `(int32_t)(deposit << -shift)` — the fix **without** the int64 routing | `test_the_solid_heat_deposit_survives_a_negative_exponent_that_overflows_int32` | landed **500 000 000** (≈7 629 game — a believable warm tile) where the exact int64 gain rails at **1 048 576 000**. **Exactly one test failed**, proving it isolates the *widening*, not the sign handling |
| **B2** | `if (s < 0) s = 0` restored in `cell_capacity_q` | `test_cap_used_round_trips_every_exponent…`, `test_the_gas_sentinel_does_not_collide…` | every negative exponent collapsed to 65536 |
| **B3** | signed twin's negative branch made lossy | `test_the_left_shift_branch_is_EXACT` (215 params) + 3 others | "a left shift loses nothing" is asserted, not just written down |
| **B4** | site 3 reverted to `shr_round0_i64` | `test_the_radiation_fold_takes_a_negative_exponent_too` | one test, precisely |
| **B5** | `thermal_solid` reverted to `int(round(tm)) > 0` | `test_a_row_at_the_representation_floor_is_accepted_not_refused` | a 2⁻¹⁶ row reclassified as GAS |
| **B6** | **only** the CUDA deposit reverted, CPU kept | `test_cuda_m1_negative_exponent` | `cpu=1048576000 gpu=0` across steep exponents — a backend divergence with **no CPU-side symptom** |
| **B7** | `rad_pair_budget_s` back to `(x << his) >> shift` | `test_the_raycaster_pair_budget_is_the_exact_signed_scaling` | budget 0 at every negative `his` — the limiter would clamp a thin object's radiative exchange to nothing |
| **B8** | `fleck_L_solid_q` back to `shr_round0_i64` | `test_the_sweep_reproduces_the_reference_at_negative_exponents` | C++ sweep diverged from the reference |
| **B9** | combustion object site back to `deposit >> shift` | `test_the_combustion_object_site_deposit_scales_with_the_signed_exponent` | the ratio identity broke |
| **B10** | the −16 floor made a **clamp** instead of a refusal | `test_a_row_below_the_REPRESENTATION_floor_is_a_named_refusal` | design §11.3's "a load-time error, not a clamp" is enforced |

**B1 is the one the brief singled out**, and it behaves as required: passes after
the fix, fails when *only* that change is reverted. A test that merely exercised a
negative exponent would have passed both before and after — the sign repair alone
compiles and runs, it just produces a wrong number.

**B7/B8/B9 matter most.** Before those tests existed, breaking all three of those
sites left the **whole suite green**. Same blind-gate pattern, found a fifth time:
the shipped table reaches no negative exponent, so nothing could see them.

---

## 5. Findings — real, not M1's to fix

1. **The design's §5 site list is incomplete (3 of 7 missing).** Not a
   contradiction — §5 reads as a worked example, and §12 states the governing rule
   ("the signed-shift twin joins the kit; no shift or rounding is re-derived at a
   call site"), which is what was followed. Recorded so M2's author does not treat
   the list as exhaustive. **If a design lists sites, it should say whether the
   list is exhaustive.**
2. **`cuda_temperature_step` could not be driven with `rad_net`** — the binding
   passed `nullptr` unconditionally while the CPU twin exposed the argument, so
   the CUDA radiation fold had **no isolated parity gate**; its only coverage was
   the engine-level A/B at shipped exponents. Fixed here, but the asymmetry is
   worth knowing about.
3. **`tests/test_temperature_conduction.py::_capacities` is a hand-written Python
   mirror of `conduction::cell_capacity_q`** carrying its own
   `np.maximum(his, 0)`; it had to be edited in lockstep. A second transcription
   of a canonical law is what the canonical-systems rule forbids; the new
   `conduction_cell_capacity_q` binding could retire it. Out of M1's scope.
4. **`pytest tests -q` in this tree does not use the conda `data` env** the
   project CLAUDE.md names — there is none on this box.

---

## 6. Contradictions with the design

**None.** Two deviations from the brief's *letter*, both serving a project rule
the design itself invokes, both reported rather than silently taken:

- the signed twin lives **only** in `fixed_point.h` (`FP_HD`), not also in
  `cuda_fixedpoint_device.cuh`.
- **no golden re-baseline and no digest bump.** §5 says "a digest bump with
  goldens (`heat_inv_shift` is synced state)". It is per-tile synced *state*, but
  not a **digest field**, and M1 moves no value — there is nothing to re-baseline,
  and doing it anyway would have thrown away the neutrality proof.

The two `ACCEPTED GAP` items were not touched, not critiqued, and no machinery was
built around them.

---

## 7. Tests added or rewritten

### `tests/test_m1_negative_thermal_mass_exponents.py` (new, 478 params)

| test | property | breaks if |
|---|---|---|
| `test_pow2_snap_returns_the_signed_exponent_not_the_value` | the snap returns `s`, `2**s` log-space nearest, either sign | the snap returns `1 << exp` again, or stops being log-space nearest |
| `test_the_design_s_thin_rows_are_expressible` | 0.120/0.104/0.260 snap to −3/−3/−2 without raising | the `exp < 0` refusal returns, or rounding moves a §6 row onto a different capacity |
| `test_derive_thermal_mass_states_the_same_answer_as_a_capacity` | value form ≡ `2**exp`, float32-exact for every legal exponent | the two forms grow separate arithmetic |
| `test_cap_used_round_trips_every_exponent_from_the_floor_to_the_ceiling` | `cap_used(s) == 2**(s+16)` for `s ∈ [−16, 12]`; never 0 | `if (s<0) s=0` returns, or `CAP_SHIFT_MIN` moves off `−FP_SHIFT` |
| `test_the_gas_sentinel_does_not_collide_with_a_sub_unit_capacity` | 0 (gas) vs 0.125 (panel) told apart by the `is_ts` flag, never the value | `thermal_solid` reverts to a rounded int, or the C++ tests the capacity value |
| `test_the_signed_shift_is_the_old_shift_on_every_non_negative_exponent` | **neutrality**: identical to `shr_round0_i64` for `s >= 0` | the twin changes the non-negative branch's rounding |
| `test_the_left_shift_branch_is_EXACT` | `s < 0` gives `x * 2**-s` exactly | the negative branch truncates or narrows first |
| `test_the_negative_branch_is_lossless_where_the_positive_branch_is_not` | the branches are not mirror images | both routed through one rounding step |
| `test_the_signed_shift_saturates_rather_than_wrapping` | int64 rails, not a wrap | the twin becomes a bare `x << (-s)` |
| `test_the_solid_heat_deposit_survives_a_negative_exponent_that_overflows_int32` | the rail is **reached, never wrapped past** | site 4's int64 routing is reverted (B1) |
| `test_the_same_deposit_at_a_non_negative_exponent_is_unchanged` | **control**: `deposit >> 3`, no rail | M1 stops being neutral on a shipped row |
| `test_a_sub_rail_negative_exponent_deposit_is_the_exact_product` | below the rail, exactly `deposit * 2**-s` | a rounding step or int32 narrow appears |
| `test_the_radiation_fold_takes_a_negative_exponent_too` | the signed fold converts gain AND loss through the same exponent | the fold reverts to `shr_round0_i64` (B4) |
| `test_the_shipped_material_table_is_unmoved_by_M1` | every shipped exponent `>= 0` and unchanged | a row is edited into M1 (rows belong to M2) |

### `tests/test_m1_extra_consumer_sites.py` (new, 347 params)

| test | property | breaks if |
|---|---|---|
| `test_the_sweep_reproduces_the_reference_at_negative_exponents` | gate 0's bit-for-bit property, extended over the negative-exponent axis | `fleck_L_solid_q` keeps the unsigned shift (B8) |
| `test_a_negative_exponent_damps_emission_harder_than_a_positive_one` | the Fleck factor is monotone decreasing in the exponent | the damping stops scaling with it (a clamp at 0 returns) |
| `test_the_raycaster_pair_budget_is_the_exact_signed_scaling` | `floor(x·2^his / 2^shift)`, either sign | the limiter keeps `(x << his) >> shift` (B7) |
| `test_the_combustion_object_site_deposit_scales_with_the_signed_exponent` | halving an object's thermal mass exactly doubles its fuel-bed rise | combustion's object site keeps `deposit >> shift` (B9) |

### `tests/cuda_m1_negative_exponent_check.py` + `tests/test_cuda_m1_negative_exponent.py` (new)

CPU/CUDA tol-0 parity over the deposit, the fold, conduction across a capacity
straddling 1 unit, and a 60-tick mixed-exponent trajectory. Parts 1 and 2 assert
the **exact** integer each config must land, so the check is an oracle and not
merely an agreement between two possibly-identical mistakes.

### Rewritten (premise overturned by the design, not bent to it)

- `test_thermal_mass_axis.py::test_a_row_lighter_than_the_column_floor_is_a_named_refusal`
  → `..._below_the_REPRESENTATION_floor_...`. Its example was a canopy at
  `thermal_mass = 0.5`, which M1 makes **expressible** — that is the patch's whole
  purpose. The *shape* of the property is preserved exactly: refuse by name, never
  clamp. New sibling `test_a_row_at_the_representation_floor_is_accepted_not_refused`
  pins the floor as inclusive (an off-by-one in the refusal alone would not be
  caught otherwise).
- `test_temperature_conduction.py::_capacities` — the Python mirror, updated to
  the lifted floor (finding 3).

---

## 8. What M2 inherits

- `heat_inv_shift` signed end to end, both backends, **seven** consumers gated at
  negative exponents.
- `derive_thermal_mass_exp` as the single derivation feeding both columns.
- `THERMAL_MASS_EXP_MIN = -16` / `conduction::CAP_SHIFT_MIN`, exposed as
  `bp.CAP_SHIFT_MIN`, with a named refusal below it.
- `test_the_shipped_material_table_is_unmoved_by_M1` will **fail by design** the
  moment M2 authors a thin row. That is intended — it marks M1's neutrality claim
  as deliberately spent, and M2 should delete it in the same commit that lands the
  rows.
