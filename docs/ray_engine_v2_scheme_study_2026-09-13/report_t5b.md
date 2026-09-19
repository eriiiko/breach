# T5b — THE FLIP (issue #12)

> **Status**: IN PROGRESS. Written incrementally, one section per step, committed
> as each step lands. A section still reading `PENDING` has not landed.
>
> **Branch** `12-t5b-the-flip` off `fire-12`, worktree
> `C:\Users\steen\projects\breach-t5b`. HUMAN-TEST: built, gated, **not merged**.
>
> **Depends on**: `docs/thermal_model_v2_design_2026-09-19.md` (R1–R14, §5 T5,
> §6 properties) · `report_t2.md` (`c_v`) · `report_t3.md` (material table, `h`,
> vacuum mask) · `report_t3b.md` (density derivation) · `report_t5a.md` (the two
> widenings + three findings) · `report_t4.md` §5.4 (dangling names).

---

## 0. What this patch is

The moment the thermal model changes. Nine ordered steps, each its own commit,
each landing a value or a structure the arc already derived and verified
elsewhere. Nothing here is chosen; what is decided here is only *how* the
settled numbers are wired, and every such call is named in its section.

**Goldens go red at step 1 and stay red until step 9.** That is the plan, not a
regression: `GOLDEN_AGGREGATE`, `test_b6_logic_golden` and `test_w6_armory` are
whole-trajectory digests, and the first behaviour change moves them. Step 9
re-baselines them ONCE, with the rationale enumerating every contributing
change. **The CUDA lockstep legs are NOT in that set** and stayed green the whole
way: every `test_cuda_*` failure in the intermediate steps is its own PART 3
golden leg, with PARTS 1–2 (the GPU-vs-CPU bit identity) passing.

**Baseline on this box** before any T5b code, both backends built:
`2557 passed, 4 skipped, 4 xfailed, 0 failed`; `GOLDEN_AGGREGATE =
167b96bddfe37c0d256afed4d3b9271371fcaf3edd7557e02cf685a17208953f`.

---

## 1. Step 1 — `c_v` -> 0.0076849

**The verification target, met.** `currency_audit_t2.py` M1, at the derived
`c_v`: **`received / delivered = 0.99996993`** (it was `0.00768487` before T5a's
structural fix, i.e. 99.23 % of every joule conducted into air was destroyed at
the face). The residue is the endpoint truncation, not a currency error.

### 1.1 What landed

| # | site | what |
|---|---|---|
| 1 | `config.toml` `[physics.thermal] c_v` | `1.0` -> **`0.0076849`**, with the derivation in place of the "no real-gas anchor yet" placeholder note. T2 D3 site 1 |
| 2 | `temperature_solver.cpp` Pass 1 | `recip_cv = make_recip(c_v_q / 65536)` — the EXACT inverse of the Q16.16 `c_v` the conduction capacity is built from, not a second quantization of the same config scalar. T2 D3 site 3b |
| 3 | `cuda_temperature.cu` | the same, with `c_v_q` hoisted above `recip_cv` (it is now the source) |
| 4 | `combustion.cpp` + `cuda_combustion.cu` | the same, **beyond T2's list**: these are a second deposit site into the same gas cells, and at 0.0724 % disagreement two deposits would have priced a heat count differently. One representation now, engine-wide |
| 5 | `tests/cuda_conduction_check.py`, `cuda_thermal_mass_check.py`, `cuda_combustion_check.py` | `c_v` 1.0 -> shipped. A parity gate frozen at a dial the engine no longer ships proves parity of a path nobody runs. T2 D3 site 8 |
| 6 | `tests/test_fixed_point_i64_twins.py` | NEW `test_staged_wide_chain_sub_unit_c_v_obeys_the_derived_bound`. T2 D3 site 9 |
| 7 | `tests/test_thermal_mass_axis.py` | the gas-branch leg now asserts the LAW, not an accidental identity (§1.3) |
| 8 | `tests/test_air_boundary.py` | gate 2's inflow rail — a finding, §1.4 |
| 9 | `tools/e2b_floor_reciprocal_probe.py`, `tests/test_eos_p2_sealed_room_energy.py` | T2 D3 sites 19, 20 — the probe now reads the shipped dial; the stale "the SAME values config.toml now ships" docstring corrected |

`c_v_safe` in `temperature_solver.cpp` Pass 1 became dead with (2) and is
deleted: the `c_v <= 0` guard is `c_v_q`'s now, at the top of `step()`, and one
guard is the point. That also keeps the TU's float ratchet **at its documented
floor of 6/4 rather than above it** — the ratchet counts LINES carrying the
token, comments included, and it caught the first draft of this patch at 9.

### 1.2 The one place T2's sub-unit `c_v` needed a derived bound

`test_fixed_point_i64_twins.py`'s one-LSB claim was explicitly written as a
`c_v >= 1` claim. The shipped value left that range, so the sub-unit regime got
its own test with its own DERIVED bound rather than a widened blanket one. With
`A = deposit·recip_n`, `stage1 = floor(A/2^16)`, `A/2^16 = stage1 + f`:

    0 <= narrow - staged <= floor(f·recip_cv / 2^32) + 1 <= (recip_cv >> 32) + 1

which is 1 at `c_v >= 1` and **131** at the shipped value. The test asserts that
bound AND that the separation genuinely exceeds one LSB, so it cannot go vacuous
if someone re-tightens it by accident.

### 1.3 An accidental identity that stopped testing its own law

`test_permeable_thermal_solid_takes_the_SHIFT_convert_not_the_gas_deposit`
asserted the gas branch with `got == deposit`. Its own comment says the law is
`deposit / (N·c_v)` — and at `c_v == 1`, `N == 1`, those coincide. It now
asserts the engine's own integer chain through the engine's own ONE `c_v`
representation, plus `got > 100·deposit` so the divide cannot silently vanish
again.

### 1.4 FINDING — `test_air_boundary` gate 2 asserted the sign of settled noise

The only pre-existing test that went red for a non-golden reason, and the
diagnosis is not "the currency broke it".

`boundary_flux()` is a **per-tick** rail (`eos_solver.cpp:348-352` zeroes it at
every `step()` entry). Gate 2 read it ONCE after 80 ticks and asserted it
negative, under the comment *"the rail recorded the boundary exchange, negative
for a net inflow"*. But the refill is over by tick 3. Measured, with the
**running total** and cross-checked to the last count against the interior's own
species mass (the cumulative rail is exactly minus the interior mass delta on
every tick):

| tick | 1 | 2 | 4 | 9 | 19 | 39 | 79 |
|---|---|---|---|---|---|---|---|
| cumulative rail, o2 | −1 365 298 | **−1 594 720** | −1 389 238 | −558 724 | +11 010 | +156 210 | +138 922 |

so the 80-tick **net is a small OUTFLOW** — the room ends ~5000 game-K hot and
vents back out — and the old assertion was reading the last tick alone: **−1678
at the old `c_v`, +46 at the new one**. A 0.1 % trajectory change flipped a
coin-flip. The property the docstring names is real and is three orders of
magnitude clear of the sign boundary: the running total bottoms out at
**−1 594 720 on tick 2, identical at both `c_v` values**. The gate now asserts
that.

**Validated by breaking it**: the old form and the new one are each other's
negative control. The old (last-tick) read passes where the new one fails
(+46 vs a required < −1e6), and the new (running-minimum) read fails where the
old one passes — the two cannot both be measuring the same thing, and only one
of them is measuring the refill.

### 1.5 The gate at step 1

| | |
|---|---|
| suite | **2544 passed, 14 failed** — all 14 golden-bound: `test_b6_logic_golden` (2), `test_w6_armory` (1), and 11 `test_cuda_*` whose GPU-vs-CPU legs passed and whose PART 3 golden leg reports the same new aggregate `5e0594d8ad4001c9…` |
| CUDA lockstep | **tol-0, every plane and counter** — P66 conduction 121 configs + 120 ticks + 30-tick engine A/B; P69 combustion edge+fuzz+2×120 ticks; P65 EOS 120 ticks over six digests and five rail counters |
| `received/delivered` | **0.99996993** |

---

## 2. Step 2 — the material rows (T3 D6)

PENDING

---

## 3. Step 3 — R10, real conduction rates, solid–solid faces only

PENDING

---

## 4. Step 4 — the vacuum mask

PENDING

---

## 5. Step 5 — `hp`/fuel separation (R14) and `H_fuel`'s value

PENDING

---

## 6. Step 6 — THE FLIP

PENDING

---

## 7. Step 7 — `cool_shift` deleted

PENDING

---

## 8. Step 8 — `k_leak` live at 0.10

PENDING

---

## 9. Step 9 — digest spec v6 and the ONE golden re-baseline

PENDING

---

## 10. The gates, and how each was validated by breaking it

PENDING

---

## 11. What Erik should expect to see when he plays it

PENDING

---

## 12. Open questions for Erik

PENDING

---

## 13. Findings

PENDING
