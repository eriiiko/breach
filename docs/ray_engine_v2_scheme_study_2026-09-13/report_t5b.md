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

T3's D6 is the execution table; every row below carries its citation in
`config.toml` itself. **`ignition_temp` moved on nothing**, per the brief (D6
§7.4 marks the two `door` rows "Optional" and the brief declines them).

### 2.1 What landed

| row | column | was | now | source |
|---|---|---|---|---|
| `hull`, `steel` | `heat_atten` | 1.0 | **0.85** | Incropera A.11, painted/oxidised mild steel 0.78–0.96 |
| `wood`, `door`, `door_closed` | `heat_atten` | 1.0 | **0.90** | Incropera A.11, planed wood 0.82–0.92 |
| **`foliage`** | `heat_atten` | **0.0** | **0.90** | **R12** — the loss-channel invariant, §2.2 |
| `air` | `conductivity` | 0.024 | **0.0257** | Incropera A.4 at 293 K. The 0.024 was reverse-engineered to make the old log-bucket land on `face[air][air] == 11`; its own comment said so |
| `steel` | `conductivity` | 45.0 | **50.0** | Incropera A.1, band 45–64 — and `hull`, the same steel, already said 50 |
| `wood` | `conductivity` | 0.15 | **0.12** | FPL-GTR-190 ch. 4, transverse at 12 % MC |
| `door`, `door_closed` | `conductivity` | 0.30 | **0.12** | it is wood; 0.30 had no source |
| `glass` | `conductivity` | 1.0 | **1.40** | Incropera A.3, plate glass at 300 K |

**The brief's abbreviated list omits `wood` 0.15 → 0.12.** D6 §7.2 carries it
(−20 %, with the same FPL citation the two `door` rows get), D6 is declared
exhaustive — *"a row missing here is a T5 bug"* — and leaving `wood` at 0.15
while moving `door` to 0.12 would put two rows of the same material on different
conductivities, which is the exact defect the `steel`-vs-`hull` row fixes. So it
is applied, and flagged here as a deliberate deviation from the brief's list.

Also landed, both from D6 §7.5 / T2 D3 site 23, neither a material row:

- `rad_scale_derived` was **already** at R13's `1.6533e-08` (T1 landed it); what
  had *not* moved was the `J_per_count` derivation in the comment above it,
  still reading `0.37013197 J` / `rho_c = 0.7 MJ/(m³·K)` / `194.06 kJ/K` /
  `~139 kg`. Restated at R13's pin: **0.4759 J / 0.9 / 249.5 kJ/K / ~154 kg**,
  and cross-checked against the `[materials.wood]` row's own authored `density`
  × `specific_heat` = 0.899.

### 2.2 R12 became a DOOR, not a value

`MaterialTable.__init__` now **refuses** `flammable && thermal_solid &&
heat_atten == 0` by name, alongside the two existing heat-extinction ingress
invariants. The reason it is a door: such a row is an energy ratchet whose every
channel is correctly booked, so no ledger, digest or conservation identity can
see it — combustion heats the tile, nothing cools it, it climbs to `T_MAX_PHYS`
and re-lights its neighbours forever. Under R10 a small `conductivity` would not
save it either.

Three new tests in `tests/test_optics_ingress.py`:
`test_flammable_thermal_solid_without_a_loss_channel_is_rejected_by_name`
(the door, plus each leg of the conjunction disarming it on its own, so it is
not a blanket "heat_atten must be positive"), and
`test_the_shipped_table_has_no_energy_ratchet` (the invariant asserted on what
actually ships, with a non-vacuity floor of ≥ 3 flammable thermal solids).

**Validated by breaking it**: reverting `foliage.heat_atten` to 0.0 turns the
shipped config into a *load failure* — 4 of the 7 tests in that module go red,
including the two new ones by name.

### 2.3 FINDING — a hull bulkhead is no longer opaque to heat

The single most feel-relevant consequence of step 2, and it needs Erik's eye.

`heat_atten` does **double duty** on the sweep's extinction plane: it is the
cell's emissivity (what it radiates, design v3 row 25) **and** its extinction
coefficient (what it stops), because the model has an absorb channel and a
transmit channel but **no reflect channel**. While every structural row carried
the placeholder 1.0 the two readings coincided and a wall was perfectly opaque.
At the real emissivity 0.85, **a hull cell transmits 15 % of the stream crossing
it.** Physically an opaque steel plate transmits nothing; the 15 % is the
reflected share, which this model has nowhere to put.

Measured on the sweep's own scene (fire-plateau source, `E°⁻¹(Φ)` at a wood
target three cells downrange):

| between source and target | Φ | `E°⁻¹(Φ)` | lights the wood? |
|---|---|---|---|
| clear line | 11 197 494 | **388 game** | yes (ignition 300) |
| **one hull cell** | 2 010 608 | **148 game** | **no** |
| two hull cells | 632 630 | 36 game | no |
| a fully opaque column (`a = ONE`) | 389 472 | 0 game | no — the ambient floor, bit for bit |

So **the gameplay property survives**: a bulkhead still stops a fire lighting
the woodwork on the other side. What changes is that the far side of a single
bulkhead now warms measurably. T3 D6 §7.3 records the same tension for `glass`
("0.3 is right as a shield and ~3× low as an emitter; no row value serves
both") and leaves it to Erik — step 2 makes it live for `hull` and `steel` too.
**Open question 1, §12.**

The three affected tests in `tests/test_sweep_heat_law_properties.py` were split
rather than loosened: the **gameplay** claim (a hull wall drops the target below
wood's own `ignition_temp`; a body behind one absorbs < ¼ of the clear-line
flux) is asserted of `A_WALL`, and the **optical** claim (nothing at all gets
through, bit for bit against the ambient floor) is asserted at `a = ONE`, which
is the only value that actually means it. Both are real, and neither could have
been stated by the merged form.

### 2.4 Two tests left red on purpose, until step 3

`test_temperature_conduction.py::test_face_table_anchor_values` and
`test_eos_p2_sealed_room_energy.py::test_solid_solid_face_shift_unaffected_by_air_conductivity`
pin anchor values of the conduction table under the **old** log-bucket law
(`wood|wood == 8`; the new κ makes it 9). Step 3 replaces that law outright, so
rewriting them here against an intermediate table and again at step 3 would be
churn. They are fixed in step 3, against the derived law.

### 2.5 The gate at step 2

`2539 passed, 21 failed`: the 14 golden-bound from step 1, plus the 2 above,
plus 5 that this step fixed and are now green
(`test_prop_entity::test_foliage_row_values_match_the_ruling`,
`test_radiation_sweep_shadow_wiring`, three in
`test_sweep_heat_law_properties`). Re-run after the fixes: **16 failed**, i.e.
the 14 goldens plus the two deferred table pins.

---

## 3. Step 3 — R10, real conduction rates, solid–solid faces only

### 3.1 The law, and why there are two of them

`SHIFT_AT_REF = 2` at `KAPPA_REF = 50` is retired from `config.toml` and from
`_THERMAL_DEFAULTS`. It was a CFL bound wearing a physics hat — "hull conducts a
quarter of the gap per tick", with no capacity, no tile size and no timestep in
it — and it ran solid–solid conduction 65 000–130 000× too fast.

    solid|solid, gas|gas:   s = round(log2( rho_c_min · dx² / (kappa_hm · dt) ))
    solid|gas:              s = round(log2( rho_c_min · dx  / (h_conv   · dt) ))

`rho_c_min` is the smaller-capacity side — `thermal_mass · THERMAL_MASS_UNIT`
for a solid, `c_v · THERMAL_MASS_UNIT` (= air's real 864.5 J/(m³·K)) for a gas
cell at N = 1 — taken from the same two columns `thermal_mass` itself derives
from, so there is no second source of truth.

**The solid|gas branch is the whole of "do not apply R10 at the wall".** A
solid–gas interface is *convection through a sub-tile boundary layer*, not a
half-tile of still air; `h` is a measured quantity (Churchill & Chu 1975), and
the conductance is the boundary layer **alone** — a convecting cell is
well-mixed, which is what `h` already describes, so the solid's own half-cell is
not put in series with it (T3 §8 q5's second option). Note `dx` rather than
`dx²`: `h` is a conductance per unit area and `kappa` is not.

### 3.2 The derived table, which reproduces T3 §3.2 to the digit

| face | derived here | T3 §3.2 predicted | shipped before |
|---|---|---|---|
| `hull`\|`hull`, `steel`\|`steel`, `hull`\|`steel` | **18** | 18 | 2 |
| `glass`\|`glass` | **22** | 22 | 6 |
| `wood`\|`wood` and every cellulosic pair | **24** | 24 | 8 |
| `hull`\|`wood`, `steel`\|`wood`, `*`\|`door_closed` | **23** | 23 | 6–7 |
| `hull`\|`glass`, `steel`\|`glass` | **21** | 21 | 5 |
| `wood`\|`glass`, `door`\|`glass` | **23** | 23 | 7–8 |
| `air`\|`air` | **16** | 16 | 11 |
| **`air`\|every solid** | **10** | 10 | **10 — unchanged** |
| `furniture` / `kindling` / `foliage`, any face | NO_FACE | — | NO_FACE |

`kappa == 0` still means NO_FACE on both laws, so the three fixture rows keep
having no conduction face at all: opening their convection face is T3 §8 q3,
Erik's.

`self_shift` stops being an independently-computed log bucket — a second source
of truth for the same rate — and becomes the table's own diagonal. Nothing in
the engine reads it.

### 3.3 The tick and the tile size

- **`dt`** is injected by `MaterialTable.from_config` from `[clock]
  ticks_per_second`, through a small read-only override view, rather than being
  copied into `[physics.thermal]`. The table can therefore never disagree with
  the clock the engine actually runs at.
- **`dx`** is a new `[physics.thermal] tile_size_ref_m = 0.333`. **This is the
  one design call step 3 had to make and it is deliberately the conservative
  one.** Under R10 the table IS tile-size dependent (solid faces as `1/dx²`, gas
  faces as `1/dx`) while the material table is global and `tile_size_m` is per
  level. T3 §8 q9 asks Erik to choose between deriving per level and keeping one
  global table; a global reference keeps today's architecture, is the same
  position `rad_scale_derived` already occupies, and leaves the question
  genuinely open instead of answering it silently. On a 1.0 m level
  (`airlock_demo`) every solid face is 3 shifts too fast and every gas face 2.
  **Open question 3, §12.**

### 3.4 The tests, and breaking them

`test_face_table_anchor_values` is **replaced**, not repaired. It pinned
`hull|hull == 2` and `wood|wood == 8` — which *are* `SHIFT_AT_REF` at
`KAPPA_REF`, so it asserted the constant back to itself and would have passed
for any physics whatsoever. Its replacements:

| test | property |
|---|---|
| `test_face_table_is_the_material_s_own_diffusivity_not_a_stability_anchor` | every solid pair's shift recomputed from the row's own `conductivity` and the `thermal_mass` currency |
| `test_a_solid_gas_face_is_convection_and_did_not_move_under_r10` | every solid\|gas pair is the derived convection shift 10, **and** that pure conduction there would be ≥ 6 shifts weaker — so "completing" R10 at the wall is caught by name |

`test_eos_p2_sealed_room_energy.py`'s redundant `2`/`8` pins move to `18`/`24`.

**Validated by breaking it:**

| control | what was broken | result |
|---|---|---|
| A | the solid\|gas branch put on Fourier (the naive-R10 error the design warns about) | RED — `test_a_solid_gas_face_is_convection_and_did_not_move_under_r10` by name |
| B | the capacity dropped out of the solid rate (back to a κ-only bucket) | RED — `test_face_table_is_the_material_s_own_diffusivity…` and two in `test_eos_p2_sealed_room_energy` |

### 3.5 Two pre-existing tests whose premise R10 changed — findings, not repairs

**(a) A corner hull tile no longer warms.**
`test_sealed_room_energy_conserved_and_walls_warm` asserted `np.all(temperature[hull_mask] > 0)` —
the *whole* ring. That held only because solid–solid conduction ran 65 000× too
fast and carried heat round the corners; a corner tile's four orthogonal
neighbours are all hull, so it touches no gas. The assertion is now split into
the real property — **every gas-facing wall tile warms, and the corners stay at
ambient** — which makes a returning stability anchor visible as a corner that
mysteriously warmed. Control B fires on it.

**(b) The sub-dead-band drain is bigger than "negligible" suggested.**
`test_sealed_room_with_one_hull_face_exposed_drains_monotonically` required the
space-facing channels to beat conduction's counted truncation by **10×**.
Measured after R10: `e_space = 2.70e+10`, `e_trunc = 1.10e+10` — a ratio of 2.5.

The cause is T3 §3.4's dead band. Below `2^s/65536` K (4 K for steel, 256 K for
wood) a face moves **nothing** to its neighbour while the hot cell still loses
one raw count to the floor division — a one-way sink with no counterparty. T3
called it "1.3 K/hour, 0.04 K over a crate burn, negligible in play", which is
true in real units; over 400 ticks against a *small* radiative drain it is 40 %
of the total. The assertion is restated as the property that survives (the
exposed face is the dominant channel, and every count is attributed), and
whether a sub-dead-band face should instead be a no-op is **T3 §8 q4, Erik's —
open question 4, §12**. Nothing here bends the law to the test.

### 3.6 The gate at step 3

**`2547 passed, 14 failed`** — back to exactly the golden-bound set from step 1.
The two tests step 2 deferred are green.

---

## 4. Step 4 — the vacuum mask

    no_cond[i] = !ts[i] && ( is_vacuum[i] || cap_real_[i] == 0 )

Every face such a cell owns becomes NO_FACE, applied per **cell** from both
ends so it is symmetric by construction and cannot produce a one-sided face.
Landed on both backends: a `no_cond_` plane built beside the capacities on the
CPU (same frozen inputs, so it is as pass-invariant as they are), and the same
predicate inline in `cuda_temperature.cu`'s `temp_conduct` (all three terms are
already kernel arguments).

**No threshold needed, and none should ever be added.** `kappa` is
density-independent (kinetic theory: `n` and the mean free path cancel) until
the gas goes free-molecular, which for a 0.333 m tile is 0.0207 Pa. `n_bulk` is
Q16.16 at 1.0 = 1 atm, so one raw count is **1.546 Pa** — the Knudsen threshold
is **0.013 of a single LSB**. `N_raw == 0` is the only free-molecular state the
field can represent (T3 D4 §5.2).

One CUDA-only detail: the CPU's masked early-out relies on `de_gas_.assign(n, 0)`
having already zeroed the parked gas sums, while the CUDA kernel writes
`de_gas[i]` per cell — so the device's masked branch writes the zero explicitly.
A masked cell that is *also* accountable (`cap_real == 0`, not vacuum, not
solid) is reachable, so this is not theoretical.

### 4.1 Validated by breaking it

| control | what was broken | result |
|---|---|---|
| A | the mask forced to `false` (i.e. deleted) | RED — `test_a_vacuum_cell_owns_no_conduction_face` |
| B | the `!ts[i]` guard dropped — the mask written as `is_vacuum \|\| cap_real == 0` | RED — `test_an_intact_hull_tile_still_conducts_even_though_it_is_is_vacuum` |

Control B is the one that matters: an intact hull tile *is*
`is_vacuum && solid && thermal_solid` (Pass 0 says so), so without the guard
the mask would have silently severed every space-facing bulkhead from the hull
behind it — and nothing else in the suite noticed. The test exists because that
failure has no other symptom.

The first test covers both disjuncts separately (`is_vacuum` = a real breach;
`cap_real == 0` = a decompressed interior, which `is_vacuum` does not mark) and
asserts the hot wall keeps **exactly** what it had — no one-count-per-tick
dribble either.

### 4.2 The gate at step 4

- **`2547 passed, 14 failed`** — still only the golden-bound set.
- **CUDA lockstep tol-0**: P66 PART 1 all 121 configs, PART 2 120 ticks, PART 3
  30-tick engine A/B, all bit-identical CPU vs GPU. The counters *moved* between
  steps 3 and 4 (`e_cond_cap_sum` 1 768 198 585 225 → 1 746 660 702 135,
  `e_cond_trunc_sum` 1 416 647 458 → 1 416 716 699), so the mask is doing
  something, and both backends do the same something.

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
