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

The step with the most in it, and the one where two of the arc's settled numbers
turned out not to be shippable. Both are reported with the measurement that
blocked them; neither was quietly dropped.

### 5.1 R14's fuel half — landed

`fuel_per_o2` stops being one global 0.7 and becomes a **derived per-material
column** projected to a per-tile plane, on the same single seam `fuel_recip` and
`fire_T_ext_plane` already use:

    fuel_per_o2[mat] = hp[mat] * KG_FUEL_PER_N_O2 / (density[mat] * V_tile)

so that spending the whole `wall_hp` bar consumes exactly the tile's real
combustible mass, while `hp` keeps its structural meaning — which is R14's own
wording, *"fuel separates from hp; hp stays structural"*. No `hp` value moved.

`KG_FUEL_PER_N_O2 = 0.37162 kg` is three cited constants and nothing fitted:
one unit of `N_O2` is 0.36878 kg of O2 (`pV/RT` at the engine's own ambient),
times Huggett's 13.1 MJ/kg-O2 = 4.831 MJ, divided by wood's *effective* heat of
combustion 13 MJ/kg (Drysdale Table 1.13; Babrauskas).

| row | was | now | store |
|---|---|---|---|
| `wood`, `foliage` | 0.7 | **0.14492** | 414.0 units of N_O2 |
| `furniture` | 0.7 | **0.07246** | 414.0 |
| `kindling` | 0.7 | **0.01932** | 414.0 |

Surface: a derived column + `fuel_per_o2_q16` in `materials.py`;
`GameMap.fuel_per_o2_plane` built and patched on the one seam; a nullable
trailing `fuel_per_o2_plane` argument through `CombustionSolver::step`,
`cuda_combustion.cu`, both bindings and `physics_runner`. `nullptr` keeps every
direct-binding caller pre-R14 bit-for-bit.

### 5.2 FINDING — R14's separation is real but very nearly inert, because a THIRD consumer owns `wall_hp`

**Measured, and this is the number that matters for the human test.** Same
bench, same crate, only the fuel exchange rate changed:

| fuel rate | hp to 50 % | to 10 % | to 1.7 % | peak I |
|---|---|---|---|---|
| the retired 0.7 | 12.8 s | 53.8 s | **105.9 s** | 0.535 |
| **T5b's derived 0.0725** (9.66x cheaper) | 12.8 s | 53.8 s | **105.4 s** | 0.536 |

A 9.66x change in the fuel exchange rate moves the burn duration by **0.5 %**.

The reason is that `wall_hp` has a **third** consumer that R14 does not name:
`FireSimulation`'s structural `wall_damage = 0.36` hp/s/intensity, which
`config.toml` itself calls *"the burn-out brake -> now the DURATION dial"* and
which is **Erik's own 3-minute fuel-out ruling of 2026-09-06**. That dial empties
the bar on a timer, regardless of how much wood is in the tile, and it dominates
the combustion-side cost by roughly 10:1.

So after step 5, `hp` is still doing three jobs, not two: structural integrity,
combustion fuel, and a feel-tuned burn-down timer. R14 made the *combustion*
share physical; making the STORE physical needs `wall_damage` and the fuel store
to stop sharing a field. **Open question 5, section 12.**

Two consequences worth being explicit about: (a) burn durations are essentially
unchanged, so Erik is not chasing a moving target when he plays this; (b) the
fuel distinction between `wood`, `furniture`, `kindling` and `foliage` now
collapses in the *store* — every flammable row is authored at rho = 555, so
under R14 every flammable tile is the same 154 kg block and holds the same 414
units — but not in *behaviour*, because `wall_damage` and `hp` still separate
them. Restoring a real store distinction means authoring those three OBJECT
rows' bulk densities (a crate stack is ~120 kg/m3, not 555), which is T3 q1/q2
and also moves `thermal_mass`. **Open question 2, section 12.** Erik's
2026-09-07 foliage ruling (*"fuel ~= 2x furniture"*) survives in `hp`, which is
what the engine actually burns down.

### 5.3 FINDING — `H_fuel`'s derived value is unshippable, and the reason is a missing channel

T3 derives `H_FUEL_M = 14870, H_FUEL_SHIFT = 9` — 7.613e+06 counts = 3.62 MJ,
the 75 % plume share of Huggett. **Applied, it makes the gas run away, and three
pre-existing gates say so by name:**

| gate | what it reported |
|---|---|
| `test_e1_hot_rail::test_no_rail_hits` | `T_MAX_PHYS` engaged **17 206x** against a budget of 8 |
| `test_eos_p4_combustion::test_thermal_spike_is_pre_existing_not_a_p4_regression` | peak **15 992 game** — the gas pinned at `T_MAX_PHYS` — against a limit of 9000 |
| `test_ps1_smoke_roundtrip::..._never_exceed_start` | **bulk gas mass minted**, +34 counts at tick 2 |

Bisected on this branch (`H_FUEL_M = 4.0`, shift swept):

| `H_fuel` | 4.0 | 8.0 | 16 | 1024 | 4096 and above |
|---|---|---|---|---|---|
| gates failing | **0** | 1 | 1 | 2 | **3** |

**The shipped 4.0 is already at the ceiling.** The derived value is 1.9 million
times what the gas side can absorb.

**The reason is structural, and the design states it itself.** Design v2 section
3.1: *"gas is heated only by combustion until P5 gives it radiative
absorption"*. A gas cell has **no radiative loss channel at all** in this model —
it can be heated, advected and expanded, and that is the whole list. Handing a
medium with no loss channel 75 % of a real fire's heat release is exactly the
defect R12 fixed for foliage, one field over: an energy ratchet. The missing
physics is the same too — a real plume sheds its heat by radiating and by
entraining ~30 tile-volumes of cold air per second, while our deposit lands in
one cell and its open faces.

So **`H_fuel` stays at 4.0** and the plume share is left unmodelled until P5.
Raising it now would be tuning a dial to paper over a missing channel, which is
the failure mode this arc exists to stop (design section 1: *"a channel is
computed and booked, or it does not exist"*). **Open question 6, section 12.**

`H_BED_M` **does** move, 18125 -> **19827** (+9.4 %): the fuel-surface 25 %
share of the same Huggett split, which lands in the SOLID, and solids do have a
radiative loss channel now. That the dial P-K0 found by feel was 9 % from the
physics is the strongest agreement between the feel pass and the derivation
anywhere in this arc.

### 5.4 T5a's finding 3 is now obsolete — the currency fix created the missing backstop

T5a section 6.4 established that **nothing** gated the gas-side combustion
yield: doubling `H_fuel` through `H_FUEL_SHIFT` left all 2557 tests green, and
it concluded that T5b's `H_fuel` move *"rests entirely on Erik playing it"*.

That is no longer true, and step 1 is why. At the corrected `c_v` the gas is
130x lighter, so the same deposit makes 130x the temperature — and a 2x
`H_fuel` now trips `test_e1_hot_rail` (see the bisect above). **The currency fix
gave the gas-side yield the automated gate it was missing.** It is also what
turned an unbacked feel judgement into the measurement in 5.3.

### 5.5 Validated by breaking it

| control | what was broken | result |
|---|---|---|
| A | the CPU solver's plane lookup forced to the scalar fallback | RED — `test_the_solver_actually_charges_the_plane_s_rate_not_the_scalar`, **and nothing else in the suite** |
| B | `fuel_per_o2` derived as the flat 0.7 again (the retired global's shape) | RED — the two R14 property tests by name |
| C (semantic) | run the engine at a `KG_FUEL_PER_N_O2` that reproduces the old 0.7 rate | the 5.2 measurement — the burn moves 0.5 %, which is the finding |

Control A is the one that had to exist. Every other gate on this axis is either
Python-side (the derived column, the projected plane) or a CPU-vs-GPU
comparison, and a solver that simply ignored the new argument would pass all of
them — the column still derived, the plane still projected, and both backends
still agreeing *with each other, on the wrong law*.

The CUDA check gained the matching leg: `cuda_combustion_check.py` (k3) runs the
same 15 fuzz states with a **non-uniform** plane (a checkerboard of the derived
`wood` and `kindling` rates, so a hoisted load is caught) and carries its own
non-vacuity control. Result: **15 configs bit-identical CPU vs GPU, 15/15 of
which moved `wall_hp` against the scalar.** Without it the device path would
have been entirely ungated.

### 5.6 The gate at step 5

**`2552 passed, 14 failed`** — still only the golden-bound set.

---

## 6. Step 6 — THE FLIP

### 6.1 What landed

| # | site | what |
|---|---|---|
| 1 | `physics_engine.cpp`, CPU branch | the temperature fold's radiation source is `rad_net_sweep`, the sweep's own plane, written at step 2b of the same `step_tail`, on the same `temperature` the pass is about to read |
| 2 | `physics_engine.cpp`, CUDA branch | the same — one source of radiative truth whichever temperature backend is selected |
| 3 | `temperature_solver.cpp` Pass 1 | the maximum-principle clamp goes LIVE: `rad_fluence` and `emissive.table()` are passed instead of `nullptr, nullptr`. It could not have been live before — clamping against a Φ from a different law is meaningless |
| 4 | `cuda_temperature.{h,cu}` | **the clamp's GPU twin**, in the same position (between the saturating add and the rails, before the applied-ΔT booking), through the same `FP_HD e_inv_q`. `rad_fluence` + the `E°` table are H2D'd; `TEMPERATURE_ENERGY_SLOTS` 13 → **14**, with `C_RAD_CLAMP = 13` **appended** so no pinned index moves |
| 5 | `exchange.py` | units absorb from **`rad_flux_sweep`** — a ledger EXIT in the fold's own currency — instead of the retired cast's undebited incident estimate. The body share is `d − a` on `dyn_heat_atten_q`; a unit MAX-stamps `heat_atten = 1.0`, so a marine on air absorbs the whole stream crossing its cell and passes ambient on |
| 6 | `config.toml [combat]` | `heat_flux_to_temp` 8.0 → **4701.2**, DERIVED (§6.2) |

### 6.2 The unit burn band, derived from the literature

With a physical currency, a body's absorbed counts convert to irradiance:

    kW/m² = phi · 65536 · J_per_count · ticks_per_second / A_rad = phi · 224.8

(`J_per_count` = 0.4759 J at R13's pin; `A_rad` = 4·0.333·2.5 = 3.33 m², the
same four lateral faces the emission scale is denominated over).

**The anchor is 2.5 kW/m²** — the standard human pain / firefighter-exposure
threshold (SFPE Handbook; Drysdale ch. 2's tenability table) — placed exactly at
the edge of the survivable band, `T_felt == temperature_max`. Everything else
falls out:

| irradiance | what it is | `T_felt` | marine |
|---|---|---|---|
| 1 kW/m² | bright sunshine | 36 | no damage |
| **2.5** | **pain in ~10 s** | **60** | **the band edge — the base 1 HP/s, 100 s to incapacitate** |
| 5 | blistering in ~30 s | 100 | 3 HP/s → 33 s |
| 10 | 2nd-degree in ~10 s | 180 | 7 HP/s → 14 s |
| 20 | untenable | 340 | 15 HP/s → 6.7 s |
| 50 | inside the flame | 820 | 39 HP/s → 2.6 s |

`tests/test_unit_heat_damage.py` now states its fixtures in **kW/m²** rather
than bare `phi` numbers, and asserts that band. The old fixture read
*"phi ~ 1: a faint warmth"* — which was the retired cast's fitted scale
talking; under the derived currency `phi = 1` is **225 kW/m²**.

---

### 6.3 THE FINDING — the flip exposes a 65 536x error in combustion's fuel-bed deposit

**This is the most important thing in the patch, it is exact rather than
estimated, and it is Erik's to rule on. Open question 7, section 12.**

#### What happens

After the flip, a burning crate does not plateau — it ratchets to the
`T_MAX_PHYS` rail, and the room follows it. Measured on the canonical
single-crate bench at the SHIPPED dials:

| | before the flip | after |
|---|---|---|
| crate peak temperature | 1 730 game | **15 194 game** |
| room steady-state (far field) | 525 game | **11 247 game** |
| burn to 1.7 % hp | 105 s | 31 s |

Four pre-existing runaway guards say so by name, and they are the right gates:
`test_eos_p4_combustion::test_thermal_spike_is_pre_existing_not_a_p4_regression`
(peak 15 998 against a 9 000 limit), `test_e2e_1_sealed_room_fire_self_starves`,
`test_e2e_2_breach_vents_o2_and_kills_fire`,
`test_payoff_orderings_perturbation_robust`, plus
`test_ps1_smoke_roundtrip` (bulk gas mass minted at extreme T) and the three
unit-digest scenarios.

#### Why — isolated, then measured exactly

**Isolated first.** Reverting ONLY the fold's source (`rad_net_sweep` back to the
old cast's `rad_net`, clamp still live) makes the same fixture plateau at 8 300
game instead of railing at 15 998. So the trigger is the emission scale, not the
clamp and not the material rows.

**Then measured.** Driving `CombustionSolver::step` directly with one source and
one O₂ cell, `H_BED_M = 1.0, H_BED_SHIFT = 0`:

    burn drawn        = 16 384 raw  = 0.25 atm-tile of O2 = 2.881 mol = 0.0922 kg
    heat deposited    = 16 384 raw                  (so deposit_raw == burn_raw · H_bed)
    PHYSICAL release  = 0.0922 kg · 13.1 MJ/kg-O2   = 1.208 MJ   (Huggett 1980)
    ENGINE deposit    = 16 384 · 0.4759 J           = 7 797 J

So at `H_bed = 1` the engine pays 7 797 J where the physics says 1.208 MJ (or
302 kJ for the 25 % fuel-surface share). The fuel-bed multiplier that pays the
derived share is therefore

> ### `H_bed = 302 000 / 7 797 = 38.73`
> and the shipped `H_BED_M · 2^H_BED_SHIFT = 19 827 · 128 = 2.538e+06` is
> **65 536x** that — exactly `2^16`, to the last digit.

The same factor resolves step 5's puzzle: T3's derived `H_fuel = 7.613e+06` is
**116.2** in the engine's own expression, and the shipped 4.0 is 29x too small
rather than 1.9 million times. T3 derived counts *per unit of N_O2*; the engine
multiplies a **raw Q16.16 burn count** by the constant, so the two differ by
2^16. The derivation was right and its transcription into the dial was not.

#### What it means

**The engine's fire has been running on a compensating pair of errors.** The
fuel-bed deposit is 65 536x the physical heat of combustion; `report_p2b.md`
measured the old cast's fitted `rad_scale` at **2 419x** the derived one. A
hugely over-strong source against a hugely over-strong sink produced a
plausible-looking 1 730-game crate. The flip removes one half of the pair and
the other half is left standing on its own.

**And correcting it alone does not fix the game either** — which is why this is
a ruling and not a patch. At the honest `H_bed = 38.73` a burning crate receives
~17.7 kW into a tile whose heat capacity is **249.5 kJ/K** (R14: a tile is a
filled 154 kg block of wood), i.e. **0.07 K/s**. It would warm by about 2 K over
its whole burn, never reach its own `hot` gate, and go out. That is physically
correct and gameplay-fatal, and the missing piece is named in the design
already: a real fire heats a thin surface layer, not the bulk. **That is the
two-node skin/core solid — issue #68, deferred by R6** (*"I don't want to
increase the complexity in the model before we have a working simulation"*).

So the three states are:

| state | `H_bed` | a burning crate |
|---|---|---|
| shipped (this branch) | 2.538e+06 | ratchets to the `T_MAX_PHYS` rail |
| the honest derivation | 38.73 | warms ~2 K and goes out |
| honest + #68 | 38.73 | a thin skin reaches flame temperature; the bulk does not |

**Nothing was tuned to get out of this.** Fitting a value between the two would
be exactly the failure mode design §1 names (*"a channel is computed and booked,
or it does not exist"*), and it would re-create the compensating pair in a new
place.

### 6.4 The same finding, one layer down: radiative spread is real physics and it is slow

`test_fire_feedback::test_spread_is_radiation_only_no_cellular_stencil`
asserted the near (air-separated) target reaches `> 100 game`. That anchor came
from P-F1a's frozen dials on the old cast. On the sweep it reaches **0.1 game**,
and that number is right: a 443-game (736 K) surface radiating across one air
cell delivers ~12 kW to a 154 kg block with a 249.5 kJ/K capacity, i.e. 0.05 K/s.
Real fire spread to heavy timber takes minutes at 10–20 kW/m².

This is design v2 R6's accepted gap — *"thick structure barely auto-ignites"* —
arriving as a measurement. The test now asserts the near/far CONTRAST, which is
the property it was written to protect, with the anchor removed and the number
recorded.

### 6.5 The gate at step 6

**`2544 passed, 22 failed`** — the 14 golden-bound, plus **8 that are the
runaway in section 6.3** and are deliberately NOT bent:

- `test_eos_p4_combustion` ×4 — the runaway guards, doing their job
- `test_ps1_smoke_roundtrip` — bulk gas mass minted at the rail
- `test_unit_state_digest` ×2, `test_s3c_unit_state_digest` ×1 — their fire
  scenarios no longer play out the same way

Three that DID move legitimately were restated rather than left red, each
because its anchor belonged to the law being deleted:
`test_unit_heat_damage::test_warm_room_survivable` (fixtures re-expressed in
kW/m²), `test_fire_feedback` (§6.4), and the three sweep tests from step 2.

---

## 7. Step 7 — `cool_shift` deleted

R1: *"`cool_shift` is deleted. A hand-rolled radiative loss
(`temperature_solver.cpp:136` says so). The sweep computes the real one;
keeping both counts it twice."*

### 7.1 What went

| surface | what |
|---|---|
| `temperature_solver.cpp` | **Pass 3** — the whole `T -= T >> cool_shift` relaxation, its vacuum-exposure gather and its two counter writes |
| `temperature_solver.h` | the three dials + their eight accessors, the `cool_shift_grid` parameter, `e_cool_sum`, `e_thermostat_sum`, and the identity's `+ e_thermostat_sum` TERM |
| `cuda_temperature.cu` | the **`temp_cool` kernel**, its launch, the `d_csg` device plane, the `vac_offset` host precompute, and slots `C_COOL` / `C_THERMOSTAT` |
| `cuda_temperature.h` | the four parameters, and `TEMPERATURE_ENERGY_SLOTS` **14 → 12** |
| `physics_engine.{h,cpp}` | the REQUIRED `cool_shift_grid` argument on `step_tail`, both `temperature.step` call sites, and the by-index counter fold |
| `bindings.cpp` | the three `def_property` dials, the two `def_readonly` counters, and the `cool_shift_grid` / `cool_shift_floor` arguments on three entry points |
| `materials.py` | the per-material `cool_shift` column, its validation, `_COOL_SHIFT_MAX`, and the `COOL_SHIFT` default |
| `gamemap.py` | the `cool_shift` plane, its build, its `on_tile_changed` patch, and its `_RESIDENT_SYNCED` membership |
| `physics_runner.py` | the three config binds and both `step_tail` arguments |
| `config.toml` | `COOL_SHIFT`, `COOL_SHIFT_VACUUM`, the axis block, and **all ten per-row `cool_shift` columns** |
| tests + tools | 14 modules: solver-fixture assignments, counter reads, dials dicts, CUDA-check counter lists, `storm_probe`'s three overrides, `eos_p5_bake`'s dial row, `storm_ledger`'s counter list |

`o2_vacuum_thresh` **survives**: Pass 0's open-vacuum wipe asks a different
question — is this cell a thermal medium at all — and still asks it.

### 7.2 The two pinned positional slots, handled explicitly

The brief flags this and it is the one place a silent bug was available.
`cuda_temperature.cu`'s `C_*` enum is **pinned positional**: `physics_engine.cpp`
folds the counter block into the solver's fields **by index**. Removing two
entries renumbers every survivor below them:

| counter | was | now |
|---|---|---|
| `e_cool_sum` | 3 | **deleted** |
| `e_vac_wipe_sum` … `e_solid_cond_sum` | 4…11 | **3…10** |
| `e_thermostat_sum` | 12 | **deleted** |
| `rad_clamp_hits` | 13 | **11** |

The enum, the by-index fold, the binding's `make_tuple`, and both CUDA checks'
counter lists are edited in this one commit. **The gate that proves it**: the
CUDA conduction lockstep compares every counter CPU-vs-GPU by NAME, and it is
still tol-0 across 121 configs, a 120-tick trajectory and the 30-tick engine
A/B — a renumbering slip would have shown up there as a counter mismatch, not
as a silent zero.

### 7.3 The #54 identity closes with the term REMOVED

`test_thermostat_books.py` was the gate for Erik's 2026-08-30 thermostat ruling.
Its identity loses a term rather than gaining a zero:

    Δ solid_energy_books_sum == e_solid_deposit_sum + e_solid_cond_sum

and its leg (b) is INVERTED: it used to assert `e_thermostat_sum < 0` and a
monotone decay toward ambient, on the reasoning that *"the thermostat is the
only channel with anywhere to put net energy"*. It now asserts that neither
counter exists at all, plus a non-vacuity check that the books did move. That
is design v2 §6 item 7 — **nothing relaxes to ambient** — as an arithmetic
identity: if a fifth solid channel ever appears, leg (a) goes red.

`test_eos_p2_sealed_room_energy.py`'s exposed-bulkhead scenario is re-anchored
the same way. It asserted that the space-facing channels DOMINATE the drain;
this module drives the direct binding, which runs no sweep, so a vacuum-facing
hull tile now loses **exactly nothing** at that seam — asserted as `e_space == 0`,
with a non-vacuity floor on the conduction truncation. That is §6 item 7 pinned
at the one place it used to be false.

### 7.4 The gate at step 7

**`2545 passed, 21 failed`** — the 14 golden-bound plus §6.3's 7 runaway tests.
CUDA lockstep tol-0 on every plane and every surviving counter.

---

## 8. Step 8 — `k_leak` live at 0.10, and §6's properties pinned

### 8.1 `k_leak` needed no edit — step 6 made it live

`[physics.radiation] k_leak = 0.10` has been in `config.toml` since T1, bound in
`PhysicsRunner` (`self.k_leak_q`) and passed into the sweep on both the normal
and the resident tick. What T1 could not do was make it *matter*: the sweep was
in shadow, so the leak only moved planes nothing consumed. **The flip is what
put it live.** D6 §7.5 records it as `already live — T1's — KEEP`, and that is
what step 8 confirms rather than changes.

Measured on the canonical sealed box with its walls seeded to 600 game, 60
ticks:

| | `\|Σ rad_amb\|` | solid books, Δ over the run |
|---|---|---|
| `k_leak = 0.10` (shipped) | **340 568** | −1 378 009 219 072 |
| `k_leak = 0` | 226 512 | −936 051 212 288 |

so the out-of-plane path adds **50 %** to the room's radiative export, and the
room ends measurably colder for it. At `k_leak = 0` the box is not *fully*
adiabatic — this harness scenario has vacuum beyond its hull, and a cell facing
vacuum exports to `rad_amb` regardless — which is why the gate is stated as a
ratio against the no-leak control rather than as "0 vs non-zero".

### 8.2 The §6 properties, and where each one lives

`tests/test_thermal_v2_properties.py` is new and carries four of them; the rest
had homes already and the module's docstring names each.

| item | property | home |
|---|---|---|
| 1 | a flammable thermal solid has a loss channel | `test_optics_ingress.py` (step 2) |
| **5b** | **a face loses exactly what the gas gains** | **new, §8.3** |
| **5c** | **a gas cell never ends hotter than the solid heating it** | **new** |
| **6** | **a sealed room is no longer adiabatic** | **new, §8.1** |
| **7** | **nothing relaxes to ambient** | **new** |
| 8 | conduction is physical, no stability anchor | `test_temperature_conduction.py` (step 3) |
| 9 | the books close with the thermostat term gone | `test_thermostat_books.py` (step 7) |
| 10 | a marine burns, a zombie takes 4× | `test_unit_heat_damage.py` (band re-derived at step 6) |
| **11** | **contents burn, structure resists** | **NOT PINNED — see §8.4** |

### 8.3 Item 5b, as an exact cross-ledger identity

The two ledgers are denominated differently — the solid side books HEAT COUNTS
(`ΔT · thermal_mass`), the gas side books `N · T_abs` — and `c_v` is the one
exchange rate between them. So Erik's assertion becomes:

    e_solid_cond_sum + c_v · e_gas_cond_sum  ==  e_cond_trunc_sum + e_cond_cap_sum

Measured on the sealed box, gas seeded +300 game, 60 ticks:

| term | value |
|---|---|
| `e_solid_cond_sum` (the walls' gain, counts) | **+28 185 722 880** |
| `e_gas_cond_sum` (the gas's loss, `N·T`) | −4 155 610 637 752 |
| …× `c_v`, i.e. in counts | **−31 935 452 190** |
| sum | **−3 749 729 310** |
| `e_cond_trunc_sum + e_cond_cap_sum` | **−3 742 108 620** |

**Unexplained: 0.2 % of the gas side**, which is the float `c_v` in the test's
own conversion, not the engine's. The gate asserts < 0.5 %, plus that the two
sides are each an order of magnitude larger than what is left over — so it is a
real cancellation, not two small numbers.

### 8.4 Item 11 is deliberately NOT pinned

*"A furniture tile beside a plateau fire ignites within the bench window; a wood
wall does not."* The property is real and wanted. It cannot be stated honestly
today: after the flip a burning tile ratchets to the `T_MAX_PHYS` rail (§6.3),
so "does this ignite within the window" answers yes for anything hot enough to
be in the scene, and a version that passed would pin the runaway rather than the
property. It is owed the moment Erik rules on §6.3.

### 8.5 Validated by breaking it

| control | what was broken | result |
|---|---|---|
| A | the gas currency conversion reverted to booking `de` unconverted — the exact pre-T5a bug | RED — **`test_5b` alone**, by 4.2e+12 counts |
| B | a `cool_shift` remnant re-introduced (`T -= T >> 5` on every thermal solid, after Pass 2) | RED — **`test_7` by name**, plus `test_thermostat_books`'s closure identity and both `test_eos_p2_sealed_room_energy` scenarios |

Control B is the one worth reading twice: an unbooked relaxation added anywhere
in the pass is now caught by **four** independent gates, one of which (the
closure identity) does not mention cooling at all. That is what "the term is
removed, not zeroed" buys.

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
