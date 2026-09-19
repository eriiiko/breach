# T2 — the currency audit: is one gas heat count one solid heat count?

> **Written incrementally** (design v2 §5, T2 row; the arc's standing rule that a
> kill must cost no resume). Branch `12-t2-currency-audit`, cut from `fire-12`
> at `6eab0f8`.
>
> **This patch changes no behaviour.** It is a measurement and an argument; the
> number it recommends is applied at **T5**.
>
> **Depends on**: `thermal_model_v2_design_2026-09-19.md` §3.2, R13, §4 ledger
> item 4, §5 T2 row, §6 item 5; `report_p2b.md` §1–§4 (the solid-side pin);
> arc #54's gas-energy seam and closure identity (CLAUDE.md, three rows).

---

## 0. Status

- [x] §1 What the currency is, on both sides, read off the code
- [x] §2 D1 — what `c_v` must be (the arithmetic)
- [x] §3 D1 — the engine check (the measurement, not the paper)
- [ ] §4 D2 — where `c_v` belongs: the representation or the seam
- [ ] §5 D2 — the #54 closure identity under the recommendation
- [ ] §6 D3 — the blast radius, by file and line
- [ ] §7 `n_floor_heat` under a rescaled `c_v`
- [ ] §8 Open questions for Erik
- [ ] §9 What did not hold


---

## 1. The currency, on both sides, read off the code

P2b §1 pinned the SOLID side. This section does the same read on the gas side,
so the two can be compared in one unit instead of by assertion.

### 1.1 The raw scalings, named

Everything below turns on integer scalings that are easy to conflate. Read off
the code, not assumed:

| quantity | raw unit | where |
|---|---|---|
| `temperature[i]` | `T_raw = T_game · 2^16` (Q16.16 game degrees = kelvin above 293 K) | `[physics.temperature_scale]`, P2b §1 |
| `heat[i]` | `E_raw` — a **heat count**, no extra shift: the solid branch is `temperature += heat >> heat_inv_shift` | `temperature_solver.cpp:337-340` |
| `cap_q` (`cell_capacity_q`) | `C · 2^16`, so `E_raw = (cap_q · T_raw) >> 16` | `temperature_solver.h:209-249` |
| `de` (a conduction face sum) | `E_raw · 2^16` — `face_energy_q` computes `g · cmin` with `g` a `T_raw` and `cmin` a `cap_q` | `temperature_solver.h:257-268` |
| `gas_energy[i]` | `N_raw · T_abs_raw` — the exact unshifted int64 product | `gamemap.py:943-967`, `gas_energy.h` |

`cell_capacity_q` gives `C = thermal_mass` for a solid and `C = N · c_v` for
gas, which is the unified FORM the design names: conduction can take
`min(cap_i, cap_j)` across a solid–gas face because both are `C · 2^16`.

### 1.2 The exact bridge between the two ledgers

Both ledgers are `Σ capacity × temperature`, but with **different capacities**:

    solid books :  solid_energy_books_sum  = Σ cap_real · T_raw = 2^16 · (C_solid · T_raw) = 2^16 · E_raw
    gas books   :  Σ_accountable gas_energy = Σ N_raw · T_abs_raw = 2^32 · (N · T_abs)

For a gas cell the heat-count energy is `E_raw = C_gas · T_raw = N·c_v·T_raw`,
so

> **`gas_energy` raw = (2^16 / c_v) · E_raw**, i.e. **one raw count of
> `gas_energy` is `c_v / 65536` heat counts.**

At `c_v = 1` — and *only* at `c_v = 1` — the two ledgers are numerically the
same unit. That coincidence is not incidental: it is **load-bearing** in the
engine, at exactly one site (§4.2).

### 1.3 What `gas_energy` actually is, physically

`gas_energy` is **not joules and was never meant to be**. It is the ideal-gas
`N·T` product — `p·V/R` — which is why the EOS reads it as a pressure
(`p* = C·E`). Internal energy is `U = N · C_v,molar · T`, so `gas_energy =
U / c_v_phys`, and the engine already says so in its own derivation.
`eos_solver.cpp:727-735`:

    k_work = (γ−1)·T_AMB_K,  from  d(E_books)/dt = −(K/c_v_phys)·p_code·div u,
    with   c_v_phys = K/((γ−1)·T_AMB)

`K = c_amb²/γ` cancels, and the residue `c_v_phys` is the gas's **physical
volumetric heat capacity** `p/((γ−1)T) = ρc_v`. The drag channel is the same
story: `k_ke = γ(γ−1)·T_AMB_K/(2·c_max²)` is `1/c_v` for *specific* kinetic
energy, derived, not dialled — arc #54 D5 retired `k_drag_heat_frac = 0.0014`
(a hand-rolled stand-in for `1/c_v,air = 1/718`) for exactly this reason.

> **The EOS's own two energy sources are ALREADY denominated at the real
> physical `c_v`, folded out of γ and `T_AMB_K`.** The only place a
> *convention* `c_v` survives is the exchange rate between heat counts and
> `gas_energy`. `[physics.thermal] c_v` is, structurally, a **seam constant
> already**. §4 argues that is where it should stay.

---

## 2. D1 — what `c_v` must be

### 2.1 The arithmetic

`c_v` is the gas cell's capacity in `thermal_mass` units at `N = 1`. R13 pins
one `thermal_mass` unit at `0.9 MJ/(m³·K) ÷ 8 = 0.1125 MJ/(m³·K)` (furniture is
`thermal_mass = 8`). So

    c_v  =  (ρc_v of ambient air)  /  0.1125 MJ/(m³·K)

**`V_tile` cancels** — the solid unit and the gas capacity are both `ρc·V_tile`
— so unlike `rad_scale`, `c_v` is **free of P2b §3.1/§3.2's two ambiguous
factors** and of the level's tile size (0.333 m vs 1.0 m, P2b §13). It depends
on exactly four numbers: `P_amb`, `T_amb`, `γ`, and the R13 pin.

Three independent routes to `ρc_v` of air, computed rather than quoted:

| route | inputs | `ρc_v` |
|---|---|---|
| **the engine's own EOS constants** | `p/((γ−1)T)`, `γ = 1.4`, `T_AMB_K = 293`, ambient `= 1 atm` (engine/04: "1.0 = standard atm") | **864.548 J/(m³·K)** |
| literature | `ρ = 1.2041 kg/m³`, `c_v = 718 J/(kg·K)`, 20 °C, 1 atm | 864.544 |
| molar | `n = P/RT = 41.592 mol/m³`, `C_v,molar = 5R/2 = 20.786` | 864.548 |

They agree to five significant figures. The first is the one to keep: `c_v` is
then *derived from constants the code already holds*, not imported.

`ρc_v = P/((γ−1)T)` is **independent of molar mass**, so `c_v` is the same for
O₂, N₂ and any mixture of them — the multi-species `n_bulk` needs no per-gas
capacity. (It would change for a monatomic or triatomic species; that is a real
extension point, not a gap today.)

### 2.2 The number

    ρc_unit (R13)  = 0.9e6 / 8  = 112 500 J/(m³·K)
    ρc_v (air)                  =     864.548 J/(m³·K)

    c_v   = 864.548 / 112 500   = 0.00768487
    1/c_v                       = 130.126

> ### `c_v` must be **`0.0076849`** (from `1.0`) — the placeholder makes an ambient air tile **130.13× too heavy thermally**.

Sanity in the units R13 pins (`J_per_count = 0.47588 J` at 0.9 — P2b §2 rerun
with ρc = 0.9): an ambient air tile's real heat capacity is
`864.548 × 0.277223 m³ = 239.67 J/K`, i.e. `239.67 / 0.47588 = 503.64` heat
counts per kelvin. And `c_v × 65536 = 503.64` — necessarily the same number,
since `cap_q` at `N = 1` *is* `quantize(c_v)`.

### 2.3 The design's `0.0098 / 101.56×` is stale after R13 — a unit, not an error

Design v2 §3.2 says air is `0.0098`, `101.56×`. Reproduced here: at the
**provisional 0.7** pin, `864.548 / (0.7e6/8) = 0.009881`, `101.21×`. Same
derivation; the residue is which air numbers were used.

R13 then moved the pin 0.7 → 0.9 and §3.2 was not restated. **In the currency
R13 actually pins, the figures are `0.0076849` and `130.13×`.** Both are right
in their own unit. T5 must apply the **0.9-pin** value; design v2 §3.2 wants
correcting to match R13 (§8, open question 1).

### 2.4 Quantization

`c_v` enters as `quantize(c_v)` (Q16.16) in two places — the capacity build
(`temperature_solver.cpp:117-119`) and `make_recip(c_v)` for the deposit divide.

    quantize(0.00768487) = round(503.6356) = 504  ->  0.00769043,  +0.072 %

Comfortably inside the ±6–10 % `thermal_mass` snap R5 already accepts, and
`c_v` is **not** snapped to a power of two (it is a Q16.16 multiplier, not a
shift), so no further quantization is owed. At the `n_floor_heat` floor
`cap_used` becomes `(655 · 504) >> 16 = 5` raw — still positive, so
`cell_capacity_q`'s divide-by-zero guard never binds, but only ~0.4 % accurate
there. See §7.

---

## 3. D1 — the engine check, and what it found

Instrument: `docs/ray_engine_v2_scheme_study_2026-09-13/currency_audit_t2.py`.
It runs the shipped `Simulation` on a sealed 12×12 room (HULL ring, air
interior, a centred 4×4 **STEEL** block at 800 game-K). Steel's conductivity is
45 and **air's is 0.024 — not zero** — so every block face is a live solid↔gas
conduction face. The only thing varied is `temperature.c_v` on the live solver.
Nothing is edited; nothing is applied.

### 3.1 M1 — the energy balance across a solid↔gas face

> **The question in its most direct form: does the gas gain the joules the
> solid loses?**

40 ticks, converted to joules through §1.2's bridge:

| | `c_v = 1` (shipped) | `c_v = 0.0076849` (physical) |
|---|---|---|
| solids LOST | 6 374 688 counts = 3 033 612 J | 59 392 counts = 28 264 J |
| …of which counted `e_cond_trunc_sum` | 35 931 counts (**0.56 %**) | 9 246 counts (**15.57 %**) |
| delivered to the gas by the faces | 6 338 757 counts = 3 016 513 J | 50 146 counts = 23 864 J |
| **gas RECEIVED** | 6 338 757 counts = 3 016 513 J | **385 counts = 183 J** |
| **received / delivered** | **1.00000000** | **0.00768487** |
| arc #54 closure identity | **0 / 40 ticks bad** | **0 / 40 ticks bad** |

> ### The ratio is `c_v`, to eight figures, at both values.
>
> At the shipped `c_v = 1` the two currencies agree **exactly** — a true null
> result, and the reason nobody has noticed. At the physical value the gas
> receives **0.77 %** of the energy the solid gave up: **99.23 % of every joule
> conducted into air is destroyed at the face.**
>
> **And the arc #54 closure identity closes exactly — 0 bad ticks — in both
> cases.** It does not see this. §5 says why, and why that is not a defect in
> the identity.

### 3.2 M2 — the same thing per cell, against the real heat capacity

Design v2 §6 item 5 asks for the per-cell form: *a known energy across a
solid–gas face raises the gas by its real heat capacity's prediction.*

`TemperatureSolver.step`'s pybind signature takes **no `gas_energy`**
(`bindings.cpp:2023-2027`), so the direct binding is always the **pre-#54
T-form law**, `ΔT_i = floordiv(Σ_faces ΔE, cap_i)`. That law is the one whose
prediction we want. One tick on `STEEL@800 | AIR | AIR`, cooling off:

| | `cap_gas` | gas ΔT | solid ΔT |
|---|---|---|---|
| `c_v = 1` | 65 536 raw | **+0.78125000 K** | −0.02441406 K |
| `c_v = 0.0076849` | 504 raw | **+0.78125000 K** | −0.00019836 K |

The T-form gas ΔT is **c_v-independent, bit for bit**, and that is exactly
right: the face moves `ΔE = (g · min(cap)) >> s` with `cap_gas` the minimum, so
`ΔT_gas = ΔE / cap_gas = g >> s` whatever `cap_gas` is. 130× less energy
crosses, into a 130× lighter cell. Meanwhile the **solid**'s ΔT falls 123× —
which is the *physics the correction buys*: a 240 J/K air tile can no longer
cool a 31 kJ/K steel tile as if it were another lump of steel.

The live engine gives `ΔT_gas = ΔE / N_raw` instead (§4.2), i.e. `c_v ×` the
T-form answer — the M1 ratio, seen per cell.

### 3.3 A second, independent finding: the endpoint truncation grows

M1's `e_cond_trunc_sum` row is not decoration. The **solid** endpoint divide
`floordiv(de, cap_solid)` destroys, counted:

    c_v = 1          ->   0.56 % of the conducted energy
    c_v = 0.0076849  ->  15.57 %

because `de` shrinks 130× while `cap_solid` does not. In M2 the steel cell's
one-tick landing is −13 raw counts — coarse, but non-zero. Under **R10**
conduction shifts get much larger (steel `α·Δt/Δx² = 4.4e-6`), and the product
of the two effects can floor a solid's conduction landing to **exactly zero**.
R10 already accepts that conduction is negligible; "negligible" and "identically
zero" are different claims, and the second one should be *chosen*, not
discovered. Flagged for T3/T5 (§8, open question 3), not a blocker for T2.
