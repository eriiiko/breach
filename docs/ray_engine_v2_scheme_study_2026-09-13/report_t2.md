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
- [x] §4 D2 — where `c_v` belongs: the representation or the seam
- [x] §5 D2 — the #54 closure identity under the recommendation
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

---

## 4. D2 — where `c_v` belongs

### 4.1 Recommendation

> ### `c_v` belongs at the **conversion seam**. `gas_energy` stays the exact integer product `N_raw · T_abs_raw` and is not redefined.
>
> It is *already* a seam constant everywhere but **one site**, and that one site
> is a shortcut taken because `c_v == 1` made it invisible. Fixing it is a
> ~6-line change on each of the two backends, and it can be landed **bit-identical
> at `c_v = 1`** — i.e. proven safe by the goldens *before* the dial moves.

### 4.2 The one site that bakes `c_v = 1` in

`temperature_solver.cpp:531-539`, the accountable-gas branch of Pass 2:

```cpp
if (e_on && acct(i)) {
    int64_t de_books = de;                                  // <-- HERE
    if (cap_real_[i] != cap_i && cap_i > 0) {
        de_books = fixedpoint::floordiv_q(de * cap_real_[i], cap_i);
        e_cond_cap_sum += de - de_books;
    }
    de_gas_[i] = de_books;
    e_gas_cond_sum += de_books;
```

and its own comment says exactly why, at `temperature_solver.cpp:507-510`:

> *"an accountable gas cell's four-face sum IS its energy change — `de` is
> already in the books' currency (a face quantum is `|ΔT|·C` with `C = N·c_v`,
> and the books' capacity IS N)"*

The books' capacity is `N`; the face quantum's capacity is `N·c_v`. Those are
the same thing **only at `c_v = 1`**, which is what §1.2 states and §3.1
measures: `de = c_v · Δgas_energy_correct`, so booking `de` directly deposits
`c_v ×` the energy the face moved. The CUDA twin is identical
(`cuda_temperature.cu:375-381`).

Two consequences of the *shortcut*, not of the dial:

1. `Δ gas_energy` from conduction is `c_v ×` too small — the 99.23 % loss of §3.1.
2. **Turning the dial alone does nothing to the conduction path.** Its gas ΔT is
   `de/N_raw`, which has no `c_v` in it. Changing `c_v` changes `de` (through
   `cmin`) and nothing else, so the gas cell's response stays pinned to the
   `c_v = 1` answer. *A T5 that only edits `config.toml` ships a half-corrected
   engine that is worse than either endpoint* — the deposit path at the real
   capacity, the conduction path still at the convention.

### 4.3 The sites that are already correct — verified, not assumed

Every other crossing already divides by `c_v` and books `N·ΔT`:

| channel | where | form | verdict |
|---|---|---|---|
| Pass-1 gas radiation/heat deposit | `temperature_solver.cpp:400-415` | `dT = E_abs/(max(N,floor)·c_v)`, books `N·dT` | **correct at any `c_v`** — §3's M3 measures the `1/c_v` scaling exactly |
| combustion's aggregate gas deposit | `combustion.cpp:1044-1095` | same form, same `recip_cv` | **correct at any `c_v`** |
| EOS compression work | `eos_solver.cpp:727-735` | `k_work = (γ−1)·T_AMB_K`, i.e. `1/c_v_phys` **derived** | correct, and already *physical* |
| EOS drag heat | `eos_solver.cpp:496-512` | `k_ke = γ(γ−1)T_AMB/(2c_max²)`, **derived** | correct, already physical |
| EOS face flux, transport, recovery | `eos_solver.cpp` steps 6–7 | pure `N·T` | **scale-free** — `c_v` cancels |
| the seam (`move`/`deposit`/`mint`/`retire`/`born_at_ambient`) | `gas_energy.h`, `gamemap.py:1066-1160` | pure `N·T` | **scale-free** |
| `refresh_gas_energy` / `reseed_gas_energy` | `gamemap.py:943-967, 1189` | `N_raw·T_abs_raw` | **unchanged** |
| FieldEdit's four gas-energy paths | `field_edit.py:655-700` | `N·ΔT` / `ΔN·T` / raw | **unchanged** |
| water evacuation export | `physics_engine` `e_water_evac_export_sum` | bulk shares, `N·T` | **unchanged** |
| Pass 3 thermostat | `temperature_solver.cpp:672-688` | `if (!ts[i]) continue;` — **solids only** | gas never reaches it |

So the seam option touches **one expression on each backend**. That is the
whole argument.

### 4.4 Why NOT put `c_v` in the field

Redefining `gas_energy := N·c_v·T_abs` was the other candidate. It is wrong
for five independent reasons, any one of which is sufficient:

1. **`gas_energy` is not an energy — it is the EOS's `N·T`.** It feeds the
   pressure (`p* = C·E`). Multiplying it by `c_v` forces a compensating `1/c_v`
   back out inside the pressure law, and `k_work` / `k_ke` — which are *already*
   at the physical `c_v` — would each need a second compensating factor. Two
   wrongs arranged to cancel.
2. **Exactness dies.** `N_raw·T_abs_raw` is an exact unshifted int64 product;
   that is the arc's whole foundation. `c_v` is a Q16.16 multiplier, not a power
   of two, so `N·c_v·T_abs` needs a shift-and-round **per write**. Every
   `parcel`, `minted`, `deposit` and rail write-back would round, and
   `Σ_region ΔE ≡ 0` — exact today — would hold only to a per-face residual.
3. **The mirror stops being a clean divide.** `mirror_q` is
   `floordiv(E, N) − t_amb`; it would become `floordiv(E, N·c_v)`, putting a
   second quantized quantity inside the once-per-tick recovery's rails, and the
   one sanctioned write-back (`E = N·(T+t_amb)`, `gas_energy.h:104`) would gain
   a rounding step in the exact place the arc forbids one.
4. **`refresh_gas_energy` / `reseed_gas_energy` would have to change**, and both
   are LEVEL-LOAD initialisers whose whole contract is "reproduce the stored
   truth exactly from `(N, T)`".
5. **CUDA cost.** The seam option is one expression in `cuda_temperature.cu`;
   the field option touches every `.cu` that reads or writes `gas_energy`
   (`cuda_eos_step`, `cuda_eos_resident`, `cuda_bulk_transport`,
   `cuda_combustion`, `cuda_temperature`) plus their lockstep checks.

The one thing the field option would buy — a `gas_energy` denominated in real
joules, so P-G5's total ledger is a genuine sum — is bought more cheaply in §6
item 11 by weighting the gas half where the sum is taken.

### 4.5 The exact form to write, and why T5 can land it before the dial moves

The shortcut and the capacity-floor shrink collapse into **one** correct
expression. Today's floor shrink divides by *capacity*; the floor is on **N**,
so write it on N:

```cpp
// `de` is a face sum in HEAT COUNTS (x2^16); the books are N*T. One
// conversion, then the floor's shrink, in the quantity the floor is on.
const int64_t de_full  = <de / c_v, through the load-time recip_cv>;
const int64_t nb       = n_books(i);                       // real N, unfloored
const int64_t nu       = (nb < (int64_t)n_floor_q) ? (int64_t)n_floor_q : nb;
const int64_t de_books = fixedpoint::floordiv_q(de_full * nb, nu);
e_cond_cap_sum += de_full - de_books;
de_gas_[i]      = de_books;
e_gas_cond_sum += de_books;
```

> **At `c_v == 1` this is bit-identical to the shipped code — verified over
> 300 000 randomized `(de, N, n_floor)` triples spanning both branches and the
> `N == 0` corner: 0 mismatches in `de_books` AND 0 in `e_cond_cap_sum`.**

That matters for the patch plan: **T5 can land the structural fix and the dial
as two commits**, the first gated by "goldens unmoved" (it is a pure refactor),
the second carrying the arc's one re-baseline. A bug in the rewrite cannot then
hide inside the retune.

Two things T5 must write down rather than inherit:

- **the int64 bound.** Today's comment justifies `de * cap_real` by "the floor
  binds". The new product is `de_full * nb`; with `|de| ≤ 4·2^31·cmin` and
  `cmin = cap_gas`, at the physical `c_v` `cap_gas ≈ 504·N`, so `|de| ≲ 2^42`
  and `|de_full·nb| ≲ 2^60` — inside int64, but *by argument*, and the argument
  has to be in the file. Use the 128-bit kit if the bound is uncomfortable.
- **the conversion's own truncation.** `de/c_v` floors; at `c_v < 1` that
  discards under one raw count per cell-tick. R3 says every residual is counted
  — give it `e_cond_trunc_sum` or a named channel of its own. It does **not**
  threaten the closure identity (§5), only the arc's honesty rule.

---

## 5. D2 — the arc #54 closure identity under the recommendation

### 5.1 It survives — and it survives the *broken* version too

**Measured** (§3.1, `currency_audit_t2.py`): at both `c_v = 1` and
`c_v = 0.0076849`, over 40 ticks of a live `Simulation`,

    d(Sum_accountable gas_energy) == EOS + thermal-gas + combustion + seams + water

closes **0 / 40 ticks bad, worst |residual| 0**, and so does the P-G5 **total**
ledger (gas books + solid books), 0 / 40 bad.

The reason is structural and worth stating plainly, because it decides how much
weight the identity can carry as a gate:

> Every gas-side counter books **what the seam actually deposited**, not a
> re-derivation of what it should have deposited. `e_gas_cond_sum += de_books`
> is, by construction, the number that was added to `gas_energy`. So the
> identity closes for **any** `c_v` — including one that destroys 99 % of the
> energy crossing every solid–gas face.

This is design v2 §6 item 3's lesson in a second place: *a conservation identity
is structural, so it holds just as exactly for an engine doing the wrong thing.*
It is not a defect in the identity — the identity's job is "nothing changes
`gas_energy` except a named channel", and it does that job perfectly. It simply
cannot answer the currency question, and **must not be cited as evidence that
the currency is sound.**

### 5.2 Which of the closure groups change

None of the four gas groups is added to or removed from. Within them:

| group | change |
|---|---|
| EOS (7 terms) | **none** — all `N·T`, `c_v` cancels or is already physical |
| thermal solver, gas (`e_gas_deposit_sum`, `e_gas_cond_sum`, `e_gas_rail_sum`) | `e_gas_cond_sum`'s **value** changes (it is now the true `N·ΔT`); no term added, no term removed |
| combustion (4 terms) | **none** |
| Python seams (`gas_energy_seam_net()`) | **none** — still exact in int64 |
| P-G5 solid side (3 terms + snapshot) | **none** in form; `e_solid_cond_sum` changes value because `cmin` shrinks |

`gas_energy_seam_net()` still closes in int64: it never sees `c_v`.
`refresh_gas_energy` / `reseed_gas_energy` are **untouched**.
**"Gas temperature is a mirror" still holds exactly** — the mirror is still
`floordiv(E, N) − t_amb`, refreshed by the same seam, and `c_v` appears nowhere
in the recovery or in `mirror_q`.

### 5.3 What it costs the CUDA twin

One kernel, one signature. `cuda_temperature.cu:375-381` carries the identical
shortcut, and `temp_conduct` (`:323-334`) does **not** currently receive
`recip_cv`, `n_floor_q` or the N source — the CPU's Pass 2 has all three in
scope already (`:116-118`, the `n_books` lambda). So T5 threads **three extra
arguments** into `temp_conduct` and writes the same expression. `temp_apply_gas_cond`
(`:400-412`) is unchanged — it deposits whatever was parked.

Nothing else on the device changes: `gas_energy.h` is shared `FP_HD` code and
is untouched, and the device capacity build (`:126-141`, `:690-700`) already
takes `c_v_q` and just carries the new value.

Cost: one `cuda_conduction_check` re-run at tol 0, plus
`tests/cuda_conduction_check.py`'s own `DIALS["c_v"] = 1.0` (`:57`) to move to
the shipped value so the gate exercises what ships. **No CUDA hardware on this
machine — the twin is written and reviewed here, gated on Erik's CUDA box.**
