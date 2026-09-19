# T3 — the real material table, derived

> **Written incrementally** (design v2 §5, T3 row; the arc's standing rule that
> a kill must cost no resume). Branch `12-t3-material-numbers`, cut from
> `fire-12` at `11af10c`.
>
> **This patch changes no code and no config.** It is a derivation with
> citations; **T5 applies it**. The material rows are Erik's.
>
> **Depends on**: `thermal_model_v2_design_2026-09-19.md` (R5, R7, R10, R12,
> **R13**, §4 ledger items 3, 6, 7a, 7b, 9, 10, 12, §5's T3 row);
> `report_p2b.md` §1–§9 (the currency pin and the row-by-row method);
> `report_t2.md` (the gas side; why `V_tile` cancels for `c_v` but not for
> `rad_scale`).

---

## 0. Status — and the gate

- [x] §1 The unit, restated (R13)
- [x] §2 **D1** — `thermal_mass` per row
- [ ] §3 **D2** — `κ` per row at real rates (R10)
- [ ] §4 **D3** — the solid–gas convection coefficient `h`
- [ ] §5 **D4** — the vacuum mask threshold
- [ ] §6 **D5** — emissivities, ignition temperatures, combustion
- [ ] §7 **D6** — the summary table T5 executes from
- [ ] §8 Open questions for Erik
- [ ] §9 What did not hold
- [ ] §10 The gate

---

## 1. The unit, restated — and the one formula everything else rides

R13 is a **unit definition**, not a measurement, and T3 does not reopen it.

    rho_c pin        = 0.9e6 J/(m3.K)      wood at ~12 % MC            (R13)
    thermal_mass pin = 8                   the furniture row
    ONE UNIT         = 0.9e6 / 8           = 112 500 J/(m3.K)
    V_tile           = 0.333^2 x 2.5       = 0.277223 m3               (P2b §3.2)
    A_face           = 0.333 x 2.5         = 0.8325 m2                 (P2b §3.1)
    J_per_count      = 0.9e6 x V_tile / (8 x 65536)  = 0.475884 J
    c_v              = 864.548 / 112 500   = 0.0076849                 (T2 §2.2)

Two consequences drive every row below.

**(a) `thermal_mass` reads directly as `rho*c` in units of 0.1125 MJ/(m3.K).**
So D1 is one division and one snap per row — exactly P2b §7's method, at the
ruled pin instead of the provisional one.

**(b) The conduction face shift is a PHYSICAL RATE, and this is the formula.**
`conduction::face_energy_q` (`cpp/src/temperature_solver.h:257-269`) moves

    dE = |T_j - T_i| * C_min >> s

and Pass 2 is a **gather**: each cell sums its four faces and divides by its
OWN capacity. For a face between two cells of the same material `C_min = C_i`,
so the pass is the textbook explicit Laplacian `T_i += r * sum_j (T_j - T_i)`
with `r = 2^-s`. Setting that equal to Fourier's law across a face of area
`A_face` at cell spacing `dx`, with `C = rho_c * V_tile` and
`V_tile / A_face = dx`:

> ### `2^-s  =  U * dt / (rho_c_min * dx)`   — THE MAPPING

where `U` is the face's conductance per unit area in W/(m2.K) and `rho_c_min`
belongs to the **smaller-capacity** side (the one `C_min` selects). For a
same-material solid face `U = kappa/dx` and this collapses to R10's own
statement, `2^-s = alpha * dt / dx^2`. For a mixed face `U` is the series
resistance of the two half-cells — which is **exactly what the shipped harmonic
mean already encodes**: `R = (dx/2)/kappa_a + (dx/2)/kappa_b = dx/kappa_hm`.

### 1.1 The mapping, measured on the engine rather than asserted

Both halves were checked against the built CPU extension, not taken on paper.

**The per-tick gap fraction is exactly `2^-s`.** A 1x2 same-material pair
(`thermal_mass = 8`), one live face, cooling disabled, one tick:

| `s` | gap (raw) | measured `dT_cold` | `gap >> s` | equal |
|---|---|---|---|---|
| 8 | 65 536 000 | 256 000 | 256 000 | yes |
| 10 | 65 536 000 | 64 000 | 64 000 | yes |
| 12 | 65 536 000 | 16 000 | 16 000 | yes |
| 16 | 65 536 000 | 1 000 | 1 000 | yes |
| 20 | 65 536 000 | 62 | 62 | yes |
| 24 | 65 536 000 | 3 | 3 | yes |

**And the pass CONVERGES to the analytic heat equation.** A 400-cell rod with a
Dirichlet hot end at `s = 8`, against the semi-infinite slab
`T(x,t) = T0 * erfc(x / (2*sqrt(alpha t)))` with `alpha = 2^-s * dx^2 / dt` —
no fitted parameter, the same `alpha` the mapping predicts:

| ticks | cells above 2 % of T0 | `sqrt(alpha t)/dx` | max relative error |
|---|---|---|---|
| 500 | 5 | 1.40 | 15.4 % |
| 2 000 | 10 | 2.80 | 6.5 % |
| 8 000 | 19 | 5.59 | 1.6 % |
| 32 000 | 37 | 11.18 | **0.22 %** |
| 128 000 | 74 | 22.36 | 0.50 % |

The error falls like the spatial discretisation as the profile spreads over more
cells, then floors at ~0.5 % on the endpoint `floordiv`. **R10's claim is
therefore not an analogy: at shift `s` the engine's conduction pass IS the heat
equation with `alpha = 2^-s dx^2/dt`, to 0.2 %.** Every `kappa` below is
converted through that one identity.

Instruments: `scratchpad/m_conduction.py`, `scratchpad/m_conv.py` — regenerable,
untracked; they build nothing and assert nothing, they only measure.

## 2. D1 — `thermal_mass` per row (ledger 3)

`thermal_mass_derived = rho*c / 112 500`, snapped to the nearest power of two
(R5). Sources in §2.2.

| row | literature `rho*c` MJ/(m3.K) | exact | **snap** | snap error | ships | moves? |
|---|---|---|---|---|---|---|
| `air` | — (gas branch) | — | **0** | — | 0 | **no** |
| `hull` | 3.60 (band 3.40–3.85) | 32.00 | **32** | **0.0 %** | 32 | **no** |
| `steel` | 3.60 (band 3.40–3.85) | 32.00 | **32** | **0.0 %** | 32 | **no** |
| `wood` | 0.90 (band 0.80–0.95) | 8.00 | **8** | **0.0 %** | 8 | **no** |
| `door` | 0.90 | 8.00 | **8** | 0.0 % | 8 | **no** |
| `door_closed` | 0.90 | 8.00 | **8** | 0.0 % | 8 | **no** |
| `furniture` | 0.90 — **the pin row** | 8.00 | **8** | 0.0 % | 8 | **no** |
| `glass` | 1.95 (band 1.88–2.10) | 17.33 | **16** | −7.7 % | 16 | **no** |
| `kindling` (material reading) | 0.90 | 8.00 | **8** | 0.0 % | 8 | **no** |
| `foliage` (material reading) | 0.90 | 8.00 | **8** | 0.0 % | 8 | **no** |

> ### D1's headline: **NOT ONE `thermal_mass` MOVES.**
> Every shipped row already equals its derived value at the R13 pin, and no snap
> is worse than 7.7 % — well inside the ±25 % flag and inside R5's stated 6–10 %
> cost. **The table has been encoding the 0.9 pin all along**, which is the claim
> R13 made; this is its row-by-row confirmation. **T5 has no D1 edit to make.**

Two rows carry an **object-vs-material** reading that the pin makes visible.
Both are P2b §7's finding restated at the ruled pin; **neither is mine to
decide** (§8, questions 1 and 2):

| row | as a MATERIAL | as an OBJECT | object `rho*c` | exact | wants |
|---|---|---|---|---|---|
| `kindling` | wood, 8 | a stick pile is 15–25 % solid wood by bulk volume | 0.13–0.23 | 1.2–2.0 | **2** (−75 %) |
| `foliage` | wood, 8 | a canopy tile is 2–5 % leaf tissue (`rho*c` 1.5–2.7) | 0.03–0.135 | 0.27–1.20 | **0.5 — NOT EXPRESSIBLE** |

`foliage`'s object reading lands at `thermal_mass = 0.5`, and **the table cannot
express it**: the column must be a power of two with `heat_inv_shift >= 0`, and
`0` is taken (it means "gas branch"). The floor is **1** — still ~2x too heavy
for a canopy, but **8x lighter than the shipped 8**. This is the identical wall
P2b §9 hit with furniture's surface-layer reading (0.25), from the other end of
the table: **the power-of-two column has ~8x of dynamic range below the pin and
it is already spent.**

### 2.1 What a snap costs, structurally

R5 accepts the quantisation on `thermal_mass`. Worth stating exactly, because
§3 uses the same bound and there it is **not** a policy choice: snapping a
quantity to the nearest power of two costs at worst **+41.4 %** (rounding down,
exact value at `x.5` in log space) or **−29.3 %** (rounding up). D1's worst row
is glass at −7.7 %, i.e. the table sits inside a fifth of the available error.

### 2.2 Sources

| quantity | value | source |
|---|---|---|
| softwood `rho*c`, 12 % MC | `rho` ~510 kg/m3, `c_p` ~1.7 kJ/(kg.K) -> **0.867**; R13 pins 0.90 | *Wood Handbook* FPL-GTR-190 ch. 4 (`c_p,dry = 0.1031 + 0.003867 T` kJ/(kg.K), plus the MC correction) |
| mild / structural steel `rho*c` | AISI 1010 at 300 K: 7832 x 434 = **3.399**; structural steel 7850 x 460–490 = **3.61–3.85** | Incropera & DeWitt, *Fundamentals of Heat and Mass Transfer*, Table A.1 |
| soda-lime plate glass `rho*c` | 2500 x 750 = **1.875**; `c_p` 750–840 -> 1.88–2.10 | Incropera & DeWitt, Table A.3 |
| leaf tissue `rho*c` | 1.5–2.7 (mostly water: `c` 2.8–3.6 kJ/(kg.K), `rho` 500–900) | Jones, *Plants and Microclimate* 3rd ed. ch. 9; Monteith & Unsworth, *Principles of Environmental Physics* 4th ed. ch. 13 |
| air `rho*c_v` | **864.548** J/(m3.K) from `p/((gamma-1)T)` — the engine's own EOS constants | T2 §2.1 (three independent routes agree to 5 s.f.) |

## 3. D2 — `κ` per row at real rates (ledger 6, R10)

*(pending)*

## 4. D3 — the solid–gas convection coefficient (ledger 7a)

*(pending)*

## 5. D4 — the vacuum mask threshold (ledger 7b)

*(pending)*

## 6. D5 — emissivities, ignition temperatures, combustion (ledger 9, 10, 12)

*(pending)*

## 7. D6 — the summary table T5 executes from

*(pending)*

## 8. Open questions for Erik

*(pending)*

## 9. What did not hold

*(pending)*

## 10. The gate

*(pending)*
