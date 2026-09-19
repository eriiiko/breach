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
- [x] §3 **D2** — `κ` per row at real rates (R10)
- [x] §4 **D3** — the solid–gas convection coefficient `h`
- [x] §5 **D4** — the vacuum mask threshold
- [x] §6 **D5** — emissivities, ignition temperatures, combustion
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

R10 removed the CFL stability anchor: `SHIFT_AT_REF` / `KAPPA_REF` stop setting
the absolute rate, and §1's mapping sets it instead. **For a same-material face
that is exactly `2^-s = alpha·dt/dx²`**, R10's own words.

### 3.1 `κ` per row

| row | **derived `κ`** W/(m·K) | band | ships | source |
|---|---|---|---|---|
| `air` | **0.0257** | 0.0223 (250 K) – 0.0263 (300 K) | 0.024 | Incropera & DeWitt Table A.4 |
| `hull` | **50.0** | 45–64 | 50.0 | Incropera Table A.1: AISI 1010 = 63.9; carbon–silicon steel = 51.9; ship-plate mild/HSLA ~50 |
| `steel` | **50.0** | 45–64 | 45.0 | ibid. |
| `wood` | **0.12** | 0.10–0.16 (transverse, 12 % MC) | 0.15 | *Wood Handbook* FPL-GTR-190 ch. 4 (`k = G(B + C·M) + A`, G = 0.51) |
| `door` / `door_closed` | **0.12** | 0.10–0.16 | 0.30 | as wood |
| `glass` | **1.40** | 1.0–1.4 | 1.0 | Incropera Table A.3, plate glass at 300 K |
| `furniture` | **0.12** as a material | 0.10–0.16 | **0.0** | Erik's fixture choice (`config.toml`'s own note) — §8 question 3 |
| `kindling` | **0.12** as a material | 0.10–0.16 | **0.0** | ibid. |
| `foliage` | **0.12** as a material | 0.10–0.16 | **0.0** | ibid. |

Only `door`/`door_closed` (0.30 → 0.12) and the three `0.0` rows are in
question; `hull`, `wood`, `glass` and `air` are already inside their bands.

### 3.2 The shift each row wants

`2^-s = alpha·dt/dx²` at `dx = 0.333 m`, `dt = 1/24 s`, `rho*c` from D1.

| face | `alpha` m²/s | per-tick gap fraction | exact `s` | **snap** | quant. err | ships | Δ |
|---|---|---|---|---|---|---|---|
| `hull`\|`hull`, `steel`\|`steel`, `hull`\|`steel` | 1.389e−05 | 5.219e−06 | 17.55 | **18** | **−26.9 %** | 2 | **+16** |
| `glass`\|`glass` | 7.179e−07 | 2.698e−07 | 21.82 | **22** | −11.6 % | 6 | **+16** |
| `wood`\|`wood` (and every cellulosic pair) | 1.333e−07 | 5.010e−08 | 24.25 | **24** | +19.0 % | 8 | **+16** |
| `hull`\|`wood`, `steel`\|`wood`, `*`\|`door_closed` | — (harmonic mean 0.719 W/(m²·K)) | 9.996e−08 | 23.25 | **23** | +19.3 % | 6–7 | **+16/+17** |
| `hull`\|`glass`, `steel`\|`glass` | — (8.179) | 5.248e−07 | 20.86 | **21** | −9.1 % | 5 | **+16** |
| `wood`\|`glass`, `door`\|`glass` | — (0.664) | 9.229e−08 | 23.37 | **23** | +29.2 % | 7–8 | **+15/+16** |
| `air`\|`air` | 2.973e−05 (`κ/rho*c_v`) | 1.117e−05 | 16.45 | **16** | +36.6 % | 11 | **+5** |

> **Every solid–solid face moves by +16 or +17 shift steps — a factor of
> 65 000–130 000 slower.** That is R10's accepted consequence stated as a
> number, and it is the same 2^16 the shipped table's log-bucket anchor was
> hiding: `KAPPA_REF = 50` with `SHIFT_AT_REF = 2` is a rate anchor with no
> capacity in it at all, so it could not have been right by construction.

**The mixed-face rule is unchanged.** The harmonic mean already *is* the series
resistance of the two half-cells (§1), so T5 keeps that formula and only changes
what it is divided by: replace the constant `KAPPA_REF` with the per-pair
quantity `rho_c_min(a,b)·dx²/dt`, i.e.

    face_shift[a][b] = clamp(SHIFT_MIN,
                             round(log2( rho_c_min(a,b) * dx^2
                                         / (kappa_hm(a,b) * dt) )),
                             NO_FACE)

`rho_c_min` is the **smaller-capacity** side (`min(thermal_mass)·112500` for
solids, `rho*c_v` at N = 1 for gas — see §4 for why the gas side is different
in kind). Quantisation is structural here, not a policy: a shift IS a power of
two, so the worst case is **+41.4 % / −29.3 %** (§2.1) and three faces land
outside ±25 % with no alternative available.

### 3.3 The analytic slab — and the brief's two numbers reconciled

State the convention, because the two published sanity numbers use different
ones and differ by exactly 4.

| convention | meaning |
|---|---|
| `tau = dx²/alpha` | the diffusion time: `sqrt(alpha·t) = dx` |
| `tau4 = dx²/(4·alpha)` | the penetration-depth convention `delta = sqrt(4·alpha·t) = dx` |

At **T3's derived inputs** (`κ` and `rho*c` from §3.1 / D1):

| row | `alpha` | `dx²/alpha` | `dx²/4alpha` | **engine `2^s·dt`** |
|---|---|---|---|---|
| `steel` / `hull` | 1.389e−05 | **2.2 h** | 0.6 h | **3.0 h** |
| `glass` | 7.179e−07 | 42.9 h | 10.7 h | 48.5 h |
| `wood` / `door` / cellulosic | 1.333e−07 | **231 h** | 57.8 h | 194 h |

At **R10's own inputs** (`κ = 45`, `rho*c = 3.8` for steel; `κ = 0.15`,
`rho*c = 0.7` for wood — the *pre-R13* pin):

| row | `alpha` | `dx²/alpha` | `dx²/4alpha` | R10's quoted rate |
|---|---|---|---|---|
| `steel` | 1.184e−05 | **2.60 h** | 0.65 h | 4.450e−06 ✓ (R10 says 4.4e−06) |
| `wood` | 2.143e−07 | 143.8 h | **35.94 h** | 8.052e−08 ✓ (R10 says 8.1e−08) |

> **So both published numbers are reproduced, and neither is wrong — they are
> in different conventions.** "steel ~2.6 h" is `dx²/alpha`; "wood ~36 h" is
> `dx²/(4·alpha)`, and both use the superseded 0.7 pin. In **one** convention
> at R13's pin the pair is **steel 2.2 h / wood 231 h** — wood is 105× steel,
> not 14×. Recorded in §9.

### 3.4 What "negligible" turns out to mean: an exact dead-band

T2 §8 question 3 asked whether conduction survives R10 combined with the `c_v`
correction, and said *"'negligible' and 'identically zero' are different claims,
and the second one should be chosen, not discovered."* **Measured, and it is
identically zero over most of the range.**

A same-material face moves `dE = (g·C) >> s` and the endpoint divides by the
same `C`, so the cold cell gains `>= 1` raw count only when `g >= 2^s` raw —
i.e. only when the two tiles differ by more than **`2^s / 65536` game degrees**.
Bisected on the built extension:

| row | `s` | predicted dead-band | **measured** |
|---|---|---|---|
| `steel` / `hull` | 18 | 4.000 K | **4.000 K** |
| `glass` | 22 | 64.000 K | **64.000 K** |
| `wood` / cellulosic | 24 | 256.000 K | **256.000 K** |
| *(wood as shipped)* | *8* | *0.004 K* | *0.004 K* |

> ### **A wood-to-wood conduction face is EXACTLY ZERO below a 256 K gap.**
> Not small — zero. Steel's dead-band is 4 K, glass's 64 K.

And below the dead-band the face is **not** inert: `floordiv` toward −∞ makes
the hot cell lose 1 raw count while the cold cell gains 0.

| wood, `s = 24`, gap | cold gains | hot loses |
|---|---|---|
| 1 K | 0 raw | **−1 raw** |
| 100 K | 0 raw | **−1 raw** |
| 255 K | 0 raw | **−1 raw** |
| 256 K | 1 raw | −1 raw |
| 1000 K | 3 raw | −4 raw |

So conduction under R10 is a **pure, booked sink** below the dead-band, at
1 raw count per cell-tick = 24/65536 K/s = **1.3 K per hour**. That is
0.04 K over a 2-minute crate burn — negligible in play, counted in
`e_cond_trunc_sum`, and *not* a violation of design §6 item 7 (it is a booked
channel). **But it should be Erik's choice, not a discovery** (§8, question 4).

Instrument: `scratchpad/m_deadband.py`.

### 3.5 The conduction table is now tile-size dependent

P2b §13 item 7 flagged `rad_scale`'s dependence on `tile_size_m`. **Under R10
the conduction table joins it**, and more strongly: the solid face goes as
`1/dx²` and the gas face as `1/dx`.

| `tile_size_m` | `wood`\|`wood` | `air`\|`hull` | `air`\|`wood` |
|---|---|---|---|
| 0.333 (playground, fire_tuning) | 24 | 10 | 13 |
| 0.5 (`bench_two_room`) | 25 | 11 | 15 |
| 1.0 (`airlock_demo`) | 27 | 12 | 16 |

The table is built at load, and `tile_size_m` **is** known at load
(`level_loader`), exactly as the water solver already takes its `dx` from the
level. Recorded for T5; not acted on here.

## 4. D3 — the solid–gas convection coefficient (ledger 7a)

**The law is wrong, and that is the whole of this section.** At a solid–gas
interface the resistance is a boundary layer far thinner than a tile, not a
half-tile of still air. `h` is a *measured* quantity with standard correlations,
so replacing `κ_air/dx` with `h` sits squarely inside R7 — it is the opposite of
a fudge, and the shipped form is the invented one.

### 4.1 The derivation

Natural convection on a vertical isothermal plate, **Churchill & Chu (1975)** —
*"Correlating equations for laminar and turbulent free convection from a
vertical plate"*, Int. J. Heat Mass Transfer **18**, 1323–1329; reproduced as
Incropera & DeWitt eqs. **9.26** (all `Ra`) and **9.27** (`Ra_L <= 1e9`):

    Ra_L = g·beta·dT·L^3 / (nu·alpha),   beta = 1/T_f   (ideal gas)

    Nu_L = ( 0.825 + 0.387·Ra^(1/6) / [1 + (0.492/Pr)^(9/16)]^(8/27) )^2

    h    = Nu_L · k / L

Air properties at the film temperature `T_f = (T_s + T_amb)/2` from Incropera &
DeWitt **Table A.4**; `L` = the deck height, 2.5 m (the same 2.5 m `A_rad` and
`k_leak` use). Computed, not quoted (`scratchpad/derive_h.py`):

| `dT` [K] | `T_s` [K] | `Ra_L` | `Nu_L` | **`h` W/(m²·K)** | regime |
|---|---|---|---|---|---|
| 5 | 298 | 7.6e+09 | 231 | 2.40 | turbulent |
| 10 | 303 | 1.5e+10 | 285 | 2.98 | turbulent |
| 50 | 343 | 5.4e+10 | 431 | 4.76 | turbulent |
| 100 | 393 | 7.6e+10 | 481 | **5.68** | turbulent |
| 200 | 493 | 8.2e+10 | 491 | **6.54** | turbulent |
| 300 | 593 | 7.2e+10 | 470 | **6.92** | turbulent |
| 500 | 793 | 4.8e+10 | 414 | **7.20** | turbulent |
| 700 | 993 | 3.3e+10 | 366 | **7.23** | turbulent |
| 1000 | 1293 | 2.0e+10 | 312 | 7.10 | turbulent |
| 1300 | 1593 | 1.2e+10 | 271 | 6.93 | turbulent |

> ### **`h = 6 W/(m²·K)`**, and **one number is honest.**

Three reasons, and the third is decisive:

1. **`h` is nearly flat across the fire-relevant band.** From `dT = 100 K` to
   `dT = 1300 K` it moves only 5.68 → 7.23 → 6.93, i.e. **±12 % about 6.4**. It
   is non-monotone: `Ra ∝ beta·dT/(nu·alpha)` peaks near `dT ≈ 250 K` because
   `nu` and `alpha` grow faster than `dT` above that.
2. **It barely depends on the characteristic length.** Every case is turbulent
   (`Ra > 1e9`), where `Nu ∝ Ra^(1/3) ∝ L`, so `h = Nu·k/L` is `L`-free:
   `h(dT = 700)` is 7.22 / 7.59 / 7.23 / 7.05 at `L =` 0.333 / 1.0 / 2.5 / 5.0 m.
   The one genuinely arbitrary geometric choice in the derivation does not
   matter.
3. **A `dT`-dependent `h` could not change the answer anyway.** The face carries
   `h` through a **bit shift**, and every `h` in `[4.8, 9.6]` rounds to the same
   shift. Only the cold end (`dT < 50 K`, `h < 4.8`) would want one step more.
   Making `h` vary with `dT` would buy **one shift step, in the band where
   nothing is burning** — and would cost the load-time table its constancy.

### 4.2 The face conductance, and the shift it implies

The solid–gas face is two resistances in series — the boundary layer, and the
solid's own half-cell (the same half-cell the harmonic mean already charges for
a solid–solid face). The gas side contributes **no** conduction resistance: a
convecting cell is well-mixed, which is what `h` already describes.

    U = 1 / ( (dx/2)/kappa_solid  +  1/h )

    2^-s = U·dt / (rho*c_v · dx)  =  U / 6909.5        [at N = 1, ambient]

| face | `R_solid` | `R_bl` | **`U` W/(m²·K)** | exact `s` | **snap** | quant | ships |
|---|---|---|---|---|---|---|---|
| `air`\|`hull`, `air`\|`steel` | 0.0033 | 0.167 | **5.88** | 10.20 | **10** | +14.7 % | 10 |
| `air`\|`glass` | 0.1189 | 0.167 | **3.50** | 10.95 | **11** | −3.6 % | 10 |
| `air`\|`wood`, `air`\|`door`, `air`\|`door_closed` | 1.3875 | 0.167 | **0.643** | 13.39 | **13** | +31.1 % | 10 |
| `air`\|`furniture`, `air`\|`kindling`, `air`\|`foliage` | 1.3875 | 0.167 | **0.643** | 13.39 | **13** | +31.1 % | **NO FACE** |

> **The metal face's derived shift is 10 — exactly what ships.** Two entirely
> different routes (a log bucket anchored on `KAPPA_REF = 50`, and `h` divided
> by `rho*c_v·dx/dt`) land on the same integer. That is a coincidence, and it is
> worth naming because it makes the design's "we are 28–139× too weak" **true of
> the law but false of the shipped rate** (§9).

**Measured on the engine** (`scratchpad/m_gasface.py`): a `SOLID(800 game,
thermal_mass 8) | GAS(N = 1)` pair, one tick, one face —

| `s` | `c_v` | gas `dT` | **solid `dT`** | implied `h` |
|---|---|---|---|---|
| 10 | 1.0 (ships) | +0.7813 K | −6400 raw | **878.0** |
| 10 | 0.0076849 | +0.7813 K | **−50 raw** | **6.748** |
| 13 | 0.0076849 | +0.0977 K | −7 raw | 0.843 |

So at shift 10, **once T2's `c_v` correction lands**, the shipped face already
implements `h = 6.75 W/(m²·K)` — 12 % above the derived 6.0, and inside the
shift quantisation. Before that correction it implements **878 W/(m²·K)**.

### 4.3 What it changes about how fast air near a hot wall heats

The gas cell's response is `dT_gas = gap · 2^-s` (T2 §3.2 measured this to be
`c_v`-independent on the T-form path), so the air beside a wall e-folds toward
the wall's temperature in `2^s` ticks:

| face | derived `s` | **gas e-fold** | ships | today's e-fold |
|---|---|---|---|---|
| metal / hull | 10 | **42.7 s** | 10 | 42.7 s — **unchanged** |
| glass | 11 | **85.3 s** | 10 | 42.7 s (2× too fast) |
| wood / door | 13 | **341 s** | 10 | 42.7 s (**8× too fast**) |
| furniture / kindling / foliage | 13 | **341 s** | NO FACE | **never — no channel at all** |

Sanity, independent of the engine: a 0.333 m column of air against 1 m² of wall
holds `rho*c_v·dx = 288 J/(m²·K)`; at `h = 6` its time constant is
`288/6 = 48 s`. The metal row's 42.7 s is that number.

**Two things this buys, and one it does not:**

- A **metal or glass** wall heats its air at the rate it already did — the
  correction there is purely in the **energy the wall loses**, which falls 130×
  with `c_v` (T2 §3.1) and is the change that matters.
- A **wood** wall heats its air **8× slower** than today, because the lumped
  tile's own half-cell resistance (1.39 m²K/W) is **8.3× the boundary layer's**
  (0.167). That is a real consequence of the one-node solid (R6 / P2b §9), not
  of `h`: a real wood wall's *surface* reaches flame temperature while its bulk
  does not, and our tile has only the bulk. Flagged, §8 question 5.
- It does **not** buy plume convection. `h = 6` is *natural* convection. A fire
  plume moving at 1–3 m/s past a surface gives `h = 10–100` (forced), and the
  face law has no velocity term. The engine gets the "hot gas arrives" half
  right (advection carries the plume) and misses the thinner boundary layer
  under it. **`ACCEPTED GAP:` the solid–gas face carries natural convection
  only.** The extension point is named, not built: `h` could read the tamed-wind
  speed the render layer already derives, which would make it a Nusselt
  correlation in `Re` instead of `Ra`.

### 4.4 Where this belongs in the table

`h` is **not** a per-material column. It is one number for the solid–gas
boundary, and it enters the existing machinery as an **effective face
conductance**: §3.2's build already forms `U` per pair, so T5 adds one branch —
*"if exactly one side is a thermal solid, `U = 1/((dx/2)/kappa_solid + 1/h)`
instead of the harmonic mean"* — keyed on the **existing derived
`thermal_solid` column**. No new per-material field, one new global
(`[physics.thermal] h_conv = 6.0`).

Why a branch rather than folding `h` into `air`'s `conductivity`: the `air` row's
`κ` also builds the **air↔air** face, where the physical process really is
molecular conduction (bulk motion between two 0.333 m air cells is advection,
which the EOS already models — using `h` there would double-count it). One
column cannot be both.

## 5. D4 — the vacuum mask threshold (ledger 7b)

### 5.1 The arithmetic, confirmed

Kinetic theory: `kappa = (1/3)·n·<v>·lambda·c_v,molecule` with `lambda ∝ 1/n`,
so `n·lambda` is constant and **`kappa` is density-independent** — until
`lambda` reaches the gap and the gas goes free-molecular (Kennard, *Kinetic
Theory of Gases*, 1938, ch. 8; Jennings 1988 for the modern value of `lambda`).

    lambda(air, 1 atm, 293 K)   = 68 nm
    lambda ∝ 1/p, so lambda = 0.333 m at  p = 101325 · 68e-9 / 0.333
                                        = 0.0207 Pa = 2.04e-07 atm

That reproduces the design's **0.02 Pa / 2×10⁻⁷ atm** exactly.

    n_floor_heat = 0.01  =  1013.2 Pa  =  4.90e+04 x the Knudsen pressure

i.e. **4.69 orders of magnitude above it** (the design says "five"; the exact
figure is 4.7). So yes: the floor invents a conducting medium.

### 5.2 The threshold is SUB-REPRESENTABLE — which hands us the mask

`n_bulk` is Q16.16 on a scale where `1.0 = 1 atm`, so **one raw count is
1/65536 atm = 1.546 Pa**, where `lambda = 4.46 mm` and `Kn = lambda/dx = 0.0134`
— still firmly a continuum conductor.

    the Knudsen threshold is 0.013 of ONE Q16.16 LSB

> ### The physics chooses the mask for us: **`N_raw == 0`.**
> Every density the engine can represent above zero is still in the continuum
> regime and *should* conduct at the full rate. The only free-molecular state
> `n_bulk` can hold is exactly zero. There is no threshold to pick, and picking
> one would be a dial where physics offers none.

### 5.3 What actually happens today — measured, not argued

`SOLID(800 game) | GAS(N)`, one face at shift 10, `c_v` corrected, one tick
(`scratchpad/m_gasface.py`):

| `N` (atm) | `N_raw` | `cap_used` | `cap_real` | gas `dT` | solid `dT` | `e_cond_cap_sum` |
|---|---|---|---|---|---|---|
| 1.0 | 65 536 | 504 | 504 | +51 200 raw | −50 raw | 0 |
| 0.05 | 3 277 | 25 | 25 | +51 200 raw | −3 raw | 0 |
| 0.01 (**at the floor**) | 655 | 5 | 5 | +51 200 raw | −1 raw | 0 |
| 0.002 | 131 | **5** | 1 | +51 200 raw | −1 raw | −204 800 |
| 1/65536 (one LSB) | 1 | **5** | 0 | +51 200 raw | −1 raw | −256 000 |
| **0.0 (hard vacuum)** | **0** | **5** | **0** | **+51 200 raw** | **−1 raw** | **−256 000** |

Two things to read off it:

1. **A hard-vacuum cell heats exactly as fast as an ambient-air cell** — the
   `dT` column is constant at +0.78 K/tick all the way down, because the T-form
   divides by the same floored `cap_used` the face multiplied by. In the live
   engine Pass 0 wipes that back to 0 every tick (and books `e_vac_wipe_sum`),
   so it never accumulates — but it *is* the temperature the sweep, the fold and
   the tile inspector see within the tick.
2. **The energy the wall loses into vacuum is the floor's own fraction.** It is
   `n_floor_heat` of the ambient rate by construction: **44.6 W per face** at a
   800 K excess, 5.6 W at 100 K. Compared with the same tile's radiative loss at
   that temperature (269 kW through the sweep) it is **0.017 %**.

> **Honest magnitude: the vacuum leak is real, is already counted
> (`e_cond_cap_sum` / `e_vac_wipe_sum`), and is energetically negligible.** It
> is worth fixing because it is a *fictional medium*, not because it is a hole —
> and this report should not be read as saying otherwise.

### 5.4 The mask T5 should apply

> ### `no_conduction(i) = !ts[i] && ( is_vacuum[i] || cap_real_[i] == 0 )`
> — every face such a cell owns becomes `NO_FACE`.

Every term is already in `TemperatureSolver::step`'s scope; **no new field, no
new dial, no threshold**:

| term | where it already lives | why it is there |
|---|---|---|
| `!ts[i]` | `temperature_solver.cpp:79` | **load-bearing.** An intact hull tile is `is_vacuum && solid && thermal_solid` (the code says so at Pass 0). It is a wall, not a breach, and must keep conducting to its solid neighbours |
| `is_vacuum[i]` | the `step()` signature; Pass 0 (`:148`) and Pass 3 (`:655`) already key on it | the **structural** statement. A cell whose `temperature` and `gas_energy` Pass 0 zeroes every tick is not a thermal medium at all |
| `cap_real_[i] == 0` | built every tick at `:120-127` | the **dynamic** statement, and §5.2's derived threshold. Catches a decompressed *interior* room, which `is_vacuum` does not mark |

**Verified on live levels** (`scratchpad/m_vac2.py`): on `airlock_demo` (35 open
vacuum cells) and `fire_tuning` (232), `is_vacuum` cells carry `n_bulk == 0`
exactly — min 0, max 0, zero nonzero — at load **and** after 120 ticks. The two
terms agree today; the disjunction is belt-and-braces for the case they stop
agreeing (a room venting through a breach), not a hedge.

### 5.5 What happens at the boundary

- **vacuum \| solid** — no face. The wall's only channels are the sweep's
  in-plane radiation and `k_leak`, which is right: across a vacuum gap,
  radiation is the *only* physics there is.
- **vacuum \| gas** — no face either, and this loses nothing real. Energy
  leaving a room through a breach leaves **with the gas**, which is bulk
  transport and the gas-energy seam (`extract_gas_n`, `e_vac_wipe_sum`), not
  conduction. Killing the conduction face there removes a fiction, not a
  channel.
- **the mask is per CELL, not per face**, so it is symmetric by construction and
  cannot produce a one-sided face (the pass already reads both ends'
  `face_shift` and takes the `max`, so a masked cell is skipped from either
  side).
- **`n_floor_heat` itself does not change.** T2 §7 established the floor is on
  `N` and keeps its job — a cell with *some* gas still needs a non-degenerate
  divisor. The mask removes only the case where there is *no* gas, which is the
  one case the floor was never meant to cover.

## 6. D5 — emissivities, ignition temperatures, combustion (ledger 9, 10, 12)

### 6.1 Emissivity (`heat_atten`) — ledger 9, R12

P2b §8 stated the complication and it still holds: in the sweep `a_i` does **two
jobs at once** — by Kirchhoff it is the cell's *emissivity*, and in the same
multiply it is the cell's *extinction*, i.e. its geometric opacity. For a solid
slab filling the tile they coincide; for a partly-filled tile they do not. This
section gives both numbers wherever they differ and says which the row is
currently encoding.

| row | ships | literature **emissivity** | literature **opacity** | source | proposal |
|---|---|---|---|---|---|
| `wood` | 1.0 | **0.82–0.92** (planed oak, pine, beech) | 1.0 (fills the tile) | Incropera & DeWitt Table A.11 | **0.90** |
| `door` / `door_closed` | 1.0 | 0.82–0.92 bare; **0.90–0.96** painted | 1.0 | ibid. | **0.90** |
| `hull` / `steel` | 1.0 | oxidised mild steel **0.78–0.82**; heavily oxidised 0.88; **painted 0.90–0.96**; *polished stainless 0.17* | 1.0 | ibid. | **0.85** (a painted / oxidised hull) |
| `glass` | 0.3 | **0.90–0.95** at its own temperature | *absorptivity for a 1556 K flame* **~0.4** | Incropera A.11; Siegel, Howell & Mengüç ch. 5; P2b §8's band arithmetic | **no single value is right** — see below |
| `air` | 0.0 | ~0 (N₂/O₂ are homonuclear: no IR-active dipole) | 0 | Siegel & Howell ch. 10 | **0.0, correct** |
| `furniture` | 0.5 | wood, 0.90 | a crate stack, ~0.5 | config's own comment says "partial: smoke/air drift past crates" | **leave 0.5 — it is an OPACITY** |
| `kindling` | 0.5 | wood, 0.90 | sticks, ≤0.5 | ibid. | **leave 0.5** |
| **`foliage`** | **0.0** | green leaf **0.94–0.99** | canopy tile **0.39–0.78** | Monteith & Unsworth *Principles of Environmental Physics* 4th ed. Table A.3 + ch. 4 (Beer–Lambert `tau = exp(−k·LAI)`, `k ≈ 0.5`, LAI 1–3 in a 0.333 m path); Jones *Plants and Microclimate* ch. 2; Campbell & Norman ch. 15 | **R12's 0.9 is supported** |

**On R12 specifically.** `foliage = 0.0` makes a flammable thermal solid strictly
adiabatic once `cool_shift` dies (design §3.1), and under R10 a small
`conductivity` provably cannot save it — §3.4 just measured that a cellulosic
conduction face is **identically zero below a 256 K gap**, so the conduction
rescue would not even engage. The fix has to be radiative, which is R12.

The literature splits the way it does for every partly-filled row: the **leaf**
is 0.94–0.99, the **canopy tile** is 0.39–0.78. R12's 0.9 sits between them,
biased toward the emissivity reading. **Any value above ~0.4 closes the
ratchet** — it is a rate question, not a yes/no — so 0.9 is safe, defensible as
the emissivity, and the number R12 ruled. **Recommend 0.9 as ruled.**

One inconsistency to name, not to fix: `foliage` also carries
`light_atten = [0, 0, 0]` ("explicit: no vision interaction"). At
`heat_atten = 0.9` the row becomes a tile that is transparent to visible light
and nearly black in the infrared. No real material does that. It is a
deliberate gameplay choice (a walk-through, see-through canopy) and it is
outside T3's scope — recorded so it is deliberate (§8, question 6).

**`glass` is still the row grey-body breaks on**, and P2b §13 item 5 has it
right: a pane's absorptivity for a 1556 K flame is ~0.4 (it passes ~52 % of the
flame's power as near-IR) while its emissivity at its own temperature is
0.90–0.95 (its own radiation is far-IR, which it is opaque to). The shipped 0.3
is about right as a *shield* and ~3× too low as an *emitter*. **No row value
fixes both**; splitting emit from absorb is a scheme change, not a row edit.
T3's recommendation is to leave 0.3 and record it, because moving it to 0.6
makes the shield wrong to fix the emitter, and T5 is not the patch to change the
scheme in.

### 6.2 Ignition temperature — ledger 10

The engine's scale: `temperature` is game degrees above `kelvin_ambient = 293`
with `k_temp_to_kelvin = 1`, so **game + 293 = kelvin** and `280 game = 573 K =
300 °C`, `300 game = 593 K = 320 °C`.

| quantity | literature | in game degrees | source |
|---|---|---|---|
| **piloted** ignition of wood, surface T | **300–365 °C** (commonly quoted 350) | **280–345** | Drysdale, *An Introduction to Fire Dynamics* 3rd ed. §6.3; Babrauskas, *Ignition Handbook* (2003) §7.6 |
| **auto**-ignition of wood, surface T | 500–600 °C | 480–580 | Babrauskas §7.6 |
| fine dead wildland fuels, piloted | 320–350 °C | 300–330 | Babrauskas §14 |

| row | ships | °C it reads | verdict | proposal |
|---|---|---|---|---|
| `wood` | 300 | 320 | inside the piloted band, slightly below its 332 °C midpoint | **300 — keep** |
| `furniture` | 280 | 300 | at the bottom edge of the band; thermally-thin crate stock ignites there | **280 — keep** |
| `kindling` | 280 | 300 | fine fuel, bottom of the band | **280 — keep** |
| `foliage` | 280 | 300 | fine dead fuel wants 300–330 | **280–300**; keep 280 |
| `door` / `door_closed` | 280 | 300 | it is wood, and `wood` is 300 | **300** — the only row that arguably moves |
| `hull` / `steel` / `glass` / `air` | 0 | — | non-flammable | **0 — correct** |

> **D5's ignition headline: the column is already right.** Every flammable row
> is inside the literature's piloted-ignition band. The only candidate edit is
> `door`/`door_closed` 280 → 300 for consistency with `wood`, and it is **not
> free**: `fire_T_ext[mat] = ignition_temp[mat] − ignition_to_ext_delta`
> (`= −200`), so it moves the `hotf` ramp's foot on those rows too.

**The caveat that matters more than the numbers.** Every literature value above
is a **surface** temperature. `combat.py::apply_temperature_ignition` compares
them against the tile's **bulk** temperature, and the tile is one node (R6). So
the model asks all 139 kg of the pinned column to reach 300 °C where reality asks
a 2.6 mm skin to. P2b §9 measured the cost: **20–45× slow to ignition.** R7
forbids correcting that in this column, and it should not be corrected here —
lowering `ignition_temp` to compensate would be tuning a *measured* quantity to
paper over a *modelling* choice. The honest fix is issue #68's two-node solid.

### 6.3 Combustion — ledger 12

#### The heat of combustion, per unit O₂

The engine's combustion is **O₂-driven**, so the right constant is the one
oxygen-consumption calorimetry is built on, and `combustion.cpp`'s own header
already cites it:

> **Huggett's constant: 13.1 ± 0.7 MJ per kg of O₂ consumed**, near-universal
> across organic fuels (±5 %).
> *Huggett, C. (1980), "Estimation of rate of heat release by means of oxygen
> consumption measurements", Fire and Materials 4, 61–65* — archived under
> `docs/papers/`.

Converting to the engine's `N_O2` unit (`N = 1.0` is 1 atm of gas in one tile):

    n_tile = P·V/(R·T) = 101325 · 0.277223 / (8.3145 · 293.15) = 11.525 mol
    1 unit of N_O2     = 11.525 mol O2 = 0.36878 kg O2
    x 13.1 MJ/kg-O2    = **4.8310 MJ per unit of N_O2**

Per unit **fuel** mass, for the cross-check: wood's gross heat of combustion is
**18–20 MJ/kg** and its *effective* (cone-calorimeter) heat of combustion is
**12–14 MJ/kg**, the difference being char that does not participate in flaming
combustion (Drysdale ch. 1 Table 1.13; Babrauskas, *Heat Release in Fires*).

#### What the engine actually pays — measured

| path | config | per unit `N_O2` | |
|---|---|---|---|
| fuel-bed deposit | `H_BED_M · 2^H_BED_SHIFT = 18125 · 128` | 2.320e+06 counts = **1.1040 MJ** | |
| flame / gas deposit | `H_fuel = 4.0` | 4.0 counts = **1.90 J** | |
| **total** | | **1.1040 MJ** | **22.9 % of Huggett** |

Cross-checked on the engine (`scratchpad/m_heat_per_o2.py`, a single crate on
the canonical `fire_timing_harness` bench): over a full burn
`Δ e_solid_deposit_sum = 8.0575e+12` raw = 1.2295e+08 counts = **58.51 MJ**,
against `Δhp = 29.5` i.e. `29.5 / 0.7 = 42.14` units of `N_O2` — so
**2.917e+06 counts = 1.388 MJ per `N_O2`**, which is `H_bed` plus 26 % of
re-absorbed radiation. The paper value is confirmed against the ledger.

> ### **The combustion headline: the fuel bed gets 580 000× more heat than the flame gas.**
> `H_BED / H_fuel = 2.32e6 / 4.0`. That ratio is two config dials and depends on
> none of the unit reasoning above. **Combustion puts 0.0002 % of a fire's heat
> into the air.**

That matters to this arc specifically. Design §3.1 says gas is heated by
combustion until P5 gives it radiative absorption — and measured, that channel
is **six orders of magnitude** below the fuel-bed one. With `cool_shift` gone
(R1) and conduction at R10 rates, a fire's plume carries essentially no energy.

#### What the derivation says the two should be

A burning solid's flame returns only part of its heat to the fuel surface; the
rest goes up in the plume. Flame-to-surface feedback for a burning solid or pool
is **~20–40 %** of the total heat release (Drysdale ch. 5's burning-rate balance
`ṁ" = (Q̇"_F + Q̇"_E − Q̇"_L)/L_v`; typical surface feedback 20–30 kW/m² against
100–300 kW/m² released). Splitting Huggett's 4.831 MJ at 25 / 75:

| dial | derived | as config | change |
|---|---|---|---|
| `H_bed` (fuel-bed feedback, 25 %) | 1.208 MJ = 2.538e+06 counts | **`H_BED_M = 19827`, `H_BED_SHIFT = 7`** | **+9.4 %** |
| `H_fuel` (plume, 75 %) | 3.623 MJ = 7.614e+06 counts | **cannot be expressed** | see below |

> **`H_BED` is right.** At the 25 % feedback reading the shipped value needs
> **+9.4 %** — one mantissa digit — which is far inside every other error in
> this report. The dial that P-K0 tuned by feel landed on the physics.

> **`H_fuel` cannot hold its derived value.** It enters as
> `quantize((double)H_fuel)`, a plain Q16.16 int32, so its ceiling is **32 768**
> and the derived value is 7.6e+06. T5 needs the same mantissa-plus-shift split
> `H_BED` already has: **`H_FUEL_M = 14872`, `H_FUEL_SHIFT = 9`**
> (`14872 · 512 = 7.614e+06`). That is a config schema change plus a
> `combustion.cpp` / `cuda_combustion.cu` pair, not a value edit — flagged as
> the single largest implementation item D5 produces.

#### Burn duration for a known fuel mass

Fuel is still hp-coupled: Pass B pays `wall_hp -= fuel_per_o2 · burn`
(`combustion.cpp:805`), so a row's O₂ budget is `hp / fuel_per_o2`.

| | `furniture` (hp 30) | `kindling` (hp 8) | `foliage` / `wood` (hp 60) |
|---|---|---|---|
| O₂ budget, units of `N_O2` | 42.86 | 11.43 | 85.71 |
| O₂ mass | 15.81 kg | 4.21 kg | 31.61 kg |
| **energy at Huggett** | **207.1 MJ** | 55.2 MJ | 414.1 MJ |
| wood burned (at 13 MJ/kg effective) | **15.9 kg** | 4.2 kg | 31.9 kg |
| energy the engine pays today | 47.3 MJ | 12.6 MJ | 94.6 MJ |

**Measured burn duration** (`fire_timing_harness.run_one`, one crate, 16×12
room, natural wind, shipped dials): a `furniture` tile burns **98.7 % of its
fuel in 142.5 s**; a second run to the `hp <= 0.5` cut took **108.3 s**. So:

| | engine today | at the derived heat of combustion |
|---|---|---|
| total energy released | 46.5 MJ | **203.6 MJ** |
| **mean heat-release rate over 108 s** | **430 kW** | **1.88 MW** |

Reality check, and it is a good one: NIST / Babrauskas furniture-calorimeter
data put a single wood pallet at 100–300 kW, a small wood crib at 200–500 kW and
an upholstered chair at 150–800 kW; a 1.5 m pallet **stack** is 3–4 MW
(Babrauskas, *Heat Release in Fires*; SFPE Handbook ch. 3-1).
**430 kW is a very plausible single crate. 1.88 MW is a small stack.**

#### The finding this exposes: `hp` and `thermal_mass` describe different objects

At Huggett, a `furniture` tile's fuel store is **15.9 kg of wood**. Its
`thermal_mass = 8` says the tile *is* **139 kg** of wood (P2b §2). So the row
stores 11.4 % of its own mass as fuel, and for the whole tile to be fuel `hp`
would have to be **262**, not 30 — **8.7×**.

Neither number is wrong on its own. A 0.333 m × 2.5 m column packed with 12 mm
plywood crates really does hold about 15–30 kg of wood, so **`hp = 30` at
`fuel_per_o2 = 0.7` describes a real crate stack almost exactly** — while
`thermal_mass = 8` describes a *solid wood column*. This is P2b §9's
object-vs-material split showing up between two columns of the **same row**, and
it is the third independent place this arc has met it (P2b §7 on `kindling` /
`foliage`, P2b §9 on the surface layer, here on `hp`). **Erik's, not mine**
(§8, question 7).

## 7. D6 — the summary table T5 executes from

*(pending)*

## 8. Open questions for Erik

*(pending)*

## 9. What did not hold

*(pending)*

## 10. The gate

*(pending)*
