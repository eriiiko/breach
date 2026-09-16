# P2b - calibration by derivation

> **Written incrementally** while the patch was implemented (design v3 section
> 11.5, row 40: a kill must cost no resume). Branch
> `12-p2b-calibration-derivation`, cut from `fire-12` at `400eaa2`.
>
> **Depends on:** `ray_engine_v2_design_v3_2026-09-15.md` section 9 (the
> calibration chain), section 2.3 (the ordinate weights sum to ONE, so `E°[T]`
> is the cell's TOTAL emission per tick), section 2.6 (the E° table and its
> owner), row 34 (calibrate against the DAMPED emission), section 11's P2b row;
> `report_p0.md` sections 0.6 and 0.8; `report_p2a.md` (floor 0 is live);
> `ray_engine_v2_survey_2026-09-12.md` section 9.3 (the reach curve).

## 0. What P2b is, in one paragraph

Today's `[physics.fire] rad_scale = 5.1427e-5` is a **fitted** dial: a number
chosen twice over (P-F1b, then re-anchored at G12) to make a two-tile ignition
inequality come out, with `sigma`, an area, `dt` and the game-Kelvin map
"folded in" but never actually evaluated. P2b evaluates them. The result,
`rad_scale_derived`, is written into `[physics.radiation]` and fed to the
**sweep's** emissive table only; `[physics.fire] rad_scale` keeps its fitted
value until P3c, so the live fold, the live game and every golden are untouched.
Nothing here is a feel decision: the number is arithmetic, and the two places
where the arithmetic is genuinely ambiguous are named, chosen with an argument,
and reported with their sensitivity (sections 3.1, 3.2).

**The headline**: physics wants `rad_scale = 2.13e-8`. The shipped dial is
**2419x larger** - 3.4 orders of magnitude - and that factor is, exactly, the
heat capacity the game currently behaves as if a furniture tile has: **80 J/K,
about 65 grams of wood**, where the pinned physical crate is **194 kJ/K, about
139 kg**. Section 6.

## 1. The currency - one heat count, verified against the code

The design states the currency and P2b checks it against the code before using
it.

**`cpp/src/temperature_solver.cpp:326-336`**, the `heat` -> `temperature`
convert branch for a thermal solid:

```cpp
int32_t deposit = heat[i];
if (deposit <= 0) continue;
if (ts[i]) {
    int shift = heat_inv_shift[i];    // log2(thermal_mass), >= 0
    int32_t gain = deposit >> shift;  // Q16.16 / 2^shift, still Q16.16
    heat_saturating_add(&temperature[i], gain);
```

`heat_inv_shift` is `log2(thermal_mass)`, frozen at table build
(`src/simulation/materials.py:326-357`, which is also where `thermal_mass` is
required to be a power of two). `temperature` is Q16.16 **game degrees**, and
`[physics.temperature_scale]` is `kelvin_ambient = 293`, `k_temp_to_kelvin = 1`
(config.toml:813-814), so **one game degree is one kelvin** and game T is an
absolute-temperature offset above 293 K, not a Celsius reading.

So `n` heat counts deposited on a tile of thermal mass `M` raise it by
`n / (M * 65536)` K. Setting `M = 1, n = 1`:

> **One heat count is the energy that raises a `thermal_mass = 1` tile by
> `1/65536` K.** Verified, not assumed.

Two consequences used below. First, `thermal_mass` is *inverse* to temperature
response - doubling it halves the degrees per count - so it is a **heat
capacity** in exactly the physical sense the moment the currency is pinned.
Second, the currency is defined by the deposit path alone; `rad_scale` is what
puts *radiation* into that currency, which is all P2b changes.

## 2. Pinning it to joules on the furniture row

Give the furniture row its real heat capacity `C_tile = rho_c * V_tile` joules
per kelvin. That row has `thermal_mass = 8` (config.toml:1575), so 8 * 65536
counts raise it by 1 K, and

    J_per_count = rho_c * V_tile / (8 * 65536)

With the two choices argued in section 3 (`rho_c = 0.7 MJ/m3/K`,
`V_tile = 0.333^2 * 2.5 = 0.277223 m3`):

    C_tile      = 0.7e6 * 0.277223       = 194 055.8 J/K = 194.06 kJ/K
    J_per_count = 194 055.8 / 524 288    = 0.37013197 J

`rho_c = 0.7 MJ/m3/K` is not mine: it is the value the config row already names
- `thermal_mass = 8  # wood-like (real rho*c ~0.7)` - and it sits inside the
literature band for wood (section 7). Choosing it means the pin *confirms* the
existing row rather than redefining it.

## 3. The two ambiguous factors

Both are simple multiplicative factors on the answer, and design section 9 does
not settle either. Each is named, chosen with an argument, and reported with its
sensitivity.

### 3.1 `A_rad` - the radiating area

**Chosen: `A_rad = 4 * tile_size_m * deck_height = 4 * 0.333 * 2.5 = 3.33 m2`
- all four lateral faces of the cell's column.**

Design section 9 says "the tile face `0.333 m x 2.5 m`", i.e. **one** face,
0.8325 m2. Design section 2.3 says something else, and section 2.3 wins:

> `w_m = ONE / 16 = 4096` ... `Sum w_m = ONE` so the baked total-power `E°`
> drops in unchanged

**The argument, done on the scheme itself.** Take an isolated opaque cell
(`a_i = ONE`) in an ambient bath. Summing its `emitted = (src * a_i) >> 16` over
the 16 ordinates gives exactly `E°[0] + f*(E°[T] - E°[0])`, because
`src = amb_m + f*ex_m` and both `amb_m` and `ex_m` carry the weight `w_m`, which
sums to ONE. Its `abs_mat` sums to `E°[0]`. Its **net** loss is therefore
`E°[T] - E°[0]` per tick, undamped - and the physical net loss of a body of
total radiating area `A` in a bath at `T_amb` is `sigma * A * (T^4 - T_amb^4)`.
Equating the two:

    E°[T] = rad_scale * K^4   must equal   sigma * K^4 * A_rad * dt / J_per_count
    =>  rad_scale = sigma * A_rad * dt / J_per_count,   A_rad = the TOTAL area

`E°` is a total, so `A_rad` must be a total. One face would be right only if
`E°` were a per-face amount, which `Sum w_m = ONE` explicitly makes it not.

**Which surface is "total" in a 2-D sweep?** The cell is a column: footprint
`tile_size_m x tile_size_m` (0.333 m - read from
`levels/fire_tuning/level.toml:13`, and the same on every shipped 0.333 m level;
`levels/airlock_demo` at 1.0 m and `levels/bench_two_room` at 0.5 m are the
exceptions, see section 13) and deck height 2.5 m (`[physics.water] ceiling_h`,
the same 2.5 m the `k_leak` derivation uses). The four **lateral** faces,
4 * 0.333 * 2.5 = 3.33 m2, are the surface that participates in in-plane
transport. The two horizontal faces, 2 * 0.333^2 = 0.222 m2, are the floor and
ceiling - they are the **out-of-plane channel**, which the scheme already has a
separate, deliberately dormant name for (`k_leak`, `[physics.radiation]`,
derived at ~0.10 for a 2.5 m deck and set to 0 until ruled). Putting the roof
and floor into `A_rad` would double-count against `k_leak` the day it is
switched on.

**Sensitivity: exactly 4x.** The one-face reading gives
`rad_scale_derived = 5.3141e-9` and a fitted/derived ratio of **9677** instead
of 2419. Reported, not chosen.

### 3.2 `V_tile` - how much solid is in a tile

**Chosen: the full column, `V_tile = 0.333^2 * 2.5 = 0.277223 m3` - the tile
treated as solid wood, 139 kg of it.**

Three reasons, in order of weight:

1. **It is geometrically consistent with 3.1.** The same block must radiate and
   store. `A_rad = 4sH` is the lateral surface of a *filled* column; pairing it
   with a hollow interior gives an object with a full-size radiating skin and a
   fraction of the matter behind it - which is a real physical object (that is
   the lumped-capacity caveat, section 9) but is not the object the extinction
   coefficient `a_i` describes. In the sweep `a_i` is the cell's extinction: a
   fully opaque cell absorbs the whole incident stream. The homogeneous filled
   column is the object the scheme already models.
2. **It makes every other row interpretable at once.** With this pin,
   `thermal_mass` reads directly as `rho_c` in MJ/m3/K, one row at a time, and
   the whole material table can be checked against literature (section 7) -
   which turned out to be the most valuable check in P2b.
3. It is design section 9's literal wording: "its real volumetric heat capacity
   x tile volume".

**Sensitivity: `rad_scale_derived` is exactly inverse in `V_tile`.** A fill
fraction `phi` multiplies the derived scale by `1/phi`:

| `phi` | what it is | `C_tile` | `rad_scale_derived` | fitted/derived |
|---|---|---|---|---|
| **1.000** | **full solid column (CHOSEN)** | **194.06 kJ/K** | **2.126e-8** | **2419** |
| 0.500 | half full | 97.03 kJ/K | 4.251e-8 | 1210 |
| 0.216 | 7.5 stacked 0.333 m crates, 12 mm plywood | 41.92 kJ/K | 9.841e-8 | 523 |
| 0.029 | ONE 0.333 m crate, alone in a 2.5 m column | 5.59 kJ/K | 7.381e-7 | 70 |

`rho_c` is a second, milder lever in the same place (`rad_scale ~ 1/rho_c`):
oven-dry softwood at 0.52 MJ/m3/K gives 2.861e-8, wood at 12 % moisture content
at 0.90 gives 1.653e-8. The whole literature band moves the answer by less than
a factor of two - which is the point: **no honest choice anywhere in this
section gets within two orders of magnitude of the fitted dial.**

This is the honest lever behind Erik's *"the tuning will be more in the heat
capacity of items"*. It is `thermal_mass` per row, and it now has units.

## 4. The derived scale

    rad_scale_derived = sigma * A_rad * dt / J_per_count

    sigma       = 5.670374419e-8  W/m2/K4   (CODATA 2018)
    A_rad       = 4 * 0.333 * 2.5 = 3.33    m2   (section 3.1)
    dt          = 1/24                      s    ([clock] ticks_per_second = 24)
    J_per_count = 0.37013197                J    (section 2)

    = 5.670374419e-8 * 3.33 / 24 / 0.37013197
    = 1.888235e-7 / 24 / 0.37013197
    = 7.867647e-9 / 0.37013197

    rad_scale_derived = 2.125632e-08   heat counts per K^4

**A consequence that falls straight out, and is the largest single result of
P2b: at the derived scale the Fleck damping never engages.** `f` is
`T_abs / max(T_abs, 4L)` with `L` the cell's own excess emission per tick
expressed in Q16.16 degrees, so `L` scales with `rad_scale` and `g = 4L/T_abs`
falls by the same 2419x. Computed through the integer reference's own
`fleck_L_solid_q` / `fleck_f_q` at both scales, for every shipped absorbing row:

| row | `a` | `thermal_mass` | undamped through, FITTED | undamped through, DERIVED |
|---|---|---|---|---|
| wood / door / door_closed | 1.0 | 8 | 1068 game | **15 996 game (the whole table)** |
| furniture / kindling | 0.5 | 8 | 1424 game | **the whole table** |
| hull / steel | 1.0 | 32 | 1872 game | **the whole table** |
| glass | 0.3 | 16 | 2272 game | **the whole table** |

The FITTED column reproduces P0b section 0.8a and P2a section 1.1 exactly -
1068 / 1424 / 1872 / 2272 - which is the check that the probe is the engine's own
arithmetic and not a re-derivation. At the derived scale `f == 2^24` exactly at
every one of the 4000 buckets for all eight rows; the worst case is wood at the
table top, `g = 0.70`, still inside the provably-monotone range `g <= 1`.

The physical statement behind it is simple: **a 139 kg wood column cannot
radiate an appreciable fraction of its own heat content in 1/24 s**, even at
16 289 K. The Fleck factor exists to stop an explicit update overshooting, and
an object with a physical heat capacity does not overshoot at 24 Hz.

Three things follow:

- Design **row 34's instruction, "calibrate against the damped emission", is
  satisfied trivially at the derived scale** - the damped emission IS the
  black-body emission, everywhere on the table. The bench (section 10) reports
  `f` at every temperature it uses, and it is 1.000000 at the derived scale.
- Design section 12 item 4, the **driven-source question** ("a burning tile held
  at 1263 game radiates only 68-73 % of black body"), **disappears at the
  derived scale**. It was an artefact of a 2419x-oversized emission, not a
  property of the scheme. Worth Erik's attention because it was live enough to
  be ruled on.
- The clamp is untouched in kind: `E°^-1(Phi)` is scale-invariant (both `Phi`
  and the table carry the same `rad_scale`), so the maximum-principle clamp
  binds at exactly the same temperatures. It does become *less* load-bearing,
  because the overshoot it caps is the same one the Fleck factor no longer has
  to damp.

## 5. Sanity checks against reality

### 5.1 The irradiance the scheme delivers, and the one thing rad_scale cannot move

The check design section 9 asks for: what irradiance does a tile one, two and
three tiles from a plateau fire actually receive, against the 10-12 kW/m2
piloted-ignition flux for wood?

Converting the sweep's fluence to W/m2 is exact and uses no new assumption,
because `rad_scale = sigma * A_rad * dt / J_per_count` *is* the conversion:

    q [W/m2]  =  Phi [counts] * J_per_count / (A_rad * dt)  =  sigma * Phi / rad_scale

**`rad_scale` cancels.** That is worth stating loudly, because it is the single
most useful structural fact in P2b: *the irradiance a receiver sees, and
therefore its radiative-equilibrium temperature `E°^-1(Phi)`, do not depend on
the calibration at all* — except through the Fleck factor, which is the only
scale-dependent term in the scheme. `E°^-1` inverts the same table `Phi` was
written in, so the `rad_scale` in the numerator and the denominator are the
same number. What the calibration sets is the **rate**: how many degrees per
tick that flux buys, given the tile's heat capacity.

Measured (section 10's bench, `tests/_fire_lab/reach.csv`), a single furniture
tile held at the 1263-game plateau (1556 K), shear S16, `k_leak = 0`:

| distance | `Phi` (derived) | **q, excess** | `E°^-1(Phi)` | vs 10-12 kW/m2 |
|---|---|---|---|---|
| 1 tile | 8 946 | **23.4 kW/m2** | 508 game (801 K) | 2.1x the threshold |
| 2 tiles | 5 990 | **15.6 kW/m2** | 432 game (725 K) | 1.4x |
| 3 tiles | 4 430 | **11.4 kW/m2** | 380 game (673 K) | **1.04x — the crossing** |
| 5 tiles | 2 746 | 6.9 kW/m2 | 304 game (597 K) | 0.6x |

A *wood* source (`a = 1.0` rather than furniture's 0.5) is exactly double:
46.9 / 31.1 / 22.8 kW/m2, which reproduces P0 section 0.6's geometric factors
(`G = 0.1414 / 0.0939 / 0.0687` shear, times `sigma * 1556^4 = 332.4 kW/m2`) to
three digits — the bench and P0's independent probe agree.

### 5.2 The survey's reach curve, hit on both of its points

Survey section 9.3 derived, from view factors and 10-12 kW/m2 and with no
reference to any of this machinery, **~3 tiles for one burning crate and ~8 for
a fully involved room**. Interpolating the bench's own `q` column for the
11 kW/m2 crossing:

| source | survey 9.3 | **bench, derived scale** |
|---|---|---|
| one furniture tile | ~3 tiles | **3.15 tiles** |
| 2x2 furniture patch | (~8 for a "room") | **7.43 tiles** |
| 4x4 furniture patch | — | 16.5 tiles |

That is a hit on both points, and it is not a fit: the survey computed it from
Stefan-Boltzmann and a view factor, and the bench measured it through the
integer sweep at a calibration derived from a heat capacity. **The design
instinct and the physics agree**, which is what section 9 hoped for.

Two caveats, both structural and both already named in the design:

- The engine's **own** ignition criterion is more permissive than 11 kW/m2. It
  is `temperature >= ignition_temp` (`combat.py::apply_temperature_ignition`),
  and 280 game is the radiative equilibrium of only
  `sigma * (573^4 - 293^4) = 5.69 kW/m2`. So judged the engine's way the same
  crate reaches **5.75** tiles, not 3.15. The two criteria differ by 2x in flux
  and therefore by ~1.8x in distance (2-D `1/r`).
- The sweep has no out-of-plane loss while `k_leak = 0`, so it is `1/r`
  everywhere, where the survey's view factor is a 3-D `1/r^2` inside a ceiling
  height (survey section 9.5). **Turning the dormant `k_leak = 0.10` on moves
  the engine-criterion reach from 5.75 tiles to 3.92** and the 11 kW/m2
  crossing from 3.15 to 2.37 — i.e. the derived leak coefficient lands the
  engine's own criterion almost exactly on the survey's view-factor number.
  That is a real argument for ruling `k_leak` on (section 13).

### 5.3 Literature values used, with sources

| quantity | value used | source |
|---|---|---|
| `sigma` | 5.670374419e-8 W/m2/K4 | CODATA 2018 |
| wood `rho*c` | **0.7 MJ/m3/K** (band 0.43-0.90) | *Wood Handbook* FPL-GTR-190 ch. 4: `c_p,dry = 0.1031 + 0.003867*T` kJ/kg/K = 1.236 at 293 K; softwood `rho` 350-550 oven-dry; at 12 % MC `c_p ~ 1.7`, `rho ~ 510` -> 0.87 |
| wood thermal diffusivity | 1.5e-7 m2/s | `k ~ 0.12 W/m/K` / `rho*c ~ 0.8 MJ/m3/K` (same source) |
| piloted ignition of wood | 10-13 kW/m2 critical flux; 30-70 s at 20-25 kW/m2 | Drysdale, *An Introduction to Fire Dynamics* 3rd ed. ch. 6; Babrauskas, *Ignition Handbook* (2003) |
| mild steel `rho*c` | 3.40-3.85 MJ/m3/K | Incropera & DeWitt Table A.1 (AISI 1010, 300 K: 7832 x 434 = 3.40); mild steel `c ~ 490` -> 3.85 |
| soda-lime glass `rho*c` | 1.88-2.10 MJ/m3/K | Incropera & DeWitt Table A.3 (plate glass 2500 x 750 = 1.88) |
| emissivities | section 8 | Incropera & DeWitt Table A.11 (total hemispherical, ~300 K); Siegel, Howell & Menguc, *Thermal Radiation Heat Transfer* |

## 6. The fitted/derived ratio, and what it means

    rad_scale_fitted  = 5.1427e-5      ([physics.fire], P-F1b then re-anchored at G12)
    rad_scale_derived = 2.125632e-8    (this patch)

    rad_scale_fitted / rad_scale_derived = 2419.4        (3.38 orders of magnitude)

**The orchestrator's back-of-envelope — "some three to four orders of magnitude
above physics" — is confirmed, at 3.4.** I re-derived it independently rather
than checking it, and the two agree.

**What the ratio IS, exactly.** `rad_scale = sigma * A_rad * dt / J_per_count`,
and `sigma`, `A_rad` and `dt` are not negotiable at the 2000x level (the widest
honest `A_rad` ambiguity is the 4x of section 3.1). So a fitted scale 2419x too
large is, term for term, a `J_per_count` 2419x too small — i.e. **the live game
behaves as if a furniture tile had**

    C_eff = 194 056 J/K / 2419.4  =  80.2 J/K

    at c_p = 1236 J/kg/K:   64.9 grams of wood
    at rho*c = 0.7 MJ/m3/K: 115 cm3 — a fill fraction of 4.1e-4 of the tile

**Sixty-five grams.** A fistful of kindling, in a tile that is nominally a
0.333 m x 0.333 m x 2.5 m crate column of 139 kg. That is the honest reading of
the shipped dial, and it is exactly why fire in the current build heats things
so fast: the radiation is not too strong, the **objects are effectively
weightless**.

Read the other way round — holding the fitted dial and asking what tile it
describes — a `thermal_mass = 8` tile at 80.2 J/K is a 12 mm plywood sheet of
about 0.09 m2. So the current dial is, unintentionally, a *very thin surface
layer* calibration (section 9 takes that seriously).

Nothing is changed by this: `[physics.fire] rad_scale` keeps its fitted value
until P3c. The ratio is what P3 has to absorb, and it is the size of the feel
change the HUMAN-TEST there will be judging.

## 7. The implied rho*c of every material row

Once the currency is pinned, **every `thermal_mass` is a heat capacity by
definition** — no further assumption:

    C_row = thermal_mass * 65536 * J_per_count      [J/K]
    rho*c = C_row / V_tile                          [J/m3/K]

Equivalently `rho*c = thermal_mass / 8 * 0.7 MJ/m3/K`, since the pin is on
`thermal_mass = 8` at 0.7. Every shipped row, against literature:

| row | `thermal_mass` | `C_tile` | **implied rho*c** | literature | ratio | verdict |
|---|---|---|---|---|---|---|
| air | 0 | — | (gas branch) | 0.0012 | — | **correct**: a solid row for air would be 600x too heavy; `thermal_mass = 0` routes it to the gas branch, which is right |
| wood | 8 | 194.1 kJ/K | 0.70 | 0.43-0.90 | 0.78-1.63 | **good** (the pin) |
| door / door_closed | 8 | 194.1 kJ/K | 0.70 | 0.43-0.90 | 0.78-1.63 | **good** |
| furniture | 8 | 194.1 kJ/K | 0.70 | 0.43-0.90 | 0.78-1.63 | **good** (the pin row) |
| kindling | 8 | 194.1 kJ/K | 0.70 | 0.43-0.90 | 0.78-1.63 | **good** as a material; see below as an *object* |
| glass | 16 | 388.1 kJ/K | 1.40 | 1.88-2.10 | 0.67-0.74 | **light by ~30 %** |
| hull / steel | 32 | 776.2 kJ/K | 2.80 | 3.40-3.85 | 0.73-0.82 | **light by ~20-27 %** |
| foliage | 8 | 194.1 kJ/K | 0.70 | 0.02-0.14 (a canopy *tile*, 1-5 % fill of leaf tissue at 1.5-2.7) | 5-35x | **wrong by an order of magnitude** — out of scope by design row 6, recorded |

**The headline is a good one: the material table's structure is sound.** Every
solid family lands inside a factor of 1.5 of literature, and — the telling part
— *all of them err in the same direction*, light by 0-30 %. That is the
signature of the power-of-two quantization, not of bad judgement: the table is
right and the pin is one notch low.

**And the pin has a better value available.** Repinning on 12 %-moisture-content
wood (`rho*c = 0.9 MJ/m3/K`, which is what furniture aboard a ship actually is,
not oven-dry stock) lands *every* row inside 15 % of literature:

| row | implied rho*c at a 0.9 pin | literature | ratio |
|---|---|---|---|
| wood / furniture / door / kindling | 0.90 | 0.43-0.90 | 1.00-2.09 |
| glass | 1.80 | 1.88-2.10 | 0.86-0.96 |
| hull / steel | 3.60 | 3.40-3.85 | 0.94-1.06 |

`rad_scale_derived` would then be **1.6533e-8** (ratio 3111). This is a real
open question for Erik (section 13), not a change made here: the shipped value
uses the 0.7 the config row itself names.

**Two rows carry an object-vs-material confusion that the pin makes visible**,
and neither is mine to fix:

- `kindling` and `furniture` have identical `thermal_mass`. As *materials* both
  are wood and that is right. As *objects* kindling is sticks with a huge
  surface-to-volume ratio and a crate is a box, and those should not heat at
  the same rate. Section 9 is the general form of this.
- `foliage` inherits wood's `thermal_mass` while being ~97 % air. It is also
  the one row with `heat_atten = 0` (design row 6), so it is radiatively inert
  and the error is currently unreachable — but the day trees are authored, the
  first thing to fix is the heat capacity, not the extinction.

## 8. Emissivities proposed per row (NOT applied)

Design section 9 item 3 wants `a_i` to be emissivities from the literature.
**Nothing here is applied** — the material rows are Erik's (D6). This is the
proposal, with its sources and, more importantly, with the reason it is not a
straight table lookup.

**The complication that has to be stated first.** In the sweep `a_i` does two
jobs at once. By Kirchhoff it is the cell's **emissivity** (the fraction of
black body it radiates) and, in the same multiply, it is the cell's
**extinction** — the fraction of a passing stream it absorbs, i.e. its
*geometric opacity*. For a solid slab filling the tile those coincide and the
literature emissivity is the right number. For a *partly filled* tile they do
not: a crate stack is ~90 % emissive on the wood it has, and maybe 50 % opaque
across the tile. The shipped `furniture = 0.5` is visibly an **opacity** number
— the config comment says so in as many words ("partial: smoke/air drift past
crates") — not an emissivity.

| row | shipped `heat_atten` | literature emissivity | source | proposal |
|---|---|---|---|---|
| wood | 1.0 | 0.82-0.92 (planed oak, pine, beech) | Incropera Table A.11 | **0.90** |
| door / door_closed | 1.0 | 0.82-0.92 (as wood); painted 0.90-0.96 | ibid. | **0.90** |
| furniture | 0.5 | wood 0.90, but the tile is partly filled | — | **leave 0.5 as an OPACITY**, and see section 13: the two jobs want separating |
| kindling | 0.5 | as furniture | — | **leave 0.5**, same reason (sticks are even less opaque) |
| hull / steel | 1.0 | oxidized mild steel 0.78-0.82; painted 0.90-0.96; polished stainless 0.17 | Incropera Table A.11 | **0.85** (a painted/oxidized hull); a *polished* interior would be 0.2 and would change shielding a lot |
| glass | 0.3 | **0.90-0.95** | Incropera Table A.11; Siegel & Howell ch. 5 | **0.92 — the shipped 0.3 is physically wrong** |
| air | 0.0 | ~0 for N2/O2 (homonuclear diatomics do not absorb in the IR) | Siegel & Howell ch. 10 | **0.0, correct** — and note it is CO2, H2O and soot that make real smoke absorb, which is P5's `heat_absorb` |
| foliage | 0.0 | green leaves 0.94-0.98 | Incropera Table A.11 | **out of scope** (design row 6); a canopy tile's opacity is the honest number, not the leaf's emissivity |

**The one row that is a genuine physical error is `glass`.** Its
`heat_atten = 0.3` is a *visible-light* intuition. Soda-lime glass is close to
opaque beyond about 4.5 um, and a 1556 K flame has ~80 % of its power above that
wavelength, so a window is nearly a black body to fire heat while being clear to
the eye. This matters in gameplay: today a glass pane lets 70 % of a fire's heat
through to whatever is behind it, and physically it would stop almost all of it
and re-radiate as a hot pane. It is also exactly the kind of case the new engine
handles naturally, since light and heat are separate channels on the same sweep
(`light_atten` stays 0.1, `heat_atten` goes to 0.92).

## 9. The lumped-capacity caveat — and whether `thermal_mass` is the right lever

**State it plainly.** The model has ONE temperature per tile. There is no
surface-versus-bulk gradient, so a tile heats as a lump: all 139 kg of the
pinned crate must come up together. Real piloted ignition is the opposite — only
a thin surface layer has to reach the ignition temperature, and the bulk behind
it is still cold.

**Quantified on the bench's own numbers.** A furniture tile one tile from a
plateau crate absorbs `a * q * A_rad = 0.5 * 23.4 kW/m2 * 3.33 m2 = 39.0 kW`.
Against `C_tile = 194 kJ/K` that is **0.201 K/s**, so from ambient to the
280-game ignition point takes **23 minutes** — and that is *ignoring* every loss.
Real wood under 23 kW/m2 ignites (piloted) in **30-70 s**. The lumped model is
**20-45x slow**.

**And the loss channel is not ignorable — it is decisive.** `furniture` has
`conductivity = 0.0`, so `cool_shift = 13` (e-fold 341 s) is its only other
channel. Solving `a*(Phi - E°[T])/M == T >> cool_shift` exactly, at the derived
scale:

| distance | `E°^-1(Phi)`, radiation alone | **T\* with `cool_shift = 13`** |
|---|---|---|
| 1 tile | 508 game | **67 game** (360 K, 87 C) |
| 2 tiles | 432 game | 45 game |
| 3 tiles | 380 game | 33 game |
| 5 tiles | 304 game | 20 game |

**At the derived scale, with the shipped `cool_shift`, radiative ignition does
not happen at any distance.** The reach curve of section 10 is a genuine upper
bound and the ambient decay takes essentially all of it back. To get the 1-tile
receiver to 280 game the row would need `cool_shift >= 16` (e-fold 2731 s, 8x
today's):

    cool_shift 13 -> 67 game    14 -> 129    15 -> 230    16 -> 343 (ignites)

**So is `thermal_mass` the right lever?** Two readings, and the difference is a
factor of ~32:

1. **`thermal_mass` = bulk heat capacity** (what P2b pinned). Then it is the
   right lever for *thermal inertia* — how long a thing stays hot, how slowly it
   cools — and it is straightforwardly interpretable per row (section 7). But it
   is then far too large for ignition timing, and ignition has to come from
   somewhere else (a longer `cool_shift`, a bigger fire, or a surface model).
2. **`thermal_mass` = effective surface-layer heat capacity.** The layer that
   must reach ignition temperature in a 45 s piloted exposure is the thermal
   penetration depth `sqrt(alpha * t)` = **2.6 mm** of wood; over the cell's four
   faces that is 8.65 litres, **6.06 kJ/K** — `C_bulk / 32`. Pinning the currency
   there instead gives `rad_scale_derived = 6.81e-7` and a fitted/derived ratio
   of **75**, and it is equivalent to `thermal_mass = 0.25` on the furniture row,
   **which the table cannot express**: `thermal_mass` must be a power of two, and
   0 is taken (it means "gas branch").

Reading 2 is very close to the `phi = 0.029` row of section 3.2's sensitivity
table (one 12 mm plywood crate alone in a 2.5 m column, 5.59 kJ/K, ratio 70) —
two independent routes to the same order. **The physically defensible span for
`J_per_count` is therefore about 35x, bulk to surface layer, and the fitted dial
sits 70x above even the most permissive end of it.**

**This is a finding, not a change** (D6). What it tells P3 and Erik: the honest
long-run fix is a two-node surface/bulk model for thermal solids (a thin
skin that ignites plus a bulk that stores) — which is a real design addition, not
a dial — and until then whichever reading is adopted must be adopted
*consistently*, because the same `thermal_mass` currently sets both ignition
timing and cool-down inertia and they want opposite values.

## 10. The reach bench

`tools/fire_tuning_lab.py --reach`. Outputs (untracked, regenerable):
**`tests/_fire_lab/reach.png`** and **`tests/_fire_lab/reach.csv`**.

### 10.1 What it measures, and why that is the engine's own criterion

`combat.py::apply_temperature_ignition` lights a flammable tile when
`temperature >= ignition_temp`. A tile heated by radiation alone settles where
absorbed equals emitted, which for the material share `a_i` (it both absorbs and
emits, Kirchhoff) is `E°[T] == Phi` — i.e. at `E°^-1(Phi)`, the inverse of the
emissive table applied to the sweep's own `rad_fluence` plane. **So the reach of
a fire is the distance at which `E°^-1(Phi)` falls through the receiver's
`ignition_temp`**, which is the same convention the P1/P2a isotropy gate already
uses ("the `E°[280]` ignition crossing") read as a profile instead of a radius.

Probes are AIR (`a = 0`): they neither absorb nor shadow, so `Phi` at a probe is
what a receiver placed there would see before it starts shielding the cells
behind it. Free field 145x145, shear S16, source held at 1263 game.

It is an **upper bound by construction** — radiative equilibrium with no
`cool_shift` and no conduction. Section 9 measures how much that costs.

### 10.2 What the curves actually do (I read the figure before writing this)

Four panels. **(a) furniture source, (b) wood source**, each with the derived
(thin solid) and fitted (thick dashed) curve for 1x1 / 2x2 / 4x4 sources plus
the dormant `k_leak = 0.10` on the 1-tile case; **(c)** the derived/fitted ratio
in absolute temperature; **(d)** the live level against the free field.

- **In panel (a) the two scales' curves are not distinguishable** — the derived
  curve lies exactly under the fitted one over the whole 56-tile profile. That
  is the section 5.1 result made visual: `rad_scale` cancels out of
  `E°^-1(Phi)`, and a *furniture* source at 1263 game is undamped at BOTH scales
  (floor 0 leaves `a = 0.5, thermal_mass = 8` undamped through 1424 game, so
  `f = 1.000000` fitted as well as derived).
- **In panel (b) they separate**, cleanly and by a constant factor near the
  source. A *wood* source (`a = 1.0`) is undamped only through 1068 game at the
  fitted scale, so at the plateau its `f` is **0.6792** fitted against
  **1.000000** derived. Panel (c) shows the ratio starting at exactly
  `(1/f)^(1/4) = 1.1016` — which is the analytic ceiling, since `T_abs^4` is
  linear in the fluence and `f` is the only scale-dependent term — and decaying
  toward 1 far out as the ambient bath comes to dominate `Phi`. **This is the
  only place in the whole patch where the calibration changes what the sweep
  delivers**, and it is a 10 % temperature effect, worth 1-3 tiles of reach.
- Every profile is a smooth monotone decay with no ray-effect scalloping at
  these distances, and the curves nest correctly by source size.
- **`k_leak = 0.10` is by far the biggest lever on the picture** — much bigger
  than the calibration. It pulls the 1-tile furniture curve down steeply and
  moves the ignition crossing from 5.75 tiles to 3.92.
- **Panel (d) is the end-to-end proof.** The live `gmap.rad_fluence`, read off a
  real level (`fire_tuning`, station-3 furniture sample at (46, 8), probed 30
  tiles down the open hall) after a whole `Simulation.step`, is **equal to the
  free-field bench at all 30 probes, integer for integer** — `Phi = 8946, 5990,
  4430, ... 166`. Two things fall out of that: air really is radiatively inert
  (`a = 0`, so a real hall reproduces a free field exactly), and **the D3
  binding works**: the engine baked its table at 2.1256e-8, on the conductor's
  own path, with no help from the bench.

### 10.3 The numbers

Reach = the last tile whose `E°^-1(Phi)` is at or above the threshold.
"q at 11 kW/m2" is the interpolated crossing of the piloted-ignition flux.

| source | scale | `f` at source | 1x1 | 2x2 | 4x4 |
|---|---|---|---|---|---|
| furniture | fitted | 1.000000 | 5 t | 11 t | 22 t |
| furniture | **derived** | 1.000000 | **5 t** | **11 t** | **22 t** |
| wood | fitted | 0.679156 | 7 t | 11 t | 20 t |
| wood | **derived** | 1.000000 | **8 t** | **13 t** | **23 t** |

(threshold: furniture's `ignition_temp = 280`; the wood receiver at 300 is one
tile shorter in each case, in the CSV.)

`k_leak = 0.10`, 1 tile: furniture **5 -> 3 t**, wood **8 -> 6 t**.

Interpolated crossings, derived scale, furniture source:

| | 11 kW/m2 (piloted) | 5.69 kW/m2 (== the engine's 280-game criterion) |
|---|---|---|
| 1x1 | 3.15 t | 5.85 t |
| 2x2 | 7.43 t | 11.38 t |
| 4x4 | 16.46 t | 22.68 t |
| 1x1, `k_leak = 0.10` | 2.37 t | 4.04 t |

The flux crossing at 5.69 kW/m2 and the `E°^-1(Phi) >= 280` crossing agree to
0.1-0.2 tiles, which is an internal consistency check on the whole conversion
chain: they are computed by different routes (a physical flux threshold vs the
integer table inverse) and they land on the same tile.

### 10.4 How it is built

- Sweeps source **material** (`furniture`, `wood` — the second included
  precisely because it is where the two scales differ), source **size**
  (1, 2x2, 4x4), **scale** (fitted, derived) and `k_leak` (0, 0.10 on the 1-tile
  case). 16 sweeps, ~20 s total.
- Both scales are two bakes of `EmissiveTable`, not two builds
  (`emissive.rad_scale` is settable from Python).
- Dials go through `fire_timing_harness.apply_overrides` / `restore_overrides`,
  the canonical bench seam, via a `REACH_DIALS` panel (empty by default, so an
  untouched panel measures `config.toml`). The interesting overrides are the two
  emission scales and `materials.*.thermal_mass`.
- The CSV carries `phi`, `t_cap` (`E°^-1(Phi)` in game units) and
  **`q_Wm2`** — the physical irradiance — per curve, so section 5's claim that
  `rad_scale` cancels can be read off the file rather than taken on trust.
- Deterministic: the sweep is pure integer and the scene is built, not sampled.

## 11. What changed in the tree

| file | change |
|---|---|
| `config.toml` | **NEW** `[physics.radiation] rad_scale_derived = 2.125632e-08`, with the full derivation, both stated assumptions and their sensitivities in the comment. `[physics.fire] rad_scale` UNTOUCHED |
| `src/simulation/physics_runner.py` | the engine's `emissive` now binds from `rad_scale_derived`; `self.raycaster.rad_scale` still binds from `[physics.fire] rad_scale`. The Kelvin map stays ONE map, shared. A positivity check on the new key (a non-positive scale bakes a non-increasing table, which breaks `E°^-1`'s binary lifting). The `rad_cfg` read moved 15 lines up so both `rad_scale_derived` and `k_leak` use it |
| `tests/test_emissive_table.py` | `test_live_runner_owns_one_bake_for_both_owners` -> `test_live_runner_bakes_each_owner_at_its_own_configured_scale` (D4, section 12.1) |
| `tools/fire_tuning_lab.py` | **NEW** `--reach` mode: the reach bench (section 10). The existing burn-curve mode is untouched; `main()` dispatches on the flag |
| this report | new |

**Not touched, on purpose (D6):** no `[materials.*]` row, no `[physics.fire]
rad_scale`, no `thermal_mass`, no `cool_shift`, no C++, no golden, no design
decision.

## 12. The gate

    C:/Users/steen/anaconda3/python.exe -m pytest tests -q
    2529 passed, 29 skipped, 4 xfailed, 3 warnings in 123.15 s

Identical counts to P2a's gate run. The 29 skips are the CUDA gates (no
`cpp/build_cuda` in this worktree, by instruction — no CUDA source changed).

**Gate 0** (`tests/test_radiation_sweep_reference.py`, the sweep held bit for
bit against `sweep_ref_q.py`): **34 passed**, unmoved — as it must be. P2b
changes no arithmetic: the reference drives its own `R.RAD_SCALE` and the gate
harness bakes at it explicitly (`tests/_radiation_sweep_harness.py:47`), so the
oracle is untouched by a runner-binding change. `config_dials_match()` still
passes because `[physics.fire] rad_scale` is exactly where it was.

**Goldens**: no golden file appears in the branch diff. Nothing the live fold
computes changed — the fold still reads the OLD cast's `rad_net`, and the old
cast still bakes at 5.1427e-5.

**The two scale-sensitive live tests were checked and are scale-relative by
construction**, which is why they did not move:
`tests/test_hover_readout.py:252` and
`tests/test_radiation_sweep_shadow_wiring.py:91` both derive their ambient-ring
baseline from the *live* table (`emissive.table()[0]`) rather than hardcoding a
count, so they compare `Phi` against the ring at whatever scale the runner bound.
That is the right way to have written them and it paid off here.

### 12.1 The one test rewritten, and why

`test_live_runner_owns_one_bake_for_both_owners` asserted
`eng.emissive.rad_scale == eng.raycaster.rad_scale` on the live runner. P2b makes
that **false by design**: two owners, two config keys, until P3c deletes the old
cast. It is replaced by
`test_live_runner_bakes_each_owner_at_its_own_configured_scale`, which asserts
the property that survives — *one bake implementation, each owner equal to the
reference bake at its own configured scale*:

- `raycaster.rad_scale == CFG.physics.fire.rad_scale` and
  `emissive.rad_scale == CFG.physics.radiation.rad_scale_derived`;
- both owners carry the one canonical Kelvin map;
- each table equals `sweep_ref_q.bake_e_table(rad_scale=<that owner's key>)`
  entry for entry — which keeps the original test's real property (an owner the
  runner forgot falls back to `EmissiveTable`'s header default 1.0e-5, which
  equals neither key and fails here);
- **non-vacuity**: the two tables must DIFFER, so nothing above can be satisfied
  by a runner that bound one key to both owners, and the two keys are asserted
  distinct up front so the test cannot silently go blind if they ever converge.

Per CLAUDE.md the docstring names the property and every change that must break
it. It is not weakened to a tautology and it is not deleted; it collapses back to
a single key at P3c, which the docstring says.

## 13. Open questions for Erik

Nothing below was decided here.

1. **The pin value: 0.7 or 0.9 MJ/m3/K?** The shipped derivation uses 0.7,
   which is the number the furniture row's own comment names. But 0.9 (wood at
   12 % moisture content, which is what furniture aboard a ship is) lands
   **every** material row inside 15 % of literature instead of 20-30 % light
   (section 7), and it costs one character in `config.toml`:
   `rad_scale_derived` would be **1.6533e-8**. This is a one-line ruling and it
   improves the whole table at once.
2. **`k_leak` — rule it.** It is dormant at 0, derived at ~0.10 for a 2.5 m
   deck, and section 10 measures it as **the largest single lever on reach**,
   far larger than the calibration: 1-tile furniture reach 5.75 -> 3.92 tiles,
   and the 11 kW/m2 crossing 3.15 -> 2.37. With `k_leak = 0.10` the engine's own
   ignition criterion lands almost exactly on survey 9.3's independently derived
   view-factor number. Erik set the reach question aside before P1; P2b's
   recommendation is that it should now be reopened, because reach is exactly
   what P2b was asked to calibrate and `k_leak` moves it more than anything P2b
   changed.
3. **`cool_shift` vs radiative ignition.** At the derived scale, with the shipped
   `cool_shift = 13`, a furniture tile one tile from a plateau fire settles at
   **67 game** and never ignites at any distance (section 9). The reach curve is
   real but unreachable. Something has to give: `cool_shift >= 16` on the
   furniture row, or a smaller effective heat capacity, or a surface model
   (item 4). This is the first thing P3's HUMAN-TEST will run into.
4. **Is `thermal_mass` bulk or surface-layer capacity?** They differ by ~32x
   (section 9), they want opposite values, and the same key currently sets both
   ignition timing and cool-down inertia. The surface-layer reading needs
   `thermal_mass = 0.25` on furniture, which the power-of-two table **cannot
   express** (and 0 means "gas branch"). The honest long-run answer is a two-node
   surface/bulk model for thermal solids — a design addition, not a dial.
5. **`glass` `heat_atten = 0.3` is physically wrong** for thermal IR (section 8):
   soda-lime glass is ~0.92 emissive/absorptive above 4.5 um, where most of a
   1556 K flame's power is. Today a window passes 70 % of a fire's heat. One row
   value, Erik's to change.
6. **`heat_atten` does two jobs** — emissivity (Kirchhoff) and geometric opacity
   — and the shipped `furniture = 0.5` is clearly the second (the config comment
   says so). For a filled tile they coincide; for a partly filled one they do
   not. Separating them is a scheme change, not a row change, and it would want
   ruling before P3b makes units absorb.
7. **`A_rad` assumes a 2.5 m deck and a 0.333 m tile.** Two shipped levels do
   not: `levels/airlock_demo` at `tile_size_m = 1.0` and `levels/bench_two_room`
   at 0.5. The derived scale is a single global constant, so on those levels the
   emission is off by `(tile_size/0.333)`. If that matters, `rad_scale_derived`
   wants to become a per-level derived quantity (`A_rad` from `tile_size_m` and
   `ceiling_h`, exactly as the water solver already takes its `dx` from the
   level) rather than a config literal. Recorded, not acted on.
8. **Design section 12 item 4 (the driven-source damping question) is moot at
   the derived scale** — `f == 1` everywhere on the table, for every shipped row
   (section 4). Worth knowing, because it was ruled on twice.
