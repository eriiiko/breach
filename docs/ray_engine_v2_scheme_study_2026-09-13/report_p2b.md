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

*(pending)*

## 6. The fitted/derived ratio, and what it means

*(pending)*

## 7. The implied rho*c of every material row

*(pending)*

## 8. Emissivities proposed per row (NOT applied)

*(pending)*

## 9. The lumped-capacity caveat

*(pending)*

## 10. The reach bench

*(pending)*

## 11. What changed in the tree

*(pending)*

## 12. The gate

*(pending)*

## 13. Open questions for Erik

*(pending)*
