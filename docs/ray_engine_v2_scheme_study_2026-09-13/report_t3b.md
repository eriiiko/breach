# T3b — materials authored by density

> **Written incrementally** (the arc's standing rule: a kill must cost no
> resume). Branch `12-t3b-density-materials`, cut from `fire-12` at `27347bc`.
>
> **This patch must be BIT-IDENTICAL.** It replaces an authored `thermal_mass`
> column with a derivation from each row's real density and specific heat, and
> the derivation must reproduce every shipped value exactly, on every shipped
> level.
>
> **Depends on**: `thermal_model_v2_design_2026-09-19.md` (**R14**, R5, R13,
> §5's T3b row); `report_t3.md` §1–§2 (**D1**, which proved the reproduction is
> achievable); `report_p2b.md` §1–§3 (the currency, `V_tile`, `A_rad`).

---

## 0. Status

- [x] §1 The design call: where the derivation lives, and why
- [x] §2 D1 — the `density` / `specific_heat` columns, with citations
- [x] §3 The bit-identity proof, per material and per level
- [x] §4 D2 — the property test that replaces the snapshot
- [x] §5 What did not hold / findings for Erik
- [x] §6 The gate

---

## 1. The design call: where the derivation lives

The brief names the tension exactly: **the material table is built from config
and is global, while `tile_size_m` is per level.** Three answers were on the
table - a tile-geometry argument on `MaterialTable`, `GameMap` recomputing the
per-id shifts, or something else. The answer is the third one, and it is not a
dodge: **the tile geometry cancels out of the `thermal_mass` derivation
identically, so the derivation needs none.**

### 1.1 Why it cancels

R13 does not pin the currency by naming a number of joules. It pins it by naming
a **material and a column value**: *wood at ~12 % moisture content,
`rho*c = 0.9 MJ/(m3.K)`, is `thermal_mass = 8`.* Write out what that makes any
other row worth, as a ratio of two tile capacities:

    thermal_mass_row = C_row / C_pin * THERMAL_MASS_PIN
                     = (rho_c_row * V_tile) / (rho_c_pin * V_tile) * 8
                     = rho_c_row / (rho_c_pin / 8)
                     = rho_c_row / 112 500 J/(m3.K)

`V_tile` appears once in each capacity and **cancels exactly** - both capacities
are measured on the *same* tile, whatever size that tile is. So:

> ### `thermal_mass = pow2_snap(rho * c / THERMAL_MASS_UNIT)`, and there is no tile geometry in it.

This is the same statement `report_t3.md` section 1(a) makes - *"`thermal_mass`
reads directly as `rho*c` in units of 0.1125 MJ/(m3.K)"* - and D1's row-by-row
table was computed that way. T3b is its implementation, not a reinterpretation.

### 1.2 What this buys, and what it does NOT buy

**It buys R14's stated motive.** Erik's objection was that the authored numbers
*"depend on the resolution as well"* and that he does not want to fix the
spatial resolution before he has to. Before T3b, `thermal_mass = 32` was a
number that only meant "steel" at one particular tile size, and the ten rows of
the table were ten places the resolution was quietly baked in. After T3b the
rows state `rho` and `c` - facts about matter, true at every resolution - and
the column is derived. **The resolution is out of the material table.**

**It does not buy a resolution-independent ENGINE, and this is the honest
caveat.** What does *not* cancel is the absolute worth of one heat count,

    J_per_count = rho_c_pin * V_tile / (8 * 65536)     (report_p2b.md section 2)

which scales with the tile volume. Every *source* of heat counts - the sweep's
`rad_scale`, the combustion fuel-bed deposit - is calibrated against that
number, and all of them are **global config literals today**. `report_p2b.md`
section 13 item 7 already recorded this ("the derived scale is a single global
constant, so on those levels the emission is off by `(tile_size/0.333)` ... if
that matters, `rad_scale_derived` wants to become a per-level derived quantity")
and deliberately did not act on it.

So the resolution dependence has not been removed from the engine; it has been
**concentrated**. It used to live in ten table rows *and* in the emission scale.
It now lives only in the emission scale - one derived scalar, in one place, next
to the water solver's `dx`, which already takes the level's own tile size. That
is a strictly better position to fix it from, and fixing it is T5's, not T3b's.

### 1.3 Why not scale `thermal_mass` with `V_tile` anyway

Because it would be **wrong on its own**, not merely different. Physically the
per-tick temperature rise of a wall under a given irradiance goes as
`A/V ~ 1/tile_size`: bigger tiles are thicker slabs and heat slower. In engine
terms that needs *both* halves to move - `thermal_mass` proportional to
`V_tile` **and** the emitted counts proportional to `A_rad`. Moving only the
capacity would take a 1.0 m level from "pinned at the 0.333 m reference" to
"9x too heavy against an unchanged emission", i.e. further from the physics, not
closer. Section 5 gives the numbers.

### 1.4 The code consequence

- The derivation is `simulation.materials.derive_thermal_mass` - **one function,
  the only place it is computed** - called by `MaterialTable.__init__`, by the
  property gate, and by nothing else.
- `MaterialTable` gains **no** tile-geometry argument, `GameMap` recomputes
  **nothing**, and there is still exactly one site that produces
  `heat_inv_shift` (`MaterialTable.__init__`, unchanged, now fed a derived
  column instead of an authored one).
- One transcription exists by design: `sweep_ref_q.py::_thermal_mass_ref`, in
  the integer reference, which is a standalone specification that transcribes
  every engine primitive it needs. Section 4's property gate holds the two equal
  on every shipped row, so the transcription cannot drift silently.

---

## 2. D1 - the authored columns

**`rho` and `c` are authored separately, not as one `rho_c`.** Three reasons,
in increasing order of weight:

1. They are two independent measured facts with separate literature sources, and
   a lumped product hides which one a re-authoring moved.
2. The design's own `ACCEPTED GAP: c_p constant per material (~30 % drift for
   wood over our range)` is a statement about `c` alone. It is only visible if
   `c` is visible.
3. **T5 needs the mass.** Fuel separates from `hp` and derives from combustible
   mass, `density * V_tile` kg per tile. A lumped `rho_c` column cannot be
   re-split into a density, so authoring it would have blocked the very patch
   R14 schedules next.

### 2.1 The rows, with citations

| row(s) | `density` kg/m3 | `specific_heat` J/(kg.K) | `rho*c` MJ/(m3.K) | source |
|---|---|---|---|---|
| `hull`, `steel` | 7850 | 460 | 3.611 | rho: structural steel, EN 1993-1-1 section 3.2.6 (Incropera and DeWitt Table A.1 gives 7832 for AISI 1010). c: mild/structural steel near 300 K - Incropera Table A.1 has AISI 1010 at 434 (300 K) rising to 487 (400 K); structural-steel handbooks quote 460-490 |
| `wood`, `door`, `door_closed`, `furniture`, `kindling`, `foliage` | 555 | 1620 | 0.899 | rho: *Wood Handbook* FPL-GTR-190 Table 5-3 specific gravities at 12 % MC - Douglas-fir (coast) 0.48 gives 538, loblolly pine 0.51 gives 571; 555 is mid-band. c: *Wood Handbook* ch. 4, the moist-wood formula at 12 % MC and 293 K - `c_dry = 0.1031 + 0.003867*T` = 1236 J/(kg.K), then the MC mix + `A_c` correction gives 1620 |
| `glass` | 2500 | 750 | 1.875 | Incropera and DeWitt Table A.3, *"Glass, plate (soda lime)"* at 300 K - both values from the same entry |
| `air` | 1.2047 | 717.6 | 8.645e-4 | the engine's OWN constants: rho = p/(R*T) at p = 101 325 Pa, R = 287.05, T = 293 K; c_v = R/(gamma-1) at gamma = 1.4. Their product, 864.5 J/(m3.K), reproduces `report_t2.md` section 2.1's EOS-derived 864.548 to **0.004 %** |

Two notes on the choices.

**The pin row is reproduced, not asserted.** R13 rules `rho*c = 0.90 MJ/(m3.K)`
for wood as a *unit definition*. The two independently cited numbers above
multiply to **0.899**, i.e. the ruled pin to **0.1 %**. That is the strongest
available evidence that R13's pin is a real material and not a fitted constant -
and it means a future re-authoring of `rho` or `c` inside their bands moves the
pin row by a fraction of a percent, not off its snap.

**`air` carries `c_v`, not `c_p`** - deliberately, per design v2 section 3.2's
CLOSED item: the gas field stores `N*T_abs` (constant volume) and the EOS
charges the expansion work separately through `k_work`, so `c_p` in the capacity
would double-count the same physics.

### 2.2 The gas row keeps `thermal_mass = 0`, and that is not an exception

`air` is the one row that still authors `thermal_mass`, and the value is the
literal `0`. This is **not** a capacity escaping the derivation: air's real
`rho*c_v` is `0.0077` column units - **seven doublings below the column's floor
of 1** - because a gas tile's capacity is not a smaller number in this column
but a *different representation*, `N * c_v` on the gas field. The `0` therefore
**declares a thermal regime**, which is exactly what the thermal-mass axis
design said it meant when it made `0` legal. Every other authored
`thermal_mass` is now rejected by name.

The door proves itself: removing air's declaration makes the loader raise
*"rho*c / 112500 = 0.00768... snaps BELOW 1"*, naming the row.

### 2.3 The snap, and how far it is from flipping

`pow2_snap` takes the nearest power of two **in log space** (R5 / T3 section
2.1's cost model), computed **without a logarithm**: the geometric midpoint
between `2^k` and `2^(k+1)` is `2^k * sqrt(2)`, so the snap is a
bracket-and-compare over exact binary scalings plus one correctly-rounded
`sqrt`. That keeps it inside the number-ingress doors with **no exemption** -
unlike `_build_conduction_tables`, whose `math.log2` carries one with a standing
TODO to remove it. (The same technique would close that TODO; out of scope.)

The reproduction is not knife-edge. A row lands on today's shift for any
`rho*c` inside:

| shift | column | `rho*c` band that snaps here, MJ/(m3.K) | the row's literature band | headroom |
|---|---|---|---|---|
| 5 | 32 | 2.546 - 5.091 | steel 3.40 - 3.85 | 1.34x below, 1.32x above |
| 4 | 16 | 1.273 - 2.546 | glass 1.88 - 2.10 | 1.47x below, 1.21x above |
| 3 | 8 | 0.636 - 1.273 | wood 0.80 - 0.95 | 1.26x below, 1.34x above |

Every shipped row's *whole literature band* snaps to the shipped value. The
reproduction survives re-authoring anywhere inside the sources, which is what
makes it a derivation rather than a coincidence.

---

## 3. Bit-identity - proved, not assumed

Measured by `scratchpad/gate_t3b.py` (regenerable, untracked). Its "before" side
is read **out of git** (`git show abcacdf:config.toml`, the report-skeleton
commit, i.e. the tree as it stood before any T3b code), never typed from memory.

### 3.1 No column but `thermal_mass` moved

Every `[materials.*]` key other than `thermal_mass` / `density` /
`specific_heat` compared old-vs-new, all ten rows: **NONE moved.** `hp`, fuel,
`conductivity`, `heat_atten`, `ignition_temp`, `cool_shift`, `permeability`,
`wave_absorb`, `blast_resist`, `burst_threshold`, `cover_exposure`,
`light_atten`, `mobility`, `flammable` - untouched, as D3 requires.

### 3.2 Per material

| id | row | `rho*c` MJ/(m3.K) | exact | **derived** | authored (before) | shift | was | snap err |
|---|---|---|---|---|---|---|---|---|
| 0 | `air` | 8.645e-4 | 0.0077 | **0** (gas regime) | 0 | 0 | 0 | - |
| 1 | `hull` | 3.6110 | 32.0978 | **32** | 32 | 5 | 5 | -0.30 % |
| 2 | `wood` | 0.8991 | 7.9920 | **8** | 8 | 3 | 3 | +0.10 % |
| 3 | `door` | 0.8991 | 7.9920 | **8** | 8 | 3 | 3 | +0.10 % |
| 4 | `steel` | 3.6110 | 32.0978 | **32** | 32 | 5 | 5 | -0.30 % |
| 5 | `glass` | 1.8750 | 16.6667 | **16** | 16 | 4 | 4 | -4.00 % |
| 6 | `furniture` | 0.8991 | 7.9920 | **8** | 8 | 3 | 3 | +0.10 % |
| 7 | `door_closed` | 0.8991 | 7.9920 | **8** | 8 | 3 | 3 | +0.10 % |
| 8 | `kindling` | 0.8991 | 7.9920 | **8** | 8 | 3 | 3 | +0.10 % |
| 9 | `foliage` | 0.8991 | 7.9920 | **8** | 8 | 3 | 3 | +0.10 % |

**Every row reproduces**, and the worst snap error in the table is glass at
-4.0 % - half of T3 D1's -7.7 % (T3 used `rho*c = 1.95` for glass; Incropera's
own plate-glass entry is 1.875), and a fifth of R5's worst case.

### 3.3 Per level, per tile

Not the columns - the actual `GameMap` planes, for every shipped level, against
`old_shift[material_grid]` rebuilt from the pre-T3b config:

| level | `tile_size_m` | tiles | material ids present | `heat_inv_shift` | `thermal_solid` |
|---|---|---|---|---|---|
| `airlock_demo` | **1.000** | 180 | 0, 1, 7 | **IDENTICAL** | IDENTICAL |
| `aquarium_demo` | 0.333 | 384 | 0, 1, 5 | IDENTICAL | IDENTICAL |
| `bake_demo` | 0.333 | 384 | 0, 1, 2, 3, 5, 6 | IDENTICAL | IDENTICAL |
| `bench_two_room` | **0.500** | 378 | 0, 1, 6 | **IDENTICAL** | IDENTICAL |
| `door_test` | 0.333 | 1200 | 0, 1, 7 | IDENTICAL | IDENTICAL |
| `fire_studio` | 0.333 | 1536 | 0, 1, 2, 6, 7 | IDENTICAL | IDENTICAL |
| `fire_tuning` | 0.333 | 3312 | 0, 1, 2, 6, 7, 8 | IDENTICAL | IDENTICAL |
| `my_ship` | 0.333 | 1536 | 0, 1, 4, 5, 6 | IDENTICAL | IDENTICAL |
| `planetside_demo` | 0.333 | 2128 | 0, 1, 2 | IDENTICAL | IDENTICAL |
| `playground` | 0.333 | 7000 | 0, 1, 2, 4, 5, 6, 7 | IDENTICAL | IDENTICAL |
| `test_level` | 0.333 | 32768 | 0, 1, 2, 4, 5, 6, 7 | IDENTICAL | IDENTICAL |
| `unhcr_vessel` | 0.333 | 6000 | 0, 1, 3 | IDENTICAL | IDENTICAL |
| `wego_test` | 0.333 | 6144 | 0, 1, 4, 7 | IDENTICAL | IDENTICAL |

**All 13 shipped levels, all 62 946 tiles, both planes: identical.** The two
levels the brief singled out as the honest test - `airlock_demo` at 1.0 m and
`bench_two_room` at 0.5 m - reproduce like the rest, *because* the tile volume
cancels. Under the derivation actually implemented there is no per-level
divergence to report. Section 5 records what the other reading would have done.

## 4. D2 - the property test that replaces the snapshot

`tests/test_thermal_mass_axis.py` pinned `hull/steel 32, glass 16, wood 8` as
literals. By the project's own standard that is a snapshot: it held the answers
without holding the reason, it listed only the rows that existed on the day it
was written (`kindling` and `foliage` were never in it), and it would have
survived the column going back to a hand-tuned list.

Six gates replace it. Each names its property and the change that must break it.

| gate | property | breaks if |
|---|---|---|
| `test_thermal_mass_is_derived_from_the_rows_real_rho_c` | **EVERY** row's `thermal_mass` equals `pow2_snap(density * specific_heat / THERMAL_MASS_UNIT)`, and every row carries a real `rho` and `c` | a row is hand-authored again; the snap stops being log-space nearest; the R13 pin moves without the rows; a tile-geometry factor is re-introduced (which would make the column per-level) |
| `test_the_derivation_reproduces_every_shipped_column_value` | the derivation lands the exact column the table shipped with, all ten rows | a row is re-authored to a `rho*c` outside its snap band - a deliberate, feel-affecting change that must not happen by accident |
| `test_the_integer_reference_derives_the_same_column` | `sweep_ref_q.py`'s transcription agrees with the engine on every shipped absorbing row, unit included | either side's snap or unit moves without the other's |
| `test_loader_accepts_the_gas_regime_declaration` | `thermal_mass = 0` is still legal and still means the gas regime | the declaration is dropped, sending air down the bit-shift convert path it must never reach |
| `test_loader_rejects_an_authored_thermal_mass_by_name` | a non-zero authored `thermal_mass` raises, naming the row | the door goes back to accepting an authored capacity - a second source of truth for `heat_inv_shift` |
| `test_the_snap_is_log_space_nearest_not_linear` | the 8/16 boundary sits at `8*sqrt(2) = 11.31`, not at 12 | `pow2_snap` is "simplified" to a linear round |
| `test_a_row_lighter_than_the_column_floor_is_a_named_refusal` | a `rho*c` snapping below 1 raises, naming the row | the floor starts silently clamping |

**Why the first two are both needed, and are not the same test.** The property
gate is self-consistent: move the pin and every row moves with it, and it still
passes. The reproduction gate is the bit-identity claim of *this patch*, and it
is the one that fails when the pin moves. Splitting them means a future,
deliberate re-pinning has exactly one red test to answer for, and it says which
one it is.

**Validated by reverting, not by passing** (the arc's standing habit):

- glass `density` 2500 to 3600 (`rho*c` 2.70, snaps to 32): **only** the
  reproduction gate goes red, naming `{'glass': 32} != {'glass': 16}`.
- `pow2_snap` perturbed to return one exponent too high: **16 gates go red**,
  including the reference-agreement gate.

`test_per_tile_shift_matches_log2_thermal_mass` and the other 41 tests in the
file are untouched: they were already properties.

---

## 5. Findings

### 5.1 The counterfactual, since the brief asked for it

The brief flagged `airlock_demo` (1.0 m) and `bench_two_room` (0.5 m) as the
cases that expose whether the derivation is honest. Under the derivation
implemented they reproduce exactly, because the tile volume cancels (§1.1).
Here is what the *other* reading - `thermal_mass` scaled by `V_tile / V_ref`,
i.e. treating one heat count as a fixed number of joules rather than a fixed
fraction of the pin material's tile - would have produced:

| `tile_size_m` | levels | `steel` | `glass` | `wood` | shift move |
|---|---|---|---|---|---|
| 0.333 | the other eleven | 32 | 16 | 8 | none |
| 0.500 | `bench_two_room` | 64 | 32 | 16 | **+1** |
| 1.000 | `airlock_demo` | 256 | 128 | 64 | **+3** |

That reading is **not** the more honest one, and it is worth being explicit
about why, because "scale it with the tile" sounds more physical than it is.
Physically the per-tick temperature rise of a wall under a given irradiance goes
as `A/V ~ 1/tile_size`. Reproducing that in the engine needs *both* halves to
move: `thermal_mass` up with `V_tile` **and** the emitted counts up with
`A_rad`. Moving only the capacity would take `airlock_demo` from "pinned at the
0.333 m reference" - which is what it is today - to "9x too heavy against an
unchanged emission". That is further from the physics, not closer, and it would
have cost the bit-identity gate for nothing.

### 5.2 What is still resolution-dependent, and whose job it is

`J_per_count = rho_c_pin * V_tile / (8 * 65536)` still scales with the tile
volume, and every heat *source* is calibrated against it as a global config
literal. `report_p2b.md` §13 item 7 recorded this before T3b existed. T3b does
not fix it and does not make it worse; it **concentrates** it, from ten table
rows plus the emission scale down to the emission scale alone.

**For T5 / Erik**: the fix is the one P2b already named - derive
`rad_scale` per level from the loaded `tile_size_m` and `ceiling_h`, the way
`WaterSolver.dx` already takes its `dx` from the level. That is a single scalar
in a single place. It is a feel-affecting change on the two off-resolution
levels, so it is not T3b's to make.

### 5.3 The gas row now carries the derivation of the corrected `c_v`

`[physics.thermal] c_v = 1.0` is still the placeholder its own config comment
says it is, and `report_t2.md` measured it 130.13x too heavy. The `air` row now
states the real numbers with their source, and **the same R14 formula that
gives every solid its `thermal_mass` gives the gas row its `c_v`, un-snapped**:

    c_v_column = density * specific_heat / THERMAL_MASS_UNIT
               = 1.2047 * 717.6 / 112 500
               = 0.0076849                      == report_t2.md §2.2 exactly

So T5's ledger-item-4 correction now has its citation site in the table rather
than as a loose number in a report. T3b does **not** apply it - changing `c_v`
moves gas behaviour everywhere and is not behaviour-neutral.

### 5.4 Glass improved without the column moving

T3 D1 used `rho*c = 1.95` for glass (a band midpoint) and reported a **-7.7 %**
snap. Incropera & DeWitt Table A.3's own *"Glass, plate (soda lime)"* entry
gives `rho = 2500` and `c = 750` from one line, i.e. **1.875** and a **-4.0 %**
snap. Same column value, better sourcing, half the error. Nothing behavioural.

### 5.5 T3 D1's open questions are now enforceable, not decided

T3 D1 flagged two rows where the OBJECT reading differs from the MATERIAL
reading - `kindling` (a stick pile is 15-25 % solid wood, wants ~2) and
`foliage` (a canopy is 2-5 % leaf tissue, wants ~0.5, **which the column cannot
express**: its floor is 1 and 0 is taken). **T3b decides neither** - both rows
are authored at the MATERIAL reading, exactly as they shipped, and the `foliage`
row says so in a comment pointing at T3's open question 2.

What changed is that the wall is now **loud**: a row whose `rho*c` snaps below 1
raises, naming the row, instead of being silently rounded to 1. `foliage`'s
object reading would hit it. Erik's call, unchanged and now un-ignorable.

### 5.6 Collateral, all mechanical, all reported

| file | what | why it was unavoidable |
|---|---|---|
| `docs/.../sweep_ref_q.py` | `_thermal_mass_ref`, a transcription of the derivation | the integer reference reads `config.toml` itself and looked up a key that no longer exists; it is a standalone spec that transcribes every engine primitive it uses, and §4's gate now holds the two equal |
| `tools/fire_tuning_lab.py` | `_mat` reads the built `MaterialTable` instead of the raw CFG row; the commented `REACH_DIALS` example becomes `density` / `specific_heat` | it read `row.thermal_mass` off the config row - which is arguably a pre-existing bug (the tool should read the one source of truth, not the raw row) that R14 merely exposed |
| `tests/test_cool_shift_axis.py` | the synthetic `_row` states `rho*c` | a dict-built row must supply what the table now requires |
| `tests/test_fuel_fraction_axis.py` | two duplicated row dicts folded into one `_rows_from_table()` helper that also carries the gas-regime declaration | same, plus the reconstructed air row needed its `thermal_mass = 0` back |

### 5.7 Nothing on the device reads `thermal_mass`

C++ and CUDA read only the int32 **`heat_inv_shift`** plane (and the
`thermal_solid` mask), never the column. Both planes are byte-identical (§3.3),
so no device change was required and no `/fp:strict` TU was added. The CUDA
build was made and the suite re-run with it present anyway, per the gate.

### 5.8 A free follow-up, deliberately not taken

`_build_conduction_tables` carries a standing `ingress-exempt: ... TODO(stats-
redesign): replace with an integer log2` on its two `math.log2` calls.
`pow2_snap`'s bracket-and-compare is exactly that replacement technique, proven
here. Closing that TODO would move conduction shifts if any of them sit near a
bucket boundary, so it is **not** behaviour-neutral and has no business in a
bit-identical patch. Recorded for whoever picks up the ratchet.

### 5.9 Nothing failed to reproduce

There is no row, and no level, where the derivation disagrees with what shipped.
The one thing that could not be expressed - air's `rho*c_v`, seven doublings
below the column's floor - is not a failure of the derivation but the reason the
gas-regime declaration exists, and it is now stated in the row rather than
implied by a bare `0`.

---

## 6. The gate

| # | requirement | result |
|---|---|---|
| 1 | derived `heat_inv_shift` == today's, every material, **every shipped level** | **PASS** - 10/10 rows, 13/13 levels, 62 946 tiles, `heat_inv_shift` and `thermal_solid` both identical (§3) |
| 2 | goldens unmoved, `DIGEST_SPEC_VERSION` unchanged | **PASS** - no golden, digest or spec file is touched by this branch (`git diff 27347bc --name-only`) |
| 3 | full suite green, CPU **and** with the CUDA build present | **PASS** - CPU: **2631 passed, 0 failed**, 29 skipped, 4 xfailed. With `cpp/build_cuda` present: **2656 passed, 0 failed**, 4 skipped, 4 xfailed (25 CUDA gates ran) |
| 4 | `git status` clean, never `git add -A` | **PASS** - every commit staged by explicit path |
| 5 | commits `feat(#12): T3b -- <what>`, co-authored | **PASS** |
| 6 | no merge, no push, no worktree deletion | **PASS** |

Also held, from D3's "what you must NOT do": `hp`, fuel, `fuel_recip` and burn
behaviour untouched; `conductivity`, `heat_atten`, `ignition_temp` and every
other row value untouched (§3.1 compares them key-by-key against the pre-T3b
config); the sweep, `cool_shift`, `c_v` and the temperature solver's laws
untouched.

### 6.1 Instruments

- `scratchpad/gate_t3b.py` - the bit-identity gate: reads the pre-T3b config out
  of git, compares every row and every level's per-tile planes, and prints the
  §5.1 counterfactual. Regenerable, untracked, asserts nothing it does not
  print.
- The permanent gates live in `tests/test_thermal_mass_axis.py` (§4).

### 6.2 What T5 inherits

A material table whose rows state matter, not tile arithmetic. `density` is in
place and cited for every row, which is the input T5's fuel-mass separation
needs (`density * V_tile` kg per tile, against `hp` staying structural). The
`c_v` correction has its citation site (§5.3). And the last place the spatial
resolution is still baked in is one derived scalar, `rad_scale`, named in
`report_p2b.md` §13 item 7 - not ten rows of a table.
