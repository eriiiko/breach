# report_m2.md — M2: dimensions in, fill derived; the four thin rows

> Implementer's report, written by the M2 agent; committed by the orchestrator
> because the session's write policy blocked the agent from creating it.
> **Design of record**: `docs/thin_material_rows_design_2026-09-20.md`.
> Branch `12-m2-dimensions-and-thin-rows`, commits `16c69a6`, `d06cfd8`.
> Merged to `12-t5b-the-flip` as `9cc1879`.

---

## 1. What landed

**The column (§4).** A row keeps its real, cited `density`/`specific_heat` and
gains `thickness_m`; `fill_fraction`, `mass_kg` and `thermal_mass` are derived at
load and none may be authored.

Omitting `thickness_m` declares a **SOLID** row (fill 1.0) — that is every
structural row, it is what they already were, and it is the one fill value that is
scale-free, so every structural row is **bit-identical across M2**.

The panel derivation collapses to a pure length ratio:

```
fill = (thickness_m * tile_w * ceiling_h) / (tile_w^2 * ceiling_h)
     = thickness_m / tile_w
```

`ceiling_h` and one factor of `tile_w` cancel, **and that cancellation IS the
tile-size invariance** — `dT/dt` reduces to `flux / (rho * c * thickness)` with no
geometry left in it. `thermal_mass` stays R14's ratio of two capacities on the same
tile, because `fill_fraction` is dimensionless.

**The pin (§7)** is now anchored on named `REFERENCE_SUBSTANCE` constants (solid
softwood at 12 % MC, ρ = 555, c = 1620, Wood Handbook), with `RHO_C_PIN = 0.9e6`
stated as the literal ruled value and `PIN_ANCHOR_TOLERANCE` asserting the two
agree (0.10 % apart). The pin cites a substance, so *"no shipped row is that
substance at full fill"* reads as obviously fine.

**The validators (§11)**, at the materials door: derived columns refused by name;
a `flammable` row must author `thickness_m` and satisfy
`thickness_m <= THIN_LIMIT_M` (0.006) or be refused by name; a fill outside (0, 1]
is a load-time error, never a clamp. `furniture` is the one exemption, in
`LUMPED_CRITERION_EXEMPT` with its reason written out `ingress-exempt`-style, plus
a `lumped-exempt:` line on the config row.

**The bench (§8).** `tools/fire_timing_harness.py` takes `--mat`, defaulting to
`kindling` — the campfire reference object. Kindling rather than `wood` because it
keeps the two properties that made the old crate a clean measurement
(`conductivity = 0.0`, `ignition_temp = 280`), so the **only** thing that changed
from the historical scenario is the fuel's mass. Its ignition seed now reads the
row's own `ignition_temp` instead of furniture's hardcoded `280.0`. Four
neighbouring instruments that share `build_level` now pass `fuel_mat=FURN`
explicitly, so none of their calibrations moved by inheriting a changed default.

**The integer reference went first**, per the CLAUDE.md rule: `_fill_fraction_ref`
added, `_thermal_mass_exp_ref` routed through it, gates re-run (all PASS) before
any engine code.

### One deviation from the design's letter, taken deliberately

§4's rule is *"derive at the base tile size, then divide across the N² runtime
tiles"*, with an added clause *"and correct across levels"*. The base-resolution
and `res_factor` halves are implemented exactly; the table was **not** made
per-level — it stays pinned to `tile_size_ref_m` and takes only the level's
`res_factor`.

Reason: `tile_size_ref_m` is the reference the *whole* thermal denomination is
already built at — the conduction table, `rad_scale`, and `J_per_count`.
`materials.py`'s own R14 block says making it per-level is "report_p2b §13 item 7
— T5's business". Making only the material table per-level would put the two
halves of the same physics on different rulers: a *new* inconsistency, and exactly
the "half-solve q3 against a requirement that does not exist yet" that §4's own
reversal forbids. §11.3b is therefore gated where the design states it — as a
property of the **derivation**, not the live engine. Erik's resolution ruling
(*"keeping the levels correct is not the main priority just now"*) is what makes
this safe. **q3 is untouched.**

---

## 2. The four rows — independent derivation, and whether it agreed with §6

`V_tile = 0.333² × 2.5 = 0.2772225 m³`; a solid tile of the reference substance is
153.858 kg.

| row | `thickness_m` | fill % | kg | raw units | snap | capacity | t_ign @17.7 kW | t_spread @12 kW |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `wood` | 0.0050 | 1.502 | **2.31** | 0.12000 | 2⁻³ (+4.2 %) | 3.90 kJ/K | **66.1 s** | 97.5 s |
| `kindling` | 0.0043 | 1.291 | **1.99** | 0.10320 | 2⁻³ (+21 %) | 3.90 kJ/K | **66.1 s** | 97.5 s |
| `foliage` | 0.0043 | 1.291 | **1.99** | 0.10320 | 2⁻³ | 3.90 kJ/K | **66.1 s** | 97.5 s |
| `furniture` | 0.0108 | 3.243 | **4.99** | 0.25957 | 2⁻² (−3.7 %) | 7.80 kJ/K | **132.2 s** | 194.9 s |

**It agreed with §6 on every column and all four exponents.** The only differences
are in the third significant figure and have one cause: §6 quotes the round-kg
*target* (2.0, 5.0) while also stating that `mm equiv` **is the authored number**.
The two-significant-figure thicknesses §6 names (4.3 mm, 10.8 mm) were authored,
landing 1.99 kg and 4.99 kg. **Authoring `0.0043286` to hit 2.000 kg exactly would
be fitting a dimension to a mass — the inversion this design exists to remove.**

Fill 1.50/1.29/1.29/3.24 % vs §6's 1.50/1.30/1.30/3.25; raw units
0.1200/0.1032/0.1032/0.2596 vs 0.120/0.104/0.104/0.260; identical snaps. The 66 s
lands inside the 30–70 s literature band with nothing tuned.

**The fuel store falls out for free as §8 predicted**: `wood` holds **6.22** O₂
units against a full tile's 414 — the same 60 hp bar over a **67× stronger physical
channel**. `fuel_per_o2` moved 0.1449 → 9.6516 (wood), 0.0725 → 2.2342
(furniture), 0.0193 → 1.4964 (kindling).

---

## 3. Gate results — exact counts

Both binaries rebuilt green. **M2 touches no C++** — `git diff cbb49f3..HEAD --
cpp/` is empty.

| | passed | failed | skipped | xfailed |
|---|---:|---:|---:|---:|
| **baseline** (`cbb49f3`) | 3390 | **7** | 4 | 4 |
| **M2 final** | **3408** | **11** | 4 | 4 |

*(Orchestrator re-ran the suite independently: 3408 / 11, matching, with the four
new reds exactly as named.)*

CUDA: `-k cuda` → **25 passed, 0 skipped** (82 s of real GPU subprocess runs).
Integer-reference gates: ALL GATES PASS on the reference's own run.

The 7 pre-existing T5b reds are unchanged and unbent. **4 new reds, every one a
finding, none bent**: G12/Fleck (§5.1), `test_fire_feedback::test_conservative_
default_does_not_firestorm_wood_room` and `test_eos_p4_combustion::test_e2e_2`
(both confirmed green under M3's derived `H_bed`), and
`test_e1_hot_rail::test_no_rail_hits` (§5.3).

---

## 4. Golden re-baseline — there isn't one, and the reason is itself a finding

`GOLDEN_AGGREGATE` is unmoved at `e369b616bb…`; all 14 golden-bound tests pass
unedited. No digest spec bump (`heat_inv_shift`/`fuel_per_o2` are not digest
fields; no field added or retyped).

**But not for the reassuring reason.** The canonical A/B scenario contains only
`hull` and `air` — `field_ab_harness._scenario_level` is an all-hull 16×16 with the
interior carved to air. It paints **no flammable tile at all**. Measured by
perturbation probe, nothing committed:

| perturbation | `GOLDEN_AGGREGATE` |
|---|---|
| baseline | `e369b616…` |
| `hull` authored as a 2 cm panel (a row the scene *does* contain) | `b8f1bb3b…` — **MOVED** |
| `wood` 5 mm → 3 mm | `e369b616…` — **unmoved** |
| all four flammable rows thinned 3–6× at once | `e369b616…` — **unmoved** |

So the golden **is** sensitive to the material table and **is structurally blind to
every row M2 authors**. Its staying put is not evidence M2 is correct.

**The independent oracles used instead:**

1. the row's own cited constants recomputed in the test — deliberately in the long
   form (object volume ÷ tile volume) rather than the engine's collapsed shortcut,
   so the two are not the same expression twice;
2. the committed integer reference, changed first, gates re-run, engine held to it
   on every shipped row;
3. closed-form identities (`dT/dt == flux/(ρ·c·thickness)`; `total mass ×
   res_factor²` constant);
4. the pre-existing conduction tests, which caught the one real bug this patch
   introduced.

---

## 5. Findings

### 5.1 — G12: the Fleck damping quantizes on thin rows, and two rulings collide

**OPEN. BLOCKS M3. Erik's ruling is owed.**

`gate12_damped_source_is_monotone` fails on the three thin rows. Its own docstring
predicted exactly this and names the remedy P0 proposed — an ingress rule
`heat_atten > 0 ⇒ thermal_mass >= 8` — **which would forbid every row this design
authors**. Left red and measured:

| row | `his` | backward steps | worst step | first at |
|---|---:|---:|---:|---:|
| hull/steel, door/door_closed, glass | +5/+3/+4 | 0 | — | — |
| `wood`/`foliage` | −3 | 542 | **0.479 %** | 7648 game |
| `kindling` | −3 | 517 | 0.223 % | 8904 game |
| `furniture` | −2 | 365 | 0.063 % | 10632 game |

Below 6000 game every shipped row is still exactly monotone; the first backward
step is at ≈7940 K (deep plasma), and 0.479 % in power is *inside* the 0.5 % bound
the same gate applies to its pathological row. Widening `f` fixes it completely
(`F_SHIFT` 28 → 0/0/15 steps; 32 → 0/0/0) but G11's headroom already measures
`max(ex_m·f_q24) = 2^61.72` against `< 2^63`, so 28 overflows and 32 badly —
restructuring that product is a P-series patch, nowhere near M2's scope.

**The adjacent observation may matter more.** `L ∝ 1/capacity`, so the damping now
bites *inside the fire range*:

| | Fleck factor `f` |
|---|---:|
| solid `door` at 1200 game | 0.848 |
| `wood` at its own ignition point | **0.223** |
| `wood` at 1200 game | **0.013** |

A 1200-game wood panel radiates **1.3 %** of its excess black-body power. Whether
that is right depends on the emission scale's calibration, which T5b showed is
entangled with the 2¹⁶ error M3 corrects. **Correcting `H_bed` without asking what
`L` means will leave this standing.**

*(Orchestrator's note: CLAUDE.md's ruled invariant states `f == 2²⁴` — full black
body — wherever the explicit update is provably monotone, and gives that threshold
as **1068 game for wood**. That was computed at `thermal_mass = 8`. A thin panel is
0.125, 64× lighter, so the threshold falls 64× to **~17 game**. The invariant's
guarantee is now false for every flammable row in the game. The risk is a NEW
compensating pair: suppressed emission keeps a burning tile hotter, partly
cancelling M3's `H_bed` cut — in the direction that makes fire look like it works
while radiative SPREAD stays suppressed 77×.)*

### 5.2 — M1's list was incomplete, and so was M1's own repair

An eighth `heat_inv_shift` consumer: the integer reference's `fold_pass1_solid`
computed its books capacity as `1 << s`, which does not exist for `s < 0` (Python
raises) and was a *different normalisation* from the engine's `1 << (s + 16)`.
Nothing consumes that sum, so the divergence was invisible. Fixed to the engine's
form. The lesson is narrower than M1's: **"already routed through the kit" is not
"every expression on this path handles the sign"**.

### 5.3 — `test_e1_hot_rail::test_no_rail_hits`

A 6×6 *furniture* block, 2000 ticks, with a "2× measured headroom" budget
calibrated when that block was 153.9 kg/tile. The load-bearing half still passes
(`t_max_phys_hits <= 8`, so the rail is still a counted bounded transient, not the
2130-hit runaway it was written to catch); the softer second bound fails, 32
ceiling ticks against 14. **Not re-baselined** — re-fitting a measured-headroom
budget in a patch that is not the one settling the fire energy is exactly the
failure the standing requirement names. **Under M3's derived `H_bed` it is still
27**, so this one is not simply the `H_bed` error.

### 5.4 — No golden in this suite can see a fire-material change

See §4. Every fire-material patch on this arc has been landing against a golden
that cannot respond to it. Worth either a scenario carrying a fuel tile, or an
explicit note beside `GOLDEN_AGGREGATE`.

### 5.5 — The conduction table is a consumer the design does not name, and it was a real bug

`_build_conduction_tables` priced a face with `rho_c_min` from `thermal_mass`
(which M2 made carry the fill) while `kappa` came from the authored *material*
conductivity. That asymmetry moved `wood|wood` from face shift 24 to **18** — heat
spreading along a wooden wall **64× faster than wood does**, from a patch whose
design says nothing about conduction.

It is wrong because in-plane diffusivity is a property of the matter: conductance
per tile pair carries `thickness·ceiling_h` and the tile capacity carries
`thickness·ceiling_h·dx`, so the fill cancels. Both sides are tile-averaged now;
the face table is **identical to pre-M2 in 96 of 100 entries**, the four movers
being `wood↔door` / `wood↔door_closed` 24 → 23 (a real consequence — a thin panel
against a solid slab is genuinely the faster face).

**Found by looking, not by a gate** — §12's consumer list does not mention
conduction. Two *pre-existing* tests went red on the asymmetry and green on the
fix, independent confirmation the fix is right and not an invention. Two
hand-written Python mirrors of that law had to be updated in lockstep — M1's
finding 3 again.

### 5.6 — The flamethrower's distance discrimination is no longer measurable

Distances 2, 3 and 8 all ignite on **tick 0** (measured). That is the design
working, but it makes the old test's real content unmeasurable at tick resolution;
restoring it needs a sub-tick observable, not a tick count. The test now asserts
the half that is still about the weapon: everything inside `range_m` ignites,
nothing beyond it does (probed at 11 and 13 tiles — never ignite).

### 5.7 — `foliage` carries `wood`'s `hp` on `kindling`'s mass

Derived `fuel_per_o2` = 11.22, the table's highest, because `hp = 60` now sits on a
1.99 kg lump. Not M2's to rule on, but the two numbers were chosen under different
premises and should be re-read together when M3 measures burn durations.

### 5.8 — A stale finding marked, not acted on

`fire_tuning_lab.py`'s 2026-09-06 wood-bonfire finding was measured against the
153.9 kg block; the conduction drain is now 64× smaller relative to the heat
arriving. Marked STALE in place with an instruction not to act on it without
re-measuring.

---

## 6. Every gate validated by breaking it

Eleven deliberate breaks; each applied, run against the full M2-relevant selection,
observed red, reverted. Baseline red set in that selection: **none**, and it
matched again after the last revert.

| # | the break | gates that caught it |
|---|---|---|
| **B1** | `fill` dropped from the `thermal_mass` derivation | 11 tests, incl. all four row derivations, the thin-vs-solid property, the integer-reference cross-check, `test_shipped_shifts` |
| **B2** | mass derived at the **live** tile size (`--res` division dropped) | **only** the two §11.3c gates — nothing else in 3408 tests sees it |
| **B3** | `fill` authored as a constant instead of `thickness/tile_w` | `test_a_panel_rows_heating_rate_is_identical_at_every_tile_size` (§11.3b) + two row derivations |
| **B4** | lumped criterion becomes a warning | `test_a_flammable_row_thicker_than_the_lumped_limit_is_refused_by_name` |
| **B5** | a flammable row need not author thickness | `test_a_flammable_row_must_author_its_thickness` |
| **B6** | fill outside (0,1] **clamped** instead of refused | `test_a_row_that_overflows_its_tile_is_a_load_time_error_not_a_clamp` |
| **B7** | the pin re-derived **off a row** | 13 tests, led by the pin gate — the blast radius *is* the hazard §7 describes, made visible |
| **B8** | fill on the capacity but **not** the conductance (the real bug) | the new conduction property + **two pre-existing** conduction tests + the EOS-P2 mirror |
| **B9** | an authored `fill_fraction` becomes an override | `test_a_derived_column_may_not_be_authored` ×3 |
| **B10** | fuel mass back to a full tile | the M2 fuel-store gate + `test_fuel_store_is_the_tile_s_real_combustible_mass` |
| **B11** | `GameMap` stops threading the level | **only** `test_the_engine_actually_threads_res_factor_into_the_material_table` |

**B2 and B11 are the ones worth reading twice** — each caught by a single gate and
nothing else, the exact shape M1 found three times.

A twelfth break happened by accident and is worth recording: the golden probe tried
to author `wood` at 15 mm and was **refused by the lumped door through the real
level-load path**, not a test fixture. Two further non-vacuity probes live inside
the tests (an authored fill is asserted to be 3× wrong at 1.0 m; a 10× less
conductive wood must conduct more slowly).

---

## 7. The re-anchored bench

`--mat {kindling|wood|furniture|foliage}`, default `kindling`. On this build (still
the uncorrected `H_bed`): kindling 1.99 kg, peak I 0.435 @ 3.5 s, **steady T railed
at 16000**, burnout 6.8 s; wood 2.31 kg, 0.408 @ 5.8 s, railed; furniture 4.99 kg,
0.591 @ 4.4 s, railed, burnout 27.8 s. All three rail — the T5b runaway on a 30–60×
lighter object. Useful, not alarming: the bench is a far sharper instrument for M3
than the crate, where the same error merely warmed the tile.

**Reference numbers re-quoted on a thin row:**

| quantity | old anchor (154 kg crate) | `kindling` | `wood` panel | `furniture` |
|---|---:|---:|---:|---:|
| capacity | 249.5 kJ/K | **3.90 kJ/K** | 3.90 | 7.80 |
| self-heat 17.7 kW | **0.07 K/s** | **4.54 K/s** | 4.54 | 2.27 |
| → 300 K rise | ~71 min | **66 s** | 66 s | 132 s |
| cross-gap 12 kW | 0.05 K/s | 3.08 K/s | 3.08 | 1.54 |
| → 300 K rise | ~104 min | **98 s** | 98 s | 195 s |
| O₂ store | 414 units | 5.4 | 6.2 | 13.4 |

**The 0.07 K/s that forced Erik's ruling becomes 4.54 K/s** — report_t5b §6.3's
"a 17.7 kW fire cannot heat this tile" disappears, measured rather than asserted.

---

## 8. Tests

**New — `tests/test_m2_dimensions_and_thin_rows.py` (23 tests).** Properties
pinned: the unit scale is a definition and is never computed from a row; each row's
mass and capacity are what its authored geometry implies (×4, recomputed from cited
constants); the rows are objects a fire can heat, 20–120× lighter than the block;
`fill_fraction`/`mass`/`mass_kg` refused by name (×3); R14's `thermal_mass` refusal
survives M2; a flammable row must state its geometry; the small-Biot criterion is a
DOOR, inclusive at the limit; `furniture` is the one exemption with a written
reason; `thickness > 0`, fill in (0,1] on every shipped tile size; omission declares
a solid row and structural rows are unmoved; §11.3b `dT/dt` identical at
0.333/0.5/1.0/2.0 m; a non-vacuity probe asserting an authored fill is exactly 3×
wrong at 1.0 m; §11.3c total mass identical at `--res` 1/2/3, and the same
end-to-end through `_upscale_level` → `GameMap` → the table on the real grid; a
row's face shift is its matter's diffusivity, not its thickness; the O₂ store is the
row's real combustible mass; kindling is inside the 1–3 kg band its own header
declares.

**Rewritten** (premise overturned by the design, not bent to it): the R14
derivation gate gains the fill term and drops `int(round(tm))` (which rounds 0.125
to **0**, the gas sentinel); `test_the_derivation_reproduces_every_shipped_column_value`
**replaced** by `test_a_flammable_row_is_thin_and_a_structural_row_is_solid` — the
old one was a hardcoded dict of ten values, a pinned exact set of a table designed
to grow; the integer-reference cross-check and the shift round-trip move to
`math.ldexp` for signed exponents; `test_furniture_row_values` asserts a **negative**
exponent equal to the row's own derivation; `test_shipped_shifts` asserts the two
anchor rows straddle the regime boundary; plus signed-shift and fill-consistency
updates across `test_temperature_convert` ×2, the two conduction mirrors, and
`test_fuel_fraction_axis` ×3; and `test_dragon_ignites_wood_within_the_derived_tick_count`
→ `..._inside_its_reach_and_nothing_outside_it`.

**Deleted, deliberately and on schedule**:
`test_the_shipped_material_table_is_unmoved_by_M1`, M1's neutrality claim, spent by
the rows exactly as M1's report instructed. A tombstone comment replaces it pointing
at what took over.

---

## 9. What contradicted the design

**One genuine collision, reported not worked around**: §5.1, G12's exact-monotonicity
criterion versus §6's thin rows. **A ruling is owed.**

**One deviation from the letter, taken deliberately**: the "correct across levels"
clause (§1), because making only the material table per-level would create a new
inconsistency and half-solve q3 — which §4's own reversal forbids.

The two `ACCEPTED GAP` items were not touched, not critiqued, and no machinery was
built around them.

---

## 10. What M3 inherits

The four rows and the `fuel_per_o2` that makes the fuel store bind instead of the
timer; the bench re-anchored with every reference number re-quoted; and **a measured
head start** — with `H_BED_M = 38.73, H_BED_SHIFT = 0` substituted as a throwaway
diagnostic (nothing committed), **5 of the 11 reds clear** (`test_eos_p4_combustion`
×3, `test_ps1_smoke_roundtrip`, the firestorm test).

The three unit-digest reds flip from "the trajectory moved" to **"the fire never
reached the unit"** — which is precisely the live tension: fires must go out *and*
still do something.

`test_e1_hot_rail` and `test_payoff_orderings_perturbation_robust` do **not** clear
on `H_bed` alone.

Plus §5.1's damping question, which should be asked at the same time as `H_bed`, and
§5.4's blind golden.
