# Thin material rows — the answer to q7 (2026-09-20)

> **Status**: design, agreed with Erik in session on 2026-09-20. Not yet built.
>
> **Amends** `thermal_model_v2_design_2026-09-19.md` R14 ("base rows are
> MASSIVE") and answers its open questions **q7** (blocking), **q2**, and part
> of **q5**.
>
> **Depends on**: `ray_engine_v2_scheme_study_2026-09-13/report_t5b.md` §6.3
> (the measurement that forced this), `thermal_solid_two_node_design_prep_2026-09-16.md`
> (the alternative, deliberately not taken), `ray_engine_v2_NEXT_SESSION_2026-09-20.md`
> (the handoff this continues).

---

## 1. The ruling chain

T5b measured that combustion's fuel-bed deposit is **65 536× (exactly 2^16) the
physical heat of combustion**, and that the old raycaster's emission was
**2 419× too strong** and had been balancing it. The flip removed one half of
that compensating pair. Three options were put to Erik:

- **A** keep the shipped `H_bed` — undoes the arc, emission goes back to fitted.
- **B** take the derived `H_bed = 38.73`, `H_fuel = 116.2` — physically exact.
- **C** B plus issue #68 (two-node skin/core).

**Erik ruled B**: take the derived values. Never fit a dial to cover a model gap.

B alone leaves a model with no fire, because a 17.7 kW fire cannot heat a
153.9 kg tile (0.07 K/s). The report presented #68 as the only way out. **It is
not.** Erik's counter-proposal, adopted here:

> **Stop authoring flammable objects as 154 kg blocks.** A 154 kg block of wood
> genuinely does not ignite from a nearby fire — that is not a model defect, it
> is correct. Author the things that are *meant* to burn at the mass they really
> have, and the one-node model works.

Erik's accompanying ruling, explicit: **no massive object ever burns.** A solid
timber wall, a heavy door and a beam are fire-resistant permanently. That is
physically true and is accepted as a game-design constraint.

---

## 2. Why not two-node, and when to revisit

The decisive observation is that **the two designs share their working part**.
The #68 prep doc sizes the skin layer at `C_bulk / 32` = **4.8 kg**; a 1 cm
wooden panel is **4.6 kg**. The thin fast node that ignites is the same object
in both designs. Two-node's only additional contribution is the **core** — the
ability to have a heavy object whose surface burns while its bulk lags.

Breach has no use for that today. The flammables are crates, kindling, props and
foliage; the structure is steel and hull. Buying the core costs a second synced
temperature field: digest spec v7 and every golden regenerated, the Recorder
contract, residency sets, the CUDA twin, a new booked channel in the arc #54
ledger, and a re-answer of "which node do I seed?" for every igniter on the
CLAUDE.md *Starting a fire* row.

**This route honours R6** (*"I don't want to increase the complexity in the model
before we have a working simulation"*) rather than overturning it. The arc did
not produce evidence that the model needs another node; it produced evidence
that the **material table** is wrong.

**When to revisit #68**: if, after this lands, a genuine gameplay need appears
for a heavy object whose surface burns — the prep doc stays valid and unspent.
Erik's expectation, recorded: *"I think new materials will be the answer."*

---

## 3. The physics: when one node is honest

Heat entering a surface penetrates roughly `d = sqrt(alpha * t)`, with
`alpha ~ 1.5e-7 m^2/s` for wood. Over a 60 s ignition exposure that is **3.0 mm**.
A lumped single-temperature node is a faithful model of an object **thin compared
to that depth** — the textbook small-Biot regime.

| exposure | penetration depth | one node honest up to |
|---|---:|---:|
| 30 s | 2.1 mm | ~4 mm |
| 60 s | 3.0 mm | ~6 mm |
| 120 s | 4.2 mm | ~8 mm |

**`THIN_LIMIT_M = 0.006` is the ruled criterion.** A flammable row whose
effective thickness exceeds it is outside the model's domain of validity.

The failure mode beyond the limit is **graceful and one-directional**: a
too-thick lump spreads incident heat through more mass than really participates,
so it ignites **slower** than reality, never faster. There is no blow-up, only a
growing conservative error. This is why `furniture` at 10.8 mm is acceptable as a
stated approximation (§6) while nothing near 154 kg is.

---

## 4. Author DIMENSIONS; derive fill

A row keeps its **real, cited** `density` and `specific_heat`, and states the
object's **physical dimensions**. `fill_fraction`, `mass` and `thermal_mass` are
all **derived at load time** from those plus the *level's actual* tile geometry.

```
thickness_m   authored          # a physical fact about the object
mass          = density * thickness_m * tile_size_m * ceiling_h      # panel form
fill_fraction = mass / (density * V_tile)                            # DERIVED
thermal_mass  = pow2_snap(density * specific_heat * fill_fraction / THERMAL_MASS_UNIT)
```

### Why dimensions and not an authored `fill_fraction` (Erik's ruling, 2026-09-20)

**`fill_fraction` is brittle against a tile system that is not fixed.** The
material table is GLOBAL while `tile_size_m` is **per level** — `config.toml:148`
says so in as many words — and `V_tile` is currently built from
`tile_size_ref_m = 0.333`, the shipped-level value. An authored fill fraction
would therefore bake the *reference* tile size into the row's physical meaning:
on a 1.0 m level the "same" row would silently become a 1.5 cm slab instead of a
0.5 cm one.

An authored **thickness** states a fact about the object that no tile size can
falsify. This is the same lesson T3b already learned one level down — author by
`density` so that `V_tile` cancels identically — applied to geometry.

**The structural payoff: ignition time becomes tile-size invariant.** For a panel,

- capacity `C = rho * c * thickness * tile_w * ceiling_h`, so `C ∝ tile_w`;
- incident power arrives on the exposed face, area `tile_w * ceiling_h`, so
  `P ∝ tile_w`;
- therefore `dT/dt = P / C` is **independent of tile size**.

With an authored fill fraction it is not. A 0.5 cm panel ignites in 66 s on every
level, which is what a physical model should say.

### Why the geometry column is separate from `density` and from `heat_atten`

1. **Honesty.** `density = 8.33` would be a lie about wood. `density = 555,
   thickness_m = 0.005` is two true statements, each independently citable.
2. **Geometry and optics are genuinely independent.** 5 mm of wood is *optically
   opaque*: `light_atten` stays 1.0 and `heat_atten` stays 0.90 (emissivity)
   while the mass drops 67x. A single smeared density cannot express that.
3. **It settles the #68 prep doc §6 inconsistency** — whether `furniture`'s
   `heat_atten = 0.5` was an emissivity or a secret fill fraction. With geometry
   explicit, `heat_atten` is purely emissivity, everywhere, forever.
4. **It makes the §3 criterion directly checkable** — `thickness_m` *is* the
   lumped-validity length. No derivation stands between the authored number and
   the rule.

### The wrinkle: panels and floor-standing objects scale differently

This is real physics, not an artifact:

| form | mass scales as | example |
|---|---|---|
| **panel** — spans the tile, full height | `tile_w` (width x ceiling x thickness) | `wood` |
| **volume** — stands on the floor | `tile_w^2` (footprint x stack height x bulk density) | `furniture`, `kindling`, `foliage` |

`wood` is genuinely a panel. The other three are approximated as an **equivalent
slab thickness** (the "mm equiv" column of §6) — which is the right single
parameter for a lump, and is one power of `tile_w` wrong if levels ever differ in
tile size. Recorded as an approximation in the same spirit as `furniture`'s
10.8 mm exemption, to be replaced by an explicit `form` discriminator when a
level with a different tile size actually ships.

### This converges with q3

The conduction table has the **identical defect**: under R10 it is tile-size
dependent and is built at `tile_size_ref_m`, so on `airlock_demo` (1.0 m) every
solid face is 3 shifts too fast and every gas face 2. Both problems are "a
global material table pinned to a reference tile size that the level need not
share," and both are fixed by the same move — **derive the tile-dependent
quantities at load from the level's real geometry**. M2 should build the seam
that serves both, even if q3's own numbers land later.

---

## 5. Negative `thermal_mass` exponents

The rows below need `thermal_mass` of 0.10–0.26. Today `pow2_snap` **refuses**
anything below 1:

> *"A material this light thermally is not expressible; report it rather than
> inflating the row"*

That refusal is a **guard, not an arithmetic limit**, and the guard did exactly
its job: it told us to come back rather than fudging a row.

`heat_inv_shift` is already `int32_t` — signed — in Python and C++ alike, and the
capacity path does not use it as a shift at all. `cell_capacity_q` builds a
Q16.16 number from it (`temperature_solver.h:207`):

```cpp
int s = (int)heat_inv_shift_i;
if (s < 0) s = 0;                                       // the guard
*cap_used = (int64_t)1 << (s + fixedpoint::FP_SHIFT);
```

At `s = -2` that is `1 << 14 = 16384`, which **is 0.25 in Q16.16, exactly**. The
representation has supported fractional thermal mass all along.

### The four sites

| # | site | change | note |
|---|---|---|---|
| 1 | `materials.py::pow2_snap` | return the **signed exponent**, floor at −16; `thermal_mass` is already `float32` | `2^-16` is the true representation floor; the rows need −2/−3 |
| 2 | `temperature_solver.h:207` `cell_capacity_q` | drop `if (s < 0) s = 0` | the whole conduction path then needs **nothing** — it is all Q16.16 capacity |
| 3 | `temperature_solver.cpp:284` radiation fold | `shr_round0_i64(rn, s)` needs a signed twin: `rn << (−s)` for `s < 0` | the negative branch is **exact** — a left shift loses nothing, where the right shift truncates |
| 4 | `temperature_solver.cpp:359` heat deposit | `int32_t gain = deposit >> shift` | **the one real risk**: `deposit << 4` overflows int32 above 2^27. Route through int64 and saturate — T5a already built those twins |

Plus the CUDA twin at the same sites, and a digest bump with goldens
(`heat_inv_shift` is synced state).

**The gas sentinel does not collide.** `thermal_solid` is `thermal_mass > 0` and
the column is already `float32`, so `0.25 > 0` holds. The C++ takes `bool is_ts`
separately and never tests the value.

---

## 6. The derived rows

Solid-wood tile = **153.9 kg**. `t_ign` at the 17.7 kW fuel-bed self-heat;
`t_spread` at the 12 kW cross-gap flux; both to a 300 K rise.

| row | mm equiv | fill % | kg | raw t_mass | snap | used | C kJ/K | t_ign | t_spread |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`wood`** (0.5 cm panel) | 5.0 | 1.50 | 2.3 | 0.120 | 2^-3 | 0.125 | 3.9 | **66 s** | 97 s |
| **`kindling`** | 4.3 | 1.30 | 2.0 | 0.104 | 2^-3 | 0.125 | 3.9 | **66 s** | 97 s |
| **`foliage`** | 4.3 | 1.30 | 2.0 | 0.104 | 2^-3 | 0.125 | 3.9 | **66 s** | 97 s |
| **`furniture`** | 10.8 | 3.25 | 5.0 | 0.260 | 2^-2 | 0.250 | 7.8 | **132 s** | 195 s |

`mm equiv` is the **authored** number (`thickness_m`); `fill %`, `kg` and
`thermal_mass` are **derived**, shown here at the shipped `tile_size_m = 0.333`.
On a different tile size the derived columns move and `t_ign` does not (§4).

**Literature check**: piloted ignition of wood at this flux is **30–70 s**. The
thin rows land at 66 s with nothing tuned. The power-of-two snap costs +4 %
(0.120 -> 0.125), negligible at this scale.

### Row-by-row rationale

- **`wood`** — repurposed from the massive structural wall to a **0.5 cm
  panel**, Erik's number. The row is *already* `flammable = true,
  ignition_temp = 300`: it has been declaring itself fuel and unable to deliver.
  **The name is kept.** "Wood" claims a material, not a thickness, and a 5 mm
  plywood partition is wood. Renaming would churn the config key, the editor
  palette (`level_edit_common.py`), `fire_tuning_lab.py`'s `REACH_SRC_MATS` and
  the docs — and would orphan the row's git history, which *is* the calibration
  record (R13's pin, T3 D6's Wood Handbook citations). The **id stays 2** either
  way; levels store numeric codes.
- **`kindling`** — restored to its own declared intent. Its header has always
  read *"the CAMPFIRE REFERENCE OBJECT — a 1–3 kg effective-class fuel row"*;
  R14's uniform rho = 555 flattened that to 153.9 kg, leaving the intent alive
  only in `hp = 8`. **A separate `10_kg_bonfire` row was considered and dropped**
  — the material id budget is full (ids 0–9; `MAT_FOLIAGE` must stay last, and id
  9 numerically equals the CSV `SPACE_CODE`), and no new id is needed: kindling
  already *is* that row.
- **`foliage`** — a canopy is mostly air. Follows kindling.
- **`furniture`** — 5 kg, roughly a few crates' worth of 1 cm board. **At
  10.8 mm this is outside the §3 lumped criterion** and is accepted as a stated
  approximation: it under-predicts ignition speed (§3's graceful direction).
  It is the heaviest row still defensible as one node, and cover the player
  hides behind should not catch as eagerly as a panel.

`furniture` and `kindling` currently carry `conductivity = 0.0` — a fixture hack
(the prep doc §5 flags it) that exists purely to stop a burning crate's heat
draining into the bulk. With the bulk gone, **the hack should be removable**;
whether it can be is a P3 measurement, not an assumption.

---

## 7. The pin is a definition, not a measurement

```
RHO_C_PIN         = 0.9e6 J/(m3.K)
THERMAL_MASS_PIN  = 8
THERMAL_MASS_UNIT = 112 500 J/(m3.K) per unit
```

This is the metre-stick: every physical quantity in the thermal model
(`J_per_count`, the ledger, `c_v`, the marine burn band in kW/m^2) is denominated
against it.

R13 chose it by **reading it off a row** — `wood` was solid, its rho*c was
0.899 MJ/(m3.K), it carried `thermal_mass = 8`. After this design **no row is
authored at full fill**, so the pin's visible anchor disappears while its value
stays exactly correct (a definition does not expire because nothing is currently
one metre long).

**The hazard is a future reader "fixing" it.** They open `wood`, see
`fill_fraction = 0.015`, conclude the pin is stale, and correct it — silently
re-scaling every material and every derived physical number, with nothing
failing, because everything is *relative* to the pin. That is the same shape as
the 2^16 error this arc just spent a patch finding.

**The rule**: *the thermal_mass unit scale is a definition anchored on a
reference substance, and is never read off, nor re-derived from, a material row.*

Enforced in three layers (§10 M4):

1. **Structural** — define the pin from named `REFERENCE_SUBSTANCE` constants
   (solid softwood, 12 % MC, rho = 555, c = 1620), not from a row lookup. The pin
   then cites a substance, and "no row is that substance" reads as obviously
   fine rather than obviously broken.
2. **Load-time validation** at the materials door, beside the existing four
   number-doors (§11).
3. **A tool and a skill** — `tools/derive_material_row.py` and an
   `adding-a-material` skill, so a new row is *derived*, never reasoned about.

---

## 8. The fuel store replaces the timer — and it falls out for free

Today `hp` is the fuel store, consumed at
`fuel_per_o2[mat] = hp * KG_FUEL_PER_N_O2 / (density * V_tile)` per unit of O2
burned, **and** drained in parallel on a timer by `[fire] wall_damage = 0.36`
(Erik's 3-minute fuel-out ruling). report_t5b §5.2 measured the timer dominating
the physical channel **~10:1** — so "fuel-out" has been a stopwatch, not physics.

The mass drop flips that automatically, with no new mechanism:

| | O2 units in the tile | hp per O2 unit | total |
|---|---:|---:|---:|
| `wood` today (153.9 kg) | 414 | 0.145 | 60 hp |
| `wood` thin (2.3 kg) | 6.2 | 9.7 | 60 hp |

The whole bar is still consumed exactly when all the fuel is burned — but over
**6.2** units of O2 instead of 414, so the physical channel gets ~67x stronger
relative to the timer. Erik's ruling: **the fuel store should bind, not the
timer.**

### RULED (Erik, 2026-09-21): Option 1 — keep the bar, delete the timer

`wall_hp` has **three** consumers, not the two R14 names:

| # | channel | law | where |
|---|---|---|---|
| 1 | structural / combat | `wall_hp -= ammo.wall_damage` | `combat.py::chew_wall` |
| 2 | **chemistry** | `wall_hp -= fuel_per_o2 * burn_i` (O2 actually drawn) | `combustion.cpp:813` |
| 3 | **the timer** | `wall_hp -= wall_damage * dt * hotf * I` | `fire_simulation.cpp:360` |

Channel 3 depends on **time, temperature and intensity — never on oxygen consumed
or on how much mass is in the tile**. A tile burns down in a fixed time whether it
is a matchstick or a tree trunk. It dominates channel 2 ~10:1, which is why a
9.66x change to the fuel rate moved burn duration 0.5 % (report_t5b §5.2).

**The ruling**: `wall_damage` goes to **0**; `hp` stays feel-authored and the
derived `fuel_per_o2` exchange rate makes spending the whole bar equal burning the
tile's real combustible mass — which is what R14 built and never got to use.
**Zero new synced state.** Burn duration becomes fuel / burn rate: physics, with
no dial.

**Option 2 is the recorded upgrade path**, not a rejected alternative: a separate
`fuel_remaining` plane, depleted only by combustion, with `wall_hp` purely
structural. It costs one synced int32 plane (digest, residency, Recorder, CUDA
twin) and buys independent control — "tough to shoot but burns fast", and shot-up
debris that is still flammable. Take it if the human test shows Option 1's
coupling wart biting; M2's derived store is what makes it cheap.

**This is not "removing a dial."** Channel 3 is load-bearing today — its own
comment calls it *"the fuel-consumption brake"*, and with a fictional 154 kg store
it was the only thing that ever ended a fire. M3 **swaps which channel ends a
fire, from a clock to stoichiometry**, and must verify that fires still go out.

Two things survive the swap (verified, not assumed):

- **R3's "hot burns faster" is not lost** — `hotf` also drives the O2 demand in
  channel 2, so a hotter fire consumes fuel faster through the physical path.
- **Starvation still works** — `combustion.cpp:511` kills the ember at
  `wall_hp <= FUEL_FLOOR`, independently of the timer.

### FINDING — only the timer can DESTROY a tile

The two channels end differently: chemistry **floors** at `FUEL_FLOOR = 1` raw and
never goes below it, while the timer **destroys** at `wall_hp <= 0 && is_wall`.

So deleting channel 3 as-is leaves a burnt-out crate **standing** — intact,
inert, having consumed all its wood, and nothing ever burns *through*.

**M3 must move the destruction decision to the chemistry channel**: a flammable
wall tile that exhausts its fuel is destroyed. One condition in the right place,
not a new field. Option 1 turns out to need the same burnt-out-falls-apart
coupling that was described under Option 2.

### What M3 still owes as a MEASUREMENT

The burn rate once the timer is gone, and therefore the real durations. The
first-order estimate at the bench's ~71 kW total fire power:

| row | kg | O2 units | chemical MJ | burn time |
|---|---:|---:|---:|---:|
| `wood` (0.5 cm panel) | 2.3 | 6.2 | 29.9 | ~7.0 min |
| `kindling` | 2.0 | 5.4 | 26.0 | ~6.1 min |
| `foliage` | 2.0 | 5.4 | 26.0 | ~6.1 min |
| `furniture` | 5.0 | 13.5 | 65.0 | ~15.3 min |
| *the old 154 kg crate, for scale* | 153.9 | 414.1 | 2 000.7 | *~7.9 h* |

**Erik's 3-minute fuel-out ruling (2026-09-06) was set against a store that
physically held 471 minutes — a 157x discrepancy.** It was a patch over a
fictional store, and with a real store it is **re-derived, not carried forward**.
Wanting ~3 minutes after seeing 7 and 15 is a legitimate game-design choice
against a physical baseline; it is simply not the same decision. M3 brings the
measured numbers back to Erik before anything is locked.

---

## 9. What this closes, and what stays open

**Closes**: q7 (the blocking one), q2 (flammable rows are objects, with real
masses), and gives q5 a mechanism.

**Still open, unchanged**: q1 (opaque rows emitting 0.85 while extinguishing
1.0), q3 (per-level conduction table), q4 (sub-dead-band faces as a one-way
sink), q6 (the gas's missing radiative loss channel, lands at P5).

**Newly relevant**: `ignition_temp` was reviewed at G12 against *massive* rows.
With thin rows reaching ignition in ~66 s the values look right, but the review
was done under a different premise and should be re-read, not re-derived.

---

## 10. Patch plan

These land **on top of `12-t5b-the-flip`**, which stays unmerged. Its 8 red
tests are the runaway from the uncorrected `H_bed`; M3 is what makes them green.
Nothing merges to `fire-12` until the whole stack is green — the branch rule
holds.

| | patch | what it does | gate |
|---|---|---|---|
| **M1** | **negative exponents** | the four §5 sites + the CUDA twin; `pow2_snap` returns a signed exponent | digest bump + goldens; CPU/CUDA bit-identity; a perturbation test that the int32 deposit really would have overflowed |
| **M2** | **dimensions + the rows** | the `thickness_m` column, the load-time derivation of `mass`/`fill_fraction`/`thermal_mass` from the **level's** geometry (the seam q3 also needs, §4), the four rows of §6, the pin decoupling (§7.1), the validators (§11) | property gates on the derived masses, the lumped criterion, and **tile-size invariance** (§11.3b); every row's ignition time measured on the bench against §6 |
| **M3** | **derived `H_bed`/`H_fuel` + fuel store** | 38.73 / 116.2; `wall_damage` -> 0 (§8, RULED); **move the destroy decision to the chemistry channel** (§8 FINDING); measure the real burn durations and bring them to Erik | **the 8 red tests go green** *without being bent*; **fires still go out, and burnt-out tiles are still destroyed** — the two properties the timer used to own; arc #54's closure identity still closes in int64 |
| **M4** | **the system** | `tools/derive_material_row.py`, the `adding-a-material` skill, the CLAUDE.md rows | the tool reproduces §6's table exactly |
| | **HUMAN TEST** | Erik plays it | this is the P3 feel gate the design always had |

### Execution mode and model tier

| | mode | tier | why |
|---|---|---|---|
| **M1** | subagent, own worktree | **Opus**, high | Its oracle is weaker than it looks: the goldens are **re-baselined** by this very patch, so a golden records whatever the code does. That is precisely the blind-gate pattern this arc hit four times. It is also the foundational representation change — everything downstream inherits its errors. |
| **M2** | subagent, own worktree | **Opus**, high | Composes the material table, level loading and the q3 tile-geometry seam. The derived numbers have no oracle but §6's table, which this patch is supposed to reproduce *from first principles*, not copy. |
| **M3** | subagent, own worktree | **Opus**, high | Physics judgement (what `wall_damage` becomes), and it must turn 8 red tests green **without bending any of them** — the failure mode the skill's rule 5 names. |
| **M4** | subagent | **Sonnet 5** | Genuinely oracle-gated: the tool must reproduce §6's table exactly. Mechanical once the numbers are settled. |

**Fable is not used on this stack** — no patch composes three or more canonical
systems with a judgement-call failure mode, and the 2026-09-16 budget finding
stands.

**ACCEPTED GAP** — `furniture` at 10.8 mm sits outside the §3 lumped criterion.
This is a decision with its reason recorded in §6, not a finding: do not build
machinery to close it, and do not critique it.

**ACCEPTED GAP** — volume-form rows (`furniture`, `kindling`, `foliage`) are
authored as equivalent slab thicknesses and so scale as `tile_w` rather than
`tile_w^2`. Erik's ruling, 2026-09-21: single `thickness_m` column now; add a
`form` discriminator when a level with a different tile size actually ships.

After the human test: **T6** (old-law deletion), **T7** (CLAUDE.md walkthrough),
**T8** (ill-posed-test sweep), then ray-engine **P4** (CUDA twin), **P5a/b**
(smoke and gas), **P6a/b/c** (light; P6b is the second human test), **P7**
(stealth + RL light).

**Standing requirement, from this arc's four blind gates**: every patch validates
its gates **by deliberately breaking the code** and names which gate caught what.
A green suite is not evidence.

---

## 11. Properties to pin

1. `thermal_mass`, `mass` and `fill_fraction` are all **derived** — from
   `density`, `specific_heat`, `thickness_m` and the **level's** tile geometry —
   and never authored. Extends the existing R14 check to geometry.
2. **The lumped criterion**: a row with `flammable = true` must satisfy
   `thickness_m <= THIN_LIMIT_M`. The authored number *is* the tested number.
   The model **refuses** a row it cannot model. `furniture` is the one recorded
   exception (§6), carried as an explicit exemption with its reason, in the style
   of the `ingress-exempt:` convention.
3. `thickness_m > 0` and the derived `fill_fraction` lands in `(0, 1]` **on every
   shipped level** — a row that overflows its tile at some tile size is a load-time
   error, not a clamp. `density` and `specific_heat` stay real cited values, so
   geometry can never be smuggled into density.
3b. **Tile-size invariance**: a panel row's derived `dT/dt` under a fixed incident
   flux is identical at `tile_size_m` of 0.333 and 1.0 (§4). This is the property
   that authoring dimensions buys, so it is the property that gates it.
4. **The pin is not read off a row** — a test asserting `THERMAL_MASS_UNIT`
   against the named reference-substance constants, whose stated property is
   *"the unit scale is a definition; moving it silently re-scales every material
   in the game."*
5. Negative exponents round-trip exactly: `cap_used(s) == 2^(s+16)` for
   `s in [-16, CAP_SHIFT_MAX]`, and the left-shift branch of the radiation fold
   is **lossless** (unlike the truncating right shift).
6. Every row's ignition time, measured on the bench, matches §6 — the property
   being *"a thin row ignites inside the literature band"*, not the exact second.

---

## 12. Systems

**Existing canonical systems this design uses** (from the project CLAUDE.md
inventory):

- **Material / gas tables** (`materials.py`) — `fill_fraction` is a new column on
  the existing row schema, not a new mechanism. **The id budget stays full**: no
  new material id is created, so the `SPACE_CODE` migration is not spent.
- **Fixed-point kits** (`fixed_point.h`, `cuda_fixedpoint_device.cuh`) — the
  signed-shift twin joins the kit; no shift or rounding is re-derived at a call
  site.
- **Temperature solver** — solids' temperature is still derived there alone.
- **Field digest / GOLDEN_AGGREGATE** — one deliberate re-baseline for M1, with
  written rationale, per the iron rule.
- **Energy closure identity** (arc #54) — unchanged: this design adds no channel
  that moves `gas_energy`, so there is no fifth group.

**New systems this design creates**, each with its draft one-line rule for the
project CLAUDE.md (written at implementation, pointing at real code):

| new system | draft rule |
|---|---|
| Material geometry column (`thickness_m`) | A row states its **real** `density`, `specific_heat` and physical **dimensions**; `mass`, `fill_fraction` and `thermal_mass` are DERIVED from those plus the **level's** tile geometry — never author a smeared density, never author a fill fraction (it bakes in a reference tile size), and never fold geometry into `heat_atten` (which is emissivity, only). |
| The lumped-validity door | A `flammable` row must satisfy the small-Biot criterion (effective thickness <= `THIN_LIMIT_M` = 6 mm) or the materials door REFUSES it. Massive objects are authored non-flammable — a 154 kg block genuinely does not ignite. Exceptions carry a named reason. |
| The reference-substance pin | `THERMAL_MASS_UNIT` is a **definition** anchored on named reference-substance constants, never read off or re-derived from a material row; moving it silently re-scales every material in the game. |
| `tools/derive_material_row.py` | THE derivation instrument for a material row (mass, `thermal_mass`, snap error, ignition times, fuel store, validity). A new or edited row is DERIVED here and its output pasted into the row's comment — never reasoned about by hand. |
| `adding-a-material` skill | The procedure around the tool: cite rho and c, derive fill from real geometry, run the tool, check the snap, archive the paper under `docs/papers/`, bump the digest if the column moved. |
